"""Read-only measurements of explicitly selected Product execution records.

The selected segments are not a task-suite denominator or proof of complete history.
为什么要有它：a selected continuation, budget reservation or completed platform
operation cannot establish independent task success or whole-task cost.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.mod_agent.evaluation.context_chain import _cost
from src.mod_agent.task_agent_v2_state import load_v2_artifact
from src.observability.dashboard.services.trace_service import TraceService
from src.observability.patch_execution_index import project_product_execution


def _local_path(value):
    raw = str(value)
    if raw.startswith(("\\\\", "//")) or Path(raw).drive.startswith("\\\\"):
        raise ValueError("Network paths are not measurement inputs.")
    return Path(raw)


def _safe_reference(ref, root):
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError("Measurement record reference is invalid.")
    path = _local_path(ref["path"])
    selected = path if path.is_absolute() else root / path
    if selected.is_symlink() or not selected.resolve().is_relative_to(root):
        raise ValueError("Measurement record reference escapes its selected root.")
    return selected.resolve()


def _preflight(root):
    manifest, digest = TraceService._read_product_json(root / "run_manifest.json", root=root)
    terminal = manifest.get("terminal_record")
    if terminal is not None and _safe_reference(terminal, root) != root / "run_result.json":
        raise ValueError("Terminal reference does not select run_result.json.")
    observations = manifest.get("execution_observations")
    if observations is not None and not isinstance(observations, dict):
        raise ValueError("Execution observations must be an object.")
    for key, row in (observations or {}).items():
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] != key:
            raise ValueError("Observation dictionary key and identity disagree.")
        attrs = row.get("attributes") or {}
        if not isinstance(attrs, dict):
            raise ValueError("Observation attributes must be an object.")
        refs = attrs.get("local_detail_refs") or {}
        if not isinstance(refs, dict):
            raise ValueError("Local detail references must be an object.")
        for ref in refs.values():
            _safe_reference(ref, root)
    parent = manifest.get("parent_result_ref")
    if isinstance(parent, dict) and isinstance(parent.get("path"), str):
        _local_path(parent["path"])
    return manifest, digest


def _read_bound(ref, root):
    value, digest = TraceService._read_product_json(Path(ref["path"]), root=root)
    if digest != ref["sha256"]:
        raise ValueError("Measurement record changed during inspection.")
    return value


def _lineage(manifest, prior):
    ref = manifest.get("parent_result_ref")
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        return {"status": "unknown_missing_parent_reference"}
    path = _local_path(ref["path"])
    if not path.is_absolute():
        return {"status": "unknown_nonabsolute_parent_reference"}
    for index, segment in enumerate(prior):
        expected = segment["result_ref"]
        if path.resolve() == Path(expected["path"]).resolve():
            if ref.get("sha256") != expected["sha256"]:
                raise ValueError("Selected parent result hash mismatch.")
            return {"status": "verified_selected_parent", "parent_segment_index": index,
                    "continues_previous_selection": index == len(prior) - 1}
    # Never open a parent merely mentioned in an untrusted record.
    # 为什么要有它：lineage metadata does not expand the caller's selected read scope.
    return {"status": "unknown_parent_not_selected"}


def _product_projection(root, manifest, result):
    if manifest.get("terminal_record") is None:
        return {"record_status": "unknown_missing_terminal_reference"}
    if not manifest.get("request_id") or (
        not result.get("request_id") and result.get("product_execution_result") is None
    ):
        return {"record_status": "unknown_missing_request_identity"}
    # The existing index validates identity inside a typed terminal result.
    # 为什么要有它：normal Product writers need not duplicate request_id at top level.
    return project_product_execution(root, run_id=result["run_id"])


def _record_identity(manifest, result, known):
    product = result.get("product_execution_result")
    records = (manifest, result, product) if isinstance(product, dict) else (manifest, result)
    # Retain identities from every selected terminal, not just its manifest.
    # 为什么要有它：legacy manifests omit fields that remain authoritative in the result.
    for name in ("run_id", "request_id", "feature_id"):
        for record in records:
            value = record.get(name)
            if value is None:
                continue
            if not isinstance(value, str) or not value:
                raise ValueError(f"Selected {name} identity is invalid.")
            if name in known and value != known[name]:
                raise ValueError(f"Selected {name} identity mismatch.")
            known[name] = value


def _terminal(result, projection):
    product = result.get("product_execution_result") or {}
    if not isinstance(product, dict):
        raise ValueError("Product execution result must be an object.")
    validated = projection.get("typed_result_status") == "validated"
    return {
        "status": result.get("status"),
        "current_stage": result.get("current_stage"),
        "requested_completion_scope": product.get("requested_completion_scope"),
        "platform_status": projection.get("platform_status"),
        "execution_disposition": projection.get("execution_disposition"),
        "recorded_product_success": (
            projection.get("execution_disposition") == "succeeded" if validated else None),
        "product_result_validation": projection.get("typed_result_status"),
        "product_record_status": projection["record_status"],
        "build_summary": result.get("build_summary"),
        "runtime_summary": result.get("runtime_summary"),
        "post_stages": projection.get("post_stages"),
        "blockers": result.get("blockers"),
        "artifact_refs": projection.get("artifact_refs", []),
    }


def _generation(call, trace_id, segment_index):
    return {key: call.get(key) for key in (
        "generation_span_id", "request_span_id", "role", "model", "provider", "status",
        "started_at", "finished_at", "elapsed_ms", "transport", "usage", "local_evidence", "response_sha256",
    )} | {"trace_id": trace_id, "segment_index": segment_index}


def _selected_calls(segments, manifests):
    observations, calls, orphan_requests, tools = {}, [], [], []
    duplicates = 0
    for index, (segment, manifest) in enumerate(zip(segments, manifests, strict=True)):
        trace_id = (manifest.get("execution_trace") or {}).get("trace_id")
        rows = manifest.get("execution_observations")
        if not isinstance(rows, dict):
            continue
        if rows and (not isinstance(trace_id, str) or not trace_id):
            raise ValueError("Observed records need a trace identity for deduplication.")
        new_ids = set()
        for row in rows.values():
            key = (trace_id, row["id"])
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if key in observations:
                if observations[key] != encoded:
                    raise ValueError("Duplicate observation identity has conflicting recorded facts.")
                duplicates += 1
                continue
            observations[key] = encoded
            new_ids.add(row["id"])
            if row.get("kind") == "tool":
                tools.append({"trace_id": trace_id, "observation_id": row["id"],
                              "segment_index": index, "name": row.get("name"),
                              "status": row.get("status"), "elapsed_ms": row.get("elapsed_ms")})
        parents = {call["request_span_id"] for call in segment["calls"]}
        calls.extend(_generation(call, trace_id, index) for call in segment["calls"]
                     if call["generation_span_id"] in new_ids)
        for row in rows.values():
            if row["id"] in new_ids and row.get("name") == "model.request" and row["id"] not in parents:
                orphan_requests.append({"trace_id": trace_id, "observation_id": row["id"],
                    "segment_index": index, "result": {"recorded_decision": {
                        "transport_started": (row.get("attributes") or {}).get("transport_started")}}})
    parents = {(row["trace_id"], row["request_span_id"]) for row in calls}
    orphan_requests = [row for row in orphan_requests
                       if (row["trace_id"], row["observation_id"]) not in parents]
    return calls, orphan_requests, tools, duplicates


def _recovery_reference(ref, root):
    path = _safe_reference(ref, root)
    digest = ref.get("sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("Recovery execution reference needs a SHA256 binding.")
    canonical = {"path": str(path), "sha256": digest}
    return _read_bound(canonical, root), canonical


def _recovery_segment(service, root, ref, known_identity, index):
    document, selected_ref = _recovery_reference(ref, root)
    if document.get("schema_version") != "task_agent_recovery_execution_v1":
        raise ValueError("Selected recovery execution schema is invalid.")
    for name in ("run_id", "request_id"):
        if not known_identity.get(name) or document.get(name) != known_identity[name]:
            raise ValueError(f"Recovery execution {name} identity mismatch.")
    # These are reference integrity checks, not a successful-recovery gate.
    # 为什么要有它：失败恢复仍须计入成本，交付资格由独立的候选链校验负责。
    for name in ("resumed_from_checkpoint_ref", "v2_checkpoint_ref"):
        load_v2_artifact(root / "a", document.get(name))
    if document.get("native_execution_ref") is not None:
        _recovery_reference(document["native_execution_ref"], root)
    trace = document.get("execution_trace")
    rows = document.get("execution_observations")
    if (not isinstance(trace, dict) or not isinstance(trace.get("trace_id"), str)
            or not trace["trace_id"] or not isinstance(rows, dict)):
        raise ValueError("Recovery execution trace or observations are malformed.")
    for key, row in rows.items():
        if (not isinstance(row, dict) or not isinstance(key, str) or not key
                or row.get("id") != key or row.get("trace_id") != trace["trace_id"]
                or not isinstance(row.get("kind"), str) or not isinstance(row.get("name"), str)
                or not isinstance(row.get("attributes", {}), dict)):
            raise ValueError("Recovery observation identity or attributes are malformed.")
        details = row.get("attributes", {}).get("local_detail_refs", {})
        if not isinstance(details, dict):
            raise ValueError("Recovery local detail references are malformed.")
        for detail in details.values():
            _safe_reference(detail, root)
            if service._local_ref_summary(detail, root=root)["integrity"] != "verified_bytes":
                raise ValueError("Recovery local detail reference is missing or changed.")
    calls = []
    for row in rows.values():
        if row.get("kind") != "generation":
            continue
        parent_id = row.get("parent_id")
        parent = rows.get(parent_id) if isinstance(parent_id, str) else None
        if not isinstance(parent, dict) or parent.get("name") != "model.request":
            raise ValueError("Recovery generation has no bound parent model.request observation.")
        calls.append(service._product_call("recovery", root, parent, row, False))
    count = document.get("model_calls_performed")
    if type(count) is not int or count < 0 or count != len(calls):
        raise ValueError("Recovery recorded generation count disagrees with observations.")
    return {"segment_index": index, "run_root": str(root), "manifest_ref": None,
        "result_ref": None, "execution_ref": selected_ref, "trace": trace,
        "lineage": {"status": "explicit_recovery_execution_not_delivery_lineage"},
        "calls": calls, "observation_coverage": "recorded",
        "terminal": {"status": document.get("status"), "recorded_product_success": None,
                     "product_record_status": "recovery_execution_only"},
        "recorded_budget": None, "recorded_model_call_count": count}, document


def measure_product_task(*, allowed_root, run_root, run_id, continuation_roots=(), execution_refs=()):
    """Measure selected records; do not invoke models, validators, or a workflow."""
    service = TraceService()
    allowed = service._existing_directory(_local_path(allowed_root), label="Allowed Runs Root")
    original = service._selected_directory(_local_path(run_root), allowed=allowed, label="Run Root")
    if original.name != run_id:
        raise ValueError("Run Root basename does not match run_id.")
    roots, duplicate_roots = [original], 0
    for raw in continuation_roots:
        selected = service._selected_directory(
            _local_path(raw), allowed=original / "continuations", label="Continuation Root")
        if selected in roots:
            duplicate_roots += 1
        else:
            roots.append(selected)

    segments, manifests, known_identity = [], [], {}
    for index, root in enumerate(roots):
        manifest, manifest_hash = _preflight(root)
        if manifest.get("run_id") != run_id:
            raise ValueError("Selected manifest run identity mismatch.")
        if isinstance(manifest.get("execution_observations"), dict):
            if index and not isinstance(manifests[0].get("execution_observations"), dict):
                view = service._product_segment(
                    "continuation", root, run_id, False, selected_run_root=original)
            else:
                view = service.inspect_product_run(
                    allowed_root=allowed, run_root=original, run_id=run_id,
                    continuation_root=root if index else None,
                )["segments"][-1]
            if view["manifest_ref"]["sha256"] != manifest_hash:
                raise ValueError("Manifest changed during measurement.")
            result = _read_bound(view["result_ref"], root)
        else:
            # Old runs may have terminal facts but no native observations.
            # 为什么要有它：absence of telemetry must not turn into zero calls or tokens.
            result, digest = service._read_product_json(root / "run_result.json", root=root)
            ref = manifest.get("terminal_record")
            if ref is not None and ref.get("sha256") != digest:
                raise ValueError("Terminal result hash mismatch.")
            view = {"calls": [], "manifest_ref": {"path": str(root / "run_manifest.json"),
                    "sha256": manifest_hash}, "result_ref": {"path": str(root / "run_result.json"),
                    "sha256": digest}, "trace": manifest.get("execution_trace")}
        if result.get("run_id") != run_id:
            raise ValueError("Selected terminal run identity mismatch.")
        terminal_ref = manifest.get("terminal_record")
        if terminal_ref is not None:
            if terminal_ref.get("sha256") != view["result_ref"]["sha256"]:
                raise ValueError("Terminal result hash mismatch.")
            view["result_ref"]["binding"] = "manifest_terminal_sha256"
        _record_identity(manifest, result, known_identity)
        lineage = _lineage(manifest, segments) if index else {"status": "selected_original"}
        projection = _product_projection(root, manifest, result)
        segments.append({"segment_index": index, "run_root": str(root),
            "manifest_ref": view["manifest_ref"], "result_ref": view["result_ref"],
            "trace": view["trace"], "lineage": lineage, "calls": view["calls"],
            "observation_coverage": "recorded" if isinstance(manifest.get("execution_observations"), dict)
                else "unknown_missing_observations",
            "terminal": _terminal(result, projection),
            "recorded_budget": result.get("model_call_budget", manifest.get("model_call_budget")),
            "recorded_model_call_count": result.get("model_call_count", manifest.get("model_call_count")),
        })
        manifests.append(manifest)

    selected_execution_refs = []
    for ref in execution_refs:
        segment, document = _recovery_segment(
            service, original, ref, known_identity, len(segments))
        if segment["execution_ref"] in selected_execution_refs:
            duplicate_roots += 1
            continue
        selected_execution_refs.append(segment["execution_ref"])
        segments.append(segment)
        manifests.append(document)

    calls, orphans, tools, duplicate_observations = _selected_calls(segments, manifests)
    cost = _cost(calls, orphans)
    missing = [row["segment_index"] for row in segments if row["observation_coverage"] != "recorded"]
    if missing:
        for key in ("input_tokens", "output_tokens"):
            cost[key]["total"] = None
        cost["generation_elapsed_ms_sum"] = None
    transport = [row["transport"].get("transport_started") for row in calls]
    transport.extend(row["result"]["recorded_decision"]["transport_started"] for row in orphans)
    origins = Counter(row["transport"].get("response_origin") or "unknown" for row in calls)
    local_missing = [{"trace_id": row["trace_id"], "generation_span_id": row["generation_span_id"],
                      "role": role, "integrity": (row["local_evidence"].get(role) or {}).get("integrity", "missing")}
                     for row in calls for role in ("request", "response")
                     if (row["local_evidence"].get(role) or {}).get("integrity") != "verified_bytes"]
    for segment in segments:
        segment.pop("calls")
    # Totals stay scoped to selected evidence even with a verified linear lineage.
    # 为什么要有它：an unselected later continuation may exist, and recorded live
    # response markers do not establish holdout eligibility or independent truth.
    return {
        "schema_version": "product_task_measurement_v1", "run_id": run_id,
        "read_only": True, "inspection_model_calls": 0,
        "selection_scope": "explicit_segments_only", "selected_scope_complete": None,
        "selected_segment_count": len(segments), "duplicate_selections_ignored": duplicate_roots,
        "selected_execution_refs": selected_execution_refs,
        "duplicate_observations_ignored": duplicate_observations, "segments": segments,
        "model": {"generation_records": len(calls), "request_records_without_generation": orphans,
            "recorded_transport_started": sum(value is True for value in transport),
            "recorded_transport_started_total": sum(value is True for value in transport)
                if not missing and all(type(value) is bool for value in transport) else None,
            "independent_http_request_count": None, "response_origins": dict(origins),
            "usage": cost, "price": None, "calls": calls,
            "accounting_basis": "deduplicated_native_observations_not_budget_reservations_or_network_capture"},
        "tools": {"observed_calls": len(tools), "selected_record_total": len(tools) if not missing else None,
                  "by_name": dict(Counter(row["name"] or "unknown" for row in tools)), "observations": tools,
                  "scope": "kind_tool_observations_not_all_OS_operations_or_planned_tool_calls"},
        "last_selected_terminal": segments[-1]["terminal"],
        "completion": None, "independent_task_count": None,
        "acceptance_boundary": "recorded_product_outcome_only; independent_task_judgment_and_execution_provenance_not_supplied",
        "task_total_input_tokens": None, "task_total_output_tokens": None,
        "task_total_tool_calls": None, "task_elapsed_ms": None,
        "coverage": {"segments_missing_observations": missing,
            "unverified_model_local_evidence": local_missing,
            "lineage_verified_between_selections": all(
                row["lineage"].get("continues_previous_selection") is True for row in segments[1:])
                if len(segments) > 1 else None,
            "whole_task_history": "unknown_no_complete_history_attestation",
            "independent_eligibility": "unknown_no_frozen_task_split_or_judgment",
            "excluded_from_success_inference": ["preparation", "controlled", "calibration", "replay"],
            "whole_task_latency": "unknown_no_complete_task_timing_boundary"},
    }
