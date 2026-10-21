"""Read Cloud execution evidence without prompts, model calls or workflow writes.

The Cloud MCP is self-describing: discover its current input and field schemas
before querying. This is a disposable projection, not another execution index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import jsonschema

from src.observability.patch_langfuse import LangfuseHttpTransport, PatchLangfuseError
from src.observability.patch_live_langfuse import SYMBOL_CHECK_OPERATION, SYMBOL_CHECK_SCORE_NAME

_TOOLS = {"getObservationFieldSchema", "listObservations", "listScores"}
_FIELDS = (
    "id", "traceId", "parentObservationId", "isRootObservation", "type", "name",
    "startTime", "endTime", "level", "statusMessage", "providedModelName", "usageDetails",
    "totalCost", "completionStartTime", "promptName", "metadata", "sessionId",
)
_EXPAND = ("artifact_refs", "consumed_tool_results", "tool_call_id_sha256s", "consumed_materials",
           "produced_materials", "selected_evidence_id_sha256s", "relationship_counts", "decision_summary", "local_detail_refs",
           "terminal_record_ref", "symbol_declaration_checks")
_EXPAND += ("requirement_findings_attempt_audit", "supplemental_findings_replay", "terminal_model_call_budget")
_FACTS = (
    "turn_index", "evaluation_index", "workflow_status", "terminal_stage", "failure_code",
    "verification_status", "failure_owner", "execution_origin", "callback_invoked",
    "checks_run", "checks_failed", "request_sha256", "response_sha256", "output_sha256",
    "output_code_sha256", "generated_changes_sha256", "behavior_report_sha256",
    "request_bytes", "encoded_payload_bytes", "encoded_payload_sha256", "byte_count",
    "failure_boundary", "failure_classification", "context_admission_mode", "provider_body_utf8_bytes",
    "configured_num_ctx", "num_predict", "conservative_required_context",
    "provider_request_constructed", "provider_request_admitted", "provider_body_sha256",
    "estimated_prompt_tokens", "estimated_required_context", "observed_calibration_count",
    "exact_prompt_tokens_available_pre_call", "context_calibration_used",
    "call_id_sha256", "transport_started", "response_received", "response_complete", "response_origin", "http_status",
    "usage_available", "usage_mapping_status", "upstream_execution_performed",
    "request_registered", "result_code", "admission_result", "tool_dispatch_performed",
    "observation_queued", "model_consumption_proven", "historical_import", "metadata_only",
    "observation_status", "source_prompt_output_included", "score_eligible", "score_basis",
    "verification_executions", "verification_matrix_sha256", *_EXPAND,
    "model_role", "reason_status", "route_decision", "trace_scope", "model_input_assembled",
    "input_payload_sha256", "feature_payload_sha256", "evidence_payload_sha256", "structured_output_sha256",
    "unresolved_gap_count", "decision_summary_origin", "decision_claim_origin", "consumed_materials_truncated",
    "produced_materials_truncated", "selected_evidence_ids_truncated",
    "retrieval_status", "decision", "decision_status", "readiness", "input_role", "material_delivery_status", "selection_mode",
    "tool_exit_code", "evidence_item_count", "query_count", "requirement_count", "relation_count",
    "projection_fallback_used", "findings_invocation_performed", "can_generate_patch", "provider_failure_count",
    "materialized_projection_sha256", "output_payload_sha256", "reason_sha256",
    "local_body_status",
    "from_node", "selected_route", "selected_next_node", "route_selector", "material_capture_status",
    "query_sha256", "route_input_sha256",
    "route_input_summary_scope", "route_input_summary_status", "route_output_summary_status",
    "failure_stage", "execution_phase", "callback_returned", "source_directory_exists", "project_file_discovered", "local_manifest_created",
    "fact_delivery_status", "input_member_access_count", "output_member_access_count", "added_member_access_count",
    "removed_member_access_count", "produced_member_access_fact_count", "input_member_access_facts_sha256",
    "output_member_access_facts_sha256", "produced_member_access_facts_sha256",
    "symbol_check_status", "symbol_lookup_method", "symbol_check_sha256", "symbol_check_basis",
    "symbol_declaration_count", "symbol_materialized_count", "symbol_check_projection_complete", "symbol_check_summary",
    "requirement_findings_attempt_audit_sha256", "timing_basis", "source_journal_sha256", "source_run_id_sha256",
)

_MATERIAL_KEYS = ("evidence_id_sha256", "excerpt_sha256")
_DIGEST = re.compile(r"(?:sha256:)?[a-f0-9]{64}\Z")


def _material_identity(item):
    if not isinstance(item, dict) or any(not isinstance(item.get(key), str)
            or not _DIGEST.fullmatch(item[key]) for key in _MATERIAL_KEYS):
        return None
    return tuple(item[key] for key in _MATERIAL_KEYS)


def _material_matches(supplied, produced):
    identity = _material_identity(supplied)
    return (identity is not None and identity == _material_identity(produced)
            and ("source_hash" not in supplied or supplied["source_hash"] == produced.get("source_hash")))


def _upstream_evidence(rows, generations):
    """Report observable alternatives, not an inferred semantic root cause."""
    producers = [row for row in rows if isinstance(row.get("metadata", {}).get("produced_materials"), list)]
    stages = [row for row in rows if isinstance(row.get("metadata", {}).get("consumed_materials"), list)
              and row.get("type") != "GENERATION"]
    material_links, diagnostics = [], []
    retrieval_meanings = {"source_root_missing": "该次检索观察到材料目录不存在。",
        "no_matches": "该次检索执行后无匹配，不能推断整个知识库没有相关材料。",
        "requested_source_no_matches": "指定材料在该次查询中无匹配。",
        "source_scope_rejected": "当前材料范围被准入检查拒绝。", "query_terms_missing": "本次没有可执行的检索词。",
        "tool_failed": "本次检索工具执行失败。", "source_decode_failed": "本次材料解码失败。",
        "source_identity_ambiguous": "本次材料身份不唯一。"}
    for row in rows:
        status = row.get("metadata", {}).get("retrieval_status")
        if isinstance(status, str) and status in retrieval_meanings:
            diagnostics.append({"observation_id": row["id"], "fact": status, "meaning": retrieval_meanings[status]})
        meta = row.get("metadata", {})
        if row.get("name") == "source.admission" and (meta.get("source_directory_exists") is False or meta.get("project_file_discovered") is False):
            diagnostics.append({"observation_id": row["id"], "fact": "source_project_material_unavailable",
                "meaning": "前置材料检查明确缺少目录或项目文件，尚不能归为检索或模型修订失败。"})
    for producer in producers:
        meta = producer["metadata"]
        if not meta["produced_materials"] and meta.get("produced_materials_truncated") is False:
            diagnostics.append({"observation_id": producer["id"], "fact": "recorded_material_result_empty",
                "meaning": "该次材料结果明确为空；不能单凭此记录区分未准备材料与检索未命中。"})
    for generation in generations:
        meta = generation.get("metadata", {})
        payload = meta.get("input_payload_sha256")
        matching_stages = [stage for stage in stages if isinstance(payload, str) and _DIGEST.fullmatch(payload)
                           and stage["metadata"].get("input_payload_sha256") == payload]
        if isinstance(meta.get("consumed_materials"), list):
            matching_stages = [generation]
        if not matching_stages:
            if meta.get("model_role"):
                diagnostics.append({"observation_id": generation["id"], "fact": "input_material_binding_unknown",
                                    "meaning": "没有与本次请求payload哈希匹配的材料记录。"})
            continue
        original_timing = (meta.get("historical_import") is False or (
            meta.get("timing_basis") == "recorded_original_execution"
            and isinstance(meta.get("source_journal_sha256"), str)
            and _DIGEST.fullmatch(meta["source_journal_sha256"]) is not None))
        attempted = (meta.get("transport_started") is True and original_timing
                     and isinstance(payload, str) and _DIGEST.fullmatch(payload) is not None)
        http_status = meta.get("http_status")
        responded = (attempted and meta.get("response_received") is True
                     and meta.get("response_origin") == "live_provider_response"
                     and (http_status is None or type(http_status) is int and 200 <= http_status < 300))
        delivery = "provider_response_received" if responded else "transport_attempted" if attempted else "assembled_only"
        for stage in matching_stages:
            stage_meta = stage.get("metadata", {})
            consumed = stage_meta["consumed_materials"]
            for item in consumed:
                matches = [producer["id"] for producer in producers
                           if _milliseconds(producer.get("endTime"), generation.get("startTime")) is not None
                           if any(_material_matches(item, found) for found in producer["metadata"]["produced_materials"])]
                material_links.append({"generation_id": generation["id"], "input_observation_id": stage["id"],
                    "input_payload_sha256": payload, "material": item, "source_observation_ids": matches,
                    "binding": "matched_excerpt_in_responded_request" if responded and matches else
                               "matched_excerpt_in_transport_attempt" if attempted and matches else
                               "assembled_material_only" if matches else "material_origin_unresolved",
                    "delivery_status": delivery,
                    "input_source_hash": item.get("source_hash"),
                    "scope": "recorded_identity_not_model_attention_or_semantic_correctness"})
            if stage_meta.get("consumed_materials_truncated") is False:
                consumed_ids = {_material_identity(item) for item in consumed}
                for producer in producers:
                    # Compare only earlier observed results. This is an input inventory
                    # difference, not proof that every result should have been selected.
                    if _milliseconds(producer.get("endTime"), generation.get("startTime")) is None:
                        continue
                    for material in producer["metadata"]["produced_materials"]:
                        identity = _material_identity(material)
                        if identity is not None and identity not in consumed_ids:
                            diagnostics.append({"observation_id": generation["id"],
                                "input_observation_id": stage["id"], "source_observation_id": producer["id"],
                                "material": material, "fact": "returned_material_not_in_recorded_input",
                                "meaning": "先前返回的这份材料不在本次完整输入清单；是否应该消费仍需核查任务。"})
    return {"material_consumption_links": material_links, "evidence_diagnostics": diagnostics,
            "upstream_decisions": [{"observation_id": row["id"], "operation": row.get("name"),
                "facts": {key: row.get("metadata", {})[key] for key in _FACTS if key in row.get("metadata", {})}}
                for row in rows if row.get("metadata", {}).get("decision_summary") or row.get("metadata", {}).get("route_decision")]}


def _fail(code):
    raise PatchLangfuseError(code)


class CloudExecutionReader:
    """Bounded, read-only MCP access through the existing HTTP transport."""

    def __init__(self, transport, *, page_budget=5):
        if type(page_budget) is not int or not 1 <= page_budget <= 10:
            _fail("LANGFUSE_QUERY_PAGE_BUDGET_INVALID")
        self.transport, self.page_budget = transport, page_budget
        self.schemas = None

    def _rpc(self, method, params=None):
        payload = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            payload["params"] = params
        envelope = self.transport._request("POST", "/api/public/mcp", payload=payload)
        if not isinstance(envelope, dict) or envelope.get("error") or not isinstance(envelope.get("result"), dict):
            _fail("LANGFUSE_QUERY_RPC_INVALID")
        return envelope["result"]

    def _discover(self):
        result = self._rpc("tools/list")
        rows = result.get("tools")
        if not isinstance(rows, list):
            _fail("LANGFUSE_QUERY_SCHEMA_UNAVAILABLE")
        self.schemas = {r["name"]: r["inputSchema"] for r in rows
                        if isinstance(r, dict) and r.get("name") in _TOOLS and isinstance(r.get("inputSchema"), dict)}
        if not _TOOLS <= self.schemas.keys():
            _fail("LANGFUSE_QUERY_SCHEMA_UNAVAILABLE")

    def _call(self, name, arguments):
        if name not in _TOOLS:
            _fail("LANGFUSE_QUERY_READ_ONLY")
        if self.schemas is None:
            self._discover()
        try:
            jsonschema.validate(arguments, self.schemas[name])
        except (jsonschema.ValidationError, jsonschema.SchemaError):
            _fail("LANGFUSE_QUERY_SCHEMA_CHANGED")
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            _fail("LANGFUSE_QUERY_TOOL_FAILED")
        content = result.get("content")
        if not isinstance(content, list):
            _fail("LANGFUSE_QUERY_CONTENT_INVALID")
        try:
            value = json.loads(next(r["text"] for r in content if isinstance(r, dict) and r.get("type") == "text"))
        except (ValueError, KeyError, StopIteration, TypeError):
            _fail("LANGFUSE_QUERY_CONTENT_INVALID")
        if not isinstance(value, dict):
            _fail("LANGFUSE_QUERY_CONTENT_INVALID")
        return value

    def _pages(self, tool, arguments):
        rows, cursor, seen = [], None, set()
        for _ in range(self.page_budget):
            value = self._call(tool, {**arguments, **({"cursor": cursor} if cursor else {})})
            data, meta = value.get("data"), value.get("meta", {})
            if not isinstance(data, list) or not isinstance(meta, dict) or any(not isinstance(r, dict) for r in data):
                _fail("LANGFUSE_QUERY_PAGE_INVALID")
            rows.extend(data)
            cursor = meta.get("cursor")
            if not cursor:
                return rows
            if not isinstance(cursor, str) or cursor in seen:
                _fail("LANGFUSE_QUERY_CURSOR_INVALID")
            seen.add(cursor)
        _fail("LANGFUSE_QUERY_PAGE_BUDGET_EXHAUSTED")

    def read(self, trace_id, *, from_start_time=None, to_start_time=None):
        if not isinstance(trace_id, str) or not re.fullmatch(r"[a-f0-9]{32}", trace_id):
            _fail("LANGFUSE_TRACE_ID_INVALID")
        # Scope: use recorded observation times, never an arbitrary all-history search.
        # 为什么要有它：云查询现在要求有限时间窗，缺少时间不能伪装为已查到全部轨迹。
        window = {}
        if from_start_time is not None or to_start_time is not None:
            try:
                start, end = (datetime.fromisoformat(value.replace("Z", "+00:00"))
                              for value in (from_start_time, to_start_time))
                if (start.utcoffset() is None or end.utcoffset() is None
                        or not 0 < (end - start).total_seconds() <= 336 * 3600):
                    raise ValueError("invalid window")
            except (AttributeError, TypeError, ValueError):
                _fail("LANGFUSE_QUERY_TIME_RANGE_INVALID")
            window = {"fromStartTime": from_start_time, "toStartTime": to_start_time}
        schema = self._call("getObservationFieldSchema", {})
        if not isinstance(schema.get("fields"), dict) or not set(_FIELDS) <= schema["fields"].keys():
            _fail("LANGFUSE_QUERY_FIELDS_CHANGED")
        observations = self._pages("listObservations", {"traceId": trace_id, "limit": 100,
            "fields": list(_FIELDS), "expandMetadataKeys": list(_EXPAND), **window})
        scores = self._pages("listScores", {"traceId": [trace_id], "limit": 100})
        result = summarize_execution(trace_id, observations, scores)
        if window:
            result["query_window"] = window
        return result


def _milliseconds(start, end):
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    try:
        a, b = (datetime.fromisoformat(x.replace("Z", "+00:00")) for x in (start, end))
        if a.utcoffset() is None or b.utcoffset() is None or b < a:
            return None
        return round((b - a).total_seconds() * 1000, 4)
    except ValueError:
        return None


def _same_observed_time(local, cloud):
    """Cloud observation APIs truncate the retained OTLP times to milliseconds."""
    if not isinstance(local, str) or not isinstance(cloud, str):
        return False
    try:
        a, b = (datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (local, cloud))
        return (a.utcoffset() is not None and b.utcoffset() is not None
                and (a == b or b.microsecond % 1000 == 0 and a.replace(microsecond=a.microsecond // 1000 * 1000) == b))
    except ValueError:
        return False


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _usage(row):
    raw = row.get("usageDetails")
    meta = row.get("metadata", {})
    if (not isinstance(raw, dict) or meta.get("usage_available") is not True
            or meta.get("usage_mapping_status") != "inclusive_provider_counts_to_exclusive_buckets"):
        return None
    # Our mapper writes exclusive buckets; Cloud may also add a derived total.
    if not all(_number(raw.get(k)) for k in ("input", "output")):
        return None
    if any(not _number(raw[k]) for k in ("input_cached_tokens", "output_reasoning_tokens") if k in raw):
        return None
    if raw.keys() - {"input", "output", "total", "input_cached_tokens", "output_reasoning_tokens"}:
        return None
    input_tokens = raw["input"] + raw.get("input_cached_tokens", 0)
    output_tokens = raw["output"] + raw.get("output_reasoning_tokens", 0)
    if "total" in raw and (not _number(raw["total"]) or raw["total"] != input_tokens + output_tokens):
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens}


def summarize_execution(trace_id, observations, scores=()):
    """Explain recorded bindings; absence is not proof that a stage never ran."""
    unique = {}
    for row in observations:
        if (not isinstance(row, dict) or row.get("traceId") != trace_id
                or not isinstance(row.get("id"), str) or not isinstance(row.get("metadata", {}), dict)):
            _fail("LANGFUSE_QUERY_OBSERVATION_IDENTITY_MISMATCH")
        if row["id"] in unique and unique[row["id"]] != row:
            _fail("LANGFUSE_QUERY_CONFLICTING_OBSERVATION")
        unique[row["id"]] = row
    rows = sorted(unique.values(), key=lambda r: (r.get("startTime") or "", r["id"]))
    generations = [r for r in rows if r.get("type") == "GENERATION"]
    roots = [r for r in rows if r.get("isRootObservation") is True]
    tools = [r for r in rows if r.get("name") == "source.inspect" and r.get("type") == "TOOL"]
    generation_results, links, verification = [], [], []
    for row in generations:
        meta = row.get("metadata", {})
        generation_results.append({"observation_id": row["id"], "parent_id": row.get("parentObservationId"),
            "model": row.get("providedModelName"), "level": row.get("level"), "status_message": row.get("statusMessage"),
            "duration_ms": _milliseconds(row.get("startTime"), row.get("endTime")),
            "ttft_ms": _milliseconds(row.get("startTime"), row.get("completionStartTime")),
            "usage": _usage(row), "cost": row.get("totalCost") if _number(row.get("totalCost")) else None,
            "prompt_name": row.get("promptName") or None, "facts": {k: meta[k] for k in _FACTS if k in meta}})
        consumed = meta.get("consumed_tool_results")
        if isinstance(consumed, list):
            for fact in consumed:
                if not isinstance(fact, dict):
                    continue
                call, digest = fact.get("call_id_sha256"), fact.get("observation_sha256")
                matches = [t["id"] for t in tools if isinstance(call, str) and isinstance(digest, str)
                           and t.get("metadata", {}).get("call_id_sha256") == call
                           and t.get("metadata", {}).get("encoded_payload_sha256") == digest]
                links.append({"generation_id": row["id"], "source_observation_ids": matches,
                    "call_id_sha256": call, "observation_sha256": digest,
                    "binding": "matched_recorded_consumption" if len(matches) == 1 else "unresolved"})
    for row in rows:
        if row.get("name") != "patch.verify":
            continue
        meta = row.get("metadata", {})
        output = meta.get("output_sha256")
        matches = [g["id"] for g in generations if isinstance(output, str)
                   and g.get("metadata", {}).get("output_sha256") == output]
        verification.append({"observation_id": row["id"], "level": row.get("level"),
            "matching_generation_ids": matches,
            "binding": "recorded_output_hash_match" if matches else "unresolved",
            "facts": {k: meta[k] for k in _FACTS if k in meta}})
    score_results = []
    for score in scores:
        subject = score.get("subject") if isinstance(score, dict) else None
        if (not isinstance(subject, dict)
                or not (subject.get("traceId") == trace_id or subject.get("kind") == "trace" and subject.get("id") == trace_id)):
            _fail("LANGFUSE_QUERY_SCORE_IDENTITY_MISMATCH")
        score_results.append({k: score[k] for k in ("id", "name", "dataType", "value", "source", "subject") if k in score})
    known_usage = [g["usage"] for g in generation_results if g["usage"] is not None]
    durations = [g for g in generation_results if g["duration_ms"] is not None]
    parent_missing = sorted({r["parentObservationId"] for r in rows
                             if r.get("parentObservationId") and r["parentObservationId"] not in unique})
    return {"trace_id": trace_id, "source": "langfuse_cloud", "read_only": True,
        "session_ids": sorted({r["sessionId"] for r in rows if isinstance(r.get("sessionId"), str)}),
        "observation_count": len(rows), "levels": dict(Counter(r.get("level") for r in rows)),
        "generation_count": len(generations), "model_request_observations": sum(r.get("name") == "model.request" for r in rows),
        "generation_count_scope": "recorded_generations_not_reserved_worker_calls",
        "generations": generation_results,
        "slowest_measured_generation_id": max(durations, key=lambda r: r["duration_ms"])["observation_id"] if durations else None,
        "usage": {"measured_generations": len(known_usage), "missing_generations": len(generations) - len(known_usage),
                  "measured_totals": {k: sum(u[k] for u in known_usage) for k in ("input_tokens", "output_tokens", "total_tokens")} if known_usage else None},
        "verification": verification, "tool_consumption_links": links,
        "symbol_materialization_checks": [{"observation_id": row["id"], "level": row.get("level"),
            "facts": {key: row.get("metadata", {})[key] for key in _FACTS if key in row.get("metadata", {})},
            "scores": [score for score in score_results if score.get("name") == SYMBOL_CHECK_SCORE_NAME
                and score.get("subject", {}).get("kind") == "observation" and score["subject"].get("id") == row["id"]],
            "scope": "declaration_identity_not_relevance_or_task_acceptance"}
            for row in rows if row.get("name") == SYMBOL_CHECK_OPERATION],
        **_upstream_evidence(rows, generations),
        "roots": [{"observation_id": r["id"], "facts": {k: r.get("metadata", {})[k] for k in _FACTS if k in r.get("metadata", {})}} for r in roots],
        "upstream_requests": [{"observation_id": r["id"], "facts": {k: r.get("metadata", {})[k] for k in _FACTS if k in r.get("metadata", {})}}
                              for r in rows if r.get("name") == "patch.upstream_request"],
        "observed_operations": sorted({r["name"] for r in rows if isinstance(r.get("name"), str)}),
        "execution_tree": [{"observation_id": r["id"], "parent_id": r.get("parentObservationId"),
            "type": r.get("type"), "name": r.get("name"), "level": r.get("level"), "status_message": r.get("statusMessage"),
            "start_time": r.get("startTime"), "end_time": r.get("endTime"),
            "duration_ms": _milliseconds(r.get("startTime"), r.get("endTime")),
            "facts": {k: r.get("metadata", {})[k] for k in _FACTS if k in r.get("metadata", {})}} for r in rows],
        "missing_parent_observation_ids": parent_missing, "scores": score_results,
        "limits": ["Default is observation severity, not task success.",
                   "Missing operations are unobserved; execution/nonexecution requires bound journal evidence.",
                   "Hash joins show recorded identity, not semantic correctness or a unique causal explanation.",
                   "No raw prompt, response or source was requested; full Mod delivery is not inferred."]}


def join_local_journal(result, journal_path, journal_sha256, *, artifact_roots=()):
    """Bind explicitly selected local evidence; do not follow Cloud-supplied paths.

    The journal hash and observation mapping must match before returning local
    references. Reading a body file additionally requires an explicit local root.
    Body content is never inserted in the Cloud projection or returned summary.
    """
    from src.observability.patch_langfuse import MAX_PAYLOAD_BYTES
    from src.observability.patch_live_langfuse import _metadata, live_observation_id, live_trace_id

    path = Path(journal_path).resolve(strict=True)
    if not isinstance(journal_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", journal_sha256):
        _fail("LANGFUSE_LOCAL_JOURNAL_HASH_INVALID")
    if path.stat().st_size > MAX_PAYLOAD_BYTES:
        _fail("LANGFUSE_LOCAL_JOURNAL_TOO_LARGE")
    raw = path.read_bytes()
    if len(raw) > MAX_PAYLOAD_BYTES or hashlib.sha256(raw).hexdigest() != journal_sha256:
        _fail("LANGFUSE_LOCAL_JOURNAL_HASH_MISMATCH")
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeError):
        _fail("LANGFUSE_LOCAL_JOURNAL_INVALID")
    local_rows = document.get("execution_observations") if isinstance(document, dict) else None
    if not isinstance(local_rows, dict):
        _fail("LANGFUSE_LOCAL_JOURNAL_OBSERVATIONS_MISSING")
    roots = [Path(root).resolve(strict=True) for root in artifact_roots]
    indexed = {}
    for row in local_rows.values():
        if not isinstance(row, dict):
            _fail("LANGFUSE_LOCAL_JOURNAL_OBSERVATION_INVALID")
        if live_trace_id(row.get("trace_id")) == result["trace_id"]:
            identity = live_observation_id(row.get("id"))
            if identity in indexed and indexed[identity] != row:
                _fail("LANGFUSE_LOCAL_JOURNAL_CONFLICTING_OBSERVATION")
            indexed[identity] = row
    if not indexed:
        _fail("LANGFUSE_LOCAL_JOURNAL_TRACE_MISMATCH")
    joins = []
    for cloud in result["execution_tree"]:
        entry = {"observation_id": cloud["observation_id"], "binding": "local_observation_not_found", "details": []}
        local = indexed.get(cloud["observation_id"])
        if local is None:
            joins.append(entry)
            continue
        attrs = local.get("attributes", {})
        mapped = _metadata(attrs)
        expected_facts = {key: mapped[key] for key in _FACTS if key in mapped}
        cloud_facts = cloud.get("facts", {})
        parent = live_observation_id(local["parent_id"]) if local.get("parent_id") is not None else None
        same = (local.get("name") == cloud.get("name") and str(local.get("kind")).upper() == cloud.get("type")
                and parent == cloud.get("parent_id")
                and _same_observed_time(local.get("started_at"), cloud.get("start_time"))
                and _same_observed_time(local.get("finished_at"), cloud.get("end_time"))
                and all(cloud_facts.get(key) == value for key, value in expected_facts.items()))
        if not same:
            entry["binding"] = "local_cloud_observation_mismatch"
            joins.append(entry)
            continue
        entry["binding"] = "verified_journal_and_observation_mapping"
        if mapped.get("requirement_findings_attempt_audit_sha256"):
            pointer_id = str(local["id"]).replace("~", "~0").replace("/", "~1")
            entry["findings_attempt_audit_ref"] = {
                "path": str(path),
                "json_pointer": "/execution_observations/" + pointer_id + "/attributes/requirement_findings_attempt_audit",
                "value_sha256": mapped["requirement_findings_attempt_audit_sha256"],
                "hash_scope": "canonical_json_value"}
        if mapped.get("symbol_check_sha256") and isinstance(attrs.get("symbol_check"), dict):
            pointer_id = str(local["id"]).replace("~", "~0").replace("/", "~1")
            entry["symbol_check_ref"] = {"path": str(path),
                "json_pointer": "/execution_observations/" + pointer_id + "/attributes/symbol_check",
                "value_sha256": mapped["symbol_check_sha256"], "hash_scope": "canonical_json_value"}
        details = attrs.get("local_detail_refs", {})
        for role in ("request", "response", "planning_receipt", "response_tools", "failure", "terminal_record"):
            reference = attrs.get("terminal_record_ref") if role == "terminal_record" else details.get(role) if isinstance(details, dict) else None
            exported = mapped.get("terminal_record_ref") if role == "terminal_record" else mapped.get("local_detail_refs", {}).get(role)
            if not isinstance(reference, dict) or exported != {"sha256": reference.get("sha256")}:
                continue
            item = {"role": role, "sha256": reference["sha256"], "file_binding": "not_checked_no_allowed_root"}
            try:
                detail_path = Path(reference["path"]).resolve(strict=True)
                item["path"] = str(detail_path)
                if roots:
                    if not any(detail_path.is_relative_to(root) for root in roots):
                        item["file_binding"] = "outside_explicit_artifact_roots"
                    elif detail_path.stat().st_size > MAX_PAYLOAD_BYTES:
                        item["file_binding"] = "file_too_large"
                    else:
                        body = detail_path.read_bytes()
                        item["file_binding"] = "verified_bytes" if len(body) <= MAX_PAYLOAD_BYTES and hashlib.sha256(body).hexdigest() == reference["sha256"] else "file_hash_mismatch"
            except (OSError, KeyError, TypeError, ValueError):
                item["file_binding"] = "file_unavailable"
            entry["details"].append(item)
        joins.append(entry)
    return {**result, "local_evidence": {"journal_path": str(path), "journal_sha256": journal_sha256,
        "journal_binding": "verified_bytes", "observations": joins,
        "local_only_observation_ids": sorted(set(indexed) - {row["observation_id"] for row in result["execution_tree"]}),
        "scope": "explicit_local_join_no_body_upload_no_semantic_correctness_claim"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read one Langfuse execution and its recorded diagnostic bindings.")
    parser.add_argument("--trace-id", required=True)
    # 为什么要有它：公共CLI不能接受缺少当前云接口必填时间窗的旧用法，再把失败留到远端。
    parser.add_argument("--from-start-time", required=True, help="Recorded inclusive observation start-time lower bound")
    parser.add_argument("--to-start-time", required=True, help="Exclusive upper bound; at most 336 hours after lower bound")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-journal", type=Path)
    parser.add_argument("--local-journal-sha256")
    parser.add_argument("--local-artifact-root", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    if bool(args.local_journal) != bool(args.local_journal_sha256) or args.local_artifact_root and not args.local_journal:
        parser.error("--local-journal and --local-journal-sha256 are required together; local roots require a journal.")
    result = CloudExecutionReader(LangfuseHttpTransport.from_environment()).read(args.trace_id,
        from_start_time=args.from_start_time, to_start_time=args.to_start_time)
    if args.local_journal:
        result = join_local_journal(result, args.local_journal, args.local_journal_sha256,
                                    artifact_roots=args.local_artifact_root)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
