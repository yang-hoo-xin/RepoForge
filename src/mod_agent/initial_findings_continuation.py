"""Prepare hash-bound Findings revisions and explicitly approved original-Graph continuation.

Read-only preparation preserves the historical wire contract. A new correction
contract requires a separately displayed input identity and exact approval.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from src.libs.llm import validate_loopback_http_base_url
from src.mod_agent import requirement_coverage_model
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.build.baseline_repair import validated_baseline_repair_observation
from src.mod_agent.evidence.optimization_policy import run_optimization_policy
from src.mod_agent.evidence import (
    EvidencePack,
    RgLiveEvidenceProvider,
    StructuredModelRequirementCoverageProvider,
    model_view,
    requirement_coverage,
    requirement_coverage_provider,
)
from src.mod_agent.evidence.requirement_coverage import materialize_requirement_coverage_projection
from src.mod_agent.evidence.model_view import expand_findings_input_text
from src.mod_agent.evidence.source_view import (
    run_small_file_max_bytes,
    source_view_policy_from_manifest,
)
from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _python_runtime_identity():
    executable = Path(sys.executable).resolve(strict=True)
    return {"executable": str(executable),
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "version": platform.python_version()}


def _input_digest(value):
    selected = deepcopy(value)
    # Revalidation latency changes on every read. Keep the measured value in
    # the record while excluding only this observational field from approval identity.
    if isinstance(selected, dict) and "retrieved_evidence" in selected:
        selected["retrieved_evidence"].pop("timing", None)
    return _digest(selected)


def _file(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("INITIAL_FINDINGS_INPUT_TOO_LARGE")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}, json.loads(raw)


def _bound_json(reference, root):
    path = Path(reference["path"]).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("INITIAL_FINDINGS_INPUT_OUTSIDE_RUN")
    observed, value = _file(path)
    if observed != {"path": str(path), "sha256": reference["sha256"]}:
        raise ValueError("INITIAL_FINDINGS_INPUT_HASH_CHANGED")
    return observed, value


def _provider_backend_identity(value):
    if value is None:
        return {"status": "not_configured"}
    if (not isinstance(value, dict) or set(value) != {
            "ollama_version", "llama_server_version", "llama_cpp_commit", "binary_refs"}
            or any(not isinstance(value[key], str) or not value[key]
                   for key in ("ollama_version", "llama_server_version", "llama_cpp_commit"))
            or not isinstance(value["binary_refs"], list) or not value["binary_refs"]):
        raise ValueError("INITIAL_FINDINGS_PROVIDER_BACKEND_INVALID")
    refs = []
    roles = set()
    for reference in value["binary_refs"]:
        if (not isinstance(reference, dict) or set(reference) != {"role", "path", "sha256"}
                or not isinstance(reference["role"], str) or not reference["role"]
                or reference["role"] in roles):
            raise ValueError("INITIAL_FINDINGS_PROVIDER_BACKEND_INVALID")
        roles.add(reference["role"])
        path = Path(reference["path"]).resolve(strict=True)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != reference["sha256"]:
            raise ValueError("INITIAL_FINDINGS_PROVIDER_BACKEND_CHANGED")
        refs.append({"role": reference["role"], "path": str(path), "sha256": observed})
    return {"status": "configured", "ollama_version": value["ollama_version"],
            "llama_server_version": value["llama_server_version"],
            "llama_cpp_commit": value["llama_cpp_commit"],
            "binary_refs": refs}


def _budget(value):
    from src.mod_agent.model_call_budget import validate_budget
    try:
        return validate_budget(value)
    except ValueError as error:
        raise ValueError("INITIAL_FINDINGS_BUDGET_INVALID") from error


def _with_baseline_compilation_observation(payload, observation):
    if observation is None:
        return payload
    if ("baseline_compilation_failure" in payload
            and payload["baseline_compilation_failure"] != observation):
        raise ValueError("INITIAL_FINDINGS_BASELINE_CHANGED")
    return {**payload, "baseline_compilation_failure": deepcopy(observation)}


def _with_baseline_compilation_evidence(pack, observation):
    return replace(pack, budget_summary=_with_baseline_compilation_observation(
        pack.budget_summary, observation)) if observation is not None else pack


def _replay_source_query_set(*, root, materialized, request, search_root, historical_payload):
    """Reconstruct the recorded source allocation before checking the full Findings input."""
    source = RgLiveEvidenceProvider(root, small_file_max_bytes=run_small_file_max_bytes(root),
        reuse_query_observations=run_optimization_policy(root)["reuse_query_observations"])
    query_count = len(materialized["queries"])
    # 为什么要有它：历史请求绑定当时的材料预算；不能用当前默认扩大或缩小检索后再把差异当作源码漂移。
    max_items = historical_payload.get("source_material_budget", {}).get("max_items", max(1, query_count))
    retrieved = source.collect_query_set(
        queries=tuple(dict(row) for row in materialized["queries"]),
        source_identity_query=request, search_root=search_root, max_items=max_items)
    if "source_material_budget_selection" in historical_payload:
        # 为什么要有它：新owner的预算选择属于原wire，逐字段重建它而非复制历史元数据，保留后续完整输入比较。
        selection = {"schema_version": "requirement_material_budget_v1",
            "scope": "requirement_coverage_source_query_set_item_allocation",
            "query_count": query_count, "max_items": max_items,
            "small_project_material_floor": source.small_project_context_policy["max_files"],
            "basis": "max_of_query_count_and_bounded_small_project_material_floor",
            "semantic_coverage_claimed": False}
        retrieved = replace(retrieved, budget_summary={**retrieved.budget_summary,
            "source_material_budget_selection": selection})
    return retrieved


class _CorrectionPreparedError(RuntimeError):
    pass


class _InputChangedError(RuntimeError):
    pass


def prepare_initial_findings_continuation(*, run_root, run_id, execution_config=None,
                                         correction_contract_version=None):
    """Prepare one exact next turn, including a truthful stage-admission refusal."""
    root = Path(run_root).resolve(strict=True)
    manifest_ref, manifest = _file(root / "run_manifest.json")
    result_ref, result = _file(root / "run_result.json")
    if (manifest.get("run_id") != run_id or result.get("run_id") != run_id
            or Path(result["run_root"]).resolve() != root
            or manifest["request_id"] != result["product_execution_result"]["request_id"]
            or result.get("patch_execution_receipts") or result.get("actual_write_performed") is not False):
        raise ValueError("INITIAL_FINDINGS_TASK_IDENTITY_INVALID")
    if not any(row.get("stage") == "retrieve_evidence_execution" for row in result.get("blockers", [])):
        raise ValueError("INITIAL_FINDINGS_FAILURE_NOT_SELECTED")
    budget = _budget(result["model_call_budget"])
    if budget["consumed"] != result["model_call_count"]:
        raise ValueError("INITIAL_FINDINGS_BUDGET_INVALID")
    rows = sorted((row for row in manifest["execution_observations"].values()
                   if row["name"] == "model.request"), key=lambda row: row["started_at"])
    roles = [row["attributes"].get("model_role") for row in rows]
    saved_query = roles == ["requirement_projection", "requirement_findings", "initial_findings_queries"]
    if (roles != ["requirement_projection", "requirement_findings"] and not saved_query
            or budget["consumed"] != len(rows)):
        raise ValueError("INITIAL_FINDINGS_HISTORY_NOT_SUPPORTED")
    model = manifest["local_model_identity"]
    if (model != result["local_model_identity"] or not model.get("digest")
            or model.get("model") != manifest["model"]):
        raise ValueError("INITIAL_FINDINGS_MODEL_IDENTITY_INVALID")
    validate_loopback_http_base_url(model["base_url"])
    references = [manifest_ref, result_ref]
    configuration_refs = []
    configuration_blockers = []
    if execution_config is not None:
        from src.mod_agent.cli import build_parser
        config_ref, configuration = _file(execution_config)
        try:
            parsed = build_parser().parse_args(["run-request", *configuration["arguments"], "--", manifest["request"]])
        except SystemExit as exc:
            raise ValueError("INITIAL_FINDINGS_EXECUTION_CONFIG_INVALID") from exc
        stage_ref, stage_limits = (_file(parsed.stage_model_call_limits)
                                   if parsed.stage_model_call_limits is not None else (None, None))
        from src.mod_agent.model_call_budget import is_meter_only, total_only
        recorded_limits = manifest["ollama_patch_generation_profile"]
        if (parsed.run_id != run_id or parsed.request_id != manifest["request_id"]
                or Path(parsed.workspace_root).resolve() / run_id != root
                or Path(parsed.target_project).resolve() != Path(manifest["target_source"]).resolve()
                or parsed.ollama_model != model["model"] or parsed.ollama_base_url != model["base_url"]
                or not parsed.local_ollama or parsed.allow_source_disclosure or not parsed.defer_approval
                or parsed.task_model_call_limit != budget["maximum"]
                or parsed.reuse_query_observations != run_optimization_policy(root)["reuse_query_observations"]
                or parsed.compress_findings_input != run_optimization_policy(root)["compress_findings_input"]
                or parsed.ollama_context_window != recorded_limits["context_window"]
                or parsed.ollama_max_output_tokens != recorded_limits["output_budget"]
                or (parsed.ollama_timeout_seconds is not None and parsed.ollama_timeout_seconds != recorded_limits["timeout_seconds"])
                or not (total_only(budget) or is_meter_only(budget))
                and stage_limits != {key: row["maximum"] for key, row in budget["by_stage"].items()}):
            raise ValueError("INITIAL_FINDINGS_EXECUTION_CONFIG_CHANGED")
        configuration_refs = [config_ref, *([stage_ref] if stage_ref is not None else [])]
        for path in (parsed.settings, parsed.patch_worker_config, parsed.game_folder_build_binding):
            if path is not None:
                selected_path = Path(path).resolve(strict=True)
                configuration_refs.append({"path": str(selected_path),
                    "sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest()})
    else:
        configuration_blockers.append("INITIAL_FINDINGS_EXECUTION_CONFIG_REQUIRED")
    bodies = []
    for row in rows:
        attributes = row["attributes"]
        if (attributes.get("transport_started") is not True or attributes.get("response_complete") is not True
                or attributes.get("model") != model["model"]):
            raise ValueError("INITIAL_FINDINGS_RESPONSE_INCOMPLETE")
        request_ref, request = _bound_json(attributes["local_detail_refs"]["request"], root)
        response_ref, response = _bound_json(attributes["local_detail_refs"]["response"], root)
        references.extend([request_ref, response_ref])
        from src.mod_agent.local_entry import _request_model_output_budget
        expected_output_budget = (_request_model_output_budget(attributes["model_role"], parsed.ollama_max_output_tokens)
                                  if execution_config is not None else None)
        if execution_config is not None and (
                request.get("max_tokens") != expected_output_budget
                or request.get("temperature") != 0.0 or request.get("stream_response") is not False):
            raise ValueError("INITIAL_FINDINGS_MODEL_REQUEST_OPTIONS_CHANGED")
        bodies.append((request, response))
    selected = [row["artifact_ref"] for row in result["product_execution_result"]["outputs"]
                if row["artifact_role"] == "feature_spec"]
    if len(selected) != 1:
        raise ValueError("INITIAL_FINDINGS_FEATURE_NOT_UNIQUE")
    feature_ref = ArtifactRef(**selected[0])
    feature = ArtifactStore(root / "a").semantic_read_immutable(feature_ref)["data"]
    # RequestPlanner owns strip normalization; the raw request stays hash-bound in manifest_ref.
    if feature["user_request"] != manifest["request"].strip() or feature_ref.feature_id != result["feature_id"]:
        raise ValueError("INITIAL_FINDINGS_REQUEST_CHANGED")
    source = Path(manifest["target_source"]).resolve(strict=True)
    generated = root / "generated_mods" / source.name
    if (source_snapshot(source) != manifest["source_snapshot"]
            or source_snapshot(generated) != manifest["source_snapshot"]
            or source_snapshot_sha256(manifest["source_snapshot"]) != manifest["source_snapshot_sha256"]):
        raise ValueError("INITIAL_FINDINGS_SOURCE_CHANGED")
    for row in manifest["source_snapshot"]:
        for base in (source, generated):
            path = (base / row["path"]).resolve(strict=True)
            if not path.is_relative_to(base) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("INITIAL_FINDINGS_SOURCE_CHANGED")

    baseline_observation = validated_baseline_repair_observation(manifest, root)
    if saved_query:
        if execution_config is None:
            raise ValueError("INITIAL_FINDINGS_EXECUTION_CONFIG_REQUIRED")
        from src.mod_agent.saved_reassessment import prepare_saved_query_finish
        return prepare_saved_query_finish(root=root, manifest=manifest, result=result,
            result_ref=result_ref, references=references, configuration_refs=configuration_refs,
            configuration_blockers=configuration_blockers, budget=budget, bodies=bodies,
            feature_ref=selected[0], feature=feature, parsed=parsed,
            contract_version=correction_contract_version)

    # Reconstruct the existing owner input, not a new prompt or interpretation.
    seen_projection = []
    def replay_projection(role, payload, schema):
        payload = _with_baseline_compilation_observation(payload, baseline_observation)
        if role != "requirement_projection" or seen_projection:
            raise _InputChangedError("INITIAL_FINDINGS_PROJECTION_CHANGED")
        seen_projection.append(role)
        original = bodies[0][0]
        if (expand_findings_input_text(json.loads(next(m["content"] for m in original["messages"] if m["role"] == "user"))) != expand_findings_input_text(payload)
                or original["response_format"] != schema):
            raise _InputChangedError("INITIAL_FINDINGS_PROJECTION_CHANGED")
        return deepcopy(bodies[0][1])
    projector = StructuredModelRequirementCoverageProvider(replay_projection, compress_findings_input=run_optimization_policy(root)["compress_findings_input"])
    projection = projector.project(feature_data=feature, request=feature["user_request"])
    materialized = materialize_requirement_coverage_projection(
        feature_data=feature, request=feature["user_request"], projection=projection)
    if result.get("source_retrieval_mode") != "rg_live":
        raise ValueError("INITIAL_FINDINGS_RETRIEVAL_MODE_NOT_SUPPORTED")
    original_findings_payload = json.loads(next(m["content"] for m in bodies[1][0]["messages"] if m["role"] == "user"))
    retrieved = _replay_source_query_set(root=root, materialized=materialized,
        request=feature["user_request"], search_root=generated.relative_to(root).as_posix(),
        historical_payload=original_findings_payload)
    prepared = {}
    def capture(role, payload, schema):
        payload = _with_baseline_compilation_observation(payload, baseline_observation)
        original = bodies[1][0]
        if role == "requirement_findings":
            # 为什么要有它：历史原文与同消息引用必须逐字段还原一致，兼容编码不能隐藏需求或元数据变更。
            if (expand_findings_input_text(json.loads(next(m["content"] for m in original["messages"] if m["role"] == "user"))) != expand_findings_input_text(payload)
                    or original["response_format"] != schema):
                raise _InputChangedError("INITIAL_FINDINGS_CURRENT_INPUT_DIFFERS")
            return deepcopy(bodies[1][1])
        if role != "requirement_findings_correction":
            raise ValueError("INITIAL_FINDINGS_ROLE_CHANGED")
        prepared.update(role=role, payload=deepcopy(payload), schema=deepcopy(schema))
        raise _CorrectionPreparedError()
    historical_version = original_findings_payload.get("findings_contract_version", "1")
    # 为什么要有它：历史零证据wire没有新policy标记时须保持旧schema，不能把协议升级当作原输入漂移。
    provider = StructuredModelRequirementCoverageProvider(capture, compress_findings_input=run_optimization_policy(root)["compress_findings_input"], findings_contract_version=historical_version,
        empty_evidence_policy=original_findings_payload.get("empty_evidence_policy"),
        findings_planning_basis_policy=original_findings_payload.get("findings_planning_basis_policy"),
        native_findings_contract=original_findings_payload.get("native_findings_contract"))
    try:
        provider.findings(materialized_projection=materialized, retrieved_pack=retrieved, request=feature["user_request"])
    except _CorrectionPreparedError:
        pass
    if not prepared or not provider.findings_attempt_audit or provider.findings_attempt_audit[0]["status"] != "rejected":
        raise ValueError("INITIAL_FINDINGS_REJECTION_NOT_REPRODUCED")
    first = deepcopy(provider.findings_attempt_audit[0])
    historical_turn = deepcopy(prepared)
    selected_historical_version = historical_version
    rejected_findings = deepcopy(bodies[1][1])
    recorded_audit = None
    terminal = result["blockers"]
    execution_parent_result_ref = result_ref

    def prepare_revision(version, rejected, *, historical_payload=None):
        captured = {}
        def capture_revision(role, payload, schema):
            payload = _with_baseline_compilation_observation(payload, baseline_observation)
            if role == "requirement_findings":
                return deepcopy(rejected)
            if role != "requirement_findings_correction":
                raise ValueError("INITIAL_FINDINGS_ROLE_CHANGED")
            captured.update(role=role, payload=deepcopy(payload), schema=deepcopy(schema))
            raise _CorrectionPreparedError()
        # 为什么要有它：重建已经执行过的纠正请求须继承其语义标记，新派生纠正才使用当前默认协议。
        historical_policies = ({key: historical_payload.get(key) for key in
            ("empty_evidence_policy", "findings_planning_basis_policy", "native_findings_contract")} if historical_payload is not None else {})
        revised = StructuredModelRequirementCoverageProvider(
            capture_revision, compress_findings_input=run_optimization_policy(root)["compress_findings_input"], findings_contract_version=version, **historical_policies)
        try:
            revised.findings(materialized_projection=materialized,
                             retrieved_pack=retrieved, request=feature["user_request"])
        except _CorrectionPreparedError:
            pass
        if not captured or not revised.findings_attempt_audit:
            raise ValueError("INITIAL_FINDINGS_CORRECTION_REVISION_NOT_PREPARED")
        return captured, deepcopy(revised.findings_attempt_audit[0])

    # Version 3 may continue only from the recorded version-2 schema failure.
    if correction_contract_version == requirement_coverage_provider.FINDINGS_CONTRACT_VERSION:
        prior = read_initial_findings_execution(root, claim_name="initial_findings.json")
        if prior is not None:
            if (prior.get("findings_contract", {}).get("selected_version")
                    != requirement_coverage_provider.PREVIOUS_FINDINGS_CONTRACT_VERSION
                    or prior.get("status") != "partial"
                    or prior.get("current_invocation_model_call_count") != 1
                    or prior.get("actual_write_performed") is not False
                    or prior.get("canonical_source_unchanged") is not True
                    or not any("REQUIREMENT_COVERAGE_FINDINGS_SCHEMA_INVALID" in row.get("message", "")
                               for row in prior.get("blockers", []))):
                raise ValueError("INITIAL_FINDINGS_PRIOR_REVISION_NOT_SELECTED")
            prior_root = Path(prior["continuation_root"]).resolve(strict=True)
            prior_manifest_ref, prior_manifest = _bound_json(
                {"path": str(prior_root / "run_manifest.json"),
                 "sha256": hashlib.sha256((prior_root / "run_manifest.json").read_bytes()).hexdigest()}, root)
            prior_result_ref, prior_result = _bound_json(prior["result_ref"], root)
            generation = [row for row in prior_manifest["execution_observations"].values()
                          if row["name"] == "requirements.findings"
                          and row["attributes"].get("model_role") == "requirement_findings_correction"]
            if len(generation) != 1 or generation[0]["attributes"].get("response_complete") is not True:
                raise ValueError("INITIAL_FINDINGS_PRIOR_REVISION_RESPONSE_INVALID")
            prior_request_ref, prior_request = _bound_json(
                generation[0]["attributes"]["local_detail_refs"]["request"], root)
            prior_response_ref, prior_response = _bound_json(
                generation[0]["attributes"]["local_detail_refs"]["response"], root)
            v2_turn, _ = prepare_revision(
                requirement_coverage_provider.PREVIOUS_FINDINGS_CONTRACT_VERSION,
                bodies[1][1], historical_payload=json.loads(next(row["content"] for row in prior_request["messages"]
                                                                 if row["role"] == "user")))
            if (not requirement_coverage_model.requirement_coverage_messages_match(
                    v2_turn["payload"], prior_request.get("messages"))
                    or prior_request.get("response_format") != v2_turn["schema"]
                    or prior_request.get("max_tokens") != parsed.ollama_max_output_tokens
                    or prior_request.get("temperature") != 0.0
                    or prior_request.get("stream_response") is not False
                    or prior_result.get("model_call_count") != prior["model_call_count"]):
                raise ValueError("INITIAL_FINDINGS_PRIOR_REVISION_INPUT_CHANGED")
            references.extend([prior_manifest_ref, prior_result_ref,
                               prior_request_ref, prior_response_ref])
            budget = _budget(prior["model_call_budget"])
            selected_historical_version = requirement_coverage_provider.PREVIOUS_FINDINGS_CONTRACT_VERSION
            historical_turn = v2_turn
            rejected_findings = prior_response
            execution_parent_result_ref = prior_result_ref
            prior_coverage = [row for row in prior_manifest["execution_observations"].values()
                              if row["name"] == "requirements.coverage"]
            recorded_audit = (prior_coverage[-1]["attributes"].get(
                "requirement_findings_attempt_audit") if prior_coverage else None)
            terminal = prior_result["blockers"]

    if correction_contract_version is not None and correction_contract_version != selected_historical_version:
        if correction_contract_version not in {
                requirement_coverage_provider.PREVIOUS_FINDINGS_CONTRACT_VERSION,
                requirement_coverage_provider.FINDINGS_CONTRACT_VERSION}:
            raise ValueError("INITIAL_FINDINGS_CORRECTION_VERSION_INVALID")
        prepared, first = prepare_revision(correction_contract_version, rejected_findings)
        if prepared == historical_turn:
            raise ValueError("INITIAL_FINDINGS_CORRECTION_REVISION_NOT_PREPARED")
    stage = "requirement_findings"
    blockers = list(configuration_blockers)
    from src.mod_agent.model_call_budget import is_meter_only, total_only
    if not is_meter_only(budget) and budget["remaining"] < 1:
        blockers.append("PRODUCT_MODEL_CALL_BUDGET_EXHAUSTED")
    if not (total_only(budget) or is_meter_only(budget)) and budget["by_stage"][stage]["remaining"] < 1:
        blockers.append("PRODUCT_STAGE_MODEL_CALL_BUDGET_EXHAUSTED")
    packet = {"run_id": run_id, "request_id": manifest["request_id"], "feature_spec_ref": selected[0],
              "parent_refs": references, "configuration_refs": configuration_refs,
              "execution_parent_result_ref": execution_parent_result_ref,
              "source_snapshot_sha256": manifest["source_snapshot_sha256"],
              "model_identity": model, "budget": budget, "budget_stage": stage, "next_turn": prepared,
              "source_retrieval_mode": result["source_retrieval_mode"],
              "evidence_optimization": run_optimization_policy(root),
              "python_runtime_identity": _python_runtime_identity()}
    packet["correction_contract"] = {"historical_version": selected_historical_version,
        "selected_version": correction_contract_version or selected_historical_version,
        "historical_turn_sha256": _digest(historical_turn),
        "selected_turn_sha256": _digest(prepared),
        "historical_turn": historical_turn,
        "revision_requires_new_approval": (correction_contract_version or selected_historical_version) != selected_historical_version}
    packet["model_limits"] = ({"timeout_seconds": recorded_limits["timeout_seconds"],
        "context_window": parsed.ollama_context_window, "output_tokens": parsed.ollama_max_output_tokens}
        if execution_config is not None else None)
    packet["admitted_projection"] = deepcopy(projection)
    packet["materialized_projection"] = deepcopy(materialized)
    packet["retrieved_evidence"] = retrieved.to_artifact_payload()
    execution_roots = [row for row in manifest["execution_observations"].values()
                       if row["name"] == "product.execution" and row["parent_id"] is None]
    packet["prior_execution_timing"] = {
        "started_at": manifest.get("started_at"),
        "elapsed_ms": execution_roots[0].get("elapsed_ms") if len(execution_roots) == 1 else None,
        "new_time_window_granted": False,
    }
    from src.mod_agent import local_entry, product_executor, request_patch_execution
    if original_findings_payload.get("native_findings_contract") is not None:
        packet["native_findings_contract"] = original_findings_payload["native_findings_contract"]
    from src.mod_agent.evidence import native_findings_contract, reassessment
    from src.mod_agent.workflow import graph
    packet["implementation_refs"] = [{"path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in (
            __file__, requirement_coverage.__file__, requirement_coverage_provider.__file__, requirement_coverage_model.__file__, model_view.__file__, native_findings_contract.__file__,
            local_entry.__file__, reassessment.__file__, product_executor.__file__, request_patch_execution.__file__, graph.__file__)]
    prepared["model_request"] = {**deepcopy(bodies[1][0]),
        "messages": [{"role": m.role, "content": m.content}
                     for m in requirement_coverage_model.requirement_coverage_messages(prepared["payload"])],
        "response_format": deepcopy(prepared["schema"])}
    for original, payload in ((bodies[0][0], json.loads(bodies[0][0]["messages"][1]["content"])),
                              (bodies[1][0], json.loads(bodies[1][0]["messages"][1]["content"]))):
        if not requirement_coverage_model.requirement_coverage_messages_match(payload, original["messages"]):
            raise _InputChangedError("INITIAL_FINDINGS_MESSAGE_CONTRACT_CHANGED")
    if recorded_audit is None:
        coverage = [row for row in manifest["execution_observations"].values()
                    if row["name"] == "requirements.coverage"]
        recorded_audit = (coverage[-1]["attributes"].get(
            "requirement_findings_attempt_audit") if coverage else None)
    return {"entry": "initial-findings-continuation", "run_id": run_id, "read_only": True,
            "status": "prepared_blocked" if blockers else "prepared_awaiting_authorization",
            "model_calls_performed": 0, "product_source_writes_performed": False, "graph_invoked": False,
            "budget": budget, "budget_stage": stage, "blockers": blockers,
            "model_invocation_authorized": False, "current_live_model_identity_checked": False,
            "cause_chain": {"historical_trace_audit": recorded_audit,
                            "first_rejection_reconstruction": first, "reconstruction_origin": "offline_exact_input_replay",
                            "original_relation_fields": deepcopy(rejected_findings["findings"]),
                            "terminal_blockers": terminal, "correction_payload_prepared": True,
                            "correction_model_request_sent": False if blockers else None,
                            "correction_model_request_sent_basis": "offline_preparation_performs_no_model_call",
                            "historical_correction_boundary": recorded_audit[-1].get("call_boundary") if recorded_audit else None,
                            "history_rewritten": False},
            "continuation_input": packet, "continuation_input_sha256": _input_digest(packet)}


def admit_initial_findings_continuation(prepared, *, run_root, run_id, execution_config=None):
    """Recheck identities and inherited allowances; this function never authorizes a call."""
    current = prepare_initial_findings_continuation(run_root=run_root, run_id=run_id, execution_config=execution_config)
    if (prepared.get("continuation_input_sha256") != _input_digest(prepared.get("continuation_input"))
            or prepared["continuation_input_sha256"] != current["continuation_input_sha256"]):
        raise ValueError("INITIAL_FINDINGS_CONTINUATION_INPUT_CHANGED")
    if current["blockers"]:
        raise ValueError(current["blockers"][0])
    return current


def _prepare_accepted_findings_reassessment(prepared, *, run_root, run_id, resume_mode,
                                          parent_override=None, feature_ref_override=None):
    """Bind the accepted v3 Findings pack as the next Evidence-node input."""
    from src.mod_agent.evidence.reassessment import (
        initial_findings_reassessment_allowed,
        prepare_initial_findings_reassessment_revision,
    )

    root = Path(run_root).resolve(strict=True)
    revision_number = None
    if resume_mode == "accepted_findings_reassessment_revision":
        parent_claim, _, revision_number = _reassessment_revision_claim_names(root)
    else:
        parent_claim = f"initial_findings-v{requirement_coverage_provider.FINDINGS_CONTRACT_VERSION}.json"
    parent = parent_override if parent_override is not None else read_initial_findings_execution(root, claim_name=parent_claim)
    saved_completion = (parent is None and revision_number == 1
        and bool(prepared["continuation_input"].get("saved_reassessment_query")))
    if saved_completion:
        parent = read_initial_findings_execution(root,
            claim_name=f"initial_findings-v{requirement_coverage_provider.FINDINGS_CONTRACT_VERSION}.json")
    accepted_parent_status = (parent or {}).get("status") in {"domain_blocked", "partial"}
    if (parent is None or parent.get("run_id") != run_id
            or not accepted_parent_status
            or (parent.get("status") == "partial"
                and parent.get("product_execution_result", {}).get("failure_stage")
                    != "retrieve_evidence_execution")
            or parent.get("model_call_count") != parent.get("model_call_budget", {}).get("consumed")
            or parent.get("current_invocation_model_call_count", 0) < 1
            or parent.get("findings_contract", {}).get("selected_version")
                != requirement_coverage_provider.FINDINGS_CONTRACT_VERSION
            or parent.get("actual_write_performed") is not False
            or parent.get("canonical_source_unchanged") is not True
            or parent.get("patch_execution_receipts")):
        raise ValueError("INITIAL_FINDINGS_ACCEPTED_PARENT_INVALID")
    outputs = [row["artifact_ref"] for row in parent["product_execution_result"]["outputs"]
               if row["artifact_role"] == "evidence_pack"]
    if resume_mode == "accepted_findings_reassessment_revision":
        current_path = parent.get("artifacts", {}).get("evidence_pack")
        outputs = [reference for reference in outputs
                   if reference.get("path") == current_path]
    if len(outputs) != 1:
        raise ValueError("INITIAL_FINDINGS_ACCEPTED_EVIDENCE_NOT_UNIQUE")
    store = ArtifactStore(root / "a")
    evidence_ref = ArtifactRef(**outputs[0])
    evidence = store.semantic_read_immutable(evidence_ref)["data"]
    feature_ref = ArtifactRef(**(feature_ref_override or prepared["continuation_input"]["feature_spec_ref"]))
    feature = store.semantic_read_immutable(feature_ref)["data"]
    prior_reassessment = None
    reuse_projected_reassessment_input = False
    admitted_evidence = deepcopy(evidence)
    completion_receipt = None
    if saved_completion:
        from src.mod_agent.saved_reassessment import verify_completed_saved_findings
        admitted_evidence, completion_receipt = verify_completed_saved_findings(
            prepared=prepared, parent=parent, evidence=evidence, root=root)
    if resume_mode == "accepted_findings_reassessment_revision":
        prior_reassessment = deepcopy(
            admitted_evidence.get("budget_summary", {}).get("initial_findings_reassessment"))
        projected_revision_number = admitted_evidence.get("budget_summary", {}).get(
            "reassessment_revision_number")
        if (parent.get("status") == "partial" and prior_reassessment is None
                and projected_revision_number in {1, 2}
                and initial_findings_reassessment_allowed(admitted_evidence, feature)):
            revision_number = projected_revision_number
            reuse_projected_reassessment_input = True
        else:
            admitted_evidence = prepare_initial_findings_reassessment_revision(
                admitted_evidence, feature, revision_number=revision_number)
    elif not initial_findings_reassessment_allowed(admitted_evidence, feature):
        raise ValueError("INITIAL_FINDINGS_ACCEPTED_REASSESSMENT_NOT_ADMITTED")
    from src.mod_agent import local_entry as owner
    admitted_pack = EvidencePack.from_artifact_payload(admitted_evidence)
    materialized = {key: deepcopy(admitted_pack.requirement_coverage[key]) for key in (
        "schema_version", "feature_spec_identity", "request_identity", "projection_identity",
        "input_obligations", "input_dispositions", "requirements", "relations", "queries")}
    feedback = {"initial_findings_reassessment": {
        "prior_evidence_hash": evidence_ref.artifact_hash,
        "findings": deepcopy(admitted_pack.requirement_coverage["findings"]),
        "sufficiency": deepcopy(admitted_pack.requirement_coverage["sufficiency"]),
    }, "interpretation_is_not_observation": True}
    query_model_contract = owner._build_request_upstream_query_contract(
        feature=deepcopy(feature), projection=materialized, old=admitted_pack,
        feedback=feedback, upstream_search_root=f"generated_mods/{feature['target_mod']}")
    _, manifest = _file(root / "run_manifest.json")
    query_model_contract["payload"] = _with_baseline_compilation_observation(
        query_model_contract["payload"], validated_baseline_repair_observation(manifest, root))
    packet = deepcopy(prepared["continuation_input"])
    if completion_receipt is not None:
        packet.pop("saved_reassessment_query")
        packet["saved_reassessment_completion"] = completion_receipt
        packet["parent_refs"].extend(deepcopy(completion_receipt[key]) for key in (
            "manifest_ref", "request_ref", "provider_request_ref", "response_ref"))
    packet["parent_refs"].append(deepcopy(parent["result_ref"]))
    packet["execution_parent_result_ref"] = deepcopy(parent["result_ref"])
    packet["budget"] = _budget(parent["model_call_budget"])
    packet["resume_mode"] = resume_mode
    packet["next_turn"] = {"role": "accepted_findings_reassessment",
                           "evidence_pack_ref": outputs[0],
                           "prior_reassessment_lineage": prior_reassessment,
                           "reassessment_revision_number": revision_number,
                           "reuse_projected_reassessment_input": reuse_projected_reassessment_input,
                           "query_model_contract": query_model_contract,
                           "reassessment_input_sha256": _digest(admitted_evidence)}
    packet["prior_execution_timing"] = {
        "started_at": parent.get("execution_time_scope", {}).get("started_at"),
        "elapsed_ms": None,
        "new_time_window_granted": False,
    }
    result = {**prepared, "budget": packet["budget"], "budget_stage": "repair",
              "blockers": [], "status": "prepared_awaiting_authorization",
              "continuation_input": packet,
              "cause_chain": {**deepcopy(prepared["cause_chain"]),
                  "accepted_findings_parent": deepcopy(parent["result_ref"]),
                  "accepted_findings_reassessment_admitted": True}}
    result["continuation_input_sha256"] = _input_digest(packet)
    return result


def _preview_first_query_provider_request(packet, *, execution_config):
    """Freeze the exact first query request without HTTP or call reservation."""
    from src.core.settings import load_settings
    from src.libs.llm.ollama_llm import OllamaLLM, OllamaLLMError
    from src.mod_agent import local_entry as owner
    from src.mod_agent.cli import build_parser

    _, configuration = _file(execution_config)
    parsed = build_parser().parse_args([
        "run-request", *configuration["arguments"], "--", "preview-only-request"])
    settings = load_settings(parsed.settings)
    settings = replace(settings, llm=replace(settings.llm, provider="ollama",
        model=parsed.ollama_model, max_tokens=parsed.ollama_max_output_tokens,
        api_key=None, base_url=parsed.ollama_base_url))
    llm = OllamaLLM(settings, base_url=parsed.ollama_base_url,
        context_window=parsed.ollama_context_window,
        timeout=float(packet["model_limits"]["timeout_seconds"]))
    contract = ({key: packet["next_turn"][key] for key in ("role", "payload", "schema")}
                if packet.get("saved_reassessment_query") else packet["next_turn"].get("query_model_contract"))
    if (not isinstance(contract, dict) or set(contract) != {"role", "payload", "schema"}):
        raise ValueError("INITIAL_FINDINGS_QUERY_CONTRACT_INVALID")
    output_budget = owner._request_model_output_budget(
        contract["role"], parsed.ollama_max_output_tokens)
    frozen = {}
    try:
        llm.prepare_structured_chat(
            requirement_coverage_model.requirement_coverage_messages(contract["payload"]),
            temperature=0.0, max_tokens=output_budget,
            response_format=contract["schema"], provider_request_recorder=frozen.update)
    except OllamaLLMError as error:
        return {"status": "blocked", "role": contract["role"],
            "output_budget": output_budget, "model_calls_performed": 0,
            "http_transport_started": False, "error": str(error),
            "diagnostics": deepcopy(error.diagnostics)}
    visible = {key: deepcopy(value) for key, value in frozen.items()
               if key not in {"body_utf8", "headers"}}
    return {"status": "admitted", "role": contract["role"],
        "output_budget": output_budget, "model_calls_performed": 0,
        "http_transport_started": False, "provider_request": visible}


def preview_initial_findings_execution(*, run_root, run_id, execution_config, allocation_config):
    """Review a separate allocation instance; never modify the original limits."""
    reference, allocation = _file(allocation_config)
    allowed = {"stage_model_call_limits", "max_execution_seconds", "findings_contract_version",
               "dependency_api", "provider_backend", "resume_mode",
               "initial_reassessment_max_cycles", "include_downstream_preflight",
               "request_interpretation_revision", "model_call_budget_policy"}
    from src.mod_agent.model_call_budget import (
        METER_ONLY,
        STAGE_AND_TOTAL,
        TASK_TOTAL_ONLY,
        as_meter_only,
        as_task_total_only,
    )
    policy = allocation.get("model_call_budget_policy", STAGE_AND_TOTAL)
    required = {"max_execution_seconds"} | (set() if policy in {TASK_TOTAL_ONLY, METER_ONLY} else {"stage_model_call_limits"})
    if set(allocation) - allowed or not required.issubset(allocation):
        raise ValueError("INITIAL_FINDINGS_ALLOCATION_SHAPE_INVALID")
    if policy not in {STAGE_AND_TOTAL, TASK_TOTAL_ONLY}:
        raise ValueError("INITIAL_FINDINGS_MODEL_BUDGET_POLICY_INVALID")
    if "findings_contract_version" in allocation and allocation["findings_contract_version"] != requirement_coverage_provider.FINDINGS_CONTRACT_VERSION:
        raise ValueError("INITIAL_FINDINGS_CORRECTION_VERSION_INVALID")
    selected_version = allocation.get("findings_contract_version")
    resume_mode = allocation.get("resume_mode")
    max_cycles = allocation.get("initial_reassessment_max_cycles", 1)
    include_downstream_preflight = allocation.get("include_downstream_preflight", False)
    if max_cycles not in {1, 2}:
        raise ValueError("INITIAL_FINDINGS_REASSESSMENT_CYCLE_LIMIT_INVALID")
    if type(include_downstream_preflight) is not bool:
        raise ValueError("INITIAL_FINDINGS_DOWNSTREAM_PREFLIGHT_INVALID")
    if resume_mode not in (None, "accepted_findings_reassessment",
                           "accepted_findings_reassessment_revision"):
        raise ValueError("INITIAL_FINDINGS_RESUME_MODE_INVALID")
    claim_name = (_reassessment_revision_claim_names(run_root)[1]
                  if resume_mode == "accepted_findings_reassessment_revision"
                  else "initial_findings-reassessment.json"
                  if resume_mode == "accepted_findings_reassessment"
                  else f"initial_findings-v{selected_version}.json"
                  if selected_version == requirement_coverage_provider.FINDINGS_CONTRACT_VERSION
                  else "initial_findings.json")
    existing = read_initial_findings_execution(run_root, claim_name=claim_name)
    if existing is not None:
        return existing
    recovery = (allocation.get("request_interpretation_revision") or {}).get("findings_schema_recovery")
    progress = None
    if allocation.get("request_interpretation_revision"):
        from src.mod_agent.request_interpretation_continuation import read_revision_selection
        progress = read_revision_selection(allocation=allocation, run_root=run_root,
            execution_config=execution_config).get("progress_parent")
    if progress:
        from src.mod_agent.request_interpretation_continuation import (
            prepare_accepted_findings_progress,
        )
        prepared = prepare_accepted_findings_progress(run_root=run_root, run_id=run_id,
            execution_config=execution_config, allocation=allocation)
    elif recovery:
        from src.mod_agent.findings_schema_recovery import prepare_findings_schema_recovery
        prepared = prepare_findings_schema_recovery(run_root=run_root, run_id=run_id,
            execution_config=execution_config, allocation=allocation)
    else:
        prepared = prepare_initial_findings_continuation(
            run_root=run_root, run_id=run_id, execution_config=execution_config,
            correction_contract_version=allocation.get("findings_contract_version"))
    if allocation.get("request_interpretation_revision") is not None and not recovery and not progress:
        from src.mod_agent.request_interpretation_continuation import (
            prepare_interpretation_revision,
        )
        prepared = prepare_interpretation_revision(prepared, run_root=run_root,
            execution_config=execution_config, allocation=allocation)
    elif not recovery and not progress and resume_mode in {"accepted_findings_reassessment",
                       "accepted_findings_reassessment_revision"}:
        prepared = _prepare_accepted_findings_reassessment(
            prepared, run_root=run_root, run_id=run_id, resume_mode=resume_mode)
    maximums = allocation.get("stage_model_call_limits")
    original = prepared["budget"]
    extended_total_allowed = resume_mode == "accepted_findings_reassessment_revision"
    selected_total = (sum(maximums.values())
                      if isinstance(maximums, dict)
                      and all(type(value) is int for value in maximums.values()) else -1)
    if policy not in {TASK_TOTAL_ONLY, METER_ONLY} and (not isinstance(maximums, dict) or set(maximums) != set(original["by_stage"])
            or any(type(value) is not int or value < max(1, original["by_stage"][key]["consumed"])
                   for key, value in maximums.items())
            or (selected_total < original["maximum"] if extended_total_allowed
                else selected_total != original["maximum"])):
        raise ValueError("INITIAL_FINDINGS_ALLOCATION_CHANGES_TOTAL_OR_REFUNDS_CALLS")
    seconds = allocation["max_execution_seconds"]
    if type(seconds) is not int or not 1 <= seconds <= 3600:
        raise ValueError("INITIAL_FINDINGS_TIME_SCOPE_INVALID")
    effective = (as_meter_only(original) if policy == METER_ONLY else
                 as_task_total_only(original) if policy == TASK_TOTAL_ONLY else deepcopy(original))
    if policy not in {TASK_TOTAL_ONLY, METER_ONLY}:
        for key, maximum in maximums.items():
            effective["by_stage"][key].update(maximum=maximum, remaining=maximum - effective["by_stage"][key]["consumed"])
        effective.update(maximum=selected_total, remaining=selected_total - effective["consumed"])
    blockers = [item for item in prepared["blockers"] if item != "PRODUCT_STAGE_MODEL_CALL_BUDGET_EXHAUSTED"]
    if policy == METER_ONLY:
        # Scope: preserve historical use and non-resource failures while changing the selected counting policy.
        # 为什么要有它：计量模式不继承旧实验额度阻断，但来源、执行故障和明确进程截止不能因此被放行。
        blockers = [item for item in blockers if item != "PRODUCT_MODEL_CALL_BUDGET_EXHAUSTED"]
    if policy not in {TASK_TOTAL_ONLY, METER_ONLY} and effective["by_stage"]["requirement_findings"]["remaining"] < 1:
        blockers.append("PRODUCT_STAGE_MODEL_CALL_BUDGET_EXHAUSTED")
    if seconds < prepared["continuation_input"]["model_limits"]["timeout_seconds"]:
        blockers.append("INITIAL_FINDINGS_TIME_WINDOW_CANNOT_FIT_ONE_CALL")
    from src.mod_agent import local_entry as owner
    dependency_api = {"status": "not_configured", "model_calls_performed": 0}
    dependency_api_factory = None
    dependency = allocation.get("dependency_api")
    if dependency is not None:
        if (not isinstance(dependency, dict) or set(dependency) != {"inspector_ref", "max_method_inspections"}
                or not isinstance(dependency["inspector_ref"], dict)
                or set(dependency["inspector_ref"]) != {"path", "sha256"}):
            raise ValueError("INITIAL_FINDINGS_DEPENDENCY_API_CONFIG_INVALID")
        _, manifest = _file(Path(run_root) / "run_manifest.json")
        dependency_api_factory, binding = owner.make_run_dependency_api_reassessment_augmentation(
            run_root=run_root, run_id=run_id,
            game_folder_dependency=manifest.get("build_dependency", {}).get("game_folder"),
            inspector_path=dependency["inspector_ref"]["path"],
            inspector_sha256=dependency["inspector_ref"]["sha256"],
            max_method_inspections=dependency["max_method_inspections"])
        dependency_api = {"status": "configured", **binding}
    contract = prepared["continuation_input"]["next_turn"].get("query_model_contract")
    if isinstance(contract, dict) and contract["payload"].get("query_contract_version") in {
            "requirement_queries_v2", "requirement_queries_v3"}:
        contract["payload"]["tool_capabilities"] = owner._query_tool_capabilities(
            upstream_search_root=contract["payload"]["search_root"], augmentation=dependency_api_factory)
        prepared["continuation_input_sha256"] = _input_digest(prepared["continuation_input"])
    provider_backend = _provider_backend_identity(allocation.get("provider_backend"))
    first_provider_request_preflight = (_preview_first_query_provider_request(
        prepared["continuation_input"], execution_config=execution_config)
        if prepared["continuation_input"].get("saved_reassessment_query") or resume_mode in {
            "accepted_findings_reassessment", "accepted_findings_reassessment_revision"}
        else {"status": "not_applicable", "model_calls_performed": 0,
              "http_transport_started": False})
    model_call_preflight = (owner._preflight_request_model_calls(effective,
        ("requirement_findings",) + (("planning", "initial_patch") if include_downstream_preflight else ()))
        if prepared["continuation_input"].get("saved_reassessment_query") else
        owner.preflight_initial_findings_model_calls(
            effective, include_correction=resume_mode not in {
                "accepted_findings_reassessment", "accepted_findings_reassessment_revision"},
            max_cycles=max_cycles, include_downstream=include_downstream_preflight))
    if prepared["continuation_input"].get("request_interpretation_revision"):
        model_call_preflight = owner._preflight_request_model_calls(effective,
            (() if progress else ("requirement_findings",) if recovery else ("requirement_projection", "requirement_findings")) + ("initial_findings_queries",
             "requirement_findings") + (("planning", "initial_patch") if include_downstream_preflight else ()))
    if (prepared["continuation_input"]["correction_contract"]["selected_version"]
            == requirement_coverage_provider.FINDINGS_CONTRACT_VERSION):
        blockers.extend(row["code"] for row in model_call_preflight["blockers"])
        if provider_backend["status"] != "configured":
            blockers.append("INITIAL_FINDINGS_PROVIDER_BACKEND_REQUIRED")
        if first_provider_request_preflight["status"] == "blocked":
            blockers.append("INITIAL_FINDINGS_PROVIDER_CONTEXT_BUDGET_EXCEEDED")
    scope = {"original_input_sha256": prepared["continuation_input_sha256"], "allocation_ref": reference,
             "effective_budget": effective, "max_execution_seconds": seconds,
             "findings_contract": {key: value for key, value in prepared["continuation_input"]["correction_contract"].items()
                                   if key != "historical_turn"},
             "model_call_preflight": model_call_preflight,
             "first_provider_request_preflight": first_provider_request_preflight,
             "dependency_api": dependency_api,
             "provider_backend": provider_backend,
             "python_runtime_identity": deepcopy(
                 prepared["continuation_input"]["python_runtime_identity"]),
             "initial_reassessment_max_cycles": None if policy == TASK_TOTAL_ONLY else max_cycles,
             "include_downstream_preflight": include_downstream_preflight,
             "resume_mode": resume_mode,
             "completion_boundary": "original_exact_diff_approval", "run_id": run_id}
    return {**prepared, "effective_budget": effective, "execution_scope": scope,
            "scope_hash": _digest(scope), "blockers": blockers,
            "status": "prepared_blocked" if blockers else "awaiting_continuation_approval"}


def _write_record(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _reassessment_revision_claim_names(run_root):
    """Append reviewed revision instances without creating another resume mode."""
    continuation_root = Path(run_root).resolve() / "continuations"
    fourth_revision = continuation_root / "initial_findings-reassessment-v5.json"
    third_revision = continuation_root / "initial_findings-reassessment-v4.json"
    second_revision = continuation_root / "initial_findings-reassessment-v3.json"
    first_revision = continuation_root / "initial_findings-reassessment-v2.json"
    if fourth_revision.is_file():
        return ("initial_findings-reassessment-v5.json",
                "initial_findings-reassessment-v6.json", 4)
    if third_revision.is_file():
        return ("initial_findings-reassessment-v4.json",
                "initial_findings-reassessment-v5.json", 3)
    if second_revision.is_file():
        return ("initial_findings-reassessment-v3.json",
                "initial_findings-reassessment-v4.json", 3)
    if first_revision.is_file():
        return ("initial_findings-reassessment-v2.json",
                "initial_findings-reassessment-v3.json", 2)
    return ("initial_findings-reassessment.json",
            "initial_findings-reassessment-v2.json", 1)


def read_initial_findings_execution(run_root, *, claim_name=None):
    """Read one claimed continuation without retrying any uncertain effects."""
    root = Path(run_root).resolve()
    continuation_root = root / "continuations"
    if claim_name is None:
        reassessment_revision_5 = continuation_root / "initial_findings-reassessment-v6.json"
        reassessment_revision_4 = continuation_root / "initial_findings-reassessment-v5.json"
        reassessment_revision_3 = continuation_root / "initial_findings-reassessment-v4.json"
        reassessment_revision_2 = continuation_root / "initial_findings-reassessment-v3.json"
        reassessment_revision = continuation_root / "initial_findings-reassessment-v2.json"
        reassessment = continuation_root / "initial_findings-reassessment.json"
        revised = continuation_root / f"initial_findings-v{requirement_coverage_provider.FINDINGS_CONTRACT_VERSION}.json"
        claim = (reassessment_revision_5 if reassessment_revision_5.is_file() else reassessment_revision_4
                 if reassessment_revision_4.is_file() else reassessment_revision_3
                 if reassessment_revision_3.is_file() else reassessment_revision_2
                 if reassessment_revision_2.is_file() else reassessment_revision
                 if reassessment_revision.is_file() else reassessment
                 if reassessment.is_file() else revised
                 if revised.is_file() else continuation_root / "initial_findings.json")
    else:
        if claim_name not in {"initial_findings.json",
                f"initial_findings-v{requirement_coverage_provider.FINDINGS_CONTRACT_VERSION}.json",
                "initial_findings-reassessment.json", "initial_findings-reassessment-v2.json",
                "initial_findings-reassessment-v3.json", "initial_findings-reassessment-v4.json",
                "initial_findings-reassessment-v5.json", "initial_findings-reassessment-v6.json"}:
            raise ValueError("INITIAL_FINDINGS_CLAIM_NAME_INVALID")
        claim = continuation_root / claim_name
    if not claim.is_file():
        if claim_name in (None, "initial_findings.json") and not any(
                continuation_root.glob("initial_findings*.json")) and any(
                continuation_root.glob("*/run_manifest.json")):
            raise ValueError("INITIAL_FINDINGS_PRIOR_EXECUTION_CLAIM_MISSING")
        return None
    _, receipt = _file(claim)
    _, parent = _bound_json(receipt["parent_result_ref"], root)
    if parent.get("run_id") != receipt["run_id"]:
        raise ValueError("INITIAL_FINDINGS_CONTINUATION_PARENT_CHANGED")
    recorded_budget = _budget(receipt["model_call_budget"])
    parent_budget = _budget(parent["model_call_budget"])
    from src.mod_agent.model_call_budget import is_meter_only
    maximum_reduced = (not is_meter_only(recorded_budget) and (
        is_meter_only(parent_budget) or recorded_budget["maximum"] < parent_budget["maximum"]))
    if (maximum_reduced
            or recorded_budget["consumed"] < parent_budget["consumed"]
            or set(recorded_budget["by_stage"]) != set(parent_budget["by_stage"])
            or any(recorded_budget["by_stage"][key]["consumed"]
                   < parent_budget["by_stage"][key]["consumed"]
                   for key in recorded_budget["by_stage"])):
        raise ValueError("INITIAL_FINDINGS_CONTINUATION_BUDGET_CHANGED")
    result_path = Path(receipt["result_path"]).resolve()
    if not result_path.is_relative_to(root / "continuations"):
        raise ValueError("INITIAL_FINDINGS_CONTINUATION_RESULT_OUTSIDE_RUN")
    if result_path.is_file():
        result_ref, result = _file(result_path)
        if result.get("run_id") != receipt["run_id"] or result.get("scope_hash") != receipt["scope_hash"]:
            raise ValueError("INITIAL_FINDINGS_CONTINUATION_RESULT_IDENTITY_INVALID")
        if result.get("model_call_budget") != recorded_budget or result.get("model_call_count") != recorded_budget["consumed"]:
            raise ValueError("INITIAL_FINDINGS_CONTINUATION_BUDGET_CHANGED")
        if result.get("status") == "awaiting_approval":
            checkpoint = ArtifactStore(root / "a").semantic_read_immutable(ArtifactRef(**result["workflow_checkpoint"]))["data"]
            if checkpoint["run_id"] != receipt["run_id"] or checkpoint["request_id"] != receipt["request_id"]:
                raise ValueError("INITIAL_FINDINGS_CONTINUATION_CHECKPOINT_CHANGED")
        return {**result, "read_only": True, "result_ref": result_ref, "model_calls_performed": 0}
    return {**receipt, "status": "continuation_started_inspect_before_retry", "read_only": True,
            "model_calls_performed": 0, "effects": "unknown_after_claim"}


def execute_initial_findings_continuation(*, run_root, run_id, execution_config, allocation_config, decision):
    """Assemble original ProductExecution adapters and invoke its original Graph once."""
    from src.core.settings import load_settings
    from src.core.trace.execution_trace import trace_operation
    from src.libs.llm.ollama_llm import OllamaLLM
    from src.mod_agent import local_entry as owner
    from src.mod_agent.evidence import build_requirement_coverage_pack
    from src.mod_agent.patch_execution_trace import product_execution_trace, product_trace_identity
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from src.mod_agent.planning.model_selection import ModelPlanningTargetSelector
    from src.mod_agent.product_execution import (
        ProductExecutionInputBindingV1,
        ProductExecutionRequestV1,
    )
    from src.mod_agent.product_executor import ProductApprovalDeferred, ProductExecutionAdaptersV1
    from src.mod_agent.request_patch_execution import RequestPatchExecution
    from src.mod_agent.upstream_model_trace import trace_upstream_model_call
    from src.mod_agent.upstream_trace import update_observation

    root = Path(run_root).resolve(strict=True)
    _, selected_allocation = _file(allocation_config)
    selected_version = selected_allocation.get("findings_contract_version")
    resume_mode = selected_allocation.get("resume_mode")
    claim_name = (_reassessment_revision_claim_names(root)[1]
                  if resume_mode == "accepted_findings_reassessment_revision"
                  else "initial_findings-reassessment.json"
                  if resume_mode == "accepted_findings_reassessment"
                  else f"initial_findings-v{selected_version}.json"
                  if selected_version == requirement_coverage_provider.FINDINGS_CONTRACT_VERSION
                  else "initial_findings.json")
    prior = read_initial_findings_execution(root, claim_name=claim_name)
    if prior is not None:
        if not isinstance(decision, dict) or decision.get("scope_hash") != prior["scope_hash"]:
            raise ValueError("INITIAL_FINDINGS_CONTINUATION_ALREADY_CLAIMED")
        return prior
    reviewed = preview_initial_findings_execution(run_root=root, run_id=run_id,
        execution_config=execution_config, allocation_config=allocation_config)
    if (not isinstance(decision, dict) or set(decision) != {"decision", "reviewer", "scope_hash"}
            or decision["decision"] != "approve" or not isinstance(decision["reviewer"], str)
            or not decision["reviewer"].strip() or decision["scope_hash"] != reviewed["scope_hash"]):
        raise ValueError("INITIAL_FINDINGS_EXACT_CONTINUATION_APPROVAL_REQUIRED")
    if "continuation_input" not in reviewed:
        return reviewed
    if reviewed["blockers"]:
        raise ValueError(reviewed["blockers"][0])
    packet = reviewed["continuation_input"]
    revision = packet.get("request_interpretation_revision")
    schema_recovery = packet.get("findings_schema_recovery")
    accepted_progress = packet.get("accepted_findings_progress")
    if _python_runtime_identity() != packet.get("python_runtime_identity"):
        raise ValueError("INITIAL_FINDINGS_PYTHON_RUNTIME_CHANGED")
    _, manifest = _file(root / "run_manifest.json")
    _, configuration = _file(execution_config)
    _, allocation = _file(allocation_config)
    from src.mod_agent.cli import build_parser
    parsed = build_parser().parse_args(["run-request", *configuration["arguments"], "--", manifest["request"]])
    if manifest.get("verification_tool_build") is not None or manifest.get("objective_verification_command") is not None:
        raise ValueError("INITIAL_FINDINGS_OPTIONAL_EXECUTION_ADAPTER_NOT_SUPPORTED")
    baseline = manifest["baseline_build"]
    baseline_observation = validated_baseline_repair_observation(manifest, root)
    if baseline_observation is None and (baseline["restore_exit_code"] != 0 or baseline["build_exit_code"] != 0
            or hashlib.sha256(Path(baseline["artifact_path"]).read_bytes()).hexdigest() != baseline["artifact_sha256"]):
        raise ValueError("INITIAL_FINDINGS_BASELINE_CHANGED")
    settings = load_settings(parsed.settings)
    settings = replace(settings, llm=replace(settings.llm, provider="ollama", model=parsed.ollama_model,
        max_tokens=parsed.ollama_max_output_tokens, api_key=None, base_url=parsed.ollama_base_url))
    timeout = float(packet["model_limits"]["timeout_seconds"])
    seconds = reviewed["execution_scope"]["max_execution_seconds"]
    if seconds < timeout:
        raise ValueError("INITIAL_FINDINGS_TIME_WINDOW_CANNOT_FIT_ONE_CALL")
    if _provider_backend_identity(allocation.get("provider_backend")) != reviewed["execution_scope"]["provider_backend"]:
        raise ValueError("INITIAL_FINDINGS_PROVIDER_BACKEND_CHANGED")
    dependency_api_factory = None
    if allocation.get("dependency_api") is not None:
        dependency = allocation["dependency_api"]
        dependency_api_factory, observed_dependency_api = owner.make_run_dependency_api_reassessment_augmentation(
            run_root=root, run_id=run_id,
            game_folder_dependency=manifest.get("build_dependency", {}).get("game_folder"),
            inspector_path=dependency["inspector_ref"]["path"],
            inspector_sha256=dependency["inspector_ref"]["sha256"],
            max_method_inspections=dependency["max_method_inspections"])
        if {"status": "configured", **observed_dependency_api} != reviewed["execution_scope"]["dependency_api"]:
            raise ValueError("INITIAL_FINDINGS_DEPENDENCY_API_BINDING_CHANGED")
    segment = root / "continuations" / reviewed["scope_hash"][:24]
    claim_path = root / "continuations" / claim_name
    claim_path.parent.mkdir(exist_ok=True)
    budget = deepcopy(reviewed["effective_budget"])
    receipt = {"entry": "initial-findings-continuation", "run_id": run_id, "request_id": packet["request_id"],
               "scope_hash": reviewed["scope_hash"], "parent_result_ref": packet["execution_parent_result_ref"],
               "result_path": str(segment / "run_result.json"), "decision": decision,
               "model_call_budget": budget, "status": "claimed"}
    if revision:
        receipt["request_interpretation_revision"] = {
            "selection_sha256": revision["selection_sha256"],
            "allocation_ref": deepcopy(reviewed["execution_scope"]["allocation_ref"])}
    with claim_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False)
    segment.mkdir(exist_ok=False)
    started = time.monotonic()
    segment_manifest = {"run_id": run_id, "request_id": packet["request_id"],
        "evidence_optimization": run_optimization_policy(root),
        "source_view_policy": source_view_policy_from_manifest(manifest),
        "parent_result_ref": packet["execution_parent_result_ref"],
        "started_at": datetime.now(timezone.utc).isoformat(), "model_call_budget": budget,
        "initial_findings_cause_chain": deepcopy(reviewed["cause_chain"]),
        "findings_contract": deepcopy(reviewed["execution_scope"]["findings_contract"]),
        "provider_backend": deepcopy(reviewed["execution_scope"]["provider_backend"]),
        "original_projection_reused": not bool(revision) or bool(schema_recovery) or bool(accepted_progress),
        "original_baseline_reused": True, "baseline_builds_performed": 0}
    _write_record(segment / "run_manifest.json", segment_manifest)
    def reserve(role):
        nonlocal budget
        if seconds - (time.monotonic() - started) < timeout:
            error = owner.LocalEntryError("PRODUCT_CONTINUATION_TIME_BUDGET_EXHAUSTED")
            error.model_call_boundary = {"budget_stage": role, "budget": deepcopy(budget),
                "reservation_succeeded": False, "model_request_invoked": False, "model_transport_started": False}
            raise error
        budget = owner._reserve_request_model_call(budget, role)
        receipt["model_call_budget"] = deepcopy(budget)
        _write_record(claim_path, receipt)  # Persist the reservation before provider invocation.
        _, observed_manifest = _file(segment / "run_manifest.json")
        observed_manifest.update(model_call_budget=deepcopy(budget), model_call_count=budget["consumed"],
            prior_model_call_count=packet["budget"]["consumed"],
            current_invocation_model_call_count=budget["consumed"] - packet["budget"]["consumed"],
            model_call_count_scope="cumulative_same_task")
        _write_record(segment / "run_manifest.json", observed_manifest)
    def identity(base_url, model):
        observed = owner._read_ollama_model_identity(base_url, model)
        owner._validate_ollama_model_identity(observed, expected_base_url=base_url, expected_model=model)
        owner._assert_ollama_model_identity_stable(expected=packet["model_identity"], observed=observed)
        owner._validate_ollama_context_window(observed, requested_context_window=parsed.ollama_context_window)
        return observed
    result = None
    projection_calls = 0
    graph_outputs = []
    graph_invocation_attempted = False
    try:
        identity(parsed.ollama_base_url, parsed.ollama_model)
        llm = OllamaLLM(settings, base_url=parsed.ollama_base_url, context_window=parsed.ollama_context_window, timeout=timeout)
        profile = owner.LocalOllamaInitialPatchProfileV1(profile_version=owner.LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
            base_url=parsed.ollama_base_url, model=parsed.ollama_model, context_window=parsed.ollama_context_window,
            output_budget=parsed.ollama_max_output_tokens, timeout_seconds=timeout)
        store = ArtifactStore(root / "a")
        feature = (deepcopy(revision["feature_data"]) if revision else
            store.semantic_read_immutable(ArtifactRef(**packet["feature_spec_ref"]))["data"])
        source = Path(manifest["target_source"])
        target = root / "generated_mods" / source.name
        native_findings_mode = packet.get("native_findings_contract")
        from src.mod_agent.saved_reassessment import restored_source_observations
        retrieved = (None if revision and not schema_recovery else restored_source_observations(packet)
            if packet.get("saved_reassessment_query") or packet.get("saved_reassessment_completion") or schema_recovery
            else _replay_source_query_set(root=root, materialized=packet["materialized_projection"],
                request=feature["user_request"], search_root=target.relative_to(root).as_posix(),
                historical_payload=packet["retrieved_evidence"]["budget_summary"]))
        if not revision and _input_digest({"retrieved_evidence": retrieved.to_artifact_payload()}) != _input_digest({"retrieved_evidence": packet["retrieved_evidence"]}):
            raise ValueError("INITIAL_FINDINGS_CURRENT_EVIDENCE_CHANGED")
        def complete(role, payload, schema):
            nonlocal projection_calls
            payload = _with_baseline_compilation_observation(payload, baseline_observation)
            identity(profile.base_url, profile.model)
            owner._assert_no_absolute_local_paths(payload)
            stage = owner._request_model_budget_stage(role)
            output_budget = owner._request_model_output_budget(role, profile.output_budget)
            messages = requirement_coverage_model.requirement_coverage_messages(payload)
            frozen_request = {}
            llm.prepare_structured_chat(messages, temperature=0.0,
                max_tokens=output_budget, response_format=schema,
                provider_request_recorder=frozen_request.update)
            reserve(stage)
            if stage == "requirement_projection":
                projection_calls += 1
            def invoke_with_provider_request(trace_recorder):
                trace_recorder(deepcopy(frozen_request))
                return llm.resume_structured_chat(
                    frozen_request, expected_output_budget=output_budget)
            response = trace_upstream_model_call(role=role, provider=type(llm).__name__,
                model=getattr(llm, "model", None), input_payload=payload,
                request_document={"messages": [{"role": item.role, "content": item.content}
                    for item in messages], "response_format": dict(schema), "temperature": 0.0,
                    "max_tokens": output_budget, "stream_response": False},
                body_root=segment, invoke_with_provider_request=invoke_with_provider_request)
            identity(profile.base_url, profile.model)
            return requirement_coverage_model.parse_requirement_coverage_json_object(response.content)
        def collect_evidence(state, incoming_feature):
            if incoming_feature != feature or state["run_id"] != run_id or state["request_id"] != packet["request_id"]:
                raise ValueError("INITIAL_FINDINGS_GRAPH_FEATURE_CHANGED")
            if accepted_progress:
                return EvidencePack.from_artifact_payload(store.semantic_read_immutable(
                    ArtifactRef(**accepted_progress["evidence_pack_ref"]))["data"])
            if schema_recovery:
                with trace_operation("requirements.coverage", attributes={
                        "saved_projection_reused": True, "findings_schema_recovery": schema_recovery}) as span:
                    semantic = StructuredModelRequirementCoverageProvider(complete, compress_findings_input=run_optimization_policy(root)["compress_findings_input"],
                        native_findings_contract=native_findings_mode)
                    findings = semantic.findings(materialized_projection=packet["materialized_projection"],
                        retrieved_pack=retrieved, request=feature["user_request"])
                    pack = build_requirement_coverage_pack(query=feature["user_request"],
                        materialized_projection=packet["materialized_projection"], retrieved_pack=retrieved, findings=findings)
                    audit = deepcopy(list(semantic.findings_attempt_audit))
                    update_observation(span, requirement_findings_attempt_audit=audit)
                    return replace(pack, budget_summary={**pack.budget_summary,
                        "requirement_findings_attempt_audit": audit, "saved_projection_recovery": deepcopy(schema_recovery)})
            if revision:
                fresh = requirement_coverage_provider.make_requirement_coverage_evidence_provider(
                    RgLiveEvidenceProvider(root, small_file_max_bytes=run_small_file_max_bytes(root),
                reuse_query_observations=run_optimization_policy(root)["reuse_query_observations"]), complete_json=complete,
                    compress_findings_input=run_optimization_policy(root)["compress_findings_input"],
                    native_findings_contract=native_findings_mode)
                return fresh.collect(query=feature["user_request"], feature_data=feature,
                    search_root=target.relative_to(root).as_posix())
            if packet.get("resume_mode") in {
                    "accepted_findings_reassessment",
                    "accepted_findings_reassessment_revision"}:
                evidence_ref = ArtifactRef(**packet["next_turn"]["evidence_pack_ref"])
                payload = store.semantic_read_immutable(evidence_ref)["data"]
                if packet.get("saved_reassessment_completion"):
                    from src.mod_agent.saved_reassessment import with_completed_findings_receipt
                    payload = with_completed_findings_receipt(payload, packet["saved_reassessment_completion"])
                pack = EvidencePack.from_artifact_payload(payload)
                if packet.get("resume_mode") == "accepted_findings_reassessment_revision":
                    from src.mod_agent.evidence.reassessment import (
                        prepare_initial_findings_reassessment_revision,
                    )
                    projected = (pack.to_artifact_payload()
                        if packet["next_turn"].get("reuse_projected_reassessment_input") is True
                        else prepare_initial_findings_reassessment_revision(
                            pack.to_artifact_payload(), feature,
                            revision_number=packet["next_turn"]["reassessment_revision_number"]))
                    if (_digest(projected) != packet["next_turn"]["reassessment_input_sha256"]
                            or (packet["next_turn"].get("reuse_projected_reassessment_input") is not True
                                and pack.budget_summary.get("initial_findings_reassessment")
                                != packet["next_turn"]["prior_reassessment_lineage"])):
                        raise ValueError("INITIAL_FINDINGS_PRIOR_REASSESSMENT_CHANGED")
                    pack = EvidencePack.from_artifact_payload(projected)
                return pack
            with trace_operation("requirements.coverage", attributes={"initial_findings_continuation": True,
                    "original_projection_reused": True, "prior_response_ref": packet["parent_refs"][-1]}) as span:
                try:
                    raw = complete(packet["next_turn"]["role"], packet["next_turn"]["payload"], packet["next_turn"]["schema"])
                    materialized = packet["materialized_projection"]
                    _, indexes, all_indexes, deferred = requirement_coverage_provider._requirement_findings_model_relations(materialized)
                    findings = requirement_coverage_provider._materialize_requirement_findings_response(raw,
                        schema=packet["next_turn"]["schema"], relation_indexes=indexes,
                        coverage_schema_version=materialized["schema_version"],
                        native_findings_contract=native_findings_mode)
                    findings = requirement_coverage_provider._merge_deferred_requirement_findings(
                        findings, all_relation_indexes=all_indexes, deferred_findings=deferred)
                    pack = build_requirement_coverage_pack(query=feature["user_request"], materialized_projection=materialized,
                        retrieved_pack=retrieved, findings=findings)
                    audit = [{"attempt_index": 2, "status": "accepted", "error_code": None,
                              "origin": "current_continuation", "prior_response_ref": packet["parent_refs"][-1]}]
                    if native_findings_mode is not None:
                        from src.mod_agent.evidence.native_findings_contract import provenance
                        audit[0]["native_findings_provenance"] = provenance(raw, findings)
                    update_observation(span, requirement_findings_attempt_audit=audit)
                    return replace(pack, budget_summary={**pack.budget_summary, "requirement_findings_attempt_audit": audit})
                except BaseException as error:
                    update_observation(span, requirement_findings_attempt_audit=[{"attempt_index": 2,
                        "status": "failed", "error_code": str(error),
                        **({"diagnostic": error.diagnostic} if hasattr(error, "diagnostic") else {}),
                        **({"call_boundary": error.model_call_boundary} if hasattr(error, "model_call_boundary") else {})}])
                    raise
        def evidence_factory(state, incoming_feature):
            return _with_baseline_compilation_evidence(
                collect_evidence(state, incoming_feature), baseline_observation)

        worker_config = PatchWorkerConfiguration.load(parsed.patch_worker_config) if parsed.patch_worker_config else None
        policy = worker_config.policy if worker_config else owner.PatchAgentLoopPolicy(
            max_total_model_turns=owner._INITIAL_PATCH_GENERATION_CALL_BUDGET, require_source_inspection=False,
            finalize_after_first_observation=False, enable_supporting_reads=True, allow_upstream_request=True,
            allow_batched_reads=True, deduplicate_source_excerpts=True, reuse_covered_source_observations=True)
        worker = RequestPatchExecution(artifact_store=store, workspace_root=root, profile=profile,
            identity_reader=identity, policy=policy, reserve_model_call=reserve, budget_snapshot=lambda: deepcopy(budget),
            behavior_evaluator=worker_config.behavior_evaluator() if worker_config else None,
            worker_configuration_ref=worker_config.reference if worker_config else None)
        class PlanningClient:
            def __init__(self, stage="planning", role="planning"):
                self.client = owner.LocalOllamaInitialPatchAdapter._default_agent_turn_client(profile)
                self.provider, self.model = self.client.provider, self.client.model
                self.stage, self.role = stage, role
            def complete_turn(self, request):
                identity(profile.base_url, profile.model)
                reserve(self.stage)
                response = trace_upstream_model_call(role=self.role, provider=self.provider, model=self.model,
                    input_payload=request.messages[1].content, request_document=asdict(request), body_root=segment,
                    invoke=lambda: self.client.complete_turn(request))
                identity(profile.base_url, profile.model)
                return response
        selector = ModelPlanningTargetSelector.from_settings(settings=settings, client=PlanningClient(), workspace_root=root,
            behavior_evaluator_configured=(worker_config is not None and worker_config.behavior_evaluator() is not None),
            receipt_path=segment / "planning_model_selection.json", model_identity=packet["model_identity"])
        def trace_cycle_observer(pack, cycle):
            provisional = {"entry": "initial-findings-cycle-checkpoint", "run_id": run_id,
                "request_id": packet["request_id"], "status": "cycle_checkpoint",
                "scope_hash": reviewed["scope_hash"], "model_call_count": budget["consumed"],
                "model_call_budget": deepcopy(budget),
                "prior_model_call_count": packet["budget"]["consumed"],
                "current_invocation_model_call_count": budget["consumed"] - packet["budget"]["consumed"],
                "blockers": [], "patch_execution_receipts": [],
                "cycle_evidence": pack.to_artifact_payload()}
            if revision:
                provisional["request_interpretation_revision"] = deepcopy(receipt["request_interpretation_revision"])
            _write_record(segment / "run_result.json", provisional)
            output = segment / f"trace_readback_cycle_{cycle}.json"
            project_root = Path(__file__).resolve().parents[2]
            completed = subprocess.run([sys.executable, "-B", str(project_root / "scripts" / "product_trace.py"),
                "--allowed-root", str(root.parents[1]), "--run-root", str(root), "--run-id", run_id,
                "--continuation-root", str(segment), "--include-local-raw", "--output", str(output)],
                cwd=project_root, capture_output=True, text=True, encoding="utf-8", errors="strict",
                timeout=120, check=False)
            if completed.returncode != 0 or not output.is_file():
                raise ValueError("INITIAL_FINDINGS_TRACE_READBACK_FAILED")
            readback = json.loads(output.read_text(encoding="utf-8"))
            latest = readback["segments"][-1]
            query_calls = [row for row in latest["calls"] if row.get("role") == "initial_findings_queries"]
            findings_calls = [row for row in latest["calls"] if row.get("role") == "requirement_findings"]
            if len(query_calls) < cycle or len(findings_calls) < cycle:
                raise ValueError("INITIAL_FINDINGS_TRACE_CALL_COUNT_MISMATCH")
            query_call = query_calls[cycle - 1]
            following_findings = ([row for row in latest["calls"][latest["calls"].index(query_call) + 1:]
                if row.get("role") == "requirement_findings"] if revision else findings_calls[cycle - 1:])
            if not following_findings:
                raise ValueError("INITIAL_FINDINGS_TRACE_CALL_COUNT_MISMATCH")
            for call in (query_call, following_findings[0]):
                transport = call.get("transport") or {}
                local = call.get("local_evidence") or {}
                if (transport.get("transport_started") is not True
                        or transport.get("response_received") is not True
                        or transport.get("response_complete") is not True
                        or transport.get("http_status") != 200
                        or (local.get("request") or {}).get("integrity") != "verified_bytes"
                        or (local.get("provider_request") or {}).get("integrity") != "verified_bytes"):
                    raise ValueError("INITIAL_FINDINGS_TRACE_TRANSPORT_MISMATCH")
            reassessment_view = latest.get("initial_findings_reassessment") or {}
            if reassessment_view.get("status") != "recorded":
                raise ValueError("INITIAL_FINDINGS_TRACE_REASSESSMENT_MISSING")
            groups = reassessment_view.get("query_groups") or []
            if not groups or any(not row.get("gap_ids") or not row.get("current_fact_needed")
                                 or not row.get("modification_impact")
                                 or row.get("authorized_source") not in {
                                     "target_source", "dependency_api"} for row in groups):
                raise ValueError("INITIAL_FINDINGS_TRACE_QUERY_BINDING_MISSING")
            metadata = reassessment_view.get("metadata_supplement") or {}
            if metadata.get("status") not in {
                    "not_configured", "not_requested", "not_authorized", "unavailable",
                    "observed", "observed_metadata_only"}:
                raise ValueError("INITIAL_FINDINGS_TRACE_METADATA_STATUS_INVALID")
            evidence = reassessment_view.get("evidence_revision") or {}
            if (evidence.get("request_integrity") != "verified_bytes"
                    or evidence.get("response_integrity") != "verified_bytes"):
                raise ValueError("INITIAL_FINDINGS_TRACE_EVIDENCE_BINDING_MISSING")
            if cycle > 1:
                raw = query_calls[cycle - 1].get("raw", {}).get("request", {})
                messages = raw.get("messages") if isinstance(raw, dict) else None
                content = next((row.get("content") for row in messages or []
                    if isinstance(row, dict) and row.get("role") == "user"), None)
                payload = json.loads(content) if isinstance(content, str) else None
                if (not isinstance(payload, dict) or "prior_query_outcomes" not in payload
                        or "prior_dependency_api_observation" not in payload):
                    raise ValueError("INITIAL_FINDINGS_TRACE_PRIOR_FEEDBACK_MISSING")
            readback_ref = {"cycle": cycle, "path": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "query_call_count": len(query_calls), "findings_call_count": len(findings_calls),
                "metadata_status": metadata.get("status"),
                "recorded_new_evidence_ids": evidence.get("recorded_new_evidence_ids")}
            _, current_manifest = _file(segment / "run_manifest.json")
            history = list(current_manifest.get("trace_cycle_readbacks", []))
            history.append(readback_ref)
            current_manifest["trace_cycle_readbacks"] = history
            stop = pack.budget_summary.get("reassessment_stop")
            if isinstance(stop, dict):
                current_manifest["initial_findings_reassessment_stop"] = deepcopy(stop)
            _write_record(segment / "run_manifest.json", current_manifest)
        upstream = owner._make_request_upstream_evidence_factory(artifact_store=store, workspace_root=root,
            compress_findings_input=run_optimization_policy(root)["compress_findings_input"],
            source_provider=RgLiveEvidenceProvider(root, small_file_max_bytes=run_small_file_max_bytes(root),
                reuse_query_observations=run_optimization_policy(root)["reuse_query_observations"]),
            upstream_search_root=target.relative_to(root).as_posix(),
            complete_requirement_coverage_json=complete, budget_snapshot=lambda: deepcopy(budget),
            source_augmentation_factory=dependency_api_factory,
            native_findings_contract=native_findings_mode,
            max_initial_reassessment_cycles=allocation.get("initial_reassessment_max_cycles", 1),
            cycle_observer=lambda pack, cycle: trace_cycle_observer(pack, cycle))
        upstream_selector = ModelPlanningTargetSelector.from_settings(settings=settings,
            behavior_evaluator_configured=(worker_config is not None and worker_config.behavior_evaluator() is not None),
            client=PlanningClient("repair", "patch_upstream_planning"), workspace_root=root,
            receipt_path=segment / "planning_upstream_model_selection.json", model_identity=packet["model_identity"])
        def approval_boundary(*_):
            raise owner.LocalEntryError("Continue through the original exact-diff approval/Build owners.")
        request = ProductExecutionRequestV1(contract_version="1", run_id=run_id, request_id=packet["request_id"],
            user_request=manifest["request"], target_domain="gtfo", target_mod=source.name,
            requested_completion_scope="compile_only", repair_budget=2, isolated_workspace_root=str(root),
            isolated_target_root=str(target), source_root=str(source), artifact_store_root=str(store.root),
            source_snapshot_sha256=manifest["source_snapshot_sha256"], isolated_source_snapshot_sha256=source_snapshot_sha256(source_snapshot(target)),
            input_bindings=tuple(ProductExecutionInputBindingV1.create(key, manifest[key] or None) for key in
                ("source_revision_binding", "source_disclosure", "model_admission", "runtime_admission")),
            completion_classification=(), confirmed_isolated_write=True, allow_build=True)
        adapters = ProductExecutionAdaptersV1(evidence_pack_factory=evidence_factory, patch_change_factory=worker,
            planning_target_selection_factory=selector, product_patch_upstream_execution=worker,
            patch_upstream_evidence_pack_factory=upstream, patch_upstream_planning_target_selection_factory=upstream_selector,
            initial_findings_reassessment_factory=(None if packet.get("saved_reassessment_query") else upstream.reassess_initial),
            dry_run_factory=owner.make_exact_no_write_dry_run_factory(root), isolated_apply_factory=approval_boundary,
            build_result_factory=approval_boundary, runtime_evidence_factory=None, compiled_artifact_verifier=None, source_change_verifier=None)
        with product_execution_trace(run_root=segment, run_id=run_id, request_id=packet["request_id"], artifact_store_root=store.root) as trace:
            preflight = owner.preflight_product_execution(request, adapters, store)
            try:
                graph_invocation_attempted = True
                product = owner.execute_product_execution(preflight, adapters, store, defer_approval=True)
                graph_outputs = product.outputs
                worker.complete_product(product)
                result = {"status": product.execution_disposition, "product_execution_result": product.to_dict(),
                          "artifacts": {item.artifact_role: item.artifact_ref.path for item in product.outputs},
                          "current_stage": product.terminal_node, "blockers": [item.to_dict() for item in product.blockers]}
            except ProductApprovalDeferred as pending:
                graph_outputs = pending.outputs
                worker.complete_deferred(pending)
                result = {**pending.to_dict(), "current_stage": "awaiting_approval",
                          "artifacts": {item.artifact_role: item.artifact_ref.path for item in pending.outputs}}
            result.update(execution_trace=product_trace_identity(trace, segment), graph_invoked=True,
                          patch_execution_receipts=[journal.projection() for journal in worker.journals])
    except Exception as error:
        result = {"status": "blocked", "reason": str(error), "failure_class": type(error).__name__,
                  "effects": "inspect_current_continuation_records", "graph_invoked": None if graph_invocation_attempted else False}
    try:
        canonical_unchanged = source_snapshot(Path(manifest["target_source"])) == manifest["source_snapshot"]
        generated_unchanged = source_snapshot(root / "generated_mods" / Path(manifest["target_source"]).name) == manifest["source_snapshot"]
    except (OSError, ValueError) as error:
        canonical_unchanged = generated_unchanged = None
        result.update(owner_status=result["status"], status="blocked", post_validation_error=str(error))
    result.update(entry="initial-findings-continuation", run_id=run_id, request_id=packet["request_id"],
        evidence_optimization=run_optimization_policy(root),
        feature_id=packet["feature_spec_ref"]["feature_id"], run_root=str(root), continuation_root=str(segment),
        scope_hash=reviewed["scope_hash"], model_call_count=budget["consumed"], model_call_budget=budget,
        prior_model_call_count=packet["budget"]["consumed"], current_invocation_model_call_count=budget["consumed"]-packet["budget"]["consumed"],
        baseline_builds_performed=0, projection_model_calls_performed=projection_calls,
        actual_write_performed=False if generated_unchanged else None,
        canonical_source_unchanged=canonical_unchanged,
        read_only=False, history_rewritten=False)
    if revision:
        result["request_interpretation_revision"] = {
            key: deepcopy(value) for key, value in revision.items()
            if key not in {"feature_data", "completion_receipt"}}
        result["request_interpretation_revision"].update(
            allocation_ref=deepcopy(reviewed["execution_scope"]["allocation_ref"]),
            graph_artifact_refs={row.artifact_role: asdict(row.artifact_ref)
                for row in graph_outputs
                if row.artifact_role in {"request_planning_report", "mod_planning_brief", "feature_spec"}})
    _, stop_manifest = _file(segment / "run_manifest.json")
    if isinstance(stop_manifest.get("initial_findings_reassessment_stop"), dict):
        result["initial_findings_reassessment_stop"] = deepcopy(
            stop_manifest["initial_findings_reassessment_stop"])
    result["cause_chain"] = deepcopy(reviewed["cause_chain"])
    result["findings_contract"] = deepcopy(reviewed["execution_scope"]["findings_contract"])
    result["provider_backend"] = deepcopy(reviewed["execution_scope"]["provider_backend"])
    result["execution_time_scope"] = {"maximum_seconds": seconds, "elapsed_seconds": time.monotonic() - started,
        "prior_execution_elapsed_ms": packet["prior_execution_timing"]["elapsed_ms"],
        "authorization_scope": "this_continuation_only"}
    _write_record(segment / "run_result.json", result)
    _, observed_manifest = _file(segment / "run_manifest.json")
    observed_manifest.update(status=result["status"], model_call_budget=budget, model_call_count=budget["consumed"],
        prior_model_call_count=packet["budget"]["consumed"], current_invocation_model_call_count=result["current_invocation_model_call_count"],
        model_call_count_scope="cumulative_same_task")
    _write_record(segment / "run_manifest.json", observed_manifest)
    receipt.update(status="recorded", model_call_budget=budget)
    _write_record(claim_path, receipt)
    return result
