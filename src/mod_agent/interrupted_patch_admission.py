"""Pure admission of a complete native response interrupted before verification."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path

from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.saved_patch_admission import _hash, _json, _read


def _require(condition, code):
    if not condition:
        raise ValueError("INTERRUPTED_PATCH_" + code)


def retained_request_from_document(document):
    """Decode the recorded effective request without calling a provider."""
    from src.libs.llm.agent_turn import (
        AgentMessage, AgentResponseFormat, AgentToolDefinition, AgentToolExchange,
        AgentToolResult, AgentTurnRequest,
    )
    from src.mod_agent.proposal.patch_agent_loop import _without_loop_states

    _require(isinstance(document, dict) and set(document) == {field.name for field in fields(AgentTurnRequest)},
             "REQUEST_SHAPE_INVALID")
    try:
        request = AgentTurnRequest(**{**document,
            "messages": tuple(AgentMessage(**item) for item in document["messages"]),
            "tools": tuple(AgentToolDefinition(**item) for item in document["tools"]),
            "tool_results": tuple(AgentToolResult(**item) for item in document["tool_results"]),
            "historical_tool_exchanges": tuple(AgentToolExchange(**{**item,
                "tool_results": tuple(AgentToolResult(**row) for row in item["tool_results"])})
                for item in document["historical_tool_exchanges"]),
            "response_format": AgentResponseFormat(**document["response_format"]) if document["response_format"] else None})
    except (TypeError, KeyError) as exc:
        raise ValueError("INTERRUPTED_PATCH_REQUEST_SHAPE_INVALID") from exc
    _require(_hash(asdict(request)) == _hash(document) and len(request.messages) >= 2,
             "REQUEST_ROUNDTRIP_MISMATCH")
    return _without_loop_states(request)


def _read_inputs(journal, store, workspace, original_worker_ref):
    from src.mod_agent.fresh_planning_continuation import (
        _admit_compilation_candidate, _read_selected_exact_revision, _validate_exact_upstream_set,
        _validate_selected_plan, _validate_source_items,
    )
    selected = journal["selected_artifacts"]
    kinds = ("feature_spec", "evidence_pack", "implementation_plan")
    _require(isinstance(selected, dict) and set(selected) == set(kinds), "SELECTED_ARTIFACTS_INVALID")
    refs = {kind: ArtifactRef(**selected[kind]) for kind in kinds}
    _require(all(ref.artifact_type == kind and ref.feature_id == journal["feature_id"]
                 for kind, ref in refs.items()), "SELECTED_ARTIFACTS_INVALID")
    envelopes = {kind: _read_selected_exact_revision(store, ref, consumer_stage=None) for kind, ref in refs.items()}
    feature, evidence, plan = (envelopes[kind]["data"] for kind in kinds)
    _require(envelopes["feature_spec"]["status"] == envelopes["implementation_plan"]["status"] == "success",
             "SELECTED_ARTIFACTS_BLOCKED")
    candidate = _admit_compilation_candidate(evidence, feature,
        worker_configuration_ref=original_worker_ref, require_selected_worker=True)
    _require(envelopes["evidence_pack"]["status"] == "success" or
        envelopes["evidence_pack"]["status"] == "blocked" and candidate is not None, "SELECTED_ARTIFACTS_BLOCKED")
    _validate_exact_upstream_set(envelopes["evidence_pack"], (refs["feature_spec"],))
    _validate_exact_upstream_set(envelopes["implementation_plan"], (refs["feature_spec"], refs["evidence_pack"]))
    _validate_selected_plan(plan, evidence, feature_id=journal["feature_id"])
    observed = _validate_source_items(evidence, workspace)
    _require(observed == journal["source_validation"] and feature["user_request"] == evidence["query"],
             "SOURCE_EVIDENCE_CHANGED")
    return feature, evidence, plan


def prepare_retained_generation(feature, evidence, plan, response):
    from src.libs.llm import ChatResponse
    from src.mod_agent.local_entry import _external_evidence_items, _patch_generation_obligation_projection
    from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator
    obligations = _patch_generation_obligation_projection(feature_spec=feature,
        implementation=plan, planned_changes=list(plan["proposed_changes"]), evidence_pack=evidence)
    generator = LLMExactPatchGenerator(None)
    prepared = generator.prepare_native_contract(user_request=feature["user_request"],
        **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
        evidence_items=_external_evidence_items(evidence["items"]), binding_registry=plan["binding_registry"],
        constraints=obligations["side_effect_boundaries"], target_behavior=obligations["target_behavior"],
        acceptance_criteria=obligations["acceptance_criteria"], runtime_verification_plan=obligations["runtime_verification_plan"],
        required_change_units=obligations["required_change_units"],
        required_validation_by_binding={row["edit_binding_id"]: row["expected_validation"] for row in obligations["required_change_units"]})
    generated = generator.materialize_prepared_native(prepared, ChatResponse(content=response["content"],
        model=response["model"], usage=response.get("usage"), raw_response=response.get("raw_response")))
    return generator, prepared, generated


def admit_interrupted_patch(selection, *, workspace_root, artifact_store, worker_configuration_ref=None):
    """Bind exact incomplete-attempt facts without manufacturing a terminal result.

    为什么要有它：完整回答可复用，但中断记录不能冒充已完成验收或刷新原任务预算。
    """
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch
    from src.mod_agent.fresh_planning_continuation import _csharp_source_snapshot, _source_tree_snapshot_sha256
    from src.mod_agent.model_call_budget import validate_budget
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256
    from src.mod_agent.proposal.patch_agent_loop import iter_patch_recovery_revisions

    _require(isinstance(selection, dict) and set(selection) == {"source_attempt_ref", "parent_manifest_ref", "execution_stop_ref"},
             "SELECTION_INVALID")
    stop = _json(selection["execution_stop_ref"])
    # 为什么要有它：完整回答的持久化不代表原进程已停止；可信host停止证明是恢复前置输入，不宣称自动检测OS全局互斥。
    _require(stop.get("schema_version") == "interrupted-execution-stop.v1" and stop.get("origin") == "host_controller"
        and stop.get("original_execution_stopped") is True and stop.get("source_model_claim") is False
        and all(stop.get(key) == selection[key] for key in ("source_attempt_ref", "parent_manifest_ref"))
        and all(isinstance(stop.get(key), str) and stop[key].strip() for key in ("stop_method", "execution_handle", "evidence"))
        and isinstance(stop.get("terminal_result"), dict) and type(stop["terminal_result"].get("exit_code")) is int
        and isinstance(stop.get("confirmation"), dict) and type(stop["confirmation"].get("matching_process_count")) is int
        and stop["confirmation"]["matching_process_count"] == 0
        and all(isinstance(stop["confirmation"].get(key), str) and stop["confirmation"][key].strip()
                for key in ("tool", "observed_utc", "scope")), "HOST_STOP_EVIDENCE_REQUIRED")
    workspace = Path(workspace_root).resolve(strict=True)
    journal = _json(selection["source_attempt_ref"])
    manifest = _json(selection["parent_manifest_ref"])
    _require(Path(selection["parent_manifest_ref"]["path"]).resolve() == workspace / "run_manifest.json"
        and Path(selection["source_attempt_ref"]["path"]).resolve().is_relative_to(artifact_store.root.resolve())
        and Path(manifest["isolated_workspace"]).resolve() == workspace
        and Path(journal["caller"]["workspace_root"]).resolve() == workspace
        and manifest["run_id"] == journal["parent_run_id"] and manifest["request_id"] == journal["request_id"],
        "PARENT_IDENTITY_MISMATCH")
    _require(journal.get("receipt_version") == "patch_continuation_attempt_v1"
        and journal.get("continuation_selection") == "product_graph_current_implementation_plan"
        and journal.get("status") == "local_patch_post_response_model_admitted"
        and journal.get("proposal_admitted") is False and not journal.get("artifacts")
        and not any(journal.get(key) for key in ("terminal_result", "parent_product_result_identity", "parent_product_run_id"))
        and not manifest.get("terminal_record") and not (workspace / "run_result.json").exists()
        and manifest.get("actual_write_performed") is False, "ATTEMPT_NOT_INTERRUPTED")
    factory = journal["factory_receipt"]
    budget = validate_budget(journal["shared_model_budget"])
    count = factory["model_call_count"]
    _require(type(count) is int and count > 0 and count == journal["model_call_count"]
        and count <= budget["by_stage"][journal["model_role"]]["consumed"]
        and count + budget["remaining"] <= journal["max_model_calls"]
        and factory["model_call_budget"]["maximum_total_model_calls"] == journal["max_model_calls"], "BUDGET_MISMATCH")
    events = journal.get("events") or []
    _require(events and events[-1].get("phase") == journal["status"]
        and events[-1].get("model_call_count") == count
        and events[-1].get("shared_model_budget") == budget
        and events[-1].get("factory_receipt") == factory, "LAST_EVENT_MISMATCH")
    _require(factory.get("status") == "model_response_persisted" and factory.get("failure") is None
        and factory.get("upstream_decision") is None, "RESPONSE_BOUNDARY_INVALID")
    original_ref = journal["worker_configuration_ref"]
    _read(original_ref)
    original_config = PatchWorkerConfiguration.load(original_ref["path"])
    selected_ref = worker_configuration_ref or original_ref
    _read(selected_ref)
    configuration = PatchWorkerConfiguration.load(selected_ref["path"])
    _require(configuration.reference == selected_ref and original_config.reference == original_ref
        and configuration.policy == original_config.policy and configuration.enable_recovery_loop
        and original_config.enable_recovery_loop and configuration.behavior_evaluator() is not None,
        "WORKER_CONFIGURATION_CHANGED")
    # 为什么要有它：恢复可显式更换验收实现，原模型策略与动态任务allowance仍须保持原义。
    policy = replace(original_config.policy, max_total_model_turns=journal["max_model_calls"],
        model_call_budget_policy=budget.get("policy", "stage_and_total"))
    _require(asdict(policy) == factory["patch_agent_loop"]["policy"]
        and journal["worker_profile_sha256"] == _hash(factory["profile"])
        and all(factory["profile"][key] == manifest["ollama_patch_generation_profile"][key]
                for key in ("context_window", "output_budget", "timeout_seconds", "automatic_retry", "automatic_provider_fallback"))
        and factory["model_identity"] == factory["post_response_model_identity"]
        and factory["model_identity"] == manifest["local_model_identity"], "MODEL_PROFILE_CHANGED")
    target = Path(manifest["nuget_isolated_build_profile"]["working_directory"]).resolve(strict=True)
    baseline = Path(manifest["baseline_build"]["build_profile"]["working_directory"]).resolve(strict=True)
    _require(target.is_relative_to(workspace / "generated_mods") and baseline.is_relative_to(workspace / "baseline_source"),
             "SOURCE_SCOPE_INVALID")
    expected = manifest["source_snapshot_sha256"]
    _require(all(source_snapshot_sha256(source_snapshot(path)) == expected
        for path in (Path(manifest["target_source"]), target, baseline)), "SOURCE_CHANGED")
    # 为什么要有它：验收产生的编译副本不是源码前像；原两份产品源码及baseline构建生成物仍按原journal整体hash绑定。
    initial_paths = tuple(sorted(((str(path.relative_to(workspace).as_posix()) + "/" + name, digest)
        for path in (baseline, target) for name, digest in _csharp_source_snapshot(path)), key=lambda item: item[0]))
    _require(_source_tree_snapshot_sha256(initial_paths) == journal["generated_against_snapshot_sha256"]
        == journal["current_snapshot_sha256"], "ORIGINAL_SNAPSHOT_CHANGED")
    feature, evidence, plan = _read_inputs(journal, artifact_store, workspace, original_ref)
    _require(feature["user_request"] == manifest["request"], "REQUEST_CHANGED")
    response = factory["model_response"]
    raw = _read({"path": response["path"], "sha256": response["output_sha256"]})
    _require(response.get("retained") is True and response.get("done") is True and response.get("encoding") == "utf-8"
        and response.get("output_utf8_bytes") == len(raw) and response.get("model") == factory["profile"]["model"],
        "RETAINED_RESPONSE_INVALID")
    turns = factory["worker_turns"]
    _require(len(turns) == count and all(turn.get("status") == "responded" and turn.get("turn_index") == index
        for index, turn in enumerate(turns, 1)), "INCOMPLETE_OR_LATER_SEND")
    calibrations, calibration_refs = [], []
    for turn in turns:
        sent = _json(turn["request_ref"])
        retained = _json(turn["response_ref"])
        wire = _json(turn["provider_request_ref"])
        _require(turn.get("response_origin") == "live_provider_response" and turn["request_sha256"] == _hash(sent)
            and turn["provider_request_sha256"] == turn["provider_request_ref"]["sha256"]
            and retained.get("provider_diagnostic", {}).get("request_identity", {}).get("provider_body_sha256")
                == turn["provider_request_ref"]["sha256"]
            and wire.get("model") == retained.get("model") == response["model"], "NATIVE_TURN_MISMATCH")
        options = wire.get("options") or {}
        _require(options.get("num_ctx") == factory["profile"]["context_window"]
            and options.get("num_predict") == factory["profile"]["output_budget"]
            and options.get("seed") == 0 and options.get("temperature") == 0, "NATIVE_PROFILE_CHANGED")
        prior_native = retained.get("raw_response") or {}
        prompt_tokens = prior_native.get("prompt_eval_count")
        if (prior_native.get("done") is True and prior_native.get("model") == response["model"]
                and type(prompt_tokens) is int and prompt_tokens > 0):
            body_bytes = len(_read(turn["provider_request_ref"]))
            _require(retained["provider_diagnostic"]["request_identity"].get("provider_body_utf8_bytes") == body_bytes,
                     "NATIVE_BODY_SIZE_MISMATCH")
            calibrations.append((body_bytes, prompt_tokens))
            calibration_refs.append({"provider_request_ref": deepcopy(turn["provider_request_ref"]),
                "response_ref": deepcopy(turn["response_ref"]), "body_utf8_bytes": body_bytes, "prompt_tokens": prompt_tokens})
    native = retained.get("raw_response") or {}
    _require(retained.get("content") == raw.decode("utf-8") and not retained.get("tool_calls")
        and native.get("done") is True and native.get("model") == response["model"]
        and native.get("message", {}).get("content") == retained["content"]
        and not native.get("message", {}).get("tool_calls"), "NATIVE_FINAL_RESPONSE_MISMATCH")
    retained_request = retained_request_from_document(sent)
    generator, prepared, generated = prepare_retained_generation(feature, evidence, plan,
        {"content": retained["content"], "model": response["model"], "usage": response.get("usage"), "raw_response": native})
    _require(prepared.schema_sha256 == factory["patch_response_contract"]["schema_sha256"]
        and retained_request.response_format is not None
        and retained_request.response_format.schema == prepared.response_schema
        and wire.get("format") == prepared.response_schema, "NATIVE_SCHEMA_CHANGED")
    materialized = materialize_exact_text_patch(workspace_root=workspace, diff_intent=generated.changes, implementation_plan=plan)
    loop = factory["patch_agent_loop"]
    revisions = list(iter_patch_recovery_revisions(loop))
    phases = [loop, *(row["execution"] for row in revisions if isinstance(row.get("execution"), dict))]
    _require(sum(phase["model_turns_used"] for phase in phases) == count, "TURN_ACCOUNTING_CHANGED")
    observations = {"consumed_tool_rounds": sum(phase["tool_rounds_used"] for phase in phases),
        "source_observation_bytes": sum(phase["total_observation_bytes"] for phase in phases),
        "recovery_observation_bytes": loop.get("recovery_observation_bytes", 0)}
    _require(observations["consumed_tool_rounds"] <= policy.max_tool_rounds
        and observations["source_observation_bytes"] + observations["recovery_observation_bytes"] <= policy.max_observation_bytes,
        "OBSERVATION_BUDGET_CHANGED")
    _require(not any(row.get("model_response_sha256") == response["output_sha256"]
        for row in loop.get("behavior_evaluations", [])), "FINAL_RESPONSE_ALREADY_EVALUATED")
    for report in loop.get("behavior_evaluations", []):
        persisted = _json(report["report_ref"])
        _require(all(persisted.get(key) == report.get(key) for key in ("status", "failure_owner", "observations")),
                 "HISTORICAL_VERIFICATION_CHANGED")
    from src.libs.llm.agent_turn import OllamaContextCalibration
    profile = factory["profile"]
    calibration = OllamaContextCalibration(
        tuple(factory["model_identity"].get(key) for key in ("base_url", "name", "model", "digest")),
        ("ollama_chat_local", profile["base_url"], profile["model"], profile["context_window"], profile["output_budget"],
         0, profile["timeout_seconds"]), _hash(profile), tuple(calibrations))
    return {"selection": deepcopy(selection), "selected_artifacts": deepcopy(journal["selected_artifacts"]),
        "feature_id": journal["feature_id"], "request_id": journal["request_id"], "parent_run_id": journal["parent_run_id"],
        "user_request": manifest["request"], "feature_spec": feature, "evidence_pack": evidence, "implementation_plan": plan,
        "shared_model_budget": budget, "factory_receipt": deepcopy(factory), "retained_request": retained_request,
        "retained_response": {"content": retained["content"], "model": response["model"], "usage": response.get("usage"), "raw_response": native},
        "worker_configuration_ref": deepcopy(selected_ref), "original_worker_configuration_ref": deepcopy(original_ref),
        "worker_policy": policy, "original_worker_allowance": journal["max_model_calls"], "inherited_model_call_count": count,
        "context_calibration": calibration, "context_calibration_refs": calibration_refs,
        "observation_budget": observations, "source_snapshot_sha256": expected,
        "materialized_code_identity": {"baseline_sha256": materialized.baseline_sha256,
            "code_sha256": materialized.code_sha256, "files": sorted(materialized.files)},
        "model_calls_performed": 0, "verifier_executions_performed": 0, "approval_inherited": False}
