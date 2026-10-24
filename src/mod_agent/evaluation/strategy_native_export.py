"""Export strategy-evaluation inputs from persisted native v2 execution records.

为什么要有它：动作、经验送达和资源来自实际执行原件；事后关联protocol不能冒充运行前冻结。
"""
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from src.libs.llm.agent_turn import AgentUsage
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.native_execution_lineage import validate_native_execution_lineage
from src.mod_agent.model_call_budget import validate_budget
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_execution import _admitted_response, _read
from src.mod_agent.task_agent_v2_experiments import _source
from src.mod_agent.task_agent_v2_state import TaskAgentV2State


def _require(condition, code):
    if not condition:
        raise ValueError("STRATEGY_NATIVE_" + code)


def _learning(native_ref, native, learning_ref):
    if learning_ref is None:
        return native["budget_after"], [], 0, None
    learning = read_bound_json(learning_ref)
    _require(learning.get("original_execution_ref") == native_ref, "LEARNING_EXECUTION_CHANGED")
    before, after = validate_budget(learning["budget_before"]), validate_budget(learning["budget_after"])
    _require(before == native["budget_after"] and after["consumed"] - before["consumed"] == learning["model_calls"],
        "LEARNING_BUDGET_CHANGED")
    configuration = read_bound_json(learning["configuration_ref"])
    store = ArtifactStore(configuration["store_root"])
    known = []
    for receipt in learning.get("call_receipts", []):
        if receipt.get("response_ref"):
            response = store.read(ArtifactRef(**receipt["response_ref"]))["data"]["response"]
            usage = AgentUsage(**response["usage"]).to_dict()
            _require(usage == receipt["usage"], "LEARNING_USAGE_CHANGED")
            known.append(usage)
        elif receipt.get("extraction_ref"):
            extraction = store.read(ArtifactRef(**receipt["extraction_ref"]))["data"]
            _require(extraction.get("status") == "execution_unknown", "LEARNING_UNKNOWN_ORIGIN_CHANGED")
    unknown = learning["model_calls"] - len(known)
    _require(unknown >= 0 and known == learning["known_usage"] and unknown == learning["unknown_usage_calls"],
        "LEARNING_USAGE_ACCOUNTING_CHANGED")
    return after, known, unknown, learning


def _host_and_freeze(state, native, request):
    anchor = state.head["operations"].get("host-recovery-config", {}).get("result", {}).get("host_config_ref")
    if anchor is None:
        return None, {"status": "freeze_binding_unproven", "reason": "No native host configuration anchor.",
            "protocol_ref": request["protocol_ref"]}
    host = state.load(anchor)
    sequence, first_model, selected = None, None, state.head["event_ref"]
    reservations = {"model-reservation:" + row["reservation_id"] for row in native["provider_turns"]}
    while selected is not None:
        event = state.load(selected)
        if event["operation_id"] == "host-recovery-config":
            _require(event["result"].get("host_config_ref") == anchor, "HOST_CONFIG_ANCHOR_CHANGED")
            sequence = event["sequence"]
        if event["operation_id"] in reservations:
            first_model = min(first_model, event["sequence"]) if first_model is not None else event["sequence"]
        selected = event["previous_event_ref"]
    _require(sequence is not None and first_model is not None and sequence < first_model,
        "HOST_CONFIGURATION_NOT_BOUND_BEFORE_MODEL")
    configuration_ref = host.get("experience_configuration_ref")
    configuration = read_bound_json(configuration_ref) if configuration_ref else {}
    trial_ref = configuration.get("trial_binding_ref")
    if trial_ref is None:
        return host, {"status": "freeze_binding_unproven", "reason": "Native host configuration has no preregistered trial binding.",
            "protocol_ref": request["protocol_ref"], "host_config_ref": anchor}
    binding = read_bound_json(trial_ref)
    _require(binding.get("schema_version") == "strategy_trial_run_binding_v1"
        and all(binding.get(key) == request[key] for key in ("protocol_ref", "task_id", "attempt", "arm")),
        "PREREGISTERED_TRIAL_CHANGED")
    return host, {"status": "protocol_bound_before_model", "protocol_ref": request["protocol_ref"],
        "trial_binding_ref": trial_ref, "configuration_ref": configuration_ref, "host_config_ref": anchor,
        "host_event_sequence": sequence, "first_model_reservation_sequence": first_model}


def _actual_conditions(state, native, host, original_request, baseline):
    profile = read_bound_json(state.contract["execution_profile_ref"])
    profile = {**profile, "dotnet_ref": {"sha256": profile["dotnet_ref"]["sha256"]}}
    initial = validate_budget(host["original_budget_before"] if host else native["budget_before"])
    return {"task": {"original_request_sha256": canonical_hash(original_request),
            "source_files_sha256": canonical_hash({p: row["sha256"] for p, row in baseline["files"].items()})},
        "model": native["model_profile"],
        "tools": {"protocol": "v2", "owner_sha256": {Path(row["path"]).name: row["sha256"]
            for row in (host or {}).get("owner_refs", [])}},
        "permissions": {"scope": state.contract["scope"], "operation": state.contract["operation"],
            "allowed_edit_paths": state.contract["allowed_edit_paths"], "read_root_ids": sorted(state.contract["read_roots"])},
        "budget": {"policy": initial.get("policy"), "maximum": initial["maximum"],
            "task_loop_policy": (host or {}).get("policy"),
            "stage_policy": {key: {k: v for k, v in row.items() if k not in {"consumed", "remaining"}}
                for key, row in initial["by_stage"].items()}},
        "judge": {"public_profile": profile, "validator_sha256": native["expected_verification_identity"]["validator_sha256"]},
        "initial_environment": {"model_identity": native["expected_model_identity"],
            "checker_environment_sha256": native["expected_verification_identity"]["environment_sha256"]},
        "feedback": {"public_acceptance": state.load(state.contract["public_acceptance_ref"])["conditions"]}}


def _timing(request, native, host, candidate_source_id):
    from src.mod_agent.task_agent_v2_execution import task_timing_snapshot

    origin_ref = (host or {}).get("task_timing_origin_ref")
    reference = request.get("timing_observation_ref")
    observation = read_bound_json(reference) if reference is not None else None
    scope = "native_boundary"
    if request.get("delivery_receipt_ref") is not None:
        delivered, _verification = _delivery(request, candidate_source_id)
        if delivered.get("task_timing_observation") is not None:
            observation = delivered["task_timing_observation"]
            scope = "verified_delivery_boundary"
    if origin_ref is None or observation is None:
        return {"status": "unknown", "duration_ms": None, "reason": "no_bound_entry_origin_or_observation"}
    _require(observation.get("origin_ref") == origin_ref
        and observation.get("native_execution_ref") in (None, request["native_execution_ref"]), "TIMING_TASK_CHANGED")
    if scope == "verified_delivery_boundary" and observation.get("status") == "unknown" and not observation.get("observed_clock"):
        return {"status": "unknown", "duration_ms": None, "origin_ref": origin_ref,
            "reason": "delivery_timing_observation_unavailable", "source": scope}
    if observation.get("native_execution_ref") is None and scope == "native_boundary":
        _require(reference == native.get("timing_observation_ref"), "TIMING_NATIVE_ANCHOR_REQUIRED")
    expected = task_timing_snapshot(origin_ref, observation["event"], observation["status"],
        observation.get("native_execution_ref"), observed_clock=observation["observed_clock"])
    _require(all(observation.get(key) == value for key, value in expected.items()), "TIMING_OBSERVATION_CHANGED")
    returned = observation["event"] in {"product_return", "resumed_product_return", "product_build_return"}
    current = scope == "verified_delivery_boundary" or observation.get("resource_event_ref") == request["resource_event_ref"]
    return {"status": observation["timing_status"] if returned and current else "observed_lower_bound_not_current_return",
        "duration_ms": observation["duration_ms"] if returned and current else None,
        "known_observed_elapsed_ms": observation["duration_ms"],
        "origin_ref": origin_ref, "observation_ref": reference, "boundary": observation["event"],
        "clock_basis": observation["clock_basis"], "scope": observation["scope"], "source": scope,
        "whole_task_finished": False}


def _technical_inputs(native, root):
    from src.mod_agent.task_agent_v2_execution import classify_v2_technical_inputs

    assessments, prior_responses = [], []
    complete = True
    for event in native["provider_turns"]:
        if event.get("technical_input_audit_ref") is None:
            complete = False
        else:
            audit = _read(event["technical_input_audit_ref"], root)
            host = TaskAgentV2State(root, native["task_contract_ref"]).load(audit["host_config_ref"])
            _require(audit.get("schema") == "task-agent-technical-input-audit.v2"
                and audit["actual_request_ref"] == event["request_ref"]
                and host["graph_state"]["run_id"] == native["run_id"]
                and host["graph_state"]["request_id"] == native["request_id"], "TECHNICAL_INPUT_ORIGIN_CHANGED")
            recorded_prior = [_read(ref, root) for ref in audit["prior_response_refs"]]
            _require(recorded_prior == prior_responses, "TECHNICAL_MODEL_HISTORY_CHANGED")
            assessment = classify_v2_technical_inputs(_read(audit["loop_request_ref"], root),
                _read(event["request_ref"], root), _read(host["initial_request_ref"], root),
                host.get("source_context_messages", []), recorded_prior)
            _require(assessment == audit["assessment"], "TECHNICAL_INPUT_ASSESSMENT_CHANGED")
            assessments.append({"audit_ref": event["technical_input_audit_ref"], "assessment": assessment})
        if event.get("response_ref"):
            prior_responses.append(_read(event["response_ref"], root))
    closed = complete and bool(assessments) and all(row["assessment"]["closed_loop_human_technical_interventions"] == 0 for row in assessments)
    return {"status": "closed_input_scope_verified" if closed else "input_scope_incomplete_or_unknown",
        "scope": "v2_model_decision_inputs_after_initial_contract", "closed_loop_human_technical_interventions": 0 if closed else None,
        "outside_scope_human_technical_interventions": None, "audit_refs": [row["audit_ref"] for row in assessments],
        "excluded_from_technical_interventions": ["initial_requirements", "permission_approval"],
        "outside_scope": ["external_requirement_or_test_authoring", "strategy_experience_authorship", "out_of_band_assistance"]}


def _project(request):
    from src.mod_agent.evaluation.strategy_adoption import _planned_pairs

    root = Path(request["artifact_store_root"]).resolve()
    native_ref = request["native_execution_ref"]
    native = _read(native_ref, root)
    _require(native.get("schema_version") == "task-agent-product-execution.v2" and native.get("protocol") == "v2",
        "NATIVE_EXECUTION_REQUIRED")
    _require(native_ref["path"] == str((Path(native_ref["path"]).parent / "execution.json").resolve()),
        "NATIVE_EXECUTION_LOCATION_REQUIRED")
    protocol = read_bound_json(request["protocol_ref"])
    _require(protocol.get("schema_version") == "strategy_trial_protocol_v1"
        and canonical_hash(protocol["invariants"]) == protocol["invariants_sha256"], "PROTOCOL_CHANGED")
    planned = _planned_pairs(protocol)
    selected = planned.get((request["task_id"], request["attempt"]))
    _require(selected is not None and request["arm"] in {"B", "B+"}, "TASK_ARM_NOT_PLANNED")
    state = TaskAgentV2State(root, native["task_contract_ref"])
    resource_ref = request["resource_event_ref"]
    selected_event, resource_event = state.head["event_ref"], None
    while selected_event is not None:
        item = state.load(selected_event)
        if selected_event == resource_ref:
            resource_event = item
            break
        selected_event = item["previous_event_ref"]
    _require(resource_event is not None and resource_event["contract_ref"] == state.contract_ref,
        "RESOURCE_EVENT_NOT_IN_NATIVE_HISTORY")
    lineage = validate_native_execution_lineage(artifact_store_root=root, native_execution_ref=native_ref)
    initial_state, initial_native = lineage["initial_state"], lineage["initial_native"]
    initial_host, freeze = _host_and_freeze(initial_state, initial_native, request)
    host = (state.load(state.head["operations"]["host-recovery-config"]["result"]["host_config_ref"])
        if state.head["operations"].get("host-recovery-config") else None)
    baseline = initial_state.load(initial_state.contract["baseline_manifest_ref"])
    source = read_bound_json(selected["source_ref"])
    original_request = initial_state.load(initial_state.contract["original_request_ref"])["original_request"]
    actual_conditions = _actual_conditions(initial_state, initial_native, initial_host, original_request, baseline)
    expected_source = {"source_files_sha256": canonical_hash({path: row["sha256"] for path, row in baseline["files"].items()}),
        "original_request_sha256": canonical_hash(original_request)}
    _require(all(source[key] == value for key, value in expected_source.items() if key in source), "PLANNED_TASK_SOURCE_CHANGED")
    source_binding = "source_and_request_verified" if all(key in source for key in expected_source) else "source_metadata_binding_unproven"
    revision, candidate_id, _files = _source(state, native["candidate_ref"] or "baseline")
    loop = _read(native["loop_result_ref"], root)
    _require(loop["model_budget"] == native["budget_after"] and loop["termination"] == native["termination"],
        "NATIVE_LOOP_CHANGED")
    records = native["completed_tool_records"]
    _require([row["event"] for row in records] == [event for event in loop["events"]
        if event.get("event") == "tool_dispatch_completed"], "NATIVE_TOOL_SEQUENCE_CHANGED")
    calls = [row for row in records if row["call_id"] == request["call_id"]]
    _require(len(calls) == 1, "ACTUAL_CALL_REQUIRED")
    call = calls[0]
    arguments, response, output = (_read(call[key], root) for key in ("arguments_ref", "response_ref", "result_ref"))
    _admitted_response(response, native["model_profile"])
    _require([row for row in response["tool_calls"] if row["call_id"] == call["call_id"]] == [{
        "call_id": call["call_id"], "name": call["name"], "arguments": arguments}], "NATIVE_MODEL_ACTION_CHANGED")
    recorded = loop["call_records"].get(call["call_id"], {})
    _require(recorded.get("status") == "completed" and recorded.get("name") == call["name"]
        and recorded.get("arguments_sha256") == canonical_hash(arguments)
        and recorded.get("result", {}).get("output") == output and output.get("execution") == "executed"
        and call["event"]["output_sha256"] == canonical_hash(output), "NATIVE_TOOL_RESULT_CHANGED")
    known_usage, provider_matches, reservations = [], [], set()
    for event in native["provider_turns"]:
        reservation = event["reservation_id"]
        _require(reservation not in reservations, "DUPLICATE_PROVIDER_RESERVATION")
        reservations.add(reservation)
        dispatch = _read(event["dispatch_ref"], root)
        provider_request = _read(event["request_ref"], root)
        _require(dispatch["request_ref"] == event["request_ref"] and dispatch["reservation_id"] == reservation
            and dispatch["status"] == "requested", "PROVIDER_DISPATCH_CHANGED")
        outcome_path = Path(event["request_ref"]["path"]).parent / ("provider-outcome-" + reservation + ".json")
        _require(_read(_ref(outcome_path), root) == event, "PROVIDER_OUTCOME_CHANGED")
        if event.get("response_ref"):
            provider_response = _read(event["response_ref"], root)
            usage = AgentUsage(**provider_response["usage"]).to_dict()
            _require(event.get("usage") == usage, "PROVIDER_USAGE_CHANGED")
            known_usage.append(usage)
            if provider_response == response:
                provider_matches.append((event, provider_request))
    _require(len(provider_matches) == 1, "NATIVE_PROVIDER_ORIGIN_REQUIRED")
    event, provider_request = provider_matches[0]
    delivered = [message["content"]["strategy_experience"] for message in provider_request["messages"]
        if isinstance(message.get("content"), dict) and "strategy_experience" in message["content"]]
    snapshot_ref, delivery_ref = None, None
    if delivered:
        _require(request["arm"] == "B+" and len(delivered) == 1
            and delivered[0].get("snapshot_hash") == protocol["snapshot_ref"]["artifact_hash"], "EXPERIENCE_ARM_CHANGED")
        matching = []
        for path in Path(native_ref["path"]).parent.glob("experience-delivery-*.json"):
            delivery = _read(_ref(path), root)
            if _read(delivery["final_request_ref"], root) == provider_request:
                matching.append((_ref(path), delivery))
        _require(len(matching) == 1, "EXPERIENCE_DELIVERY_ORIGIN_REQUIRED")
        delivery_ref, delivery = matching[0]
        selection = _read(delivery["selection_ref"], root)
        _require(delivery["snapshot_ref"] == protocol["snapshot_ref"]
            and selection["model_view"] == delivered[0]
            and delivery["model_view_sha256"] == selection["model_view_sha256"] == canonical_hash(delivered[0]),
            "EXPERIENCE_DELIVERY_CHANGED")
        snapshot_ref = delivery["snapshot_ref"]
    else:
        _require(request["arm"] == "B", "EXPERIENCE_NOT_DELIVERED")
    final_budget, learning_usage, unknown_learning, learning = _learning(native_ref, native, request.get("learning_receipt_ref"))
    known_usage.extend(learning_usage)
    reported_budget_calls = validate_budget(final_budget)["consumed"]
    total_calls = resource_event["resources"].get("model_calls")
    _require(type(total_calls) is int and total_calls >= reported_budget_calls, "RESOURCE_LEDGER_BEHIND_EXECUTION")
    unknown_calls = total_calls - len(known_usage)
    _require(unknown_calls >= unknown_learning >= 0, "NATIVE_RESOURCE_ACCOUNTING_CHANGED")
    known_sums = {key: sum(row[key] for row in known_usage) for key in ("input_tokens", "output_tokens")}
    timing = _timing(request, native, host, candidate_id)
    human_inputs = _technical_inputs(native, root)
    # 为什么要有它：时间从原入口量到已观察边界；封闭循环之外的人工帮助仍unknown，不能用局部0冒充全局无人帮助。
    costs = {"model_calls": total_calls, "input_tokens": known_sums["input_tokens"] if not unknown_calls else None,
        "output_tokens": known_sums["output_tokens"] if not unknown_calls else None, "duration_ms": timing["duration_ms"]}
    data = {"schema_version": "strategy_execution_receipt_v1", "task_id": request["task_id"],
        "attempt": request["attempt"], "arm": request["arm"], "source_ref": selected["source_ref"],
        "exposure_ref": selected["exposure_ref"], "conditions_sha256": protocol["invariants_sha256"],
        "candidate_source_id": candidate_id, "candidate_state": {"artifact_store_root": str(root),
            "task_contract_ref": state.contract_ref, "candidate_ref": revision},
        "model_request_ref": event["request_ref"], "experience_snapshot_ref": snapshot_ref,
        "native_execution_ref": native_ref, "native_call_id": call["call_id"], "known_cost": costs,
        "execution_lineage": {"initial_native_ref": lineage["initial_native_ref"],
            "initial_checkpoint_ref": lineage["initial_checkpoint_ref"], "chain": lineage["chain"]},
        "human_technical_interventions": None,
        "resource_evidence": {"budget_after": final_budget, "known_usage_sums": known_sums,
            "unknown_usage_calls": unknown_calls, "unknown_learning_usage_calls": unknown_learning,
            "resource_event_ref": resource_ref, "ledger_model_calls": total_calls,
            "unreconciled_calls_after_native_and_learning": total_calls - reported_budget_calls,
            "learning_receipt_ref": request.get("learning_receipt_ref"), "native_provider_turn_count": len(native["provider_turns"]),
            "count_basis": final_budget.get("count_basis"), "duration_source": timing,
            "human_intervention_source": human_inputs},
        "experience_delivery_ref": delivery_ref,
        "freeze_binding": freeze, "actual_common_conditions": actual_conditions,
        "actual_common_conditions_sha256": canonical_hash(actual_conditions),
        "planned_source_binding": source_binding,
        "native_deadline_utc": (host or {}).get("deadline_utc"),
        "native_status": native["status"], "exporter_ref": _ref(__file__)}
    action = {"schema_version": "strategy_action_receipt_v1", "request_ref": event["request_ref"],
        "response_ref": event["response_ref"], "call_id": call["call_id"], "execution_origin": "executed",
        "action": {"name": call["name"], "arguments": arguments}, "native_execution_ref": native_ref,
        "native_output_ref": call["result_ref"]}
    output_projection = {"call_id": call["call_id"], "native_result_ref": call["result_ref"], "output": output}
    return data, action, output_projection


def _delivery(request, candidate_source_id):
    if request.get("delivery_receipt_ref") is None:
        return {"status": "not_verified_by_native_execution", "candidate_source_id": candidate_source_id,
            "native_execution_ref": request["native_execution_ref"]}, None
    from src.mod_agent.task_agent_v2_delivery import read_completed_v2_delivery

    result = read_completed_v2_delivery(artifact_store_root=request["artifact_store_root"],
        receipt_ref=request["delivery_receipt_ref"], native_execution_ref=request["native_execution_ref"])
    _require(result["candidate_source_id"] == candidate_source_id, "DELIVERY_CANDIDATE_CHANGED")
    return {key: value for key, value in result.items() if key != "verification"}, result["verification"]


def export_strategy_execution(*, native_execution_ref, call_id, output_root, artifact_store_root=None,
                              protocol_ref=None, task_id=None, arm=None, attempt=None, learning_receipt_ref=None,
                              delivery_receipt_ref=None):
    """Export actual facts; callers choose an existing call, never its contents."""
    if artifact_store_root is None:
        header = read_bound_json(native_execution_ref)
        _require(header.get("schema_version") == "task-agent-product-execution.v2", "NATIVE_EXECUTION_REQUIRED")
        contract_ref = header.get("task_contract_ref", "")
        _require(isinstance(contract_ref, str) and re.fullmatch(r"v2/contract/[0-9a-f]{64}\.json", contract_ref), "NATIVE_CONTRACT_REQUIRED")
        roots = [parent for parent in Path(native_execution_ref["path"]).resolve().parents if (parent / contract_ref).is_file()]
        _require(len(roots) == 1, "NATIVE_STORE_LOCATION_REQUIRED")
        artifact_store_root = roots[0]
    else:
        artifact_store_root = Path(artifact_store_root).resolve()
    native = _read(native_execution_ref, artifact_store_root)
    _require(native.get("schema_version") == "task-agent-product-execution.v2", "NATIVE_EXECUTION_REQUIRED")
    state = TaskAgentV2State(artifact_store_root, native["task_contract_ref"])
    anchor = state.head["operations"].get("host-recovery-config", {}).get("result", {}).get("host_config_ref")
    host = state.load(anchor) if anchor else {}
    config = read_bound_json(host["experience_configuration_ref"]) if host.get("experience_configuration_ref") else {}
    binding = read_bound_json(config["trial_binding_ref"]) if config.get("trial_binding_ref") else {}
    arguments = {"protocol_ref": protocol_ref, "task_id": task_id, "arm": arm, "attempt": attempt}
    for key, value in arguments.items():
        if value is None:
            arguments[key] = binding.get(key)
        elif key in binding:
            _require(value == binding[key], "PREREGISTERED_TRIAL_CHANGED")
    _require(all(value is not None for value in arguments.values()), "TRIAL_BINDING_OR_EXPLICIT_DIAGNOSTIC_REQUIRED")
    if learning_receipt_ref is None:
        matches = [_ref(path) for path in Path(native_execution_ref["path"]).parent.glob("terminal-learning-receipt-*.json")
            if read_bound_json(_ref(path)).get("original_execution_ref") == native_execution_ref]
        _require(len(matches) <= 1, "AMBIGUOUS_LEARNING_RECEIPTS")
        learning_receipt_ref = matches[0] if matches else None
    request = {"native_execution_ref": native_execution_ref, "artifact_store_root": str(Path(artifact_store_root).resolve()),
        **arguments,
        "call_id": call_id, "learning_receipt_ref": learning_receipt_ref, "resource_event_ref": state.head["event_ref"],
        "delivery_receipt_ref": delivery_receipt_ref}
    observations = [(_ref(path), read_bound_json(_ref(path))) for path in Path(native_execution_ref["path"]).parent.glob("task-observation-*.json")]
    observations = [(ref, value) for ref, value in observations if value.get("native_execution_ref") == native_execution_ref]
    request["timing_observation_ref"] = (max(observations, key=lambda row: row[1]["observation_sequence"])[0]
        if observations else native.get("timing_observation_ref"))
    execution, action, output = _project(request)
    delivered, verification = _delivery(request, execution["candidate_source_id"])
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    request_ref = _save(root / "export-request.json", request)
    output_ref = _save(root / "tool-output.json", output)
    action_ref = _save(root / "action.json", {**action, "output_ref": output_ref})
    verification_ref = _save(root / "delivery-verification.json", verification) if verification is not None else None
    delivery_ref = _save(root / "delivery.json", {**delivered, "verification_ref": verification_ref})
    reference = _save(root / "execution.json", {**execution, "action_ref": action_ref, "delivery_ref": delivery_ref,
        "native_export_request_ref": request_ref})
    validate_native_strategy_execution(reference)
    return reference


def validate_native_strategy_execution(reference):
    execution = read_bound_json(reference)
    _require(execution.get("native_export_request_ref") and execution.get("exporter_ref") == _ref(__file__),
        "PRODUCTION_EXPORT_REQUIRED")
    request = read_bound_json(execution["native_export_request_ref"])
    expected, action, output = _project(request)
    _require(all(execution.get(key) == value for key, value in expected.items()), "EXPORTED_NATIVE_FACTS_CHANGED")
    actual_action = read_bound_json(execution["action_ref"])
    _require({key: value for key, value in actual_action.items() if key != "output_ref"} == action
        and read_bound_json(actual_action["output_ref"]) == output, "EXPORTED_NATIVE_ACTION_CHANGED")
    delivery = read_bound_json(execution["delivery_ref"])
    delivered, verification = _delivery(request, expected["candidate_source_id"])
    _require({key: value for key, value in delivery.items() if key != "verification_ref"} == delivered
        and (read_bound_json(delivery["verification_ref"]) if delivery["verification_ref"] else None) == verification,
        "EXPORTED_DELIVERY_CHANGED")
    return {"status": "native_execution_verified", "freeze_binding": deepcopy(execution["freeze_binding"]),
        "actual_common_conditions_sha256": execution["actual_common_conditions_sha256"],
        "planned_source_binding": execution["planned_source_binding"],
        "unknown_usage_calls": execution["resource_evidence"]["unknown_usage_calls"],
        "scope": "native_action_delivery_and_metering_only_not_fair_comparison_or_product_delivery"}
