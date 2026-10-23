"""Evidence-based decisions for a frozen strategy trial, never model self-rating.

为什么要有它：送达经验、实际行动变化和独立任务结果必须分别成立，汇总描述不能自动启用策略。
"""
from __future__ import annotations

import math
from dataclasses import asdict

from src.mod_agent.evaluation.memory_comparison import INVARIANTS
from src.mod_agent.model_qualification import (
    ModelQualificationError,
    canonical_hash,
    read_bound_json,
)


def _require(condition, message):
    if not condition:
        raise ValueError("STRATEGY_ADOPTION_" + message)


_COST_FIELDS = ("model_calls", "input_tokens", "output_tokens", "duration_ms", "human_technical_interventions")


def _planned_pairs(protocol):
    rows = protocol.get("planned_pairs")
    _require(isinstance(rows, list) and rows, "EXACT_PLANNED_PAIRS_REQUIRED")
    planned = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {
            "task_id", "task_family", "attempt", "source_ref", "exposure_ref", "current_context"}, "PLANNED_PAIR_FIELDS_INVALID")
        _require(isinstance(row["task_id"], str) and row["task_id"] and type(row["attempt"]) is int
            and row["attempt"] > 0 and row["task_family"] in protocol["task_families"], "PLANNED_PAIR_IDENTITY_INVALID")
        key = (row["task_id"], row["attempt"])
        _require(key not in planned, "DUPLICATE_PLANNED_PAIR")
        source, exposure = read_bound_json(row["source_ref"]), read_bound_json(row["exposure_ref"])
        _require(source.get("task_id") == row["task_id"] and source.get("task_family") == row["task_family"]
            and isinstance(source.get("source_version"), str) and source["source_version"], "TASK_SOURCE_IDENTITY_MISMATCH")
        _require(exposure.get("task_id") == row["task_id"] and exposure.get("source_ref") == row["source_ref"]
            and exposure.get("attempt") == row["attempt"] and exposure.get("exposure") in {"unseen", "exposed"}
            and (row["attempt"] == 1 or exposure["exposure"] == "exposed"), "TASK_EXPOSURE_IDENTITY_MISMATCH")
        _require(isinstance(row["current_context"], dict), "TASK_CONTEXT_REQUIRED")
        planned[key] = {**row, "exposure": exposure["exposure"]}
    return planned


def _cost(execution):
    reported = execution.get("known_cost", {})
    _require(isinstance(reported, dict), "COST_SHAPE_INVALID")
    values = {**reported, "human_technical_interventions": execution.get("human_technical_interventions")}
    _require(all(value is None or type(value) in {int, float} and math.isfinite(value) and value >= 0
        for value in values.values()), "COST_VALUE_INVALID")
    return {"values": {key: values.get(key) for key in _COST_FIELDS},
        "unknown_fields": [key for key in _COST_FIELDS if values.get(key) is None],
        "additional_reported_metrics": {key: value for key, value in reported.items() if key not in _COST_FIELDS}}


def _evaluator_evidence(verdict):
    """Check optional reference linkage, without inventing a qualified adapter.

    为什么要有它：independent 自述和文件哈希仅证明记录绑定；缺少真实runner验真接口时不能批准采用。
    """
    from src.mod_agent.evaluation.strategy_restricted_evaluator import (
        SCHEMA,
        validate_restricted_strategy_verdict,
    )

    configuration = read_bound_json(verdict["evaluator_ref"])
    from src.mod_agent.evaluation.strategy_repository_evaluator import (
        SCHEMA as REPOSITORY_SCHEMA,
    )
    from src.mod_agent.evaluation.strategy_repository_evaluator import (
        validate_repository_strategy_verdict,
    )

    if configuration.get("schema_version") == REPOSITORY_SCHEMA:
        if not verdict.get("evaluator_result_ref"):
            return {"status": "missing_original_evaluator_evidence", "missing": ["evaluator_result_ref"],
                "validation_gap": "repository_csharp_original_report_missing"}
        try:
            return validate_repository_strategy_verdict(verdict)
        except (OSError, ModelQualificationError) as error:
            if isinstance(error, ModelQualificationError) and error.status != "pending":
                raise
            return {"status": "missing_original_evaluator_evidence", "missing": ["referenced_original_unavailable"],
                "validation_gap": "repository_csharp_original_report_unavailable"}
    if configuration.get("schema_version") == SCHEMA:
        if not verdict.get("evaluator_result_ref"):
            return {"status": "missing_original_evaluator_evidence", "missing": ["evaluator_result_ref"],
                "validation_gap": "restricted_csharp_original_bundle_missing"}
        try:
            return validate_restricted_strategy_verdict(verdict)
        except (OSError, ModelQualificationError) as error:
            if isinstance(error, ModelQualificationError) and error.status != "pending":
                raise
            return {"status": "missing_original_evaluator_evidence", "missing": ["referenced_original_unavailable"],
                "validation_gap": "restricted_csharp_original_bundle_unavailable"}
    required = ("evaluator_execution_ref", "evaluator_result_ref")
    missing = [key for key in required if not verdict.get(key)]
    if missing:
        return {"status": "missing_original_evaluator_evidence", "missing": missing,
            "validation_gap": "independent_strategy_evaluator_adapter_not_registered"}
    execution = read_bound_json(verdict["evaluator_execution_ref"])
    result = read_bound_json(verdict["evaluator_result_ref"])
    _require(execution.get("evaluator_ref") == verdict["evaluator_ref"]
        and execution.get("task_execution_ref") == verdict["execution_ref"]
        and execution.get("result_ref") == verdict["evaluator_result_ref"]
        and all(execution.get(key) == verdict[key] for key in ("task_id", "candidate_source_id"))
        and all(result.get(key) == verdict[key] for key in (
            "task_id", "candidate_source_id", "task_success", "preservation_passed")), "EVALUATOR_ORIGINAL_LINK_MISMATCH")
    return {"status": "references_bound_runner_not_validated", "missing": [],
        "validation_gap": "independent_strategy_evaluator_adapter_not_registered",
        **{key: verdict[key] for key in required}}


def _evaluate(memory, strategy_ref, protocol_ref, result_refs, policy_ref):
    policy = read_bound_json(policy_ref)
    protocol = read_bound_json(protocol_ref)
    _require(policy.get("schema_version") == "strategy_adoption_policy_v1", "POLICY_REQUIRED")
    _require(protocol.get("schema_version") == "strategy_trial_protocol_v1"
        and protocol.get("batch_complete") is True, "COMPLETE_FROZEN_BATCH_REQUIRED")
    planned = _planned_pairs(protocol)
    invariants = protocol["invariants"]
    _require(set(invariants) >= INVARIANTS | {"feedback"} and all(isinstance(v, dict) and v for v in invariants.values()),
        "COMMON_CONDITIONS_REQUIRED")
    _require(protocol["invariants_sha256"] == canonical_hash(invariants), "COMMON_CONDITIONS_CHANGED")
    snapshot = memory.read(protocol["snapshot_ref"])
    # 为什么要有它：首版对照只支持单策略归因，不能把其他被选中经验的行动或收益归给目标策略。
    _require(len(snapshot.get("strategy_refs", [])) == 1, "SINGLE_STRATEGY_COMPARISON_REQUIRED")
    strategy = memory.read(strategy_ref)
    frozen = [memory.read(ref) for ref in snapshot["strategy_refs"]]
    _require(any(all(record.get(key) == strategy.get(key) for key in (
        "strategy_id", "content_revision", "strategy")) for record in frozen), "STRATEGY_NOT_IN_FROZEN_TRIAL")
    _require(policy.get("allow_limited_activation") in {True, False}
        and type(policy.get("minimum_unseen_pairs")) is int and policy["minimum_unseen_pairs"] > 0,
        "ACTIVATION_POLICY_REQUIRED")
    judge_ref = invariants["judge"]["evaluator_ref"]
    read_bound_json(judge_ref)
    gains = regressions = action_changes = unseen = 0
    seen = set()
    results = []
    for reference in result_refs:
        pair = read_bound_json(reference)
        _require(pair.get("schema_version") == "strategy_pair_result_v1" and pair["protocol_ref"] == protocol_ref,
            "INDEPENDENT_PAIR_REQUIRED")
        key = (pair["task_id"], pair["attempt"])
        _require(key not in seen, "DUPLICATE_OR_RETRIED_PAIR")
        _require(key in planned, "UNPLANNED_PAIR")
        seen.add(key)
        expected = planned[key]
        _require(all(pair.get(field) == expected[field] for field in (
            "task_family", "source_ref", "exposure_ref", "exposure", "current_context")), "FROZEN_TASK_PROVENANCE_CHANGED")
        _require(set(pair["arms"]) == {"B", "B+"}, "PAIRED_ARMS_REQUIRED")
        view = memory.advisory_model_view(memory.select(protocol["snapshot_ref"], pair["current_context"]))
        verdicts, actions, costs, evaluator_evidence, native_evidence = {}, {}, {}, {}, {}
        for arm in ("B", "B+"):
            row = pair["arms"][arm]
            execution = read_bound_json(row["execution_ref"])
            if execution.get("native_execution_ref") or execution.get("native_export_request_ref"):
                from src.mod_agent.evaluation.strategy_native_export import (
                    validate_native_strategy_execution,
                )
                native_evidence[arm] = validate_native_strategy_execution(row["execution_ref"])
            else:
                native_evidence[arm] = {"status": "native_execution_evidence_missing"}
            verdict = read_bound_json(row["evaluation_ref"])
            _require(execution.get("schema_version") == "strategy_execution_receipt_v1"
                and execution["arm"] == arm and execution["task_id"] == pair["task_id"]
                and execution["conditions_sha256"] == protocol["invariants_sha256"]
                and all(execution.get(field) == expected[field] for field in (
                    "attempt", "source_ref", "exposure_ref")), "EXECUTION_CONDITIONS_MISMATCH")
            _require(verdict.get("schema_version") == "independent_strategy_verdict_v1"
                and verdict.get("independent") is True and verdict.get("batch_complete") is True
                and verdict["task_id"] == pair["task_id"] and verdict["execution_ref"] == row["execution_ref"]
                and verdict["evaluator_ref"] == judge_ref and verdict["candidate_source_id"] == execution["candidate_source_id"],
                "INDEPENDENT_VERDICT_MISMATCH")
            _require(set(verdict) <= {"schema_version", "independent", "batch_complete", "task_id", "execution_ref",
                "evaluator_ref", "candidate_source_id", "task_success", "preservation_passed",
                "evaluator_execution_ref", "evaluator_result_ref"}, "HIDDEN_DETAILS_NOT_ALLOWED")
            _require(type(verdict["task_success"]) is bool and type(verdict["preservation_passed"]) is bool,
                "UNKNOWN_VERDICT")
            evaluator_evidence[arm] = _evaluator_evidence(verdict)
            request = read_bound_json(execution["model_request_ref"])
            delivered = [message["content"]["strategy_experience"] for message in request["messages"]
                if isinstance(message.get("content"), dict) and "strategy_experience" in message["content"]]
            if arm == "B+":
                _require(execution["experience_snapshot_ref"] == protocol["snapshot_ref"] and delivered == [view],
                    "STRATEGY_NOT_ACTUALLY_DELIVERED")
            else:
                _require(not delivered and execution.get("experience_snapshot_ref") is None, "BASELINE_CONTAMINATED")
            action = read_bound_json(execution["action_ref"])
            _require(action.get("schema_version") == "strategy_action_receipt_v1"
                and action.get("execution_origin") == "executed" and action["request_ref"] == execution["model_request_ref"],
                "ACTUAL_ACTION_REQUIRED")
            response = read_bound_json(action["response_ref"])
            selected_calls = [call for call in response["tool_calls"] if call.get("call_id") == action["call_id"]]
            _require(len(selected_calls) == 1 and {key: selected_calls[0][key] for key in ("name", "arguments")} == action["action"],
                "MODEL_ACTION_MISMATCH")
            output = read_bound_json(action["output_ref"])
            _require(output.get("call_id") == action["call_id"], "ACTION_OUTPUT_MISMATCH")
            actions[arm] = action["action"]
            costs[arm] = _cost(execution)
            success = verdict["task_success"] and verdict["preservation_passed"]
            if success:
                delivery = read_bound_json(execution["delivery_ref"])
                _require(delivery.get("status") == "delivered"
                    and delivery["candidate_source_id"] == execution["candidate_source_id"], "DELIVERY_VERSION_MISMATCH")
                delivery_check = read_bound_json(delivery["verification_ref"])
                _require(delivery_check.get("passed") is True
                    and delivery_check.get("candidate_source_id") == execution["candidate_source_id"], "DELIVERY_CHECK_MISMATCH")
            verdicts[arm] = success
        is_unseen = pair["exposure"] == "unseen" and bool(view["strategies"])
        unseen += is_unseen
        action_changes += actions["B"] != actions["B+"]
        gains += is_unseen and verdicts["B+"] and not verdicts["B"] and actions["B"] != actions["B+"]
        regressions += verdicts["B"] and not verdicts["B+"]
        if all(native_evidence[arm]["status"] == "native_execution_verified" for arm in ("B", "B+")):
            _require(native_evidence["B"]["actual_common_conditions_sha256"] == native_evidence["B+"]["actual_common_conditions_sha256"],
                "ACTUAL_COMMON_CONDITIONS_MISMATCH")
        results.append({"task_id": pair["task_id"], "attempt": pair["attempt"], "unseen": is_unseen,
            "source_ref": pair["source_ref"], "exposure_ref": pair["exposure_ref"], "reported_results": verdicts,
            "costs": costs, "evaluator_evidence": evaluator_evidence, "native_evidence": native_evidence,
            "cost_difference_Bplus_minus_B": {key: costs["B+"]["values"][key] - costs["B"]["values"][key]
                if costs["B+"]["values"][key] is not None and costs["B"]["values"][key] is not None else None
                for key in _COST_FIELDS}})
    missing = [planned[key] for key in sorted(set(planned) - seen)]
    totals = {}
    for arm in ("B", "B+"):
        totals[arm] = {}
        for field in _COST_FIELDS:
            known = [row["costs"][arm]["values"][field] for row in results if row["costs"][arm]["values"][field] is not None]
            totals[arm][field] = {"known_sum": sum(known), "known_pair_count": len(known),
                "unknown_or_missing_pair_count": len(planned) - len(known),
                "total": sum(known) if len(known) == len(planned) else None}
    # 为什么要有它：只对实际有验真适配且原件合格的范围作采用决定，其他评估器不能凭自述升级。
    verified = bool(results) and all(row["evaluator_evidence"][arm]["status"] in {"verified_restricted_csharp", "verified_repository_csharp"}
        for row in results for arm in ("B", "B+"))
    blockers = [] if verified else ["independent_strategy_evaluator_adapter_not_registered_or_evidence_missing"]
    if any(row["native_evidence"][arm]["status"] != "native_execution_verified" for row in results for arm in ("B", "B+")):
        blockers.append("native_execution_evidence_missing")
    if any(row["native_evidence"][arm].get("freeze_binding", {}).get("status") != "protocol_bound_before_model"
            for row in results for arm in ("B", "B+")):
        blockers.append("protocol_preregistration_binding_unproven")
    if any(row["native_evidence"][arm].get("planned_source_binding") != "source_and_request_verified"
            for row in results for arm in ("B", "B+")):
        blockers.append("planned_source_binding_unproven")
    if missing:
        blockers.append("planned_pair_results_missing")
    if any(row["costs"][arm]["unknown_fields"] for row in results for arm in ("B", "B+")):
        blockers.append("cost_or_assistance_unknown")
    if any(row["costs"][arm]["values"]["human_technical_interventions"] != 0 for row in results for arm in ("B", "B+")):
        blockers.append("human_assistance_not_comparable_to_autonomy")
    cost_policy = policy.get("cost_policy")
    if cost_policy != {"mode": "report_only"}:
        blockers.append("explicit_cost_adoption_policy_required")
    target = "trial"
    # 为什么要有它：停用的因果判断也须有完整执行和比较依据，不能让手写动作记录借真实评分自动停用经验。
    if not blockers and regressions:
        retirement = policy.get("retire_after_regressions")
        target = "retired" if type(retirement) is int and retirement > 0 and regressions >= retirement else "disabled"
    elif (not blockers and policy["allow_limited_activation"] and unseen >= policy["minimum_unseen_pairs"]
            and gains >= max(1, policy.get("minimum_success_gains", 1)) and action_changes):
        target = "active"
    return {"target_state": target, "scope": strategy["strategy"]["applicability"],
        "unseen_pairs": unseen, "reported_success_gains": gains, "reported_regressions": regressions,
        "action_changes": action_changes, "planned_pair_count": len(planned), "received_pair_count": len(results),
        "batch_complete_verified": not missing, "missing_pairs": missing, "activation_blockers": blockers,
        "cost_comparison": {"by_arm": totals, "difference_Bplus_minus_B": {field:
            totals["B+"][field]["total"] - totals["B"][field]["total"]
            if totals["B+"][field]["total"] is not None and totals["B"][field]["total"] is not None else None
            for field in _COST_FIELDS}},
        "results": results, "claim": ("verified_repository_csharp_paired_evidence" if any(
            row["evaluator_evidence"][arm]["status"] == "verified_repository_csharp" for row in results for arm in ("B", "B+"))
            else "verified_restricted_csharp_paired_evidence") if verified else
            "paired_receipts_without_independent_evaluator_validation", "protocol_ref": dict(protocol_ref),
        "policy_ref": dict(policy_ref), "result_refs": list(result_refs), "strategy_ref": strategy_ref}


def decide_strategy_adoption(memory, strategy_ref, *, protocol_ref, result_refs, policy_ref):
    if not isinstance(strategy_ref, dict):
        strategy_ref = asdict(strategy_ref)
    decision = _evaluate(memory, strategy_ref, protocol_ref, result_refs, policy_ref)
    return memory._write("strategy_decision_" + canonical_hash(decision), "adoption_decision", decision)


def validate_strategy_decision(memory, decision_ref):
    decision = memory.read(decision_ref)
    if decision.get("kind") == "withdrawal_decision":
        original = read_bound_json(decision["withdrawal_ref"])
        record = memory.read(decision["strategy_ref"])
        _require(original.get("schema_version") == "strategy_withdrawal_v1"
            and original.get("strategy_id") == record["strategy_id"]
            and original.get("target_state") == decision["target_state"] in {"disabled", "retired"}
            and isinstance(original.get("actor"), str) and original["actor"]
            and isinstance(original.get("reason"), str) and original["reason"], "WITHDRAWAL_ORIGIN_REQUIRED")
        return decision
    _require(decision.get("kind") == "adoption_decision", "DECISION_REQUIRED")
    actual = _evaluate(memory, decision["strategy_ref"], decision["protocol_ref"],
        decision["result_refs"], decision["policy_ref"])
    _require(all(decision.get(k) == v for k, v in actual.items()), "DECISION_CHANGED")
    return decision


def record_strategy_withdrawal(memory, strategy_ref, *, withdrawal_ref):
    """Record a host-provided withdrawal; it cannot promote any strategy.

    为什么要有它：撤回需要立即影响后续决策，不应等待一轮新评测才能停止已知不适用策略。
    """
    if not isinstance(strategy_ref, dict):
        strategy_ref = asdict(strategy_ref)
    original = read_bound_json(withdrawal_ref)
    decision = {"strategy_ref": strategy_ref, "withdrawal_ref": dict(withdrawal_ref),
        "target_state": original.get("target_state")}
    reference = memory._write("strategy_withdrawal_" + canonical_hash(decision), "withdrawal_decision", decision)
    validate_strategy_decision(memory, reference)
    return reference
