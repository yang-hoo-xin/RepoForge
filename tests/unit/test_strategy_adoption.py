"""Adoption consumes original frozen paired evidence, not a success summary."""
from dataclasses import asdict

import pytest

from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.evaluation.memory_comparison import INVARIANTS
from src.mod_agent.evaluation.strategy_adoption import (
    decide_strategy_adoption,
    record_strategy_withdrawal,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from tests.unit import test_strategy_experience as experience_fixtures

manual = experience_fixtures.manual


@pytest.fixture
def setup(tmp_path):
    return experience_fixtures.setup.__wrapped__(tmp_path)


def pair(tmp_path, setup, *, outcomes=(False, True), minimum_pairs=2, same_action=False,
         omit_delivery=False, context=None, attempt=1, missing_pair=False, costs=None, evaluator_originals=False,
         qualified_evaluator=None):
    memory, _, _, _ = setup
    _, snapshot = manual(setup)
    strategy_ref = memory.read(snapshot)["strategy_refs"][0]
    policy = _save(tmp_path / "adoption-policy.json", {"schema_version": "strategy_adoption_policy_v1",
        "allow_limited_activation": True, "minimum_unseen_pairs": minimum_pairs, "minimum_success_gains": 1,
        "cost_policy": {"mode": "report_only"}})
    judge = qualified_evaluator["evaluator_ref"] if qualified_evaluator else _save(
        tmp_path / "judge.json", {"source": "controlled independent assertion checker", "version": 1})
    invariants = {key: {"fixture_identity": key} for key in INVARIANTS | {"feedback"}}
    invariants["judge"] = {"evaluator_ref": judge}
    conditions = canonical_hash(invariants)
    current = context if context is not None else {"mechanism": "retained_state"}
    source = _save(tmp_path / "task-source.json", {"task_id": "unseen-1", "task_family": "retained-reader",
        "source_version": "frozen-source-1"})
    exposure = _save(tmp_path / "task-exposure.json", {"task_id": "unseen-1", "attempt": 1,
        "source_ref": source, "exposure": "unseen"})
    planned = [{"task_id": "unseen-1", "task_family": "retained-reader", "attempt": 1,
        "source_ref": source, "exposure_ref": exposure, "current_context": current}]
    if missing_pair:
        second_source = _save(tmp_path / "task-source-2.json", {"task_id": "unseen-2", "task_family": "retained-reader",
            "source_version": "frozen-source-2"})
        second_exposure = _save(tmp_path / "task-exposure-2.json", {"task_id": "unseen-2", "attempt": 1,
            "source_ref": second_source, "exposure": "unseen"})
        planned.append({**planned[0], "task_id": "unseen-2", "source_ref": second_source, "exposure_ref": second_exposure})
    protocol = _save(tmp_path / "protocol.json", {"schema_version": "strategy_trial_protocol_v1",
        "batch_complete": True, "snapshot_ref": asdict(snapshot), "invariants": invariants,
        "invariants_sha256": conditions, "task_families": ["retained-reader"], "planned_pairs": planned})
    view = memory.advisory_model_view(memory.select(snapshot, current))
    arms = {}
    for arm, success in zip(("B", "B+"), outcomes):
        folder = tmp_path / arm.replace("+", "plus")
        request = _save(folder / "request.json", {"messages": [{"role": "user", "content":
            {"strategy_experience": view} if arm == "B+" and not omit_delivery else {"task": "repair"}}]})
        call = {"call_id": "call-1", "name": "source_read", "arguments": {"target": "same" if same_action else arm}}
        response = _save(folder / "response.json", {"tool_calls": [call]})
        output = _save(folder / "tool-output.json", {"call_id": "call-1", "observed": "public observation"})
        action = _save(folder / "action.json", {"schema_version": "strategy_action_receipt_v1",
            "execution_origin": "executed", "request_ref": request, "output_ref": output,
            "response_ref": response, "call_id": "call-1",
            "action": {"name": "source_read", "arguments": {"target": "same" if same_action else arm}}})
        selected_candidate = qualified_evaluator["candidates"][arm] if qualified_evaluator else None
        candidate = selected_candidate["candidate_source_id"] if selected_candidate else canonical_hash({"arm": arm, "code": "candidate"})
        verification = _save(folder / "verification.json", {"passed": success, "candidate_source_id": candidate})
        delivery = _save(folder / "delivery.json", {"status": "delivered" if success else "not_delivered",
            "candidate_source_id": candidate, "verification_ref": verification})
        execution = _save(folder / "execution.json", {"schema_version": "strategy_execution_receipt_v1",
            "arm": arm, "task_id": "unseen-1", "conditions_sha256": conditions, "candidate_source_id": candidate,
            "attempt": attempt, "source_ref": source, "exposure_ref": exposure,
            "model_request_ref": request, "action_ref": action, "delivery_ref": delivery,
            "experience_snapshot_ref": asdict(snapshot) if arm == "B+" else None,
            **({"candidate_state": selected_candidate["candidate_state"]} if selected_candidate else {}),
            "human_technical_interventions": 0, "known_cost": costs[arm] if costs is not None else {
                "model_calls": 2, "input_tokens": 70, "output_tokens": 30, "duration_ms": 1000}})
        originals = {}
        if evaluator_originals:
            raw_result = _save(folder / "evaluator-result.json", {"task_id": "unseen-1", "candidate_source_id": candidate,
                "task_success": success, "preservation_passed": True})
            raw_execution = _save(folder / "evaluator-execution.json", {"task_id": "unseen-1", "candidate_source_id": candidate,
                "evaluator_ref": judge, "task_execution_ref": execution, "result_ref": raw_result})
            originals = {"evaluator_execution_ref": raw_execution, "evaluator_result_ref": raw_result}
        if selected_candidate:
            from src.mod_agent.evaluation.strategy_restricted_evaluator import (
                generate_restricted_strategy_verdict,
            )
            verdict = generate_restricted_strategy_verdict(evaluator_ref=judge, execution_ref=execution,
                evidence_bundle_ref=selected_candidate["bundle_ref"], output_path=folder / "independent-verdict.json")
        else:
            verdict = _save(folder / "independent-verdict.json", {"schema_version": "independent_strategy_verdict_v1",
                "independent": True, "batch_complete": True, "task_id": "unseen-1", "execution_ref": execution,
                "evaluator_ref": judge, "candidate_source_id": candidate, "task_success": success, "preservation_passed": True,
                **originals})
        arms[arm] = {"execution_ref": execution, "evaluation_ref": verdict}
    result = _save(tmp_path / "paired-result.json", {"schema_version": "strategy_pair_result_v1",
        "protocol_ref": protocol, "task_id": "unseen-1", "task_family": "retained-reader", "exposure": "unseen",
        "attempt": attempt, "source_ref": source, "exposure_ref": exposure, "current_context": current, "arms": arms})
    return memory, strategy_ref, snapshot, {"protocol_ref": protocol, "result_refs": [result], "policy_ref": policy}


def test_small_positive_batch_remains_trial(tmp_path, setup):
    memory, strategy, snapshot, args = pair(tmp_path, setup)
    decision = decide_strategy_adoption(memory, strategy, **args)
    assert memory.read(decision)["target_state"] == "trial"
    revised = memory.transition(strategy, decision)
    assert memory.read(revised)["state"] == "trial"
    assert memory.check_snapshot(snapshot)
    with pytest.raises(ValueError, match="REVISION_CONFLICT"):
        memory.transition(strategy, decision)


def test_reported_negative_transfer_is_not_certified_without_evaluator_adapter(tmp_path, setup):
    memory, strategy, snapshot, args = pair(tmp_path, setup, outcomes=(True, False))
    decision = decide_strategy_adoption(memory, strategy, **args)
    revised = memory.transition(strategy, decision)
    assert memory.read(revised)["state"] == "trial"
    assert memory.read(decision)["reported_regressions"] == 1
    assert memory.check_snapshot(snapshot)


def test_self_declared_independence_cannot_activate_even_if_scores_improve(tmp_path, setup):
    memory, strategy, _, args = pair(tmp_path, setup, minimum_pairs=1)
    decision = decide_strategy_adoption(memory, strategy, **args)
    assert memory.read(decision)["target_state"] == "trial"
    assert "independent_strategy_evaluator_adapter_not_registered_or_evidence_missing" in memory.read(decision)["activation_blockers"]
    assert memory.read(memory.transition(strategy, decision))["state"] == "trial"


@pytest.mark.parametrize("kwargs,code", [({"omit_delivery": True}, "STRATEGY_NOT_ACTUALLY_DELIVERED"),
    ({"attempt": 2}, "UNPLANNED_PAIR")])
def test_delivery_and_exposure_receipts_required(tmp_path, setup, kwargs, code):
    memory, strategy, _, args = pair(tmp_path, setup, **kwargs)
    with pytest.raises(ValueError, match=code):
        decide_strategy_adoption(memory, strategy, **args)


@pytest.mark.parametrize("kwargs", [{"same_action": True}, {"context": {"mechanism": "different"}}])
def test_improved_outcome_without_applicable_changed_action_does_not_activate(tmp_path, setup, kwargs):
    memory, strategy, _, args = pair(tmp_path, setup, minimum_pairs=1, **kwargs)
    decision = decide_strategy_adoption(memory, strategy, **args)
    assert memory.read(decision)["target_state"] == "trial"


def test_summary_cannot_replace_original_independent_result(tmp_path, setup):
    memory, strategy, _, args = pair(tmp_path, setup)
    args["result_refs"] = [_save(tmp_path / "summary.json", {"successes": 99, "model_says_better": True})]
    with pytest.raises(ValueError, match="INDEPENDENT_PAIR_REQUIRED"):
        decide_strategy_adoption(memory, strategy, **args)


def test_explicit_withdrawal_invalidates_existing_advisory_before_execution(tmp_path, setup):
    memory, _, _, _ = setup
    _, snapshot = manual(setup)
    selection = memory.select(snapshot, {"mechanism": "retained_state"})
    reference = memory.current_strategy("retained-reader")
    withdrawal = _save(tmp_path / "withdrawal.json", {"schema_version": "strategy_withdrawal_v1",
        "actor": "user", "reason": "Observed conflicting evidence", "strategy_id": "retained-reader", "target_state": "disabled"})
    decision = record_strategy_withdrawal(memory, reference, withdrawal_ref=withdrawal)
    memory.transition(reference, decision)
    with pytest.raises(ValueError, match="SNAPSHOT_REVOKED_OR_REVISED"):
        memory.advisory_model_view(selection)


def test_winning_subset_does_not_complete_frozen_batch(tmp_path, setup):
    memory, strategy, _, args = pair(tmp_path, setup, minimum_pairs=1, missing_pair=True)
    decision = memory.read(decide_strategy_adoption(memory, strategy, **args))
    assert decision["target_state"] == "trial" and decision["batch_complete_verified"] is False
    assert decision["planned_pair_count"] == 2 and decision["received_pair_count"] == 1
    assert decision["missing_pairs"][0]["task_id"] == "unseen-2"
    assert "planned_pair_results_missing" in decision["activation_blockers"]
    assert decision["cost_comparison"]["by_arm"]["B+"]["model_calls"] == {
        "known_sum": 2, "known_pair_count": 1, "unknown_or_missing_pair_count": 1, "total": None}


def test_no_results_preserves_all_missing_pairs_and_unknown_totals(tmp_path, setup):
    memory, strategy, _, args = pair(tmp_path, setup)
    args["result_refs"] = []
    decision = memory.read(decide_strategy_adoption(memory, strategy, **args))
    assert decision["target_state"] == "trial" and len(decision["missing_pairs"]) == 1
    assert decision["cost_comparison"]["difference_Bplus_minus_B"]["model_calls"] is None


@pytest.mark.parametrize("field,value", [("exposure", "exposed"), ("source_ref", {}),
    ("exposure_ref", {}), ("current_context", {"mechanism": "invented"})])
def test_result_cannot_relabel_frozen_source_exposure_or_context(tmp_path, setup, field, value):
    memory, strategy, _, args = pair(tmp_path, setup)
    modified = {**read_bound_json(args["result_refs"][0]), field: value}
    args["result_refs"] = [_save(tmp_path / "modified-result.json", modified)]
    with pytest.raises(ValueError, match="FROZEN_TASK_PROVENANCE_CHANGED"):
        decide_strategy_adoption(memory, strategy, **args)


def test_unplanned_task_and_duplicate_pair_are_rejected(tmp_path, setup):
    memory, strategy, _, args = pair(tmp_path, setup)
    duplicate = {**args, "result_refs": args["result_refs"] * 2}
    with pytest.raises(ValueError, match="DUPLICATE_OR_RETRIED_PAIR"):
        decide_strategy_adoption(memory, strategy, **duplicate)
    modified = {**read_bound_json(args["result_refs"][0]), "task_id": "unplanned-winner"}
    args["result_refs"] = [_save(tmp_path / "unplanned-result.json", modified)]
    with pytest.raises(ValueError, match="UNPLANNED_PAIR"):
        decide_strategy_adoption(memory, strategy, **args)


@pytest.mark.parametrize("value", [-1, True, "unknown"])
def test_invalid_cost_does_not_become_adoption_evidence(tmp_path, setup, value):
    memory, strategy, _, args = pair(tmp_path, setup, costs={"B": {"model_calls": 1}, "B+": {"model_calls": value}})
    with pytest.raises(ValueError, match="COST_VALUE_INVALID"):
        decide_strategy_adoption(memory, strategy, **args)


def test_missing_cost_fields_are_unknown_not_zero(tmp_path, setup):
    memory, strategy, _, args = pair(tmp_path, setup, costs={"B": {"model_calls": 2}, "B+": {"model_calls": 3}})
    decision = memory.read(decide_strategy_adoption(memory, strategy, **args))
    assert "cost_or_assistance_unknown" in decision["activation_blockers"]
    assert decision["cost_comparison"]["difference_Bplus_minus_B"]["model_calls"] == 1
    assert decision["cost_comparison"]["difference_Bplus_minus_B"]["input_tokens"] is None
    assert decision["results"][0]["costs"]["B"]["unknown_fields"] == ["input_tokens", "output_tokens", "duration_ms"]


def test_hash_bound_evaluator_metadata_is_not_a_real_runner_adapter(tmp_path, setup):
    memory, strategy, _, args = pair(tmp_path, setup, minimum_pairs=1, evaluator_originals=True)
    decision = memory.read(decide_strategy_adoption(memory, strategy, **args))
    assert decision["target_state"] == "trial"
    assert decision["results"][0]["evaluator_evidence"]["B+"]["status"] == "references_bound_runner_not_validated"


def test_multistrategy_snapshot_cannot_attribute_another_strategy_to_target(tmp_path, setup):
    memory, target, snapshot, args = pair(tmp_path, setup)
    _, evidence, strategy, _ = setup
    other = memory.record_strategy("other-strategy", {**strategy, "applicability": {"mechanism": "other"}},
        evidence_refs=evidence, origin={"kind": "human", "actor": "controlled-test"})
    other_policy = _save(tmp_path / "other-trial-policy.json", {"schema_version": "strategy_trial_policy_v1",
        "allowed_origin_kinds": ["human"], "allowed_domains": {"mechanism": ["other"]}})
    other_snapshot = memory.admit_trial(other, policy_ref=other_policy)
    combined = memory._write("combined-strategy-fixture", "snapshot", {**memory.read(snapshot),
        "strategy_refs": [target, memory.read(other_snapshot)["strategy_refs"][0]]})
    protocol = read_bound_json(args["protocol_ref"])
    args["protocol_ref"] = _save(tmp_path / "multistrategy-protocol.json", {**protocol, "snapshot_ref": asdict(combined)})
    with pytest.raises(ValueError, match="SINGLE_STRATEGY_COMPARISON_REQUIRED"):
        decide_strategy_adoption(memory, target, **args)
