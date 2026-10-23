"""Derived facts must not turn unknown or duplicated evidence into achievements."""
from copy import deepcopy

from src.mod_agent.evaluation.benchmark_outcome_ledger import project_native_outcome


def ref(name):
    return {"path": "/evidence/" + name, "sha256": name.encode().hex().ljust(64, "0")[:64]}


def fact(name, **value):
    return {"ref": ref(name), "value": value}


def turn(number=1, input_tokens=100, output_tokens=10):
    return {"request_ref": ref("request" + str(number)), "response_ref": ref("response" + str(number)),
        "reservation_id": str(number), "response_observed": True, "send_attempt_observed": True,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cached_input_tokens": 60}}


def inputs():
    return {"report": {"run_id": "run1", "executions": [{"turns": [turn()]}], "complete": True,
        "terminal": {"status": "unknown"}, "artifacts": []},
        "native_facts": [fact("native", run_id="run1", request_id="r1", task_contract_ref="contract1",
            budget_after={"consumed": 1}, candidate_ref="candidate1")], "context_facts": []}


def test_pure_no_rewrite_and_unknown_is_not_zero():
    source = inputs()
    previous = deepcopy(source)
    outcome = project_native_outcome(source)
    assert source == previous
    assert outcome["result"]["correct_delivery"] is None
    assert outcome["interventions"]["human_active_seconds"] is None
    assert outcome["identity"]["registered_task_denominator"] is None
    assert outcome["costs"]["task_runtime"]["total_tokens"] == 110
    assert outcome["costs"]["per_successful_delivery"]["value"] is None


def test_recovery_deduplicates_actual_requests_not_just_response_contents():
    source = inputs()
    source["report"]["executions"].append({"turns": [turn(), turn(2)]})
    source["native_facts"].append(fact("recovered", run_id="run1", task_contract_ref="contract1", budget_after={"consumed": 2}))
    cost = project_native_outcome(source)["costs"]["task_runtime"]
    assert cost["observed_responses"] == 2
    assert cost["input_tokens"]["known_sum"] == 200
    assert cost["total_tokens"] == 220  # Cache is included in input, not added.
    assert cost["reserved_budget_calls"] == 2


def test_response_conflict_stays_unknown_after_later_duplicate():
    source = inputs()
    conflict = {**turn(), "response_ref": ref("different")}
    source["report"]["executions"][0]["turns"] += [conflict, turn()]
    cost = project_native_outcome(source)["costs"]["task_runtime"]
    assert cost["input_tokens"]["known_sum"] == 0
    assert cost["input_tokens"]["unknown_calls"] == 1


def test_native_summary_prior_cost_and_unverified_learning_are_not_usage():
    source = inputs()
    source["native_facts"][0]["value"]["provider_turns"] = [{"usage": {"input_tokens": 99999}}]
    source["context_facts"] = [fact("started", prior_known_tokens=99999), fact("learning",
        original_execution_ref=ref("native"), model_calls=1, budget_after={"consumed": 2}, call_receipts=[{"response_ref": ref("learnresp"),
            "usage": {"input_tokens": 500, "output_tokens": 20, "cached_input_tokens": 0}}])]
    cost = project_native_outcome(source)["costs"]["task_runtime"]
    assert cost["total_tokens"] == 110
    assert cost["unknown_learning_calls"] == 1


def test_bound_learning_actual_response_is_counted_once():
    source = inputs()
    usage = {"input_tokens": 500, "output_tokens": 20, "cached_input_tokens": 0}
    source["context_facts"] = [fact("learning", original_execution_ref=ref("native"), model_calls=1, budget_after={"consumed": 2},
        call_receipts=[{"response_ref": ref("learnresp"), "usage": usage}] ),
        fact("learnresp", data={"response": {"usage": usage}})]
    source["context_facts"].append(deepcopy(source["context_facts"][0]))
    cost = project_native_outcome(source)["costs"]["task_runtime"]
    assert cost["total_tokens"] == 630
    assert cost["known_learning_responses"] == 1


def test_explicit_correction_overrides_old_aggregate_without_reexecution():
    source = inputs()
    old = fact("old", independent_acceptance_passed=False, preservation_suite_passed=False, candidate_source_id="source1")
    corrected = fact("correction", independent_acceptance_passed=False, preservation_suite_passed=True,
        full_selected_suite_passed=False, candidate_source_id="source1", supersedes_aggregate_ref=old["ref"],
        native_execution_ref=ref("native"),
        delivered=False, model_calls=0, total_results=220, outcomes={"Failed": 5})
    source["context_facts"] = [old, corrected]
    outcome = project_native_outcome(source)
    assert len(outcome["result"]["independent_acceptance"]) == 1
    assert outcome["result"]["independent_acceptance"][0]["preservation_suite_passed"] is True
    assert outcome["result"]["correct_delivery"] is False
    assert outcome["costs"]["per_successful_delivery"]["reason"] == "zero_confirmed_success_no_finite_estimate"


def test_delivery_requires_reader_validation_and_same_accepted_source():
    source = inputs()
    source["context_facts"] = [fact("acceptance", independent_acceptance_passed=True, preservation_suite_passed=True,
        candidate_source_id="source1", tests_executed=True, native_execution_ref=ref("native")), fact("delivery", delivered=True,
        candidate_source_id="source1", delivered_artifact_refs=[ref("dll")])]
    assert project_native_outcome(source)["result"]["correct_delivery"] is None
    source["context_facts"][1]["verification"] = {"status": "verified_delivery", "value": {
        "native_execution_ref": ref("native"),
        "status": "delivered", "candidate_source_id": "different", "delivered_artifact_refs": [ref("dll")],
        "verification": {"passed": True, "candidate_source_id": "different", "native_verification_ref": ref("build")}}}
    assert project_native_outcome(source)["result"]["correct_delivery"] is None
    check = source["context_facts"][1]["verification"]["value"]
    check["candidate_source_id"] = check["verification"]["candidate_source_id"] = "source1"
    assert project_native_outcome(source)["result"]["correct_delivery"] is None
    source["context_facts"][0]["verification"] = {"status": "verified_independent_acceptance",
        "native_execution_ref": ref("native"), "candidate_source_id": "source1", "judge_ref": ref("independent_judge")}
    assert project_native_outcome(source)["result"]["correct_delivery"] is True


def test_candidate_resubmission_and_plan_update_are_not_repairs():
    source = inputs()
    source["report"]["artifacts"] = [{"schema": "task-agent-candidate.v2", "candidate_source_id": "same"}] * 2
    source["report"]["executions"][0]["actions"] = [{"name": "plan_update"}]
    feedback = project_native_outcome(source)["feedback_revision"]
    assert feedback["distinct_candidate_sources"] == 1
    assert feedback["effective_repairs"] is None


def test_human_interval_union_and_outer_codex_not_human_minutes():
    source = inputs()
    def event(event_id, start, finish, actor="human"):
        return {"event_id": event_id, "person_id": "p1", "actor": actor, "kind": "technical_hint",
            "cost_scope": "task_runtime",
            "timing_basis": "measured", "started_at": "2026-10-23T10:" + start + "+00:00",
            "finished_at": "2026-10-23T10:" + finish + "+00:00"}
    source["context_facts"] = [fact("interventions", interventions=[event("a", "00:00", "02:00"),
        event("b", "01:00", "03:00"), event("c", "00:00", "50:00", "outer_codex")])]
    outcome = project_native_outcome(source)["interventions"]
    assert outcome["human_active_seconds"] == 180
    assert outcome["observed_event_count"] == 3
    assert outcome["technical_assistance_observed"] is True
    assert outcome["total_technical_assistance"] is None


def test_declared_overlaps_not_invented_as_complete_human_time():
    source = inputs()
    source["context_facts"] = [fact("declared", interventions=[{"event_id": str(i), "person_id": "p1",
        "actor": "human", "kind": "review", "timing_basis": "declared", "active_seconds": 60} for i in range(2)])]
    outcome = project_native_outcome(source)["interventions"]
    assert outcome["human_active_seconds"] is None
    assert "declared_time_overlap_unknown" in outcome["unknown"]


def test_process_time_bound_to_run_deduplicated_and_never_human_time():
    source = inputs()
    process = {"argv": ["--execution-config", "/inputs/run1/execution.json"], "process_id": 4,
        "wall_seconds": 60, "started_at": "2026-10-23T10:00:00+00:00", "finished_at": "2026-10-23T10:01:00+00:00"}
    source["context_facts"] = [fact("process", **process), fact("result", process=process)]
    outcome = project_native_outcome(source)
    assert outcome["costs"]["task_runtime"]["known_process_wall_seconds"] == 60
    assert outcome["interventions"]["human_active_seconds"] is None


def test_actual_trace_declarations_are_consumed_and_cost_scopes_stay_separate():
    from src.core.trace.execution_trace import ExecutionTrace

    trace = ExecutionTrace("trace", "session", {})
    with trace.activate():
        trace.record_intervention(event_id="review1", run_id="run1", actor="human", kind="review",
            cost_scope="task_runtime", subject_ref=ref("native"), person_id="p1", active_seconds=60,
            timing_basis="declared")
        trace.record_intervention(event_id="rescue1", run_id="run1", actor="outer_codex", kind="technical_hint",
            cost_scope="platform_development", subject_ref=ref("native"))
    source = inputs()
    source["context_facts"] = [{"ref": ref("trace"), "value": trace.context.to_dict()}]
    output = project_native_outcome(source)["interventions"]
    assert output["observed_event_count"] == 2  # Trace start/finish snapshots deduplicate by event ID.
    assert output["human_active_seconds"] is None
    assert output["by_cost_scope"]["task_runtime"]["human_active_seconds"] == 60
    assert output["by_cost_scope"]["platform_development"]["human_active_seconds"] is None


def test_declared_active_time_does_not_use_entire_wait_interval():
    source = inputs()
    source["context_facts"] = [fact("human", interventions=[{"event_id": "a", "person_id": "p",
        "actor": "human", "kind": "review", "cost_scope": "task_runtime", "timing_basis": "declared",
        "active_seconds": 60, "started_at": "2026-10-23T10:00:00+00:00", "finished_at": "2026-10-23T11:00:00+00:00"}])]
    output = project_native_outcome(source)["interventions"]
    assert output["human_active_seconds"] == 60
    assert output["measured_interval_union_seconds"] is None


def test_recovery_boundaries_remain_separate_from_original_terminal():
    source = inputs()
    source["context_facts"] = [fact("boundary", schema="run_outcome_boundary_v1", run_id="run1",
        event="resume_return", original_terminal_ref=ref("terminal"), result={"status": "blocked", "whole_task_delivered": False})]
    result = project_native_outcome(source)["result"]
    assert result["terminal"]["status"] == "unknown"
    assert result["boundaries"][0]["result"]["status"] == "blocked"
    assert result["correct_delivery"] is False


def test_parent_non_delivery_does_not_set_child_outcome_and_reservations_do_not_merge_runs():
    source = inputs()
    source["native_facts"].append(fact("parent", run_id="parent", task_contract_ref="contract1", budget_after={"consumed": 3}))
    source["context_facts"] = [fact("parent_result", native_execution_ref=ref("parent"), delivered=False)]
    output = project_native_outcome(source)
    assert output["result"]["correct_delivery"] is None
    assert output["costs"]["task_runtime"]["reserved_budget_calls"] == 4
    assert output["costs"]["task_runtime"]["included_run_ids"] == ["parent", "run1"]


def test_unknown_learning_calls_across_runs_add_but_cumulative_receipts_do_not():
    source = inputs()
    source["native_facts"].append(fact("parent", run_id="parent", task_contract_ref="contract1"))
    source["context_facts"] = [fact("learn1", original_execution_ref=ref("native"), model_calls=2,
        budget_after={"consumed": 3}, unknown_usage_calls=2), fact("learnparent", original_execution_ref=ref("parent"),
        model_calls=3, budget_after={"consumed": 3}, unknown_usage_calls=3)]
    source["context_facts"].append(fact("learn1copy", **source["context_facts"][0]["value"]))
    output = project_native_outcome(source)["costs"]["task_runtime"]
    assert output["unknown_learning_calls"] == 5


def test_native_time_observations_not_added_to_process_or_human_time():
    source = inputs()
    source["context_facts"] = [fact("clock", schema="task-agent-time-observation.v2",
        native_execution_ref=ref("native"), origin_ref=ref("origin"), duration_ms=3000,
        clock_basis="monotonic", event="product_return", whole_task_finished=True)]
    output = project_native_outcome(source)
    cost = output["costs"]["task_runtime"]
    assert cost["time_observations"][0]["observed_elapsed_ms"] == 3000
    assert cost["time_observations"][0]["whole_task_finished"] is False
    assert cost["known_process_wall_seconds"] is None
    assert output["interventions"]["human_active_seconds"] is None


def test_recovery_segments_do_not_increase_attempt_denominator():
    source = inputs()
    source["native_facts"].append(fact("resume", run_id="run1", request_id="r1", task_contract_ref="contract1"))
    identity = project_native_outcome(source)["identity"]
    assert len(identity["attempts"]) == 1
    assert len(identity["native_segments"]) == 2
    assert len(identity["attempts"][0]["native_execution_refs"]) == 2
    assert identity["registered_task_denominator"] is None


def test_native_explicit_no_delivery_does_not_convert_unknown_execution_to_failure():
    source = inputs()
    source["native_facts"][0]["value"].update(status="unknown", whole_task_delivered=False)
    result = project_native_outcome(source)["result"]
    assert result["correct_delivery"] is False
    assert result["terminal"]["status"] == "unknown"
