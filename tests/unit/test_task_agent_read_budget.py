"""Real read-budget views and charged atomic shrink-batch feedback; no real model/Build."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.libs.llm.agent_turn import AgentTurnProviderError, AgentTurnResponse
from src.mod_agent.task_agent_loop import TaskReadBudgetError, read_budget_view
from tests.unit.test_task_agent_loop import Budget, Client, Owners, call, response, run
from tests.unit.test_task_agent_tools import setup as setup


class AtomicReadBudget(Budget):
    def charge(self, *, reads, observation_bytes):
        if self.bytes + observation_bytes > self.max_bytes:
            raise ValueError("observation exhausted")
        if self.reads + reads > self.max_reads:
            raise TaskReadBudgetError(requested=reads, used=self.reads, limit=self.max_reads)
        super().charge(reads=reads, observation_bytes=observation_bytes)


class ReadOwners(Owners):
    def prepare(self, role, arguments):
        prepared = super().prepare(role, arguments)
        return replace(prepared, read_cost=1) if role == "source_search" else prepared


def owner(*, maximum=20, max_bytes=100_000, used=4):
    selected = ReadOwners(AtomicReadBudget(maximum=maximum, max_reads=6, max_bytes=max_bytes))
    selected.budget.reads = used
    selected.fixed, selected.report = True, {"candidate_id": "c0", "passed": True, "report_id": "existing-check"}
    return selected


def searches():
    return tuple(call("source_search", {"query": query}, "search-" + str(index)) for index, query in enumerate(
        ("Receive logic", "Trigger logic", "ISignalTransport IsHost")))


def test_three_requested_two_remaining_returns_atomic_feedback_then_same_two_ids_execute_once():
    selected = owner()
    batch = searches()
    client = Client([response(*batch), response(*batch[:2]), response(call("finish", {"candidate_id": "c0"}))])
    result = run(client, selected)
    assert result.status == "completed", result.termination
    assert selected.budget.reads == 6 and result.read_units == 2 and result.tool_argument_correction_used
    assert [role for role, _ in selected.effects] == ["source_search", "source_search", "finish"]
    feedback = client.requests[1].tool_results
    assert [row.call_id for row in feedback] == [item.call_id for item in batch]
    assert all(row.output["status"] == "rejected_not_executed" and row.output["read_cost"] == 0
        and row.output["requested_reads"] == 3 and row.output["read_budget"] == read_budget_view(used=4, limit=6) for row in feedback)
    assert not any(event.get("event") == "tool_dispatch_prepared" and event.get("turn") == 1 for event in result.events)


@pytest.mark.parametrize("kind", ["again", "no_model_calls", "no_feedback_bytes", "already_corrected"])
def test_read_correction_repeats_only_within_existing_resource_limits(kind):
    selected = owner(maximum=1 if kind == "no_model_calls" else 3 if kind in {"again", "already_corrected"} else 20,
        max_bytes=0 if kind == "no_feedback_bytes" else 100_000)
    client = Client([response(*searches()) for _ in range(3 if kind in {"again", "already_corrected"} else 1)])
    result = run(client, selected, tool_argument_correction_used=kind == "already_corrected")
    assert result.termination["code"] == {"again": "TASK_AGENT_MODEL_BUDGET_EXHAUSTED",
        "already_corrected": "TASK_AGENT_MODEL_BUDGET_EXHAUSTED",
        "no_model_calls": "TASK_AGENT_MODEL_BUDGET_EXHAUSTED", "no_feedback_bytes": "TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED"}[kind]
    assert selected.effects == [] and selected.budget.reads == 4
    assert len(client.requests) == (3 if kind in {"again", "already_corrected"} else 1)
    assert bool(result.unadmitted_tool_results) is (kind == "no_feedback_bytes")


def test_zero_read_remainder_can_only_switch_to_another_permitted_action():
    selected = owner(used=6)
    client = Client([response(searches()[0]), response(call("finish", {"candidate_id": "c0"}))])
    result = run(client, selected)
    assert result.status == "completed" and selected.effects == [("finish", {"candidate_id": "c0"})]
    assert client.requests[1].tool_results[0].output["read_budget"]["remaining"] == 0
    assert selected.budget.reads == 6


def test_read_correction_survives_history_resume_without_reset_or_replay():
    selected = owner()
    stopped = run(Client([response(*searches()), AgentTurnProviderError("controlled pause")]), selected)
    assert stopped.tool_argument_correction_used and selected.budget.reads == 4
    corrected = run(Client([response(*searches()[:2]), response(call("finish", {"candidate_id": "c0"}))]), selected,
        initial_request=stopped.request, prior_call_records=stopped.call_records)
    assert corrected.status == "completed" and corrected.tool_argument_correction_used
    assert selected.budget.reads == 6 and selected.budget.consumed == 4


@pytest.mark.parametrize("earlier", ["argument", "mixed", "read"])
def test_all_safe_corrections_continue_across_resume_without_reset(earlier):
    selected = owner()
    first = response(call("source_read", {}, "bad-arg")) if earlier == "argument" else (
        response(searches()[0], call("upstream", {}, "upstream")) if earlier == "mixed" else response(*searches()))
    stopped = run(Client([first, AgentTurnProviderError("controlled pause")]), selected)
    assert stopped.tool_argument_correction_used
    next_error = response(call("source_read", {}, "another-arg")) if earlier == "read" else response(*searches())
    result = run(Client([next_error, AgentTurnProviderError("pause after another correction")]), selected,
        initial_request=stopped.request, prior_call_records=stopped.call_records)
    assert result.termination["code"] == "TASK_AGENT_PROVIDER_TURN_FAILED"
    assert selected.effects == [] and selected.budget.reads == 4


def test_safe_argument_then_read_batch_corrections_use_the_original_model_budget():
    selected = owner(maximum=3)
    batch = searches()
    client = Client([
        response(call("source_read", {}, "missing-binding")),
        response(*batch),
        response(*batch),
    ])

    result = run(client, selected)

    assert result.termination["code"] == "TASK_AGENT_MODEL_BUDGET_EXHAUSTED"
    assert result.model_turns == result.model_budget["consumed"] == 3
    assert selected.effects == [] and selected.budget.reads == 4
    assert client.requests[1].tool_results[0].output["code"] == "TASK_AGENT_SOURCE_ARGUMENTS_INVALID"
    assert all(item.output["code"] == "TASK_AGENT_READ_BUDGET_EXCEEDED"
        for item in client.requests[2].tool_results)


@pytest.mark.parametrize("fault", ["outside", "duplicate", "identity", "version_after_feedback", "unknown_charge"])
def test_budget_feedback_does_not_override_identity_scope_version_or_unknown_failures(fault):
    selected = owner()
    batch, options = list(searches()), {}
    if fault == "outside":
        batch[0] = call("source_read", {"binding_id": "outside"}, "outside")
    elif fault == "duplicate":
        batch[1] = replace(batch[1], call_id=batch[0].call_id)
    elif fault == "identity":
        def reject(_response):
            raise ValueError("changed provider identity")
        options["response_admission"] = reject
    elif fault == "version_after_feedback":
        stopped = run(Client([response(*batch), AgentTurnProviderError("controlled pause")]), selected)
        original = selected.prepare
        selected.prepare = lambda role, arguments: replace(original(role, arguments), version_identity="changed-reference")
        options.update(initial_request=stopped.request, prior_call_records=stopped.call_records)
        batch = batch[:2]
    else:
        def unknown(**_kwargs):
            raise RuntimeError("unknown charge state")
        selected.budget.charge = unknown
    result = run(Client([response(*batch)]), selected, **options)
    assert result.termination["code"] in {"TASK_AGENT_BATCH_NOT_ADMITTED", "TASK_AGENT_RESPONSE_NOT_ADMITTED", "TASK_AGENT_READ_BUDGET_EXHAUSTED"}
    assert selected.effects == [] and selected.budget.reads == 4


def test_executed_read_id_in_oversized_batch_remains_hard_rejected():
    selected = owner(used=3)
    batch = searches()
    client = Client([response(batch[0]), response(*batch, call("source_search", {"query": "extra"}, "extra"))])
    result = run(client, selected)
    assert result.termination["code"] == "TASK_AGENT_READ_BUDGET_EXHAUSTED"
    assert selected.budget.reads == 4 and len(selected.effects) == 1
    assert result.call_records[batch[0].call_id]["status"] == "completed" and not result.tool_argument_correction_used


def test_session_returns_current_aggregate_charged_budget_in_every_read_result(setup):
    _inputs, session, _evaluator, budget = setup
    budget.max_reads = 6
    session.policy = replace(session.policy, max_tool_rounds=6)
    session.read_budget_snapshot = lambda: read_budget_view(used=budget.reads, limit=budget.max_reads)
    requests = []

    class Scripted:
        def complete_turn(self, request):
            requests.append(request)
            if len(requests) == 1:
                return response(*(call("source_search", {"query": query}, "q" + str(index)) for index, query in enumerate(
                    ("ChangeMarker", "KeepExistingFeedback", "RecordRuntimeMarker", "Configuration"))))
            if len(requests) == 2:
                return response(call("source_read", {"binding_id": session.inspector.available_read_binding_ids[0]}, "read"))
            return AgentTurnResponse("scripted", "controlled", "stop observing")

    result = session.run(client=Scripted())
    assert result.termination["code"] == "TASK_AGENT_FINISH_TOOL_REQUIRED" and budget.reads == 5
    assert session.model_context()["read_budget"] == read_budget_view(used=5, limit=6)
    assert all(row.output["value"]["read_budget"] == read_budget_view(used=4, limit=6) for row in requests[1].tool_results)
    assert requests[2].tool_results[0].output["value"]["read_budget"] == read_budget_view(used=5, limit=6)


@pytest.mark.parametrize("value", [{"used": 4, "limit": 6, "remaining": 3, "count_unit": "one_source_read_or_source_search_call"},
    {"used": 0, "limit": 100, "remaining": 100, "count_unit": "one_source_read_or_source_search_call"},
    {"used": True, "limit": 6, "remaining": 5, "count_unit": "one_source_read_or_source_search_call"}])
def test_session_rejects_invalid_or_inflated_budget_view(setup, value):
    _inputs, session, _evaluator, _budget = setup
    session.policy = replace(session.policy, max_tool_rounds=6)
    session.read_budget_snapshot = lambda: value
    with pytest.raises(ValueError):
        session.initial_request()
