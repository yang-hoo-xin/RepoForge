"""Candidate reads reuse the existing read role and its actual budget; no model calls."""
from dataclasses import replace

from tests.unit.test_task_agent_loop import Client, call, response, run
from tests.unit.test_task_agent_read_budget import owner


def candidate_actions(selected):
    actions = selected.actions()
    source = next(action for action in actions if action.definition.name == "source_read")
    return [*actions, replace(source, definition=replace(source.definition, name="candidate_read"))]


def test_mixed_feedback_names_registered_candidate_read_and_charges_once():
    selected = owner()
    read = call("candidate_read", {"binding_id": "one"}, "candidate-read")
    client = Client([response(read, call("upstream", {}, "unused-upstream")), response(read),
        response(call("finish", {"candidate_id": "c0"}))])
    result = run(client, selected, actions=candidate_actions(selected))
    assert result.status == "completed"
    assert "candidate_read" in client.requests[1].tool_results[0].output["message"]
    assert selected.budget.reads == 5 and result.read_units == 1
    assert [role for role, _ in selected.effects] == ["source_read", "finish"]
    assert result.tool_argument_correction_used


def test_over_budget_candidate_reads_use_existing_atomic_refusal_and_feedback():
    selected = owner()
    reads = [call("candidate_read", {"binding_id": value}, "candidate-" + value) for value in ("one", "two")]
    client = Client([response(*reads, call("source_search", {"query": "excess"}, "excess")), response(*reads),
        response(call("finish", {"candidate_id": "c0"}))])
    result = run(client, selected, actions=candidate_actions(selected))
    assert result.status == "completed"
    feedback = client.requests[1].tool_results
    assert all(row.output["status"] == "rejected_not_executed" and row.output["read_budget"]["remaining"] == 2
        and "candidate_read" in row.output["message"] for row in feedback)
    assert selected.budget.reads == 6 and result.read_units == 2
    assert [role for role, _ in selected.effects] == ["source_read", "source_read", "finish"]
