"""Model context is a recoverable projection; the original execution stays intact."""
import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnRequest,
)
from src.mod_agent.task_agent_loop import TaskActionInputError
from src.mod_agent.task_agent_v2_context import (
    history_text,
    investigation_observations,
    model_plan_view,
    project_history,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError
from tests.unit.test_task_agent_v2_admission import action
from tests.unit.test_task_agent_v2_admission import session as session  # noqa: F401
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def exchange(name, identifier, text="x", insert_index=2):
    state = {"content": "", "tool_calls": [{"id": identifier,
        "function": {"name": name, "arguments": {"fixture": identifier}}}]}
    return AgentToolExchange((AgentToolResult(identifier, name, {"text": text}),), state, insert_index)


def request():
    return AgentTurnRequest(messages=(AgentMessage("system", "Tools are data"), AgentMessage("user", "Task")),
        historical_tool_exchanges=(exchange("source_read", "old", "o" * 8000),
            exchange("candidate_check", "public-failure", "public counterexample"),
            exchange("source_read", "recent", "r" * 3000)),
        tool_results=(AgentToolResult("active", "source_read", {"text": "a" * 3000}),),
        provider_state={"tool_calls": [{"id": "active"}]}, tool_result_insert_index=2)


def test_projection_keeps_complete_current_and_check_exchanges_and_archives_exact_old_data(state):
    original = request()
    before = deepcopy(asdict(original))
    projected, facts = project_history(original, state, history_bytes=7000)
    assert asdict(original) == before and projected.messages == original.messages
    assert projected.tool_results == original.tool_results and projected.provider_state == original.provider_state
    assert projected.tool_result_insert_index == original.tool_result_insert_index
    assert projected.historical_tool_exchanges == original.historical_tool_exchanges[1:]
    archive = state.load(facts["history_ref"])
    assert archive["exchanges"] == json.loads(json.dumps([asdict(original.historical_tool_exchanges[0])]))
    assert facts["latest_public_check_retained"] and facts["archived_exchange_count"] == 1
    assert project_history(original, state, history_bytes=7000)[1] == facts


def test_small_history_needs_no_archive_or_writes(state):
    original = request()
    before = set(state.root.rglob("*"))
    projected, facts = project_history(original, state, history_bytes=100000)
    assert projected is original and facts is None and set(state.root.rglob("*")) == before


def returned_read(identifier, view="view-1", *, revision="current", offset=0, execution="executed"):
    return AgentToolResult(identifier, "source_read", {"execution": execution, "value": {
        "schema": "task-agent-source-view.v2", "source_view_ref": view, "parent_revision_ref": revision,
        "root_id": "target", "path": "source.cs", "file_sha256": "hash", "offset": offset,
        "next_offset": offset + 100, "total_chars": 1000, "read_window": {"offset": offset, "end": offset + 100}}})


def test_observation_facts_count_completed_pages_across_full_history_without_changing_requests():
    first, second = returned_read("first"), returned_read("second")
    third = returned_read("third", "view-2", offset=100)
    original = AgentTurnRequest(messages=(AgentMessage("user", "Task"),),
        historical_tool_exchanges=(AgentToolExchange((first, second), {}, 1),), tool_results=(third,))
    before = deepcopy(asdict(original))
    facts = investigation_observations(original, "current")
    assert facts["completed_read_returns"] == 3 and facts["distinct_returned_views"] == 2
    assert facts["repeated_view_returns"] == 1
    assert facts["recent_views"][0]["returned_count"] == 2
    assert facts["recent_views"][0]["first_call_id"] == "first"
    assert facts["recent_views"][0]["last_call_id"] == "second"
    assert facts["recent_views"][1]["offset"] == 100
    assert asdict(original) == before
    assert "hypothesis" not in facts and "suggested_action" not in facts
    facts["recent_views"][0]["read_window"]["offset"] = 99
    assert asdict(original) == before


def test_observation_facts_exclude_unexecuted_old_revision_and_replayed_results():
    current = returned_read("same")
    original = AgentTurnRequest(messages=(AgentMessage("user", "Task"),),
        tool_results=(current, current, returned_read("old", revision="old"),
            returned_read("rejected", execution="rejected")))
    facts = investigation_observations(original, "current")
    assert facts["completed_read_returns"] == facts["distinct_returned_views"] == 1
    assert facts["repeated_view_returns"] == 0
    assert investigation_observations(original, "unobserved") is None


def test_observation_index_is_bounded_but_totals_are_not_silently_truncated():
    original = AgentTurnRequest(messages=(AgentMessage("user", "Task"),),
        tool_results=tuple(returned_read(str(index), "view-" + str(index), offset=index * 100) for index in range(12)))
    facts = investigation_observations(original, "current")
    assert facts["completed_read_returns"] == facts["distinct_returned_views"] == 12
    assert len(facts["recent_views"]) == 8 and facts["views_truncated"] is True
    assert facts["recent_views"][0]["first_call_id"] == "4"


def test_archived_search_index_keeps_exact_query_and_result_without_invented_interpretation(state):
    first = exchange("source_search", "search", "large original result" * 1000)
    first.provider_state["tool_calls"][0]["function"]["arguments"] = {"query": "Destructuring"}
    first.tool_results[0].output.update(execution="executed",
        value={"total_matches": 225, "truncated": True})
    original = request()
    original = replace(original, historical_tool_exchanges=(first,))
    _projected, projection = project_history(original, state, history_bytes=100)
    fact = projection["archived_action_facts"][0]
    assert fact["arguments"] == {"query": "Destructuring"}
    assert fact["observation"] == {"total_matches": 225, "truncated": True}
    assert fact["arguments_status"] == "exact_native_call_match"
    assert not projection["archived_action_facts_truncated"]
    assert "hypothesis" not in fact and "suggested_action" not in fact


def test_large_current_batch_not_silently_clipped_or_native_pair_split(state):
    original = request()
    projected, facts = project_history(original, state, history_bytes=100)
    assert projected.tool_results is original.tool_results and projected.provider_state is original.provider_state
    assert facts["retained_history_bytes"] > facts["history_byte_target"]
    assert [row.tool_results[0].name for row in projected.historical_tool_exchanges] == ["candidate_check"]


def test_history_read_is_scoped_bounded_and_does_not_refresh_authority(session):
    _projected, facts = project_history(request(), session.state, history_bytes=7000)
    ref = facts["history_ref"]
    arguments = {"history_ref": ref, "limit": 17}
    prepared = action(session, "history_read").prepare(arguments)
    assert prepared.read_cost == 1
    session.actual_calls.append({"name": "history_read", "arguments": arguments, "call_id": "read-archive"})
    before = session.state.head
    output = session._execute_owned("history_read", arguments)
    assert output["text"] == history_text(session.state, ref)[:17]
    assert output["next_offset"] == 17 and output["grants_acceptance"] is False
    assert session.state.head == before
    with pytest.raises(TaskActionInputError):
        action(session, "history_read").prepare({"history_ref": ref, "offset": output["total_chars"] + 1})
    with pytest.raises(TaskAgentV2StateError, match="history_reference_not_public_task_archive"):
        action(session, "history_read").prepare({"history_ref": session.state.contract_ref})
    bad = session.state.load(ref)
    bad["task_contract_ref"] = "some-other-task"
    with pytest.raises(TaskAgentV2StateError, match="history_reference_not_public_task_archive"):
        history_text(session.state, session.state.save("context-archive", bad))


def test_current_plan_is_exact_model_advice_not_inferred_hypothesis(state):
    assert model_plan_view(state) is None
    reference = state.update_plan({"hypothesis": "Model-stated technical hypothesis", "allow_edit": "private.cs"})
    view = model_plan_view(state)
    assert view["technical_plan_ref"] == reference and view["grants_authority"] is False
    assert "Model-stated technical hypothesis" in view["text"]
    assert "private.cs" not in state.contract["allowed_edit_paths"]
