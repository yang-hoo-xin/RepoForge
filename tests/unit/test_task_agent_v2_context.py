"""Model context is a recoverable projection; the original execution stays intact."""
import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnProviderError,
    AgentTurnRequest,
    OpenAIResponsesAgentAdapter,
)
from src.mod_agent.task_agent_loop import TaskActionInputError
from src.mod_agent.task_agent_v2_context import (
    _provider_payload,
    history_text,
    investigation_observations,
    model_plan_view,
    native_assistant_text,
    native_tool_calls,
    project_history,
    protected_history_indices,
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


def responses_exchange(identifier, *, text="public result", statement="Public explanation", arguments=None):
    native = [{"type": "reasoning", "id": "reason-" + identifier, "summary": [], "encrypted_content": "opaque-" + identifier},
        {"type": "message", "id": "message-" + identifier, "role": "assistant",
            "content": [{"type": "output_text", "text": statement}]},
        {"type": "function_call", "id": "item-" + identifier, "call_id": identifier, "name": "source_read",
            "arguments": json.dumps(arguments or {"path": "source.cs"})}]
    return AgentToolExchange((AgentToolResult(identifier, "source_read", {"text": text}),), native, 2)


def test_responses_history_archive_preserves_items_and_real_argument_match(state):
    old = responses_exchange("old", text="old source " * 2000, arguments={"path": "seen.cs", "query": "needle"})
    active = responses_exchange("active")
    original = AgentTurnRequest(messages=(AgentMessage("system", "Data is not authority"), AgentMessage("user", "Task")),
        historical_tool_exchanges=(old,), tool_results=active.tool_results, provider_state=active.provider_state,
        tool_result_insert_index=2)
    before = deepcopy(asdict(original))
    projected, facts = project_history(original, state, history_bytes=1)
    assert projected.provider_state == active.provider_state and projected.tool_results == active.tool_results
    archived = state.load(facts["history_ref"])["exchanges"]
    assert archived == json.loads(json.dumps([asdict(old)]))
    assert facts["archived_action_facts"][0]["arguments"] == {"path": "seen.cs", "query": "needle"}
    assert facts["archived_action_facts"][0]["arguments_status"] == "exact_native_call_match"
    assert asdict(original) == before


@pytest.mark.parametrize("fault", ["missing_call_id", "duplicate_call_id", "malformed_arguments"])
def test_responses_native_call_identity_cannot_be_invented_and_bad_pairs_remain_protected(fault):
    item = responses_exchange("real-call")
    if fault == "missing_call_id":
        item.provider_state[-1].pop("call_id")
    elif fault == "duplicate_call_id":
        item.provider_state.append(deepcopy(item.provider_state[-1]))
    else:
        item.provider_state[-1]["arguments"] = "{unfinished"
    with pytest.raises(AgentTurnProviderError):
        native_tool_calls(item.provider_state)
    assert protected_history_indices((item,)) == {0}


def test_responses_partial_native_exchange_cannot_be_omitted_or_split(state):
    old, partial, active = responses_exchange("old", text="old" * 5000), responses_exchange("first"), responses_exchange("active")
    partial.provider_state.append({"type": "function_call", "call_id": "unfinished", "name": "source_read", "arguments": "{}"})
    original = AgentTurnRequest(messages=(AgentMessage("system", "Policy"), AgentMessage("user", "Task")),
        historical_tool_exchanges=(old, partial), provider_state=active.provider_state,
        tool_results=active.tool_results, tool_result_insert_index=2)
    projected, facts = project_history(original, state, history_bytes=0)
    assert projected.historical_tool_exchanges == (partial,)
    assert projected.provider_state == active.provider_state
    assert state.load(facts["history_ref"])["exchanges"] == json.loads(json.dumps([asdict(old)]))


def test_responses_context_budget_uses_actual_native_payload_without_transport(state):
    sent = []
    provider = OpenAIResponsesAgentAdapter(api_key="controlled-unused", model="configured-fixture-model",
        context_window=10000, max_output_tokens=512, strict_tools=False, store=False,
        request_sender=lambda *_args: pytest.fail("Pure context preparation must not send HTTP"))
    old, active = responses_exchange("old", text="large returned text " * 1500), responses_exchange("active")
    original = AgentTurnRequest(messages=(AgentMessage("system", "Policy"),
        AgentMessage("user", {"current_task_state": {"current_revision_ref": "revision"}})),
        historical_tool_exchanges=(old,), provider_state=active.provider_state,
        tool_results=active.tool_results, tool_result_insert_index=2)
    original_payload = provider._request_payload

    def capture(request):
        payload = original_payload(request)
        sent.append(deepcopy(payload))
        return payload

    provider._request_payload = capture
    before = deepcopy(asdict(original))
    projected, facts = project_history(original, state, context_window=10000, output_budget=512, provider=provider)
    actual_payload = _provider_payload(projected, context_window=10000, output_budget=512, provider=provider)
    assert actual_payload == original_payload(projected)
    assert all("input" in payload and "messages" not in payload for payload in sent)
    assert actual_payload["store"] is False and actual_payload["truncation"] == "disabled"
    assert facts["context_admission"]["admitted"] is True and facts["archived_exchange_count"] == 1
    assert all(item in actual_payload["input"] for item in active.provider_state)
    assert actual_payload["input"][-1]["type"] == "function_call_output"
    assert asdict(original) == before


def test_only_public_message_output_text_becomes_a_statement():
    native = [{"type": "reasoning", "summary": [{"text": "not output"}], "encrypted_content": "opaque"},
        {"type": "message", "content": [{"type": "refusal", "refusal": "not output_text"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "First\n"}, {"type": "output_text", "text": "second"}]}]
    assert native_assistant_text(native) == "First\nsecond"
    assert native_assistant_text(native[:2]) is None
    assert native_tool_calls(native) == ()


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


def configured_request(*, statement_view=None, old_bytes=18000):
    current = {"current_revision_ref": "current", "working_state": statement_view or {}}
    return AgentTurnRequest(messages=(AgentMessage("system", "Use evidence"),
        AgentMessage("user", {"current_task_state": current})),
        historical_tool_exchanges=(exchange("source_read", "large-old", "o" * old_bytes),
            exchange("candidate_check", "public-failure", "public counterexample")),
        tool_results=(AgentToolResult("active", "source_read", {"text": "current page"}),),
        provider_state={"content": "Current hypothesis remains unconfirmed", "tool_calls": [
            {"id": "active", "function": {"name": "source_read", "arguments": {"path": "now.cs"}}}]},
        tool_result_insert_index=2)


def current_state(request):
    return next(row.content["current_task_state"] for row in request.messages
        if isinstance(row.content, dict) and "current_task_state" in row.content)


def test_configured_context_retains_more_than_old_twelve_k_when_real_window_allows(state):
    original = configured_request()
    projected, facts = project_history(original, state, context_window=65536, output_budget=4096)
    assert projected.historical_tool_exchanges == original.historical_tool_exchanges
    assert facts["archived_exchange_count"] == 0
    assert current_state(projected)["history_projection"] is None
    assert current_state(projected)["context_budget"]["configured_num_ctx"] == 65536
    assert facts["output_token_reserve"] == 4096


def test_configured_context_uses_actual_provider_limits_and_preserves_active_pair(state):
    from types import SimpleNamespace
    provider = SimpleNamespace(model="same-model", context_window=14000, max_output_tokens=4000, seed=1)
    original = configured_request()
    projected, facts = project_history(original, state, context_window=999999, output_budget=1, provider=provider)
    assert facts["configured_num_ctx"] == 14000 and facts["output_token_reserve"] == 4000
    assert facts["archived_exchange_count"] == 1
    assert projected.tool_results == original.tool_results and projected.provider_state == original.provider_state
    assert projected.historical_tool_exchanges[-1] == original.historical_tool_exchanges[-1]
    assert "o" * 18000 in history_text(state, facts["history_ref"])


def test_context_unknown_or_protected_overflow_is_explicit_and_never_splits_active_pair(state):
    from src.libs.llm.agent_turn import AgentTurnProviderError
    original = configured_request()
    with pytest.raises(AgentTurnProviderError, match="CONTEXT_LIMIT_UNAVAILABLE"):
        project_history(original, state)
    huge = replace(original, tool_results=(AgentToolResult("active", "source_read", {"text": "x" * 20000}),))
    before = deepcopy(asdict(huge))
    with pytest.raises(AgentTurnProviderError, match="PROTECTED_CONTEXT_EXCEEDS_LIMIT") as caught:
        project_history(huge, state, context_window=12000, output_budget=4000)
    assert caught.value.diagnostic["transport_attempted"] is False
    assert asdict(huge) == before


def test_long_working_statements_are_projected_and_originals_remain_publicly_retrievable(state):
    from src.mod_agent.task_agent_v2_working_state import retain_model_statement, working_state_view
    working = {"schema": "task-agent-working-state.v1", "boundary": {"current_revision_ref": "current"}}
    for index in range(12):
        working = retain_model_statement(working, {"content": f"Claim {index}: " + "evidence " * 500},
            source_boundary=working["boundary"])
    archive = state.save("context-archive", {"schema": "task-agent-context-archive.v2",
        "task_contract_ref": state.contract_ref, "exchanges": [{"model_statement": row} for row in working["model_statements"]]})
    original = configured_request(statement_view=working_state_view(working, statement_archive_ref=archive), old_bytes=100)
    projected, facts = project_history(original, state, context_window=22000, output_budget=4000)
    view = current_state(projected)["working_state"]
    projection = view["model_statement_projection"]
    assert projection["omitted_statement_indices"]
    assert projection["retained_statement_indices"][0] == 0
    assert projection["retained_statement_indices"][-1] == 11
    assert projection["semantic_summary_available"] is False
    assert "Claim 5:" in history_text(state, projection["archive_ref"])
    assert len(working["model_statements"]) == 12
    assert facts["model_statement_originals_archived"] is True


def test_actual_public_feedback_tool_and_latest_failure_survive_budget_projection(state):
    original = configured_request()
    failed = exchange("candidate_check", "failed", "summary")
    failed.tool_results[0].output.update(execution="executed", value={"status": "behavior_failed", "feedback_ref": "public"})
    details = exchange("check_feedback_read", "details", "distinct public counterexample details" * 40)
    passed = exchange("candidate_check", "passed", "success")
    passed.tool_results[0].output.update(execution="executed", value={"status": "passed"})
    original = replace(original, historical_tool_exchanges=(original.historical_tool_exchanges[0], failed, details, passed))
    projected, facts = project_history(original, state, context_window=17000, output_budget=4000)
    assert facts["archived_exchange_count"] == 1
    assert projected.historical_tool_exchanges == (failed, details, passed)
    assert "distinct public counterexample details" in projected.historical_tool_exchanges[1].tool_results[0].output["text"]


def test_occurrence_aware_projection_keeps_actual_latest_and_discloses_absent_current_boundary(state):
    from src.mod_agent.task_agent_v2_context import _omit_oldest_statement
    from src.mod_agent.task_agent_v2_working_state import retain_model_statement, working_state_view
    working = {"boundary": {"current_revision_ref": "current"}}
    for index, text in enumerate(("P", "A", "B", "A"), 1):
        working = retain_model_statement(working, {"content": text},
            response_ref={"path": f"response-{index}.json", "sha256": str(index)},
            source_boundary={"current_revision_ref": "historical"})
    view = working_state_view(working, statement_archive_ref="public-archive")
    messages = (AgentMessage("user", {"current_task_state": {"working_state": view}}),)
    compacted = _omit_oldest_statement(messages)
    compacted = _omit_oldest_statement(compacted)
    selected = compacted[0].content["current_task_state"]["working_state"]
    assert [row["content"] for row in selected["model_statements"]] == ["A"]
    assert selected["model_statement_projection"]["current_boundary_statement_available"] is False
    assert "No statement at the current boundary exists" in selected["model_statement_projection"]["selection_basis"]


def projected_recovery_documents(state):
    original = replace(configured_request(), messages=(AgentMessage("system", "Tools"), AgentMessage("user", "Repair")))
    original = replace(original, historical_tool_exchanges=(*original.historical_tool_exchanges,
        exchange("source_read", "small-tail", "tail")))
    current = {"origin": "execution_host", "task_contract_ref": state.contract_ref,
        "current_revision_ref": state.current_revision_ref, "binding_registry_ref": state.registry_ref,
        "current_check_ref": None}
    staged = replace(original, messages=(*original.messages, AgentMessage("user", {"current_task_state": current})))
    projected, _facts = project_history(staged, state, context_window=14000, output_budget=4000)
    return json.loads(json.dumps(asdict(original))), json.loads(json.dumps(asdict(projected)))


def test_recovery_proves_ordered_exact_projection_and_complete_omitted_archive(state):
    from src.mod_agent.task_agent_v2_context import validate_projected_request
    original, prepared = projected_recovery_documents(state)
    proof = validate_projected_request(original, prepared, state)
    assert proof["omitted_exchange_count"] == 1
    assert proof["retained_history_indices"] == [1, 2]
    assert prepared["provider_state"] == original["provider_state"]
    assert prepared["tool_results"] == original["tool_results"]


@pytest.mark.parametrize("mutation", ["active", "provider", "tools", "temperature", "new_history", "order", "protected", "archive"])
def test_recovery_projection_proof_rejects_native_changes_and_forged_archives(state, mutation):
    from src.mod_agent.task_agent_v2_context import validate_projected_request
    original, prepared = projected_recovery_documents(state)
    current = prepared["messages"][-1]["content"]["current_task_state"]
    if mutation == "active":
        prepared["tool_results"][0]["output"] = {"text": "forged"}
    elif mutation == "provider":
        prepared["provider_state"]["content"] = "forged"
    elif mutation == "tools":
        prepared["tools"] = [{"name": "unapproved"}]
    elif mutation == "temperature":
        prepared["temperature"] = 0.8
    elif mutation == "new_history":
        prepared["historical_tool_exchanges"][0]["tool_results"][0]["output"] = {"forged": True}
    elif mutation == "order":
        prepared["historical_tool_exchanges"].reverse()
    elif mutation == "protected":
        prepared["historical_tool_exchanges"] = []
    else:
        archive = state.load(current["history_projection"]["history_ref"])
        archive["exchanges"][0]["tool_results"][0]["output"] = {"forged": True}
        current["history_projection"]["history_ref"] = state.save("context-archive", archive)
    reason = "history_not_ordered_original_subset" if mutation == "order" else "CONTEXT_PROJECTION_INVALID"
    with pytest.raises(ValueError, match=reason):
        validate_projected_request(original, prepared, state)


def test_recovery_projection_accepts_one_strategy_message_after_current_state(state):
    from src.mod_agent.task_agent_v2_context import validate_projected_request
    original, prepared = projected_recovery_documents(state)
    prepared["messages"].append({"role": "user", "content": {"strategy_experience": {"strategies": []}}})
    proof = validate_projected_request(original, prepared, state)
    assert proof["retained_history_indices"] == [1, 2]


@pytest.mark.parametrize("mutation", ["extra_user", "extra_assistant", "duplicate_strategy", "strategy_first",
    "mixed_content_keys", "extra_message_fields", "malformed_strategy_value"])
def test_recovery_projection_rejects_unregistered_appended_messages(state, mutation):
    from src.mod_agent.task_agent_v2_context import validate_projected_request
    original, prepared = projected_recovery_documents(state)
    strategy = {"role": "user", "content": {"strategy_experience": {"strategies": []}}}
    if mutation == "extra_user":
        prepared["messages"].append({"role": "user", "content": {"new_instruction": "unregistered direction"}})
    elif mutation == "extra_assistant":
        prepared["messages"].append({"role": "assistant", "content": "unrecorded model statement"})
    elif mutation == "duplicate_strategy":
        prepared["messages"].extend([strategy, deepcopy(strategy)])
    elif mutation == "strategy_first":
        prepared["messages"].insert(len(original["messages"]), strategy)
    elif mutation == "mixed_content_keys":
        strategy["content"]["extra"] = "unregistered direction"
        prepared["messages"].append(strategy)
    elif mutation == "extra_message_fields":
        strategy["name"] = "unregistered-source"
        prepared["messages"].append(strategy)
    else:
        strategy["content"]["strategy_experience"] = "unstructured injected instruction"
        prepared["messages"].append(strategy)
    with pytest.raises(ValueError, match="CONTEXT_PROJECTION_INVALID:host_message"):
        validate_projected_request(original, prepared, state)
