"""Normal Patch loop controls; scripted provider, real Plan-bound read owner."""
import json
from dataclasses import replace

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolCall,
    AgentToolExchange,
    AgentTurnProviderError,
)
from src.mod_agent.proposal.patch_agent_loop import (
    PATCH_SOURCE_INSPECTION_TOOL_NAME,
    PatchAgentLoopError,
    PatchAgentLoopPolicy,
    PlanBoundSourceInspector,
    run_patch_agent_loop,
)
from tests.unit.test_patch_agent_loop import (
    _final_format,
    _fixture,
    _response,
    _run,
    _ScriptedClient,
    _tool_call,
)


def call(arguments, identity="invalid-1"):
    return AgentToolCall(call_id=identity, name=PATCH_SOURCE_INSPECTION_TOOL_NAME, arguments=arguments)


def no_read_policy(**kwargs):
    return PatchAgentLoopPolicy(require_source_inspection=False, finalize_after_first_observation=False, **kwargs)


def test_single_argument_error_is_feedback_then_corrected_in_the_same_loop(tmp_path):
    _, _, _, inspector = _fixture(tmp_path)
    bad = AgentToolCall(call_id="bad-1", name=PATCH_SOURCE_INSPECTION_TOOL_NAME, arguments={})
    client = _ScriptedClient([
        _response(turn=1, tool_calls=(bad,)),
        _response(turn=2, tool_calls=(_tool_call("edit_binding_001", turn=2),)),
        _response(turn=3, content='{"changes": []}'),
    ])
    result = _run(client, inspector)
    assert result.completed and result.model_turns_used == 3 and result.tool_rounds_used == 1
    feedback = client.requests[1].tool_results
    assert len(feedback) == 1 and feedback[0].call_id == "bad-1"
    assert feedback[0].output["status"] == "not_executed"
    assert feedback[0].output["error_code"] == "PATCH_AGENT_TOOL_ARGUMENTS_INVALID"
    assert feedback[0].output["argument_errors"]
    assert result.tool_call_trace[0]["dispatch_started"] is False
    assert result.tool_call_trace[0]["read_cost"] == 0
    assert result.tool_call_trace[0]["feedback_payload_bytes"] > 0
    assert result.reused_source_observations == 0
    assert result.turn_progress_trace[0]["reused_source_observations"] == 0


def test_whole_read_batch_is_rejected_before_a_corrected_batch_runs(tmp_path):
    policy = PatchAgentLoopPolicy(allow_batched_reads=True, finalize_after_first_observation=False)
    _, _, _, inspector = _fixture(tmp_path, binding_count=2, policy=policy)
    bad = AgentToolCall(call_id="bad-1", name=PATCH_SOURCE_INSPECTION_TOOL_NAME, arguments={})
    good = _tool_call("edit_binding_002", turn=1)
    client = _ScriptedClient([
        _response(turn=1, tool_calls=(bad, good)),
        _response(turn=2, tool_calls=(_tool_call("edit_binding_001", turn=2), _tool_call("edit_binding_002", turn=3))),
        _response(turn=3, content='{"changes": []}'),
    ])
    result = _run(client, inspector)
    assert result.completed and result.tool_rounds_used == 2 and result.model_turns_used == 3
    feedback = client.requests[1].tool_results
    assert [item.call_id for item in feedback] == [bad.call_id, good.call_id]
    assert all(item.output["status"] == "not_executed" for item in feedback)
    assert all(item["dispatch_started"] is False and item["read_cost"] == 0 for item in result.tool_call_trace[:2])
    assert result.dispatched_binding_reads == 2


@pytest.mark.parametrize("arguments", [{}, {"binding_id": None}, {"binding_id": False}, {"binding_id": 12},
    {"binding_id": []}, {"binding_id": ""}, {"binding_id": "  "}, [], "not-an-object",
    {"binding_id": "edit_binding_001", "comment": "不要回显这段内容"},
    {"binding_id": "edit_binding_001", "reason": "not execution authority"}])
def test_ordinary_errors_reach_native_feedback_without_inspection(tmp_path, arguments):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    client = _ScriptedClient([_response(turn=1, tool_calls=(call(arguments),)), _response(turn=2, content='{"changes":[]}')])
    result = _run(client, inspector)
    assert result.completed and result.tool_rounds_used == result.dispatched_binding_reads == 0
    feedback = client.requests[1].tool_results[0].output
    assert feedback["argument_errors"] and feedback["allowed_format"]["additionalProperties"] is False
    assert "不要回显" not in json.dumps(feedback, ensure_ascii=False)
    assert result.tool_argument_correction_used is True
    assert result.tool_call_trace[0]["feedback_queued"] is True
    assert result.tool_call_trace[0]["feedback_in_request"] is True
    assert result.tool_call_trace[0]["followup_response_observed"] is True
    assert result.tool_call_trace[0]["consumed_in_model_turn"] == 2


@pytest.mark.parametrize("field", ["path", "file", "source", "target", "source_hash", "identity", "permissions", "command", "execute", "unexpected"])
def test_non_annotation_extra_fields_remain_hard_rejected(tmp_path, field):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    client = _ScriptedClient([_response(turn=1, tool_calls=(call({"binding_id": "edit_binding_001", field: "sensitive attempt"}),))])
    result = _run(client, inspector)
    assert result.status == "blocked" and result.tool_argument_correction_used is False
    assert result.tool_rounds_used == 0 and len(client.requests) == 1
    assert result.tool_call_trace[0]["feedback_queued"] is False


@pytest.mark.parametrize("unsafe", ["unknown_binding", "duplicate_id", "missing_id", "unknown_tool", "path", "duplicate_binding"])
def test_later_unsafe_batch_item_cannot_be_hidden_by_first_missing_argument(tmp_path, unsafe):
    _, _, _, inspector = _fixture(tmp_path, binding_count=2, policy=no_read_policy(allow_batched_reads=True))
    bad, other = call({}), call({"binding_id": "edit_binding_001"}, "other")
    if unsafe == "unknown_binding":
        other = call({"binding_id": "outside_plan"}, "other")
    elif unsafe == "duplicate_id":
        other = call({"binding_id": "edit_binding_001"})
    elif unsafe == "missing_id":
        other = call({"binding_id": "edit_binding_001"}, "")
    elif unsafe == "unknown_tool":
        other = replace(other, name="arbitrary_read")
    elif unsafe == "path":
        other = call({"binding_id": "edit_binding_001", "path": "private"}, "other")
    else:
        bad = call({"binding_id": "edit_binding_001", "comment": "unused"})
    client = _ScriptedClient([_response(turn=1, tool_calls=(bad, other))])
    result = _run(client, inspector)
    assert result.status == "blocked" and not result.tool_argument_correction_used
    assert result.tool_rounds_used == result.dispatched_binding_reads == 0
    assert all(not entry["feedback_queued"] for entry in result.tool_call_trace)


def test_bad_arguments_cannot_reuse_an_executed_id_even_in_cached_mode(tmp_path):
    policy = no_read_policy(reuse_covered_source_observations=True, allow_batched_reads=True)
    _, _, _, inspector = _fixture(tmp_path, policy=policy)
    client = _ScriptedClient([_response(turn=1, tool_calls=(_tool_call("edit_binding_001", turn=1),)),
        _response(turn=2, tool_calls=(call({"binding_id": "edit_binding_001", "comment": "malformed"}, "call-001"),))])
    result = _run(client, inspector)
    assert result.termination["code"] == "PATCH_AGENT_TOOL_CALL_ID_REUSED"
    assert result.tool_rounds_used == 1 and not result.tool_argument_correction_used


def test_rejected_argument_call_id_cannot_be_reused_for_corrected_arguments(tmp_path):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    client = _ScriptedClient([_response(turn=1, tool_calls=(call({}),)),
        _response(turn=2, tool_calls=(call({"binding_id": "edit_binding_001"}),))])
    result = _run(client, inspector)
    assert result.termination["code"] == "PATCH_AGENT_TOOL_CALL_ID_REUSED"
    assert result.tool_rounds_used == 0 and result.tool_argument_correction_used


def test_second_distinct_argument_error_has_no_second_opportunity(tmp_path):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    client = _ScriptedClient([_response(turn=1, tool_calls=(call({}),)),
        _response(turn=2, tool_calls=(call({"binding_id": 5}, "new-call"),))])
    result = _run(client, inspector)
    assert result.termination["code"] == "PATCH_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED"
    assert result.total_observation_bytes == result.tool_call_trace[0]["feedback_payload_bytes"]
    assert result.tool_call_trace[1]["feedback_queued"] is False


@pytest.mark.parametrize("location", ["current", "historical"])
def test_same_plan_resume_retains_single_opportunity_and_budget_floors(tmp_path, location):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    first = _run(_ScriptedClient([_response(turn=1, tool_calls=(call({}),)),
        _response(turn=2, content='{"changes":[]}')]), inspector)
    previous = first.final_request
    if location == "historical":
        previous = replace(previous, historical_tool_exchanges=(*previous.historical_tool_exchanges,
            AgentToolExchange(previous.tool_results, previous.provider_state, previous.tool_result_insert_index)),
            tool_results=(), provider_state=None, tool_result_insert_index=None)
    client = _ScriptedClient([_response(turn=3, tool_calls=(call({}, "resume-error"),))])
    options = dict(client=client, initial_messages=previous.messages, resume_request=previous,
        final_response_format=_final_format(), source_inspector=inspector,
        consumed_model_turns=2, consumed_observation_bytes=first.total_observation_bytes)
    resumed = run_patch_agent_loop(**options)
    assert resumed.termination["code"] == "PATCH_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED"
    assert resumed.model_turns_used == 1 and resumed.total_observation_bytes == 0
    for changed in ({"consumed_model_turns": 1}, {"consumed_observation_bytes": first.total_observation_bytes - 1}):
        with pytest.raises(PatchAgentLoopError, match="PATCH_AGENT_LOOP_BUDGET_INVALID"):
            run_patch_agent_loop(**{**options, **changed, "client": _ScriptedClient([])})


def test_provider_failure_keeps_queued_feedback_and_consumed_opportunity_for_resume(tmp_path):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    def fail_second(number):
        if number == 2:
            raise AgentTurnProviderError("controlled provider failure")
    client = _ScriptedClient([_response(turn=1, tool_calls=(call({}),))], before_response=fail_second)
    first = _run(client, inspector)
    assert first.termination["code"] == "PATCH_AGENT_PROVIDER_TURN_FAILED" and first.final_request is not None
    trace = first.tool_call_trace[0]
    assert trace["feedback_queued"] and trace["feedback_request_attempted"]
    assert not trace["followup_response_observed"] and "consumed_in_model_turn" not in trace
    second = run_patch_agent_loop(client=_ScriptedClient([_response(turn=3, tool_calls=(call({}, "later"),))]),
        initial_messages=first.final_request.messages, resume_request=first.final_request,
        final_response_format=_final_format(), source_inspector=inspector,
        consumed_model_turns=first.model_turns_used, consumed_observation_bytes=first.total_observation_bytes)
    assert second.termination["code"] == "PATCH_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED"


def test_snapshot_drift_after_model_cannot_be_hidden_as_argument_error(tmp_path):
    _, source, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    client = _ScriptedClient([_response(turn=1, tool_calls=(call({}),))],
        before_response=lambda _: source.write_text("changed source", encoding="utf-8"))
    result = _run(client, inspector)
    assert result.status == "blocked" and result.termination["code"] != "PATCH_AGENT_TOOL_ARGUMENTS_INVALID"
    assert not result.tool_argument_correction_used and result.tool_rounds_used == 0


@pytest.mark.parametrize("delta", [-1, 0])
def test_utf8_feedback_is_admitted_as_a_complete_batch_with_reservation(tmp_path, delta):
    workspace, _, plan, _ = _fixture(tmp_path, policy=no_read_policy())
    plan["binding_registry"]["edit_bindings"][0]["binding_id"] = "编辑一"
    plan["proposed_changes"][0]["edit_binding_id"] = "编辑一"
    first_inspector = PlanBoundSourceInspector(workspace_root=workspace, implementation_plan=plan, policy=no_read_policy())
    first_client = _ScriptedClient([_response(turn=1, tool_calls=(call({}),)), _response(turn=2, content='{"changes":[]}')])
    first = _run(first_client, first_inspector)
    payload = first_client.requests[1].tool_results[0].output
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert len(encoded.encode()) == first.total_observation_bytes > len(encoded)
    inspector = PlanBoundSourceInspector(workspace_root=workspace, implementation_plan=plan,
        policy=no_read_policy(max_observation_bytes=first.total_observation_bytes + 20 + delta))
    client = _ScriptedClient([_response(turn=1, tool_calls=(call({}),)), _response(turn=2, content='{"changes":[]}')])
    result = run_patch_agent_loop(client=client, initial_messages=(AgentMessage("user", "controlled"),),
        final_response_format=_final_format(), source_inspector=inspector, reserve_observation_bytes=20)
    if delta < 0:
        assert result.termination["code"] == "PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED"
        assert result.total_observation_bytes == 0 and not result.tool_call_trace[0]["feedback_queued"]
    else:
        assert result.completed and result.total_observation_bytes == first.total_observation_bytes
    assert result.tool_rounds_used == 0


@pytest.mark.parametrize("kind", ["model", "read", "observation"])
def test_exhausted_or_reserved_budget_never_gets_a_new_correction(tmp_path, kind):
    policy = no_read_policy(max_total_model_turns=1) if kind == "model" else no_read_policy()
    _, _, _, inspector = _fixture(tmp_path, policy=policy)
    kwargs = {"reserve_tool_reads": 3} if kind == "read" else {"consumed_observation_bytes": policy.max_observation_bytes} if kind == "observation" else {}
    result = run_patch_agent_loop(client=_ScriptedClient([_response(turn=1, tool_calls=(call({}),))]),
        initial_messages=(AgentMessage("user", "controlled"),), final_response_format=_final_format(),
        source_inspector=inspector, **kwargs)
    assert result.status == "blocked" and not result.tool_argument_correction_used and result.tool_rounds_used == 0


def test_two_distinct_retained_correction_exchanges_are_rejected(tmp_path):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    first = _run(_ScriptedClient([_response(turn=1, tool_calls=(call({}),)), _response(turn=2, content='{"changes":[]}')]), inspector)
    prior = first.final_request
    duplicate = tuple(replace(item, call_id="second-opportunity") for item in prior.tool_results)
    prior = replace(prior, historical_tool_exchanges=(AgentToolExchange(duplicate, prior.provider_state, 0),))
    with pytest.raises(ValueError, match="ARGUMENT_CORRECTION_STATE_INVALID"):
        run_patch_agent_loop(client=_ScriptedClient([]), initial_messages=prior.messages, resume_request=prior,
            source_inspector=inspector, final_response_format=_final_format(), consumed_model_turns=2,
            consumed_observation_bytes=first.total_observation_bytes)


def test_argument_error_after_an_actual_read_delivers_new_feedback_not_old_observation(tmp_path):
    _, _, _, inspector = _fixture(tmp_path, binding_count=2,
        policy=no_read_policy(allow_batched_reads=True, max_total_model_turns=5))
    client = _ScriptedClient([_response(turn=1, tool_calls=(_tool_call("edit_binding_001", turn=1),)),
        _response(turn=2, tool_calls=(call({}),)),
        _response(turn=3, tool_calls=(_tool_call("edit_binding_002", turn=3),)),
        _response(turn=4, content='{"changes":[]}')])
    result = _run(client, inspector)
    assert result.completed and result.dispatched_binding_reads == result.tool_rounds_used == 2
    assert [item.call_id for item in client.requests[2].tool_results] == ["invalid-1"]
    assert client.requests[2].tool_results[0].output["status"] == "not_executed"
    assert result.reused_source_observations == 0


def test_failed_diagnosis_resume_preserves_actual_native_history(tmp_path):
    from src.libs.llm.agent_turn import AgentTurnRequest
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy(max_total_model_turns=5))
    first = _run(_ScriptedClient([_response(turn=1, tool_calls=(call({}),)),
        _response(turn=2, content='{"changes":[]}')]), inspector)
    def fail(_):
        raise AgentTurnProviderError("controlled diagnosis failure")
    failed = run_patch_agent_loop(client=_ScriptedClient([], before_response=fail),
        initial_messages=first.final_request.messages, resume_request=first.final_request,
        diagnosis_request=AgentTurnRequest(messages=(AgentMessage("user", "diagnose only"),)),
        final_response_format=_final_format(), source_inspector=inspector,
        consumed_model_turns=2, consumed_observation_bytes=first.total_observation_bytes)
    assert failed.tool_argument_correction_used and failed.final_request.tool_results == first.final_request.tool_results


@pytest.mark.parametrize("provider", ["ollama", "qwen", "openai"])
@pytest.mark.parametrize("count", [1, 2])
def test_argument_feedback_uses_actual_provider_single_and_batch_protocol(tmp_path, provider, count):
    from src.libs.llm import (
        OllamaChatAgentAdapter,
        OpenAIResponsesAgentAdapter,
        QwenChatAgentAdapter,
    )
    captured = []
    def send(_url, payload, _headers):
        captured.append(payload)
        calls = [{"id": f"bad-{i}", "type": "function", "function": {
            "index": i, "name": PATCH_SOURCE_INSPECTION_TOOL_NAME,
            "arguments": {} if i == 0 else {"binding_id": 5}}} for i in range(count)]
        if provider == "openai":
            return {"model": "controlled", "output": ([{"type": "function_call", "call_id": c["id"],
                "name": c["function"]["name"], "arguments": json.dumps(c["function"]["arguments"])} for c in calls] if len(captured) == 1 else
                [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": '{"changes":[]}'}]}])}
        message = {"content": "", "tool_calls": calls} if len(captured) == 1 else {"content": '{"changes":[]}'}
        if provider == "ollama":
            return {"model": "controlled", "message": message, "done": True}
        for c in calls:
            c["function"]["arguments"] = json.dumps(c["function"]["arguments"])
        return {"model": "controlled", "choices": [{"message": message, "finish_reason": "tool_calls" if len(captured) == 1 else "stop"}]}
    if provider == "ollama":
        client = OllamaChatAgentAdapter(base_url="http://127.0.0.1:11434", model="controlled", max_output_tokens=128, request_sender=send)
    elif provider == "qwen":
        client = QwenChatAgentAdapter(api_key="test", base_url="https://example.invalid/v1", model="controlled", max_completion_tokens=128, request_sender=send)
    else:
        client = OpenAIResponsesAgentAdapter(api_key="test", model="controlled", max_output_tokens=128, request_sender=send)
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy(allow_batched_reads=True))
    result = _run(client, inspector)
    assert result.completed and result.model_turns_used == 2 and result.tool_rounds_used == 0
    assert len(captured) == 2
    messages = captured[1]["input" if provider == "openai" else "messages"]
    feedback = [item for item in messages if item.get("type") == "function_call_output" or item.get("role") == "tool"]
    assert len(feedback) == count
    if provider == "ollama":
        # Scope: Ollama correlates by ordered tool_name; the existing adapter records deterministic call IDs.
        # 为什么要有它：不能为真实协议并不存在的tool_call_id字段伪造通过条件。
        assert [item["tool_name"] for item in feedback] == [PATCH_SOURCE_INSPECTION_TOOL_NAME] * count
        assert len({entry["tool_call_id"] for entry in result.tool_call_trace}) == count
    else:
        assert [item.get("call_id", item.get("tool_call_id")) for item in feedback] == [f"bad-{i}" for i in range(count)]
    for i, item in enumerate(feedback):
        output = json.loads(item.get("output", item.get("content")))
        assert output["status"] == "not_executed" and output["tool_execution_performed"] is False
        assert output["argument_errors"] == [{"field": "binding_id", "issue": "required" if i == 0 else "string_required"}]


@pytest.mark.parametrize("resume_after_provider_failure", [False, True])
def test_native_ollama_corrected_batch_reuses_unexecuted_valid_call_without_changing_id(tmp_path, resume_after_provider_failure):
    from src.libs.llm import OllamaChatAgentAdapter
    captured = []
    def raw(arguments):
        return {"model": "controlled", "done": True, "message": {"content": "", "tool_calls": [
            {"function": {"name": PATCH_SOURCE_INSPECTION_TOOL_NAME, "arguments": value}} for value in arguments]}}
    responses = [raw([{}, {"binding_id": "edit_binding_002"}]),
        raw([{"binding_id": "edit_binding_001"}, {"binding_id": "edit_binding_002"}]),
        {"model": "controlled", "done": True, "message": {"content": '{"changes":[]}'}}]
    def send(_url, payload, _headers):
        captured.append(payload)
        if resume_after_provider_failure and len(captured) == 2:
            raise AgentTurnProviderError("controlled transport failure")
        return responses.pop(0)
    client = OllamaChatAgentAdapter(base_url="http://127.0.0.1:11434", model="controlled", max_output_tokens=128, request_sender=send)
    _, _, _, inspector = _fixture(tmp_path, binding_count=2,
        policy=PatchAgentLoopPolicy(allow_batched_reads=True, finalize_after_first_observation=False, max_total_model_turns=5))
    first = _run(client, inspector)
    result = first
    if resume_after_provider_failure:
        assert first.termination["code"] == "PATCH_AGENT_PROVIDER_TURN_FAILED"
        result = run_patch_agent_loop(client=client, initial_messages=first.final_request.messages,
            resume_request=first.final_request, final_response_format=_final_format(), source_inspector=inspector,
            consumed_model_turns=first.model_turns_used, consumed_observation_bytes=first.total_observation_bytes)
    assert result.completed, result.termination
    assert result.tool_rounds_used == result.dispatched_binding_reads == 2
    initial_good_id = first.tool_call_trace[1]["tool_call_id"]
    executed_good = [entry for entry in result.tool_call_trace if entry["dispatch_started"] and entry["binding_id"] == "edit_binding_002"]
    assert len(executed_good) == 1 and executed_good[0]["tool_call_id"] == initial_good_id
    assert all(not entry["dispatch_started"] for entry in first.tool_call_trace[:2])
    assert result.reused_source_observations == 0 and not responses


@pytest.mark.parametrize("change", ["different_binding", "malformed", "different_tool", "duplicate_id", "executed_again"])
def test_unexecuted_valid_call_identity_does_not_authorize_other_or_repeated_reads(tmp_path, change):
    _, _, _, inspector = _fixture(tmp_path, binding_count=2,
        policy=no_read_policy(allow_batched_reads=True, max_total_model_turns=5))
    good = call({"binding_id": "edit_binding_002"}, "valid-unexecuted")
    second = good
    if change == "different_binding":
        second = replace(good, arguments={"binding_id": "edit_binding_001"})
    elif change == "malformed":
        second = replace(good, arguments={})
    elif change == "different_tool":
        second = replace(good, name="request_patch_upstream")
    replies = [_response(turn=1, tool_calls=(call({}), good)),
        _response(turn=2, tool_calls=(second, second) if change == "duplicate_id" else (second,))]
    if change == "executed_again":
        replies.append(_response(turn=3, tool_calls=(good,)))
    result = _run(_ScriptedClient(replies), inspector)
    assert result.status == "blocked" and result.termination["code"] == "PATCH_AGENT_TOOL_CALL_ID_REUSED"
    assert result.dispatched_binding_reads == (1 if change == "executed_again" else 0)
    assert result.reused_source_observations == 0


def test_valid_call_after_argument_feedback_preserves_existing_exact_cache_reuse(tmp_path):
    _, _, _, inspector = _fixture(tmp_path, binding_count=2,
        policy=no_read_policy(allow_batched_reads=True, reuse_covered_source_observations=True))
    good = call({"binding_id": "edit_binding_002"}, "valid-unexecuted")
    result = _run(_ScriptedClient([_response(turn=1, tool_calls=(call({}), good)),
        _response(turn=2, tool_calls=(good,)), _response(turn=3, tool_calls=(good,)),
        _response(turn=4, content='{"changes":[]}')]), inspector)
    assert result.completed and result.dispatched_binding_reads == 2 and result.tool_rounds_used == 1
    assert result.reused_source_observations == 1


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_inherited_correction_marker_requires_boolean_before_any_provider_turn(tmp_path, invalid):
    _, _, _, inspector = _fixture(tmp_path, policy=no_read_policy())
    client = _ScriptedClient([])
    with pytest.raises(ValueError, match="ARGUMENT_CORRECTION_STATE_INVALID"):
        run_patch_agent_loop(client=client, initial_messages=(AgentMessage("user", "controlled"),),
            final_response_format=_final_format(), source_inspector=inspector,
            tool_argument_correction_used=invalid)
    assert client.requests == []
