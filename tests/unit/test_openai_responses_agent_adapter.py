"""Offline Responses wire/accounting checks; no external provider requests."""
import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnProviderError,
    AgentTurnRequest,
    OpenAIResponsesAgentAdapter,
)
from src.libs.llm.transport_observation import observe_transport
from tests.unit.test_task_agent_v2_admission import session as session
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def native_response(**changes):
    return {"id": "resp_1", "model": "test-model", "status": "completed", "output": [],
        "usage": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100,
            "input_tokens_details": {"cached_tokens": 30}, "output_tokens_details": {"reasoning_tokens": 12}}, **changes}


def request(**changes):
    return AgentTurnRequest(messages=(AgentMessage("user", {"goal": "保留原文\r\n"}),), **changes)


def adapter(data, **changes):
    return OpenAIResponsesAgentAdapter(api_key="test-only", model="test-model", max_output_tokens=128,
        request_sender=lambda *_: deepcopy(data), **changes)


def test_normal_task_schema_uses_explicit_nonstrict_without_changing_host_contract(session):
    actions = session.actions()
    tools = tuple(action.definition for action in actions)
    before = deepcopy(tools)
    client = adapter(native_response(), strict_tools=False)
    payload = client._request_payload(request(tools=tools))
    assert len(payload["tools"]) == len(tools)
    for sent, original in zip(payload["tools"], tools):
        assert sent["strict"] is False and original.strict is True
        assert sent["parameters"] == original.parameters
    read = next(action for action in actions if action.definition.name == "source_read")
    assert "root_id" not in read.definition.parameters["required"]
    accepted = read.prepare({"parent_revision_ref": "baseline", "path": "A.cs", "start_line": 1})
    assert accepted.value == {"parent_revision_ref": "baseline", "path": "A.cs", "start_line": 1}
    assert tools == before
    assert adapter(native_response())._request_payload(request(tools=tools))["tools"][0]["strict"] is True


def test_actual_responses_http_body_is_retained_and_matches_preview_and_diagnostic(monkeypatch):
    bodies, retained, facts, sends = [], [], [], []
    def respond(actual):
        bodies.append(actual.content)
        return httpx.Response(200, json=native_response())
    client_type = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    client = OpenAIResponsesAgentAdapter(api_key="test-only", model="test-model", max_output_tokens=128,
        context_window=4096, timeout_seconds=7, reasoning_effort="low", strict_tools=False, store=False)
    original = request()
    preview = client._request_payload(original)
    with observe_transport(lambda: sends.append(True), lambda **value: facts.append(value),
            request_callback=lambda **value: facts.append(value), local_request_body_callback=retained.append):
        response = client.complete_turn(original)
    assert sends == [True] and retained == bodies and len(bodies) == 1
    assert json.loads(bodies[0]) == preview
    assert preview["store"] is False and preview["include"] == ["reasoning.encrypted_content"]
    assert preview["reasoning"] == {"effort": "low"} and preview["truncation"] == "disabled"
    assert response.provider_diagnostic["request_identity"] == {"provider_body_sha256": hashlib.sha256(bodies[0]).hexdigest(),
        "provider_body_utf8_bytes": len(bodies[0]), "hash_basis": "httpx_json_body_bytes"}
    assert any(row.get("provider_request_constructed") for row in facts)
    assert any(row.get("response_received") and row["http_status"] == 200 for row in facts)
    assert "Authorization" not in str(facts) and "test-only" not in str(facts)


@pytest.mark.parametrize("usage,expected", [(None, None), ({}, None),
    ({"input_tokens": 0, "output_tokens": 0}, {"input_tokens": 0, "output_tokens": 0}),
    ({"input_tokens": True, "output_tokens": 3}, None), ({"input_tokens": -1, "output_tokens": 3}, None)])
def test_responses_missing_usage_is_unknown_and_measured_zero_remains_zero(usage, expected):
    response = adapter(native_response(usage=usage)).complete_turn(request())
    assert response.provider_diagnostic["measured_usage"] == expected
    assert response.provider_diagnostic["usage_available"] is (expected is not None)


@pytest.mark.parametrize("status,reason", [("incomplete", "max_output_tokens"), ("failed", "failed"),
    ("cancelled", "cancelled"), ("queued", "incomplete"), ("in_progress", "incomplete")])
def test_stopped_responses_retain_tokens_without_parsing_or_dispatching_partial_arguments(status, reason):
    data = native_response(status=status, incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
        output=[{"type": "function_call", "call_id": "call_1", "name": "candidate_submit", "arguments": '{"edits":'}])
    response = adapter(data).complete_turn(request())
    assert response.stop_reason == reason and response.tool_calls == ()
    assert response.usage.total_tokens == 100 and response.usage.reasoning_tokens == 12
    assert response.usage.cached_input_tokens == 30
    assert response.raw_response == data and response.provider_state == data["output"]
    assert response.provider_diagnostic["measured_usage"] == {"input_tokens": 80, "output_tokens": 20,
        "cached_input_tokens": 30, "reasoning_tokens": 12}


def test_refusal_blocks_even_valid_tool_calls_and_retains_native_output():
    data = native_response(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "Cannot comply."}]},
        {"type": "function_call", "call_id": "call_1", "name": "finish", "arguments": "{}"}])
    response = adapter(data).complete_turn(request())
    assert response.stop_reason == "refusal" and not response.tool_calls
    assert response.raw_response == data and response.provider_state == data["output"]


@pytest.mark.parametrize("reason", ["new_provider_reason", {"unknown": "shape"}, None])
def test_unknown_incomplete_reason_still_stops_without_losing_native_reason(reason):
    response = adapter(native_response(status="incomplete", incomplete_details={"reason": reason},
        output=[{"type": "message", "content": [{"type": "output_text", "text": "partial text"}]}])).complete_turn(request())
    assert response.stop_reason == "incomplete" and response.tool_calls == ()
    assert response.provider_diagnostic["native_incomplete_reason"] == reason
    assert response.usage.total_tokens == 100


def test_reasoning_and_cached_usage_are_subsets_not_extra_tokens():
    response = adapter(native_response(usage={"input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
        "input_tokens_details": {"cached_tokens": 60}, "output_tokens_details": {"reasoning_tokens": 25}})).complete_turn(request())
    assert response.usage.total_tokens == 140
    assert response.usage.cached_input_tokens == 60 and response.usage.reasoning_tokens == 25


@pytest.mark.parametrize("fault", ["arguments", "no_call_id", "duplicate_call_id", "output", "item"])
def test_malformed_completed_response_keeps_original_and_measured_cost_in_error(fault):
    item = {"type": "function_call", "id": "fc_item_not_a_call_id", "call_id": "call_1", "name": "inspect", "arguments": "{}"}
    data = native_response(output=[item])
    if fault == "arguments":
        item["arguments"] = "{"
    elif fault == "no_call_id":
        item.pop("call_id")
    elif fault == "duplicate_call_id":
        data["output"].append(deepcopy(item))
    elif fault == "output":
        data["output"] = None
    else:
        data["output"] = ["invalid"]
    with pytest.raises(AgentTurnProviderError) as caught:
        adapter(data).complete_turn(request())
    diagnostic = caught.value.diagnostic
    assert diagnostic["raw_response"] == data and diagnostic["measured_usage"]["output_tokens"] == 20
    assert diagnostic["usage_available"] is True and diagnostic["response_id"] == "resp_1"
    assert diagnostic["request_identity"]["provider_body_sha256"]


def test_continuation_preserves_native_messages_reasoning_and_call_id_without_reconstruction():
    items = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-native-content"},
        {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "Inspecting."}]},
        {"type": "function_call", "id": "fc_item", "call_id": "call_1", "name": "inspect", "arguments": '{"path":"A.cs"}'}]
    client = adapter(native_response(output=items))
    response = client.complete_turn(request())
    assert response.tool_calls[0].call_id == "call_1" and response.provider_state == items
    followup = request(provider_state=response.provider_state, tool_results=(AgentToolResult("call_1", "inspect", {"ok": True}),))
    sent = client._request_payload(followup)
    assert sent["input"][1:4] == items
    assert sent["input"][-1] == {"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'}
    assert response.provider_state == items
    with pytest.raises(AgentTurnProviderError, match="CALL_ID_MISMATCH"):
        client._request_payload(request(provider_state=items, tool_results=(AgentToolResult("fc_item", "inspect", {}),)))


def test_context_rejection_occurs_without_transport_and_preserves_unknown_usage():
    client = OpenAIResponsesAgentAdapter(api_key="test-only", model="test-model", max_output_tokens=128,
        context_window=10, request_sender=lambda *_: pytest.fail("Local gate must not send"))
    with pytest.raises(AgentTurnProviderError, match="CONTEXT_BUDGET_EXCEEDED") as caught:
        client.complete_turn(request())
    assert caught.value.diagnostic["failure_boundary"] == "provider_request_gate"
    assert caught.value.diagnostic["measured_usage"] is None


def test_text_only_responses_continuation_keeps_native_items_at_each_original_boundary():
    from src.libs.llm.agent_turn import _tool_exchanges

    native = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "No tool yet."}]}]
    client = adapter(native_response())
    original = AgentTurnRequest(messages=(AgentMessage("user", "task"), AgentMessage("user", "please choose a tool")),
        historical_tool_exchanges=(AgentToolExchange((), deepcopy(native), 1),),
        provider_state=deepcopy(native), tool_result_insert_index=2)
    body = client._request_payload(original)
    assert body["input"][1:3] == native and body["input"][4:6] == native
    assert body["input"][3] == {"role": "user", "content": "please choose a tool"}
    assert not any(item.get("type") == "function_call_output" for item in body["input"])
    assert original.provider_state == native
    with pytest.raises(AgentTurnProviderError, match="TOOL_HISTORY_INVALID"):
        _tool_exchanges(original, 2)
    with pytest.raises(AgentTurnProviderError, match="CONTINUATION_WITHOUT_OUTPUTS"):
        client._request_payload(request(provider_state=[{"type": "function_call", "call_id": "unanswered"}]))


def test_explicit_proxy_is_per_client_and_does_not_read_environment_or_follow_redirects(monkeypatch):
    captured = []
    class Client:
        def __init__(self, **kwargs):
            captured.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def post(self, *_args, **_kwargs):
            return httpx.Response(200, json=native_response())
    monkeypatch.setattr(httpx, "Client", Client)
    client = OpenAIResponsesAgentAdapter(api_key="test-only", model="test-model", max_output_tokens=128,
        proxy_url="http://127.0.0.1:10808", timeout_seconds=11)
    client.complete_turn(request())
    assert captured[0]["proxy"] == "http://127.0.0.1:10808"
    assert captured[0]["trust_env"] is False and captured[0]["follow_redirects"] is False
    assert captured[0]["timeout"] == 11
    assert len(captured[0]["event_hooks"]["request"]) == 1
    with pytest.raises(ValueError, match="proxy URL"):
        adapter(native_response(), proxy_url="http://user:secret@127.0.0.1:10808")


@pytest.mark.parametrize("fault", ["missing_result", "duplicate_result", "extra_result", "wrong_name"])
def test_native_function_call_results_must_close_each_call_exactly_once(fault):
    items = [{"type": "function_call", "call_id": key, "name": "inspect", "arguments": "{}"} for key in ("call_1", "call_2")]
    results = [AgentToolResult(key, "inspect", {"ok": True}) for key in ("call_1", "call_2")]
    if fault == "missing_result":
        results.pop()
    elif fault == "duplicate_result":
        results.append(results[0])
    elif fault == "extra_result":
        results.append(AgentToolResult("unrequested", "inspect", {}))
    else:
        results[1] = AgentToolResult("call_2", "other_tool", {})
    with pytest.raises(AgentTurnProviderError, match="CALL_ID_MISMATCH"):
        adapter(native_response())._request_payload(request(provider_state=items, tool_results=tuple(results)))


@pytest.mark.parametrize("body", [b"test-only invalid JSON" + b"x" * 9000, b'["test-only", "not an object"]'])
def test_http_success_with_invalid_protocol_retains_bounded_redacted_body_and_exact_identity(monkeypatch, body):
    client_type = httpx.Client
    def respond(_request):
        return httpx.Response(200, content=body, headers={"x-request-id": "req-http-1"})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    client = OpenAIResponsesAgentAdapter(api_key="test-only", model="test-model", max_output_tokens=128)
    with pytest.raises(AgentTurnProviderError) as caught:
        client.complete_turn(request())
    info = caught.value.diagnostic
    assert info["http_status"] == 200 and info["response_received"] is True
    assert info["provider_http_request_id"] == "req-http-1"
    assert info["raw_response_body_sha256"] == hashlib.sha256(body).hexdigest()
    assert info["raw_response_body_bytes"] == len(body)
    assert len(info["response_body_preview"].encode()) <= 8192
    assert "test-only" not in str(info) and "test-only" not in str(caught.value)
    assert "[redacted]" in info["response_body_preview"]
    assert info["response_body_preview_truncated"] is (len(body) > 8192)
    assert info["measured_usage"] is None and info["usage_available"] is False
    assert info["request_identity"]["provider_body_sha256"]


@pytest.mark.parametrize("invalid", ["cached", "reasoning", "both"])
def test_invalid_usage_subcounts_do_not_erase_valid_native_totals(invalid):
    data = native_response()
    if invalid in {"cached", "both"}:
        data["usage"]["input_tokens_details"]["cached_tokens"] = 81
    if invalid in {"reasoning", "both"}:
        data["usage"]["output_tokens_details"]["reasoning_tokens"] = 21
    response = adapter(data).complete_turn(request())
    measured = response.provider_diagnostic["measured_usage"]
    assert measured["input_tokens"] == 80 and measured["output_tokens"] == 20
    assert response.usage.total_tokens == 100 and response.raw_response == data
    assert ("cached_input_tokens" not in measured) is (invalid in {"cached", "both"})
    assert ("reasoning_tokens" not in measured) is (invalid in {"reasoning", "both"})


def test_native_text_exchange_replaces_only_the_exact_duplicate_host_assistant_message():
    native = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []},
        {"type": "message", "role": "assistant", "id": "msg_1", "content": [{"type": "output_text", "text": "Choose a tool."}]}]
    original = AgentTurnRequest(messages=(AgentMessage("user", "task"), AgentMessage("assistant", "Choose a tool."),
        AgentMessage("user", "please continue")), historical_tool_exchanges=(AgentToolExchange((), deepcopy(native), 1),))
    before = deepcopy(original)
    body = adapter(native_response())._request_payload(original)
    assert body["input"] == [{"role": "user", "content": "task"}, *native, {"role": "user", "content": "please continue"}]
    assert original == before
    changed = AgentTurnRequest(messages=(AgentMessage("user", "task"), AgentMessage("assistant", "Different text")),
        historical_tool_exchanges=original.historical_tool_exchanges)
    with pytest.raises(AgentTurnProviderError, match="NATIVE_ASSISTANT_TEXT_MISMATCH"):
        adapter(native_response())._request_payload(changed)
