"""Native call identity and exact legacy replay; no network or build execution.

为什么要有它：调用身份区别于参数相等，兼容历史也不能给真实新调用伪造身份或额外权限。
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentTurnProviderError,
    AgentTurnRequest,
    OllamaChatAgentAdapter,
)
from tests.unit.test_task_agent_loop import Budget, Owners, run


def _wire(call_id, *, name="candidate_check", arguments=None):
    call = {"function": {"name": name, "arguments": arguments or {"candidate_id": "c0"}}}
    if call_id is not None:
        call["id"] = call_id
    return {"model": "local", "done": True, "done_reason": "stop", "prompt_eval_count": 10,
        "eval_count": 5, "message": {"role": "assistant", "content": "", "tool_calls": [call]}}


def _adapter(responses):
    pending = iter(responses)
    def send(*_):
        item = next(pending)
        if isinstance(item, Exception):
            raise item
        return deepcopy(item)
    return OllamaChatAgentAdapter(base_url="http://127.0.0.1:11434", model="local", max_output_tokens=64,
        request_sender=send)


def test_native_ids_are_preserved_when_arguments_are_identical():
    originals = [_wire("call_one"), _wire("call_two")]
    adapter = _adapter(originals)
    request = AgentTurnRequest((AgentMessage("user", "Check the candidate"),))
    responses = [adapter.complete_turn(request) for _ in originals]
    assert [row.tool_calls[0].call_id for row in responses] == ["call_one", "call_two"]
    assert [row.provider_state["tool_calls"] for row in responses] == [row["message"]["tool_calls"] for row in originals]
    assert [row.provider_diagnostic["tool_calls"][0]["tool_call_id"] for row in responses] == ["call_one", "call_two"]


def test_missing_native_ids_keep_the_exact_legacy_identity():
    adapter = _adapter([_wire(None), _wire(None)])
    request = AgentTurnRequest((AgentMessage("user", "Check the candidate"),))
    responses = [adapter.complete_turn(request) for _ in range(2)]
    assert responses[0].tool_calls[0].call_id == responses[1].tool_calls[0].call_id
    assert responses[0].tool_calls[0].call_id == "ollama-call-f9779605c2c33cfc"


def test_adapter_to_loop_distinguishes_new_native_invocations_not_automatic_retries():
    owners = Owners(Budget())
    owners.fixed = True
    result = run(_adapter([_wire("check-one"), _wire("check-two"), _wire("finish", name="finish")]), owners)
    assert result.status == "completed" and result.model_turns == 3
    assert [role for role, _ in owners.effects] == ["candidate_check", "candidate_check", "finish"]
    assert result.call_records["check-one"]["status"] == result.call_records["check-two"]["status"] == "completed"


@pytest.mark.parametrize("changed", [False, True])
def test_adapter_to_loop_rejects_same_native_id_replay_or_changed_arguments(changed):
    owners = Owners(Budget())
    second = _wire("original-check", arguments={"candidate_id": "c1" if changed else "c0"})
    result = run(_adapter([_wire("original-check"), second]), owners)
    expected = "TASK_AGENT_TOOL_CALL_IDENTITY_CHANGED" if changed else "TASK_AGENT_REPEATED_TOOL_CALL"
    assert expected in result.termination["detail"]["reasons"]
    assert [role for role, _ in owners.effects] == ["candidate_check"]


def test_completed_native_id_is_still_protected_after_history_restore():
    owners = Owners(Budget())
    retained = run(_adapter([_wire("original-check"), AgentTurnProviderError("controlled pause")]), owners)
    result = run(_adapter([_wire("original-check")]), owners,
        initial_request=retained.request, prior_call_records=retained.call_records)
    assert "TASK_AGENT_REPEATED_TOOL_CALL" in result.termination["detail"]["reasons"]
    assert [role for role, _ in owners.effects] == ["candidate_check"]


def test_unexecuted_native_batch_ids_can_resume_without_fabricated_identity():
    owners = Owners(Budget())
    owners.fixed = True
    read = _wire("read-one", name="source_read", arguments={"binding_id": "one"})
    check = _wire("check-one")
    mixed = deepcopy(read)
    mixed["message"]["tool_calls"].extend(check["message"]["tool_calls"])
    retained = run(_adapter([mixed, AgentTurnProviderError("controlled pause")]), owners)
    assert owners.effects == [] and retained.tool_argument_correction_used
    result = run(_adapter([read, check, _wire("finish", name="finish")]), owners,
        initial_request=retained.request, prior_call_records=retained.call_records)
    assert result.status == "completed" and result.tool_argument_correction_used
    assert [role for role, _ in owners.effects] == ["source_read", "candidate_check", "finish"]


@pytest.mark.parametrize("invalid", ["", "   ", 12, True, {}])
def test_present_invalid_native_id_is_not_silently_replaced(invalid):
    with pytest.raises(AgentTurnProviderError, match="TOOL_CALL_IDENTITY_INVALID"):
        _adapter([_wire(invalid)]).complete_turn(AgentTurnRequest((AgentMessage("user", "check"),)))


def test_replay_identity_preserves_json_argument_types():
    from src.libs.llm.agent_turn import _ollama_replay_call_identity

    raw = _wire("original", arguments={"index": 0})["message"]["tool_calls"]
    changed = [{"call_id": "original", "name": "candidate_check", "arguments": {"index": False}}]
    with pytest.raises(AgentTurnProviderError, match="REPLAY_TOOL_CALL_IDENTITY_INVALID"):
        _ollama_replay_call_identity(raw, changed)
