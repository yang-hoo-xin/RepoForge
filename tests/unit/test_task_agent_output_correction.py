"""One charged host correction with native history and real session owners, no models.

为什么要有它：可纠正输出不能重领预算、误认交付或破坏已有工具交换和安全停止。
"""
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolResult,
    AgentTurnProviderError,
    AgentTurnResponse,
)
from src.mod_agent.comparison_checkpoint import _request
from src.mod_agent.model_qualification import canonical_hash
from tests.unit.test_task_agent_loop import (
    Budget,
    Client,
    Owners,
    call,
    native,
    native_client,
    response,
    run,
)
from tests.unit.test_task_agent_tools import patch
from tests.unit.test_task_agent_tools import setup as setup


def text(value="The candidate failed; please revise it.", *, stop_reason="stop"):
    return AgentTurnResponse("scripted", "controlled", value, stop_reason=stop_reason)


@pytest.mark.parametrize("reason", ["refusal", "content_filter", "length", "max_tokens", "max_output_tokens", "incomplete"])
def test_stopped_plain_provider_output_never_gets_a_correction(reason):
    owner = Owners(Budget())
    result = run(Client([text(stop_reason=reason)]), owner)
    assert not result.output_correction_used and not result.unadmitted_host_messages
    assert result.model_turns == 1 and owner.budget.bytes == 0 and not owner.effects


@pytest.mark.parametrize("value", [None, "", "  \n", {"invalid": "content"}])
def test_missing_or_invalid_plain_content_does_not_gain_another_call(value):
    result = run(Client([text(value)]), Owners(Budget()))
    assert result.termination["code"] == "TASK_AGENT_FINISH_TOOL_REQUIRED"
    assert not result.output_correction_used and result.model_turns == 1


@pytest.mark.parametrize("limit", ["model", "bytes"])
def test_plain_output_correction_does_not_exceed_existing_budget(limit):
    owner = Owners(Budget(maximum=1 if limit == "model" else 20, max_bytes=0 if limit == "bytes" else 100_000))
    result = run(Client([text()]), owner)
    assert result.termination["code"] == ("TASK_AGENT_MODEL_BUDGET_EXHAUSTED" if limit == "model"
        else "TASK_AGENT_HOST_FEEDBACK_NOT_ADMITTED")
    assert result.request.messages[-1] == AgentMessage("assistant", text().content)
    assert not result.output_correction_used and not result.unadmitted_tool_results
    assert bool(result.unadmitted_host_messages) is (limit == "bytes")
    assert result.model_turns == 1 and owner.budget.bytes == 0 and not owner.effects


@pytest.mark.parametrize("error", [TimeoutError("deadline"), RuntimeError("owner state unavailable")])
def test_host_charge_failure_preserves_time_limit_or_unknown_owner_cause(error):
    owner = Owners(Budget())

    def refuse(**_kwargs):
        raise error

    owner.budget.charge = refuse
    result = run(Client([text()]), owner)
    assert result.termination == {"code": "TASK_AGENT_WALL_BUDGET_EXHAUSTED" if isinstance(error, TimeoutError)
        else "TASK_AGENT_HOST_FEEDBACK_NOT_ADMITTED", "detail": {"error_type": type(error).__name__}}
    assert result.model_turns == 1 and not result.output_correction_used
    assert len(result.unadmitted_host_messages) == 1 and result.observation_bytes == 0


def test_second_plain_response_stops_with_both_real_texts_and_only_one_host_message():
    owner = Owners(Budget())
    result = run(Client([text("first"), text("second")]), owner, tool_argument_correction_used=True)
    assert result.termination["code"] == "TASK_AGENT_FINISH_TOOL_REQUIRED"
    assert result.output_correction_used and result.tool_argument_correction_used
    assert [item.content for item in result.request.messages if item.role == "assistant"] == ["first", "second"]
    assert sum(isinstance(item.content, dict) and item.content.get("message_type") == "task_agent_output_correction"
        for item in result.request.messages) == 1
    assert result.model_turns == 2 and result.read_units == 0


def test_output_correction_is_retained_through_json_restore_and_provider_interruption():
    owner = Owners(Budget())
    first = run(Client([text(), AgentTurnProviderError("controlled interruption")]), owner)
    assert first.output_correction_used and first.model_turns == 2
    before = owner.budget.bytes
    second = run(Client([text("Another plain response")]), owner,
        initial_request=_request(asdict(first.request)), prior_call_records=first.call_records)
    assert second.termination["code"] == "TASK_AGENT_FINISH_TOOL_REQUIRED"
    assert second.output_correction_used and second.model_turns == 1
    assert owner.budget.bytes == before and owner.budget.consumed == 3


@pytest.mark.parametrize("tamper", ["text", "hash", "role", "duplicate", "counter"])
def test_changed_output_correction_history_is_rejected_before_model(tamper):
    owner = Owners(Budget())
    first = run(Client([text(), AgentTurnProviderError("controlled pause")]), owner)
    messages = deepcopy(list(first.request.messages))
    if tamper == "text":
        messages[-2] = replace(messages[-2], content="different assistant output")
    elif tamper == "role":
        messages[-1] = replace(messages[-1], role="assistant")
    elif tamper == "duplicate":
        messages.extend(messages[-2:])
    else:
        messages[-1].content["assistant_message_sha256" if tamper == "hash" else "correction_number"] = (
            "0" * 64 if tamper == "hash" else True)
    before = owner.budget.consumed
    result = run(Client([]), owner, initial_request=replace(first.request, messages=tuple(messages)))
    assert result.termination["code"] == "TASK_AGENT_OUTPUT_CORRECTION_HISTORY_INVALID"
    assert owner.budget.consumed == before


def test_identity_failure_and_new_owner_stop_do_not_become_output_correction():
    owner = Owners(Budget())

    def reject(_response):
        raise PermissionError("actual model identity changed")

    result = run(Client([text()]), owner, response_admission=reject)
    assert result.termination["code"] == "TASK_AGENT_RESPONSE_NOT_ADMITTED" and not result.output_correction_used
    stopped = {"value": False}

    class StopDuringResponse:
        def complete_turn(self, _request):
            stopped["value"] = True
            return text()

    result = run(StopDuringResponse(), Owners(Budget()), stop_check=lambda: {
        "code": "PATCH_BEHAVIOR_NO_PROGRESS"} if stopped["value"] else None)
    assert result.termination["code"] == "PATCH_BEHAVIOR_NO_PROGRESS" and not result.output_correction_used
    assert result.observation_bytes == 0


@pytest.mark.parametrize("already_corrected", [False, True])
def test_prior_unknown_effect_cannot_continue_via_plain_text_correction(already_corrected):
    owner = Owners(Budget())
    prior_call = call("candidate_check", {"candidate_id": "c0"}, "unknown-check")
    prior = AgentToolResult(prior_call.call_id, prior_call.name, {"status": "unknown"})
    record = {"name": prior.name, "arguments_sha256": canonical_hash(prior_call.arguments),
        "status": "unknown", "result": asdict(prior)}
    from src.libs.llm.agent_turn import AgentTurnRequest

    initial = (run(Client([text(), AgentTurnProviderError("controlled pause")]), owner).request
        if already_corrected else AgentTurnRequest((AgentMessage("user", "Task"),)))
    before = owner.budget.bytes
    result = run(Client([text()]), owner, initial_request=replace(initial,
        tool_results=(prior,), provider_state={"calls": []}), prior_call_records={prior.call_id: record})
    assert result.unknown_effects and result.termination["code"] == "TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE"
    assert result.output_correction_used is already_corrected and result.observation_bytes == 0
    assert owner.budget.bytes == before


def test_actual_ollama_wire_preserves_rejected_finish_then_assistant_then_host_then_tools():
    owner, payloads = Owners(Budget()), []
    actual_text = "Failed. The user should repair it."
    replies = [native(("finish", {"candidate_id": "c0"})),
        {"model": "controlled", "done": True, "done_reason": "stop", "message": {"content": actual_text}},
        native(("candidate_submit", {"candidate_id": "c1", "fixed": True})),
        native(("candidate_check", {"candidate_id": "c1"})), native(("finish", {"candidate_id": "c1"}))]
    result = run(native_client(replies, payloads), owner)
    assert result.status == "completed" and result.output_correction_used and not replies
    messages = payloads[2]["messages"]
    assert [row["role"] for row in messages] == ["user", "assistant", "tool", "assistant", "user"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "finish"
    assert '"accepted":false' in messages[2]["content"]
    assert messages[3] == {"role": "assistant", "content": actual_text}
    assert "task_agent_output_correction" in messages[4]["content"]
    assert [row["tool_name"] for row in payloads[4]["messages"] if row["role"] == "tool"] == [
        "finish", "candidate_submit", "candidate_check"]


@pytest.mark.parametrize("recover", [True, False])
def test_normal_session_keeps_failed_check_and_can_continue_after_one_plain_output(setup, recover):
    inputs, session, evaluator, budget = setup
    before = {path: path.read_bytes() for path in inputs.workspace.rglob("*.cs")}
    steps = [lambda: response(call("candidate_submit", patch(session, correct=False), "bad-submit")),
        lambda: response(call("candidate_check", {"candidate_id": session.current_candidate}, "bad-check")),
        lambda: response(call("finish", {"candidate_id": session.current_candidate}, "bad-finish")),
        lambda: text()]
    if recover:
        steps.extend([lambda: response(call("candidate_submit", patch(session), "good-submit")),
            lambda: response(call("candidate_check", {"candidate_id": session.current_candidate}, "good-check")),
            lambda: response(call("finish", {"candidate_id": session.current_candidate}, "good-finish"))])
    else:
        steps.append(lambda: text("Still no tool action"))
    requests = []

    class Scripted:
        def complete_turn(self, request):
            requests.append(request)
            return steps.pop(0)()

    result = session.run(client=Scripted())
    assert result.output_correction_used and not steps
    assert result.status == ("completed" if recover else "blocked")
    assert requests[3].tool_results[0].output["completion_check"]["accepted"] is False
    assert requests[4].messages[-1].content["message_type"] == "task_agent_output_correction"
    assert budget.consumed == len(requests) == (7 if recover else 5)
    assert len(evaluator.calls) == len(session.candidates) == (2 if recover else 1)
    assert result.observation_bytes == budget.bytes
    assert all(path.read_bytes() == raw for path, raw in before.items())
    if recover:
        assert result.completion["whole_task_delivered"] is False
    else:
        assert result.termination["code"] == "TASK_AGENT_FINISH_TOOL_REQUIRED"
