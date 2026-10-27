"""Actual candidate owners reject a no-op, then allow a bounded useful revision.

为什么要有它：未修改代码的提交必须被拒绝，但一次可纠正输入不能自动撤销继续修复的机会。
"""
from dataclasses import asdict

import pytest

from src.libs.llm.agent_turn import AgentTurnProviderError
from src.mod_agent.comparison_checkpoint import _request
from tests.unit.test_current_candidate import current_setup, revision  # noqa: F401
from tests.unit.test_task_agent_loop import call, response
from tests.unit.test_task_agent_tools import patch, setup  # noqa: F401


def _noop(session):
    return revision(session, after="RecordMissingMarker();")


def test_failed_behavior_noop_feedback_then_actual_revision_and_recheck(current_setup):  # noqa: F811
    _inputs, session, oracle, budget = current_setup
    requests = []

    class Scripted:
        def complete_turn(self, request):
            requests.append(request)
            turn = len(requests)
            if turn == 1:
                return response(call("candidate_submit", patch(session, correct=False), "initial"))
            if turn in (2, 5):
                return response(call("candidate_check", {"candidate_id": session.current_candidate}, "check-" + str(turn)))
            if turn == 3:
                return response(call("candidate_submit", _noop(session), "noop"))
            if turn == 4:
                feedback = request.tool_results[0]
                assert feedback.call_id == "noop"
                assert feedback.output["status"] == "rejected_not_executed"
                assert feedback.output["code"] == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"
                assert len(session.candidates) == 1 and len(oracle.calls) == 1
                return response(call("candidate_submit", revision(session), "changed"))
            if turn == 6:
                return response(call("finish", {"candidate_id": session.current_candidate}, "finish"))
            pytest.fail("Unexpected additional model turn")

    result = session.run(client=Scripted())
    assert result.status == "completed", result.termination
    assert result.tool_argument_correction_used and result.model_turns == 6
    assert len(session.candidates) == 2 and len(oracle.calls) == 2
    assert result.call_records["noop"]["status"] == "not_dispatched"
    assert budget.reads == 0 and budget.bytes == result.observation_bytes > 0
    assert "RecordRuntimeMarker();" in session._candidate(session.current_candidate)["generated_changes"][0]["patch"]["after_text"]


def test_repeated_noop_correction_stops_after_recovery_without_new_feedback(current_setup):  # noqa: F811
    _inputs, session, _oracle, budget = current_setup
    responses = [lambda: response(call("candidate_submit", patch(session, correct=False), "initial")),
                 lambda: response(call("candidate_submit", _noop(session), "noop"))]

    class Interrupted:
        def complete_turn(self, request):
            if responses:
                return responses.pop(0)()
            raise AgentTurnProviderError("controlled interruption")

    first = session.run(client=Interrupted())
    assert first.tool_argument_correction_used, first.termination
    charged = budget.bytes
    budget.maximum = budget.consumed + 2

    class Repeated:
        def complete_turn(self, request):
            return response(call("candidate_submit", _noop(session), "again"))

    second = session.run(client=Repeated(), initial_request=_request(asdict(first.request)),
                         prior_call_records=first.call_records)
    assert second.termination["code"] == "TASK_AGENT_CORRECTION_NO_PROGRESS"
    assert second.tool_argument_correction_used and budget.bytes == charged
    assert any(event["event"] == "correction_no_progress" for event in second.events)
    assert second.call_records["again"]["status"] == "not_dispatched"
    assert len(session.candidates) == 1


@pytest.mark.parametrize("fault", ["parent", "hash", "binding"])
def test_noop_never_downgrades_parent_version_or_authority_failure(current_setup, fault):  # noqa: F811
    _inputs, session, _oracle, _budget = current_setup
    turns = 0

    class Scripted:
        def complete_turn(self, request):
            nonlocal turns
            turns += 1
            if turns == 1:
                return response(call("candidate_submit", patch(session, correct=False), "initial"))
            args = _noop(session)
            if fault == "parent":
                args["parent_candidate_id"] = "not-the-current-parent"
            elif fault == "hash":
                args["parent_code_sha256"] = "sha256:" + "0" * 64
            else:
                args["edits"][0]["edit_binding_id"] = "not-authorized"
            return response(call("candidate_submit", args, "bad-noop"))

    result = session.run(client=Scripted())
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert not result.tool_argument_correction_used and result.model_turns == 2
    assert len(session.candidates) == 1


@pytest.mark.parametrize("limit", ["calls", "feedback"])
def test_noop_feedback_cannot_expand_resource_budget(current_setup, limit):  # noqa: F811
    _inputs, session, _oracle, budget = current_setup
    turns = 0
    if limit == "calls":
        budget.maximum = 2

    class Scripted:
        def complete_turn(self, request):
            nonlocal turns
            turns += 1
            if turns == 1:
                return response(call("candidate_submit", patch(session, correct=False), "initial"))
            if limit == "feedback":
                budget.max_bytes = budget.bytes
            return response(call("candidate_submit", _noop(session), "noop"))

    result = session.run(client=Scripted())
    assert result.termination["code"] == ("TASK_AGENT_MODEL_BUDGET_EXHAUSTED" if limit == "calls"
                                           else "TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED")
    assert result.model_turns == 2 and len(session.candidates) == 1
    assert not result.tool_argument_correction_used
    if limit == "feedback":
        assert len(result.unadmitted_tool_results) == 1
        assert result.unadmitted_tool_results[0].output["code"] == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"
