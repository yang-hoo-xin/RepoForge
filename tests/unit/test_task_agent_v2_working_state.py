"""Counterexamples for durable public statements, repeated reads and recovered counters."""
import json

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnRequest,
    AgentTurnResponse,
)
from src.mod_agent.task_agent_v2_working_state import (
    retain_model_statement,
    update_working_state,
    working_state_view,
)


def observed(identifier, *, text="same source", name="source_read", content=""):
    native = {"content": content, "tool_calls": [{"id": identifier,
        "function": {"name": name, "arguments": {"path": "source.cs"}}}]}
    result = AgentToolResult(identifier, name, {"execution": "executed", "value": {"text": text}})
    return AgentToolExchange((result,), native, 1)


def advance(previous, exchanges, **changes):
    request = AgentTurnRequest(messages=(AgentMessage("user", "repair"),), historical_tool_exchanges=tuple(exchanges))
    return update_working_state(previous, request, **{"task_contract_ref": "task", "current_revision_ref": "revision",
        "registry_ref": "bindings", "current_check_ref": None, **changes})


def test_explicit_model_claim_survives_many_reads_without_optional_plan():
    first = observed("first", content="The null payload may fail before conversion; test that hypothesis.")
    original = advance(None, [first])
    updated = advance(original, [first, *[observed(str(i), text=str(i)) for i in range(25)]])
    view = working_state_view(updated)
    assert view["model_statements"][0]["content"] == first.provider_state["content"]
    assert "hypothesis" not in view and "plan" not in view
    assert view["counts"]["completed_returns"] == 26
    assert "observed_call_ids" not in view


def test_repeated_return_feedback_is_factual_and_recovery_does_not_reset_or_double_count():
    history = [observed(str(i)) for i in range(4)]
    before = advance(None, history)
    assert before["consecutive_repeated_returns"] == 3
    assert before["stagnation"]["triggered"] is True
    restored = json.loads(json.dumps(before))
    assert advance(restored, history) == restored
    after = advance(restored, [*history, observed("four")])
    assert after["counts"]["completed_returns"] == 5
    assert after["counts"]["repeated_returns"] == 4
    assert after["stagnation"]["business_progress_inferred"] is False
    assert after["stagnation"]["authority_changed"] is False
    assert "No patch or plan tool is mandatory" in after["stagnation"]["guidance"]
    changed = advance(after, [*history, observed("four"), observed("new", text="different returned evidence")])
    assert changed["consecutive_repeated_returns"] == 0


def test_changed_advisory_metadata_does_not_make_an_identical_read_new_evidence():
    from src.mod_agent.model_qualification import canonical_hash

    first, second = observed("first"), observed("second")
    first.provider_state["tool_calls"][0]["function"]["arguments"]["working_state"] = {"hypothesis": "first"}
    second.provider_state["tool_calls"][0]["function"]["arguments"]["working_state"] = {"hypothesis": "revised"}
    raw_first = first.provider_state["tool_calls"][0]["function"]["arguments"]
    raw_second = second.provider_state["tool_calls"][0]["function"]["arguments"]
    assert canonical_hash(raw_first) != canonical_hash(raw_second)
    working = advance(None, [first, second])
    assert working["counts"]["completed_returns"] == 2
    assert working["counts"]["distinct_returns"] == working["counts"]["repeated_returns"] == 1
    assert working["latest_action"]["arguments"] == {"path": "source.cs"}
    assert raw_first["working_state"] == {"hypothesis": "first"}
    assert raw_second["working_state"] == {"hypothesis": "revised"}


def test_model_response_content_is_exact_and_never_promoted_to_fact():
    response = AgentTurnResponse(provider="fixture", model="fixture", content="Hypothesis A is unconfirmed. " * 1000)
    working = retain_model_statement(None, response, response_ref={"path": "response.json", "sha256": "hash"})
    assert working["model_statements"][0]["content"] == response.content
    assert len(retain_model_statement(working, response)["model_statements"]) == 1
    assert working["model_statements"][0]["interpretation"].startswith("Model-authored statement")


def test_latest_public_failure_survives_later_pass_and_has_exact_output():
    failure = observed("failed", name="candidate_check")
    failure.tool_results[0].output["value"].update(status="behavior_failed", check_ref="failed-check", feedback_ref="public")
    passed = observed("passed", name="candidate_check")
    passed.tool_results[0].output["value"].update(status="passed", check_ref="passing-check")
    working = advance(None, [failure, passed], current_check_ref="passing-check")
    assert working["latest_public_failure"]["output"] == failure.tool_results[0].output
    assert working["latest_public_check"]["output"] == passed.tool_results[0].output
    assert working["counts"]["accepted_candidate_returns"] == 0


def test_recovery_rejects_an_invented_model_statement_or_public_feedback():
    import pytest

    from src.mod_agent.task_agent_v2_working_state import validate_working_state
    item = observed("read", content="Unconfirmed actual hypothesis")
    working = advance(None, [item])
    records = {"read": {"result": {"call_id": "read", "name": "source_read", "output": item.tool_results[0].output}}}
    validate_working_state(working, responses=[{"content": item.provider_state["content"]}], call_records=records)
    with pytest.raises(ValueError, match="WORKING_STATEMENT_NOT_OBSERVED"):
        validate_working_state(working, responses=[{"content": "different"}], call_records=records)
    forged = json.loads(json.dumps(working))
    forged["latest_public_failure"] = {"call_id": "read", "name": "candidate_check", "output": {"status": "failed"}}
    with pytest.raises(ValueError, match="WORKING_FEEDBACK_NOT_OBSERVED"):
        validate_working_state(forged, responses=[{"content": item.provider_state["content"]}], call_records=records)


def test_first_received_response_can_be_reconciled_before_any_previous_working_state():
    from src.mod_agent.task_agent_v2_working_state import validate_working_state
    working = retain_model_statement(None, {"content": "First actual claim"})
    validate_working_state(working, responses=[{"content": "First actual claim"}], call_records={})


def test_unconfigured_learning_seals_public_episode_and_reports_storage_failure_without_raising(monkeypatch):
    from src.mod_agent import task_agent_v2_experience
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
    owner = TaskAgentExecutionV2.__new__(TaskAgentExecutionV2)
    reference = {"path": "public-episode.json", "sha256": "hash"}
    monkeypatch.setattr(task_agent_v2_experience, "seal_current_public_episode", lambda supplied: reference)
    assert owner.complete_terminal_learning() == {"status": "not_configured", "episode_status": "sealed", "episode_ref": reference}

    def unavailable(supplied):
        raise OSError("controlled storage failure")

    monkeypatch.setattr(task_agent_v2_experience, "seal_current_public_episode", unavailable)
    result = owner.complete_terminal_learning()
    assert result["status"] == "not_configured" and result["episode_status"] == "episode_unavailable"
    assert result["diagnostic"]["error_type"] == "OSError"


def test_repeated_content_keeps_latest_occurrence_and_new_boundary_without_duplicate_body():
    old = {"current_revision_ref": "old"}
    new = {"current_revision_ref": "new"}
    working = None
    for index, text in enumerate(("P", "A", "B", "A"), 1):
        working = retain_model_statement(working, {"content": text},
            response_ref={"path": f"response-{index}.json", "sha256": str(index)},
            source_boundary=new if index == 4 else old)
    assert [row["content"] for row in working["model_statements"]] == ["P", "A", "B"]
    last = next(row for row in working["model_statements"] if row["content"] == "A")
    assert last["response_ref"]["path"] == "response-4.json"
    assert last["first_response_ref"]["path"] == "response-2.json"
    assert last["source_boundary"] == new
    assert working["latest_model_statement_sha256"] == last["content_sha256"]
    replay = retain_model_statement(working, {"content": "A"})
    assert replay == working
    replay = retain_model_statement(working, {"content": "A"}, response_ref=last["response_ref"], source_boundary=new)
    assert replay == working


def test_same_public_failure_is_counted_across_receipts_and_candidates_without_causal_claim():
    history, working = [], None
    for index in range(3):
        check = observed(f"check-{index}", name="candidate_check")
        check.tool_results[0].output.update(version_identity=f"revision-{index}")
        check.tool_results[0].output["value"] = {"status": "behavior_failed", "candidate_ref": f"candidate-{index}",
            "candidate_source_id": f"source-{index}", "check_ref": f"check-{index}", "feedback_ref": f"feedback-{index}",
            "observations": [{"check_id": "null-input", "test_name": "NullInput", "expected": 2,
                "observed": {"actual": 1, "passed": False}, "raw_report_ref": {"path": f"temp-{index}"},
                "details": f"temporary execution {index}"}], "observation_count": 1}
        history.append(check)
        working = advance(working, history, current_revision_ref=f"candidate-{index}", current_check_ref=f"check-{index}")
    assert working["consecutive_repeated_returns"] == 0
    assert working["public_failure_repetition"]["consecutive_checks"] == 3
    assert working["stagnation"]["same_visible_public_failure"] is True
    assert working["stagnation"]["business_progress_inferred"] is False
    changed = observed("changed", name="candidate_check")
    changed.tool_results[0].output["value"] = {"status": "behavior_failed", "check_ref": "changed",
        "observations": [{"check_id": "null-input", "expected": 2, "observed": {"actual": 0, "passed": False}}]}
    updated = advance(working, [*history, changed], current_check_ref="changed")
    assert updated["public_failure_repetition"]["consecutive_checks"] == 1
    assert updated["stagnation"]["same_visible_public_failure"] is False
