"""Actual-visible quotes, stale state and advisory update recovery counterexamples."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolCall,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnRequest,
    AgentTurnResponse,
)
from src.mod_agent.task_agent_v2_semantic_state import (
    record_semantic_state,
    semantic_state_view,
    strip_working_state,
    validate_semantic_state,
    with_semantic_evidence,
)


def boundary(revision="r1", check=None):
    return {"task_contract_ref": "contract", "current_revision_ref": revision, "current_check_ref": check}


def request(working=None, *, revision="r1", check=None, include_source=True, tool_name="source_read"):
    current = {**boundary(revision, check), "origin": "execution_host",
        "experiment_capabilities": {"registered_tests": {"status": "configured"}},
        "semantic_work_state": semantic_state_view(working, boundary(revision, check))}
    output = {"schema": "task-agent-source-view.v2", "contract_ref": "contract", "root_id": "target",
        "parent_revision_ref": "r1", "source_view_ref": "view-r1", "path": "A.cs", "file_sha256": "source-hash",
        "text": "public int Compute()\n    => 1;", "read_window": {"offset": 0, "end": 30}}
    if tool_name == "history_read":
        output = {"history_ref": "archive", "text": 'Model previously said: "the source is certainly correct"', "offset": 0}
    exchanges = (AgentToolExchange((AgentToolResult("read-1", tool_name,
        {"execution": "executed", "value": output}),), {"tool_calls": []}, 1),) if include_source else ()
    return with_semantic_evidence(AgentTurnRequest(messages=(
        AgentMessage("user", {"protocol": "task_agent_v2", "task_contract_ref": "contract",
            "task_requirements": {"original_request": "Change Compute while preserving public behavior"}}),
        AgentMessage("user", {"current_task_state": current})), historical_tool_exchanges=exchanges))


def current(request):
    return request.messages[-1].content["current_task_state"]


def state(request, *, quote="public int Compute()\n    => 1;", kind="source_view"):
    source = next(row for row in current(request)["semantic_evidence_catalog"] if row["kind"] == kind)
    evidence = [{"id": source["id"], "quote": quote}]
    return {"observations": [{"statement": "The returned implementation returns one", "evidence": evidence}],
        "hypotheses": [{"id": "h1", "statement": "The returned constant may explain the failure", "status": "open", "evidence": evidence}],
        "open_questions": ["Which public observation distinguishes this hypothesis?"],
        "next_action": {"purpose": "Check the relevant public behavior", "expected_observation": "A discriminating public result",
            "decision_if_observed": "Revise or retain the hypothesis using the actual result"}}


def response(*metadata, identifier="action"):
    calls = tuple(AgentToolCall(identifier + str(index), "source_read", {"path": "A.cs", "working_state": value})
        for index, value in enumerate(metadata))
    return AgentTurnResponse(provider="fixture", model="fixture", content=None, tool_calls=calls)


def turn(request, response, index=1):
    return {"request": asdict(request), "response": asdict(response),
        "request_ref": {"path": f"request-{index}.json", "sha256": f"request-{index}"},
        "response_ref": {"path": f"response-{index}.json", "sha256": f"response-{index}"}}


def record(working, item):
    return record_semantic_state(working, item["response"], actual_request=item["request"],
        request_ref=item["request_ref"], response_ref=item["response_ref"])


def test_state_cites_actual_page_and_does_not_read_full_source_artifact(monkeypatch):
    from pathlib import Path

    supplied = request()
    payload = state(supplied)
    original = deepcopy(payload)
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("Semantic evidence must use only actual request values"))
    item = turn(supplied, response(payload))
    working = record(None, item)
    view = semantic_state_view(working, boundary())
    assert view["state"] == payload == original and view["latest_update"]["status"] == "accepted"
    assert view["freshness"] == "current" and view["authority"] == "advisory_only"
    assert view["evidence_quotes"][0]["origin"]["request_ref"] == item["request_ref"]
    unseen = state(supplied, quote="UNSEEN SECOND PAGE")
    rejected = record(None, turn(supplied, response(unseen)))
    assert semantic_state_view(rejected, boundary())["latest_update"]["status"] == "invalid"
    assert semantic_state_view(rejected, boundary())["state"] is None


@pytest.mark.parametrize("metadata, expected", [((), "missing"), (({},), "empty"), (({"unexpected": "value"},), "invalid")])
def test_absent_empty_and_invalid_metadata_are_diagnostics(metadata, expected):
    supplied = request()
    original = response(*metadata)
    before = asdict(original)
    working = record(None, turn(supplied, original))
    assert semantic_state_view(working, boundary())["latest_update"]["status"] == expected
    assert asdict(original) == before
    for call in original.tool_calls:
        assert strip_working_state(call.arguments) == {"path": "A.cs"}
        assert "working_state" in call.arguments


def test_conflicting_batch_keeps_raw_calls_but_no_state_and_identical_batch_records_once():
    supplied = request()
    first = state(supplied)
    other = deepcopy(first)
    other["open_questions"] = ["A different unresolved question"]
    invalid = record(None, turn(supplied, response(first, other)))
    assert semantic_state_view(invalid, boundary())["latest_update"]["status"] == "conflicting"
    same = turn(supplied, response(first, deepcopy(first)))
    working = record(None, same)
    assert len(working["semantic_state"]["snapshots"]) == 1
    assert record(working, same) == working
    next_request = request(working)
    working = record(working, turn(next_request, response(first), 2))
    assert semantic_state_view(working, boundary())["latest_update"]["status"] == "reaffirmed"
    assert len(working["semantic_state"]["snapshots"]) == 1
    assert len(working["semantic_state"]["snapshots"][0]["occurrences"]) == 2


def test_old_quote_remains_available_and_historical_without_promoting_model_summary():
    supplied = request()
    first = turn(supplied, response(state(supplied)))
    working = record(None, first)
    newer = request(working, revision="r2", check="new-check", include_source=False)
    view = current(newer)["semantic_work_state"]
    assert view["freshness"] == "historical"
    retained = next(row for row in current(newer)["semantic_evidence_catalog"] if "retained_origin" in row)
    assert retained["source_boundary"]["current_revision_ref"] == "r1"
    assert not any("state" in row["locator"] for row in current(newer)["semantic_evidence_catalog"])
    payload = state(newer)
    second = turn(newer, response(payload), 2)
    working = record(working, second)
    next_view = semantic_state_view(working, boundary("r2", "new-check"))
    assert next_view["latest_update"]["status"] == "accepted"
    assert next_view["freshness"] == "historical"
    assert next_view["evidence_quotes"][0]["origin"]["request_ref"] == first["request_ref"]
    validate_semantic_state(working, [first, second])


def test_history_excerpt_keeps_mixed_kind_when_quoted_again():
    supplied = request(tool_name="history_read")
    first = turn(supplied, response(state(supplied, quote="the source is certainly correct", kind="history_excerpt")))
    working = record(None, first)
    view = semantic_state_view(working, boundary())
    assert view["evidence_quotes"][0]["origin"]["kind"] == "history_excerpt"
    assert view["freshness"] == "historical"
    later = request(working, include_source=False)
    retained = next(row for row in current(later)["semantic_evidence_catalog"] if "retained_origin" in row)
    assert retained["kind"] == "history_excerpt" and retained["source_boundary"]["mixed_historical_text"] is True


def test_recovery_rejects_forged_state_or_truthful_words_bound_to_wrong_turn():
    supplied = request()
    item = turn(supplied, response(state(supplied)))
    working = record(None, item)
    validate_semantic_state(working, [item])
    changed = deepcopy(working)
    changed["semantic_state"]["snapshots"][0]["state"]["hypotheses"][0]["status"] = "refuted"
    with pytest.raises(ValueError, match="SEMANTIC_STATE_NOT_OBSERVED"):
        validate_semantic_state(changed, [item])
    wrong_turn = deepcopy(working)
    wrong_turn["semantic_state"]["snapshots"][0]["evidence_quotes"][0]["origin"]["request_ref"] = {"path": "other.json", "sha256": "other"}
    with pytest.raises(ValueError, match="SEMANTIC_STATE_NOT_OBSERVED"):
        validate_semantic_state(wrong_turn, [item])


def test_forged_old_quote_or_catalog_cannot_enter_as_original_evidence():
    supplied = request()
    first = turn(supplied, response(state(supplied)))
    working = record(None, first)
    forged = request(working, include_source=False)
    current(forged)["semantic_work_state"]["evidence_quotes"][0]["quote"] = "Invented memory quote"
    forged = with_semantic_evidence(forged)
    invalid = record(working, turn(forged, response(state(forged, quote="Invented memory quote")), 2))
    assert semantic_state_view(invalid, boundary())["latest_update"]["diagnostics"] == ["prior_model_view_not_verified"]
    removed = request(working, include_source=False)
    current(removed)["semantic_evidence_catalog"] = current(supplied)["semantic_evidence_catalog"]
    invalid = record(working, turn(removed, response(state(supplied)), 2))
    assert semantic_state_view(invalid, boundary())["latest_update"]["diagnostics"] == ["evidence_catalog_not_current_visible_values"]
