"""Uncapped accounting retains real counts and existing identity checks."""
import json
from copy import deepcopy
from math import inf
from types import SimpleNamespace

import pytest

from src.mod_agent.local_entry import (
    LocalEntryError,
    _preflight_request_model_calls,
    _reserve_request_model_call,
)
from src.mod_agent.model_call_budget import (
    as_meter_only,
    call_allowance,
    is_meter_only,
    validate_budget,
)
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
from tests.unit.test_patch_agent_loop import _fixture, _response, _run, _ScriptedClient, _tool_call
from tests.unit.test_task_total_budget import historical_budget


def test_meter_keeps_counts_after_previous_limits_and_never_serializes_infinity():
    original = historical_budget()
    before = deepcopy(original)
    metered = as_meter_only(original)
    assert is_meter_only(metered)
    for _ in range(20):
        metered = _reserve_request_model_call(metered, "requirement_findings")
    assert metered["consumed"] == 28
    assert metered["by_stage"]["requirement_findings"]["consumed"] == 25
    assert metered["maximum"] is metered["remaining"] is None
    assert call_allowance(metered, "requirement_findings", local_limit=0) == inf
    assert call_allowance(metered, "unknown") == 0
    assert json.loads(json.dumps(metered, allow_nan=False)) == validate_budget(metered)
    assert original == before
    projected = _preflight_request_model_calls(metered, ("requirement_findings", "planning"))
    assert projected["admitted"] and projected["projected_budget"]["consumed"] == 30
    assert metered["consumed"] == 28
    with pytest.raises(LocalEntryError, match="PRODUCT_MODEL_CALL_BUDGET_EXHAUSTED"):
        _reserve_request_model_call(metered, "unknown")


@pytest.mark.parametrize("mutation", ["numeric_limit", "negative_count", "wrong_total"])
def test_meter_rejects_invalid_accounting(mutation):
    value = as_meter_only(historical_budget())
    if mutation == "numeric_limit":
        value["maximum"] = 99
    elif mutation == "negative_count":
        value["by_stage"]["repair"]["consumed"] = -1
    else:
        value["consumed"] += 1
    with pytest.raises(ValueError, match="PRODUCT_MODEL_CALL_BUDGET_INVALID"):
        validate_budget(value)


def test_patch_loop_meter_passes_legacy_turn_read_and_byte_limits(tmp_path):
    policy = PatchAgentLoopPolicy(model_call_budget_policy="meter_only",
        max_total_model_turns=1, max_tool_rounds=0, max_observation_bytes=0,
        finalize_after_first_observation=False, require_source_inspection=False)
    assert policy.max_total_model_turns is policy.max_tool_rounds is policy.max_observation_bytes is None
    _, _, _, inspector = _fixture(tmp_path, binding_count=6, policy=policy)
    replies = [_response(turn=i, tool_calls=(_tool_call(f"edit_binding_{i:03d}", turn=i),))
        for i in range(1, 7)]
    replies.append(_response(turn=7, content='{"changes":[]}'))
    client = _ScriptedClient(replies)
    result = _run(client, inspector)
    assert result.completed and result.model_turns_used == 7 and result.tool_rounds_used == 6
    assert result.total_observation_bytes > 0
    for request in client.requests:
        states = [message.content["budget_state"] for message in request.messages
            if isinstance(message.content, dict) and "budget_state" in message.content]
        assert states
        assert states[-1]["remaining_model_turns"] is None
        assert states[-1]["remaining_tool_rounds"] is None
        assert states[-1]["remaining_observation_bytes"] is None
        json.dumps(states[-1], allow_nan=False)


def test_patch_loop_meter_preserves_source_change_rejection(tmp_path):
    policy = PatchAgentLoopPolicy(model_call_budget_policy="meter_only", finalize_after_first_observation=False)
    _, source, _, inspector = _fixture(tmp_path, policy=policy)
    client = _ScriptedClient([_response(turn=1, tool_calls=(_tool_call("edit_binding_001", turn=1),))],
        before_response=lambda _: source.write_text("changed", encoding="utf-8"))
    result = _run(client, inspector)
    assert not result.completed
    assert result.termination["code"] == "PATCH_AGENT_SOURCE_SNAPSHOT_DRIFT"


@pytest.mark.parametrize("repeat", [False, True])
def test_meter_submission_feedback_still_repairs_or_stops_no_progress(tmp_path, repeat):
    from tests.unit.test_patch_submission_static_loop import _reject
    from tests.unit.test_patch_submission_static_loop import _run as run_submission
    def validate(response):
        if repeat or response.content == '{"changes":[]}':
            _reject(response)
    result, client = run_submission(tmp_path, [_response(turn=1, content='{"changes":[]}'),
        _response(turn=2, content='{"changes":[]}' if repeat else '{"changes": []}')], validate,
        policy=PatchAgentLoopPolicy(model_call_budget_policy="meter_only", require_source_inspection=False))
    assert len(client.requests) == 2
    assert result.completed is not repeat
    if repeat:
        assert result.termination["code"] == "PATCH_AGENT_REPEATED_INVALID_SUBMISSION"
    assert result.submission_trace[0]["released_recovery_reservation"] == {
        "model_turns": 0, "tool_reads": 0, "observation_bytes": 0}


def test_adapter_meter_keeps_correction_counts_and_truthful_unread_sources(tmp_path):
    from tests.unit.test_fresh_planning_continuation import _inputs, _PatchAgentTurnClient
    from tests.unit.test_patch_behavior_feedback import _adapter, _BehaviorOracle
    adapter = _adapter(_inputs(tmp_path), _PatchAgentTurnClient(), _BehaviorOracle(tmp_path / "observations"))
    adapter.patch_agent_loop_policy = PatchAgentLoopPolicy(
        model_call_budget_policy="meter_only", finalize_after_first_observation=False)
    for _ in range(20):
        adapter._consume_correction_call()
    assert adapter._receipt["model_call_count"] == 20
    adapter._receipt["patch_agent_loop"] = {
        "tool_rounds_used": 12, "total_observation_bytes": 100_000,
        "observation_trace": [{"binding_id": "already_read"}],
    }
    inspector = SimpleNamespace(available_read_binding_ids=("already_read", "unread"))
    result = adapter._failure_capacity(inspector, {"observed": "failed"})
    assert result["remaining_model_turns"] is result["remaining_binding_reads"] is None
    assert result["direct_revision_available"]
    assert result["affordable_unread_binding_ids"] == ["unread"]
    assert adapter._initial_recovery_reservations(inspector) == {}
    json.dumps(result, allow_nan=False)
