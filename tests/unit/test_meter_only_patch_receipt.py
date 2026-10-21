"""Metered receipts waive quotas, not evidence, accounting or completion checks."""
from copy import deepcopy

import pytest

from src.mod_agent.workflow.graph import _valid_patch_agent_loop_receipt
from tests.unit.test_patch_recovery_ledger import _continuous


def metered_receipt():
    value = _continuous()
    value["policy"].update(model_call_budget_policy="meter_only",
        max_tool_rounds=None, max_total_model_turns=None, max_observation_bytes=None)
    value.update(model_turns_used=8, usage_by_turn=[{} for _ in range(8)],
        total_observation_bytes=1_000_000, recovery_observation_bytes=2_000_000,
        prior_worker_budget={"consumed_model_turns": 20, "consumed_tool_rounds": 10,
            "consumed_observation_bytes": 3_000_000})
    return value


def test_meter_receipt_accepts_actual_cumulative_work_without_legacy_quotas():
    value = metered_receipt()
    original = deepcopy(value)
    assert _valid_patch_agent_loop_receipt(value, model_call_count=10)
    assert value == original


@pytest.mark.parametrize("damage", [
    "usage", "calls", "read_count", "read_origin", "negative_bytes",
    "negative_history", "missing_revision", "failed_behavior", "bad_finish",
    "numeric_meter_limit", "missing_meter_limit",
])
def test_meter_receipt_rejects_unverifiable_or_unsuccessful_results(damage):
    value = metered_receipt()
    if damage == "usage":
        value["usage_by_turn"].pop()
    elif damage == "calls":
        value["recovery_revisions"][0]["model_call_count"] += 1
    elif damage == "read_count":
        value["tool_rounds_used"] += 1
    elif damage == "read_origin":
        value["observation_trace"][1]["observation_origin"] = "current_source_read"
    elif damage == "negative_bytes":
        value["recovery_revisions"][0]["execution"]["total_observation_bytes"] = -1
    elif damage == "negative_history":
        value["prior_worker_budget"]["consumed_model_turns"] = -1
    elif damage == "missing_revision":
        value["recovery_revisions"].pop(0)
    elif damage == "failed_behavior":
        value["behavior_evaluations"][-1]["status"] = "behavior_failed"
    elif damage == "bad_finish":
        value["termination"]["status"] = "blocked"
    elif damage == "numeric_meter_limit":
        value["policy"]["max_tool_rounds"] = 99
    else:
        del value["policy"]["max_tool_rounds"]
    assert not _valid_patch_agent_loop_receipt(value, model_call_count=10)


def test_none_limits_without_explicit_meter_policy_are_not_accepted():
    value = metered_receipt()
    del value["policy"]["model_call_budget_policy"]
    assert not _valid_patch_agent_loop_receipt(value, model_call_count=10)
