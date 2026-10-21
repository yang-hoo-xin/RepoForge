"""Read compatibility only; the recovery Lua is verified by real Redis separately."""
import json
from copy import deepcopy

from src.service_layer.api import public_job
from src.service_layer.store import _job
from tests.unit.test_service_result_compaction import retained_job


def test_current_error_reference_is_not_recursively_expanded_by_normal_reader():
    original = {"detail": '"\\汉' * 100, "empty": []}
    legacy = {"code": "lease_expired", "cost_unknown": True, "prior_error_json": json.dumps(original)}
    new = {"schema": "service-lease-expired-error.v2", "code": "lease_expired", "cost_unknown": True,
           "prior_error_ref": {"version": 1, "attempt": 1, "field": "error"}, "reason": "retained summary"}
    raw = retained_job(None)
    raw.update(state="failed", error_json=json.dumps(new))
    raw["attempts_history"][0].update(state="lease_expired", error_json=json.dumps(legacy))
    raw["attempts_history"][1].update(state="lease_expired", error_json=json.dumps(new))
    before = deepcopy(raw)
    value = _job(raw)
    assert value["error"] == new and public_job(value)["error"] == new
    assert "attempts_history" not in public_job(value) and "prior_error_json" not in value["error"]
    assert value["attempts_history"][0]["error"] == legacy and raw == before
    assert value["error"] is not value["attempts_history"][-1]["error"]


def test_old_error_structure_is_read_unchanged_and_not_rewritten():
    legacy = {"code": "lease_expired", "cost_unknown": True, "prior_error_json": '{"original":[]}'}
    raw = retained_job(None)
    raw.update(state="failed", error_json=json.dumps(legacy))
    raw["attempts_history"][-1].update(state="lease_expired", error_json=json.dumps(legacy))
    before = deepcopy(raw)
    assert public_job(_job(raw))["error"] == legacy
    assert raw == before
