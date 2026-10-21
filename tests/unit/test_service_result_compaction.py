"""Storage decoding controls only; real Lua/API verification remains separate."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src.service_layer.api import public_job
from src.service_layer.store import RedisJobStore, _job


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def retained_job(value, *, compact=False):
    result = encoded(value)
    prior = {"attempt": 1, "state": "retry_scheduled", "started_at": 2,
             "finished_at": 3, "lease_token": "old-token", "worker_id": "old-worker",
             "result_json": "null", "error_json": encoded({"code": "old failure", "notes": []}),
             "cost_unknown": True}
    last = {"attempt": 2, "state": "completed", "started_at": 4, "finished_at": 5,
            "lease_token": "final-token", "worker_id": "final-worker", "result_json": result,
            "error_json": "null", "cost_unknown": None}
    if compact:
        del last["result_json"]
        last["_result_from_job"] = {"version": 1, "attempt": 2}
    return {"id": "a" * 32, "kind": "query", "state": "completed", "attempt": 2,
            "max_attempts": 3, "timeout_seconds": 10, "payload_json": encoded({"query": "原问题", "options": []}),
            "result_json": result, "error_json": "null", "attempts_history": [prior, last],
            "created_at": 1, "started_at": 4, "finished_at": 5, "available_at": 3,
            "lease_until": None, "lease_token": None, "attempt_deadline_at": 14,
            "worker_id": "final-worker", "_request_hash": "internal", "_idempotency_digest": "internal"}


@pytest.mark.parametrize("value", [{"answer": "完成", "arrays": [], "objects": {}, "typed": [0, False, None, "0"]},
                                  [], {}, False, 0, ""])
def test_compact_and_legacy_decode_to_identical_public_and_private_results(value):
    legacy, compact = retained_job(value), retained_job(value, compact=True)
    observed = _job(compact)
    assert observed == _job(legacy)
    assert public_job(observed) == public_job(_job(retained_job(value)))
    assert observed["result"] == observed["attempts_history"][-1]["result"] == value
    assert observed["attempts_history"][0]["error"] == {"code": "old failure", "notes": []}
    assert observed["attempts_history"][0]["cost_unknown"] is True
    assert "_result_from_job" not in observed["attempts_history"][-1]


def test_reconstructed_result_objects_are_independent_and_input_is_not_mutated():
    raw = retained_job({"nested": [{"items": []}]}, compact=True)
    original = deepcopy(raw)
    result = _job(raw)
    result["result"]["nested"][0]["items"].append("only top")
    assert result["attempts_history"][-1]["result"]["nested"][0]["items"] == []
    result["attempts_history"][-1]["result"]["nested"].append("only history")
    assert len(result["result"]["nested"]) == 1
    assert raw == original


@pytest.mark.parametrize("fault", ["unknown_version", "boolean_version", "wrong_attempt", "boolean_attempt",
                                  "marker_extra_field", "marker_not_object", "duplicate_result", "job_not_completed",
                                  "attempt_not_completed", "job_attempt_mismatch", "historical_marker", "null_result"])
def test_invalid_or_conflicting_result_reference_is_rejected(fault):
    raw = retained_job({"answer": "retained"}, compact=True)
    last = raw["attempts_history"][-1]
    marker = last["_result_from_job"]
    if fault == "unknown_version":
        marker["version"] = 2
    elif fault == "boolean_version":
        marker["version"] = True
    elif fault == "wrong_attempt":
        marker["attempt"] = 1
    elif fault == "boolean_attempt":
        marker["attempt"] = True
    elif fault == "marker_extra_field":
        marker["other"] = "unrecognized"
    elif fault == "marker_not_object":
        last["_result_from_job"] = "result_json"
    elif fault == "duplicate_result":
        last["result_json"] = raw["result_json"]
    elif fault == "job_not_completed":
        raw["state"] = "failed"
    elif fault == "attempt_not_completed":
        last["state"] = "failed"
    elif fault == "job_attempt_mismatch":
        raw["attempt"] = 3
    elif fault == "historical_marker":
        raw["attempts_history"][0].pop("result_json")
        raw["attempts_history"][0]["_result_from_job"] = {"version": 1, "attempt": 1}
    else:
        raw["result_json"] = "null"
    with pytest.raises(ValueError, match="SERVICE_JOB_RESULT_REFERENCE_INVALID"):
        _job(raw)


def test_legacy_null_result_and_empty_history_remain_supported():
    raw = retained_job(None)
    result = _job(raw)
    assert result["result"] is result["attempts_history"][-1]["result"] is None
    queued = retained_job(None)
    queued.update(state="queued", attempt=0, attempts_history={})
    assert _job(queued)["attempts_history"] == []


class ReadOnlyTransport:
    """Return stored bytes to real Store readers; this does not emulate Lua."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get(self, key):
        return encoded(self.rows[key.rsplit(":", 1)[-1]])

    def register_script(self, script):
        def existing(*, keys, args):
            self.calls.append(args[0])
            assert args[0] == "submit"
            return encoded({"code": "existing", "job": self.rows["a" * 32]})
        return existing


def test_normal_store_get_and_idempotent_submit_read_mixed_formats_identically():
    compact = retained_job({"value": []}, compact=True)
    legacy = retained_job({"value": []})
    legacy["id"] = "b" * 32
    transport = ReadOnlyTransport({compact["id"]: compact, legacy["id"]: legacy})
    store = RedisJobStore(transport, namespace="pure-decoder-test")
    first = store.get(compact["id"])
    second = store.get(legacy["id"])
    assert first["result"] == second["result"]
    assert first["attempts_history"] == second["attempts_history"]
    same, created = store.submit("query", {"query": "原问题", "options": []}, "retained", 10, 3)
    assert same == first and created is False and transport.calls == ["submit"]
    assert store.get(compact["id"]) == first
