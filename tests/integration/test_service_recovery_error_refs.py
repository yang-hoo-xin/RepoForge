"""Real Redis recovery/error-history checks; no in-memory Lua simulation.

Scope: fixture-owned random namespaces. 为什么要有它：真实Lua原子预核和保留字节必须在Redis验证，不能拿Python投影当实际结果。
"""
import json
import time

import pytest
from redis.exceptions import ResponseError

from src.service_layer.store import _MAX_ERROR_BYTES, _json
from tests.integration import test_service_api_redis as api_fixtures
from tests.integration import test_service_job_store as store_fixtures

SCHEMA = "service-lease-expired-error.v2"
api_pair = api_fixtures.api_pair
store = store_fixtures.store


def near_limit_error():
    error = {"detail": '"\\汉' * 9250, "empty": [], "object": {}, "flag": False}
    assert _MAX_ERROR_BYTES - 1024 <= len(_json(error, _MAX_ERROR_BYTES).encode()) <= _MAX_ERROR_BYTES
    return error


def raw_job(store, job_id):
    return json.loads(store.client.get(store.key_prefix + "job:" + job_id))


def metrics_without_clock(store):
    # Scope: only the observation timestamp differs between reads.
    # 为什么要有它：排除自然时间流逝造成的误判，仍逐项比较全部计数和队列状态。
    value = store.metrics()
    value.pop("timestamp")
    return value


def expire(store):
    active = store.claim("error-history-worker", 0.02)
    assert active is not None
    time.sleep(0.04)
    return active


def seeded_retry(store, attempts=3, error=None):
    job, created = store.submit("query", {"query": "synthetic history fixture"}, "one", 10, attempts)
    assert created
    first = store.claim("initial-error-worker", 30)
    assert store.retry(job["id"], first["lease_token"], near_limit_error() if error is None else error, 0)
    return job["id"]


def assert_current_reference(raw, prior_attempt):
    current = json.loads(raw["error_json"])
    assert current["schema"] == SCHEMA and current["code"] == "lease_expired" and current["cost_unknown"] is True
    assert "prior_error_json" not in current
    assert current["prior_error_ref"] == (None if prior_attempt is None else
        {"version": 1, "attempt": prior_attempt, "field": "error"})
    assert raw["attempts_history"][-1]["error_json"] == raw["error_json"]
    assert len(raw["error_json"].encode()) < 512
    return current


def test_store_ten_attempt_bound_retains_initial_error_without_reescaping(store, record_property):
    job_id = seeded_retry(store, attempts=10)
    previous = [row["error_json"] for row in raw_job(store, job_id)["attempts_history"]]
    initial = previous[0]
    for number in range(2, 11):
        active = expire(store)
        assert active["attempt"] == number
        outcome = store.recover_expired()
        assert outcome["expired"] == 1
        assert outcome["queries_exhausted" if number == 10 else "queries_requeued"] == 1
        raw = raw_job(store, job_id)
        assert_current_reference(raw, number - 1)
        history = [row["error_json"] for row in raw["attempts_history"]]
        assert history[:-1] == previous and history[0] == initial
        assert sum(len(x.encode()) for x in history) < len(initial.encode()) + 512 * (number - 1)
        previous = history
        assert not store.finish(job_id, active["lease_token"], "completed", result="late")
    assert store.get(job_id)["state"] == "failed" and store.claim("no-eleventh-attempt", 30) is None
    assert 0 < store.client.pttl(store.key_prefix + "job:" + job_id) <= 86400000
    record_property("scope", "Store explicit max_attempts=10, not the HTTP default or a model run")


def test_http_explicit_three_attempts_retains_prior_error_but_returns_only_summary(api_pair, record_property):
    clients, stores, auth = api_pair
    body = {"kind": "query", "payload": {"query": "synthetic HTTP history fixture"}, "max_attempts": 3}
    headers = {**auth, "Idempotency-Key": "three-attempt-history"}
    accepted = clients[0].post("/v1/jobs", headers=headers, json=body)
    assert accepted.status_code == 202 and accepted.json()["job"]["max_attempts"] == 3
    assert clients[0].post("/v1/jobs", headers=headers, json={**body, "max_attempts": 4}).status_code == 422
    assert clients[0].post("/v1/jobs", headers=headers, json={"kind": "agent", "max_attempts": 2,
        "payload": {"profile_id": "synthetic", "description": "not executed"}}).status_code == 422
    owner, job_id = stores[0], accepted.json()["job"]["id"]
    first = owner.claim("initial-http-error", 30)
    error = near_limit_error()
    assert owner.retry(job_id, first["lease_token"], error, 0)
    original = raw_job(owner, job_id)["attempts_history"][0]["error_json"]
    for number in (2, 3):
        expire(owner)
        assert owner.recover_expired()["expired"] == 1
        raw = raw_job(owner, job_id)
        current = assert_current_reference(raw, number - 1)
        assert raw["attempts_history"][0]["error_json"] == original
        response = clients[1].get("/v1/jobs/" + job_id, headers=auth)
        assert response.status_code == 200 and response.json()["error"] == current
        assert "attempts_history" not in response.json() and "prior_error_json" not in response.text
        assert owner.get(job_id)["attempts_history"][0]["error"] == error
    repeated = clients[1].post("/v1/jobs", headers=headers, json=body)
    assert repeated.status_code == 200 and repeated.json()["job"]["state"] == "failed"
    record_property("scope", "Normal ASGI API explicitly requested 3 attempts; HTTP default remains 1 and Agent allows only 1")


@pytest.mark.parametrize("kind", ["query", "agent"])
def test_first_expiry_has_null_reference_and_agent_is_never_requeued(store, kind):
    job, _ = store.submit(kind, {"query": "synthetic"}, "first-expiry", 10, 3)
    expire(store)
    outcome = store.recover_expired()
    raw = raw_job(store, job["id"])
    assert_current_reference(raw, None)
    if kind == "agent":
        assert outcome["agents_indeterminate"] == 1 and outcome["queries_requeued"] == 0
        assert raw["state"] == "indeterminate" and store.claim("not-retried", 30) is None
    else:
        assert raw["state"] == "queued"


def test_old_v1_to_new_v2_and_retry_new_error_preserve_each_raw_history(store):
    job_id = seeded_retry(store, attempts=5, error={"original": [False, [], {}]})
    expire(store)
    assert store.recover_expired()["expired"] == 1
    raw = raw_job(store, job_id)
    original = raw["attempts_history"][0]["error_json"]
    legacy = json.dumps({"code": "lease_expired", "cost_unknown": True, "prior_error_json": original,
                         "reason": "retained legacy representation"}, separators=(",", ":"))
    raw["error_json"] = raw["attempts_history"][1]["error_json"] = legacy
    # Scope: only this fixture's synthetic job. 为什么要有它：验证旧存储格式转入新恢复，不迁移或重写任何真实旧数据。
    store.client.set(store.key_prefix + "job:" + job_id, json.dumps(raw), keepttl=True)
    expire(store)
    assert store.recover_expired()["expired"] == 1
    current = raw_job(store, job_id)
    assert_current_reference(current, 2)
    assert [x["error_json"] for x in current["attempts_history"][:2]] == [original, legacy]
    fourth = store.claim("new-error-worker", 30)
    new_error = {"schema": SCHEMA, "prior_error_ref": {"attempt": 999}, "application_data": []}
    assert store.retry(job_id, fourth["lease_token"], new_error, 0)
    raw_fourth_error = raw_job(store, job_id)["attempts_history"][3]["error_json"]
    expire(store)
    assert store.recover_expired()["queries_exhausted"] == 1
    final = raw_job(store, job_id)
    assert_current_reference(final, 4)
    assert final["attempts_history"][3]["error_json"] == raw_fourth_error
    assert store.get(job_id)["attempts_history"][3]["error"] == new_error


@pytest.mark.parametrize("fault", ["self", "future", "missing_target", "missing_reference", "extra_field", "bad_version",
                                  "top_history_conflict", "history_number", "missing_history"])
def test_corrupt_error_history_reference_is_rejected_without_mutating_job(store, fault):
    job_id = seeded_retry(store, error={"known": "first"})
    expire(store)
    store.recover_expired()
    expire(store)
    key = store.key_prefix + "job:" + job_id
    raw = raw_job(store, job_id)
    value = json.loads(raw["attempts_history"][1]["error_json"])
    if fault in {"self", "future", "missing_target"}:
        value["prior_error_ref"]["attempt"] = {"self": 2, "future": 3, "missing_target": 999}[fault]
    elif fault == "missing_reference":
        del value["prior_error_ref"]
    elif fault == "extra_field":
        value["prior_error_ref"]["operation"] = "replay"
    elif fault == "bad_version":
        value["prior_error_ref"]["version"] = True
    elif fault == "history_number":
        raw["attempts_history"][0]["attempt"] = 2
    elif fault == "missing_history":
        raw["attempts_history"].pop(0)
    if fault not in {"history_number", "missing_history"}:
        raw["attempts_history"][1]["error_json"] = json.dumps(value)
        raw["error_json"] = '"conflict"' if fault == "top_history_conflict" else raw["attempts_history"][1]["error_json"]
    stored = json.dumps(raw)
    store.client.set(key, stored, keepttl=True)
    metrics = metrics_without_clock(store)
    processing = store.client.zrange(store.key_prefix + "processing", 0, -1, withscores=True)
    with pytest.raises(ResponseError, match="RECOVERY_ERROR_HISTORY_INCONSISTENT"):
        store.recover_expired()
    observed = store.client.get(key)
    assert (observed.decode() if isinstance(observed, bytes) else observed) == stored
    assert metrics_without_clock(store) == metrics
    assert store.client.zrange(store.key_prefix + "processing", 0, -1, withscores=True) == processing


def test_hundredth_invalid_history_prevents_all_batch_writes(store):
    # Scope: this 100-job corruption fixture only, before the first write.
    # 为什么要有它：新生命周期预留不能把原第100项损坏的回归变成配额测试；明确提供合成额度，仍逐字检查整批没有写入。
    store = store_fixtures.RedisJobStore(store.client, namespace=store.namespace,
        retention_seconds=store.retention_seconds, max_pending=store.max_pending,
        capacity_bytes=2*1024**3, cache_capacity_bytes=store.cache_capacity_bytes,
        cache_ttl_seconds=store.cache_ttl_seconds)
    ids = [store.submit("query", {"query": "batch fixture"}, "batch-" + str(i), 10, 1)[0]["id"] for i in range(100)]
    for _ in ids:
        assert store.claim("batch-worker", 0.02) is not None
    time.sleep(0.04)
    ordered = store.client.zrange(store.key_prefix + "processing", 0, -1)
    final_id = ordered[-1].decode() if isinstance(ordered[-1], bytes) else ordered[-1]
    key = store.key_prefix + "job:" + final_id
    bad = raw_job(store, final_id)
    bad["error_json"] = '"not in first-attempt history"'
    store.client.set(key, json.dumps(bad), keepttl=True)
    before = {job_id: store.client.get(store.key_prefix + "job:" + job_id) for job_id in ids}
    counters = metrics_without_clock(store)
    with pytest.raises(ResponseError, match="RECOVERY_ERROR_HISTORY_INCONSISTENT"):
        store.recover_expired()
    assert {job_id: store.client.get(store.key_prefix + "job:" + job_id) for job_id in ids} == before
    assert metrics_without_clock(store) == counters and store.client.zcard(store.key_prefix + "processing") == 100
