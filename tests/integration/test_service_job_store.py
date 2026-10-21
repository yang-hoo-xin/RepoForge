"""Real Redis integration checks; never substitute an in-memory fake service."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

redis = pytest.importorskip("redis", reason="Real Redis tests require the redis client package")

from src.service_layer.store import IdempotencyConflict, QueueFull, RedisJobStore  # noqa: E402


@pytest.fixture
def store():
    url = os.environ.get("SERVICE_TEST_REDIS_URL")
    if not url:
        pytest.skip("SERVICE_TEST_REDIS_URL is unset; a real Redis service is required")
    client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
    try:
        client.ping()
    except redis.RedisError as exc:
        client.close()
        pytest.skip(f"Configured real Redis is unavailable ({type(exc).__name__}); no fake fallback")
    value = RedisJobStore(client, namespace="service-store-test-" + uuid4().hex)
    try:
        yield value
    finally:
        # 为什么要有它：只清理本测试独占的随机命名空间，不能flush共享Redis或触及其他作业。
        keys = list(client.scan_iter(match=value.key_prefix + "*"))
        if keys:
            client.delete(*keys)
        client.close()


def submit(store, key="request", *, kind="query", attempts=2, timeout=20, payload=None):
    return store.submit(kind, {"query": "test"} if payload is None else payload, key, timeout, attempts)


def test_atomic_concurrent_idempotency_and_conflict(store):
    payload = {"query": "原始问题不应成为Redis键", "options": {"b": 2, "a": 1}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: submit(store, payload=payload), range(16)))
    assert len({job["id"] for job, _ in outcomes}) == 1
    assert sum(created for _, created in outcomes) == 1
    existing, created = submit(store, payload={"options": {"a": 1, "b": 2}, "query": payload["query"]})
    assert not created and existing["id"] == outcomes[0][0]["id"]
    with pytest.raises(IdempotencyConflict) as error:
        submit(store, payload={"query": "different"})
    assert error.value.job_id == existing["id"]
    metrics = store.metrics()
    assert metrics["queued"] == metrics["pending"] == 1
    assert metrics["counters"]["submitted"] == 1
    assert metrics["counters"]["idempotency_conflicts"] == 1
    assert all(payload["query"] not in key for key in store.client.scan_iter(match=store.key_prefix + "*"))


def test_json_empty_types_unicode_and_attempt_history_survive_lua(store):
    payload = {"profile_id": "例子", "empty_array": [], "empty_object": {}, "typed": [0, False, None, "0"]}
    job, _ = submit(store, payload=payload)
    assert job["payload"] == payload and job["attempts_history"] == []
    active = store.claim("worker", 10)
    assert active["id"] == job["id"] and active["attempt"] == 1
    result = {"answer": "完成", "citations": [], "metadata": {}}
    assert store.finish(job["id"], active["lease_token"], "completed", result=result)
    raw = json.loads(store.client.get(store.key_prefix + "job:" + job["id"]))
    assert json.loads(raw["result_json"]) == result
    assert "result_json" not in raw["attempts_history"][-1]
    assert raw["attempts_history"][-1]["_result_from_job"] == {"version": 1, "attempt": 1}
    terminal = store.get(job["id"])
    assert terminal["payload"] == payload and terminal["result"] == result
    assert terminal["attempts_history"][0]["result"] == result
    assert terminal["lease_token"] is None
    assert all(type(terminal[name]) is float for name in ("created_at", "started_at", "finished_at"))


@pytest.mark.parametrize("state,result,compact", [
    ("completed", None, False), ("completed", False, True), ("completed", 0, True),
    ("completed", [], True), ("completed", {}, True),
    ("failed", {"preserved": []}, False), ("timed_out", {"preserved": []}, False),
    ("indeterminate", {"preserved": []}, False), ("cancelled", {"preserved": []}, False),
])
def test_lua_compacts_only_completed_nonnull_results_and_keeps_idempotent_reads(store, state, result, compact):
    job, _ = submit(store)
    active = store.claim("worker", 10)
    error = None if state == "completed" else {"code": "retained error", "details": []}
    assert store.finish(job["id"], active["lease_token"], state, result=result, error=error)
    raw = json.loads(store.client.get(store.key_prefix + "job:" + job["id"]))
    last = raw["attempts_history"][-1]
    assert ("_result_from_job" in last) is compact
    assert ("result_json" not in last) is compact
    terminal = store.get(job["id"])
    repeated, created = submit(store)
    assert not created and repeated == terminal
    assert terminal["result"] == terminal["attempts_history"][-1]["result"] == result
    assert terminal["error"] == terminal["attempts_history"][-1]["error"] == error
    assert not store.finish(job["id"], active["lease_token"], "failed", error="late")
    assert store.get(job["id"]) == terminal and store.metrics()["pending"] == 0


def test_claim_race_leases_only_one_worker_and_fences_wrong_tokens(store):
    job, _ = submit(store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = list(pool.map(lambda i: store.claim(f"worker-{i}", 10), range(8)))
    active = [item for item in claimed if item is not None]
    assert len(active) == 1
    assert not store.heartbeat(job["id"], "wrong-token", 10)
    assert not store.finish(job["id"], "wrong-token", "completed", result={"wrong": True})
    assert not store.retry(job["id"], "wrong-token", {"wrong": True}, 0)
    assert store.get(job["id"])["state"] == "running"
    assert store.heartbeat(job["id"], active[0]["lease_token"], 10)
    assert store.finish(job["id"], active[0]["lease_token"], "completed", result={"right": True})
    assert not store.finish(job["id"], active[0]["lease_token"], "failed", error="late")
    assert store.get(job["id"])["result"] == {"right": True}


def test_expired_query_requeues_bounded_attempts_and_rejects_late_result(store):
    job, _ = submit(store, attempts=2)
    first = store.claim("old-worker", 0.2)
    time.sleep(0.26)
    assert not store.heartbeat(job["id"], first["lease_token"], 10)
    assert not store.finish(job["id"], first["lease_token"], "completed", result="late")
    recovered = store.recover_expired()
    assert recovered["expired"] == recovered["queries_requeued"] == 1
    second = store.claim("new-worker", 0.2)
    assert second["attempt"] == 2 and second["lease_token"] != first["lease_token"]
    assert second["attempts_history"][0]["state"] == "lease_expired"
    assert second["attempts_history"][0]["cost_unknown"] is True
    assert not store.finish(job["id"], first["lease_token"], "completed")
    assert not store.heartbeat(job["id"], first["lease_token"], 10)
    assert not store.retry(job["id"], first["lease_token"], {"stale": True}, 0)
    assert store.get(job["id"])["lease_token"] == second["lease_token"]
    time.sleep(0.26)
    assert store.recover_expired()["queries_exhausted"] == 1
    assert store.claim("third-worker", 10) is None
    terminal = store.get(job["id"])
    assert terminal["state"] == "failed" and len(terminal["attempts_history"]) == 2
    assert all(item["cost_unknown"] is True for item in terminal["attempts_history"])


def test_expired_agent_is_indeterminate_and_never_retried(store):
    job, _ = submit(store, kind="agent", payload={"profile_id": "bound-profile"}, attempts=3)
    active = store.claim("worker", 0.2)
    assert not store.retry(job["id"], active["lease_token"], "unknown side effects", 0)
    time.sleep(0.26)
    recovered = store.recover_expired()
    assert recovered["agents_indeterminate"] == 1 and recovered["queries_requeued"] == 0
    assert store.get(job["id"])["state"] == "indeterminate"
    assert store.get(job["id"])["error"]["cost_unknown"] is True
    assert store.claim("other-worker", 10) is None
    assert not store.finish(job["id"], active["lease_token"], "completed")


def test_retry_preserves_error_evidence_delay_and_exhaustion(store):
    job, _ = submit(store, attempts=2)
    active = store.claim("worker", 10)
    error = {"code": "safe_retry", "attempt_evidence_ref": {"path": "attempt.json", "sha256": "a" * 64}, "notes": []}
    assert store.retry(job["id"], active["lease_token"], error, 0.2)
    assert store.claim("worker", 10) is None
    time.sleep(0.26)
    second = store.claim("worker", 10)
    assert second["attempts_history"][0]["error"] == error
    assert store.retry(job["id"], second["lease_token"], {"code": "again"}, 0)
    terminal = store.get(job["id"])
    assert terminal["state"] == "failed" and terminal["attempt"] == 2
    assert terminal["attempts_history"][1]["error"] == {"code": "again"}
    assert store.claim("worker", 10) is None


def test_active_jobs_and_idempotency_survive_retention_then_terminal_expires(store):
    short = RedisJobStore(store.client, namespace=store.namespace, retention_seconds=0.2)
    job, _ = submit(short)
    assert short.client.pttl(short.key_prefix + "job:" + job["id"]) == -1
    time.sleep(0.26)
    same, created = submit(short)
    assert same["id"] == job["id"] and not created
    active = short.claim("worker", 10)
    time.sleep(0.26)
    assert short.get(job["id"])["state"] == "running"
    assert submit(short)[1] is False
    assert short.finish(job["id"], active["lease_token"], "completed", result=[])
    time.sleep(0.26)
    assert short.get(job["id"]) is None
    next_job, created = submit(short)
    assert created and next_job["id"] != job["id"]


def test_capacity_includes_running_but_existing_idempotency_still_returns(store):
    limited = RedisJobStore(store.client, namespace=store.namespace, max_pending=1)
    job, _ = submit(limited)
    active = limited.claim("worker", 10)
    with pytest.raises(QueueFull):
        submit(limited, "another")
    assert submit(limited)[1] is False
    assert limited.finish(job["id"], active["lease_token"], "cancelled")
    assert submit(limited, "another")[1] is True


def test_concurrent_distinct_submissions_cannot_exceed_capacity(store):
    limited = RedisJobStore(store.client, namespace=store.namespace, max_pending=1)

    def admit(index):
        try:
            return submit(limited, f"distinct-{index}")
        except QueueFull:
            return None

    # 为什么要有它：同键去重不能证明不同请求的容量控制原子性；并发入队仍必须遵守同一上限。
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(admit, range(8)))
    accepted = [item for item in outcomes if item is not None]
    assert len(accepted) == 1 and accepted[0][1] is True
    assert limited.metrics()["pending"] == 1
    assert len(list(store.client.scan_iter(match=store.key_prefix + "job:*"))) == 1
    assert len(list(store.client.scan_iter(match=store.key_prefix + "idem:*"))) == 1


@pytest.mark.parametrize("changed", [
    {"kind": "agent"}, {"timeout": 21}, {"attempts": 3},
])
def test_idempotency_covers_execution_parameters_and_terminal_state(store, changed):
    job, _ = submit(store)
    running = store.claim("worker", 10)
    assert submit(store)[0]["id"] == job["id"]
    assert store.finish(job["id"], running["lease_token"], "completed", result={"answer": "retained"})
    retained, created = submit(store)
    assert not created and retained["state"] == "completed"
    assert retained["result"] == {"answer": "retained"}
    # 为什么要有它：终态重发不能重新排队，同幂等键也不能静默改变执行类型或预算。
    with pytest.raises(IdempotencyConflict):
        submit(store, **changed)
    assert store.claim("another-worker", 10) is None
    assert store.metrics()["pending"] == 0


def test_expired_recovery_is_bounded_and_preserves_remaining_work(store):
    # Scope: this 101-job recovery fixture only, before its namespace is initialized.
    # 为什么要有它：每任务生命周期预留后默认64MiB会先拒绝准备；显式测试额度不改变生产默认，也不绕过配额或原100+1恢复断言。
    store = RedisJobStore(store.client, namespace=store.namespace,
        retention_seconds=store.retention_seconds, max_pending=store.max_pending,
        capacity_bytes=2*1024**3, cache_capacity_bytes=store.cache_capacity_bytes,
        cache_ttl_seconds=store.cache_ttl_seconds)
    for index in range(101):
        submit(store, f"recovery-{index}", attempts=1)
    for index in range(101):
        assert store.claim(f"worker-{index}", 0.02) is not None
    time.sleep(0.04)
    # 为什么要有它：一次恢复有界不能吞掉超出批次的过期任务，后续调用必须能继续收口。
    first = store.recover_expired()
    assert first["expired"] == first["queries_exhausted"] == 100
    assert first["remaining_expired"] == 1
    second = store.recover_expired()
    assert second["expired"] == second["queries_exhausted"] == 1
    assert second["remaining_expired"] == 0
    assert store.metrics()["pending"] == 0


def test_rate_limit_atomic_and_identity_not_used_as_key(store):
    identity = "SECRET-IDENTITY user@example.test"
    with ThreadPoolExecutor(max_workers=8) as pool:
        decisions = list(pool.map(lambda _: store.rate_limit(identity, 3, 30), range(16)))
    assert sum(allowed for allowed, _ in decisions) == 3
    assert all(delay == 0 if allowed else 0 < delay <= 30 for allowed, delay in decisions)
    assert all(identity not in key for key in store.client.scan_iter(match=store.key_prefix + "*"))


def test_redis_server_error_is_not_success_or_empty_queue(store):
    # 为什么要有它：真实Redis的WRONGTYPE是存储故障，不能吞掉后伪称提交成功或空队列。
    store.client.set(store.key_prefix + "ready", "wrong-type")
    with pytest.raises(redis.RedisError):
        submit(store)
    assert not list(store.client.scan_iter(match=store.key_prefix + "idem:*"))


@pytest.mark.parametrize("invalid", [0, -1, float("inf"), float("nan"), True])
def test_invalid_numeric_contract_is_rejected_before_enqueue(store, invalid):
    with pytest.raises(ValueError):
        submit(store, timeout=invalid)
    with pytest.raises(ValueError):
        store.claim("worker", invalid)
    assert store.metrics()["pending"] == 0


def test_utf8_bounds_nonjson_keys_and_nonfinite_payload_are_rejected(store):
    for payload in ({"query": "界" * 50000}, {1: "not-string-key"}, {"value": float("nan")}):
        with pytest.raises(ValueError):
            submit(store, payload=payload)
    assert store.metrics()["pending"] == 0


def test_worker_can_record_timeout_while_lease_is_owned(store):
    job, _ = submit(store, timeout=0.1)
    active = store.claim("worker", 10)
    time.sleep(0.15)
    # 为什么要有它：执行超时由worker收尾，不能把仍持有租约的超时结果误当失租迟到写；失租测试独立覆盖。
    assert store.finish(job["id"], active["lease_token"], "timed_out", error={"code": "backend_timeout"})
    assert store.get(job["id"])["state"] == "timed_out"
