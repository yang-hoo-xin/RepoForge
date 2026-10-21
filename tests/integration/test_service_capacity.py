"""Encoded-byte admission checks using the existing real Redis fixture.

Scope: synthetic jobs and fixture-owned namespaces, never business execution.
为什么要有它：并发预留、Lua中途失败和TTL释放必须由真实Redis证明，不能用内存替身或物理内存读数代替。
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from redis import ConnectionPool, Redis
from redis.exceptions import RedisError

from src.service_layer.api import create_app
from src.service_layer.capacity import BASE_BYTES
from src.service_layer.config import ServiceConfig
from src.service_layer.store import (
    CapacityExceeded,
    CapacityPolicyMismatch,
    CapacityStateError,
    RedisJobStore,
)
from tests.integration import test_service_job_store as store_fixtures

pytestmark = pytest.mark.integration
store = store_fixtures.store
MIB = 1024**2


@pytest.fixture
def capacity_factory(store):
    owned = []

    def make(**options):
        value = RedisJobStore(store.client, namespace=store.namespace + "-" + str(len(owned)), **options)
        owned.append(value)
        return value

    try:
        yield make
    finally:
        # Scope: exact randomly allocated fixture namespaces, including their legacy cache prefix.
        # 为什么要有它：校准与反例需要不同policy，但清理不能flush共享Redis或触及既有服务数据。
        for owner in owned:
            prefixes = (owner.key_prefix, owner.namespace + ":query-cache:")
            keys = [key for prefix in prefixes for key in owner.client.scan_iter(match=prefix + "*")]
            assert all(any(key.startswith(prefix) for prefix in prefixes) for key in keys)
            if keys:
                owner.client.delete(*keys)


def _capacity(owner):
    value = owner.metrics()["capacity"]
    assert 0 <= value["cache_charged_bytes"] <= value["charged_bytes"] <= value["capacity_bytes"]
    assert value["cache_charged_bytes"] <= value["cache_capacity_bytes"]
    return value


def _submit(owner, key="synthetic", *, attempts=1):
    return owner.submit("query", {"query": "capacity-only synthetic input"}, key, 30, attempts)


def _reservation(capacity_factory, *, attempts=1):
    calibration = capacity_factory()
    assert _capacity(calibration)["charged_bytes"] == 0
    _submit(calibration, attempts=attempts)
    charged = _capacity(calibration)["charged_bytes"]
    assert charged > 0
    return charged


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@contextmanager
def _api(owner, tmp_path):
    settings = tmp_path / (uuid4().hex + ".yaml")
    settings.write_text("query: {}\n", encoding="utf-8")
    config = ServiceConfig(settings_path=settings, api_token=uuid4().hex, requests_per_minute=1000)
    with TestClient(create_app(config, store=owner)) as client:
        yield client, {"Authorization": "Bearer " + config.api_token}


def _api_post(client, auth):
    return client.post("/v1/jobs", headers={**auth, "Idempotency-Key": "api-new"},
                       json={"kind": "query", "payload": {"query": "capacity-only API input"}})


def test_concurrent_submit_cannot_double_spend_last_lifecycle_reservation(capacity_factory, tmp_path):
    reservation = _reservation(capacity_factory)
    owner = capacity_factory(capacity_bytes=reservation, cache_capacity_bytes=reservation)
    barrier = Barrier(8)

    def admit(index):
        barrier.wait(timeout=5)
        try:
            return _submit(owner, f"request-{index}")
        except CapacityExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(admit, range(8)))
    accepted = [item for item in outcomes if item is not None]
    assert len(accepted) == 1 and accepted[0][1] is True
    assert _capacity(owner)["charged_bytes"] == reservation
    assert owner.metrics()["pending"] == 1
    assert len(list(owner.client.scan_iter(match=owner.key_prefix + "job:*"))) == 1
    assert len(list(owner.client.scan_iter(match=owner.key_prefix + "idem:*"))) == 1
    with _api(owner, tmp_path) as (client, auth):
        rejected = _api_post(client, auth)
        assert rejected.status_code == 429 and "retry-after" in rejected.headers
        readable = client.get("/v1/jobs/" + accepted[0][0]["id"], headers=auth)
        assert readable.status_code == 200
    assert _capacity(owner)["charged_bytes"] == reservation


def test_reserved_ten_attempt_job_finishes_after_cache_fills_with_escaped_failed_result(capacity_factory):
    reservation = _reservation(capacity_factory, attempts=10)
    owner = capacity_factory(capacity_bytes=reservation + 512 * 1024, cache_capacity_bytes=512 * 1024)
    job, _ = _submit(owner, attempts=10)
    cached = []
    for index in range(40):
        digest, encoded = _digest(str(index)), json.dumps("x" * (16 * 1024))
        try:
            owner.cache_put(digest, encoded)
        except CapacityExceeded:
            break
        cached.append((digest, encoded))
    else:
        pytest.fail("The configured cache sub-budget never rejected further entries")
    assert cached
    full = _capacity(owner)
    assert full["charged_bytes"] > reservation
    worker_id = "worker" + '\x00"\\\n\t' * 40
    error = {"detail": '\x00"\\\n' * 5000}
    result = {"detail": '\x00"\\\n' * 170000}
    assert len(json.dumps(error, separators=(",", ":")).encode()) <= 64 * 1024
    assert MIB < len(json.dumps(result, separators=(",", ":")).encode()) <= 2 * MIB
    for attempt in range(1, 11):
        active = owner.claim(worker_id, 30)
        assert active["id"] == job["id"] and active["attempt"] == attempt
        assert owner.heartbeat(job["id"], active["lease_token"], 30)
        if attempt < 10:
            assert owner.retry(job["id"], active["lease_token"], error, 0)
            assert _capacity(owner)["charged_bytes"] == full["charged_bytes"]
        else:
            assert owner.finish(job["id"], active["lease_token"], "failed", result=result, error=error)
    terminal = owner.get(job["id"])
    assert terminal["state"] == "failed" and len(terminal["attempts_history"]) == 10
    assert all(row["worker_id"] == worker_id and row["error"] == error for row in terminal["attempts_history"])
    assert terminal["result"] == terminal["attempts_history"][-1]["result"] == result
    raw = json.loads(owner.client.get(owner.key_prefix + "job:" + job["id"]))
    assert raw["result_json"] == raw["attempts_history"][-1]["result_json"]
    assert "_result_from_job" not in raw["attempts_history"][-1]
    settled = _capacity(owner)
    assert settled["charged_bytes"] < full["charged_bytes"]
    assert settled["cache_charged_bytes"] == full["cache_charged_bytes"]
    assert all(owner.cache_get(digest) == encoded for digest, encoded in cached)
    assert not owner.finish(job["id"], active["lease_token"], "completed", result="late")
    assert _capacity(owner)["charged_bytes"] == settled["charged_bytes"]


def test_cache_and_rate_share_total_budget_without_bypassing_single_entry_limit(capacity_factory):
    calibration = capacity_factory()
    digest, encoded = _digest("one"), json.dumps("x" * 128)
    calibration.cache_put(digest, encoded)
    cache_cost = _capacity(calibration)["charged_bytes"]
    assert calibration.rate_limit("one", 100, 60)[0]
    total = _capacity(calibration)["charged_bytes"]
    assert total > cache_cost > 0
    owner = capacity_factory(capacity_bytes=total, cache_capacity_bytes=total)
    owner.cache_put(digest, encoded)
    assert owner.rate_limit("one", 100, 60)[0]
    assert _capacity(owner)["charged_bytes"] == total
    for operation in (lambda: owner.cache_put(_digest("two"), encoded),
                      lambda: owner.rate_limit("two", 100, 60)):
        with pytest.raises(CapacityExceeded):
            operation()
        assert _capacity(owner)["charged_bytes"] == total
    assert owner.rate_limit("one", 100, 60)[0]
    assert owner.cache_get(digest) == encoded
    roomy = capacity_factory()
    with pytest.raises(CapacityExceeded) as rejected:
        roomy.cache_put(_digest("too-large"), json.dumps("x" * (2 * MIB)))
    assert rejected.value.code == "SERVICE_CACHE_ENTRY_TOO_LARGE"
    assert roomy.cache_get(_digest("too-large")) is None
    assert _capacity(roomy)["charged_bytes"] == 0


def test_expired_capacity_is_ttl_checked_bounded_and_released_once(capacity_factory):
    owner = capacity_factory(cache_ttl_seconds=2)
    for index in range(101):
        owner.cache_put(_digest(str(index)), '"retained"')
    before = _capacity(owner)
    deadline = time.monotonic() + 6
    while any(owner.client.exists(owner.cache_key(_digest(str(index)))) for index in range(101)):
        if time.monotonic() >= deadline:
            pytest.fail("Real Redis cache TTL did not expire within the bounded wait")
        time.sleep(0.02)
    assert _capacity(owner)["charged_bytes"] == before["charged_bytes"]
    first = owner.reclaim_expired_capacity()
    assert first["released"] == 100 and first["remaining_expired"] == 1
    assert 0 < _capacity(owner)["charged_bytes"] < before["charged_bytes"]
    second = owner.reclaim_expired_capacity()
    assert second["released"] == 1 and second["remaining_expired"] == 0
    third = owner.reclaim_expired_capacity()
    assert third["released"] == third["remaining_expired"] == 0
    assert _capacity(owner)["charged_bytes"] == BASE_BYTES == 8192
    assert _capacity(owner)["cache_charged_bytes"] == 0

    digest = _digest("still-live")
    owner.cache_put(digest, '"not expired"')
    charged = _capacity(owner)["charged_bytes"]
    # Scope: corrupt only this synthetic expiry hint, leaving the actual Redis TTL untouched.
    # 为什么要有它：索引分数到期不等于数据已过期；先核原key的TTL才能返还预算。
    owner.client.zadd(owner.key_prefix + "capacity-expiry", {"c:" + digest: 0})
    assert owner.client.pttl(owner.cache_key(digest)) > 0
    assert owner.reclaim_expired_capacity()["released"] == 0
    assert _capacity(owner)["charged_bytes"] == charged
    assert owner.cache_get(digest) == '"not expired"'


@pytest.mark.parametrize("policy_fault", ["legacy", "mismatch"])
def test_legacy_or_different_policy_rejects_api_writes_but_preserves_get(capacity_factory, tmp_path, policy_fault):
    source = capacity_factory()
    job, _ = _submit(source)
    if policy_fault == "legacy":
        owner = capacity_factory()
        original = source.client.get(source.key_prefix + "job:" + job["id"])
        owner.client.set(owner.key_prefix + "job:" + job["id"], original)
        error_type = CapacityStateError
    else:
        owner = RedisJobStore(source.client, namespace=source.namespace, capacity_bytes=65 * MIB)
        error_type = CapacityPolicyMismatch
    original = owner.client.get(owner.key_prefix + "job:" + job["id"])
    with pytest.raises(error_type):
        _submit(owner, "must-not-write")
    with _api(owner, tmp_path) as (client, auth):
        response = client.get("/v1/jobs/" + job["id"], headers=auth)
        assert response.status_code == 200 and response.json()["id"] == job["id"]
        rejected = _api_post(client, auth)
        assert rejected.status_code == 503 and "retry-after" in rejected.headers
    assert owner.client.get(owner.key_prefix + "job:" + job["id"]) == original
    assert len(list(owner.client.scan_iter(match=owner.key_prefix + "job:*"))) == 1


def test_real_partial_write_fault_keeps_dirty_charge_and_readable_jobs(capacity_factory, tmp_path):
    owner = capacity_factory()
    retained, _ = _submit(owner, "retained")
    original = owner.client.get(owner.key_prefix + "job:" + retained["id"])
    charge_before = _capacity(owner)["charged_bytes"]
    # Scope: a temporary ACL user, restricted to this fixture namespace and required commands.
    # 为什么要有它：类型错误本应写前发现；合法类型下最后HINCRBY真实被拒，才能检验中途失败不回滚、不返还额度。
    username, password = "capacity-test-" + uuid4().hex, uuid4().hex
    assert owner.client.acl_getuser(username) is None
    restricted_client = None
    try:
        commands = ("evalsha", "eval", "script|load", "get", "set", "persist", "pexpire", "time",
                    "hget", "hgetall", "hset", "hdel", "type", "scan", "zrangebyscore", "zcount",
                    "zcard", "zscore", "zadd", "zrem", "exists", "pttl", "ping", "client|setinfo")
        owner.client.execute_command("ACL", "SETUSER", username, "reset", "on", ">" + password,
                                     "~" + owner.key_prefix + "*", *("+" + command for command in commands))
        connection_options = dict(owner.client.connection_pool.connection_kwargs)
        connection_options.update(username=username, password=password)
        restricted_client = Redis(connection_pool=ConnectionPool(**connection_options))
        restricted = RedisJobStore(restricted_client, namespace=owner.namespace)
        assert owner.client.type(owner.key_prefix + "metrics") == "hash"
        with pytest.raises(RedisError):
            _submit(restricted, "partial-write")
    finally:
        if restricted_client is not None:
            restricted_client.close()
            restricted_client.connection_pool.disconnect()
        owner.client.acl_deluser(username)
        assert owner.client.acl_getuser(username) is None
    capacity_key = owner.key_prefix + "capacity"
    state = owner.client.hgetall(capacity_key)
    assert state["dirty"] and int(state["used"]) >= charge_before > 0
    keys = list(owner.client.scan_iter(match=owner.key_prefix + "job:*"))
    assert len(keys) == 2
    assert all(owner.get(key.rsplit(":", 1)[-1]) is not None for key in keys)
    for operation in (lambda: _submit(owner, "blocked"),
                      lambda: owner.cache_put(_digest("blocked"), "{}"),
                      lambda: owner.rate_limit("blocked", 100, 60),
                      owner.reclaim_expired_capacity):
        with pytest.raises(CapacityStateError):
            operation()
        assert owner.client.hgetall(capacity_key) == state
    with _api(owner, tmp_path) as (client, auth):
        readable = client.get("/v1/jobs/" + retained["id"], headers=auth)
        assert readable.status_code == 200 and readable.json()["id"] == retained["id"]
        assert _api_post(client, auth).status_code == 503
    assert owner.client.get(owner.key_prefix + "job:" + retained["id"]) == original
    assert owner.client.hgetall(capacity_key) == state
