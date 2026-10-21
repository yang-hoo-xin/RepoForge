"""Pure interface/bound checks; real Lua behavior is separately qualified in Redis."""
import json

import pytest
from redis.exceptions import ResponseError

from src.service_layer.capacity import BASE_BYTES, ITEM_METADATA_BYTES, reservation_bytes
from src.service_layer.store import (
    CapacityExceeded,
    CapacityPolicyMismatch,
    CapacityStateError,
    RedisJobStore,
)


class CapturingRedis:
    def __init__(self):
        self.calls = []
        self.values = {}

    def register_script(self, source):
        self.source = source
        def script(**kwargs):
            self.calls.append(kwargs)
            return json.dumps({"code": "accepted"})
        return script

    def get(self, key):
        return self.values.get(key)


def test_capacity_policy_is_explicit_and_cache_cannot_bypass_registered_script():
    client = CapturingRedis()
    owner = RedisJobStore(client, namespace="core", capacity_bytes=64*1024**2,
        cache_capacity_bytes=16*1024**2, cache_ttl_seconds=60)
    owner.cache_put("a"*64, '{"value":"controlled"}')
    assert len(client.calls) == 1 and client.calls[0]["args"][0] == "cache_put"
    assert owner.cache_key("a"*64).startswith(owner.key_prefix)


def test_legacy_cache_is_read_only_and_does_not_initialize_policy():
    client = CapturingRedis()
    client.values["core:query-cache:"+"b"*64] = "legacy"
    owner = RedisJobStore(client, namespace="core")
    assert owner.cache_get("b"*64) == "legacy"
    assert client.calls == []


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_capacity_requires_explicit_positive_integer_bytes(invalid):
    with pytest.raises(ValueError):
        RedisJobStore(CapturingRedis(), capacity_bytes=invalid)


def test_policy_binds_only_explicit_capacity_and_ttl_contract_and_keeps_integer_precision():
    owner = RedisJobStore(CapturingRedis(), retention_seconds=.2, cache_ttl_seconds=1,
        capacity_bytes=2**40, cache_capacity_bytes=2**40)
    assert json.loads(owner.capacity_policy) == {"version": "service-capacity.v1", "capacity_bytes": 2**40,
        "cache_capacity_bytes": 2**40, "retention_ms": 200, "cache_ttl_ms": 1000, "cache_entry_bytes": 2*1024**2}
    for options in ({"capacity_bytes": 2**40+1}, {"capacity_bytes": BASE_BYTES-1},
                    {"cache_capacity_bytes": 65*1024**2}, {"cache_ttl_seconds": True}):
        with pytest.raises(ValueError):
            RedisJobStore(CapturingRedis(), **options)


def test_cache_byte_limit_uses_utf8_and_rejects_before_any_redis_send():
    client = CapturingRedis()
    owner = RedisJobStore(client)
    exact = "中"*(2*1024**2//3)+"x"*(2*1024**2%3)
    owner.cache_put("a"*64, exact)
    assert len(client.calls) == 1
    with pytest.raises(CapacityExceeded, match="SERVICE_CACHE_ENTRY_TOO_LARGE"):
        owner.cache_put("a"*64, exact+"x")
    assert len(client.calls) == 1


@pytest.mark.parametrize("code,expected", [
    ("SERVICE_CAPACITY_EXCEEDED", CapacityExceeded),
    ("SERVICE_CACHE_CAPACITY_EXCEEDED", CapacityExceeded),
    ("SERVICE_CAPACITY_POLICY_MISMATCH", CapacityPolicyMismatch),
    ("SERVICE_CAPACITY_DIRTY", CapacityStateError),
    ("SERVICE_CAPACITY_INITIALIZATION_SCAN_LIMIT", CapacityStateError),
    ("SERVICE_CAPACITY_KEY_TYPE_INVALID", CapacityStateError),
])
def test_only_fixed_capacity_codes_are_classified_without_retry(code, expected):
    client = CapturingRedis()
    owner = RedisJobStore(client)
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise ResponseError(code+" script: retained-sha, on @user_script:12.")
    owner._script = fail
    with pytest.raises(expected) as result:
        owner.cache_put("a"*64, "safe")
    assert result.value.code == code and len(calls) == 1


def test_unknown_physical_redis_failure_is_not_misreported_as_quota_rejection():
    owner = RedisJobStore(CapturingRedis())
    original = ResponseError("OOM command not allowed when used memory > maxmemory")
    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise original
    owner._script = fail
    with pytest.raises(ResponseError) as result:
        owner.cache_put("a"*64, "safe")
    assert result.value is original and len(calls) == 1


@pytest.mark.parametrize("attempts", [1, 3, 10])
def test_reservation_covers_encoded_failed_results_history_and_maximum_control_worker(attempts):
    # Scope: arithmetic/string-encoding bound only, not simulated Lua or actual RSS savings.
    # 为什么要有它：双份结果和每次原错误都会再作为JSON字符串转义，普通小样本不能检验此最坏路径。
    def text(limit):
        return json.dumps("\\"*((limit-2)//2), ensure_ascii=False, separators=(",", ":"))
    payload, result, error = text(128*1024), text(2*1024**2), text(64*1024)
    worker = "\x00"*256
    history = [{"attempt": number, "state": "failed" if number == attempts else "retry_scheduled",
        "started_at": 1792000000.123, "finished_at": 1792000001.123,
        "lease_token": "f"*32, "worker_id": worker, "result_json": result if number == attempts else "null",
        "error_json": error, "cost_unknown": None} for number in range(1, attempts+1)]
    job = {"id": "a"*32, "kind": "query", "state": "failed", "payload_json": payload,
        "result_json": result, "error_json": error, "attempt": attempts, "max_attempts": attempts,
        "timeout_seconds": 86400, "created_at": 1792000000.123, "started_at": 1792000000.123,
        "finished_at": 1792000001.123, "available_at": 1792000000.123, "lease_token": None,
        "lease_until": None, "attempt_deadline_at": 1792086400.123, "worker_id": worker,
        "attempts_history": history, "_request_hash": "b"*64, "_idempotency_digest": "c"*64}
    encoded = json.dumps(job, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(encoded)+ITEM_METADATA_BYTES+512 < reservation_bytes(len(payload.encode()), attempts)
