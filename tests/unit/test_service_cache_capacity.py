"""Cache adapter capacity-owner boundary; no Redis server or business provider."""
import hashlib
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.core.query_engine.result_cache import canonical, fingerprint
from src.core.settings import QuerySettings
from src.service_layer import backends


class MemoryRedis:
    def __init__(self):
        self.values, self.owner_writes, self.reads = {}, [], []

    def ping(self):
        return True

    def get(self, key):
        self.reads.append(key)
        return self.values.get(key)

    def set(self, *_args, **_kwargs):
        raise AssertionError("Cache adapter bypassed the capacity owner")


@pytest.fixture
def owner(monkeypatch):
    instances = []

    class Owner:
        def __init__(self, client, **policy):
            self.client, self.policy, self.error = client, policy, None
            instances.append(self)

        def cache_key(self, digest):
            prefix = hashlib.sha256(self.policy["namespace"].encode()).hexdigest()
            return "agent-service:{" + prefix + "}:cache:" + digest

        def cache_get(self, digest):
            raw = self.client.get(self.cache_key(digest))
            return self.client.get(self.policy["namespace"] + ":query-cache:" + digest) if raw is None else raw

        def cache_put(self, digest, encoded):
            if self.error is not None:
                raise self.error
            self.client.owner_writes.append((digest, encoded, self.policy["cache_ttl_seconds"]))
            self.client.values[self.cache_key(digest)] = encoded

    monkeypatch.setattr(backends, "RedisJobStore", Owner, raising=False)
    return instances


def test_cache_write_uses_capacity_owner_and_keeps_integrity(owner):
    client = MemoryRedis()
    cache = backends.RedisQueryCache(client, "capacity-adapter", 45)
    binding, value = {"index": "v1"}, {"answer": ["unchanged"]}
    cache.put(binding, value)
    assert len(owner) == 1
    assert owner[0].policy == {"namespace": "capacity-adapter", "retention_seconds": 86400,
        "max_pending": 1000, "capacity_bytes": 64 * 1024**2,
        "cache_capacity_bytes": 16 * 1024**2, "cache_ttl_seconds": 45}
    assert len(client.owner_writes) == 1
    assert cache.get(binding) == value
    copied = cache.get(binding)
    copied["answer"].append("not retained")
    assert cache.get(binding) == value


def test_legacy_cache_read_is_compatible_without_writing_or_promotion(owner):
    client = MemoryRedis()
    binding, value = {"index": "old"}, {"answer": "old result"}
    legacy_key = "capacity-adapter:query-cache:" + fingerprint(binding)
    client.values[legacy_key] = canonical({"version": "query_model_cache_v1", "binding": binding,
        "value": value, "value_sha256": fingerprint(value)})
    before = dict(client.values)
    cache = backends.RedisQueryCache(client, "capacity-adapter", 45)
    assert cache.get(binding) == value
    assert len(owner) == 1 and client.values == before and not client.owner_writes
    assert client.reads == [owner[0].cache_key(fingerprint(binding)), legacy_key]


def test_run_query_passes_complete_nondefault_capacity_policy(tmp_path, monkeypatch, owner):
    @dataclass(frozen=True)
    class Settings:
        query: QuerySettings

    settings = tmp_path / "settings.yaml"
    settings.write_text("controlled settings", encoding="utf-8")
    config = SimpleNamespace(settings_path=settings, namespace="custom-policy", redis_url="redis://unused",
        query_cache_ttl=79, retention_seconds=987, max_pending=7,
        capacity_bytes=40 * 1024**2, cache_capacity_bytes=3 * 1024**2)
    monkeypatch.setattr("src.core.settings.load_settings", lambda _path: Settings(QuerySettings()))
    client = MemoryRedis()
    service = SimpleNamespace()

    def query(_request):
        service.cache.put({"index": "fixed"}, {"value": "fixed"})
        return SimpleNamespace(to_dict=lambda: {"status": "completed"})

    service.query = query
    result = backends.run_query({"query": "fixture", "_settings_sha256": hashlib.sha256(settings.read_bytes()).hexdigest()},
        config, job_id="job", timeout_seconds=5, redis_client=client,
        service_factory=lambda *_args, **_kwargs: service)
    assert result == {"status": "completed"}
    assert owner[0].policy == {"namespace": "custom-policy", "retention_seconds": 987, "max_pending": 7,
        "capacity_bytes": 40 * 1024**2, "cache_capacity_bytes": 3 * 1024**2, "cache_ttl_seconds": 79}


@pytest.mark.parametrize("error_type,code", [
    (backends.CapacityExceeded, "SERVICE_CAPACITY_EXCEEDED"),
    (backends.CapacityExceeded, "SERVICE_CACHE_CAPACITY_EXCEEDED"),
    (backends.CapacityExceeded, "SERVICE_CACHE_ENTRY_TOO_LARGE"),
    (backends.CapacityPolicyMismatch, "SERVICE_CAPACITY_POLICY_MISMATCH"),
    (backends.CapacityStateError, "SERVICE_CAPACITY_DIRTY"),
    (backends.CapacityStateError, "SERVICE_CAPACITY_LEGACY_DATA"),
])
def test_capacity_rejection_is_io_failure_without_direct_write(owner, error_type, code):
    client = MemoryRedis()
    cache = backends.RedisQueryCache(client, "capacity-adapter", 45)
    assert cache.get({}) is None
    rejected = error_type(code)
    owner[0].error = rejected
    with pytest.raises(OSError, match="QUERY_CACHE_WRITE_UNAVAILABLE") as observed:
        cache.put({}, {"value": "unchanged"})
    assert observed.value.__cause__ is rejected and not client.owner_writes


@pytest.mark.parametrize("error_type", [backends.CapacityExceeded, backends.CapacityPolicyMismatch,
                                       backends.CapacityStateError])
def test_final_query_retained_when_capacity_owner_refuses(tmp_path, owner, error_type):
    from tests.unit.test_query_service import make_service, request

    client = MemoryRedis()
    service, model, _, calls, _ = make_service(tmp_path)
    service.cache = backends.RedisQueryCache(client, "capacity-adapter", 45)
    assert service.cache.get({}) is None
    owner[0].error = error_type("controlled capacity refusal")
    response = service.query(request(answer=False, rewrite=False, rerank=False)).to_dict()
    assert response["status"] == "completed" and response["results"]
    assert response["warnings"] == ["complete_result_cache_write_failed"]
    assert "result" not in response["cache"] and not client.owner_writes
    assert len(calls) == 2 and not model.calls


def test_corrupt_new_record_does_not_fall_back_to_a_legacy_success(owner):
    client = MemoryRedis()
    binding, value = {"index": "old"}, {"answer": "old result"}
    cache = backends.RedisQueryCache(client, "capacity-adapter", 45)
    assert cache.get(binding) is None
    digest = fingerprint(binding)
    client.values["capacity-adapter:query-cache:" + digest] = canonical({
        "version": "query_model_cache_v1", "binding": binding, "value": value,
        "value_sha256": fingerprint(value)})
    client.values[owner[0].cache_key(digest)] = canonical({"version": "query_model_cache_v1",
        "binding": binding, "value": value, "value_sha256": "0" * 64})
    client.reads.clear()
    assert cache.get(binding) is None
    assert client.reads == [owner[0].cache_key(digest)] and not client.owner_writes


def test_invalid_value_is_rejected_before_capacity_owner_is_created(owner):
    cache = backends.RedisQueryCache(MemoryRedis(), "capacity-adapter", 45)
    with pytest.raises(TypeError):
        cache.put({}, {"value": object()})
    assert owner == []


def test_capacity_owner_construction_refusal_cannot_bypass_policy(monkeypatch):
    failure = backends.CapacityStateError("controlled initializing state")

    def rejected(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(backends, "RedisJobStore", rejected)
    cache = backends.RedisQueryCache(MemoryRedis(), "capacity-adapter", 45)
    with pytest.raises(OSError) as observed:
        cache.put({}, {"value": "unchanged"})
    assert observed.value.__cause__ is failure
