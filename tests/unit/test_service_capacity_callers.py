"""Capacity policy propagation and HTTP boundaries; no Redis/model execution."""
import pytest

from src.service_layer.config import ServiceConfig
from src.service_layer.worker import backend_configuration


def test_capacity_configuration_roundtrip_and_environment(monkeypatch):
    monkeypatch.setenv("SERVICE_CAPACITY_BYTES", "33554432")
    monkeypatch.setenv("SERVICE_CACHE_CAPACITY_BYTES", "4194304")
    config = ServiceConfig.from_env()
    assert config.capacity_bytes == 33554432
    assert config.cache_capacity_bytes == 4194304
    assert ServiceConfig.from_dict(config.to_dict()) == config


def test_worker_passes_exact_capacity_policy_to_child(tmp_path):
    config = ServiceConfig(work_root=tmp_path, capacity_bytes=33554432,
                           cache_capacity_bytes=4194304, retention_seconds=120,
                           max_pending=12, query_cache_ttl=45, api_token="not-for-child")
    child = backend_configuration(config)
    assert {name: child[name] for name in (
        "capacity_bytes", "cache_capacity_bytes", "retention_seconds", "max_pending", "query_cache_ttl"
    )} == {"capacity_bytes":33554432, "cache_capacity_bytes":4194304,
          "retention_seconds":120, "max_pending":12, "query_cache_ttl":45}
    assert "api_token" not in child and "redis_url" not in child


def test_capacity_metrics_are_encoded_data_only_and_finite():
    from src.service_layer.metrics import queue_metrics
    text = queue_metrics({"capacity":{"capacity_bytes":67108864, "cache_capacity_bytes":16777216,
        "charged_bytes":123, "cache_charged_bytes":5, "dirty":False, "initialized":True,
        "PRIVATE_BODY":"do not export"}}).decode()
    assert "service_encoded_charged_bytes 123" in text
    assert "service_encoded_cache_charged_bytes 5" in text
    assert "service_capacity_dirty 0" in text and "service_capacity_initialized 1" in text
    assert "PRIVATE_BODY" not in text and "do not export" not in text and "rss" not in text
    invalid = queue_metrics({"capacity":{"charged_bytes":True, "capacity_bytes":float("inf"),
        "cache_charged_bytes":-1, "dirty":"yes"}}).decode()
    assert "service_encoded_" not in invalid and "service_capacity_dirty" not in invalid


@pytest.mark.parametrize("overrides", [
    {"capacity_bytes": True}, {"capacity_bytes": 0}, {"capacity_bytes": 1.5},
    {"capacity_bytes": 2**40+1}, {"cache_capacity_bytes": False},
    {"cache_capacity_bytes": 0}, {"cache_capacity_bytes": 1.5},
    {"capacity_bytes": 8191, "cache_capacity_bytes": 100},
    {"capacity_bytes": 8192, "cache_capacity_bytes": 8193},
])
def test_invalid_capacity_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        ServiceConfig(**overrides)


def test_api_constructor_passes_exact_shared_policy(tmp_path, monkeypatch):
    from src.service_layer import api
    from tests.unit.test_service_http_api import Store

    config = ServiceConfig(work_root=tmp_path, api_token="test-only",
        capacity_bytes=33554432, cache_capacity_bytes=4194304,
        retention_seconds=120, max_pending=12, query_cache_ttl=45)
    calls = []
    def construct(client, **kwargs):
        calls.append(kwargs)
        return Store()
    monkeypatch.setattr(api.redis.Redis, "from_url", lambda *args, **kwargs: object())
    monkeypatch.setattr(api, "RedisJobStore", construct)
    api.create_app(config)
    assert calls == [{"namespace":config.namespace, "retention_seconds":120,
        "max_pending":12, "capacity_bytes":33554432, "cache_capacity_bytes":4194304,
        "cache_ttl_seconds":45}]


@pytest.mark.parametrize("stage", ["rate_limit", "submit"])
@pytest.mark.parametrize("kind,status", [
    ("CapacityExceeded",429), ("CapacityPolicyMismatch",503), ("CapacityStateError",503),
])
def test_api_does_not_accept_capacity_failures(tmp_path, stage, kind, status):
    from fastapi.testclient import TestClient

    from src.service_layer import store as store_module
    from src.service_layer.api import create_app
    from tests.unit.test_service_http_api import HEADERS, Store

    settings = tmp_path / "settings.yaml"
    settings.write_text("query: {}\n", encoding="utf-8")
    config = ServiceConfig(settings_path=settings, api_token="unit-test-only")
    store = Store()
    def reject(*_args):
        raise getattr(store_module, kind)("controlled-capacity-test")
    setattr(store, stage, reject)
    with TestClient(create_app(config, store=store)) as client:
        response = client.post("/v1/jobs", headers=HEADERS,
                               json={"kind":"query", "payload":{"query":"test"}})
        assert response.status_code == status
        assert response.headers["retry-after"] == "1"
        assert "controlled-capacity-test" not in response.text
        assert client.get("/v1/jobs/"+"a"*32, headers=HEADERS).status_code == 404
    assert store.calls == []


@pytest.mark.parametrize("kind", ["CapacityExceeded", "CapacityPolicyMismatch", "CapacityStateError"])
def test_worker_capacity_failures_stop_without_retry_and_release_resources(tmp_path, monkeypatch, capsys, kind):
    from src.service_layer import store as store_module
    from src.service_layer import worker
    from tests.unit.test_service_worker_shutdown import Store, setup_main

    class RefusedStore(Store):
        def recover_expired(self):
            raise getattr(store_module, kind)("controlled-capacity-test")
    store = RefusedStore()
    events = setup_main(monkeypatch, tmp_path, store)
    assert worker.main([]) == 2
    assert store.claims == 0 and not store.finished and not store.retried
    assert events[-2:] == ["runner_closed", "client_closed"]
    error = capsys.readouterr().err
    assert '"retry": false' in error
    assert "controlled-capacity-test" not in error
