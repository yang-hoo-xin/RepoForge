"""Real Redis adapter exceptions, fake transport only; no Redis server or provider."""
import hashlib
import json
from dataclasses import dataclass

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import DataError, OutOfMemoryError, RedisError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.core.settings import QuerySettings
from src.service_layer.backends import BackendError, RedisQueryCache, run_query
from tests.unit.test_query_service import make_service, request
from tests.unit.test_service_worker import cache_owner_boundary as cache_owner_boundary
from tests.unit.test_service_worker import config


class RejectingRedis:
    def __init__(self, error):
        self.error, self.writes = error, []

    def get(self, key):
        return None

    def set(self, key, value, **options):
        self.writes.append((key, value, options))
        raise self.error


def test_final_cache_oom_retains_completed_query_and_true_warning(tmp_path):
    client = RejectingRedis(OutOfMemoryError("controlled Redis SET refusal"))
    service, model, _, calls, _ = make_service(tmp_path)
    service.cache = RedisQueryCache(client, "controlled-cache-error", 30)
    events = []
    response = service.query(request(answer=False, rewrite=False, rerank=False), on_event=events.append).to_dict()
    assert response["status"] == "completed" and response["results"]
    assert response["warnings"] == ["complete_result_cache_write_failed"]
    assert "result" not in response["cache"]  # No successful hit/write marker is fabricated.
    assert len(client.writes) == 1 and client.writes[0][2] == {"ex": 30}
    assert len(calls) == 2 and not model.calls
    assert events[-1]["kind"] == "final" and events[-1]["response"] == response
    assert service._requests[response["request_id"]]["response"] == response


@pytest.mark.parametrize("error_type", [OutOfMemoryError, RedisConnectionError, RedisTimeoutError, ResponseError])
def test_adapter_maps_only_redis_write_error_to_existing_io_contract(error_type):
    original = error_type("controlled storage refusal")
    cache = RedisQueryCache(RejectingRedis(original), "controlled-cache-error", 30)
    with pytest.raises(OSError, match="QUERY_CACHE_WRITE_UNAVAILABLE") as observed:
        cache.put({"index": "fixture"}, {"answer": "not a model output"})
    assert observed.value.__cause__ is original


@pytest.mark.parametrize("error_type", [TypeError, ValueError, RuntimeError, DataError, RedisError])
def test_adapter_does_not_translate_programming_exceptions(error_type):
    original = error_type("controlled programming failure")
    cache = RedisQueryCache(RejectingRedis(original), "controlled-cache-error", 30)
    with pytest.raises(error_type) as observed:
        cache.put({"index": "fixture"}, {"value": "fixture"})
    assert observed.value is original


def test_final_cache_sdk_data_error_is_not_hidden_as_warning(tmp_path):
    original = DataError("controlled SDK argument error")
    service, model, _, _, _ = make_service(tmp_path)
    service.cache = RedisQueryCache(RejectingRedis(original), "controlled-cache-error", 30)
    with pytest.raises(DataError) as observed:
        service.query(request(answer=False, rewrite=False, rerank=False))
    assert observed.value is original and not model.calls


def test_invalid_cache_value_is_not_sent_to_redis():
    client = RejectingRedis(OutOfMemoryError("must not be reached"))
    with pytest.raises(TypeError):
        RedisQueryCache(client, "controlled-cache-error", 30).put({}, {"value": object()})
    assert client.writes == []


def test_cache_read_error_keeps_existing_failure_path(tmp_path):
    original = RedisConnectionError("controlled read error")
    client = RejectingRedis(original)
    def get(key):
        raise original
    client.get = get
    cache = RedisQueryCache(client, "controlled-cache-error", 30)
    with pytest.raises(RedisConnectionError) as observed:
        cache.get({"index": "fixture"})
    assert observed.value is original
    service, model, _, calls, _ = make_service(tmp_path)
    service.cache = cache
    result = service.query(request(answer=False, rewrite=False, rerank=False)).to_dict()
    assert result["status"] == "failed" and not calls and not model.calls and not client.writes


@pytest.mark.parametrize("operation", ["rewrite_v2", "answer_v3_original_source_context"])
def test_intermediate_cache_write_still_raises_for_owner(operation):
    cache = RedisQueryCache(RejectingRedis(OutOfMemoryError("controlled")), "controlled-cache-error", 30)
    with pytest.raises(OSError):
        cache.put({"operation": operation}, {"value": "fixture"})


@pytest.mark.parametrize("error_type,safe_retry", [(RedisConnectionError, True), (OutOfMemoryError, False)])
def test_preexecution_ping_and_retry_classification_are_unchanged(tmp_path, monkeypatch, error_type, safe_retry):
    @dataclass(frozen=True)
    class Settings:
        query: QuerySettings
    cfg = config(tmp_path)
    cfg.settings_path.write_text("controlled configuration", encoding="utf-8")
    monkeypatch.setattr("src.core.settings.load_settings", lambda _: Settings(QuerySettings()))
    client = RejectingRedis(error_type("controlled ping error"))
    def ping():
        raise client.error
    client.ping = ping
    with pytest.raises(BackendError, match="QUERY_CACHE_UNAVAILABLE_BEFORE_EXECUTION") as observed:
        run_query({"query": "fixture", "_settings_sha256": hashlib.sha256(cfg.settings_path.read_bytes()).hexdigest()},
                  cfg, job_id="fixture", timeout_seconds=5, redis_client=client,
                  service_factory=lambda *a, **k: pytest.fail("business factory must not run"))
    assert observed.value.safe_retry is safe_retry and not client.writes


def test_normal_execute_request_retains_completed_envelope_after_final_cache_rejection(tmp_path, monkeypatch):
    from src.service_layer import backends, execute, worker

    @dataclass(frozen=True)
    class Settings:
        query: QuerySettings
    cfg = config(tmp_path)
    cfg.settings_path.write_text("controlled configuration", encoding="utf-8")
    monkeypatch.setattr("src.core.settings.load_settings", lambda _: Settings(QuerySettings()))
    service, model, _, calls, _ = make_service(tmp_path)
    client = RejectingRedis(OutOfMemoryError("controlled Redis SET refusal"))
    client.ping = lambda: True
    def normal_query(payload, config, **kwargs):
        return run_query(payload, config, redis_client=client, service_factory=lambda *a, **k: service, **kwargs)
    monkeypatch.setattr(backends, "run_query", normal_query)
    payload = {"query": "fixture", "collection": "docs", "answer": False, "rewrite": False, "rerank": False,
               "_settings_sha256": hashlib.sha256(cfg.settings_path.read_bytes()).hexdigest()}
    execute.save(tmp_path / "request.json", {"schema": "service-attempt-request.v1",
        "job": {"id": "cache-oom", "attempt": 1, "kind": "query", "payload": payload, "timeout_seconds": 5},
        "server_config": worker.backend_configuration(cfg)})
    assert execute.execute_request(tmp_path / "request.json", tmp_path / "result.json") == 0
    result = json.loads((tmp_path / "result.json").read_bytes())
    assert result["job_id"] == "cache-oom" and result["attempt"] == 1
    assert result["ok"] is True and result["safe_retry"] is False and result["business_status"] == "completed"
    assert result["result"]["warnings"] == ["complete_result_cache_write_failed"]
    assert "result" not in result["result"]["cache"] and result["result"]["results"]
    assert len(client.writes) == 1 and len(calls) == 2 and not model.calls
