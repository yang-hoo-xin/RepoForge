"""Startup measurements with fake Redis/query boundaries, no models or service.

Scope: additive attempt/process observations only. 为什么要有它：拆分启动成本不能改变业务返回、
重试资格或泄露查询和凭据，也不能把嵌套跨度求和冒充耗时。
"""
from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

import pytest

from src.core.settings import QuerySettings
from src.core.trace.execution_trace import ExecutionTrace, current_trace, trace_operation
from src.service_layer import backends, execute

QUERY_CANARY = "PRIVATE_QUERY_CANARY"
SECRET_CANARY = "PRIVATE_ENV_CREDENTIAL"
DOMAIN_TRACE_ID = "b8f1b2da-57d1-456f-bf92-a6a0f626c211"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Startup trace tests do not call networks or models.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def _request(tmp_path, monkeypatch, *, kind="query"):
    settings = tmp_path / "settings.yaml"
    settings.write_text("fixture settings", encoding="utf-8")
    payload = {"query": QUERY_CANARY, "answer": False, "rewrite": False, "rerank": False,
               "_settings_sha256": hashlib.sha256(settings.read_bytes()).hexdigest()}
    request = {
        "schema": "service-attempt-request.v1",
        "job": {"id": "job-trace-fixture", "attempt": 2, "kind": kind,
                "timeout_seconds": 5, "payload": payload if kind == "query" else {"description": QUERY_CANARY}},
        "server_config": {"namespace": "trace-test", "work_root": str(tmp_path),
                          "settings_path": str(settings), "agent_profiles": {}, "query_cache_ttl": 30,
                          "retention_seconds": 86400, "max_pending": 1000,
                          "capacity_bytes": 64*1024*1024, "cache_capacity_bytes": 16*1024*1024},
    }
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setenv("SERVICE_CHILD_REDIS_URL", "redis://:" + SECRET_CANARY + "@localhost/0")
    monkeypatch.setenv("FAKE_API_KEY", SECRET_CANARY)
    return source, tmp_path / "result.json"


def _query_backend(monkeypatch, *, fail_redis=False):
    @dataclass(frozen=True)
    class Settings:
        query: QuerySettings
        retrieval: object

    monkeypatch.setattr("src.core.settings.load_settings", lambda _path: Settings(
        QuerySettings(), SimpleNamespace(dense_enabled=False, dense_provider="disabled")))
    calls = []
    business = {"status": "completed", "trace_id": DOMAIN_TRACE_ID, "answer": "unchanged",
                "cache": {"result": "miss"}, "query": QUERY_CANARY}

    class Client:
        def ping(self):
            calls.append("redis_ping")
            if fail_redis:
                raise ConnectionError(SECRET_CANARY)
            return True

    def query(request):
        assert request.query == QUERY_CANARY
        calls.append("query")
        with trace_operation("unrelated.business", attributes={"query": QUERY_CANARY, "secret": SECRET_CANARY}):
            pass
        # Scope: observer filtering. 为什么要有它：即使相同阶段误传额外属性，启动Trace也只保留固定安全字段。
        with trace_operation("service.query.execute", attributes={"query": QUERY_CANARY, "secret": SECRET_CANARY}):
            pass
        return SimpleNamespace(to_dict=lambda: dict(business))

    def factory(settings, **_kwargs):
        assert settings.query.timeout_seconds == 5
        calls.append("factory")
        return SimpleNamespace(query=query)

    original = backends.run_query
    monkeypatch.setattr(backends, "run_query", lambda payload, config, **kwargs: original(
        payload, config, **kwargs, redis_client=Client(), service_factory=factory))
    return business, calls


def _trace(result_path):
    envelope = json.loads(result_path.read_bytes())
    reference = envelope["startup_trace_ref"]
    assert execute.reference(reference["path"]) == reference
    return envelope, json.loads(Path(reference["path"]).read_bytes())


def test_query_startup_phases_link_job_attempt_and_domain_without_business_changes(tmp_path, monkeypatch):
    request, result = _request(tmp_path, monkeypatch)
    business, calls = _query_backend(monkeypatch)
    assert execute.execute_request(request, result) == 0
    envelope, trace = _trace(result)
    assert envelope["result"] == business
    assert envelope["ok"] is True and envelope["safe_retry"] is False
    assert calls == ["redis_ping", "factory", "query"]
    assert trace["service_attempt"] == {
        "job_id": "job-trace-fixture", "attempt": 2, "kind": "query",
        "child_pid": execute.os.getpid(), "domain_query_trace_id": DOMAIN_TRACE_ID,
    }
    rows = list(trace["execution_observations"].values())
    names = {row["name"] for row in rows}
    assert names == {
        "service.child.execution", "service.child.backend_import",
        "service.query.imports", "service.query.settings", "service.query.redis_connect",
        "service.query.factory", "service.query.execute",
        "service.child.business_result_persist", "service.child.public_result",
    }
    assert all(row["status"] == "completed" and row["elapsed_ms"] >= 0 for row in rows)
    assert all(row["trace_id"] == trace["execution_trace"]["trace_id"] for row in rows)
    assert all(row["finished_at"] for row in rows)
    assert "not additive" in trace["timing_scope"]
    assert QUERY_CANARY not in json.dumps(trace) and SECRET_CANARY not in json.dumps(trace)
    assert "query" not in envelope.keys()
    boundaries = trace["timing_boundaries"]
    assert boundaries["module_entry"]["monotonic_ns"] <= boundaries["execute_request_entered"]["monotonic_ns"]
    assert boundaries["trace_bootstrap_started"]["perf_counter_ns"] <= boundaries["trace_bootstrap_finished"]["perf_counter_ns"]
    assert all(value["utc"].endswith("+00:00") for value in boundaries.values())


def test_redis_failure_trace_keeps_original_safe_retry_and_hides_exception_text(tmp_path, monkeypatch):
    request, result = _request(tmp_path, monkeypatch)
    _, calls = _query_backend(monkeypatch, fail_redis=True)
    assert execute.execute_request(request, result) == 0
    envelope, trace = _trace(result)
    assert envelope["ok"] is False and envelope["safe_retry"] is True
    assert envelope["execution_state"] == "not_started"
    assert envelope["error"]["code"] == "QUERY_CACHE_UNAVAILABLE_BEFORE_EXECUTION"
    rows = list(trace["execution_observations"].values())
    redis = next(row for row in rows if row["name"] == "service.query.redis_connect")
    assert redis["status"] == "failed" and redis["attributes"]["exception_type"] == "BackendError"
    assert calls == ["redis_ping"]
    assert not any(row["name"] == "service.query.factory" for row in rows)
    assert QUERY_CANARY not in json.dumps(trace) and SECRET_CANARY not in json.dumps(trace)
    assert SECRET_CANARY not in json.dumps(envelope)


def test_trace_persistence_failure_does_not_change_success_or_retry(tmp_path, monkeypatch):
    request, result = _request(tmp_path, monkeypatch)
    business, _ = _query_backend(monkeypatch)

    def fail(*_args):
        raise OSError(SECRET_CANARY)

    monkeypatch.setattr(execute, "_retain_startup_trace", fail)
    assert execute.execute_request(request, result) == 0
    envelope = json.loads(result.read_bytes())
    assert envelope["ok"] is True and envelope["safe_retry"] is False
    assert envelope["result"] == business
    assert envelope["startup_trace_error"] == {"type": "OSError"}
    assert "startup_trace_ref" not in envelope
    assert SECRET_CANARY not in json.dumps(envelope)


@pytest.mark.parametrize("has_active_trace", [False, True])
def test_agent_dispatch_and_active_trace_semantics_are_unchanged(tmp_path, monkeypatch, has_active_trace):
    request, result = _request(tmp_path, monkeypatch, kind="agent")
    calls = []
    observed = {}
    original_trace = ExecutionTrace(
        "a" * 32, "original-agent-session", "fixture", recorder=lambda row: observed.update({row["id"]: row}),
        root_name="product.execution",
    ) if has_active_trace else None

    def agent(payload, config, **kwargs):
        assert current_trace() is original_trace
        calls.append((payload, kwargs))
        with trace_operation("product.agent.operation"):
            pass
        return {"status": "awaiting_approval", "actual_write_performed": False}

    monkeypatch.setattr(backends, "run_agent", agent)
    with original_trace.activate() if original_trace is not None else nullcontext():
        assert execute.execute_request(request, result) == 0
        assert current_trace() is original_trace
    envelope = json.loads(result.read_bytes())
    assert calls == [({"description": QUERY_CANARY}, {"job_id": "job-trace-fixture"})]
    assert envelope["result"] == {"status": "awaiting_approval", "actual_write_performed": False}
    assert envelope["business_status"] == "awaiting_approval" and envelope["ok"] is True
    assert "startup_trace_ref" not in envelope and "startup_trace_error" not in envelope
    assert not (tmp_path / "execution").exists()
    assert not any(row["name"].startswith("service.") for row in observed.values())


def test_process_boundaries_correlate_spawn_and_exit_without_claiming_interpreter_time(tmp_path):
    result = execute.execute_process(
        [sys.executable, "-I", "-B", "-c", "pass"], cwd=tmp_path,
        output=tmp_path / "process", deadline=time.monotonic() + 15,
    )
    assert result["exit_code"] == 0 and result["timeout"] is False
    names = ["execute_entered", "popen_requested", "popen_returned",
             "child_released", "exit_observed", "cleanup_finished"]
    boundaries = result["timing_boundaries"]
    assert set(boundaries) == set(names)
    for clock in ("monotonic_ns", "perf_counter_ns"):
        values = [boundaries[name][clock] for name in names]
        assert values == sorted(values) and all(type(value) is int for value in values)
    assert all(boundaries[name]["utc"].endswith("+00:00") for name in names)
    assert "interpreter_startup_ms" not in result
    assert json.loads((tmp_path / "process/process.json").read_bytes())["timing_boundaries"] == boundaries
