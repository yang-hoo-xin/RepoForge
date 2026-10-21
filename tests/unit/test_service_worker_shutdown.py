"""Controlled worker shutdown; no Redis server, Docker or provider calls."""
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.service_layer import execute, worker


def config(tmp_path):
    return SimpleNamespace(work_root=tmp_path, settings_path=tmp_path / "settings.yaml",
        agent_profiles={}, namespace="shutdown-test", redis_url="redis://127.0.0.1:6379/0",
        query_cache_ttl=30, max_job_seconds=10, lease_seconds=3, retry_delay_seconds=1,
        retention_seconds=60, max_pending=10, capacity_bytes=64*1024*1024,
        cache_capacity_bytes=16*1024*1024)


class Store:
    def __init__(self, kind="query"):
        self.job = {"id": "shutdown-job", "kind": kind, "payload": {}, "attempt": 1,
                    "timeout_seconds": 10, "lease_token": "original-token"}
        self.claims, self.finished, self.retried = 0, [], []

    def claim(self, *args):
        self.claims += 1
        return self.job

    def heartbeat(self, *args):
        return True

    def finish(self, *args, **kwargs):
        self.finished.append((args, kwargs))
        return True

    def retry(self, *args):
        self.retried.append(args)
        return True


def test_stop_before_claim_does_not_claim_or_execute(tmp_path):
    stop = worker.WorkerShutdown()
    stop.request(signal.SIGTERM, None)
    store = Store()
    assert worker.run_once(store, config(tmp_path), worker_id="test", shutdown=stop,
        runner=lambda *a, **k: pytest.fail("started after shutdown")) is False
    assert store.claims == 0 and not store.finished


@pytest.mark.parametrize("kind,state", [("query", "cancelled"), ("agent", "indeterminate")])
def test_active_shutdown_cancel_keeps_original_fence_and_never_retries(tmp_path, kind, state):
    stop, store = worker.WorkerShutdown(), Store(kind)
    def runner(*args, **kwargs):
        assert not kwargs["cancel_event"].is_set()
        stop.request(signal.SIGTERM, None)
        assert kwargs["cancel_event"].is_set()
        return {"status": "cancelled", "cancelled": True, "started": True, "exit_code": -9}
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner, shutdown=stop)
    assert len(store.finished) == 1 and not store.retried
    assert store.finished[0][0] == ("shutdown-job", "original-token", state)
    evidence = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert evidence["shutdown_requested"] is True and evidence["lease_lost"] is False


def test_stop_during_claim_never_starts_claimed_agent(tmp_path):
    stop, store = worker.WorkerShutdown(), Store("agent")
    def claim(*args):
        stop.request(signal.SIGINT, None)
        return store.job
    store.claim = claim
    def runner(*args, **kwargs):
        assert kwargs["cancel_event"].is_set()
        return execute.execute_process(*args, **kwargs)
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner, shutdown=stop)
    assert store.finished[0][0][2] == "cancelled" and not store.retried
    receipt = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert receipt["process"]["started"] is False
    assert receipt["process"]["status"] == "cancelled_before_start"


def completed_runner(stop):
    def runner(argv, **kwargs):
        request = json.loads(Path(argv[argv.index("--request") + 1]).read_bytes())
        execute.save(argv[argv.index("--result") + 1], {"schema": "service-attempt-result.v1",
            "job_id": request["job"]["id"], "attempt": request["job"]["attempt"],
            "ok": True, "safe_retry": False, "result": {"status": "completed"}})
        stop.request(signal.SIGTERM, None)
        return {"status": "exited", "started": True, "exit_code": 0}
    return runner


def test_late_signal_preserves_completed_result_and_original_deadline(tmp_path):
    stop, store = worker.WorkerShutdown(), Store()
    worker.run_once(store, config(tmp_path), worker_id="test", runner=completed_runner(stop), shutdown=stop)
    assert store.finished[0][0][2] == "completed" and not store.retried
    receipt = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert receipt["shutdown_requested"] is True and receipt["publication"] == "accepted"


def test_shutdown_unknown_finish_reply_never_republishes(tmp_path):
    stop, store = worker.WorkerShutdown(), Store()
    def finish(*args, **kwargs):
        store.finished.append((args, kwargs))
        raise ConnectionError("possible commit; reply lost")
    store.finish = finish
    worker.run_once(store, config(tmp_path), worker_id="test", runner=completed_runner(stop), shutdown=stop)
    assert len(store.finished) == 1 and not store.retried
    receipt = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert receipt["publication"] == "store_response_unknown"


def test_shutdown_does_not_override_lease_loss(tmp_path, monkeypatch):
    class Lost(worker.LeaseHeartbeat):
        def start(self):
            self.lost.set()
        def stop(self, shutdown=None):
            pass
    monkeypatch.setattr(worker, "LeaseHeartbeat", Lost)
    stop, store = worker.WorkerShutdown(), Store("agent")
    worker.run_once(store, config(tmp_path), worker_id="test", runner=completed_runner(stop), shutdown=stop)
    assert not store.finished and not store.retried
    receipt = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert receipt["publication"] == "suppressed_lease_lost"


def test_heartbeat_stop_observes_signal_and_repeated_stop_has_one_join_budget(monkeypatch):
    stop, now, waits = worker.WorkerShutdown(), [0.0], []
    def join(timeout):
        now[0] += timeout
        waits.append(timeout)
        stop.request(signal.SIGTERM, None)
    monkeypatch.setattr(worker.time, "monotonic", lambda: now[0])
    heartbeat = worker.LeaseHeartbeat(Store(), {}, 30)
    heartbeat.thread = SimpleNamespace(is_alive=lambda: True, join=join)
    heartbeat.stop(shutdown=stop)
    first_wait_count = len(waits)
    heartbeat.stop(shutdown=stop)
    assert heartbeat.lost.is_set() and now[0] <= 1.051
    assert len(waits) == first_wait_count and max(waits) <= 0.05


def setup_main(monkeypatch, tmp_path, store, *, close_error=None, client_error=None):
    cfg, events = config(tmp_path), []
    cfg.retention_seconds, cfg.max_pending, cfg.poll_seconds = 60, 10, 30
    def close_runner():
        events.append("runner_closed")
        if close_error:
            raise close_error
    def close_client():
        events.append("client_closed")
        if client_error:
            raise client_error
    monkeypatch.setattr("src.service_layer.config.ServiceConfig.from_env", lambda: cfg)
    monkeypatch.setattr(worker, "worker_redis_client", lambda _: SimpleNamespace(close=close_client))
    monkeypatch.setattr("src.service_layer.store.RedisJobStore", lambda *a, **k: store)
    monkeypatch.setattr("src.service_layer.sparse_process.SparseProcess", lambda _: SimpleNamespace(
        reap_idle=lambda: events.append("reaped"), close=close_runner))
    return events


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("during", ["idle", "backoff", "recovery"])
def test_normal_main_signal_interrupts_wait_and_stops_claims(tmp_path, monkeypatch, signum, during):
    import redis
    class Idle(Store):
        def recover_expired(self):
            if during == "recovery":
                signal.raise_signal(signum)
            elif during == "backoff":
                raise redis.ConnectionError("controlled")
        def claim(self, *args):
            self.claims += 1
            return None
    store = Idle()
    events = setup_main(monkeypatch, tmp_path, store)
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    def sleep(seconds):
        assert seconds <= 0.05
        events.append("wait_interrupted")
        signal.raise_signal(signum)
    monkeypatch.setattr(worker.time, "sleep", sleep)
    assert worker.main([]) == 0
    assert store.claims == (1 if during == "idle" else 0)
    assert events[-2:] == ["runner_closed", "client_closed"]
    assert {s: signal.getsignal(s) for s in before} == before


@pytest.mark.parametrize("client_error", [None, RuntimeError("client close failed")])
def test_cleanup_failure_still_releases_client_and_restores_handlers(tmp_path, monkeypatch, client_error):
    class Idle(Store):
        def recover_expired(self):
            pass
        def claim(self, *args):
            return None
    events = setup_main(monkeypatch, tmp_path, Idle(), close_error=ValueError("runner close failed"),
                        client_error=client_error)
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    with pytest.raises(RuntimeError if client_error else ValueError):
        worker.main(["--once"])
    assert events[-2:] == ["runner_closed", "client_closed"]
    assert {s: signal.getsignal(s) for s in before} == before


def test_worker_redis_url_cannot_enable_command_retries_or_expand_socket_timeouts():
    import redis
    client = worker.worker_redis_client("redis://127.0.0.1:6379/0?socket_timeout=99&"
        "socket_connect_timeout=99&retry_on_timeout=True&retry=invalid&retry_on_error=TimeoutError")
    try:
        options = client.connection_pool.connection_kwargs
        assert options["socket_timeout"] == options["socket_connect_timeout"] == 3
        assert options["retry_on_timeout"] is False and options["retry_on_error"] == []
        assert client.auto_close_connection_pool is True
        conn = client.connection_pool.connection_class(**options)  # No socket opened.
        calls = []
        def operation():
            calls.append(1)
            raise redis.TimeoutError("reply unknown")
        with pytest.raises(redis.TimeoutError):
            conn.retry.call_with_retry(operation, lambda _: None)
        assert calls == [1]
    finally:
        client.close()


def test_shutdown_cleanup_failure_remains_uncertain_and_is_not_retried(tmp_path):
    stop, store = worker.WorkerShutdown(), Store("agent")
    def runner(*args, **kwargs):
        stop.request(signal.SIGTERM, None)
        return {"status": "infrastructure_failed", "cancelled": True, "started": True,
                "cleanup_error_type": "TimeoutError", "exit_code": None}
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner, shutdown=stop)
    assert store.finished[0][0][2] == "indeterminate" and not store.retried
    assert store.finished[0][1]["error"]["code"] == "CHILD_PROCESS_RESULT_UNAVAILABLE"
