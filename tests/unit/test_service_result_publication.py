"""Actual Store/worker local result-limit controls; no Redis server or provider.

Scope: only fake transport beneath the real Store and an already-completed child.
为什么要有它：必须区分调用Python方法与实际发送Redis写入，不能把本地拒绝误记为网络结果未知。
"""
import json
from pathlib import Path

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from src.service_layer import execute, worker
from src.service_layer.store import (
    _MAX_ERROR_BYTES,
    _MAX_PAYLOAD_BYTES,
    _MAX_RESULT_BYTES,
    RedisJobStore,
    ResultTooLargeError,
)
from tests.unit.test_service_worker import config


class RecordingTransport:
    def __init__(self, kind="query", outcome="accepted"):
        self.operations, self.terminal_writes = [], []
        self.outcome = outcome
        self.job = {"id": "a" * 32, "kind": kind, "state": "running", "attempt": 1,
            "max_attempts": 1 if kind == "agent" else 2, "timeout_seconds": 10, "lease_token": "original-fence",
            "payload_json": '{"query":"controlled"}', "result_json": "null", "error_json": "null",
            "attempts_history": [{"attempt": 1, "state": "running", "result_json": "null", "error_json": "null"}]}

    def register_script(self, script):
        def call(*, keys, args):
            self.operations.append(args[0])
            if args[0] == "claim":
                return json.dumps({"code": "claimed", "job": self.job})
            assert args[0] in {"heartbeat", "finish", "retry"}
            if args[0] == "finish":
                self.terminal_writes.append(json.loads(args[2]))
                if self.outcome == "reply_unknown":
                    raise RedisConnectionError("controlled reply lost after possible commit")
                if self.outcome == "rejected":
                    return '{"code":"rejected"}'
            return '{"code":"accepted"}'
        return call


class CountingStore(RedisJobStore):
    finish_calls = 0

    def finish(self, *args, **kwargs):
        self.finish_calls += 1
        return super().finish(*args, **kwargs)


def oversized_result(status="completed"):
    return {"status": status, "answer": "x" * (_MAX_RESULT_BYTES + 1)}


def completed_runner(original, saved_refs):
    def runner(argv, **kwargs):
        request = json.loads(Path(argv[argv.index("--request") + 1]).read_bytes())
        saved_refs.append(execute.save(argv[argv.index("--result") + 1], {"schema": "service-attempt-result.v1",
            "job_id": request["job"]["id"], "attempt": request["job"]["attempt"],
            "ok": True, "safe_retry": False, "business_status": original["status"], "result": original}))
        return {"status": "exited", "started": True, "exit_code": 0, "timeout": False}
    return runner


def receipt_at(tmp_path):
    return json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())


def test_real_store_oversized_finish_rejects_before_any_redis_operation():
    transport = RecordingTransport()
    store = RedisJobStore(transport, namespace="result-publication-control")
    with pytest.raises(ResultTooLargeError, match="JSON value exceeds UTF-8 size limit") as observed:
        store.finish(transport.job["id"], transport.job["lease_token"], "completed", result=oversized_result())
    assert transport.operations == []
    assert observed.value.limit_bytes == _MAX_RESULT_BYTES
    assert observed.value.size_bytes == len(json.dumps(oversized_result(), separators=(",", ":")).encode())


@pytest.mark.parametrize("kind", ["query", "agent"])
@pytest.mark.parametrize("domain_status", ["completed", "awaiting_approval", "failed"])
def test_worker_must_not_label_proven_local_result_refusal_as_unknown_redis_write(tmp_path, kind, domain_status):
    transport = RecordingTransport(kind)
    store = CountingStore(transport, namespace="result-publication-control")
    original, saved_refs = oversized_result(domain_status), []
    assert worker.run_once(store, config(tmp_path), worker_id="size-control", runner=completed_runner(original, saved_refs))
    receipt = receipt_at(tmp_path)
    result_path = Path(receipt["result_ref"]["path"])
    assert execute.reference(result_path) == receipt["result_ref"] == saved_refs[0]
    retained = json.loads(result_path.read_bytes())
    assert retained["ok"] is True and retained["business_status"] == domain_status and retained["result"] == original
    assert receipt["process"]["exit_code"] == 0 and receipt["lease_lost"] is False
    assert store.finish_calls == 2 and transport.operations.count("finish") == 1 and "retry" not in transport.operations
    assert receipt["publication"] == "accepted" and receipt["requested_state"] == "failed"
    rejected = receipt["result_publication_rejection"]
    assert rejected["redis_write_attempted"] is False and rejected["original_result_ref"] == saved_refs[0]
    sent = transport.terminal_writes[0]
    assert sent["job_id"] == transport.job["id"] and sent["lease_token"] == "original-fence"
    assert sent["state"] == "failed" and sent["result_json"] == "null"
    error = json.loads(sent["error_json"])
    assert error == {"code": "SERVICE_RESULT_TOO_LARGE", "failure_stage": "result_publication",
        "backend_returned": True, "result_retained": True, "size_bytes": rejected["size_bytes"], "limit_bytes": _MAX_RESULT_BYTES}
    assert "path" not in error and "original_result_ref" not in error and len(sent["error_json"].encode()) < 1024


@pytest.mark.parametrize("outcome,publication", [("reply_unknown", "store_response_unknown"), ("rejected", "suppressed_fence_rejected")])
def test_small_terminal_reply_unknown_or_rejected_never_causes_third_finish(tmp_path, outcome, publication):
    transport, refs = RecordingTransport(outcome=outcome), []
    store = CountingStore(transport, namespace="result-publication-control")
    worker.run_once(store, config(tmp_path), worker_id="size-control", runner=completed_runner(oversized_result(), refs))
    receipt = receipt_at(tmp_path)
    assert store.finish_calls == 2 and transport.operations.count("finish") == 1 and "retry" not in transport.operations
    assert receipt["publication"] == publication and receipt["result_publication_rejection"]["redis_write_attempted"] is False
    assert receipt["result_ref"] == refs[0]


def test_ordinary_network_reply_unknown_has_no_local_refusal_fallback(tmp_path):
    transport, refs = RecordingTransport(outcome="reply_unknown"), []
    store = CountingStore(transport, namespace="result-publication-control")
    worker.run_once(store, config(tmp_path), worker_id="size-control", runner=completed_runner({"status": "completed"}, refs))
    receipt = receipt_at(tmp_path)
    assert store.finish_calls == transport.operations.count("finish") == 1
    assert receipt["publication"] == "store_response_unknown" and "result_publication_rejection" not in receipt


def test_lost_lease_after_local_refusal_suppresses_even_small_terminal(tmp_path, monkeypatch):
    captured = []
    class Heartbeat(worker.LeaseHeartbeat):
        def start(self):
            captured.append(self)
        def stop(self, shutdown=None):
            pass
    class LostAfterRefusal(CountingStore):
        def finish(self, *args, **kwargs):
            try:
                return super().finish(*args, **kwargs)
            except ResultTooLargeError:
                captured[0].lost.set()
                raise
    monkeypatch.setattr(worker, "LeaseHeartbeat", Heartbeat)
    transport, refs = RecordingTransport(), []
    store = LostAfterRefusal(transport, namespace="result-publication-control")
    worker.run_once(store, config(tmp_path), worker_id="size-control", runner=completed_runner(oversized_result(), refs))
    receipt = receipt_at(tmp_path)
    assert store.finish_calls == 1 and "finish" not in transport.operations and "retry" not in transport.operations
    assert receipt["publication"] == "suppressed_lease_lost" and receipt["lease_lost"] is True
    assert receipt["result_publication_rejection"]["original_result_ref"] == refs[0]


@pytest.mark.parametrize("fault", ["invalid_result", "nonfinite_result", "oversized_error", "oversized_payload"])
def test_other_local_validation_errors_are_not_result_size_proofs(fault):
    transport = RecordingTransport()
    store = RedisJobStore(transport, namespace="result-publication-control")
    with pytest.raises(ValueError) as observed:
        if fault == "oversized_payload":
            store.submit("query", {"query": "x" * (_MAX_PAYLOAD_BYTES + 1)}, "key", 10, 1)
        else:
            result = object() if fault == "invalid_result" else float("nan") if fault == "nonfinite_result" else None
            error = "x" * (_MAX_ERROR_BYTES + 1) if fault == "oversized_error" else None
            store.finish(transport.job["id"], "original-fence", "failed", result=result, error=error)
    assert not isinstance(observed.value, ResultTooLargeError) and transport.operations == []


@pytest.mark.parametrize("unicode", [False, True])
def test_exact_utf8_result_limit_is_unchanged(unicode):
    transport = RecordingTransport()
    store = RedisJobStore(transport, namespace="result-publication-control")
    body_bytes = _MAX_RESULT_BYTES - 2
    value = "汉" * (body_bytes // 3) + "x" * (body_bytes % 3) if unicode else "x" * body_bytes
    assert store.finish(transport.job["id"], "original-fence", "completed", result=value)
    assert len(transport.terminal_writes[0]["result_json"].encode()) == _MAX_RESULT_BYTES
    with pytest.raises(ResultTooLargeError):
        store.finish(transport.job["id"], "original-fence", "completed", result=value + "汉")
    assert transport.operations == ["finish"]


def test_worker_generic_error_limit_refusal_keeps_existing_no_second_write_semantics(tmp_path):
    transport = RecordingTransport()
    store = CountingStore(transport, namespace="result-publication-control")
    def runner(argv, **kwargs):
        execute.save(argv[argv.index("--result") + 1], {"schema": "service-attempt-result.v1",
            "job_id": transport.job["id"], "attempt": 1, "ok": False, "safe_retry": False,
            "error": {"details": "x" * (_MAX_ERROR_BYTES + 1)}})
        return {"status": "exited", "started": True, "exit_code": 0}
    worker.run_once(store, config(tmp_path), worker_id="size-control", runner=runner)
    receipt = receipt_at(tmp_path)
    assert store.finish_calls == 1 and "finish" not in transport.operations and "retry" not in transport.operations
    assert receipt["publication"] == "store_response_unknown" and "result_publication_rejection" not in receipt
