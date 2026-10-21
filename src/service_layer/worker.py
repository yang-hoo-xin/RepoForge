"""Leased Redis jobs with bounded isolated or pure-sparse process execution."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from src.service_layer.execute import REPO, execute_process, reference, save
from src.service_layer.store import ResultTooLargeError


class WorkerShutdown:
    """Signal state only; cleanup remains in the ordinary worker control flow."""

    def __init__(self):
        self.requested = False
        self.signal_number = None

    def request(self, signum, frame):
        # Scope: Python signal handlers. 为什么要有它：handler可能中断锁或I/O，不能在其中续租、发布、关闭进程或获取Event锁。
        self.requested = True
        self.signal_number = signum

    def wait(self, seconds):
        until = time.monotonic() + seconds
        while not self.requested:
            remaining = until - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
        return self.requested


class AttemptCancellation(threading.Event):
    def __init__(self, lease_lost, shutdown):
        super().__init__()
        self.lease_lost, self.shutdown = lease_lost, shutdown

    def is_set(self):
        return (super().is_set() or self.lease_lost.is_set()
                or self.shutdown is not None and self.shutdown.requested)


def worker_redis_client(url):
    import redis
    from redis.backoff import NoBackoff
    from redis.connection import parse_url
    from redis.retry import Retry

    # Scope: worker-owned queue connections, after URL parsing. 为什么要有它：URL不得覆盖停止预算或自动重发回复丢失的claim/finish，轮询恢复仍由worker外层处理。
    options = parse_url(url)
    options.update(socket_connect_timeout=3, socket_timeout=3, retry=Retry(NoBackoff(), 0),
                   retry_on_timeout=False, retry_on_error=[])
    return redis.Redis.from_pool(redis.ConnectionPool(**options))


def backend_configuration(config):
    """Only server-owned routing/storage settings; credentials travel in child env."""
    return {"namespace": config.namespace, "work_root": str(config.work_root),
            "settings_path": str(config.settings_path),
            "agent_profiles": {key: str(value) for key, value in config.agent_profiles.items()},
            "query_cache_ttl": config.query_cache_ttl,
            "retention_seconds": config.retention_seconds, "max_pending": config.max_pending,
            "capacity_bytes": config.capacity_bytes, "cache_capacity_bytes": config.cache_capacity_bytes}


class LeaseHeartbeat:
    def __init__(self, store, job, lease_seconds):
        self.store, self.job, self.lease_seconds = store, job, lease_seconds
        self.stopped, self.lost = threading.Event(), threading.Event()
        self.error_type = None
        self.shutdown_join_deadline = None
        self.thread = threading.Thread(target=self._run, name="service-lease-heartbeat", daemon=True)

    def _run(self):
        interval = max(0.05, min(5.0, self.lease_seconds / 3))
        while not self.stopped.wait(interval):
            try:
                owned = self.store.heartbeat(self.job["id"], self.job["lease_token"], self.lease_seconds)
                if owned is not True:
                    self.lost.set()
                    return
            except Exception as error:
                # 为什么要有它：Redis故障不能当作仍持有租约；原进程仍受绝对deadline约束，但不能发布晚到结果。
                self.error_type = type(error).__name__
                self.lost.set()
                return

    def start(self):
        self.thread.start()

    def stop(self, shutdown=None):
        self.stopped.set()
        until = time.monotonic() + max(1.0, self.lease_seconds)
        while self.thread.is_alive():
            if shutdown is not None and shutdown.requested:
                # Scope: stopping heartbeat only. 为什么要有它：阻塞的Redis续租不能两次各等待整个租约期，无法确认退出时保守禁止发布。
                if self.shutdown_join_deadline is None:
                    self.shutdown_join_deadline = time.monotonic() + 1.0
                until = min(until, self.shutdown_join_deadline)
            remaining = until - time.monotonic()
            if remaining <= 0:
                break
            self.thread.join(timeout=min(0.05, remaining))
        if self.thread.is_alive():
            self.lost.set()


def _attempt_directory(config, job):
    if not isinstance(job.get("id"), str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job["id"]) is None:
        raise ValueError("SERVICE_JOB_ID_INVALID")
    if type(job.get("attempt")) is not int or job["attempt"] < 1:
        raise ValueError("SERVICE_ATTEMPT_INVALID")
    root = Path(config.work_root).resolve()
    path = root / "jobs" / job["id"] / (str(job["attempt"]) + "-" + uuid4().hex[:12])
    if not path.resolve().is_relative_to(root):
        raise ValueError("SERVICE_ATTEMPT_PATH_ESCAPE")
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def run_once(store, config, *, worker_id, runner=execute_process, shutdown=None):
    if shutdown is not None and shutdown.requested:
        return False
    claimed_at = time.monotonic()
    job = store.claim(worker_id, config.lease_seconds)
    if job is None:
        if hasattr(runner, "reap_idle"):
            runner.reap_idle()
        return False
    path = _attempt_directory(config, job)
    timeout = job.get("timeout_seconds")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("SERVICE_JOB_TIMEOUT_INVALID")
    timeout = min(timeout, config.max_job_seconds)
    deadline = claimed_at + timeout
    request = {"schema": "service-attempt-request.v1", "job": {
        "id": job["id"], "kind": job["kind"], "payload": job["payload"],
        "attempt": job["attempt"], "timeout_seconds": timeout}, "server_config": backend_configuration(config)}
    request_ref = save(path / "request.json", request)
    result_path = path / "result.json"
    environment = os.environ.copy()
    environment.pop("SERVICE_API_TOKEN", None)
    environment["SERVICE_CHILD_REDIS_URL"] = config.redis_url
    command = [sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__).with_name("execute.py").resolve()),
               "--request", str(path / "request.json"), "--result", str(result_path)]
    heartbeat = LeaseHeartbeat(store, job, config.lease_seconds)
    heartbeat.start()
    evidence = {"schema": "service-attempt-evidence.v1", "job_id": job["id"], "attempt": job["attempt"],
                "request_ref": request_ref, "command": command, "deadline_monotonic": deadline,
                "publication": "not_attempted", "local_process_cancel_on_lease_loss": True}
    envelope = None
    publication_attempted = False
    cancellation = AttemptCancellation(heartbeat.lost, shutdown)

    def stop_heartbeat():
        heartbeat.stop() if shutdown is None else heartbeat.stop(shutdown=shutdown)

    def stopping():
        return shutdown is not None and shutdown.requested

    try:
        # Leave a small bounded margin for durable terminal publication; never refresh the original deadline.
        process = runner(command, cwd=REPO, output=path / "process", deadline=deadline-min(1.0, timeout*0.05),
                         environment=environment, cancel_event=cancellation)
        evidence["process"] = process
        stop_heartbeat()
        # Scope: resident request completion is not an OS exit. 为什么要有它：存活子进程不能伪报退出0来绕过原封套与lease fence。
        request_succeeded = (
            process.get("request_completed") is True and process.get("request_exit_code") == 0
            and process.get("status") == "request_completed"
            if process.get("execution_mode") == "resident_sparse"
            else process.get("exit_code") == 0
        )
        if result_path.is_file():
            envelope = json.loads(result_path.read_bytes())
            if (not isinstance(envelope, dict) or envelope.get("schema") != "service-attempt-result.v1"
                    or envelope.get("job_id") != job["id"] or envelope.get("attempt") != job["attempt"]
                    or type(envelope.get("ok")) is not bool or type(envelope.get("safe_retry")) is not bool
                    or envelope.get("ok") is True and not isinstance(envelope.get("result"), dict)):
                raise ValueError("SERVICE_CHILD_RESULT_BINDING_INVALID")
            evidence["result_ref"] = reference(result_path)
        if process.get("timeout") or process.get("status") == "deadline_exhausted_before_start":
            state = "indeterminate" if job["kind"] == "agent" else "timed_out"
            error = {"code": "JOB_DEADLINE_EXCEEDED", "provider_server_state": "unknown"}
        elif (stopping() and process.get("cancelled") is True
              and process.get("status") in {"cancelled", "cancelled_before_start"}
              and not process.get("cleanup_error_type")):
            # Scope: actual runner cancellation, not just a late signal. 为什么要有它：已启动Agent的副作用未知，不能当成功或自动重试；已完成有效结果仍按原fence发布。
            not_started = process.get("started") is False and process.get("status") == "cancelled_before_start"
            state = "indeterminate" if job["kind"] == "agent" and not not_started else "cancelled"
            error = {"code": "WORKER_SHUTDOWN_CANCELLED", "execution_state": "not_started" if not_started else "unknown",
                     "provider_server_state": "unknown", "cost_unknown": not not_started}
        elif (process.get("status") == "infrastructure_failed" or process.get("cancelled")
              or not request_succeeded or envelope is None):
            state = "indeterminate" if job["kind"] == "agent" else "failed"
            error = {"code": "CHILD_PROCESS_RESULT_UNAVAILABLE", "cost_unknown": True}
        elif envelope["ok"]:
            state, error = "completed", None
        else:
            state = "indeterminate" if job["kind"] == "agent" and envelope.get("execution_state") != "not_started" else "failed"
            error = envelope.get("error") or {"code": "BACKEND_EXECUTION_FAILED"}
        if heartbeat.lost.is_set():
            if hasattr(runner, "close") and not stopping():
                evidence["post_attempt_cleanup"] = runner.close()
            evidence.update(publication="suppressed_lease_lost", heartbeat_error_type=heartbeat.error_type)
        else:
            # 为什么要有它：租约失效后结果只留在原attempt证据，Redis的token fence决定是否允许发布，不能复写新owner结果。
            safe_retry = (not stopping() and job["kind"] == "query" and envelope is not None and envelope.get("safe_retry") is True
                          and request_succeeded and not process.get("timeout") and envelope.get("ok") is False
                          and envelope.get("execution_state") == "not_started" and isinstance(error, dict)
                          and error.get("code") == "QUERY_CACHE_UNAVAILABLE_BEFORE_EXECUTION")
            if safe_retry:
                publication_attempted = True
                published = store.retry(job["id"], job["lease_token"], error, config.retry_delay_seconds)
            else:
                publication_attempted = True
                try:
                    published = store.finish(job["id"], job["lease_token"], state,
                                             result=envelope.get("result") if state == "completed" else None, error=error)
                except ResultTooLargeError as rejected:
                    if state != "completed" or envelope is None or envelope.get("ok") is not True:
                        raise
                    # Scope: proven local result-size refusal only. 为什么要有它：backend返回不等于领域成功；完整JSON已留存，服务发布失败只能写一次小终态，不能截断正文或重跑业务。
                    publication_attempted = False
                    error = {"code": "SERVICE_RESULT_TOO_LARGE", "failure_stage": "result_publication",
                             "backend_returned": True, "result_retained": True,
                             "size_bytes": rejected.size_bytes, "limit_bytes": rejected.limit_bytes}
                    evidence["result_publication_rejection"] = {**error, "redis_write_attempted": False,
                        "original_result_ref": evidence["result_ref"]}
                    state, published = "failed", False
                    if not heartbeat.lost.is_set():
                        publication_attempted = True
                        published = store.finish(job["id"], job["lease_token"], state, result=None, error=error)
            evidence.update(publication=("suppressed_lease_lost" if not publication_attempted and heartbeat.lost.is_set()
                                         else "accepted" if published is True else "suppressed_fence_rejected"),
                            requested_state="retry" if safe_retry else state)
            if published is not True and hasattr(runner, "close") and not stopping():
                evidence["post_attempt_cleanup"] = runner.close()
    except Exception as error:
        if hasattr(runner, "close") and not stopping():
            try:
                evidence["post_attempt_cleanup"] = runner.close()
            except Exception as cleanup_error:
                evidence["cleanup_error_type"] = type(cleanup_error).__name__
        evidence.update(error_type=type(error).__name__)
        if publication_attempted:
            # 为什么要有它：Redis回复丢失时首次写入可能已成功，不能再用不确定终态覆盖这次结果或触发第二次重排。
            evidence.update(publication="store_response_unknown", publication_error_type=type(error).__name__)
        elif not heartbeat.lost.is_set():
            try:
                published = store.finish(job["id"], job["lease_token"], "indeterminate" if job["kind"] == "agent" else "failed",
                    error={"code": "WORKER_ATTEMPT_UNCERTAIN", "type": type(error).__name__, "cost_unknown": True})
                evidence["publication"] = "accepted_uncertain" if published is True else "suppressed_fence_rejected"
            except Exception as publication_error:
                evidence.update(publication="store_unavailable", publication_error_type=type(publication_error).__name__)
    finally:
        stop_heartbeat()
        evidence["lease_lost"] = heartbeat.lost.is_set()
        evidence["shutdown_requested"] = stopping()
        evidence["shutdown_signal"] = shutdown.signal_number if shutdown is not None else None
        save(path / "attempt.json", evidence)
    return True


def main(argv=None):
    import redis

    from src.service_layer.config import ServiceConfig
    from src.service_layer.store import (
        CapacityExceeded,
        CapacityPolicyMismatch,
        CapacityStateError,
        RedisJobStore,
    )

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--worker-id", default=socket.gethostname() + ":" + str(os.getpid()) + ":" + uuid4().hex[:8])
    args = parser.parse_args(argv)
    config = ServiceConfig.from_env()
    from src.service_layer.sparse_process import SparseProcess
    shutdown = WorkerShutdown()
    handlers, client, runner = {}, None, None
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, shutdown.request)
        client = worker_redis_client(config.redis_url)
        store = RedisJobStore(client, namespace=config.namespace, retention_seconds=config.retention_seconds,
                             max_pending=config.max_pending, capacity_bytes=config.capacity_bytes,
                             cache_capacity_bytes=config.cache_capacity_bytes,
                             cache_ttl_seconds=config.query_cache_ttl)
        runner = SparseProcess(config)
        failures = 0
        while not shutdown.requested:
            try:
                store.recover_expired()
                worked = run_once(store, config, worker_id=args.worker_id, runner=runner, shutdown=shutdown)
            except (redis.RedisError, CapacityExceeded, CapacityPolicyMismatch, CapacityStateError) as error:
                # 为什么要有它：连接暂断不能终止常驻worker；只重试队列轮询，未知Agent仍由原租约恢复规则终结，不能本地重新派发。
                transient = (isinstance(error, (redis.ConnectionError, redis.TimeoutError,
                                               redis.exceptions.BusyLoadingError))
                             and not isinstance(error, redis.exceptions.AuthenticationError))
                # Scope: capacity policy/corruption failures are not transport retries.
                # 为什么要有它：旧数据、配置冲突或未完成写入不能靠重启循环清零；停止并保留原状态，GET仍可调查。
                retry = transient and not args.once and not shutdown.requested
                failures = min(failures + 1, 16)
                delay = min(30.0, max(config.poll_seconds, config.retry_delay_seconds) * 2 ** (failures-1))
                # 为什么要有它：Redis异常文本可能包含连接URL/密码；诊断只保存类型、是否续试和退避时间，不打印原异常。
                print(json.dumps({"event":"worker_store_unavailable", "error_type":type(error).__name__,
                    "retry":retry, "retry_delay_seconds":delay if retry else None}), file=sys.stderr, flush=True)
                if not retry:
                    return 0 if shutdown.requested and transient else 2
                runner.reap_idle()
                shutdown.wait(delay)
                continue
            failures = 0
            if args.once:
                return 0
            if not worked:
                shutdown.wait(config.poll_seconds)
        return 0
    finally:
        # Scope: worker-owned resources. 为什么要有它：resident清理失败也必须尝试释放Redis并恢复handler；内核/远端故障仍由容器硬停止界限处理。
        try:
            if runner is not None:
                runner.close()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
