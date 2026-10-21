"""One bounded Linux interpreter for explicitly model-free sparse requests.

Every command still calls execute_request; no QueryService/request/result is retained.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 为什么要有它：与原execute.py相同，隔离解释器只在固定仓库根加入正常业务入口。
from src.service_layer.execute import (  # noqa: E402
    _clock_boundary,
    execute_process,
    execute_request,
    reference,
    save,
)

MAX_JOBS = 128
IDLE_SECONDS = 60
MAX_COMMAND_BYTES = 65536


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _bound_path(value, root):
    path = Path(value).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path == Path(root).resolve():
        raise ValueError("SPARSE_PATH_ESCAPE")
    return path


def _file_identity(paths, check_active=lambda: None):
    values = []
    for path in sorted(set(paths)):
        check_active()
        if path.is_file():
            values.append({"path": str(path), "sha256": reference(path)["sha256"]})
    check_active()
    return values


def sparse_identity(request, work_root, *, check_active=lambda: None):
    """Fail closed before either process admits a resident query; no model imports."""
    from src.core.settings import load_settings, require_storage_path, resolve_storage_path

    if (not isinstance(request, dict) or request.get("schema") != "service-attempt-request.v1"
            or set(request) != {"schema", "job", "server_config"}):
        return None
    job, server = request["job"], request["server_config"]
    payload = job.get("payload")
    # Scope: explicit pure sparse requests only. 为什么要有它：默认值或配置回退不能偷偷引入模型/Agent副作用。
    if (job.get("kind") != "query" or not isinstance(payload, dict)
            or any(payload.get(key) is not False for key in ("answer", "rewrite", "rerank"))
            or payload.get("stream") not in (None, False)
            or Path(server["work_root"]).resolve() != Path(work_root).resolve()):
        return None
    path = Path(server["settings_path"]).resolve()
    expected = payload.get("_settings_sha256")
    if not isinstance(expected, str) or reference(path)["sha256"] != expected:
        return None
    settings = load_settings(path)
    if (reference(path)["sha256"] != expected
            or settings.retrieval.dense_enabled is not False
            or settings.retrieval.dense_provider != "disabled"
            or settings.rerank.enabled is not False
            or settings.vector_store.provider != "chroma"
            or getattr(settings.query, "rewrite_enabled", False) is not False):
        return None
    collection = payload.get("collection", "default")
    import re
    if not isinstance(collection, str) or re.fullmatch(r"\w[\w.-]{0,127}", collection) is None:
        return None
    bm25 = resolve_storage_path(settings, "db", "bm25", collection)
    chroma = require_storage_path(settings, settings.vector_store.persist_directory)
    # Scope: lifecycle invalidation, not a substitute for QueryService's content/cache identity.
    # 为什么要有它：常驻资源绑定settings、collection、源码和BM25，正文仍由QueryService逐次检查，SQLite维护字节不能冒充资料变化。
    sources = _file_identity((REPO / "src").rglob("*.py"), check_active)
    corpus = _file_identity(bm25.rglob("*"), check_active)
    database = chroma / "chroma.sqlite3"
    info = database.stat() if database.is_file() else None
    return {"settings_sha256": expected, "settings_path": str(path), "collection": collection,
            "server_sha256": _digest(server), "source_sha256": _digest(sources),
            "corpus_sha256": _digest(corpus), "bm25_root": str(bm25), "chroma_root": str(chroma),
            "database_file_identity": {"device": info.st_dev, "inode": info.st_ino} if info else None}


def _same_code_and_scope(before, after):
    return (isinstance(after, dict)
            and {k: v for k, v in before.items() if k != "corpus_sha256"}
            == {k: v for k, v in after.items() if k != "corpus_sha256"})


def _atomic_marker(path, value):
    temporary = path.with_name("." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fail if a prior/hostile marker already exists.
    finally:
        temporary.unlink(missing_ok=True)


def _parent_death_guard(parent_pid):
    if sys.platform != "linux" or parent_pid != 1 or os.getpgrp() != os.getpid():
        raise RuntimeError("SPARSE_REQUIRES_OWN_LINUX_PROCESS_GROUP")

    def terminate_group(_signal, _frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    # Scope: this daemon's session only. 为什么要有它：worker异常退出不能留下仍能执行请求的自有子进程树。
    signal.signal(signal.SIGTERM, terminate_group)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "parent-death registration failed")
    if os.getppid() != parent_pid:
        terminate_group(None, None)


def _validate_command(command, *, work_root, generation, sequence, validate_identity=True):
    if (not isinstance(command, dict) or set(command) != {
            "schema", "generation", "sequence", "request_ref", "result_path", "marker_path",
            "identity", "job_id", "attempt", "deadline_monotonic"}
            or command["schema"] != "sparse-command.v1" or command["generation"] != generation
            or type(command["sequence"]) is not int or command["sequence"] != sequence):
        raise ValueError("SPARSE_COMMAND_BINDING_INVALID")
    request_path = _bound_path(command["request_ref"]["path"], work_root)
    result_path = _bound_path(command["result_path"], work_root)
    marker_path = _bound_path(command["marker_path"], work_root)
    attempt_root = request_path.parent
    if (request_path.name != "request.json" or result_path != attempt_root / "result.json"
            or marker_path != attempt_root / "process" / "completed.json"
            or attempt_root.parent != Path(work_root).resolve() / "jobs" / command["job_id"]
            or not attempt_root.name.startswith(str(command["attempt"]) + "-")
            or reference(request_path) != command["request_ref"]
            or result_path.exists() or marker_path.exists()):
        raise ValueError("SPARSE_COMMAND_PATH_OR_DIGEST_INVALID")
    request = json.loads(request_path.read_bytes())
    if (request["job"]["id"] != command["job_id"] or request["job"]["attempt"] != command["attempt"]
            or validate_identity and sparse_identity(request, work_root) != command["identity"]):
        raise ValueError("SPARSE_REQUEST_IDENTITY_CHANGED")
    return request, request_path, result_path, marker_path


def child_main(work_root, parent_pid, generation, max_jobs=MAX_JOBS):
    _parent_death_guard(parent_pid)
    for sequence in range(1, max_jobs + 1):
        raw = sys.stdin.buffer.readline(MAX_COMMAND_BYTES + 1)
        if not raw:
            return 0
        if len(raw) > MAX_COMMAND_BYTES or not raw.endswith(b"\n"):
            return 2
        command = json.loads(raw)
        request, request_path, result_path, marker = _validate_command(
            command, work_root=work_root, generation=generation, sequence=sequence)
        if os.getppid() != parent_pid or time.monotonic() >= command["deadline_monotonic"]:
            return 2
        output = marker.parent
        completed = False
        exit_code = None
        error_type = None
        with (output / "stdout.bin").open("xb") as stdout, (output / "stderr.bin").open("xb") as stderr:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            try:
                exit_code = execute_request(request_path, result_path)
                envelope = json.loads(result_path.read_bytes())
                completed = True
                post_identity = sparse_identity(request, work_root)
                if (reference(request_path) != command["request_ref"]
                        or not _same_code_and_scope(command["identity"], post_identity)):
                    raise ValueError("SPARSE_SCOPE_CHANGED_DURING_EXECUTION")
            except Exception as error:
                error_type = type(error).__name__
                post_identity = None
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                ctypes.CDLL(None).fflush(None)
                with open(os.devnull, "wb") as idle:
                    os.dup2(idle.fileno(), 1)
                    os.dup2(idle.fileno(), 2)
        reusable = (completed and exit_code == 0 and error_type is None
                    and envelope.get("ok") is True
                    and envelope.get("result", {}).get("status") not in {"failed", "timed_out", "cancelled", "rejected"}
                    and sequence < max_jobs)
        _atomic_marker(marker, {"schema": "sparse-completion.v1", "generation": generation,
            "sequence": sequence, "child_pid": os.getpid(), "job_id": command["job_id"],
            "attempt": command["attempt"], "request_ref": command["request_ref"],
            "result_ref": reference(result_path) if result_path.is_file() else None,
            "identity": command["identity"], "post_identity": post_identity,
            "request_completed": completed, "request_exit_code": exit_code, "error_type": error_type,
            "reusable": reusable, "completed_monotonic": time.monotonic(),
            "stdout_ref": reference(output / "stdout.bin"), "stderr_ref": reference(output / "stderr.bin")})
        if not reusable:
            return 0 if error_type is None else 2
    return 0


class SparseProcess:
    """Worker-owned serial lifecycle; an inapplicable request uses the original runner."""

    def __init__(self, config, *, fallback=execute_process, max_jobs=MAX_JOBS, idle_seconds=IDLE_SECONDS):
        if type(max_jobs) is not int or not 1 <= max_jobs <= MAX_JOBS or idle_seconds <= 0:
            raise ValueError("SPARSE_LIFECYCLE_LIMIT_INVALID")
        self.config, self.fallback = config, fallback
        self.max_jobs, self.idle_seconds = max_jobs, idle_seconds
        self.process = None
        self.identity = None
        self.generation = None
        self.sequence = 0
        self.last_used = 0.0
        self.cleanup_pending = False

    def close(self):
        process = self.process
        self.identity = None
        if process is None:
            return "no_process_created"
        self.cleanup_pending = True
        # Scope: only the session created by this owner. 为什么要有它：异常、失租与边界回收不得影响其他worker或服务。
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=10)
        if process.poll() is None:
            raise RuntimeError("SPARSE_PROCESS_CLEANUP_UNCONFIRMED")
        # Scope: retain ownership until termination is confirmed. 为什么要有它：清理失败不能丢句柄后启动第二个resident。
        self.process = None
        self.cleanup_pending = False
        return "owned_process_group_terminated"

    def reap_idle(self):
        if self.process is not None and (self.cleanup_pending or self.process.poll() is not None
                or time.monotonic() - self.last_used >= self.idle_seconds):
            self.close()

    def __call__(self, argv, *, cwd, output, deadline, environment=None, cancel_event=None):
        entered = _clock_boundary()
        options = {"cwd": cwd, "output": output, "deadline": deadline,
                   "environment": environment, "cancel_event": cancel_event}
        def check_active():
            if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() >= deadline:
                raise TimeoutError("SPARSE_ADMISSION_CANCELLED_OR_DEADLINE")

        if (cancel_event is not None and cancel_event.is_set()) or time.monotonic() >= deadline:
            self.close()
            return self.fallback(argv, **options)
        identity = None
        request_path = Path(argv[argv.index("--request") + 1])
        result_path = Path(argv[argv.index("--result") + 1])
        try:
            request = json.loads(request_path.read_bytes())
            # Scope: qualified container worker is PID 1; other Linux keeps isolation.
            # 为什么要有它：native调用可能延迟Python信号处理，PID namespace退出提供父死整树终止兜底。
            if (sys.platform == "linux" and os.getpid() == 1
                    and getattr(self.config, "query_execution_mode", "resident_sparse") == "resident_sparse"):
                identity = sparse_identity(request, self.config.work_root, check_active=check_active)
        except (OSError, ValueError, KeyError, TypeError):
            identity = None
        if identity is None:
            self.close()
            return self.fallback(argv, **options)
        return self._run(request, request_path, result_path, identity, entered=entered, **options)

    def _run(self, request, request_path, result_path, identity, *, entered, cwd, output, deadline, environment, cancel_event):
        started = time.monotonic()
        output = _bound_path(output, self.config.work_root)
        output.mkdir(parents=True, exist_ok=False)
        marker = output / "completed.json"
        result = {"execution_mode": "resident_sparse", "started": False, "timeout": False,
                  "cancelled": False, "request_completed": False, "request_exit_code": None,
                  "process_alive": False, "reused": False, "exit_code": None,
                  "timing_boundaries": {"execute_entered": entered, "resident_admission_completed": _clock_boundary()},
                  "resident_module_entry_scope": "generation_start_not_request_start",
                  "provider_server_state": "not_established_by_client_process_cleanup"}

        def interrupted():
            if cancel_event is not None and cancel_event.is_set():
                result.update(cancelled=True, status="cancelled")
                return True
            if time.monotonic() >= deadline:
                result.update(timeout=True, status="timed_out")
                return True
            return False

        try:
            self.reap_idle()
            if self.identity != identity or self.sequence >= self.max_jobs:
                self.close()
            if interrupted():
                return result
            reused = self.process is not None
            if self.process is None:
                self.generation, self.sequence = uuid4().hex, 0
                command = [sys.executable, "-I", "-B", "-X", "utf8", str(Path(__file__).resolve()),
                           "--child", "--work-root", str(self.config.work_root), "--parent-pid", str(os.getpid()),
                           "--generation", self.generation]
                result["timing_boundaries"]["popen_requested"] = _clock_boundary()
                # Scope: this generation's first attempt only. 为什么要有它：初始化失败需原始stderr，后续请求不得共用可变日志摘要。
                with (output / "bootstrap-stdout.bin").open("xb") as bootstrap_out, (output / "bootstrap-stderr.bin").open("xb") as bootstrap_err:
                    self.process = subprocess.Popen(command, cwd=cwd, env=environment, stdin=subprocess.PIPE,
                        stdout=bootstrap_out, stderr=bootstrap_err, start_new_session=True)
                result["timing_boundaries"]["popen_returned"] = _clock_boundary()
                os.set_blocking(self.process.stdin.fileno(), False)
                result["spawn_argv"] = command
            process = self.process
            self.sequence += 1
            result.update(started=True, process_id=process.pid, generation=self.generation,
                          request_sequence=self.sequence, reused=reused,
                          process_tree_ownership="POSIX process group")
            command = {"schema": "sparse-command.v1", "generation": self.generation,
                "sequence": self.sequence, "request_ref": reference(request_path), "result_path": str(result_path.resolve()),
                "marker_path": str(marker), "identity": identity, "job_id": request["job"]["id"],
                "attempt": request["job"]["attempt"], "deadline_monotonic": deadline}
            _validate_command(command, work_root=self.config.work_root, generation=self.generation,
                              sequence=self.sequence, validate_identity=False)
            data = (json.dumps(command, ensure_ascii=False, allow_nan=False) + "\n").encode()
            if len(data) > MAX_COMMAND_BYTES:
                raise ValueError("SPARSE_COMMAND_TOO_LARGE")
            while data:
                if interrupted():
                    return result
                if process.poll() is not None:
                    raise RuntimeError("SPARSE_CHILD_EXITED_BEFORE_SEND")
                _, writable, _ = select.select([], [process.stdin], [], min(0.02, max(0, deadline-time.monotonic())))
                if writable:
                    try:
                        data = data[os.write(process.stdin.fileno(), data):]
                    except BlockingIOError:
                        pass
            result["timing_boundaries"]["command_sent"] = _clock_boundary()
            while True:
                if interrupted():
                    return result
                if marker.is_file():
                    completion = json.loads(marker.read_bytes())
                    self._validate_completion(completion, command, process.pid, output, deadline)
                    # Scope: marker is not authority after lease/deadline loss. 为什么要有它：迟到结果不能复活过期attempt。
                    if interrupted():
                        return result
                    result.update(status="request_completed", request_completed=completion["request_completed"],
                        request_exit_code=completion["request_exit_code"], completion_ref=reference(marker),
                        result_ref=completion["result_ref"], stdout_ref=completion["stdout_ref"], stderr_ref=completion["stderr_ref"])
                    result["timing_boundaries"]["completion_observed"] = _clock_boundary()
                    if completion["error_type"] is not None:
                        raise RuntimeError("SPARSE_REQUEST_EXCEPTION")
                    self.identity = completion["post_identity"]
                    self.last_used = time.monotonic()
                    if not completion["reusable"] or self.sequence >= self.max_jobs:
                        result["owned_process_tree_cleanup"] = self.close()
                    else:
                        result["owned_process_tree_cleanup"] = "resident_retained_for_bounded_reuse"
                    break
                if process.poll() is not None:
                    raise RuntimeError("SPARSE_CHILD_EXITED_WITHOUT_COMPLETION")
                time.sleep(min(0.01, max(0, deadline-time.monotonic())))
        except Exception as error:
            result.update(status="infrastructure_failed", error_type=type(error).__name__)
        finally:
            if result.get("status") != "request_completed":
                try:
                    result["owned_process_tree_cleanup"] = self.close()
                except Exception as error:
                    result.update(status="infrastructure_failed", cleanup_error_type=type(error).__name__)
            result.update(process_alive=self.process is not None and self.process.poll() is None,
                          wall_seconds=time.monotonic()-entered["monotonic_ns"]/1e9,
                          resident_execution_wall_seconds=time.monotonic()-started)
            result["timing_boundaries"]["cleanup_finished"] = _clock_boundary()
            # Scope: publish hashes only after fd redirection or confirmed termination.
            # 为什么要有它：清理未完成时日志仍可能增长，不能把暂时SHA称作不可变attempt证据。
            if result.get("completion_ref") is not None or self.process is None:
                for name in ("stdout", "stderr"):
                    path = output / (name + ".bin")
                    if path.is_file():
                        result[name + "_ref"] = reference(path)
                    bootstrap = output / ("bootstrap-" + name + ".bin")
                    if bootstrap.is_file():
                        result["bootstrap_" + name + "_ref"] = reference(bootstrap)
            else:
                result["log_capture_status"] = "unsealed_process_cleanup_pending"
            save(output / "process.json", result)
        return result

    @staticmethod
    def _validate_completion(completion, command, pid, output, deadline):
        if (not isinstance(completion, dict) or completion.get("schema") != "sparse-completion.v1"
                or any(completion.get(key) != command[key] for key in
                       ("generation", "sequence", "job_id", "attempt", "request_ref", "identity"))
                or completion.get("child_pid") != pid
                or type(completion.get("request_completed")) is not bool
                or type(completion.get("request_exit_code")) is not int
                or completion.get("request_exit_code") != 0 or not completion["request_completed"]
                or type(completion.get("reusable")) is not bool
                or type(completion.get("completed_monotonic")) not in (int, float)
                or not 0 < completion["completed_monotonic"] <= deadline
                or completion.get("result_ref") != reference(command["result_path"])
                or command["request_ref"] != reference(command["request_ref"]["path"])
                or not _same_code_and_scope(command["identity"], completion.get("post_identity"))
                or any(completion.get(name + "_ref") != reference(output / (name + ".bin")) for name in ("stdout", "stderr"))):
            raise ValueError("SPARSE_COMPLETION_BINDING_INVALID")


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args(argv)
    return child_main(args.work_root, args.parent_pid, args.generation)


if __name__ == "__main__":
    raise SystemExit(main())
