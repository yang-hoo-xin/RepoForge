"""Controlled lifecycle and binding checks; Linux process qualification is separate."""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.service_layer import sparse_process as sparse
from src.service_layer import worker
from src.service_layer.config import ServiceConfig
from tests.unit.test_service_worker import Store, config, runner_result


def request_files(root, name="job", attempt=1):
    path = root / "jobs" / name / f"{attempt}-controlled"
    path.mkdir(parents=True)
    request = {"schema": "service-attempt-request.v1", "job": {"id": name, "attempt": attempt,
        "kind": "query", "payload": {"query": name, "answer": False, "rewrite": False,
        "rerank": False, "request_id": "same-client-request"}, "timeout_seconds": 10},
        "server_config": {"work_root": str(root)}}
    sparse.save(path / "request.json", request)
    return ["python", "--request", str(path / "request.json"), "--result", str(path / "result.json")], path


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(sparse.sys, "platform", "linux")
    monkeypatch.setattr(sparse.os, "getpid", lambda: 1)
    monkeypatch.setattr(sparse.signal, "SIGKILL", 9, raising=False)
    identity = {"source_sha256": "fixed-code", "settings_sha256": "fixed-settings", "corpus_sha256": "fixed-corpus"}
    monkeypatch.setattr(sparse, "sparse_identity", lambda *a, **k: copy.deepcopy(identity))
    children, commands, killed = [], [], []
    behavior = {"mode": "ok"}
    actual_monotonic = time.monotonic
    clock = {"offset": 0}
    monkeypatch.setattr(sparse.time, "monotonic", lambda: actual_monotonic() + clock["offset"])

    class Pipe:
        def fileno(self):
            return 42
        def close(self):
            pass

    class Child:
        def __init__(self, *args, **kwargs):
            self.pid, self.returncode, self.stdin = 2000 + len(children), None, Pipe()
            children.append(self)
            assert kwargs["start_new_session"] is True
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            return self.returncode

    def kill(pid, sig):
        killed.append(pid)
        next(p for p in children if p.pid == pid).returncode = -9

    def send(fd, raw):
        command = json.loads(raw)
        commands.append(command)
        mode = behavior["mode"]
        if mode == "cancel":
            behavior["cancel"].set()
            return len(raw)
        if mode == "timeout":
            clock["offset"] += 20
            return len(raw)
        output = Path(command["marker_path"]).parent
        sparse.save(command["result_path"], {"schema": "service-attempt-result.v1", "job_id": command["job_id"],
            "attempt": command["attempt"], "ok": mode != "business_failure", "safe_retry": False,
            "result": {"status": "completed", "query": command["job_id"]}})
        (output / "stdout.bin").write_bytes(command["job_id"].encode())
        (output / "stderr.bin").write_bytes(b"")
        completion = {"schema": "sparse-completion.v1", **{k: command[k] for k in
            ("generation", "sequence", "job_id", "attempt", "request_ref", "identity")},
            "child_pid": children[-1].pid, "result_ref": sparse.reference(command["result_path"]),
            "post_identity": copy.deepcopy(identity), "request_completed": True, "request_exit_code": 0,
            "error_type": None, "reusable": mode != "business_failure", "completed_monotonic": time.monotonic(),
            "stdout_ref": sparse.reference(output / "stdout.bin"), "stderr_ref": sparse.reference(output / "stderr.bin")}
        if mode == "wrong_sequence":
            completion["sequence"] += 1
        if mode == "wrong_pid":
            completion["child_pid"] += 1
        if mode == "wrong_result_sha":
            completion["result_ref"]["sha256"] = "0" * 64
        if mode == "late_marker":
            completion["completed_monotonic"] = command["deadline_monotonic"] + 1
        sparse._atomic_marker(Path(command["marker_path"]), completion)
        return len(raw)

    monkeypatch.setattr(sparse.subprocess, "Popen", Child)
    monkeypatch.setattr(sparse.os, "set_blocking", lambda *a: None, raising=False)
    monkeypatch.setattr(sparse.os, "killpg", kill, raising=False)
    monkeypatch.setattr(sparse.os, "write", send)
    monkeypatch.setattr(sparse.select, "select", lambda r, w, e, timeout: ([], w, []))
    runner = sparse.SparseProcess(SimpleNamespace(work_root=tmp_path, query_execution_mode="resident_sparse"))

    def run(name, *, deadline=None, cancel=None):
        argv, path = request_files(tmp_path, name)
        return runner(argv, cwd=tmp_path, output=path / "process", deadline=deadline or time.monotonic()+10,
                      environment={}, cancel_event=cancel)
    yield SimpleNamespace(runner=runner, run=run, identity=identity, children=children, commands=commands,
                          killed=killed, behavior=behavior, root=tmp_path)
    runner.close()


def test_reuse_has_request_completion_not_fake_exit_and_private_logs(lifecycle):
    first = lifecycle.run("first")
    before = Path(first["stdout_ref"]["path"]).read_bytes()
    second = lifecycle.run("second")
    assert first["request_completed"] and second["request_completed"]
    assert first["exit_code"] is second["exit_code"] is None
    assert not first["reused"] and second["reused"] and second["process_alive"]
    assert first["generation"] == second["generation"] and first["request_sequence"] == 1 and second["request_sequence"] == 2
    assert Path(first["stdout_ref"]["path"]).read_bytes() == before
    assert first["stdout_ref"] != second["stdout_ref"]
    # Scope: same client request_id across jobs. 为什么要有它：只复用解释器，不能复用QueryService的进程内请求记录。
    assert [json.loads(Path(c["result_path"]).read_bytes())["result"]["query"] for c in lifecycle.commands] == ["first", "second"]


@pytest.mark.parametrize("field", ["settings_sha256", "source_sha256", "corpus_sha256"])
def test_identity_drift_rebuilds_before_next_request(lifecycle, field):
    first = lifecycle.run("first")
    lifecycle.identity[field] = "new"
    second = lifecycle.run("second")
    assert first["generation"] != second["generation"] and not second["reused"]
    assert first["process_id"] in lifecycle.killed


@pytest.mark.parametrize("mode", ["wrong_sequence", "wrong_pid", "wrong_result_sha", "late_marker"])
def test_bad_or_late_marker_kills_generation_then_next_job_recovers(lifecycle, mode):
    lifecycle.behavior["mode"] = mode
    bad = lifecycle.run("bad")
    assert bad["status"] == "infrastructure_failed" and not bad["process_alive"]
    assert bad["process_id"] in lifecycle.killed
    lifecycle.behavior["mode"] = "ok"
    good = lifecycle.run("good")
    assert good["request_completed"] and good["generation"] != bad["generation"]


@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_inflight_cancel_or_deadline_kills_generation_and_recovers(lifecycle, mode):
    lifecycle.behavior["mode"] = mode
    cancel = threading.Event()
    lifecycle.behavior["cancel"] = cancel
    bad = lifecycle.run("bad", cancel=cancel, deadline=time.monotonic() + 10)
    assert bad["cancelled" if mode == "cancel" else "timeout"] is True
    assert not bad["process_alive"] and bad["process_id"] in lifecycle.killed
    lifecycle.behavior["mode"] = "ok"
    good = lifecycle.run("good")
    assert good["request_completed"] and not good["reused"]


def test_business_failure_and_max_jobs_both_destroy_resident(lifecycle):
    lifecycle.runner.max_jobs = 2
    first, second = lifecycle.run("first"), lifecycle.run("second")
    assert second["reused"] and not second["process_alive"]
    third = lifecycle.run("third")
    assert not third["reused"] and third["generation"] != first["generation"]
    lifecycle.behavior["mode"] = "business_failure"
    failed = lifecycle.run("failed")
    assert failed["request_completed"] and not failed["process_alive"]


def test_idle_cleanup_is_bounded(lifecycle):
    result = lifecycle.run("first")
    lifecycle.runner.last_used = time.monotonic() - 61
    lifecycle.runner.reap_idle()
    assert lifecycle.runner.process is None and result["process_id"] in lifecycle.killed


@pytest.mark.parametrize("failure", ["kill_failed", "wait_timeout"])
def test_cleanup_failure_keeps_owner_and_blocks_new_resident(lifecycle, monkeypatch, failure):
    first = lifecycle.run("first")
    original = lifecycle.runner.process
    kill, wait = sparse.os.killpg, original.wait
    if failure == "kill_failed":
        def fail_kill(*args):
            raise PermissionError("controlled cleanup denial")
        monkeypatch.setattr(sparse.os, "killpg", fail_kill)
    else:
        monkeypatch.setattr(sparse.os, "killpg", lambda *args: None)
        def fail_wait(timeout):
            raise sparse.subprocess.TimeoutExpired("controlled-child", timeout)
        original.wait = fail_wait
    with pytest.raises((PermissionError, sparse.subprocess.TimeoutExpired)):
        lifecycle.runner.close()
    assert lifecycle.runner.process is original and lifecycle.runner.cleanup_pending
    blocked = lifecycle.run("blocked")
    assert blocked["status"] == "infrastructure_failed" and len(lifecycle.children) == 1
    assert lifecycle.runner.process is original and lifecycle.runner.cleanup_pending
    monkeypatch.setattr(sparse.os, "killpg", kill)
    original.wait = wait
    recovered = lifecycle.run("recovered")
    assert recovered["request_completed"] and recovered["generation"] != first["generation"]
    assert len(lifecycle.children) == 2 and not lifecycle.runner.cleanup_pending


@pytest.mark.parametrize("reason", ["ineligible", "windows", "linux_non_pid1", "isolated"])
def test_fallback_closes_idle_resident_before_isolated_path(lifecycle, monkeypatch, reason):
    first = lifecycle.run("first")
    if reason == "ineligible":
        monkeypatch.setattr(sparse, "sparse_identity", lambda *a, **k: None)
    elif reason == "windows":
        monkeypatch.setattr(sparse.sys, "platform", "win32")
    elif reason == "linux_non_pid1":
        monkeypatch.setattr(sparse.os, "getpid", lambda: 123)
    else:
        lifecycle.runner.config.query_execution_mode = "isolated"
    lifecycle.runner.fallback = lambda *a, **k: {"isolated": True, "resident_closed": lifecycle.runner.process is None}
    result = lifecycle.run("second")
    assert result == {"isolated": True, "resident_closed": True}
    assert first["process_id"] in lifecycle.killed


@pytest.mark.parametrize("expired", [False, True])
def test_preexpired_or_precancelled_never_hashes_or_spawns(lifecycle, monkeypatch, expired):
    monkeypatch.setattr(sparse, "sparse_identity", lambda *a, **k: pytest.fail("admission after budget loss"))
    lifecycle.runner.fallback = lambda *a, **k: {"stopped": True}
    cancel = threading.Event()
    if not expired:
        cancel.set()
    result = lifecycle.run("stop", cancel=cancel, deadline=time.monotonic()-1 if expired else None)
    assert result == {"stopped": True} and not lifecycle.children


def test_worker_accepts_bound_request_completion_without_os_exit(tmp_path):
    store = Store()
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner_result(
        {"ok": True, "safe_retry": False, "result": {"status": "completed"}},
        execution_mode="resident_sparse", request_completed=True, request_exit_code=0,
        process_alive=True, status="request_completed", exit_code=None))
    assert store.finished[0][0][2] == "completed"


@pytest.mark.parametrize("field,value", [("request_completed", False), ("request_exit_code", 7), ("status", "alive")])
def test_worker_does_not_publish_uncompleted_resident(tmp_path, field, value):
    store = Store()
    process = dict(execution_mode="resident_sparse", request_completed=True, request_exit_code=0,
                   process_alive=True, status="request_completed", exit_code=None)
    process[field] = value
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner_result(
        {"ok": True, "safe_retry": False, "result": {"status": "completed"}}, **process))
    assert store.finished[0][0][2] == "failed"


def test_config_mode_is_explicit_and_validated(monkeypatch, tmp_path):
    assert ServiceConfig(work_root=tmp_path).query_execution_mode == "resident_sparse"
    monkeypatch.setenv("SERVICE_QUERY_EXECUTION_MODE", "isolated")
    assert ServiceConfig.from_env().query_execution_mode == "isolated"
    with pytest.raises(ValueError, match="execution mode"):
        ServiceConfig(query_execution_mode="fork")
    with pytest.raises(ValueError, match="LIMIT"):
        sparse.SparseProcess(SimpleNamespace(), max_jobs=129)


def test_parent_death_guard_registers_group_cleanup_and_rechecks_parent(monkeypatch):
    monkeypatch.setattr(sparse.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(sparse.sys, "platform", "linux")
    monkeypatch.setattr(sparse.os, "getpid", lambda: 9)
    monkeypatch.setattr(sparse.os, "getpgrp", lambda: 9, raising=False)
    monkeypatch.setattr(sparse.os, "getppid", lambda: 999)
    callbacks, registered, killed = [], [], []
    monkeypatch.setattr(sparse.signal, "signal", lambda sig, callback: callbacks.append(callback))
    monkeypatch.setattr(sparse.ctypes, "CDLL", lambda *a, **k: SimpleNamespace(prctl=lambda *a: registered.append(a) or 0))
    monkeypatch.setattr(sparse.os, "killpg", lambda *args: killed.append(args), raising=False)
    sparse._parent_death_guard(1)
    assert registered == [(1, sparse.signal.SIGTERM, 0, 0, 0)] and killed == [(9, sparse.signal.SIGKILL)]
    callbacks[0](None, None)
    assert len(killed) == 2


@pytest.fixture
def admission(tmp_path, monkeypatch):
    argv, directory = request_files(tmp_path)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("controlled settings", encoding="utf-8")
    request = json.loads((directory / "request.json").read_bytes())
    request["server_config"]["settings_path"] = str(settings_path)
    request["job"]["payload"]["_settings_sha256"] = sparse.reference(settings_path)["sha256"]
    settings = SimpleNamespace(storage_root=str(tmp_path),
        retrieval=SimpleNamespace(dense_enabled=False, dense_provider="disabled"),
        rerank=SimpleNamespace(enabled=False, provider="none"), query=SimpleNamespace(rewrite_enabled=False),
        vector_store=SimpleNamespace(provider="chroma", persist_directory=str(tmp_path / "db" / "chroma")))
    monkeypatch.setattr("src.core.settings.load_settings", lambda _: settings)
    monkeypatch.setattr(sparse, "REPO", tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "owner.py").write_text("VERSION = 1", encoding="utf-8")
    return request, settings, tmp_path


@pytest.mark.parametrize("mode", ["agent", "answer", "rewrite", "rerank", "missing_flag", "dense", "provider", "settings_rewrite", "settings_rerank", "settings_drift"])
def test_shared_parent_child_admission_fails_closed(admission, mode):
    request, settings, root = admission
    assert sparse.sparse_identity(request, root) is not None
    if mode == "agent":
        request["job"]["kind"] = "agent"
    elif mode in {"answer", "rewrite", "rerank"}:
        request["job"]["payload"][mode] = True
    elif mode == "missing_flag":
        request["job"]["payload"].pop("rewrite")
    elif mode == "dense":
        settings.retrieval.dense_enabled = True
    elif mode == "provider":
        settings.retrieval.dense_provider = "vector_store"
    elif mode == "settings_rewrite":
        settings.query.rewrite_enabled = True
    elif mode == "settings_rerank":
        settings.rerank.enabled = True
    else:
        Path(request["server_config"]["settings_path"]).write_text("changed")
    assert sparse.sparse_identity(request, root) is None


def test_identity_binds_collection_and_actual_source_bytes(admission):
    request, _, root = admission
    original = sparse.sparse_identity(request, root)
    request["job"]["payload"]["collection"] = "another"
    assert sparse.sparse_identity(request, root) != original
    request["job"]["payload"]["collection"] = "default"
    (root / "src" / "owner.py").write_text("VERSION = 2")
    assert sparse.sparse_identity(request, root)["source_sha256"] != original["source_sha256"]


def test_disabled_rerank_provider_does_not_change_original_load_settings(admission):
    request, settings, root = admission
    settings.rerank.provider = "cross_encoder"
    assert sparse.sparse_identity(request, root) is not None


def test_sqlite_maintenance_bytes_do_not_rebuild_but_bm25_does(admission):
    request, settings, root = admission
    database = Path(settings.vector_store.persist_directory) / "chroma.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"before-maintenance")
    original = sparse.sparse_identity(request, root)
    database.write_bytes(b"after-maintenance-bigger")
    database.with_name("chroma.sqlite3-wal").write_bytes(b"normal wal")
    assert sparse.sparse_identity(request, root) == original
    bm25 = root / "db" / "bm25" / "default"
    bm25.mkdir(parents=True)
    (bm25 / "index.json").write_text("new index")
    assert sparse.sparse_identity(request, root)["corpus_sha256"] != original["corpus_sha256"]


@pytest.mark.parametrize("mutate", ["path_escape", "request_digest", "generation", "sequence", "job_id", "attempt", "identity"])
def test_child_rejects_command_path_and_binding_changes(lifecycle, mutate):
    lifecycle.run("first")
    command = copy.deepcopy(lifecycle.commands[0])
    argv, path = request_files(lifecycle.root, "new")
    command.update(request_ref=sparse.reference(path / "request.json"), result_path=str(path / "result.json"),
        marker_path=str(path / "process" / "completed.json"), job_id="new")
    if mutate == "path_escape":
        command["result_path"] = str(lifecycle.root.parent / "outside-result.json")
    elif mutate == "request_digest":
        command["request_ref"]["sha256"] = "0" * 64
    elif mutate in {"generation", "job_id"}:
        command[mutate] = "wrong"
    elif mutate in {"sequence", "attempt"}:
        command[mutate] += 1
    else:
        command["identity"]["source_sha256"] = "wrong"
    with pytest.raises(ValueError):
        sparse._validate_command(command, work_root=lifecycle.root, generation=lifecycle.runner.generation, sequence=1)


def test_worker_lease_loss_after_completion_closes_idle_resident(tmp_path, monkeypatch):
    class Lost(worker.LeaseHeartbeat):
        def start(self):
            self.lost.set()
        def stop(self):
            pass
    monkeypatch.setattr(worker, "LeaseHeartbeat", Lost)
    class Runner:
        closed = False
        def __call__(self, *args, **kwargs):
            return runner_result({"ok": True, "safe_retry": False, "result": {"status": "completed"}},
                execution_mode="resident_sparse", request_completed=True, request_exit_code=0,
                process_alive=True, status="request_completed", exit_code=None)(*args, **kwargs)
        def close(self):
            self.closed = True
            return "owned_process_group_terminated"
    runner, store = Runner(), Store()
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner)
    assert runner.closed and not store.finished and not store.retried
