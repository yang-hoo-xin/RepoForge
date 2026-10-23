"""Run one preregistered benchmark arm and its host acceptance under one wall deadline."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID


def ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read(reference):
    if ref(reference["path"]) != reference:
        raise ValueError("BENCHMARK_INPUT_CHANGED")
    return json.loads(Path(reference["path"]).read_bytes())


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return ref(path)


def codex_argv(executable, workspace, last_message, deny_roots):
    """Explicit fixed model/effort; no invented total token or provider-call cap."""
    filesystem = {":root": "read", ":workspace_roots": "write"}
    filesystem.update({Path(path).resolve().as_posix(): "deny" for path in deny_roots})
    filesystem[(Path(workspace) / "materials").resolve().as_posix()] = "read"
    table = "{" + ",".join(json.dumps(k) + "=" + json.dumps(v) for k, v in filesystem.items()) + "}"
    argv = [str(executable), "--no-daemon", "exec", "--ignore-user-config", "--ignore-rules", "--strict-config",
        "--skip-git-repo-check", "--json", "--color", "never", "--model", "gpt-5.6-sol",
        "--cd", str(workspace), "--output-last-message", str(last_message)]
    settings = ['model_reasoning_effort="medium"', 'approval_policy="never"', 'web_search="disabled"',
        'default_permissions="benchmark"', 'permissions.benchmark.filesystem=' + table,
        'permissions.benchmark.network.enabled=false', 'windows.sandbox="elevated"',
        'shell_environment_policy.inherit="core"', 'shell_environment_policy.ignore_default_excludes=false',
        'project_doc_max_bytes=0', 'mcp_servers={}']
    for value in settings:
        argv.extend(["-c", value])
    for feature in ("apps", "plugins", "hooks", "memories", "browser_use", "browser_use_external", "computer_use",
                    "multi_agent", "remote_plugin", "skill_search", "skill_mcp_dependency_install", "goals", "image_generation"):
        argv.extend(["--disable", feature])
    argv.extend(["--enable", "skip_host_skill_discovery", "-"])
    return argv


def codex_resume_argv(initial_argv, thread_id, last_message):
    """Keep the frozen model/policy on an exact native session, never --last."""
    if str(UUID(thread_id)) != thread_id:
        raise ValueError("BENCHMARK_EXACT_CODEX_SESSION_REQUIRED")
    boundary = initial_argv.index("exec")
    result = [*initial_argv[:boundary + 1], "resume"]
    index = boundary + 1
    while index < len(initial_argv):
        value = initial_argv[index]
        if value in {"--cd", "--color"}:
            index += 2
            continue
        if value == "--output-last-message":
            result.extend([value, str(last_message)])
            index += 2
            continue
        if value != "-":
            result.append(value)
        index += 1
    result.extend([thread_id, "-"])
    return result


class WindowsJob:
    """Own every descendant of a suspended child before it starts executing."""
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", w.DWORD),
                ("min_working_set", ctypes.c_size_t), ("max_working_set", ctypes.c_size_t), ("active_limit", w.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        settings = Extended()
        settings.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.handle or not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(settings), ctypes.sizeof(settings)):
            raise ctypes.WinError(ctypes.get_last_error())

    def start(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        native = ctypes.WinDLL("ntdll")
        native.NtResumeProcess.argtypes = [ctypes.c_void_p]
        native.NtResumeProcess.restype = ctypes.c_long
        status = native.NtResumeProcess(int(process._handle))
        if status != 0:
            raise OSError("BENCHMARK_PROCESS_RESUME_FAILED", status)

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 124):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        self.kernel.CloseHandle(self.handle)


def execute(argv, *, cwd, stdin_path, output, deadline, environment=None):
    if os.name != "nt":
        raise ValueError("BENCHMARK_RUNNER_CURRENTLY_VERIFIED_FOR_WINDOWS_ONLY")
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError("BENCHMARK_DEADLINE_INVALID")
    child_environment = dict(os.environ if environment is None else environment)
    clock, utc = time.monotonic(), time.time()
    deadline_utc = utc + (deadline - clock)
    for inherited in (os.environ.get("REPOFORGE_TASK_DEADLINE_UTC"),
                      child_environment.get("REPOFORGE_TASK_DEADLINE_UTC")):
        if inherited is not None:
            inherited = float(inherited)
            if not math.isfinite(inherited):
                raise ValueError("BENCHMARK_INHERITED_DEADLINE_INVALID")
            deadline_utc = min(deadline_utc, inherited)
            deadline = min(deadline, clock + (deadline_utc - utc))
    # 为什么要有它：正常入口与恢复需要原截止事实，嵌套进程不得重新取得更长执行窗口。
    child_environment["REPOFORGE_TASK_DEADLINE_UTC"] = str(deadline_utc)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {"status": "deadline_exhausted_before_start", "started": False, "argv": argv,
                "deadline_utc": deadline_utc}
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    job, process = WindowsJob(), None
    result = {"argv": argv, "cwd": str(cwd), "started_at": started, "timeout": False, "started": False,
        "deadline_utc": deadline_utc, "deadline_scope": "owned_process_tree; inherited_earlier_deadline_retained"}
    try:
        with (output / "stdout.bin").open("xb") as stdout, (output / "stderr.bin").open("xb") as stderr:
            with Path(stdin_path).open("rb") if stdin_path else open(os.devnull, "rb") as stdin:
                # 为什么要有它：600秒截止必须覆盖整棵自有进程树；先加入Job再运行，不能只杀父CLI却让编译或工具继续修改。
                process = subprocess.Popen(argv, cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr,
                    env=child_environment, creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004)
                job.start(process)
                result.update(started=True, process_id=process.pid, process_tree_ownership="Windows Job Object")
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    result["timeout"] = True
                    job.terminate()
                    process.wait(timeout=15)
                result["exit_code"] = process.returncode
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error), status="infrastructure_failed")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=15)
        job.close()
    result.update(status=result.get("status", "timed_out" if result["timeout"] else "exited"), finished_at=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.monotonic() - clock, stdout_ref=ref(output / "stdout.bin"), stderr_ref=ref(output / "stderr.bin"),
        owned_process_tree_cleanup="job_closed_kill_on_close",
        broker_descendant_cleanup="not_established_by_generic_job_ownership",
        side_effects="inspect_final_workspace_even_on_nonzero_or_timeout; broker work may require separate stop evidence")
    save(output / "process.json", result)
    return result


def codex_events(path):
    events, invalid = [], []
    for number, line in enumerate(Path(path).read_bytes().splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("object required")
            events.append(row)
        except (ValueError, UnicodeError):
            invalid.append({"line": number, "sha256": hashlib.sha256(line).hexdigest()})
    turns = [row for row in events if row.get("type") in {"turn.completed", "turn.failed"}]
    usage = [row["usage"] for row in turns if isinstance(row.get("usage"), dict)]
    return {"event_count": len(events), "invalid_jsonl_lines": invalid,
        "thread_ids": [row.get("thread_id") for row in events if row.get("type") == "thread.started"],
        "terminal_events": turns, "native_usage_records": usage,
        "usage_totals": {name: sum(row[name] for row in usage) if usage and all(type(row.get(name)) is int for row in usage) else None
            for name in ("input_tokens", "cached_input_tokens", "output_tokens")},
        "provider_send_count": None, "provider_send_count_scope": "not inferred from item or tool counts",
        "tool_items": [row for row in events if row.get("type") == "item.completed"
            and row.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}]}


def source_snapshot(workspace):
    original = Path(workspace)
    if original.is_symlink() or getattr(original.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError("BENCHMARK_WORKSPACE_REPARSE_POINT_NOT_ALLOWED")
    root = original.resolve(strict=True)
    result = {}
    pending = [root]
    while pending:
        for entry in sorted(os.scandir(pending.pop()), key=lambda item: item.name):
            path = Path(entry.path)
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or getattr(metadata, "st_file_attributes", 0) & 0x400 or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError("BENCHMARK_WORKSPACE_REPARSE_POINT_NOT_ALLOWED")
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
            elif entry.is_file(follow_symlinks=False) and not {"bin", "obj", ".checks", ".git"}.intersection(path.relative_to(root).parts):
                result[path.relative_to(root).as_posix()] = ref(path)["sha256"]
    return result


def run(plan_reference):
    plan = read(plan_reference)
    protocol = read(plan["protocol_ref"])
    if (plan["task_id"] not in protocol["task_ids"] or plan["arm"] not in {"platform", "codex"}
            or plan["attempt"] != 1 or plan["wall_seconds"] != protocol["proposed_shared_wall_budget_seconds"]
            or not plan.get("public_checker_command") or not plan.get("acceptance_command")):
        raise ValueError("BENCHMARK_PREREGISTRATION_OR_CHECKER_REQUIRED")
    for reference in plan["input_refs"]:
        if ref(reference["path"]) != reference:
            raise ValueError("BENCHMARK_BOUND_INPUT_CHANGED")
    workspace = Path(plan["workspace"]).resolve(strict=True)
    output = Path(plan["output_root"]).resolve()
    if output.is_relative_to(workspace) or workspace.is_relative_to(output):
        raise ValueError("BENCHMARK_HOST_RECORDS_MUST_BE_SEPARATE")
    output.mkdir(parents=True, exist_ok=False)
    audit_root = Path(plan["source_audit_root"]).resolve(strict=True)
    if not audit_root.is_relative_to(workspace):
        raise ValueError("BENCHMARK_SOURCE_AUDIT_ROOT_OUTSIDE_WORKSPACE")
    before = source_snapshot(audit_root)
    save(output / "admission.json", {"plan_ref": plan_reference, "source_before": before, "started_at": datetime.now(timezone.utc).isoformat()})
    start = time.monotonic()
    deadline = start + plan["wall_seconds"]
    agent = execute(plan["command"], cwd=workspace, stdin_path=plan["prompt_path"], output=output / "agent", deadline=deadline)
    events = codex_events(agent["stdout_ref"]["path"]) if plan["arm"] == "codex" and agent.get("started") else None
    if events is not None:
        save(output / "native-event-summary.json", events)
    after = source_snapshot(audit_root)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    forbidden = [path for path in changed if path not in plan["allowed_source_paths"]]
    acceptance = None
    if not agent.get("timeout") and agent.get("exit_code") == 0 and not forbidden:
        remaining = max(0, deadline - time.monotonic())
        command = [value.format(workspace=str(workspace), output_root=str(output), remaining_seconds=f"{remaining:.3f}", arm=plan["arm"])
                   for value in plan["acceptance_command"]]
        acceptance = execute(command, cwd=workspace, stdin_path=None, output=output / "acceptance", deadline=deadline)
    result = {"status": "recorded_first_outcome", "task_id": plan["task_id"], "arm": plan["arm"], "plan_ref": plan_reference,
        "agent_process": agent, "acceptance_process": acceptance, "changed_source_paths": changed,
        "forbidden_source_changes": forbidden, "source_before": before, "source_after_agent": after,
        "source_audit_root": str(audit_root), "source_after_acceptance": source_snapshot(audit_root), "wall_seconds": time.monotonic() - start,
        "wall_budget_seconds": plan["wall_seconds"], "delivery_success": None,
        "delivery_success_basis": "host acceptance evidence must be evaluated; CLI exit or agent text alone never counts as delivery",
        "native_usage": events["usage_totals"] if events is not None else None}
    return save(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    print(json.dumps(run(ref(args.plan))))


if __name__ == "__main__":
    main()
