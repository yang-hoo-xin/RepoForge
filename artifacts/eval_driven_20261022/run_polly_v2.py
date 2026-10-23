"""Freeze product code and run the public Polly task using v2, without scripted actions.

为什么要有它：真实B选择必须来自模型；独立代码快照允许其他实施继续而不改变本次模型运行的执行器身份。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
BASE = "5d81f99ef5f523032c17b6b59b32d575ec9dfb06"
SOURCE = Path("C:/tmp/rfext-polly-2164")
TASK = Path("C:/tmp/rfext-polly-2164-task")
DOTNET = Path("C:/tmp/rf-dotnet-8.0.302/sdk/dotnet.exe")
MODEL = "qwen3-coder:30b"


def ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    return ref(path)


def freeze_code(destination):
    if destination.exists():
        raise ValueError("CODE_SNAPSHOT_ALREADY_EXISTS")
    destination.mkdir()
    manifest = []
    for folder in ("src", "scripts", "config"):
        for path in sorted((REPO / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            before = ref(path)
            relative = path.relative_to(REPO)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if ref(path) != before or ref(target)["sha256"] != before["sha256"]:
                raise ValueError("CODE_CHANGED_DURING_SNAPSHOT")
            manifest.append({"path": relative.as_posix(), "sha256": before["sha256"]})
    for row in manifest:
        if ref(REPO / row["path"])["sha256"] != row["sha256"]:
            raise ValueError("CODE_CHANGED_DURING_SNAPSHOT")
    return save(destination / "snapshot.json", {"files": manifest,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "includes_working_tree": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-snapshot", type=Path, required=True)
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--question", default="Observe autonomous investigation, revision and public acceptance under the frozen capability set.")
    args = parser.parse_args()
    root, snapshot = args.root.resolve(), args.code_snapshot.resolve()
    if root.exists():
        raise ValueError("RUN_ALREADY_EXISTS")
    code_ref = freeze_code(snapshot)
    root.mkdir()
    public_patch = REPO / "artifacts/eval_driven_20261021/polly-2164-public-reproduction.patch"
    qualification_path = Path("C:/tmp/rfext-polly-2164-public-qualification/qualification.json")
    qualification = json.loads(qualification_path.read_text())
    if not qualification["qualified"] or qualification["public_reproduction_ref"]["sha256"] != ref(public_patch)["sha256"]:
        raise ValueError("PUBLIC_REPRODUCTION_NOT_QUALIFIED")
    source = root / "input/Polly"
    subprocess.run(["git", "-C", str(SOURCE), "worktree", "add", "--detach", str(source), BASE],
                   check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    subprocess.run(["git", "apply", str(public_patch)], cwd=source, check=True,
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    allowed = sorted(p.relative_to(source).as_posix() for folder in ("src/Polly.Core", "src/LegacySupport")
                     for p in (source / folder).rglob("*.cs"))
    settings = yaml.safe_load((snapshot / "config/settings.yaml").read_text(encoding="utf-8"))
    settings["task_memory"] = {"enabled": False, "store_root": None, "max_context_chars": 3000}
    settings["llm"].update(provider="ollama", model=MODEL, base_url="http://127.0.0.1:11434", max_tokens=4096)
    settings_path = root / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
    profile = {"profile_version": "csharp_repository_profile_v1", "dotnet_ref": ref(DOTNET),
        "project_path": "src/Polly.Core/Polly.Core.csproj", "test_project_path": "test/Polly.Core.Tests/Polly.Core.Tests.csproj",
        "nuget_config_path": "NuGet.config", "configuration": "Release", "allowed_edit_paths": allowed,
        "build_properties": {"MinVerSkip": "true"},
        "expected_artifacts": ["artifacts/bin/Polly.Core/release_net8.0/Polly.Core.dll"],
        "test_command": ["test", "{test_project}", "--framework", "net8.0", "--configuration", "{configuration}",
            "-p:MinVerSkip=true", "-p:CollectCoverage=false", "--logger", "trx;LogFileName={report_path}"],
        "test_report": {"format": "trx", "path": "public-net8.trx"}, "timeout_seconds": 900}
    profile_ref = save(root / "repository-profile.json", profile)
    arguments = ["--target-project", str(source), "--workspace-root", str(root / "runs"),
        "--settings", str(settings_path), "--repository-profile", profile_ref["path"],
        "--local-ollama", "--ollama-model", MODEL, "--ollama-base-url", "http://127.0.0.1:11434",
        "--ollama-context-window", "65536", "--ollama-max-output-tokens", "4096", "--ollama-timeout-seconds", "600",
        "--model-call-budget-policy", "meter_only", "--source-retrieval", "rg_live", "--confirm-isolated-write",
        "--allow-build", "--defer-approval", "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
        "--constraint-policy", "source_priority_v1", "--task-agent-edit-paths", json.dumps(allowed)]
    execution = root / "execution.json"
    if args.learn:
        learning = save(root / "experience-config.json", {"schema_version": "task_agent_experience_config_v1",
            "store_root": str(root / "strategy-memory"), "snapshot_ref": None, "extract_on_terminal": True})
        arguments.extend(["--task-agent-experience-config", learning["path"]])
    save(execution, {"arguments": arguments})
    sys.path.insert(0, str(snapshot))
    from scripts.dsh_run_request import prepare_new_task

    problem = TASK / "solver-input/problem_statement.txt"
    allocated = prepare_new_task(execution_config=str(execution), description=problem.read_text(encoding="utf-8"))
    started = save(root / "started.json", {"code_snapshot_ref": code_ref, "runner_ref": ref(Path(__file__)),
        "evaluation_question": args.question, "public_problem_ref": ref(problem),
        "public_reproduction_ref": ref(public_patch), "qualification_ref": ref(qualification_path),
        "allocated_config": allocated, "model": MODEL, "memory_enabled": False,
        "terminal_learning_enabled": args.learn, "hidden_material_supplied": False,
        "scripted_actions": False, "outer_supplied_solution": False})
    env = {**os.environ, "PYTHONPATH": str(snapshot), "MinVerSkip": "true", "DOTNET_CLI_UI_LANGUAGE": "en-US"}
    try:
        process = subprocess.run([sys.executable, "-B", str(snapshot / "scripts/dsh_run_request.py"),
            "--execution-config", allocated], cwd=snapshot, env=env, input=problem.read_bytes(),
            capture_output=True, timeout=1800, creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired as error:
        (root / "stdout.txt").write_bytes(error.stdout or b"")
        (root / "stderr.txt").write_bytes(error.stderr or b"")
        save(root / "result.json", {"status": "runner_timeout", "started_ref": started, "whole_task_delivered": False})
        raise
    (root / "stdout.json").write_bytes(process.stdout)
    (root / "stderr.txt").write_bytes(process.stderr)
    parsed = json.loads(process.stdout) if process.stdout.strip().startswith(b"{") else {}
    result = {"status": "process_completed", "exit_code": process.returncode, "started_ref": started,
        "product_status": parsed.get("status"), "task_outcome": parsed.get("task_agent_outcome"),
        "strategy_learning": parsed.get("strategy_learning"),
        "run_root": parsed.get("run_root"), "stdout_ref": ref(root / "stdout.json"), "stderr_ref": ref(root / "stderr.txt")}
    save(root / "result.json", result)
    print(json.dumps({"exit_code": process.returncode, "status": parsed.get("status"),
        "termination": (parsed.get("task_agent_outcome") or {}).get("termination"), "result_ref": ref(root / "result.json")}))


if __name__ == "__main__":
    main()
