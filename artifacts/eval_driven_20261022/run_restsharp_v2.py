"""Use frozen normal v2 entry on the selected public RestSharp issue.

为什么要有它：模型独立选择调查和修改；运行包装器只固定输入、环境、版本及原件。
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("C:/tmp/rfext-restsharp-2241")
TASK = Path("C:/tmp/rfext-restsharp-2241-task")
DOTNET = Path("C:/tmp/rf-dotnet-8.0.302/sdk/dotnet.exe")
MODEL = "qwen3-coder:30b"
BASE = "dd52ff6cd18cce2a74475f97374d431280ac10cf"


def repository_profile(allowed, dotnet_ref):
    return {"profile_version": "csharp_repository_profile_v1", "dotnet_ref": dotnet_ref,
        "project_path": "src/RestSharp/RestSharp.csproj", "test_project_path": "test/RestSharp.Tests/RestSharp.Tests.csproj",
        "nuget_config_path": "NuGet.config", "configuration": "Release", "allowed_edit_paths": allowed,
        "build_properties": {"MinVerSkip": "true", "MinVerMajor": "0", "MinVerMinor": "0", "MinVerPatch": "0"},
        "expected_artifacts": ["src/RestSharp/bin/Release/net8.0/RestSharp.dll"],
        "test_command": ["test", "{test_project}", "--framework", "net8.0", "--configuration", "{configuration}",
            "-p:MinVerSkip=true", "-p:MinVerMajor=0", "-p:MinVerMinor=0", "-p:MinVerPatch=0", "-p:CollectCoverage=false", "-m:1",
            "--logger", "trx;LogFileName={report_path}"],
        "test_report": {"format": "trx", "path": "public-net8.trx"}, "timeout_seconds": 900}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-snapshot", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    original = REPO / "artifacts/eval_driven_20261022/run_polly_v2.py"
    spec = importlib.util.spec_from_file_location("existing_frozen_run_helpers", original)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    ref, save = helpers.ref, helpers.save
    root, snapshot = args.root.resolve(), args.code_snapshot.resolve()
    qualification = json.loads(args.qualification.read_bytes())
    public_patch = REPO / "artifacts/eval_driven_20261022/restsharp-2241-public-reproduction-v2.patch"
    environment_patch = REPO / "artifacts/eval_driven_20261022/restsharp-2241-public-environment.patch"
    if root.exists() or not qualification.get("qualified"):
        raise ValueError("RUN_PRECONDITION_FAILED")
    public_ref = qualification.get("public_reproduction_ref", {})
    if (Path(public_ref.get("path", "")).resolve() != public_patch.resolve()
            or public_ref.get("sha256") != ref(public_patch)["sha256"]
            or "bytes" in public_ref and public_ref["bytes"] != public_patch.stat().st_size):
        raise ValueError("PUBLIC_FEEDBACK_NOT_QUALIFIED")
    if snapshot.exists():
        code_ref = ref(snapshot / "snapshot.json")
        frozen = json.loads((snapshot / "snapshot.json").read_bytes())
        if not frozen.get("files") or any(ref(snapshot / row["path"])["sha256"] != row["sha256"] for row in frozen["files"]):
            raise ValueError("FROZEN_CODE_CHANGED")
    else:
        code_ref = helpers.freeze_code(snapshot)
    root.mkdir()
    source = root / "input/RestSharp"
    subprocess.run(["git", "-C", str(SOURCE), "worktree", "add", "--detach", str(source), BASE],
        check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    for patch in (public_patch, environment_patch):
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True, capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
    allowed = sorted(p.relative_to(source).as_posix() for p in (source / "src/RestSharp").rglob("*.cs"))
    settings = yaml.safe_load((snapshot / "config/settings.yaml").read_text(encoding="utf-8"))
    settings["task_memory"] = {"enabled": False, "store_root": None, "max_context_chars": 3000}
    settings["llm"].update(provider="ollama", model=MODEL, base_url="http://127.0.0.1:11434", max_tokens=4096)
    settings_path = root / "settings.yaml"
    with settings_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(settings, stream, allow_unicode=True, sort_keys=False)
    profile = repository_profile(allowed, ref(DOTNET))
    profile_ref = save(root / "repository-profile.json", profile)
    experience = save(root / "experience-config.json", {"schema_version": "task_agent_experience_config_v1",
        "store_root": str(root / "strategy-memory"), "snapshot_ref": None, "extract_on_terminal": True})
    arguments = ["--target-project", str(source), "--workspace-root", str(root / "runs"), "--settings", str(settings_path),
        "--repository-profile", profile_ref["path"], "--local-ollama", "--ollama-model", MODEL,
        "--ollama-base-url", "http://127.0.0.1:11434", "--ollama-context-window", "65536",
        "--ollama-max-output-tokens", "4096", "--ollama-timeout-seconds", "600", "--model-call-budget-policy", "meter_only",
        "--source-retrieval", "rg_live", "--confirm-isolated-write", "--allow-build", "--defer-approval",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2", "--constraint-policy", "source_priority_v1",
        "--task-agent-edit-paths", json.dumps(allowed), "--task-agent-experience-config", experience["path"]]
    execution = root / "execution.json"
    save(execution, {"arguments": arguments})
    sys.path.insert(0, str(snapshot))
    from scripts.dsh_run_request import prepare_new_task
    problem = TASK / "solver-input/problem_statement.txt"
    description = problem.read_text(encoding="utf-8") + "\nFix the reported compatibility regression while preserving applicable existing behavior."
    allocated = prepare_new_task(execution_config=str(execution), description=description)
    started = save(root / "started.json", {"code_snapshot_ref": code_ref, "runner_ref": ref(Path(__file__)),
        "helper_ref": ref(original), "qualification_ref": ref(args.qualification), "public_problem_ref": ref(problem),
        "public_reproduction_ref": ref(public_patch), "environment_patch_ref": ref(environment_patch),
        "allocated_config": allocated, "model": MODEL, "experience_injection": False, "terminal_learning_enabled": True,
        "hidden_material_supplied": False, "scripted_actions": False, "outer_supplied_solution": False})
    env = {**os.environ, "PYTHONPATH": str(snapshot), "MinVerSkip": "true", "DOTNET_CLI_UI_LANGUAGE": "en-US"}
    with (root / "stdout.json").open("xb") as stdout, (root / "stderr.txt").open("xb") as stderr:
        try:
            process = subprocess.run([sys.executable, "-B", str(snapshot / "scripts/dsh_run_request.py"),
                "--execution-config", allocated], cwd=snapshot, env=env, input=description.encode(),
                stdout=stdout, stderr=stderr, timeout=1800, creationflags=subprocess.CREATE_NO_WINDOW)
            code, status = process.returncode, "process_completed"
        except subprocess.TimeoutExpired:
            code, status = None, "runner_timeout"
    try:
        parsed = json.loads((root / "stdout.json").read_bytes())
    except ValueError:
        parsed = {}
    result = save(root / "result.json", {"status": status, "exit_code": code, "started_ref": started,
        "product_status": parsed.get("status"), "task_outcome": parsed.get("task_agent_outcome"),
        "strategy_learning": parsed.get("strategy_learning"), "run_root": parsed.get("run_root"),
        "stdout_ref": ref(root / "stdout.json"), "stderr_ref": ref(root / "stderr.txt")})
    print(json.dumps({"result_ref": result, "status": status, "product_status": parsed.get("status")}))


if __name__ == "__main__":
    main()
