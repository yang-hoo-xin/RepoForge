"""New exposed-task diagnostic from saved model code and verified public feedback.

为什么要有它：检查器修复后验证真实修订能力，原unknown尝试、资源和独立评分仍保持原状。
"""
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.prepare_reload_trials import ref, save  # noqa: E402
from scripts.run_benchmark_arm import WindowsJob, execute  # noqa: E402
from src.mod_agent.domain.csharp_repository_backend import _IGNORED, _snapshot  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402
from src.mod_agent.task_agent_v2_state import TaskAgentV2State  # noqa: E402

ROOT = Path("C:/tmp/rfext-restsharp-v2-revision-1")
SNAPSHOT = Path("C:/tmp/rfcode-restsharp-v2-revision-1")
PREVIOUS = Path("C:/tmp/rfext-restsharp-v2-b1")
REPLAY = {"path": "C:/tmp/rfext-restsharp-v2-b1-check-replay/result.json",
    "sha256": "7494107b9cffb62b0c41f1b710d861dec12cd9c7a1acfe3aac66575c88049dd6"}
OLLAMA = Path("C:/Users/10160/AppData/Local/Programs/Ollama/ollama.exe")


def get(endpoint):
    with urlopen("http://127.0.0.1:11434/api/" + endpoint, timeout=2) as response:
        return json.load(response)


def main():
    replay = read_bound_json(REPLAY)
    report = read_bound_json(replay["report_ref"])
    native = read_bound_json(replay["native_execution_ref"])
    native_path = Path(replay["native_execution_ref"]["path"])
    stores = [parent for parent in native_path.parents if (parent / native["task_contract_ref"]).is_file()]
    if len(stores) != 1 or report["status"] != "behavior_failed" or not report["verification_summary"]["test_command_executed"]:
        raise ValueError("VERIFIED_PUBLIC_FAILURE_REQUIRED")
    state = TaskAgentV2State(stores[0], native["task_contract_ref"])
    candidate = state.validate_candidate(replay["candidate_ref"])
    if candidate["candidate_source_id"] != replay["candidate_source_id"]:
        raise ValueError("ORIGINAL_CANDIDATE_CHANGED")
    old_head = ref(state.head_path)
    source = Path(report["source_root"])
    before = _snapshot(source)
    if before != report["source_inputs"] or any((source / path).read_bytes() != text.encode()
            for path, text in state.materialize(replay["candidate_ref"]).items()):
        raise ValueError("PUBLIC_FEEDBACK_SOURCE_CHANGED")
    with socket.socket() as probe:
        existing_service = probe.connect_ex(("127.0.0.1", 11434)) == 0
    helper_path = REPO / "artifacts/eval_driven_20261022/run_polly_v2.py"
    spec = importlib.util.spec_from_file_location("freeze_helpers", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    snapshot_ref = helper.freeze_code(SNAPSHOT)
    ROOT.mkdir()
    target = ROOT / "input/RestSharp"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*_IGNORED))
    if _snapshot(target) != before:
        raise ValueError("DIAGNOSTIC_SOURCE_COPY_CHANGED")
    for name in ("settings.yaml", "repository-profile.json"):
        shutil.copyfile(PREVIOUS / name, ROOT / name)
    experience = save(ROOT / "experience-config.json", {"schema_version": "task_agent_experience_config_v1",
        "store_root": str(ROOT / "strategy-memory"), "snapshot_ref": None, "extract_on_terminal": True})
    arguments = json.loads((PREVIOUS / "execution.json").read_bytes())["arguments"]
    replacements = {"--target-project": str(target), "--workspace-root": str(ROOT / "runs"),
        "--settings": str(ROOT / "settings.yaml"), "--repository-profile": str(ROOT / "repository-profile.json"),
        "--task-agent-experience-config": experience["path"]}
    for flag, value in replacements.items():
        arguments[arguments.index(flag) + 1] = value
    config = ROOT / "execution.json"
    save(config, {"arguments": arguments})
    problem = Path("C:/tmp/rfext-restsharp-2241-task/solver-input/problem_statement.txt")
    request = problem.read_text(encoding="utf-8") + "\nFix the reported regression while preserving applicable existing behavior."
    request += "\nThe starting source is a saved model-produced candidate, not the original repository baseline. "
    request += "A real public check of these exact starting bytes failed. Investigate the current code and use the observations to decide your changes.\n"
    request += json.dumps({"public_check": {"summary": report["verification_summary"], "observations": report["observations"]}}, ensure_ascii=False)
    request_path = ROOT / "request.txt"
    request_path.write_text(request, encoding="utf-8")
    from scripts.dsh_run_request import prepare_new_task
    allocated = prepare_new_task(execution_config=str(config), description=request)
    save(ROOT / "started.json", {"code_snapshot_ref": snapshot_ref, "runner_ref": ref(Path(__file__)),
        "helper_ref": ref(helper_path), "original_native_ref": replay["native_execution_ref"], "original_head_ref": old_head,
        "starting_candidate_ref": replay["candidate_ref"], "starting_candidate_source_id": candidate["candidate_source_id"],
        "public_feedback_ref": replay["report_ref"], "request_ref": ref(request_path), "allocated_config": allocated,
        "new_task": True, "original_task_resumed": False, "exposure": "exposed_development_revision_diagnostic",
        "prior_model_calls": 14, "prior_known_tokens": 217646, "experience_injection": False,
        "terminal_learning_enabled": True, "hidden_feedback_supplied": False,
        "developer_assistance": "Prepared saved candidate and its actual public feedback; no technical fix or action order supplied.",
        "reuse_existing_service": existing_service,
        "preparation_events": ["Earlier pre-allocation invocation stopped on occupied port; zero model calls."],
        "wall_limit_seconds": 1800})
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MODELS": "C:/Users/10160/.ollama/models",
        "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": "-1",
        "PYTHONPATH": str(SNAPSHOT), "DOTNET_CLI_UI_LANGUAGE": "en-US"}
    job, service = WindowsJob(), None
    try:
        with (ROOT / "service.stdout.bin").open("xb") as stdout, (ROOT / "service.stderr.bin").open("xb") as stderr:
            if not existing_service:
                service = subprocess.Popen([str(OLLAMA), "serve"], env=env, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004)
                job.start(service)
            deadline = time.monotonic() + 30
            while True:
                if service is not None and service.poll() is not None:
                    raise RuntimeError("OWNED_SERVICE_EXITED")
                try:
                    version = get("version")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("SERVICE_START_TIMEOUT")
                    time.sleep(0.25)
            owner = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                "Get-NetTCPConnection -LocalPort 11434 -State Listen | Select-Object -ExpandProperty OwningProcess | ConvertTo-Json -Compress"],
                capture_output=True, check=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            pids = json.loads(owner.stdout.decode("utf-8-sig"))
            if service is not None and set(pids if isinstance(pids, list) else [pids]) != {service.pid}:
                raise ValueError("SERVICE_OWNER_CHANGED")
            model = next(row for row in get("tags")["models"] if row["name"] == "qwen3-coder:30b")
            if model["digest"] != "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca":
                raise ValueError("MODEL_CHANGED")
            save(ROOT / "ready.json", {"service_pid": service.pid if service else None, "listener_pids": pids,
                "service_owned": service is not None, "version": version, "model": model, "warmup_calls": 0,
                "cold_start_verified": False, "external_service_configuration": "unknown" if existing_service else None})
            process = execute([sys.executable, "-B", str(SNAPSHOT / "scripts/dsh_run_request.py"), "--execution-config", allocated],
                cwd=SNAPSHOT, stdin_path=request_path, output=ROOT / "process", deadline=time.monotonic() + 1800, environment=env)
            if ref(state.head_path) != old_head or _snapshot(source) != before:
                raise ValueError("ORIGINAL_EVIDENCE_CHANGED")
            result = save(ROOT / "result.json", {"process": process, "new_task": True, "old_head_unchanged": True,
                "starting_source_unchanged": True, "prior_known_model_calls": 14, "prior_known_tokens": 217646})
            print(json.dumps({"result_ref": result, "process_status": process["status"]}))
    finally:
        job.close()
        if service is not None:
            service.wait(timeout=15)


if __name__ == "__main__":
    main()
