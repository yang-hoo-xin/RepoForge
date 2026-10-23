"""Supply the missing local service to the exact pre-model task allocation.

为什么要有它：模型准入前的环境失败继续原任务；自有服务与进程树需有确定清理边界。
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.prepare_reload_trials import ref, save
from scripts.run_benchmark_arm import WindowsJob, execute

ROOT = Path("C:/tmp/rfext-restsharp-v2-b1")
OUTPUT = ROOT / "service-resume-1"
SNAPSHOT = Path("C:/tmp/rfcode-restsharp-v2-b1")
EXECUTABLE = Path("C:/Users/10160/AppData/Local/Programs/Ollama/ollama.exe")


def get(endpoint):
    with urlopen("http://127.0.0.1:11434/api/" + endpoint, timeout=2) as response:
        return json.load(response)


def main():
    prior = json.loads((ROOT / "stdout.json").read_bytes())
    started = json.loads((ROOT / "started.json").read_bytes())
    config = Path(started["allocated_config"])
    arguments = json.loads(config.read_bytes())["arguments"]
    run_id = arguments[arguments.index("--run-id") + 1]
    if (prior.get("reason") != "OLLAMA_MODEL_CATALOG_UNAVAILABLE:ConnectError"
            or (ROOT / "runs" / run_id).exists() or OUTPUT.exists()):
        raise ValueError("EXACT_PRE_MODEL_ALLOCATION_REQUIRED")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 11434)) == 0:
            raise ValueError("SERVICE_PORT_ALREADY_OWNED")
    OUTPUT.mkdir()
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MODELS": "C:/Users/10160/.ollama/models",
        "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KEEP_ALIVE": "-1",
        "PYTHONPATH": str(SNAPSHOT), "MinVerSkip": "true", "DOTNET_CLI_UI_LANGUAGE": "en-US"}
    request = Path("C:/tmp/rfext-restsharp-2241-task/solver-input/problem_statement.txt").read_text(encoding="utf-8")
    request += "\nFix the reported compatibility regression while preserving applicable existing behavior."
    with (OUTPUT / "request.txt").open("x", encoding="utf-8") as stream:
        stream.write(request)
    save(OUTPUT / "start.json", {"original_started_ref": ref(ROOT / "started.json"), "config_ref": ref(config),
        "service_executable_ref": ref(EXECUTABLE), "run_id": run_id, "new_task_allocated": False,
        "model_calls_before": 0, "wall_limit_seconds": 1800})
    job, service = WindowsJob(), None
    try:
        with (OUTPUT / "service.stdout.bin").open("xb") as stdout, (OUTPUT / "service.stderr.bin").open("xb") as stderr:
            service = subprocess.Popen([str(EXECUTABLE), "serve"], env=env, stdout=stdout, stderr=stderr,
                stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004)
            job.start(service)
            deadline = time.monotonic() + 30
            while True:
                if service.poll() is not None:
                    raise RuntimeError("OWNED_SERVICE_EXITED")
                try:
                    version = get("version")
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("OWNED_SERVICE_START_TIMEOUT")
                    time.sleep(0.25)
            owner = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                "Get-NetTCPConnection -LocalPort 11434 -State Listen | Select-Object -ExpandProperty OwningProcess | ConvertTo-Json -Compress"],
                capture_output=True, check=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            pids = json.loads(owner.stdout.decode("utf-8-sig"))
            if set(pids if isinstance(pids, list) else [pids]) != {service.pid}:
                raise ValueError("SERVICE_LISTENER_OWNER_CHANGED")
            model = next(row for row in get("tags")["models"] if row["name"] == "qwen3-coder:30b")
            if model["digest"] != "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca":
                raise ValueError("SELECTED_MODEL_CHANGED")
            save(OUTPUT / "ready.json", {"service_pid": service.pid, "version": version, "model": model,
                "warmup_model_calls": 0, "model_downloads": 0})
            result = execute([sys.executable, "-B", str(SNAPSHOT / "scripts/dsh_run_request.py"),
                "--execution-config", str(config)], cwd=SNAPSHOT, stdin_path=OUTPUT / "request.txt",
                output=OUTPUT / "process", deadline=time.monotonic() + 1800, environment=env)
            save(OUTPUT / "result.json", {"process": result, "original_started_ref": ref(ROOT / "started.json"),
                "new_task_allocated": False})
            print(json.dumps({"result_ref": ref(OUTPUT / "result.json"), "process_status": result["status"]}))
    finally:
        job.close()
        if service is not None:
            service.wait(timeout=15)


if __name__ == "__main__":
    main()
