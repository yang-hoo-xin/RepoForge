"""Run the public issue reproduction with original unit tests and retain logs.

为什么要有它：真实模型前先确认公开反馈实际执行，并保留环境失败与业务失败的区别。
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("C:/tmp/rfext-restsharp-2241-public-check-v2")
SOURCE = Path("C:/tmp/rfext-restsharp-2241-public")
DOTNET = Path("C:/tmp/rf-dotnet-8.0.302/sdk/dotnet.exe")


def ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, indent=2).encode())
    return ref(path)


def main():
    ROOT.mkdir()
    argv = [str(DOTNET), "test", "test/RestSharp.Tests/RestSharp.Tests.csproj", "--framework", "net8.0",
        "--configuration", "Release", "-p:MinVerSkip=true", "-p:MinVerMajor=0", "-p:MinVerMinor=0", "-p:MinVerPatch=0", "-p:CollectCoverage=false",
        "-m:1", "--logger", "trx;LogFileName=public.trx", "--results-directory", str(ROOT / "results")]
    env = {**os.environ, "DOTNET_ROOT": str(DOTNET.parent), "DOTNET_MULTILEVEL_LOOKUP": "0", "DOTNET_NOLOGO": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_CLI_HOME": "C:/tmp/rf-dotnet-8.0.302/cli-home",
        "NUGET_PACKAGES": "C:/tmp/rfext-restsharp-packages", "NUGET_HTTP_CACHE_PATH": "C:/tmp/rfext-restsharp-http-cache",
        "DOTNET_CLI_UI_LANGUAGE": "en-US"}
    started = time.monotonic()
    save(ROOT / "started.json", {"argv": argv, "source": str(SOURCE), "public_only": True, "model_calls": 0})
    try:
        process = subprocess.run(argv, cwd=SOURCE, env=env, capture_output=True, timeout=900,
            creationflags=subprocess.CREATE_NO_WINDOW)
        stdout, stderr, code, status = process.stdout, process.stderr, process.returncode, "process_completed"
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code, status = error.stdout or b"", error.stderr or b"", None, "timeout"
    out = save(ROOT / "stdout.txt", stdout)
    err = save(ROOT / "stderr.txt", stderr)
    result = save(ROOT / "result.json", {"argv": argv, "exit_code": code, "status": status,
        "elapsed_seconds": time.monotonic() - started, "stdout_ref": out, "stderr_ref": err,
        "trx_refs": [ref(path) for path in (ROOT / "results").glob("*.trx")], "model_calls": 0})
    print(json.dumps({"result_ref": result, "exit_code": code, "status": status}))


if __name__ == "__main__":
    main()
