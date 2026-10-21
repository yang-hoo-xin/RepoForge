"""Run the deterministic RepoForge v0.1 normal-entry demonstration.

The demonstration uses a scripted model and scripted approval while exercising
the real public CLI, .NET build, failing and passing behavior tests, isolated
Apply, Build continuation, and final readback. It is platform evidence, not an
autonomous-model result.

为什么要有它：发布演示需要从干净源码复现完整产品入口，同时明确隔离脚本模型与真实构建测试所能支持的结论。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "repoforge-v0.1-demo.v1"
TEST_PATH = "tests/unit/test_csharp_repository_product_entry.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_ref(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run(*argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _resolve_dotnet(explicit: str | None) -> Path:
    selected = explicit or os.environ.get("REPOFORGE_DOTNET") or shutil.which("dotnet")
    if not selected:
        raise ValueError("REPOFORGE_DOTNET_REQUIRED")
    path = Path(selected).resolve()
    if not path.is_file():
        raise ValueError("REPOFORGE_DOTNET_NOT_FOUND")
    return path


def _git_state() -> dict[str, object]:
    revision = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    if revision.returncode or status.returncode:
        raise ValueError("RELEASE_DEMO_GIT_STATE_UNAVAILABLE")
    return {
        "commit_sha": revision.stdout.strip(),
        "clean": not bool(status.stdout.strip()),
        "porcelain": status.stdout.splitlines(),
    }


def _junit_summary(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        name: sum(int(suite.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def run_demo(
    *,
    output: Path,
    dotnet: Path,
    require_clean: bool,
    workspace_parent: Path | None = None,
) -> dict[str, object]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    git_state = _git_state()
    dotnet_version = _run(str(dotnet), "--version")
    if dotnet_version.returncode:
        raise ValueError("REPOFORGE_DOTNET_VERSION_FAILED")
    if require_clean and not git_state["clean"]:
        raise ValueError("RELEASE_DEMO_REQUIRES_CLEAN_GIT_STATE")

    junit = output / "junit.xml"
    command = (
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "--basetemp",
        str(output / "pytest-temp"),
        "--junitxml",
        str(junit),
        TEST_PATH,
    )
    started = datetime.now(timezone.utc).isoformat()
    selected_parent = workspace_parent.resolve() if workspace_parent is not None else None
    if selected_parent is not None and not selected_parent.is_dir():
        raise ValueError("RELEASE_DEMO_WORKSPACE_PARENT_NOT_FOUND")
    with tempfile.TemporaryDirectory(prefix="rfv01-", dir=selected_parent) as temporary_workspace:
        workspaces = Path(temporary_workspace).resolve()
        environment = os.environ.copy()
        environment.update(
            {
                "REPOFORGE_REAL_REPOSITORY_CLI_INTEGRATION": "1",
                "REPOFORGE_DOTNET": str(dotnet),
                "TMP": str(workspaces),
                "TEMP": str(workspaces),
                "TMPDIR": str(workspaces),
            }
        )
        completed = _run(*command, env=environment)
        integrations = sorted(workspaces.rglob("integration-result.json"))
        retained = output / "integration-results"
        retained.mkdir()
        integration_refs = []
        for index, path in enumerate(integrations, start=1):
            value = json.loads(path.read_text(encoding="utf-8"))
            executor = value.get("technical_executor")
            label = executor if executor in {"task_agent", "workflow"} else str(index)
            destination = retained / f"{label}.json"
            shutil.copyfile(path, destination)
            integration_refs.append(_file_ref(destination))
    finished = datetime.now(timezone.utc).isoformat()
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    junit_result = _junit_summary(junit)
    passed = (
        completed.returncode == 0
        and junit_result == {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}
        and len(integration_refs) == 2
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "claim_scope": (
            "Real normal CLI, local .NET build/tests, scripted model and scripted approval; "
            "not autonomous-model evidence."
        ),
        "started_utc": started,
        "finished_utc": finished,
        "source_state": git_state,
        "dotnet": {"path": str(dotnet), "version": dotnet_version.stdout.strip()},
        "command": list(command),
        "pytest_exit_code": completed.returncode,
        "junit": junit_result,
        "integration_results": integration_refs,
        "stdout_ref": _file_ref(stdout_path),
        "stderr_ref": _file_ref(stderr_path),
        "junit_ref": _file_ref(junit) if junit.is_file() else None,
    }
    _write_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dotnet")
    parser.add_argument(
        "--workspace-parent",
        type=Path,
        help="Optional existing short-path parent for temporary build workspaces.",
    )
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_demo(
            output=args.output,
            dotnet=_resolve_dotnet(args.dotnet),
            require_clean=args.require_clean,
            workspace_parent=args.workspace_parent,
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
