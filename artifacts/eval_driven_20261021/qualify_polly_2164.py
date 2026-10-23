"""Qualify the separated Polly task on an isolated net8 test project.

为什么要有它：模型运行前必须证明基线失败、参考修复通过且既有标签保持；测试补丁和参考实现不进入求解器输入。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

SOURCE = Path("C:/tmp/rfext-polly-2164")
TASK_ROOT = Path("C:/tmp/rfext-polly-2164-task")
ROOT = Path("C:/tmp/rfext-polly-2164-qualification")
DOTNET = Path("C:/tmp/rf-dotnet-8.0.302/sdk/dotnet.exe")
BASE = "5d81f99ef5f523032c17b6b59b32d575ec9dfb06"
NS = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def save(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    return ref(path)


def run(argv: list[str], *, cwd: Path, label: str, env: dict[str, str] | None = None,
        allowed: tuple[int, ...] = (0,)) -> dict[str, object]:
    output = ROOT / "commands" / label
    output.mkdir(parents=True, exist_ok=False)
    completed = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=900, creationflags=subprocess.CREATE_NO_WINDOW)
    stdout = output / "stdout.txt"
    stderr = output / "stderr.txt"
    stdout.write_bytes(completed.stdout)
    stderr.write_bytes(completed.stderr)
    result = {"argv": argv, "cwd": str(cwd.resolve()), "exit_code": completed.returncode,
        "stdout_ref": ref(stdout), "stderr_ref": ref(stderr)}
    save(output / "process.json", result)
    if completed.returncode not in allowed:
        raise RuntimeError(f"PROCESS_FAILED:{label}:{completed.returncode}")
    return result


def patch_paths(path: Path) -> list[str]:
    result = []
    for match in re.finditer(r"^\+\+\+ b/(.+)$", path.read_text(encoding="utf-8"), re.MULTILINE):
        value = match.group(1).strip()
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or "\\" in value:
            raise ValueError("PATCH_PATH_INVALID")
        result.append(value)
    if not result:
        raise ValueError("TEST_PATCH_PATH_REQUIRED")
    return result


def test_project(test_patch: Path) -> str:
    projects = set()
    for value in patch_paths(test_patch):
        parent = (SOURCE / value).parent
        while parent != SOURCE and parent.is_relative_to(SOURCE):
            matches = list(parent.glob("*.csproj"))
            if len(matches) == 1:
                projects.add(matches[0].relative_to(SOURCE).as_posix())
                break
            if len(matches) > 1:
                raise ValueError("AMBIGUOUS_TEST_PROJECT")
            parent = parent.parent
    if len(projects) != 1:
        raise ValueError("ONE_TEST_PROJECT_REQUIRED")
    return projects.pop()


def environment() -> dict[str, str]:
    value = dict(os.environ)
    value.update(DOTNET_ROOT=str(DOTNET.parent), DOTNET_CLI_HOME="C:/tmp/rf-dotnet-8.0.302/cli-home",
        DOTNET_MULTILEVEL_LOOKUP="0", DOTNET_NOLOGO="1", DOTNET_CLI_TELEMETRY_OPTOUT="1",
        NUGET_PACKAGES="C:/tmp/rfext-polly-packages", NUGET_HTTP_CACHE_PATH="C:/tmp/rfext-polly-http-cache",
        DOTNET_CLI_UI_LANGUAGE="en-US", LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8")
    return value


def trx_results(root: Path) -> dict[str, list[str]]:
    outcomes: dict[str, list[str]] = {}
    files = list(root.glob("*.trx"))
    if not files:
        raise ValueError("TRX_REQUIRED")
    for path in files:
        document = ET.fromstring(path.read_bytes())
        definitions = {row.get("id"): row.find("t:TestMethod", NS)
                       for row in document.findall(".//t:UnitTest", NS)}
        for row in document.findall(".//t:UnitTestResult", NS):
            method = definitions.get(row.get("testId"))
            if method is None:
                raise ValueError("TRX_DEFINITION_REQUIRED")
            name = method.get("className") + "." + method.get("name")
            outcomes.setdefault(name, []).append(row.get("outcome"))
    return outcomes


def label_outcomes(observed: dict[str, list[str]], labels: list[str]) -> dict[str, object]:
    missing = [label for label in labels if label not in observed]
    counts = {"Passed": 0, "Failed": 0, "Other": 0}
    for label in labels:
        for outcome in observed.get(label, []):
            counts[outcome if outcome in counts else "Other"] += 1
    return {"label_count": len(labels), "missing_count": len(missing), "outcome_counts": counts}


def prepare_variant(name: str, *, reference: bool) -> Path:
    target = ROOT / "variants" / name / "source"
    run(["git", "-C", str(SOURCE), "worktree", "add", "--detach", str(target), BASE],
        cwd=SOURCE, label=f"{name}-worktree")
    test_patch = TASK_ROOT / "evaluation-only/test.patch"
    run(["git", "apply", "--check", str(test_patch)], cwd=target, label=f"{name}-test-check")
    run(["git", "apply", str(test_patch)], cwd=target, label=f"{name}-test-apply")
    if reference:
        patch = TASK_ROOT / "evaluation-only/reference.patch"
        run(["git", "apply", "--check", str(patch)], cwd=target, label=f"{name}-reference-check")
        run(["git", "apply", str(patch)], cwd=target, label=f"{name}-reference-apply")
    return target


def test_variant(name: str, source: Path, project: str) -> dict[str, object]:
    results = ROOT / "variants" / name / "results"
    results.mkdir(parents=True)
    process = run([str(DOTNET), "test", str(source / project), "--framework", "net8.0",
        "--configuration", "Release", "-p:MinVerSkip=true", "--logger", f"trx;LogFileName={name}.trx",
        "--results-directory", str(results)], cwd=source, label=f"{name}-test", env=environment(),
        allowed=(0, 1))
    task = json.loads((TASK_ROOT / "evaluation-only/task.json").read_bytes())
    observed = trx_results(results)
    return {"process_ref": ref(ROOT / "commands" / f"{name}-test/process.json"),
        "exit_code": process["exit_code"], "fail_to_pass": label_outcomes(observed, task["FAIL_TO_PASS"]),
        "pass_to_pass": label_outcomes(observed, task["PASS_TO_PASS"]),
        "total_results": sum(len(value) for value in observed.values()),
        "trx_refs": [ref(path) for path in sorted(results.glob("*.trx"))]}


def main() -> int:
    opening = json.loads((TASK_ROOT / "task-opened.json").read_bytes())
    if (ROOT.exists() or opening.get("reference_exposed_to_solver") is not False
            or opening.get("hidden_tests_exposed_to_solver") is not False
            or not DOTNET.is_file()
            or subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip() != BASE):
        raise ValueError("QUALIFICATION_PRECONDITION_FAILED")
    ROOT.mkdir()
    project = test_project(TASK_ROOT / "evaluation-only/test.patch")
    baseline = test_variant("baseline", prepare_variant("baseline", reference=False), project)
    reference = test_variant("reference", prepare_variant("reference", reference=True), project)
    baseline_expected = (baseline["exit_code"] == 1
        and baseline["fail_to_pass"]["missing_count"] == 0
        and baseline["fail_to_pass"]["outcome_counts"]["Failed"] > 0
        and baseline["pass_to_pass"]["missing_count"] == 0
        and baseline["pass_to_pass"]["outcome_counts"]["Failed"] == 0)
    reference_expected = (reference["exit_code"] == 0
        and reference["fail_to_pass"]["missing_count"] == 0
        and reference["fail_to_pass"]["outcome_counts"]["Failed"] == 0
        and reference["pass_to_pass"]["missing_count"] == 0
        and reference["pass_to_pass"]["outcome_counts"]["Failed"] == 0)
    qualified = baseline_expected and reference_expected
    result = {"schema_version": "polly-2164-grader-qualification.v1",
        "status": "qualified" if qualified else "qualification_failed", "qualified": qualified,
        "base_commit": BASE, "test_project": project, "target_framework": "net8.0",
        "environment_differences": ["MinVerSkip=true avoids this host's cmd PATH resolution defect.",
            "Only the task patch's owning net8.0 test project is used; whole-solution net6/net481 parity is not claimed."],
        "baseline": baseline, "reference": reference, "public_problem_ref": opening["public_problem_ref"],
        "reference_exposed_to_solver": False, "hidden_tests_exposed_to_solver": False,
        "model_calls": 0, "resource_policy": "meter_only"}
    receipt = save(ROOT / "qualification.json", result)
    print(json.dumps({"status": result["status"], "qualified": qualified, "test_project": project,
        "baseline_exit": baseline["exit_code"], "reference_exit": reference["exit_code"],
        "baseline_fail_to_pass": baseline["fail_to_pass"], "reference_fail_to_pass": reference["fail_to_pass"],
        "qualification_ref": receipt, "model_calls": 0}))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
