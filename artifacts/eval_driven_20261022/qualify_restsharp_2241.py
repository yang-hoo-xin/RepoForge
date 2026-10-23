"""Independent evaluator qualification; never include this file in solver input.

为什么要有它：验收须先识别基线、正确、等价与错误实现，私有变体不得变成模型提示。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qualification_helpers", REPO / "artifacts/eval_driven_20261021/qualify_polly_2164.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.SOURCE = Path("C:/tmp/rfext-restsharp-2241")
h.TASK_ROOT = Path("C:/tmp/rfext-restsharp-2241-task")
h.ROOT = Path("C:/tmp/rfext-restsharp-2241-qualification-v3")
h.BASE = "dd52ff6cd18cce2a74475f97374d431280ac10cf"
PROJECTS = ["test/RestSharp.Tests/RestSharp.Tests.csproj", "test/RestSharp.Tests.Integrated/RestSharp.Tests.Integrated.csproj"]
PUBLIC_PATCH = REPO / "artifacts/eval_driven_20261022/restsharp-2241-public-reproduction.patch"


def environment():
    env = h.environment()
    env.update(NUGET_PACKAGES="C:/tmp/rfext-restsharp-packages", NUGET_HTTP_CACHE_PATH="C:/tmp/rfext-restsharp-http-cache")
    return env


def replace_exact(path, before, after):
    text = path.read_text(encoding="utf-8-sig")
    if text.count(before) != 1:
        raise ValueError("VARIANT_PREIMAGE_NOT_UNIQUE")
    path.write_text(text.replace(before, after), encoding="utf-8")


def prepare(name):
    source = h.prepare_variant(name, reference=name != "baseline")
    # 为什么要有它：仅约束本轮库/测试目标，不能覆盖 netstandard2.0 源码生成器。
    replace_exact(source / "test/Directory.Build.props", "<TargetFrameworks>net48;net6.0;net7.0;net8.0</TargetFrameworks>", "<TargetFrameworks>net8.0</TargetFrameworks>")
    replace_exact(source / "src/Directory.Build.props", "<TargetFrameworks>netstandard2.0;net471;net48;net6.0;net7.0;net8.0</TargetFrameworks>", "<TargetFrameworks>net8.0</TargetFrameworks>")
    h.run(["git", "apply", str(PUBLIC_PATCH)], cwd=source, label=f"{name}-public-apply")
    if name == "equivalent":
        replace_exact(source / "src/RestSharp/Parameters/HeaderParameter.cs",
            "Ensure.NotNull(value, nameof(value)),", "value ?? throw new ArgumentNullException(nameof(value)),")
    if name == "wrong":
        replace_exact(source / "src/RestSharp/Parameters/HeaderParameter.cs",
            "Ensure.NotNull(value, nameof(value)),", "value,")
    if name == "overreject":
        replace_exact(source / "src/RestSharp/Parameters/HeaderParameter.cs",
            "Ensure.NotNull(value, nameof(value)),", "Ensure.NotNull((string?)null, nameof(value)),")
    h.run(["git", "diff", "--binary"], cwd=source, label=f"{name}-diff")
    return source


def test(name, source, *, public_only=False):
    observed = {}
    processes = []
    trxs = []
    for index, project in enumerate(PROJECTS[:1] if public_only else PROJECTS):
        results = h.ROOT / "variants" / name / f"results-{index}"
        results.mkdir(parents=True)
        argv = [str(h.DOTNET), "test", str(source / project), "--framework", "net8.0", "--configuration", "Release",
            "-p:MinVerSkip=true", "-p:MinVerMajor=0", "-p:MinVerMinor=0", "-p:MinVerPatch=0", "-m:1", "--logger", f"trx;LogFileName={name}-{index}.trx",
            "--results-directory", str(results)]
        if public_only:
            argv += ["--filter", "FullyQualifiedName~RestSharp.Tests.PublicIssue2241Tests"]
        elif index == 1:
            argv += ["--filter", "FullyQualifiedName~RestSharp.Tests.Integrated.HttpHeadersTests"]
        label = f"{name}-test-{index}"
        process = h.run(argv, cwd=source, label=label, env=environment(), allowed=(0, 1))
        processes.append({"exit_code": process["exit_code"], "process_ref": h.ref(h.ROOT / "commands" / label / "process.json")})
        for key, value in h.trx_results(results).items():
            observed.setdefault(key, []).extend(value)
        trxs.extend(h.ref(path) for path in results.glob("*.trx"))
    task = json.loads((h.TASK_ROOT / "evaluation-only/task.json").read_bytes())
    public = {key: values for key, values in observed.items() if "PublicIssue2241Tests" in key}
    return {"processes": processes, "trx_refs": trxs, "total_results": sum(map(len, observed.values())),
        "fail_to_pass": h.label_outcomes(observed, task["FAIL_TO_PASS"]),
        "pass_to_pass": h.label_outcomes(observed, task["PASS_TO_PASS"]),
        "public_reproduction": {"count": sum(map(len, public.values())), "passed": sum(v == "Passed" for values in public.values() for v in values)},
        "all_outcomes": {status: sum(v == status for values in observed.values() for v in values) for status in ("Passed", "Failed", "NotExecuted")}}


def accepted(result):
    return all(result[key]["missing_count"] == 0 and result[key]["outcome_counts"]["Passed"] > 0
        and result[key]["outcome_counts"]["Failed"] == 0 and result[key]["outcome_counts"]["Other"] == 0
        for key in ("fail_to_pass", "pass_to_pass"))


def public_recheck():
    global PUBLIC_PATCH
    original = h.ROOT / "qualification.json"
    previous = json.loads(original.read_bytes())
    h.ROOT = Path("C:/tmp/rfext-restsharp-2241-public-qualification-v2")
    PUBLIC_PATCH = REPO / "artifacts/eval_driven_20261022/restsharp-2241-public-reproduction-v2.patch"
    h.ROOT.mkdir(exist_ok=False)
    h.save(h.ROOT / "inputs.json", {"script_ref": h.ref(Path(__file__)), "independent_qualification_ref": h.ref(original), "public_reproduction_ref": h.ref(PUBLIC_PATCH)})
    results = {}
    for name in ("baseline", "reference", "equivalent"):
        result = test(name, prepare(name), public_only=True)
        results[name] = result
        h.save(h.ROOT / f"{name}.json", result)
        print(json.dumps({"variant": name, "public": result["public_reproduction"]}), flush=True)
    # 为什么要有它：公开反馈版本改变后须用真实执行复验，不能沿用此前误拒的资格。
    public_ok = all(result["public_reproduction"]["count"] == 2 for result in results.values())
    public_ok = public_ok and results["baseline"]["public_reproduction"]["passed"] == 0
    public_ok = public_ok and all(results[name]["public_reproduction"]["passed"] == 2 and results[name]["processes"][0]["exit_code"] == 0 for name in ("reference", "equivalent"))
    baseline = previous["variants"]["baseline"]
    independent_ok = previous["qualified"] and baseline["fail_to_pass"]["missing_count"] == 0 and baseline["fail_to_pass"]["outcome_counts"]["Failed"] > 0
    independent_ok = independent_ok and baseline["pass_to_pass"]["missing_count"] == 0 and baseline["pass_to_pass"]["outcome_counts"] == {"Passed": 6, "Failed": 0, "Other": 0}
    qualified = bool(public_ok and independent_ok)
    receipt = h.save(h.ROOT / "qualification.json", {"schema_version": "restsharp-2241-public-and-independent-qualification.v1",
        "qualified": qualified, "public_reproduction_ref": h.ref(PUBLIC_PATCH), "independent_qualification_ref": h.ref(original),
        "public_results": results, "test_projects": PROJECTS, "target_framework": "net8.0", "model_calls": 0,
        "scope": "Public reproduction v2 and separated official acceptance qualified; v1 public false rejection retained; no solver success claimed."})
    print(json.dumps({"qualified": qualified, "qualification_ref": receipt}), flush=True)
    return 0 if qualified else 1


def main():
    if h.ROOT.exists() or subprocess.check_output(["git", "-C", str(h.SOURCE), "rev-parse", "HEAD"], text=True).strip() != h.BASE:
        raise ValueError("QUALIFICATION_PRECONDITION_FAILED")
    h.ROOT.mkdir()
    h.save(h.ROOT / "inputs.json", {"base": h.BASE, "script_ref": h.ref(Path(__file__)), "opening_ref": h.ref(h.TASK_ROOT / "task-opened.json"),
        "public_patch_ref": h.ref(PUBLIC_PATCH), "test_patch_ref": h.ref(h.TASK_ROOT / "evaluation-only/test.patch"),
        "reference_patch_ref": h.ref(h.TASK_ROOT / "evaluation-only/reference.patch"), "labels_ref": h.ref(h.TASK_ROOT / "evaluation-only/task.json")})
    results = {}
    for name in ("baseline", "reference", "equivalent", "wrong", "overreject"):
        result = test(name, prepare(name))
        results[name] = result
        h.save(h.ROOT / f"{name}.json", result)
        print(json.dumps({"variant": name, "accepted": accepted(result), "total_results": result["total_results"], "public_reproduction": result["public_reproduction"]}), flush=True)
    qualified = not accepted(results["baseline"]) and accepted(results["reference"]) and accepted(results["equivalent"]) and not accepted(results["wrong"]) and not accepted(results["overreject"])
    result = {"schema_version": "restsharp-2241-grader-qualification.v1", "qualified": qualified,
        "test_projects": PROJECTS, "target_framework": "net8.0", "variants": results,
        "scope": "Official FAIL_TO_PASS and PASS_TO_PASS labels; full unit project plus integrated HttpHeaders group; public reproduction separately reported.",
        "environment_differences": ["net8.0 only; generator retains netstandard2.0", "MinVerSkip=true with MinVerMajor/Minor/Patch=0", "reference patch includes its upstream dependency changes"],
        "reference_exposed_to_solver": False, "hidden_tests_exposed_to_solver": False, "model_calls": 0}
    reference = h.save(h.ROOT / "qualification.json", result)
    print(json.dumps({"qualified": qualified, "qualification_ref": reference}), flush=True)
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(public_recheck() if "--public-recheck" in sys.argv else main())
