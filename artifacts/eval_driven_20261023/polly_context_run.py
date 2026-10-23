"""Replay an existing public candidate, then freeze one normal v2 task.

The controller supplies no model actions or solution. It reuses product owners.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.dsh_run_request import (  # noqa: E402
    _configuration_arguments,
    build_product_argv,
    prepare_new_task,
)
from scripts.run_benchmark_arm import execute  # noqa: E402
from src.mod_agent.domain.csharp_repository_backend import (  # noqa: E402
    _IGNORED,
    CSharpRepositoryBackend,
    CSharpRepositoryProfile,
)
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.evaluation.strategy_trial_preparation import _source_snapshot  # noqa: E402
from src.mod_agent.model_admission import read_ollama_model_identity  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402
from src.mod_agent.platform_execution_snapshot import (  # noqa: E402
    FrozenExecution,
    prepare_snapshot,
)
from src.mod_agent.public_failure_feedback import validated_public_checks  # noqa: E402
from src.mod_agent.task_agent_v2_state import TaskAgentV2State  # noqa: E402

ROOT = Path("C:/tmp/rf-polly-v2-context-main")
OLD = Path("C:/tmp/rfext-polly-v2-b3")
OLD_RUN = OLD / "runs/dsh_379b036c2fdd40c8a2c7c7fe18843ab6"
SNAPSHOT = Path("C:/tmp/rfcode-polly-v2-context-B")
CACHE = Path("C:/tmp/rfext-polly-packages")
MODEL = "qwen3-coder:30b"
DIGEST = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"
ENV = {"RestoreSources": str(CACHE), "RestoreFallbackFolders": str(CACHE), "NuGetAudit": "false",
    "MinVerSkip": "true", "DOTNET_CLI_UI_LANGUAGE": "en-US"}


def prepare():
    if ROOT.exists() or SNAPSHOT.exists():
        raise ValueError("ALREADY_PREPARED_INSPECT_ORIGINAL")
    if not CACHE.is_dir():
        raise ValueError("LOCAL_PACKAGE_CACHE_REQUIRED")
    original = read_bound_json(_ref(OLD_RUN / "a/task-agent-execution/execution.json"))
    state = TaskAgentV2State(OLD_RUN / "a", original["task_contract_ref"])
    baseline = state.load(state.contract["baseline_manifest_ref"])
    source = Path(baseline["source_root"])
    checkout = OLD / "input/Polly"
    if subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() != "5d81f99ef5f523032c17b6b59b32d575ec9dfb06":
        raise ValueError("SOURCE_VERSION_CHANGED")
    for path, row in baseline["files"].items():
        if (source / path).read_bytes().decode("utf-8") != row["text"]:
            raise ValueError("PUBLIC_BASELINE_CHANGED")
    qualification = Path("C:/tmp/rfext-polly-2164-public-qualification/qualification.json")
    qualified = read_bound_json(_ref(qualification))
    public_patch = REPO / "artifacts/eval_driven_20261021/polly-2164-public-reproduction.patch"
    if not qualified["qualified"] or _ref(public_patch)["sha256"] != qualified["public_reproduction_ref"]["sha256"]:
        raise ValueError("PUBLIC_CHECKER_NOT_QUALIFIED")
    ROOT.mkdir()
    shutil.copytree(source, ROOT / "public-source", ignore=shutil.ignore_patterns(*_IGNORED))
    shutil.copyfile(OLD / "repository-profile.json", ROOT / "repository-profile.json")
    shutil.copyfile(OLD / "settings.yaml", ROOT / "settings.yaml")
    problem = Path("C:/tmp/rfext-polly-2164-task/solver-input/problem_statement.txt")
    profile = CSharpRepositoryProfile(_ref(ROOT / "repository-profile.json"))
    public = {"schema": "public-polly-context-input.v1", "source_root": str(ROOT / "public-source"),
        "source_snapshot": _source_snapshot(ROOT / "public-source", include_binary=True),
        "profile_ref": profile.reference, "fixed_inputs": profile.bind(ROOT / "public-source"),
        "settings_ref": _ref(ROOT / "settings.yaml"), "problem_ref": _ref(problem), "qualification_ref": _ref(qualification),
        "old_native_ref": _ref(OLD_RUN / "a/task-agent-execution/execution.json"),
        "old_candidate_ref": original["candidate_ref"], "old_task_contract_ref": original["task_contract_ref"],
        "old_store_root": str(OLD_RUN / "a"), "intent_ref": _ref(REPO / "artifacts/eval_driven_20261023/polly-context-intent.json"),
        "exposure": "exposed development task; not held-out or official resource-constrained benchmark result"}
    print(json.dumps(_save(ROOT / "public-input.json", public)))


def preflight(*, use_baseline=False):
    public = read_bound_json(_ref(ROOT / "public-input.json"))
    state = TaskAgentV2State(public["old_store_root"], public["old_task_contract_ref"])
    candidate = state.materialize("baseline" if use_baseline else public["old_candidate_ref"])
    label = "baseline-preflight" if use_baseline else "preflight"
    target = ROOT / (label + "-source")
    if target.exists() or (ROOT / (label + "-result.json")).exists():
        raise ValueError("PREFLIGHT_ALREADY_STARTED_INSPECT_ORIGINAL")
    shutil.copytree(public["source_root"], target, ignore=shutil.ignore_patterns(*_IGNORED))
    for path, text in candidate.items():
        (target / path).write_bytes(text.encode("utf-8"))
    os.environ.update(ENV)
    backend = CSharpRepositoryBackend(profile_ref=public["profile_ref"], source_root=target, workspace_root=ROOT,
        output_root=ROOT / label, expected_fixed_inputs=public["fixed_inputs"])
    result = backend.run_build_test(target, "public-baseline" if use_baseline else "saved-public-candidate", "development_preflight")
    report, checks = validated_public_checks(result["report_ref"])
    sources = set()
    for path in target.rglob("project.assets.json"):
        sources.update(json.loads(path.read_bytes()).get("project", {}).get("restore", {}).get("sources", {}))
    local = bool(sources) and all(Path(path).resolve() == CACHE.resolve() for path in sources)
    receipt = _save(ROOT / (label + "-result.json"), {"report_ref": result["report_ref"], "status": report["status"],
        "summary": report["verification_summary"], "failed_public_checks": [row["name"] for row in checks if not row["passed"]],
        "local_feed_only": local, "restore_sources": sorted(sources), "model_calls": 0,
        "basis": "original public baseline, zero model" if use_baseline else "old saved candidate, not baseline or new model result"})
    print(json.dumps(receipt))
    if not local or report["status"] != "behavior_failed":
        raise ValueError("PUBLIC_PREFLIGHT_NOT_READY")


def freeze():
    public = read_bound_json(_ref(ROOT / "public-input.json"))
    preflight = read_bound_json(_ref(ROOT / "baseline-preflight-result.json"))
    if preflight["status"] != "behavior_failed" or not preflight["local_feed_only"]:
        raise ValueError("PREFLIGHT_REQUIRED")
    if _source_snapshot(public["source_root"], include_binary=True) != public["source_snapshot"]:
        raise ValueError("PUBLIC_INPUT_CHANGED")
    model = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    if model["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    config = read_bound_json(_ref(OLD / "execution.json"))
    args = config["arguments"]
    for option, value in (("--target-project", public["source_root"]), ("--workspace-root", str(ROOT / "runs")),
            ("--settings", public["settings_ref"]["path"]), ("--repository-profile", public["profile_ref"]["path"])):
        args[args.index(option) + 1] = value
    experience = _save(ROOT / "experience.json", {"schema_version": "task_agent_experience_config_v1",
        "store_root": str(ROOT / "strategy-memory"), "snapshot_ref": None, "extract_on_terminal": True})
    if "--task-agent-experience-config" in args:
        args[args.index("--task-agent-experience-config") + 1] = experience["path"]
    else:
        args.extend(["--task-agent-experience-config", experience["path"]])
    base = _save(ROOT / "base-execution.json", {"arguments": args})
    description = Path(public["problem_ref"]["path"]).read_text(encoding="utf-8")
    selected = prepare_new_task(execution_config=base["path"], description=description)
    argv = build_product_argv(execution_config=selected, description=description)
    _, options = _configuration_arguments(selected)
    run_id = options["--run-id"]
    refs = [_ref(selected), public["profile_ref"], public["settings_ref"], public["problem_ref"], experience]
    platform = prepare_snapshot(source_root=REPO, snapshot_root=SNAPSHOT, business_argv=argv,
        run_root=ROOT / "runs" / run_id, bound_inputs=[(ref["path"], ref["sha256"]) for ref in refs])
    plan = _save(ROOT / "frozen-plan.json", {"schema": "polly-context-diagnostic.v1", "run_id": run_id,
        "run_root": str(ROOT / "runs" / run_id), "execution_config_ref": _ref(selected), "platform_ref": platform,
        "controller_ref": _ref(__file__), "public_input_ref": _ref(ROOT / "public-input.json"),
        "preflight_ref": _ref(ROOT / "baseline-preflight-result.json"), "intent_ref": public["intent_ref"],
        "compile_replay_ref": _ref(REPO / "artifacts/eval_driven_20261023/polly-saved-candidate-preflight.json"),
        "model_identity": model, "environment": ENV, "wall_seconds": 1800,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "planned_attempts": 1, "model_calls_during_freeze": 0, "independent_from_old_attempts": True})
    print(json.dumps(plan))


def run():
    plan_ref = _ref(ROOT / "frozen-plan.json")
    plan = read_bound_json(plan_ref)
    if _ref(__file__) != plan["controller_ref"] or (ROOT / "started.json").exists() or Path(plan["run_root"]).exists():
        raise ValueError("INPUT_CHANGED_OR_ALREADY_STARTED")
    public = read_bound_json(plan["public_input_ref"])
    if _source_snapshot(public["source_root"], include_binary=True) != public["source_snapshot"]:
        raise ValueError("PUBLIC_SOURCE_CHANGED")
    FrozenExecution(plan["platform_ref"]).verify_files()
    model = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    if model["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    clock = time.monotonic()
    body = json.dumps({"model": MODEL, "keep_alive": "30m", "stream": False, "options": {"num_ctx": 65536}}).encode()
    with urlopen(Request("http://127.0.0.1:11434/api/generate", data=body,
            headers={"Content-Type": "application/json"}), timeout=180) as response:
        warmup = _save(ROOT / "warmup.json", {"response": json.load(response), "seconds": time.monotonic() - clock,
            "scope": "model loading without task prompt"})
    _save(ROOT / "started.json", {"plan_ref": plan_ref, "run_id": plan["run_id"], "execution_config_ref": plan["execution_config_ref"],
        "warmup_ref": warmup, "scripted_actions": False, "outer_supplied_solution": False,
        "assistance": "Public issue/reproduction and platform/environment preparation; no private repair or patch supplied.",
        "human_active_seconds": None, "human_timing_basis": "unknown"})
    manifest = plan["platform_ref"]
    result = execute([sys.executable, "-I", "-B", str(SNAPSHOT / "scripts/run_frozen_product.py"), "--_child",
        manifest["path"], manifest["sha256"]], cwd=SNAPSHOT, stdin_path=None, output=ROOT / "execution",
        deadline=time.monotonic() + plan["wall_seconds"], environment={**os.environ, **plan["environment"], "PYTHONUTF8": "1"})
    print(json.dumps({"run_id": plan["run_id"], "status": result["status"], "process_ref": _ref(ROOT / "execution/process.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "baseline", "freeze", "run"])
    args = parser.parse_args()
    preflight(use_baseline=True) if args.mode == "baseline" else globals()[args.mode]()
