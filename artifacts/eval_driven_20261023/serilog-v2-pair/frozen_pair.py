"""Freeze and run an explicit exposed-task pair using existing normal owners.

为什么要有它：控制器只绑定输入、记录准备成本和进程边界，不给模型预设工具顺序或补丁。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.run_benchmark_arm import execute  # noqa: E402
from src.mod_agent.artifacts import ArtifactStore  # noqa: E402
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.evaluation.strategy_repository_evaluator import _configuration  # noqa: E402
from src.mod_agent.evaluation.strategy_trial_preparation import (  # noqa: E402
    _source_snapshot,
    prepare_strategy_trials,
    verify_strategy_trials,
)
from src.mod_agent.model_admission import read_ollama_model_identity  # noqa: E402
from src.mod_agent.model_qualification import canonical_hash, read_bound_json  # noqa: E402
from src.mod_agent.planning.strategy_experience import StrategyExperience  # noqa: E402
from src.mod_agent.platform_execution_snapshot import (  # noqa: E402
    FrozenExecution,
    prepare_snapshot,
)
from src.mod_agent.public_failure_feedback import validated_public_checks  # noqa: E402

ROOT = Path("C:/tmp/rf-serilog-v2-pair-main")
BASE = Path("C:/tmp/rfcode-serilog-v2-pair-base")
MODEL = "qwen3-coder:30b"
DIGEST = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"
JUDGE = {"path": str(Path("C:/tmp/rf-serilog-v2-pair-eval/qualification-final/evaluator.json").resolve()),
    "sha256": "4b6f85bd95d7ed30a555f4ffb58f207b618f413a4c941b1cf8788014ac9f3bef"}


def api(name, body=None, timeout=10):
    req = Request("http://127.0.0.1:11434/api/" + name,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def freeze():
    public = read_bound_json(_ref(ROOT / "public-preparation.json"))
    baseline = read_bound_json(_ref(ROOT / "baseline-result.json"))
    strategy = read_bound_json(_ref(ROOT / "strategy-preparation.json"))
    qualification_ref = {"path": "C:/tmp/rf-serilog-v2-pair-eval/public-controls/qualification.json",
        "sha256": "5bf99dd44b5543b0ce49e3917b5faba31de36e99ccaf3368658bdb100bf19c1d"}
    qualification = read_bound_json(qualification_ref)
    if (qualification.get("qualified") is not True or qualification.get("hidden_tests_copied") is not False
            or qualification["public_input_ref"] != _ref(ROOT / "public-preparation.json")):
        raise ValueError("PUBLIC_QUALIFICATION_CHANGED")
    for role, row in qualification["controls"].items():
        report, checks = validated_public_checks(row["report_ref"])
        if report["status"] != ("behavior_failed" if role == "wrong" else "passed") or not checks:
            raise ValueError("PUBLIC_CONTROL_CHANGED")
    if baseline["status"] != "behavior_failed" or not baseline["local_feed_only"]:
        raise ValueError("PUBLIC_BASELINE_REQUIRED")
    _configuration(JUDGE)
    identity = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    if identity["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    model_ref = _save(ROOT / "model-before-freeze.json", {"identity": identity, "service_version": api("version"),
        "loaded_models": api("ps"), "model_calls": 0})
    code_ref = prepare_snapshot(source_root=REPO, snapshot_root=BASE, business_argv=["--help"])
    memory = StrategyExperience(ArtifactStore(strategy["store_root"]))
    snapshot = memory.check_snapshot(strategy["snapshot_ref"])
    task_id = public["task_id"]
    request = read_bound_json(public["request_ref"])
    source_ref = _save(ROOT / "task-source.json", {"task_id": task_id, "task_family": "object_representation",
        "source_version": "serilog-3f6d855a-plus-public-reproduction-v1",
        "source_files_sha256": canonical_hash(_source_snapshot(public["source_root"])),
        "original_request_sha256": canonical_hash(request["description"]),
        "public_preparation_ref": _ref(ROOT / "public-preparation.json")})
    exposure = _save(ROOT / "task-exposure.json", {"task_id": task_id, "source_ref": source_ref, "attempt": 1,
        "exposure": "exposed", "reason": "Earlier Serilog development attempts; new v2 pair is not held-out migration."})
    profile = read_bound_json(public["profile_ref"])
    invariants = {"task": {"source_ref": source_ref, "request_ref": public["request_ref"]},
        "model": {"name": MODEL, "digest": DIGEST, "context_window": 65536, "output_tokens": 4096, "identity_ref": model_ref},
        "tools": {"platform_ref": code_ref, "protocol": "v2", "tool_rules": "identical"},
        "permissions": {"scope": "file", "operation": "modify_existing", "allowed_edit_paths": profile["allowed_edit_paths"]},
        "budget": {"policy": "meter_only", "process_wall_seconds_per_arm": 1800, "request_timeout_seconds": 600},
        "judge": {"evaluator_ref": JUDGE},
        "initial_environment": {"dotnet_ref": profile["dotnet_ref"], "restore_environment": baseline["environment"],
            "warmup": "model_load_without_task_prompt_before_each_arm", "shared_service_configuration": "not_owned_or_reconfigured"},
        "feedback": {"public_profile_ref": public["profile_ref"], "public_qualification_ref": qualification_ref}}
    protocol = _save(ROOT / "trial-protocol.json", {"schema_version": "strategy_trial_protocol_v1", "batch_complete": True,
        "snapshot_ref": strategy["snapshot_ref"], "invariants": invariants, "invariants_sha256": canonical_hash(invariants),
        "task_families": ["object_representation"], "planned_pairs": [{"task_id": task_id, "task_family": "object_representation",
            "attempt": 1, "source_ref": source_ref, "exposure_ref": exposure,
            "current_context": {"runtime": "dotnet", "target_domain": "csharp_repository"}}],
        "scope": "exposed_manual_strategy_development_comparison_no_activation_or_migration_claim",
        "strategy_ref": snapshot["strategy_refs"][0], "public_qualification_summary_fields": sorted(qualification)})
    config = _save(ROOT / "base-execution.json", {"arguments": ["--target-project", public["source_root"],
        "--workspace-root", str(ROOT / "unused-base-runs"), "--settings", public["settings_ref"]["path"],
        "--repository-profile", public["profile_ref"]["path"], "--local-ollama", "--ollama-model", MODEL,
        "--ollama-base-url", "http://127.0.0.1:11434", "--ollama-context-window", "65536",
        "--ollama-max-output-tokens", "4096", "--ollama-timeout-seconds", "600",
        "--model-call-budget-policy", "meter_only", "--source-retrieval", "rg_live", "--confirm-isolated-write",
        "--allow-build", "--defer-approval", "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
        "--constraint-policy", "source_priority_v1", "--task-agent-edit-paths", json.dumps(profile["allowed_edit_paths"])]})
    input_ref = _save(ROOT / "preparation-input.json", {"schema_version": "strategy_trial_preparation_input_v1",
        "protocol_ref": protocol, "store_root": strategy["store_root"], "extract_on_terminal": True,
        "tasks": [{"task_id": task_id, "attempt": 1, "execution_config_ref": config, "request_ref": public["request_ref"]}],
        "order": [{"task_id": task_id, "attempt": 1, "arm": arm} for arm in ("B", "B+")]})
    prepared_ref = prepare_strategy_trials(input_ref, ROOT / "paired-inputs")
    prepared = verify_strategy_trials(prepared_ref)
    arms = []
    for trial in prepared["trials"]:
        folder = Path("C:/tmp/rfcode-serilog-v2-pair-" + trial["arm"].replace("+", "plus"))
        refs = {r["path"]: r for r in prepared["input_refs"] + trial["generated_refs"]}
        ref = prepare_snapshot(source_root=BASE, snapshot_root=folder, business_argv=trial["argv"],
            run_root=trial["run_root"], bound_inputs=[(r["path"], r["sha256"]) for r in refs.values()])
        arms.append({**trial, "platform_ref": ref})
    plan = _save(ROOT / "frozen-plan.json", {"prepared_ref": prepared_ref, "base_platform_ref": code_ref,
        "controller_ref": _ref(Path(__file__)), "model_identity_ref": model_ref, "arms": arms,
        "environment": baseline["environment"], "wall_seconds_per_arm": 1800,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "prior_results_not_pooled": True, "model_calls": 0})
    print(json.dumps(plan))


def run(arm):
    plan_ref = _ref(ROOT / "frozen-plan.json")
    plan = read_bound_json(plan_ref)
    if _ref(Path(__file__)) != plan["controller_ref"]:
        raise ValueError("CONTROLLER_CHANGED")
    verify_strategy_trials(plan["prepared_ref"])
    selected = next(row for row in plan["arms"] if row["arm"] == arm)
    if arm == "B+" and not (ROOT / "execution-B/process.json").is_file():
        raise ValueError("PREDECLARED_ORDER_REQUIRED")
    output = ROOT / ("execution-" + arm.replace("+", "plus"))
    if output.exists() or Path(selected["run_root"]).exists() or (ROOT / ("started-" + arm.replace("+", "plus") + ".json")).exists():
        raise ValueError("ATTEMPT_ALREADY_STARTED_INSPECT_ORIGINAL")
    FrozenExecution(selected["platform_ref"]).verify_files()
    identity = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    if identity["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    clock = time.monotonic()
    warmed = api("generate", {"model": MODEL, "keep_alive": "30m", "stream": False,
        "options": {"num_ctx": 65536}}, timeout=180)
    warmup = _save(ROOT / ("warmup-" + arm.replace("+", "plus") + ".json"), {
        "response": warmed, "seconds": time.monotonic() - clock, "scope": "preparation_without_task_prompt"})
    _save(ROOT / ("started-" + arm.replace("+", "plus") + ".json"), {"plan_ref": plan_ref,
        "arm": arm, "run_id": selected["run_id"], "execution_config_ref": selected["execution_config_ref"],
        "warmup_ref": warmup, "model_identity": identity, "scripted_actions": False, "outer_supplied_solution": False,
        "developer_assistance": "Public issue reproduction, registered environment and manual generic strategy; no patch or hidden case supplied."})
    manifest = selected["platform_ref"]
    snapshot_root = Path(manifest["path"]).parent
    outcome = execute([sys.executable, "-I", "-B", str(snapshot_root / "scripts/run_frozen_product.py"),
        "--_child", manifest["path"], manifest["sha256"]], cwd=snapshot_root, stdin_path=None, output=output,
        deadline=time.monotonic() + plan["wall_seconds_per_arm"], environment={**os.environ, **plan["environment"],
            "PYTHONUTF8": "1", "DOTNET_CLI_UI_LANGUAGE": "en-US"})
    print(json.dumps({"arm": arm, "process_status": outcome["status"], "run_id": selected["run_id"],
        "result_ref": _ref(output / "process.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "B", "B+"])
    args = parser.parse_args()
    freeze() if args.mode == "freeze" else run(args.mode)
