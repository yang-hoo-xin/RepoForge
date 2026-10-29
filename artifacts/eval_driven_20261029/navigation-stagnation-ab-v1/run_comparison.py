"""One preregistered development comparison using existing Task Run owners.

为什么要有它：两臂复用正常入口和硬停止执行器，只有冻结平台中的导航反馈机制不同。
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import httpx
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.dsh_run_request import OpenTaskRun, _configuration_arguments
from scripts import run_repository_product_pilot as pilot
from scripts.run_repository_product_pilot import frozen_task_run_command, source_snapshot
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile
from src.mod_agent.model_admission import read_ollama_model_identity
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.platform_execution_snapshot import prepare_snapshot, FrozenExecution

MODEL = "qwen3-coder:30b"
DIGEST = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"
BASELINE = "f059c4a1a8de4290924ff76c486e7fe5631c6c31"
TREATMENT = "ea4e532085861e3ea07bccc4461ef9ff91527b45"


def prepare(config_path):
    config_ref = _ref(config_path)
    config = read_bound_json(config_ref)
    qualification = read_bound_json(config["qualification_ref"])
    if qualification.get("qualified") is not True:
        raise ValueError("TASK_NOT_ADMITTED")
    # 为什么要有它：qualified自报状态不足以准入，须由原有验收器校验全部已执行控制原件。
    pilot.validate_frozen_evaluator(qualification)
    if (qualification["public_profile_ref"] != config["public_profile_ref"]
            or Path(qualification["public_source_root"]).resolve() != Path(config["public_source_root"]).resolve()
            or qualification["task_id"] != config["task_id"]):
        raise ValueError("QUALIFICATION_TASK_CHANGED")
    profile = CSharpRepositoryProfile(config["public_profile_ref"])
    source = Path(config["public_source_root"])
    profile.bind(source)
    if read_ollama_model_identity("http://127.0.0.1:11434", MODEL)["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "config"], cwd=REPO, text=True)
    runtime_diff = subprocess.run(["git", "diff", "--quiet", TREATMENT, "HEAD", "--", "src", "scripts", "config"], cwd=REPO).returncode
    # 为什么要有它：文档提交不改变冻结运行代码；运行目录仍须无修改，且与登记版本完全相同。
    if runtime_diff != 0 or dirty:
        raise ValueError("PLATFORM_NOT_FROZEN_CLEAN")
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=False)
    sources = {}
    for arm, revision in (("B0", BASELINE), ("B1", TREATMENT)):
        exported = root / (arm + "-platform-source")
        archive = root / (arm + "-source.zip")
        subprocess.run(["git", "archive", "--format=zip", "--output", str(archive), revision,
            "src", "scripts", "config"], cwd=REPO, check=True)
        with zipfile.ZipFile(archive) as z:
            if any(not (exported / name).resolve().is_relative_to(exported.resolve()) for name in z.namelist()):
                raise ValueError("ARCHIVE_PATH_ESCAPE")
            z.extractall(exported)
        # 为什么要有它：Git换行规范化不能把相同验收器变成不同原始字节；共同模块复用已验证checkout字节。
        for path in exported.rglob("*"):
            if not path.is_file():
                continue
            current = REPO / path.relative_to(exported)
            raw, actual = path.read_bytes(), current.read_bytes()
            if raw.replace(b"\r\n", b"\n") == actual.replace(b"\r\n", b"\n"):
                path.write_bytes(actual)
            elif arm == "B1":
                raise ValueError("CHECKOUT_DIFFERS_FROM_COMMIT")
        sources[arm] = exported
    problem = Path(config["problem_ref"]["path"])
    if _ref(problem) != config["problem_ref"]:
        raise ValueError("PROBLEM_CHANGED")
    intent = problem.read_text(encoding="utf-8")
    plans = {}
    for arm in ("B0", "B1"):
        arm_root = root / arm
        workspace = arm_root / "workspace"
        target = workspace / "source"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "bin", "obj", "TestResults", "__pycache__"))
        if source_snapshot(target) != source_snapshot(source):
            raise ValueError("COMMON_SOURCE_CHANGED")
        settings = yaml.safe_load((REPO / "config/settings.yaml").read_text(encoding="utf-8"))
        settings["task_memory"] = {"enabled": False, "store_root": None, "max_context_chars": 3000}
        settings["llm"].update(provider="ollama", model=MODEL, base_url="http://127.0.0.1:11434", max_tokens=4096)
        settings_path = arm_root / "settings.yaml"
        with settings_path.open("x", encoding="utf-8") as f:
            yaml.safe_dump(settings, f, allow_unicode=True)
        arguments = ["--target-project", str(target), "--workspace-root", str(arm_root / "runs"),
            "--settings", str(settings_path), "--repository-profile", profile.reference["path"],
            "--local-ollama", "--ollama-model", MODEL, "--ollama-base-url", "http://127.0.0.1:11434",
            "--ollama-context-window", "65536", "--ollama-max-output-tokens", "4096", "--ollama-timeout-seconds", "600",
            "--model-call-budget-policy", "task_total_only", "--task-model-call-limit", "16",
            "--source-retrieval", "rg_live", "--confirm-isolated-write", "--allow-build", "--defer-approval",
            "--technical-executor", "task_agent", "--task-agent-protocol", "v2", "--task-agent-semantic-state", "enabled",
            "--constraint-policy", "source_priority_v1", "--task-agent-edit-paths", json.dumps(profile.data["allowed_edit_paths"])]
        common_environment = {"PYTHONUTF8": "1", "DOTNET_CLI_UI_LANGUAGE": "en-US",
            "DOTNET_ROOT": str(Path(profile.data["dotnet_ref"]["path"]).parent),
            "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "", "NO_PROXY": "127.0.0.1,localhost"}
        arm_config = _save(arm_root / "configuration.json", {
            "mode": "absolute_real_B", "intent_ref": config["intent_ref"],
            "public_problem_ref": config["problem_ref"],
            "public_contract_ref": profile.reference,
            "public_contract_scope": "bound repository execution profile; no additional solver prompt",
            "public_profile_ref": profile.reference, "qualification_ref": config["qualification_ref"],
            "settings_ref": _ref(settings_path), "environment": common_environment})
        base = _save(arm_root / "base-execution.json", {"arguments": arguments})
        command = OpenTaskRun(execution_config_ref=base, intent=intent).bind()
        _, options = _configuration_arguments(command.execution_config_ref["path"])
        run_root = arm_root / "runs" / options["--run-id"]
        refs = [config_ref, arm_config, qualification["evaluator_ref"], command.execution_config_ref, profile.reference, config["problem_ref"],
            config["qualification_ref"], _ref(settings_path), _ref(__file__)]
        snapshot = prepare_snapshot(source_root=sources[arm],
            snapshot_root=arm_root / "platform", business_argv=list(command.business_argv), run_root=run_root,
            bound_inputs=[(r["path"], r["sha256"]) for r in refs])
        policy_id = "user_delegated_navigation_diagnostic_isolated_delivery"
        authorization = _save(arm_root / "host-delivery-authorization.json", {
            "origin": "task_delegation_policy", "decision": "approve", "policy_id": policy_id,
            "runtime": False, "deployment": False, "run_id": options["--run-id"],
            "request_id": options["--request-id"], "isolated_root": str(arm_root / "delivery"),
            "allowed_effects": ["isolated_apply", "build"], "human_semantic_review_completed": False,
            "authorization_source": "existing user delegation for the single registered B0/B1 comparison",
            "why": "为什么要有它：实际交付仍由精确候选和前像绑定的原有委托校验，不能把模型建议当授权。"})
        plan = {"arm": "RepoForge_B", "comparison_arm": arm, "root": str(arm_root), "workspace": str(workspace), "source_root": str(target),
            "source_before": source_snapshot(target), "configuration_ref": arm_config,
            "controller_ref": _ref(pilot.__file__), "environment": common_environment,
            "allowed_edit_paths": profile.data["allowed_edit_paths"],
            "model_identity": read_ollama_model_identity("http://127.0.0.1:11434", MODEL),
            "authorization_ref": authorization, "policy_id": policy_id,
            "run_id": options["--run-id"], "request_id": options["--request-id"], "run_root": str(run_root),
            "execution_config_ref": command.execution_config_ref, "platform_ref": snapshot, "bound_inputs": refs,
            "command": frozen_task_run_command(snapshot), "wall_seconds": 900, "model_call_limit": 16,
            "model": MODEL, "digest": DIGEST, "strategy_injected": False,
            "evaluator_ref": qualification["evaluator_ref"], "public_profile_ref": profile.reference}
        plans[arm] = _save(arm_root / "plan.json", plan)
        _save(arm_root / "pilot.json", {"mode": "absolute_real_B", "order": ["RepoForge_B"],
            "plans": {"RepoForge_B": plans[arm]}, "configuration_ref": arm_config})
    documents = [read_bound_json(read_bound_json(plans[a])["platform_ref"])["files"] for a in ("B0", "B1")]
    changed = sorted(k for k in documents[0].keys() | documents[1].keys()
        if not k.startswith("external/") and documents[0].get(k) != documents[1].get(k))
    expected = ["src/mod_agent/task_agent_v2_experience.py", "src/mod_agent/task_agent_v2_recovery.py",
        "src/mod_agent/task_agent_v2_working_state.py"]
    if changed != expected:
        raise ValueError("UNREGISTERED_PLATFORM_DIFFERENCE: " + str(changed))
    reference = _save(root / "pair.json", {"order": ["B0", "B1"], "plans": plans,
        "changed_platform_files": changed, "task_id": config["task_id"], "model_calls": 0,
        "preparation_checkout_commit": head, "baseline_commit": BASELINE, "treatment_commit": TREATMENT,
        "source_byte_policy": "Git commit exports; files equivalent after CRLF normalization use the same verified checkout bytes in both arms, preserving common evaluator identity.",
        "scope": "development diagnostic; one run per arm; no task-specific rescue"})
    print(json.dumps(reference), flush=True)


def run(pair_path, arm, expected_sha256):
    pair_ref = {"path": str(Path(pair_path).resolve()), "sha256": expected_sha256}
    pair = read_bound_json(pair_ref)
    plan_ref = pair["plans"][arm]
    plan = read_bound_json(plan_ref)
    root = Path(plan["root"])
    for previous in pair["order"][:pair["order"].index(arm)]:
        previous_result = Path(pair["plans"][previous]["path"]).parent / "result.json"
        if not previous_result.is_file():
            raise ValueError("SERIAL_ORDER_REQUIRED")
        previous_process = read_bound_json(read_bound_json(_ref(previous_result))["process_ref"])
        if previous_process.get("status") == "infrastructure_failed" or previous_process.get("owned_process_residuals"):
            raise ValueError("PREVIOUS_PLATFORM_FAILURE_COMPARISON_STOPPED")
    if (root / "started.json").exists():
        raise ValueError("ALREADY_STARTED_NO_RETRY")
    FrozenExecution(plan["platform_ref"]).verify_files()
    for ref in plan["bound_inputs"]:
        if _ref(ref["path"]) != ref:
            raise ValueError("BOUND_INPUT_CHANGED")
    if source_snapshot(plan["source_root"]) != plan["source_before"]:
        raise ValueError("SOURCE_CHANGED")
    if read_ollama_model_identity("http://127.0.0.1:11434", MODEL)["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    # 为什么要有它：两臂均卸载驻留模型并冻结版本，缓存差异不能被误当作反馈机制效果。
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post("http://127.0.0.1:11434/api/generate", json={"model": MODEL, "keep_alive": 0})
        response.raise_for_status()
        if client.get("http://127.0.0.1:11434/api/ps").json().get("models"):
            raise ValueError("MODEL_NOT_UNLOADED")
    arm_pilot = read_bound_json(_ref(root / "pilot.json"))
    if arm_pilot["plans"] != {"RepoForge_B": plan_ref}:
        raise ValueError("ARM_PLAN_CHANGED")
    pilot.run(_ref(root / "pilot.json"), "RepoForge_B")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("path")
    parser.add_argument("--arm", choices=["B0", "B1"])
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.path)
    else:
        if not args.arm or not args.sha256:
            parser.error("run requires the frozen arm and pair sha256")
        run(args.path, args.arm, args.sha256)
