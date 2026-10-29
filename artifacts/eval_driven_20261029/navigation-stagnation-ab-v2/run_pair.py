"""One preregistered development comparison using existing Task Run owners.

为什么要有它：两臂复用正常入口和硬停止执行器，只有冻结平台中的导航反馈机制不同。
"""
import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import httpx
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts import run_repository_product_pilot as pilot  # noqa: E402
from scripts.dsh_run_request import OpenTaskRun, _configuration_arguments  # noqa: E402
from scripts.run_repository_product_pilot import (  # noqa: E402
    frozen_task_run_command,
    source_snapshot,
)
from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile  # noqa: E402
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.model_admission import read_ollama_model_identity  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402
from src.mod_agent.platform_execution_snapshot import (  # noqa: E402
    FrozenExecution,
    prepare_snapshot,
)

MODEL = "qwen3-coder:30b"
DIGEST = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"
BASELINE = TREATMENT = "3569e61bf541288a02485f93b5cc43341d1f5c3a"
CHANGED = "src/mod_agent/task_agent_v2_working_state.py"
ADMISSION_SHA = "c3455e5a04a7ebc9af3bfac52adf326da4f0e9676aeb2122f4aad236ddf42bd6"
PREPARED_SHA = "875283f7a9e77e31cb07f22993ef7353dfd91b601be7c08f5c945e145acfe802"
REQUIREMENTS = {"evaluation_version": "navigation-stagnation-ab-v2-restsharp-2241", "threshold": 8,
    "navigation_observation_bytes": 131072, "navigation_seconds": 600,
    "post_trigger_reserve": {"reads": 2, "model_calls": 3, "observation_bytes": 65536, "seconds": 180}}


def verify_refs(refs):
    for reference in refs:
        if _ref(reference["path"]) != reference:
            raise ValueError("BOUND_INPUT_CHANGED: " + reference["path"])


def verify_owner(compatibility):
    refs = compatibility["imported_owner_file_refs"]
    if len(refs) != 175:
        raise ValueError("OWNER_DEPENDENCY_COUNT_CHANGED")
    verify_refs([compatibility["python_executable_ref"], compatibility["original_evaluator_ref"], *refs])


def qualify(admission, compatibility, root):
    # 为什么要有它：仅调用原owner的只读验真入口，禁止重跑控制或用通用资格schema替代。
    verify_owner(compatibility)
    evaluator = admission["original_evaluator_ref"]
    if evaluator != compatibility["original_evaluator_ref"]:
        raise ValueError("OWNER_CHANGED")
    code = ("import importlib.util,sys; p=sys.argv[1]; "
        "s=importlib.util.spec_from_file_location('original_evaluator',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "q=m.verified_qualification(); assert q['qualified'] is True; print('verified')")
    result = subprocess.run([compatibility["python_executable_ref"]["path"], "-B", "-c", code,
        evaluator["path"]], cwd=Path(evaluator["path"]).parents[2], capture_output=True,
        env={**os.environ, "PYTHONUTF8": "1"}, timeout=120)
    (root / "qualification.stdout.txt").write_bytes(result.stdout)
    (root / "qualification.stderr.txt").write_bytes(result.stderr)
    verify_owner(compatibility)
    if result.returncode != 0 or result.stdout.strip() != b"verified":
        raise ValueError("ORIGINAL_QUALIFICATION_REVALIDATION_FAILED")
    return _save(root / "qualification-revalidation.json", {"status": "passed",
        "evaluator_ref": evaluator, "compatibility": compatibility,
        "stdout_ref": _ref(root / "qualification.stdout.txt"),
        "stderr_ref": _ref(root / "qualification.stderr.txt"), "controls_rerun": False})


def disk_admit(path, gib):
    free = shutil.disk_usage(path).free
    if free < gib * 1024**3:
        raise ValueError(f"DISK_BOUNDARY: available={free}, required_gib={gib}; no old data deleted")
    return free


def model_identity():
    identity = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.get("http://127.0.0.1:11434/api/version")
        response.raise_for_status()
        version = response.json()["version"]
    if identity["digest"] != DIGEST or version != "0.40.11":
        raise ValueError("MODEL_OR_OLLAMA_CHANGED")
    return {**identity, "ollama_version": version}



def prepare(config_path):
    config_ref = _ref(config_path)
    config = read_bound_json(config_ref)
    protocol = read_bound_json(config["intent_ref"])
    if protocol["navigation_feedback_experiment"] != REQUIREMENTS:
        raise ValueError("REGISTERED_RESOURCE_REQUIREMENTS_CHANGED")
    common = protocol["common_conditions"]
    expected_common = {"model": MODEL, "digest": DIGEST, "ollama_version": "0.40.11",
        "seed": 0, "temperature": 0, "context_tokens": 65536, "max_output_tokens": 4096,
        "wall_seconds": 900, "model_reservations": 16, "read_calls": 12,
        "observation_bytes": 262144, "strategy_experience": False, "task_memory": False, "retries": 0}
    if any(common.get(key) != value for key, value in expected_common.items()):
        raise ValueError("REGISTERED_COMMON_CONDITIONS_CHANGED")
    if config["admission_ref"]["sha256"] != ADMISSION_SHA or config["prepared_ref"]["sha256"] != PREPARED_SHA:
        raise ValueError("REGISTERED_INPUT_CHANGED")
    admission = read_bound_json(config["admission_ref"])
    prepared = read_bound_json(config["prepared_ref"])
    compatibility = read_bound_json(config["compatibility_ref"])
    if _ref(sys.executable) != compatibility["python_executable_ref"]:
        raise ValueError("CONFIGURED_PYTHON_REQUIRED")
    if admission["status"] != "admitted" or admission["task"] != config["task_id"]:
        raise ValueError("TASK_NOT_ADMITTED")
    if prepared["admission_ref"] != config["admission_ref"]:
        raise ValueError("PREPARATION_ADMISSION_CHANGED")
    config.update(public_profile_ref=prepared["profile_ref"], public_source_root=prepared["source_root"],
        problem_ref=prepared["problem_ref"], qualification_ref=admission["qualification_ref"])
    qualification = {"evaluator_ref": admission["original_evaluator_ref"]}
    verify_refs([*admission["public_input_refs"].values(), admission["sdk_ref"],
        prepared["problem_ref"], config["intent_ref"]])
    if prepared["problem_ref"]["sha256"] != admission["public_input_refs"]["public_problem_ref"]["sha256"]:
        raise ValueError("PUBLIC_PROBLEM_CHANGED")
    profile = CSharpRepositoryProfile(config["public_profile_ref"])
    source = Path(config["public_source_root"])
    profile.verify_fixed_inputs(source, prepared["fixed_input_binding"])
    # 为什么要有它：准备清单包含全部公开文件，运行投影排除.idea；先逐字节核验完整原件，不能混用两种清单。
    full_source = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in source.rglob("*") if path.is_file()}
    if full_source != {row["path"]: row["sha256"] for row in prepared["source_manifest"]}:
        raise ValueError("PREPARED_SOURCE_CHANGED")
    model_identity()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "config"], cwd=REPO, text=True)
    runtime_diff = subprocess.run(["git", "diff", "--quiet", TREATMENT, "HEAD", "--", "src", "scripts", "config"], cwd=REPO).returncode
    # 为什么要有它：文档提交不改变冻结运行代码；运行目录仍须无修改，且与登记版本完全相同。
    ci_diff = subprocess.run(["git", "diff", "--quiet", "0c644d0", TREATMENT,
        "--", "src", "scripts", "config"], cwd=REPO).returncode
    if runtime_diff != 0 or ci_diff != 0 or dirty:
        raise ValueError("PLATFORM_NOT_FROZEN_CLEAN")
    root = Path(config["root"])
    if root.resolve() != Path("C:/tmp/rfn2241v2").resolve():
        raise ValueError("UNREGISTERED_ROOT")
    disk_before = disk_admit(root.parent, 3)
    root.mkdir(parents=True, exist_ok=False)
    qualification_check = qualify(admission, compatibility, root)
    worker_ref = _save(root / "worker-policy.json", {"schema_version": "patch_worker_configuration_v1",
        "policy": {"model_call_budget_policy": "task_total_only", "max_tool_rounds": 12, "max_observation_bytes": 262144,
                   "max_total_model_turns": 16}, "enable_recovery_loop": False, "behavior_evaluator": None})
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
            else:
                raise ValueError("CHECKOUT_DIFFERS_FROM_COMMIT")
        sources[arm] = exported
    before = (sources["B0"] / CHANGED).read_bytes()
    old = b"navigation_only = navigation >= NAVIGATION_STAGNATION_THRESHOLD"
    if before.count(old) != 1:
        raise ValueError("ABLATION_PREIMAGE_CHANGED")
    after = before.replace(old, b"navigation_only = False")
    (sources["B0"] / CHANGED).write_bytes(after)
    difference = "".join(difflib.unified_diff(before.decode().splitlines(True),
        after.decode().splitlines(True), fromfile="B1/" + CHANGED, tofile="B0/" + CHANGED))
    (root / "ablation.diff").write_bytes(difference.encode())
    difference_ref = _save(root / "ablation-manifest.json", {"diff_ref": _ref(root / "ablation.diff"),
        "B0_ref": _ref(sources["B0"] / CHANGED), "B1_ref": _ref(sources["B1"] / CHANGED),
        "threshold": 8, "only_replacement": [old.decode(), "navigation_only = False"]})
    problem = Path(config["problem_ref"]["path"])
    if _ref(problem) != config["problem_ref"]:
        raise ValueError("PROBLEM_CHANGED")
    intent = problem.read_bytes().decode("utf-8")
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
        settings["llm"].update(provider="ollama", model=MODEL, base_url="http://127.0.0.1:11434", max_tokens=4096, temperature=0.0)
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
        arguments.extend(["--patch-worker-config", worker_ref["path"]])
        common_environment = {"PYTHONUTF8": "1", "DOTNET_CLI_UI_LANGUAGE": "en-US",
            "DOTNET_ROOT": str(Path(profile.data["dotnet_ref"]["path"]).parent),
            "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "", "NO_PROXY": "127.0.0.1,localhost"}
        arm_config = _save(arm_root / "configuration.json", {
            "mode": "absolute_real_B", "intent_ref": config["intent_ref"],
            "public_problem_ref": config["problem_ref"],
            "public_contract_ref": profile.reference,
            "public_contract_scope": "bound repository execution profile; no additional solver prompt",
            "public_profile_ref": profile.reference, "qualification_ref": config["qualification_ref"],
            "settings_ref": _ref(settings_path), "environment": common_environment, "patch_worker_config_ref": worker_ref})
        base = _save(arm_root / "base-execution.json", {"arguments": arguments})
        command = OpenTaskRun(execution_config_ref=base, intent=intent).bind()
        _, options = _configuration_arguments(command.execution_config_ref["path"])
        copied_worker = _ref(options["--patch-worker-config"])
        if read_bound_json(copied_worker) != read_bound_json(worker_ref):
            raise ValueError("WORKER_CHANGED_DURING_BIND")
        capacity = pilot.validate_navigation_experiment(command.execution_config_ref, REQUIREMENTS, wall_seconds=900)
        if capacity["worker_configuration_ref"] != copied_worker or capacity["capacity"] != {
                "reads": 12, "model_calls": 16, "observation_bytes": 262144, "seconds": 900}:
            raise ValueError("RESOURCE_POLICY_CHANGED")
        capacity_ref = _save(arm_root / "navigation-admission.json", capacity)
        run_root = arm_root / "runs" / options["--run-id"]
        refs = [config_ref, arm_config, qualification["evaluator_ref"], command.execution_config_ref, profile.reference, config["problem_ref"],
            config["qualification_ref"], _ref(settings_path), _ref(__file__)]
        refs.extend([config["admission_ref"], config["prepared_ref"], config["compatibility_ref"],
            config["intent_ref"], worker_ref, copied_worker, capacity_ref, qualification_check,
            admission["sdk_ref"], difference_ref, compatibility["python_executable_ref"],
            *compatibility["imported_owner_file_refs"], *admission["public_input_refs"].values()])
        if "protocol_ref" in config:
            refs.append(config["protocol_ref"])
        verify_refs(refs)
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
            "model_identity": model_identity(),
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
    expected = [CHANGED]
    if changed != expected:
        raise ValueError("UNREGISTERED_PLATFORM_DIFFERENCE: " + str(changed))
    reference = _save(root / "pair.json", {"order": ["B0", "B1"], "plans": plans,
        "changed_platform_files": changed, "ablation_manifest_ref": difference_ref, "disk_free_before": disk_before,
        "compatibility_ref": config["compatibility_ref"], "controller_ref": _ref(__file__), "task_id": config["task_id"], "model_calls": 0,
        "preparation_checkout_commit": head, "baseline_commit": BASELINE, "treatment_commit": TREATMENT,
        "source_byte_policy": "Git commit exports; files equivalent after CRLF normalization use the same verified checkout bytes in both arms, preserving common evaluator identity.",
        "scope": "development diagnostic; one run per arm; no task-specific rescue"})
    print(json.dumps(reference), flush=True)


def run(pair_path, arm, expected_sha256):
    pair_ref = {"path": str(Path(pair_path).resolve()), "sha256": expected_sha256}
    pair = read_bound_json(pair_ref)
    verify_refs([pair["controller_ref"]])
    if pair["controller_ref"] != _ref(__file__):
        raise ValueError("PAIR_CONTROLLER_CHANGED")
    plan_ref = pair["plans"][arm]
    plan = read_bound_json(plan_ref)
    root = Path(plan["root"])
    for previous in pair["order"][:pair["order"].index(arm)]:
        previous_result = Path(pair["plans"][previous]["path"]).parent / "result.json"
        if not previous_result.is_file():
            raise ValueError("SERIAL_ORDER_REQUIRED")
        previous_process = read_bound_json(read_bound_json(_ref(previous_result))["process_ref"])
        previous_plan = read_bound_json(pair["plans"][previous])
        previous_result_data = read_bound_json(_ref(previous_result))
        processes = [previous_process]
        if previous_result_data.get("delivery_process"):
            processes.append(previous_result_data["delivery_process"])
        for process in processes:
            if process.get("status") == "infrastructure_failed" or process.get("owned_process_residuals"):
                raise ValueError("PREVIOUS_PLATFORM_FAILURE_COMPARISON_STOPPED")
            for key in ("stdout_ref", "stderr_ref"):
                if process.get(key):
                    verify_refs([process[key]])
                    if b"Traceback (most recent call last)" in Path(process[key]["path"]).read_bytes():
                        raise ValueError("PREVIOUS_TRACEBACK_COMPARISON_STOPPED")
        if previous_process.get("timeout") is not True:
            if previous_process.get("exit_code") not in (0, 2):
                raise ValueError("PREVIOUS_ABNORMAL_EXIT")
            pilot._task_run_process_result(previous_process, previous_plan)
        disk_admit(root, 1.5)
    if (root / "started.json").exists() or (root / "controller-started.json").exists():
        raise ValueError("ALREADY_STARTED_NO_RETRY")
    FrozenExecution(plan["platform_ref"]).verify_files()
    for ref in plan["bound_inputs"]:
        if _ref(ref["path"]) != ref:
            raise ValueError("BOUND_INPUT_CHANGED")
    if source_snapshot(plan["source_root"]) != plan["source_before"]:
        raise ValueError("SOURCE_CHANGED")
    model_identity()
    compatibility = read_bound_json(pair["compatibility_ref"])
    if _ref(sys.executable) != compatibility["python_executable_ref"]:
        raise ValueError("CONFIGURED_PYTHON_REQUIRED")
    verify_owner(compatibility)
    # 为什么要有它：两臂均卸载驻留模型并冻结版本，缓存差异不能被误当作反馈机制效果。
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post("http://127.0.0.1:11434/api/generate", json={"model": MODEL, "keep_alive": 0})
        response.raise_for_status()
        if client.get("http://127.0.0.1:11434/api/ps").json().get("models"):
            raise ValueError("MODEL_NOT_UNLOADED")
    arm_pilot = read_bound_json(_ref(root / "pilot.json"))
    if arm_pilot["plans"] != {"RepoForge_B": plan_ref}:
        raise ValueError("ARM_PLAN_CHANGED")
    # 为什么要有它：控制器级独占标记把冷启动准备也纳入一次尝试，派发后绝不重试。
    with (root / "controller-started.json").open("x", encoding="utf-8") as stream:
        json.dump({"pair_ref": pair_ref, "plan_ref": plan_ref}, stream)
    try:
        pilot.run(_ref(root / "pilot.json"), "RepoForge_B")
    finally:
        verify_owner(compatibility)


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
