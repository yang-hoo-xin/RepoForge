"""Bind and run a registered repository product pilot using existing owners.

No solver actions, repair hints or acceptance answers are selected here.
为什么要有它：产品对照需要同一公开任务与真实交付，控制脚本不能成为外层解题器。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import yaml

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dsh_run_request import (
    _configuration_arguments,
    build_product_argv,
    evidence_locator,
    prepare_new_task,
    validate_external_settings_credentials,
)
from scripts.run_benchmark_arm import codex_argv, codex_events, execute
from src.mod_agent.domain.csharp_repository_backend import (
    _IGNORED,
    CSharpRepositoryProfile,
    _relative,
    _snapshot,
)
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.strategy_trial_preparation import verify_strategy_trials
from src.mod_agent.evaluation.trial_comparison_identity import platform_identity
from src.mod_agent.model_admission import read_ollama_model_identity
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.platform_execution_snapshot import FrozenExecution, prepare_snapshot
from src.mod_agent.task_agent_v2_state import _no_links

REPO = Path(__file__).resolve().parents[1]
MODEL = "qwen3-coder:30b"
DIGEST = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"
ORDER = ("Codex_local", "RepoForge_B")


def source_snapshot(root):
    return {row["path"]: row["sha256"] for row in _snapshot(root)}


def local_codex_command(executable, workspace, last_message, deny_roots, dotnet, allowed_edit_paths):
    command = codex_argv(executable, workspace, last_message, deny_roots)
    command[command.index("--model") + 1] = MODEL
    index = command.index('model_reasoning_effort="medium"')
    del command[index - 1:index + 1]
    # 为什么要有它：工具只能查看本臂公开材料，独立验收与其他臂不能成为解题上下文。
    filesystem = {":root": "read", ":workspace_roots": "read", "C:/tmp": "deny",
                  Path(workspace).resolve().as_posix(): "write",
                  Path(dotnet).resolve().parent.as_posix(): "read"}
    filesystem.update({Path(path).resolve().as_posix(): "deny" for path in deny_roots})
    source = Path(workspace) / "source"
    filesystem[source.resolve().as_posix()] = "read"
    for relative in allowed_edit_paths:
        _relative(relative)
        parts = Path(relative).parts
        if Path(relative).is_absolute() or len(parts) < 3 or any(part in {".", ".."} for part in parts):
            raise ValueError("EXACT_ALLOWED_SOURCE_PATH_REQUIRED")
        group = source.joinpath(*parts[:2]).resolve().as_posix()
        filesystem.setdefault(group, {".": "read"})[Path(*parts[2:]).as_posix()] = "write"
    filesystem[(Path(workspace) / "materials").resolve().as_posix()] = "read"
    # Test files are read-only; generated build outputs remain writable.
    # 为什么要有它：公开测试允许执行，不允许改写测试答案；编译输出不是业务源码授权。
    for relative in ("test", "src"):
        for project in (source / relative).glob("*"):
            if project.is_dir():
                for output in ("bin", "obj", "TestResults"):
                    filesystem[(project / output).resolve().as_posix()] = "write"
    for project in source.rglob("*.csproj"):
        filesystem[project.resolve().as_posix()] = "read"
    def toml(value):
        if isinstance(value, dict):
            return "{" + ",".join(json.dumps(k) + "=" + toml(v) for k, v in value.items()) + "}"
        return json.dumps(value)
    table = toml(filesystem)
    index = next(i for i, value in enumerate(command) if value.startswith("permissions.benchmark.filesystem="))
    command[index] = "permissions.benchmark.filesystem=" + table
    command[-1:-1] = ["--oss", "--local-provider", "ollama", "--disable", "unbounded_connection_retries"]
    if len(subprocess.list2cmdline(command)) >= 32767:
        raise ValueError("EXACT_PERMISSION_COMMAND_EXCEEDS_WINDOWS_LIMIT")
    return command


def external_pilot_profile(config):
    """Select the existing external profile only for one absolute TaskAgent attempt.

    为什么要有它：更换模型的参考运行不是同模型经验配对，也不能暗中改变Codex产品对照的共同条件。
    """
    reference = config.get("external_profile_ref")
    if "external_profile_ref" not in config:
        return None
    if config.get("mode") != "absolute_real_B":
        raise ValueError("EXTERNAL_PROFILE_REQUIRES_ABSOLUTE_TASK_AGENT")
    from src.mod_agent.external_patch import (
        ExternalOpenAIResponsesProfile,
        external_profile_from_document,
    )
    profile = external_profile_from_document(read_bound_json(reference))
    if not isinstance(profile, ExternalOpenAIResponsesProfile):
        raise ValueError("EXTERNAL_PILOT_REQUIRES_RESPONSES_PROFILE")
    # 为什么要有它：Windows环境变量名不区分大小写，换一种拼写仍不能把当前凭据值保存进计划。
    if any(str(key).casefold() == profile.credential_env.casefold() for key in config.get("environment", {})):
        raise ValueError("PILOT_CREDENTIAL_VALUE_MUST_NOT_BE_FROZEN")
    return profile


def external_pilot_identity(profile, expected=None):
    from src.mod_agent.external_patch import (
        read_external_openai_model_identity,
        validate_openai_task_identity,
    )
    observed = read_external_openai_model_identity(profile)
    validate_openai_task_identity(profile, observed, expected)
    return observed


def freeze(configuration_ref):
    config = read_bound_json(configuration_ref)
    external = external_pilot_profile(config)
    external_settings = None
    if external is not None:
        # 为什么要有它：冻结配置不能复制其他provider凭据，拒绝应早于外部目录查询或文件写出；原用户配置不改写。
        _verify(config["settings_ref"])
        external_settings = yaml.safe_load(Path(config["settings_ref"]["path"]).read_text(encoding="utf-8"))
        validate_external_settings_credentials(external_settings)
    mode = config.get("mode", "matched_product_pair")
    if mode == "strategy_pair":
        return freeze_strategy_pair(configuration_ref, config)
    if mode not in {"matched_product_pair", "absolute_real_B"}:
        raise ValueError("EXPLICIT_PILOT_MODE_REQUIRED")
    semantic_state = config.get("task_agent_semantic_state", "enabled")
    if type(semantic_state) is not str or semantic_state not in {"enabled", "disabled"}:
        raise ValueError("PILOT_SEMANTIC_STATE_POLICY_INVALID")
    experience_ref = config.get("experience_config_ref")
    experience = None
    if "experience_config_ref" in config:
        experience = read_bound_json(experience_ref)
        if (experience.get("schema_version") != "task_agent_experience_config_v1"
                or not {"store_root", "snapshot_ref", "extract_on_terminal"} <= set(experience)
                or type(experience["extract_on_terminal"]) is not bool):
            raise ValueError("PILOT_EXPERIENCE_CONFIG_INVALID")
    # 为什么要有它：普通产品试跑的语义状态选择必须进入已绑定argv；经验配对仍保留其原冻结输入。
    for key in configuration_keys(config):
        read_bound_json(config[key]) if key in {"intent_ref", "public_profile_ref", "qualification_ref"} else _verify(config[key])
    qualification = read_bound_json(config["qualification_ref"])
    if qualification.get("qualified") is not True:
        raise ValueError("QUALIFIED_ACCEPTANCE_REQUIRED")
    validate_frozen_evaluator(qualification)
    root = Path(config["root"]).resolve()
    public = Path(config["public_source_root"]).resolve(strict=True)
    if root.exists() or root.is_relative_to(public) or public.is_relative_to(root):
        raise ValueError("FRESH_SEPARATE_PILOT_ROOT_REQUIRED")
    profile = CSharpRepositoryProfile(config["public_profile_ref"])
    profile.bind(public)
    if external is None:
        identity = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
        if identity["digest"] != DIGEST:
            raise ValueError("MODEL_CHANGED")
    else:
        identity = external_pilot_identity(external)
    root.mkdir(parents=True)
    public_contract = Path(config["public_contract_ref"]["path"]).read_text(encoding="utf-8")
    common = Path(config["public_problem_ref"]["path"]).read_text(encoding="utf-8") + "\n\n" + public_contract
    common_path = root / "common-request.txt"
    with common_path.open("x", encoding="utf-8") as stream:
        stream.write(common)
    common_ref = _ref(common_path)
    original = source_snapshot(public)
    plans = {}
    arms = ORDER if mode == "matched_product_pair" else (ORDER[1],)
    for arm in arms:
        arm_root = root / arm
        workspace = arm_root / "workspace"
        source = workspace / "source"
        shutil.copytree(public, source, ignore=shutil.ignore_patterns(*_IGNORED))
        if source_snapshot(source) != original:
            raise ValueError("COMMON_SOURCE_COPY_CHANGED")
        materials = workspace / "materials"
        materials.mkdir()
        shutil.copyfile(common_path, materials / "request.txt")
        base = {"schema_version": "repository_product_arm_plan_v1", "arm": arm,
                "root": str(arm_root), "workspace": str(workspace), "source_root": str(source),
                "source_before": original, "common_request_ref": common_ref,
                "configuration_ref": configuration_ref, "wall_seconds": 1800,
                "model_identity": identity, "allowed_edit_paths": profile.data["allowed_edit_paths"],
                "public_profile_ref": profile.reference, "controller_ref": _ref(__file__),
                "environment": config["environment"], "evaluator_ref": qualification["evaluator_ref"]}
        if external is not None:
            base["external_profile_ref"] = config["external_profile_ref"]
        if arm == "Codex_local":
            deny = [REPO, Path.home() / ".codex", Path.home() / ".ssh", Path.home() / ".aws"]
            base["command"] = local_codex_command(config["codex_executable_ref"]["path"], workspace,
                arm_root / "last-message.txt", deny, profile.data["dotnet_ref"]["path"], profile.data["allowed_edit_paths"])
            base["prompt_path"] = str(common_path)
            base["configuration_difference"] = "native OSS adapter; probe context4096, sampling/output options not independently captured; final wire not instrumented"
        else:
            settings = (deepcopy(external_settings) if external is not None else
                yaml.safe_load(Path(config["settings_ref"]["path"]).read_text(encoding="utf-8")))
            settings["task_memory"] = {"enabled": False, "store_root": None, "max_context_chars": 3000}
            if external is None:
                settings["llm"].update(provider="ollama", model=MODEL, base_url="http://127.0.0.1:11434", max_tokens=4096)
                model_arguments = ["--local-ollama", "--ollama-model", MODEL, "--ollama-base-url", "http://127.0.0.1:11434",
                    "--ollama-context-window", "65536", "--ollama-max-output-tokens", "4096", "--ollama-timeout-seconds", "600"]
                model_description = "explicit context65536/output4096/temperature0/seed0"
            else:
                settings["llm"].update(provider="openai", model=external.model,
                    base_url=external.base_url, max_tokens=external.output_budget)
                model_arguments = ["--external-patch-profile", config["external_profile_ref"]["path"],
                    "--api-key-env", external.credential_env, "--base-url-env", "REPOFORGE_OPENAI_BASE_URL",
                    "--allow-source-disclosure"]
                model_description = (f"Responses declared context{external.context_window}/output{external.output_budget}"
                    f"/reasoning {external.reasoning_effort}; remote model ID without binary digest")
            settings_path = arm_root / "settings.yaml"
            with settings_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(settings, stream, allow_unicode=True, sort_keys=False)
            arguments = ["--target-project", str(source), "--workspace-root", str(arm_root / "runs"),
                "--settings", str(settings_path), "--repository-profile", profile.reference["path"],
                *model_arguments, "--model-call-budget-policy", "meter_only",
                "--source-retrieval", "rg_live", "--confirm-isolated-write", "--allow-build", "--defer-approval",
                "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
                "--task-agent-semantic-state", semantic_state,
                "--constraint-policy", "source_priority_v1", "--task-agent-edit-paths", json.dumps(profile.data["allowed_edit_paths"])]
            if experience_ref is not None:
                arguments.extend(["--task-agent-experience-config", experience_ref["path"]])
            execution = _save(arm_root / "base-execution.json", {"arguments": arguments})
            selected = prepare_new_task(execution_config=execution["path"], description=common)
            business = build_product_argv(execution_config=selected, description=common)
            _, options = _configuration_arguments(selected)
            run_root = arm_root / "runs" / options["--run-id"]
            bound = [_ref(selected), _ref(settings_path), profile.reference, common_ref, configuration_ref, _ref(__file__)]
            if external is not None:
                bound.append(config["external_profile_ref"])
            if experience_ref is not None:
                bound.append(experience_ref)
            platform = prepare_snapshot(source_root=REPO, snapshot_root=root / "platform",
                business_argv=business, run_root=run_root, bound_inputs=[(r["path"], r["sha256"]) for r in bound])
            policy_id = "user_goal_disposable_product_pilot_compile_only"
            authorization = _save(arm_root / "host-delivery-authorization.json", {
                "origin": "task_delegation_policy", "decision": "approve", "policy_id": policy_id,
                "runtime": False, "deployment": False, "run_id": options["--run-id"], "request_id": options["--request-id"],
                "isolated_root": str(arm_root / "delivery"), "allowed_effects": ["isolated_apply", "build"],
                "human_semantic_review_completed": False,
                "authorization_source": "existing user continuous engineering delegation for scoped disposable repository writes and build/test; not model output or human technical review",
                "why": "为什么要有它：有效委托仍须核对精确候选、前像和交付范围，不能由模型授予权限。"})
            base.update(run_id=options["--run-id"], run_root=str(run_root), platform_ref=platform,
                execution_config_ref=_ref(selected), authorization_ref=authorization, policy_id=policy_id,
                inspection=evidence_locator(execution_config=selected, run_id=options["--run-id"]),
                bound_inputs=bound, task_agent_semantic_state=semantic_state,
                configuration_difference=model_description + "; semantic working state="
                    + semantic_state + ("; host experience policy explicitly bound" if experience_ref is not None
                        else "; no injected strategy or terminal extraction configured"),
                command=[sys.executable, "-I", "-B", str(root / "platform/scripts/run_frozen_product.py"),
                    "--_child", platform["path"], platform["sha256"]])
            if experience_ref is not None:
                # 为什么要有它：显式配置须同时进入真实请求、冻结输入和可核验计划，不能静默漏传或在派发前变更。
                base.update(experience_config_ref=experience_ref,
                    experience_policy={key: experience.get(key) for key in ("snapshot_ref", "extract_on_terminal", "episode_catalog")})
        plans[arm] = _save(arm_root / "frozen-plan.json", base)
    result = _save(root / "frozen-pilot.json", {"configuration_ref": configuration_ref, "common_request_ref": common_ref,
        "order": list(arms), "plans": plans, "model_calls": 0, "mode": mode,
        "scope": "whole product configurations" if mode == "matched_product_pair" else "absolute real B result; product comparison not covered"})
    print(json.dumps({"pilot_ref": result, "arms_started": 0}))


def _verify(reference):
    # 为什么要有它：同一绝对路径的分隔符写法不应改变文件身份；仍拒绝链接、额外字段和实际字节变化。
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("BOUND_INPUT_CHANGED")
    path = Path(reference["path"])
    if not path.is_absolute():
        raise ValueError("BOUND_INPUT_CHANGED")
    _no_links(path)
    if _ref(path)["sha256"] != reference["sha256"]:
        raise ValueError("BOUND_INPUT_CHANGED")


def validate_frozen_evaluator(qualification):
    """Consume qualification with its original adapter, not the mutable checkout.

    为什么要有它：资格绑定的是实际执行版本及路径，消费端不能改写原件或放宽身份检查。
    """
    reference = qualification["evaluator_ref"]
    snapshot = qualification.get("snapshot_ref", qualification.get("platform_ref"))
    if not snapshot or (qualification.get("snapshot_ref") and qualification.get("platform_ref")
            and qualification["snapshot_ref"] != qualification["platform_ref"]):
        raise ValueError("QUALIFIED_PLATFORM_IDENTITY_REQUIRED")
    evaluator = read_bound_json(reference)
    frozen = FrozenExecution(snapshot)
    frozen.verify_files()
    expected = _ref(frozen.root / "src/mod_agent/evaluation/strategy_repository_evaluator.py")
    if evaluator["adoption_adapter_ref"] != expected:
        raise ValueError("QUALIFIED_ADAPTER_OUTSIDE_FROZEN_PLATFORM")
    code = (
        "import json,sys; sys.path.insert(0,sys.argv[1]); "
        "from src.mod_agent.evaluation.strategy_repository_evaluator import _configuration; "
        "_configuration(json.loads(sys.argv[2]))"
    )
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(frozen.root), json.dumps(reference)],
        cwd=frozen.root, check=True, timeout=60, capture_output=True)
    frozen.verify_files()


def configuration_keys(config):
    if config.get("mode") == "strategy_pair":
        return ("intent_ref", "prepared_ref", "public_profile_ref", "qualification_ref")
    keys = ("intent_ref", "public_problem_ref", "public_contract_ref", "public_profile_ref", "qualification_ref", "settings_ref")
    if "experience_config_ref" in config:
        keys += ("experience_config_ref",)
    if "external_profile_ref" in config:
        keys += ("external_profile_ref",)
    return (*keys, "codex_executable_ref") if config.get("mode", "matched_product_pair") == "matched_product_pair" else keys


def freeze_strategy_pair(configuration_ref, config):
    """Freeze already allocated B/B+ requests; never reconstruct their argv.

    为什么要有它：经验只改变已经登记的输入边界，派发不能另造一套技术流程或刷新原任务身份。
    """
    for key in configuration_keys(config):
        _verify(config[key])
    prepared = verify_strategy_trials(config["prepared_ref"])
    qualification = read_bound_json(config["qualification_ref"])
    if qualification.get("qualified") is not True:
        raise ValueError("QUALIFIED_ACCEPTANCE_REQUIRED")
    validate_frozen_evaluator(qualification)
    if qualification.get("profile_ref") != config["public_profile_ref"]:
        raise ValueError("QUALIFIED_PUBLIC_PROFILE_CHANGED")
    intent = read_bound_json(config["intent_ref"])
    wall = config.get("wall_seconds")
    if (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0
            or intent.get("wall_seconds") != wall):
        raise ValueError("PREREGISTERED_ARM_DEADLINE_REQUIRED")
    trials = prepared["trials"]
    if (len(trials) != 2 or {t["arm"] for t in trials} != {"B", "B+"}
            or len({(t["task_id"], t["attempt"]) for t in trials}) != 1):
        raise ValueError("ONE_PREPARED_STRATEGY_PAIR_REQUIRED")
    protocol = read_bound_json(prepared["protocol_ref"])
    model = dict(protocol["invariants"]["model"])
    if model.get("name") and model.get("model") and model["name"] != model["model"]:
        raise ValueError("REGISTERED_MODEL_IDENTITY_CONFLICT")
    model["model"] = model.get("model", model.get("name"))
    if not all(isinstance(model.get(k), str) and model[k] for k in ("model", "base_url", "digest")):
        raise ValueError("REGISTERED_MODEL_IDENTITY_REQUIRED")
    profile = CSharpRepositoryProfile(config["public_profile_ref"])
    root = Path(config["root"]).resolve()
    if root.exists():
        raise ValueError("FRESH_SEPARATE_PILOT_ROOT_REQUIRED")
    rows, common_options, common_request = [], None, None
    for trial in trials:
        _, options = _configuration_arguments(trial["execution_config_ref"]["path"])
        source = Path(options["--target-project"]).resolve(strict=True)
        run_root = Path(trial["run_root"])
        arm_root = run_root.parent.parent
        if (root.is_relative_to(source) or source.is_relative_to(root)
                or root.is_relative_to(arm_root) or arm_root.is_relative_to(root)
                or run_root.exists() or (arm_root / "frozen-plan.json").exists()
                or (arm_root / "started.json").exists()):
            raise ValueError("FRESH_SEPARATE_PILOT_ROOT_REQUIRED")
        if (options.get("--repository-profile") != profile.reference["path"]
                or options.get("--ollama-model") != model["model"]
                or options.get("--ollama-base-url", "http://127.0.0.1:11434") != model["base_url"]
                or "--local-ollama" not in options):
            raise ValueError("PREPARED_MODEL_OR_PROFILE_CHANGED")
        if not {"--confirm-isolated-write", "--allow-build", "--defer-approval"} <= options.keys():
            raise ValueError("PREPARED_ISOLATED_DELIVERY_AUTHORITY_REQUIRED")
        for key, option in (("context_window", "--ollama-context-window"),
                ("output_tokens", "--ollama-max-output-tokens")):
            if key in model and str(model[key]) != options.get(option):
                raise ValueError("PREPARED_MODEL_RESOURCE_CHANGED")
        qualified_source = qualification.get("public_source_root")
        if not qualified_source or source_snapshot(source) != source_snapshot(qualified_source):
            raise ValueError("QUALIFIED_PUBLIC_SOURCE_CHANGED")
        # Allocation and experience are the only permitted arm differences.
        # 为什么要有它：两臂必须面对同一源码、请求、工具、权限和资源，不把其他配置差异计作经验收益。
        comparison = {k: v for k, v in options.items() if k not in {
            "--workspace-root", "--run-id", "--request-id", "--task-agent-experience-config", "--patch-worker-config"}}
        request = read_bound_json(trial["request_ref"])["description"]
        if common_options is not None and (comparison != common_options or request != common_request):
            raise ValueError("STRATEGY_PAIR_COMMON_CONDITIONS_CHANGED")
        common_options, common_request = comparison, request
        profile.bind(source)
        rows.append((trial, options, source, run_root, arm_root))
    identity = read_ollama_model_identity(model["base_url"], model["model"])
    if identity.get("digest") != model["digest"]:
        raise ValueError("MODEL_CHANGED")
    root.mkdir(parents=True, exist_ok=False)
    plans, common_platform_files = {}, None
    for trial, options, source, run_root, arm_root in rows:
        arm = trial["arm"]
        platform_root = root / arm / "platform"
        bound = [configuration_ref, config["prepared_ref"], config["intent_ref"],
            config["qualification_ref"], profile.reference, _ref(__file__),
            *prepared["input_refs"], *trial["generated_refs"]]
        platform = prepare_snapshot(source_root=REPO, snapshot_root=platform_root,
            business_argv=trial["argv"], run_root=run_root,
            bound_inputs=[(r["path"], r["sha256"]) for r in bound])
        # 为什么要有它：两臂的输入配置与输出位置本来不同，平台一致性应复用源码身份而不是比较绑定输入字节。
        files = platform_identity(read_bound_json(platform))["platform_files_sha256"]
        if common_platform_files is not None and files != common_platform_files:
            raise ValueError("STRATEGY_PAIR_PLATFORM_CHANGED_DURING_FREEZE")
        common_platform_files = files
        policy_id = "user_goal_disposable_strategy_pair_compile_only"
        authorization = _save(arm_root / "host-delivery-authorization.json", {
            "origin": "task_delegation_policy", "decision": "approve", "policy_id": policy_id,
            "runtime": False, "deployment": False, "run_id": trial["run_id"],
            "request_id": options["--request-id"], "isolated_root": str(arm_root / "delivery"),
            "allowed_effects": ["isolated_apply", "build"], "human_semantic_review_completed": False,
            "authorization_source": "existing user continuous engineering delegation for scoped disposable repository writes and build/test; not model output or technical review",
            "why": "为什么要有它：经验不授予权限，交付仍核验真实委托、精确候选和前像。"})
        plan = {"schema_version": "repository_product_arm_plan_v1", "arm": arm,
            "task_id": trial["task_id"], "attempt": trial["attempt"], "root": str(arm_root),
            "workspace": str(arm_root), "source_root": str(source), "source_before": source_snapshot(source),
            "common_request_ref": trial["request_ref"], "configuration_ref": configuration_ref,
            "prepared_ref": config["prepared_ref"], "wall_seconds": wall, "model_identity": identity,
            "model_endpoint": model, "allowed_edit_paths": profile.data["allowed_edit_paths"],
            "public_profile_ref": profile.reference, "controller_ref": _ref(__file__),
            "environment": config["environment"], "evaluator_ref": qualification["evaluator_ref"],
            "run_id": trial["run_id"], "run_root": str(run_root), "platform_ref": platform,
            "execution_config_ref": trial["execution_config_ref"], "authorization_ref": authorization,
            "policy_id": policy_id, "bound_inputs": bound,
            "inspection": evidence_locator(execution_config=trial["execution_config_ref"]["path"], run_id=trial["run_id"]),
            "configuration_difference": "frozen strategy snapshot only; original prepared argv retained",
            "command": [sys.executable, "-I", "-B", str(platform_root / "scripts/run_frozen_product.py"),
                "--_child", platform["path"], platform["sha256"]]}
        plans[arm] = _save(arm_root / "frozen-plan.json", plan)
    result = _save(root / "frozen-pilot.json", {"configuration_ref": configuration_ref,
        "prepared_ref": config["prepared_ref"], "order": [t["arm"] for t in trials],
        "plans": plans, "model_calls": 0, "mode": "strategy_pair",
        "scope": "internal frozen B/B+ strategy comparison; no external product superiority claim"})
    print(json.dumps({"pilot_ref": result, "arms_started": 0}))


def deliver(plan_ref):
    plan = read_bound_json(plan_ref)
    FrozenExecution(plan["platform_ref"]).verify_files()
    from src.mod_agent.reviewed_patch_continuation import continue_v2_delivery
    from src.mod_agent.task_agent_v2_delivery import (
        prepare_v2_task_delegation,
        read_completed_v2_delivery,
    )
    root = Path(plan["root"])
    result_path = Path(plan["run_root"]) / "run_result.json"
    if not result_path.is_file():
        return {"status": "native_result_missing", "delivered": False}
    native = json.loads(result_path.read_bytes())
    checkpoint = native.get("v2_delivery_checkpoint_ref")
    if not checkpoint:
        return {"status": "no_product_handoff", "delivered": False}
    destination = root / "delivery"
    destination.mkdir()
    store = str(Path(plan["run_root"]) / "a")
    prepared = continue_v2_delivery(artifact_store_root=store, checkpoint_ref=checkpoint, action="prepare",
        isolated_root=str(destination), operation_id="registered-product-pilot-delivery")
    _save(root / "delivery-preview.json", prepared)
    delegation = prepare_v2_task_delegation(artifact_store_root=store,
        displayed_binding=prepared["displayed_binding"], policy_id=plan["policy_id"],
        authorization_ref=plan["authorization_ref"], allow_build=True)
    applied = continue_v2_delivery(artifact_store_root=store, checkpoint_ref=checkpoint, action="apply",
        displayed_binding=prepared["displayed_binding"], task_delegation_ref=delegation)
    _save(root / "apply-result.json", applied)
    if applied.get("status") != "applied_in_isolation":
        return {"status": applied["status"], "delivered": False}
    built = continue_v2_delivery(artifact_store_root=store, checkpoint_ref=applied["receipt_ref"], action="build",
        operation_id="registered-product-pilot-build", allow_build=True, task_delegation_ref=delegation)
    _save(root / "build-result.json", built)
    if built.get("status") != "built":
        return {"status": built["status"], "delivered": False}
    return read_completed_v2_delivery(artifact_store_root=store, receipt_ref=built["receipt_ref"],
        native_execution_ref=_ref(Path(store) / "task-agent-execution/execution.json"))


def seal_stopped_process(plan, process_ref, *, plan_ref):
    """Seal only the known completed prefix; never resume or call a model.

    为什么要有它：外层硬停止不能跳过过程经历；缺失或歧义只保留缺口，不能猜测最新任务或重发动作。
    """
    if "platform_ref" not in plan or not plan.get("run_root") or not plan.get("run_id"):
        return {"status": "not_applicable", "model_calls": 0}
    try:
        from src.mod_agent.task_agent_v2_experience import seal_saved_public_episode
        from src.mod_agent.task_agent_v2_state import _no_links, load_v2_artifact

        if read_bound_json(plan_ref) != plan:
            raise ValueError("STOP_PLAN_CHANGED")
        process_path = Path(plan["root"]) / "execution/process.json"
        _no_links(process_path)
        if Path(process_ref["path"]).resolve(strict=True) != process_path.resolve(strict=True):
            raise ValueError("STOP_PROCESS_OUTSIDE_ARM")
        process = read_bound_json(process_ref)
        timeout = process.get("timeout") is True or process.get("status") == "timed_out"
        failed = process.get("exit_code") not in (None, 0)
        if not timeout and not failed:
            return {"status": "not_required", "model_calls": 0}
        if (process.get("started") is not True or process.get("argv") != plan["command"]
                or Path(process.get("cwd", "")).resolve(strict=True) != Path(plan["workspace"]).resolve(strict=True)):
            raise ValueError("STOP_PROCESS_EXECUTION_CHANGED")
        run_root = Path(plan["run_root"]).absolute()
        _no_links(run_root)
        if not run_root.resolve(strict=True).is_relative_to(Path(plan["root"]).resolve(strict=True)):
            raise ValueError("STOP_RUN_OUTSIDE_ARM")
        manifest_path = run_root / "run_manifest.json"
        _no_links(manifest_path)
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("run_id") != plan["run_id"]:
            raise ValueError("STOP_RUN_IDENTITY_CHANGED")
        store = run_root / "a"
        pointers_root = store / "v2/session-checkpoints"
        _no_links(pointers_root)
        pointers = list(pointers_root.glob("*.json"))
        if len(pointers) != 1:
            raise ValueError("STOP_CHECKPOINT_MISSING_OR_AMBIGUOUS")
        _no_links(pointers[0])
        checkpoint_ref = json.loads(pointers[0].read_bytes())["checkpoint_ref"]
        checkpoint = load_v2_artifact(store, checkpoint_ref)
        details = checkpoint["details"]
        if details.get("run_id") != plan["run_id"] or details.get("request_id") != manifest.get("request_id"):
            raise ValueError("STOP_CHECKPOINT_IDENTITY_CHANGED")
        stop = {"schema": "task-agent-process-stop.v1", "checkpoint_ref": checkpoint_ref,
            "task_contract_ref": checkpoint["task_contract_ref"], "run_id": plan["run_id"],
            "request_id": manifest["request_id"], "stop_kind": "timed_out" if timeout else "process_failed",
            "status": process.get("status"), "exit_code": process.get("exit_code"), "source_ref": process_ref,
            "plan_ref": plan_ref,
            "model_calls": 0, "resumed": False}
        stop_path = run_root / "process-stops" / (canonical_hash(stop) + ".json")
        _no_links(stop_path)
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with stop_path.open("x", encoding="utf-8") as stream:
                json.dump(stop, stream, ensure_ascii=False, indent=2)
        except FileExistsError:
            if json.loads(stop_path.read_bytes()) != stop:
                raise ValueError("STOP_RECORD_CHANGED") from None
        stop_ref = _ref(stop_path)
        episode_ref = seal_saved_public_episode(artifact_store_root=store,
            checkpoint_ref=checkpoint_ref, stop_ref=stop_ref)
        return {"status": "sealed", "episode_ref": episode_ref, "stop_ref": stop_ref,
            "model_calls": 0, "resumed": False}
    except Exception as error:
        return {"status": "unavailable", "error_type": type(error).__name__, "reason": str(error),
            "model_calls": 0, "resumed": False}


def run(pilot_ref, arm):
    pilot = read_bound_json(pilot_ref)
    plan_ref = pilot["plans"][arm]
    plan = read_bound_json(plan_ref)
    _verify(plan["configuration_ref"])
    _verify(plan["controller_ref"])
    config = read_bound_json(plan["configuration_ref"])
    for key in configuration_keys(config):
        _verify(config[key])
    root = Path(plan["root"])
    if (root / "started.json").exists() or source_snapshot(plan["source_root"]) != plan["source_before"]:
        raise ValueError("ALREADY_STARTED_OR_SOURCE_CHANGED")
    for previous in pilot["order"][:pilot["order"].index(arm)]:
        if not (Path(pilot["plans"][previous]["path"]).parent / "result.json").is_file():
            raise ValueError("SERIAL_ARM_ORDER_REQUIRED")
    if pilot.get("mode") == "strategy_pair":
        verify_strategy_trials(plan["prepared_ref"])
    if "platform_ref" in plan:
        FrozenExecution(plan["platform_ref"]).verify_files()
        for reference in plan["bound_inputs"]:
            _verify(reference)
    external = external_pilot_profile(config)
    if external is not None:
        if plan.get("external_profile_ref") != config["external_profile_ref"]:
            raise ValueError("PILOT_EXTERNAL_PROFILE_BINDING_CHANGED")
        external_pilot_identity(external, plan["model_identity"])
    else:
        if "external_profile_ref" in plan:
            raise ValueError("PILOT_EXTERNAL_PROFILE_BINDING_CHANGED")
        model = plan.get("model_endpoint", {"base_url": "http://127.0.0.1:11434", "model": MODEL, "digest": DIGEST})
        if read_ollama_model_identity(model["base_url"], model["model"])["digest"] != model["digest"]:
            raise ValueError("MODEL_CHANGED")
    clock, utc = time.monotonic(), time.time()
    deadline, deadline_utc = clock + plan["wall_seconds"], utc + plan["wall_seconds"]
    for inherited in (os.environ.get("REPOFORGE_TASK_DEADLINE_UTC"),
            plan["environment"].get("REPOFORGE_TASK_DEADLINE_UTC")):
        if inherited is not None:
            value = float(inherited)
            if not math.isfinite(value):
                raise ValueError("INHERITED_DEADLINE_INVALID")
            deadline_utc = min(deadline_utc, value)
    deadline = clock + (deadline_utc - utc)
    if deadline <= clock:
        raise ValueError("INHERITED_DEADLINE_EXHAUSTED")
    started = {"plan_ref": plan_ref, "deadline_utc": deadline_utc,
        **({"run_id": plan["run_id"]} if "run_id" in plan else {}),
        "human_active_seconds": None, "technical_hint_or_patch": False, "scope": "one registered attempt; no automatic retry"}
    # 为什么要有它：同一臂的并发派发也不能通过先检查后覆盖的竞态获得第二次执行。
    with (root / "started.json").open("x", encoding="utf-8") as stream:
        json.dump(started, stream, ensure_ascii=False, indent=2)
    environment = {**os.environ, **plan["environment"], "PYTHONUTF8": "1", "DOTNET_CLI_UI_LANGUAGE": "en-US",
                   "REPOFORGE_TASK_DEADLINE_UTC": str(deadline_utc)}
    process = execute(plan["command"], cwd=plan["workspace"], stdin_path=plan.get("prompt_path"),
        output=root / "execution", deadline=deadline, environment=environment)
    delivery_process = None
    if "platform_ref" in plan and not process.get("timeout") and deadline > time.monotonic():
        delivery_process = execute([sys.executable, "-B", str(Path(plan["platform_ref"]["path"]).parent /
            "scripts/run_repository_product_pilot.py"), "deliver", "--reference", plan_ref["path"], "--sha256", plan_ref["sha256"]],
            cwd=plan["workspace"], stdin_path=None, output=root / "delivery-process", deadline=deadline, environment=environment)
    after = source_snapshot(plan["source_root"])
    changed = sorted(path for path in after.keys() | plan["source_before"].keys() if after.get(path) != plan["source_before"].get(path))
    forbidden = [path for path in changed if path not in plan["allowed_edit_paths"]]
    process_ref = _ref(root / "execution/process.json")
    facts = {"plan_ref": plan_ref, "process_ref": process_ref,
        "delivery_process": delivery_process, "changed_source_paths": changed, "forbidden_source_changes": forbidden,
        "source_after": after, "task_wall_seconds": time.monotonic() - clock,
        "independent_business_result": None, "correct_delivery": None,
        "scope": "execution facts only; native claims do not prove independent acceptance or same-version delivery"}
    if arm == ORDER[0] and process.get("started"):
        facts["native_events_ref"] = _save(root / "native-events.json", codex_events(process["stdout_ref"]["path"]))
    # 为什么要有它：先保存已发生的进程与源码事实；经历封存失败不能吞掉本次执行结果。
    facts["execution_facts_ref"] = _save(root / "execution-facts.json", facts)
    facts["process_episode"] = seal_stopped_process(plan, process_ref, plan_ref=plan_ref)
    print(json.dumps({"result_ref": _save(root / "result.json", facts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("freeze", "run", "deliver"))
    parser.add_argument("--reference", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--arm", choices=(*ORDER, "B", "B+"))
    args = parser.parse_args()
    reference = {"path": args.reference, "sha256": args.sha256}
    if args.mode == "freeze":
        freeze(reference)
    elif args.mode == "run":
        if args.arm is None:
            parser.error("--arm required for run")
        run(reference, args.arm)
    else:
        result = deliver(reference)
        plan = read_bound_json(reference)
        print(json.dumps({"delivery_ref": _save(Path(plan["root"]) / "delivery-result.json", result)}))


if __name__ == "__main__":
    main()
