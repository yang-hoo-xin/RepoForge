"""Bind and run a registered repository product pilot using existing owners.

No solver actions, repair hints or acceptance answers are selected here.
为什么要有它：产品对照需要同一公开任务与真实交付，控制脚本不能成为外层解题器。
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

import yaml

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dsh_run_request import (
    _configuration_arguments,
    build_product_argv,
    evidence_locator,
    prepare_new_task,
)
from scripts.run_benchmark_arm import codex_argv, codex_events, execute
from src.mod_agent.domain.csharp_repository_backend import (
    _IGNORED,
    CSharpRepositoryProfile,
    _relative,
    _snapshot,
)
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_admission import read_ollama_model_identity
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.platform_execution_snapshot import FrozenExecution, prepare_snapshot

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


def freeze(configuration_ref):
    config = read_bound_json(configuration_ref)
    mode = config.get("mode", "matched_product_pair")
    if mode not in {"matched_product_pair", "absolute_real_B"}:
        raise ValueError("EXPLICIT_PILOT_MODE_REQUIRED")
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
    identity = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    if identity["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
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
        if arm == "Codex_local":
            deny = [REPO, Path.home() / ".codex", Path.home() / ".ssh", Path.home() / ".aws"]
            base["command"] = local_codex_command(config["codex_executable_ref"]["path"], workspace,
                arm_root / "last-message.txt", deny, profile.data["dotnet_ref"]["path"], profile.data["allowed_edit_paths"])
            base["prompt_path"] = str(common_path)
            base["configuration_difference"] = "native OSS adapter; probe context4096, sampling/output options not independently captured; final wire not instrumented"
        else:
            settings = yaml.safe_load(Path(config["settings_ref"]["path"]).read_text(encoding="utf-8"))
            settings["task_memory"] = {"enabled": False, "store_root": None, "max_context_chars": 3000}
            settings["llm"].update(provider="ollama", model=MODEL, base_url="http://127.0.0.1:11434", max_tokens=4096)
            settings_path = arm_root / "settings.yaml"
            with settings_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(settings, stream, allow_unicode=True, sort_keys=False)
            arguments = ["--target-project", str(source), "--workspace-root", str(arm_root / "runs"),
                "--settings", str(settings_path), "--repository-profile", profile.reference["path"],
                "--local-ollama", "--ollama-model", MODEL, "--ollama-base-url", "http://127.0.0.1:11434",
                "--ollama-context-window", "65536", "--ollama-max-output-tokens", "4096",
                "--ollama-timeout-seconds", "600", "--model-call-budget-policy", "meter_only",
                "--source-retrieval", "rg_live", "--confirm-isolated-write", "--allow-build", "--defer-approval",
                "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
                "--constraint-policy", "source_priority_v1", "--task-agent-edit-paths", json.dumps(profile.data["allowed_edit_paths"])]
            execution = _save(arm_root / "base-execution.json", {"arguments": arguments})
            selected = prepare_new_task(execution_config=execution["path"], description=common)
            business = build_product_argv(execution_config=selected, description=common)
            _, options = _configuration_arguments(selected)
            run_root = arm_root / "runs" / options["--run-id"]
            bound = [_ref(selected), _ref(settings_path), profile.reference, common_ref, configuration_ref, _ref(__file__)]
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
                bound_inputs=bound, configuration_difference="explicit context65536/output4096/temperature0/seed0; no injected strategy or terminal extraction configured",
                command=[sys.executable, "-I", "-B", str(root / "platform/scripts/run_frozen_product.py"),
                    "--_child", platform["path"], platform["sha256"]])
        plans[arm] = _save(arm_root / "frozen-plan.json", base)
    result = _save(root / "frozen-pilot.json", {"configuration_ref": configuration_ref, "common_request_ref": common_ref,
        "order": list(arms), "plans": plans, "model_calls": 0, "mode": mode,
        "scope": "whole product configurations" if mode == "matched_product_pair" else "absolute real B result; product comparison not covered"})
    print(json.dumps({"pilot_ref": result, "arms_started": 0}))


def _verify(reference):
    if _ref(reference["path"]) != reference:
        raise ValueError("BOUND_INPUT_CHANGED")


def validate_frozen_evaluator(qualification):
    """Consume qualification with its original adapter, not the mutable checkout.

    为什么要有它：资格绑定的是实际执行版本及路径，消费端不能改写原件或放宽身份检查。
    """
    reference = qualification["evaluator_ref"]
    evaluator = read_bound_json(reference)
    frozen = FrozenExecution(qualification["snapshot_ref"])
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
    keys = ("intent_ref", "public_problem_ref", "public_contract_ref", "public_profile_ref", "qualification_ref", "settings_ref")
    return (*keys, "codex_executable_ref") if config.get("mode", "matched_product_pair") == "matched_product_pair" else keys


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
    if (arm == ORDER[1] and ORDER[0] in pilot["plans"]
            and not (Path(pilot["plans"][ORDER[0]]["path"]).parent / "result.json").is_file()):
        raise ValueError("SERIAL_ARM_ORDER_REQUIRED")
    if read_ollama_model_identity("http://127.0.0.1:11434", MODEL)["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    if "platform_ref" in plan:
        FrozenExecution(plan["platform_ref"]).verify_files()
        for reference in plan["bound_inputs"]:
            _verify(reference)
    clock, utc = time.monotonic(), time.time()
    deadline, deadline_utc = clock + plan["wall_seconds"], utc + plan["wall_seconds"]
    _save(root / "started.json", {"plan_ref": plan_ref, "deadline_utc": deadline_utc,
        **({"run_id": plan["run_id"]} if "run_id" in plan else {}),
        "human_active_seconds": None, "technical_hint_or_patch": False, "scope": "one registered attempt; no automatic retry"})
    environment = {**os.environ, **plan["environment"], "PYTHONUTF8": "1", "DOTNET_CLI_UI_LANGUAGE": "en-US",
                   "REPOFORGE_TASK_DEADLINE_UTC": str(deadline_utc)}
    process = execute(plan["command"], cwd=plan["workspace"], stdin_path=plan.get("prompt_path"),
        output=root / "execution", deadline=deadline, environment=environment)
    delivery_process = None
    if arm == ORDER[1] and not process.get("timeout") and deadline > time.monotonic():
        delivery_process = execute([sys.executable, "-B", str(Path(plan["platform_ref"]["path"]).parent /
            "scripts/run_repository_product_pilot.py"), "deliver", "--reference", plan_ref["path"], "--sha256", plan_ref["sha256"]],
            cwd=plan["workspace"], stdin_path=None, output=root / "delivery-process", deadline=deadline, environment=environment)
    after = source_snapshot(plan["source_root"])
    changed = sorted(path for path in after.keys() | plan["source_before"].keys() if after.get(path) != plan["source_before"].get(path))
    forbidden = [path for path in changed if path not in plan["allowed_edit_paths"]]
    facts = {"plan_ref": plan_ref, "process_ref": _ref(root / "execution/process.json"),
        "delivery_process": delivery_process, "changed_source_paths": changed, "forbidden_source_changes": forbidden,
        "source_after": after, "task_wall_seconds": time.monotonic() - clock,
        "independent_business_result": None, "correct_delivery": None,
        "scope": "execution facts only; native claims do not prove independent acceptance or same-version delivery"}
    if arm == ORDER[0] and process.get("started"):
        facts["native_events_ref"] = _save(root / "native-events.json", codex_events(process["stdout_ref"]["path"]))
    print(json.dumps({"result_ref": _save(root / "result.json", facts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("freeze", "run", "deliver"))
    parser.add_argument("--reference", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--arm", choices=ORDER)
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
