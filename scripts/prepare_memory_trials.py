"""Freeze the already preselected memory transfer pairs; run only an explicit arm."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_reload_trials import ref, same_source_snapshot, save
from src.mod_agent.build.dependencies import _binary_tree_snapshot, _binary_tree_snapshot_sha256
from src.mod_agent.domain.full_project_compilation import _source_snapshot
from src.mod_agent.evaluation.memory_comparison import (
    digest,
    prepare_memory_pair,
    verify_memory_pair,
)
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.platform_execution_snapshot import FrozenExecution, prepare_snapshot


def _verify(reference):
    if ref(reference["path"]) != reference:
        raise ValueError("MEMORY_TRANSFER_BOUND_INPUT_CHANGED")


def prepare(*, comparison_path, worker_template_path, output_root):
    comparison_ref = ref(comparison_path)
    comparison = read_bound_json(comparison_ref)
    selection = read_bound_json(comparison["preselection_ref"])
    candidate = read_bound_json(comparison["candidate_ref"])
    pairs = [read_bound_json(reference) for reference in comparison["pairs"]]
    for reference in comparison["pairs"]:
        verify_memory_pair(reference)
    for reference in selection["base_refs"].values():
        _verify(reference)
    if selection["repetitions_per_task_arm"] != 1 or len(pairs) != 2:
        raise ValueError("MEMORY_TRANSFER_PRESELECTION_CHANGED")
    for pair in pairs:
        invariants = pair["invariants"]
        task = next(row for row in selection["tasks"] if row["task_id"] == invariants["task"]["task_id"])
        if (invariants["task"]["task_sha256"] != digest(task)
                or invariants["budget"] != selection["budget"] or invariants["permissions"] != selection["permissions"]
                or invariants["model"]["digest"] != candidate["model_identity"]["digest"]
                or invariants["initial_environment"]["source_ref"] != selection["base_refs"]["source"]
                or invariants["tools"]["knowledge_augmentation"] is not False
                or invariants["tools"]["dependency_metadata_augmentation"] is not False):
            raise ValueError("MEMORY_TRANSFER_PAIR_INVARIANT_CHANGED")
    source = Path(selection["base_refs"]["source"]["path"]).parent
    source_rows = _source_snapshot(source)
    worker_template = read_bound_json(ref(worker_template_path))
    dependency = worker_template["behavior_evaluator"]["kwargs"]["compilation_kwargs"]["dependency_binding"]
    tree = _binary_tree_snapshot(Path(dependency["source_root"]))
    if (_binary_tree_snapshot_sha256(tree) != dependency["snapshot_sha256"]
            or dependency["snapshot_sha256"] != candidate["dependency_snapshot_sha256"]
            or len(tree) != dependency["file_count"]):
        raise ValueError("MEMORY_TRANSFER_DEPENDENCY_CHANGED")
    matching = [row for row in tree if row["path"] == candidate["dependency_relative_path"]]
    if len(matching) != 1 or matching[0]["sha256"] != candidate["dependency_ref"]["sha256"]:
        raise ValueError("MEMORY_TRANSFER_LESSON_DEPENDENCY_NOT_APPLICABLE")
    probe_ref = comparison["judge"]["probe_configuration_ref"]
    probe = read_bound_json(probe_ref)
    for reference in [*probe["files"], probe["dotnet"]]:
        _verify(reference)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    copied = root / "i" / "StartupNotice"
    for row in source_rows:
        destination = copied / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / row["path"]).read_bytes())
    if not same_source_snapshot(_source_snapshot(copied), source_rows):
        raise ValueError("MEMORY_TRANSFER_SOURCE_COPY_CHANGED")
    dependency_ref = save(root / "dependency.json", dependency)
    project = ET.parse(copied / "Mod.csproj").getroot()
    framework, assembly = project.findtext(".//TargetFramework"), project.findtext(".//AssemblyName")
    if not framework or not assembly:
        raise ValueError("MEMORY_TRANSFER_PROJECT_OUTPUT_UNKNOWN")
    platform = root / "p"
    platform_ref = prepare_snapshot(source_root=Path(__file__).resolve().parents[1], snapshot_root=platform,
                                    business_argv=["--help"])
    rows, new_pairs, revisions = [], [], []
    for number, (old_pair_ref, old) in enumerate(zip(comparison["pairs"], pairs), 1):
        task_id = old["invariants"]["task"]["task_id"]
        task = next(task for task in selection["tasks"] if task["task_id"] == task_id)
        invariants = deepcopy(old["invariants"])
        for key, relative in (("normal_owner_ref", "src/mod_agent/local_entry.py"),
                              ("planning_owner_ref", "src/mod_agent/planning/model_selection.py")):
            previous = invariants["tools"][key]
            invariants["tools"][key] = ref(platform / relative)
            revisions.append({"role": key, "previous_ref": previous, "new_ref": invariants["tools"][key],
                              "bytes_changed": previous["sha256"] != invariants["tools"][key]["sha256"]})
        invariants["tools"]["platform_snapshot_ref"] = platform_ref
        invariants["judge"]["factory_ref"] = ref(platform / "src/mod_agent/domain/logging_transfer_evaluator.py")
        pair_ref = prepare_memory_pair(settings_path=old["base_settings_ref"]["path"],
            memory_root=old["arms"][0]["memory_root"], invariants=invariants, output_root=root / str(number))
        pair = read_bound_json(pair_ref)
        if pair["snapshot_files"] != old["snapshot_files"]:
            raise ValueError("MEMORY_TRANSFER_LESSON_SNAPSHOT_CHANGED")
        new_pairs.append(pair_ref)
        request_ref = save(root / str(number) / "task.json", task)
        for arm in pair["arms"]:
            base = Path(arm["receipts_root"]).parent
            worker = deepcopy(worker_template)
            selected = worker["behavior_evaluator"]
            selected["factory_ref"] = invariants["judge"]["factory_ref"]
            selected["factory_name"] = "make_logging_transfer_evaluator"
            compilation = selected["kwargs"]["compilation_kwargs"]
            compilation.update(project_relative_path="Mod.csproj", expected_artifact=f"bin/Debug/{framework}/{assembly}.dll",
                source_root="allocated-by-existing-bridge", workspace_root="allocated-by-existing-bridge",
                artifact_store_root="allocated-by-existing-bridge", output_root="allocated-by-existing-bridge")
            selected["kwargs"] = {"compilation_kwargs": compilation, "output_root": "allocated-by-existing-bridge",
                "probe_configuration_ref": probe_ref, "preselection_ref": comparison["preselection_ref"], "task_id": task_id}
            worker["policy"]["max_total_model_turns"] = selection["budget"]["product_model_calls"]
            worker_ref = save(base / "worker.json", worker)
            budget, model = selection["budget"], invariants["model"]
            arguments = ["--target-project", str(copied), "--workspace-root", arm["workspace_root"],
                "--settings", arm["settings_ref"]["path"], "--game-folder-build-binding", dependency_ref["path"],
                "--local-ollama", "--ollama-model", model["model"], "--ollama-base-url", model["base_url"],
                "--ollama-context-window", str(budget["context_window"]), "--ollama-max-output-tokens", str(budget["output_tokens"]),
                "--ollama-timeout-seconds", str(budget["provider_timeout_seconds"]),
                "--task-model-call-limit", str(budget["product_model_calls"]), "--model-call-budget-policy", "task_total_only",
                "--patch-worker-config", worker_ref["path"], "--source-retrieval", "rg_live",
                "--confirm-isolated-write", "--allow-build", "--defer-approval"]
            bindings = [{"path": ["behavior_evaluator", "kwargs", "compilation_kwargs", key], "relative_path": value}
                for key, value in (("source_root", "generated_mods/StartupNotice"), ("workspace_root", "."),
                                   ("artifact_store_root", "a"), ("output_root", "c"))]
            bindings.append({"path": ["behavior_evaluator", "kwargs", "output_root"], "relative_path": "v"})
            profile_ref = save(base / "profile.json", {"arguments": arguments, "worker_path_bindings": bindings})
            rows.append({**arm, "trial_id": f"{number}-{arm['arm']}", "task_id": task_id, "root": str(base),
                "task_ref": request_ref, "pair_ref": pair_ref, "original_pair_ref": old_pair_ref,
                "profile_ref": profile_ref, "worker_ref": worker_ref, "product_snapshot_root": str(base / "p")})
    return save(root / "ready.json", {"schema_version": "memory_transfer_dispatch_v1", "status": "prepared_not_executed",
        "root": str(root), "comparison_ref": comparison_ref, "preselection_ref": comparison["preselection_ref"],
        "candidate_ref": comparison["candidate_ref"], "source_root": str(copied), "source_snapshot": source_rows,
        "dependency_ref": dependency_ref, "platform_root": str(platform), "platform_snapshot_ref": platform_ref,
        "worker_template_ref": ref(worker_template_path), "source_revision_changes": revisions,
        "pairs": new_pairs, "trials": rows, "model_calls": 0, "builds": 0, "applies": 0,
        "order": [row["trial_id"] for row in rows], "memory_mutation_during_pair": False,
        "adoption_decision": "pending_preselected_actual_delivery_misleading_use_cost_comparison"})


def _executor(trial, prepared):
    from scripts.dsh_run_request import _configuration_arguments
    from src.mod_agent.cli import build_parser
    original, _ = _configuration_arguments(trial["profile_ref"]["path"])
    profile = read_bound_json(trial["profile_ref"])
    task = read_bound_json(trial["task_ref"])
    expected = build_parser().parse_args(["run-request", *original, "--", task["request"]])

    def execute(argv):
        common = FrozenExecution(prepared["platform_snapshot_ref"])
        common.verify_files()
        common.verify_loaded_modules()
        actual = build_parser().parse_args(argv)
        ignored = {"run_id", "request_id", "patch_worker_config"}
        if {k: v for k, v in vars(actual).items() if k not in ignored} != {
                k: v for k, v in vars(expected).items() if k not in ignored}:
            raise ValueError("MEMORY_TRANSFER_DISPATCH_ARGUMENT_CHANGED")
        if (not actual.run_id or re.fullmatch(r"dsh_[0-9a-f]{32}", actual.run_id) is None
                or actual.request_id != "request_" + actual.run_id.removeprefix("dsh_")):
            raise ValueError("MEMORY_TRANSFER_DISPATCH_IDENTITY_CHANGED")
        run_root = (Path(trial["workspace_root"]) / actual.run_id).resolve()
        worker_path = Path(actual.patch_worker_config).resolve(strict=True)
        if worker_path.parent != Path(trial["workspace_root"]) / ".dsh-inputs" / actual.run_id:
            raise ValueError("MEMORY_TRANSFER_WORKER_ALLOCATION_CHANGED")
        worker = read_bound_json(trial["worker_ref"])
        for binding in profile["worker_path_bindings"]:
            current = worker
            for key in binding["path"][:-1]:
                current = current[key]
            current[binding["path"][-1]] = str((run_root / binding["relative_path"]).resolve())
        if json.loads(worker_path.read_bytes()) != worker:
            raise ValueError("MEMORY_TRANSFER_WORKER_CHANGED")
        verify_memory_pair(trial["pair_ref"])
        inputs = [ref(worker_path), trial["settings_ref"], prepared["dependency_ref"], trial["task_ref"],
                  prepared["preselection_ref"], trial["pair_ref"]]
        pair = read_bound_json(trial["pair_ref"])
        inputs.extend(ref(Path(trial["memory_root"]) / row["path"]) for row in pair["snapshot_files"])
        factory = worker["behavior_evaluator"]["factory_ref"]
        snapshot = prepare_snapshot(source_root=prepared["platform_root"], snapshot_root=trial["product_snapshot_root"],
            business_argv=argv, run_root=run_root, extra_load_refs=[(factory["path"], factory["sha256"])],
            bound_inputs=[(row["path"], row["sha256"]) for row in inputs])
        save(Path(trial["root"]) / "dispatch.json", {"argv": argv, "run_root": str(run_root), "snapshot_ref": snapshot,
            "bound_inputs": inputs, "extra_load_refs": [factory], "allocated_worker_ref": ref(worker_path)})
        command = [sys.executable, "-I", "-B", str(Path(trial["product_snapshot_root"]) / "scripts/run_frozen_product.py"),
                   "--_child", snapshot["path"], snapshot["sha256"]]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
        save(Path(trial["root"]) / "product-process.json", {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr})
        print(result.stdout, end="")
        return result.returncode
    return execute


def run(*, ready_path, trial_id):
    from scripts.dsh_request_routing import run_natural_request
    from src.mod_agent.local_entry import _read_ollama_model_identity
    prepared = read_bound_json(ref(ready_path))
    if Path(__file__).resolve().parents[1] != Path(prepared["platform_root"]):
        raise ValueError("MEMORY_TRANSFER_REQUIRES_FROZEN_PLATFORM")
    frozen = FrozenExecution(prepared["platform_snapshot_ref"])
    frozen.verify_files()
    frozen.verify_loaded_modules()
    trial = next(row for row in prepared["trials"] if row["trial_id"] == trial_id)
    output = Path(trial["root"]) / "natural-result.json"
    if output.exists():
        return ref(output)
    # 为什么要有它：同一臂首次运行才是配对样本；未知中断不能自动重发并刷新本题模型预算。
    if (Path(trial["root"]) / "started.json").exists():
        raise ValueError("MEMORY_TRANSFER_PRIOR_ATTEMPT_REQUIRES_INSPECTION")
    for previous in prepared["trials"][:prepared["trials"].index(trial)]:
        if not (Path(previous["root"]) / "natural-result.json").exists():
            raise ValueError("MEMORY_TRANSFER_PREDECLARED_ORDER_REQUIRED")
    verify_memory_pair(trial["pair_ref"])
    for reference in (trial["profile_ref"], trial["worker_ref"], trial["task_ref"], prepared["dependency_ref"]):
        _verify(reference)
    if not same_source_snapshot(_source_snapshot(Path(prepared["source_root"])), prepared["source_snapshot"]):
        raise ValueError("MEMORY_TRANSFER_SOURCE_CHANGED")
    expected = read_bound_json(trial["pair_ref"])["invariants"]["model"]
    actual = _read_ollama_model_identity(expected["base_url"], expected["model"])
    if actual.get("digest") != expected["digest"]:
        raise ValueError("MEMORY_TRANSFER_MODEL_CHANGED")
    save(Path(trial["root"]) / "started.json", {"ready_ref": ref(ready_path), "trial_id": trial_id, "model_identity": actual})
    result = run_natural_request(execution_config=trial["profile_ref"]["path"],
        description=read_bound_json(trial["task_ref"])["request"], conversation_id="memory-" + trial_id,
        turn_id="initial", product_executor=_executor(trial, prepared))
    result_ref = save(output, result)
    save(Path(trial["root"]) / "memory-unchanged.json", verify_memory_pair(trial["pair_ref"]))
    return result_ref


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("comparison-path", "worker-template-path", "output-root"):
        prepare_parser.add_argument("--" + name, required=True)
    run_parser = sub.add_parser("run")
    for name in ("ready-path", "trial-id"):
        run_parser.add_argument("--" + name, required=True)
    v2 = sub.add_parser("prepare-v2", help="Prepare registered v2 B/B+ inputs without running models.")
    v2.add_argument("--input-record", required=True)
    v2.add_argument("--output-root", required=True)
    check_v2 = sub.add_parser("verify-v2", help="Recheck saved v2 inputs and live strategy withdrawal state.")
    check_v2.add_argument("--prepared-record", required=True)
    args = vars(parser.parse_args(argv))
    mode = args.pop("mode")
    if mode in {"prepare-v2", "verify-v2"}:
        from src.mod_agent.evaluation.strategy_trial_preparation import (
            prepare_strategy_trials,
            verify_strategy_trials,
        )

        # 为什么要有它：新协议复用既有准备入口，准备与核对不能暗中派发求解或重启旧尝试。
        try:
            result = (prepare_strategy_trials(input_ref=ref(args["input_record"]), output_root=args["output_root"])
                if mode == "prepare-v2" else verify_strategy_trials(prepared_ref=ref(args["prepared_record"])))
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=True))
            return 2
        print(json.dumps(result, ensure_ascii=True))
        return 0
    print(json.dumps(prepare(**args) if mode == "prepare" else run(**args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
