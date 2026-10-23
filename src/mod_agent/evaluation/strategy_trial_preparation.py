"""Prepare paired public v2 inputs, without executing or claiming a frozen model.

为什么要有它：同一任务的经验比较需要可核验的实际入口输入，而不是另一套执行器。
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import yaml

from scripts.dsh_run_request import (
    _configuration_arguments,
    build_product_argv,
    prepare_new_task,
    validate_product_profile,
)
from src.core.settings import TaskMemorySettings
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.evaluation.independent_preparation import file_ref, write_once
from src.mod_agent.evaluation.memory_comparison import INVARIANTS
from src.mod_agent.evaluation.strategy_adoption import _planned_pairs
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.planning.strategy_experience import StrategyExperience
from src.mod_agent.task_agent_v2_state import _no_links, _text_hash


def _require(condition, code):
    if not condition:
        raise ValueError("STRATEGY_PREPARATION_" + code)


def _source_snapshot(root, *, include_binary=False):
    root = Path(root)
    _no_links(root.absolute())
    root = root.resolve(strict=True)
    files = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if any(part.casefold() in {".git", ".env", "bin", "obj", ".venv", "__pycache__",
                ".aws", ".ssh", ".codex", "credentials", ".idea", ".vs", "testresults"}
                or part.casefold().startswith(".env.") for part in relative.parts):
            continue
        _no_links(item)
        if item.is_file():
            raw = item.read_bytes()
            if include_binary:
                files[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            files[relative.as_posix()] = _text_hash(text)
    return files


def _refs(value):
    result = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and Path(value["path"]).is_absolute() and "sha256" in value:
            _require(file_ref(value["path"])["sha256"] == value["sha256"], "REFERENCE_CHANGED")
            result.append(file_ref(value["path"]))
        for child in value.values():
            result.extend(_refs(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_refs(child))
    return result


def _validate(input_ref):
    document = read_bound_json(input_ref)
    _require(set(document) == {"schema_version", "protocol_ref", "store_root", "tasks", "order", "extract_on_terminal"}
        and document["schema_version"] == "strategy_trial_preparation_input_v1"
        and type(document["extract_on_terminal"]) is bool, "INPUT_INVALID")
    protocol = read_bound_json(document["protocol_ref"])
    _require(protocol.get("schema_version") == "strategy_trial_protocol_v1", "PROTOCOL_INVALID")
    invariants = protocol.get("invariants")
    _require(isinstance(invariants, dict) and set(invariants) >= INVARIANTS | {"feedback"}
        and all(isinstance(v, dict) and v for v in invariants.values())
        and canonical_hash(invariants) == protocol.get("invariants_sha256"), "INVARIANTS_INVALID")
    planned = _planned_pairs(protocol)
    store = Path(document["store_root"])
    _require(store.is_absolute() and store.is_dir(), "STORE_REQUIRED")
    _no_links(store)
    StrategyExperience(ArtifactStore(store)).check_snapshot(protocol["snapshot_ref"])
    _require(isinstance(document["tasks"], list) and isinstance(document["order"], list), "TASKS_INVALID")
    tasks, snapshots, refs = {}, {}, [input_ref, document["protocol_ref"], *_refs(protocol)]
    for task in document["tasks"]:
        _require(set(task) == {"task_id", "attempt", "execution_config_ref", "request_ref"}, "TASK_INVALID")
        key = (task["task_id"], task["attempt"])
        _require(type(task["attempt"]) is int and key in planned and key not in tasks, "TASK_NOT_PLANNED_OR_DUPLICATE")
        request = read_bound_json(task["request_ref"])
        _require(set(request) == {"schema_version", "visibility", "description"}
            and request["schema_version"] == "public_development_request_v1"
            and request["visibility"] == "public", "PUBLIC_REQUEST_REQUIRED")
        config = read_bound_json(task["execution_config_ref"])
        _, options = _configuration_arguments(task["execution_config_ref"]["path"])
        validate_product_profile(execution_config=task["execution_config_ref"]["path"], description=request["description"])
        _require(options.get("--technical-executor") == "task_agent" and options.get("--task-agent-protocol") == "v2"
            and options.get("--repository-profile"), "V2_REPOSITORY_REQUIRED")
        _require(not options.get("--task-agent-experience-config")
            and not any(k in config for k in ("new_project_binding", "followup_binding")), "BASE_CONFIG_REQUIRED")
        worker = read_bound_json(file_ref(options["--patch-worker-config"])) if options.get("--patch-worker-config") else None
        bindings = config.get("worker_path_bindings", [])
        _require(isinstance(bindings, list) and (not bindings or worker is not None)
            and not (worker and worker.get("behavior_evaluator") is not None and not bindings), "WORKER_BINDINGS_REQUIRED")
        locators = set()
        for item in bindings:
            _require(isinstance(item, dict) and set(item) == {"path", "relative_path"}, "WORKER_BINDING_INVALID")
            locator, relative = item["path"], item["relative_path"]
            _require(isinstance(locator, list) and len(locator) >= 3
                and all(isinstance(k, str) and k for k in locator)
                and locator[:2] == ["behavior_evaluator", "kwargs"] and tuple(locator) not in locators
                and isinstance(relative, str) and relative and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts, "WORKER_BINDING_INVALID")
            locators.add(tuple(locator))
            parent = worker
            for locator_key in locator:
                _require(isinstance(parent, dict) and locator_key in parent, "WORKER_BINDING_MISSING")
                parent = parent[locator_key]
            _require(isinstance(parent, str), "WORKER_BINDING_INVALID")
        settings = yaml.safe_load(Path(options["--settings"]).read_text(encoding="utf-8"))
        _require(isinstance(settings, dict) and not TaskMemorySettings(**settings.get("task_memory", {})).enabled,
            "PLANNING_MEMORY_MUST_BE_OFF")
        source = Path(options["--target-project"]).resolve(strict=True)
        snapshot = _source_snapshot(source)
        source_record = read_bound_json(planned[key]["source_ref"])
        _require(source_record.get("source_files_sha256") == canonical_hash(snapshot)
            and source_record.get("original_request_sha256") == canonical_hash(request["description"]), "SOURCE_OR_REQUEST_CHANGED")
        snapshots[str(source)] = snapshot
        refs.extend([task["execution_config_ref"], task["request_ref"], planned[key]["source_ref"], planned[key]["exposure_ref"]])
        for option, path in options.items():
            if path and Path(path).is_absolute() and Path(path).is_file():
                refs.append(file_ref(path))
                if Path(path).suffix.lower() == ".json":
                    refs.extend(_refs(read_bound_json(file_ref(path))))
        tasks[key] = (task, config, options, request)
    _require(set(tasks) == set(planned), "EXACT_PLANNED_TASKS_REQUIRED")
    order = []
    for row in document["order"]:
        _require(set(row) == {"task_id", "attempt", "arm"} and type(row["attempt"]) is int, "ORDER_INVALID")
        order.append((row["task_id"], row["attempt"], row["arm"]))
    expected = {(task, attempt, arm) for task, attempt in planned for arm in ("B", "B+")}
    _require(len(order) == len(expected) and set(order) == expected, "EXACT_ARM_ORDER_REQUIRED")
    return document, protocol, tasks, snapshots, refs


def prepare_strategy_trials(input_ref, output_root):
    document, protocol, tasks, snapshots, refs = _validate(input_ref)
    root = Path(output_root)
    _require(root.is_absolute(), "ABSOLUTE_OUTPUT_REQUIRED")
    _no_links(root)
    root = root.resolve()
    for protected in [Path(document["store_root"]).resolve(), *(Path(p) for p in snapshots)]:
        _require(not root.is_relative_to(protected) and not protected.is_relative_to(root), "OUTPUT_OVERLAP")
    _require(not root.exists(), "OUTPUT_EXISTS")
    byte_snapshots = {p: _source_snapshot(p, include_binary=True) for p in snapshots}
    root.mkdir(parents=True, exist_ok=False)
    trials = []
    for index, row in enumerate(document["order"]):
        task, config, options, request = tasks[(row["task_id"], row["attempt"])]
        folder = root / str(index + 1)
        binding = write_once(folder / "trial-binding.json", {"schema_version": "strategy_trial_run_binding_v1",
            "protocol_ref": document["protocol_ref"], **row})
        experience = write_once(folder / "experience.json", {"schema_version": "task_agent_experience_config_v1",
            "store_root": document["store_root"], "snapshot_ref": protocol["snapshot_ref"] if row["arm"] == "B+" else None,
            "extract_on_terminal": document["extract_on_terminal"], "trial_binding_ref": binding})
        selected = deepcopy(options)
        selected.pop("--run-id", None)
        selected.pop("--request-id", None)
        selected.update({"--workspace-root": str(folder / "runs"), "--task-agent-experience-config": experience["path"]})
        base = write_once(folder / "base.json", {**config, "arguments": [v for k, value in selected.items()
            for v in ([k] if value is None else [k, value])]})
        allocated = prepare_new_task(execution_config=base["path"], description=request["description"])
        argv = build_product_argv(execution_config=allocated, description=request["description"])
        _, actual = _configuration_arguments(allocated)
        generated_refs = [base, file_ref(allocated), experience, binding]
        if actual.get("--patch-worker-config"):
            generated_refs.append(file_ref(actual["--patch-worker-config"]))
        trials.append({**row, "execution_config_ref": file_ref(allocated), "request_ref": task["request_ref"],
            "experience_config_ref": experience, "trial_binding_ref": binding, "argv": argv,
            "run_id": actual["--run-id"], "run_root": str(Path(actual["--workspace-root"]) / actual["--run-id"]),
            "generated_refs": generated_refs})
    prepared = write_once(root / "prepared.json", {"schema_version": "strategy_trial_preparation_v1",
        "scope": "input_preparation_only", "input_ref": input_ref, "protocol_ref": document["protocol_ref"],
        "input_refs": refs, "source_snapshots": snapshots,
        "source_byte_snapshots": byte_snapshots, "trials": trials})
    verify_strategy_trials(prepared)
    return prepared


def verify_strategy_trials(prepared_ref):
    prepared = read_bound_json(prepared_ref)
    _require(prepared.get("schema_version") == "strategy_trial_preparation_v1"
        and prepared.get("scope") == "input_preparation_only", "PREPARED_INVALID")
    document, protocol, tasks, snapshots, refs = _validate(prepared["input_ref"])
    _require(prepared["protocol_ref"] == document["protocol_ref"] and snapshots == prepared["source_snapshots"]
        and refs == prepared["input_refs"], "PREPARED_INPUT_CHANGED")
    _require(prepared["source_byte_snapshots"] == {p: _source_snapshot(p, include_binary=True) for p in snapshots},
        "SOURCE_BYTES_CHANGED")
    _require([{k: t[k] for k in ("task_id", "attempt", "arm")} for t in prepared["trials"]] == document["order"], "ORDER_CHANGED")
    for reference in refs + [r for trial in prepared["trials"] for r in trial["generated_refs"]]:
        _require(file_ref(reference["path"]) == reference, "REFERENCE_CHANGED")
    root = Path(prepared_ref["path"]).resolve().parent
    for index, trial in enumerate(prepared["trials"]):
        task, config, options, _ = tasks[(trial["task_id"], trial["attempt"])]
        folder = root / str(index + 1)
        _require(trial["request_ref"] == task["request_ref"], "REQUEST_CHANGED")
        binding = read_bound_json(trial["trial_binding_ref"])
        expected_binding = {"schema_version": "strategy_trial_run_binding_v1", "protocol_ref": document["protocol_ref"],
            **{k: trial[k] for k in ("task_id", "attempt", "arm")}}
        _require(binding == expected_binding, "BINDING_CHANGED")
        experience = read_bound_json(trial["experience_config_ref"])
        _require(experience == {"schema_version": "task_agent_experience_config_v1", "store_root": document["store_root"],
            "snapshot_ref": protocol["snapshot_ref"] if trial["arm"] == "B+" else None,
            "extract_on_terminal": document["extract_on_terminal"], "trial_binding_ref": trial["trial_binding_ref"]},
            "EXPERIENCE_CHANGED")
        request = read_bound_json(trial["request_ref"])
        validate_product_profile(execution_config=trial["execution_config_ref"]["path"], description=request["description"])
        arguments, actual = _configuration_arguments(trial["execution_config_ref"]["path"])
        expected = deepcopy(options)
        expected.update({"--workspace-root": str(folder / "runs"), "--run-id": actual["--run-id"],
            "--request-id": actual["--request-id"], "--task-agent-experience-config": trial["experience_config_ref"]["path"]})
        run_root = folder / "runs" / actual["--run-id"]
        _require(trial["run_id"] == actual["--run-id"] and trial["run_root"] == str(run_root)
            and trial["argv"] == ["run-request", *arguments, "--", request["description"]], "INVOCATION_CHANGED")
        if options.get("--patch-worker-config"):
            expected["--patch-worker-config"] = actual["--patch-worker-config"]
            worker = read_bound_json(file_ref(options["--patch-worker-config"]))
            for item in config.get("worker_path_bindings", []):
                parent = worker
                for key in item["path"][:-1]:
                    parent = parent[key]
                destination = (run_root / item["relative_path"]).resolve()
                _require(destination.is_relative_to(run_root), "WORKER_OUTPUT_OUTSIDE_RUN")
                parent[item["path"][-1]] = str(destination)
            _require(read_bound_json(file_ref(actual["--patch-worker-config"])) == worker, "WORKER_CHANGED")
        _require(actual == expected, "EXECUTION_OPTIONS_CHANGED")
    return prepared
