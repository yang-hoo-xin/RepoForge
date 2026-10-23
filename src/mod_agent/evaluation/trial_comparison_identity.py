"""Separate frozen platform bytes from per-run inputs and compare effective limits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def platform_identity(snapshot):
    """Retained inputs stay bound inputs, not a different platform per run.

    为什么要有它：worker 输出路径变化会改变配置字节，但不能据此声称平台代码版本变化。
    """
    files = snapshot["files"]
    external = snapshot.get("external_refs", [])
    retained, code = set(), set()
    for row in external:
        if row.get("role") not in {"bound_input", "extra_python_load"}:
            raise ValueError("Unknown frozen external reference role.")
        name = row["retained_path"]
        if name not in files or files[name]["sha256"] != row["sha256"]:
            raise ValueError("Frozen external file identity disagrees with its manifest.")
        retained.add(name)
        if row["role"] == "extra_python_load":
            code.add(row["sha256"])
    identity = {"platform_files": {name: value for name, value in files.items() if name not in retained},
        "external_python_sha256": sorted(code)}
    return {"platform_files_sha256": _hash(identity),
        "identity_scope": "platform_source_static_resources_and_extra_python_load_excluding_bound_inputs",
        "bound_input_refs": [row for row in external if row["role"] == "bound_input"]}


def _bound_input(snapshot, root, flag):
    argv = snapshot.get("business_argv", [])
    if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
        return None
    path = Path(argv[argv.index(flag) + 1]).resolve()
    matches = [row for row in snapshot.get("external_refs", []) if row["role"] == "bound_input"
        and Path(row["path"]).resolve() == path]
    if len(matches) != 1:
        return None
    selected = matches[0]
    retained = (root / selected["retained_path"]).resolve()
    if not retained.is_relative_to(root):
        raise ValueError("Retained budget input escapes its execution snapshot.")
    raw = retained.read_bytes()
    if hashlib.sha256(raw).hexdigest() != selected["sha256"]:
        raise ValueError("Retained budget input changed.")
    return json.loads(raw)


def resource_budget_identity(snapshot, snapshot_root, manifest, result, process_deadline_seconds):
    root = Path(snapshot_root).resolve()
    worker = _bound_input(snapshot, root, "--patch-worker-config")
    policy = worker.get("policy", {}) if isinstance(worker, dict) else {}
    model = result.get("model_call_budget") or manifest.get("model_call_budget") or {}
    native = manifest.get("ollama_patch_generation_profile") or {}
    build = manifest.get("nuget_isolated_build_profile") or {}
    total_only = model.get("policy") == "task_total_only"
    metered = model.get("policy") == "meter_only"
    no_stage_caps = total_only or metered
    if metered and model.get("maximum") is not None:
        raise ValueError("Meter-only model budget must declare a null maximum.")
    # The execution owner replaces worker quotas under this explicit policy.
    # 为什么要有它：计量策略取消的上限应标为不执行，不能误报为缺失或沿用配置中的旧配额。
    budget = {"task_model_call_maximum": model.get("maximum"), "model_call_policy": model.get("policy"),
        "source_tool_read_maximum": None if metered else policy.get("max_tool_rounds"),
        "observation_bytes_maximum": None if metered else policy.get("max_observation_bytes"),
        "normal_process_deadline_seconds": process_deadline_seconds,
        "model_request_timeout_seconds": native.get("timeout_seconds"),
        "build_process_timeout_seconds": build.get("timeout_seconds"),
        "stage_model_limits": None if no_stage_caps else _bound_input(snapshot, root, "--stage-model-call-limits"),
        "worker_model_turn_maximum": None if no_stage_caps else policy.get("max_total_model_turns")}
    required = [key for key in budget if key not in {"stage_model_limits", "worker_model_turn_maximum"}]
    if not no_stage_caps:
        required += ["stage_model_limits", "worker_model_turn_maximum"]
    unenforced = ["stage_model_limits", "worker_model_turn_maximum"] if no_stage_caps else []
    if metered:
        unenforced += ["task_model_call_maximum", "source_tool_read_maximum", "observation_bytes_maximum"]
        required = [key for key in required if key not in unenforced]
    missing = [key for key in required if budget[key] is None]
    if metered and "maximum" not in model:
        missing.append("task_model_call_maximum")
    if metered and (not isinstance(worker, dict) or not isinstance(worker.get("policy"), dict)):
        missing.append("worker_policy")
    return {"effective_limits": budget, "complete": not missing, "missing": missing,
        "declared_worker_model_turn_maximum": policy.get("max_total_model_turns"),
        "stage_limits_enforced": not no_stage_caps, "unenforced_limits": unenforced}
