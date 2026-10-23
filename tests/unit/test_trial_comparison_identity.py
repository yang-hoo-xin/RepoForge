"""Per-run retained configuration is not platform code or extra budget credit."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from src.mod_agent.evaluation.trial_comparison_identity import (
    platform_identity,
    resource_budget_identity,
)


def _snapshot(tmp_path, *, reads=6, observation_bytes=32768):
    original = tmp_path.parent / "worker.json"
    raw = json.dumps({"policy": {"max_tool_rounds": reads, "max_observation_bytes": observation_bytes,
        "max_total_model_turns": 8}, "output_root": str(tmp_path / "run")}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    relative = "external/" + digest + "/worker.json"
    retained = tmp_path / relative
    retained.parent.mkdir(parents=True, exist_ok=True)
    retained.write_bytes(raw)
    return {"files": {"src/code.py": {"sha256": "a" * 64, "size": 10}, relative: {"sha256": digest, "size": len(raw)},
        "external/code/factory.py": {"sha256": "b" * 64, "size": 20}},
        "business_argv": ["run-request", "task", "--patch-worker-config", str(original)],
        "external_refs": [{"role": "bound_input", "path": str(original), "retained_path": relative, "sha256": digest},
            {"role": "extra_python_load", "path": str(tmp_path.parent / "factory.py"),
                "retained_path": "external/code/factory.py", "sha256": "b" * 64}]}


def test_relocated_bound_worker_input_changes_old_files_hash_but_not_platform(tmp_path):
    first = _snapshot(tmp_path / "a")
    second = _snapshot(tmp_path / "b")
    assert first["files"] != second["files"]
    assert platform_identity(first)["platform_files_sha256"] == platform_identity(second)["platform_files_sha256"]
    assert platform_identity(first)["bound_input_refs"] != platform_identity(second)["bound_input_refs"]
    code_changed = deepcopy(second)
    code_changed["files"]["src/code.py"]["sha256"] = "c" * 64
    assert platform_identity(first)["platform_files_sha256"] != platform_identity(code_changed)["platform_files_sha256"]


def test_external_python_code_remains_part_of_platform_identity(tmp_path):
    first = _snapshot(tmp_path)
    changed = deepcopy(first)
    changed["external_refs"][1]["sha256"] = "c" * 64
    changed["files"]["external/code/factory.py"]["sha256"] = "c" * 64
    assert platform_identity(first)["platform_files_sha256"] != platform_identity(changed)["platform_files_sha256"]


@pytest.mark.parametrize("changed", ["reads", "bytes", "deadline", "build", "model_timeout"])
def test_actual_resource_limit_increase_changes_comparison_identity(tmp_path, changed):
    manifest = {"ollama_patch_generation_profile": {"timeout_seconds": 180}, "nuget_isolated_build_profile": {"timeout_seconds": 120}}
    result = {"model_call_budget": {"maximum": 12, "policy": "task_total_only"}}
    first = _snapshot(tmp_path / "first")
    other = _snapshot(tmp_path / "other", reads=12 if changed == "reads" else 6,
        observation_bytes=65536 if changed == "bytes" else 32768)
    baseline = resource_budget_identity(first, tmp_path / "first", manifest, result, 1800)
    if changed == "build":
        manifest["nuget_isolated_build_profile"]["timeout_seconds"] = 240
    if changed == "model_timeout":
        manifest["ollama_patch_generation_profile"]["timeout_seconds"] = 360
    candidate = resource_budget_identity(other, tmp_path / "other", manifest, result, 3600 if changed == "deadline" else 1800)
    assert baseline["complete"] and candidate["complete"]
    assert baseline["effective_limits"] != candidate["effective_limits"]


def test_missing_deadline_stays_unknown_and_changed_retained_config_is_rejected(tmp_path):
    snapshot = _snapshot(tmp_path)
    manifest = {"ollama_patch_generation_profile": {"timeout_seconds": 180}, "nuget_isolated_build_profile": {"timeout_seconds": 120}}
    result = {"model_call_budget": {"maximum": 12, "policy": "task_total_only"}}
    identity = resource_budget_identity(snapshot, tmp_path, manifest, result, None)
    assert not identity["complete"] and "normal_process_deadline_seconds" in identity["missing"]
    (tmp_path / snapshot["external_refs"][0]["retained_path"]).write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        resource_budget_identity(snapshot, tmp_path, manifest, result, 1800)


def test_explicit_meter_only_disables_quotas_but_preserves_timeouts(tmp_path):
    snapshot = _snapshot(tmp_path)
    manifest = {"ollama_patch_generation_profile": {"timeout_seconds": 180},
        "nuget_isolated_build_profile": {"timeout_seconds": 120}}
    result = {"model_call_budget": {"maximum": None, "policy": "meter_only"}}
    identity = resource_budget_identity(snapshot, tmp_path, manifest, result, 1800)
    assert identity["complete"] and not identity["stage_limits_enforced"]
    assert identity["declared_worker_model_turn_maximum"] == 8
    assert set(identity["unenforced_limits"]) == {"task_model_call_maximum", "source_tool_read_maximum",
        "observation_bytes_maximum", "stage_model_limits", "worker_model_turn_maximum"}
    assert all(identity["effective_limits"][key] is None for key in identity["unenforced_limits"])
    assert identity["effective_limits"]["normal_process_deadline_seconds"] == 1800
    assert identity["effective_limits"]["model_request_timeout_seconds"] == 180
    assert identity["effective_limits"]["build_process_timeout_seconds"] == 120
    other = _snapshot(tmp_path / "other", reads=100, observation_bytes=100000)
    assert identity["effective_limits"] == resource_budget_identity(other, tmp_path / "other", manifest, result, 1800)["effective_limits"]
    missing = resource_budget_identity(snapshot, tmp_path, manifest, result, None)
    assert missing["missing"] == ["normal_process_deadline_seconds"]


@pytest.mark.parametrize("omitted", ["maximum", "policy", "worker", "model_timeout", "build_timeout"])
def test_meter_only_does_not_turn_missing_facts_into_unlimited_policy(tmp_path, omitted):
    snapshot = _snapshot(tmp_path)
    manifest = {"ollama_patch_generation_profile": {"timeout_seconds": 180},
        "nuget_isolated_build_profile": {"timeout_seconds": 120}}
    result = {"model_call_budget": {"maximum": None, "policy": "meter_only"}}
    if omitted in {"maximum", "policy"}:
        del result["model_call_budget"][omitted]
    elif omitted == "worker":
        snapshot["business_argv"] = []
    else:
        manifest.pop("ollama_patch_generation_profile" if omitted == "model_timeout" else "nuget_isolated_build_profile")
    identity = resource_budget_identity(snapshot, tmp_path, manifest, result, 1800)
    assert not identity["complete"]
    if omitted == "policy":
        assert identity["stage_limits_enforced"]
        assert identity["unenforced_limits"] == []


def test_meter_only_rejects_contradictory_numeric_maximum(tmp_path):
    snapshot = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="null maximum"):
        resource_budget_identity(snapshot, tmp_path, {},
            {"model_call_budget": {"maximum": 10, "policy": "meter_only"}}, 1800)


def test_legacy_stage_policy_keeps_stage_and_worker_limits_required(tmp_path):
    snapshot = _snapshot(tmp_path)
    manifest = {"ollama_patch_generation_profile": {"timeout_seconds": 180},
        "nuget_isolated_build_profile": {"timeout_seconds": 120}}
    identity = resource_budget_identity(snapshot, tmp_path, manifest,
        {"model_call_budget": {"maximum": 10, "policy": "stage_and_total"}}, 1800)
    assert identity["missing"] == ["stage_model_limits"]
    assert identity["stage_limits_enforced"]
    assert identity["effective_limits"]["worker_model_turn_maximum"] == 8
    assert identity["effective_limits"]["source_tool_read_maximum"] == 6
