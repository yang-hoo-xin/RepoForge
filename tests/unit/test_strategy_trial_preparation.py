"""Paired preparation uses real CLI parsing without model or execution calls."""
from dataclasses import asdict
from pathlib import Path

import pytest

from src.mod_agent.evaluation.independent_preparation import file_ref, write_once
from src.mod_agent.evaluation.memory_comparison import INVARIANTS
from src.mod_agent.evaluation.strategy_trial_preparation import (
    _source_snapshot,
    prepare_strategy_trials,
    verify_strategy_trials,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from tests.unit.test_csharp_repository_backend import _profile
from tests.unit.test_strategy_experience import manual
from tests.unit.test_strategy_experience import setup as experience_setup


@pytest.fixture
def trial_input(tmp_path):
    memory_setup = experience_setup.__wrapped__(tmp_path)
    _, snapshot = manual(memory_setup)
    profile, source = _profile(tmp_path / "repository", real=False)
    settings = tmp_path / "settings.yaml"
    settings.write_text("task_memory:\n  enabled: false\n", encoding="utf-8")
    config = write_once(tmp_path / "config.json", {"arguments": [
        "--target-project", str(source), "--workspace-root", str(tmp_path / "old-runs"),
        "--settings", str(settings), "--local-ollama", "--ollama-model", "controlled-model",
        "--ollama-context-window", "32768", "--ollama-max-output-tokens", "2048",
        "--model-call-budget-policy", "meter_only", "--defer-approval", "--confirm-isolated-write", "--allow-build",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2", "--repository-profile", profile.reference["path"]]})
    request = write_once(tmp_path / "request.json", {"schema_version": "public_development_request_v1",
        "visibility": "public", "description": "Preserve arithmetic behavior."})
    source_ref = write_once(tmp_path / "source.json", {"task_id": "one", "task_family": "arithmetic",
        "source_version": "fixture", "source_files_sha256": canonical_hash(_source_snapshot(source)),
        "original_request_sha256": canonical_hash("Preserve arithmetic behavior.")})
    exposure = write_once(tmp_path / "exposure.json", {"task_id": "one", "attempt": 1,
        "source_ref": source_ref, "exposure": "unseen"})
    invariants = {k: {"fixture": k} for k in INVARIANTS | {"feedback"}}
    protocol = write_once(tmp_path / "protocol.json", {"schema_version": "strategy_trial_protocol_v1",
        "invariants": invariants, "invariants_sha256": canonical_hash(invariants),
        "snapshot_ref": asdict(snapshot), "task_families": ["arithmetic"], "planned_pairs": [
            {"task_id": "one", "task_family": "arithmetic", "attempt": 1, "source_ref": source_ref,
             "exposure_ref": exposure, "current_context": {}}]})
    return write_once(tmp_path / "input.json", {"schema_version": "strategy_trial_preparation_input_v1",
        "protocol_ref": protocol, "store_root": str(tmp_path / "artifacts"), "extract_on_terminal": False,
        "tasks": [{"task_id": "one", "attempt": 1, "execution_config_ref": config, "request_ref": request}],
        "order": [{"task_id": "one", "attempt": 1, "arm": arm} for arm in ("B+", "B")]}), memory_setup, source


def test_prepare_real_bridge_and_verify_inputs(tmp_path, trial_input):
    input_ref, _, source = trial_input
    before = _source_snapshot(source)
    prepared_ref = prepare_strategy_trials(input_ref, tmp_path / "prepared")
    prepared = verify_strategy_trials(prepared_ref)
    assert prepared["scope"] == "input_preparation_only"
    assert [t["arm"] for t in prepared["trials"]] == ["B+", "B"]
    plus, base = prepared["trials"]
    assert read_bound_json(plus["experience_config_ref"])["snapshot_ref"]
    assert read_bound_json(base["experience_config_ref"])["snapshot_ref"] is None
    for trial in prepared["trials"]:
        assert trial["argv"][0] == "run-request"
        assert trial["argv"][-1] == "Preserve arithmetic behavior."
        assert Path(trial["run_root"]).is_relative_to(tmp_path / "prepared")
        assert not Path(trial["run_root"]).exists()
    assert before == _source_snapshot(source)
    with pytest.raises(ValueError, match="OUTPUT_EXISTS"):
        prepare_strategy_trials(input_ref, tmp_path / "prepared")


@pytest.mark.parametrize("change", ["source", "settings", "experience", "revocation"])
def test_verify_rechecks_changes(tmp_path, trial_input, change):
    input_ref, setup, source = trial_input
    prepared = prepare_strategy_trials(input_ref, tmp_path / "prepared")
    if change == "source":
        (source / "new.cs").write_text("public class Other {}")
    elif change == "settings":
        (tmp_path / "settings.yaml").write_text("task_memory:\n  enabled: true\n  store_root: memory\n")
    elif change == "experience":
        Path(read_bound_json(prepared)["trials"][0]["experience_config_ref"]["path"]).write_text("{}")
    else:
        memory, evidence, strategy, _ = setup
        memory.record_strategy("retained-reader", {**strategy, "suggested_action": "Revised strategy"},
            evidence_refs=evidence, origin={"kind": "human", "actor": "test-developer"},
            expected_revision=memory.current_strategy("retained-reader"))
    with pytest.raises(ValueError):
        verify_strategy_trials(prepared)


@pytest.mark.parametrize("change", ["order", "private", "overlap", "source_hash"])
def test_reject_before_output(tmp_path, trial_input, change):
    input_ref, _, source = trial_input
    document = read_bound_json(input_ref)
    destination = tmp_path / "prepared"
    if change == "order":
        document["order"][1] = document["order"][0]
    elif change == "private":
        document["tasks"][0]["request_ref"] = write_once(tmp_path / "private.json", {
            **read_bound_json(document["tasks"][0]["request_ref"]), "visibility": "hidden"})
    elif change == "overlap":
        destination = source / "prepared"
    else:
        (source / "changed.cs").write_text("changed")
    changed = write_once(tmp_path / "changed-input.json", document)
    with pytest.raises(ValueError):
        prepare_strategy_trials(changed, destination)
    assert not destination.exists()


def test_explicit_worker_relocation_and_binary_drift(tmp_path, trial_input):
    input_ref, _, source = trial_input
    (source / "resource.bin").write_bytes(b"\xff\x00binary")
    document = read_bound_json(input_ref)
    config = read_bound_json(document["tasks"][0]["execution_config_ref"])
    worker = write_once(tmp_path / "worker.json", {"behavior_evaluator": {"kwargs": {
        "output_root": str(tmp_path / "old-output"), "expected_fixed_inputs": {
            "fixed_files": [{"path": "resource.bin", "sha256": "1" * 64}]}}}})
    config["arguments"].extend(["--patch-worker-config", worker["path"]])
    config["worker_path_bindings"] = [{"path": ["behavior_evaluator", "kwargs", "output_root"], "relative_path": "checks"}]
    document["tasks"][0]["execution_config_ref"] = write_once(tmp_path / "worker-config.json", config)
    changed_input = write_once(tmp_path / "worker-input.json", document)
    reference = prepare_strategy_trials(changed_input, tmp_path / "prepared")
    prepared = verify_strategy_trials(reference)
    from scripts.dsh_run_request import _configuration_arguments
    for trial in prepared["trials"]:
        _, options = _configuration_arguments(trial["execution_config_ref"]["path"])
        saved = read_bound_json(file_ref(options["--patch-worker-config"]))
        assert saved["behavior_evaluator"]["kwargs"]["output_root"] == str(Path(trial["run_root"]) / "checks")
    (source / "resource.bin").write_bytes(b"\xff\x00changed")
    with pytest.raises(ValueError, match="SOURCE_BYTES_CHANGED"):
        verify_strategy_trials(reference)


@pytest.mark.parametrize("change", ["arm", "model", "argv"])
def test_rehashed_prepared_record_cannot_relabel_actual_inputs(tmp_path, trial_input, change):
    input_ref, _, _ = trial_input
    reference = prepare_strategy_trials(input_ref, tmp_path / "prepared")
    prepared = read_bound_json(reference)
    first = prepared["trials"][0]
    if change == "arm":
        experience = read_bound_json(first["experience_config_ref"])
        experience["snapshot_ref"] = None
        first["experience_config_ref"] = write_once(tmp_path / "changed-experience.json", experience)
    elif change == "model":
        config = read_bound_json(first["execution_config_ref"])
        config["arguments"][config["arguments"].index("--ollama-model") + 1] = "different-model"
        first["execution_config_ref"] = write_once(tmp_path / "changed-config.json", config)
        first["argv"] = ["run-request", *config["arguments"], "--", "Preserve arithmetic behavior."]
    else:
        first["argv"][-1] = "Different request"
    changed = write_once(tmp_path / "prepared" / "rehashed.json", prepared)
    with pytest.raises(ValueError, match="EXPERIENCE_CHANGED|EXECUTION_OPTIONS_CHANGED|INVOCATION_CHANGED"):
        verify_strategy_trials(changed)
