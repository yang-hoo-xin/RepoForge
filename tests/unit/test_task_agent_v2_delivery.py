"""Controlled normal-entry delivery: no real model, SDK, or user source writes."""
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from src.mod_agent import task_agent_v2_delivery as delivery_module
from src.mod_agent.build.adapter import BuildCommandResult
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.reviewed_patch_continuation import continue_v2_delivery
from src.mod_agent.task_agent_v2_delivery import (
    apply_v2_delivery,
    build_v2_delivery,
    delivery_recovery_context,
    prepare_v2_delivery,
    prepare_v2_delivery_rebase,
    prepare_v2_task_delegation,
    validate_v2_delivery_rebase,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, load_v2_artifact, save_v2_artifact
from tests.unit.test_task_agent_execution import file_ref
from tests.unit.test_task_agent_v2_product_entry import (
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff as run_controlled_entry,
)


@pytest.fixture
def delivery_case(short_workspace, monkeypatch, capsys):
    run_controlled_entry(short_workspace, monkeypatch, capsys, False)
    pending = json.loads((short_workspace / "v2-entry-result.json").read_text())
    store = Path(pending["run_root"]) / "a"
    refs = list((store / "v2/delivery-checkpoint").glob("*.json"))
    assert len(refs) == 1
    isolated = short_workspace / "delivery"
    isolated.mkdir()
    ref = refs[0].relative_to(store).as_posix()
    prepared = prepare_v2_delivery(artifact_store_root=store, checkpoint_ref=ref,
        isolated_root=str(isolated), operation_id="controlled-apply")
    return store, ref, isolated, prepared


def approve(prepared):
    return {**prepared["human_semantic_decision_binding"], "decision": "approve", "reviewer_identity": "CONTROLLED TEST HUMAN"}


def test_exact_approval_apply_build_and_replay_preserve_original(delivery_case, capsys):
    store, checkpoint_ref, _isolated, prepared = delivery_case
    checkpoint = load_v2_artifact(store, checkpoint_ref)
    original = Path(checkpoint["source_root"]) / "Library/Arithmetic.cs"
    before = original.read_bytes()
    decision = approve(prepared)
    wrong = {**decision, "diff_hash": "0" * 64}
    with pytest.raises(RuntimeError, match="diff_hash"):
        apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"], human_semantic_decision=wrong)
    assert not Path(prepared["displayed_binding"]["isolated_workspace"]).exists()
    applied = continue_v2_delivery(artifact_store_root=store, checkpoint_ref=checkpoint_ref, action="apply",
        displayed_binding=prepared["displayed_binding"], human_semantic_decision=decision)
    assert applied["status"] == "applied_in_isolation"
    assert original.read_bytes() == before
    assert "Factor.Multiplier" in (Path(applied["applied_source_root"]) / "Library/Arithmetic.cs").read_text()
    replay = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"], human_semantic_decision=decision)
    assert replay["receipt_ref"] == applied["receipt_ref"] and replay["replayed"]
    wrong_receipt = load_v2_artifact(store, applied["receipt_ref"])
    wrong_receipt["candidate_source_id"] = "unrelated-candidate"
    wrong_ref = save_v2_artifact(store, "delivery-receipt", wrong_receipt)
    with pytest.raises(ValueError, match="APPLY_BINDING_CHANGED"):
        build_v2_delivery(artifact_store_root=store, apply_ref=wrong_ref, operation_id="wrong", allow_build=True)
    factor = Path(applied["applied_source_root"]) / "Library/Factor.cs"
    saved_bytes = factor.read_bytes()
    factor.write_bytes(saved_bytes.replace(b"Multiplier = 2", b"Multiplier = 9"))
    forged = load_v2_artifact(store, applied["receipt_ref"])
    forged["applied_source_snapshot"] = delivery_module._snapshot(factor.parent.parent)
    forged_ref = save_v2_artifact(store, "delivery-receipt", forged)
    key = canonical_hash({"kind": "apply", "operation_id": prepared["displayed_binding"]["operation_id"]})
    pointer = store / "v2/delivery-operations" / (key + ".result.json")
    pointer_bytes = pointer.read_bytes()
    pointer.write_text(json.dumps({"receipt_ref": forged_ref}))
    try:
        with pytest.raises(ValueError, match="APPLIED_POSTIMAGE_CHANGED"):
            build_v2_delivery(artifact_store_root=store, apply_ref=forged_ref, operation_id="forged", allow_build=True)
    finally:
        factor.write_bytes(saved_bytes)
        pointer.write_bytes(pointer_bytes)
    with pytest.raises(ValueError, match="BUILD_NOT_AUTHORIZED"):
        build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="build-1")
    built = continue_v2_delivery(artifact_store_root=store, checkpoint_ref=applied["receipt_ref"], action="build",
        operation_id="build-1", allow_build=True)
    assert built["status"] == "built", built
    assert built["resource_record"]["delivery_applies"] == built["resource_record"]["delivery_builds"] == 1
    assert built["resource_record"]["model_budget"] == checkpoint["resource_record"]["model_budget"]
    assert built["whole_task_delivered"] is True
    assert built["delivery_scope"] == "approved_isolated_source_and_bound_public_build_tests"
    assert build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"],
        operation_id="build-1", allow_build=True)["replayed"]
    with pytest.raises(ValueError, match="OPERATION_ID_REUSED"):
        build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="build-2", allow_build=True)
    recovery = delivery_recovery_context(artifact_store_root=store, receipt_ref=built["receipt_ref"])
    assert recovery["requires_fresh_source_views"] is True and recovery["approval_inherited"] is False
    native_ref = checkpoint["handoff"]["execution_ref"]
    observed = delivery_module.read_completed_v2_delivery(artifact_store_root=store,
        receipt_ref=built["receipt_ref"], native_execution_ref=native_ref)
    assert observed["status"] == "delivered" and observed["candidate_source_id"] == built["candidate_source_id"]
    assert observed["verification"]["native_verification_ref"] == built["verification_ref"]
    with pytest.raises(ValueError, match="DELIVERY_NATIVE_EXECUTION_CHANGED"):
        delivery_module.read_completed_v2_delivery(artifact_store_root=store,
            receipt_ref=built["receipt_ref"], native_execution_ref={**native_ref, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="COMPLETED_DELIVERY_REQUIRED"):
        delivery_module.read_completed_v2_delivery(artifact_store_root=store,
            receipt_ref=applied["receipt_ref"], native_execution_ref=native_ref)
    _export_completed_delivery(store, checkpoint, built, capsys)


def _export_completed_delivery(store, checkpoint, built, capsys):
    from src.mod_agent import cli
    from src.mod_agent.domain.full_project_compilation import _save
    from src.mod_agent.model_qualification import read_bound_json

    task = TaskAgentV2State(store, checkpoint["handoff"]["task_contract_ref"])
    baseline = task.load(task.contract["baseline_manifest_ref"])
    source = _save(store.parent / "diagnostic-source.json", {"task_id": "delivery-fixture", "task_family": "controlled-arithmetic",
        "source_version": "controlled-v1", "source_files_sha256": canonical_hash({p: row["sha256"] for p, row in baseline["files"].items()}),
        "original_request_sha256": canonical_hash(task.load(task.contract["original_request_ref"])["original_request"])})
    exposure = _save(store.parent / "diagnostic-exposure.json", {"task_id": "delivery-fixture", "source_ref": source,
        "attempt": 1, "exposure": "exposed"})
    protocol = _save(store.parent / "diagnostic-protocol.json", {"schema_version": "strategy_trial_protocol_v1",
        "batch_complete": True, "snapshot_ref": None, "invariants": {}, "invariants_sha256": canonical_hash({}),
        "task_families": ["controlled-arithmetic"], "planned_pairs": [{"task_id": "delivery-fixture",
            "task_family": "controlled-arithmetic", "attempt": 1, "source_ref": source, "exposure_ref": exposure,
            "current_context": {}}]})
    native_ref = checkpoint["handoff"]["execution_ref"]
    native = read_bound_json(native_ref)
    timing = built["task_timing_observation"]
    assert timing["origin_ref"] == native["task_timing_origin_ref"]
    assert timing["event"] == "product_build_return" and timing["status"] == "built"
    assert timing["timing_status"] == "observed" and timing["duration_ms"] >= 0
    call_id = next(row["call_id"] for row in native["completed_tool_records"] if row["name"] == "finish")
    code = cli.main(["export-strategy-execution", "--execution-record", native_ref["path"], "--call-id", call_id,
        "--output-root", str(store.parent / "strategy-delivery-export"), "--protocol-record", protocol["path"],
        "--task-id", "delivery-fixture", "--attempt", "1", "--arm", "B", "--delivery-receipt", built["receipt_ref"]])
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    exported = read_bound_json(result["execution_ref"])
    delivery = read_bound_json(exported["delivery_ref"])
    assert delivery["status"] == "delivered" and delivery["candidate_source_id"] == built["candidate_source_id"]
    assert read_bound_json(delivery["verification_ref"])["native_verification_ref"] == built["verification_ref"]
    assert delivery["task_timing_observation"] == timing
    assert exported["known_cost"]["duration_ms"] == timing["duration_ms"]
    assert exported["freeze_binding"]["status"] == "freeze_binding_unproven"


def test_public_build_failure_binds_applied_preimages_and_budget(delivery_case, monkeypatch):
    store, checkpoint_ref, _isolated, prepared = delivery_case
    applied = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
        human_semantic_decision=approve(prepared))
    def fail(_profile, command):
        Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1", "tests": [
            {"name": "public-regression", "passed": False, "expected": 6, "actual": 3}]}))
        return BuildCommandResult(1, "controlled public failure", "", 1)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", fail)
    failed = build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="failed-build", allow_build=True)
    assert failed["status"] == "build_failed"
    feedback = load_v2_artifact(store, failed["failure_feedback_ref"])
    assert feedback["visibility"] == "public" and feedback["verification_ref"]
    assert feedback["candidate_ref"] == applied["candidate_ref"]
    assert feedback["applied_source_snapshot"] == applied["applied_source_snapshot"]
    assert feedback["resource_record"]["delivery_builds"] == 1
    assert feedback["observations"]
    (Path(applied["applied_source_root"]) / "Library/Factor.cs").write_text("changed after Apply")
    with pytest.raises(ValueError, match="RECOVERY_SOURCE_CHANGED"):
        delivery_recovery_context(artifact_store_root=store, receipt_ref=failed["receipt_ref"])


def test_explicit_host_delegation_is_bound_and_not_human_review(delivery_case):
    store, ref, isolated, prepared = delivery_case
    checkpoint = load_v2_artifact(store, ref)
    authorization = {"origin": "task_delegation_policy", "decision": "approve", "policy_id": "controlled-host-policy",
        "runtime": False, "deployment": False, "run_id": checkpoint["run_id"], "request_id": checkpoint["request_id"],
        "isolated_root": str(isolated), "allowed_effects": ["isolated_apply", "build"]}
    path = isolated / "host-authorization.json"
    path.write_text(json.dumps(authorization))
    delegated = prepare_v2_task_delegation(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
        policy_id=authorization["policy_id"], authorization_ref=file_ref(path), allow_build=True)
    record = load_v2_artifact(store, delegated)
    assert record["human_review_completed"] is False
    wrong = deepcopy(prepared["displayed_binding"])
    wrong["candidate_source_id"] = "other"
    with pytest.raises(ValueError, match="DISPLAY_BINDING_CHANGED"):
        apply_v2_delivery(artifact_store_root=store, displayed_binding=wrong, task_delegation_ref=delegated)
    applied = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"], task_delegation_ref=delegated)
    assert applied["approval"]["origin"] == "task_delegation_policy"
    assert applied["approval"]["human_review_completed"] is False
    built = build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="delegated-build",
        task_delegation_ref=delegated)
    assert built["status"] == "built"


def test_uncertain_build_and_interrupt_cannot_be_replayed_as_new_execution(delivery_case, monkeypatch):
    store, checkpoint_ref, isolated, prepared = delivery_case
    applied = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
        human_semantic_decision=approve(prepared))
    calls = []
    def unknown(*_args, **_kwargs):
        calls.append("unknown")
        raise RuntimeError("controlled loss of execution response")
    monkeypatch.setattr(backend.CSharpRepositoryBackend, "run_build_test", unknown)
    result = build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="unknown", allow_build=True)
    assert result["status"] == "build_unknown" and result["builds_performed"] is None
    assert build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="unknown", allow_build=True)["replayed"]
    assert calls == ["unknown"]
    with pytest.raises(ValueError, match="RECOVERY_RECEIPT_INVALID"):
        delivery_recovery_context(artifact_store_root=store, receipt_ref=result["receipt_ref"])
    prepared = prepare_v2_delivery(artifact_store_root=store, checkpoint_ref=checkpoint_ref,
        isolated_root=str(isolated), operation_id="interrupt-apply")
    applied = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
        human_semantic_decision=approve(prepared))
    def interrupt(*_args, **_kwargs):
        calls.append("interrupt")
        raise KeyboardInterrupt()
    monkeypatch.setattr(backend.CSharpRepositoryBackend, "run_build_test", interrupt)
    with pytest.raises(KeyboardInterrupt):
        build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="interrupt", allow_build=True)
    with pytest.raises(ValueError, match="INFLIGHT_REQUIRES_RECONCILIATION"):
        build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="interrupt", allow_build=True)
    assert calls == ["unknown", "interrupt"]


def test_partial_apply_retains_public_failure_and_does_not_repeat(delivery_case, monkeypatch):
    store, _checkpoint_ref, _isolated, prepared = delivery_case
    actual_replace = delivery_module.os.replace
    calls = []
    def fail_second(source, target):
        calls.append(str(target))
        if len(calls) == 2:
            raise OSError("controlled second replacement failure")
        return actual_replace(source, target)
    monkeypatch.setattr(delivery_module.os, "replace", fail_second)
    with pytest.raises(OSError) as raised:
        apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
            human_semantic_decision=approve(prepared))
    retained = load_v2_artifact(store, raised.value.delivery_receipt_ref)
    feedback = load_v2_artifact(store, raised.value.failure_feedback_ref)
    assert retained["status"] == "apply_failed"
    assert retained["write_observations"][0]["state"] == "confirmed"
    assert feedback["execution_status"] == "apply_failed" and feedback["visibility"] == "public"
    assert feedback["applied_source_snapshot"] == retained["applied_source_snapshot"]
    replay = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
        human_semantic_decision=approve(prepared))
    assert replay["status"] == "apply_failed" and replay["replayed"] and len(calls) == 2


def test_human_approval_cannot_move_to_another_destination(delivery_case):
    store, checkpoint_ref, isolated, prepared = delivery_case
    other = isolated.parent / "another-delivery-root"
    other.mkdir()
    moved = prepare_v2_delivery(artifact_store_root=store, checkpoint_ref=checkpoint_ref,
        isolated_root=str(other), operation_id=prepared["displayed_binding"]["operation_id"])
    assert moved["candidate_diff_hash"] == prepared["candidate_diff_hash"]
    assert moved["human_semantic_decision_binding"]["diff_hash"] != prepared["human_semantic_decision_binding"]["diff_hash"]
    with pytest.raises(RuntimeError, match="diff_hash"):
        apply_v2_delivery(artifact_store_root=store, displayed_binding=moved["displayed_binding"],
            human_semantic_decision=approve(prepared))
    assert not Path(moved["displayed_binding"]["isolated_workspace"]).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse point counterexample")
@pytest.mark.parametrize("position", ["ancestor", "folder", "intent", "result"])
def test_operation_rejects_windows_junctions(short_workspace, position):
    store = short_workspace / "store"
    (store / "v2").mkdir(parents=True)
    external = short_workspace / "external"
    external.mkdir()
    key = canonical_hash({"kind": "apply", "operation_id": "junction"})
    if position == "ancestor":
        (store / "v2").rmdir()
        link = store / "v2"
    elif position == "folder":
        link = store / "v2/delivery-operations"
    else:
        (store / "v2/delivery-operations").mkdir()
        link = store / "v2/delivery-operations" / (key + f".{position}.json")
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(external)], capture_output=True)
    if result.returncode:
        pytest.skip("Junction creation unavailable")
    with pytest.raises(ValueError, match="link"):
        delivery_module._operation(store, "apply", "junction", {})
    assert list(external.iterdir()) == []


def test_applied_rebase_creates_real_current_contract_and_preserves_resources(delivery_case, monkeypatch):
    store, checkpoint_ref, _isolated, prepared = delivery_case
    checkpoint = load_v2_artifact(store, checkpoint_ref)
    prior = TaskAgentV2State(store, checkpoint["handoff"]["task_contract_ref"])
    old_view = prior.read_source(prior.current_revision_ref, "Library/Arithmetic.cs")
    old_resources = prior.head["resources"].copy()
    applied = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
        human_semantic_decision=approve(prepared))
    def failed_test(_profile, command):
        Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1", "tests": [
            {"name": "public-copy-behavior", "passed": False, "expected": 6, "actual": 3}]}))
        return BuildCommandResult(1, "controlled copy-specific failure", "", 1)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", failed_test)
    failed = build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="rebase-build", allow_build=True)
    assert failed["status"] == "build_failed", failed
    prepared_rebase = prepare_v2_delivery_rebase(artifact_store_root=store, receipt_ref=failed["receipt_ref"], operation_id="rebase-1")
    record = validate_v2_delivery_rebase(artifact_store_root=store, rebase_ref=prepared_rebase["rebase_ref"])
    assert record["run_id"] == checkpoint["run_id"] and record["request_id"] == checkpoint["request_id"]
    assert record["new_task_created"] is False and record["new_apply_approval_granted"] is False
    assert record["failure_feedback_ref"] == failed["failure_feedback_ref"]
    current = TaskAgentV2State(store, record["new_task_contract_ref"])
    assert current.contract["task_id"] == prior.contract["task_id"]
    assert current.contract["resource_policy_ref"] == prior.contract["resource_policy_ref"]
    assert current.head["resources"] == old_resources
    assert current.contract["prior_delivery_resource_record"]["delivery_applies"] == 1
    assert current.contract["prior_delivery_resource_record"]["delivery_builds"] == 1
    assert current.load(current.registry_ref)["bindings"] == {}
    assert current.current_revision_ref == current.contract["baseline_manifest_ref"]
    assert current.contract["historical_candidate_ref"] == checkpoint["handoff"]["candidate_ref"]
    assert prior.head["superseded_by_contract_ref"] == current.contract_ref
    read = current.read_source(current.current_revision_ref, "Library/Arithmetic.cs")
    assert "value * Factor.Multiplier" in read["text"]
    with pytest.raises(ValueError, match="source_view_not_edit_authority"):
        current.add_binding(current.registry_ref, old_view["source_view_ref"], "Library/Arithmetic.cs")
    replay = prepare_v2_delivery_rebase(artifact_store_root=store, receipt_ref=failed["receipt_ref"], operation_id="rebase-1")
    assert replay["rebase_ref"] == prepared_rebase["rebase_ref"] and replay["replayed"]
    bad = deepcopy(record)
    bad["host_overrides"]["allowed_edit_paths"].append("not-authorized.cs")
    with pytest.raises(ValueError, match="REBASE_ORIGIN_CHANGED"):
        validate_v2_delivery_rebase(artifact_store_root=store, rebase_ref=save_v2_artifact(store, "delivery-rebase", bad))
