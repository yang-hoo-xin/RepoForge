"""Exact accepted operations may consume their historical checkpoint approvals."""

from pathlib import Path

import pytest

from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.workflow.checkpointing import (
    WorkflowCheckpointStore,
    resumed_state_from_decision,
)
from src.mod_agent.workflow.graph import _route_runtime_continuation_entry
from tests.unit.test_workflow_graph import (
    _build_runtime_resume_payload,
    _clone_as_latest_revision,
    _runtime_workflow_graph_operation,
    _successful_build_handoff,
)


def _accepted_chain(tmp_path: Path, historical_changes: dict | None = None):
    store, _, prior_checkpoint = _successful_build_handoff(tmp_path)
    owner = WorkflowCheckpointStore(store)
    prior_decision, prior_report = owner.resume(
        prior_checkpoint,
        _build_runtime_resume_payload(
            checkpoint=store.read(prior_checkpoint)["data"],
            resume_id="prior-stage-approval",
        ),
    )
    if historical_changes:
        prior_report = _clone_as_latest_revision(
            store, prior_report, message_suffix="foreign historical control",
            payload_updates=historical_changes,
        )
    historical_state = resumed_state_from_decision(
        prior_decision,
        checkpoint_ref=prior_checkpoint,
        resume_report_ref=prior_report,
    )
    historical_state["artifacts"].pop("workflow_checkpoint")
    historical_state["artifact_hashes"].pop("workflow_checkpoint")
    checkpoint_ref = owner.create_checkpoint(
        historical_state,
        interrupt_reason="Build complete; separate Runtime approval required.",
        resume_allowed_fields=("decision", "runtime_admission"),
    )
    decision, report_ref = owner.resume(
        checkpoint_ref,
        _build_runtime_resume_payload(
            checkpoint=store.read(checkpoint_ref)["data"],
            resume_id="exact-runtime-operation",
        ),
    )
    assert decision.allowed
    state = resumed_state_from_decision(
        decision, checkpoint_ref=checkpoint_ref, resume_report_ref=report_ref
    )
    return store, checkpoint_ref, report_ref, state


def test_accepted_operation_does_not_stale_its_own_checkpoint(tmp_path: Path):
    store, checkpoint, report, state = _accepted_chain(tmp_path)
    assert not store.check_freshness(checkpoint).is_fresh
    operation = _runtime_workflow_graph_operation(
        store, checkpoint_ref=checkpoint, resume_report_ref=report
    )
    assert _route_runtime_continuation_entry(state, store, operation) == "continue"


def test_same_accepted_resume_recovery_is_read_only(tmp_path: Path):
    store, checkpoint, report, _ = _accepted_chain(tmp_path)
    before = {p: p.read_bytes() for p in store.root.rglob("*.json")}
    decision, recovered_report = WorkflowCheckpointStore(
        store
    ).preflight_accepted_resume_recovery(
        checkpoint, resume_id="exact-runtime-operation"
    )
    assert decision.allowed and recovered_report == report
    assert {p: p.read_bytes() for p in store.root.rglob("*.json")} == before


def _read_accepted(store, checkpoint, report, **changes):
    values = {
        "resume_report_ref": report,
        "resume_id": "exact-runtime-operation",
        "approved_by": "human",
        "runtime_action_scope": store.read(report)["data"]["applied_updates"]["runtime_admission"],
    }
    values.update(changes)
    return WorkflowCheckpointStore(store).read_accepted_runtime_checkpoint(checkpoint, **values)


@pytest.mark.parametrize("role", ["feature_spec", "build_result", "patch_proposal"])
def test_accepted_history_does_not_exempt_material_drift(tmp_path: Path, role: str):
    store, checkpoint, report, state = _accepted_chain(tmp_path)
    ref = ArtifactRef(
        feature_id=checkpoint.feature_id,
        artifact_type=role,
        path=state["artifacts"][role],
        artifact_hash=state["artifact_hashes"][role],
    )
    _clone_as_latest_revision(store, ref, message_suffix="changed material")
    with pytest.raises(ValueError, match="STALE|INVALID"):
        _read_accepted(store, checkpoint, report)


def test_accepted_history_still_verifies_physical_build_bytes(tmp_path: Path):
    store, checkpoint, report, state = _accepted_chain(tmp_path)
    build = store.read(ArtifactRef(
        feature_id=checkpoint.feature_id,
        artifact_type="build_result",
        path=state["artifacts"]["build_result"],
        artifact_hash=state["artifact_hashes"]["build_result"],
    ))["data"]
    Path(build["verified_artifacts"][0]["resolved_path"]).write_bytes(b"different DLL")
    with pytest.raises(ValueError, match="BUILD_OUTPUT_BYTES_CHANGED"):
        _read_accepted(store, checkpoint, report)


@pytest.mark.parametrize("changes", [
    {"resume_id": "other-operation"},
    {"approved_by": "other-human"},
    {"runtime_action_scope": {"requested_side_effects": ["process_launch"]}},
])
def test_accepted_identity_and_scope_cannot_change(tmp_path: Path, changes: dict):
    store, checkpoint, report, _ = _accepted_chain(tmp_path)
    with pytest.raises(ValueError, match="MISMATCH"):
        _read_accepted(store, checkpoint, report, **changes)


@pytest.mark.parametrize("changes", [
    {"run_id": "another-run"},
    {"checkpoint_id": "another-checkpoint"},
    {"decision": "reject"},
    {"allowed": False},
    {"status": "blocked"},
])
def test_invalid_current_report_never_unlocks_history(tmp_path: Path, changes: dict):
    store, checkpoint, report, _ = _accepted_chain(tmp_path)
    changed = _clone_as_latest_revision(
        store, report, message_suffix="invalid report", payload_updates=changes
    )
    with pytest.raises(ValueError, match="BINDING_MISMATCH"):
        _read_accepted(store, checkpoint, changed)


def test_superseded_current_report_is_not_historical_exemption(tmp_path: Path):
    store, checkpoint, report, _ = _accepted_chain(tmp_path)
    _clone_as_latest_revision(store, report, message_suffix="different current report")
    with pytest.raises(ValueError, match="ACCEPTED_RUNTIME_REPORT_STALE"):
        _read_accepted(store, checkpoint, report)
    with pytest.raises(ValueError, match="exactly one"):
        WorkflowCheckpointStore(store).preflight_accepted_resume_recovery(
            checkpoint, resume_id="exact-runtime-operation"
        )


@pytest.mark.parametrize("changes", [
    {"run_id": "other-historical-run"},
    {"checkpoint_id": "other-historical-checkpoint"},
    {"resume_stage": "dry_run_apply"},
    {"decision": "reject"},
])
def test_historical_control_exemption_requires_same_accepted_lineage(tmp_path: Path, changes: dict):
    store, checkpoint, report, _ = _accepted_chain(tmp_path, changes)
    with pytest.raises(ValueError, match="HISTORICAL_RESUME_REPORT_BINDING_MISMATCH"):
        _read_accepted(store, checkpoint, report)
