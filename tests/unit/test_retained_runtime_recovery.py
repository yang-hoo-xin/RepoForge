"""Normal accepted-resume recovery for source-bound retained evidence adapters."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from src.mod_agent.local_entry import (
    AcceptedRuntimeEvidenceRecoveryFactory,
    LocalEntryError,
    _runtime_resume_admission_snapshot,
    resume_local_runtime,
)
from src.mod_agent.workflow import ResumePayload, WorkflowCheckpointStore
from tests.unit.test_workflow_graph import (
    _fake_runtime_factory,
    _successful_build_handoff,
    _write_runtime_handoff_manifest,
)


def _ref(path):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _retained_fixture_factory(feature_id, build, feature_spec):
    return _fake_runtime_factory(feature_id, build, feature_spec)


@pytest.fixture
def retained(tmp_path, monkeypatch):
    store, _, checkpoint_ref = _successful_build_handoff(tmp_path, artifact_directory="a")
    checkpoint = store.read(checkpoint_ref)["data"]
    _write_runtime_handoff_manifest(tmp_path, checkpoint=checkpoint, checkpoint_ref=checkpoint_ref)
    original = tmp_path / "original-verifier.py"
    manifest = tmp_path / "original-input.json"
    original.write_text("# original approved retained verifier\n", encoding="utf-8")
    manifest.write_text('{"source_sessions": 2}', encoding="utf-8")
    admission = {
        "status": "admitted", "authorization_scope": "this_resume_only",
        "requested_side_effects": ["runtime_execution"], "authorized_side_effects": ["runtime_execution"],
        "evidence_source": "retained_test_evidence", "factory_ref": _ref(original), "input_manifest_ref": _ref(manifest),
        "new_process_launch": False, "new_game_directory_write": False, "new_build": False,
        "new_product_source_write": False,
    }
    decision, report = WorkflowCheckpointStore(store).resume(checkpoint_ref, ResumePayload(
        resume_id="same-accepted-retained-id", run_id=checkpoint["run_id"], request_id=checkpoint["request_id"],
        feature_id=checkpoint["feature_id"], stage="build", decision="approve", approved_by="original-human",
        approved_at="2026-10-14T00:00:00Z", artifact_refs=tuple(checkpoint["artifact_refs"]),
        runtime_admission=_runtime_resume_admission_snapshot(admission),
    ))
    assert decision.allowed
    for name in ("load_settings", "_runtime_admission_record", "_make_deploying_runtime_evidence_factory"):
        monkeypatch.setattr("src.mod_agent.local_entry." + name,
                            lambda *a, **k: pytest.fail("retained recovery entered settings/deployment"))
    adapter = AcceptedRuntimeEvidenceRecoveryFactory(
        factory=_retained_fixture_factory, original_factory_ref=admission["factory_ref"],
        input_manifest_ref=admission["input_manifest_ref"], recovery_source_ref=_ref(__file__))
    output = tmp_path / "new-recovery-result"
    output.mkdir()
    arguments = dict(run_root=tmp_path, artifact_store_root=tmp_path / "a", feature_id=checkpoint_ref.feature_id,
        checkpoint_path=checkpoint_ref.path, checkpoint_hash=checkpoint_ref.artifact_hash,
        approved_by="current-developer", resume_id="same-accepted-retained-id", recover_accepted_resume=True,
        runtime_output_root=output, allow_game_launch=False, allow_game_directory_write=False)
    return store, checkpoint_ref, report, admission, adapter, arguments


def test_normal_recovery_reuses_exact_report_without_deployment_or_new_approval(retained):
    store, checkpoint, report, admission, adapter, arguments = retained
    originals = {ref.path: Path(ref.path).read_bytes() for ref in (checkpoint, report)}
    result = resume_local_runtime(**arguments, runtime_evidence_factory=adapter)
    assert result["status"] == "completed" and result["current_stage"] == "done"
    assert result["accepted_resume_recovery"] is True
    assert result["artifacts"]["resume_report"] == report.path
    assert result["runtime_admission"] == admission
    assert result["deployment_backup"] is None and result["model_call_count"] == 0
    assert result["retained_factory_recovery"]["aggregation_origin"] == "development_assisted_retained_evidence_recovery"
    assert result["retained_factory_recovery"]["original_factory_ref"] == adapter.original_factory_ref
    assert result["retained_factory_recovery"]["recovery_source_ref"] == _ref(__file__)
    assert len(store.list_feature_artifact_revisions(checkpoint.feature_id, "resume_report")) == 1
    assert all(Path(path).read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("mismatch", ["original_ref", "manifest_ref", "source_file", "original_bytes", "manifest_bytes", "source_sha"])
def test_recovery_rejects_changed_origin_or_inputs_before_graph(retained, tmp_path, mismatch):
    store, checkpoint, _, _, adapter, arguments = retained
    other = tmp_path / "other.py"
    other.write_text("# unrelated file", encoding="utf-8")
    if mismatch == "original_ref":
        adapter = replace(adapter, original_factory_ref=_ref(other))
    elif mismatch == "manifest_ref":
        adapter = replace(adapter, input_manifest_ref=_ref(other))
    elif mismatch == "source_file":
        adapter = replace(adapter, recovery_source_ref=_ref(other))
    elif mismatch == "original_bytes":
        Path(adapter.original_factory_ref["path"]).write_text("changed verifier")
    elif mismatch == "manifest_bytes":
        Path(adapter.input_manifest_ref["path"]).write_text("changed inputs")
    else:
        adapter = replace(adapter, recovery_source_ref={**adapter.recovery_source_ref, "sha256": "0" * 64})
    with pytest.raises(LocalEntryError, match="RETAINED_RECOVERY_"):
        resume_local_runtime(**arguments, runtime_evidence_factory=adapter)
    assert not store.list_feature_artifact_revisions(checkpoint.feature_id, "runtime_evidence")
    assert len(store.list_feature_artifact_revisions(checkpoint.feature_id, "resume_report")) == 1


@pytest.mark.parametrize("change", ["raw_callback", "new_admission", "game_launch", "game_write", "destination"])
def test_recovery_rejects_arbitrary_callback_or_effect_expansion(retained, tmp_path, change):
    store, checkpoint, _, admission, adapter, arguments = retained
    if change == "raw_callback":
        adapter = _retained_fixture_factory
    elif change == "new_admission":
        arguments["injected_runtime_admission"] = admission
    elif change == "game_launch":
        arguments["allow_game_launch"] = True
    elif change == "game_write":
        arguments["allow_game_directory_write"] = True
    else:
        arguments["runtime_plugin_destination"] = tmp_path / "game" / "candidate.dll"
    with pytest.raises(LocalEntryError, match="BOUND_FACTORY_REQUIRED|does not accept injected"):
        resume_local_runtime(**arguments, runtime_evidence_factory=adapter)
    assert not store.list_feature_artifact_revisions(checkpoint.feature_id, "runtime_evidence")


@pytest.mark.parametrize("field,value", [
    ("new_process_launch", True), ("new_game_directory_write", True), ("new_build", True),
    ("new_product_source_write", True), ("requested_side_effects", ["runtime_execution", "process_launch"]),
    ("authorized_side_effects", ["runtime_execution", "process_launch"]),
])
def test_retained_adapter_cannot_reinterpret_broader_accepted_scope(retained, field, value):
    _, _, _, admission, adapter, _ = retained
    with pytest.raises(LocalEntryError, match="ACCEPTED_SCOPE_MISMATCH"):
        adapter.validate_admission({**admission, field: value})
