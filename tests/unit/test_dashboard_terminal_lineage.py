"""Current terminal projection must not confuse retained historical outcomes."""
from pathlib import Path

from src.mod_agent.api.http_server import ModAgentHttpApi
from src.mod_agent.artifacts import ArtifactStore
from tests.unit.test_dashboard_projection import (
    _eval_report_payload,
    _feature_spec_payload,
    _workflow_summary_payload,
    _write_full_chain,
)

FEATURE = "feature_001"


def _latest(store, kind):
    current = next(ref for ref in store.list_feature_artifacts(FEATURE) if ref.artifact_type == kind)
    return next(ref for ref in store.list_feature_artifact_revisions(FEATURE, kind) if ref.artifact_hash == current.artifact_hash)


def _evaluate(store, *, accepted):
    payload = _eval_report_payload()
    payload.update(message="current accepted" if accepted else "current blocked", verdict="accepted" if accepted else "blocked",
        claims_real_success=accepted, v1_acceptance_status="accepted", v1_acceptance_blockers=[])
    payload["evaluation_gate"].update(status="passed" if accepted else "failed")
    blockers = [] if accepted else [{"stage": "evaluation", "code": "EXACT_EVAL_FAILURE", "message": "current evaluation failed"}]
    return store.write(feature_id=FEATURE, artifact_type="eval_report", payload=payload,
        status="success" if accepted else "blocked", blockers=blockers,
        upstream_refs=[_latest(store, "build_result"), _latest(store, "runtime_evidence")])


def _blocked(store, evaluation):
    return store.write(feature_id=FEATURE, artifact_type="blocked_report", status="blocked", payload={
        "node_name": "blocked", "stage": "blocked", "message": "retained terminal failure",
        "blockers": [{"stage": "evaluation", "code": "EXACT_EVAL_FAILURE", "message": "current evaluation failed"}]},
        upstream_refs=[evaluation])


def _accepted_summary(store, evaluation, *, bind=True):
    payload = _workflow_summary_payload()
    payload.update(final_status="accepted", claims_real_mod_success=True, claims_real_build_success=True,
                   claims_real_runtime_success=True, eval_verdict="accepted", v1_acceptance_status="accepted")
    return store.write(feature_id=FEATURE, artifact_type="workflow_summary", payload=payload,
                       upstream_refs=[evaluation] if bind else [])


def test_old_blocked_then_new_fresh_accepted_is_current_without_deleting_history(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    _write_full_chain(store)
    previous = _blocked(store, _evaluate(store, accepted=False))
    original = Path(previous.path).read_bytes()
    _accepted_summary(store, _evaluate(store, accepted=True))
    api = ModAgentHttpApi(store)
    status = api.handle_get(f"/features/{FEATURE}/status").payload
    dashboard = api.handle_get(f"/features/{FEATURE}/dashboard").payload
    assert status["status"] == "available" and status["workflow"]["final_status"] == "accepted"
    assert status["blockers"] == []
    assert dashboard["historical_blockers"][0]["artifact_type"] == "blocked_report"
    assert dashboard["evaluation"]["verdict"] == "accepted"
    assert any(row["artifact_type"] == "blocked_report" and row["freshness"]["status"] == "stale" for row in dashboard["artifacts"])
    assert Path(previous.path).read_bytes() == original


def test_old_accepted_cannot_override_new_current_blocked(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    _write_full_chain(store)
    summary = _accepted_summary(store, _evaluate(store, accepted=True))
    original = Path(summary.path).read_bytes()
    _blocked(store, _evaluate(store, accepted=False))
    status = ModAgentHttpApi(store).handle_get(f"/features/{FEATURE}/status").payload
    assert status["status"] == "blocked" and status["blockers"]
    assert status["workflow"]["final_status"] == "blocked"
    assert status["workflow"]["claims_real_mod_success"] is False
    assert Path(summary.path).read_bytes() == original


def test_upstream_drift_invalidates_accepted_terminal_claim(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    _write_full_chain(store)
    _accepted_summary(store, _evaluate(store, accepted=True))
    store.write(feature_id=FEATURE, artifact_type="feature_spec", payload=_feature_spec_payload(message="new source contract"))
    dashboard = ModAgentHttpApi(store).handle_get(f"/features/{FEATURE}/dashboard").payload
    assert dashboard["status"] == "blocked"
    assert dashboard["workflow"]["final_status"] != "accepted"
    assert dashboard["workflow"]["claims_real_mod_success"] is False
    assert dashboard["evaluation"]["status"] == "stale"
    assert any(row["code"] == "CURRENT_TERMINAL_STALE" for row in dashboard["blockers"])


def test_blocked_record_bound_to_current_eval_is_not_hidden_by_summary(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    _write_full_chain(store)
    evaluation = _evaluate(store, accepted=True)
    _accepted_summary(store, evaluation)
    _blocked(store, evaluation)
    dashboard = ModAgentHttpApi(store).handle_get(f"/features/{FEATURE}/dashboard").payload
    assert dashboard["status"] == "blocked"
    assert dashboard.get("historical_blockers", []) == []
    assert dashboard["workflow"]["final_status"] == "blocked"
    assert dashboard["workflow"]["claims_real_mod_success"] is False
    assert any(row["artifact_type"] == "blocked_report" for row in dashboard["blockers"])


def test_accepted_summary_without_current_eval_lineage_is_not_authority(tmp_path):
    store = ArtifactStore(tmp_path / "a")
    _write_full_chain(store)
    _accepted_summary(store, _evaluate(store, accepted=True), bind=False)
    status = ModAgentHttpApi(store).handle_get(f"/features/{FEATURE}/status").payload
    assert status["status"] == "blocked"
    assert status["workflow"]["claims_real_mod_success"] is False
    assert any(row["code"] == "CURRENT_TERMINAL_LINEAGE_INVALID" for row in status["blockers"])
