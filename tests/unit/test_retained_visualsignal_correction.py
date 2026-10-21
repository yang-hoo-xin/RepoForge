"""Controlled fixtures for one retained binding revision and normal EvalGate rerun."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.local_entry import (
    AcceptedRuntimeEvidenceRecoveryFactory,
    _runtime_resume_admission_snapshot,
    correct_retained_visualsignal_evidence,
    preflight_retained_visualsignal_correction,
)
from src.mod_agent.runtime import (
    RuntimeEvidenceItem,
    bind_runtime_evidence_context,
    build_real_runtime_evidence_payload,
)
from src.mod_agent.runtime.retained_visualsignal import (
    PREFIX,
    RetainedVisualSignalCorrection,
    project_standard_bindings,
)
from src.mod_agent.workflow import (
    ResumePayload,
    WorkflowCheckpointStore,
    build_graph,
    resumed_state_from_decision,
)
from src.mod_agent.workflow.graph import build_retained_visualsignal_correction_graph
from tests.unit.test_workflow_graph import (
    _clone_as_latest_revision,
    _runtime_workflow_graph_operation,
    _successful_build_handoff,
)


def _ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return _ref(path)


@pytest.fixture
def correction_case(tmp_path):
    store, state, _ = _successful_build_handoff(tmp_path, artifact_directory="a")
    feature = state["feature_id"]
    def from_state(kind):
        return ArtifactRef(feature_id=feature, artifact_type=kind, path=state["artifacts"][kind],
                           artifact_hash=state["artifact_hashes"][kind])
    build_ref = _clone_as_latest_revision(store, from_state("build_result"), message_suffix="isolated fixture",
                                        payload_updates={"input_execution_mode": "isolated_apply"})
    state["artifacts"]["build_result"], state["artifact_hashes"]["build_result"] = build_ref.path, build_ref.artifact_hash
    state["build_summary"]["input_execution_mode"] = "isolated_apply"
    state["artifacts"].pop("workflow_checkpoint", None)
    state["artifact_hashes"].pop("workflow_checkpoint", None)
    checkpoint = WorkflowCheckpointStore(store).create_checkpoint(state, interrupt_reason="fixture Build handoff",
                                                                  resume_allowed_fields=("runtime_admission",))
    cp = store.read(checkpoint)["data"]
    feature_ref = from_state("feature_spec")
    criteria = store.read(feature_ref)["data"]["acceptance_criteria"]
    build = store.read(build_ref)["data"]
    candidate = {"path": build["verified_artifacts"][0]["resolved_path"], "sha256": build["verified_artifacts"][0]["sha256"]}
    manifest_ref = _json(tmp_path / "input.json", {"scope": "controlled unit fixture; not a real session"})
    source = tmp_path / "original.py"
    source.write_text("# controlled original verifier\n")
    facts, sources = [], []
    for role, sid, pid in (("native", "session-one", 101), ("display_restore", "session-two", 202)):
        rows = [{"kind": "loaded", "seq": 1, "owner_pid": pid, "session_id": "",
                 "data": {"subject": {"sha256": candidate["sha256"]}}},
                {"kind": "session_started", "seq": 2, "owner_pid": pid, "session_id": sid}]
        if role == "native":
            count = 0
            for index, phase in enumerate(("default", "refresh_disabled", "refresh_reenabled"), 3):
                before = {"count": count}
                if phase != "refresh_disabled":
                    count += 1
                after = {"count": count}
                rows.append({"kind": "native_end", "seq": index, "owner_pid": pid, "session_id": sid,
                    "data": {"event_id": index, "before": before, "after": after}})
                facts.append({"fact": "natural_refresh:"+phase, "session_id": sid, "owner_pid": pid,
                    "seq": [index], "event_id": index, "before": before, "after": after})
        path = tmp_path / (role + ".log")
        path.write_text("\n".join(PREFIX + json.dumps(row) for row in rows)+"\n", encoding="utf-8")
        sources.append({"role": role, "session_id": sid, "owner_pid": pid, "raw_log_ref": _ref(path),
                        "original_status": "insufficient_evidence"})
    for fact in facts:
        fact["raw_log_ref"] = sources[0]["raw_log_ref"]
    humans = {}
    for role in ("native_visible", "disabled_frames_visible", "both_hidden_visible", "restored_visible"):
        selected = sources[1 if role == "restored_visible" else 0]
        item = {"question_call_id": "controlled-"+role, "user_reply": "controlled fixture reply",
                "session_id": selected["session_id"], "owner_pid": selected["owner_pid"]}
        record = {"schema": "operator-observation.v1", "origin": "explicit human reply in current conversation", **item}
        humans[role] = {"ref": _json(tmp_path / (role + ".json"), record), **item}
    proof = {"schema": "visualsignal-retained-session-proof.v1", "sources": sources, "facts": facts,
        "human_observations": humans, "same_candidate_and_environment": True, "source_statuses_rewritten": False,
        "cross_session_counter_continuity_claimed": False}
    payload = build_real_runtime_evidence_payload(feature_id=feature, input_execution_mode="isolated_apply",
        evidence_items=(RuntimeEvidenceItem("log", "controlled_fixture", str(tmp_path / "native.log"), tuple(criteria), 1.0),),
        startup_loaded=True, gameplay_behavior_verified=True, verification_status="verified", acceptance_criteria=criteria,
        runtime_trust_binding={"status": "pending_graph_binding", "checks": {"controlled_fixture": {"status": "passed"}}, "blockers": []})
    payload["evidence_source"] = "verified_retained_visualsignal_sessions"
    payload["visualsignal_session_evidence"] = {"input_manifest_ref": manifest_ref, "source_runtime_execution_count": 2,
        "new_runtime_performed": False, "proof": proof, "candidate_ref": candidate}
    payload = bind_runtime_evidence_context(payload, run_id=cp["run_id"], build_artifact_ref=build_ref)
    admission = {"status": "admitted", "authorization_scope": "this_resume_only",
        "requested_side_effects": ["runtime_execution"], "authorized_side_effects": ["runtime_execution"],
        "factory_ref": _ref(source), "input_manifest_ref": manifest_ref, "evidence_source": "retained_test_fixture",
        "new_process_launch": False, "new_game_directory_write": False, "new_build": False, "new_product_source_write": False}
    decision, accepted = WorkflowCheckpointStore(store).resume(checkpoint, ResumePayload(resume_id="one-original-resume",
        run_id=cp["run_id"], request_id=cp["request_id"], feature_id=feature, stage="build", decision="approve",
        approved_by="original-human", approved_at="2026-10-14", artifact_refs=tuple(cp["artifact_refs"]),
        runtime_admission=_runtime_resume_admission_snapshot(admission)))
    assert decision.allowed
    def fresh_verifier(*args):
        return deepcopy(payload)
    old_graph = build_graph(store, runtime_evidence_factory=fresh_verifier,
        verification_operation=_runtime_workflow_graph_operation(store, checkpoint_ref=checkpoint, resume_report_ref=accepted),
        use_default_proposal_git_binding=False)
    blocked = old_graph.invoke(resumed_state_from_decision(decision, checkpoint_ref=checkpoint, resume_report_ref=accepted))
    assert blocked["current_stage"] == "blocked"
    def old_ref(kind):
        return ArtifactRef(feature_id=feature, artifact_type=kind, path=blocked["artifacts"][kind], artifact_hash=blocked["artifact_hashes"][kind])
    verifier = AcceptedRuntimeEvidenceRecoveryFactory(factory=fresh_verifier, original_factory_ref=admission["factory_ref"],
        input_manifest_ref=manifest_ref, recovery_source_ref=_ref(Path(__file__)))
    request = RetainedVisualSignalCorrection(checkpoint_ref=checkpoint, accepted_resume_report_ref=accepted,
        prior_runtime_ref=old_ref("runtime_evidence"), prior_eval_ref=old_ref("eval_report"), prior_blocked_ref=old_ref("blocked_report"),
        feature_spec_ref=feature_ref, build_result_ref=build_ref, resume_id="one-original-resume", approved_by="original-human",
        original_factory_ref=admission["factory_ref"], input_manifest_ref=manifest_ref, correction_source_ref=_ref(Path(__file__)))
    return store, request, verifier, payload


def _arguments(case):
    store, request, verifier, _ = case
    return dict(artifact_store_root=store.root, correction=request, verifier=verifier,
                expected_correction_identity_hash=request.identity_hash)


def test_normal_correction_adds_one_revision_then_unchanged_evalgate_accepts(correction_case):
    store, request, verifier, prior = correction_case
    original = {ref.path: Path(ref.path).read_bytes() for ref in (
        request.prior_runtime_ref, request.prior_eval_ref, request.prior_blocked_ref, request.accepted_resume_report_ref)}
    preflight = preflight_retained_visualsignal_correction(**_arguments(correction_case))
    corrected = preflight["runtime_evidence_preview"]
    assert {key for key in corrected if corrected[key] != prior.get(key)} == {
        "plugin_marker_binding", "runtime_trust_binding", "retained_binding_correction"}
    assert corrected["visualsignal_session_evidence"] == prior["visualsignal_session_evidence"]
    assert all(item["record_session_id"] == "" and item["bound_session_id"] == item["session_start_binding"]["session_id"]
               for item in corrected["plugin_marker_binding"]["startup_observations"])
    assert {m["acceptance_criterion"] for m in corrected["plugin_marker_binding"]["markers"]} == {prior["acceptance_contract"]["acceptance_criteria"][0]}
    graph = build_retained_visualsignal_correction_graph(store, correction=request, verifier=verifier, expected_identity_hash=request.identity_hash)
    assert set(graph.get_graph().nodes) == {"__start__", "__end__", "correct_retained_runtime_binding", "evaluate", "done", "blocked"}
    result = correct_retained_visualsignal_evidence(**_arguments(correction_case))
    assert result["status"] == "accepted" and result["current_stage"] == "done"
    assert result["runtime_summary"]["terminal_final_status"] == "accepted"
    assert all(Path(path).read_bytes() == raw for path, raw in original.items())
    assert len(store.list_feature_artifact_revisions(request.checkpoint_ref.feature_id, "runtime_evidence")) == 2
    assert len(store.list_feature_artifact_revisions(request.checkpoint_ref.feature_id, "resume_report")) == 1
    revisions = store.list_feature_artifacts(request.checkpoint_ref.feature_id)
    with pytest.raises(ValueError, match="PREDECESSOR_NOT_CURRENT|ALREADY_CORRECTED"):
        correct_retained_visualsignal_evidence(**_arguments(correction_case))
    assert store.list_feature_artifacts(request.checkpoint_ref.feature_id) == revisions


@pytest.mark.parametrize("kind", ["missing_phase", "wrong_pid", "wrong_event", "wrong_counter", "mixed_session", "fake_human_origin"])
def test_projection_refuses_unobserved_or_mixed_claims(correction_case, kind):
    _, request, _, prior = correction_case
    prior = deepcopy(prior)
    proof = prior["visualsignal_session_evidence"]["proof"]
    if kind == "missing_phase":
        proof["facts"].pop()
    elif kind == "wrong_pid":
        proof["facts"][0]["owner_pid"] += 1
    elif kind == "wrong_event":
        proof["facts"][0]["event_id"] += 1
    elif kind == "wrong_counter":
        proof["facts"][0]["after"] = {"count": 999}
    elif kind == "mixed_session":
        proof["facts"][0]["session_id"] = proof["sources"][1]["session_id"]
    else:
        item = proof["human_observations"]["restored_visible"]
        path = Path(item["ref"]["path"])
        data = json.loads(path.read_bytes())
        data["origin"] = "assistant inference"
        item["ref"] = _json(path, data)
    with pytest.raises(ValueError, match="RETAINED_CORRECTION_"):
        project_standard_bindings(prior, request)


@pytest.mark.parametrize("kind", ["action_hash", "resume_id", "approver", "verifier_ref", "raw_log", "dll", "different_fresh_result"])
def test_admission_refuses_scope_and_material_drift(correction_case, kind):
    store, request, verifier, prior = correction_case
    args = _arguments(correction_case)
    if kind == "action_hash":
        args["expected_correction_identity_hash"] = "sha256:" + "0"*64
    elif kind in {"resume_id", "approver"}:
        request = replace(request, **{("approved_by" if kind == "approver" else kind): "unrelated"})
        args.update(correction=request, expected_correction_identity_hash=request.identity_hash)
    elif kind == "verifier_ref":
        args["verifier"] = replace(verifier, original_factory_ref=verifier.input_manifest_ref)
    elif kind == "raw_log":
        Path(prior["visualsignal_session_evidence"]["proof"]["sources"][0]["raw_log_ref"]["path"]).write_text("changed")
    elif kind == "dll":
        Path(prior["visualsignal_session_evidence"]["candidate_ref"]["path"]).write_bytes(b"changed")
    else:
        def wrong(*args):
            return {**prior, "message": "different result"}
        args["verifier"] = replace(verifier, factory=wrong)
    before = store.list_feature_artifacts(request.checkpoint_ref.feature_id)
    with pytest.raises((ValueError, RuntimeError)):
        preflight_retained_visualsignal_correction(**args)
    assert store.list_feature_artifacts(request.checkpoint_ref.feature_id) == before


def test_other_eval_failure_cannot_be_reinterpreted_as_binding_fix(correction_case):
    store, request, _, _ = correction_case
    old = store.read(request.prior_eval_ref)["data"]
    gate = deepcopy(old["evaluation_gate"])
    gate["blockers"].append({"code": "OTHER_REAL_FAILURE", "message": "not a binding defect"})
    changed = _clone_as_latest_revision(store, request.prior_eval_ref, message_suffix="other failure",
                                       payload_updates={"evaluation_gate": gate})
    blocked = _clone_as_latest_revision(store, request.prior_blocked_ref, message_suffix="other failure",
                                       replace_upstream={"eval_report": changed})
    request = replace(request, prior_eval_ref=changed, prior_blocked_ref=blocked)
    args = _arguments(correction_case)
    args.update(correction=request, expected_correction_identity_hash=request.identity_hash)
    with pytest.raises(ValueError, match="NOT_EXACT_BINDING_FAILURE"):
        preflight_retained_visualsignal_correction(**args)


@pytest.mark.parametrize("case", ["loaded_missing_sid", "loaded_wrong_sid", "no_start", "duplicate_start", "wrong_start_pid", "wrong_start_sid", "reversed_seq", "reversed_line"])
def test_loaded_session_identity_requires_actual_same_window_start_chain(correction_case, case):
    _, request, _, prior = correction_case
    prior = deepcopy(prior)
    proof = prior["visualsignal_session_evidence"]["proof"]
    source = proof["sources"][0]
    path = Path(source["raw_log_ref"]["path"])
    rows = [json.loads(line[len(PREFIX):]) for line in path.read_text(encoding="utf-8").splitlines()]
    if case == "loaded_missing_sid":
        rows[0].pop("session_id")
    elif case == "loaded_wrong_sid":
        rows[0]["session_id"] = "invented-session"
    elif case == "no_start":
        rows.pop(1)
    elif case == "duplicate_start":
        rows.append(deepcopy(rows[1]))
    elif case == "wrong_start_pid":
        rows[1]["owner_pid"] += 1
    elif case == "wrong_start_sid":
        rows[1]["session_id"] = "foreign-session"
    elif case == "reversed_seq":
        rows[1]["seq"] = 0
    else:
        rows[0], rows[1] = rows[1], rows[0]
    path.write_text("\n".join(PREFIX + json.dumps(row) for row in rows)+"\n", encoding="utf-8")
    source["raw_log_ref"] = _ref(path)
    for fact in proof["facts"]:
        fact["raw_log_ref"] = source["raw_log_ref"]
    with pytest.raises(ValueError, match="LOADED_SUBJECT_MISMATCH|LOADED_SESSION_CHAIN_MISMATCH"):
        project_standard_bindings(prior, request)
