"""Bounded standard-binding correction for retained VisualSignal observations.

This owner does not run Runtime, Build, Apply, models, or user callbacks on game objects.
It rechecks an approved retained verifier and projects already-observed records.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.mod_agent.action_identity import hash_canonical_value
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.workflow.checkpointing import WorkflowCheckpointStore

CORRECTION_KIND = "retained_visualsignal_standard_binding_correction_v1"
GATE_CODE = "EVAL_GATE_RUNTIME_CLAIM_BOUNDARY_BINDING_FAILED"
PREFIX = "[Info   :C4 Runtime Observer] C4_RUNTIME_V1 "


@dataclass(frozen=True)
class RetainedVisualSignalCorrection:
    checkpoint_ref: ArtifactRef
    accepted_resume_report_ref: ArtifactRef
    prior_runtime_ref: ArtifactRef
    prior_eval_ref: ArtifactRef
    prior_blocked_ref: ArtifactRef
    feature_spec_ref: ArtifactRef
    build_result_ref: ArtifactRef
    resume_id: str
    approved_by: str
    original_factory_ref: Mapping[str, str]
    input_manifest_ref: Mapping[str, str]
    correction_source_ref: Mapping[str, str]

    def identity(self):
        return {"kind": CORRECTION_KIND, "refs": {name: vars(getattr(self, name)) for name in (
            "checkpoint_ref", "accepted_resume_report_ref", "prior_runtime_ref", "prior_eval_ref",
            "prior_blocked_ref", "feature_spec_ref", "build_result_ref")},
            "resume_id": self.resume_id, "approved_by": self.approved_by,
            "original_factory_ref": dict(self.original_factory_ref), "input_manifest_ref": dict(self.input_manifest_ref),
            "correction_source_ref": dict(self.correction_source_ref),
            "effect_scope": {"runtime_evidence_revision": 1, "eval_gate_reevaluation": 1,
                "new_model_calls": 0, "new_build": False, "new_apply": False, "new_game_run": False,
                "product_source_write": False, "new_approval": False, "automatic_retry": False}}

    @property
    def identity_hash(self):
        return hash_canonical_value(self.identity())


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _read_file(reference):
    path = Path(reference["path"])
    _require(path.is_absolute() and not str(path).startswith(("\\\\", "//")), "RETAINED_CORRECTION_LOCAL_FILE_REQUIRED")
    for item in (path, *path.parents):
        _require(not item.is_symlink() and not (item.exists() and getattr(item.lstat(), "st_file_attributes", 0) & 0x400),
                 "RETAINED_CORRECTION_REPARSE_PATH")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "RETAINED_CORRECTION_SOURCE_DIGEST_CHANGED")
    return raw


def _records(reference):
    records = []
    offset = 0
    for number, line in enumerate(_read_file(reference).splitlines(keepends=True), 1):
        text = line.decode("utf-8-sig").rstrip("\r\n")
        if text.startswith(PREFIX):
            records.append((json.loads(text[len(PREFIX):]), {"marker": text, "source_ref": dict(reference),
                "line_number": number, "line_start_byte": offset, "line_end_byte": offset + len(line),
                "line_sha256": hashlib.sha256(line).hexdigest()}))
        offset += len(line)
    return records


def project_standard_bindings(prior: Mapping[str, Any], request: RetainedVisualSignalCorrection) -> dict[str, Any]:
    """Project actual passive-observer lines and existing human statements only."""
    _require(prior.get("evidence_source") == "verified_retained_visualsignal_sessions"
             and prior.get("input_execution_mode") == "isolated_apply"
             and prior.get("verification_status") == "verified" and prior.get("acceptance_verified") is True
             and prior.get("execution_mode") == "real" and prior.get("startup_loaded") is True
             and prior.get("gameplay_behavior_verified") is True and prior.get("claims_real_runtime_success") is True
             and prior.get("runtime_trust_binding", {}).get("status") == "trusted"
             and prior.get("plugin_marker_binding") == {}, "RETAINED_CORRECTION_WRONG_PREDECESSOR")
    retained = prior["visualsignal_session_evidence"]
    _require(retained["input_manifest_ref"] == request.input_manifest_ref
             and retained.get("source_runtime_execution_count") == 2
             and retained.get("new_runtime_performed") is False, "RETAINED_CORRECTION_INPUT_CHANGED")
    proof = retained["proof"]
    _require(proof["schema"] == "visualsignal-retained-session-proof.v1"
             and proof.get("same_candidate_and_environment") is True
             and proof.get("source_statuses_rewritten") is False
             and proof.get("cross_session_counter_continuity_claimed") is False,
             "RETAINED_CORRECTION_PROOF_SCOPE_CHANGED")
    sources = proof["sources"]
    _require(len(sources) == 2 and {item["role"] for item in sources} == {"native", "display_restore"},
             "RETAINED_CORRECTION_SESSION_ROLES_CHANGED")
    by_role = {item["role"]: item for item in sources}
    by_session = {item["session_id"]: item for item in sources}
    _require(len(by_session) == 2, "RETAINED_CORRECTION_SESSIONS_NOT_INDEPENDENT")
    parsed = {sid: _records(source["raw_log_ref"]) for sid, source in by_session.items()}
    startup = []
    candidate_sha = retained["candidate_ref"]["sha256"]
    for sid, source in by_session.items():
        loaded = [(row, binding) for row, binding in parsed[sid] if row.get("kind") == "loaded"]
        starts = [(row, binding) for row, binding in parsed[sid] if row.get("kind") == "session_started"]
        _require(len(loaded) == 1 and loaded[0][0].get("owner_pid") == source["owner_pid"]
                 and loaded[0][0].get("session_id") == ""
                 and loaded[0][0]["data"]["subject"]["sha256"] == candidate_sha,
                 "RETAINED_CORRECTION_LOADED_SUBJECT_MISMATCH")
        # Scope: loaded predates host session assignment in the actual observer log.
        # 为什么要有它：空session_id不能被改写成已观察身份；只能通过同窗口同PID后续唯一started记录建立明确关联。
        _require(len(starts) == 1 and starts[0][0].get("session_id") == sid
                 and starts[0][0].get("owner_pid") == source["owner_pid"]
                 and type(loaded[0][0].get("seq")) is int and type(starts[0][0].get("seq")) is int
                 and loaded[0][0]["seq"] < starts[0][0]["seq"]
                 and loaded[0][1]["line_number"] < starts[0][1]["line_number"],
                 "RETAINED_CORRECTION_LOADED_SESSION_CHAIN_MISMATCH")
        startup.append({**loaded[0][1], "record_session_id": loaded[0][0]["session_id"],
            "bound_session_id": sid, "owner_pid": source["owner_pid"], "subject_sha256": candidate_sha,
            "record_seq": loaded[0][0]["seq"], "session_start_binding": {**starts[0][1],
                "session_id": starts[0][0]["session_id"], "owner_pid": starts[0][0]["owner_pid"],
                "record_seq": starts[0][0]["seq"]}})
    criteria = prior["acceptance_contract"]["acceptance_criteria"]
    _require(len(criteria) == 3, "RETAINED_CORRECTION_CRITERIA_CHANGED")
    markers = []
    for fact in proof["facts"]:
        if not fact["fact"].startswith("natural_refresh:"):
            continue
        native = by_role["native"]
        _require(fact["session_id"] == native["session_id"] and fact["owner_pid"] == native["owner_pid"]
                 and fact["raw_log_ref"] == native["raw_log_ref"], "RETAINED_CORRECTION_NATURAL_SOURCE_MIXED")
        selected = [(row, binding) for row, binding in parsed[native["session_id"]]
                    if row.get("seq") == fact["seq"][-1]]
        _require(len(selected) == 1, "RETAINED_CORRECTION_MARKER_MISSING_OR_DUPLICATED")
        row, binding = selected[0]
        _require(row.get("kind") == "native_end" and row.get("session_id") == fact["session_id"]
                 and row.get("owner_pid") == fact["owner_pid"]
                 and row["data"]["event_id"] == fact["event_id"]
                 and row["data"]["before"] == fact["before"] and row["data"]["after"] == fact["after"],
                 "RETAINED_CORRECTION_MARKER_FACT_MISMATCH")
        markers.append({**binding, "acceptance_criterion": criteria[0], "observed": True,
            "session_id": fact["session_id"], "owner_pid": fact["owner_pid"], "record_seq": row["seq"],
            "observation_fact": fact["fact"], "subject_sha256": candidate_sha,
            "observation_scope": "natural callback/config clause; remaining obligations retain source/control/human evidence"})
    _require({item["observation_fact"] for item in markers} == {
        "natural_refresh:default", "natural_refresh:refresh_disabled", "natural_refresh:refresh_reenabled"},
        "RETAINED_CORRECTION_NATURAL_PHASE_MISSING")
    humans = proof["human_observations"]
    _require(set(humans) == {"native_visible", "disabled_frames_visible", "both_hidden_visible", "restored_visible"},
             "RETAINED_CORRECTION_HUMAN_ROLES_CHANGED")
    for role, item in humans.items():
        record = json.loads(_read_file(item["ref"]))
        source = by_role["display_restore" if role == "restored_visible" else "native"]
        _require(record.get("schema") == "operator-observation.v1"
                 and record.get("origin") == "explicit human reply in current conversation"
                 and all(isinstance(item.get(key), str) and item[key] for key in ("question_call_id", "user_reply"))
                 and all(record.get(key) == item[key] for key in ("question_call_id", "user_reply", "session_id", "owner_pid"))
                 and item["session_id"] == source["session_id"] and item["owner_pid"] == source["owner_pid"],
                 "RETAINED_CORRECTION_HUMAN_BINDING_MISMATCH")
    corrected = deepcopy(dict(prior))
    # Scope: markers cover actual native observations, not Build/source obligations.
    # 为什么要有它：非运行条款不能伪装为插件输出；既有人工观察与源码证据仍按原验收分工保留。
    corrected["plugin_marker_binding"] = {"marker_source": "gtfo_persistent_log", "markers": markers,
        "marker_plugin": "C4 Runtime Observer", "startup_observations": startup,
        "new_marker_emitted": False, "binding_origin": "retained_passive_observer_records"}
    trust = corrected["runtime_trust_binding"]
    trust["operator_completion_required"] = True
    trust["checks"]["operator_completion"] = {"status": "passed", "required": True,
        "origin": "retained explicit human replies; no new reply", "observations": deepcopy(humans)}
    trust["checks"]["acceptance_markers"] = {"status": "passed", "marker_source": "gtfo_persistent_log",
        "observed_native_record_count": len(markers), "source_refs": [item["raw_log_ref"] for item in sources]}
    corrected["retained_binding_correction"] = {"kind": CORRECTION_KIND, "identity_hash": request.identity_hash,
        "identity": request.identity(), "predecessor_ref": vars(request.prior_runtime_ref),
        "changed_fields": ["plugin_marker_binding", "runtime_trust_binding.operator_completion_required",
            "runtime_trust_binding.checks.operator_completion", "runtime_trust_binding.checks.acceptance_markers"],
        "aggregation_origin": "development_assisted_retained_evidence_correction", "new_runtime_performed": False}
    return corrected


def prepare_correction(store: ArtifactStore, request: RetainedVisualSignalCorrection, verifier):
    """Read-only admission with exact predecessor, current-source and single-successor fences."""
    refs = request.identity()["refs"]
    feature = request.checkpoint_ref.feature_id
    _require(all(ref["feature_id"] == feature for ref in refs.values()), "RETAINED_CORRECTION_FEATURE_MISMATCH")
    expected_types = {"checkpoint_ref": "workflow_checkpoint", "accepted_resume_report_ref": "resume_report",
        "prior_runtime_ref": "runtime_evidence", "prior_eval_ref": "eval_report", "prior_blocked_ref": "blocked_report",
        "feature_spec_ref": "feature_spec", "build_result_ref": "build_result"}
    _require(all(refs[key]["artifact_type"] == kind for key, kind in expected_types.items()), "RETAINED_CORRECTION_REF_TYPE_MISMATCH")
    envelopes = {name: store.semantic_read_immutable(ArtifactRef(**ref)) for name, ref in refs.items()}
    report = envelopes["accepted_resume_report_ref"]["data"]
    scope = report["applied_updates"]["runtime_admission"]
    admission = scope["runtime_admission"]
    _require(verifier.original_factory_ref == request.original_factory_ref
             and verifier.input_manifest_ref == request.input_manifest_ref
             and verifier.recovery_source_ref == request.correction_source_ref,
             "RETAINED_CORRECTION_VERIFIER_IDENTITY_CHANGED")
    verifier.validate_admission(admission)
    checkpoint = WorkflowCheckpointStore(store).read_accepted_runtime_checkpoint(request.checkpoint_ref,
        resume_report_ref=request.accepted_resume_report_ref, resume_id=request.resume_id, approved_by=request.approved_by,
        runtime_action_scope=scope, allow_prior_runtime_state=True)["data"]
    latest = {ref.artifact_type: ref for ref in store.list_feature_artifacts(feature)}
    for name in ("prior_runtime_ref", "prior_eval_ref", "prior_blocked_ref", "feature_spec_ref", "build_result_ref"):
        ref = ArtifactRef(**refs[name])
        _require(ref.artifact_type in latest and latest[ref.artifact_type].artifact_hash == ref.artifact_hash,
                 "RETAINED_CORRECTION_PREDECESSOR_NOT_CURRENT")
        _require(store.check_freshness(ref).is_fresh, "RETAINED_CORRECTION_MATERIAL_STALE")
    for ref in store.list_feature_artifact_revisions(feature, "runtime_evidence"):
        existing = store.semantic_read_immutable(ref)["data"].get("retained_binding_correction") or {}
        _require(existing.get("predecessor_ref") != vars(request.prior_runtime_ref), "RETAINED_CORRECTION_PREDECESSOR_ALREADY_CORRECTED")
    prior = envelopes["prior_runtime_ref"]["data"]
    build = envelopes["build_result_ref"]["data"]
    feature_spec = envelopes["feature_spec_ref"]["data"]
    snapshot = checkpoint["state_snapshot"]
    _require(all(snapshot["artifacts"][kind] == refs[name]["path"]
                 and snapshot["artifact_hashes"][kind] == refs[name]["artifact_hash"]
                 for kind, name in (("build_result", "build_result_ref"), ("feature_spec", "feature_spec_ref"))),
             "RETAINED_CORRECTION_BUILD_FEATURE_CHANGED")
    _require(build.get("input_execution_mode") == "isolated_apply"
             and prior.get("input_execution_mode") == build.get("input_execution_mode")
             and prior.get("run_id") == checkpoint["run_id"] and prior.get("feature_id") == feature
             and prior.get("build_artifact_ref") == vars(request.build_result_ref)
             and prior.get("acceptance_contract", {}).get("acceptance_criteria") == feature_spec["acceptance_criteria"],
             "RETAINED_CORRECTION_RUNTIME_LINEAGE_CHANGED")
    outputs = build.get("verified_artifacts") or []
    _require(len(outputs) == 1 and prior.get("visualsignal_session_evidence", {}).get("candidate_ref") == {
        "path": outputs[0]["resolved_path"], "sha256": outputs[0]["sha256"]}, "RETAINED_CORRECTION_CANDIDATE_CHANGED")
    def binds(envelope, ref):
        return any(item.get("artifact_type") == ref.artifact_type and item.get("path") == ref.path
                   and item.get("artifact_hash") == ref.artifact_hash for item in envelope["inputs"]["upstream_artifacts"])
    _require(binds(envelopes["prior_eval_ref"], request.prior_runtime_ref)
             and binds(envelopes["prior_eval_ref"], request.build_result_ref)
             and binds(envelopes["prior_blocked_ref"], request.prior_eval_ref), "RETAINED_CORRECTION_TERMINAL_LINEAGE_CHANGED")
    evaluation = envelopes["prior_eval_ref"]["data"]
    gate = evaluation.get("evaluation_gate") or {}
    _require(envelopes["prior_runtime_ref"]["status"] == "success"
             and envelopes["prior_eval_ref"]["status"] == envelopes["prior_blocked_ref"]["status"] == "blocked"
             and evaluation.get("verdict") == "blocked" and evaluation.get("v1_acceptance_status") == "accepted"
             and gate.get("status") == "failed" and len(gate.get("blockers", [])) == 1
             and gate["blockers"][0].get("code") == GATE_CODE
             and gate["blockers"][0].get("message") == "Runtime claims real success without required binding fields: plugin_marker_binding."
             and [row["case_id"] for row in gate["cases"] if row["status"] != "passed"] == ["runtime_claim_boundary_binding"]
             and [row["code"] for row in envelopes["prior_blocked_ref"]["data"]["blockers"]] == [GATE_CODE],
             "RETAINED_CORRECTION_NOT_EXACT_BINDING_FAILURE")
    fresh = verifier(feature, build, feature_spec)
    verifier.validate_admission(admission)
    _require(fresh == prior, "RETAINED_CORRECTION_FRESH_VERIFIER_DIFFERS_FROM_PREDECESSOR")
    corrected = project_standard_bindings(prior, request)
    return {"checkpoint": checkpoint, "report": report, "payload": corrected,
            "source_snapshot": deepcopy(snapshot), "admission": deepcopy(admission)}
