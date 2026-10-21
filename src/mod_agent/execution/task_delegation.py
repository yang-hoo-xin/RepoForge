"""Exact host-issued delegation for one disposable, compile-only product run.

This module is not a model tool or a protection boundary against arbitrary code
on the trusted host. A host caller selects the task policy and required checks.
为什么要有它：任务委托是明确的执行授权来源，不能冒充人工语义审阅或由模型字段自行授予。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ORIGIN = "task_delegation_policy"
SCHEMA = "task_delegation_decision_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_KINDS = ("patch_proposal", "proposal_review", "implementation_plan", "dry_run_result")
_FIELDS = {"schema_version", "origin", "decision", "policy_id", "created_at", "scope",
           "checkpoint_ref", "request", "artifact_refs", "dry_run_binding",
           "materialized_code_identity", "verification_report_ref", "required_check_ids"}


def _require(value, code):
    if not value:
        raise ValueError("TASK_DELEGATION_" + code)


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _path(value):
    _require(isinstance(value, str) and Path(value).is_absolute()
             and not value.startswith(("\\\\", "//")), "LOCAL_ABSOLUTE_PATH_REQUIRED")
    path = Path(value)
    _require(not path.is_symlink() and str(path.resolve()) == value, "NORMALIZED_PATH_REQUIRED")
    return path


def _read(reference):
    _require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256"}
             and isinstance(reference["sha256"], str) and _SHA.fullmatch(reference["sha256"]),
             "EXACT_REFERENCE_REQUIRED")
    raw = _path(reference["path"]).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "REFERENCED_BYTES_CHANGED")
    return json.loads(raw)


def _verify_file(reference, root):
    _require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256"}
             and isinstance(reference["sha256"], str) and _SHA.fullmatch(reference["sha256"]),
             "EXACT_REFERENCE_REQUIRED")
    path = _path(reference["path"])
    _require(path.is_relative_to(root) and hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"],
             "VERIFIED_OUTPUT_CHANGED")


def _bundle(artifact_store_root, checkpoint_ref):
    from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
    from src.mod_agent.product_execution import ProductExecutionRequestV1

    _require(isinstance(checkpoint_ref, ArtifactRef)
             and checkpoint_ref.artifact_type == "workflow_checkpoint", "CHECKPOINT_REQUIRED")
    store = ArtifactStore(artifact_store_root)
    checkpoint = store.semantic_read_immutable(checkpoint_ref)["data"]
    state = checkpoint["state_snapshot"]
    _require(checkpoint["checkpoint_stage"] == state["current_stage"] == "dry_run_apply",
             "DEFERRED_DRY_RUN_REQUIRED")
    request = ProductExecutionRequestV1.from_dict(
        json.loads(state["runtime_summary"]["deferred_product_request_json"]))
    _require(request.run_id == checkpoint["run_id"] and request.request_id == checkpoint["request_id"]
             and str(store.root.resolve()) == request.artifact_store_root, "RUN_IDENTITY_MISMATCH")
    refs = {kind: ArtifactRef(feature_id=checkpoint_ref.feature_id, artifact_type=kind,
            path=state["artifacts"][kind], artifact_hash=state["artifact_hashes"][kind]) for kind in _KINDS}
    documents = {kind: store.semantic_read_immutable(reference)["data"] for kind, reference in refs.items()}
    return request, refs, documents


def _scope(request, *, live=True):
    source, root, target, store = ((_path if live else Path)(getattr(request, name)) for name in
        ("source_root", "isolated_workspace_root", "isolated_target_root", "artifact_store_root"))
    _require(request.requested_completion_scope == "compile_only"
             and request.confirmed_isolated_write is True and request.allow_build is True,
             "COMPILE_ONLY_AUTHORIZATION_REQUIRED")
    _require((not live or all(path.is_dir() for path in (source, root, target, store)))
             and target != root and target.is_relative_to(root) and store.is_relative_to(root)
             and not source.is_relative_to(root) and not root.is_relative_to(source),
             "DISPOSABLE_ISOLATION_REQUIRED")
    return {"kind": "disposable_isolated_evaluation", "workspace_root": str(root),
            "allowed_source_directory": str(target), "allowed_effects": ["isolated_apply", "build"],
            "runtime": False, "deployment": False, "game_launch": False}


def _check_sources(request, *, before_apply):
    from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256

    _require(source_snapshot_sha256(source_snapshot(Path(request.source_root)))
             == request.source_snapshot_sha256, "ORIGINAL_SOURCE_CHANGED")
    if before_apply:
        _require(source_snapshot_sha256(source_snapshot(Path(request.isolated_target_root)))
                 == request.isolated_source_snapshot_sha256, "ISOLATED_SOURCE_CHANGED")


def _code_identity(materialized):
    return {"baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256,
            "files": sorted(materialized.files)}


def _native_verification_binding(*, request, plan_ref, code_identity, proposal, reference, report,
                                 recorded_binding=None, before_apply=True):
    """Bind exact native origin before Apply, then recheck its immutable witnesses.

    为什么要有它：原生工具响应不使用旧worker的generation字段；Apply后的来源复核也不能把已修改源码误当成B0再次物化。
    """
    from src.mod_agent.artifacts import ArtifactStore
    from src.mod_agent.proposal.builder import PatchProposalBuilder
    from src.mod_agent.task_agent_execution import (
        load_task_agent_outcome,
        validate_task_agent_handoff,
    )

    root, store = Path(request.isolated_workspace_root), ArtifactStore(request.artifact_store_root)
    lineage = proposal.get("context_pack", {}).get("task_agent_lineage")
    _require(isinstance(lineage, dict) and set(lineage) == {
        "execution_ref", "candidate_ref", "check_ref", "approval_inherited", "whole_task_delivered"}
        and lineage["approval_inherited"] is False and lineage["whole_task_delivered"] is False,
        "NATIVE_LINEAGE_REQUIRED")
    outcome = load_task_agent_outcome(lineage["execution_ref"], artifact_store_root=store.root)
    execution = _read(lineage["execution_ref"])
    _require(execution.get("run_id") == request.run_id and execution.get("request_id") == request.request_id
        and outcome.status == "candidate_ready" and not outcome.unknown_effects and not outcome.whole_task_delivered
        and outcome.candidate_ref == lineage["candidate_ref"] and outcome.check_ref == lineage["check_ref"]
        and asdict(outcome.final_context.implementation_plan_ref) == plan_ref, "NATIVE_EXECUTION_CHANGED")
    candidate, check = _read(outcome.candidate_ref), _read(outcome.check_ref)
    _require(check.get("report", {}).get("report_ref") == reference
        and candidate.get("materialized_code_identity") == code_identity
        and candidate.get("context") == outcome.final_context.to_dict()
        and report.get("model_response_sha256") == candidate.get("model_origin", {}).get("derived_content_ref", {}).get("sha256")
        and proposal["diff_intent"] == [PatchProposalBuilder._change_intent(row)
            for row in candidate.get("generated_changes", [])]
        and proposal["context_pack"].get("implementation_plan_hash") == plan_ref["artifact_hash"],
        "NATIVE_CANDIDATE_CHANGED")
    identity = {"schema_version": "native-delegation-binding.v1", "execution_ref": lineage["execution_ref"],
        "candidate_ref": outcome.candidate_ref, "check_ref": outcome.check_ref,
        "verification_report_ref": dict(reference), "model_response_sha256": report["model_response_sha256"],
        "materialized_code_identity": code_identity, "implementation_plan_ref": plan_ref}
    if not before_apply:
        _require(isinstance(recorded_binding, dict)
            and set(recorded_binding) == {*identity, "immutable_file_refs"}
            and all(recorded_binding.get(key) == value for key, value in identity.items())
            and isinstance(recorded_binding["immutable_file_refs"], list)
            and bool(recorded_binding["immutable_file_refs"]), "NATIVE_ADMISSION_WITNESS_REQUIRED")
        for selected in recorded_binding["immutable_file_refs"]:
            _require(isinstance(selected, dict) and set(selected) == {"path", "sha256"}
                and hashlib.sha256(_path(selected["path"]).read_bytes()).hexdigest() == selected["sha256"],
                "NATIVE_IMMUTABLE_ORIGIN_CHANGED")
        return recorded_binding
    handoff = validate_task_agent_handoff(outcome, store=store, workspace_root=root,
        expected_initial_context=outcome.initial_context, allowed_edit_paths=tuple(execution["allowed_edit_paths"]),
        worker_configuration_ref=execution["worker_binding"]["configuration_ref"],
        expected_verification_identity=execution["expected_verification_identity"],
        expected_candidate_mode=execution.get("candidate_mode", "baseline"))
    _require(handoff.candidate == candidate and handoff.check == check, "NATIVE_HANDOFF_CHANGED")
    references = [outcome.execution_ref, outcome.claim_ref, outcome.loop_result_ref, outcome.check_ref,
        outcome.finish_response_ref, reference, execution["worker_binding"]["configuration_ref"],
        execution["worker_binding"]["factory_ref"]]
    for context in (outcome.initial_context, outcome.final_context):
        for selected in context.to_dict().values():
            path = _path(selected["path"])
            references.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for call in execution.get("completed_tool_records", []):
        if call.get("role") in {"candidate_submit", "candidate_check"}:
            references.extend([call["response_ref"], call["arguments_ref"]])
    selected_ref, visited = outcome.candidate_ref, set()
    while selected_ref is not None:
        _require(selected_ref["path"] not in visited and _path(selected_ref["path"]).is_relative_to(root),
            "NATIVE_PARENT_CYCLE_OR_SCOPE")
        visited.add(selected_ref["path"])
        selected = _read(selected_ref)
        origin = selected["model_origin"]
        references.extend([selected_ref, origin["arguments_ref"], origin["response_ref"], origin["derived_content_ref"]])
        selected_ref = origin.get("parent_candidate_ref")
    unique = {}
    for selected in references:
        path = _path(selected["path"])
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == selected["sha256"], "NATIVE_ORIGIN_CHANGED")
        unique[str(path)] = dict(selected)
    identity["immutable_file_refs"] = [unique[key] for key in sorted(unique)]
    _require(recorded_binding is None or recorded_binding == identity, "NATIVE_ADMISSION_CHANGED")
    return identity


def _verification(reference, required_checks, *, request, plan_ref, code_identity, proposal,
                  native_binding=None, before_apply=True):
    root = Path(request.isolated_workspace_root)
    _require(_path(reference["path"]).is_relative_to(root), "VERIFICATION_OUTSIDE_RUN")
    report = _read(reference)
    _require(isinstance(required_checks, list) and required_checks
             and all(isinstance(x, str) and x for x in required_checks)
             and len(required_checks) == len(set(required_checks)), "CHECK_SET_REQUIRED")
    summary = report.get("verification_summary", {})
    _require(report.get("status") == "passed" and report.get("execution_origin") == "executed"
             and report.get("failure_owner") is None and report.get("observations") == []
             and report.get("materialized_code_identity") == code_identity
             and report.get("implementation_plan_ref") == plan_ref
             and summary.get("checks") == len(required_checks) and summary.get("failed_checks") == 0
             and type(summary.get("executions")) is int and summary["executions"] > 0,
             "VERIFICATION_NOT_PASSED_OR_BOUND")
    native = proposal.get("context_pack", {}).get("task_agent_lineage") is not None
    if native:
        native_binding = _native_verification_binding(request=request, plan_ref=plan_ref,
            code_identity=code_identity, proposal=proposal, reference=reference, report=report,
            recorded_binding=native_binding, before_apply=before_apply)
    else:
        _require(native_binding is None, "NATIVE_BINDING_WITHOUT_LINEAGE")
        response_hashes = {row.get("generation", {}).get("model_response_sha256")
                           for row in proposal["diff_intent"]}
        _require(len(response_hashes) == 1 and None not in response_hashes
                 and isinstance(report.get("model_response_sha256"), str)
                 and _SHA.fullmatch(report["model_response_sha256"])
                 and report.get("model_response_sha256") in response_hashes, "MODEL_RESPONSE_MISMATCH")
    detail_ref = report.get("verification_ref")
    _require(isinstance(detail_ref, Mapping) and _path(detail_ref["path"]).is_relative_to(root),
             "CHECK_DETAILS_REQUIRED")
    detail = _read(detail_ref)
    rows = detail.get("checks", [])
    _require(detail.get("status") == "passed" and detail.get("execution_origin") == "executed"
             and detail.get("materialized_code_identity") == code_identity
             and [row.get("check_id", row.get("check")) for row in rows] == required_checks
             and all(row.get("passed") is True for row in rows), "CHECK_DETAILS_NOT_PASSED")
    compilation = report.get("validation_results", {}).get("full_compilation", {})
    _require(compilation.get("status") == "passed" and compilation.get("all_required_files_compiled") is True,
             "FULL_COMPILATION_REQUIRED")
    compile_ref = compilation.get("report_ref")
    _require(isinstance(compile_ref, Mapping) and _path(compile_ref["path"]).is_relative_to(root),
             "COMPILATION_REFERENCE_REQUIRED")
    compiled = _read(compile_ref)
    _require(compiled.get("status") == "passed" and compiled.get("execution_origin") == "executed"
             and compiled.get("all_required_files_compiled") is True
             and compiled.get("materialized_code_identity") == code_identity,
             "COMPILATION_NOT_BOUND")
    _verify_file(compiled.get("built_artifact_ref"), root)
    _require(detail.get("built_artifact_ref") == compiled["built_artifact_ref"]
             and detail.get("compilation_report_ref") == compile_ref,
             "CHECKS_DO_NOT_BIND_COMPILED_OUTPUT")
    return native_binding


def prepare_task_delegation(*, artifact_store_root, checkpoint_ref, verification_report_ref,
                            required_check_ids, disposable_run_root, policy_id):
    """Trusted host selection only: retain one exact decision, without Apply or Build."""
    from src.mod_agent.artifacts import ArtifactStore
    from src.mod_agent.execution.patch_apply import (
        ExactTextPatchApplier,
        materialize_exact_text_patch,
    )
    from src.mod_agent.workflow.checkpointing import WorkflowCheckpointStore

    request, refs, documents = _bundle(artifact_store_root, checkpoint_ref)
    scope = _scope(request)
    _require(str(_path(disposable_run_root)) == request.isolated_workspace_root, "RUN_ROOT_MISMATCH")
    _require(isinstance(policy_id, str) and policy_id.strip(), "HOST_POLICY_REQUIRED")
    allowed = WorkflowCheckpointStore(ArtifactStore(artifact_store_root)).preflight_resume(
        checkpoint_ref, resume_id="task-delegation-preparation")
    _require(allowed.allowed, "CHECKPOINT_NOT_RESUMABLE")
    _check_sources(request, before_apply=True)
    proposal, plan, review, dry = (documents[k] for k in
        ("patch_proposal", "implementation_plan", "proposal_review", "dry_run_result"))
    current = ExactTextPatchApplier().run(workspace_root=request.isolated_workspace_root,
        proposal=proposal, implementation_plan=plan, review=review, allow_write=False)
    keys = ("diff_hash", "unified_diff_sha256", "context_hashes")
    _require(all(current[key] == dry[key] for key in keys), "CURRENT_DIFF_CHANGED")
    materialized = materialize_exact_text_patch(workspace_root=request.isolated_workspace_root,
        diff_intent=proposal["diff_intent"], implementation_plan=plan)
    code = _code_identity(materialized)
    target = Path(request.isolated_target_root)
    _require(bool(code["files"]) and all((Path(request.isolated_workspace_root) / path).resolve().is_relative_to(target)
             for path in code["files"]), "CHANGE_OUTSIDE_ALLOWED_DIRECTORY")
    checks = list(required_check_ids)
    native = _verification(verification_report_ref, checks, request=request, plan_ref=asdict(refs["implementation_plan"]),
                           code_identity=code, proposal=proposal)
    record = {"schema_version": SCHEMA, "origin": ORIGIN, "decision": "approve", "policy_id": policy_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "scope": scope,
        "checkpoint_ref": asdict(checkpoint_ref), "request": request.to_dict(),
        "artifact_refs": {kind: asdict(value) for kind, value in refs.items()},
        "dry_run_binding": {key: dry[key] for key in keys}, "materialized_code_identity": code,
        "verification_report_ref": dict(verification_report_ref), "required_check_ids": checks}
    if native is not None:
        record["native_candidate_binding"] = native
    raw = _bytes(record)
    digest = hashlib.sha256(raw).hexdigest()
    path = Path(artifact_store_root) / "task_delegations" / (digest + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path.resolve()), "sha256": digest}


def validate_task_delegation(reference, *, workspace_root=None, diff_hash=None, context_hashes=None,
                             materialized_code_identity=None, before_apply=False, resume_report_ref=None,
                             fresh_verification=False):
    from src.mod_agent.artifacts import ArtifactRef

    record = _read(reference)
    _require(isinstance(record, dict) and set(record) in (_FIELDS, _FIELDS | {"native_candidate_binding"})
             and record["schema_version"] == SCHEMA
             and record["origin"] == ORIGIN and record["decision"] == "approve"
             and isinstance(record["policy_id"], str) and bool(record["policy_id"].strip())
             and isinstance(record["created_at"], str) and bool(record["created_at"]), "ORIGIN_OR_SCHEMA_INVALID")
    request, refs, documents = _bundle(record["request"]["artifact_store_root"], ArtifactRef(**record["checkpoint_ref"]))
    _require(record["request"] == request.to_dict()
             and record["scope"] == _scope(request, live=before_apply or fresh_verification)
             and record["artifact_refs"] == {k: asdict(v) for k, v in refs.items()}, "BINDING_CHANGED")
    _require((documents["patch_proposal"].get("context_pack", {}).get("task_agent_lineage") is not None)
        == ("native_candidate_binding" in record), "NATIVE_ADMISSION_WITNESS_REQUIRED")
    expected_path = Path(request.artifact_store_root) / "task_delegations" / (reference["sha256"] + ".json")
    _require(_path(reference["path"]) == expected_path, "DECISION_LOCATION_INVALID")
    dry = documents["dry_run_result"]
    _require(record["dry_run_binding"] == {k: dry[k] for k in ("diff_hash", "unified_diff_sha256", "context_hashes")},
             "DRY_RUN_BINDING_CHANGED")
    _require(bool(record["materialized_code_identity"].get("files")) and all(
        (Path(request.isolated_workspace_root) / path).resolve().is_relative_to(Path(request.isolated_target_root))
        for path in record["materialized_code_identity"]["files"]), "CHANGE_OUTSIDE_ALLOWED_DIRECTORY")
    if workspace_root is not None:
        _require(Path(workspace_root).resolve() == Path(request.isolated_workspace_root), "WORKSPACE_MISMATCH")
    if diff_hash is not None:
        _require(diff_hash == dry["diff_hash"], "DIFF_MISMATCH")
    if context_hashes is not None:
        _require(dict(context_hashes) == dry["context_hashes"], "CONTEXT_MISMATCH")
    if materialized_code_identity is not None:
        _require(dict(materialized_code_identity) == record["materialized_code_identity"], "CODE_MISMATCH")
    if before_apply or fresh_verification:
        # Historical artifact reads validate the immutable decision chain only.
        # 为什么要有它：旧编译输出或原工程的后续清理不能重写历史授权事实；真正执行前仍须重验实际输入与验收。
        _verification(record["verification_report_ref"], record["required_check_ids"], request=request,
            plan_ref=record["artifact_refs"]["implementation_plan"], code_identity=record["materialized_code_identity"],
            proposal=documents["patch_proposal"], native_binding=record.get("native_candidate_binding"),
            before_apply=before_apply)
        _check_sources(request, before_apply=before_apply)
    if before_apply:
        _validate_apply_resume_receipt(reference, record, resume_report_ref)
    return record


def _validate_apply_resume_receipt(reference, record, resume_reference):
    from src.mod_agent.artifacts import ArtifactRef, ArtifactStore

    _require(isinstance(resume_reference, Mapping), "RESUME_RECEIPT_REQUIRED")
    selected = ArtifactRef(**dict(resume_reference))
    _require(selected.artifact_type == "resume_report"
             and selected.feature_id == record["checkpoint_ref"]["feature_id"], "RESUME_RECEIPT_MISMATCH")
    envelope = ArtifactStore(record["request"]["artifact_store_root"]).semantic_read_immutable(selected)
    data = envelope["data"]
    _require(envelope["status"] == "success" and data.get("allowed") is True
             and data.get("decision") == "approve" and data.get("resume_stage") == "dry_run_apply"
             and data.get("run_id") == record["request"]["run_id"]
             and data.get("authority_request_hash") == reference["sha256"]
             and data.get("authorization_origin") == ORIGIN and data.get("task_delegation_ref") == reference
             and data.get("approved_by") == ORIGIN + ":" + reference["sha256"]
             and any(all(upstream.get(key) == record["checkpoint_ref"][key]
                         for key in ("artifact_type", "path", "artifact_hash"))
                     for upstream in envelope["inputs"]["upstream_artifacts"]), "RESUME_RECEIPT_MISMATCH")


def validate_delegated_resume(reference, payload, *, run_id, request_id, checkpoint_ref=None):
    record = validate_task_delegation(reference)
    _require(record["request"]["run_id"] == payload.run_id == run_id
             and record["request"]["request_id"] == payload.request_id == request_id
             and record["checkpoint_ref"]["feature_id"] == payload.feature_id,
             "RESUME_RUN_MISMATCH")
    _require(payload.decision == "approve" and payload.stage in {"dry_run_apply", "isolated_apply"}
             and payload.runtime_admission is None and payload.authority_request_hash == reference["sha256"]
             and payload.diff_hash == record["dry_run_binding"]["diff_hash"]
             and payload.approved_by == ORIGIN + ":" + reference["sha256"], "RESUME_AUTHORITY_MISMATCH")
    if checkpoint_ref is not None:
        _require(record["checkpoint_ref"] == asdict(checkpoint_ref), "RESUME_CHECKPOINT_MISMATCH")
    return record


def delegation_result_fields(reference, resume_report_ref):
    return {"authorization_origin": ORIGIN, "task_delegation_ref": dict(reference),
            "task_delegation_resume_ref": dict(resume_report_ref),
            "approved_by_user": False, "human_semantic_review_required": False,
            "human_semantic_review_completed": False, "human_semantic_review_result": None,
            "human_semantic_decision": None, "human_semantic_decision_binding": None}


def delegation_resume_report_fields(payload, checkpoint):
    """Keep the decision reference in the original ResumeReport, not in free-form notes."""
    if not isinstance(payload.approved_by, str) or not payload.approved_by.startswith(ORIGIN + ":"):
        return {}
    digest = payload.authority_request_hash
    _require(isinstance(digest, str) and _SHA.fullmatch(digest), "RESUME_AUTHORITY_MISMATCH")
    request = json.loads(checkpoint["state_snapshot"]["runtime_summary"]["deferred_product_request_json"])
    reference = {"path": str(Path(request["artifact_store_root"]) / "task_delegations" / (digest + ".json")),
                 "sha256": digest}
    validate_delegated_resume(reference, payload, run_id=checkpoint["run_id"], request_id=checkpoint["request_id"])
    return {"authorization_origin": ORIGIN, "task_delegation_ref": reference}


def has_verified_apply_authorization(payload, *, require_human_semantic=True, run_id=None, request_id=None):
    """Shared consumer check; an origin label or boolean never grants authority."""
    origin = payload.get("authorization_origin")
    if "authorization_origin" not in payload:
        _require("task_delegation_ref" not in payload and "task_delegation_resume_ref" not in payload,
                 "APPLY_ORIGIN_REQUIRED")
        return (payload.get("approved_by_user") is True
                and (not require_human_semantic or payload.get("human_semantic_review_completed") is True))
    _require(origin == ORIGIN and payload.get("approved_by_user") is False
             and payload.get("human_semantic_review_required") is False
             and payload.get("human_semantic_review_completed") is False
             and payload.get("human_semantic_decision") is None
             and payload.get("human_semantic_decision_binding") is None
             and payload.get("human_semantic_review_result") is None, "APPLY_PROVENANCE_INVALID")
    diff_hash = payload.get("diff_hash")
    context_hashes = payload.get("context_hashes")
    _require(isinstance(diff_hash, str) and diff_hash.startswith("sha256:") and _SHA.fullmatch(diff_hash[7:])
             and payload.get("approved_diff_hash") == diff_hash, "APPLY_DIFF_MISMATCH")
    _require(isinstance(context_hashes, Mapping) and context_hashes, "APPLY_CONTEXT_REQUIRED")
    record = validate_task_delegation(payload["task_delegation_ref"], diff_hash=diff_hash,
                                       context_hashes=context_hashes)
    _validate_apply_resume_receipt(payload["task_delegation_ref"], record,
                                 payload.get("task_delegation_resume_ref"))
    if run_id is not None:
        _require(record["request"]["run_id"] == run_id, "APPLY_RUN_MISMATCH")
    if request_id is not None:
        _require(record["request"]["request_id"] == request_id, "APPLY_REQUEST_MISMATCH")
    return True
