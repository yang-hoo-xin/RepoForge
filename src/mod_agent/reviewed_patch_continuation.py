"""Review retained exact artifacts and resume human-approved isolated application.

Read/preparation functions grant no approval. Explicit Apply and Build resume
decisions enter the original Graph effects through their existing contracts.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.execution.patch_apply import (
    PatchApplyError,
    _build_unified_diff,
    _diff_hash,
    _sha256_json,
    materialize_exact_text_patch,
)
from src.mod_agent.fresh_planning_continuation import (
    PATCH_CONTINUATION_RECEIPT_VERSION,
    _canonical_sha256,
)

_ARTIFACT_FIELDS = {"patch_proposal": "patch_proposal_artifact", "proposal_review": "proposal_review_artifact",
                    "dry_run_result": "dry_run_artifact"}
_REF_FIELDS = ("feature_id", "artifact_type", "path", "artifact_hash")
_INPUT_FIELDS = {"feature_spec": "feature_spec_artifact", "evidence_pack": "evidence_pack_artifact",
                 "implementation_plan": "implementation_plan_artifact"}


def continue_v2_delivery(*, artifact_store_root, checkpoint_ref, action, isolated_root=None,
        operation_id=None, displayed_binding=None, human_semantic_decision=None,
        task_delegation_ref=None, allow_build=False):
    """Normal Apply entry dispatch for v2, independent of the historical Plan.

    为什么要有它：同一用户入口需要显式协议消费，不能因恢复而把新候选伪装为旧Plan。
    """
    from src.mod_agent.task_agent_v2_delivery import apply_v2_delivery, build_v2_delivery, prepare_v2_delivery
    if action == "prepare":
        return prepare_v2_delivery(artifact_store_root=artifact_store_root, checkpoint_ref=checkpoint_ref,
            isolated_root=isolated_root, operation_id=operation_id)
    if action == "apply":
        if not isinstance(displayed_binding, dict) or displayed_binding.get("checkpoint_ref") != checkpoint_ref:
            raise ValueError("V2_DELIVERY_EXACT_DISPLAY_BINDING_REQUIRED")
        return apply_v2_delivery(artifact_store_root=artifact_store_root, displayed_binding=displayed_binding,
            human_semantic_decision=human_semantic_decision, task_delegation_ref=task_delegation_ref)
    if action == "build":
        return build_v2_delivery(artifact_store_root=artifact_store_root, apply_ref=checkpoint_ref,
            operation_id=operation_id, allow_build=allow_build, task_delegation_ref=task_delegation_ref)
    raise ValueError("V2_DELIVERY_ACTION_UNSUPPORTED")


def _ref(value, feature_id, kind):
    if (not isinstance(value, dict) or set(value) != set(_REF_FIELDS)
            or any(not isinstance(value[key], str) or not value[key] for key in _REF_FIELDS)):
        raise ValueError("REVIEW_ARTIFACT_REFERENCE_INVALID")
    result = {key: value[key] for key in _REF_FIELDS}
    if result["feature_id"] != feature_id or result["artifact_type"] != kind:
        raise ValueError("REVIEW_ARTIFACT_IDENTITY_MISMATCH")
    return result


def _read_retained_journal(store_root, feature_id, run_id):
    # Same locator as PatchContinuationAttemptJournal and its terminal-cache owner.
    path = Path(store_root).resolve() / "cr" / (hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32] + ".json")
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError("RUN_NOT_UNIQUE_OR_MISSING") from exc
    if (not isinstance(document, dict) or document.get("receipt_version") != PATCH_CONTINUATION_RECEIPT_VERSION
            or document.get("run_id") != run_id or document.get("feature_id") != feature_id):
        raise ValueError("RUN_NOT_UNIQUE_OR_MISSING")
    return document, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _verified_terminal(document, feature_id, run_id):
    terminal = document.get("terminal_result")
    if (not isinstance(terminal, dict) or document.get("terminal_result_sha256") != _canonical_sha256(terminal)
            or terminal.get("run_id") != run_id or terminal.get("feature_id") != feature_id
            or terminal.get("request_id") != document.get("request_id")):
        raise ValueError("TERMINAL_RECORD_UNVERIFIED")
    return terminal


def _terminal_ref(document, terminal, feature_id, kind, field):
    value = terminal.get(field)
    current = (document.get("artifacts") or {}).get(kind)
    if current is not None and (value is None or _ref(current, feature_id, kind) != _ref(value, feature_id, kind)):
        raise ValueError("CURRENT_TERMINAL_ARTIFACT_CONFLICT")
    return None if value is None else _ref(value, feature_id, kind)


def _recorded_behavior_summaries(document):
    evaluations = ((document.get("factory_receipt") or {}).get("patch_agent_loop") or {}).get("behavior_evaluations") or []
    fields = ("status", "failure_owner", "evaluation_index", "report_ref", "generated_changes_sha256",
              "model_response_sha256", "execution_origin", "callback_invoked")
    return [{key: row[key] for key in fields if key in row} for row in evaluations if isinstance(row, dict)]


def _has_upstream(document, ref):
    return any(isinstance(item, dict) and all(item.get(key) == ref[key]
        for key in ("artifact_type", "path", "artifact_hash"))
        for item in document.get("inputs", {}).get("upstream_artifacts", []))


def _source_comparison(workspace, dry_run, proposal, plan):
    files = {}
    for change in dry_run["changes"]:
        relative = change["path"]
        target = (workspace / relative).resolve()
        if Path(relative).is_absolute() or not target.is_relative_to(workspace):
            raise ValueError("REVIEW_SOURCE_PATH_OUTSIDE_CONFIGURED_WORKSPACE")
        if relative in files:
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        expected = change["raw_before_sha256"]
        files[relative] = {"path": relative, "expected_baseline_sha256": expected,
                           "current_sha256": actual, "matches": expected == actual}
    result = {"scope": "changed_files_and_plan_bound_in_memory_materialization_only",
              "files": list(files.values()), "first_difference": next((row for row in files.values() if not row["matches"]), None),
              "source_tree_wide_freshness": "not_checked", "materializer_executed_in_memory": False,
              "approval_or_apply_eligibility": "not_granted_by_this_view"}
    if result["first_difference"] is not None:
        return {**result, "status": "source_changed", "diff_recomputed_matches": None}
    try:
        materialized = materialize_exact_text_patch(workspace_root=workspace,
            diff_intent=proposal["diff_intent"], implementation_plan=plan)
        rendered = _build_unified_diff(root=workspace,
            original_by_target={workspace / path: raw for path, raw in materialized.baseline_files.items()},
            staged={workspace / path: raw for path, raw in materialized.files.items()})
        matches = rendered == dry_run["unified_diff"]
        return {**result, "status": "matches_original_baseline" if matches else "materialized_diff_mismatch",
                "materializer_executed_in_memory": True, "diff_recomputed_matches": matches,
                "baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256}
    except (PatchApplyError, OSError, ValueError):
        return {**result, "status": "source_or_binding_check_failed", "diff_recomputed_matches": None}


def review_run(*, artifact_store_root: str, workspace_root: str, feature_id: str, run_id: str) -> dict:
    for value in (feature_id, run_id):
        if not isinstance(value, str) or not value.strip() or len(value) > 300 or any(c in value for c in "\r\n\0"):
            raise ValueError("REVIEW_ID_INVALID")
    workspace, store_root = Path(workspace_root), Path(artifact_store_root)
    if not workspace.is_absolute() or not workspace.is_dir() or not store_root.is_absolute() or not store_root.is_dir():
        raise ValueError("REVIEW_CONFIGURED_ROOT_UNAVAILABLE")
    workspace = workspace.resolve()
    document, journal_ref = _read_retained_journal(store_root, feature_id, run_id)
    terminal = document.get("terminal_result") or {}
    result = {"entry": "review-run", "read_only": True, "run_id": run_id, "feature_id": feature_id,
              "journal_ref": journal_ref, "artifacts": {}, "unified_diff": None,
              "source_currentness": {"status": "not_checked"},
              "recorded_stage": terminal.get("terminal_stage") if isinstance(terminal, dict) else None,
              "recorded_status": terminal.get("status") if isinstance(terminal, dict) else None,
              "behavior_verification_recorded": _recorded_behavior_summaries(document),
              "query_model_calls": 0, "query_verifier_executions": 0, "product_source_writes_performed": False,
              "artifact_writes_performed": False, "approval_created": False, "apply_allowed": False,
              "product_execution_started": False, "review_is_historical_not_current_apply_approval": True}
    if "saved_patch" in document:
        result["saved_patch_reassembly"] = {
            key: document.get(key) for key in (
                "generation_origin", "saved_patch", "inherited_model_budget", "logical_budget",
                "new_provider_sends", "new_compilations", "parent_run_id",
            )
        }
    try:
        terminal = _verified_terminal(document, feature_id, run_id)
        for kind, field in _ARTIFACT_FIELDS.items():
            selected = _terminal_ref(document, terminal, feature_id, kind, field)
            if selected is not None:
                result["artifacts"][kind] = selected
        if "patch_proposal" not in result["artifacts"]:
            return {**result, "status": "no_reviewable_diff", "reason": "THIS_RUN_HAS_NO_NEW_PROPOSAL"}
        if len(result["artifacts"]) != 3:
            return {**result, "status": "no_reviewable_diff", "reason": "THIS_RUN_HAS_NO_COMPLETE_REVIEWED_DRY_RUN"}
        store = ArtifactStore(store_root)
        documents = {kind: store.semantic_read_immutable(ArtifactRef(**ref)) for kind, ref in result["artifacts"].items()}
        proposal, review, dry_run = (documents[kind]["data"] for kind in _ARTIFACT_FIELDS)
        if (not _has_upstream(documents["proposal_review"], result["artifacts"]["patch_proposal"])
                or not all(_has_upstream(documents["dry_run_result"], result["artifacts"][kind])
                           for kind in ("patch_proposal", "proposal_review"))):
            raise ValueError("REVIEW_DRY_RUN_LINEAGE_MISMATCH")
        plan_ref = _terminal_ref(document, terminal, feature_id, "implementation_plan", "implementation_plan_artifact")
        if not _has_upstream(documents["patch_proposal"], plan_ref):
            raise ValueError("PROPOSAL_PLAN_LINEAGE_MISMATCH")
        plan = store.semantic_read_immutable(ArtifactRef(**plan_ref))["data"]
        context = dry_run["context_hashes"]
        if any(context.get(key) != _sha256_json(value) for key, value in (
                ("proposal_payload_sha256", proposal), ("proposal_review_payload_sha256", review),
                ("implementation_plan_payload_sha256", plan))):
            raise ValueError("DRY_RUN_CONTEXT_HASH_MISMATCH")
        raw = dry_run["unified_diff"].encode("utf-8")
        if (hashlib.sha256(raw).hexdigest() != dry_run["unified_diff_sha256"] or len(raw) != dry_run["unified_diff_bytes"]
                or dry_run["diff_hash"] != _diff_hash(proposal_id=proposal["proposal_id"],
                    diff_preview=dry_run["diff_preview"], context_hashes=context)):
            raise ValueError("DRY_RUN_DIFF_IDENTITY_MISMATCH")
        result.update(status="reviewable_diff", unified_diff=dry_run["unified_diff"], diff_hash=dry_run["diff_hash"],
            unified_diff_sha256=dry_run["unified_diff_sha256"], diff_stats=dry_run["diff_stats"],
            original_review={key: review.get(key) for key in ("decision", "allowed_next_stage", "findings", "required_changes",
                "human_semantic_review_required", "semantic_assurance")},
            original_dry_run={key: dry_run.get(key) for key in ("execution_mode", "actual_write_performed", "human_semantic_review_completed",
                "human_semantic_review_result", "approved_by_user", "static_checks", "limitations")},
            source_currentness=_source_comparison(workspace, dry_run, proposal, plan),
            stage_limits={"human_semantic_review": "not_performed_by_this_view", "exact_diff_approval": "not_granted_by_this_view",
                "apply_build_runtime": "not_executed_by_this_view", "historical_execution_counts": terminal["execution_counts"]})
        return result
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {**result, "status": "review_unavailable", "unified_diff": None,
                "reason": str(exc) if isinstance(exc, ValueError) and str(exc).isupper() else "EXACT_REVIEW_RECORD_INVALID"}


_COPY_IGNORED = frozenset({"bin", "obj", ".git", ".idea"})


def _copy_source_snapshot(root, *, excluded_copy_paths=()):
    from src.mod_agent.product_executor import source_snapshot

    excluded = []
    for relative in excluded_copy_paths:
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or path == root or not path.is_relative_to(root):
            raise ValueError("APPLY_COPY_EXCLUSION_OUTSIDE_WORKSPACE")
        excluded.append(path)
    files = []
    for path in root.rglob("*"):
        if any(part in _COPY_IGNORED for part in path.relative_to(root).parts):
            continue
        if any(path == omitted or path.is_relative_to(omitted) for omitted in excluded):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("APPLY_COPY_LINK_OUTSIDE_SOURCE")
        if excluded and path.is_file():
            raw = path.read_bytes()
            files.append({"path": path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)})
    if excluded:
        return sorted(files, key=lambda row: row["path"])
    return [row for row in source_snapshot(root)
            if not any(part in _COPY_IGNORED for part in Path(row["path"]).parts)]


def _apply_inputs(*, artifact_store_root, workspace_root, feature_id, run_id):
    from src.mod_agent.execution.patch_apply import ExactTextPatchApplier
    from src.mod_agent.fresh_planning_continuation import (
        PATCH_CONTINUATION_CONTRACT_VERSION,
        PatchContinuationInputV1,
        preflight_patch_continuation,
    )

    viewed = review_run(artifact_store_root=artifact_store_root, workspace_root=workspace_root,
                       feature_id=feature_id, run_id=run_id)
    if viewed["status"] != "reviewable_diff":
        raise ValueError("APPLY_EXACT_REVIEW_UNAVAILABLE")
    if viewed["source_currentness"]["status"] != "matches_original_baseline":
        raise ValueError("APPLY_REVIEW_SOURCE_CHANGED")
    document, journal_ref = _read_retained_journal(artifact_store_root, feature_id, run_id)
    terminal = _verified_terminal(document, feature_id, run_id)
    if journal_ref != viewed["journal_ref"]:
        raise ValueError("APPLY_REVIEW_RECORD_CHANGED")
    if (terminal["status"] != "terminal" or terminal["terminal_stage"] != "dry_run_apply"
            or terminal.get("first_failure") is not None or document.get("proposal_admitted") is not True):
        raise ValueError("APPLY_RETAINED_RUN_NOT_READY")
    selected = {kind: ArtifactRef(**_terminal_ref(document, terminal, feature_id, kind, field))
                for kind, field in _INPUT_FIELDS.items()}
    store = ArtifactStore(artifact_store_root)
    saved_selection = None
    if document.get("continuation_selection") == "saved_generated_patch_reassembly":
        saved = document.get("saved_patch") or {}
        budget = document.get("logical_budget") or {}
        if (document.get("generation_origin") != "reused_saved_response"
                or not isinstance(saved.get("source_attempt_ref"), dict)
                or not isinstance(budget.get("logical_budget_ref"), dict)):
            raise ValueError("APPLY_SAVED_SOURCE_SELECTION_MISSING")
        saved_selection = {"source_attempt_ref": saved["source_attempt_ref"],
                           "logical_budget_ref": budget["logical_budget_ref"]}
    request = PatchContinuationInputV1(contract_version=PATCH_CONTINUATION_CONTRACT_VERSION,
        artifact_store_root=artifact_store_root, workspace_root=workspace_root,
        feature_spec_ref=selected["feature_spec"], evidence_pack_ref=selected["evidence_pack"],
        implementation_plan_ref=selected["implementation_plan"], saved_patch_selection=saved_selection,
        # This is a read-only re-admission of the inspected run, not a new
        # reassembly execution or a claim about its original CLI attempt label.
        attempt_id=run_id if saved_selection is not None else None)
    admitted = preflight_patch_continuation(request, artifact_store=store)
    refs = {kind: ArtifactRef(**ref) for kind, ref in viewed["artifacts"].items()}
    documents = {kind: store.semantic_read_immutable(ref) for kind, ref in refs.items()}
    if any(document["status"] != "success" for document in documents.values()):
        raise ValueError("APPLY_ARTIFACT_STATUS_NOT_SUCCESS")
    payloads = {kind: document["data"] for kind, document in documents.items()}
    if admitted.saved_patch is not None:
        from src.mod_agent.proposal.builder import PatchProposalBuilder
        expected_changes = [PatchProposalBuilder._change_intent(change) for change in admitted.saved_patch["changes"]]
        if payloads["patch_proposal"]["diff_intent"] != expected_changes:
            raise ValueError("APPLY_SAVED_PATCH_DIFF_CHANGED")
    dry = ExactTextPatchApplier().run(workspace_root=workspace_root,
        proposal=payloads["patch_proposal"], implementation_plan=admitted.implementation_plan,
        review=payloads["proposal_review"], allow_write=False)
    if dry["diff_hash"] != viewed["diff_hash"] or dry["unified_diff_sha256"] != viewed["unified_diff_sha256"]:
        raise ValueError("APPLY_CURRENT_DIFF_DIFFERS_FROM_DISPLAYED")
    if dry["human_semantic_review_required"] is not True:
        raise ValueError("APPLY_STRUCTURED_SEMANTIC_REVIEW_REQUIRED")
    return viewed, store, admitted, {**selected, **refs}, payloads, dry


def prepare_reviewed_patch_apply(*, artifact_store_root, workspace_root, feature_id, run_id,
                                isolated_root, operation_id):
    """Prepare a human action using current original contracts, without writes."""
    if (not isinstance(operation_id, str) or not operation_id.strip() or len(operation_id) > 200
            or any(char in operation_id for char in "\r\n\0")):
        raise ValueError("APPLY_OPERATION_ID_INVALID")
    source = Path(workspace_root).resolve()
    destination_root = Path(isolated_root)
    if not destination_root.is_absolute() or not destination_root.is_dir():
        raise ValueError("APPLY_CONFIGURED_ISOLATED_ROOT_UNAVAILABLE")
    destination_root = destination_root.resolve()
    store_root = Path(artifact_store_root).resolve()
    if (destination_root.is_relative_to(source) or source.is_relative_to(destination_root)
            or destination_root.is_relative_to(store_root) or store_root.is_relative_to(destination_root)):
        raise ValueError("APPLY_SOURCE_STORE_DESTINATION_OVERLAP")
    viewed, _store, _admitted, _refs, _payloads, dry = _apply_inputs(
        artifact_store_root=artifact_store_root, workspace_root=str(source), feature_id=feature_id, run_id=run_id)
    destination = destination_root / hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:32]
    if destination.exists():
        raise ValueError("APPLY_OPERATION_DESTINATION_ALREADY_EXISTS")
    excluded_copy_paths = (
        (store_root.relative_to(source).as_posix(),)
        if _admitted.saved_patch is not None and store_root.is_relative_to(source) else ()
    )
    snapshot = _copy_source_snapshot(source, excluded_copy_paths=excluded_copy_paths)
    binding = {"run_id": run_id, "feature_id": feature_id, "operation_id": operation_id,
        "journal_sha256": viewed["journal_ref"]["sha256"], "diff_hash": dry["diff_hash"],
        "unified_diff_sha256": dry["unified_diff_sha256"],
        "source_snapshot_sha256": _sha256_json(snapshot), "workspace_root": str(source),
        "isolated_root": str(destination_root), "isolated_workspace": str(destination),
        "human_semantic_decision_binding": dry["human_semantic_decision_binding"]}
    if excluded_copy_paths:
        binding["excluded_copy_paths"] = list(excluded_copy_paths)
    return {**viewed, "apply_preparation": {"binding": binding,
        "write_target": str(destination), "copy_scope": "configured_workspace_tree",
        "copy_exclusions": sorted(_COPY_IGNORED), "source_file_count": len(snapshot),
        "source_root_provenance": "configured_source_verified_against_retained_baseline",
        "stop_before": "build", "build_runtime_started": False,
        "instruction": "Review this exact diff and isolation target, then explicitly approve or reject with your reviewer identity."}}


def apply_reviewed_patch(*, artifact_store_root, workspace_root, feature_id, run_id,
                         isolated_root, displayed_binding, human_semantic_decision,
                         execution_trace_configuration=None):
    """Trace this approval invocation using the original prospective trace owner."""
    from src.mod_agent.patch_execution_trace import PatchExecutionTraceConfiguration

    if not isinstance(displayed_binding, dict) or not isinstance(displayed_binding.get("operation_id"), str):
        raise ValueError("APPLY_DISPLAY_BINDING_REQUIRED")
    observations = {}
    configuration = execution_trace_configuration or PatchExecutionTraceConfiguration()
    logical_request_id = _retained_trace_request_id(
        artifact_store_root, feature_id, run_id, displayed_binding.get("journal_sha256"))
    trace = configuration.make_trace(run_id=_apply_run_id(displayed_binding["operation_id"]),
        feature_id=feature_id, artifact_store_root=artifact_store_root,
        recorder=lambda row: observations.update({row["id"]: row}),
        **({"request_id": logical_request_id} if logical_request_id is not None else {}))
    failure = None
    try:
        with trace.activate():
            result = _apply_reviewed_patch_once(artifact_store_root=artifact_store_root,
                workspace_root=workspace_root, feature_id=feature_id, run_id=run_id,
                isolated_root=isolated_root, displayed_binding=displayed_binding,
                human_semantic_decision=human_semantic_decision)
            trace.update(workflow_status=result["status"],
                terminal_stage="isolated_apply" if result["isolated_workspace_created"] else "human_decision")
    except BaseException as error:
        failure = error
        raise
    finally:
        summary = configuration.finish(trace)
        summary.update(trace_id=trace.context.trace_id, session_id=trace.session_id, environment=trace.environment)
        if failure is not None:
            # Preserve measured failure facts without changing or swallowing the
            # original exception, and without writing a rejected action to disk.
            try:
                failure.execution_trace = summary
                failure.execution_observations = observations
            except Exception:
                pass
    result.update(execution_trace=summary, execution_observations=observations)
    if result["isolated_workspace_created"]:
        path = Path(result["isolated_workspace"]) / "reviewed-apply-result.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True)
    return result


def _retained_trace_request_id(store_root, feature_id, run_id, journal_sha256):
    """Read correlation only from the already displayed exact original journal.

    Old feature-based traces retain that identity; this does not infer a logical
    task from a newly generated continuation request identifier.
    """
    try:
        document, ref = _read_retained_journal(store_root, feature_id, run_id)
        _verified_terminal(document, feature_id, run_id)
        logical = (document.get("execution_trace") or {}).get("logical_request_id")
        if (isinstance(logical, str) and logical and ref["sha256"] == journal_sha256
                and logical == document["request_id"]):
            return logical
    except Exception:
        pass
    return None


def _apply_run_id(operation_id):
    return "reviewed_apply_" + hashlib.sha256(operation_id.encode()).hexdigest()[:24]


def _apply_reviewed_patch_once(*, artifact_store_root, workspace_root, feature_id, run_id,
                              isolated_root, displayed_binding, human_semantic_decision):
    """Re-admit a displayed action, then use the existing Graph and real Applier.

    This boundary stops before Build; it never updates the source workspace.
    The passed human decision is the existing human_semantic_decision_v1 object.
    """
    from src.mod_agent.execution.patch_apply import (
        ExactTextPatchApplier,
        _validate_human_semantic_decision,
    )
    from src.mod_agent.workflow.checkpointing import WorkflowCheckpointStore
    from src.mod_agent.workflow.graph import build_graph
    from src.mod_agent.workflow.state import create_initial_state

    if not isinstance(displayed_binding, dict):
        raise ValueError("APPLY_DISPLAY_BINDING_REQUIRED")
    prepared = prepare_reviewed_patch_apply(artifact_store_root=artifact_store_root,
        workspace_root=workspace_root, feature_id=feature_id, run_id=run_id, isolated_root=isolated_root,
        operation_id=displayed_binding.get("operation_id"))
    current = prepared["apply_preparation"]["binding"]
    if displayed_binding != current:
        raise ValueError("APPLY_DISPLAY_BINDING_STALE")
    decision = _validate_human_semantic_decision(human_semantic_decision,
        expected_binding=current["human_semantic_decision_binding"])
    base_result = {"entry": "apply-patch", "run_id": run_id, "feature_id": feature_id,
        "operation_id": current["operation_id"], "diff_hash": current["diff_hash"],
        "source_workspace_written": False, "model_calls_performed": 0, "builds_performed": 0,
        "runtime_executions_performed": 0, "stopped_before": "build", "human_decision": decision}
    if decision["decision"] == "reject":
        return {**base_result, "status": "rejected", "product_source_writes_performed": False,
                "artifact_writes_performed": False, "isolated_workspace_created": False}
    _, store, admitted, refs, payloads, _ = _apply_inputs(artifact_store_root=artifact_store_root,
        workspace_root=workspace_root, feature_id=feature_id, run_id=run_id)
    source, destination = Path(workspace_root).resolve(), Path(current["isolated_workspace"])
    excluded_copy_paths = tuple(current.get("excluded_copy_paths", ()))
    before = _copy_source_snapshot(source, excluded_copy_paths=excluded_copy_paths)
    if _sha256_json(before) != current["source_snapshot_sha256"]:
        raise ValueError("APPLY_SOURCE_CHANGED_BEFORE_COPY")
    ignored_names = shutil.ignore_patterns(*sorted(_COPY_IGNORED))
    def ignored(directory, names):
        excluded = set(ignored_names(directory, names))
        for name in names:
            relative = (Path(directory) / name).relative_to(source).as_posix()
            if relative in excluded_copy_paths:
                excluded.add(name)
        return excluded
    shutil.copytree(source, destination, ignore=ignored)
    if (_copy_source_snapshot(destination) != before
            or _copy_source_snapshot(source, excluded_copy_paths=excluded_copy_paths) != before):
        raise ValueError("APPLY_ISOLATED_COPY_SOURCE_MISMATCH")
    applier = ExactTextPatchApplier()

    def dry_run_factory(proposal, implementation, review):
        dry = applier.run(workspace_root=destination, proposal=proposal,
            implementation_plan=implementation, review=review, allow_write=False)
        if dry["diff_hash"] != current["diff_hash"] or dry["unified_diff_sha256"] != current["unified_diff_sha256"]:
            raise ValueError("APPLY_ISOLATED_DIFF_CHANGED")
        return dry

    def isolated_apply_factory(proposal, implementation, review):
        if _copy_source_snapshot(source, excluded_copy_paths=excluded_copy_paths) != before:
            raise ValueError("APPLY_SOURCE_CHANGED_AFTER_APPROVAL")
        return applier.run(workspace_root=destination, proposal=proposal,
            implementation_plan=implementation, review=review, allow_write=True,
            approved_diff_hash=current["diff_hash"], human_semantic_decision=decision)

    state = create_initial_state(run_id=_apply_run_id(current["operation_id"]),
        request_id="reviewed_apply_" + current["operation_id"], user_request=admitted.feature_spec["user_request"],
        confirmed_isolated_write=True)
    state.update(feature_id=feature_id, current_stage="proposal_review",
        review_summary={key: payloads["proposal_review"].get(key) for key in ("decision", "allowed_next_stage")},
        artifacts={kind: ref.path for kind, ref in refs.items()},
        artifact_hashes={kind: ref.artifact_hash for kind, ref in refs.items()})
    graph = build_graph(store, dry_run_factory=dry_run_factory, isolated_apply_factory=isolated_apply_factory,
        checkpointer=WorkflowCheckpointStore.langgraph_memory_checkpointer(), interrupt_before=["build"])
    config = {"configurable": {"thread_id": state["run_id"]}}
    resumed = graph.update_state(config, state, as_node="review_proposal")
    finished = graph.invoke(None, resumed)
    source_unchanged = _copy_source_snapshot(source, excluded_copy_paths=excluded_copy_paths) == before
    apply_ref = ArtifactRef(feature_id=feature_id, artifact_type="isolated_apply_result",
        path=finished["artifacts"]["isolated_apply_result"],
        artifact_hash=finished["artifact_hashes"]["isolated_apply_result"])
    applied = store.read(apply_ref)["data"]
    complete = applied["actual_write_performed"] is True and not finished["blockers"] and source_unchanged
    checkpoint_ref = None
    if complete:
        # Persist the actual Graph state at its existing Apply boundary.  The
        # instance binding travels with the original checkpoint, not a second
        # execution registry or an approval inferred from a query projection.
        checkpoint_ref = _persist_applied_checkpoint(store, finished, current, before, destination)
    result = {**base_result, "status": "applied_in_isolation" if complete else "blocked",
        "source_workspace_written": not source_unchanged, "isolated_workspace": str(destination),
        "isolated_workspace_created": True, "artifact_writes_performed": True,
        "product_source_writes_performed": applied["actual_write_performed"],
        "apply_write_details": {key: applied[key] for key in (
            "actual_write_performed", "real_patch_applied", "changed_files", "write_completion", "write_failure") if key in applied},
        "isolated_apply_result": {"feature_id": apply_ref.feature_id, "artifact_type": apply_ref.artifact_type,
            "path": apply_ref.path, "artifact_hash": apply_ref.artifact_hash},
        "workflow_checkpoint": _artifact_reference(checkpoint_ref),
        "next_nodes": list(graph.get_state(config).next), "blockers": finished["blockers"],
        "trace_events": finished["trace_events"],
        "delivery_scope": "approved_bytes_in_isolated_workspace_not_built_or_runtime_verified"}
    return result


def _artifact_reference(reference):
    return None if reference is None else {key: getattr(reference, key) for key in _REF_FIELDS}


def _persist_applied_checkpoint(store, state, binding, source_snapshot, applied_source_root):
    from src.core.trace.execution_trace import current_trace
    from src.mod_agent.workflow.checkpointing import WorkflowCheckpointStore

    state["runtime_summary"]["reviewed_apply"] = {
        "displayed_binding": binding, "source_snapshot": source_snapshot,
        "applied_source_root": str(applied_source_root),
        "applied_source_snapshot": _copy_source_snapshot(applied_source_root),
    }
    original_request = state["runtime_summary"].get("deferred_product_request_json")
    if original_request is not None:
        from src.mod_agent.domain.csharp_repository_delivery import repository_binding_from_request
        from src.mod_agent.product_execution import ProductExecutionRequestV1
        repository = repository_binding_from_request(ProductExecutionRequestV1.from_dict(json.loads(original_request)))
        if repository is not None:
            state["runtime_summary"]["reviewed_apply"]["repository_build_and_test"] = repository
    trace = current_trace()
    if trace is not None and "logical_request_id" in trace.context.metadata:
        state["runtime_summary"]["reviewed_apply"]["trace_correlation"] = {
            "logical_request_id": trace.context.metadata["logical_request_id"],
        }
    return WorkflowCheckpointStore(store).create_checkpoint(state,
        interrupt_reason="Approved isolated Apply awaits separate Build authorization.", resume_allowed_fields=())


def _applied_source_snapshot(root):
    # This owner writes its presentation result only after the Apply checkpoint.
    return [row for row in _copy_source_snapshot(root) if row["path"] != "reviewed-apply-result.json"]


def _reviewed_build_inputs(store, checkpoint_ref, resume_payload):
    from src.mod_agent.workflow.interrupts import ResumePayload

    if not isinstance(checkpoint_ref, ArtifactRef) or checkpoint_ref.artifact_type != "workflow_checkpoint":
        raise ValueError("REVIEWED_BUILD_CHECKPOINT_REQUIRED")
    if not isinstance(resume_payload, ResumePayload):
        raise ValueError("REVIEWED_BUILD_RESUME_PAYLOAD_REQUIRED")
    checkpoint = store.semantic_read_immutable(checkpoint_ref)["data"]
    state = checkpoint["state_snapshot"]
    if checkpoint["checkpoint_stage"] != "isolated_apply" or state["current_stage"] != "isolated_apply":
        raise ValueError("REVIEWED_BUILD_REQUIRES_APPLY_CHECKPOINT")
    if (resume_payload.feature_id != checkpoint["feature_id"]
            or resume_payload.request_id != checkpoint["request_id"]):
        raise ValueError("REVIEWED_BUILD_RESUME_IDENTITY_MISMATCH")
    selected = state["runtime_summary"].get("reviewed_apply")
    if not isinstance(selected, dict):
        raise ValueError("REVIEWED_BUILD_APPLY_BINDING_MISSING")
    binding = selected["displayed_binding"]
    if (resume_payload.diff_hash != binding["diff_hash"]
            or checkpoint["resume_binding"]["diff_hash"] != binding["diff_hash"]):
        raise ValueError("REVIEWED_BUILD_DIFF_IDENTITY_MISMATCH")
    apply_ref = ArtifactRef(feature_id=checkpoint_ref.feature_id, artifact_type="isolated_apply_result",
        path=state["artifacts"]["isolated_apply_result"], artifact_hash=state["artifact_hashes"]["isolated_apply_result"])
    applied = store.semantic_read_immutable(apply_ref)["data"]
    from src.mod_agent.execution.task_delegation import (
        has_verified_apply_authorization,
        validate_delegated_resume,
        validate_task_delegation,
    )
    if applied.get("authorization_origin") is not None:
        validate_delegated_resume(applied["task_delegation_ref"], resume_payload,
            run_id=checkpoint["run_id"], request_id=checkpoint["request_id"])
        validate_task_delegation(applied["task_delegation_ref"], fresh_verification=True)
    if (applied["actual_write_performed"] is not True or applied["execution_mode"] != "isolated_apply"
            or applied["approved_diff_hash"] != binding["diff_hash"]
            or not has_verified_apply_authorization(applied, run_id=checkpoint["run_id"],
                request_id=checkpoint["request_id"])):
        raise ValueError("REVIEWED_BUILD_APPROVED_APPLY_REQUIRED")
    source, destination = Path(binding["workspace_root"]), Path(binding["isolated_workspace"])
    if (not source.is_absolute() or not destination.is_absolute() or not source.is_dir() or not destination.is_dir()
            or destination.resolve().is_relative_to(source.resolve())
            or source.resolve().is_relative_to(destination.resolve())):
        raise ValueError("REVIEWED_BUILD_WORKSPACE_BINDING_INVALID")
    excluded_copy_paths = tuple(binding.get("excluded_copy_paths", ()))
    if excluded_copy_paths and (not store.root.resolve().is_relative_to(source.resolve())
            or excluded_copy_paths != (store.root.resolve().relative_to(source.resolve()).as_posix(),)):
        raise ValueError("REVIEWED_BUILD_COPY_BOUNDARY_CHANGED")
    if _copy_source_snapshot(source, excluded_copy_paths=excluded_copy_paths) != selected["source_snapshot"]:
        raise ValueError("REVIEWED_BUILD_ORIGINAL_SOURCE_CHANGED")
    # Older reviewed-copy checkpoints inventory their workspace root. Product
    # checkpoints explicitly inventory only their declared source subtree, so
    # ArtifactStore and manifest writes cannot invalidate source identity.
    applied_source_root = Path(selected.get("applied_source_root", str(destination))).resolve()
    if not applied_source_root.is_relative_to(destination.resolve()) or not applied_source_root.is_dir():
        raise ValueError("REVIEWED_BUILD_SOURCE_ROOT_OUTSIDE_WORKSPACE")
    if _applied_source_snapshot(applied_source_root) != selected["applied_source_snapshot"]:
        raise ValueError("REVIEWED_BUILD_APPLIED_SOURCE_CHANGED")
    for change in applied["changes"]:
        path = (destination / change["path"]).resolve()
        if (not path.is_relative_to(destination.resolve()) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != change["raw_after_sha256"]):
            raise ValueError("REVIEWED_BUILD_APPLIED_BYTES_MISMATCH")
    return checkpoint, selected, apply_ref, destination.resolve()


def _verify_reviewed_build_result(result, *, destination, feature_id, mode, source_binding):
    from src.mod_agent.build.adapter import BuildResult

    if (not isinstance(result, BuildResult) or result.feature_id != feature_id
            or result.input_execution_mode != mode or result.real_build_performed is not True
            or result.execution_mode not in {"local", "real"}):
        raise ValueError("REVIEWED_BUILD_RESULT_IDENTITY_MISMATCH")
    profile = result.build_profile
    if (not isinstance(profile, dict) or not isinstance(profile.get("working_directory"), str)
            or not Path(profile["working_directory"]).is_absolute()
            or not Path(profile["working_directory"]).resolve().is_relative_to(destination)):
        raise ValueError("REVIEWED_BUILD_PROFILE_OUTSIDE_APPLIED_WORKSPACE")
    applied_source_root = Path(source_binding.get("applied_source_root", str(destination))).resolve()
    if _applied_source_snapshot(applied_source_root) != source_binding["applied_source_snapshot"]:
        raise ValueError("REVIEWED_BUILD_INPUT_CHANGED_DURING_BUILD")
    for row in source_binding["applied_source_snapshot"]:
        path = applied_source_root / row["path"]
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(destination)
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError("REVIEWED_BUILD_INPUT_CHANGED_DURING_BUILD")
    if _copy_source_snapshot(Path(source_binding["displayed_binding"]["workspace_root"]),
            excluded_copy_paths=tuple(source_binding["displayed_binding"].get("excluded_copy_paths", ()))) != source_binding["source_snapshot"]:
        raise ValueError("REVIEWED_BUILD_ORIGINAL_SOURCE_CHANGED")
    if result.status == "success" and (
        result.claims_real_build_success is not True or result.artifact_verification.get("status") != "verified"
            or not result.verified_artifacts):
        raise ValueError("REVIEWED_BUILD_VERIFIED_OUTPUT_REQUIRED")
    for artifact in result.verified_artifacts:
        path = Path(artifact["resolved_path"])
        if (not path.is_absolute() or not path.resolve().is_relative_to(destination) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]):
            raise ValueError("REVIEWED_BUILD_OUTPUT_BINDING_INVALID")
    return result


def continue_reviewed_patch(*, artifact_store_root, checkpoint_ref, resume_payload,
                           allow_build=False, build_result_factory=None,
                           execution_trace_configuration=None):
    """Consume one original Apply checkpoint and execute only its authorized Build.

    ``build_result_factory(destination, feature_id, input_execution_mode)`` is an
    explicitly supplied host adapter.  Runtime continues through the existing
    Build checkpoint/Runtime entry; no model, Apply replay, or repair is started
    here.  A consumed checkpoint cannot launch Build again, even after a crash.
    """
    from src.mod_agent.patch_execution_trace import PatchExecutionTraceConfiguration
    from src.mod_agent.workflow.checkpointing import (
        BUILD_CHECKPOINT_EXACT_REVISION_TYPES,
        WorkflowCheckpointStore,
        resumed_state_from_decision,
    )
    from src.mod_agent.workflow.graph import build_graph

    if allow_build is not True:
        raise ValueError("REVIEWED_BUILD_AUTHORIZATION_REQUIRED")
    store = ArtifactStore(artifact_store_root)
    checkpoint, source_binding, apply_ref, destination = _reviewed_build_inputs(store, checkpoint_ref, resume_payload)
    repository_binding = source_binding.get("repository_build_and_test")
    if repository_binding is not None:
        from src.mod_agent.domain.csharp_repository_delivery import repository_build_factory
        if build_result_factory is not None:
            raise ValueError("REPOSITORY_BUILD_FACTORY_OVERRIDE_FORBIDDEN")
        build_result_factory = repository_build_factory(repository_binding, destination)
    if not callable(build_result_factory):
        raise ValueError("REVIEWED_BUILD_FACTORY_REQUIRED")
    checkpoint_store = WorkflowCheckpointStore(store)
    base = {"entry": "continue-reviewed-patch", "run_id": checkpoint["run_id"],
        "feature_id": checkpoint["feature_id"], "isolated_workspace": str(destination),
        "diff_hash": source_binding["displayed_binding"]["diff_hash"],
        "isolated_apply_result": _artifact_reference(apply_ref),
        "build_input_checkpoint": _artifact_reference(checkpoint_ref),
        "source_snapshot_sha256": _sha256_json(source_binding["applied_source_snapshot"]),
        "model_calls_performed": 0, "builds_performed": 0, "runtime_executions_performed": 0,
        "product_source_writes_performed": False, "source_workspace_written": False,
        "build_result": None, "workflow_checkpoint": None, "stopped_before": "runtime_evidence"}
    # ResumeReport is the existing durable consumption owner.  Do not turn a
    # repeated invocation or an uncertain previous Build into another attempt.
    preflight = checkpoint_store.preflight_resume(checkpoint_ref, resume_id=resume_payload.resume_id)
    if not preflight.allowed:
        return {**base, "status": "blocked", "resume_report": None, "blockers": list(preflight.blockers)}
    decision, report_ref = checkpoint_store.resume(checkpoint_ref, resume_payload)
    base["resume_report"] = _artifact_reference(report_ref)
    if not decision.allowed or resume_payload.decision != "approve":
        return {**base, "status": "rejected" if resume_payload.decision == "reject" else "blocked",
                "blockers": list(decision.blockers)}
    observations = {}
    configuration = execution_trace_configuration or PatchExecutionTraceConfiguration()
    logical_request_id = (source_binding.get("trace_correlation") or {}).get("logical_request_id")
    trace = configuration.make_trace(run_id=checkpoint["run_id"] + ":build:" + resume_payload.resume_id,
        feature_id=checkpoint["feature_id"], artifact_store_root=artifact_store_root,
        recorder=lambda row: observations.update({row["id"]: row}),
        **({"request_id": logical_request_id} if logical_request_id is not None else {}))
    build_calls = 0

    def bound_build(feature_id, mode):
        nonlocal build_calls
        # Recheck after durable admission, immediately before the physical effect.
        _reviewed_build_inputs(store, checkpoint_ref, resume_payload)
        build_calls += 1
        result = build_result_factory(destination, feature_id, mode)
        return _verify_reviewed_build_result(result, destination=destination, feature_id=feature_id,
            mode=mode, source_binding=source_binding)

    def forbidden_apply(*args, **kwargs):
        raise AssertionError("Reviewed Build continuation cannot repeat Apply.")

    failure = None
    try:
        with trace.activate():
            # ResumeReport already binds the consumed checkpoint by exact
            # revision. Re-adding that old checkpoint as a latest-required
            # sibling would make the new Build checkpoint stale on creation.
            state = resumed_state_from_decision(decision, resume_report_ref=report_ref)
            graph = build_graph(store, isolated_apply_factory=forbidden_apply,
                build_result_factory=bound_build, stop_after_build=True, persist_build_runtime_handoff=True,
                build_checkpoint_exact_revision_upstream_types=BUILD_CHECKPOINT_EXACT_REVISION_TYPES,
                checkpointer=WorkflowCheckpointStore.langgraph_memory_checkpointer())
            config = {"configurable": {"thread_id": checkpoint["run_id"]}}
            restored = graph.update_state(config, state, as_node="isolated_apply")
            finished = graph.invoke(None, restored)
            build_ref = ArtifactRef(feature_id=checkpoint["feature_id"], artifact_type="build_result",
                path=finished["artifacts"]["build_result"], artifact_hash=finished["artifact_hashes"]["build_result"])
            built = store.semantic_read_immutable(build_ref)["data"]
            runtime_checkpoint = None
            if finished["artifacts"].get("workflow_checkpoint") is not None:
                runtime_checkpoint = ArtifactRef(feature_id=checkpoint["feature_id"], artifact_type="workflow_checkpoint",
                    path=finished["artifacts"]["workflow_checkpoint"], artifact_hash=finished["artifact_hashes"]["workflow_checkpoint"])
            status = "built" if built["status"] == "success" and not finished["blockers"] else "failed"
            result = {**base, "status": status, "builds_performed": build_calls,
                "build_result": _artifact_reference(build_ref), "workflow_checkpoint": _artifact_reference(runtime_checkpoint),
                "blockers": finished["blockers"], "trace_events": finished["trace_events"],
                "stopped_before": "runtime_evidence" if status == "built" else "repair_plan",
                "delivery_scope": "approved_isolated_bytes_built_not_runtime_verified" if status == "built" else "bound_build_failure_not_repaired"}
            trace.update(workflow_status=status, terminal_stage="build")
    except BaseException as error:
        failure = error
        raise
    finally:
        summary = configuration.finish(trace)
        summary.update(trace_id=trace.context.trace_id, session_id=trace.session_id, environment=trace.environment)
        if failure is not None:
            failure.execution_trace = summary
            failure.execution_observations = observations
            failure.resume_report = _artifact_reference(report_ref)
    result.update(execution_trace=summary, execution_observations=observations)
    return result


def resume_deferred_approval(*, artifact_store_root, checkpoint_ref, resume_payload,
                             human_semantic_decision=None, task_delegation_ref=None,
                             execution_trace_configuration=None):
    """Approve a saved product diff in its existing isolated workspace, once."""
    from src.mod_agent.execution.patch_apply import (
        ExactTextPatchApplier,
        _validate_human_semantic_decision,
    )
    from src.mod_agent.patch_execution_trace import PatchExecutionTraceConfiguration
    from src.mod_agent.product_execution import ProductExecutionRequestV1
    from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256
    from src.mod_agent.workflow.checkpointing import (
        WorkflowCheckpointStore,
        resumed_state_from_decision,
    )
    from src.mod_agent.workflow.graph import build_graph
    from src.mod_agent.workflow.interrupts import ResumePayload

    if not isinstance(checkpoint_ref, ArtifactRef) or checkpoint_ref.artifact_type != "workflow_checkpoint":
        raise ValueError("DEFERRED_APPROVAL_CHECKPOINT_REQUIRED")
    if not isinstance(resume_payload, ResumePayload):
        raise ValueError("DEFERRED_APPROVAL_RESUME_REQUIRED")
    store = ArtifactStore(artifact_store_root)
    checkpoint = store.semantic_read_immutable(checkpoint_ref)["data"]
    paused = checkpoint["state_snapshot"]
    if checkpoint["checkpoint_stage"] != "dry_run_apply" or paused["current_stage"] != "dry_run_apply":
        raise ValueError("DEFERRED_APPROVAL_DRY_RUN_CHECKPOINT_REQUIRED")
    request = ProductExecutionRequestV1.from_dict(json.loads(paused["runtime_summary"]["deferred_product_request_json"]))
    if (request.run_id != checkpoint["run_id"] or request.request_id != checkpoint["request_id"]
            or Path(request.artifact_store_root).resolve() != Path(artifact_store_root).resolve()
            or resume_payload.feature_id != checkpoint_ref.feature_id
            or resume_payload.request_id != request.request_id):
        raise ValueError("DEFERRED_APPROVAL_IDENTITY_MISMATCH")
    base = {"entry": "apply-patch", "run_id": request.run_id, "request_id": request.request_id,
        "feature_id": checkpoint_ref.feature_id, "isolated_workspace": request.isolated_workspace_root,
        "model_calls_performed": 0, "builds_performed": 0, "runtime_executions_performed": 0,
        "product_source_writes_performed": False, "source_workspace_written": False,
        "isolated_workspace_created": False, "workflow_checkpoint": None, "stopped_before": "build"}
    checkpoint_store = WorkflowCheckpointStore(store)
    preflight = checkpoint_store.preflight_resume(checkpoint_ref, resume_id=resume_payload.resume_id)
    if not preflight.allowed:
        return {**base, "status": "blocked", "resume_report": None, "blockers": list(preflight.blockers)}
    documents = {kind: store.semantic_read_immutable(ArtifactRef(feature_id=checkpoint_ref.feature_id,
        artifact_type=kind, path=paused["artifacts"][kind], artifact_hash=paused["artifact_hashes"][kind]))["data"]
        for kind in ("patch_proposal", "proposal_review", "implementation_plan", "dry_run_result")}
    dry = documents["dry_run_result"]
    if resume_payload.diff_hash != dry["diff_hash"]:
        raise ValueError("DEFERRED_APPROVAL_DIFF_MISMATCH")
    human = None
    if task_delegation_ref is not None:
        from src.mod_agent.execution.task_delegation import validate_delegated_resume

        if human_semantic_decision is not None:
            raise ValueError("DEFERRED_APPROVAL_AUTHORIZATION_SOURCE_AMBIGUOUS")
        delegated = validate_delegated_resume(task_delegation_ref, resume_payload,
            run_id=request.run_id, request_id=request.request_id, checkpoint_ref=checkpoint_ref)
        approval_decision = delegated["decision"]
        base.update(authorization_origin="task_delegation_policy", task_delegation_ref=dict(task_delegation_ref))
    else:
        human = _validate_human_semantic_decision(human_semantic_decision,
            expected_binding=dry["human_semantic_decision_binding"])
        approval_decision = human["decision"]
    if resume_payload.decision != approval_decision:
        raise ValueError("DEFERRED_APPROVAL_DECISIONS_DISAGREE")
    source = Path(request.source_root)
    destination = Path(request.isolated_workspace_root)
    target = Path(request.isolated_target_root)
    from src.mod_agent.domain.csharp_repository_delivery import (
        repository_binding_from_request,
        validate_repository_changes,
    )
    repository_binding = repository_binding_from_request(request)

    def verify_source():
        if source_snapshot_sha256(source_snapshot(source)) != request.source_snapshot_sha256:
            raise ValueError("DEFERRED_APPROVAL_ORIGINAL_SOURCE_CHANGED")
        if source_snapshot_sha256(source_snapshot(target)) != request.isolated_source_snapshot_sha256:
            raise ValueError("DEFERRED_APPROVAL_ISOLATED_SOURCE_CHANGED")

    verify_source()
    applier = ExactTextPatchApplier()
    current = applier.run(workspace_root=destination, proposal=documents["patch_proposal"],
        implementation_plan=documents["implementation_plan"], review=documents["proposal_review"], allow_write=False)
    if current["diff_hash"] != dry["diff_hash"] or current["unified_diff_sha256"] != dry["unified_diff_sha256"]:
        raise ValueError("DEFERRED_APPROVAL_CURRENT_DIFF_CHANGED")
    if repository_binding is not None:
        validate_repository_changes(repository_binding, destination, current["changes"])
    decision, report_ref = checkpoint_store.resume(checkpoint_ref, resume_payload)
    base.update(resume_report=_artifact_reference(report_ref), diff_hash=dry["diff_hash"])
    if not decision.allowed or approval_decision != "approve":
        return {**base, "status": "rejected" if approval_decision == "reject" else "blocked",
                "blockers": list(decision.blockers)}
    source_before = _copy_source_snapshot(source)

    def apply_current(proposal, implementation, review):
        verify_source()
        return applier.run(workspace_root=destination, proposal=proposal, implementation_plan=implementation,
            review=review, allow_write=True, approved_diff_hash=dry["diff_hash"], human_semantic_decision=human,
            task_delegation_ref=task_delegation_ref,
            task_delegation_resume_ref=_artifact_reference(report_ref) if task_delegation_ref else None)

    observations = {}
    configuration = execution_trace_configuration or PatchExecutionTraceConfiguration()
    trace = configuration.make_trace(run_id=request.run_id + ":approval:" + resume_payload.resume_id,
        feature_id=checkpoint_ref.feature_id, request_id=request.request_id,
        artifact_store_root=artifact_store_root, recorder=lambda row: observations.update({row["id"]: row}))
    try:
        with trace.activate():
            state = resumed_state_from_decision(decision, resume_report_ref=report_ref)
            graph = build_graph(store, isolated_apply_factory=apply_current,
                checkpointer=WorkflowCheckpointStore.langgraph_memory_checkpointer(), interrupt_before=["build"])
            config = {"configurable": {"thread_id": request.run_id}}
            resumed = graph.update_state(config, state, as_node="dry_run_apply")
            finished = graph.invoke(None, resumed)
            apply_ref = ArtifactRef(feature_id=checkpoint_ref.feature_id, artifact_type="isolated_apply_result",
                path=finished["artifacts"]["isolated_apply_result"], artifact_hash=finished["artifact_hashes"]["isolated_apply_result"])
            applied = store.semantic_read_immutable(apply_ref)["data"]
            source_unchanged = _copy_source_snapshot(source) == source_before
            complete = applied["actual_write_performed"] is True and not finished["blockers"] and source_unchanged
            following = None
            if complete:
                following = _persist_applied_checkpoint(store, finished, {
                    "workspace_root": str(source), "isolated_workspace": str(destination),
                    "diff_hash": dry["diff_hash"], "unified_diff_sha256": dry["unified_diff_sha256"],
                    "human_semantic_decision_binding": dry["human_semantic_decision_binding"],
                    **({"authorization_origin": "task_delegation_policy",
                        "task_delegation_ref": dict(task_delegation_ref)} if task_delegation_ref else {}),
                }, source_before, target)
            result = {**base, "status": "applied_in_isolation" if complete else "blocked",
                "isolated_apply_result": _artifact_reference(apply_ref), "workflow_checkpoint": _artifact_reference(following),
                "product_source_writes_performed": applied["actual_write_performed"],
                "apply_write_details": {key: applied[key] for key in (
                    "actual_write_performed", "real_patch_applied", "changed_files", "write_completion", "write_failure") if key in applied},
                "source_workspace_written": not source_unchanged, "blockers": finished["blockers"],
                "trace_events": finished["trace_events"], "next_nodes": list(graph.get_state(config).next)}
            trace.update(workflow_status=result["status"], terminal_stage="isolated_apply")
    finally:
        summary = configuration.finish(trace)
        summary.update(trace_id=trace.context.trace_id, session_id=trace.session_id, environment=trace.environment)
    result.update(execution_trace=summary, execution_observations=observations)
    return result
