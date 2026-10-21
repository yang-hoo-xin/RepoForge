"""Zero-model recovery of an already checked native candidate into normal approval."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.candidate_verification_continuation import (
    _checked,
    _checkpoint,
    _completed_delivery_view,
    _file,
    _hash,
    _write,
)
from src.mod_agent.model_call_budget import validate_budget
from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256
from src.mod_agent.task_agent_execution import load_task_agent_outcome, validate_task_agent_handoff


def _require(condition, code):
    if not condition:
        raise ValueError("RETAINED_TASK_AGENT_" + code)


def _retained_authority(execution_config, root, manifest, original, execution):
    import json

    from scripts.dsh_run_request import validate_product_profile

    reference, _ = _file(execution_config)
    _require(Path(reference["path"]) == root.parent / ".dsh-inputs" / root.name / "execution.json",
        "ALLOCATED_CONFIGURATION_REQUIRED")
    _, parsed = validate_product_profile(execution_config=reference["path"], description=manifest["request"])
    selection = manifest["technical_execution"]
    # Scope: reconstruct only the original no-write approval boundary, not Apply authority.
    # 为什么要有它：普通新请求的配置没有后续任务绑定，仍须核原分配路径和全部有效权限；真正Apply继续要求精确审批。
    _require(parsed.run_id == manifest["run_id"] and parsed.request_id == manifest["request_id"]
        and Path(parsed.workspace_root).resolve() / parsed.run_id == root
        and Path(parsed.target_project).resolve() == Path(manifest["target_source"]).resolve()
        and parsed.confirm_isolated_write and parsed.allow_build and parsed.defer_approval
        and parsed.task_model_call_limit == original["model_call_budget"]["maximum"]
        and parsed.model_call_budget_policy == original["model_call_budget"].get("policy")
        and _file(parsed.patch_worker_config)[0] == execution["worker_binding"]["configuration_ref"]
        and parsed.technical_executor == selection["technical_executor"]
        and parsed.constraint_policy == selection["constraint_policy"]
        and parsed.task_agent_candidate_mode == selection.get("candidate_mode", "baseline")
        and json.loads(parsed.task_agent_edit_paths) == selection["allowed_edit_paths_relative_to_project"]
        and manifest.get("runtime_admission", {}).get("requested") is False,
        "AUTHORIZATION_CONFIGURATION_CHANGED")
    return reference


@dataclass(frozen=True)
class RetainedHandoffAdmission:
    root: Path
    identity: dict[str, Any]
    manifest: dict[str, Any]
    original: dict[str, Any]
    execution: dict[str, Any]
    outcome: Any
    handoff: Any

    def revalidate(self):
        current = admit_retained_task_agent_handoff(
            execution_config=self.identity["authorization_configuration_ref"]["path"],
            run_root=self.root, run_id=self.manifest["run_id"],
            ready_execution_ref=self.identity["ready_execution_ref"])
        _require(current.identity == self.identity, "ADMISSION_CHANGED")
        return current


def admit_retained_task_agent_handoff(*, execution_config, run_root, run_id, ready_execution_ref):
    """Read-only exact admission; no model, evaluator, source write or Build.

    为什么要有它：保存的通过状态只有与原失败、模型动作、预算和当前源码一致时才能续接，不能把失败记录改成成功。
    """
    root = Path(run_root).resolve(strict=True)
    _require(root.name == run_id, "RUN_IDENTITY_CHANGED")
    manifest_ref, manifest = _file(root / "run_manifest.json")
    result_ref, original = _file(root / "run_result.json")
    _require(manifest.get("run_id") == original.get("run_id") == run_id
        and manifest.get("terminal_record") == result_ref
        and manifest.get("status") == "blocked" and original.get("current_stage") == "blocked"
        and original.get("actual_write_performed") is False
        and original.get("canonical_source_unchanged") in {None, True}
        and not original.get("patch_execution_receipts"), "PARENT_NOT_RECOVERABLE")
    execution_root = root / "a" / "task-agent-execution"
    _require(Path(ready_execution_ref["path"]).resolve() == execution_root / "execution.json",
        "READY_EXECUTION_OUTSIDE_ORIGINAL")
    ready_ref, execution = _file(execution_root / "execution.json")
    _require(ready_ref == ready_execution_ref, "READY_EXECUTION_CHANGED")
    error_ref, error = _file(execution_root / "execution-error.json")
    old = original.get("task_agent_outcome", {})
    _require(original.get("task_agent_execution_ref") == error_ref and old.get("execution_ref") == error_ref
        and old.get("status") == error.get("status") == "failed"
        and old.get("unknown_effects") is False and error.get("unknown_effects") is False
        and old.get("termination") == error.get("termination") == {
            "code": "TASK_AGENT_EXECUTION_FAILED", "error_type": "ValueError",
            "reason": "TASK_AGENT_EXECUTION_REFERENCE_OUTSIDE_STORE",
            "prior_loop_termination": {"code": "TASK_AGENT_COMPLETION_ACCEPTED"}},
        "ORIGINAL_FAILURE_CHANGED")
    for key in old:
        if key != "execution_ref":
            _require(old[key] == error.get(key), "ORIGINAL_OUTCOME_CHANGED")
        if key not in {"execution_ref", "status", "termination"}:
            _require(old[key] == execution.get(key), "READY_ERROR_LINEAGE_CHANGED")
    _require(execution.get("run_id") == run_id
        and execution.get("request_id") == manifest.get("request_id"), "EXECUTION_IDENTITY_CHANGED")
    for key in ("worker_binding", "policy", "model_profile", "allowed_edit_paths", "source_before",
            "expected_model_identity", "expected_verification_identity", "candidate_mode"):
        _require(execution.get(key) == error.get(key), "EXECUTION_OWNER_CHANGED")
    budget = validate_budget(original["model_call_budget"])
    _require(execution.get("budget_after") == budget and original.get("model_call_count") == budget["consumed"]
        and manifest.get("model_call_count") == budget["consumed"], "BUDGET_CHANGED")
    selection = manifest.get("technical_execution", {})
    _require(selection == original.get("technical_execution")
        and selection.get("technical_executor") == "task_agent"
        and selection.get("constraint_policy") == "source_priority_v1"
        and selection.get("candidate_mode", "baseline") == execution.get("candidate_mode", "baseline")
        and manifest.get("local_model_identity") == execution.get("expected_model_identity"),
        "TECHNICAL_OWNER_CHANGED")
    target = Path(manifest["nuget_isolated_build_profile"]["working_directory"]).resolve(strict=True)
    _require(target.is_relative_to(root / "generated_mods")
        and execution.get("allowed_edit_paths") == [
            (target.relative_to(root) / path).as_posix()
            for path in selection.get("allowed_edit_paths_relative_to_project", [])], "EDIT_SCOPE_CHANGED")
    _require(source_snapshot(target) == original.get("initial_isolated_source_snapshot")
        and original.get("final_isolated_source_snapshot") in (None, original.get("initial_isolated_source_snapshot"))
        and source_snapshot_sha256(source_snapshot(Path(manifest["target_source"])))
        == manifest.get("source_snapshot_sha256"), "SOURCE_CHANGED")
    worker_ref = execution["worker_binding"]["configuration_ref"]
    authority_ref = _retained_authority(execution_config, root, manifest, original, execution)
    store = ArtifactStore(root / "a")
    outcome = load_task_agent_outcome(ready_ref, artifact_store_root=store.root)
    handoff = validate_task_agent_handoff(outcome, store=store, workspace_root=root,
        expected_initial_context=outcome.initial_context, allowed_edit_paths=tuple(execution["allowed_edit_paths"]),
        worker_configuration_ref=worker_ref, expected_verification_identity=execution["expected_verification_identity"],
        expected_candidate_mode=execution.get("candidate_mode", "baseline"))
    identity = {"parent_manifest_ref": manifest_ref, "parent_result_ref": result_ref,
        "error_execution_ref": error_ref, "ready_execution_ref": ready_ref,
        "authorization_configuration_ref": authority_ref}
    return RetainedHandoffAdmission(root, identity, manifest, original, execution, outcome, handoff)


def _fresh_view(state, store):
    roles = {"feature_spec": "feature_spec_artifact", "evidence_pack": "evidence_pack_artifact",
        "implementation_plan": "implementation_plan_artifact", "patch_proposal": "patch_proposal_artifact",
        "proposal_review": "proposal_review_artifact", "dry_run_result": "dry_run_artifact"}
    result = {"run_id": state["run_id"], "current_stage": state["current_stage"],
        "blockers": deepcopy(state["blockers"]), "trace_events": deepcopy(state["trace_events"])}
    for role, key in roles.items():
        if role in state["artifacts"]:
            ref = ArtifactRef(feature_id=state["feature_id"], artifact_type=role,
                path=state["artifacts"][role], artifact_hash=state["artifact_hashes"][role])
            store.semantic_read_immutable(ref)
            result[key] = {"feature_id": ref.feature_id, "artifact_type": ref.artifact_type,
                "path": ref.path, "artifact_hash": ref.artifact_hash}
    return result


def resume_retained_task_agent_handoff(*, execution_config, run_root, run_id, ready_execution_ref, attempt_id):
    """Continue one native result through normal Proposal, Review and exact dry run.

    为什么要有它：候选已验收后交接失败不应重新调用模型；新检查点仍必须等待独立的精确审批，不能隐式Apply。
    """
    _require(isinstance(attempt_id, str) and bool(attempt_id.strip()), "ATTEMPT_REQUIRED")
    root = Path(run_root).resolve(strict=True)
    segment = root / "continuations" / ("ta_" + _hash(ready_execution_ref)[:20])
    if (root / "delivery").exists():
        # Scope: already-started delivery is a read-only operation, never a new approval.
        # 为什么要有它：Apply之后初始源码应已改变，不能因此再建恢复任务，也不能盲重试未知交付副作用。
        _require((segment / "completion.json").is_file(), "DELIVERY_ALREADY_STARTED")
        saved = _file(segment / "completion.json")[1]
        cached = _checked(saved["result_ref"])
        claim = _checked(cached["continuation_claim_ref"])
        _require(root.name == run_id and cached["run_id"] == run_id and claim["attempt_id"] == attempt_id
            and cached["ready_execution_ref"] == ready_execution_ref
            and cached["authorization_configuration_ref"] == _file(execution_config)[0]
            and _file(root / "run_manifest.json")[0] == cached["parent_manifest_ref"]
            and _file(root / "run_result.json")[0] == cached["parent_result_ref"]
            and saved == {"result_ref": _file(segment / "run_result.json")[0],
                "claim_ref": cached["continuation_claim_ref"], "fresh_result_ref": cached["fresh_result_ref"]},
            "DELIVERY_ORIGIN_CHANGED")
        _require((root / "delivery/authorization.json").is_file()
            and (root / "delivery/result.json").is_file(), "DELIVERY_INCOMPLETE_REVIEW_REQUIRED")
        return _completed_delivery_view(execution_config=execution_config, root=root,
            run_id=run_id, result_ref=saved["result_ref"])
    admitted = admit_retained_task_agent_handoff(execution_config=execution_config, run_root=run_root,
        run_id=run_id, ready_execution_ref=ready_execution_ref)
    identity = {**deepcopy(admitted.identity), "attempt_id": attempt_id}
    # Scope: one owner per retained execution, even if callers invent another attempt id.
    # 为什么要有它：进程中断后未完成的工件写入必须核对，不能换ID盲目重复提案和审批检查点。
    segment = admitted.root / "continuations" / ("ta_" + _hash(ready_execution_ref)[:20])
    segment.parent.mkdir(exist_ok=True)
    _require(not segment.is_symlink() and segment.resolve().is_relative_to(admitted.root), "OUTPUT_OUTSIDE_RUN")
    try:
        segment.mkdir()
    except FileExistsError:
        _require(_checked(_file(segment / "selection.json")[0]) == identity, "OPERATION_CONFLICT")
        if (segment / "completion.json").exists():
            return validate_retained_handoff_result(admitted.root,
                _file(segment / "completion.json")[1]["result_ref"])
        return {"status": "blocked", "run_id": run_id, "request_id": admitted.manifest["request_id"],
            "reason": "RETAINED_TASK_AGENT_INCOMPLETE_REVIEW_REQUIRED", "new_provider_sends": 0,
            "continuation_root": str(segment)}
    _write(segment / "selection.json", identity)
    from src.mod_agent.workflow.graph import build_retained_task_agent_handoff_graph
    from src.mod_agent.workflow.state import create_initial_state

    store = ArtifactStore(admitted.root / "a")
    state = create_initial_state(run_id=run_id, request_id=admitted.manifest["request_id"],
        user_request=admitted.manifest["request"], target_domain="gtfo",
        target_mod=Path(admitted.manifest["target_source"]).name, confirmed_isolated_write=True)
    state = build_retained_task_agent_handoff_graph(store, admission=admitted).invoke(state)
    fresh = _fresh_view(state, store)
    _write(segment / "graph_result.json", fresh)
    public = {"status": "blocked", "current_stage": state["current_stage"], "blockers": state["blockers"]}
    if state["current_stage"] == "dry_run_apply" and not state["blockers"]:
        public = _checkpoint(root=admitted.root, manifest=admitted.manifest, original=admitted.original,
            fresh=fresh, authorization_ref=admitted.identity["authorization_configuration_ref"])
    public.update(entry="retained-task-agent-handoff", generation_origin="retained_task_agent_candidate",
        run_id=run_id, parent_run_id=run_id, request_id=admitted.manifest["request_id"],
        **deepcopy(admitted.identity), task_agent_execution_ref=deepcopy(ready_execution_ref),
        task_agent_outcome=admitted.outcome.to_dict(), technical_execution=deepcopy(admitted.manifest["technical_execution"]),
        new_provider_sends=0, model_calls_performed=0, model_call_count=admitted.original["model_call_count"],
        model_call_budget=deepcopy(admitted.original["model_call_budget"]), patch_execution_receipts=[],
        actual_write_performed=False, new_builds=0, continuation_root=str(segment), run_root=str(admitted.root),
        continuation_claim_ref=_file(segment / "selection.json")[0], fresh_result_ref=_file(segment / "graph_result.json")[0])
    source_root = Path(__file__).parent
    public["recovery_implementation_snapshot"] = [{"path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (
            Path(__file__), source_root / "task_agent_execution.py", source_root / "workflow/graph.py",
            source_root / "candidate_verification_continuation.py")]
    admitted.revalidate()
    _write(segment / "run_result.json", public)
    _write(segment / "completion.json", {"result_ref": _file(segment / "run_result.json")[0],
        "claim_ref": public["continuation_claim_ref"], "fresh_result_ref": public["fresh_result_ref"]})
    return validate_retained_handoff_result(admitted.root, _file(segment / "run_result.json")[0])


def validate_retained_handoff_result(run_root, result_ref):
    """Read-only validation for normal delivery; this never manufactures approval."""
    root = Path(run_root).resolve(strict=True)
    public = _checked(result_ref)
    segment = Path(public["continuation_root"]).resolve(strict=True)
    _require(segment == root / "continuations" / ("ta_" + _hash(public["ready_execution_ref"])[:20])
        and Path(result_ref["path"]).resolve() == segment / "run_result.json", "RESULT_OUTSIDE_CONTINUATION")
    identity = _checked(public["continuation_claim_ref"])
    _require(_file(segment / "selection.json")[0] == public["continuation_claim_ref"]
        and _file(segment / "completion.json")[1] == {"result_ref": result_ref,
            "claim_ref": public["continuation_claim_ref"], "fresh_result_ref": public["fresh_result_ref"]},
        "COMPLETION_CHANGED")
    admitted = admit_retained_task_agent_handoff(execution_config=identity["authorization_configuration_ref"]["path"],
        run_root=root, run_id=public["run_id"], ready_execution_ref=identity["ready_execution_ref"])
    _require({key: identity[key] for key in admitted.identity} == admitted.identity
        and all(public.get(key) == value for key, value in admitted.identity.items())
        and public.get("entry") == "retained-task-agent-handoff"
        and public.get("generation_origin") == "retained_task_agent_candidate"
        and public.get("task_agent_execution_ref") == admitted.outcome.execution_ref
        and public.get("task_agent_outcome") == admitted.outcome.to_dict()
        and public.get("model_call_budget") == admitted.original["model_call_budget"]
        and public.get("model_call_count") == admitted.original["model_call_count"]
        and public.get("patch_execution_receipts") == []
        and all(public.get(key) == 0 for key in ("new_provider_sends", "model_calls_performed", "new_builds"))
        and public.get("actual_write_performed") is False, "PUBLIC_RESULT_CHANGED")
    fresh = _checked(public["fresh_result_ref"])
    _require(_file(segment / "graph_result.json")[0] == public["fresh_result_ref"], "GRAPH_RESULT_CHANGED")
    if public.get("status") == "awaiting_approval":
        store = ArtifactStore(root / "a")
        checkpoint = store.semantic_read_immutable(ArtifactRef(**public["workflow_checkpoint"]))["data"]
        snapshot = checkpoint["state_snapshot"]
        _require(checkpoint.get("run_id") == public["run_id"]
            and checkpoint.get("request_id") == admitted.manifest["request_id"]
            and checkpoint.get("checkpoint_stage") == snapshot.get("current_stage") == "dry_run_apply"
            and snapshot["artifacts"] == public["artifacts"]
            and snapshot["runtime_summary"]["authorization_configuration_ref"]
            == admitted.identity["authorization_configuration_ref"], "CHECKPOINT_CHANGED")
        for role, key in (("feature_spec", "feature_spec_artifact"), ("evidence_pack", "evidence_pack_artifact"),
                ("implementation_plan", "implementation_plan_artifact"), ("patch_proposal", "patch_proposal_artifact"),
                ("proposal_review", "proposal_review_artifact"), ("dry_run_result", "dry_run_artifact")):
            ref = ArtifactRef(**fresh[key])
            _require(ref.path == snapshot["artifacts"][role] and ref.artifact_hash == snapshot["artifact_hashes"][role],
                "CHECKPOINT_ARTIFACT_CHANGED")
            store.semantic_read_immutable(ref)
    return public
