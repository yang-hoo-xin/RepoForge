"""Exact v2 delivery checkpoints, isolated Apply and public Build continuations.

为什么要有它：新合同不能借旧Plan恢复；真正授权、执行原件和Apply后的源码前像须共同连续。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.execution.patch_apply import (
    ExactTextPatchApplier,
    _human_semantic_decision_binding,
    _validate_human_semantic_decision,
    attach_patch_apply_failure,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_handoff import revalidate_handoff_v2
from src.mod_agent.task_agent_v2_state import (
    TaskAgentV2State,
    _no_links,
    load_v2_artifact,
    save_v2_artifact,
)


def _require(value, reason):
    if not value:
        raise ValueError("V2_DELIVERY_" + reason)


def _snapshot(root):
    from src.mod_agent.reviewed_patch_continuation import _copy_source_snapshot
    _no_links(Path(root))
    for item in Path(root).rglob("*"):
        if any(part in {".git", "bin", "obj", ".idea"} for part in item.relative_to(root).parts):
            continue
        _no_links(item)
    return _copy_source_snapshot(Path(root))


def _artifact(store, reference, kind):
    ref = ArtifactRef(**reference)
    _require(ref.artifact_type == kind, "ARTIFACT_ROLE_CHANGED")
    envelope = store.semantic_read_immutable(ref)
    _require(envelope["status"] == "success", "ARTIFACT_NOT_SUCCESSFUL")
    return envelope["data"]


def create_v2_delivery_checkpoint(*, store, feature_id, run_id, request_id, proposal_ref, review_ref, dry_run_ref):
    """Persist exact read-only handoff; creation grants no Apply permission.

    为什么要有它：正常入口的待交付状态需要可独立恢复的原件，不能依赖仍在内存的Agent对象。
    """
    refs = {"patch_proposal": asdict(proposal_ref), "proposal_review": asdict(review_ref), "dry_run_result": asdict(dry_run_ref)}
    proposal, review, dry = (_artifact(store, refs[k], k) for k in refs)
    handoff = revalidate_handoff_v2(proposal["context_pack"]["task_agent_handoff_v2"], store=store)
    actual = ExactTextPatchApplier().dry_run_v2(proposal=proposal, review=review, store=store)
    _require(all(dry.get(k) == v for k, v in actual.items()), "DRY_RUN_CHANGED")
    task = TaskAgentV2State(store.root, handoff.task_contract_ref)
    baseline = task.load(task.contract["baseline_manifest_ref"])
    execution = read_bound_json(handoff.execution_ref)
    _require(execution["run_id"] == run_id and execution["request_id"] == request_id
        and execution["initial_context"]["feature_spec_ref"]["feature_id"] == feature_id, "TASK_IDENTITY_CHANGED")
    prior_resources = task.contract.get("prior_delivery_resource_record", {})
    _require(all(type(prior_resources.get(key, 0)) is int and prior_resources.get(key, 0) >= 0
        for key in ("delivery_applies", "delivery_builds")), "RESOURCE_RECORD_INVALID")
    _require(execution["budget_after"]["consumed"] >= prior_resources.get("model_budget", {}).get("consumed", 0),
        "RESOURCE_RECORD_RESET")
    return save_v2_artifact(store.root, "delivery-checkpoint", {
        "schema": "task-agent-delivery-checkpoint.v2", "stage": "awaiting_apply", "feature_id": feature_id,
        "run_id": run_id, "request_id": request_id, "artifact_refs": refs, "handoff": handoff.to_dict(),
        "source_root": baseline["source_root"], "source_snapshot": _snapshot(baseline["source_root"]),
        "candidate_source_id": handoff.candidate_source_id, "diff_hash": actual["diff_hash"],
        "resource_record": {"model_budget": execution["budget_after"],
            "delivery_applies": prior_resources.get("delivery_applies", 0),
            "delivery_builds": prior_resources.get("delivery_builds", 0)},
        "approval_granted": False, "whole_task_delivered": False})


def finish_resumed_v2_handoff(*, owner, outcome, state):
    """Consume an already executed resumed outcome through the normal product nodes.

    为什么要有它：跨进程恢复后的候选仍须走同一Proposal/Review/Dry-run，不能再次调用模型或凭恢复状态直接交付。
    """
    from langgraph.graph import END, START, StateGraph

    from src.mod_agent.workflow.graph import (
        _make_real_graph_node,
        _run_dry_run_apply,
        _run_generate_patch_proposal_v2,
        _run_review_proposal,
    )
    from src.mod_agent.workflow.state import ModAgentState

    _require(getattr(owner, "protocol", None) == "v2" and outcome.status == "candidate_ready", "RESUMED_CANDIDATE_REQUIRED")
    execution = read_bound_json(outcome.execution_ref)
    _require(state["run_id"] == execution["run_id"] and state["request_id"] == execution["request_id"], "RESUMED_TASK_CHANGED")
    arguments = {"execution_ref": outcome.execution_ref, "initial_context": outcome.initial_context}
    _intent, result_file, replay = _operation(owner.store.root, "product_handoff", outcome.execution_ref["sha256"], arguments)
    if replay is not None:
        _checkpoint(owner.store.root, replay["v2_delivery_checkpoint_ref"])
        return replay
    graph = StateGraph(ModAgentState)
    graph.add_node("generate_patch_proposal", _make_real_graph_node("generate_patch_proposal", lambda current:
        _run_generate_patch_proposal_v2(current, owner.store, owner=owner, factory=lambda *_args: outcome)))
    graph.add_node("review_proposal", _make_real_graph_node("review_proposal", lambda current:
        _run_review_proposal(current, owner.store, require_executable_patch=True)))
    graph.add_node("dry_run_apply", _make_real_graph_node("dry_run_apply", lambda current:
        _run_dry_run_apply(current, owner.store)))
    graph.add_edge(START, "generate_patch_proposal")
    graph.add_conditional_edges("generate_patch_proposal", lambda current: "stop" if current["blockers"] else "continue",
        {"stop": END, "continue": "review_proposal"})
    graph.add_conditional_edges("review_proposal", lambda current: "stop" if current["blockers"] else "continue",
        {"stop": END, "continue": "dry_run_apply"})
    graph.add_edge("dry_run_apply", END)
    finished = graph.compile().invoke(deepcopy(state))
    _require(not finished["blockers"] and finished["current_stage"] == "dry_run_apply", "RESUMED_PRODUCT_HANDOFF_FAILED")
    refs = {kind: {"feature_id": finished["feature_id"], "artifact_type": kind,
        "path": finished["artifacts"][kind], "artifact_hash": finished["artifact_hashes"][kind]}
        for kind in ("patch_proposal", "proposal_review", "dry_run_result")}
    result = {"schema": "task-agent-delivery-handoff.v2", "status": "awaiting_delivery", "protocol": "v2",
        "run_id": state["run_id"], "request_id": state["request_id"], "feature_id": finished["feature_id"],
        "task_agent_outcome": outcome.to_dict(), "artifact_refs": refs,
        "v2_delivery_checkpoint_ref": finished["runtime_summary"]["v2_delivery_checkpoint_ref"],
        "trace_events": finished["trace_events"], "whole_task_delivered": False,
        "new_model_calls": 0, "product_source_writes_performed": False}
    return _finish(owner.store.root, result_file, result)


def _checkpoint(artifact_store_root, checkpoint_ref, *, live=True):
    store = ArtifactStore(artifact_store_root)
    checkpoint = load_v2_artifact(store.root, checkpoint_ref)
    _require(checkpoint.get("schema") == "task-agent-delivery-checkpoint.v2"
        and checkpoint.get("stage") == "awaiting_apply", "CHECKPOINT_REQUIRED")
    refs = checkpoint["artifact_refs"]
    proposal, review, dry = (_artifact(store, refs[k], k) for k in ("patch_proposal", "proposal_review", "dry_run_result"))
    handoff = revalidate_handoff_v2(checkpoint["handoff"], store=store)
    _require(proposal["context_pack"]["task_agent_handoff_v2"] == checkpoint["handoff"]
        and dry["candidate_ref"] == handoff.candidate_ref and dry["diff_hash"] == checkpoint["diff_hash"], "CHECKPOINT_CONTENT_CHANGED")
    if live:
        current = ExactTextPatchApplier().dry_run_v2(proposal=proposal, review=review, store=store)
        _require(current["diff_hash"] == checkpoint["diff_hash"]
            and _snapshot(checkpoint["source_root"]) == checkpoint["source_snapshot"], "SOURCE_CHANGED")
    return store, checkpoint, handoff, proposal, review, dry


def prepare_v2_delivery(*, artifact_store_root, checkpoint_ref, isolated_root, operation_id):
    """Display one exact human/delegated action without writing or consuming it."""
    store, checkpoint, handoff, proposal, review, dry = _checkpoint(artifact_store_root, checkpoint_ref)
    _require(isinstance(operation_id, str) and operation_id.strip() and len(operation_id) <= 200, "OPERATION_ID_INVALID")
    root = Path(isolated_root)
    _no_links(root)
    _require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "ISOLATED_ROOT_REQUIRED")
    root = root.resolve()
    source, artifacts = Path(checkpoint["source_root"]).resolve(), Path(store.root).resolve()
    _require(not any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in ((root, source), (root, artifacts))),
        "SOURCE_STORE_DESTINATION_OVERLAP")
    destination = root / hashlib.sha256(operation_id.encode()).hexdigest()[:32]
    binding = {"schema": "task-agent-delivery-binding.v2", "checkpoint_ref": checkpoint_ref,
        "operation_id": operation_id, "isolated_root": str(root), "isolated_workspace": str(destination),
        "source_snapshot_sha256": canonical_hash(checkpoint["source_snapshot"]), "diff_hash": dry["diff_hash"],
        "candidate_source_id": handoff.candidate_source_id, "candidate_ref": handoff.candidate_ref,
        "proposal_ref": checkpoint["artifact_refs"]["patch_proposal"], "allowed_effects": ["isolated_apply"]}
    # 为什么要有它：相同代码diff的审批不能被搬到另一个写入目标；标准审批hash同时承诺展示的执行范围。
    approval_diff_hash = canonical_hash({"candidate_diff_hash": dry["diff_hash"], "execution_binding": binding})
    human = _human_semantic_decision_binding(proposal=proposal, diff_hash=approval_diff_hash, context_hashes={
        "proposal_payload_sha256": canonical_hash(proposal), "proposal_review_payload_sha256": canonical_hash(review),
        "semantic_assurance_payload_sha256": canonical_hash(proposal["semantic_assurance"])})
    return {"entry": "apply-patch", "protocol": "v2", "status": "awaiting_approval", "read_only": True,
        "displayed_binding": binding, "human_semantic_decision_binding": human,
        "unified_diff": dry["unified_diff"], "diff_hash": approval_diff_hash, "candidate_diff_hash": dry["diff_hash"],
        "approval_identity_scope": "candidate_diff_and_displayed_execution_binding",
        "product_source_writes_performed": False, "builds_performed": 0, "model_calls_performed": 0}


def prepare_v2_task_delegation(*, artifact_store_root, displayed_binding, policy_id, authorization_ref,
        allow_build=False):
    """Trusted host API; consume real preexisting delegation, never model output.

    为什么要有它：已有效委托可以继续使用，但新协议不能自行授予权限或冒充人工语义审阅。
    """
    authorization = read_bound_json(authorization_ref)
    _require(authorization.get("origin") == "task_delegation_policy" and authorization.get("decision") == "approve"
        and authorization.get("policy_id") == policy_id and authorization.get("runtime") is False
        and authorization.get("deployment") is False, "HOST_DELEGATION_REQUIRED")
    prepared = prepare_v2_delivery(artifact_store_root=artifact_store_root, checkpoint_ref=displayed_binding["checkpoint_ref"],
        isolated_root=displayed_binding["isolated_root"], operation_id=displayed_binding["operation_id"])
    _require(prepared["displayed_binding"] == displayed_binding, "DISPLAY_BINDING_CHANGED")
    checkpoint = load_v2_artifact(artifact_store_root, displayed_binding["checkpoint_ref"])
    _require(authorization.get("request_id") == checkpoint["request_id"]
        and authorization.get("run_id") == checkpoint["run_id"]
        and authorization.get("isolated_root") == displayed_binding["isolated_root"]
        and "isolated_apply" in authorization.get("allowed_effects", [])
        and (not allow_build or "build" in authorization.get("allowed_effects", [])), "DELEGATION_SCOPE_CHANGED")
    return save_v2_artifact(artifact_store_root, "delivery-delegation", {
        "schema": "task-agent-delivery-delegation.v2", "origin": "task_delegation_policy",
        "authorization_ref": authorization_ref, "policy_id": policy_id, "displayed_binding": displayed_binding,
        "allow_build": allow_build, "human_review_completed": False})


def _operation(root, kind, operation_id, arguments, *, reserve=True):
    folder = Path(root) / "v2" / "delivery-operations"
    _no_links(folder)
    folder.mkdir(parents=True, exist_ok=True)
    key = canonical_hash({"kind": kind, "operation_id": operation_id})
    intent, result = folder / (key + ".intent.json"), folder / (key + ".result.json")
    _no_links(intent)
    _no_links(result)
    expected = {"kind": kind, "operation_id": operation_id, "arguments": arguments}
    if intent.exists():
        _require(json.loads(intent.read_bytes()) == expected, "OPERATION_ID_REUSED")
        if result.exists():
            ref = json.loads(result.read_bytes())["receipt_ref"]
            return intent, result, {**load_v2_artifact(root, ref), "receipt_ref": ref, "replayed": True}
        raise ValueError("V2_DELIVERY_OPERATION_INFLIGHT_REQUIRES_RECONCILIATION")
    if not reserve:
        return intent, result, None
    with intent.open("x", encoding="utf-8") as stream:
        json.dump(expected, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return intent, result, None


def _finish(root, result_path, payload):
    ref = save_v2_artifact(root, "delivery-receipt", payload)
    _no_links(result_path)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump({"receipt_ref": ref}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {**payload, "receipt_ref": ref, "replayed": False}


def _validate_applied_receipt(store, apply_ref):
    """Reconstruct applied bytes and their operation origin before another effect.

    为什么要有它：自洽的副本snapshot不是原候选证据，必须从原baseline与实际批准补丁重建结果。
    """
    applied = load_v2_artifact(store.root, apply_ref)
    _require(applied.get("schema") == "task-agent-delivery-receipt.v2"
        and applied.get("status") == "applied_in_isolation", "APPLIED_RECEIPT_REQUIRED")
    _s, checkpoint, handoff, proposal, review, _dry = _checkpoint(store.root, applied["checkpoint_ref"], live=False)
    binding = applied["displayed_binding"]
    root = Path(binding["isolated_root"])
    _no_links(root)
    destination = root / hashlib.sha256(binding["operation_id"].encode()).hexdigest()[:32]
    _no_links(destination)
    _require(root.is_absolute() and str(root.resolve()) == str(root)
        and binding["checkpoint_ref"] == applied["checkpoint_ref"]
        and binding["diff_hash"] == checkpoint["diff_hash"]
        and binding["proposal_ref"] == checkpoint["artifact_refs"]["patch_proposal"]
        and binding["source_snapshot_sha256"] == canonical_hash(checkpoint["source_snapshot"])
        and binding["allowed_effects"] == ["isolated_apply"]
        and binding["candidate_ref"] == applied["candidate_ref"] == handoff.candidate_ref
        and binding["candidate_source_id"] == applied["candidate_source_id"] == handoff.candidate_source_id
        and binding["isolated_workspace"] == applied["applied_source_root"] == str(destination), "APPLY_BINDING_CHANGED")
    folder = Path(store.root) / "v2/delivery-operations"
    key = canonical_hash({"kind": "apply", "operation_id": binding["operation_id"]})
    intent, result = folder / (key + ".intent.json"), folder / (key + ".result.json")
    _no_links(intent)
    _no_links(result)
    _require(json.loads(intent.read_bytes()) == {"kind": "apply", "operation_id": binding["operation_id"],
        "arguments": {"displayed_binding": binding, "approval": applied["approval"]}}
        and json.loads(result.read_bytes()).get("receipt_ref") == apply_ref, "APPLY_OPERATION_ORIGIN_CHANGED")
    if applied["approval"]["origin"] == "human_review":
        human = _human_semantic_decision_binding(proposal=proposal,
            diff_hash=canonical_hash({"candidate_diff_hash": checkpoint["diff_hash"], "execution_binding": binding}),
            context_hashes={"proposal_payload_sha256": canonical_hash(proposal), "proposal_review_payload_sha256": canonical_hash(review),
                "semantic_assurance_payload_sha256": canonical_hash(proposal["semantic_assurance"])})
        decision = _validate_human_semantic_decision(applied["approval"]["decision"], expected_binding=human)
        _require(decision["decision"] == "approve", "APPLY_APPROVAL_CHANGED")
    else:
        _require(applied["approval"]["origin"] == "task_delegation_policy", "APPLY_APPROVAL_CHANGED")
        delegated = load_v2_artifact(store.root, applied["approval"]["delegation_ref"])
        _require(delegated["displayed_binding"] == binding and delegated["origin"] == "task_delegation_policy", "APPLY_DELEGATION_CHANGED")
        authorization = read_bound_json(delegated["authorization_ref"])
        _require(authorization["decision"] == "approve", "APPLY_DELEGATION_CHANGED")
    expected = {row["path"]: deepcopy(row) for row in checkpoint["source_snapshot"]}
    for change in handoff.changes:
        path, raw = change["display_file"], change["patch"]["after"].encode("utf-8")
        _require(path in expected and expected[path]["sha256"] == change["baseline_file_sha256"], "APPLY_BASELINE_CHANGED")
        expected[path].update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
    expected_snapshot = [expected[path] for path in sorted(expected)]
    _require(applied["applied_source_snapshot"] == expected_snapshot
        and _snapshot(destination) == expected_snapshot, "APPLIED_POSTIMAGE_CHANGED")
    _require(applied["resource_record"] == {**checkpoint["resource_record"],
        "delivery_applies": checkpoint["resource_record"].get("delivery_applies", 0) + 1}, "APPLY_RESOURCE_RECORD_CHANGED")
    return applied, checkpoint, handoff, destination


def apply_v2_delivery(*, artifact_store_root, displayed_binding, human_semantic_decision=None, task_delegation_ref=None):
    """Apply approved bytes only to a new isolated copy; repeated IDs reuse facts."""
    _require((human_semantic_decision is not None) != (task_delegation_ref is not None), "ONE_APPROVAL_SOURCE_REQUIRED")
    checkpoint_ref = displayed_binding["checkpoint_ref"]
    prepared = prepare_v2_delivery(artifact_store_root=artifact_store_root, checkpoint_ref=checkpoint_ref,
        isolated_root=displayed_binding["isolated_root"], operation_id=displayed_binding["operation_id"])
    _require(prepared["displayed_binding"] == displayed_binding, "DISPLAY_BINDING_CHANGED")
    store, checkpoint, handoff, _proposal, _review, _dry = _checkpoint(artifact_store_root, checkpoint_ref)
    if human_semantic_decision is not None:
        decision = _validate_human_semantic_decision(human_semantic_decision,
            expected_binding=prepared["human_semantic_decision_binding"])
        if decision["decision"] == "reject":
            return {"entry": "apply-patch", "protocol": "v2", "status": "rejected", "product_source_writes_performed": False}
        approval = {"origin": "human_review", "decision": decision}
    else:
        delegated = load_v2_artifact(store.root, task_delegation_ref)
        expected = prepare_v2_task_delegation(artifact_store_root=store.root, displayed_binding=displayed_binding,
            policy_id=delegated["policy_id"], authorization_ref=delegated["authorization_ref"], allow_build=delegated["allow_build"])
        _require(expected == task_delegation_ref, "DELEGATION_CHANGED")
        approval = {"origin": "task_delegation_policy", "delegation_ref": task_delegation_ref, "human_review_completed": False}
    arguments = {"displayed_binding": displayed_binding, "approval": approval}
    _intent, result_file, replay = _operation(store.root, "apply", displayed_binding["operation_id"], arguments)
    if replay is not None:
        _require(_snapshot(replay["applied_source_root"]) == replay["applied_source_snapshot"], "APPLIED_SOURCE_CHANGED")
        return replay
    source, destination = Path(checkpoint["source_root"]), Path(displayed_binding["isolated_workspace"])
    payload = {"schema": "task-agent-delivery-receipt.v2", "entry": "apply-patch", "protocol": "v2",
        "stage": "applied", "checkpoint_ref": checkpoint_ref, "approval": approval,
        "displayed_binding": displayed_binding, "candidate_ref": handoff.candidate_ref,
        "candidate_source_id": handoff.candidate_source_id, "applied_source_root": str(destination),
        "resource_record": {**checkpoint["resource_record"],
            "delivery_applies": checkpoint["resource_record"].get("delivery_applies", 0) + 1},
        "whole_task_delivered": False, "model_calls_performed": 0, "builds_performed": 0,
        "source_workspace_written": False, "write_observations": []}
    try:
        _require(not destination.exists(), "DESTINATION_ALREADY_EXISTS")
        from src.mod_agent.reviewed_patch_continuation import _COPY_IGNORED
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*sorted(_COPY_IGNORED)))
        _require(_snapshot(destination) == checkpoint["source_snapshot"]
            and _snapshot(source) == checkpoint["source_snapshot"], "COPY_SOURCE_CHANGED")
        for change in handoff.changes:
            raw_target = destination / change["display_file"]
            _no_links(raw_target)
            target = raw_target.resolve(strict=True)
            _require(target.is_relative_to(destination), "TARGET_OUTSIDE_COPY")
            _require(hashlib.sha256(target.read_bytes()).hexdigest() == change["baseline_file_sha256"], "PREIMAGE_CHANGED")
            content = change["patch"]["after"].encode("utf-8")
            temporary = target.with_name(target.name + ".repoforge-delivery.tmp")
            _no_links(temporary)
            observation = {"path": change["display_file"], "expected_sha256": hashlib.sha256(content).hexdigest(), "state": "started"}
            payload["write_observations"].append(observation)
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            _require(temporary.read_bytes() == content, "TEMPORARY_READBACK_FAILED")
            _no_links(raw_target)
            _no_links(temporary)
            os.replace(temporary, target)
            _require(target.read_bytes() == content, "APPLY_READBACK_FAILED")
            observation["state"] = "confirmed"
        _require(_snapshot(source) == checkpoint["source_snapshot"], "ORIGINAL_SOURCE_CHANGED")
        payload.update(status="applied_in_isolation", applied_source_snapshot=_snapshot(destination),
            product_source_writes_performed=True, actual_write_performed=True, stopped_before="build")
    except Exception as error:
        confirmed = [row["path"] for row in payload["write_observations"] if row["state"] == "confirmed"]
        payload.update(status="apply_failed", error_type=type(error).__name__, reason=str(error),
            applied_source_snapshot=_snapshot(destination) if destination.is_dir() else [],
            product_source_writes_performed=True if confirmed else None,
            actual_write_performed=True if confirmed else None, stopped_before="build")
        attach_patch_apply_failure(error, {"actual_write_performed": payload["actual_write_performed"],
            "real_patch_applied": False, "changed_files": confirmed, "write_completion": "partial" if confirmed else "unknown",
            "write_failure": {"operation": "v2_isolated_apply", "file_observations": payload["write_observations"]}})
        payload["failure_feedback_ref"] = save_v2_artifact(store.root, "delivery-feedback", {
            "schema": "task-agent-delivery-feedback.v2", "visibility": "public", "execution_status": "apply_failed",
            "checkpoint_ref": checkpoint_ref, "candidate_ref": handoff.candidate_ref,
            "candidate_source_id": handoff.candidate_source_id, "applied_source_root": str(destination),
            "applied_source_snapshot": payload["applied_source_snapshot"], "resource_record": payload["resource_record"],
            "observations": [{"kind": "apply_failure", "error_type": type(error).__name__, "message": str(error),
                "write_observations": payload["write_observations"]}],
            "retry_allowed": False, "approval_inherited": False, "authority_inherited": False})
        retained = _finish(store.root, result_file, payload)
        error.delivery_receipt_ref = retained["receipt_ref"]
        error.failure_feedback_ref = payload["failure_feedback_ref"]
        raise
    return _finish(store.root, result_file, payload)


def build_v2_delivery(*, artifact_store_root, apply_ref, operation_id, allow_build=False, task_delegation_ref=None):
    """Run the frozen repository Build/test contract on the actual applied copy."""
    store = ArtifactStore(artifact_store_root)
    applied, checkpoint, handoff, destination = _validate_applied_receipt(store, apply_ref)
    original_execution = read_bound_json(handoff.execution_ref)
    if task_delegation_ref is not None:
        delegated = load_v2_artifact(store.root, task_delegation_ref)
        _require(applied["approval"].get("delegation_ref") == task_delegation_ref and delegated["allow_build"] is True,
            "BUILD_DELEGATION_CHANGED")
        authorization = read_bound_json(delegated["authorization_ref"])
        _require("build" in authorization["allowed_effects"] and authorization["decision"] == "approve", "BUILD_NOT_AUTHORIZED")
    else:
        _require(allow_build is True, "BUILD_NOT_AUTHORIZED")
    _require(isinstance(operation_id, str) and operation_id.strip(), "OPERATION_ID_INVALID")
    arguments = {"apply_ref": apply_ref, "operation_id": operation_id, "allow_build": allow_build, "task_delegation_ref": task_delegation_ref}
    # 为什么要有它：一个已Apply版本仅消费一次Build，换operation_id不能清零或绕过崩溃后的未确认执行。
    _intent, result_file, replay = _operation(store.root, "build", apply_ref, arguments)
    if replay is not None:
        if replay.get("verification_ref") is not None:
            report = read_bound_json(replay["verification_ref"])
            for artifact in report.get("built_artifact_refs", []):
                _require(hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"], "BUILT_ARTIFACT_CHANGED")
        return replay
    task = TaskAgentV2State(store.root, handoff.task_contract_ref)
    output_root = destination.parent / (destination.name + "-build-" + canonical_hash(operation_id)[:12])
    resources = {**applied["resource_record"], "delivery_builds": applied["resource_record"].get("delivery_builds", 0) + 1}
    payload = {"schema": "task-agent-delivery-receipt.v2", "entry": "apply-patch", "protocol": "v2", "stage": "build",
        "apply_ref": apply_ref, "checkpoint_ref": applied["checkpoint_ref"], "candidate_ref": handoff.candidate_ref,
        "candidate_source_id": handoff.candidate_source_id, "applied_source_root": str(destination),
        "applied_source_snapshot": applied["applied_source_snapshot"], "resource_record": resources,
        "builds_performed": 1, "model_calls_performed": 0, "product_source_writes_performed": False,
        "runtime_executions_performed": 0, "whole_task_delivered": False}
    execution_started = False
    try:
        from src.mod_agent.domain.csharp_repository_backend import (
            CSharpRepositoryBackend,
            CSharpRepositoryProfile,
        )
        profile = CSharpRepositoryProfile(task.contract["execution_profile_ref"])
        worker = read_bound_json(original_execution["worker_binding"]["configuration_ref"])
        fixed = worker["behavior_evaluator"]["kwargs"]["expected_fixed_inputs"]
        backend = CSharpRepositoryBackend(profile_ref=profile.reference, source_root=destination,
            workspace_root=destination.parent, output_root=output_root, expected_fixed_inputs=fixed)
        execution_started = True
        result = backend.run_build_test(source_root=destination, feature_id=checkpoint["feature_id"], input_execution_mode="isolated_apply")
        raw = read_bound_json(result["report_ref"])
        _require(raw["status"] == result["status"] and raw["execution_origin"] == "executed"
            and Path(raw["source_root"]).resolve() == destination
            and _snapshot(destination) == applied["applied_source_snapshot"], "BUILD_SOURCE_OR_REPORT_CHANGED")
        for artifact in raw.get("built_artifact_refs", []):
            path = Path(artifact["path"]).resolve(strict=True)
            _require(path.is_relative_to(destination) and hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"], "BUILD_ARTIFACT_CHANGED")
        payload.update(status="built" if result["status"] == "passed" else "build_failed", verification_ref=result["report_ref"],
            build_result=result["build_result"].to_artifact_payload(), observations=result.get("observations", []),
            delivered_artifact_refs=raw.get("built_artifact_refs", []),
            whole_task_delivered=result["status"] == "passed",
            delivery_scope="approved_isolated_source_and_bound_public_build_tests",
            limitations=["Delivery covers the frozen public build/test contract; independent hidden scoring and runtime are not asserted."])
    except Exception as error:
        # 为什么要有它：没有执行终态的异常不能伪装为业务反例；中断保留未决intent，恢复不能自动重复副作用。
        payload.update(status="build_unknown" if execution_started else "build_blocked", builds_performed=None if execution_started else 0,
            error_type=type(error).__name__, reason=str(error), verification_ref=None,
            observations=[{"kind": "execution_failure", "error_type": type(error).__name__, "message": str(error)}])
    if payload["status"] != "built":
        payload["failure_feedback_ref"] = save_v2_artifact(store.root, "delivery-feedback", {
            "schema": "task-agent-delivery-feedback.v2", "visibility": "public", "apply_ref": apply_ref,
            "checkpoint_ref": applied["checkpoint_ref"], "candidate_ref": handoff.candidate_ref,
            "candidate_source_id": handoff.candidate_source_id, "applied_source_root": str(destination),
            "applied_source_snapshot": applied["applied_source_snapshot"], "resource_record": resources,
            "verification_ref": payload.get("verification_ref"), "observations": payload.get("observations", []),
            "execution_status": payload["status"], "retry_allowed": False,
            "authority_inherited": False, "approval_inherited": False})
    timing_origin = original_execution.get("task_timing_origin_ref")
    if timing_origin is not None:
        from src.mod_agent.task_agent_v2_execution import task_timing_snapshot
        try:
            # 为什么要有它：交付等待须从原任务入口延续；时间记录失败不能抹去已经执行的Build与工件收据。
            payload["task_timing_observation"] = task_timing_snapshot(timing_origin,
                "product_build_return", payload["status"], native_execution_ref=handoff.execution_ref)
        except (OSError, ValueError, TypeError, KeyError) as error:
            payload["task_timing_observation"] = {"status": "unknown", "origin_ref": timing_origin,
                "event": "product_build_return", "native_execution_ref": handoff.execution_ref,
                "reason": str(error)}
    return _finish(store.root, result_file, payload)


def delivery_recovery_context(*, artifact_store_root, receipt_ref):
    """Return applied preimages and cumulative facts; grant no new edit permission.

    为什么要有它：Apply后恢复须针对实际代码重新读取，旧baseline和审批不能套用到新修订。
    """
    store = ArtifactStore(artifact_store_root)
    receipt = load_v2_artifact(artifact_store_root, receipt_ref)
    _require(receipt.get("status") in {"applied_in_isolation", "built", "build_failed"}, "RECOVERY_RECEIPT_INVALID")
    _require(_snapshot(receipt["applied_source_root"]) == receipt["applied_source_snapshot"], "RECOVERY_SOURCE_CHANGED")
    applied_ref = receipt_ref if receipt["status"] == "applied_in_isolation" else receipt["apply_ref"]
    applied, _checkpoint_data, _handoff, _destination = _validate_applied_receipt(store, applied_ref)
    if receipt["status"] != "applied_in_isolation":
        key = canonical_hash({"kind": "build", "operation_id": applied_ref})
        pointer = Path(store.root) / "v2/delivery-operations" / (key + ".result.json")
        _no_links(pointer)
        _require(json.loads(pointer.read_bytes()).get("receipt_ref") == receipt_ref
            and all(receipt[k] == applied[k] for k in ("checkpoint_ref", "candidate_ref", "candidate_source_id",
                "applied_source_root", "applied_source_snapshot")), "RECOVERY_BUILD_ORIGIN_CHANGED")
        _require(receipt["resource_record"] == {**applied["resource_record"],
            "delivery_builds": applied["resource_record"].get("delivery_builds", 0) + 1}, "RECOVERY_RESOURCE_RECORD_CHANGED")
        report = read_bound_json(receipt["verification_ref"])
        _require(report["source_root"] == receipt["applied_source_root"]
            and (report["status"] == "passed") == (receipt["status"] == "built"), "RECOVERY_BUILD_REPORT_CHANGED")
        for artifact in report.get("built_artifact_refs", []):
            _require(hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"], "RECOVERY_BUILD_ARTIFACT_CHANGED")
    checkpoint = load_v2_artifact(artifact_store_root, receipt["checkpoint_ref"])
    task = TaskAgentV2State(artifact_store_root, checkpoint["handoff"]["task_contract_ref"])
    return {"schema": "task-agent-delivery-recovery.v2", "receipt_ref": receipt_ref,
        "prior_candidate_ref": receipt["candidate_ref"], "applied_source_root": receipt["applied_source_root"],
        "current_source_snapshot": receipt["applied_source_snapshot"], "resource_record": receipt["resource_record"],
        "failure_feedback_ref": receipt.get("failure_feedback_ref"), "requires_fresh_source_views": True,
        "original_authority_ref": task.contract["authority_ref"], "allowed_edit_paths": task.contract["allowed_edit_paths"],
        "host_scope_rebind_required": True,
        "approval_inherited": False, "authority_inherited": False}


def read_completed_v2_delivery(*, artifact_store_root, receipt_ref, native_execution_ref):
    """Read a completed product delivery for the exact native technical execution.

    为什么要有它：经验评测需要消费已有真实交付及产物原件，不能把候选通过或调用者填写的delivered当作交付。
    """
    context = delivery_recovery_context(artifact_store_root=artifact_store_root, receipt_ref=receipt_ref)
    receipt = load_v2_artifact(artifact_store_root, receipt_ref)
    _require(receipt.get("status") == "built" and receipt.get("whole_task_delivered") is True,
        "COMPLETED_DELIVERY_REQUIRED")
    checkpoint = load_v2_artifact(artifact_store_root, receipt["checkpoint_ref"])
    _require(checkpoint["handoff"]["execution_ref"] == native_execution_ref,
        "DELIVERY_NATIVE_EXECUTION_CHANGED")
    report = read_bound_json(receipt["verification_ref"])
    checks = report.get("checks", [])
    _require(report.get("execution_origin") == "executed" and report.get("status") == "passed"
        and report.get("test_execution", {}).get("status") == "executed" and checks
        and all(row.get("passed") is True for row in checks)
        and report["verification_summary"]["checks"] == len(checks)
        and report["verification_summary"].get("failed_checks") == 0
        and report.get("built_artifact_refs") == receipt.get("delivered_artifact_refs")
        and receipt.get("delivered_artifact_refs"), "COMPLETED_DELIVERY_CHECK_CHANGED")
    return {"status": "delivered", "candidate_source_id": receipt["candidate_source_id"],
        "candidate_ref": receipt["candidate_ref"], "native_execution_ref": native_execution_ref,
        "delivery_receipt_ref": receipt_ref, "resource_record": context["resource_record"],
        "delivery_scope": receipt["delivery_scope"], "delivered_artifact_refs": receipt["delivered_artifact_refs"],
        "task_timing_observation": deepcopy(receipt.get("task_timing_observation")),
        "verification": {"passed": True, "candidate_source_id": receipt["candidate_source_id"],
            "native_verification_ref": receipt["verification_ref"], "delivery_receipt_ref": receipt_ref}}


def _save_rebase_worker(root, document):
    raw = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    path = Path(root) / "v2/rebase-workers" / (hashlib.sha256(raw).hexdigest() + ".json")
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        _require(path.read_bytes() == raw, "REBASE_WORKER_CHANGED")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def prepare_v2_delivery_rebase(*, artifact_store_root, receipt_ref, operation_id):
    """Derive a current-code contract from verified Apply and original task scope.

    为什么要有它：正式交付失败后的修订以已Apply字节为起点；继承原任务权限不等于继承旧精确diff审批。
    """
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from src.mod_agent.task_agent_execution import _source_facts, _worker_binding
    from src.mod_agent.task_agent_v2_recovery import load_task_agent_checkpoint, single_task_lock

    _require(isinstance(operation_id, str) and operation_id.strip(), "OPERATION_ID_INVALID")
    root = Path(artifact_store_root).resolve(strict=True)
    context = delivery_recovery_context(artifact_store_root=root, receipt_ref=receipt_ref)
    receipt = load_v2_artifact(root, receipt_ref)
    delivery = load_v2_artifact(root, receipt["checkpoint_ref"])
    prior_ref = delivery["handoff"]["task_contract_ref"]
    logical_lock = "logical-task:" + delivery["run_id"] + ":" + delivery["request_id"]
    with single_task_lock(root, logical_lock):
        prior = TaskAgentV2State(root, prior_ref)
        request = {"receipt_ref": receipt_ref, "requested_operation_id": operation_id, "prior_task_contract_ref": prior_ref}
        _intent, result_file, replay = _operation(root, "delivery_rebase", receipt_ref, request, reserve=False)
        if replay is not None:
            result = validate_v2_delivery_rebase(artifact_store_root=root, rebase_ref=replay["receipt_ref"])
            return {**result, "rebase_ref": replay["receipt_ref"], "replayed": True}
        _require(not prior.head.get("superseded_by_contract_ref"), "PRIOR_TASK_ALREADY_REBASED")
        latest_path = prior.root / "session-checkpoints" / (canonical_hash(prior_ref) + ".json")
        _no_links(latest_path)
        prior_checkpoint_ref = json.loads(latest_path.read_bytes())["checkpoint_ref"]
        loaded = load_task_agent_checkpoint(artifact_store_root=root, checkpoint_ref=prior_checkpoint_ref)
        host, prior_checkpoint, old_head = loaded["host"], loaded["checkpoint"], prior.head
        _require(host["graph_state"]["run_id"] == delivery["run_id"]
            and host["graph_state"]["request_id"] == delivery["request_id"], "REBASE_LOGICAL_TASK_CHANGED")
        authority = prior.load(prior.contract["authority_ref"])
        _require(authority.get("scope") == "file" and authority.get("operation") == "modify_existing",
            "REBASE_MODIFY_EXISTING_FILE_AUTHORITY_REQUIRED")
        paths = prior.contract["allowed_edit_paths"]
        _require(set(paths) <= set(authority["allowed_edit_paths"]), "REBASE_EDIT_SCOPE_NOT_AUTHORIZED")
        original_worker = PatchWorkerConfiguration.load(host["worker_configuration_ref"]["path"])
        _require(original_worker.reference == host["worker_configuration_ref"]
            and authority["worker_configuration_ref"] == original_worker.reference, "REBASE_WORKER_ORIGIN_CHANGED")
        source = Path(context["applied_source_root"]).resolve(strict=True)
        workspace = source.parent
        read_roots = {"target": source}
        for root_id, snapshot_ref in prior.contract["read_roots"].items():
            if root_id == "target":
                continue
            selected = Path(prior.load(snapshot_ref)["source_root"])
            _require(selected.is_relative_to(workspace), "REBASE_READ_DEPENDENCY_OUTSIDE_AUTHORIZED_WORKSPACE")
            read_roots[root_id] = selected
        config = read_bound_json(original_worker.reference)
        kwargs = config["behavior_evaluator"]["kwargs"]
        _require(config["behavior_evaluator"]["factory_name"] == "make_csharp_repository_evaluator"
            and kwargs["profile_ref"] == prior.contract["execution_profile_ref"], "REBASE_CHECKER_PROFILE_CHANGED")
        output_root = workspace / ("rebase-checks-" + canonical_hash(receipt_ref)[:16])
        kwargs.update(source_root=str(source), workspace_root=str(workspace), output_root=str(output_root))
        worker_ref = _save_rebase_worker(root, config)
        worker = PatchWorkerConfiguration.load(worker_ref["path"])
        evaluator = worker.behavior_evaluator()
        identity = evaluator.verification_identity()
        _require(identity == host["verification_identity"], "REBASE_PUBLIC_ACCEPTANCE_CHANGED")
        budget = deepcopy(prior_checkpoint["loop"]["model_budget"])
        _require(budget["consumed"] == old_head["resources"].get("model_calls", 0)
            and budget["consumed"] >= context["resource_record"]["model_budget"]["consumed"], "REBASE_RESOURCE_RECORD_CHANGED")
        resources = {**context["resource_record"], "model_budget": budget}
        origin_ref = prior.save("rebase-origin", {"schema": "task-agent-rebase-origin.v2", "receipt_ref": receipt_ref,
            "prior_task_contract_ref": prior_ref, "prior_checkpoint_ref": prior_checkpoint_ref,
            "prior_host_config_ref": prior_checkpoint["host_config_ref"], "prior_task_head": old_head,
            "prior_execution_ref": delivery["handoff"]["execution_ref"], "prior_candidate_ref": receipt["candidate_ref"],
            "run_id": delivery["run_id"], "request_id": delivery["request_id"], "operation_id": operation_id,
            "source_snapshot": context["current_source_snapshot"], "resource_record": resources,
            "original_authority_ref": prior.contract["authority_ref"], "approval_inherited": False})
        new_authority = prior.save("authority", {"schema": "task-agent-authority.v2", "scope": "file",
            "operation": "modify_existing", "allowed_edit_paths": paths, "read_roots": {key: str(value) for key, value in read_roots.items()},
            "origin": "host_derivation_from_task_authority_and_verified_applied_copy",
            "original_authority_ref": prior.contract["authority_ref"], "rebase_origin_ref": origin_ref,
            "worker_configuration_ref": worker_ref, "execution_profile_ref": prior.contract["execution_profile_ref"],
            "run_id": delivery["run_id"], "request_id": delivery["request_id"], "new_apply_approval_granted": False})
        acceptance = deepcopy(prior.load(prior.contract["public_acceptance_ref"]))
        acceptance["worker_binding"] = _worker_binding(worker_ref, evaluator)
        acceptance_ref = prior.save("acceptance", acceptance)
        reserved = {"schema", "task_id", "authority_ref", "baseline_manifest_ref", "allowed_edit_paths", "read_roots", "scope", "operation", "task_requirements"}
        fields = {key: deepcopy(value) for key, value in prior.contract.items() if key not in reserved}
        fields.update(public_acceptance_ref=acceptance_ref, resource_policy_ref=prior.contract["resource_policy_ref"],
            prior_delivery_resource_record=resources, rebase_origin_ref=origin_ref,
            historical_task_contract_ref=prior_ref, historical_candidate_ref=receipt["candidate_ref"],
            historical_execution_ref=delivery["handoff"]["execution_ref"], failure_feedback_ref=context["failure_feedback_ref"])
        _operation(root, "delivery_rebase", receipt_ref, request)
        current = TaskAgentV2State.create(root, prior.contract["task_id"], source,
            read_roots=read_roots, allowed_edit_paths=paths, authority_ref=new_authority,
            task_requirements=deepcopy(prior.contract["task_requirements"]), contract_fields=fields)
        current.record_resources(deepcopy(old_head["resources"]), operation_id="rebase-resource-baseline:" + prior_ref)
        overrides = {"workspace_root": str(workspace), "worker_configuration_ref": worker_ref,
            "allowed_read_roots": [str(value) for value in read_roots.values()],
            "allowed_edit_paths": [(source / path).relative_to(workspace).as_posix() for path in paths],
            "verification_identity": identity, "source_facts": _source_facts(tuple(read_roots.values())),
            "graph_state": deepcopy(host["graph_state"]), "task_contract_ref": current.contract_ref,
            "model_budget": budget, "deadline_utc": host["deadline_utc"]}
        record = {"schema": "task-agent-delivery-rebase.v2", "status": "prepared", "receipt_ref": receipt_ref,
            "new_task_contract_ref": current.contract_ref, "prior_task_contract_ref": prior_ref,
            "prior_checkpoint_ref": prior_checkpoint_ref, "prior_host_config_ref": prior_checkpoint["host_config_ref"],
            "rebase_origin_ref": origin_ref, "host_overrides": overrides, "resource_record": resources,
            "failure_feedback_ref": context["failure_feedback_ref"], "run_id": delivery["run_id"],
            "request_id": delivery["request_id"], "historical_execution_ref": delivery["handoff"]["execution_ref"],
            "logical_lock_key": logical_lock, "old_source_views_valid": False, "old_check_valid": False,
            "new_apply_approval_granted": False, "new_task_created": False}
        ref = current.save("delivery-rebase", record)
        with prior._transaction("delivery-rebase:" + canonical_hash(receipt_ref),
                {"kind": "delivery_rebase", "receipt_ref": receipt_ref, "new_task_contract_ref": current.contract_ref}) as (head, result, _replay):
            head["superseded_by_contract_ref"] = current.contract_ref
            result.update(rebase_ref=ref, new_task_contract_ref=current.contract_ref)
        with result_file.open("x", encoding="utf-8") as stream:
            json.dump({"receipt_ref": ref}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return {**record, "rebase_ref": ref, "replayed": False}


def validate_v2_delivery_rebase(*, artifact_store_root, rebase_ref):
    """Validate derived authority, exact postimage and original operation before resume."""
    root = Path(artifact_store_root).resolve(strict=True)
    record = load_v2_artifact(root, rebase_ref)
    _require(record.get("schema") == "task-agent-delivery-rebase.v2", "REBASE_RECORD_REQUIRED")
    context = delivery_recovery_context(artifact_store_root=root, receipt_ref=record["receipt_ref"])
    prior = TaskAgentV2State(root, record["prior_task_contract_ref"])
    current = TaskAgentV2State(root, record["new_task_contract_ref"])
    origin = current.load(record["rebase_origin_ref"])
    transition = prior.head["operations"].get("delivery-rebase:" + canonical_hash(record["receipt_ref"]), {})
    _require(prior.head.get("superseded_by_contract_ref") == current.contract_ref
        and transition.get("result", {}).get("rebase_ref") == rebase_ref
        and current.contract.get("rebase_origin_ref") == record["rebase_origin_ref"]
        and origin["prior_task_contract_ref"] == prior.contract_ref
        and origin["receipt_ref"] == record["receipt_ref"], "REBASE_ORIGIN_CHANGED")
    authority, old_authority = current.load(current.contract["authority_ref"]), prior.load(prior.contract["authority_ref"])
    _require(authority["original_authority_ref"] == prior.contract["authority_ref"]
        and authority["allowed_edit_paths"] == current.contract["allowed_edit_paths"] == prior.contract["allowed_edit_paths"]
        and set(authority["allowed_edit_paths"]) <= set(old_authority["allowed_edit_paths"])
        and authority["scope"] == old_authority["scope"] == "file"
        and authority["operation"] == old_authority["operation"] == "modify_existing"
        and authority["new_apply_approval_granted"] is False, "REBASE_AUTHORITY_CHANGED")
    baseline = current.load(current.contract["baseline_manifest_ref"])
    expected = {row["path"]: row["sha256"] for row in context["current_source_snapshot"]}
    utf8_paths = set()
    for path in expected:
        if any(part.casefold() in {".git", ".env", "bin", "obj", ".venv", "__pycache__", ".aws", ".ssh", ".codex",
                "credentials", ".idea", ".vs", "testresults"} or part.casefold().startswith(".env.") for part in Path(path).parts):
            continue
        try:
            (Path(context["applied_source_root"]) / path).read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        utf8_paths.add(path)
    _require(baseline["source_root"] == context["applied_source_root"]
        and set(baseline["files"]) == utf8_paths
        and all(expected.get(path) == row["sha256"] for path, row in baseline["files"].items())
        and current.contract["task_id"] == prior.contract["task_id"]
        and current.contract["task_requirements"] == prior.contract["task_requirements"], "REBASE_BASELINE_CHANGED")
    _require(current.contract["prior_delivery_resource_record"] == record["resource_record"] == origin["resource_record"]
        and record["resource_record"]["model_budget"]["consumed"] >= context["resource_record"]["model_budget"]["consumed"]
        and all(current.head["resources"].get(key, 0) >= value for key, value in origin["prior_task_head"]["resources"].items()),
        "REBASE_RESOURCE_RECORD_CHANGED")
    _require(authority["worker_configuration_ref"] == record["host_overrides"]["worker_configuration_ref"], "REBASE_WORKER_CHANGED")
    host = prior.load(record["prior_host_config_ref"])
    overrides = record["host_overrides"]
    source = Path(context["applied_source_root"])
    _require(overrides["workspace_root"] == str(source.parent)
        and overrides["allowed_edit_paths"] == [(source / path).relative_to(source.parent).as_posix()
            for path in prior.contract["allowed_edit_paths"]]
        and overrides["allowed_read_roots"][0] == str(source)
        and set(overrides["allowed_read_roots"]) == set(authority["read_roots"].values())
        and overrides["graph_state"] == host["graph_state"]
        and overrides["deadline_utc"] == host["deadline_utc"]
        and overrides["verification_identity"] == host["verification_identity"]
        and overrides["model_budget"] == record["resource_record"]["model_budget"]
        and current.contract["resource_policy_ref"] == prior.contract["resource_policy_ref"], "REBASE_HOST_SCOPE_CHANGED")
    original_worker = read_bound_json(host["worker_configuration_ref"])
    expected_worker = deepcopy(original_worker)
    expected_worker["behavior_evaluator"]["kwargs"].update(source_root=str(source), workspace_root=str(source.parent),
        output_root=str(source.parent / ("rebase-checks-" + canonical_hash(record["receipt_ref"])[:16])))
    _require(read_bound_json(authority["worker_configuration_ref"]) == expected_worker, "REBASE_WORKER_SCOPE_CHANGED")
    return record
