"""Bound one explicit semantic return to existing Repair, Evidence and Planning owners."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.libs.llm.agent_turn import AgentToolCall
from src.mod_agent.fresh_planning_continuation import (
    PatchContinuationError,
    PatchContinuationInputV1,
    _artifact_ref_dict,
    _canonical_sha256,
    _patch_continuation_state,
    _read_hashed_repair_file,
    _repair_json,
    _source_tree_snapshot_sha256,
    admit_active_worker_failure,
    preflight_patch_continuation,
)
from src.mod_agent.planning.model_selection import ModelPlanningTargetSelector
from src.mod_agent.proposal.behavior_feedback_view import project_execution_reference_paths
from src.mod_agent.proposal.patch_agent_loop import (
    PATCH_UPSTREAM_REQUEST_TOOL_NAME,
    PatchAgentLoopPolicy,
    PlanBoundSourceInspector,
    _validate_upstream_request,
)


def validate_product_worker_upstream(*, execution, state, artifact_store):
    """Admit a request from this Product worker, with optional current failure.

    A request before the first generated Patch has no failed-code/report lineage.
    The Product journal owns both cases; no Fresh continuation is constructed.
    """
    from src.mod_agent.artifacts import ArtifactRef
    from src.mod_agent.fresh_planning_continuation import (
        _csharp_source_snapshot, _validate_selected_plan, _validate_source_items,
    )

    def require(condition, message):
        if not condition:
            raise PatchContinuationError("PATCH_UPSTREAM_ATTEMPT_MISMATCH", "upstream_admission", message)

    from src.mod_agent.model_call_budget import is_meter_only, total_only
    budget = execution.budget_snapshot()
    if execution.replan_count and not (total_only(budget) or is_meter_only(budget)):
        raise PatchContinuationError("PATCH_UPSTREAM_CONTINUATION_BUDGET_EXHAUSTED", "upstream_admission",
            "This logical Product task has already consumed its one upstream cycle.")
    require(execution.last_worker is not None and execution.last_state is not None
            and bool(execution.journals), "No current Product worker journal owns this request.")
    selected = {kind: _artifact_ref_dict(ArtifactRef(feature_id=state["feature_id"], artifact_type=kind,
        path=state["artifacts"][kind], artifact_hash=state["artifact_hashes"][kind]))
        for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
    original = execution.last_state
    require(all(state[key] == original[key] for key in ("run_id", "request_id", "feature_id"))
            and all(original["artifacts"].get(kind) == ref["path"]
                    and original["artifact_hashes"].get(kind) == ref["artifact_hash"]
                    for kind, ref in selected.items()), "The upstream request belongs to another Product task or Plan.")
    journal = json.loads(Path(execution.journals[-1].path).read_text(encoding="utf-8"))
    receipt = execution.patch_attempt_receipt()
    snapshot = _source_tree_snapshot_sha256(_csharp_source_snapshot(execution.workspace_root))
    require(journal.get("continuation_selection") == "product_graph_current_implementation_plan"
            and journal.get("request_id") == state["request_id"]
            and journal.get("parent_run_id") == state["run_id"]
            and journal.get("selected_artifacts") == selected
            and journal.get("factory_receipt") == receipt
            and receipt == execution.last_worker.patch_attempt_receipt()
            and journal.get("generated_against_snapshot_sha256") == snapshot
            and journal.get("current_snapshot_sha256") == snapshot,
            "The current Product journal, worker output and live source must agree.")
    payloads = {kind: artifact_store.semantic_read_immutable(ArtifactRef(**ref))["data"]
                for kind, ref in selected.items()}
    _validate_selected_plan(payloads["implementation_plan"], payloads["evidence_pack"], feature_id=state["feature_id"])
    _validate_source_items(payloads["evidence_pack"], execution.workspace_root)
    policy = execution.last_worker.patch_agent_loop_policy
    inspector = PlanBoundSourceInspector(workspace_root=execution.workspace_root,
        implementation_plan=payloads["implementation_plan"], evidence_pack=payloads["evidence_pack"], policy=policy)
    terminal = receipt.get("upstream_decision") or {}
    arguments = terminal.get("upstream_request") or {}
    require(terminal.get("code") == "PATCH_AGENT_UPSTREAM_REQUESTED", "There is no explicit worker upstream request.")
    call_arguments = {key: arguments.get(key) for key in ("required_owner", "reason", "source_refs")}
    code, validated = _validate_upstream_request(AgentToolCall(call_id="product-upstream-admission",
        name=PATCH_UPSTREAM_REQUEST_TOOL_NAME, arguments=call_arguments), inspector, policy, tool_available=True)
    require(validated is not None and arguments == validated, f"Current upstream request failed source admission: {code}.")
    turns = receipt.get("worker_turns") or []
    require(bool(turns) and turns[-1].get("status") == "responded", "No retained final worker response exists.")
    response = _repair_json(turns[-1].get("response_ref"))
    calls = response.get("tool_calls") or []
    require(len(calls) == 1 and calls[0].get("name") == PATCH_UPSTREAM_REQUEST_TOOL_NAME
            and calls[0].get("arguments") == call_arguments, "The retained worker response requests different upstream material.")
    context = {**deepcopy(payloads), "selected_artifacts": selected,
        "workspace_snapshot_sha256": snapshot, "source_snapshot_sha256": inspector.source_snapshot_sha256,
        "failure_report": None, "failed_response_ref": None, "failure_report_ref": None,
        "previous_response": None, "generated_changes_sha256": None,
        "current_failure_available": False, "approval_inherited": False}
    evaluations = (receipt.get("patch_agent_loop") or {}).get("behavior_evaluations") or []
    if evaluations:
        context.update(_product_current_worker_failure(context=context, receipt=receipt,
            workspace_root=execution.workspace_root))
        _validate_current_failure_feedback(context, receipt)
    elif getattr(execution.last_worker, "_comparison_checkpoint", None) is not None:
        from src.mod_agent.comparison_checkpoint import admit_checkpoint_verifier

        checked = execution.last_worker._comparison_checkpoint
        require(checked.provenance["current_artifacts"] == selected
                and receipt.get("comparison_checkpoint") == checked.provenance,
                "Inherited diagnostic facts must belong to this exact checked trial selection.")
        verifier = admit_checkpoint_verifier(checked, execution.last_worker.behavior_evaluator)
        # Scope: an admitted experiment prefix is diagnostic history, not a newly executed candidate.
        # 为什么要有它：首次新动作可以请求重新调查；不能因本段尚未再次Build而把真实C1/F1丢成“没有失败”，也不能伪造本段执行次数。
        context.update(_product_current_worker_failure(context=context, receipt=checked.history,
            workspace_root=execution.workspace_root))
        context.update(failure_origin="verified_historical_comparison_prefix",
            current_segment_failure_executed=False,
            inherited_failure_provenance=deepcopy(verifier["provenance"]))
        actual_request = _repair_json(turns[-1]["request_ref"])
        retained_messages = [asdict(message) for message in checked.request.messages[2:]]
        require(_canonical_sha256(actual_request) == turns[-1]["request_sha256"]
                and actual_request["messages"][2:2 + len(retained_messages)] == retained_messages,
                "The new upstream response must consume the actual inherited C1/F1 messages.")
    else:
        require(not receipt.get("model_response") and not receipt.get("model_responses")
                and execution.last_worker._repair_context is None,
                "Generated or inherited code cannot be treated as a request before the first Patch.")
    upstream = {"origin": "model", **call_arguments, "validated_request": validated,
        "source_worker_response_ref": deepcopy(turns[-1]["response_ref"]),
        "source_attempt_id": journal["run_id"], "failure_report_ref": deepcopy(context["failure_report_ref"]),
        "preserved_plan_boundaries": deepcopy(payloads["implementation_plan"].get("side_effect_boundaries", []))}
    return context, upstream


def _product_current_worker_failure(*, context, receipt, workspace_root):
    """Reconstruct the latest executed output from the current Product receipt."""
    from src.libs.llm import ChatResponse
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch
    from src.mod_agent.local_entry import _external_evidence_items, _patch_generation_obligation_projection
    from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator

    try:
        evaluations = receipt["patch_agent_loop"]["behavior_evaluations"]
        latest = evaluations[-1]
        if (latest.get("admission") != "accepted" or latest.get("execution_origin") != "executed"
                or latest.get("status") not in {"behavior_failed", "needs_evidence"}):
            raise ValueError("Upstream recovery requires the current admitted executed failure.")
        report_ref = latest["report_ref"]
        report = _repair_json(report_ref)
        if any(latest.get(key) != report.get(key) for key in (
                "status", "failure_owner", "owner_binding_ids", "observations", "source_snapshot_sha256",
                "generated_changes_sha256", "execution_origin")):
            raise ValueError("The failure differs from its saved result.")
        if report["source_snapshot_sha256"] != context["source_snapshot_sha256"]:
            raise ValueError("The failure belongs to different selected source bytes.")
        response_hash = latest.get("model_response_sha256") or report.get("model_response_sha256")
        if "model_response_sha256" in report and report["model_response_sha256"] != response_hash:
            raise ValueError("The failure names a different model response.")
        identities = [receipt.get("model_response"), *(receipt.get("model_responses") or []),
            ((receipt.get("patch_agent_loop") or {}).get("behavior_correction") or {}).get("response")]
        identities = [item for item in identities if isinstance(item, Mapping) and item.get("output_sha256") == response_hash]
        if not identities or not response_hash:
            raise ValueError("No current retained response matches this checked output.")
        response = identities[-1]
        response_ref = {"path": response["path"], "sha256": response_hash}
        raw = _read_hashed_repair_file(response_ref).decode("utf-8")
        implementation = context["implementation_plan"]
        obligations = _patch_generation_obligation_projection(feature_spec=context["feature_spec"],
            implementation=implementation, planned_changes=list(implementation["proposed_changes"]),
            evidence_pack=context["evidence_pack"])
        generator = LLMExactPatchGenerator(None)
        prepared = generator.prepare_native_contract(user_request=context["feature_spec"]["user_request"],
            **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
            evidence_items=_external_evidence_items(context["evidence_pack"]["items"]),
            binding_registry=implementation["binding_registry"], constraints=obligations["side_effect_boundaries"],
            target_behavior=obligations["target_behavior"], acceptance_criteria=obligations["acceptance_criteria"],
            runtime_verification_plan=obligations["runtime_verification_plan"],
            required_change_units=obligations["required_change_units"], required_validation_by_binding={
                item["edit_binding_id"]: item["expected_validation"] for item in obligations["required_change_units"]})
        generated = generator.materialize_prepared_native(prepared, ChatResponse(content=raw, model=response["model"]))
        if _canonical_sha256(generated.changes) != report["generated_changes_sha256"]:
            raise ValueError("The observations were measured against different generated changes.")
        materialized = materialize_exact_text_patch(workspace_root=workspace_root,
            diff_intent=generated.changes, implementation_plan=implementation)
        if latest.get("materialized_code_identity") != {"baseline_sha256": materialized.baseline_sha256,
                "code_sha256": materialized.code_sha256, "files": sorted(materialized.files)}:
            raise ValueError("The observations do not match the actual materialized code.")
        if "verification_ref" in report:
            _read_hashed_repair_file(report["verification_ref"])
        for observation in report["observations"]:
            if "result_ref" in observation:
                _read_hashed_repair_file(observation["result_ref"])
        return {"previous_response": raw, "failed_response_ref": response_ref,
            "failed_diff_intent": list(generated.changes), "generated_changes_sha256": _canonical_sha256(generated.changes),
            "failure_report_ref": deepcopy(report_ref), "failure_report": report,
            "active_failure_evaluation_index": len(evaluations), "current_failure_available": True}
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise PatchContinuationError("PATCH_UPSTREAM_FAILURE_BINDING_REQUIRED", "upstream_admission", str(exc)) from exc


def prepare_product_upstream_assessment(*, execution, state, artifact_store):
    """Revalidate Product dependencies and the retained current request before assessment."""
    from src.mod_agent.artifacts import ArtifactRef

    original = execution.last_state
    context, upstream = validate_product_worker_upstream(execution=execution, state=original, artifact_store=artifact_store)
    repair_ref = ArtifactRef(feature_id=state["feature_id"], artifact_type="repair_plan",
        path=state["artifacts"]["repair_plan"], artifact_hash=state["artifact_hashes"]["repair_plan"])
    repair = artifact_store.semantic_read_immutable(repair_ref)["data"]
    recorded = state["runtime_summary"].get("patch_upstream_return") or {}
    if (recorded.get("assessment_only") is not True or recorded.get("repair_plan_hash") != repair_ref.artifact_hash
            or repair["repair_context"]["upstream_request"] != upstream
            or repair["repair_context"]["product_context_sha256"] != _canonical_sha256(context)
            or context != execution.upstream_context):
        raise PatchContinuationError("PATCH_UPSTREAM_ASSESSMENT_INVALID", "upstream_admission",
            "The current Product assessment no longer matches its admitted worker context.")
    unchanged = all(state["artifacts"].get(kind) == ref["path"]
                    and state["artifact_hashes"].get(kind) == ref["artifact_hash"]
                    for kind, ref in context["selected_artifacts"].items())
    model_report = project_execution_reference_paths(context["failure_report"])
    return {"current_plan": deepcopy(context["implementation_plan"]),
        "selected_artifact_hashes": {kind: ref["artifact_hash"] for kind, ref in context["selected_artifacts"].items()},
        "repair_plan_hash": repair_ref.artifact_hash, "previous_response": context["previous_response"],
        "failed_response_sha256": (context["failed_response_ref"] or {}).get("sha256"),
        "failure_report": model_report, "failure_report_sha256": (context["failure_report_ref"] or {}).get("sha256"),
        "generated_changes_sha256": context["generated_changes_sha256"],
        "source_snapshot_sha256": context["source_snapshot_sha256"],
        "current_failure_available": context["current_failure_available"],
        "retain_current_plan_allowed": unchanged, "approval_inherited": False}


def prepare_patch_upstream_assessment(*, request, state, artifact_store):
    """Read the original Plan and current bound failure for one model assessment.

    Exact reference and live-source checks remain deterministic. The returned
    model view contains hashes instead of local Artifact/file locations.
    """
    from src.mod_agent.artifacts import ArtifactRef

    upstream = state["runtime_summary"].get("patch_upstream_return")
    if (not isinstance(upstream, Mapping) or upstream.get("assessment_only") is not True
            or upstream.get("repair_plan_hash") != state["artifact_hashes"].get("repair_plan")):
        raise PatchContinuationError("PATCH_UPSTREAM_ASSESSMENT_INVALID", "upstream_admission",
            "The current RepairPlan must own this explicit upstream assessment.")
    admitted = preflight_patch_continuation(request, artifact_store=artifact_store)
    original_refs = {kind: getattr(request, kind + "_ref")
                     for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
    unchanged = all(state["artifacts"].get(kind) == ref.path
                    and state["artifact_hashes"].get(kind) == ref.artifact_hash
                    for kind, ref in original_refs.items())
    repair_ref = ArtifactRef(feature_id=state["feature_id"], artifact_type="repair_plan",
        path=state["artifacts"]["repair_plan"], artifact_hash=state["artifact_hashes"]["repair_plan"])
    repair = artifact_store.semantic_read_immutable(repair_ref)["data"]
    context = repair["repair_context"]
    report = _repair_json(context["failure_report_ref"])
    if (report != context["failure_report"]
            or context["upstream_request"]["failure_report_ref"] != context["failure_report_ref"]
            or context["workspace_snapshot_sha256"] != _source_tree_snapshot_sha256(admitted.source_tree_snapshot)):
        raise PatchContinuationError("PATCH_UPSTREAM_FAILURE_BINDING_REQUIRED", "upstream_admission",
            "The assessment must use the current checked failure over unchanged source bytes.")
    raw = _read_hashed_repair_file(context["failed_response_ref"]).decode("utf-8")
    model_report = project_execution_reference_paths(report)
    return {
        "current_plan": deepcopy(admitted.implementation_plan),
        "selected_artifact_hashes": {kind: ref.artifact_hash for kind, ref in original_refs.items()},
        "repair_plan_hash": repair_ref.artifact_hash,
        "previous_response": raw,
        "failed_response_sha256": context["failed_response_ref"]["sha256"],
        "failure_report": model_report,
        "failure_report_sha256": context["failure_report_ref"]["sha256"],
        "generated_changes_sha256": context["generated_changes_sha256"],
        "source_snapshot_sha256": context["source_snapshot_sha256"],
        "retain_current_plan_allowed": unchanged,
        "approval_inherited": False,
    }


def validate_patch_upstream_selection(request: PatchContinuationInputV1, selection: Mapping[str, Any], *, artifact_store=None):
    """A recommendation selects an assessment owner, never writable source or approval."""
    if not isinstance(selection, Mapping) or set(selection) != {
        "origin", "required_owner", "reason", "source_refs", "source_attempt_ref"
    } or selection["origin"] not in {"human", "model"}:
        raise PatchContinuationError("PATCH_UPSTREAM_SELECTION_INVALID", "upstream_admission", "Explicit semantic owner selection is required.")
    admitted = preflight_patch_continuation(request, artifact_store=artifact_store)
    if admitted.repair_context is None:
        raise PatchContinuationError("PATCH_UPSTREAM_FAILURE_REQUIRED", "upstream_admission", "This return requires the existing bound post-Proposal failure context.")
    arguments = {key: deepcopy(selection[key]) for key in ("required_owner", "reason", "source_refs")}
    policy = PatchAgentLoopPolicy(enable_supporting_reads=True, allow_upstream_request=True)
    inspector = PlanBoundSourceInspector(workspace_root=request.workspace_root,
        implementation_plan=admitted.implementation_plan, evidence_pack=admitted.evidence_pack, policy=policy)
    code, validated = _validate_upstream_request(
        AgentToolCall(call_id="upstream-admission", name=PATCH_UPSTREAM_REQUEST_TOOL_NAME, arguments=arguments),
        inspector, policy, tool_available=True)
    if validated is None:
        raise PatchContinuationError(code, "upstream_admission", "The owner recommendation has invalid refs or source identity.")
    if selection["origin"] == "model":
        source = _repair_json(selection["source_attempt_ref"])
        expected = {kind: _artifact_ref_dict(ref) for kind, ref in (
            ("feature_spec", request.feature_spec_ref), ("evidence_pack", request.evidence_pack_ref),
            ("implementation_plan", request.implementation_plan_ref))}
        receipt = source.get("factory_receipt") or {}
        terminal = receipt.get("upstream_decision") or {}
        turns = receipt.get("worker_turns") or []
        snapshot = _source_tree_snapshot_sha256(admitted.source_tree_snapshot)
        if (source.get("selected_artifacts") != expected or terminal.get("code") != "PATCH_AGENT_UPSTREAM_REQUESTED"
                or source.get("generated_against_snapshot_sha256") != snapshot
                or source.get("current_snapshot_sha256") != snapshot
                or terminal.get("upstream_request") != validated or not turns):
            raise PatchContinuationError("PATCH_UPSTREAM_ATTEMPT_MISMATCH", "upstream_admission", "The source attempt does not carry this exact model return.")
        admitted, _ = _validate_worker_failure_binding(admitted, receipt, arguments,
            request=request, artifact_store=admitted.artifact_store)
    elif selection["source_attempt_ref"] is not None:
        raise PatchContinuationError("PATCH_UPSTREAM_SELECTION_INVALID", "upstream_admission", "A human selection must not impersonate a model attempt.")
    return admitted, {**deepcopy(dict(selection)), "validated_request": validated,
        "failure_report_ref": deepcopy(admitted.repair_context["failure_report_ref"]),
        "preserved_plan_boundaries": deepcopy(admitted.implementation_plan.get("side_effect_boundaries", []))}


def _validate_worker_failure_binding(admitted, receipt, arguments, *, request, artifact_store):
    context = admitted.repair_context
    recorded = receipt.get("repair_context") or {}
    if (recorded.get("context_sha256") != _canonical_sha256(context)
            or recorded.get("failure_report_sha256") != _canonical_sha256(context["failure_report"])
            or recorded.get("previous_response_sha256") != hashlib.sha256(context["previous_response"].encode()).hexdigest()):
        raise PatchContinuationError("PATCH_UPSTREAM_FAILURE_BINDING_REQUIRED", "upstream_admission",
            "The worker must originate from this exact bound C1/F1 before any newer failure is consumed.")
    turns = receipt.get("worker_turns") or []
    if not turns or turns[-1].get("status") != "responded":
        raise PatchContinuationError("PATCH_UPSTREAM_RESPONSE_MISMATCH", "upstream_admission", "No retained final worker response exists.")
    response = _repair_json(turns[-1].get("response_ref"))
    calls = response.get("tool_calls") or []
    if len(calls) != 1 or calls[0].get("name") != PATCH_UPSTREAM_REQUEST_TOOL_NAME or calls[0].get("arguments") != arguments:
        raise PatchContinuationError("PATCH_UPSTREAM_RESPONSE_MISMATCH", "upstream_admission", "The retained model response does not request this owner and refs.")
    if (receipt.get("patch_agent_loop") or {}).get("behavior_evaluations"):
        admitted = admit_active_worker_failure(request, receipt, artifact_store)
        _validate_current_failure_feedback(admitted.repair_context, receipt)
    return admitted, deepcopy(turns[-1]["response_ref"])


def _validate_current_failure_feedback(context, receipt):
    """An upstream request must consume F2, not merely coexist with its saved result."""
    loop = receipt.get("patch_agent_loop") or {}
    correction = loop.get("behavior_correction") or {}
    execution = correction.get("execution") or {}
    feedback = correction.get("validator") or {}
    observations = deepcopy(context["failure_report"]["observations"])
    for observation in observations:
        if isinstance(observation, dict) and isinstance(observation.get("result_ref"), dict):
            observation["result_ref"] = {"sha256": observation["result_ref"].get("sha256")}
    if (execution.get("termination") != receipt.get("upstream_decision")
            or execution.get("status") != "blocked" or correction.get("status") != "blocked"
            or feedback.get("report_sha256") != context["failure_report_ref"]["sha256"]
            or feedback.get("generated_changes_sha256") != context["generated_changes_sha256"]
            or feedback.get("source_snapshot_sha256") != context["source_snapshot_sha256"]
            or feedback.get("observations") != observations):
        raise PatchContinuationError("PATCH_UPSTREAM_FAILURE_BINDING_REQUIRED", "upstream_admission",
            "The upstream response is not from the revision segment that consumed this current failure.")


def validate_active_worker_upstream(request, receipt, *, artifact_store):
    terminal = receipt.get("upstream_decision") or {}
    arguments = terminal.get("upstream_request") or {}
    if terminal.get("code") != "PATCH_AGENT_UPSTREAM_REQUESTED":
        raise PatchContinuationError("PATCH_UPSTREAM_SELECTION_INVALID", "upstream_admission", "There is no explicit worker upstream request.")
    selection = {"origin": "human", "source_attempt_ref": None,
        **{key: arguments.get(key) for key in ("required_owner", "reason", "source_refs")}}
    admitted, validated = validate_patch_upstream_selection(request, selection, artifact_store=artifact_store)
    if arguments != validated["validated_request"]:
        raise PatchContinuationError("PATCH_UPSTREAM_ATTEMPT_MISMATCH", "upstream_admission", "The live request's source identity differs.")
    admitted, ref = _validate_worker_failure_binding(admitted, receipt,
        {key: selection[key] for key in ("required_owner", "reason", "source_refs")},
        request=request, artifact_store=artifact_store)
    return admitted, {**validated, "origin": "model", "source_worker_response_ref": ref,
        "failure_report_ref": deepcopy(admitted.repair_context["failure_report_ref"]),
        "source_attempt_id": request.attempt_id}


def run_patch_upstream_replanning(*, request: PatchContinuationInputV1, selection: Mapping[str, Any],
        planning_target_selection_factory: ModelPlanningTargetSelector, evidence_pack_factory=None,
        run_id: str | None = None):
    """Run the existing Graph through one upstream assessment, ending at its new Plan."""
    from src.mod_agent.artifacts import ArtifactStore
    from src.mod_agent.workflow import build_graph

    if run_id is not None and (not isinstance(run_id, str) or not run_id or run_id.strip() != run_id):
        raise ValueError("An explicit nonempty invocation identity is required.")
    store = ArtifactStore(request.artifact_store_root)
    admitted, validated = validate_patch_upstream_selection(request, selection, artifact_store=store)
    if not isinstance(planning_target_selection_factory, ModelPlanningTargetSelector):
        raise ValueError("Upstream replanning requires the model-owned target selector; explicit target lists are not a fallback.")
    identity = hashlib.sha256(json.dumps(validated, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    state = _patch_continuation_state(run_id=run_id or f"patch_upstream_{identity[:24]}", request_id=f"patch_upstream_{identity[:24]}",
        request=request, admitted=admitted)
    graph = build_graph(store, use_default_proposal_git_binding=False, patch_continuation_mode=request,
        patch_upstream_request=selection, planning_target_selection_factory=planning_target_selection_factory,
        evidence_pack_factory=evidence_pack_factory)
    return graph.invoke(state)


def prepare_replanned_patch_input(*, prior_request: PatchContinuationInputV1, state: Mapping[str, Any],
        attempt_id: str | None = None) -> PatchContinuationInputV1:
    """Resolve new exact refs; no inherited failed-run context, worker config or execution approval."""
    from src.mod_agent.artifacts import ArtifactRef

    if state.get("current_stage") != "implementation_plan" or state.get("blockers"):
        raise ValueError("A nonblocking replanned result is required before preparing Patch.")
    refs = {kind: ArtifactRef(feature_id=state["feature_id"], artifact_type=kind,
        path=state["artifacts"][kind], artifact_hash=state["artifact_hashes"][kind])
        for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
    if refs["feature_spec"] != prior_request.feature_spec_ref:
        raise ValueError("Upstream reassessment must preserve the original FeatureSpec.")
    result = PatchContinuationInputV1(contract_version=prior_request.contract_version,
        artifact_store_root=prior_request.artifact_store_root, workspace_root=prior_request.workspace_root,
        feature_spec_ref=refs["feature_spec"], evidence_pack_ref=refs["evidence_pack"],
        implementation_plan_ref=refs["implementation_plan"], attempt_id=attempt_id)
    preflight_patch_continuation(result)
    return result
