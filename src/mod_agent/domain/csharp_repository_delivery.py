"""Restore the declared repository build/test contract through existing Apply/Build APIs.

为什么要有它：当前候选的测试合同必须随原checkpoint恢复，不能在审批后降级为只编译或改用其他测试。
"""
from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path, PurePosixPath

from src.mod_agent.domain.csharp_repository_backend import (
    CSharpRepositoryBackend,
    CSharpRepositoryProfile,
)
from src.mod_agent.model_qualification import read_bound_json


def repository_binding_from_request(request):
    values = [item.canonical_payload for item in request.input_bindings if item.binding_kind == "repository_build_and_test"]
    if not values:
        return None
    if len(values) != 1 or request.target_domain != "csharp_repository" or request.requested_completion_scope != "compile_only":
        raise ValueError("REPOSITORY_DELIVERY_DOMAIN_MISMATCH")
    return deepcopy(values[0])


def validate_repository_binding(binding, workspace_root):
    if not isinstance(binding, dict) or set(binding) != {
            "configuration_ref", "fixed_inputs", "target_relative_root", "worker_configuration_ref"}:
        raise ValueError("REPOSITORY_DELIVERY_BINDING_INVALID")
    root = Path(workspace_root).resolve(strict=True)
    relative = PurePosixPath(binding["target_relative_root"])
    if relative.is_absolute() or "\\" in str(relative) or ".." in relative.parts:
        raise ValueError("REPOSITORY_DELIVERY_TARGET_INVALID")
    target = (root / relative.as_posix()).resolve(strict=True)
    if target == root or not target.is_relative_to(root) or not target.is_dir():
        raise ValueError("REPOSITORY_DELIVERY_TARGET_INVALID")
    profile = CSharpRepositoryProfile(binding["configuration_ref"])
    profile.verify_fixed_inputs(target, binding["fixed_inputs"])
    worker = read_bound_json(binding["worker_configuration_ref"])
    evaluator = worker["behavior_evaluator"]
    parameters = evaluator["kwargs"]
    from src.mod_agent.domain import csharp_repository_backend
    factory = evaluator["factory_ref"]
    factory_path = Path(factory["path"]).resolve(strict=True)
    if (evaluator["factory_name"] != "make_csharp_repository_evaluator"
            or hashlib.sha256(factory_path.read_bytes()).hexdigest() != factory["sha256"]
            or hashlib.sha256(Path(csharp_repository_backend.__file__).read_bytes()).hexdigest() != factory["sha256"]
            or parameters["profile_ref"] != binding["configuration_ref"]
            or parameters["expected_fixed_inputs"] != binding["fixed_inputs"]
            or Path(parameters["workspace_root"]).resolve() != root
            or Path(parameters["source_root"]).resolve() != target):
        raise ValueError("REPOSITORY_DELIVERY_CHECKER_CHANGED")
    read_bound_json(binding["configuration_ref"])
    return profile, target


def validate_repository_changes(binding, workspace_root, changes):
    profile, target = validate_repository_binding(binding, workspace_root)
    prefix = target.relative_to(Path(workspace_root).resolve()).as_posix()
    allowed = {prefix + "/" + relative for relative in profile.data["allowed_edit_paths"]}
    if any(row.get("path") not in allowed for row in changes):
        raise ValueError("REPOSITORY_DELIVERY_EDIT_SCOPE_CHANGED")


def repository_build_factory(binding, workspace_root):
    profile, target = validate_repository_binding(binding, workspace_root)
    root = Path(workspace_root).resolve(strict=True)
    backend = CSharpRepositoryBackend(profile_ref=profile.reference, source_root=target,
        workspace_root=root, output_root=root / "repository_formal_validation",
        expected_fixed_inputs=binding["fixed_inputs"])

    def build(destination, feature_id, mode):
        if Path(destination).resolve() != root or mode != "isolated_apply":
            raise ValueError("REPOSITORY_DELIVERY_BUILD_SCOPE_CHANGED")
        validate_repository_binding(binding, root)
        outcome = backend.run_build_test(source_root=target, feature_id=feature_id, input_execution_mode=mode)
        return outcome["build_result"]

    return build


def validate_repository_candidate(*, binding, workspace_root, store, initial, verification, report, proposal):
    """Reuse the actual candidate handoff validator, without frozen benchmark-specific callers."""
    from src.mod_agent.candidate_execution import verification_identity
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from src.mod_agent.proposal.builder import PatchProposalBuilder
    from src.mod_agent.task_agent_execution import (
        load_task_agent_outcome,
        validate_task_agent_handoff,
    )

    profile, target = validate_repository_binding(binding, workspace_root)
    root = Path(workspace_root).resolve()
    worker = PatchWorkerConfiguration.load(binding["worker_configuration_ref"]["path"])
    if worker.reference != binding["worker_configuration_ref"]:
        raise ValueError("REPOSITORY_DELIVERY_WORKER_CHANGED")
    current_identity = verification_identity(worker.behavior_evaluator())
    if report.get("verification_identity") != current_identity:
        raise ValueError("REPOSITORY_DELIVERY_VERIFIER_CHANGED")
    execution_ref = initial.get("task_agent_execution_ref")
    native_lineage = proposal.get("context_pack", {}).get("task_agent_lineage")
    if execution_ref is None and native_lineage is None:
        return _validate_workflow_candidate(binding=binding, root=root, store=store, initial=initial,
            verification=verification, report=report, proposal=proposal, worker=worker,
            current_identity=current_identity)
    if not isinstance(execution_ref, dict):
        raise ValueError("REPOSITORY_DELIVERY_NATIVE_EXECUTION_REQUIRED")
    outcome = load_task_agent_outcome(execution_ref, artifact_store_root=store.root)
    execution = read_bound_json(execution_ref)
    if (initial.get("task_agent_outcome") != outcome.to_dict()
            or outcome.budget_after != initial.get("model_call_budget")
            or execution["run_id"] != initial["run_id"] or execution["request_id"] != initial["request_id"]):
        raise ValueError("REPOSITORY_DELIVERY_OUTCOME_CHANGED")
    handoff = validate_task_agent_handoff(outcome, store=store, workspace_root=root,
        expected_initial_context=outcome.initial_context,
        allowed_edit_paths=tuple((target / name).relative_to(root).as_posix() for name in profile.data["allowed_edit_paths"]),
        worker_configuration_ref=worker.reference, expected_verification_identity=current_identity,
        expected_candidate_mode=execution.get("candidate_mode", "baseline"))
    if (handoff.check["report"]["report_ref"] != verification
            or [PatchProposalBuilder._change_intent(change) for change in handoff.changes] != proposal["diff_intent"]
            or proposal.get("context_pack", {}).get("task_agent_lineage", {}).get("execution_ref") != execution_ref):
        raise ValueError("REPOSITORY_DELIVERY_PROPOSAL_CHANGED")
    return {"source": "normal_product_repository_binding_and_native_candidate",
        "validator_sha256": current_identity["validator_sha256"], "worker_configuration_ref": worker.reference,
        "repository_profile_ref": profile.reference, "task_agent_execution_ref": execution_ref,
        "candidate_ref": outcome.candidate_ref, "check_ref": outcome.check_ref}


def _validate_workflow_candidate(*, binding, root, store, initial, verification, report, proposal,
                                 worker, current_identity):
    """Verify the original workflow response and executed checker, not a native-tool receipt.

    为什么要有它：两种执行器的真实来源不同，workflow不能因没有B专用回执被拒，也不能借别人的通过报告交付。
    """
    from src.mod_agent.artifacts import ArtifactRef
    from src.mod_agent.candidate_execution import materialize_candidate, materialized_code_identity
    from src.mod_agent.model_qualification import canonical_hash
    from src.mod_agent.reviewed_patch_continuation import _verified_terminal

    def inside_ref(value):
        reference = {key: value[key] for key in ("path", "sha256")}
        path = Path(reference["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_REFERENCE_OUTSIDE_RUN")
        return reference

    if report != read_bound_json(inside_ref(verification)) or report.get("status") != "passed":
        raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_REPORT_CHANGED")
    checkpoint = store.semantic_read_immutable(ArtifactRef(**initial["workflow_checkpoint"]))["data"]
    state = checkpoint["state_snapshot"]
    proposal_ref = ArtifactRef(feature_id=initial["feature_id"], artifact_type="patch_proposal",
        path=state["artifacts"]["patch_proposal"], artifact_hash=state["artifact_hashes"]["patch_proposal"])
    plan_ref = ArtifactRef(feature_id=initial["feature_id"], artifact_type="implementation_plan",
        path=state["artifacts"]["implementation_plan"], artifact_hash=state["artifact_hashes"]["implementation_plan"])
    if (store.semantic_read_immutable(proposal_ref)["data"] != proposal
            or report.get("implementation_plan_ref") != {
                key: getattr(plan_ref, key) for key in ("feature_id", "artifact_type", "path", "artifact_hash")}
            or proposal.get("context_pack", {}).get("implementation_plan_hash") != plan_ref.artifact_hash):
        raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_PROPOSAL_CHANGED")
    materialized = materialize_candidate(workspace_root=root,
        implementation_plan=store.semantic_read_immutable(plan_ref)["data"], changes=proposal["diff_intent"])
    if materialized_code_identity(materialized) != report.get("materialized_code_identity"):
        raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_CODE_CHANGED")
    response_sha = report.get("model_response_sha256")
    if {row.get("generation", {}).get("model_response_sha256") for row in proposal["diff_intent"]} != {response_sha}:
        raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_RESPONSE_CHANGED")
    configured = read_bound_json(worker.reference)["behavior_evaluator"]
    expected_configuration = {"configuration_ref": worker.reference, "factory_ref": configured["factory_ref"],
        "factory_name": configured["factory_name"], "kwargs_sha256": canonical_hash(configured["kwargs"])}
    matches = []
    for selected in initial.get("patch_execution_receipts", []):
        reference = inside_ref(selected)
        journal = read_bound_json(reference)
        if (journal.get("parent_run_id") != initial["run_id"]
                or journal.get("request_id") != initial["request_id"]
                or journal.get("feature_id") != initial["feature_id"]):
            raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_RUN_CHANGED")
        terminal = _verified_terminal(journal, initial["feature_id"], journal["run_id"])
        if (terminal.get("status") != "awaiting_approval"
                or terminal.get("parent_product_run_id") != initial["run_id"]
                or terminal.get("workflow_checkpoint") != initial["workflow_checkpoint"]
                or terminal.get("patch_proposal_artifact") != {
                    key: getattr(proposal_ref, key) for key in ("feature_id", "artifact_type", "path", "artifact_hash")}):
            continue
        receipt = journal.get("factory_receipt", {})
        if receipt.get("behavior_evaluator_configuration") != expected_configuration:
            continue
        response = receipt.get("model_response", {})
        if response.get("output_sha256") != response_sha:
            continue
        response_ref = inside_ref({"path": response["path"], "sha256": response_sha})
        raw = Path(response_ref["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != response_sha or response.get("output_utf8_bytes") != len(raw):
            raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_RESPONSE_CHANGED")
        evaluations = receipt.get("patch_agent_loop", {}).get("behavior_evaluations", [])
        if any(row.get("report_ref") == verification and row.get("admission") == "accepted"
                and row.get("callback_invoked") is True and row.get("execution_origin") == "executed"
                and row.get("status") == "passed" and row.get("model_response_sha256") == response_sha
                and row.get("verification_identity") == current_identity
                and row.get("materialized_code_identity") == report["materialized_code_identity"]
                and row.get("generated_changes_sha256") == report["generated_changes_sha256"]
                for row in evaluations):
            matches.append({"receipt_ref": reference, "model_response_ref": response_ref})
    if len(matches) != 1:
        raise ValueError("REPOSITORY_DELIVERY_WORKFLOW_CHECKER_EXECUTION_REQUIRED")
    return {"source": "normal_product_repository_binding_and_workflow_candidate",
        "validator_sha256": current_identity["validator_sha256"], "worker_configuration_ref": worker.reference,
        "repository_profile_ref": binding["configuration_ref"], "verification_report_ref": verification,
        "worker_receipt_refs": [matches[0]["receipt_ref"]], "model_response_ref": matches[0]["model_response_ref"]}
