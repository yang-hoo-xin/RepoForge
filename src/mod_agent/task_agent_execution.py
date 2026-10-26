"""Optional Product technical executor over the existing task session and owners."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.libs.llm import ChatResponse
from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.candidate_execution import (
    materialize_candidate,
    materialized_code_identity,
    validate_candidate_report,
    verification_identity,
)
from src.mod_agent.model_call_budget import is_meter_only, validate_budget
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy, PlanBoundSourceInspector
from src.mod_agent.task_agent_loop import TaskReadBudgetError, read_budget_view
from src.mod_agent.task_agent_owners import make_task_agent_owners
from src.mod_agent.task_agent_tools import TaskAgentContext, TaskAgentSession, TaskContextDocuments


def _require(value, code):
    if not value:
        raise ValueError("TASK_AGENT_EXECUTION_" + code)


def _ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(reference, root):
    _require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256"}, "REFERENCE_INVALID")
    selected = Path(reference["path"]).resolve(strict=True)
    _require(selected.is_relative_to(Path(root).resolve()) and selected != Path(root).resolve(), "REFERENCE_OUTSIDE_STORE")
    return read_bound_json(dict(reference))


def _read_checker_report(reference, *, store_root, workspace_root, worker_configuration_ref):
    """Admit only the checker output root declared by this exact worker configuration.

    为什么要有它：真实检查报告可位于run/v而非ArtifactStore；只扩展检查报告读取，不能把整个run或任意引用都变成可信工件。
    """
    _require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256"}, "REFERENCE_INVALID")
    selected, store, workspace = (Path(reference["path"]).resolve(strict=True),
        Path(store_root).resolve(strict=True), Path(workspace_root).resolve(strict=True))
    if selected != store and selected.is_relative_to(store):
        return _read(reference, store)
    worker = read_bound_json(dict(worker_configuration_ref))
    kwargs = worker.get("behavior_evaluator", {}).get("kwargs", {})
    configured = kwargs.get("output_root") if isinstance(kwargs, Mapping) else None
    _require(isinstance(configured, str) and bool(configured) and Path(configured).is_absolute(),
        "CHECKER_OUTPUT_ROOT_REQUIRED")
    output = Path(configured).resolve(strict=True)
    _require(output.is_dir() and output != workspace and output.is_relative_to(workspace),
        "CHECKER_OUTPUT_ROOT_OUTSIDE_WORKSPACE")
    compilation = kwargs.get("compilation_kwargs")
    if compilation is not None:
        _require(isinstance(compilation, Mapping)
            and Path(compilation.get("workspace_root", "")).resolve() == workspace
            and Path(compilation.get("artifact_store_root", "")).resolve() == store,
            "CHECKER_WORKSPACE_BINDING_CHANGED")
    _require(selected != output and selected.is_relative_to(output), "CHECKER_REPORT_OUTSIDE_OUTPUT")
    return _read(reference, output)


def _worker_binding(reference, evaluator=None):
    worker = read_bound_json(dict(reference))
    selected = worker["behavior_evaluator"]
    _require(isinstance(selected, dict) and _ref(selected["factory_ref"]["path"]) == selected["factory_ref"], "FACTORY_CHANGED")
    binding = {"configuration_ref": dict(reference), "factory_ref": deepcopy(selected["factory_ref"]),
        "factory_name": selected["factory_name"], "kwargs_sha256": canonical_hash(selected["kwargs"])}
    if evaluator is not None:
        receipt = getattr(evaluator, "configuration_receipt", None)
        _require(callable(receipt) and receipt() == binding, "EVALUATOR_CONFIGURATION_CHANGED")
    return binding


def _context(state):
    return TaskAgentContext(**{kind + "_ref": ArtifactRef(feature_id=state["feature_id"], artifact_type=kind,
        path=state["artifacts"][kind], artifact_hash=state["artifact_hashes"][kind])
        for kind in ("feature_spec", "evidence_pack", "implementation_plan")})


def _source_facts(roots):
    from src.mod_agent.product_executor import source_snapshot

    return [{"root": str(root), "files": source_snapshot(root)} for root in roots]


def _admitted_response(value, profile):
    provider = "openai_responses" if profile.get("profile_version") == "external_openai_responses_profile_v1" else "ollama_chat_local"
    _require(value.get("provider") == provider and value.get("model") == profile["model"]
        and value.get("provider_state") is not None and value.get("stop_reason") not in {
            "length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal", "failed", "cancelled"}, "RESPONSE_IDENTITY_CHANGED")
    if provider == "openai_responses":
        raw = value.get("raw_response")
        _require(isinstance(raw, dict) and raw.get("model") == profile["model"] and raw.get("status") == "completed"
            and isinstance(value["provider_state"], list), "RESPONSE_NATIVE_IDENTITY_CHANGED")


def prepare_task_agent_contract(documents, *, workspace_root, evaluator, policy, documentation_holder=None,
        verification_configured=None, protocol="v1"):
    """Reuse the normal Patch contract and its dependency/verification model views."""
    if protocol == "v2":
        # 为什么要有它：v2合同直接消费宿主任务与授权，不能从技术计划生成必改义务或伪造旧Plan。
        from src.mod_agent.task_agent_v2_tools import prepare_v2_contract
        return prepare_v2_contract(documents)
    _require(protocol == "v1", "PROTOCOL_INVALID")
    from src.mod_agent.local_entry import (
        _external_evidence_items,
        _patch_agent_initial_messages,
        _patch_generation_obligation_projection,
    )
    from src.mod_agent.proposal.dependency_documentation import prepare_dependency_documentation
    from src.mod_agent.proposal.verification_capability import (
        PatchVerificationCapability,
        prepared_verification_capability,
    )

    feature, evidence, plan = documents.feature_spec, documents.evidence_pack, documents.implementation_plan
    obligations = _patch_generation_obligation_projection(feature_spec=feature, evidence_pack=evidence,
        implementation=plan, planned_changes=plan["proposed_changes"])
    prepared = LLMExactPatchGenerator(None).prepare_native_contract(user_request=feature["user_request"],
        evidence_items=_external_evidence_items(evidence["items"]), binding_registry=plan["binding_registry"],
        constraints=obligations["side_effect_boundaries"], target_behavior=obligations["target_behavior"],
        acceptance_criteria=obligations["acceptance_criteria"], runtime_verification_plan=obligations["runtime_verification_plan"],
        required_change_units=obligations["required_change_units"], required_validation_by_binding={
            row["edit_binding_id"]: row["expected_validation"] for row in obligations["required_change_units"]},
        **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}))
    documentation = prepare_dependency_documentation(workspace_root=workspace_root, implementation_plan=plan)
    if documentation_holder is not None:
        documentation_holder["current"] = documentation
    inspector = PlanBoundSourceInspector(workspace_root=workspace_root, implementation_plan=plan,
        evidence_pack=evidence, policy=policy)
    capability = (PatchVerificationCapability(True) if verification_configured is True else
        prepared_verification_capability(evaluator, recovery_enabled=True))
    return replace(prepared, messages=_patch_agent_initial_messages(prepared.messages,
        on_demand_binding_ids=tuple(value for value in inspector.available_read_binding_ids
            if value not in inspector.available_binding_ids),
        verification_questions=tuple(obligations.get("verification_questions", ())),
        dependency_documentation=documentation.model_view() if documentation is not None else None,
        candidate_verification=capability.model_view() if capability is not None else None,
        baseline_compilation_failure=evidence.get("budget_summary", {}).get("baseline_compilation_failure")))


@dataclass(frozen=True)
class TaskAgentPatchOutcome:
    status: str
    termination: dict[str, Any]
    initial_context: TaskAgentContext
    final_context: TaskAgentContext
    execution_ref: dict[str, str] | None
    loop_result_ref: dict[str, str] | None
    claim_ref: dict[str, str] | None
    candidate_ref: dict[str, str] | None
    check_ref: dict[str, str] | None
    finish_response_ref: dict[str, str] | None
    budget_before: dict[str, Any]
    budget_after: dict[str, Any]
    unknown_effects: bool = False
    whole_task_delivered: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        _require(isinstance(value, dict) and set(value) == {field.name for field in fields(cls)}, "OUTCOME_SHAPE_CHANGED")
        data = deepcopy(value)
        for name in ("initial_context", "final_context"):
            selected = data[name]
            _require(isinstance(selected, dict) and set(selected) == {field.name for field in fields(TaskAgentContext)},
                "OUTCOME_CONTEXT_INVALID")
            data[name] = TaskAgentContext(**{kind: ArtifactRef(**reference) for kind, reference in selected.items()})
        return cls(**data)


def load_task_agent_outcome(execution_ref, *, artifact_store_root):
    """Load the exact saved typed outcome; this does not itself admit delivery."""
    document = _read(execution_ref, artifact_store_root)
    _require(document.get("schema_version") == "task-agent-product-execution.v1", "OUTCOME_VERSION_CHANGED")
    return TaskAgentPatchOutcome.from_dict({field.name: deepcopy(document[field.name])
        if field.name != "execution_ref" else dict(execution_ref) for field in fields(TaskAgentPatchOutcome)})


@dataclass(frozen=True)
class ValidatedTaskAgentHandoff:
    final_context: TaskAgentContext
    changes: tuple[dict[str, Any], ...]
    candidate: dict[str, Any]
    check: dict[str, Any]
    execution_ref: dict[str, str]


def validate_task_agent_handoff(outcome, *, store, workspace_root, expected_initial_context,
        allowed_edit_paths, worker_configuration_ref, expected_verification_identity, expected_candidate_mode="baseline"):
    """Read and rebind a real finish, exact tool output, candidate and checker result.

    为什么要有它：完成状态和passed字典本身不授予交付资格；必须证明最后模型动作对应当前计划、源码和实际检查过的候选。
    """
    _require(isinstance(outcome, TaskAgentPatchOutcome) and outcome.status == "candidate_ready"
        and outcome.whole_task_delivered is False and outcome.unknown_effects is False, "HANDOFF_NOT_READY")
    root, workspace = Path(store.root).resolve(), Path(workspace_root).resolve()
    execution = _read(outcome.execution_ref, root)
    claimed = {key: value for key, value in outcome.to_dict().items() if key != "execution_ref"}
    _require(execution.get("schema_version") == "task-agent-product-execution.v1"
        and all(execution.get(key) == value for key, value in claimed.items()), "OUTCOME_RECORD_CHANGED")
    claim = _read(outcome.claim_ref, root)
    _require(expected_candidate_mode in {"baseline", "current"}
        and claim.get("candidate_mode", "baseline") == execution.get("candidate_mode", "baseline") == expected_candidate_mode,
        "CANDIDATE_MODE_CHANGED")
    _require(claim.get("schema_version") == "task-agent-product-claim.v1"
        and all(claim.get(key) == execution.get(key) for key in ("run_id", "request_id", "initial_context", "budget_before",
            "worker_binding", "policy", "allowed_edit_paths", "source_before", "expected_model_identity",
            "model_profile", "expected_verification_identity")), "CLAIM_BINDING_CHANGED")
    _require(outcome.initial_context == expected_initial_context
        and outcome.final_context.feature_spec_ref == expected_initial_context.feature_spec_ref, "INITIAL_CONTEXT_CHANGED")
    _require(execution.get("worker_binding") == _worker_binding(worker_configuration_ref)
        and execution.get("expected_verification_identity") == expected_verification_identity,
        "VERIFICATION_OWNER_CHANGED")
    _require(execution.get("allowed_edit_paths") == list(allowed_edit_paths), "EDIT_AUTHORITY_CHANGED")
    before, after = validate_budget(outcome.budget_before), validate_budget(outcome.budget_after)
    _require(before["maximum"] == after["maximum"] and before.get("policy") == after.get("policy")
        and after["consumed"] >= before["consumed"] and set(before["by_stage"]) == set(after["by_stage"])
        and all(after["by_stage"][role]["consumed"] >= row["consumed"] for role, row in before["by_stage"].items()),
        "BUDGET_CHANGED")
    loop = _read(outcome.loop_result_ref, root)
    _require(loop.get("status") == "completed" and loop.get("unknown_effects") is False
        and loop.get("termination") == {"code": "TASK_AGENT_COMPLETION_ACCEPTED"}
        and outcome.termination == loop["termination"] and loop.get("model_budget") == after,
        "REAL_FINISH_REQUIRED")
    _require(type(loop.get("model_turns")) is int and 0 < loop["model_turns"] <= after["consumed"] - before["consumed"],
        "MODEL_ACCOUNTING_CHANGED")
    final = outcome.final_context
    envelopes = {kind: store.semantic_read_immutable(getattr(final, kind + "_ref"))
        for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
    for kind in envelopes:
        _require(store.check_freshness(getattr(final, kind + "_ref")).is_fresh, "FINAL_CONTEXT_STALE")
    for kind, upstream in (("evidence_pack", (final.feature_spec_ref,)),
            ("implementation_plan", (final.feature_spec_ref, final.evidence_pack_ref))):
        actual = envelopes[kind]["inputs"].get("upstream_artifacts", [])
        _require(sorted((row["artifact_type"], row["path"], row["artifact_hash"]) for row in actual) == sorted(
            (ref.artifact_type, ref.path, ref.artifact_hash) for ref in upstream), "FINAL_CONTEXT_LINEAGE_CHANGED")
    from src.mod_agent.fresh_planning_continuation import (
        _validate_selected_plan,
        _validate_source_items,
    )

    feature, evidence, plan = (envelopes[kind]["data"] for kind in ("feature_spec", "evidence_pack", "implementation_plan"))
    _validate_selected_plan(plan, evidence, feature_id=final.feature_spec_ref.feature_id)
    _validate_source_items(evidence, workspace)
    _require(all(row.get("canonical_target_path") in allowed_edit_paths
        for row in plan["binding_registry"]["edit_bindings"]), "PLAN_EDIT_SCOPE_CHANGED")
    candidate = _read(outcome.candidate_ref, root)
    check = _read(outcome.check_ref, root)
    _require(candidate.get("candidate_id") == "candidate-" + canonical_hash({key: value for key, value in candidate.items()
        if key != "candidate_id"})[:20], "CANDIDATE_ID_CHANGED")
    _require(candidate.get("context") == final.to_dict() and check.get("context_id") == candidate.get("context_id")
        and check.get("candidate_id") == candidate.get("candidate_id") and check.get("candidate_ref") == outcome.candidate_ref,
        "CANDIDATE_CONTEXT_CHANGED")
    from src.mod_agent.current_candidate import (
        candidate_projection_contract,
        context_transfer_content,
        derive_current_revision,
    )

    policy = PatchAgentLoopPolicy(**execution["policy"])
    transitions = []
    for row in execution.get("completed_tool_records", []):
        if (row.get("role") in {"candidate_submit", "upstream"} and row.get("candidate_ref") is not None
                and (not transitions or row["candidate_ref"] != transitions[-1]["candidate_ref"])):
            transitions.append(row)
    # Scope: advancing code includes an authenticated host transfer, not a fictitious model submission.
    # 为什么要有它：重新规划只改变表示且保留字节；交付仍须验证真实调用顺序、每代原始合同和最终检查。
    _require(expected_candidate_mode != "current" or transitions and transitions[-1].get("candidate_ref") == outcome.candidate_ref,
        "CURRENT_CANDIDATE_CHANGED")
    visiting = set()

    def contract_for(selected):
        selected_context = TaskAgentContext(**{key: ArtifactRef(**value) for key, value in selected["context"].items()})
        _require(selected_context.feature_spec_ref == final.feature_spec_ref, "CANDIDATE_REQUIREMENTS_CHANGED")
        historical = {kind: store.semantic_read_immutable(getattr(selected_context, kind + "_ref"))
            for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
        for kind, upstream in (("evidence_pack", (selected_context.feature_spec_ref,)),
                ("implementation_plan", (selected_context.feature_spec_ref, selected_context.evidence_pack_ref))):
            actual = historical[kind]["inputs"].get("upstream_artifacts", [])
            _require(sorted((row["artifact_type"], row["path"], row["artifact_hash"]) for row in actual) == sorted(
                (ref.artifact_type, ref.path, ref.artifact_hash) for ref in upstream), "CANDIDATE_CONTEXT_LINEAGE_CHANGED")
        selected_feature, selected_evidence, selected_plan = (historical[kind]["data"] for kind in
            ("feature_spec", "evidence_pack", "implementation_plan"))
        _validate_selected_plan(selected_plan, selected_evidence, feature_id=final.feature_spec_ref.feature_id)
        _validate_source_items(selected_evidence, workspace)
        _require(all(row.get("canonical_target_path") in allowed_edit_paths
            for row in selected_plan["binding_registry"]["edit_bindings"]), "PLAN_EDIT_SCOPE_CHANGED")
        documents = TaskContextDocuments(selected_context, selected_feature, selected_evidence, selected_plan)
        prepared = prepare_task_agent_contract(documents, workspace_root=workspace,
            evaluator=None, policy=policy, verification_configured=True)
        source_hash = PlanBoundSourceInspector(workspace_root=workspace, implementation_plan=selected_plan,
            evidence_pack=selected_evidence, policy=policy).source_snapshot_sha256
        return selected_plan, prepared, source_hash

    def actual_tool_record(row, *, name, arguments, response):
        _admitted_response(response, execution["model_profile"])
        _require([item for item in response.get("tool_calls", []) if item.get("call_id") == row["call_id"]] == [
            {"name": name, "call_id": row["call_id"], "arguments": arguments}], "TOOL_RESPONSE_CHANGED")
        actual = loop.get("call_records", {}).get(row["call_id"], {})
        output = actual.get("result", {}).get("output", {})
        _require(actual.get("status") == "completed" and actual.get("name") == name
            and actual.get("arguments_sha256") == canonical_hash(arguments)
            and row.get("event") in loop.get("events", [])
            and row["event"].get("output_sha256") == canonical_hash(output)
            and output.get("execution") == "executed", "CANDIDATE_SUBMISSION_CHANGED")
        return output.get("value", {})

    def rebuild_candidate(selected, reference):
        identity = (reference["path"], reference["sha256"])
        _require(identity not in visiting and len(visiting) <= len(transitions), "CANDIDATE_PARENT_CYCLE")
        visiting.add(identity)
        _require(selected.get("candidate_id") == "candidate-" + canonical_hash({key: value for key, value in selected.items()
            if key != "candidate_id"})[:20], "CANDIDATE_PARENT_CHANGED")
        selected_plan, prepared, source_hash = contract_for(selected)
        origin = selected["model_origin"]
        transition_index = next((index for index, row in enumerate(transitions) if row.get("candidate_ref") == reference), None)
        if expected_candidate_mode == "current":
            _require(transition_index is not None, "CANDIDATE_SUBMISSION_REQUIRED")
            transition = transitions[transition_index]
            _require(transition.get("context") == selected["context"]
                and transition.get("context_id") == selected["context_id"]
                and selected.get("source_snapshot_sha256") == source_hash, "CANDIDATE_SUBMISSION_CHANGED")
        parent_materialized = None
        if origin.get("kind") == "derived_from_context_transition":
            _require(expected_candidate_mode == "current" and origin.get("derivation") == "exact_code_context_transfer_v1"
                and transition_index > 0 and transition.get("role") == "upstream", "CONTEXT_TRANSFER_ORIGIN_CHANGED")
            parent_ref = origin.get("parent_candidate_ref")
            _require(parent_ref == transitions[transition_index - 1].get("candidate_ref"), "CURRENT_PARENT_CHANGED")
            parent = _read(parent_ref, root)
            _generated, parent_materialized = rebuild_candidate(parent, parent_ref)
            assessment = _read(origin["assessment_ref"], root)
            arguments, response = _read(transition["arguments_ref"], root), _read(transition["response_ref"], root)
            observed = actual_tool_record(transition, name="upstream", arguments=arguments, response=response)
            _require(arguments.get("context_id") == parent["context_id"]
                and assessment.get("input_context") == parent["context"]
                and assessment.get("output_context") == selected["context"]
                and selected["context"] != parent["context"]
                and assessment.get("request_sha256") == canonical_hash(arguments.get("request"))
                and assessment.get("outcome") == "replanned" and assessment.get("approval_inherited") is False
                and assessment.get("source_modified") is False and assessment.get("model_history_reset") is False
                and observed.get("assessment_sha256") == origin["assessment_ref"]["sha256"]
                and observed.get("current_code_preserved") is True and observed.get("new_check_required") is True
                and observed.get("context_id") == selected["context_id"], "CONTEXT_TRANSFER_ASSESSMENT_CHANGED")
            for owner_ref in assessment["owner_result_refs"]:
                _read(owner_ref, root)
            content, prepared = context_transfer_content(baseline_files=parent_materialized.baseline_files,
                current_files=parent_materialized.files, implementation_plan=selected_plan, prepared_contract=prepared)
            model = parent["model_response"]["model"]
        else:
            arguments, response = _read(origin["arguments_ref"], root), _read(origin["response_ref"], root)
            model = response["model"]
            _admitted_response(response, execution["model_profile"])
            _require([row for row in response.get("tool_calls", []) if row.get("call_id") == origin["call_id"]] == [
                {"name": "candidate_submit", "call_id": origin["call_id"], "arguments": arguments}], "TOOL_RESPONSE_CHANGED")
            if expected_candidate_mode == "current":
                _require(transition.get("role") == "candidate_submit" and transition.get("call_id") == origin["call_id"]
                    and transition.get("response_ref") == origin["response_ref"]
                    and transition.get("arguments_ref") == origin["arguments_ref"], "CANDIDATE_SUBMISSION_CHANGED")
                observed = actual_tool_record(transition, name="candidate_submit", arguments=arguments, response=response)
                _require(observed.get("candidate_id") == selected["candidate_id"]
                    and observed.get("code_sha256") == selected["materialized_code_identity"]["code_sha256"],
                    "CANDIDATE_SUBMISSION_CHANGED")
        if origin.get("kind") == "derived_from_actual_tool_call_arguments":
            _require(origin.get("field") == "patch_json" and set(arguments) == {"context_id", "patch_json"}
                and arguments["context_id"] == selected["context_id"] and isinstance(arguments["patch_json"], str),
                "TOOL_ARGUMENTS_CHANGED")
            _require(expected_candidate_mode != "current" or transition_index == 0, "CURRENT_REVISION_REQUIRED")
            content = arguments["patch_json"]
        elif origin.get("kind") != "derived_from_context_transition":
            _require(expected_candidate_mode == "current" and origin.get("kind") == "derived_from_current_candidate_revision"
                and origin.get("derivation") == "binding_exact_revision_v1" and transition_index is not None
                and transition_index > 0, "NATIVE_TOOL_ORIGIN_REQUIRED")
            parent_ref = origin.get("parent_candidate_ref")
            _require(parent_ref == transitions[transition_index - 1].get("candidate_ref"), "CURRENT_PARENT_CHANGED")
            parent = _read(parent_ref, root)
            rebuild_candidate(parent, parent_ref)
            _require(parent["context"] == selected["context"] and parent["context_id"] == selected["context_id"],
                "CURRENT_PARENT_CHANGED")
            content = derive_current_revision(parent, arguments=arguments, prepared_contract=prepared)
        derived_path = Path(origin["derived_content_ref"]["path"]).resolve(strict=True)
        _require(derived_path.is_relative_to(root) and _ref(derived_path) == origin["derived_content_ref"]
            and derived_path.read_bytes() == content.encode("utf-8"), "DERIVED_CONTENT_CHANGED")
        _require(selected.get("model_response") == {"path": origin["derived_content_ref"]["path"],
            "output_sha256": origin["derived_content_ref"]["sha256"], "output_utf8_bytes": len(content.encode("utf-8")),
            "model": model, "origin": origin["kind"]}, "CANDIDATE_RESPONSE_CHANGED")
        if origin.get("kind") == "derived_from_current_candidate_revision":
            prepared = candidate_projection_contract(prepared, content)
        generated = LLMExactPatchGenerator(None).materialize_prepared_native(prepared,
            ChatResponse(content=content, model=model))
        _require(list(generated.changes) == selected.get("generated_changes")
            and canonical_hash(generated.changes) == selected.get("generated_changes_sha256"), "CANDIDATE_CONTENT_CHANGED")
        materialized = materialize_candidate(workspace_root=workspace, implementation_plan=selected_plan, changes=generated.changes)
        _require(materialized_code_identity(materialized) == selected.get("materialized_code_identity")
            and all(path in allowed_edit_paths for path in materialized.files), "CANDIDATE_SOURCE_OR_SCOPE_CHANGED")
        if parent_materialized is not None:
            _require(materialized.files == parent_materialized.files
                and materialized.baseline_files == parent_materialized.baseline_files, "CONTEXT_TRANSFER_CODE_CHANGED")
        visiting.remove(identity)
        return generated, materialized

    generated, _materialized = rebuild_candidate(candidate, outcome.candidate_ref)
    origin = candidate["model_origin"]
    report = _read_checker_report(check["report"]["report_ref"], store_root=root,
        workspace_root=workspace, worker_configuration_ref=worker_configuration_ref)
    admitted, _ = validate_candidate_report({**report, "report_ref": check["report"]["report_ref"]},
        changes_sha256=candidate["generated_changes_sha256"], source_snapshot_sha256=candidate["source_snapshot_sha256"],
        allowed_binding_ids=[row["binding_id"] for row in plan["binding_registry"]["edit_bindings"]])
    _require(report.get("status") == "passed" and report.get("execution_origin") == "executed"
        and report.get("materialized_code_identity") == candidate["materialized_code_identity"]
        and report.get("implementation_plan_ref") == asdict(final.implementation_plan_ref)
        and report.get("model_response_sha256") == origin["derived_content_ref"]["sha256"]
        and check.get("verification_identity") == report.get("verification_identity") == expected_verification_identity
        and all(check["report"].get(key) == value for key, value in admitted.items()), "CHECK_BINDING_CHANGED")
    checked_calls = [row for row in execution.get("completed_tool_records", []) if row.get("role") == "candidate_check"
        and row.get("check_ref") == outcome.check_ref and row.get("candidate_ref") == outcome.candidate_ref
        and row.get("context") == final.to_dict()]
    _require(len(checked_calls) == 1, "CHECK_TOOL_EXECUTION_REQUIRED")
    checked_call = checked_calls[0]
    check_arguments = _read(checked_call["arguments_ref"], root)
    check_response = _read(checked_call["response_ref"], root)
    _admitted_response(check_response, execution["model_profile"])
    _require(check_arguments == {"candidate_id": candidate["candidate_id"]}
        and [row for row in check_response.get("tool_calls", []) if row.get("call_id") == checked_call["call_id"]] == [
            {"name": "candidate_check", "call_id": checked_call["call_id"], "arguments": check_arguments}], "CHECK_TOOL_RESPONSE_CHANGED")
    checked_record = loop.get("call_records", {}).get(checked_call["call_id"], {})
    checked_output = checked_record.get("result", {}).get("output", {})
    _require(checked_record.get("status") == "completed" and checked_record.get("name") == "candidate_check"
        and checked_record.get("arguments_sha256") == canonical_hash(check_arguments)
        and checked_output.get("execution") == "executed" and checked_output.get("value", {}).get("status") == "passed"
        and checked_call.get("event") in loop.get("events", [])
        and checked_call["event"].get("output_sha256") == canonical_hash(checked_output), "CHECK_TOOL_RECORD_CHANGED")
    finish = _read(outcome.finish_response_ref, root)
    _admitted_response(finish, execution["model_profile"])
    _require(len(finish.get("tool_calls", [])) == 1, "FINISH_RESPONSE_REQUIRED")
    call = finish["tool_calls"][0]
    _require(call.get("name") == "finish" and call.get("arguments") == {"candidate_id": candidate["candidate_id"]},
        "FINISH_CANDIDATE_CHANGED")
    completion = loop.get("completion")
    expected_completion = {"accepted": True, "candidate_id": candidate["candidate_id"],
        "checked_report_ref": outcome.check_ref, "scope": "qualified_candidate_validation_only", "whole_task_delivered": False}
    record = loop.get("call_records", {}).get(call["call_id"], {})
    _require(completion == expected_completion and record.get("status") == "completed" and record.get("name") == "finish"
        and record.get("arguments_sha256") == canonical_hash(call["arguments"])
        and record.get("result", {}).get("output", {}).get("completion_check") == completion
        and any(row.get("event") == "tool_dispatch_completed" and row.get("role") == "finish"
                and row.get("call_id") == call["call_id"] for row in loop.get("events", [])), "FINISH_RECORD_CHANGED")
    source_before = execution.get("source_before")
    _require(source_before == execution.get("source_after") and isinstance(source_before, list)
        and _source_facts([Path(row["root"]) for row in source_before]) == source_before, "SOURCE_CHANGED")
    return ValidatedTaskAgentHandoff(final, tuple(deepcopy(generated.changes)), candidate, check, dict(outcome.execution_ref))


class TaskAgentExecution:
    """One local-Ollama session; Product keeps approval and formal delivery.

    deadline cooperatively stops between actions, not an in-flight provider or
    compiler. The outer owned-process controller remains the hard deadline owner.
    """

    def __init__(self, *, artifact_store, workspace_root, profile, policy, behavior_evaluator,
            worker_configuration_ref, expected_model_identity, identity_reader, providers, validate_providers,
            reserve_model_call, budget_snapshot, allowed_edit_paths, allowed_read_roots, output_root,
            record_event=None, client_factory=None, deadline=None, candidate_mode="baseline"):
        from src.mod_agent.external_patch import ExternalOpenAIResponsesProfile
        from src.mod_agent.local_entry import LocalOllamaInitialPatchProfileV1

        _require(isinstance(profile, LocalOllamaInitialPatchProfileV1)
            or getattr(self, "protocol", None) == "v2" and isinstance(profile, ExternalOpenAIResponsesProfile), "LOCAL_OLLAMA_REQUIRED")
        _require(isinstance(policy, PatchAgentLoopPolicy) and all(callable(value) for value in (
            identity_reader, validate_providers, reserve_model_call, budget_snapshot)), "OWNERS_REQUIRED")
        _require(deadline is None or type(deadline) in {int, float} and math.isfinite(deadline), "DEADLINE_INVALID")
        _require(candidate_mode in {"baseline", "current"}, "CANDIDATE_MODE_INVALID")
        self.candidate_mode = candidate_mode
        self.store, self.workspace_root = artifact_store, Path(workspace_root).resolve(strict=True)
        self.profile, self.policy, self.evaluator = profile, policy, behavior_evaluator
        self._worker_configuration_ref = deepcopy(worker_configuration_ref)
        self.expected_model_identity, self.identity_reader = deepcopy(expected_model_identity), identity_reader
        self.providers, self.validate_providers = dict(providers), validate_providers
        self.reserve_model_call, self.budget_snapshot = reserve_model_call, budget_snapshot
        self._allowed_edit_paths = tuple(allowed_edit_paths)
        _require(bool(self._allowed_edit_paths) and len(set(self._allowed_edit_paths)) == len(self._allowed_edit_paths), "EDIT_SCOPE_REQUIRED")
        self.allowed_read_roots = tuple(Path(path).resolve(strict=True) for path in allowed_read_roots)
        _require(bool(self.allowed_read_roots) and all(path.is_relative_to(self.workspace_root) and path.is_dir()
            for path in self.allowed_read_roots), "READ_SCOPE_REQUIRED")
        self.output_root = Path(output_root).resolve()
        _require(self.output_root != self.store.root.resolve() and self.output_root.is_relative_to(self.store.root.resolve()), "OUTPUT_OUTSIDE_STORE")
        self.record_event, self.client_factory, self.deadline = record_event, client_factory, deadline
        self._bound_store = self._used = False
        self._receipt = None
        self._verification_identity = None
        self._reads = self._observation_bytes = 0

    @property
    def allowed_edit_paths(self):
        return self._allowed_edit_paths

    @property
    def worker_configuration_ref(self):
        return deepcopy(self._worker_configuration_ref)

    @property
    def expected_verification_identity(self):
        return deepcopy(self._verification_identity)

    def bind_product_store(self, recorder):
        _require(not self._used and not self._bound_store and recorder.root.resolve() == self.store.root.resolve(), "STORE_BINDING_CHANGED")
        self.store, self._bound_store = recorder, True

    def execution_receipt(self):
        return deepcopy(self._receipt)

    def validate_handoff(self, outcome, expected_initial_context):
        return validate_task_agent_handoff(outcome, store=self.store, workspace_root=self.workspace_root,
            expected_initial_context=expected_initial_context, allowed_edit_paths=self.allowed_edit_paths,
            worker_configuration_ref=self.worker_configuration_ref, expected_verification_identity=self.expected_verification_identity,
            expected_candidate_mode=self.candidate_mode)

    def _save(self, name, value):
        with (self.output_root / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        return _ref(self.output_root / name)

    def _charge(self, *, reads, observation_bytes):
        _require(type(reads) is int and reads >= 0 and type(observation_bytes) is int and observation_bytes >= 0,
            "TOOL_CHARGE_INVALID")
        if not is_meter_only(self.budget_snapshot()):
            _require(self._observation_bytes + observation_bytes <= self.policy.max_observation_bytes, "TOOL_BUDGET_EXHAUSTED")
            if self._reads + reads > self.policy.max_tool_rounds:
                raise TaskReadBudgetError(requested=reads, used=self._reads, limit=self.policy.max_tool_rounds)
        self._reads += reads
        self._observation_bytes += observation_bytes

    def read_budget_snapshot(self):
        return read_budget_view(used=self._reads,
            limit=None if is_meter_only(self.budget_snapshot()) else self.policy.max_tool_rounds)

    def __call__(self, state, implementation, evidence_pack):
        from src.mod_agent.local_entry import LocalOllamaInitialPatchAdapter
        from src.mod_agent.model_admission import (
            assert_ollama_model_identity_stable,
            validate_ollama_context_window,
            validate_ollama_model_identity,
        )

        _require(not self._used, "SINGLE_INVOCATION_REQUIRED")
        self._used = True
        initial = _context(state)
        budget_before = validate_budget(self.budget_snapshot())
        self._receipt = TaskAgentPatchOutcome("incomplete", {"code": "TASK_AGENT_EXECUTION_NOT_FINISHED"},
            initial, initial, None, None, None, None, None, None, budget_before, deepcopy(budget_before))
        try:
            self.output_root.parent.mkdir(parents=True, exist_ok=True)
            self.output_root.mkdir(exist_ok=False)
        except Exception as error:
            self._receipt = replace(self._receipt, status="failed", termination={
                "code": "TASK_AGENT_EXECUTION_OUTPUT_CLAIM_FAILED", "error_type": type(error).__name__})
            raise
        details = {"run_id": state["run_id"], "request_id": state["request_id"]}
        completed_tools, provider_turns = [], []
        session, loop, claim_ref, loop_ref, finish_ref = None, None, None, None, None
        try:
            _require(self.store.semantic_read_immutable(initial.implementation_plan_ref)["data"] == implementation
                and self.store.semantic_read_immutable(initial.evidence_pack_ref)["data"] == evidence_pack, "INITIAL_PAYLOAD_CHANGED")
            self.validate_providers()
            details = {"run_id": state["run_id"], "request_id": state["request_id"],
                "worker_binding": _worker_binding(self.worker_configuration_ref, self.evaluator),
                "model_profile": self.profile.to_dict(), "expected_model_identity": deepcopy(self.expected_model_identity),
                "policy": asdict(self.policy), "allowed_edit_paths": list(self.allowed_edit_paths),
                **({"candidate_mode": self.candidate_mode} if self.candidate_mode != "baseline" else {}),
                "source_before": _source_facts(self.allowed_read_roots), "deadline": self.deadline,
                "scope": "common_initial_prefix_then_task_agent_candidate_only; Product owns final delivery"}
            self._verification_identity = verification_identity(self.evaluator)
            _require(isinstance(self._verification_identity, dict), "VERIFICATION_IDENTITY_REQUIRED")
            details["expected_verification_identity"] = self.expected_verification_identity
            claim_ref = self._save("claim.json", {"schema_version": "task-agent-product-claim.v1",
                "run_id": state["run_id"], "request_id": state["request_id"], "initial_context": initial.to_dict(),
                "budget_before": budget_before, **details})
            gate = LocalOllamaInitialPatchAdapter(profile=self.profile, artifact_store=self.store,
                workspace_root=self.workspace_root, patch_agent_loop_policy=self.policy,
                behavior_evaluator=self.evaluator, identity_reader=self.identity_reader, enable_recovery_loop=True)

            def identity():
                observed = self.identity_reader(self.profile.base_url, self.profile.model)
                validate_ollama_model_identity(observed, expected_base_url=self.profile.base_url, expected_model=self.profile.model)
                assert_ollama_model_identity_stable(expected=self.expected_model_identity, observed=observed)
                validate_ollama_context_window(observed, requested_context_window=self.profile.context_window)
                return observed

            def record(event):
                if session is not None and event.get("event") == "tool_dispatch_completed":
                    calls = [row for row in session._actual_calls if row["call_id"] == event["call_id"]]
                    _require(len(calls) == 1, "COMPLETED_TOOL_ORIGIN_MISSING")
                    completed_tools.append({"call_id": event["call_id"], "role": event["role"], "event": deepcopy(event),
                        "response_ref": deepcopy(calls[0]["response_ref"]), "arguments_ref": deepcopy(calls[0]["arguments_ref"]),
                        "context": session.context.to_dict(), "check_ref": deepcopy(session.current_check),
                        **({"context_id": session.context_id} if self.candidate_mode == "current" else {}),
                        "candidate_ref": deepcopy(session.candidates[session.current_candidate]["ref"])
                        if session.current_candidate else None})
                if self.record_event is not None:
                    self.record_event(deepcopy(event))

            owners = make_task_agent_owners(store=self.store, workspace_root=self.workspace_root,
                providers=self.providers, validate_providers=self.validate_providers, initial_context=initial,
                failure_context_provider=lambda: session.upstream_failure_context(),
                current_code_view_provider=lambda: session.current_code_view(),
                validate_context_transfer=lambda **payload: session.validate_context_transfer(**payload),
                reserve_model_call=self.reserve_model_call, budget_snapshot=self.budget_snapshot,
                record=record, output_root=self.output_root / "owners", constraint_policy="source_priority_v1")
            documentation = {}
            deadline = self.deadline

            class TimedSession(TaskAgentSession):
                def stop_check(self):
                    if deadline is not None and time.monotonic() >= deadline:
                        return {"code": "TASK_AGENT_WALL_BUDGET_EXHAUSTED"}
                    return super().stop_check()

            session = TimedSession(store=self.store, workspace_root=self.workspace_root, context=initial,
                candidate_mode=self.candidate_mode,
                policy=self.policy, generator=LLMExactPatchGenerator(None), evaluator=self.evaluator,
                allowed_edit_paths=self.allowed_edit_paths, allowed_read_roots=self.allowed_read_roots,
                output_root=self.output_root / "session", prepare_contract=lambda documents: prepare_task_agent_contract(
                    documents, workspace_root=self.workspace_root, evaluator=self.evaluator, policy=self.policy,
                    documentation_holder=documentation), source_search=owners.source_search, upstream=owners.upstream,
                reserve_model_call=lambda: self.reserve_model_call("initial_patch"), budget_snapshot=self.budget_snapshot,
                charge_tool_budget=self._charge, read_budget_snapshot=self.read_budget_snapshot,
                model_response_admission=lambda response: gate._admit_response_model_identity(
                    ChatResponse(content=response.content or "", model=response.model, raw_response=response.raw_response),
                    self.expected_model_identity), record_event=record)
            client = (self.client_factory or LocalOllamaInitialPatchAdapter._default_agent_turn_client)(self.profile)
            _require(getattr(client, "provider", None) == "ollama_chat_local" and getattr(client, "model", None) == self.profile.model,
                "PROVIDER_CHANGED")

            def complete(request):
                from src.core.trace.execution_trace import current_trace
                from src.libs.llm.transport_observation import observe_transport
                from src.mod_agent.patch_transport_bodies import PatchTransportBodyRecorder

                observed_identity = identity()
                bind_prefix = getattr(client, "bind_context_identity", None)
                if callable(bind_prefix):
                    bind_prefix(model_identity=observed_identity,
                        owner_profile_sha256=canonical_hash(self.profile.to_dict()))
                current = documentation.get("current")
                if current is not None:
                    current.validate()
                turn = len(provider_turns) + 1
                event = {"turn": turn, "status": "requested", "request_ref": self._save(
                    f"request-{turn:03d}.json", asdict(request)), "budget": validate_budget(self.budget_snapshot())}
                provider_turns.append(event)
                try:
                    trace = current_trace()
                    if trace is None:
                        response = client.complete_turn(request)
                    else:
                        with trace.model_call({"provider": client.provider, "model": client.model, "turn_index": turn,
                                "usage_available": False, "response_received": False}) as observation:
                            bodies = PatchTransportBodyRecorder(record=event, observation=observation, body_root=self.output_root)
                            with observe_transport(observation.transport_started, observation.update,
                                    request_callback=observation.update, local_request_body_callback=bodies.callback):
                                response = client.complete_turn(request)
                            observation.record_response(usage=response.usage.to_dict(), usage_available=True,
                                response_sha256=canonical_hash(asdict(response)), response_origin="live_provider_response",
                                stop_reason=response.stop_reason, tool_call_ids=[row.call_id for row in response.tool_calls])
                            bodies.publish_refs()
                    event.update(status="responded", response_ref=self._save(f"response-{turn:03d}.json", asdict(response)),
                        usage=response.usage.to_dict())
                except Exception as error:
                    event.update(status="failed", error_type=type(error).__name__)
                    raise
                if current is not None:
                    current.validate()
                identity()
                return response

            loop = session.run(client=SimpleNamespace(provider=client.provider, model=client.model, complete_turn=complete),
                model_role="initial_patch")
            loop_ref = self._save("loop-result.json", loop.to_dict())
            candidate_ref = session.candidates[session.current_candidate]["ref"] if session.current_candidate else None
            if loop.status == "completed":
                finishes = [row for row in session._actual_calls if row["name"] == "finish"]
                _require(len(finishes) == 1, "FINISH_RESPONSE_REQUIRED")
                finish_ref = deepcopy(finishes[0]["response_ref"])
            outcome = TaskAgentPatchOutcome("candidate_ready" if loop.status == "completed" else
                "unknown" if loop.unknown_effects else "blocked", deepcopy(loop.termination), initial, session.context,
                None, loop_ref, claim_ref, deepcopy(candidate_ref), deepcopy(session.current_check), finish_ref,
                budget_before, validate_budget(self.budget_snapshot()), loop.unknown_effects)
            details.update(source_after=_source_facts(self.allowed_read_roots),
                tool_budget={"reads": self._reads, "observation_bytes": self._observation_bytes},
                completed_tool_records=deepcopy(completed_tools), provider_turns=deepcopy(provider_turns))
            _require(details["source_before"] == details["source_after"], "SOURCE_MUTATED")
            getter = getattr(self.store, "task_agent_store_attempts", None)
            details["product_store_attempts"] = getter(self) if callable(getter) else []
            self._receipt = outcome
            details["completed_tool_records"] = deepcopy(completed_tools)
            details["provider_turns"] = deepcopy(provider_turns)
            reference = self._save("execution.json", {"schema_version": "task-agent-product-execution.v1",
                **{key: value for key, value in outcome.to_dict().items() if key != "execution_ref"}, **details})
            self._receipt = replace(outcome, execution_ref=reference)
            if outcome.status == "candidate_ready":
                self.validate_handoff(self._receipt, initial)
            return self.execution_receipt()
        except Exception as error:
            # Scope: final recording failures cannot erase a previously unknown effect.
            # 为什么要有它：上游动作可能已经写入工件；收尾异常不能把未知副作用降成可随意重试的普通失败。
            unknown = bool((loop is not None and loop.unknown_effects)
                or (self._receipt is not None and self._receipt.unknown_effects)
                or (session is not None and session._check_inflight))
            outcome = TaskAgentPatchOutcome("unknown" if unknown else "failed",
                {"code": "TASK_AGENT_EXECUTION_FAILED", "error_type": type(error).__name__,
                 "reason": str(error), **({"prior_loop_termination": deepcopy(loop.termination)} if loop else {})},
                initial, session.context if session else initial, None, loop_ref, claim_ref,
                session.candidates[session.current_candidate]["ref"] if session and session.current_candidate else None,
                deepcopy(session.current_check) if session else None, finish_ref, budget_before,
                validate_budget(self.budget_snapshot()), unknown)
            self._receipt = outcome
            getter = getattr(self.store, "task_agent_store_attempts", None)
            details["product_store_attempts"] = getter(self) if callable(getter) else []
            path = "execution-error.json" if (self.output_root / "execution.json").exists() else "execution.json"
            reference = self._save(path, {"schema_version": "task-agent-product-execution.v1",
                **{key: value for key, value in outcome.to_dict().items() if key != "execution_ref"}, **details})
            self._receipt = replace(outcome, execution_ref=reference)
            return self.execution_receipt()
