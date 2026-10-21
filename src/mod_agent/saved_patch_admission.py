"""Read-only admission of a retained, verified response for context rebuilding.

Historical verification stays historical. This module neither calls a provider
or verifier nor writes an Artifact, source file, approval or budget allowance.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


def _require(condition, code):
    if not condition:
        from src.mod_agent.fresh_planning_continuation import PatchContinuationError
        raise PatchContinuationError(code, "saved_patch_admission", code)


def _read(ref):
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}
             and isinstance(ref["path"], str) and Path(ref["path"]).is_absolute()
             and isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64,
             "SAVED_PATCH_FILE_REFERENCE_INVALID")
    raw = Path(ref["path"]).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "SAVED_PATCH_FILE_HASH_MISMATCH")
    return raw


def _json(ref):
    value = json.loads(_read(ref))
    _require(isinstance(value, dict), "SAVED_PATCH_OBJECT_REQUIRED")
    return value


def _hash(value):
    from src.mod_agent.fresh_planning_continuation import _canonical_sha256
    return _canonical_sha256(value)


def _check_budget(value):
    _require(isinstance(value, dict), "SAVED_PATCH_BUDGET_INVALID")
    for key in ("maximum", "consumed", "remaining"):
        _require(type(value.get(key)) is int and value[key] >= 0, "SAVED_PATCH_BUDGET_INVALID")
    _require(value["consumed"] + value["remaining"] == value["maximum"], "SAVED_PATCH_BUDGET_INVALID")
    stages = value.get("by_stage")
    _require(isinstance(stages, dict) and stages, "SAVED_PATCH_BUDGET_INVALID")
    for stage in stages.values():
        _require(isinstance(stage, dict) and all(type(stage.get(k)) is int and stage[k] >= 0
                 for k in ("maximum", "consumed", "remaining"))
                 and stage["consumed"] + stage["remaining"] == stage["maximum"],
                 "SAVED_PATCH_BUDGET_INVALID")
    _require(sum(s["consumed"] for s in stages.values()) == value["consumed"], "SAVED_PATCH_BUDGET_INVALID")


def _parent_and_budget(request, journal, workspace):
    """Read the existing parent owner's manifest, exact result and prior receipt."""
    from src.mod_agent.product_execution import ProductExecutionResultV1
    from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256

    manifest_path = workspace / "run_manifest.json"  # Original run-request owner layout.
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    parent = _json(manifest["terminal_record"])
    product = ProductExecutionResultV1.from_dict(parent["product_execution_result"]).to_dict()
    identity = journal["parent_product_result_identity"]
    _require(manifest["run_id"] == parent["run_id"] == product["run_id"] == journal["parent_run_id"]
             == journal["parent_product_run_id"] == identity["run_id"]
             and manifest["request_id"] == product["request_id"] == journal["request_id"]
             and product["feature_id"] == journal["feature_id"]
             and Path(manifest["isolated_workspace"]).resolve() == workspace,
             "SAVED_PATCH_PARENT_IDENTITY_MISMATCH")
    _require(identity.get("owner") == "ProductExecutionResultV1"
             and identity.get("hash_basis") == "canonical_owner_to_dict"
             and identity.get("sha256") == _hash(product), "SAVED_PATCH_PARENT_RESULT_MISMATCH")
    _require(parent.get("actual_write_performed") is False
             and all(row.get("effect_role") not in {"isolated_apply", "build", "runtime"}
                     for row in product["effect_events"]), "SAVED_PATCH_PARENT_EFFECT_ALREADY_STARTED")
    _require(source_snapshot_sha256(source_snapshot(Path(manifest["target_source"])))
             == manifest["source_snapshot_sha256"], "SAVED_PATCH_CANONICAL_SOURCE_CHANGED")
    budget = journal["shared_model_budget"]
    _check_budget(budget)
    _require(parent["model_call_budget"] == budget and parent["model_call_count"] == budget["consumed"],
             "SAVED_PATCH_PARENT_BUDGET_MISMATCH")
    receipt_ref = request.saved_patch_selection["logical_budget_ref"]
    receipt = _json(receipt_ref)
    prior_ref = receipt["previous_summary_ref"]
    prior = _json(prior_ref)
    prior_result_ref = prior["public_result_ref"]
    prior_result = _json(prior_result_ref)
    prior_budget = prior_result["model_call_budget"]
    _check_budget(prior_budget)
    _require(receipt["current_run_id"] == parent["run_id"]
             and receipt["previous_run_id"] == prior["run_id"] == prior_result["run_id"]
             and prior_result["product_execution_result"]["request_id"] == receipt["logical_request_id"]
             == journal["request_id"] and prior["stage_budget"] == prior_budget
             and receipt["previously_consumed"] == prior_budget["consumed"]
             and receipt["original_maximum"] == prior_budget["maximum"]
             and receipt["this_invocation_maximum"] == prior_budget["remaining"] == budget["maximum"]
             and receipt["previous_stage_budget"] == prior_budget["by_stage"]
             and receipt["count_basis"] == prior_budget["count_basis"] == budget["count_basis"],
             "SAVED_PATCH_LOGICAL_BUDGET_MISMATCH")
    limits = receipt["this_invocation_stage_limits"]
    _require(set(limits) == set(budget["by_stage"]) == set(prior_budget["by_stage"])
             and all(type(limits[k]) is int and limits[k] == budget["by_stage"][k]["maximum"]
                     == prior_budget["by_stage"][k]["remaining"] for k in limits)
             and journal["max_model_calls"] == receipt["worker_segment_maximum"]
             and journal["model_call_count"] == journal["factory_receipt"]["model_call_count"]
             and 0 < journal["model_call_count"] <= journal["max_model_calls"]
             and journal["model_call_count"] <= budget["by_stage"][journal["model_role"]]["consumed"],
             "SAVED_PATCH_STAGE_BUDGET_MISMATCH")
    total = prior_budget["consumed"] + budget["consumed"]
    cumulative_stages = {key: {"maximum": prior_budget["by_stage"][key]["maximum"],
        "consumed": prior_budget["by_stage"][key]["consumed"] + value["consumed"],
        "remaining": value["remaining"]} for key, value in budget["by_stage"].items()}
    return parent, {"original_maximum": prior_budget["maximum"], "consumed": total,
        "remaining": prior_budget["maximum"] - total, "new_model_allowance": 0,
        "by_stage": cumulative_stages, "count_basis": budget["count_basis"],
        "logical_budget_ref": deepcopy(receipt_ref), "previous_summary_ref": deepcopy(prior_ref),
        "previous_result_ref": deepcopy(prior_result_ref), "parent_result_ref": deepcopy(manifest["terminal_record"]),
        "parent_manifest_ref": {"path": str(manifest_path), "sha256": hashlib.sha256(manifest_raw).hexdigest()}}


def _bound_refs(value):
    """Recheck only explicitly stored exact-file references, without executing them."""
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            _read(value)
        else:
            for child in value.values():
                _bound_refs(child)
    elif isinstance(value, list):
        for child in value:
            _bound_refs(child)


def _historical_verification(factory, native, inspector, materialized, plan_ref, response_ref):
    evaluations = factory["patch_agent_loop"]["behavior_evaluations"]
    _require(isinstance(evaluations, list) and evaluations, "SAVED_PATCH_VERIFICATION_REQUIRED")
    last = evaluations[-1]
    report_ref = last["report_ref"]
    report = _json(report_ref)
    identity = {"baseline_sha256": materialized.baseline_sha256,
                "code_sha256": materialized.code_sha256, "files": sorted(materialized.files)}
    expected = {"status": "passed", "execution_origin": "executed",
                "generated_changes_sha256": _hash(native), "source_snapshot_sha256": inspector.source_snapshot_sha256,
                "model_response_sha256": response_ref["sha256"], "materialized_code_identity": identity,
                "implementation_plan_ref": plan_ref}
    _require(last.get("admission") == "accepted" and last.get("callback_invoked") is True
             and all(report.get(k) == v and last.get(k) == v for k, v in expected.items())
             and last.get("verification_identity") == report.get("verification_identity")
             and report.get("failure_owner") is None, "SAVED_PATCH_VERIFICATION_BINDING_MISMATCH")
    _bound_refs(report)
    config = factory["behavior_evaluator_configuration"]
    worker = _json(config["configuration_ref"])
    evaluator = worker["behavior_evaluator"]
    _require(evaluator["factory_ref"] == config["factory_ref"]
             and evaluator["factory_name"] == config["factory_name"]
             and _hash(evaluator["kwargs"]) == config["kwargs_sha256"],
             "SAVED_PATCH_VERIFIER_CONFIGURATION_MISMATCH")
    _bound_refs(evaluator)
    kwargs = evaluator["kwargs"]
    probe = _json(kwargs["configuration_kwargs"]["probe_configuration_ref"])
    _read(probe["dotnet"])
    for reference in probe["files"]:
        _read(reference)
    _require(any(row["path"] == probe["assembly_path"] for row in probe["files"]),
             "SAVED_PATCH_PROBE_ASSEMBLY_UNBOUND")
    # The full-project/configuration receipts expose code -> assembly -> executed
    # checks directly. Recheck that historical chain; do not invoke its factory.
    phases = report["validation_results"]
    compilation_ref = phases["full_compilation"]["report_ref"]
    compilation = _json(compilation_ref)
    configuration_ref = phases["configuration"]["report_ref"]
    configuration = _json(configuration_ref)
    _require(compilation.get("status") == configuration.get("status") == "passed"
             and compilation.get("execution_origin") == configuration.get("execution_origin") == "executed"
             and compilation.get("model_response_ref") == response_ref
             and compilation.get("implementation_plan_ref") == plan_ref
             and compilation.get("materialized_code_identity") == configuration.get("materialized_code_identity") == identity
             and configuration.get("compilation_report_ref") == compilation_ref
             and configuration.get("built_artifact_ref") == compilation.get("built_artifact_ref")
             and compilation.get("all_required_files_compiled") is True
             and compilation.get("product_apply_performed") is False
             and compilation.get("formal_build_stage_performed") is False,
             "SAVED_PATCH_VERIFICATION_CHAIN_MISMATCH")
    checks = configuration["checks"]
    _require(isinstance(checks, list) and checks and all(row.get("passed") is True for row in checks)
             and len({row["check_id"] for row in checks}) == len(checks)
             and report["verification_ref"] == configuration_ref
             and report["verification_summary"] == {"checks": len(checks), "failed_checks": 0,
                                                    "executions": len(configuration["observations"])},
             "SAVED_PATCH_VERIFICATION_CHECKS_MISMATCH")
    for field in ("built_artifact_ref", "compile_inputs_ref", "measurement_targets_ref",
                  "nuget_config_input_ref", "source_inputs_ref", "stdout_ref", "stderr_ref"):
        _read(compilation[field])
    for reference in configuration["process_refs"]:
        _read(reference)
    compiled_root = Path(compilation["project_root"]).resolve()
    for row in compilation["compile_inputs"]:
        target = (compiled_root / row["path"]).resolve()
        _require(target.is_relative_to(compiled_root), "SAVED_PATCH_COMPILE_SOURCE_OUTSIDE_ROOT")
        _read({"path": str(target), "sha256": row["sha256"]})
    from src.mod_agent.product_executor import source_snapshot
    source_root = Path(kwargs["compilation_kwargs"]["source_root"]).resolve()
    workspace = Path(kwargs["compilation_kwargs"]["workspace_root"]).resolve()
    expected_sources = {row["path"]: row["sha256"] for row in source_snapshot(source_root)}
    for relative, raw in materialized.files.items():
        target = (workspace / relative).resolve()
        _require(target.is_relative_to(source_root), "SAVED_PATCH_COMPILE_SOURCE_OUTSIDE_ROOT")
        expected_sources[target.relative_to(source_root).as_posix()] = hashlib.sha256(raw).hexdigest()
    _require(_json(compilation["source_inputs_ref"]) == expected_sources,
             "SAVED_PATCH_COMPILE_BASELINE_CHANGED")
    # Environment identity is the explicitly declared dependency tree, not the host.
    from src.mod_agent.build.dependencies import (
        _binary_tree_snapshot,
        _binary_tree_snapshot_sha256,
    )
    binding = compilation["dependency_binding"]
    snapshot = _binary_tree_snapshot(Path(binding["source_root"]))
    _require(len(snapshot) == binding["file_count"]
             and _binary_tree_snapshot_sha256(snapshot) == binding["snapshot_sha256"],
             "SAVED_PATCH_VERIFICATION_ENVIRONMENT_CHANGED")
    return {"status": "historical_binding_revalidated", "executed": False,
            "new_verifier_executions": 0, "new_compilations": 0, "new_model_calls": 0,
            "report_ref": deepcopy(report_ref), "historical_status": report["status"],
            "historical_execution_origin": report["execution_origin"],
            "verification_identity": deepcopy(report["verification_identity"]),
            "verification_summary": deepcopy(report["verification_summary"]),
            "materialized_code_identity": identity, "approval_inherited": False,
            "scope": "retained_code_and_historical_verification_binding_not_new_execution"}


def admit_saved_patch(request, admitted) -> dict:
    """Recover only saved code rejected by the original context-pack review."""
    selection = getattr(request, "saved_patch_selection", None)
    _require(isinstance(selection, dict) and set(selection) == {"source_attempt_ref", "logical_budget_ref"},
             "SAVED_PATCH_SELECTION_INVALID")
    journal = _json(selection["source_attempt_ref"])
    from src.libs.llm import ChatResponse
    from src.mod_agent.artifacts import ArtifactRef
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch
    from src.mod_agent.fresh_planning_continuation import (
        _artifact_ref_dict,
        _csharp_source_snapshot,
        _read_selected_exact_revision,
        _source_tree_snapshot_sha256,
        _validate_exact_upstream_set,
    )
    from src.mod_agent.local_entry import (
        _external_evidence_items,
        _overlay_generated_patch_changes_on_plan,
        _patch_generation_obligation_projection,
    )
    from src.mod_agent.proposal.builder import PatchProposalBuilder
    from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator
    from src.mod_agent.proposal.patch_agent_loop import (
        PatchAgentLoopPolicy,
        PlanBoundSourceInspector,
    )
    from src.mod_agent.reviewed_patch_continuation import _terminal_ref, _verified_terminal

    workspace, store = Path(request.workspace_root).resolve(), admitted.artifact_store
    selected = {k: _artifact_ref_dict(getattr(request, k + "_ref"))
                for k in ("feature_spec", "evidence_pack", "implementation_plan")}
    _require(journal.get("receipt_version") == "patch_continuation_attempt_v1"
             and journal.get("continuation_selection") == "product_graph_current_implementation_plan"
             and journal.get("selected_artifacts") == selected
             and journal.get("feature_id") == request.feature_spec_ref.feature_id
             and journal.get("proposal_admitted") is True
             and journal.get("status") == "product_execution_finished"
             and Path(journal["caller"]["workspace_root"]).resolve() == workspace,
             "SAVED_PATCH_ATTEMPT_IDENTITY_MISMATCH")
    terminal = _verified_terminal(journal, journal["feature_id"], journal["run_id"])
    _require(terminal.get("terminal_stage") == "review_proposal"
             and terminal.get("approval_granted") is False and terminal.get("dry_run_artifact") is None
             and all(terminal["execution_counts"].get(k) == 0 for k in ("apply", "build", "runtime")),
             "SAVED_PATCH_STAGE_NOT_RECOVERABLE")
    source_hash = _source_tree_snapshot_sha256(_csharp_source_snapshot(workspace))
    _require(source_hash == journal["generated_against_snapshot_sha256"] == journal["current_snapshot_sha256"]
             == _source_tree_snapshot_sha256(admitted.source_tree_snapshot)
             and admitted.source_validation == journal["source_validation"], "SAVED_PATCH_SOURCE_CHANGED")
    _require(all(not (workspace / row["source_path"]).resolve().is_relative_to(store.root.resolve())
                 for row in admitted.source_validation["files"]), "SAVED_PATCH_SOURCE_IN_STORE")
    proposal_ref = ArtifactRef(**_terminal_ref(journal, terminal, journal["feature_id"], "patch_proposal", "patch_proposal_artifact"))
    review_ref = ArtifactRef(**_terminal_ref(journal, terminal, journal["feature_id"], "proposal_review", "proposal_review_artifact"))
    proposal = _read_selected_exact_revision(store, proposal_ref, consumer_stage=None)
    review = _read_selected_exact_revision(store, review_ref, consumer_stage=None)
    _validate_exact_upstream_set(proposal, (request.implementation_plan_ref, request.evidence_pack_ref))
    _validate_exact_upstream_set(review, (proposal_ref,))
    findings = review["data"]["findings"]
    blocking = [row for row in findings if row.get("level") == "blocking"]
    _require(proposal["status"] == "success" and review["data"]["decision"] == "revise_proposal"
             and review["data"]["allowed_next_stage"] is None and blocking
             and all(row.get("rule_id") == "context_pack" for row in blocking),
             "SAVED_PATCH_REVIEW_NOT_CONTEXT_ONLY")
    from src.mod_agent.domain import GTFOAdapter
    from src.mod_agent.proposal.reviewer import ProposalReviewer
    # The original run-request product owner fixes this domain explicitly. This
    # recovery does not infer a domain from the model, Mod name or source path.
    domain_policy = GTFOAdapter().identify_project(str(workspace)).review_policy()
    rereview = ProposalReviewer().review(proposal["data"], evidence_pack=admitted.evidence_pack,
        implementation_plan=admitted.implementation_plan, feature_spec=admitted.feature_spec,
        artifact_hashes={key: ref["artifact_hash"] for key, ref in selected.items()},
        domain_policy=domain_policy, permission_level="dry_run_write", require_executable_patch=True).to_artifact_payload()
    observed_blocking = [row for row in rereview["findings"] if row.get("level") == "blocking"]
    _require(rereview["decision"] == "revise_proposal" and observed_blocking
             and all(row.get("rule_id") == "context_pack" for row in observed_blocking),
             "SAVED_PATCH_CURRENT_REVIEW_NOT_CONTEXT_ONLY")
    factory = journal["factory_receipt"]
    response = factory["model_response"]
    response_ref = {"path": response["path"], "sha256": response["output_sha256"]}
    raw = _read(response_ref)
    _require(factory.get("status") == "completed" and response.get("retained") is True
             and response.get("encoding") == "utf-8" and response.get("output_utf8_bytes") == len(raw)
             and response.get("model") == factory["profile"]["model"], "SAVED_PATCH_RESPONSE_IDENTITY_MISMATCH")
    plan = admitted.implementation_plan
    obligations = _patch_generation_obligation_projection(feature_spec=admitted.feature_spec,
        implementation=plan, planned_changes=list(plan["proposed_changes"]), evidence_pack=admitted.evidence_pack)
    generator = LLMExactPatchGenerator(None)  # Existing no-provider parser API.
    prepared = generator.prepare_native_contract(user_request=admitted.feature_spec["user_request"],
        **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
        evidence_items=_external_evidence_items(admitted.evidence_pack["items"]), binding_registry=plan["binding_registry"],
        constraints=obligations["side_effect_boundaries"], target_behavior=obligations["target_behavior"],
        acceptance_criteria=obligations["acceptance_criteria"], runtime_verification_plan=obligations["runtime_verification_plan"],
        required_change_units=obligations["required_change_units"],
        required_validation_by_binding={r["edit_binding_id"]: r["expected_validation"] for r in obligations["required_change_units"]})
    native = generator.materialize_prepared_native(prepared, ChatResponse(content=raw.decode("utf-8"), model=response["model"]))
    changes = _overlay_generated_patch_changes_on_plan(planned_changes=list(plan["proposed_changes"]), generated_changes=native.changes)
    original = proposal["data"]["diff_intent"]
    _require(len(changes) == len(original), "SAVED_PATCH_PROPOSAL_CODE_MISMATCH")
    complete = []
    for change, old in zip(changes, original):
        generation = old["generation"]
        _require(generation.get("model_response_sha256") == response_ref["sha256"]
                 and generation.get("model_input_sha256") == factory["model_input_sha256"]
                 and generation.get("model") == response["model"]
                 and generation.get("patch_response_contract") == factory["patch_response_contract"]
                 and generation["patch_response_contract"]["schema_sha256"] == prepared.schema_sha256,
                 "SAVED_PATCH_GENERATION_METADATA_MISMATCH")
        recovered = {**change, "generation": deepcopy(generation)}
        _require(PatchProposalBuilder._change_intent(recovered) == old, "SAVED_PATCH_PROPOSAL_CODE_MISMATCH")
        complete.append(recovered)
    _require(_hash(complete) == factory["generated_changes_sha256"], "SAVED_PATCH_GENERATED_CHANGES_MISMATCH")
    materialized = materialize_exact_text_patch(workspace_root=workspace, diff_intent=complete, implementation_plan=plan)
    _require(all(not (workspace / p).resolve().is_relative_to(store.root.resolve()) for p in materialized.files),
             "SAVED_PATCH_SOURCE_IN_STORE")
    inspector = PlanBoundSourceInspector(workspace_root=workspace, implementation_plan=plan, policy=PatchAgentLoopPolicy())
    history = _historical_verification(factory, native.changes, inspector, materialized, selected["implementation_plan"], response_ref)
    _parent, budget = _parent_and_budget(request, journal, workspace)
    _require(_source_tree_snapshot_sha256(_csharp_source_snapshot(workspace)) == source_hash,
             "SAVED_PATCH_SOURCE_CHANGED")
    return {"changes": tuple(complete), "source_attempt_ref": deepcopy(selection["source_attempt_ref"]),
            "source_proposal_ref": proposal_ref, "source_review_ref": review_ref, "response_ref": response_ref,
            "historical_verification": history, "request_id": journal["request_id"],
            "parent_run_id": journal["parent_run_id"], "inherited_model_budget": deepcopy(journal["shared_model_budget"]),
            "logical_budget": budget, "source_snapshot_sha256": source_hash}
