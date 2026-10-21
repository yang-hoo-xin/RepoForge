"""Admit a retained candidate, then recheck it through the existing evaluator owner."""
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path

from src.mod_agent.saved_patch_admission import _hash, _json, _read


def _require(condition, code):
    if not condition:
        from src.mod_agent.fresh_planning_continuation import PatchContinuationError
        raise PatchContinuationError(code, "saved_candidate_reverification", code)


def _report_refs(value):
    """Recheck explicit report references without mistaking compile-input rows for refs."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_ref") and isinstance(child, dict) and "sha256" in child:
                _read(child)
            else:
                _report_refs(child)
    elif isinstance(value, list):
        for child in value:
            _report_refs(child)


def admit_saved_candidate(request, admitted):
    """Read exact original identities; never construct an evaluator or execute a tool."""
    from src.libs.llm import ChatResponse
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch
    from src.mod_agent.fresh_planning_continuation import (
        _patch_request_source_snapshot, _source_tree_snapshot_sha256,
    )
    from src.mod_agent.local_entry import (
        _external_evidence_items, _overlay_generated_patch_changes_on_plan,
        _patch_agent_loop_generation_ref, _patch_generation_obligation_projection,
    )
    from src.mod_agent.model_call_budget import validate_budget
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from src.mod_agent.product_execution import ProductExecutionResultV1
    from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256
    from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator
    from src.mod_agent.proposal.patch_agent_loop import PlanBoundSourceInspector
    from src.mod_agent.reviewed_patch_continuation import _verified_terminal

    selection = request.candidate_reverification_selection
    _require(isinstance(selection, dict) and set(selection) == {
        "source_attempt_ref", "parent_manifest_ref", "parent_result_ref"}, "SAVED_CANDIDATE_SELECTION_INVALID")
    journal = _json(selection["source_attempt_ref"])
    manifest = _json(selection["parent_manifest_ref"])
    parent = _json(selection["parent_result_ref"])
    workspace = Path(request.workspace_root).resolve()
    selected = {key: asdict(getattr(request, key + "_ref"))
                for key in ("feature_spec", "evidence_pack", "implementation_plan")}
    product = ProductExecutionResultV1.from_dict(parent["product_execution_result"]).to_dict()
    identity = journal["parent_product_result_identity"]
    _require(manifest["terminal_record"] == selection["parent_result_ref"]
        and Path(selection["parent_manifest_ref"]["path"]).resolve() == workspace / "run_manifest.json"
        and Path(manifest["isolated_workspace"]).resolve() == workspace
        and Path(journal["caller"]["workspace_root"]).resolve() == workspace
        and manifest["run_id"] == parent["run_id"] == product["run_id"] == journal["parent_run_id"]
            == journal["parent_product_run_id"] == identity["run_id"]
        and manifest["request_id"] == product["request_id"] == journal["request_id"]
        and product["feature_id"] == journal["feature_id"] == request.feature_spec_ref.feature_id
        and identity.get("owner") == "ProductExecutionResultV1"
        and identity.get("hash_basis") == "canonical_owner_to_dict"
        and identity["sha256"] == _hash(product), "SAVED_CANDIDATE_PARENT_IDENTITY_MISMATCH")
    receipt_refs = parent.get("patch_execution_receipts", [])
    if isinstance(receipt_refs, dict):
        receipt_refs = [receipt_refs]
    _require(any({"path": row.get("path"), "sha256": row.get("sha256")} == selection["source_attempt_ref"]
                 for row in receipt_refs), "SAVED_CANDIDATE_PARENT_RECEIPT_MISMATCH")
    terminal = _verified_terminal(journal, journal["feature_id"], journal["run_id"])
    _require(journal.get("receipt_version") == "patch_continuation_attempt_v1"
        and journal.get("continuation_selection") == "product_graph_current_implementation_plan"
        and journal.get("status") == "product_execution_finished" and journal.get("proposal_admitted") is False
        and journal.get("selected_artifacts") == selected
        and terminal.get("terminal_stage") == "generate_patch_proposal"
        and terminal.get("approval_granted") is False
        and all(terminal["execution_counts"].get(key) == 0 for key in ("apply", "build", "runtime", "dry_run"))
        and all(terminal.get(key) is None for key in ("patch_proposal_artifact", "proposal_review_artifact", "dry_run_artifact"))
        and parent.get("actual_write_performed") is False
        and all(row.get("effect_role") not in {"isolated_apply", "build", "runtime"}
                for row in product["effect_events"]), "SAVED_CANDIDATE_STAGE_NOT_RECOVERABLE")
    target = Path(manifest["nuget_isolated_build_profile"]["working_directory"]).resolve()
    snapshot = _source_tree_snapshot_sha256(_patch_request_source_snapshot(request))
    _require(journal["generated_against_snapshot_sha256"] == journal["current_snapshot_sha256"]
        and snapshot == _source_tree_snapshot_sha256(admitted.source_tree_snapshot)
        and admitted.source_validation == journal["source_validation"]
        and source_snapshot(target) == parent["initial_isolated_source_snapshot"]
        and all((workspace / row["source_path"]).resolve().is_relative_to(target)
                for row in admitted.source_validation["files"])
        and source_snapshot_sha256(source_snapshot(Path(manifest["target_source"]))) == manifest["source_snapshot_sha256"],
        "SAVED_CANDIDATE_SOURCE_CHANGED")
    _require(all(not (workspace / row["source_path"]).resolve().is_relative_to(admitted.artifact_store.root.resolve())
                 for row in admitted.source_validation["files"]), "SAVED_CANDIDATE_SOURCE_IN_STORE")
    budget = validate_budget(journal["shared_model_budget"])
    factory = journal["factory_receipt"]
    _require(parent["model_call_budget"] == budget and parent["model_call_count"] == budget["consumed"]
        and type(journal["model_call_count"]) is int and 0 < journal["model_call_count"] <= journal["max_model_calls"]
        and journal["model_call_count"] == factory["model_call_count"]
        and journal["model_call_count"] <= budget["by_stage"][journal["model_role"]]["consumed"],
        "SAVED_CANDIDATE_BUDGET_MISMATCH")
    _read(request.worker_configuration_ref)
    configuration = PatchWorkerConfiguration.load(request.worker_configuration_ref["path"])
    _require(configuration.reference == request.worker_configuration_ref and configuration.enable_recovery_loop
        and configuration.behavior_evaluator() is not None
        and request.worker_profile_sha256 == _hash(factory["profile"]), "SAVED_CANDIDATE_CONFIGURATION_MISMATCH")
    response = factory["model_response"]
    response_ref = {"path": response["path"], "sha256": response["output_sha256"]}
    raw = _read(response_ref)
    _require(factory.get("status") == "failed" and response.get("retained") is True
        and response.get("encoding") == "utf-8" and response.get("output_utf8_bytes") == len(raw)
        and response.get("model") == factory["profile"]["model"] == factory["model_identity"]["model"],
        "SAVED_CANDIDATE_RESPONSE_MISMATCH")
    turns = factory.get("worker_turns") or []
    _require(turns and turns[-1].get("status") == "responded", "SAVED_CANDIDATE_FINAL_TURN_MISSING")
    last_turn = _json(turns[-1]["response_ref"])
    _require(last_turn.get("content") == raw.decode("utf-8") and not last_turn.get("tool_calls")
        and last_turn.get("model") == response["model"], "SAVED_CANDIDATE_FINAL_TURN_MISMATCH")
    plan = admitted.implementation_plan
    obligations = _patch_generation_obligation_projection(feature_spec=admitted.feature_spec,
        implementation=plan, planned_changes=list(plan["proposed_changes"]), evidence_pack=admitted.evidence_pack)
    generator = LLMExactPatchGenerator(None)
    prepared = generator.prepare_native_contract(user_request=admitted.feature_spec["user_request"],
        **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
        evidence_items=_external_evidence_items(admitted.evidence_pack["items"]), binding_registry=plan["binding_registry"],
        constraints=obligations["side_effect_boundaries"], target_behavior=obligations["target_behavior"],
        acceptance_criteria=obligations["acceptance_criteria"], runtime_verification_plan=obligations["runtime_verification_plan"],
        required_change_units=obligations["required_change_units"],
        required_validation_by_binding={r["edit_binding_id"]: r["expected_validation"] for r in obligations["required_change_units"]})
    _require(factory["patch_response_contract"]["schema_sha256"] == prepared.schema_sha256,
             "SAVED_CANDIDATE_SCHEMA_MISMATCH")
    generated = generator.materialize_prepared_native(prepared, ChatResponse(content=raw.decode("utf-8"),
        model=response["model"], usage=response.get("usage")))
    materialized = materialize_exact_text_patch(workspace_root=workspace, diff_intent=generated.changes,
        implementation_plan=plan)
    code = {"baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256,
            "files": sorted(materialized.files)}
    inspector = PlanBoundSourceInspector(workspace_root=workspace, implementation_plan=plan, policy=configuration.policy)
    evaluations = factory["patch_agent_loop"].get("behavior_evaluations") or []
    _require(evaluations, "SAVED_CANDIDATE_FAILURE_REPORT_MISSING")
    failure = evaluations[-1]
    persisted = _json(failure["report_ref"])
    _require(failure.get("status") == "infrastructure_failed" and failure.get("failure_owner") in {"environment", "validator"}
        and failure.get("admission") == "accepted" and failure.get("callback_invoked") is True
        and failure.get("execution_origin") == "executed"
        and failure.get("generated_changes_sha256") == _hash(generated.changes)
        and failure.get("model_response_sha256") == response_ref["sha256"]
        and failure.get("source_snapshot_sha256") == inspector.source_snapshot_sha256
        and failure.get("materialized_code_identity") == code
        and persisted.get("materialized_code_identity") == code
        and all(failure.get(key) == persisted.get(key) for key in (
            "status", "failure_owner", "generated_changes_sha256", "source_snapshot_sha256",
            "owner_binding_ids", "observations", "execution_origin")), "SAVED_CANDIDATE_FAILURE_IDENTITY_MISMATCH")
    _report_refs(persisted)
    changes = _overlay_generated_patch_changes_on_plan(planned_changes=list(plan["proposed_changes"]),
        generated_changes=generated.changes)
    generation = {"provider": "ollama", "model_digest": factory["model_identity"]["digest"],
        "model": generated.model, "usage": deepcopy(response.get("usage")),
        "profile_version": factory["profile"]["profile_version"],
        "patch_response_contract": deepcopy(factory["patch_response_contract"]),
        "model_input_sha256": factory["model_input_sha256"], "model_response_sha256": response_ref["sha256"],
        "patch_agent_loop": _patch_agent_loop_generation_ref(factory["patch_agent_loop"]),
        "response_origin": "verified_prior_response", "new_model_calls": 0,
        "source_attempt_ref": deepcopy(selection["source_attempt_ref"])}
    _require(_source_tree_snapshot_sha256(_patch_request_source_snapshot(request)) == snapshot,
             "SAVED_CANDIDATE_SOURCE_CHANGED")
    return {"changes": tuple({**change, "generation": deepcopy(generation)} for change in changes),
        "source_attempt_ref": deepcopy(selection["source_attempt_ref"]), "response_ref": response_ref,
        "request_id": journal["request_id"], "parent_run_id": journal["parent_run_id"],
        "inherited_model_budget": budget, "source_snapshot_sha256": snapshot,
        "parent_manifest_ref": deepcopy(selection["parent_manifest_ref"]),
        "parent_result_ref": deepcopy(selection["parent_result_ref"]), "original_request": manifest.get("request"),
        "historical_failure_ref": deepcopy(failure["report_ref"]), "materialized_code_identity": code,
        "generated": generated, "configuration": configuration, "factory_receipt": deepcopy(factory),
        "worker_configuration_ref": deepcopy(request.worker_configuration_ref)}


def execute_saved_candidate(request, admitted, recorder):
    """Execute the current evaluator once; Graph owns attempt claiming and terminal caching."""
    from src.mod_agent.local_entry import LocalOllamaInitialPatchAdapter, LocalOllamaInitialPatchProfileV1
    from src.mod_agent.proposal.patch_agent_loop import PlanBoundSourceInspector

    saved = admit_saved_candidate(request, admitted)
    configuration = saved["configuration"]
    original = saved["factory_receipt"]
    profile = LocalOllamaInitialPatchProfileV1(**{field.name: original["profile"][field.name]
        for field in fields(LocalOllamaInitialPatchProfileV1)})

    def no_model(*args, **kwargs):
        raise RuntimeError("SAVED_CANDIDATE_MODEL_EXECUTION_FORBIDDEN")

    worker = LocalOllamaInitialPatchAdapter(profile=profile, artifact_store=admitted.artifact_store,
        workspace_root=request.workspace_root, patch_agent_loop_policy=configuration.policy,
        identity_reader=no_model, llm_factory=no_model, agent_turn_client_factory=no_model,
        behavior_evaluator=configuration.behavior_evaluator(), enable_recovery_loop=True)
    worker.bind_attempt_recorder(recorder)
    worker._implementation_plan = deepcopy(admitted.implementation_plan)
    worker._implementation_plan_ref = asdict(request.implementation_plan_ref)
    worker._current_run_id = "candidate_reverification_" + _hash(request.attempt_id)[:24]
    worker._receipt.update(adapter="RetainedCandidateReverification", status="reverification_started",
        model_response=deepcopy(original["model_response"]), model_call_count=0,
        response_origin="verified_prior_response", source_attempt_ref=deepcopy(saved["source_attempt_ref"]),
        model_call_budget={"minimum_total_model_calls": 0, "maximum_total_model_calls": 0},
        inherited_model_budget=deepcopy(saved["inherited_model_budget"]))
    inspector = PlanBoundSourceInspector(workspace_root=Path(request.workspace_root),
        implementation_plan=admitted.implementation_plan, policy=configuration.policy)
    try:
        # 为什么要有它：准入可重复读取，只有执行节点能够构造验证器并产生新的验收事实。
        evaluator = worker.behavior_evaluator
        evaluator.prepare_for_patch_attempt()
        worker._receipt["behavior_evaluator_configuration"] = evaluator.configuration_receipt()
        worker._emit("saved_candidate_reverification_started")
        report = worker._evaluate_generated_behavior_once(saved["generated"], inspector)
        _require(report.get("status") == "passed" and report.get("admission") == "accepted"
            and report.get("execution_origin") == "executed" and report.get("callback_invoked") is True,
            "SAVED_CANDIDATE_REVERIFICATION_NOT_PASSED")
        persisted = _json(report["report_ref"])
        _require(persisted.get("materialized_code_identity") == saved["materialized_code_identity"]
            and (report.get("verification_identity") is None
                 or persisted.get("verification_identity") == report["verification_identity"]),
            "SAVED_CANDIDATE_REVERIFICATION_CODE_MISMATCH")
        _report_refs(persisted)
        # 为什么要有它：验证器可能读写外部环境，进入Proposal前必须重新核验输入和完整源码前像。
        checked = admit_saved_candidate(request, admitted)
        _require(checked["materialized_code_identity"] == saved["materialized_code_identity"],
                 "SAVED_CANDIDATE_SOURCE_CHANGED")
        worker._receipt["status"] = "reverification_passed"
        worker._emit("saved_candidate_reverification_passed")
    except Exception as exc:
        worker._receipt.update(status="reverification_failed", failure={
            "code": getattr(exc, "code", type(exc).__name__), "message": str(exc)})
        worker._emit("saved_candidate_reverification_failed")
        raise
    changes = deepcopy(saved["changes"])
    for change in changes:
        # 为什么要有它：模型原始来源保持不变，新的实际验收必须单独绑定到这一次执行。
        change["generation"]["patch_agent_loop"]["behavior_evaluation_refs"] = [{
            key: deepcopy(report.get(key)) for key in ("status", "failure_owner", "generated_changes_sha256",
                "model_response_sha256", "report_ref")}]
    result = {key: value for key, value in saved.items() if key not in {"generated", "configuration", "factory_receipt"}}
    result.update(changes=changes, verification=deepcopy(report), worker_receipt=worker.patch_attempt_receipt(),
        model_calls_performed=0, verifier_executions_performed=1, approval_inherited=False)
    return result
