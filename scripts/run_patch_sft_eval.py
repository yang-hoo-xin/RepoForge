"""Explicit one-cell SFT/Base evaluation through the existing controlled worker.

Preparation never calls a model. fixed_failure consumes a labelled teacher C1,
not invented provider history. initial_patch starts at its given operator Plan.
Neither mode proves natural-language planning or Runtime delivery.
External execution requires --source-disclosure and --source-disclosure-sha256;
preparation records the disclosure requirements without creating consent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from copy import deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path

from scripts.prepare_patch_sft_data import execute_version, ref, save, toolchain
from src.libs.llm import ChatResponse
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.execution.patch_apply import ExactTextPatchApplier, materialize_exact_text_patch
from src.mod_agent.external_patch import (
    EXTERNAL_PATCH_PROFILE_VERSION,
    PATCH_FEEDBACK_DATA_CLASSES,
    ExternalQwenPatchProfile,
    run_external_patch_continuation,
)
from src.mod_agent.fresh_planning_continuation import (
    PatchContinuationInputV1,
    _csharp_source_snapshot,
    _source_tree_snapshot_sha256,
    preflight_patch_continuation,
    run_fresh_patch_continuation,
)
from src.mod_agent.local_entry import (
    LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
    LocalOllamaInitialPatchAdapter,
    LocalOllamaInitialPatchProfileV1,
    _external_evidence_items,
    _overlay_generated_patch_changes_on_plan,
    _patch_generation_obligation_projection,
    make_exact_no_write_dry_run_factory,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
from src.mod_agent.product_executor import SOURCE_DISCLOSURE_DATA_CLASSES
from src.mod_agent.proposal import LLMExactPatchGenerator, PatchProposalBuilder, ProposalReviewer
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy, PlanBoundSourceInspector


def _request(document):
    value = deepcopy(document)
    for key in ("feature_spec_ref", "evidence_pack_ref", "implementation_plan_ref",
                "repair_parent_proposal_ref", "repair_parent_review_ref"):
        if value.get(key) is not None:
            value[key] = ArtifactRef(**value[key])
    return PatchContinuationInputV1(**value)


def _case(manifest_ref, case_id):
    manifest = read_bound_json(manifest_ref)
    matches = [row for row in manifest["tasks"] if row["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError("SFT_EVAL_CASE_NOT_UNIQUE")
    return matches[0]


def _contract(request):
    admitted = preflight_patch_continuation(request)
    feature, plan = admitted.feature_spec, admitted.implementation_plan
    obligations = _patch_generation_obligation_projection(feature_spec=feature, implementation=plan,
                                                          planned_changes=plan["proposed_changes"], evidence_pack=admitted.evidence_pack)
    generator = LLMExactPatchGenerator(None)
    prepared = generator.prepare_native_contract(user_request=feature["user_request"],
        **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
        evidence_items=_external_evidence_items(admitted.evidence_pack["items"]), binding_registry=plan["binding_registry"],
        constraints=obligations["side_effect_boundaries"], target_behavior=obligations["target_behavior"],
        acceptance_criteria=obligations["acceptance_criteria"], runtime_verification_plan=obligations["runtime_verification_plan"],
        required_change_units=obligations["required_change_units"],
        required_validation_by_binding={row["edit_binding_id"]: row["expected_validation"] for row in obligations["required_change_units"]})
    return admitted, generator, prepared


def _observation_input(check, *, case_id):
    """Preserve explicitly recorded inputs; labels cannot reconstruct execution."""
    if check.get("input") is not None:
        return deepcopy(check["input"])
    return {"case_id": case_id, "group": check["group"],
            "execution_context_status": "unknown"}


def _check_definition_signature(checks):
    if (not isinstance(checks, list) or not checks
            or any(not isinstance(row, dict) or not isinstance(row.get("check"), str)
                   or not row["check"] or not isinstance(row.get("group"), str)
                   or not row["group"] for row in checks)):
        raise ValueError("SFT_EVAL_CHECK_SET_INVALID")
    signature = [(row["check"], row["group"]) for row in checks]
    if len({check for check, _ in signature}) != len(signature):
        raise ValueError("SFT_EVAL_CHECK_SET_DUPLICATE")
    return signature


def _teacher_parent(request, task, output):
    admitted, generator, contract = _contract(request)
    raw_request = read_bound_json(task["raw_request_ref"])
    values = [row["content"] for row in raw_request["messages"] if row["role"] == "assistant"]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("SFT_EVAL_FIXED_C1_NOT_UNIQUE")
    raw = values[0]
    generated = generator.materialize_prepared_native(contract, ChatResponse(content=raw, model="operator_authored_fixed_failure"))
    materialized = materialize_exact_text_patch(workspace_root=request.workspace_root, diff_intent=generated.changes,
                                               implementation_plan=admitted.implementation_plan)
    measured = read_bound_json(task["version_reports"]["C1"])
    compilation = read_bound_json(measured["compilation_ref"])
    if (measured["status"] != "behavior_failed" or len(materialized.files) != 1
            or measured.get("execution_origin") != "actual_csharp_process" or compilation.get("exit_code") != 0
            or ref(measured["assembly_ref"]["path"]) != measured["assembly_ref"]):
        raise ValueError("SFT_EVAL_REAL_C1_FAILURE_REQUIRED")
    source_ref = measured["source_ref"]
    if ref(source_ref["path"]) != source_ref or next(iter(materialized.files.values())) != Path(source_ref["path"]).read_bytes():
        raise ValueError("SFT_EVAL_C1_MEASURED_BYTES_MISMATCH")
    response_path = output / "teacher-C1.response.txt"
    with response_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(raw)
    response_ref = ref(response_path)
    planned = admitted.implementation_plan
    overlaid = _overlay_generated_patch_changes_on_plan(planned_changes=planned["proposed_changes"], generated_changes=generated.changes)
    overlaid = [{**row, "generation": {"model_response_sha256": response_ref["sha256"],
        "origin": "operator_authored_fixed_failure", "model_calls": 0}} for row in overlaid]
    hashes = {"feature_spec": request.feature_spec_ref.artifact_hash, "evidence_pack": request.evidence_pack_ref.artifact_hash,
              "implementation_plan": request.implementation_plan_ref.artifact_hash}
    proposal = PatchProposalBuilder().build({**planned, "proposed_changes": overlaid}, evidence_pack=admitted.evidence_pack,
                                            artifact_hashes=hashes).to_artifact_payload()
    review = ProposalReviewer().review(proposal, evidence_pack=admitted.evidence_pack, implementation_plan=planned,
        feature_spec=admitted.feature_spec, artifact_hashes=hashes, require_executable_patch=True).to_artifact_payload()
    if review["allowed_next_stage"] != "dry_run_apply":
        raise ValueError("SFT_EVAL_TEACHER_STRUCTURAL_REVIEW_BLOCKED: " + json.dumps(review))
    store = admitted.artifact_store
    proposal_ref = store.write(feature_id=request.feature_spec_ref.feature_id, artifact_type="patch_proposal", payload=proposal,
                               upstream_refs=[request.implementation_plan_ref, request.evidence_pack_ref])
    review_ref = store.write(feature_id=request.feature_spec_ref.feature_id, artifact_type="proposal_review", payload=review,
                             upstream_refs=[proposal_ref, request.implementation_plan_ref, request.evidence_pack_ref])
    dry_run = ExactTextPatchApplier().run(workspace_root=request.workspace_root, proposal=proposal,
                                        implementation_plan=planned, review=review, allow_write=False)
    save(output / "teacher-dry-run.json", dry_run)
    inspector = PlanBoundSourceInspector(workspace_root=request.workspace_root, implementation_plan=planned, policy=PatchAgentLoopPolicy())
    failure = {"status": "behavior_failed", "failure_owner": "patch_implementation",
        "owner_binding_ids": list(inspector.available_binding_ids), "generated_changes_sha256": canonical_hash(generated.changes),
        "source_snapshot_sha256": inspector.source_snapshot_sha256,
        "observations": [{"check_id": row["check"], "input": _observation_input(row, case_id=task["case_id"]),
            "expected": row["expected"], "observed": row["observed"], "result_ref": task["version_reports"]["C1"]}
            for row in measured["checks"] if not row["passed"]],
        "verification_ref": task["version_reports"]["C1"], "execution_origin": "unknown",
        "origin": "derived_from_executed_teacher_C1_not_new_model_or_verifier_run"}
    failure_ref = save(output / "fixed-failure.json", failure)
    parent = {"receipt_version": "operator_authored_fixed_failure_receipt_v1", "origin": "operator_authored_fixed_failure",
        "model_call_count": 0, "provider_send_count": 0, "historical_model_attempt_claimed": False,
        "generated_against_snapshot_sha256": _source_tree_snapshot_sha256(_csharp_source_snapshot(Path(request.workspace_root))),
        "artifacts": {"patch_proposal": asdict(proposal_ref), "proposal_review": asdict(review_ref)},
        "factory_receipt": {"model_call_count": 0, "model_response": {"path": response_ref["path"],
            "output_sha256": response_ref["sha256"], "model": "operator_authored_fixed_failure"}}}
    parent_ref = save(output / "teacher-parent-receipt.json", parent)
    return replace(request, repair_parent_proposal_ref=proposal_ref, repair_parent_review_ref=review_ref,
        repair_failure_report_ref=failure_ref, repair_parent_response_ref=response_ref, repair_parent_attempt_ref=parent_ref)


class OwnedSftOracle:
    """Only this verifier reads hidden test code; C2 is not supplied to a model."""
    def __init__(self, *, manifest_ref, case_id, output_root):
        self.task = _case(manifest_ref, case_id)
        hidden = read_bound_json(self.task["hidden_oracle_ref"])
        self.program = hidden["checks_source"]
        self.root, self.calls = Path(output_root), 0
        self.tools = toolchain()
        self.expected = read_bound_json(self.task["version_reports"]["C1"])["checks"]

    def verification_identity(self):
        return {"validator_sha256": canonical_hash({"runner": ref(__file__), "compiler_owner": ref(Path(execute_version.__code__.co_filename)),
                    "oracle_source_sha256": hashlib.sha256(self.program.encode()).hexdigest()}),
                "inputs_sha256": canonical_hash({"case_id": self.task["case_id"], "source_root": self.task["source_root"],
                    "original_checks": [{key: row[key] for key in ("check", "expected", "group")} for row in self.expected]}),
                "environment_sha256": canonical_hash(self.tools), "deterministic_reuse_allowed": False}

    def __call__(self, context):
        identity = self.verification_identity()
        self.calls += 1
        run = self.root / ("evaluation_" + str(self.calls))
        materialized = materialize_exact_text_patch(workspace_root=self.task["source_root"],
            diff_intent=context["generated_changes"], implementation_plan=context["implementation_plan"])
        code = {"baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256, "files": sorted(materialized.files)}
        if code != context["materialized_code_identity"] or len(materialized.files) != 1:
            raise ValueError("SFT_EVAL_CODE_IDENTITY_MISMATCH")
        status, owner, observations, matrix = "infrastructure_failed", "environment", [], None
        matrix_verified = False
        try:
            matrix = execute_version(run / "execution", next(iter(materialized.files.values())).decode("utf-8"), self.program, self.tools)
            if _check_definition_signature(matrix["checks"]) != _check_definition_signature(self.expected):
                raise ValueError("SFT_EVAL_CHECK_SET_CHANGED")
            matrix_verified = True
            failed = [row for row in matrix["checks"] if not row["passed"]]
            status, owner = ("behavior_failed", "patch_implementation") if failed else ("passed", None)
            observations = [{"check_id": row["check"], "input": _observation_input(row, case_id=self.task["case_id"]),
                "expected": row["expected"], "observed": row["observed"]} for row in failed]
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            compilation_path = run / "execution/compilation.json"
            if compilation_path.is_file():
                compilation = json.loads(compilation_path.read_text(encoding="utf-8"))
                if compilation["exit_code"] != 0:
                    status, owner = "behavior_failed", "patch_implementation"
                    observations = [{"check_id": "csharp_compilation", "input": {"source_identity": code},
                        "expected": {"exit_code": 0}, "observed": compilation}]
            if not observations:
                observations = [{"check_id": "verification_execution", "input": {}, "expected": "fixed checks execute",
                                 "observed": {"exception_type": type(exc).__name__}}]
        if self.verification_identity() != identity:
            raise ValueError("SFT_EVAL_VERIFIER_CHANGED")
        record = {"status": status, "failure_owner": owner,
            "owner_binding_ids": [row["edit_binding_id"] for row in context["generated_changes"]] if owner == "patch_implementation" else [],
            "observations": observations, "generated_changes_sha256": context["generated_changes_sha256"],
            "source_snapshot_sha256": context["source_snapshot_sha256"], "materialized_code_identity": code,
            "verification_identity": identity, "model_response_sha256": context["model_response"]["output_sha256"],
            "execution_origin": "executed" if matrix_verified else "incomplete",
            "verification_summary": {"checks": len(matrix["checks"]) if matrix else 0,
                "failed_checks": len(matrix["failed_checks"]) if matrix else 0, "executions": 1 if matrix else 0},
            "preservation_passed": matrix["preservation_passed"] if matrix else None,
            "verification_scope": "independent_owned_CSharp_checks_not_GTFO_or_formal_Build_Runtime"}
        if matrix is not None:
            record["verification_ref"] = matrix["report_ref"]
        return {**record, "report_ref": save(run / "report.json", record)}


def make_evaluator(**kwargs):
    return OwnedSftOracle(**kwargs)


def _profile(document):
    profile_type = {
        LOCAL_OLLAMA_PATCH_PROFILE_VERSION: LocalOllamaInitialPatchProfileV1,
        EXTERNAL_PATCH_PROFILE_VERSION: ExternalQwenPatchProfile,
    }.get(document.get("profile_version"))
    if profile_type is None:
        raise ValueError("SFT_EVAL_PROFILE_VERSION_UNSUPPORTED")
    # Both existing to_dict projections contain derived identity fields.
    return profile_type(**{field.name: document[field.name] for field in fields(profile_type)
                           if field.name in document})


def prepare_eval(*, manifest_ref, case_id, profile_ref, mode, attempt_id, output_root):
    if mode not in ("fixed_failure", "initial_patch") or not attempt_id:
        raise ValueError("SFT_EVAL_MODE_AND_ATTEMPT_REQUIRED")
    task = _case(manifest_ref, case_id)
    profile = _profile(read_bound_json(profile_ref))
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = replace(_request(read_bound_json(task["continuation_input_ref"])), attempt_id=attempt_id)
    if mode == "fixed_failure":
        request = _teacher_parent(request, task, output)
    policy = PatchAgentLoopPolicy(max_total_model_turns=5, max_tool_rounds=6, max_observation_bytes=32768,
        require_source_inspection=False, finalize_after_first_observation=False, enable_supporting_reads=True,
        allow_batched_reads=True, allow_upstream_request=True, reuse_covered_source_observations=True)
    worker_ref = save(output / "worker.json", {"schema_version": "patch_worker_configuration_v1", "policy": asdict(policy),
        "enable_recovery_loop": True, "behavior_evaluator": {"factory_ref": ref(__file__), "factory_name": "make_evaluator",
            "kwargs": {"manifest_ref": dict(manifest_ref), "case_id": case_id, "output_root": str(output / "verification")}}})
    request = replace(request, worker_configuration_ref=worker_ref, worker_profile_sha256=canonical_hash(profile.to_dict()))
    preflight_patch_continuation(request)
    request_ref = save(output / "continuation-input.json", asdict(request))
    repair_ref = None
    if mode == "fixed_failure":
        repair_ref = save(output / "repair-inputs.json", {key: asdict(getattr(request,key)) if key in
            ("repair_parent_proposal_ref", "repair_parent_review_ref") else dict(getattr(request,key)) for key in
            ("repair_parent_proposal_ref", "repair_parent_review_ref", "repair_failure_report_ref", "repair_parent_response_ref", "repair_parent_attempt_ref")})
    argv = None
    if isinstance(profile, LocalOllamaInitialPatchProfileV1):
        argv = ["continue-fresh-patch", "--artifact-root", request.artifact_store_root, "--workspace-root", request.workspace_root,
            "--feature-id", request.feature_spec_ref.feature_id, "--feature-spec-path", request.feature_spec_ref.path,
            "--feature-spec-hash", request.feature_spec_ref.artifact_hash, "--evidence-pack-path", request.evidence_pack_ref.path,
            "--evidence-pack-hash", request.evidence_pack_ref.artifact_hash, "--implementation-plan-path", request.implementation_plan_ref.path,
            "--implementation-plan-hash", request.implementation_plan_ref.artifact_hash, "--ollama-model", profile.model,
            "--ollama-base-url", profile.base_url, "--ollama-context-window", str(profile.context_window), "--ollama-output-budget", str(profile.output_budget),
            "--ollama-timeout-seconds", str(profile.timeout_seconds), "--patch-worker-config", worker_ref["path"], "--attempt-id", attempt_id]
        if repair_ref:
            argv.extend(["--repair-inputs", repair_ref["path"]])
    record = {"manifest_ref": dict(manifest_ref), "case_id": case_id, "family_id": task["family_id"], "split": task["split"],
        "mode": mode, "profile_ref": dict(profile_ref), "profile": profile.to_dict(), "attempt_id": attempt_id,
        "request_ref": request_ref, "worker_ref": worker_ref, "runner_ref": ref(__file__), "output_root": str(output),
        "cli_arguments": argv, "prepared_only": True, "model_calls": 0,
        "scope": "given_operator_Plan_not_natural_language_to_Runtime", "teacher_parent_is_model_history": False}
    if isinstance(profile, ExternalQwenPatchProfile):
        record.update(execution_entry="run_external_patch_continuation",
            source_files=dict(_csharp_source_snapshot(Path(request.workspace_root))),
            disclosure_requirements={
                "destination": {"provider": "qwen_chat", "model": profile.model, "base_url": profile.base_url,
                                "request_url": profile.base_url + "/chat/completions"},
                "required_data_classes": [*SOURCE_DISCLOSURE_DATA_CLASSES, *PATCH_FEEDBACK_DATA_CLASSES],
                "subsequent_feedback_data_classes": list(PATCH_FEEDBACK_DATA_CLASSES),
                "source_scope": "independent_owned_CSharp_only",
            })
    return {**record, "prepared_ref": save(output / "prepared.json", record)}


def execute_prepared(prepared_ref, *, execution_trace_configuration=None, source_disclosure_ref=None):
    prepared = read_bound_json(prepared_ref)
    if ref(__file__) != prepared["runner_ref"]:
        raise ValueError("SFT_EVAL_RUNNER_CHANGED")
    read_bound_json(prepared["profile_ref"])
    request = _request(read_bound_json(prepared["request_ref"]))
    profile = _profile(prepared["profile"])
    disclosure = None
    if isinstance(profile, ExternalQwenPatchProfile):
        if source_disclosure_ref is None:
            raise ValueError("SFT_EVAL_EXTERNAL_DISCLOSURE_REQUIRED")
        disclosure = read_bound_json(source_disclosure_ref)
    elif source_disclosure_ref is not None:
        raise ValueError("SFT_EVAL_LOCAL_DISCLOSURE_UNSUPPORTED")
    output = Path(prepared["output_root"]) / "actual"
    output.mkdir(exist_ok=False)
    config = PatchWorkerConfiguration.load(prepared["worker_ref"]["path"])
    started = time.perf_counter()
    if isinstance(profile, ExternalQwenPatchProfile):
        result = run_external_patch_continuation(request, profile=profile, worker_configuration=config,
            source_disclosure=disclosure, approved_source_files=prepared["source_files"],
            execution_trace_configuration=execution_trace_configuration)
    else:
        adapter = LocalOllamaInitialPatchAdapter(profile=profile, artifact_store=ArtifactStore(request.artifact_store_root),
            workspace_root=request.workspace_root, patch_agent_loop_policy=config.policy,
            behavior_evaluator=config.behavior_evaluator(), enable_recovery_loop=config.enable_recovery_loop)
        result = run_fresh_patch_continuation(request, patch_change_factory=adapter,
            dry_run_factory=make_exact_no_write_dry_run_factory(request.workspace_root), max_model_calls=config.policy.max_total_model_turns,
            execution_trace_configuration=execution_trace_configuration)
    elapsed = time.perf_counter()-started
    receipt = read_bound_json({"path": result.attempt_receipt["path"], "sha256": result.attempt_receipt["sha256"]}) if result.attempt_receipt else {}
    factory = receipt.get("factory_receipt", {})
    evaluations = factory.get("patch_agent_loop", {}).get("behavior_evaluations", [])
    final = evaluations[-1] if evaluations else {}
    code_revisions = factory.get("code_revision_comparisons", [])
    success = result.status == "terminal" and final.get("status") == "passed"
    first_pass = len(evaluations) == 1 and success and prepared["mode"] == "initial_patch"
    summary = {"case_id": prepared["case_id"], "family_id": prepared["family_id"], "mode": prepared["mode"],
        "profile": prepared["profile"], "task_succeeded": success, "initial_output_correct": first_pass,
        "repair_required": False if first_pass else prepared["mode"] == "fixed_failure" or len(evaluations)>1,
        "effective_revision": success and any(row.get("same_materialized_code") is False and row.get("comparable_to_known_failed") for row in code_revisions),
        "preservation_passed": read_bound_json(final["report_ref"]).get("preservation_passed") if final.get("report_ref") else None,
        "latency_seconds": elapsed, "execution_counts": result.execution_counts, "first_failure": result.first_failure,
        "verification_summary": final.get("verification_summary"), "attempt_ref": result.attempt_receipt,
        "actual_model_calls_recorded": factory.get("model_call_count"),
        "protocol_results": {"loop_termination": factory.get("patch_agent_loop", {}).get("termination"),
            "initial_submissions": factory.get("patch_agent_loop", {}).get("submission_trace", []),
            "correction_status": factory.get("patch_agent_loop", {}).get("behavior_correction", {}).get("status")},
        "token_usage": [{"turn_index": turn.get("turn_index"), "usage": turn.get("usage")} for turn in factory.get("worker_turns",[])],
        "natural_language_to_runtime_proven": False, "runtime_performed": False}
    if source_disclosure_ref is not None:
        summary["source_disclosure_ref"] = dict(source_disclosure_ref)
    save(output / "result.json", result.to_dict())
    return {**summary, "summary_ref": save(output / "summary.json", summary)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--case-id")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--mode", choices=("fixed_failure", "initial_patch"))
    parser.add_argument("--attempt-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-disclosure", type=Path,
                        help="External execution only: this attempt's explicit existing source disclosure record.")
    parser.add_argument("--source-disclosure-sha256", help="Exact SHA-256 of --source-disclosure.")
    parser.add_argument("--langfuse-metadata", action="store_true",
                        help="Execute only: export metadata via the existing configured Langfuse sink; never source or prompt bodies.")
    args = parser.parse_args(argv)
    if args.langfuse_metadata and not args.execute:
        parser.error("--langfuse-metadata requires --execute; preparation cannot enable export.")
    if args.source_disclosure or args.source_disclosure_sha256:
        if not args.execute or not (args.source_disclosure and args.source_disclosure_sha256):
            parser.error("Source disclosure requires --execute and both --source-disclosure and --source-disclosure-sha256.")
    if args.execute:
        if not args.prepared or not args.prepared_sha256:
            parser.error("Execution requires one exact --prepared and --prepared-sha256.")
        trace_configuration = None
        if args.langfuse_metadata:
            from src.mod_agent.patch_execution_trace import PatchExecutionTraceConfiguration
            from src.observability.patch_langfuse import LangfuseHttpTransport
            from src.observability.patch_live_langfuse import PatchLiveLangfuseSink

            trace_configuration = PatchExecutionTraceConfiguration(
                sink=PatchLiveLangfuseSink(LangfuseHttpTransport.from_environment()))
        execution_options = {"execution_trace_configuration": trace_configuration}
        if args.source_disclosure:
            execution_options["source_disclosure_ref"] = {
                "path": str(args.source_disclosure.resolve()), "sha256": args.source_disclosure_sha256}
        result = execute_prepared({"path":str(args.prepared.resolve()), "sha256":args.prepared_sha256},
                                  **execution_options)
    else:
        if not all((args.manifest,args.manifest_sha256,args.case_id,args.profile,args.profile_sha256,args.mode,args.attempt_id,args.output_root)):
            parser.error("Prepare requires explicit manifest/profile hashes, case, mode, attempt and output root.")
        result = prepare_eval(manifest_ref={"path":str(args.manifest.resolve()),"sha256":args.manifest_sha256},
            case_id=args.case_id, profile_ref={"path":str(args.profile.resolve()),"sha256":args.profile_sha256},
            mode=args.mode, attempt_id=args.attempt_id, output_root=args.output_root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
