"""Local CLI for RepoForge."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

from .actual_execution import (
    ActualProductExecutionSelectionV1,
    ActualRunError,
    ActualRunPreparationRequestV1,
    execute_sealed_actual_run,
    prepare_sealed_actual_run,
)
from .artifacts import ArtifactRef, ArtifactStore
from .artifacts.store import ArtifactStorePreflightError
from .build.dependencies import RunLocalGameFolderBindingV1
from .edit_authority_capabilities import (
    parse_current_edit_authority_review_result,
)
from .evidence import R8EvidenceError, R8ILInspectionError
from .evidence_only import EvidenceOnlyError, run_gtfo_r8_evidence_only
from .fresh_coverage import (
    FreshCoverageError,
    FreshCoverageProgressReporter,
    blocked_fresh_coverage_result,
    create_local_ollama_requirement_coverage_completion,
    load_feature_spec_payload,
    run_fresh_requirement_coverage,
)
from .fresh_planning_continuation import (
    PATCH_CONTINUATION_CONTRACT_VERSION,
    PLANNING_CONTINUATION_CONTRACT_VERSION,
    PatchContinuationError,
    PatchContinuationInputV1,
    PlanningContinuationError,
    PlanningContinuationInputV1,
    _canonical_sha256,
    bind_patch_repair_verification,
    blocked_patch_continuation_result,
    blocked_planning_continuation_result,
    run_fresh_patch_continuation,
    run_fresh_planning_continuation,
)
from .intelligent_evidence_only import (
    IntelligentEvidenceEntryError,
    make_fresh_api_source_augmentation_factory,
    make_fresh_assisted_api_source_augmentation_factory,
    run_intelligent_evidence_only,
)
from .local_entry import (
    LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
    EditAuthorityPresentationError,
    LocalEntryError,
    LocalOllamaInitialPatchAdapter,
    LocalOllamaInitialPatchProfileV1,
    make_exact_no_write_dry_run_factory,
    preflight_local_runtime,
    preflight_stagger_edit_authority_review,
    resume_local_runtime,
    resume_stagger_edit_authority_review,
    run_local_request,
)
from .model_qualification import ModelQualificationError, resolve_qualified_patch_profile
from .patch_execution_trace import exception_execution_trace_projection
from .patch_upstream_configuration import PatchUpstreamConfiguration
from .patch_worker_configuration import PatchWorkerConfiguration, read_json_object
from .planning import ModPlanningBriefBuilder, RequestPlanner
from .preflight_preparation import (
    DeterministicPostChangeConditionsV1,
    PreflightPreparationError,
    RejectedPreflightIdentityReservationV1,
    RunFamilyContinuationProfileV1,
    RunFamilyContinuationRequestV1,
    SealedPreflightPreparationRequestV1,
    SelectedPreflightConfigurationV1,
    prepare_sealed_entry_preflight,
)
from .product_execution import (
    ProductExecutionContractError,
    ProductExecutionResultV1,
)
from .product_executor import ProductExecutionPresentationError
from .proposal.patch_agent_loop import PatchAgentLoopPolicy
from .runtime import (
    StaggerDiagnosticDiffPreflightError,
    StaggerEditAuthorityApprovalError,
    StaggerEditAuthorityPreflightError,
    create_stagger_edit_authority_request,
    create_stagger_existing_binding_reauthorization_request,
    create_stagger_hud_parent_observation_authority_request,
    preflight_stagger_edit_authority,
    preflight_stagger_existing_binding_reauthorization,
    preflight_stagger_hud_parent_diagnostic_exact_diff,
    preflight_stagger_hud_parent_observation_authority,
)
from .tools import PermissionModel


@dataclass(frozen=True)
class PlanRequestResult:
    """Structured no-write planning result returned by plan-request."""

    status: str
    command: str
    request: str
    execution_mode: str
    writes_performed: bool
    stopped_before: str
    request_planning_report: dict[str, object]
    mod_planning_brief: dict[str, object]


def _task_agent_edit_paths(value):
    if value is None:
        return ()
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or any(type(item) is not str for item in parsed):
        raise ValueError("RUN_REQUEST_TASK_AGENT_EDIT_SCOPE_INVALID")
    return tuple(parsed)


def _apply_failure_cli_details(error):
    from .execution.patch_apply import patch_apply_failure_details

    details = patch_apply_failure_details(error)
    return {"product_source_writes_performed": details["actual_write_performed"],
            "apply_write_details": details, **exception_execution_trace_projection(error)}


def _execution_trace_configuration(args):
    from src.mod_agent.patch_execution_trace import PatchExecutionTraceConfiguration

    sink = None
    if args.langfuse_metadata:
        from src.observability.patch_langfuse import LangfuseHttpTransport
        from src.observability.patch_live_langfuse import PatchLiveLangfuseSink

        sink = PatchLiveLangfuseSink(LangfuseHttpTransport.from_environment())
    return PatchExecutionTraceConfiguration(environment=args.trace_environment, sink=sink)


def _add_execution_trace_arguments(parser):
    parser.add_argument("--trace-environment", default="local",
        help="Deployment label for measured execution observations; default local.")
    parser.add_argument("--langfuse-metadata", action="store_true",
        help="Explicitly export allowlisted execution metadata to the configured Langfuse project; no source, prompts or raw replies.")


def _load_reviewed_build_resume(path):
    from src.mod_agent.workflow.interrupts import ResumePayload

    value = read_json_object(path)
    declared = fields(ResumePayload)
    required = {item.name for item in declared if item.default is MISSING and item.default_factory is MISSING}
    if not required.issubset(value) or not set(value).issubset({item.name for item in declared}):
        raise ValueError("REVIEWED_BUILD_RESUME_FIELDS_INVALID")
    if (not isinstance(value["artifact_refs"], list)
            or any(not isinstance(item, str) or not item for item in value["artifact_refs"])):
        raise ValueError("REVIEWED_BUILD_RESUME_REFERENCES_INVALID")
    return ResumePayload(**{**value, "artifact_refs": tuple(value["artifact_refs"])})


def _load_reviewed_build_profile(path, destination):
    from src.mod_agent.build.adapter import BuildProfile

    value = read_json_object(path)
    declared = fields(BuildProfile)
    required = {item.name for item in declared if item.default is MISSING and item.default_factory is MISSING}
    if not required.issubset(value) or not set(value).issubset({item.name for item in declared}):
        raise ValueError("REVIEWED_BUILD_PROFILE_FIELDS_INVALID")
    if any(not isinstance(value[name], str) or not value[name]
            for name in ("profile_id", "build_system", "working_directory")):
        raise ValueError("REVIEWED_BUILD_PROFILE_VALUES_INVALID")
    if (type(value["timeout_seconds"]) is not int or value["timeout_seconds"] < 1
            or not isinstance(value["command"], list) or not value["command"]
            or any(not isinstance(item, str) or not item for item in value["command"])):
        raise ValueError("REVIEWED_BUILD_PROFILE_COMMAND_INVALID")
    if (not Path(value["working_directory"]).is_absolute()
            or Path(value["working_directory"]).resolve() != destination):
        raise ValueError("REVIEWED_BUILD_PROFILE_WORKSPACE_MISMATCH")
    expected = value.get("expected_artifacts")
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(item, str) or not item for item in expected)):
        raise ValueError("REVIEWED_BUILD_EXPECTED_ARTIFACTS_REQUIRED")
    for item in expected:
        relative = Path(item)
        output = (destination / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not output.is_relative_to(destination):
            raise ValueError("REVIEWED_BUILD_EXPECTED_ARTIFACT_PATH_INVALID")
        if output.exists():
            raise ValueError("REVIEWED_BUILD_PREEXISTING_OUTPUT_REJECTED")
    if value.get("log_path") is not None and (not isinstance(value["log_path"], str) or not value["log_path"]):
        raise ValueError("REVIEWED_BUILD_PROFILE_LOG_PATH_INVALID")
    return BuildProfile(**{**value, "command": tuple(value["command"]), "expected_artifacts": tuple(expected)})


def _continue_reviewed_build_from_cli(args):
    from src.mod_agent.build.adapter import BuildAdapter, run_build_profile_command
    from src.mod_agent.reviewed_patch_continuation import (
        _reviewed_build_inputs,
        continue_reviewed_patch,
    )

    if (not args.allow_build or not all((args.artifact_store_root, args.feature_id,
            args.checkpoint_path, args.checkpoint_hash, args.resume_payload))):
        raise ValueError("REVIEWED_BUILD_EXPLICIT_CONFIGURATION_REQUIRED")
    if any((args.confirm, args.workspace_root, args.run_id, args.isolated_root)):
        raise ValueError("REVIEWED_BUILD_APPLY_ARGUMENTS_FORBIDDEN")
    checkpoint_ref = ArtifactRef(feature_id=args.feature_id, artifact_type="workflow_checkpoint",
        path=args.checkpoint_path, artifact_hash=args.checkpoint_hash)
    payload = _load_reviewed_build_resume(args.resume_payload)
    _, source_binding, _, destination = _reviewed_build_inputs(ArtifactStore(args.artifact_store_root), checkpoint_ref, payload)
    if source_binding.get("repository_build_and_test") is not None:
        if args.build_profile is not None:
            raise ValueError("REPOSITORY_BUILD_PROFILE_OVERRIDE_FORBIDDEN")
        return continue_reviewed_patch(artifact_store_root=args.artifact_store_root,
            checkpoint_ref=checkpoint_ref, resume_payload=payload, allow_build=True,
            execution_trace_configuration=_execution_trace_configuration(args))
    if not args.build_profile:
        raise ValueError("REVIEWED_BUILD_EXPLICIT_CONFIGURATION_REQUIRED")
    profile = _load_reviewed_build_profile(args.build_profile, destination)

    def build_applied(selected_destination, feature_id, mode):
        if selected_destination != destination:
            raise ValueError("REVIEWED_BUILD_PROFILE_WORKSPACE_MISMATCH")
        # The original BuildAdapter owns diagnostics and output verification;
        # the shared command runner owns process execution. No nested CLI runs.
        return BuildAdapter().run_local(feature_id=feature_id, profile=profile,
            command_runner=lambda selected: run_build_profile_command(selected, environment=os.environ.copy()),
            allow_execution=True, input_execution_mode=mode)

    return continue_reviewed_patch(artifact_store_root=args.artifact_store_root,
        checkpoint_ref=checkpoint_ref, resume_payload=payload, allow_build=True,
        build_result_factory=build_applied, execution_trace_configuration=_execution_trace_configuration(args))


def _strict_cli_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("value must be exactly true or false")


def _read_exact_diff_approval_from_stdin(dry_run: dict[str, object]) -> str:
    semantic_review_required = (
        dry_run.get("human_semantic_review_required") is True
    )
    semantic_binding = dry_run.get("human_semantic_decision_binding")
    semantic_decision_template = (
        {
            **dict(semantic_binding),
            "decision": "approve_or_reject",
            "reviewer_identity": "<non-empty human reviewer identity>",
        }
        if semantic_review_required and isinstance(semantic_binding, Mapping)
        else None
    )
    print(
        json.dumps(
            {
                "status": "AWAITING_EXACT_DIFF_APPROVAL",
                "diff_hash": dry_run.get("diff_hash"),
                "context_hashes": dry_run.get("context_hashes"),
                "unified_diff": dry_run.get("unified_diff"),
                "unified_diff_sha256": dry_run.get("unified_diff_sha256"),
                "unified_diff_bytes": dry_run.get("unified_diff_bytes"),
                "diff_stats": dry_run.get("diff_stats"),
                "static_checks": dry_run.get("static_checks"),
                "diff_preview": dry_run.get("diff_preview"),
                "human_semantic_review_required": semantic_review_required,
                "human_semantic_decision_binding": semantic_binding,
                "human_semantic_decision_template": semantic_decision_template,
                "instruction": (
                    (
                        "Paste one JSON object matching human_semantic_decision_template; "
                        "set decision to approve or reject and provide a non-empty "
                        "reviewer_identity. Every displayed binding value must remain exact."
                    )
                    if semantic_review_required
                    else (
                        "Paste the exact diff_hash to approve isolated apply; "
                        "enter REJECTED_BY_USER to explicitly reject it; any "
                        "other input blocks the write as a hash mismatch."
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )
    prompt = (
        "Exact diff human semantic decision JSON: "
        if semantic_review_required
        else "Exact diff hash approval: "
    )
    print(prompt, end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip()


def _read_typed_json_object_file(path_value: str, role: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path_value).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} file is not valid readable UTF-8 JSON.") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} file must contain one JSON object.")
    return dict(value)


def _read_patch_repair_inputs(path: str) -> dict[str, object]:
    """Deserialize existing Repair refs; their owner performs semantic admission."""
    value = read_json_object(path)
    expected = {
        "repair_parent_proposal_ref", "repair_parent_review_ref",
        "repair_failure_report_ref", "repair_parent_response_ref", "repair_parent_attempt_ref",
    }
    if set(value) != expected:
        raise ValueError("Repair inputs must contain exactly the five existing Repair reference fields.")
    for name in ("repair_parent_proposal_ref", "repair_parent_review_ref"):
        ref = value[name]
        if not isinstance(ref, dict) or set(ref) != {"feature_id", "artifact_type", "path", "artifact_hash"}:
            raise ValueError(f"{name} must be an exact ArtifactRef object.")
        value[name] = ArtifactRef(**ref)
    return value


def _reassemble_saved_patch_from_cli(args) -> int:
    """Use the existing continuation owner without constructing any model adapter."""
    started = time.perf_counter()
    request = None
    try:
        if any(value is not None for value in (
            args.ollama_base_url, args.ollama_model, args.ollama_context_window,
            args.ollama_output_budget, args.ollama_timeout_seconds,
            args.qualified_model_routing, args.patch_worker_config, args.patch_upstream_config,
        )) or args.patch_agent_loop:
            raise PatchContinuationError("PATCH_SAVED_MODEL_ARGUMENT_CONFLICT", "request",
                "Saved Patch reassembly cannot accept model, worker or upstream execution arguments.")
        request = PatchContinuationInputV1(
            contract_version=PATCH_CONTINUATION_CONTRACT_VERSION,
            artifact_store_root=args.artifact_root,
            workspace_root=args.workspace_root,
            feature_spec_ref=ArtifactRef(feature_id=args.feature_id, artifact_type="feature_spec",
                path=args.feature_spec_path, artifact_hash=args.feature_spec_hash),
            evidence_pack_ref=ArtifactRef(feature_id=args.feature_id, artifact_type="evidence_pack",
                path=args.evidence_pack_path, artifact_hash=args.evidence_pack_hash),
            implementation_plan_ref=ArtifactRef(feature_id=args.feature_id, artifact_type="implementation_plan",
                path=args.implementation_plan_path, artifact_hash=args.implementation_plan_hash),
            attempt_id=args.attempt_id,
            saved_patch_selection=read_json_object(args.reassemble_saved_patch),
        )
        result = run_fresh_patch_continuation(
            request, patch_change_factory=None,
            dry_run_factory=make_exact_no_write_dry_run_factory(args.workspace_root),
            max_model_calls=0,
            execution_trace_configuration=_execution_trace_configuration(args),
        )
    except (PatchContinuationError, LocalEntryError, OSError, ValueError) as exc:
        result = blocked_patch_continuation_result(
            code=str(getattr(exc, "code", type(exc).__name__)),
            stage=str(getattr(exc, "stage", "request")),
            reason=str(getattr(exc, "reason", exc)),
            duration_seconds=time.perf_counter() - started,
            feature_id=args.feature_id,
        )
    payload = result.to_dict()
    payload["generation_origin"] = "reused_saved_response"
    payload["new_provider_sends"] = 0
    exit_code = 0 if result.status == "terminal" else 1
    if result.status == "terminal":
        from src.mod_agent.reviewed_patch_continuation import review_run

        view = review_run(artifact_store_root=args.artifact_root, workspace_root=args.workspace_root,
            feature_id=args.feature_id, run_id=result.run_id)
        payload["review"] = view
        if view["status"] == "reviewable_diff":
            payload.update(
                approval_status="awaiting_exact_diff_approval",
                unified_diff=view["unified_diff"], diff_hash=view["diff_hash"],
            )
        else:
            payload["approval_status"] = "review_unavailable"
            exit_code = 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return exit_code


def plan_request(
    description: str,
    *,
    target_game: str | None = None,
    target_mod: str | None = None,
) -> PlanRequestResult:
    """Run request planning and stop before feature specification."""

    request_plan = RequestPlanner().plan(description, target_mod=target_mod)
    planning_brief = ModPlanningBriefBuilder().build_planning_only(
        request_plan,
        target_game=target_game,
    )
    return PlanRequestResult(
        status=planning_brief.decision,
        command="plan-request",
        request=description,
        execution_mode="planning_only",
        writes_performed=False,
        stopped_before="feature_spec",
        request_planning_report=request_plan.to_artifact_payload(),
        mod_planning_brief=planning_brief.to_artifact_payload(),
    )


def _add_stagger_edit_authority_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--artifact-store-root", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-spec-path", required=True)
    parser.add_argument("--feature-spec-hash", required=True)
    parser.add_argument("--runtime-evidence-path", required=True)
    parser.add_argument("--runtime-evidence-hash", required=True)
    parser.add_argument("--implementation-plan-path", required=True)
    parser.add_argument("--implementation-plan-hash", required=True)
    parser.add_argument("--patch-proposal-path", required=True)
    parser.add_argument("--patch-proposal-hash", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--canonical-target-path", required=True)
    parser.add_argument("--target-symbol", required=True)
    parser.add_argument("--target-start-line", required=True, type=int)
    parser.add_argument("--target-end-line", required=True, type=int)
    parser.add_argument("--proposed-operation", required=True)


def _stagger_edit_authority_arguments(
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "artifact_store_root": args.artifact_store_root,
        "feature_id": args.feature_id,
        "run_id": args.run_id,
        "feature_spec_path": args.feature_spec_path,
        "feature_spec_hash": args.feature_spec_hash,
        "runtime_evidence_path": args.runtime_evidence_path,
        "runtime_evidence_hash": args.runtime_evidence_hash,
        "implementation_plan_path": args.implementation_plan_path,
        "implementation_plan_hash": args.implementation_plan_hash,
        "patch_proposal_path": args.patch_proposal_path,
        "patch_proposal_hash": args.patch_proposal_hash,
        "source_root": args.source_root,
        "canonical_target_path": args.canonical_target_path,
        "target_symbol": args.target_symbol,
        "target_start_line": args.target_start_line,
        "target_end_line": args.target_end_line,
        "proposed_operation": args.proposed_operation,
    }


def _add_stagger_existing_binding_reauthorization_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--artifact-store-root", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-spec-path", required=True)
    parser.add_argument("--feature-spec-hash", required=True)
    parser.add_argument("--runtime-evidence-path", required=True)
    parser.add_argument("--runtime-evidence-hash", required=True)
    parser.add_argument(
        "--predecessor-implementation-plan-path",
        required=True,
    )
    parser.add_argument(
        "--predecessor-implementation-plan-hash",
        required=True,
    )
    parser.add_argument("--patch-proposal-path", required=True)
    parser.add_argument("--patch-proposal-hash", required=True)
    parser.add_argument(
        "--current-implementation-plan-path",
        required=True,
    )
    parser.add_argument(
        "--current-implementation-plan-hash",
        required=True,
    )
    parser.add_argument(
        "--existing-binding-resolution-path",
        required=True,
    )
    parser.add_argument(
        "--existing-binding-resolution-hash",
        required=True,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--canonical-target-path", required=True)
    parser.add_argument("--target-symbol", required=True)
    parser.add_argument("--target-start-line", required=True, type=int)
    parser.add_argument("--target-end-line", required=True, type=int)


def _stagger_existing_binding_reauthorization_arguments(
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "artifact_store_root": args.artifact_store_root,
        "feature_id": args.feature_id,
        "run_id": args.run_id,
        "feature_spec_path": args.feature_spec_path,
        "feature_spec_hash": args.feature_spec_hash,
        "runtime_evidence_path": args.runtime_evidence_path,
        "runtime_evidence_hash": args.runtime_evidence_hash,
        "predecessor_implementation_plan_path": (
            args.predecessor_implementation_plan_path
        ),
        "predecessor_implementation_plan_hash": (
            args.predecessor_implementation_plan_hash
        ),
        "patch_proposal_path": args.patch_proposal_path,
        "patch_proposal_hash": args.patch_proposal_hash,
        "current_implementation_plan_path": (
            args.current_implementation_plan_path
        ),
        "current_implementation_plan_hash": (
            args.current_implementation_plan_hash
        ),
        "existing_binding_resolution_path": (
            args.existing_binding_resolution_path
        ),
        "existing_binding_resolution_hash": (
            args.existing_binding_resolution_hash
        ),
        "source_root": args.source_root,
        "canonical_target_path": args.canonical_target_path,
        "target_symbol": args.target_symbol,
        "target_start_line": args.target_start_line,
        "target_end_line": args.target_end_line,
    }


def _add_stagger_observation_authority_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--artifact-store-root", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-spec-path", required=True)
    parser.add_argument("--feature-spec-hash", required=True)
    parser.add_argument("--runtime-evidence-path", required=True)
    parser.add_argument("--runtime-evidence-hash", required=True)
    parser.add_argument("--current-implementation-plan-path", required=True)
    parser.add_argument("--current-implementation-plan-hash", required=True)
    parser.add_argument("--patch-proposal-path", required=True)
    parser.add_argument("--patch-proposal-hash", required=True)
    parser.add_argument("--existing-binding-resolution-path", required=True)
    parser.add_argument("--existing-binding-resolution-hash", required=True)
    parser.add_argument(
        "--existing-edit-authority-approval-path",
        required=True,
    )
    parser.add_argument(
        "--existing-edit-authority-approval-hash",
        required=True,
    )
    parser.add_argument("--evidence-pack-path", required=True)
    parser.add_argument("--evidence-pack-hash", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--build-dependency-root", required=True)
    parser.add_argument("--unity-core-module-hash", required=True)
    parser.add_argument("--modules-asm-hash", required=True)
    parser.add_argument("--il2cpp-interop-runtime-hash", required=True)


def _stagger_observation_authority_arguments(
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "artifact_store_root": args.artifact_store_root,
        "feature_id": args.feature_id,
        "run_id": args.run_id,
        "feature_spec_path": args.feature_spec_path,
        "feature_spec_hash": args.feature_spec_hash,
        "runtime_evidence_path": args.runtime_evidence_path,
        "runtime_evidence_hash": args.runtime_evidence_hash,
        "current_implementation_plan_path": (
            args.current_implementation_plan_path
        ),
        "current_implementation_plan_hash": (
            args.current_implementation_plan_hash
        ),
        "patch_proposal_path": args.patch_proposal_path,
        "patch_proposal_hash": args.patch_proposal_hash,
        "existing_binding_resolution_path": (
            args.existing_binding_resolution_path
        ),
        "existing_binding_resolution_hash": (
            args.existing_binding_resolution_hash
        ),
        "existing_edit_authority_approval_path": (
            args.existing_edit_authority_approval_path
        ),
        "existing_edit_authority_approval_hash": (
            args.existing_edit_authority_approval_hash
        ),
        "evidence_pack_path": args.evidence_pack_path,
        "evidence_pack_hash": args.evidence_pack_hash,
        "source_root": args.source_root,
        "build_dependency_root": args.build_dependency_root,
        "unity_core_module_hash": args.unity_core_module_hash,
        "modules_asm_hash": args.modules_asm_hash,
        "il2cpp_interop_runtime_hash": (
            args.il2cpp_interop_runtime_hash
        ),
    }


def _add_stagger_diagnostic_diff_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--artifact-store-root", required=True)
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--edit-authority-approval-path", required=True)
    parser.add_argument("--edit-authority-approval-hash", required=True)
    parser.add_argument("--evidence-pack-path", required=True)
    parser.add_argument("--evidence-pack-hash", required=True)
    parser.add_argument("--implementation-plan-path", required=True)
    parser.add_argument("--implementation-plan-hash", required=True)
    parser.add_argument("--source-root", required=True)


def build_parser(
    *,
    _prepare_continuation: bool = False,
) -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(
        prog="mod-agent",
        description="RepoForge — evidence-grounded Coding Agent CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    retained_handoff = subparsers.add_parser("resume-retained-task-agent-handoff",
        help="Recover an already checked native candidate to exact approval without models, Build or Apply.")
    retained_handoff.add_argument("--execution-config", required=True)
    retained_handoff.add_argument("--run-root", required=True)
    retained_handoff.add_argument("--run-id", required=True)
    retained_handoff.add_argument("--execution-path", required=True)
    retained_handoff.add_argument("--execution-sha256", required=True)
    retained_handoff.add_argument("--attempt-id", required=True)

    plan_parser = subparsers.add_parser(
        "plan-request",
        help="Create a no-write request plan and planning brief.",
    )
    plan_parser.add_argument("description", help="Natural-language request description.")
    plan_parser.add_argument("--target-game")
    plan_parser.add_argument("--target-mod")

    prepare_entry_parser = subparsers.add_parser(
        "prepare-sealed-entry-preflight",
        help=(
            "Derive the next identities and atomically seal one five-file "
            "preflight-only Entry packet."
        ),
    )
    explicit_required = not _prepare_continuation
    prepare_entry_parser.add_argument("--packet-root", required=True)
    prepare_entry_parser.add_argument(
        "--run-family-continuation-profile",
        help=(
            "Use one typed finite-anchor profile; when present, all explicit "
            "identity, lineage, platform, source, model, and configuration "
            "arguments are forbidden."
        ),
    )
    prepare_entry_parser.add_argument("--packet-date", required=explicit_required)
    prepare_entry_parser.add_argument("--task-id", required=explicit_required)
    prepare_entry_parser.add_argument("--request-text", required=explicit_required)
    prepare_entry_parser.add_argument("--identity-series", required=explicit_required)
    prepare_entry_parser.add_argument(
        "--identity-sequence",
        required=explicit_required,
        type=int,
    )
    prepare_entry_parser.add_argument("--configuration-id", required=explicit_required)
    prepare_entry_parser.add_argument("--route", required=explicit_required)
    prepare_entry_parser.add_argument(
        "--main-ollama-timeout-seconds",
        required=explicit_required,
        type=float,
    )
    prepare_entry_parser.add_argument("--timeout-owner", required=explicit_required)
    prepare_entry_parser.add_argument(
        "--repair-ollama-timeout-seconds",
        type=float,
    )
    prepare_entry_parser.add_argument(
        "--repair-timeout-selection",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--projection-contract-checkpoint",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--model-projection-field",
        action="append",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--model-relations-authorized",
        required=explicit_required,
        type=_strict_cli_bool,
    )
    prepare_entry_parser.add_argument(
        "--deterministic-relation-completion",
        required=explicit_required,
        type=_strict_cli_bool,
    )
    prepare_entry_parser.add_argument(
        "--deterministic-relation-rule",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--deterministic-post-change-only-unaccounted",
        required=explicit_required,
        type=_strict_cli_bool,
    )
    prepare_entry_parser.add_argument(
        "--deterministic-post-change-required-before-patch-policy",
        required=explicit_required,
        type=_strict_cli_bool,
    )
    prepare_entry_parser.add_argument(
        "--deterministic-post-change-allowed-disposition",
        action="append",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--deterministic-post-change-all-other-omissions",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--initial-projection-attempts",
        required=explicit_required,
        type=int,
    )
    prepare_entry_parser.add_argument(
        "--maximum-correction-attempts-after-initial",
        required=explicit_required,
        type=int,
    )
    prepare_entry_parser.add_argument(
        "--provider-fallback",
        required=explicit_required,
        type=_strict_cli_bool,
    )
    prepare_entry_parser.add_argument(
        "--automatic-retry",
        required=explicit_required,
        type=_strict_cli_bool,
    )
    prepare_entry_parser.add_argument(
        "--external-disclosure",
        required=explicit_required,
        type=_strict_cli_bool,
    )
    prepare_entry_parser.add_argument("--platform-root", required=explicit_required)
    prepare_entry_parser.add_argument(
        "--allowed-dirty-path",
        action="append",
        default=[],
    )
    prepare_entry_parser.add_argument("--source-root", required=explicit_required)
    prepare_entry_parser.add_argument("--target-project", required=explicit_required)
    prepare_entry_parser.add_argument("--project-file", required=explicit_required)
    prepare_entry_parser.add_argument(
        "--primary-source",
        action="append",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument("--git-executable", required=explicit_required)
    prepare_entry_parser.add_argument("--ollama-base-url", required=explicit_required)
    prepare_entry_parser.add_argument("--ollama-model", required=explicit_required)
    prepare_entry_parser.add_argument(
        "--ollama-context-window",
        required=explicit_required,
        type=int,
    )
    prepare_entry_parser.add_argument(
        "--ollama-output-budget",
        required=explicit_required,
        type=int,
    )
    prepare_entry_parser.add_argument(
        "--preflight-lineage-packet",
        action="append",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--predecessor-receipt",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--predecessor-receipt-manifest",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--actual-lineage-packet",
        action="append",
        required=explicit_required,
    )
    prepare_entry_parser.add_argument(
        "--rejected-preflight-identity-reservation",
        action="append",
        nargs=3,
        default=[],
        metavar=("PACKET", "RECEIPT", "RECEIPT_MANIFEST"),
    )

    prepare_actual_parser = subparsers.add_parser(
        "prepare-sealed-actual",
        help="Compile a ready Entry receipt into a sealed, preparation-only Actual spec.",
    )
    prepare_actual_parser.add_argument("--accepted-preflight-packet", required=True)
    prepare_actual_parser.add_argument(
        "--accepted-preflight-packet-sha256", required=True
    )
    prepare_actual_parser.add_argument(
        "--accepted-preflight-manifest-sha256", required=True
    )
    prepare_actual_parser.add_argument(
        "--accepted-payload-manifest-sha256", required=True
    )
    prepare_actual_parser.add_argument("--accepted-receipt", required=True)
    prepare_actual_parser.add_argument("--accepted-receipt-sha256", required=True)
    prepare_actual_parser.add_argument("--accepted-receipt-manifest", required=True)
    prepare_actual_parser.add_argument(
        "--accepted-receipt-manifest-sha256", required=True
    )
    prepare_actual_parser.add_argument("--workspace-root", required=True)
    prepare_actual_parser.add_argument("--settings", required=True)
    prepare_actual_parser.add_argument(
        "--completion-classification",
        action="append",
        nargs=2,
        default=[],
        metavar=("NAME", "TRUE_OR_FALSE"),
    )
    prepare_actual_parser.add_argument(
        "--objective-verification-argument", action="append", default=[]
    )
    prepare_actual_parser.add_argument("--game-folder-build-binding-file")
    prepare_actual_parser.add_argument(
        "--compiled-artifact-inspection-profile-file"
    )

    execute_actual_parser = subparsers.add_parser(
        "execute-sealed-actual",
        help="Execute one sealed Actual spec after separate explicit authorization.",
    )
    execute_actual_parser.add_argument("--packet-root", required=True)
    execute_actual_parser.add_argument("--expected-packet-sha256", required=True)
    execute_actual_parser.add_argument("--expected-seal-sha256", required=True)
    execute_actual_parser.add_argument(
        "--authorize-actual-execution", action="store_true"
    )

    evidence_parser = subparsers.add_parser(
        "evidence-request",
        help="Read the approved GTFO R8 snapshot and stop before implementation planning.",
    )
    evidence_parser.add_argument("description", help="Natural-language Mod request.")
    evidence_parser.add_argument("--evidence-root", required=True)
    evidence_parser.add_argument("--artifact-root", default="artifacts/evidence_only")
    evidence_parser.add_argument("--target-mod", default="WallCrawler")
    evidence_parser.add_argument("--assembly-path")
    evidence_parser.add_argument("--current-interop-path")
    evidence_parser.add_argument("--enemy-animation-fix-path")
    evidence_parser.add_argument("--stuck-enemy-fix-path")

    fresh_coverage_parser = subparsers.add_parser(
        "fresh-coverage",
        help=(
            "Run requirement projection, current-source rg retrieval, findings, "
            "and deterministic coverage resolution, then stop."
        ),
    )
    fresh_coverage_parser.add_argument("--feature-spec", required=True)
    fresh_coverage_parser.add_argument("--workspace-root", required=True)
    fresh_coverage_parser.add_argument("--search-root", required=True)
    fresh_coverage_parser.add_argument("--artifact-root", required=True)
    fresh_coverage_parser.add_argument(
        "--settings", default="config/settings.yaml"
    )
    fresh_coverage_parser.add_argument(
        "--local-ollama",
        action="store_true",
        help=(
            "Explicitly select loopback-only Ollama execution; external model "
            "disclosure is not admitted by fresh-coverage."
        ),
    )
    fresh_coverage_parser.add_argument(
        "--ollama-base-url", default="http://127.0.0.1:11434"
    )
    fresh_coverage_parser.add_argument("--ollama-model")
    fresh_coverage_parser.add_argument(
        "--ollama-context-window", type=int
    )
    fresh_coverage_parser.add_argument(
        "--ollama-output-budget", type=int, default=4096
    )
    fresh_coverage_parser.add_argument(
        "--ollama-timeout-seconds", type=float
    )
    fresh_coverage_parser.add_argument(
        "--api-selection-mode",
        choices=("autonomous", "assisted_explicit"),
        help=(
            "Explicitly select model-driven or operator-selected API symbol "
            "resolution. Requires --api-source-registry and --api-source-id."
        ),
    )
    fresh_coverage_parser.add_argument(
        "--api-source-registry",
        help=(
            "Enable same-run read-only API evidence augmentation from this "
            "existing source registry. Requires --api-source-id."
        ),
    )
    fresh_coverage_parser.add_argument(
        "--api-source-id",
        help=(
            "Exact decompiled_api source_id selected for Fresh augmentation. "
            "Requires --api-source-registry."
        ),
    )
    fresh_coverage_parser.add_argument(
        "--api-symbol",
        action="append",
        default=[],
        help=(
            "One exact API method symbol selected by the operator. Repeat for "
            "assisted_explicit mode; forbidden in autonomous mode."
        ),
    )

    pretransport_parser = subparsers.add_parser("continue-pretransport-patch",
        help="Resume an exact terminal Patch pre-transport failure within its inherited task budget.")
    pretransport_parser.add_argument("--run-root", required=True)
    pretransport_parser.add_argument("--run-id", required=True)
    pretransport_parser.add_argument("--execution-config", required=True)
    pretransport_parser.add_argument("--recovery-config", required=True)
    pretransport_parser.add_argument("--prepare-only", action="store_true")
    planning_continuation_parser = subparsers.add_parser(
        "continue-fresh-planning",
        help=(
            "Consume exact Fresh Artifact revisions through implementation "
            "planning and stop before patch generation."
        ),
    )
    planning_continuation_parser.add_argument("--artifact-root", required=True)
    planning_continuation_parser.add_argument("--workspace-root", required=True)
    planning_continuation_parser.add_argument("--feature-id", required=True)
    planning_continuation_parser.add_argument(
        "--feature-spec-path", required=True
    )

    patch_continuation_parser = subparsers.add_parser(
        "continue-fresh-patch",
        help=(
            "Consume exact Fresh and ImplementationPlan revisions through one "
            "local Patch call, deterministic Review, and no-write Dry-run."
        ),
    )
    patch_continuation_parser.add_argument("--artifact-root", required=True)
    patch_continuation_parser.add_argument("--workspace-root", required=True)
    _add_execution_trace_arguments(patch_continuation_parser)
    patch_continuation_parser.add_argument("--feature-id", required=True)
    patch_continuation_parser.add_argument("--feature-spec-path", required=True)
    patch_continuation_parser.add_argument("--feature-spec-hash", required=True)
    patch_continuation_parser.add_argument("--evidence-pack-path", required=True)
    patch_continuation_parser.add_argument("--evidence-pack-hash", required=True)
    patch_continuation_parser.add_argument(
        "--implementation-plan-path", required=True
    )
    patch_continuation_parser.add_argument(
        "--implementation-plan-hash", required=True
    )
    patch_continuation_parser.add_argument(
        "--ollama-base-url",
        default=None,
    )
    patch_continuation_parser.add_argument("--ollama-model")
    patch_continuation_parser.add_argument(
        "--qualified-model-routing", help="Optional exact local role-profile selection from bound task qualification evidence.",
    )
    patch_continuation_parser.add_argument(
        "--ollama-context-window",
        type=int,
    )
    patch_continuation_parser.add_argument(
        "--ollama-output-budget",
        type=int,
        default=None,
    )
    patch_continuation_parser.add_argument(
        "--ollama-timeout-seconds",
        type=float,
    )
    patch_continuation_parser.add_argument(
        "--patch-agent-loop",
        action="store_true",
        help=(
            "Require a bounded Plan-binding source inspection loop before "
            "the final structured Patch response."
        ),
    )
    patch_continuation_parser.add_argument(
        "--patch-worker-config",
        help=("Explicit local worker JSON configuration. Overrides the legacy loop defaults; "
              "its hash-pinned evaluator factory is trusted local Python code executed by the host, not a model tool."),
    )
    patch_continuation_parser.add_argument(
        "--patch-upstream-config",
        help=("Optional local Evidence/Planning provider configuration for the existing Repair return. "
              "All provider calls share this attempt's model budget; it grants no additional edit or apply authority."),
    )
    patch_repair_selection = patch_continuation_parser.add_mutually_exclusive_group()
    patch_repair_selection.add_argument(
        "--repair-inputs", help="JSON object containing the five existing exact Repair references.",
    )
    patch_repair_selection.add_argument(
        "--repair-from-verification",
        help=("JSON containing source_attempt_ref and verification_result_ref; "
              "the product binds the current failed output and formal ancestry automatically."),
    )
    patch_repair_selection.add_argument(
        "--reassemble-saved-patch",
        help=("Exact source_attempt_ref and logical_budget_ref JSON for no-model "
              "reassembly of a verified saved Patch through Review and Dry-run."),
    )
    patch_continuation_parser.add_argument(
        "--attempt-id", help="Stable attempt ID; requires --patch-worker-config and is mandatory for Repair.",
    )
    planning_continuation_parser.add_argument(
        "--feature-spec-hash", required=True
    )
    planning_continuation_parser.add_argument(
        "--evidence-pack-path", required=True
    )
    planning_continuation_parser.add_argument(
        "--evidence-pack-hash", required=True
    )
    planning_continuation_parser.add_argument(
        "--planning-target-evidence-id",
        action="append",
        default=[],
        help=(
            "Repeat for each human-assisted Planning edit target selected "
            "from the complete change-support evidence pool."
        ),
    )

    intelligent_evidence_parser = subparsers.add_parser(
        "intelligent-evidence-request",
        help="Let the Evidence specialist select read-only assembly tools and stop after evidence.",
    )
    intelligent_evidence_parser.add_argument(
        "description", help="Natural-language evidence request."
    )
    intelligent_evidence_parser.add_argument("--source-registry", required=True)
    intelligent_evidence_parser.add_argument(
        "--artifact-root", default="artifacts/intelligent_evidence_only"
    )
    intelligent_evidence_parser.add_argument("--settings", default="config/settings.yaml")
    intelligent_evidence_parser.add_argument("--target-domain", default="gtfo")
    intelligent_evidence_parser.add_argument("--target-mod", default="WallCrawler")
    intelligent_evidence_parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    intelligent_evidence_parser.add_argument(
        "--base-url-env", default="DASHSCOPE_BASE_URL"
    )
    intelligent_evidence_parser.add_argument(
        "--allow-evidence-disclosure",
        action="store_true",
        help=(
            "Approve this run to send the natural-language request, path-hidden source "
            "views, and metadata/IL tool observations to the configured model."
        ),
    )
    intelligent_evidence_parser.add_argument(
        "--local-ollama",
        action="store_true",
        help=(
            "Use Ollama through a literal loopback endpoint; no external model "
            "disclosure is authorized."
        ),
    )
    intelligent_evidence_parser.add_argument(
        "--ollama-base-url", default="http://127.0.0.1:11434"
    )
    intelligent_evidence_parser.add_argument(
        "--ollama-model", default="qwen3:4b-instruct"
    )
    intelligent_evidence_parser.add_argument(
        "--ollama-max-output-tokens", type=int, default=4096
    )
    intelligent_evidence_parser.add_argument(
        "--ollama-context-window", type=int
    )
    intelligent_evidence_parser.add_argument(
        "--ollama-request-timeout-seconds", type=float, default=180.0
    )

    apply_parser = subparsers.add_parser(
        "apply-patch",
        help="Apply a reviewed exact diff in isolation, or explicitly continue its checkpoint through Build.",
    )
    apply_parser.add_argument("--confirm", action="store_true", help="Request confirmed apply.")
    for name in ("artifact-store-root", "workspace-root", "feature-id", "run-id", "isolated-root"):
        apply_parser.add_argument("--" + name)
    apply_mode = apply_parser.add_mutually_exclusive_group()
    apply_mode.add_argument("--human-decision-stdin", action="store_true",
        help="Consume the displayed binding and existing human semantic decision from JSON stdin.")
    apply_mode.add_argument("--continue-build", action="store_true",
        help="Consume the existing Apply checkpoint and ResumePayload for a separately authorized Build; does not repeat Apply.")
    apply_mode.add_argument("--resume-approval", action="store_true",
        help="Resume a deferred product diff in its existing isolated workspace using the original checkpoint and exact human decision.")
    apply_parser.add_argument("--checkpoint-path")
    apply_parser.add_argument("--checkpoint-hash")
    apply_parser.add_argument("--resume-payload", help="JSON file containing the existing exact ResumePayload fields.")
    apply_parser.add_argument("--build-profile", help="JSON file containing the existing BuildProfile fields; its working_directory must be this Apply's isolated workspace.")
    apply_parser.add_argument("--human-semantic-decision", help="JSON file containing the existing exact human_semantic_decision_v1 decision.")
    apply_parser.add_argument("--v2-delivery-checkpoint", help="Exact v2 delivery checkpoint, or Apply receipt for the build action.")
    apply_parser.add_argument("--v2-delivery-action", choices=("prepare", "apply", "build"), default="prepare")
    apply_parser.add_argument("--v2-displayed-binding", help="JSON file containing the exact v2 displayed_binding returned by prepare.")
    apply_parser.add_argument("--task-delegation-ref", help="Existing host-authorized v2 task delegation reference; never a model approval.")
    apply_parser.add_argument("--operation-id", help="Stable v2 delivery operation identity for retries and reconciliation.")
    apply_parser.add_argument("--allow-build", action="store_true",
        help="Explicitly authorize this Build continuation only; Runtime needs its own existing authorization.")
    _add_execution_trace_arguments(apply_parser)

    repository_qualification = subparsers.add_parser("qualify-repository-evaluator",
        help="Validate retained independent SDK-project control reports; does not execute code or models.")
    repository_qualification.add_argument("--input-record", required=True)
    repository_qualification.add_argument("--output-root", required=True)
    repository_verdict = subparsers.add_parser("evaluate-repository-strategy",
        help="Verify a retained independent repository report against native candidate evidence.")
    repository_verdict.add_argument("--evaluator-record", required=True)
    repository_verdict.add_argument("--execution-record", required=True)
    repository_verdict.add_argument("--evidence-record", required=True)
    repository_verdict.add_argument("--output", required=True)

    strategy_export_parser = subparsers.add_parser("export-strategy-execution",
        help="Export retained native Task Agent evidence for strategy evaluation without running a model.")
    strategy_export_parser.add_argument("--execution-record", required=True, help="Original native execution.json.")
    strategy_export_parser.add_argument("--call-id", required=True, help="Actual completed decision/tool call to examine.")
    strategy_export_parser.add_argument("--output-root", required=True)
    strategy_export_parser.add_argument("--artifact-store-root")
    strategy_export_parser.add_argument("--protocol-record", help="Legacy diagnostic protocol; cannot establish pre-run freezing.")
    strategy_export_parser.add_argument("--task-id", help="Legacy diagnostic label; pre-bound trials derive this from native evidence.")
    strategy_export_parser.add_argument("--arm", choices=("B", "B+"))
    strategy_export_parser.add_argument("--attempt", type=int)
    strategy_export_parser.add_argument("--learning-receipt", help="Original terminal learning receipt, if retained.")
    strategy_export_parser.add_argument("--delivery-receipt", help="Exact completed v2 delivery receipt reference, if available.")

    run_parser = subparsers.add_parser(
        "run-request",
        help="Run one controlled request against an isolated copy of a real target project.",
    )
    run_parser.add_argument("description", nargs="?", help="Natural-language request; omitted only for an exact v2 resume.")
    run_parser.add_argument("--target-project")
    run_parser.add_argument("--resume-task-agent", help="Exact persisted v2 task checkpoint reference; restores the original task and resources.")
    run_parser.add_argument("--artifact-store-root", help="Original store for --resume-task-agent; never a new task store.")
    run_parser.add_argument("--repository-profile",
        help="Explicit SDK-style C# repository build/test profile; keeps the existing product workflow and disables game scope.")
    run_parser.add_argument("--game-folder-build-binding", help="JSON with the existing exact read-only GameFolder binding.")
    run_parser.add_argument("--external-patch-profile", help="Explicit existing external Qwen worker profile JSON.")
    run_parser.add_argument("--run-id", help="Explicit identity for one new product invocation.")
    run_parser.add_argument("--request-id", help="Logical request identity; supplied together with --run-id.")
    run_parser.add_argument(
        "--workspace-root",
        default="workspaces/v2_codegen_real_001",
    )
    run_parser.add_argument("--settings", default="config/settings.yaml")
    run_parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    run_parser.add_argument("--base-url-env", default="DASHSCOPE_BASE_URL")
    run_parser.add_argument(
        "--local-ollama",
        action="store_true",
        help=(
            "Use Ollama through a literal loopback endpoint for this run; "
            "no external source disclosure is authorized."
        ),
    )
    run_parser.add_argument(
        "--ollama-base-url", default="http://127.0.0.1:11434"
    )
    run_parser.add_argument("--ollama-model")
    run_parser.add_argument("--ollama-timeout-seconds", type=float)
    run_parser.add_argument(
        "--ollama-max-output-tokens", type=int, default=4096
    )
    run_parser.add_argument("--ollama-context-window", type=int)
    run_parser.add_argument("--source-retrieval", choices=("rg_live", "indexed_source"), default="rg_live",
        help="Select current-source retrieval for initial Evidence and every reassessment; rg_live remains the default.")
    run_parser.add_argument("--reuse-query-observations", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse identical source-bound queries within a batch; disable for a recorded online comparison.")
    run_parser.add_argument("--compress-findings-input", action=argparse.BooleanOptionalAction, default=True,
        help="Use reversible Findings input text references; disable for a recorded online comparison.")
    run_parser.add_argument("--patch-worker-config",
        help="Explicit existing PatchWorkerConfiguration JSON for the shared worker and optional evaluator.")
    run_parser.add_argument("--defer-approval", action="store_true",
        help="Persist the completed exact Dry-run and pause before asking for or performing Apply; resume through apply-patch --resume-approval.")
    run_parser.add_argument("--task-model-call-limit", type=int,
        help="Positive total model-call limit shared across this product task.")
    run_parser.add_argument("--model-call-budget-policy", choices=("task_total_only", "stage_and_total", "meter_only"),
        default="task_total_only", help="meter_only counts resources without quotas; other policies reproduce bounded runs.")
    run_parser.add_argument("--technical-executor", choices=("workflow", "task_agent"), default="workflow",
        help="Explicit experimental technical executor; workflow remains the default.")
    run_parser.add_argument("--constraint-policy", choices=("source_priority_v1",),
        help="Separate model-derived preservation hints from user and platform constraints.")
    run_parser.add_argument("--task-agent-edit-paths",
        help="JSON list of operator-authorized project-relative existing files; required only for task_agent.")
    run_parser.add_argument("--task-agent-candidate-mode", choices=("baseline", "current"), default="baseline",
        help="Experimental task_agent revision base; baseline preserves the existing candidate protocol.")
    run_parser.add_argument("--task-agent-protocol", choices=("v1", "v2"), default="v1",
        help="Explicit task-agent protocol; v2 separates authority, technical plans and file bindings.")
    run_parser.add_argument("--task-agent-experience-config", help="Hash-bound host strategy snapshot and terminal learning configuration for v2.")
    run_parser.add_argument("--task-agent-semantic-state", choices=("enabled", "disabled"), default="enabled",
        help="Maintain model-authored evidence and hypotheses alongside normal v2 actions; disabled retains the legacy decision view.")
    run_parser.add_argument("--stage-model-call-limits",
        help="JSON object with the existing requirement_projection, requirement_findings, planning, initial_patch and repair call limits.")
    run_parser.add_argument("--dependency-api-config", help="Explicit hash-bound metadata/IL inspector configuration for the run-local GameFolder copy.")
    run_parser.add_argument("--knowledge-evidence-config", help="Explicit source/hash/budget-bound read-only MCP document knowledge configuration.")
    run_parser.add_argument("--resume-initial-findings", help="Exact original execution configuration for one same-task Findings continuation.")
    run_parser.add_argument("--findings-stage-allocation", help="Separate reviewed stage-allocation/time configuration; original limits stay unchanged.")
    run_parser.add_argument("--findings-approval-stdin", action="store_true")
    _add_execution_trace_arguments(run_parser)
    run_parser.add_argument(
        "--allow-source-disclosure",
        action="store_true",
        help=(
            "Explicitly allow this run to send retrieved local source excerpts "
            "and binding context to the configured external LLM."
        ),
    )
    run_parser.add_argument("--confirm-isolated-write", action="store_true")
    run_parser.add_argument("--allow-build", action="store_true")
    run_parser.add_argument(
        "--planning-target-evidence-id",
        action="append",
        default=[],
        help=(
            "Repeat for each precomputed human-assisted Planning edit target "
            "from an already-known current Fresh EvidencePack identity."
        ),
    )
    run_parser.add_argument(
        "--run-gtfo-runtime",
        action="store_true",
        help=(
            "Continue the admitted run after Build through configured GTFO Runtime "
            "Verification, Evaluation, and Terminal Record."
        ),
    )
    run_parser.add_argument(
        "--allow-game-launch",
        action="store_true",
        help=(
            "Authorize this run only to launch GTFO through the configured Runtime "
            "profile; requires --run-gtfo-runtime."
        ),
    )
    run_parser.add_argument(
        "--runtime-plugin-destination",
        help=(
            "Absolute game-directory DLL path for the verified final Build; "
            "requires Runtime and game-directory write approval."
        ),
    )
    run_parser.add_argument(
        "--allow-game-directory-write",
        action="store_true",
        help=(
            "Authorize this run only to deploy the verified final DLL and restore "
            "the destination after Runtime capture."
        ),
    )
    resume_runtime_parser = subparsers.add_parser(
        "resume-runtime",
        help=(
            "Continue a validated successful Build checkpoint through the "
            "existing Runtime, Evaluation, and Terminal graph path."
        ),
    )
    resume_runtime_parser.add_argument("--run-root", required=True)
    resume_runtime_parser.add_argument("--artifact-store-root")
    resume_runtime_parser.add_argument("--feature-id", required=True)
    resume_runtime_parser.add_argument("--checkpoint-path", required=True)
    resume_runtime_parser.add_argument("--checkpoint-hash", required=True)
    resume_runtime_parser.add_argument("--resume-id")
    resume_runtime_parser.add_argument(
        "--recover-accepted-resume",
        action="store_true",
        help=(
            "Reuse the exact accepted ResumeReport for a bounded pre-Runtime "
            "recovery; requires --resume-id and creates no second ResumeReport."
        ),
    )
    resume_runtime_parser.add_argument(
        "--retry-runtime-observation",
        action="store_true",
        help=(
            "Reuse the exact accepted ResumeReport only when the latest "
            "RuntimeEvidence and RepairPlan require a bounded observation retry."
        ),
    )
    resume_runtime_parser.add_argument(
        "--runtime-output-root",
        help=(
            "Absolute empty run-local output directory required for an initial "
            "Stagger repair Runtime and for --retry-runtime-observation."
        ),
    )
    resume_runtime_parser.add_argument("--approved-by", required=True)
    resume_runtime_parser.add_argument("--settings", default="config/settings.yaml")
    resume_runtime_parser.add_argument("--allow-game-launch", action="store_true")
    resume_runtime_parser.add_argument(
        "--runtime-plugin-destination",
        required=True,
    )
    resume_runtime_parser.add_argument(
        "--allow-game-directory-write",
        action="store_true",
    )
    preflight_runtime_parser = subparsers.add_parser(
        "preflight-runtime",
        help=(
            "Read-validate a Build-to-Runtime continuation without writing, "
            "consuming the checkpoint, entering the graph, copying a DLL, or "
            "launching a process."
        ),
    )
    preflight_runtime_parser.add_argument("--run-root", required=True)
    preflight_runtime_parser.add_argument("--artifact-store-root")
    preflight_runtime_parser.add_argument("--feature-id", required=True)
    preflight_runtime_parser.add_argument("--checkpoint-path", required=True)
    preflight_runtime_parser.add_argument("--checkpoint-hash", required=True)
    preflight_runtime_parser.add_argument("--resume-id")
    preflight_runtime_parser.add_argument(
        "--recover-accepted-resume",
        action="store_true",
        help=(
            "Read-validate the exact accepted ResumeReport recovery boundary "
            "without writing or launching."
        ),
    )
    preflight_runtime_parser.add_argument(
        "--retry-runtime-observation",
        action="store_true",
        help=(
            "Read-validate an observation-owned Runtime retry without writing "
            "or launching."
        ),
    )
    preflight_runtime_parser.add_argument(
        "--runtime-output-root",
        help=(
            "Absolute empty run-local output directory required for an initial "
            "Stagger repair Runtime and for --retry-runtime-observation."
        ),
    )
    preflight_runtime_parser.add_argument("--settings")
    preflight_runtime_parser.add_argument("--runtime-plugin-destination")
    stagger_edit_authority_parser = subparsers.add_parser(
        "preflight-stagger-edit-authority",
        help=(
            "Read-validate the current Stagger Runtime failure and emit an "
            "exact no-write Edit Authority proposal."
        ),
    )
    _add_stagger_edit_authority_arguments(
        stagger_edit_authority_parser
    )
    request_stagger_edit_authority_parser = subparsers.add_parser(
        "request-stagger-edit-authority",
        help=(
            "Persist one exact technical request without granting Edit "
            "Authority or mutating the Registry."
        ),
    )
    _add_stagger_edit_authority_arguments(
        request_stagger_edit_authority_parser
    )
    preflight_existing_binding_parser = subparsers.add_parser(
        "preflight-stagger-existing-binding-reauthorization",
        help=(
            "Read-validate the exact existing Stagger repair binding and "
            "emit a no-write reauthorization request payload."
        ),
    )
    _add_stagger_existing_binding_reauthorization_arguments(
        preflight_existing_binding_parser
    )
    request_existing_binding_parser = subparsers.add_parser(
        "request-stagger-existing-binding-reauthorization",
        help=(
            "Persist one no-authority request for the exact existing "
            "Stagger repair binding."
        ),
    )
    _add_stagger_existing_binding_reauthorization_arguments(
        request_existing_binding_parser
    )
    preflight_observation_authority_parser = subparsers.add_parser(
        "preflight-stagger-hud-parent-observation-authority",
        help=(
            "Read-validate the complete bounded Stagger HUD parent "
            "diagnostic Plan and emit a no-write authority proposal."
        ),
    )
    _add_stagger_observation_authority_arguments(
        preflight_observation_authority_parser
    )
    request_observation_authority_parser = subparsers.add_parser(
        "request-stagger-hud-parent-observation-authority",
        help=(
            "Persist one no-authority Request for the complete bounded "
            "Stagger HUD parent diagnostic Plan."
        ),
    )
    _add_stagger_observation_authority_arguments(
        request_observation_authority_parser
    )
    stagger_diagnostic_diff_parser = subparsers.add_parser(
        "preflight-stagger-hud-parent-diagnostic-exact-diff",
        help=(
            "Generate the exact four-change Stagger HUD diagnostic diff "
            "without writing source or Artifacts."
        ),
    )
    _add_stagger_diagnostic_diff_arguments(
        stagger_diagnostic_diff_parser
    )
    preflight_stagger_review_parser = subparsers.add_parser(
        "preflight-stagger-edit-authority-review",
        help=(
            "Enter the exact Graph review interrupt without writing a "
            "checkpoint, ResumeReport, Approval, or BindingResolution."
        ),
    )
    preflight_stagger_review_parser.add_argument(
        "--artifact-store-root",
        required=True,
    )
    preflight_stagger_review_parser.add_argument(
        "--feature-id",
        required=True,
    )
    preflight_stagger_review_parser.add_argument(
        "--request-path",
        required=True,
    )
    preflight_stagger_review_parser.add_argument(
        "--request-artifact-hash",
        required=True,
    )
    preflight_stagger_review_parser.add_argument(
        "--authority-request-hash",
        required=True,
    )
    resume_stagger_edit_authority_parser = subparsers.add_parser(
        "resume-stagger-edit-authority-review",
        help=(
            "Carry one exact human decision through the existing LangGraph "
            "Edit Authority interrupt/resume gate."
        ),
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--artifact-store-root",
        required=True,
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--feature-id",
        required=True,
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--request-path",
        required=True,
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--request-artifact-hash",
        required=True,
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--authority-request-hash",
        required=True,
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--resume-id",
        required=True,
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--decision",
        required=True,
        choices=("approve", "reject"),
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--approved-by",
        required=True,
    )
    resume_stagger_edit_authority_parser.add_argument(
        "--approved-at",
        required=True,
    )

    return parser


_VERIFICATION_OPERATION_EXIT_CODES = {
    "succeeded": 0,
    "domain_blocked": 1,
    "operator_input_required": 1,
    "platform_blocked": 2,
    "partial": 2,
}

_LEGACY_ADVISORY_READY_STATUSES = {
    "READY_FOR_RUNTIME_PROFILE",
    "READY_FOR_EXTERNAL_AUTHORIZATION",
}


def _verification_operation_exit_code(result: object) -> int:
    """Map one public verification result to its declaration-owned exit."""

    if not isinstance(result, Mapping):
        return 2
    operation_result = result.get("verification_operation_result")
    if not isinstance(operation_result, Mapping):
        return 2
    disposition = operation_result.get("execution_disposition")
    if not isinstance(disposition, str):
        return 2
    return _VERIFICATION_OPERATION_EXIT_CODES.get(disposition, 2)


def _verification_preflight_exit_code(result: object) -> int:
    """Map one execution-grade preflight result."""

    if not isinstance(result, Mapping):
        return 2
    operation_result = result.get("verification_operation_preflight_result")
    if not isinstance(operation_result, Mapping):
        return 2
    return 0 if operation_result.get("preflight_status") == "ready" else 2


def _legacy_advisory_preflight_exit_code(result: object) -> int:
    """Preserve the exact CLI-only legacy advisory mapping boundary."""

    if not isinstance(result, Mapping):
        return 2
    if (
        "verification_operation_preflight_result" in result
        or "verification_operation_result" in result
    ):
        return 2
    return 0 if result.get("status") in _LEGACY_ADVISORY_READY_STATUSES else 2


def _edit_authority_review_exit_code(result: object) -> int:
    """Project the declaration-owned Edit Authority result to CLI exit status."""

    if not isinstance(result, Mapping):
        return 2
    projection = result.get("edit_authority_review_result")
    feature_id = result.get("feature_id")
    approval_identity_hash = result.get("request_hash")
    if (
        not isinstance(projection, Mapping)
        or type(feature_id) is not str
        or not feature_id
        or type(approval_identity_hash) is not str
        or not approval_identity_hash
    ):
        return 2
    try:
        machine_result = parse_current_edit_authority_review_result(
            projection,
            feature_id=feature_id,
            approval_identity_hash=approval_identity_hash,
        )
    except ValueError:
        return 2
    return (
        0
        if machine_result.platform_status == "completed"
        and machine_result.review_outcome == "approved"
        and machine_result.materialization_status == "completed"
        else 2
    )


def _run_preflight_runtime_command(args: argparse.Namespace) -> int:
    """Prepare one legacy Runtime preflight call and present its public result."""

    try:
        result = preflight_local_runtime(
            run_root=args.run_root,
            artifact_store_root=args.artifact_store_root,
            feature_id=args.feature_id,
            checkpoint_path=args.checkpoint_path,
            checkpoint_hash=args.checkpoint_hash,
            resume_id=args.resume_id,
            recover_accepted_resume=bool(args.recover_accepted_resume),
            retry_runtime_observation=bool(args.retry_runtime_observation),
            runtime_output_root=args.runtime_output_root,
            settings_path=args.settings,
            runtime_plugin_destination=args.runtime_plugin_destination,
        )
    except LocalEntryError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "command": "preflight-runtime",
                    "reason": str(exc),
                    "writes_performed": False,
                    "graph_invoked": False,
                    "checkpoint_consumed": False,
                    "resume_report_created": False,
                    "runtime_evidence_created": False,
                    "process_launched": False,
                    "dll_copied": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return _legacy_advisory_preflight_exit_code(result)


def _run_resume_runtime_command(args: argparse.Namespace) -> int:
    """Prepare one legacy Runtime execution call and present its public result."""

    try:
        result = resume_local_runtime(
            run_root=args.run_root,
            artifact_store_root=args.artifact_store_root,
            feature_id=args.feature_id,
            checkpoint_path=args.checkpoint_path,
            checkpoint_hash=args.checkpoint_hash,
            resume_id=args.resume_id,
            recover_accepted_resume=bool(args.recover_accepted_resume),
            retry_runtime_observation=bool(args.retry_runtime_observation),
            runtime_output_root=args.runtime_output_root,
            approved_by=args.approved_by,
            settings_path=args.settings,
            allow_game_launch=bool(args.allow_game_launch),
            runtime_plugin_destination=args.runtime_plugin_destination,
            allow_game_directory_write=bool(args.allow_game_directory_write),
        )
    except LocalEntryError as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "command": "resume-runtime",
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return _verification_operation_exit_code(result)


def _explicit_preflight_request_from_args(
    args: argparse.Namespace,
) -> SealedPreflightPreparationRequestV1:
    return SealedPreflightPreparationRequestV1(
        contract_version="1",
        packet_root=args.packet_root,
        packet_date=args.packet_date,
        task_id=args.task_id,
        request_text=args.request_text,
        identity_series=args.identity_series,
        identity_sequence=args.identity_sequence,
        selected_configuration=SelectedPreflightConfigurationV1(
            configuration_id=args.configuration_id,
            route=args.route,
            main_ollama_timeout_seconds=args.main_ollama_timeout_seconds,
            timeout_owner=args.timeout_owner,
            repair_ollama_timeout_seconds=args.repair_ollama_timeout_seconds,
            repair_timeout_selection=args.repair_timeout_selection,
            projection_contract_checkpoint=args.projection_contract_checkpoint,
            model_projection_fields=tuple(args.model_projection_field),
            model_relations_authorized=args.model_relations_authorized,
            deterministic_relation_completion=(
                args.deterministic_relation_completion
            ),
            deterministic_relation_rule=args.deterministic_relation_rule,
            deterministic_post_change=DeterministicPostChangeConditionsV1(
                only_unaccounted=(
                    args.deterministic_post_change_only_unaccounted
                ),
                required_before_patch_policy=(
                    args.deterministic_post_change_required_before_patch_policy
                ),
                allowed_dispositions_exact=tuple(
                    args.deterministic_post_change_allowed_disposition
                ),
                all_other_omissions=(
                    args.deterministic_post_change_all_other_omissions
                ),
            ),
            initial_projection_attempts=args.initial_projection_attempts,
            maximum_correction_attempts_after_initial=(
                args.maximum_correction_attempts_after_initial
            ),
            provider_fallback=args.provider_fallback,
            automatic_retry=args.automatic_retry,
            external_disclosure=args.external_disclosure,
        ),
        platform_root=args.platform_root,
        allowed_dirty_paths=tuple(args.allowed_dirty_path),
        source_root=args.source_root,
        target_project=args.target_project,
        project_file=args.project_file,
        primary_source_paths=tuple(args.primary_source),
        git_executable=args.git_executable,
        ollama_base_url=args.ollama_base_url,
        ollama_model=args.ollama_model,
        ollama_context_window=args.ollama_context_window,
        ollama_output_budget=args.ollama_output_budget,
        preflight_lineage_packet_paths=tuple(args.preflight_lineage_packet),
        predecessor_receipt_path=args.predecessor_receipt,
        predecessor_receipt_manifest_path=args.predecessor_receipt_manifest,
        actual_lineage_packet_paths=tuple(args.actual_lineage_packet),
        rejected_preflight_identity_reservations=tuple(
            RejectedPreflightIdentityReservationV1(
                packet_path=values[0],
                receipt_path=values[1],
                receipt_manifest_path=values[2],
            )
            for values in args.rejected_preflight_identity_reservation
        ),
    )


def _write_fresh_coverage_progress(event: dict[str, object]) -> None:
    print(
        json.dumps(event, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local controlled-development CLI."""

    # JSON results and exact diff displays must preserve every source character,
    # including when redirected under a Windows legacy console encoding.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")

    selected_argv = tuple(sys.argv[1:] if argv is None else argv)
    prepare_continuation = (
        "prepare-sealed-entry-preflight" in selected_argv
        and "--run-family-continuation-profile" in selected_argv
    )
    parser = build_parser(_prepare_continuation=prepare_continuation)
    args = parser.parse_args(selected_argv)

    # Scope: opt-in host resource grant, never an authorization to build or edit.
    # 为什么要有它：普通CLI与冻结入口必须遵守同一构建总额，不能绕过父进程批准的资源上限。
    from src.mod_agent.build.execution_budget import BuildBudgetError, install_build_budget_from_environment
    try:
        install_build_budget_from_environment()
    except (BuildBudgetError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "command": args.command,
            "code": str(error) if isinstance(error, BuildBudgetError) else "BUILD_BUDGET_CONFIGURATION_NOT_ADMITTED",
            "process_launched": False, "source_modified": False}, ensure_ascii=False))
        return 2

    if args.command == "fresh-coverage":
        fresh_coverage_started = time.perf_counter()
        progress = FreshCoverageProgressReporter(_write_fresh_coverage_progress)
        progress.start()
        result = None
        try:
            if not args.local_ollama:
                raise FreshCoverageError(
                    "FRESH_COVERAGE_MODEL_BOUNDARY_NOT_AUTHORIZED",
                    "model_admission",
                    "fresh-coverage requires explicit --local-ollama.",
                )
            api_symbols = tuple(args.api_symbol)
            api_source_pair_complete = (
                args.api_source_registry is not None
                and args.api_source_id is not None
            )
            if args.api_selection_mode is None and (
                args.api_source_registry is not None
                or args.api_source_id is not None
                or api_symbols
            ):
                raise FreshCoverageError(
                    "FRESH_COVERAGE_API_SELECTION_MODE_REQUIRED",
                    "request",
                    "API augmentation requires explicit --api-selection-mode.",
                )
            if args.api_selection_mode is not None and not api_source_pair_complete:
                raise FreshCoverageError(
                    "FRESH_COVERAGE_API_SOURCE_SELECTION_INCOMPLETE",
                    "request",
                    "--api-source-registry and --api-source-id must be provided together.",
                )
            if args.api_selection_mode == "autonomous" and api_symbols:
                raise FreshCoverageError(
                    "FRESH_COVERAGE_AUTONOMOUS_API_SYMBOL_FORBIDDEN",
                    "request",
                    "autonomous API selection does not accept --api-symbol.",
                )
            if args.api_selection_mode == "assisted_explicit" and (
                not api_symbols
                or len(api_symbols) > 50
                or any(
                    not symbol or symbol != symbol.strip()
                    for symbol in api_symbols
                )
                or len(api_symbols) != len(set(api_symbols))
            ):
                raise FreshCoverageError(
                    "FRESH_COVERAGE_ASSISTED_API_SYMBOL_SELECTION_INVALID",
                    "request",
                    "assisted_explicit requires one to fifty unique exact --api-symbol values.",
                )
            feature_data = load_feature_spec_payload(args.feature_spec)
            completion = create_local_ollama_requirement_coverage_completion(
                settings_path=args.settings,
                ollama_base_url=args.ollama_base_url,
                ollama_model=args.ollama_model,
                ollama_context_window=args.ollama_context_window,
                ollama_output_budget=args.ollama_output_budget,
                ollama_timeout_seconds=args.ollama_timeout_seconds,
                stream_activity_recorder=progress.record_stream_activity,
            )
            source_augmentation_factory = None
            if args.api_selection_mode == "autonomous":
                source_augmentation_factory = (
                    make_fresh_api_source_augmentation_factory(
                        request=str(feature_data["user_request"]),
                        source_registry_path=args.api_source_registry,
                        source_id=args.api_source_id,
                        workspace_root=args.workspace_root,
                        search_root=args.search_root,
                        ollama_base_url=completion.base_url,
                        ollama_model=completion.model,
                        ollama_max_output_tokens=completion.output_budget,
                        ollama_context_window=completion.context_window,
                        ollama_request_timeout_seconds=completion.llm.timeout,
                    )
                )
            elif args.api_selection_mode == "assisted_explicit":
                source_augmentation_factory = (
                    make_fresh_assisted_api_source_augmentation_factory(
                        request=str(feature_data["user_request"]),
                        source_registry_path=args.api_source_registry,
                        source_id=args.api_source_id,
                        api_symbols=api_symbols,
                    )
                )
            result = run_fresh_requirement_coverage(
                feature_data=feature_data,
                workspace_root=args.workspace_root,
                search_root=args.search_root,
                artifact_root=args.artifact_root,
                completion=completion,
                progress_reporter=progress,
                source_augmentation_factory=source_augmentation_factory,
            )
        except (FreshCoverageError, OSError, RuntimeError, ValueError) as exc:
            failure_code = getattr(
                exc,
                "code",
                "FRESH_COVERAGE_EXECUTION_FAILED",
            )
            failure_stage = getattr(exc, "stage", "requirement_coverage")
            result = blocked_fresh_coverage_result(
                code=failure_code,
                stage=failure_stage,
                reason=failure_code,
                duration_seconds=time.perf_counter() - fresh_coverage_started,
            )
        except BaseException:
            progress.finish(
                status="failed",
                terminal_stage="execution",
            )
            raise
        assert result is not None
        progress.stop_heartbeat()
        try:
            print(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        except BaseException:
            progress.finish(status="failed", terminal_stage="result_output")
            raise
        progress.finish(
            status=result.status,
            terminal_stage=result.terminal_stage,
        )
        if result.status == "terminal":
            return 0
        return 1 if result.evidence_pack_artifact is not None else 2

    if args.command == "continue-pretransport-patch":
        from src.mod_agent.interrupted_patch_continuation import prepare_interrupted_patch, resume_interrupted_patch
        try:
            kwargs = dict(run_root=args.run_root, run_id=args.run_id, execution_config=args.execution_config,
                recovery_config=read_json_object(args.recovery_config), pretransport=True)
            if args.prepare_only:
                prepared = prepare_interrupted_patch(**kwargs)
                result = {"status": "ready", "scope": prepared["scope"], "scope_sha256": prepared["scope_sha256"],
                    "model_calls_performed": 0, "apply_calls_performed": 0}
            else:
                result = resume_interrupted_patch(**kwargs)
        except (ValueError, OSError, KeyError, TypeError) as error:
            result = {"status": "blocked", "reason": str(error)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") in {"ready", "awaiting_approval"} else 2

    if args.command == "continue-fresh-planning":
        continuation_started = time.perf_counter()
        try:
            request = PlanningContinuationInputV1(
                contract_version=PLANNING_CONTINUATION_CONTRACT_VERSION,
                artifact_store_root=args.artifact_root,
                workspace_root=args.workspace_root,
                feature_spec_ref=ArtifactRef(
                    feature_id=args.feature_id,
                    artifact_type="feature_spec",
                    path=args.feature_spec_path,
                    artifact_hash=args.feature_spec_hash,
                ),
                evidence_pack_ref=ArtifactRef(
                    feature_id=args.feature_id,
                    artifact_type="evidence_pack",
                    path=args.evidence_pack_path,
                    artifact_hash=args.evidence_pack_hash,
                ),
                planning_target_evidence_ids=tuple(
                    args.planning_target_evidence_id
                ),
            )
            result = run_fresh_planning_continuation(request)
        except PlanningContinuationError as exc:
            result = blocked_planning_continuation_result(
                code=exc.code,
                stage=exc.stage,
                reason=exc.code,
                duration_seconds=(
                    time.perf_counter() - continuation_started
                ),
            )
        print(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if result.status == "terminal":
            return 0
        if (
            result.status == "blocked"
            and result.implementation_plan_artifact is not None
        ):
            return 1
        return 2

    if args.command == "continue-fresh-patch":
        if args.reassemble_saved_patch is not None:
            return _reassemble_saved_patch_from_cli(args)
        continuation_started = time.perf_counter()
        request: PatchContinuationInputV1 | None = None
        try:
            worker_config = (
                PatchWorkerConfiguration.load(args.patch_worker_config)
                if args.patch_worker_config else None
            )
            upstream_configuration = (
                PatchUpstreamConfiguration.load(args.patch_upstream_config)
                if args.patch_upstream_config else None
            )
            repair_inputs = _read_patch_repair_inputs(args.repair_inputs) if args.repair_inputs else {}
            recovery_selection = read_json_object(args.repair_from_verification) if args.repair_from_verification else None
            if recovery_selection is not None and set(recovery_selection) != {"source_attempt_ref", "verification_result_ref"}:
                raise PatchContinuationError(
                    "PATCH_REPAIR_SELECTION_INVALID", "request",
                    "Automatic binding requires exactly source_attempt_ref and verification_result_ref.",
                )
            if (repair_inputs or recovery_selection is not None) and (worker_config is None or not worker_config.enable_recovery_loop):
                raise PatchContinuationError(
                    "PATCH_REPAIR_WORKER_CONFIGURATION_REQUIRED", "request",
                    "Repair inputs require a worker configuration with a behavior evaluator and recovery enabled.",
                )
            if args.attempt_id is not None and worker_config is None:
                raise PatchContinuationError(
                    "PATCH_WORKER_CONFIGURATION_REQUIRED", "request",
                    "A named CLI attempt requires an explicit worker configuration.",
                )
            qualified_selection = None
            if args.qualified_model_routing:
                if worker_config is None:
                    raise ModelQualificationError("MODEL_QUALIFICATION_PENDING", "Qualified routing requires explicit worker configuration.", status="pending")
                selected_feature = ArtifactStore(args.artifact_root).read(ArtifactRef(
                    feature_id=args.feature_id, artifact_type="feature_spec",
                    path=args.feature_spec_path, artifact_hash=args.feature_spec_hash,
                ))
                qualified_selection = resolve_qualified_patch_profile(
                    routing_path=args.qualified_model_routing,
                    role="repair" if repair_inputs or recovery_selection is not None else "initial_patch",
                    target_mod=selected_feature["data"].get("target_mod"), worker_policy=worker_config.policy,
                )
                profile = qualified_selection.profile
                explicit_values = {
                    "base_url": args.ollama_base_url, "model": args.ollama_model,
                    "context_window": args.ollama_context_window, "output_budget": args.ollama_output_budget,
                    "timeout_seconds": args.ollama_timeout_seconds,
                }
                if any(value is not None and value != getattr(profile, name) for name, value in explicit_values.items()):
                    raise ModelQualificationError("MODEL_QUALIFIED_PROFILE_ARGUMENT_CONFLICT", "Explicit model arguments disagree with the qualified role profile.")
            else:
                profile = LocalOllamaInitialPatchProfileV1(
                    profile_version=LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
                    base_url=args.ollama_base_url if args.ollama_base_url is not None else "http://127.0.0.1:11434",
                    model=args.ollama_model, context_window=args.ollama_context_window,
                    output_budget=args.ollama_output_budget if args.ollama_output_budget is not None else 4096,
                    timeout_seconds=args.ollama_timeout_seconds,
                )
            if upstream_configuration is not None:
                if not repair_inputs and recovery_selection is None:
                    raise PatchContinuationError(
                        "PATCH_UPSTREAM_FAILURE_REQUIRED", "request",
                        "Upstream return requires the existing exact Repair failure input.",
                    )
                upstream_configuration = upstream_configuration.with_worker(
                    profile=profile, worker_configuration=worker_config,
                    qualified_profile_selection=qualified_selection,
                )
            request = PatchContinuationInputV1(
                contract_version=PATCH_CONTINUATION_CONTRACT_VERSION,
                artifact_store_root=args.artifact_root,
                workspace_root=args.workspace_root,
                feature_spec_ref=ArtifactRef(
                    feature_id=args.feature_id,
                    artifact_type="feature_spec",
                    path=args.feature_spec_path,
                    artifact_hash=args.feature_spec_hash,
                ),
                evidence_pack_ref=ArtifactRef(
                    feature_id=args.feature_id,
                    artifact_type="evidence_pack",
                    path=args.evidence_pack_path,
                    artifact_hash=args.evidence_pack_hash,
                ),
                implementation_plan_ref=ArtifactRef(
                    feature_id=args.feature_id,
                    artifact_type="implementation_plan",
                    path=args.implementation_plan_path,
                    artifact_hash=args.implementation_plan_hash,
                ),
                **repair_inputs,
                attempt_id=args.attempt_id,
                worker_configuration_ref=worker_config.reference if worker_config else None,
                worker_profile_sha256=_canonical_sha256(profile.to_dict()) if worker_config else None,
                qualified_model_routing_ref=qualified_selection.routing_ref if qualified_selection else None,
                **({"patch_upstream_configuration_ref": upstream_configuration.reference}
                   if upstream_configuration is not None else {}),
            )
            if recovery_selection is not None:
                request = bind_patch_repair_verification(request=request, **recovery_selection)
            policy = (
                worker_config.policy if worker_config else
                PatchAgentLoopPolicy() if args.patch_agent_loop else None
            )
            configured_worker = {
                "behavior_evaluator": worker_config.behavior_evaluator(),
                "enable_recovery_loop": worker_config.enable_recovery_loop,
            } if worker_config else {}
            if qualified_selection is not None:
                configured_worker["qualified_profile_selection"] = qualified_selection
            trace_configuration = _execution_trace_configuration(args)
            store = ArtifactStore(args.artifact_root)
            result = run_fresh_patch_continuation(
                request,
                patch_change_factory=LocalOllamaInitialPatchAdapter(
                    profile=profile,
                    artifact_store=store,
                    workspace_root=args.workspace_root if policy is not None else None,
                    patch_agent_loop_policy=policy,
                    **configured_worker,
                ),
                dry_run_factory=make_exact_no_write_dry_run_factory(
                    args.workspace_root
                ),
                max_model_calls=policy.max_total_model_turns if policy is not None else 1,
                execution_trace_configuration=trace_configuration,
                **({"upstream_configuration": upstream_configuration}
                   if upstream_configuration is not None else {}),
            )
        except (PatchContinuationError, LocalEntryError, OSError, ValueError) as exc:
            result = blocked_patch_continuation_result(
                code=str(getattr(exc, "code", type(exc).__name__)),
                stage=str(getattr(exc, "stage", "request")),
                reason=str(getattr(exc, "reason", exc)),
                duration_seconds=time.perf_counter() - continuation_started,
                feature_id=(
                    None if request is None else request.feature_spec_ref.feature_id
                ),
                feature_spec_ref=(
                    None if request is None else request.feature_spec_ref
                ),
                evidence_pack_ref=(
                    None if request is None else request.evidence_pack_ref
                ),
                implementation_plan_ref=(
                    None if request is None else request.implementation_plan_ref
                ),
            )
        print(
            json.dumps(
                result.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        if result.status == "terminal":
            return 0
        if result.patch_proposal_artifact is not None or (
            result.attempt_receipt is not None
            and result.attempt_receipt.get("model_call_count") == 1
        ):
            return 1
        return 2

    if args.command == "plan-request":
        result = plan_request(
            args.description,
            target_game=args.target_game,
            target_mod=args.target_mod,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0

    if args.command == "prepare-sealed-entry-preflight":
        try:
            if args.run_family_continuation_profile:
                explicit_fields = (
                    "packet_date",
                    "task_id",
                    "request_text",
                    "identity_series",
                    "identity_sequence",
                    "configuration_id",
                    "route",
                    "main_ollama_timeout_seconds",
                    "timeout_owner",
                    "repair_ollama_timeout_seconds",
                    "repair_timeout_selection",
                    "projection_contract_checkpoint",
                    "model_projection_field",
                    "model_relations_authorized",
                    "deterministic_relation_completion",
                    "deterministic_relation_rule",
                    "deterministic_post_change_only_unaccounted",
                    "deterministic_post_change_required_before_patch_policy",
                    "deterministic_post_change_allowed_disposition",
                    "deterministic_post_change_all_other_omissions",
                    "initial_projection_attempts",
                    "maximum_correction_attempts_after_initial",
                    "provider_fallback",
                    "automatic_retry",
                    "external_disclosure",
                    "platform_root",
                    "allowed_dirty_path",
                    "source_root",
                    "target_project",
                    "project_file",
                    "primary_source",
                    "git_executable",
                    "ollama_base_url",
                    "ollama_model",
                    "ollama_context_window",
                    "ollama_output_budget",
                    "preflight_lineage_packet",
                    "predecessor_receipt",
                    "predecessor_receipt_manifest",
                    "actual_lineage_packet",
                    "rejected_preflight_identity_reservation",
                )
                supplied = [
                    name
                    for name in explicit_fields
                    if getattr(args, name) not in (None, [], ())
                ]
                if supplied:
                    raise ValueError(
                        "Continuation mode forbids explicit identity, lineage, "
                        "platform, source, model, and selected-configuration inputs: "
                        + ", ".join(supplied)
                    )
                profile_value = _read_typed_json_object_file(
                    args.run_family_continuation_profile,
                    "Run-family continuation profile",
                )
                request = RunFamilyContinuationRequestV1(
                    contract_version="1",
                    packet_root=args.packet_root,
                    profile=RunFamilyContinuationProfileV1.from_dict(
                        profile_value
                    ),
                )
            else:
                request = _explicit_preflight_request_from_args(args)
            result = prepare_sealed_entry_preflight(request)
        except PreflightPreparationError as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "command": "prepare-sealed-entry-preflight",
                        "stage": exc.stage,
                        "code": exc.code,
                        "reason": str(exc),
                        "diagnostics": exc.diagnostics,
                        "packet_root_created": False,
                        "sealed_files_written": 0,
                        "model_generation_calls": 0,
                        "ProductExecution_runs": 0,
                        "Builds": 0,
                        "Runtime_runs": 0,
                        "Apply_runs": 0,
                        "execution_root_created": False,
                        "apply_workspace_created": False,
                        "canary_authorized": False,
                        "actual_authorized": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "command": "prepare-sealed-entry-preflight",
                        "stage": "request",
                        "code": "PREPARATION_REQUEST_INVALID",
                        "reason": str(exc),
                        "packet_root_created": False,
                        "sealed_files_written": 0,
                        "model_generation_calls": 0,
                        "ProductExecution_runs": 0,
                        "Builds": 0,
                        "Runtime_runs": 0,
                        "Apply_runs": 0,
                        "execution_root_created": False,
                        "apply_workspace_created": False,
                        "canary_authorized": False,
                        "actual_authorized": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "prepare-sealed-actual":
        try:
            classification = tuple(
                sorted(
                    (
                        (name, _strict_cli_bool(value))
                        for name, value in args.completion_classification
                    ),
                    key=lambda item: item[0].encode("utf-8"),
                )
            )
            result = prepare_sealed_actual_run(
                ActualRunPreparationRequestV1(
                    contract_version="1",
                    accepted_preflight_packet_path=(
                        args.accepted_preflight_packet
                    ),
                    accepted_preflight_packet_sha256=(
                        args.accepted_preflight_packet_sha256
                    ),
                    accepted_preflight_manifest_sha256=(
                        args.accepted_preflight_manifest_sha256
                    ),
                    accepted_payload_manifest_sha256=(
                        args.accepted_payload_manifest_sha256
                    ),
                    accepted_receipt_path=args.accepted_receipt,
                    accepted_receipt_sha256=args.accepted_receipt_sha256,
                    accepted_receipt_manifest_path=(
                        args.accepted_receipt_manifest
                    ),
                    accepted_receipt_manifest_sha256=(
                        args.accepted_receipt_manifest_sha256
                    ),
                    product_execution=ActualProductExecutionSelectionV1(
                        contract_version="1",
                        workspace_base=args.workspace_root,
                        settings_path=args.settings,
                        requested_completion_scope="compile_only",
                        completion_classification=classification,
                        objective_verification_command=(
                            tuple(args.objective_verification_argument)
                            if args.objective_verification_argument
                            else None
                        ),
                        game_folder_build_binding=(
                            _read_typed_json_object_file(
                                args.game_folder_build_binding_file,
                                "GameFolder binding",
                            )
                            if args.game_folder_build_binding_file
                            else None
                        ),
                        compiled_artifact_inspection_profile=(
                            _read_typed_json_object_file(
                                args.compiled_artifact_inspection_profile_file,
                                "compiled-artifact inspection profile",
                            )
                            if args.compiled_artifact_inspection_profile_file
                            else None
                        ),
                    ),
                )
            )
        except (ActualRunError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "command": "prepare-sealed-actual",
                        "stage": getattr(exc, "stage", "request"),
                        "code": getattr(exc, "code", "ACTUAL_REQUEST_INVALID"),
                        "reason": str(exc),
                        "actual_authorized": False,
                        "model_generation_calls": 0,
                        "ProductExecution_runs": 0,
                        "Builds": 0,
                        "Apply_runs": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "execute-sealed-actual":
        try:
            result = execute_sealed_actual_run(
                args.packet_root,
                expected_packet_sha256=args.expected_packet_sha256,
                expected_seal_sha256=args.expected_seal_sha256,
                actual_execution_authorized=bool(
                    args.authorize_actual_execution
                ),
                diff_approval_reader=_read_exact_diff_approval_from_stdin,
            )
        except ActualRunError as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "command": "execute-sealed-actual",
                        "stage": exc.stage,
                        "code": exc.code,
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.status == "terminal" else 2

    if args.command == "apply-patch":
        if args.v2_delivery_checkpoint:
            from src.mod_agent.reviewed_patch_continuation import continue_v2_delivery
            try:
                if not args.artifact_store_root or any((args.resume_approval, args.human_decision_stdin,
                        args.resume_payload, args.checkpoint_path, args.checkpoint_hash, args.build_profile)):
                    raise ValueError("V2_DELIVERY_EXPLICIT_CONFIGURATION_REQUIRED")
                action = args.v2_delivery_action
                if (action == "apply" and not args.confirm
                        or action != "apply" and args.confirm
                        or action != "build" and (args.continue_build or args.allow_build)):
                    raise ValueError("V2_DELIVERY_EFFECT_AUTHORIZATION_MISMATCH")
                # 为什么要有它：v2从同一公开命令消费精确显示绑定与真实批准，不把旧协议字段或模型文本当新权限。
                displayed = (json.loads(Path(args.v2_displayed_binding).read_text(encoding="utf-8-sig"))
                             if args.v2_displayed_binding else None)
                decision = (json.loads(Path(args.human_semantic_decision).read_text(encoding="utf-8-sig"))
                            if args.human_semantic_decision else None)
                result = continue_v2_delivery(artifact_store_root=args.artifact_store_root,
                    checkpoint_ref=args.v2_delivery_checkpoint, action=action, isolated_root=args.isolated_root,
                    operation_id=args.operation_id, displayed_binding=displayed, human_semantic_decision=decision,
                    task_delegation_ref=args.task_delegation_ref, allow_build=args.allow_build)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
                print(json.dumps({"status": "blocked", "protocol": "v2", "reason": str(error),
                    **{key: getattr(error, key) for key in ("delivery_receipt_ref", "failure_feedback_ref")
                       if getattr(error, key, None) is not None}}, ensure_ascii=False))
                return 2
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("status") in {"ready", "prepared", "awaiting_approval", "applied_in_isolation", "built"} else 1
        if args.resume_approval:
            from src.mod_agent.reviewed_patch_continuation import resume_deferred_approval

            try:
                if not all((args.artifact_store_root, args.feature_id, args.checkpoint_path,
                        args.checkpoint_hash, args.resume_payload, args.human_semantic_decision)):
                    raise ValueError("DEFERRED_APPROVAL_EXPLICIT_CONFIGURATION_REQUIRED")
                if any((args.confirm, args.workspace_root, args.run_id, args.isolated_root, args.allow_build, args.build_profile)):
                    raise ValueError("DEFERRED_APPROVAL_UNRELATED_ARGUMENTS_FORBIDDEN")
                result = resume_deferred_approval(artifact_store_root=args.artifact_store_root,
                    checkpoint_ref=ArtifactRef(feature_id=args.feature_id, artifact_type="workflow_checkpoint",
                        path=args.checkpoint_path, artifact_hash=args.checkpoint_hash),
                    resume_payload=_load_reviewed_build_resume(args.resume_payload),
                    human_semantic_decision=read_json_object(args.human_semantic_decision),
                    execution_trace_configuration=_execution_trace_configuration(args))
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                result = {"entry": "apply-patch", "status": "blocked", "feature_id": args.feature_id,
                    **_apply_failure_cli_details(exc),
                    "reason": str(exc), "execution_outcome": "not_confirmed_check_resume_report_and_apply_artifacts"}
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] == "applied_in_isolation" else 2
        if args.continue_build:
            try:
                if args.human_semantic_decision:
                    raise ValueError("REVIEWED_BUILD_APPLY_ARGUMENTS_FORBIDDEN")
                result = _continue_reviewed_build_from_cli(args)
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                result = {"entry": "apply-patch", "status": "blocked", "feature_id": args.feature_id,
                    "reason": str(exc), "execution_outcome": "not_confirmed_check_resume_report_and_build_artifacts",
                    "builds_performed": None, "model_calls_performed": 0, "runtime_executions_performed": 0,
                    "product_source_writes_performed": False,
                    "resume_report": getattr(exc, "resume_report", None),
                    "execution_trace": getattr(exc, "execution_trace", None)}
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] == "built" else 1 if result["status"] == "failed" else 2
        if any((args.allow_build, args.checkpoint_path, args.checkpoint_hash, args.resume_payload, args.build_profile, args.human_semantic_decision)):
            print(json.dumps({"entry": "apply-patch", "status": "blocked",
                "reason": "REVIEWED_BUILD_MODE_REQUIRED"}, ensure_ascii=False))
            return 2
        if args.human_decision_stdin:
            from src.mod_agent.reviewed_patch_continuation import apply_reviewed_patch

            try:
                if not args.confirm or not all((args.artifact_store_root, args.workspace_root,
                        args.feature_id, args.run_id, args.isolated_root)):
                    raise ValueError("APPLY_EXPLICIT_CONFIGURATION_REQUIRED")
                request = json.loads(sys.stdin.read(65537))
                if not isinstance(request, dict) or set(request) != {"displayed_binding", "human_semantic_decision"}:
                    raise ValueError("APPLY_HUMAN_REQUEST_INVALID")
                result = apply_reviewed_patch(artifact_store_root=args.artifact_store_root,
                    workspace_root=args.workspace_root, feature_id=args.feature_id, run_id=args.run_id,
                    isolated_root=args.isolated_root, execution_trace_configuration=_execution_trace_configuration(args), **request)
            except (ValueError, RuntimeError, OSError) as exc:
                result = {"entry": "apply-patch", "status": "blocked", "run_id": args.run_id,
                    "feature_id": args.feature_id, "reason": str(exc),
                    "execution_outcome": "not_confirmed_check_isolated_output_if_created",
                    **_apply_failure_cli_details(exc), "model_calls_performed": 0,
                    "builds_performed": 0, "runtime_executions_performed": 0, "stopped_before": "build"}
            print(json.dumps(result, ensure_ascii=False))
            return 0
        decision = PermissionModel().decide("confirmed_write")
        print(
            json.dumps(
                {
                    "status": "BLOCKED" if not decision.allowed else "allowed",
                    "command": "apply-patch",
                    "reason": decision.reason,
                    "required_approval": decision.required_approval,
                    "confirmed": bool(args.confirm),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "evidence-request":
        try:
            result = run_gtfo_r8_evidence_only(
                description=args.description,
                evidence_root=args.evidence_root,
                artifact_root=args.artifact_root,
                target_mod=args.target_mod,
                assembly_path=args.assembly_path,
                current_interop_path=args.current_interop_path,
                enemy_animation_fix_path=args.enemy_animation_fix_path,
                stuck_enemy_fix_path=args.stuck_enemy_fix_path,
            )
        except (
            EvidenceOnlyError,
            R8EvidenceError,
            R8ILInspectionError,
            OSError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {"status": "BLOCKED", "command": "evidence-request", "reason": str(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "intelligent-evidence-request":
        try:
            result = run_intelligent_evidence_only(
                description=args.description,
                source_registry_path=args.source_registry,
                artifact_root=args.artifact_root,
                settings_path=args.settings,
                allow_evidence_disclosure=bool(args.allow_evidence_disclosure),
                target_domain=args.target_domain,
                target_mod=args.target_mod,
                api_key_env=args.api_key_env,
                base_url_env=args.base_url_env,
                local_ollama=bool(args.local_ollama),
                ollama_base_url=args.ollama_base_url,
                ollama_model=args.ollama_model,
                ollama_max_output_tokens=args.ollama_max_output_tokens,
                ollama_context_window=args.ollama_context_window,
                ollama_request_timeout_seconds=args.ollama_request_timeout_seconds,
            )
        except ArtifactStorePreflightError as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": "intelligent-evidence-request",
                        "reason": str(exc),
                        "diagnostic": {
                            "category": "infrastructure_preflight_failure",
                            "component": "artifact_store",
                            "stage": "artifact_preflight",
                            "code": "artifact_path_unaddressable",
                            "operation": "os.replace",
                            "model_execution_started": False,
                            "tool_execution_started": False,
                            "model_call_count": 0,
                            "tool_call_count": 0,
                            "details": {
                                "artifact_type": exc.artifact_type,
                                "path_role": exc.path_role,
                                "source_path_length": exc.source_path_length,
                                "target_path_length": exc.target_path_length,
                                "target_parent_exists": exc.target_parent_exists,
                                "winerror": exc.winerror,
                            },
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        except (IntelligentEvidenceEntryError, OSError, RuntimeError, ValueError) as exc:
            blocked_payload: dict[str, object] = {
                "status": "BLOCKED",
                "command": "intelligent-evidence-request",
                "reason": str(exc),
            }
            if isinstance(exc, IntelligentEvidenceEntryError) and exc.diagnostic:
                blocked_payload["diagnostic"] = dict(exc.diagnostic)
                if exc.diagnostic.get("run_status") == "budget_exhausted":
                    blocked_payload["status"] = "budget_exhausted"
            print(
                json.dumps(
                    blocked_payload,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "resume-retained-task-agent-handoff":
        from src.mod_agent.local_entry import resume_retained_task_agent_handoff

        try:
            result = resume_retained_task_agent_handoff(execution_config=args.execution_config,
                run_root=args.run_root, run_id=args.run_id,
                ready_execution_ref={"path": str(Path(args.execution_path).resolve()), "sha256": args.execution_sha256},
                attempt_id=args.attempt_id)
        except (OSError, ValueError, KeyError) as error:
            print(json.dumps({"status": "blocked", "command": args.command,
                "reason": str(error), "new_provider_sends": 0, "new_builds": 0,
                "actual_write_performed": False}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") in {"awaiting_approval", "delivered"} else 2

    if args.command in {"qualify-repository-evaluator", "evaluate-repository-strategy"}:
        from src.mod_agent.evaluation.independent_preparation import file_ref
        from src.mod_agent.evaluation.strategy_repository_evaluator import (
            generate_repository_strategy_verdict,
            qualify_repository_strategy_evaluator,
        )
        from src.mod_agent.model_qualification import read_bound_json as read_repository_record

        try:
            # 为什么要有它：正常入口只消费已执行原件，资格与判分不能由调用者传入passed摘要或私下执行新代码。
            if args.command == "qualify-repository-evaluator":
                document = read_repository_record(file_ref(args.input_record))
                fields = {"profile_ref", "controls", "production_paths", "target_test_ids", "preservation_test_ids"}
                if (set(document) != fields | {"schema_version"}
                        or document["schema_version"] != "repository_strategy_qualification_input_v1"):
                    raise ValueError("REPOSITORY_QUALIFICATION_INPUT_INVALID")
                reference = qualify_repository_strategy_evaluator(**{key: document[key] for key in fields},
                    output_root=args.output_root)
                result = {"status": "qualified", "evaluator_ref": reference}
            else:
                reference = generate_repository_strategy_verdict(evaluator_ref=file_ref(args.evaluator_record),
                    execution_ref=file_ref(args.execution_record), evidence_bundle_ref=file_ref(args.evidence_record),
                    output_path=args.output)
                result = {"status": "recorded", "verdict_ref": reference}
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
            print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=False))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export-strategy-execution":
        from src.mod_agent.evaluation.independent_preparation import file_ref
        from src.mod_agent.evaluation.strategy_native_export import export_strategy_execution

        try:
            # 为什么要有它：评测通过正常命令消费真实执行原件，动作、经验送达和成本由导出器核验，调用者不能手填成绩。
            reference = export_strategy_execution(native_execution_ref=file_ref(args.execution_record),
                call_id=args.call_id, output_root=args.output_root, artifact_store_root=args.artifact_store_root,
                protocol_ref=file_ref(args.protocol_record) if args.protocol_record else None,
                task_id=args.task_id, arm=args.arm, attempt=args.attempt,
                learning_receipt_ref=file_ref(args.learning_receipt) if args.learning_receipt else None,
                delivery_receipt_ref=args.delivery_receipt)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
            print(json.dumps({"status": "blocked", "reason": str(error)}, ensure_ascii=False))
            return 2
        print(json.dumps({"status": "exported", "execution_ref": reference}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run-request":
        if args.resume_task_agent:
            from src.mod_agent.task_agent_v2_recovery import resume_v2_task_agent
            recovery_invoked = False
            try:
                selected = list(argv if argv is not None else sys.argv[1:])
                allowed_flags = {"--resume-task-agent", "--artifact-store-root"}
                if (not args.artifact_store_root or args.description is not None or args.target_project is not None
                        or any(value.startswith("--") and value.split("=", 1)[0] not in allowed_flags for value in selected)):
                    raise ValueError("TASK_V2_RESUME_ORIGINAL_CONFIGURATION_ONLY")
                # 为什么要有它：恢复必须重用原模型、授权和累计资源，不让新的命令行参数悄悄创建另一套执行条件。
                recovery_invoked = True
                result = resume_v2_task_agent(artifact_store_root=args.artifact_store_root,
                                             checkpoint_ref=args.resume_task_agent)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
                summary = getattr(error, "outcome_finalization", {})
                if not recovery_invoked and args.artifact_store_root:
                    from src.mod_agent.run_outcome_finalization import finalize_continuation
                    summary = finalize_continuation(artifact_store_root=args.artifact_store_root,
                        event="recovery_admission_rejected", checkpoint_ref=args.resume_task_agent, error=error)
                print(json.dumps({"status": "blocked", "protocol": "v2", "reason": str(error), **summary}, ensure_ascii=False))
                return 2
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("status") in {"awaiting_delivery", "candidate_ready"} else 1
        if args.resume_initial_findings is not None:
            from src.mod_agent.initial_findings_continuation import (
                execute_initial_findings_continuation,
            )
            continuation_invoked = False
            try:
                if not args.findings_stage_allocation or not args.findings_approval_stdin:
                    raise ValueError("INITIAL_FINDINGS_EXPLICIT_APPROVAL_CHANNEL_REQUIRED")
                original = read_json_object(args.resume_initial_findings)
                expected = build_parser().parse_args(["run-request", *original["arguments"], "--", args.description])
                excluded = {"resume_initial_findings", "findings_stage_allocation", "findings_approval_stdin"}
                if {key: value for key, value in vars(args).items() if key not in excluded} != {
                        key: value for key, value in vars(expected).items() if key not in excluded}:
                    raise ValueError("INITIAL_FINDINGS_CLI_CONFIGURATION_CHANGED")
                original_manifest = read_json_object(Path(args.workspace_root) / args.run_id / "run_manifest.json")
                if args.description != original_manifest["request"]:
                    raise ValueError("INITIAL_FINDINGS_CLI_REQUEST_CHANGED")
                raw = sys.stdin.buffer.read(16385)
                if len(raw) > 16384:
                    raise ValueError("INITIAL_FINDINGS_APPROVAL_TOO_LARGE")
                decision = json.loads(raw)
                continuation_invoked = True
                result = execute_initial_findings_continuation(
                    run_root=Path(args.workspace_root) / args.run_id, run_id=args.run_id,
                    execution_config=args.resume_initial_findings, allocation_config=args.findings_stage_allocation,
                    decision=decision)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                result = {"entry": "initial-findings-continuation", "run_id": args.run_id,
                          "status": "blocked", "reason": str(error),
                          "model_calls_performed": None if continuation_invoked else 0,
                          "continuation_invocation_attempted": continuation_invoked}
            print(json.dumps(result, ensure_ascii=True))
            return 0 if result["status"] == "awaiting_approval" else 2
        if args.findings_stage_allocation or args.findings_approval_stdin:
            raise ValueError("INITIAL_FINDINGS_RESUME_SELECTOR_REQUIRED")
        try:
            if not args.target_project or not args.description:
                raise ValueError("RUN_REQUEST_TARGET_AND_DESCRIPTION_REQUIRED")
            dependency_binding = None
            if args.game_folder_build_binding:
                dependency_document = read_json_object(args.game_folder_build_binding)
                if dependency_document.pop("binding_version", None) != "run_local_game_folder_binding_v1":
                    raise ValueError("RUN_REQUEST_GAME_FOLDER_BINDING_VERSION_INVALID")
                dependency_binding = RunLocalGameFolderBindingV1(**dependency_document)
            external_profile = None
            if args.external_patch_profile:
                from .external_patch import ExternalQwenPatchProfile
                external_profile = ExternalQwenPatchProfile(**read_json_object(args.external_patch_profile))
            worker_configuration = PatchWorkerConfiguration.load(args.patch_worker_config) if args.patch_worker_config else None
            stage_limits = read_json_object(args.stage_model_call_limits) if args.stage_model_call_limits else None
            if stage_limits is not None and (not stage_limits or any(type(value) is not int or value < 1 for value in stage_limits.values())):
                raise ValueError("RUN_REQUEST_STAGE_MODEL_LIMITS_INVALID")
            if args.task_model_call_limit is not None and args.task_model_call_limit < 1:
                raise ValueError("RUN_REQUEST_TASK_MODEL_LIMIT_INVALID")
            result = run_local_request(
                description=args.description,
                target_project=args.target_project,
                repository_profile_path=args.repository_profile,
                run_id=args.run_id,
                request_id=args.request_id,
                game_folder_build_binding=dependency_binding,
                dependency_api_config=args.dependency_api_config,
                knowledge_evidence_config=args.knowledge_evidence_config,
                external_patch_profile=external_profile,
                workspace_base=args.workspace_root,
                settings_path=args.settings,
                api_key_env=args.api_key_env,
                base_url_env=args.base_url_env,
                confirm_isolated_write=bool(args.confirm_isolated_write),
                allow_build=bool(args.allow_build),
                allow_source_disclosure=bool(args.allow_source_disclosure),
                run_gtfo_runtime=bool(args.run_gtfo_runtime),
                allow_game_launch=bool(args.allow_game_launch),
                runtime_plugin_destination=args.runtime_plugin_destination,
                allow_game_directory_write=bool(args.allow_game_directory_write),
                diff_approval_reader=_read_exact_diff_approval_from_stdin,
                local_ollama=bool(args.local_ollama),
                ollama_base_url=args.ollama_base_url,
                ollama_model=args.ollama_model,
                ollama_max_output_tokens=args.ollama_max_output_tokens,
                ollama_context_window=args.ollama_context_window,
                planning_target_evidence_ids=tuple(
                    args.planning_target_evidence_id
                ),
                **({"ollama_timeout_seconds": args.ollama_timeout_seconds}
                   if args.ollama_timeout_seconds is not None else {}),
                patch_worker_configuration=worker_configuration,
                source_retrieval_mode=args.source_retrieval,
                reuse_query_observations=args.reuse_query_observations,
                compress_findings_input=args.compress_findings_input,
                task_model_call_limit=args.task_model_call_limit,
                model_call_budget_policy=args.model_call_budget_policy,
                **({"technical_executor": args.technical_executor,
                    "constraint_policy": args.constraint_policy,
                    "task_agent_candidate_mode": args.task_agent_candidate_mode,
                    "task_agent_protocol": args.task_agent_protocol,
                    "task_agent_experience_config": args.task_agent_experience_config,
                    "task_agent_semantic_state": args.task_agent_semantic_state,
                    "task_agent_allowed_edit_paths": _task_agent_edit_paths(args.task_agent_edit_paths)}
                   if (args.technical_executor != "workflow" or args.constraint_policy is not None
                       or args.task_agent_edit_paths is not None or args.task_agent_candidate_mode != "baseline"
                       or args.task_agent_protocol != "v1" or args.task_agent_experience_config is not None
                       or args.task_agent_semantic_state != "enabled") else {}),
                stage_model_call_limits=stage_limits,
                execution_trace_configuration=_execution_trace_configuration(args),
                defer_approval=bool(args.defer_approval),
            )
        except ProductExecutionPresentationError as exc:
            print(
                json.dumps(
                    {
                        "status": "PRESENTATION_FAILED",
                        **exception_execution_trace_projection(exc),
                        "product_execution_result": (
                            exc.product_execution_result.to_dict()
                        ),
                        "audit_error": {
                            "failure_class": exc.failure_class,
                            "failure_message": exc.failure_message,
                            "sink": exc.sink,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        except ProductExecutionContractError as exc:
            print(
                json.dumps(
                    {
                        "status": "CONTRACT_ERROR",
                        **exception_execution_trace_projection(exc),
                        "reason": exc.reason,
                        "side_effects_may_have_started": (
                            exc.side_effects_may_have_started
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        except (LocalEntryError, OSError, ValueError) as exc:
            print(
                json.dumps(
                    {"status": "BLOCKED", "reason": str(exc),
                     **exception_execution_trace_projection(exc)},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("status") == "awaiting_approval":
            return 0
        structured = result.get("product_execution_result")
        if not isinstance(structured, Mapping):
            return 2
        try:
            machine_result = ProductExecutionResultV1.from_dict(structured)
        except (ProductExecutionContractError, ValueError, TypeError):
            return 2
        return {
            "succeeded": 0,
            "awaiting_delivery": 0,
            "domain_blocked": 1,
            "platform_blocked": 2,
            "partial": 2,
        }[machine_result.execution_disposition]

    if args.command == "preflight-stagger-edit-authority":
        try:
            result = preflight_stagger_edit_authority(
                **_stagger_edit_authority_arguments(args)
            )
        except (
            StaggerEditAuthorityPreflightError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": "preflight-stagger-edit-authority",
                        "reason": str(exc),
                        "approval_required": False,
                        "registry_mutation_allowed": False,
                        "repair_allowed": False,
                        "writes_performed": False,
                        "artifact_writes_performed": False,
                        "repair_planner_invoked": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "request-stagger-edit-authority":
        try:
            result = create_stagger_edit_authority_request(
                **_stagger_edit_authority_arguments(args),
            )
        except (
            StaggerEditAuthorityApprovalError,
            StaggerEditAuthorityPreflightError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": "request-stagger-edit-authority",
                        "reason": str(exc),
                        "writes_performed": None,
                        "artifact_store_inspection_required": True,
                        "repair_planner_invoked": False,
                        "repair_allowed": False,
                        "source_patch_allowed": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if (
        args.command
        == "preflight-stagger-existing-binding-reauthorization"
    ):
        try:
            result = preflight_stagger_existing_binding_reauthorization(
                **_stagger_existing_binding_reauthorization_arguments(
                    args
                )
            )
        except (
            StaggerEditAuthorityPreflightError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": args.command,
                        "reason": str(exc),
                        "approval_required": False,
                        "registry_mutation_allowed": False,
                        "repair_allowed": False,
                        "writes_performed": False,
                        "artifact_writes_performed": False,
                        "repair_planner_invoked": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if (
        args.command
        == "request-stagger-existing-binding-reauthorization"
    ):
        try:
            result = (
                create_stagger_existing_binding_reauthorization_request(
                    **_stagger_existing_binding_reauthorization_arguments(
                        args
                    )
                )
            )
        except (
            StaggerEditAuthorityApprovalError,
            StaggerEditAuthorityPreflightError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": args.command,
                        "reason": str(exc),
                        "writes_performed": None,
                        "artifact_store_inspection_required": True,
                        "repair_planner_invoked": False,
                        "repair_allowed": False,
                        "source_patch_allowed": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if (
        args.command
        == "preflight-stagger-hud-parent-observation-authority"
    ):
        try:
            result = preflight_stagger_hud_parent_observation_authority(
                **_stagger_observation_authority_arguments(args)
            )
        except (
            StaggerEditAuthorityPreflightError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": args.command,
                        "reason": str(exc),
                        "approval_required": False,
                        "registry_mutation_allowed": False,
                        "repair_allowed": False,
                        "writes_performed": False,
                        "artifact_writes_performed": False,
                        "repair_planner_invoked": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if (
        args.command
        == "request-stagger-hud-parent-observation-authority"
    ):
        try:
            result = (
                create_stagger_hud_parent_observation_authority_request(
                    **_stagger_observation_authority_arguments(args)
                )
            )
        except (
            StaggerEditAuthorityApprovalError,
            StaggerEditAuthorityPreflightError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": args.command,
                        "reason": str(exc),
                        "writes_performed": None,
                        "artifact_store_inspection_required": True,
                        "repair_planner_invoked": False,
                        "repair_allowed": False,
                        "source_patch_allowed": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if (
        args.command
        == "preflight-stagger-hud-parent-diagnostic-exact-diff"
    ):
        try:
            result = preflight_stagger_hud_parent_diagnostic_exact_diff(
                artifact_store_root=args.artifact_store_root,
                feature_id=args.feature_id,
                run_id=args.run_id,
                edit_authority_approval_path=(
                    args.edit_authority_approval_path
                ),
                edit_authority_approval_hash=(
                    args.edit_authority_approval_hash
                ),
                evidence_pack_path=args.evidence_pack_path,
                evidence_pack_hash=args.evidence_pack_hash,
                implementation_plan_path=args.implementation_plan_path,
                implementation_plan_hash=args.implementation_plan_hash,
                source_root=args.source_root,
            )
        except StaggerDiagnosticDiffPreflightError as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": (
                            "preflight-stagger-hud-parent-diagnostic-"
                            "exact-diff"
                        ),
                        "reason": str(exc),
                        "writes_performed": False,
                        "artifact_writes_performed": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "preflight-stagger-edit-authority-review":
        try:
            result = preflight_stagger_edit_authority_review(
                artifact_store_root=args.artifact_store_root,
                feature_id=args.feature_id,
                edit_authority_request_path=args.request_path,
                edit_authority_request_hash=(
                    args.request_artifact_hash
                ),
                authority_request_hash=args.authority_request_hash,
            )
        except LocalEntryError as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": (
                            "preflight-stagger-edit-authority-review"
                        ),
                        "reason": str(exc),
                        "writes_performed": False,
                        "artifact_writes_performed": False,
                        "repair_planner_invoked": False,
                        "source_patch_allowed": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "resume-stagger-edit-authority-review":
        try:
            result = resume_stagger_edit_authority_review(
                artifact_store_root=args.artifact_store_root,
                feature_id=args.feature_id,
                edit_authority_request_path=args.request_path,
                edit_authority_request_hash=args.request_artifact_hash,
                authority_request_hash=args.authority_request_hash,
                resume_id=args.resume_id,
                decision=args.decision,
                approved_by=args.approved_by,
                approved_at=args.approved_at,
            )
        except EditAuthorityPresentationError as exc:
            print(
                json.dumps(
                    {
                        "result_type": "edit_authority_presentation_error_v1",
                        "status": "PRESENTATION_FAILED",
                        "command": "resume-stagger-edit-authority-review",
                        "reason": str(exc),
                        "edit_authority_review_result": (
                            exc.edit_authority_review_result.to_dict()
                        ),
                        "presentation_error": exc.presentation_error,
                        "audit_error": exc.audit_error,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        except (LocalEntryError, OSError, RuntimeError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "command": (
                            "resume-stagger-edit-authority-review"
                        ),
                        "reason": str(exc),
                        "artifact_store_inspection_required": True,
                        "repair_planner_invoked": False,
                        "source_patch_allowed": False,
                        "source_modified": False,
                        "build_performed": False,
                        "process_launched": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return _edit_authority_review_exit_code(result)

    if args.command == "preflight-runtime":
        return _run_preflight_runtime_command(args)

    if args.command == "resume-runtime":
        return _run_resume_runtime_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
