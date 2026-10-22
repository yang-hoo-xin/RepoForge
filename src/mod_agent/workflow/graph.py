"""LangGraph StateGraph construction for the controlled development workflow."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.core.trace.execution_trace import current_trace, trace_operation
from src.mod_agent.artifact_semantics import project_artifact_write_intent
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore, ArtifactStoreError
from src.mod_agent.bindings import resolve_runtime_diagnosis_binding
from src.mod_agent.build import BuildResult, build_fake_build_result
from src.mod_agent.domain import (
    GTFO_DOMAIN_NAME,
    DomainAdapterRegistry,
    DomainReviewPolicy,
    GTFOAdapter,
    UnsupportedDomainAdapter,
)
from src.mod_agent.edit_authority_capabilities import (
    EditAuthorityCapabilityDeclaration,
    EditAuthorityStoreStepV1,
    InvalidEditAuthorityApprovalIdentityError,
    UnsupportedEditAuthorityCapabilityError,
    bind_edit_authority_exact_input_upstreams,
    edit_authority_identity_sections,
    project_edit_authority_review_result,
    resolve_edit_authority_request_capability,
    validate_edit_authority_exact_input_admission,
    validate_edit_authority_request_identity,
)
from src.mod_agent.evaluation import build_transition_eval_gate_report
from src.mod_agent.evidence import (
    DenseRetrievalProvider,
    DeterministicEvidenceRetriever,
    EvidencePack,
    EvidenceRequest,
)
from src.mod_agent.execution import ExactDiffApprovalRejectedError
from src.mod_agent.fresh_planning_continuation import (
    PatchContinuationError,
    PatchContinuationInputV1,
    PlanningContinuationArtifactWriteError,
    PlanningContinuationError,
    PlanningContinuationInputV1,
    preflight_patch_continuation,
    preflight_planning_continuation,
)
from src.mod_agent.git import GitCommandError, GitRepository
from src.mod_agent.knowledge import (
    CollectionResolutionError,
    CollectionResolver,
    KnowledgeDatabase,
)
from src.mod_agent.planning import (
    FeatureSpec,
    FeatureSpecBuilder,
    ImplementationPlanner,
    ModPlanningBriefBuilder,
    RequestPlan,
    RequestPlanner,
    Uncertainty,
)
from src.mod_agent.planning.model_selection import (
    ModelPlanningTargetSelector,
    PlanningTargetSelectionError,
)
from src.mod_agent.product_execution import (
    ProductExecutionContractError,
    ProductExecutionGraphModeV1,
)
from src.mod_agent.proposal import (
    CRITIC_DIMENSIONS,
    CriticAssessment,
    CriticFinding,
    PatchProposalBuilder,
    ProposalCritic,
    ProposalReviewer,
)
from src.mod_agent.proposal.models import SemanticAssuranceV1
from src.mod_agent.repair import RepairPatchProposalBuilder, RepairPlanner
from src.mod_agent.runtime import (
    build_fake_runtime_evidence_payload,
    make_gtfo_runtime_evidence_factory_from_settings,
)
from src.mod_agent.runtime.evidence import bind_runtime_evidence_context
from src.mod_agent.runtime.stagger_edit_authority import (
    StaggerEditAuthorityApprovalError,
    StaggerEditAuthorityMaterializationError,
    materialize_stagger_edit_authority_approval,
)
from src.mod_agent.runtime.stagger_planner_admission import (
    materialize_stagger_planning_approval,
)
from src.mod_agent.runtime_evidence_consolidation import (
    project_operation_payload as project_consolidated_runtime_evidence,
)
from src.mod_agent.upstream_trace import payload_sha256, update_observation
from src.mod_agent.verification_operations import (
    EVIDENCE_CONSOLIDATION_COMPLETION,
    RUNTIME_EVIDENCE_ONLY_COMPLETION,
    RUNTIME_WORKFLOW_COMPLETION,
    BuildInvocationProtocolBindingV1,
    EvidenceConsolidationOperationV1,
    NewRuntimeResumeProtocolBindingV1,
    ObservationRetryAttemptV1,
    RuntimeTerminalizationOperationV1,
    VerificationBuildOperationV1,
    VerificationOperationDeclaration,
    VerificationOperationEnvelope,
    VerificationProtocolBinding,
    resolve_verification_operation,
    validate_verification_operation_envelope,
)

from .checkpointing import (
    BUILD_CHECKPOINT_EXACT_REVISION_TYPES,
    WorkflowCheckpointStore,
)
from .interrupts import InterruptRequest, ResumePayload
from .nodes import StubNodeRunner
from .state import ModAgentState, _copy_mod_agent_state
from .transitions import (
    route_after_binding_resolution,
    route_after_build,
    route_after_dry_run,
    route_after_evaluate,
    route_after_evidence,
    route_after_feature_spec,
    route_after_human_review_gate,
    route_after_implementation_plan,
    route_after_isolated_apply,
    route_after_patch_proposal,
    route_after_plan_request,
    route_after_planning_brief,
    route_after_repair_plan,
    route_after_review,
    route_after_runtime_evidence,
    route_registry,
)
from .verification_only import (
    VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
    VerificationOnlyContinuationInput,
    VerificationOnlyRuntimeTerminalInput,
    preflight_verification_only_continuation,
    validate_verification_only_build_result,
    verification_only_lineage_matches,
    verification_only_lineage_payload,
)

GraphNode = Callable[[ModAgentState], ModAgentState]
BuildResultFactory = Callable[[str, str | None], BuildResult]
RuntimeEvidenceFactory = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]
EvaluationContextFactory = Callable[[ModAgentState], dict[str, Any]]
ProposalGitBindingFactory = Callable[[], dict[str, Any]]
EvidencePackFactory = Callable[[ModAgentState, dict[str, Any]], EvidencePack]
PlanningTargetSelectionFactory = Callable[
    [dict[str, Any], dict[str, Any]], tuple[str, ...]
]
PatchChangeFactory = Callable[
    [ModAgentState, dict[str, Any], dict[str, Any]], tuple[dict[str, Any], ...]
]
ProposalReviewContextFactory = Callable[
    [ModAgentState, dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any], dict[str, Any]],
]
PatchExecutionFactory = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]
]
PatchContinuationAttemptRecorder = Callable[[Mapping[str, Any]], None]


class _RuntimeEvidenceExecutionBoundaryError(RuntimeError):
    """Carry a platform-owned RuntimeEvidence failure stage across the Graph."""

    _ALLOWED_STAGES = frozenset(
        {"store_preflight", "runtime_execution", "post_runtime_validation"}
    )

    def __init__(self, failure_stage: str, cause: Exception) -> None:
        if failure_stage not in self._ALLOWED_STAGES:
            raise ValueError(
                f"Unsupported RuntimeEvidence failure stage: {failure_stage}"
            )
        super().__init__(str(cause))
        self.failure_stage = failure_stage
        self.failure_class = type(cause).__name__


CriticRequirement = Literal["not_required", "required"]
_FEATURE_SPEC_REVISION_APPROVAL_KEY = "feature_spec_revision"
_FEATURE_SPEC_REVISION_PROJECTION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "feature_id",
        "producer_stage",
        "status",
        "payload",
        "inputs",
        "blockers",
        "trace_refs",
    }
)
_FEATURE_SPEC_REVISION_PAYLOAD_FIELDS = frozenset(
    {
        "acceptance_criteria",
        "admission_boundary",
        "automation_claim",
        "blocked",
        "blockers",
        "evidence_requirements",
        "feature_id",
        "hook_requirement",
        "message",
        "node_name",
        "non_goals",
        "operator_guided_technical_hypotheses",
        "revision_reason",
        "risk_level",
        "runtime_verification_plan",
        "source_lineage",
        "specification_mode",
        "stage",
        "success_marker_contract",
        "supersedes_artifact_hash",
        "target_behavior",
        "target_mod",
        "user_request",
    }
)
_DEFERRED_CORRELATION_HYPOTHESIS_IDS = frozenset(
    {
        "input_event_mapping",
        "result_event_mapping",
        "enemy_identity_mapping",
        "exactly_once_correlation",
    }
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _verification_build_continuation(
    operation: VerificationOperationEnvelope,
) -> VerificationOnlyContinuationInput:
    declaration = resolve_verification_operation(
        validate_verification_operation_envelope(operation)
    )
    variant = operation.variant_payload
    if (
        declaration.graph_handler != "verification_build"
        or not isinstance(variant, VerificationBuildOperationV1)
    ):
        raise ValueError("Verification operation is not a Build operation.")
    return VerificationOnlyContinuationInput(
        feature_spec_ref=variant.feature_spec_ref,
        evidence_pack_ref=variant.evidence_pack_ref,
    )


def _verification_runtime_checkpoint_id(
    operation: VerificationOperationEnvelope,
) -> str:
    declaration = resolve_verification_operation(
        validate_verification_operation_envelope(operation)
    )
    if declaration.graph_handler != "runtime_execution":
        raise ValueError("Verification operation is not a Runtime execution.")
    checkpoint = operation.approval_identity.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Runtime execution identity lacks checkpoint facts.")
    checkpoint_id = checkpoint.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("Runtime execution checkpoint_id is invalid.")
    return checkpoint_id


def _verification_runtime_terminal_input(
    operation: VerificationOperationEnvelope,
) -> VerificationOnlyRuntimeTerminalInput:
    declaration = resolve_verification_operation(
        validate_verification_operation_envelope(operation)
    )
    variant = operation.variant_payload
    protocol = operation.protocol_binding
    if (
        declaration.graph_handler != "runtime_terminalization"
        or not isinstance(variant, RuntimeTerminalizationOperationV1)
        or not isinstance(protocol, VerificationProtocolBinding)
    ):
        raise ValueError("Verification operation is not a Runtime terminalization.")
    return VerificationOnlyRuntimeTerminalInput(
        checkpoint_ref=protocol.checkpoint_ref,
        resume_report_ref=protocol.accepted_resume_report_ref,
        runtime_evidence_ref=variant.runtime_evidence_ref,
        resume_id=protocol.resume_id,
        approved_by=protocol.approved_by,
    )


def _verification_operation_writes(
    declaration: VerificationOperationDeclaration | None,
    artifact_type: str,
) -> bool:
    return declaration is not None and artifact_type in declaration.output_artifact_types


@dataclass(frozen=True)
class _VerificationGraphControls:
    handler: str | None
    runtime_evidence_store_results: bool
    terminal_store_results: bool
    stop_after_build: bool
    persist_build_handoff: bool
    stop_after_runtime_evidence: bool


def _verification_graph_controls(
    declaration: VerificationOperationDeclaration | None,
) -> _VerificationGraphControls:
    return _VerificationGraphControls(
        handler=None if declaration is None else declaration.graph_handler,
        runtime_evidence_store_results=_verification_operation_writes(
            declaration,
            "runtime_evidence",
        ),
        terminal_store_results=any(
            _verification_operation_writes(declaration, artifact_type)
            for artifact_type in (
                "eval_report",
                "blocked_report",
                "workflow_summary",
            )
        ),
        stop_after_build=(
            declaration is not None
            and declaration.graph_handler == "verification_build"
        ),
        persist_build_handoff=(
            declaration is not None
            and declaration.graph_handler == "verification_build"
        ),
        stop_after_runtime_evidence=(
            declaration is not None
            and declaration.completion_scope
            in {
                RUNTIME_EVIDENCE_ONLY_COMPLETION,
                EVIDENCE_CONSOLIDATION_COMPLETION,
            }
        ),
    )


def _route_after_typed_runtime_evidence(state: ModAgentState) -> str:
    """Stop platform failures before product repair routing consumes no Artifact."""

    runtime_summary = state.get("runtime_summary") or {}
    if (
        runtime_summary.get("platform_failure_stage") is not None
        or (
            state.get("blockers")
            and "runtime_evidence" not in (state.get("artifacts") or {})
        )
    ):
        return "stop"
    return route_after_runtime_evidence(state)


def _validate_product_graph_start(
    state: ModAgentState,
    mode: ProductExecutionGraphModeV1,
    artifact_store: ArtifactStore | None = None,
) -> str:
    """Reject typed product request drift before the first node can act."""

    request = mode.request
    expected = {
        "run_id": request.run_id,
        "request_id": request.request_id,
        "user_request": request.user_request,
        "target_domain": request.target_domain,
        "target_mod": request.target_mod,
    }
    observed = {field: state.get(field) for field in expected}
    if observed != expected:
        raise ProductExecutionContractError(
            "Product Graph initial state changed after preflight.",
            side_effects_may_have_started=False,
        )
    if state.get("approvals") != {"confirmed_write": True}:
        raise ProductExecutionContractError(
            "Product Graph isolated-write authority changed after preflight.",
            side_effects_may_have_started=False,
        )
    if mode.interrupted_patch_selection is not None or mode.pretransport_patch_selection is not None:
        from src.mod_agent.interrupted_patch_admission import admit_interrupted_patch
        from src.mod_agent.workflow.state import create_initial_state
        if artifact_store is None:
            raise ProductExecutionContractError("Interrupted Patch START requires its original Store.",
                                                side_effects_may_have_started=False)
        if mode.pretransport_patch_selection is not None:
            from src.mod_agent.pretransport_patch_admission import admit_pretransport_patch
            admit_interrupted_patch = admit_pretransport_patch
        try:
            admitted = admit_interrupted_patch(mode.pretransport_patch_selection or mode.interrupted_patch_selection,
                workspace_root=Path(request.isolated_workspace_root), artifact_store=artifact_store)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise ProductExecutionContractError("Interrupted Patch START admission changed: " + str(exc),
                                                side_effects_may_have_started=False) from exc
        expected_state = create_initial_state(run_id=request.run_id, request_id=request.request_id,
            user_request=request.user_request, target_domain=request.target_domain,
            target_mod=request.target_mod, confirmed_isolated_write=True)
        expected_state.update(feature_id=admitted["feature_id"], risk_level=admitted["feature_spec"]["risk_level"],
            current_stage="implementation_plan", next_stage="patch_proposal",
            artifacts={role: ref["path"] for role, ref in admitted["selected_artifacts"].items()},
            artifact_hashes={role: ref["artifact_hash"] for role, ref in admitted["selected_artifacts"].items()})
        # 为什么要有它：Graph START重新核对原始只读准入及最小初态，避免调用者借恢复入口继承审批或注入重试/状态。
        if (state != expected_state or admitted["request_id"] != request.request_id
                or admitted["parent_run_id"] != request.run_id
                or admitted["user_request"] != request.user_request):
            raise ProductExecutionContractError("Interrupted Patch Graph initial state changed after admission.",
                                                side_effects_may_have_started=False)
    return "continue"


def _validate_planning_continuation_graph_start(
    state: ModAgentState,
    mode: PlanningContinuationInputV1,
    artifact_store: ArtifactStore,
) -> str:
    """Reject exact-ref state drift before the Planning node can act."""

    admitted = preflight_planning_continuation(
        mode,
        artifact_store=artifact_store,
    )
    expected_artifacts = {
        "feature_spec": mode.feature_spec_ref.path,
        "evidence_pack": mode.evidence_pack_ref.path,
    }
    expected_hashes = {
        "feature_spec": mode.feature_spec_ref.artifact_hash,
        "evidence_pack": mode.evidence_pack_ref.artifact_hash,
    }
    if (
        state.get("feature_id") != mode.feature_spec_ref.feature_id
        or state.get("user_request") != admitted.feature_spec["user_request"]
        or state.get("target_mod") != admitted.feature_spec["target_mod"]
        or state.get("risk_level") != admitted.feature_spec["risk_level"]
        or state.get("current_stage") != "evidence_retrieval"
        or state.get("next_stage") != "implementation_plan"
        or state.get("artifacts") != expected_artifacts
        or state.get("artifact_hashes") != expected_hashes
        or state.get("approvals") != {}
        or state.get("blockers") != []
    ):
        raise ValueError(
            "Planning continuation state changed after exact-ref admission."
        )
    for field in ("run_id", "request_id", "user_request"):
        value = state.get(field)
        if type(value) is not str or not value:
            raise ValueError(
                "Planning continuation state identity is incomplete."
            )
    return "continue"


def _validate_patch_continuation_graph_start(
    state: ModAgentState,
    mode: PatchContinuationInputV1,
    artifact_store: ArtifactStore,
) -> str:
    """Reject state, exact-ref, binding, or source drift before Patch."""

    admitted = preflight_patch_continuation(
        mode,
        artifact_store=artifact_store,
    )
    if mode.saved_patch_selection is not None and (
        admitted.saved_patch is None
        or state.get("request_id") != admitted.saved_patch["request_id"]
    ):
        raise PatchContinuationError(
            "SAVED_PATCH_LOGICAL_REQUEST_DRIFT", "patch_admission",
            "Saved output continuation must retain its original logical request.",
        )
    if mode.candidate_reverification_selection is not None and (
            admitted.saved_candidate is None or state.get("request_id") != admitted.saved_candidate["request_id"]):
        raise PatchContinuationError("SAVED_CANDIDATE_LOGICAL_REQUEST_DRIFT", "patch_admission", "Candidate retry must retain the original request.")
    expected_artifacts = {
        "feature_spec": mode.feature_spec_ref.path,
        "evidence_pack": mode.evidence_pack_ref.path,
        "implementation_plan": mode.implementation_plan_ref.path,
    }
    expected_hashes = {
        "feature_spec": mode.feature_spec_ref.artifact_hash,
        "evidence_pack": mode.evidence_pack_ref.artifact_hash,
        "implementation_plan": mode.implementation_plan_ref.artifact_hash,
    }
    recovery = admitted.repair_context is not None
    if recovery:
        for ref in (mode.repair_parent_proposal_ref, mode.repair_parent_review_ref):
            expected_artifacts[ref.artifact_type] = ref.path
            expected_hashes[ref.artifact_type] = ref.artifact_hash
    if (
        state.get("feature_id") != mode.feature_spec_ref.feature_id
        or state.get("user_request") != admitted.feature_spec["user_request"]
        or state.get("target_mod") != admitted.feature_spec["target_mod"]
        or state.get("risk_level") != admitted.feature_spec["risk_level"]
        or state.get("current_stage") != ("behavior_report" if recovery else "implementation_plan")
        or state.get("next_stage") != ("repair_plan" if recovery else "patch_proposal")
        or state.get("retry_count") != (admitted.repair_context["parent_retry_count"] if recovery else 0)
        or state.get("artifacts") != expected_artifacts
        or state.get("artifact_hashes") != expected_hashes
        or state.get("approvals") != {}
        or state.get("blockers") != []
    ):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_STATE_DRIFT",
            "patch_admission",
            "Patch continuation state changed after exact-ref admission.",
        )
    for field in ("run_id", "request_id", "user_request"):
        value = state.get(field)
        if type(value) is not str or not value:
            raise PatchContinuationError(
                "PATCH_CONTINUATION_STATE_IDENTITY_INVALID",
                "patch_admission",
                "Patch continuation state identity is incomplete.",
            )
    return "repair" if recovery else "continue"


def build_graph(
    artifact_store: ArtifactStore,
    *,
    settings: Any | None = None,
    knowledge_database: KnowledgeDatabase | None = None,
    collection_id: str | None = None,
    collection_resolver: CollectionResolver | None = None,
    dense_provider: DenseRetrievalProvider | None = None,
    dense_enabled: bool = False,
    dense_required: bool = False,
    dense_top_k: int | None = None,
    domain_adapter_registry: DomainAdapterRegistry | None = None,
    domain_review_policy: DomainReviewPolicy | None = None,
    build_result_factory: BuildResultFactory | None = None,
    runtime_evidence_factory: RuntimeEvidenceFactory | None = None,
    allow_game_launch: bool = False,
    proposal_git_binding_factory: ProposalGitBindingFactory | None = None,
    use_default_proposal_git_binding: bool = True,
    evidence_pack_factory: EvidencePackFactory | None = None,
    planning_target_selection_factory: PlanningTargetSelectionFactory | None = None,
    patch_change_factory: PatchChangeFactory | None = None,
    proposal_review_context_factory: ProposalReviewContextFactory | None = None,
    proposal_source_snippet_char_limit: int = 6000,
    proposal_critic: ProposalCritic | None = None,
    critic_requirement: CriticRequirement = "not_required",
    dry_run_factory: PatchExecutionFactory | None = None,
    isolated_apply_factory: PatchExecutionFactory | None = None,
    stop_after_evidence: bool = False,
    stop_after_build: bool = False,
    persist_build_runtime_handoff: bool = False,
    build_checkpoint_exact_revision_upstream_types: frozenset[str] = frozenset(),
    edit_authority_request_ref: ArtifactRef | None = None,
    repair_planning_authority_request_ref: ArtifactRef | None = None,
    evaluation_context_factory: EvaluationContextFactory | None = None,
    require_external_evaluation: bool = False,
    require_review_approval_for_dry_run: bool = False,
    enable_dynamic_review_interrupt: bool = False,
    checkpointer: Any | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    feature_spec_revision_projection: Mapping[str, Any] | None = None,
    domain_feature_spec_factory: Callable[..., FeatureSpec] | None = None,
    approved_feature_spec_revision_payload_hash: str | None = None,
    approved_feature_spec_revision_projection_hash: str | None = None,
    verification_operation: VerificationOperationEnvelope | None = None,
    product_execution_mode: ProductExecutionGraphModeV1 | None = None,
    planning_continuation_mode: PlanningContinuationInputV1 | None = None,
    patch_continuation_mode: PatchContinuationInputV1 | None = None,
    patch_upstream_request: Mapping[str, Any] | None = None,
    enable_patch_upstream_replanning: bool = False,
    product_patch_upstream_execution: Any = None,
    task_agent_execution: Any = None,
    patch_upstream_evidence_pack_factory: EvidencePackFactory | None = None,
    patch_upstream_planning_target_selection_factory: PlanningTargetSelectionFactory | None = None,
    initial_findings_reassessment_factory: EvidencePackFactory | None = None,
    patch_replanning_context: Any = None,
    patch_continuation_max_model_calls: int = 1,
    patch_continuation_attempt_recorder: (
        PatchContinuationAttemptRecorder | None
    ) = None,
):
    """Build a compiled LangGraph graph with real planning/review routing.

    Planning, evidence retrieval, implementation planning, patch proposal, and
    review are real controlled nodes. Dry-run defaults to the no-write fake path
    and can use an injected exact no-write adapter; build and runtime evidence
    can be injected for controlled contract tests.
    """

    if (
        enable_dynamic_review_interrupt
        or edit_authority_request_ref is not None
        or repair_planning_authority_request_ref is not None
    ) and checkpointer is None:
        raise ValueError(
            "Human approval interrupts require a LangGraph checkpointer."
        )
    validated_verification_operation = (
        None
        if verification_operation is None
        else validate_verification_operation_envelope(verification_operation)
    )
    validated_product_execution_mode = product_execution_mode
    if validated_product_execution_mode is not None and not isinstance(
        validated_product_execution_mode, ProductExecutionGraphModeV1
    ):
        raise ProductExecutionContractError(
            "product_execution_mode must be ProductExecutionGraphModeV1.",
            side_effects_may_have_started=False,
        )
    if task_agent_execution is not None:
        from src.mod_agent.task_agent_execution import TaskAgentExecution

        if (not isinstance(task_agent_execution, TaskAgentExecution) or validated_product_execution_mode is None
                or product_patch_upstream_execution is not None or enable_patch_upstream_replanning
                or patch_continuation_mode is not None or patch_replanning_context is not None
                or patch_change_factory is None):
            raise ProductExecutionContractError("Task execution requires its explicit Product-only technical owner.",
                side_effects_may_have_started=False)
    validated_planning_continuation_mode = planning_continuation_mode
    if validated_planning_continuation_mode is not None and not isinstance(
        validated_planning_continuation_mode,
        PlanningContinuationInputV1,
    ):
        raise ValueError(
            "planning_continuation_mode must be PlanningContinuationInputV1."
        )
    validated_patch_continuation_mode = patch_continuation_mode
    if validated_patch_continuation_mode is not None and not isinstance(
        validated_patch_continuation_mode,
        PatchContinuationInputV1,
    ):
        raise ValueError(
            "patch_continuation_mode must be PatchContinuationInputV1."
        )
    saved_patch_reassembly = (
        validated_patch_continuation_mode is not None
        and validated_patch_continuation_mode.reuses_model_output
    )
    if saved_patch_reassembly and (
        patch_change_factory is not None
        or patch_upstream_request is not None
        or enable_patch_upstream_replanning
        or patch_replanning_context is not None
        or validated_patch_continuation_mode.repair_parent_proposal_ref is not None
        or type(patch_continuation_max_model_calls) is not int
        or patch_continuation_max_model_calls != 0
    ):
        raise ValueError(
            "Saved output reassembly requires zero new model calls and no model, Repair, or upstream adapter."
        )
    active_controlled_modes = sum(
        value is not None
        for value in (
            validated_product_execution_mode,
            validated_verification_operation,
            validated_planning_continuation_mode,
            validated_patch_continuation_mode,
        )
    )
    if active_controlled_modes > 1:
        raise ValueError(
            "Product, verification, Planning continuation, and Patch "
            "continuation modes are mutually exclusive."
        )
    if (
        validated_product_execution_mode is not None
        and validated_verification_operation is not None
    ):
        raise ProductExecutionContractError(
            "Product and verification operation modes cannot share one Graph.",
            side_effects_may_have_started=False,
        )
    if validated_planning_continuation_mode is not None:
        incompatible_values = (
            settings,
            knowledge_database,
            collection_id,
            collection_resolver,
            dense_provider,
            dense_top_k,
            domain_adapter_registry,
            domain_review_policy,
            build_result_factory,
            runtime_evidence_factory,
            proposal_git_binding_factory,
            evidence_pack_factory,
            planning_target_selection_factory,
            patch_change_factory,
            proposal_review_context_factory,
            proposal_critic,
            dry_run_factory,
            isolated_apply_factory,
            edit_authority_request_ref,
            repair_planning_authority_request_ref,
            evaluation_context_factory,
            checkpointer,
            feature_spec_revision_projection,
            approved_feature_spec_revision_payload_hash,
            approved_feature_spec_revision_projection_hash,
            validated_verification_operation,
            validated_product_execution_mode,
            validated_patch_continuation_mode,
            patch_continuation_attempt_recorder,
        )
        incompatible_flags = (
            dense_enabled,
            dense_required,
            allow_game_launch,
            use_default_proposal_git_binding,
            stop_after_evidence,
            stop_after_build,
            persist_build_runtime_handoff,
            bool(build_checkpoint_exact_revision_upstream_types),
            require_external_evaluation,
            require_review_approval_for_dry_run,
            enable_dynamic_review_interrupt,
            bool(interrupt_before),
            bool(interrupt_after),
            critic_requirement != "not_required",
            proposal_source_snippet_char_limit != 6000,
            patch_continuation_max_model_calls != 1,
        )
        if any(value is not None for value in incompatible_values) or any(
            incompatible_flags
        ):
            raise ValueError(
                "Planning continuation cannot receive unrelated workflow, "
                "model, effect, approval, or stopping controls."
            )
    if type(enable_patch_upstream_replanning) is not bool:
        raise ValueError("Automatic upstream replanning must be an explicit boolean.")
    upstream_replanning = patch_upstream_request is not None or enable_patch_upstream_replanning
    if initial_findings_reassessment_factory is not None and (
            validated_product_execution_mode is None or not callable(initial_findings_reassessment_factory)):
        raise ValueError("Initial Findings reassessment requires a Product execution provider.")
    if enable_patch_upstream_replanning:
        if patch_upstream_request is not None:
            raise ValueError("Automatic and explicit upstream entries cannot be combined.")
        if product_patch_upstream_execution is not None:
            if validated_product_execution_mode is None or patch_replanning_context is not None:
                raise ValueError("Product upstream execution requires its own Product Graph mode.")
            if any(not callable(getattr(product_patch_upstream_execution, method, None))
                   for method in ("validate_upstream", "prepare_assessment", "after_plan")):
                raise ValueError("Product upstream execution requires its current worker owner.")
        else:
            if validated_patch_continuation_mode is None:
                raise ValueError("Automatic upstream return requires a controlled Product or Patch entry.")
            if preflight_patch_continuation(validated_patch_continuation_mode, artifact_store=artifact_store).repair_context is None:
                raise ValueError("Automatic upstream return currently requires bound Repair C1/F1 input.")
            if not isinstance(planning_target_selection_factory, ModelPlanningTargetSelector):
                raise ValueError("Automatic upstream return requires the model-owned Planning provider.")
    elif any(value is not None for value in (product_patch_upstream_execution,
            patch_upstream_evidence_pack_factory, patch_upstream_planning_target_selection_factory)):
        raise ValueError("Product upstream adapters require enabled automatic upstream execution.")
    if patch_upstream_request is not None:
        from src.mod_agent.repair.upstream import validate_patch_upstream_selection
        if validated_patch_continuation_mode is None:
            raise ValueError("An upstream assessment requires the exact existing Patch continuation input.")
        _, validated_upstream = validate_patch_upstream_selection(validated_patch_continuation_mode,
            patch_upstream_request, artifact_store=artifact_store)
        patch_upstream_request = copy.deepcopy(dict(patch_upstream_request))
        if not isinstance(planning_target_selection_factory, ModelPlanningTargetSelector):
            raise ValueError("Upstream assessment requires model-owned Planning target selection.")
        from src.mod_agent.evidence.reassessment import RequirementCoverageReassessmentFactory
        if validated_upstream["required_owner"] == "evidence" and not isinstance(
                evidence_pack_factory, RequirementCoverageReassessmentFactory):
            raise ValueError("Evidence return requires the complete-coverage reassessment provider.")
    if validated_patch_continuation_mode is not None:
        if (
            isinstance(patch_continuation_max_model_calls, bool)
            or not isinstance(patch_continuation_max_model_calls, int)
            or (not saved_patch_reassembly and not 1 <= patch_continuation_max_model_calls <= 5)
        ):
            raise ValueError(
                "Patch continuation model-call budget must be between 1 and 5."
            )
        incompatible_values = (
            settings,
            knowledge_database,
            collection_id,
            collection_resolver,
            dense_provider,
            dense_top_k,
            domain_adapter_registry,
            domain_review_policy,
            build_result_factory,
            runtime_evidence_factory,
            proposal_git_binding_factory,
            evidence_pack_factory if not upstream_replanning else None,
            planning_target_selection_factory if not upstream_replanning else None,
            proposal_review_context_factory,
            proposal_critic,
            isolated_apply_factory,
            edit_authority_request_ref,
            repair_planning_authority_request_ref,
            evaluation_context_factory,
            checkpointer,
            feature_spec_revision_projection,
            approved_feature_spec_revision_payload_hash,
            approved_feature_spec_revision_projection_hash,
            validated_verification_operation,
            validated_product_execution_mode,
            validated_planning_continuation_mode,
        )
        incompatible_flags = (
            dense_enabled,
            dense_required,
            allow_game_launch,
            use_default_proposal_git_binding,
            stop_after_evidence,
            stop_after_build,
            persist_build_runtime_handoff,
            bool(build_checkpoint_exact_revision_upstream_types),
            require_external_evaluation,
            require_review_approval_for_dry_run,
            enable_dynamic_review_interrupt,
            bool(interrupt_before),
            bool(interrupt_after),
            critic_requirement != "not_required",
        )
        if any(value is not None for value in incompatible_values) or any(
            incompatible_flags
        ):
            raise ValueError(
                "Patch continuation cannot receive unrelated workflow, "
                "approval, Apply, Build, Runtime, or repair controls."
            )
        if patch_upstream_request is None and (
            (patch_change_factory is None and not saved_patch_reassembly) or dry_run_factory is None
        ):
            raise ValueError(
                "Patch continuation requires explicit Patch and no-write "
                "Dry-run adapters."
            )
        if patch_upstream_request is None and patch_continuation_attempt_recorder is None:
            raise ValueError(
                "Patch continuation requires a durable attempt recorder."
            )
    elif patch_continuation_attempt_recorder is not None:
        raise ValueError(
            "Patch continuation attempt recorder requires patch_continuation_mode."
        )
    elif patch_continuation_max_model_calls != 1:
        raise ValueError(
            "Patch continuation model-call budget requires patch continuation mode."
        )
    if validated_product_execution_mode is not None:
        if (
            stop_after_evidence
            or stop_after_build
            or persist_build_runtime_handoff
            or build_checkpoint_exact_revision_upstream_types
            or enable_dynamic_review_interrupt
            or edit_authority_request_ref is not None
            or repair_planning_authority_request_ref is not None
            or feature_spec_revision_projection is not None
            or approved_feature_spec_revision_payload_hash is not None
            or approved_feature_spec_revision_projection_hash is not None
            or interrupt_before
            or interrupt_after
        ):
            raise ProductExecutionContractError(
                "Product Graph routing comes from its typed request, not legacy "
                "stopping, authority, revision, or interrupt controls.",
                side_effects_may_have_started=False,
            )
        if build_result_factory is None:
            raise ProductExecutionContractError(
                "Product execution requires an explicit Build adapter.",
                side_effects_may_have_started=False,
            )
        if evidence_pack_factory is None or patch_change_factory is None:
            raise ProductExecutionContractError(
                "Product execution requires explicit Evidence and Patch adapters.",
                side_effects_may_have_started=False,
            )
        if dry_run_factory is None or isolated_apply_factory is None:
            raise ProductExecutionContractError(
                "Product execution requires exact Dry-run and Isolated Apply adapters.",
                side_effects_may_have_started=False,
            )
        request_scope = (
            validated_product_execution_mode.request.requested_completion_scope
        )
        if (runtime_evidence_factory is None) is not (request_scope == "compile_only"):
            raise ProductExecutionContractError(
                "Product Runtime adapter topology disagrees with completion scope.",
                side_effects_may_have_started=False,
            )
        stop_after_build = request_scope == "compile_only"
        persist_build_runtime_handoff = request_scope == "compile_only"
    operation_declaration = (
        None
        if validated_verification_operation is None
        else resolve_verification_operation(validated_verification_operation)
    )
    operation_controls = _verification_graph_controls(operation_declaration)
    operation_handler = operation_controls.handler
    if operation_declaration is not None:
        if (
            edit_authority_request_ref is not None
            or repair_planning_authority_request_ref is not None
            or feature_spec_revision_projection is not None
        ):
            raise ValueError(
                "Verification operation and another controlled Graph entry "
                "cannot share one invocation."
            )
        if (
            stop_after_evidence
            or stop_after_build
            or persist_build_runtime_handoff
            or build_checkpoint_exact_revision_upstream_types
            or allow_game_launch
            or interrupt_before
            or interrupt_after
        ):
            raise ValueError(
                "Verification operation routing, side-effect scope, stopping, "
                "checkpointing, and interrupts come from its declaration, not "
                "generic Graph controls."
            )
        if operation_handler == "verification_build":
            if build_result_factory is None:
                raise ValueError(
                    "Verification Build requires an explicit build_result_factory."
                )
            if runtime_evidence_factory is not None:
                raise ValueError(
                    "Verification Build cannot receive Runtime effect adapters."
                )
            if build_checkpoint_exact_revision_upstream_types:
                raise ValueError(
                    "Verification Build uses its declared fresh FeatureSpec and "
                    "EvidencePack lineage, not mutation exact-revision roles."
                )
        elif operation_handler == "runtime_execution":
            if build_result_factory is not None:
                raise ValueError(
                    "Runtime execution cannot receive a Build effect adapter."
                )
            if runtime_evidence_factory is None:
                raise ValueError(
                    "Runtime execution requires an explicit runtime_evidence_factory."
                )
        elif operation_handler in {
            "runtime_terminalization",
            "runtime_evidence_consolidation",
        }:
            if (
                build_result_factory is not None
                or runtime_evidence_factory is not None
                or persist_build_runtime_handoff
            ):
                raise ValueError(
                    "Store-only verification operations cannot receive effect "
                    "adapters or Build handoff configuration."
                )
        else:
            raise ValueError("Verification operation Graph handler is unsupported.")
    revision_configuration = (
        feature_spec_revision_projection,
        approved_feature_spec_revision_payload_hash,
        approved_feature_spec_revision_projection_hash,
    )
    if any(value is not None for value in revision_configuration) and not all(
        value is not None for value in revision_configuration
    ):
        raise ValueError(
            "FeatureSpec revision materialization requires the complete "
            "projection and both approved hashes."
        )
    if feature_spec_revision_projection is not None:
        if checkpointer is None:
            raise ValueError(
                "FeatureSpec revision materialization requires a LangGraph "
                "checkpointer for controlled node-level resume."
            )
        if "draft_feature_spec" not in set(interrupt_after or []):
            raise ValueError(
                "FeatureSpec revision materialization must interrupt after "
                "draft_feature_spec before any downstream node."
            )
    if (
        edit_authority_request_ref is not None
        and repair_planning_authority_request_ref is not None
    ):
        raise ValueError(
            "Edit Authority and Repair planning approval require separate "
            "graph entries."
        )
    if persist_build_runtime_handoff and not stop_after_build:
        raise ValueError(
            "Build-to-Runtime handoff persistence requires stop_after_build=True."
        )
    if (
        build_checkpoint_exact_revision_upstream_types
        and not persist_build_runtime_handoff
    ):
        raise ValueError(
            "Build checkpoint exact revision roles require persisted "
            "Build-to-Runtime handoff."
        )
    unsupported_checkpoint_exact_types = (
        set(build_checkpoint_exact_revision_upstream_types)
        - BUILD_CHECKPOINT_EXACT_REVISION_TYPES
    )
    if unsupported_checkpoint_exact_types:
        raise ValueError(
            "Build checkpoint exact revision roles are unsupported: "
            f"{sorted(unsupported_checkpoint_exact_types)}"
        )
    missing_checkpoint_exact_types = (
        BUILD_CHECKPOINT_EXACT_REVISION_TYPES
        - set(build_checkpoint_exact_revision_upstream_types)
        if build_checkpoint_exact_revision_upstream_types
        else set()
    )
    if missing_checkpoint_exact_types:
        raise ValueError(
            "Build checkpoint exact revision roles are incomplete: "
            f"{sorted(missing_checkpoint_exact_types)}"
        )
    if require_external_evaluation and evaluation_context_factory is None:
        raise ValueError("require_external_evaluation requires evaluation_context_factory")
    if (
        isinstance(proposal_source_snippet_char_limit, bool)
        or not isinstance(proposal_source_snippet_char_limit, int)
        or proposal_source_snippet_char_limit <= 0
    ):
        raise ValueError("proposal_source_snippet_char_limit must be a positive integer.")
    if critic_requirement not in {"not_required", "required"}:
        raise ValueError(
            "critic_requirement must be exactly 'not_required' or 'required'."
        )
    if critic_requirement == "not_required" and proposal_critic is not None:
        raise ValueError(
            "proposal_critic requires critic_requirement='required'; "
            "implicit triggering is not supported."
        )
    if domain_review_policy is not None and domain_adapter_registry is not None:
        raise ValueError(
            "domain_review_policy and domain_adapter_registry cannot both "
            "be supplied."
        )

    verification_build_continuation = (
        _verification_build_continuation(validated_verification_operation)
        if operation_handler == "verification_build"
        and validated_verification_operation is not None
        else None
    )
    effective_interrupt_after = list(interrupt_after or ())
    if (
        operation_controls.stop_after_runtime_evidence
        and "collect_runtime_evidence" not in effective_interrupt_after
    ):
        effective_interrupt_after.append("collect_runtime_evidence")

    stub_runner = StubNodeRunner(artifact_store)
    effective_domain_registry = domain_adapter_registry or _default_domain_adapter_registry()
    effective_runtime_evidence_factory = (
        None
        if operation_handler == "runtime_evidence_consolidation"
        else (
            runtime_evidence_factory
            or _runtime_evidence_factory_from_settings(
                settings,
                allow_game_launch=allow_game_launch,
            )
        )
    )
    effective_proposal_git_binding_factory = proposal_git_binding_factory
    if effective_proposal_git_binding_factory is None and use_default_proposal_git_binding:
        effective_proposal_git_binding_factory = _default_proposal_git_binding_factory
    graph = StateGraph(ModAgentState)

    if verification_build_continuation is not None:
        graph.add_node(
            "verification_only_admission",
            _make_real_graph_node(
                "verification_only_admission",
                lambda state: _run_verification_only_admission(
                    state,
                    artifact_store,
                    validated_verification_operation,
                ),
            ),
        )

    graph.add_node(
        "plan_request",
        _make_real_graph_node(
            "plan_request",
            lambda state: _run_plan_request(state, artifact_store),
        ),
    )
    graph.add_node(
        "draft_feature_spec",
        _make_real_graph_node(
            "draft_feature_spec",
            lambda state: _run_draft_feature_spec(
                state,
                artifact_store,
                domain_feature_spec_factory=domain_feature_spec_factory,
                revision_projection=feature_spec_revision_projection,
                approved_payload_hash=approved_feature_spec_revision_payload_hash,
                approved_projection_hash=(
                    approved_feature_spec_revision_projection_hash
                ),
            ),
        ),
    )
    graph.add_node(
        "planning_brief",
        _make_real_graph_node(
            "planning_brief",
            lambda state: _run_planning_brief(state, artifact_store),
        ),
    )
    graph.add_node(
        "retrieve_evidence",
        _make_real_graph_node(
            "retrieve_evidence",
            lambda state: _run_retrieve_evidence(
                state,
                artifact_store,
                knowledge_database,
                collection_id,
                collection_resolver,
                dense_provider,
                dense_enabled,
                dense_required,
                dense_top_k,
                (patch_upstream_evidence_pack_factory
                 if product_patch_upstream_execution is not None
                 and state["runtime_summary"].get("patch_upstream_return") else evidence_pack_factory),
                initial_findings_reassessment_factory=initial_findings_reassessment_factory,
                product_execution_mode=validated_product_execution_mode,
                enable_initial_findings_reassessment=(initial_findings_reassessment_factory is not None
                    or product_patch_upstream_execution is not None),
            ),
        ),
    )
    graph.add_node(
        "create_implementation_plan",
        _make_real_graph_node(
            "create_implementation_plan",
            lambda state: _finish_replanned_selection(_run_create_implementation_plan(
                state,
                artifact_store,
                formal_requirement_coverage_required=(
                    validated_product_execution_mode is not None
                    or validated_planning_continuation_mode is not None
                    or upstream_replanning
                ),
                use_store_result=(
                    validated_planning_continuation_mode is not None
                ),
                planning_continuation_mode=(
                    validated_planning_continuation_mode
                ),
                planning_target_selection_factory=(
                    patch_upstream_planning_target_selection_factory
                    if product_patch_upstream_execution is not None
                    and state["runtime_summary"].get("patch_upstream_return")
                    else planning_target_selection_factory
                ),
                patch_continuation_mode=(patch_replanning_context.request
                    if patch_replanning_context is not None else validated_patch_continuation_mode),
                product_patch_upstream_execution=product_patch_upstream_execution,
            ), product_patch_upstream_execution or patch_replanning_context,
                product_execution_mode=validated_product_execution_mode),
        ),
    )
    graph.add_node(
        "generate_patch_proposal",
        _make_real_graph_node(
            "generate_patch_proposal",
            lambda state: _run_generate_patch_with_upstream(
                state,
                artifact_store,
                enable_upstream_replanning=(enable_patch_upstream_replanning
                    and (patch_replanning_context is None or patch_replanning_context.replan_count == 0)),
                upstream_evidence_factory=(patch_upstream_evidence_pack_factory
                    if product_patch_upstream_execution is not None else evidence_pack_factory),
                upstream_planning_factory=(patch_upstream_planning_target_selection_factory
                    if product_patch_upstream_execution is not None else planning_target_selection_factory),
                product_patch_upstream_execution=product_patch_upstream_execution,
                task_agent_execution=task_agent_execution,
                upstream_continues_patch=(patch_replanning_context is not None or product_patch_upstream_execution is not None),
                proposal_git_binding_factory=effective_proposal_git_binding_factory,
                patch_change_factory=(patch_replanning_context.factory if patch_replanning_context is not None else patch_change_factory),
                source_snippet_char_limit=(patch_replanning_context.source_snippet_char_limit
                    if patch_replanning_context is not None else (
                        None if validated_product_execution_mode is not None or saved_patch_reassembly
                        else proposal_source_snippet_char_limit)),
                patch_continuation_mode=(patch_replanning_context.request if patch_replanning_context is not None else validated_patch_continuation_mode),
                patch_continuation_attempt_recorder=(
                    patch_replanning_context.record if patch_replanning_context is not None else patch_continuation_attempt_recorder
                ),
                patch_continuation_max_model_calls=(
                    patch_replanning_context.allocation if patch_replanning_context is not None else patch_continuation_max_model_calls
                ),
            ),
        ),
    )
    graph.add_node(
        "review_proposal",
        _make_real_graph_node(
            "review_proposal",
            lambda state: _run_review_proposal(
                state,
                artifact_store,
                domain_adapter_registry=effective_domain_registry,
                domain_review_policy=domain_review_policy,
                require_executable_patch=patch_change_factory is not None or saved_patch_reassembly,
                proposal_critic=proposal_critic,
                critic_requirement=critic_requirement,
                proposal_review_context_factory=proposal_review_context_factory,
                patch_continuation_attempt_recorder=(
                    patch_continuation_attempt_recorder
                    if validated_patch_continuation_mode is not None
                    else None
                ),
            ),
        ),
    )
    if enable_dynamic_review_interrupt:
        graph.add_node(
            "human_review_gate",
            _make_real_graph_node(
                "human_review_gate",
                lambda state: _run_dynamic_human_review_gate(state, artifact_store),
            ),
        )
    if edit_authority_request_ref is not None:
        graph.add_node(
            "stagger_edit_authority_gate",
            _make_real_graph_node(
                "stagger_edit_authority_gate",
                lambda state: _run_stagger_edit_authority_gate(
                    state,
                    artifact_store,
                    edit_authority_request_ref,
                ),
            ),
        )
    if repair_planning_authority_request_ref is not None:
        graph.add_node(
            "stagger_repair_planning_gate",
            _make_real_graph_node(
                "stagger_repair_planning_gate",
                lambda state: _run_stagger_repair_planning_gate(
                    state,
                    artifact_store,
                    repair_planning_authority_request_ref,
                ),
            ),
        )

    graph.add_node(
        "dry_run_apply",
        _make_real_graph_node(
            "dry_run_apply",
            lambda state: _run_dry_run_apply(
                state,
                artifact_store,
                require_review_approval=require_review_approval_for_dry_run,
                dry_run_factory=dry_run_factory,
            ),
        ),
    )
    if isolated_apply_factory is not None:
        graph.add_node(
            "isolated_apply",
            _make_real_graph_node(
                "isolated_apply",
                lambda state: _run_isolated_apply(
                    state,
                    artifact_store,
                    isolated_apply_factory=isolated_apply_factory,
                ),
            ),
        )
    graph.add_node(
        "build",
        _make_real_graph_node(
            "build",
            lambda state: _run_build(
                state,
                artifact_store,
                build_result_factory,
                verification_operation=validated_verification_operation,
                persist_build_runtime_handoff=(
                    persist_build_runtime_handoff
                    or operation_controls.persist_build_handoff
                ),
                exact_revision_upstream_types=(
                    build_checkpoint_exact_revision_upstream_types
                ),
            ),
        ),
    )
    graph.add_node(
        "collect_runtime_evidence",
        _make_real_graph_node(
            "collect_runtime_evidence",
            lambda state: _run_runtime_evidence(
                state,
                artifact_store,
                effective_runtime_evidence_factory,
                use_store_result=operation_controls.runtime_evidence_store_results,
                verification_operation=validated_verification_operation,
            ),
        ),
    )
    graph.add_node(
        "evaluate",
        _make_real_graph_node(
            "evaluate",
            lambda state: _run_terminal_node_with_boundary(
                state,
                lambda selected: _run_fake_evaluate(
                    selected,
                    artifact_store,
                    evaluation_context_factory,
                    require_external_evaluation,
                    use_store_result=operation_controls.terminal_store_results,
                ),
                failure_stage="eval_report_execution",
                blocker_code="EVAL_REPORT_EXECUTION_FAILED",
                use_boundary=operation_controls.terminal_store_results,
            ),
        ),
    )
    graph.add_node(
        "create_repair_plan",
        _make_real_graph_node(
            "create_repair_plan",
            lambda state: _run_create_repair_plan(
                state,
                artifact_store,
                product_execution_mode=validated_product_execution_mode,
                patch_continuation_mode=validated_patch_continuation_mode,
                patch_continuation_attempt_recorder=patch_continuation_attempt_recorder,
                patch_upstream_request=patch_upstream_request,
                patch_upstream_factory=patch_change_factory if enable_patch_upstream_replanning else None,
                product_patch_upstream_execution=product_patch_upstream_execution,
            ),
        ),
    )
    graph.add_node(
        "resolve_runtime_binding",
        _make_real_graph_node(
            "resolve_runtime_binding",
            lambda state: _run_resolve_runtime_binding(
                state,
                artifact_store,
            ),
        ),
    )
    graph.add_node(
        "done",
        _make_real_graph_node(
            "done",
            lambda state: _run_terminal_node_with_boundary(
                state,
                lambda selected: _run_workflow_summary(
                    selected,
                    artifact_store,
                    use_store_result=operation_controls.terminal_store_results,
                ),
                failure_stage="workflow_summary_execution",
                blocker_code="WORKFLOW_SUMMARY_EXECUTION_FAILED",
                use_boundary=operation_controls.terminal_store_results,
            ),
        ),
    )
    graph.add_node(
        "blocked",
        _make_stub_graph_node(
            "blocked",
            stub_runner,
            use_store_result=operation_controls.terminal_store_results,
        ),
    )

    if (
        operation_handler == "runtime_terminalization"
        and validated_verification_operation is not None
    ):
        _add_traced_conditional_edges(graph,
            START,
            lambda state: _route_runtime_terminal_continuation_entry(
                state,
                artifact_store,
                _verification_runtime_terminal_input(
                    validated_verification_operation
                ),
            ),
            {
                "blocked": "blocked",
                "continue": "evaluate",
            },
        )
    elif (
        operation_handler == "runtime_evidence_consolidation"
        and validated_verification_operation is not None
    ):
        _add_traced_conditional_edges(graph,
            START,
            lambda state: _route_runtime_evidence_consolidation_entry(
                state,
                artifact_store,
                validated_verification_operation,
            ),
            {"continue": "collect_runtime_evidence"},
        )
    elif operation_handler == "verification_build":
        graph.add_edge(START, "verification_only_admission")
    elif (
        operation_handler == "runtime_execution"
        and validated_verification_operation is not None
    ):
        _add_traced_conditional_edges(graph,
            START,
            lambda state: _route_runtime_continuation_entry(
                state,
                artifact_store,
                validated_verification_operation,
            ),
            {"continue": "collect_runtime_evidence"},
        )
    elif validated_planning_continuation_mode is not None:
        _add_traced_conditional_edges(graph,
            START,
            lambda state: _validate_planning_continuation_graph_start(
                state,
                validated_planning_continuation_mode,
                artifact_store,
            ),
            {"continue": "create_implementation_plan"},
        )
    elif validated_patch_continuation_mode is not None:
        _add_traced_conditional_edges(graph,
            START,
            lambda state: _validate_patch_continuation_graph_start(
                state,
                validated_patch_continuation_mode,
                artifact_store,
            ),
            ({"repair": "create_repair_plan"}
             if validated_patch_continuation_mode.repair_parent_proposal_ref is not None
             else {"continue": "generate_patch_proposal"}),
        )
    elif validated_product_execution_mode is not None:
        _add_traced_conditional_edges(graph,
            START,
            lambda state: _validate_product_graph_start(
                state,
                validated_product_execution_mode,
                artifact_store,
            ),
            {"continue": ("generate_patch_proposal" if (validated_product_execution_mode.interrupted_patch_selection is not None
                    or validated_product_execution_mode.pretransport_patch_selection is not None)
                          else "plan_request")},
        )
    elif repair_planning_authority_request_ref is not None:
        graph.add_edge(START, "stagger_repair_planning_gate")
    elif edit_authority_request_ref is not None:
        graph.add_edge(START, "stagger_edit_authority_gate")
    else:
        graph.add_edge(START, "plan_request")
    if operation_handler == "verification_build":
        _add_traced_conditional_edges(graph,
            "verification_only_admission",
            lambda state: "blocked" if state["blockers"] else "continue",
            {"blocked": END, "continue": "build"},
        )
    registry = route_registry()
    _add_traced_conditional_edges(graph,
        "plan_request",
        route_after_plan_request,
        dict(registry["plan_request"]),
    )
    _add_traced_conditional_edges(graph,
        "planning_brief",
        route_after_planning_brief,
        dict(registry["planning_brief"]),
    )
    _add_traced_conditional_edges(graph,
        "draft_feature_spec",
        route_after_feature_spec,
        ({**dict(registry["draft_feature_spec"]), "continue": "generate_patch_proposal"}
            if getattr(task_agent_execution, "protocol", None) == "v2" else dict(registry["draft_feature_spec"])),
    )
    if stop_after_evidence:
        _add_traced_conditional_edges(graph,
            "retrieve_evidence",
            route_after_evidence,
            {"continue": END, "blocked": END},
        )
    else:
        _add_traced_conditional_edges(graph,
            "retrieve_evidence",
            (lambda state: "blocked" if state["blockers"]
                and state["runtime_summary"].get("initial_findings_reassessment")
                and "implementation_plan" not in state["artifacts"] else route_after_evidence(state))
                if validated_product_execution_mode is not None else route_after_evidence,
            dict(registry["retrieve_evidence"]),
        )
    _add_traced_conditional_edges(graph,
        "create_implementation_plan",
        route_after_implementation_plan,
        (
            {"continue": "generate_patch_proposal", "blocked": "blocked"}
            if product_patch_upstream_execution is not None
            else
            {"continue": "generate_patch_proposal", "blocked": END}
            if patch_replanning_context is not None
            else {"continue": END, "blocked": END}
            if validated_planning_continuation_mode is not None or upstream_replanning
            else dict(registry["create_implementation_plan"])
        ),
    )
    _add_traced_conditional_edges(graph,
        "generate_patch_proposal",
        route_after_patch_proposal,
        (
            {"continue": "review_proposal", "blocked": END, **({"upstream": "create_repair_plan"}
                if enable_patch_upstream_replanning else {})}
            if validated_patch_continuation_mode is not None
            else {**dict(registry["generate_patch_proposal"]),
                  **({"upstream": "create_repair_plan"} if product_patch_upstream_execution is not None else {})}
        ),
    )
    _add_traced_conditional_edges(graph,
        "review_proposal",
        route_after_review,
        (
            {"continue": "dry_run_apply", "blocked": END, "repair": END}
            if validated_patch_continuation_mode is not None or getattr(task_agent_execution, "protocol", None) == "v2"
            else _review_route_map(
                registry["review_proposal"],
                enable_dynamic_review_interrupt=enable_dynamic_review_interrupt,
            )
        ),
    )
    if enable_dynamic_review_interrupt:
        _add_traced_conditional_edges(graph,
            "human_review_gate",
            route_after_human_review_gate,
            dict(registry["human_review_gate"]),
        )
    if edit_authority_request_ref is not None:
        graph.add_edge("stagger_edit_authority_gate", END)
    if repair_planning_authority_request_ref is not None:
        graph.add_edge("stagger_repair_planning_gate", END)
    dry_run_routes = dict(registry["dry_run_apply"])
    if isolated_apply_factory is not None:
        dry_run_routes["continue"] = "isolated_apply"
    _add_traced_conditional_edges(graph,
        "dry_run_apply",
        route_after_dry_run,
        (
            {"continue": END, "blocked": END}
            if validated_patch_continuation_mode is not None or getattr(task_agent_execution, "protocol", None) == "v2"
            else dry_run_routes
        ),
    )
    if isolated_apply_factory is not None:
        _add_traced_conditional_edges(graph,
            "isolated_apply",
            route_after_isolated_apply,
            dict(registry["isolated_apply"]),
        )
    if (
        validated_product_execution_mode is not None
        and validated_product_execution_mode.request.requested_completion_scope
        == "compile_only"
    ):
        _add_traced_conditional_edges(graph,
            "build",
            route_after_build,
            {
                "blocked": END,
                "continue": END,
                "repair": registry["build"]["repair"],
            },
        )
    elif stop_after_build or operation_controls.stop_after_build:
        _add_traced_conditional_edges(graph,
            "build",
            route_after_build,
            {"blocked": END, "continue": END, "repair": END},
        )
    else:
        _add_traced_conditional_edges(graph,
            "build",
            route_after_build,
            dict(registry["build"]),
        )
    if operation_controls.stop_after_runtime_evidence:
        graph.add_edge("collect_runtime_evidence", END)
    elif operation_handler == "runtime_execution":
        runtime_routes = dict(registry["collect_runtime_evidence"])
        runtime_routes["stop"] = END
        _add_traced_conditional_edges(graph,
            "collect_runtime_evidence",
            _route_after_typed_runtime_evidence,
            runtime_routes,
        )
    else:
        _add_traced_conditional_edges(graph,
            "collect_runtime_evidence",
            route_after_runtime_evidence,
            dict(registry["collect_runtime_evidence"]),
        )
    _add_traced_conditional_edges(graph,
        "resolve_runtime_binding",
        route_after_binding_resolution,
        dict(registry["resolve_runtime_binding"]),
    )
    _add_traced_conditional_edges(graph,
        "create_repair_plan",
        route_after_repair_plan,
        ({"evidence": "retrieve_evidence", "planning": "create_implementation_plan",
            "blocked": "blocked" if product_patch_upstream_execution is not None else END,
            **({"continue": "generate_patch_proposal"} if enable_patch_upstream_replanning else {})}
         if upstream_replanning else {"continue": "generate_patch_proposal", "blocked": END}
         if validated_patch_continuation_mode is not None
         else dict(registry["create_repair_plan"])),
    )
    _add_traced_conditional_edges(graph,
        "evaluate",
        (
            _route_after_terminal_evaluate
            if operation_controls.terminal_store_results
            else route_after_evaluate
        ),
        (
            {"stop": END, "blocked": "blocked", "continue": "done"}
            if operation_controls.terminal_store_results
            else dict(registry["evaluate"])
        ),
    )
    graph.add_edge("done", END)
    graph.add_edge("blocked", END)
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=effective_interrupt_after,
    )



def build_retained_task_agent_handoff_graph(artifact_store: ArtifactStore, *, admission):
    """Re-enter only the normal proposal, review and no-write dry-run nodes.

    为什么要有它：恢复已通过候选不应重新执行技术模型，也不能绕过正常提案审查或直接Apply。
    """
    from src.mod_agent.local_entry import make_exact_no_write_dry_run_factory
    from src.mod_agent.retained_task_agent_handoff import RetainedHandoffAdmission

    if type(admission) is not RetainedHandoffAdmission or artifact_store.root.resolve() != admission.root / "a":
        raise ValueError("RETAINED_TASK_AGENT_GRAPH_ADMISSION_REQUIRED")

    def proposal(state):
        current = admission.revalidate()
        if (state["run_id"] != current.manifest["run_id"]
                or state["request_id"] != current.manifest["request_id"]):
            raise ValueError("RETAINED_TASK_AGENT_GRAPH_IDENTITY_CHANGED")
        selected = _copy_state(state)
        selected["feature_id"] = current.outcome.final_context.feature_spec_ref.feature_id
        selected["current_stage"] = "implementation_plan"
        for role in ("feature_spec", "evidence_pack", "implementation_plan"):
            _record_artifact(selected, role, getattr(current.outcome.final_context, role + "_ref"))
        return _run_generate_patch_proposal(selected, artifact_store,
            source_snippet_char_limit=None, retained_task_agent_handoff=current)

    graph = StateGraph(ModAgentState)
    graph.add_node("generate_patch_proposal", _make_real_graph_node("generate_patch_proposal", proposal))
    graph.add_node("review_proposal", _make_real_graph_node("review_proposal", lambda state:
        _run_review_proposal(state, artifact_store, require_executable_patch=True)))
    graph.add_node("dry_run_apply", _make_real_graph_node("dry_run_apply", lambda state:
        _run_dry_run_apply(state, artifact_store,
            dry_run_factory=make_exact_no_write_dry_run_factory(admission.root))))
    graph.add_edge(START, "generate_patch_proposal")
    _add_traced_conditional_edges(graph, "generate_patch_proposal",
        lambda state: "stop" if state["blockers"] else "continue", {"stop": END, "continue": "review_proposal"})
    _add_traced_conditional_edges(graph, "review_proposal",
        lambda state: "stop" if state["blockers"] else "continue", {"stop": END, "continue": "dry_run_apply"})
    graph.add_edge("dry_run_apply", END)
    return graph.compile()


def build_retained_visualsignal_correction_graph(
    artifact_store: ArtifactStore, *, correction, verifier, expected_identity_hash: str,
):
    """One retained binding revision followed by the existing mutation EvalGate.

    Scope: no execution/review/Build/Apply/game nodes are included.
    为什么要有它：表示修正须重新经过正常评估门禁，不能以重跑Runtime或verification_only绕过已存在的失败。
    """
    from src.mod_agent.runtime.retained_visualsignal import (
        RetainedVisualSignalCorrection,
        prepare_correction,
    )

    if type(correction) is not RetainedVisualSignalCorrection or correction.identity_hash != expected_identity_hash:
        raise ValueError("RETAINED_CORRECTION_ACTION_IDENTITY_CHANGED")

    def correct_binding(state):
        prepared = prepare_correction(artifact_store, correction, verifier)
        if correction.identity_hash != expected_identity_hash:
            raise ValueError("RETAINED_CORRECTION_ACTION_IDENTITY_CHANGED")
        next_state = ModAgentState(**copy.deepcopy(prepared["source_snapshot"]))
        next_state["trace_events"] = list(state["trace_events"])
        _record_artifact(next_state, "workflow_checkpoint", correction.checkpoint_ref)
        _record_artifact(next_state, "resume_report", correction.accepted_resume_report_ref)
        payload = prepared["payload"]
        _validate_runtime_evidence_result_identity(next_state, artifact_store, payload,
            build=artifact_store.read(correction.build_result_ref)["data"],
            build_artifact_ref=correction.build_result_ref, requires_verification_only_input=False)
        artifact_store.preflight_atomic_commit(feature_id=correction.checkpoint_ref.feature_id,
            artifact_types=("runtime_evidence", "eval_report", "workflow_summary", "blocked_report"))
        write_result = artifact_store.write_with_result(feature_id=correction.checkpoint_ref.feature_id,
            artifact_type="runtime_evidence", payload=payload, status="success",
            upstream_refs=[correction.build_result_ref, correction.feature_spec_ref, correction.prior_runtime_ref,
                           correction.prior_eval_ref, correction.prior_blocked_ref],
            exact_revision_upstream_types={"runtime_evidence", "eval_report", "blocked_report"})
        next_state["current_stage"] = "runtime_evidence"
        next_state["next_stage"] = "evaluation"
        next_state["runtime_summary"] = _runtime_summary_from_payload(payload)
        ready = write_result.outcome == "committed" and write_result.artifact_ref is not None
        next_state["runtime_summary"].update(runtime_evidence_ready=ready,
            runtime_evidence_store_result=write_result.to_dict(),
            retained_binding_correction=payload["retained_binding_correction"],
            platform_failure_stage=None if ready else "runtime_evidence_correction_store")
        if write_result.artifact_ref is not None:
            _record_artifact(next_state, "runtime_evidence", write_result.artifact_ref)
        if not ready:
            next_state["blockers"].append({"stage": "runtime_evidence", "code": "RETAINED_CORRECTION_NOT_COMMITTED",
                                           "message": "Correction revision did not reach a committed Store outcome."})
        return next_state

    graph = StateGraph(ModAgentState)
    graph.add_node("correct_retained_runtime_binding", _make_real_graph_node("correct_retained_runtime_binding", correct_binding))
    graph.add_node("evaluate", _make_real_graph_node("evaluate", lambda state: _run_terminal_node_with_boundary(
        state, lambda selected: _run_fake_evaluate(selected, artifact_store, use_store_result=True),
        failure_stage="eval_report_execution", blocker_code="EVAL_REPORT_EXECUTION_FAILED", use_boundary=True)))
    graph.add_node("done", _make_real_graph_node("done", lambda state: _run_terminal_node_with_boundary(
        state, lambda selected: _run_workflow_summary(selected, artifact_store, use_store_result=True),
        failure_stage="workflow_summary_execution", blocker_code="WORKFLOW_SUMMARY_EXECUTION_FAILED", use_boundary=True)))
    graph.add_node("blocked", _make_stub_graph_node("blocked", StubNodeRunner(artifact_store), use_store_result=True))
    graph.add_edge(START, "correct_retained_runtime_binding")
    _add_traced_conditional_edges(graph, "correct_retained_runtime_binding",
        lambda state: "continue" if state["runtime_summary"].get("runtime_evidence_ready") else "stop",
        {"continue": "evaluate", "stop": END})
    _add_traced_conditional_edges(graph, "evaluate", _route_after_terminal_evaluate,
        {"continue": "done", "blocked": "blocked", "stop": END})
    graph.add_edge("done", END)
    graph.add_edge("blocked", END)
    return graph.compile()


def _route_input_summary(state: ModAgentState, from_node: str) -> dict[str, Any]:
    """Project selector fields; never copy blocker messages or model rationale."""
    blockers = state.get("blockers", [])
    values = {
        "blocker_count": len(blockers),
        "blocker_codes": [row.get("code") for row in blockers if isinstance(row, Mapping)],
        "blocker_stages": [row.get("stage") for row in blockers if isinstance(row, Mapping)],
    }
    if from_node == "plan_request":
        values["execution_path"] = state.get("execution_path")
    elif from_node == "retrieve_evidence":
        values["evidence_pack_present"] = "evidence_pack" in state.get("artifacts", {})
    elif from_node == "review_proposal":
        review = state.get("review_summary", {})
        values.update(review_allowed_next_stage=review.get("allowed_next_stage"),
                      review_decision=review.get("decision"))
    elif from_node == "generate_patch_proposal":
        pending = state.get("runtime_summary", {}).get("patch_upstream_pending")
        values.update(
            upstream_pending_present=isinstance(pending, Mapping),
            upstream_plan_hash_matches=(isinstance(pending, Mapping) and
                pending.get("source_plan_hash") == state.get("artifact_hashes", {}).get("implementation_plan")),
        )
    elif from_node == "create_repair_plan":
        upstream = state.get("runtime_summary", {}).get("patch_upstream_return")
        values.update(upstream_return_present=isinstance(upstream, Mapping))
        if isinstance(upstream, Mapping):
            values.update(
                upstream_assessment_only=upstream.get("assessment_only"),
                upstream_required_owner=upstream.get("required_owner"),
                upstream_repair_plan_hash_matches=(upstream.get("repair_plan_hash")
                    == state.get("artifact_hashes", {}).get("repair_plan")),
            )
    elif from_node in {"build", "collect_runtime_evidence", "evaluate"}:
        build = state.get("build_summary", {})
        runtime = state.get("runtime_summary", {})
        values.update(build_status=build.get("status"), input_execution_mode=build.get("input_execution_mode"))
        if from_node != "build":
            values["platform_failure_stage"] = runtime.get("platform_failure_stage")
        if from_node == "collect_runtime_evidence":
            failure = runtime.get("failure_ownership")
            values.update(
                runtime_acceptance_verified=runtime.get("acceptance_verified"),
                has_structured_diagnosis=runtime.get("has_structured_diagnosis"),
                runtime_artifact_present="runtime_evidence" in state.get("artifacts", {}),
                failure_owner=failure.get("owner") if isinstance(failure, Mapping) else None,
                source_repair_allowed=failure.get("source_repair_allowed") if isinstance(failure, Mapping) else None,
            )
    return values


def _add_traced_conditional_edges(graph, from_node, selector, route_map):
    """Observe only a real selector invocation; keep its return and route map."""
    execution_trace = current_trace()

    @wraps(selector)
    def observed(state):
        if (current_trace() is None and execution_trace is not None
                and execution_trace.context.finished_at is None):
            with execution_trace.activate():
                return select(state)
        return select(state)

    def select(state):
        with trace_operation("workflow.route", kind="event", attributes={
            "workflow_node": from_node,
            "from_node": from_node,
            "route_selector": getattr(selector, "__name__", type(selector).__name__),
            "decision_summary_origin": "deterministic_route_selector",
        }) as span:
            try:
                inputs = _route_input_summary(state, from_node)
                update_observation(span, route_inputs=inputs,
                    route_input_sha256=payload_sha256(inputs),
                    route_input_summary_scope="structured_selector_fields",
                    artifact_refs=[{"artifact_type": kind, "artifact_hash": digest}
                        for kind, digest in state.get("artifact_hashes", {}).items()])
            except Exception:
                update_observation(span, route_input_summary_status="unavailable")
            selected = selector(state)
            try:
                update_observation(span, selected_route=selected,
                    selected_next_node=route_map.get(selected),
                    decision_summary=f"路由 {from_node} 选择 {selected}，映射到 {route_map.get(selected)}；目标是否执行由后续独立观察确认。")
            except Exception:
                update_observation(span, route_output_summary_status="unavailable")
            return selected

    return graph.add_conditional_edges(from_node, observed, route_map)

def _review_route_map(
    route_map: dict[str, str] | Any,
    *,
    enable_dynamic_review_interrupt: bool,
) -> dict[str, str]:
    mapped = dict(route_map)
    if enable_dynamic_review_interrupt:
        mapped["continue"] = "human_review_gate"
    return mapped


def _make_real_graph_node(node_name: str, run: GraphNode) -> GraphNode:
    execution_trace = current_trace()
    def _node(state: ModAgentState) -> ModAgentState:
        # Preserve LangGraph's runtime ContextVars (especially interrupt/resume).
        # Only bind our trace when a worker does not inherit it automatically.
        if (current_trace() is None and execution_trace is not None
                and execution_trace.context.finished_at is None):
            with execution_trace.activate():
                return _traced_node(state)
        return _traced_node(state)

    def _traced_node(state: ModAgentState) -> ModAgentState:
        names = {"generate_patch_proposal": "patch.worker", "review_proposal": "proposal.review",
                 "dry_run_apply": "patch.dry_run"}
        with trace_operation(names.get(node_name, "workflow." + node_name),
                attributes={"workflow_node": node_name}) as observation:
            result = _execute_node(state)
            paths = set(_node_artifact_refs(result, node_name))
            observation.update(workflow_status=_node_status(result, node_name), artifact_refs=[
                {"artifact_type": kind, "artifact_hash": result["artifact_hashes"].get(kind)}
                for kind, path in result["artifacts"].items() if path in paths])
            return result

    def _execute_node(state: ModAgentState) -> ModAgentState:
        started = _trace_event(
            state=state,
            node_name=node_name,
            event_type="node_started",
            status="started",
            artifact_refs=[],
            summary=f"{node_name} started.",
        )
        state_with_start = _copy_state_with_trace(state, started)
        try:
            result = run(state_with_start)
        except Exception as exc:
            setattr(exc, "_mod_agent_graph_node", node_name)
            setattr(exc, "_mod_agent_graph_state", state_with_start)
            raise
        finished = _trace_event(
            state=result,
            node_name=node_name,
            event_type="node_finished",
            status=_node_status(result, node_name),
            artifact_refs=_node_artifact_refs(result, node_name),
            summary=f"{node_name} completed.",
            details=_node_trace_details(result, node_name),
        )
        return _copy_state_with_trace(result, finished)

    return _node


def _make_stub_graph_node(
    node_name: str,
    runner: StubNodeRunner,
    *,
    use_store_result: bool = False,
) -> GraphNode:
    def _node(state: ModAgentState) -> ModAgentState:
        started = _trace_event(
            state=state,
            node_name=node_name,
            event_type="node_started",
            status="started",
            artifact_refs=[],
            summary=f"{node_name} started.",
        )
        state_with_start = _copy_state_with_trace(state, started)
        additional_upstreams: tuple[ArtifactRef, ...] = ()
        if use_store_result and node_name == "blocked":
            source_type = (
                "eval_report"
                if "eval_report" in state_with_start["artifacts"]
                else "runtime_evidence"
            )
            additional_upstreams = (
                _artifact_ref_from_state(state_with_start, source_type),
            )
        try:
            result = runner.run(
                node_name,
                state_with_start,
                use_store_result=use_store_result,
                additional_upstream_refs=additional_upstreams,
            )
        except Exception as exc:
            setattr(exc, "_mod_agent_graph_node", node_name)
            raise
        finished_state = _copy_state_with_trace(
            result.state,
            result.trace_event.to_dict(),
        )
        return finished_state

    return _node


def _run_terminal_node_with_boundary(
    state: ModAgentState,
    run: GraphNode,
    *,
    failure_stage: str,
    blocker_code: str,
    use_boundary: bool,
) -> ModAgentState:
    """Normalize pre-Store terminal node failures for the public result path."""

    if not use_boundary:
        return run(state)
    try:
        return run(state)
    except Exception as exc:
        next_state = _copy_state(state)
        next_state["runtime_summary"].update(
            {
                "platform_failure_stage": failure_stage,
                "terminal_ready": False,
            }
        )
        next_state["blockers"].append(
            {
                "stage": "evaluation" if failure_stage.startswith("eval_") else "done",
                "code": blocker_code,
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return next_state


def _run_plan_request(
    state: ModAgentState,
    artifact_store: ArtifactStore,
) -> ModAgentState:
    plan = RequestPlanner().plan(
        state["user_request"],
        target_mod=state["target_mod"],
    )
    artifact_ref = artifact_store.write(
        feature_id=plan.feature_id,
        artifact_type="request_planning_report",
        payload=plan.to_artifact_payload(),
        status="blocked" if plan.blocked else "success",
    )
    next_state = _copy_state(state)
    next_state["feature_id"] = plan.feature_id
    next_state["target_mod"] = plan.target_mod
    next_state["execution_path"] = plan.route
    next_state["risk_level"] = plan.risk_level
    next_state["current_stage"] = "request_planning"
    _record_artifact(next_state, "request_planning_report", artifact_ref)
    next_state["warnings"].extend(
        uncertainty.to_dict()
        for uncertainty in plan.uncertainties
        if uncertainty.severity != "blocking"
    )
    if plan.blocked:
        next_state["blockers"].extend(
            {
                "stage": "request_planning",
                "code": uncertainty.uncertainty_id.upper(),
                "message": uncertainty.question,
            }
            for uncertainty in plan.uncertainties
            if uncertainty.severity == "blocking"
        )
    return next_state


def _run_draft_feature_spec(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    domain_feature_spec_factory: Callable[..., FeatureSpec] | None = None,
    revision_projection: Mapping[str, Any] | None = None,
    approved_payload_hash: str | None = None,
    approved_projection_hash: str | None = None,
) -> ModAgentState:
    if revision_projection is not None:
        if approved_payload_hash is None or approved_projection_hash is None:
            raise ValueError(
                "FeatureSpec revision approval hashes are incomplete."
            )
        return _materialize_feature_spec_revision(
            state,
            artifact_store,
            projection=revision_projection,
            approved_payload_hash=approved_payload_hash,
            approved_projection_hash=approved_projection_hash,
        )

    plan_data = _read_artifact_data(state, artifact_store, "request_planning_report")
    plan = _request_plan_from_payload(plan_data)
    planning_brief = _read_artifact_data(state, artifact_store, "mod_planning_brief")
    feature_spec = (domain_feature_spec_factory or FeatureSpecBuilder().build)(
        plan,
        planning_brief=planning_brief,
    )
    # Scope: an explicitly injected domain adapter may change domain defaults, not the user's task.
    # 为什么要有它：共享外部工程接入需要去掉游戏模板，但不得借适配器替换原始需求或任务身份。
    if (not isinstance(feature_spec, FeatureSpec)
            or feature_spec.feature_id != plan.feature_id
            or feature_spec.user_request != plan.user_request):
        raise ValueError("DOMAIN_FEATURE_SPEC_TASK_IDENTITY_CHANGED")
    artifact_ref = artifact_store.write(
        feature_id=feature_spec.feature_id,
        artifact_type="feature_spec",
        payload=feature_spec.to_artifact_payload(),
        status="blocked" if feature_spec.blocked else "success",
        upstream_refs=[
            _artifact_ref_from_state(state, "request_planning_report"),
            _artifact_ref_from_state(state, "mod_planning_brief"),
        ],
    )
    next_state = _copy_state(state)
    next_state["current_stage"] = "feature_spec"
    _record_artifact(next_state, "feature_spec", artifact_ref)
    if feature_spec.blocked:
        next_state["blockers"].extend(feature_spec.blockers)
    return next_state


def _run_retrieve_evidence(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    knowledge_database: KnowledgeDatabase | None,
    collection_id: str | None,
    collection_resolver: CollectionResolver | None,
    dense_provider: DenseRetrievalProvider | None,
    dense_enabled: bool,
    dense_required: bool,
    dense_top_k: int | None,
    evidence_pack_factory: EvidencePackFactory | None,
    *,
    initial_findings_reassessment_factory: EvidencePackFactory | None = None,
    product_execution_mode: ProductExecutionGraphModeV1 | None = None,
    enable_initial_findings_reassessment: bool = False,
) -> ModAgentState:
    feature_data = _read_artifact_data(state, artifact_store, "feature_spec")
    if evidence_pack_factory is not None:
        pack = evidence_pack_factory(state, feature_data)
    elif knowledge_database is None:
        from src.mod_agent.evidence import EvidencePack

        pack = EvidencePack.empty(
            query=str(feature_data.get("target_behavior") or state["user_request"]),
            reason="Knowledge database is not configured for this graph run.",
        )
    else:
        request = EvidenceRequest(
            query=str(feature_data.get("target_behavior") or state["user_request"]),
            execution_path=state["execution_path"] or "standard_feature",
            requires_hook_evidence=(
                "target method" in feature_data.get("evidence_requirements", [])
            ),
            hook_requirement=_hook_requirement_from_payload(feature_data),
        )
        try:
            pack = DeterministicEvidenceRetriever(
                knowledge_database,
                collection_id=collection_id,
                collection_resolver=collection_resolver,
                dense_provider=dense_provider,
                dense_enabled=dense_enabled,
                dense_required=dense_required,
                dense_top_k=dense_top_k,
            ).retrieve(request)
        except CollectionResolutionError as exc:
            from src.mod_agent.evidence import EvidencePack

            pack = EvidencePack.empty(
                query=request.query,
                reason=f"Collection resolution failed: {exc}",
            )
    from src.mod_agent.evidence.reassessment import (
        RequirementCoverageReassessmentFactory, validate_unchanged_source_reassessment,
    )

    if isinstance(evidence_pack_factory, RequirementCoverageReassessmentFactory):
        reuse = validate_unchanged_source_reassessment(pack, state=state, artifact_store=artifact_store,
            workspace_root=(product_execution_mode.request.isolated_workspace_root if product_execution_mode is not None
                else getattr(evidence_pack_factory, "workspace_root", None)), feature_data=feature_data)
        if reuse is not None:
            next_state = _copy_state(state)
            next_state["current_stage"] = "evidence_retrieval"
            next_state["runtime_summary"]["evidence_reassessment"] = reuse
            next_state["runtime_summary"].setdefault("evidence_reassessment_history", []).append(copy.deepcopy(reuse))
            # 为什么要有它：本次真实查询保存在执行状态/原查询trace中，不能为相同证据改写head而使旧Plan依赖失效。
            return next_state
    artifact_ref = artifact_store.write(
        feature_id=state["feature_id"] or state["request_id"],
        artifact_type="evidence_pack",
        payload=pack.to_artifact_payload(),
        status="success" if pack.can_generate_patch else "blocked",
        upstream_refs=[_artifact_ref_from_state(state, "feature_spec")],
        trace_refs=([f"{state['artifacts']['repair_plan']}#sha256={state['artifact_hashes']['repair_plan']}"]
            if state["runtime_summary"].get("patch_upstream_return") else []),
    )
    next_state = _copy_state(state)
    _record_artifact(next_state, "evidence_pack", artifact_ref)
    from src.mod_agent.evidence.compilation_candidate import compilation_candidate_scope

    candidate_scope = compilation_candidate_scope(pack.to_artifact_payload(), feature_data)
    if (product_execution_mode is not None and enable_initial_findings_reassessment and not state["blockers"]
            and candidate_scope is None
            and not any(kind in state["artifacts"] for kind in ("evidence_pack", "implementation_plan", "repair_plan", "patch_proposal"))):
        from src.mod_agent.evidence.reassessment import initial_findings_reassessment_allowed

        if initial_findings_reassessment_allowed(pack.to_artifact_payload(), feature_data):
            next_state["runtime_summary"]["initial_findings_reassessment"] = {
                "prior_evidence_hash": artifact_ref.artifact_hash,
                "provider_configured": initial_findings_reassessment_factory is not None,
                "maximum_reassessments": 1,
            }
            if initial_findings_reassessment_factory is None:
                next_state["blockers"].append({"stage": "evidence_retrieval",
                    "code": "INITIAL_FINDINGS_REASSESSMENT_PROVIDER_REQUIRED",
                    "message": "The current Findings require more evidence, but no reassessment provider is configured."})
            else:
                previous_ref = artifact_ref
                # One bounded semantic pass in the existing Evidence node. Its
                # original insufficient Artifact stays immutable and queryable.
                pack = initial_findings_reassessment_factory(next_state, feature_data)
                artifact_ref = artifact_store.write(
                    feature_id=state["feature_id"], artifact_type="evidence_pack",
                    payload=pack.to_artifact_payload(),
                    status="success" if pack.can_generate_patch else "blocked",
                    upstream_refs=[_artifact_ref_from_state(state, "feature_spec")],
                    trace_refs=[f"{previous_ref.path}#sha256={previous_ref.artifact_hash}"],
                )
    next_state["current_stage"] = "evidence_retrieval"
    next_state["evidence_summary"] = {
        "readiness": pack.readiness,
        "can_generate_patch": pack.can_generate_patch,
        "num_items": len(pack.items),
        "gaps": list(pack.gaps),
        "retrieval_methods_used": list(pack.retrieval_methods_used),
        "missing_evidence": list(pack.missing_evidence),
        "disabled_modes": dict(pack.disabled_modes),
        "timing": dict(pack.timing),
        **({"compilation_repair_candidate": candidate_scope} if candidate_scope is not None else {}),
    }
    _record_artifact(next_state, "evidence_pack", artifact_ref)
    # 为什么要有它：保留原不足Evidence与疑问，已绑定源码错误可进入只读候选生成；执行仍由原验收和批准约束。
    if not pack.can_generate_patch and candidate_scope is None:
        next_state["blockers"].extend(pack.blockers)
    return next_state


def _finish_replanned_selection(state, context, *, product_execution_mode=None):
    assessed = bool(state["runtime_summary"].get("patch_upstream_return"))
    result = context.after_plan(state) if context is not None else state
    if assessed and product_execution_mode is not None and not result["blockers"]:
        # Rebuild only this request's admission to the exact-approval reader.
        # Prior diff/review approvals remain cleared by the upstream owner.
        result["approvals"] = {
            "confirmed_write": product_execution_mode.request.confirmed_isolated_write,
        }
    return result


def _run_create_implementation_plan(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    formal_requirement_coverage_required: bool = False,
    use_store_result: bool = False,
    planning_continuation_mode: PlanningContinuationInputV1 | None = None,
    planning_target_selection_factory: PlanningTargetSelectionFactory | None = None,
    patch_continuation_mode: PatchContinuationInputV1 | None = None,
    product_patch_upstream_execution: Any = None,
) -> ModAgentState:
    feature_spec = _feature_spec_from_payload(
        _read_artifact_data(state, artifact_store, "feature_spec")
    )
    evidence_pack = _read_artifact_data(state, artifact_store, "evidence_pack")
    formal_requirement_coverage_required = formal_requirement_coverage_required or isinstance(
        planning_target_selection_factory, ModelPlanningTargetSelector
    )
    prior_feedback = None
    prior_assessment = None
    assessment_context = None
    preserved_plan_boundaries = ()
    if state["runtime_summary"].get("patch_upstream_return") and isinstance(
            planning_target_selection_factory, ModelPlanningTargetSelector):
        from src.mod_agent.model_call_budget import is_meter_only, total_only
        if product_patch_upstream_execution is not None and (
                total_only(product_patch_upstream_execution.budget_snapshot())
                or is_meter_only(product_patch_upstream_execution.budget_snapshot())):
            planning_target_selection_factory = planning_target_selection_factory.for_product_assessment(
                state["artifact_hashes"]["repair_plan"])
        repair = artifact_store.semantic_read_immutable(_artifact_ref_from_state(state, "repair_plan"))["data"]
        upstream = repair["repair_context"]["upstream_request"]
        prior_feedback = planning_target_selection_factory.semantic_feedback
        prior_assessment = planning_target_selection_factory.upstream_assessment
        preserved_plan_boundaries = tuple(upstream["preserved_plan_boundaries"])
        recommendation = {key: copy.deepcopy(upstream[key]) for key in (
            "origin", "required_owner", "reason", "source_refs")}
        if upstream.get("source_worker_response_ref") is not None:
            recommendation["source_worker_response_ref"] = {
                "sha256": upstream["source_worker_response_ref"]["sha256"]}
        from src.mod_agent.proposal.behavior_feedback_view import project_execution_reference_paths

        planning_target_selection_factory.semantic_feedback = tuple(dict.fromkeys((
            *prior_feedback, *preserved_plan_boundaries,
            json.dumps(project_execution_reference_paths({"recommendation": recommendation,
                "observed_failure": repair["failure_findings"], "failure_report_ref": upstream["failure_report_ref"],
                "recommendation_is_not_observation": True}), ensure_ascii=False, sort_keys=True))))
    try:
        if prior_feedback is not None and (patch_continuation_mode is not None or product_patch_upstream_execution is not None):
            from src.mod_agent.repair.upstream import prepare_patch_upstream_assessment
            try:
                assessment_context = (product_patch_upstream_execution.prepare_assessment(state, artifact_store)
                    if product_patch_upstream_execution is not None else prepare_patch_upstream_assessment(
                        request=patch_continuation_mode, state=state, artifact_store=artifact_store))
            except PatchContinuationError as exc:
                raise PlanningTargetSelectionError(exc.code, exc.reason) from exc
            planning_target_selection_factory.upstream_assessment = copy.deepcopy(assessment_context)
        reused_evidence = state["runtime_summary"].get("evidence_reassessment")
        if (assessment_context is not None and assessment_context["retain_current_plan_allowed"] is True
                and all(value in assessment_context["current_plan"]["side_effect_boundaries"]
                    for value in (prior_feedback or ()))
                and isinstance(reused_evidence, dict)
                and reused_evidence.get("origin") == "executed_source_recheck_reusing_unchanged_judgments"
                and reused_evidence.get("repair_plan_hash") == assessment_context["repair_plan_hash"]
                and reused_evidence.get("selected_artifact_hashes") == assessment_context["selected_artifact_hashes"]):
            next_state = _copy_state(state)
            next_state["current_stage"] = "implementation_plan"
            next_state["approvals"] = {}
            next_state["runtime_summary"]["planning_upstream_assessment"] = {
                "decision": "retain_current_plan", "origin": reused_evidence["origin"],
                "rationale": "The requested queries executed and returned no new source facts. "
                    "The same complete source observations remain fresh; original Findings, bindings, "
                    "requirements and scope are unchanged. This is not proof of a repair. "
                    "Use the retained observations, or request evidence that is still missing.",
                "implementation_plan": {"path": state["artifacts"]["implementation_plan"],
                    "artifact_hash": state["artifact_hashes"]["implementation_plan"]},
                "repair_plan_hash": assessment_context["repair_plan_hash"],
                "trace_refs": [], "approval_inherited": False}
            return next_state
        if planning_continuation_mode is not None and isinstance(
            planning_target_selection_factory, ModelPlanningTargetSelector
        ):
            raise PlanningTargetSelectionError(
                "PLANNING_MODEL_OPERATOR_SELECTION_CONFLICT",
                "An autonomous selector cannot silently use explicit continuation target IDs.",
            )
        planning_target_evidence_ids = (
            planning_continuation_mode.planning_target_evidence_ids
            if planning_continuation_mode is not None
            else planning_target_selection_factory(
                copy.deepcopy(feature_spec.to_artifact_payload()),
                copy.deepcopy(evidence_pack),
            )
            if formal_requirement_coverage_required
            and planning_target_selection_factory is not None
            else None
        )
        if (isinstance(planning_target_selection_factory, ModelPlanningTargetSelector)
                and planning_target_selection_factory.selection_result.get("decision") == "retain_current_plan"):
            if assessment_context is None or assessment_context["retain_current_plan_allowed"] is not True:
                raise PlanningTargetSelectionError("PLANNING_MODEL_RETAIN_PLAN_NOT_ADMITTED",
                    "The current Graph does not own an unchanged Plan retention assessment.")
            try:
                current_assessment = (product_patch_upstream_execution.prepare_assessment(state, artifact_store)
                    if product_patch_upstream_execution is not None else prepare_patch_upstream_assessment(
                        request=patch_continuation_mode, state=state, artifact_store=artifact_store))
            except PatchContinuationError as exc:
                raise PlanningTargetSelectionError(exc.code, exc.reason) from exc
            if current_assessment != assessment_context:
                raise PlanningTargetSelectionError("PLANNING_MODEL_RETAIN_PLAN_CONTEXT_CHANGED",
                    "The Plan dependencies or checked failure changed during assessment.")
            next_state = _copy_state(state)
            next_state["current_stage"] = "implementation_plan"
            next_state["approvals"] = {}
            next_state["runtime_summary"]["planning_upstream_assessment"] = {
                **planning_target_selection_factory.selection_result,
                "implementation_plan": {
                    "path": state["artifacts"]["implementation_plan"],
                    "artifact_hash": state["artifact_hashes"]["implementation_plan"],
                },
                "repair_plan_hash": assessment_context["repair_plan_hash"],
                "trace_refs": planning_target_selection_factory.planning_trace_refs(),
                "approval_inherited": False,
            }
            return next_state
        implementation = ImplementationPlanner().build(
            feature_spec=feature_spec,
            evidence_pack=evidence_pack,
            formal_requirement_coverage_required=formal_requirement_coverage_required,
            planning_target_evidence_ids=planning_target_evidence_ids,
            constraint_policy=getattr(planning_target_selection_factory, "constraint_policy", None),
        )
        from src.mod_agent.planning.target_correction import correct_unsafe_target_selection

        implementation = correct_unsafe_target_selection(
            implementation=implementation, selector=planning_target_selection_factory,
            feature_spec=feature_spec, evidence_pack=evidence_pack,
            formal_requirement_coverage_required=formal_requirement_coverage_required,
            allow_correction=(planning_continuation_mode is None and patch_continuation_mode is None
                              and not state["runtime_summary"].get("patch_upstream_return")),
        )
        if not implementation.blocking and isinstance(
            planning_target_selection_factory, ModelPlanningTargetSelector
        ):
            # 为什么要有它：上游诊断可供规划推翻，不能把模型猜想或检查观测升级为 Patch 的硬约束；原边界仍继承。
            boundary_feedback = (planning_target_selection_factory.semantic_feedback
                if prior_feedback is None else (*prior_feedback, *preserved_plan_boundaries))
            implementation = replace(
                implementation,
                verification_questions=planning_target_selection_factory.verification_questions,
                side_effect_boundaries=tuple(
                    dict.fromkeys(
                        (
                            *implementation.side_effect_boundaries,
                            *boundary_feedback,
                        )
                    )
                ),
            )
            if implementation.constraint_context is not None:
                from src.mod_agent.planning.constraint_context import build_constraint_context

                # Scope: synchronize explicit experimental context with retained caller constraints.
                # 为什么要有它：未知来源的既有边界不能被文本巧合降级；重规划也不能让来源上下文与实际硬边界分离。
                context = implementation.constraint_context
                context = build_constraint_context(feature_spec=feature_spec.to_artifact_payload(),
                    evidence_pack=evidence_pack, plan_boundaries=context["source_plan_boundaries"],
                    retained_hard_boundaries=(*context["retained_hard_boundaries"], *boundary_feedback))
                implementation = replace(implementation, constraint_context=context,
                    side_effect_boundaries=tuple(context["hard_boundaries"]))
    except PlanningTargetSelectionError as exc:
        implementation = ImplementationPlanner._blocked(
            feature_id=feature_spec.feature_id,
            blockers=(exc.as_blocker(),),
            open_questions=(exc.reason,),
        )
    finally:
        if prior_feedback is not None:
            planning_target_selection_factory.semantic_feedback = prior_feedback
            planning_target_selection_factory.upstream_assessment = prior_assessment
    if planning_continuation_mode is not None:
        preflight_planning_continuation(
            planning_continuation_mode,
            artifact_store=artifact_store,
        )
    write_arguments = {
        "feature_id": implementation.feature_id,
        "artifact_type": "implementation_plan",
        "payload": implementation.to_artifact_payload(),
        "status": "blocked" if implementation.blocking else "success",
        "upstream_refs": [
            _artifact_ref_from_state(state, "feature_spec"),
            _artifact_ref_from_state(state, "evidence_pack"),
        ],
    }
    if isinstance(planning_target_selection_factory, ModelPlanningTargetSelector):
        write_arguments["trace_refs"] = planning_target_selection_factory.planning_trace_refs()
    if state["runtime_summary"].get("patch_upstream_return"):
        write_arguments.setdefault("trace_refs", []).append(
            f"{state['artifacts']['repair_plan']}#sha256={state['artifact_hashes']['repair_plan']}")
    write_result = None
    if use_store_result:
        write_result = artifact_store.write_with_result(**write_arguments)
        artifact_ref = write_result.artifact_ref
        if write_result.outcome != "committed" or artifact_ref is None:
            raise PlanningContinuationArtifactWriteError(write_result)
    else:
        artifact_ref = artifact_store.write(**write_arguments)
    next_state = _copy_state(state)
    next_state["current_stage"] = "implementation_plan"
    _record_artifact(next_state, "implementation_plan", artifact_ref)
    if write_result is not None:
        next_state["runtime_summary"][
            "planning_artifact_write_result"
        ] = write_result.to_dict()
    if implementation.blocking:
        next_state["blockers"].extend(implementation.blockers)
    return next_state


def _patch_factory_attempt_receipt(
    patch_change_factory: PatchChangeFactory,
) -> dict[str, Any]:
    reporter = getattr(patch_change_factory, "patch_attempt_receipt", None)
    if reporter is None:
        return {}
    if not callable(reporter):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_FACTORY_RECEIPT_INVALID",
            "patch_generation",
            "Patch factory receipt reporter must be callable.",
        )
    receipt = reporter()
    if not isinstance(receipt, Mapping):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_FACTORY_RECEIPT_INVALID",
            "patch_generation",
            "Patch factory receipt reporter must return an object.",
        )
    return json.loads(
        json.dumps(
            dict(receipt),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _patch_continuation_current_snapshot_sha256(
    mode: PatchContinuationInputV1,
) -> str | None:
    try:
        root = Path(mode.workspace_root).resolve(strict=True)
        snapshot = tuple(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(root.rglob("*.cs"), key=lambda item: item.as_posix())
            if path.is_file()
        )
    except OSError:
        return None
    return hashlib.sha256(
        json.dumps(
            [{"path": path, "sha256": sha256} for path, sha256 in snapshot],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _valid_patch_read_accounting(value: Mapping[str, Any], *, reuse_enabled: bool) -> bool:
    """Validate producer accounting without treating reused bytes as a new read."""
    reads, observations = value.get("tool_rounds_used"), value.get("observation_trace")
    if type(reads) is not int or reads < 0 or not isinstance(observations, list):
        return False
    if not reuse_enabled:
        return (value.get("read_accounting_version") in (None, "binding_dispatch_v1")
                and len(observations) == reads)
    if value.get("read_accounting_version") != "new_source_window_v2":
        return False
    if any(not isinstance(item, Mapping) or type(item.get("read_cost")) is not int
           or item["read_cost"] not in (0, 1)
           or item.get("observation_origin") != (
               "current_source_read" if item["read_cost"] else "prior_visible_source_reuse")
           for item in observations):
        return False
    dispatched, reused = value.get("dispatched_binding_reads"), value.get("reused_source_observations")
    return (type(dispatched) is int and dispatched == len(observations)
            and type(reused) is int and reused == sum(item["read_cost"] == 0 for item in observations)
            and reads == sum(item["read_cost"] for item in observations))


def _valid_patch_agent_loop_receipt(
    value: Any,
    *,
    model_call_count: int,
) -> bool:
    from src.mod_agent.proposal.patch_agent_loop import iter_patch_recovery_revisions

    if not isinstance(value, Mapping):
        return False
    usage = value.get("usage_by_turn")
    observations = value.get("observation_trace")
    termination = value.get("termination")
    tool_rounds = value.get("tool_rounds_used")
    loop_model_turns = value.get("model_turns_used")
    policy = value.get("policy", {})
    if not isinstance(policy, Mapping):
        return False
    metered = policy.get("model_call_budget_policy") == "meter_only"
    if metered and any(policy.get(name, False) is not None for name in (
            "max_tool_rounds", "max_total_model_turns", "max_observation_bytes")):
        return False
    read_limit = policy.get("max_tool_rounds", 3)
    require_read = policy.get("require_source_inspection", True)
    reuse_enabled = policy.get("reuse_covered_source_observations", False)
    if (not metered and (type(read_limit) is not int or read_limit < 0)
            or type(require_read) is not bool or (not metered and require_read and read_limit == 0)
            or type(reuse_enabled) is not bool):
        return False
    origin = value.get("prior_worker_budget", {
        "consumed_model_turns": 0, "consumed_tool_rounds": 0, "consumed_observation_bytes": 0,
    })
    if (not isinstance(origin, Mapping) or set(origin) != {
            "consumed_model_turns", "consumed_tool_rounds", "consumed_observation_bytes"}
            or any(type(number) is not int or number < 0 for number in origin.values())):
        return False
    if "prior_worker_budget" in value and "recovery_revisions" not in value:
        return False
    try:
        corrections = iter_patch_recovery_revisions(value)
    except ValueError:
        return False
    if "recovery_revisions" in value:
        latest = {}
        for index, correction in enumerate(corrections, 1):
            kind = correction.get("message_type")
            if (type(correction.get("revision_index")) is not int or correction["revision_index"] != index
                    or kind not in {"patch_validator_correction", "patch_behavior_correction"}):
                return False
            latest[kind.removeprefix("patch_")] = correction
        if any(value.get(name) != latest.get(name) for name in ("validator_correction", "behavior_correction")):
            return False
        phases = [value, *(item.get("execution") for item in corrections)]
        extra_bytes = value.get("recovery_observation_bytes")
        byte_limit = policy.get("max_observation_bytes", 32_768)
        model_limit = policy.get("max_total_model_turns", 5)
        if (type(extra_bytes) is not int or extra_bytes < 0
                or not metered and (type(byte_limit) is not int or byte_limit < 0)
                or not metered and (type(model_limit) is not int or not 1 <= model_limit <= 5)
                or type(model_call_count) is not int or model_call_count < 1
                or not metered and model_call_count + origin["consumed_model_turns"] > model_limit
                or any(not isinstance(phase, Mapping) or type(phase.get("total_observation_bytes")) is not int
                    or phase["total_observation_bytes"] < 0 for phase in phases)
                or not metered and (origin["consumed_observation_bytes"] + extra_bytes
                    + sum(phase["total_observation_bytes"] for phase in phases) > byte_limit)):
            return False
    recovery_reads = 0
    correction_calls = 0
    for correction in corrections:
        if not (
            isinstance(correction, Mapping)
            and correction.get("status") == "completed"
            and type(correction.get("attempt_count")) is int
            and correction["attempt_count"] == 1
            and isinstance(correction.get("response"), Mapping)
        ):
            return False
        count = correction.get("model_call_count", 1)
        if type(count) is not int or count < 1:
            return False
        correction_calls += count
        execution = correction.get("execution")
        if execution is not None:
            if not isinstance(execution, Mapping):
                return False
            if "recovery_revisions" in value and (
                    execution.get("status") != "completed" or execution.get("model_turns_used") != count
                    or not isinstance(execution.get("usage_by_turn"), list)
                    or len(execution["usage_by_turn"]) != count
                    or any(not isinstance(item, Mapping) or item.get("admitted_to_model") is not True
                        for item in execution.get("observation_trace", []))):
                return False
            reads = execution.get("tool_rounds_used")
            if not _valid_patch_read_accounting(execution, reuse_enabled=reuse_enabled):
                return False
            recovery_reads += reads
    behavior = value.get("behavior_evaluations")
    if behavior is not None and not (
        isinstance(behavior, list) and behavior
        and isinstance(behavior[-1], Mapping)
        and behavior[-1].get("status") == "passed"
        and behavior[-1].get("admission") == "accepted"
        and behavior[-1].get("execution_origin") != "incomplete"
    ):
        return False
    return (
        value.get("enabled") is True
        and value.get("status") == "completed"
        and type(loop_model_turns) is int
        and loop_model_turns + correction_calls == model_call_count
        and isinstance(usage, list)
        and len(usage) == loop_model_turns
        and type(tool_rounds) is int
        and (1 if require_read else 0) <= tool_rounds + origin["consumed_tool_rounds"]
        and (metered or origin["consumed_tool_rounds"] + tool_rounds + recovery_reads <= read_limit)
        and isinstance(observations, list)
        and _valid_patch_read_accounting(value, reuse_enabled=reuse_enabled)
        and all(
            isinstance(item, Mapping)
            and item.get("admitted_to_model") is True
            for item in observations
        )
        and isinstance(termination, Mapping)
        and termination.get("status") == "completed"
        and termination.get("code") == "PATCH_AGENT_MODEL_FINISHED"
    )


def _run_patch_continuation_factory(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    implementation: dict[str, Any],
    evidence_pack: dict[str, Any],
    patch_change_factory: PatchChangeFactory,
    mode: PatchContinuationInputV1,
    recorder: PatchContinuationAttemptRecorder,
    max_model_calls: int,
) -> tuple[dict[str, Any], ...]:
    admitted_before = preflight_patch_continuation(
        mode,
        artifact_store=artifact_store,
    )
    admitted_snapshot = hashlib.sha256(
        json.dumps(
            [
                {"path": path, "sha256": sha256}
                for path, sha256 in admitted_before.source_tree_snapshot
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    generated_against = _patch_continuation_current_snapshot_sha256(mode)
    if generated_against is None or generated_against != admitted_snapshot:
        error = PatchContinuationError(
            "PATCH_CONTINUATION_SOURCE_CHANGED_BEFORE_MODEL",
            "pre_call_admission",
            "Product source changed after Patch preflight and before model invocation.",
        )
        recorder(
            {
                "phase": "pre_call_admission_rejected",
                "generated_against_snapshot_sha256": generated_against,
                "current_snapshot_sha256": generated_against,
                "model_call_count": 0,
                "failure": {
                    "code": error.code,
                    "failure_class": type(error).__name__,
                    "stage": error.stage,
                },
                "proposal_admitted": False,
            }
        )
        raise error
    recorder(
        {
            "phase": "patch_generation_started",
            "generated_against_snapshot_sha256": generated_against,
            "proposal_admitted": False,
        }
    )
    if admitted_before.repair_context is not None:
        binder = getattr(patch_change_factory, "bind_repair_context", None)
        if not callable(binder):
            raise PatchContinuationError("PATCH_REPAIR_FACTORY_CONTEXT_UNSUPPORTED", "patch_generation", "The selected Patch worker cannot consume the admitted repair context.")
        binder(admitted_before.repair_context)
    try:
        generated = tuple(
            patch_change_factory(state, implementation, evidence_pack)
        )
    except Exception as exc:
        factory_receipt = _patch_factory_attempt_receipt(patch_change_factory)
        recorder(
            {
                "phase": "patch_generation_failed",
                "factory_receipt": factory_receipt,
                "model_call_count": factory_receipt.get("model_call_count", 0),
                "failure": {
                    "code": str(getattr(exc, "code", type(exc).__name__)),
                    "failure_class": type(exc).__name__,
                    "stage": "patch_generation",
                },
                "proposal_admitted": False,
            }
        )
        raise
    factory_receipt = _patch_factory_attempt_receipt(patch_change_factory)
    model_call_count = factory_receipt.get("model_call_count")
    generated_sha256 = hashlib.sha256(
        json.dumps(
            generated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    recorder(
        {
            "phase": "model_output_persisted_before_post_call_admission",
            "factory_receipt": factory_receipt,
            "model_call_count": model_call_count,
            "generated_changes_sha256": generated_sha256,
            "generated_change_count": len(generated),
            "generated_against_snapshot_sha256": generated_against,
            "proposal_admitted": False,
        }
    )
    model_call_count_valid = (
        type(model_call_count) is int
        and 1 <= model_call_count <= max_model_calls
        and (
            model_call_count == 1
            or _valid_patch_agent_loop_receipt(
                factory_receipt.get("patch_agent_loop"),
                model_call_count=model_call_count,
            )
        )
    )
    if not model_call_count_valid:
        error = PatchContinuationError(
            "PATCH_CONTINUATION_MODEL_CALL_COUNT_INVALID",
            "post_call_admission",
            "Patch continuation model calls do not match the admitted budget.",
        )
        recorder(
            {
                "phase": "post_call_admission_rejected",
                "failure": {
                    "code": error.code,
                    "failure_class": type(error).__name__,
                    "stage": error.stage,
                },
                "proposal_admitted": False,
            }
        )
        raise error
    try:
        admitted_after = preflight_patch_continuation(
            mode,
            artifact_store=artifact_store,
        )
    except (PatchContinuationError, PlanningContinuationError) as exc:
        current_snapshot = _patch_continuation_current_snapshot_sha256(mode)
        recorder(
            {
                "phase": "post_call_admission_rejected",
                "current_snapshot_sha256": current_snapshot,
                "failure": {
                    "code": str(getattr(exc, "code", type(exc).__name__)),
                    "failure_class": type(exc).__name__,
                    "stage": str(getattr(exc, "stage", "post_call_admission")),
                },
                "proposal_admitted": False,
            }
        )
        raise PatchContinuationError(
            str(getattr(exc, "code", "PATCH_CONTINUATION_POST_CALL_DRIFT")),
            "post_call_admission",
            "Exact Artifact, binding, or source admission changed during Patch generation.",
        ) from exc
    current_snapshot = _patch_continuation_current_snapshot_sha256(mode)
    if admitted_after.source_tree_snapshot != admitted_before.source_tree_snapshot:
        error = PatchContinuationError(
            "PATCH_CONTINUATION_SOURCE_CHANGED_AFTER_MODEL",
            "post_call_admission",
            "Product source changed while Patch generation was running.",
        )
        recorder(
            {
                "phase": "post_call_admission_rejected",
                "current_snapshot_sha256": current_snapshot,
                "failure": {
                    "code": error.code,
                    "failure_class": type(error).__name__,
                    "stage": error.stage,
                },
                "proposal_admitted": False,
            }
        )
        raise error
    recorder(
        {
            "phase": "post_call_admission_passed",
            "current_snapshot_sha256": current_snapshot,
            "proposal_admitted": False,
        }
    )
    return generated


def _run_generate_patch_with_upstream(state, artifact_store, *, enable_upstream_replanning=False,
        upstream_evidence_factory=None, upstream_planning_factory=None, upstream_continues_patch=False,
        product_patch_upstream_execution=None, **kwargs):
    from src.mod_agent.evidence.reassessment import RequirementCoverageReassessmentFactory

    effect = ("assess_and_continue" if upstream_continues_patch else "assess_then_stop") if enable_upstream_replanning else "record_and_stop"
    effects = {"planning": effect, "evidence": effect}
    if enable_upstream_replanning and not isinstance(upstream_evidence_factory, RequirementCoverageReassessmentFactory):
        effects["evidence"] = "provider_unavailable_stop"
    if enable_upstream_replanning and product_patch_upstream_execution is not None and not isinstance(
            upstream_planning_factory, ModelPlanningTargetSelector):
        effects = {"planning": "provider_unavailable_stop", "evidence": "provider_unavailable_stop"}
    from src.mod_agent.model_call_budget import is_meter_only, total_only
    if (product_patch_upstream_execution is not None and product_patch_upstream_execution.replan_count
            and not (total_only(product_patch_upstream_execution.budget_snapshot())
                     or is_meter_only(product_patch_upstream_execution.budget_snapshot()))):
        effects = {"planning": "record_and_stop", "evidence": "record_and_stop"}
    binder = getattr(product_patch_upstream_execution or kwargs.get("patch_change_factory"), "bind_upstream_effects", None)
    if callable(binder):
        binder(effects)
    try:
        return _run_generate_patch_proposal(state, artifact_store, **kwargs)
    except Exception as exc:
        if not enable_upstream_replanning or str(exc) != "PATCH_AGENT_UPSTREAM_REQUESTED":
            raise
        from src.mod_agent.repair.upstream import validate_active_worker_upstream
        next_state = _copy_state(state)
        try:
            if product_patch_upstream_execution is not None:
                _, upstream = product_patch_upstream_execution.validate_upstream(state, artifact_store)
                if not isinstance(upstream_planning_factory, ModelPlanningTargetSelector):
                    raise PatchContinuationError("PATCH_UPSTREAM_PLANNING_PROVIDER_REQUIRED", "upstream_admission",
                        "No fresh model-owned Planning reassessment provider was configured.")
            else:
                _, upstream = validate_active_worker_upstream(kwargs["patch_continuation_mode"],
                    _patch_factory_attempt_receipt(kwargs["patch_change_factory"]), artifact_store=artifact_store)
            if upstream["required_owner"] == "evidence" and not isinstance(
                    upstream_evidence_factory, RequirementCoverageReassessmentFactory):
                raise PatchContinuationError("PATCH_UPSTREAM_EVIDENCE_PROVIDER_REQUIRED", "upstream_admission",
                    "No full-coverage Evidence reassessment provider was configured.")
        except (PatchContinuationError, ValueError) as invalid:
            next_state["blockers"].append({"stage": "upstream_admission",
                "code": getattr(invalid, "code", "PATCH_UPSTREAM_ADMISSION_FAILED"), "message": str(invalid)})
            return next_state
        next_state["current_stage"] = "patch_upstream_request" if product_patch_upstream_execution is not None else "behavior_report"
        next_state["runtime_summary"]["patch_upstream_pending"] = {
            "required_owner": upstream["required_owner"], "source_worker_response_ref": upstream["source_worker_response_ref"],
            "source_plan_hash": state["artifact_hashes"]["implementation_plan"], "proposal_admitted": False}
        recorder = kwargs.get("patch_continuation_attempt_recorder")
        if recorder is not None:
            recorder({"phase": "patch_upstream_assessment_admitted", "upstream_request": upstream, "proposal_admitted": False})
        return next_state


def _run_generate_patch_proposal_v2(state, store, *, owner, factory):
    """Run the same product node with an explicit v2 contract consumer.

    为什么要有它：U1必须从真实FeatureSpec进入Proposal，不能在旧Plan后接一个演示适配器。
    """
    from src.mod_agent.task_agent_v2_handoff import TaskAgentPatchOutcomeV2
    initial = {"feature_spec_ref": asdict(_artifact_ref_from_state(state, "feature_spec"))}
    outcome = factory(state, None, None)
    if not isinstance(outcome, TaskAgentPatchOutcomeV2) or outcome.initial_context != initial:
        raise ProductExecutionContractError("V2 task inputs changed.", side_effects_may_have_started=True)
    selected = _copy_state(state)
    selected["runtime_summary"]["task_agent_execution"] = outcome.to_dict()
    if outcome.status != "candidate_ready":
        selected["blockers"].append({"stage": "generate_patch_proposal",
            "code": outcome.termination.get("code", "V2_CANDIDATE_UNAVAILABLE"),
            "message": "V2 execution has no admitted candidate.", "execution_ref": outcome.execution_ref})
        return selected
    handoff = owner.validate_handoff(outcome, expected_initial_context=initial)
    proposal = PatchProposalBuilder().build_v2(handoff, feature_id=state["feature_id"],
        feature_spec_ref=initial["feature_spec_ref"], store=store)
    ref = store.write(feature_id=state["feature_id"], artifact_type="patch_proposal",
        payload=proposal.to_artifact_payload(), status="success",
        upstream_refs=[_artifact_ref_from_state(state, "feature_spec")],
        exact_revision_upstream_types={"feature_spec"})
    _record_artifact(selected, "patch_proposal", ref)
    selected["current_stage"] = "patch_proposal"
    return selected


def _run_generate_patch_proposal(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    proposal_git_binding_factory: ProposalGitBindingFactory | None = None,
    patch_change_factory: PatchChangeFactory | None = None,
    source_snippet_char_limit: int | None = 6000,
    patch_continuation_mode: PatchContinuationInputV1 | None = None,
    patch_continuation_attempt_recorder: (
        PatchContinuationAttemptRecorder | None
    ) = None,
    patch_continuation_max_model_calls: int = 1,
    task_agent_execution: Any = None,
    retained_task_agent_handoff: Any = None,
) -> ModAgentState:
    if getattr(task_agent_execution, "protocol", None) == "v2":
        return _run_generate_patch_proposal_v2(state, artifact_store,
            owner=task_agent_execution, factory=patch_change_factory)
    saved_patch = None
    saved_candidate = None
    task_handoff = None
    if retained_task_agent_handoff is not None:
        from src.mod_agent.retained_task_agent_handoff import RetainedHandoffAdmission

        if (type(retained_task_agent_handoff) is not RetainedHandoffAdmission
                or task_agent_execution is not None or patch_change_factory is not None
                or patch_continuation_mode is not None):
            raise ValueError("RETAINED_TASK_AGENT_GRAPH_OWNER_CONFLICT")
        admitted = retained_task_agent_handoff.revalidate()
        task_handoff, outcome = admitted.handoff, admitted.outcome
        for role in ("feature_spec", "evidence_pack", "implementation_plan"):
            if _artifact_ref_from_state(state, role) != getattr(outcome.final_context, role + "_ref"):
                raise ValueError("RETAINED_TASK_AGENT_GRAPH_CONTEXT_CHANGED")
        patch_change_factory = lambda *_args: task_handoff.changes
    if task_agent_execution is not None:
        from src.mod_agent.task_agent_execution import TaskAgentPatchOutcome
        from src.mod_agent.task_agent_tools import TaskAgentContext

        initial_context = TaskAgentContext(**{role + "_ref": _artifact_ref_from_state(state, role)
            for role in ("feature_spec", "evidence_pack", "implementation_plan")})
        outcome = patch_change_factory(state,
            _read_artifact_data(state, artifact_store, "implementation_plan"),
            _read_artifact_data(state, artifact_store, "evidence_pack"))
        if not isinstance(outcome, TaskAgentPatchOutcome) or outcome.initial_context != initial_context:
            raise ProductExecutionContractError("Task technical outcome does not identify its actual inputs.",
                side_effects_may_have_started=True)
        if outcome.status != "candidate_ready":
            next_state = _copy_state(state)
            next_state["runtime_summary"]["task_agent_execution"] = outcome.to_dict()
            next_state["blockers"].append({"stage": "generate_patch_proposal",
                "code": outcome.termination.get("code", "TASK_AGENT_EXECUTION_BLOCKED"),
                "message": "Technical execution did not produce a current qualified candidate.",
                "execution_ref": outcome.execution_ref, "unknown_effects": outcome.unknown_effects})
            return next_state
        task_handoff = task_agent_execution.validate_handoff(outcome, expected_initial_context=initial_context)
        # Scope: admitted final refs only; the original builder and all delivery nodes remain owners.
        # 为什么要有它：B可能已重新调查和规划，旧Plan不能绑定新候选；候选通过也不能替代Review、审批或正式Build。
        state = _copy_state(state)
        for role in ("feature_spec", "evidence_pack", "implementation_plan"):
            _record_artifact(state, role, getattr(task_handoff.final_context, role + "_ref"))
        state["runtime_summary"]["task_agent_execution"] = outcome.to_dict()
        patch_change_factory = lambda *_args: task_handoff.changes
    if patch_continuation_mode is not None and patch_continuation_mode.candidate_reverification_selection is not None:
        admitted = preflight_patch_continuation(patch_continuation_mode, artifact_store=artifact_store)
        if (patch_change_factory is not None or patch_continuation_max_model_calls != 0
                or patch_continuation_attempt_recorder is None
                or admitted.saved_candidate is None or state["request_id"] != admitted.saved_candidate["request_id"]):
            raise PatchContinuationError("SAVED_CANDIDATE_GRAPH_INPUT_INVALID", "patch_proposal", "Candidate requires a zero-model execution.")
        from src.mod_agent.saved_candidate_reverification import execute_saved_candidate
        # 为什么要有它：只在真实执行节点调用验收器，START准入和重复读取不能重新编译或消耗模型预算。
        saved_candidate = execute_saved_candidate(patch_continuation_mode, admitted, patch_continuation_attempt_recorder)
    if patch_continuation_mode is not None and patch_continuation_mode.saved_patch_selection is not None:
        # Re-admit the retained output at the node boundary, after START admission.
        # No model factory or synthetic generation receipt participates here.
        admitted = preflight_patch_continuation(patch_continuation_mode, artifact_store=artifact_store)
        saved_patch = admitted.saved_patch
        if (
            saved_patch is None or patch_change_factory is not None
            or patch_continuation_max_model_calls != 0
            or patch_continuation_attempt_recorder is None
            or state.get("request_id") != saved_patch["request_id"]
        ):
            raise PatchContinuationError(
                "SAVED_PATCH_GRAPH_INPUT_INVALID", "patch_proposal",
                "Saved output is not bound to this zero-model continuation.",
            )
    if state["current_stage"] == "repair_plan" and "repair_plan" in state["artifacts"]:
        repair_plan = _read_artifact_data(state, artifact_store, "repair_plan")
        implementation = _read_artifact_data(
            state,
            artifact_store,
            "implementation_plan",
        )
        evidence_pack = _read_artifact_data(state, artifact_store, "evidence_pack")
        if patch_change_factory is not None:
            generated_changes = (
                _run_patch_continuation_factory(
                    state, artifact_store, implementation=implementation,
                    evidence_pack=evidence_pack, patch_change_factory=patch_change_factory,
                    mode=patch_continuation_mode, recorder=patch_continuation_attempt_recorder,
                    max_model_calls=patch_continuation_max_model_calls,
                ) if patch_continuation_mode is not None and patch_continuation_attempt_recorder is not None
                else patch_change_factory(state, implementation, evidence_pack)
            )
            implementation = {
                **implementation,
                "proposed_changes": list(generated_changes),
            }
        if repair_plan.get("failure_source") == "behavior_report":
            proposal = PatchProposalBuilder().build(
                implementation, evidence_pack=evidence_pack,
                artifact_hashes={key: value for key, value in state["artifact_hashes"].items()
                    if key in {"feature_spec", "evidence_pack", "implementation_plan"}},
                require_git_binding=False,
                source_snippet_char_limit=(
                    PatchProposalBuilder.required_source_snippet_char_limit(implementation, evidence_pack)
                    if source_snippet_char_limit is None else source_snippet_char_limit
                ),
            )
        else:
            proposal = RepairPatchProposalBuilder().build(repair_plan, implementation_plan=implementation)
        repair_context_pack = dict(proposal.context_pack)
        repair_context_pack["artifact_hashes"] = {
            key: value
            for key, value in state["artifact_hashes"].items()
            if key in {"feature_spec", "evidence_pack", "implementation_plan"}
        }
        repair_context_pack["implementation_plan_hash"] = state[
            "artifact_hashes"
        ].get("implementation_plan")
        repair_context_pack["repair_lineage"] = {
            "repair_plan": {
                "path": state["artifacts"]["repair_plan"],
                "artifact_hash": state["artifact_hashes"]["repair_plan"],
            },
            "parent_proposal": repair_plan.get("source_proposal"),
            "failure_source": repair_plan.get("failure_source"),
            "base_relationship": (repair_plan.get("repair_context") or {}).get("base_relationship"),
            "approval_inherited": False,
        }
        proposal = replace(proposal, context_pack=repair_context_pack)
        upstream_refs = [
            _artifact_ref_from_state(state, "implementation_plan"),
            _artifact_ref_from_state(state, "repair_plan"),
            _artifact_ref_from_state(state, "evidence_pack"),
        ]
        if patch_continuation_mode is not None and patch_continuation_mode.repair_parent_proposal_ref is not None:
            upstream_refs.append(patch_continuation_mode.repair_parent_proposal_ref)
    else:
        implementation = _read_artifact_data(state, artifact_store, "implementation_plan")
        evidence_pack = _read_artifact_data(state, artifact_store, "evidence_pack")
        if saved_candidate is not None:
            implementation = {**implementation, "proposed_changes": copy.deepcopy(saved_candidate["changes"])}
        elif saved_patch is not None:
            implementation = {**implementation, "proposed_changes": copy.deepcopy(saved_patch["changes"])}
        elif patch_change_factory is not None:
            generated_changes = (
                _run_patch_continuation_factory(
                    state,
                    artifact_store,
                    implementation=implementation,
                    evidence_pack=evidence_pack,
                    patch_change_factory=patch_change_factory,
                    mode=patch_continuation_mode,
                    recorder=patch_continuation_attempt_recorder,
                    max_model_calls=patch_continuation_max_model_calls,
                )
                if patch_continuation_mode is not None
                and patch_continuation_attempt_recorder is not None
                else patch_change_factory(state, implementation, evidence_pack)
            )
            implementation = {
                **implementation,
                "proposed_changes": list(generated_changes),
            }
        proposal = PatchProposalBuilder().build(
            implementation,
            evidence_pack=evidence_pack,
            artifact_hashes=dict(state["artifact_hashes"]),
            git_binding=(
                proposal_git_binding_factory()
                if proposal_git_binding_factory is not None
                else None
            ),
            require_git_binding=proposal_git_binding_factory is not None,
            source_snippet_char_limit=(
                PatchProposalBuilder.required_source_snippet_char_limit(implementation, evidence_pack)
                if source_snippet_char_limit is None else source_snippet_char_limit
            ),
        )
        upstream_refs = [
            _artifact_ref_from_state(state, "implementation_plan"),
            _artifact_ref_from_state(state, "evidence_pack"),
        ]
        lineage = state["runtime_summary"].get("replanned_patch_lineage")
        if lineage is not None:
            proposal = replace(proposal, context_pack={**proposal.context_pack,
                "repair_lineage": copy.deepcopy(lineage)})
    if saved_candidate is not None:
        lineage = {key: copy.deepcopy(saved_candidate[key]) for key in
            ("source_attempt_ref", "response_ref", "request_id", "parent_run_id", "inherited_model_budget")}
        lineage.update(new_model_calls=0, approval_inherited=False,
            verification=copy.deepcopy(saved_candidate["verification"]))
        proposal = replace(proposal, context_pack={**proposal.context_pack, "saved_candidate_lineage": lineage})
        patch_continuation_attempt_recorder({"phase": "saved_candidate_reverified", "saved_candidate_lineage": lineage,
            "model_call_count": 0, "proposal_admitted": False})
    if saved_patch is not None:
        saved_lineage = {
            key: copy.deepcopy(saved_patch[key])
            for key in (
                "source_attempt_ref", "response_ref", "historical_verification", "request_id",
                "parent_run_id", "inherited_model_budget", "logical_budget", "source_snapshot_sha256",
            )
        }
        saved_lineage.update(
            source_proposal_ref=asdict(saved_patch["source_proposal_ref"]),
            source_review_ref=asdict(saved_patch["source_review_ref"]),
            model_calls_performed=0, verifier_executions_performed=0, approval_inherited=False,
        )
        proposal = replace(proposal, context_pack={**proposal.context_pack,
            "saved_patch_lineage": saved_lineage})
        # The existing registry permits these exact historical ancestors. Their
        # immutable identity remains valid after this invocation moves latest.
        upstream_refs.extend([
            _artifact_ref_from_state(state, "feature_spec"),
            saved_patch["source_proposal_ref"], saved_patch["source_review_ref"],
        ])
        patch_continuation_attempt_recorder({
            "phase": "saved_patch_reassembled", "saved_patch_lineage": saved_lineage,
            "model_call_count": 0, "proposal_admitted": False,
            "generated_changes_sha256": hashlib.sha256(json.dumps(
                saved_patch["changes"], ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "generated_change_count": len(saved_patch["changes"]),
            "generated_against_snapshot_sha256": saved_patch["source_snapshot_sha256"],
            "current_snapshot_sha256": saved_patch["source_snapshot_sha256"],
        })
    if task_handoff is not None:
        proposal = replace(proposal, context_pack={**proposal.context_pack, "task_agent_lineage": {
            "execution_ref": task_handoff.execution_ref, "candidate_ref": outcome.candidate_ref,
            "check_ref": outcome.check_ref, "approval_inherited": False, "whole_task_delivered": False}})
    artifact_ref = artifact_store.write(
        feature_id=proposal.feature_id,
        artifact_type="patch_proposal",
        payload=proposal.to_artifact_payload(),
        status="blocked" if proposal.blocked else "success",
        upstream_refs=upstream_refs,
        exact_revision_upstream_types=(
            {ref.artifact_type for ref in upstream_refs}
            if patch_continuation_mode is not None
            else None
        ),
    )
    if (
        patch_continuation_mode is not None
        and patch_continuation_attempt_recorder is not None
    ):
        patch_continuation_attempt_recorder(
            {
                "phase": "patch_proposal_committed",
                "proposal_admitted": not proposal.blocked,
                "artifacts": {
                    "patch_proposal": {
                        "feature_id": artifact_ref.feature_id,
                        "artifact_type": artifact_ref.artifact_type,
                        "path": artifact_ref.path,
                        "artifact_hash": artifact_ref.artifact_hash,
                    }
                },
                "patch_proposal_upstream_artifacts": [
                    {
                        "artifact_type": ref.artifact_type,
                        "path": ref.path,
                        "artifact_hash": ref.artifact_hash,
                    }
                    for ref in upstream_refs
                ],
            }
        )
    next_state = _copy_state(state)
    next_state["current_stage"] = "patch_proposal"
    _record_artifact(next_state, "patch_proposal", artifact_ref)
    if proposal.blocked:
        next_state["blockers"].extend(proposal.blockers)
    return next_state


def _default_proposal_git_binding_factory() -> dict[str, Any]:
    try:
        snapshot = _project_git_repository().working_tree_snapshot()
    except GitCommandError as exc:
        return {
            "status": "unavailable",
            "base_revision": None,
            "error": str(exc),
        }
    return {
        "status": "available",
        "base_revision": snapshot.base_revision,
        "branch": snapshot.branch,
        "is_dirty": snapshot.is_dirty,
        "diff_hash": snapshot.diff_hash,
        "dirty_tree_policy": snapshot.dirty_tree_policy,
    }


def _project_git_repository() -> GitRepository:
    base_repository = GitRepository(_PROJECT_ROOT)
    git_dir = _PROJECT_ROOT / ".git"
    return GitRepository(
        _PROJECT_ROOT,
        git_executable=(
            *base_repository.git_command,
            "--git-dir",
            str(git_dir),
            "--work-tree",
            str(_PROJECT_ROOT),
        ),
    )


def _run_review_proposal(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    domain_adapter_registry: DomainAdapterRegistry | None = None,
    domain_review_policy: DomainReviewPolicy | None = None,
    require_executable_patch: bool = False,
    proposal_critic: ProposalCritic | None = None,
    critic_requirement: CriticRequirement = "not_required",
    proposal_review_context_factory: ProposalReviewContextFactory | None = None,
    patch_continuation_attempt_recorder: (
        PatchContinuationAttemptRecorder | None
    ) = None,
) -> ModAgentState:
    proposal = _read_artifact_data(state, artifact_store, "patch_proposal")
    if proposal.get("context_pack", {}).get("task_agent_protocol") == "v2":
        review = ProposalReviewer().review_v2(proposal, store=artifact_store,
            feature_spec_ref=asdict(_artifact_ref_from_state(state, "feature_spec")))
        ref = artifact_store.write(feature_id=state["feature_id"], artifact_type="proposal_review",
            payload=review.to_artifact_payload(), status="success",
            upstream_refs=[_artifact_ref_from_state(state, "patch_proposal")])
        selected = _copy_state(state)
        selected["current_stage"] = "proposal_review"
        selected["review_summary"] = {"decision": review.decision, "allowed_next_stage": review.allowed_next_stage,
            "num_findings": 0, "critic_gate": review.critic_gate,
            "semantic_assurance": review.semantic_assurance.to_dict(), "human_semantic_review_required": True}
        _record_artifact(selected, "proposal_review", ref)
        return selected
    implementation_plan = _read_artifact_data(
        state,
        artifact_store,
        "implementation_plan",
    )
    evidence_pack = _read_artifact_data(state, artifact_store, "evidence_pack")
    feature_spec = _read_artifact_data(state, artifact_store, "feature_spec")
    repair_plan = (
        _read_artifact_data(state, artifact_store, "repair_plan")
        if "repair_plan" in state["artifacts"]
        else None
    )
    if repair_plan is not None and proposal_review_context_factory is not None:
        implementation_plan, evidence_pack = proposal_review_context_factory(
            state,
            implementation_plan,
            evidence_pack,
        )
        if not isinstance(implementation_plan, dict) or not isinstance(
            evidence_pack, dict
        ):
            raise TypeError(
                "Proposal review context factory must return implementation and "
                "evidence dictionaries."
            )
    proposal_ref = _artifact_ref_from_state(state, "patch_proposal")
    critic_payload: dict[str, Any] | None = None
    critic_ref: ArtifactRef | None = None
    if critic_requirement == "required" and proposal_critic is not None:
        try:
            critic_assessment = proposal_critic.assess(
                proposal,
                subject_artifact_ref=proposal_ref.path,
                subject_artifact_hash=proposal_ref.artifact_hash,
            )
        except Exception as exc:  # fail closed at the external Critic boundary
            critic_assessment = CriticAssessment(
                assessor_id=type(proposal_critic).__name__ or "unidentified_critic",
                assessor_kind="unresolved",
                subject_artifact_ref=proposal_ref.path,
                subject_artifact_hash=proposal_ref.artifact_hash,
                dimension_scores={dimension: 0 for dimension in CRITIC_DIMENSIONS},
                total_score=0,
                verdict="unresolved",
                findings=(
                    CriticFinding(
                        dimension="evidence_grounding",
                        level="blocking",
                        message=f"Critic execution failed with {type(exc).__name__}.",
                    ),
                ),
                limitations=("The Critic result was not available for this review.",),
            )
        critic_payload = critic_assessment.to_artifact_payload()
        critic_ref = artifact_store.write(
            feature_id=proposal["feature_id"],
            artifact_type="critic_assessment",
            payload=critic_payload,
            status="success" if critic_assessment.verdict == "pass" else "blocked",
            upstream_refs=[
                _artifact_ref_from_state(state, "feature_spec"),
                _artifact_ref_from_state(state, "evidence_pack"),
                _artifact_ref_from_state(state, "implementation_plan"),
                proposal_ref,
            ],
            exact_revision_upstream_types={"patch_proposal"},
        )

    review = ProposalReviewer().review(
        proposal,
        evidence_pack=evidence_pack,
        repair_plan=repair_plan,
        implementation_plan=implementation_plan,
        feature_spec=feature_spec,
        artifact_hashes=dict(state["artifact_hashes"]),
        critic_assessment=critic_payload,
        require_critic_assessment=critic_requirement == "required",
        domain_policy=(
            domain_review_policy
            if domain_review_policy is not None
            else _domain_review_policy_from_state(
                state,
                domain_adapter_registry,
            )
        ),
        permission_level="dry_run_write",
        require_executable_patch=require_executable_patch,
    )
    artifact_ref = artifact_store.write(
        feature_id=proposal["feature_id"],
        artifact_type="proposal_review",
        payload=review.to_artifact_payload(),
        status="success" if review.allowed_next_stage == "dry_run_apply" else "blocked",
        upstream_refs=[proposal_ref, *([critic_ref] if critic_ref is not None else [])],
    )
    if patch_continuation_attempt_recorder is not None:
        patch_continuation_attempt_recorder(
            {
                "phase": "proposal_review_committed",
                "proposal_admitted": review.allowed_next_stage == "dry_run_apply",
                "artifacts": {
                    "proposal_review": {
                        "feature_id": artifact_ref.feature_id,
                        "artifact_type": artifact_ref.artifact_type,
                        "path": artifact_ref.path,
                        "artifact_hash": artifact_ref.artifact_hash,
                    }
                },
            }
        )
    next_state = _copy_state(state)
    next_state["current_stage"] = "proposal_review"
    next_state["review_summary"] = {
        "decision": review.decision,
        "allowed_next_stage": review.allowed_next_stage,
        "num_findings": len(review.findings),
        "critic_gate": dict(review.critic_gate),
        "semantic_assurance": review.semantic_assurance.to_dict(),
        "human_semantic_review_required": (
            review.human_semantic_review_required
        ),
    }
    _record_artifact(next_state, "proposal_review", artifact_ref)
    if critic_ref is not None:
        _record_artifact(next_state, "critic_assessment", critic_ref)
    if review.allowed_next_stage != "dry_run_apply":
        next_state["blockers"].extend(
            {
                "stage": "proposal_review",
                "code": finding.rule_id.upper(),
                "message": finding.message,
            }
            for finding in review.findings
            if finding.level == "blocking"
        )
    return next_state


def _domain_review_policy_from_state(
    state: ModAgentState,
    domain_adapter_registry: DomainAdapterRegistry | None,
) -> DomainReviewPolicy | None:
    """Resolve the review policy for a workflow run."""

    registry = domain_adapter_registry or _default_domain_adapter_registry()
    target_domain = state.get("target_domain")
    if not target_domain:
        return None

    adapter_name = target_domain
    if target_domain == "gtfo" and GTFO_DOMAIN_NAME in registry.names():
        adapter_name = GTFO_DOMAIN_NAME

    try:
        adapter = registry.get(adapter_name)
    except KeyError:
        adapter = UnsupportedDomainAdapter()

    return adapter.identify_project(".").review_policy()


def _run_dynamic_human_review_gate(
    state: ModAgentState,
    artifact_store: ArtifactStore,
) -> ModAgentState:
    review = _read_artifact_data(state, artifact_store, "proposal_review")
    if review.get("allowed_next_stage") != "dry_run_apply":
        return _copy_state(state)
    if state["approvals"].get("review") is True:
        return _copy_state(state)

    request = _dynamic_review_interrupt_request(state, artifact_store)
    resume_value = interrupt(request.to_dict())

    checkpoint_store = WorkflowCheckpointStore(artifact_store)
    checkpoint_ref = checkpoint_store.create_checkpoint(
        state,
        interrupt_reason=request.reason,
        resume_allowed_fields=("approvals.review",),
    )
    checkpoint = artifact_store.read(checkpoint_ref)["data"]
    payload = _resume_payload_from_dynamic_value(
        resume_value,
        state=state,
        checkpoint=checkpoint,
    )
    decision, resume_report_ref = checkpoint_store.resume(checkpoint_ref, payload)

    next_state = _copy_state(state)
    _record_artifact(next_state, "workflow_checkpoint", checkpoint_ref)
    _record_artifact(next_state, "resume_report", resume_report_ref)
    approvals = decision.applied_updates.get("approvals")
    if isinstance(approvals, dict):
        next_state["approvals"] = {
            **dict(next_state["approvals"]),
            **approvals,
        }
    if not decision.allowed:
        next_state["blockers"].extend(decision.blockers)
    return next_state


def _validate_graph_edit_authority_exact_inputs(
    *,
    artifact_store: ArtifactStore,
    feature_id: str,
    request: Mapping[str, Any],
    request_envelope: Mapping[str, Any],
) -> None:
    upstreams = request_envelope.get("inputs", {}).get(
        "upstream_artifacts",
        (),
    )
    if not isinstance(upstreams, list):
        raise ValueError("Edit Authority Request upstreams are invalid.")
    try:
        bindings = bind_edit_authority_exact_input_upstreams(request, upstreams)
    except InvalidEditAuthorityApprovalIdentityError as exc:
        raise ValueError(str(exc)) from exc
    if not bindings:
        return
    refs_by_identity: dict[tuple[str, str], list[ArtifactRef]] = {}
    for item in upstreams:
        if not isinstance(item, Mapping):
            raise ValueError("Edit Authority Request upstream is invalid.")
        ref = ArtifactRef(
            feature_id=feature_id,
            artifact_type=str(item.get("artifact_type", "")),
            path=str(item.get("path", "")),
            artifact_hash=str(item.get("artifact_hash", "")),
        )
        refs_by_identity.setdefault(
            (ref.artifact_type, ref.artifact_hash),
            [],
        ).append(ref)
    latest_by_type = {
        ref.artifact_type: ref
        for ref in artifact_store.list_feature_artifacts(feature_id)
    }
    input_facts: list[dict[str, Any]] = []
    for requirement, _ in bindings:
        key = (
            str(requirement.get("artifact_type")),
            str(requirement.get("artifact_hash")),
        )
        matches = refs_by_identity.get(key)
        if not matches:
            raise ValueError("Edit Authority exact input lineage is incomplete.")
        ref = matches.pop(0)
        artifact_store.read(ref)
        latest = latest_by_type.get(ref.artifact_type)
        input_facts.append(
            {
                "role": requirement["role"],
                "artifact_type": ref.artifact_type,
                "artifact_hash": ref.artifact_hash,
                "registered": True,
                "latest_artifact_hash": (
                    latest.artifact_hash if latest is not None else None
                ),
                "recursive_fresh": artifact_store.check_freshness(ref).is_fresh,
                "exact_recursive_fresh": (
                    artifact_store.check_exact_revision_freshness(ref).is_fresh
                ),
            }
        )
    try:
        validate_edit_authority_exact_input_admission(request, input_facts)
    except InvalidEditAuthorityApprovalIdentityError as exc:
        raise ValueError(str(exc)) from exc


def _finish_stagger_edit_authority_gate(
    state: ModAgentState,
    *,
    capability: EditAuthorityCapabilityDeclaration,
    feature_id: str,
    approval_identity_hash: str,
    store_steps: tuple[EditAuthorityStoreStepV1, ...],
    payload_decision: str | None,
    resume_decision_status: str | None,
    resume_decision_allowed: bool | None,
    execution_failure_stage: str | None = None,
    platform_blockers: tuple[dict[str, Any], ...] = (),
) -> ModAgentState:
    """Project the sole machine result, then derive legacy Graph state."""

    machine_result = project_edit_authority_review_result(
        capability=capability,
        feature_id=feature_id,
        approval_identity_hash=approval_identity_hash,
        store_steps=store_steps,
        payload_decision=payload_decision,
        resume_decision_status=resume_decision_status,
        resume_decision_allowed=resume_decision_allowed,
        execution_failure_stage=execution_failure_stage,
        platform_blockers=platform_blockers,
    )
    next_state = _copy_state(state)
    next_state["edit_authority_review_result"] = machine_result.to_dict()
    for ref in (
        *machine_result.protocol_outputs,
        *machine_result.materialization_outputs,
    ):
        _record_artifact(next_state, ref.artifact_type, ref)

    if (
        machine_result.platform_status == "completed"
        and machine_result.review_outcome == "approved"
        and machine_result.materialization_status == "completed"
    ):
        next_state["current_stage"] = capability.terminal_stage
        next_state["approvals"] = {
            **dict(next_state["approvals"]),
            "edit_authority": True,
        }
        return next_state

    next_state["current_stage"] = "blocked"
    details = (
        machine_result.platform_blockers
        or machine_result.review_reasons
        or (
            {
                "stage": machine_result.failure_stage
                or "edit_authority_approval",
                "code": "EDIT_AUTHORITY_REVIEW_NOT_COMPLETED",
                "message": (
                    "Edit Authority review stopped before a complete approved "
                    "materialization."
                ),
            },
        )
    )
    next_state["blockers"].extend(copy.deepcopy(list(details)))
    return next_state


def _run_stagger_edit_authority_gate(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    request_ref: ArtifactRef,
) -> ModAgentState:
    """Approve one exact request in-graph, then project bounded records."""

    feature_id = state["feature_id"] or state["request_id"]
    if request_ref.feature_id != feature_id:
        raise ValueError(
            "EditAuthorityRequest feature does not match graph state."
        )
    request_envelope = artifact_store.read_for_stage(
        request_ref,
        consumer_stage="edit_authority_approval",
    )
    request = request_envelope["data"]
    if (
        request.get("feature_id") != feature_id
        or request.get("run_id") != state["run_id"]
        or request.get("authority_granted") is not False
    ):
        raise ValueError(
            "EditAuthorityRequest identity or no-authority boundary is invalid."
        )
    _validate_graph_edit_authority_exact_inputs(
        artifact_store=artifact_store,
        feature_id=feature_id,
        request=request,
        request_envelope=request_envelope,
    )

    gate_state = _copy_state(state)
    gate_state["current_stage"] = "edit_authority_approval"
    _record_artifact(gate_state, "edit_authority_request", request_ref)
    request_payload = request.get("request_payload")
    try:
        capability, _, _ = validate_edit_authority_request_identity(
            request,
            allow_legacy=False,
        )
        identity_sections = edit_authority_identity_sections(
            request,
            allow_legacy=False,
        )
    except (
        InvalidEditAuthorityApprovalIdentityError,
        UnsupportedEditAuthorityCapabilityError,
    ) as exc:
        raise ValueError(str(exc)) from exc
    if (
        capability.checkpoint_input_mode
        == "request_and_registered_upstreams"
    ):
        for upstream in request_envelope.get("inputs", {}).get(
            "upstream_artifacts",
            (),
        ):
            if (
                isinstance(upstream, dict)
                and isinstance(upstream.get("artifact_type"), str)
                and isinstance(upstream.get("path"), str)
                and isinstance(
                    upstream.get("artifact_hash"),
                    str,
                )
            ):
                _record_artifact(
                    gate_state,
                    str(upstream["artifact_type"]),
                    ArtifactRef(
                        feature_id=feature_id,
                        artifact_type=str(
                            upstream["artifact_type"]
                        ),
                        path=str(upstream["path"]),
                        artifact_hash=str(
                            upstream["artifact_hash"]
                        ),
                    ),
                )
    execution_contract = identity_sections["execution_contract_summary"]
    allowed_actions = list(execution_contract["allowed_actions"])
    forbidden_actions = list(execution_contract["forbidden_actions"])
    interrupt_request = InterruptRequest(
        interrupt_id=f"interrupt_{uuid4().hex}",
        run_id=state["run_id"],
        feature_id=feature_id,
        stage="edit_authority_approval",
        reason=(
            "exact EditAuthorityRequest requires human approval before "
            "Registry projection"
        ),
        payload={
            "node_name": "stagger_edit_authority_gate",
            "stage": "edit_authority_approval",
            "required_decision": ["approve", "reject"],
            "edit_authority_request_revision": {
                "feature_id": request_ref.feature_id,
                "path": request_ref.path,
                "artifact_hash": request_ref.artifact_hash,
            },
            "request_hash": request["request_hash"],
            "approval_identity_mode": identity_sections["identity_mode"],
            "approval_identity_schema_version": identity_sections[
                "identity_schema_version"
            ],
            "allowed_actions": allowed_actions,
            "forbidden_actions": forbidden_actions,
        },
        resume_allowed_fields=("approvals.edit_authority",),
        artifact_refs=tuple(gate_state["artifacts"].values()),
    )
    resume_value = interrupt(interrupt_request.to_dict())

    checkpoint_store = WorkflowCheckpointStore(artifact_store)
    store_steps: list[EditAuthorityStoreStepV1] = []
    try:
        checkpoint_result = checkpoint_store.create_checkpoint_with_result(
            gate_state,
            interrupt_reason=interrupt_request.reason,
            resume_allowed_fields=("approvals.edit_authority",),
        )
    except Exception as exc:
        return _finish_stagger_edit_authority_gate(
            gate_state,
            capability=capability,
            feature_id=feature_id,
            approval_identity_hash=str(request["request_hash"]),
            store_steps=(),
            payload_decision=None,
            resume_decision_status=None,
            resume_decision_allowed=None,
            execution_failure_stage="checkpoint_construction",
            platform_blockers=(
                {
                    "stage": "checkpoint_construction",
                    "code": "EDIT_AUTHORITY_CHECKPOINT_CONSTRUCTION_FAILED",
                    "message": str(exc),
                },
            ),
        )
    store_steps.append(
        EditAuthorityStoreStepV1(
            role="workflow_checkpoint",
            store_result=checkpoint_result,
        )
    )
    checkpoint_ref = checkpoint_result.artifact_ref
    if checkpoint_result.outcome != "committed" or checkpoint_ref is None:
        return _finish_stagger_edit_authority_gate(
            gate_state,
            capability=capability,
            feature_id=feature_id,
            approval_identity_hash=str(request["request_hash"]),
            store_steps=tuple(store_steps),
            payload_decision=None,
            resume_decision_status=None,
            resume_decision_allowed=None,
        )
    try:
        checkpoint = artifact_store.read(checkpoint_ref)["data"]
        payload = _resume_payload_from_dynamic_value(
            resume_value,
            state=gate_state,
            checkpoint=checkpoint,
        )
        decision, report_result = checkpoint_store.resume_with_result(
            checkpoint_ref,
            payload,
        )
    except Exception as exc:
        return _finish_stagger_edit_authority_gate(
            gate_state,
            capability=capability,
            feature_id=feature_id,
            approval_identity_hash=str(request["request_hash"]),
            store_steps=tuple(store_steps),
            payload_decision=None,
            resume_decision_status=None,
            resume_decision_allowed=None,
            execution_failure_stage="resume_validation",
            platform_blockers=(
                {
                    "stage": "resume_validation",
                    "code": "EDIT_AUTHORITY_RESUME_VALIDATION_FAILED",
                    "message": str(exc),
                },
            ),
        )
    store_steps.append(
        EditAuthorityStoreStepV1(
            role="resume_report",
            store_result=report_result,
        )
    )
    report_ref = report_result.artifact_ref
    if report_result.outcome != "committed" or report_ref is None:
        return _finish_stagger_edit_authority_gate(
            gate_state,
            capability=capability,
            feature_id=feature_id,
            approval_identity_hash=str(request["request_hash"]),
            store_steps=tuple(store_steps),
            payload_decision=payload.decision,
            resume_decision_status=decision.status,
            resume_decision_allowed=decision.allowed,
            platform_blockers=tuple(decision.blockers),
        )
    if not decision.allowed or payload.decision != "approve":
        return _finish_stagger_edit_authority_gate(
            gate_state,
            capability=capability,
            feature_id=feature_id,
            approval_identity_hash=str(request["request_hash"]),
            store_steps=tuple(store_steps),
            payload_decision=payload.decision,
            resume_decision_status=decision.status,
            resume_decision_allowed=decision.allowed,
            platform_blockers=tuple(decision.blockers),
        )

    try:
        materialized = materialize_stagger_edit_authority_approval(
            artifact_store_root=artifact_store.root,
            feature_id=feature_id,
            edit_authority_request_path=request_ref.path,
            edit_authority_request_hash=request_ref.artifact_hash,
            resume_report_path=report_ref.path,
            resume_report_hash=report_ref.artifact_hash,
        )
        materialization_steps = materialized.get("store_steps")
        if not isinstance(materialization_steps, tuple) or any(
            not isinstance(step, EditAuthorityStoreStepV1)
            for step in materialization_steps
        ):
            raise StaggerEditAuthorityApprovalError(
                "Edit Authority materializer omitted typed Store results."
            )
        store_steps.extend(materialization_steps)
    except StaggerEditAuthorityMaterializationError as exc:
        store_steps.extend(exc.store_steps)
        return _finish_stagger_edit_authority_gate(
            gate_state,
            capability=capability,
            feature_id=feature_id,
            approval_identity_hash=str(request["request_hash"]),
            store_steps=tuple(store_steps),
            payload_decision=payload.decision,
            resume_decision_status=decision.status,
            resume_decision_allowed=decision.allowed,
            execution_failure_stage=exc.failure_stage,
            platform_blockers=(
                {
                    "stage": exc.failure_stage,
                    "code": "EDIT_AUTHORITY_MATERIALIZATION_FAILED",
                    "message": str(exc),
                },
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return _finish_stagger_edit_authority_gate(
            gate_state,
            capability=capability,
            feature_id=feature_id,
            approval_identity_hash=str(request["request_hash"]),
            store_steps=tuple(store_steps),
            payload_decision=payload.decision,
            resume_decision_status=decision.status,
            resume_decision_allowed=decision.allowed,
            execution_failure_stage="materialization_validation",
            platform_blockers=(
                {
                    "stage": "materialization_validation",
                    "code": "EDIT_AUTHORITY_MATERIALIZATION_VALIDATION_FAILED",
                    "message": str(exc),
                },
            ),
        )

    return _finish_stagger_edit_authority_gate(
        gate_state,
        capability=capability,
        feature_id=feature_id,
        approval_identity_hash=str(request["request_hash"]),
        store_steps=tuple(store_steps),
        payload_decision=payload.decision,
        resume_decision_status=decision.status,
        resume_decision_allowed=decision.allowed,
    )


def _materialize_feature_spec_revision(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    projection: Mapping[str, Any],
    approved_payload_hash: str,
    approved_projection_hash: str,
) -> ModAgentState:
    """Materialize one approved FeatureSpec revision through its graph node."""

    if state["approvals"].get(_FEATURE_SPEC_REVISION_APPROVAL_KEY) is not True:
        raise ValueError(
            "FeatureSpec revision materialization lacks node-level human approval."
        )
    if set(projection) != _FEATURE_SPEC_REVISION_PROJECTION_FIELDS:
        raise ValueError(
            "FeatureSpec revision projection fields do not match the approved shape."
        )
    payload = projection.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("FeatureSpec revision payload must be an object.")
    if set(payload) != _FEATURE_SPEC_REVISION_PAYLOAD_FIELDS:
        raise ValueError(
            "FeatureSpec revision payload fields do not match the approved shape."
        )

    payload_hash = _canonical_sha256(payload)
    projection_hash = _canonical_sha256(projection)
    if payload_hash != approved_payload_hash:
        raise ValueError("FeatureSpec revision payload hash is not approved.")
    if projection_hash != approved_projection_hash:
        raise ValueError("FeatureSpec revision projection hash is not approved.")

    feature_id = state.get("feature_id")
    if not isinstance(feature_id, str) or not feature_id:
        raise ValueError("FeatureSpec revision state feature_id is missing.")
    if (
        projection.get("schema_version") != "1.0"
        or projection.get("artifact_type") != "feature_spec"
        or projection.get("feature_id") != feature_id
        or projection.get("producer_stage") != "feature_spec"
        or projection.get("status") != "success"
        or projection.get("blockers") != []
        or payload.get("feature_id") != feature_id
        or payload.get("node_name") != "draft_feature_spec"
        or payload.get("stage") != "feature_spec"
    ):
        raise ValueError("FeatureSpec revision identity metadata drifted.")

    contract = artifact_store.registry.validate_write(
        artifact_type="feature_spec",
        status="success",
        payload=payload,
    )
    if (
        contract.schema_version != projection["schema_version"]
        or contract.producer_stage != projection["producer_stage"]
    ):
        raise ValueError("FeatureSpec revision Registry contract drifted.")
    _validate_feature_spec_revision_semantics(payload)

    inputs = projection.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {"upstream_artifacts"}:
        raise ValueError("FeatureSpec revision inputs do not match the approved shape.")
    upstreams = inputs.get("upstream_artifacts")
    if not isinstance(upstreams, list) or len(upstreams) != 1:
        raise ValueError(
            "FeatureSpec revision requires exactly one RequestPlan upstream."
        )
    upstream = upstreams[0]
    if (
        not isinstance(upstream, Mapping)
        or set(upstream)
        != {"artifact_type", "path", "artifact_hash", "require_latest"}
        or upstream.get("artifact_type") != "request_planning_report"
        or upstream.get("require_latest") is not True
    ):
        raise ValueError("FeatureSpec revision RequestPlan upstream drifted.")
    request_ref = ArtifactRef(
        feature_id=feature_id,
        artifact_type="request_planning_report",
        path=str(upstream["path"]),
        artifact_hash=str(upstream["artifact_hash"]),
    )
    if (
        state["artifacts"].get("request_planning_report") != request_ref.path
        or state["artifact_hashes"].get("request_planning_report")
        != request_ref.artifact_hash
    ):
        raise ValueError(
            "FeatureSpec revision state does not bind the approved RequestPlan."
        )
    artifact_store.read(request_ref)
    latest_request = _latest_feature_artifact_ref(
        artifact_store,
        feature_id=feature_id,
        artifact_type="request_planning_report",
    )
    if latest_request.artifact_hash != request_ref.artifact_hash:
        raise ValueError("FeatureSpec revision RequestPlan is not latest.")

    predecessor_hash = payload.get("supersedes_artifact_hash")
    if not isinstance(predecessor_hash, str) or not predecessor_hash:
        raise ValueError("FeatureSpec revision predecessor identity is missing.")
    latest_feature_spec = _latest_feature_artifact_ref(
        artifact_store,
        feature_id=feature_id,
        artifact_type="feature_spec",
    )
    artifact_store.read(latest_feature_spec)
    if latest_feature_spec.artifact_hash != predecessor_hash:
        raise ValueError(
            "FeatureSpec revision predecessor is not the current latest revision."
        )

    trace_refs = projection.get("trace_refs")
    if not isinstance(trace_refs, list) or any(
        not isinstance(item, str) or not item for item in trace_refs
    ):
        raise ValueError("FeatureSpec revision trace refs are invalid.")
    expected_artifact_hash = artifact_store._stable_hash(
        {
            "schema_version": projection["schema_version"],
            "artifact_type": projection["artifact_type"],
            "feature_id": feature_id,
            "producer_stage": projection["producer_stage"],
            "status": projection["status"],
            "data": dict(payload),
            "inputs": dict(inputs),
            "blockers": [],
            "trace_refs": list(trace_refs),
        }
    )
    before_revisions = artifact_store.list_feature_artifact_revisions(
        feature_id,
        "feature_spec",
    )
    if any(ref.artifact_hash == expected_artifact_hash for ref in before_revisions):
        raise ValueError(
            "FeatureSpec revision approval was already materialized and cannot replay."
        )

    artifact_ref = artifact_store.write(
        feature_id=feature_id,
        artifact_type="feature_spec",
        payload=payload,
        status="success",
        blockers=[],
        trace_refs=list(trace_refs),
        upstream_refs=[request_ref],
    )
    after_revisions = artifact_store.list_feature_artifact_revisions(
        feature_id,
        "feature_spec",
    )
    if (
        artifact_ref.artifact_hash != expected_artifact_hash
        or len(after_revisions) != len(before_revisions) + 1
    ):
        raise ValueError(
            "FeatureSpec revision materialization did not create exactly one revision."
        )

    next_state = _copy_state(state)
    next_state["current_stage"] = "feature_spec"
    next_state["next_stage"] = None
    _record_artifact(next_state, "feature_spec", artifact_ref)
    return next_state


def _validate_feature_spec_revision_semantics(payload: Mapping[str, Any]) -> None:
    hypotheses = payload.get("operator_guided_technical_hypotheses")
    if not isinstance(hypotheses, Mapping):
        raise ValueError("FeatureSpec revision hypotheses are invalid.")
    items = hypotheses.get("hypotheses")
    if not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items):
        raise ValueError("FeatureSpec revision active hypotheses are invalid.")
    active_ids = {item.get("id") for item in items}
    if active_ids & _DEFERRED_CORRELATION_HYPOTHESIS_IDS:
        raise ValueError(
            "FeatureSpec revision reintroduced deferred correlation hypotheses."
        )
    evidence_requirements = payload.get("evidence_requirements")
    runtime_plan = payload.get("runtime_verification_plan")
    if not isinstance(evidence_requirements, list) or not isinstance(runtime_plan, list):
        raise ValueError("FeatureSpec revision verification fields are invalid.")
    if any(
        token in item
        for item in evidence_requirements
        if isinstance(item, str)
        for token in (
            "one-time consumption",
            "Dam_EnemyDamageBase.BulletDamage may",
            "ES_Hitreact.DoHitReact may",
        )
    ) or any(
        token in item
        for item in runtime_plan
        if isinstance(item, str)
        for token in ("correlation/deduplication", "same physical hit")
    ):
        raise ValueError(
            "FeatureSpec revision reintroduced deferred correlation requirements."
        )


def _latest_feature_artifact_ref(
    artifact_store: ArtifactStore,
    *,
    feature_id: str,
    artifact_type: str,
) -> ArtifactRef:
    refs = [
        ref
        for ref in artifact_store.list_feature_artifacts(feature_id)
        if ref.artifact_type == artifact_type
    ]
    if len(refs) != 1:
        raise ValueError(
            f"FeatureSpec revision expected one latest {artifact_type} Artifact."
        )
    return refs[0]


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _run_stagger_repair_planning_gate(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    request_ref: ArtifactRef,
) -> ModAgentState:
    """Approve one exact planning Request before materialization or Planner."""

    feature_id = state["feature_id"] or state["request_id"]
    if request_ref.feature_id != feature_id:
        raise ValueError(
            "Repair planning Request feature does not match graph state."
        )
    request_envelope = artifact_store.read_for_stage(
        request_ref,
        consumer_stage="repair_planning_approval",
    )
    request = request_envelope["data"]
    if (
        request.get("feature_id") != feature_id
        or request.get("run_id") != state["run_id"]
        or request.get("authority_granted") is not False
        or request.get("artifact_materialization_allowed") is not False
        or request.get("repair_planner_allowed") is not False
    ):
        raise ValueError(
            "Repair planning Request identity or no-authority boundary is invalid."
        )
    gate_state = _copy_state(state)
    gate_state["current_stage"] = "repair_planning_approval"
    _record_artifact(
        gate_state,
        "repair_planning_request",
        request_ref,
    )
    interrupt_request = InterruptRequest(
        interrupt_id=f"interrupt_{uuid4().hex}",
        run_id=state["run_id"],
        feature_id=feature_id,
        stage="repair_planning_approval",
        reason=(
            "exact Repair planning Request requires human approval before "
            "Artifact materialization or RepairPlanner invocation"
        ),
        payload={
            "node_name": "stagger_repair_planning_gate",
            "stage": "repair_planning_approval",
            "required_decision": ["approve", "reject"],
            "repair_planning_authority_request_revision": {
                "feature_id": request_ref.feature_id,
                "path": request_ref.path,
                "artifact_hash": request_ref.artifact_hash,
            },
            "request_hash": request["request_hash"],
            "allowed_actions": [
                "materialize_root_cause_evidence",
                "materialize_repair_hypothesis_set",
                "invoke_repair_planner_for_semantic_decision",
                "materialize_repair_plan",
            ],
            "forbidden_actions": [
                "modify_source",
                "build",
                "launch_process",
                "write_game_directory",
                "claim_product_acceptance",
                "claim_audio_audibility",
            ],
        },
        resume_allowed_fields=("approvals.repair_planning",),
        artifact_refs=tuple(gate_state["artifacts"].values()),
    )
    resume_value = interrupt(interrupt_request.to_dict())
    checkpoint_store = WorkflowCheckpointStore(artifact_store)
    checkpoint_ref = checkpoint_store.create_checkpoint(
        gate_state,
        interrupt_reason=interrupt_request.reason,
        resume_allowed_fields=("approvals.repair_planning",),
    )
    checkpoint = artifact_store.read(checkpoint_ref)["data"]
    payload = _resume_payload_from_dynamic_value(
        resume_value,
        state=gate_state,
        checkpoint=checkpoint,
    )
    decision, report_ref = checkpoint_store.resume(
        checkpoint_ref,
        payload,
    )
    next_state = _copy_state(gate_state)
    _record_artifact(next_state, "workflow_checkpoint", checkpoint_ref)
    _record_artifact(next_state, "resume_report", report_ref)
    if not decision.allowed or payload.decision != "approve":
        next_state["current_stage"] = "blocked"
        next_state["blockers"].extend(
            decision.blockers
            or (
                {
                    "stage": "repair_planning_approval",
                    "code": "REPAIR_PLANNING_AUTHORITY_REJECTED",
                    "message": (
                        "Human review did not approve the exact Repair "
                        "planning Request."
                    ),
                },
            )
        )
        return next_state
    try:
        materialized = materialize_stagger_planning_approval(
            artifact_store_root=artifact_store.root,
            feature_id=feature_id,
            repair_planning_authority_request_path=request_ref.path,
            repair_planning_authority_request_hash=(
                request_ref.artifact_hash
            ),
            resume_report_path=report_ref.path,
            resume_report_hash=report_ref.artifact_hash,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        next_state["current_stage"] = "blocked"
        next_state["blockers"].append(
            {
                "stage": "repair_planning_approval",
                "code": "REPAIR_PLANNING_APPROVED_NOT_MATERIALIZED",
                "message": str(exc),
            }
        )
        return next_state
    for artifact_type, field_name in (
        (
            "repair_planning_approval",
            "repair_planning_authority_approval_revision",
        ),
        ("root_cause_evidence", "root_cause_evidence_revision"),
        (
            "repair_hypothesis_set",
            "repair_hypothesis_set_revision",
        ),
        ("repair_plan", "repair_plan_revision"),
    ):
        revision = materialized[field_name]
        _record_artifact(
            next_state,
            artifact_type,
            ArtifactRef(
                feature_id=feature_id,
                artifact_type=artifact_type,
                path=str(revision["path"]),
                artifact_hash=str(revision["artifact_hash"]),
            ),
        )
    next_state["current_stage"] = "repair_plan"
    next_state["approvals"] = {
        **dict(next_state["approvals"]),
        "repair_planning": True,
    }
    semantic = materialized["semantic_decision"]
    if semantic.get("decision") != "repair_ready":
        next_state["blockers"].extend(
            [
                dict(blocker)
                for blocker in semantic.get("blockers", ())
                if isinstance(blocker, dict)
            ]
        )
    return next_state


def _dynamic_review_interrupt_request(
    state: ModAgentState,
    artifact_store: ArtifactStore,
) -> InterruptRequest:
    feature_id = state["feature_id"] or state["request_id"]
    return InterruptRequest(
        interrupt_id=f"interrupt_{uuid4().hex}",
        run_id=state["run_id"],
        feature_id=feature_id,
        stage="proposal_review",
        reason="proposal review requires human approval before dry run",
        payload={
            "node_name": "human_review_gate",
            "stage": "proposal_review",
            "required_decision": ["approve", "reject"],
            "resume_binding": _dynamic_review_resume_binding(state, artifact_store),
            "artifact_hashes": dict(state["artifact_hashes"]),
        },
        resume_allowed_fields=("approvals.review",),
        artifact_refs=tuple(state["artifacts"].values()),
    )


def _dynamic_review_resume_binding(
    state: ModAgentState,
    artifact_store: ArtifactStore,
) -> dict[str, Any]:
    proposal_ref = _artifact_ref_from_state(state, "patch_proposal")
    review_ref = _artifact_ref_from_state(state, "proposal_review")
    proposal_payload = artifact_store.read(proposal_ref)["data"]
    serialized = json.dumps(
        proposal_payload.get("diff_intent") or [],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "patch_proposal_ref": proposal_ref.path,
        "patch_proposal_hash": proposal_ref.artifact_hash,
        "proposal_review_ref": review_ref.path,
        "proposal_review_hash": review_ref.artifact_hash,
        "diff_hash": "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _resume_payload_from_dynamic_value(
    resume_value: Any,
    *,
    state: ModAgentState,
    checkpoint: dict[str, Any],
) -> ResumePayload:
    value = resume_value if isinstance(resume_value, dict) else {}
    return ResumePayload(
        resume_id=str(value.get("resume_id") or f"resume_{uuid4().hex}"),
        run_id=str(value.get("run_id") or state["run_id"]),
        stage=str(value.get("stage") or checkpoint.get("checkpoint_stage") or state["current_stage"]),
        decision=str(value.get("decision") or "reject"),
        approved_by=str(value.get("approved_by") or ""),
        approved_at=str(value.get("approved_at") or ""),
        artifact_refs=tuple(value.get("artifact_refs") or ()),
        diff_hash=value.get("diff_hash"),
        notes=str(value.get("notes") or ""),
        authority_request_hash=value.get("authority_request_hash"),
    )


def _default_domain_adapter_registry() -> DomainAdapterRegistry:
    """Return the default domain policy registry for workflow review gates."""

    return DomainAdapterRegistry((GTFOAdapter(), UnsupportedDomainAdapter()))


def _runtime_evidence_factory_from_settings(
    settings: Any | None,
    *,
    allow_game_launch: bool,
) -> RuntimeEvidenceFactory | None:
    if settings is None:
        return None
    runtime = getattr(settings, "runtime", None)
    gtfo = getattr(runtime, "gtfo", None)
    if gtfo is not None and getattr(gtfo, "enabled", False):
        return make_gtfo_runtime_evidence_factory_from_settings(
            settings,
            allow_game_launch=allow_game_launch,
        )
    return None


def _run_dry_run_apply(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    require_review_approval: bool = False,
    dry_run_factory: PatchExecutionFactory | None = None,
) -> ModAgentState:
    proposal = _read_artifact_data(state, artifact_store, "patch_proposal")
    review = _read_artifact_data(state, artifact_store, "proposal_review")
    approved_by_review = review.get("allowed_next_stage") == "dry_run_apply"
    approved_by_human = state["approvals"].get("review") is True
    approved = approved_by_review and (
        not require_review_approval or approved_by_human
    )
    semantic_assurance, semantic_contract_present = (
        _semantic_assurance_for_execution(proposal, review)
    )
    human_semantic_review_required = (
        semantic_contract_present and semantic_assurance.verdict == "unproven"
    )
    changed_files = list(proposal.get("files") or [])
    diff_intent = list(proposal.get("diff_intent") or [])
    v2 = proposal.get("context_pack", {}).get("task_agent_protocol") == "v2"
    v2_payload = None
    if v2 and approved:
        from src.mod_agent.execution.patch_apply import ExactTextPatchApplier
        v2_payload = ExactTextPatchApplier().dry_run_v2(proposal=proposal, review=review, store=artifact_store)
    generated_payload = (
        v2_payload if v2 else
        dry_run_factory(
            proposal,
            _read_artifact_data(state, artifact_store, "implementation_plan"),
            review,
        )
        if dry_run_factory is not None and approved
        else None
    )
    payload = {
        "node_name": "dry_run_apply",
        "stage": "dry_run_apply",
        "message": (
            "Executable patch dry-run validated without writing source files."
            if generated_payload is not None
            else "Fake dry-run completed. No source files were written and no patch was applied."
        ),
        "execution_mode": "dry_run" if generated_payload is not None else "fake",
        "actual_write_performed": False,
        "real_patch_applied": False,
        "approved_by_review": approved_by_review,
        "human_approval_required": require_review_approval,
        "approved_by_human": approved_by_human,
        "proposal_id": proposal.get("proposal_id"),
        "review_decision": review.get("decision"),
        "semantic_assurance": semantic_assurance.to_dict(),
        "human_semantic_review_required": human_semantic_review_required,
        "human_semantic_review_completed": False,
        "human_semantic_review_result": (
            "pending" if human_semantic_review_required else "not_required"
        ),
        "human_semantic_decision_binding": None,
        "human_semantic_decision": None,
        "would_write": bool(diff_intent),
        "changed_files": changed_files,
        "diff_stats": {
            "files": len(changed_files),
            "change_units": len(diff_intent),
            "insertions": 0,
            "deletions": 0,
        },
        "limitations": [
            "Dry-run validation does not prove the isolated write or build succeeds."
            if generated_payload is not None
            else "This is an isolated fake dry-run artifact."
        ],
        **dict(generated_payload or {}),
    }
    status = "success" if approved else "blocked"
    artifact_ref = artifact_store.write(
        feature_id=proposal["feature_id"],
        artifact_type="dry_run_result",
        payload=payload,
        status=status,
        upstream_refs=[
            _artifact_ref_from_state(state, "patch_proposal"),
            _artifact_ref_from_state(state, "proposal_review"),
        ],
    )
    next_state = _copy_state(state)
    next_state["current_stage"] = "dry_run_apply"
    _record_artifact(next_state, "dry_run_result", artifact_ref)
    if v2 and approved:
        from src.mod_agent.task_agent_v2_delivery import create_v2_delivery_checkpoint
        next_state["runtime_summary"]["v2_delivery_checkpoint_ref"] = create_v2_delivery_checkpoint(
            store=artifact_store, feature_id=state["feature_id"], run_id=state["run_id"], request_id=state["request_id"],
            proposal_ref=_artifact_ref_from_state(state, "patch_proposal"),
            review_ref=_artifact_ref_from_state(state, "proposal_review"), dry_run_ref=artifact_ref)
    if not approved:
        next_state["blockers"].append(
            {
                "stage": "dry_run_apply",
                "code": "REVIEW_NOT_APPROVED",
                "message": "Dry-run cannot proceed without review approval.",
            }
        )
    return next_state


def _run_build(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    build_result_factory: BuildResultFactory | None,
    *,
    verification_operation: VerificationOperationEnvelope | None = None,
    persist_build_runtime_handoff: bool = False,
    exact_revision_upstream_types: frozenset[str] = frozenset(),
) -> ModAgentState:
    feature_id = state["feature_id"] or state["request_id"]
    build_continuation = (
        None
        if verification_operation is None
        else _verification_build_continuation(verification_operation)
    )
    if build_continuation is not None:
        preflight = preflight_verification_only_continuation(
            artifact_store,
            build_continuation,
            run_id=state["run_id"],
            request_id=state["request_id"],
            user_request=state["user_request"],
            target_mod=state["target_mod"],
        )
        if (
            state.get("feature_id") != build_continuation.feature_spec_ref.feature_id
            or preflight["status"] != "ready_for_build_authorization"
        ):
            next_state = _copy_state(state)
            next_state["current_stage"] = "build"
            next_state["build_summary"] = {
                "status": "blocked",
                "input_execution_mode": VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
                "preflight_hash": preflight["projection_hash"],
                "platform_failure_stage": "admission",
            }
            preflight_blockers = list(preflight["blockers"])
            if (
                state.get("feature_id")
                != build_continuation.feature_spec_ref.feature_id
            ):
                preflight_blockers.append(
                    {
                        "code": "VERIFICATION_STATE_FEATURE_ID_MISMATCH",
                        "message": "Graph state feature_id changed after verification admission.",
                    }
                )
            next_state["blockers"].extend(
                {
                    "stage": "build",
                    "code": blocker["code"],
                    "message": blocker["message"],
                }
                for blocker in preflight_blockers
            )
            return next_state
        input_execution_mode = VERIFICATION_ONLY_INPUT_EXECUTION_MODE
        build_upstream_refs = [
            build_continuation.feature_spec_ref,
            build_continuation.evidence_pack_ref,
        ]
    else:
        dry_run = _read_artifact_data(state, artifact_store, "dry_run_result")
        apply_result = (
            _read_artifact_data(state, artifact_store, "isolated_apply_result")
            if "isolated_apply_result" in state["artifacts"]
            else None
        )
        input_execution_mode = (
            apply_result.get("execution_mode")
            if apply_result is not None
            else dry_run.get("execution_mode")
        )
        build_upstream_refs = [
            _artifact_ref_from_state(state, "dry_run_result"),
            *(
                [_artifact_ref_from_state(state, "isolated_apply_result")]
                if "isolated_apply_result" in state["artifacts"]
                else []
            ),
        ]
    result = (
        build_result_factory(feature_id, input_execution_mode)
        if build_result_factory is not None
        else build_fake_build_result(
            feature_id=feature_id,
            input_execution_mode=input_execution_mode,
        )
    )
    try:
        payload = result.to_artifact_payload()
        post_build_preflight = (
            preflight_verification_only_continuation(
                artifact_store,
                build_continuation,
                run_id=state["run_id"],
                request_id=state["request_id"],
                user_request=state["user_request"],
                target_mod=state["target_mod"],
            )
            if build_continuation is not None
            else None
        )
        build_validation_blockers = (
            validate_verification_only_build_result(payload, post_build_preflight)
            if post_build_preflight is not None
            else []
        )
    except Exception as exc:
        if build_continuation is None:
            raise
        next_state = _copy_state(state)
        next_state["current_stage"] = "build"
        next_state["build_summary"] = {
            "status": "blocked",
            "input_execution_mode": VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
            "preflight_hash": preflight["projection_hash"],
            "platform_failure_stage": "post_build_validation",
        }
        next_state["blockers"].append(
            {
                "stage": "build",
                "code": "VERIFICATION_POST_BUILD_VALIDATION_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return next_state
    if build_continuation is not None:
        if post_build_preflight is None:
            raise AssertionError("Verification post-Build preflight is missing.")
        if (
            post_build_preflight["status"]
            != "ready_for_build_authorization"
            or post_build_preflight["projection_hash"]
            != preflight["projection_hash"]
        ):
            next_state = _copy_state(state)
            next_state["current_stage"] = "build"
            next_state["build_summary"] = {
                "status": "blocked",
                "input_execution_mode": VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
                "preflight_hash": preflight["projection_hash"],
                "post_build_preflight_hash": post_build_preflight[
                    "projection_hash"
                ],
                "platform_failure_stage": "post_build_validation",
            }
            next_state["blockers"].append(
                {
                    "stage": "build",
                    "code": "VERIFICATION_BUILD_INPUT_DRIFT",
                    "message": (
                        "Verification source or Artifact identity changed during Build."
                    ),
                }
            )
            return next_state
        if build_validation_blockers:
            next_state = _copy_state(state)
            next_state["current_stage"] = "build"
            next_state["build_summary"] = {
                "status": "blocked",
                "input_execution_mode": VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
                "preflight_hash": preflight["projection_hash"],
                "platform_failure_stage": "post_build_validation",
            }
            next_state["blockers"].extend(
                {
                    "stage": "build",
                    "code": blocker["code"],
                    "message": blocker["message"],
                }
                for blocker in build_validation_blockers
            )
            return next_state
        payload["verification_only_lineage"] = verification_only_lineage_payload(
            post_build_preflight
        )
    if build_continuation is not None and (
        payload.get("feature_id") != feature_id
        or payload.get("input_execution_mode")
        != VERIFICATION_ONLY_INPUT_EXECUTION_MODE
    ):
        next_state = _copy_state(state)
        next_state["current_stage"] = "build"
        next_state["build_summary"] = {
            "status": "blocked",
            "input_execution_mode": VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
            "platform_failure_stage": "post_build_validation",
        }
        next_state["blockers"].append(
            {
                "stage": "build",
                "code": "VERIFICATION_BUILD_RESULT_IDENTITY_MISMATCH",
                "message": (
                    "BuildResult does not preserve the verification-only "
                    "Feature and input execution identity."
                ),
            }
        )
        return next_state
    build_write_result = None
    if build_continuation is not None:
        build_write_result = artifact_store.write_with_result(
            feature_id=feature_id,
            artifact_type="build_result",
            payload=payload,
            status="failed" if payload["status"] == "failed" else "success",
            upstream_refs=build_upstream_refs,
        )
        artifact_ref = build_write_result.artifact_ref
    else:
        artifact_ref = artifact_store.write(
            feature_id=feature_id,
            artifact_type="build_result",
            payload=payload,
            status="failed" if payload["status"] == "failed" else "success",
            upstream_refs=build_upstream_refs,
        )
    next_state = _copy_state(state)
    next_state["current_stage"] = "build"
    next_state["build_summary"] = {
        "status": payload["status"],
        "build_status": payload["build_status"],
        "execution_mode": payload["execution_mode"],
        "real_build_performed": payload["real_build_performed"],
        "error_class": payload["error_class"],
        "repair_handoff": payload["repair_handoff"],
        "claims_real_build_success": payload["claims_real_build_success"],
        "input_execution_mode": payload.get("input_execution_mode"),
    }
    if build_continuation is not None:
        next_state["build_summary"]["domain_status"] = payload["status"]
    if build_write_result is not None and (
        build_write_result.outcome != "committed" or artifact_ref is None
    ):
        next_state["build_summary"].update(
            {
                "status": "blocked",
                "platform_failure_stage": "build_result_store",
                "build_result_store_result": build_write_result.to_dict(),
                "workflow_checkpoint_store_result": None,
            }
        )
        next_state["blockers"].append(
            {
                "stage": "build",
                "code": "VERIFICATION_BUILD_RESULT_STORE_NOT_COMMITTED",
                "message": (
                    "Verification BuildResult did not reach a committed Store outcome."
                ),
            }
        )
        return next_state
    if artifact_ref is None:
        raise ArtifactStoreError("Committed BuildResult is missing artifact_ref.")
    _record_artifact(next_state, "build_result", artifact_ref)
    checkpoint_write_result = None
    if (
        persist_build_runtime_handoff
        and payload["status"] == "success"
        and payload["claims_real_build_success"] is True
    ):
        next_state["next_stage"] = "runtime_evidence"
        checkpoint_store = WorkflowCheckpointStore(artifact_store)
        if build_continuation is not None:
            try:
                checkpoint_write_result = checkpoint_store.create_checkpoint_with_result(
                    next_state,
                    interrupt_reason=(
                        "Successful Build is ready for fresh-process Runtime continuation."
                    ),
                    resume_allowed_fields=(),
                    exact_revision_upstream_types=exact_revision_upstream_types,
                )
            except ArtifactStoreError as exc:
                next_state["build_summary"].update(
                    {
                        "status": "blocked",
                        "platform_failure_stage": "checkpoint_prewrite",
                        "build_result_store_result": build_write_result.to_dict(),
                        "workflow_checkpoint_store_result": None,
                    }
                )
                next_state["blockers"].append(
                    {
                        "stage": "build",
                        "code": "VERIFICATION_CHECKPOINT_PREWRITE_BLOCKED",
                        "message": str(exc),
                    }
                )
                return next_state
            checkpoint_ref = checkpoint_write_result.artifact_ref
            if (
                checkpoint_write_result.outcome != "committed"
                or checkpoint_ref is None
            ):
                next_state["build_summary"].update(
                    {
                        "status": "blocked",
                        "platform_failure_stage": "checkpoint_store",
                        "build_result_store_result": build_write_result.to_dict(),
                        "workflow_checkpoint_store_result": (
                            checkpoint_write_result.to_dict()
                        ),
                    }
                )
                next_state["blockers"].append(
                    {
                        "stage": "build",
                        "code": "VERIFICATION_CHECKPOINT_STORE_NOT_COMMITTED",
                        "message": (
                            "Verification workflow checkpoint did not reach a "
                            "committed Store outcome."
                        ),
                    }
                )
                return next_state
        else:
            checkpoint_ref = checkpoint_store.create_checkpoint(
                next_state,
                interrupt_reason=(
                    "Successful Build is ready for fresh-process Runtime continuation."
                ),
                resume_allowed_fields=(),
                exact_revision_upstream_types=exact_revision_upstream_types,
            )
        _record_artifact(next_state, "workflow_checkpoint", checkpoint_ref)
    if build_continuation is not None:
        next_state["build_summary"].update(
            {
                "build_result_store_result": build_write_result.to_dict(),
                "workflow_checkpoint_store_result": (
                    None
                    if checkpoint_write_result is None
                    else checkpoint_write_result.to_dict()
                ),
                "handoff_eligible": (
                    payload["status"] == "success"
                    and payload["claims_real_build_success"] is True
                ),
            }
        )
    return next_state


def _run_verification_only_admission(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    operation: VerificationOperationEnvelope,
) -> ModAgentState:
    """Hydrate one exact no-mutation lineage after a no-write preflight."""

    envelope = validate_verification_operation_envelope(operation)
    declaration = resolve_verification_operation(envelope)
    variant = envelope.variant_payload
    protocol = envelope.protocol_binding
    if (
        declaration.graph_handler != "verification_build"
        or not isinstance(variant, VerificationBuildOperationV1)
        or not isinstance(protocol, BuildInvocationProtocolBindingV1)
    ):
        raise ValueError("Verification Build operation shape changed.")
    continuation = _verification_build_continuation(envelope)
    preflight = preflight_verification_only_continuation(
        artifact_store,
        continuation,
        run_id=state["run_id"],
        request_id=state["request_id"],
        user_request=state["user_request"],
        target_mod=state["target_mod"],
    )
    next_state = _copy_state(state)
    next_state["current_stage"] = "evidence_retrieval"
    next_state["evidence_summary"] = {
        "readiness": "partial",
        "can_generate_patch": False,
        "verification_only_preflight": preflight,
    }
    operation_mismatches = []
    if state.get("run_id") != protocol.run_id:
        operation_mismatches.append("run_id")
    if state.get("request_id") != protocol.request_id:
        operation_mismatches.append("request_id")
    if state.get("user_request") != variant.user_request:
        operation_mismatches.append("user_request")
    if state.get("target_domain") != variant.target_domain:
        operation_mismatches.append("target_domain")
    if state.get("target_mod") != variant.target_mod:
        operation_mismatches.append("target_mod")
    if operation_mismatches:
        next_state["build_summary"] = {
            "status": "blocked",
            "input_execution_mode": VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
            "preflight_hash": preflight["projection_hash"],
            "platform_failure_stage": "admission",
        }
        next_state["blockers"].append(
            {
                "stage": "evidence_retrieval",
                "code": "VERIFICATION_OPERATION_STATE_MISMATCH",
                "message": (
                    "Verification Build Graph state changed approved fields: "
                    + ", ".join(operation_mismatches)
                ),
            }
        )
        return next_state
    if preflight["status"] != "ready_for_build_authorization":
        next_state["build_summary"] = {
            "status": "blocked",
            "input_execution_mode": VERIFICATION_ONLY_INPUT_EXECUTION_MODE,
            "preflight_hash": preflight["projection_hash"],
            "platform_failure_stage": "admission",
        }
        next_state["blockers"].extend(
            {
                "stage": "evidence_retrieval",
                "code": blocker["code"],
                "message": blocker["message"],
            }
            for blocker in preflight["blockers"]
        )
        return next_state
    next_state["feature_id"] = continuation.feature_spec_ref.feature_id
    next_state["next_stage"] = "build"
    _record_artifact(next_state, "feature_spec", continuation.feature_spec_ref)
    _record_artifact(next_state, "evidence_pack", continuation.evidence_pack_ref)
    return next_state


def _route_runtime_continuation_entry(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    operation: VerificationOperationEnvelope,
) -> str:
    """Admit only a validated Build checkpoint into the existing Runtime node."""

    envelope = validate_verification_operation_envelope(operation)
    declaration = resolve_verification_operation(envelope)
    if declaration.graph_handler != "runtime_execution":
        raise ValueError("Verification operation is not a Runtime execution.")
    checkpoint_id = _verification_runtime_checkpoint_id(envelope)
    protocol = envelope.protocol_binding
    if not isinstance(
        protocol,
        (NewRuntimeResumeProtocolBindingV1, VerificationProtocolBinding),
    ):
        raise ValueError("Runtime execution protocol binding changed.")
    if state.get("current_stage") != "build":
        raise ValueError("Runtime continuation state must restore the Build boundary.")
    if state.get("next_stage") != "runtime_evidence":
        raise ValueError("Runtime continuation state has an unexpected next_stage.")
    if state.get("blockers"):
        raise ValueError("Runtime continuation state contains unresolved blockers.")
    for artifact_type in (
        "workflow_checkpoint",
        "resume_report",
        "feature_spec",
        "build_result",
    ):
        if artifact_type not in state.get("artifacts", {}):
            raise ValueError(
                f"Runtime continuation state is missing {artifact_type}."
            )
        if artifact_type not in state.get("artifact_hashes", {}):
            raise ValueError(
                f"Runtime continuation state is missing {artifact_type} hash."
            )
    checkpoint_ref = _artifact_ref_from_state(state, "workflow_checkpoint")
    report_ref = _artifact_ref_from_state(state, "resume_report")
    if checkpoint_ref != protocol.checkpoint_ref:
        raise ValueError(
            "Runtime continuation state changed the approved checkpoint ref."
        )
    if (
        isinstance(protocol, VerificationProtocolBinding)
        and report_ref != protocol.accepted_resume_report_ref
    ):
        raise ValueError(
            "Runtime continuation state changed the accepted ResumeReport ref."
        )
    checkpoint = WorkflowCheckpointStore(artifact_store).read_accepted_runtime_checkpoint(
        checkpoint_ref,
        resume_report_ref=report_ref,
        resume_id=protocol.resume_id,
        approved_by=protocol.approved_by,
        runtime_action_scope=dict(envelope.variant_payload.runtime_action_scope),
        allow_prior_runtime_state=isinstance(envelope.attempt_policy, ObservationRetryAttemptV1),
    )["data"]
    report = artifact_store.read(report_ref)["data"]
    if (
        checkpoint.get("checkpoint_id") != checkpoint_id
        or checkpoint.get("checkpoint_stage") != "build"
        or checkpoint.get("run_id") != state.get("run_id")
        or checkpoint.get("request_id") != state.get("request_id")
        or checkpoint.get("feature_id") != state.get("feature_id")
        or report.get("checkpoint_id") != checkpoint_id
        or report.get("run_id") != state.get("run_id")
        or report.get("resume_stage") != "build"
        or report.get("resume_id") != protocol.resume_id
        or report.get("approved_by") != protocol.approved_by
        or report.get("decision") != "approve"
        or report.get("allowed") is not True
    ):
        raise ValueError(
            "Runtime continuation checkpoint and resume report are not bound "
            "to the restored state."
        )
    return "continue"


def _route_runtime_terminal_continuation_entry(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    continuation: VerificationOnlyRuntimeTerminalInput,
) -> str:
    """Admit one exact committed RuntimeEvidence into terminal routing."""

    if state.get("current_stage") != "runtime_evidence":
        raise ValueError(
            "Runtime terminal continuation must restore the RuntimeEvidence boundary."
        )
    if state.get("blockers"):
        raise ValueError(
            "Runtime terminal continuation state contains unresolved blockers."
        )
    exact_refs = {
        "workflow_checkpoint": continuation.checkpoint_ref,
        "resume_report": continuation.resume_report_ref,
        "runtime_evidence": continuation.runtime_evidence_ref,
    }
    for artifact_type, expected_ref in exact_refs.items():
        actual_ref = _artifact_ref_from_state(state, artifact_type)
        if actual_ref != expected_ref:
            raise ValueError(
                f"Runtime terminal continuation state changed {artifact_type}."
            )
    decision, runtime_payload, terminal_refs = (
        WorkflowCheckpointStore(
            artifact_store
        ).preflight_runtime_terminal_continuation(
            continuation.checkpoint_ref,
            resume_report_ref=continuation.resume_report_ref,
            runtime_evidence_ref=continuation.runtime_evidence_ref,
            resume_id=continuation.resume_id,
            approved_by=continuation.approved_by,
        )
    )
    if not decision.allowed or terminal_refs:
        raise ValueError(
            "Runtime terminal continuation admission changed before Graph routing."
        )
    expected_summary = _runtime_summary_from_payload(runtime_payload)
    if state.get("runtime_summary") != expected_summary:
        raise ValueError(
            "Runtime terminal continuation summary does not match RuntimeEvidence."
        )
    return route_after_runtime_evidence(state)


def _route_runtime_evidence_consolidation_entry(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    verification_operation: VerificationOperationEnvelope,
) -> str:
    """Admit one exact implementation-evidence delta before its only write."""

    envelope = validate_verification_operation_envelope(verification_operation)
    operation = envelope.variant_payload
    protocol = envelope.protocol_binding
    if state.get("current_stage") != "runtime_evidence":
        raise ValueError(
            "RuntimeEvidence consolidation must restore the RuntimeEvidence boundary."
        )
    if state.get("blockers"):
        raise ValueError(
            "RuntimeEvidence consolidation state contains unresolved blockers."
        )
    exact_refs = {
        "workflow_checkpoint": protocol.checkpoint_ref,
        "resume_report": protocol.accepted_resume_report_ref,
        "runtime_evidence": operation.prior_runtime_evidence_ref,
        "evidence_pack": operation.evidence_pack_ref,
        "feature_spec": operation.feature_spec_ref,
        "build_result": operation.build_result_ref,
    }
    for artifact_type, expected_ref in exact_refs.items():
        if _artifact_ref_from_state(state, artifact_type) != expected_ref:
            raise ValueError(
                f"RuntimeEvidence consolidation state changed {artifact_type}."
            )
    decision, _prior_payload, child_refs = WorkflowCheckpointStore(
        artifact_store
    ).preflight_runtime_evidence_consolidation(
        protocol.checkpoint_ref,
        resume_report_ref=protocol.accepted_resume_report_ref,
        prior_runtime_evidence_ref=operation.prior_runtime_evidence_ref,
        resume_id=protocol.resume_id,
        approved_by=protocol.approved_by,
    )
    if not decision.allowed or child_refs:
        raise ValueError(
            "RuntimeEvidence consolidation admission changed before Graph routing."
        )
    project_consolidated_runtime_evidence(envelope, artifact_store)
    return "continue"


def _route_after_terminal_evaluate(state: ModAgentState) -> str:
    """Stop a terminal continuation after any non-committed EvalReport result."""

    if state["runtime_summary"].get("platform_failure_stage") in {
        "eval_report_execution",
        "eval_report_store",
    }:
        return "stop"
    return route_after_evaluate(state)


def _run_create_repair_plan(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    product_execution_mode: ProductExecutionGraphModeV1 | None = None,
    patch_continuation_mode: PatchContinuationInputV1 | None = None,
    patch_continuation_attempt_recorder: PatchContinuationAttemptRecorder | None = None,
    patch_upstream_request: Mapping[str, Any] | None = None,
    patch_upstream_factory: PatchChangeFactory | None = None,
    product_patch_upstream_execution: Any = None,
) -> ModAgentState:
    feature_spec = _read_artifact_data(state, artifact_store, "feature_spec")
    product_upstream = (product_patch_upstream_execution is not None
                        and state["runtime_summary"].get("patch_upstream_pending"))
    proposal_ref = None if product_upstream else _artifact_ref_from_state(state, "patch_proposal")
    proposal_context_pack = _read_artifact_data(
        state,
        artifact_store,
        "patch_proposal",
    ).get("context_pack") if not product_upstream else None
    proposal_context_for_repair = (
        {
            **proposal_context_pack,
            "source_patch_proposal_ref": proposal_ref.path,
            "source_patch_proposal_hash": proposal_ref.artifact_hash,
        }
        if isinstance(proposal_context_pack, dict)
        else None
    )
    planner = RepairPlanner()
    feature_id = state["feature_id"] or state["request_id"]
    failed_stage = state["current_stage"]
    if product_upstream:
        context, upstream = product_patch_upstream_execution.validate_upstream(state, artifact_store)
        plan = planner.from_product_patch_upstream_request(context=context, upstream=upstream,
            feature_id=feature_id, target_behavior=str(feature_spec.get("target_behavior") or ""),
            retry_count=state["retry_count"])
        upstream_refs = [_artifact_ref_from_state(state, kind)
                         for kind in ("feature_spec", "evidence_pack", "implementation_plan")]
    elif failed_stage == "behavior_report" and patch_continuation_mode is not None:
        admitted = preflight_patch_continuation(patch_continuation_mode, artifact_store=artifact_store)
        if admitted.repair_context is None:
            raise PatchContinuationError("PATCH_REPAIR_CONTEXT_MISSING", "repair_plan", "Behavior recovery context was not admitted.")
        plan = planner.from_behavior_failure(
            repair_context=admitted.repair_context, feature_id=feature_id,
            target_behavior=str(feature_spec.get("target_behavior") or ""),
            retry_count=state["retry_count"],
        )
        pending_upstream = state["runtime_summary"].get("patch_upstream_pending")
        if pending_upstream and patch_upstream_factory is not None:
            from src.mod_agent.repair.upstream import validate_active_worker_upstream
            admitted, upstream = validate_active_worker_upstream(patch_continuation_mode,
                _patch_factory_attempt_receipt(patch_upstream_factory), artifact_store=artifact_store)
            plan = planner.from_patch_upstream_request(repair_context=admitted.repair_context,
                upstream=upstream, feature_id=feature_id, target_behavior=str(feature_spec.get("target_behavior") or ""),
                retry_count=admitted.repair_context["parent_retry_count"])
        elif patch_upstream_request is not None:
            from src.mod_agent.repair.upstream import validate_patch_upstream_selection
            admitted, upstream = validate_patch_upstream_selection(patch_continuation_mode, patch_upstream_request,
                artifact_store=artifact_store)
            plan = planner.from_patch_upstream_request(repair_context=admitted.repair_context,
                upstream=upstream, feature_id=feature_id, target_behavior=str(feature_spec.get("target_behavior") or ""),
                retry_count=state["retry_count"])
        # The Review is a passed parent-stage fact, not the source of this failure.
        upstream_refs = [proposal_ref, patch_continuation_mode.repair_parent_review_ref]
    elif failed_stage == "build" and state["build_summary"].get("status") == "failed":
        build_ref = _artifact_ref_from_state(state, "build_result")
        plan = planner.from_build_failure(
            build_result=_read_artifact_data(state, artifact_store, "build_result"),
            build_result_ref=build_ref.path,
            retry_count=state["retry_count"],
            target_behavior=str(feature_spec.get("target_behavior") or ""),
            proposal_context_pack=proposal_context_for_repair,
            implementation_plan=_read_artifact_data(
                state,
                artifact_store,
                "implementation_plan",
            ),
        )
        upstream_refs = [build_ref, proposal_ref]
    elif failed_stage in {"runtime_evidence", "binding_resolution"}:
        runtime_ref = _artifact_ref_from_state(state, "runtime_evidence")
        runtime_evidence = _read_artifact_data(
            state,
            artifact_store,
            "runtime_evidence",
        )
        binding_resolution = (
            _read_artifact_data(
                state,
                artifact_store,
                "binding_resolution",
            )
            if "binding_resolution" in state["artifacts"]
            else None
        )
        resolved_repair_target = (
            dict(binding_resolution["resolved_repair_target"])
            if isinstance(binding_resolution, dict)
            and binding_resolution.get("result_type")
            == "resolved_repair_target"
            and isinstance(
                binding_resolution.get("resolved_repair_target"),
                dict,
            )
            else {}
        )
        plan = planner.from_runtime_failure(
            runtime_evidence=runtime_evidence,
            runtime_evidence_ref=runtime_ref.path,
            retry_count=state["retry_count"],
            target_behavior=str(feature_spec.get("target_behavior") or ""),
            proposal_context_pack=proposal_context_for_repair,
            resolved_repair_target=resolved_repair_target,
        )
        upstream_refs = [runtime_ref, proposal_ref]
        if binding_resolution is not None:
            upstream_refs.append(
                _artifact_ref_from_state(state, "binding_resolution")
            )
    elif failed_stage == "proposal_review":
        review_ref = _artifact_ref_from_state(state, "proposal_review")
        plan = planner.from_review_failure(
            proposal_review=_read_artifact_data(
                state,
                artifact_store,
                "proposal_review",
            ),
            proposal_review_ref=review_ref.path,
            retry_count=state["retry_count"],
            target_behavior=str(feature_spec.get("target_behavior") or ""),
            proposal_context_pack=proposal_context_for_repair,
            implementation_plan=_read_artifact_data(
                state,
                artifact_store,
                "implementation_plan",
            ),
        )
        upstream_refs = [review_ref, proposal_ref]
    else:
        raise ValueError(
            f"Unsupported repair-plan source stage: {failed_stage}"
        )
    artifact_ref = artifact_store.write(
        feature_id=feature_id,
        artifact_type="repair_plan",
        payload=plan.to_artifact_payload(),
        status="blocked" if plan.blockers else "success",
        upstream_refs=upstream_refs,
        exact_revision_upstream_types=(
            {ref.artifact_type for ref in upstream_refs}
            if product_execution_mode is not None or patch_continuation_mode is not None
            else {"patch_proposal"}
        ),
    )
    next_state = _copy_state(state)
    next_state["current_stage"] = "repair_plan"
    next_state["retry_count"] = plan.retry_count
    _record_artifact(next_state, "repair_plan", artifact_ref)
    if (patch_upstream_request is not None or state["runtime_summary"].get("patch_upstream_pending")) and not plan.blockers:
        next_state["runtime_summary"]["patch_upstream_return"] = {
            "required_owner": plan.semantic_decision["required_next_owner"],
            "repair_plan_hash": artifact_ref.artifact_hash,
            "assessment_only": True, "approval_inherited": False,
        }
        next_state["runtime_summary"].pop("patch_upstream_pending", None)
    if patch_continuation_attempt_recorder is not None:
        patch_continuation_attempt_recorder({
            "phase": "repair_plan_committed",
            "repair_semantic_decision": plan.semantic_decision,
            "artifacts": {"repair_plan": {"feature_id": artifact_ref.feature_id,
                "artifact_type": artifact_ref.artifact_type, "path": artifact_ref.path,
                "artifact_hash": artifact_ref.artifact_hash}},
        })
    if plan.blockers:
        next_state["blockers"].extend(plan.blockers)
    elif (product_execution_mode is not None and failed_stage == "build"
          and _read_artifact_data(state, artifact_store, "build_result").get("input_execution_mode") == "isolated_apply"):
        try:
            next_state = _reanchor_applied_product_repair(next_state, artifact_store,
                workspace_root=product_execution_mode.request.isolated_workspace_root)
        except (ArtifactStoreError, ProductExecutionContractError):
            # Store/contract failures retain the product executor's platform
            # classification; they are not missing business-source evidence.
            raise
        except (ValueError, OSError) as exc:
            blocker = {"stage": "repair_plan", "code": "PRODUCT_REPAIR_SOURCE_REANCHORING_REQUIRED",
                "message": str(exc), "required_next_owner": "evidence"}
            # The semantic Repair decision remains history. The conditional
            # edge owns stopping and the existing blocked node persists this
            # physical evidence gap without a second Repair decision/attempt.
            next_state["blockers"].append(blocker)
    return next_state


def _reanchor_applied_product_repair(state, artifact_store, *, workspace_root):
    """Publish physical C1 identities for the original admitted Repair targets.

    This is a deterministic source observation, not another Findings/Planning
    model turn. Existing semantic obligations and binding IDs remain unchanged.
    No revision is written until the full in-memory projection has validated.
    """
    from src.mod_agent.fresh_planning_continuation import (
        _validate_selected_plan,
        _validate_source_items,
    )

    def require(condition, message):
        if not condition:
            raise ValueError("PRODUCT_REPAIR_REANCHORING_REQUIRED: " + message)

    def read(ref):
        require(ref.feature_id == state["feature_id"], "Artifact belongs to another feature.")
        return artifact_store.semantic_read_immutable(ref)

    def upstream(envelope, kind):
        rows = [row for row in envelope["inputs"]["upstream_artifacts"] if row["artifact_type"] == kind]
        require(len(rows) == 1, "Expected one exact " + kind + " upstream.")
        return ArtifactRef(feature_id=state["feature_id"], artifact_type=kind,
            path=rows[0]["path"], artifact_hash=rows[0]["artifact_hash"])

    def selected(kind):
        return _artifact_ref_from_state(state, kind)

    def sha(value):
        return hashlib.sha256(value).hexdigest()

    root = Path(workspace_root).resolve(strict=True)
    require(root.is_dir(), "The product workspace is unavailable.")
    repair_ref = selected("repair_plan")
    repair_envelope = read(repair_ref)
    repair = repair_envelope["data"]
    require(state["current_stage"] == "repair_plan" and repair_envelope["status"] == "success"
            and repair["failure_source"] == "build" and not repair.get("blockers")
            and repair.get("allowed_next_stage") == "generate_patch_proposal",
            "The Repair owner has not admitted an applied Build repair.")
    build_ref = upstream(repair_envelope, "build_result")
    proposal_ref = upstream(repair_envelope, "patch_proposal")
    require(build_ref == selected("build_result") and proposal_ref == selected("patch_proposal"),
            "Repair failure lineage differs from the current state.")
    require(repair["source_proposal"] == {"ref": proposal_ref.path, "hash": proposal_ref.artifact_hash},
            "Repair's declared parent Proposal differs from its upstream.")
    build = read(build_ref)["data"]
    require(build.get("status") == "failed" and build.get("real_build_performed") is True
            and build.get("execution_mode") == "local" and build.get("input_execution_mode") == "isolated_apply"
            and build.get("diagnostics"), "Build lacks an executed failed applied input and diagnostics.")
    apply_ref = upstream(read(build_ref), "isolated_apply_result")
    apply_envelope = read(apply_ref)
    review_ref = upstream(apply_envelope, "proposal_review")
    require(apply_ref == selected("isolated_apply_result") and review_ref == selected("proposal_review")
            and upstream(apply_envelope, "patch_proposal") == proposal_ref,
            "Apply does not bind the selected Proposal and Review.")
    review = read(review_ref)
    require(review["status"] == "success" and upstream(review, "patch_proposal") == proposal_ref
            and review["data"].get("allowed_next_stage") == "dry_run_apply",
            "The original Proposal did not pass its bound Review.")
    applied = apply_envelope["data"]
    from src.mod_agent.execution.task_delegation import has_verified_apply_authorization

    require(apply_envelope["status"] == "success" and applied.get("execution_mode") == "isolated_apply"
            and applied.get("actual_write_performed") is True and applied.get("real_patch_applied") is True
            and has_verified_apply_authorization(applied, require_human_semantic=False)
            and applied.get("approved_by_review") is True
            and applied.get("diff_hash") and applied.get("approved_diff_hash") == applied["diff_hash"],
            "Apply lacks exact approved written bytes.")
    dry_run_ref = upstream(apply_envelope, "dry_run_result")
    dry_run = read(dry_run_ref)
    require(dry_run_ref == selected("dry_run_result") and dry_run["status"] == "success"
            and upstream(dry_run, "patch_proposal") == proposal_ref
            and upstream(dry_run, "proposal_review") == review_ref
            and dry_run["data"].get("diff_hash") == applied["diff_hash"],
            "Approved Apply does not match its exact displayed dry-run diff.")
    proposal = read(proposal_ref)
    require(proposal["status"] == "success", "The failed parent Proposal was not admitted.")
    plan_ref, evidence_ref = upstream(proposal, "implementation_plan"), upstream(proposal, "evidence_pack")
    require(plan_ref == selected("implementation_plan") and evidence_ref == selected("evidence_pack"),
            "Reanchoring must start from the failed Proposal's original Plan and Evidence.")
    plan, evidence = copy.deepcopy(read(plan_ref)["data"]), copy.deepcopy(read(evidence_ref)["data"])
    _validate_selected_plan(plan, evidence, feature_id=state["feature_id"])
    registry = plan["binding_registry"]
    bindings = {row["binding_id"]: row for row in registry["edit_bindings"]}
    require(set(bindings).issubset(set(repair["allowed_edit_binding_ids"])),
            "Current Plan extends beyond Repair's admitted binding IDs.")
    changes, receipts, previews = proposal["data"]["diff_intent"], applied.get("changes"), applied.get("diff_preview")
    require(isinstance(receipts, list) and isinstance(previews, list)
            and len(changes) == len(receipts) == len(previews) and changes,
            "Apply change coverage is incomplete; real before/after receipts are required.")
    per_path = {}
    for change, receipt, preview in zip(changes, receipts, previews, strict=True):
        binding = bindings.get(change["edit_binding_id"])
        require(binding is not None, "An applied change has no original edit binding.")
        path = binding["canonical_target_path"]
        require(all(row.get("edit_binding_id") == binding["binding_id"] and row.get("path") == path
                    and row.get("change_id") == change["change_id"] for row in (receipt, preview)),
                "Apply's path or change identity differs from the original binding.")
        require(change["patch"].get("kind") == "exact_text_replace"
                and preview["before_text"] == change["patch"]["before_text"]
                and preview["after_text"] == change["patch"]["after_text"]
                and preview["before_sha256"] == receipt["before_sha256"]
                and preview["after_sha256"] == receipt["after_sha256"],
                "Apply preview differs from the approved Proposal bytes.")
        per_path.setdefault(path, []).append((change, receipt))

    observations = []
    current_files = {}
    for path, records in per_path.items():
        relative = PurePosixPath(path)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in path,
                "An applied path is not a canonical workspace-relative path.")
        source = (root / path).resolve(strict=True)
        require(source.is_relative_to(root) and source.is_file(), "Applied source escaped the workspace.")
        live = source.read_bytes()
        current_files[path] = live
        require(sha(live) == records[-1][1]["after_sha256"], "Live bytes differ from the final approved Apply output.")
        baseline = live
        for change, receipt in reversed(records):
            before, after = (change["patch"][key].encode("utf-8") for key in ("before_text", "after_text"))
            require(sha(baseline) == receipt["after_sha256"] and after and baseline.count(after) == 1,
                    "Approved output location is absent or ambiguous; exact source evidence is required.")
            baseline = baseline.replace(after, before, 1)
            require(sha(baseline) == receipt["before_sha256"], "Reverse Apply did not recover the recorded base bytes.")
        target_items = [item for item in evidence["items"] if item["source_type"] == "target_source"
                        and item["source_path"] == path]
        require(target_items, "Applied source has no original source evidence.")
        offsets = [0]
        for line in baseline.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        spans = {}
        for item in target_items:
            require(item.get("source_hash") == sha(baseline) and item.get("file_hash") in (None, sha(baseline)),
                    "Original source evidence does not identify the recovered Apply baseline.")
            require(not item.get("verified_member_accesses"),
                    "Changed source carries compiled member observations that require fresh evidence.")
            span = item["span"]
            start, end = span["start_line"], span["end_line"]
            require(type(start) is int and type(end) is int and 1 <= start <= end < len(offsets),
                    "Original evidence does not carry an exact line span.")
            spans[item["evidence_id"]] = [offsets[start - 1], offsets[end]]
        replay = baseline
        for change, receipt in records:
            before, after = (change["patch"][key].encode("utf-8") for key in ("before_text", "after_text"))
            require(before and replay.count(before) == 1, "Approved input location is ambiguous during replay.")
            position, length = replay.index(before), len(before)
            delta = len(after) - length
            for span in spans.values():
                start, end = span
                if position + length <= start:
                    span[:] = [start + delta, end + delta]
                elif position >= end:
                    continue
                elif start <= position and position + length <= end:
                    span[1] += delta
                else:
                    raise ValueError("PRODUCT_REPAIR_REANCHORING_REQUIRED: Approved replacement crosses an evidence boundary.")
            replay = replay.replace(before, after, 1)
            require(sha(replay) == receipt["after_sha256"], "Approved replay did not match its recorded output.")
        require(replay == live, "Approved replay differs from current source.")
        current_offsets = [0]
        for line in live.splitlines(keepends=True):
            current_offsets.append(current_offsets[-1] + len(line))
        for item in target_items:
            start, end = spans[item["evidence_id"]]
            require(start < end and start in current_offsets and end in current_offsets,
                    "The admitted target no longer maps to a complete nonempty line span.")
            original_span = dict(item["span"])
            item.update(span={"start_line": current_offsets.index(start) + 1, "end_line": current_offsets.index(end)},
                source_ref=str(source), source_hash=sha(live), file_hash=sha(live), doc_hash=sha(live),
                excerpt=live[start:end].decode("utf-8"), retrieval_method="approved_apply_byte_mapping")
            item["risk_notes"] = [*item["risk_notes"], "Physical identity refreshed from approved Apply; semantic Findings were not re-executed."]
            observations.append({"evidence_id": item["evidence_id"], "source_path": path,
                "before_span": original_span, "after_span": dict(item["span"]),
                "before_sha256": sha(baseline), "after_sha256": sha(live)})
    items = {item["evidence_id"]: item for item in evidence["items"]}
    for binding in registry["edit_bindings"]:
        item = items[binding["evidence_id"]]
        binding.update(source_hash=item["source_hash"], target_span=dict(item["span"]))
    for binding in registry.get("supporting_bindings", []):
        item = items[binding["evidence_id"]]
        if item["source_type"] == "target_source" and item["source_path"] in per_path:
            binding["target_span"] = dict(item["span"])
    refs = [repair_ref, plan_ref, evidence_ref, proposal_ref, review_ref, dry_run_ref, apply_ref, build_ref]
    trace_refs = [f"{ref.path}#sha256={ref.artifact_hash}" for ref in refs]
    provenance = {"repair_plan_ref": {"path": repair_ref.path, "artifact_hash": repair_ref.artifact_hash},
        "base_relationship": "revision_against_verified_applied_parent", "findings_reexecuted": False,
        "planning_model_reexecuted": False, "approval_inherited": False, "observations": observations}
    evidence["provenance_validation"] = {**evidence.get("provenance_validation", {}), "repair_source_reanchoring": provenance}
    evidence["trace_refs"] = list(dict.fromkeys([*evidence["trace_refs"], *trace_refs]))
    evidence["retrieval_methods_used"] = list(dict.fromkeys([*evidence["retrieval_methods_used"], "approved_apply_byte_mapping"]))
    EvidencePack.from_artifact_payload(evidence)
    _validate_selected_plan(plan, evidence, feature_id=state["feature_id"])
    _validate_source_items(evidence, root)
    require(all((root / path).read_bytes() == content for path, content in current_files.items()),
            "Source changed during repair identity observation.")
    feature_ref = selected("feature_spec")
    writer_scope = getattr(artifact_store, "repair_source_revision_scope", None)
    with trace_operation("repair.reanchor_applied_source", attributes={"repair_plan_hash": repair_ref.artifact_hash,
            "findings_reexecuted": False, "refreshed_source_items": len(observations)}), (
            writer_scope(repair_ref) if writer_scope is not None else nullcontext()):
        fresh_evidence = artifact_store.write(feature_id=state["feature_id"], artifact_type="evidence_pack",
            payload=evidence, status="success", upstream_refs=[feature_ref],
            exact_revision_upstream_types={"feature_spec"}, trace_refs=trace_refs)
        fresh_plan = artifact_store.write(feature_id=state["feature_id"], artifact_type="implementation_plan",
            payload=plan, status="success", upstream_refs=[feature_ref, fresh_evidence],
            exact_revision_upstream_types={"feature_spec", "evidence_pack"}, trace_refs=trace_refs)
    next_state = _copy_state(state)
    _record_artifact(next_state, "evidence_pack", fresh_evidence)
    _record_artifact(next_state, "implementation_plan", fresh_plan)
    return next_state


def _run_resolve_runtime_binding(
    state: ModAgentState,
    artifact_store: ArtifactStore,
) -> ModAgentState:
    feature_id = state["feature_id"] or state["request_id"]
    runtime_ref = _artifact_ref_from_state(state, "runtime_evidence")
    runtime_evidence = _read_artifact_data(
        state,
        artifact_store,
        "runtime_evidence",
    )
    implementation_plan_ref = _artifact_ref_from_state(
        state,
        "implementation_plan",
    )
    implementation_plan = _read_artifact_data(
        state,
        artifact_store,
        "implementation_plan",
    )
    proposal_ref = _artifact_ref_from_state(state, "patch_proposal")
    proposal = _read_artifact_data(
        state,
        artifact_store,
        "patch_proposal",
    )
    context_pack = proposal.get("context_pack")
    allowed_binding_ids = {
        str(value)
        for value in (
            context_pack.get("allowed_edit_binding_ids", ())
            if isinstance(context_pack, dict)
            else ()
        )
        if isinstance(value, str)
    }
    result = resolve_runtime_diagnosis_binding(
        diagnosis=runtime_evidence.get("runtime_failure_diagnosis"),
        feature_id=feature_id,
        run_id=state["run_id"],
        runtime_evidence_revision={
            "feature_id": runtime_ref.feature_id,
            "path": runtime_ref.path,
            "artifact_hash": runtime_ref.artifact_hash,
        },
        implementation_plan_revision={
            "feature_id": implementation_plan_ref.feature_id,
            "path": implementation_plan_ref.path,
            "artifact_hash": implementation_plan_ref.artifact_hash,
        },
        patch_proposal_revision={
            "feature_id": proposal_ref.feature_id,
            "path": proposal_ref.path,
            "artifact_hash": proposal_ref.artifact_hash,
        },
        implementation_plan=implementation_plan,
        allowed_edit_binding_ids=allowed_binding_ids,
    )
    artifact_ref = artifact_store.write(
        feature_id=feature_id,
        artifact_type="binding_resolution",
        payload=result.to_artifact_payload(),
        status="blocked" if result.blocked else "success",
        upstream_refs=[
            runtime_ref,
            implementation_plan_ref,
            proposal_ref,
        ],
    )
    next_state = _copy_state(state)
    next_state["current_stage"] = "binding_resolution"
    _record_artifact(next_state, "binding_resolution", artifact_ref)
    next_state["blockers"].extend(result.blockers)
    return next_state


def _run_runtime_evidence(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    runtime_evidence_factory: RuntimeEvidenceFactory | None,
    *,
    use_store_result: bool = False,
    verification_operation: VerificationOperationEnvelope | None = None,
) -> ModAgentState:
    build = _read_artifact_data(state, artifact_store, "build_result")
    feature_spec = _read_artifact_data(state, artifact_store, "feature_spec")
    feature_id = state["feature_id"] or state["request_id"]
    build_artifact_ref = _artifact_ref_from_state(state, "build_result")
    operation_declaration = (
        None
        if verification_operation is None
        else resolve_verification_operation(
            validate_verification_operation_envelope(verification_operation)
        )
    )
    is_consolidation = (
        operation_declaration is not None
        and operation_declaration.graph_handler
        == "runtime_evidence_consolidation"
    )
    requires_verification_only_input = (
        operation_declaration is not None
        and operation_declaration.completion_scope
        != RUNTIME_WORKFLOW_COMPLETION
    )
    try:
        payload = (
            project_consolidated_runtime_evidence(
                validate_verification_operation_envelope(
                    verification_operation
                ),
                artifact_store,
            )
            if is_consolidation
            else (
                runtime_evidence_factory(feature_id, build, feature_spec)
                if runtime_evidence_factory is not None
                else build_fake_runtime_evidence_payload(
                    feature_id=feature_id,
                    input_execution_mode=build.get("execution_mode"),
                    acceptance_criteria=tuple(
                        feature_spec.get("acceptance_criteria") or ()
                    ),
                    runtime_verification_plan=tuple(
                        feature_spec.get("runtime_verification_plan") or ()
                    ),
                )
            )
        )
    except Exception as exc:
        if not use_store_result:
            raise
        if isinstance(exc, _RuntimeEvidenceExecutionBoundaryError):
            failure_stage = exc.failure_stage
            failure_class = exc.failure_class
        else:
            failure_stage = (
                "graph_consolidation_admission"
                if is_consolidation
                else "runtime_execution"
            )
            failure_class = type(exc).__name__
        next_state = _copy_state(state)
        next_state["current_stage"] = "runtime_evidence"
        next_state["runtime_summary"] = {
            "status": "blocked",
            "platform_failure_stage": failure_stage,
            "runtime_evidence_store_result": None,
            "runtime_evidence_ready": False,
            "failure_class": failure_class,
            "failure_message": str(exc),
        }
        next_state["blockers"].append(
            {
                "stage": "runtime_evidence",
                "code": (
                    "RUNTIME_EVIDENCE_STORE_PREFLIGHT_FAILED"
                    if failure_stage == "store_preflight"
                    else (
                        "RUNTIME_EVIDENCE_POST_VALIDATION_FAILED"
                        if failure_stage == "post_runtime_validation"
                        else "RUNTIME_EVIDENCE_FACTORY_FAILED"
                    )
                ),
                "message": f"{failure_class}: {exc}",
            }
        )
        return next_state
    try:
        payload = bind_runtime_evidence_context(
            payload,
            run_id=state["run_id"],
            build_artifact_ref=build_artifact_ref,
        )
        if use_store_result:
            _validate_runtime_evidence_result_identity(
                state,
                artifact_store,
                payload,
                build=build,
                build_artifact_ref=build_artifact_ref,
                requires_verification_only_input=(
                    requires_verification_only_input
                ),
            )
        acceptance_verified = bool(payload["acceptance_verified"])
    except Exception as exc:
        if not use_store_result:
            raise
        validation_stage = (
            "graph_consolidation_admission"
            if is_consolidation
            else "post_runtime_validation"
        )
        next_state = _copy_state(state)
        next_state["current_stage"] = "runtime_evidence"
        next_state["runtime_summary"] = {
            "status": "blocked",
            "platform_failure_stage": validation_stage,
            "runtime_evidence_store_result": None,
            "runtime_evidence_ready": False,
            "failure_class": type(exc).__name__,
            "failure_message": str(exc),
        }
        next_state["blockers"].append(
            {
                "stage": "runtime_evidence",
                "code": (
                    "RUNTIME_EVIDENCE_CONSOLIDATION_VALIDATION_FAILED"
                    if is_consolidation
                    else "RUNTIME_EVIDENCE_POST_VALIDATION_FAILED"
                ),
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return next_state
    upstream_refs = [
        build_artifact_ref,
        _artifact_ref_from_state(state, "feature_spec"),
    ]
    exact_revision_upstream_types: set[str] | None = None
    semantic_intent = None
    if is_consolidation:
        if verification_operation is None:
            raise AssertionError("Consolidation operation is missing.")
        operation = verification_operation.variant_payload
        if not isinstance(operation, EvidenceConsolidationOperationV1):
            raise AssertionError("Consolidation variant type changed.")
        upstream_refs.extend(
            [
                operation.prior_runtime_evidence_ref,
                operation.evidence_pack_ref,
            ]
        )
        exact_revision_upstream_types = {
            "runtime_evidence",
            "evidence_pack",
        }
        semantic_intent = project_artifact_write_intent(
            reader=artifact_store,
            feature_id=feature_id,
            artifact_type="runtime_evidence",
            payload=payload,
            status="blocked",
            upstream_refs=upstream_refs,
            exact_revision_upstream_types=exact_revision_upstream_types,
            blockers=(),
            trace_refs=(),
            protocol_refs=(
                verification_operation.protocol_binding.checkpoint_ref,
                verification_operation.protocol_binding.accepted_resume_report_ref,
            ),
            execution_file_bindings=(
                operation.execution_file_bindings
            ),
        )
    write_arguments = {
        "feature_id": feature_id,
        "artifact_type": "runtime_evidence",
        "payload": payload,
        "status": (
            "blocked"
            if is_consolidation
            else ("success" if acceptance_verified else "blocked")
        ),
        "upstream_refs": upstream_refs,
        "exact_revision_upstream_types": exact_revision_upstream_types,
        "semantic_intent": semantic_intent,
    }
    write_result = None
    if use_store_result:
        write_result = artifact_store.write_with_result(**write_arguments)
        artifact_ref = write_result.artifact_ref
    else:
        artifact_ref = artifact_store.write(**write_arguments)
    next_state = _copy_state(state)
    if is_consolidation:
        next_state["artifacts"].pop("runtime_evidence", None)
        next_state["artifact_hashes"].pop("runtime_evidence", None)
    next_state["current_stage"] = "runtime_evidence"
    next_state["runtime_summary"] = _runtime_summary_from_payload(payload)
    if write_result is not None:
        next_state["runtime_summary"].update(
            {
                "runtime_evidence_store_result": write_result.to_dict(),
                "runtime_evidence_ready": (
                    write_result.outcome == "committed" and artifact_ref is not None
                ),
                "platform_failure_stage": (
                    None
                    if write_result.outcome == "committed"
                    else "runtime_evidence_store"
                ),
            }
        )
    if artifact_ref is not None:
        _record_artifact(next_state, "runtime_evidence", artifact_ref)
    elif use_store_result:
        next_state["blockers"].append(
            {
                "stage": "runtime_evidence",
                "code": "RUNTIME_EVIDENCE_STORE_NOT_COMMITTED",
                "message": (
                    "RuntimeEvidence did not reach a committed Store outcome."
                ),
            }
        )
    return next_state


def _runtime_summary_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project durable RuntimeEvidence facts without ephemeral Store receipts."""

    return {
        "status": payload["verification_status"],
        "execution_mode": payload["execution_mode"],
        "real_runtime_performed": payload["real_runtime_performed"],
        "startup_loaded": payload["startup_loaded"],
        "gameplay_behavior_verified": payload["gameplay_behavior_verified"],
        "acceptance_status": payload["acceptance_status"],
        "acceptance_verified": bool(payload["acceptance_verified"]),
        "acceptance_contract": payload["acceptance_contract"],
        "claims_real_runtime_success": payload["claims_real_runtime_success"],
        "run_id": payload["run_id"],
        "build_artifact_hash": payload["build_artifact_hash"],
        "plugin_marker_binding": payload["plugin_marker_binding"],
        "screenshot_refs": payload["screenshot_refs"],
        "runtime_trust_binding": payload["runtime_trust_binding"],
        "failure_ownership": payload["failure_ownership"],
        "has_structured_diagnosis": isinstance(
            payload.get("runtime_failure_diagnosis"),
            dict,
        ),
        "plugin_deployment_binding": payload.get("plugin_deployment_binding"),
        "non_goal_preservation": payload["non_goal_preservation"],
    }


def _validate_runtime_evidence_result_identity(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    payload: Mapping[str, Any],
    *,
    build: Mapping[str, Any],
    build_artifact_ref: ArtifactRef,
    requires_verification_only_input: bool,
) -> None:
    """Validate exact post-effect RuntimeEvidence identity before Store write."""

    feature_id = state["feature_id"] or state["request_id"]
    expected_build_ref = {
        "feature_id": build_artifact_ref.feature_id,
        "artifact_type": build_artifact_ref.artifact_type,
        "path": build_artifact_ref.path,
        "artifact_hash": build_artifact_ref.artifact_hash,
    }
    expected_input_mode = build.get("input_execution_mode")
    mismatches = []
    if (
        requires_verification_only_input
        and expected_input_mode != VERIFICATION_ONLY_INPUT_EXECUTION_MODE
    ):
        mismatches.append("verification_only_input_execution_mode")
    if payload.get("feature_id") != feature_id:
        mismatches.append("feature_id")
    if payload.get("run_id") != state["run_id"]:
        mismatches.append("run_id")
    if payload.get("build_artifact_ref") != expected_build_ref:
        mismatches.append("build_artifact_ref")
    if payload.get("build_artifact_hash") != build_artifact_ref.artifact_hash:
        mismatches.append("build_artifact_hash")
    if (
        requires_verification_only_input
        and expected_input_mode is not None
        and payload.get("input_execution_mode") != expected_input_mode
    ):
        mismatches.append("input_execution_mode")
    artifact_store.registry.validate_write(
        artifact_type="runtime_evidence",
        status=("success" if payload.get("acceptance_verified") is True else "blocked"),
        payload=dict(payload),
    )
    for artifact_type in ("feature_spec", "build_result"):
        artifact_ref = _artifact_ref_from_state(state, artifact_type)
        freshness = artifact_store.check_freshness(artifact_ref)
        if not freshness.is_fresh:
            mismatches.append(f"{artifact_type}_freshness")
    if mismatches:
        raise ValueError(
            "RuntimeEvidence post-effect identity validation failed: "
            + ", ".join(sorted(mismatches))
        )


def _run_fake_evaluate(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    evaluation_context_factory: EvaluationContextFactory | None = None,
    require_external_evaluation: bool = False,
    *,
    use_store_result: bool = False,
) -> ModAgentState:
    build = _read_artifact_data(state, artifact_store, "build_result")
    runtime = _read_artifact_data(state, artifact_store, "runtime_evidence")
    verification_only = (
        build.get("input_execution_mode")
        == VERIFICATION_ONLY_INPUT_EXECUTION_MODE
    )
    verification_preflight: dict[str, Any] | None = None
    mutation_artifacts = {
        "implementation_plan",
        "patch_proposal",
        "proposal_review",
        "dry_run_result",
        "isolated_apply_result",
    }.intersection(state["artifacts"])
    if verification_only:
        review = {}
        dry_run = {}
        if {"feature_spec", "evidence_pack"}.issubset(state["artifacts"]):
            verification_preflight = preflight_verification_only_continuation(
                artifact_store,
                VerificationOnlyContinuationInput(
                    feature_spec_ref=_artifact_ref_from_state(state, "feature_spec"),
                    evidence_pack_ref=_artifact_ref_from_state(state, "evidence_pack"),
                ),
                run_id=state["run_id"],
                request_id=state["request_id"],
                user_request=state["user_request"],
                target_mod=state["target_mod"],
            )
    else:
        review = _read_artifact_data(state, artifact_store, "proposal_review")
        dry_run = _read_artifact_data(state, artifact_store, "dry_run_result")
    feature_id = state["feature_id"] or state["request_id"]
    acceptance_verified = bool(runtime.get("acceptance_verified"))
    v1_boundary = _v1_acceptance_boundary(build=build, runtime=runtime)
    v1_accepted = v1_boundary["v1_acceptance_status"] == "accepted"
    evaluation_gate = build_transition_eval_gate_report(
        review=review,
        dry_run=dry_run,
        build=build,
        runtime=runtime,
        v1_boundary=v1_boundary,
        claims_real_success=v1_accepted,
        evaluation_context=(
            evaluation_context_factory(state)
            if evaluation_context_factory is not None
            else None
        ),
        evaluation_required=require_external_evaluation,
        continuation_mode=(
            VERIFICATION_ONLY_INPUT_EXECUTION_MODE
            if verification_only
            else "mutation"
        ),
        verification_only_lineage_verified=(
            verification_preflight is not None
            and verification_preflight["status"]
            == "ready_for_build_authorization"
            and not mutation_artifacts
            and verification_only_lineage_matches(build, verification_preflight)
        ),
    )
    gate_failed = evaluation_gate.get("status") == "failed"
    accepted = v1_accepted and not gate_failed
    execution_mode = "real" if runtime.get("execution_mode") == "real" else "fake"
    workflow_completion_status = (
        "real_runtime_acceptance_completed"
        if accepted
        else ("evaluation_blocked" if gate_failed else "fake_execution_completed")
    )
    payload = {
        "node_name": "evaluate",
        "stage": "evaluation",
        "message": (
            "Evaluation completed with real build and runtime acceptance evidence."
            if accepted
            else (
                "Evaluation blocked by deterministic transition gates."
                if gate_failed
                else (
                    "Evaluation completed for a fake/local workflow path. Result "
                    "remains inconclusive for real build and runtime success."
                )
            )
        ),
        "execution_mode": execution_mode,
        "workflow_completion_status": workflow_completion_status,
        "requested_behavior_status": (
            "verified" if acceptance_verified and not gate_failed else "not_verified"
        ),
        "verdict": "accepted" if accepted else ("blocked" if gate_failed else "inconclusive"),
        "claims_real_success": accepted,
        "checks": [
            {
                "check": (
                    "verification_only_lineage_admitted"
                    if verification_only
                    else "review_gate_approved"
                ),
                "status": (
                    "passed"
                    if not verification_only
                    or (
                        verification_preflight is not None
                        and verification_preflight["status"]
                        == "ready_for_build_authorization"
                        and not mutation_artifacts
                        and verification_only_lineage_matches(
                            build,
                            verification_preflight,
                        )
                    )
                    else "failed"
                ),
            },
            {
                "check": "real_build",
                "status": "passed"
                if build.get("claims_real_build_success")
                else "not_run",
            },
            {
                "check": "real_runtime",
                "status": "passed"
                if runtime.get("claims_real_runtime_success")
                else "not_run",
            },
        ],
        "input_runtime_status": runtime.get("verification_status"),
        "input_acceptance_status": runtime.get("acceptance_status"),
        "input_acceptance_verified": acceptance_verified,
        "acceptance_contract": runtime.get("acceptance_contract"),
        "evaluation_gate": evaluation_gate,
        **v1_boundary,
    }
    gate_blockers = list(evaluation_gate.get("blockers") or [])
    write_arguments = {
        "feature_id": feature_id,
        "artifact_type": "eval_report",
        "payload": payload,
        "status": "blocked" if gate_failed else "success",
        "blockers": gate_blockers,
        "upstream_refs": [
            _artifact_ref_from_state(state, "build_result"),
            _artifact_ref_from_state(state, "runtime_evidence"),
        ],
    }
    write_result = None
    if use_store_result:
        write_result = artifact_store.write_with_result(**write_arguments)
        artifact_ref = write_result.artifact_ref
    else:
        artifact_ref = artifact_store.write(**write_arguments)
    next_state = _copy_state(state)
    next_state["current_stage"] = "evaluation"
    if artifact_ref is not None:
        _record_artifact(next_state, "eval_report", artifact_ref)
    if write_result is not None:
        eval_ready = (
            write_result.outcome == "committed" and artifact_ref is not None
        )
        next_state["runtime_summary"].update(
            {
                "eval_report_store_result": write_result.to_dict(),
                "eval_report_ready": eval_ready,
                "platform_failure_stage": (
                    None if eval_ready else "eval_report_store"
                ),
                "terminal_route": (
                    "blocked" if gate_failed and eval_ready else "evaluate"
                ),
                "terminal_ready": False,
            }
        )
        if not eval_ready:
            next_state["blockers"].append(
                {
                    "stage": "evaluation",
                    "code": "EVAL_REPORT_STORE_NOT_COMMITTED",
                    "message": "EvalReport did not reach a committed Store outcome.",
                }
            )
    if gate_failed:
        next_state["blockers"].extend(
            {
                "stage": "evaluation",
                "code": blocker.get("code", "EVAL_GATE_FAILED"),
                "message": blocker.get("message", "Evaluation gate failed."),
            }
            for blocker in gate_blockers
            if isinstance(blocker, dict)
        )
    return next_state


def _run_isolated_apply(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    isolated_apply_factory: PatchExecutionFactory,
) -> ModAgentState:
    from langgraph.errors import GraphInterrupt

    proposal = _read_artifact_data(state, artifact_store, "patch_proposal")
    review = _read_artifact_data(state, artifact_store, "proposal_review")
    semantic_assurance, semantic_contract_present = (
        _semantic_assurance_for_execution(proposal, review)
    )
    human_semantic_review_required = (
        semantic_contract_present and semantic_assurance.verdict == "unproven"
    )
    approved_by_user = state["approvals"].get("confirmed_write") is True
    next_state = _copy_state(state)
    status = "success"
    if not approved_by_user:
        payload = {
            "execution_mode": "blocked",
            "actual_write_performed": False,
            "real_patch_applied": False,
            "approved_by_review": review.get("allowed_next_stage") == "dry_run_apply",
            "changed_files": [],
        }
        status = "blocked"
        next_state["blockers"].append(
            {
                "stage": "isolated_apply",
                "code": "ISOLATED_WRITE_NOT_APPROVED",
                "message": "Isolated patch application requires explicit user approval.",
            }
        )
    else:
        try:
            payload = isolated_apply_factory(
                proposal,
                _read_artifact_data(state, artifact_store, "implementation_plan"),
                review,
            )
        except ExactDiffApprovalRejectedError as exc:
            payload = {
                "execution_mode": "rejected",
                "actual_write_performed": False,
                "real_patch_applied": False,
                "approved_by_review": review.get("allowed_next_stage")
                == "dry_run_apply",
                "changed_files": [],
                "rejection": str(exc),
                "human_semantic_decision": exc.human_semantic_decision,
                "human_semantic_review_completed": False,
                "human_semantic_review_result": "rejected",
            }
            status = "blocked"
            next_state["blockers"].append(
                {
                    "stage": "isolated_apply",
                    "code": "EXACT_DIFF_REJECTED",
                    "message": str(exc),
                }
            )
        except GraphInterrupt:
            raise
        except Exception as exc:
            from src.mod_agent.execution.patch_apply import patch_apply_failure_details

            payload = {
                "execution_mode": "isolated_apply",
                **patch_apply_failure_details(exc),
                "approved_by_review": review.get("allowed_next_stage") == "dry_run_apply",
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = "failed"
            next_state["blockers"].append(
                {
                    "stage": "isolated_apply",
                    "code": "ISOLATED_APPLY_FAILED",
                    "message": payload["error"],
                }
            )
    exact_diff_approved = bool(
        payload.get("approved_diff_hash")
        and payload.get("approved_diff_hash") == payload.get("diff_hash")
    )
    human_semantic_decision = payload.get("human_semantic_decision")
    human_semantic_review_completed = bool(
        payload.get("human_semantic_review_completed") is True
        and isinstance(human_semantic_decision, dict)
        and human_semantic_decision.get("decision") == "approve"
    )
    delegated = False
    if ("authorization_origin" in payload or "task_delegation_ref" in payload
            or "task_delegation_resume_ref" in payload):
        from src.mod_agent.execution.task_delegation import has_verified_apply_authorization

        try:
            delegated = has_verified_apply_authorization(
                payload, run_id=state["run_id"], request_id=state["request_id"])
        except (KeyError, OSError, TypeError, ValueError) as error:
            status = "failed"
            next_state["blockers"].append({"stage": "isolated_apply",
                "code": "TASK_DELEGATION_AUTHORIZATION_INVALID", "message": str(error)})
    if (
        status == "success"
        and human_semantic_review_required
        and not (human_semantic_review_completed or delegated)
    ):
        status = "failed"
        next_state["blockers"].append(
            {
                "stage": "isolated_apply",
                "code": "HUMAN_SEMANTIC_DECISION_MISSING",
                "message": (
                    "Unproven semantic assurance requires a bound structured "
                    "human approve decision before isolated apply can succeed."
                ),
            }
        )
    complete_payload = {
        "node_name": "isolated_apply",
        "stage": "isolated_apply",
        "message": (
            "Approved patch was applied inside the isolated workspace."
            if status == "success"
            else "Isolated patch application did not complete."
        ),
        **payload,
        "write_scope_preapproved": approved_by_user,
        "approved_by_user": bool(
            payload.get("approved_by_user", approved_by_user if status == "success" else False)
        ),
        "exact_diff_approved": exact_diff_approved,
        "semantic_assurance": semantic_assurance.to_dict(),
        "human_semantic_review_required": human_semantic_review_required and not delegated,
        "human_semantic_review_completed": human_semantic_review_completed,
        "human_semantic_review_result": payload.get(
            "human_semantic_review_result",
            "approved" if human_semantic_review_completed else "missing",
        ),
        "human_semantic_decision": human_semantic_decision,
    }
    try:
        artifact_ref = artifact_store.write(
            feature_id=proposal["feature_id"],
            artifact_type="isolated_apply_result",
            payload=complete_payload,
            status=status,
            upstream_refs=[
                _artifact_ref_from_state(state, "patch_proposal"),
                _artifact_ref_from_state(state, "proposal_review"),
                _artifact_ref_from_state(state, "dry_run_result"),
            ],
        )
    except BaseException as error:
        from src.mod_agent.execution.patch_apply import attach_patch_apply_failure

        # An ArtifactStore failure after an Apply return is not evidence of
        # zero filesystem effects. Preserve facts on the escaping exception.
        attach_patch_apply_failure(error, {
            **{key: complete_payload[key] for key in (
                "actual_write_performed", "real_patch_applied", "changed_files",
                "diff_hash", "approved_diff_hash", "context_hashes", "approved_by_user",
                "authorization_origin", "task_delegation_ref", "task_delegation_resume_ref",
            ) if key in complete_payload},
            "write_completion": complete_payload.get("write_completion",
                "complete" if complete_payload.get("real_patch_applied") is True else
                "not_started" if complete_payload.get("actual_write_performed") is False else "unknown"),
            "write_failure": {"operation": "result_persistence",
                "file_observations": (complete_payload.get("write_failure") or {}).get("file_observations", [])},
        })
        raise
    next_state["current_stage"] = "isolated_apply"
    _record_artifact(next_state, "isolated_apply_result", artifact_ref)
    return next_state


def _semantic_assurance_for_execution(
    proposal: dict[str, Any],
    review: dict[str, Any],
) -> tuple[SemanticAssuranceV1, bool]:
    proposal_has_contract = "semantic_assurance" in proposal
    review_has_contract = "semantic_assurance" in review
    proposal_value = proposal.get("semantic_assurance")
    review_value = review.get("semantic_assurance")
    if proposal_has_contract and not isinstance(proposal_value, dict):
        raise ValueError("Proposal semantic assurance is malformed.")
    if review_has_contract and not isinstance(review_value, dict):
        raise ValueError("Review semantic assurance is malformed.")
    proposal_assurance = (
        SemanticAssuranceV1.from_dict(proposal_value)
        if proposal_has_contract
        else SemanticAssuranceV1.unproven()
    )
    review_assurance = (
        SemanticAssuranceV1.from_dict(review_value)
        if review_has_contract
        else SemanticAssuranceV1.unproven()
    )
    if proposal_assurance != review_assurance:
        raise ValueError(
            "Proposal and review semantic assurance results do not match."
        )
    semantic_contract_present = proposal_has_contract or review_has_contract
    if semantic_contract_present and proposal_assurance.verdict != "unproven":
        raise ValueError(
            "Semantic Assurance V0 accepts only the unproven machine verdict."
        )
    if (
        semantic_contract_present
        and review.get("human_semantic_review_required") is not True
    ):
        raise ValueError(
            "Unproven semantic assurance is missing its human review obligation."
        )
    return proposal_assurance, semantic_contract_present


def _run_workflow_summary(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    *,
    use_store_result: bool = False,
) -> ModAgentState:
    eval_report = _read_artifact_data(state, artifact_store, "eval_report")
    feature_id = state["feature_id"] or state["request_id"]
    evaluation_gate = eval_report.get("evaluation_gate")
    evaluation_gate_passed = (
        isinstance(evaluation_gate, dict)
        and evaluation_gate.get("status") == "passed"
    )
    accepted = (
        eval_report.get("v1_acceptance_status") == "accepted"
        and eval_report.get("verdict") == "accepted"
        and eval_report.get("claims_real_success") is True
        and evaluation_gate_passed
    )
    evaluation_blocked = (
        eval_report.get("verdict") == "blocked" or not evaluation_gate_passed
    )
    verification_only = (
        state["build_summary"].get("input_execution_mode")
        == VERIFICATION_ONLY_INPUT_EXECUTION_MODE
    )
    durable_runtime_summary = (
        _runtime_summary_from_payload(
            _read_artifact_data(state, artifact_store, "runtime_evidence")
        )
        if use_store_result
        else dict(state["runtime_summary"])
    )
    payload = {
        "node_name": "done",
        "stage": "done",
        "message": (
            "Workflow reached done with real build and runtime acceptance evidence."
            if accepted
            else (
                "Workflow stopped because deterministic evaluation gates blocked "
                "acceptance. No real Mod success is claimed."
                if evaluation_blocked
                else
                "Workflow reached the end of the controlled fake execution loop. "
                "No real file write, real build, or real runtime success is claimed."
            )
        ),
        "execution_mode": eval_report.get("execution_mode"),
        "workflow_completion_status": eval_report.get("workflow_completion_status"),
        "requested_behavior_status": eval_report.get("requested_behavior_status"),
        "final_status": (
            "accepted" if accepted else ("blocked" if evaluation_blocked else "inconclusive")
        ),
        "claims_real_mod_success": accepted,
        "claims_real_build_success": bool(
            state["build_summary"].get("claims_real_build_success")
        ),
        "claims_real_runtime_success": bool(
            durable_runtime_summary.get("claims_real_runtime_success")
        ),
        "artifact_refs": dict(state["artifacts"]),
        "build_summary": dict(state["build_summary"]),
        "runtime_summary": durable_runtime_summary,
        "eval_verdict": eval_report.get("verdict"),
        "v1_acceptance_status": eval_report.get("v1_acceptance_status"),
        "v1_acceptance_blockers": list(eval_report.get("v1_acceptance_blockers") or []),
        "next_required_evidence": (
            []
            if accepted
            else (
                ["real build output", "real runtime or manual verification evidence"]
                if verification_only
                else [
                    "real dry-run/apply result",
                    "real build output",
                    "real runtime or manual verification evidence",
                ]
            )
        ),
    }
    write_arguments = {
        "feature_id": feature_id,
        "artifact_type": "workflow_summary",
        "payload": payload,
        "status": "success",
        "upstream_refs": [_artifact_ref_from_state(state, "eval_report")],
    }
    write_result = None
    if use_store_result:
        write_result = artifact_store.write_with_result(**write_arguments)
        artifact_ref = write_result.artifact_ref
    else:
        artifact_ref = artifact_store.write(**write_arguments)
    next_state = _copy_state(state)
    next_state["current_stage"] = "done"
    if artifact_ref is not None:
        _record_artifact(next_state, "workflow_summary", artifact_ref)
    if write_result is not None:
        summary_ready = (
            write_result.outcome == "committed" and artifact_ref is not None
        )
        next_state["runtime_summary"].update(
            {
                "workflow_summary_store_result": write_result.to_dict(),
                "workflow_summary_ready": summary_ready,
                "platform_failure_stage": (
                    None if summary_ready else "workflow_summary_store"
                ),
                "terminal_route": "done",
                "terminal_ready": summary_ready,
                "terminal_final_status": payload["final_status"],
                "terminal_claims_real_mod_success": payload[
                    "claims_real_mod_success"
                ],
            }
        )
        if not summary_ready:
            next_state["blockers"].append(
                {
                    "stage": "done",
                    "code": "WORKFLOW_SUMMARY_STORE_NOT_COMMITTED",
                    "message": (
                        "WorkflowSummary did not reach a committed Store outcome."
                    ),
                }
            )
    return next_state


def _run_planning_brief(
    state: ModAgentState,
    artifact_store: ArtifactStore,
) -> ModAgentState:
    plan_data = _read_artifact_data(state, artifact_store, "request_planning_report")
    plan = _request_plan_from_payload(plan_data)
    planning_brief = ModPlanningBriefBuilder().build(
        plan,
        target_game=state["target_domain"],
    )
    artifact_ref = artifact_store.write(
        feature_id=planning_brief.feature_id,
        artifact_type="mod_planning_brief",
        payload=planning_brief.to_artifact_payload(),
        status="blocked" if planning_brief.blocked else "success",
        upstream_refs=[_artifact_ref_from_state(state, "request_planning_report")],
    )
    next_state = _copy_state(state)
    next_state["current_stage"] = "planning_brief"
    _record_artifact(next_state, "mod_planning_brief", artifact_ref)
    if planning_brief.blocked:
        next_state["blockers"].extend(planning_brief.blockers)
    return next_state


def _v1_acceptance_boundary(
    *,
    build: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    if build.get("claims_real_build_success") is not True:
        blockers.append(
            {
                "code": "REAL_BUILD_EVIDENCE_MISSING",
                "message": "V1 acceptance requires real build success evidence.",
            }
        )
    if runtime.get("execution_mode") != "real":
        blockers.append(
            {
                "code": "REAL_RUNTIME_MODE_REQUIRED",
                "message": "V1 acceptance requires runtime evidence from real execution mode.",
            }
        )
    if runtime.get("real_runtime_performed") is not True:
        blockers.append(
            {
                "code": "REAL_RUNTIME_EVIDENCE_MISSING",
                "message": "V1 acceptance requires real runtime execution evidence.",
            }
        )
    if runtime.get("claims_real_runtime_success") is not True:
        blockers.append(
            {
                "code": "REAL_RUNTIME_SUCCESS_NOT_CLAIMED",
                "message": "V1 acceptance requires an explicit real runtime success claim.",
            }
        )
    if runtime.get("acceptance_verified") is not True:
        blockers.append(
            {
                "code": "ACCEPTANCE_NOT_VERIFIED",
                "message": "V1 acceptance requires verified runtime acceptance evidence.",
            }
        )
    if (
        runtime.get("evidence_source") == "gtfo_real_runtime_automation"
        and _runtime_trust_binding_status(runtime) != "trusted"
    ):
        blockers.append(
            {
                "code": "GTFO_RUNTIME_TRUST_BINDING_MISSING",
                "message": (
                    "V1 acceptance for GTFO automation requires trusted current-run "
                    "log, process, marker, build, and screenshot binding."
                ),
            }
        )
    return {
        "v1_acceptance_status": "blocked" if blockers else "accepted",
        "v1_acceptance_blockers": blockers,
    }


def _runtime_trust_binding_status(runtime: dict[str, Any]) -> str | None:
    binding = runtime.get("runtime_trust_binding")
    if not isinstance(binding, dict):
        return None
    return binding.get("status")


def _request_plan_from_payload(data: dict[str, Any]) -> RequestPlan:
    return RequestPlan(
        feature_id=data["feature_id"],
        user_request=data["user_request"],
        route=data["route"],
        risk_level=data["risk_level"],
        target_mod=data.get("target_mod"),
        requires_hook_evidence=bool(data["requires_hook_evidence"]),
        requires_runtime_evidence=bool(data["requires_runtime_evidence"]),
        hook_requirement=_hook_requirement_from_payload(data),
        request_interpretation=dict(data.get("request_interpretation") or {}),
        uncertainties=tuple(
            Uncertainty(
                uncertainty_id=item["uncertainty_id"],
                severity=item["severity"],
                question=item["question"],
                allowed_resolution=tuple(item.get("allowed_resolution", ())),
                default_action=item.get("default_action", "block"),
            )
            for item in data.get("uncertainties", [])
        ),
        blocked=bool(data["blocked"]),
    )


def _feature_spec_from_payload(data: dict[str, Any]) -> FeatureSpec:
    return FeatureSpec(
        feature_id=data["feature_id"],
        user_request=data["user_request"],
        target_behavior=data["target_behavior"],
        non_goals=tuple(data["non_goals"]),
        target_mod=data.get("target_mod"),
        risk_level=data["risk_level"],
        acceptance_criteria=tuple(data["acceptance_criteria"]),
        evidence_requirements=tuple(data["evidence_requirements"]),
        runtime_verification_plan=tuple(data["runtime_verification_plan"]),
        hook_requirement=_hook_requirement_from_payload(data),
        request_interpretation=dict(data.get("request_interpretation") or {}),
        blocked=bool(data["blocked"]),
        blockers=tuple(data.get("blockers", [])),
    )


def _hook_requirement_from_payload(data: dict[str, Any]) -> dict[str, Any]:
    hook_requirement = data.get("hook_requirement")
    if isinstance(hook_requirement, dict):
        return {
            "required": hook_requirement.get("required") is True,
            "reason": str(hook_requirement.get("reason") or "not_required"),
        }
    return {"required": False, "reason": "not_required"}


def _read_artifact_data(
    state: ModAgentState,
    artifact_store: ArtifactStore,
    artifact_type: str,
) -> dict[str, Any]:
    ref = _artifact_ref_from_state(state, artifact_type)
    envelope = artifact_store.read(ref)
    return envelope["data"]


def _artifact_ref_from_state(state: ModAgentState, artifact_type: str) -> ArtifactRef:
    return ArtifactRef(
        feature_id=state["feature_id"] or state["request_id"],
        artifact_type=artifact_type,
        path=state["artifacts"][artifact_type],
        artifact_hash=state["artifact_hashes"][artifact_type],
    )


def _record_artifact(
    state: ModAgentState,
    artifact_type: str,
    artifact_ref: ArtifactRef,
) -> None:
    state["artifacts"][artifact_type] = artifact_ref.path
    state["artifact_hashes"][artifact_type] = artifact_ref.artifact_hash


def _node_status(state: ModAgentState, node_name: str) -> str:
    if node_name == "generate_patch_proposal" and state["runtime_summary"].get("patch_upstream_pending"):
        return "partial"
    if node_name == "retrieve_evidence":
        if state["evidence_summary"].get("compilation_repair_candidate"):
            return "partial"
        return "success" if state["evidence_summary"].get("can_generate_patch") else "blocked"
    if node_name == "review_proposal":
        return (
            "success"
            if state["review_summary"].get("allowed_next_stage") == "dry_run_apply"
            else "blocked"
        )
    if node_name == "build" and state["build_summary"]:
        return str(state["build_summary"].get("status", "success"))
    if node_name == "collect_runtime_evidence" and state["runtime_summary"]:
        return str(state["runtime_summary"].get("status", "success"))
    if node_name == "create_repair_plan":
        return "blocked" if state["blockers"] else "success"
    if node_name == "resolve_runtime_binding":
        return "blocked" if state["blockers"] else "success"
    if node_name == "done":
        return "inconclusive"
    if state["blockers"]:
        return "blocked"
    return "success"


def _node_artifact_refs(state: ModAgentState, node_name: str) -> list[str]:
    if node_name == "generate_patch_proposal" and state["runtime_summary"].get("patch_upstream_pending"):
        return []
    artifact_by_node = {
        "plan_request": "request_planning_report",
        "planning_brief": "mod_planning_brief",
        "draft_feature_spec": "feature_spec",
        "retrieve_evidence": "evidence_pack",
        "create_implementation_plan": "implementation_plan",
        "generate_patch_proposal": "patch_proposal",
        "review_proposal": "proposal_review",
        "human_review_gate": "resume_report",
        "dry_run_apply": "dry_run_result",
        "build": "build_result",
        "collect_runtime_evidence": "runtime_evidence",
        "correct_retained_runtime_binding": "runtime_evidence",
        "resolve_runtime_binding": "binding_resolution",
        "create_repair_plan": "repair_plan",
        "evaluate": "eval_report",
        "done": "workflow_summary",
    }
    artifact_type = artifact_by_node.get(node_name)
    if artifact_type is None or artifact_type not in state["artifacts"]:
        return []
    return [state["artifacts"][artifact_type]]


def _node_trace_details(state: ModAgentState, node_name: str) -> dict[str, Any]:
    if node_name == "generate_patch_proposal" and state["runtime_summary"].get("patch_upstream_pending"):
        return copy.deepcopy(state["runtime_summary"]["patch_upstream_pending"])
    if node_name == "verification_only_admission":
        preflight = state["evidence_summary"].get("verification_only_preflight")
        if isinstance(preflight, dict):
            return {
                "status": preflight.get("status"),
                "projection_hash": preflight.get("projection_hash"),
                "source_files_verified": (
                    preflight.get("source_lineage", {}).get("all_files_verified")
                ),
            }
    if node_name == "retrieve_evidence" and state["evidence_summary"]:
        missing_evidence = list(state["evidence_summary"].get("missing_evidence") or [])
        return {
            "readiness": state["evidence_summary"].get("readiness"),
            "can_generate_patch": state["evidence_summary"].get("can_generate_patch"),
            "retrieval_methods_used": list(
                state["evidence_summary"].get("retrieval_methods_used") or []
            ),
            "missing_evidence_count": len(missing_evidence),
            "disabled_modes": dict(state["evidence_summary"].get("disabled_modes") or {}),
            "timing": dict(state["evidence_summary"].get("timing") or {}),
            **({"source_reassessment": copy.deepcopy(state["runtime_summary"]["evidence_reassessment"])}
                if isinstance(state["runtime_summary"].get("evidence_reassessment"), dict)
                    and state["runtime_summary"]["evidence_reassessment"].get("repair_plan_hash")
                    == state["artifact_hashes"].get("repair_plan") else {}),
        }
    if node_name == "create_repair_plan" and "repair_plan" in state["artifacts"]:
        return {
            "retry_count": state["retry_count"],
            "blocked": bool(state["blockers"]),
        }
    if (
        node_name == "resolve_runtime_binding"
        and "binding_resolution" in state["artifacts"]
    ):
        return {
            "result_type": (
                "binding_gap" if state["blockers"] else "resolved_repair_target"
            ),
            "blocked": bool(state["blockers"]),
        }
    return {}


def _trace_event(
    *,
    state: ModAgentState,
    node_name: str,
    event_type: str,
    status: str,
    artifact_refs: list[str],
    summary: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "event_id": f"evt_{uuid4().hex}",
        "run_id": state["run_id"],
        "feature_id": state.get("feature_id") or state["request_id"],
        "node_name": node_name,
        "stage": state["current_stage"],
        "event_type": event_type,
        "status": status,
        "artifact_refs": artifact_refs,
        "summary": summary,
    }
    if details:
        event["details"] = details
    return event


def _copy_state_with_trace(
    state: ModAgentState,
    trace_event: dict[str, Any],
) -> ModAgentState:
    next_state = _copy_state(state)
    next_state["trace_events"].append(trace_event)
    return next_state


def _copy_state(state: ModAgentState) -> ModAgentState:
    return _copy_mod_agent_state(state)
