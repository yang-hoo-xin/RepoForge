"""Artifact-backed checkpoint and resume helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

from src.mod_agent.artifacts import (
    ArtifactRef,
    ArtifactStore,
    ArtifactStoreError,
    ArtifactWriteResult,
)
from src.mod_agent.verification_operations import (
    AcceptedResumeRecoveryAttemptV1,
    EvidenceConsolidationOperationV1,
    InitialAttemptV1,
    NewRuntimeResumeProtocolBindingV1,
    ObservationRetryAttemptV1,
    RuntimeTerminalizationOperationV1,
    VerificationAttemptPolicy,
    VerificationOperationDeclaration,
    VerificationOperationEnvelope,
    VerificationProtocolBinding,
    resolve_verification_operation,
    validate_verification_operation_envelope,
)

from .interrupts import ResumeDecision, ResumePayload, ResumeValidator
from .state import ModAgentState
from .verification_only import VERIFICATION_ONLY_INPUT_EXECUTION_MODE

BUILD_CHECKPOINT_EXACT_REVISION_TYPES = frozenset(
    {"evidence_pack", "implementation_plan"}
)

VerificationCheckpointAdmissionResult = (
    None
    | ResumeDecision
    | tuple[ResumeDecision, ArtifactRef]
    | tuple[ResumeDecision, dict[str, Any], tuple[ArtifactRef, ...]]
    | tuple[
        ResumeDecision,
        ArtifactRef,
        ArtifactRef,
        ArtifactRef,
        dict[str, Any],
    ]
)


@dataclass(frozen=True)
class CheckpointPolicy:
    """Local checkpoint policy for human-controlled workflow boundaries."""

    interrupt_stages: tuple[str, ...] = (
        "proposal_review",
        "review_proposal",
        "dry_run_apply",
        "build",
        "collect_runtime_evidence",
        "edit_authority_approval",
        "repair_planning_approval",
    )
    backend: str = "langgraph_memory"

    def should_checkpoint(self, stage: str) -> bool:
        """Return whether the stage is a checkpoint boundary."""

        return stage in self.interrupt_stages

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable policy details."""

        return {
            "interrupt_stages": list(self.interrupt_stages),
            "backend": self.backend,
        }


@dataclass(frozen=True)
class WorkflowCheckpoint:
    """Structured checkpoint payload."""

    checkpoint_id: str
    run_id: str
    feature_id: str
    request_id: str
    stage: str
    status: str
    state_snapshot: dict[str, Any]
    artifact_refs: tuple[str, ...]
    artifact_hashes: dict[str, str]
    resume_binding: dict[str, Any]
    resume_allowed_fields: tuple[str, ...]
    interrupt_reason: str
    policy: CheckpointPolicy

    def to_artifact_payload(self) -> dict[str, Any]:
        """Return payload compatible with the workflow_checkpoint contract."""

        return {
            "node_name": "workflow_checkpoint",
            "stage": "checkpoint",
            "message": "Workflow checkpoint recorded for controlled resume.",
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "feature_id": self.feature_id,
            "request_id": self.request_id,
            "checkpoint_stage": self.stage,
            "status": self.status,
            "state_snapshot": self.state_snapshot,
            "artifact_refs": list(self.artifact_refs),
            "artifact_hashes": self.artifact_hashes,
            "resume_binding": self.resume_binding,
            "resume_allowed_fields": list(self.resume_allowed_fields),
            "interrupt_reason": self.interrupt_reason,
            "checkpoint_policy": self.policy.to_dict(),
        }


@dataclass(frozen=True)
class ResumeReport:
    """Structured resume report payload."""

    resume_id: str
    checkpoint_id: str
    run_id: str
    stage: str
    decision: str
    status: str
    allowed: bool
    reason: str
    applied_updates: dict[str, Any]
    stale_reasons: tuple[str, ...]
    blockers: tuple[dict[str, Any], ...]
    approved_by: str = ""
    approved_at: str = ""
    authority_request_hash: str | None = None

    def to_artifact_payload(self) -> dict[str, Any]:
        """Return payload compatible with the resume_report contract."""

        return {
            "node_name": "workflow_resume",
            "stage": "resume",
            "message": f"Workflow resume {self.status}.",
            "resume_id": self.resume_id,
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "resume_stage": self.stage,
            "decision": self.decision,
            "status": self.status,
            "allowed": self.allowed,
            "reason": self.reason,
            "applied_updates": self.applied_updates,
            "stale_reasons": list(self.stale_reasons),
            "blockers": list(self.blockers),
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "authority_request_hash": self.authority_request_hash,
        }


def resumed_state_from_decision(
    decision: ResumeDecision,
    *,
    checkpoint_ref: ArtifactRef | None = None,
    resume_report_ref: ArtifactRef | None = None,
) -> ModAgentState:
    """Build a graph-resumable state from an accepted resume decision."""

    if not decision.allowed:
        raise ValueError("Cannot build graph state from a blocked resume decision.")
    restored_state = decision.applied_updates.get("restored_state")
    if not isinstance(restored_state, dict):
        raise ValueError("Accepted resume decision does not contain restored_state.")
    state = ModAgentState(**restored_state)
    if decision.applied_updates.get("clear_checkpointed_approvals") is True:
        state["approvals"] = {}
    approvals = decision.applied_updates.get("approvals")
    if isinstance(approvals, dict):
        state["approvals"] = {
            **dict(state["approvals"]),
            **approvals,
        }
    if checkpoint_ref is not None:
        state["artifacts"] = {
            **dict(state["artifacts"]),
            "workflow_checkpoint": checkpoint_ref.path,
        }
        state["artifact_hashes"] = {
            **dict(state["artifact_hashes"]),
            "workflow_checkpoint": checkpoint_ref.artifact_hash,
        }
    if resume_report_ref is not None:
        state["artifacts"] = {
            **dict(state["artifacts"]),
            "resume_report": resume_report_ref.path,
        }
        state["artifact_hashes"] = {
            **dict(state["artifact_hashes"]),
            "resume_report": resume_report_ref.artifact_hash,
        }
    return state


class WorkflowCheckpointStore:
    """Write checkpoints and validate resume attempts through ArtifactStore."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        policy: CheckpointPolicy | None = None,
        resume_validator: ResumeValidator | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.policy = policy or CheckpointPolicy()
        self.resume_validator = resume_validator or ResumeValidator()
        self._consumed_checkpoint_ids: set[str] = set()
        self._seen_resume_ids: set[str] = set()

    def create_checkpoint(
        self,
        state: ModAgentState,
        *,
        interrupt_reason: str,
        resume_allowed_fields: tuple[str, ...],
        exact_revision_upstream_types: frozenset[str] = frozenset(),
    ) -> ArtifactRef:
        """Persist a workflow_checkpoint artifact from current state."""

        write_arguments = self._checkpoint_write_arguments(
            state,
            interrupt_reason=interrupt_reason,
            resume_allowed_fields=resume_allowed_fields,
            exact_revision_upstream_types=exact_revision_upstream_types,
        )
        return self.artifact_store.write(**write_arguments)

    def create_checkpoint_with_result(
        self,
        state: ModAgentState,
        *,
        interrupt_reason: str,
        resume_allowed_fields: tuple[str, ...],
        exact_revision_upstream_types: frozenset[str] = frozenset(),
    ) -> ArtifactWriteResult:
        """Persist a checkpoint and return the Store-owned commit outcome."""

        try:
            write_arguments = self._checkpoint_write_arguments(
                state,
                interrupt_reason=interrupt_reason,
                resume_allowed_fields=resume_allowed_fields,
                exact_revision_upstream_types=exact_revision_upstream_types,
            )
        except Exception as exc:
            raise ArtifactStoreError(
                f"Workflow checkpoint prewrite validation failed: {exc}"
            ) from exc
        return self.artifact_store.write_with_result(**write_arguments)

    def _checkpoint_write_arguments(
        self,
        state: ModAgentState,
        *,
        interrupt_reason: str,
        resume_allowed_fields: tuple[str, ...],
        exact_revision_upstream_types: frozenset[str],
    ) -> dict[str, Any]:
        """Build one checkpoint write without changing the write API contract."""

        feature_id = state["feature_id"] or state["request_id"]
        exact_revision_types = tuple(
            sorted(exact_revision_upstream_types)
        )
        exact_role_policy_blockers = self._build_checkpoint_exact_role_policy_blockers(
            frozenset(exact_revision_types)
        )
        if exact_role_policy_blockers:
            raise ArtifactStoreError(
                "; ".join(
                    str(blocker["message"])
                    for blocker in exact_role_policy_blockers
                )
            )
        artifact_refs = tuple(state["artifacts"].values())
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=f"checkpoint_{uuid4().hex}",
            run_id=state["run_id"],
            feature_id=feature_id,
            request_id=state["request_id"],
            stage=state["current_stage"],
            status="interrupted",
            state_snapshot=self._state_snapshot(state),
            artifact_refs=artifact_refs,
            artifact_hashes=dict(state["artifact_hashes"]),
            resume_binding=self._resume_binding_from_state(state),
            resume_allowed_fields=resume_allowed_fields,
            interrupt_reason=interrupt_reason,
            policy=self.policy,
        )
        checkpoint_payload = checkpoint.to_artifact_payload()
        if exact_revision_types:
            prewrite_blockers = self._build_checkpoint_exact_lineage_blockers(
                state,
                exact_revision_types=frozenset(exact_revision_types),
            )
            prewrite_blockers.extend(
                self._build_handoff_blockers(checkpoint_payload)
            )
            prewrite_blockers.extend(
                self._checkpoint_execution_diff_binding_blockers(
                    checkpoint_payload
                )
            )
            if prewrite_blockers:
                raise ArtifactStoreError(
                    "; ".join(
                        str(blocker["message"])
                        for blocker in prewrite_blockers
                    )
                )
        return {
            "feature_id": feature_id,
            "artifact_type": "workflow_checkpoint",
            "payload": checkpoint_payload,
            "status": "success",
            "upstream_refs": self._artifact_refs_from_state(state),
            "exact_revision_upstream_types": set(exact_revision_types),
        }

    def resume(
        self,
        checkpoint_ref: ArtifactRef,
        payload: ResumePayload,
    ) -> tuple[ResumeDecision, ArtifactRef]:
        """Validate a resume payload and write a resume_report artifact."""

        from .resume_lock import checkpoint_resume_lock

        with checkpoint_resume_lock(self.artifact_store.root, checkpoint_ref):
            checkpoint, decision, write_arguments = self._resume_write_arguments(checkpoint_ref, payload)
            report_ref = self.artifact_store.write(**write_arguments)
            self._record_consumed_resume(checkpoint, payload, decision=decision, committed=True)
            return decision, report_ref

    def resume_with_result(
        self,
        checkpoint_ref: ArtifactRef,
        payload: ResumePayload,
    ) -> tuple[ResumeDecision, ArtifactWriteResult]:
        """Validate one resume and return the Store-owned commit outcome."""

        from .resume_lock import checkpoint_resume_lock

        with checkpoint_resume_lock(self.artifact_store.root, checkpoint_ref):
            checkpoint, decision, write_arguments = self._resume_write_arguments(checkpoint_ref, payload)
            write_result = self.artifact_store.write_with_result(**write_arguments)
            self._record_consumed_resume(checkpoint, payload, decision=decision,
                                         committed=write_result.outcome == "committed")
            return decision, write_result

    def _record_consumed_resume(
        self,
        checkpoint: dict[str, Any],
        payload: ResumePayload,
        *,
        decision: ResumeDecision,
        committed: bool,
    ) -> None:
        """Record idempotency only after the accepted report is durable."""

        if decision.allowed and committed:
            self._seen_resume_ids.add(payload.resume_id)
            self._consumed_checkpoint_ids.add(checkpoint["checkpoint_id"])

    def _resume_write_arguments(
        self,
        checkpoint_ref: ArtifactRef,
        payload: ResumePayload,
    ) -> tuple[dict[str, Any], ResumeDecision, dict[str, Any]]:
        """Build the shared ResumeReport decision and Store write arguments."""

        checkpoint = self.artifact_store.read(checkpoint_ref)["data"]
        decision = self.preflight_resume(
            checkpoint_ref,
            resume_id=payload.resume_id,
        )
        if decision.allowed:
            decision = self.resume_validator.validate(payload, checkpoint=checkpoint)
        report = ResumeReport(
            resume_id=payload.resume_id,
            checkpoint_id=str(checkpoint["checkpoint_id"]),
            run_id=payload.run_id,
            stage=payload.stage,
            decision=payload.decision,
            status=decision.status,
            allowed=decision.allowed,
            reason=decision.reason,
            applied_updates=decision.applied_updates,
            stale_reasons=tuple(decision.stale_reasons),
            blockers=tuple(decision.blockers),
            approved_by=payload.approved_by,
            approved_at=payload.approved_at,
            authority_request_hash=payload.authority_request_hash,
        )
        from src.mod_agent.execution.task_delegation import delegation_resume_report_fields

        report_payload = report.to_artifact_payload()
        report_payload.update(delegation_resume_report_fields(payload, checkpoint))
        return checkpoint, decision, {
            "feature_id": checkpoint_ref.feature_id,
            "artifact_type": "resume_report",
            "payload": report_payload,
            "status": "success" if decision.allowed else "blocked",
            "upstream_refs": [checkpoint_ref],
            "allow_stale_upstreams": (
                not decision.allowed and bool(decision.stale_reasons)
            ),
            "exact_revision_upstream_types": {"workflow_checkpoint"},
        }

    def _checkpoint_freshness(
        self,
        checkpoint_ref: ArtifactRef,
        checkpoint_envelope: dict[str, Any],
        *,
        additional_exact_revision_types: frozenset[str] = frozenset(),
    ):
        checkpoint = checkpoint_envelope["data"]
        if checkpoint.get("checkpoint_stage") != "build":
            return self.artifact_store.check_direct_freshness(checkpoint_ref)
        exact_types = {
            str(item.get("artifact_type"))
            for item in checkpoint_envelope.get("inputs", {}).get(
                "upstream_artifacts",
                [],
            )
            if isinstance(item, dict) and item.get("require_latest") is False
        }
        exact_types.update(additional_exact_revision_types)
        return self.artifact_store.check_freshness_with_exact_revision_types(
            checkpoint_ref,
            exact_revision_types=exact_types,
        )

    def read_accepted_runtime_checkpoint(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_report_ref: ArtifactRef,
        resume_id: str,
        approved_by: str,
        runtime_action_scope: dict[str, Any],
        allow_prior_runtime_state: bool = False,
    ) -> dict[str, Any]:
        """Read one accepted operation's immutable Build checkpoint, without writes."""

        checkpoint_envelope = self.artifact_store.read(checkpoint_ref)
        checkpoint = checkpoint_envelope["data"]
        report_envelope = self.artifact_store.read(resume_report_ref)
        report = report_envelope["data"]
        updates = report.get("applied_updates") or {}
        admission = updates.get("runtime_admission")
        upstreams = report_envelope.get("inputs", {}).get("upstream_artifacts", [])
        if (
            checkpoint_ref.artifact_type != "workflow_checkpoint"
            or resume_report_ref.artifact_type != "resume_report"
            or resume_report_ref.feature_id != checkpoint_ref.feature_id
            or checkpoint.get("checkpoint_stage") != "build"
            or report.get("checkpoint_id") != checkpoint.get("checkpoint_id")
            or report.get("run_id") != checkpoint.get("run_id")
            or report.get("resume_stage") != "build"
            or report.get("resume_id") != resume_id
            or not resume_id
            or report.get("approved_by") != approved_by
            or not approved_by
            or report.get("status") != "accepted"
            or report.get("decision") != "approve"
            or report.get("allowed") is not True
            or updates.get("restored_state") != checkpoint.get("state_snapshot")
            or not isinstance(admission, dict)
            or not isinstance(upstreams, list)
            or len(upstreams) != 1
            or upstreams[0].get("artifact_type") != "workflow_checkpoint"
            or upstreams[0].get("path") != checkpoint_ref.path
            or upstreams[0].get("artifact_hash") != checkpoint_ref.artifact_hash
        ):
            raise ValueError("ACCEPTED_RUNTIME_CHECKPOINT_BINDING_MISMATCH")
        # Scope: accept the persisted snapshot or its exact initial-admission input.
        # 为什么要有它：初次执行与恢复使用不同投影，但不能扩大已批准动作。
        if runtime_action_scope != admission and runtime_action_scope != admission.get(
            "runtime_admission"
        ):
            raise ValueError("ACCEPTED_RUNTIME_ACTION_SCOPE_MISMATCH")
        approval = self.resume_validator.validate(
            ResumePayload(
                resume_id=resume_id,
                run_id=str(checkpoint["run_id"]),
                request_id=str(checkpoint["request_id"]),
                feature_id=checkpoint_ref.feature_id,
                stage="build",
                decision="approve",
                approved_by=approved_by,
                approved_at=str(report.get("approved_at") or ""),
                artifact_refs=tuple(checkpoint["artifact_refs"]),
                runtime_admission=admission,
            ),
            checkpoint=checkpoint,
        )
        if not approval.allowed:
            raise ValueError("ACCEPTED_RUNTIME_ADMISSION_INVALID")
        current_reports = [
            ref for ref in self.artifact_store.list_feature_artifacts(checkpoint_ref.feature_id)
            if ref.artifact_type == "resume_report"
        ]
        if (
            len(current_reports) != 1
            or self.artifact_store.read(current_reports[0])["artifact_hash"]
            != resume_report_ref.artifact_hash
        ):
            raise ValueError("ACCEPTED_RUNTIME_REPORT_STALE: current report changed")
        report_freshness = self.artifact_store.check_freshness(resume_report_ref)
        if not report_freshness.is_fresh:
            raise ValueError(
                "ACCEPTED_RUNTIME_REPORT_STALE: "
                + "; ".join(report_freshness.stale_reasons)
            )
        blockers = self._checkpoint_lineage_blockers(
            checkpoint_ref,
            checkpoint_envelope,
            allow_prior_runtime_state=allow_prior_runtime_state,
        )
        if blockers:
            raise ValueError(
                "ACCEPTED_RUNTIME_CHECKPOINT_INVALID: "
                + ", ".join(str(item.get("code")) for item in blockers)
            )
        self._validate_historical_resume_reports(checkpoint_ref, checkpoint)
        # Scope: only history consumed by the exact current accepted operation.
        # 为什么要有它：当前批准会推进报告指针，历史批准仍是不可变输入；
        # FeatureSpec、Build、源码与其他材料的最新性和内容校验不能被放宽。
        freshness = self._checkpoint_freshness(
            checkpoint_ref,
            checkpoint_envelope,
            additional_exact_revision_types=frozenset({"resume_report"}),
        )
        if not freshness.is_fresh:
            raise ValueError(
                "STALE_RESUME_RECOVERY_REJECTED: "
                + "; ".join(freshness.stale_reasons)
            )
        return checkpoint_envelope

    def _validate_historical_resume_reports(
        self, checkpoint_ref: ArtifactRef, checkpoint: dict[str, Any]
    ) -> None:
        """Constrain exact control history to accepted approvals in this same run."""

        pending = [checkpoint_ref]
        seen: set[ArtifactRef] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            envelope = self.artifact_store.read(current)
            upstreams = envelope.get("inputs", {}).get("upstream_artifacts", [])
            if current.artifact_type == "resume_report":
                report = envelope["data"]
                if len(upstreams) != 1 or upstreams[0].get("artifact_type") != "workflow_checkpoint":
                    raise ValueError("HISTORICAL_RESUME_REPORT_BINDING_MISMATCH")
                bound_checkpoint = self.artifact_store.read(ArtifactRef(
                    feature_id=checkpoint_ref.feature_id,
                    artifact_type="workflow_checkpoint",
                    path=upstreams[0]["path"],
                    artifact_hash=upstreams[0]["artifact_hash"],
                ))["data"]
                # Scope: only same-run, exact accepted historical control edges.
                # 为什么要有它：按类型保留历史不能借同 feature 的其他运行或拒绝记录绕过批准。
                if (
                    report.get("run_id") != checkpoint.get("run_id")
                    or bound_checkpoint.get("run_id") != checkpoint.get("run_id")
                    or bound_checkpoint.get("request_id") != checkpoint.get("request_id")
                    or bound_checkpoint.get("feature_id") != checkpoint_ref.feature_id
                    or report.get("checkpoint_id") != bound_checkpoint.get("checkpoint_id")
                    or report.get("resume_stage") != bound_checkpoint.get("checkpoint_stage")
                    or report.get("decision") != "approve"
                    or report.get("status") != "accepted"
                    or report.get("allowed") is not True
                    or not report.get("approved_by")
                    or not report.get("resume_id")
                    or (report.get("applied_updates") or {}).get("restored_state")
                    != bound_checkpoint.get("state_snapshot")
                ):
                    raise ValueError("HISTORICAL_RESUME_REPORT_BINDING_MISMATCH")
            pending.extend(
                ArtifactRef(
                    feature_id=checkpoint_ref.feature_id,
                    artifact_type=item["artifact_type"],
                    path=item["path"],
                    artifact_hash=item["artifact_hash"],
                )
                for item in upstreams
            )

    def preflight_verification_operation(
        self,
        envelope: VerificationOperationEnvelope,
        declaration: VerificationOperationDeclaration | None = None,
    ) -> VerificationCheckpointAdmissionResult:
        """Admit one normalized operation's checkpoint protocol and attempt."""

        validated = validate_verification_operation_envelope(envelope)
        resolved = resolve_verification_operation(validated)
        if declaration is not None and declaration != resolved:
            raise ValueError(
                "Verification operation declaration changed before checkpoint admission."
            )

        shape_id = resolved.shape_id
        protocol = validated.protocol_binding
        attempt = validated.attempt_policy
        variant = validated.variant_payload

        if shape_id == "verification_build_initial":
            return None

        if shape_id in {
            "runtime_execution_initial_evidence_only",
            "runtime_execution_initial_workflow",
        }:
            if (
                type(protocol) is not NewRuntimeResumeProtocolBindingV1
                or type(attempt) is not InitialAttemptV1
            ):
                raise ValueError(
                    "Initial Runtime checkpoint protocol or attempt changed."
                )
            return self._preflight_resume_facts(
                protocol.checkpoint_ref,
                resume_id=protocol.resume_id,
            )

        if shape_id == "runtime_execution_observation_retry":
            if (
                type(protocol) is not VerificationProtocolBinding
                or type(attempt) is not ObservationRetryAttemptV1
            ):
                raise ValueError(
                    "Observation retry checkpoint protocol or attempt changed."
                )
            admitted = self._preflight_runtime_observation_retry_facts(
                protocol.checkpoint_ref,
                resume_id=protocol.resume_id,
            )
            (
                _decision,
                report_ref,
                prior_runtime_ref,
                repair_plan_ref,
                retry_binding,
            ) = admitted
            report = self.artifact_store.read(report_ref)["data"]
            remaining_before = retry_binding.get(
                "retry_budget_remaining_before_attempt"
            )
            if (
                report_ref != protocol.accepted_resume_report_ref
                or report.get("approved_by") != protocol.approved_by
                or prior_runtime_ref != attempt.prior_runtime_evidence_ref
                or repair_plan_ref != attempt.repair_plan_ref
                or retry_binding.get("retry_number") != attempt.retry_number
                or retry_binding.get("retry_budget") != attempt.retry_budget
                or type(remaining_before) is not int
                or max(remaining_before - 1, 0)
                != attempt.retry_budget_remaining
            ):
                raise ValueError(
                    "Observation retry protocol or attempt does not match current facts."
                )
            return admitted

        if shape_id == "runtime_execution_accepted_resume_recovery":
            if (
                type(protocol) is not VerificationProtocolBinding
                or type(attempt) is not AcceptedResumeRecoveryAttemptV1
            ):
                raise ValueError(
                    "Accepted-resume recovery protocol or attempt changed."
                )
            admitted = self._preflight_accepted_resume_recovery_facts(
                protocol.checkpoint_ref,
                resume_id=protocol.resume_id,
            )
            _decision, report_ref = admitted
            report = self.artifact_store.read(report_ref)["data"]
            if (
                report_ref != protocol.accepted_resume_report_ref
                or report_ref != attempt.accepted_resume_report_ref
                or report.get("approved_by") != protocol.approved_by
                or attempt.recovery_of_resume_id != protocol.resume_id
            ):
                raise ValueError(
                    "Accepted-resume recovery protocol does not match current facts."
                )
            return admitted

        if shape_id == "runtime_terminalization_initial":
            if (
                type(protocol) is not VerificationProtocolBinding
                or type(attempt) is not InitialAttemptV1
                or type(variant) is not RuntimeTerminalizationOperationV1
            ):
                raise ValueError(
                    "Runtime terminalization protocol or attempt changed."
                )
            return self._preflight_runtime_terminal_continuation_facts(
                protocol.checkpoint_ref,
                resume_report_ref=protocol.accepted_resume_report_ref,
                runtime_evidence_ref=variant.runtime_evidence_ref,
                resume_id=protocol.resume_id,
                approved_by=protocol.approved_by,
            )

        if shape_id == "runtime_evidence_consolidation_initial":
            if (
                type(protocol) is not VerificationProtocolBinding
                or type(attempt) is not VerificationAttemptPolicy
                or type(variant) is not EvidenceConsolidationOperationV1
            ):
                raise ValueError(
                    "RuntimeEvidence consolidation protocol changed."
                )
            return self._preflight_runtime_evidence_consolidation_facts(
                protocol.checkpoint_ref,
                resume_report_ref=protocol.accepted_resume_report_ref,
                prior_runtime_evidence_ref=variant.prior_runtime_evidence_ref,
                resume_id=protocol.resume_id,
                approved_by=protocol.approved_by,
            )

        raise ValueError(
            "Verification operation has no checkpoint admission handler."
        )

    def preflight_resume(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_id: str | None = None,
    ) -> ResumeDecision:
        """Compatibility wrapper for initial checkpoint admission."""

        return self._preflight_resume_facts(
            checkpoint_ref,
            resume_id=resume_id,
        )

    def _preflight_resume_facts(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_id: str | None = None,
    ) -> ResumeDecision:
        """Validate checkpoint integrity and consumption without writing or consuming."""

        checkpoint_envelope = self.artifact_store.read(checkpoint_ref)
        checkpoint = checkpoint_envelope["data"]
        lineage_blockers = self._checkpoint_lineage_blockers(
            checkpoint_ref,
            checkpoint_envelope,
        )
        freshness = self._checkpoint_freshness(
            checkpoint_ref,
            checkpoint_envelope,
        )
        consumed, seen_resume_id = self._durable_consumption_state(
            checkpoint_ref.feature_id,
            checkpoint_id=str(checkpoint.get("checkpoint_id") or ""),
            resume_id=resume_id or "",
        )
        if lineage_blockers:
            return ResumeDecision(
                status="blocked",
                allowed=False,
                reason="Resume blocked because checkpoint lineage is invalid.",
                applied_updates={},
                blockers=tuple(lineage_blockers),
            )
        if not freshness.is_fresh:
            return ResumeDecision(
                status="blocked",
                allowed=False,
                reason="Resume blocked because checkpoint inputs are stale.",
                applied_updates={},
                blockers=(
                    {
                        "code": "STALE_RESUME_REJECTED",
                        "message": "Checkpoint upstream artifacts changed.",
                    },
                ),
                stale_reasons=tuple(freshness.stale_reasons),
            )
        if (
            (resume_id is not None and resume_id in self._seen_resume_ids)
            or checkpoint["checkpoint_id"] in self._consumed_checkpoint_ids
            or seen_resume_id
            or consumed
        ):
            return ResumeDecision(
                status="blocked",
                allowed=False,
                reason="Resume blocked because checkpoint or resume_id was already used.",
                applied_updates={},
                blockers=(
                    {
                        "code": "DOUBLE_RESUME_REJECTED",
                        "message": "Checkpoint resume is idempotent and was already consumed.",
                    },
                ),
            )
        return ResumeDecision(
            status="ready",
            allowed=True,
            reason="Checkpoint is ready for a separately authorized resume.",
            applied_updates={},
        )

    def preflight_accepted_resume_recovery(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_id: str,
    ) -> tuple[ResumeDecision, ArtifactRef]:
        """Compatibility wrapper for accepted-resume recovery admission."""

        return self._preflight_accepted_resume_recovery_facts(
            checkpoint_ref,
            resume_id=resume_id,
        )

    def _preflight_accepted_resume_recovery_facts(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_id: str,
    ) -> tuple[ResumeDecision, ArtifactRef]:
        """Read-validate one exact accepted pre-Runtime resume without writing."""

        if not resume_id:
            raise ValueError(
                "Accepted resume recovery requires the exact prior resume_id."
            )
        checkpoint_envelope = self.artifact_store.read(checkpoint_ref)
        checkpoint = checkpoint_envelope["data"]
        blockers = self._checkpoint_lineage_blockers(
            checkpoint_ref,
            checkpoint_envelope,
        )
        accepted_reports: list[tuple[ArtifactRef, dict[str, Any], dict[str, Any]]] = []
        for report_ref in self.artifact_store.list_feature_artifact_revisions(
            checkpoint_ref.feature_id,
            "resume_report",
        ):
            envelope = self.artifact_store.read(report_ref)
            report = envelope["data"]
            if (
                report.get("checkpoint_id") == checkpoint.get("checkpoint_id")
                and report.get("allowed") is True
            ):
                accepted_reports.append((report_ref, envelope, report))
        if len(accepted_reports) != 1:
            raise ValueError(
                "Accepted resume recovery requires exactly one accepted "
                "ResumeReport for the checkpoint."
            )
        report_ref, report_envelope, report = accepted_reports[0]
        if report.get("resume_id") != resume_id:
            raise ValueError(
                "Accepted resume recovery resume_id does not match ResumeReport."
            )
        upstream = report_envelope.get("inputs", {}).get("upstream_artifacts", [])
        if (
            len(upstream) != 1
            or upstream[0].get("artifact_type") != "workflow_checkpoint"
            or upstream[0].get("path") != checkpoint_ref.path
            or upstream[0].get("artifact_hash") != checkpoint_ref.artifact_hash
        ):
            blockers.append(
                {
                    "code": "RESUME_RECOVERY_REPORT_BINDING_MISMATCH",
                    "message": (
                        "Accepted ResumeReport is not bound to the exact checkpoint."
                    ),
                }
            )
        restored_state = report.get("applied_updates", {}).get("restored_state")
        if (
            report.get("status") != "accepted"
            or report.get("run_id") != checkpoint.get("run_id")
            or report.get("resume_stage") != checkpoint.get("checkpoint_stage")
            or not isinstance(report.get("applied_updates"), dict)
            or not isinstance(restored_state, dict)
            or restored_state != checkpoint.get("state_snapshot")
        ):
            blockers.append(
                {
                    "code": "RESUME_RECOVERY_REPORT_INVALID",
                    "message": (
                        "Accepted ResumeReport cannot reconstruct the exact "
                        "pre-Runtime graph state."
                    ),
                }
            )
        for artifact_type in ("runtime_evidence", "eval_report", "workflow_summary"):
            for ref in self.artifact_store.list_feature_artifact_revisions(
                checkpoint_ref.feature_id,
                artifact_type,
            ):
                artifact = self.artifact_store.read(ref)["data"]
                if artifact.get("run_id") == checkpoint.get("run_id"):
                    blockers.append(
                        {
                            "code": "RESUME_RECOVERY_DOWNSTREAM_STATE_PRESENT",
                            "message": (
                                f"{artifact_type} already exists for this handoff run."
                            ),
                        }
                    )
                    break
        if blockers:
            raise ValueError(
                "Accepted resume recovery was blocked: "
                + ", ".join(str(item.get("code")) for item in blockers)
            )
        self.read_accepted_runtime_checkpoint(
            checkpoint_ref,
            resume_report_ref=report_ref,
            resume_id=resume_id,
            approved_by=str(report.get("approved_by") or ""),
            runtime_action_scope=dict(report["applied_updates"].get("runtime_admission") or {}),
        )
        return (
            ResumeDecision(
                status="accepted",
                allowed=True,
                reason="Exact accepted ResumeReport is ready for bounded recovery.",
                applied_updates=dict(report["applied_updates"]),
            ),
            report_ref,
        )

    def preflight_runtime_terminal_continuation(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_report_ref: ArtifactRef,
        runtime_evidence_ref: ArtifactRef,
        resume_id: str,
        approved_by: str,
    ) -> tuple[ResumeDecision, dict[str, Any], tuple[ArtifactRef, ...]]:
        """Compatibility wrapper for terminal checkpoint admission."""

        return self._preflight_runtime_terminal_continuation_facts(
            checkpoint_ref,
            resume_report_ref=resume_report_ref,
            runtime_evidence_ref=runtime_evidence_ref,
            resume_id=resume_id,
            approved_by=approved_by,
        )

    def _runtime_resume_preflight_facts(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_report_ref: ArtifactRef,
        runtime_evidence_ref: ArtifactRef,
        resume_id: str,
        approved_by: str,
        require_exact_runtime_upstream_types: bool,
    ) -> dict[str, Any]:
        """Load and validate facts shared by the two post-Runtime admissions."""

        checkpoint_envelope = self.artifact_store.read(checkpoint_ref)
        checkpoint = checkpoint_envelope["data"]
        blockers = self._checkpoint_lineage_blockers(
            checkpoint_ref,
            checkpoint_envelope,
            allow_prior_runtime_state=True,
            require_current_build_output=False,
        )
        checkpoint_freshness = self._checkpoint_freshness(
            checkpoint_ref,
            checkpoint_envelope,
        )

        report_envelope = self.artifact_store.read(resume_report_ref)
        report = report_envelope["data"]
        report_upstreams = report_envelope.get("inputs", {}).get(
            "upstream_artifacts",
            [],
        )
        restored_state = (report.get("applied_updates") or {}).get(
            "restored_state"
        )
        report_valid = not (
            resume_report_ref.feature_id != checkpoint_ref.feature_id
            or report.get("checkpoint_id") != checkpoint.get("checkpoint_id")
            or report.get("run_id") != checkpoint.get("run_id")
            or report.get("resume_stage") != checkpoint.get("checkpoint_stage")
            or report.get("resume_id") != resume_id
            or report.get("approved_by") != approved_by
            or report.get("decision") != "approve"
            or report.get("status") != "accepted"
            or report.get("allowed") is not True
            or restored_state != checkpoint.get("state_snapshot")
            or not isinstance(report_upstreams, list)
            or len(report_upstreams) != 1
            or report_upstreams[0].get("artifact_type")
            != "workflow_checkpoint"
            or report_upstreams[0].get("path") != checkpoint_ref.path
            or report_upstreams[0].get("artifact_hash")
            != checkpoint_ref.artifact_hash
        )
        report_freshness = self.artifact_store.check_freshness(
            resume_report_ref
        )

        runtime_envelope = self.artifact_store.read(runtime_evidence_ref)
        runtime = runtime_envelope["data"]
        runtime_upstreams = runtime_envelope.get("inputs", {}).get(
            "upstream_artifacts",
            [],
        )
        snapshot = checkpoint.get("state_snapshot") or {}
        snapshot_artifacts = snapshot.get("artifacts") or {}
        snapshot_hashes = snapshot.get("artifact_hashes") or {}
        runtime_upstreams_by_type = {
            item.get("artifact_type"): item
            for item in runtime_upstreams
            if isinstance(item, dict)
            and isinstance(item.get("artifact_type"), str)
        }
        required_runtime_upstream_types = {"build_result", "feature_spec"}
        observed_runtime_upstream_types = set(runtime_upstreams_by_type)
        runtime_upstream_types_valid = (
            observed_runtime_upstream_types == required_runtime_upstream_types
            if require_exact_runtime_upstream_types
            else required_runtime_upstream_types.issubset(
                observed_runtime_upstream_types
            )
        )
        expected_build_ref = {
            "feature_id": checkpoint_ref.feature_id,
            "artifact_type": "build_result",
            "path": snapshot_artifacts.get("build_result"),
            "artifact_hash": snapshot_hashes.get("build_result"),
        }
        build_upstream = runtime_upstreams_by_type.get("build_result", {})
        feature_upstream = runtime_upstreams_by_type.get("feature_spec", {})
        runtime_lineage_valid = not (
            runtime_evidence_ref.feature_id != checkpoint_ref.feature_id
            or runtime.get("feature_id") != checkpoint_ref.feature_id
            or runtime.get("run_id") != checkpoint.get("run_id")
            or runtime.get("input_execution_mode")
            != VERIFICATION_ONLY_INPUT_EXECUTION_MODE
            or runtime.get("build_artifact_ref") != expected_build_ref
            or runtime.get("build_artifact_hash")
            != snapshot_hashes.get("build_result")
            or not runtime_upstream_types_valid
            or build_upstream.get("path")
            != snapshot_artifacts.get("build_result")
            or build_upstream.get("artifact_hash")
            != snapshot_hashes.get("build_result")
            or feature_upstream.get("path")
            != snapshot_artifacts.get("feature_spec")
            or feature_upstream.get("artifact_hash")
            != snapshot_hashes.get("feature_spec")
        )
        runtime_freshness = self.artifact_store.check_freshness(
            runtime_evidence_ref
        )
        return {
            "checkpoint_freshness": checkpoint_freshness,
            "report": report,
            "report_freshness": report_freshness,
            "report_valid": report_valid,
            "runtime": runtime,
            "runtime_freshness": runtime_freshness,
            "runtime_lineage_valid": runtime_lineage_valid,
            "blockers": blockers,
        }

    def _preflight_runtime_terminal_continuation_facts(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_report_ref: ArtifactRef,
        runtime_evidence_ref: ArtifactRef,
        resume_id: str,
        approved_by: str,
    ) -> tuple[ResumeDecision, dict[str, Any], tuple[ArtifactRef, ...]]:
        """Validate one exact post-Runtime terminal continuation without writes."""

        facts = self._runtime_resume_preflight_facts(
            checkpoint_ref,
            resume_report_ref=resume_report_ref,
            runtime_evidence_ref=runtime_evidence_ref,
            resume_id=resume_id,
            approved_by=approved_by,
            require_exact_runtime_upstream_types=True,
        )
        checkpoint_freshness = facts["checkpoint_freshness"]
        report = facts["report"]
        report_freshness = facts["report_freshness"]
        runtime = facts["runtime"]
        runtime_freshness = facts["runtime_freshness"]
        blockers = facts["blockers"]
        if not checkpoint_freshness.is_fresh:
            blockers.append(
                {
                    "code": "RUNTIME_TERMINAL_CHECKPOINT_STALE",
                    "message": "Post-Runtime checkpoint inputs are stale.",
                }
            )

        if not facts["report_valid"]:
            blockers.append(
                {
                    "code": "RUNTIME_TERMINAL_RESUME_REPORT_INVALID",
                    "message": (
                        "Accepted ResumeReport does not bind the exact Build "
                        "checkpoint and approved resume identity."
                    ),
                }
            )
        if not report_freshness.is_fresh:
            blockers.append(
                {
                    "code": "RUNTIME_TERMINAL_RESUME_REPORT_STALE",
                    "message": "Accepted ResumeReport inputs are stale.",
                }
            )

        if not facts["runtime_lineage_valid"]:
            blockers.append(
                {
                    "code": "RUNTIME_TERMINAL_EVIDENCE_LINEAGE_INVALID",
                    "message": (
                        "RuntimeEvidence does not bind the exact checkpoint "
                        "FeatureSpec and BuildResult lineage."
                    ),
                }
            )
        if not runtime_freshness.is_fresh:
            blockers.append(
                {
                    "code": "RUNTIME_TERMINAL_EVIDENCE_STALE",
                    "message": "RuntimeEvidence inputs are stale.",
                }
            )

        terminal_refs = self._runtime_terminal_artifact_refs(
            runtime_evidence_ref
        )
        if terminal_refs:
            blockers.append(
                {
                    "code": "RUNTIME_TERMINALIZATION_ALREADY_STARTED",
                    "message": (
                        "Terminal Artifacts already exist for the exact "
                        "RuntimeEvidence."
                    ),
                }
            )
        if blockers:
            return (
                ResumeDecision(
                    status="blocked",
                    allowed=False,
                    reason="Post-Runtime terminal continuation was blocked.",
                    applied_updates={},
                    blockers=tuple(blockers),
                    stale_reasons=tuple(
                        checkpoint_freshness.stale_reasons
                        + report_freshness.stale_reasons
                        + runtime_freshness.stale_reasons
                    ),
                ),
                runtime,
                terminal_refs,
            )
        return (
            ResumeDecision(
                status="accepted",
                allowed=True,
                reason=(
                    "Exact committed RuntimeEvidence is ready for terminal "
                    "Graph continuation."
                ),
                applied_updates=dict(report["applied_updates"]),
            ),
            runtime,
            (),
        )

    def preflight_runtime_evidence_consolidation(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_report_ref: ArtifactRef,
        prior_runtime_evidence_ref: ArtifactRef,
        resume_id: str,
        approved_by: str,
    ) -> tuple[ResumeDecision, dict[str, Any], tuple[ArtifactRef, ...]]:
        """Compatibility wrapper for consolidation checkpoint admission."""

        return self._preflight_runtime_evidence_consolidation_facts(
            checkpoint_ref,
            resume_report_ref=resume_report_ref,
            prior_runtime_evidence_ref=prior_runtime_evidence_ref,
            resume_id=resume_id,
            approved_by=approved_by,
        )

    def _preflight_runtime_evidence_consolidation_facts(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_report_ref: ArtifactRef,
        prior_runtime_evidence_ref: ArtifactRef,
        resume_id: str,
        approved_by: str,
    ) -> tuple[ResumeDecision, dict[str, Any], tuple[ArtifactRef, ...]]:
        """Validate one exact no-Runtime RuntimeEvidence consolidation."""

        facts = self._runtime_resume_preflight_facts(
            checkpoint_ref,
            resume_report_ref=resume_report_ref,
            runtime_evidence_ref=prior_runtime_evidence_ref,
            resume_id=resume_id,
            approved_by=approved_by,
            require_exact_runtime_upstream_types=False,
        )
        checkpoint_freshness = facts["checkpoint_freshness"]
        report = facts["report"]
        report_freshness = facts["report_freshness"]
        runtime = facts["runtime"]
        runtime_freshness = facts["runtime_freshness"]
        blockers = facts["blockers"]
        if not checkpoint_freshness.is_fresh:
            blockers.append(
                {
                    "code": "RUNTIME_CONSOLIDATION_CHECKPOINT_STALE",
                    "message": "RuntimeEvidence consolidation checkpoint is stale.",
                }
            )

        if not facts["report_valid"]:
            blockers.append(
                {
                    "code": "RUNTIME_CONSOLIDATION_RESUME_REPORT_INVALID",
                    "message": (
                        "Accepted ResumeReport does not bind the exact Build "
                        "checkpoint and approved resume identity."
                    ),
                }
            )
        if not report_freshness.is_fresh:
            blockers.append(
                {
                    "code": "RUNTIME_CONSOLIDATION_RESUME_REPORT_STALE",
                    "message": "Accepted ResumeReport inputs are stale.",
                }
            )

        if not facts["runtime_lineage_valid"]:
            blockers.append(
                {
                    "code": "RUNTIME_CONSOLIDATION_EVIDENCE_LINEAGE_INVALID",
                    "message": (
                        "Prior RuntimeEvidence does not bind the exact "
                        "checkpoint FeatureSpec and BuildResult lineage."
                    ),
                }
            )
        if not runtime_freshness.is_fresh:
            blockers.append(
                {
                    "code": "RUNTIME_CONSOLIDATION_EVIDENCE_STALE",
                    "message": "Prior RuntimeEvidence inputs are stale.",
                }
            )
        latest_runtime = [
            ref
            for ref in self.artifact_store.list_feature_artifacts(
                checkpoint_ref.feature_id
            )
            if ref.artifact_type == "runtime_evidence"
        ]
        if (
            len(latest_runtime) != 1
            or latest_runtime[0].artifact_hash
            != prior_runtime_evidence_ref.artifact_hash
        ):
            blockers.append(
                {
                    "code": "RUNTIME_CONSOLIDATION_EVIDENCE_NOT_LATEST",
                    "message": "Prior RuntimeEvidence is not the latest revision.",
                }
            )

        if blockers:
            return (
                ResumeDecision(
                    status="blocked",
                    allowed=False,
                    reason="RuntimeEvidence consolidation was blocked.",
                    applied_updates={},
                    blockers=tuple(blockers),
                    stale_reasons=tuple(
                        checkpoint_freshness.stale_reasons
                        + report_freshness.stale_reasons
                        + runtime_freshness.stale_reasons
                    ),
                ),
                runtime,
                (),
            )
        return (
            ResumeDecision(
                status="accepted",
                allowed=True,
                reason=(
                    "Exact RuntimeEvidence predecessor is ready for one "
                    "implementation-evidence consolidation."
                ),
                applied_updates=dict(report["applied_updates"]),
            ),
            runtime,
            (),
        )

    def _runtime_terminal_artifact_refs(
        self,
        runtime_evidence_ref: ArtifactRef,
    ) -> tuple[ArtifactRef, ...]:
        """Return immutable terminal revisions rooted at RuntimeEvidence."""

        feature_id = runtime_evidence_ref.feature_id
        eval_refs: list[ArtifactRef] = []
        terminal_refs: list[ArtifactRef] = []

        def binds(
            envelope: dict[str, Any],
            upstream_ref: ArtifactRef,
        ) -> bool:
            return any(
                isinstance(item, dict)
                and item.get("artifact_type") == upstream_ref.artifact_type
                and item.get("path") == upstream_ref.path
                and item.get("artifact_hash") == upstream_ref.artifact_hash
                for item in envelope.get("inputs", {}).get(
                    "upstream_artifacts",
                    [],
                )
            )

        for ref in self.artifact_store.list_feature_artifact_revisions(
            feature_id,
            "eval_report",
        ):
            envelope = self.artifact_store.read(ref)
            if binds(envelope, runtime_evidence_ref):
                eval_refs.append(ref)
                terminal_refs.append(ref)
        for ref in self.artifact_store.list_feature_artifact_revisions(
            feature_id,
            "blocked_report",
        ):
            envelope = self.artifact_store.read(ref)
            if binds(envelope, runtime_evidence_ref) or any(
                binds(envelope, eval_ref) for eval_ref in eval_refs
            ):
                terminal_refs.append(ref)
        for ref in self.artifact_store.list_feature_artifact_revisions(
            feature_id,
            "workflow_summary",
        ):
            envelope = self.artifact_store.read(ref)
            if any(binds(envelope, eval_ref) for eval_ref in eval_refs):
                terminal_refs.append(ref)
        return tuple(
            sorted(
                terminal_refs,
                key=lambda ref: (
                    ref.artifact_type,
                    ref.path,
                    ref.artifact_hash,
                ),
            )
        )

    def preflight_runtime_observation_retry(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_id: str,
    ) -> tuple[
        ResumeDecision,
        ArtifactRef,
        ArtifactRef,
        ArtifactRef,
        dict[str, Any],
    ]:
        """Compatibility wrapper for observation-retry checkpoint admission."""

        return self._preflight_runtime_observation_retry_facts(
            checkpoint_ref,
            resume_id=resume_id,
        )

    def _preflight_runtime_observation_retry_facts(
        self,
        checkpoint_ref: ArtifactRef,
        *,
        resume_id: str,
    ) -> tuple[
        ResumeDecision,
        ArtifactRef,
        ArtifactRef,
        ArtifactRef,
        dict[str, Any],
    ]:
        """Read-validate one bounded observation retry without writing."""

        if not resume_id:
            raise ValueError(
                "Runtime observation retry requires the exact prior resume_id."
            )
        checkpoint_envelope = self.artifact_store.read(checkpoint_ref)
        checkpoint = checkpoint_envelope["data"]
        blockers = self._checkpoint_lineage_blockers(
            checkpoint_ref,
            checkpoint_envelope,
            allow_prior_runtime_state=True,
        )
        freshness = self._checkpoint_freshness(
            checkpoint_ref,
            checkpoint_envelope,
            additional_exact_revision_types=frozenset({"repair_plan"}),
        )
        if not freshness.is_fresh:
            blockers.append(
                {
                    "code": "OBSERVATION_RETRY_STALE_CHECKPOINT",
                    "message": (
                        "Checkpoint inputs changed after the prior Runtime "
                        "observation."
                    ),
                }
            )

        accepted_reports: list[tuple[ArtifactRef, dict[str, Any], dict[str, Any]]] = []
        for report_ref in self.artifact_store.list_feature_artifact_revisions(
            checkpoint_ref.feature_id,
            "resume_report",
        ):
            envelope = self.artifact_store.read(report_ref)
            report = envelope["data"]
            if (
                report.get("checkpoint_id") == checkpoint.get("checkpoint_id")
                and report.get("allowed") is True
            ):
                accepted_reports.append((report_ref, envelope, report))
        if len(accepted_reports) != 1:
            raise ValueError(
                "Runtime observation retry requires exactly one accepted "
                "ResumeReport for the checkpoint."
            )
        report_ref, report_envelope, report = accepted_reports[0]
        if report.get("resume_id") != resume_id:
            raise ValueError(
                "Runtime observation retry resume_id does not match ResumeReport."
            )
        report_upstream = report_envelope.get("inputs", {}).get(
            "upstream_artifacts",
            [],
        )
        restored_state = report.get("applied_updates", {}).get("restored_state")
        if (
            len(report_upstream) != 1
            or report_upstream[0].get("artifact_type") != "workflow_checkpoint"
            or report_upstream[0].get("path") != checkpoint_ref.path
            or report_upstream[0].get("artifact_hash")
            != checkpoint_ref.artifact_hash
            or report.get("status") != "accepted"
            or report.get("run_id") != checkpoint.get("run_id")
            or report.get("resume_stage") != checkpoint.get("checkpoint_stage")
            or not isinstance(report.get("applied_updates"), dict)
            or not isinstance(restored_state, dict)
            or restored_state != checkpoint.get("state_snapshot")
        ):
            blockers.append(
                {
                    "code": "OBSERVATION_RETRY_RESUME_BINDING_INVALID",
                    "message": (
                        "Accepted ResumeReport does not reconstruct the exact "
                        "pre-Runtime state."
                    ),
                }
            )

        runtime_revisions: list[tuple[ArtifactRef, dict[str, Any]]] = []
        for runtime_ref in self.artifact_store.list_feature_artifact_revisions(
            checkpoint_ref.feature_id,
            "runtime_evidence",
        ):
            runtime = self.artifact_store.read(runtime_ref)["data"]
            if runtime.get("run_id") == checkpoint.get("run_id"):
                runtime_revisions.append((runtime_ref, runtime))
        latest_runtime_ref = self._latest_revision_ref(
            checkpoint_ref.feature_id,
            "runtime_evidence",
        )
        if latest_runtime_ref is None:
            raise ValueError(
                "Runtime observation retry requires prior RuntimeEvidence."
            )
        latest_runtime = self.artifact_store.read(latest_runtime_ref)["data"]
        runtime_owner = latest_runtime.get("failure_ownership")
        checkpoint_snapshot = checkpoint.get("state_snapshot") or {}
        checkpoint_artifacts = checkpoint_snapshot.get("artifacts") or {}
        checkpoint_hashes = checkpoint_snapshot.get("artifact_hashes") or {}
        build_ref = latest_runtime.get("build_artifact_ref") or {}
        if (
            latest_runtime.get("run_id") != checkpoint.get("run_id")
            or latest_runtime.get("verification_status") != "inconclusive"
            or latest_runtime.get("acceptance_verified") is not False
            or latest_runtime.get("claims_real_runtime_success") is not False
            or not isinstance(runtime_owner, dict)
            or runtime_owner.get("owner") != "observation"
            or runtime_owner.get("required_action") != "repair_runtime_observation"
            or runtime_owner.get("source_repair_allowed") is not False
            or not isinstance(build_ref, dict)
            or build_ref.get("path") != checkpoint_artifacts.get("build_result")
            or build_ref.get("artifact_hash")
            != checkpoint_hashes.get("build_result")
            or latest_runtime.get("build_artifact_hash")
            != checkpoint_hashes.get("build_result")
        ):
            blockers.append(
                {
                    "code": "OBSERVATION_RETRY_RUNTIME_EVIDENCE_INELIGIBLE",
                    "message": (
                        "Latest RuntimeEvidence is not an inconclusive, "
                        "observation-owned result bound to this Build."
                    ),
                }
            )

        latest_repair_ref = self._latest_revision_ref(
            checkpoint_ref.feature_id,
            "repair_plan",
        )
        if latest_repair_ref is None:
            raise ValueError(
                "Runtime observation retry requires a prior RepairPlan."
            )
        latest_repair_envelope = self.artifact_store.read(latest_repair_ref)
        latest_repair = latest_repair_envelope["data"]
        repair_context = latest_repair.get("repair_context")
        repair_upstreams = latest_repair_envelope.get("inputs", {}).get(
            "upstream_artifacts",
            [],
        )
        bound_runtime_upstreams = [
            item
            for item in repair_upstreams
            if isinstance(item, dict)
            and item.get("artifact_type") == "runtime_evidence"
        ]
        retry_budget = latest_repair.get("retry_budget")
        retry_budget_remaining = latest_repair.get("retry_budget_remaining")
        retry_number = len(runtime_revisions)
        if (
            latest_repair.get("failure_source") != "runtime"
            or latest_repair.get("requires_new_evidence") is not True
            or latest_repair.get("allowed_next_stage") is not None
            or latest_repair.get("retry_exhausted") is not False
            or not isinstance(repair_context, dict)
            or repair_context.get("failure_owner") != "observation"
            or repair_context.get("repair_decision")
            != "repair_runtime_observation"
            or repair_context.get("source_repair_allowed") is not False
            or bound_runtime_upstreams
            != [
                {
                    "artifact_type": "runtime_evidence",
                    "path": latest_runtime_ref.path,
                    "artifact_hash": latest_runtime_ref.artifact_hash,
                    "require_latest": True,
                }
            ]
            or isinstance(retry_budget, bool)
            or not isinstance(retry_budget, int)
            or retry_budget < 1
            or isinstance(retry_budget_remaining, bool)
            or not isinstance(retry_budget_remaining, int)
            or retry_budget_remaining < 1
        ):
            blockers.append(
                {
                    "code": "OBSERVATION_RETRY_REPAIR_PLAN_INELIGIBLE",
                    "message": (
                        "Latest RepairPlan does not authorize new observation "
                        "evidence without source repair."
                    ),
                }
            )
        if isinstance(retry_budget, int) and retry_number > retry_budget:
            blockers.append(
                {
                    "code": "OBSERVATION_RETRY_BUDGET_EXHAUSTED",
                    "message": "Runtime observation retry budget is exhausted.",
                }
            )
        effective_retry_budget_remaining = (
            min(
                retry_budget_remaining,
                retry_budget - retry_number + 1,
            )
            if isinstance(retry_budget, int)
            and isinstance(retry_budget_remaining, int)
            else 0
        )

        for artifact_type in ("eval_report", "workflow_summary"):
            for artifact_ref in self.artifact_store.list_feature_artifact_revisions(
                checkpoint_ref.feature_id,
                artifact_type,
            ):
                artifact = self.artifact_store.read(artifact_ref)["data"]
                if artifact.get("run_id") == checkpoint.get("run_id"):
                    blockers.append(
                        {
                            "code": "OBSERVATION_RETRY_TERMINAL_STATE_PRESENT",
                            "message": (
                                f"{artifact_type} already exists for this "
                                "Runtime handoff."
                            ),
                        }
                    )
                    break

        if blockers:
            raise ValueError(
                "Runtime observation retry was blocked: "
                + ", ".join(str(item.get("code")) for item in blockers)
            )
        decision = ResumeDecision(
            status="accepted",
            allowed=True,
            reason=(
                "Observation-owned Runtime result is ready for one bounded "
                "evidence retry."
            ),
            applied_updates=dict(report["applied_updates"]),
        )
        retry_binding = {
            "retry_number": retry_number,
            "retry_budget": retry_budget,
            "retry_budget_remaining_before_attempt": (
                effective_retry_budget_remaining
            ),
            "prior_runtime_evidence_ref": latest_runtime_ref.path,
            "prior_runtime_evidence_hash": latest_runtime_ref.artifact_hash,
            "prior_repair_plan_ref": latest_repair_ref.path,
            "prior_repair_plan_hash": latest_repair_ref.artifact_hash,
            "resume_report_ref": report_ref.path,
            "resume_report_hash": report_ref.artifact_hash,
            "new_checkpoint_would_be_created": False,
            "new_resume_report_would_be_created": False,
        }
        return (
            decision,
            report_ref,
            latest_runtime_ref,
            latest_repair_ref,
            retry_binding,
        )

    @staticmethod
    def langgraph_memory_checkpointer() -> MemorySaver:
        """Return the available local LangGraph memory checkpointer."""

        return MemorySaver()

    @staticmethod
    def _state_snapshot(state: ModAgentState) -> dict[str, Any]:
        # This is the historical durable checkpoint projection, not a generic
        # state clone. edit_authority_review_result remains intentionally absent
        # from persisted checkpoint payloads.
        return {
            "run_id": state["run_id"],
            "request_id": state["request_id"],
            "feature_id": state["feature_id"],
            "user_request": state["user_request"],
            "target_domain": state["target_domain"],
            "target_mod": state["target_mod"],
            "execution_path": state["execution_path"],
            "risk_level": state["risk_level"],
            "current_stage": state["current_stage"],
            "next_stage": state["next_stage"],
            "artifacts": dict(state["artifacts"]),
            "artifact_hashes": dict(state["artifact_hashes"]),
            "approvals": dict(state["approvals"]),
            "blockers": list(state["blockers"]),
            "warnings": list(state["warnings"]),
            "evidence_summary": dict(state["evidence_summary"]),
            "review_summary": dict(state["review_summary"]),
            "build_summary": dict(state["build_summary"]),
            "runtime_summary": dict(state["runtime_summary"]),
            "trace_events": list(state["trace_events"]),
            "retry_count": state["retry_count"],
        }

    def _resume_binding_from_state(self, state: ModAgentState) -> dict[str, Any]:
        binding: dict[str, Any] = {
            "patch_proposal_ref": None,
            "patch_proposal_hash": None,
            "proposal_review_ref": None,
            "proposal_review_hash": None,
            "diff_hash": None,
            "feature_spec_ref": None,
            "feature_spec_hash": None,
            "build_result_ref": None,
            "build_result_hash": None,
            "next_graph_node": None,
            "edit_authority_request_ref": None,
            "edit_authority_request_artifact_hash": None,
            "edit_authority_request_hash": None,
            "repair_planning_authority_request_ref": None,
            "repair_planning_authority_request_artifact_hash": None,
            "repair_planning_authority_request_hash": None,
        }
        if "patch_proposal" in state["artifacts"]:
            proposal_ref = self._artifact_ref_from_state(state, "patch_proposal")
            proposal_payload = self.artifact_store.read(proposal_ref)["data"]
            binding["patch_proposal_ref"] = proposal_ref.path
            binding["patch_proposal_hash"] = proposal_ref.artifact_hash
            binding["diff_hash"] = self._diff_hash_from_patch_proposal(
                proposal_payload,
            )
        execution_diff_hash = self._persisted_execution_diff_hash(state)
        if execution_diff_hash is not None:
            binding["diff_hash"] = execution_diff_hash
        if "proposal_review" in state["artifacts"]:
            review_ref = self._artifact_ref_from_state(state, "proposal_review")
            binding["proposal_review_ref"] = review_ref.path
            binding["proposal_review_hash"] = review_ref.artifact_hash
        if "feature_spec" in state["artifacts"]:
            feature_ref = self._artifact_ref_from_state(state, "feature_spec")
            binding["feature_spec_ref"] = feature_ref.path
            binding["feature_spec_hash"] = feature_ref.artifact_hash
        if "build_result" in state["artifacts"]:
            build_ref = self._artifact_ref_from_state(state, "build_result")
            binding["build_result_ref"] = build_ref.path
            binding["build_result_hash"] = build_ref.artifact_hash
        if "edit_authority_request" in state["artifacts"]:
            request_ref = self._artifact_ref_from_state(
                state,
                "edit_authority_request",
            )
            request_payload = self.artifact_store.read(request_ref)["data"]
            binding["edit_authority_request_ref"] = request_ref.path
            binding["edit_authority_request_artifact_hash"] = (
                request_ref.artifact_hash
            )
            binding["edit_authority_request_hash"] = request_payload.get(
                "request_hash"
            )
        if "repair_planning_request" in state["artifacts"]:
            request_ref = self._artifact_ref_from_state(
                state,
                "repair_planning_request",
            )
            request_payload = self.artifact_store.read(request_ref)["data"]
            binding["repair_planning_authority_request_ref"] = (
                request_ref.path
            )
            binding[
                "repair_planning_authority_request_artifact_hash"
            ] = request_ref.artifact_hash
            binding["repair_planning_authority_request_hash"] = (
                request_payload.get("request_hash")
            )
        if state["current_stage"] == "build":
            binding["next_graph_node"] = "collect_runtime_evidence"
        return binding

    def _persisted_execution_diff_hash(
        self,
        state: ModAgentState | dict[str, Any],
    ) -> str | None:
        execution_diff_hash: str | None = None
        for artifact_type in ("dry_run_result", "isolated_apply_result"):
            if artifact_type not in state["artifacts"]:
                continue
            artifact_ref = self._artifact_ref_from_state(state, artifact_type)
            value = self.artifact_store.read(artifact_ref)["data"].get("diff_hash")
            if isinstance(value, str) and value:
                execution_diff_hash = value
        return execution_diff_hash

    def _checkpoint_execution_diff_binding_blockers(
        self,
        checkpoint: dict[str, Any],
    ) -> list[dict[str, Any]]:
        state = checkpoint["state_snapshot"]
        dry_run_data: dict[str, Any] = {}
        apply_data: dict[str, Any] = {}
        if "dry_run_result" in state["artifacts"]:
            dry_run_data = self.artifact_store.read(
                self._artifact_ref_from_state(state, "dry_run_result")
            )["data"]
        if "isolated_apply_result" in state["artifacts"]:
            apply_data = self.artifact_store.read(
                self._artifact_ref_from_state(state, "isolated_apply_result")
            )["data"]
        execution_diff_hash = self._persisted_execution_diff_hash(state)
        blockers: list[dict[str, Any]] = []
        if apply_data.get("actual_write_performed") is True:
            apply_diff_hash = apply_data.get("diff_hash")
            if (
                not isinstance(apply_diff_hash, str)
                or not apply_diff_hash
                or dry_run_data.get("diff_hash") != apply_diff_hash
                or apply_data.get("approved_diff_hash") != apply_diff_hash
                or apply_data.get("exact_diff_approved") is not True
            ):
                blockers.append(
                    {
                        "code": "BUILD_CHECKPOINT_EXACT_DIFF_IDENTITY_INVALID",
                        "message": (
                            "Build checkpoint DryRun/Apply exact diff identity is "
                            "missing or inconsistent."
                        ),
                    }
                )
        resume_binding = checkpoint.get("resume_binding")
        if (
            execution_diff_hash is not None
            and (
                not isinstance(resume_binding, dict)
                or resume_binding.get("diff_hash") != execution_diff_hash
            )
        ):
            blockers.append(
                {
                    "code": "CHECKPOINT_EXACT_DIFF_BINDING_MISMATCH",
                    "message": (
                        "Checkpoint resume diff hash disagrees with its "
                        "persisted DryRun/Apply execution identity."
                    ),
                }
            )
        return blockers

    def _durable_consumption_state(
        self,
        feature_id: str,
        *,
        checkpoint_id: str,
        resume_id: str,
    ) -> tuple[bool, bool]:
        consumed = False
        seen_resume_id = False
        for ref in self.artifact_store.list_feature_artifact_revisions(
            feature_id,
            "resume_report",
        ):
            report = self.artifact_store.read(ref)["data"]
            if report.get("allowed") is not True:
                continue
            consumed = consumed or report.get("checkpoint_id") == checkpoint_id
            seen_resume_id = seen_resume_id or report.get("resume_id") == resume_id
        return consumed, seen_resume_id

    def _latest_revision_ref(
        self,
        feature_id: str,
        artifact_type: str,
    ) -> ArtifactRef | None:
        """Resolve the latest pointer back to its immutable revision."""

        latest_refs = [
            ref
            for ref in self.artifact_store.list_feature_artifacts(feature_id)
            if ref.artifact_type == artifact_type
        ]
        if not latest_refs:
            return None
        if len(latest_refs) != 1:
            raise ArtifactStoreError(
                f"Multiple latest pointers found for {artifact_type}."
            )
        matching_revisions = [
            ref
            for ref in self.artifact_store.list_feature_artifact_revisions(
                feature_id,
                artifact_type,
            )
            if ref.artifact_hash == latest_refs[0].artifact_hash
        ]
        if len(matching_revisions) != 1:
            raise ArtifactStoreError(
                f"Latest {artifact_type} pointer does not resolve uniquely."
            )
        return matching_revisions[0]

    def _build_checkpoint_exact_lineage_blockers(
        self,
        state: ModAgentState | dict[str, Any],
        *,
        exact_revision_types: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Validate approved exact roles against one Build's immutable ancestry."""

        stage = state.get("current_stage")
        if stage != "build":
            if exact_revision_types:
                return [
                    {
                        "code": "CHECKPOINT_EXACT_LINEAGE_STAGE_INVALID",
                        "message": (
                            "Exact revision checkpoint roles are supported only "
                            "at the Build handoff."
                        ),
                    }
                ]
            return []

        feature_id = state.get("feature_id") or state.get("request_id")
        if (
            not isinstance(feature_id, str)
            or not feature_id
            or not isinstance(state.get("run_id"), str)
            or not state.get("run_id")
            or not isinstance(state.get("request_id"), str)
            or not state.get("request_id")
        ):
            return [
                {
                    "code": "BUILD_CHECKPOINT_IDENTITY_INVALID",
                    "message": (
                        "Build checkpoint Feature, Run, and Request identities "
                        "must be non-empty."
                    ),
                }
            ]

        artifacts = state.get("artifacts")
        artifact_hashes = state.get("artifact_hashes")
        if not isinstance(artifacts, dict) or not isinstance(artifact_hashes, dict):
            return [
                {
                    "code": "BUILD_CHECKPOINT_STATE_INVALID",
                    "message": "Build checkpoint Artifact bindings are incomplete.",
                }
            ]
        if "build_result" not in artifacts or "build_result" not in artifact_hashes:
            return [
                {
                    "code": "BUILD_CHECKPOINT_RESULT_MISSING",
                    "message": "Build checkpoint is missing its BuildResult binding.",
                }
            ]

        build_ref = ArtifactRef(
            feature_id=feature_id,
            artifact_type="build_result",
            path=str(artifacts["build_result"]),
            artifact_hash=str(artifact_hashes["build_result"]),
        )
        blockers: list[dict[str, Any]] = []
        ancestry, ancestry_blockers = self._build_upstream_ancestry(build_ref)
        blockers.extend(ancestry_blockers)
        lineage_roles = set(exact_revision_types)
        lineage_roles.update(
            artifact_type
            for artifact_type in ("dry_run_result", "isolated_apply_result")
            if artifact_type in artifacts
        )
        for artifact_type in sorted(lineage_roles):
            path = artifacts.get(artifact_type)
            artifact_hash = artifact_hashes.get(artifact_type)
            if not isinstance(path, str) or not isinstance(artifact_hash, str):
                blockers.append(
                    {
                        "code": "CHECKPOINT_EXACT_ROLE_MISSING",
                        "message": (
                            "Approved exact revision role is missing from state: "
                            f"{artifact_type}."
                        ),
                    }
                )
                continue
            state_ref = ArtifactRef(
                feature_id=feature_id,
                artifact_type=artifact_type,
                path=path,
                artifact_hash=artifact_hash,
            )
            role_refs = ancestry.get(artifact_type, {})
            expected_identity = (state_ref.path, state_ref.artifact_hash)
            nearest_depth = min(role_refs.values()) if role_refs else None
            nearest_refs = {
                identity
                for identity, depth in role_refs.items()
                if depth == nearest_depth
            }
            if nearest_refs != {expected_identity}:
                role_kind = (
                    "exact revision role"
                    if artifact_type in exact_revision_types
                    else "execution role"
                )
                blockers.append(
                    {
                        "code": "CHECKPOINT_EXACT_ROLE_NOT_BUILD_ANCESTOR",
                        "message": (
                            f"Build checkpoint {role_kind} is not the unique nearest "
                            f"{artifact_type} ancestor of this BuildResult."
                        ),
                    }
                )

        return blockers

    @staticmethod
    def _build_checkpoint_exact_role_policy_blockers(
        exact_revision_types: frozenset[str],
    ) -> list[dict[str, Any]]:
        if not exact_revision_types:
            return []
        unsupported = sorted(
            set(exact_revision_types) - BUILD_CHECKPOINT_EXACT_REVISION_TYPES
        )
        missing = sorted(
            BUILD_CHECKPOINT_EXACT_REVISION_TYPES - set(exact_revision_types)
        )
        blockers: list[dict[str, Any]] = []
        if unsupported:
            blockers.append(
                {
                    "code": "BUILD_CHECKPOINT_EXACT_ROLE_UNSUPPORTED",
                    "message": (
                        "Build checkpoint exact revision roles are unsupported: "
                        f"{unsupported}"
                    ),
                }
            )
        if missing:
            blockers.append(
                {
                    "code": "BUILD_CHECKPOINT_EXACT_ROLE_SET_INCOMPLETE",
                    "message": (
                        "Build checkpoint exact revision roles are incomplete: "
                        f"{missing}"
                    ),
                }
            )
        return blockers

    def _build_upstream_ancestry(
        self,
        build_ref: ArtifactRef,
    ) -> tuple[dict[str, dict[tuple[str, str], int]], list[dict[str, Any]]]:
        ancestry: dict[str, dict[tuple[str, str], int]] = {}
        blockers: list[dict[str, Any]] = []
        pending = [(build_ref, 0)]
        visited_depths: dict[tuple[str, str, str], int] = {}
        while pending:
            current, current_depth = pending.pop()
            identity = (
                current.artifact_type,
                current.path,
                current.artifact_hash,
            )
            prior_depth = visited_depths.get(identity)
            if prior_depth is not None and prior_depth <= current_depth:
                continue
            visited_depths[identity] = current_depth
            try:
                envelope = self.artifact_store.read(current)
            except ArtifactStoreError as exc:
                blockers.append(
                    {
                        "code": "BUILD_LINEAGE_ARTIFACT_INVALID",
                        "message": (
                            "Build lineage Artifact content validation failed: "
                            f"{current.artifact_type}: {exc}"
                        ),
                    }
                )
                continue
            inputs = envelope.get("inputs", {}).get("upstream_artifacts", [])
            if not isinstance(inputs, list):
                blockers.append(
                    {
                        "code": "BUILD_LINEAGE_INPUTS_INVALID",
                        "message": (
                            "Build lineage upstream representation is invalid for "
                            f"{current.artifact_type}."
                        ),
                    }
                )
                continue
            for item in inputs:
                if not isinstance(item, dict):
                    blockers.append(
                        {
                            "code": "BUILD_LINEAGE_INPUT_INVALID",
                            "message": "Build lineage contains a malformed upstream.",
                        }
                    )
                    continue
                artifact_type = item.get("artifact_type")
                path = item.get("path")
                artifact_hash = item.get("artifact_hash")
                if (
                    not isinstance(artifact_type, str)
                    or not isinstance(path, str)
                    or not isinstance(artifact_hash, str)
                ):
                    blockers.append(
                        {
                            "code": "BUILD_LINEAGE_INPUT_INVALID",
                            "message": "Build lineage upstream path/hash is incomplete.",
                        }
                    )
                    continue
                upstream = ArtifactRef(
                    feature_id=build_ref.feature_id,
                    artifact_type=artifact_type,
                    path=path,
                    artifact_hash=artifact_hash,
                )
                upstream_depth = current_depth + 1
                role_depths = ancestry.setdefault(artifact_type, {})
                upstream_identity = (path, artifact_hash)
                role_depths[upstream_identity] = min(
                    upstream_depth,
                    role_depths.get(upstream_identity, upstream_depth),
                )
                pending.append((upstream, upstream_depth))
        return ancestry, blockers

    def _checkpoint_lineage_blockers(
        self,
        checkpoint_ref: ArtifactRef,
        checkpoint_envelope: dict[str, Any],
        *,
        allow_prior_runtime_state: bool = False,
        require_current_build_output: bool = True,
    ) -> list[dict[str, Any]]:
        checkpoint = checkpoint_envelope["data"]
        snapshot = checkpoint.get("state_snapshot")
        artifact_hashes = checkpoint.get("artifact_hashes")
        artifact_refs = checkpoint.get("artifact_refs")
        upstream = checkpoint_envelope.get("inputs", {}).get("upstream_artifacts", [])
        blockers: list[dict[str, Any]] = []
        if (
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("artifacts"), dict)
            or not isinstance(snapshot.get("artifact_hashes"), dict)
            or not isinstance(artifact_hashes, dict)
            or not isinstance(artifact_refs, list)
            or not isinstance(upstream, list)
        ):
            return [
                {
                    "code": "CHECKPOINT_LINEAGE_INVALID",
                    "message": "Checkpoint state and Artifact bindings are incomplete.",
                }
            ]
        upstream_by_type = {
            item.get("artifact_type"): item
            for item in upstream
            if isinstance(item, dict) and isinstance(item.get("artifact_type"), str)
        }
        snapshot_artifacts = snapshot["artifacts"]
        snapshot_hashes = snapshot["artifact_hashes"]
        actual_exact_types = {
            str(item.get("artifact_type"))
            for item in upstream
            if isinstance(item, dict) and item.get("require_latest") is False
        }
        if (
            checkpoint.get("run_id") != snapshot.get("run_id")
            or checkpoint.get("request_id") != snapshot.get("request_id")
            or checkpoint.get("feature_id") != snapshot.get("feature_id")
            or checkpoint_ref.feature_id != checkpoint.get("feature_id")
            or checkpoint.get("checkpoint_stage") != snapshot.get("current_stage")
            or set(artifact_refs) != set(snapshot_artifacts.values())
            or artifact_hashes != snapshot_hashes
            or set(upstream_by_type) != set(snapshot_artifacts)
        ):
            blockers.append(
                {
                    "code": "CHECKPOINT_LINEAGE_MISMATCH",
                    "message": "Checkpoint identity, state, or Artifact bindings disagree.",
                }
            )
            return blockers
        for artifact_type, path in snapshot_artifacts.items():
            item = upstream_by_type.get(artifact_type)
            if (
                item is None
                or item.get("path") != path
                or item.get("artifact_hash") != snapshot_hashes.get(artifact_type)
            ):
                blockers.append(
                    {
                        "code": "CHECKPOINT_ARTIFACT_BINDING_MISMATCH",
                        "message": (
                            f"Checkpoint Artifact binding disagrees for {artifact_type}."
                        ),
                    }
                )
        if checkpoint.get("checkpoint_stage") == "build":
            exact_role_policy_blockers = (
                self._build_checkpoint_exact_role_policy_blockers(
                    frozenset(actual_exact_types)
                )
            )
            blockers.extend(exact_role_policy_blockers)
            if actual_exact_types and not exact_role_policy_blockers:
                blockers.extend(
                    self._build_checkpoint_exact_lineage_blockers(
                        snapshot,
                        exact_revision_types=frozenset(actual_exact_types),
                    )
                )
                blockers.extend(
                    self._checkpoint_execution_diff_binding_blockers(checkpoint)
                )
            blockers.extend(
                self._build_handoff_blockers(
                    checkpoint,
                    allow_prior_runtime_state=allow_prior_runtime_state,
                    require_current_build_output=require_current_build_output,
                )
            )
        return blockers

    def _build_handoff_blockers(
        self,
        checkpoint: dict[str, Any],
        *,
        allow_prior_runtime_state: bool = False,
        require_current_build_output: bool = True,
    ) -> list[dict[str, Any]]:
        snapshot = checkpoint["state_snapshot"]
        required = {"feature_spec", "build_result"}
        missing = sorted(required - set(snapshot["artifacts"]))
        if missing:
            return [
                {
                    "code": "BUILD_HANDOFF_ARTIFACT_MISSING",
                    "message": f"Build handoff is missing required Artifacts: {missing}",
                }
            ]
        binding = checkpoint.get("resume_binding") or {}
        if (
            binding.get("feature_spec_ref") != snapshot["artifacts"]["feature_spec"]
            or binding.get("feature_spec_hash")
            != snapshot["artifact_hashes"]["feature_spec"]
            or binding.get("build_result_ref") != snapshot["artifacts"]["build_result"]
            or binding.get("build_result_hash")
            != snapshot["artifact_hashes"]["build_result"]
            or binding.get("next_graph_node") != "collect_runtime_evidence"
        ):
            return [
                {
                    "code": "BUILD_HANDOFF_BINDING_MISMATCH",
                    "message": "Build handoff does not bind the expected Runtime graph entry.",
                }
            ]
        build_ref = ArtifactRef(
            feature_id=checkpoint["feature_id"],
            artifact_type="build_result",
            path=snapshot["artifacts"]["build_result"],
            artifact_hash=snapshot["artifact_hashes"]["build_result"],
        )
        build = self.artifact_store.read(build_ref)["data"]
        if (
            build.get("status") != "success"
            or build.get("claims_real_build_success") is not True
        ):
            return [
                {
                    "code": "BUILD_HANDOFF_NOT_SUCCESSFUL",
                    "message": "Runtime continuation requires a successful real Build.",
                }
            ]
        verified_artifacts = build.get("verified_artifacts")
        if (
            not isinstance(verified_artifacts, list)
            or not verified_artifacts
            or (build.get("artifact_verification") or {}).get("status") != "verified"
        ):
            return [
                {
                    "code": "BUILD_OUTPUT_VERIFICATION_MISSING",
                    "message": (
                        "Runtime continuation requires at least one verified "
                        "physical Build output."
                    ),
                }
            ]
        blockers: list[dict[str, Any]] = []
        if not allow_prior_runtime_state:
            for runtime_ref in self.artifact_store.list_feature_artifact_revisions(
                checkpoint["feature_id"],
                "runtime_evidence",
            ):
                runtime = self.artifact_store.read(runtime_ref)["data"]
                if runtime.get("run_id") == checkpoint.get("run_id"):
                    blockers.append(
                        {
                            "code": "PRIOR_RUNTIME_STATE_PRESENT",
                            "message": (
                                "Runtime evidence already exists for this "
                                "handoff run; automatic repetition is blocked."
                            ),
                        }
                    )
                    return blockers
        for artifact in verified_artifacts:
            if not isinstance(artifact, dict):
                blockers.append(
                    {
                        "code": "BUILD_OUTPUT_BINDING_INVALID",
                        "message": "Build output binding is not structured.",
                    }
                )
                continue
            resolved_path = artifact.get("resolved_path")
            expected_hash = artifact.get("sha256")
            if not isinstance(resolved_path, str) or not isinstance(expected_hash, str):
                blockers.append(
                    {
                        "code": "BUILD_OUTPUT_BINDING_INVALID",
                        "message": "Build output path or SHA256 is missing.",
                    }
                )
                continue
            if require_current_build_output:
                path = Path(resolved_path)
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
                ):
                    blockers.append(
                        {
                            "code": "BUILD_OUTPUT_BYTES_CHANGED",
                            "message": (
                                f"Verified Build output changed: {resolved_path}"
                            ),
                        }
                    )

        return blockers

    @staticmethod
    def _diff_hash_from_patch_proposal(proposal_payload: dict[str, Any]) -> str:
        serialized = json.dumps(
            proposal_payload["diff_intent"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _artifact_refs_from_state(state: ModAgentState) -> list[ArtifactRef]:
        refs = []
        for artifact_type, path in state["artifacts"].items():
            artifact_hash = state["artifact_hashes"].get(artifact_type)
            if artifact_hash is None:
                raise ArtifactStoreError(
                    f"Missing artifact hash for state artifact: {artifact_type}"
                )
            refs.append(
                ArtifactRef(
                    feature_id=state["feature_id"] or state["request_id"],
                    artifact_type=artifact_type,
                    path=path,
                    artifact_hash=artifact_hash,
                )
            )
        return refs

    @staticmethod
    def _artifact_ref_from_state(
        state: ModAgentState,
        artifact_type: str,
    ) -> ArtifactRef:
        artifact_hash = state["artifact_hashes"].get(artifact_type)
        if artifact_hash is None:
            raise ArtifactStoreError(
                f"Missing artifact hash for state artifact: {artifact_type}"
            )
        return ArtifactRef(
            feature_id=state["feature_id"] or state["request_id"],
            artifact_type=artifact_type,
            path=state["artifacts"][artifact_type],
            artifact_hash=artifact_hash,
        )
