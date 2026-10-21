"""Structured planning contracts for Phase E."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.mod_agent.workflow.state import ExecutionPath, RiskLevel


@dataclass(frozen=True)
class Uncertainty:
    """A planning uncertainty that may require blocking or human input."""

    uncertainty_id: str
    severity: str
    question: str
    allowed_resolution: tuple[str, ...] = ("human_answer", "artifact_context")
    default_action: str = "block"

    def to_dict(self) -> dict[str, Any]:
        return {
            "uncertainty_id": self.uncertainty_id,
            "severity": self.severity,
            "question": self.question,
            "allowed_resolution": list(self.allowed_resolution),
            "default_action": self.default_action,
        }


@dataclass(frozen=True)
class RequestPlan:
    """Route decision created from the user request."""

    feature_id: str
    user_request: str
    route: ExecutionPath
    risk_level: RiskLevel
    target_mod: str | None
    requires_hook_evidence: bool
    requires_runtime_evidence: bool
    hook_requirement: dict[str, Any] = field(
        default_factory=lambda: {"required": False, "reason": "not_required"}
    )
    uncertainties: tuple[Uncertainty, ...] = ()
    blocked: bool = False
    request_interpretation: dict[str, Any] = field(default_factory=dict)

    def to_artifact_payload(self) -> dict[str, Any]:
        return {
            "node_name": "plan_request",
            "stage": "request_planning",
            "message": (
                "Request planning completed."
                if not self.blocked
                else "Request planning blocked on unresolved uncertainty."
            ),
            "feature_id": self.feature_id,
            "user_request": self.user_request,
            "route": self.route,
            "risk_level": self.risk_level,
            "target_mod": self.target_mod,
            "requires_hook_evidence": self.requires_hook_evidence,
            "requires_runtime_evidence": self.requires_runtime_evidence,
            "hook_requirement": dict(self.hook_requirement),
            "uncertainties": [uncertainty.to_dict() for uncertainty in self.uncertainties],
            **({"request_interpretation": dict(self.request_interpretation)}
               if self.request_interpretation else {}),
            "blocked": self.blocked,
        }


@dataclass(frozen=True)
class FeatureSpec:
    """Verifiable feature contract derived from a request plan."""

    feature_id: str
    user_request: str
    target_behavior: str
    non_goals: tuple[str, ...]
    target_mod: str | None
    risk_level: RiskLevel
    acceptance_criteria: tuple[str, ...]
    evidence_requirements: tuple[str, ...]
    runtime_verification_plan: tuple[str, ...]
    hook_requirement: dict[str, Any] = field(
        default_factory=lambda: {"required": False, "reason": "not_required"}
    )
    blocked: bool = False
    blockers: tuple[dict[str, Any], ...] = ()
    request_interpretation: dict[str, Any] = field(default_factory=dict)

    def to_artifact_payload(self) -> dict[str, Any]:
        return {
            "node_name": "draft_feature_spec",
            "stage": "feature_spec",
            "message": (
                "Feature spec drafted."
                if not self.blocked
                else "Feature spec blocked before implementation planning."
            ),
            "feature_id": self.feature_id,
            "user_request": self.user_request,
            "target_behavior": self.target_behavior,
            "non_goals": list(self.non_goals),
            "target_mod": self.target_mod,
            "risk_level": self.risk_level,
            "acceptance_criteria": list(self.acceptance_criteria),
            "evidence_requirements": list(self.evidence_requirements),
            "runtime_verification_plan": list(self.runtime_verification_plan),
            "hook_requirement": dict(self.hook_requirement),
            **({"request_interpretation": dict(self.request_interpretation)}
               if self.request_interpretation else {}),
            "blocked": self.blocked,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ModPlanningBrief:
    """Negotiated planning handoff before feature specification."""

    feature_id: str
    decision: str
    user_intent: dict[str, Any]
    request_type: dict[str, Any]
    knowledge_query_plan: dict[str, Any]
    retrieved_experience: dict[str, Any]
    planning_options: tuple[dict[str, Any], ...]
    selected_plan: dict[str, Any]
    handoff: dict[str, Any]
    verification_boundary: tuple[str, ...]
    next_action: str
    blockers: tuple[dict[str, Any], ...] = ()

    @property
    def blocked(self) -> bool:
        return self.decision in {"needs_clarification", "needs_evidence", "blocked"}

    def to_artifact_payload(self) -> dict[str, Any]:
        return {
            "node_name": "build_mod_planning_brief",
            "stage": "planning_brief",
            "message": (
                "Planning brief is ready for feature specification."
                if not self.blocked
                else "Planning brief needs clarification or evidence before handoff."
            ),
            "feature_id": self.feature_id,
            "decision": self.decision,
            "user_intent": self.user_intent,
            "request_type": self.request_type,
            "knowledge_query_plan": self.knowledge_query_plan,
            "retrieved_experience": self.retrieved_experience,
            "planning_options": list(self.planning_options),
            "selected_plan": self.selected_plan,
            "handoff": self.handoff,
            "verification_boundary": list(self.verification_boundary),
            "next_action": self.next_action,
            "blocked": self.blocked,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ImplementationPlan:
    """Evidence-bound plan for how code should change, without patch content."""

    feature_id: str
    target_files: tuple[str, ...]
    target_symbols: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    proposed_changes: tuple[dict[str, Any], ...]
    side_effect_boundaries: tuple[str, ...]
    test_plan: tuple[str, ...]
    rollback_plan: tuple[str, ...]
    open_questions: tuple[str, ...] = ()
    blocking: bool = False
    blockers: tuple[dict[str, Any], ...] = ()
    evidence_source_paths: tuple[str, ...] = ()
    binding_registry: dict[str, Any] | None = None
    verification_questions: tuple[str, ...] = ()
    constraint_context: dict[str, Any] | None = None

    def to_artifact_payload(self) -> dict[str, Any]:
        return {
            "node_name": "create_implementation_plan",
            "stage": "implementation_plan",
            "message": (
                "Implementation plan is evidence-bound."
                if not self.blocking
                else "Implementation planning blocked by evidence or open questions."
            ),
            "feature_id": self.feature_id,
            "evidence_source_paths": list(self.evidence_source_paths),
            "target_files": list(self.target_files),
            "target_symbols": list(self.target_symbols),
            "evidence_refs": list(self.evidence_refs),
            "binding_registry": dict(self.binding_registry or {}),
            **({"verification_questions": list(self.verification_questions)}
               if self.verification_questions else {}),
            "proposed_changes": list(self.proposed_changes),
            "side_effect_boundaries": list(self.side_effect_boundaries),
            **({"constraint_context": deepcopy(self.constraint_context)} if self.constraint_context is not None else {}),
            "test_plan": list(self.test_plan),
            "rollback_plan": list(self.rollback_plan),
            "open_questions": list(self.open_questions),
            "blocking": self.blocking,
            "blockers": list(self.blockers),
        }


def stable_feature_id(user_request: str) -> str:
    """Create a stable feature id from request text."""

    return "feature_" + hashlib.sha256(user_request.strip().encode("utf-8")).hexdigest()[:12]
