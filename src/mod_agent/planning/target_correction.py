"""One normalizer-feedback correction inside the existing Planning owner."""
from __future__ import annotations

from copy import deepcopy

from .implementation_plan import ImplementationPlanner
from .model_selection import ModelPlanningTargetSelector, PlanningTargetSelectionError


def correct_unsafe_target_selection(*, implementation, selector, feature_spec, evidence_pack,
                                    formal_requirement_coverage_required, allow_correction):
    """Reconsider only this exact initial blocker, using the same budgeted client.

    为什么要有它：正常化失败是可反馈给模型的业务结果，但不能创建新预算、放松正常化或扩展续接授权。
    """
    if (not allow_correction or not isinstance(selector, ModelPlanningTargetSelector)
            or selector.upstream_assessment is not None or selector.correction_context is not None
            or not implementation.blocking or len(implementation.blockers) != 1
            or implementation.blockers[0].get("code") != "EDIT_TARGET_NORMALIZATION_UNSAFE"):
        return implementation
    try:
        child = selector.for_target_correction(implementation.blockers[0])
        selected = child(deepcopy(feature_spec.to_artifact_payload()), deepcopy(evidence_pack))
    except PlanningTargetSelectionError as error:
        return ImplementationPlanner._blocked(feature_id=feature_spec.feature_id,
            blockers=(*implementation.blockers, error.as_blocker()),
            open_questions=(*implementation.open_questions, error.reason))
    return ImplementationPlanner().build(feature_spec=feature_spec, evidence_pack=evidence_pack,
        formal_requirement_coverage_required=formal_requirement_coverage_required,
        planning_target_evidence_ids=selected, constraint_policy=child.constraint_policy)
