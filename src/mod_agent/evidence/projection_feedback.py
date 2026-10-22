"""Project exact input-accounting diagnostics into bounded correction guidance.

Only existing obligation policy is restated; no model requirement is completed.
为什么要有它：遗漏项反馈应说明合法响应位置，不能替模型生成语义或放宽原校验。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .requirement_coverage import (
    RequirementCoverageProjectionError,
    _normalized_projection,
)


def multiple_accounting_conflict_feedback(
    completed_projection: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose each overlap using the unchanged validator, without repairing data."""
    result = deepcopy(dict(diagnostic))
    if diagnostic.get("failure_category") != "obligation_already_accounted_and_dispositioned":
        return result
    requirements = completed_projection["requirements"]
    dispositions = completed_projection["input_dispositions"]
    accounted = {index for row in requirements for index in row["input_obligation_indexes"]}
    positions = [index for index, row in enumerate(dispositions)
                 if row["input_obligation_index"] in accounted]
    if len(positions) < 2:
        return result
    conflicts = []
    for position in positions:
        # Only reorder a private list: the validator still sees every obligation,
        # requirement and disposition, with their original semantics and values.
        # 为什么要有它：首错校验器须逐项提供真实拒绝边界，反馈层不能另写放宽的判断或替模型删字段。
        reordered = list(dispositions)
        reordered[0], reordered[position] = reordered[position], reordered[0]
        try:
            _normalized_projection(
                raw_requirements=requirements,
                raw_relations=completed_projection["relations"],
                raw_dispositions=reordered,
                input_obligations=list(obligations),
            )
        except RequirementCoverageProjectionError as error:
            detail = deepcopy(error.diagnostic)
        else:
            raise ValueError("PROJECTION_ACCOUNTING_CONFLICT_NOT_REPRODUCED")
        if (detail.get("failure_category") != "obligation_already_accounted_and_dispositioned"
                or detail.get("input_obligation_index")
                != dispositions[position]["input_obligation_index"]):
            raise ValueError("PROJECTION_ACCOUNTING_CONFLICT_DIAGNOSTIC_MISMATCH")
        detail = _restore_disposition_paths(detail, position)
        detail["disposition_item_index"] = position
        detail["input_policy"] = deepcopy(dict(obligations[detail["input_obligation_index"] - 1]))
        conflicts.append(detail)
    result["accounting_conflicts"] = conflicts
    return result


def _restore_disposition_paths(value: Any, original_position: int) -> Any:
    if isinstance(value, dict):
        return {key: _restore_disposition_paths(item, original_position)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_disposition_paths(item, original_position) for item in value]
    prefix = "$.input_dispositions[0]"
    if isinstance(value, str) and value.startswith(prefix):
        return f"$.input_dispositions[{original_position}]" + value[len(prefix):]
    return value


def multiple_accounting_conflict_rules() -> list[str]:
    return [
        "For each conflict in rejection_diagnostic.accounting_conflicts, choose exactly one "
        "of that conflict's correction_directions with locally_applicable equal to true; "
        "do not combine both directions for the same conflict or invent a direction when none is applicable. "
        "为什么要有它：多个独立冲突必须在同一次纠错中逐项处理，合法选择仍由模型负责。",
        "All JSON paths refer to previous_projection before any removals. Apply the union of the selected "
        "remove_json_paths without shifting the original array coordinates, and preserve fields unrelated "
        "to that union. 为什么要有它：逐次删除会移动数组下标，误删其他义务会破坏原始语义。",
        "Evaluate the chosen directions together: every existing requirement must remain non-empty, "
        "every input obligation must remain accounted, and the complete corrected projection must pass "
        "the unchanged schema and semantic validator. Local applicability does not guarantee that all "
        "chosen removals are jointly valid. 为什么要有它：多个单独可行的删除可能共同清空同一需求，不能用局部条件冒充整份响应通过。",
    ]


def missing_obligation_feedback(
    obligations: Sequence[Mapping[str, Any]], diagnostic: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if diagnostic.get("failure_category") not in {
        "input_accounting_incomplete", "requirement_role_not_allowed"
    }:
        return []
    indexes = diagnostic.get("missing_input_obligation_indexes")
    if indexes is None and diagnostic.get("failure_category") == "requirement_role_not_allowed":
        return []
    if (not isinstance(indexes, list) or not indexes
            or any(type(index) is not int for index in indexes)
            or len(indexes) != len(set(indexes))):
        raise ValueError("PROJECTION_MISSING_INPUT_DIAGNOSTIC_INVALID")
    by_index = {row["input_obligation_index"]: row for row in obligations}
    if any(index not in by_index for index in indexes):
        raise ValueError("PROJECTION_MISSING_INPUT_DIAGNOSTIC_INVALID")
    result = []
    for index in indexes:
        obligation = deepcopy(dict(by_index[index]))
        locations = ["requirements"]
        instruction = (
            "Account for this input index in the complete corrected response. "
            "A requirements entry must reference this index and use one of this "
            "input's allowed_roles, retaining its original obligation. "
        )
        if obligation["allowed_dispositions"]:
            locations.append("input_dispositions")
            instruction += (
                "Alternatively, use an explicitly allowed input_dispositions value "
                "with a reason; do not account for the same input in both locations. "
            )
        else:
            instruction += (
                "Do not add an input_dispositions entry for this index: no "
                "disposition is allowed, so it must appear in requirements. "
            )
        instruction += (
            "This is input accounting only, not a finding that evidence supports "
            "the requirement or authority to edit source. "
            "为什么要有它：响应结构合法仍可能遗漏必需输入，补齐承接不等于证据已充分。"
        )
        result.append({"input": obligation, "response_locations": locations,
                       "instruction": instruction})
    return result
