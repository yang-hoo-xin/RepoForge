"""Deterministic current-candidate editing over the existing immutable snapshots.

Scope: reconstruct a cumulative native patch, not an Apply or model response.
为什么要有它：模型修改当前代码，执行层仍须保留可重建的原始调用和父版本，而不是伪造模型提交的累计补丁。
"""
from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace


def _unique(text, fragment):
    position = text.find(fragment)
    return bool(fragment) and position >= 0 and text.find(fragment, position + 1) < 0


def context_transfer_content(*, baseline_files, current_files, implementation_plan, prepared_contract):
    """Re-express identical bytes under a new, independently admitted Plan.

    Only existing changed regions are mapped; untouched new bindings remain in
    the full Plan, not as fabricated no-op patches. 为什么要有它：计划可改变绑定和范围，
    但转移不能丢掉旧代码、替模型完成新需求或给未修改区域伪造检查结果。
    """
    validate_current_candidate_contract(implementation_plan, prepared_contract)
    bindings = {row["binding_id"]: row for row in implementation_plan["binding_registry"]["edit_bindings"]}
    units = prepared_contract.materialization_context["required_change_units"]
    replacements = {}
    for path, raw in current_files.items():
        before = baseline_files[path].decode("utf-8")
        after = raw.decode("utf-8")
        replacements[path] = [(a, b, after[c:d]) for tag, a, b, c, d in
            difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes() if tag != "equal"]
    changes, covered = [], set()
    for unit in units:
        binding = bindings[unit["edit_binding_id"]]
        path = binding["canonical_target_path"]
        if path not in replacements:
            continue
        original = baseline_files[path].decode("utf-8")
        lines = original.splitlines(keepends=True)
        span = binding["target_span"]
        start, end = sum(map(len, lines[:span["start_line"] - 1])), sum(map(len, lines[:span["end_line"]]))
        owned = [(index, a, b, text) for index, (a, b, text) in enumerate(replacements[path])
            if start <= a and b <= end and (a < end or b > a)]
        if not owned:
            continue
        selected_before = original[start:end]
        selected_after = selected_before
        for index, a, b, text in reversed(owned):
            if (path, index) in covered:
                raise ValueError("TASK_CURRENT_TRANSFER_AMBIGUOUS_SCOPE")
            covered.add((path, index))
            selected_after = selected_after[:a - start] + text + selected_after[b - start:]
        changes.append({"edit_binding_id": unit["edit_binding_id"], "before_text": selected_before,
            "after_text": selected_after, "rationale": "Preserve the existing candidate bytes under the admitted new plan.",
            "rollback_hint": "Restore the original source preimage."})
    expected = {(path, index) for path, rows in replacements.items() for index in range(len(rows))}
    if covered != expected or not changes:
        raise ValueError("TASK_CURRENT_TRANSFER_SCOPE_NOT_COVERED")
    content = json.dumps({"changes": changes}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return content, candidate_projection_contract(prepared_contract, content)


def candidate_projection_contract(prepared_contract, content):
    """Project code-present units without removing any requirement from the Plan.

    Scope: host transfer/current cumulative materialization only. 为什么要有它：
    新计划可以含尚未修改的区域；代码快照不能为了满足列表长度伪造修改，最终检查仍使用完整计划。
    """
    selected = {row["edit_binding_id"] for row in json.loads(content)["changes"]}
    context = deepcopy(prepared_contract.materialization_context)
    allowed = {row["edit_binding_id"] for row in context["required_change_units"]}
    if not selected or not selected <= allowed:
        raise ValueError("TASK_CURRENT_EDIT_BINDING_NOT_AUTHORIZED")
    context["required_change_units"] = [row for row in context["required_change_units"] if row["edit_binding_id"] in selected]
    return replace(prepared_contract, materialization_context=context)


def validate_current_candidate_contract(implementation_plan, prepared_contract):
    """Reject unsupported overlapping edit regions before any model invocation."""
    if prepared_contract.contract_version != "3":
        raise ValueError("TASK_CURRENT_NATIVE_V3_REQUIRED")
    required = {row["edit_binding_id"] for row in prepared_contract.materialization_context["required_change_units"]}
    bindings = {row["binding_id"]: row for row in implementation_plan["binding_registry"]["edit_bindings"]}
    selected = []
    for binding_id in required:
        row = bindings[binding_id]
        span = row.get("target_span")
        if (not isinstance(span, Mapping) or type(span.get("start_line")) is not int
                or type(span.get("end_line")) is not int or span["start_line"] < 1
                or span["end_line"] < span["start_line"]):
            raise ValueError("TASK_CURRENT_EXPLICIT_BINDING_SPAN_REQUIRED")
        selected.append((row["canonical_target_path"], span["start_line"], span["end_line"]))
    selected.sort()
    for left, right in zip(selected, selected[1:]):
        if left[0] == right[0] and left[2] >= right[1]:
            raise ValueError("TASK_CURRENT_OVERLAPPING_BINDINGS_UNSUPPORTED")


def derive_current_revision(parent_candidate, *, arguments, prepared_contract):
    """Return cumulative native JSON from authenticated parent and actual edit args.

The caller authenticates the parent origin and its materialized bytes first.
Edits are sequential within one binding; other bindings retain parent changes.
为什么要有它：局部修订必须保留之前的有效修改，并继续受原生生成器、Plan和精确源码校验约束。
    """
    if (not isinstance(arguments, Mapping) or set(arguments) != {
            "context_id", "parent_candidate_id", "parent_code_sha256", "edits"}
            or arguments["context_id"] != parent_candidate.get("context_id")
            or arguments["parent_candidate_id"] != parent_candidate.get("candidate_id")
            or arguments["parent_code_sha256"] != parent_candidate.get("materialized_code_identity", {}).get("code_sha256")):
        raise ValueError("TASK_CURRENT_PARENT_VERSION_CHANGED")
    edits = arguments["edits"]
    if not isinstance(edits, list) or not edits:
        raise ValueError("TASK_CURRENT_EDITS_REQUIRED")
    contexts = prepared_contract.materialization_context
    originals = contexts["binding_evidence_excerpts"]
    units = contexts["required_change_units"]
    previous = {row["edit_binding_id"]: row for row in parent_candidate["generated_changes"]}
    if len(previous) != len(parent_candidate["generated_changes"]) or not set(previous) <= {row["edit_binding_id"] for row in units}:
        raise ValueError("TASK_CURRENT_PARENT_BINDINGS_CHANGED")
    current = {unit["edit_binding_id"]: originals[unit["edit_binding_id"]] for unit in units}
    rationales = {binding: "Retain current code." for binding in current}
    for binding_id, row in previous.items():
        patch = row["patch"]
        before, after = patch["before_text"], patch["after_text"]
        original = originals[binding_id]
        if not _unique(original, before):
            raise ValueError("TASK_CURRENT_PARENT_EXCERPT_CHANGED")
        current[binding_id] = original.replace(before, after, 1)
        rationales[binding_id] = row["rationale"]
    # Validate every target before classifying an ordinary no-op as recoverable.
    # 为什么要有它：一个无改动不能掩盖同一批次后面的越权binding。
    for edit in edits:
        if (not isinstance(edit, Mapping) or set(edit) != {"edit_binding_id", "before_text", "after_text", "rationale"}
                or any(type(edit[key]) is not str for key in edit)
                or not edit["before_text"] or not edit["rationale"]):
            raise ValueError("TASK_CURRENT_EDIT_ARGUMENTS_INVALID")
        binding_id = edit["edit_binding_id"]
        if binding_id not in current:
            raise PermissionError("TASK_CURRENT_EDIT_BINDING_NOT_AUTHORIZED")
    contains_noop = False
    for edit in edits:
        binding_id = edit["edit_binding_id"]
        before, after = edit["before_text"], edit["after_text"]
        if not _unique(current[binding_id], before):
            raise ValueError("TASK_CURRENT_BEFORE_TEXT_NOT_UNIQUE")
        if before.startswith("\ufeff") != after.startswith("\ufeff"):
            raise ValueError("TASK_CURRENT_BOM_CHANGED")
        if before == after:
            contains_noop = True
        current[binding_id] = current[binding_id].replace(before, after, 1)
        rationales[binding_id] = edit["rationale"]
    if contains_noop:
        from src.mod_agent.task_agent_loop import TaskCandidateRevisionInputError
        raise TaskCandidateRevisionInputError(("edits.before_text and after_text are identical; no code was changed",))
    changes = []
    for unit in units:
        binding_id = unit["edit_binding_id"]
        original, updated = originals[binding_id], current[binding_id]
        if original == updated:
            continue
        # One native v3 replacement per existing binding; never promote a local
        # binding into whole-file authority. 为什么要有它：累计补丁不能越过Plan绑定范围。
        changes.append({"edit_binding_id": binding_id, "before_text": original, "after_text": updated,
            "rationale": rationales[binding_id], "rollback_hint": previous.get(binding_id, {}).get("rollback_hint", "Restore original source.")})
    return json.dumps({"changes": changes}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
