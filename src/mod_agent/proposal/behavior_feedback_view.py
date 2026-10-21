"""Lossless public expectations and exact code references for behavior feedback only."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import PurePosixPath, PureWindowsPath


def project_execution_reference_paths(value):
    """Retain failure facts and hashes; keep known local execution addresses host-only.

    为什么要有它：嵌套执行工件地址不是业务观察，不能进入模型或阻断合法上游调查；未识别或无有效哈希的路径仍交原门禁拒绝。
    """
    reference_fields = {"result_ref", "execution_ref", "verification_ref", "failure_report_ref"}

    def project(item):
        if isinstance(item, dict):
            result = {}
            for key, nested in item.items():
                if (key in reference_fields and isinstance(nested, dict)
                        and isinstance(nested.get("path"), str)
                        and isinstance(nested.get("sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", nested["sha256"])
                        and (PureWindowsPath(nested["path"]).is_absolute()
                             or PurePosixPath(nested["path"]).is_absolute())):
                    result[key] = {name: project(child) for name, child in nested.items() if name != "path"}
                else:
                    result[key] = project(nested)
            return result
        if isinstance(item, list):
            return [project(child) for child in item]
        if isinstance(item, tuple):
            return tuple(project(child) for child in item)
        return deepcopy(item)

    return project(value)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _select(value, path):
    for key in path:
        if isinstance(value, dict) and isinstance(key, str):
            value = value[key]
        elif isinstance(value, list) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise ValueError("Unsupported public observation path")
    return value


def _differences(expected, observed, path=()):
    if _json(expected) == _json(observed):
        return []
    if isinstance(expected, dict) and isinstance(observed, dict) and expected.keys() == observed.keys():
        return [difference for key in expected for difference in _differences(expected[key], observed[key], (*path, key))]
    if isinstance(expected, list) and isinstance(observed, list) and len(expected) == len(observed):
        return [difference for index, value in enumerate(expected)
            for difference in _differences(value, observed[index], (*path, index))]
    return [{"path": list(path), "expected": deepcopy(expected), "observed": deepcopy(observed)}]


def _restore_expected(observed, differences):
    restored = deepcopy(observed)
    for difference in differences:
        path = difference["path"]
        if _json(_select(observed, path)) != _json(difference["observed"]):
            raise ValueError("Public observation difference changed")
        if not path:
            restored = deepcopy(difference["expected"])
        else:
            _select(restored, path[:-1])[path[-1]] = deepcopy(difference["expected"])
    return restored


def _public_observation(row):
    if not isinstance(row, dict) or row.get("origin") != "explicit_public_development_case":
        return deepcopy(row)
    expected = row.get("expected")
    if not isinstance(expected, list) or not expected or "observed" not in row:
        return deepcopy(row)
    projected = deepcopy(row)
    expectations = []
    try:
        for assertion in expected:
            if set(assertion) != {"path", "equals"} or not isinstance(assertion["path"], list):
                return deepcopy(row)
            observed = _select(row["observed"], assertion["path"])
            differences = _differences(assertion["equals"], observed)
            if _json(_restore_expected(observed, differences)) != _json(assertion["equals"]):
                return deepcopy(row)
            expectations.append({"path": assertion["path"], "matches_observed_except": differences})
    except (KeyError, IndexError, TypeError, ValueError):
        return deepcopy(row)
    projected.pop("expected")
    projected["expectations"] = expectations
    return projected if len(_json(projected)) < len(_json(row)) else deepcopy(row)


def project_behavior_feedback(feedback, previous_response):
    """Change model display only; raw verifier artifacts and all measured values remain intact.

    为什么要有它：失败反馈的重复代码和相同预期不应占满纠错预算，但必须能精确定位公开输入的预期/实际差异。
    """
    result = deepcopy(dict(feedback))
    notes = []
    original_rows = result.get("observations", [])
    rows = [_public_observation(row) for row in original_rows]
    if rows != original_rows:
        notes.append("Each expectations item compares at path in observed. All expected values equal observed "
            "except matches_observed_except; its paths are relative to that selected value and give exact expected/observed differences.")
    fields = ("case_bundle_sha256", "public_worker_sha256", "materialized_code_identity")
    public = [row for row in rows if isinstance(row, dict) and row.get("origin") == "explicit_public_development_case"]
    if len(public) > 1 and all(all(key in row for key in fields) for row in public):
        shared = {key: deepcopy(public[0][key]) for key in fields}
        if all(all(_json(row[key]) == _json(shared[key]) for key in fields) for row in public):
            result["public_development_provenance"] = shared
            for row in public:
                for key in fields:
                    row.pop(key)
            notes.append("public_development_provenance applies to every explicit_public_development_case observation.")
    result["observations"] = rows
    view = result.get("working_code_view")
    try:
        response = json.loads(previous_response)
        changes = response["changes"]
        if not isinstance(changes, list) or not changes:
            raise ValueError("No exact candidate text")
        files = view["files"]
        raw_files = {}
        references = []
        for file in files:
            if file["complete"] is not True or len(file["hunks"]) != 1:
                raise ValueError("Partial code view")
            hunk = file["hunks"][0]
            raw = hunk["text"].encode("utf-8")
            if hunk["start_line"] != 1 or hunk["end_line"] != file["line_count"]:
                raise ValueError("Incomplete code coordinates")
            if hashlib.sha256(raw).hexdigest() != file["raw_bytes_sha256"]:
                raise ValueError("Code view bytes differ")
            matches = [change for change in changes if isinstance(change, dict)
                and change.get("after_text") == hunk["text"] and isinstance(change.get("edit_binding_id"), str)]
            if len(matches) != 1:
                raise ValueError("No unique full-file candidate text")
            references.append(matches[0]["edit_binding_id"])
            raw_files[file["path"]] = raw
        from src.mod_agent.execution.patch_apply import _materialized_files_sha256

        if _materialized_files_sha256(raw_files) != view["code_sha256"]:
            raise ValueError("Aggregate code identity differs")
        for file, binding_id in zip(files, references):
            file["hunks"][0].pop("text")
            file["hunks"][0]["text_from_checked_patch"] = {"edit_binding_id": binding_id,
                "after_text_sha256": file["raw_bytes_sha256"]}
            file.pop("unified_diff", None)
        notes.append("working_code_view text_from_checked_patch is the exact after_text for that binding in the "
            "checked assistant Patch C1 identified by this correction. Original file/line/code identities remain; "
            "the derived duplicate diff display is omitted. B0 and edit authority are unchanged.")
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    if notes:
        result["model_view"] = " ".join(notes)
    from .observation_value_refs import project_observation_values

    result = project_observation_values(result)
    return result if len(_json(result).encode("utf-8")) < len(_json(feedback).encode("utf-8")) else deepcopy(dict(feedback))



def _visible_patch_documents(messages):
    patches = []
    for message in messages:
        if message.role != "assistant" or not isinstance(message.content, str):
            continue
        try:
            value = json.loads(message.content)
            if isinstance(value, dict) and isinstance(value.get("changes"), list):
                patches.append((message.content, value))
        except (TypeError, ValueError):
            continue
    return patches


def restore_visible_patch_response(projected, messages):
    """Expand input-only code refs against earlier literal assistant code."""
    value = json.loads(projected)
    guide = value.pop("checked_patch_input_view")
    if set(guide) != {"version", "original_object_sha256", "guide"} or guide["version"] != "same_visible_after_text_v1":
        raise ValueError("Unsupported visible code reference version")
    patches = _visible_patch_documents(messages)
    for change in value["changes"]:
        if "after_text_ref" not in change:
            continue
        reference = change.pop("after_text_ref")
        if set(reference) != {"earlier_assistant_patch", "response_sha256"}:
            raise ValueError("Invalid visible code reference fields")
        index = reference["earlier_assistant_patch"]
        if type(index) is not int or not 1 <= index <= len(patches) or "after_text" in change:
            raise ValueError("Invalid visible code reference")
        text, anchor = patches[index - 1]
        matches = [row for row in anchor["changes"] if isinstance(row, dict) and row.get("edit_binding_id") == change.get("edit_binding_id")]
        if (len(matches) != 1 or not isinstance(matches[0].get("after_text"), str)
                or reference["response_sha256"] != hashlib.sha256(text.encode("utf-8")).hexdigest()):
            raise ValueError("Visible code source changed")
        change["after_text"] = matches[0]["after_text"]
    if hashlib.sha256(_json(value).encode("utf-8")).hexdigest() != guide["original_object_sha256"]:
        raise ValueError("Expanded checked Patch differs from original")
    return value


def project_visible_patch_response(previous_response, messages):
    """Retain each revision's other fields while sharing byte-identical code only.

    为什么要有它：不同解释不能让相同失败代码重复占用上下文；每次显示仍须精确还原该次原响应，原始响应记录保持不变。
    """
    try:
        original = json.loads(previous_response)
        if (not isinstance(original, dict) or "checked_patch_input_view" in original
                or not isinstance(original.get("changes"), list)
                or any(not isinstance(row, dict) for row in original["changes"])):
            return previous_response
        projected = deepcopy(original)
        changed = False
        patches = _visible_patch_documents(messages)
        for change in projected["changes"]:
            if not isinstance(change.get("after_text"), str) or "after_text_ref" in change:
                continue
            for index, (text, anchor) in enumerate(patches, 1):
                matches = [row for row in anchor["changes"] if isinstance(row, dict) and row.get("edit_binding_id") == change.get("edit_binding_id")]
                if len(matches) == 1 and matches[0].get("after_text") == change["after_text"]:
                    change.pop("after_text")
                    change["after_text_ref"] = {"earlier_assistant_patch": index,
                        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
                    changed = True
                    break
        if not changed:
            return previous_response
        projected["checked_patch_input_view"] = {"version": "same_visible_after_text_v1",
            "original_object_sha256": hashlib.sha256(_json(original).encode("utf-8")).hexdigest(),
            "guide": "Input only: after_text_ref selects the same binding's literal after_text in the Nth earlier assistant Patch (1-based). Other fields belong to this revision. Output literal after_text; edit authority is unchanged."}
        rendered = _json(projected)
        if restore_visible_patch_response(rendered, messages) != original:
            raise ValueError("Changed visible Patch input")
        return rendered if len(_json(rendered).encode("utf-8")) < len(_json(previous_response).encode("utf-8")) else previous_response
    except (KeyError, IndexError, TypeError, ValueError):
        return previous_response
