"""Bounded MCP argument diagnostics from public schema, never caller values.

为什么要有它：参数填写错误需要可操作反馈，但验证器错误文本可能含凭据、未知键或整个输入，不能直接转发。
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

_CODES = {"required": "required", "type": "type", "minimum": "minimum", "maximum": "maximum",
    "exclusiveMinimum": "exclusive_minimum", "exclusiveMaximum": "exclusive_maximum",
    "multipleOf": "multiple_of", "minLength": "min_length", "maxLength": "max_length",
    "minItems": "min_items", "maxItems": "max_items", "minProperties": "min_properties",
    "maxProperties": "max_properties", "pattern": "pattern", "enum": "enum",
    "additionalProperties": "additional_properties"}
_TYPES = ["object", "array", "string", "integer", "number", "boolean", "null"]
_LIMITS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"}
_NEXT = "Correct arguments using tools/list inputSchema and retry; no handler was executed."
_REFERENCE = {"schema_ref": "inputSchema"}
_MAX_ISSUES, _MAX_BYTES = 8, 8192
_EXPECTED_PROPERTIES = {"schema_ref": {"const": "inputSchema"}, "required": {"const": True},
    "type": {"anyOf": [{"enum": _TYPES}, {"type": "array", "items": {"enum": _TYPES}, "maxItems": 7}]},
    "pattern": {"type": "string"}, "enum": {"type": "array", "maxItems": 32,
        "items": {"type": ["string", "number", "boolean", "null"]}}, "additionalProperties": {"const": False},
    **{key: {"type": "number"} for key in _LIMITS}}
ARGUMENT_FEEDBACK_SCHEMA = {"type": "object", "additionalProperties": False,
    "required": ["execution", "retryable", "issues", "issues_truncated", "next_action"], "properties": {
        "execution": {"const": "not_executed"}, "retryable": {"const": True},
        "issues": {"type": "array", "minItems": 1, "maxItems": _MAX_ISSUES, "items": {
            "type": "object", "additionalProperties": False, "required": ["path", "code", "expected"],
            "properties": {"path": {"type": "string"}, "code": {"enum": sorted({*_CODES.values(), "validation"})},
                "expected": {"type": "object", "additionalProperties": False,
                    "minProperties": 1, "properties": _EXPECTED_PROPERTIES}}}},
        "issues_truncated": {"type": "boolean"}, "next_action": {"const": _NEXT}}}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _pointer(parts):
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _location(schema, path):
    """Stop at the last registered path; dynamic object keys are never returned.

    为什么要有它：error.absolute_path中的动态键来自请求本身，可能是密钥或其他敏感文本。
    """
    current, parts = schema, []
    for part in path:
        if not isinstance(current, Mapping):
            return parts, current, False
        properties = current.get("properties", {})
        if type(part) is str and isinstance(properties, Mapping) and part in properties:
            selected = properties[part]
        elif type(part) is int and part >= 0:
            prefix = current.get("prefixItems", [])
            if isinstance(prefix, list) and part < len(prefix):
                selected = prefix[part]
            elif isinstance(current.get("items"), (Mapping, bool)):
                selected = current["items"]
            else:
                return parts, current, False
        else:
            return parts, current, False
        try:
            oversized = len(_pointer([*parts, part]).encode("utf-8")) > 1024
        except UnicodeError:
            oversized = True
        if oversized:
            return parts, current, False
        parts.append(part)
        current = selected
    return parts, current, True


def _public_expected(keyword, schema):
    if not isinstance(schema, Mapping) or keyword not in schema:
        return dict(_REFERENCE), False
    value = schema[keyword]
    accepted = False
    if keyword == "type":
        accepted = type(value) is str and value in _TYPES or isinstance(value, list) and len(value) <= 7 and all(item in _TYPES for item in value)
    elif keyword in _LIMITS:
        accepted = type(value) is int or type(value) is float and math.isfinite(value)
    elif keyword == "pattern":
        accepted = type(value) is str
    elif keyword == "enum":
        accepted = isinstance(value, list) and len(value) <= 32 and all(type(item) in (str, int, float, bool, type(None)) for item in value)
    elif keyword == "additionalProperties":
        accepted = value is False
    if not accepted:
        return dict(_REFERENCE), True
    try:
        result = {keyword: value}
        if _size(result) <= 2048:
            return result, False
    except (TypeError, ValueError, OverflowError):
        pass
    return dict(_REFERENCE), True


def _issues(error, root):
    parts, selected, exact = _location(root, error.absolute_path)
    keyword = error.validator
    code = _CODES.get(keyword, "validation") if type(keyword) is str else "validation"
    if not exact:
        yield {"path": _pointer(parts), "code": "validation", "expected": dict(_REFERENCE)}, True
        return
    if keyword == "required" and isinstance(error.instance, Mapping) and isinstance(selected, Mapping):
        properties = selected.get("properties", {})
        produced, omitted = False, False
        required = error.validator_value if isinstance(error.validator_value, list) else []
        for name in required:
            if type(name) is not str or name in error.instance:
                continue
            if not isinstance(properties, Mapping) or name not in properties:
                omitted = True
                continue
            path, _schema, safe = _location(root, [*parts, name])
            if not safe:
                omitted = True
                continue
            expected = {"required": True}
            field_type, _ = _public_expected("type", properties[name])
            if "type" in field_type:
                expected.update(field_type)
            produced = True
            yield {"path": _pointer(path), "code": "required", "expected": expected}, False
        if produced and not omitted:
            return
    expected, truncated = _public_expected(keyword, error.schema) if code != "validation" else (dict(_REFERENCE), False)
    yield {"path": _pointer(parts), "code": code, "expected": expected}, truncated


def validation_feedback(arguments, schema, validator_cls):
    """Return safe correction hints, or None; this function never invokes a handler.

    At most eight issues and 8 KiB are returned; omitted detail is explicit.
    为什么要有它：反馈应足够纠正参数但不能耗尽上下文；MCP参数必须是对象，即使工具schema省略了type。
    """
    result = {"execution": "not_executed", "retryable": True, "issues": [],
        "issues_truncated": False, "next_action": _NEXT}
    if not isinstance(arguments, dict):
        result["issues"] = [{"path": "", "code": "type", "expected": {"type": "object"}}]
        return result
    seen = set()
    # Bound duplicate/dynamic error enumeration too, not only displayed rows.
    # 为什么要有它：大量未知键可能产生同一安全父路径，不能靠去重后数量很小就无限扫描错误。
    for index, error in enumerate(validator_cls(schema).iter_errors(arguments)):
        if index >= 64:
            result["issues_truncated"] = True
            break
        for issue, truncated in _issues(error, schema):
            result["issues_truncated"] |= truncated
            identity = _json(issue)
            if identity in seen:
                continue
            if len(result["issues"]) >= _MAX_ISSUES:
                result["issues_truncated"] = True
                return result
            proposed = {**result, "issues": [*result["issues"], issue]}
            if _size(proposed) > _MAX_BYTES - 256:
                issue = {**issue, "expected": dict(_REFERENCE)}
                result["issues_truncated"] = True
                proposed = {**result, "issues": [*result["issues"], issue]}
                if _size(proposed) > _MAX_BYTES - 256:
                    return result
            result["issues"].append(issue)
            seen.add(identity)
    return result if result["issues"] else None
