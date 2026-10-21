"""Field-level MCP validation feedback must not echo caller-controlled secrets.

为什么要有它：客户端需要可纠正的字段信息，但参数错误不能变成输入内容或未知字段名称的披露通道。
"""
from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator, validators

StrictValidator = validators.extend(Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine("integer", lambda _checker, value: type(value) is int))


def _feedback(arguments, schema):
    from src.mcp_server.argument_feedback import validation_feedback

    return validation_feedback(arguments, schema, StrictValidator)


def _schema(properties, **extra):
    return {"type": "object", "properties": properties, "additionalProperties": False, **extra}


def test_required_and_type_issues_use_known_paths_without_input_values():
    schema = _schema({"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}}, required=["query"])
    result = _feedback({"limit": "SECRET-PARAMETER-VALUE"}, schema)
    assert {(row["path"], row["code"]) for row in result["issues"]} == {("/query", "required"), ("/limit", "type")}
    assert result["execution"] == "not_executed" and result["retryable"] is True
    assert "SECRET-PARAMETER-VALUE" not in json.dumps(result)
    assert next(row for row in result["issues"] if row["code"] == "type")["expected"]["type"] == "integer"


def test_valid_arguments_return_none_and_strict_integer_validator_is_used():
    schema = _schema({"limit": {"type": "integer", "minimum": 1}})
    assert _feedback({"limit": 2}, schema) is None
    assert _feedback({"limit": 2.0}, schema)["issues"][0]["code"] == "type"
    assert _feedback({"limit": True}, schema)["issues"][0]["code"] == "type"


def test_unknown_property_name_and_value_are_never_reported():
    result = _feedback({"TOP-SECRET-DYNAMIC-KEY": "SECRET-VALUE"}, _schema({"query": {"type": "string"}}))
    assert result["issues"] == [{"path": "", "code": "additional_properties", "expected": {"additionalProperties": False}}]
    assert "SECRET" not in json.dumps(result)


def test_nested_arrays_have_registered_escaped_paths_and_numeric_indexes():
    schema = _schema({"a/b~c": {"type": "array", "items": _schema({"limit": {"type": "integer", "minimum": 1}})}})
    result = _feedback({"a/b~c": [{"limit": 0}]}, schema)
    assert result["issues"] == [{"path": "/a~1b~0c/0/limit", "code": "minimum", "expected": {"minimum": 1}}]


@pytest.mark.parametrize("keyword,value,actual,code", [
    ("minimum", 2, 1, "minimum"), ("exclusiveMinimum", 2, 2, "exclusive_minimum"),
    ("maximum", 2, 3, "maximum"), ("exclusiveMaximum", 2, 2, "exclusive_maximum"),
    ("minLength", 3, "x", "min_length"), ("maxLength", 1, "xx", "max_length"),
    ("minItems", 2, [], "min_items"), ("maxItems", 1, [1, 2], "max_items"),
    ("minProperties", 2, {}, "min_properties"), ("maxProperties", 1, {"SECRET-A": 1, "SECRET-B": 2}, "max_properties"),
    ("pattern", "^[a-z]+$", "SECRET-123", "pattern"), ("enum", ["short", "full"], "SECRET", "enum"),
])
def test_public_constraints_are_actionable_without_echo(keyword, value, actual, code):
    result = _feedback({"field": actual}, _schema({"field": {keyword: value}}))
    assert result["issues"] == [{"path": "/field", "code": code, "expected": {keyword: value}}]
    assert "SECRET" not in json.dumps(result)


def test_dynamic_object_key_is_not_exposed_by_nested_type_error():
    schema = _schema({"metadata": {"type": "object", "additionalProperties": {"type": "integer"}}})
    result = _feedback({"metadata": {"PRIVATE-KEY": "PRIVATE-VALUE"}}, schema)
    assert result["issues"][0]["path"] == "/metadata"
    assert "PRIVATE" not in json.dumps(result)


def test_required_but_undeclared_field_and_unknown_validator_use_schema_reference():
    result = _feedback({}, _schema({}, required=["UNDECLARED-KEY"]))
    assert "UNDECLARED-KEY" not in json.dumps(result)
    assert result["issues"][0]["path"] == ""
    result = _feedback({"field": "SECRET"}, _schema({"field": {"not": {"type": "string"}}}))
    assert result["issues"] == [{"path": "/field", "code": "validation", "expected": {"schema_ref": "inputSchema"}}]


def test_output_is_bounded_with_explicit_truncation_and_schema_validates_it():
    from src.mcp_server.argument_feedback import ARGUMENT_FEEDBACK_SCHEMA

    schema = _schema({f"field{index}": {"type": "string"} for index in range(30)})
    result = _feedback({f"field{index}": 0 for index in range(30)}, schema)
    assert len(result["issues"]) == 8 and result["issues_truncated"] is True
    Draft202012Validator(ARGUMENT_FEEDBACK_SCHEMA).validate(result)
    large = _feedback({"field": "bad"}, _schema({"field": {"enum": ["中" * 6000]}}))
    assert len(json.dumps(large, ensure_ascii=False).encode("utf-8")) <= 8192
    assert large["issues_truncated"] is True
    Draft202012Validator(ARGUMENT_FEEDBACK_SCHEMA).validate(large)


@pytest.mark.parametrize("schema", [{}, {"properties": {}}, _schema({})])
def test_root_argument_type_error_never_echoes_array_or_scalar_contents(schema):
    result = _feedback(["SECRET"], schema)
    assert result["issues"] == [{"path": "", "code": "type", "expected": {"type": "object"}}]
    assert "SECRET" not in json.dumps(result)


def test_complex_enum_values_and_unregistered_required_names_are_not_copied():
    result = _feedback({"field": {}}, _schema({"field": {"enum": [{"DYNAMIC-SCHEMA-KEY": "public"}]}},
        required=["field", "OTHER-UNREGISTERED-KEY"]))
    assert "DYNAMIC-SCHEMA-KEY" not in json.dumps(result)
    assert "OTHER-UNREGISTERED-KEY" not in json.dumps(result)
    assert result["issues_truncated"] is True


def test_multibyte_many_constraint_hints_stay_within_serialized_byte_limit():
    properties = {f"field{index}": {"enum": [f"{number}-" + "例" * 16 for number in range(30)]} for index in range(8)}
    result = _feedback({key: "invalid" for key in properties}, _schema(properties))
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 8192
    assert result["issues_truncated"] is True


def test_unusual_public_schema_names_and_large_numeric_bound_fail_safely():
    result = _feedback({}, _schema({"\ud800": {"type": "string"}}, required=["\ud800"]))
    assert result["issues"][0]["path"] == "" and result["issues_truncated"]
    result = _feedback({"number": 1}, _schema({"number": {"minimum": 10 ** 400}}))
    assert result["issues"][0]["code"] == "minimum"
