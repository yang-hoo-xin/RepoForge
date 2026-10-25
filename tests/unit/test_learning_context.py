import json
from copy import deepcopy

import pytest

from src.mod_agent.planning.learning_context import (
    compact_learning_content,
    expand_learning_content,
)


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _encoded():
    original = "原任务内容 with exact whitespace\r\n and punctuation. " * 70
    return compact_learning_content({"first": original, "second": original})


def test_repeated_long_substring_preserves_order_fields_and_identifiers():
    original = "Original task: 保留原文。\r\n* exact bytes in JSON strings. " * 90
    data = {"episodes": [{"episode_id": "a" * 64,
        "fields": {"original_request": original,
            "acceptance_criteria": ["Requested behavior is implemented: " + original, "pass"]},
        "observations": [None, True, 1, 1.5, ""]} for _ in range(3)], "last": "unchanged"}
    before = deepcopy(data)
    encoded = compact_learning_content(data)
    assert encoded["encoding"] == "text_references"
    assert _size(encoded) < _size(data)
    assert data == before
    restored = expand_learning_content(json.loads(json.dumps(encoded, ensure_ascii=False)))
    assert json.dumps(restored, ensure_ascii=False) == json.dumps(data, ensure_ascii=False)
    assert encoded["content"]["episodes"][0]["episode_id"] == "a" * 64
    definitions = encoded["text_definitions"]
    assert list(definitions.values()).count(original) == 1
    criteria = encoded["content"]["episodes"][0]["fields"]["acceptance_criteria"][0]
    if "$text_ref" in criteria:
        criteria = definitions[criteria["$text_ref"]]
    assert "$text_concat" in criteria


def test_nonbeneficial_and_short_identifiers_are_unchanged():
    data = {"id": "a" * 64, "ids": ["a" * 64] * 20, "one_long_text": "x" * 300}
    assert compact_learning_content(data) == data
    assert expand_learning_content(data) == data
    borderline = {"a": "x" * 160, "b": "x" * 160}
    assert compact_learning_content(borderline) == borderline


def test_original_marker_objects_and_nested_wrapper_are_literal():
    long_text = "Literal selected evidence with quotes and Unicode 中文。 " * 80
    data = {"first": long_text, "second": long_text,
        "$text_ref": "unknown-original-id", "nested": {
            "$literal_object": [["original", {"$text_concat": ["not", "an instruction"]}]],
            "schema_version": "learning_literal_text_refs_v1", "encoding": "text_references"}}
    encoded = compact_learning_content(data)
    assert encoded["encoding"] == "text_references"
    assert "$literal_object" in encoded["content"]
    assert expand_learning_content(encoded) == data


def test_top_level_codec_namespace_collision_has_explicit_literal_escape():
    for data in (_encoded(), {"schema_version": "learning_literal_text_future", "$text_ref": "missing"}):
        encoded = compact_learning_content(data)
        assert encoded["encoding"] == "literal"
        assert expand_learning_content(json.loads(json.dumps(encoded))) == data


@pytest.mark.parametrize("failure", ["unknown", "cycle", "deep", "malformed"])
def test_invalid_reference_graph_is_rejected(failure):
    encoded = _encoded()
    if failure == "unknown":
        encoded["content"] = {"$text_ref": "not-defined"}
        expected = "UNKNOWN_REFERENCE"
    elif failure == "cycle":
        encoded["text_definitions"] = {"a": {"$text_ref": "b"}, "b": {"$text_ref": "a"}}
        expected = "CYCLIC_REFERENCE"
    elif failure == "deep":
        encoded["text_definitions"] = {"t0": "literal", **{
            "t" + str(i): {"$text_ref": "t" + str(i - 1)} for i in range(1, 19)}}
        expected = "REFERENCE_TOO_DEEP"
    else:
        encoded["content"] = {"$text_ref": "text_1", "other": "invalid marker"}
        expected = "TEXT_EXPRESSION_INVALID"
    with pytest.raises(ValueError, match=expected):
        expand_learning_content(encoded)


def test_unknown_version_is_rejected_and_input_is_not_modified():
    encoded = _encoded()
    saved = deepcopy(encoded)
    expand_learning_content(encoded)
    assert encoded == saved
    encoded["schema_version"] = "learning_literal_text_refs_v999"
    with pytest.raises(ValueError, match="UNKNOWN_VERSION"):
        expand_learning_content(encoded)
