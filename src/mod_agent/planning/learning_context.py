"""Reversible, readable literal-text deduplication for selected learning content.

为什么要有它：重复原文不应挤掉已选择的经历；共享文字仍须逐字可还原，不能用摘要替换证据。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from copy import deepcopy

_NAMESPACE = "learning_literal_text_"
_VERSION = _NAMESPACE + "refs_v1"
_MARKERS = {"$text_ref", "$text_concat", "$literal_object"}
_MIN_TEXT_LENGTH = 96
_MAX_REFERENCE_DEPTH = 16
_MAX_STRUCTURE_DEPTH = 128
_EXPLANATION = (
    "Lossless literal text encoding, not a summary. Read text_definitions first. "
    "$text_ref names its exact text; $text_concat joins its string/reference parts in order "
    "without separators. Resolve definitions by name, regardless of table order. $literal_object "
    "preserves an original object as ordered key/value pairs; its values use the same encoding. "
    "Expand these text values in content to recover the original selected learning input. "
    "All original evidence IDs, locators and authority remain unchanged. 为什么要有它：共享重复原文，保留全部已选择证据。"
)


def _require(condition, code):
    if not condition:
        raise ValueError("LEARNING_CONTEXT_" + code)


def _json_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _namespace_collision(value):
    return isinstance(value, dict) and isinstance(value.get("schema_version"), str) \
        and value["schema_version"].startswith(_NAMESPACE)


def _strings(value, depth=0):
    _require(depth <= _MAX_STRUCTURE_DEPTH, "STRUCTURE_TOO_DEEP")
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        _require(all(isinstance(key, str) for key in value), "JSON_OBJECT_KEYS_REQUIRED")
        for item in value.values():
            yield from _strings(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, depth + 1)
    else:
        _require(value is None or type(value) in {bool, int, float}, "JSON_VALUE_REQUIRED")


def _text_encoder(text_ids):
    if not text_ids:
        return lambda text: text
    pattern = re.compile("|".join(re.escape(text) for text in sorted(text_ids, key=len, reverse=True)))

    def encode(text):
        parts, offset = [], 0
        for match in pattern.finditer(text):
            if match.start() > offset:
                parts.append(text[offset:match.start()])
            parts.append({"$text_ref": text_ids[match.group()]})
            offset = match.end()
        if not parts:
            return text
        if offset < len(text):
            parts.append(text[offset:])
        return parts[0] if len(parts) == 1 else {"$text_concat": parts}

    return encode


def compact_learning_content(content):
    """Return smaller readable encoding, or an unchanged copy of selected JSON data.

    A top-level codec namespace collision is escaped even if larger.
    为什么要有它：合法原数据可能长得像编码封套；仅此碰撞例外优先保证无歧义还原。
    """
    counts = Counter(_strings(content))
    original_size = _json_size(content)
    if _namespace_collision(content):
        return {"schema_version": _VERSION, "encoding": "literal", "content": deepcopy(content)}
    candidates = [text for text in counts if len(text) >= _MIN_TEXT_LENGTH]
    definitions, text_ids, reference_depths = {}, {}, {}
    for text in sorted(candidates, key=len):
        occurrences = sum(other.count(text) * count for other, count in counts.items())
        if occurrences < 2:
            continue
        name = "t" + str(len(definitions) + 1)
        available = {value: key for value, key in text_ids.items()
            if reference_depths[key] < _MAX_REFERENCE_DEPTH - 1}
        expression = _text_encoder(available)(text)
        definition_size = _json_size({name: expression})
        if occurrences * (_json_size(expression) - _json_size({"$text_ref": name})) <= definition_size:
            continue
        parts = expression.get("$text_concat", [expression]) if isinstance(expression, dict) else []
        reference_depths[name] = max((1 + reference_depths[part["$text_ref"]]
            for part in parts if isinstance(part, dict)), default=0)
        definitions[name], text_ids[text] = expression, name
    if not definitions:
        return deepcopy(content)
    encode_text = _text_encoder(text_ids)

    def encode(value):
        if isinstance(value, str):
            return encode_text(value)
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, dict):
            if _MARKERS.intersection(value):
                return {"$literal_object": [[key, encode(item)] for key, item in value.items()]}
            return {key: encode(item) for key, item in value.items()}
        return value

    encoded = {"schema_version": _VERSION, "encoding": "text_references", "explanation": _EXPLANATION,
        "text_definitions": definitions, "content": encode(content)}
    return encoded if _json_size(encoded) < original_size else deepcopy(content)


def expand_learning_content(encoded):
    """Restore exact selected JSON data; reject malformed or recursive text references."""
    if not _namespace_collision(encoded):
        return deepcopy(encoded)
    _require(encoded.get("schema_version") == _VERSION, "UNKNOWN_VERSION")
    mode = encoded.get("encoding")
    if mode == "literal":
        _require(set(encoded) == {"schema_version", "encoding", "content"}, "WRAPPER_INVALID")
        return deepcopy(encoded["content"])
    _require(mode == "text_references" and set(encoded) == {
        "schema_version", "encoding", "explanation", "text_definitions", "content"}, "WRAPPER_INVALID")
    _require(encoded["explanation"] == _EXPLANATION, "INSTRUCTIONS_CHANGED")
    definitions = encoded["text_definitions"]
    _require(isinstance(definitions, dict) and all(isinstance(key, str) and key for key in definitions),
        "DEFINITIONS_INVALID")
    resolved, resolving = {}, set()

    def resolve(name):
        _require(isinstance(name, str) and name in definitions, "UNKNOWN_REFERENCE")
        _require(name not in resolving, "CYCLIC_REFERENCE")
        if name not in resolved:
            _require(len(resolving) < _MAX_REFERENCE_DEPTH, "REFERENCE_TOO_DEEP")
            resolving.add(name)
            resolved[name] = text_value(definitions[name])
            resolving.remove(name)
        value, depth = resolved[name]
        _require(depth < _MAX_REFERENCE_DEPTH, "REFERENCE_TOO_DEEP")
        return value, depth + 1

    def text_value(value):
        if isinstance(value, str):
            return value, 0
        _require(isinstance(value, dict), "TEXT_EXPRESSION_INVALID")
        if set(value) == {"$text_ref"}:
            return resolve(value["$text_ref"])
        _require(set(value) == {"$text_concat"} and isinstance(value["$text_concat"], list)
            and bool(value["$text_concat"]), "TEXT_EXPRESSION_INVALID")
        parts = []
        for part in value["$text_concat"]:
            _require(isinstance(part, str) or isinstance(part, dict) and set(part) == {"$text_ref"},
                "TEXT_EXPRESSION_INVALID")
            parts.append(text_value(part))
        return "".join(text for text, _ in parts), max(depth for _, depth in parts)

    def decode(value, depth=0):
        _require(depth <= _MAX_STRUCTURE_DEPTH, "STRUCTURE_TOO_DEEP")
        if isinstance(value, list):
            return [decode(item, depth + 1) for item in value]
        if isinstance(value, dict):
            if set(value) == {"$literal_object"}:
                pairs = value["$literal_object"]
                _require(isinstance(pairs, list), "LITERAL_OBJECT_INVALID")
                restored = {}
                for pair in pairs:
                    _require(isinstance(pair, list) and len(pair) == 2 and isinstance(pair[0], str)
                        and pair[0] not in restored, "LITERAL_OBJECT_INVALID")
                    restored[pair[0]] = decode(pair[1], depth + 1)
                return restored
            if _MARKERS.intersection(value):
                return text_value(value)[0]
            return {key: decode(item, depth + 1) for key, item in value.items()}
        return deepcopy(value)

    for name in definitions:
        resolve(name)
    return decode(encoded["content"])
