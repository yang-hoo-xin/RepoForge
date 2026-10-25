"""Model-authored working state with evidence from the actual visible request.

为什么要有它：持续判断需要模型明确表达并保留来源；引用校验不证明判断正确，也不构成动作授权。
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace

from jsonschema import Draft202012Validator

from src.mod_agent.model_qualification import canonical_hash

SCHEMA = "task-agent-semantic-state.v1"
_BOUNDARY = ("task_contract_ref", "current_revision_ref", "current_check_ref")


def _document(value):
    return json.loads(json.dumps(asdict(value) if is_dataclass(value) else value, ensure_ascii=False, allow_nan=False))


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _quote_seen(value, quote):
    if quote in _text(value):
        return True
    if isinstance(value, dict):
        return any(_quote_seen(item, quote) for item in value.values())
    if isinstance(value, list):
        return any(_quote_seen(item, quote) for item in value)
    return False


def strip_working_state(arguments):
    """Leave the native arguments untouched and remove advisory metadata for dispatch."""
    copied = deepcopy(arguments)
    if isinstance(copied, dict):
        copied.pop("working_state", None)
    return copied


def semantic_contract():
    text = {"type": "string", "minLength": 1}
    evidence = {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"id": text, "quote": text}, "required": ["id", "quote"]}}
    observation = {"type": "object", "additionalProperties": False,
        "properties": {"statement": text, "evidence": {**evidence, "minItems": 1}},
        "required": ["statement", "evidence"]}
    hypothesis = {"type": "object", "additionalProperties": False,
        "properties": {"id": text, "statement": text, "status": {"enum": ["open", "supported", "refuted"]},
            "evidence": evidence}, "required": ["id", "statement", "status", "evidence"]}
    return {"type": "object", "additionalProperties": False, "properties": {
        "observations": {"type": "array", "items": observation},
        "hypotheses": {"type": "array", "items": hypothesis},
        "open_questions": {"type": "array", "items": text},
        "next_action": {"type": "object", "additionalProperties": False,
            "properties": {key: text for key in ("purpose", "expected_observation", "decision_if_observed")},
            "required": ["purpose", "expected_observation", "decision_if_observed"]}},
        "required": ["observations", "hypotheses", "open_questions", "next_action"],
        "description": "Your explicit advisory observations, hypotheses and next discriminating action. Cite catalog IDs with verbatim visible quotes. Supported/refuted are your judgments, not platform facts. Missing or invalid metadata does not authorize or prevent an action."}


def _current(document):
    rows = [(index, row["content"]["current_task_state"]) for index, row in enumerate(document["messages"])
        if row.get("role") == "user" and isinstance(row.get("content"), dict)
        and set(row["content"]) == {"current_task_state"}]
    if len(rows) != 1 or not isinstance(rows[0][1], dict):
        raise ValueError("SEMANTIC_CURRENT_STATE_AMBIGUOUS")
    return rows[0]


def _boundary(current):
    return {key: deepcopy(current.get(key)) for key in _BOUNDARY}


def _at(document, locator):
    value = document
    for key in locator:
        value = value[key]
    return value


def _source_boundary(kind, value, boundary):
    if kind in {"task_requirements", "capabilities"}:
        return {"task_contract_ref": boundary.get("task_contract_ref")}
    if kind == "history_excerpt":
        return {"task_contract_ref": boundary.get("task_contract_ref"), "mixed_historical_text": True}
    if isinstance(value, dict):
        source = {"task_contract_ref": value.get("contract_ref", value.get("task_contract_ref", boundary.get("task_contract_ref")))}
        revision = value.get("parent_revision_ref", value.get("candidate_ref", value.get("source_revision_ref")))
        if revision is not None and value.get("root_id", "target") == "target":
            source["current_revision_ref"] = revision
        if value.get("check_ref") is not None:
            source["current_check_ref"] = value["check_ref"]
        if len(source) > 1:
            return source
    return deepcopy(boundary)


def _sources(document):
    """Build only locators into sent values, never dereference a filesystem artifact.

    为什么要有它：源码原件可能包含未读页；归档混合文字仍可引用，但不能洗成独立源码或检查证据。
    """
    index, current = _current(document)
    boundary, sources = _boundary(current), []

    def add(kind, locator, label, identity, origin=None):
        value = _at(document, locator)
        digest = canonical_hash(value)
        identifier = "e_" + canonical_hash({"kind": kind, "identity": identity, "value": digest})[:24]
        if any(row["id"] == identifier for row in sources):
            return
        sources.append({"id": identifier, "kind": kind, "locator": locator,
            "value_sha256": digest, "label": label, "source_boundary": (
                deepcopy(origin["source_boundary"]) if origin is not None else _source_boundary(kind, value, boundary)),
            **({"retained_origin": deepcopy(origin)} if origin is not None else {})})

    for message_index, message in enumerate(document["messages"]):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, dict):
            continue
        if content.get("protocol") == "task_agent_v2" and "task_requirements" in content:
            add("task_requirements", ["messages", message_index, "content", "task_requirements"],
                "Original task requirements", content.get("task_contract_ref"))
        transition = content.get("source_context_transition")
        if isinstance(transition, dict) and transition.get("origin") == "execution_host" and transition.get("public_delivery_feedback") is not None:
            add("public_delivery_feedback", ["messages", message_index, "content", "source_context_transition", "public_delivery_feedback"],
                "Actual public delivery feedback", transition.get("rebase_ref"))
    if "experiment_capabilities" in current:
        add("capabilities", ["messages", index, "content", "current_task_state", "experiment_capabilities"],
            "Host-reported experiment capabilities", boundary.get("task_contract_ref"))
    if document.get("tools"):
        add("tool_definitions", ["tools"], "Available tool definitions; no extra authority", boundary)
    exchanges = [(row, ["historical_tool_exchanges", i, "tool_results"])
        for i, row in enumerate(document.get("historical_tool_exchanges", []))]
    exchanges.append(({"tool_results": document.get("tool_results", [])}, ["tool_results"]))
    for exchange, prefix in exchanges:
        for result_index, result in enumerate(exchange["tool_results"]):
            output = result.get("output")
            value = output.get("value", output) if isinstance(output, dict) else output
            locator = [*prefix, result_index, "output"] + (["value"] if isinstance(output, dict) and "value" in output else [])
            name = result.get("name")
            kind = ("history_excerpt" if name == "history_read" else "source_view" if isinstance(value, dict)
                and value.get("schema") == "task-agent-source-view.v2" else "public_check_return" if name in
                {"candidate_check", "check_feedback_read"} else "tool_return")
            add(kind, locator, str(name) + ":" + str(result.get("call_id")),
                {"call_id": result.get("call_id"), "name": name})
    view = current.get("semantic_work_state") or {}
    for quote_index, quote in enumerate(view.get("evidence_quotes", [])):
        if not isinstance(quote, dict) or not isinstance(quote.get("origin"), dict):
            continue
        # A quoted original remains that original kind; model prose is never a new source.
        # 为什么要有它：跨轮保留引文不能把旧模型摘要或混合归档升级为新的独立观察。
        add(quote["origin"]["kind"], ["messages", index, "content", "current_task_state", "semantic_work_state",
            "evidence_quotes", quote_index, "quote"], "Retained original quote " + quote["id"],
            {"retained_quote": quote["id"]}, quote["origin"])
    return sources


def with_semantic_evidence(request):
    document = _document(request)
    index, current = _current(document)
    current.pop("semantic_evidence_catalog", None)
    current["semantic_evidence_catalog"] = _sources(document)
    messages = list(request.messages)
    messages[index] = replace(messages[index], content={"current_task_state": current})
    return replace(request, messages=tuple(messages))


def _historical(source, boundary):
    return source.get("mixed_historical_text") is True or any(
        key in source and source[key] != boundary.get(key) for key in _BOUNDARY)


def semantic_state_view(working, current_boundary):
    semantic = (working or {}).get("semantic_state") or {}
    selected = next((row for row in semantic.get("snapshots", []) if row["id"] == semantic.get("latest_snapshot_id")), None)
    boundary = {key: deepcopy(current_boundary.get(key)) for key in _BOUNDARY}
    origin = selected["latest_origin"] if selected else None
    quotes = deepcopy(selected["evidence_quotes"]) if selected else []
    for quote in quotes:
        quote["freshness"] = "historical" if _historical(quote["origin"]["source_boundary"], boundary) else "current"
    freshness = ("unavailable" if selected is None else "historical" if
        _historical(origin["source_boundary"], boundary) or any(row["freshness"] == "historical" for row in quotes) else "current")
    return {"schema": SCHEMA, "origin": "model_authored_working_state", "status": "available" if selected else "not_expressed",
        "authority": "advisory_only", "snapshot_id": selected["id"] if selected else None,
        "state": deepcopy(selected["state"]) if selected else None, "authored_boundary": deepcopy(origin["source_boundary"]) if selected else None,
        "current_boundary": boundary, "freshness": freshness, "evidence_quotes": quotes,
        "latest_update": deepcopy(semantic.get("events", [None])[-1]) if semantic.get("events") else None,
        "snapshot_count": len(semantic.get("snapshots", [])),
        "interpretation": "All statements and hypothesis statuses are model judgments. Quotes prove visible wording only; history_excerpt may contain model statements. Historical evidence is not an edit preimage or a current passing check."}


def _empty(state):
    return state is None or state == {} or isinstance(state, dict) and set(state) == {
        "observations", "hypotheses", "open_questions", "next_action"} and not any(state.get(key) for key in
        ("observations", "hypotheses", "open_questions")) and isinstance(state.get("next_action"), dict) and not any(state["next_action"].values())


def record_semantic_state(working, response, *, actual_request, request_ref, response_ref):
    """Record one native decision's advisory update, without rejecting its action.

    为什么要有它：坏摘要不应变成修复门禁；响应原文和预算仍归既有执行owner保存、计量和恢复。
    """
    value = deepcopy(working or {})
    semantic = value.setdefault("semantic_state", {"schema": SCHEMA, "snapshots": [], "events": [], "latest_snapshot_id": None})
    request, reply = _document(actual_request), _document(response)
    identity = canonical_hash({"request_ref": request_ref, "response_ref": response_ref})
    if any(row["response_identity"] == identity for row in semantic["events"]):
        return value
    event = {"response_identity": identity, "request_ref": deepcopy(request_ref), "response_ref": deepcopy(response_ref),
        "status": "missing", "call_ids": [], "diagnostics": []}
    proposed = []
    for call in reply.get("tool_calls", []):
        arguments = call.get("arguments")
        if isinstance(arguments, dict) and "working_state" in arguments:
            event["call_ids"].append(call["call_id"])
            proposed.append(arguments["working_state"])
    if not proposed:
        semantic["events"].append(event)
        return value
    nonempty = [row for row in proposed if not _empty(row)]
    if not nonempty:
        event["status"] = "empty"
        semantic["events"].append(event)
        return value
    try:
        if len({canonical_hash(row) for row in nonempty}) != 1:
            event["status"] = "conflicting"
        else:
            state = nonempty[0]
            if list(Draft202012Validator(semantic_contract()).iter_errors(state)):
                raise ValueError("metadata_schema_invalid")
            hypothesis_ids = [row["id"] for row in state["hypotheses"]]
            if len(set(hypothesis_ids)) != len(hypothesis_ids):
                raise ValueError("hypothesis_identity_duplicated")
            if any(row["status"] != "open" and not row["evidence"] for row in state["hypotheses"]):
                raise ValueError("hypothesis_judgment_without_evidence")
            _, current = _current(request)
            boundary = _boundary(current)
            shown = current.get("semantic_work_state")
            if shown is not None and shown != semantic_state_view(working, boundary):
                raise ValueError("prior_model_view_not_verified")
            sources = _sources(request)
            if current.get("semantic_evidence_catalog") != sources:
                raise ValueError("evidence_catalog_not_current_visible_values")
            catalog = {row["id"]: row for row in sources}
            evidence = []
            for row in [*state["observations"], *state["hypotheses"]]:
                for citation in row["evidence"]:
                    source = catalog.get(citation["id"])
                    if source is None:
                        raise ValueError("evidence_not_visible")
                    visible = _at(request, source["locator"])
                    quote = citation["quote"]
                    if not quote.strip() or not _quote_seen(visible, quote):
                        raise ValueError("quote_not_visible")
                    appearance = {"request_ref": deepcopy(request_ref), "response_ref": deepcopy(response_ref),
                        "locator": deepcopy(source["locator"])}
                    origin = deepcopy(source.get("retained_origin")) or {**appearance,
                        "kind": source["kind"], "value_sha256": source["value_sha256"],
                        "source_boundary": deepcopy(source["source_boundary"])}
                    identifier = "q_" + canonical_hash({"quote": quote, "origin": origin})[:24]
                    if not any(item["id"] == identifier for item in evidence):
                        evidence.append({"id": identifier, "quote": quote, "origin": origin, "appearance": appearance})
            snapshot_id = canonical_hash(state)
            snapshot = next((row for row in semantic["snapshots"] if row["id"] == snapshot_id), None)
            occurrence = {"request_ref": deepcopy(request_ref), "response_ref": deepcopy(response_ref), "source_boundary": boundary}
            event.update(status="reaffirmed" if snapshot else "accepted", snapshot_id=snapshot_id)
            if snapshot is None:
                snapshot = {"id": snapshot_id, "state": deepcopy(state), "first_origin": occurrence, "occurrences": []}
                semantic["snapshots"].append(snapshot)
            snapshot.update(latest_origin=occurrence, evidence_quotes=evidence)
            snapshot["occurrences"].append(occurrence)
            semantic["latest_snapshot_id"] = snapshot_id
    except (ValueError, TypeError, KeyError, IndexError) as error:
        event.update(status="invalid", diagnostics=[str(error) if isinstance(error, ValueError) else type(error).__name__])
    semantic["events"].append(event)
    return value


def validate_semantic_state(working, turns):
    """Recompute state from authenticated request/response pairs, never from checkpoint claims."""
    expected = {}
    for turn in turns:
        expected = record_semantic_state(expected, turn["response"], actual_request=turn["request"],
            request_ref=turn["request_ref"], response_ref=turn["response_ref"])
    observed = (working or {}).get("semantic_state")
    if observed != expected.get("semantic_state"):
        raise ValueError("TASK_AGENT_SEMANTIC_STATE_NOT_OBSERVED")
