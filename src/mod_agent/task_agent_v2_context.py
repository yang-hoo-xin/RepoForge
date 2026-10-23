"""Bound the model's history projection, not the authoritative execution history."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

from src.libs.llm.agent_turn import AgentTurnProviderError, _parse_ollama_tool_call
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError

HISTORY_BYTES = 12000


def _text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _action_facts(exchanges):
    facts = []
    total = sum(len(exchange["tool_results"]) for exchange in exchanges)
    for exchange_index in range(len(exchanges) - 1, -1, -1):
        exchange = exchanges[exchange_index]
        calls = {}
        for raw in (exchange.get("provider_state") or {}).get("tool_calls", []):
            try:
                call = _parse_ollama_tool_call(raw)
                calls[call.call_id] = call
            except (AgentTurnProviderError, ValueError):
                continue
        for result_index in range(len(exchange["tool_results"]) - 1, -1, -1):
            result = exchange["tool_results"][result_index]
            output = result["output"] if isinstance(result["output"], dict) else {}
            value = output.get("value", output)
            value = value if isinstance(value, dict) else {}
            call = calls.get(result["call_id"])
            matched = call is not None and call.name == result["name"]
            arguments = {key: val for key, val in (call.arguments if matched else {}).items()
                if key in {"query", "path", "root_id", "offset", "limit", "candidate_ref", "source_revision_ref"}}
            clipped = [key for key, val in arguments.items() if isinstance(val, str) and len(val) > 256]
            fact = {"call_id": result["call_id"], "name": result["name"],
                "archive_exchange_index": exchange_index, "tool_result_index": result_index,
                "arguments_status": "exact_native_call_match" if matched else "unavailable",
                "arguments": {key: val[:256] if key in clipped else val for key, val in arguments.items()},
                "arguments_truncated_fields": clipped,
                "observation": {key: value[key] for key in ("total_matches", "truncated", "status", "executed",
                    "candidate_ref", "check_ref") if key in value}, "execution": output.get("execution")}
            if len(facts) == 12 or len(_text([fact, *facts]).encode("utf-8")) > 4000:
                return facts, total
            facts.insert(0, fact)
    return facts, total


def project_history(request, state, *, history_bytes=HISTORY_BYTES):
    """Keep whole recent exchanges and the last public check; archive older ones.

    为什么要有它：已闭合原生调用可整体省略，但不能拆散请求响应、删执行事实，或把摘要当成模型当时的推理。
    """
    history = request.historical_tool_exchanges
    latest_check = next((index for index in range(len(history) - 1, -1, -1)
        if any(row.name == "candidate_check" for row in history[index].tool_results)), None)
    retained = set()
    used = len(_text([asdict(row) for row in request.tool_results]).encode("utf-8"))
    if latest_check is not None:
        retained.add(latest_check)
        used += len(_text(asdict(history[latest_check])).encode("utf-8"))
    # Always retain the active exchange; never synthesize its provider state or tool result.
    for index in range(len(history) - 1, -1, -1):
        if index in retained:
            continue
        size = len(_text(asdict(history[index])).encode("utf-8"))
        if used + size <= history_bytes:
            retained.add(index)
            used += size
        else:
            break
    archived = [asdict(exchange) for index, exchange in enumerate(history) if index not in retained]
    if not archived:
        return request, None
    reference = state.save("context-archive", {"schema": "task-agent-context-archive.v2",
        "task_contract_ref": state.contract_ref, "exchanges": archived})
    action_facts, archived_actions = _action_facts(archived)
    facts = {"schema": "task-agent-history-projection.v2", "origin": "execution_host",
        "history_ref": reference, "archived_exchange_count": len(archived),
        "retained_exchange_count": len(retained) + int(bool(request.tool_results)),
        "retained_history_bytes": used, "history_byte_target": history_bytes,
        "archived_action_facts": action_facts, "archived_action_facts_truncated": len(action_facts) < archived_actions,
        "action_facts_scope": "Bounded index of exact requested arguments and recorded results, not advice or inferred hypotheses.",
        "latest_public_check_retained": latest_check is not None or any(row.name == "candidate_check" for row in request.tool_results),
        "meaning": "Older closed exchanges omitted from this input only. history_read can retrieve their exact public data. "
            "This is not a model reasoning summary. Current source must be read at its exact revision before editing."}
    return replace(request, historical_tool_exchanges=tuple(exchange for index, exchange in enumerate(history)
        if index in retained)), facts


def history_text(state, reference):
    archive = state.load(reference)
    if (archive.get("schema") != "task-agent-context-archive.v2"
            or archive.get("task_contract_ref") != state.contract_ref or not isinstance(archive.get("exchanges"), list)):
        raise TaskAgentV2StateError("history_reference_not_public_task_archive")
    return _text(archive["exchanges"])


def model_plan_view(state):
    reference = state.head["plan_ref"]
    if reference is None:
        return None
    plan = state.load(reference)
    if plan.get("schema") != "task-agent-technical-plan.v2" or plan.get("contract_ref") != state.contract_ref:
        raise TaskAgentV2StateError("technical_plan_contract_mismatch")
    text = _text(plan["plan"])
    return {"technical_plan_ref": reference, "origin": "model_authored_advisory_plan", "text": text[:4000],
        "truncated": len(text) > 4000, "total_chars": len(text), "grants_authority": False}
