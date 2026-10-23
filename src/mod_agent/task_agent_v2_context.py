"""Bound the model's history projection, not the authoritative execution history."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError

HISTORY_BYTES = 12000


def _text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    facts = {"schema": "task-agent-history-projection.v2", "origin": "execution_host",
        "history_ref": reference, "archived_exchange_count": len(archived),
        "retained_exchange_count": len(retained) + int(bool(request.tool_results)),
        "retained_history_bytes": used, "history_byte_target": history_bytes,
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
