"""Bound the model's history projection, not the authoritative execution history."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace

from src.libs.llm.agent_turn import (
    AgentToolCall,
    AgentTurnProviderError,
    _chat_message,
    _chat_tool,
    _json_text,
    _parse_arguments,
    _parse_ollama_tool_call,
    _tool_exchanges,
)
from src.libs.llm.ollama_llm import build_ollama_context_admission
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError


def _text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def native_tool_calls(native):
    """Read actual native call identities without rewriting the retained provider state.

    为什么要有它：Responses用有序output items，Ollama用message对象；统一观察不能补造call_id或把原生历史改成另一协议。
    """
    if native is None:
        return ()
    if isinstance(native, dict):
        raw = native.get("tool_calls", [])
        if not isinstance(raw, list):
            raise AgentTurnProviderError("TASK_AGENT_NATIVE_CALLS_INVALID")
        calls = tuple(_parse_ollama_tool_call(item) for item in raw)
    elif isinstance(native, list):
        if not all(isinstance(item, dict) for item in native):
            raise AgentTurnProviderError("TASK_AGENT_NATIVE_ITEMS_INVALID")
        calls = []
        for item in native:
            if item.get("type") != "function_call":
                continue
            call_id, name = item.get("call_id"), item.get("name")
            if (not isinstance(call_id, str) or not call_id.strip() or not isinstance(name, str) or not name.strip()
                    or not isinstance(item.get("arguments"), str)):
                raise AgentTurnProviderError("TASK_AGENT_NATIVE_CALL_INVALID")
            calls.append(AgentToolCall(call_id, name, _parse_arguments(item["arguments"])))
        calls = tuple(calls)
    else:
        raise AgentTurnProviderError("TASK_AGENT_NATIVE_STATE_INVALID")
    if len({call.call_id for call in calls}) != len(calls):
        raise AgentTurnProviderError("TASK_AGENT_NATIVE_CALL_ID_DUPLICATED")
    return calls


def native_assistant_text(native):
    """Extract public output text only; reasoning and encrypted items remain opaque.

    为什么要有它：保持原生continuation不等于公开隐藏推理，工作陈述只来自模型实际输出的文字。
    """
    if isinstance(native, dict):
        return native.get("content") if isinstance(native.get("content"), str) else None
    if not isinstance(native, list):
        return None
    parts = [str(part.get("text") or "") for item in native
        if isinstance(item, dict) and item.get("type") == "message" and isinstance(item.get("content"), list)
        for part in item["content"] if isinstance(part, dict) and part.get("type") == "output_text"]
    return "".join(parts) or None


def investigation_observations(request, current_revision_ref, *, max_views=8):
    """Count returned immutable read views, never inferred investigation progress.

    为什么要有它：有限上下文可以省略旧正文，但不应让重复读取伪装成新的观察；是否继续重读仍由模型决定。
    """
    views, seen_calls, completed = {}, set(), 0
    exchanges = [*request.historical_tool_exchanges]
    results = [row for exchange in exchanges for row in exchange.tool_results] + list(request.tool_results)
    for result in results:
        output = result.output
        if (result.name not in {"source_read", "candidate_read"} or not isinstance(output, dict)
                or output.get("execution") != "executed"):
            continue
        value = output.get("value")
        if (not isinstance(value, dict) or value.get("schema") != "task-agent-source-view.v2"
                or value.get("parent_revision_ref") != current_revision_ref
                or not isinstance(value.get("source_view_ref"), str)):
            continue
        call = (result.name, result.call_id)
        if call in seen_calls:
            continue
        seen_calls.add(call)
        completed += 1
        reference = value["source_view_ref"]
        fact = views.pop(reference, None)
        if fact is None:
            fact = {key: deepcopy(value[key]) for key in ("source_view_ref", "parent_revision_ref", "root_id", "path",
                "file_sha256", "offset", "next_offset", "total_chars", "read_window") if key in value}
            fact.update(first_call_id=result.call_id, returned_count=0)
        fact.update(last_call_id=result.call_id, returned_count=fact["returned_count"] + 1)
        views[reference] = fact
    if not completed:
        return None
    selected = list(views.values())[-max_views:]
    return {"scope": "executed_current_revision_source_reads_only", "current_revision_ref": current_revision_ref,
        "completed_read_returns": completed, "distinct_returned_views": len(views),
        "repeated_view_returns": completed - len(views), "recent_views": selected,
        "views_truncated": len(selected) < len(views),
        "meaning": "Counts exact immutable returned views, not model understanding, novel knowledge or task progress. "
            "Re-reading may reconfirm facts but does not expose a later page. No action is required or forbidden by this index."}


def _action_facts(exchanges):
    facts = []
    total = sum(len(exchange["tool_results"]) for exchange in exchanges)
    for exchange_index in range(len(exchanges) - 1, -1, -1):
        exchange = exchanges[exchange_index]
        try:
            calls = {call.call_id: call for call in native_tool_calls(exchange.get("provider_state"))}
        except (AgentTurnProviderError, ValueError):
            calls = {}
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


def project_history(request, state, *, history_bytes=None, context_window=None, output_budget=None, provider=None,
                    decorate=None):
    """Keep whole recent exchanges and the last public check; archive older ones.

    为什么要有它：已闭合原生调用可整体省略，但不能拆散请求响应、删执行事实，或把摘要当成模型当时的推理。
    """
    if history_bytes is None:
        return _project_configured_history(request, state, context_window=context_window,
            output_budget=output_budget, provider=provider, decorate=decorate)
    if type(history_bytes) is not int or history_bytes < 0:
        raise ValueError("TASK_AGENT_HISTORY_BUDGET_INVALID")
    history = request.historical_tool_exchanges
    latest_check = next((index for index in range(len(history) - 1, -1, -1)
        if any(row.name == "candidate_check" for row in history[index].tool_results)), None)
    retained = protected_history_indices(history, current_revision_ref=_request_revision(request, state),
        active_results=request.tool_results)
    used = len(_text([asdict(row) for row in request.tool_results]).encode("utf-8"))
    used += sum(len(_text(asdict(history[index])).encode("utf-8")) for index in retained)
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
        return decorate(request) if decorate is not None else request, None
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
    projected = replace(request, historical_tool_exchanges=tuple(exchange for index, exchange in enumerate(history)
        if index in retained))
    return decorate(projected) if decorate is not None else projected, facts


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
    return {"technical_plan_ref": reference, "origin": "model_authored_advisory_plan", "text": text,
        "truncated": False, "total_chars": len(text), "grants_authority": False}


def _provider_payload(request, *, context_window, output_budget, provider):
    """Use the adapter's native message helpers for the same pre-call budget projection."""
    builder = getattr(provider, "_request_payload", None)
    if callable(builder):
        payload = builder(request)
        if not isinstance(payload, dict):
            raise AgentTurnProviderError("TASK_AGENT_NATIVE_PAYLOAD_INVALID")
        return payload
    if getattr(provider, "provider", "ollama_chat_local") != "ollama_chat_local":
        raise AgentTurnProviderError("TASK_AGENT_NATIVE_PAYLOAD_UNAVAILABLE")
    messages = [_chat_message(row) for row in request.messages]
    for recorded in _tool_exchanges(request, len(messages)):
        native = recorded.provider_state
        if not isinstance(native, dict) or not isinstance(native.get("tool_calls"), list):
            raise AgentTurnProviderError("OLLAMA_TOOL_RESULT_STATE_MISSING")
        exchange = [{"role": "assistant", "content": native.get("content") or "",
            "tool_calls": native["tool_calls"]}]
        exchange.extend({"role": "tool", "tool_name": result.name, "content": _json_text(result.output)}
            for result in recorded.tool_results)
        messages[recorded.insert_index:recorded.insert_index] = exchange
    payload = {"model": getattr(provider, "model", "configured-model"), "messages": messages,
        "stream": False, "think": False, "options": {"temperature": request.temperature,
            "num_predict": output_budget, "num_ctx": context_window, "seed": getattr(provider, "seed", 0)}}
    if request.tools:
        payload["tools"] = [_chat_tool(tool) for tool in request.tools]
    if request.response_format is not None:
        payload["format"] = request.response_format.schema
    elif request.tool_choice == "required" and len(request.tools) == 1:
        payload["format"] = request.tools[0].parameters
    return payload


def with_history_projection(request, facts):
    messages = []
    for message in request.messages:
        content = message.content
        if message.role == "user" and isinstance(content, dict) and set(content) == {"current_task_state"}:
            content = deepcopy(content)
            content["current_task_state"]["context_budget"] = deepcopy(facts)
            content["current_task_state"]["history_projection"] = deepcopy(facts) if facts.get("history_ref") else None
            message = replace(message, content=content)
        messages.append(message)
    return replace(request, messages=tuple(messages))


def _request_revision(request, state=None):
    for message in reversed(request.messages):
        if isinstance(message.content, dict) and isinstance(message.content.get("current_task_state"), dict):
            return message.content["current_task_state"].get("current_revision_ref")
    return getattr(state, "current_revision_ref", None)


def protected_history_indices(history, *, current_revision_ref=None, active_results=()):
    """Identify native exchanges that projection may not omit.

    为什么要有它：正常投影和恢复证明必须共用未闭合调用与公开反馈的保护条件，不能各自形成不同门禁。
    """
    protected = set()
    # 为什么要有它：活动代码窗口和最近成功编辑比旧旁白更接近下一次决策；仍只保护实际返回的完整调用/输出组。
    def current_work(result, kind):
        output = result.output if isinstance(result.output, dict) else {}
        value = output.get("value")
        if output.get("execution") != "executed" or not isinstance(value, dict):
            return False
        if kind == "source":
            return (result.name in {"source_read", "candidate_read"} and value.get("schema") == "task-agent-source-view.v2"
                and isinstance(value.get("source_view_ref"), str)
                and (current_revision_ref is None or value.get("parent_revision_ref") == current_revision_ref))
        return (result.name in {"candidate_submit", "candidate_edit"} and isinstance(value.get("candidate_ref"), str)
            and (current_revision_ref is None or value["candidate_ref"] == current_revision_ref))

    for kind in ("source", "edit"):
        if not any(current_work(result, kind) for result in active_results):
            latest = next((index for index in range(len(history) - 1, -1, -1)
                if any(current_work(result, kind) for result in history[index].tool_results)), None)
            if latest is not None:
                protected.add(latest)
    for wanted in ("candidate_check", "check_feedback_read", "feedback_read"):
        latest = next((index for index in range(len(history) - 1, -1, -1)
            if any(row.name == wanted for row in history[index].tool_results)), None)
        if latest is not None:
            protected.add(latest)
    for index in range(len(history) - 1, -1, -1):
        failed = False
        for row in history[index].tool_results:
            output = row.output if isinstance(row.output, dict) else {}
            value = output.get("value", output)
            if row.name == "candidate_check" and isinstance(value, dict) and value.get("status") not in {None, "passed"}:
                failed = True
        if failed:
            protected.add(index)
            break
    for index, exchange in enumerate(history):
        try:
            expected = {call.call_id for call in native_tool_calls(exchange.provider_state)}
        except (AgentTurnProviderError, ValueError):
            protected.add(index)
            continue
        if expected != {row.call_id for row in exchange.tool_results}:
            protected.add(index)
    return protected


def _project_configured_history(request, state, *, context_window, output_budget, provider, decorate=None):
    """Fit complete closed exchanges against the real provider limit, not a fixed byte tail.

    为什么要有它：工具、任务状态和输出预留共同占用窗口；预算不足必须披露，不能静默丢失当前调用或公开反例。
    """
    context_window = getattr(provider, "context_window", None) or context_window
    output_budget = getattr(provider, "max_output_tokens", None) or output_budget
    if (type(context_window) is not int or type(output_budget) is not int
            or output_budget <= 0 or context_window <= output_budget):
        raise AgentTurnProviderError("TASK_AGENT_CONTEXT_LIMIT_UNAVAILABLE", diagnostic={
            "failure_boundary": "context_projection", "configured_num_ctx": context_window,
            "num_predict": output_budget, "transport_attempted": False})
    history = request.historical_tool_exchanges
    protected = protected_history_indices(history, current_revision_ref=_request_revision(request, state),
        active_results=request.tool_results)
    retained = set(range(len(history)))
    removable = [index for index in range(len(history)) if index not in protected]
    estimator = getattr(provider, "_context_admission", None)

    def decorated(candidate):
        # 为什么要有它：引用目录必须随实际保留的历史重建并计入容量，不能继续声称已归档正文仍在本轮输入中。
        return decorate(candidate) if decorate is not None else candidate

    def measure(candidate):
        payload = _provider_payload(candidate, context_window=context_window, output_budget=output_budget, provider=provider)
        return (estimator(payload) if callable(estimator) else build_ollama_context_admission(
            payload=payload, configured_num_ctx=context_window, num_predict=output_budget))

    statements_omitted = any(isinstance(message.content, dict)
        and ((message.content.get("current_task_state") or {}).get("working_state") or {}).get(
            "model_statement_projection", {}).get("omitted_statement_count", 0) > 0 for message in request.messages)
    facts = {"schema": "task-agent-history-projection.v2", "origin": "execution_host",
        "projection_policy": "current_work_before_archived_narrative_v1", "history_ref": None,
        "archived_exchange_count": 0, "retained_exchange_count": len(history) + int(bool(request.tool_results)),
        "protected_history_indices": sorted(protected), "active_exchange_preserved": True,
        "model_statements_preserved": not statements_omitted, "model_statement_originals_archived": statements_omitted,
        "configured_num_ctx": context_window, "output_token_reserve": output_budget,
        "exact_prompt_tokens_available_pre_call": False,
        "meaning": "Omit archived older raw statements before closed historical exchanges. Preserve current source/edit, "
            "latest public check/failure and complete native call/output groups; history_read returns original public data."}
    projected_messages = request.messages
    while True:
        facts.update(archived_exchange_count=len(history) - len(retained),
            retained_exchange_count=len(retained) + int(bool(request.tool_results)))
        # The final reference has bounded content-addressed length; account for it before saving.
        facts["history_ref"] = "context-archive/" + "0" * 192 if len(retained) < len(history) else None
        candidate = replace(request, messages=projected_messages,
            historical_tool_exchanges=tuple(row for index, row in enumerate(history) if index in retained))
        facts["context_admission"] = measure(decorated(with_history_projection(candidate, facts)))
        checked = decorated(with_history_projection(candidate, facts))
        admission = measure(checked)
        if admission.get("admitted") is True:
            break
        compacted = _omit_oldest_statement(projected_messages)
        if compacted is not None:
            projected_messages = compacted
            facts["model_statements_preserved"] = False
            facts["model_statement_originals_archived"] = True
            continue
        if not removable:
            raise AgentTurnProviderError("TASK_AGENT_PROTECTED_CONTEXT_EXCEEDS_LIMIT", diagnostic={
                "failure_boundary": "context_projection", "context_admission": admission,
                "protected_history_indices": sorted(protected), "active_exchange_preserved": True,
                "model_statements_preserved": facts["model_statements_preserved"], "transport_attempted": False})
        retained.remove(removable.pop(0))
    archived = [asdict(exchange) for index, exchange in enumerate(history) if index not in retained]
    facts["history_ref"] = state.save("context-archive", {"schema": "task-agent-context-archive.v2",
        "task_contract_ref": state.contract_ref, "exchanges": archived}) if archived else None
    final = decorated(with_history_projection(candidate, facts))
    final_admission = measure(final)
    if final_admission.get("admitted") is not True:
        raise AgentTurnProviderError("TASK_AGENT_FINAL_CONTEXT_EXCEEDS_LIMIT", diagnostic={
            "context_admission": final_admission, "transport_attempted": False})
    return final, facts


def _omit_oldest_statement(messages):
    """Remove one non-anchor raw statement from input, never from its public archive.

    为什么要有它：阅读旁白不能自动替代先前工作判断；时间锚点仅保留原文，不冒称理解哪些假设仍然成立。
    """
    for index, message in enumerate(messages):
        content = message.content
        if not isinstance(content, dict) or set(content) != {"current_task_state"}:
            continue
        working = content["current_task_state"].get("working_state") or {}
        rows = working.get("model_statements", [])
        projection = working.get("model_statement_projection") or {}
        if not projection.get("archive_ref") or len(rows) <= 1:
            return None
        boundary = working.get("boundary")
        latest = next((row for row in rows
            if row["content_sha256"] == working.get("latest_model_statement_sha256")),
            max(rows, key=lambda row: row.get("last_observed_sequence", row["statement_index"])))
        victim = next((row for row in rows if row["statement_index"] != latest["statement_index"]), None)
        if victim is None:
            return None
        changed = deepcopy(content)
        view = changed["current_task_state"]["working_state"]
        view["model_statements"] = [row for row in view["model_statements"]
            if row["statement_index"] != victim["statement_index"]]
        selected = view["model_statement_projection"]
        selected["retained_statement_indices"] = [row["statement_index"] for row in view["model_statements"]]
        selected.pop("omitted_statement_indices", None)
        selected["omitted_statement_count"] = selected["total_statements"] - len(view["model_statements"])
        selected["current_boundary_statement_available"] = any(row.get("source_boundary") == boundary for row in view["model_statements"])
        selected["latest_statement_sha256"] = latest["content_sha256"]
        selected["selection_basis"] = ("Budgeted raw statements: actual latest occurrence retained; "
            + ("A statement at the current recorded boundary is retained. " if selected["current_boundary_statement_available"]
                else "No statement at the current boundary exists; prior statements are historical. ")
            + "No semantic hypothesis status is inferred.")
        return (*messages[:index], replace(message, content=changed), *messages[index + 1:])
    return None


def validate_projected_request(original, prepared, state):
    """Prove a prepared request is an exact projection of one retained native request.

    为什么要有它：恢复可以接受有据可查的闭合历史省略，但不能改变活动调用、工具、参数或伪造归档来注入新历史。
    """
    from src.mod_agent.interrupted_patch_admission import retained_request_from_document

    def require(condition, reason):
        if not condition:
            raise ValueError("TASK_AGENT_CONTEXT_PROJECTION_INVALID:" + reason)

    require(isinstance(original, dict) and isinstance(prepared, dict)
        and set(original) == set(prepared), "request_fields")
    require(all(prepared[key] == original[key] for key in original
        if key not in {"messages", "historical_tool_exchanges"}), "native_fields_changed")
    require(isinstance(prepared["messages"], list) and isinstance(original["messages"], list)
        and prepared["messages"][:len(original["messages"])] == original["messages"], "message_prefix")
    additions = prepared["messages"][len(original["messages"]):]
    # Scope: exactly the host-state message, then its optional strategy projection.
    # 为什么要有它：找到合法状态消息不代表其旁边的任意文字也有来源；恢复不得接受额外或重复的指导消息。
    require(all(isinstance(row, dict) and set(row) == {"role", "content"} and row["role"] == "user"
        and isinstance(row["content"], dict) for row in additions), "host_message_shape")
    require(len(additions) in {1, 2} and set(additions[0]["content"]) == {"current_task_state"},
        "host_message_sequence")
    if len(additions) == 2:
        require(set(additions[1]["content"]) == {"strategy_experience"}
            and isinstance(additions[1]["content"]["strategy_experience"], dict), "host_message_strategy")
    current = additions[0]["content"]["current_task_state"]
    require(isinstance(current, dict), "current_task_state_missing_or_ambiguous")
    require(current.get("origin") == "execution_host" and current.get("task_contract_ref") == state.contract_ref,
        "task_state_origin")
    history, selected = original["historical_tool_exchanges"], prepared["historical_tool_exchanges"]
    require(isinstance(history, list) and isinstance(selected, list), "history_shape")
    indices, cursor = [], 0
    for exchange in selected:
        index = next((index for index in range(cursor, len(history)) if history[index] == exchange), None)
        require(index is not None, "history_not_ordered_original_subset")
        indices.append(index)
        cursor = index + 1
    original_request = retained_request_from_document(original)
    protected = protected_history_indices(original_request.historical_tool_exchanges,
        current_revision_ref=current.get("current_revision_ref"), active_results=original_request.tool_results)
    require(protected.issubset(indices), "protected_exchange_omitted")
    omitted = [row for index, row in enumerate(history) if index not in indices]
    projection = current.get("history_projection")
    if omitted:
        require(isinstance(projection, dict) and isinstance(projection.get("history_ref"), str), "archive_missing")
        archive = state.load(projection["history_ref"])
        require(archive.get("schema") == "task-agent-context-archive.v2"
            and archive.get("task_contract_ref") == state.contract_ref
            and archive.get("exchanges") == omitted, "archive_not_exact_omitted_history")
        require(type(projection.get("archived_exchange_count")) is int
            and projection["archived_exchange_count"] == len(omitted)
            and type(projection.get("retained_exchange_count")) is int
            and projection["retained_exchange_count"] == len(selected) + int(bool(original["tool_results"])),
            "projection_count_changed")
    else:
        require(projection is None or isinstance(projection, dict)
            and projection.get("history_ref") is None and projection.get("archived_exchange_count") == 0,
            "unexpected_archive_claim")
    return {"current_task_state": current, "retained_history_indices": indices,
        "omitted_exchange_count": len(omitted)}
