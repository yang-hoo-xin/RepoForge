"""One continuous model session over explicitly supplied task tool owners.

Scope: dispatch, native history and shared-budget accounting, never task authority.
为什么要有它：模型可选择下一动作，但源码范围、候选版本、执行权限和最终验收仍由真实owner核准。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Any

from src.core.trace.execution_trace import trace_operation
from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolCall,
    AgentToolDefinition,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnClient,
    AgentTurnRequest,
    AgentTurnResponse,
    AgentUsage,
)
from src.mod_agent.model_call_budget import call_allowance, validate_budget

_ROLES = frozenset({"source_read", "source_search", "upstream", "candidate_submit", "candidate_check", "finish"})
_STOPPED = frozenset({"length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal", "failed", "cancelled"})
_CORRECTION_CODES = frozenset({"TASK_AGENT_SOURCE_ARGUMENTS_INVALID", "TASK_AGENT_MIXED_TOOL_BATCH",
    "TASK_AGENT_READ_BUDGET_EXCEEDED", "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"})
_OUTPUT_CORRECTION = "task_agent_output_correction"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def measured_turn_usage(response):
    """Keep missing provider measurements unknown instead of charging dataclass zeros.

    为什么要有它：未知token数与零token不同；同一纯函数供正常执行、Trace和恢复读取既有测量标记。
    """
    value = asdict(response) if isinstance(response, AgentTurnResponse) else response
    diagnostic = value.get("provider_diagnostic") or {}
    if "measured_usage" in diagnostic:
        return deepcopy(diagnostic["measured_usage"])
    if value.get("provider") == "openai_responses":
        return None
    return AgentUsage(**value["usage"]).to_dict()


def _output_correction_in_history(messages):
    """Recover the one host correction from its unchanged preceding assistant message.

    为什么要有它：中断后不能重新领取纠正机会，也不能把模型自己写的完成声明当成宿主状态。
    """
    found = 0
    for index, message in enumerate(messages):
        content = message.content
        if not isinstance(content, Mapping) or content.get("message_type") != _OUTPUT_CORRECTION:
            continue
        if (message.role != "user" or content.get("schema_version") != "task-agent-output-correction.v1"
                or content.get("origin") != "execution_host" or type(content.get("correction_number")) is not int
                or content["correction_number"] != 1
                or content.get("code") != "TASK_AGENT_FINISH_TOOL_REQUIRED" or index == 0
                or messages[index - 1].role != "assistant"
                or content.get("assistant_message_sha256") != _hash(asdict(messages[index - 1]))):
            raise ValueError("TASK_AGENT_OUTPUT_CORRECTION_HISTORY_INVALID")
        found += 1
    if found > 1:
        raise ValueError("TASK_AGENT_OUTPUT_CORRECTION_HISTORY_INVALID")
    return bool(found)


class TaskActionInputError(ValueError):
    """A source-read owner may identify ordinary missing/type errors before dispatch."""

    def __init__(self, argument_errors: Sequence[str]):
        if not argument_errors or any(type(item) is not str or not item for item in argument_errors):
            raise ValueError("TASK_AGENT_ARGUMENT_FEEDBACK_INVALID")
        self.argument_errors = tuple(argument_errors)
        super().__init__("TASK_AGENT_SOURCE_ARGUMENTS_INVALID")


class TaskCandidateRevisionInputError(ValueError):
    """A candidate owner rejected correctable submission input before dispatch.

    Scope: the owner must first validate identity, parent version and edit authority.
    为什么要有它：格式错误或无变化修改可返回一次明确反馈，但不能把越权或版本冲突降级为可重试输入。
    """

    def __init__(self, argument_errors: Sequence[str]):
        if not argument_errors or any(type(item) is not str or not item for item in argument_errors):
            raise ValueError("TASK_AGENT_ARGUMENT_FEEDBACK_INVALID")
        self.argument_errors = tuple(argument_errors)
        super().__init__("TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID")


class TaskCandidateReadinessError(ValueError):
    """A check or handoff lacks a candidate or check, before any dispatch.

    为什么要有它：尚未产生的候选或检查不应耗尽参数纠错机会，也不能被记录成已运行的验收。
    """

    def __init__(self, argument_errors: Sequence[str]):
        if not argument_errors or any(item not in {"candidate_required", "current_check_required"}
                for item in argument_errors):
            raise ValueError("TASK_AGENT_ARGUMENT_FEEDBACK_INVALID")
        self.argument_errors = tuple(argument_errors)
        super().__init__("TASK_AGENT_CANDIDATE_NOT_READY")


def read_budget_view(*, used, limit):
    """Project an existing owner's counter; never allocate or reset an allowance."""
    if (type(used) is not int or used < 0
            or limit is not None and (type(limit) is not int or not used <= limit)):
        raise ValueError("TASK_AGENT_READ_BUDGET_SNAPSHOT_INVALID")
    return {"used": used, "limit": limit, "remaining": None if limit is None else limit - used,
        "count_unit": "one_source_read_or_source_search_call",
        **({"policy": "meter_only"} if limit is None else {})}


class TaskReadBudgetError(ValueError):
    """Trusted atomic refusal before any read charge or tool dispatch."""

    def __init__(self, *, requested, used, limit):
        current = read_budget_view(used=used, limit=limit)
        if limit is None or type(requested) is not int or requested <= current["remaining"]:
            raise ValueError("TASK_AGENT_READ_BUDGET_ERROR_INVALID")
        self.requested, self.used, self.limit = requested, used, limit
        super().__init__("TASK_AGENT_READ_BUDGET_EXCEEDED")

    def model_view(self):
        return read_budget_view(used=self.used, limit=self.limit)


@dataclass(frozen=True)
class PreparedTaskAction:
    value: Any
    read_cost: int
    version_identity: str | None = None


@dataclass(frozen=True)
class TaskAgentAction:
    definition: AgentToolDefinition
    role: str
    prepare: Callable[[dict[str, Any]], PreparedTaskAction]
    execute: Callable[[Any], Any]
    read_only: bool
    confirm_reuse: Callable[[PreparedTaskAction, Mapping[str, Any]], Mapping[str, Any] | None] | None = None


@dataclass(frozen=True)
class TaskAgentLoopResult:
    status: str
    termination: dict[str, Any]
    request: AgentTurnRequest
    model_budget: dict[str, Any]
    model_turns: int
    read_units: int
    observation_bytes: int
    tool_argument_correction_used: bool
    events: tuple[dict[str, Any], ...]
    usage_by_turn: tuple[dict[str, Any], ...]
    call_records: dict[str, dict[str, Any]]
    completion: dict[str, Any] | None = None
    unknown_effects: bool = False
    unadmitted_tool_results: tuple[AgentToolResult, ...] = ()
    output_correction_used: bool = False
    unadmitted_host_messages: tuple[AgentMessage, ...] = ()
    execution_history_requests: tuple[AgentTurnRequest, ...] = ()

    def to_dict(self):
        value = asdict(self)
        # Scope: local returned state; callers choose their existing private journal/ArtifactStore.
        # 为什么要有它：中断时必须保留原生工具交换，不能靠重建摘要重新获得调用机会或丢失未知执行。
        value["observation_accounting"] = "new_canonical_AgentToolResult_and_host_AgentMessage_utf8_bytes"
        return value


def run_task_agent_loop(*, client: AgentTurnClient, initial_request: AgentTurnRequest,
                        actions: Sequence[TaskAgentAction], reserve_model_call: Callable[[], Any],
                        budget_snapshot: Callable[[], dict[str, Any]],
                        charge_tool_budget: Callable[..., Any], completion_check: Callable[..., Mapping[str, Any]],
                        response_admission: Callable[[AgentTurnResponse], None] | None = None,
                        model_role: str = "patch", tool_argument_correction_used: bool = False,
                        prior_call_records: Mapping[str, Mapping[str, Any]] | None = None,
                        prior_execution_requests: Sequence[AgentTurnRequest] | None = None,
                        stop_check: Callable[[], Mapping[str, Any] | None] | None = None,
                        tool_definitions: Callable[[], Sequence[AgentToolDefinition]] | None = None,
                        record_event: Callable[[dict[str, Any]], None] | None = None,
                        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
                        resume_checkpoint: Mapping[str, Any] | None = None) -> TaskAgentLoopResult:
    """Let the model choose actions; every execution and result remains owner-bound.

    prepare/confirm_reuse/completion_check are trusted, read-only host checks.
    charge_tool_budget atomically refuses or charges its supplied nonnegative deltas.
    Upstream owners must use the same reserve_model_call/budget_snapshot for nested sends.
    为什么要有它：工具选择不产生新权限或隐藏预算；核心不重试未知副作用，也不自建第二份预算存储。
    """
    request = initial_request
    events, usage = [], []
    seen: dict[str, dict[str, Any]] = {}
    turns = reads = observation_bytes = 0
    argument_used = tool_argument_correction_used
    output_corrected = False
    execution_history = ()
    last_budget: dict[str, Any] = {}
    phase = "ready"
    pending_response = None
    reads_charged = observations_charged = False
    resume_pending = False
    pending_charge_replay = None

    def checkpoint(next_phase=None, *, terminal=None, completion=None):
        nonlocal phase
        if next_phase is not None:
            phase = next_phase
        if checkpoint_callback is not None:
            checkpoint_callback({"schema": "task-agent-loop-checkpoint.v2", "phase": phase,
                "request": asdict(request), "pending_response": asdict(pending_response) if pending_response else None,
                "reads_charged": reads_charged, "observations_charged": observations_charged,
                "pending_charge_replay": deepcopy(pending_charge_replay),
                "model_turns": turns, "read_units": reads, "observation_bytes": observation_bytes,
                "tool_argument_correction_used": argument_used, "output_correction_used": output_corrected,
                "events": deepcopy(events), "usage_by_turn": deepcopy(usage), "call_records": deepcopy(seen),
                "model_budget": snapshot(), "terminal": terminal, "completion": completion})

    def emit(value):
        event = _copy(value)
        events.append(event)
        if record_event is not None:
            record_event(deepcopy(event))
        if event.get("event") == "provider_send_reserved":
            checkpoint("provider_pending")
        elif event.get("event") == "tool_dispatch_started":
            checkpoint("tool_pending")
        elif event.get("event") == "tool_dispatch_completed":
            checkpoint("dispatch_ready")

    def snapshot():
        nonlocal last_budget
        current = validate_budget(budget_snapshot())
        if last_budget and (current["maximum"] != last_budget["maximum"]
                or current.get("policy", "stage_and_total") != last_budget.get("policy", "stage_and_total")
                or current["consumed"] < last_budget["consumed"]):
            raise ValueError("TASK_AGENT_SHARED_BUDGET_CHANGED")
        last_budget = current
        return deepcopy(last_budget)

    def stop(code, *, unknown=False, completion=None, unadmitted=(), host_unadmitted=(), detail=None):
        try:
            snapshot()
        except Exception:
            pass
        checkpoint(terminal={"code": code, "unknown_effects": unknown, "detail": detail}, completion=completion)
        return TaskAgentLoopResult("completed" if completion is not None else "blocked",
            {"code": code, **({"detail": detail} if detail is not None else {})}, request,
            deepcopy(last_budget), turns, reads, observation_bytes, argument_used,
            tuple(deepcopy(events)), tuple(deepcopy(usage)), deepcopy(seen), completion,
            unknown, tuple(unadmitted), output_corrected, tuple(host_unadmitted),
            tuple(deepcopy((*execution_history, request))))

    def charge(*, read_count=0, byte_count=0):
        nonlocal reads, observation_bytes, pending_charge_replay
        if pending_charge_replay is not None:
            if pending_charge_replay != {"reads": read_count, "observation_bytes": byte_count}:
                raise ValueError("TASK_AGENT_RECONCILED_CHARGE_CHANGED")
            pending_charge_replay = None
            return
        decision = charge_tool_budget(reads=read_count, observation_bytes=byte_count)
        if decision is False:
            raise ValueError("TASK_AGENT_TOOL_BUDGET_EXHAUSTED")
        reads += read_count
        observation_bytes += byte_count

    def retain(results, response):
        nonlocal request
        historical = request.historical_tool_exchanges
        if request.tool_results:
            historical = (*historical, AgentToolExchange(request.tool_results, request.provider_state,
                len(request.messages) if request.tool_result_insert_index is None else request.tool_result_insert_index))
        request = replace(request, tool_results=tuple(results), provider_state=response.provider_state,
            tool_result_insert_index=len(request.messages), historical_tool_exchanges=historical,
            response_format=None, tool_choice="auto", parallel_tool_calls=False)

    def retain_assistant_text(text, response):
        nonlocal request
        historical = request.historical_tool_exchanges
        if request.tool_results:
            historical = (*historical, AgentToolExchange(request.tool_results, request.provider_state,
                len(request.messages) if request.tool_result_insert_index is None else request.tool_result_insert_index))
        assistant = AgentMessage("assistant", text)
        if response.provider == "openai_responses":
            # 为什么要有它：纠正文字输出也必须保留原生opaque续接；提供方序列化时替换同位置的重复公开文本。
            historical = (*historical, AgentToolExchange((), deepcopy(response.provider_state), len(request.messages)))
        request = replace(request, messages=(*request.messages, assistant), tool_results=(), provider_state=None,
            tool_result_insert_index=None, historical_tool_exchanges=historical)
        return assistant

    try:
        if (not isinstance(initial_request, AgentTurnRequest) or type(argument_used) is not bool
                or prior_execution_requests is not None and (
                    not isinstance(prior_execution_requests, (list, tuple))
                    or any(not isinstance(item, AgentTurnRequest) for item in prior_execution_requests))
                or tool_definitions is not None and not callable(tool_definitions)):
            return stop("TASK_AGENT_INPUT_INVALID")
        execution_history = tuple(prior_execution_requests or ())
        if execution_history and (request.tool_results or request.historical_tool_exchanges):
            return stop("TASK_AGENT_EXECUTION_HISTORY_CONFLICT")
        # Scope: trusted owner may retain verified execution history without resending it as model context.
        # 为什么要有它：压缩或重建模型输入不能清空已执行动作、未知副作用和纠错状态；执行历史与模型可见上下文不是同一对象。
        binding_requests = (*execution_history, request)
        try:
            output_corrected = any([_output_correction_in_history(item.messages) for item in binding_requests])
        except ValueError:
            return stop("TASK_AGENT_OUTPUT_CORRECTION_HISTORY_INVALID")
        # Scope: a genuinely unknown action may have no tool result to bind yet.
        # 为什么要有它：缺失回执不能把未知副作用降为普通输入错误，重建上下文前必须由宿主核对实际执行。
        if execution_history and any(row.get("status") in {"pending", "unknown"}
                for row in (prior_call_records or {}).values()):
            seen.update(_copy(prior_call_records))
            return stop("TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE", unknown=True)
        registered = {}
        for action in actions:
            if (not isinstance(action, TaskAgentAction) or not isinstance(action.definition, AgentToolDefinition)
                    or not action.definition.name or action.definition.name in registered or action.role not in _ROLES
                    or type(action.read_only) is not bool or not callable(action.prepare) or not callable(action.execute)
                    or action.read_only and action.role not in {"source_read", "source_search", "finish"}
                    or action.role == "finish" and not action.read_only):
                return stop("TASK_AGENT_ACTION_CONFIGURATION_INVALID")
            registered[action.definition.name] = action
        missing = sorted(_ROLES - {action.role for action in registered.values()})
        if missing or not callable(completion_check):
            return stop("TASK_AGENT_CAPABILITIES_INCOMPLETE", detail={"missing_roles": missing})
        # Scope: describe the admitted read surface, without changing its costs or retry policy.
        # 为什么要有它：新增候选读取沿用原只读角色，纠错信息也必须准确列出实际可调用的只读工具。
        read_tool_names = [name for name, action in registered.items()
            if action.read_only and action.role in {"source_read", "source_search"}]
        request = replace(request, tools=tuple(deepcopy(item.definition) for item in registered.values()),
            parallel_tool_calls=False, tool_choice="auto", response_format=None)
        result_batches = [results for item in binding_requests
            for results in [*(exchange.tool_results for exchange in item.historical_tool_exchanges), item.tool_results]]
        historical_results = [result for results in result_batches for result in results]
        correction_exchanges = sum(any(isinstance(result.output, Mapping)
            and result.output.get("code") in _CORRECTION_CODES for result in results)
            for results in result_batches)
        if correction_exchanges > 1:
            return stop("TASK_AGENT_ARGUMENT_HISTORY_INVALID")
        argument_used = argument_used or bool(correction_exchanges)
        visible_results = {result.call_id: asdict(result) for result in historical_results}
        if execution_history and set(prior_call_records or {}) != set(visible_results):
            return stop("TASK_AGENT_PRIOR_CALL_BINDING_INVALID")
        for call_id, result in visible_results.items():
            seen[call_id] = {"name": result["name"], "arguments_sha256": None,
                             "status": "prior_result_unbound", "result": result}
        for call_id, record in (prior_call_records or {}).items():
            copied = _copy(record)
            if (call_id not in visible_results or copied.get("result") != visible_results[call_id]
                    or copied.get("name") != visible_results[call_id]["name"]
                    or type(copied.get("arguments_sha256")) is not str):
                return stop("TASK_AGENT_PRIOR_CALL_BINDING_INVALID")
            seen[call_id] = copied
        if execution_history:
            if any(row["status"] not in {"completed", "not_dispatched"} for row in seen.values()):
                return stop("TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE", unknown=True)
            emit({"event": "execution_history_restored", "model_history_exposed": False,
                "prior_request_sha256s": [_hash(asdict(item)) for item in execution_history],
                "prior_result_count": len(visible_results)})
        snapshot()
        if resume_checkpoint is not None:
            restored = _copy(resume_checkpoint)
            if (restored.get("schema") != "task-agent-loop-checkpoint.v2"
                    or restored.get("phase") not in {"ready", "response_ready", "dispatch_ready", "observations_charged"}
                    or restored.get("model_budget") != snapshot()
                    or _hash(restored.get("request")) != _hash(asdict(initial_request))
                    or any(row.get("status") in {"pending", "unknown"} for row in restored.get("call_records", {}).values())):
                raise ValueError("TASK_AGENT_RESUME_CHECKPOINT_NOT_ADMITTED")
            for call_id, record in seen.items():
                if restored["call_records"].get(call_id) != record:
                    if restored["call_records"].get(call_id, {}).get("result") != record.get("result"):
                        raise ValueError("TASK_AGENT_RESUME_HISTORY_CHANGED")
            seen = restored["call_records"]
            events, usage = restored["events"], restored["usage_by_turn"]
            turns, reads, observation_bytes = restored["model_turns"], restored["read_units"], restored["observation_bytes"]
            argument_used = restored["tool_argument_correction_used"]
            output_corrected = restored["output_correction_used"]
            reads_charged, observations_charged = restored["reads_charged"], restored["observations_charged"]
            pending_charge_replay = restored.get("pending_charge_replay")
            phase = restored["phase"]
            if restored.get("completion") is not None:
                return stop("TASK_AGENT_COMPLETION_ACCEPTED", completion=restored["completion"])
            if restored.get("pending_response") is not None:
                document = restored["pending_response"]
                pending_response = AgentTurnResponse(**{**document,
                    "tool_calls": tuple(AgentToolCall(**row) for row in document["tool_calls"]),
                    "usage": AgentUsage(**document["usage"])})
                resume_pending = True
            elif phase != "ready":
                raise ValueError("TASK_AGENT_RESUME_RESPONSE_REQUIRED")
    except Exception as error:
        return stop("TASK_AGENT_INPUT_INVALID", detail={"error_type": type(error).__name__})

    while True:
        # Scope: trusted execution-owner termination only, including before the first send.
        # 为什么要有它：已确认的无进展停止不是未知副作用，也不能再预留或发送一轮模型；此入口不接受模型文本作为成功。
        if not resume_pending:
            pending_response = None
            reads_charged = observations_charged = False
            checkpoint("ready")
        try:
            termination = stop_check() if stop_check is not None else None
            if termination is not None:
                if (not isinstance(termination, Mapping) or type(termination.get("code")) is not str
                        or not termination["code"]):
                    raise ValueError("TASK_AGENT_STOP_CHECK_INVALID")
                return replace(stop(termination["code"]), termination=_copy(termination))
        except Exception as error:
            return stop("TASK_AGENT_STOP_CHECK_FAILED", detail={"error_type": type(error).__name__})
        if tool_definitions is not None:
            # Scope: refresh descriptions and schemas, never execution owners or tool authority.
            # 为什么要有它：当前候选变化后只显示此时可用的表单，旧请求保持原样，工具名和执行权限不随展示刷新而扩大。
            try:
                definitions = tuple(deepcopy(tool_definitions()))
                if (any(not isinstance(item, AgentToolDefinition) or type(item.name) is not str
                        or type(item.description) is not str or not isinstance(item.parameters, dict)
                        or type(item.strict) is not bool for item in definitions)
                        or len(definitions) != len(registered)
                        or {item.name for item in definitions} != set(registered)):
                    raise ValueError("TASK_AGENT_TOOL_DEFINITIONS_INVALID")
                _json([asdict(item) for item in definitions])
                request = replace(request, tools=definitions)
            except Exception as error:
                return stop("TASK_AGENT_TOOL_DEFINITIONS_INVALID", detail={"error_type": type(error).__name__})
        try:
            before = snapshot()
            if not resume_pending and call_allowance(before, model_role) <= 0:
                return stop("TASK_AGENT_MODEL_BUDGET_EXHAUSTED")
            if not resume_pending and reserve_model_call() is False:
                return stop("TASK_AGENT_MODEL_BUDGET_EXHAUSTED")
            reserved = snapshot()
            if not resume_pending and (reserved["maximum"] != before["maximum"]
                    or reserved.get("policy", "stage_and_total") != before.get("policy", "stage_and_total")
                    or reserved["consumed"] != before["consumed"] + 1):
                return stop("TASK_AGENT_MODEL_RESERVATION_INVALID")
            if not resume_pending:
                turns += 1
                emit({"event": "provider_send_reserved", "turn": turns, "model_budget": reserved})
        except Exception as error:
            return stop("TASK_AGENT_MODEL_RESERVATION_FAILED", detail={"error_type": type(error).__name__})
        try:
            if resume_pending:
                response = pending_response
            else:
                with trace_operation("task.agent.turn", kind="llm", attributes={"turn": turns}) as span:
                    response = client.complete_turn(request)
                    if isinstance(response, AgentTurnResponse):
                        span.update(provider=response.provider, model=response.model, stop_reason=response.stop_reason)
        except Exception as error:
            measured = (getattr(error, "diagnostic", None) or {}).get("measured_usage")
            usage.append({"turn": turns, "usage": deepcopy(measured), "usage_available": measured is not None})
            return stop("TASK_AGENT_PROVIDER_TURN_FAILED", detail={"error_type": type(error).__name__,
                "provider_diagnostic": deepcopy(getattr(error, "diagnostic", None))})
        if not isinstance(response, AgentTurnResponse):
            return stop("TASK_AGENT_RESPONSE_INVALID")
        if not resume_pending:
            measured = measured_turn_usage(response)
            usage.append({"turn": turns, "provider": response.provider, "model": response.model,
                          "request_id": response.request_id, "stop_reason": response.stop_reason,
                          "usage": measured, "usage_available": measured is not None})
        try:
            if response_admission is not None and not resume_pending:
                response_admission(response)
        except Exception as error:
            return stop("TASK_AGENT_RESPONSE_NOT_ADMITTED", detail={"error_type": type(error).__name__,
                "code": getattr(error, "code", None)})
        pending_response = response
        if not resume_pending:
            checkpoint("response_ready")
        if response.stop_reason in _STOPPED:
            return stop("TASK_AGENT_RESPONSE_INCOMPLETE" if response.stop_reason in
                {"length", "max_tokens", "max_output_tokens", "incomplete"} else "TASK_AGENT_PROVIDER_STOPPED")
        if not response.tool_calls:
            has_text = type(response.content) is str and bool(response.content.strip())
            assistant = retain_assistant_text(response.content, response) if has_text else None
            if any(row.get("status") in {"pending", "unknown"} for row in seen.values()):
                return stop("TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE", unknown=True)
            # Scope: only complete, admitted text is correctable; real owner stops remain authoritative.
            # 为什么要有它：拒绝错误交付不等于撤销修订权限，但权限拒绝、无进展和未知副作用不能借纠正绕过。
            try:
                termination = stop_check() if stop_check is not None else None
                if termination is not None:
                    if (not isinstance(termination, Mapping) or type(termination.get("code")) is not str
                            or not termination["code"]):
                        raise ValueError("TASK_AGENT_STOP_CHECK_INVALID")
                    return replace(stop(termination["code"]), termination=_copy(termination))
            except Exception as error:
                return stop("TASK_AGENT_STOP_CHECK_FAILED", detail={"error_type": type(error).__name__})
            if output_corrected or not has_text:
                return stop("TASK_AGENT_FINISH_TOOL_REQUIRED")
            try:
                current_budget = snapshot()
                if call_allowance(current_budget, model_role) <= 0:
                    return stop("TASK_AGENT_MODEL_BUDGET_EXHAUSTED")
            except Exception as error:
                return stop("TASK_AGENT_MODEL_RESERVATION_FAILED", detail={"error_type": type(error).__name__})
            notice = AgentMessage("user", {
                "message_type": _OUTPUT_CORRECTION, "schema_version": "task-agent-output-correction.v1",
                "origin": "execution_host", "code": "TASK_AGENT_FINISH_TOOL_REQUIRED", "correction_number": 1,
                "assistant_message_sha256": _hash(asdict(assistant)), "task_completed": False,
                "message": "Text neither executes an action nor completes this task. A rejected finish does not revoke tools. "
                    "Continue with a permitted tool under the existing owner checks; finish requires an accepted current check. "
                    "This one correction grants no new permission or budget. "
                    "为什么要有它：一次普通输出错误不应结束可继续的任务，文字不能代替验收。",
                "available_actions": list(registered), "model_budget": current_budget,
                "budget_scope": "after_last_response_before_next_reservation; remaining includes the next model call",
                "resource_information": {"read_budget": "not_supplied", "observation_budget": "not_supplied"}})
            size = len(_json(asdict(notice)).encode("utf-8"))
            try:
                charge(byte_count=size)
            except TimeoutError as error:
                return stop("TASK_AGENT_WALL_BUDGET_EXHAUSTED", host_unadmitted=(notice,),
                    detail={"error_type": type(error).__name__})
            except Exception as error:
                # Scope: only the new host-message charge; preserve unknown owner refusal causes.
                # 为什么要有它：预算owner也检查时限和状态，不能把所有拒绝都误报为观察字节耗尽。
                detail = {"error_type": type(error).__name__}
                if type(getattr(error, "code", None)) is str:
                    detail["owner_error_code"] = error.code
                return stop("TASK_AGENT_HOST_FEEDBACK_NOT_ADMITTED", host_unadmitted=(notice,), detail=detail)
            request = replace(request, messages=(*request.messages, notice))
            output_corrected = True
            emit({"event": "output_format_correction", "turn": turns, "observation_bytes": size,
                "host_message_sha256": _hash(asdict(notice)), "output_correction_used": True})
            resume_pending = False
            pending_response = None
            reads_charged = observations_charged = False
            checkpoint("ready")
            continue
        if response.provider_state is None:
            return stop("TASK_AGENT_TOOL_STATE_MISSING")

        prepared_rows, input_errors, hard_errors = [], {}, []
        input_error_codes = {}
        batch_ids = set()
        for call in response.tool_calls:
            if (not isinstance(call, AgentToolCall) or type(call.call_id) is not str or not call.call_id
                    or call.call_id in batch_ids or type(call.name) is not str
                    or call.name not in registered or not isinstance(call.arguments, dict)):
                hard_errors.append("TASK_AGENT_TOOL_CALL_INVALID")
                continue
            batch_ids.add(call.call_id)
            action = registered[call.name]
            try:
                arguments = _copy(call.arguments)
                args_hash = _hash(arguments)
                previous = seen.get(call.call_id)
                if previous and (previous["name"] != call.name or previous["arguments_sha256"] != args_hash):
                    raise ValueError("TASK_AGENT_TOOL_CALL_IDENTITY_CHANGED")
                if previous and previous["status"] not in {"not_dispatched", "completed"}:
                    raise ValueError("TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE")
                if resume_pending and previous and previous["status"] == "completed":
                    prepared_rows.append((call, action, PreparedTaskAction(arguments, 0, previous.get("version_identity")),
                        "resume_completed", None, args_hash))
                    continue
                prepared = action.prepare(arguments)
                if (not isinstance(prepared, PreparedTaskAction) or type(prepared.read_cost) is not int
                        or prepared.read_cost < 0 or prepared.version_identity is not None
                        and (type(prepared.version_identity) is not str or not prepared.version_identity)):
                    raise ValueError("TASK_AGENT_PREPARATION_INVALID")
                mode, proof = "execute", None
                if previous and previous["status"] == "not_dispatched":
                    if previous.get("version_identity") != prepared.version_identity:
                        raise ValueError("TASK_AGENT_TOOL_VERSION_CHANGED")
                elif previous:
                    if not action.read_only or not callable(action.confirm_reuse):
                        raise ValueError("TASK_AGENT_REPEATED_TOOL_CALL")
                    proof = action.confirm_reuse(prepared, deepcopy(previous))
                    if (not isinstance(proof, Mapping) or proof.get("version_identity") != prepared.version_identity
                            or prepared.version_identity is None or not isinstance(proof.get("evidence"), Mapping)
                            or not proof["evidence"]):
                        raise ValueError("TASK_AGENT_REUSE_NOT_CONFIRMED")
                    mode = proof.get("mode")
                    if mode == "reuse":
                        if action.role == "finish" or previous.get("version_identity") != prepared.version_identity:
                            raise ValueError("TASK_AGENT_REUSE_NOT_CONFIRMED")
                    elif mode != "refresh" or action.role != "finish":
                        raise ValueError("TASK_AGENT_REUSE_NOT_CONFIRMED")
                    proof = _copy(proof)
                prepared_rows.append((call, action, prepared, mode, proof, args_hash))
            except TaskCandidateReadinessError as error:
                if action.role not in {"candidate_check", "finish"}:
                    hard_errors.append("TASK_AGENT_ACTION_NOT_ADMITTED")
                else:
                    input_errors[call.call_id] = list(error.argument_errors)
                    input_error_codes[call.call_id] = "TASK_AGENT_CANDIDATE_NOT_READY"
            except TaskCandidateRevisionInputError as error:
                if action.role != "candidate_submit" or action.read_only:
                    hard_errors.append("TASK_AGENT_ACTION_NOT_ADMITTED")
                else:
                    input_errors[call.call_id] = list(error.argument_errors)
                    input_error_codes[call.call_id] = "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"
            except TaskActionInputError as error:
                if action.role != "source_read":
                    hard_errors.append("TASK_AGENT_ACTION_NOT_ADMITTED")
                else:
                    input_errors[call.call_id] = list(error.argument_errors)
                    input_error_codes[call.call_id] = "TASK_AGENT_SOURCE_ARGUMENTS_INVALID"
            except Exception as error:
                hard_errors.append(getattr(error, "code", None) or str(error)
                    if isinstance(error, ValueError) else "TASK_AGENT_ACTION_NOT_ADMITTED")
        mixed_batch = len(response.tool_calls) > 1 and any(type(call.name) is str and registered.get(call.name) is not None and
                (not registered[call.name].read_only or registered[call.name].role == "finish")
                for call in response.tool_calls if isinstance(call, AgentToolCall))
        if mixed_batch:
            hard_errors.append("TASK_AGENT_MIXED_TOOL_BATCH")
        # Scope: only a completely admitted batch with the sole scheduling error is correctable.
        # 为什么要有它：还未派发的混合批次可让模型拆开；身份、范围、版本或参数问题不能借此降级为重试。
        correctable_mixed = (mixed_batch and not input_errors and len(prepared_rows) == len(response.tool_calls)
            and hard_errors == ["TASK_AGENT_MIXED_TOOL_BATCH"]
            and all(call.call_id not in seen or seen[call.call_id]["status"] == "not_dispatched"
                for call in response.tool_calls))
        if hard_errors and not correctable_mixed:
            return stop("TASK_AGENT_BATCH_NOT_ADMITTED", detail={"reasons": hard_errors})

        if input_errors or correctable_mixed:
            readiness_only = bool(input_errors) and set(input_error_codes.values()) == {"TASK_AGENT_CANDIDATE_NOT_READY"}
            if argument_used and not readiness_only:
                return stop("TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED")
            if call_allowance(snapshot(), model_role) <= 0:
                return stop("TASK_AGENT_MODEL_BUDGET_EXHAUSTED")
            corrections = []
            prepared_by_id = {row[0].call_id: row[2] for row in prepared_rows}
            for call in response.tool_calls:
                prepared = prepared_by_id.get(call.call_id)
                corrections.append(AgentToolResult(call.call_id, call.name, {
                    "status": "not_ready" if readiness_only else "rejected_not_executed",
                    **({"executed": False} if readiness_only else {}), "code": (
                        "TASK_AGENT_MIXED_TOOL_BATCH" if correctable_mixed else
                        input_error_codes.get(call.call_id, "TASK_AGENT_SOURCE_ARGUMENTS_INVALID")),
                    "argument_errors": input_errors.get(call.call_id, []), "read_cost": 0,
                    "message": ("No candidate check or handoff executed. Submit a valid source change before checking; "
                        "finish requires a current passed check. Continue using the existing task tools and budget."
                        if readiness_only else
                        "No calls in this batch executed. Only " + "/".join(read_tool_names) + " may be batched together. "
                        "Send upstream, candidate_submit, candidate_check or finish alone. Unchanged never-dispatched calls may keep their IDs."
                        " 为什么要有它：本批没有执行，必须先拆开动作再按原预算继续，不能依赖尚未产生的结果。"
                        if correctable_mixed else
                        "No candidate was created or checked; any existing candidate and check are unchanged. "
                        "Correct the indicated fields using the active candidate_submit schema and argument_errors. "
                        "Use the fields required by the currently supplied candidate_submit schema, "
                        "with the current source identity and authorized targets. You may also inspect authorized code. "
                        "This correction grants no new permission or budget. "
                        "为什么要有它：首次提交与后续修订都可能有可纠正的格式错误，未执行不等于禁止继续合法修改。"
                        if input_error_codes.get(call.call_id) == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID" else
                        "No calls in this batch executed. Correct missing/type errors; unchanged valid calls may be resent."),
                }))
                # Scope: only never-dispatched calls may keep the same native stable ID after correction.
                # 为什么要有它：Ollama按name/args生成ID，不能要求未执行good调用修改内容来绕过重复ID；已执行记录不能被覆盖为未执行。
                if call.call_id not in seen:
                    seen[call.call_id] = {"name": call.name, "arguments_sha256": _hash(call.arguments),
                        "status": "not_dispatched", "version_identity": prepared.version_identity if prepared else None,
                        "result": asdict(corrections[-1])}
            size = sum(len(_json(asdict(result)).encode("utf-8")) for result in corrections)
            try:
                charge(byte_count=size)
                argument_used = argument_used or not readiness_only
                emit({"event": "candidate_readiness_feedback" if readiness_only else
                      "mixed_batch_correction" if correctable_mixed else
                      "candidate_argument_correction" if "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID" in input_error_codes.values()
                      else "source_argument_correction",
                      "turn": turns, "observation_bytes": size,
                      "tool_argument_correction_used": argument_used})
            except Exception:
                return stop("TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED", unadmitted=corrections)
            retain(corrections, response)
            resume_pending = False
            pending_response = None
            reads_charged = observations_charged = False
            checkpoint("ready")
            continue

        requested_reads = sum(row[2].read_cost for row in prepared_rows if row[3] != "reuse")
        try:
            if not reads_charged:
                charge(read_count=requested_reads)
                reads_charged = True
                checkpoint("dispatch_ready")
        except TaskReadBudgetError as error:
            # Scope: an entirely legal, never-dispatched read/search batch may shrink once under the same budgets.
            # 为什么要有它：模型缺少额度信息时可调整调用数量，但不能部分执行、重放历史动作、扩大权限或重新领取纠错机会。
            eligible = (error.requested == requested_reads and len(prepared_rows) == len(response.tool_calls)
                and all(action.role in {"source_read", "source_search"} and action.read_only and prepared.read_cost == 1
                    and mode == "execute" and (call.call_id not in seen or seen[call.call_id]["status"] == "not_dispatched")
                    for call, action, prepared, mode, _proof, _hash_value in prepared_rows))
            if not eligible:
                return stop("TASK_AGENT_READ_BUDGET_EXHAUSTED")
            if argument_used:
                return stop("TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED")
            if call_allowance(snapshot(), model_role) <= 0:
                return stop("TASK_AGENT_MODEL_BUDGET_EXHAUSTED")
            read_budget = error.model_view()
            corrections = []
            for call, _action, prepared, _mode, _proof, args_hash in prepared_rows:
                result = AgentToolResult(call.call_id, call.name, {"status": "rejected_not_executed",
                    "code": "TASK_AGENT_READ_BUDGET_EXCEEDED", "read_cost": 0, "requested_reads": requested_reads,
                    "read_budget": read_budget,
                    "message": "No calls in this batch executed. Resend at most read_budget.remaining read/search calls, "
                        "or use another permitted action. A " + " or ".join(read_tool_names) + " costs one read. "
                        "Unchanged never-dispatched calls may keep their IDs. "
                        "为什么要有它：整批未执行且额度不增加，只能在原剩余额度内缩小批次或使用已有证据。"})
                corrections.append(result)
                seen[call.call_id] = {"name": call.name, "arguments_sha256": args_hash, "status": "not_dispatched",
                    "version_identity": prepared.version_identity, "result": asdict(result)}
            size = sum(len(_json(asdict(result)).encode("utf-8")) for result in corrections)
            try:
                charge(byte_count=size)
                argument_used = True
                emit({"event": "read_budget_correction", "turn": turns, "requested_reads": requested_reads,
                    "read_budget": read_budget, "observation_bytes": size, "tool_argument_correction_used": True})
            except Exception:
                return stop("TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED", unadmitted=corrections)
            retain(corrections, response)
            resume_pending = False
            pending_response = None
            reads_charged = observations_charged = False
            checkpoint("ready")
            continue
        except Exception:
            return stop("TASK_AGENT_READ_BUDGET_EXHAUSTED")
        results, final_decision = [], None
        for call, action, prepared, mode, proof, args_hash in prepared_rows:
            if mode == "resume_completed":
                previous_result = AgentToolResult(**deepcopy(seen[call.call_id]["result"]))
                results.append(previous_result)
                decision = previous_result.output.get("completion_check")
                if action.role == "finish" and isinstance(decision, dict) and decision.get("accepted") is True:
                    final_decision = decision
                continue
            started = False
            try:
                emit({"event": "tool_dispatch_prepared", "turn": turns, "call_id": call.call_id,
                    "name": call.name, "role": action.role, "mode": mode,
                    "version_identity": prepared.version_identity, "arguments_sha256": args_hash,
                    "read_cost": 0 if mode == "reuse" else prepared.read_cost})
                with trace_operation("task.agent.action", kind="tool", attributes={
                        "tool_name": call.name, "call_id": call.call_id, "role": action.role}) as span:
                    if mode == "reuse":
                        previous = seen[call.call_id]
                        previous_output = previous["result"]["output"]
                        output = deepcopy(previous_output["value"]
                            if previous.get("output_format") == "task_agent_tool_result_v1" else previous_output)
                    else:
                        seen[call.call_id] = {"name": call.name, "arguments_sha256": args_hash,
                            "status": "pending", "version_identity": prepared.version_identity}
                        emit({"event": "tool_dispatch_started", "call_id": call.call_id,
                              "role": action.role, "operation_state": "pending"})
                        started = True
                        output = _copy(action.execute(prepared.value))
                    result_output = {"value": output, "execution": "reused" if mode == "reuse" else "executed",
                        "version_identity": prepared.version_identity,
                        **({"reuse_confirmation": proof} if proof is not None else {})}
                    if action.role == "finish":
                        decision = _copy(completion_check(prepared=prepared, output=deepcopy(output)))
                        if not isinstance(decision, dict) or type(decision.get("accepted")) is not bool:
                            raise ValueError("TASK_AGENT_COMPLETION_CHECK_INVALID")
                        result_output["completion_check"] = decision
                        if decision["accepted"]:
                            final_decision = decision
                    result = AgentToolResult(call.call_id, call.name, result_output)
                    results.append(result)
                    seen[call.call_id] = {"name": call.name, "arguments_sha256": args_hash,
                        "status": "completed", "version_identity": prepared.version_identity,
                        "output_format": "task_agent_tool_result_v1", "result": asdict(result)}
                    span.update(execution=result_output["execution"], output_sha256=_hash(result_output))
                emit({"event": "tool_dispatch_completed", "call_id": call.call_id,
                      "role": action.role, "output_sha256": _hash(result_output), "model_budget": snapshot()})
            except Exception as error:
                seen[call.call_id] = {"name": call.name, "arguments_sha256": args_hash,
                    "status": "unknown" if started else "not_dispatched",
                    "version_identity": prepared.version_identity, "error_type": type(error).__name__}
                return stop("TASK_AGENT_ACTION_OUTCOME_UNKNOWN" if started else "TASK_AGENT_ACTION_RECORDING_FAILED",
                            unknown=started, unadmitted=results, detail={"call_id": call.call_id,
                                "error_type": type(error).__name__})
        # Scope: completed local results stay recorded, but a stopped task sends no extra model observations.
        # 为什么要有它：第三次同失败已经确定停止时，未发送F3不能再抢观测额度或把真实无进展终态覆盖成字节超限。
        try:
            termination = stop_check() if stop_check is not None else None
            if termination is not None:
                if (not isinstance(termination, Mapping) or type(termination.get("code")) is not str
                        or not termination["code"]):
                    raise ValueError("TASK_AGENT_STOP_CHECK_INVALID")
                return replace(stop(termination["code"], unadmitted=results), termination=_copy(termination))
        except Exception as error:
            return stop("TASK_AGENT_STOP_CHECK_FAILED", unadmitted=results,
                        detail={"error_type": type(error).__name__})
        size = sum(len(_json(asdict(result)).encode("utf-8")) for result in results)
        try:
            if not observations_charged:
                charge(byte_count=size)
                observations_charged = True
                checkpoint("observations_charged")
        except Exception:
            return stop("TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED", unadmitted=results)
        retain(results, response)
        resume_pending = False
        pending_response = None
        reads_charged = observations_charged = False
        checkpoint("ready", completion=final_decision)
        if final_decision is not None:
            return stop("TASK_AGENT_COMPLETION_ACCEPTED", completion=final_decision)
