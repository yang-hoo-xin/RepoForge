"""One bounded schema correction for the registered read-only source tool."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from src.libs.llm.agent_turn import AgentToolCall, AgentToolResult, AgentTurnRequest

ARGUMENT_ERROR = "PATCH_AGENT_TOOL_ARGUMENTS_INVALID"
STATE_KEY = "tool_argument_correction"


@dataclass(frozen=True)
class ArgumentRejection:
    code: str
    reason: str
    issues: tuple[tuple[dict[str, str], ...], ...] = ()


def correction_used(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("PATCH_AGENT_ARGUMENT_CORRECTION_STATE_INVALID")
    return value


def receipt_correction_used(receipt: Mapping) -> bool:
    """Read cumulative state only from a caller-validated worker receipt."""
    used = correction_used(receipt.get("tool_argument_correction_used", False))
    loop = receipt.get("patch_agent_loop") or {}
    used = correction_used(loop.get("tool_argument_correction_used", False)) or used
    revisions = loop.get("recovery_revisions") or [loop.get(name) for name in ("validator_correction", "behavior_correction")]
    for revision in revisions:
        if isinstance(revision, Mapping) and isinstance(revision.get("execution"), Mapping):
            used = correction_used(revision["execution"].get("tool_argument_correction_used", False)) or used
    return used


def _issues(arguments: object) -> tuple[dict[str, str], ...]:
    if not isinstance(arguments, dict):
        return ({"field": "$", "issue": "object_required"},)
    errors = []
    if "binding_id" not in arguments:
        errors.append({"field": "binding_id", "issue": "required"})
    elif type(arguments["binding_id"]) is not str:
        errors.append({"field": "binding_id", "issue": "string_required"})
    elif not arguments["binding_id"].strip():
        errors.append({"field": "binding_id", "issue": "nonempty_string_required"})
    for field in arguments.keys() - {"binding_id"}:
        # Scope: only flat, string-valued annotations can be ordinary excess arguments.
        # 为什么要有它：不能猜测path/source/权限/执行字段是否已授权，未知额外字段继续硬拒；也不回显任意值。
        if field not in {"comment", "reason"} or type(arguments[field]) is not str:
            errors.append({"field": "$", "issue": "non_annotation_extra_field"})
        else:
            errors.append({"field": field, "issue": "unexpected_annotation"})
    return tuple(errors)


def classify_source_arguments(calls: tuple[AgentToolCall, ...], *, source_tool: str,
    available_bindings: tuple[str, ...], inspected_bindings: set[str], executed_call_ids: set[str],
    rejected_argument_call_ids: set[str], reusable_call_bindings: Mapping[str, str] | None,
    retryable_argument_calls: Mapping[str, str], tool_available: bool, allow_batches: bool) -> ArgumentRejection | None:
    problems = tuple(_issues(call.arguments) if call.name == source_tool else () for call in calls)
    for call in calls:
        if isinstance(call.call_id, str) and call.call_id in rejected_argument_call_ids:
            binding = retryable_argument_calls.get(call.call_id)
            # Scope: an exact valid member of a wholly rejected batch has not yet executed.
            # 为什么要有它：Ollama按工具名和参数生成ID，模型无法换ID；只允许原合法调用第一次执行，不放过畸形参数或借ID换来源。
            if (binding is None or call.name != source_tool or call.arguments != {"binding_id": binding}
                    or call.call_id in executed_call_ids and (
                        reusable_call_bindings is None or reusable_call_bindings.get(call.call_id) != binding)):
                return ArgumentRejection("PATCH_AGENT_TOOL_CALL_ID_REUSED", "A rejected call id cannot authorize changed arguments or an unpermitted repeated read.")
    if not any(problems):
        return None
    # Scope: validate the complete exchange before allowing any ordinary correction.
    # 为什么要有它：首项缺参不能掩盖后项越界身份、重复ID或权限错误；整批一律不执行。
    ids: set[str] = set()
    for call in calls:
        if type(call.call_id) is not str or not call.call_id:
            return ArgumentRejection("PATCH_AGENT_TOOL_CALL_ID_INVALID", "A provider call id is missing.")
        if call.call_id in ids:
            return ArgumentRejection("PATCH_AGENT_TOOL_CALL_ID_REUSED", "A provider call id was reused.")
        ids.add(call.call_id)
    if any(call.name != source_tool for call in calls):
        return ArgumentRejection("PATCH_AGENT_UNKNOWN_TOOL" if any(call.name not in {source_tool, "request_patch_upstream"} for call in calls)
            else "PATCH_AGENT_MIXED_TOOL_BATCH_FORBIDDEN", "Argument correction is limited to source reads alone.")
    if len(calls) > 1 and not allow_batches:
        return ArgumentRejection("PATCH_AGENT_PARALLEL_TOOL_CALLS_FORBIDDEN", "Batched source reads are unavailable.")
    selected: set[str] = set()
    for call, errors in zip(calls, problems):
        binding = call.arguments.get("binding_id") if isinstance(call.arguments, dict) else None
        if call.call_id in executed_call_ids and (errors or reusable_call_bindings is None
                or reusable_call_bindings.get(call.call_id) != binding):
            return ArgumentRejection("PATCH_AGENT_TOOL_CALL_ID_REUSED", "Malformed arguments cannot establish cached-read identity.")
        if isinstance(binding, str) and binding.strip():
            if binding not in available_bindings:
                return ArgumentRejection("PATCH_AGENT_BINDING_UNKNOWN", "A binding is outside the current admitted source scope.")
            if binding in selected or binding in inspected_bindings and reusable_call_bindings is None:
                return ArgumentRejection("PATCH_AGENT_REPEATED_TOOL_CALL", "A source binding cannot be inspected again.")
            selected.add(binding)
    if not tool_available:
        return ArgumentRejection("PATCH_AGENT_TOOL_FORBIDDEN", "Source inspection is unavailable on this turn.")
    if any(error["issue"] == "non_annotation_extra_field" for errors in problems for error in errors):
        return ArgumentRejection(ARGUMENT_ERROR, "Source, target, execution, authority and unknown excess fields are not correctable annotations.")
    return ArgumentRejection(ARGUMENT_ERROR, "Source arguments must follow the currently allowed binding-only schema.", problems)


def argument_feedback(calls, rejection, *, allowed_format, model_turns_used, observation_bytes_before):
    return tuple(AgentToolResult(call_id=call.call_id, name=call.name, output={
        "status": "not_executed", "error_code": ARGUMENT_ERROR, "batch_rejected": True,
        "tool_execution_performed": False, "argument_errors": list(issues),
        "allowed_format": allowed_format,
        **({"validated_retry_arguments": {"binding_id": call.arguments["binding_id"]}} if not issues else {}),
        "required_action": "Correct only invalid arguments. A valid, unexecuted call may be repeated with exactly the same arguments and id; otherwise use fresh call ids, or finish from existing context. "
                           "All original source permissions and model/read/observation budgets still apply; this is the only argument-correction opportunity.",
        STATE_KEY: {"version": 1, "used": True, "model_turns_used": model_turns_used,
                    "observation_bytes_before": observation_bytes_before},
    }) for call, issues in zip(calls, rejection.issues))


def retained_correction_state(request: AgentTurnRequest, *, source_tool: str):
    used, ids, model_floor, byte_floor = False, set(), 0, 0
    retryable = {}
    groups = [exchange.tool_results for exchange in request.historical_tool_exchanges]
    if request.tool_results:
        groups.append(request.tool_results)
    for group in groups:
        selected = [result for result in group if isinstance(result.output, Mapping) and STATE_KEY in result.output]
        if not selected:
            continue
        # Scope: one exchange may contain several correlated call results, but not two opportunities.
        # 为什么要有它：历史中两次独立纠错不能仅OR成true后被当成合法一次；同一批多call仍只算一次。
        if used:
            raise ValueError("PATCH_AGENT_ARGUMENT_CORRECTION_STATE_INVALID")
        states = []
        for result in selected:
            state = result.output[STATE_KEY]
            if (result.name != source_tool or result.output.get("status") != "not_executed"
                    or result.output.get("error_code") != ARGUMENT_ERROR or not isinstance(state, Mapping)
                    or type(state.get("version")) is not int or state["version"] != 1
                    or correction_used(state.get("used")) is not True
                    or any(type(state.get(key)) is not int or state[key] < 0
                           for key in ("model_turns_used", "observation_bytes_before"))):
                raise ValueError("PATCH_AGENT_ARGUMENT_CORRECTION_STATE_INVALID")
            if type(result.call_id) is not str or not result.call_id or result.call_id in ids:
                raise ValueError("PATCH_AGENT_ARGUMENT_CORRECTION_STATE_INVALID")
            if "validated_retry_arguments" in result.output:
                arguments = result.output["validated_retry_arguments"]
                if _issues(arguments) or result.output.get("argument_errors") != []:
                    raise ValueError("PATCH_AGENT_ARGUMENT_CORRECTION_STATE_INVALID")
                retryable[result.call_id] = arguments["binding_id"]
            ids.add(result.call_id)
            states.append((state["model_turns_used"], state["observation_bytes_before"]))
        if len(selected) != len(group) or len(set(states)) != 1:
            raise ValueError("PATCH_AGENT_ARGUMENT_CORRECTION_STATE_INVALID")
        used = True
        model_floor = max(model_floor, states[0][0])
        byte_floor = max(byte_floor, states[0][1] + sum(len(json.dumps(result.output,
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")) for result in selected))
    return used, ids, model_floor, byte_floor, retryable
