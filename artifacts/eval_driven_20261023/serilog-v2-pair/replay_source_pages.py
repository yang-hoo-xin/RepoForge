"""Read-only, zero-provider counterfactual replay of saved source tool results.

This is platform admission evidence, never a reconstructed autonomous success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentResponseFormat,
    AgentToolDefinition,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnProviderError,
    AgentTurnRequest,
    OllamaChatAgentAdapter,
    _httpx_json_body_identity,
)
from src.mod_agent.task_agent_v2_state import SOURCE_READ_DEFAULT_LIMIT, TaskAgentV2State, _hash
from src.mod_agent.task_agent_v2_tools import TaskAgentSessionV2


def reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_ref(ref, root):
    path = Path(ref["path"]).resolve()
    if not path.is_relative_to(root) or reference(path)["sha256"] != ref["sha256"]:
        raise ValueError("saved reference outside selected native root or hash mismatch")
    return json.loads(path.read_bytes())


def decode(document):
    request = AgentTurnRequest(**{**document,
        "messages": tuple(AgentMessage(**item) for item in document["messages"]),
        "tools": tuple(AgentToolDefinition(**item) for item in document["tools"]),
        "tool_results": tuple(AgentToolResult(**item) for item in document["tool_results"]),
        "historical_tool_exchanges": tuple(AgentToolExchange(**{**item,
            "tool_results": tuple(AgentToolResult(**row) for row in item["tool_results"])})
            for item in document["historical_tool_exchanges"]),
        "response_format": AgentResponseFormat(**document["response_format"]) if document["response_format"] else None})
    if json.loads(json.dumps(asdict(request))) != document:
        raise ValueError("saved request does not round-trip")
    return request


class OfflineStopError(Exception):
    pass


def admission(request, execution):
    profile = execution["model_profile"]
    captured = {}

    def never_send(*_args):
        raise OfflineStopError()

    adapter = OllamaChatAgentAdapter(base_url=profile["base_url"], model=profile["model"],
        max_output_tokens=profile["output_budget"], context_window=profile["context_window"],
        request_timeout_seconds=profile["timeout_seconds"], request_sender=never_send)
    adapter._prompt_token_calibrations = [(row["provider_request_bytes"], row["usage"]["input_tokens"])
        for row in execution["provider_turns"] if row["status"] == "responded"]
    original = adapter._context_admission

    def observe(payload):
        captured.update(request_identity=_httpx_json_body_identity(profile["base_url"] + "/api/chat", payload),
            context_admission=original(payload))
        return captured["context_admission"]

    adapter._context_admission = observe
    try:
        adapter.complete_turn(request)
    except OfflineStopError:
        pass
    except AgentTurnProviderError as error:
        if error.diagnostic.get("failure_boundary") != "provider_request_gate":
            raise
    if not captured:
        raise ValueError("payload not reconstructed")
    return captured


def inspect(native_root, default_limit=SOURCE_READ_DEFAULT_LIMIT):
    root = Path(native_root).resolve()
    execution_path = root / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    last_turn = execution["provider_turns"][-1]
    request_ref = last_turn["request_ref"]
    document = read_ref(request_ref, root)
    request = decode(document)
    recorded = admission(request, execution)
    diagnostic = execution["termination"]["detail"]["provider_diagnostic"]
    if recorded["request_identity"]["provider_body_utf8_bytes"] != diagnostic["request_identity"]["provider_body_utf8_bytes"]:
        raise ValueError("restored serialized payload size differs from recorded rejected request")
    if recorded["context_admission"] != diagnostic["context_admission"]:
        raise ValueError("restored context estimate differs from recorded rejected request")
    recorded["recorded_request_identity"] = diagnostic["request_identity"]
    recorded["literal_body_hash_matches"] = recorded["request_identity"] == diagnostic["request_identity"]
    state = TaskAgentV2State(root.parent, execution["task_contract_ref"])
    # 为什么要有它：只计算假设的新读取结果及其身份，不改原件、状态、资源，也不构造模型从未作出的后续动作。
    state.save = lambda kind, data: f"v2/{kind}/{_hash(data)}.json"
    calls = {row["call_id"]: row for row in execution["completed_tool_records"]}
    changed = []

    def project(result):
        if result.name not in {"source_read", "candidate_read", "source_search"}:
            return result
        call = calls[result.call_id]
        raw_result = read_ref(call["result_ref"], root)
        if raw_result != result.output:
            raise ValueError("tool result differs from saved execution receipt")
        args = call["arguments"]
        if result.name == "source_search":
            value = state.search_source(args["parent_revision_ref"], args["query"], args.get("root_id", "target"))
        else:
            value = state.read_source(args["parent_revision_ref"], args["path"], args.get("root_id", "target"),
                offset=args.get("offset", 0), limit=args.get("limit", default_limit))
        changed.append({"call_id": result.call_id, "name": result.name, "result_ref": call["result_ref"],
            "path": args.get("path"), "returned_chars": len(value.get("text", "")),
            "next_offset": value.get("next_offset"), "truncated": value.get("truncated")})
        return replace(result, output={**deepcopy(result.output), "value": value})

    paged = replace(request, tool_results=tuple(project(row) for row in request.tool_results),
        historical_tool_exchanges=tuple(replace(exchange,
            tool_results=tuple(project(row) for row in exchange.tool_results)) for exchange in request.historical_tool_exchanges))
    session = TaskAgentSessionV2(state=state, evaluator=None, owner=SimpleNamespace(), feature_id="offline-projection")
    definitions = {item.definition.name: item.definition for item in session.actions()}
    current = replace(paged, tools=tuple(definitions.get(tool.name, tool)
        if tool.name in {"source_read", "candidate_read"} else tool for tool in paged.tools))
    responded = [row for row in execution["provider_turns"] if row["status"] == "responded"]
    usage = {key: sum(row["usage"].get(key, 0) for row in responded)
        for key in ("input_tokens", "output_tokens", "cached_input_tokens")}
    first_payload = read_ref(responded[0]["provider_request_ref"], root)
    experience_messages = []
    for message in first_payload["messages"]:
        if message["role"] != "user":
            continue
        try:
            content = json.loads(message["content"])
        except (ValueError, TypeError):
            continue
        if isinstance(content, dict) and "strategy_experience" in content:
            experience_messages.append(content["strategy_experience"])
    return {"execution_ref": reference(execution_path), "request_ref": request_ref, "run_id": execution["run_id"],
        "projection_program_ref": reference(__file__),
        "projection_owner_refs": [reference(Path(__file__).resolve().parents[3] / "src/mod_agent" / name)
            for name in ("task_agent_v2_state.py", "task_agent_v2_tools.py")],
        "actual": {"status": execution["status"], "responses": len(responded), "reservations": len(execution["provider_turns"]),
            "known_usage": usage, "candidate_ref": execution["candidate_ref"], "check_ref": execution["check_ref"],
            "first_actual_request_ref": responded[0]["provider_request_ref"],
            "strategy_experience_messages_in_first_actual_request": len(experience_messages),
            "whole_task_delivered": execution["whole_task_delivered"],
            "executed_actions": [{"name": row["name"], "call_id": row["call_id"],
                "arguments_ref": row["arguments_ref"], "result_ref": row["result_ref"]}
                for row in execution["completed_tool_records"]]},
        "original": recorded, "counterfactual_paged_outputs": admission(paged, execution),
        "counterfactual_current_read_schemas": admission(current, execution), "projected_calls": changed,
        "limits": ["No provider, model, build, test or recovery executed; original files unchanged.",
            "Archived JSON key ordering may change literal body hash; byte count and full original admission diagnostic must match.",
            "Counterfactual tool results do not show which pages or edits the model would choose.",
            "Historical context still accumulates; this does not prove unlimited task continuity.",
            "Known usage is across requests; cached tokens are a subset of input tokens, not additional usage."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--default-limit", type=int, default=SOURCE_READ_DEFAULT_LIMIT)
    args = parser.parse_args()
    print(json.dumps(inspect(args.native_root, args.default_limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
