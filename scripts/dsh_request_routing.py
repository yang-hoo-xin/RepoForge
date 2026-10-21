"""Natural-language selection in front of the existing DSH/product owners."""
from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.core.trace.execution_trace import ExecutionTrace
from src.libs.llm.ollama_llm import OllamaLLM
from src.mod_agent.local_entry import _read_ollama_model_identity
from src.mod_agent.planning.clarification_state import (
    ClarificationState,
    read_reference,
    reference,
    transcript,
)
from src.mod_agent.planning.clarified_request import (
    ANCHOR_PROTOCOL,
    UNIT_ROLE_PROTOCOL,
    product_description,
)
from src.mod_agent.planning.task_intent import select_task_intent
from src.mod_agent.upstream_model_trace import _usage


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def routing_context(*, execution_config, active_run_id=None, parent_execution_config=None, delivery_root=None):
    from scripts import dsh_run_request as bridge

    _, options = bridge._configuration_arguments(execution_config)
    context = {"projects": [{"id": "configured_project", "name": Path(options["--target-project"]).name}],
               "tasks": [], "active_task_id": active_run_id}
    bindings = {"execution_config_ref": bridge._file_ref(Path(execution_config))}
    if options.get("--dependency-api-config"):
        bindings["dependency_api"] = bridge.dependency_api_binding(options)
    if options.get("--knowledge-evidence-config"):
        bindings["knowledge_evidence"] = bridge.knowledge_evidence_binding(options)
    if options.get("--repository-profile"):
        bindings["repository_profile"] = bridge.repository_profile_binding(options)
    if parent_execution_config is not None and active_run_id is None:
        raise ValueError("TASK_INTENT_PARENT_REQUIRES_ACTIVE_TASK")
    if active_run_id is not None:
        selected = bridge.task_execution_config(execution_config=parent_execution_config or execution_config, run_id=active_run_id)
        if parent_execution_config is not None:
            bindings["parent_profile_ref"] = bridge._file_ref(Path(parent_execution_config))
        record = bridge.inspect_run(execution_config=selected, run_id=active_run_id)
        manifest, result = record["manifest"], record["result"]
        if manifest is None or result is None:
            document = bridge.read_json_object(selected)
            binding = document.get("followup_binding") or document.get("new_project_binding")
            if manifest is not None or result is not None or binding is None:
                raise ValueError("TASK_INTENT_ACTIVE_TASK_RECORDED_RESULT_REQUIRED")
            request = bridge.read_json_object(bridge._verify_ref(binding["request_ref"]))
            _, prepared_options = bridge._configuration_arguments(selected)
            # 为什么要有它：初始化或派生完成后进程中断，继续命令应识别已绑定原需求，不再分配工程或预算。
            manifest = {"request": request["description"], "target_project": prepared_options["--target-project"]}
            meter_only = prepared_options.get("--model-call-budget-policy") == "meter_only"
            maximum = None if meter_only else int(prepared_options["--task-model-call-limit"])
            result = {"status": "prepared_not_started", "model_call_budget": {
                "maximum": maximum, "consumed": 0, "remaining": maximum,
                **({"policy": "meter_only"} if meter_only else {})}}
            bindings["prepared_request_ref"] = binding["request_ref"]
        context["tasks"].append({"id": active_run_id, "original_request": manifest["request"],
            "status": result.get("status"), "remaining_model_budget": result.get("model_call_budget")})
        context["projects"].append({"id": active_run_id, "name": Path(manifest["target_project"]).name
            if "target_project" in manifest else manifest.get("target_mod", active_run_id),
            "origin": "selected_task_output_requires_verified_delivery_before_new_requirement"})
        project_binding = bridge.read_json_object(selected).get("new_project_binding")
        if project_binding is not None:
            # 为什么要有它：动态工程目录是内部标识，用户的工程名称应来自已绑定初始化原件。
            _, selected_options = bridge._configuration_arguments(selected)
            bound_project = bridge._verify_project_configuration(Path(selected), selected_options)
            receipt = bound_project["receipt"]
            if receipt["project_id"] != active_run_id:
                raise ValueError("TASK_INTENT_PROJECT_IDENTITY_CHANGED")
            context["projects"][-1]["name"] = receipt["identity"]["plugin_name"]
            bindings["active_project_initialization_ref"] = project_binding["receipt_ref"]
        root = Path(record["run_root"])
        if (root / "delivery/result.json").is_file() or delivery_root is not None:
            delivered = bridge._verified_delivered_source(selected, active_run_id, delivery_root)
            bindings["active_delivery_refs"] = delivered["evidence_refs"]
            bindings["active_delivered_source_sha256"] = delivered["source_snapshot_sha256"]
            context["tasks"][-1]["status"] = "delivered"
        bindings["active_configuration_ref"] = bridge._file_ref(Path(selected))
        if record["manifest"] is not None:
            bindings["active_manifest_ref"] = bridge._file_ref(root / "run_manifest.json")
            continuation = record.get("continuation")
            if continuation is not None:
                result_ref = continuation.get("result_ref")
                if result_ref is None:
                    pointer = root / "continuations/interrupted_latest.json"
                    bindings["active_continuation_pointer_ref"] = bridge._file_ref(pointer)
                    result_ref = bridge.read_json_object(pointer)["result_ref"]
                bridge._verify_ref(result_ref)
                bindings["active_result_ref"] = result_ref
            elif record["result"] is not None:
                bindings["active_result_ref"] = bridge._file_ref(root / "run_result.json")
    return context, bindings


def run_natural_request(*, execution_config, description, active_run_id=None,
                        allocation_config=None, delivery_root=None, parent_execution_config=None,
                        preview_only=False, llm=None, identity_reader=None,
                        conversation_id=None, clarification_for=None, turn_id=None, product_executor=None, unified_evaluation=None):
    """Route once, then call the existing owner with unchanged literal request/permissions.

    为什么要有它：自然语言路由不另造执行循环；其一次调用独立记账，同任务续接不获得新预算。
    """
    from scripts import dsh_run_request as bridge

    _, parsed = bridge.validate_product_profile(execution_config=execution_config, description=description)
    workspace = Path(parsed.workspace_root).resolve()
    state = None
    if conversation_id is None and (clarification_for is not None or turn_id is not None):
        raise ValueError("CLARIFICATION_CONVERSATION_REQUIRED")
    if conversation_id is not None:
        state = ClarificationState(workspace=workspace, conversation_id=conversation_id,
            turn_id=turn_id, clarification_for=clarification_for, request={
                "description": description, "execution_config": str(Path(execution_config).resolve()),
                "active_run_id": active_run_id, "parent_execution_config": parent_execution_config,
                "allocation_config": allocation_config, "delivery_root": delivery_root, "preview_only": preview_only})
        try:
            replay = state.begin(clarification_for, preview_only=preview_only)
        except ValueError as error:
            if "IN_PROGRESS_OR_INTERRUPTED" not in str(error):
                raise
            return {"status": "routing_indeterminate", "reason": str(error),
                    "process_liveness": "unknown", "product_owner_invoked": False,
                    "product_execution_started": False, "routing_model_calls_reserved": 0,
                    "conversation_id": conversation_id, "turn_id": state.turn_id}
        if replay is not None:
            state.close()
            return replay
    try:
        context, bindings = routing_context(execution_config=execution_config, active_run_id=active_run_id,
            parent_execution_config=parent_execution_config, delivery_root=delivery_root)
        turns = [{"role": "original_user_request", "content": description}]
        context_drift = False
        resolved_new_input = False
        if state is not None and state.resolved_reply is not None:
            prior_result = state.replay(state.resolved_reply)
            prior_request = read_reference(prior_result["request_ref"], workspace)
            if (state.key == state.resolved_reply["request_hash"]
                    and (context, bindings) == (prior_request["context"], prior_request["bindings"])):
                state.close()
                return prior_result
            resolved_new_input = True
        if state is not None and state.parent is not None:
            parent_root = Path(state.parent["routing_record_path"]).parent
            receipt = bridge.read_json_object(parent_root / "conversation_receipt.json")
            parent_request = read_reference(receipt["request_ref"], workspace)
            turns = parent_request["conversation_turns"] + [
                {"role": "assistant_clarification_context", "content": state.parent["question"]},
                {"role": "user_answer_or_correction", "content": description}]
            context_drift = (context, bindings) != (parent_request["context"], parent_request["bindings"])
        effective_description = transcript(turns)
    except BaseException:
        if state is not None:
            state.close()
        raise
    route_id = "route_" + uuid4().hex
    root = bridge._contained_record_path(workspace / ".dsh-routing" / route_id, workspace)
    root.mkdir(parents=True, exist_ok=False)
    meter_only = parsed.model_call_budget_policy == "meter_only"
    budget = {"routing_model_call_limit": None if meter_only else 1,
        "product_new_task_call_limit": None if meter_only else parsed.task_model_call_limit,
        "new_task_combined_maximum": None if meter_only else 1 + parsed.task_model_call_limit,
        "new_task_combined_maximum_scope": "current routing turn plus a new product task only; prior routing costs are additional",
        "resume_product_budget": "unchanged inherited task budget",
        "scope": "routing cost is additional and separate from product task budget"}
    if meter_only:
        budget.update(policy="meter_only",
            new_task_combined_maximum_scope="No count ceiling; routing and product usage are retained separately.")
    if state is not None:
        state.claim(root, preview_only)
    _save(root / "request.json", {"description": description, "effective_description": effective_description,
        "conversation_turns": turns, "conversation_id": conversation_id,
        "clarification_for": state.parent["route_id"] if state and state.parent else None,
        "turn_id": state.turn_id if state else None, "context": context,
        "bindings": bindings, "budget": budget, "preview_only": preview_only})
    started = time.perf_counter()
    trace_rows = {}
    trace = ExecutionTrace(route_id, route_id, "local", recorder=lambda row: trace_rows.update({row["id"]: row}),
        root_name="task.routing")
    selected_identity = None
    result = {"status": "routing_failed", "route_id": route_id, "budget": budget,
              "product_execution_started": False, "product_owner_invoked": False,
              "routing_model_calls_reserved": 0, "routing_usage": None,
              "preview_only": preview_only, "conversation_id": conversation_id,
              "turn_id": state.turn_id if state else None}
    try:
        if resolved_new_input:
            result.update(status="routing_rejected", reason="CLARIFICATION_ALREADY_RESOLVED_NEW_INPUT",
                question="该澄清已用于执行先前答复。本次内容或绑定不同，请作为新的需求明确提交，原任务不会重复执行。")
            return result
        if context_drift:
            # 为什么要有它：目标或权限配置改变后，旧澄清不能静默授权在新对象上执行。
            result.update(status="needs_clarification", question="目标工程、活动任务或执行配置已改变。请确认按当前目标与配置继续，或明确要选择的目标。",
                          reason="CLARIFICATION_CONTEXT_CHANGED")
            return result
        identity_reader = identity_reader or _read_ollama_model_identity
        selected_identity = identity_reader(parsed.ollama_base_url, parsed.ollama_model)
        client = llm or OllamaLLM(SimpleNamespace(llm=SimpleNamespace(model=parsed.ollama_model,
            temperature=0.0, max_tokens=768)), base_url=parsed.ollama_base_url,
            timeout=parsed.ollama_timeout_seconds, context_window=parsed.ollama_context_window)
        result["routing_model_calls_reserved"] = 1
        with trace.activate():
            decision, response, provenance = select_task_intent(description=effective_description, context=context,
                llm=client, body_root=root, current_user_message=description, conversation_turns=turns)
        result.update(decision=decision, routing_usage=_usage(response), routing_protocol=provenance["protocol"],
                      assessment=provenance["assessment"], intent_provenance=provenance)
        result["routing_elapsed_seconds"] = time.perf_counter() - started
        if identity_reader(parsed.ollama_base_url, parsed.ollama_model) != selected_identity:
            raise ValueError("TASK_INTENT_MODEL_CHANGED")
        if response.model != parsed.ollama_model:
            raise ValueError("TASK_INTENT_RESPONSE_MODEL_CHANGED")
        current_context, current_bindings = routing_context(execution_config=execution_config,
            active_run_id=active_run_id, parent_execution_config=parent_execution_config, delivery_root=delivery_root)
        if (current_context, current_bindings) != (context, bindings):
            raise ValueError("TASK_INTENT_CONTEXT_CHANGED")
        # 为什么要有它：模型返回后重核真实输入，选路不能执行在过时配置或被替换任务上。
        _save(root / "selection.json", {"decision": decision, "bindings": bindings,
            "model_identity": selected_identity, "budget": budget, "usage": result.get("routing_usage"),
            "assessment": provenance["assessment"], "intent_provenance": provenance})
        if decision["action"] == "clarify":
            result.update(status="needs_clarification", question=decision["clarification"])
        elif preview_only:
            result["status"] = "route_selected"
        else:
            action, target = decision["action"], decision["target_id"]
            if action == "resume_task":
                if any(row["id"] == target and row.get("status") == "delivered" for row in context["tasks"]):
                    # 为什么要有它：已完成交付的续跑只读返回真实交付证据，不能重开旧待批准状态。
                    product = {"status": "delivered", "run_id": target, "read_only": True,
                        "delivery_refs": bindings["active_delivery_refs"], "model_calls_performed": 0}
                else:
                    result.update(product_owner_invoked=True, product_execution_started=None)
                    product = bridge.advance_task(execution_config=parent_execution_config or execution_config, run_id=target,
                        allocation_config=allocation_config)
            else:
                anchor_binding = ({"conversation_id": conversation_id, "turn_id": state.turn_id,
                    "request_ref": reference(root / "request.json"), "selection_ref": reference(root / "selection.json")}
                    if len(turns) > 1 and provenance["protocol"] in {ANCHOR_PROTOCOL, UNIT_ROLE_PROTOCOL} else None)
                effective_description = product_description(turns, decision, routing_binding=anchor_binding)
                if len(turns) > 1:
                    _save(root / "product_request.json", {"description": effective_description,
                        "routing_request_ref": reference(root / "request.json"),
                        "routing_selection_ref": reference(root / "selection.json")})
                # 为什么要有它：准备工程也可能产生副作用，准备中断后不得当作零派发重新分配工程。
                result.update(product_owner_invoked=True, product_execution_started=False)
                if action == "modify_project" and target != "configured_project":
                    selected = bridge.prepare_followup_task(execution_config=execution_config,
                        parent_run_id=target, description=effective_description, delivery_root=delivery_root,
                        parent_execution_config=parent_execution_config)
                else:
                    selected = bridge.prepare_new_task(execution_config=execution_config,
                        description=effective_description,
                        project_name=decision["project_name"] if action == "new_project" else None)
                result["execution_config_path"] = selected
                # 为什么要有它：实际工作仍由原CLI/Graph负责，原需求文字不被分类模型改写。
                _save(root / "dispatch.json", {"execution_config_ref": bridge._file_ref(Path(selected)),
                    "literal_description": effective_description,
                    **({"product_request_ref": reference(root / "product_request.json")} if len(turns) > 1 else {})})
                output = io.StringIO()
                result.update(product_owner_invoked=True, product_execution_started=True)
                with redirect_stdout(output):
                    argv = bridge.build_product_argv(execution_config=selected, description=effective_description)
                    # 为什么要有它：可信调用方可选择冻结进程传输，但argv仍由原正常入口分配，模型不能选择可执行代码。
                    code = (product_executor or bridge.cli.main)(argv)
                _save(root / "product_output.json", {"exit_code": code, "stdout": output.getvalue()})
                product = json.loads(output.getvalue())
            status = product.get("status") or (product.get("product_execution_result") or {}).get("execution_disposition")
            result.update(status=status or "unknown", product_result=product)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        result.update(status="dispatch_failed" if result["product_owner_invoked"] else "routing_failed",
            reason=str(error), error_type=type(error).__name__)
    finally:
        model_rows = [row["attributes"] for row in trace_rows.values() if row["name"] == "model.request"]
        transports = [row.get("transport_started") for row in model_rows]
        result["routing_transport_calls"] = (0 if not result["routing_model_calls_reserved"] else
            sum(v is True for v in transports) if transports and all(type(v) is bool for v in transports) else None)
        result["routing_transport_calls_known_subtotal"] = sum(v is True for v in transports)
        if result["routing_usage"] is None and len(model_rows) == 1:
            result["routing_usage"] = model_rows[0].get("usage")
        _save(root / "trace.json", {"observations": list(trace_rows.values())})
        result.update(model_identity=selected_identity, elapsed_seconds=time.perf_counter() - started,
            routing_record_path=str(root / "result.json"))
        result.setdefault("routing_elapsed_seconds", result["elapsed_seconds"])
        result["clarification_id"] = route_id if result["status"] == "needs_clarification" and not preview_only else None
        result["request_ref"] = reference(root / "request.json")
        try:
            if state is not None:
                result.update(state.totals(result))
                result["budget"]["conversation_reserved_routing_plus_new_product_maximum"] = (
                    None if parsed.model_call_budget_policy == "meter_only"
                    else result["cumulative_routing_calls"] + parsed.task_model_call_limit)
            if unified_evaluation is not None:
                try:
                    result["unified_evaluation"] = unified_evaluation(result, routing_root=root)
                except Exception as error:
                    result["unified_evaluation"] = {"status": "evaluation_error", "error_type": type(error).__name__}
            _save(root / "result.json", result)
            if state is not None:
                state.finish(root)
        finally:
            if state is not None:
                state.close()
    return result
