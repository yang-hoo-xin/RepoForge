"""Content-bound checkpoints for the existing continuous task loop.

为什么要有它：恢复继续原任务和累计成本，已确认回执不重做，未知派发先对账。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from src.mod_agent.model_call_budget import validate_budget
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_state import (
    TaskAgentV2State,
    _no_links,
    load_v2_artifact,
)


def _require(value, code):
    if not value:
        raise ValueError("TASK_V2_RECOVERY_" + code)


def _ref(path):
    path = Path(path).resolve(strict=True)
    _no_links(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_bound(path, value):
    path = Path(path)
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    return _ref(path)


@contextmanager
def single_task_lock(store_root, contract_ref):
    path = Path(store_root) / "v2" / "session-locks" / (canonical_hash(contract_ref) + ".lock")
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("TASK_V2_RECOVERY_SESSION_ALREADY_RUNNING") from error
        yield
    finally:
        handle.close()


def session_snapshot(session):
    return {name: deepcopy(getattr(session, name)) for name in (
        "actual_calls", "submission_origins", "current_check", "public_checks", "_check_inflight")}


def restore_session(session, snapshot):
    for name in ("actual_calls", "submission_origins", "current_check", "public_checks", "_check_inflight"):
        setattr(session, name, deepcopy(snapshot[name]))


class TaskAgentRecoveryJournal:
    def __init__(self, owner, session, graph_state, original_budget, details, *, host_config_ref=None):
        from src.mod_agent import (
            public_failure_feedback,
            task_agent_execution,
            task_agent_loop,
            task_agent_v2_context,
            task_agent_v2_execution,
            task_agent_v2_experience,
            task_agent_v2_experiment_tools,
            task_agent_v2_experiments,
            task_agent_v2_semantic_state,
            task_agent_v2_state,
            task_agent_v2_tools,
            task_agent_v2_working_state,
        )
        from src.mod_agent.planning import learning_request, strategy_experience

        self.owner, self.session, self.state, self.details = owner, session, session.state, details
        self.lock_key = "logical-task:" + graph_state["run_id"] + ":" + graph_state["request_id"]
        if host_config_ref is None:
            from src.mod_agent.task_agent_execution import _source_facts
            deadline = getattr(owner, "deadline_utc", None)
            if deadline is None and owner.deadline is not None:
                deadline = time.time() + (owner.deadline - time.monotonic())
            host = {"schema": "task-agent-host-recovery-config.v2", "task_contract_ref": self.state.contract_ref,
                "workspace_root": str(owner.workspace_root), "artifact_store_root": str(owner.store.root),
                "original_output_root": str(owner.output_root), "model_profile": asdict(owner.profile),
                "policy": asdict(owner.policy), "worker_configuration_ref": owner.worker_configuration_ref,
                "expected_model_identity": owner.expected_model_identity,
                "allowed_edit_paths": list(owner.allowed_edit_paths), "allowed_read_roots": [str(path) for path in owner.allowed_read_roots],
                "source_facts": _source_facts(owner.allowed_read_roots), "verification_identity": owner.expected_verification_identity,
                "graph_state": deepcopy(graph_state), "original_budget_before": validate_budget(original_budget),
                "logical_session_key": self.lock_key,
                "candidate_mode": owner.candidate_mode, "deadline_utc": deadline,
                "semantic_state_enabled": getattr(owner, "semantic_state_enabled", False),
                "experience_configuration_ref": owner._experience.configuration_ref if owner._experience is not None else None,
                "task_timing_origin_ref": deepcopy(getattr(owner, "_task_timing_origin_ref", None)),
                "initial_request_ref": owner._save("recovery-initial-request.json", asdict(session.initial_request())),
                "owner_refs": [_ref(module.__file__) for module in (
                    task_agent_execution, task_agent_loop, task_agent_v2_execution, task_agent_v2_tools,
                    task_agent_v2_state, task_agent_v2_experience, task_agent_v2_experiments,
                    task_agent_v2_experiment_tools, task_agent_v2_context, task_agent_v2_working_state,
                    task_agent_v2_semantic_state, public_failure_feedback,
                    strategy_experience, learning_request)] + [_ref(__file__)]}
            self.host_config_ref = self.state.save("host-recovery-config", host)
            with self.state._transaction("host-recovery-config", {"kind": "host_recovery_config", "host_config_ref": self.host_config_ref}) as (_head, result, _replay):
                result["host_config_ref"] = self.host_config_ref
        else:
            self.host_config_ref = host_config_ref
        self.latest_path = self.state.root / "session-checkpoints" / (canonical_hash(self.state.contract_ref) + ".json")
        self.latest_ref = None
        self.latest_loop_state = None

    def save(self, loop_state):
        from src.mod_agent.task_agent_v2_execution import task_timing_snapshot
        owner = self.owner
        payload = {"schema": "task-agent-session-checkpoint.v2", "host_config_ref": self.host_config_ref,
            "task_contract_ref": self.state.contract_ref, "task_head": self.state.head,
            "loop": deepcopy(loop_state), "session": session_snapshot(self.session), "details": deepcopy(self.details),
            "owner": {"output_root": str(owner.output_root), "reads": owner._reads, "observation_bytes": owner._observation_bytes,
                "deadline_utc": getattr(owner, "deadline_utc", None) if getattr(owner, "deadline_utc", None) is not None
                    else self.state.load(self.host_config_ref)["deadline_utc"],
                "turn_count": getattr(owner, "_v2_turn_count", 0), "tool_charge_count": getattr(owner, "_v2_tool_charge_count", 0),
                "reservation": deepcopy(getattr(owner, "_v2_reservation", None)),
                "working_state": deepcopy(getattr(owner, "_v2_working_state", None)),
                "context_projection": deepcopy(getattr(owner, "_v2_context_projection", None))},
            "terminal_learning_receipt": deepcopy(owner._experience._terminal_receipt) if owner._experience is not None else None,
            "task_timing_observation": task_timing_snapshot(getattr(owner, "_task_timing_origin_ref", None),
                "checkpoint_observed", loop_state["phase"]),
            "previous_checkpoint_ref": self.latest_ref}
        ref = self.state.save("session-checkpoint", payload)
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.latest_path.with_name(self.latest_path.name + "." + uuid4().hex)
        _write_bound(temporary, {"checkpoint_ref": ref, "host_config_ref": self.host_config_ref})
        _no_links(self.latest_path)
        os.replace(temporary, self.latest_path)
        self.latest_ref = ref
        self.latest_loop_state = deepcopy(loop_state)
        owner.v2_checkpoint_ref = ref
        return ref

    def refresh_after_learning(self, receipt):
        _require(self.latest_loop_state is not None, "LOOP_CHECKPOINT_REQUIRED")
        if not receipt.get("receipt_ref"):
            return self.latest_ref
        original = read_bound_json(receipt["receipt_ref"])
        _require(original == {key: value for key, value in receipt.items() if key != "receipt_ref"}
            and original["budget_after"] == validate_budget(self.owner.budget_snapshot()), "LEARNING_RECEIPT_CHANGED")
        updated = deepcopy(self.latest_loop_state)
        updated["post_loop_learning"] = {"receipt_ref": receipt["receipt_ref"],
            "original_execution_ref": original["original_execution_ref"],
            "loop_budget_before_learning": updated.get("post_loop_learning", {}).get("loop_budget_before_learning", updated["model_budget"])}
        updated["model_budget"] = validate_budget(self.owner.budget_snapshot())
        return self.save(updated)

    def assert_current(self):
        _no_links(self.latest_path)
        _require(json.loads(self.latest_path.read_bytes())["checkpoint_ref"] == self.latest_ref
            and self.state.head == self.state.load(self.latest_ref)["task_head"], "SESSION_ADVANCED")


def _trusted_read(reference, store_root):
    path = Path(reference["path"])
    _no_links(path)
    _require(path.resolve().is_relative_to(Path(store_root).resolve()), "EXECUTION_REFERENCE_OUTSIDE_STORE")
    return read_bound_json(reference)


def _validate_completed(checkpoint, store_root, host=None):
    from src.mod_agent.task_agent_execution import _admitted_response

    calls = {row["call_id"]: row for row in checkpoint["session"]["actual_calls"]}
    receipts_by_call = {}
    for call_id, record in checkpoint["loop"]["call_records"].items():
        if record["status"] != "completed":
            continue
        _require(call_id in calls, "COMPLETED_ORIGIN_MISSING")
        call = calls[call_id]
        response, arguments = _trusted_read(call["response_ref"], store_root), _trusted_read(call["arguments_ref"], store_root)
        _admitted_response(response, checkpoint["details"]["model_profile"])
        _require([item for item in response["tool_calls"] if item["call_id"] == call_id]
            == [{"call_id": call_id, "name": call["name"], "arguments": arguments}]
            and arguments == call["arguments"] and canonical_hash(arguments) == record["arguments_sha256"], "COMPLETED_ARGUMENTS_CHANGED")
        _require(call.get("execution_receipt_ref"), "COMPLETED_EXECUTION_RECEIPT_REQUIRED")
        receipt = _trusted_read(call["execution_receipt_ref"], store_root)
        receipts_by_call[call_id] = receipt
        _require(receipt["origin"] == {key: item for key, item in call.items() if key != "execution_receipt_ref"}
            and receipt["output"] == record["result"]["output"].get("value")
            and record["result"]["output"].get("execution") == "executed", "COMPLETED_EXECUTION_RECEIPT_CHANGED")
        _require(any(event.get("event") == "tool_dispatch_completed" and event.get("call_id") == call_id
            and event.get("output_sha256") == canonical_hash(record["result"]["output"])
            for event in checkpoint["loop"]["events"]), "COMPLETED_RECEIPT_MISSING")
        decision = record["result"]["output"].get("completion_check")
        if call["name"] == "finish":
            state = TaskAgentV2State(store_root, checkpoint["task_contract_ref"])
            output = receipt["output"]
            accepted = False
            if receipt["session"]["current_check"] == output["check_ref"]:
                check = state.load(output["check_ref"])
                accepted = (check["status"] == "passed" and check["candidate_ref"] == output["candidate_ref"]
                    and receipt["task_head"]["current_revision_ref"] == output["candidate_ref"])
            expected = {"accepted": accepted, "candidate_ref": output["candidate_ref"], "check_ref": output["check_ref"], "whole_task_delivered": False}
            _require(decision == expected, "FINISH_DECISION_CHANGED")
    completion = checkpoint["loop"].get("completion")
    completed = [event for event in checkpoint["loop"]["events"] if event.get("event") == "tool_dispatch_completed"]
    if completed:
        last = receipts_by_call[completed[-1]["call_id"]]
        if last["task_head"]["contract_ref"] == checkpoint["task_contract_ref"]:
            _require(checkpoint["session"]["current_check"] == last["session"]["current_check"]
                and checkpoint["session"]["public_checks"] == last["session"]["public_checks"], "SESSION_CHECK_STATE_CHANGED")
            if completed[-1]["role"] == "finish" and checkpoint["loop"]["phase"] == "ready":
                decision = checkpoint["loop"]["call_records"][completed[-1]["call_id"]]["result"]["output"]["completion_check"]
                if decision["accepted"]:
                    _require(completion == decision, "COMPLETION_WAS_DROPPED")
        else:
            _require(host is not None and host.get("rebase_ref") and checkpoint["session"]["current_check"] is None,
                "SESSION_CONTEXT_CHANGED")
    if completion is not None:
        _require(completed and completed[-1]["role"] == "finish", "COMPLETION_EXECUTION_MISSING")
        record = checkpoint["loop"]["call_records"][completed[-1]["call_id"]]
        _require(record["result"]["output"]["completion_check"] == completion and completion["accepted"] is True,
            "COMPLETION_RESULT_CHANGED")
    if host is not None:
        initial = _trusted_read(host["initial_request_ref"], store_root)
        request = checkpoint["loop"]["request"]
        _require(request["messages"][:len(initial["messages"])] == initial["messages"], "INITIAL_REQUEST_CHANGED")
        transitions = host.get("source_context_messages", [])
        observed_transitions = [message for message in request["messages"] if isinstance(message.get("content"), dict)
            and "source_context_transition" in message["content"]]
        _require(observed_transitions == transitions, "SOURCE_CONTEXT_TRANSITION_CHANGED")
        ordinary_messages = [message for message in request["messages"] if message not in observed_transitions]
        from src.mod_agent.interrupted_patch_admission import retained_request_from_document
        from src.mod_agent.task_agent_loop import _output_correction_in_history
        decoded = retained_request_from_document(request)
        corrected = _output_correction_in_history(decoded.messages)
        _require(corrected == checkpoint["loop"]["output_correction_used"], "OUTPUT_CORRECTION_CHANGED")
        if len(ordinary_messages) > len(initial["messages"]):
            _require(corrected and len(ordinary_messages) == len(initial["messages"]) + 2, "UNRECORDED_REQUEST_MESSAGE")
        for exchange in [*request["historical_tool_exchanges"], {"tool_results": request["tool_results"], "provider_state": request["provider_state"]}]:
            for result in exchange["tool_results"]:
                _require(result == checkpoint["loop"]["call_records"].get(result["call_id"], {}).get("result"), "REQUEST_TOOL_RESULT_CHANGED")
                source = calls.get(result["call_id"])
                _require(source is not None and _trusted_read(source["response_ref"], store_root)["provider_state"] == exchange["provider_state"],
                    "REQUEST_PROVIDER_HISTORY_CHANGED")
        for index, event in enumerate(checkpoint["details"]["provider_turns"], 1):
            _require(event["turn"] == index, "PROVIDER_TURN_SEQUENCE_CHANGED")
            reference = _ref(Path(event["request_ref"]["path"]).parent / ("provider-outcome-" + event["reservation_id"] + ".json"))
            _require(_trusted_read(reference, store_root) == event, "PROVIDER_EXECUTION_RECORD_CHANGED")
        turns = checkpoint["details"]["provider_turns"]
        _require(len(turns) == checkpoint["loop"]["model_turns"], "LOOP_TURN_COUNT_CHANGED")
        pending = checkpoint["loop"].get("pending_response")
        if pending is not None:
            _require(turns and turns[-1].get("response_ref")
                and _trusted_read(turns[-1]["response_ref"], store_root) == pending, "PENDING_RESPONSE_CHANGED")


def _validate_resource_chain(state, checkpoint, host):
    head = state.head
    current, sequence, ids, totals = head["event_ref"], head["sequence"], set(), {}
    while current is not None:
        event = state.load(current)
        operation_id = event["operation_id"]
        _require(event["schema"] == "task-agent-operation.v2" and event["contract_ref"] == state.contract_ref
            and event["sequence"] == sequence and operation_id not in ids
            and canonical_hash(event["request"]) == event["request_hash"]
            and head["operations"].get(operation_id) == {"request_hash": event["request_hash"], "result": event["result"]},
            "RESOURCE_OPERATION_CHAIN_CHANGED")
        ids.add(operation_id)
        if event["request"].get("kind") == "resource_record":
            for key, amount in event["request"]["usage"].items():
                _require(type(amount) in {int, float} and amount >= 0, "RESOURCE_AMOUNT_INVALID")
                totals[key] = totals.get(key, 0) + amount
        current, sequence = event["previous_event_ref"], sequence - 1
    _require(sequence == 0 and ids == set(head["operations"]) and totals == head["resources"], "RESOURCE_TOTAL_CHANGED")
    _require(checkpoint["loop"]["read_units"] == checkpoint["owner"]["reads"] == totals.get("reads", 0)
        and checkpoint["loop"]["observation_bytes"] == checkpoint["owner"]["observation_bytes"] == totals.get("observation_bytes", 0),
        "OBSERVATION_RESOURCE_CHANGED")
    budget, original = validate_budget(checkpoint["loop"]["model_budget"]), validate_budget(host["original_budget_before"])
    _require(budget["consumed"] == head["resources"].get("model_calls", 0)
        and budget["maximum"] == original["maximum"] and budget.get("policy") == original.get("policy")
        and set(budget["by_stage"]) == set(original["by_stage"])
        and all(budget["by_stage"][key].get("maximum") == row.get("maximum")
            and budget["by_stage"][key]["consumed"] >= row["consumed"] for key, row in original["by_stage"].items()),
        "MODEL_BUDGET_CHANGED")
    learning = checkpoint["loop"].get("post_loop_learning")
    if learning is not None:
        receipt = _trusted_read(learning["receipt_ref"], state.store_root)
        execution = _trusted_read(learning["original_execution_ref"], state.store_root)
        _require(receipt["original_execution_ref"] == learning["original_execution_ref"]
            and receipt["budget_after"] == budget
            and execution["budget_after"] == learning["loop_budget_before_learning"]
            and receipt["budget_before"] == learning["loop_budget_before_learning"]
            and receipt["budget_before"]["consumed"] + receipt["model_calls"] == receipt["budget_after"]["consumed"],
            "LEARNING_BUDGET_RECEIPT_CHANGED")


def _validate_host_binding(state, checkpoint, host):
    from src.mod_agent.artifacts import ArtifactRef, ArtifactStore

    registered = state.head["operations"].get("host-recovery-config", {})
    _require(registered.get("result", {}).get("host_config_ref") == checkpoint["host_config_ref"], "HOST_CONFIG_NOT_REGISTERED")
    if "deadline_utc" in checkpoint["owner"]:
        from src.mod_agent.task_agent_v2_execution import earliest_task_deadline_utc

        deadline = checkpoint["owner"]["deadline_utc"]
        previous = state.load(checkpoint["previous_checkpoint_ref"]) if checkpoint.get("previous_checkpoint_ref") else {}
        prior = previous.get("owner", {}).get("deadline_utc")
        _require(deadline == earliest_task_deadline_utc(host["deadline_utc"], prior, deadline), "DEADLINE_EXTENDED")
    authority = state.load(state.contract["authority_ref"])
    baseline = state.load(state.contract["baseline_manifest_ref"])
    acceptance = state.load(state.contract["public_acceptance_ref"])
    resource = state.load(state.contract["resource_policy_ref"])
    workspace, target = Path(host["workspace_root"]).resolve(), Path(baseline["source_root"]).resolve()
    expected = {str((target / path).resolve()) for path in state.contract["allowed_edit_paths"]}
    reads = {str(Path(state.load(ref)["source_root"]).resolve()) for ref in state.contract["read_roots"].values()}
    graph = host["graph_state"]
    if host.get("task_timing_origin_ref") is not None:
        timing = read_bound_json(host["task_timing_origin_ref"])
        _require(timing.get("schema") == "task-agent-entry-timing.v2" and timing["run_id"] == graph["run_id"]
            and timing["request_id"] == graph["request_id"], "TIMING_ORIGIN_CHANGED")
    feature = state.contract["feature_spec_ref"]
    _require(target.is_relative_to(workspace)
        and host["logical_session_key"] == "logical-task:" + graph["run_id"] + ":" + graph["request_id"]
        and {str((workspace / path).resolve()) for path in host["allowed_edit_paths"]} == expected
        and {str(Path(path).resolve()) for path in host["allowed_read_roots"]} == reads
        and host["worker_configuration_ref"] == authority["worker_configuration_ref"]
        and host["verification_identity"] == acceptance["verification_identity"]
        and validate_budget(host["original_budget_before"]) == resource["initial_budget"]
        and authority["run_id"] == graph["run_id"] and authority["request_id"] == graph["request_id"]
        and state.contract["task_id"] == graph["request_id"]
        and feature == {"feature_id": graph["feature_id"], "artifact_type": "feature_spec",
            "path": graph["artifacts"]["feature_spec"], "artifact_hash": graph["artifact_hashes"]["feature_spec"]},
        "HOST_AUTHORITY_OR_TASK_CHANGED")
    envelope = ArtifactStore(state.store_root).semantic_read_immutable(ArtifactRef(**feature))
    _require(envelope["status"] == "success", "FEATURE_SPEC_NOT_ADMITTED")


def _reconcile_checkpoint(state, checkpoint_ref, checkpoint, latest_path):
    from src.libs.llm.agent_turn import AgentUsage
    from src.mod_agent.task_agent_execution import _admitted_response

    value = deepcopy(checkpoint)
    loop, owner = value["loop"], value["owner"]
    receipts = []
    output_root = Path(owner["output_root"])
    _require(output_root.resolve().is_relative_to(state.store_root), "RECEIPT_DIRECTORY_OUTSIDE_STORE")
    if loop["phase"] == "provider_pending":
        reservation = owner["reservation"]
        reference = _ref(output_root / ("provider-outcome-" + reservation["reservation_id"] + ".json"))
        event = read_bound_json(reference)
        _require(event.get("status") == "responded" and event["reservation_id"] == reservation["reservation_id"]
            and event["turn"] == reservation["turn"] and event["reserved_ref"] == reservation["reserved_ref"],
            "PROVIDER_OUTCOME_UNCONFIRMED")
        response = read_bound_json(event["response_ref"])
        _admitted_response(response, value["details"]["model_profile"])
        from src.mod_agent.interrupted_patch_admission import retained_request_from_document
        from src.mod_agent.task_agent_v2_context import validate_projected_request
        from src.mod_agent.task_agent_v2_working_state import (
            retain_model_statement,
            update_working_state,
        )
        effective = read_bound_json(event["request_ref"])
        expected = loop["request"]
        projection = validate_projected_request(expected, effective, state)
        current = projection["current_task_state"]
        boundary = {key: current.get(key) for key in ("task_contract_ref", "current_revision_ref",
            "binding_registry_ref", "current_check_ref")}
        _require(boundary == {"task_contract_ref": state.contract_ref,
            "current_revision_ref": value["task_head"]["current_revision_ref"],
            "binding_registry_ref": value["task_head"]["registry_ref"],
            "current_check_ref": value["session"]["current_check"]}, "PROVIDER_WORKING_BOUNDARY_CHANGED")
        # 为什么要有它：provider_pending在请求准备前落盘，恢复应重建实际请求的观察状态而不是沿用旧边界。
        working = update_working_state(owner.get("working_state"), retained_request_from_document(expected),
            task_contract_ref=boundary["task_contract_ref"], current_revision_ref=boundary["current_revision_ref"],
            registry_ref=boundary["binding_registry_ref"], current_check_ref=boundary["current_check_ref"])
        owner["working_state"] = retain_model_statement(working, response,
            response_ref=event["response_ref"], source_boundary=boundary)
        if state.load(value["host_config_ref"]).get("semantic_state_enabled", False):
            from src.mod_agent.task_agent_v2_semantic_state import record_semantic_state
            # 为什么要有它：收到响应后的中断重建同一模型声明，不新增调用，也不按未投影历史扩大可引用范围。
            owner["working_state"] = record_semantic_state(owner["working_state"], response,
                actual_request=effective, request_ref=event["request_ref"], response_ref=event["response_ref"])
        owner["context_projection"] = deepcopy(current.get("context_budget"))
        usage = AgentUsage(**response["usage"]).to_dict()
        old_head, new_head = value["task_head"], state.head
        operation_id = "model-usage:" + reservation["reservation_id"]
        _require(set(new_head["operations"]) == set(old_head["operations"]) | {operation_id}
            and new_head["operations"][operation_id]["request_hash"] == canonical_hash({"kind": "resource_record", "usage": usage})
            and new_head["resources"] == {key: old_head["resources"].get(key, 0) + usage.get(key, 0)
                for key in set(old_head["resources"]) | set(usage)}
            and all(new_head[key] == old_head[key] for key in ("registry_ref", "current_revision_ref", "plan_ref")),
            "PROVIDER_RESOURCE_RECONCILIATION_FAILED")
        for call in response["tool_calls"]:
            arguments_ref = _write_bound(state.root / "reconciled" / (uuid4().hex + ".json"), call["arguments"])
            value["session"]["actual_calls"].append({**call, "response_ref": event["response_ref"], "arguments_ref": arguments_ref})
        value["details"]["provider_turns"].append(event)
        loop["usage_by_turn"].append({"turn": loop["model_turns"], "provider": response["provider"], "model": response["model"],
            "request_id": response.get("request_id"), "stop_reason": response.get("stop_reason"), "usage": usage, "usage_available": True})
        loop.update(phase="response_ready", pending_response=response, terminal=None)
        owner["reservation"].update(request_ref=event["request_ref"], response_ref=event["response_ref"],
            call_ids=[call["call_id"] for call in response["tool_calls"]])
        receipts.append(reference)
    elif loop["phase"] == "tool_pending":
        pending = [key for key, row in loop["call_records"].items() if row["status"] == "pending"]
        _require(len(pending) == 1, "PENDING_TOOL_IDENTITY_INVALID")
        call_id = pending[0]
        reference = _ref(output_root / ("tool-receipt-" + canonical_hash(call_id) + ".json"))
        receipt = read_bound_json(reference)
        origin = next(row for row in value["session"]["actual_calls"] if row["call_id"] == call_id)
        _require(receipt["origin"] == {key: item for key, item in origin.items() if key != "execution_receipt_ref"}
            and receipt["task_head"] == state.head, "TOOL_RECEIPT_CHANGED")
        output, arguments, name = receipt["output"], origin["arguments"], origin["name"]
        if name in {"candidate_submit", "binding_add", "plan_update"}:
            operation = state.head["operations"].get(call_id)
            _require(operation is not None and all(output.get(key) == item for key, item in operation["result"].items()),
                "MUTATION_RECEIPT_MISSING")
            if name == "candidate_submit":
                candidate = state.validate_candidate(output["candidate_ref"])
                _require(candidate["model_origin"] == {key: origin[key] for key in ("call_id", "response_ref", "arguments_ref")},
                    "CANDIDATE_ORIGIN_CHANGED")
        elif name == "candidate_check":
            from src.mod_agent.domain.csharp_repository_backend import validate_candidate_report_v2
            checked = state.load(output["check_ref"])
            _require(checked["candidate_ref"] == arguments["candidate_ref"], "CHECK_CANDIDATE_CHANGED")
            validate_candidate_report_v2(checked["report"], context=read_bound_json(checked["request_ref"]))
        elif name not in {"source_read", "candidate_read", "source_search", "check_feedback_read", "history_read", "finish"}:
            raise ValueError("TASK_V2_RECOVERY_TOOL_REQUIRES_SPECIALIZED_RECONCILIATION")
        previous = loop["call_records"][call_id]
        projected = {"value": output, "execution": "executed", "version_identity": previous["version_identity"]}
        if name == "finish":
            checked = state.load(output["check_ref"])
            _require(receipt["session"]["current_check"] == output["check_ref"]
                and output["candidate_ref"] == state.current_revision_ref
                and checked["report"]["status"] == "passed", "FINISH_NOT_ACCEPTED")
            projected["completion_check"] = {"accepted": True, "candidate_ref": output["candidate_ref"],
                "check_ref": output["check_ref"], "whole_task_delivered": False}
        result = {"call_id": call_id, "name": name, "output": projected}
        previous.update(status="completed", result=result, output_format="task_agent_tool_result_v1")
        role = next(event["role"] for event in reversed(loop["events"])
            if event.get("event") == "tool_dispatch_started" and event.get("call_id") == call_id)
        event = {"event": "tool_dispatch_completed", "call_id": call_id, "role": role,
            "output_sha256": canonical_hash(projected), "model_budget": loop["model_budget"], "reconciled_receipt_ref": reference}
        loop["events"].append(event)
        origin["execution_receipt_ref"] = reference
        value["details"]["completed_tool_records"].append({**origin, "role": role, "event": event})
        value["session"] = receipt["session"]
        next(row for row in value["session"]["actual_calls"] if row["call_id"] == call_id)["execution_receipt_ref"] = reference
        loop.update(phase="dispatch_ready", terminal=None)
        receipts.append(reference)
    elif loop["phase"] in {"response_ready", "dispatch_ready"}:
        before, after = value["task_head"], state.head
        extra = set(after["operations"]) - set(before["operations"])
        expected_id = owner["reservation"]["reservation_id"] + ":tools:" + str(owner["tool_charge_count"] + 1)
        _require(extra == {expected_id} and all(after[key] == before[key]
            for key in ("registry_ref", "current_revision_ref", "plan_ref")), "CHARGE_RECONCILIATION_CHANGED")
        matched = []
        for path in output_root.glob("tool-resource-intent-*.json"):
            reference = _ref(path)
            intent = read_bound_json(reference)
            if intent.get("charge_id") == expected_id:
                matched.append((reference, intent))
        _require(len(matched) == 1, "CHARGE_INTENT_NOT_UNIQUE")
        reference, intent = matched[0]
        usage = intent["usage"]
        _require(intent["reservation"] == owner["reservation"]
            and after["operations"][expected_id]["request_hash"] == canonical_hash({"kind": "resource_record", "usage": usage})
            and set(usage) == {"reads", "observation_bytes"}
            and all(type(amount) is int and amount >= 0 for amount in usage.values())
            and after["resources"] == {key: before["resources"].get(key, 0) + usage.get(key, 0)
                for key in set(before["resources"]) | set(usage)}, "CHARGE_RECEIPT_CHANGED")
        owner["reads"] += usage["reads"]
        owner["observation_bytes"] += usage["observation_bytes"]
        owner["tool_charge_count"] += 1
        loop["read_units"] += usage["reads"]
        loop["observation_bytes"] += usage["observation_bytes"]
        if usage["observation_bytes"] == 0:
            loop["reads_charged"] = True
            loop["phase"] = "dispatch_ready"
        else:
            loop["pending_charge_replay"] = usage
        loop["terminal"] = None
        receipts.extend([reference, {"state_operation_ref": after["event_ref"]}])
    else:
        raise ValueError("TASK_V2_RECOVERY_UNKNOWN_ADVANCE_REQUIRES_RECONCILIATION")
    value["task_head"] = state.head
    value["reconciliation_ref"] = state.save("session-reconciliation", {"schema": "task-agent-session-reconciliation.v2",
        "previous_checkpoint_ref": checkpoint_ref, "receipts": receipts, "new_task_head": state.head,
        "actions_reexecuted": False, "model_calls_reissued": False})
    value["previous_checkpoint_ref"] = checkpoint_ref
    ref = state.save("session-checkpoint", value)
    temporary = latest_path.with_name(latest_path.name + "." + uuid4().hex)
    _write_bound(temporary, {"checkpoint_ref": ref, "host_config_ref": value["host_config_ref"]})
    os.replace(temporary, latest_path)
    return ref, value


def load_task_agent_checkpoint(*, artifact_store_root, checkpoint_ref, reconcile=False):
    from src.mod_agent.task_agent_execution import _source_facts

    root = Path(artifact_store_root).resolve(strict=True)
    if Path(checkpoint_ref).is_absolute():
        checkpoint_ref = Path(checkpoint_ref).resolve(strict=True).relative_to(root).as_posix()
    checkpoint = load_v2_artifact(root, checkpoint_ref)
    _require(checkpoint.get("schema") == "task-agent-session-checkpoint.v2", "CHECKPOINT_SCHEMA_INVALID")
    state = TaskAgentV2State(root, checkpoint["task_contract_ref"])
    _require(not state.head.get("superseded_by_contract_ref"), "TASK_CONTRACT_SUPERSEDED")
    latest = state.root / "session-checkpoints" / (canonical_hash(state.contract_ref) + ".json")
    _no_links(latest)
    _require(json.loads(latest.read_bytes())["checkpoint_ref"] == checkpoint_ref, "CHECKPOINT_SUPERSEDED")
    host = state.load(checkpoint["host_config_ref"])
    _require(type(host.get("semantic_state_enabled", False)) is bool, "SEMANTIC_STATE_FLAG_INVALID")
    _require(host["task_contract_ref"] == state.contract_ref and Path(host["artifact_store_root"]).resolve() == root, "HOST_TASK_CHANGED")
    _validate_host_binding(state, checkpoint, host)
    _require(all(_ref(ref["path"]) == ref for ref in host["owner_refs"]), "OWNER_CODE_CHANGED")
    read_bound_json(host["worker_configuration_ref"])
    if host["experience_configuration_ref"] is not None:
        read_bound_json(host["experience_configuration_ref"])
    _require(_source_facts(tuple(Path(path) for path in host["allowed_read_roots"])) == host["source_facts"], "SOURCE_CHANGED")
    _require(not state.head.get("pending_operation"), "STATE_OPERATION_REQUIRES_RECONCILIATION")
    if state.head != checkpoint["task_head"] or checkpoint["loop"]["phase"] in {"provider_pending", "tool_pending"}:
        _require(reconcile, "PENDING_DISPATCH_REQUIRES_RECONCILIATION")
        checkpoint_ref, checkpoint = _reconcile_checkpoint(state, checkpoint_ref, checkpoint, latest)
    _require(not checkpoint["session"]["_check_inflight"], "CHECK_REQUIRES_RECONCILIATION")
    _validate_resource_chain(state, checkpoint, host)
    _validate_completed(checkpoint, root, host)
    from src.mod_agent.task_agent_v2_working_state import validate_working_state
    validate_working_state(checkpoint["owner"].get("working_state"),
        responses=[_trusted_read(row["response_ref"], root) for row in checkpoint["details"]["provider_turns"]
            if row.get("response_ref")], call_records=checkpoint["loop"]["call_records"])
    if host.get("semantic_state_enabled", False):
        from src.mod_agent.task_agent_v2_semantic_state import validate_semantic_state
        validate_semantic_state(checkpoint["owner"].get("working_state"), [
            {"request_ref": row["request_ref"], "response_ref": row["response_ref"],
             "request": _trusted_read(row["request_ref"], root), "response": _trusted_read(row["response_ref"], root)}
            for row in checkpoint["details"]["provider_turns"] if row.get("response_ref")])
    return {"checkpoint_ref": checkpoint_ref, "checkpoint": checkpoint, "host": host, "state": state}


def _activate_delivery_rebase(root, rebase_ref):
    from src.mod_agent.task_agent_v2_delivery import validate_v2_delivery_rebase

    record = validate_v2_delivery_rebase(artifact_store_root=root, rebase_ref=rebase_ref)
    state = TaskAgentV2State(root, record["new_task_contract_ref"])
    latest = state.root / "session-checkpoints" / (canonical_hash(state.contract_ref) + ".json")
    _no_links(latest)
    if latest.exists():
        return json.loads(latest.read_bytes())["checkpoint_ref"]
    prior = state.load(record["prior_checkpoint_ref"])
    old_host = state.load(record["prior_host_config_ref"])
    _require(prior["host_config_ref"] == record["prior_host_config_ref"]
        and prior["task_contract_ref"] == record["prior_task_contract_ref"]
        and all(_ref(reference["path"]) == reference for reference in old_host["owner_refs"]), "REBASE_PRIOR_HOST_CHANGED")
    _validate_completed(prior, root, old_host)
    overrides = record["host_overrides"]
    host = {**deepcopy(old_host), **{key: deepcopy(value) for key, value in overrides.items() if key != "model_budget"},
        "rebase_ref": rebase_ref, "prior_host_config_ref": record["prior_host_config_ref"]}
    feedback = state.load(record["failure_feedback_ref"]) if record.get("failure_feedback_ref") else None
    message = {"role": "user", "content": {"source_context_transition": {
        "origin": "execution_host", "rebase_ref": rebase_ref, "prior_task_contract_ref": record["prior_task_contract_ref"],
        "task_contract_ref": state.contract_ref, "current_revision_ref": state.current_revision_ref,
        "binding_registry_ref": state.registry_ref, "public_delivery_feedback": feedback,
        "old_source_views_valid": False, "old_check_valid": False, "new_apply_approval_granted": False,
        "reason": "为什么要有它：交付失败后的修订以已核验的实际源码为起点，历史观察保留，旧前像和精确diff批准不沿用。"}}}
    host["source_context_messages"] = [*host.get("source_context_messages", []), message]
    host_ref = state.save("host-recovery-config", host)
    with state._transaction("host-recovery-config", {"kind": "host_recovery_config", "host_config_ref": host_ref}) as (_head, result, _replay):
        result["host_config_ref"] = host_ref
    checkpoint = deepcopy(prior)
    checkpoint.update(host_config_ref=host_ref, task_contract_ref=state.contract_ref,
        task_head=state.head, previous_checkpoint_ref=record["prior_checkpoint_ref"], rebase_ref=rebase_ref)
    checkpoint["loop"]["request"]["messages"].append(message)
    checkpoint["loop"].update(phase="ready", pending_response=None, reads_charged=False, observations_charged=False,
        pending_charge_replay=None, completion=None, terminal=None, model_budget=deepcopy(overrides["model_budget"]))
    checkpoint["session"].update(current_check=None, _check_inflight=False)
    checkpoint["details"].update(workspace_root=host["workspace_root"], allowed_edit_paths=host["allowed_edit_paths"],
        source_before=host["source_facts"], rebase_ref=rebase_ref,
        historical_execution_ref=record["historical_execution_ref"])
    ref = state.save("session-checkpoint", checkpoint)
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_name(latest.name + "." + uuid4().hex)
    _write_bound(temporary, {"checkpoint_ref": ref, "host_config_ref": host_ref})
    os.replace(temporary, latest)
    return ref


@contextmanager
def _recovery_execution_trace(owner, host, checkpoint_ref):
    """Retain one recovery invocation without replacing the original run trace.

    为什么要有它：恢复模型的真实发送内容及新增用量也必须可核验，不能由旧调用的真实性代替。
    """
    from src.mod_agent.patch_execution_trace import PatchExecutionTraceConfiguration

    graph = host["graph_state"]
    configuration = PatchExecutionTraceConfiguration()
    observations, returned = {}, {}
    trace = configuration.make_trace(run_id=graph["run_id"], request_id=graph["request_id"],
        feature_id=graph["feature_id"], artifact_store_root=owner.store.root,
        recorder=lambda row: observations.update({row["id"]: row}), root_name="product.recovery")
    trace.context.metadata["local_body_root"] = str(owner.output_root)
    failure = None
    try:
        with trace.activate():
            yield returned
    except BaseException as error:
        failure = error
        raise
    finally:
        summary = configuration.finish(trace)
        summary.update(trace_id=trace.context.trace_id, session_id=trace.session_id, environment=trace.environment)
        receipt = getattr(owner, "_receipt", None)
        document = {"schema_version": "task_agent_recovery_execution_v1",
            "run_id": graph["run_id"], "request_id": graph["request_id"],
            "resumed_from_checkpoint_ref": checkpoint_ref,
            "v2_checkpoint_ref": owner.v2_checkpoint_ref,
            "native_execution_ref": receipt.execution_ref if receipt is not None else None,
            "status": "failed" if failure else "returned", "error_type": type(failure).__name__ if failure else None,
            "execution_trace": summary, "execution_observations": observations,
            "model_calls_performed": sum(row.get("kind") == "generation" for row in observations.values()),
            "model_call_count_basis": "observed_transport_started_in_this_invocation_not_cumulative_budget"}
        try:
            returned["recovery_execution_ref"] = owner._save("recovery-execution.json", document)
            if failure is not None:
                failure.recovery_execution_ref = returned["recovery_execution_ref"]
        except Exception as error:
            returned["recovery_execution_status"] = "unavailable"
            returned["recovery_execution_error"] = type(error).__name__


def resume_v2_task_agent(*, artifact_store_root, checkpoint_ref):
    from src.mod_agent.run_outcome_finalization import finalize_continuation

    try:
        result = _resume_v2_task_agent(artifact_store_root=artifact_store_root, checkpoint_ref=checkpoint_ref)
    except Exception as error:
        error.outcome_finalization = finalize_continuation(artifact_store_root=artifact_store_root, event="recovery_return",
            checkpoint_ref=checkpoint_ref, error=error)
        raise
    return {**result, **finalize_continuation(artifact_store_root=artifact_store_root,
        event="recovery_return", checkpoint_ref=checkpoint_ref, result=result)}


def _resume_v2_task_agent(*, artifact_store_root, checkpoint_ref):
    """Reconstruct the same trusted owner and shared loop; never allocate a task.

    为什么要有它：CLI恢复必须消耗原合同、原模型和原累计账本，而不是暗中重开一次尝试。
    """
    from src.mod_agent.artifacts import ArtifactStore
    from src.mod_agent.local_entry import (
        LocalOllamaInitialPatchProfileV1,
        _reserve_request_model_call,
    )
    from src.mod_agent.model_admission import (
        assert_ollama_model_identity_stable,
        read_ollama_model_identity,
        validate_ollama_context_window,
        validate_ollama_model_identity,
    )
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from src.mod_agent.task_agent_v2_execution import (
        TaskAgentExecutionV2,
        earliest_task_deadline_utc,
        task_deadline_utc_from_environment,
    )

    root = Path(artifact_store_root).resolve(strict=True)
    if Path(checkpoint_ref).is_absolute():
        checkpoint_ref = Path(checkpoint_ref).resolve(strict=True).relative_to(root).as_posix()
    initial = load_v2_artifact(root, checkpoint_ref)
    if initial.get("schema") == "task-agent-delivery-receipt.v2":
        _require(initial.get("status") == "build_failed", "CONFIRMED_DELIVERY_FAILURE_REQUIRED")
        from src.mod_agent.task_agent_v2_delivery import prepare_v2_delivery_rebase
        prepared = prepare_v2_delivery_rebase(artifact_store_root=root, receipt_ref=checkpoint_ref,
            operation_id="delivery-feedback:" + canonical_hash(checkpoint_ref))
        resumed = resume_v2_task_agent(artifact_store_root=root, checkpoint_ref=prepared["rebase_ref"])
        return {**resumed, "resumed_from_delivery_receipt_ref": checkpoint_ref}
    rebase_ref = checkpoint_ref if initial.get("schema") == "task-agent-delivery-rebase.v2" else None
    initial_host = load_v2_artifact(root, initial["prior_host_config_ref"] if rebase_ref else initial["host_config_ref"])
    with single_task_lock(artifact_store_root, initial_host["logical_session_key"]):
        if rebase_ref is not None:
            checkpoint_ref = _activate_delivery_rebase(root, rebase_ref)
        loaded = load_task_agent_checkpoint(artifact_store_root=artifact_store_root, checkpoint_ref=checkpoint_ref, reconcile=True)
        host, checkpoint = loaded["host"], loaded["checkpoint"]
        deadline_utc = earliest_task_deadline_utc(host["deadline_utc"], checkpoint["owner"].get("deadline_utc"),
            task_deadline_utc_from_environment())
        _require(deadline_utc is None or time.time() < deadline_utc, "DEADLINE_EXPIRED")
        worker = PatchWorkerConfiguration.load(host["worker_configuration_ref"]["path"])
        _require(worker.reference == host["worker_configuration_ref"], "WORKER_CHANGED")
        evaluator = worker.behavior_evaluator()
        _require(evaluator.verification_identity() == host["verification_identity"], "VERIFIER_CHANGED")
        observed = read_ollama_model_identity(host["model_profile"]["base_url"], host["model_profile"]["model"])
        validate_ollama_model_identity(observed, expected_base_url=host["model_profile"]["base_url"],
            expected_model=host["model_profile"]["model"])
        assert_ollama_model_identity_stable(expected=host["expected_model_identity"], observed=observed)
        validate_ollama_context_window(observed, requested_context_window=host["model_profile"]["context_window"])
        budget = validate_budget(checkpoint["loop"]["model_budget"])

        def reserve(role):
            updated = _reserve_request_model_call(budget, role)
            budget.clear()
            budget.update(updated)

        def validate_providers():
            _require(PatchWorkerConfiguration.load(host["worker_configuration_ref"]["path"]).reference == worker.reference
                and evaluator.verification_identity() == host["verification_identity"], "PROVIDERS_CHANGED")

        output = Path(host["original_output_root"]) / ("resume-" + uuid4().hex)
        owner = TaskAgentExecutionV2(artifact_store=ArtifactStore(artifact_store_root), workspace_root=host["workspace_root"],
            profile=LocalOllamaInitialPatchProfileV1(**host["model_profile"]), policy=worker.policy,
            behavior_evaluator=evaluator, worker_configuration_ref=worker.reference,
            expected_model_identity=host["expected_model_identity"], identity_reader=read_ollama_model_identity,
            providers={}, validate_providers=validate_providers, reserve_model_call=reserve,
            budget_snapshot=lambda: deepcopy(budget), allowed_edit_paths=tuple(host["allowed_edit_paths"]),
            allowed_read_roots=tuple(host["allowed_read_roots"]), output_root=output,
            deadline_utc=deadline_utc, candidate_mode=host["candidate_mode"],
            experience_configuration_ref=host["experience_configuration_ref"],
            semantic_state_enabled=host.get("semantic_state_enabled", False))
        _require(asdict(owner.policy) == host["policy"], "POLICY_CHANGED")
        owner._resume_v2 = loaded
        owner._resume_lock_held = True
        with _recovery_execution_trace(owner, host, loaded["checkpoint_ref"]) as recovery_trace:
            outcome = owner(deepcopy(host["graph_state"]))
            if outcome.status == "candidate_ready":
                from src.mod_agent.task_agent_v2_delivery import finish_resumed_v2_handoff
                result = finish_resumed_v2_handoff(owner=owner, outcome=outcome, state=deepcopy(host["graph_state"]))
            else:
                result = {"status": outcome.status, "run_id": host["graph_state"]["run_id"],
                    "request_id": host["graph_state"]["request_id"], "outcome": outcome.to_dict()}
            learning = owner.complete_terminal_learning()
            timing_ref = owner.observe_task_boundary("resumed_product_return", status=result["status"])
        return {**result, "v2_checkpoint_ref": owner.v2_checkpoint_ref,
            **({"resumed_from_rebase_ref": rebase_ref} if rebase_ref else {}),
            "resumed_from_checkpoint_ref": loaded["checkpoint_ref"], "new_task_created": False,
            "strategy_learning": learning,
            "task_timing_ref": timing_ref, **recovery_trace,
            "cumulative_resources": loaded["state"].head["resources"]}
