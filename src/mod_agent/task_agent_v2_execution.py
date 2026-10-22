"""Product-owned v2 execution using the shared loop and repository checker.

为什么要有它：正常入口需建立可信任务合同并记录真实模型动作，不能用演示脚本或旧Plan伪造新协议交接。
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from uuid import uuid4

from src.libs.llm.agent_turn import AgentMessage, AgentTurnResponse
from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.model_call_budget import validate_budget
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_execution import (
    TaskAgentExecution,
    _require,
    _source_facts,
    _worker_binding,
)
from src.mod_agent.task_agent_v2_handoff import (
    TaskAgentPatchOutcomeV2,
    validate_task_agent_handoff_v2,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact
from src.mod_agent.task_agent_v2_tools import TaskAgentSessionV2


class TaskAgentExecutionV2(TaskAgentExecution):
    protocol = "v2"

    def __init__(self, *, experience_configuration_ref=None, **kwargs):
        super().__init__(**kwargs)
        self._experience = None
        if experience_configuration_ref is not None:
            from src.mod_agent.task_agent_v2_experience import TaskAgentV2Experience
            self._experience = TaskAgentV2Experience(self, experience_configuration_ref)

    def check_experience_boundary(self):
        if getattr(self, "_experience", None) is not None:
            self._experience.check_boundary()

    def prepare_decision_request(self, request, session):
        state = session.state
        registry = state.validate_registry(state.registry_ref)
        current_check = state.load(session.current_check) if session.current_check is not None else None
        current = {"origin": "execution_host", "task_contract_ref": state.contract_ref,
            "current_revision_ref": state.current_revision_ref, "binding_registry_ref": state.registry_ref,
            "valid_bindings": [{"binding_id": key, "path": item["canonical_path"]} for key, item in sorted(registry["bindings"].items())],
            "current_check_ref": session.current_check, "current_check_status": current_check["status"] if current_check else None,
            "technical_plan_ref": state.head["plan_ref"], "baseline_revision_ref": state.contract["baseline_manifest_ref"],
            "baseline_usage": "Historical source reference; editing must name current_revision_ref and a matching source_view_ref."}
        prepared = replace(request, messages=(*request.messages, AgentMessage("user", {"current_task_state": current})))
        return self._experience.prepare_decision_request(prepared, session) if getattr(self, "_experience", None) is not None else prepared

    def complete_terminal_learning(self):
        if getattr(self, "_experience", None) is None:
            return {"status": "not_configured"}
        from src.mod_agent.task_agent_v2_recovery import single_task_lock
        lock = nullcontext() if getattr(self, "_resume_lock_held", False) else single_task_lock(self.store.root, self._recovery_journal.lock_key)
        with lock:
            journal = getattr(self, "_recovery_journal", None)
            if journal is not None:
                journal.assert_current()
            self._learning_lock_held = True
            try:
                result = self._experience.complete_terminal_learning()
                self.refresh_recovery_checkpoint_after_learning(result)
                return result
            finally:
                self._learning_lock_held = False

    def refresh_recovery_checkpoint_after_learning(self, receipt):
        journal = getattr(self, "_recovery_journal", None)
        if journal is None:
            return None
        from src.mod_agent.task_agent_v2_recovery import single_task_lock
        held = (getattr(self, "_resume_lock_held", False) or getattr(self, "_learning_lock_held", False)
            or getattr(self, "_active_session_lock", None) is not None)
        with nullcontext() if held else single_task_lock(self.store.root, journal.lock_key):
            return journal.refresh_after_learning(receipt)

    def stop_v2_check(self):
        try:
            self.check_experience_boundary()
        except (ValueError, OSError) as error:
            return {"code": "TASK_AGENT_EXPERIENCE_WITHDRAWN_OR_CHANGED", "reason": str(error)}
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return {"code": "TASK_AGENT_WALL_BUDGET_EXHAUSTED"}
        return None

    def reserve_v2_call(self):
        reservation_id = uuid4().hex
        before = validate_budget(self.budget_snapshot())
        turn = getattr(self, "_v2_turn_count", 0) + 1
        intent = self._save("resource-intent-" + reservation_id + ".json", {
            "reservation_id": reservation_id, "turn": turn,
            "kind": "model_call", "task_contract_ref": self._v2_state.contract_ref, "before": before})
        result = self.reserve_model_call("initial_patch")
        after = validate_budget(self.budget_snapshot())
        reserved = self._save("resource-reserved-" + reservation_id + ".json", {
            "reservation_id": reservation_id, "turn": turn, "intent_ref": intent,
            "kind": "model_call", "after": after})
        self._v2_state.record_resources({"model_calls": after["consumed"] - before["consumed"]},
            operation_id="model-reservation:" + reservation_id)
        self._v2_reservation = {"reservation_id": reservation_id, "turn": turn,
            "intent_ref": intent, "reserved_ref": reserved}
        self._v2_turn_count = turn
        self._v2_tool_charge_count = 0
        return result

    def charge_v2_tools(self, *, reads, observation_bytes):
        reservation = deepcopy(self._v2_reservation)
        index = self._v2_tool_charge_count + 1
        charge_id = reservation["reservation_id"] + ":tools:" + str(index)
        usage = {"reads": reads, "observation_bytes": observation_bytes}
        intent = self._save("tool-resource-intent-" + uuid4().hex + ".json", {
            "charge_id": charge_id, "reservation": reservation, "usage": usage})
        self._charge(reads=reads, observation_bytes=observation_bytes)
        self._v2_state.record_resources(usage, operation_id=charge_id)
        self._v2_tool_charge_count = index
        self._save("resource-tools-" + uuid4().hex + ".json", {"charge_id": charge_id,
            "intent_ref": intent, "reservation": reservation, "usage": usage,
            "reads": self._reads, "observation_bytes": self._observation_bytes,
            "model_budget": validate_budget(self.budget_snapshot())})

    def validate_handoff(self, outcome, expected_initial_context):
        return validate_task_agent_handoff_v2(outcome, store=self.store, workspace_root=self.workspace_root,
            expected_initial_context=expected_initial_context, allowed_edit_paths=self.allowed_edit_paths,
            worker_configuration_ref=self.worker_configuration_ref,
            expected_verification_identity=self.expected_verification_identity)

    def record_v2_tool_receipt(self, origin, output, session):
        from src.mod_agent.task_agent_v2_recovery import session_snapshot
        return self._save("tool-receipt-" + canonical_hash(origin["call_id"]) + ".json", {
            "origin": deepcopy(origin), "output": deepcopy(output), "session": session_snapshot(session),
            "task_head": self._v2_state.head})

    def __call__(self, state, implementation=None, evidence_pack=None):
        try:
            return self._call_with_recovery(state, implementation, evidence_pack)
        finally:
            lock = getattr(self, "_active_session_lock", None)
            if lock is not None:
                self._active_session_lock = None
                lock.__exit__(None, None, None)

    def _call_with_recovery(self, state, implementation=None, evidence_pack=None):
        from src.core.trace.execution_trace import current_trace
        from src.libs.llm.transport_observation import observe_transport
        from src.mod_agent.local_entry import LocalOllamaInitialPatchAdapter
        from src.mod_agent.model_admission import (
            assert_ollama_model_identity_stable,
            validate_ollama_context_window,
            validate_ollama_model_identity,
        )
        from src.mod_agent.patch_transport_bodies import PatchTransportBodyRecorder

        _require(not self._used, "SINGLE_INVOCATION_REQUIRED")
        self._used = True
        self.output_root.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(exist_ok=False)
        feature_ref = ArtifactRef(feature_id=state["feature_id"], artifact_type="feature_spec",
            path=state["artifacts"]["feature_spec"], artifact_hash=state["artifact_hashes"]["feature_spec"])
        initial = {"feature_spec_ref": asdict(feature_ref)}
        resumed = getattr(self, "_resume_v2", None)
        before = (deepcopy(resumed["host"]["original_budget_before"]) if resumed is not None
            else validate_budget(self.budget_snapshot()))
        details = {"run_id": state["run_id"], "request_id": state["request_id"], "workspace_root": str(self.workspace_root),
            "allowed_edit_paths": list(self.allowed_edit_paths), "model_profile": self.profile.to_dict(),
            "product_store_attempts": [], "completed_tool_records": [], "provider_turns": []}
        if resumed is not None:
            details = deepcopy(resumed["checkpoint"]["details"])
            counters = resumed["checkpoint"]["owner"]
            self._reads, self._observation_bytes = counters["reads"], counters["observation_bytes"]
            self._v2_turn_count, self._v2_tool_charge_count = counters["turn_count"], counters["tool_charge_count"]
            self._v2_reservation = deepcopy(counters["reservation"])
        task = session = loop = None
        loop_ref = finish_ref = None
        try:
            feature = self.store.semantic_read_immutable(feature_ref)["data"]
            _require(not feature.get("blocked", False), "REQUIREMENTS_BLOCKED")
            self.validate_providers()
            binding = _worker_binding(self.worker_configuration_ref, self.evaluator)
            self._verification_identity = self.evaluator.verification_identity()
            _require(isinstance(self._verification_identity, dict), "VERIFICATION_IDENTITY_REQUIRED")
            details.update(worker_binding=binding, expected_verification_identity=self.expected_verification_identity,
                expected_model_identity=deepcopy(self.expected_model_identity),
                source_before=_source_facts(self.allowed_read_roots), protocol="v2")
            worker = read_bound_json(self.worker_configuration_ref)
            profile_ref = worker["behavior_evaluator"]["kwargs"].get("profile_ref")
            _require(profile_ref is not None, "V2_REPOSITORY_PROFILE_REQUIRED")
            target = self.allowed_read_roots[0]
            paths = [str((self.workspace_root / path).relative_to(target).as_posix()) for path in self.allowed_edit_paths]
            root = self.store.root
            if resumed is not None:
                task = resumed["state"]
            else:
                authority = save_v2_artifact(root, "authority", {"schema": "task-agent-authority.v2",
                "scope": "file", "operation": "modify_existing", "allowed_edit_paths": paths,
                "read_roots": {"target": str(target)},
                "origin": "host_run_request_edit_scope_and_repository_profile",
                "worker_configuration_ref": self.worker_configuration_ref, "execution_profile_ref": profile_ref,
                "run_id": state["run_id"], "request_id": state["request_id"]})
                request_ref = save_v2_artifact(root, "request", {"original_request": feature["user_request"],
                "feature_spec_ref": asdict(feature_ref)})
                acceptance = save_v2_artifact(root, "acceptance", {"conditions": feature["acceptance_criteria"],
                "verification_identity": self.expected_verification_identity, "worker_binding": binding})
                resource = save_v2_artifact(root, "resource", {"initial_budget": before,
                "policy": before.get("policy"), "deadline": self.deadline})
                task = TaskAgentV2State.create(store_root=root, task_id=state["request_id"], target_root=target,
                allowed_edit_paths=paths, authority_ref=authority,
                task_requirements={"original_request": feature["user_request"],
                    "acceptance_criteria": feature["acceptance_criteria"],
                    "non_goals": feature.get("non_goals", [])},
                contract_fields={"original_request_ref": request_ref, "feature_spec_ref": asdict(feature_ref),
                    "public_acceptance_ref": acceptance, "execution_profile_ref": profile_ref,
                    "resource_policy_ref": resource})
            self._v2_state = task
            if resumed is None:
                task.record_resources({"model_calls": before["consumed"]}, operation_id="initial-resource-baseline")
            session = TaskAgentSessionV2(state=task, evaluator=self.evaluator, owner=self, feature_id=state["feature_id"])
            from src.mod_agent.task_agent_v2_recovery import (
                TaskAgentRecoveryJournal,
                restore_session,
                single_task_lock,
            )
            if resumed is not None:
                restore_session(session, resumed["checkpoint"]["session"])
                if self._experience is not None:
                    self._experience._terminal_receipt = deepcopy(resumed["checkpoint"].get("terminal_learning_receipt"))
            journal = TaskAgentRecoveryJournal(self, session, state, before, details,
                host_config_ref=resumed["checkpoint"]["host_config_ref"] if resumed is not None else None)
            if resumed is not None:
                journal.latest_ref = resumed["checkpoint_ref"]
                journal.latest_loop_state = deepcopy(resumed["checkpoint"]["loop"])
            self._recovery_journal = journal
            self.v2_checkpoint_callback = journal.save
            if not getattr(self, "_resume_lock_held", False):
                self._active_session_lock = single_task_lock(self.store.root, journal.lock_key)
                self._active_session_lock.__enter__()
            self._v2_session = session
            client = (self.client_factory or LocalOllamaInitialPatchAdapter._default_agent_turn_client)(self.profile)
            self._v2_client = client
            _require(getattr(client, "provider", None) == "ollama_chat_local"
                     and getattr(client, "model", None) == self.profile.model, "PROVIDER_CHANGED")

            def identity():
                observed = self.identity_reader(self.profile.base_url, self.profile.model)
                validate_ollama_model_identity(observed, expected_base_url=self.profile.base_url, expected_model=self.profile.model)
                assert_ollama_model_identity_stable(expected=self.expected_model_identity, observed=observed)
                validate_ollama_context_window(observed, requested_context_window=self.profile.context_window)
                return observed

            self._v2_identity = identity

            def complete(request):
                if resumed is not None and self._experience is not None:
                    self._experience._terminal_receipt = None
                request = self.prepare_decision_request(request, session)
                observed = identity()
                bind = getattr(client, "bind_context_identity", None)
                if callable(bind):
                    bind(model_identity=observed, owner_profile_sha256=canonical_hash(self.profile.to_dict()))
                index = len(details["provider_turns"]) + 1
                reservation = self._v2_reservation
                _require(reservation["turn"] == index, "V2_RESERVATION_TURN_CHANGED")
                event = {**deepcopy(reservation), "status": "requested", "request_ref": self._save(
                    f"request-{index:03d}.json", asdict(request)), "budget": validate_budget(self.budget_snapshot())}
                reservation["request_ref"] = event["request_ref"]
                event["dispatch_ref"] = self._save("provider-dispatch-" + reservation["reservation_id"] + ".json", event)
                details["provider_turns"].append(event)
                try:
                    trace = current_trace()
                    if trace is None:
                        response = client.complete_turn(request)
                    else:
                        with trace.model_call({"provider": client.provider, "model": client.model,
                                "turn_index": index, "usage_available": False, "response_received": False}) as observation:
                            bodies = PatchTransportBodyRecorder(record=event, observation=observation, body_root=self.output_root)
                            with observe_transport(observation.transport_started, observation.update,
                                    request_callback=observation.update, local_request_body_callback=bodies.callback):
                                response = client.complete_turn(request)
                            if isinstance(response, AgentTurnResponse):
                                observation.record_response(usage=response.usage.to_dict(), usage_available=True,
                                    response_sha256=canonical_hash(asdict(response)), response_origin="live_provider_response",
                                    stop_reason=response.stop_reason, tool_call_ids=[row.call_id for row in response.tool_calls])
                            bodies.publish_refs()
                    _require(isinstance(response, AgentTurnResponse), "RESPONSE_INVALID")
                    event.update(status="responded", response_ref=self._save(f"response-{index:03d}.json", asdict(response)),
                                 usage=response.usage.to_dict())
                    reservation.update(response_ref=event["response_ref"], call_ids=[call.call_id for call in response.tool_calls])
                    task.record_resources(response.usage.to_dict(),
                        operation_id="model-usage:" + reservation["reservation_id"])
                    identity()
                    return response
                except Exception as error:
                    event.update(status="failed", error_type=type(error).__name__)
                    raise
                finally:
                    self._save("provider-outcome-" + reservation["reservation_id"] + ".json", event)

            def record(event):
                self._save("loop-event-" + uuid4().hex + ".json", event)
                session.observe_dispatch_event(event)
                if event.get("event") == "tool_dispatch_completed":
                    calls = [row for row in session.actual_calls if row["call_id"] == event["call_id"]]
                    _require(len(calls) == 1, "COMPLETED_TOOL_ORIGIN_MISSING")
                    details["completed_tool_records"].append({**deepcopy(calls[0]), "role": event["role"],
                        "event": deepcopy(event)})
                if self.record_event is not None:
                    self.record_event(deepcopy(event))

            loop = session.run(SimpleNamespace(provider=client.provider, model=client.model, complete_turn=complete), record)
            loop_ref = self._save("loop-result.json", loop.to_dict())
            for row in details["completed_tool_records"]:
                raw = loop.call_records[row["call_id"]]["result"]["output"]
                row["result_ref"] = self._save("tool-result-" + uuid4().hex + ".json", raw)
            finished = [row for row in session.actual_calls if row["name"] == "finish"]
            if loop.status == "completed":
                _require(bool(finished), "REAL_FINISH_REQUIRED")
                finish_ref = finished[-1]["response_ref"]
            details["source_after"] = _source_facts(self.allowed_read_roots)
            _require(details["source_before"] == details["source_after"], "SOURCE_MUTATED")
            status = "candidate_ready" if loop.status == "completed" else "unknown" if loop.unknown_effects else "blocked"
            termination = deepcopy(loop.termination)
        except Exception as error:
            status = "unknown" if session is not None and session._check_inflight else "failed"
            termination = {"code": "TASK_AGENT_V2_EXECUTION_FAILED", "error_type": type(error).__name__, "reason": str(error)}
        candidate_ref = None
        if task is not None and task.current_revision_ref != task.contract["baseline_manifest_ref"]:
            candidate_ref = task.current_revision_ref
        getter = getattr(self.store, "task_agent_store_attempts", None)
        details["product_store_attempts"] = getter(self) if callable(getter) else []
        outcome = TaskAgentPatchOutcomeV2(status=status, termination=termination,
            initial_context=initial, final_context=initial,
            task_contract_ref=task.contract_ref if task else None, binding_registry_ref=task.registry_ref if task else None,
            candidate_ref=candidate_ref, check_ref=session.current_check if session else None,
            execution_ref=None, loop_result_ref=loop_ref, finish_response_ref=finish_ref,
            budget_before=before, budget_after=validate_budget(self.budget_snapshot()), unknown_effects=status == "unknown")
        ref = self._save("execution.json", {**outcome.to_dict(), **details})
        self._receipt = replace(outcome, execution_ref=ref)
        return self.execution_receipt()
