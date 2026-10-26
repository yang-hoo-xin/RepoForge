"""Product-owned v2 execution using the shared loop and repository checker.

为什么要有它：正常入口需建立可信任务合同并记录真实模型动作，不能用演示脚本或旧Plan伪造新协议交接。
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from src.libs.llm.agent_turn import AgentMessage, AgentTurnProviderError, AgentTurnResponse
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

_PROCESS_INSTANCE = uuid4().hex


def task_deadline_utc_from_environment():
    """Read a host-supplied limiting fact, not authority to extend resources."""
    value = os.environ.get("REPOFORGE_TASK_DEADLINE_UTC")
    if value is None:
        return None
    value = float(value)
    _require(math.isfinite(value), "WALL_DEADLINE_INVALID")
    return value


def earliest_task_deadline_utc(*values):
    selected = [value for value in values if value is not None]
    _require(all(type(value) in (int, float) and math.isfinite(value) for value in selected), "WALL_DEADLINE_INVALID")
    return min(selected) if selected else None


def capture_clock():
    return {"received_at": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
        "process_id": os.getpid(), "process_instance": _PROCESS_INSTANCE}


def task_timing_snapshot(origin_ref, event, status, native_execution_ref=None, *, observed_clock=None):
    """Measure to an observed boundary, never infer the end of an unknown task.

    为什么要有它：恢复继承原起点并包含中断等待；没有结束原件的中断不能被补写为完成。
    """
    clock = deepcopy(observed_clock) if observed_clock is not None else capture_clock()
    value = {"schema": "task-agent-time-observation.v2", "origin_ref": origin_ref,
        "event": event, "status": status, "observed_clock": clock, "native_execution_ref": native_execution_ref,
        "scope": "entry_received_to_observed_boundary_including_recovery_wait", "whole_task_finished": False,
        "duration_ms": None, "clock_basis": None, "timing_status": "origin_unavailable"}
    if origin_ref is None:
        return value
    origin = read_bound_json(origin_ref)
    _require(origin.get("schema") == "task-agent-entry-timing.v2", "TIMING_ORIGIN_INVALID")
    started = origin["entry_clock"]
    if (started.get("process_instance") and started["process_instance"] == clock.get("process_instance")
            and started.get("process_id") == clock.get("process_id")):
        elapsed = clock["monotonic"] - started["monotonic"]
        basis = "same_process_monotonic"
    else:
        begin, observed = datetime.fromisoformat(started["received_at"]), datetime.fromisoformat(clock["received_at"])
        _require(begin.tzinfo is not None and observed.tzinfo is not None, "TIMING_TIMEZONE_REQUIRED")
        elapsed, basis = (observed - begin).total_seconds(), "cross_process_utc_wall_clock"
    value.update(timing_status="observed" if elapsed >= 0 else "clock_regressed",
        duration_ms=max(0, int(elapsed * 1000)) if elapsed >= 0 else None, clock_basis=basis)
    return value


def classify_v2_technical_inputs(loop_request, prepared_request, initial_request, source_context_messages, prior_responses):
    """Audit closed-loop input channels; external preparation remains unknown."""
    initial, messages = list(initial_request["messages"]), list(loop_request["messages"])
    unknown = []
    if messages[:len(initial)] != initial:
        unknown.append("initial_contract_prefix_changed")
    for index, message in enumerate(messages[len(initial):], len(initial)):
        content = message.get("content")
        if message.get("role") == "assistant" and any(response.get("content") == content for response in prior_responses):
            continue
        if message in source_context_messages:
            continue
        if (message.get("role") == "user" and isinstance(content, dict)
                and content.get("message_type") == "task_agent_output_correction"
                and content.get("origin") == "execution_host" and index > 0
                and content.get("assistant_message_sha256") == canonical_hash(messages[index - 1])):
            continue
        unknown.append("unclassified_message:" + str(index))
    prepared_messages = list(prepared_request["messages"])
    if prepared_messages[:len(messages)] != messages:
        unknown.append("prepared_history_changed")
    for index, message in enumerate(prepared_messages[len(messages):], len(messages)):
        content = message.get("content")
        if (message.get("role") != "user" or not isinstance(content, dict)
                or set(content) not in ({"current_task_state"}, {"strategy_experience"})):
            unknown.append("unclassified_host_input:" + str(index))
    return {"policy": "closed_v2_technical_input_channels_v1", "scope": "v2_model_decision_inputs_after_initial_contract",
        "closed_loop_human_technical_interventions": 0 if not unknown else None,
        "unknown_input_sources": unknown, "outside_scope_human_technical_interventions": None,
        "excluded_from_technical_interventions": ["initial_requirements", "permission_approval"],
        "outside_scope": ["external_requirement_or_test_authoring", "strategy_experience_authorship", "out_of_band_assistance"]}


class TaskAgentExecutionV2(TaskAgentExecution):
    protocol = "v2"

    def __init__(self, *, experience_configuration_ref=None, deadline_utc=None, semantic_state_enabled=True,
                 source_disclosure=None, **kwargs):
        _require(type(semantic_state_enabled) is bool, "SEMANTIC_STATE_POLICY_INVALID")
        self.semantic_state_enabled = semantic_state_enabled
        self.source_disclosure = deepcopy(source_disclosure)
        clock, utc = time.monotonic(), time.time()
        relative = kwargs.get("deadline")
        if relative is not None:
            _require(type(relative) in (int, float) and math.isfinite(relative), "DEADLINE_INVALID")
        self.deadline_utc = earliest_task_deadline_utc(deadline_utc,
            None if relative is None else utc + (relative - clock))
        if self.deadline_utc is not None:
            kwargs["deadline"] = clock + (self.deadline_utc - utc)
        super().__init__(**kwargs)
        self._experience = None
        if experience_configuration_ref is not None:
            from src.mod_agent.external_patch import (
                ExternalOpenAIResponsesProfile,
                validate_openai_experience_configuration,
            )
            if isinstance(self.profile, ExternalOpenAIResponsesProfile):
                validate_openai_experience_configuration(experience_configuration_ref)
            from src.mod_agent.task_agent_v2_experience import TaskAgentV2Experience
            self._experience = TaskAgentV2Experience(self, experience_configuration_ref)

    def check_experience_boundary(self):
        if getattr(self, "_experience", None) is not None:
            self._experience.check_boundary()

    def observe_task_boundary(self, event, *, status):
        receipt = self.execution_receipt()
        native_ref = receipt.execution_ref if receipt is not None else None
        value = task_timing_snapshot(getattr(self, "_task_timing_origin_ref", None), event, status, native_ref)
        self._task_boundary_sequence = getattr(self, "_task_boundary_sequence", 0) + 1
        value["observation_sequence"] = self._task_boundary_sequence
        if getattr(self, "_v2_state", None) is not None:
            value["resource_event_ref"] = self._v2_state.head["event_ref"]
        return self._save("task-observation-" + uuid4().hex + ".json", value)

    def prepare_decision_request(self, request, session):
        from src.mod_agent.task_agent_v2_context import (
            investigation_observations,
            model_plan_view,
            project_history,
            with_history_projection,
        )
        from src.mod_agent.task_agent_v2_experiment_tools import experiment_configuration_facts
        from src.mod_agent.task_agent_v2_semantic_state import (
            semantic_contract,
            semantic_state_view,
            with_semantic_evidence,
        )
        from src.mod_agent.task_agent_v2_working_state import (
            update_working_state,
            working_state_view,
        )

        state = session.state
        observation_facts = investigation_observations(request, state.current_revision_ref)
        self._v2_working_state = update_working_state(getattr(self, "_v2_working_state", None), request,
            task_contract_ref=state.contract_ref, current_revision_ref=state.current_revision_ref,
            registry_ref=state.registry_ref, current_check_ref=session.current_check)
        working_ref = state.save("working-state", self._v2_working_state)
        statement_archive = state.save("context-archive", {"schema": "task-agent-context-archive.v2",
            "task_contract_ref": state.contract_ref, "exchanges": [
                {"model_statement": deepcopy(row)} for row in self._v2_working_state["model_statements"]]})

        registry = state.validate_registry(state.registry_ref)
        current_check = state.load(session.current_check) if session.current_check is not None else None
        current = {"origin": "execution_host", "task_contract_ref": state.contract_ref,
            "current_revision_ref": state.current_revision_ref, "binding_registry_ref": state.registry_ref,
            "current_candidate_ref": state.current_candidate_ref,
            "valid_bindings": [{"binding_id": key, "path": item["canonical_path"]} for key, item in sorted(registry["bindings"].items())],
            "current_check_ref": session.current_check, "current_check_status": current_check["status"] if current_check else None,
            "technical_plan_ref": state.head["plan_ref"], "baseline_revision_ref": state.contract["baseline_manifest_ref"],
            "model_technical_plan": model_plan_view(state),
            "history_projection": None,
            "working_state_ref": working_ref, "working_state": working_state_view(self._v2_working_state, statement_archive_ref=statement_archive),
            "investigation_observations": observation_facts,
            "cumulative_resources": deepcopy(state.head["resources"]),
            "model_call_budget": validate_budget(self.budget_snapshot()),
            "execution_window": {"deadline_utc": getattr(self, "deadline_utc", None),
                "remaining_wall_seconds": max(0.0, self.deadline - time.monotonic()) if self.deadline is not None else None,
                "scope": "host_wall_window_not_tokens_or_success_estimate",
                "enforcement": "cooperative_between_actions; outer_controller_owns_hard_process_stop"},
            "experiment_capabilities": experiment_configuration_facts(self),
            "baseline_usage": "Historical source reference; editing must name current_revision_ref and a matching source_view_ref."}
        semantic_enabled = getattr(self, "semantic_state_enabled", True)
        if semantic_enabled:
            current["semantic_state_contract"] = semantic_contract()
            current["semantic_work_state"] = semantic_state_view(
                self._v2_working_state, self._v2_working_state["boundary"])
        prepared = replace(request, messages=(*request.messages, AgentMessage("user", {"current_task_state": current})))
        if getattr(self, "_experience", None) is not None:
            prepared = self._experience.prepare_decision_request(prepared, session)
        prepared, projection = project_history(prepared, state,
            context_window=getattr(self.profile, "context_window", None),
            output_budget=getattr(self.profile, "output_budget", None), provider=getattr(self, "_v2_client", None),
            decorate=with_semantic_evidence if semantic_enabled else None)
        self._v2_context_projection = projection
        prepared = with_history_projection(prepared, projection) if projection is not None else prepared
        return with_semantic_evidence(prepared) if semantic_enabled else prepared

    def complete_terminal_learning(self):
        if getattr(self, "_experience", None) is None:
            from src.mod_agent.task_agent_v2_experience import seal_current_public_episode
            # 为什么要有它：无经验配置仍保留公开执行经历；封存失败必须披露，不能反转既有业务结果。
            try:
                episode_ref = seal_current_public_episode(self)
            except Exception as error:
                return {"status": "not_configured", "episode_status": "episode_unavailable",
                    "episode_ref": None, "diagnostic": {"error_type": type(error).__name__, "message": str(error)}}
            return {"status": "not_configured", "episode_status": "sealed" if episode_ref else "no_observed_episode",
                "episode_ref": episode_ref}
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

    def require_model_send_window(self, purpose):
        termination = self.stop_v2_check()
        if termination is not None:
            receipt = self._save("model-not-sent-" + uuid4().hex + ".json", {
                "status": "not_sent", "purpose": purpose, "termination": termination,
                "deadline_utc": getattr(self, "deadline_utc", None),
                "budget": validate_budget(self.budget_snapshot()), "transport_attempted": False})
            raise AgentTurnProviderError("Task stopped before model transport", diagnostic={
                "termination": termination, "transport_attempted": False, "not_sent_receipt_ref": receipt})

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
        origin_path = self.workspace_root / "entry-timing.json"
        if resumed is not None:
            self._task_timing_origin_ref = deepcopy(resumed["host"].get("task_timing_origin_ref"))
        elif origin_path.is_file():
            from src.mod_agent.domain.full_project_compilation import _ref
            self._task_timing_origin_ref = _ref(origin_path)
        else:
            self._task_timing_origin_ref = None
        if self._task_timing_origin_ref is not None:
            origin = read_bound_json(self._task_timing_origin_ref)
            _require(origin["run_id"] == state["run_id"] and origin["request_id"] == state["request_id"], "TIMING_TASK_CHANGED")
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
            self._v2_working_state = deepcopy(counters.get("working_state"))
            self._v2_context_projection = deepcopy(counters.get("context_projection"))
        details["task_timing_origin_ref"] = deepcopy(self._task_timing_origin_ref)
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
                disclosure_fields = {}
                if self.source_disclosure is not None:
                    disclosure_fields["source_disclosure_ref"] = save_v2_artifact(root, "source-disclosure", {
                        "schema": "task-agent-source-disclosure.v2", "record": self.source_disclosure,
                        "model_profile": asdict(self.profile)})
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
                    "resource_policy_ref": resource, **disclosure_fields})
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
            from src.mod_agent.external_patch import (
                ExternalOpenAIResponsesProfile,
                validate_openai_task_identity,
            )
            openai = isinstance(self.profile, ExternalOpenAIResponsesProfile)
            _require(getattr(client, "provider", None) == ("openai_responses" if openai else "ollama_chat_local")
                     and getattr(client, "model", None) == self.profile.model, "PROVIDER_CHANGED")

            def identity():
                observed = self.identity_reader(self.profile.base_url, self.profile.model)
                if openai:
                    validate_openai_task_identity(self.profile, observed, self.expected_model_identity)
                    return observed
                validate_ollama_model_identity(observed, expected_base_url=self.profile.base_url, expected_model=self.profile.model)
                assert_ollama_model_identity_stable(expected=self.expected_model_identity, observed=observed)
                validate_ollama_context_window(observed, requested_context_window=self.profile.context_window)
                return observed

            self._v2_identity = identity

            def complete(request):
                from src.mod_agent.task_agent_loop import measured_turn_usage
                if resumed is not None and self._experience is not None:
                    self._experience._terminal_receipt = None
                loop_request = asdict(request)
                observed = identity()
                bind = getattr(client, "bind_context_identity", None)
                if callable(bind):
                    bind(model_identity=observed, owner_profile_sha256=canonical_hash(self.profile.to_dict()))
                request = self.prepare_decision_request(request, session)
                index = len(details["provider_turns"]) + 1
                reservation = self._v2_reservation
                _require(reservation["turn"] == index, "V2_RESERVATION_TURN_CHANGED")
                event = {**deepcopy(reservation), "status": "requested", "request_ref": self._save(
                    f"request-{index:03d}.json", asdict(request)), "budget": validate_budget(self.budget_snapshot())}
                session.bind_actual_request(event["request_ref"])
                host = task.load(journal.host_config_ref)
                prior_refs = [row["response_ref"] for row in details["provider_turns"] if row.get("response_ref")]
                input_audit = classify_v2_technical_inputs(loop_request, asdict(request), read_bound_json(host["initial_request_ref"]),
                    host.get("source_context_messages", []), [read_bound_json(ref) for ref in prior_refs])
                event["technical_input_audit_ref"] = self._save("technical-input-audit-" + uuid4().hex + ".json", {
                    "schema": "task-agent-technical-input-audit.v2", "host_config_ref": journal.host_config_ref,
                    "loop_request_ref": self._save("loop-request-" + uuid4().hex + ".json", loop_request),
                    "actual_request_ref": event["request_ref"], "prior_response_refs": prior_refs, "assessment": input_audit})
                reservation["request_ref"] = event["request_ref"]
                event["dispatch_ref"] = self._save("provider-dispatch-" + reservation["reservation_id"] + ".json", event)
                details["provider_turns"].append(event)
                try:
                    trace = current_trace()
                    if trace is None:
                        self.require_model_send_window("technical_decision")
                        response = client.complete_turn(request)
                    else:
                        with trace.model_call({"provider": client.provider, "model": client.model,
                                "turn_index": index, "usage_available": False, "response_received": False}) as observation:
                            bodies = PatchTransportBodyRecorder(record=event, observation=observation,
                                body_root=self.output_root, provider=client.provider)
                            with observe_transport(observation.transport_started, observation.update,
                                    request_callback=observation.update, local_request_body_callback=bodies.callback):
                                self.require_model_send_window("technical_decision")
                                response = client.complete_turn(request)
                            if isinstance(response, AgentTurnResponse):
                                measured = measured_turn_usage(response)
                                observation.record_response(usage=measured, usage_available=measured is not None,
                                    response_sha256=canonical_hash(asdict(response)), response_origin="live_provider_response",
                                    stop_reason=response.stop_reason, tool_call_ids=[row.call_id for row in response.tool_calls])
                            bodies.publish_refs()
                    _require(isinstance(response, AgentTurnResponse), "RESPONSE_INVALID")
                    measured = measured_turn_usage(response)
                    event.update(status="responded", response_ref=self._save(f"response-{index:03d}.json", asdict(response)),
                                 usage=measured, usage_available=measured is not None)
                    from src.mod_agent.task_agent_v2_working_state import retain_model_statement
                    self._v2_working_state = retain_model_statement(getattr(self, "_v2_working_state", None),
                        response, response_ref=event["response_ref"], source_boundary=deepcopy(self._v2_working_state.get("boundary")))
                    if self.semantic_state_enabled:
                        from src.mod_agent.task_agent_v2_semantic_state import record_semantic_state
                        # 为什么要有它：语义记录属于本次原始模型响应，只能引用实际发送请求；动作尚未执行，不能借未来结果补写判断。
                        self._v2_working_state = record_semantic_state(self._v2_working_state, asdict(response),
                            actual_request=read_bound_json(event["request_ref"]),
                            request_ref=event["request_ref"], response_ref=event["response_ref"])
                    reservation.update(response_ref=event["response_ref"], call_ids=[call.call_id for call in response.tool_calls])
                    task.record_resources(measured if measured is not None else {"unknown_usage_calls": 1},
                        operation_id="model-usage:" + reservation["reservation_id"])
                    identity()
                    return response
                except Exception as error:
                    event.update(status="failed", error_type=type(error).__name__)
                    diagnostic = deepcopy(getattr(error, "diagnostic", None) or {})
                    if diagnostic.get("raw_response") is not None:
                        event["provider_error_ref"] = self._save("provider-error-" + reservation["reservation_id"] + ".json",
                            {"error_type": type(error).__name__, "diagnostic": diagnostic})
                    measured = event.get("usage", diagnostic.get("measured_usage"))
                    event.update(usage=deepcopy(measured), usage_available=measured is not None)
                    task.record_resources(measured if measured is not None else {"unknown_usage_calls": 1},
                        operation_id="model-usage:" + reservation["reservation_id"])
                    diagnostic.update(measured_usage=deepcopy(measured), usage_available=measured is not None)
                    try:
                        error.diagnostic = diagnostic
                    except (AttributeError, TypeError):
                        pass
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
        details["timing_observation_ref"] = self.observe_task_boundary("native_execution_return", status=status)
        ref = self._save("execution.json", {**outcome.to_dict(), **details})
        self._receipt = replace(outcome, execution_ref=ref)
        return self.execution_receipt()
