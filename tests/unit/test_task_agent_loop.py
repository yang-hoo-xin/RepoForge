"""Scripted task-loop controls, including actual Ollama wire parsing; no providers.

为什么要有它：连续会话要证明模型实际选择的动作、预算和绑定，不用Graph外层重跑或伪造新tool ID代替。
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolCall,
    AgentToolDefinition,
    AgentToolResult,
    AgentTurnProviderError,
    AgentTurnRequest,
    AgentTurnResponse,
    OllamaChatAgentAdapter,
)
from src.mod_agent.task_agent_loop import (
    PreparedTaskAction,
    TaskActionInputError,
    TaskAgentAction,
    run_task_agent_loop,
)


class Budget:
    def __init__(self, maximum=20, max_reads=8, max_bytes=100_000):
        self.maximum, self.consumed = maximum, 0
        self.max_reads, self.max_bytes = max_reads, max_bytes
        self.reads = self.bytes = 0

    def snapshot(self):
        return {"policy": "task_total_only", "maximum": self.maximum, "consumed": self.consumed,
                "remaining": self.maximum - self.consumed, "by_stage": {"patch": {"consumed": self.consumed}},
                "count_basis": "reserved_provider_invocations_not_transport_proof"}

    def reserve(self):
        assert self.consumed < self.maximum
        self.consumed += 1

    def charge(self, *, reads, observation_bytes):
        if self.reads + reads > self.max_reads or self.bytes + observation_bytes > self.max_bytes:
            raise ValueError("shared tool budget exhausted")
        self.reads += reads
        self.bytes += observation_bytes


def call(name, arguments=None, identifier=None):
    return AgentToolCall(identifier or name, name, {} if arguments is None else arguments)


def response(*calls, stop_reason=None):
    return AgentTurnResponse("scripted", "controlled", "", tuple(calls), stop_reason=stop_reason,
                             provider_state={"calls": [asdict(item) for item in calls]})


class Client:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def complete_turn(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Owners:
    def __init__(self, budget):
        self.budget = budget
        self.candidate = "c0"
        self.fixed = False
        self.report = None
        self.plan_revision = 0
        self.effects = []
        self.raise_after_effect = None

    def prepare(self, role, arguments):
        if role == "source_read":
            if "binding_id" not in arguments or type(arguments["binding_id"]) is not str:
                raise TaskActionInputError(("binding_id requires a string",))
            if set(arguments) != {"binding_id"} or arguments["binding_id"] not in {"one", "two"}:
                raise PermissionError("not an authorized source")
            version, cost = "source-v1", 1
        elif role in {"candidate_check", "finish"}:
            if arguments != {"candidate_id": self.candidate}:
                raise PermissionError("stale candidate selection")
            version, cost = self.state_version(), 0
        else:
            version, cost = "scope-v1", 0
        return PreparedTaskAction(deepcopy(arguments), cost, version)

    def state_version(self):
        return self.candidate + ":" + json.dumps(self.report, sort_keys=True)

    def execute(self, role, arguments):
        self.effects.append((role, deepcopy(arguments)))
        if self.raise_after_effect == role:
            raise OSError("controlled operation completion unknown")
        if role == "upstream":
            self.budget.reserve()
            self.plan_revision += 1
            return {"plan_revision": self.plan_revision, "authority_changed": False}
        if role == "candidate_submit":
            self.candidate, self.fixed, self.report = arguments["candidate_id"], arguments["fixed"], None
            return {"candidate_id": self.candidate}
        if role == "candidate_check":
            self.report = {"candidate_id": self.candidate, "passed": self.fixed, "report_id": "checked-" + self.candidate}
            return deepcopy(self.report)
        if role == "finish":
            return {"candidate_id": self.candidate}
        return {"observed": arguments, "source_version": "source-v1"}

    def completion(self, *, prepared, output):
        accepted = bool(self.report and self.report["candidate_id"] == output["candidate_id"] == self.candidate
                        and self.report["passed"] is True and prepared.version_identity == self.state_version())
        return {"accepted": accepted, "candidate_id": self.candidate,
                "report_id": None if self.report is None else self.report["report_id"]}

    def reuse(self, role, prepared, previous):
        return {"mode": "refresh" if role == "finish" else "reuse",
                "version_identity": prepared.version_identity,
                "evidence": {"checked_version": self.state_version() if role == "finish" else "source-v1"}}

    def actions(self):
        result = []
        for role in ("source_read", "source_search", "upstream", "candidate_submit", "candidate_check", "finish"):
            result.append(TaskAgentAction(AgentToolDefinition(role, "Controlled " + role, {"type": "object"}),
                role, lambda arguments, role=role: self.prepare(role, arguments),
                lambda arguments, role=role: self.execute(role, arguments),
                role in {"source_read", "source_search", "finish"},
                lambda prepared, previous, role=role: self.reuse(role, prepared, previous)))
        return result


def run(client, owners, *, actions=None, **options):
    return run_task_agent_loop(client=client,
        initial_request=options.pop("initial_request", AgentTurnRequest((AgentMessage("user", "Frozen task and failure"),))),
        actions=owners.actions() if actions is None else actions,
        reserve_model_call=owners.budget.reserve, budget_snapshot=owners.budget.snapshot,
        charge_tool_budget=owners.budget.charge, completion_check=owners.completion, **options)


@pytest.mark.parametrize("fault", ["foreign", "missing", "duplicate"])
def test_definition_refresh_cannot_change_registered_names_or_spend_a_call(fault):
    owners = Owners(Budget())
    definitions = [action.definition for action in owners.actions()]
    if fault == "foreign":
        definitions[-1] = replace(definitions[-1], name="unregistered_execution")
    elif fault == "missing":
        definitions.pop()
    else:
        definitions[-1] = definitions[0]
    client = Client([])
    result = run(client, owners, tool_definitions=lambda: definitions)
    assert result.termination["code"] == "TASK_AGENT_TOOL_DEFINITIONS_INVALID"
    assert not client.requests and not owners.effects and owners.budget.consumed == 0


def test_definition_refresh_deepcopies_owner_schemas_without_mutating_past_requests():
    owners = Owners(Budget())
    definitions = [action.definition for action in owners.actions()]
    submit = next(item for item in definitions if item.name == "candidate_submit")
    submit.parameters["description"] = "before"
    requests = []

    class Scripted:
        def complete_turn(self, request):
            requests.append(request)
            if len(requests) == 1:
                submit.parameters["description"] = "after"
                return response(call("candidate_submit", {"candidate_id": "c1", "fixed": True}))
            if len(requests) == 2:
                return response(call("candidate_check", {"candidate_id": "c1"}))
            return response(call("finish", {"candidate_id": "c1"}))

    result = run(Scripted(), owners, tool_definitions=lambda: definitions)
    assert result.status == "completed", result.termination
    assert next(item for item in requests[0].tools if item.name == "candidate_submit").parameters["description"] == "before"
    assert all(next(item for item in request.tools if item.name == "candidate_submit").parameters["description"] == "after"
        for request in requests[1:])
    assert [role for role, _ in owners.effects] == ["candidate_submit", "candidate_check", "finish"]


def test_failed_candidate_read_upstream_new_candidate_check_and_finish_keep_one_history():
    owners = Owners(Budget())
    client = Client([
        response(call("candidate_check", {"candidate_id": "c0"}, "check0")),
        response(call("source_read", {"binding_id": "one"}, "read1")),
        response(call("upstream", {"question": "revise technical assumption"}, "upstream1")),
        response(call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "submit1")),
        response(call("candidate_check", {"candidate_id": "c1"}, "check1")),
        response(call("finish", {"candidate_id": "c1"}, "finish1")),
    ])
    result = run(client, owners)
    assert result.status == "completed" and result.completion["candidate_id"] == "c1"
    assert result.model_turns == 6 and result.model_budget["consumed"] == 7
    assert owners.plan_revision == 1 and result.read_units == owners.budget.reads == 1
    after_upstream = client.requests[3]
    assert [x.call_id for exchange in after_upstream.historical_tool_exchanges for x in exchange.tool_results] == ["check0", "read1"]
    assert after_upstream.tool_results[0].call_id == "upstream1"
    returned = [item for exchange in result.request.historical_tool_exchanges for item in exchange.tool_results]
    returned.extend(result.request.tool_results)
    actual_bytes = sum(len(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()) for item in returned)
    assert result.observation_bytes == owners.budget.bytes == actual_bytes


def test_missing_upstream_is_not_a_qualified_equal_tool_surface():
    owners, client = Owners(Budget()), Client([])
    result = run(client, owners, actions=[item for item in owners.actions() if item.role != "upstream"])
    assert result.termination["code"] == "TASK_AGENT_CAPABILITIES_INCOMPLETE"
    assert result.termination["detail"]["missing_roles"] == ["upstream"]
    assert owners.budget.consumed == 0 and client.requests == []


@pytest.mark.parametrize("budget_kind", ["models", "reads", "bytes"])
def test_shared_budget_refusal_never_hides_extra_send_or_observation(budget_kind):
    budget = Budget(maximum=0 if budget_kind == "models" else 2,
                    max_reads=0 if budget_kind == "reads" else 8,
                    max_bytes=0 if budget_kind == "bytes" else 100_000)
    owners = Owners(budget)
    client = Client([response(call("source_read", {"binding_id": "one"}))])
    result = run(client, owners)
    assert result.status == "blocked"
    assert len(client.requests) == (0 if budget_kind == "models" else 1)
    assert len(owners.effects) == (1 if budget_kind == "bytes" else 0)
    assert result.observation_bytes == 0
    assert bool(result.unadmitted_tool_results) is (budget_kind == "bytes")


def test_failed_provider_send_is_reserved_and_not_retried():
    owners = Owners(Budget())
    client = Client([AgentTurnProviderError("controlled transport failure")])
    result = run(client, owners)
    assert result.termination["code"] == "TASK_AGENT_PROVIDER_TURN_FAILED"
    assert result.model_budget["consumed"] == result.model_turns == 1
    assert len(client.requests) == 1 and owners.effects == []
    assert result.usage_by_turn[-1]["usage_available"] is False


@pytest.mark.parametrize("stopped", ["length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal"])
def test_stopped_provider_output_cannot_dispatch_tools(stopped):
    owners = Owners(Budget())
    result = run(Client([response(call("candidate_submit", {"candidate_id": "c1", "fixed": True}), stop_reason=stopped)]), owners)
    assert result.status == "blocked" and owners.effects == []


def test_post_response_identity_gate_runs_before_any_tool():
    owners = Owners(Budget())

    def reject(_response):
        raise PermissionError("changed model identity")

    result = run(Client([response(call("source_read", {"binding_id": "one"}))]), owners, response_admission=reject)
    assert result.termination["code"] == "TASK_AGENT_RESPONSE_NOT_ADMITTED" and owners.effects == []


@pytest.mark.parametrize("unsafe", ["out_of_scope", "mixed_effects", "duplicate_id"])
def test_entire_batch_preflight_prevents_partial_reads_or_argument_feedback(unsafe):
    owners = Owners(Budget())
    bad = call("source_read", {}, "missing")
    other = call("source_read", {"binding_id": "outside"}, "unauthorized")
    if unsafe == "mixed_effects":
        other = call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "submit")
    elif unsafe == "duplicate_id":
        other = call("source_read", {"binding_id": "one"}, "missing")
    result = run(Client([response(bad, other)]), owners)
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert not result.tool_argument_correction_used and owners.effects == [] and owners.budget.bytes == 0


def test_same_call_id_with_changed_arguments_is_rejected():
    owners = Owners(Budget())
    result = run(Client([response(call("source_read", {"binding_id": "one"}, "same")),
                         response(call("source_read", {"binding_id": "two"}, "same"))]), owners)
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED" and len(owners.effects) == 1


def test_pure_mixed_batch_gets_one_not_executed_correction_then_same_ids_execute_once():
    owners = Owners(Budget())
    read = call("source_read", {"binding_id": "one"}, "mixed-read")
    upstream = call("upstream", {"question": "reassess"}, "mixed-upstream")
    client = Client([response(read, upstream), response(read), response(upstream),
        response(call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "submit")),
        response(call("candidate_check", {"candidate_id": "c1"}, "check")),
        response(call("finish", {"candidate_id": "c1"}, "finish"))])
    result = run(client, owners)
    assert result.status == "completed", result.termination
    assert result.tool_argument_correction_used is True
    feedback = client.requests[1].tool_results
    assert [row.call_id for row in feedback] == ["mixed-read", "mixed-upstream"]
    assert all(row.output["status"] == "rejected_not_executed"
        and row.output["code"] == "TASK_AGENT_MIXED_TOOL_BATCH" and row.output["read_cost"] == 0 for row in feedback)
    assert [role for role, _ in owners.effects] == ["source_read", "upstream", "candidate_submit", "candidate_check", "finish"]
    assert result.read_units == 1 and result.model_turns == 6 and result.model_budget["consumed"] == 7
    assert not any(row.get("event") == "tool_dispatch_prepared" and row.get("turn") == 1 for row in result.events)


def _mixed_calls():
    return (call("source_read", {"binding_id": "one"}, "mixed-read"),
        call("upstream", {"question": "reassess"}, "mixed-upstream"))


@pytest.mark.parametrize("limit", ["second_error", "model_budget", "observation_budget"])
def test_mixed_batch_correction_stops_within_original_limits_without_dispatch(limit):
    owners = Owners(Budget(maximum=1 if limit == "model_budget" else 20,
        max_bytes=0 if limit == "observation_budget" else 100_000))
    client = Client([response(*_mixed_calls()), response(*_mixed_calls())])
    result = run(client, owners)
    expected = {"second_error": "TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED",
        "model_budget": "TASK_AGENT_MODEL_BUDGET_EXHAUSTED", "observation_budget": "TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED"}
    assert result.termination["code"] == expected[limit]
    assert owners.effects == [] and owners.budget.reads == result.read_units == 0
    assert len(client.requests) == (2 if limit == "second_error" else 1)
    assert result.tool_argument_correction_used is (limit == "second_error")
    assert bool(result.unadmitted_tool_results) is (limit == "observation_budget")
    if limit == "second_error":
        feedback = client.requests[1].tool_results
        assert owners.budget.bytes == sum(len(json.dumps(asdict(row), sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()) for row in feedback)
    else:
        assert owners.budget.bytes == 0


@pytest.mark.parametrize("bad", ["ordinary_argument", "outside_scope", "duplicate_id", "spoofed_mixed_error", "snapshot_changed"])
def test_mixed_batch_with_any_other_fault_is_hard_rejected_without_correction(bad):
    owners = Owners(Budget())
    read, upstream = _mixed_calls()
    actions = owners.actions()
    if bad == "ordinary_argument":
        read = replace(read, arguments={})
    elif bad == "outside_scope":
        read = replace(read, arguments={"binding_id": "outside"})
    elif bad == "duplicate_id":
        upstream = replace(upstream, call_id=read.call_id)
    else:
        def refuse(_arguments):
            raise ValueError("TASK_AGENT_MIXED_TOOL_BATCH" if bad == "spoofed_mixed_error" else "SOURCE_SNAPSHOT_CHANGED")
        actions = [replace(action, prepare=refuse) if action.role == "source_read" else action for action in actions]
    result = run(Client([response(read, upstream)]), owners, actions=actions)
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert owners.effects == [] and owners.budget.bytes == owners.budget.reads == 0
    assert result.tool_argument_correction_used is False


@pytest.mark.parametrize("first_kind", ["argument", "mixed"])
def test_argument_and_mixed_errors_share_one_correction_even_after_history_restore(first_kind):
    owners = Owners(Budget())
    first = (response(call("source_read", {}, "bad-argument")) if first_kind == "argument" else response(*_mixed_calls()))
    retained = run(Client([first, AgentTurnProviderError("controlled pause")]), owners)
    assert retained.tool_argument_correction_used
    second = response(*_mixed_calls()) if first_kind == "argument" else response(call("source_read", {}, "bad-argument"))
    before = owners.budget.consumed
    result = run(Client([second]), owners, initial_request=retained.request,
        prior_call_records=retained.call_records)
    assert result.termination["code"] == "TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED"
    assert owners.budget.consumed == before + 1 and owners.effects == []


def test_corrected_never_dispatched_ids_resume_from_native_history_without_an_extra_chance():
    owners = Owners(Budget())
    owners.fixed = True
    retained = run(Client([response(*_mixed_calls()), AgentTurnProviderError("controlled pause")]), owners)
    assert owners.budget.reads == 0 and not owners.effects
    before = owners.budget.consumed
    read, upstream = _mixed_calls()
    result = run(Client([response(read), response(upstream),
        response(call("candidate_check", {"candidate_id": "c0"}, "check")),
        response(call("finish", {"candidate_id": "c0"}, "finish"))]), owners,
        initial_request=retained.request, prior_call_records=retained.call_records)
    assert result.status == "completed" and result.tool_argument_correction_used
    assert [role for role, _ in owners.effects] == ["source_read", "upstream", "candidate_check", "finish"]
    assert owners.budget.reads == 1 and owners.budget.consumed == before + 5


def test_corrected_never_dispatched_same_id_cannot_change_version():
    owners = Owners(Budget())
    retained = run(Client([response(*_mixed_calls()), AgentTurnProviderError("controlled pause")]), owners)
    actions = owners.actions()
    source = next(action for action in actions if action.role == "source_read")
    actions = [replace(action, prepare=lambda arguments: replace(source.prepare(arguments), version_identity="changed-source"))
        if action.role == "source_read" else action for action in actions]
    result = run(Client([response(_mixed_calls()[0])]), owners, actions=actions,
        initial_request=retained.request, prior_call_records=retained.call_records)
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert "TASK_AGENT_TOOL_VERSION_CHANGED" in result.termination["detail"]["reasons"] and not owners.effects


def test_mixed_batch_does_not_downgrade_an_already_executed_effect_for_replay():
    owners = Owners(Budget())
    submit = call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "same-submit")
    client = Client([response(submit), response(call("source_read", {"binding_id": "one"}, "read"), submit)])
    result = run(client, owners)
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert [role for role, _ in owners.effects] == ["candidate_submit"]
    assert result.call_records["same-submit"]["status"] == "completed" and not result.tool_argument_correction_used


def test_mixed_batch_with_prior_executed_read_stays_hard_rejected_without_rewriting_history():
    owners = Owners(Budget())
    read, upstream = _mixed_calls()
    result = run(Client([response(read), response(read, upstream)]), owners)
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert result.call_records[read.call_id]["status"] == "completed"
    assert result.call_records[read.call_id]["result"]["output"]["execution"] == "executed"
    assert owners.effects == [("source_read", {"binding_id": "one"})]
    assert owners.budget.reads == 1 and not result.tool_argument_correction_used


def test_mixed_batch_cannot_correct_a_prior_unknown_effect_id():
    from src.mod_agent.model_qualification import canonical_hash

    owners = Owners(Budget())
    owners.budget.consumed = 2
    read, upstream = _mixed_calls()
    prior = AgentToolResult(upstream.call_id, upstream.name, {"status": "unknown"})
    initial = AgentTurnRequest((AgentMessage("user", "Original task"),), tool_results=(prior,), provider_state={"prior": True})
    record = {"name": upstream.name, "arguments_sha256": canonical_hash(upstream.arguments),
        "status": "unknown", "result": asdict(prior)}
    result = run(Client([response(read, upstream)]), owners, initial_request=initial,
        prior_call_records={upstream.call_id: record})
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert "TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE" in result.termination["detail"]["reasons"]
    assert result.call_records[upstream.call_id]["status"] == "unknown"
    assert not result.tool_argument_correction_used and owners.effects == [] and owners.budget.consumed == 3


@pytest.mark.parametrize("role", ["upstream", "candidate_submit", "candidate_check", "finish"])
def test_each_solo_action_in_a_fresh_valid_mixed_batch_is_not_executed(role):
    owners = Owners(Budget())
    arguments = ({"question": "reassess"} if role == "upstream" else
        {"candidate_id": "c1", "fixed": True} if role == "candidate_submit" else {"candidate_id": "c0"})
    client = Client([response(call("source_read", {"binding_id": "one"}, "read"), call(role, arguments, "solo")),
        AgentTurnProviderError("stop after observing one correction")])
    result = run(client, owners)
    assert result.tool_argument_correction_used and result.termination["code"] == "TASK_AGENT_PROVIDER_TURN_FAILED"
    assert owners.effects == [] and owners.budget.reads == 0
    assert all(row.output["status"] == "rejected_not_executed" for row in client.requests[1].tool_results)


def test_same_version_read_reuses_only_after_owner_confirmation_and_charges_visible_bytes():
    owners = Owners(Budget(maximum=2))
    one = call("source_read", {"binding_id": "one"}, "stable")
    result = run(Client([response(one), response(one)]), owners)
    assert result.termination["code"] == "TASK_AGENT_MODEL_BUDGET_EXHAUSTED"
    assert len(owners.effects) == 1 and result.read_units == 1
    assert result.request.tool_results[0].output["execution"] == "reused"
    assert result.request.tool_results[0].output["reuse_confirmation"]["evidence"]
    assert result.observation_bytes > len(json.dumps(result.request.tool_results[0].output).encode())


def test_no_reexecution_of_candidate_submit_or_unknown_effect():
    owners = Owners(Budget())
    submit = call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "same")
    repeated = run(Client([response(submit), response(submit)]), owners)
    assert repeated.status == "blocked" and len(owners.effects) == 1
    owners = Owners(Budget())
    owners.raise_after_effect = "candidate_check"
    client = Client([response(call("candidate_check", {"candidate_id": "c0"}, "check"))])
    unknown = run(client, owners)
    assert unknown.unknown_effects and unknown.termination["code"] == "TASK_AGENT_ACTION_OUTCOME_UNKNOWN"
    assert len(owners.effects) == len(client.requests) == 1 and unknown.call_records["check"]["status"] == "unknown"
    pending = [row for row in unknown.events if row["event"] == "tool_dispatch_started"]
    assert pending == [{"event": "tool_dispatch_started", "call_id": "check",
                        "role": "candidate_check", "operation_state": "pending"}]


def test_finish_does_not_accept_stale_report_or_model_text():
    owners = Owners(Budget(maximum=1))
    owners.report = {"candidate_id": "old", "passed": True, "report_id": "stale"}
    result = run(Client([response(call("finish", {"candidate_id": "c0"}))]), owners)
    assert result.status == "blocked" and result.completion is None
    assert result.request.tool_results[0].output["completion_check"]["accepted"] is False
    text = run(Client([AgentTurnResponse("scripted", "controlled", "Everything passed"),
        AgentTurnResponse("scripted", "controlled", "Everything passed")]), Owners(Budget()))
    assert text.termination["code"] == "TASK_AGENT_FINISH_TOOL_REQUIRED"


def test_plain_text_after_rejected_finish_gets_one_host_correction_then_real_repair():
    owners = Owners(Budget())
    text = "The check failed. You should revise this candidate."
    client = Client([response(call("finish", {"candidate_id": "c0"}, "rejected-finish")),
        AgentTurnResponse("scripted", "controlled", text, stop_reason="stop"),
        response(call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "submit")),
        response(call("candidate_check", {"candidate_id": "c1"}, "check")),
        response(call("finish", {"candidate_id": "c1"}, "finish"))])
    result = run(client, owners)
    assert result.status == "completed", result.termination
    assert result.output_correction_used and not result.tool_argument_correction_used
    corrected = client.requests[2]
    assert corrected.messages[-2] == AgentMessage("assistant", text)
    notice = corrected.messages[-1]
    assert notice.role == "user" and notice.content["message_type"] == "task_agent_output_correction"
    assert notice.content["model_budget"]["consumed"] == 2
    assert notice.content["model_budget"]["remaining"] == 18
    assert notice.content["resource_information"] == {"read_budget": "not_supplied", "observation_budget": "not_supplied"}
    assert set(notice.content["available_actions"]) == {action.definition.name for action in owners.actions()}
    assert not corrected.tool_results and corrected.provider_state is None
    assert corrected.historical_tool_exchanges[-1].tool_results[0].call_id == "rejected-finish"
    assert "The check failed" not in json.dumps(notice.content)
    assert "Session" not in json.dumps(notice.content)
    recorded = [row for exchange in result.request.historical_tool_exchanges for row in exchange.tool_results]
    recorded.extend(result.request.tool_results)
    def canonical(row):
        return len(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())

    assert owners.budget.bytes == result.observation_bytes == sum(map(canonical, recorded)) + canonical(notice)


def native_client(responses, captured):
    def send(_url, payload, _headers):
        captured.append(deepcopy(payload))
        return responses.pop(0)

    return OllamaChatAgentAdapter(base_url="http://127.0.0.1:11434", model="controlled",
                                  max_output_tokens=128, request_sender=send)


def native(*calls):
    return {"model": "controlled", "done": True, "message": {"content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}} for name, arguments in calls]}}


def test_native_ollama_rejected_finish_check_then_same_id_finish_refreshes_real_state():
    owners, captured = Owners(Budget()), []
    owners.fixed = True
    replies = [native(("finish", {"candidate_id": "c0"})),
        native(("candidate_check", {"candidate_id": "c0"})),
        native(("finish", {"candidate_id": "c0"}))]
    result = run(native_client(replies, captured), owners)
    assert result.status == "completed" and len(captured) == 3 and not replies
    finishes = [row for row in result.events if row["event"] == "tool_dispatch_prepared" and row["role"] == "finish"]
    assert len(finishes) == 2 and finishes[0]["call_id"] == finishes[1]["call_id"]
    assert finishes[1]["mode"] == "refresh" and finishes[0]["version_identity"] != finishes[1]["version_identity"]
    third_tools = [json.loads(message["content"]) for message in captured[2]["messages"] if message["role"] == "tool"]
    assert third_tools[0]["completion_check"]["accepted"] is False and third_tools[1]["value"]["passed"] is True


def test_native_ollama_one_argument_feedback_preserves_unchanged_unexecuted_good_id():
    owners, captured = Owners(Budget()), []
    owners.fixed, owners.report = True, {"candidate_id": "c0", "passed": True, "report_id": "prior-check"}
    replies = [native(("source_read", {}), ("source_read", {"binding_id": "two"})),
        native(("source_read", {"binding_id": "one"}), ("source_read", {"binding_id": "two"})),
        native(("finish", {"candidate_id": "c0"}))]
    result = run(native_client(replies, captured), owners)
    assert result.status == "completed" and result.tool_argument_correction_used
    assert result.read_units == 2 and [args["binding_id"] for role, args in owners.effects if role == "source_read"] == ["one", "two"]
    feedback = [json.loads(message["content"]) for message in captured[1]["messages"] if message["role"] == "tool"]
    assert len(feedback) == 2 and all(row["status"] == "rejected_not_executed" for row in feedback)
    assert feedback[1]["argument_errors"] == []


def test_argument_correction_is_shared_once_and_retained_across_provider_interruption():
    owners = Owners(Budget())
    first = run(Client([response(call("source_read", {}, "bad1")),
                        AgentTurnProviderError("controlled interruption")]), owners)
    assert first.tool_argument_correction_used and first.model_budget["consumed"] == 2
    second = run(Client([response(call("source_read", {"binding_id": 3}, "bad2"))]), owners,
        initial_request=first.request, prior_call_records=first.call_records)
    assert second.tool_argument_correction_used
    assert second.termination["code"] == "TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED"
    assert owners.effects == [] and second.model_budget["consumed"] == 3


def test_host_only_execution_history_allows_new_repair_without_resending_old_messages():
    owners = Owners(Budget())
    retained = run(Client([response(call("candidate_submit", {"candidate_id": "c1", "fixed": False}, "old-submit")),
        AgentTurnProviderError("controlled pause")]), owners)
    client = Client([response(call("candidate_submit", {"candidate_id": "c2", "fixed": True}, "new-submit")),
        response(call("candidate_check", {"candidate_id": "c2"}, "new-check")),
        response(call("finish", {"candidate_id": "c2"}, "new-finish"))])
    result = run(client, owners, prior_execution_requests=(retained.request,),
        prior_call_records=retained.call_records)
    assert result.status == "completed"
    assert not client.requests[0].tool_results and not client.requests[0].historical_tool_exchanges
    assert "old-submit" not in json.dumps(asdict(client.requests[0]))
    assert "old-submit" in result.call_records and result.model_budget["consumed"] == 5
    assert any(row.get("event") == "execution_history_restored" and row["model_history_exposed"] is False
        for row in result.events)


@pytest.mark.parametrize("fault", ["repeat_effect", "changed_result", "missing_record", "conflicting_visible_history"])
def test_host_only_execution_history_cannot_reset_replay_or_binding_checks(fault):
    owners = Owners(Budget())
    previous = call("candidate_submit", {"candidate_id": "c1", "fixed": False}, "old-submit")
    retained = run(Client([response(previous), AgentTurnProviderError("controlled pause")]), owners)
    records = deepcopy(retained.call_records)
    options = {}
    if fault == "changed_result":
        records["old-submit"]["result"]["output"]["value"]["candidate_id"] = "forged"
    elif fault == "missing_record":
        records = {}
    elif fault == "conflicting_visible_history":
        options["initial_request"] = retained.request
    client = Client([response(previous)])
    result = run(client, owners, prior_execution_requests=(retained.request,), prior_call_records=records, **options)
    assert owners.effects == [("candidate_submit", previous.arguments)]
    if fault == "repeat_effect":
        assert "TASK_AGENT_REPEATED_TOOL_CALL" in result.termination["detail"]["reasons"]
    else:
        assert not client.requests
        assert result.termination["code"] in {"TASK_AGENT_PRIOR_CALL_BINDING_INVALID", "TASK_AGENT_EXECUTION_HISTORY_CONFLICT"}


@pytest.mark.parametrize("kind", ["argument", "output"])
def test_host_only_execution_history_preserves_used_correction(kind):
    owners = Owners(Budget())
    bad = (response(call("source_read", {}, "invalid")) if kind == "argument"
        else AgentTurnResponse("scripted", "controlled", "Finished without tools"))
    retained = run(Client([bad, AgentTurnProviderError("controlled pause")]), owners)
    result = run(Client([bad]), owners, prior_execution_requests=(retained.request,),
        prior_call_records=retained.call_records)
    assert result.termination["code"] == (
        "TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED" if kind == "argument" else "TASK_AGENT_FINISH_TOOL_REQUIRED")
    assert result.model_budget["consumed"] == 3 and not owners.effects


def test_host_only_execution_history_rejects_unknown_effects_before_another_send():
    from src.mod_agent.model_qualification import canonical_hash

    owners = Owners(Budget())
    previous = call("upstream", {}, "uncertain")
    tool_result = AgentToolResult(previous.call_id, previous.name, {"status": "unknown"})
    retained = AgentTurnRequest((AgentMessage("user", "Earlier task"),), tool_results=(tool_result,))
    records = {previous.call_id: {"name": previous.name, "arguments_sha256": canonical_hash(previous.arguments),
        "status": "unknown", "result": asdict(tool_result)}}
    client = Client([])
    result = run(client, owners, prior_execution_requests=(retained,), prior_call_records=records)
    assert result.unknown_effects and result.termination["code"] == "TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE"
    assert not client.requests and not owners.effects and owners.budget.consumed == 0


def test_second_fresh_restore_retains_first_context_effects():
    owners = Owners(Budget())
    previous = call("candidate_submit", {"candidate_id": "c1", "fixed": False}, "old-submit")
    first = run(Client([response(previous), AgentTurnProviderError("first pause")]), owners)
    middle = run(Client([response(call("source_read", {"binding_id": "one"}, "new-read")),
        AgentTurnProviderError("second pause")]), owners,
        prior_execution_requests=first.execution_history_requests, prior_call_records=first.call_records)
    assert len(middle.execution_history_requests) == 2 and len(middle.call_records) == 2
    final = run(Client([response(previous)]), owners,
        prior_execution_requests=middle.execution_history_requests, prior_call_records=middle.call_records)
    assert final.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert "TASK_AGENT_REPEATED_TOOL_CALL" in final.termination["detail"]["reasons"]
    assert len(owners.effects) == 2 and owners.budget.consumed == 5


@pytest.mark.parametrize("kind", ["argument", "output"])
def test_second_fresh_restore_does_not_reissue_used_correction(kind):
    owners = Owners(Budget())
    bad = (response(call("source_read", {}, "invalid")) if kind == "argument"
        else AgentTurnResponse("scripted", "controlled", "Finished without tools"))
    first = run(Client([bad, AgentTurnProviderError("first pause")]), owners)
    middle = run(Client([response(call("source_read", {"binding_id": "one"}, "new-read")),
        AgentTurnProviderError("second pause")]), owners,
        prior_execution_requests=first.execution_history_requests, prior_call_records=first.call_records)
    final = run(Client([bad]), owners,
        prior_execution_requests=middle.execution_history_requests, prior_call_records=middle.call_records)
    assert final.termination["code"] == (
        "TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED" if kind == "argument" else "TASK_AGENT_FINISH_TOOL_REQUIRED")
    assert owners.effects == [("source_read", {"binding_id": "one"})] and owners.budget.consumed == 5


def test_actual_unknown_execution_without_result_stays_unknown_during_fresh_restore():
    owners = Owners(Budget())
    owners.raise_after_effect = "candidate_check"
    first = run(Client([response(call("candidate_check", {"candidate_id": "c0"}, "uncertain"))]), owners)
    assert first.unknown_effects and "result" not in first.call_records["uncertain"]
    client = Client([])
    final = run(client, owners, prior_execution_requests=first.execution_history_requests,
        prior_call_records=first.call_records)
    assert final.unknown_effects and final.termination["code"] == "TASK_AGENT_PREVIOUS_OUTCOME_NOT_REUSABLE"
    assert final.call_records["uncertain"] == first.call_records["uncertain"]
    assert not client.requests and len(owners.effects) == 1 and owners.budget.consumed == 1


def test_finish_refresh_requires_owner_confirmation_not_stale_cached_denial():
    owners = Owners(Budget())
    actions = [replace(item, confirm_reuse=None) if item.role == "finish" else item for item in owners.actions()]
    finish = call("finish", {"candidate_id": "c0"}, "stable")
    result = run(Client([response(finish), response(finish)]), owners, actions=actions)
    assert result.status == "blocked" and len(owners.effects) == 1


@pytest.mark.parametrize("fault", ["unconfirmed", "changed_version"])
def test_read_reuse_does_not_grant_new_source_version_or_skip_owner_confirmation(fault):
    owners = Owners(Budget())
    actions = owners.actions()
    action = actions[0]
    calls = 0

    def prepare(arguments):
        nonlocal calls
        calls += 1
        prepared = action.prepare(arguments)
        return replace(prepared, version_identity="source-v2") if fault == "changed_version" and calls == 2 else prepared

    actions[0] = replace(action, prepare=prepare,
        confirm_reuse=None if fault == "unconfirmed" else action.confirm_reuse)
    repeated = call("source_read", {"binding_id": "one"}, "same")
    result = run(Client([response(repeated), response(repeated)]), owners, actions=actions)
    assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
    assert len(owners.effects) == 1 and result.read_units == 1


def test_source_feedback_cannot_advertise_an_unavailable_next_model_turn():
    owners = Owners(Budget(maximum=1))
    result = run(Client([response(call("source_read", {}, "bad"))]), owners)
    assert result.termination["code"] == "TASK_AGENT_MODEL_BUDGET_EXHAUSTED"
    assert not result.tool_argument_correction_used and owners.budget.bytes == 0
    assert not result.request.tool_results and owners.effects == []


@pytest.mark.parametrize("initial", [True, False])
def test_trusted_stop_check_retains_known_failure_without_next_reservation(initial):
    owners = Owners(Budget())
    client = Client([] if initial else [response(call("candidate_check", {"candidate_id": "c0"}, "check"))])

    def stop_check():
        if initial or owners.report is not None:
            return {"code": "PATCH_BEHAVIOR_NO_PROGRESS", "same_state_observations": 3}
        return None

    result = run(client, owners, stop_check=stop_check)
    assert result.status == "blocked" and not result.unknown_effects
    assert result.termination == {"code": "PATCH_BEHAVIOR_NO_PROGRESS", "same_state_observations": 3}
    assert result.model_budget["consumed"] == len(client.requests) == (0 if initial else 1)
    if not initial:
        assert result.call_records["check"]["status"] == "completed"
        assert result.unadmitted_tool_results[0].output["value"]["passed"] is False
        assert result.observation_bytes == 0


def test_stop_check_error_is_pre_send_admission_failure_not_unknown_execution():
    owners, client = Owners(Budget()), Client([])

    def stop_check():
        raise ValueError("controlled invalid checkpoint")

    result = run(client, owners, stop_check=stop_check)
    assert result.termination["code"] == "TASK_AGENT_STOP_CHECK_FAILED"
    assert not result.unknown_effects and owners.budget.consumed == 0 and not client.requests


def test_completed_known_stop_precedes_unneeded_feedback_byte_budget():
    owners, client = Owners(Budget(max_bytes=0)), Client([
        response(call("candidate_check", {"candidate_id": "c0"}, "last-failed-check"))])
    result = run(client, owners, stop_check=lambda: {"code": "PATCH_BEHAVIOR_NO_PROGRESS",
        "same_state_observations": 3} if owners.report is not None else None)
    assert result.termination["code"] == "PATCH_BEHAVIOR_NO_PROGRESS" and not result.unknown_effects
    assert result.model_budget["consumed"] == 1 and result.observation_bytes == owners.budget.bytes == 0
    assert result.call_records["last-failed-check"]["status"] == "completed"
    assert result.unadmitted_tool_results[0].output["value"]["passed"] is False
