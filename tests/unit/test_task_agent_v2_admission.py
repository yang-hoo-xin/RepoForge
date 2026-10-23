"""Pre-dispatch rejection and linked accounting through the actual task loop.

为什么要有它：只读参数错误不能伪造未知副作用，真实调用和资源记录必须互相关联。
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.libs.llm.agent_turn import AgentMessage, AgentToolCall, AgentTurnRequest, AgentTurnResponse
from src.mod_agent.task_agent_loop import TaskActionInputError, TaskCandidateRevisionInputError
from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError
from src.mod_agent.task_agent_v2_tools import TaskAgentSessionV2
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


@pytest.fixture
def session(state, tmp_path):
    owner = object.__new__(TaskAgentExecutionV2)
    owner._v2_state = state
    owner.output_root = tmp_path / "execution"
    owner.output_root.mkdir()
    owner.profile = SimpleNamespace(model="local")
    owner.policy = SimpleNamespace(max_observation_bytes=1000000, max_tool_rounds=1000)
    owner.workspace_root = tmp_path
    owner.deadline = None
    owner._reads = owner._observation_bytes = 0
    consumed = [0]

    def reserve(_role):
        consumed[0] += 1

    owner.reserve_model_call = reserve
    owner.budget_snapshot = lambda: {"policy": "task_total_only", "maximum": 20,
        "consumed": consumed[0], "remaining": 20 - consumed[0],
        "by_stage": {"initial_patch": {"consumed": consumed[0]}},
        "count_basis": "reserved_provider_invocations_not_transport_proof"}
    return TaskAgentSessionV2(state=state, evaluator=None, owner=owner, feature_id="fixture")


def action(session, name):
    return next(item for item in session.actions() if item.definition.name == name)


def test_current_decision_receives_remaining_window_and_cumulative_resources(session):
    import time

    session.owner.deadline = time.monotonic() + 30
    prepared = session.owner.prepare_decision_request(session.initial_request(), session)
    current = prepared.messages[-1].content['current_task_state']
    assert 0 < current['execution_window']['remaining_wall_seconds'] <= 30
    assert current['cumulative_resources'] == session.state.head['resources']
    assert current['model_call_budget'] == session.owner.budget_snapshot()


def response(name, arguments, identifier="call-1"):
    call = AgentToolCall(identifier, name, arguments)
    return AgentTurnResponse("ollama_chat_local", "local", "", (call,), provider_state={"calls": [identifier]})


def test_missing_read_is_correctable_before_dispatch_and_prepare_writes_nothing(session):
    state = session.state
    before = set(state.root.rglob("*"))
    with pytest.raises(TaskActionInputError) as error:
        action(session, "source_read").prepare({"parent_revision_ref": "baseline", "path": "missing.cs"})
    assert "source_not_in_snapshot" in error.value.argument_errors
    action(session, "source_read").prepare({"parent_revision_ref": "baseline", "path": "A.cs"})
    action(session, "source_search").prepare({"parent_revision_ref": "baseline", "query": "a"})
    assert set(state.root.rglob("*")) == before


@pytest.mark.parametrize("args,code", [
    ({"parent_revision_ref": "baseline", "path": "A.cs", "root_id": "other"}, "read_root_denied"),
    ({"parent_revision_ref": "baseline", "path": "../A.cs"}, "invalid_path"),
    ({"parent_revision_ref": "baseline", "path": ".env"}, "read_path_denied"),
])
def test_read_authority_rejection_is_hard_and_known(session, args, code):
    with pytest.raises(TaskAgentV2StateError, match=code):
        action(session, "source_read").prepare(args)
    replies = [response("source_read", args)]
    client = SimpleNamespace(complete_turn=lambda _request: replies.pop(0))
    result = session.run(client, lambda _event: None)
    assert result.unknown_effects is False
    assert not any(event["event"] == "tool_dispatch_started" for event in result.events)


def test_exact_preimage_and_valid_historical_read_errors_are_correctable(session):
    state = session.state
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    edit = {"binding_id": binding["binding_id"], "source_view_ref": view["source_view_ref"],
        "before_text": "missing", "after_text": "2", "rationale": "fix"}
    submit = action(session, "candidate_submit")
    with pytest.raises(TaskCandidateRevisionInputError) as error:
        submit.prepare({"parent_revision_ref": state.current_revision_ref, "edits": [edit]})
    assert "before_text_not_unique" in error.value.argument_errors
    edit.update(before_text="1", after_text="1")
    with pytest.raises(TaskCandidateRevisionInputError) as error:
        submit.prepare({"parent_revision_ref": state.current_revision_ref, "edits": [edit]})
    assert "candidate_has_no_changes" in error.value.argument_errors
    edit["after_text"] = "2"
    state.submit("baseline", [edit])
    with pytest.raises(TaskAgentV2StateError, match="stale_parent_revision"):
        submit.prepare({"parent_revision_ref": "baseline", "edits": [edit]})
    with pytest.raises(TaskCandidateRevisionInputError) as error:
        submit.prepare({"parent_revision_ref": state.current_revision_ref, "edits": [edit]})
    assert "historical revision" in str(error.value.argument_errors)


@pytest.mark.parametrize("change", [{"provider": "different"}, {"stop_reason": "refusal"}, {"stop_reason": "length"}])
def test_unadmitted_provider_or_stopped_response_cannot_dispatch(session, change):
    before = len(list(session.owner.output_root.iterdir()))
    item = replace(response("source_read", {"parent_revision_ref": "baseline", "path": "A.cs"}), **change)
    with pytest.raises(ValueError, match="RESPONSE_IDENTITY_CHANGED"):
        session.response_admission(item)
    assert session.actual_calls == []
    assert len(list(session.owner.output_root.iterdir())) == before


def test_accounting_receipts_link_reservation_to_tools_and_sync_once(session):
    import json

    owner, state = session.owner, session.state
    owner.reserve_v2_call()
    reservation = owner._v2_reservation.copy()
    owner._v2_reservation.update(call_ids=["actual-call"], request_ref={"path": "request"}, response_ref={"path": "response"})
    owner.charge_v2_tools(reads=1, observation_bytes=17)
    intent = json.loads(next(owner.output_root.glob("resource-intent-*.json")).read_text())
    reserved = json.loads(next(owner.output_root.glob("resource-reserved-*.json")).read_text())
    tools = json.loads(next(owner.output_root.glob("resource-tools-*.json")).read_text())
    assert intent["reservation_id"] == reserved["reservation_id"] == reservation["reservation_id"]
    assert reserved["intent_ref"] == reservation["intent_ref"]
    assert tools["reservation"]["reserved_ref"] == reservation["reserved_ref"]
    assert tools["reservation"]["call_ids"] == ["actual-call"]
    state.record_resources(tools["usage"], operation_id=tools["charge_id"])
    assert state.head["resources"] == {"model_calls": 1, "reads": 1, "observation_bytes": 17}


def test_current_state_and_parent_schema_follow_candidate_without_rewriting_baseline(session):
    from tests.unit.test_task_agent_v2_state import edit

    request = AgentTurnRequest((AgentMessage("system", "Host rules"), AgentMessage("user", {"original": "task"})))
    first = session.owner.prepare_decision_request(request, session)
    baseline = session.state.current_revision_ref
    assert first.messages[-1].content["current_task_state"]["current_revision_ref"] == baseline
    changed = edit(session.state, "A.cs", "1", "2")
    current = session.owner.prepare_decision_request(request, session)
    facts = current.messages[-1].content["current_task_state"]
    assert facts["current_revision_ref"] == changed["candidate_ref"]
    assert facts["baseline_revision_ref"] == baseline
    assert facts["valid_bindings"][0]["path"] == "A.cs"
    assert len(current.messages) == len(first.messages) == len(request.messages) + 1
    assert action(session, "candidate_submit").definition.parameters["properties"]["parent_revision_ref"]["enum"] == [changed["candidate_ref"]]


def test_identical_batched_reads_keep_distinct_actual_call_receipts(session):
    import json

    arguments = {"parent_revision_ref": "baseline", "path": "A.cs"}
    calls = (AgentToolCall("read-1", "source_read", arguments), AgentToolCall("read-2", "source_read", arguments))
    replies = [AgentTurnResponse("ollama_chat_local", "local", "", calls, provider_state={"calls": ["read-1", "read-2"]}),
        AgentTurnResponse("ollama_chat_local", "local", "", (), stop_reason="refusal", provider_state={})]
    result = session.run(SimpleNamespace(complete_turn=lambda _request: replies.pop(0)), session.observe_dispatch_event)
    assert result.unknown_effects is False
    assert all(result.call_records[call_id]["status"] == "completed" for call_id in ("read-1", "read-2"))
    receipts = [json.loads(path.read_text()) for path in session.owner.output_root.glob("tool-receipt-*.json")]
    assert {receipt["origin"]["call_id"] for receipt in receipts} == {"read-1", "read-2"}
    assert len({row["execution_receipt_ref"]["path"] for row in session.actual_calls}) == 2


def test_v2_correction_uses_current_schema_without_legacy_patch_format(session):
    state = session.state
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    arguments = {"parent_revision_ref": state.current_revision_ref, "edits": [{"binding_id": binding["binding_id"],
        "source_view_ref": view["source_view_ref"], "after_text": "changed", "rationale": "controlled missing before_text"}]}
    replies = [response("candidate_submit", arguments),
        AgentTurnResponse("ollama_chat_local", "local", "", (), stop_reason="refusal", provider_state={})]
    requests = []

    def complete(request):
        requests.append(request)
        return replies.pop(0)

    result = session.run(SimpleNamespace(complete_turn=complete), session.observe_dispatch_event)
    assert result.tool_argument_correction_used and not result.unknown_effects
    correction = requests[1].tool_results[0].output
    assert "candidate_submit schema" in correction["message"]
    assert "patch_json" not in correction["message"] and "context_id" not in correction["message"]
    assert any("before_text" in error for error in correction["argument_errors"])


def test_missing_edit_explanation_is_recorded_without_blocking_authorized_code_change(session):
    state = session.state
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    arguments = {"parent_revision_ref": state.current_revision_ref, "edits": [{"binding_id": binding["binding_id"],
        "source_view_ref": view["source_view_ref"], "before_text": "a = 1;", "after_text": "a = 3;"}]}
    replies = [response("candidate_submit", arguments, "unexplained-edit"),
        AgentTurnResponse("ollama_chat_local", "local", "", (), stop_reason="refusal", provider_state={})]
    requests = []

    def complete(request):
        requests.append(request)
        return replies.pop(0)

    result = session.run(SimpleNamespace(complete_turn=complete), session.observe_dispatch_event)
    assert not result.tool_argument_correction_used and not result.unknown_effects
    assert result.call_records["unexplained-edit"]["status"] == "completed"
    output = requests[1].tool_results[-1].output["value"]
    assert output["edit_explanations"] == [{"binding_id": binding["binding_id"], "status": "not_expressed", "rationale": None}]
    assert state.materialize(output["candidate_ref"])["A.cs"] == "a = 3;\n"
    assert session.actual_calls[-1]["arguments"] == arguments
    assert "rationale" not in session.actual_calls[-1]["arguments"]["edits"][0]
    with pytest.raises(TaskAgentV2StateError, match="stale_parent_revision"):
        action(session, "candidate_submit").prepare(arguments)


def test_missing_experiment_catalog_returns_known_rejection_and_keeps_same_session(session, monkeypatch):
    from src.mod_agent import task_agent_v2_experiment_tools as adapters
    from src.mod_agent.task_agent_v2_experiments import TaskAgentExperiments

    def unexpected_runner():
        pytest.fail("A nonexistent catalog cannot dispatch tests")

    experiments = TaskAgentExperiments(state=session.state, test_runner=SimpleNamespace(identity=unexpected_runner))
    monkeypatch.setattr(adapters, "_experiments", lambda _session: experiments)
    baseline = session.state.current_revision_ref
    arguments = {"kind": "registered_test", "source_revision_ref": baseline,
        "catalog_ref": "v2/test-catalog/" + "7" * 64 + ".json", "test_ids": ["invented"],
        "hypothesis": "Check public behavior", "expected_observation": "Actual result"}
    replies = [response("experiment_run", arguments, "missing-catalog"),
        response("test_catalog", {}, "discover-options"),
        AgentTurnResponse("ollama_chat_local", "local", "", (), stop_reason="refusal", provider_state={})]
    requests = []

    def complete(request):
        requests.append(request)
        return replies.pop(0)

    result = session.run(SimpleNamespace(complete_turn=complete), session.observe_dispatch_event)
    rejection = requests[1].tool_results[-1].output["value"]
    assert rejection["status"] == "rejected" and rejection["executed"] is False
    assert "CATALOG_REFERENCE_NOT_FOUND" in rejection["reason"]
    assert requests[2].tool_results[-1].output["value"]["discovery_available"] is True
    assert not result.unknown_effects
    assert result.call_records["missing-catalog"]["status"] == "completed"
    assert session.state.current_revision_ref == baseline
    assert session.state.head["resources"]["model_calls"] == 3
    assert not any(key.startswith("experiment:") for key in session.state.head["operations"])
