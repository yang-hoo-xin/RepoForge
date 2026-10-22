"""Binding prerequisites are explicit without weakening authority or version checks."""
from types import SimpleNamespace

import pytest

from src.mod_agent.task_agent_loop import TaskCandidateRevisionInputError
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError
from tests.unit.test_task_agent_v2_admission import action, response
from tests.unit.test_task_agent_v2_admission import session as session
from tests.unit.test_task_agent_v2_state import state as state


def edits(session, path="A.cs", binding_id="snapshot-is-not-binding"):
    view = session.state.read_source(session.state.current_revision_ref, path)
    return {"parent_revision_ref": session.state.current_revision_ref, "edits": [{
        "binding_id": binding_id, "source_view_ref": view["source_view_ref"],
        "before_text": view["text"], "after_text": view["text"] + "\n", "rationale": "controlled correction"}]}


def test_empty_registry_feedback_does_not_execute_or_create_authority(session):
    arguments = edits(session)
    before = set(session.state.root.rglob("*"))
    with pytest.raises(TaskCandidateRevisionInputError) as error:
        action(session, "candidate_submit").prepare(arguments)
    assert "Call binding_add" in error.value.argument_errors[0]
    assert "valid_binding_ids=[]" in error.value.argument_errors
    assert set(session.state.root.rglob("*")) == before
    assert session.state.load(session.state.registry_ref)["bindings"] == {}


def test_binding_schema_refreshes_after_registration(session):
    definition = action(session, "candidate_submit").definition
    assert "registry is empty" in definition.description
    arguments = edits(session)
    view = arguments["edits"][0]["source_view_ref"]
    binding = session.state.add_binding(session.state.registry_ref, view, "A.cs")
    updated = action(session, "candidate_submit").definition
    assert updated.parameters["properties"]["edits"]["items"]["properties"]["binding_id"]["enum"] == [binding["binding_id"]]
    assert binding["binding_id"] in updated.description


def test_unknown_binding_cannot_hide_out_of_scope_or_stale_source(session):
    with pytest.raises(TaskAgentV2StateError, match="edit_path_denied"):
        action(session, "candidate_submit").prepare(edits(session, "private.cs"))
    old = edits(session)
    view = old["edits"][0]["source_view_ref"]
    binding = session.state.add_binding(session.state.registry_ref, view, "A.cs")
    session.state.submit(session.state.current_revision_ref, [{**old["edits"][0], "binding_id": binding["binding_id"]}])
    with pytest.raises(TaskAgentV2StateError, match="stale_parent_revision"):
        action(session, "candidate_submit").prepare(old)
    old["parent_revision_ref"] = session.state.current_revision_ref
    with pytest.raises(TaskAgentV2StateError, match="stale_or_invalid_source_view"):
        action(session, "candidate_submit").prepare(old)


def test_unknown_binding_correction_is_bounded_and_never_dispatches_edit(session):
    arguments = edits(session)
    requests = []
    def complete(request):
        requests.append(request)
        return response("candidate_submit", arguments, "call-" + str(len(requests)))
    result = session.run(SimpleNamespace(complete_turn=complete), lambda event: None)
    assert len(requests) == 2
    assert result.tool_argument_correction_used
    assert not any(event["event"] == "tool_dispatch_started" for event in result.events)
    assert session.state.current_revision_ref == session.state.contract["baseline_manifest_ref"]


def test_each_decision_displays_current_parent_without_rewriting_model_arguments(session):
    from src.libs.llm.agent_turn import AgentMessage, AgentTurnRequest
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2

    original = edits(session)
    baseline = session.state.current_revision_ref
    binding = session.state.add_binding(session.state.registry_ref, original["edits"][0]["source_view_ref"], "A.cs")
    submitted = session.state.submit(baseline, [{**original["edits"][0], "binding_id": binding["binding_id"]}])
    request = AgentTurnRequest(messages=(AgentMessage("user", {"initial_parent": baseline}),))
    prepared = TaskAgentExecutionV2.prepare_decision_request(session.owner, request, session)
    current = prepared.messages[-1].content["current_task_state"]
    assert current["current_revision_ref"] == submitted["candidate_ref"] != baseline
    assert current["baseline_revision_ref"] == baseline
    assert current["valid_bindings"] == [{"binding_id": binding["binding_id"], "path": "A.cs"}]
    assert prepared.messages[0] == request.messages[0]
    definition = action(session, "candidate_submit").definition
    assert definition.parameters["properties"]["parent_revision_ref"]["enum"] == [submitted["candidate_ref"]]
    with pytest.raises(TaskAgentV2StateError, match="stale_parent_revision"):
        action(session, "candidate_submit").prepare(original)
