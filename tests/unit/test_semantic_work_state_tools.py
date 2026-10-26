"""Advisory state travels with existing actions without becoming authority.

为什么要有它：模型说明和实际调用原文均需保留，合法动作仍按原权限、版本及前像规则执行。
"""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.libs.llm.agent_turn import AgentTurnResponse
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_loop import TaskActionInputError
from src.mod_agent.task_agent_v2_experiments import TaskAgentExperiments
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError
from tests.unit.test_task_agent_v2_admission import action, response
from tests.unit.test_task_agent_v2_admission import session as session
from tests.unit.test_task_agent_v2_state import edit
from tests.unit.test_task_agent_v2_state import state as state

SEMANTIC_GUIDANCE = (
    "working_state is optional. When useful, record concise observations, hypotheses, "
    "open questions and the purpose/expected observation of that action using the semantic contract "
    "and actually seen evidence provided in current_task_state. Omit it when no update is needed; "
    "in a read-only batch include at most one nonempty update. These are your explicit advisory statements, "
    "not hidden reasoning, established truth, new permission or completion. "
    "Do not invent observations or cite unseen evidence."
)


def stopped_response():
    return AgentTurnResponse("ollama_chat_local", "local", "", (), stop_reason="refusal", provider_state={})


def test_enabled_metadata_uses_existing_tools_and_disabled_input_remains_legacy(session):
    session.owner.semantic_state_enabled = False
    legacy_request = session.initial_request()
    legacy = {row.definition.name: row for row in session.actions()}
    assert all("working_state" not in row.definition.parameters["properties"] for row in legacy.values())
    session.owner.semantic_state_enabled = True
    enabled_request = session.initial_request()
    enabled = {row.definition.name: row for row in session.actions()}
    assert enabled.keys() == legacy.keys()
    # 为什么要有它：官方Ollama v0.40.9 Qwen3CoderRenderer只采用首条非空system；HTTP中的后置system不证明模型收到它。
    assert enabled_request.messages[0].role == "system"
    assert enabled_request.messages[0].content == legacy_request.messages[0].content + "\n\n" + SEMANTIC_GUIDANCE
    assert enabled_request.messages[1:] == legacy_request.messages[1:]
    assert sum(message.role == "system" for message in enabled_request.messages) == 1
    for name, row in enabled.items():
        assert row.role == legacy[name].role and row.read_only == legacy[name].read_only
        assert row.definition.parameters["properties"]["working_state"]["type"] == "object"
        assert "working_state" not in row.definition.parameters["required"]
        assert "observations" not in row.definition.parameters["properties"]["working_state"]
    session.owner.semantic_state_enabled = False
    assert session.initial_request() == legacy_request
    with pytest.raises(TaskActionInputError):
        action(session, "source_read").prepare({"parent_revision_ref": "baseline", "path": "A.cs", "working_state": {}})


def test_invalid_state_does_not_block_read_or_rewrite_actual_call_identity(session):
    arguments = {"parent_revision_ref": "baseline", "path": "A.cs", "working_state": "not an object"}
    original = deepcopy(arguments)
    before = set(session.state.root.rglob("*"))
    prepared = action(session, "source_read").prepare(arguments)
    assert prepared.value == {"parent_revision_ref": "baseline", "path": "A.cs"}
    assert arguments == original and set(session.state.root.rglob("*")) == before
    replies = [response("source_read", arguments, "actual-read"), stopped_response()]
    result = session.run(SimpleNamespace(complete_turn=lambda _request: replies.pop(0)), session.observe_dispatch_event)
    record = result.call_records["actual-read"]
    assert record["status"] == "completed" and record["result"]["output"]["value"]["text"] == "a = 1;\n"
    assert record["arguments_sha256"] == canonical_hash(original)
    origin = session.actual_calls[0]
    assert origin["arguments"] == original
    assert json.loads(Path(origin["arguments_ref"]["path"]).read_text(encoding="utf-8")) == original
    session._dispatch_call_id = "actual-read"
    with pytest.raises(ValueError, match="REAL_MODEL_CALL_REQUIRED"):
        session._origin("source_read", {"parent_revision_ref": "baseline", "path": "B.cs"})


def test_invalid_state_does_not_block_candidate_or_grant_out_of_scope_edit(session):
    state = session.state
    view = state.read_source("baseline", "A.cs")
    arguments = {"parent_revision_ref": state.current_revision_ref, "working_state": {"observations": "invalid"},
        "edits": [{"path": "A.cs", "source_view_ref": view["source_view_ref"],
                   "before_text": "a = 1;", "after_text": "a = 3;"}]}
    replies = [response("candidate_submit", arguments, "actual-edit"), stopped_response()]
    result = session.run(SimpleNamespace(complete_turn=lambda _request: replies.pop(0)), session.observe_dispatch_event)
    assert result.call_records["actual-edit"]["status"] == "completed"
    candidate = state.validate_candidate(state.current_revision_ref)
    origin = session.actual_calls[0]
    assert state.materialize(state.current_revision_ref)["A.cs"] == "a = 3;\n"
    assert candidate["model_origin"] == {key: origin[key] for key in ("call_id", "response_ref", "arguments_ref")}
    assert origin["arguments"] == arguments
    private = state.read_source(state.current_revision_ref, "private.cs")
    with pytest.raises(TaskAgentV2StateError, match="edit_path_denied"):
        action(session, "candidate_submit").prepare({"parent_revision_ref": state.current_revision_ref,
            "working_state": {"permission": "allowed"}, "edits": [{"path": "private.cs",
                "source_view_ref": private["source_view_ref"], "before_text": "private", "after_text": "changed"}]})


def test_finish_keeps_raw_state_argument_and_requires_original_passing_check(session):
    state = session.state
    candidate = edit(state, "A.cs", "1", "2")
    identity = {"controlled": "verification"}
    session.evaluator = SimpleNamespace(verification_identity=lambda: identity)
    session.current_check = state.save("check", {"status": "passed", "candidate_ref": candidate["candidate_ref"],
        "candidate_source_id": candidate["candidate_source_id"], "report": {"status": "passed", "verification_identity": identity}})
    arguments = {"candidate_ref": candidate["candidate_ref"], "check_ref": session.current_check, "working_state": []}
    replies = [response("finish", arguments, "actual-finish")]
    result = session.run(SimpleNamespace(complete_turn=lambda _request: replies.pop(0)), session.observe_dispatch_event)
    assert result.completion["accepted"] is True and result.completion["whole_task_delivered"] is False
    assert result.call_records["actual-finish"]["arguments_sha256"] == canonical_hash(arguments)
    assert session.actual_calls[0]["arguments"] == arguments
    assert result.model_turns == 1


def test_probe_and_unavailable_experiment_keep_raw_origin_without_executing_sdk(session):
    state = session.state
    session._experiments = TaskAgentExperiments(state=state)
    baseline = state.materialize(state.current_revision_ref)
    raw = []

    def complete(request):
        if not raw:
            name, args = "probe_submit", {"source_revision_ref": state.current_revision_ref,
                "files": [{"relative_path": "probe/Main.cs", "content": "class Probe {}"}],
                "entrypoint_id": "csharp_console", "rationale": "Controlled probe artifact",
                "expected_observation": "No code executes while submitting", "working_state": "invalid metadata"}
        elif len(raw) == 1:
            probe = request.tool_results[-1].output["value"]
            name, args = "experiment_run", {"source_revision_ref": state.current_revision_ref,
                "kind": "temporary_probe", "probe_artifact_ref": probe["probe_artifact_ref"],
                "hypothesis": "The unavailable runner should remain unavailable",
                "expected_observation": "No SDK execution", "working_state": {"permission": "run anyway"}}
        else:
            return stopped_response()
        raw.append(deepcopy(args))
        return response(name, args, "actual-probe-" + str(len(raw)))

    result = session.run(SimpleNamespace(complete_turn=complete), session.observe_dispatch_event)
    submitted = result.call_records["actual-probe-1"]["result"]["output"]["value"]
    experiment = result.call_records["actual-probe-2"]["result"]["output"]["value"]
    probe = state.load(submitted["probe_artifact_ref"])
    observed = state.load(experiment["result_ref"])
    for index, record in enumerate((probe, observed["request"])):
        origin = session.actual_calls[index]
        assert record["model_origin"] == {key: origin[key] for key in ("call_id", "response_ref", "arguments_ref")}
        assert origin["arguments"] == raw[index]
    assert experiment["status"] == "capability_unavailable" and experiment["executed"] is False
    assert state.materialize(state.current_revision_ref) == baseline
