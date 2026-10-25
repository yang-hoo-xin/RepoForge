"""Current-boundary strategy conditions do not invent missing or recovered facts."""
import json
from copy import deepcopy
from types import SimpleNamespace

from src.libs.llm.agent_turn import AgentTurnRequest
from src.mod_agent.task_agent_v2_experience import TaskAgentV2Experience
from src.mod_agent.task_agent_v2_state import save_v2_artifact
from src.mod_agent.task_agent_v2_working_state import update_working_state
from tests.unit.test_task_agent_v2_working_state import observed


def session_fixture(tmp_path):
    documents = {}
    state = SimpleNamespace(contract_ref="task", current_revision_ref="baseline", registry_ref="bindings",
        contract={"baseline_manifest_ref": "baseline"}, store_root=tmp_path,
        validate_candidate=lambda ref: {"candidate_source_id": "source:" + ref}, load=documents.__getitem__)
    session = SimpleNamespace(state=state, owner=SimpleNamespace(), current_check=None, _check_inflight=False,
        public_checks=[], actual_calls=[])
    return session, documents


def boundary(session):
    return {"task_contract_ref": session.state.contract_ref, "current_revision_ref": session.state.current_revision_ref,
        "binding_registry_ref": session.state.registry_ref, "current_check_ref": session.current_check}


def source_state(session, working):
    reference = save_v2_artifact(session.state.store_root, "working-state", working)
    return {"origin": "execution_host", **boundary(session), "working_state_ref": reference}


def advance(session, previous, history):
    return update_working_state(previous, AgentTurnRequest(messages=(), historical_tool_exchanges=tuple(history)),
        task_contract_ref=session.state.contract_ref, current_revision_ref=session.state.current_revision_ref,
        registry_ref=session.state.registry_ref, current_check_ref=session.current_check)


def test_unknown_stagnation_is_omitted_and_same_boundary_replay_does_not_trigger(tmp_path):
    session, _ = session_fixture(tmp_path)
    facts = TaskAgentV2Experience.context(session)
    assert facts["has_candidate"] is False and facts["task_phase"] == "investigation"
    assert "stagnation_triggered" not in facts and "consecutive_repeated_returns" not in facts
    history = [observed("one"), observed("two")]
    working = advance(session, None, history)
    restored = advance(session, json.loads(json.dumps(working)), history)
    assert restored == working
    facts, sources = TaskAgentV2Experience.context_with_sources(session, source_state(session, restored))
    assert facts["stagnation_triggered"] is False and facts["consecutive_repeated_returns"] == 1
    assert sources["working_state_ref"].startswith("v2/working-state/")
    corrupted = deepcopy(restored)
    corrupted["stagnation"]["triggered"] = True
    assert "stagnation_triggered" not in TaskAgentV2Experience.context(session, source_state(session, corrupted))


def test_candidate_change_cannot_reuse_old_boundary_trigger(tmp_path):
    session, _ = session_fixture(tmp_path)
    history = [observed(str(index)) for index in range(4)]
    working = advance(session, None, history)
    old = source_state(session, working)
    assert TaskAgentV2Experience.context(session, old)["stagnation_triggered"] is True
    session.state.current_revision_ref = "candidate-new"
    stale = TaskAgentV2Experience.context(session, old)
    assert stale["has_candidate"] is True and stale["task_phase"] == "candidate_unchecked"
    assert "stagnation_triggered" not in stale
    renewed = advance(session, working, history)
    assert TaskAgentV2Experience.context(session, source_state(session, renewed))["stagnation_triggered"] is False


def test_public_phase_requires_check_bound_to_exact_current_candidate(tmp_path):
    session, documents = session_fixture(tmp_path)
    session.state.current_revision_ref = "candidate-new"
    session.current_check = "retained-check"
    documents["retained-check"] = {"schema": "task-agent-check.v2", "task_contract_ref": "task",
        "candidate_ref": "candidate-old", "candidate_source_id": "source:candidate-old", "status": "behavior_failed"}
    assert "task_phase" not in TaskAgentV2Experience.context(session)
    documents["retained-check"].update(candidate_ref="candidate-new")
    assert "task_phase" not in TaskAgentV2Experience.context(session)
    documents["retained-check"].update(candidate_source_id="source:candidate-new")
    facts, sources = TaskAgentV2Experience.context_with_sources(session)
    assert facts["task_phase"] == "public_check_failed" and sources["current_check_ref"] == "retained-check"
    documents["retained-check"].update(status="passed")
    assert TaskAgentV2Experience.context(session)["task_phase"] == "public_check_passed"
