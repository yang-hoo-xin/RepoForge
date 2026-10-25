"""Explicit learning purpose is separate from selected task facts and model conclusions."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.libs.llm.agent_turn import AgentTurnResponse, AgentUsage
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.planning.learning_request import task_context_selections
from src.mod_agent.planning.strategy_experience import StrategyExperience


def question(tmp_path, name="question", text="Which observations justify changing the investigation decision?"):
    return _save(tmp_path / (name + ".json"), {"schema_version": "strategy_learning_request_v1",
        "visibility": "public", "data_use": "strategy_learning", "question": text,
        "origin": {"kind": "outer_technical_assistance", "description": "Controlled caller question; not a reference strategy."}})


def inputs(tmp_path, *, task_context=True, run="first"):
    facts = _save(tmp_path / (run + "-facts.json"), {"visible": {"status": "timed_out"}, "not_selected": "PRIVATE_NOT_SENT"})
    contract = _save(tmp_path / (run + "-contract.json"), {"schema": "task-agent-contract.v2", "task_id": run + "-request",
        "task_requirements": {"original_request": "Preserve signatures while correcting the original behavior.",
            "acceptance_criteria": ["Public behavior passes."], "non_goals": ["Do not change public APIs."]}})
    contract_path = tmp_path / run / "v2" / "contract" / (contract["sha256"] + ".json")
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_bytes(Path(contract["path"]).read_bytes())
    contract = {**contract, "path": str(contract_path.resolve())}
    references = [{"kind": "json_file", "evidence_id": "facts", **facts, "visibility": "public", "allowed_uses": ["strategy_learning"]}]
    selectors = []
    if task_context:
        references.append({"kind": "json_file", "evidence_id": "contract", **contract,
            "visibility": "public", "allowed_uses": ["strategy_learning"]})
        selectors = task_context_selections({"task_requirements": {key: None for key in (
            "original_request", "acceptance_criteria", "non_goals")}}, "contract")
    episode = _save(tmp_path / (run + "-episode.json"), {"schema_version": "public_strategy_episode_v1",
        "visibility": "public", "data_use": "strategy_learning", "information_gain": ["public_process_observation"],
        "run_id": run, "request_id": run + "-request", "task_contract_ref": "v2/contract/" + contract["sha256"] + ".json", "applicability": {"has_candidate": True},
        "evidence_refs": references, "selections": [{"evidence_id": "facts", "locator": ["visible"]}],
        "task_context_selections": selectors})
    admission = _save(tmp_path / (run + "-admission.json"), {"status": "admitted", "purpose": "strategy_learning",
        "provider": "controlled", "model": "fixed"})
    return episode, admission


def client_with(make_output):
    requests = []

    def complete(request):
        requests.append(request)
        output = make_output(request.messages[-1].content)
        return AgentTurnResponse("controlled", "fixed", json.dumps(output), stop_reason="stop",
            usage=AgentUsage(input_tokens=8, output_tokens=5))

    return SimpleNamespace(provider="controlled", model="fixed", complete_turn=complete, requests=requests)


def test_explicit_question_source_task_and_requested_strategy_scope_remain_distinct(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "memory"))
    episode, admission = inputs(tmp_path)
    learning = question(tmp_path)
    frozen = memory.freeze_episode_set([episode], applicability={"has_candidate": False}, learning_request_ref=learning)

    def negative(sent):
        row = sent["episodes"][0]
        assert sent["learning_question"]["reference"] == learning
        assert sent["requested_strategy_scope"] == {"has_candidate": False}
        assert row["source_applicability"] == {"has_candidate": True}
        assert row["source_task_context"]["fields"]["original_request"].startswith("Preserve signatures")
        assert row["source_role"] == "context" and "PRIVATE_NOT_SENT" not in str(sent)
        source = row["source_task_context"]["field_sources"]["original_request"]
        return {"status": "insufficient_information", "reason": "An observed task is not a transfer comparison.",
            "source_assessments": [{"episode_id": row["episode_id"], "relation": "insufficient",
                "reason": "Only the original task and outcome are observed.", "evidence": [source]}],
            "evidence_gaps": [{"status": "not_observed", "description": "An independent transfer result is not observed."}]}

    client, receipts = client_with(negative), []
    result = memory.propose_from_episodes(client, frozen, admission_ref=admission, before_call=lambda _ref: True, after_call=receipts.append)
    assert result["status"] == "insufficient_information" and result["strategy_ref"] is None
    assert result["assessment_kind"] == "model_interpretation_not_verified_fact"
    assert result["evidence_gaps"][0]["status"] == "not_observed"
    assert receipts[0]["usage"]["total_tokens"] == 13
    assert memory.read(result["request_ref"])["learning_request_ref"] == learning
    repeated = memory.propose_from_episodes(client, frozen, admission_ref=admission,
        before_call=lambda _ref: pytest.fail("No repeated reservation"), after_call=lambda _ref: pytest.fail("No repeated charge"))
    assert repeated["extraction_ref"] == result["extraction_ref"] and len(client.requests) == 1


def test_legacy_source_missing_task_context_is_unknown_and_question_change_is_new_attempt(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "memory"))
    episode, admission = inputs(tmp_path, task_context=False)
    first, second = question(tmp_path), question(tmp_path, "second-question", "Which evidence is missing for a different learning question?")

    def negative(sent):
        assert sent["episodes"][0]["source_task_context"]["availability"] == "not_selected"
        return {"status": "insufficient_information", "reason": "Source task context was not selected.",
            "evidence_gaps": [{"status": "unknown", "description": "The original non-goals are unknown.", "evidence": []}]}

    client = client_with(negative)
    one = memory.propose_from_episode(client, episode, admission_ref=admission,
        learning_request_ref=first, before_call=lambda _: True, after_call=lambda _: None)
    two = memory.propose_from_episode(client, episode, admission_ref=admission,
        learning_request_ref=second, before_call=lambda _: True, after_call=lambda _: None)
    assert one["extraction_ref"] != two["extraction_ref"] and len(client.requests) == 2
    assert memory.read(one["extraction_ref"])["learning_request_ref"] == first
    assert memory.read(two["extraction_ref"])["learning_request_ref"] == second


@pytest.mark.parametrize("bad", ["unselected", "other_episode", "duplicate_episode", "omission_file"])
def test_invalid_model_feedback_is_charged_retained_and_not_retried(tmp_path, bad):
    memory = StrategyExperience(ArtifactStore(tmp_path / "memory"))
    episode, admission = inputs(tmp_path)
    learning = question(tmp_path)

    def output(sent):
        episode_id = sent["episodes"][0]["episode_id"]
        row = {"episode_id": episode_id, "relation": "insufficient", "reason": "Controlled unsupported citation.",
            "evidence": [{"evidence_id": "facts", "locator": ["not_selected"] if bad == "unselected" else ["visible", "status"]}]}
        if bad == "other_episode":
            row["episode_id"] = "unknown-episode"
        if bad == "omission_file":
            row["evidence"][0]["evidence_id"] = "omission_manifest_ref"
        return {"status": "insufficient_information", "reason": "Controlled result.",
            "source_assessments": [row, deepcopy(row)] if bad == "duplicate_episode" else [row]}

    client, receipts = client_with(output), []
    with pytest.raises(ValueError):
        memory.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
            before_call=lambda _: True, after_call=receipts.append)
    assert len(receipts) == 1 and receipts[0]["usage"]["total_tokens"] == 13
    repeated = memory.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
        before_call=lambda _: pytest.fail("Invalid received output must not retry"), after_call=lambda _: None)
    assert repeated["status"] == "response_rejected" and len(client.requests) == 1


def test_unknown_dispatch_keeps_question_binding_and_never_retries(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "memory"))
    episode, admission = inputs(tmp_path)
    learning, requests, receipts = question(tmp_path), [], []

    def fail(request):
        requests.append(request)
        raise TimeoutError("controlled unknown outcome")

    client = SimpleNamespace(provider="controlled", model="fixed", complete_turn=fail)
    with pytest.raises(TimeoutError):
        memory.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
            before_call=lambda _: True, after_call=receipts.append)
    repeated = memory.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
        before_call=lambda _: pytest.fail("Unknown must not reserve again"), after_call=lambda _: None)
    assert repeated["status"] == "execution_unknown" and repeated["learning_request_ref"] == learning
    assert len(requests) == 1 and receipts[0]["status"] == "execution_unknown"


def test_same_schema_wrong_task_contract_is_rejected_before_reservation(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "memory"))
    episode, admission = inputs(tmp_path)
    other, _ = inputs(tmp_path, run="other")
    original = json.loads(Path(episode["path"]).read_text(encoding="utf-8"))
    other_episode = json.loads(Path(other["path"]).read_text(encoding="utf-8"))
    original["evidence_refs"][1] = other_episode["evidence_refs"][1]
    changed = _save(tmp_path / "wrong-contract-episode.json", original)
    with pytest.raises(ValueError, match="TASK_CONTRACT_REFERENCE_CHANGED"):
        memory.propose_from_episode(client_with(lambda _: pytest.fail("Must not call model")), changed,
            admission_ref=admission, learning_request_ref=question(tmp_path),
            before_call=lambda _: pytest.fail("Must not reserve"), after_call=lambda _: None)
