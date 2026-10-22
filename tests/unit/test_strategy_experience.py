"""Actual artifact and request provenance with a controlled provider."""
import json
from dataclasses import asdict

import pytest

from src.libs.llm.agent_turn import AgentTurnResponse, AgentUsage
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.planning.strategy_experience import StrategyExperience


@pytest.fixture
def setup(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "artifacts"))
    source = _save(tmp_path / "public-source.json", {"request": "Existing readers must see updates.",
        "observation": "Old reader saw stale state; a new reader saw the update."})
    evidence = [{"kind": "json_file", "evidence_id": "public", **source,
        "visibility": "public", "allowed_uses": ["strategy_learning"]}]
    strategy = {"applicability": {"mechanism": "retained_state"},
        "suggested_action": "Compare objects created before and after the update.",
        "expected_observation": "Only the old reader remains stale.",
        "counterevidence": ["Both reader instances see the same state."]}
    policy = _save(tmp_path / "trial-policy.json", {"schema_version": "strategy_trial_policy_v1",
        "allowed_origin_kinds": ["human", "model"], "allowed_domains": {"mechanism": ["retained_state"]}})
    return memory, evidence, strategy, policy


def manual(setup):
    memory, evidence, strategy, policy = setup
    proposed = memory.record_strategy("retained-reader", strategy, evidence_refs=evidence,
        origin={"kind": "human", "actor": "test-developer"})
    return proposed, memory.admit_trial(proposed, policy_ref=policy)


def test_trial_keeps_version_and_omits_unknown_applicability(setup):
    memory, _, _, _ = setup
    proposed, snapshot = manual(setup)
    assert memory.read(proposed)["state"] == "proposed"
    assert not memory.select(snapshot, {})["selected"]
    assert not memory.select(snapshot, {"mechanism": "different"})["selected"]
    selection = memory.select(snapshot, {"mechanism": "retained_state"})
    view = memory.advisory_model_view(selection)
    assert view["authority"] == "advisory_only"
    assert view["strategies"][0]["origin"] == "human"
    assert "public-source.json" not in json.dumps(view)


def test_revised_strategy_does_not_replace_frozen_trial(setup):
    memory, evidence, strategy, _ = setup
    _, snapshot = manual(setup)
    current = memory.current_strategy("retained-reader")
    revised = {**strategy, "applicability": {**strategy["applicability"], "update_kind": "in_place"}}
    new = memory.record_strategy("retained-reader", revised, evidence_refs=evidence,
        origin={"kind": "human", "actor": "test-developer"}, expected_revision=current)
    assert memory.read(new)["state"] == "proposed"
    with pytest.raises(ValueError, match="SNAPSHOT_REVOKED_OR_REVISED"):
        memory.check_snapshot(snapshot)
    with pytest.raises(ValueError, match="REVISION_CONFLICT"):
        memory.record_strategy("retained-reader", revised, evidence_refs=evidence,
            origin={"kind": "human", "actor": "test-developer"}, expected_revision=current)


def test_source_changes_invalidate_selection(setup):
    memory, evidence, _, _ = setup
    _, snapshot = manual(setup)
    from pathlib import Path
    Path(evidence[0]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash changed"):
        memory.select(snapshot, {"mechanism": "retained_state"})


class Provider:
    provider, model = "controlled", "fixed-weights"

    def __init__(self, strategy):
        self.strategy, self.requests = strategy, []

    def complete_turn(self, request):
        self.requests.append(request)
        return AgentTurnResponse(self.provider, self.model,
            json.dumps({k: v for k, v in self.strategy.items() if k != "applicability"}),
            stop_reason="stop", usage=AgentUsage(input_tokens=23, output_tokens=17))


def extraction_inputs(tmp_path, setup):
    _, evidence, strategy, _ = setup
    episode = _save(tmp_path / "episode.json", {"schema_version": "public_strategy_episode_v1",
        "visibility": "public", "data_use": "strategy_learning", "information_gain": ["public_counterexample"],
        "evidence_refs": evidence, "selections": [{"evidence_id": "public", "locator": ["observation"]}],
        "applicability": strategy["applicability"]})
    admission = _save(tmp_path / "admission.json", {"status": "admitted", "purpose": "strategy_learning",
        "provider": Provider.provider, "model": Provider.model})
    return episode, admission


def test_real_request_response_and_cost_receipt_are_saved_once(tmp_path, setup):
    memory, _, strategy, _ = setup
    episode, admission = extraction_inputs(tmp_path, setup)
    client, receipts = Provider(strategy), []
    result = memory.propose_from_episode(client, episode, admission_ref=admission,
        before_call=lambda ref: True, after_call=receipts.append)
    assert result["status"] == "completed"
    assert len(client.requests) == 1 and receipts[0]["usage"]["total_tokens"] == 40
    request = memory.read(result["request_ref"])
    assert request["request"] == json.loads(json.dumps(asdict(client.requests[0])))
    assert memory.read(result["response_ref"])["response"]["content"]
    assert memory.read(result["strategy_ref"])["state"] == "proposed"
    again = memory.propose_from_episode(client, episode, admission_ref=admission,
        before_call=lambda ref: True, after_call=receipts.append)
    assert again["strategy_ref"] == result["strategy_ref"] and len(client.requests) == 1


def test_existing_budget_denial_records_pending_without_provider_call(tmp_path, setup):
    memory, _, strategy, _ = setup
    episode, admission = extraction_inputs(tmp_path, setup)
    client = Provider(strategy)
    result = memory.propose_from_episode(client, episode, admission_ref=admission,
        before_call=lambda ref: False, after_call=lambda r: pytest.fail("not called"))
    assert result["status"] == "pending_budget" and not client.requests


def test_nonpublic_evidence_rejected_before_source_read_or_model(tmp_path, setup):
    memory, evidence, strategy, _ = setup
    denied = [{**evidence[0], "visibility": "hidden", "path": str(tmp_path / "nonexistent-hidden.json")}]
    with pytest.raises(ValueError, match="DATA_USE_DENIED"):
        memory.record_strategy("bad", strategy, evidence_refs=denied, origin={"kind": "human", "actor": "test"})


def test_unadmitted_provider_cannot_extract(tmp_path, setup):
    memory, _, strategy, _ = setup
    episode, admission = extraction_inputs(tmp_path, setup)
    client = Provider(strategy)
    client.model = "different"
    with pytest.raises(ValueError, match="PROVIDER_NOT_ADMITTED"):
        memory.propose_from_episode(client, episode, admission_ref=admission,
            before_call=lambda ref: True, after_call=lambda r: None)
    assert not client.requests


def test_unknown_provider_execution_is_recorded_and_never_replayed(tmp_path, setup):
    memory, _, strategy, _ = setup
    episode, admission = extraction_inputs(tmp_path, setup)
    class FailingProvider(Provider):
        def complete_turn(self, request):
            self.requests.append(request)
            raise TimeoutError("controlled unknown outcome")
    client, receipts = FailingProvider(strategy), []
    with pytest.raises(TimeoutError):
        memory.propose_from_episode(client, episode, admission_ref=admission,
            before_call=lambda ref: True, after_call=receipts.append)
    assert receipts[0]["status"] == "execution_unknown" and receipts[0]["model_calls"] == 1
    repeated = memory.propose_from_episode(client, episode, admission_ref=admission,
        before_call=lambda ref: True, after_call=receipts.append)
    assert repeated["status"] == "execution_unknown" and len(client.requests) == 1
