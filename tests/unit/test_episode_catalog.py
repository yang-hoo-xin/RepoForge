"""Public catalog selection is actual-data selection, not an inferred success history.

为什么要有它：新增运行、同run变体和输出schema变化必须影响正确的身份，同时旧冻结集及未知派发保持可追溯。
"""
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.libs.llm.agent_turn import AgentResponseFormat
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.planning import strategy_experience as module
from src.mod_agent.planning.strategy_experience import StrategyExperience
from tests.unit.test_strategy_learning_request import client_with, inputs, question


def memory(tmp_path):
    return StrategyExperience(ArtifactStore(tmp_path / "memory"))


def native_views(tmp_path, request_id="request-one"):
    sources, calls = [], []
    for index in range(2):
        source = _save(tmp_path / ("action-" + str(index) + ".json"), {
            "arguments": {"path": str(index) + ".cs"}, "output": {"text": "public " + str(index)}})
        sources.append({"kind": "json_file", "evidence_id": "action_" + str(index), **source,
            "visibility": "public", "allowed_uses": ["strategy_learning"]})
        calls.append({"call_id": "call-" + str(index), "name": "source_read",
            "arguments_ref": source, "response_ref": source, "execution_receipt_ref": source})

    def view(name, count, terminal):
        execution = {"run_id": "run-one", "request_id": request_id, "completed_tool_records": calls[:count],
            "status": "failed_no_candidate" if terminal else "timed_out"}
        execution.update({"schema_version": "task-agent-product-execution.v2"} if terminal else
            {"schema": "task-agent-public-process-observation.v1", "native_terminal": False})
        outcome = _save(tmp_path / (name + "-execution.json"), execution)
        references = sources[:count] + [{"kind": "json_file", "evidence_id": "technical_outcome", **outcome,
            "visibility": "public", "allowed_uses": ["strategy_learning"]}]
        selectors = [{"evidence_id": "action_" + str(index), "locator": [field],
            "role": role, "action_index": index} for index in range(count)
            for field, role in [("arguments", "action_request"), ("output", "action_return")]]
        selectors.append({"evidence_id": "technical_outcome", "locator": ["status"], "role": "technical_outcome"})
        return _save(tmp_path / (name + "-episode.json"), {"schema_version": "public_strategy_episode_v1",
            "visibility": "public", "data_use": "strategy_learning", "run_id": "run-one",
            "request_id": request_id, "task_contract_ref": "contract-one", "execution_ref": outcome,
            "projection_version": "ordered_public_actions_v1", "information_gain": ["public_process_observation"],
            "applicability": {"runtime": "dotnet"}, "evidence_refs": references, "selections": selectors})
    return view("prefix", 1, False), view("terminal", 2, True)


def test_registration_is_idempotent_and_new_run_enters_next_frozen_set(tmp_path):
    owner = memory(tmp_path)
    first, _ = inputs(tmp_path, run="first")
    registered = owner.register_public_episode("project", first)
    assert registered == owner.register_public_episode("project", first)
    old = owner.freeze_catalog("project", applicability={"runtime": "dotnet"})
    second, _ = inputs(tmp_path, run="second")
    owner.register_public_episode("project", second)
    new = owner.freeze_catalog("project", applicability={"runtime": "dotnet"},
        selection={"required_episode_refs": [second]})
    assert len(owner.verify_catalog_selection(old)["episodes"]) == 1
    assert {row["run_id"] for row in owner.verify_catalog_selection(new)["episodes"]} == {"first", "second"}


def test_native_v2_and_stop_prefix_count_one_run_and_disclose_unmerged_values(tmp_path):
    owner = memory(tmp_path)
    prefix, terminal = native_views(tmp_path)
    owner.register_public_episode("project", prefix)
    owner.register_public_episode("project", terminal)
    frozen = owner.freeze_catalog("project", applicability={"runtime": "dotnet"},
        selection={"required_episode_refs": [terminal]})
    data = owner.verify_catalog_selection(frozen)
    receipt = owner.read(data["catalog_selection_ref"])
    assert [row["episode_ref"] for row in data["episodes"]] == [terminal]
    run = receipt["runs"][0]
    assert run["covered_refs"] == [terminal] and run["excluded_refs"] == [prefix]
    assert "not_fact_superset" in run["selection_reason"]
    omitted = run["omitted_selected_facts"][0]
    assert omitted["episode_ref"] == prefix
    assert any(fact["locator"] == ["status"] for fact in omitted["facts"])
    with pytest.raises(ValueError, match="REQUIRED_EPISODE_NOT_REPRESENTED"):
        owner.freeze_catalog("project", applicability={"runtime": "dotnet"},
            selection={"required_episode_refs": [prefix]})
    chosen = owner.freeze_catalog("project", applicability={"runtime": "dotnet"},
        selection={"required_episode_refs": [prefix], "run_choices": {"run-one": prefix}})
    assert owner.read(chosen)["episodes"][0]["episode_ref"] == prefix


def test_episode_alias_keeps_both_references_without_two_tasks(tmp_path):
    owner = memory(tmp_path)
    first, _ = inputs(tmp_path)
    alias = _save(tmp_path / "alias.json", json.loads(Path(first["path"]).read_bytes()))
    owner.register_public_episode("project", first)
    owner.register_public_episode("project", alias)
    frozen = owner.freeze_catalog("project", applicability={"runtime": "dotnet"},
        selection={"required_episode_refs": [first, alias]})
    receipt = owner.read(owner.read(frozen)["catalog_selection_ref"])
    assert len(receipt["runs"]) == len(owner.read(frozen)["episodes"]) == 1
    assert {canonical_hash(r) for r in receipt["runs"][0]["covered_refs"]} == {canonical_hash(first), canonical_hash(alias)}


def test_request_identity_unknown_is_retained_and_not_guessed(tmp_path):
    owner = memory(tmp_path)
    first, _ = inputs(tmp_path, task_context=False)
    document = json.loads(Path(first["path"]).read_bytes())
    unknown = _save(tmp_path / "unknown-request.json", {**document, "request_id": None})
    owner.register_public_episode("project", unknown)
    owner.register_public_episode("project", first)
    with pytest.raises(ValueError, match="CONFLICT_OR_COVERAGE_UNPROVEN"):
        owner.freeze_catalog("project", applicability={"runtime": "dotnet"})
    frozen = owner.freeze_catalog("project", applicability={"runtime": "dotnet"},
        selection={"run_choices": {"first": unknown}})
    receipt = owner.read(owner.read(frozen)["catalog_selection_ref"])
    assert next(row for row in receipt["entries"] if row["episode_ref"] == unknown)["request_id"] is None


def test_private_registration_is_rejected_and_unregistered_choice_cannot_expand_set(tmp_path):
    owner = memory(tmp_path)
    first, _ = inputs(tmp_path)
    document = json.loads(Path(first["path"]).read_bytes())
    private = _save(tmp_path / "private-episode.json", {**document, "visibility": "private"})
    with pytest.raises(ValueError, match="PUBLIC_EPISODE_REQUIRED"):
        owner.register_public_episode("project", private)
    owner.register_public_episode("project", first)
    with pytest.raises(ValueError, match="UNREGISTERED"):
        owner.freeze_catalog("project", applicability={"runtime": "dotnet"},
            selection={"run_choices": {"first": private}})


def test_frozen_episode_list_cannot_be_resigned_to_hide_a_selected_run(tmp_path):
    owner = memory(tmp_path)
    for run in ["first", "second"]:
        episode, _ = inputs(tmp_path, run=run)
        owner.register_public_episode("project", episode)
    frozen = owner.freeze_catalog("project", applicability={"runtime": "dotnet"})
    changed = owner.read(frozen)
    changed["episodes"] = changed["episodes"][:1]
    changed.pop("logical_namespace")
    forged = owner._write("forged-catalog-set", "episode_set", changed)
    with pytest.raises(ValueError, match="FROZEN_SELECTION_CHANGED"):
        owner.verify_catalog_selection(forged)


def test_schema_change_changes_request_identity_even_with_same_question(tmp_path, monkeypatch):
    owner = memory(tmp_path)
    episode, admission = inputs(tmp_path)
    learning = question(tmp_path)
    client = client_with(lambda _: {"status": "insufficient_information", "reason": "Controlled no proposal."})
    first = owner.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
        before_call=lambda _: True, after_call=lambda _: None)
    original = module._build_extraction_request
    def changed(*args, **kwargs):
        request = original(*args, **kwargs)
        schema = deepcopy(request.response_format.schema)
        schema["description"] = "A newly registered output contract."
        return replace(request, response_format=AgentResponseFormat("strategy", "Conditional strategy hypothesis", schema))
    monkeypatch.setattr(module, "_build_extraction_request", changed)
    second = owner.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
        before_call=lambda _: True, after_call=lambda _: None)
    assert first["request_sha256"] != second["request_sha256"] and len(client.requests) == 2


def test_legacy_unknown_same_actual_request_is_not_dispatched_again(tmp_path):
    owner = memory(tmp_path)
    episode, admission = inputs(tmp_path)
    client = client_with(lambda _: {"status": "insufficient_information", "reason": "Controlled."})
    current = owner.propose_from_episode(client, episode, admission_ref=admission,
        before_call=lambda _: False, after_call=lambda _: None)
    identity = {"source": json.loads(Path(episode["path"]).read_bytes()), "extraction_version": module._EXTRACTION_VERSION,
        "model": {"provider": "controlled", "model": "fixed", "model_identity": None, "profile_sha256": None}}
    legacy_feature = "strategy_episode_" + canonical_hash(identity)
    old = {key: deepcopy(value) for key, value in current.items() if key not in {
        "logical_namespace", "kind", "schema_version", "extraction_ref", "request_sha256"}}
    old["status"] = "execution_unknown"
    legacy = owner._write(legacy_feature, "extraction", old)
    replay = owner.propose_from_episode(client, episode, admission_ref=admission,
        before_call=lambda _: pytest.fail("Unknown prior request must not reserve"),
        after_call=lambda _: pytest.fail("No new charge"))
    assert replay["extraction_ref"] == asdict(legacy) and replay["replayed_legacy_request"] is True
    assert replay["status"] == "execution_unknown" and client.requests == []


def long_episode(tmp_path):
    episode, admission = inputs(tmp_path)
    document = json.loads(Path(episode["path"]).read_bytes())
    text = "Recorded public observation with exact source provenance. " * 30
    facts = _save(tmp_path / "long-facts.json", {"visible": {str(index): text for index in range(12)}})
    document["evidence_refs"][0].update(facts)
    return _save(tmp_path / "long-episode.json", document), admission, text


def test_compacted_request_still_validates_citations_against_original_selected_value(tmp_path):
    from src.mod_agent.planning.learning_context import expand_learning_content

    owner = memory(tmp_path)
    episode, admission, original_text = long_episode(tmp_path)

    def respond(encoded):
        assert encoded["schema_version"] == "learning_literal_text_refs_v1"
        original = expand_learning_content(encoded)
        row = original["episodes"][0]
        assert row["observations"][0]["value"]["0"] == original_text
        return {"status": "insufficient_information", "reason": "Controlled observation remains inconclusive.",
            "source_assessments": [{"episode_id": row["episode_id"], "relation": "insufficient",
                "reason": "Exact public value", "evidence": [{"evidence_id": "facts", "locator": ["visible", "0"]}]}]}

    client = client_with(respond)
    result = owner.propose_from_episode(client, episode, admission_ref=admission,
        learning_request_ref=question(tmp_path), before_call=lambda _: True, after_call=lambda _: None)
    assert result["status"] == "insufficient_information"
    assert result["source_assessments"][0]["evidence"] == [{"evidence_id": "facts", "locator": ["visible", "0"]}]


def test_compacted_unknown_request_keeps_identity_and_does_not_send_again(tmp_path):
    from types import SimpleNamespace

    owner = memory(tmp_path)
    episode, admission, _ = long_episode(tmp_path)
    learning = question(tmp_path)
    calls = []

    def fail(request):
        calls.append(request)
        assert request.messages[-1].content["schema_version"] == "learning_literal_text_refs_v1"
        raise TimeoutError("controlled unknown")

    client = SimpleNamespace(provider="controlled", model="fixed", complete_turn=fail)
    with pytest.raises(TimeoutError):
        owner.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
            before_call=lambda _: True, after_call=lambda _: None)
    repeated = owner.propose_from_episode(client, episode, admission_ref=admission, learning_request_ref=learning,
        before_call=lambda _: pytest.fail("Unknown compressed request cannot reserve again"), after_call=lambda _: None)
    assert repeated["status"] == "execution_unknown" and len(calls) == 1
    assert repeated["request_sha256"] == canonical_hash(asdict(calls[0]))
