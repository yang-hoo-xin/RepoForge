"""Dynamic public collections reach the existing extraction command and HTTP client."""
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from scripts import extract_strategy_experience as runner
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.planning.strategy_experience import StrategyExperience
from tests.unit.test_extract_strategy_experience import identity
from tests.unit.test_strategy_learning_request import question


def episode(tmp_path, run_id):
    source = runner.write_once(tmp_path / run_id / "facts.json", {
        "facts": {"status": "timed_out", "completed_reads": 4, "candidate_count": 0}})
    return runner.write_once(tmp_path / run_id / "episode.json", {
        "schema_version": "public_strategy_episode_v1", "visibility": "public", "data_use": "strategy_learning",
        "information_gain": ["public_process_observation"], "run_id": run_id, "request_id": "request-" + run_id,
        "task_contract_ref": "contract-" + run_id, "applicability": {"runtime": "dotnet"},
        "evidence_refs": [{"kind": "json_file", "evidence_id": "facts", **source,
            "visibility": "public", "allowed_uses": ["strategy_learning"]}],
        "selections": [{"evidence_id": "facts", "locator": ["facts"]}]})


def registered_input(tmp_path, memory, *, selection=None):
    return runner.write_once(tmp_path / "registered" / "input.json", {
        "schema_version": "strategy_extraction_input_v1", "base_url": "http://127.0.0.1:11434",
        "model": "local", "digest": "d" * 64, "context_window": 65536, "output_budget": 4096,
        "request_timeout_seconds": 600, "process_window_seconds": 600,
        "learning_request_ref": question(tmp_path), "applicability": {"runtime": "dotnet"},
        "episode_catalog": {"store_root": str(memory.store.root), "catalog_id": "investigation",
            **({"selection": selection} if selection is not None else {})}})


def test_catalog_growth_preserves_prepared_set_and_reaches_actual_http(tmp_path, monkeypatch):
    memory = StrategyExperience(ArtifactStore(tmp_path / "shared-memory"))
    first = episode(tmp_path, "first-run")
    duplicate = runner.write_once(tmp_path / "alias" / "episode.json", runner.read_public(first))
    second = episode(tmp_path, "second-run")
    for reference in (first, duplicate, second):
        memory.register_public_episode("investigation", reference)
    input_ref = registered_input(tmp_path, memory)
    earlier = runner.prepare(input_ref, tmp_path / "earlier")
    earlier_doc = runner.read_public(earlier)
    source = earlier_doc["catalog_source"]
    receipt = memory.read(source["selection_ref"])
    assert len(receipt["entries"]) == 3 and len(receipt["runs"]) == 2
    assert len(earlier_doc["episode_refs"]) == 2

    # 为什么要有它：新失败影响下一次选择，不能悄悄改变已经冻结、可能已经计费的请求。
    newest = episode(tmp_path, "new-no-candidate-run")
    memory.register_public_episode("investigation", newest)
    runner.verify_prepared(earlier)
    old_preview = runner.read_public(runner.preview(earlier, tmp_path / "old-preview"))
    old_wire = runner.read_public(old_preview["payload_ref"])
    assert {row["run_id"] for row in json.loads(old_wire["messages"][-1]["content"])["episodes"]} == {
        "first-run", "second-run"}

    current = runner.prepare(input_ref, tmp_path / "current")
    current_doc = runner.read_public(current)
    assert len(current_doc["episode_refs"]) == 3
    assert current_doc["catalog_source"]["selection_ref"] != source["selection_ref"]
    preview = runner.read_public(runner.preview(current, tmp_path / "current-preview"))
    expected_wire = runner.read_public(preview["payload_ref"])
    calls = []

    def transport(_transport, request):
        payload = json.loads(request.content)
        assert payload == expected_wire
        data = json.loads(payload["messages"][-1]["content"])
        assert {row["run_id"] for row in data["episodes"]} == {
            "first-run", "second-run", "new-no-candidate-run"}
        assert all(row["observations"][0]["value"]["candidate_count"] == 0 for row in data["episodes"])
        calls.append(payload)
        value = {"status": "insufficient_information", "reason": "No comparative outcome is available.",
            "source_assessments": [{"episode_id": row["episode_id"], "relation": "insufficient",
                "reason": "This is a stopped investigation, not an experiment result.",
                "evidence": [{"evidence_id": "facts", "locator": ["facts", "status"]}]} for row in data["episodes"]],
            "evidence_gaps": [{"status": "not_observed", "description": "No alternative action outcome was observed."}]}
        return httpx.Response(200, request=request, json={"model": "local", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(value)}, "prompt_eval_count": 50, "eval_count": 20})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    result = runner.read_public(runner.execute(current, tmp_path / "attempt",
        (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(), identity_reader=identity))
    assert len(calls) == 1 and result["model_calls_reserved"] == 1
    assert result["status"] == "insufficient_information" and result["strategy_ref"] is None
    assert result["unknown_usage_calls"] == 0
    runner.verify_prepared(earlier)


def test_catalog_snapshot_or_selected_source_tampering_is_rejected(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "shared-memory"))
    original = episode(tmp_path, "first-run")
    memory.register_public_episode("investigation", original)
    prepared = runner.prepare(registered_input(tmp_path, memory), tmp_path / "prepared")
    source = runner.read_public(prepared)["catalog_source"]
    document = memory.read(source["frozen_ref"])
    assert document["catalog_selection_ref"] == source["selection_ref"]
    Path(source["selection_ref"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        runner.verify_prepared(prepared)


def test_resigned_subset_cannot_keep_a_full_catalog_selection_receipt(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "shared-memory"))
    for run_id in ("first-run", "second-run"):
        memory.register_public_episode("investigation", episode(tmp_path, run_id))
    prepared = runner.prepare(registered_input(tmp_path, memory), tmp_path / "prepared")
    document = runner.read_public(prepared)
    original = memory.read(document["catalog_source"]["frozen_ref"])
    subset = memory._write("forged-source-subset", "episode_set", {**original, "episodes": original["episodes"][:1]})
    copied_memory = StrategyExperience(ArtifactStore(document["memory_root"]))
    copied = copied_memory.read(document["selection"]["reference"])
    copied_subset = copied_memory._write("forged-copy-subset", "episode_set", {**copied, "episodes": copied["episodes"][:1]})
    forged = {**document, "catalog_source": {**document["catalog_source"], "frozen_ref": asdict(subset)},
        "selection": {"kind": "episode_set", "reference": asdict(copied_subset)},
        "episode_refs": document["episode_refs"][:1]}
    # 为什么要有它：重新签名的合法JSON仍不能把两次运行的选择凭据冒用于只含一次运行的集合。
    reference = runner.write_once(Path(prepared["path"]).parent / "forged-prepared.json", forged)
    with pytest.raises(ValueError, match="FROZEN_SELECTION_CHANGED"):
        runner.verify_prepared(reference)


def test_catalog_cannot_replace_an_explicit_frozen_input_or_overlap_output(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "shared-memory"))
    original = episode(tmp_path, "first-run")
    memory.register_public_episode("investigation", original)
    record = registered_input(tmp_path, memory)
    ambiguous = runner.write_once(tmp_path / "ambiguous.json", {**runner.read_public(record), "episode_ref": original})
    with pytest.raises(ValueError, match="EXACTLY_ONE_INPUT_MODE"):
        runner.prepare(ambiguous, tmp_path / "unused")
    with pytest.raises(ValueError, match="OUTPUT_OVERLAP"):
        runner.prepare(record, memory.store.root / "must-not-write")
    assert not (memory.store.root / "must-not-write").exists()
