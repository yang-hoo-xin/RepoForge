"""The normal extraction command uses controlled responses, never a real provider or SDK."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import extract_strategy_experience as runner
from src.libs.llm.agent_turn import AgentTurnResponse, AgentUsage


@pytest.fixture
def prepared(tmp_path):
    original = tmp_path / "original"
    data = runner.write_once(original / "facts.json", {"facts": {"status": "timed_out", "completed_reads": 4}})
    episode = runner.write_once(original / "episode.json", {"schema_version": "public_strategy_episode_v1",
        "visibility": "public", "data_use": "strategy_learning", "information_gain": ["public_process_observation"],
        "run_id": "original-run", "task_contract_ref": "original-contract", "applicability": {"runtime": "dotnet"},
        "evidence_refs": [{"kind": "json_file", "evidence_id": "facts", **data, "visibility": "public", "allowed_uses": ["strategy_learning"]}],
        "selections": [{"evidence_id": "facts", "locator": ["facts"]}]})
    input_ref = runner.write_once(tmp_path / "registered" / "input.json", {"schema_version": "strategy_extraction_input_v1",
        "base_url": "http://127.0.0.1:11434", "model": "local", "digest": "d" * 64, "context_window": 65536,
        "output_budget": 4096, "request_timeout_seconds": 600, "process_window_seconds": 600, "episode_ref": episode})
    return runner.prepare(input_ref, tmp_path / "prepared")


def identity(_base_url, _model):
    return {"name": "local", "model": "local", "base_url": "http://127.0.0.1:11434", "digest": "d" * 64,
        "details": {"context_length": 65536}, "capabilities": ["completion"]}


def test_offline_preview_constructs_actual_schema_and_never_sends(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "read_ollama_model_identity", lambda *_args: pytest.fail("No network in preview"))
    reference = runner.preview(prepared, tmp_path / "preview")
    result = runner.read_public(reference)
    assert result["status"] == "preview_only" and result["model_calls"] == 0
    assert result["admission"]["admitted"] is True and not result["live_model_verified"]
    payload = runner.read_public(result["payload_ref"])
    assert payload["model"] == "local" and payload["options"]["num_ctx"] == 65536
    assert "anyOf" in payload["format"] and "timed_out" in str(payload["messages"])
    assert not (Path(prepared["path"]).parent / "execution-started.json").exists()


@pytest.mark.parametrize("outcome", ["completed", "insufficient_information", "unknown"])
def test_at_most_one_controlled_dispatch_and_unknown_is_not_retried(prepared, tmp_path, outcome):
    from src.libs.llm.transport_observation import transport_local_request_body, transport_started
    calls = []

    def complete(request):
        calls.append(request)
        transport_started()
        transport_local_request_body(SimpleNamespace(content=b'{"controlled":true}'))
        if outcome == "unknown":
            raise TimeoutError("controlled transport timeout")
        value = {"status": "insufficient_information", "reason": "No supported strategy."} if outcome == "insufficient_information" else {
            "suggested_action": "Inspect the unresolved question before repeating the same view.",
            "expected_observation": "A distinct observation addresses that question.",
            "counterevidence": ["The prior read has not returned."]}
        return AgentTurnResponse("ollama_chat_local", "local", json.dumps(value), stop_reason="stop", usage=AgentUsage(input_tokens=5, output_tokens=7))

    client = SimpleNamespace(provider="ollama_chat_local", model="local", complete_turn=complete, bind_context_identity=lambda **_kwargs: None)
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    result = runner.read_public(runner.execute(prepared, tmp_path / "attempt", deadline,
        client_factory=lambda _profile: client, identity_reader=identity))
    assert len(calls) == 1 and result["model_calls_reserved"] == 1
    assert result["status"] == ("failed" if outcome == "unknown" else outcome)
    assert result["unknown_usage_calls"] == (1 if outcome == "unknown" else 0)
    assert runner.read_public(runner.file_ref(tmp_path / "attempt/context-preflight.json"))["context_admission"]["admitted"] is True
    assert result["provider_events"][0]["provider_request_ref_scope"] == "actual_loopback_httpx_request_content"
    with pytest.raises(FileExistsError):
        runner.execute(prepared, tmp_path / "second-output", deadline, client_factory=lambda _profile: pytest.fail("Must not retry"), identity_reader=identity)


def test_context_rejection_happens_before_reservation_or_model_send(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "build_ollama_context_admission", lambda **_kwargs: {"admitted": False})
    result = runner.read_public(runner.execute(prepared, tmp_path / "too-large",
        (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
        client_factory=lambda _profile: pytest.fail("No model before preflight"), identity_reader=identity))
    assert result["status"] == "failed" and result["model_calls_reserved"] == 0
    assert result["unknown_usage_calls"] == 0
    assert not (tmp_path / "too-large/reservation.json").exists()


def test_expired_parent_stops_before_output_and_provider(prepared, tmp_path, monkeypatch):
    monkeypatch.setenv("REPOFORGE_TASK_DEADLINE_UTC", "0")
    with pytest.raises(ValueError, match="REGISTERED_DEADLINE"):
        runner.execute(prepared, tmp_path / "expired", (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
            identity_reader=lambda *_args: pytest.fail("No provider after stop"))
    assert not (tmp_path / "expired").exists()


def test_changed_original_evidence_is_rejected_without_model(prepared, tmp_path):
    (tmp_path / "original/facts.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError):
        runner.preview(prepared, tmp_path / "changed")


def test_source_bytes_are_retained_and_tampering_is_rejected(prepared):
    value = runner.read_public(prepared)
    assert len(value["retained_owner_sources"]) > 10
    for row in value["retained_owner_sources"]:
        assert runner.file_ref(row["retained_ref"]["path"])["sha256"] == row["original_ref"]["sha256"]
    Path(value["retained_owner_sources"][0]["retained_ref"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="RETAINED_OWNER_CHANGED"):
        runner.verify_prepared(prepared)


def test_generic_window_is_positive_finite_not_a_new_fixed_600_second_gate(prepared):
    record = runner.read_public(runner.read_public(prepared)["input_ref"])
    runner.profile_for({**record, "process_window_seconds": 1200})
    for value in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="BOUNDED_EXTRACTION_WINDOW"):
            runner.profile_for({**record, "process_window_seconds": value})
