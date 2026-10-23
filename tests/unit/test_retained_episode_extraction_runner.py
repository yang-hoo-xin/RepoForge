"""Pure controlled API-runner checks; no model, SDK or network is invoked."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.libs.llm.agent_turn import AgentTurnResponse, AgentUsage
from src.mod_agent.evaluation.independent_preparation import file_ref, write_once


@pytest.fixture
def runner(tmp_path, monkeypatch):
    path = Path("artifacts/eval_driven_20261023/retained-episode-learning/run_extraction.py").resolve()
    spec = importlib.util.spec_from_file_location("retained_episode_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "original"
    store = original / "store"
    store.mkdir(parents=True)
    write_once(store / "old.json", {"status": "dispatch_started"})
    refs = [{"kind": "json_file", "evidence_id": str(index), "visibility": "public",
        "allowed_uses": ["strategy_learning"], **write_once(original / f"source-{index}.json", {"observation": "failed compilation"})}
        for index in range(25)]
    episode = write_once(original / "episode.json", {"schema_version": "public_strategy_episode_v1",
        "visibility": "public", "data_use": "strategy_learning", "information_gain": ["public_counterexample"],
        "evidence_refs": refs, "selections": [{"evidence_id": "0", "locator": ["observation"]}],
        "applicability": {"counterexample_kind": "public_compilation_counterexample"}})
    intent = json.loads(module.INTENT.read_text(encoding="utf-8"))
    intent.update(episode_ref=episode, original_store=str(store), original_run_root=str(original))
    reference = write_once(tmp_path / "intent.json", intent)
    monkeypatch.setattr(module, "INTENT", Path(reference["path"]))
    monkeypatch.setattr(module, "EPISODE_HASH", episode["sha256"])
    return module


def test_preflight_has_no_execution_or_writes(runner, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runner, "read_ollama_model_identity", lambda *a: pytest.fail("network not allowed"))
    before = {str(p): file_ref(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert runner.main([]) == 0
    assert json.loads(capsys.readouterr().out)["writes"] == 0
    assert before == {str(p): file_ref(p) for p in tmp_path.rglob("*") if p.is_file()}
    intent, _ = runner.preflight()
    with pytest.raises(ValueError, match="OVERLAP"):
        runner.output_path(tmp_path / "original" / "new", intent)


@pytest.mark.parametrize("fail", [False, True])
def test_one_controlled_call_and_preserved_originals(runner, monkeypatch, tmp_path, fail,
                                                   unreadable_postcheck=False, expire_before_send=False):
    from src.libs.llm.transport_observation import (
        transport_local_request_body,
        transport_response_received,
        transport_started,
    )
    intent, original = runner.preflight()
    identity = {"name": intent["model"], "model": intent["model"], "digest": intent["digest"],
        "base_url": "http://127.0.0.1:11434", "details": {"context_length": 65536}, "capabilities": ["completion"]}
    monkeypatch.setattr(runner, "read_ollama_model_identity", lambda *a: identity)
    calls = []

    def complete(request):
        calls.append(request)
        transport_started()
        transport_local_request_body(SimpleNamespace(content=b'{"controlled":true}'))
        if fail:
            raise TimeoutError("controlled transport timeout")
        transport_response_received(http_status=200, response_complete=True)
        return AgentTurnResponse("ollama_chat_local", intent["model"], json.dumps({
            "suggested_action": "Read the reported API signature before revising a call.",
            "expected_observation": "The signature differs from the assumed argument shape.",
            "counterevidence": ["The call already matches the signature."]}),
            stop_reason="stop", usage=AgentUsage(input_tokens=5, output_tokens=7))

    original_file_ref = runner.file_ref
    def bind(**_kwargs):
        if expire_before_send:
            class ExpiredDateTime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return datetime.now(timezone.utc) + timedelta(seconds=120)
            monkeypatch.setattr(runner, "datetime", ExpiredDateTime)
        if unreadable_postcheck:
            def blocked(path):
                if Path(path) == Path(original[0]["path"]):
                    raise PermissionError("controlled post-response access failure")
                return original_file_ref(path)
            monkeypatch.setattr(runner, "file_ref", blocked)
    client = SimpleNamespace(provider="ollama_chat_local", model=intent["model"], complete_turn=complete,
        bind_context_identity=bind)
    monkeypatch.setattr(runner.LocalOllamaInitialPatchAdapter, "_default_agent_turn_client", lambda profile: client)
    output = tmp_path / "new-attempt"
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    result_ref = runner.execute(str(output), deadline, identity["base_url"])
    result = json.loads(Path(result_ref["path"]).read_text(encoding="utf-8"))
    assert len(calls) == (0 if expire_before_send else 1)
    assert result["budget"]["consumed"] == 1
    assert result["status"] == ("integrity_unconfirmed" if unreadable_postcheck
        else "failed" if fail or expire_before_send else "completed")
    assert result["original_refs_unchanged"] is (None if unreadable_postcheck else True)
    assert all(file_ref(r["path"]) == r for r in original)
    if expire_before_send:
        assert "provider_request_ref" not in result["provider_events"][0]
    else:
        assert result["provider_events"][0]["provider_request_ref_scope"] == "actual_loopback_httpx_request_content"
    assert result["trace_recording_failures"]["recorder"]["total"] == 0
    if not fail and not expire_before_send:
        record = json.loads(Path(result["strategy_ref"]["path"]).read_text(encoding="utf-8"))["data"]
        assert record["state"] == "proposed" and record["origin"]["kind"] == "model"
        assert result["call_receipts"][0]["usage"]["total_tokens"] == 12
        if unreadable_postcheck:
            assert result["extraction_status"] == "completed"
            assert result["provider_events"][0]["response_ref"]
    else:
        assert result["strategy_ref"] is None
        assert result["call_receipts"][0]["status"] == "execution_unknown"
    with pytest.raises(ValueError, match="NEW_ABSOLUTE_OUTPUT_REQUIRED"):
        runner.execute(str(output), deadline, identity["base_url"])


def test_postresponse_integrity_failure_retains_actual_response_and_cost(runner, monkeypatch, tmp_path):
    test_one_controlled_call_and_preserved_originals(runner, monkeypatch, tmp_path, False, unreadable_postcheck=True)


def test_expiry_after_identity_binding_does_not_send_or_refund_reservation(runner, monkeypatch, tmp_path):
    test_one_controlled_call_and_preserved_originals(runner, monkeypatch, tmp_path, False, expire_before_send=True)


def test_inherited_deadline_cannot_be_extended_by_arguments(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("REPOFORGE_TASK_DEADLINE_UTC", "0")
    monkeypatch.setattr(runner, "read_ollama_model_identity", lambda *a: pytest.fail("Expired operation made a network request"))
    deadline = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    with pytest.raises(ValueError, match="PARENT_UTC_DEADLINE"):
        runner.execute(str(tmp_path / "not-created"), deadline, "http://127.0.0.1:11434")
    assert not (tmp_path / "not-created").exists()
