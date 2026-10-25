"""Receipt-backed calibration tests use controlled native responses, never a real model."""
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from scripts import extract_strategy_experience as runner
from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentResponseFormat,
    AgentTurnRequest,
    OllamaChatAgentAdapter,
)
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.planning.learning_calibration import (
    inherit_calibration,
    load_calibration_sources,
)
from tests.unit.test_extract_strategy_experience import identity
from tests.unit.test_extract_strategy_experience import prepared as prepared
from tests.unit.test_strategy_learning_request import question


def source_attempt(tmp_path, profile, *, name="source-attempt", prompt_tokens=1600):
    root = tmp_path / name
    observed = identity(profile.base_url, profile.model)
    request = AgentTurnRequest(messages=(AgentMessage("system", "Original controlled learning task."),
        AgentMessage("user", {"facts": "observed public text " * 250})), tool_choice="none",
        response_format=AgentResponseFormat("strategy", "fixture", {"type": "object"}))
    captured = {}

    def sender(url, payload, headers):
        captured["body"] = httpx.Request("POST", url, json=payload, headers=headers).content
        return {"model": profile.model, "done": True, "done_reason": "stop", "prompt_eval_count": prompt_tokens,
            "eval_count": 20, "prompt_eval_cached_count": 0,
            "message": {"role": "assistant", "content": '{"status":"insufficient_information","reason":"Controlled observation."}'}}

    client = OllamaChatAgentAdapter(base_url=profile.base_url, model=profile.model, context_window=profile.context_window,
        max_output_tokens=profile.output_budget, seed=0, request_timeout_seconds=profile.timeout_seconds, request_sender=sender)
    client.bind_context_identity(model_identity=observed, owner_profile_sha256=canonical_hash(profile.to_dict()))
    response = client.complete_turn(request)
    wire = runner.write_once(root / "placeholder.json", {})  # Directory creation only; not calibration evidence.
    Path(wire["path"]).unlink()
    body_path = root / "model_trace_bodies/request.json"
    body_path.parent.mkdir(parents=True)
    body_path.write_bytes(captured["body"])
    wire = runner.file_ref(body_path)
    application = runner.write_once(root / "application-request.json", asdict(request))
    received = runner.write_once(root / "application-response.json", asdict(response))
    preparation = {"path": str((tmp_path / "old-prepared.json").resolve()), "sha256": "e" * 64}
    started = runner.write_once(root / "started.json", {"profile": profile.to_dict(), "prepared_ref": preparation})
    before = runner.write_once(root / "identity-before.json", observed)
    after = runner.write_once(root / "identity-after.json", observed)
    event = {"request_ref": application, "response_ref": received, "provider_request_ref": wire,
        "provider_request_bytes": len(captured["body"]), "provider_request_sha256": wire["sha256"],
        "provider_request_capture_boundary": "loopback_httpx_request_event",
        "provider_request_ref_scope": "actual_loopback_httpx_request_content", "provider_request_local_body_status": "saved"}
    result = runner.write_once(root / "result.json", {"status": "insufficient_information", "prepared_ref": preparation,
        "model_calls_reserved": 1, "unknown_usage_calls": 0, "budget": {"consumed": 1}, "provider_events": [event],
        "call_receipts": [{"model_calls": 1, "usage": response.usage.to_dict()}]})
    return {"attempt_root": str(root.resolve()), "result_ref": result, "started_ref": started,
        "identity_before_ref": before, "identity_after_ref": after}


def rewrite(reference, value):
    path = Path(reference["path"])
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return runner.file_ref(path)


def profile_record(prepared):
    return runner.read_public(runner.read_public(prepared)["input_ref"])


@pytest.mark.parametrize("mutation", ["body", "digest", "profile", "usage", "done", "truncated", "native_body_link", "outside_root"])
def test_invalid_source_is_rejected_before_reservation_or_model(prepared, tmp_path, monkeypatch, mutation):
    record = profile_record(prepared)
    profile = runner.profile_for(record)
    source = source_attempt(tmp_path, profile)
    result = runner.read_public(source["result_ref"])
    if mutation == "digest":
        before = runner.read_public(source["identity_before_ref"])
        source["identity_before_ref"] = rewrite(source["identity_before_ref"], {**before, "digest": "f" * 64})
    elif mutation == "profile":
        started = runner.read_public(source["started_ref"])
        started["profile"]["output_budget"] = 2048
        source["started_ref"] = rewrite(source["started_ref"], started)
    elif mutation == "body":
        result["provider_events"][0]["provider_request_bytes"] += 1
    else:
        reference = result["provider_events"][0]["response_ref"]
        response = runner.read_public(reference)
        if mutation == "usage":
            response["raw_response"]["prompt_eval_count"] += 1
        elif mutation == "done":
            response["raw_response"]["done"] = False
        elif mutation == "truncated":
            response["raw_response"]["truncated"] = True
        elif mutation == "native_body_link":
            response["provider_diagnostic"]["request_identity"]["provider_body_sha256"] = "0" * 64
        else:
            reference = runner.write_once(tmp_path / "outside-response.json", response)
        result["provider_events"][0]["response_ref"] = reference if mutation == "outside_root" else rewrite(reference, response)
    source["result_ref"] = rewrite(source["result_ref"], result)
    changed = runner.write_once(tmp_path / "calibration-input/input.json", {**record, "context_calibration_sources": [source]})
    monkeypatch.setattr(runner, "read_ollama_model_identity", lambda *_args: pytest.fail("No live identity request in preparation"))
    with pytest.raises(ValueError):
        runner.prepare(changed, tmp_path / "must-not-prepare")
    assert not (tmp_path / "must-not-prepare").exists()


def test_all_selected_observations_use_existing_max_ratio_not_a_favorable_subset(prepared, tmp_path):
    record = profile_record(prepared)
    profile = runner.profile_for(record)
    low = source_attempt(tmp_path, profile, name="low", prompt_tokens=500)
    high = source_attempt(tmp_path, profile, name="high", prompt_tokens=3000)
    bundle = load_calibration_sources([low, high], profile, record["digest"])
    assert [pair[1] for pair in bundle["observations"]] == [500, 3000]
    client = runner.LocalOllamaInitialPatchAdapter._default_agent_turn_client(profile)
    inherit_calibration(client, bundle, profile, identity(profile.base_url, profile.model))
    payload = {"model": profile.model, "messages": [{"role": "user", "content": "x" * 80000}],
        "stream": False, "think": False, "options": {"temperature": 0, "num_ctx": 65536, "num_predict": 4096, "seed": 0}}
    admission = client._context_admission(payload)
    assert admission["observed_calibration_count"] == 2
    assert admission["observed_max_prompt_tokens_per_body_byte"] == max(tokens / size for size, tokens in bundle["observations"])
    assert admission["safety_multiplier"] == 2.0 and admission["admitted"] is False
    with pytest.raises(ValueError, match="DUPLICATE"):
        load_calibration_sources([low, low], profile, record["digest"])


def test_preview_reserve_and_actual_http_share_calibration_without_timeout_rebinding(prepared, tmp_path, monkeypatch):
    record = profile_record(prepared)
    source = source_attempt(tmp_path, runner.profile_for(record))
    learning = question(tmp_path, text="Which evidence is missing? " + "x" * 70000)
    input_ref = runner.write_once(tmp_path / "calibrated-input/input.json", {**record,
        "learning_request_ref": learning, "context_calibration_sources": [source]})
    selected = runner.prepare(input_ref, tmp_path / "calibrated-prepared")
    preview = runner.read_public(runner.preview(selected, tmp_path / "calibrated-preview"))
    expected = runner.read_public(preview["payload_ref"])
    assert preview["admission"]["mode"] == "same_model_observed_prompt_ratio_upper_bound_v1"
    assert preview["admission"]["conservative_fallback_required_context"] > 65536
    assert preview["admission"]["admitted"] is True and preview["model_calls"] == 0
    calls, clients = [], []

    def transport(_transport, request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload == expected
        return httpx.Response(200, request=request, json={"model": "local", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": '{"status":"insufficient_information","reason":"Controlled negative result."}'},
            "prompt_eval_count": 20000, "eval_count": 20, "prompt_eval_cached_count": 0})

    def factory(profile):
        client = runner.LocalOllamaInitialPatchAdapter._default_agent_turn_client(profile)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    attempt = tmp_path / "calibrated-attempt"
    result = runner.read_public(runner.execute(selected, attempt,
        (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(), client_factory=factory, identity_reader=identity))
    assert result["status"] == "insufficient_information", result
    assert len(calls) == len(clients) == 1 and result["model_calls_reserved"] == 1
    preflight = runner.read_public(runner.file_ref(attempt / "context-preflight.json"))["context_admission"]
    response = runner.read_public(result["provider_events"][0]["response_ref"])
    assert preflight == preview["admission"] == response["provider_diagnostic"]["context_admission"]
    assert clients[0].request_timeout_seconds == 600 and clients[0]._current_prefix_binding() is not None
    assert 0 < clients[0]._send.timeout_seconds < 60
    assert result["provider_events"][0]["configured_request_timeout_seconds"] == 600
    assert result["provider_events"][0]["calibration_observation_count"] == 1
    assert result["known_usage"] if "known_usage" in result else result["call_receipts"][0]["usage"]["input_tokens"] == 20000
    assert result["strategy_ref"] is None
