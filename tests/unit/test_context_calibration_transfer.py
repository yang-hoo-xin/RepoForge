"""Real native request construction with controlled senders; no model endpoint."""
from dataclasses import FrozenInstanceError, replace

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentTurnProviderError,
    AgentTurnRequest,
    OllamaChatAgentAdapter,
)
from src.mod_agent.local_entry import LocalOllamaInitialPatchAdapter


def _identity(**changes):
    return {"base_url": "http://127.0.0.1:11434", "name": "qwen3-coder:30b",
            "model": "qwen3-coder:30b", "digest": "a" * 64, **changes}


def _adapter(sender, **changes):
    return OllamaChatAgentAdapter(**{"base_url": "http://127.0.0.1:11434", "model": "qwen3-coder:30b",
        "context_window": 32768, "max_output_tokens": 4096, "seed": 0,
        "request_timeout_seconds": 180.0, "request_sender": sender, **changes})


def _response(**changes):
    return {"model": "qwen3-coder:30b", "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": "done"}, "prompt_eval_count": 200,
        "eval_count": 1, **changes}


def _seed(adapter):
    adapter.bind_context_identity(model_identity=_identity(), owner_profile_sha256="f" * 64)
    adapter.complete_turn(AgentTurnRequest(messages=(AgentMessage("user", "first observed input " * 100),)))
    return adapter.export_context_calibration(model_identity=_identity(), owner_profile_sha256="f" * 64)


def test_detached_calibration_admits_large_request_with_original_32k_gate_and_new_sender():
    old_sent, new_sent = [], []
    source = _adapter(lambda _url, body, _headers: old_sent.append(body) or _response())
    calibration = _seed(source)
    target = _adapter(lambda _url, body, _headers: new_sent.append(body) or _response())
    large = AgentTurnRequest(messages=(AgentMessage("user", "x" * 34001),))
    with pytest.raises(AgentTurnProviderError, match="pre-call context budget"):
        target.complete_turn(large)
    assert new_sent == []
    assert target.inherit_context_calibration(calibration,
        model_identity=_identity(), owner_profile_sha256="f" * 64)
    source._send = lambda *_: pytest.fail("Old transport/recording/replay closure must not execute")
    source._prompt_token_calibrations.clear()
    result = target.complete_turn(large)
    admission = result.provider_diagnostic["context_admission"]
    assert len(old_sent) == len(new_sent) == 1
    assert new_sent[0]["options"]["num_ctx"] == 32768
    assert new_sent[0]["options"]["num_predict"] == 4096
    assert admission["mode"] == "same_model_observed_prompt_ratio_upper_bound_v1"
    assert admission["observed_calibration_count"] == 1
    assert admission["conservative_fallback_required_context"] > 32768
    assert admission["estimated_required_context"] <= 32768
    with pytest.raises(FrozenInstanceError):
        calibration.owner_profile_sha256 = "e" * 64
    too_large = AgentTurnRequest(messages=(AgentMessage("user", "x" * 70000),))
    with pytest.raises(AgentTurnProviderError, match="pre-call context budget"):
        target.complete_turn(too_large)
    assert len(new_sent) == 1  # Calibration never removes the context gate.


@pytest.mark.parametrize("drift", ["digest", "base_url", "model", "context_window", "max_output_tokens", "seed", "request_timeout_seconds", "owner_profile"])
def test_identity_and_every_profile_drift_cannot_borrow_calibration(drift):
    calibration = _seed(_adapter(lambda *_: _response()))
    changes, identity, owner = {}, _identity(), "f" * 64
    if drift == "digest":
        identity["digest"] = "b" * 64
    elif drift == "owner_profile":
        owner = "e" * 64
    else:
        changes[drift] = {"base_url": "http://127.0.0.1:11435", "model": "different-model",
            "context_window": 16384, "max_output_tokens": 2048, "seed": 1,
            "request_timeout_seconds": 120.0}[drift]
        if drift == "base_url":
            identity["base_url"] = changes[drift]
        elif drift == "model":
            identity.update(name=changes[drift], model=changes[drift])
    client = _adapter(lambda *_: pytest.fail("No send for mismatched calibration"), **changes)
    assert client.inherit_context_calibration(calibration, model_identity=identity, owner_profile_sha256=owner) is False
    assert client._prompt_token_calibrations == []


@pytest.mark.parametrize("changes", [{"prompt_eval_count": None}, {"prompt_eval_count": True},
    {"prompt_eval_count": 0}, {"prompt_eval_count": -1}, {"done": False}, {"model": "different-model"}])
def test_unknown_incomplete_or_other_model_counts_are_not_exported(changes):
    client = _adapter(lambda *_: _response(**changes))
    assert _seed(client).observations == ()
    if changes.get("done") is False:
        assert len(client._prompt_token_calibrations) == 1  # Existing in-client/replay behavior is unchanged.


def test_import_rejects_malformed_observations_and_replay_worker_exports_none():
    calibration = _seed(_adapter(lambda *_: _response()))
    client = _adapter(lambda *_: pytest.fail("No provider call"))
    with pytest.raises(AgentTurnProviderError, match="OBSERVATIONS_INVALID"):
        client.inherit_context_calibration(replace(calibration, observations=((2000, True),)),
            model_identity=_identity(), owner_profile_sha256="f" * 64)
    worker = object.__new__(LocalOllamaInitialPatchAdapter)
    worker._context_calibration_client = client
    worker._replayed_turn_count = 1
    assert worker.retained_context_calibration() is None
