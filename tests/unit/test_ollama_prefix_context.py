"""A measured text prefix is reusable only under the same verified local identity."""
from copy import deepcopy
from dataclasses import replace

import pytest

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentResponseFormat,
    AgentToolDefinition,
    AgentTurnProviderError,
    AgentTurnRequest,
    OllamaChatAgentAdapter,
)


def identity(**changes):
    return {"base_url": "http://127.0.0.1:11434", "name": "qwen3-coder:30b",
        "model": "qwen3-coder:30b", "digest": "a" * 64, **changes}


def answer(**changes):
    return {"model": "qwen3-coder:30b", "done": True, "done_reason": "stop",
        "message": {"role": "assistant", "content": "done"},
        "prompt_eval_count": 100, "eval_count": 1, **changes}


def setup(response=None, *, bound=True):
    sent = []
    client = OllamaChatAgentAdapter(base_url="http://127.0.0.1:11434", model="qwen3-coder:30b",
        context_window=4096, max_output_tokens=128,
        request_sender=lambda _url, body, _headers: sent.append(deepcopy(body)) or (response or answer()))
    if bound:
        client.bind_context_identity(model_identity=identity(), owner_profile_sha256="f" * 64)
    first = AgentTurnRequest(messages=(AgentMessage("system", "base"),
        AgentMessage("user", "x" * 3300)))
    client.complete_turn(first)
    appended = replace(first, messages=(*first.messages, AgentMessage("user", "y" * 1400)))
    return client, sent, first, appended


def observed_admission(client, request):
    try:
        return client.complete_turn(request).provider_diagnostic["context_admission"]
    except AgentTurnProviderError as error:
        return error.diagnostic["context_admission"]


def test_complete_prefix_uses_observed_tokens_plus_new_utf8_without_double_counting():
    client, sent, first, appended = setup()
    result = client.complete_turn(appended)
    view = result.provider_diagnostic["context_admission"]
    assert len(sent) == 2 and sent[1]["messages"][:2] == sent[0]["messages"]
    assert view["mode"] == "bound_observed_prefix_plus_appended_utf8_v1"
    assert view["anchor_prompt_tokens"] == 100 and view["anchor_message_count"] == 2
    assert view["appended_messages_utf8_bytes"] >= 1400
    assert view["estimated_required_context"] == (
        100 + view["appended_messages_utf8_bytes"] + view["template_margin_tokens"] + 128)
    assert view["conservative_fallback_required_context"] > 4096
    assert view["estimated_required_context"] < 4096
    assert sent[-1]["options"]["num_ctx"] == 4096 and sent[-1]["options"]["num_predict"] == 128
    assert first.messages[1].content == "x" * 3300


def test_prefix_still_rejects_addition_that_cannot_fit_before_send():
    client, sent, first, _ = setup()
    large = replace(first, messages=(*first.messages, AgentMessage("user", "大" * 2000)))
    with pytest.raises(AgentTurnProviderError) as caught:
        client.complete_turn(large)
    view = caught.value.diagnostic["context_admission"]
    assert view["mode"] == "bound_observed_prefix_plus_appended_utf8_v1"
    assert not view["admitted"] and view["appended_messages_utf8_bytes"] >= 6000
    assert len(sent) == 1


@pytest.mark.parametrize("change", ["edit", "delete", "reorder", "tools", "format", "temperature"])
def test_changed_prefix_or_prompt_contract_uses_existing_fallback(change):
    client, _, first, request = setup()
    if change == "edit":
        request = replace(request, messages=(AgentMessage("system", "changed"), *request.messages[1:]))
    elif change == "delete":
        request = replace(request, messages=request.messages[1:])
    elif change == "reorder":
        request = replace(request, messages=(first.messages[1], first.messages[0], request.messages[-1]))
    elif change == "tools":
        request = replace(request, tools=(AgentToolDefinition("search", "Search", {"type": "object"}),))
    elif change == "format":
        request = replace(request, response_format=AgentResponseFormat("answer", "Answer", {"type": "object"}))
    else:
        request = replace(request, temperature=0.2)
    result = client.complete_turn(request)
    assert result.provider_diagnostic["context_admission"]["mode"] != "bound_observed_prefix_plus_appended_utf8_v1"


@pytest.mark.parametrize("response", [
    answer(prompt_eval_count=None), answer(prompt_eval_count=True), answer(prompt_eval_count=0),
    answer(eval_count=None), answer(eval_count=True), answer(done=False),
    answer(done_reason="length"), answer(model="other"), answer(truncated=True),
])
def test_incomplete_or_unknown_usage_is_not_a_measured_prefix(response):
    client, _, _, appended = setup(response)
    assert observed_admission(client, appended)["mode"] != "bound_observed_prefix_plus_appended_utf8_v1"


@pytest.mark.parametrize("change", ["unbound", "digest", "owner_profile", "adapter_profile", "model"])
def test_unbound_or_changed_identity_cannot_use_a_prefix(change):
    client, _, _, appended = setup(bound=change != "unbound")
    if change == "digest":
        client.bind_context_identity(model_identity=identity(digest="b" * 64), owner_profile_sha256="f" * 64)
    elif change == "owner_profile":
        client.bind_context_identity(model_identity=identity(), owner_profile_sha256="e" * 64)
    elif change == "adapter_profile":
        client.seed = 42
    elif change == "model":
        client.model = "different-model"
    view = observed_admission(client, appended)
    assert view["mode"] != "bound_observed_prefix_plus_appended_utf8_v1"
    if change != "unbound":
        assert view["mode"] == "provider_body_utf8_proxy_plus_output_budget_v1"
        assert not view["admitted"]
        assert client._prompt_token_calibrations == client._transferable_prompt_calibrations == []


def test_multimodal_payload_does_not_use_text_prefix():
    client, sent, _, _ = setup()
    payload = deepcopy(sent[0])
    payload["messages"].append({"role": "user", "content": "image", "images": ["encoded-image"]})
    assert client._context_admission(payload)["mode"] != "bound_observed_prefix_plus_appended_utf8_v1"


def test_export_import_transfers_counts_but_not_a_prefix_anchor():
    client, _, _, appended = setup()
    calibration = client.export_context_calibration(model_identity=identity(), owner_profile_sha256="f" * 64)
    target = OllamaChatAgentAdapter(base_url="http://127.0.0.1:11434", model="qwen3-coder:30b",
        context_window=4096, max_output_tokens=128, request_sender=lambda *_: answer())
    assert target.inherit_context_calibration(calibration,
        model_identity=identity(), owner_profile_sha256="f" * 64)
    assert target._prefix_identity_binding is not None and target._prompt_prefix_anchor is None
    assert target._prompt_token_calibrations == list(calibration.observations)
    result = target.complete_turn(appended)
    assert result.provider_diagnostic["context_admission"]["mode"] != "bound_observed_prefix_plus_appended_utf8_v1"
    later = replace(appended, messages=(*appended.messages, AgentMessage("user", "new" * 100)))
    result = target.complete_turn(later)
    assert result.provider_diagnostic["context_admission"]["mode"] == "bound_observed_prefix_plus_appended_utf8_v1"


@pytest.mark.parametrize("operation", ["bind", "export"])
def test_first_identity_binding_discards_preidentity_samples(operation):
    client, _, _, appended = setup(bound=False)
    assert client._prompt_token_calibrations and client._transferable_prompt_calibrations
    if operation == "bind":
        client.bind_context_identity(model_identity=identity(), owner_profile_sha256="f" * 64)
    else:
        exported = client.export_context_calibration(model_identity=identity(), owner_profile_sha256="f" * 64)
        assert not exported.observations
    assert client._prompt_token_calibrations == client._transferable_prompt_calibrations == []
    view = observed_admission(client, appended)
    assert view["mode"] == "provider_body_utf8_proxy_plus_output_budget_v1"
    assert not view["admitted"]


def test_failed_send_clears_the_prefix_instead_of_claiming_a_completed_observation():
    client, _, _, appended = setup()
    def fail(*_args):
        raise AgentTurnProviderError("controlled transport failure")
    client._send = fail
    with pytest.raises(AgentTurnProviderError, match="controlled transport failure"):
        client.complete_turn(appended)
    assert client._prompt_prefix_anchor is None
