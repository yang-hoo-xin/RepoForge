"""Import measured standalone-extraction context observations through the existing adapter.

为什么要有它：实际请求和原生计数可校准已有保守估计，但手填数字、不同模型或不完整响应不能放宽上下文准入。
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from src.libs.llm.agent_turn import OllamaChatAgentAdapter, OllamaContextCalibration
from src.libs.llm.ollama_llm import ollama_provider_body_utf8_policy_projection
from src.mod_agent.model_admission import (
    assert_ollama_model_identity_stable,
    validate_ollama_context_window,
    validate_ollama_model_identity,
)
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_v2_state import _no_links


def require(value, code):
    if not value:
        raise ValueError("LEARNING_CALIBRATION_" + code)


def _read(reference, root, *, name=None, raw=False):
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "BOUND_REFERENCE_REQUIRED")
    path = Path(reference["path"])
    require(path.is_absolute(), "ABSOLUTE_REFERENCE_REQUIRED")
    _no_links(path)
    path = path.resolve(strict=True)
    require(path.is_relative_to(root) and (name is None or path == root / name), "SOURCE_OUTSIDE_ATTEMPT")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == reference["sha256"], "SOURCE_BYTES_CHANGED")
    return content if raw else json.loads(content)


def _application_matches_wire(application, payload, profile):
    require(not application.get("tools") and not application.get("tool_results")
        and not application.get("historical_tool_exchanges") and application.get("provider_state") is None
        and application.get("tool_choice") == "none" and application.get("parallel_tool_calls") is False,
        "EXTRACTION_REQUEST_REQUIRED")
    require(payload.get("model") == profile["model"] and payload.get("stream") is False and payload.get("think") is False
        and set(payload) == {"model", "messages", "stream", "think", "options", "format"}
        and canonical_hash(payload["options"]) == canonical_hash({"num_ctx": profile["context_window"], "num_predict": profile["output_budget"],
            "seed": 0, "temperature": profile["temperature"]})
        and application.get("temperature") == profile["temperature"]
        and isinstance(application.get("response_format"), dict)
        and payload["format"] == application["response_format"]["schema"], "REQUEST_PROFILE_CHANGED")
    expected, actual = application["messages"], payload["messages"]
    require(isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual), "REQUEST_MESSAGES_CHANGED")
    for before, wire in zip(expected, actual, strict=True):
        require(isinstance(wire, dict) and set(wire) == {"role", "content"}
            and wire["role"] == before["role"], "REQUEST_MESSAGES_CHANGED")
        value = wire["content"] if isinstance(before["content"], str) else json.loads(wire["content"])
        require(canonical_hash(value) == canonical_hash(before["content"]), "REQUEST_CONTENT_CHANGED")


def load_calibration_sources(sources, profile, expected_digest):
    """Validate every selected attempt and keep every measured observation in order.

    为什么要有它：只读取显式选择的根内绑定原件，不能扫描其他历史、挑选较低比例或跟随请求正文中的引用。
    """
    if sources is None:
        return None
    require(isinstance(sources, list) and bool(sources), "NONEMPTY_SOURCE_LIST_REQUIRED")
    configuration = profile.to_dict()
    selected, observations, seen_roots, seen_responses = [], [], set(), set()
    common_identity = None
    for source in sources:
        require(isinstance(source, dict) and set(source) == {"attempt_root", "result_ref", "started_ref",
            "identity_before_ref", "identity_after_ref"}, "SOURCE_FIELDS_INVALID")
        root = Path(source["attempt_root"])
        require(root.is_absolute(), "ABSOLUTE_ATTEMPT_ROOT_REQUIRED")
        _no_links(root)
        root = root.resolve(strict=True)
        require(root.is_dir() and str(root) not in seen_roots, "DUPLICATE_OR_INVALID_ATTEMPT")
        seen_roots.add(str(root))
        result = _read(source["result_ref"], root, name="result.json")
        started = _read(source["started_ref"], root, name="started.json")
        before = _read(source["identity_before_ref"], root, name="identity-before.json")
        after = _read(source["identity_after_ref"], root, name="identity-after.json")
        require(started.get("profile") == configuration, "CONFIGURED_PROFILE_CHANGED")
        require(isinstance(started.get("prepared_ref"), dict)
            and result.get("prepared_ref") == started["prepared_ref"], "ATTEMPT_IDENTITY_CHANGED")
        for identity in (before, after):
            validate_ollama_model_identity(identity, expected_base_url=profile.base_url, expected_model=profile.model)
            validate_ollama_context_window(identity, requested_context_window=profile.context_window)
            require(identity.get("digest") == expected_digest, "MODEL_DIGEST_CHANGED")
        assert_ollama_model_identity_stable(expected=before, observed=after)
        if common_identity is not None:
            assert_ollama_model_identity_stable(expected=common_identity, observed=before)
        else:
            common_identity = deepcopy(before)
        events, receipts = result.get("provider_events"), result.get("call_receipts")
        require(result.get("model_calls_reserved") == 1 and result.get("unknown_usage_calls") == 0
            and result.get("budget", {}).get("consumed") == 1
            and isinstance(events, list) and len(events) == 1
            and isinstance(receipts, list) and len(receipts) == 1
            and receipts[0].get("model_calls") == 1, "COMPLETE_SINGLE_ATTEMPT_REQUIRED")
        event = events[0]
        require(event.get("provider_request_capture_boundary") == "loopback_httpx_request_event"
            and event.get("provider_request_ref_scope") == "actual_loopback_httpx_request_content"
            and event.get("provider_request_local_body_status") == "saved", "ACTUAL_HTTP_BODY_REQUIRED")
        body = _read(event["provider_request_ref"], root, raw=True)
        body_hash = hashlib.sha256(body).hexdigest()
        require(type(event.get("provider_request_bytes")) is int and event["provider_request_bytes"] == len(body)
            and event.get("provider_request_sha256") == body_hash, "REQUEST_BODY_IDENTITY_CHANGED")
        payload = json.loads(body)
        application = _read(event["request_ref"], root)
        _application_matches_wire(application, payload, configuration)
        response = _read(event["response_ref"], root)
        response_identity = (str(Path(event["response_ref"]["path"]).resolve()), event["response_ref"]["sha256"])
        require(response_identity not in seen_responses, "DUPLICATE_RESPONSE")
        seen_responses.add(response_identity)
        native, diagnostic, usage = response.get("raw_response"), response.get("provider_diagnostic"), response.get("usage")
        require(isinstance(native, dict) and isinstance(diagnostic, dict) and isinstance(usage, dict), "NATIVE_RESPONSE_REQUIRED")
        # 为什么要有它：显式截断后的prompt计数不能与完整HTTP正文配对；字段缺省不被补写为未截断的证明。
        require(response.get("provider") == "ollama_chat_local" and response.get("model") == profile.model
            and native.get("model") == profile.model and native.get("done") is True
            and native.get("truncated") is not True
            and native.get("done_reason") == response.get("stop_reason") == "stop", "RESPONSE_NOT_COMPLETE")
        require(isinstance(native.get("message"), dict) and native["message"].get("role") == "assistant"
            and native["message"].get("content") == response.get("content"), "NATIVE_RESPONSE_CONTENT_CHANGED")
        prompt, output = native.get("prompt_eval_count"), native.get("eval_count")
        require(type(prompt) is int and prompt > 0 and type(output) is int and output >= 0
            and prompt + output <= profile.context_window and output <= profile.output_budget
            and usage.get("input_tokens") == prompt and usage.get("output_tokens") == output,
            "NATIVE_USAGE_CHANGED")
        if "prompt_eval_cached_count" in native:
            require(type(native["prompt_eval_cached_count"]) is int and 0 <= native["prompt_eval_cached_count"] <= prompt
                and usage.get("cached_input_tokens") == native["prompt_eval_cached_count"], "NATIVE_CACHE_USAGE_CHANGED")
        receipt_usage = {**usage, "total_tokens": prompt + output}
        require(receipts[0].get("usage") == receipt_usage
            and diagnostic.get("usage") == receipt_usage
            and diagnostic.get("measured_usage", {}).get("input_tokens") == prompt
            and diagnostic.get("measured_usage", {}).get("output_tokens") == output, "RECEIPT_USAGE_CHANGED")
        require(diagnostic.get("api_type") == "ollama_native_chat"
            and diagnostic.get("requested_model") == diagnostic.get("returned_model") == profile.model
            and diagnostic.get("configured_context_window") == profile.context_window
            and diagnostic.get("configured_max_output_tokens") == profile.output_budget
            and diagnostic.get("configured_seed") == 0
            and diagnostic.get("configured_temperature") == configuration["temperature"]
            and diagnostic.get("finish_reason") == "stop", "NATIVE_PROFILE_CHANGED")
        identity = diagnostic.get("request_identity", {})
        require(identity == {"hash_basis": "httpx_json_body_bytes", "provider_body_sha256": body_hash,
            "provider_body_utf8_bytes": len(body)}, "NATIVE_REQUEST_IDENTITY_CHANGED")
        policy_bytes = len(ollama_provider_body_utf8_policy_projection(payload))
        admission = diagnostic.get("context_admission", {})
        require(admission.get("admitted") is True and admission.get("configured_num_ctx") == profile.context_window
            and admission.get("num_predict") == profile.output_budget
            and admission.get("provider_body_utf8_bytes") == policy_bytes, "SOURCE_ADMISSION_CHANGED")
        observation = [policy_bytes, prompt]
        observations.append(observation)
        selected.append({"source": deepcopy(source), "request_ref": deepcopy(event["request_ref"]),
            "provider_request_ref": deepcopy(event["provider_request_ref"]), "response_ref": deepcopy(event["response_ref"]),
            "actual_body_bytes": len(body), "observation": observation, "semantic_status": result.get("status")})
    return {"schema_version": "verified_learning_context_calibration_v1", "model_identity": common_identity,
        "profile_sha256": canonical_hash(configuration), "observations": observations, "sources": selected,
        "policy": "existing_adapter_max_observed_ratio_with_2x_margin_and_body_byte_half_floor",
        "scope": "Only complete native measurements from every explicitly selected attempt; semantic strategy outcome is not a filter."}


def inherit_calibration(client, calibration, profile, model_identity):
    if calibration is None:
        return
    require(isinstance(client, OllamaChatAgentAdapter), "EXISTING_ADAPTER_REQUIRED")
    assert_ollama_model_identity_stable(expected=calibration["model_identity"], observed=model_identity)
    require(calibration["profile_sha256"] == canonical_hash(profile.to_dict())
        and client.base_url == profile.base_url and client.model == profile.model
        and client.context_window == profile.context_window and client.max_output_tokens == profile.output_budget
        and client.seed == 0 and client.request_timeout_seconds == profile.timeout_seconds, "CLIENT_PROFILE_CHANGED")
    binding = client._calibration_binding(model_identity, calibration["profile_sha256"])
    require(client.inherit_context_calibration(OllamaContextCalibration(*binding,
        tuple(tuple(pair) for pair in calibration["observations"])), model_identity=model_identity,
        owner_profile_sha256=calibration["profile_sha256"]), "CALIBRATION_NOT_INHERITED")
