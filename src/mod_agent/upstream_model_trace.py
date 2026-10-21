"""Observe upstream provider calls; bodies remain local and never control routing."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from src.core.trace.execution_trace import current_trace
from src.libs.llm.base_llm import ChatResponse
from src.libs.llm.ollama_llm import ollama_cached_input_tokens
from src.libs.llm.transport_observation import observe_transport
from src.mod_agent.upstream_trace import model_input_materials

_NAMES = {"task_intent": "task.intent", "requirement_projection": "requirements.project",
          "requirement_projection_correction": "requirements.project",
          "requirement_findings": "requirements.findings",
          "requirement_findings_correction": "requirements.findings",
          "planning": "planning.select",
          "upstream_evidence_queries": "evidence.query",
          "initial_findings_queries": "evidence.query",
          "patch_upstream_planning": "planning.select", "self_rag_critic": "self_rag.critic"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _local_body(root, observation_id, role, raw):
    if root is None:
        return None
    path = Path(root) / "model_trace_bodies" / (observation_id + "." + role)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _usage(response):
    if isinstance(response, ChatResponse):
        usage = response.usage
        raw = response.raw_response
        # The old Ollama adapter defaults a missing side of usage to zero.
        # Its raw counts let tracing preserve that side as unknown instead.
        if isinstance(raw, Mapping) and ("prompt_eval_count" in raw or "eval_count" in raw):
            usage = {"prompt_tokens": raw.get("prompt_eval_count"), "completion_tokens": raw.get("eval_count")}
            cached = ollama_cached_input_tokens(raw)
            if cached is not None:
                usage["cached_input_tokens"] = cached
        if not isinstance(usage, Mapping):
            return None
        normalized = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
        cached = usage.get("cached_input_tokens")
        if type(cached) is int and cached >= 0:
            normalized["cached_input_tokens"] = cached
        usage = normalized
    else:
        diagnostic = getattr(response, "provider_diagnostic", None)
        usage = diagnostic.get("measured_usage") if isinstance(diagnostic, Mapping) else None
    if (not isinstance(usage, Mapping)
            or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens"))):
        return None
    return dict(usage)


def trace_upstream_model_call(*, role, provider, model, input_payload, request_document,
        body_root, invoke=None, invoke_with_provider_request=None):
    """Same invocation/result/exception, with actual transport-driven generations.

    The request file contains selected application messages/schema/options, not
    an HTTP header capture. Hashes of that file and of the semantic input payload
    are distinct. No new model request is made to produce a decision explanation.
    """
    trace = current_trace()
    if (invoke is None) == (invoke_with_provider_request is None):
        raise ValueError("Exactly one upstream provider invocation must be supplied.")
    if trace is None or role not in _NAMES:
        return invoke() if invoke is not None else invoke_with_provider_request(lambda _value: None)
    try:
        raw_request = _json(request_document)
        attrs = {"model_role": role, "provider": provider, "model": model,
                 "request_sha256": hashlib.sha256(raw_request).hexdigest(),
                 "input_payload_sha256": hashlib.sha256(_json(input_payload)).hexdigest(),
                 "request_bytes": len(raw_request), "usage_available": False, "response_received": False}
        try:
            attrs["consumed_materials"] = model_input_materials(input_payload)
        except Exception:
            attrs["material_capture_status"] = "unavailable"
    except Exception:
        # Serialization for telemetry must not decide whether the provider runs.
        raw_request, attrs = None, {"model_role": role, "model": model, "provider": provider}
    with trace.model_call(attrs, generation_name=_NAMES[role]) as call:
        refs = {}
        def record_local_provider_body(raw):
            call.update(provider_request_sha256=hashlib.sha256(raw).hexdigest(),
                        provider_request_bytes=len(raw),
                        provider_request_capture_boundary="loopback_httpx_request_event")
            refs.pop("provider_request", None)
            try:
                ref = _local_body(body_root, call.observation_id, "provider_request.json", raw)
                if ref is not None:
                    refs["provider_request"] = ref
                call.update(local_detail_refs=dict(refs),
                            provider_request_local_body_status="saved" if ref is not None else "not_configured")
            except Exception:
                call.update(local_detail_refs=dict(refs), provider_request_local_body_status="write_failed")
        def record_provider_request(value):
            if not isinstance(value, Mapping):
                raise ValueError("UPSTREAM_PROVIDER_REQUEST_INVALID")
            body = value.get("body_utf8")
            if not isinstance(body, str):
                raise ValueError("UPSTREAM_PROVIDER_REQUEST_BODY_MISSING")
            raw = body.encode("utf-8")
            ref = _local_body(body_root, call.observation_id, "provider_request.json", raw)
            if ref is not None:
                refs["provider_request"] = ref
            safe = {key: value.get(key) for key in (
                "method", "destination", "content_type", "timeout_seconds",
                "trust_environment", "follow_redirects", "context_admission",
                "model", "output_budget", "context_window", "temperature",
                "messages_sha256", "response_format_sha256")}
            call.update(provider_request_sha256=hashlib.sha256(raw).hexdigest(),
                        provider_request_bytes=len(raw),
                        provider_request_metadata=safe,
                        local_detail_refs=dict(refs))
        try:
            ref = _local_body(body_root, call.observation_id, "request.json", raw_request) if raw_request is not None else None
            if ref is not None:
                refs["request"] = ref
                call.update(local_detail_refs=dict(refs))
        except Exception:
            call.update(local_body_status="request_write_failed")
        with observe_transport(call.transport_started, call.update, call.update,
                               local_request_body_callback=record_local_provider_body):
            try:
                response = (invoke() if invoke is not None
                            else invoke_with_provider_request(record_provider_request))
            except BaseException as error:
                # Keep incomplete local text when the existing Ollama parser
                # explicitly exposes it; do not copy arbitrary error dictionaries.
                diagnostic = getattr(error, "diagnostics", None)
                if isinstance(diagnostic, Mapping) and isinstance(diagnostic.get("message_content"), str):
                    try:
                        ref = _local_body(body_root, call.observation_id, "response.txt", diagnostic["message_content"].encode("utf-8"))
                        if ref is not None:
                            refs["response"] = ref
                            call.update(local_detail_refs=dict(refs), response_origin="incomplete_provider_response")
                    except Exception:
                        pass
                raise
        # Only telemetry code lives in this try: never retry invoke() on a
        # formatting or persistence error after the provider has returned.
        try:
            native = getattr(response, "raw_response", None)
            if native is None:
                call.update(provider_response_local_body_status="not_available")
            else:
                try:
                    native_body = _json(native)
                    ref = _local_body(body_root, call.observation_id, "provider_response.json", native_body)
                    if ref is not None:
                        refs["provider_response"] = ref
                    call.update(provider_response_local_body_status="saved" if ref is not None else "not_configured",
                                provider_response_sha256=hashlib.sha256(native_body).hexdigest(),
                                provider_response_bytes=len(native_body),
                                provider_response_capture_boundary="adapter_raw_response_json_not_wire_bytes")
                except (TypeError, ValueError):
                    call.update(provider_response_local_body_status="not_serializable")
                except Exception:
                    call.update(provider_response_local_body_status="write_failed")
            content = getattr(response, "content", None)
            if isinstance(content, str):
                raw = content.encode("utf-8")
                call.update(output_sha256=hashlib.sha256(raw).hexdigest(), response_sha256=hashlib.sha256(raw).hexdigest())
                try:
                    ref = _local_body(body_root, call.observation_id, "response.txt", raw)
                    if ref is not None:
                        refs["response"] = ref
                except Exception:
                    call.update(local_body_status="response_write_failed")
            tool_calls = getattr(response, "tool_calls", ())
            if tool_calls:
                try:
                    ref = _local_body(body_root, call.observation_id, "response_tools.json", _json([asdict(v) for v in tool_calls]))
                    if ref is not None:
                        refs["response_tools"] = ref
                except Exception:
                    call.update(local_body_status="response_tools_write_failed")
            usage = _usage(response)
            call.record_response(usage=usage, usage_available=usage is not None,
                                 response_origin="live_provider_response", local_detail_refs=dict(refs))
        except Exception:
            call.update(local_body_status="response_metadata_unavailable")
        return response
