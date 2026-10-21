"""Provider-neutral structured Agent turns and bounded provider adapters."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from src.libs.llm.ollama_llm import (
    build_ollama_context_admission,
    nonnegative_token_count,
    ollama_cached_input_tokens,
)
from src.libs.llm.transport_observation import (
    transport_local_request_body,
    transport_request_prepared,
    transport_response_received,
    transport_started,
)


@dataclass(frozen=True)
class AgentMessage:
    """One provider-neutral instruction or context message."""

    role: Literal["system", "user", "assistant"]
    content: object


@dataclass(frozen=True)
class AgentToolDefinition:
    """One callable tool described by an exact JSON Schema."""

    name: str
    description: str
    parameters: dict[str, object]
    strict: bool = True


@dataclass(frozen=True)
class AgentResponseFormat:
    """Structured final-output contract."""

    name: str
    description: str
    schema: dict[str, object]
    strict: bool = True


@dataclass(frozen=True)
class AgentToolResult:
    """A deterministic local result linked to one provider tool call."""

    call_id: str
    name: str
    output: object


@dataclass(frozen=True)
class AgentToolExchange:
    """A completed native tool exchange retained at its original message boundary."""

    tool_results: tuple[AgentToolResult, ...]
    provider_state: object
    insert_index: int


@dataclass(frozen=True)
class AgentTurnRequest:
    """One bounded structured turn, independent of provider wire format."""

    messages: tuple[AgentMessage, ...]
    tools: tuple[AgentToolDefinition, ...] = ()
    tool_choice: Literal["auto", "required", "none"] = "auto"
    parallel_tool_calls: bool = False
    response_format: AgentResponseFormat | None = None
    tool_results: tuple[AgentToolResult, ...] = ()
    tool_result_insert_index: int | None = None
    provider_state: object | None = None
    temperature: float = 0.0
    historical_tool_exchanges: tuple[AgentToolExchange, ...] = ()


@dataclass(frozen=True)
class AgentToolCall:
    """One normalized provider-requested tool call."""

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class AgentUsage:
    """Normalized totals; a missing cache measurement is explicitly unknown."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        usage = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cached_input_tokens is not None:
            usage["cached_input_tokens"] = self.cached_input_tokens
        return usage


@dataclass(frozen=True)
class AgentTurnResponse:
    """Normalized Agent response; raw provider data is diagnostic-only."""

    provider: str
    model: str
    content: str | None
    tool_calls: tuple[AgentToolCall, ...] = ()
    stop_reason: str | None = None
    request_id: str | None = None
    usage: AgentUsage = field(default_factory=AgentUsage)
    provider_state: object | None = None
    provider_diagnostic: dict[str, object] | None = None
    raw_response: object | None = None


class AgentTurnClient(Protocol):
    """Execute one provider-neutral structured Agent turn."""

    provider: str
    model: str

    def complete_turn(self, request: AgentTurnRequest) -> AgentTurnResponse:
        ...


JsonRequestSender = Callable[[str, dict[str, object], dict[str, str]], dict[str, Any]]


class AgentTurnProviderError(RuntimeError):
    """Raised when a provider rejects or returns an invalid Agent turn."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = dict(diagnostic or {})


class _HttpJsonSender:
    def __init__(self, *, api_key: str, timeout_seconds: float = 60.0) -> None:
        if not api_key:
            raise ValueError("Agent provider API key is required.")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        import httpx

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                transport_started()
                response = client.post(url, json=payload, headers=headers)
                transport_response_received(http_status=response.status_code, response_complete=True)
        except httpx.TimeoutException as exc:
            raise AgentTurnProviderError(
                f"Agent provider request timed out after {self.timeout_seconds:g} seconds."
            ) from exc
        except httpx.RequestError as exc:
            raise AgentTurnProviderError(
                f"Agent provider connection failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code != 200:
            try:
                error = response.json()
            except Exception:
                error = response.text
            raise AgentTurnProviderError(
                f"Agent provider HTTP {response.status_code}: {error}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise AgentTurnProviderError("Agent provider response must be an object.")
        return data


def validate_loopback_http_base_url(base_url: str) -> str:
    """Return one normalized literal-loopback HTTP origin or raise."""

    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("LOCAL_AGENT_ENDPOINT_INVALID") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("LOCAL_AGENT_ENDPOINT_MUST_BE_LITERAL_LOOPBACK_HTTP_ORIGIN")
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    return f"http://{host}:{port}"


class _LoopbackHttpJsonSender:
    def __init__(self, *, timeout_seconds: float = 180.0) -> None:
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        import httpx

        endpoint = urlsplit(url)
        if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "::1"}:
            raise AgentTurnProviderError("LOCAL_AGENT_REQUEST_LEFT_LOOPBACK")
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                trust_env=False,
                event_hooks={"request": [transport_local_request_body]},
            ) as client:
                transport_started()
                response = client.post(url, json=payload, headers=headers)
                transport_response_received(http_status=response.status_code, response_complete=True)
        except httpx.TimeoutException as exc:
            raise AgentTurnProviderError(
                f"Local Agent request timed out after {self.timeout_seconds:g} seconds."
            ) from exc
        except httpx.RequestError as exc:
            raise AgentTurnProviderError(
                f"Local Agent connection failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code != 200:
            try:
                error = response.json()
            except Exception:
                error = response.text
            raise AgentTurnProviderError(
                f"Local Agent HTTP {response.status_code}: {error}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise AgentTurnProviderError("Local Agent response must be an object.")
        return data


def _httpx_json_body_identity(url: str, payload: dict[str, object]) -> dict[str, object]:
    """Use the sender's JSON encoder without changing its dispatched payload."""
    import httpx

    body = httpx.Request("POST", url, json=payload).content
    return {
        "provider_body_sha256": sha256(body).hexdigest(),
        "provider_body_utf8_bytes": len(body),
        "hash_basis": "httpx_json_body_bytes",
    }


@dataclass(frozen=True)
class OllamaContextCalibration:
    """Detached observations only: no client, sender, recorder or replay state."""

    model_identity: tuple[str, str, str, str]
    adapter_profile: tuple[object, ...]
    owner_profile_sha256: str
    observations: tuple[tuple[int, int], ...]


class OllamaChatAgentAdapter:
    """Ollama native Chat adapter restricted to a literal loopback origin."""

    provider = "ollama_chat_local"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        max_output_tokens: int,
        context_window: int | None = None,
        seed: int = 0,
        request_timeout_seconds: float = 180.0,
        request_sender: JsonRequestSender | None = None,
    ) -> None:
        if (
            not model.strip()
            or max_output_tokens <= 0
            or (
                context_window is not None
                and (
                    isinstance(context_window, bool)
                    or not isinstance(context_window, int)
                    or context_window <= max_output_tokens
                )
            )
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or request_timeout_seconds <= 0
        ):
            raise ValueError("Ollama Agent adapter configuration is invalid.")
        self.base_url = validate_loopback_http_base_url(base_url)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.context_window = context_window
        self.seed = seed
        self.request_timeout_seconds = request_timeout_seconds
        self._send = request_sender or _LoopbackHttpJsonSender(
            timeout_seconds=request_timeout_seconds
        )
        self._prompt_token_calibrations: list[tuple[int, int]] = []
        self._transferable_prompt_calibrations: list[tuple[int, int]] = []
        self._prefix_identity_binding = None
        self._prompt_prefix_anchor = None

    def _calibration_binding(self, model_identity, owner_profile_sha256):
        identity = tuple(model_identity.get(key) for key in ("base_url", "name", "model", "digest"))
        if (identity[:3] != (self.base_url, self.model, self.model)
                or not isinstance(identity[3], str) or len(identity[3]) != 64
                or any(c not in "0123456789abcdef" for c in identity[3])
                or not isinstance(owner_profile_sha256, str) or len(owner_profile_sha256) != 64
                or any(c not in "0123456789abcdef" for c in owner_profile_sha256)):
            raise AgentTurnProviderError("OLLAMA_CONTEXT_CALIBRATION_IDENTITY_INVALID")
        profile = (self.provider, self.base_url, self.model, self.context_window,
                   self.max_output_tokens, self.seed, self.request_timeout_seconds)
        return identity, profile, owner_profile_sha256

    def bind_context_identity(self, *, model_identity, owner_profile_sha256):
        """Use the owner's already verified model manifest and profile, without a new request."""
        binding = self._calibration_binding(model_identity, owner_profile_sha256)
        if binding != self._prefix_identity_binding:
            self._prompt_prefix_anchor = None
            self._prompt_token_calibrations.clear()
            self._transferable_prompt_calibrations.clear()
        self._prefix_identity_binding = binding

    def _current_prefix_binding(self):
        binding = self._prefix_identity_binding
        profile = (self.provider, self.base_url, self.model, self.context_window,
                   self.max_output_tokens, self.seed, self.request_timeout_seconds)
        if binding is None:
            self._prompt_prefix_anchor = None
            return None
        if binding[1] != profile:
            self._prompt_prefix_anchor = None
            self._prompt_token_calibrations.clear()
            self._transferable_prompt_calibrations.clear()
            self._prefix_identity_binding = None
            return None
        return binding

    def export_context_calibration(self, *, model_identity, owner_profile_sha256):
        """Bind previously observed native counts to the admitted identity/profile."""
        binding = self._calibration_binding(model_identity, owner_profile_sha256)
        # Scope: exported numeric observations obey the same identity boundary as prefix anchors.
        # 为什么要有它：首次绑定或模型/配置变化不能把旧未知来源计数追认为当前身份的观测。
        self.bind_context_identity(model_identity=model_identity, owner_profile_sha256=owner_profile_sha256)
        return OllamaContextCalibration(*binding, tuple(self._transferable_prompt_calibrations))

    def inherit_context_calibration(self, calibration, *, model_identity, owner_profile_sha256):
        # 为什么要有它：同 Plan 可续用实测 tokenizer 比例，但模型/配置漂移不能借用旧样本绕过原上下文门禁。
        if type(calibration) is not OllamaContextCalibration or self._prompt_token_calibrations:
            raise AgentTurnProviderError("OLLAMA_CONTEXT_CALIBRATION_IMPORT_INVALID")
        if self._calibration_binding(model_identity, owner_profile_sha256) != (
                calibration.model_identity, calibration.adapter_profile, calibration.owner_profile_sha256):
            return False
        if any(len(row) != 2 or any(type(value) is not int or value <= 0 for value in row)
               for row in calibration.observations):
            raise AgentTurnProviderError("OLLAMA_CONTEXT_CALIBRATION_OBSERVATIONS_INVALID")
        self.bind_context_identity(model_identity=model_identity, owner_profile_sha256=owner_profile_sha256)
        self._prompt_token_calibrations = list(calibration.observations)
        self._transferable_prompt_calibrations = list(calibration.observations)
        return True

    def complete_turn(self, request: AgentTurnRequest) -> AgentTurnResponse:
        messages = [_chat_message(item) for item in request.messages]
        required_single_tool = (
            request.tool_choice == "required"
            and len(request.tools) == 1
            and request.response_format is None
        )
        for recorded_exchange in _tool_exchanges(request, len(messages)):
            state = recorded_exchange.provider_state
            if not isinstance(state, dict) or not isinstance(state.get("tool_calls"), list):
                raise AgentTurnProviderError("OLLAMA_TOOL_RESULT_STATE_MISSING")
            exchange: list[dict[str, object]] = [
                {
                    "role": "assistant",
                    "content": state.get("content") or "",
                    "tool_calls": state["tool_calls"],
                }
            ]
            for result in recorded_exchange.tool_results:
                exchange.append(
                    {
                        "role": "tool",
                        "tool_name": result.name,
                        "content": _json_text(result.output),
                    }
                )
            insert_at = recorded_exchange.insert_index
            messages[insert_at:insert_at] = exchange
        options: dict[str, object] = {
            "temperature": request.temperature,
            "num_predict": self.max_output_tokens,
            "seed": self.seed,
        }
        if self.context_window is not None:
            options["num_ctx"] = self.context_window
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }
        if request.tools:
            payload["tools"] = [_chat_tool(tool) for tool in request.tools]
        if request.response_format is not None:
            payload["format"] = request.response_format.schema
        elif required_single_tool:
            payload["format"] = request.tools[0].parameters
        request_url = f"{self.base_url}/api/chat"
        request_identity = _httpx_json_body_identity(request_url, payload)
        context_admission = self._context_admission(payload)
        transport_request_prepared(request_identity=request_identity, context_admission=context_admission)
        if (
            self.context_window is not None
            and context_admission["admitted"] is not True
        ):
            raise AgentTurnProviderError(
                "Ollama Agent request exceeds the configured pre-call context budget.",
                diagnostic={
                    "failure_boundary": "provider_request_gate",
                    "failure_classification": (
                        "provider_request_context_budget_exceeded"
                    ),
                    "context_admission": context_admission,
                    "request_identity": request_identity,
                },
            )
        request_started = time.perf_counter()
        anchor_binding = self._current_prefix_binding()
        anchor_payload = deepcopy(payload) if anchor_binding is not None else None
        try:
            data = self._send(request_url, payload, {"Content-Type": "application/json"})
        except AgentTurnProviderError as exc:
            self._prompt_prefix_anchor = None
            raise AgentTurnProviderError(
                str(exc), diagnostic={**exc.diagnostic, "request_identity": request_identity},
            ) from exc
        except Exception:
            self._prompt_prefix_anchor = None
            raise
        client_elapsed_seconds = time.perf_counter() - request_started
        self._prompt_prefix_anchor = None
        prompt_eval_count = _int_value(data.get("prompt_eval_count"))
        if prompt_eval_count > 0:
            self._prompt_token_calibrations.append(
                (int(context_admission["provider_body_utf8_bytes"]), prompt_eval_count))
        message = data.get("message")
        if not isinstance(message, dict):
            raise AgentTurnProviderError(
                "OLLAMA_CHAT_RESPONSE_INVALID", diagnostic={"request_identity": request_identity}
            )
        stop_reason = "incomplete" if data.get("done") is False else str(data.get("done_reason") or "") or None
        stopped = stop_reason in {"length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal"}
        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise AgentTurnProviderError(
                "OLLAMA_CHAT_TOOL_CALLS_INVALID", diagnostic={"request_identity": request_identity}
            )
        try:
            if not stopped and not raw_tool_calls and required_single_tool:
                raw_tool_calls = [
                    {
                        "function": {
                            "name": request.tools[0].name,
                            "arguments": _parse_arguments(message.get("content")),
                        }
                    }
                ]
            tool_calls = () if stopped else tuple(_parse_ollama_tool_call(item) for item in raw_tool_calls)
        except AgentTurnProviderError as exc:
            raise AgentTurnProviderError(
                str(exc), diagnostic={**exc.diagnostic, "request_identity": request_identity},
            ) from exc
        # 为什么要有它：只有完整同名模型响应中的真实正整数计数才形成可跨 worker 携带的校准，未知/畸形计数不填零冒充观测。
        prompt_eval_count = data.get("prompt_eval_count")
        if (data.get("model") == self.model and data.get("done") is True
                and type(prompt_eval_count) is int and prompt_eval_count > 0):
            self._transferable_prompt_calibrations.append(
                (int(context_admission["provider_body_utf8_bytes"]), prompt_eval_count))
            output_tokens = data.get("eval_count")
            if (not stopped and type(output_tokens) is int and output_tokens >= 0
                    and self.context_window is not None
                    and prompt_eval_count + output_tokens <= self.context_window
                    and data.get("truncated") is not True
                    and anchor_binding is not None and anchor_binding == self._current_prefix_binding()
                    and anchor_payload == payload and self._text_prefix_payload(anchor_payload)):
                self._prompt_prefix_anchor = {
                    "binding": anchor_binding, "payload": anchor_payload,
                    "prompt_tokens": prompt_eval_count,
                    "request_sha256": request_identity["provider_body_sha256"],
                }
        return AgentTurnResponse(
            provider=self.provider,
            model=str(data.get("model") or self.model),
            content=str(message.get("content") or "") or None,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=AgentUsage(
                input_tokens=_int_value(data.get("prompt_eval_count")),
                output_tokens=_int_value(data.get("eval_count")),
                cached_input_tokens=ollama_cached_input_tokens(data),
            ),
            provider_state={
                "content": message.get("content"),
                "tool_calls": raw_tool_calls,
            },
            provider_diagnostic=_ollama_provider_diagnostic(
                data=data,
                requested_model=self.model,
                max_output_tokens=self.max_output_tokens,
                context_window=self.context_window,
                temperature=request.temperature,
                seed=self.seed,
                client_elapsed_seconds=client_elapsed_seconds,
                tool_calls=tool_calls,
                context_admission=context_admission,
                request_identity=request_identity,
            ),
            raw_response=data,
        )

    def _context_admission(
        self,
        payload: Mapping[str, object],
    ) -> dict[str, Any]:
        conservative = build_ollama_context_admission(
            payload=payload,
            configured_num_ctx=self.context_window,
            num_predict=self.max_output_tokens,
        )
        if (
            conservative["admitted"] is True
            or self.context_window is None
        ):
            return conservative
        prefix = self._prefix_context_admission(payload, conservative)
        if prefix is not None:
            return prefix
        if not self._prompt_token_calibrations:
            return conservative
        body_bytes = int(conservative["provider_body_utf8_bytes"])
        observed_upper_ratio = max(
            prompt_tokens / prior_body_bytes
            for prior_body_bytes, prompt_tokens in self._prompt_token_calibrations
            if prior_body_bytes > 0
        )
        estimated_prompt_tokens = max(
            math.ceil(body_bytes * observed_upper_ratio * 2.0),
            math.ceil(body_bytes / 2),
        )
        estimated_required_context = (
            estimated_prompt_tokens + self.max_output_tokens
        )
        return {
            "mode": "same_model_observed_prompt_ratio_upper_bound_v1",
            "exact_prompt_tokens_available_pre_call": False,
            "provider_body_utf8_bytes": body_bytes,
            "configured_num_ctx": self.context_window,
            "num_predict": self.max_output_tokens,
            "observed_calibration_count": len(
                self._prompt_token_calibrations
            ),
            "observed_max_prompt_tokens_per_body_byte": (
                observed_upper_ratio
            ),
            "safety_multiplier": 2.0,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "estimated_required_context": estimated_required_context,
            "conservative_fallback_required_context": conservative[
                "conservative_required_context"
            ],
            "admitted": estimated_required_context <= self.context_window,
        }

    @staticmethod
    def _text_prefix_payload(payload):
        if not isinstance(payload, Mapping) or not isinstance(payload.get("messages"), list):
            return False
        return bool(payload["messages"]) and all(
            isinstance(message, dict)
            and set(message) <= {"role", "content", "tool_name", "tool_calls"}
            and message.get("role") in {"system", "user", "assistant", "tool"}
            and isinstance(message.get("content"), str)
            for message in payload["messages"]
        )

    def _prefix_context_admission(self, payload, conservative):
        binding = self._current_prefix_binding()
        anchor = self._prompt_prefix_anchor
        if (binding is None or anchor is None or anchor["binding"] != binding
                or not self._text_prefix_payload(payload)):
            return None
        previous = anchor["payload"]
        def encode(item):
            return json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        try:
            if encode({key: value for key, value in previous.items() if key != "messages"}) != encode(
                    {key: value for key, value in payload.items() if key != "messages"}):
                return None
            count = len(previous["messages"])
            if len(payload["messages"]) <= count or any(
                    encode(before) != encode(after)
                    for before, after in zip(previous["messages"], payload["messages"])):
                return None
            appended = payload["messages"][count:]
            added_bytes = len(encode(appended))
        except (TypeError, ValueError):
            return None
        # Scope: text-only append estimates under a verified, unchanged model/profile.
        # 为什么要有它：已实测的相同前缀不用再次整体乘倍数；新增内容仍按UTF8计入，并为聊天模板/拼接边界保守留量。
        # This is a conservative heuristic, not a tokenizer measurement or a universal template proof.
        per_message_margin = 256
        boundary_margin = 256
        template_margin = per_message_margin * len(appended) + boundary_margin
        estimated_prompt = anchor["prompt_tokens"] + added_bytes + template_margin
        required = estimated_prompt + self.max_output_tokens
        return {
            "mode": "bound_observed_prefix_plus_appended_utf8_v1",
            "exact_prompt_tokens_available_pre_call": False,
            "provider_body_utf8_bytes": conservative["provider_body_utf8_bytes"],
            "configured_num_ctx": self.context_window, "num_predict": self.max_output_tokens,
            "anchor_request_sha256": anchor["request_sha256"],
            "anchor_prompt_tokens": anchor["prompt_tokens"],
            "anchor_message_count": count, "appended_message_count": len(appended),
            "appended_messages_utf8_bytes": added_bytes,
            "template_margin_tokens_per_message": per_message_margin,
            "boundary_margin_tokens": boundary_margin,
            "template_margin_tokens": template_margin,
            "model_manifest_digest": binding[0][3],
            "owner_profile_sha256": binding[2],
            "estimated_prompt_tokens": estimated_prompt, "estimated_required_context": required,
            "conservative_fallback_required_context": conservative["conservative_required_context"],
            "admitted": required <= self.context_window,
        }


class QwenChatAgentAdapter:
    """Qwen OpenAI-compatible Chat Completions structured-turn adapter."""

    provider = "qwen_chat"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_completion_tokens: int,
        request_sender: JsonRequestSender | None = None,
    ) -> None:
        if not base_url or not model or max_completion_tokens <= 0:
            raise ValueError("Qwen Agent adapter configuration is invalid.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self._send = request_sender or _HttpJsonSender(api_key=api_key)

    def complete_turn(self, request: AgentTurnRequest) -> AgentTurnResponse:
        messages = [_chat_message(item) for item in request.messages]
        for recorded_exchange in _tool_exchanges(request, len(messages)):
            state = recorded_exchange.provider_state
            if not isinstance(state, dict) or not isinstance(state.get("tool_calls"), list):
                raise AgentTurnProviderError("QWEN_TOOL_RESULT_STATE_MISSING")
            exchange: list[dict[str, object]] = [
                {
                    "role": "assistant",
                    "content": state.get("content"),
                    "tool_calls": state["tool_calls"],
                }
            ]
            for result in recorded_exchange.tool_results:
                exchange.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "name": result.name,
                        "content": _json_text(result.output),
                    }
                )
            insert_at = recorded_exchange.insert_index
            messages[insert_at:insert_at] = exchange
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": request.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "enable_thinking": False,
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        if request.tools:
            payload["tools"] = [_chat_tool(tool) for tool in request.tools]
            payload["tool_choice"] = request.tool_choice
        elif request.tool_choice == "none":
            payload["tool_choice"] = "none"
        if request.response_format is not None:
            # Qwen's documented JSON mode guarantees an object; deterministic
            # platform validation still enforces the exact local schema.
            payload["response_format"] = {"type": "json_object"}
        data = self._send(
            f"{self.base_url}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentTurnProviderError("QWEN_CHAT_RESPONSE_INVALID") from exc
        usage_data = data.get("usage") or {}
        usage = AgentUsage(
            input_tokens=_int_value(usage_data.get("prompt_tokens")),
            output_tokens=_int_value(usage_data.get("completion_tokens")),
            reasoning_tokens=_int_value(
                (usage_data.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
            ),
            cached_input_tokens=_cached_detail_count(usage_data.get("prompt_tokens_details")),
        )
        stop_reason = "refusal" if message.get("refusal") else str(choice.get("finish_reason") or "") or None
        stopped = stop_reason in {"length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal"}
        raw_tool_calls = message.get("tool_calls") or []
        parsed_tool_calls: list[AgentToolCall] = []
        try:
            # A stopped response may contain truncated tool arguments. Retain
            # them in raw/state data, but never parse them as executable actions.
            for item in (() if stopped else raw_tool_calls):
                parsed_tool_calls.append(_parse_chat_tool_call(item))
        except AgentTurnProviderError as exc:
            raise AgentTurnProviderError(
                str(exc),
                diagnostic=_qwen_chat_response_diagnostic(
                    data=data,
                    raw_tool_calls=raw_tool_calls,
                    tool_calls=tuple(parsed_tool_calls),
                    usage=usage,
                ),
            ) from exc
        tool_calls = tuple(parsed_tool_calls)
        provider_diagnostic = _qwen_chat_response_diagnostic(
            data=data,
            raw_tool_calls=raw_tool_calls,
            tool_calls=tool_calls,
            usage=usage,
        )
        return AgentTurnResponse(
            provider=self.provider,
            model=str(data.get("model") or self.model),
            content=message.get("content"),
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            request_id=str(data.get("id") or "") or None,
            usage=usage,
            provider_state={
                "content": message.get("content"),
                "tool_calls": raw_tool_calls,
            },
            provider_diagnostic=provider_diagnostic,
            raw_response=data,
        )


class OpenAIResponsesAgentAdapter:
    """OpenAI Responses API structured-turn adapter."""

    provider = "openai_responses"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str,
        max_output_tokens: int,
        request_sender: JsonRequestSender | None = None,
    ) -> None:
        if not base_url or not model or max_output_tokens <= 0:
            raise ValueError("OpenAI Responses Agent adapter configuration is invalid.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._send = request_sender or _HttpJsonSender(api_key=api_key)

    def complete_turn(self, request: AgentTurnRequest) -> AgentTurnResponse:
        input_items: list[dict[str, object]] = [
            {"role": item.role, "content": _json_text(item.content)}
            for item in request.messages
        ]
        for recorded_exchange in _tool_exchanges(request, len(input_items)):
            state = recorded_exchange.provider_state
            if not isinstance(state, list):
                raise AgentTurnProviderError("OPENAI_TOOL_RESULT_STATE_MISSING")
            exchange = [dict(item) for item in state if isinstance(item, dict)]
            exchange.extend(
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": _json_text(result.output),
                }
                for result in recorded_exchange.tool_results
            )
            insert_at = recorded_exchange.insert_index
            input_items[insert_at:insert_at] = exchange
        payload: dict[str, object] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": self.max_output_tokens,
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        if request.tools:
            payload["tools"] = [_responses_tool(tool) for tool in request.tools]
            payload["tool_choice"] = request.tool_choice
        elif request.tool_choice == "none":
            payload["tool_choice"] = "none"
        if request.response_format is not None:
            fmt = request.response_format
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": fmt.name,
                    "description": fmt.description,
                    "schema": fmt.schema,
                    "strict": fmt.strict,
                }
            }
        data = self._send(
            f"{self.base_url}/responses",
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        output = data.get("output")
        if not isinstance(output, list):
            raise AgentTurnProviderError("OPENAI_RESPONSES_OUTPUT_INVALID")
        tool_calls: list[AgentToolCall] = []
        text_parts: list[str] = []
        continuation: list[dict[str, object]] = []
        for item in output:
            if not isinstance(item, dict):
                raise AgentTurnProviderError("OPENAI_RESPONSES_ITEM_INVALID")
            item_type = item.get("type")
            if item_type == "function_call":
                tool_calls.append(
                    AgentToolCall(
                        call_id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=_parse_arguments(item.get("arguments")),
                    )
                )
                continuation.append(dict(item))
            elif item_type == "reasoning":
                continuation.append(dict(item))
            elif item_type == "message":
                content = item.get("content") or []
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text_parts.append(str(part.get("text") or ""))
        usage_data = data.get("usage") or {}
        output_details = usage_data.get("output_tokens_details") or {}
        input_details = usage_data.get("input_tokens_details") or {}
        usage = AgentUsage(
            input_tokens=_int_value(usage_data.get("input_tokens")),
            output_tokens=_int_value(usage_data.get("output_tokens")),
            reasoning_tokens=_int_value(output_details.get("reasoning_tokens")),
            cached_input_tokens=_cached_detail_count(input_details),
        )
        incomplete = data.get("incomplete_details") or {}
        stop_reason = incomplete.get("reason") or data.get("status")
        return AgentTurnResponse(
            provider=self.provider,
            model=str(data.get("model") or self.model),
            content="".join(text_parts) or None,
            tool_calls=tuple(tool_calls),
            stop_reason=str(stop_reason or "") or None,
            request_id=str(data.get("id") or "") or None,
            usage=usage,
            provider_state=continuation,
            raw_response=data,
        )


def _chat_message(message: AgentMessage) -> dict[str, object]:
    return {"role": message.role, "content": _json_text(message.content)}


def _chat_tool(tool: AgentToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _responses_tool(tool: AgentToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": tool.strict,
    }


def _measured_chat_usage(data):
    """Do not turn legacy zero defaults into measured provider usage."""
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    values = {"input_tokens": raw.get("prompt_tokens"), "output_tokens": raw.get("completion_tokens")}
    if any(type(value) is not int or value < 0 for value in values.values()):
        return None
    for group, key, target in (("prompt_tokens_details", "cached_tokens", "cached_input_tokens"),
                               ("completion_tokens_details", "reasoning_tokens", "reasoning_tokens")):
        detail = raw.get(group)
        if isinstance(detail, dict) and type(detail.get(key)) is int and detail[key] >= 0:
            values[target] = detail[key]
    return values


def _qwen_chat_response_diagnostic(
    *,
    data: dict[str, Any],
    raw_tool_calls: object,
    tool_calls: tuple[AgentToolCall, ...],
    usage: AgentUsage,
) -> dict[str, object]:
    raw_items = raw_tool_calls if isinstance(raw_tool_calls, list) else []
    tool_call_diagnostics: list[dict[str, object]] = []
    for index, raw in enumerate(raw_items):
        raw_call = raw if isinstance(raw, dict) else {}
        function = raw_call.get("function")
        raw_function = function if isinstance(function, dict) else {}
        provider_name = str(raw_function.get("name") or "")
        adapter_name = tool_calls[index].name if index < len(tool_calls) else ""
        arguments_audit = _provider_arguments_audit(raw_function.get("arguments"))
        tool_call_diagnostics.append(
            {
                "tool_call_id": str(raw_call.get("id") or ""),
                "provider_reported_tool_name": provider_name,
                "adapter_extracted_tool_name": adapter_name,
                "normalized_tool_name": adapter_name,
                "normalization_applied": False,
                **arguments_audit,
            }
        )
    choice = {}
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
    return {
        "measured_usage": _measured_chat_usage(data),
        "api_type": "openai_compatible_chat_completions",
        "streaming": False,
        "provider_request_id": str(data.get("id") or "") or None,
        "finish_reason": str(choice.get("finish_reason") or "") or None,
        "tool_calls": tool_call_diagnostics,
        "usage": usage.to_dict(),
    }


def _provider_arguments_audit(value: object) -> dict[str, object]:
    valid_json = True
    parsed: object = value
    if isinstance(value, str):
        serialized = value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            valid_json = False
            parsed = value
    elif not isinstance(value, dict):
        valid_json = False
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            serialized = repr(value)
    else:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    valid_object = valid_json and isinstance(parsed, dict)
    return {
        "arguments_valid_json": valid_json,
        "arguments_is_object": valid_object,
        "arguments_sha256": sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _parse_chat_tool_call(item: object) -> AgentToolCall:
    if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
        raise AgentTurnProviderError("QWEN_CHAT_TOOL_CALL_INVALID")
    function = item["function"]
    call_id = str(item.get("id") or "")
    name = str(function.get("name") or "")
    if not call_id or not name:
        raise AgentTurnProviderError("QWEN_CHAT_TOOL_CALL_IDENTITY_INVALID")
    return AgentToolCall(
        call_id=call_id,
        name=name,
        arguments=_parse_arguments(function.get("arguments")),
    )


def _parse_ollama_tool_call(item: object, *, legacy_identity: bool = False) -> AgentToolCall:
    if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
        raise AgentTurnProviderError("OLLAMA_CHAT_TOOL_CALL_INVALID")
    function = item["function"]
    name = str(function.get("name") or "")
    arguments = _parse_arguments(function.get("arguments"))
    if not name:
        raise AgentTurnProviderError("OLLAMA_CHAT_TOOL_CALL_IDENTITY_INVALID")
    native_id = item.get("id")
    if native_id is not None and (type(native_id) is not str or not native_id.strip()):
        raise AgentTurnProviderError("OLLAMA_CHAT_TOOL_CALL_IDENTITY_INVALID")
    # Scope: live native IDs identify invocations, not their argument values.
    # 为什么要有它：同参数的新调用不是原调用重放；缺ID时仍保留旧哈希兼容边界，不能假称可区分两次同参调用。
    if native_id is not None and not legacy_identity:
        return AgentToolCall(call_id=native_id, name=name, arguments=arguments)
    identity = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AgentToolCall(
        call_id="ollama-call-" + sha256(identity).hexdigest()[:16],
        name=name,
        arguments=arguments,
    )


def _ollama_replay_call_identity(raw_calls, retained_calls):
    """Accept one complete exact historical mapping, never a mixed/guessed one.

    Caller must authenticate the retained response file and unexecuted replay
    boundary. 为什么要有它：历史记录曾用参数哈希表示调用，兼容只能在已核准恢复入口使用，不能让实时provider自行降级身份。
    """
    if not isinstance(raw_calls, list) or not raw_calls or not isinstance(retained_calls, list):
        raise AgentTurnProviderError("OLLAMA_REPLAY_TOOL_CALL_IDENTITY_INVALID")
    # Scope: exact JSON values, including bool versus number, across a whole batch.
    # 为什么要有它：Python把False与0视为相等，但工具调用的原始参数类型不能因此被替换。
    def canonical(value):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

    current = [asdict(_parse_ollama_tool_call(item)) for item in raw_calls]
    if canonical(current) == canonical(retained_calls):
        return "native_id_or_missing_id_fallback"
    legacy = [asdict(_parse_ollama_tool_call(item, legacy_identity=True)) for item in raw_calls]
    if canonical(legacy) == canonical(retained_calls):
        return "legacy_name_arguments_sha256_v1"
    raise AgentTurnProviderError("OLLAMA_REPLAY_TOOL_CALL_IDENTITY_INVALID")


def _ollama_provider_diagnostic(
    *,
    data: dict[str, Any],
    requested_model: str,
    max_output_tokens: int,
    context_window: int | None,
    temperature: float,
    seed: int,
    client_elapsed_seconds: float,
    tool_calls: tuple[AgentToolCall, ...],
    context_admission: dict[str, Any],
    request_identity: dict[str, object],
) -> dict[str, object]:
    """Retain bounded native timing facts without treating them as evidence."""

    eval_count = _int_value(data.get("eval_count"))
    eval_duration_ns = _int_value(data.get("eval_duration"))
    tokens_per_second = (
        eval_count / (eval_duration_ns / 1_000_000_000)
        if eval_count > 0 and eval_duration_ns > 0
        else None
    )
    measured_usage = None
    cached = ollama_cached_input_tokens(data)
    if all(type(data.get(key)) is int and data[key] >= 0 for key in ("prompt_eval_count", "eval_count")):
        measured_usage = {"input_tokens": data["prompt_eval_count"], "output_tokens": data["eval_count"]}
        if cached is not None:
            measured_usage["cached_input_tokens"] = cached
    return {
        "measured_usage": measured_usage,
        "api_type": "ollama_native_chat",
        "streaming": False,
        "thinking": False,
        "requested_model": requested_model,
        "returned_model": str(data.get("model") or ""),
        "configured_context_window": context_window,
        "configured_temperature": temperature,
        "configured_seed": seed,
        "configured_max_output_tokens": max_output_tokens,
        "context_admission": dict(context_admission),
        "request_identity": dict(request_identity),
        "client_total_duration_seconds": client_elapsed_seconds,
        "ollama_total_duration_ns": _int_value(data.get("total_duration")),
        "ollama_load_duration_ns": _int_value(data.get("load_duration")),
        "ollama_prompt_eval_duration_ns": _int_value(
            data.get("prompt_eval_duration")
        ),
        "ollama_eval_duration_ns": eval_duration_ns,
        "tokens_per_second": tokens_per_second,
        "first_token_duration_available": False,
        "finish_reason": str(data.get("done_reason") or "") or None,
        "tool_calls": [
            {"tool_call_id": call.call_id, "normalized_tool_name": call.name}
            for call in tool_calls
        ],
        "usage": AgentUsage(
            input_tokens=_int_value(data.get("prompt_eval_count")),
            output_tokens=eval_count,
            cached_input_tokens=cached,
        ).to_dict(),
    }


def _parse_arguments(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise AgentTurnProviderError("AGENT_TOOL_ARGUMENTS_INVALID")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AgentTurnProviderError("AGENT_TOOL_ARGUMENTS_INVALID_JSON") from exc
    if not isinstance(parsed, dict):
        raise AgentTurnProviderError("AGENT_TOOL_ARGUMENTS_MUST_BE_OBJECT")
    return parsed


def _json_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _int_value(value: object) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _cached_detail_count(details: object) -> int | None:
    return nonnegative_token_count(details.get("cached_tokens")) if isinstance(details, Mapping) else None


def _tool_exchanges(request: AgentTurnRequest, message_count: int) -> tuple[AgentToolExchange, ...]:
    """Render later boundaries first so native exchanges do not shift earlier indexes."""

    exchanges = list(request.historical_tool_exchanges)
    if any(not isinstance(item, AgentToolExchange) or not item.tool_results
           or type(item.insert_index) is not int or not 0 <= item.insert_index <= message_count
           for item in exchanges):
        raise AgentTurnProviderError("AGENT_TOOL_HISTORY_INVALID")
    if request.tool_results:
        exchanges.append(AgentToolExchange(request.tool_results, request.provider_state,
            _insert_index(request.tool_result_insert_index, message_count)))
    return tuple(item for _, item in sorted(enumerate(exchanges),
        key=lambda pair: (pair[1].insert_index, pair[0]), reverse=True))


def _insert_index(value: int | None, length: int) -> int:
    if value is None:
        return length
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= length:
        raise AgentTurnProviderError("AGENT_TOOL_RESULT_INSERT_INDEX_INVALID")
    return value
