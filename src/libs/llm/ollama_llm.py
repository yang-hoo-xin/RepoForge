"""Ollama LLM implementation for local model inference.

This module provides the Ollama LLM implementation that works with
locally running Ollama instances. Ollama enables running LLMs like
Llama, Mistral, CodeLlama, etc. on local hardware.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message
from src.libs.llm.transport_observation import transport_started, transport_response_received


def nonnegative_token_count(value: object) -> int | None:
    """Read a measured integer count without inventing an absent zero."""
    return value if type(value) is int and value >= 0 else None


def ollama_cached_input_tokens(response_data: Mapping[str, Any]) -> int | None:
    """Preserve the native inclusive cache count; absent/invalid is unknown."""
    return nonnegative_token_count(response_data.get("prompt_eval_cached_count"))


def ollama_provider_body_utf8_policy_projection(
    payload: Mapping[str, Any],
) -> bytes:
    """Return the stable UTF-8 policy projection used for context admission."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _ollama_canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_frozen_provider_json(text: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonstandard_number(value: str) -> Any:
        raise ValueError(f"non-standard JSON number: {value}")

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonstandard_number,
    )


def build_ollama_context_admission(
    *,
    payload: Mapping[str, Any],
    configured_num_ctx: int | None,
    num_predict: int,
) -> dict[str, Any]:
    """Project configured or explicitly unavailable pre-call admission."""

    if (
        isinstance(num_predict, bool)
        or not isinstance(num_predict, int)
        or num_predict <= 0
    ):
        raise ValueError("Ollama context admission budgets must be positive integers.")
    provider_body_utf8_bytes = len(
        ollama_provider_body_utf8_policy_projection(payload)
    )
    if configured_num_ctx is None:
        return {
            "mode": "context_admission_unavailable_not_configured_v1",
            "exact_prompt_tokens_available_pre_call": False,
            "provider_body_utf8_bytes": provider_body_utf8_bytes,
            "configured_num_ctx": None,
            "num_predict": num_predict,
            "conservative_required_context": None,
            "admitted": None,
        }
    if (
        isinstance(configured_num_ctx, bool)
        or not isinstance(configured_num_ctx, int)
        or configured_num_ctx <= 0
    ):
        raise ValueError("Ollama context admission budgets must be positive integers.")
    conservative_required_context = provider_body_utf8_bytes + num_predict
    return {
        "mode": "provider_body_utf8_proxy_plus_output_budget_v1",
        "exact_prompt_tokens_available_pre_call": False,
        "provider_body_utf8_bytes": provider_body_utf8_bytes,
        "configured_num_ctx": configured_num_ctx,
        "num_predict": num_predict,
        "conservative_required_context": conservative_required_context,
        "admitted": conservative_required_context <= configured_num_ctx,
    }


def _context_request_identity_sha256(payload: Mapping[str, Any]) -> str:
    """Bind a structured request while allowing only num_ctx to change."""

    normalized = dict(payload)
    options = dict(normalized.get("options") or {})
    options.pop("num_ctx", None)
    normalized["options"] = options
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class OllamaLLMError(RuntimeError):
    """Raised when Ollama API call fails.
    
    This exception provides clear error messages without exposing
    sensitive configuration details like internal URLs or credentials.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def _schema_constraint_loss(value: object, path: str = "$") -> dict[str, object] | None:
    """Detect the exact mixed-union shape the bundled llama.cpp drops."""
    if isinstance(value, Mapping):
        for combinator in ("oneOf", "anyOf"):
            if combinator in value and any(
                key in value for key in ("properties", "required", "additionalProperties")
            ):
                ignored = sorted(set(value).intersection(
                    {"type", "properties", "required", "additionalProperties"}))
                return {"json_path": path, "combinator": combinator,
                        "ignored_sibling_keywords": ignored}
        for key, nested in value.items():
            failure = _schema_constraint_loss(nested, f"{path}.{key}")
            if failure is not None:
                return failure
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            failure = _schema_constraint_loss(nested, f"{path}[{index}]")
            if failure is not None:
                return failure
    return None


class OllamaLLM(BaseLLM):
    """Ollama LLM provider implementation for local inference.
    
    This class implements the BaseLLM interface for Ollama's chat API,
    enabling local LLM inference without cloud dependencies.
    
    Attributes:
        base_url: The base URL for the Ollama server (default: http://localhost:11434).
        model: The model identifier to use (e.g., 'llama3', 'mistral').
        default_temperature: Default temperature for generation.
        default_max_tokens: Default max tokens for generation (num_predict in Ollama).
        context_window: Explicit Ollama num_ctx request value when configured.
        timeout: Request timeout in seconds.
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = OllamaLLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # Longer timeout for local inference
    _FROZEN_STRUCTURED_REQUEST_FIELDS = frozenset(
        {
            "method",
            "destination",
            "headers",
            "content_type",
            "body_utf8",
            "body_sha256",
            "body_size_bytes",
            "timeout_seconds",
            "trust_environment",
            "follow_redirects",
            "context_admission",
            "model",
            "output_budget",
            "context_window",
            "temperature",
            "messages_sha256",
            "response_format_sha256",
        }
    )
    _STRUCTURED_BODY_FIELDS = frozenset(
        {"model", "messages", "stream", "options", "think", "format"}
    )
    _STRUCTURED_MESSAGE_FIELDS = frozenset({"role", "content"})
    
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        context_window: int | None = None,
        observed_prompt_tokens: int | None = None,
        observed_prompt_identity_sha256: str | None = None,
        observed_prompt_token_source: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama LLM provider.
        
        Args:
            settings: Application settings containing LLM configuration.
            base_url: Optional override (then settings.llm.base_url, environment, default).
            timeout: Optional timeout override for requests.
            context_window: Optional explicit Ollama num_ctx request value.
            **kwargs: Additional configuration overrides.
        
        Raises:
            ValueError: If required configuration is missing.
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        if (
            context_window is not None
            and (
                isinstance(context_window, bool)
                or not isinstance(context_window, int)
                or context_window <= 0
            )
        ):
            raise ValueError("Ollama context_window must be a positive integer.")
        self.context_window = context_window
        prompt_observation = (
            observed_prompt_tokens,
            observed_prompt_identity_sha256,
            observed_prompt_token_source,
        )
        if any(value is not None for value in prompt_observation):
            if not all(value is not None for value in prompt_observation):
                raise ValueError(
                    "Ollama observed prompt admission requires tokens, "
                    "identity SHA256, and source together."
                )
            if (
                isinstance(observed_prompt_tokens, bool)
                or not isinstance(observed_prompt_tokens, int)
                or observed_prompt_tokens <= 0
            ):
                raise ValueError(
                    "Ollama observed_prompt_tokens must be a positive integer."
                )
            if (
                not isinstance(observed_prompt_identity_sha256, str)
                or len(observed_prompt_identity_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in observed_prompt_identity_sha256
                )
            ):
                raise ValueError(
                    "Ollama observed_prompt_identity_sha256 must be lowercase "
                    "SHA256 hex."
                )
            if (
                not isinstance(observed_prompt_token_source, str)
                or not observed_prompt_token_source.strip()
            ):
                raise ValueError(
                    "Ollama observed_prompt_token_source must be non-empty."
                )
        self.observed_prompt_tokens = observed_prompt_tokens
        self.observed_prompt_identity_sha256 = observed_prompt_identity_sha256
        self.observed_prompt_token_source = observed_prompt_token_source
        self._structured_prompt_calibrations: list[tuple[str, int, int]] = []
        
        # 为什么要有它：工厂调用必须使用配置的endpoint，不能让重排等调用绕开配置的代理、权限或预算记录。
        configured_base_url = getattr(settings.llm, "base_url", None)
        if not isinstance(configured_base_url, str):
            configured_base_url = None
        # Base URL: explicit > settings > env var > default
        self.base_url = (
            base_url 
            or configured_base_url
            or os.environ.get("OLLAMA_BASE_URL") 
            or self.DEFAULT_BASE_URL
        )
        
        # Timeout: explicit > default
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        
        # Store any additional kwargs for future use
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate a chat completion using Ollama API.
        
        Args:
            messages: List of conversation messages.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Override parameters. ``stream_response`` is immediate
                transport only and does not create a resumable frozen request.
        
        Returns:
            ChatResponse with generated content and metadata.
        
        Raises:
            ValueError: If messages are invalid.
            OllamaLLMError: If API call fails.
        """
        # Validate input
        self.validate_messages(messages)
        
        # Prepare request parameters
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)
        response_format = kwargs.get("response_format")
        provider_request_recorder = kwargs.get("provider_request_recorder")
        provider_response_recorder = kwargs.get("provider_response_recorder")
        stream_response = kwargs.get("stream_response", False)
        stream_activity_recorder = kwargs.get("stream_activity_recorder")
        stream_content_recorder = kwargs.get("stream_content_recorder")
        if stream_content_recorder is not None and (not callable(stream_content_recorder) or not stream_response):
            raise ValueError("Ollama stream_content_recorder requires a callable and stream_response=true.")
        if response_format is not None and not isinstance(response_format, Mapping):
            raise ValueError("Ollama response_format must be a JSON Schema object.")
        if provider_request_recorder is not None and not callable(provider_request_recorder):
            raise ValueError("Ollama provider_request_recorder must be callable.")
        if provider_response_recorder is not None and not callable(provider_response_recorder):
            raise ValueError("Ollama provider_response_recorder must be callable.")
        if type(stream_response) is not bool:
            raise ValueError("Ollama stream_response must be a boolean.")
        if stream_activity_recorder is not None and not callable(
            stream_activity_recorder
        ):
            raise ValueError("Ollama stream_activity_recorder must be callable.")
        if stream_activity_recorder is not None and not stream_response:
            raise ValueError(
                "Ollama stream_activity_recorder requires stream_response=true."
            )
        
        # Convert messages to Ollama API format
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Make API call
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=self.context_window,
                response_format=(
                    dict(response_format) if response_format is not None else None
                ),
                provider_request_recorder=provider_request_recorder,
                stream_response=stream_response,
                stream_activity_recorder=stream_activity_recorder,
                **({"stream_content_recorder": stream_content_recorder} if stream_content_recorder is not None else {}),
            )
            if provider_response_recorder is not None:
                # 为什么要有它：完整性门禁拒绝的原生响应也须保留，不能把截断错误变成没有原件的推测。
                try:
                    provider_response_recorder(deepcopy(response_data))
                except Exception as record_error:
                    raise OllamaLLMError(
                        "[Ollama] Provider response persistence failed.",
                        diagnostics={
                            "failure_boundary": "provider_response_recording",
                            "failure_classification": "durable_provider_response_failure",
                        },
                    ) from record_error
            if response_format is not None:
                content, usage = self._admit_chat_response(
                    response_data,
                    expected_model=str(model),
                )
            else:
                if "message" in response_data:
                    content = response_data["message"]["content"]
                elif "response" in response_data:
                    content = response_data["response"]
                else:
                    raise OllamaLLMError(
                        "[Ollama] Unexpected response format: missing "
                        "'message' or 'response' key"
                    )
                usage = self._usage_from_response(response_data)
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except Exception as e:
            if isinstance(e, OllamaLLMError):
                raise
            raise OllamaLLMError(
                f"[Ollama] API call failed: {type(e).__name__}: {e}",
                diagnostics={
                    "failure_boundary": "provider_request_gate",
                    "failure_classification": "durable_provider_request_failure",
                },
            ) from e

    def prepare_structured_chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Freeze one structured request durably without opening an HTTP client."""

        del trace
        self.validate_messages(messages)
        response_format = kwargs.get("response_format")
        provider_request_recorder = kwargs.get("provider_request_recorder")
        if not isinstance(response_format, Mapping):
            raise ValueError(
                "Ollama prepare_structured_chat requires a JSON Schema object."
            )
        if not callable(provider_request_recorder):
            raise ValueError(
                "Ollama prepare_structured_chat requires a durable provider "
                "request recorder."
            )
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)
        api_messages = [
            {"role": item.role, "content": item.content} for item in messages
        ]
        try:
            return self._prepare_structured_api_request(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=self.context_window,
                response_format=dict(response_format),
                provider_request_recorder=provider_request_recorder,
            )
        except Exception as exc:
            if isinstance(exc, OllamaLLMError):
                raise
            raise OllamaLLMError(
                f"[Ollama] API call failed: {type(exc).__name__}: {exc}",
                diagnostics={
                    "failure_boundary": "provider_request_gate",
                    "failure_classification": "durable_provider_request_failure",
                },
            ) from exc

    def resume_structured_chat(
        self,
        frozen_request: Mapping[str, Any],
        *,
        expected_output_budget: int | None = None,
    ) -> ChatResponse:
        """Send one already-frozen structured request using its original bytes."""

        request = self.validate_frozen_structured_chat(
            frozen_request,
            expected_output_budget=(
                self.default_max_tokens
                if expected_output_budget is None
                else expected_output_budget
            ),
        )
        response_data = self._send_api_request(
            url=str(request["destination"]),
            headers=dict(request["headers"]),
            payload=None,
            frozen_request_bytes=str(request["body_utf8"]).encode("utf-8"),
            stream_response=False,
            stream_activity_recorder=None,
        )
        content, usage = self._admit_chat_response(
            response_data,
            expected_model=str(request["model"]),
        )
        self._record_structured_prompt_calibration(
            response_data=response_data,
            request=request,
        )
        return ChatResponse(
            content=content,
            model=str(response_data.get("model", request["model"])),
            usage=usage,
            raw_response=response_data,
        )

    def validate_frozen_structured_chat(
        self,
        frozen_request: Mapping[str, Any],
        *,
        expected_output_budget: int | None = None,
    ) -> dict[str, Any]:
        """Validate one frozen transport without opening an HTTP client."""

        return self._revalidate_frozen_structured_request(
            frozen_request,
            expected_output_budget=(
                self.default_max_tokens
                if expected_output_budget is None
                else expected_output_budget
            ),
        )
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        context_window: int | None = None,
        response_format: Dict[str, Any] | None = None,
        provider_request_recorder: Callable[[dict[str, Any]], None] | None = None,
        stream_response: bool = False,
        stream_activity_recorder: Callable[[dict[str, Any]], None] | None = None,
        stream_content_recorder: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        """Make the actual API call to Ollama.
        
        This method is separated to allow easy mocking in tests.
        
        Args:
            messages: Messages in API format.
            model: Model identifier.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate (num_predict in Ollama).
            context_window: Explicit Ollama num_ctx request value when configured.
        
        Returns:
            Raw API response as dictionary.
        
        Raises:
            OllamaLLMError: If the API call fails.
        """
        url = f"{self.base_url.rstrip('/')}/api/chat"
        headers = {
            "Content-Type": "application/json",
        }
        
        # Ollama uses 'num_predict' instead of 'max_tokens'
        # and 'options' object for model parameters
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if context_window is not None:
            options["num_ctx"] = context_window
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream_response,
            "options": options,
        }
        frozen_request_bytes: bytes | None = None
        if response_format is not None:
            frozen_request = self._prepare_structured_api_request(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                context_window=context_window,
                response_format=response_format,
                provider_request_recorder=provider_request_recorder,
                stream_response=stream_response,
            )
            frozen_request_bytes = str(frozen_request["body_utf8"]).encode("utf-8")
            headers = dict(frozen_request["headers"])
            url = str(frozen_request["destination"])
            payload = {}

        return self._send_api_request(
            url=url,
            headers=headers,
            payload=payload,
            frozen_request_bytes=frozen_request_bytes,
            stream_response=stream_response,
            stream_activity_recorder=stream_activity_recorder,
            **({"stream_content_recorder": stream_content_recorder} if stream_content_recorder is not None else {}),
        )

    def _prepare_structured_api_request(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        context_window: int | None,
        response_format: Dict[str, Any],
        provider_request_recorder: Callable[[dict[str, Any]], None] | None,
        stream_response: bool = False,
    ) -> dict[str, Any]:
        """Build one exact immediate request and record its transport facts.

        Only ``prepare_structured_chat`` creates a resumable frozen request; it
        always uses the existing non-streaming transport profile.
        """

        if provider_request_recorder is None:
            raise OllamaLLMError(
                "[Ollama] Structured generation requires a durable "
                "provider request recorder.",
                diagnostics={
                    "failure_boundary": "provider_request_gate",
                    "failure_classification": "durable_provider_request_missing",
                },
            )
        schema_failure = _schema_constraint_loss(response_format)
        if schema_failure is not None:
            raise OllamaLLMError(
                "[Ollama] Structured schema would lose common constraints in "
                "the admitted local backend.",
                diagnostics={
                    "failure_boundary": "provider_request_gate",
                    "failure_classification": "provider_request_schema_constraint_loss",
                    **schema_failure,
                },
            )
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if context_window is not None:
            options["num_ctx"] = context_window
        payload = {
            "model": model,
            "messages": deepcopy(messages),
            "stream": stream_response,
            "options": options,
            "think": False,
            "format": deepcopy(response_format),
        }
        frozen_request_bytes = ollama_provider_body_utf8_policy_projection(payload)
        context_admission = self._structured_context_admission(
            payload=payload,
            frozen_request_bytes=frozen_request_bytes,
            context_window=context_window,
            max_tokens=max_tokens,
        )
        url = f"{self.base_url.rstrip('/')}/api/chat"
        headers = {"Content-Type": "application/json"}
        frozen_request = {
            "method": "POST",
            "destination": url,
            "headers": headers,
            "content_type": "application/json",
            "body_utf8": frozen_request_bytes.decode("utf-8"),
            "body_sha256": hashlib.sha256(frozen_request_bytes).hexdigest(),
            "body_size_bytes": len(frozen_request_bytes),
            "timeout_seconds": self.timeout,
            "trust_environment": False,
            "follow_redirects": False,
            "context_admission": context_admission,
            "model": model,
            "output_budget": max_tokens,
            "context_window": context_window,
            "temperature": temperature,
            "messages_sha256": _ollama_canonical_json_sha256(payload["messages"]),
            "response_format_sha256": _ollama_canonical_json_sha256(
                payload["format"]
            ),
        }
        provider_request_recorder(deepcopy(frozen_request))
        return frozen_request

    def _structured_context_admission(
        self,
        *,
        payload: Mapping[str, Any],
        frozen_request_bytes: bytes,
        context_window: int | None,
        max_tokens: int,
    ) -> dict[str, Any] | None:
        if context_window is None:
            return None
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise OllamaLLMError(
                "[Ollama] Structured request output budget must be a positive integer.",
                diagnostics={
                    "failure_boundary": "provider_request_gate",
                    "failure_classification": "provider_request_output_budget_invalid",
                },
            )
        if self.observed_prompt_tokens is None:
            context_admission = build_ollama_context_admission(
                payload=payload,
                configured_num_ctx=context_window,
                num_predict=max_tokens,
            )
            if context_admission["admitted"] is not True:
                model = str(payload.get("model") or "")
                observations = [
                    (body_bytes, prompt_tokens)
                    for observed_model, body_bytes, prompt_tokens
                    in self._structured_prompt_calibrations
                    if observed_model == model and body_bytes > 0
                ]
                if observations:
                    provider_body_bytes = int(
                        context_admission["provider_body_utf8_bytes"]
                    )
                    observed_upper_ratio = max(
                        prompt_tokens / body_bytes
                        for body_bytes, prompt_tokens in observations
                    )
                    estimated_prompt_tokens = max(
                        math.ceil(
                            provider_body_bytes
                            * observed_upper_ratio
                            * 2.0
                        ),
                        math.ceil(provider_body_bytes / 2),
                    )
                    estimated_required_context = (
                        estimated_prompt_tokens + max_tokens
                    )
                    context_admission = {
                        "mode": (
                            "same_model_observed_prompt_ratio_upper_bound_v1"
                        ),
                        "exact_prompt_tokens_available_pre_call": False,
                        "provider_body_utf8_bytes": provider_body_bytes,
                        "configured_num_ctx": context_window,
                        "num_predict": max_tokens,
                        "observed_calibration_count": len(observations),
                        "observed_max_prompt_tokens_per_body_byte": (
                            observed_upper_ratio
                        ),
                        "safety_multiplier": 2.0,
                        "estimated_prompt_tokens": estimated_prompt_tokens,
                        "estimated_required_context": (
                            estimated_required_context
                        ),
                        "conservative_fallback_required_context": (
                            context_admission[
                                "conservative_required_context"
                            ]
                        ),
                        "admitted": (
                            estimated_required_context <= context_window
                        ),
                    }
        else:
            observed_identity = _context_request_identity_sha256(payload)
            if observed_identity != self.observed_prompt_identity_sha256:
                raise OllamaLLMError(
                    "[Ollama] Structured request does not match the bound "
                    "prompt-token observation.",
                    diagnostics={
                        "failure_boundary": "provider_request_gate",
                        "failure_classification": (
                            "provider_request_prompt_identity_mismatch"
                        ),
                        "expected_prompt_identity_sha256": (
                            self.observed_prompt_identity_sha256
                        ),
                        "observed_prompt_identity_sha256": observed_identity,
                    },
                )
            observed_required_context = self.observed_prompt_tokens + max_tokens
            context_admission = {
                "mode": "bound_observed_prompt_tokens_plus_output_budget_v1",
                "exact_prompt_tokens_available_pre_call": True,
                "provider_body_utf8_bytes": len(frozen_request_bytes),
                "observed_prompt_tokens": self.observed_prompt_tokens,
                "prompt_identity_sha256": observed_identity,
                "prompt_token_source": self.observed_prompt_token_source,
                "configured_num_ctx": context_window,
                "num_predict": max_tokens,
                "observed_required_context": observed_required_context,
                "admitted": observed_required_context <= context_window,
            }
        if context_admission["admitted"] is not True:
            raise OllamaLLMError(
                "[Ollama] Structured request exceeds the configured pre-call "
                "context budget.",
                diagnostics={
                    "failure_boundary": "provider_request_gate",
                    "failure_classification": (
                        "provider_request_context_budget_exceeded"
                    ),
                    "context_admission": context_admission,
                },
            )
        return context_admission

    def _record_structured_prompt_calibration(
        self,
        *,
        response_data: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> None:
        admission = request.get("context_admission")
        prompt_tokens = response_data.get("prompt_eval_count")
        model = response_data.get("model")
        if (
            response_data.get("done") is not True
            or not isinstance(admission, Mapping)
            or type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or not isinstance(model, str)
            or model != request.get("model")
        ):
            return
        body_bytes = admission.get("provider_body_utf8_bytes")
        if type(body_bytes) is not int or body_bytes <= 0:
            return
        # 为什么要有它：后续结构化阶段可复用同模型真实分词比例，避免把UTF-8字节数永久当成Token数；有限历史防止长任务无界增长。
        self._structured_prompt_calibrations.append(
            (model, body_bytes, prompt_tokens)
        )
        del self._structured_prompt_calibrations[:-64]

    def _revalidate_frozen_structured_request(
        self,
        frozen_request: Mapping[str, Any],
        *,
        expected_output_budget: int,
    ) -> dict[str, Any]:
        diagnostics = {
            "failure_boundary": "frozen_provider_request_revalidation",
            "failure_classification": "frozen_provider_request_identity_mismatch",
        }
        if set(frozen_request) != self._FROZEN_STRUCTURED_REQUEST_FIELDS:
            raise OllamaLLMError(
                "[Ollama] Frozen provider request fields are invalid.",
                diagnostics=diagnostics,
            )
        request = dict(frozen_request)
        body_utf8 = request.get("body_utf8")
        if not isinstance(body_utf8, str):
            raise OllamaLLMError(
                "[Ollama] Frozen provider request body is invalid.",
                diagnostics=diagnostics,
            )
        body_bytes = body_utf8.encode("utf-8")
        if (
            request.get("method") != "POST"
            or request.get("destination") != f"{self.base_url.rstrip('/')}/api/chat"
            or request.get("headers") != {"Content-Type": "application/json"}
            or request.get("content_type") != "application/json"
            or request.get("body_sha256")
            != hashlib.sha256(body_bytes).hexdigest()
            or request.get("body_size_bytes") != len(body_bytes)
            or request.get("timeout_seconds") != self.timeout
            or request.get("trust_environment") is not False
            or request.get("follow_redirects") is not False
            or request.get("model") != self.model
            or request.get("output_budget") != expected_output_budget
            or request.get("context_window") != self.context_window
        ):
            raise OllamaLLMError(
                "[Ollama] Frozen provider request identity does not match the "
                "current provider binding.",
                diagnostics=diagnostics,
            )
        try:
            payload = _strict_frozen_provider_json(body_utf8)
        except (TypeError, ValueError) as exc:
            raise OllamaLLMError(
                "[Ollama] Frozen provider request body shape is invalid.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": (
                        "frozen_provider_request_body_shape_invalid"
                    ),
                },
            ) from exc
        body_shape_diagnostics = {
            **diagnostics,
            "failure_classification": "frozen_provider_request_body_shape_invalid",
        }
        if not isinstance(payload, dict) or set(payload) != self._STRUCTURED_BODY_FIELDS:
            raise OllamaLLMError(
                "[Ollama] Frozen provider request body fields are invalid.",
                diagnostics=body_shape_diagnostics,
            )
        options = payload.get("options")
        messages = payload.get("messages")
        response_format = payload.get("format")
        output_budget = request.get("output_budget")
        expected_option_fields = {"temperature", "num_predict"}
        if request["context_window"] is not None:
            expected_option_fields.add("num_ctx")
        if (
            not isinstance(options, dict)
            or set(options) != expected_option_fields
            or not isinstance(messages, list)
            or not messages
            or not all(
                isinstance(message, dict)
                and set(message) == self._STRUCTURED_MESSAGE_FIELDS
                and message.get("role") in {"system", "user", "assistant"}
                and isinstance(message.get("content"), str)
                and bool(message["content"].strip())
                for message in messages
            )
            or not isinstance(response_format, dict)
            or payload.get("model") != request["model"]
            or payload.get("stream") is not False
            or payload.get("think") is not False
            or options.get("num_predict") != output_budget
            or options.get("num_ctx") != request["context_window"]
            or (
                isinstance(options.get("temperature"), bool)
                or not isinstance(options.get("temperature"), (int, float))
                or not math.isfinite(options["temperature"])
                or options["temperature"] != request.get("temperature")
            )
            or request.get("messages_sha256")
            != _ollama_canonical_json_sha256(messages)
            or request.get("response_format_sha256")
            != _ollama_canonical_json_sha256(response_format)
            or ollama_provider_body_utf8_policy_projection(payload) != body_bytes
        ):
            raise OllamaLLMError(
                "[Ollama] Frozen provider request body shape is invalid.",
                diagnostics=body_shape_diagnostics,
            )
        expected_admission = self._structured_context_admission(
            payload=payload,
            frozen_request_bytes=body_bytes,
            context_window=request["context_window"],
            max_tokens=output_budget,
        )
        if request.get("context_admission") != expected_admission:
            raise OllamaLLMError(
                "[Ollama] Frozen provider request context admission has drifted.",
                diagnostics=diagnostics,
            )
        return request

    def _send_api_request(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        frozen_request_bytes: bytes | None,
        stream_response: bool,
        stream_activity_recorder: Callable[[dict[str, Any]], None] | None,
        stream_content_recorder: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        import httpx

        try:
            with httpx.Client(
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                if stream_response:
                    if frozen_request_bytes is None:
                        stream_context = client.stream(
                            "POST",
                            url,
                            json=dict(payload or {}),
                            headers=dict(headers),
                        )
                    else:
                        stream_context = client.stream(
                            "POST",
                            url,
                            content=frozen_request_bytes,
                            headers=dict(headers),
                        )
                    transport_started()
                    with stream_context as response:
                        transport_response_received(http_status=response.status_code, response_complete=False)
                        if response.status_code != 200:
                            try:
                                response.read()
                            except httpx.RequestError:
                                pass
                            error_detail = self._parse_error_response(response)
                            raise OllamaLLMError(
                                "[Ollama] Streaming API error "
                                f"(HTTP {response.status_code}): {error_detail}",
                                diagnostics={
                                    "failure_boundary": "provider_response_admission",
                                    "failure_classification": "provider_http_error",
                                    "http_status": response.status_code,
                                },
                            )
                        result = self._collect_streaming_response(
                            response,
                            activity_recorder=stream_activity_recorder,
                            content_recorder=stream_content_recorder,
                        )
                        transport_response_received(http_status=response.status_code, response_complete=True)
                        return result
                transport_started()
                if frozen_request_bytes is None:
                    response = client.post(url, json=payload, headers=dict(headers))
                else:
                    response = client.post(
                        url,
                        content=frozen_request_bytes,
                        headers=dict(headers),
                    )
                transport_response_received(http_status=response.status_code, response_complete=True)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise OllamaLLMError(
                        f"[Ollama] API error (HTTP {response.status_code}): {error_detail}",
                        diagnostics={
                            "failure_boundary": "provider_response_admission",
                            "failure_classification": "provider_http_error",
                            "http_status": response.status_code,
                        },
                    )
                try:
                    response_data = response.json()
                except (TypeError, ValueError) as exc:
                    raise OllamaLLMError(
                        "[Ollama] Response envelope is not valid JSON.",
                        diagnostics={
                            "failure_boundary": "provider_response_admission",
                            "failure_classification": (
                                "provider_envelope_json_decoding_failure"
                            ),
                            "http_status": response.status_code,
                        },
                    ) from exc
                if not isinstance(response_data, dict):
                    raise OllamaLLMError(
                        "[Ollama] Response envelope root must be an object.",
                        diagnostics={
                            "failure_boundary": "provider_response_admission",
                            "failure_classification": "provider_envelope_shape_failure",
                            "http_status": response.status_code,
                        },
                    )
                return response_data
        except httpx.TimeoutException as e:
            raise OllamaLLMError(
                f"[Ollama] Request timed out after {self.timeout} seconds. "
                "Consider increasing timeout for larger models or longer responses.",
                diagnostics={
                    "failure_boundary": "provider_response_admission",
                    "failure_classification": "provider_timeout",
                },
            ) from e
        except httpx.ConnectError as e:
            raise OllamaLLMError(
                "[Ollama] Connection failed. Ensure Ollama is running locally. "
                "Start it with 'ollama serve' command.",
                diagnostics={
                    "failure_boundary": "provider_response_admission",
                    "failure_classification": "provider_connection_failure",
                },
            ) from e
        except httpx.RequestError as e:
            raise OllamaLLMError(
                f"[Ollama] Request failed: {type(e).__name__}",
                diagnostics={
                    "failure_boundary": "provider_response_admission",
                    "failure_classification": "provider_request_failure",
                },
            ) from e

    def _collect_streaming_response(
        self,
        response: Any,
        *,
        activity_recorder: Callable[[dict[str, Any]], None] | None,
        content_recorder: Callable[[str], None] | None = None,
    ) -> Dict[str, Any]:
        """Accumulate one Ollama NDJSON response without exposing its content."""

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[Any] = []
        final_chunk: dict[str, Any] | None = None
        observed_model: str | None = None
        cumulative_content_bytes = 0
        chunk_index = 0

        for line in self._iter_strict_stream_lines(response):
            if not line.strip():
                continue
            if final_chunk is not None:
                raise OllamaLLMError(
                    "[Ollama] Streaming response continued after completion.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": "provider_stream_trailing_data",
                    },
                )
            try:
                chunk = _strict_frozen_provider_json(line)
            except (TypeError, ValueError) as exc:
                raise OllamaLLMError(
                    "[Ollama] Streaming response chunk is not valid JSON.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": (
                            "provider_stream_json_decoding_failure"
                        ),
                    },
                ) from exc
            if not isinstance(chunk, dict):
                raise OllamaLLMError(
                    "[Ollama] Streaming response chunk must be an object.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": "provider_stream_chunk_shape_failure",
                    },
                )
            if "error" in chunk:
                raise OllamaLLMError(
                    "[Ollama] Streaming response reported an error.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": "provider_stream_error",
                    },
                )

            chunk_model = chunk.get("model")
            if not isinstance(chunk_model, str) or not chunk_model:
                raise OllamaLLMError(
                    "[Ollama] Streaming response model is invalid.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": "provider_stream_model_shape_failure",
                    },
                )
            if observed_model is None:
                observed_model = chunk_model
            elif chunk_model != observed_model:
                raise OllamaLLMError(
                    "[Ollama] Streaming response model identity changed.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": (
                            "provider_stream_model_identity_drift"
                        ),
                    },
                )

            done = chunk.get("done")
            message = chunk.get("message")
            if type(done) is not bool or not isinstance(message, dict):
                raise OllamaLLMError(
                    "[Ollama] Streaming response chunk shape is invalid.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": "provider_stream_chunk_shape_failure",
                    },
                )
            content = message.get("content", "")
            thinking = message.get("thinking", "")
            raw_tool_calls = message.get("tool_calls", [])
            if (
                not isinstance(content, str)
                or (
                    thinking is not None
                    and not isinstance(thinking, str)
                )
                or (
                    raw_tool_calls is not None
                    and not isinstance(raw_tool_calls, list)
                )
            ):
                raise OllamaLLMError(
                    "[Ollama] Streaming response message shape is invalid.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": "provider_stream_message_shape_failure",
                    },
                )
            content_parts.append(content)
            if content_recorder is not None:
                # 为什么要有它：正文消费者及合作取消是执行边界，异常须关闭现有stream；不发送thinking或吞掉取消。
                content_recorder(content)
            if thinking:
                thinking_parts.append(thinking)
            if raw_tool_calls:
                tool_calls.extend(raw_tool_calls)
            content_bytes = len(content.encode("utf-8"))
            cumulative_content_bytes += content_bytes
            chunk_index += 1
            self._record_stream_activity(
                activity_recorder,
                {
                    "chunk_index": chunk_index,
                    "done": done,
                    "content_utf8_bytes": content_bytes,
                    "cumulative_content_utf8_bytes": cumulative_content_bytes,
                },
            )
            if done:
                final_chunk = dict(chunk)

        if final_chunk is None:
            raise OllamaLLMError(
                "[Ollama] Streaming response ended before completion.",
                diagnostics={
                    "failure_boundary": "provider_response_admission",
                    "failure_classification": "provider_stream_incomplete",
                },
            )
        final_message = dict(final_chunk["message"])
        final_message["content"] = "".join(content_parts)
        if thinking_parts:
            final_message["thinking"] = "".join(thinking_parts)
        if tool_calls:
            final_message["tool_calls"] = tool_calls
        final_chunk["message"] = final_message
        final_chunk["stream_activity"] = {
            "chunk_count": chunk_index,
            "content_utf8_bytes": cumulative_content_bytes,
        }
        return final_chunk

    @staticmethod
    def _iter_strict_stream_lines(response: Any) -> Iterator[str]:
        """Split NDJSON as bytes so malformed UTF-8 cannot be replaced."""

        buffer = bytearray()

        def decode_line(raw_line: bytes) -> str:
            try:
                return raw_line.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise OllamaLLMError(
                    "[Ollama] Streaming response is not valid UTF-8.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": (
                            "provider_stream_utf8_decoding_failure"
                        ),
                    },
                ) from exc

        for raw_chunk in response.iter_bytes():
            if not isinstance(raw_chunk, (bytes, bytearray)):
                raise OllamaLLMError(
                    "[Ollama] Streaming response chunk has an invalid type.",
                    diagnostics={
                        "failure_boundary": "provider_response_admission",
                        "failure_classification": "provider_stream_chunk_shape_failure",
                    },
                )
            buffer.extend(raw_chunk)
            while True:
                newline_index = buffer.find(b"\n")
                if newline_index < 0:
                    break
                raw_line = bytes(buffer[:newline_index])
                del buffer[: newline_index + 1]
                yield decode_line(raw_line)
        if buffer:
            yield decode_line(bytes(buffer))

    @staticmethod
    def _record_stream_activity(
        recorder: Callable[[dict[str, Any]], None] | None,
        event: dict[str, Any],
    ) -> None:
        if recorder is None:
            return
        try:
            recorder(dict(event))
        except Exception:
            # Progress reporting is deliberately non-authoritative.
            return

    def _admit_chat_response(
        self,
        response_data: Dict[str, Any],
        *,
        expected_model: str,
    ) -> tuple[str, dict[str, int] | None]:
        """Apply the bounded complete local response profile."""

        response_model = response_data.get("model")
        done = response_data.get("done")
        done_reason = response_data.get("done_reason")
        message = response_data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        diagnostics = {
            "failure_boundary": "provider_response_admission",
            "response_model": response_model,
            "done": done,
            "done_reason": done_reason,
            "message_content": content if isinstance(content, str) else None,
        }
        if response_model != expected_model:
            raise OllamaLLMError(
                "[Ollama] Response model does not match the admitted exact model.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": "response_model_identity_mismatch",
                },
            )
        if done is not True:
            raise OllamaLLMError(
                "[Ollama] Response is not a complete completion.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": "provider_response_incomplete",
                },
            )
        if done_reason != "stop":
            raise OllamaLLMError(
                "[Ollama] Response did not finish with the admitted stop reason.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": "provider_response_stop_reason_rejected",
                },
            )
        if not isinstance(message, dict):
            raise OllamaLLMError(
                "[Ollama] Response message must be an object.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": "provider_message_shape_failure",
                },
            )
        if not isinstance(content, str) or not content.strip():
            raise OllamaLLMError(
                "[Ollama] Response message.content must be non-empty.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": "provider_message_content_empty",
                },
            )
        if message.get("thinking") not in (None, ""):
            raise OllamaLLMError(
                "[Ollama] Thinking content is not admitted for patch generation.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": "provider_thinking_not_admitted",
                },
            )
        if message.get("tool_calls") not in (None, []):
            raise OllamaLLMError(
                "[Ollama] Tool calls are not admitted for patch generation.",
                diagnostics={
                    **diagnostics,
                    "failure_classification": "provider_tool_calls_not_admitted",
                },
            )
        return content, self._usage_from_response(response_data)

    @staticmethod
    def _usage_from_response(
        response_data: Mapping[str, Any],
    ) -> dict[str, int] | None:
        if "eval_count" not in response_data and "prompt_eval_count" not in response_data:
            return None
        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            return None
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        cached = ollama_cached_input_tokens(response_data)
        if cached is not None:
            usage["cached_input_tokens"] = cached
        return usage
    
    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response.
        
        Args:
            response: The HTTP response object.
        
        Returns:
            Human-readable error message without exposing sensitive details.
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                return str(error_data["error"])
            return response.text[:200] if response.text else "Unknown error"
        except Exception:
            return response.text[:200] if response.text else "Unknown error"
