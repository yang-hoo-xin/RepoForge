"""Unit tests for Ollama LLM implementation.

This module tests the Ollama LLM provider using mocked HTTP responses
to avoid requiring a running Ollama instance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.libs.llm import LLMFactory, Message, OllamaLLM, OllamaLLMError
from src.libs.llm.ollama_llm import _context_request_identity_sha256


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@dataclass
class MockLLMSettings:
    """Mock settings for LLM testing."""
    
    provider: str = "ollama"
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class MockSettings:
    """Mock application settings."""
    
    llm: MockLLMSettings = None
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = MockLLMSettings()


def make_ollama_response(
    content: str = "Hello! I'm running locally via Ollama.",
    model: str = "llama3",
    status_code: int = 200,
    prompt_eval_count: int = 15,
    eval_count: int = 25,
) -> MagicMock:
    """Create a mock Ollama HTTP response.
    
    Ollama's /api/chat endpoint returns a different format than OpenAI:
    - Uses 'message' object with 'role' and 'content'
    - Uses 'eval_count' and 'prompt_eval_count' for token counts
    """
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "model": model,
        "created_at": "2026-01-28T12:00:00.000000Z",
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 1234567890,
        "load_duration": 12345678,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": 123456789,
        "eval_count": eval_count,
        "eval_duration": 987654321,
    }
    return response


def make_error_response(
    status_code: int = 400,
    error_message: str = "model not found",
) -> MagicMock:
    """Create a mock error HTTP response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"error": error_message}
    response.text = f"Error: {error_message}"
    return response


def make_ollama_stream_response(
    chunks: list[dict[str, Any] | str | bytes],
    *,
    status_code: int = 200,
) -> MagicMock:
    """Create one mocked NDJSON streaming response."""

    response = MagicMock()
    response.status_code = status_code
    response.iter_bytes.return_value = [
        (
            item
            if isinstance(item, bytes)
            else (
                item
                if isinstance(item, str)
                else json.dumps(item, ensure_ascii=False)
            ).encode("utf-8")
        )
        + b"\n"
        for item in chunks
    ]
    response.read.return_value = b""
    return response


# -----------------------------------------------------------------------------
# Factory Registration Tests
# -----------------------------------------------------------------------------


class TestOllamaFactoryRegistration:
    """Tests for Ollama provider factory registration."""
    
    @pytest.fixture(autouse=True)
    def ensure_provider_registered(self):
        """Ensure Ollama provider is registered before each test."""
        if "ollama" not in LLMFactory._PROVIDERS:
            LLMFactory.register_provider("ollama", OllamaLLM)
        yield
    
    def test_ollama_registered(self):
        """Ollama provider should be registered with factory."""
        assert "ollama" in LLMFactory.list_providers()
    
    def test_factory_creates_ollama(self):
        """Factory should create OllamaLLM instance when provider=ollama."""
        settings = MockSettings(llm=MockLLMSettings(provider="ollama"))
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OllamaLLM)
    
    def test_factory_creates_ollama_case_insensitive(self):
        """Factory should handle case-insensitive provider name."""
        settings = MockSettings(llm=MockLLMSettings(provider="OLLAMA"))
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OllamaLLM)


# -----------------------------------------------------------------------------
# Initialization Tests
# -----------------------------------------------------------------------------


class TestOllamaInit:
    """Tests for OllamaLLM initialization."""
    
    def test_init_default_base_url(self):
        """Should use default localhost URL when none specified."""
        settings = MockSettings()
        llm = OllamaLLM(settings)
        assert llm.base_url == "http://localhost:11434"
    
    def test_init_custom_base_url(self):
        """Should use provided base URL."""
        settings = MockSettings()
        llm = OllamaLLM(settings, base_url="http://192.168.1.100:11434")
        assert llm.base_url == "http://192.168.1.100:11434"
    
    def test_init_base_url_from_env(self):
        """Should read base URL from environment variable."""
        settings = MockSettings()
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://remote:11434"}):
            llm = OllamaLLM(settings)
            assert llm.base_url == "http://remote:11434"
    
    def test_init_explicit_base_url_overrides_env(self):
        """Explicit base URL should override environment variable."""
        settings = MockSettings()
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env:11434"}):
            llm = OllamaLLM(settings, base_url="http://explicit:11434")
            assert llm.base_url == "http://explicit:11434"

    def test_factory_uses_settings_endpoint_before_environment(self):
        settings = MockSettings()
        settings.llm.base_url = "http://127.0.0.1:59123"
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env:11434"}):
            assert LLMFactory.create(settings).base_url == settings.llm.base_url
            assert LLMFactory.create(settings, base_url="http://explicit:11434").base_url == "http://explicit:11434"

    def test_reranker_uses_configured_endpoint(self, tmp_path):
        from src.libs.reranker.llm_reranker import LLMReranker
        settings = MockSettings()
        settings.llm.base_url = "http://127.0.0.1:59124"
        prompt = tmp_path / "rerank.txt"
        prompt.write_text("Rank the supplied passages.", encoding="utf-8")
        assert LLMReranker(settings, prompt_path=str(prompt)).llm.base_url == settings.llm.base_url
    
    def test_init_model_from_settings(self):
        """Should read model from settings."""
        settings = MockSettings(llm=MockLLMSettings(model="mistral"))
        llm = OllamaLLM(settings)
        assert llm.model == "mistral"
    
    def test_init_temperature_from_settings(self):
        """Should read temperature from settings."""
        settings = MockSettings(llm=MockLLMSettings(temperature=0.5))
        llm = OllamaLLM(settings)
        assert llm.default_temperature == 0.5
    
    def test_init_max_tokens_from_settings(self):
        """Should read max_tokens from settings."""
        settings = MockSettings(llm=MockLLMSettings(max_tokens=4096))
        llm = OllamaLLM(settings)
        assert llm.default_max_tokens == 4096
    
    def test_init_custom_timeout(self):
        """Should accept custom timeout."""
        settings = MockSettings()
        llm = OllamaLLM(settings, timeout=300.0)
        assert llm.timeout == 300.0

    def test_init_explicit_context_window(self):
        settings = MockSettings()
        llm = OllamaLLM(settings, context_window=32768)

        assert llm.context_window == 32768

    @pytest.mark.parametrize("value", (0, -1, True, 1.5, "32768"))
    def test_init_rejects_invalid_context_window(self, value):
        settings = MockSettings()

        with pytest.raises(ValueError, match="positive integer"):
            OllamaLLM(settings, context_window=value)

    def test_init_rejects_partial_observed_prompt_admission(self):
        settings = MockSettings()

        with pytest.raises(ValueError, match="requires tokens"):
            OllamaLLM(
                settings,
                observed_prompt_tokens=32,
                observed_prompt_identity_sha256="0" * 64,
            )

    @pytest.mark.parametrize("value", (0, -1, True, 1.5, "32"))
    def test_init_rejects_invalid_observed_prompt_tokens(self, value):
        settings = MockSettings()

        with pytest.raises(ValueError, match="positive integer"):
            OllamaLLM(
                settings,
                observed_prompt_tokens=value,
                observed_prompt_identity_sha256="0" * 64,
                observed_prompt_token_source="bound_test_observation",
            )

    @pytest.mark.parametrize("value", ("", "A" * 64, "g" * 64, "0" * 63))
    def test_init_rejects_invalid_observed_prompt_identity(self, value):
        settings = MockSettings()

        with pytest.raises(ValueError, match="lowercase SHA256"):
            OllamaLLM(
                settings,
                observed_prompt_tokens=32,
                observed_prompt_identity_sha256=value,
                observed_prompt_token_source="bound_test_observation",
            )
    
    def test_init_default_timeout(self):
        """Should use default timeout when none specified."""
        settings = MockSettings()
        llm = OllamaLLM(settings)
        assert llm.timeout == 120.0  # Longer default for local inference


# -----------------------------------------------------------------------------
# Chat Tests with Mocked HTTP
# -----------------------------------------------------------------------------


class TestOllamaChat:
    """Tests for OllamaLLM chat functionality with mocked HTTP."""
    
    @pytest.fixture
    def llm(self):
        """Create an OllamaLLM instance for testing."""
        settings = MockSettings()
        return OllamaLLM(settings)
    
    def test_chat_success(self, llm):
        """Should successfully process a chat request."""
        mock_response = make_ollama_response(content="Hello from Ollama!")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.content == "Hello from Ollama!"
            assert response.model == "llama3"
    
    def test_chat_returns_usage_stats(self, llm):
        """Should include usage statistics in response."""
        mock_response = make_ollama_response(
            prompt_eval_count=20,
            eval_count=50,
        )
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.usage is not None
            assert response.usage["prompt_tokens"] == 20
            assert response.usage["completion_tokens"] == 50
            assert response.usage["total_tokens"] == 70
    
    def test_chat_preserves_raw_response(self, llm):
        """Should preserve raw response for debugging."""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat([Message(role="user", content="Hello")])
            
            assert response.raw_response is not None
            assert "model" in response.raw_response
    
    def test_chat_with_system_message(self, llm):
        """Should handle system messages correctly."""
        mock_response = make_ollama_response(content="I understand the context.")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
            response = llm.chat(messages)
            
            assert response.content == "I understand the context."
            
            # Verify API was called with both messages
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert len(payload["messages"]) == 2
            assert payload["messages"][0]["role"] == "system"
    
    def test_chat_with_conversation(self, llm):
        """Should handle multi-turn conversations."""
        mock_response = make_ollama_response(content="The capital is Paris.")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            messages = [
                Message(role="user", content="What is the capital of France?"),
                Message(role="assistant", content="Paris is the capital of France."),
                Message(role="user", content="Tell me more about it."),
            ]
            response = llm.chat(messages)
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert len(payload["messages"]) == 3
    
    def test_chat_with_temperature_override(self, llm):
        """Should allow temperature override in chat call."""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")], temperature=0.1)
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["temperature"] == 0.1
    
    def test_chat_with_max_tokens_override(self, llm):
        """Should allow max_tokens override in chat call."""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")], max_tokens=500)
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["options"]["num_predict"] == 500  # Ollama uses num_predict
    
    def test_chat_uses_stream_false(self, llm):
        """Should disable streaming for synchronous response."""
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")])
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["stream"] is False

    def test_structured_chat_streams_and_accumulates_content(self, llm):
        schema = {
            "type": "object",
            "required": ["changes", "message"],
            "properties": {
                "changes": {"type": "array"},
                "message": {"type": "string"},
            },
        }
        chunks = [
            {
                "model": "llama3",
                "message": {"role": "assistant", "content": '{"changes":[],'},
                "done": False,
            },
            {
                "model": "llama3",
                "message": {"role": "assistant", "content": '"message":"完成😀"}'},
                "done": False,
            },
            {
                "model": "llama3",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": 4_000_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_count": 123,
                "prompt_eval_duration": 1_000_000_000,
                "eval_count": 17,
                "eval_duration": 2_000_000_000,
            },
        ]
        stream_response = make_ollama_stream_response(chunks)
        activity: list[dict[str, Any]] = []
        recorded: dict[str, Any] = {}

        with patch("httpx.Client") as mock_client:
            client = mock_client.return_value.__enter__.return_value
            client.stream.return_value.__enter__.return_value = stream_response
            result = llm.chat(
                [Message(role="user", content="return structured JSON")],
                temperature=0.0,
                response_format=schema,
                provider_request_recorder=recorded.update,
                stream_response=True,
                stream_activity_recorder=activity.append,
            )

        stream_call = client.stream.call_args
        assert stream_call.args[:2] == (
            "POST",
            "http://localhost:11434/api/chat",
        )
        payload = json.loads(stream_call.kwargs["content"].decode("utf-8"))
        assert payload["stream"] is True
        assert (
            recorded["body_utf8"].encode("utf-8")
            == stream_call.kwargs["content"]
        )
        assert result.content == '{"changes":[],"message":"完成😀"}'
        assert result.usage == {
            "prompt_tokens": 123,
            "completion_tokens": 17,
            "total_tokens": 140,
        }
        assert result.raw_response["done"] is True
        assert result.raw_response["done_reason"] == "stop"
        assert result.raw_response["stream_activity"] == {
            "chunk_count": 3,
            "content_utf8_bytes": len(result.content.encode("utf-8")),
        }
        assert [item["chunk_index"] for item in activity] == [1, 2, 3]
        assert [item["done"] for item in activity] == [False, False, True]
        assert all(
            set(item)
            == {
                "chunk_index",
                "done",
                "content_utf8_bytes",
                "cumulative_content_utf8_bytes",
            }
            for item in activity
        )
        client.post.assert_not_called()

        with pytest.raises(OllamaLLMError) as exc_info:
            llm.validate_frozen_structured_chat(recorded)
        assert exc_info.value.diagnostics["failure_classification"] == (
            "frozen_provider_request_body_shape_invalid"
        )
        assert mock_client.call_count == 1

    @pytest.mark.parametrize(
        ("chunks", "classification"),
        [
            (
                [
                    {
                        "model": "llama3",
                        "message": {"role": "assistant", "content": "partial"},
                        "done": False,
                    },
                    {"error": "private provider detail"},
                ],
                "provider_stream_error",
            ),
            (
                [
                    {
                        "model": "llama3",
                        "message": {"role": "assistant", "content": "partial"},
                        "done": False,
                    }
                ],
                "provider_stream_incomplete",
            ),
            (["not-json"], "provider_stream_json_decoding_failure"),
            (
                [b'{"model":"llama3","message":{"content":"\xff"},"done":true}'],
                "provider_stream_utf8_decoding_failure",
            ),
            (
                [
                    {
                        "model": "llama3",
                        "message": {"role": "assistant", "content": "complete"},
                        "done": True,
                        "done_reason": "stop",
                    },
                    {
                        "model": "llama3",
                        "message": {"role": "assistant", "content": "trailing"},
                        "done": False,
                    },
                ],
                "provider_stream_trailing_data",
            ),
        ],
    )
    def test_streaming_response_failures_are_terminal_without_retry(
        self,
        llm,
        chunks,
        classification,
    ):
        stream_response = make_ollama_stream_response(chunks)

        with patch("httpx.Client") as mock_client:
            client = mock_client.return_value.__enter__.return_value
            client.stream.return_value.__enter__.return_value = stream_response
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat(
                    [Message(role="user", content="return structured JSON")],
                    response_format={"type": "object"},
                    provider_request_recorder=lambda _request: None,
                    stream_response=True,
                )

        assert (
            exc_info.value.diagnostics["failure_classification"]
            == classification
        )
        assert "private provider detail" not in str(exc_info.value)
        assert client.stream.call_count == 1
        client.post.assert_not_called()

    def test_stream_activity_recorder_failure_does_not_change_result(self, llm):
        stream_response = make_ollama_stream_response(
            [
                {
                    "model": "llama3",
                    "message": {"role": "assistant", "content": '{"ok":true}'},
                    "done": True,
                    "done_reason": "stop",
                }
            ]
        )

        with patch("httpx.Client") as mock_client:
            client = mock_client.return_value.__enter__.return_value
            client.stream.return_value.__enter__.return_value = stream_response
            result = llm.chat(
                [Message(role="user", content="return structured JSON")],
                response_format={"type": "object"},
                provider_request_recorder=lambda _request: None,
                stream_response=True,
                stream_activity_recorder=lambda _event: (_ for _ in ()).throw(
                    RuntimeError("progress sink failed")
                ),
            )

        assert result.content == '{"ok":true}'

    def test_structured_patch_request_is_frozen_recorded_then_sent(
        self,
        llm,
        tmp_path,
    ):
        schema = {
            "type": "object",
            "required": ["changes"],
            "properties": {"changes": {"type": "array"}},
        }
        response = make_ollama_response(content='{"changes":[]}')
        audit_path = tmp_path / "attempt.json"
        recorded = {}

        def record(request):
            recorded.update(request)
            audit_path.write_text(json.dumps(request), encoding="utf-8")

        def post(*args, **kwargs):
            assert audit_path.is_file()
            return response

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = post
            result = llm.chat(
                [Message(role="user", content="return a patch")],
                temperature=0.0,
                max_tokens=4096,
                response_format=schema,
                provider_request_recorder=record,
            )

        mock_client.assert_called_once_with(
            timeout=120.0,
            trust_env=False,
            follow_redirects=False,
        )
        call = mock_client.return_value.__enter__.return_value.post.call_args
        sent_bytes = call.kwargs["content"]
        payload = json.loads(sent_bytes.decode("utf-8"))
        assert result.content == '{"changes":[]}'
        assert payload["stream"] is False
        assert payload["think"] is False
        assert "tools" not in payload
        assert payload["format"] == schema
        assert payload["options"] == {"temperature": 0.0, "num_predict": 4096}
        assert recorded["body_utf8"].encode("utf-8") == sent_bytes
        assert recorded["body_sha256"] == hashlib.sha256(sent_bytes).hexdigest()
        assert recorded["body_size_bytes"] == len(sent_bytes)
        assert recorded["timeout_seconds"] == 120.0
        assert recorded["trust_environment"] is False
        assert recorded["follow_redirects"] is False

    def test_prepare_structured_chat_matches_immediate_body_without_http(
        self,
        llm,
    ):
        schema = {
            "type": "object",
            "required": ["changes"],
            "properties": {"changes": {"type": "array"}},
        }
        prepared = {}

        with patch("httpx.Client") as mock_client:
            frozen = llm.prepare_structured_chat(
                [Message(role="user", content="return a patch 内容")],
                temperature=0.0,
                max_tokens=4096,
                response_format=schema,
                provider_request_recorder=prepared.update,
            )

        mock_client.assert_not_called()
        assert frozen == prepared

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_ollama_response(content='{"changes":[]}')
            )
            llm.chat(
                [Message(role="user", content="return a patch 内容")],
                temperature=0.0,
                max_tokens=4096,
                response_format=schema,
                provider_request_recorder=lambda _: None,
            )

        sent = mock_client.return_value.__enter__.return_value.post.call_args.kwargs[
            "content"
        ]
        assert prepared["body_utf8"].encode("utf-8") == sent
        assert prepared["body_sha256"] == hashlib.sha256(sent).hexdigest()
        assert prepared["body_size_bytes"] == len(sent)

    def test_prepare_structured_chat_persistence_failure_has_zero_http(self, llm):
        def fail_record(_):
            raise OSError("freeze persistence failed")

        with patch("httpx.Client") as mock_client:
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.prepare_structured_chat(
                    [Message(role="user", content="return a patch")],
                    response_format={"type": "object"},
                    provider_request_recorder=fail_record,
                )

        assert exc_info.value.diagnostics["failure_classification"] == (
            "durable_provider_request_failure"
        )
        mock_client.assert_not_called()

    def test_structured_recorder_nested_mutation_cannot_change_immediate_send(self):
        settings = MockSettings()
        llm = OllamaLLM(settings, context_window=32768)
        response = make_ollama_response(content='{"changes":[]}')

        def mutate_record(record):
            record["headers"]["Content-Type"] = "text/plain"
            record["context_admission"]["admitted"] = False
            record["body_utf8"] = "{}"

        with patch("httpx.Client") as mock_client:
            post = mock_client.return_value.__enter__.return_value.post
            post.return_value = response
            llm.chat(
                [Message(role="user", content="return a patch")],
                response_format={"type": "object"},
                provider_request_recorder=mutate_record,
            )

        call = post.call_args
        assert call.kwargs["headers"] == {"Content-Type": "application/json"}
        assert json.loads(call.kwargs["content"])["messages"] == [
            {"role": "user", "content": "return a patch"}
        ]

    def test_structured_recorder_nested_mutation_cannot_change_frozen_return(self):
        settings = MockSettings()
        llm = OllamaLLM(settings, context_window=32768)

        def mutate_record(record):
            record["headers"]["Content-Type"] = "text/plain"
            record["context_admission"]["admitted"] = False
            record["body_utf8"] = "{}"

        frozen = llm.prepare_structured_chat(
            [Message(role="user", content="return a patch")],
            response_format={"type": "object"},
            provider_request_recorder=mutate_record,
        )

        assert frozen["headers"] == {"Content-Type": "application/json"}
        assert frozen["context_admission"]["admitted"] is True
        assert json.loads(frozen["body_utf8"])["messages"] == [
            {"role": "user", "content": "return a patch"}
        ]

    def test_frozen_request_json_roundtrip_preserves_exact_bytes_and_resume_send(
        self,
        llm,
        tmp_path,
    ):
        frozen_path = tmp_path / "frozen.json"
        frozen = llm.prepare_structured_chat(
            [Message(role="user", content="return 内容")],
            temperature=0.0,
            max_tokens=4096,
            response_format={"type": "object"},
            provider_request_recorder=lambda value: frozen_path.write_text(
                json.dumps(value, ensure_ascii=False),
                encoding="utf-8",
            ),
        )
        loaded = json.loads(frozen_path.read_text(encoding="utf-8"))
        expected_bytes = frozen["body_utf8"].encode("utf-8")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_ollama_response(content='{"changes":[]}')
            )
            response = llm.resume_structured_chat(
                loaded,
                expected_output_budget=4096,
            )

        sent = mock_client.return_value.__enter__.return_value.post.call_args.kwargs[
            "content"
        ]
        assert response.content == '{"changes":[]}'
        assert sent == expected_bytes
        assert loaded["body_sha256"] == hashlib.sha256(sent).hexdigest()
        assert loaded["body_size_bytes"] == len(sent)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("body_utf8", "{}"),
            ("model", "another-model"),
            ("destination", "http://127.0.0.1:11434/api/chat"),
            ("output_budget", 1),
            ("context_window", 1),
            ("method", "GET"),
            ("content_type", "text/plain"),
            ("timeout_seconds", 1.0),
            ("trust_environment", True),
            ("follow_redirects", True),
        ],
    )
    def test_resume_structured_chat_rejects_frozen_transport_drift_before_http(
        self,
        llm,
        field,
        value,
    ):
        frozen = llm.prepare_structured_chat(
            [Message(role="user", content="return a patch")],
            max_tokens=4096,
            response_format={"type": "object"},
            provider_request_recorder=lambda _: None,
        )
        frozen[field] = value

        with patch("httpx.Client") as mock_client:
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.resume_structured_chat(frozen)

        assert exc_info.value.diagnostics["failure_boundary"] == (
            "frozen_provider_request_revalidation"
        )
        mock_client.assert_not_called()

    @pytest.mark.parametrize("change", ("missing", "extra"))
    def test_resume_structured_chat_rejects_missing_or_extra_frozen_fields(
        self,
        llm,
        change,
    ):
        frozen = llm.prepare_structured_chat(
            [Message(role="user", content="return a patch")],
            response_format={"type": "object"},
            provider_request_recorder=lambda _: None,
        )
        if change == "missing":
            frozen.pop("method")
        else:
            frozen["unexpected"] = True

        with patch("httpx.Client") as mock_client:
            with pytest.raises(OllamaLLMError):
                llm.resume_structured_chat(frozen)

        mock_client.assert_not_called()

    @pytest.mark.parametrize(
        "mutation",
        (
            "root_missing",
            "root_extra",
            "options_missing",
            "options_extra",
            "message_extra",
            "message_count",
            "message_order",
            "schema_drift",
            "tools",
            "tool_choice",
            "keep_alive",
            "duplicate_key",
            "nan",
            "infinity",
        ),
    )
    def test_resume_rejects_noncanonical_frozen_body_shape_before_http(
        self,
        llm,
        mutation,
    ):
        frozen = llm.prepare_structured_chat(
            [
                Message(role="system", content="system"),
                Message(role="user", content="return a patch"),
            ],
            response_format={"type": "object"},
            provider_request_recorder=lambda _: None,
        )
        body = json.loads(frozen["body_utf8"])
        if mutation == "root_missing":
            body.pop("messages")
        elif mutation == "root_extra":
            body["unexpected"] = True
        elif mutation == "options_missing":
            body["options"].pop("temperature")
        elif mutation == "options_extra":
            body["options"]["unexpected"] = True
        elif mutation == "message_extra":
            body["messages"][0]["name"] = "unexpected"
        elif mutation == "message_count":
            body["messages"].append(dict(body["messages"][-1]))
        elif mutation == "message_order":
            body["messages"] = list(reversed(body["messages"]))
        elif mutation == "schema_drift":
            body["format"] = {"type": "array"}
        elif mutation in {"tools", "tool_choice", "keep_alive"}:
            body[mutation] = [] if mutation == "tools" else "unexpected"
        elif mutation == "nan":
            body["options"]["temperature"] = float("nan")
        elif mutation == "infinity":
            body["options"]["temperature"] = float("inf")
        body_utf8 = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if mutation == "duplicate_key":
            body_utf8 = body_utf8[:-1] + ',"model":"llama3"}'
        body_bytes = body_utf8.encode("utf-8")
        frozen["body_utf8"] = body_utf8
        frozen["body_sha256"] = hashlib.sha256(body_bytes).hexdigest()
        frozen["body_size_bytes"] = len(body_bytes)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_ollama_response(content='{"changes":[]}')
            )
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.resume_structured_chat(frozen)

        assert exc_info.value.diagnostics["failure_classification"] == (
            "frozen_provider_request_body_shape_invalid"
        )
        mock_client.assert_not_called()

    @pytest.mark.parametrize("failure", ("timeout", "http"))
    def test_resume_structured_chat_failure_has_one_post_without_retry(
        self,
        llm,
        failure,
    ):
        import httpx

        frozen = llm.prepare_structured_chat(
            [Message(role="user", content="return a patch")],
            response_format={"type": "object"},
            provider_request_recorder=lambda _: None,
        )
        with patch("httpx.Client") as mock_client:
            post = mock_client.return_value.__enter__.return_value.post
            if failure == "timeout":
                post.side_effect = httpx.TimeoutException("timeout")
            else:
                post.return_value = make_error_response(status_code=503)
            with pytest.raises(OllamaLLMError):
                llm.resume_structured_chat(frozen)

        assert post.call_count == 1

    def test_structured_patch_request_sends_and_audits_explicit_num_ctx(self):
        settings = MockSettings()
        llm = OllamaLLM(settings, context_window=32768)
        response = make_ollama_response(content='{"changes":[]}')
        recorded = {}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = response
            llm.chat(
                [Message(role="user", content="return a patch")],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "object"},
                provider_request_recorder=recorded.update,
            )

        sent_bytes = (
            mock_client.return_value.__enter__.return_value.post.call_args.kwargs[
                "content"
            ]
        )
        payload = json.loads(sent_bytes.decode("utf-8"))
        assert payload["options"] == {
            "temperature": 0.0,
            "num_predict": 4096,
            "num_ctx": 32768,
        }
        assert recorded["context_admission"] == {
            "mode": "provider_body_utf8_proxy_plus_output_budget_v1",
            "exact_prompt_tokens_available_pre_call": False,
            "provider_body_utf8_bytes": len(sent_bytes),
            "configured_num_ctx": 32768,
            "num_predict": 4096,
            "conservative_required_context": len(sent_bytes) + 4096,
            "admitted": True,
        }

    def test_structured_patch_context_budget_blocks_before_http(self):
        settings = MockSettings()
        llm = OllamaLLM(settings, context_window=128)
        recorded = []

        with patch("httpx.Client") as mock_client:
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat(
                    [Message(role="user", content="x" * 256)],
                    max_tokens=64,
                    response_format={"type": "object"},
                    provider_request_recorder=recorded.append,
                )

        assert exc_info.value.diagnostics["failure_boundary"] == (
            "provider_request_gate"
        )
        assert exc_info.value.diagnostics["failure_classification"] == (
            "provider_request_context_budget_exceeded"
        )
        admission = exc_info.value.diagnostics["context_admission"]
        assert admission["configured_num_ctx"] == 128
        assert admission["num_predict"] == 64
        assert admission["admitted"] is False
        assert recorded == []
        mock_client.assert_not_called()

    def test_completed_structured_request_calibrates_later_same_model_gate(self):
        settings = MockSettings()
        llm = OllamaLLM(settings, context_window=1024)
        schema = {"type": "object"}
        first = llm.prepare_structured_chat(
            [Message(role="user", content="short request")],
            max_tokens=64,
            response_format=schema,
            provider_request_recorder=lambda _: None,
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_ollama_response(
                    content='{"changes":[]}',
                    prompt_eval_count=25,
                )
            )
            llm.resume_structured_chat(first, expected_output_budget=64)

        recorded = {}
        second = llm.prepare_structured_chat(
            [Message(role="user", content="x" * 1400)],
            max_tokens=64,
            response_format=schema,
            provider_request_recorder=recorded.update,
        )

        admission = second["context_admission"]
        assert admission["mode"] == (
            "same_model_observed_prompt_ratio_upper_bound_v1"
        )
        assert admission["observed_calibration_count"] == 1
        assert admission["conservative_fallback_required_context"] > 1024
        assert admission["estimated_required_context"] <= 1024
        assert admission["admitted"] is True
        assert recorded == second

    def test_structured_calibration_is_not_reused_for_another_model(self):
        settings = MockSettings()
        llm = OllamaLLM(settings, context_window=1024)
        first = llm.prepare_structured_chat(
            [Message(role="user", content="short request")],
            max_tokens=64,
            response_format={"type": "object"},
            provider_request_recorder=lambda _: None,
        )
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_ollama_response(
                    content='{"changes":[]}',
                    prompt_eval_count=25,
                )
            )
            llm.resume_structured_chat(first, expected_output_budget=64)

        with patch("httpx.Client") as mock_client:
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat(
                    [Message(role="user", content="x" * 1400)],
                    model="another-model",
                    max_tokens=64,
                    response_format={"type": "object"},
                    provider_request_recorder=lambda _: None,
                )

        assert exc_info.value.diagnostics["failure_classification"] == (
            "provider_request_context_budget_exceeded"
        )
        mock_client.assert_not_called()

    def test_bound_observed_prompt_tokens_allow_exact_smaller_context(self):
        settings = MockSettings()
        schema = {"type": "object"}
        messages = [{"role": "user", "content": "x" * 256}]
        payload = {
            "model": "llama3",
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 64,
                "num_ctx": 32768,
            },
            "think": False,
            "format": schema,
        }
        identity = _context_request_identity_sha256(payload)
        llm = OllamaLLM(
            settings,
            context_window=128,
            observed_prompt_tokens=32,
            observed_prompt_identity_sha256=identity,
            observed_prompt_token_source="bound_test_observation",
        )
        recorded = {}

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = (
                make_ollama_response(content='{"changes":[]}')
            )
            llm.chat(
                [Message(role="user", content="x" * 256)],
                max_tokens=64,
                response_format=schema,
                provider_request_recorder=recorded.update,
            )

        assert recorded["context_admission"] == {
            "mode": "bound_observed_prompt_tokens_plus_output_budget_v1",
            "exact_prompt_tokens_available_pre_call": True,
            "provider_body_utf8_bytes": recorded["body_size_bytes"],
            "observed_prompt_tokens": 32,
            "prompt_identity_sha256": identity,
            "prompt_token_source": "bound_test_observation",
            "configured_num_ctx": 128,
            "num_predict": 64,
            "observed_required_context": 96,
            "admitted": True,
        }

    def test_bound_observed_prompt_tokens_block_identity_drift(self):
        settings = MockSettings()
        llm = OllamaLLM(
            settings,
            context_window=128,
            observed_prompt_tokens=32,
            observed_prompt_identity_sha256="0" * 64,
            observed_prompt_token_source="bound_test_observation",
        )
        recorded = []

        with patch("httpx.Client") as mock_client:
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat(
                    [Message(role="user", content="x" * 256)],
                    max_tokens=64,
                    response_format={"type": "object"},
                    provider_request_recorder=recorded.append,
                )

        assert exc_info.value.diagnostics["failure_classification"] == (
            "provider_request_prompt_identity_mismatch"
        )
        assert recorded == []
        mock_client.assert_not_called()

    def test_structured_patch_request_persistence_failure_prevents_http_call(
        self,
        llm,
    ):
        def fail_record(_):
            raise OSError("attempt persistence failed")

        with patch("httpx.Client") as mock_client:
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat(
                    [Message(role="user", content="return a patch")],
                    response_format={"type": "object"},
                    provider_request_recorder=fail_record,
                )

        assert exc_info.value.diagnostics["failure_classification"] == (
            "durable_provider_request_failure"
        )
        mock_client.return_value.__enter__.return_value.post.assert_not_called()

    @pytest.mark.parametrize(
        ("top_level_updates", "message_updates", "classification"),
        [
            ({"done": False}, {}, "provider_response_incomplete"),
            (
                {"done_reason": "length"},
                {},
                "provider_response_stop_reason_rejected",
            ),
            (
                {"model": "another-model"},
                {},
                "response_model_identity_mismatch",
            ),
            ({}, {"content": "   "}, "provider_message_content_empty"),
            ({}, {"thinking": "hidden"}, "provider_thinking_not_admitted"),
            (
                {},
                {"tool_calls": [{"function": {"name": "x"}}]},
                "provider_tool_calls_not_admitted",
            ),
        ],
    )
    def test_structured_patch_response_profile_fails_closed(
        self,
        llm,
        top_level_updates,
        message_updates,
        classification,
    ):
        response = make_ollama_response(content='{"changes":[]}')
        response.json.return_value.update(top_level_updates)
        response.json.return_value["message"].update(message_updates)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = response
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat(
                    [Message(role="user", content="return a patch")],
                    response_format={"type": "object"},
                    provider_request_recorder=lambda _: None,
                )

        assert exc_info.value.diagnostics["failure_boundary"] == (
            "provider_response_admission"
        )
        assert exc_info.value.diagnostics["failure_classification"] == classification
        assert "thinking" not in exc_info.value.diagnostics
        assert "tool_calls" not in exc_info.value.diagnostics

    def test_provider_envelope_json_decoding_is_separate_admission_failure(
        self,
        llm,
    ):
        response = MagicMock()
        response.status_code = 200
        response.json.side_effect = ValueError("invalid envelope")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = response
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat(
                    [Message(role="user", content="return a patch")],
                    response_format={"type": "object"},
                    provider_request_recorder=lambda _: None,
                )

        assert exc_info.value.diagnostics["failure_classification"] == (
            "provider_envelope_json_decoding_failure"
        )

    def test_structured_success_without_usage_remains_admissible(self, llm):
        response = make_ollama_response(content='{"changes":[]}')
        response.json.return_value.pop("prompt_eval_count")
        response.json.return_value.pop("eval_count")

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = response
            result = llm.chat(
                [Message(role="user", content="return a patch")],
                response_format={"type": "object"},
                provider_request_recorder=lambda _: None,
            )

        assert result.usage is None


# -----------------------------------------------------------------------------
# Validation Tests
# -----------------------------------------------------------------------------


class TestOllamaValidation:
    """Tests for input validation."""
    
    @pytest.fixture
    def llm(self):
        """Create an OllamaLLM instance for testing."""
        settings = MockSettings()
        return OllamaLLM(settings)
    
    def test_chat_empty_messages_raises(self, llm):
        """Should raise ValueError for empty messages list."""
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            llm.chat([])
    
    def test_chat_invalid_role_raises(self, llm):
        """Should raise ValueError for invalid message role."""
        with pytest.raises(ValueError, match="invalid role"):
            llm.chat([Message(role="invalid", content="Hello")])
    
    def test_chat_empty_content_raises(self, llm):
        """Should raise ValueError for empty message content."""
        with pytest.raises(ValueError, match="empty content"):
            llm.chat([Message(role="user", content="")])


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestOllamaErrorHandling:
    """Tests for error handling scenarios."""
    
    @pytest.fixture
    def llm(self):
        """Create an OllamaLLM instance for testing."""
        settings = MockSettings()
        return OllamaLLM(settings)
    
    def test_api_error_response(self, llm):
        """Should raise OllamaLLMError on API error response."""
        mock_response = make_error_response(status_code=404, error_message="model not found")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "HTTP 404" in str(exc_info.value)
            assert "model not found" in str(exc_info.value)
    
    def test_timeout_error(self, llm):
        """Should raise OllamaLLMError on timeout with helpful message."""
        import httpx
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException("timeout")
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "timed out" in str(exc_info.value).lower()
            assert "120" in str(exc_info.value)  # Default timeout value
    
    def test_connection_error(self, llm):
        """Should raise OllamaLLMError on connection failure with helpful message."""
        import httpx
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = httpx.ConnectError("Connection refused")
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            error_msg = str(exc_info.value)
            assert "Connection failed" in error_msg
            assert "ollama serve" in error_msg.lower()  # Helpful hint
    
    def test_request_error(self, llm):
        """Should raise OllamaLLMError on general request failure."""
        import httpx
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = httpx.RequestError("Network error")
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "[Ollama]" in str(exc_info.value)
    
    def test_error_does_not_leak_sensitive_info(self, llm):
        """Error messages should not expose internal URLs or config details."""
        mock_response = make_error_response(status_code=500, error_message="internal error")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            error_msg = str(exc_info.value)
            # Should not contain the full internal URL
            assert "localhost:11434" not in error_msg
    
    def test_unexpected_response_format(self, llm):
        """Should handle unexpected response format gracefully."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"unexpected": "format"}
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = response
            
            with pytest.raises(OllamaLLMError) as exc_info:
                llm.chat([Message(role="user", content="Hello")])
            
            assert "Unexpected response format" in str(exc_info.value)


# -----------------------------------------------------------------------------
# API URL Construction Tests
# -----------------------------------------------------------------------------


class TestOllamaAPIEndpoint:
    """Tests for API endpoint URL construction."""
    
    def test_api_url_construction(self):
        """Should construct correct API URL."""
        settings = MockSettings()
        llm = OllamaLLM(settings, base_url="http://localhost:11434")
        
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")])
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_args.args[0] == "http://localhost:11434/api/chat"
    
    def test_api_url_strips_trailing_slash(self):
        """Should handle base URL with trailing slash."""
        settings = MockSettings()
        llm = OllamaLLM(settings, base_url="http://localhost:11434/")
        
        mock_response = make_ollama_response()
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            llm.chat([Message(role="user", content="Hello")])
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            assert call_args.args[0] == "http://localhost:11434/api/chat"


# -----------------------------------------------------------------------------
# Model Override Tests
# -----------------------------------------------------------------------------


class TestOllamaModelOverride:
    """Tests for model override functionality."""
    
    def test_model_override_in_chat(self):
        """Should allow model override per chat call."""
        settings = MockSettings(llm=MockLLMSettings(model="llama3"))
        llm = OllamaLLM(settings)
        
        mock_response = make_ollama_response(model="mistral")
        
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            
            response = llm.chat(
                [Message(role="user", content="Hello")],
                model="mistral",
            )
            
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["model"] == "mistral"
            assert response.model == "mistral"
