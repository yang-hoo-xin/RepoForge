import asyncio
import json
import logging
import threading
from types import SimpleNamespace

import pytest
from jsonschema import validate

from src.core.settings import load_settings
from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.registered_tools import ERROR_SCHEMA, register_tools
from src.mcp_server.safety import Redactor, install_log_redaction
from src.mod_agent.artifacts import ArtifactStore


@pytest.mark.parametrize("arguments", [{}, {"number": True}, {"number": 1.0}, {"number": "1"}, {"number": 1, "unknown": "CANARY"}])
def test_invalid_arguments_never_enter_handler(arguments):
    async def handler(**kwargs):
        pytest.fail("invalid arguments reached handler")
    protocol = ProtocolHandler("test", "1")
    protocol.register_tool("number", "number", {"type": "object", "required": ["number"],
        "properties": {"number": {"type": "integer"}}}, handler)
    result = asyncio.run(protocol.execute_tool("number", arguments))
    assert result.isError and result.structuredContent["error"]["code"] == "INVALID_PARAMS"
    validate(result.structuredContent, ERROR_SCHEMA)
    assert "CANARY" not in result.model_dump_json()


def test_invalid_argument_feedback_supports_correction_without_executing_or_echoing_input():
    invoked = []

    async def handler(number):
        invoked.append(number)
        return {"number": number}

    protocol = ProtocolHandler("test", "1")
    protocol.register_tool("number", "number", {"type": "object", "required": ["number"],
        "properties": {"number": {"type": "integer", "minimum": 1, "maximum": 20}}}, handler)
    invalid = asyncio.run(protocol.execute_tool("number", {"number": "private-input-value", "private-input-key": "secret"}))
    feedback = invalid.structuredContent["error"]["details"]
    assert feedback["execution"] == "not_executed" and feedback["retryable"] is True
    assert any(issue["path"] == "/number" and issue["code"] == "type"
               and issue["expected"]["type"] == "integer" for issue in feedback["issues"])
    assert any(issue["code"] == "additional_properties" for issue in feedback["issues"])
    assert not invoked
    assert "private-input" not in invalid.model_dump_json()
    validate(invalid.structuredContent, ERROR_SCHEMA)
    assert json.loads(invalid.content[0].text) == invalid.structuredContent
    corrected = asyncio.run(protocol.execute_tool("number", {"number": 3}))
    assert not corrected.isError and corrected.structuredContent == {"number": 3}
    assert invoked == [3]


def test_internal_type_error_not_misclassified_or_exposed(caplog):
    marker = "synthetic-secret-canary-123456"
    async def handler():
        raise TypeError(marker)
    protocol = ProtocolHandler("test", "1", redactor=Redactor([marker]))
    protocol.register_tool("broken", "broken", {"type": "object", "properties": {}}, handler)
    result = asyncio.run(protocol.execute_tool("broken", {}))
    assert result.structuredContent["error"]["code"] == "INTERNAL_ERROR"
    assert marker not in result.model_dump_json() + caplog.text


def test_structured_content_and_log_redaction(tmp_path, caplog):
    marker = "synthetic-secret-canary-123456"
    redactor = Redactor([marker])
    old = install_log_redaction(redactor)
    try:
        logging.getLogger("mcp-canary").warning("secret=%s", marker)
    finally:
        logging.setLogRecordFactory(old)
    async def handler():
        return {"text": marker}
    protocol = ProtocolHandler("test", "1", redactor=redactor)
    protocol.register_tool("text", "text", {"type": "object"}, handler,
        output_schema={"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}})
    result = asyncio.run(protocol.execute_tool("text", {}))
    assert result.structuredContent["text"] == "[REDACTED]"
    assert marker not in result.model_dump_json() + caplog.text


def test_artifact_tools_registered_to_actual_sdk_contract(tmp_path):
    protocol = ProtocolHandler("test", "1")
    register_tools(protocol, settings=load_settings("config/settings.yaml"), artifact_store=ArtifactStore(tmp_path))
    schemas = protocol.get_tool_schemas()
    assert len(schemas) == 11
    assert all(tool.outputSchema is not None and tool.inputSchema["additionalProperties"] is False for tool in schemas)
    assert not {"apply_patch", "run_build", "run_game"} & {tool.name for tool in schemas}
    result = asyncio.run(protocol.execute_tool("list_features", {}))
    assert result.structuredContent["schema_version"] == "artifact_read_v1"
    assert not result.isError
    validate(result.structuredContent, next(tool.outputSchema for tool in schemas if tool.name == "list_features"))
    assert json.loads(result.content[0].text) == result.structuredContent


def test_cancelled_mcp_await_signals_inflight_owner_without_claiming_provider_stop(monkeypatch):
    from src.mcp_server import registered_tools
    started, stopped = threading.Event(), threading.Event()
    class Owner:
        trace_collector = None
        def query(self, request, *, cancel_event):
            started.set()
            assert cancel_event.wait(timeout=3)
            stopped.set()
            return SimpleNamespace(to_dict=lambda: {"status": "cancelled"})
    monkeypatch.setattr(registered_tools, "create_query_service", lambda *_: Owner())
    monkeypatch.setattr(registered_tools.ListCollectionsTool, "list_collections", lambda *_: [SimpleNamespace(name="default")])
    protocol = ProtocolHandler("test", "1")
    register_tools(protocol, settings=load_settings("config/settings.yaml"))
    async def run():
        task = asyncio.create_task(protocol.execute_tool("query_knowledge_hub", {"query": "question", "collection": "default"}))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await asyncio.to_thread(stopped.wait, 2)
    asyncio.run(run())


def test_probe_rejects_nonprotocol_stdout(tmp_path):
    import sys

    from scripts.probe_mcp_stdio import probe_stdio
    result = probe_stdio(command=sys.executable, arguments=["-c", "print('not-json-protocol')"], calls=[],
                         output_root=tmp_path / "probe", timeout_seconds=5)
    assert result["status"] == "failed" and result["protocol_error"] == "NON_JSONRPC_STDOUT"
    assert result["all_processes_exited"]
