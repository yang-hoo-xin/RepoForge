"""MCP Protocol Handler for JSON-RPC 2.0 message handling.

This module provides the ProtocolHandler class that encapsulates:
- Tool registration and schema management
- JSON-RPC error code handling
- Capability negotiation during initialize
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from jsonschema import Draft202012Validator, validators
from mcp import types
from mcp.server.lowlevel import Server

from src.mcp_server.argument_feedback import validation_feedback
from src.mcp_server.safety import Redactor
from src.observability.logger import get_logger

StrictInputValidator = validators.extend(Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine("integer", lambda checker, value: type(value) is int))


# JSON-RPC 2.0 Error Codes
class JSONRPCErrorCodes:
    """Standard JSON-RPC 2.0 error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]
    output_schema: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class ProtocolHandler:
    """Handles MCP protocol operations including tool registration and execution.

    This class encapsulates:
    - Tool registration with schema validation
    - Tool execution with error handling
    - Capability declaration for initialize response

    Attributes:
        server_name: Name of the MCP server.
        server_version: Version string of the server.
        tools: Registry of available tools.
    """

    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    redactor: Any = field(default_factory=Redactor)

    def __post_init__(self) -> None:
        """Initialize logger after dataclass initialization."""
        self._logger = get_logger(log_level="INFO")

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
        output_schema: Optional[Dict[str, Any]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool with the protocol handler.

        Args:
            name: Unique name for the tool.
            description: Human-readable description of what the tool does.
            input_schema: JSON Schema for the tool's input parameters.
            handler: Async function that executes the tool logic.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")

        input_schema = deepcopy(input_schema)
        input_schema.setdefault("additionalProperties", False)
        Draft202012Validator.check_schema(input_schema)
        if output_schema is not None:
            Draft202012Validator.check_schema(output_schema)
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler, output_schema=output_schema, annotations=annotations,
        )
        self._logger.info("Registered tool: %s", name)

    def get_tool_schemas(self) -> List[types.Tool]:
        """Get list of tool schemas for tools/list response.

        Returns:
            List of Tool objects with name, description, and inputSchema.
        """
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
                outputSchema=tool.output_schema,
                annotations=types.ToolAnnotations(**tool.annotations) if tool.annotations is not None else None,
            )
            for tool in self.tools.values()
        ]

    async def execute_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Execute a registered tool by name.

        Args:
            name: Name of the tool to execute.
            arguments: Arguments to pass to the tool handler.

        Returns:
            CallToolResult with content blocks or error indication.

        Raises:
            ValueError: If tool is not found.
        """
        if name not in self.tools:
            return self.error_result("UNKNOWN_TOOL", "Tool not found")
        tool = self.tools[name]
        feedback = validation_feedback(arguments, tool.input_schema, StrictInputValidator)
        if feedback is not None:
            return self.error_result("INVALID_PARAMS", "Invalid parameters", details=feedback)
        try:
            result = await tool.handler(**arguments)
            if isinstance(result, types.CallToolResult):
                raw = result.model_dump(mode="json", exclude_none=True)
                payload = self.redactor(raw)
                structured = payload.get("structuredContent", {})
                if raw != payload and structured.get("schema_version") == "query_response_v1":
                    structured["warnings"].append("MCP_SECRET_REDACTION_APPLIED")
                    payload["content"] = [{"type": "text", "text": json.dumps(structured, ensure_ascii=False)}]
                normalized = types.CallToolResult.model_validate(payload)
            elif isinstance(result, dict):
                payload = self.redactor(result)
                normalized = types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
                                                  structuredContent=payload, isError=False)
            elif isinstance(result, str):
                normalized = types.CallToolResult(content=[types.TextContent(type="text", text=self.redactor(result))], isError=False)
            else:
                normalized = types.CallToolResult(content=result, isError=False)
            if tool.output_schema is not None and not Draft202012Validator(tool.output_schema).is_valid(normalized.structuredContent):
                return self.error_result("OUTPUT_CONTRACT_INVALID", "Tool output contract failed")
            return normalized
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._logger.error("Tool execution failed: %s (%s)", name, type(error).__name__)
            return self.error_result("INTERNAL_ERROR", "Internal server error")

    def error_result(self, code, message, *, details=None):
        payload = {"schema_version": "mcp_error_v1", "status": "error", "request_id": uuid4().hex,
                   "error": {"code": code, "message": message}}
        if details is not None:
            payload["error"]["details"] = details
        # Scope: both text and structured error views share the output redaction boundary.
        # 为什么要有它：纠正反馈不能回显凭据，两个客户端可见表示也不能因脱敏而互相矛盾。
        payload = self.redactor(payload)
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(payload))],
                                    structuredContent=payload, isError=True)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities for initialize response.

        Returns:
            Dictionary of server capabilities.
        """
        return {
            "tools": {} if self.tools else {},
        }


def _register_default_tools(protocol_handler: ProtocolHandler) -> None:
    """Register all default MCP tools with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register tools with.
    """
    from src.mcp_server.registered_tools import register_tools
    register_tools(protocol_handler)


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
    settings=None,
    artifact_store=None,
) -> Server:
    """Create and configure an MCP server with the protocol handler.

    This factory function creates a low-level MCP Server instance and
    registers the necessary handlers for tools/list and tools/call.

    Args:
        server_name: Name of the server.
        server_version: Version string.
        protocol_handler: Optional pre-configured protocol handler.
            If None, a new one will be created.
        register_tools: Whether to register default tools (default: True).

    Returns:
        Configured Server instance ready to run.
    """
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            server_version=server_version,
        )

    # Register default tools if requested
    if register_tools:
        from src.mcp_server.registered_tools import register_tools as register_selected_tools
        register_selected_tools(protocol_handler, settings=settings, artifact_store=artifact_store)

    # Create low-level server
    server = Server(server_name)

    # Register tools/list handler
    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        """Handle tools/list request."""
        return protocol_handler.get_tool_schemas()

    # Register tools/call handler
    @server.call_tool(validate_input=False)
    async def handle_call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> types.CallToolResult:
        """Handle tools/call request."""
        return await protocol_handler.execute_tool(name, arguments)

    # Store protocol handler on server for access
    server._protocol_handler = protocol_handler  # type: ignore[attr-defined]

    return server


def get_protocol_handler(server: Server) -> ProtocolHandler:
    """Get the protocol handler from a server instance.

    Args:
        server: Server instance created by create_mcp_server.

    Returns:
        The ProtocolHandler associated with the server.

    Raises:
        AttributeError: If server was not created with create_mcp_server.
    """
    return server._protocol_handler  # type: ignore[attr-defined]
