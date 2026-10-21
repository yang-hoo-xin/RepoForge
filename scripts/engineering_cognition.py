"""Read-only, reproducible engineering-cognition projections for this repository.

The command deliberately builds views from current source, registries, Git, and
the bounded task packet on every invocation.  It never creates an Artifact,
cache, database, snapshot, or other durable state.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "engineering_cognition_projection_v0"
TASK_PACKET_PATH = Path(".codex/task-packet.json")
HOOK_CONFIG_PATH = Path(".codex/hooks.json")
HOOK_RUNTIME_PATH = Path(".codex/.runtime/context-snapshots")

VIEW_NAMES = (
    "summary",
    "current",
    "surfaces",
    "workflow",
    "artifacts",
    "ownership",
    "source-dependency",
    "api-usage",
    "behavior-dataflow",
    "verification-claims",
    "patch-execution",  # Compatibility alias for a Verification/Claim subquery.
    "product-api",
    "all",
)
ENTITY_KINDS = ("symbol", "artifact", "route", "state-field")

PRODUCT_API_QUESTIONS = (
    "read",
    "write",
    "call",
    "subscription",
    "lifecycle",
    "accessibility",
    "Harmony target",
    "parameters",
    "caller",
    "return propagation",
    "version",
    "signature",
    "legacy compatibility",
)
MAX_GIT_STATUS_LINES = 100

_PRODUCT_API_SYMBOLS = (
    "EvidenceDiscoverySpecialist",
    "LocalAssemblyEvidenceToolExecutor",
    "RunLocalMemberAccessResolver",
)


class EngineeringCognitionError(RuntimeError):
    """Raised when a requested projection cannot be built exactly."""


def repository_root() -> Path:
    """Return the repository that owns this script."""

    return Path(__file__).resolve().parents[1]


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            return path.as_posix()


def _source_ref(path: Path, line: int, root: Path) -> str:
    return f"{_relative(path, root)}:{line}"


@cache
def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise EngineeringCognitionError(f"Unable to read {_relative(path, path.parent)}") from exc


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


@cache
def _parse_python(path: Path) -> ast.Module:
    try:
        return ast.parse(_read_text(path), filename=str(path))
    except SyntaxError as exc:
        raise EngineeringCognitionError(f"Unable to parse {path.as_posix()}") from exc


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            env=environment,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineeringCognitionError("Git facts are unavailable.") from exc


def _git_facts(root: Path) -> dict[str, Any]:
    head_result = _run_git(root, "rev-parse", "--verify", "HEAD")
    branch_result = _run_git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    status_result = _run_git(root, "status", "--short", "--untracked-files=all")
    if head_result.returncode != 0:
        return {
            "available": False,
            "branch": None,
            "dirty": None,
            "head_sha": None,
            "status_count": None,
            "status_short": [],
            "status_truncated": False,
        }
    all_status_lines = status_result.stdout.splitlines() if status_result.returncode == 0 else []
    status_lines = all_status_lines[:MAX_GIT_STATUS_LINES]
    return {
        "available": status_result.returncode == 0,
        "branch": branch_result.stdout.strip() if branch_result.returncode == 0 else None,
        "dirty": bool(all_status_lines) if status_result.returncode == 0 else None,
        "head_sha": head_result.stdout.strip(),
        "status_count": len(all_status_lines) if status_result.returncode == 0 else None,
        "status_short": status_lines,
        "status_truncated": len(all_status_lines) > len(status_lines),
    }


def _task_packet_projection(root: Path, current_git: Mapping[str, Any]) -> dict[str, Any]:
    path = root / TASK_PACKET_PATH
    raw = _read_json(path) if path.is_file() else None
    if not isinstance(raw, Mapping):
        return {
            "context_role": "context_only",
            "exists": path.is_file(),
            "freshness": "unknown",
            "reason": "missing_or_invalid_task_packet",
            "path": TASK_PACKET_PATH.as_posix(),
            "stored_base_sha": None,
            "stored_head_sha": None,
            "task_id": None,
        }

    git_snapshot = raw.get("git_snapshot")
    stored_head = git_snapshot.get("head_sha") if isinstance(git_snapshot, Mapping) else None
    stored_head = stored_head if isinstance(stored_head, str) else None
    stored_base = raw.get("base_sha")
    stored_base = stored_base if isinstance(stored_base, str) else None
    stored_branch = git_snapshot.get("branch") if isinstance(git_snapshot, Mapping) else None
    stored_branch = stored_branch if isinstance(stored_branch, str) else None
    stored_status = git_snapshot.get("status_short") if isinstance(git_snapshot, Mapping) else None
    stored_status = (
        [item for item in stored_status if isinstance(item, str)]
        if isinstance(stored_status, list)
        else None
    )
    stored_clean = git_snapshot.get("is_clean") if isinstance(git_snapshot, Mapping) else None
    stored_clean = stored_clean if isinstance(stored_clean, bool) else None
    current_head = current_git.get("head_sha")

    if current_head is None:
        freshness = "unknown"
        reason = "current_head_unavailable"
    elif stored_head is None:
        freshness = "unknown"
        reason = "stored_head_unavailable"
    elif stored_head != current_head:
        freshness = "stale"
        reason = "head_changed"
    elif stored_branch is not None and stored_branch != current_git.get("branch"):
        freshness = "stale"
        reason = "branch_changed"
    elif stored_status is not None and stored_status != current_git.get("status_short"):
        freshness = "stale"
        reason = "status_changed"
    elif stored_clean is False or current_git.get("dirty") is True:
        freshness = "unknown"
        reason = "dirty_content_not_compared"
    elif stored_status is None or stored_clean is None:
        freshness = "unknown"
        reason = "stored_status_unavailable"
    else:
        freshness = "current"
        reason = "clean_git_identity_matches"

    task_id = raw.get("task_id")
    return {
        "context_role": "context_only",
        "exists": True,
        "freshness": freshness,
        "goal": raw.get("goal") if isinstance(raw.get("goal"), str) else None,
        "path": TASK_PACKET_PATH.as_posix(),
        "reason": reason,
        "stored_base_sha": stored_base,
        "stored_branch": stored_branch,
        "stored_clean": stored_clean,
        "stored_head_sha": stored_head,
        "stored_status_count": len(stored_status) if stored_status is not None else None,
        "task_id": task_id if isinstance(task_id, str) else None,
    }


def _hook_events(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    hooks = raw.get("hooks")
    if not isinstance(hooks, Mapping):
        return []
    events: list[dict[str, Any]] = []
    for event_name, groups in sorted(hooks.items(), key=lambda item: str(item[0])):
        if not isinstance(event_name, str) or not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            commands = group.get("hooks")
            if not isinstance(commands, list):
                continue
            for command in commands:
                if not isinstance(command, Mapping):
                    continue
                events.append(
                    {
                        "additional_context_limit": command.get("additionalContextLimit"),
                        "event": event_name,
                        "matcher": group.get("matcher"),
                        "timeout_seconds": command.get("timeout"),
                        "type": command.get("type"),
                        "windows_command_configured": isinstance(
                            command.get("commandWindows"), str
                        ),
                    }
                )
    return sorted(events, key=lambda item: (str(item["event"]), str(item["matcher"])))


def _hook_projection(root: Path) -> dict[str, Any]:
    config_path = root / HOOK_CONFIG_PATH
    raw = _read_json(config_path) if config_path.is_file() else None
    events = _hook_events(raw) if isinstance(raw, Mapping) else []
    runtime_dir = root / HOOK_RUNTIME_PATH
    snapshot_files = sorted(runtime_dir.glob("*.json")) if runtime_dir.is_dir() else []
    valid_snapshots: list[Path] = []
    for path in snapshot_files:
        value = _read_json(path)
        if (
            isinstance(value, Mapping)
            and value.get("schema_version") == "codex_context_snapshot_v0"
            and isinstance(value.get("session"), Mapping)
            and isinstance(value.get("git"), Mapping)
        ):
            valid_snapshots.append(path)
    event_names = {str(event["event"]) for event in events}
    return {
        "configuration": {
            "configured": bool(events),
            "events": events,
            "path": HOOK_CONFIG_PATH.as_posix(),
            "pre_compact_configured": "PreCompact" in event_names,
            "session_start_configured": "SessionStart" in event_names,
            "shape_valid": bool(events) and isinstance(raw, Mapping),
        },
        "lifecycle_acceptance": {
            "reason": "no_durable_session_start_acceptance_receipt_is_defined",
            "status": "not_observed",
        },
        "runtime_snapshot": {
            "invalid_snapshot_count": len(snapshot_files) - len(valid_snapshots),
            "observed": bool(valid_snapshots),
            "path": HOOK_RUNTIME_PATH.as_posix(),
            "snapshot_count": len(snapshot_files),
            "valid_snapshot_count": len(valid_snapshots),
        },
    }


def project_current(root: Path) -> dict[str, Any]:
    """Project current Git, task-location, and Hook facts without mutation."""

    git = _git_facts(root)
    return {
        "git": git,
        "hooks": _hook_projection(root),
        "task_packet": _task_packet_projection(root, git),
    }


def _literal_assignment(module: ast.Module, name: str) -> Any:
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        try:
            return ast.literal_eval(node.value)
        except (TypeError, ValueError):
            return None
    return None


def _calls_with_string_argument(path: Path, attribute: str, root: Path) -> list[dict[str, Any]]:
    module = _parse_python(path)
    found: list[dict[str, Any]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != attribute or not node.args:
            continue
        value = node.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.append({"name": value.value, "source_ref": _source_ref(path, node.lineno, root)})
    return sorted(found, key=lambda item: str(item["name"]))


def _standard_mcp_tools(root: Path) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for path in sorted((root / "src/mcp_server/tools").glob("*.py")):
        value = _literal_assignment(_parse_python(path), "TOOL_NAME")
        if isinstance(value, str):
            line = next(
                (
                    node.lineno
                    for node in _parse_python(path).body
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and any(
                        isinstance(target, ast.Name) and target.id == "TOOL_NAME"
                        for target in (
                            node.targets if isinstance(node, ast.Assign) else [node.target]
                        )
                    )
                ),
                1,
            )
            tools.append({"name": value, "source_ref": _source_ref(path, line, root)})
    return sorted(tools, key=lambda item: str(item["name"]))


def _controlled_mcp_tools(root: Path) -> list[dict[str, Any]]:
    path = root / "src/mod_agent/mcp/tools.py"
    module = _parse_python(path)
    values = _literal_assignment(module, "CONTROLLED_AGENT_TOOL_NAMES")
    if not isinstance(values, tuple) or not all(isinstance(value, str) for value in values):
        return []
    line = next(
        (
            node.lineno
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CONTROLLED_AGENT_TOOL_NAMES"
                for target in node.targets
            )
        ),
        1,
    )
    return [
        {"name": value, "source_ref": _source_ref(path, line, root)} for value in sorted(values)
    ]


def _http_get_routes(root: Path) -> list[dict[str, Any]]:
    path = root / "src/mod_agent/api/http_server.py"
    module = _parse_python(path)
    handler = next(
        (
            node
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "handle_get"
        ),
        None,
    )
    if handler is None:
        return []
    routes: dict[str, int] = {}
    dynamic_actions: dict[str, int] = {}
    for node in ast.walk(handler):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        comparator = node.comparators[0]
        if isinstance(node.left, ast.Name) and node.left.id == "parts":
            try:
                parts = ast.literal_eval(comparator)
            except (TypeError, ValueError):
                parts = None
            if isinstance(parts, tuple) and parts and all(isinstance(item, str) for item in parts):
                routes["/" + "/".join(parts)] = node.lineno
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == "action"
            and isinstance(comparator, ast.Constant)
            and isinstance(comparator.value, str)
        ):
            dynamic_actions[comparator.value] = node.lineno
    for action, line in dynamic_actions.items():
        routes[f"/features/{{feature_id}}/{action}"] = line
    return [
        {"method": "GET", "path": route, "source_ref": _source_ref(path, line, root)}
        for route, line in sorted(routes.items())
    ]


def _dashboard_pages(root: Path) -> list[dict[str, Any]]:
    path = root / "src/observability/dashboard/app.py"
    module = _parse_python(path)
    pages: list[dict[str, Any]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "st"
            and node.func.attr == "Page"
        ):
            continue
        title = next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "title"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            None,
        )
        if title is not None:
            pages.append({"name": title, "source_ref": _source_ref(path, node.lineno, root)})
    return sorted(pages, key=lambda item: str(item["name"]))


def _project_scripts(root: Path) -> dict[str, str]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    in_scripts = False
    result: dict[str, str] = {}
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            continue
        if not in_scripts or not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)\s*=\s*\"([^\"]+)\"",
            line,
        )
        if match is not None:
            result[match.group(1)] = match.group(2)
    return dict(sorted(result.items()))


def _python_entrypoint(root: Path, path: Path, function_name: str) -> str | None:
    definitions = {
        node.name
        for node in ast.walk(_parse_python(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function_name not in definitions:
        return None
    relative = path.relative_to(root).with_suffix("")
    return f"{'.'.join(relative.parts)}:{function_name}"


def project_surfaces(root: Path) -> dict[str, Any]:
    """Project the registered entry surfaces; this is not a whole-project API claim."""

    cli_path = root / "src/mod_agent/cli.py"
    cli_commands = _calls_with_string_argument(cli_path, "add_parser", root)
    standard_tools = _standard_mcp_tools(root)
    controlled_tools = _controlled_mcp_tools(root)
    http_routes = _http_get_routes(root)
    dashboard_pages = _dashboard_pages(root)
    server_path = root / "src/mcp_server/server.py"
    server_text = _read_text(server_path)
    controlled_protocol_text = _read_text(root / "src/mod_agent/mcp/protocol.py")
    http_text = _read_text(root / "src/mod_agent/api/http_server.py")
    dashboard_launcher = root / "scripts/start_dashboard.py"
    project_scripts = _project_scripts(root)
    product_cli_implementation = _python_entrypoint(root, cli_path, "main")
    standard_mcp_implementation = _python_entrypoint(root, server_path, "main")
    product_cli_package_entry = project_scripts.get("mod-agent")
    standard_mcp_package_entry = project_scripts.get("mcp-server")
    return {
        "scope": "registered_entry_surfaces_not_all_internal_apis",
        "surfaces": [
            {
                "entries": cli_commands,
                "entry_count": len(cli_commands),
                "entrypoint_alignment": (
                    "match"
                    if product_cli_package_entry == product_cli_implementation
                    else "mismatch"
                ),
                "implementation_entrypoint": product_cli_implementation,
                "maturity": "implemented_parser",
                "package_entrypoint": product_cli_package_entry,
                "surface": "product_cli",
            },
            {
                "entries": standard_tools,
                "entry_count": len(standard_tools),
                "entrypoint_alignment": (
                    "match"
                    if standard_mcp_package_entry == standard_mcp_implementation
                    else "mismatch"
                ),
                "implementation_entrypoint": standard_mcp_implementation,
                "installed_entry_health": "not_inferred",
                "maturity": (
                    "implemented_stdio_source"
                    if "mcp.server.stdio.stdio_server" in server_text
                    else "transport_not_observed"
                ),
                "package_entrypoint": standard_mcp_package_entry,
                "surface": "standard_rag_mcp",
            },
            {
                "entries": controlled_tools,
                "entry_count": len(controlled_tools),
                "maturity": (
                    "implemented_in_process_json_rpc"
                    if 'request.get("jsonrpc")' in controlled_protocol_text
                    else "protocol_not_observed"
                ),
                "surface": "controlled_agent_mcp",
                "transport_limit": "no_server_or_cli_entry_observed",
            },
            {
                "entries": http_routes,
                "entry_count": len(http_routes),
                "maturity": (
                    "implemented_local_dev_library"
                    if '"deployment": "local_dev_only"' in http_text
                    else "deployment_boundary_not_observed"
                ),
                "surface": "http_get",
                "transport_limit": "no_repository_launcher_observed",
            },
            {
                "entries": dashboard_pages,
                "entry_count": len(dashboard_pages),
                "maturity": (
                    "implemented_streamlit_launcher"
                    if dashboard_launcher.is_file()
                    else "page_registry_only"
                ),
                "surface": "dashboard_pages",
                "effect_boundary": "page_registration_does_not_prove_read_only_behavior",
            },
        ],
        "limitations": [
            "MCP is an entry protocol surface, not the whole project API.",
            "Registration proves a declared entry, not deployment, invocation, or successful runtime use.",
        ],
    }


def _graph_nodes(root: Path) -> list[dict[str, Any]]:
    return _calls_with_string_argument(root / "src/mod_agent/workflow/graph.py", "add_node", root)


def _state_field_types(root: Path) -> dict[str, tuple[str, int]]:
    path = root / "src/mod_agent/workflow/state.py"
    module = _parse_python(path)
    state_class = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "ModAgentState"
        ),
        None,
    )
    if state_class is None:
        return {}
    result: dict[str, tuple[str, int]] = {}
    for node in state_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = (ast.unparse(node.annotation), node.lineno)
    return result


def _static_value(node: ast.AST | None) -> Any:
    """Evaluate only literal containers and frozenset calls used by registries."""

    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (TypeError, ValueError):
        pass
    if isinstance(node, ast.Tuple):
        values = tuple(_static_value(item) for item in node.elts)
        return values if all(value is not None for value in values) else None
    if isinstance(node, ast.List):
        values = [_static_value(item) for item in node.elts]
        return values if all(value is not None for value in values) else None
    if isinstance(node, ast.Set):
        values = {_static_value(item) for item in node.elts}
        return values if None not in values else None
    if isinstance(node, ast.Dict):
        keys = [_static_value(item) for item in node.keys]
        values = [_static_value(item) for item in node.values]
        if all(key is not None for key in keys) and all(value is not None for value in values):
            return dict(zip(keys, values))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) <= 1
        and not node.keywords
    ):
        value = _static_value(node.args[0]) if node.args else ()
        if isinstance(value, (list, set, tuple, frozenset)):
            return frozenset(value)
    return None


def _state_writer_map(root: Path) -> dict[str, list[str]]:
    path = root / "src/mod_agent/workflow/state.py"
    module = _parse_python(path)
    policy_class = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "StateOwnershipPolicy"
        ),
        None,
    )
    if policy_class is None:
        return {}
    writers_node: ast.AST | None = None
    for node in policy_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "_WRITERS":
                writers_node = node.value
                break
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_WRITERS" for target in node.targets
        ):
            writers_node = node.value
            break
    if not isinstance(writers_node, ast.Dict):
        return {}
    result: dict[str, list[str]] = {}
    for key_node, value_node in zip(writers_node.keys, writers_node.values):
        key = _static_value(key_node)
        value = _static_value(value_node)
        if (
            isinstance(key, str)
            and isinstance(value, frozenset)
            and all(isinstance(item, str) for item in value)
        ):
            result[key] = sorted(value)
    return result


def _route_source_lines(root: Path) -> dict[str, int]:
    path = root / "src/mod_agent/workflow/transitions.py"
    module = _parse_python(path)
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "route_registry"
        ),
        None,
    )
    if function is None:
        return {}
    returned = next(
        (node.value for node in ast.walk(function) if isinstance(node, ast.Return)), None
    )
    if not isinstance(returned, ast.Dict):
        return {}
    result: dict[str, int] = {}
    for key in returned.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            result[key.value] = key.lineno
    return result


def _route_registry_value(root: Path) -> dict[str, dict[str, str]]:
    path = root / "src/mod_agent/workflow/transitions.py"
    module = _parse_python(path)
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "route_registry"
        ),
        None,
    )
    if function is None:
        return {}
    returned = next(
        (node.value for node in ast.walk(function) if isinstance(node, ast.Return)),
        None,
    )
    value = _static_value(returned)
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, str]] = {}
    for source, labels in value.items():
        if not isinstance(source, str) or not isinstance(labels, dict):
            continue
        if not all(
            isinstance(label, str) and isinstance(target, str) for label, target in labels.items()
        ):
            continue
        result[source] = dict(labels)
    return result


def _workflow_components(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    route_path = root / "src/mod_agent/workflow/transitions.py"
    route_lines = _route_source_lines(root)
    routes: list[dict[str, Any]] = []
    for source, labels in sorted(_route_registry_value(root).items()):
        for label, destination in sorted(labels.items()):
            routes.append(
                {
                    "destination": destination,
                    "label": label,
                    "source": source,
                    "source_ref": _source_ref(route_path, route_lines.get(source, 1), root),
                }
            )
    state_path = root / "src/mod_agent/workflow/state.py"
    writers = _state_writer_map(root)
    fields: list[dict[str, Any]] = []
    for field_name, (field_type, line) in sorted(_state_field_types(root).items()):
        fields.append(
            {
                "field": field_name,
                "source_ref": _source_ref(state_path, line, root),
                "type": field_type,
                "writers": writers.get(field_name, []),
            }
        )
    return _graph_nodes(root), routes, fields


def project_workflow(root: Path) -> dict[str, Any]:
    """Project Graph node declarations, route declarations, and state writers."""

    nodes, routes, fields = _workflow_components(root)
    return {
        "graph_nodes": nodes,
        "node_count": len(nodes),
        "route_count": len(routes),
        "routes": routes,
        "state_field_count": len(fields),
        "state_fields": fields,
        "limitations": [
            "A Graph node declaration does not prove that a node ran.",
            "A route declaration does not prove that a route was selected.",
            "State writers describe runtime-state mutation policy, not product-source write authority.",
        ],
    }


def _artifact_contract_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "src/mod_agent/artifacts/registry.py"
    module = _parse_python(path)
    function = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "default_artifact_contracts"
        ),
        None,
    )
    if function is None:
        return []
    rows: list[dict[str, Any]] = []
    for call in ast.walk(function):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "ArtifactContract":
            continue
        values = {
            keyword.arg: _static_value(keyword.value) for keyword in call.keywords if keyword.arg
        }
        artifact_type = values.get("artifact_type")
        schema_version = values.get("schema_version")
        producer_stage = values.get("producer_stage")
        consumer_stages = values.get("consumer_stages")
        if not (
            isinstance(artifact_type, str)
            and isinstance(schema_version, str)
            and isinstance(producer_stage, str)
            and isinstance(consumer_stages, frozenset)
        ):
            continue
        required_upstream = values.get("required_upstream_artifacts") or frozenset()
        exact_upstream = values.get("required_exact_revision_upstream_artifacts") or frozenset()
        one_of = values.get("required_one_of_upstream_artifact_groups") or ()
        rows.append(
            {
                "artifact_type": artifact_type,
                "consumer_stages": sorted(consumer_stages),
                "producer_stage": producer_stage,
                "required_exact_revision_upstream_artifacts": sorted(exact_upstream),
                "required_one_of_upstream_artifact_groups": sorted(
                    [sorted(group) for group in one_of]
                ),
                "required_upstream_artifacts": sorted(required_upstream),
                "schema_version": schema_version,
                "source_ref": _source_ref(path, call.lineno, root),
            }
        )
    return sorted(rows, key=lambda item: str(item["artifact_type"]))


def project_artifacts(root: Path) -> dict[str, Any]:
    """Project the registered producer/consumer/upstream Artifact DAG."""

    contracts = _artifact_contract_rows(root)
    return {
        "artifact_count": len(contracts),
        "contracts": contracts,
        "source_ref": _definition_ref(root, "default_artifact_contracts"),
        "limitations": [
            "A registered Artifact contract does not prove that an Artifact instance exists.",
            "An existing Artifact does not by itself prove its product claim is verified.",
        ],
    }


@cache
def _definition_index(root: Path, include_tests: bool = False) -> dict[str, tuple[str, ...]]:
    search_roots = [root / "src", root / "scripts"]
    if include_tests:
        search_roots.append(root / "tests")
    refs: dict[str, set[str]] = {}
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.rglob("*.py")):
            try:
                module = _parse_python(path)
            except EngineeringCognitionError:
                continue
            for node in ast.walk(module):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    refs.setdefault(node.name, set()).add(_source_ref(path, node.lineno, root))
    return {name: tuple(sorted(items)) for name, items in sorted(refs.items())}


def _definition_refs(root: Path, name: str, *, include_tests: bool = False) -> list[str]:
    return list(_definition_index(root, include_tests).get(name, ()))


def _definition_ref(root: Path, name: str) -> str | None:
    refs = _definition_refs(root, name)
    return refs[0] if refs else None


def _text_refs(
    root: Path,
    name: str,
    *,
    under: Path,
    limit: int = 80,
) -> dict[str, Any]:
    refs: list[str] = []
    if not under.is_dir():
        return {"refs": refs, "truncated": False}
    identifier_pattern = re.compile(rf"\b{re.escape(name)}\b") if name.isidentifier() else None
    for path in sorted(under.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(lines, 1):
            matched = bool(identifier_pattern.search(line)) if identifier_pattern else name in line
            if matched:
                refs.append(_source_ref(path, line_number, root))
    unique = sorted(set(refs))
    return {"refs": unique[:limit], "truncated": len(unique) > limit}


def _text_refs_in_paths(root: Path, name: str, paths: Sequence[Path]) -> list[str]:
    pattern = re.compile(rf"\b{re.escape(name)}\b") if name.isidentifier() else None
    refs: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            lines = _read_text(path).splitlines()
        except EngineeringCognitionError:
            continue
        for line_number, line in enumerate(lines, 1):
            matched = bool(pattern.search(line)) if pattern else name in line
            if matched:
                refs.append(_source_ref(path, line_number, root))
    return sorted(set(refs))


def _actual_evidence(root: Path, entity_name: str | None = None) -> dict[str, Any]:
    artifact_root = root / "artifacts"
    journals: list[Path] = []
    if artifact_root.is_dir():
        # 为什么要有它：本地产物目录可由并行运行清理或替换；只读工程视图应
        # 忽略一次消失的目录，而不能让历史临时文件阻断当前源码查询。
        for directory, child_directories, filenames in os.walk(
            artifact_root,
            topdown=True,
            onerror=lambda _error: None,
        ):
            child_directories.sort()
            if "actual_journal.json" in filenames:
                journals.append(Path(directory) / "actual_journal.json")
        journals.sort()
    observed: list[tuple[str, Path, Mapping[str, Any]]] = []
    for path in journals:
        raw = _read_json(path)
        if not isinstance(raw, Mapping) or raw.get("journal_version") != "actual_run_journal_v1":
            continue
        if raw.get("terminal_summary_complete") is not True:
            continue
        finished_at = raw.get("finished_at")
        observed.append((finished_at if isinstance(finished_at, str) else "", path, raw))
    if entity_name is not None:
        return {
            "binding": "not_available",
            "reason": (
                "complete_actual_journals_have_no_structured_entity_binding_projection"
                if observed
                else "no_complete_actual_journal_observed"
            ),
            "search_scope": "artifacts/**/actual_journal.json",
            "status": "unknown",
        }
    if not observed:
        return {
            "binding": "not_available",
            "reason": "no_complete_actual_journal_observed_in_artifacts_tree",
            "search_scope": "artifacts/**/actual_journal.json",
            "status": "not_observed",
        }
    _, path, raw = sorted(observed, key=lambda item: (item[0], item[1].as_posix()))[-1]
    return {
        "actual_attempt_id": raw.get("actual_attempt_id"),
        "binding": "journal_shape_only",
        "execution_verification": "not_performed_by_projection",
        "finished_at": raw.get("finished_at"),
        "journal_ref": _relative(path, root),
        "search_scope": "artifacts/**/actual_journal.json",
        "stage": raw.get("stage"),
        "status": "journal_shape_observed",
        "terminal_status": (
            raw.get("terminal", {}).get("status")
            if isinstance(raw.get("terminal"), Mapping)
            else None
        ),
    }


def project_ownership(root: Path) -> dict[str, Any]:
    """Project declared mechanical ownership and its authority limitations."""

    _, routes, fields = _workflow_components(root)
    contracts = _artifact_contract_rows(root)
    anchor_paths = (
        root / "src/mod_agent/evidence/requirement_coverage.py",
        root / "src/mod_agent/planning/implementation_plan.py",
        root / "src/mod_agent/bindings/resolver.py",
    )
    anchors = _text_refs_in_paths(root, "authorized_edit_target_ids", anchor_paths)
    return {
        "artifact_producers": [
            {"artifact_type": row["artifact_type"], "producer_stage": row["producer_stage"]}
            for row in contracts
        ],
        "coverage_planning_binding_drift_anchor": {
            "inspection_anchor": "authorized_edit_target_ids",
            "source_refs": anchors,
            "status": (
                "cross_module_reference_points_observed_semantic_owner_not_inferred"
                if anchors
                else "not_observed"
            ),
        },
        "route_declarations": routes,
        "state_writers": [
            {
                "field": field["field"],
                "writers": field["writers"],
                "source_ref": field["source_ref"],
            }
            for field in fields
        ],
        "limitations": [
            "Artifact producer_stage is not proof of the business-semantic owner.",
            "A ModAgentState writer is not product-source write authority.",
            "A route declaration is not evidence that the route ran.",
            "Artifact existence is not evidence that the represented claim is verified.",
        ],
    }


def _internal_import_edges(root: Path) -> tuple[list[dict[str, str]], int]:
    source_root = root / "src"
    edges: set[tuple[str, str]] = set()
    file_count = 0
    for path in sorted(source_root.rglob("*.py")):
        file_count += 1
        try:
            module = _parse_python(path)
        except EngineeringCognitionError:
            continue
        source = _relative(path, root)
        for node in ast.walk(module):
            target: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src" or alias.name.startswith("src."):
                        edges.add((source, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "src" or node.module.startswith("src."):
                    target = node.module
                if target is not None:
                    edges.add((source, target))
    rows = [{"source": source, "target_module": target} for source, target in sorted(edges)]
    return rows, file_count


def project_source_dependency(root: Path) -> dict[str, Any]:
    """Project bounded static internal-import facts, not a runtime dependency graph."""

    edges, file_count = _internal_import_edges(root)
    limit = 300
    return {
        "internal_import_edge_count": len(edges),
        "internal_import_edges": edges[:limit],
        "python_source_file_count": file_count,
        "relative_imports_omitted": True,
        "status": "partial_static_projection",
        "truncated": len(edges) > limit,
        "limitations": [
            "Dynamic imports, callbacks, dependency injection, and runtime calls are not reconstructed.",
            "Relative imports are intentionally omitted from this v0 import-edge projection.",
            "An import edge does not prove a call or successful integration.",
        ],
    }


def _symbol_connection(root: Path, name: str) -> dict[str, Any]:
    definitions = _definition_refs(root, name)
    source_refs = _text_refs(root, name, under=root / "src")
    test_refs = _text_refs(root, name, under=root / "tests")
    return {
        "definition_refs": definitions,
        "name": name,
        "source_reference_count": len(source_refs["refs"]),
        "source_refs": source_refs["refs"],
        "source_refs_truncated": source_refs["truncated"],
        "status": (
            "source_and_test_references_observed"
            if definitions and test_refs["refs"]
            else "source_reference_only"
        ),
        "test_refs": test_refs["refs"],
        "test_refs_truncated": test_refs["truncated"],
        "verification_status": "not_executed_by_projection",
    }


def _fresh_api_augmentation_wiring(root: Path) -> dict[str, Any]:
    """Index the existing cross-module callback chain; never execute providers."""
    paths = {name: root / relative for name, relative in {
        "cli": "src/mod_agent/cli.py", "fresh": "src/mod_agent/fresh_coverage.py",
        "provider": "src/mod_agent/evidence/requirement_coverage_provider.py",
        "factories": "src/mod_agent/intelligent_evidence_only.py",
    }.items()}
    trees = {name: _parse_python(path) for name, path in paths.items() if path.is_file()}

    def calls(owner: str, name: str, *, forwarded: bool = False) -> list[str]:
        tree = trees.get(owner)
        if tree is None:
            return []
        result = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if callee != name:
                continue
            if forwarded and not any(keyword.arg == "source_augmentation_factory"
                    and isinstance(keyword.value, ast.Name) and keyword.value.id == "source_augmentation_factory"
                    for keyword in node.keywords):
                continue
            result.append(_source_ref(paths[owner], node.lineno, root))
        return result

    factory_names = (
        "make_fresh_api_source_augmentation_factory",
        "make_fresh_assisted_api_source_augmentation_factory",
    )
    definitions = {node.name: _source_ref(paths["factories"], node.lineno, root)
                   for node in ast.walk(trees["factories"])
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))} if "factories" in trees else {}
    factories = {name: calls("cli", name) for name in factory_names if name in definitions}
    seams = {
        "cli_to_fresh": calls("cli", "run_fresh_requirement_coverage", forwarded=True),
        "fresh_to_provider": calls("fresh", "make_requirement_coverage_evidence_provider", forwarded=True),
        "provider_constructor": calls("provider", "RequirementCoverageEvidenceProvider", forwarded=True),
        "augmentation_invocation": calls("provider", "source_augmentation_factory"),
    }
    connected = len(factories) == len(factory_names) and all(factories.values()) and all(seams.values())
    return {"connected": connected, "api_factory_names": sorted(name for name, refs in factories.items() if refs),
            "source_refs": sorted({ref for refs in (*factories.values(), *seams.values()) for ref in refs}
                                  | {definitions[name] for name in factories}),
            "seams": seams, "verification_status": "source_wiring_not_executed_by_projection"}


def project_product_api(root: Path) -> dict[str, Any]:
    """Project current product-Agent API evidence connections and honest gaps."""

    components = [_symbol_connection(root, name) for name in _PRODUCT_API_SYMBOLS]
    verified_member = _text_refs(root, "verified_member_accesses", under=root / "src")
    verified_member_tests = _text_refs(root, "verified_member_accesses", under=root / "tests")
    components.append(
        {
            "definition_refs": [],
            "name": "verified_member_accesses",
            "source_reference_count": len(verified_member["refs"]),
            "source_refs": verified_member["refs"],
            "source_refs_truncated": verified_member["truncated"],
            "status": (
                "source_and_test_references_observed"
                if verified_member["refs"] and verified_member_tests["refs"]
                else "source_reference_only"
            ),
            "test_refs": verified_member_tests["refs"],
            "test_refs_truncated": verified_member_tests["truncated"],
            "verification_status": "not_executed_by_projection",
        }
    )
    fresh_path = root / "src/mod_agent/fresh_coverage.py"
    fresh_text = _read_text(fresh_path)
    assembly_names = {
        "EvidenceDiscoverySpecialist",
        "LocalAssemblyEvidenceToolExecutor",
        "RunLocalMemberAccessResolver",
    }
    connected_names = sorted(name for name in assembly_names if name in fresh_text)
    augmentation = _fresh_api_augmentation_wiring(root)
    intelligent_path = root / "src/mod_agent/intelligent_evidence_only.py"
    local_entry_path = root / "src/mod_agent/local_entry.py"
    metadata_path = root / "src/mod_agent/evidence/dotnet_metadata.py"
    planning_path = root / "src/mod_agent/planning/implementation_plan.py"
    generator_path = root / "src/mod_agent/proposal/code_generator.py"
    return {
        "api_semantic_questions": list(PRODUCT_API_QUESTIONS),
        "components": components,
        "connection_boundaries": [
            {
                "connection": "EvidenceDiscoverySpecialist receives LocalAssemblyEvidenceToolExecutor",
                "source_refs": sorted(
                    set(
                        _text_refs_in_paths(
                            root,
                            "EvidenceDiscoverySpecialist",
                            (intelligent_path,),
                        )
                        + _text_refs_in_paths(
                            root,
                            "LocalAssemblyEvidenceToolExecutor",
                            (intelligent_path,),
                        )
                    )
                ),
                "verification_status": "source_reference_observed",
            },
            {
                "connection": "local_entry wires RunLocalMemberAccessResolver into enrich_source_evidence_with_member_accesses",
                "source_refs": sorted(
                    set(
                        _text_refs_in_paths(
                            root, "RunLocalMemberAccessResolver", (local_entry_path,)
                        )
                        + _text_refs_in_paths(
                            root,
                            "enrich_source_evidence_with_member_accesses",
                            (local_entry_path,),
                        )
                    )
                ),
                "verification_status": "source_reference_observed",
            },
            {
                "connection": "metadata enrichment writes verified_member_accesses and Planning/Patch read it",
                "source_refs": sorted(
                    set(
                        _text_refs_in_paths(root, "verified_member_accesses", (metadata_path,))
                        + _text_refs_in_paths(
                            root,
                            "verified_member_accesses",
                            (planning_path, generator_path),
                        )
                    )
                ),
                "verification_status": "source_reference_observed",
            },
        ],
        "fresh_coverage": {
            "assembly_api_components_present": connected_names,
            "evidence_provider": (
                "RgLiveEvidenceProvider" if "RgLiveEvidenceProvider" in fresh_text else None
            ),
            "provider_scope": "RgLive_plus_optional_API_augmentation" if augmentation["connected"] else "RgLive_only",
            "source_ref": _definition_ref(root, "run_fresh_requirement_coverage"),
            "status": "partially_connected" if connected_names or augmentation["connected"] else "not_connected_to_fresh",
            "api_factory_names": augmentation["api_factory_names"],
            "augmentation_source_refs": augmentation["source_refs"],
            "verification_status": augmentation["verification_status"],
        },
        "general_documentation_gap": {
            "reason": "Fresh has optional registered assembly/API augmentation; these source links do not establish a general documentation ingestion or discovery path.",
            "source_refs": _text_refs_in_paths(
                root, "*.cs", (root / "src/mod_agent/evidence/rg_live.py",)
            ),
            "status": "not_connected_to_fresh",
        },
        "general_product_api_semantics": {
            "status": "general_semantics_unproven",
            "reason": (
                "Narrow metadata/member-access components and tests do not prove general Mod API "
                "read/write/call/subscription/lifecycle behavior or runtime correctness."
            ),
        },
    }


def project_api_usage(root: Path) -> dict[str, Any]:
    product_api = project_product_api(root)
    return {
        "connection_points": product_api["components"],
        "fresh_coverage": product_api["fresh_coverage"],
        "questions": product_api["api_semantic_questions"],
        "status": "partial_connection_index",
        "limitations": [
            "Static symbol references do not prove correct call semantics or runtime compatibility.",
            "General API semantics remain unproven.",
        ],
    }


def project_behavior_dataflow(root: Path) -> dict[str, Any]:
    names = (
        "run_fresh_requirement_coverage",
        "make_requirement_coverage_evidence_provider",
        "RgLiveEvidenceProvider",
        "RequirementCoverageView",
        "ImplementationPlanner",
        "BindingResolver",
    )
    return {
        "inspection_anchors": [
            {"name": name, "source_ref": _definition_ref(root, name)} for name in names
        ],
        "edges_inferred": False,
        "status": "partial_static_projection",
        "limitations": [
            "This view indexes source connection points; it does not reconstruct all callbacks or intermediate values.",
            "No route execution, returned value, model decision, or runtime behavior is inferred from this chain.",
        ],
    }


def project_verification_claims(root: Path) -> dict[str, Any]:
    unit_files = sorted((root / "tests/unit").glob("test_*.py"))
    integration_files = sorted((root / "tests/integration").glob("test_*.py"))
    return {
        "actual_evidence": _actual_evidence(root),
        "test_inventory": {
            "integration_file_count": len(integration_files),
            "unit_file_count": len(unit_files),
        },
        "status": "inventory_only",
        "limitations": [
            "Test discovery is not a test result.",
            "A green unit test does not prove Build or Runtime behavior.",
            "A shape-qualified terminal journal is only a journal observation; this projection does not prove a bound Actual execution.",
        ],
    }


def _summary(root: Path) -> dict[str, Any]:
    current = project_current(root)
    surfaces = project_surfaces(root)
    nodes, routes, fields = _workflow_components(root)
    contracts = _artifact_contract_rows(root)
    producer_counts = Counter(str(row["producer_stage"]) for row in contracts)
    fresh_text = _read_text(root / "src/mod_agent/fresh_coverage.py")
    fresh_api_connected = (_fresh_api_augmentation_wiring(root)["connected"]
                           or any(name in fresh_text for name in _PRODUCT_API_SYMBOLS))
    return {
        "current": {
            "branch": current["git"]["branch"],
            "dirty": current["git"]["dirty"],
            "head_sha": current["git"]["head_sha"],
            "hook_runtime_snapshot_observed": current["hooks"]["runtime_snapshot"]["observed"],
            "hooks_configured": current["hooks"]["configuration"]["configured"],
            "task_packet": current["task_packet"],
        },
        "counts": {
            "artifact_contracts": len(contracts),
            "graph_nodes_declared": len(nodes),
            "registered_routes": len(routes),
            "state_fields": len(fields),
            "surface_entries": {
                str(surface["surface"]): surface["entry_count"] for surface in surfaces["surfaces"]
            },
        },
        "eight_views": [
            {"name": "Entry/Surface", "view": "surfaces", "status": "mechanical_projection"},
            {"name": "Workflow/Route", "view": "workflow", "status": "mechanical_projection"},
            {"name": "Artifact DAG", "view": "artifacts", "status": "mechanical_projection"},
            {
                "name": "Ownership/Authority",
                "view": "ownership",
                "status": "mechanical_projection_with_limits",
            },
            {
                "name": "Source/Dependency",
                "view": "source-dependency",
                "status": "partial_static_projection",
            },
            {"name": "API Usage", "view": "api-usage", "status": "partial_connection_index"},
            {
                "name": "Behavior/Dataflow",
                "view": "behavior-dataflow",
                "status": "partial_static_projection",
            },
            {
                "name": "Verification/Claim",
                "view": "verification-claims",
                "status": "inventory_only",
            },
        ],
        "product_api": {
            "fresh_status": (
                "partially_connected" if fresh_api_connected else "not_connected_to_fresh"
            ),
            "general_status": "general_semantics_unproven",
        },
        "producer_stage_counts": dict(sorted(producer_counts.items())),
        "read_only": True,
        "regeneration": "built_from_current_sources_registries_git_and_task_packet",
    }


def _route_entity(root: Path, name: str) -> dict[str, Any]:
    _, routes, _ = _workflow_components(root)
    selected = [route for route in routes if route["source"] == name]
    tests = _text_refs(root, name, under=root / "tests")
    return {
        "destinations": selected,
        "latest_actual_evidence": _actual_evidence(root, name),
        "name": name,
        "final_authority": "not_inferred",
        "semantic_owner": "not_inferred",
        "source_refs": sorted({str(route["source_ref"]) for route in selected}),
        "status": "found" if selected else "not_found_exact",
        "test_refs": tests["refs"],
        "test_refs_truncated": tests["truncated"],
    }


def _artifact_entity(root: Path, name: str) -> dict[str, Any]:
    row = next(
        (item for item in _artifact_contract_rows(root) if item["artifact_type"] == name), None
    )
    tests = _text_refs(root, name, under=root / "tests")
    return {
        "contract": row,
        "latest_actual_evidence": _actual_evidence(root, name),
        "name": name,
        "final_authority": "not_inferred",
        "semantic_owner": "not_inferred",
        "source_refs": [
            ref for ref in (_definition_ref(root, "default_artifact_contracts"),) if ref
        ],
        "status": "found" if row is not None else "not_found_exact",
        "test_refs": tests["refs"],
        "test_refs_truncated": tests["truncated"],
        "validator_refs": [
            ref
            for ref in (
                _definition_ref(root, "validate_write"),
                _definition_ref(root, "validate_envelope"),
            )
            if ref
        ],
        "validator_scope": "generic_registry_entrypoints_only",
        "artifact_specific_validator": "not_inferred",
    }


def _state_field_entity(root: Path, name: str) -> dict[str, Any]:
    _, _, fields = _workflow_components(root)
    row = next((item for item in fields if item["field"] == name), None)
    tests = _text_refs(root, name, under=root / "tests")
    return {
        "field": row,
        "latest_actual_evidence": _actual_evidence(root, name),
        "name": name,
        "final_authority": "not_inferred",
        "semantic_owner": "not_inferred",
        "source_refs": [row["source_ref"]] if row is not None else [],
        "status": "found" if row is not None else "not_found_exact",
        "test_refs": tests["refs"],
        "test_refs_truncated": tests["truncated"],
    }


def _symbol_entity(root: Path, name: str) -> dict[str, Any]:
    definitions = _definition_refs(root, name)
    references = _text_refs(root, name, under=root / "src")
    tests = _text_refs(root, name, under=root / "tests")
    return {
        "definition_refs": definitions,
        "latest_actual_evidence": _actual_evidence(root, name),
        "name": name,
        "final_authority": "not_inferred",
        "reference_refs": references["refs"],
        "reference_refs_truncated": references["truncated"],
        "status": "found" if definitions else "not_found_exact",
        "semantic_owner": "not_inferred",
        "test_refs": tests["refs"],
        "test_refs_truncated": tests["truncated"],
    }


def query_entity(root: Path, kind: str, name: str) -> dict[str, Any]:
    """Return an exact-name entity projection without fuzzy or case matching."""

    if kind not in ENTITY_KINDS:
        raise EngineeringCognitionError(f"Unsupported entity kind: {kind}")
    if not name:
        raise EngineeringCognitionError("Entity name must be non-empty.")
    builders = {
        "artifact": _artifact_entity,
        "route": _route_entity,
        "state-field": _state_field_entity,
        "symbol": _symbol_entity,
    }
    result = builders[kind](root, name)
    result["test_execution"] = "not_performed_by_projection"
    result["test_reference_semantics"] = "exact_lexical_reference_only"
    return {
        "exact_match_only": True,
        "kind": kind,
        "result": result,
    }


def build_projection(
    view: str = "summary",
    *,
    root: Path | None = None,
    kind: str | None = None,
    name: str | None = None,
    artifact_store_root: Path | str | None = None,
    feature_id: str | None = None,
    run_id: str | None = None,
    attempt_locator: str | None = None,
    query: str | None = None,
    run_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build one deterministic projection from the current repository."""

    selected_root = (root or repository_root()).resolve()
    execution_filters = (artifact_store_root, feature_id, run_id, attempt_locator, run_root)
    product_query = view == "verification-claims" and query == "product-execution"
    investigation_query = view == "verification-claims" and query == "patch-investigation"
    execution_query = view == "patch-execution" or (
        view == "verification-claims" and query in ("patch-execution", "patch-investigation")
    )
    if query is not None and not (product_query or investigation_query or execution_query and query == "patch-execution"):
        raise EngineeringCognitionError("Execution subqueries require --view verification-claims.")
    if any(value is not None for value in execution_filters) and not (execution_query or product_query):
        raise EngineeringCognitionError("Execution filters require an execution subquery.")
    if product_query:
        if run_root is None or any(value is not None for value in (artifact_store_root, attempt_locator, kind, name)):
            raise EngineeringCognitionError("product-execution requires --run-root and no Patch-store or entity filters.")
        from src.observability.patch_execution_index import project_product_execution

        try:
            data = project_product_execution(run_root, feature_id=feature_id, run_id=run_id)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EngineeringCognitionError(str(exc)) from exc
        selected_view = view
    elif execution_query:
        if artifact_store_root is None or kind is not None or name is not None or run_root is not None:
            raise EngineeringCognitionError(
                "patch-execution requires --artifact-store-root and no entity query."
            )
        from src.observability.patch_execution_index import project_patch_executions

        try:
            data = project_patch_executions(
                artifact_store_root, feature_id=feature_id, run_id=run_id,
                attempt_locator=attempt_locator,
                **({"include_investigation": True} if investigation_query else {}),
            )
        except (OSError, ValueError) as exc:
            raise EngineeringCognitionError(str(exc)) from exc
        selected_view = view
    elif kind is not None or name is not None:
        if kind is None or name is None:
            raise EngineeringCognitionError("--kind and --name must be provided together.")
        data = query_entity(selected_root, kind, name)
        selected_view = "entity"
    else:
        builders = {
            "summary": _summary,
            "current": project_current,
            "surfaces": project_surfaces,
            "workflow": project_workflow,
            "artifacts": project_artifacts,
            "ownership": project_ownership,
            "source-dependency": project_source_dependency,
            "api-usage": project_api_usage,
            "behavior-dataflow": project_behavior_dataflow,
            "verification-claims": project_verification_claims,
            "product-api": project_product_api,
        }
        if view == "all":
            data = {
                key: builder(selected_root) for key, builder in builders.items() if key != "summary"
            }
        elif view in builders:
            data = builders[view](selected_root)
        else:
            raise EngineeringCognitionError(f"Unsupported view: {view}")
        selected_view = view
    return {
        "data": data,
        "persistence": "none",
        "read_only": True,
        "repository_root": str(selected_root),
        "schema_version": SCHEMA_VERSION,
        "view": selected_view,
        **({"view_family": "verification-claims", "subquery": "product-execution" if product_query else "patch-investigation" if investigation_query else "patch-execution"}
           if execution_query or product_query else {}),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project current engineering structure as read-only JSON.",
    )
    parser.add_argument("--view", choices=VIEW_NAMES, default="summary")
    parser.add_argument("--kind", choices=ENTITY_KINDS)
    parser.add_argument("--name", help="Exact entity identifier; no fuzzy or case matching.")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--query", choices=("patch-execution", "patch-investigation", "product-execution"),
                        help="Bounded subquery within verification-claims.")
    parser.add_argument("--artifact-store-root", type=Path, help="Read-only Patch journal store.")
    parser.add_argument("--run-root", type=Path, help="Explicit product run root containing its original run_manifest.json.")
    parser.add_argument("--feature-id", help="Exact journal feature_id filter.")
    parser.add_argument("--run-id", help="Exact journal run_id filter.")
    parser.add_argument("--attempt-locator", help="Exact journal path within the store cr directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.kind is None) is not (args.name is None):
        parser.error("--kind and --name must be provided together")
    try:
        payload = build_projection(
            args.view, kind=args.kind, name=args.name,
            artifact_store_root=args.artifact_store_root, feature_id=args.feature_id,
            run_id=args.run_id, attempt_locator=args.attempt_locator,
            query=args.query, run_root=args.run_root,
        )
    except EngineeringCognitionError as exc:
        sys.stderr.write(f"engineering-cognition: {exc}\n")
        return 2
    sys.stdout.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
