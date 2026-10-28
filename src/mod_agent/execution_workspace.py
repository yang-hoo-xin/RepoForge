"""Short, recoverable workspaces whose bindings remain in durable task evidence.

为什么要有它：执行路径的宿主限制不应迫使任务证据迁移到临时目录，重启也不能为同一操作重新选择路径。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable


class ExecutionWorkspaceError(ValueError):
    """A workspace could not be allocated without losing its binding."""


@dataclass(frozen=True)
class ExecutionWorkspace:
    execution_root: Path
    evidence_root: Path
    mapping_ref: str
    workspace_id: str
    recovered: bool


def _bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _relative(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or "\x00" in value):
        raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_RELATIVE_PATH_REQUIRED")
    return value


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return bool(path.is_symlink()
        or getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _unlinked_evidence_root(value: str | Path) -> Path:
    """Preserve and inspect the configured lexical root before resolving it.

    Scope: the evidence root itself and every existing filesystem parent.
    为什么要有它：若构造时先resolve，根或父级链接身份会丢失，随后所有期望路径都会错误地接受同一外部目标。
    """
    selected = Path(value).absolute()
    for component in reversed((selected, *selected.parents)):
        if _is_link_or_reparse(component):
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_EVIDENCE_LINKED")
    return selected.resolve()


class ExecutionWorkspaceManager:
    """Allocate one deterministic short root behind a single task-facing interface."""

    def __init__(self, *, state, evidence_root: str | Path,
                 execution_base: str | Path | None = None, windows_path_limit: int = 248):
        self.state = state
        self.evidence_root = _unlinked_evidence_root(evidence_root)
        self.execution_base = Path(execution_base).resolve() if execution_base is not None else (
            Path(tempfile.gettempdir()).resolve() / "rfw")
        if windows_path_limit < 128:
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_PATH_LIMIT_INVALID")
        self.windows_path_limit = windows_path_limit

    def allocate(self, *, kind: str, operation_id: str, source_revision_ref: str | None = None,
                 source_identity: str | None = None, projected_paths: Iterable[str] = ()) -> ExecutionWorkspace:
        identity = self._identity(kind=kind, operation_id=operation_id,
            source_revision_ref=source_revision_ref, source_identity=source_identity)
        workspace_id = _hash(identity)
        index = self._evidence_path(self.evidence_root / "workspace-mappings" / (workspace_id + ".json"))
        recovered = index.is_file()
        if recovered:
            binding = self._read_index(index, workspace_id)
            mapping_ref = binding["mapping_ref"]
            mapping = self.state.load(mapping_ref)
            self._validate_mapping(mapping, identity, workspace_id)
        else:
            mapping = {"schema": "task-agent-execution-workspace.v2", "workspace_id": workspace_id,
                "identity": identity, "execution_root": str(self.execution_base / workspace_id[:32]),
                "evidence_root": str(self.evidence_root / kind / workspace_id),
                "path_policy": {"kind": "short_host_temporary_root", "windows_limit": self.windows_path_limit}}
            mapping_ref = self.state.save("execution-workspace", mapping)
            self._write_index(index, {"schema": "task-agent-execution-workspace-binding.v2",
                "workspace_id": workspace_id, "mapping_ref": mapping_ref})
            binding = self._read_index(index, workspace_id)
            if binding["mapping_ref"] != mapping_ref:
                mapping_ref = binding["mapping_ref"]
                mapping = self.state.load(mapping_ref)
                self._validate_mapping(mapping, identity, workspace_id)
                recovered = True

        return self._workspace(mapping, mapping_ref=mapping_ref, workspace_id=workspace_id,
            projected_paths=projected_paths, recovered=recovered)

    def recover(self, *, mapping_ref: str, kind: str, operation_id: str,
                source_revision_ref: str | None = None, source_identity: str | None = None,
                projected_paths: Iterable[str] = ()) -> ExecutionWorkspace:
        """Recover one exact mapping under this manager's configured roots.

        Scope: existing content-addressed workspace mappings only; this grants no new operation or path.
        为什么要有它：资格和恢复调用方必须复用同一根绑定校验，不能各自重写一份容易漂移的映射验证。
        """
        identity = self._identity(kind=kind, operation_id=operation_id,
            source_revision_ref=source_revision_ref, source_identity=source_identity)
        workspace_id = _hash(identity)
        mapping = self.state.load(mapping_ref)
        self._validate_mapping(mapping, identity, workspace_id)
        return self._workspace(mapping, mapping_ref=mapping_ref, workspace_id=workspace_id,
            projected_paths=projected_paths, recovered=True)

    def _identity(self, *, kind: str, operation_id: str, source_revision_ref: str | None,
                  source_identity: str | None) -> dict[str, object]:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", kind):
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_KIND_INVALID")
        if not isinstance(operation_id, str) or not operation_id or len(operation_id) > 1024:
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_OPERATION_INVALID")
        return {"schema": "task-agent-execution-workspace-identity.v2",
            "task_contract_ref": self.state.contract_ref, "kind": kind, "operation_id": operation_id,
            "source_revision_ref": source_revision_ref, "source_identity": source_identity,
            "artifact_store_root": str(self.state.store_root)}

    def _workspace(self, mapping, *, mapping_ref: str, workspace_id: str,
                   projected_paths: Iterable[str], recovered: bool) -> ExecutionWorkspace:
        execution_root = Path(mapping["execution_root"])
        evidence_root = self._evidence_path(Path(mapping["evidence_root"]))
        self._check_projected_paths(execution_root, projected_paths)
        self._bind_execution_root(execution_root, workspace_id, mapping_ref)
        evidence_root.mkdir(parents=True, exist_ok=True)
        return ExecutionWorkspace(execution_root=execution_root, evidence_root=evidence_root,
            mapping_ref=mapping_ref, workspace_id=workspace_id, recovered=recovered)

    def _validate_mapping(self, mapping, identity, workspace_id):
        expected = {"schema", "workspace_id", "identity", "execution_root", "evidence_root", "path_policy"}
        if (not isinstance(mapping, dict) or set(mapping) != expected
                or mapping["schema"] != "task-agent-execution-workspace.v2"
                or mapping["workspace_id"] != workspace_id or mapping["identity"] != identity
                or mapping["path_policy"] != {"kind": "short_host_temporary_root",
                    "windows_limit": self.windows_path_limit}):
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_MAPPING_CHANGED")
        execution_root = Path(mapping["execution_root"])
        evidence_root = self._evidence_path(Path(mapping["evidence_root"]))
        expected_execution_root = (self.execution_base / workspace_id[:32]).resolve()
        expected_evidence_root = (self.evidence_root / identity["kind"] / workspace_id).resolve()
        if (not execution_root.is_absolute() or not evidence_root.is_absolute()
                or execution_root.resolve() != expected_execution_root
                or evidence_root.resolve() != expected_evidence_root
                or execution_root.resolve().is_relative_to(evidence_root.resolve())
                or evidence_root.resolve().is_relative_to(execution_root.resolve())):
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_MAPPING_SCOPE_INVALID")

    def _evidence_path(self, path: Path) -> Path:
        """Return one lexical descendant after rejecting linked existing components.

        Scope: the configured evidence root and every existing component down to one index or evidence target.
        为什么要有它：先resolve再比较会把链接后的外部位置同时当成期望值，证据和索引必须在写入前拒绝symlink或reparse重定向。
        """
        configured = self.evidence_root.absolute()
        selected = path.absolute()
        try:
            relative = selected.relative_to(configured)
        except ValueError as error:
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_EVIDENCE_SCOPE_INVALID") from error
        current = configured
        for part in (None, *relative.parts):
            if part is not None:
                current = current / part
            if _is_link_or_reparse(current):
                raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_EVIDENCE_LINKED")
        if not selected.resolve().is_relative_to(configured.resolve()):
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_EVIDENCE_SCOPE_INVALID")
        return selected

    def _check_projected_paths(self, root: Path, projected_paths: Iterable[str]) -> None:
        for value in projected_paths:
            target = root / Path(*PurePosixPath(_relative(value)).parts)
            if os.name == "nt" and len(str(target)) >= self.windows_path_limit:
                raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_TARGET_PATH_TOO_LONG: " + str(target))

    @staticmethod
    def _read_index(path: Path, workspace_id: str) -> dict[str, str]:
        try:
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_BINDING_LINKED")
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_BINDING_UNREADABLE") from error
        if (not isinstance(value, dict) or set(value) != {"schema", "workspace_id", "mapping_ref"}
                or value["schema"] != "task-agent-execution-workspace-binding.v2"
                or value["workspace_id"] != workspace_id or not isinstance(value["mapping_ref"], str)):
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_BINDING_CHANGED")
        return value

    @staticmethod
    def _write_index(path: Path, value: dict[str, str]) -> None:
        raw = _bytes(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(raw)
        except FileExistsError:
            if path.read_bytes() != raw:
                return

    @staticmethod
    def _bind_execution_root(root: Path, workspace_id: str, mapping_ref: str) -> None:
        marker = root / ".repoforge-workspace.json"
        value = {"schema": "repoforge-execution-workspace-marker.v2",
            "workspace_id": workspace_id, "mapping_ref": mapping_ref}
        raw = _bytes(value)
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or getattr(root.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_ROOT_LINKED")
        try:
            with marker.open("xb") as stream:
                stream.write(raw)
        except FileExistsError:
            if marker.read_bytes() != raw:
                raise ExecutionWorkspaceError("EXECUTION_WORKSPACE_MARKER_CHANGED") from None
