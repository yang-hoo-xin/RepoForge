"""Persistent, file-scoped task execution state independent of a technical plan.

为什么要有它：技术计划可以改变，授权、源码身份和执行原件仍须独立核验。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4


class TaskAgentV2StateError(ValueError):
    @property
    def code(self) -> str:
        return str(self)


def _bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise TaskAgentV2StateError("invalid_path")
    p = PurePosixPath(value)
    if p.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise TaskAgentV2StateError("invalid_path")
    return p.as_posix()


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink():
            raise TaskAgentV2StateError("linked_path_not_supported")
        try:
            attributes = part.lstat()
        except FileNotFoundError:
            continue
        if getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024):
            raise TaskAgentV2StateError("linked_path_not_supported")


def save_v2_artifact(store_root: str | Path, kind: str, payload: Mapping[str, Any]) -> str:
    _no_links(Path(store_root).absolute())
    obj = object.__new__(TaskAgentV2State)
    obj.store_root = Path(store_root).resolve()
    obj.root = obj.store_root / "v2"
    return obj.save(kind, payload)


def load_v2_artifact(store_root: str | Path, ref: str) -> dict[str, Any]:
    _no_links(Path(store_root).absolute())
    obj = object.__new__(TaskAgentV2State)
    obj.store_root = Path(store_root).resolve()
    obj.root = obj.store_root / "v2"
    return obj.load(ref)


class TaskAgentV2State:
    """Host-created contract plus immutable records and a CAS-controlled head.

    为什么要有它：重复操作不能重新执行，未确认操作必须留给恢复处理。
    """

    def __init__(self, store_root: str | Path, contract_ref: str):
        _no_links(Path(store_root).absolute())
        self.store_root = Path(store_root).resolve()
        self.root = self.store_root / "v2"
        self.contract_ref = contract_ref
        self.contract = self.load(contract_ref)
        if self.contract.get("schema") != "task-agent-contract.v2":
            raise TaskAgentV2StateError("invalid_contract")
        self.head_path = self.root / "heads" / (_hash(contract_ref) + ".json")
        authority = self.load(self.contract["authority_ref"])
        if (authority.get("schema") != "task-agent-authority.v2" or authority.get("scope") != "file"
                or authority.get("operation") != "modify_existing" or not isinstance(authority.get("allowed_edit_paths"), list)
                or not isinstance(authority.get("read_roots"), dict) or self.contract.get("scope") != "file"
                or self.contract.get("operation") != "modify_existing"
                or not set(self.contract["allowed_edit_paths"]).issubset(authority["allowed_edit_paths"])
                or set(self.contract["read_roots"]) != set(authority["read_roots"])):
            raise TaskAgentV2StateError("invalid_contract_authority")
        for root_id, snapshot_ref in self.contract["read_roots"].items():
            snapshot = self.load(snapshot_ref)
            if (snapshot.get("schema") != "task-agent-source-snapshot.v2" or snapshot.get("root_id") != root_id
                    or Path(snapshot["source_root"]).resolve() != Path(authority["read_roots"][root_id]).resolve()
                    or any(_text_hash(row["text"]) != row["sha256"] for row in snapshot["files"].values())):
                raise TaskAgentV2StateError("invalid_source_snapshot")
        if self.contract["baseline_manifest_ref"] != self.contract["read_roots"].get("target"):
            raise TaskAgentV2StateError("invalid_baseline_snapshot")

    @classmethod
    def create(cls, store_root: str | Path, task_id: str, target_root: str | Path,
               read_roots: Mapping[str, str | Path] | None = None,
               allowed_edit_paths: tuple[str, ...] | list[str] = (), authority_ref: str = "",
               task_requirements: Any = (), scope: str = "file",
               contract_fields: Mapping[str, Any] | None = None) -> TaskAgentV2State:
        if scope != "file":
            raise TaskAgentV2StateError("unsupported_scope")
        obj = object.__new__(cls)
        _no_links(Path(store_root).absolute())
        obj.store_root = Path(store_root).resolve()
        obj.root = obj.store_root / "v2"
        obj.root.mkdir(parents=True, exist_ok=True)
        if not authority_ref:
            raise TaskAgentV2StateError("authority_required")
        authority = obj.load(authority_ref)
        if authority.get("scope") != "file":
            raise TaskAgentV2StateError("unsupported_scope")
        if (authority.get("schema") != "task-agent-authority.v2" or authority.get("operation") != "modify_existing"
                or not isinstance(authority.get("allowed_edit_paths"), list) or not isinstance(authority.get("read_roots"), dict)):
            raise TaskAgentV2StateError("invalid_authority")
        paths = sorted({_path(p) for p in allowed_edit_paths})
        if not set(paths).issubset(authority["allowed_edit_paths"]):
            raise TaskAgentV2StateError("authority_scope_mismatch")
        roots = {**dict(read_roots or {}), "target": target_root}
        if set(roots) != set(authority["read_roots"]) or any(
                Path(roots[key]).resolve() != Path(authority["read_roots"][key]).resolve() for key in roots):
            raise TaskAgentV2StateError("authority_read_scope_mismatch")
        snapshots = {}
        for root_id, directory in roots.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", root_id):
                raise TaskAgentV2StateError("invalid_root_id")
            _no_links(Path(directory).absolute())
            source_root = Path(directory).resolve(strict=True)
            files = {}
            for item in sorted(source_root.rglob("*")):
                rel = item.relative_to(source_root).as_posix()
                if any(part.casefold() in {".git", ".env", "bin", "obj", ".venv", "__pycache__", ".aws", ".ssh", ".codex", "credentials", ".idea", ".vs", "testresults"}
                       or part.casefold().startswith(".env.") for part in item.relative_to(source_root).parts):
                    continue
                _no_links(item)
                if not item.is_file():
                    continue
                if not item.resolve().is_relative_to(source_root):
                    raise TaskAgentV2StateError("source_outside_root")
                try:
                    text = item.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                files[rel] = {"text": text, "sha256": _text_hash(text)}
            snapshots[root_id] = obj.save("snapshot", {"schema": "task-agent-source-snapshot.v2",
                "root_id": root_id, "source_root": str(source_root), "files": files})
        baseline = obj.load(snapshots["target"])
        if not set(paths).issubset(baseline["files"]):
            raise TaskAgentV2StateError("authorized_file_missing_or_not_utf8")
        fields = dict(contract_fields or {})
        reserved = {"schema", "task_id", "authority_ref", "baseline_manifest_ref", "allowed_edit_paths", "read_roots", "scope", "operation", "task_requirements"}
        if reserved.intersection(fields):
            raise TaskAgentV2StateError("reserved_contract_field")
        contract = {**fields, "schema": "task-agent-contract.v2", "task_id": task_id,
            "authority_ref": authority_ref, "baseline_manifest_ref": snapshots["target"],
            "allowed_edit_paths": paths, "read_roots": snapshots, "scope": "file",
            "operation": "modify_existing", "task_requirements": task_requirements}
        ref = obj.save("contract", contract)
        obj = cls(store_root, ref)
        registry = obj.save("registry", {"schema": "task-agent-binding-registry.v2", "contract_ref": ref, "bindings": {}})
        obj.head_path.parent.mkdir(parents=True, exist_ok=True)
        initial = {"contract_ref": ref, "registry_ref": registry, "current_revision_ref": snapshots["target"],
            "plan_ref": None, "operations": {}, "resources": {}, "sequence": 0, "event_ref": None}
        try:
            with obj.head_path.open("xb") as handle:
                handle.write(_bytes(initial))
        except FileExistsError:
            raise TaskAgentV2StateError("task_already_exists") from None
        return obj

    def save(self, kind: str, payload: Mapping[str, Any]) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", kind):
            raise TaskAgentV2StateError("invalid_artifact_kind")
        data = dict(payload)
        data.setdefault("schema", f"task-agent-{kind}.v2")
        raw = _bytes(data)
        digest = hashlib.sha256(raw).hexdigest()
        path = self.root / kind / (digest + ".json")
        _no_links(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(raw)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise TaskAgentV2StateError("artifact_corrupted") from None
        return path.relative_to(self.store_root).as_posix()

    def load(self, ref: str) -> dict[str, Any]:
        if not isinstance(ref, str) or not re.fullmatch(r"v2/[a-z][a-z0-9_-]*/[0-9a-f]{64}\.json", ref):
            raise TaskAgentV2StateError("invalid_artifact_ref")
        path = self.store_root / ref
        _no_links(path)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root.resolve()):
            raise TaskAgentV2StateError("invalid_artifact_path")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != path.stem:
            raise TaskAgentV2StateError("artifact_hash_mismatch")
        data = json.loads(raw)
        if not isinstance(data, dict) or not str(data.get("schema", "")).endswith(".v2"):
            raise TaskAgentV2StateError("invalid_artifact_schema")
        return data

    @property
    def head(self) -> dict[str, Any]:
        _no_links(self.head_path)
        data = json.loads(self.head_path.read_bytes())
        if data["contract_ref"] != self.contract_ref:
            raise TaskAgentV2StateError("head_contract_mismatch")
        return data

    @property
    def registry_ref(self) -> str:
        return self.head["registry_ref"]

    @property
    def current_revision_ref(self) -> str:
        return self.head["current_revision_ref"]

    def validate_registry(self, registry_ref: str) -> dict[str, Any]:
        registry = self.load(registry_ref)
        if registry.get("schema") != "task-agent-binding-registry.v2" or registry.get("contract_ref") != self.contract_ref:
            raise TaskAgentV2StateError("registry_contract_mismatch")
        baseline = self.load(self.contract["baseline_manifest_ref"])["files"]
        for binding_id, binding in registry["bindings"].items():
            path = binding.get("canonical_path")
            if (path not in self.contract["allowed_edit_paths"] or binding.get("authority_ref") != self.contract["authority_ref"]
                    or binding.get("scope") != "file" or binding.get("operation") != "modify_existing"
                    or binding.get("baseline_manifest_ref") != self.contract["baseline_manifest_ref"]
                    or binding.get("baseline_file_sha256") != baseline[path]["sha256"]
                    or binding.get("binding_id") != binding_id
                    or binding_id != "file-" + _hash({k: v for k, v in binding.items() if k != "binding_id"})):
                raise TaskAgentV2StateError("invalid_binding_authority")
        return registry

    def _write_head(self, head: dict[str, Any]) -> None:
        temp = self.head_path.with_name(self.head_path.name + "." + uuid4().hex)
        _no_links(temp)
        with temp.open("xb") as handle:
            handle.write(_bytes(head))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.head_path)

    @contextmanager
    def _transaction(self, operation_id: str, request: dict[str, Any]):
        lock = self.head_path.with_suffix(".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise TaskAgentV2StateError("operation_in_progress_or_unknown") from None
        os.close(fd)
        try:
            head = self.head
            if head.get("pending_operation"):
                raise TaskAgentV2StateError("operation_outcome_unknown")
            identity = _hash(request)
            existing = head["operations"].get(operation_id)
            if existing:
                if existing["request_hash"] != identity:
                    raise TaskAgentV2StateError("operation_identity_conflict")
                yield head, existing["result"], True
                return
            result: dict[str, Any] = {}
            pending = self.save("intent", {"schema": "task-agent-operation-intent.v2", "contract_ref": self.contract_ref,
                "operation_id": operation_id, "request_hash": identity, "request": request, "previous_event_ref": head["event_ref"]})
            head["pending_operation"] = pending
            self._write_head(head)
            try:
                yield head, result, False
            except TaskAgentV2StateError:
                previous = self.head
                previous.pop("pending_operation", None)
                self._write_head(previous)
                raise
            head["operations"][operation_id] = {"request_hash": identity, "result": result}
            head["sequence"] += 1
            event = self.save("operation", {"schema": "task-agent-operation.v2", "contract_ref": self.contract_ref,
                "operation_id": operation_id, "request_hash": identity, "request": request, "result": result,
                "previous_event_ref": head["event_ref"], "sequence": head["sequence"], "resources": head["resources"]})
            head["event_ref"] = event
            head.pop("pending_operation", None)
            self._write_head(head)
        finally:
            lock.unlink(missing_ok=True)

    def read_source(self, parent_revision_ref: str, path: str, root_id: str = "target") -> dict[str, Any]:
        self.validate_read_request(parent_revision_ref, path=path, root_id=root_id)
        path = _path(path)
        if root_id not in self.contract["read_roots"]:
            raise TaskAgentV2StateError("read_root_denied")
        if root_id == "target":
            revision = self._revision(parent_revision_ref)
            files = self.materialize(revision)
        else:
            if parent_revision_ref not in ("baseline", self.contract["read_roots"][root_id]):
                raise TaskAgentV2StateError("dependency_revision_mismatch")
            revision = self.contract["read_roots"][root_id]
            files = {p: row["text"] for p, row in self.load(revision)["files"].items()}
        if path not in files:
            raise TaskAgentV2StateError("source_not_in_snapshot")
        data = {"schema": "task-agent-source-view.v2", "contract_ref": self.contract_ref,
            "root_id": root_id, "parent_revision_ref": revision, "path": path,
            "file_sha256": _text_hash(files[path]), "text": files[path]}
        return {**data, "source_view_ref": self.save("source-view", data)}

    def _revision(self, ref: str) -> str:
        return self.contract["baseline_manifest_ref"] if ref in ("baseline", "B0") else ref

    def validate_read_request(self, parent_revision_ref: str, *, path: str | None = None,
                              root_id: str = "target", query: str | None = None) -> None:
        if root_id not in self.contract["read_roots"]:
            raise TaskAgentV2StateError("read_root_denied")
        if root_id == "target":
            files = self.materialize(parent_revision_ref)
        else:
            if parent_revision_ref not in ("baseline", self.contract["read_roots"][root_id]):
                raise TaskAgentV2StateError("dependency_revision_mismatch")
            files = self.load(self.contract["read_roots"][root_id])["files"]
        if path is not None:
            path = _path(path)
            if any(part.casefold() in {".git", ".env", ".aws", ".ssh", ".codex", "credentials"}
                   or part.casefold().startswith(".env.") for part in path.split("/")):
                raise TaskAgentV2StateError("read_path_denied")
            if path not in files:
                raise TaskAgentV2StateError("source_not_in_snapshot")
        elif not isinstance(query, str) or not query:
            raise TaskAgentV2StateError("invalid_search")

    def search_source(self, parent_revision_ref: str, query: str, root_id: str = "target",
                      max_matches: int = 40) -> dict[str, Any]:
        self.validate_read_request(parent_revision_ref, root_id=root_id, query=query)
        if not isinstance(query, str) or not query or max_matches < 1:
            raise TaskAgentV2StateError("invalid_search")
        if root_id not in self.contract["read_roots"]:
            raise TaskAgentV2StateError("read_root_denied")
        if root_id == "target":
            files = self.materialize(parent_revision_ref)
        else:
            if parent_revision_ref not in ("baseline", self.contract["read_roots"][root_id]):
                raise TaskAgentV2StateError("dependency_revision_mismatch")
            files = {p: row["text"] for p, row in self.load(self.contract["read_roots"][root_id])["files"].items()}
        matches = []
        count = 0
        for path, content in sorted(files.items()):
            selected = [(i, line) for i, line in enumerate(content.splitlines(), 1) if query.casefold() in line.casefold()]
            if not selected:
                continue
            count += len(selected)
            if len(matches) >= max_matches:
                continue
            view = self.read_source(parent_revision_ref, path, root_id)
            for number, line in selected[:max_matches - len(matches)]:
                matches.append({"path": path, "root_id": root_id, "line": number, "text": line,
                    "source_view_ref": view["source_view_ref"], "file_sha256": view["file_sha256"]})
        return {"matches": matches, "total_matches": count, "truncated": count > len(matches)}

    def validate_binding_request(self, expected_registry_ref: str, source_view_ref: str, path: str) -> None:
        path = _path(path)
        if self.registry_ref != expected_registry_ref:
            raise TaskAgentV2StateError("stale_registry")
        view = self.load(source_view_ref)
        if view.get("contract_ref") != self.contract_ref or view.get("root_id") != "target" or view.get("path") != path:
            raise TaskAgentV2StateError("source_view_not_edit_authority")
        if path not in self.contract["allowed_edit_paths"]:
            raise TaskAgentV2StateError("edit_path_denied")
        self._validate_view(view, view["parent_revision_ref"], path)

    def validate_submission(self, parent_revision_ref: str, edits: list[dict[str, Any]]) -> None:
        parent = self._revision(parent_revision_ref)
        if parent != self.current_revision_ref:
            raise TaskAgentV2StateError("stale_parent_revision")
        if not edits:
            raise TaskAgentV2StateError("edits_required")
        registry = self.validate_registry(self.registry_ref)
        original = self.materialize(parent)
        working = dict(original)
        for edit in edits:
            binding = registry["bindings"].get(edit.get("binding_id"))
            if binding is None:
                raise TaskAgentV2StateError("unknown_binding")
            path = binding["canonical_path"]
            self._validate_view(self.load(edit.get("source_view_ref", "")), parent, path)
            before, after = edit.get("before_text"), edit.get("after_text")
            if not isinstance(before, str) or not before or not isinstance(after, str):
                raise TaskAgentV2StateError("explicit_nonempty_before_and_after_required")
            if original[path].count(before) != 1 or working[path].count(before) != 1:
                raise TaskAgentV2StateError("before_text_not_unique")
            working[path] = working[path].replace(before, after, 1)
        if working == original:
            raise TaskAgentV2StateError("candidate_has_no_changes")

    def add_binding(self, expected_registry_ref: str, source_view_ref: str, path: str,
                    operation_id: str | None = None) -> dict[str, Any]:
        path = _path(path)
        request = {"kind": "binding_add", "registry_ref": expected_registry_ref, "source_view_ref": source_view_ref, "path": path}
        with self._transaction(operation_id or uuid4().hex, request) as (head, result, replay):
            if replay:
                return dict(result)
            if head["registry_ref"] != expected_registry_ref:
                raise TaskAgentV2StateError("stale_registry")
            view = self.load(source_view_ref)
            if view.get("contract_ref") != self.contract_ref or view.get("root_id") != "target" or view.get("path") != path:
                raise TaskAgentV2StateError("source_view_not_edit_authority")
            if path not in self.contract["allowed_edit_paths"]:
                raise TaskAgentV2StateError("edit_path_denied")
            self._validate_view(view, view["parent_revision_ref"], path)
            baseline = self.load(self.contract["baseline_manifest_ref"])["files"][path]
            binding = {"authority_ref": self.contract["authority_ref"], "baseline_manifest_ref": self.contract["baseline_manifest_ref"],
                "canonical_path": path, "operation": "modify_existing", "scope": "file", "baseline_file_sha256": baseline["sha256"]}
            binding_id = "file-" + _hash(binding)
            binding["binding_id"] = binding_id
            registry = self.validate_registry(expected_registry_ref)
            registry["bindings"][binding_id] = binding
            head["registry_ref"] = self.save("registry", registry)
            result.update(binding_id=binding_id, registry_ref=head["registry_ref"])
        return dict(result)

    def update_plan(self, plan: Mapping[str, Any], operation_id: str | None = None) -> str:
        with self._transaction(operation_id or uuid4().hex, {"kind": "plan_update", "plan": dict(plan)}) as (head, result, replay):
            if not replay:
                head["plan_ref"] = self.save("technical-plan", {"schema": "task-agent-technical-plan.v2", "contract_ref": self.contract_ref, "plan": dict(plan)})
                result["plan_ref"] = head["plan_ref"]
        return result["plan_ref"]

    def _validate_view(self, view: dict[str, Any], parent: str, path: str,
                       files: dict[str, str] | None = None) -> None:
        files = self.materialize(parent) if files is None else files
        if (view.get("schema") != "task-agent-source-view.v2" or view.get("contract_ref") != self.contract_ref
                or view.get("root_id") != "target" or view.get("parent_revision_ref") != parent
                or view.get("path") != path or view.get("text") != files.get(path)
                or view.get("file_sha256") != _text_hash(files.get(path, ""))):
            raise TaskAgentV2StateError("stale_or_invalid_source_view")

    def submit(self, parent_revision_ref: str, edits: list[dict[str, Any]], operation_id: str | None = None,
               model_origin: Mapping[str, Any] | None = None, rationale: str = "") -> dict[str, Any]:
        parent = self._revision(parent_revision_ref)
        request = {"kind": "candidate_submit", "parent_revision_ref": parent, "edits": edits,
            "model_origin": dict(model_origin or {}), "rationale": rationale}
        with self._transaction(operation_id or uuid4().hex, request) as (head, result, replay):
            if replay:
                return dict(result)
            if parent != head["current_revision_ref"]:
                raise TaskAgentV2StateError("stale_parent_revision")
            if not edits:
                raise TaskAgentV2StateError("edits_required")
            original = self.materialize(parent)
            files = dict(original)
            registry = self.validate_registry(head["registry_ref"])
            changes = []
            for edit in edits:
                binding = registry["bindings"].get(edit.get("binding_id"))
                if binding is None:
                    raise TaskAgentV2StateError("unknown_binding")
                path = binding["canonical_path"]
                view = self.load(edit.get("source_view_ref", ""))
                self._validate_view(view, parent, path)
                before, after = edit.get("before_text"), edit.get("after_text")
                if not isinstance(before, str) or not before or not isinstance(after, str):
                    raise TaskAgentV2StateError("explicit_nonempty_before_and_after_required")
                if original[path].count(before) != 1 or files[path].count(before) != 1:
                    raise TaskAgentV2StateError("before_text_not_unique")
                previous_text = files[path]
                files[path] = previous_text.replace(before, after, 1)
                changes.append({"path": path, "binding_id": binding["binding_id"], "source_view_ref": edit["source_view_ref"],
                    "before_text": before, "after_text": after, "before_sha256": _text_hash(previous_text), "after_sha256": _text_hash(files[path])})
            if files == original:
                raise TaskAgentV2StateError("candidate_has_no_changes")
            baseline = self.load(self.contract["baseline_manifest_ref"])["files"]
            overlay = {p: t for p, t in files.items() if t != baseline[p]["text"]}
            identity = _hash({"baseline_manifest_ref": self.contract["baseline_manifest_ref"], "files": {p: _text_hash(t) for p, t in sorted(files.items())}})
            candidate = {"schema": "task-agent-candidate.v2", "contract_ref": self.contract_ref,
                "baseline_manifest_ref": self.contract["baseline_manifest_ref"], "parent_revision_ref": parent,
                "registry_ref": head["registry_ref"], "files": overlay, "changes": changes,
                "candidate_source_id": identity, "model_origin": dict(model_origin or {})}
            ref = self.save("candidate", candidate)
            head["current_revision_ref"] = ref
            result.update(candidate_ref=ref, candidate_source_id=identity, changes=changes)
        return dict(result)

    def materialize(self, candidate_ref: str) -> dict[str, str]:
        return self._materialize(candidate_ref, {})

    def _materialize(self, candidate_ref: str, validated: dict[str, Any]) -> dict[str, str]:
        ref = self._revision(candidate_ref)
        baseline = self.load(self.contract["baseline_manifest_ref"])
        files = {p: row["text"] for p, row in baseline["files"].items()}
        if ref == self.contract["baseline_manifest_ref"]:
            return files
        candidate = self._validate_candidate(ref, validated)
        files.update(candidate["files"])
        return files

    def validate_candidate(self, candidate_ref: str) -> dict[str, Any]:
        return self._validate_candidate(candidate_ref, {})

    def _validate_candidate(self, candidate_ref: str, validated: dict[str, Any]) -> dict[str, Any]:
        if candidate_ref in validated:
            if validated[candidate_ref] is None:
                raise TaskAgentV2StateError("cyclic_candidate_chain")
            return validated[candidate_ref]
        validated[candidate_ref] = None
        candidate = self.load(candidate_ref)
        if (candidate.get("schema") != "task-agent-candidate.v2" or candidate.get("contract_ref") != self.contract_ref
                or candidate.get("baseline_manifest_ref") != self.contract["baseline_manifest_ref"]):
            raise TaskAgentV2StateError("candidate_contract_mismatch")
        parent = self._materialize(candidate["parent_revision_ref"], validated)
        expected = dict(parent)
        registry = self.validate_registry(candidate["registry_ref"])
        if registry.get("contract_ref") != self.contract_ref:
            raise TaskAgentV2StateError("registry_contract_mismatch")
        baseline = self.load(self.contract["baseline_manifest_ref"])["files"]
        for change in candidate["changes"]:
            path = _path(change["path"])
            binding = registry["bindings"].get(change["binding_id"], {})
            if (path not in self.contract["allowed_edit_paths"] or binding.get("canonical_path") != path
                    or binding.get("authority_ref") != self.contract["authority_ref"] or binding.get("scope") != "file"
                    or binding.get("operation") != "modify_existing" or binding.get("baseline_file_sha256") != baseline[path]["sha256"]
                    or binding.get("baseline_manifest_ref") != self.contract["baseline_manifest_ref"]
                    or binding.get("binding_id") != change["binding_id"]
                    or binding.get("binding_id") != "file-" + _hash({k: v for k, v in binding.items() if k != "binding_id"})):
                raise TaskAgentV2StateError("invalid_candidate_authority")
            self._validate_view(self.load(change["source_view_ref"]), candidate["parent_revision_ref"], path, parent)
            before, after = change["before_text"], change["after_text"]
            if not before or expected[path].count(before) != 1 or _text_hash(expected[path]) != change["before_sha256"]:
                raise TaskAgentV2StateError("invalid_candidate_preimage")
            expected[path] = expected[path].replace(before, after, 1)
            if _text_hash(expected[path]) != change["after_sha256"]:
                raise TaskAgentV2StateError("invalid_candidate_postimage")
        overlay = {p: t for p, t in expected.items() if t != baseline[p]["text"]}
        identity = _hash({"baseline_manifest_ref": self.contract["baseline_manifest_ref"], "files": {p: _text_hash(t) for p, t in sorted(expected.items())}})
        if candidate["files"] != overlay or candidate["candidate_source_id"] != identity:
            raise TaskAgentV2StateError("candidate_content_mismatch")
        validated[candidate_ref] = candidate
        return candidate

    def changes_from_baseline(self, candidate_ref: str) -> list[dict[str, Any]]:
        candidate = self.validate_candidate(candidate_ref)
        baseline = self.load(self.contract["baseline_manifest_ref"])["files"]
        registry = self.load(candidate["registry_ref"])["bindings"]
        by_path = {binding["canonical_path"]: binding_id for binding_id, binding in registry.items()}
        return [{"path": path, "binding_id": by_path[path], "before_text": baseline[path]["text"],
            "after_text": text, "before_sha256": baseline[path]["sha256"], "after_sha256": _text_hash(text)}
            for path, text in sorted(candidate["files"].items())]

    def record_resources(self, usage: Mapping[str, int | float], operation_id: str) -> dict[str, Any]:
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 for v in usage.values()):
            raise TaskAgentV2StateError("invalid_resource_delta")
        with self._transaction(operation_id, {"kind": "resource_record", "usage": dict(usage)}) as (head, result, replay):
            if not replay:
                for key, value in usage.items():
                    head["resources"][key] = head["resources"].get(key, 0) + value
                result.update(head["resources"])
        return dict(result)
