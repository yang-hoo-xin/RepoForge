"""Read-only discovery and browsing of existing local run records.

为什么要有它：不同运行格式共享授权、完整性与脱敏读取，而不是伪造另一套 Trace 或补写历史记录。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from src.mcp_server.safety import Redactor

_TREE_CHARS = 128 * 1024
_INDEX_BYTES = 8 * 1024 * 1024
_TIMELINE_BYTES = 512 * 1024
_PAGE_CHARS = 256 * 1024
_METRIC_KEYS = {"input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens",
                "prompt_tokens", "completion_tokens", "max_tokens", "max_output_tokens", "token_count",
                "token_budget", "tokens_per_second", "prompt_eval_count", "eval_count"}
_SECRET_KEY = re.compile(r"(?:^|[_\-. ])(?:api[_-]?(?:key|token)|access[_-]?token|refresh[_-]?token|auth[_-]?token|password|passwd|secrets?|credentials?|authorization|cookie)(?:$|[_\-. ])", re.I)
_BEARER = re.compile(r"(?i)\b(Bearer\s+)[^\s\"'<>\\,;]+")
_AUTH_HEADER = re.compile(r"(?i)(\b(?:proxy-)?authorization\s*[:=]\s*)[^\r\n]*")
_TEXT_SECRET = re.compile(r"(?i)([\"']?(?:api[_-]?(?:key|token)|access[_-]?token|refresh[_-]?token|password|passwd|secrets?|authorization|credentials?)[\"']?\s*[:=]\s*)([\"'])(.*?)(?:\2|$)", re.S)
_LINE_SECRET = re.compile(r"(?im)^(\s*(?:(?:[A-Z][A-Z0-9_]*_)?(?:api[_-]?(?:key|token)|password|secrets?)|Authorization|Proxy-Authorization|Cookie|Set-Cookie)\s*[:=]\s*)[^\r\n]*")
_CONTAINER_SECRET = re.compile(r"(?i)([\"']?(?:api[_-]?(?:key|token)|access[_-]?token|password|secrets?|credentials?|authorization)[\"']?\s*[:=]\s*)[\[{]")
_UNQUOTED_SECRET = re.compile(r"(?i)([\"']?(?:api[_-]?(?:key|token)|access[_-]?token|password|secrets?|credentials?|authorization)[\"']?\s*[:=]\s*)([^\s\"'\[{][^\s,}\]\r\n]*)")
_PRIVATE_KEY = re.compile(r"-----BEGIN [^-\r\n]*PRIVATE KEY(?: BLOCK)?-----.*?(?:-----END [^-\r\n]*PRIVATE KEY(?: BLOCK)?-----|$)", re.S)


class _AccessError(ValueError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _plain(info: os.stat_result) -> bool:
    return not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400)


def _timeline_order(records):
    """Order recorded instants only; unavailable clocks are not invented.

    为什么要有它：文件顺序、mtime或无时区时间不能证明执行先后；相同时间只保持稳定显示，不宣称因果顺序。
    """
    dated, undated = [], []
    for original in records:
        row, parsed, errors = dict(original), {}, []
        for field in ("started_at", "finished_at"):
            value = row.get(field)
            if value is None or value == "":
                continue
            try:
                if not isinstance(value, str):
                    raise ValueError("invalid timestamp type")
                instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if instant.tzinfo is None or instant.utcoffset() is None:
                    errors.append(field + ":timezone_missing")
                    continue
                parsed[field] = instant.astimezone(timezone.utc)
                row[field + "_utc"] = parsed[field].isoformat().replace("+00:00", "Z")
            except (ValueError, OverflowError):
                errors.append(field + ":invalid")
        if parsed.get("finished_at") and parsed.get("started_at") and parsed["finished_at"] < parsed["started_at"]:
            errors.append("finished_before_started")
        if errors or not parsed:
            row["time_issues"] = errors or ["recorded_timestamp_missing"]
            undated.append(row)
            continue
        basis = "started_at" if "started_at" in parsed else "finished_at"
        row.update(sort_time_utc=row[basis + "_utc"], time_basis=basis)
        dated.append((parsed[basis], row))
    dated.sort(key=lambda item: item[0])
    return [row for _, row in dated], undated


def _sensitive(key: str, value: Any) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized in _METRIC_KEYS and isinstance(value, (int, float, type(None))):
        return False
    return bool(_SECRET_KEY.search(key)) or normalized in {"token", "tokens", "apikey", "private_key"}


def _safe_text(text: str, notes: list[str], secrets=None) -> str:
    # Scope: inspect full strings before pagination, including encoded JSON payloads.
    # 为什么要有它：分页边界不能切开凭证或让内嵌消息中的密钥绕过结构化脱敏。
    if secrets is None:
        secrets = Redactor.configured().secrets
    for secret in secrets:
        if secret in text:
            notes.append("configured_secret_value")
            text = text.replace(secret, "[redacted]")
    text, private_keys = _PRIVATE_KEY.subn("[redacted private key]", text)
    notes.extend(["private_key_block"] * private_keys)
    stripped = text.lstrip()
    if stripped.startswith(("{", "[", '"')):
        try:
            nested = json.loads(text)
        except (ValueError, RecursionError):
            nested = None
        if isinstance(nested, (dict, list)):
            return json.dumps(_sanitize(nested, notes, secrets=secrets), ensure_ascii=False, separators=(",", ":"))
        if isinstance(nested, str) and nested != text:
            return json.dumps(_safe_text(nested, notes, secrets), ensure_ascii=False)
    lines = text.splitlines(keepends=True)
    if len(lines) > 1:
        try:
            values = [json.loads(line) if line.strip() else None for line in lines]
        except (ValueError, RecursionError):
            pass
        else:
            return "".join((json.dumps(_sanitize(value, notes, secrets=secrets), ensure_ascii=False) +
                            ("\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""))
                           if line.strip() else line for line, value in zip(lines, values))
    container = _CONTAINER_SECRET.search(text)
    if container:
        notes.append("malformed_sensitive_structure_tail_withheld")
        text = text[:container.start()] + container.group(1) + '"[redacted: unparseable sensitive value and following text]"'
    changed, count = _TEXT_SECRET.subn(lambda m: m.group(1) + m.group(2) + "[redacted]" + m.group(2), text)
    changed, headers = _AUTH_HEADER.subn(lambda m: m.group(1) + "[redacted]", changed)
    changed, bearers = _BEARER.subn(lambda m: m.group(1) + "[redacted]", changed)
    changed, unquoted = _UNQUOTED_SECRET.subn(lambda m: m.group(1) + "[redacted]", changed)
    changed, lines = _LINE_SECRET.subn(lambda m: m.group(1) + "[redacted]", changed)
    notes.extend(["sensitive_text"] * (count + headers + unquoted + lines + bearers))
    return changed


def _sanitize(value: Any, notes: list[str], path=(), *, secrets=None) -> Any:
    if secrets is None:
        secrets = Redactor.configured().secrets
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            safe_key = _safe_text(str(key), notes, secrets)
            if safe_key in result and safe_key != key:
                safe_key += f" [redacted-key-{index}]"
            sensitive_schema_value = str(key) in {"default", "examples", "example", "const", "enum"} and any(
                part == "properties" and index + 1 < len(path) and _sensitive(path[index + 1], None)
                for index, part in enumerate(path))
            schema_property = bool(path and path[-1] == "properties" and isinstance(item, dict)
                                   and any(name in item for name in ("type", "$ref", "oneOf", "anyOf", "allOf")))
            bound_reference = str(key).endswith("_ref") and isinstance(item, dict) and isinstance(item.get("path"), str) and (
                "sha256" in item or "artifact_hash" in item)
            if sensitive_schema_value or _sensitive(str(key), item) and not schema_property and not bound_reference:
                notes.append("sensitive_field")
                result[safe_key] = "[redacted]"
            else:
                result[safe_key] = _sanitize(item, notes, (*path, str(key)), secrets=secrets)
        return result
    if isinstance(value, list):
        return [_sanitize(item, notes, path, secrets=secrets) for item in value]
    return _safe_text(value, notes, secrets) if isinstance(value, str) else value


def _historical_import(value):
    return isinstance(value, dict) and (isinstance(value.get("checkpoint_import"), dict)
        or isinstance(value.get("origin"), str) and value["origin"].startswith("historical_"))


class RunRecordBrowser:
    def __init__(self, allowed_roots):
        self.roots: dict[str, Path] = {}
        self.issues = []
        if isinstance(allowed_roots, (str, Path)):
            raise ValueError("allowed_roots must be an explicit list of local directories")
        for item in allowed_roots:
            try:
                root = Path(item)
                if not root.is_absolute() or root == Path(root.anchor):
                    raise _AccessError("forbidden", "Select a concrete task directory, not a volume root")
                for part in [root, *root.parents]:
                    if not _plain(part.lstat()):
                        raise _AccessError("forbidden", "Linked or reparse directory is not an allowed root")
                root = root.resolve(strict=True)
                if not root.is_dir():
                    raise _AccessError("forbidden", "Allowed root is not a directory")
                root_id = hashlib.sha256(str(root).encode()).hexdigest()[:24]
                self.roots[root_id] = root
            except (OSError, ValueError) as error:
                self.issues.append({"status": getattr(error, "status", "missing"), "message": str(error)})

    def _id(self, root_id: str, relative: str) -> str:
        raw = json.dumps([root_id, relative], separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def _decode(self, locator: str) -> tuple[str, str, Path]:
        try:
            root_id, relative = json.loads(base64.urlsafe_b64decode(locator + "=" * (-len(locator) % 4)))
            if root_id not in self.roots or not isinstance(relative, str):
                raise ValueError()
            path = self._path(root_id, relative)
            return root_id, relative, path
        except _AccessError:
            raise
        except (ValueError, TypeError, OSError):
            raise _AccessError("forbidden", "Unknown or unauthorized record locator") from None

    def _path(self, root_id: str, relative: str) -> Path:
        if ("\\" in relative or ":" in relative or "\x00" in relative
                or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute()
                or any(part in {"..", "."} for part in relative.split("/") if part)):
            raise _AccessError("forbidden", "Record path is not a safe relative path")
        root = self.roots[root_id]
        path = root
        for part in PurePosixPath(relative).parts:
            path = path / part
            try:
                info = path.lstat()
            except FileNotFoundError:
                raise _AccessError("missing", "Recorded file or directory is missing") from None
            if not _plain(info):
                raise _AccessError("forbidden", "Linked or reparse entries are not readable")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise _AccessError("forbidden", "Record escapes its allowed root")
        return resolved

    def _locator(self, path: Path) -> str | None:
        for key, root in sorted(self.roots.items(), key=lambda item: len(str(item[1])), reverse=True):
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            return self._id(key, "" if relative == "." else relative)
        return None

    def _read(self, locator: str, expected=None, *, retain=True):
        key, relative, path = self._decode(locator)
        # Scope: even explicitly selected local roots do not enable a credential-content viewer.
        # 为什么要有它：.env、私钥文件不能通过普通日志或哈希引用入口绕过默认凭证保护。
        if retain and (path.name.lower() == ".env" or path.name.lower().startswith(".env.")
                       or path.suffix.lower() in {".key", ".p12", ".pfx"}
                       or path.name.lower() in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}):
            raise _AccessError("forbidden", "Credential file content is not exposed by the run-record browser")
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise _AccessError("forbidden", "Only regular files can be read")
        digest, blocks = hashlib.sha256(), []
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise _AccessError("changed", "Record changed before opening")
            while block := handle.read(64 * 1024):
                digest.update(block)
                if retain:
                    blocks.append(block)
            after = os.fstat(handle.fileno())
        current = self._path(key, relative).stat()
        if any((item.st_size, item.st_mtime_ns, item.st_ino) !=
               (before.st_size, before.st_mtime_ns, before.st_ino) for item in (after, current)):
            raise _AccessError("changed", "Record changed during inspection; reload its index")
        sha = digest.hexdigest()
        if expected is not None and sha != expected:
            raise _AccessError("hash_mismatch", "Recorded byte SHA256 does not match")
        return path, b"".join(blocks), sha

    def _walk(self, root: Path):
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                entries = sorted(directory.iterdir(), key=lambda item: item.name)
            except OSError as error:
                yield directory, True, {"status": "read_error", "message": str(error)}
                continue
            yield directory, True, None
            subdirectories = []
            for path in entries:
                try:
                    info = path.lstat()
                    if not _plain(info):
                        yield path, False, {"status": "forbidden", "message": "Link or reparse entry skipped"}
                    elif stat.S_ISDIR(info.st_mode):
                        subdirectories.append(path)
                    elif stat.S_ISREG(info.st_mode):
                        yield path, False, None
                except OSError as error:
                    yield path, False, {"status": "read_error", "message": str(error)}
            pending.extend(reversed(subdirectories))

    def _object(self, path: Path):
        locator = self._locator(path)
        if locator is None or path.stat().st_size > _INDEX_BYTES:
            return None, None
        _, raw, sha = self._read(locator)
        value = json.loads(raw)
        return (value if isinstance(value, dict) else None), sha

    def _signature(self, path: Path):
        damaged = None
        for name, kind in (("run_manifest.json", "product"), ("controller-result.json", "experiment"),
                           ("start.json", "experiment"), ("run_result.json", "product"),
                           ("result.json", "retained_or_generic_run")):
            file = path / name
            try:
                if not file.is_file():
                    continue
                value, sha = self._object(file)
                if value is None:
                    continue
                if name == "run_manifest.json" and not isinstance(value.get("run_id"), str):
                    continue
                if name in {"start.json", "controller-result.json"} and not ("protocol_ref" in value or "candidate_mode" in value):
                    continue
                if name == "result.json" and not any(key in value for key in (
                        "run_id", "checkpoint_ref", "delivery_ref", "delivery", "candidate_mode", "initial_result_ref", "run_result_ref", "scope")):
                    continue
                return {"run_id": self._locator(path), "run_root": str(path), "format": kind,
                        "status": value.get("status") or value.get("current_stage") or "unknown", "identity_source": name,
                        "summary_ref": {"path": str(file), "sha256": sha},
                        "recorded_run_id": value.get("run_id"), "live_process_status": "not_queried"}
            except (OSError, ValueError, UnicodeError) as error:
                damaged = (file, name)
                self.issues.append({"path": str(file), "status": getattr(error, "status", "corrupt"),
                                    "message": "Run identification record could not be parsed or read"})
                continue
        if damaged:
            file, name = damaged
            return {"run_id": self._locator(path), "run_root": str(path), "format": "incomplete_run",
                    "status": "corrupt", "identity_source": name, "summary_ref": {"path": str(file), "sha256": None},
                    "recorded_run_id": None, "live_process_status": "not_queried"}
        return None

    def discover(self, *, cursor=0, limit=100):
        if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("Invalid discovery page")
        runs, issues, visited, scanned, known_roots = [], self.issues, 0, 0, []
        cap = max(1000, limit * 100)
        for root in self.roots.values():
            for path, directory, issue in self._walk(root):
                visited += 1
                found = None
                if directory and issue is None:
                    ancestors = [(old, kind) for old, kind in known_roots if path != old and path.is_relative_to(old)]
                    if not ancestors:
                        found = self._signature(path)
                    elif max(ancestors, key=lambda item: len(item[0].parts))[1] == "experiment":
                        manifest_path = path / "run_manifest.json"
                        try:
                            manifest, _ = self._object(manifest_path) if manifest_path.is_file() else (None, None)
                            if isinstance(manifest, dict) and isinstance(manifest.get("run_id"), str):
                                found = self._signature(path)
                        except (OSError, ValueError, UnicodeError):
                            pass
                    if found:
                        known_roots.append((path, found["format"]))
                if visited <= cursor:
                    continue
                scanned += 1
                if issue:
                    issues.append({**issue, "path": str(path)})
                elif found:
                    runs.append(found)
                if len(runs) >= limit or scanned >= cap:
                    return {"runs": runs, "issues": issues, "next_cursor": visited, "complete": False,
                            "scan_entries": scanned, "scan_limit": cap, "read_only": True}
        return {"runs": runs, "issues": issues, "next_cursor": None, "complete": True,
                "scan_entries": scanned, "scan_limit": cap, "read_only": True}

    def _owner_root(self, path: Path):
        owner, owner_kind, product_id = None, None, None
        for parent in path.parents:
            if self._locator(parent) is None:
                break
            signature = self._signature(parent)
            if not signature:
                continue
            kind = signature["format"]
            if kind == "product":
                recorded_id = signature.get("recorded_run_id")
                if owner_kind != "product" or recorded_id == product_id and isinstance(recorded_id, str):
                    owner, owner_kind, product_id = parent, kind, recorded_id
                else:
                    break
            elif owner is None or owner_kind == "retained_or_generic_run" and kind == "experiment":
                owner, owner_kind = parent, kind
        return owner

    def _references(self, value, *, source_id: str, source_path: Path, run_root=None):
        links = []
        owner = run_root if run_root is not None else self._owner_root(source_path)
        pending = [(value, "")]
        while pending:
            item, pointer = pending.pop()
            if isinstance(item, dict):
                if isinstance(item.get("path"), str) and ("sha256" in item or "artifact_hash" in item):
                    target = Path(item["path"])
                    if not target.is_absolute():
                        target = source_path.parent / target
                    locator = self._locator(target)
                    link = {"source_record_id": source_id, "json_pointer": pointer,
                            "target_record_id": locator, "target_path": str(target) if locator else "[outside_allowed_roots]",
                            "relationship": "recorded_reference", "historical": not target.is_relative_to(owner) if owner else None,
                            "expected_sha256": item.get("sha256"), "artifact_hash": item.get("artifact_hash"),
                            "integrity": "unknown", "status": "forbidden" if locator is None else "unknown"}
                    if locator:
                        try:
                            _, _, sha = self._read(locator, item.get("sha256"), retain=False)
                            link.update(status="ok", integrity="verified_bytes" if "sha256" in item else "semantic_hash_not_byte_sha256", actual_sha256=sha)
                            if "artifact_hash" in item:
                                from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
                                try:
                                    _, envelope_bytes, _ = self._read(locator)
                                    envelope = json.loads(envelope_bytes)
                                    if not isinstance(envelope, dict) or not isinstance(envelope.get("artifact_hash"), str):
                                        raise KeyError("Artifact envelope identity is unavailable")
                                    if envelope["artifact_hash"] != item["artifact_hash"]:
                                        raise ValueError("Artifact envelope identity mismatch")
                                    parts = target.parts
                                    position = parts.index("features")
                                    store = ArtifactStore(Path(*parts[:position]))
                                    store.read(ArtifactRef(feature_id=item.get("feature_id", envelope["feature_id"]),
                                        artifact_type=item.get("artifact_type", envelope["artifact_type"]),
                                        path=str(target), artifact_hash=item["artifact_hash"]))
                                    link["integrity"] = "verified_artifact_semantics"
                                except (ValueError, KeyError, TypeError) as error:
                                    mismatch = "hash mismatch" in str(error).lower() or "identity mismatch" in str(error).lower()
                                    link.update(status="hash_mismatch" if mismatch else "unknown",
                                                integrity="semantic_hash_mismatch" if mismatch else "semantic_validation_unavailable")
                        except (OSError, ValueError) as error:
                            link.update(status=getattr(error, "status", "read_error"), integrity=getattr(error, "status", "read_error"))
                    links.append(link)
                pending.extend((child, pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")) for key, child in item.items())
            elif isinstance(item, list):
                pending.extend((child, pointer + "/" + str(index)) for index, child in enumerate(item))
        return links

    def _run_summary(self, run, root, all_files):
        calls, terminals, timeline, issues, incoming, terminal_links = {}, [], [], [], {}, []
        outcome_read_errors = []
        has_request_records = False
        for path, issue in all_files:
            if issue or not (path.name in {"run_manifest.json", "run_result.json", "controller-result.json", "result.json"}
                             or path.parent.name == "events"):
                continue
            try:
                document, document_sha = self._object(path)
                if not isinstance(document, dict):
                    continue
                source_id = self._locator(path)
                if path.name == "run_manifest.json":
                    for relation in self._references(document, source_id=source_id, source_path=path, run_root=root):
                        incoming.setdefault(relation.get("target_record_id"), []).append(relation)
                        terminal_links.append(relation)
                records = [document]
                if isinstance(document.get("execution_observations"), dict):
                    records = list(document["execution_observations"].values())
                for item in records:
                    if not isinstance(item, dict):
                        continue
                    attributes = item.get("attributes") or {}
                    if item.get("name") == "model.request" and isinstance(attributes, dict):
                        has_request_records = True
                        identity = (item.get("trace_id"), item.get("id"))
                        if not all(isinstance(value, str) for value in identity):
                            issues.append("Model request lacks a trace/span identity; it is not guessed or counted")
                            continue
                        old = calls.get(identity)
                        if old is None or item.get("finished_at") or not old[0].get("finished_at"):
                            calls[identity] = (item, source_id)
                if path.name in {"run_result.json", "controller-result.json", "result.json"}:
                    relations = self._references(document, source_id=source_id, source_path=path, run_root=root)
                    terminal_links.extend(relations)
                    product = document.get("product_execution_result") or {}
                    if not isinstance(product, dict):
                        product = {}
                    relative = path.relative_to(root)
                    association = None
                    recorded_run = run.get("recorded_run_id")
                    if (relative.parts[0] == "continuations" and path.name == "run_result.json"
                            and recorded_run and document.get("run_id") == recorded_run
                            and document.get("parent_run_id", recorded_run) == recorded_run):
                        parent_links = [link for link in relations if link["json_pointer"] == "/parent_result_ref"]
                        if not parent_links or any(link.get("target_record_id") == self._locator(root / "run_result.json") for link in parent_links):
                            association = "recorded_parent_result_reference" if parent_links else "same_run_identity_in_continuations"
                    if (len(relative.parts) == 2 and (relative.parts[0] == "delivery" or relative.parts[0].startswith("delivery-"))
                            and type(document.get("delivery_complete")) is bool and type(document.get("acceptance_passed")) is bool
                            and isinstance(document.get("source_origin"), str)
                            and isinstance(document.get("execution_refs"), list) and isinstance(document.get("evidence_refs"), list)):
                        association = "delivery_record_contract_in_selected_run"
                    terminals.append({"record_id": source_id, "path": relative.as_posix(), "sha256": document_sha,
                        "status": document.get("status") or document.get("current_stage") or "unknown",
                        "reported_status": document.get("status") or document.get("current_stage") or "unknown",
                        "recorded_status": document.get("status"), "current_stage": document.get("current_stage"),
                        "platform_status": product.get("platform_status"), "execution_disposition": product.get("execution_disposition"),
                        "status_source": "/status" if document.get("status") else "/current_stage" if document.get("current_stage") else None,
                        "completion": document.get("completion_classification"),
                        "entry": document.get("entry"), "stage": document.get("stage"), "role": document.get("role"),
                        "source_origin": document.get("source_origin"),
                        "association_basis": association,
                        "verification_scope": document.get("candidate_verification_scope", document.get("verification_scope", document.get("scope"))),
                        "total_wall_seconds": document.get("total_wall_seconds"),
                        "integrity": "referenced_bytes_invalid" if any(link["status"] in {"hash_mismatch", "corrupt", "changed"} for link in relations) else "file_bytes_read",
                        "reference_access_complete": all(link["status"] == "ok" for link in relations),
                        "reference_count": len(relations),
                        "unverified_reference_count": sum(link["status"] != "ok" for link in relations),
                        "basis": "reported_state_not_independent_delivery_verdict"})
            except (OSError, ValueError, UnicodeError) as error:
                issues.append({"record_id": self._locator(path), "status": getattr(error, "status", "corrupt"), "message": str(error)})
                relative = path.relative_to(root)
                if (len(relative.parts) == 1 or relative.parts[0] == "continuations"
                        or relative.parts[0] == "delivery" or relative.parts[0].startswith("delivery-")):
                    outcome_read_errors.append({"record_id": self._locator(path), "path": relative.as_posix(),
                        "status": getattr(error, "status", "corrupt")})
        started, usages = [], []
        for item, source_id in calls.values():
            attributes = item["attributes"]
            _, _, source_path = self._decode(source_id)
            relations = self._references(attributes.get("local_detail_refs", {}), source_id=source_id,
                                         source_path=source_path, run_root=root)
            timeline.append({"record_id": source_id, "name": "model.request", "status": item.get("status", "unknown"),
                "started_at": item.get("started_at"), "finished_at": item.get("finished_at"), "kind": "model_request",
                "refs": relations, "basis": "unique_recorded_trace_and_span_identity_not_live_process",
                "trace_id": item.get("trace_id"), "span_id": item.get("id"),
                "transport_started": attributes.get("transport_started"), "usage": attributes.get("usage")})
            if attributes.get("transport_started") is True:
                started.append(item)
                usage = attributes.get("usage")
                if isinstance(usage, dict) and all(type(usage.get(key)) is int for key in ("input_tokens", "output_tokens")):
                    usages.append(usage)
        if len(usages) != len(started):
            issues.append("Some started model requests have no complete recorded usage; displayed token sums are known portions only")
        preferred = "run_result.json" if run["format"] == "product" else "trial/result.json"
        for terminal in terminals:
            bound = incoming.get(terminal["record_id"], [])
            terminal["incoming_reference_count"] = len(bound)
            if any(link["status"] in {"hash_mismatch", "corrupt", "changed"} for link in bound):
                terminal["integrity"] = "referenced_bytes_invalid"
            terminal["incoming_unverified_reference_count"] = sum(link["status"] != "ok" for link in bound)
        selected = next((item for item in terminals if item["path"] == preferred), None)
        original = selected or next((item for item in terminals if item["path"] == run["identity_source"]), None)
        followups = [item for item in terminals if item is not original and item["association_basis"] is not None]
        scoped = ([original] if original else []) + followups
        if (any(item["integrity"] == "referenced_bytes_invalid" for item in scoped)
                or any(item["status"] in {"corrupt", "hash_mismatch", "changed"} for item in outcome_read_errors)):
            evidence_status = "invalid"
        elif not scoped:
            evidence_status = "unknown"
        elif outcome_read_errors or not original or any(item["status"] == "unknown" or item["unverified_reference_count"]
                                or item["incoming_unverified_reference_count"] for item in scoped):
            evidence_status = "partial"
        else:
            evidence_status = "complete"
        elapsed = next((item["total_wall_seconds"] for item in terminals
                        if item["path"] == "controller-result.json" and isinstance(item["total_wall_seconds"], (int, float))), None)
        summary = {"status": "integrity_error" if selected and selected["integrity"] == "referenced_bytes_invalid"
                   else selected["status"] if selected else run["status"],
            "reported_status": selected["status"] if selected else run["status"],
            "completion": selected["completion"] if selected else None, "elapsed_seconds": elapsed,
            "input_tokens": sum(item["input_tokens"] for item in usages) if usages else None,
            "output_tokens": sum(item["output_tokens"] for item in usages) if usages else None,
            "model_calls": len(started) if has_request_records else None,
            "model_calls_with_usage": len(usages), "usage_complete": bool(started) and len(usages) == len(started),
            "cost_scope": {"selected_run_root": str(root), "basis": "unique_recorded_model.request_spans_with_transport_started",
                           "coverage": "recorded_spans_only_not_automatically_the_whole_user_task",
                           "outside_reference_calls_added": False},
            "usage_complete_scope": "known_started_model_request_spans_only",
            "terminal_records": terminals,
            "outcome_scopes": {"original": original, "followups": followups, "evidence_status": evidence_status,
                "read_errors": outcome_read_errors,
                "evidence_status_scope": "recorded_files_and_reference_integrity_only_not_independent_business_acceptance"},
            "unknowns": issues + ["Stage terminal records are shown together; later delivery does not rewrite the original result.",
                "Imported candidates/checks and historical budget totals are not new executions.",
                "Upstream routing without an associated request span is not included in these call/token sums; inspect its explicit recorded references separately.",
                "Recorded running is not a live process observation."]}
        return summary, timeline, terminal_links

    def read_record(self, record_id, *, offset=0, limit=65536, expected_sha256=None):
        result = {"record": {"record_id": record_id}, "status": "read_error", "format": "unknown", "json_value": None,
                  "text": "", "range": {"offset": offset, "length": 0, "total_chars": None, "next_offset": None, "complete": False},
                  "redaction": {"applied": False, "count": 0, "notes": []}, "links": [], "issues": [], "read_only": True}
        try:
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= _PAGE_CHARS:
                raise _AccessError("forbidden", "Invalid display page")
            path, raw, sha = self._read(record_id, expected_sha256)
            _, relative, _ = self._decode(record_id)
            result["record"].update(path=str(path), relative_path=relative, size_bytes=len(raw), sha256=sha)
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeError:
                raise _AccessError("unsupported_binary", "This is not a UTF-8 text record") from None
            notes = []
            try:
                value = json.loads(text)
            except (ValueError, RecursionError):
                value = None
                result.update(status="corrupt" if path.suffix.lower() == ".json" else "ok", format="text")
                if path.suffix.lower() == ".jsonl":
                    result["format"] = "jsonl"
                    try:
                        for line in text.splitlines():
                            if line.strip():
                                json.loads(line)
                    except (ValueError, RecursionError):
                        result["status"] = "corrupt"
                text = _safe_text(text, notes)
                if result["status"] == "corrupt":
                    result["issues"].append("Invalid JSON; showing protected text, not a parsed document")
            else:
                result.update(status="ok", format="json")
                result["record"]["historical_import"] = _historical_import(value)
                result["links"] = self._references(value, source_id=record_id, source_path=path)
                safe = _sanitize(value, notes)
                text = json.dumps(safe, ensure_ascii=False, indent=2)
                if len(text) <= _TREE_CHARS:
                    result["json_value"] = safe
            page = text[offset:offset + limit]
            end = offset + len(page)
            result.update(text=page, range={"offset": offset, "length": len(page), "total_chars": len(text),
                          "next_offset": end if end < len(text) else None, "complete": end >= len(text)})
            result["record"]["format"] = result["format"]
            result["redaction"] = {"applied": bool(notes), "count": len(notes), "notes": sorted(set(notes)) + [
                "Display offsets count protected text characters; SHA256 identifies original bytes.",
                "Pagination limits display size; this version parses the selected document in memory before redaction."]}
        except (OSError, ValueError, MemoryError, RecursionError) as error:
            result.update(status=getattr(error, "status", "read_error"), text="", json_value=None)
            result["issues"].append(str(error))
        return result

    def _trace_sources(self, all_files):
        """Read original trace identities; never manufacture a manifest for event files.

        为什么要有它：云端和本地关联必须来自相同原件，缺少结束回执或身份时只能显示未记录。
        """
        traces, observations, issues = [], [], []
        for path, issue in all_files:
            if issue or not (path.name == "run_manifest.json" or path.parent.name == "events"):
                continue
            try:
                document, digest = self._object(path)
                if not isinstance(document, dict):
                    issues.append({"record_id": self._locator(path), "status": "trace_source_not_indexed",
                                   "message": "Trace source is too large or not a JSON object; use the original record viewer."})
                    continue
                source = {"path": str(path), "sha256": digest, "record_id": self._locator(path)}
                if isinstance(document.get("execution_trace"), dict):
                    traces.append({"trace": document["execution_trace"],
                                   "source": {**source, "json_pointer": "/execution_trace"}})
                if document.get("phase") == "trace_finished" and isinstance(document.get("summary"), dict):
                    traces.append({"trace": document["summary"],
                                   "source": {**source, "json_pointer": "/summary"}})
                rows = document.get("execution_observations")
                if isinstance(rows, dict):
                    for key, row in rows.items():
                        if isinstance(row, dict):
                            pointer = str(key).replace("~", "~0").replace("/", "~1")
                            observations.append({"observation": row,
                                "source": {**source, "json_pointer": "/execution_observations/" + pointer}})
                elif isinstance(document.get("trace_id"), str) and isinstance(document.get("id"), str):
                    observations.append({"observation": document, "source": {**source, "json_pointer": ""}})
            except (OSError, ValueError, UnicodeError) as error:
                issues.append({"record_id": self._locator(path), "status": getattr(error, "status", "corrupt")})
        identity = sorted({(item["source"]["path"], item["source"]["sha256"])
                           for item in traces + observations})
        fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        return traces, observations, fingerprint, issues

    def verify_cloud(self, run_id, trace_id, *, expected_source_fingerprint):
        from .cloud_trace_link import verify_cloud_trace

        _, _, root = self._decode(run_id)
        if not root.is_dir() or self._signature(root) is None:
            raise ValueError("Run locator must identify a recognized run directory")
        files = [(path, issue) for path, directory, issue in self._walk(root)
                 if not directory and path.suffix.lower() == ".json"]
        _, observations, fingerprint, issues = self._trace_sources(files)
        if fingerprint != expected_source_fingerprint:
            return {"status": "local_sources_changed", "network_performed": False,
                    "message": "Refresh the local run before verifying its cloud association."}
        if issues:
            return {"status": "local_sources_unreadable", "network_performed": False, "issues": issues}
        selected = [item for item in observations if item["observation"].get("trace_id") == trace_id]
        if not selected:
            return {"status": "trace_not_in_selected_run", "network_performed": False}
        result = verify_cloud_trace(trace_id, selected)
        final_files = [(path, issue) for path, directory, issue in self._walk(root)
                       if not directory and path.suffix.lower() == ".json"]
        _, _, final_fingerprint, final_issues = self._trace_sources(final_files)
        if final_fingerprint != fingerprint or final_issues:
            return {"status": "local_sources_changed_during_cloud_check",
                    "network_performed": result.get("network_performed", False),
                    "checked_at": result.get("checked_at"), "source_fingerprint": fingerprint,
                    "message": "Cloud response cannot verify the refreshed local source version; refresh before another check."}
        return {**result, "source_fingerprint": fingerprint}

    def _record_timeline(self, root, all_files):
        """Enumerate bounded JSON events independently of the record-index page.

        为什么要有它：翻页只改变文件索引，不应删掉先前事件；大文件或无法读取的记录明确省略，原件仍走正常读取入口。
        """
        rows, links, omissions = [], [], []
        inspected = 0
        for path, issue in all_files:
            locator = self._locator(path)
            omitted = {"record_id": locator, "path": path.relative_to(root).as_posix()}
            if issue or path.suffix.lower() != ".json":
                omissions.append({**omitted, "reason": issue["status"] if issue else "jsonl_not_indexed_as_single_event"})
                continue
            try:
                size = path.stat().st_size
                if size > _TIMELINE_BYTES:
                    omissions.append({**omitted, "reason": "above_timeline_json_limit", "size_bytes": size})
                    continue
                value, digest = self._object(path)
                inspected += 1
                if not isinstance(value, dict):
                    continue
                events = [(value, "")]
                if isinstance(value.get("execution_observations"), dict):
                    events.extend((item, "/execution_observations/" + str(key).replace("~", "~0").replace("/", "~1"))
                        for key, item in value["execution_observations"].items())
                for event, pointer in events:
                    if (not isinstance(event, dict) or event.get("name") == "model.request"
                            or not any(name in event for name in ("status", "name", "event", "phase"))):
                        continue
                    relations = self._references(event, source_id=locator, source_path=path, run_root=root)
                    if pointer:
                        relations = [{**link, "json_pointer": pointer + link["json_pointer"]} for link in relations]
                    links.extend(relations)
                    attributes = event.get("attributes") if isinstance(event.get("attributes"), dict) else {}
                    row = {"record_id": locator, "sha256": digest, "json_pointer": pointer,
                        "name": event.get("name", event.get("event", event.get("phase", path.name))),
                        "status": event.get("status", "unknown"), "started_at": event.get("started_at"),
                        "finished_at": event.get("finished_at"), "kind": event.get("kind", "record"), "refs": relations,
                        "basis": "recorded_file_not_live_process_or_causal_diagnosis", "historical_import": _historical_import(event)}
                    for field in ("role", "stage", "phase"):
                        if isinstance(event.get(field), str):
                            row[field] = event[field]
                        elif isinstance(attributes.get(field), str):
                            row[field] = attributes[field]
                    rows.append(row)
            except (OSError, ValueError, UnicodeError) as error:
                omissions.append({**omitted, "reason": getattr(error, "status", "corrupt")})
        return rows, links, {"record_index_pagination_independent": True, "small_json_limit_bytes": _TIMELINE_BYTES,
            "json_records_inspected": inspected, "record_count": len(all_files), "omitted_records": omissions,
            "complete": not omissions, "scope": "recorded_small_JSON_events_plus_unique_model_request_spans",
            "ordering": "timezone_aware_recorded_instants_only; ties_preserve_display_order_without_causal_claim"}

    def inspect(self, run_id, *, record_cursor=0, record_limit=200):
        key, _, root = self._decode(run_id)
        if not root.is_dir():
            raise ValueError("Run locator must identify a directory")
        run = self._signature(root)
        if run is None:
            raise ValueError("Selected directory has no recognized run record")
        records, links, issues = [], [], list(self.issues)
        all_files = []
        for path, directory, issue in self._walk(root):
            if issue:
                issues.append({**issue, "path": str(path)})
            if not directory and path.suffix.lower() in {".json", ".jsonl"}:
                all_files.append((path, issue))
        if type(record_cursor) is not int or record_cursor < 0 or type(record_limit) is not int or not 1 <= record_limit <= 1000:
            raise ValueError("Invalid record index page")
        for path, issue in all_files[record_cursor:record_cursor + record_limit]:
            locator = self._locator(path)
            item = {"record_id": locator, "relative_path": path.relative_to(root).as_posix(), "size_bytes": None,
                    "format": "json" if path.suffix == ".json" else "jsonl", "status": issue["status"] if issue else "unread"}
            records.append(item)
            if issue:
                continue
            try:
                item["size_bytes"] = path.stat().st_size
                if item["size_bytes"] > _INDEX_BYTES:
                    item["status"] = "available_on_demand"
                    continue
                value, digest = self._object(path)
                item.update(status="ok", sha256=digest, historical_import=_historical_import(value))
                if value is None:
                    continue
                relations = self._references(value, source_id=locator, source_path=path, run_root=root)
                links.extend(relations)
            except (OSError, ValueError, UnicodeError) as error:
                item["status"] = getattr(error, "status", "corrupt")
                issues.append({"record_id": locator, "status": item["status"], "message": str(error)})
        summary, model_timeline, terminal_links = self._run_summary(run, root, all_files)
        from .cloud_trace_link import cloud_trace_summary

        traces, observations, fingerprint, trace_issues = self._trace_sources(all_files)
        summary["cloud_association"] = {**cloud_trace_summary(traces, observations),
            "source_fingerprint": fingerprint, "source_issues": trace_issues}
        event_timeline, event_links, coverage = self._record_timeline(root, all_files)
        combined = links + terminal_links + event_links
        links = list({(item["source_record_id"], item["json_pointer"], item["target_path"],
                       item.get("expected_sha256"), item.get("artifact_hash")): item for item in combined}.values())
        timeline, undated = _timeline_order(model_timeline + event_timeline)
        coverage.update(dated_records=len(timeline), undated_records=len(undated))
        next_cursor = record_cursor + len(records)
        return {"run": run, "summary": summary, "timeline": timeline, "records": records, "links": links,
                "undated_records": undated, "timeline_coverage": coverage,
                "issues": issues, "next_record_cursor": next_cursor if next_cursor < len(all_files) else None,
                "index_complete": next_cursor >= len(all_files), "record_count": len(all_files), "read_only": True,
                "scope": "Existing files under the selected run; external references never expand allowed_roots."}
