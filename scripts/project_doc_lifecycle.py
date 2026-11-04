"""Explicit, preimage-bound draft updates and publication into repository text.

为什么要有它：固定草稿身份、真实决定和原文哈希，避免每次讨论生成一份无人维护的新正文。
"""
import hashlib
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from scripts.project_docs import (
    INDEX,
    PAGE,
    REGISTRY,
    TEXT_HASH_BASIS,
    DocumentError,
    catalog,
    normalized_text,
    page,
    require,
    safe_path,
    serialize,
    write_atomic,
)

FORMAL = {"specs", "adr", "research", "reports"}
KINDS = ("spec", "prd", "adr", "analysis", "report", "plan", "document")


def register_commands(commands):
    draft = commands.add_parser("draft", help="创建或按原文哈希更新同一 ID 的 Markdown 草稿")
    draft.add_argument("--id", required=True)
    draft.add_argument("--title", required=True)
    draft.add_argument("--kind", choices=KINDS, required=True)
    draft.add_argument("--content-file", required=True)
    draft.add_argument("--expected-sha256")
    promote = commands.add_parser("promote", help="按明确决定将草稿晋升为正式正文")
    promote.add_argument("id")
    promote.add_argument("--to", required=True)
    promote.add_argument("--decision-ref", required=True)
    promote.add_argument("--expected-sha256", required=True)
    retire = commands.add_parser("retire", help="保留原文，将废弃草稿退出默认视图")
    retire.add_argument("id")
    retire.add_argument("--reason", required=True)
    retire.add_argument("--decision-ref", required=True)
    retire.add_argument("--expected-sha256", required=True)
    retire.add_argument("--replacement")
    for command in (draft, promote, retire):
        import argparse
        command.add_argument("--root", default=argparse.SUPPRESS)


def raw_sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_id(value):
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value),
        "Draft ID requires 1-64 lowercase letters, digits or hyphens")
    require(value not in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10))}, "Reserved Windows draft ID")


@contextmanager
def locked(root):
    lock = safe_path(root, "docs/.document-lifecycle.lock")
    try:
        stream = lock.open("xb")
    except FileExistsError as error:
        raise DocumentError("Document lifecycle lock exists; inspect the active or interrupted operation") from error
    try:
        with stream:
            stream.write(str(os.getpid()).encode())
            stream.flush()
        yield
    finally:
        lock.unlink()


def verified_draft(root, row, identity, expected):
    require(row is not None and row.get("status") == "draft", "Only registered draft documents may change lifecycle")
    require(row.get("path") == f"docs/drafts/{identity}.md", "Draft path does not match its stable ID")
    raw = safe_path(root, row["path"]).read_bytes()
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)
        and raw_sha256(raw) == expected, "Draft raw SHA-256 changed or is required")
    return raw


def relocated_markdown(root, old_path, target, raw):
    """Relocate common Markdown links; reject forms we cannot preserve reliably.

    为什么要有它：晋升改变正文目录，引用与图片必须仍指向同一仓库文件；代码示例和本页锚点保持原文。
    """
    text = raw.decode("utf-8-sig")
    old = safe_path(root, old_path)

    def destination(token):
        wrapped = token.startswith("<") and token.endswith(">")
        value = token[1:-1] if wrapped else token
        if value.startswith("#"):
            return token
        uri = urlsplit(value)
        if uri.scheme or uri.netloc:
            require(uri.scheme in {"", "https", "http", "mailto"}, "Unsupported Markdown link scheme")
            return token
        require(uri.path and not uri.path.startswith("/") and "\\" not in unquote(uri.path),
            "Unsupported Markdown link path")
        resolved = (old.parent / unquote(uri.path)).resolve()
        require(resolved.is_relative_to(root), "Markdown link escapes repository")
        require(resolved.exists(), "Missing Markdown link target: " + uri.path)
        if resolved == old:
            resolved = target
        relative = Path(os.path.relpath(resolved, target.parent)).as_posix()
        changed = urlunsplit(("", "", quote(relative, safe="/"), uri.query, uri.fragment))
        return "<" + changed + ">" if wrapped else changed

    fence, result = None, []
    inline = re.compile(r'!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s()]+)(?:\s+(?:"[^"\n]*"|\x27[^\x27\n]*\x27|\([^\n)]*\)))?\s*\)')
    for line in text.splitlines(keepends=True):
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if re.match(r"^ {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}\s*$", line):
                fence = None
            result.append(line)
            continue
        if opening:
            fence = opening[1]
            result.append(line)
            continue
        if line.startswith(("    ", "\t")):
            result.append(line)
            continue
        code = [match.span() for match in re.finditer(r"(?<!`)(`+)(?!`)(.*?)\1(?!`)", line)]
        def protected(at):
            return any(start <= at < end for start, end in code)
        for tag in re.finditer(r"<(?:a|img)\b[^>]*(?:href|src)\s*=", line, re.IGNORECASE):
            require(protected(tag.start()), "Use Markdown links instead of inline HTML before promotion")
        replacements, matches = [], []
        definition = re.match(r"^ {0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|[^\s]+)", line)
        if definition:
            replacements.append((*definition.span(1), destination(definition[1])))
        elif re.match(r"^ {0,3}\[[^\]\n]+\]:", line):
            raise DocumentError("Multiline Markdown link definitions require explicit normalization")
        else:
            for match in inline.finditer(line):
                if not protected(match.start()):
                    replacements.append((*match.span(1), destination(match[1])))
                    matches.append(match.span())
            for marker in re.finditer(r"\]\(", line):
                require(protected(marker.start()) or any(start <= marker.start() < end for start, end in matches),
                    "Unsupported Markdown link syntax; normalize the destination before promotion")
        for start, end, replacement in reversed(replacements):
            line = line[:start] + replacement + line[end:]
        result.append(line)
    encoded = "".join(result).encode("utf-8")
    return (b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b"") + encoded


def commit_files(root, updates, expected):
    """Check every location first and restore our own writes on ordinary failures.

    为什么要有它：拒绝旧哈希和路径冲突时不能留下半份正文；写入错误也要保留原内容和原始错误。
    """
    for name in (*updates, INDEX, PAGE):
        path = safe_path(root, name)
        require(not path.exists() or path.is_file(), "Document destination is not a file")
        require(all(not parent.exists() or parent.is_dir() for parent in path.parents if parent != root),
            "Document parent is not a directory")
    backups = {name: safe_path(root, name).read_bytes() if safe_path(root, name).exists() else None
        for name in (*updates, INDEX, PAGE)}
    for name, raw in expected.items():
        path = safe_path(root, name)
        actual = path.read_bytes() if path.exists() else None
        require(actual == raw, "Document changed before write: " + name)
    written, created_files, created_directories = {}, {}, []
    try:
        for name, raw in updates.items():
            path = safe_path(root, name)
            if name in expected:
                actual = path.read_bytes() if path.exists() else None
                require(actual == expected[name], "Document changed before write: " + name)
            absent = [parent for parent in path.parents if parent != root and not parent.exists()]
            for parent in reversed(absent):
                parent.mkdir()
                created_directories.append(parent)
            if backups[name] is None:
                with path.open("xb") as stream:
                    identity = os.fstat(stream.fileno())
                    created_files[name] = (identity.st_dev, identity.st_ino)
                    written[name] = b""
                    stream.write(raw)
                    written[name] = raw
            else:
                written[name] = raw
                write_atomic(path, raw.decode("utf-8"))
        index = catalog(root)
        for name, raw in {INDEX: serialize(index).encode("utf-8"), PAGE: page(index).encode("utf-8")}.items():
            written[name] = raw
            write_atomic(safe_path(root, name), raw.decode("utf-8"))
        return len(index["documents"])
    except BaseException:
        for name, own_content in reversed(list(written.items())):
            path = safe_path(root, name)
            if path.exists():
                identity = path.stat()
                if (created_files.get(name) == (identity.st_dev, identity.st_ino)
                        or backups[name] is None and path.read_bytes() == own_content):
                    # 为什么要有它：独占创建后写入也可能部分失败，清理应依据本次文件身份而非完整内容匹配。
                    path.unlink()
                elif path.read_bytes() == own_content and backups[name] is not None:
                    # Existing UTF-8 input bytes, including CRLF/BOM, survive rollback unchanged.
                    with path.open("wb") as stream:
                        stream.write(backups[name])
        for directory in reversed(created_directories):
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        raise


def run(root, args):
    import json

    safe_id(args.id)
    with locked(root):
        # Read-only validation precedes lifecycle edits, including unrelated broken references.
        catalog(root)
        registry_raw = safe_path(root, REGISTRY).read_bytes()
        registry = json.loads(registry_raw.decode("utf-8-sig"))
        row = next((value for value in registry["documents"] if value["id"] == args.id), None)
        moment = datetime.now(timezone.utc).isoformat(timespec="seconds")
        event = {"action": args.command, "at": moment}
        updates, expected = {}, {REGISTRY: registry_raw}
        if args.command == "draft":
            require(args.title.strip(), "Draft title is required")
            source = safe_path(root, args.content_file)
            require(source.suffix == ".md" and source.is_file(), "Draft content must be a repository Markdown file")
            raw = source.read_bytes()
            raw.decode("utf-8-sig")
            if row is not None:
                prior = verified_draft(root, row, args.id, args.expected_sha256)
                expected[row["path"]] = prior
                event["previous_raw_sha256"] = raw_sha256(prior)
            else:
                require(args.expected_sha256 is None, "New drafts have no expected previous hash")
                name = f"docs/drafts/{args.id}.md"
                require(not safe_path(root, name).exists(), "Draft destination already exists")
                row = {"id": args.id, "path": name, "status": "draft"}
                registry["documents"].append(row)
                expected[name] = None
            row.update(title=args.title.strip(), kind=args.kind)
            source_raw = raw
            raw = relocated_markdown(root, args.content_file, safe_path(root, row["path"]), source_raw)
            event.update(content_file=args.content_file, content_file_raw_sha256=raw_sha256(source_raw),
                raw_sha256=raw_sha256(raw))
            updates[row["path"]] = raw
        else:
            raw = verified_draft(root, row, args.id, args.expected_sha256)
            require(args.decision_ref.strip(), "An explicit decision reference is required")
            expected[row["path"]] = raw
            event.update(decision_ref=args.decision_ref.strip(), raw_sha256=raw_sha256(raw))
            if args.command == "promote":
                target = safe_path(root, args.to)
                parts = Path(args.to).parts
                require(len(parts) >= 3 and parts[0] == "docs" and parts[1] in FORMAL and target.suffix == ".md",
                    "Formal Markdown belongs under docs/specs, docs/adr, docs/research or docs/reports")
                require(not target.exists(), "Formal document destination already exists")
                old_path = row["path"]
                published = relocated_markdown(root, old_path, target, raw)
                link = quote(Path(os.path.relpath(target, safe_path(root, old_path).parent)).as_posix(), safe="/")
                redirect = ("# 文档已迁移\n\n正文已晋升：[查看正式正文](" + link + ").\n").encode("utf-8")
                registry.setdefault("redirects", []).append({"path": old_path, "document_id": args.id,
                    "sha256": raw_sha256(normalized_text(redirect).encode("utf-8")), "hash_basis": TEXT_HASH_BASIS})
                event.update(from_path=old_path, to_path=args.to, published_raw_sha256=raw_sha256(published))
                row.update(path=args.to, status="accepted")
                expected[args.to] = None
                updates.update({args.to: published, old_path: redirect})
            else:
                require(args.reason.strip(), "Retirement reason is required")
                event["reason"] = args.reason.strip()
                if args.replacement:
                    require(args.replacement != args.id and any(value["id"] == args.replacement
                        for value in registry["documents"]), "Unknown or self replacement ID")
                    row["related"] = sorted(set(row.get("related", [])) | {args.replacement})
                    event["replacement"] = args.replacement
                row["status"] = "superseded" if args.replacement else "archived-draft"
        require(isinstance(row.get("lifecycle", []), list), "Invalid document lifecycle history")
        row.setdefault("lifecycle", []).append(deepcopy(event))
        registry["updated_at"] = moment
        updates[REGISTRY] = serialize(registry).encode("utf-8")
        count = commit_files(root, updates, expected)
        return {"status": "ok", "action": args.command, "id": args.id, "path": row["path"],
            "document_status": row["status"], "raw_sha256": raw_sha256(safe_path(root, row["path"]).read_bytes()),
            "indexed_documents": count}
