"""Offline document discovery, safe navigation, and stale-index acceptance."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def repository(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# 项目\n根目录介绍", encoding="utf-8")
    (tmp_path / "docs/spec.md").write_text("# 功能规格\n费用恢复与预算", encoding="utf-8")
    registry = {"schema_version": "project-documents.v1", "repository_url": "https://github.com/example/repo",
        "updated_at": "2026-11-04", "workstreams": [{"id": "preview", "title": "试用版"}],
        "documents": [{"id": "spec", "title": "试用规格", "kind": "spec", "status": "accepted",
            "path": "docs/spec.md", "workstream": "preview", "source": {"commit": "abc", "path": "docs/spec.md"}},
            {"id": "issue", "title": "关联任务", "kind": "issue", "status": "open",
                "url": "https://github.com/example/repo/issues/1", "related": ["spec"]}]}
    write_registry(tmp_path, registry)
    return tmp_path


def write_registry(root, registry):
    (root / "docs/document-registry.json").write_text(json.dumps(registry), encoding="utf-8")


def invoke(root, *args):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "PYTHONUTF8": "1"}
    return subprocess.run([sys.executable, "-m", "scripts.project_docs", *args, "--root", str(root)],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", check=False)


def test_actual_cli_build_check_search_and_unclassified_discovery(repository):
    built = invoke(repository, "build")
    assert built.returncode == 0, built.stderr + built.stdout
    index = json.loads((repository / "docs/document-index.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in index["documents"]] == sorted(row["id"] for row in index["documents"])
    spec = next(row for row in index["documents"] if row["id"] == "spec")
    assert spec["source"]["commit"] == "abc"
    assert len(spec["sha256"]) == 64
    assert next(row for row in index["documents"] if row.get("path") == "README.md")["status"] == "unclassified"
    assert invoke(repository, "check").returncode == 0
    search = invoke(repository, "search", "预算", "--kind", "spec", "--status", "accepted", "--workstream", "preview")
    assert search.returncode == 0, search.stdout
    assert [row["id"] for row in json.loads(search.stdout)["documents"]] == ["spec"]
    assert json.loads(invoke(repository, "search", "预算", "--kind", "adr").stdout)["documents"] == []


@pytest.mark.parametrize("changed", ["source", "registry", "html", "index", "new_document"])
def test_check_detects_changed_inputs_or_generated_views(repository, changed):
    assert invoke(repository, "build").returncode == 0
    if changed == "source":
        (repository / "docs/spec.md").write_text("# 新规格", encoding="utf-8")
    elif changed == "registry":
        registry = json.loads((repository / "docs/document-registry.json").read_text())
        registry["documents"][0]["status"] = "superseded"
        write_registry(repository, registry)
    elif changed == "new_document":
        (repository / "docs/new.md").write_text("# 新报告", encoding="utf-8")
    else:
        target = repository / ("docs/index.html" if changed == "html" else "docs/document-index.json")
        target.write_text("stale", encoding="utf-8")
    result = invoke(repository, "check")
    assert result.returncode != 0
    assert "stale" in result.stdout.lower()


def test_crlf_build_remains_current_after_lf_checkout_without_rewriting_sources(repository):
    original_body = "# 功能规格\r\n费用恢复与预算\r\n".encode()
    source_hash = hashlib.sha256(original_body).hexdigest()
    registry = json.loads((repository / "docs/document-registry.json").read_text())
    registry["documents"][0]["source"]["sha256"] = source_hash
    (repository / "docs/document-registry.json").write_bytes(
        json.dumps(registry, indent=2, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8"))
    (repository / "docs/spec.md").write_bytes(original_body)
    (repository / "README.md").write_bytes("# 项目\r\n根目录介绍\r\n".encode())
    originals = {name: (repository / name).read_bytes()
        for name in ("docs/document-registry.json", "docs/spec.md", "README.md")}
    assert invoke(repository, "build").returncode == 0
    assert all((repository / name).read_bytes() == raw for name, raw in originals.items())
    generated = (repository / "docs/document-index.json").read_bytes()
    for name, raw in originals.items():
        (repository / name).write_bytes(raw.replace(b"\r\n", b"\n"))
    checked = invoke(repository, "check")
    assert checked.returncode == 0, checked.stdout
    assert (repository / "docs/document-index.json").read_bytes() == generated
    index = json.loads(generated)
    spec = next(row for row in index["documents"] if row["id"] == "spec")
    assert spec["hash_basis"] == index["registry_hash_basis"] == "utf8-text-lf-v1"
    assert spec["sha256"] == hashlib.sha256(original_body.replace(b"\r\n", b"\n")).hexdigest()
    assert spec["source"]["sha256"] == source_hash
    (repository / "docs/spec.md").write_bytes(original_body.replace("预算".encode(), b"other"))
    assert invoke(repository, "check").returncode != 0


@pytest.mark.parametrize("fault", ["duplicate_id", "duplicate_path", "unknown_related", "missing_file",
    "escape", "absolute", "backslash", "noncanonical", "javascript", "credentials", "both", "neither"])
def test_registry_errors_are_explicit_and_do_not_generate_output(repository, fault):
    registry = json.loads((repository / "docs/document-registry.json").read_text())
    doc = registry["documents"][0]
    if fault == "duplicate_id":
        registry["documents"][1]["id"] = "spec"
    elif fault == "duplicate_path":
        registry["documents"].append({**doc, "id": "copy"})
    elif fault == "unknown_related":
        doc["related"] = ["missing"]
    elif fault in {"missing_file", "escape", "absolute", "backslash", "noncanonical"}:
        doc["path"] = {"missing_file": "docs/gone.md", "escape": "../outside.md",
            "absolute": str(repository / "docs/spec.md"), "backslash": "docs\\spec.md",
            "noncanonical": "docs//spec.md"}[fault]
    elif fault in {"javascript", "credentials"}:
        registry["documents"][1]["url"] = {"javascript": "javascript:alert(1)",
            "credentials": "https://user:secret@example.com/report"}[fault]
    elif fault == "both":
        doc["url"] = "https://example.com"
    else:
        del doc["path"]
    write_registry(repository, registry)
    result = invoke(repository, "build")
    assert result.returncode != 0
    assert json.loads(result.stdout)["status"] == "error"
    assert not (repository / "docs/index.html").exists()


def test_html_report_is_text_and_data_cannot_break_out_of_json_script(repository):
    payload = '</script><script>alert("xss")</script>'
    registry = json.loads((repository / "docs/document-registry.json").read_text())
    registry["documents"][0]["title"] = payload
    write_registry(repository, registry)
    (repository / "docs/report.html").write_text(
        '<title>架构分析</title><style>hidden-css</style><h1>报告</h1><p>检查结论</p>'
        '<script>hidden-javascript</script>', encoding="utf-8")
    result = invoke(repository, "build")
    assert result.returncode == 0, result.stdout
    html = (repository / "docs/index.html").read_text(encoding="utf-8")
    assert payload not in html
    assert "\\u003c/script\\u003e" in html
    assert "innerHTML" not in html and "fetch(" not in html
    index = json.loads((repository / "docs/document-index.json").read_text(encoding="utf-8"))
    report = next(row for row in index["documents"] if row.get("path") == "docs/report.html")
    assert report["title"] == "架构分析" and "检查结论" in report["text"]
    assert "hidden-javascript" not in report["text"] and "hidden-css" not in report["text"]
    assert next(row for row in index["documents"] if row.get("path") == "README.md")["href"] == "../README.md"
    assert report["href"] == "report.html"


def test_binary_registered_document_is_linked_without_utf8_decode(repository):
    registry = json.loads((repository / "docs/document-registry.json").read_text())
    registry["documents"].append({"id": "pdf", "title": "资料", "kind": "reference",
        "status": "historical", "path": "docs/manual.pdf"})
    (repository / "docs/manual.pdf").write_bytes(b"%PDF\xff\xfe")
    write_registry(repository, registry)
    assert invoke(repository, "build").returncode == 0
    assert invoke(repository, "check").returncode == 0
    index = json.loads((repository / "docs/document-index.json").read_text(encoding="utf-8"))
    assert next(row for row in index["documents"] if row["id"] == "pdf")["hash_basis"] == "raw-bytes-v1"


def test_html_visibility_controls_search_and_explicit_status(repository):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is only needed to execute the generated HTML interaction test")
    registry = json.loads((repository / "docs/document-registry.json").read_text())
    (repository / "docs/old.md").write_text("# 旧分析\n过期证据仍可追溯", encoding="utf-8")
    registry["documents"].append({"id": "old", "title": "旧分析", "kind": "analysis",
        "status": "historical", "path": "docs/old.md"})
    write_registry(repository, registry)
    assert invoke(repository, "build").returncode == 0
    html = (repository / "docs/index.html").read_text(encoding="utf-8")
    assert 'id="include-history" type="checkbox"' in html
    scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    data = json.loads((repository / "docs/document-index.json").read_text(encoding="utf-8"))
    # 为什么要有它：直接执行生成页面的筛选代码，验证可见结果而非复制一份相同的筛选实现。
    harness = r'''
const {script,ids,data}=JSON.parse(require('fs').readFileSync(0,'utf8'));
class Element{constructor(){this.children=[];this.value='';this.textContent='';this.checked=false;this.events={};}
append(...items){this.children.push(...items);}replaceChildren(...items){this.children=[...items];}
addEventListener(name,fn){this.events[name]=fn;}}
const nodes=Object.fromEntries(ids.map(id=>[id,new Element()]));nodes.catalog.textContent=JSON.stringify(data);
const document={getElementById(id){if(!nodes[id])throw Error('missing '+id);return nodes[id];},createElement(){return new Element();}};
require('vm').runInNewContext(script,{document});
const rows=()=>nodes.results.children.map(e=>e.id), result={initial:rows()};
nodes.query.value='过期证据';nodes.query.events.input();result.searched=rows();
nodes.query.value='';nodes.query.events.input();result.cleared=rows();
nodes['include-history'].checked=true;nodes['include-history'].events.change();result.all=rows();
nodes['include-history'].checked=false;nodes.status.value='historical';nodes.status.events.change();result.explicit=rows();
result.labels=nodes.kind.children.map(e=>e.textContent);console.log(JSON.stringify(result));
'''
    run = subprocess.run([node, "-e", harness], input=json.dumps({"script": scripts[0],
        "ids": re.findall(r'id="([^"]+)"', html), "data": data}, ensure_ascii=False),
        text=True, encoding="utf-8", capture_output=True, check=False)
    assert run.returncode == 0, run.stderr
    actual = json.loads(run.stdout)
    assert actual["initial"] == ["issue", "spec"]
    assert actual["searched"] == ["old"]
    assert actual["cleared"] == ["issue", "spec"]
    assert set(actual["all"]) == {row["id"] for row in data["documents"]}
    assert actual["explicit"] == ["old"]
    assert {"分析", "需求规格", "GitHub 任务"} <= set(actual["labels"])


def test_invalid_utf8_prose_is_not_silently_omitted(repository):
    (repository / "docs/spec.md").write_bytes(b"\xff\xfe")
    assert invoke(repository, "build").returncode != 0


def test_symlink_output_cannot_overwrite_outside_repository(repository, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.html")
    outside.write_text("keep", encoding="utf-8")
    try:
        (repository / "docs/index.html").symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this host")
    result = invoke(repository, "build")
    assert result.returncode != 0
    assert outside.read_text(encoding="utf-8") == "keep"


def create_draft(root, identity="sample"):
    result = invoke(root, "draft", "--id", identity, "--title", "维护草稿", "--kind", "spec",
        "--content-file", "docs/spec.md")
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def file_snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_draft_update_promote_is_one_identity_and_redirect_is_not_a_second_document(repository):
    created = create_draft(repository)
    path = repository / "docs/drafts/sample.md"
    assert created["raw_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert created["document_status"] == "draft"
    (repository / "docs/spec.md").write_text("# 维护草稿\nStatus: accepted\n新正文", encoding="utf-8")
    updated = invoke(repository, "draft", "--id", "sample", "--title", "维护草稿", "--kind", "spec",
        "--content-file", "docs/spec.md", "--expected-sha256", created["raw_sha256"])
    assert updated.returncode == 0, updated.stdout
    receipt = json.loads(updated.stdout)
    assert receipt["document_status"] == "draft"  # The body cannot approve itself.
    raw = path.read_bytes()
    promoted = invoke(repository, "promote", "sample", "--to", "docs/specs/sample.md",
        "--decision-ref", "user:approved-sample", "--expected-sha256", receipt["raw_sha256"])
    assert promoted.returncode == 0, promoted.stdout
    assert (repository / "docs/specs/sample.md").read_bytes() == raw
    assert "../specs/sample.md" in path.read_text(encoding="utf-8")
    registry = json.loads((repository / "docs/document-registry.json").read_text(encoding="utf-8"))
    item = next(row for row in registry["documents"] if row["id"] == "sample")
    assert item["path"] == "docs/specs/sample.md" and item["status"] == "accepted"
    assert item["lifecycle"][-1]["decision_ref"] == "user:approved-sample"
    assert item["lifecycle"][-1]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert registry["redirects"][0]["document_id"] == "sample"
    index = json.loads((repository / "docs/document-index.json").read_text(encoding="utf-8"))
    assert sum(row["id"] == "sample" for row in index["documents"]) == 1
    assert not any(row.get("path") == "docs/drafts/sample.md" for row in index["documents"])
    assert invoke(repository, "check").returncode == 0


@pytest.mark.parametrize("operation", ["draft", "promote", "retire"])
def test_old_draft_hash_rejects_without_any_file_changes(repository, operation):
    create_draft(repository)
    before = file_snapshot(repository)
    args = {"draft": ["draft", "--id", "sample", "--title", "new", "--kind", "spec", "--content-file", "docs/spec.md"],
        "promote": ["promote", "sample", "--to", "docs/specs/sample.md", "--decision-ref", "user:ok"],
        "retire": ["retire", "sample", "--reason", "unused", "--decision-ref", "user:ok"]}[operation]
    result = invoke(repository, *args, "--expected-sha256", "0" * 64)
    assert result.returncode != 0
    assert file_snapshot(repository) == before


@pytest.mark.parametrize("target", ["docs/specs/taken.md", "docs/spec.md", "../outside.md", "docs/index.html",
    "docs/document-registry.json", "docs/policies/new.md", "docs/specs/../stolen.md"])
def test_promote_refuses_collisions_and_nonformal_paths_without_effect(repository, target):
    created = create_draft(repository)
    (repository / "docs/specs").mkdir()
    (repository / "docs/specs/taken.md").write_text("Existing formal text", encoding="utf-8")
    before = file_snapshot(repository)
    result = invoke(repository, "promote", "sample", "--to", target, "--decision-ref", "user:ok",
        "--expected-sha256", created["raw_sha256"])
    assert result.returncode != 0
    assert file_snapshot(repository) == before


@pytest.mark.parametrize("identity", ["../escape", "bad:id", "con", "nul", "com1", "bad.", "has space"])
def test_draft_id_is_a_safe_windows_filename(repository, identity):
    before = file_snapshot(repository)
    result = invoke(repository, "draft", "--id", identity, "--title", "草稿", "--kind", "spec",
        "--content-file", "docs/spec.md")
    assert result.returncode != 0
    assert file_snapshot(repository) == before


def test_retire_retains_body_and_records_replacement_and_reason(repository):
    created = create_draft(repository)
    path = repository / "docs/drafts/sample.md"
    original = path.read_bytes()
    retired = invoke(repository, "retire", "sample", "--reason", "已由规格收录", "--decision-ref", "issue:12",
        "--replacement", "spec", "--expected-sha256", created["raw_sha256"])
    assert retired.returncode == 0, retired.stdout
    assert path.read_bytes() == original
    registry = json.loads((repository / "docs/document-registry.json").read_text(encoding="utf-8"))
    item = next(row for row in registry["documents"] if row["id"] == "sample")
    assert item["status"] == "superseded" and "spec" in item["related"]
    assert item["lifecycle"][-1]["reason"] == "已由规格收录"
    assert invoke(repository, "check").returncode == 0


def test_failed_partial_new_file_write_removes_only_its_own_target(repository, monkeypatch):
    from scripts.project_doc_lifecycle import commit_files

    before = file_snapshot(repository)
    original_open = Path.open

    class PartialWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def write(self, raw):
            self.stream.write(raw[:3])
            self.stream.flush()
            raise OSError("injected partial write")

    def fail_new_write(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return PartialWrite(stream) if mode == "xb" and path.name == "sample.md" else stream

    monkeypatch.setattr(Path, "open", fail_new_write)
    with pytest.raises(OSError, match="injected partial write"):
        commit_files(repository, {"docs/drafts/sample.md": b"new document"}, {})
    assert file_snapshot(repository) == before
    assert not (repository / "docs/drafts").exists()


def promotion_args(created):
    from argparse import Namespace
    return Namespace(command="promote", id="sample", to="docs/specs/sample.md",
        decision_ref="user:accepted", expected_sha256=created["raw_sha256"])


def test_keyboard_interrupt_rolls_back_promoted_text_and_registry(repository, monkeypatch):
    from scripts import project_doc_lifecycle as lifecycle

    created = create_draft(repository)
    before = file_snapshot(repository)
    original_write = lifecycle.write_atomic

    def interrupt_registry(path, content):
        if path.name == "document-registry.json":
            raise KeyboardInterrupt()
        original_write(path, content)

    monkeypatch.setattr(lifecycle, "write_atomic", interrupt_registry)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.run(repository, promotion_args(created))
    assert file_snapshot(repository) == before


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_first_generated_view_is_removed_when_second_output_fails(repository, monkeypatch, failure):
    from argparse import Namespace

    from scripts import project_doc_lifecycle as lifecycle

    before = file_snapshot(repository)
    original_write = lifecycle.write_atomic

    def fail_html(path, content):
        if path.name == "index.html":
            raise failure("injected HTML write failure")
        original_write(path, content)

    monkeypatch.setattr(lifecycle, "write_atomic", fail_html)
    with pytest.raises(failure, match="injected HTML write failure"):
        lifecycle.run(repository, Namespace(command="draft", id="sample", title="Draft", kind="spec",
            content_file="docs/spec.md", expected_sha256=None))
    assert file_snapshot(repository) == before


@pytest.mark.parametrize("changed", ["docs/drafts/sample.md", "docs/document-registry.json"])
def test_promote_rechecks_each_preimage_and_preserves_external_changes(repository, monkeypatch, changed):
    from scripts import project_doc_lifecycle as lifecycle

    created = create_draft(repository)
    before = file_snapshot(repository)
    original_open = Path.open
    modified = before[changed] + b"\n "

    class EditAfterTarget:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, *args):
            self.stream.close()
            (repository / changed).write_bytes(modified)

    def edit_after_new_target(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return EditAfterTarget(stream) if mode == "xb" and path == repository / "docs/specs/sample.md" else stream

    monkeypatch.setattr(Path, "open", edit_after_new_target)
    with pytest.raises(ValueError, match="changed before write"):
        lifecycle.run(repository, promotion_args(created))
    assert file_snapshot(repository) == {**before, changed: modified}


def test_promote_preserves_relative_markdown_targets_and_code_examples(repository):
    create_draft(repository)
    (repository / "docs/drafts/other.md").write_text("# Other", encoding="utf-8")
    (repository / "docs/报告图.png").write_bytes(b"image")
    body = ('# Draft\n[other](other.md#section)\n![图](../报告图.png)\n'
        '[ref][x]\n[x]: other.md "Title"\n[web](https://example.com/page)\n[here](#local)\n'
        '`[sample](missing.md)`\n```markdown\n[code](missing.md)\n```\n')
    path = repository / "docs/drafts/sample.md"
    raw = body.encode("utf-8")
    path.write_bytes(raw)
    result = invoke(repository, "promote", "sample", "--to", "docs/specs/topic/sample.md",
        "--decision-ref", "issue:1", "--expected-sha256", hashlib.sha256(raw).hexdigest())
    assert result.returncode == 0, result.stdout + result.stderr
    formal = (repository / "docs/specs/topic/sample.md").read_text(encoding="utf-8")
    assert "[other](../../drafts/other.md#section)" in formal
    assert "![图](../../%E6%8A%A5%E5%91%8A%E5%9B%BE.png)" in formal
    assert '[x]: ../../drafts/other.md "Title"' in formal
    assert "[web](https://example.com/page)" in formal and "[here](#local)" in formal
    assert '`[sample](missing.md)`\n```markdown\n[code](missing.md)\n```' in formal
    registry = json.loads((repository / "docs/document-registry.json").read_text(encoding="utf-8"))
    event = next(row for row in registry["documents"] if row["id"] == "sample")["lifecycle"][-1]
    assert event["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    published_hash = hashlib.sha256((repository / "docs/specs/topic/sample.md").read_bytes()).hexdigest()
    assert event["published_raw_sha256"] == json.loads(result.stdout)["raw_sha256"] == published_hash


@pytest.mark.parametrize("link", ["[missing](missing.md)", "[escape](../../../outside.md)",
    '<img src="../image.png">'])
def test_promote_rejects_broken_or_unsupported_relative_links_without_changes(repository, link):
    create_draft(repository)
    path = repository / "docs/drafts/sample.md"
    path.write_text("# Draft\n" + link, encoding="utf-8")
    before = file_snapshot(repository)
    result = invoke(repository, "promote", "sample", "--to", "docs/specs/sample.md", "--decision-ref", "user:ok",
        "--expected-sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert result.returncode != 0
    assert file_snapshot(repository) == before


def test_draft_copy_relocates_links_from_the_content_file(repository):
    (repository / "docs/spec.md").write_text("# Draft\n[root](../README.md)", encoding="utf-8")
    source_raw = (repository / "docs/spec.md").read_bytes()
    created = create_draft(repository)
    draft = (repository / "docs/drafts/sample.md").read_text(encoding="utf-8")
    assert "[root](../../README.md)" in draft
    assert created["raw_sha256"] != hashlib.sha256(source_raw).hexdigest()
    registry = json.loads((repository / "docs/document-registry.json").read_text(encoding="utf-8"))
    event = next(row for row in registry["documents"] if row["id"] == "sample")["lifecycle"][-1]
    assert event["content_file_raw_sha256"] == hashlib.sha256(source_raw).hexdigest()


@pytest.mark.parametrize("operation", ["draft", "promote", "retire"])
def test_pending_publication_task_cannot_be_promoted_or_rewritten_as_draft(repository, operation):
    created = create_draft(repository)
    registry_path = repository / "docs/document-registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    next(row for row in registry["documents"] if row["id"] == "sample")["status"] = "pending-publication"
    write_registry(repository, registry)
    before = file_snapshot(repository)
    args = {"draft": ["draft", "--id", "sample", "--title", "new", "--kind", "spec", "--content-file", "docs/spec.md"],
        "promote": ["promote", "sample", "--to", "docs/specs/sample.md", "--decision-ref", "user:ok"],
        "retire": ["retire", "sample", "--reason", "unused", "--decision-ref", "user:ok"]}[operation]
    result = invoke(repository, *args, "--expected-sha256", created["raw_sha256"])
    assert result.returncode != 0
    assert file_snapshot(repository) == before
