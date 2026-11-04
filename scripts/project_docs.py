"""Offline document discovery and navigation; repository text remains authoritative.

为什么要有它：一个可重建索引连接正式正文、任务和原件，避免各工作树各自维护一份全文目录。
"""
import argparse
import hashlib
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit
from uuid import uuid4

REGISTRY = "docs/document-registry.json"
INDEX = "docs/document-index.json"
PAGE = "docs/index.html"
PROSE = {".md", ".markdown", ".html", ".htm", ".rst", ".txt", ".json", ".yaml", ".yml"}
DISCOVER = {".md", ".html", ".htm", ".rst"}
TEXT_HASH_BASIS = "utf8-text-lf-v1"


class DocumentError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise DocumentError(message)


def safe_path(root, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
        "Document paths must be repository-relative POSIX paths")
    parts = PurePosixPath(name).parts
    require(not name.startswith("/") and PurePosixPath(name).as_posix() == name
        and all(part not in (".", "..") for part in parts),
        "Document path escapes or is not canonical")
    path = root
    for part in parts:
        path /= part
        require(not path.is_symlink(), "Symlink document paths are unsupported")
    require(path.resolve().is_relative_to(root), "Document path escapes the repository")
    return path


def safe_url(value):
    require(isinstance(value, str) and not any(ord(c) < 33 for c in value) and "\\" not in value,
        "Invalid document URL")
    parsed = urlsplit(value)
    require(parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password,
        "Document URLs require HTTPS without credentials")
    require(parsed.port is None or 0 < parsed.port <= 65535, "Invalid document URL port")
    return value


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.title, self.hidden, self.in_title = [], [], 0, False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "template"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "template"}:
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)
            if self.in_title:
                self.title.append(data)


def normalized_text(raw):
    """Normalize only the rebuildable view; preserve files and source provenance.

    为什么要有它：Git 的 CRLF/LF 签出差异不代表文档过期，原件哈希仍由 source 元数据保留。
    """
    return raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")


def read_document(path):
    raw = path.read_bytes()
    title, text = path.stem, ""
    hashed, hash_basis = raw, "raw-bytes-v1"
    if path.suffix.lower() in PROSE:
        text = normalized_text(raw)
        hashed, hash_basis = text.encode("utf-8"), TEXT_HASH_BASIS
        if path.suffix.lower() in {".html", ".htm"}:
            parser = TextExtractor()
            parser.feed(text)
            title = " ".join(parser.title).strip() or title
            text = " ".join(parser.text)
        else:
            heading = re.search(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
            if heading:
                title = heading[1].strip()
    return title, " ".join(text.split()), hashlib.sha256(hashed).hexdigest(), hash_basis


def discover(root):
    paths = []

    def report_read_error(error):
        raise error

    for directory, children, files in os.walk(root / "docs", followlinks=False, onerror=report_read_error):
        for child in children:
            safe_path(root, (Path(directory) / child).relative_to(root).as_posix())
        for filename in files:
            path = Path(directory) / filename
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() in DISCOVER and relative != PAGE:
                paths.append(relative)
    paths.extend(p.name for p in root.glob("*SPEC.md") if p.is_file())
    if (root / "README.md").is_file():
        paths.append("README.md")
    return sorted(set(paths))


def catalog(root):
    registry_text = normalized_text(safe_path(root, REGISTRY).read_bytes())
    registry = json.loads(registry_text)
    require(isinstance(registry, dict) and registry.get("schema_version") == "project-documents.v1",
        "Unsupported document registry schema")
    documents = registry.get("documents")
    require(isinstance(documents, list), "Registry documents must be a list")
    paths, ids, rows = set(), set(), []
    for entry in documents:
        require(isinstance(entry, dict), "Document entries must be objects")
        require(all(isinstance(entry.get(k), str) and entry[k].strip() for k in ("id", "title", "kind", "status")),
            "Document id, title, kind and status must be nonempty strings")
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", entry["id"]), "Invalid document id")
        require(entry["id"] not in ids, "Duplicate document id: " + entry["id"])
        require(all(key not in entry or isinstance(entry[key], str) for key in ("summary", "workstream")),
            "Document summary and workstream must be strings")
        ids.add(entry["id"])
        require(("path" in entry) != ("url" in entry), "Each document requires exactly one path or URL")
        row = dict(entry)
        if "path" in entry:
            path = safe_path(root, entry["path"])
            require(entry["path"] not in {INDEX, PAGE}, "Generated views cannot register themselves")
            require(path.is_file(), "Missing document: " + entry["path"])
            key = str(path.resolve()).casefold()
            require(key not in paths, "Duplicate document path: " + entry["path"])
            paths.add(key)
        else:
            row["url"] = safe_url(entry["url"])
        related = row.get("related", [])
        require(isinstance(related, list) and all(isinstance(value, str) for value in related), "Invalid related IDs")
        rows.append(row)
    for row in rows:
        require(all(value in ids for value in row.get("related", [])), "Unknown related ID in " + row["id"])
    registered = {row["path"].casefold() for row in rows if "path" in row}
    redirects = registry.get("redirects", [])
    require(isinstance(redirects, list), "Registry redirects must be a list")
    redirected = set()
    for redirect in redirects:
        require(isinstance(redirect, dict) and redirect.get("document_id") in ids, "Unknown redirect document")
        name = redirect.get("path")
        path = safe_path(root, name)
        require(name.startswith("docs/") and name not in {INDEX, PAGE} and name.casefold() not in registered
            and name.casefold() not in redirected, "Invalid or duplicate redirect path")
        require(path.is_file(), "Missing redirect document: " + name)
        require(redirect.get("hash_basis") == TEXT_HASH_BASIS and redirect.get("sha256") ==
            hashlib.sha256(normalized_text(path.read_bytes()).encode("utf-8")).hexdigest(), "Redirect content changed: " + name)
        redirected.add(name.casefold())
    for name in discover(root):
        if name.casefold() not in registered | redirected:
            identity = "doc:" + hashlib.sha256(name.encode()).hexdigest()[:16]
            require(identity not in ids, "Discovered document ID collision")
            ids.add(identity)
            rows.append({"id": identity, "path": name, "title": "", "kind": "document", "status": "unclassified"})
    for row in rows:
        if "path" in row:
            path = safe_path(root, row["path"])
            title, text, digest, hash_basis = read_document(path)
            row.update(title=row["title"] or title, text=text, sha256=digest, hash_basis=hash_basis,
                href=quote(Path(os.path.relpath(path, root / "docs")).as_posix(), safe="/"))
        else:
            row.update(text="", href=row["url"])
    return {**registry, "registry_sha256": hashlib.sha256(registry_text.encode("utf-8")).hexdigest(),
        "registry_hash_basis": TEXT_HASH_BASIS,
        "documents": sorted(rows, key=lambda row: row["id"])}


def serialize(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def page(index):
    # 为什么要有它：报告和标题只作为文本；JSON中的HTML结束标记不能变成索引页面脚本。
    data = serialize(index).replace("&", "\\u0026").replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RepoForge · 文档总览</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172c39;background:#f3f5f6}body{max-width:1120px;margin:auto;padding:32px 20px}
h1{font-size:30px;margin:0}header p{color:#526775;line-height:1.7}nav{display:flex;flex-wrap:wrap;gap:14px;margin:25px 0}
label{display:flex;flex-direction:column;gap:6px;font-size:13px}input,select{padding:12px;border:1px solid #aebcc4;border-radius:7px;background:white;color:#172c39;font-size:15px}
input{width:min(390px,80vw)}.history-toggle{flex-direction:row;align-items:center;align-self:end;padding:12px 0}.history-toggle input{width:auto}#count{color:#526775}article{border:1px solid #d8e1e6;border-radius:10px;background:white;padding:20px;margin:13px 0}
a{color:#075e87;text-decoration:none}a:hover{text-decoration:underline}h2{font-size:19px;margin:0 0 10px}article p{line-height:1.7;margin:8px 0;overflow-wrap:anywhere}
.meta{color:#5a6e7a;font-size:13px}.tag{display:inline-block;background:#eaf2f5;border-radius:4px;padding:3px 8px;margin-right:5px;font-size:12px}footer{margin-top:32px;color:#617481;font-size:13px}
</style>
<header><h1>RepoForge · 文档总览</h1><p>仓库保存正文，GitHub 管理任务与 PR。按主题查找规格、决策、分析和证据，核对状态及来源。</p></header>
<nav aria-label="筛选文档"><label>搜索标题与正文<input id="query" type="search" placeholder="例如：B4、费用恢复、候选编辑"></label>
<label>类型<select id="kind"><option value="">全部类型</option></select></label>
<label>状态<select id="status"><option value="">全部状态</option></select></label>
<label>工作线<select id="workstream"><option value="">全部工作线</option></select></label>
<label class="history-toggle"><input id="include-history" type="checkbox">包含历史与待分类</label></nav>
<p id="count" role="status"></p><main id="results"></main><footer>默认展示当前登记资料；搜索关键词或选择明确状态可查历史。未分类不代表现行；GitHub 状态为登记时快照。</footer>
<script id="catalog" type="application/json">''' + data + '''</script>
<script>
const catalog=JSON.parse(document.getElementById('catalog').textContent), docs=catalog.documents;
const labels={accepted:'已接受',active:'当前',current:'当前',historical:'历史',superseded:'已替代',archived:'已归档','archived-draft':'归档草稿',deprecated:'已弃用',unclassified:'待分类',open:'开放',closed:'关闭',merged:'已合入',draft:'草稿','pending-integration':'待集成','accepted-design':'设计已接受','recorded-unmerged':'已记录，未合入','pending-publication':'待发布',spec:'需求规格',adr:'架构决策',analysis:'分析',report:'报告',issue:'GitHub 任务',pr:'合并请求',task:'实施任务',policy:'项目约定',document:'文档',reference:'参考资料'};
const historicalStates=new Set(['historical','superseded','archived','archived-draft','deprecated','unclassified']);
const filters=['kind','status','workstream'];
for(const field of filters){const select=document.getElementById(field);for(const value of [...new Set(docs.map(d=>d[field]).filter(Boolean))].sort()){const option=document.createElement('option');option.value=value;option.textContent=labels[value]||value;select.append(option);}select.addEventListener('change',render);}
document.getElementById('query').addEventListener('input',render);
document.getElementById('include-history').addEventListener('change',render);
function add(parent,tag,text,cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;parent.append(e);return e;}
function render(){const query=document.getElementById('query').value.trim().toLocaleLowerCase();const includeHistory=document.getElementById('include-history').checked||Boolean(query)||Boolean(document.getElementById('status').value);const found=docs.filter(d=>(includeHistory||!historicalStates.has(d.status))&&filters.every(f=>!document.getElementById(f).value||d[f]===document.getElementById(f).value)&&[d.title,d.summary,d.path,d.text,d.id,d.workstream].filter(Boolean).join(' ').toLocaleLowerCase().includes(query));
const results=document.getElementById('results');results.replaceChildren();document.getElementById('count').textContent=`显示 ${found.length} / ${docs.length} 项 · 登记日期 ${catalog.updated_at||'未知'}`;
for(const d of found){const article=add(results,'article','');article.id=d.id;const h=add(article,'h2','');const a=add(h,'a',d.title);a.href=d.href;if(d.url)a.rel='noopener noreferrer';const meta=add(article,'div','');for(const value of [d.kind,d.status,d.workstream].filter(Boolean))add(meta,'span',labels[value]||value,'tag');if(d.summary)add(article,'p',d.summary);add(article,'p',d.path||d.url,'meta');if(d.source)add(article,'p','来源：'+JSON.stringify(d.source),'meta');if(d.related?.length){const related=add(article,'p','关联：','meta');for(const id of d.related){const target=docs.find(x=>x.id===id);if(target){const link=add(related,'a',target.title+'  ');link.href=target.href;}}}}
}render();
</script></html>
'''


def write_atomic(path, content):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线构建、校验和搜索项目文档索引；不删除正文、不写 GitHub。")
    parser.add_argument("--root", default=".", help="仓库根目录")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "check", "search"):
        command = commands.add_parser(name)
        command.add_argument("--root", default=argparse.SUPPRESS)
        if name == "search":
            command.add_argument("query", nargs="?", default="")
            for field in ("kind", "status", "workstream"):
                command.add_argument("--" + field)
    from scripts.project_doc_lifecycle import register_commands, run
    register_commands(commands)
    args = parser.parse_args(argv)
    try:
        root = Path(args.root).resolve(strict=True)
        if args.command in {"draft", "promote", "retire"}:
            print(serialize(run(root, args)), end="")
            return 0
        index = catalog(root)
        warnings = sum(row["status"] == "unclassified" for row in index["documents"])
        if args.command == "search":
            query = args.query.casefold()
            rows = [row for row in index["documents"] if all(not getattr(args, key) or row.get(key) == getattr(args, key)
                for key in ("kind", "status", "workstream")) and query in " ".join(str(row.get(key, ""))
                for key in ("title", "summary", "path", "text", "id", "workstream")).casefold()]
            result = {"status": "ok", "documents": [{k: v for k, v in row.items() if k != "text"} for row in rows]}
        else:
            outputs = {INDEX: serialize(index), PAGE: page(index)}
            destinations = {name: safe_path(root, name) for name in outputs}
            if args.command == "build":
                for name, value in outputs.items():
                    write_atomic(destinations[name], value)
            else:
                stale = [name for name, value in outputs.items() if not destinations[name].is_file()
                    or destinations[name].read_bytes() != value.encode("utf-8")]
                require(not stale, "Generated index is stale or missing: " + ", ".join(stale))
            result = {"status": "ok", "documents": len(index["documents"]), "unclassified_warning_count": warnings}
        print(serialize(result), end="")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(serialize({"status": "error", "message": str(error)}), end="")
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
