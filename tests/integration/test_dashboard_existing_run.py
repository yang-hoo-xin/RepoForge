"""Opt-in read-only acceptance through the real Dashboard, not an experiment adapter.

为什么要有它：同一正常入口需能读取另一个既有任务；样本位置仅是测试输入，不能成为产品分支。
"""
import hashlib
import json
import os
from pathlib import Path
import socket

import pytest
from streamlit.testing.v1 import AppTest

from src.observability.dashboard.services.trace_service import TraceService


def render_page():
    from src.observability.dashboard.pages.mod_agent_dashboard import render
    render()


def _hashes(root):
    result = {}
    for directory, names, leaves in os.walk(root, followlinks=False):
        parent = Path(directory)
        names[:] = [name for name in names if not (parent / name).is_symlink()
                    and not getattr((parent / name).lstat(), "st_file_attributes", 0) & 0x400]
        for name in leaves:
            path = parent / name
            if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
                continue
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                result[str(path.relative_to(root))] = digest.hexdigest()
    return result


def _widget(app, kind, prefix):
    return next(item for item in getattr(app, kind) if item.key == prefix or item.key.startswith(prefix + "::"))


def test_existing_run_uses_public_dashboard_without_network(monkeypatch, record_property):
    configured = os.environ.get("REPOFORGE_ACCEPTANCE_RUN_ROOT")
    if not configured:
        pytest.skip("Explicit existing-run root required; never scan the user's disk")
    root = Path(configured).resolve(strict=True)
    before = _hashes(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    monkeypatch.setenv("REPOFORGE_RUN_RECORD_ROOTS", json.dumps([str(root)]))
    monkeypatch.setattr(socket.socket, "connect", lambda *_a, **_k: pytest.fail("Browser must not contact a model or cloud automatically"))
    service = TraceService()
    run = next(row for row in service.discover_runs(allowed_roots=[str(root)])["runs"]
               if Path(row["run_root"]).resolve() == root)
    app = AppTest.from_function(render_page, default_timeout=45).run()
    app.button(key="record_browser_discover").click().run()
    app.selectbox(key="record_browser_run").set_value(run["run_id"]).run()
    assert not app.exception
    view = app.session_state["record_browser_inspection"]
    assert view["run"]["recorded_run_id"] == manifest["run_id"]
    cloud = view["summary"]["cloud_association"]
    assert any(row["trace_id"] == manifest["execution_trace"]["trace_id"] for row in cloud["traces"])
    assert cloud["network_performed"] is False
    event_index, link_index, link = next((i, j, link)
        for i, event in enumerate(view["timeline"])
        for j, link in enumerate(event.get("refs") or [])
        if "request" in link.get("json_pointer", "") and link.get("status") == "ok")
    _widget(app, "selectbox", "record_browser_event").select_index(event_index).run()
    _widget(app, "selectbox", "record_browser_event_ref").select_index(link_index).run()
    _widget(app, "button", "record_browser_event_ref_open").click().run()
    assert not app.exception
    assert app.session_state["record_browser_focus"] == link["target_record_id"]
    assert app.session_state["record_browser_expected_sha256"] == (link.get("expected_sha256") or link["actual_sha256"])
    assert any(isinstance(value, dict) and value.get("source", {}).get("path") == link["target_path"]
               for value in (json.loads(item.value) for item in app.json))
    assert before == _hashes(root)
    record_property("normal_entry", "mod_agent_dashboard.render -> TraceService -> RunRecordBrowser")
    record_property("run_id", manifest["run_id"])
    record_property("trace_id", manifest["execution_trace"]["trace_id"])
    record_property("original_files_unchanged", len(before))
    record_property("actual_request_ref", link["target_path"])
    record_property("actual_request_sha256", app.session_state["record_browser_expected_sha256"])
