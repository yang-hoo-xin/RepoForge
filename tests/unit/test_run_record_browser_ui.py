"""Read-only record-browser AppTest flows with an explicit service contract.

为什么要有它：界面必须按来源和完整分页展示事实，不能把历史状态、缺失数据或摘要当成新交付。
"""
from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def render_page():
    from src.observability.dashboard.pages.mod_agent_dashboard import render
    render()


class Reader:
    def __init__(self):
        self.calls = []
        self.runs = [{"run_id": "opaque-run-one", "run_root": "allowed/one", "format": "product",
            "status": "blocked", "identity_source": "recorded execution"}]
        self.records = [
            {"record_id": "opaque-request", "relative_path": "requests/new-format.json", "size_bytes": 100,
             "format": "json", "status": "ok", "sha256": "a" * 64},
            {"record_id": "opaque-large", "relative_path": "unknown-format.json", "size_bytes": 70004,
             "format": "json", "status": "ok", "sha256": "b" * 64},
            {"record_id": "opaque-candidate", "relative_path": "immutable-candidate.json", "size_bytes": 200,
             "format": "json", "status": "ok", "sha256": "c" * 64}]
        self.status = "ok"
        self.record_hashes = {row["record_id"]: row["sha256"] for row in self.records}
        self.extra_records = []
        self.event_refs = []
        self.reference_status = "ok"

    def discover_runs(self, **kwargs):
        self.calls.append(("discover", kwargs))
        return {"runs": deepcopy(self.runs), "issues": [], "next_cursor": None, "complete": True, "read_only": True}

    def inspect_run(self, **kwargs):
        self.calls.append(("inspect", kwargs))
        selected = next(row for row in self.runs if row["run_id"] == kwargs["run_id"])
        return {"run": deepcopy(selected), "summary": {"status": selected["status"], "completion": "not_recorded",
            "model_calls": None, "unknowns": ["process liveness not recorded"]},
            "timeline": [{"record_id": "opaque-request", "name": "model.request", "status": "running",
                "kind": "fact", "basis": "retained trace only", "started_at": None, "finished_at": None,
                "refs": deepcopy(self.event_refs)}],
            "records": deepcopy(self.records), "links": [], "issues": [], "next_record_cursor": None,
            "index_complete": True, "read_only": True}

    def read_record(self, **kwargs):
        self.calls.append(("read", kwargs))
        record = next(row for row in [*self.records, *self.extra_records] if row["record_id"] == kwargs["record_id"])
        record = {**record, "path": "allowed/one/" + record["relative_path"], "sha256": self.record_hashes[record["record_id"]]}
        value = {"new_schema": {"tool_results": [{"execution": "not_executed", "output": "[redacted]"}]}}
        text = '{"new_schema":{"tool_results":[{"execution":"not_executed","output":"[redacted]"}]}}'
        links = [{"source_record_id": record["record_id"], "json_pointer": "/candidate_ref",
            "target_record_id": "opaque-candidate", "target_path": "allowed/one/immutable-candidate.json",
            "relationship": "explicit_reference", "integrity": "verified", "status": self.reference_status,
            "historical": True, "actual_sha256": "c" * 64}]
        total, following, redacted = len(text), None, True
        if kwargs["record_id"] == "opaque-large":
            full = "BEGIN" + "x" * 70000 + "TAIL-MARKER"
            text = full[kwargs["offset"]:kwargs["offset"] + kwargs["limit"]]
            total = len(full)
            following = kwargs["offset"] + len(text) if kwargs["offset"] + len(text) < total else None
            value, links, redacted = None, [], False
        elif kwargs["record_id"] == "opaque-candidate":
            value = {"candidate_id": "historical-candidate", "changes": [{"patch": {
                "before_text": "old();\n", "after_text": "changed();\n"}}]}
            text, total, links, redacted = '{"candidate_id":"historical-candidate"}', 39, [], False
        result = {"record": record, "status": self.status, "format": "json", "json_value": value, "text": text,
            "range": {"offset": kwargs["offset"], "length": len(text), "total_chars": total,
                "next_offset": following, "complete": following is None},
            "redaction": {"applied": redacted, "count": int(redacted), "notes": ["sensitive values protected"]},
            "links": links, "issues": [], "read_only": True}
        if kwargs.get("expected_sha256") and kwargs["expected_sha256"] != record["sha256"]:
            result.update(status="hash_mismatch", text="", json_value=None)
        return result


@pytest.fixture
def browser(monkeypatch):
    from src.observability.dashboard.pages import mod_agent_dashboard as page

    reader = Reader()
    monkeypatch.delenv("REPOFORGE_RUN_RECORD_ROOTS", raising=False)
    monkeypatch.setattr(page, "TraceService", lambda **_kwargs: reader)
    return reader


def open_browser():
    app = AppTest.from_function(render_page).run()
    app.button(key="record_browser_discover").click().run()
    assert not app.exception
    return app


def text_content(app):
    return "\n".join(str(row.value) for name in ("json", "caption", "warning", "info", "code")
        for row in getattr(app, name))


def widget(app, kind, prefix):
    return next(item for item in getattr(app, kind) if item.key == prefix or item.key.startswith(prefix + "::"))


def test_cloud_verification_is_explicit_and_does_not_replace_task_outcome(browser):
    original = browser.inspect_run

    def inspect(**kwargs):
        result = original(**kwargs)
        result["summary"]["cloud_association"] = {"source_fingerprint": "d" * 64,
            "traces": [{"trace_id": "e" * 32, "mode": "local_only", "cloud_readback": "not_performed"}],
            "unassociated_export_reports": [{"mode": "local_only", "identity": "not_recorded"}]}
        return result

    def verify(**kwargs):
        browser.calls.append(("cloud", kwargs))
        return {"status": "not_found_at_check_time", "network_performed": True, "observations": 0}

    browser.inspect_run, browser.verify_run_cloud = inspect, verify
    app = open_browser()
    assert not any(kind == "cloud" for kind, _ in browser.calls)
    assert "Langfuse" in text_content(app) and "未记录 Trace ID" in text_content(app)
    widget(app, "button", "record_cloud_verify").click().run()
    assert not app.exception
    calls = [args for kind, args in browser.calls if kind == "cloud"]
    assert calls == [{"allowed_roots": [str(Path.home() / 'ModAgentRuns')],
        "run_id": "opaque-run-one", "trace_id": "e" * 32, "expected_source_fingerprint": "d" * 64}]
    assert "not_found_at_check_time" in text_content(app)
    assert app.session_state["record_browser_inspection"]["summary"]["status"] == "blocked"


def test_cross_run_selectors_get_new_source_bound_widget_identity(browser):
    browser.runs.append({"run_id": "opaque-run-two", "run_root": "allowed/two", "format": "experiment",
        "status": "blocked", "identity_source": "another execution"})
    browser.event_refs = [{"source_record_id": "opaque-request", "json_pointer": "/request",
        "target_record_id": "opaque-candidate", "target_path": "first-run/candidate.json",
        "status": "ok", "actual_sha256": "c" * 64}]
    app = open_browser()
    prefixes = ["record_browser_event", "record_browser_event_ref", "record_browser_selection", "record_browser_link"]
    first = {key: widget(app, "selectbox", key).key for key in prefixes}
    browser.event_refs[0]["target_path"] = "second-run/candidate.json"
    browser.records[0]["relative_path"] = "second-run/request.json"
    app.selectbox(key="record_browser_run").set_value("opaque-run-two").run()
    assert not app.exception
    second = {key: widget(app, "selectbox", key).key for key in prefixes}
    assert all(first[key] != second[key] for key in prefixes)
    assert any("second-run/candidate.json" in option for option in widget(app, "selectbox", "record_browser_event_ref").options)
    assert any("second-run/request.json" in option for option in widget(app, "selectbox", "record_browser_selection").options)


def test_browse_record_preserves_sources_unknown_state_and_redaction(browser):
    app = open_browser()
    rendered = text_content(app)
    assert "not_executed" in rendered and "[redacted]" in rendered
    assert "process liveness not recorded" in rendered
    assert "状态字段不证明进程仍在运行" in rendered and "不是本轮新成绩" in rendered
    assert "allowed/one/requests/new-format.json" in rendered
    assert "敏感内容已脱敏" in rendered
    assert browser.calls[-1][1]["record_id"] == "opaque-request"
    assert browser.calls[-1][1]["expected_sha256"] == "a" * 64
    assert set(name for name, _ in browser.calls) == {"discover", "inspect", "read"}


def test_unknown_large_json_can_be_viewed_through_final_page(browser):
    app = open_browser()
    widget(app, "selectbox", "record_browser_selection").set_value("opaque-large").run()
    assert not app.exception and "BEGIN" in text_content(app)
    assert "TAIL-MARKER" not in text_content(app)
    app.button(key="record_browser_next_text").click().run()
    assert not app.exception and "TAIL-MARKER" in text_content(app)
    assert browser.calls[-1][1]["offset"] == 65536
    assert "65536" in text_content(app) and "已到记录末尾" in text_content(app)
    app.button(key="record_browser_first_text").click().run()
    assert browser.calls[-1][1]["offset"] == 0


def test_real_reference_opens_historical_candidate_and_literal_diff(browser):
    app = open_browser()
    assert "历史记录引用" in text_content(app)
    widget(app, "button", "record_browser_link_open").click().run()
    assert not app.exception and browser.calls[-1][1]["record_id"] == "opaque-candidate"
    assert "+changed();" in text_content(app) and "-old();" in text_content(app)
    assert any("/changes/0/patch" in item.label for item in app.expander)
    assert "不是应用、测试通过或交付证明" in text_content(app)
    assert widget(app, "selectbox", "record_browser_selection").value == "opaque-candidate"
    assert "正在查看：allowed/one/immutable-candidate.json" in "\n".join(row.value for row in app.info)


def test_original_and_followup_outcomes_are_visible_without_opening_json(browser):
    original = browser.inspect_run

    def inspect(**kwargs):
        view = original(**kwargs)
        view["summary"]["outcome_scopes"] = {"original": {"reported_status": "blocked"},
            "followups": [{"reported_status": "delivered", "path": "delivery/result.json"}],
            "evidence_status": "partial"}
        return view

    browser.inspect_run = inspect
    app = open_browser()
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["原运行记录"] == "blocked"
    assert metrics["后续恢复／交付记录"] == "存在已交付记录"
    assert metrics["原件引用核验"] == "部分"
    assert any("存在后续 delivered 记录" in row.value for row in app.info)
    assert not any(row.value == "Run Summary" for row in app.subheader)


def test_trace_selection_uses_the_same_run_and_does_not_keep_previous_result(browser, monkeypatch):
    from src.observability.dashboard.pages import mod_agent_dashboard as page

    allowed = Path.home() / "ModAgentRuns"
    browser.runs = [{"run_id": "one", "run_root": str(allowed / "one"), "recorded_run_id": "one",
        "format": "product", "status": "blocked", "identity_source": "run_manifest.json"},
        {"run_id": "two", "run_root": str(allowed / "two"), "recorded_run_id": "two",
         "format": "product", "status": "awaiting_approval", "identity_source": "run_manifest.json"}]

    def inspect_product(**kwargs):
        browser.calls.append(("product", kwargs))
        return {"selected_source": kwargs["run_id"]}

    def render_trace(value, **kwargs):
        import streamlit as st
        st.info("trace-selected-" + value["selected_source"])

    browser.inspect_product_run = inspect_product
    monkeypatch.setattr(page, "_render_product_trace", render_trace)
    app = open_browser()
    assert not any(item.key == "product_trace_run_root" for item in app.text_input)
    app.button(key="product_trace_inspect").click().run()
    assert browser.calls[-1][0] == "product"
    assert browser.calls[-1][1]["run_root"] == str(allowed / "one")
    assert any(row.value == "trace-selected-one" for row in app.info)
    app.selectbox(key="record_browser_run").set_value("two").run()
    assert not any(row.value == "trace-selected-one" for row in app.info)
    app.button(key="product_trace_inspect").click().run()
    assert browser.calls[-1][1]["run_id"] == "two"
    assert any(row.value == "trace-selected-two" for row in app.info)


@pytest.mark.parametrize("status", ["missing", "forbidden", "hash_mismatch", "changed", "read_error"])
def test_bad_record_status_never_renders_content_as_complete(browser, status):
    browser.status = status
    app = open_browser()
    assert not app.exception
    assert any(f"记录状态：{status}" in row.value for row in app.warning)
    assert not app.code


def test_new_run_format_is_discovered_without_page_configuration(browser):
    app = open_browser()
    browser.runs.append({"run_id": "opaque-run-two", "run_root": "allowed/two", "format": "new_future_format",
        "status": "unknown", "identity_source": "new record set"})
    app.button(key="record_browser_discover").click().run()
    app.selectbox(key="record_browser_run").set_value("opaque-run-two").run()
    assert not app.exception
    assert any(kwargs.get("run_id") == "opaque-run-two" for name, kwargs in browser.calls if name == "inspect")
    assert "new record set" in text_content(app)


def test_corrupt_json_warns_but_keeps_safe_text_pagination(browser):
    browser.status = "corrupt"
    app = open_browser()
    widget(app, "selectbox", "record_browser_selection").set_value("opaque-large").run()
    assert any("JSON 解析失败" in row.value for row in app.warning)
    assert "BEGIN" in text_content(app)
    assert not any("JSON tree" in row.label for row in app.expander)
    app.button(key="record_browser_next_text").click().run()
    assert not app.exception and "TAIL-MARKER" in text_content(app)


def test_reference_outside_index_keeps_verified_hash_and_rejects_changed_target(browser):
    browser.extra_records.append(browser.records.pop())
    app = open_browser()
    browser.record_hashes["opaque-candidate"] = "d" * 64
    widget(app, "button", "record_browser_link_open").click().run()
    assert not app.exception
    assert browser.calls[-1][1]["record_id"] == "opaque-candidate"
    assert browser.calls[-1][1]["expected_sha256"] == "c" * 64
    assert any("hash_mismatch" in row.value for row in app.warning)
    assert not app.code


def test_unhashed_large_record_pins_first_read_version_for_following_pages(browser):
    browser.records[1].pop("sha256")
    app = open_browser()
    widget(app, "selectbox", "record_browser_selection").set_value("opaque-large").run()
    assert browser.calls[-1][1]["expected_sha256"] is None
    browser.record_hashes["opaque-large"] = "d" * 64
    app.button(key="record_browser_next_text").click().run()
    assert browser.calls[-1][1]["expected_sha256"] == "b" * 64
    assert any("hash_mismatch" in row.value for row in app.warning)
    assert not app.code


def test_timeline_reference_opens_actual_target_with_bound_version(browser):
    browser.event_refs = [{"json_pointer": "/request", "target_path": "allowed/one/immutable-candidate.json",
        "target_record_id": "opaque-candidate", "status": "ok", "expected_sha256": "c" * 64,
        "actual_sha256": "c" * 64, "historical": False, "execution": "not_executed"}]
    app = open_browser()
    widget(app, "button", "record_browser_event_ref_open").click().run()
    assert not app.exception and browser.calls[-1][1]["record_id"] == "opaque-candidate"
    assert browser.calls[-1][1]["expected_sha256"] == "c" * 64
    assert "+changed();" in text_content(app) and "not_executed" in text_content(app)


@pytest.mark.parametrize("status", ["changed", "read_error", "forbidden", "hash_mismatch", "unknown"])
def test_unverified_reference_cannot_open_as_a_verified_target(browser, status):
    browser.reference_status = status
    app = open_browser()
    assert widget(app, "button", "record_browser_link_open").disabled


def test_changing_allowed_roots_requires_new_explicit_discovery(browser):
    app = open_browser()
    before = len(browser.calls)
    app.text_area(key="record_browser_roots").set_value("another/allowed/root").run()
    assert len(browser.calls) == before and not app.exception
    assert any("点击 Discover" in row.value for row in app.info)


def test_patch_diff_uses_literal_fields_not_names_or_inferred_versions():
    from src.observability.dashboard.pages.mod_agent_dashboard import _patch_comparisons

    assert _patch_comparisons({"candidate_before": "x", "candidate_after": "y"}) == []
    assert _patch_comparisons({"a/b": {"~patch": {"before_text": "x", "after_text": "y"}}}) == [
        {"pointer": "/a~1b/~0patch", "before": "x", "after": "y"}]


def test_process_configuration_supplies_multiple_explicit_roots(browser, monkeypatch):
    monkeypatch.setenv("REPOFORGE_RUN_RECORD_ROOTS", '["allowed/one", "allowed/two"]')
    app = open_browser()
    assert app.text_area(key="record_browser_roots").value == "allowed/one\nallowed/two"
    assert all(kwargs["allowed_roots"] == ["allowed/one", "allowed/two"] for _, kwargs in browser.calls)


@pytest.mark.parametrize("configured", ['"not-an-array"', '[1]', '{bad json', '["one\\ntwo"]'])
def test_invalid_process_root_configuration_never_becomes_an_implicit_scope(monkeypatch, configured):
    from src.observability.dashboard.pages.mod_agent_dashboard import (
        DEFAULT_PRODUCT_RUNS_ROOT,
        _record_roots_default,
    )

    monkeypatch.setenv("REPOFORGE_RUN_RECORD_ROOTS", configured)
    value, warning = _record_roots_default()
    assert value == str(DEFAULT_PRODUCT_RUNS_ROOT) and "JSON" in warning
