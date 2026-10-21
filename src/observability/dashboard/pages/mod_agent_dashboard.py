"""Controlled-agent dashboard page."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.mod_agent.artifacts import ArtifactStore, ArtifactStoreError
from src.mod_agent.dashboard import DashboardProjectionBuilder
from src.observability.dashboard.services.trace_service import TraceService

DEFAULT_PRODUCT_RUNS_ROOT = Path.home() / "ModAgentRuns"


def render() -> None:
    """Render read-only controlled-agent artifact summaries."""

    import streamlit as st

    st.header("Controlled Agent")
    selected = _render_run_records()
    if selected:
        _render_record_outcome(selected["inspection"])
    records_tab, trace_tab, artifacts_tab = st.tabs(["Run Records", "Trace Drill-down", "Artifact Projection"])
    with records_tab:
        if selected:
            _render_run_record_index(selected["service"], selected["roots"], selected["inspection"])
    with trace_tab:
        _render_product_trace_drilldown(selected=selected)
    with artifacts_tab:
        _render_artifact_projection(selected=selected)


def _record_issues(items: list[Any]) -> None:
    import streamlit as st

    for item in items:
        st.warning(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False))


def _record_jump(record_id: str, *, offset: int = 0, expected_sha256: str | None = None,
                 preserve_version: bool = False, reference: dict[str, Any] | None = None) -> None:
    import streamlit as st

    if not preserve_version or st.session_state.get("record_browser_focus") != record_id:
        st.session_state["record_browser_expected_sha256"] = expected_sha256
        st.session_state["record_browser_opened_reference"] = reference
    st.session_state["record_browser_focus"] = record_id
    st.session_state["record_browser_offset"] = offset
    st.session_state.pop("record_browser_link", None)


def _record_selection_changed(selection_key: str) -> None:
    import streamlit as st

    _record_jump(st.session_state[selection_key])


def _selection_key(prefix: str, *identity: Any) -> str:
    """Bind frontend selection state to the actual source/version, not its row number.

    为什么要有它：不同运行可能都有第0行，复用控件身份会让闭合标签停留在旧来源而后台已换成新记录。
    """
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return prefix + "::" + hashlib.sha256(raw).hexdigest()[:20]


def _record_roots_default() -> tuple[str, str | None]:
    """Use explicit process configuration, never a built-in experiment directory list.

    为什么要有它：同一浏览器可接入用户指定的不同运行目录，缺省值不能隐式授权整个磁盘。
    """
    configured = os.environ.get("REPOFORGE_RUN_RECORD_ROOTS")
    if configured is None:
        return str(DEFAULT_PRODUCT_RUNS_ROOT), None
    try:
        roots = json.loads(configured)
        if not isinstance(roots, list) or any(not isinstance(root, str) or not root.strip()
                or any(char in root for char in ("\n", "\r", "\x00")) for root in roots):
            raise ValueError("invalid roots")
        return "\n".join(roots), None
    except (ValueError, json.JSONDecodeError):
        return str(DEFAULT_PRODUCT_RUNS_ROOT), "REPOFORGE_RUN_RECORD_ROOTS 必须是 JSON 路径字符串数组；当前使用默认目录。"


def _render_run_records() -> dict[str, Any] | None:
    """Browse existing authorized records without a separate source of results.

    为什么要有它：运行选择和未知JSON读取复用同一只读服务，界面不能靠题目清单或自行遍历目录绕过边界。
    """
    import streamlit as st

    st.subheader("Run record browser / 运行记录")
    st.caption("只读浏览已有记录；不启动任务、模型或构建，不修改原件。历史状态与后续交付分别展示。")
    default_roots, configuration_issue = _record_roots_default()
    if configuration_issue:
        st.warning(configuration_issue)
    raw_roots = st.text_area("Allowed record roots / 允许读取的目录（每行一个）",
        value=default_roots, key="record_browser_roots")
    roots = tuple(line.strip() for line in raw_roots.splitlines() if line.strip())
    refresh = st.button("Discover / refresh runs", key="record_browser_discover", disabled=not roots)
    if refresh:
        st.session_state["record_browser_authorized_roots"] = roots
        st.session_state["record_browser_run_cursor"] = 0
        st.session_state.pop("record_browser_inspection", None)
    if not roots or st.session_state.get("record_browser_authorized_roots") != roots:
        st.info("选择明确的运行目录后点击 Discover；不会搜索整个磁盘。")
        return
    service = TraceService()
    try:
        discovered = service.discover_runs(allowed_roots=list(roots),
            cursor=st.session_state.get("record_browser_run_cursor", 0), limit=100)
        _record_issues(discovered.get("issues") or [])
        runs = discovered.get("runs") or []
        if not runs:
            st.info("没有发现可读取的运行。缺失记录不等于任务成功或失败。")
            return
        options = {row["run_id"]: row for row in runs}
        run_id = st.selectbox("Run / 运行", list(options), key="record_browser_run",
            format_func=lambda value: f"{options[value].get('format')} · {options[value].get('recorded_run_id') or Path(options[value]['run_root']).name} · {options[value].get('status', 'unknown')}")
        st.caption(f"当前运行目录：{options[run_id].get('run_root')}。下面三个视图共用此选择。")
        st.caption(f"运行来源：{options[run_id].get('identity_source', '未记录')}；定位符：{run_id}")
        next_cursor = discovered.get("next_cursor")
        if next_cursor is not None and st.button("Next runs", key="record_browser_next_runs"):
            st.session_state["record_browser_run_cursor"] = next_cursor
            st.session_state.pop("record_browser_inspection", None)
            st.rerun()
        if st.session_state.get("record_browser_run_cursor", 0) and st.button("First runs", key="record_browser_first_runs"):
            st.session_state["record_browser_run_cursor"] = 0
            st.session_state.pop("record_browser_inspection", None)
            st.rerun()
        st.caption("运行发现：完整" if discovered.get("complete") else "运行发现尚未完整；可继续下一页或刷新。")
        stored = st.session_state.get("record_browser_inspection")
        refresh_run = st.button("Refresh selected run", key="record_browser_refresh_run")
        if stored is None or stored.get("run", {}).get("run_id") != run_id or refresh_run:
            stored = service.inspect_run(allowed_roots=list(roots), run_id=run_id, record_cursor=0, record_limit=200)
            st.session_state["record_browser_inspection"] = stored
            st.session_state["record_browser_inspection_revision"] = st.session_state.get("record_browser_inspection_revision", 0) + 1
            st.session_state.pop("record_browser_selection", None)
            st.session_state.pop("record_browser_focus", None)
            st.session_state.pop("record_browser_expected_sha256", None)
            st.session_state.pop("record_browser_opened_reference", None)
            st.session_state.pop("record_browser_event", None)
            st.session_state.pop("record_browser_link", None)
            st.session_state["record_browser_offset"] = 0
        if stored.get("next_record_cursor") is not None and st.button("Load more records", key="record_browser_more_records"):
            following = service.inspect_run(allowed_roots=list(roots), run_id=run_id,
                record_cursor=stored["next_record_cursor"], record_limit=200)
            stored = {**following, "records": [*stored.get("records", []), *following.get("records", [])],
                "links": [*stored.get("links", []), *following.get("links", [])]}
            st.session_state["record_browser_inspection"] = stored
            st.session_state.pop("record_browser_event", None)
        return {"service": service, "roots": list(roots), "inspection": stored}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"运行记录无法读取：{exc}")
    return None


def _render_record_outcome(inspection: dict[str, Any]) -> None:
    """Show separate recorded stages once, not three competing task verdicts.

    为什么要有它：原失败不能被覆盖，但后续恢复也不能藏在JSON里；引用完整性不是独立业务验收。
    """
    import streamlit as st

    summary = inspection.get("summary") or {}
    scopes = summary.get("outcome_scopes") or {}
    original = scopes.get("original") or {}
    followups = scopes.get("followups") or []
    states = list(dict.fromkeys(str(row.get("reported_status") or row.get("status") or "unknown") for row in followups))
    st.subheader("Recorded outcome / 已记录结果")
    cols = st.columns(3)
    original_status = ("未记录终态" if scopes and scopes.get("original") is None else
        original.get("reported_status") or original.get("status") or summary.get("reported_status") or summary.get("status", "unknown"))
    cols[0].metric("原运行记录", original_status)
    if original_status == "未记录终态":
        st.caption("历史阶段状态：" + str(summary.get("reported_status") or summary.get("status", "unknown")) + "；不能据此判断进程现在仍在运行。")
    cols[1].metric("后续恢复／交付记录", "存在已交付记录" if "delivered" in states
                   else " / ".join(states) if states else "未找到已关联记录")
    integrity = scopes.get("evidence_status", "unknown")
    cols[2].metric("原件引用核验", {"complete": "完整", "partial": "部分", "unknown": "未知", "invalid": "存在无效引用"}.get(integrity, integrity))
    if "delivered" in states:
        st.info("存在后续 delivered 记录；原运行结果保留。是否满足全部交付条件，应回查该记录的验收范围。")
    if followups:
        with st.expander(f"后续处理阶段与来源（{len(followups)}）"):
            st.dataframe([{key: row.get(key) for key in ("path", "reported_status", "verification_scope", "integrity")}
                          for row in followups], use_container_width=True)
    st.caption("unknown/null 表示未记录；状态字段不证明进程仍在运行。导入的历史候选与检查不是本轮新成绩。")
    with st.expander("Outcome sources and cost details / 结果来源与成本详情"):
        st.json({key: value for key, value in summary.items() if key != "cloud_association"})


def _render_run_record_index(service: TraceService, roots: list[str], inspection: dict[str, Any]) -> None:
    import streamlit as st

    _record_issues(inspection.get("issues") or [])
    _render_cloud_association(service, roots, inspection)
    records = {row["record_id"]: row for row in inspection.get("records") or []}
    run_identity = (inspection["run"]["run_id"], st.session_state.get("record_browser_inspection_revision", 0))
    st.subheader("Execution timeline / 执行时间线")
    timeline = inspection.get("timeline") or []
    if timeline:
        st.caption("按记录中的带时区时间升序排列；顺序不代表因果关系。")
        st.dataframe([{key: row.get(key) for key in ("sort_time_utc", "name", "role", "status", "started_at", "finished_at")} for row in timeline], use_container_width=True)
        event_options = {_selection_key("event", item, index): (index, item) for index, item in enumerate(timeline)}
        event_id = st.selectbox("Timeline event", list(event_options), key=_selection_key("record_browser_event", run_identity),
            format_func=lambda value: f"{event_options[value][0] + 1}. {event_options[value][1].get('role') or event_options[value][1].get('name', 'record')} · {event_options[value][1].get('status', 'unknown')}")
        event = event_options[event_id][1]
        st.json(event)
        _render_record_links(event.get("refs") or [], key="record_browser_event_ref",
            label="Event reference / 事件对应记录", open_label="Open event reference", identity=(run_identity, event_id))
        if event.get("record_id"):
            st.button("Open event record", key="record_browser_event_open", on_click=_record_jump,
                args=(event["record_id"],), kwargs={"expected_sha256": records.get(event["record_id"], {}).get("sha256")})
    else:
        st.info("没有可核准的时间线；仍可查看下面的原始记录。")
    undated = inspection.get("undated_records") or []
    if undated:
        with st.expander(f"无可用执行时间的记录（{len(undated)}）"):
            st.caption("缺失、无时区或无效时间单独显示；未使用文件修改时间推断执行顺序。")
            st.dataframe(undated, use_container_width=True)
    if inspection.get("timeline_coverage"):
        with st.expander("时间线覆盖范围"):
            st.json(inspection["timeline_coverage"])
    st.subheader("All authorized records / 全部授权记录")
    st.caption(f"已列出 {len(records)} 条；索引{'完整' if inspection.get('index_complete') else '尚未完整，可继续加载'}。新增或未知格式使用同一读取入口。")
    if not records:
        st.info("没有已枚举的记录。")
        return
    selection_key = _selection_key("record_browser_selection", run_identity)
    focus = st.session_state.get("record_browser_focus")
    if focus and focus not in records:
        reference = st.session_state.get("record_browser_opened_reference") or {}
        records[focus] = {"relative_path": reference.get("target_path") or focus,
                          "status": "linked_record_not_in_current_index_page"}
    if focus and st.session_state.get(selection_key) != focus:
        st.session_state[selection_key] = focus
    selected = st.selectbox("Record / 记录", list(records), key=selection_key,
        format_func=lambda value: f"{records[value].get('relative_path')} · {records[value].get('size_bytes', '?')} bytes · {records[value].get('status', 'unknown')}",
        on_change=_record_selection_changed, args=(selection_key,))
    focus = st.session_state.setdefault("record_browser_focus", selected)
    offset = st.number_input("Text offset / 脱敏文本字符偏移", min_value=0, value=0,
        step=65536, key="record_browser_offset")
    expected_sha = st.session_state.get("record_browser_expected_sha256") or records.get(focus, {}).get("sha256")
    document = service.read_record(allowed_roots=roots, record_id=focus, offset=int(offset), limit=65536,
        expected_sha256=expected_sha)
    st.info("正在查看：" + str(document.get("record", {}).get("path") or focus))
    # Scope: bind every page to the first verified bytes, including records not pre-hashed in the index.
    # 为什么要有它：跨引用或大文件分页不能悄悄混合多个文件版本，刷新运行才重新选择当前字节。
    if document.get("status") in {"ok", "ready", "available", "corrupt"} and document.get("record", {}).get("sha256"):
        st.session_state["record_browser_expected_sha256"] = document["record"]["sha256"]
    opened_reference = st.session_state.get("record_browser_opened_reference")
    if opened_reference:
        st.caption("当前记录来自选中的显式引用；引用状态不证明工具已经执行。")
        st.json(opened_reference)
        if opened_reference.get("historical"):
            st.warning("正在查看历史引用的原记录，不计为本轮新执行或新验收。")
    _render_record_document(document, focus)
    links = document.get("links") or [row for row in inspection.get("links") or []
        if row.get("source_record_id") == focus]
    if links:
        st.subheader("Related records / 关联记录")
        _render_record_links(links, key="record_browser_link", label="Reference / 引用", open_label="Open referenced record",
            identity=(run_identity, focus, document.get("record", {}).get("sha256")))


def _render_cloud_association(service, roots, inspection):
    """Show shared identity and recorded export facts, not a second cloud trace UI.

    为什么要有它：本地交付关系与云端通用观测分工，但必须可核对同一Trace，不能各自宣称独立真相。
    """
    import streamlit as st

    association = inspection.get("summary", {}).get("cloud_association")
    if not isinstance(association, dict):
        return
    with st.expander("Langfuse association / 云端关联与导出状态"):
        st.caption("通用 Trace、模型调用和成本分析优先使用 Langfuse；本页负责工程原件、候选、验收、恢复与交付关联。")
        st.caption("原 Trace ID 不变。导出记录、HTTP 接收和云端回读是不同事实；云端缺失不会改写本地任务结果。")
        traces = {row["trace_id"]: row for row in association.get("traces", []) if row.get("trace_id")}
        if association.get("unassociated_export_reports"):
            st.warning("存在未记录 Trace ID 的导出报告；保留原件，不猜测它属于哪条 Trace。")
            st.json(association["unassociated_export_reports"])
        if association.get("source_issues"):
            _record_issues(association["source_issues"])
        if not traces:
            st.info("没有可核准的原 Trace 身份，不能生成云端成功关联。")
            return
        identity = (inspection["run"]["run_id"], association.get("source_fingerprint"))
        trace_id = st.selectbox("Original Trace ID / 原轨迹标识", list(traces),
            key=_selection_key("record_cloud_trace", identity))
        st.json(traces[trace_id])
        st.caption("核验只查询已配置的 Langfuse 项目和本 Trace 的记录时间窗，不上传本地内容、不重试导出、不调用模型。")
        result_key = _selection_key("record_cloud_result", identity, trace_id)
        if st.button("Verify cloud records (read only) / 只读核验云端", key=_selection_key("record_cloud_verify", identity, trace_id)):
            with st.spinner("核对原 Trace 身份与已记录观察；本地结果保持不变……"):
                st.session_state[result_key] = service.verify_run_cloud(allowed_roots=roots,
                    run_id=inspection["run"]["run_id"], trace_id=trace_id,
                    expected_source_fingerprint=association.get("source_fingerprint"))
        if result_key in st.session_state:
            result = st.session_state[result_key]
            st.json(result)
            if result.get("cloud_base_url"):
                st.link_button("Open configured Langfuse / 打开已配置的 Langfuse", result["cloud_base_url"])
                st.caption("这是服务入口，不是已验证的项目深链接；请使用上方原 Trace ID。未找到云记录时不生成成功链接。")
        else:
            st.info("本次浏览尚未核验云端；没有回读证据时不显示已同步。")


def _render_record_links(links: list[dict[str, Any]], *, key: str, label: str, open_label: str, identity: Any) -> None:
    import streamlit as st

    if not links:
        return
    options = {_selection_key("reference", item, index): item for index, item in enumerate(links)}
    selection_key = _selection_key(key, identity, list(options))
    selected = st.selectbox(label, list(options), key=selection_key,
        format_func=lambda value: f"{options[value].get('json_pointer', '?')} → {options[value].get('target_path', '?')} · {options[value].get('status', 'unknown')}")
    link = options[selected]
    st.json(link)
    if link.get("historical"):
        st.warning("这是历史记录引用；不能算成本轮新执行或新验收。")
    expected = link.get("expected_sha256") or link.get("actual_sha256")
    verified_hash = isinstance(expected, str) and len(expected) == 64 and all(char in "0123456789abcdefABCDEF" for char in expected)
    permitted = bool(link.get("target_record_id")) and link.get("status") in {"ok", "corrupt"} and verified_hash
    if not permitted:
        st.caption("引用缺失、受限、版本不符或未核准字节身份，不能作为已验证引用打开。")
    if link.get("status") == "corrupt":
        st.warning("引用字节可核对，但内容损坏；只能查看安全文本，不能视为有效结果。")
    st.button(open_label, key=_selection_key(key + "_open", identity, selected), disabled=not permitted, on_click=_record_jump,
        args=(link.get("target_record_id"),), kwargs={"expected_sha256": expected, "reference": link})


def _patch_comparisons(value: Any) -> list[dict[str, Any]]:
    """Find literal before/after pairs; never infer a source version from a filename.

    为什么要有它：字段差异有真实记录依据，但不证明修改已应用、构建通过或实际交付。
    """
    pending, pairs = [("", value)], []
    while pending:
        pointer, item = pending.pop()
        if isinstance(item, dict):
            if isinstance(item.get("before_text"), str) and isinstance(item.get("after_text"), str):
                pairs.append({"pointer": pointer or "/", "before": item["before_text"], "after": item["after_text"]})
            pending.extend((pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"), child)
                for key, child in reversed(list(item.items())))
        elif isinstance(item, list):
            pending.extend((pointer + "/" + str(index), item[index]) for index in reversed(range(len(item))))
    return pairs


def _render_record_document(document: dict[str, Any], record_id: str) -> None:
    import streamlit as st

    st.subheader("Record content / 记录内容")
    st.json({"source": document.get("record"), "status": document.get("status"), "format": document.get("format"),
        "range": document.get("range"), "redaction": document.get("redaction"), "read_only": document.get("read_only")})
    _record_issues(document.get("issues") or [])
    status = document.get("status")
    if status == "corrupt":
        st.warning("记录状态：corrupt。JSON 解析失败；仍可逐页查看服务提供的安全文本，不展示为有效 JSON 树。")
    elif status not in {"ok", "ready", "available"}:
        st.warning(f"记录状态：{status}。缺失、损坏、已变更或未授权内容不会伪装为完整原文。")
        return
    redaction = document.get("redaction") or {}
    if redaction.get("applied"):
        st.warning(f"敏感内容已脱敏（{redaction.get('count', '?')}处）；下面不是未经处理的原始字节。")
    span = document.get("range") or {}
    st.caption(f"当前显示字符范围：{span.get('offset', 0)}–{span.get('offset', 0) + span.get('length', 0)} / {span.get('total_chars', '?')}；原件字节大小和SHA见来源。")
    if status != "corrupt" and document.get("json_value") is not None:
        with st.expander("JSON tree / 完整脱敏 JSON 树", expanded=False):
            st.json(document["json_value"])
    st.code(document.get("text") or "", language="json" if document.get("format") == "json" else "text")
    st.download_button("Download displayed text page / 下载当前文本页", data=document.get("text") or "",
        file_name="record-page.txt", mime="text/plain", key="record_browser_download")
    st.caption("下载仅包含当前脱敏展示页。分页不会省略中间内容；下一页使用服务返回的字符偏移。")
    if span.get("next_offset") is not None:
        st.button("Next text page", key="record_browser_next_text", on_click=_record_jump,
            args=(record_id,), kwargs={"offset": span["next_offset"], "preserve_version": True})
    if span.get("offset", 0):
        st.button("First text page", key="record_browser_first_text", on_click=_record_jump,
            args=(record_id,), kwargs={"preserve_version": True})
    if span.get("complete"):
        st.caption("已到记录末尾。")
    for pair in _patch_comparisons(document.get("json_value") if status != "corrupt" else None):
        with st.expander("Recorded patch diff · " + pair["pointer"]):
            st.caption("仅比较本记录的 before_text / after_text 字段；不是应用、测试通过或交付证明。")
            diff = "".join(difflib.unified_diff(pair["before"].splitlines(keepends=True),
                pair["after"].splitlines(keepends=True), fromfile="recorded before_text", tofile="recorded after_text"))
            st.code(diff or "(no textual difference)", language="diff")


def _selected_artifact_store(inspection: dict[str, Any]) -> ArtifactStore | None:
    """Use only recognized local stores inside the already selected run.

    为什么要有它：自动关联不能把外部引用或目录链接变成新读取权限；工件状态也不等于整项任务的最新结果。
    """
    root = Path(inspection["run"]["run_root"])
    if not root.is_absolute() or not root.is_dir():
        return None
    if (root.is_symlink() or getattr(root.lstat(), "st_file_attributes", 0) & 0x400
            or root.resolve(strict=True) != root.absolute()):
        return None
    root = root.resolve(strict=True)
    for parts in (("a",), ("artifacts",), ("trial", "a")):
        candidate = root
        valid = True
        for part in parts:
            candidate = candidate / part
            if (not candidate.is_dir() or candidate.is_symlink()
                    or getattr(candidate.lstat(), "st_file_attributes", 0) & 0x400):
                valid = False
                break
        if valid and candidate.resolve(strict=True).is_relative_to(root):
            return ArtifactStore(candidate)
    return None


def _render_artifact_projection(*, selected=None) -> None:
    import streamlit as st

    if selected:
        inspection = selected["inspection"]
        st.caption("与上方共用运行选择。下面仅展示该运行本地 ArtifactStore 的状态；后续恢复／交付结果见上方记录，不覆盖原工件。")
        try:
            store = _selected_artifact_store(inspection)
            if store is None:
                st.info("本运行未发现可安全关联的 ArtifactStore；仍可在 Run Records 查看原件及显式引用。")
                return
            ids = store.list_feature_ids()
            if not ids:
                st.info("所选运行的工件库没有已登记任务。")
                return
            feature_id = st.selectbox("Feature in selected run / 当前运行的任务", ids,
                key=_selection_key("mod_agent_selected_feature", inspection["run"]["run_id"]))
            st.caption("工件来源：" + str(store.root))
            projection = DashboardProjectionBuilder(store).build_feature_projection(feature_id)
        except (ArtifactStoreError, OSError, ValueError) as exc:
            st.error(f"所选运行的工件无法读取：{exc}")
            return
        _render_projection(projection)
        return

    st.caption("独立工件检查：尚未选择运行，这里的状态不代表其他运行或后续交付。")
    feature_id = st.text_input(
        "Feature ID",
        value="",
        key="mod_agent_feature_id",
    ).strip()
    artifact_root = st.text_input(
        "Artifact Root",
        value="artifacts",
        key="mod_agent_artifact_root",
    ).strip()

    if not feature_id:
        st.info("Enter a feature_id to inspect controlled-agent artifacts.")
        return

    try:
        store = _artifact_store_from_input(artifact_root or "artifacts")
        projection = DashboardProjectionBuilder(store).build_feature_projection(feature_id)
    except ArtifactStoreError as exc:
        st.error(str(exc))
        return

    _render_projection(projection)


def _render_product_trace_drilldown(*, selected=None) -> None:
    """Render one explicit local Product run without starting work."""
    import streamlit as st

    st.subheader("Product Run Trace")
    st.caption(
        "Read-only local inspection. The selected run must stay under Allowed Runs Root; "
        "Cloud-supplied paths are never followed."
    )
    if selected:
        inspection = selected["inspection"]
        run = inspection["run"]
        run_root, run_id = run["run_root"], run.get("recorded_run_id")
        st.caption("当前运行：" + run_root)
        if not run_id or run.get("format") != "product":
            st.info("该运行不是带产品运行标识的 Trace 格式；同一运行的实际请求、事件和终态可在 Run Records 查看，不要求补造 manifest。")
            return
        current_path = Path(run_root).resolve()
        allowed = [Path(value).resolve() for value in selected["roots"]
                   if current_path.is_relative_to(Path(value).resolve())]
        if not allowed:
            st.error("所选运行不在当前允许目录内。")
            return
        allowed_root = str(max(allowed, key=lambda path: len(path.parts)))
        continuations = {}
        for item in (inspection.get("summary", {}).get("outcome_scopes") or {}).get("followups", []):
            relative = Path(item.get("path", ""))
            if relative.parts and relative.parts[0] == "continuations" and relative.name == "run_result.json":
                continuations[str(Path(run_root) / relative.parent)] = item
        selection = st.selectbox("Continuation / 后续恢复记录", ["", *continuations],
            key=_selection_key("product_trace_shared_continuation", run["run_id"]),
            format_func=lambda value: "仅原运行" if not value else f"{continuations[value].get('reported_status') or continuations[value].get('status', 'unknown')} · {Path(value).name}")
        continuation_root = selection or None
    else:
        allowed_root = st.text_input(
            "Allowed Runs Root", value=str(DEFAULT_PRODUCT_RUNS_ROOT), key="product_trace_allowed_root",
        ).strip()
        run_root = st.text_input("Run Root", value="", key="product_trace_run_root").strip()
        run_id = st.text_input("Run ID", value="", key="product_trace_run_id").strip()
        continuation_root = st.text_input(
            "Continuation Root (optional)", value="", key="product_trace_continuation_root",
        ).strip()
    show_raw = st.checkbox(
        "Show hash-verified local request/response bodies",
        value=False,
        key="product_trace_show_raw",
        help="Reads only request/response refs recorded inside the explicitly selected local run.",
    )
    source_identity = (allowed_root, run_root, run_id, continuation_root, show_raw)
    if not st.button("Inspect Product Trace", key="product_trace_inspect", disabled=not (run_root and run_id)):
        if st.session_state.get("product_trace_selected_source") == source_identity:
            _render_product_trace(st.session_state["product_trace_drilldown"], show_raw=show_raw)
        return
    try:
        drilldown = TraceService(max_string_chars=262_144).inspect_product_run(
            allowed_root=allowed_root,
            run_root=run_root,
            run_id=run_id,
            continuation_root=continuation_root or None,
            include_local_raw=show_raw,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Trace inspection failed: {exc}")
        return
    st.session_state["product_trace_drilldown"] = drilldown
    st.session_state["product_trace_selected_source"] = source_identity
    _render_product_trace(drilldown, show_raw=show_raw)


def _render_product_trace(drilldown: dict[str, Any], *, show_raw: bool) -> None:
    import streamlit as st

    protection = drilldown.get("redaction") or {}
    if protection.get("applied"):
        st.warning("此 Trace 展示中的敏感内容已脱敏；身份与哈希先按原件核验，显示内容不是完整原文。")
    accounting = drilldown["call_accounting"]
    cols = st.columns(4)
    cols[0].metric("Actual calls", accounting["actual_generation_records"])
    original = accounting["original"]
    latest = accounting["latest"]
    findings = accounting["latest_requirement_findings"]
    cols[1].metric("Original budget", f"{original['consumed']}/{original['maximum']}")
    cols[2].metric("Latest cumulative", f"{latest['consumed']}/{latest['maximum']}")
    cols[3].metric("Findings", f"{findings['consumed']}/{findings['maximum']}")
    st.caption(accounting["counting_rule"])

    st.subheader("Trace Sources")
    for segment in drilldown["segments"]:
        trace = segment["trace"]
        with st.expander(
            f"{segment['segment']} · trace {trace.get('trace_id')} · {segment['model_call_count']} cumulative calls",
            expanded=True,
        ):
            st.json({"run_root": segment["run_root"], "status": segment["status"],
                     "manifest_ref": segment["manifest_ref"], "result_ref": segment["result_ref"],
                     "trace": trace, "observation_count": segment["observation_count"],
                     **{key: segment.get(key) for key in ("record_format", "manifest_status", "trace_status",
                         "identity_status", "missing_identity_fields", "current_invocation_model_call_count")}})
            if segment.get("manifest_status") == "not_recorded":
                st.info("此恢复阶段未记录 run_manifest.json；按原结果和恢复引用核对，不补造 manifest 或新模型调用。")
            if segment.get("trace_status") == "not_recorded":
                st.info("此阶段未记录新的 Trace 观察；累计预算不是本次新增调用。")
            elif trace.get("mode") != "local_only" or trace.get("cloud_readback") != "not_performed":
                st.warning("Trace mode/readback differs from the expected local-only, not-performed state.")
            else:
                st.info("Local-only trace; Cloud readback was not performed. No Cloud link is available.")

    st.subheader("Model Calls")
    for call in drilldown["calls"]:
        label = (f"Call {call['call_number']} · {call['segment']} · {call['role']} · "
                 f"{call['model']} · {call['elapsed_ms']:.0f} ms")
        with st.expander(label, expanded=True):
            st.json({key: call[key] for key in (
                "request_span_id", "generation_span_id", "parent_id", "role", "model", "provider",
                "status", "started_at", "finished_at", "elapsed_ms", "transport", "usage",
                "input_payload_sha256", "output_sha256", "local_evidence", "counting")})
            if show_raw and call["raw"]:
                request = call["raw"].get("request")
                response = call["raw"].get("response")
                provider_request = call["raw"].get("provider_request")
                if request:
                    st.markdown("**Model messages**")
                    st.caption("Application-layer messages. The output constraint Schema is shown separately below.")
                    st.json(request.get("messages"))
                    st.markdown("**Output constraint Schema**")
                    st.caption(request.get("schema_visibility"))
                    st.json(request.get("response_format"))
                    with st.expander("Other application request fields"):
                        st.json(request.get("other_fields"))
                if response:
                    st.markdown("**Recorded application response**")
                    st.code(response.get("text", ""), language="json")
                if provider_request:
                    st.markdown("**Native provider request projection**")
                    st.caption("Only the hash-bound native format and non-content fields are shown; native messages and the complete body remain hidden.")
                    st.json(provider_request)

    st.subheader("Five-layer Schema Failure Chain")
    layers = drilldown["schema_failure_drilldown"]
    layer_labels = {
        "layer_1_schema_builder": "1. Schema builder constraints",
        "layer_2_application_to_native_provider": "2. Application response_format → native provider format",
        "layer_3_provider_runtime_support": "3. Ollama / llama.cpp version and support evidence",
        "layer_4_raw_response_against_recorded_schema": "4. Raw response vs recorded Schema",
        "layer_5_local_validators": "5. Local Schema and business validators",
    }
    for key, label in layer_labels.items():
        with st.expander(label, expanded=True):
            st.json(layers[key])
    st.caption(
        "Official documentation and fixed-version source analysis describe capability; "
        "they do not prove the native conversion used by this run."
    )

    latest_gate = drilldown["segments"][-1].get("post_findings_gate") or {}
    st.subheader("Current Post-Findings Gate")
    st.json({"blockers": drilldown["segments"][-1].get("blockers") or [], **latest_gate})

    reassessment = drilldown["segments"][-1].get("initial_findings_reassessment") or {}
    st.subheader("Initial Findings Query and Evidence Chain")
    st.caption(
        "Source retrieval, dependency-metadata requests, model HTTP transport, and Findings use are "
        "separate recorded boundaries. Input presence does not prove model attention or usefulness."
    )
    st.json(reassessment)

    st.subheader("Validation and Failure Timeline")
    for index, event in enumerate(drilldown["diagnostic_timeline"], 1):
        origin = event.get("origin")
        time_value = event.get("time") or "unknown (offline reconstruction has no original event time)"
        with st.expander(
            f"{index}. {event.get('segment')} · {event.get('error_code')} · {origin}", expanded=True,
        ):
            st.json({**event, "time": time_value})
            if origin == "offline_exact_input_replay":
                st.warning("Supplemental offline reconstruction; this was not backfilled as an original realtime span.")

    st.subheader("Recorded Blind Spots")
    for item in drilldown["blind_spots"]:
        st.write(f"- {item}")


def _artifact_store_from_input(artifact_root: str) -> ArtifactStore:
    root = Path(artifact_root)
    if root.is_absolute():
        resolved = root.resolve()
    else:
        resolved = (Path.cwd() / root).resolve()
        try:
            resolved.relative_to(Path.cwd().resolve())
        except ValueError as exc:
            raise ArtifactStoreError(
                "Relative Artifact Root cannot escape the project root."
            ) from exc
    return ArtifactStore(resolved)


def _render_projection(projection: dict[str, Any]) -> None:
    import streamlit as st

    st.subheader("Selected artifact-store state / 所选工件库状态")
    st.caption("这是工件投影，不是整个任务的最终交付结论；原失败与后续恢复记录分别保留。")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Status", projection.get("status", "unknown"))
    with cols[1]:
        st.metric("Artifacts", len(projection.get("artifacts") or []))
    with cols[2]:
        st.metric("Missing", len(projection.get("missing_artifacts") or []))
    with cols[3]:
        st.metric("Blockers", len(projection.get("blockers") or []))

    workflow, evidence, proposal, build, runtime, repair, evaluation = st.tabs(
        ["Workflow", "Evidence", "Proposal", "Build", "Runtime", "Repair", "Eval"]
    )
    with workflow:
        st.json(projection.get("workflow") or {})
    with evidence:
        st.json(projection.get("evidence") or {})
    with proposal:
        st.json(projection.get("proposal") or {})
    with build:
        st.json(projection.get("build") or {})
    with runtime:
        st.json(projection.get("runtime") or {})
    with repair:
        st.json(projection.get("repair") or {})
    with evaluation:
        st.json(projection.get("evaluation") or {})

    st.subheader("Artifacts")
    artifacts = projection.get("artifacts") or []
    if artifacts:
        st.dataframe(artifacts, use_container_width=True)
    else:
        st.info("No artifacts found for this feature_id.")

    missing = projection.get("missing_artifacts") or []
    if missing:
        st.subheader("Missing Artifacts")
        st.write(missing)

    blockers = projection.get("blockers") or []
    if blockers:
        st.subheader("Blockers")
        st.dataframe(blockers, use_container_width=True)
