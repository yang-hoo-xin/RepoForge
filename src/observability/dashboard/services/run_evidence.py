"""Read-only joins over retained run evidence, shared by CLI and Dashboard.

为什么要有它：精确引用与实际请求决定关联，目录邻近或时间顺序不证明采纳或成功。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.mod_agent.evaluation.context_chain import _json_equal, _payloads

from .run_records import RunRecordBrowser, _AccessError, _sanitize

MAX_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 1500
MAX_ENTRIES = 20000
MAX_TOTAL_BYTES = 64 * 1024 * 1024
_V2 = re.compile(r"v2/[a-z][a-z0-9_-]*/([0-9a-f]{64})\.json")


def _preview(value, limit=2000):
    text = json.dumps(value, ensure_ascii=False)
    return value if len(text) <= limit else {"preview": text[:limit], "truncated": True, "total_chars": len(text)}


class EvidenceReader:
    """Reuse browser path/byte checks; references never expand selected roots."""

    def __init__(self, root):
        self.root = root
        self.browser = RunRecordBrowser([root])
        self.records, self.values, self.links, self.issues = {}, {}, [], []
        self.expanded = set()
        self.total_bytes = 0

    def read(self, path, expected=None):
        path = Path(path)
        key = str(path.absolute())
        cache = (key, expected)
        if expected is None:
            bound = [(value, self.records[k]) for k, value in self.values.items() if k[0] == key and k[1] is not None]
            if bound:
                failed = next(((v, r) for v, r in bound if r["status"] != "verified"), None)
                return failed or bound[0]
        if cache in self.values:
            return self.values[cache], self.records[cache]
        if len(self.records) >= MAX_RECORDS:
            issue = {"status": "truncated", "reason": "Evidence reference count limit reached"}
            if issue not in self.issues:
                self.issues.append(issue)
            return None, issue
        row = {"path": key, "expected_sha256": expected, "status": "unknown"}
        self.records[cache] = row
        value = None
        try:
            if len(self.records) > MAX_RECORDS:
                raise _AccessError("truncated", "Evidence reference count limit reached")
            locator = self.browser._locator(path)
            if locator is None:
                row["path"] = "[outside_selected_run]"
                raise _AccessError("forbidden", "Reference does not grant access outside selected run")
            _, _, selected = self.browser._decode(locator)
            row["record_id"] = locator
            row["size_bytes"] = selected.stat().st_size
            if row["size_bytes"] > MAX_BYTES:
                raise _AccessError("truncated", "File exceeds bounded inspection; original not modified")
            if self.total_bytes + row["size_bytes"] > MAX_TOTAL_BYTES:
                raise _AccessError("truncated", "Cumulative evidence read limit reached")
            _, raw, digest = self.browser._read(locator, expected)
            self.total_bytes += len(raw)
            row.update(sha256=digest, status="verified" if expected else "indexed_unbound")
            if selected.suffix == ".json":
                value = json.loads(raw)
                if not isinstance(value, dict):
                    value = None
            row["binding"] = "expected_byte_hash" if expected else "current_bytes_only_not_authenticity"
        except (OSError, ValueError, UnicodeError) as error:
            row.update(status=getattr(error, "status", "unreadable"), reason=str(error))
            self.issues.append(dict(row))
        self.values[cache] = value
        return value, row

    def reference(self, ref, origin, store=None):
        if isinstance(ref, str) and _V2.fullmatch(ref) and store is not None:
            path, expected = store / ref, _V2.fullmatch(ref).group(1)
        elif isinstance(ref, dict) and isinstance(ref.get("path"), str) and isinstance(ref.get("sha256"), str):
            path = Path(ref["path"])
            path = path if path.is_absolute() else origin.parent / path
            expected = ref["sha256"]
        else:
            issue = {"status": "missing", "reason": "Expected bound reference absent", "source": str(origin)}
            if issue not in self.issues:
                self.issues.append(issue)
            return None, issue
        return self.read(path, expected)

    def expand(self, value, origin, store=None):
        store = store or _store_for(origin, self.root)
        identity = (str(origin), str(store) if store else None)
        if not isinstance(value, dict) or identity in self.expanded:
            return
        self.expanded.add(identity)
        pending = [(value, "")]
        children = []
        while pending:
            if len(self.links) >= MAX_RECORDS * 8:
                issue = {"status": "truncated", "reason": "Reference edge limit reached"}
                if issue not in self.issues:
                    self.issues.append(issue)
                break
            item, pointer = pending.pop()
            if isinstance(item, dict) and isinstance(item.get("path"), str) and "artifact_hash" in item:
                # 为什么要有它：产品ArtifactStore语义哈希不是字节哈希，沿用浏览器原校验而不混用两种身份。
                target = Path(item["path"])
                target = target if target.is_absolute() else origin.parent / target
                bounded, record = self.read(target)
                if bounded is None:
                    self.links.append({"source": str(origin), "pointer": pointer, "target": record.get("path"),
                        "status": record["status"], "relation": "artifact_semantic_reference", "integrity": "unconfirmed"})
                    continue
                relations = self.browser._references(item, source_id=self.browser._locator(origin), source_path=origin, run_root=self.root)
                for relation in relations:
                    self.links.append({"source": str(origin), "pointer": pointer, "target": relation["target_path"],
                        "status": relation["status"], "relation": "artifact_semantic_reference", "integrity": relation["integrity"]})
                    if relation["integrity"] == "verified_artifact_semantics":
                        child, row = self.read(Path(relation["target_path"]), relation["actual_sha256"])
                        if child:
                            children.append((child, Path(row["path"]), store))
                    else:
                        self.issues.append({"status": relation["status"], "reason": "Product artifact reference not verified", "source": str(origin)})
                continue
            is_ref = isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str)
            if is_ref and not Path(item["path"]).is_absolute() and not any(
                    part.endswith(("_ref", "_refs", "_record")) for part in pointer.split("/")):
                # 为什么要有它：源码清单中的相对文件身份不是相对报告目录的可读引用，不能凭猜测拼接路径。
                self.links.append({"source": str(origin), "pointer": pointer, "target": item["path"],
                    "status": "identity_only", "relation": "relative_source_identity_not_resolved_as_record"})
                continue
            is_v2 = isinstance(item, str) and _V2.fullmatch(item) and store is not None
            if is_ref or is_v2:
                child, row = self.reference(item, origin, store)
                self.links.append({"source": str(origin), "pointer": pointer, "target": row.get("path"),
                    "status": row["status"], "relation": "explicit_reference_not_causal_claim"})
                if child and row["status"] == "verified":
                    children.append((child, Path(row["path"]), store))
                continue
            if isinstance(item, dict):
                pending.extend((v, pointer + "/" + str(k).replace("~", "~0").replace("/", "~1")) for k, v in item.items())
            elif isinstance(item, list):
                pending.extend((v, pointer + "/" + str(i)) for i, v in enumerate(item))
        for child, path, child_store in children:
            if len(self.records) <= MAX_RECORDS:
                self.expand(child, path, child_store)


def _resolve(browser, run_id):
    matches, issues, visited = {}, list(browser.issues), 0
    pending = list(browser.roots.values())
    while pending:
        root = pending.pop()
        visited += 1
        if visited > MAX_ENTRIES:
            issues.append({"status": "truncated", "reason": "Run discovery limit reached; uniqueness unconfirmed"})
            break
        try:
            locator = browser._locator(root)
            browser._decode(locator)
            signature = browser._signature(root)
            if signature:
                if signature.get("recorded_run_id") == run_id or root.name == run_id:
                    matches[str(root)] = signature
                # 为什么要有它：子候选、恢复和嵌套原件不是另一业务运行，不能按最新文件重新选主任务。
                continue
            pending.extend(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        except (OSError, ValueError) as error:
            issues.append({"status": getattr(error, "status", "unreadable"), "reason": str(error)})
    issues.extend(row for row in browser.issues if row not in issues)
    return list(matches.values()), issues


def _store_for(path, root):
    for parent in path.parents:
        if not parent.is_relative_to(root):
            break
        if (parent / "v2").is_dir():
            return parent
    return None


def _tool_outputs(request):
    output = list(request.get("tool_results") or [])
    for exchange in request.get("historical_tool_exchanges") or []:
        output.extend(exchange.get("tool_results") or [])
    return output


def _provider_contains(provider, name, output):
    for message in provider.get("messages") or []:
        if message.get("role") != "tool" or message.get("tool_name", message.get("name")) != name:
            continue
        try:
            value = json.loads(message["content"])
        except (ValueError, TypeError, KeyError):
            continue
        if _json_equal(value, output):
            return True
    return False


def _native_chain(reader, native, path):
    store = _store_for(path, reader.root)
    reader.expand(native, path, store)
    loop, _ = reader.reference(native.get("loop_result_ref"), path, store)
    calls = (loop or {}).get("call_records", {})
    turns, actions, feedback = [], [], []
    response_turns = {}
    for event in native.get("provider_turns", []):
        request, request_ref = reader.reference(event.get("request_ref"), path, store)
        provider, provider_ref = reader.reference(event.get("provider_request_ref"), path, store)
        response, response_ref = reader.reference(event.get("response_ref"), path, store)
        dispatch, dispatch_ref = reader.reference(event.get("dispatch_ref"), path, store)
        row = {"turn": event.get("turn"), "reservation_id": event.get("reservation_id"),
            "request_ref": request_ref, "provider_request_ref": provider_ref, "response_ref": response_ref,
            "request_prepared": bool(request), "dispatch_intent_recorded": bool(dispatch),
            "send_attempt_observed": True if provider and provider_ref.get("status") == "verified"
                and event.get("provider_request_capture_boundary") == "loopback_httpx_request_event" else None,
            "send_attempt_basis": event.get("provider_request_capture_boundary"),
            "response_observed": bool(response), "usage": (response or {}).get("usage"),
            "requested_actions": [{"call_id": c.get("call_id"), "name": c.get("name"), "arguments": _preview(c.get("arguments"))}
                for c in (response or {}).get("tool_calls", [])],
            "payload_types": [p.get("message_type") for _, p in _payloads(request)],
            "understanding_or_adoption": "not_inferred"}
        if response:
            response_turns.setdefault(response_ref.get("sha256"), []).append(row)
        turns.append(row)
        for observed in _tool_outputs(request or {}):
            name, call_id, output = observed.get("name"), observed.get("call_id"), observed.get("output")
            if name not in {"candidate_check", "check_feedback_read"}:
                continue
            result = calls.get(call_id, {}).get("result")
            equal = isinstance(result, dict) and _json_equal(result, observed)
            if not equal:
                reader.issues.append({"status": "unconfirmed", "reason": "Feedback in request lacks matching result in selected execution",
                    "call_id": call_id, "turn": event.get("turn")})
            value = output.get("value", {}) if isinstance(output, dict) else {}
            item = {"turn": event.get("turn"), "call_id": call_id, "name": name,
                **{key: value.get(key) for key in ("candidate_ref", "candidate_source_id", "check_ref", "feedback_ref", "status")},
                "request_ref": request_ref, "matches_recorded_tool_result": equal,
                "included_in_prepared_request": True,
                "included_in_provider_request": _provider_contains(provider or {}, name, output) if provider else None,
                "provider_request_ref": provider_ref, "projection": _preview(output),
                "related_requested_actions": [{"call_id": c.get("call_id"), "name": c.get("name")}
                    for c in row["requested_actions"]], "adoption_or_repair_proven": False}
            feedback.append(item)
    for call in native.get("completed_tool_records", []):
        result, result_ref = reader.reference(call.get("result_ref"), path, store)
        receipt, receipt_ref = reader.reference(call.get("execution_receipt_ref"), path, store)
        response, response_ref = reader.reference(call.get("response_ref"), path, store)
        arguments, arguments_ref = reader.reference(call.get("arguments_ref"), path, store)
        matches = [c for c in (response or {}).get("tool_calls", []) if c.get("call_id") == call.get("call_id")
            and c.get("name") == call.get("name") and _json_equal(c.get("arguments"), call.get("arguments"))]
        options = response_turns.get(response_ref.get("sha256"), [])
        exact = [row for row in options if row["response_ref"].get("path") == response_ref.get("path")]
        options = exact or options
        turn = options[0] if len(options) == 1 else None
        if len(options) > 1:
            reader.issues.append({"status": "ambiguous", "reason": "Identical response copies do not uniquely identify a turn",
                "call_id": call.get("call_id")})
        loop_call = calls.get(call.get("call_id"), {})
        output_equal = bool(result) and _json_equal(result, loop_call.get("result", {}).get("output"))
        receipt_equal = bool(receipt) and _json_equal(receipt.get("output"), (result or {}).get("value"))
        origin = (receipt or {}).get("origin", {})
        origin_equal = all(_json_equal(origin.get(k), call.get(k)) for k in ("call_id", "name", "arguments", "response_ref"))
        verified = (len(matches) == 1 and arguments is not None and _json_equal(arguments, call.get("arguments"))
            and output_equal and receipt_equal and origin_equal and loop_call.get("status") == "completed")
        actions.append({"call_id": call.get("call_id"), "name": call.get("name"),
            "turn": turn["turn"] if turn else None, "status": "executed_bound" if verified else "unconfirmed",
            "action_kind": "plan_update_not_code_edit" if call.get("name") == "plan_update" else "tool_action",
            "arguments": _preview(call.get("arguments")), "arguments_ref": arguments_ref,
            "result": _preview(result), "result_ref": result_ref, "receipt_ref": receipt_ref,
            "response_ref": response_ref, "match_basis": "call_id_name_arguments_response_hash_and_execution_receipt"})
        if not verified:
            reader.issues.append({"status": "unconfirmed", "reason": "Tool origin/result binding incomplete", "call_id": call.get("call_id")})
    return {"native_ref": reader.read(path)[1], "status": native.get("status"),
        "task_contract_ref": native.get("task_contract_ref"), "candidate_ref": native.get("candidate_ref"),
        "source_identity": _preview(native.get("source_before")), "model_identity": _preview(native.get("expected_model_identity")),
        "budget_after": native.get("budget_after"), "turns": turns, "actions": actions, "feedback_links": feedback,
        "uncompleted_actions": [{"call_id": key, "name": row.get("name"), "status": row.get("status"),
            "claim": "not_a_confirmed_execution"} for key, row in calls.items() if row.get("status") != "completed"]}


def query_run_evidence(*, allowed_roots, run_id):
    """Developer/CI read-only view. Not a product-model tool or authorization."""
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
        raise ValueError("EVIDENCE_RUN_ID_INVALID")
    browser = RunRecordBrowser(allowed_roots)
    matches, issues = _resolve(browser, run_id)
    report = {"schema_version": "run_evidence_v1", "run_id": run_id, "view": "developer_evidence",
        "read_only": True, "query_effects": {"model_calls": 0, "builds": 0, "tests": 0, "recoveries": 0, "writes": 0},
        "issues": issues, "complete": False, "executions": [], "records": [], "links": [],
        "completeness_scope": "bounded_index_and_reference_checks_not_task_acceptance",
        "delivery_qualification": "not_evaluated_use_original_acceptance_and_authorization",
        "limits": ["Associations do not prove understanding, adoption, correctness or delivery.",
            "No timestamp-only causal links. Developer view is not supplied to product models.",
            "Integrity is hash/reference consistency, not external authenticity or independent business grading."]}
    if len(matches) != 1 or issues:
        report.update(status="ambiguous" if len(matches) > 1 else "incomplete_index" if issues else "not_found",
            matches=matches)
        return _sanitize(report, [])
    root = Path(matches[0]["run_root"])
    reader = EvidenceReader(root)
    manifest, manifest_ref = reader.read(root / "run_manifest.json")
    terminal = (manifest or {}).get("terminal_record")
    result, result_ref = reader.reference(terminal, root / "run_manifest.json") if terminal else reader.read(root / "run_result.json")
    if any(v.get("run_id") != run_id for v in (manifest, result) if v is not None):
        report.update(status="identity_conflict", issues=[{"status": "identity_conflict", "reason": "Manifest/result business identity mismatch"}])
        return _sanitize(report, [])
    for value, ref in ((manifest, manifest_ref), (result, result_ref)):
        if value:
            reader.expand(value, Path(ref["path"]), root / "a" if (root / "a" / "v2").is_dir() else None)
    if not terminal:
        reader.issues.append({"status": "missing", "reason": "No manifest-bound terminal record; terminal is indexed only"})
    selected = []
    for i, (path, directory, issue) in enumerate(reader.browser._walk(root)):
        if i >= MAX_ENTRIES:
            reader.issues.append({"status": "truncated", "reason": "Run index scan limit reached"})
            break
        if issue:
            reader.issues.append({**issue, "path": str(path)})
        if directory or issue:
            continue
        if (path.name in {"execution.json", "integrity-error.json"} or path.name.endswith("-fact.json")
                or path.parent.name in {"session-checkpoints", "delivery-receipts"}
                or path.name.startswith("provider-outcome-")):
            selected.append(path)
    executions, facts, lineage = [], [], []
    for key in ("resumed_from_checkpoint_ref", "resumed_from_delivery_receipt_ref", "resumed_from_rebase_ref", "original_execution_ref", "parent_run_ref"):
        if (result or {}).get(key):
            value, ref = reader.reference(result[key], root / "run_result.json", root / "a")
            lineage.append({"kind": key, "ref": ref, "status": "reference_verified" if value else "unconfirmed",
                "claim": "explicit_recorded_lineage_not_new_execution_or_success"})
    for path in selected:
        value, ref = reader.read(path)
        if not value:
            continue
        schema = value.get("schema_version", value.get("schema"))
        if schema == "task-agent-product-execution.v2":
            if value.get("run_id") != run_id:
                lineage.append({"kind": "separate_run_not_merged", "run_id": value.get("run_id"), "ref": ref})
                continue
            executions.append(_native_chain(reader, value, path))
        elif schema in {"csharp_repository_stage_fact_v1", "csharp_repository_integrity_v1"}:
            reader.expand(value, path, _store_for(path, root))
            facts.append({"ref": ref, "schema": schema, "stage": value.get("stage"), "status": value.get("status"),
                "candidate_ref": value.get("context", {}).get("candidate_ref"),
                "candidate_source_id": value.get("context", {}).get("candidate_source_id"),
                "execution": _preview(value.get("execution")), "checks": value.get("checks"),
                "claim": "recorded_execution_fact_not_acceptance"})
        elif path.parent.name in {"session-checkpoints", "delivery-receipts"}:
            reader.expand(value, path, _store_for(path, root))
            lineage.append({"kind": "recovery_or_delivery_reference", "ref": ref,
                "claim": "checkpoint_presence_does_not_prove_resume", "fields": list(value)})
    # 为什么要有它：恢复的累计原件可能重复包含旧调用，只按准确请求引用去重，不把预算或缓存输入再加一次。
    unique = {}
    for execution in executions:
        for turn in execution["turns"]:
            identity = (turn["request_ref"].get("path"), turn["request_ref"].get("sha256"), turn.get("reservation_id"))
            if identity[1]:
                old = unique.get(identity)
                if old and old["response_ref"].get("sha256") and turn["response_ref"].get("sha256") != old["response_ref"]["sha256"]:
                    reader.issues.append({"status": "ambiguous", "reason": "One request identity has conflicting response records"})
                    unique[identity] = {**old, "usage": None}
                    continue
                unique[identity] = turn
    cost = {"scope": "unique_v2_technical_requests_only_not_whole_task_or_single_context", "requests": len(unique),
        "identity_basis": "request_path_sha256_and_reservation_id_not_content_only",
        "recorded_run_budget": (result or {}).get("model_call_budget", (manifest or {}).get("model_call_budget")),
        "whole_task_cost_complete": False, "money": None, "human_help": "not_inferred"}
    for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
        amounts = [(turn.get("usage") or {}).get(key) for turn in unique.values()]
        known = [n for n in amounts if type(n) is int and n >= 0]
        cost[key] = {"known_sum": sum(known), "unknown_requests": len(amounts) - len(known)}
    projections = []
    for (path, _), value in list(reader.values.items()):
        if not isinstance(value, dict):
            continue
        if value.get("schema") in {"task-agent-candidate.v2", "task-agent-check.v2", "task-agent-public-check-feedback.v2", "task-agent-delivery-receipt.v2", "task-agent-delivery-checkpoint.v2"}:
            projections.append({"path": path, "schema": value["schema"], "candidate_source_id": value.get("candidate_source_id"),
                "candidate_ref": value.get("candidate_ref"), "parent_revision_ref": value.get("parent_revision_ref"),
                "status": value.get("status"), "changes": _preview(value.get("changes")),
                "delivered_artifact_refs": value.get("delivered_artifact_refs"),
                "claim": "indexed_original_not_new_acceptance"})
    if not executions:
        reader.issues.append({"status": "missing", "reason": "No v2 native execution indexed; legacy manifest references remain available"})
    report.update(status="partial" if reader.issues else "ok", complete=not reader.issues,
        run_root=str(root), manifest_ref=manifest_ref, result_ref=result_ref,
        terminal={key: _preview((result or {}).get(key)) for key in ("status", "current_stage", "completion_classification", "task_agent_outcome", "delivery_ref")},
        executions=executions, stage_facts=facts, lineage=lineage, artifacts=projections, cost=cost,
        records=list(reader.records.values()), links=reader.links, issues=reader.issues)
    return _sanitize(report, [])


def render_evidence_markdown(report):
    lines = [f"# 运行证据：{report['run_id']}", "", f"状态：{report['status']}；索引及引用检查完整：{report.get('complete', False)}（非任务验收）。只读查询，不执行任务。", "",
        "关联只证明记录之间的联系，不证明模型理解、采纳或业务成功。", ""]
    for execution in report.get("executions", []):
        lines += ["## 实际请求与行动", "", "| 轮次 | 请求准备 | 发送尝试 | 响应 | 请求动作 |", "|---|---|---|---|---|"]
        for turn in execution["turns"]:
            names = ", ".join(c.get("name", "unknown") for c in turn["requested_actions"])
            lines.append(f"| {turn['turn']} | {turn['request_prepared']} | {turn['send_attempt_observed']} | {turn['response_observed']} | {names} |")
        lines += ["", "已执行动作（不将计划内容当作代码修改）：", "```json", json.dumps([
            {key: row.get(key) for key in ("turn", "call_id", "name", "status", "action_kind", "result_ref")}
            for row in execution["actions"]], ensure_ascii=False, indent=2), "```", "",
            "公开反馈进入请求（不证明采纳）：", "```json", json.dumps([
                {key: row.get(key) for key in ("turn", "call_id", "matches_recorded_tool_result", "included_in_provider_request", "request_ref")}
                for row in execution["feedback_links"]], ensure_ascii=False, indent=2), "```", ""]
    for title, key in (("累计成本与口径", "cost"), ("阶段事实", "stage_facts"), ("恢复或另开诊断", "lineage"),
                       ("缺口", "issues"), ("原件索引", "records"), ("精确引用关联", "links")):
        value = report.get(key)
        if isinstance(value, list) and len(value) > 30:
            value = {"items": value[:30], "display_truncated": True, "total_items": len(value),
                "full_index": "same command with --format json"}
        lines += [f"## {title}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)
