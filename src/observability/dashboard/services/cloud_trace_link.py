"""Small read-only Cloud association for the existing local execution identities.

为什么要有它：本地原件与 Langfuse 用相同观察身份核对，云查询不是新执行，也不把 HTTP 接受当远端已验证。
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from src.observability.patch_langfuse import LangfuseHttpTransport
from src.observability.patch_langfuse_query import CloudExecutionReader, _FACTS, _same_observed_time
from src.observability.patch_live_langfuse import _metadata, live_observation_id, live_trace_id


_EXPORT_FIELDS = {"status", "received", "queued", "running_not_exported", "duplicate_records", "conflicting_updates",
    "invalid_records", "queue_dropped", "post_attempts", "http_accepted", "partial_responses", "send_failures",
    "pending", "queued_pending", "active_sends", "closed_to_new_records", "worker_alive", "last_error_type",
    "deadline_exhausted", "remote_readback", "delivery_scope", "retry_policy", "deduplication_scope",
    "score_queued", "score_post_attempts", "score_http_accepted", "score_send_failures", "score_skipped_unaccepted_observation"}
_SAFE_ERROR_CODES = {"LANGFUSE_QUERY_SCHEMA_CHANGED", "LANGFUSE_QUERY_RPC_INVALID", "LANGFUSE_QUERY_TOOL_FAILED",
    "LANGFUSE_QUERY_CONTENT_INVALID", "LANGFUSE_QUERY_PAGE_INVALID", "LANGFUSE_QUERY_PAGE_BUDGET_EXHAUSTED",
    "LANGFUSE_QUERY_CURSOR_INVALID", "LANGFUSE_QUERY_SCHEMA_UNAVAILABLE", "LANGFUSE_QUERY_FIELDS_CHANGED",
    "LANGFUSE_TRACE_ID_INVALID", "LANGFUSE_QUERY_TIME_WINDOW_INVALID", "LANGFUSE_QUERY_TIME_WINDOW_REQUIRED",
    "LANGFUSE_QUERY_TIME_RANGE_INVALID",
    "LANGFUSE_BASE_URL_INVALID", "LANGFUSE_PROJECT_KEYS_REQUIRED", "LANGFUSE_TRANSPORT_BUDGET_INVALID",
    "LANGFUSE_RESPONSE_TOO_LARGE", "LANGFUSE_RESPONSE_INVALID"}


def _source(value):
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str)
            or not value["path"] or not isinstance(value.get("sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", value["sha256"]) is None
            or not isinstance(value.get("json_pointer"), str)
            or value["json_pointer"] and not value["json_pointer"].startswith("/")):
        raise ValueError("LOCAL_TRACE_SOURCE_REFERENCE_REQUIRED")
    return {key: value[key] for key in ("path", "sha256", "json_pointer")}


def _state(trace):
    export = trace.get("export")
    safe = {}
    if isinstance(export, dict):
        for key in _EXPORT_FIELDS & export.keys():
            value = export[key]
            if value is None or type(value) in (bool, int) or type(value) is float and math.isfinite(value):
                safe[key] = value
            elif isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
                safe[key] = value
    return {"mode": trace.get("mode") if trace.get("mode") in {"local_only", "metadata_only"} else "not_recorded",
            "export": safe if isinstance(export, dict) else "not_recorded",
            "cloud_readback": trace.get("cloud_readback") if trace.get("cloud_readback") in {
                "not_performed", "verified", "failed", "unknown"} else "not_recorded",
            "basis": "recorded_local_metadata_not_a_fresh_cloud_check"}


def _observations(sources):
    indexed, conflicts = {}, []
    for item in sources:
        if not isinstance(item, dict) or not isinstance(item.get("observation"), dict):
            raise ValueError("LOCAL_OBSERVATION_REQUIRED")
        row, source = item["observation"], _source(item.get("source"))
        key = (live_trace_id(row.get("trace_id")), live_observation_id(row.get("id")))
        old = indexed.get(key)
        if old is not None:
            previous = old["observation"]
            identity_changed = any(previous.get(field) != row.get(field) for field in ("name", "kind", "parent_id", "started_at"))
            if identity_changed or previous.get("finished_at") and row.get("finished_at") and previous != row:
                conflicts.append({"trace_id": key[0], "observation_id": key[1], "sources": [old["source"], source]})
                continue
            if previous.get("finished_at") and not row.get("finished_at"):
                continue
        indexed[key] = {"observation": row, "source": source}
    return indexed, conflicts


def cloud_trace_summary(trace_records, observation_sources):
    """Summarize caller-verified local rows without reading files or networking."""
    grouped, unassociated, issues = {}, [], []
    try:
        observations, conflicts = _observations(observation_sources)
    except ValueError as error:
        return {"traces": [], "unassociated_export_reports": [], "issues": [str(error)], "network_performed": False}
    for (trace_id, _), item in observations.items():
        row = item["observation"]
        group = grouped.setdefault(trace_id, {"trace_id": row["trace_id"], "cloud_trace_id": trace_id,
            "session_ids": [], "sources": [], "export_records": [], "local_observation_count": 0,
            "finished_observation_count": 0, "mode": "not_recorded", "export": "not_recorded", "cloud_readback": "not_recorded"})
        group["local_observation_count"] += 1
        group["finished_observation_count"] += bool(row.get("finished_at"))
        if isinstance(row.get("session_id"), str) and row["session_id"] not in group["session_ids"]:
            group["session_ids"].append(row["session_id"])
        group["sources"].append(item["source"])
    for item in trace_records:
        try:
            trace, source = item["trace"], _source(item["source"])
            if not isinstance(trace, dict):
                raise ValueError("LOCAL_TRACE_METADATA_REQUIRED")
            state = _state(trace)
            if not trace.get("trace_id"):
                unassociated.append({**state, "source": source, "binding": "no_trace_identity_recorded"})
                continue
            trace_id = live_trace_id(trace["trace_id"])
            group = grouped.setdefault(trace_id, {"trace_id": trace["trace_id"], "cloud_trace_id": trace_id,
                "session_ids": [], "sources": [], "export_records": [], "local_observation_count": 0, "finished_observation_count": 0})
            group["export_records"].append({**state, "source": source})
            group.update({key: state[key] for key in ("mode", "export", "cloud_readback")})
            group["sources"].append(source)
            if isinstance(trace.get("session_id"), str) and trace["session_id"] not in group["session_ids"]:
                group["session_ids"].append(trace["session_id"])
        except (ValueError, KeyError, TypeError):
            issues.append("Invalid local trace metadata or source reference")
    for group in grouped.values():
        if len(group["export_records"]) > 1:
            states = {json.dumps({key: row[key] for key in ("mode", "export", "cloud_readback")}, sort_keys=True)
                      for row in group["export_records"]}
            if len(states) > 1:
                group.update(mode="multiple_recorded_states", export="see_export_records", cloud_readback="see_export_records")
        group["source_count"] = len(group["sources"])
        group["identity_conflict"] = any(row["trace_id"] == group["cloud_trace_id"] for row in conflicts)
    return {"traces": list(grouped.values()), "unassociated_export_reports": unassociated,
            "issues": issues, "conflicts": conflicts, "network_performed": False, "uploads_performed": 0,
            "scope": "Caller verified local file bytes; no paths or contents are sent to Cloud by this summary."}


def _time(value):
    if not isinstance(value, str):
        raise ValueError("LOCAL_TRACE_TIME_REQUIRED")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("LOCAL_TRACE_TIMEZONE_REQUIRED")
    return parsed.astimezone(timezone.utc)


def verify_cloud_trace(trace_id, observations, transport=None):
    """Explicit bounded readback; never upload, retry POST ingestion or read local paths.

    为什么要有它：空查询、读取失败与已核对匹配是不同事实，不能凭一个 trace ID 宣称云端收到或从未收到。
    """
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "verification_unavailable",
              "trace_id": trace_id, "network_performed": False, "uploads_performed": 0,
              "local_state_modified": False, "project_url": None,
              "limits": {"page_budget_per_tool": 2, "default_http_timeout_seconds": 5, "automatic_retry": False}}
    try:
        identity = live_trace_id(trace_id)
        indexed, conflicts = _observations(observations)
        selected = {key[1]: item for key, item in indexed.items() if key[0] == identity}
        if not selected:
            result["status"] = "local_trace_not_found"
            return result
        if any(item["trace_id"] == identity for item in conflicts):
            result["status"] = "local_observation_conflict"
            return result
        times = [_time(item["observation"].get("started_at")) for item in selected.values()]
        ends = [_time(item["observation"]["finished_at"]) for item in selected.values() if item["observation"].get("finished_at")]
        start, end = min(times) - timedelta(milliseconds=1), max(times + ends) + timedelta(milliseconds=1)
        if end - start > timedelta(days=14):
            result["status"] = "local_time_window_exceeds_query_limit"
            return result
        result.update(trace_id=identity, query_window={"from_start_time": start.isoformat(), "to_start_time": end.isoformat()},
                      local_observation_count=len(selected), local_sources=[item["source"] for item in selected.values()])
        unfinished = sorted(key for key, item in selected.items() if not item["observation"].get("finished_at"))
        result["incomplete_local_observation_ids"] = unfinished
        if transport is None:
            transport = LangfuseHttpTransport.from_environment()
            transport.timeout_seconds = 5.0
        base = getattr(transport, "base_url", None)
        if isinstance(base, str):
            address = urlsplit(base)
            if (address.scheme == "https" or address.scheme == "http" and address.hostname in {"127.0.0.1", "localhost", "::1"}) and (
                    address.hostname and not address.username and not address.password and not address.query
                    and not address.fragment and address.path in {"", "/"}):
                result["cloud_base_url"] = base.rstrip("/")
        reader = CloudExecutionReader(transport, page_budget=2)
        result["network_performed"] = True
        cloud = reader.read(identity, from_start_time=start.isoformat(), to_start_time=end.isoformat())
        if cloud.get("trace_id") != identity or not isinstance(cloud.get("execution_tree"), list):
            raise ValueError("CLOUD_TRACE_RESPONSE_IDENTITY_INVALID")
        rows = cloud["execution_tree"]
        result["cloud_observation_count"] = len(rows)
        if not rows:
            result.update(status="not_found_at_check_time", note="No observations returned for this bounded query; this does not prove never exported.")
            return result
        matched, mismatched, cloud_only, incomplete, seen = [], [], [], unfinished, set()
        for row in rows:
            observation_id = row.get("observation_id")
            if not isinstance(observation_id, str) or observation_id in seen:
                raise ValueError("CLOUD_OBSERVATION_IDENTITY_INVALID")
            seen.add(observation_id)
            item = selected.get(observation_id)
            if item is None:
                cloud_only.append(observation_id)
                continue
            local = item["observation"]
            if not local.get("finished_at"):
                continue
            mapped = _metadata(local.get("attributes", {}))
            expected = {key: mapped[key] for key in _FACTS if key in mapped}
            facts = row.get("facts", {})
            parent = live_observation_id(local["parent_id"]) if local.get("parent_id") is not None else None
            matches = (local.get("name") == row.get("name") and str(local.get("kind")).upper() == row.get("type")
                       and parent == row.get("parent_id")
                       and _same_observed_time(local.get("started_at"), row.get("start_time"))
                       and _same_observed_time(local.get("finished_at"), row.get("end_time"))
                       and all(facts.get(key) == value for key, value in expected.items()))
            (matched if matches else mismatched).append(observation_id)
        expected_ids = {key for key, item in selected.items() if item["observation"].get("finished_at")}
        missing = sorted(expected_ids - seen)
        result.update(status="local_cloud_mismatch" if mismatched else "partial_readback_match" if
                      cloud_only or incomplete or missing else "verified_readback_at_check_time",
                      matched_observation_ids=matched, mismatched_observation_ids=mismatched,
                      cloud_only_observation_ids=cloud_only, incomplete_local_observation_ids=incomplete,
                      local_only_finished_observation_ids=missing,
                      binding_scope="Existing live IDs, names, parent IDs, recorded times and allowlisted metadata mapping; not business correctness.")
        return result
    except Exception as error:
        result.update(status="verification_unavailable", error_type=type(error).__name__)
        code = getattr(error, "code", None)
        if code is None and str(error) in _SAFE_ERROR_CODES:
            code = str(error)
        if isinstance(code, str) and (code in _SAFE_ERROR_CODES or re.fullmatch(r"LANGFUSE_HTTP_[1-5][0-9]{2}", code)):
            result["error_code"] = code
        return result
