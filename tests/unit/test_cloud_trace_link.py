"""Offline checks for local/Cloud identity binding; no network or model execution.

为什么要有它：核验状态只来自已有真实身份与有界读取结果，不能因为未查到、异常或缺少结束记录而捏造同步结论。
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.observability.dashboard.services import cloud_trace_link as subject


def observation(**changes):
    row = {"trace_id": "a" * 32, "id": "b" * 16, "parent_id": None, "session_id": "session-fixture",
           "name": "patch.generate", "kind": "generation", "status": "completed",
           "started_at": "2026-10-18T11:00:00.123456+00:00", "finished_at": "2026-10-18T11:00:01.123456+00:00",
           "attributes": {"transport_started": True, "input_payload_sha256": "f" * 64, "private_text": "PRIVATE_SOURCE_PAYLOAD"}}
    row.update(changes)
    return {"observation": row, "source": {"path": "/authorized/task/events/one.json", "sha256": "d" * 64, "json_pointer": ""}}


def cloud_row(item):
    row = item["observation"]
    mapped = subject._metadata(row["attributes"])
    return {"observation_id": subject.live_observation_id(row["id"]), "name": row["name"], "type": row["kind"].upper(),
            "parent_id": None, "start_time": "2026-10-18T11:00:00.123Z", "end_time": "2026-10-18T11:00:01.123Z",
            "facts": {key: mapped[key] for key in subject._FACTS if key in mapped}}


def reader(monkeypatch, rows, *, error=None):
    calls = []
    class FakeReader:
        def __init__(self, transport, *, page_budget):
            assert page_budget == 2
            self.transport = transport
        def read(self, trace_id, **window):
            calls.append({"trace_id": trace_id, "window": window})
            if error:
                raise error
            return {"trace_id": trace_id, "execution_tree": rows}
    monkeypatch.setattr(subject, "CloudExecutionReader", FakeReader)
    return calls


def test_summary_does_not_bind_unidentified_export_report_or_network(monkeypatch):
    monkeypatch.setattr(subject, "CloudExecutionReader", lambda *_args, **_kwargs: pytest.fail("No networking in summary"))
    item = observation()
    report = {"trace": {"mode": "local_only", "cloud_readback": "not_performed"},
              "source": {"path": "/authorized/task/events/finish.json", "sha256": "e" * 64, "json_pointer": "/summary"}}
    result = subject.cloud_trace_summary([report], [item])
    assert result["traces"][0]["cloud_trace_id"] == "a" * 32
    assert result["traces"][0]["mode"] == "not_recorded"
    assert result["unassociated_export_reports"][0]["mode"] == "local_only"
    assert result["network_performed"] is False


def test_named_metadata_keeps_http_acceptance_distinct_from_readback():
    item = observation()
    report = {"trace": {"trace_id": "a" * 32, "mode": "metadata_only", "cloud_readback": "not_performed",
              "export": {"http_accepted": 1, "remote_readback": "not_performed", "password": "not exported"}},
              "source": {"path": "/authorized/run_manifest.json", "sha256": "e" * 64, "json_pointer": "/execution_trace"}}
    result = subject.cloud_trace_summary([report], [item])["traces"][0]
    assert result["export"]["http_accepted"] == 1
    assert result["cloud_readback"] == "not_performed"
    assert "password" not in result["export"]


def test_verified_mapping_uses_existing_ids_metadata_and_millisecond_times(monkeypatch):
    item = observation()
    calls = reader(monkeypatch, [cloud_row(item)])
    result = subject.verify_cloud_trace("a" * 32, [item], transport=SimpleNamespace(base_url="https://langfuse.example"))
    assert result["status"] == "verified_readback_at_check_time"
    assert result["matched_observation_ids"] == ["b" * 16]
    assert result["uploads_performed"] == 0 and result["local_state_modified"] is False
    assert "execution_tree" not in result
    assert result["cloud_base_url"] == "https://langfuse.example" and result["project_url"] is None
    assert set(calls[0]["window"]) == {"from_start_time", "to_start_time"}
    assert "/authorized" not in repr(calls) and "PRIVATE_SOURCE_PAYLOAD" not in repr(calls)


def test_empty_query_is_not_proof_of_never_exported(monkeypatch):
    reader(monkeypatch, [])
    result = subject.verify_cloud_trace("a" * 32, [observation()], transport=object())
    assert result["status"] == "not_found_at_check_time"
    assert "never exported" in result["note"]


@pytest.mark.parametrize("field", ["name", "parent_id", "start_time", "facts"])
def test_mismatched_mapping_is_not_verified(monkeypatch, field):
    item = observation()
    remote = cloud_row(item)
    remote[field] = {"transport_started": False} if field == "facts" else "different"
    reader(monkeypatch, [remote])
    result = subject.verify_cloud_trace("a" * 32, [item], transport=object())
    assert result["status"] == "local_cloud_mismatch"


def test_missing_local_unfinished_observation_prevents_full_readback_claim(monkeypatch):
    complete = observation()
    unfinished = observation(id="c" * 16, status="running", finished_at=None)
    reader(monkeypatch, [cloud_row(complete)])
    result = subject.verify_cloud_trace("a" * 32, [complete, unfinished], transport=object())
    assert result["status"] == "partial_readback_match"
    assert result["incomplete_local_observation_ids"] == ["c" * 16]


def test_old_running_snapshot_does_not_replace_completed_local_row(monkeypatch):
    complete = observation()
    reader(monkeypatch, [cloud_row(complete)])
    earlier = observation(status="running", finished_at=None)
    result = subject.verify_cloud_trace("a" * 32, [complete, earlier], transport=object())
    assert result["status"] == "verified_readback_at_check_time"


def test_conflicting_local_final_rows_and_large_time_window_do_not_query(monkeypatch):
    calls = reader(monkeypatch, [])
    changed = observation()
    changed["observation"]["attributes"]["transport_started"] = False
    assert subject.verify_cloud_trace("a" * 32, [observation(), changed], transport=object())["status"] == "local_observation_conflict"
    old = observation(started_at="2026-01-01T00:00:00+00:00")
    assert subject.verify_cloud_trace("a" * 32, [old], transport=object())["status"] == "local_time_window_exceeds_query_limit"
    assert calls == []


def test_safe_schema_error_and_private_exception_body(monkeypatch):
    reader(monkeypatch, [], error=ValueError("LANGFUSE_QUERY_SCHEMA_CHANGED"))
    result = subject.verify_cloud_trace("a" * 32, [observation()], transport=object())
    assert result["error_code"] == "LANGFUSE_QUERY_SCHEMA_CHANGED"
    reader(monkeypatch, [], error=RuntimeError("private credential and provider body"))
    result = subject.verify_cloud_trace("a" * 32, [observation()], transport=object())
    assert result["status"] == "verification_unavailable"
    assert "private credential" not in repr(result)


def test_default_transport_timeout_and_no_unsafe_cloud_url(monkeypatch):
    transport = SimpleNamespace(timeout_seconds=10, base_url="https://user:private@langfuse.example/?key=hidden")
    monkeypatch.setattr(subject.LangfuseHttpTransport, "from_environment", lambda: transport)
    reader(monkeypatch, [])
    result = subject.verify_cloud_trace("a" * 32, [observation()])
    assert transport.timeout_seconds == 5
    assert "cloud_base_url" not in result
    assert "private@" not in repr(result)


def test_missing_bound_local_source_does_not_query(monkeypatch):
    calls = reader(monkeypatch, [])
    item = deepcopy(observation())
    item["source"].pop("sha256")
    result = subject.verify_cloud_trace("a" * 32, [item], transport=object())
    assert result["status"] == "verification_unavailable" and result["network_performed"] is False
    assert calls == []
