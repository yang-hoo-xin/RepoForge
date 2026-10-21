"""Run formats and recorded lifecycle facts share a read-only browser.

为什么要有它：产品、实验和后续交付不能因文件布局不同而丢失记录或混算历史成绩。
"""
import hashlib
import json
from pathlib import Path

from src.observability.dashboard.services.trace_service import TraceService


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def discover(root):
    return TraceService().discover_runs(allowed_roots=[str(root)])["runs"]


def inspect(root, run, **kwargs):
    return TraceService().inspect_run(allowed_roots=[str(root)], run_id=run["run_id"], **kwargs)


def test_same_feature_different_runs_and_delivery_do_not_rewrite_original(tmp_path):
    for name in ("ordinary", "restored"):
        root = tmp_path / name
        terminal = write(root / "run_result.json", {"run_id": name, "feature_id": "same-feature", "status": "blocked"})
        write(root / "run_manifest.json", {"run_id": name, "terminal_record": terminal})
    restored = tmp_path / "restored"
    write(restored / "continuations/one/run_result.json", {"run_id": "restored", "status": "awaiting_approval"})
    write(restored / "delivery/result.json", {"status": "delivered", "scope": "isolated formal delivery"})
    runs = discover(tmp_path)
    assert len(runs) == 2 and len({run["run_id"] for run in runs}) == 2
    result = inspect(tmp_path, next(run for run in runs if Path(run["run_root"]).name == "restored"))
    assert result["summary"]["status"] == "blocked"
    assert {row["status"] for row in result["summary"]["terminal_records"]} == {"blocked", "awaiting_approval", "delivered"}


def test_outcome_scopes_separate_original_recovery_and_reported_delivery(tmp_path):
    root = tmp_path / "restored"
    original = write(root / "run_result.json", {"run_id": "restored", "current_stage": "blocked"})
    write(root / "run_manifest.json", {"run_id": "restored", "terminal_record": original})
    write(root / "continuations/one/run_result.json", {"run_id": "restored", "parent_run_id": "restored",
        "entry": "retained-task-agent-handoff", "parent_result_ref": original, "status": "awaiting_approval"})
    proof = write(root / "delivery/check.json", {"status": "passed", "scope": "controlled not game"})
    write(root / "delivery/result.json", {"status": "delivered", "source_origin": "product_model_proposal",
        "delivery_complete": True, "acceptance_passed": True, "execution_refs": [], "evidence_refs": [proof],
        "candidate_verification_scope": "controlled not game"})
    write(root / "unrelated/result.json", {"status": "delivered"})
    result = inspect(tmp_path, discover(tmp_path)[0])["summary"]
    assert result["status"] == "blocked"
    scopes = result["outcome_scopes"]
    assert scopes["original"]["reported_status"] == "blocked"
    assert {row["reported_status"] for row in scopes["followups"]} == {"awaiting_approval", "delivered"}
    delivered = next(row for row in scopes["followups"] if row["reported_status"] == "delivered")
    assert delivered["verification_scope"] == "controlled not game"
    assert delivered["basis"] == "reported_state_not_independent_delivery_verdict"
    assert scopes["evidence_status"] == "complete"
    assert "independent" in scopes["evidence_status_scope"]


def test_timeline_is_time_sorted_and_record_pagination_does_not_drop_events(tmp_path):
    root = tmp_path / "clock"
    write(root / "start.json", {"candidate_mode": "current", "status": "starting"})
    write(root / "00-later.json", {"event": "build_finished", "status": "failed", "role": "compiler",
        "started_at": "2026-10-19T10:00:00+08:00", "finished_at": "2026-10-19T10:00:03+08:00"})
    write(root / "99-earlier.json", {"event": "candidate_submitted", "status": "saved", "stage": "patch",
        "finished_at": "2026-10-19T01:00:00Z"})
    write(root / "no-time.json", {"event": "missing_clock", "status": "saved"})
    write(root / "bad-time.json", {"event": "bad_clock", "started_at": "yesterday"})
    write(root / "naive-time.json", {"event": "naive_clock", "started_at": "2026-10-19T01:00:00"})
    run = discover(tmp_path)[0]
    first = inspect(tmp_path, run, record_limit=1)
    second = inspect(tmp_path, run, record_limit=1, record_cursor=first["next_record_cursor"])
    assert first["timeline"] == second["timeline"]
    assert [row["name"] for row in first["timeline"]] == ["candidate_submitted", "build_finished"]
    assert first["timeline"][0]["sort_time_utc"] == "2026-10-19T01:00:00Z"
    assert first["timeline"][1]["started_at_utc"] == "2026-10-19T02:00:00Z"
    assert first["timeline"][1]["role"] == "compiler"
    assert {row["name"] for row in first["undated_records"]} >= {"missing_clock", "bad_clock", "naive_clock"}
    assert first["timeline_coverage"]["record_index_pagination_independent"] is True


def test_experiment_counts_actual_unique_requests_not_budget_or_imported_check(tmp_path):
    root = tmp_path / "trial-owner"
    write(root / "controller-result.json", {"candidate_mode": "current", "status": "trial_ended_not_success_claim", "total_wall_seconds": 12})
    write(root / "trial/w/run_manifest.json", {"build_dependency": {"not_a_main_manifest": True}})
    write(root / "trial/result.json", {"candidate_mode": "current", "status": "blocked", "model_budget": {"consumed": 13}})
    event = {"id": "request-one", "trace_id": "trace-one", "name": "model.request", "kind": "span", "status": "running",
             "attributes": {"transport_started": True}}
    write(root / "trial/events/001.json", event)
    event = {**event, "status": "completed", "finished_at": "2026-10-19T01:00:00+00:00",
             "attributes": {"transport_started": True, "usage": {"input_tokens": 14, "output_tokens": 3}}}
    write(root / "trial/events/002.json", event)
    write(root / "trial/events/003.json", {**event, "id": "generation", "name": "patch.generate", "kind": "generation"})
    write(root / "trial/a/session/imported-check.json", {"status": "behavior_failed", "verification_summary": {"checks": 26, "failed_checks": 18},
                                                       "origin": "historical_business_failure_not_new_execution"})
    write(root / "trial/response-001.json", {"content": '{"task_status":"completed"}'})
    runs = discover(tmp_path)
    assert len(runs) == 1 and runs[0]["format"] == "experiment"
    view = inspect(tmp_path, runs[0], record_limit=1)
    assert view["summary"]["status"] == "blocked"
    assert view["summary"]["model_calls"] == 1
    assert view["summary"]["input_tokens"] == 14
    assert view["summary"]["output_tokens"] == 3
    assert view["summary"]["elapsed_seconds"] == 12
    assert view["next_record_cursor"] == 1
    assert view["index_complete"] is False


def test_interrupted_run_keeps_unknown_result_and_usage(tmp_path):
    root = tmp_path / "interrupted"
    write(root / "start.json", {"candidate_mode": "baseline", "status": "starting_owned_service"})
    write(root / "trial/events/001.json", {"id": "pending", "trace_id": "trace", "name": "model.request", "status": "running",
        "attributes": {"transport_started": True, "response_received": False}})
    view = inspect(tmp_path, discover(tmp_path)[0])
    assert view["summary"]["model_calls"] == 1
    assert view["summary"]["input_tokens"] is None
    assert view["summary"]["usage_complete"] is False
    assert view["summary"]["terminal_records"] == []
    assert view["run"]["live_process_status"] == "not_queried"


def test_cloud_sources_reuse_original_trace_and_block_changed_local_bytes(tmp_path, monkeypatch):
    from src.observability.dashboard.services import cloud_trace_link

    root = tmp_path / "trace"
    trace_id = "a" * 32
    manifest = root / "run_manifest.json"
    original = {"run_id": "trace", "execution_trace": {"trace_id": trace_id, "mode": "local_only",
        "cloud_readback": "not_performed"}, "execution_observations": {"one": {"id": "b" * 16,
        "trace_id": trace_id, "name": "model.request", "kind": "span", "status": "completed",
        "started_at": "2026-10-19T01:00:00Z", "finished_at": "2026-10-19T01:00:01Z", "attributes": {}}}}
    write(manifest, original)
    run = discover(tmp_path)[0]
    view = inspect(tmp_path, run)
    association = view["summary"]["cloud_association"]
    assert association["traces"][0]["trace_id"] == trace_id
    calls = []

    def verify(identity, rows):
        calls.append((identity, rows))
        return {"status": "not_found_at_check_time", "network_performed": True}

    monkeypatch.setattr(cloud_trace_link, "verify_cloud_trace", verify)
    args = {"allowed_roots": [str(tmp_path)], "run_id": run["run_id"], "trace_id": trace_id,
            "expected_source_fingerprint": association["source_fingerprint"]}
    result = TraceService().verify_run_cloud(**args)
    assert result["status"] == "not_found_at_check_time" and len(calls) == 1
    assert calls[0][1][0]["source"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    write(manifest, {**original, "additional_fact": "changed"})
    result = TraceService().verify_run_cloud(**args)
    assert result == {"status": "local_sources_changed", "network_performed": False,
        "message": "Refresh the local run before verifying its cloud association."}
    assert len(calls) == 1


def test_unbound_export_report_keeps_missing_trace_identity(tmp_path):
    root = tmp_path / "b"
    write(root / "start.json", {"candidate_mode": "current", "status": "starting"})
    write(root / "trial/events/001.json", {"id": "b" * 16, "trace_id": "a" * 32,
        "name": "model.request", "kind": "span", "status": "completed", "attributes": {}})
    write(root / "trial/events/002.json", {"phase": "trace_finished", "summary": {"mode": "local_only"}})
    association = inspect(tmp_path, discover(tmp_path)[0])["summary"]["cloud_association"]
    assert len(association["unassociated_export_reports"]) == 1
    assert "trace_id" not in association["unassociated_export_reports"][0].get("trace", {})


def test_unknown_new_json_array_and_root_record_pagination(tmp_path):
    root = tmp_path / "new-run"
    write(root / "start.json", {"candidate_mode": "current", "status": "starting"})
    run = discover(tmp_path)[0]
    write(root / "new-format.json", [1, {"new-field": "last"}])
    first = inspect(tmp_path, run, record_limit=1)
    second = inspect(tmp_path, run, record_limit=1, record_cursor=first["next_record_cursor"])
    rows = first["records"] + second["records"]
    row = next(item for item in rows if item["relative_path"] == "new-format.json")
    body = TraceService().read_record(allowed_roots=[str(tmp_path)], record_id=row["record_id"])
    assert body["status"] == "ok" and body["json_value"] == [1, {"new-field": "last"}]
    assert second["index_complete"] is True


def test_bad_manifest_reference_does_not_validate_delivered_result(tmp_path):
    root = tmp_path / "product"
    terminal = write(root / "run_result.json", {"run_id": "product", "status": "delivered"})
    write(root / "run_manifest.json", {"run_id": "product", "terminal_record": {**terminal, "sha256": "0" * 64}})
    view = inspect(tmp_path, discover(tmp_path)[0])
    assert view["summary"]["reported_status"] == "delivered"
    assert view["summary"]["status"] == "integrity_error"
    assert view["summary"]["terminal_records"][0]["integrity"] == "referenced_bytes_invalid"
    assert view["summary"]["outcome_scopes"]["evidence_status"] == "invalid"
    assert view["summary"]["outcome_scopes"]["original"]["reported_status"] == "delivered"


def test_outcome_reference_access_is_partial_not_a_new_delivery_verdict(tmp_path):
    root = tmp_path / "root"
    external = write(tmp_path / "not-authorized.json", {"private": "not shown"})
    write(root / "run_manifest.json", {"run_id": "root"})
    write(root / "run_result.json", {"run_id": "root", "status": "blocked", "external_ref": external})
    view = inspect(root, discover(root)[0])
    assert view["summary"]["outcome_scopes"]["evidence_status"] == "partial"
    assert view["summary"]["outcome_scopes"]["original"]["integrity"] == "file_bytes_read"
    interrupted = tmp_path / "interrupted"
    write(interrupted / "start.json", {"candidate_mode": "current"})
    summary = inspect(interrupted, discover(interrupted)[0])["summary"]
    assert summary["outcome_scopes"]["original"] is None
    assert summary["outcome_scopes"]["evidence_status"] == "unknown"


def test_timeline_explicitly_lists_omissions_and_invalid_clock_order(tmp_path, monkeypatch):
    from src.observability.dashboard.services import run_records

    root = tmp_path / "events"
    write(root / "start.json", {"candidate_mode": "current"})
    write(root / "clock.json", {"event": "backwards", "started_at": "2026-10-19T02:00:00Z",
        "finished_at": "2026-10-19T01:00:00Z"})
    write(root / "large.json", {"event": "large", "started_at": "2026-10-19T01:00:00Z", "body": "x" * 1000})
    (root / "bad.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(run_records, "_TIMELINE_BYTES", 300)
    files = {path: path.read_bytes() for path in root.rglob("*.json")}
    view = inspect(tmp_path, discover(tmp_path)[0], record_limit=1)
    assert any("finished_before_started" in row["time_issues"] for row in view["undated_records"])
    omitted = {row["path"]: row["reason"] for row in view["timeline_coverage"]["omitted_records"]}
    assert omitted == {"bad.json": "corrupt", "large.json": "above_timeline_json_limit"}
    assert view["timeline_coverage"]["complete"] is False
    assert all(path.read_bytes() == raw for path, raw in files.items())


def test_discovery_cursor_does_not_rediscover_inner_trial(tmp_path):
    for name in ("one", "two"):
        root = tmp_path / name
        write(root / "start.json", {"candidate_mode": "current"})
        write(root / "trial/result.json", {"candidate_mode": "current", "status": "blocked"})
    service = TraceService()
    cursor, all_runs = 0, []
    while cursor is not None:
        page = service.discover_runs(allowed_roots=[str(tmp_path)], cursor=cursor, limit=1)
        all_runs.extend(page["runs"])
        cursor = page["next_cursor"]
    assert {Path(row["run_root"]).name for row in all_runs} == {"one", "two"}
    assert len(all_runs) == 2


def test_corrupt_manifest_does_not_hide_existing_run_result(tmp_path):
    root = tmp_path / "broken"
    write(root / "run_result.json", {"run_id": "broken", "current_stage": "blocked",
        "product_execution_result": {"platform_status": "completed", "execution_disposition": "domain_blocked"}})
    (root / "run_manifest.json").write_text('{"broken":', encoding="utf-8")
    page = TraceService().discover_runs(allowed_roots=[str(tmp_path)])
    assert len(page["runs"]) == 1
    assert page["issues"]
    summary = inspect(tmp_path, page["runs"][0])["summary"]
    assert summary["status"] == "blocked"
    assert summary["terminal_records"][0]["platform_status"] == "completed"
    assert summary["terminal_records"][0]["execution_disposition"] == "domain_blocked"


def test_discovery_scan_limit_has_working_continuation(tmp_path):
    for index in range(1005):
        (tmp_path / f"filler-{index:04d}.txt").write_text("metadata only", encoding="utf-8")
    write(tmp_path / "z-last-run/start.json", {"candidate_mode": "current", "status": "starting"})
    service = TraceService()
    first = service.discover_runs(allowed_roots=[str(tmp_path)], limit=1)
    assert not first["complete"] and first["runs"] == []
    second = service.discover_runs(allowed_roots=[str(tmp_path)], cursor=first["next_cursor"], limit=1)
    assert len(second["runs"]) == 1 and Path(second["runs"][0]["run_root"]).name == "z-last-run"


def test_artifact_reference_without_feature_id_uses_verified_target_envelope(tmp_path):
    from src.mod_agent.artifacts import ArtifactStore
    from tests.unit.test_artifact_store import _feature_payload

    root = tmp_path / "product"
    write(root / "start.json", {"candidate_mode": "current"})
    store = ArtifactStore(root / "a")
    artifact = store.write(feature_id="feature_001", artifact_type="feature_spec", payload=_feature_payload())
    write(root / "upstream.json", {"input": {"artifact_type": artifact.artifact_type,
        "artifact_hash": artifact.artifact_hash, "path": artifact.path}})
    view = inspect(tmp_path, discover(tmp_path)[0])
    links = [row for row in view["links"] if row["json_pointer"] == "/input"]
    assert len(links) == 1 and links[0]["status"] == "ok"
    assert links[0]["integrity"] == "verified_artifact_semantics"
    assert links[0]["actual_sha256"] != artifact.artifact_hash


def test_jsonl_sensitive_container_and_corrupt_unquoted_value_are_protected(tmp_path):
    root = tmp_path / "product"
    write(root / "start.json", {"candidate_mode": "current"})
    (root / "log.jsonl").write_text('{"secrets":["JSONL_HIDDEN"]}\n{"status":"done"}\n', encoding="utf-8")
    (root / "bad.json").write_text('{"api_token": UNQUOTED_HIDDEN}', encoding="utf-8")
    service = TraceService()
    view = inspect(tmp_path, discover(tmp_path)[0])
    for name, secret in (("log.jsonl", "JSONL_HIDDEN"), ("bad.json", "UNQUOTED_HIDDEN")):
        row = next(item for item in view["records"] if item["relative_path"] == name)
        result = service.read_record(allowed_roots=[str(tmp_path)], record_id=row["record_id"])
        assert secret not in result["text"]
        assert result["redaction"]["applied"] is True
        if name == "log.jsonl":
            assert "done" in result["text"]


def test_controller_does_not_hide_a_real_nested_product_manifest(tmp_path):
    controller = tmp_path / "controller"
    write(controller / "controller-result.json", {"protocol_ref": {}, "status": "ended"})
    write(controller / "trial/w/run_manifest.json", {"build_dependency": {}})
    product = controller / "trial/runs/product-one"
    write(product / "run_manifest.json", {"run_id": "product-one", "status": "awaiting_approval"})
    write(product / "run_result.json", {"run_id": "product-one", "status": "awaiting_approval"})
    runs = discover(tmp_path)
    assert {(Path(row["run_root"]).name, row["format"]) for row in runs} == {
        ("controller", "experiment"), ("product-one", "product")}


def test_authorization_header_inside_text_redacts_entire_credential(tmp_path):
    root = tmp_path / "product"
    write(root / "start.json", {"candidate_mode": "current"})
    write(root / "text.json", {"message": "error: Authorization: Basic BASIC_PRIVATE_123"})
    view = inspect(tmp_path, discover(tmp_path)[0])
    item = next(row for row in view["records"] if row["relative_path"] == "text.json")
    result = TraceService().read_record(allowed_roots=[str(tmp_path)], record_id=item["record_id"])
    assert "BASIC_PRIVATE_123" not in result["text"]
    assert result["redaction"]["applied"]


def test_product_record_read_marks_parent_controller_reference_outside_its_run(tmp_path):
    controller = tmp_path / "controller"
    controller_ref = write(controller / "controller-result.json", {"protocol_ref": {}, "status": "ended"})
    product = controller / "trial/runs/product-one"
    write(product / "run_manifest.json", {"run_id": "product-one"})
    write(product / "details.json", {"controller_ref": controller_ref})
    runs = discover(tmp_path)
    run = next(item for item in runs if item["format"] == "product")
    view = inspect(tmp_path, run)
    item = next(row for row in view["records"] if row["relative_path"] == "details.json")
    result = TraceService().read_record(allowed_roots=[str(tmp_path)], record_id=item["record_id"])
    assert result["links"][0]["status"] == "ok"
    assert result["links"][0]["historical"] is True
