"""Counterexamples for selected-run accounting, without model or build execution."""
from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from scripts.measure_product_task import main
from src.mod_agent.evaluation.task_measurement import measure_product_task
from src.mod_agent.task_agent_v2_state import save_v2_artifact


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("Measurement must be offline"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode("utf-8")
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def _segment(root, trace="1" * 32, parent=None, usage=None):
    usage = usage if usage is not None else {"input_tokens": 10, "output_tokens": 3}
    request = _write(root / "details/request.json", {"messages": [{"role": "user", "content": "source"}]})
    response = _write(root / "details/response.json", {"findings": []})
    attrs = {"model_role": "requirement_findings", "model": "local-model", "provider": "ollama",
        "transport_started": True, "response_received": True,
        "response_origin": "live_provider_response", "usage": usage,
        "local_detail_refs": {"request": request, "response": response}}
    observations = {"request": {"id": "request", "name": "model.request", "kind": "span",
        "trace_id": trace, "attributes": {"transport_started": True}},
        "generation": {"id": "generation", "name": "requirements.findings", "kind": "generation",
        "trace_id": trace, "parent_id": "request", "status": "completed", "elapsed_ms": 100,
        "attributes": attrs},
        "tool": {"id": "tool", "name": "evidence.source_query", "kind": "tool",
        "trace_id": trace, "status": "completed", "elapsed_ms": 25, "attributes": {}}}
    result = {"run_id": "run-one", "request_id": "request-one", "status": "blocked",
        "current_stage": "blocked", "model_call_count": 8,
        "model_call_budget": {"consumed": 8}, "blockers": [{"code": "TEST_BLOCKED"}]}
    result_ref = _write(root / "run_result.json", result)
    manifest = {"run_id": "run-one", "request_id": "request-one", "status": "blocked",
        "terminal_stage": "blocked", "terminal_record": result_ref,
        "execution_trace": {"trace_id": trace, "mode": "local_only"},
        "execution_observations": observations}
    if parent:
        manifest["parent_result_ref"] = parent
    _write(root / "run_manifest.json", manifest)
    return manifest


def _fixture(tmp_path):
    root = tmp_path / "runs/run-one"
    manifest = _segment(root)
    continuation = root / "continuations/second"
    _segment(continuation, "2" * 32, manifest["terminal_record"])
    return root, continuation


def _measure(root, *continuations):
    return measure_product_task(allowed_root=root.parent, run_root=root, run_id=root.name,
                                continuation_roots=continuations)


def _change_result(root, **changes):
    result = _read(root, "run_result.json")
    result.update(changes)
    manifest = _read(root, "run_manifest.json")
    manifest["terminal_record"] = _write(root / "run_result.json", result)
    _write(root / "run_manifest.json", manifest)


def test_selected_costs_and_tools_do_not_sum_cumulative_budget_or_claim_complete_task(tmp_path):
    root, continuation = _fixture(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    report = _measure(root, continuation)
    assert report["model"]["generation_records"] == 2
    assert report["model"]["recorded_transport_started_total"] == 2
    assert report["model"]["usage"]["input_tokens"] == {"total": 20, "known_subtotal": 20, "observed_calls": 2}
    assert report["model"]["usage"]["output_tokens"]["total"] == 6
    assert report["tools"]["observed_calls"] == 2
    assert report["coverage"]["lineage_verified_between_selections"] is True
    assert report["selected_scope_complete"] is None
    assert report["task_total_input_tokens"] is report["task_elapsed_ms"] is None
    assert report["completion"] is report["independent_task_count"] is None
    assert report["model"]["price"] is report["model"]["independent_http_request_count"] is None
    assert report["last_selected_terminal"]["blockers"] == [{"code": "TEST_BLOCKED"}]
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("invalid", [None, -1, True, 1.5, "3"])
def test_missing_or_invalid_usage_keeps_subtotal_and_unknown_total(tmp_path, invalid):
    root, continuation = _fixture(tmp_path)
    manifest = _read(continuation, "run_manifest.json")
    manifest["execution_observations"]["generation"]["attributes"]["usage"]["input_tokens"] = invalid
    _write(continuation / "run_manifest.json", manifest)
    cost = _measure(root, continuation)["model"]["usage"]
    assert cost["input_tokens"] == {"total": None, "known_subtotal": 10, "observed_calls": 1}
    assert cost["output_tokens"]["total"] == 6


def test_duplicate_continuation_selection_is_not_a_second_attempt(tmp_path):
    root, continuation = _fixture(tmp_path)
    report = _measure(root, continuation, continuation / "../second")
    assert report["selected_segment_count"] == 2
    assert report["duplicate_selections_ignored"] == 1
    assert report["model"]["usage"]["input_tokens"]["total"] == 20


def test_repeated_observation_identity_is_deduplicated_but_conflicts_fail(tmp_path):
    root, continuation = _fixture(tmp_path)
    first = _read(root, "run_manifest.json")
    first["execution_observations"]["generation"]["attributes"].pop("local_detail_refs")
    _write(root / "run_manifest.json", first)
    second = _read(continuation, "run_manifest.json")
    second["execution_trace"] = first["execution_trace"]
    second["execution_observations"] = first["execution_observations"]
    _write(continuation / "run_manifest.json", second)
    report = _measure(root, continuation)
    assert report["duplicate_observations_ignored"] == 3
    assert report["model"]["generation_records"] == 1
    assert report["tools"]["observed_calls"] == 1
    second["execution_observations"]["generation"]["attributes"]["usage"]["input_tokens"] = 999
    _write(continuation / "run_manifest.json", second)
    with pytest.raises(ValueError, match="conflicting recorded facts"):
        _measure(root, continuation)


@pytest.mark.parametrize("started,total", [(True, None), (None, None), (False, 0)])
def test_prepared_request_is_not_a_generation_and_unknown_transport_is_not_free(tmp_path, started, total):
    root, _ = _fixture(tmp_path)
    manifest = _read(root, "run_manifest.json")
    del manifest["execution_observations"]["generation"]
    manifest["execution_observations"]["request"]["attributes"]["transport_started"] = started
    _write(root / "run_manifest.json", manifest)
    report = _measure(root)
    assert report["model"]["generation_records"] == 0
    assert report["model"]["usage"]["input_tokens"]["total"] == total
    assert report["completion"] is None


def test_missing_native_observations_are_not_zero_tokens_or_tools(tmp_path):
    root, _ = _fixture(tmp_path)
    manifest = _read(root, "run_manifest.json")
    del manifest["execution_observations"]
    _write(root / "run_manifest.json", manifest)
    report = _measure(root)
    assert report["model"]["usage"]["input_tokens"]["total"] is None
    assert report["model"]["recorded_transport_started_total"] is None
    assert report["tools"]["selected_record_total"] is None
    assert report["coverage"]["segments_missing_observations"] == [0]


def test_legacy_run_without_observations_can_include_traced_continuation(tmp_path):
    root, continuation = _fixture(tmp_path)
    manifest = _read(root, "run_manifest.json")
    del manifest["execution_observations"]
    _write(root / "run_manifest.json", manifest)
    report = _measure(root, continuation)
    assert report["model"]["generation_records"] == 1
    assert report["model"]["usage"]["input_tokens"] == {
        "total": None, "known_subtotal": 10, "observed_calls": 1}
    assert report["tools"]["selected_record_total"] is None


def test_request_bound_by_later_snapshot_is_not_also_an_orphan(tmp_path):
    root, continuation = _fixture(tmp_path)
    first = _read(root, "run_manifest.json")
    del first["execution_observations"]["generation"]
    _write(root / "run_manifest.json", first)
    second = _read(continuation, "run_manifest.json")
    second["execution_trace"] = first["execution_trace"]
    for row in second["execution_observations"].values():
        row["trace_id"] = first["execution_trace"]["trace_id"]
    _write(continuation / "run_manifest.json", second)
    report = _measure(root, continuation)
    assert report["model"]["request_records_without_generation"] == []
    assert report["model"]["usage"]["input_tokens"]["total"] == 10
    assert report["model"]["recorded_transport_started_total"] == 1


@pytest.mark.parametrize("mode", ["controlled", "calibration", "preparation", "replay"])
def test_platform_completed_and_budget_do_not_establish_real_task_success(tmp_path, mode):
    root, _ = _fixture(tmp_path)
    _change_result(root, status="completed", execution_mode=mode, build_summary={"status": "success"})
    report = _measure(root)
    assert report["completion"] is None
    assert report["last_selected_terminal"]["recorded_product_success"] is None
    assert report["independent_task_count"] is None


def test_unselected_parent_is_not_opened_and_wrong_selected_parent_hash_fails(tmp_path):
    root, continuation = _fixture(tmp_path)
    manifest = _read(continuation, "run_manifest.json")
    manifest["parent_result_ref"] = {"path": str(tmp_path / "unselected/missing.json"), "sha256": "0" * 64}
    _write(continuation / "run_manifest.json", manifest)
    report = _measure(root, continuation)
    assert report["segments"][1]["lineage"]["status"] == "unknown_parent_not_selected"
    assert report["coverage"]["lineage_verified_between_selections"] is False
    manifest["parent_result_ref"]["path"] = str(root / "run_result.json")
    _write(continuation / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="parent result hash mismatch"):
        _measure(root, continuation)


@pytest.mark.parametrize("damage", ["outside_root", "outside_terminal", "outside_detail", "hash", "identity"])
def test_untrusted_paths_or_changed_identity_fail_closed(tmp_path, damage):
    root, continuation = _fixture(tmp_path)
    manifest = _read(root, "run_manifest.json")
    if damage == "outside_root":
        continuation = tmp_path / "unselected"
        _segment(continuation)
    elif damage == "outside_terminal":
        manifest["terminal_record"]["path"] = str(tmp_path / "outside.json")
    elif damage == "outside_detail":
        manifest["execution_observations"]["generation"]["attributes"]["local_detail_refs"]["request"]["path"] = "../../outside.json"
    elif damage == "hash":
        (root / "run_result.json").write_text("{}")
    else:
        manifest["request_id"] = "not-the-task"
    _write(root / "run_manifest.json", manifest)
    with pytest.raises(ValueError):
        _measure(root, continuation)


def test_unc_input_is_rejected_before_resolution(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda *_a, **_k: pytest.fail("Do not access a network path"))
    with pytest.raises(ValueError, match="Network paths"):
        measure_product_task(allowed_root="//host/share", run_root="//host/share/run", run_id="run")


def test_missing_raw_body_is_a_coverage_gap_without_echoing_private_messages(tmp_path):
    root, _ = _fixture(tmp_path)
    (root / "details/request.json").unlink()
    report = _measure(root)
    assert report["coverage"]["unverified_model_local_evidence"][0]["role"] == "request"
    assert "messages" not in json.dumps(report)


def test_cli_uses_same_measurement_and_preserves_existing_reports_and_history(tmp_path, capsys):
    root, continuation = _fixture(tmp_path)
    args = ["--allowed-root", str(root.parent), "--run-root", str(root), "--run-id", root.name,
            "--continuation-root", str(continuation)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model"]["generation_records"] == 2
    assert main([*args, "--output", str(root / "measurement.json")]) == 2
    assert not (root / "measurement.json").exists()
    output = tmp_path / "measurement.json"
    assert main([*args, "--output", str(output)]) == 0
    content = output.read_bytes()
    assert main([*args, "--output", str(output)]) == 2
    assert output.read_bytes() == content


@pytest.mark.parametrize("identity_matches", [True, False])
def test_typed_terminal_identity_without_duplicate_top_level_request_id(tmp_path, identity_matches):
    from tests.unit.test_product_execution_projection_v2 import (
        _strict_result_with_reused_patch_revision,
    )

    typed = _strict_result_with_reused_patch_revision(tmp_path).to_dict()
    root = tmp_path / "runs" / typed["run_id"]
    manifest = _segment(root)
    result = _read(root, "run_result.json")
    result.pop("request_id")
    result.update(run_id=typed["run_id"], product_execution_result=typed)
    manifest.update(run_id=typed["run_id"], request_id=(
        typed["request_id"] if identity_matches else "different-request"))
    manifest["terminal_record"] = _write(root / "run_result.json", result)
    _write(root / "run_manifest.json", manifest)
    if not identity_matches:
        with pytest.raises(ValueError, match="request_id identity mismatch"):
            _measure(root)
        return
    report = _measure(root)
    terminal = report["last_selected_terminal"]
    assert terminal["product_result_validation"] == "validated"
    assert terminal["product_record_status"] == "verified_terminal_bytes"
    assert terminal["execution_disposition"] == "partial"
    assert terminal["recorded_product_success"] is False
    assert report["completion"] is None


def test_prior_terminal_identity_is_retained_when_its_manifest_omits_identity(tmp_path):
    root, continuation = _fixture(tmp_path)
    manifest = _read(root, "run_manifest.json")
    manifest.pop("request_id")
    _write(root / "run_manifest.json", manifest)
    other_result = _read(continuation, "run_result.json")
    other_result["request_id"] = "different-request"
    other_manifest = _read(continuation, "run_manifest.json")
    other_manifest["request_id"] = "different-request"
    other_manifest["terminal_record"] = _write(continuation / "run_result.json", other_result)
    _write(continuation / "run_manifest.json", other_manifest)
    with pytest.raises(ValueError, match="request_id identity mismatch"):
        _measure(root, continuation)


def test_continuation_terminal_hash_is_checked_even_without_request_identity(tmp_path):
    root, continuation = _fixture(tmp_path)
    manifest = _read(continuation, "run_manifest.json")
    manifest.pop("request_id")
    _write(continuation / "run_manifest.json", manifest)
    tampered = _read(continuation, "run_result.json")
    tampered["blockers"] = [{"code": "UNBOUND_CHANGED_CONTENT"}]
    _write(continuation / "run_result.json", tampered)
    with pytest.raises(ValueError, match="Terminal result hash mismatch"):
        _measure(root, continuation)


def _recovery(root, *, trace="3" * 32, observations=None):
    manifest = _read(root, "run_manifest.json")
    rows = observations if observations is not None else manifest["execution_observations"]
    if observations is None:
        for row in rows.values():
            row["trace_id"] = trace
    checkpoint = save_v2_artifact(root / "a", "session-checkpoint",
        {"schema": "task-agent-session-checkpoint.v2", "run_id": "run-one"})
    document = {"schema_version": "task_agent_recovery_execution_v1",
        "run_id": "run-one", "request_id": "request-one", "status": "failed",
        "resumed_from_checkpoint_ref": checkpoint, "v2_checkpoint_ref": checkpoint,
        "native_execution_ref": None, "execution_trace": {"trace_id": trace},
        "execution_observations": rows,
        "model_calls_performed": sum(row.get("kind") == "generation" for row in rows.values()),
        "model_call_budget": {"consumed": 999}}
    return _write(root / "recoveries/execution.json", document)


def _measure_recovery(root, *refs):
    return measure_product_task(allowed_root=root.parent, run_root=root, run_id=root.name,
                                execution_refs=refs)


def test_recovery_costs_merge_without_terminal_success_or_cumulative_budget(tmp_path):
    root, _ = _fixture(tmp_path)
    ref = _recovery(root)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    report = _measure_recovery(root, ref, ref)
    assert report["model"]["generation_records"] == 2
    assert report["model"]["usage"]["input_tokens"]["total"] == 20
    assert report["tools"]["observed_calls"] == 2
    assert report["selected_execution_refs"] == [ref]
    assert report["duplicate_selections_ignored"] == 1
    assert report["segments"][1]["manifest_ref"] is None
    assert report["segments"][1]["execution_ref"] == ref
    assert report["last_selected_terminal"]["recorded_product_success"] is None
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_recovery_duplicate_observations_deduplicate_and_changed_facts_reject(tmp_path):
    root, _ = _fixture(tmp_path)
    rows = _read(root, "run_manifest.json")["execution_observations"]
    ref = _recovery(root, trace="1" * 32, observations=rows)
    report = _measure_recovery(root, ref)
    assert report["model"]["generation_records"] == 1
    assert report["duplicate_observations_ignored"] == 3
    rows["generation"]["attributes"]["usage"]["input_tokens"] = 999
    changed = _recovery(root, trace="1" * 32, observations=rows)
    with pytest.raises(ValueError, match="conflicting recorded facts"):
        _measure_recovery(root, changed)


def test_failed_recovery_orphan_transport_preserves_unknown_tail(tmp_path):
    root, _ = _fixture(tmp_path)
    rows = {"tail": {"id": "tail", "name": "model.request", "kind": "span",
        "trace_id": "3" * 32, "attributes": {"transport_started": True}}}
    ref = _recovery(root, observations=rows)
    report = _measure_recovery(root, ref)
    assert report["model"]["usage"]["input_tokens"] == {
        "total": None, "known_subtotal": 10, "observed_calls": 1}
    assert len(report["model"]["request_records_without_generation"]) == 1
    assert report["model"]["usage"]["generation_elapsed_ms_sum"] is None


@pytest.mark.parametrize("damage", ["run", "request", "hash", "outside", "parent", "trace",
    "row_key", "attributes", "details", "detail_hash", "detail_outside", "count", "checkpoint",
    "checkpoint_outside", "checkpoint_dict", "checkpoint_missing"])
def test_recovery_invalid_selected_evidence_rejected(tmp_path, damage):
    root, _ = _fixture(tmp_path)
    ref = _recovery(root)
    document = json.loads(Path(ref["path"]).read_text())
    rows = document["execution_observations"]
    if damage in {"run", "request"}:
        document[f"{damage}_id"] = "different"
    elif damage == "parent":
        del rows["request"]
    elif damage == "trace":
        rows["generation"]["trace_id"] = "other-trace"
    elif damage == "row_key":
        rows["generation"]["id"] = "other-id"
    elif damage == "attributes":
        rows["generation"]["attributes"] = []
    elif damage == "details":
        rows["generation"]["attributes"]["local_detail_refs"] = []
    elif damage == "detail_hash":
        rows["generation"]["attributes"]["local_detail_refs"]["request"]["sha256"] = "0" * 64
    elif damage == "detail_outside":
        rows["generation"]["attributes"]["local_detail_refs"]["request"]["path"] = str(tmp_path / "outside")
    elif damage == "count":
        document["model_calls_performed"] = 999
    elif damage == "checkpoint":
        checkpoint_path = root / "a" / document["v2_checkpoint_ref"]
        checkpoint_path.write_text('{"schema":"tampered"}', encoding="utf-8")
    elif damage == "checkpoint_outside":
        document["resumed_from_checkpoint_ref"] = "../outside.json"
    elif damage == "checkpoint_dict":
        document["resumed_from_checkpoint_ref"] = {"path": "fake.json", "sha256": "0" * 64}
    elif damage == "checkpoint_missing":
        document["resumed_from_checkpoint_ref"] = None
    ref = _write(Path(ref["path"]), document)
    if damage == "hash":
        ref["sha256"] = "0" * 64
    elif damage == "outside":
        ref = _write(tmp_path / "outside.json", document)
    with pytest.raises(ValueError):
        _measure_recovery(root, ref)
