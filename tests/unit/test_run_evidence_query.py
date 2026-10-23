"""Retained fixture evidence queried through the normal read-only command."""
import hashlib
import json

import pytest

from scripts import dsh_run_request as bridge
from tests.unit.test_task_agent_v2_admission import session as session
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _fixture(tmp_path, *, status="delivered", native=None):
    workspace = tmp_path / "runs"
    root = workspace / "one"
    native = native or {"schema_version": "task-agent-product-execution.v2", "run_id": "one",
        "status": "candidate_checked", "provider_turns": [], "completed_tool_records": []}
    native["loop_result_ref"] = _save(root / "loop.json", {"call_records": {}})
    native_ref = _save(root / "native" / "execution.json", native)
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": status, "execution_ref": native_ref})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    config = tmp_path / "config.json"
    _save(config, {"arguments": ["--workspace-root", str(workspace)]})
    return config, root


def _query(config, capsys, run_id="one"):
    assert bridge.main(["--execution-config", str(config), "--inspect-run", run_id,
        "--view", "evidence", "--format", "json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_normal_delivery_readonly_and_no_dispatch(tmp_path, capsys, monkeypatch):
    config, root = _fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(bridge.cli, "main", lambda *a: pytest.fail("execution attempted"))
    monkeypatch.setattr(bridge, "advance_task", lambda **k: pytest.fail("recovery attempted"))
    report = _query(config, capsys)
    assert report["terminal"]["status"] == "delivered"
    assert report["query_effects"] == {"model_calls": 0, "builds": 0, "tests": 0, "recoveries": 0, "writes": 0}
    assert len(report["executions"]) == 1
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["missing", "tampered", "outside", "oversize"])
def test_untrustworthy_references_never_claim_complete(tmp_path, capsys, kind):
    config, root = _fixture(tmp_path)
    path = root / "detail.json"
    ref = _save(path, {"private_payload": "never deliver this as verified"})
    if kind == "missing":
        ref["path"] = str(root / "absent.json")
    elif kind == "tampered":
        path.write_text("{}", encoding="utf-8")
    elif kind == "outside":
        ref = _save(tmp_path / "outside.json", {"marker": "FORBIDDEN CONTENT"})
    else:
        path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
        ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((root / "run_manifest.json").read_text())
    _save(root / "run_manifest.json", {**manifest, "detail_ref": ref})
    report = _query(config, capsys)
    assert report["complete"] is False
    statuses = {row["status"] for row in report["issues"]}
    assert {"missing": "missing", "tampered": "hash_mismatch", "outside": "forbidden", "oversize": "truncated"}[kind] in statuses
    assert "FORBIDDEN CONTENT" not in json.dumps(report)
    if kind == "oversize":
        assert "truncated" in statuses
    if kind == "outside":
        assert "forbidden" in statuses


def test_duplicate_business_id_is_ambiguous_and_bound_config_conflicts(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    _save(root.parent / "copy" / "run_manifest.json", {"run_id": "one"})
    report = _query(config, capsys)
    assert report["status"] == "ambiguous" and not report["complete"]
    _save(config, {"arguments": ["--workspace-root", str(root.parent), "--run-id", "other"]})
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence"]) == 2
    assert "IDENTITY_CONFLICT" in json.loads(capsys.readouterr().out)["reason"]


def test_distinct_request_reservations_count_but_native_copies_do_not(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    turns = []
    for number in (1, 2):
        turns.append({"turn": number, "reservation_id": f"reserve-{number}",
            "request_ref": _save(root / f"request-{number}.json", {"same": "input"}),
            "response_ref": _save(root / f"response-{number}.json", {"usage": {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 3}})})
    native = {"schema_version": "task-agent-product-execution.v2", "run_id": "one",
        "provider_turns": turns, "completed_tool_records": [], "loop_result_ref": _save(root / "loop.json", {"call_records": {}})}
    ref = _save(root / "native" / "execution.json", native)
    copy = _save(root / "snapshot" / "execution.json", native)
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": "unknown", "execution_ref": ref, "snapshot_ref": copy})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    report = _query(config, capsys)
    assert len(report["executions"]) == 2
    assert report["cost"]["requests"] == 2
    assert report["cost"]["input_tokens"]["known_sum"] == 20
    assert report["cost"]["cached_input_tokens"]["known_sum"] == 6
    assert not report["cost"]["whole_task_cost_complete"]


def test_relative_source_identity_is_not_a_guessed_record_path(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    _save(root / "run_manifest.json", {"run_id": "one", "source_inputs": [{"path": "src/Real.cs", "sha256": "a" * 64}]})
    report = _query(config, capsys)
    assert any(link["status"] == "identity_only" and link["target"] == "src/Real.cs" for link in report["links"])
    assert not any(row["path"].endswith("Real.cs") for row in report["records"])


def test_postcheck_unknown_preserves_fact_without_becoming_delivery(tmp_path, capsys):
    config, root = _fixture(tmp_path, status="unknown")
    fact = _save(root / "checks" / "test-fact.json", {"schema": "csharp_repository_stage_fact_v1", "stage": "test_execution",
        "context": {"candidate_ref": "candidate-A", "candidate_source_id": "source-A"}, "execution": {"exit_code": 1}})
    _save(root / "checks" / "integrity-error.json", {"schema": "csharp_repository_integrity_v1", "status": "unknown", "stage_receipts": [fact]})
    report = _query(config, capsys)
    assert report["terminal"]["status"] == "unknown"
    assert any(row["stage"] == "test_execution" and row["execution"]["exit_code"] == 1 for row in report["stage_facts"])
    assert all(row["claim"] == "recorded_execution_fact_not_acceptance" for row in report["stage_facts"])


def test_separate_diagnostic_and_checkpoint_not_merged_as_model_turns(tmp_path, capsys):
    config, root = _fixture(tmp_path, status="unknown")
    checkpoint = _save(root / "session-checkpoints" / "checkpoint.json", {"schema": "task-agent-checkpoint.v2"})
    _save(root / "diagnostic" / "execution.json", {"schema_version": "task-agent-product-execution.v2", "run_id": "diagnostic_other",
        "provider_turns": [], "resumed_from_checkpoint_ref": checkpoint})
    report = _query(config, capsys)
    assert len(report["executions"]) == 1
    assert any(row["kind"] == "separate_run_not_merged" and row["run_id"] == "diagnostic_other" for row in report["lineage"])
    assert any(row.get("claim") == "checkpoint_presence_does_not_prove_resume" for row in report["lineage"])


def test_product_tool_catalog_exposes_only_scoped_public_feedback(session):
    names = {action.definition.name for action in session.actions()}
    assert "check_feedback_read" in names
    assert not names & {"query_run_evidence", "inspect_evidence", "read_record", "inspect_run"}


def test_explicit_result_resume_reference_is_verified_not_new_success(tmp_path, capsys):
    config, root = _fixture(tmp_path, status="unknown")
    checkpoint = _save(root / "session-checkpoints" / "checkpoint.json", {"schema": "task-agent-checkpoint.v2"})
    result = json.loads((root / "run_result.json").read_text())
    terminal = _save(root / "run_result.json", {**result, "resumed_from_checkpoint_ref": checkpoint})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    report = _query(config, capsys)
    recorded = next(row for row in report["lineage"] if row["kind"] == "resumed_from_checkpoint_ref")
    assert recorded["status"] == "reference_verified"
    assert recorded["claim"] == "explicit_recorded_lineage_not_new_execution_or_success"
    assert report["terminal"]["status"] == "unknown" and report["cost"]["requests"] == 0


def test_duplicate_response_alias_is_ambiguous_not_guessed_turn(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    call = {"call_id": "call-1", "name": "plan_update", "arguments": {"plan": {}}}
    response = {"tool_calls": [call]}
    turns = [{"turn": number, "reservation_id": f"r-{number}",
        "request_ref": _save(root / f"request-{number}.json", {"input": "same"}),
        "response_ref": _save(root / f"response-{number}.json", response)} for number in (1, 2)]
    native = {"schema_version": "task-agent-product-execution.v2", "run_id": "one", "provider_turns": turns,
        "loop_result_ref": _save(root / "loop.json", {"call_records": {}}),
        "completed_tool_records": [{**call, "response_ref": _save(root / "origin-response.json", response)}]}
    native_ref = _save(root / "native" / "execution.json", native)
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": "unknown", "execution_ref": native_ref})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    report = _query(config, capsys)
    assert report["executions"][0]["actions"][0]["turn"] is None
    assert any(row["status"] == "ambiguous" and row.get("call_id") == "call-1" for row in report["issues"])
