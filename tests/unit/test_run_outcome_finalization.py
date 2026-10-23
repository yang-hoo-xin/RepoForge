"""Automatic summaries through original finalization and controlled public entry.

为什么要有它：通路通过不代表模型自主成功，摘要故障也不能抹掉产品失败。
"""
import hashlib
import json
from pathlib import Path

import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.run_outcome_finalization import finalize_continuation, finalize_run_outcome
from src.observability.dashboard.services import run_evidence
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _read(ref):
    raw = Path(ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
    return json.loads(raw)


def _run(tmp_path):
    root = tmp_path / "run"
    _save(root / "run_manifest.json", {"run_id": "one", "request_id": "request-one"})
    (root / "a").mkdir()
    return root


def test_normal_terminal_and_repeated_projection_do_not_feed_summary_back(tmp_path):
    root = _run(tmp_path)
    result = local_entry._persist_graph_terminal_result(run_root=root, run_id="one", request_id="request-one",
        summary={"run_id": "one", "current_stage": "blocked", "whole_task_delivered": False},
        model_call_count=0, patch_payload_parser_entered=False, actual_write_performed=False)
    assert result["outcome_summary_status"] == "available"
    first = _read(result["outcome_summary_ref"])
    manifest = json.loads((root / "run_manifest.json").read_bytes())
    terminal = _read(manifest["terminal_record"])
    assert "outcome_summary_ref" not in terminal
    again = finalize_run_outcome(run_root=root, run_id="one")
    assert _read(again["outcome_summary_ref"])["outcome"] == first["outcome"]
    report = run_evidence.query_run_evidence(allowed_roots=[root], run_id="one")
    assert not any("run-outcomes" in row.get("path", "") for row in report["records"])


@pytest.mark.parametrize("stage", ["pre_graph", "graph"])
def test_failure_paths_commit_original_failure_and_automatic_outcome(tmp_path, stage):
    root = _run(tmp_path)
    common = dict(run_root=root, run_id="one", request_id="request-one", error=ValueError("original failure"))
    if stage == "pre_graph":
        local_entry._persist_pre_graph_failure(**common, failure_stage="admission",
            source_revision_binding=None, model_admission={})
    else:
        local_entry._persist_graph_terminal_failure(**common, model_call_count=2)
    manifest = json.loads((root / "run_manifest.json").read_bytes())
    assert manifest["status"] == "failed"
    assert _read(manifest["terminal_record"])["error"] == "original failure"
    assert _read(manifest["outcome_summary_ref"])["run_id"] == "one"


def test_summary_exception_does_not_change_product_result(tmp_path, monkeypatch):
    root = _run(tmp_path)
    def broken(**kwargs):
        raise RuntimeError("controlled summary failure")
    monkeypatch.setattr(run_evidence, "query_run_evidence", broken)
    result = local_entry._persist_graph_terminal_result(run_root=root, run_id="one", request_id="request-one",
        summary={"run_id": "one", "current_stage": "blocked", "actual_write_performed": False},
        model_call_count=3, patch_payload_parser_entered=False, actual_write_performed=False)
    assert result == {"outcome_summary_status": "summary_unavailable", "outcome_summary_error": "RuntimeError"}
    manifest = json.loads((root / "run_manifest.json").read_bytes())
    assert manifest["terminal_stage"] == "blocked" and manifest["model_call_count"] == 3
    assert _read(manifest["terminal_record"])["current_stage"] == "blocked"


def test_recovery_rejection_persists_boundary_without_replacing_terminal(tmp_path, monkeypatch, capsys):
    from src.mod_agent import task_agent_v2_recovery as recovery
    root = _run(tmp_path)
    terminal = _save(root / "run_result.json", {"run_id": "one", "current_stage": "unknown"})
    _save(root / "run_manifest.json", {"run_id": "one", "request_id": "request-one", "terminal_record": terminal})
    checkpoint = _save(root / "a" / "checkpoint.json", {"task_contract_ref": "contract-one"})
    def rejected(**kwargs):
        raise ValueError("DEADLINE_EXPIRED")
    monkeypatch.setattr(recovery, "_resume_v2_task_agent", rejected)
    code = cli.main(["run-request", "--resume-task-agent", checkpoint["path"],
        "--artifact-store-root", str(root / "a")])
    assert code == 2 and json.loads(capsys.readouterr().out)["reason"] == "DEADLINE_EXPIRED"
    manifest = json.loads((root / "run_manifest.json").read_bytes())
    assert manifest["terminal_record"] == terminal
    boundary = _read(manifest["outcome_boundaries"][0])
    assert boundary["status"] == "blocked" and boundary["task_contract_ref"] == "contract-one"
    assert boundary["run_id"] == "one" and boundary["request_id"] == "request-one"
    assert boundary["resumed_from_checkpoint_ref"] == checkpoint
    assert _read(manifest["outcome_summary_ref"])["boundary_refs"] == manifest["outcome_boundaries"]


def test_delivery_receipt_commits_before_summary_and_retains_known_failure(tmp_path):
    from src.mod_agent.task_agent_v2_delivery import _finish
    root = _run(tmp_path)
    returned = _finish(root / "a", root / "a" / "delivery-result.json", {
        "schema": "task-agent-delivery-receipt.v2", "status": "build_failed", "whole_task_delivered": False})
    assert returned["status"] == "build_failed"
    assert returned["receipt_ref"] and returned["outcome_summary_status"] == "available"
    manifest = json.loads((root / "run_manifest.json").read_bytes())
    boundary = _read(manifest["outcome_boundaries"][0])
    assert boundary["result"]["receipt_ref"] == returned["receipt_ref"]
    assert "resource_record" not in boundary["result"]


def test_controlled_normal_cli_automatically_writes_outcome(repository_workspace, monkeypatch, capsys, tmp_path):
    from tests.unit.test_task_agent_v2_product_entry import (
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff,
    )
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(repository_workspace, monkeypatch, capsys, False)
    result = json.loads((repository_workspace / "v2-entry-result.json").read_text(encoding="utf-8"))
    assert result["status"] == "awaiting_delivery"
    assert result["outcome_summary_status"] == "available"
    summary = _read(result["outcome_summary_ref"])
    assert summary["run_id"] == result["run_id"]
    assert summary["outcome"]["costs"]["task_runtime"]["unique_requests"] == 13
    assert summary["outcome"]["costs"]["task_runtime"]["total_tokens"] == 390
    _save(tmp_path / "controlled-cli-artifacts.json", {"run_root": result["run_root"],
        "summary_ref": result["outcome_summary_ref"], "origin": "controlled_model_and_build"})


def test_controlled_recovery_cli_preserves_task_and_deduplicates_calls(repository_workspace, monkeypatch, capsys, tmp_path):
    from tests.unit.test_task_agent_v2_recovery import (
        test_normal_entry_interruption_reconstructs_owner_and_finishes_same_product,
    )
    test_normal_entry_interruption_reconstructs_owner_and_finishes_same_product(
        repository_workspace, monkeypatch, capsys, "tool_receipt")
    manifests = list((repository_workspace / "runs").rglob("run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_bytes())
    assert manifest["outcome_boundaries"]
    summary = _read(manifest["outcome_summary_ref"])
    assert summary["run_id"] == manifest["run_id"]
    assert summary["outcome"]["costs"]["task_runtime"]["unique_requests"] == 13
    assert summary["outcome"]["costs"]["task_runtime"]["total_tokens"] == 390

    _save(tmp_path / "controlled-recovery-artifacts.json", {"run_id": manifest["run_id"],
        "run_root": str(manifests[0].parent), "summary_ref": manifest["outcome_summary_ref"],
        "boundary_refs": manifest["outcome_boundaries"], "origin": "controlled_model_and_build"})


def test_missing_checkpoint_rejection_still_retains_boundary(tmp_path, monkeypatch, capsys):
    from src.mod_agent import task_agent_v2_recovery as recovery
    root = _run(tmp_path)
    def missing(**kwargs):
        raise FileNotFoundError("original missing checkpoint")
    monkeypatch.setattr(recovery, "_resume_v2_task_agent", missing)
    assert cli.main(["run-request", "--resume-task-agent", "missing.json", "--artifact-store-root", str(root / "a")]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "original missing checkpoint"
    boundary = _read(result["outcome_boundary_ref"])
    assert boundary["checkpoint_status"] == "unavailable_or_unverified"
    assert "resumed_from_checkpoint_ref" not in boundary


@pytest.mark.parametrize("operation", ["apply", "build"])
def test_delivery_admission_failure_is_recorded_without_repeating_effect(tmp_path, monkeypatch, operation):
    from src.mod_agent import task_agent_v2_delivery as delivery
    root = _run(tmp_path)
    calls = []
    def rejected(**kwargs):
        calls.append(kwargs)
        raise ValueError("APPROVAL_CHANGED")
    monkeypatch.setattr(delivery, "_" + operation + "_v2_delivery", rejected)
    with pytest.raises(ValueError, match="APPROVAL_CHANGED"):
        if operation == "apply":
            delivery.apply_v2_delivery(artifact_store_root=root / "a", displayed_binding={"checkpoint_ref": "missing.json"})
        else:
            delivery.build_v2_delivery(artifact_store_root=root / "a", apply_ref="missing.json", operation_id="one")
    assert len(calls) == 1
    manifest = json.loads((root / "run_manifest.json").read_bytes())
    boundary = _read(manifest["outcome_boundaries"][0])
    assert boundary["event"] == operation + "_return" and boundary["status"] == "blocked"


def test_continuation_retains_refs_without_copying_cumulative_ledger(tmp_path):
    root = _run(tmp_path)
    native_ref = _save(root / "native" / "execution.json", {"schema_version": "task-agent-product-execution.v2",
        "run_id": "one", "provider_turns": [], "completed_tool_records": []})
    result = finalize_continuation(artifact_store_root=root / "a", event="recovery_return", result={
        "status": "blocked", "run_id": "one", "request_id": "request-one",
        "cumulative_resources": {"model_calls": 99}, "outcome": {"execution_ref": native_ref,
            "budget_after": {"consumed": 99}}})
    boundary = _read(result["outcome_boundary_ref"])
    assert boundary["result"]["native_execution_ref"] == native_ref
    assert "cumulative_resources" not in boundary["result"] and "outcome" not in boundary["result"]


def test_existing_immutable_summary_is_not_overwritten(tmp_path):
    from src.mod_agent.run_outcome_finalization import _immutable
    ref = _immutable(tmp_path, "run-outcomes", {"value": "original"})
    path = Path(ref["path"])
    path.write_bytes(b"existing changed content")
    with pytest.raises(ValueError, match="IMMUTABLE_RECORD_CHANGED"):
        _immutable(tmp_path, "run-outcomes", {"value": "original"})
    assert path.read_bytes() == b"existing changed content"


def test_product_terminal_change_during_summary_is_preserved(tmp_path, monkeypatch):
    root = _run(tmp_path)
    original = _save(root / "first.json", {"status": "unknown"})
    latest = _save(root / "latest.json", {"status": "blocked"})
    _save(root / "run_manifest.json", {"run_id": "one", "request_id": "request-one", "terminal_record": original})
    def concurrent(**kwargs):
        _save(root / "run_manifest.json", {"run_id": "one", "request_id": "request-one", "terminal_record": latest})
        return {"outcome": {"result": "old"}, "complete": True, "status": "available"}
    monkeypatch.setattr(run_evidence, "query_run_evidence", concurrent)
    result = finalize_run_outcome(run_root=root, run_id="one")
    assert result["outcome_summary_status"] == "summary_unavailable"
    assert json.loads((root / "run_manifest.json").read_bytes())["terminal_record"] == latest


def test_reparse_path_is_rejected_on_python310(tmp_path, monkeypatch):
    import stat
    from types import SimpleNamespace

    from src.mod_agent.run_outcome_finalization import _safe_path

    original = Path.lstat
    def reparse(path, *args, **kwargs):
        actual = original(path, *args, **kwargs)
        if path == tmp_path:
            return SimpleNamespace(st_mode=actual.st_mode, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return actual
    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(ValueError, match="OUTCOME_LINK_NOT_ALLOWED"):
        _safe_path(tmp_path / "result.json")


def test_admission_rejection_is_not_an_unknown_dispatched_recovery(tmp_path):
    root = _run(tmp_path)
    result = finalize_continuation(artifact_store_root=root / "a", event="recovery_admission_rejected",
        error=ValueError("configuration override rejected"))
    boundary = _read(result["outcome_boundary_ref"])
    assert boundary["execution_status"] == "not_executed"
    assert boundary["original_terminal_ref"] is None


def test_manifest_path_checked_before_its_first_read(tmp_path, monkeypatch):
    from src.mod_agent import run_outcome_finalization as module

    root = _run(tmp_path)
    target = root / "run_manifest.json"
    check, read = module._safe_path, Path.read_bytes
    def guard(path):
        if Path(path) == target:
            raise ValueError("OUTCOME_LINK_NOT_ALLOWED")
        return check(path)
    def forbidden_read(path):
        if path == target:
            pytest.fail("manifest read before path admission")
        return read(path)
    monkeypatch.setattr(module, "_safe_path", guard)
    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    assert finalize_run_outcome(run_root=root, run_id="one")["outcome_summary_status"] == "summary_unavailable"
