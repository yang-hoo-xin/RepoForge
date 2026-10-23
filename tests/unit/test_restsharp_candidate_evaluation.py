"""Final-evaluation admission guards; no real candidate or dotnet execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[2] / "artifacts/eval_driven_20261022/evaluate_restsharp_v2_candidate.py"
    spec = importlib.util.spec_from_file_location("final_restsharp_eval", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("changes,reason", [
    ({"candidate_ref": "different"}, "FINAL_CANDIDATE_MISMATCH"),
    ({"unknown_effects": True}, "KNOWN_TERMINAL_EXECUTION_REQUIRED"),
    ({"status": "running"}, "KNOWN_TERMINAL_EXECUTION_REQUIRED"),
    ({"source_after": {"changed": True}}, "NATIVE_SOURCE_CHANGED"),
])
def test_unproven_final_native_rejected(module, tmp_path, changes, reason):
    native = {"schema_version": "task-agent-product-execution.v2", "status": "blocked", "unknown_effects": False,
        "candidate_ref": "final", "source_before": {"source": "hash"}, "source_after": {"source": "hash"}, **changes}
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(native), encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        module.validate_native(path, "final")


def fixture_state(tmp_path, *, name="src/RestSharp/A.cs", before="old", snapshot="old", allowed=True):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old", encoding="utf-8")
    state = SimpleNamespace(contract={"baseline_manifest_ref": "baseline", "allowed_edit_paths": [name] if allowed else []},
        load=lambda _: {"files": {name: {"text": snapshot}}},
        changes_from_baseline=lambda _: [{"path": name, "before_text": before, "after_text": "new",
            "before_sha256": "before", "after_sha256": "after", "binding_id": "binding"}])
    return state, path


@pytest.mark.parametrize("kwargs,reason", [
    ({"snapshot": "other"}, "BASELINE_SOURCE_IDENTITY_MISMATCH"),
    ({"before": "other"}, "CANDIDATE_PREIMAGE_MISMATCH"),
    ({"allowed": False}, "OUTSIDE_QUALIFIED_EDIT_SCOPE"),
    ({"name": "src/RestSharp/Project.csproj"}, "OUTSIDE_QUALIFIED_EDIT_SCOPE"),
])
def test_exact_production_edit_guards(module, tmp_path, monkeypatch, kwargs, reason):
    state, path = fixture_state(tmp_path, **kwargs)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: path.relative_to(tmp_path).as_posix().encode() + b"\0")
    with pytest.raises(ValueError, match=reason):
        module.apply_changes(tmp_path, state, "candidate")
    assert path.read_text() == "old"


def test_matching_authorized_diff_applied_without_other_files(module, tmp_path, monkeypatch):
    state, path = fixture_state(tmp_path)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: b"src/RestSharp/A.cs\0")
    assert module.apply_changes(tmp_path, state, "candidate")[0]["path"] == "src/RestSharp/A.cs"
    assert path.read_text() == "new"


@pytest.mark.parametrize("changes", [{"status": "running"}, {"timeout": True}, {"owned_process_tree_cleanup": "unknown"}])
def test_diagnostic_requires_closed_owned_process(module, tmp_path, changes):
    path = tmp_path / "process.json"
    path.write_text(json.dumps({"status": "exited", "started": True, "timeout": False, "exit_code": 1,
        "finished_at": "2026-10-22T12:47:40Z", "process_tree_ownership": "Windows Job Object",
        "owned_process_tree_cleanup": "job_closed_kill_on_close", **changes}))
    with pytest.raises(ValueError, match="DIAGNOSTIC_PROCESS_NOT_CLOSED"):
        module.validate_diagnostic_process({}, {}, path)


def test_diagnostic_without_process_and_known_execution_cannot_bypass(module, tmp_path):
    with pytest.raises(ValueError, match="DIAGNOSTIC_PROCESS_RECEIPT_REQUIRED"):
        module.validate_diagnostic_process({}, {}, None)
    path = tmp_path / "execution.json"
    path.write_text(json.dumps({"schema_version": "task-agent-product-execution.v2", "status": "blocked", "unknown_effects": False}))
    with pytest.raises(ValueError, match="DIAGNOSTIC_UNKNOWN_CHECK_REQUIRED"):
        module.validate_native(path, "candidate", diagnostic=True)


def test_target_failures_are_not_reported_as_preservation_regressions(module):
    tested = {"fail_to_pass": {"missing_count": 0, "outcome_counts": {"Passed": 0, "Failed": 3, "Other": 0}},
        "pass_to_pass": {"missing_count": 0, "outcome_counts": {"Passed": 6, "Failed": 0, "Other": 0}},
        "public_reproduction": {"count": 2, "passed": 0}, "all_outcomes": {"Passed": 215, "Failed": 5, "NotExecuted": 0},
        "processes": [{"exit_code": 1}, {"exit_code": 0}], "total_results": 220}
    result = module.aggregate_test_result(tested)
    assert result["status"] == "failed" and result["full_selected_suite_passed"] is False
    assert result["preservation_suite_passed"] is True
