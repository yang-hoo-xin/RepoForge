"""Focused shared candidate checks; no provider, real compiler, or game process.

为什么要有它：共同后端须保持旧候选核验边界，不能以不同执行/权限条件制造A与B收益差异。
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from src.mod_agent.candidate_execution import (
    CandidateExecutionError,
    invoke_candidate_evaluator,
    materialize_candidate,
    materialized_code_identity,
    validate_candidate_report,
    verification_identity,
)
from src.mod_agent.execution.patch_apply import PatchApplyError, materialize_exact_text_patch
from tests.unit.test_patch_apply import _plan, _proposal


def _persist(tmp_path, report):
    path = tmp_path / "report.json"
    raw = json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _report(tmp_path, **updates):
    report = {
        "status": "behavior_failed", "failure_owner": "patch_implementation",
        "owner_binding_ids": ["edit_binding_001"],
        "observations": [{"check_id": "synthetic", "expected": 1, "observed": 0}],
        "generated_changes_sha256": "a" * 64, "source_snapshot_sha256": "b" * 64,
        "execution_origin": "executed",
    }
    report.update(updates)
    return {**report, "report_ref": _persist(tmp_path, report)}


def _validate(report):
    return validate_candidate_report(report, changes_sha256="a" * 64,
        source_snapshot_sha256="b" * 64, allowed_binding_ids=("edit_binding_001",))


def _context():
    return {"generated_changes": [{"edit_binding_id": "edit_binding_001"}],
        "generated_changes_sha256": "a" * 64, "source_snapshot_sha256": "b" * 64,
        "model_response": {"output_sha256": "c" * 64}, "evaluation_index": 1}


def test_materializer_matches_exact_owner_bytes_without_writing(short_workspace):
    source = short_workspace / "Patches" / "Behavior.cs"
    source.parent.mkdir()
    before = b"\xef\xbb\xbfvalue = Color.Red;\r\n"
    source.write_bytes(before)
    plan = _plan()
    plan["binding_registry"]["edit_bindings"][0]["source_hash"] = hashlib.sha256(before).hexdigest()
    plan["binding_registry"]["edit_bindings"][0]["target_span"] = {"start_line": 1, "end_line": 1}
    changes = _proposal()["diff_intent"]
    actual = materialize_candidate(workspace_root=short_workspace, implementation_plan=plan, changes=changes)
    expected = materialize_exact_text_patch(workspace_root=short_workspace,
        implementation_plan=plan, diff_intent=changes)
    assert actual == expected
    assert actual.files["Patches/Behavior.cs"] == before.replace(b"Color.Red", b"Color.Pink")
    assert source.read_bytes() == before
    assert materialized_code_identity(actual) == {
        "baseline_sha256": expected.baseline_sha256, "code_sha256": expected.code_sha256,
        "files": ["Patches/Behavior.cs"],
    }


@pytest.mark.parametrize("fault", ["stale_hash", "outside_span", "outside_workspace"])
def test_materialization_keeps_existing_exact_source_rejections(short_workspace, fault):
    source = short_workspace / "Patches" / "Behavior.cs"
    source.parent.mkdir()
    before = b"value = Color.Red;\nother = true;\n"
    source.write_bytes(before)
    plan = _plan()
    binding = plan["binding_registry"]["edit_bindings"][0]
    binding["source_hash"] = hashlib.sha256(before).hexdigest()
    if fault == "stale_hash":
        binding["source_hash"] = "0" * 64
    elif fault == "outside_span":
        binding["target_span"] = {"start_line": 2, "end_line": 2}
    else:
        binding["canonical_target_path"] = "../Behavior.cs"
    with pytest.raises(PatchApplyError):
        materialize_candidate(workspace_root=short_workspace, implementation_plan=plan,
                              changes=_proposal()["diff_intent"])
    assert source.read_bytes() == before


def test_optional_verifier_identity_is_detached_and_not_invented():
    assert verification_identity(lambda context: context) is None
    value = {"validator_sha256": "a" * 64, "inputs_sha256": "b" * 64,
        "environment_sha256": "c" * 64, "deterministic_reuse_allowed": False,
        "details": {"source": "synthetic"}}

    class Evaluator:
        def verification_identity(self):
            return value

    actual = verification_identity(Evaluator())
    assert actual == value
    actual["details"]["source"] = "changed outside"
    assert value["details"]["source"] == "synthetic"


@pytest.mark.parametrize("fault", ["missing_hash", "not_mapping", "boolean_integer"])
def test_invalid_verifier_identity_stays_rejected(fault):
    value = {"validator_sha256": "a" * 64, "inputs_sha256": "b" * 64,
        "environment_sha256": "c" * 64, "deterministic_reuse_allowed": False}
    if fault == "missing_hash":
        del value["inputs_sha256"]
    elif fault == "boolean_integer":
        value["deterministic_reuse_allowed"] = 1
    else:
        value = []

    class Evaluator:
        def verification_identity(self):
            return value

    with pytest.raises(CandidateExecutionError) as rejected:
        verification_identity(Evaluator())
    assert rejected.value.code == "PATCH_VERIFICATION_IDENTITY_INVALID"
    assert rejected.value.automatic_retry_allowed is False


def test_evaluator_gets_detached_context_and_returns_detached_report():
    context = _context()
    original = deepcopy(context)
    returned = {"observations": [{"value": [1]}]}
    calls = []

    def evaluator(value):
        calls.append(value)
        value["generated_changes"].clear()
        return returned

    report = invoke_candidate_evaluator(evaluator, context)
    assert len(calls) == 1 and context == original
    report["observations"][0]["value"].append(2)
    assert returned["observations"][0]["value"] == [1]


@pytest.mark.parametrize("fault", ["raises_after_effect", "invalid_output", "invalid_input"])
def test_evaluator_failure_preserves_dispatch_fact_without_retry(fault):
    calls = []
    context = _context()
    if fault == "invalid_input":
        context["not_json"] = object()

    def evaluator(value):
        calls.append(value)
        if fault == "raises_after_effect":
            raise OSError("synthetic unknown completion")
        return []

    with pytest.raises(CandidateExecutionError) as rejected:
        invoke_candidate_evaluator(evaluator, context)
    failure = rejected.value
    assert failure.code == "PATCH_BEHAVIOR_EVALUATOR_FAILED"
    assert failure.dispatch_started is (fault != "invalid_input")
    assert len(calls) == (0 if fault == "invalid_input" else 1)
    assert failure.automatic_retry_allowed is False
    assert failure.rejected_report["status"] == "infrastructure_failed"
    assert failure.rejected_report["admission"] == "rejected"
    assert failure.rejected_report["failure_owner"] == "unknown"
    assert failure.rejected_report["generated_changes_sha256"] == "a" * 64


@pytest.mark.parametrize("status,owner,origin", [
    ("behavior_failed", "patch_implementation", "executed"),
    ("needs_evidence", "evidence", "executed"),
    ("infrastructure_failed", "environment", "incomplete"),
    ("identity_failed", "validator", "unknown"),
    ("passed", None, "executed"),
])
def test_report_acceptance_matches_old_bound_status_contract(tmp_path, status, owner, origin):
    raw = _report(tmp_path, status=status, failure_owner=owner, execution_origin=origin)
    before = deepcopy(raw)
    admitted, persisted = _validate(raw)
    assert admitted["admission"] == "accepted" and admitted["execution_origin"] == origin
    assert raw == before
    assert persisted == {key: value for key, value in before.items() if key != "report_ref"}
    assert "callback_invoked" not in admitted
    assert "whole_task_delivered" not in admitted


def test_legacy_missing_execution_origin_remains_unknown_not_real_execution(tmp_path):
    raw = _report(tmp_path, status="passed", failure_owner=None)
    del raw["execution_origin"]
    raw["report_ref"] = _persist(tmp_path, {key: value for key, value in raw.items() if key != "report_ref"})
    admitted, _ = _validate(raw)
    assert admitted["execution_origin"] == "unknown"
    assert "real_build_performed" not in admitted


@pytest.mark.parametrize("fault", [
    "wrong_candidate", "wrong_snapshot", "outside_owner", "missing_ref", "wrong_file_hash",
    "summary_disagrees", "empty_failure", "passed_with_owner", "passed_incomplete",
    "forged_reuse", "origin_disagrees", "bad_failure_observations", "bad_owner_type",
])
def test_report_binding_and_execution_rejections_preserve_raw_evidence(tmp_path, fault):
    raw = _report(tmp_path)
    if fault == "wrong_candidate":
        raw["generated_changes_sha256"] = "d" * 64
    elif fault == "wrong_snapshot":
        raw["source_snapshot_sha256"] = "d" * 64
    elif fault == "outside_owner":
        raw["owner_binding_ids"] = ["unselected_binding"]
    elif fault == "missing_ref":
        del raw["report_ref"]
    elif fault == "wrong_file_hash":
        raw["report_ref"]["sha256"] = "d" * 64
    elif fault == "summary_disagrees":
        raw["observations"][0]["observed"] = 1
    elif fault == "origin_disagrees":
        raw["execution_origin"] = "unknown"
    else:
        if fault == "empty_failure":
            raw["observations"] = []
        elif fault == "passed_with_owner":
            raw["status"] = "passed"
        elif fault == "passed_incomplete":
            raw.update(status="passed", failure_owner=None, execution_origin="incomplete")
        elif fault == "forged_reuse":
            raw["execution_origin"] = "reused"
        elif fault == "bad_failure_observations":
            raw["failure_state_observations"] = {}
        elif fault == "bad_owner_type":
            raw["owner_binding_ids"] = [True]
        raw["report_ref"] = _persist(tmp_path, {key: value for key, value in raw.items() if key != "report_ref"})
    original = deepcopy(raw)
    with pytest.raises(CandidateExecutionError) as rejected:
        _validate(raw)
    failure = rejected.value
    assert failure.code == "PATCH_BEHAVIOR_OBSERVATION_INVALID"
    assert failure.automatic_retry_allowed is False
    assert failure.rejected_report["admission"] == "rejected"
    assert failure.rejected_report.get("report_ref") == original.get("report_ref")
    assert raw == original
