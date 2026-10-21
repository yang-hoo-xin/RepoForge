"""Shared failure feedback keeps observations, but not host execution addresses."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.mod_agent.evidence import RgLiveEvidenceProvider
from src.mod_agent.local_entry import (
    LocalEntryError,
    _assert_no_absolute_local_paths,
    _make_request_upstream_evidence_factory,
)
from src.mod_agent.proposal.behavior_feedback_view import project_execution_reference_paths
from src.mod_agent.repair.upstream import run_patch_upstream_replanning
from tests.unit.test_model_planning_selection import _Client, _selector
from tests.unit.test_patch_repair_continuation import _file_ref, _repair_case
from tests.unit.test_patch_upstream_replanning import _artifact, _selection


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No real model or network is allowed")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


@pytest.mark.parametrize("path", [r"C:\private\execution.json", "/private/execution.json", r"\\host\share\execution.json"])
def test_nested_execution_reference_projection_preserves_every_fact_and_hash(path):
    ref = {"path": path, "sha256": "a" * 64, "format": "original-json"}
    original = {"observations": [{"check_id": "client_repeated_packet_once",
        "input": {"execution_ref": ref, "packet": {"session": 1, "sequence": 2}},
        "observed": {"actual": 3, "expected": 1, "passed": False},
        "expected": {"all_controlled_checks": True, "whole_task_delivered": False}, "result_ref": ref}],
        "verification_ref": ref, "failure_report_ref": ref}
    before = deepcopy(original)
    model = project_execution_reference_paths(original)
    _assert_no_absolute_local_paths(model)
    assert original == before
    expected = deepcopy(original)
    for reference in (expected["observations"][0]["input"]["execution_ref"], expected["observations"][0]["result_ref"],
            expected["verification_ref"], expected["failure_report_ref"]):
        reference.pop("path", None)
    assert model == expected
    assert model["observations"][0]["observed"] == {"actual": 3, "expected": 1, "passed": False}


@pytest.mark.parametrize("value", [
    {"observed": {"path": "C:/private/business-input.txt"}},
    {"unknown_ref": {"path": "C:/private/execution.json", "sha256": "a" * 64}},
    {"execution_ref": {"path": "C:/private/execution.json"}},
    {"execution_ref": {"path": "C:/private/execution.json", "sha256": "not-an-identity"}},
    {"execution_ref": {"path": "C:/private/execution.json", "sha256": "a" * 64,
        "unexpected_private_path": "C:/other/private.txt"}},
])
def test_projection_does_not_disable_the_original_guard_for_unknown_or_unbound_paths(value):
    with pytest.raises(LocalEntryError, match="absolute local path"):
        _assert_no_absolute_local_paths(project_execution_reference_paths(value))


def test_relative_business_paths_and_non_reference_data_are_not_rewritten():
    value = {"execution_ref": {"path": "testdata/input.json", "sha256": "a" * 64},
        "observed": {"expected_path": "source/Thing.cs", "reason": "C:/private/execution.json was not found\n"}}
    assert project_execution_reference_paths(value) == value
    assert project_execution_reference_paths(None) is None


def _nested_failure(tmp_path):
    inputs, request, parent = _repair_case(tmp_path)
    path = Path(request.repair_failure_report_ref["path"])
    report = json.loads(path.read_bytes())
    execution_path = tmp_path / "controlled-execution.json"
    execution_path.write_text('{"actual":3,"expected":1}', encoding="utf-8")
    reference = _file_ref(execution_path)
    report["observations"][0]["input"]["execution_ref"] = reference
    report["observations"][0]["result_ref"] = reference
    path.write_text(json.dumps(report), encoding="utf-8")
    return inputs, replace(request, repair_failure_report_ref=_file_ref(path)), parent, report


def test_real_graph_planning_projects_nested_failure_in_both_assessment_and_json_feedback(tmp_path):
    inputs, request, _parent, original = _nested_failure(tmp_path)
    before = Path(request.repair_failure_report_ref["path"]).read_bytes()
    client = _Client()
    state = run_patch_upstream_replanning(request=request, selection=_selection(),
        planning_target_selection_factory=_selector(inputs, client))
    assert not state["blockers"] and len(client.calls) == 1
    payload = client.calls[0].messages[1].content
    _assert_no_absolute_local_paths(payload)
    observed = payload["upstream_assessment"]["failure_report"]["observations"]
    assert observed == project_execution_reference_paths(original["observations"])
    feedback = [json.loads(value) for value in payload["semantic_feedback"]
        if value.startswith("{") and "recommendation_is_not_observation" in value]
    assert len(feedback) == 1
    _assert_no_absolute_local_paths(feedback[0])
    assert feedback[0]["observed_failure"] == observed
    assert _artifact(inputs, state, "repair_plan")["data"]["failure_findings"] == original["observations"]
    assert Path(request.repair_failure_report_ref["path"]).read_bytes() == before


def test_shared_product_evidence_factory_reaches_original_guard_with_all_observations(tmp_path):
    inputs, request, _parent, original = _nested_failure(tmp_path)
    captured = []

    class StopBeforeProviderError(RuntimeError):
        pass

    def complete(role, payload, schema):
        _assert_no_absolute_local_paths(payload)
        captured.append((role, deepcopy(payload), schema))
        raise StopBeforeProviderError("No response or provider send is authorized by this test")

    factory = _make_request_upstream_evidence_factory(artifact_store=inputs.store,
        workspace_root=inputs.workspace, source_provider=RgLiveEvidenceProvider(inputs.workspace),
        upstream_search_root="generated_mods/SyntheticMod", complete_requirement_coverage_json=complete)
    with pytest.raises(StopBeforeProviderError):
        run_patch_upstream_replanning(request=request, selection=_selection("evidence"),
            planning_target_selection_factory=_selector(inputs, _Client()), evidence_pack_factory=factory)
    assert len(captured) == 1 and captured[0][0] == "upstream_evidence_queries"
    assert captured[0][1]["failure_feedback"]["observed_failure"] == project_execution_reference_paths(original["observations"])
    assert json.loads(Path(request.repair_failure_report_ref["path"]).read_bytes()) == original
