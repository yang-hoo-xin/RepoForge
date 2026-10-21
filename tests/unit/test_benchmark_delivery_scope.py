"""Exact file authority before delegated Apply, with real patch materialization."""
from types import SimpleNamespace

import pytest

from scripts import finalize_benchmark_platform as entry
from src.mod_agent.evaluation import isolated_delivery_adapter as delivery
from tests.unit.test_task_agent_execution import case as case
from tests.unit.test_task_agent_execution import execute


def materialized_scope(case, allowed):
    owner, outcome = execute(case)
    assert outcome.status == "candidate_ready"
    handoff = owner.validate_handoff(outcome, case.context)
    return entry.declared_source_scope(workspace=case.inputs.workspace,
        target=case.inputs.workspace / "generated_mods/SyntheticMod",
        proposal={"diff_intent": list(handoff.changes)}, implementation=case.plan, allowed_paths=allowed)


def test_exact_declared_file_passes_without_writing_source(case):
    before = case.inputs.behavior.read_bytes()
    result = materialized_scope(case, ["Behavior.cs"])
    assert result["status"] == "passed" and result["changed_paths"] == ["Behavior.cs"]
    assert result["checked_before_apply"] is True and result["apply_performed_by_this_check"] is False
    assert case.inputs.behavior.read_bytes() == before


@pytest.mark.parametrize("allowed", [[], ["Plugin.cs"], ["../Behavior.cs"], [True]])
def test_candidate_cannot_expand_explicit_file_authority(case, allowed):
    before = case.inputs.behavior.read_bytes()
    with pytest.raises(ValueError):
        materialized_scope(case, allowed)
    assert case.inputs.behavior.read_bytes() == before


@pytest.mark.parametrize("continuation", [False, True])
@pytest.mark.parametrize("new_records", [False, True])
def test_finalizer_rejects_scope_before_delegated_apply_or_build(tmp_path, monkeypatch, continuation, new_records):
    ready = {"records_root": str(tmp_path), "bundle_ref": {"which": "bundle"},
             "product_snapshot_root": str(tmp_path)}
    dispatch = {"run_root": str(tmp_path), "bound_inputs": [], "extra_load_refs": []}
    bundle = {"expected_check_count": 1, "cases": 1, "source_snapshot": [{"path": "Protected.cs", "sha256": "a"}],
              "allowed_edit_paths": ["Allowed.cs"]}
    report = {"verification_summary": {"checks": 1, "failed_checks": 0, "executions": 1}}
    checkpoint = SimpleNamespace(feature_id="feature", artifact_type="workflow_checkpoint", path="saved", artifact_hash="a" * 64)
    state = {"artifacts": {"implementation_plan": "plan", "patch_proposal": "proposal"},
             "artifact_hashes": {"implementation_plan": "b" * 64, "patch_proposal": "c" * 64}}
    store = SimpleNamespace(semantic_read_immutable=lambda selected: {
        "data": {"state_snapshot": state} if selected is checkpoint else {}})
    prepared = {"verification_report_ref": {"which": "report"}, "manifest": {"source_snapshot": bundle["source_snapshot"]},
                "store": store, "checkpoint": checkpoint, "initial": {"feature_id": "feature"},
                "run_root": tmp_path, "target": tmp_path / "source"}
    reads = iter((ready, dispatch, bundle, report))
    monkeypatch.setattr(entry, "ref", lambda path: {"path": str(path)})
    monkeypatch.setattr(entry, "read_bound_json", lambda _: next(reads))
    selected = {"path": str(tmp_path / "continuation.json"), "sha256": "d" * 64} if continuation else None
    contexts = []
    monkeypatch.setattr(entry, "prepare_delivery", lambda context: contexts.append(context) or prepared)
    def reject(**_kwargs):
        raise ValueError("BENCHMARK_PRE_APPLY_SCOPE_CHANGED")
    monkeypatch.setattr(entry, "declared_source_scope", reject)
    monkeypatch.setattr(entry, "deliver", lambda *_args, **_kwargs: pytest.fail("Apply/Build must not be called"))
    with pytest.raises(ValueError, match="PRE_APPLY_SCOPE_CHANGED"):
        entry.finalize("ready.json", initial_result_ref=selected,
            records_output=tmp_path / "attempt-02" if new_records else None)
    assert contexts[0]["trial"]["delivery"].get("initial_result_ref") == selected
    assert (tmp_path / "attempt-02").exists() is new_records


def test_changed_result_between_precheck_and_delivery_never_reaches_delegation(tmp_path, monkeypatch):
    path = tmp_path / "run_result.json"
    path.write_text('{"checkpoint":"allowed"}', encoding="utf-8")
    checked = entry.ref(path)
    path.write_text('{"checkpoint":"different"}', encoding="utf-8")
    actual = entry.ref(path)
    monkeypatch.setattr(delivery, "prepare_delivery", lambda _: {"initial_ref": actual})
    monkeypatch.setattr(delivery, "prepare_task_delegation",
                        lambda **_: pytest.fail("Changed checkpoint must not receive delegation"))
    result = delivery.deliver({}, expected_initial_result_ref=checked)
    assert result["delivery_complete"] is False
    assert result["reason"] == "DELIVERY_PRECHECKED_INITIAL_RESULT_CHANGED"
    assert result["execution_refs"] == []


def test_finalizer_installs_build_admission_before_reading_or_applying(monkeypatch):
    from src.mod_agent.build import execution_budget
    calls = []
    def install():
        calls.append("install")
        raise execution_budget.BuildBudgetError("BUILD_BUDGET_CONFIGURATION_CHANGED")
    monkeypatch.setattr(execution_budget, "install_build_budget_from_environment", install)
    monkeypatch.setattr(entry, "ref", lambda _: pytest.fail("must stop before input and Apply"))
    monkeypatch.setattr(entry, "deliver", lambda *_args, **_kwargs: pytest.fail("must not deliver"))
    with pytest.raises(execution_budget.BuildBudgetError, match="CONFIGURATION_CHANGED"):
        entry.finalize("not-read.json")
    assert calls == ["install"]


@pytest.mark.parametrize("quota", [None, {"authorization_ref": {"other": "grant"}, "maximum": 8},
    {"authorization_ref": {"same": "grant"}, "maximum": 9}])
def test_development_delivery_rejects_missing_or_different_shared_quota(monkeypatch, quota):
    from src.mod_agent.build import execution_budget
    monkeypatch.setattr(execution_budget, "install_build_budget_from_environment",
        lambda: SimpleNamespace(configuration=quota) if quota is not None else None)
    monkeypatch.setattr(entry, "ref", lambda path: {"path": str(path)})
    monkeypatch.setattr(entry, "read_bound_json", lambda _: {
        "development_authorization_ref": {"same": "grant"}, "shared_build_limit": 8})
    monkeypatch.setattr(entry, "prepare_delivery", lambda *_: pytest.fail("must not prepare delivery"))
    with pytest.raises(ValueError, match="SHARED_BUILD_BUDGET_REQUIRED"):
        entry.finalize("ready.json")
