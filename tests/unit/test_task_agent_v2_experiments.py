"""Controlled runner evidence verifies exploration wiring, not OS isolation.

为什么要有它：受控runner只能证明权限与记录合同，不能计为真实沙箱或模型自主实验成果。
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.mod_agent.task_agent_v2_experiments import (
    CSharpRegisteredTestRunner,
    ExperimentOutcomeUnknownError,
    ExperimentRejectedError,
    TaskAgentExperiments,
    _file_ref,
    _source,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, TaskAgentV2StateError
from tests.unit.test_task_agent_v2_state import edit  # noqa: F401
from tests.unit.test_task_agent_v2_state import state as state


class ControlledTests:
    def __init__(self):
        self.calls = []
        self.zero = False
        self.extra = False
        self.fail = False
        self.crash = False

    def identity(self):
        return {"runner": "controlled-tests", "environment": "unit-fixture"}

    def read_public_report(self, _reference):
        return {"checks": [{"check_id": str(index), "test_definitions": [{"class_name": "Tests.Cache",
            "method_name": method}]} for index, method in enumerate(["Existing", "Existing", "Fresh"])],
            "raw_test_report_ref": "controlled-report", "runner_identity": self.identity(), "selection_available": True}

    def run_selected(self, *, state, source_revision_ref, selected, **_kwargs):
        self.calls.append(deepcopy(selected))
        if self.crash:
            raise OSError("process completion unavailable")
        ref, identity, _files = _source(state, source_revision_ref)
        groups = [] if self.zero else [row["method_group"] for row in selected]
        if self.extra:
            groups.append("Tests.Unselected.Method")
        return {"source_revision_ref": ref, "source_identity": identity, "runner_identity": self.identity(),
            "execution_origin": "executed", "execution_kind": "controlled", "exit_code": 1 if self.fail else 0,
            "tests_executed": len(groups), "executed_method_groups": groups, "passed": not self.fail,
            "logs_ref": state.save("experiment-logs", {"observations": ["controlled fixture"]})}


def setup_tests(state):
    runner = ControlledTests()
    owner = TaskAgentExperiments(state=state, test_runner=runner)
    catalog = owner.test_catalog("controlled-public-report")
    return owner, runner, catalog


def test_catalog_groups_parameter_cases_and_selects_registered_ids(state):
    owner, runner, catalog = setup_tests(state)
    existing = next(row for row in catalog["entries"] if row["method_group"].endswith("Existing"))
    assert existing["observed_cases"] == 2
    assert existing["selection_scope"] == "all_parameter_cases_in_method_group"
    args = {"source_revision_ref": "baseline", "catalog_ref": catalog["catalog_ref"],
        "test_ids": [existing["test_id"]], "operation_id": "select-1"}
    result = owner.registered_tests_run(**args)
    assert result["status"] == "completed" and result["grants_acceptance"] is False
    assert "check_ref" not in result
    assert owner.registered_tests_run(**args) == result
    assert len(runner.calls) == 1
    record = state.load(result["result_ref"])
    assert record["request"]["verification_scope"] == "exploration_only"
    assert record["request"]["source_revision_ref"] == state.contract["baseline_manifest_ref"]


@pytest.mark.parametrize("selection", [[], ["arbitrary --filter code"], ["same", "same"]])
def test_unregistered_or_empty_selection_never_dispatches(state, selection):
    owner, runner, catalog = setup_tests(state)
    with pytest.raises(ExperimentRejectedError):
        owner.registered_tests_run(source_revision_ref="baseline", catalog_ref=catalog["catalog_ref"],
            test_ids=selection, operation_id="invalid")
    assert runner.calls == []


@pytest.mark.parametrize("mode", ["zero", "extra", "fail"])
def test_zero_or_unselected_tests_and_real_failure_are_failed_observations(state, mode):
    owner, runner, catalog = setup_tests(state)
    setattr(runner, mode, True)
    result = owner.registered_tests_run(source_revision_ref="baseline", catalog_ref=catalog["catalog_ref"],
        test_ids=[catalog["entries"][0]["test_id"]], operation_id="failure")
    assert result["status"] == "failed" and result["executed"] is True
    assert not state.head.get("pending_operation")
    if mode in {"zero", "extra"}:
        assert result["observations"]["failure_code"] == "SELECTED_TESTS_NOT_EXECUTED"


def test_unknown_execution_is_persistent_and_cannot_repeat(state):
    owner, runner, catalog = setup_tests(state)
    runner.crash = True
    args = {"source_revision_ref": "baseline", "catalog_ref": catalog["catalog_ref"],
        "test_ids": [catalog["entries"][0]["test_id"]], "operation_id": "unknown"}
    with pytest.raises(ExperimentOutcomeUnknownError):
        owner.registered_tests_run(**args)
    with pytest.raises(TaskAgentV2StateError, match="operation_outcome_unknown"):
        owner.registered_tests_run(**args)
    assert len(runner.calls) == 1
    statuses = [state.load(path.relative_to(state.store_root).as_posix())["status"]
        for path in (state.root / "experiment-operation").glob("*.json")]
    assert set(statuses) == {"prepared", "dispatched", "unknown"}


def submit_probe(owner, **extra):
    return owner.probe_submit(source_revision_ref="baseline", files=[{"relative_path": "probe/Check.cs", "content": "class Check {}"}],
        entrypoint_id="csharp_console", rationale="Compare existing and fresh readers",
        expected_observation="Existing reader follows latest state", **extra)


def test_probe_submission_without_sandbox_does_not_block_tests_or_candidates(state):
    owner, runner, catalog = setup_tests(state)
    baseline = state.current_revision_ref
    submitted = submit_probe(owner)
    result = owner.experiment_run(source_revision_ref="baseline", probe_artifact_ref=submitted["probe_artifact_ref"], operation_id="probe-unavailable")
    assert result["status"] == "capability_unavailable" and result["executed"] is False
    assert state.current_revision_ref == baseline
    test = owner.registered_tests_run(source_revision_ref="baseline", catalog_ref=catalog["catalog_ref"],
        test_ids=[catalog["entries"][0]["test_id"]], operation_id="test-after-probe")
    assert test["status"] == "completed" and len(runner.calls) == 1
    candidate = edit(state, "A.cs", "1", "2")
    with pytest.raises(ExperimentRejectedError, match="PROBE_REVISION_CHANGED"):
        owner.experiment_run(source_revision_ref=candidate["candidate_ref"],
            probe_artifact_ref=submitted["probe_artifact_ref"], operation_id="wrong-version")


@pytest.mark.parametrize("path", ["../A.cs", "Library/A.cs", "probe/build.targets", "probe/x.csproj", "probe/../A.cs",
    "probe/.git/Own.cs", "probe/CON.cs"])
def test_probe_cannot_submit_project_scripts_or_escape_experiment_paths(state, path):
    owner = TaskAgentExperiments(state=state)
    with pytest.raises(ValueError):
        owner.probe_submit(source_revision_ref="baseline", files=[{"relative_path": path, "content": "text"}],
            entrypoint_id="csharp_console", rationale="reason", expected_observation="result")


def test_probe_size_and_entrypoint_are_host_limits(state):
    owner = TaskAgentExperiments(state=state)
    with pytest.raises(ExperimentRejectedError, match="PROBE_ENTRYPOINT_DENIED"):
        owner.probe_submit(source_revision_ref="baseline", files=[], entrypoint_id="cmd.exe", rationale="x", expected_observation="y")
    with pytest.raises(ExperimentRejectedError, match="PROBE_FILE_TOO_LARGE"):
        owner.probe_submit(source_revision_ref="baseline", files=[{"relative_path": "probe/x.cs", "content": "x" * 32769}],
            entrypoint_id="csharp_console", rationale="x", expected_observation="y")


def test_controlled_probe_runner_requires_evidence_and_remains_labelled_controlled(state):
    owner = TaskAgentExperiments(state=state)
    calls = []
    capability = {"status": "qualified", "profile_ref": owner.probe_profile_ref,
        "execution_kind": "controlled", "runner_identity": {"kind": "controlled-fixture"},
        "boundaries": {"filesystem": "isolated", "network": "denied", "process": "isolated"}}
    evidence = state.save("probe-qualification", capability)
    capability["evidence_refs"] = [evidence]

    def run_probe(**args):
        calls.append(args)
        ref, identity, _files = _source(state, args["source_revision_ref"])
        return {"source_revision_ref": ref, "source_identity": identity, "runner_identity": capability["runner_identity"],
            "execution_kind": "controlled", "execution_origin": "executed", "exit_code": 0,
            "logs_ref": state.save("probe-logs", {"message": "controlled test, no process launched"})}

    owner.probe_runner = SimpleNamespace(verify_capability=lambda _profile: deepcopy(capability), run_probe=run_probe)
    probe = submit_probe(owner)
    result = owner.experiment_run(source_revision_ref="baseline", probe_artifact_ref=probe["probe_artifact_ref"], operation_id="controlled-probe")
    assert result["observations"]["execution_kind"] == "controlled" and len(calls) == 1
    assert result["grants_acceptance"] is False


def test_csharp_catalog_reparses_real_trx_definitions_and_rejects_stale_report(state, tmp_path):
    import json

    from src.mod_agent.domain.csharp_repository_backend import _read_tests

    root = tmp_path / "runner"
    root.mkdir()
    trx = root / "tests.trx"
    trx.write_bytes(b'<TestRun><TestDefinitions><UnitTest id="one" name="Existing"><TestMethod className="Tests.Cache" name="Existing" /></UnitTest></TestDefinitions><Results><UnitTestResult testId="one" testName="Existing(1)" outcome="Passed" /><UnitTestResult testId="one" testName="Existing(2)" outcome="Passed" /></Results><ResultSummary outcome="Completed"><Counters executed="2" /></ResultSummary></TestRun>')
    checks = _read_tests(trx, "trx")
    profile_ref = {"path": "controlled-profile", "sha256": "hash"}
    backend = SimpleNamespace(output_root=root, fixed_inputs=[], verification_identity=lambda: {"runner": "fixture"},
        profile=SimpleNamespace(reference=profile_ref, data={"test_report": {"format": "trx"}, "test_command": ["test"]}))
    report = root / "report.json"
    report.write_text(json.dumps({"execution_origin": "executed", "profile_ref": profile_ref,
        "verification_identity": backend.verification_identity(), "fixed_inputs": [],
        "test_execution": {"status": "executed", "report_ref": _file_ref(trx)}, "checks": checks}), encoding="utf-8")
    runner = CSharpRegisteredTestRunner(backend)
    owner = TaskAgentExperiments(state=state, test_runner=runner)
    catalog = owner.test_catalog(_file_ref(report))
    assert len(catalog["entries"]) == 1 and catalog["entries"][0]["observed_cases"] == 2
    trx.write_bytes(b"changed")
    with pytest.raises(ExperimentRejectedError, match="CATALOG_TEST_REPORT_CHANGED"):
        owner.test_catalog(_file_ref(report))


def test_derived_test_profile_only_changes_registered_filter_and_keeps_formal_profile(state, tmp_path, monkeypatch):
    from src.mod_agent.domain import csharp_repository_backend as backend_module

    source = state.load(state.contract["baseline_manifest_ref"])["source_root"]
    from pathlib import Path
    source = Path(source)
    original_profile = {"test_command": ["test", "{test_project}", "--filter", "Category=Unit"],
        "test_report": {"format": "trx"}}
    profile_ref = {"path": "profile", "sha256": "unchanged"}
    authority = state.load(state.contract["authority_ref"])
    state = TaskAgentV2State.create(state.store_root, "experiment-source", source,
        read_roots={key: value for key, value in authority["read_roots"].items() if key != "target"},
        allowed_edit_paths=["A.cs", "B.cs"], authority_ref=state.contract["authority_ref"],
        contract_fields={"execution_profile_ref": profile_ref})
    host = SimpleNamespace(profile=SimpleNamespace(data=deepcopy(original_profile), reference=profile_ref,
        task_work_root=lambda _root: tmp_path / "scratch"), source_root=source, workspace_root=tmp_path,
        verification_identity=lambda: {"runner": "host", "environment": "fixture"})
    observed = []

    class ControlledBackend:
        def __init__(self, **kwargs):
            import json
            observed.append(json.loads(Path(kwargs["profile_ref"]["path"]).read_text()))

        def run_build_test(self, *_args):
            return {"checks": [{"check_id": "case-1", "test_definitions": [{"class_name": "Tests.Cache", "method_name": "Existing"}]}],
                "test_execution": {"exit_code": 0}, "build_execution": {}, "report_ref": {"path": "controlled-report"},
                "status": "passed", "observations": []}

    monkeypatch.setattr(backend_module, "CSharpRepositoryBackend", ControlledBackend)
    runner = CSharpRegisteredTestRunner(host)
    result = runner.run_selected(state=state, source_revision_ref="baseline", catalog={"runner_identity": runner.identity()},
        selected=[{"method_group": "Tests.Cache.Existing"}], operation_id="selection")
    assert result["selection_complete"] is True
    assert observed[0]["test_command"] == ["test", "{test_project}", "--filter", "(Category=Unit)&(FullyQualifiedName=Tests.Cache.Existing)"]
    assert host.profile.data == original_profile
    assert state.current_revision_ref == state.contract["baseline_manifest_ref"]


@pytest.mark.parametrize("damage", ["runner_identity", "source_identity", "source_file"])
def test_report_binding_or_original_source_mutation_is_never_admitted(state, monkeypatch, damage):
    from pathlib import Path

    owner, runner, catalog = setup_tests(state)
    original = runner.run_selected

    def damaged(**kwargs):
        result = original(**kwargs)
        if damage == "source_file":
            source = state.load(state.contract["baseline_manifest_ref"])["source_root"]
            Path(source, "A.cs").write_bytes(b"unauthorized effect")
        else:
            result[damage] = "different"
        return result

    monkeypatch.setattr(runner, "run_selected", damaged)
    with pytest.raises(ExperimentOutcomeUnknownError):
        owner.registered_tests_run(source_revision_ref="baseline", catalog_ref=catalog["catalog_ref"],
            test_ids=[catalog["entries"][0]["test_id"]], operation_id="damaged")
    assert state.head["pending_operation"]
