"""Repository originals through controlled build transport; no model success claim."""
import json
import shutil
from pathlib import Path

import pytest

from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation import strategy_repository_evaluator as owner
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.task_agent_execution import _worker_binding
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact
from tests.unit.test_csharp_repository_backend import _backend, _fake_execution, _profile
from tests.unit.test_task_agent_v2_state import edit


@pytest.fixture
def evidence(tmp_path, monkeypatch, request):
    profile, source = _profile(tmp_path)
    if hasattr(request, "param"):
        data = profile.data
        data["test_command"] = ["run", "--project", "{test_project}", *request.param, "--", "{report_path}"]
        profile = owner.backend.CSharpRepositoryProfile(_save(tmp_path / "selected-profile.json", data))
    (source / "Library/key.snk").write_bytes(b"\xff\xfe\x80\x00")
    controls = {}
    for role, value in (("correct", "value * 2"), ("equivalent", "value + value"), ("wrong", "value + 1")):
        copied = tmp_path / role
        shutil.copytree(source, copied)
        path = copied / "Library/Arithmetic.cs"
        path.write_text(path.read_text().replace("=> value;", f"=> {value};"))
        _fake_execution(monkeypatch, tests=[
            {"name": "target", "passed": role != "wrong", "expected": 6, "actual": 4 if role == "wrong" else 6},
            {"name": "preserve", "passed": True, "expected": 0, "actual": 0}], test_exit=1 if role == "wrong" else 0)
        controls[role] = _backend(tmp_path, profile, copied).run_build_test(copied, "controlled", "controlled_fixture")["report_ref"]
    kwargs = dict(profile_ref=profile.reference, controls=controls,
                  production_paths=["Library/Arithmetic.cs", "Library/Library.csproj", "Library/key.snk", "nuget.config"],
                  target_test_ids=["target"], preservation_test_ids=["preserve"], output_root=tmp_path / "qualification")
    return kwargs, source


def test_original_controls_qualify_and_parameter_failures_remain_false(evidence):
    kwargs, _ = evidence
    reference = owner.qualify_repository_strategy_evaluator(**kwargs)
    config = owner._configuration(reference)
    failed = owner._report(config, kwargs["controls"]["wrong"])
    assert failed["task_success"] is False and failed["preservation_passed"] is True
    assert config["execution_scope"] == "host_workspace_execution_no_os_isolation"


def test_completed_json_driver_with_failed_assertion_stays_failed_even_with_zero_exit(evidence, tmp_path, monkeypatch):
    kwargs, _ = evidence
    retained = read_bound_json(kwargs["controls"]["wrong"])
    source = Path(retained["source_root"])
    profile = owner.backend.CSharpRepositoryProfile(kwargs["profile_ref"])
    _fake_execution(monkeypatch, tests=[
        {"name": "target", "passed": False, "expected": 6, "actual": 4},
        {"name": "preserve", "passed": True, "expected": 0, "actual": 0}], test_exit=0)
    report = _backend(tmp_path, profile, source).run_build_test(source, "controlled", "controlled_fixture")["report_ref"]
    kwargs["controls"]["wrong"] = report
    reference = owner.qualify_repository_strategy_evaluator(**kwargs)
    verdict = owner._report(owner._configuration(reference), report)
    assert verdict["task_success"] is False and verdict["preservation_passed"] is True


@pytest.mark.parametrize("mutation,code", [
    ("omit_project", "PRODUCTION_SOURCE_OMITTED"), ("overlap", "TEST_SCOPE_INCOMPLETE"),
    ("missing_tests", "TEST_SCOPE_INCOMPLETE"), ("wrong_pass", "QUALIFICATION_CONTROL_FAILED")])
def test_qualification_rejects_incomplete_contracts(evidence, mutation, code):
    kwargs, _ = evidence
    if mutation == "omit_project":
        kwargs["production_paths"].remove("Library/Library.csproj")
    elif mutation == "overlap":
        kwargs["preservation_test_ids"] = ["target"]
    elif mutation == "missing_tests":
        kwargs["target_test_ids"] = ["unexecuted"]
    else:
        kwargs["controls"]["wrong"] = kwargs["controls"]["correct"]
    with pytest.raises(ValueError, match=code):
        owner.qualify_repository_strategy_evaluator(**kwargs)


@pytest.mark.parametrize("mutation", ["compile_capture", "dll", "test_report", "production", "summary"])
def test_rechecks_originals_including_failed_control(evidence, mutation, tmp_path):
    kwargs, _ = evidence
    evaluator = owner.qualify_repository_strategy_evaluator(**kwargs)
    report = read_bound_json(kwargs["controls"]["wrong"])
    if mutation == "compile_capture":
        (Path(kwargs["controls"]["wrong"]["path"]).parent / "compile-inputs.txt").write_text("")
    elif mutation == "dll":
        Path(report["built_artifact_refs"][0]["path"]).write_text("changed")
    elif mutation == "test_report":
        Path(report["test_execution"]["report_ref"]["path"]).write_text("{}")
    elif mutation == "production":
        (Path(report["source_root"]) / "Library/Library.csproj").write_text("changed")
    else:
        config = read_bound_json(evaluator)
        report["status"] = "passed"
        bad = _save(tmp_path / "report.json", report)
        with pytest.raises(ValueError):
            owner._report(config, bad)
        return
    with pytest.raises(ValueError):
        owner._configuration(evaluator)


def test_aggregate_matches_whole_candidate_and_rejects_wrong_candidate(evidence, tmp_path):
    kwargs, source = evidence
    evaluator = owner.qualify_repository_strategy_evaluator(**kwargs)
    store = tmp_path / "store"
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": ["Library/Arithmetic.cs"], "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "task", source, {}, ["Library/Arithmetic.cs"], authority)
    candidate = edit(state, "Library/Arithmetic.cs", "=> value;", "=> value * 2;")
    profile = owner.backend.CSharpRepositoryProfile(kwargs["profile_ref"])
    worker_ref = _save(tmp_path / "worker.json", {"behavior_evaluator": {
        "factory_ref": _ref(owner.backend.__file__), "factory_name": "make_csharp_repository_evaluator",
        "kwargs": {"profile_ref": kwargs["profile_ref"], "expected_fixed_inputs": profile.bind(source)}}})
    native_ref = _save(tmp_path / "native.json", {"worker_binding": _worker_binding(worker_ref)})
    execution = _save(tmp_path / "execution.json", {"task_id": "task", "candidate_source_id": candidate["candidate_source_id"],
        "native_execution_ref": native_ref,
        "candidate_state": {"artifact_store_root": str(store), "task_contract_ref": state.contract_ref,
                            "candidate_ref": candidate["candidate_ref"]}})
    verdict = owner.generate_repository_strategy_verdict(evaluator_ref=evaluator, execution_ref=execution,
        evidence_bundle_ref=kwargs["controls"]["correct"], output_path=tmp_path / "verdict.json")
    value = read_bound_json(verdict)
    assert owner.validate_repository_strategy_verdict(value)["status"] == "verified_repository_csharp"
    assert "checks" not in value
    value["evaluator_result_ref"] = kwargs["controls"]["equivalent"]
    with pytest.raises(ValueError, match="VERDICT_CHANGED"):
        owner.validate_repository_strategy_verdict(value)
    worker = read_bound_json(worker_ref)
    row = next(row for row in worker["behavior_evaluator"]["kwargs"]["expected_fixed_inputs"]["fixed_files"]
               if row["path"] == "Library/key.snk")
    row["sha256"] = "0" * 64
    changed_worker = _save(tmp_path / "changed-worker.json", worker)
    changed_native = _save(tmp_path / "changed-native.json", {"worker_binding": _worker_binding(changed_worker)})
    changed_execution = read_bound_json(execution)
    changed_execution["native_execution_ref"] = changed_native
    value["evaluator_result_ref"] = kwargs["controls"]["correct"]
    value["execution_ref"] = _save(tmp_path / "changed-execution.json", changed_execution)
    with pytest.raises(ValueError, match="VERDICT_CHANGED"):
        owner.validate_repository_strategy_verdict(value)


def test_trx_labels_use_complete_method_groups_and_require_definition():
    row = {"test_definitions": [{"class_name": "Tests.Cache", "method_name": "Update"}], "passed": True}
    assert owner._groups([row, {**row, "passed": False}], "trx") == {"Tests.Cache.Update": [True, False]}
    with pytest.raises(ValueError, match="TEST_METHOD_IDENTITY_MISSING"):
        owner._groups([{**row, "test_definitions": []}], "trx")


@pytest.mark.parametrize("arguments", [
    [], ["--configuration", "Debug"], ["--configuration", "Release", "-c", "Release"],
    ["--", "--configuration", "Release"], ["--configuration", "Release", "-p:Configuration=Debug"],
    ["--configuration", "Release", "@override.rsp"],
])
def test_test_configuration_cannot_select_stale_or_overridden_outputs(arguments):
    with pytest.raises(ValueError, match="TEST_CONFIGURATION"):
        owner._validate_test_configuration({"configuration": "Release", "build_properties": {},
            "test_command": ["run", "--project", "{test_project}", *arguments]})


@pytest.mark.parametrize("arguments", [["--configuration", "{configuration}"], ["-c", "Release"],
    ["--configuration=Release"], ["-c=Release"]])
def test_explicit_equivalent_configuration_flags_are_supported(arguments):
    owner._validate_test_configuration({"configuration": "Release", "build_properties": {},
        "test_command": ["run", "--project", "{test_project}", *arguments, "--", "{report_path}"]})


@pytest.mark.parametrize("evidence", [[], ["--configuration", "Debug"],
    ["--configuration", "Release", "-p:Configuration=Debug"]], indirect=True)
def test_matching_native_reports_still_reject_wrong_test_configuration(evidence):
    kwargs, _ = evidence
    # The reports genuinely match this host profile, so profile identity alone cannot catch the defect.
    with pytest.raises(ValueError, match="TEST_CONFIGURATION"):
        owner.qualify_repository_strategy_evaluator(**kwargs)
    assert not kwargs["output_root"].exists()


@pytest.mark.parametrize("mutation", ["unknown", "missing_stage", "changed_result"])
def test_stage_facts_cannot_be_replaced_by_final_summary(evidence, mutation):
    kwargs, _ = evidence
    report = read_bound_json(kwargs["controls"]["correct"])
    root = Path(kwargs["controls"]["correct"]["path"]).parent
    integrity = read_bound_json(report["integrity_ref"])
    if mutation == "unknown":
        integrity["status"] = "unknown"
    elif mutation == "missing_stage":
        report["stage_receipts"].pop("test_execution")
        integrity["stage_receipts"] = report["stage_receipts"]
    else:
        report["test_execution"]["exit_code"] = 99
    (root / "integrity.json").write_text(json.dumps(integrity), encoding="utf-8")
    report["integrity_ref"] = _ref(root / "integrity.json")
    with pytest.raises(ValueError, match="INTEGRITY_UNCONFIRMED|STAGE_MISSING|STAGE_RESULT_CHANGED"):
        owner._stage_facts(report, root)


@pytest.mark.parametrize("selector", [["--framework", "net7.0"], ["--runtime=win-x64"],
    ["-o", "old-output"], ["/p:TargetFramework=net7.0"], ["--property", "OutputPath=old-output"]])
def test_other_artifact_selectors_are_not_silently_qualified(selector):
    with pytest.raises(ValueError, match="TEST_ARTIFACT_SELECTION_UNSUPPORTED"):
        owner._validate_test_configuration({"configuration": "Release", "test_command": [
            "run", "--project", "{test_project}", "--configuration", "Release", *selector, "--", "{report_path}"]})


def test_build_only_target_framework_override_cannot_select_another_test_artifact():
    with pytest.raises(ValueError, match="TEST_ARTIFACT_SELECTION_UNSUPPORTED"):
        owner._validate_test_configuration({"configuration": "Release", "build_properties": {"TargetFramework": "net8.0"},
            "test_command": ["run", "--project", "{test_project}", "--configuration", "Release"]})
