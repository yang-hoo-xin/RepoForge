"""Host-declared exclusions remain not executed, never successful checks.

为什么要有它：允许原始套件显式跳过不能让新跳过、伪造计数或丢失来源获得验收资格。
"""
import json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

import pytest

from src.mod_agent.build.adapter import BuildCommandResult
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation import strategy_repository_evaluator as evaluator
from src.mod_agent.public_failure_feedback import validated_public_checks
from src.mod_agent.task_agent_v2_experiments import CSharpRegisteredTestRunner
from tests.unit.test_csharp_repository_backend import _backend, _fake_execution, _profile

SKIP = {"name": "Fixture.Disabled", "class_name": "Fixture", "method_name": "Disabled", "max_instances": 1}


def write_trx(path, *, failed=False, skipped=1, skip_name="Fixture.Disabled", extra_outcome=None):
    root = ElementTree.Element("TestRun")
    results = ElementTree.SubElement(root, "Results")
    rows = [("target", "Fixture.Target", "Target", "Failed" if failed else "Passed"),
            ("preserve", "Fixture.Preserve", "Preserve", "Passed")]
    rows += [("skip", skip_name, "Disabled", "NotExecuted")] * skipped
    if extra_outcome:
        rows.append(("extra", "Fixture.Extra", "Extra", extra_outcome))
    definitions = ElementTree.SubElement(root, "TestDefinitions")
    seen = set()
    for test_id, name, method, outcome in rows:
        row = ElementTree.SubElement(results, "UnitTestResult", testId=test_id, testName=name, outcome=outcome)
        if outcome == "Failed":
            ElementTree.SubElement(row, "Output").text = "Expected twice input, observed original input."
        if test_id not in seen:
            definition = ElementTree.SubElement(definitions, "UnitTest", id=test_id, name=method)
            ElementTree.SubElement(definition, "TestMethod", className="Fixture", name=method)
            seen.add(test_id)
    summary = ElementTree.SubElement(root, "ResultSummary", outcome="Failed" if failed else "Completed")
    ElementTree.SubElement(summary, "Counters", total=str(len(rows)), executed="2", passed=str(1 if failed else 2),
                          failed=str(int(failed)), notExecuted="0")
    path.write_bytes(ElementTree.tostring(root))
    return path


def skip_profile(tmp_path):
    profile, source = _profile(tmp_path)
    data = profile.data
    data.update(test_report={"path": "tests.trx", "format": "trx"}, allowed_skipped_tests=[SKIP])
    return backend.CSharpRepositoryProfile(_save(tmp_path / "skip-profile.json", data)), source


def controlled_runner(monkeypatch, *, failed=False):
    _fake_execution(monkeypatch, capture_outputs=True)

    def verify(_selected, command):
        write_trx(Path(command[-1]), failed=failed)
        return BuildCommandResult(int(failed))

    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)


@pytest.mark.parametrize("failed", [False, True])
def test_declared_skip_is_not_a_check_and_counter_discrepancy_is_retained(tmp_path, failed):
    path = write_trx(tmp_path / "tests.trx", failed=failed)
    before = _ref(path)
    with pytest.raises(ValueError, match="TESTS_NOT_EXECUTED"):
        backend._read_tests(path, "trx")
    metadata = {}
    checks = backend._read_tests(path, "trx", [SKIP], metadata=metadata)
    assert len(checks) == 2 and sum(not row["passed"] for row in checks) == int(failed)
    assert metadata["counts"] == {"total": 3, "executed": 2, "passed": 1 if failed else 2, "failed": int(failed), "skipped": 1}
    assert metadata["skipped_tests"][0]["actual"] == "NotExecuted"
    assert "passed" not in metadata["skipped_tests"][0]
    assert metadata["skipped_tests"][0]["report_ref"] == before
    assert metadata["counter_discrepancies"] == [{"counter": "notExecuted", "reported": "0", "observed": 1}]
    assert _ref(path) == before


@pytest.mark.parametrize("mutation,code", [
    ("unknown", "SKIPPED_TEST_NOT_ALLOWED"), ("excess", "SKIPPED_TEST_NOT_ALLOWED"),
    ("ambiguous", "SKIPPED_TEST_IDENTITY_AMBIGUOUS"), ("wrong_class", "SKIPPED_TEST_NOT_ALLOWED"),
    ("aborted", "TESTS_NOT_EXECUTED"), ("not_runnable", "TESTS_NOT_EXECUTED"),
    ("zero", "TESTS_NOT_EXECUTED"), ("total", "TRX_COUNTER_MISMATCH"),
    ("passed", "TRX_COUNTER_MISMATCH"), ("failed", "TRX_COUNTER_MISMATCH"),
    ("executed", "TESTS_NOT_EXECUTED"), ("missing_counter", "TRX_COUNTER_MISMATCH"),
    ("summary", "TRX_EXECUTION_INCOMPLETE"), ("aborted_counter", "TRX_EXECUTION_INCOMPLETE"),
])
def test_skips_do_not_admit_incomplete_or_changed_execution(tmp_path, mutation, code):
    path = write_trx(tmp_path / "tests.trx", skipped=2 if mutation == "excess" else 1,
        skip_name="Other.Disabled" if mutation == "unknown" else "Fixture.Disabled",
        extra_outcome={"aborted": "Aborted", "not_runnable": "NotRunnable"}.get(mutation))
    tree = ElementTree.parse(path)
    root = tree.getroot()
    counters = root.find(".//Counters")
    if mutation in {"total", "passed", "failed", "executed"}:
        counters.set(mutation, "8")
    elif mutation == "missing_counter":
        del counters.attrib["passed"]
    elif mutation == "zero":
        for row in list(root.find("Results"))[:2]:
            root.find("Results").remove(row)
        counters.set("executed", "0")
    elif mutation == "summary":
        root.find("ResultSummary").set("outcome", "Aborted")
    elif mutation == "aborted_counter":
        counters.set("aborted", "1")
    elif mutation == "ambiguous":
        root.find("TestDefinitions").append(deepcopy(root.find(".//UnitTest[@id='skip']")))
    elif mutation == "wrong_class":
        root.find(".//UnitTest[@id='skip']/TestMethod").set("className", "Other")
    tree.write(path)
    with pytest.raises(ValueError, match=code):
        backend._read_tests(path, "trx", [SKIP])


def test_policy_subset_is_allowed_but_not_a_full_acceptance_claim(tmp_path):
    path = write_trx(tmp_path / "tests.trx", skipped=0)
    metadata = {}
    assert len(backend._read_tests(path, "trx", [SKIP], metadata=metadata)) == 2
    assert metadata["counts"]["skipped"] == 0


@pytest.mark.parametrize("policy", [None, {}, [{**SKIP, "max_instances": True}], [{**SKIP, "max_instances": 0}],
                                     [SKIP, SKIP], [{"name": "*", "max_instances": 1}]])
def test_profile_rejects_non_explicit_or_ambiguous_skip_policy(tmp_path, policy):
    profile, _source = skip_profile(tmp_path)
    data = profile.data
    data["allowed_skipped_tests"] = policy
    with pytest.raises(ValueError, match="SKIP_POLICY"):
        backend.CSharpRepositoryProfile(_save(tmp_path / "bad-profile.json", data))


def test_policy_changes_bound_profile_and_cannot_mutate_in_place(tmp_path):
    profile, source = skip_profile(tmp_path)
    binding = profile.bind(source)
    data = profile.data
    data["allowed_skipped_tests"][0]["max_instances"] = 2
    changed = backend.CSharpRepositoryProfile(_save(tmp_path / "changed-profile.json", data))
    assert changed.reference["sha256"] != profile.reference["sha256"]
    with pytest.raises(ValueError, match="FIXED_INPUTS_CHANGED"):
        changed.verify_fixed_inputs(source, binding)
    # 为什么要有它：反例需真实篡改已有测试工件，生产的独占保存函数不应为测试允许覆盖。
    Path(profile.reference["path"]).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        profile.bind(source)


@pytest.mark.parametrize("failed", [False, True])
def test_shared_execution_feedback_catalog_and_independent_reader_agree(tmp_path, monkeypatch, failed):
    profile, source = skip_profile(tmp_path)
    controlled_runner(monkeypatch, failed=failed)
    runner = _backend(tmp_path, profile, source)
    result = runner.run_build_test(source, "declared-skips", "controlled_fixture")
    assert result["status"] == ("behavior_failed" if failed else "passed")
    assert result["verification_summary"] == {"checks": 2, "failed_checks": int(failed), "executions": 1, "test_command_executed": True}
    assert len(result["observations"]) == int(failed)
    report, checks = validated_public_checks(result["report_ref"])
    assert checks == result["checks"]
    assert CSharpRegisteredTestRunner(runner).read_public_report(result["report_ref"])["checks"] == checks
    config = {"profile_ref": profile.reference, "fixed_inputs": profile.bind(source),
        "production_paths": ["Library/Arithmetic.cs", "Library/Library.csproj", "nuget.config"],
        "target_test_ids": ["Fixture.Target"], "preservation_test_ids": ["Fixture.Preserve"]}
    observed = evaluator._report(config, result["report_ref"])
    assert observed["task_success"] is (not failed)
    assert observed["test_counts"] == {"Fixture.Target": 1, "Fixture.Preserve": 1}
    assert observed["test_execution_metadata"]["counts"]["skipped"] == 1
    report["test_execution_metadata"]["counts"]["skipped"] = 0
    report_path = Path(result["report_ref"]["path"])
    report_path.write_text(json.dumps(report), encoding="utf-8")
    bad = _ref(report_path)
    with pytest.raises(ValueError, match="TEST_EXECUTION_METADATA_CHANGED"):
        validated_public_checks(bad)
    with pytest.raises(ValueError, match="TEST_EXECUTION_METADATA_CHANGED"):
        evaluator._report(config, bad)


def test_independent_scope_cannot_count_skips_as_preservation(tmp_path, monkeypatch):
    profile, source = skip_profile(tmp_path)
    controlled_runner(monkeypatch)
    result = _backend(tmp_path, profile, source).run_build_test(source, "declared-skips", "controlled_fixture")
    config = {"profile_ref": profile.reference, "fixed_inputs": profile.bind(source),
        "production_paths": ["Library/Arithmetic.cs", "Library/Library.csproj", "nuget.config"],
        "target_test_ids": ["Fixture.Target"], "preservation_test_ids": ["Fixture.Preserve", "Fixture.Disabled"]}
    with pytest.raises(ValueError, match="TEST_SCOPE_INCOMPLETE"):
        evaluator._report(config, result["report_ref"])
