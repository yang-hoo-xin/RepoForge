"""Exercise the actual shared backend boundary with controlled execution.

为什么要有它：诊断合法性必须影响真实共享检查结论，不能只验证独立辅助函数。
"""
import json
from pathlib import Path

import pytest

from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain import repository_diagnostics as diagnostics
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.evaluation.strategy_repository_evaluator import _stage_facts
from tests.unit.test_csharp_repository_backend import _backend, _fake_execution, _profile, _write


def diagnostic_profile(tmp_path, *, real=False):
    profile, source = _profile(tmp_path, real=real)
    _write(source / "tests/Expectations/Example.verified.txt", "expected\n")
    value = profile.data
    value["diagnostics"] = {"kind": "verify_snapshots_v1", "expectation_root": "tests/Expectations",
        "compiler_output_roots": ["Library/bin", "Library/obj", "tests/bin", "tests/obj"],
        "received_markers": [".received.", ".DotNet8_0.received."]}
    path = Path(profile.reference["path"])
    path.write_text(json.dumps(value), encoding="utf-8")
    return backend.CSharpRepositoryProfile.load(path), source


def executing_fixture(tmp_path, monkeypatch, *, mutation=None, failed=False):
    profile, source = diagnostic_profile(tmp_path)
    _fake_execution(monkeypatch, capture_outputs=True,
        tests=[{"name": "example", "passed": not failed, "expected": 1, "actual": 0 if failed else 1}])
    original = backend.nuget._run_nuget_isolated_verification_command

    def verify(selected, command):
        output = Path(selected.environment["REPOFORGE_DIAGNOSTIC_ROOT"])
        assert not output.is_relative_to(source)
        assert (output / "Example.verified.txt").read_bytes() == (source / "tests/Expectations/Example.verified.txt").read_bytes()
        assert selected.environment["REPOFORGE_EXPECTATION_ROOT"] == str(source / "tests/Expectations")
        (output / "Example.DotNet8_0.received.txt").write_bytes(b"actual observation\n")
        if mutation:
            mutation(source, output)
        return original(selected, command)

    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)
    return profile, source, _backend(tmp_path, profile, source)


@pytest.mark.parametrize("failed", [False, True])
def test_shared_check_accepts_declared_diagnostics_and_keeps_behavior_outcome(tmp_path, monkeypatch, failed):
    _profile_value, source, owner = executing_fixture(tmp_path, monkeypatch, failed=failed)
    before = backend._snapshot(source)
    result = owner.run_build_test(source, "diagnostic", "controlled")
    assert result["status"] == ("behavior_failed" if failed else "passed")
    report = json.loads(Path(result["report_ref"]["path"]).read_bytes())
    root = Path(result["report_ref"]["path"]).parent
    facts = _stage_facts(report, root)
    assert "test_execution" in facts and backend._snapshot(source) == before
    integrity = json.loads(Path(report["integrity_ref"]["path"]).read_bytes())
    assert integrity["checks"]["diagnostic_files"] == {"status": "verified"}
    receipt = diagnostics.validate_receipt(report["diagnostics_ref"], integrity["context"],
        environment=report["isolated_build_profile"]["environment"])
    assert len(receipt["expected_copies"]) == len(receipt["received_refs"]) == 1
    assert not (source / "tests/TestResults").exists()


@pytest.mark.parametrize("mutation", [
    lambda source, output: (output / "Example.verified.txt").write_bytes(b"tampered"),
    lambda source, output: (output / "unexpected.received.txt").write_bytes(b"unregistered"),
    lambda source, output: _write(source / "TestResults/hidden.cs", "new source"),
    lambda source, output: _write(source / "extra/bin/hidden.cs", "new source"),
    lambda source, output: _write(source / "tests/Expectations/Example.verified.txt", "new expectation"),
])
def test_shared_check_rejects_tamper_but_retains_actual_execution_facts(tmp_path, monkeypatch, mutation):
    _profile_value, source, owner = executing_fixture(tmp_path, monkeypatch, mutation=mutation)
    with pytest.raises(ValueError):
        owner.run_build_test(source, "diagnostic", "controlled")
    failures = list(owner.output_root.glob("*/integrity-error.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_bytes())
    assert failure["status"] == "unknown"
    assert {"build", "test_build", "test_execution", "raw_test_result"} <= failure["stage_receipts"].keys()
    assert not (failures[0].parent / "report.json").exists()


def test_profile_cannot_hide_arbitrary_source_as_compiler_output(tmp_path):
    profile, source = diagnostic_profile(tmp_path)
    value = profile.data
    value["diagnostics"]["compiler_output_roots"].append("unregistered/bin")
    reference = _save(tmp_path / "invalid-profile.json", value)
    with pytest.raises(ValueError, match="COMPILER_ROOT_NOT_PROJECT_OUTPUT"):
        backend.CSharpRepositoryProfile(reference).bind(source)


def test_independent_consumer_rechecks_copied_expectation_and_environment(tmp_path, monkeypatch):
    from src.mod_agent.public_failure_feedback import validated_public_checks
    _profile_value, source, owner = executing_fixture(tmp_path, monkeypatch)
    result = owner.run_build_test(source, "diagnostic", "controlled")
    report = json.loads(Path(result["report_ref"]["path"]).read_bytes())
    root = Path(result["report_ref"]["path"]).parent
    assert validated_public_checks(result["report_ref"])[0] == report
    altered = dict(report["isolated_build_profile"]["environment"])
    altered["REPOFORGE_DIAGNOSTIC_ROOT"] = str(tmp_path / "elsewhere")
    integrity = json.loads(Path(report["integrity_ref"]["path"]).read_bytes())
    with pytest.raises(ValueError, match="EXECUTION_ENVIRONMENT_CHANGED"):
        diagnostics.validate_receipt(report["diagnostics_ref"], integrity["context"], environment=altered)
    (root / "diagnostics/Example.verified.txt").write_bytes(b"changed after completed check")
    with pytest.raises(ValueError, match="EXPECTED_COPY_CHANGED"):
        _stage_facts(report, root)
    with pytest.raises(ValueError, match="EXPECTED_COPY_CHANGED"):
        validated_public_checks(result["report_ref"])
