"""Shared repository backend contracts; real compilation is an explicit selection."""
import json
import os
import shutil
from pathlib import Path
from xml.etree import ElementTree

import pytest

from src.mod_agent.build.adapter import BuildCommandResult, BuildResult
from src.mod_agent.build.nuget import NuGetIsolatedBuildExecution
from src.mod_agent.domain import csharp_repository_backend as owner
from src.mod_agent.domain.full_project_compilation import _ref


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_snapshot_handles_generated_paths_at_win32_limit_and_hashes_long_source(tmp_path):
    from src.mod_agent.build.dependencies import _windows_extended_length_path

    source = tmp_path / "source"
    source.mkdir()
    (source / "A.cs").write_text("original", encoding="utf-8")
    generated = source / "obj"
    generated.mkdir()
    name = "g" * max(16, 260 - len(str(generated)) - 4) + ".cs"
    target = generated / name
    io_target = Path(_windows_extended_length_path(str(target))) if os.name == "nt" else target
    io_target.write_bytes(b"generated")
    baseline = owner._snapshot(source)
    assert baseline == [{"path": "A.cs", "sha256": _ref(source / "A.cs")["sha256"]}]
    (source / "src").mkdir()
    long_name = "included" + name
    long_source = source / "src" / long_name
    io_source = Path(_windows_extended_length_path(str(long_source))) if os.name == "nt" else long_source
    io_source.write_bytes(b"included source")
    observed = owner._snapshot(source)
    assert len(observed) == 2
    assert len(str(long_source)) > 260
    assert next(row for row in observed if row["path"] == "src/" + long_name)["sha256"] == _ref(io_source)["sha256"]


def _profile(tmp_path, *, real=False, build_properties=None):
    source = tmp_path / "source"
    if real:
        selected = os.environ.get("REPOFORGE_DOTNET") or shutil.which("dotnet")
        if not selected:
            pytest.skip("Real repository integration requires REPOFORGE_DOTNET or dotnet on PATH.")
        sdk = Path(selected).resolve()
        if not sdk.is_file():
            raise ValueError("REPOFORGE_DOTNET_NOT_FOUND")
    else:
        sdk = tmp_path / "sdk/dotnet.exe"
    if not real:
        _write(sdk, "CONTROLLED TEST DOUBLE; NOT AN EXECUTABLE")
    _write(source / "Library/Library.csproj",
           '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>')
    _write(source / "Library/Arithmetic.cs",
           "public static class Arithmetic { public static int Double(int value) => value; }")
    _write(source / "tests/Tests.csproj",
           '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net8.0</TargetFramework>'
           '<OutputType>Exe</OutputType></PropertyGroup><ItemGroup>'
           '<ProjectReference Include="../Library/Library.csproj" /></ItemGroup></Project>')
    _write(source / "tests/Program.cs", """
using System.IO;
using System.Text.Json;
var actual = Arithmetic.Double(3);
File.WriteAllText(args[0], JsonSerializer.Serialize(new {
  schema_version = "csharp-test-report.v1",
  tests = new[] { new { name = "double", passed = actual == 6, expected = 6, actual = actual } }
}));
""")
    _write(source / "nuget.config", '<configuration><packageSources><clear /></packageSources></configuration>')
    data = {"profile_version": "csharp_repository_profile_v1", "dotnet_ref": _ref(sdk),
            "project_path": "Library/Library.csproj", "test_project_path": "tests/Tests.csproj",
            "nuget_config_path": "nuget.config", "configuration": "Release",
            "allowed_edit_paths": ["Library/Arithmetic.cs"],
            "expected_artifacts": ["Library/bin/Release/net8.0/Library.dll"],
            "test_command": ["run", "--project", "{test_project}", "--configuration", "{configuration}", "--", "{report_path}"],
            "test_report": {"path": "assertions.json", "format": "json_assertions"},
            "timeout_seconds": 60}
    if build_properties is not None:
        data["build_properties"] = build_properties
    path = tmp_path / "repository-profile.json"
    _write(path, json.dumps(data))
    return owner.CSharpRepositoryProfile.load(path), source


def _backend(tmp_path, profile, source, **kwargs):
    return owner.CSharpRepositoryBackend(profile_ref=profile.reference, source_root=source,
        workspace_root=tmp_path, output_root=tmp_path / "build-records", **kwargs)


def _fake_execution(monkeypatch, *, tests=None, test_exit=0, missing=False, stale=False, compile_paths=True,
                    capture_outputs=False, mutate_test_output=False):
    calls = []
    def build(profile):
        calls.append(profile)
        if any("RepositoryCompileInputs=" in part for part in profile.build_command):
            source = profile.working_directory
            _write(source / "Library/bin/Release/net8.0/Library.dll", "CONTROLLED NOT A REAL DLL")
            capture = Path(next(part.split("=", 1)[1] for part in profile.build_command if "RepositoryCompileInputs=" in part))
            if compile_paths:
                _write(capture, f"{source / 'Library/Library.csproj'}|{source / 'Library/Arithmetic.cs'}\n")
        if capture_outputs and any("RepositoryTestOutputs=" in part for part in profile.build_command):
            source = profile.working_directory
            artifact = source / "tests/bin/Release/net8.0/Tests.dll"
            _write(artifact, "CONTROLLED TEST OUTPUT")
            capture = Path(next(part.split("=", 1)[1] for part in profile.build_command if "RepositoryTestOutputs=" in part))
            _write(capture, f"{source / 'tests/Tests.csproj'}|{artifact}|Release|net8.0|\n")
        return NuGetIsolatedBuildExecution(0, 0, "", "", "", "", 1)
    def verify(profile, command):
        calls.append(command)
        report = Path(command[-1])
        assert not report.exists()
        assert "--no-build" in command and "--no-restore" in command
        if mutate_test_output:
            _write(profile.working_directory / "tests/bin/Release/net8.0/Tests.dll", "MUTATED")
        if not missing:
            _write(report, json.dumps({"schema_version": "csharp-test-report.v1",
                                      "tests": tests if tests is not None else [
                                          {"name": "double", "passed": True, "expected": 6, "actual": 6}]}))
            if stale:
                os.utime(report, (1, 1))
        return BuildCommandResult(test_exit, "fixture", "", 1)
    monkeypatch.setattr(owner.nuget, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(owner.nuget, "_run_nuget_isolated_verification_command", verify)
    return calls


def test_test_build_outputs_bind_actual_framework_and_hash(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path)
    _fake_execution(monkeypatch, capture_outputs=True)
    result = _backend(tmp_path, profile, source).run_build_test(source, "test", "controlled_fixture")
    outputs = result["test_execution"]["test_build_outputs"]
    assert outputs["status"] == "captured"
    assert outputs["outputs"] == [{"project": str(source / "tests/Tests.csproj"),
        "target_path": str(source / "tests/bin/Release/net8.0/Tests.dll"), "configuration": "Release",
        "target_framework": "net8.0", "runtime_identifier": "",
        "artifact_ref": _ref(source / "tests/bin/Release/net8.0/Tests.dll")}]
    assert _ref(outputs["capture_ref"]["path"]) == outputs["capture_ref"]
    fact = json.loads(Path(result["stage_receipts"]["test_build_outputs"]["path"]).read_text())
    assert fact["execution"] == outputs


def test_test_build_outputs_mutation_retains_execution_and_denies_result(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path)
    _fake_execution(monkeypatch, capture_outputs=True, mutate_test_output=True)
    with pytest.raises(ValueError, match="BUILT_ARTIFACT_CHANGED_DURING_TEST"):
        _backend(tmp_path, profile, source).run_build_test(source, "test", "controlled_fixture")
    error = json.loads(next((tmp_path / "build-records").glob("*/integrity-error.json")).read_text())
    assert {"test_build", "test_build_outputs", "test_execution", "raw_test_result"} <= error["stage_receipts"].keys()
    assert error["status"] == "unknown"
    assert not list((tmp_path / "build-records").glob("*/report.json"))


@pytest.mark.parametrize("defect", ["outside", "wrong_project", "malformed", "targets_changed"])
def test_test_output_capture_rejects_unbound_metadata(tmp_path, defect):
    profile, source = _profile(tmp_path)
    targets = tmp_path / "capture.targets"
    targets.write_bytes(owner.test_output_capture_targets())
    artifact = source / "tests/bin/Tests.dll"
    project = source / "tests/Tests.csproj"
    if defect == "outside":
        artifact = tmp_path / "outside.dll"
    if defect == "wrong_project":
        project = source / "Library/Library.csproj"
    _write(artifact, "fixture")
    capture = tmp_path / "capture.txt"
    _write(capture, "bad" if defect == "malformed" else f"{project}|{artifact}|Release|net8.0|\n")
    if defect == "targets_changed":
        targets.write_bytes(b"changed")
    with pytest.raises(ValueError, match="TEST_OUTPUT"):
        owner._test_build_outputs(profile.data, source, targets, capture)


def test_profile_binding_relocates_but_tests_and_config_are_immutable(tmp_path):
    profile, source = _profile(tmp_path)
    binding = profile.bind(source)
    copied = tmp_path / "copy"
    shutil.copytree(source, copied)
    profile.verify_fixed_inputs(copied, binding)
    _write(copied / "Library/Arithmetic.cs", "public static class Changed {}")
    profile.verify_fixed_inputs(copied, binding)
    _write(copied / "tests/Program.cs", "changed test")
    with pytest.raises(ValueError, match="FIXED_INPUTS_CHANGED"):
        profile.verify_fixed_inputs(copied, binding)
    with pytest.raises(ValueError, match="FIXED_INPUTS_CHANGED"):
        _backend(tmp_path, profile, copied, expected_fixed_inputs=binding)


@pytest.mark.parametrize("field,value", [
    ("allowed_edit_paths", ["tests/Program.cs"]), ("allowed_edit_paths", ["Library/Library.csproj"]),
    ("project_path", "../outside.csproj"), ("test_project_path", "Test.csproj"),
    ("test_command", ["bash", "{test_project}", "{report_path}"]),
    ("test_command", ["run", "--project"]), ("test_command", ["run", "--project", "{test_project}", "{evil}"]),
    ("timeout_seconds", 0), ("expected_artifacts", ["C:/outside.dll"]),
])
def test_profile_rejects_unbound_or_unsafe_scope(tmp_path, field, value):
    profile, _ = _profile(tmp_path)
    data = profile.data
    data[field] = value
    path = Path(profile.reference["path"])
    _write(path, json.dumps(data))
    with pytest.raises(ValueError):
        owner.CSharpRepositoryProfile.load(path)


def test_declared_sdk_and_project_used_for_restore_and_build(tmp_path):
    profile, source = _profile(tmp_path, build_properties={"MinVerSkip": "true"})
    prepared = profile.create_nuget_profile(tmp_path / "baseline", source)
    assert prepared.restore_command[:3] == (profile.data["dotnet_ref"]["path"], "restore", str(source / "Library/Library.csproj"))
    assert prepared.build_command[:3] == (profile.data["dotnet_ref"]["path"], "build", str(source / "Library/Library.csproj"))
    assert prepared.working_directory == source
    assert prepared.environment["NUGET_PACKAGES"].startswith(str(tmp_path / "baseline"))
    assert prepared.environment["DOTNET_ROOT"] == str(tmp_path / "sdk")
    assert prepared.environment["DOTNET_CLI_HOME"].startswith(str(tmp_path / "baseline"))
    assert prepared.environment["DOTNET_MULTILEVEL_LOOKUP"] == "0"
    assert prepared.environment["MSBUILDDISABLENODEREUSE"] == "1"
    assert prepared.environment["DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER"] == "1"
    assert "-p:UseSharedCompilation=false" in prepared.build_command
    assert "-p:MinVerSkip=true" in prepared.build_command
    assert not (source / "Library/bin").exists()  # Profile creation performs no Build/test.


def test_shared_build_and_test_result_is_real_object_and_reference_bound(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path, build_properties={"MinVerSkip": "true"})
    backend = _backend(tmp_path, profile, source)
    identity = backend.verification_identity()
    calls = _fake_execution(monkeypatch)
    result = backend.run_build_test(source, "test", "controlled_fixture")
    assert isinstance(result["build_result"], BuildResult)
    assert result["status"] == "passed"
    assert result["test_execution"]["test_build_outputs"]["status"] == "unavailable"
    assert result["verification_summary"] == {"checks": 1, "failed_checks": 0, "executions": 1, "test_command_executed": True}
    assert result["checks"][0]["check_id"] == result["checks"][0]["name"] == "double"
    assert result["materialized_code_identity"] is None
    compilation = json.loads(Path(result["compilation_report_ref"]["path"]).read_text(encoding="utf-8"))
    assert compilation["verification_scope"] == "formal_repository_compilation"
    assert len(calls) == 3
    assert "-p:BuildProjectReferences=false" not in calls[1].build_command
    assert calls[1].environment["DOTNET_ROOT"] == calls[0].environment["DOTNET_ROOT"]
    for prepared in calls[:2]:
        assert prepared.environment["MSBUILDDISABLENODEREUSE"] == "1"
        assert prepared.environment["DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER"] == "1"
        assert "-p:UseSharedCompilation=false" in prepared.build_command
        assert "-p:MinVerSkip=true" in prepared.build_command
    saved = json.loads(Path(result["report_ref"]["path"]).read_text(encoding="utf-8"))
    returned = result["build_result"].to_artifact_payload()
    assert returned["artifact_verification"]["repository_validation_ref"] == result["report_ref"]
    assert returned["artifact_verification"]["repository_test_report_ref"] == result["test_execution"]["report_ref"]
    assert "repository_validation_ref" not in saved["build_result"]["artifact_verification"]
    assert saved["build_result"]["verified_artifacts"] == returned["verified_artifacts"]
    assert result["report_ref"] == _ref(result["report_ref"]["path"])
    assert backend.verification_identity() == identity
    assert _backend(tmp_path, profile, source).verification_identity() == identity


@pytest.mark.parametrize("properties", [
    {"UseSharedCompilation": "true"}, {"BuildProjectReferences": "true"},
    {"bad;name": "true"}, {"MinVerSkip": "line\nbreak"}, [],
])
def test_profile_rejects_unsafe_or_reserved_build_properties(tmp_path, properties):
    profile, _ = _profile(tmp_path)
    data = profile.data
    data["build_properties"] = properties
    path = Path(profile.reference["path"])
    _write(path, json.dumps(data))
    with pytest.raises(ValueError, match="BUILD_PROPERTIES_INVALID"):
        owner.CSharpRepositoryProfile.load(path)


def test_verifier_identity_survives_exact_source_relocation_but_not_drift(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path)
    backend = _backend(tmp_path, profile, source)
    identity = backend.verification_identity()
    copied_owner = tmp_path / "snapshot/csharp_repository_backend.py"
    copied_build_owner = tmp_path / "snapshot/nuget.py"
    copied_owner.parent.mkdir(parents=True)
    copied_owner.write_bytes(Path(owner.__file__).read_bytes())
    copied_build_owner.write_bytes(Path(owner.nuget.__file__).read_bytes())

    monkeypatch.setattr(owner, "__file__", str(copied_owner))
    monkeypatch.setattr(owner.nuget, "__file__", str(copied_build_owner))
    assert backend.verification_identity() == identity

    copied_build_owner.write_bytes(copied_build_owner.read_bytes() + b"\n# drift")
    assert backend.verification_identity() != identity


def test_baseline_candidate_and_formal_profiles_share_only_task_local_cache(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path)
    baseline = profile.create_nuget_profile(tmp_path, source)
    calls = _fake_execution(monkeypatch)
    backend = _backend(tmp_path, profile, source)
    first = backend.run_build_test(source, "first", "controlled_fixture")
    copied = tmp_path / "other-source"
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns("bin", "obj"))
    formal_backend = owner.CSharpRepositoryBackend(
        profile_ref=profile.reference, source_root=copied, workspace_root=tmp_path,
        output_root=tmp_path / "formal-records", expected_fixed_inputs=backend.fixed_inputs)
    second = formal_backend.run_build_test(copied, "second", "controlled_fixture")
    assert first["report_ref"]["path"] != second["report_ref"]["path"]
    assert first["test_execution"]["report_ref"]["path"] != second["test_execution"]["report_ref"]["path"]
    for prepared in [calls[0], calls[1], calls[3], calls[4]]:
        assert prepared.packages_path == baseline.packages_path
        assert prepared.http_cache_path == baseline.http_cache_path
        assert prepared.environment["DOTNET_CLI_HOME"] == baseline.environment["DOTNET_CLI_HOME"]
        assert prepared.packages_path.is_relative_to(tmp_path)
        assert not prepared.packages_path.is_relative_to(backend.output_root)
        assert prepared.packages_path == tmp_path / ".cs" / profile.reference["sha256"][:12] / "p"
    unrelated_task = tmp_path / "other-task"
    other = profile.create_nuget_profile(unrelated_task, source)
    assert other.packages_path != baseline.packages_path
    assert other.packages_path.is_relative_to(unrelated_task)


def test_task_cache_checks_full_identity_and_does_not_overwrite(tmp_path):
    profile, source = _profile(tmp_path)
    prepared = profile.create_nuget_profile(tmp_path, source)
    binding = prepared.run_root / "profile.sha256"
    binding.write_text("different-full-profile-hash", encoding="ascii")
    with pytest.raises(ValueError, match="TASK_CACHE_PROFILE_CHANGED"):
        profile.create_nuget_profile(tmp_path, source)
    assert binding.read_text(encoding="ascii") == "different-full-profile-hash"


@pytest.mark.parametrize("settings,expected", [
    ({"tests": []}, "infrastructure_failed"),
    ({"missing": True}, "infrastructure_failed"),
    ({"stale": True}, "infrastructure_failed"),
    ({"test_exit": 7}, "infrastructure_failed"),
    ({"compile_paths": False}, "infrastructure_failed"),
    ({"tests": [{"name": "double", "passed": False, "expected": 6, "actual": 3}]}, "behavior_failed"),
])
def test_compile_success_never_substitutes_for_test_success(tmp_path, monkeypatch, settings, expected):
    profile, source = _profile(tmp_path)
    _fake_execution(monkeypatch, **settings)
    result = _backend(tmp_path, profile, source).run_build_test(source, "test", "controlled_fixture")
    assert result["status"] == expected
    assert not result["build_result"].claims_real_build_success
    assert result["build_result"].status == "failed"
    assert result["build_result"].artifact_verification["repository_validation_ref"] == result["report_ref"]
    assert result["observations"]


def test_timeout_retains_failed_build_without_running_test(tmp_path, monkeypatch):
    import subprocess
    profile, source = _profile(tmp_path)
    def timeout(profile):
        raise subprocess.TimeoutExpired(profile.build_command, 60)
    monkeypatch.setattr(owner.nuget, "_run_nuget_isolated_build", timeout)
    result = _backend(tmp_path, profile, source).run_build_test(source, "test", "controlled_fixture")
    assert result["status"] == "infrastructure_failed"
    assert result["test_execution"]["status"] == "not_executed"
    assert result["verification_summary"]["executions"] == 0
    assert result["build_execution"]["build_executed"] is False


@pytest.mark.parametrize("target,code", [
    ("tests/Program.cs", "FIXED_INPUTS_CHANGED"),
    ("Library/Arithmetic.cs", "INPUT_CHANGED_DURING_EXECUTION"),
    ("Library/bin/Release/net8.0/Library.dll", "BUILT_ARTIFACT_CHANGED_DURING_TEST"),
])
def test_test_execution_cannot_change_inputs_or_built_artifact(tmp_path, monkeypatch, target, code):
    profile, source = _profile(tmp_path)
    _fake_execution(monkeypatch)
    verify = owner.nuget._run_nuget_isolated_verification_command
    def mutate(profile, command):
        result = verify(profile, command)
        _write(source / target, "changed during test")
        return result
    monkeypatch.setattr(owner.nuget, "_run_nuget_isolated_verification_command", mutate)
    with pytest.raises(ValueError, match=code):
        _backend(tmp_path, profile, source).run_build_test(source, "test", "controlled_fixture")
    run = next((tmp_path / "build-records").iterdir())
    integrity = owner.read_bound_json(_ref(run / "integrity-error.json"))
    assert integrity["status"] == "unknown"
    assert code in integrity["error"]
    assert not (run / "report.json").exists()
    assert not (run / "candidate-report.json").exists()
    stages = integrity["stage_receipts"]
    assert set(stages) == {"build", "test_build", "test_build_outputs", "test_execution", "raw_test_result"}
    execution = owner.read_bound_json(stages["test_execution"])
    assert execution["execution"]["exit_code"] == 0
    assert execution["execution"]["stdout"] == "fixture"
    assert Path(execution["logs"]["stdout"]["path"]).read_text() == "fixture"
    raw = owner.read_bound_json(stages["raw_test_result"])["raw_report_ref"]
    assert owner.read_bound_json(raw)["tests"][0]["name"] == "double"


def test_post_test_snapshot_exception_retains_stage_facts_without_acceptance(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path)
    backend = _backend(tmp_path, profile, source)
    _fake_execution(monkeypatch)
    verify = owner.nuget._run_nuget_isolated_verification_command
    def fail_after_return(selected, command):
        result = verify(selected, command)
        def inaccessible(_):
            raise OSError("saved post-test snapshot failure")
        monkeypatch.setattr(owner, "_snapshot", inaccessible)
        return result
    monkeypatch.setattr(owner.nuget, "_run_nuget_isolated_verification_command", fail_after_return)
    with pytest.raises(OSError, match="saved post-test snapshot failure"):
        backend.run_build_test(source, "test", "controlled_fixture")
    run = next(backend.output_root.iterdir())
    assert not (run / "report.json").exists()
    receipt = owner.read_bound_json(_ref(run / "integrity-error.json"))
    assert receipt["status"] == "unknown"
    assert set(receipt["stage_receipts"]) == {"build", "test_build", "test_build_outputs", "test_execution", "raw_test_result"}
    for stage, reference in receipt["stage_receipts"].items():
        fact = owner.read_bound_json(reference)
        assert fact["stage"] == stage
        assert fact["context"]["source_root"] == str(source.resolve())
        assert fact["context"]["profile_ref"] == profile.reference
    raw = owner.read_bound_json(receipt["stage_receipts"]["raw_test_result"])["raw_report_ref"]
    assert owner._read_tests(Path(raw["path"]), "json_assertions")[0]["passed"] is True


def test_stage_facts_exist_before_parsing_and_success_binds_integrity(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path)
    backend = _backend(tmp_path, profile, source)
    calls = _fake_execution(monkeypatch)
    parse = owner._read_tests
    def inspect_before_parse(path, kind):
        run = next(backend.output_root.iterdir())
        raw = owner.read_bound_json(_ref(run / "raw_test_result-fact.json"))
        assert raw["raw_report_ref"] == _ref(path)
        assert not (run / "integrity.json").exists()
        assert not (run / "report.json").exists()
        test = owner.read_bound_json(_ref(run / "test_execution-fact.json"))
        assert test["commands"]["test"] == list(calls[-1])
        assert test["execution"]["exit_code"] == 0
        return parse(path, kind)
    monkeypatch.setattr(owner, "_read_tests", inspect_before_parse)
    result = backend.run_build_test(source, "test", "controlled_fixture")
    integrity = owner.read_bound_json(result["integrity_ref"])
    assert integrity["status"] == "validated"
    assert integrity["stage_receipts"] == result["stage_receipts"]
    assert integrity["context"]["source_inputs"] == result["source_inputs"]
    assert integrity["context"]["verification_identity"] == result["verification_identity"]


def test_candidate_materialization_copies_full_repository_without_changing_original(tmp_path, monkeypatch):
    from src.mod_agent.candidate_execution import materialized_code_identity
    from src.mod_agent.execution.patch_apply import MaterializedPatchBytes
    profile, source = _profile(tmp_path)
    backend = _backend(tmp_path, profile, source)
    before = (source / "Library/Arithmetic.cs").read_bytes()
    after = before.replace(b"=> value;", b"=> value * 2;")
    relative = "source/Library/Arithmetic.cs"
    materialized = MaterializedPatchBytes({relative: before}, {relative: after}, (), (), ())
    monkeypatch.setattr(owner, "materialize_candidate", lambda **kwargs: materialized)
    _fake_execution(monkeypatch)
    response = tmp_path / "response.json"
    _write(response, '{"changes":[]}')
    changes = [{"edit_binding_id": "binding-1"}]
    context = {"generated_changes": changes, "generated_changes_sha256": owner.canonical_hash(changes),
               "implementation_plan": {"feature_id": "test"}, "implementation_plan_ref": {"feature_id": "test"},
               "materialized_code_identity": materialized_code_identity(materialized),
               "model_response": {"path": str(response), "output_sha256": _ref(response)["sha256"]},
               "source_snapshot_sha256": "a" * 64}
    result = backend(context)
    assert result["status"] == "passed"
    assert (source / "Library/Arithmetic.cs").read_bytes() == before
    execution = json.loads(Path(result["verification_ref"]["path"]).read_text(encoding="utf-8"))
    copied = Path(execution["source_root"])
    assert copied.is_relative_to(tmp_path / ".cs" / profile.reference["sha256"][:12] / "c")
    assert not copied.is_relative_to(backend.output_root)
    assert len(copied.parent.name) == 8
    assert Path(result["report_ref"]["path"]).is_relative_to(backend.output_root)
    assert (copied / "Library/Arithmetic.cs").read_bytes() == after
    assert (copied / "tests/Program.cs").read_bytes() == (source / "tests/Program.cs").read_bytes()
    assert result["materialized_code_identity"] == context["materialized_code_identity"]
    assert execution["materialized_code_identity"] == context["materialized_code_identity"]
    compilation_ref = result["validation_results"]["full_compilation"]["report_ref"]
    compilation = json.loads(Path(compilation_ref["path"]).read_text(encoding="utf-8"))
    assert compilation["verification_scope"] == "isolated_candidate_compilation"
    assert compilation["materialized_code_identity"] == context["materialized_code_identity"]
    assert execution["compilation_report_ref"] == compilation_ref
    assert execution["built_artifact_ref"] == compilation["built_artifact_ref"]
    # Scope: public delegation contract against controlled execution, not an actual delegated Apply.
    # 为什么要有它：正常交付消费者应能核对同一候选和实际测试条目，而不是只验证backend自己的字段。
    from types import SimpleNamespace

    from src.mod_agent.execution.task_delegation import _verification
    assert _verification(result["report_ref"], ["double"],
        request=SimpleNamespace(isolated_workspace_root=str(tmp_path.resolve())),
        plan_ref=context["implementation_plan_ref"], code_identity=context["materialized_code_identity"],
        proposal={"diff_intent": [{"generation": {"model_response_sha256": context["model_response"]["output_sha256"]}}]}) is None
    again = backend(context)
    repeated = json.loads(Path(again["verification_ref"]["path"]).read_text(encoding="utf-8"))
    assert repeated["source_root"] != execution["source_root"]
    assert (copied / "Library/Arithmetic.cs").read_bytes() == after
    assert result["materialized_code_identity"] == again["materialized_code_identity"]


def test_trx_zero_skipped_or_incomplete_cannot_pass(tmp_path):
    path = tmp_path / "tests.trx"
    _write(path, '<TestRun><Results/><ResultSummary outcome="Completed"><Counters executed="0" /></ResultSummary></TestRun>')
    with pytest.raises(ValueError, match="TESTS_NOT_EXECUTED"):
        owner._read_tests(path, "trx")
    _write(path, '<TestRun><Results><UnitTestResult testName="x" outcome="Passed"/></Results>'
           '<ResultSummary outcome="Completed"><Counters executed="1"/></ResultSummary></TestRun>')
    assert owner._read_tests(path, "trx")[0]["passed"]
    _write(path, path.read_text().replace('outcome="Completed"', 'outcome="Aborted"'))
    with pytest.raises(ValueError, match="TRX_EXECUTION_INCOMPLETE"):
        owner._read_tests(path, "trx")


def _write_trx(path, rows, definitions=()):
    root = ElementTree.Element("TestRun", xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010")
    results = ElementTree.SubElement(root, "Results")
    for row in rows:
        ElementTree.SubElement(results, "UnitTestResult", row)
    declared = ElementTree.SubElement(root, "TestDefinitions")
    for test_id, class_name, method_name in definitions:
        test = ElementTree.SubElement(declared, "UnitTest", id=test_id, name=method_name)
        ElementTree.SubElement(test, "TestMethod", className=class_name, name=method_name)
    summary = ElementTree.SubElement(root, "ResultSummary", outcome="Completed")
    ElementTree.SubElement(summary, "Counters", executed=str(len(rows)))
    path.write_bytes(ElementTree.tostring(root))


def test_trx_parameterized_and_same_names_keep_unique_runner_identities_across_order(tmp_path):
    path = tmp_path / "parameterized.trx"
    rows = [
        {"testId": "method-a", "testName": "Case(1)", "outcome": "Passed", "executionId": "run-a-1"},
        {"testId": "method-a", "testName": "Case(2)", "outcome": "Failed", "executionId": "run-a-2"},
        {"testId": "method-b", "testName": "Case(1)", "outcome": "Passed", "executionId": "run-b-1"},
    ]
    definitions = [("method-a", "First.Class", "Case"), ("method-b", "Second.Class", "Case")]
    _write_trx(path, rows, definitions)
    original_ref = _ref(path)
    parsed = owner._read_tests(path, "trx")
    assert len(parsed) == len({row["check_id"] for row in parsed}) == 3
    assert [row["name"] for row in parsed] == [row["testName"] for row in rows]
    assert all(row["identity_scope"] == "runner_test_id_and_display_name" for row in parsed)
    assert all(row["cross_run_semantics_verified"] is False for row in parsed)
    assert all(row["report_ref"] == original_ref for row in parsed)
    assert parsed[0]["test_definitions"][0]["class_name"] == "First.Class"
    assert parsed[2]["test_definitions"][0]["class_name"] == "Second.Class"
    assert _ref(path) == original_ref
    reordered = tmp_path / "reordered.trx"
    _write_trx(reordered, [{**row, "executionId": "different-" + str(index)}
                         for index, row in enumerate(reversed(rows))], definitions)
    second = owner._read_tests(reordered, "trx")
    def identities(items):
        return {(row["test_id"], row["name"]): row["check_id"] for row in items}
    assert identities(parsed) == identities(second)
    assert [row["result_index"] for row in second] == [0, 1, 2]


def test_trx_true_duplicate_pairs_keep_every_outcome_with_report_local_occurrence(tmp_path):
    path = tmp_path / "duplicates.trx"
    rows = [
        {"testId": "method", "testName": "Case(1)", "outcome": "Passed", "executionId": "execution-1"},
        {"testId": "method", "testName": "Case(1)", "outcome": "Failed", "executionId": "execution-2"},
    ]
    _write_trx(path, rows)
    parsed = owner._read_tests(path, "trx")
    assert len(parsed) == len({row["check_id"] for row in parsed}) == 2
    assert [row["actual"] for row in parsed] == ["Passed", "Failed"]
    assert [row["occurrence"] for row in parsed] == [1, 2]
    assert all(row["group_size"] == 2 and row["identity_scope"] == "report_local_occurrence" for row in parsed)
    reordered = tmp_path / "duplicates-reordered.trx"
    _write_trx(reordered, list(reversed(rows)))
    second = owner._read_tests(reordered, "trx")
    assert {row["check_id"] for row in parsed}.isdisjoint(row["check_id"] for row in second)
    assert [row["actual"] for row in second] == ["Failed", "Passed"]


def test_trx_missing_runner_id_is_report_scoped_not_inferred_from_display_name(tmp_path):
    path = tmp_path / "missing-id.trx"
    _write_trx(path, [{"testName": "Case(1)", "outcome": "Passed"}])
    row = owner._read_tests(path, "trx")[0]
    assert row["test_id"] is None
    assert row["identity_scope"] == "report_local_occurrence"
    assert row["occurrence"] == 1
    assert row["report_ref"] == _ref(path)


def test_backend_preserves_trx_identity_through_candidate_feedback(tmp_path, monkeypatch):
    profile, source = _profile(tmp_path)
    data = profile.data
    data["test_report"] = {"path": "report.trx", "format": "trx"}
    _write(Path(profile.reference["path"]), json.dumps(data))
    profile = owner.CSharpRepositoryProfile.load(profile.reference["path"])
    _fake_execution(monkeypatch)
    def verify(profile, command):
        _write_trx(Path(command[-1]), [
            {"testId": "first-method", "testName": "same display", "outcome": "Passed"},
            {"testId": "other-method", "testName": "same display", "outcome": "Failed"},
        ])
        return BuildCommandResult(1)
    monkeypatch.setattr(owner.nuget, "_run_nuget_isolated_verification_command", verify)
    result = _backend(tmp_path, profile, source).run_build_test(source, "trx-fixture", "controlled_fixture")
    assert result["status"] == "behavior_failed"
    assert len({row["check_id"] for row in result["checks"]}) == 2
    assert [row["name"] for row in result["checks"]] == ["same display", "same display"]
    assert result["observations"][0]["check_id"] == result["checks"][1]["check_id"]
    assert all(row["report_ref"] == result["test_execution"]["report_ref"] for row in result["checks"])


@pytest.mark.skipif(os.environ.get("REPOFORGE_REAL_CSHARP_BACKEND_TEST") != "1",
                    reason="Explicit no-package local SDK integration selection, not routine unit testing.")
def test_real_no_package_repository_failure_then_repair(tmp_path):
    profile, source = _profile(tmp_path, real=True)
    backend = _backend(tmp_path, profile, source)
    failed = backend.run_build_test(source, "real-library", "local_developer_verification")
    assert failed["status"] == "behavior_failed", failed
    assert failed["checks"][0]["actual"] == 3
    emitted = failed["test_execution"]["test_build_outputs"]
    assert emitted["status"] == "captured"
    assert len(emitted["outputs"]) == 1
    assert emitted["outputs"][0]["target_framework"] == "net8.0"
    assert emitted["outputs"][0]["project"] == str(source / "tests/Tests.csproj")
    path = source / "Library/Arithmetic.cs"
    path.write_text(path.read_text().replace("=> value;", "=> value * 2;"), encoding="utf-8")
    passed = backend.run_build_test(source, "real-library", "local_developer_verification")
    assert passed["status"] == "passed", passed
    assert passed["build_result"].claims_real_build_success
    assert passed["checks"][0]["actual"] == 6
    assert failed["report_ref"] != passed["report_ref"]
