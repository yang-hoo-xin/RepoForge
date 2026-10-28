"""Batch admission behavior for the mature native backend.

为什么要有它：业务批次必须消费一个确定的准入决定，不能自行解释零散的启动计时与权限回执。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_native_backend_admission as admission_runner
from scripts.run_benchmark_arm import codex_argv
from src.mod_agent.evaluation.native_backend_admission import (
    decide_native_backend_admission,
    freeze_native_backend_admission,
    require_native_backend_admission,
)


def _measurement(first_outputs: list[float]) -> dict:
    return {
        "schema_version": "native_backend_admission_measurement_v1",
        "configuration_identity": "frozen-native-route-v1",
        "plan_ref": {"path": "C:/evidence/frozen-plan.json", "sha256": "1" * 64},
        "permission_profile_ref": {"path": "C:/evidence/policy.toml", "sha256": "2" * 64},
        "thresholds": {
            "consecutive_commands": 3,
            "first_output_median_seconds_max": 15.0,
            "first_output_single_seconds_max": 30.0,
        },
        "admission_trials": [
            {
                "sequence": index,
                "configuration_identity": "frozen-native-route-v1",
                "command_kind": "cmd",
                "first_output_seconds": seconds,
                "exit_code": 0,
                "timeout": False,
                "process_ref": {"path": f"C:/evidence/process-{index}.json", "sha256": str(index) * 64},
            }
            for index, seconds in enumerate(first_outputs, 1)
        ],
        "permission_matrix": {
            "passed": True,
            "required_checks": ["allowed_source_write", "private_read_denied"],
            "checks": [
                {"name": "allowed_source_write", "passed": True},
                {"name": "private_read_denied", "passed": True},
            ],
            "evidence_ref": {"path": "C:/evidence/permission-matrix.json", "sha256": "a" * 64},
        },
        "comparison_axes": {
            "permission_scale": {"status": "measured", "evidence_refs": []},
            "cold_vs_repeated": {"status": "measured", "evidence_refs": []},
            "cmd_vs_powershell": {"status": "measured", "evidence_refs": []},
        },
    }


def test_batch_admission_rejects_one_slow_command_and_retains_evidence() -> None:
    measurement = _measurement([12.0, 14.0, 31.0])

    result = decide_native_backend_admission(measurement)

    assert result["decision"] == "not_admitted"
    assert result["business_runs_allowed"] is False
    assert result["timing"] == {
        "complete_consecutive_samples": True,
        "first_output_seconds": [12.0, 14.0, 31.0],
        "median_seconds": 14.0,
        "maximum_seconds": 31.0,
        "passed": False,
    }
    assert result["reasons"] == ["FIRST_OUTPUT_SINGLE_LIMIT_EXCEEDED"]
    assert result["permission_matrix"]["evidence_ref"] == measurement["permission_matrix"]["evidence_ref"]
    assert result["configuration_identity"] == "frozen-native-route-v1"
    assert result["result_version"].startswith("native-backend-admission-v1:")


def test_batch_admission_allows_only_three_fast_commands_with_complete_permissions() -> None:
    measurement = _measurement([14.0, 15.0, 15.0])

    first = decide_native_backend_admission(measurement)
    second = decide_native_backend_admission(measurement)

    assert first["decision"] == "admitted"
    assert first["business_runs_allowed"] is True
    assert first["timing"]["median_seconds"] == 15.0
    assert first["timing"]["maximum_seconds"] == 15.0
    assert first["reasons"] == []
    assert second == first


def test_one_observed_slow_command_is_enough_to_freeze_not_admitted() -> None:
    measurement = _measurement([221.093])

    result = decide_native_backend_admission(measurement)

    assert result["decision"] == "not_admitted"
    assert result["timing"]["complete_consecutive_samples"] is False
    assert result["timing"]["first_output_seconds"] == [221.093]
    assert result["timing"]["maximum_seconds"] == 221.093
    assert result["reasons"] == [
        "THREE_CONSECUTIVE_COMMANDS_NOT_ESTABLISHED",
        "FIRST_OUTPUT_SINGLE_LIMIT_EXCEEDED",
    ]


def _save(path: Path, value: object) -> dict:
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _native_run_inputs(tmp_path: Path, baseline_change: dict | None = None) -> tuple[dict, dict, dict]:
    workspace, source, codex_home = (tmp_path / name for name in ("workspace", "source", "codex-home"))
    for path in (workspace, source, codex_home):
        path.mkdir()
    (source / "Example.cs").write_text("public class Example {}", encoding="utf-8")
    shell = tmp_path / "pwsh.exe"
    shell.write_bytes(b"controlled shell; never executed")
    policy = codex_home / "permission.config.toml"
    policy.write_text(
        'permissions.benchmark.filesystem={":root"="deny","C:/workspace"={"."="read","out"="write"}}\n',
        encoding="utf-8",
    )
    policy_ref = {"path": str(policy.resolve()), "sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}
    profile_ref = _save(tmp_path / "native-profile.json", {"codex_home": str(codex_home)})
    command = ["codex", "--no-daemon", "exec", "--cd", str(workspace), "--profile", "permission",
        "-c", 'shell_environment_policy.set={"DOTNET_ROOT"="C:/dotnet"}', "-"]
    plan_ref = _save(tmp_path / "plan.json", {"native_permission_profile_ref": policy_ref,
        "workspace": str(workspace), "source_root": str(source), "native_reference_profile_ref": profile_ref,
        "environment": {}, "command": command, "native_shell_ref": {
            "path": str(shell.resolve()), "sha256": hashlib.sha256(shell.read_bytes()).hexdigest()}})
    baseline_process_ref = _save(tmp_path / "baseline-process.json", {"first_output_seconds": 172.0})
    stage = {"label": "cmd", "first_output_elapsed_seconds": 172.0, "exit_code": 0,
        "timeout": False, "payload_observed": True, "process_ref": baseline_process_ref}
    stage.update(baseline_change or {})
    baseline_ref = _save(tmp_path / "baseline.json", {"plan_ref": plan_ref, "policy_ref": policy_ref,
        "stages": [stage]})
    permission_ref = _save(tmp_path / "permission-matrix.json", {"plan_ref": plan_ref,
        "policy_ref": policy_ref, "passed": True, "permission_matrix": {"passed": True,
            "checks": [{"name": "allowed", "passed": True}, {"name": "denied", "passed": True}]}})
    return plan_ref, baseline_ref, permission_ref


def test_frozen_admission_is_bound_to_plan_policy_and_original_evidence(tmp_path: Path) -> None:
    plan_ref = _save(tmp_path / "plan.json", {"kind": "controlled-plan"})
    policy_ref = _save(tmp_path / "policy.json", {"kind": "exact-policy"})
    matrix_ref = _save(tmp_path / "matrix.json", {"passed": True})
    measurement = _measurement([1.0, 2.0, 3.0])
    measurement["configuration_identity"] = plan_ref["sha256"]
    measurement["plan_ref"] = plan_ref
    measurement["permission_profile_ref"] = policy_ref
    measurement["permission_matrix"]["evidence_ref"] = matrix_ref
    for trial in measurement["admission_trials"]:
        trial["configuration_identity"] = plan_ref["sha256"]
        trial["process_ref"] = _save(tmp_path / f"process-{trial['sequence']}.json", trial)
    measurement_ref = _save(tmp_path / "measurement.json", measurement)

    result_ref = freeze_native_backend_admission(measurement_ref, tmp_path / "admission.json")
    result = require_native_backend_admission(
        result_ref,
        plan_ref=plan_ref,
        permission_profile_ref=policy_ref,
        configuration_identity=plan_ref["sha256"],
    )

    assert result["decision"] == "admitted"
    assert result["measurement_ref"] == measurement_ref
    assert result["admission_trials"] == measurement["admission_trials"]
    Path(matrix_ref["path"]).write_text('{"passed":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="NATIVE_BACKEND_ADMISSION_EVIDENCE_CHANGED"):
        require_native_backend_admission(
            result_ref,
            plan_ref=plan_ref,
            permission_profile_ref=policy_ref,
            configuration_identity=plan_ref["sha256"],
        )


def test_frozen_admission_cannot_self_sign_a_decision_refuted_by_measurement(tmp_path: Path) -> None:
    plan_ref = _save(tmp_path / "plan.json", {"kind": "controlled-plan"})
    policy_ref = _save(tmp_path / "policy.json", {"kind": "exact-policy"})
    matrix_ref = _save(tmp_path / "matrix.json", {"passed": True})
    measurement = _measurement([31.0, 31.0, 31.0])
    measurement["configuration_identity"] = plan_ref["sha256"]
    measurement["plan_ref"] = plan_ref
    measurement["permission_profile_ref"] = policy_ref
    measurement["permission_matrix"]["evidence_ref"] = matrix_ref
    for trial in measurement["admission_trials"]:
        trial["configuration_identity"] = plan_ref["sha256"]
        trial["process_ref"] = _save(tmp_path / f"process-{trial['sequence']}.json", trial)
    measurement_ref = _save(tmp_path / "measurement.json", measurement)
    admission_ref = freeze_native_backend_admission(measurement_ref, tmp_path / "admission.json")
    forged = json.loads(Path(admission_ref["path"]).read_text(encoding="utf-8"))
    assert forged["decision"] == "not_admitted"
    forged.update(decision="admitted", business_runs_allowed=True)
    forged.pop("result_version")
    encoded = json.dumps(forged, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    forged["result_version"] = "native-backend-admission-v1:" + hashlib.sha256(encoded).hexdigest()
    forged_ref = _save(tmp_path / "forged-admission.json", forged)

    with pytest.raises(ValueError, match="NATIVE_BACKEND_ADMISSION_RESULT_INVALID"):
        require_native_backend_admission(
            forged_ref,
            plan_ref=plan_ref,
            permission_profile_ref=policy_ref,
            configuration_identity=plan_ref["sha256"],
        )


def test_native_admission_probe_rejects_unknown_frozen_command_option() -> None:
    command = ["codex", "--no-daemon", "exec", "--strict-config", "--future-switch",
        "--cd", "C:/workspace", "-"]

    with pytest.raises(ValueError, match="NATIVE_ADMISSION_PLAN_COMMAND_INVALID"):
        admission_runner._sandbox_base(command, {"C:/workspace": "read"})


@pytest.mark.parametrize("baseline_change", [
    {"exit_code": 1},
    {"timeout": True},
    {"payload_observed": False},
    {"first_output_elapsed_seconds": True},
    {"first_output_elapsed_seconds": float("nan")},
    {"first_output_elapsed_seconds": -0.1},
])
def test_native_admission_rejects_unsuccessful_or_invalid_plan_bound_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline_change: dict,
) -> None:
    plan_ref, baseline_ref, permission_ref = _native_run_inputs(tmp_path, baseline_change)
    monkeypatch.setattr(admission_runner, "execute", lambda *_args, **_kwargs:
        pytest.fail("Invalid baseline must fail before a sandbox process starts"))

    with pytest.raises(ValueError, match="NATIVE_ADMISSION_FULL_POLICY_BASELINE_INVALID"):
        admission_runner.run(plan_ref, baseline_ref, permission_ref, tmp_path / "admission-run")


def test_native_admission_probe_classifies_every_supported_exec_option() -> None:
    command = ["codex", "--no-daemon", "exec", "--ignore-rules", "--strict-config",
        "--skip-git-repo-check", "--json", "--color", "never", "--model", "gpt-5.6-sol",
        "--cd", "C:/workspace", "--output-last-message", "C:/last-message.txt",
        "-c", 'approval_policy="never"', "--disable", "apps", "--enable", "skip_host_skill_discovery",
        "--ephemeral", "--profile", "repoforge-native", "-"]

    result = admission_runner._sandbox_base(command, {"C:/workspace": "read"})

    assert result[:6] == ["codex", "--no-daemon", "sandbox", "--include-managed-config", "-P", "benchmark"]
    assert result[6:14] == ["--cd", "C:/workspace", "-c", 'approval_policy="never"',
        "--disable", "apps", "--enable", "skip_host_skill_discovery"]
    assert result[-2] == "-c"
    assert result[-1].startswith("permissions.benchmark.filesystem=")
    assert not any(option in result for option in ("--json", "--model", "--profile", "--ephemeral"))


def test_native_admission_probe_classifies_the_formal_codex_command(tmp_path: Path) -> None:
    command = codex_argv("codex", tmp_path / "workspace", tmp_path / "last-message.txt", [tmp_path / "host"])

    result = admission_runner._sandbox_base(command, {str(tmp_path / "workspace"): "read"})

    assert result[:6] == ["codex", "--no-daemon", "sandbox", "--include-managed-config", "-P", "benchmark"]
    assert "--ignore-user-config" not in result
    assert result[result.index("--cd") + 1] == str(tmp_path / "workspace")
    assert 'approval_policy="never"' in result


@pytest.mark.parametrize("policy_location", ["codex_home", "same_name_elsewhere", "environment_changed",
    "declared_environment_changed"])
def test_native_admission_run_uses_three_current_full_policy_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy_location: str,
) -> None:
    workspace = tmp_path / "workspace"
    source = tmp_path / "source"
    codex_home = tmp_path / "codex-home"
    for path in (workspace, source, codex_home):
        path.mkdir()
    (source / "Example.cs").write_text("public class Example {}", encoding="utf-8")
    shell = tmp_path / "pwsh.exe"
    shell.write_bytes(b"controlled shell; never executed")
    policy_parent = tmp_path / "other-codex-home" if policy_location == "same_name_elsewhere" else codex_home
    policy_parent.mkdir(exist_ok=True)
    policy = policy_parent / "permission.config.toml"
    policy.write_text('permissions.benchmark.filesystem={":root"="deny","C:/workspace"={"."="read","out"="write"}}\n',
        encoding="utf-8")
    policy_ref = {"path": str(policy.resolve()), "sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}
    profile_ref = _save(tmp_path / "native-profile.json", {"codex_home": str(codex_home)})
    command = codex_argv("codex", workspace, tmp_path / "last-message.txt", [tmp_path / "host"])
    permission_index = next(index for index, value in enumerate(command)
        if value.startswith("permissions.benchmark.filesystem="))
    assert command[permission_index - 1] == "-c"
    del command[permission_index - 1:permission_index + 1]
    declared_environment = '{"DOTNET_ROOT"="C:/dotnet"}'
    if policy_location == "declared_environment_changed":
        changed_declared_home = tmp_path / "changed-declared-home"
        changed_declared_home.mkdir()
        declared_environment = '{"DOTNET_ROOT"="C:/dotnet","CoDeX_HoMe"=' + json.dumps(
            str(changed_declared_home)) + "}"
    command[-1:-1] = ["--profile", "permission", "-c",
        "shell_environment_policy.set=" + declared_environment]
    plan = {"native_permission_profile_ref": policy_ref, "workspace": str(workspace),
        "source_root": str(source), "native_reference_profile_ref": profile_ref,
        "environment": {}, "command": command, "native_shell_ref": {
            "path": str(shell.resolve()), "sha256": hashlib.sha256(shell.read_bytes()).hexdigest()}}
    plan_ref = _save(tmp_path / "plan.json", plan)
    baseline_process_ref = _save(tmp_path / "historical-process.json", {"first_output_seconds": 221.0})
    baseline_ref = _save(tmp_path / "baseline.json", {"plan_ref": plan_ref, "policy_ref": policy_ref,
        "stages": [{"label": "cmd", "first_output_elapsed_seconds": 221.0, "exit_code": 0,
            "timeout": False, "payload_observed": True, "process_ref": baseline_process_ref}]})
    permission_ref = _save(tmp_path / "permission-matrix.json", {"plan_ref": plan_ref,
        "policy_ref": policy_ref, "passed": True, "permission_matrix": {"passed": True,
            "checks": [{"name": "allowed", "passed": True}, {"name": "denied", "passed": True}]}})
    seconds = [1.0, 2.0, 3.0, 0.4, 0.5, 0.6]
    calls: list[list[str]] = []

    def execute(argv, **kwargs):
        sequence = len(calls)
        calls.append(argv)
        output = Path(kwargs["output"])
        output.mkdir()
        stdout = output / "stdout.bin"
        stdout.write_bytes(admission_runner.READY_MARKER + b"\n")
        receipt = {"started": True, "exit_code": 0, "timeout": False,
            "stdout_ref": {"path": str(stdout.resolve()), "sha256": hashlib.sha256(stdout.read_bytes()).hexdigest()},
            "phase_timing": {"stdout_first_output_seconds": seconds[sequence]},
            "owned_process_residuals": []}
        _save(output / "process.json", receipt)
        return receipt

    monkeypatch.setattr(admission_runner, "execute", execute)
    if policy_location == "environment_changed":
        changed_home = tmp_path / "changed-environment-home"
        changed_home.mkdir()
        monkeypatch.setattr(admission_runner, "native_reference_environment",
            lambda *_args, **_kwargs: {"CODEX_HOME": str(changed_home)})
    if policy_location != "codex_home":
        with pytest.raises(ValueError, match="NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID"):
            admission_runner.run(plan_ref, baseline_ref, permission_ref, tmp_path / "admission-run")
        assert calls == []
        return
    result_ref = admission_runner.run(plan_ref, baseline_ref, permission_ref, tmp_path / "admission-run")
    result = json.loads(Path(result_ref["path"]).read_text(encoding="utf-8"))
    measurement = json.loads(Path(result["measurement_ref"]["path"]).read_text(encoding="utf-8"))
    admission = json.loads(Path(result["admission_ref"]["path"]).read_text(encoding="utf-8"))

    assert len(calls) == 6
    assert [row["first_output_seconds"] for row in measurement["admission_trials"]] == [1.0, 2.0, 3.0]
    assert all(call[call.index("--profile") + 1] == "permission"
        and not any(value.startswith("permissions.benchmark.filesystem=") for value in call)
        for call in calls[:3])
    assert all('"out"="write"' not in next(value for value in call
        if value.startswith("permissions.benchmark.filesystem=")) for call in calls[3:])
    assert measurement["comparison_axes"]["permission_scale"]["historical_full_first_output_seconds"] == 221.0
    assert admission["decision"] == "admitted" and admission["business_runs_allowed"] is True


def test_successful_narrowed_comparisons_survive_full_policy_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_ref, baseline_ref, permission_ref = _native_run_inputs(tmp_path)
    narrowed_seconds = [0.5, 0.4, 0.8]
    calls = []

    def execute(argv, **kwargs):
        sequence = len(calls)
        calls.append(argv)
        output = Path(kwargs["output"])
        output.mkdir()
        stdout = output / "stdout.bin"
        timed_out = sequence < 3
        stdout.write_bytes(b"" if timed_out else admission_runner.READY_MARKER + b"\n")
        receipt = {"started": True, "exit_code": 124 if timed_out else 0, "timeout": timed_out,
            "stdout_ref": {"path": str(stdout.resolve()), "sha256": hashlib.sha256(stdout.read_bytes()).hexdigest()},
            "phase_timing": {"stdout_first_output_seconds": None if timed_out else narrowed_seconds[sequence - 3]},
            "owned_process_residuals": []}
        _save(output / "process.json", receipt)
        return receipt

    monkeypatch.setattr(admission_runner, "execute", execute)
    result_ref = admission_runner.run(plan_ref, baseline_ref, permission_ref, tmp_path / "admission-run")
    result = json.loads(Path(result_ref["path"]).read_text(encoding="utf-8"))
    measurement = json.loads(Path(result["measurement_ref"]["path"]).read_text(encoding="utf-8"))
    admission = json.loads(Path(result["admission_ref"]["path"]).read_text(encoding="utf-8"))

    assert [measurement["comparison_axes"][name]["status"] for name in (
        "permission_scale", "cold_vs_repeated", "cmd_vs_powershell")] == ["measured"] * 3
    assert admission["reasons"] == ["THREE_CONSECUTIVE_COMMANDS_NOT_ESTABLISHED"]
    assert admission["decision"] == "not_admitted" and admission["business_runs_allowed"] is False


@pytest.mark.parametrize(("failed_index", "expected_statuses"), [
    (3, {"permission_scale": "not_measured", "cold_vs_repeated": "not_measured", "cmd_vs_powershell": "measured"}),
    (4, {"permission_scale": "measured", "cold_vs_repeated": "not_measured", "cmd_vs_powershell": "not_measured"}),
    (5, {"permission_scale": "measured", "cold_vs_repeated": "measured", "cmd_vs_powershell": "not_measured"}),
])
def test_one_failed_narrowed_probe_only_invalidates_dependent_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_index: int, expected_statuses: dict,
) -> None:
    plan_ref, baseline_ref, permission_ref = _native_run_inputs(tmp_path)
    seconds = [1.0, 2.0, 3.0, 0.5, 0.4, 0.8]
    calls = []

    def execute(argv, **kwargs):
        sequence = len(calls)
        calls.append(argv)
        output = Path(kwargs["output"])
        output.mkdir()
        stdout = output / "stdout.bin"
        failed = sequence == failed_index
        stdout.write_bytes(b"" if failed else admission_runner.READY_MARKER + b"\n")
        receipt = {"started": True, "exit_code": 124 if failed else 0, "timeout": failed,
            "stdout_ref": {"path": str(stdout.resolve()), "sha256": hashlib.sha256(stdout.read_bytes()).hexdigest()},
            "phase_timing": {"stdout_first_output_seconds": None if failed else seconds[sequence]},
            "owned_process_residuals": []}
        _save(output / "process.json", receipt)
        return receipt

    monkeypatch.setattr(admission_runner, "execute", execute)
    result_ref = admission_runner.run(plan_ref, baseline_ref, permission_ref, tmp_path / "admission-run")
    result = json.loads(Path(result_ref["path"]).read_text(encoding="utf-8"))
    measurement = json.loads(Path(result["measurement_ref"]["path"]).read_text(encoding="utf-8"))
    admission = json.loads(Path(result["admission_ref"]["path"]).read_text(encoding="utf-8"))

    assert {name: axis["status"] for name, axis in measurement["comparison_axes"].items()} == expected_statuses
    assert admission["decision"] == "not_admitted"
    assert "REQUIRED_STARTUP_COMPARISON_NOT_ESTABLISHED" in admission["reasons"]
