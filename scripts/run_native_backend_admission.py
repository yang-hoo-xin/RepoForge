"""Measure and freeze the zero-model admission decision for one native plan.

The runner retains the earlier full-policy baseline for comparison, measures
three current commands with the frozen full policy, then executes a strictly
narrower permission profile for bounded scale/repeat/shell diagnosis.
It never calls a task model or evaluates a business task.
为什么要有它：成熟后端的业务运行必须由可复算的性能与权限结论准入，不能用延长任务等待替代执行器诊断。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ImportError:
    import tomli as tomllib

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_benchmark_arm import execute, read, ref, save, source_snapshot
from scripts.run_repository_product_pilot import native_reference_environment
from src.mod_agent.evaluation.native_backend_admission import freeze_native_backend_admission

READY_MARKER = b"REPOFORGE_NATIVE_ADMISSION_READY"
_SANDBOX_PASSTHROUGH_OPTIONS = frozenset({"--cd", "-c", "--disable", "--enable"})
_SANDBOX_REPLACED_OPTIONS = frozenset({"--color", "--model", "--output-last-message"})
_SANDBOX_EXEC_ONLY_FLAGS = frozenset({"--ignore-user-config", "--ignore-rules", "--strict-config",
    "--skip-git-repo-check", "--json", "--ephemeral"})


def _toml(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(str(key)) + "=" + _toml(nested) for key, nested in value.items()) + "}"
    if not isinstance(value, str):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_VALUE_INVALID")
    return json.dumps(value)


def _permission_leaves(value: Any, prefix: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _permission_leaves(nested, (*prefix, str(key)))
    else:
        yield prefix, value


def narrowed_permission_table(table: dict[str, Any]) -> dict[str, Any]:
    """Remove nested write grants while preserving every other exact rule."""
    if not isinstance(table, dict):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_TABLE_INVALID")
    result: dict[str, Any] = {}
    for root, value in table.items():
        if not isinstance(value, dict):
            result[root] = value
            continue
        retained = {relative: permission for relative, permission in value.items() if permission != "write"}
        if retained:
            result[root] = retained
    original = dict(_permission_leaves(table))
    narrowed = dict(_permission_leaves(result))
    if any(original[path] != permission for path, permission in narrowed.items()):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_NARROWING_INVALID")
    removed = {path: permission for path, permission in original.items() if path not in narrowed}
    if not removed or any(len(path) < 2 or permission != "write" for path, permission in removed.items()):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_NARROWING_INVALID")
    return result


def _sandbox_base(command: list[str], permission_table: dict[str, Any], *,
                  retain_config_profile: bool = False) -> list[str]:
    if (len(command) < 4 or command[1:3] != ["--no-daemon", "exec"] or command[-1] != "-"
            or type(retain_config_profile) is not bool):
        raise ValueError("NATIVE_ADMISSION_PLAN_COMMAND_INVALID")
    result = [command[0], "--no-daemon", "sandbox", "--include-managed-config", "-P", "benchmark"]
    index = 3
    while index < len(command) - 1:
        option = command[index]
        # Scope: only options whose sandbox meaning is explicitly classified may be copied or removed.
        # 为什么要有它：未知exec参数不能被静默丢弃后让探针测量另一份未冻结配置。
        if option in _SANDBOX_PASSTHROUGH_OPTIONS:
            if index + 1 >= len(command) - 1:
                raise ValueError("NATIVE_ADMISSION_PLAN_COMMAND_INVALID")
            result.extend(command[index:index + 2])
            index += 2
        elif option == "--profile":
            if index + 1 >= len(command) - 1:
                raise ValueError("NATIVE_ADMISSION_PLAN_COMMAND_INVALID")
            if retain_config_profile:
                result.extend(command[index:index + 2])
            index += 2
        elif option in _SANDBOX_REPLACED_OPTIONS:
            if index + 1 >= len(command) - 1:
                raise ValueError("NATIVE_ADMISSION_PLAN_COMMAND_INVALID")
            index += 2
        elif option in _SANDBOX_EXEC_ONLY_FLAGS:
            index += 1
        else:
            raise ValueError("NATIVE_ADMISSION_PLAN_COMMAND_INVALID")
    # Scope: full-policy trials reuse the frozen named config; narrowed diagnostics replace it with one explicit table.
    # 为什么要有它：完整权限不能因Windows命令长度重新内联，收窄对照也不能同时继承完整写规则。
    if not retain_config_profile:
        result.extend(["-c", "permissions.benchmark.filesystem=" + _toml(permission_table)])
    if len(subprocess.list2cmdline(result)) >= 32767:
        raise ValueError("NATIVE_ADMISSION_PERMISSION_COMMAND_EXCEEDS_WINDOWS_LIMIT")
    if any(value in result for value in ("--yolo", "--dangerously-bypass-approvals-and-sandbox")):
        raise ValueError("NATIVE_ADMISSION_SANDBOX_BYPASS_DENIED")
    return result


def _declared_shell_environment(command: list[str]) -> dict[str, str]:
    documents = [value for index, value in enumerate(command) if index and command[index - 1] == "-c"
        and value.startswith("shell_environment_policy.set=")]
    if len(documents) != 1:
        raise ValueError("NATIVE_ADMISSION_SHELL_ENVIRONMENT_INVALID")
    value = tomllib.loads(documents[0]).get("shell_environment_policy", {}).get("set")
    if not isinstance(value, dict) or not value:
        raise ValueError("NATIVE_ADMISSION_SHELL_ENVIRONMENT_INVALID")
    return {str(key): str(item) for key, item in value.items()}


def _process_observation(label: str, receipt: dict[str, Any], process_ref: dict[str, str]) -> dict[str, Any]:
    raw_phases = receipt.get("phase_timing")
    phases: dict[str, Any] = raw_phases if isinstance(raw_phases, dict) else {}
    try:
        stdout = Path(receipt["stdout_ref"]["path"]).read_bytes()
    except (KeyError, OSError, TypeError):
        stdout = b""
    payload_observed = READY_MARKER in stdout
    first_output = phases.get("stdout_first_output_seconds") if payload_observed else None
    return {
        "label": label,
        "process_ref": process_ref,
        "started": receipt.get("started") is True,
        "exit_code": receipt.get("exit_code"),
        "timeout": receipt.get("timeout"),
        "payload_observed": payload_observed,
        "payload_marker": READY_MARKER.decode("ascii"),
        "first_output_seconds": first_output
            if isinstance(first_output, (int, float)) and not isinstance(first_output, bool)
            and math.isfinite(first_output) else None,
        "phase_timing": deepcopy(phases),
        "owned_process_residuals": deepcopy(receipt.get("owned_process_residuals")),
    }


def run(plan_ref: dict[str, str], baseline_ref: dict[str, str], permission_matrix_ref: dict[str, str],
        output_root: str | Path, *, case_timeout_seconds: float = 35.0) -> dict[str, str]:
    if type(case_timeout_seconds) not in (int, float) or not math.isfinite(case_timeout_seconds) or case_timeout_seconds <= 30:
        raise ValueError("NATIVE_ADMISSION_CASE_TIMEOUT_MUST_EXCEED_SINGLE_LIMIT")
    plan = read(plan_ref)
    baseline = read(baseline_ref)
    permission_result = read(permission_matrix_ref)
    policy_ref = plan.get("native_permission_profile_ref")
    if not isinstance(policy_ref, dict) or set(policy_ref) != {"path", "sha256"}:
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID")
    if (
        baseline.get("plan_ref") != plan_ref
        or baseline.get("policy_ref") != policy_ref
        or permission_result.get("plan_ref") != plan_ref
        or permission_result.get("policy_ref") != policy_ref
    ):
        raise ValueError("NATIVE_ADMISSION_BASELINE_IDENTITY_CHANGED")
    stages = [stage for stage in baseline.get("stages", []) if stage.get("label") == "cmd"]
    if len(stages) != 1 or not isinstance(stages[0].get("first_output_elapsed_seconds"), (int, float)):
        raise ValueError("NATIVE_ADMISSION_FULL_POLICY_BASELINE_INVALID")
    raw_matrix = permission_result.get("permission_matrix")
    if not isinstance(raw_matrix, dict) or not isinstance(raw_matrix.get("checks"), list):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_MATRIX_INVALID")
    matrix: dict[str, Any] = raw_matrix

    try:
        native_profile = read(plan["native_reference_profile_ref"])
        codex_home = Path(native_profile["codex_home"]).resolve(strict=True)
        policy_before = ref(policy_ref["path"])
    except (KeyError, OSError, TypeError, ValueError):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID") from None
    # Scope: the short profile name must resolve to the exact hash-bound file in the final CODEX_HOME.
    # 为什么要有它：另一目录同名文件不能让计划校验A却让Codex实际加载B，也不能借链接或漂移替换权限。
    if (not codex_home.is_dir() or policy_before != policy_ref
            or Path(policy_before["path"]).parent != codex_home):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID")

    policy = tomllib.loads(Path(policy_before["path"]).read_text(encoding="utf-8"))
    full_table = policy.get("permissions", {}).get("benchmark", {}).get("filesystem")
    narrowed_table = narrowed_permission_table(full_table)
    full_leaves = list(_permission_leaves(full_table))
    narrowed_leaves = list(_permission_leaves(narrowed_table))
    if len(narrowed_leaves) >= len(full_leaves):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_SCALE_NOT_REDUCED")
    expected_profile = Path(policy_ref["path"]).name.removesuffix(".config.toml")
    selected_profiles = [value for index, value in enumerate(plan["command"])
        if index and plan["command"][index - 1] == "--profile"]
    if selected_profiles != [expected_profile]:
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID")

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = Path(plan["workspace"]).resolve(strict=True)
    source = Path(plan["source_root"]).resolve(strict=True)
    source_before = source_snapshot(source)
    environment = native_reference_environment({**os.environ, **plan.get("environment", {})}, native_profile,
        workspace=workspace)
    try:
        environment_codex_home = Path(environment["CODEX_HOME"]).resolve(strict=True)
    except (KeyError, OSError, TypeError, ValueError):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID") from None
    if environment_codex_home != codex_home:
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID")
    declared = _declared_shell_environment(plan["command"])
    for key, value in declared.items():
        for inherited in list(environment):
            if inherited.casefold() == key.casefold():
                del environment[inherited]
        environment[key] = value
    # Scope: validate the actual post-merge Windows environment, including case-insensitive key replacement.
    # 为什么要有它：shell_environment_policy不能在早期校验后用CoDeX_HoMe等拼写切换实际profile目录。
    final_codex_homes = [value for key, value in environment.items() if str(key).casefold() == "codex_home"]
    try:
        final_codex_home = Path(final_codex_homes[0]).resolve(strict=True)
    except (IndexError, OSError, TypeError, ValueError):
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID") from None
    if len(final_codex_homes) != 1 or final_codex_home != codex_home:
        raise ValueError("NATIVE_ADMISSION_PERMISSION_PROFILE_INVALID")

    full_base = _sandbox_base(plan["command"], full_table, retain_config_profile=True)
    narrowed_base = _sandbox_base(plan["command"], narrowed_table)
    cmd = [os.environ.get("COMSPEC", "C:/Windows/System32/cmd.exe"), "/d", "/c",
        "echo REPOFORGE_NATIVE_ADMISSION_READY"]
    powershell = [plan["native_shell_ref"]["path"], "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
        "[Console]::WriteLine('REPOFORGE_NATIVE_ADMISSION_READY')"]
    cases = (("full_cmd_1", full_base, cmd), ("full_cmd_2", full_base, cmd),
        ("full_cmd_3", full_base, cmd), ("narrowed_cmd_cold", narrowed_base, cmd),
        ("narrowed_cmd_repeated", narrowed_base, cmd),
        ("narrowed_powershell_repeated", narrowed_base, powershell))
    observations = []
    started = time.monotonic()
    for label, base, child in cases:
        case_root = output / label
        receipt = execute([*base, "--", *child], cwd=workspace, stdin_path=None, output=case_root,
            deadline=time.monotonic() + case_timeout_seconds, environment=environment, observe_output=True)
        observations.append(_process_observation(label, receipt, ref(case_root / "process.json")))

    if source_snapshot(source) != source_before or ref(policy_ref["path"]) != policy_before:
        raise ValueError("NATIVE_ADMISSION_PROBE_SIDE_EFFECT_DETECTED")
    if any(observation["owned_process_residuals"] for observation in observations):
        raise ValueError("NATIVE_ADMISSION_OWNED_PROCESS_RESIDUAL_DETECTED")
    comparisons_measured = all(observation["started"] and observation["exit_code"] == 0
        and observation["timeout"] is False and observation["payload_observed"] for observation in observations)
    baseline_stage = stages[0]
    full_observations, narrowed_observations = observations[:3], observations[3:]
    permission_checks = [{"name": row.get("name"), "passed": row.get("passed") is True}
        for row in matrix["checks"]]
    measurement = {
        "schema_version": "native_backend_admission_measurement_v1",
        "configuration_identity": plan_ref["sha256"],
        "plan_ref": plan_ref,
        "permission_profile_ref": policy_ref,
        "thresholds": {"consecutive_commands": 3, "first_output_median_seconds_max": 15.0,
            "first_output_single_seconds_max": 30.0},
        "admission_trials": [{"sequence": sequence,
            "configuration_identity": plan_ref["sha256"], "command_kind": "cmd",
            "first_output_seconds": observation["first_output_seconds"],
            "exit_code": observation["exit_code"], "timeout": observation["timeout"],
            "process_ref": observation["process_ref"]}
            for sequence, observation in enumerate(full_observations, 1)],
        "permission_matrix": {
            "passed": permission_result.get("passed") is True and matrix.get("passed") is True,
            "required_checks": [row["name"] for row in permission_checks],
            "checks": permission_checks,
            "evidence_ref": permission_matrix_ref,
        },
        "comparison_axes": {
            "permission_scale": {"status": "measured" if comparisons_measured else "not_measured",
                "full_permission_leaf_count": len(full_leaves),
                "narrowed_permission_leaf_count": len(narrowed_leaves),
                "narrowing": "nested write grants removed; no read, deny, or top-level temporary write rule changed",
                "historical_full_first_output_seconds": float(baseline_stage["first_output_elapsed_seconds"]),
                "full_observation": full_observations[0],
                "narrowed_observation": narrowed_observations[0],
                "evidence_refs": [baseline_ref, full_observations[0]["process_ref"],
                    narrowed_observations[0]["process_ref"]]},
            "cold_vs_repeated": {"status": "measured" if comparisons_measured else "not_measured",
                "observations": narrowed_observations[:2],
                "evidence_refs": [narrowed_observations[0]["process_ref"], narrowed_observations[1]["process_ref"]]},
            "cmd_vs_powershell": {"status": "measured" if comparisons_measured else "not_measured",
                "observations": narrowed_observations[1:],
                "evidence_refs": [narrowed_observations[1]["process_ref"], narrowed_observations[2]["process_ref"]]},
        },
        "baseline_evidence_refs": [baseline_ref, permission_matrix_ref],
        "probe_observations": observations,
        "source_unchanged": True,
        "permission_profile_unchanged": True,
        "model_calls": 0,
        "task_runs": 0,
        "wall_seconds": time.monotonic() - started,
        "case_timeout_seconds": float(case_timeout_seconds),
    }
    measurement_ref = save(output / "measurement.json", measurement)
    admission_ref = freeze_native_backend_admission(measurement_ref, output / "admission.json")
    result = {
        "schema_version": "native_backend_admission_run_v1",
        "plan_ref": plan_ref,
        "measurement_ref": measurement_ref,
        "admission_ref": admission_ref,
        "model_calls": 0,
        "task_runs": 0,
        "source_unchanged": True,
        "permission_profile_unchanged": True,
        "owned_process_residuals": [],
        "wall_seconds": measurement["wall_seconds"],
    }
    return cast(dict[str, str], save(output / "result.json", result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan-reference", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--baseline-reference", required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--permission-matrix-reference", required=True)
    parser.add_argument("--permission-matrix-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--case-timeout-seconds", type=float, default=35.0)
    args = parser.parse_args()
    result_ref = run(
        {"path": args.plan_reference, "sha256": args.plan_sha256},
        {"path": args.baseline_reference, "sha256": args.baseline_sha256},
        {"path": args.permission_matrix_reference, "sha256": args.permission_matrix_sha256},
        args.output_root,
        case_timeout_seconds=args.case_timeout_seconds,
    )
    print(json.dumps({"result_ref": result_ref}, ensure_ascii=False))


if __name__ == "__main__":
    main()
