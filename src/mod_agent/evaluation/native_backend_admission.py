"""Freeze one deterministic admission decision for the mature native backend.

The public interface consumes measured execution facts and returns the one result
that a batch gate needs.  It does not execute a model or infer missing evidence.
为什么要有它：延迟、权限和配置身份必须在一处共同判定，避免每个批次调用者自行解释相同证据。
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

MEASUREMENT_SCHEMA = "native_backend_admission_measurement_v1"
RESULT_SCHEMA = "native_backend_admission_v1"
RESULT_VERSION_PREFIX = "native-backend-admission-v1:"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_seconds(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _file_reference(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _read_reference(reference: Any) -> bytes:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("NATIVE_BACKEND_ADMISSION_EVIDENCE_CHANGED")
    try:
        observed = _file_reference(reference["path"])
    except (OSError, TypeError, ValueError):
        raise ValueError("NATIVE_BACKEND_ADMISSION_EVIDENCE_CHANGED") from None
    if observed != reference:
        raise ValueError("NATIVE_BACKEND_ADMISSION_EVIDENCE_CHANGED")
    return Path(observed["path"]).read_bytes()


def _evidence_references(value: Any):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            yield value
        else:
            for nested in value.values():
                yield from _evidence_references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _evidence_references(nested)


def _result_version(result: dict[str, Any]) -> str:
    versioned = deepcopy(result)
    versioned.pop("result_version", None)
    return RESULT_VERSION_PREFIX + _canonical_hash(versioned)


def decide_native_backend_admission(measurement: dict[str, Any]) -> dict[str, Any]:
    """Return an admitted/not-admitted result from one frozen measurement.

    The decision fails closed.  Missing samples, identity drift, incomplete
    permission checks, or absent comparison axes are evidence against admission;
    none are filled in by the platform.
    """
    if not isinstance(measurement, dict) or measurement.get("schema_version") != MEASUREMENT_SCHEMA:
        raise ValueError("NATIVE_BACKEND_ADMISSION_MEASUREMENT_INVALID")
    configuration_identity = measurement.get("configuration_identity")
    if not isinstance(configuration_identity, str) or not configuration_identity:
        raise ValueError("NATIVE_BACKEND_ADMISSION_CONFIGURATION_IDENTITY_REQUIRED")
    thresholds = measurement.get("thresholds")
    if thresholds != {
        "consecutive_commands": 3,
        "first_output_median_seconds_max": 15.0,
        "first_output_single_seconds_max": 30.0,
    }:
        raise ValueError("NATIVE_BACKEND_ADMISSION_THRESHOLDS_CHANGED")

    raw_trials = measurement.get("admission_trials")
    trials: list[Any] = raw_trials if isinstance(raw_trials, list) else []
    valid_trials: list[dict[str, Any]] = []
    if trials:
        for expected_sequence, trial in enumerate(trials, 1):
            if not (
                isinstance(trial, dict)
                and trial.get("sequence") == expected_sequence
                and trial.get("configuration_identity") == configuration_identity
                and trial.get("timeout") is False
                and trial.get("exit_code") == 0
                and _finite_seconds(trial.get("first_output_seconds"))
            ):
                break
            valid_trials.append(trial)
    complete = len(valid_trials) == 3 and len(trials) == 3
    seconds = [float(trial["first_output_seconds"]) for trial in valid_trials]
    median = float(statistics.median(seconds)) if seconds else None
    maximum = max(seconds) if seconds else None
    timing_passed = bool(
        complete and median is not None and maximum is not None
        and median <= 15.0 and maximum <= 30.0
    )

    reasons: list[str] = []
    if not complete:
        reasons.append("THREE_CONSECUTIVE_COMMANDS_NOT_ESTABLISHED")
    else:
        if median is not None and median > 15.0:
            reasons.append("FIRST_OUTPUT_MEDIAN_LIMIT_EXCEEDED")
    if maximum is not None and maximum > 30.0:
        reasons.append("FIRST_OUTPUT_SINGLE_LIMIT_EXCEEDED")

    permission_matrix = measurement.get("permission_matrix")
    matrix_passed = False
    if isinstance(permission_matrix, dict):
        required = permission_matrix.get("required_checks")
        checks = permission_matrix.get("checks")
        if isinstance(required, list) and required and len(required) == len(set(required)) and isinstance(checks, list):
            observed = [check.get("name") for check in checks if isinstance(check, dict)]
            matrix_passed = bool(
                permission_matrix.get("passed") is True
                and observed == required
                and all(check.get("passed") is True for check in checks)
            )
    if not matrix_passed:
        reasons.append("PERMISSION_MATRIX_NOT_ESTABLISHED")

    axes = measurement.get("comparison_axes")
    required_axes = ("permission_scale", "cold_vs_repeated", "cmd_vs_powershell")
    comparisons_complete = bool(
        isinstance(axes, dict)
        and set(axes) == set(required_axes)
        and all(isinstance(axes[name], dict) and axes[name].get("status") == "measured" for name in required_axes)
    )
    if not comparisons_complete:
        reasons.append("REQUIRED_STARTUP_COMPARISON_NOT_ESTABLISHED")

    admitted = timing_passed and matrix_passed and comparisons_complete
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "decision": "admitted" if admitted else "not_admitted",
        "business_runs_allowed": admitted,
        "configuration_identity": configuration_identity,
        "plan_ref": deepcopy(measurement.get("plan_ref")),
        "permission_profile_ref": deepcopy(measurement.get("permission_profile_ref")),
        "thresholds": deepcopy(thresholds),
        "timing": {
            "complete_consecutive_samples": complete,
            "first_output_seconds": seconds,
            "median_seconds": median,
            "maximum_seconds": maximum,
            "passed": timing_passed,
        },
        "admission_trials": deepcopy(trials),
        "permission_matrix": deepcopy(permission_matrix),
        "comparison_axes": deepcopy(axes),
        "reasons": reasons,
    }
    result["result_version"] = _result_version(result)
    return result


def freeze_native_backend_admission(
    measurement_ref: dict[str, str], output_path: str | Path
) -> dict[str, str]:
    """Verify all cited evidence and create one immutable admission artifact."""
    try:
        measurement = json.loads(_read_reference(measurement_ref))
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("NATIVE_BACKEND_ADMISSION_MEASUREMENT_INVALID") from None
    for reference in _evidence_references(measurement):
        _read_reference(reference)
    result = decide_native_backend_admission(measurement)
    result["measurement_ref"] = deepcopy(measurement_ref)
    result["result_version"] = _result_version(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return _file_reference(path)


def require_native_backend_admission(
    admission_ref: dict[str, str], *, plan_ref: dict[str, str],
    permission_profile_ref: dict[str, str], configuration_identity: str,
) -> dict[str, Any]:
    """Read a frozen result and fail before business execution unless admitted."""
    try:
        result = json.loads(_read_reference(admission_ref))
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("NATIVE_BACKEND_ADMISSION_RESULT_INVALID") from None
    if (
        not isinstance(result, dict)
        or result.get("schema_version") != RESULT_SCHEMA
        or result.get("result_version") != _result_version(result)
        or result.get("plan_ref") != plan_ref
        or result.get("permission_profile_ref") != permission_profile_ref
        or result.get("configuration_identity") != configuration_identity
    ):
        raise ValueError("NATIVE_BACKEND_ADMISSION_RESULT_INVALID")
    for reference in _evidence_references(result):
        _read_reference(reference)
    try:
        measurement = json.loads(_read_reference(result.get("measurement_ref")))
    except (json.JSONDecodeError, UnicodeError, TypeError, ValueError):
        raise ValueError("NATIVE_BACKEND_ADMISSION_RESULT_INVALID") from None
    for reference in _evidence_references(measurement):
        _read_reference(reference)
    # Scope: frozen admission is a cache of the deterministic decision, never an authority that can sign itself.
    # 为什么要有它：有效哈希只证明文件未变，必须从权威测量重算才能拒绝伪造的准入结论。
    expected = decide_native_backend_admission(measurement)
    expected["measurement_ref"] = deepcopy(result["measurement_ref"])
    expected["result_version"] = _result_version(expected)
    if result != expected:
        raise ValueError("NATIVE_BACKEND_ADMISSION_RESULT_INVALID")
    if result.get("decision") != "admitted" or result.get("business_runs_allowed") is not True:
        raw_reasons = result.get("reasons")
        reasons: list[Any] = raw_reasons if isinstance(raw_reasons, list) else []
        raise ValueError("NATIVE_BACKEND_NOT_ADMITTED:" + ",".join(str(reason) for reason in reasons))
    return result
