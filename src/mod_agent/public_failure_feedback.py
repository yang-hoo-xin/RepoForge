"""Bounded public observations with retained, readable original details.

为什么要有它：失败名称和反例必须到达模型，完整来源仍受候选身份与公开报告范围约束。
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from src.mod_agent.model_qualification import read_bound_json


def _bounded(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return deepcopy(value)
    return {"preview": text[:limit], "truncated": True, "total_chars": len(text)}


def _locations(details, source_root):
    root = Path(source_root).resolve()
    rows = []
    for match in re.finditer(r"(?:\bin\s+)([^\r\n]+?):line\s+(\d+)", details):
        path = Path(match.group(1).strip())
        if not path.is_absolute():
            continue
        resolved = path.resolve()
        if resolved.is_relative_to(root) and resolved.is_file():
            row = {"path": resolved.relative_to(root).as_posix(), "line": int(match.group(2)),
                "origin": "reported_public_stack_trace"}
            if row not in rows:
                rows.append(row)
    return rows[:8]


def project_failed_checks(checks, *, raw_report_ref, source_root):
    result = []
    for row in checks:
        if row["passed"]:
            continue
        projected = {"check_id": row["check_id"], "expected": _bounded(row["expected"], 2000),
            "observed": {"actual": _bounded(row["actual"], 2000), "passed": False},
            "raw_report_ref": deepcopy(raw_report_ref)}
        if "name" in row:
            projected["test_name"] = _bounded(row["name"], 1000)
        if "details" in row:
            details = row["details"]
            projected.update(details=details[:2000], details_truncated=len(details) > 2000,
                details_total_chars=len(details), source_locations=_locations(details, source_root))
        result.append(projected)
    return result


def failed_checks_text(checks):
    return json.dumps([row for row in checks if not row["passed"]], ensure_ascii=False, sort_keys=True, indent=2)


def validated_public_checks(reference):
    """Reparse existing public originals; no invented details or rerun.

    为什么要有它：旧报告重投影也要核验源码、固定输入及真实测试原件，不能只信汇总状态。
    """
    from src.mod_agent.domain.csharp_repository_backend import (
        CSharpRepositoryProfile,
        _read_tests,
        _snapshot,
    )
    from src.mod_agent.domain.full_project_compilation import _ref

    report = read_bound_json(reference)
    profile = CSharpRepositoryProfile(report["profile_ref"])
    source = Path(report["source_root"]).resolve(strict=True)
    profile.verify_fixed_inputs(source, report["fixed_inputs"])
    if _snapshot(source) != report["source_inputs"]:
        raise ValueError("PUBLIC_FEEDBACK_SOURCE_CHANGED")
    execution = report["test_execution"]
    compilation = read_bound_json(report["compilation_report_ref"])
    if (report.get("execution_origin") != "executed" or report.get("status") not in {"passed", "behavior_failed"}
            or report["build_execution"].get("restore_exit_code") != 0
            or report["build_execution"].get("build_exit_code") != 0
            or compilation.get("source_inputs") != report["source_inputs"]
            or not report.get("built_artifact_refs")
            or compilation.get("built_artifact_refs") != report["built_artifact_refs"]
            or any(_ref(row["path"]) != row for row in report["built_artifact_refs"])
            or execution.get("exit_code") not in (0, 1)):
        raise ValueError("PUBLIC_FEEDBACK_EXECUTION_UNCONFIRMED")
    raw = execution.get("report_ref")
    if execution.get("status") != "executed" or not raw:
        raise ValueError("PUBLIC_FEEDBACK_TESTS_NOT_EXECUTED")
    path = Path(raw["path"]).resolve(strict=True)
    if not path.is_relative_to(Path(reference["path"]).parent / "test-output") or _ref(path) != raw:
        raise ValueError("PUBLIC_FEEDBACK_REPORT_CHANGED")
    checks = _read_tests(path, profile.data["test_report"]["format"])
    if checks != report["checks"]:
        raise ValueError("PUBLIC_FEEDBACK_CHECKS_CHANGED")
    if report["verification_summary"] != {"checks": len(checks), "failed_checks": sum(not row["passed"] for row in checks),
            "executions": 1, "test_command_executed": True}:
        raise ValueError("PUBLIC_FEEDBACK_SUMMARY_CHANGED")
    if report.get("integrity_ref"):
        integrity = read_bound_json(report["integrity_ref"])
        if integrity.get("status") != "validated":
            raise ValueError("PUBLIC_FEEDBACK_INTEGRITY_UNCONFIRMED")
    return report, checks


def project_saved_candidate_failure(*, report_ref, state, candidate_ref, original_execution_ref=None):
    from src.mod_agent.task_agent_execution import _worker_binding

    if original_execution_ref is None:
        raise ValueError("PUBLIC_FEEDBACK_ORIGINAL_EXECUTION_REQUIRED")
    native = read_bound_json(original_execution_ref)
    if (native.get("schema_version") != "task-agent-product-execution.v2"
            or native.get("task_contract_ref") != state.contract_ref or native.get("candidate_ref") != candidate_ref):
        raise ValueError("PUBLIC_FEEDBACK_ORIGINAL_BINDING_CHANGED")
    binding = native["worker_binding"]
    worker_ref = binding["configuration_ref"]
    if _worker_binding(worker_ref) != binding:
        raise ValueError("PUBLIC_FEEDBACK_WORKER_CHANGED")
    selected = read_bound_json(worker_ref)["behavior_evaluator"]
    if (selected["factory_name"] != "make_csharp_repository_evaluator"
            or selected["kwargs"]["profile_ref"] != state.contract["execution_profile_ref"]):
        raise ValueError("PUBLIC_FEEDBACK_ORIGINAL_PROFILE_CHANGED")
    candidate = state.validate_candidate(candidate_ref)
    report, checks = validated_public_checks(report_ref)
    # 为什么要有它：公开来源由原任务冻结的完整输入集合界定，不能让报告自带的额外测试进入反馈。
    frozen = selected["kwargs"]["expected_fixed_inputs"]
    if (report["fixed_inputs"] != frozen
            or set(row["path"] for row in report["source_inputs"]) != set(frozen["source_paths"])):
        raise ValueError("PUBLIC_FEEDBACK_INPUTS_CHANGED")
    if report["profile_ref"] != state.contract["execution_profile_ref"]:
        raise ValueError("PUBLIC_FEEDBACK_PROFILE_CHANGED")
    source = Path(report["source_root"])
    if any((source / path).read_bytes() != text.encode("utf-8")
           for path, text in state.materialize(candidate_ref).items()):
        raise ValueError("PUBLIC_FEEDBACK_CANDIDATE_CHANGED")
    failed = project_failed_checks(checks, raw_report_ref=report["test_execution"]["report_ref"], source_root=source)
    return {"candidate_ref": candidate_ref, "candidate_source_id": candidate["candidate_source_id"],
        "report_ref": report_ref, "observations": failed[:8], "failed_checks": len(failed),
        "observations_truncated": len(failed) > 8, "complete_public_details": failed_checks_text(checks),
        "scope": "saved_public_report_projection_not_model_execution_or_delivery"}
