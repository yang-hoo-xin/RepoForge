"""Shared, stateless candidate execution primitives; no loop or retry policy.

Scope: exact bytes, one evaluator invocation, and its persisted report binding.
为什么要有它：不同模型编排必须共用同一候选核验边界，不能复制另一套较宽松的执行成功条件。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.core.trace.execution_trace import trace_operation
from src.mod_agent.execution.patch_apply import MaterializedPatchBytes, materialize_exact_text_patch


class CandidateExecutionError(RuntimeError):
    """Retain the original gate code and rejected evidence, never grant a retry."""

    def __init__(self, code: str, *, rejected_report: dict[str, Any] | None = None,
                 dispatch_started: bool | None = None):
        super().__init__(code)
        self.code = code
        self.rejected_report = deepcopy(rejected_report)
        self.dispatch_started = dispatch_started
        self.automatic_retry_allowed = False


def materialize_candidate(*, workspace_root: str | Path,
                          implementation_plan: dict[str, Any],
                          changes: Sequence[Mapping[str, Any]]) -> MaterializedPatchBytes:
    """Reuse the formal applier's no-write byte materialization and trace span."""
    changes_hash = hashlib.sha256(json.dumps(changes, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    with trace_operation("patch.materialize", attributes={"generated_changes_sha256": changes_hash}) as operation:
        result = materialize_exact_text_patch(workspace_root=workspace_root,
            diff_intent=changes, implementation_plan=implementation_plan)
        operation.update(output_code_sha256=result.code_sha256, baseline_sha256=result.baseline_sha256,
                         file_count=len(result.files))
        return result


def materialized_code_identity(materialized: MaterializedPatchBytes) -> dict[str, Any]:
    return {"baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256,
            "files": sorted(materialized.files)}


def verification_identity(evaluator: Any) -> dict[str, Any] | None:
    """Keep the existing optional identity contract; absence does not qualify reuse."""
    reader = getattr(evaluator, "verification_identity", None)
    if not callable(reader):
        return None
    value = reader()
    if not isinstance(value, Mapping):
        raise CandidateExecutionError("PATCH_VERIFICATION_IDENTITY_INVALID", dispatch_started=False)
    identity = json.loads(json.dumps(dict(value), allow_nan=False))
    if any(type(identity.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", identity[key]) is None
           for key in ("validator_sha256", "inputs_sha256", "environment_sha256")):
        raise CandidateExecutionError("PATCH_VERIFICATION_IDENTITY_INVALID", dispatch_started=False)
    if type(identity.get("deterministic_reuse_allowed")) is not bool:
        raise CandidateExecutionError("PATCH_VERIFICATION_IDENTITY_INVALID", dispatch_started=False)
    return identity


def invoke_candidate_evaluator(evaluator: Callable[[dict[str, Any]], Any],
                               context: Mapping[str, Any]) -> dict[str, Any]:
    """Call the selected owner exactly once, without sharing mutable model inputs.

    Scope: invocation only; callers retain snapshot-finally and before/after verifier checks.
    为什么要有它：构造输入失败与调用后未知结果必须可区分，两者都不能由本函数自动重跑或声称领域通过。
    """
    dispatched = False
    try:
        detached = json.loads(json.dumps(dict(context), allow_nan=False))
        dispatched = True
        report = evaluator(detached)
        if not isinstance(report, Mapping):
            raise ValueError("Behavior evaluator must return a structured observation.")
        return json.loads(json.dumps(dict(report), allow_nan=False))
    except Exception as exc:
        rejected = {
            "status": "infrastructure_failed", "admission": "rejected", "failure_owner": "unknown",
            "message": str(exc), "generated_changes_sha256": context["generated_changes_sha256"],
            "model_response_sha256": context["model_response"]["output_sha256"],
        }
        raise CandidateExecutionError("PATCH_BEHAVIOR_EVALUATOR_FAILED", rejected_report=rejected,
                                      dispatch_started=dispatched) from exc


def validate_candidate_report(report: Mapping[str, Any], *, changes_sha256: str,
                              source_snapshot_sha256: str,
                              allowed_binding_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Admit an owner report using the existing exact candidate/source/file checks.

    This does not execute the callback, authorize Apply, or certify whole-task delivery.
    为什么要有它：报告准入只证明已检查的绑定关系，历史unknown执行来源不能被提取重构升级为真实执行。
    """
    admitted = deepcopy(dict(report))
    statuses = {"passed", "behavior_failed", "infrastructure_failed", "identity_failed", "needs_evidence"}
    owners = {None, "patch_implementation", "validator", "environment", "evidence", "planning", "unknown"}
    try:
        if (admitted.get("status") not in statuses or admitted.get("failure_owner") not in owners
                or admitted.get("generated_changes_sha256") != changes_sha256
                or admitted.get("source_snapshot_sha256") != source_snapshot_sha256
                or not isinstance(admitted.get("observations"), list)
                or ("failure_state_observations" in admitted
                    and not isinstance(admitted["failure_state_observations"], list))
                or not isinstance(admitted.get("owner_binding_ids"), list)):
            raise ValueError("Behavior observation is not bound to the current input.")
        if any(type(item) is not str or item not in allowed_binding_ids for item in admitted["owner_binding_ids"]):
            raise ValueError("Behavior owner is outside the selected Plan bindings.")
        reference = admitted.get("report_ref")
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("Behavior observation needs an exact persisted report reference.")
        raw = Path(reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Behavior report bytes differ from their bound hash.")
        persisted = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(persisted, dict) or any(persisted.get(key) != admitted.get(key) for key in (
                "status", "failure_owner", "owner_binding_ids", "observations",
                "generated_changes_sha256", "source_snapshot_sha256", "failure_state_observations")):
            raise ValueError("Behavior report content differs from the returned observation.")
        if admitted["status"] == "behavior_failed" and not admitted["observations"]:
            raise ValueError("Behavior failure has no executable observations.")
        if admitted["status"] == "passed" and admitted["failure_owner"] is not None:
            raise ValueError("A passed report cannot still declare a failure owner.")
        origin = admitted.get("execution_origin", "unknown")
        if origin not in {"executed", "incomplete", "unknown"}:
            raise ValueError("Only the execution owner may produce a verified reuse record.")
        if origin != persisted.get("execution_origin", "unknown"):
            raise ValueError("Validation execution origin differs from its persisted report.")
        if admitted["status"] == "passed" and origin == "incomplete":
            raise ValueError("A passed behavior report cannot declare incomplete execution.")
        admitted.update(admission="accepted", execution_origin=origin)
        return admitted, persisted
    except (ValueError, TypeError, KeyError, OSError) as exc:
        admitted.update(admission="rejected", admission_error=str(exc))
        raise CandidateExecutionError("PATCH_BEHAVIOR_OBSERVATION_INVALID",
                                      rejected_report=admitted) from exc
