"""Typed Fresh Artifact continuation into the existing Planning graph node."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from src.mod_agent.artifacts import (
    ArtifactRef,
    ArtifactRegistryError,
    ArtifactStore,
    ArtifactStoreError,
    ArtifactWriteResult,
    StaleArtifactError,
)
from src.mod_agent.bindings import BindingResolver
from src.mod_agent.evidence import EvidencePack
from src.mod_agent.evidence.compilation_candidate import compilation_candidate_scope
from src.mod_agent.evidence.identity import feature_spec_payload_sha256
from src.mod_agent.fresh_coverage import (
    FRESH_COVERAGE_MODEL_CALL_BUDGET,
    strict_feature_spec_payload,
)

PLANNING_CONTINUATION_CONTRACT_VERSION = "fresh_planning_continuation_v1"
PLANNING_CONTINUATION_RESULT_TYPE = "planning_continuation_result_v1"
PLANNING_CONSUMER_STAGE = "create_implementation_plan"
PATCH_CONTINUATION_CONTRACT_VERSION = "fresh_patch_continuation_v1"
PATCH_CONTINUATION_RESULT_TYPE = "patch_continuation_result_v1"
PATCH_CONTINUATION_RECEIPT_VERSION = "patch_continuation_attempt_v1"
PATCH_CONSUMER_STAGE = "generate_patch_proposal"


class PlanningContinuationError(ValueError):
    """Reject an invalid continuation before a later product stage can run."""

    def __init__(self, code: str, stage: str, reason: str) -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage
        self.reason = reason


class PlanningContinuationArtifactWriteError(RuntimeError):
    """Carry an exact non-committed Store result out of the Planning node."""

    def __init__(self, write_result: ArtifactWriteResult) -> None:
        super().__init__("Planning continuation Artifact write did not commit.")
        self.write_result = write_result


class PatchContinuationError(ValueError):
    """Reject an invalid exact Plan continuation before unsafe progress."""

    def __init__(self, code: str, stage: str, reason: str) -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class PlanningContinuationInputV1:
    """Complete identity required to continue two Fresh revisions."""

    contract_version: Literal["fresh_planning_continuation_v1"]
    artifact_store_root: str
    workspace_root: str
    feature_spec_ref: ArtifactRef
    evidence_pack_ref: ArtifactRef
    planning_target_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.contract_version != PLANNING_CONTINUATION_CONTRACT_VERSION:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_VERSION_INVALID",
                "request",
                "Planning continuation contract version is invalid.",
            )
        for value, name in (
            (self.artifact_store_root, "artifact_store_root"),
            (self.workspace_root, "workspace_root"),
        ):
            if type(value) is not str or not value.strip():
                raise PlanningContinuationError(
                    "PLANNING_CONTINUATION_PATH_INVALID",
                    "request",
                    f"Planning continuation {name} must be non-empty text.",
                )
        _validate_artifact_ref(
            self.feature_spec_ref,
            field_name="feature_spec_ref",
            expected_type="feature_spec",
        )
        _validate_artifact_ref(
            self.evidence_pack_ref,
            field_name="evidence_pack_ref",
            expected_type="evidence_pack",
        )
        if self.feature_spec_ref.feature_id != self.evidence_pack_ref.feature_id:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_FEATURE_MISMATCH",
                "request",
                "Fresh FeatureSpec and EvidencePack refs have different feature_id values.",
            )


@dataclass(frozen=True)
class PatchContinuationInputV1:
    """Exact immutable Artifact selection for one Patch-to-Dry-run continuation."""

    contract_version: Literal["fresh_patch_continuation_v1"]
    artifact_store_root: str
    workspace_root: str
    feature_spec_ref: ArtifactRef
    evidence_pack_ref: ArtifactRef
    implementation_plan_ref: ArtifactRef
    repair_parent_proposal_ref: ArtifactRef | None = None
    repair_parent_review_ref: ArtifactRef | None = None
    repair_failure_report_ref: Mapping[str, str] | None = None
    repair_parent_response_ref: Mapping[str, str] | None = None
    repair_parent_attempt_ref: Mapping[str, str] | None = None
    attempt_id: str | None = None
    worker_configuration_ref: Mapping[str, str] | None = None
    worker_profile_sha256: str | None = None
    repair_source_attempt_ref: Mapping[str, str] | None = None
    repair_verification_ref: Mapping[str, str] | None = None
    qualified_model_routing_ref: Mapping[str, str] | None = None
    patch_upstream_configuration_ref: Mapping[str, str] | None = None
    saved_patch_selection: Mapping[str, Any] | None = None
    candidate_reverification_selection: Mapping[str, Any] | None = None

    @property
    def reuses_model_output(self) -> bool:
        return self.saved_patch_selection is not None or self.candidate_reverification_selection is not None

    def __post_init__(self) -> None:
        if self.candidate_reverification_selection is not None:
            # 为什么要有它：未通过验收的候选与历史已通过补丁分开准入，精确来源和当前检查器不能被模型修订或隐式重选。
            selected = self.candidate_reverification_selection
            if (self.saved_patch_selection is not None or self.repair_parent_proposal_ref is not None
                    or self.patch_upstream_configuration_ref is not None or not self.attempt_id
                    or not isinstance(selected, Mapping)
                    or set(selected) != {"source_attempt_ref", "parent_manifest_ref", "parent_result_ref"}
                    or self.worker_configuration_ref is None):
                raise PatchContinuationError("SAVED_CANDIDATE_SELECTION_INVALID", "request", "Exact retained candidate and new verifier selection required.")
            for reference in selected.values():
                if (not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}
                        or not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute()
                        or not _is_sha256(reference["sha256"])):
                    raise PatchContinuationError("SAVED_CANDIDATE_SELECTION_INVALID", "request", "Exact candidate reference required.")
            object.__setattr__(self, "candidate_reverification_selection", deepcopy(dict(selected)))
        if self.contract_version != PATCH_CONTINUATION_CONTRACT_VERSION:
            raise PatchContinuationError(
                "PATCH_CONTINUATION_VERSION_INVALID",
                "request",
                "Patch continuation contract version is invalid.",
            )
        for value, name in (
            (self.artifact_store_root, "artifact_store_root"),
            (self.workspace_root, "workspace_root"),
        ):
            if type(value) is not str or not value.strip():
                raise PatchContinuationError(
                    "PATCH_CONTINUATION_PATH_INVALID",
                    "request",
                    f"Patch continuation {name} must be non-empty text.",
                )
        for ref, field_name, artifact_type in (
            (self.feature_spec_ref, "feature_spec_ref", "feature_spec"),
            (self.evidence_pack_ref, "evidence_pack_ref", "evidence_pack"),
            (
                self.implementation_plan_ref,
                "implementation_plan_ref",
                "implementation_plan",
            ),
        ):
            try:
                _validate_artifact_ref(
                    ref,
                    field_name=field_name,
                    expected_type=artifact_type,
                )
            except PlanningContinuationError as exc:
                raise PatchContinuationError(
                    "PATCH_CONTINUATION_REF_INVALID",
                    "request",
                    exc.reason,
                ) from exc
        feature_ids = {
            self.feature_spec_ref.feature_id,
            self.evidence_pack_ref.feature_id,
            self.implementation_plan_ref.feature_id,
        }
        if len(feature_ids) != 1:
            raise PatchContinuationError(
                "PATCH_CONTINUATION_FEATURE_MISMATCH",
                "request",
                "Selected FeatureSpec, EvidencePack, and Plan feature_id values differ.",
            )
        recovery_fields = (
            self.repair_parent_proposal_ref, self.repair_parent_review_ref,
            self.repair_failure_report_ref, self.repair_parent_response_ref,
            self.repair_parent_attempt_ref,
        )
        if any(value is not None for value in recovery_fields):
            if any(value is None for value in recovery_fields):
                raise PatchContinuationError(
                    "PATCH_REPAIR_CONTEXT_INCOMPLETE", "request",
                    "Repair requires the exact parent Proposal, Review, response, attempt, and failure report.",
                )
            if self.attempt_id is None:
                raise PatchContinuationError("PATCH_REPAIR_ATTEMPT_ID_REQUIRED", "request", "Repair execution requires a stable attempt identity for duplicate-call prevention.")
            for ref, kind in ((self.repair_parent_proposal_ref, "patch_proposal"),
                              (self.repair_parent_review_ref, "proposal_review")):
                try:
                    _validate_artifact_ref(ref, field_name=kind, expected_type=kind)
                except PlanningContinuationError as exc:
                    raise PatchContinuationError("PATCH_REPAIR_REF_INVALID", "request", exc.reason) from exc
                if ref.feature_id != self.feature_spec_ref.feature_id:
                    raise PatchContinuationError("PATCH_REPAIR_FEATURE_MISMATCH", "request", "Repair parent belongs to another FeatureSpec.")
        if self.attempt_id is not None and (
            type(self.attempt_id) is not str or not self.attempt_id.strip()
            or len(self.attempt_id) > 200
        ):
            raise PatchContinuationError("PATCH_CONTINUATION_ATTEMPT_ID_INVALID", "request", "Attempt identity must be non-empty text of at most 200 characters.")
        _validate_worker_configuration_identity(self.worker_configuration_ref, self.worker_profile_sha256)
        if self.qualified_model_routing_ref is not None:
            ref = self.qualified_model_routing_ref
            if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
                    or not isinstance(ref.get("path"), str) or not Path(ref["path"]).is_absolute()
                    or not _is_sha256(ref.get("sha256"))):
                raise PatchContinuationError("PATCH_MODEL_ROUTING_REF_INVALID", "request", "Exact model routing reference required.")
            object.__setattr__(self, "qualified_model_routing_ref", dict(ref))
        if self.worker_configuration_ref is not None:
            # Detach the caller's mutable mapping while preserving its exact values.
            object.__setattr__(self, "worker_configuration_ref", dict(self.worker_configuration_ref))
        if self.patch_upstream_configuration_ref is not None:
            ref = self.patch_upstream_configuration_ref
            if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
                    or not isinstance(ref.get("path"), str) or not Path(ref["path"]).is_absolute()
                    or not _is_sha256(ref.get("sha256"))):
                raise PatchContinuationError("PATCH_UPSTREAM_CONFIGURATION_INVALID", "request", "Exact upstream configuration reference required.")
            _read_hashed_repair_file(ref)
            object.__setattr__(self, "patch_upstream_configuration_ref", dict(ref))
        if (self.repair_source_attempt_ref is None) != (self.repair_verification_ref is None):
            raise PatchContinuationError("PATCH_REPAIR_BINDING_INPUT_INVALID", "request",
                "A repair source attempt and separate verification reference are required together.")
        if self.repair_source_attempt_ref is not None and self.repair_parent_proposal_ref is None:
            raise PatchContinuationError("PATCH_REPAIR_BINDING_INPUT_INVALID", "request",
                "Cross-attempt repair must retain its explicitly selected formal parent lineage.")
        for name in ("repair_source_attempt_ref", "repair_verification_ref"):
            ref = getattr(self, name)
            if ref is not None:
                if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
                        or type(ref.get("path")) is not str or not Path(ref["path"]).is_absolute()
                        or not _is_sha256(ref.get("sha256"))):
                    raise PatchContinuationError("PATCH_REPAIR_BINDING_INPUT_INVALID", "request",
                        "Repair binding sources require exact absolute paths and lowercase SHA-256 values.")
                object.__setattr__(self, name, dict(ref))
        if self.saved_patch_selection is not None:
            selection = self.saved_patch_selection
            if (not isinstance(selection, Mapping)
                    or set(selection) != {"source_attempt_ref", "logical_budget_ref"}
                    or self.attempt_id is None
                    or any(value is not None for value in (
                        *recovery_fields, self.repair_source_attempt_ref, self.repair_verification_ref,
                        self.worker_configuration_ref, self.worker_profile_sha256,
                        self.qualified_model_routing_ref, self.patch_upstream_configuration_ref))):
                raise PatchContinuationError("PATCH_SAVED_SELECTION_INVALID", "request",
                    "Saved Patch reassembly requires its exact selection and attempt ID, without model or Repair inputs.")
            for ref in selection.values():
                if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
                        or type(ref.get("path")) is not str or not Path(ref["path"]).is_absolute()
                        or not _is_sha256(ref.get("sha256"))):
                    raise PatchContinuationError("PATCH_SAVED_SELECTION_INVALID", "request",
                        "Saved Patch inputs require exact absolute paths and SHA-256 values.")
            object.__setattr__(self, "saved_patch_selection",
                {key: dict(value) for key, value in selection.items()})


@dataclass(frozen=True)
class PlanningContinuationResultV1:
    """Stable public result for one bounded Fresh-to-Planning continuation."""

    status: Literal["terminal", "blocked", "partial"]
    terminal_stage: str
    first_failure: dict[str, str] | None
    run_id: str
    request_id: str
    feature_id: str | None
    feature_spec_artifact: dict[str, str] | None
    evidence_pack_artifact: dict[str, str] | None
    implementation_plan_artifact: dict[str, str] | None
    plan_blocking: bool | None
    artifact_write: dict[str, Any] | None
    source_validation: dict[str, Any]
    planning_execution_count: int
    source_modified_during_run: bool | None
    duration_seconds: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_type": PLANNING_CONTINUATION_RESULT_TYPE,
            "status": self.status,
            "command": "continue-fresh-planning",
            "terminal_stage": self.terminal_stage,
            "first_failure": (None if self.first_failure is None else dict(self.first_failure)),
            "run_id": self.run_id,
            "request_id": self.request_id,
            "feature_id": self.feature_id,
            "feature_spec_artifact": (
                None if self.feature_spec_artifact is None else dict(self.feature_spec_artifact)
            ),
            "evidence_pack_artifact": (
                None if self.evidence_pack_artifact is None else dict(self.evidence_pack_artifact)
            ),
            "implementation_plan_artifact": (
                None
                if self.implementation_plan_artifact is None
                else dict(self.implementation_plan_artifact)
            ),
            "plan_blocking": self.plan_blocking,
            "artifact_write": (None if self.artifact_write is None else dict(self.artifact_write)),
            "source_validation": dict(self.source_validation),
            "stopped_before": "patch_proposal",
            "execution_counts": {
                "fresh_coverage": 0,
                "model": 0,
                "planning": self.planning_execution_count,
                "patch": 0,
                "apply": 0,
                "build": 0,
                "runtime": 0,
            },
            "product_source_writes_performed": False,
            "source_modified_during_run": self.source_modified_during_run,
            "duration_seconds": self.duration_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PatchContinuationResultV1:
    """Public result for exact Plan through deterministic no-write Dry-run."""

    status: Literal["terminal", "blocked", "partial"]
    terminal_stage: str
    first_failure: dict[str, str] | None
    run_id: str
    request_id: str
    feature_id: str | None
    feature_spec_artifact: dict[str, str] | None
    evidence_pack_artifact: dict[str, str] | None
    implementation_plan_artifact: dict[str, str] | None
    patch_proposal_artifact: dict[str, str] | None
    proposal_review_artifact: dict[str, str] | None
    dry_run_artifact: dict[str, str] | None
    source_validation: dict[str, Any]
    execution_counts: dict[str, int]
    attempt_receipt: dict[str, Any] | None
    source_modified_during_run: bool | None
    duration_seconds: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_type": PATCH_CONTINUATION_RESULT_TYPE,
            "status": self.status,
            "command": "continue-fresh-patch",
            "terminal_stage": self.terminal_stage,
            "first_failure": (
                None if self.first_failure is None else dict(self.first_failure)
            ),
            "run_id": self.run_id,
            "request_id": self.request_id,
            "feature_id": self.feature_id,
            "feature_spec_artifact": self.feature_spec_artifact,
            "evidence_pack_artifact": self.evidence_pack_artifact,
            "implementation_plan_artifact": self.implementation_plan_artifact,
            "patch_proposal_artifact": self.patch_proposal_artifact,
            "proposal_review_artifact": self.proposal_review_artifact,
            "dry_run_artifact": self.dry_run_artifact,
            "source_validation": dict(self.source_validation),
            "stopped_before": "exact_diff_approval",
            "execution_counts": dict(self.execution_counts),
            "attempt_receipt": (
                None if self.attempt_receipt is None else dict(self.attempt_receipt)
            ),
            "product_source_writes_performed": False,
            "source_modified_during_run": self.source_modified_during_run,
            "duration_seconds": self.duration_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _AdmittedPlanningInputs:
    artifact_store: ArtifactStore
    feature_spec: dict[str, Any]
    evidence_pack: dict[str, Any]
    source_validation: dict[str, Any]
    source_tree_snapshot: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _AdmittedPatchInputs:
    artifact_store: ArtifactStore
    feature_spec: dict[str, Any]
    evidence_pack: dict[str, Any]
    implementation_plan: dict[str, Any]
    source_validation: dict[str, Any]
    source_tree_snapshot: tuple[tuple[str, str], ...]
    latest_observations: dict[str, dict[str, Any]]
    repair_context: dict[str, Any] | None = None
    saved_patch: dict[str, Any] | None = None
    saved_candidate: dict[str, Any] | None = None


PatchContinuationFactory = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    tuple[dict[str, Any], ...],
]
PatchContinuationDryRunFactory = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]],
    dict[str, Any],
]


def _validate_artifact_ref(
    ref: Any,
    *,
    field_name: str,
    expected_type: str,
) -> None:
    if not isinstance(ref, ArtifactRef):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_REF_INVALID",
            "request",
            f"Planning continuation {field_name} must be ArtifactRef.",
        )
    if (
        type(ref.feature_id) is not str
        or not ref.feature_id
        or ref.artifact_type != expected_type
        or type(ref.path) is not str
        or not ref.path
        or not Path(ref.path).is_absolute()
        or not _is_sha256(ref.artifact_hash)
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_REF_INVALID",
            "request",
            f"Planning continuation {field_name} identity is invalid.",
        )


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_worker_configuration_identity(
    ref: Mapping[str, str] | None, profile_sha256: str | None, *, verify_bytes: bool = False,
) -> None:
    """Bind an optional worker configuration file without interpreting its contents."""
    if ref is None and profile_sha256 is None:
        return
    if ref is None or not _is_sha256(profile_sha256):
        raise PatchContinuationError(
            "PATCH_WORKER_CONFIGURATION_IDENTITY_INVALID", "request",
            "Worker configuration and its lowercase model-profile SHA-256 are required together.",
        )
    if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
            or type(ref.get("path")) is not str or not ref["path"].strip()
            or not _is_sha256(ref.get("sha256"))):
        raise PatchContinuationError(
            "PATCH_WORKER_CONFIGURATION_IDENTITY_INVALID", "request",
            "Worker configuration requires exactly a non-empty path and lowercase SHA-256.",
        )
    if verify_bytes:
        try:
            digest = hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise PatchContinuationError(
                "PATCH_WORKER_CONFIGURATION_IDENTITY_INVALID", "configuration_admission",
                "The exact worker configuration file is unavailable.",
            ) from exc
        if digest != ref["sha256"]:
            raise PatchContinuationError(
                "PATCH_WORKER_CONFIGURATION_IDENTITY_INVALID", "configuration_admission",
                "Worker configuration file bytes differ from the selected SHA-256.",
            )


def _admit_compilation_candidate(
    evidence_pack: dict[str, Any],
    feature_spec: dict[str, Any],
    *,
    worker_configuration_ref: Mapping[str, str] | None = None,
    require_selected_worker: bool = False,
) -> dict[str, Any] | None:
    """Recheck the candidate's exact worker configuration without invoking it.

    为什么要有它：源码绑定的候选准入依赖原检查器，续接不能换成另一配置或把未配置检查器的任务当作可验证候选。
    """
    scope = compilation_candidate_scope(evidence_pack, feature_spec)
    if scope is None:
        return None
    reference = scope["worker_configuration_ref"]
    if require_selected_worker and worker_configuration_ref != reference:
        raise ValueError("COMPILATION_CANDIDATE_WORKER_MISMATCH")
    return scope


def _same_path(left: Path, right: Path) -> bool:
    try:
        return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))
    except (OSError, RuntimeError):
        return False


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _artifact_ref_dict(ref: ArtifactRef) -> dict[str, str]:
    return {
        "feature_id": ref.feature_id,
        "artifact_type": ref.artifact_type,
        "path": ref.path,
        "artifact_hash": ref.artifact_hash,
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _artifact_write_projection(
    value: ArtifactWriteResult | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    raw = value.to_dict() if isinstance(value, ArtifactWriteResult) else dict(value)
    fields = (
        "outcome",
        "artifact_hash",
        "revision_path",
        "latest_alias_path",
        "commit_started",
        "revision_state",
        "latest_alias_state",
        "pending_marker_state",
        "failure_stage",
        "failure_class",
        "recovery_available",
    )
    return {field: raw.get(field) for field in fields}


def _resolve_roots(
    request: PlanningContinuationInputV1,
    *,
    allow_product_store_subdir: bool = False,
) -> tuple[Path, Path]:
    try:
        artifact_root = Path(request.artifact_store_root).resolve(strict=True)
        workspace_root = Path(request.workspace_root).resolve(strict=True)
    except OSError as exc:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_ROOT_INVALID",
            "artifact_admission",
            "Planning continuation roots must already exist.",
        ) from exc
    if not artifact_root.is_dir() or not workspace_root.is_dir():
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_ROOT_INVALID",
            "artifact_admission",
            "Planning continuation roots must be directories.",
        )
    if (
        _same_path(artifact_root, workspace_root)
        or (_is_within(artifact_root, workspace_root) and not allow_product_store_subdir)
        or _is_within(workspace_root, artifact_root)
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_ROOTS_OVERLAP",
            "artifact_admission",
            "ArtifactStore and product workspace roots must not overlap.",
        )
    return artifact_root, workspace_root


def _latest_artifact_ref(
    store: ArtifactStore,
    *,
    feature_id: str,
    artifact_type: str,
) -> ArtifactRef | None:
    latest = [
        ref
        for ref in store.list_feature_artifacts(feature_id)
        if ref.artifact_type == artifact_type
    ]
    if not latest:
        return None
    if len(latest) != 1:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_LATEST_ALIAS_INVALID",
            "artifact_admission",
            "ArtifactStore has multiple latest aliases for one Artifact type.",
        )
    store.read(latest[0])
    return latest[0]


def _read_exact_revision(
    store: ArtifactStore,
    ref: ArtifactRef,
) -> dict[str, Any]:
    try:
        revisions = store.list_feature_artifact_revisions(
            ref.feature_id,
            ref.artifact_type,
        )
        if ref not in revisions:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_EXACT_REVISION_REQUIRED",
                "artifact_admission",
                "Planning continuation requires an immutable Artifact revision ref.",
            )
        latest_ref = _latest_artifact_ref(
            store,
            feature_id=ref.feature_id,
            artifact_type=ref.artifact_type,
        )
        if latest_ref is None or latest_ref.artifact_hash != ref.artifact_hash:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_ARTIFACT_STALE",
                "artifact_admission",
                "Planning continuation requires the current latest Artifact revision.",
            )
        envelope = store.read_for_stage(
            ref,
            consumer_stage=PLANNING_CONSUMER_STAGE,
        )
        exact_freshness = store.check_exact_revision_freshness(ref)
    except PlanningContinuationError:
        raise
    except (
        ArtifactStoreError,
        StaleArtifactError,
        OSError,
        ValueError,
    ) as exc:
        code = (
            "PLANNING_CONTINUATION_ARTIFACT_STALE"
            if "stale" in type(exc).__name__.lower()
            else "PLANNING_CONTINUATION_ARTIFACT_INVALID"
        )
        raise PlanningContinuationError(
            code,
            "artifact_admission",
            "Planning continuation Artifact admission failed.",
        ) from exc
    if not exact_freshness.is_fresh:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_ARTIFACT_STALE",
            "artifact_admission",
            "Planning continuation exact revision is stale.",
        )
    return envelope


def _validate_upstream(
    evidence_envelope: Mapping[str, Any],
    feature_spec_ref: ArtifactRef,
) -> None:
    upstreams = evidence_envelope.get("inputs", {}).get("upstream_artifacts")
    if not isinstance(upstreams, list) or len(upstreams) != 1:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_UPSTREAM_INVALID",
            "artifact_admission",
            "Fresh EvidencePack must have exactly one FeatureSpec upstream.",
        )
    upstream = upstreams[0]
    if not isinstance(upstream, Mapping) or (
        upstream.get("artifact_type") != "feature_spec"
        or upstream.get("path") != feature_spec_ref.path
        or upstream.get("artifact_hash") != feature_spec_ref.artifact_hash
        or upstream.get("require_latest") is not True
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_UPSTREAM_MISMATCH",
            "artifact_admission",
            "Fresh EvidencePack is not bound to the selected FeatureSpec revision.",
        )


def _source_hash(item: Mapping[str, Any]) -> str:
    source_hash = item.get("source_hash")
    file_hash = item.get("file_hash")
    if source_hash not in (None, "") and not _is_sha256(source_hash):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_SOURCE_HASH_INVALID",
            "source_validation",
            "Target-source evidence has an invalid source_hash.",
        )
    if file_hash not in (None, "") and not _is_sha256(file_hash):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_SOURCE_HASH_INVALID",
            "source_validation",
            "Target-source evidence has an invalid file_hash.",
        )
    if source_hash and file_hash and source_hash != file_hash:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_SOURCE_HASH_CONFLICT",
            "source_validation",
            "Target-source evidence hash aliases disagree.",
        )
    selected = source_hash or file_hash
    if not isinstance(selected, str):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_SOURCE_HASH_MISSING",
            "source_validation",
            "Target-source evidence has no physical source hash.",
        )
    return selected


def _validate_source_items(
    evidence_pack: Mapping[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    raw_items = evidence_pack.get("items")
    if not isinstance(raw_items, list):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_EVIDENCE_ITEMS_INVALID",
            "source_validation",
            "Fresh EvidencePack items must be an array.",
        )
    target_items = [
        item
        for item in raw_items
        if isinstance(item, Mapping) and item.get("source_type") == "target_source"
    ]
    if not target_items:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_TARGET_SOURCE_MISSING",
            "source_validation",
            "Planning continuation found no target-source evidence.",
        )

    files: dict[str, dict[str, Any]] = {}
    for item in target_items:
        source_path = item.get("source_path")
        source_ref = item.get("source_ref")
        if type(source_path) is not str or not source_path.strip():
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_PATH_INVALID",
                "source_validation",
                "Target-source evidence source_path is invalid.",
            )
        pure_path = PurePosixPath(source_path)
        if pure_path.is_absolute() or Path(source_path).is_absolute() or ".." in pure_path.parts:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_PATH_INVALID",
                "source_validation",
                "Target-source evidence source_path escapes the workspace.",
            )
        expected_path = (workspace_root / Path(*pure_path.parts)).resolve()
        if not _is_within(expected_path, workspace_root) or not expected_path.is_file():
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_MISSING",
                "source_validation",
                "A target-source evidence file is unavailable.",
            )
        if (
            type(source_ref) is not str
            or not Path(source_ref).is_absolute()
            or not _same_path(Path(source_ref), expected_path)
        ):
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_REF_MISMATCH",
                "source_validation",
                "Target-source evidence source_ref does not match source_path.",
            )
        try:
            source_bytes = expected_path.read_bytes()
            line_count = len(source_bytes.decode("utf-8").splitlines())
        except (OSError, UnicodeDecodeError) as exc:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_READ_FAILED",
                "source_validation",
                "A target-source evidence file could not be read as UTF-8.",
            ) from exc
        expected_hash = _source_hash(item)
        observed_hash = hashlib.sha256(source_bytes).hexdigest()
        if observed_hash != expected_hash:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_HASH_MISMATCH",
                "source_validation",
                "A target-source evidence file has changed.",
            )
        span = item.get("span")
        if not isinstance(span, Mapping):
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_SPAN_INVALID",
                "source_validation",
                "Target-source evidence span is invalid.",
            )
        start_line = span.get("start_line")
        end_line = span.get("end_line")
        if (
            type(start_line) is not int
            or type(end_line) is not int
            or start_line < 1
            or end_line < start_line
            or end_line > line_count
        ):
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_SPAN_INVALID",
                "source_validation",
                "Target-source evidence span is outside the current source file.",
            )
        existing = files.setdefault(
            pure_path.as_posix(),
            {
                "source_path": pure_path.as_posix(),
                "sha256": observed_hash,
                "item_count": 0,
            },
        )
        if existing["sha256"] != observed_hash:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_SOURCE_HASH_CONFLICT",
                "source_validation",
                "One source path has conflicting evidence hashes.",
            )
        existing["item_count"] += 1

    file_rows = [files[path] for path in sorted(files)]
    return {
        "status": "passed",
        "checked_item_count": len(target_items),
        "checked_file_count": len(file_rows),
        "files": file_rows,
        "snapshot_sha256": _canonical_sha256(file_rows),
    }


def _csharp_source_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
    try:
        return tuple(
            (
                path.relative_to(root).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(root.rglob("*.cs"), key=lambda item: item.as_posix())
            if path.is_file()
        )
    except OSError as exc:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_SOURCE_SNAPSHOT_FAILED",
            "source_validation",
            "Current C# source snapshot could not be read.",
        ) from exc


def _validate_fresh_origin(evidence_pack: Mapping[str, Any]) -> None:
    budget_summary = evidence_pack.get("budget_summary")
    execution = (
        budget_summary.get("fresh_coverage_model_execution")
        if isinstance(budget_summary, Mapping)
        else None
    )
    if not isinstance(execution, Mapping):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_FRESH_ORIGIN_MISSING",
            "artifact_admission",
            "EvidencePack has no Fresh Coverage model execution receipt.",
        )
    roles = execution.get("roles")
    attempts = execution.get("attempts")
    call_count = execution.get("call_count")
    budget = execution.get("budget")
    admission = execution.get("admission")
    if (
        not isinstance(roles, list)
        or not roles
        or not all(type(role) is str and role for role in roles)
        or "requirement_projection" not in roles
        or "requirement_findings" not in roles
        or type(call_count) is not int
        or call_count != len(roles)
        or not isinstance(attempts, list)
        or len(attempts) != call_count
        or not isinstance(budget, Mapping)
        or budget.get("automatic_provider_fallback") is not False
        or budget.get("automatic_retry") is not False
        or not isinstance(admission, Mapping)
        or admission.get("provider") != "ollama"
        or admission.get("execution_locality") != "local"
        or admission.get("external_disclosure_occurred") is not False
        or budget_summary.get("all_queries_executed") is not True
        or not {
            "rg_live_query_set",
            "rg_live_read",
        }.issubset(set(evidence_pack.get("retrieval_methods_used") or []))
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_FRESH_ORIGIN_INVALID",
            "artifact_admission",
            "EvidencePack Fresh Coverage execution receipt is invalid.",
        )
    for index, (role, attempt) in enumerate(zip(roles, attempts, strict=True), start=1):
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt_index") != index
            or attempt.get("role") != role
            or attempt.get("status") not in {"accepted", "rejected"}
            or not _is_sha256(attempt.get("payload_canonical_sha256"))
            or not _is_sha256(attempt.get("schema_canonical_sha256"))
            or not _is_sha256(attempt.get("output_canonical_sha256"))
        ):
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_FRESH_ORIGIN_INVALID",
                "artifact_admission",
                "EvidencePack Fresh Coverage attempt receipt is invalid.",
            )

    valid_role_sequences = {
        (
            "requirement_projection",
            "requirement_findings",
        ),
        (
            "requirement_projection",
            "requirement_projection_correction",
            "requirement_findings",
        ),
        (
            "requirement_projection",
            "requirement_findings",
            "requirement_findings_correction",
        ),
        tuple(FRESH_COVERAGE_MODEL_CALL_BUDGET),
    }
    role_sequence = tuple(roles)
    role_call_counts = execution.get("role_call_counts")
    expected_role_call_counts = {
        role: role_sequence.count(role) for role in FRESH_COVERAGE_MODEL_CALL_BUDGET
    }
    if (
        role_sequence not in valid_role_sequences
        or role_call_counts != expected_role_call_counts
        or execution.get("correction_count")
        != sum(role.endswith("_correction") for role in role_sequence)
        or budget.get("maximum_total_calls") != sum(FRESH_COVERAGE_MODEL_CALL_BUDGET.values())
        or budget.get("maximum_calls_by_role") != FRESH_COVERAGE_MODEL_CALL_BUDGET
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_FRESH_ORIGIN_INVALID",
            "artifact_admission",
            "EvidencePack Fresh Coverage role sequence is invalid.",
        )

    attempt_by_role = {str(attempt["role"]): attempt for attempt in attempts}
    for base_role, correction_role in (
        ("requirement_projection", "requirement_projection_correction"),
        ("requirement_findings", "requirement_findings_correction"),
    ):
        base_attempt = attempt_by_role[base_role]
        correction_attempt = attempt_by_role.get(correction_role)
        if correction_attempt is None:
            valid_stage = (
                base_attempt.get("status") == "accepted" and base_attempt.get("error_code") is None
            )
        else:
            valid_stage = (
                base_attempt.get("status") == "rejected"
                and type(base_attempt.get("error_code")) is str
                and bool(base_attempt["error_code"])
                and correction_attempt.get("status") == "accepted"
                and correction_attempt.get("error_code") is None
            )
        if not valid_stage:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_FRESH_ORIGIN_INVALID",
                "artifact_admission",
                "EvidencePack Fresh Coverage stage has no accepted final output.",
            )


def preflight_planning_continuation(
    request: PlanningContinuationInputV1,
    *,
    artifact_store: ArtifactStore | None = None,
) -> _AdmittedPlanningInputs:
    artifact_root, workspace_root = _resolve_roots(request)
    if artifact_store is not None and not _same_path(
        artifact_store.root,
        artifact_root,
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_STORE_BINDING_MISMATCH",
            "artifact_admission",
            "Graph ArtifactStore does not match the continuation input root.",
        )
    store = artifact_store or ArtifactStore(artifact_root)
    feature_envelope = _read_exact_revision(store, request.feature_spec_ref)
    evidence_envelope = _read_exact_revision(store, request.evidence_pack_ref)
    if (feature_envelope.get("status") != "success"
            or evidence_envelope.get("status") not in {"success", "blocked"}):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_ARTIFACT_STATUS_INVALID",
            "artifact_admission",
            "Planning continuation requires successful Fresh input Artifacts.",
        )
    _validate_upstream(evidence_envelope, request.feature_spec_ref)
    try:
        feature_spec = strict_feature_spec_payload(feature_envelope["data"])
        pack = EvidencePack.from_artifact_payload(evidence_envelope["data"])
        candidate_scope = _admit_compilation_candidate(pack.to_artifact_payload(), feature_spec)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_PAYLOAD_INVALID",
            "artifact_admission",
            "Fresh input payload failed strict semantic round-trip.",
        ) from exc
    evidence_pack = pack.to_artifact_payload()
    if evidence_envelope.get("status") != "success" and candidate_scope is None:
        raise PlanningContinuationError("PLANNING_CONTINUATION_ARTIFACT_STATUS_INVALID", "artifact_admission",
            "Blocked Evidence requires a current compilation candidate scope.")
    if candidate_scope is None:
        _validate_fresh_origin(evidence_pack)
    if (
        feature_spec["feature_id"] != request.feature_spec_ref.feature_id
        or evidence_pack["query"] != feature_spec["user_request"]
        or (candidate_scope is None and (evidence_pack["readiness"] != "sufficient"
            or evidence_pack["can_generate_patch"] is not True
            or evidence_pack["authoritative_for_patch"] is not True))
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_SEMANTIC_ADMISSION_FAILED",
            "artifact_admission",
            "Fresh inputs are not eligible for formal implementation planning.",
        )
    requirement_coverage = evidence_pack.get("requirement_coverage")
    expected_feature_identity = {
        "feature_id": feature_spec["feature_id"],
        "feature_spec_sha256": feature_spec_payload_sha256(feature_spec),
    }
    if (
        not isinstance(requirement_coverage, Mapping)
        or requirement_coverage.get("feature_spec_identity") != expected_feature_identity
    ):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_COVERAGE_IDENTITY_MISMATCH",
            "artifact_admission",
            "Requirement coverage is not bound to the selected FeatureSpec payload.",
        )
    source_validation = _validate_source_items(evidence_pack, workspace_root)
    return _AdmittedPlanningInputs(
        artifact_store=store,
        feature_spec=feature_spec,
        evidence_pack=evidence_pack,
        source_validation=source_validation,
        source_tree_snapshot=_csharp_source_snapshot(workspace_root),
    )


def _read_selected_exact_revision(
    store: ArtifactStore,
    ref: ArtifactRef,
    *,
    consumer_stage: str | None,
) -> dict[str, Any]:
    """Read a selected immutable revision without promoting latest to authority."""

    try:
        revisions = store.list_feature_artifact_revisions(
            ref.feature_id,
            ref.artifact_type,
        )
        if ref not in revisions:
            raise PatchContinuationError(
                "PATCH_CONTINUATION_EXACT_REVISION_REQUIRED",
                "artifact_admission",
                "Patch continuation requires immutable Artifact revision refs.",
            )
        if consumer_stage is not None:
            store.registry.validate_consumer(ref.artifact_type, consumer_stage)
        envelope = dict(store.semantic_read_immutable(ref))
        freshness = store.check_exact_revision_freshness(ref)
    except PatchContinuationError:
        raise
    except (ArtifactRegistryError, ArtifactStoreError, OSError, ValueError) as exc:
        raise PatchContinuationError(
            "PATCH_CONTINUATION_ARTIFACT_INVALID",
            "artifact_admission",
            "Selected exact Artifact revision failed immutable admission.",
        ) from exc
    if not freshness.is_fresh:
        raise PatchContinuationError(
            "PATCH_CONTINUATION_ANCESTRY_STALE",
            "artifact_admission",
            "Selected exact Artifact ancestry is missing or content-invalid.",
        )
    return envelope


def _latest_observation(
    store: ArtifactStore,
    selected_ref: ArtifactRef,
) -> dict[str, Any]:
    """Observe the latest alias without granting it continuation authority."""

    try:
        latest_ref = _latest_artifact_ref(
            store,
            feature_id=selected_ref.feature_id,
            artifact_type=selected_ref.artifact_type,
        )
    except (ArtifactStoreError, PatchContinuationError, PlanningContinuationError) as exc:
        return {
            "status": "unavailable",
            "selected_is_latest": None,
            "failure_class": type(exc).__name__,
        }
    if latest_ref is None:
        return {
            "status": "absent",
            "selected_is_latest": False,
            "artifact_hash": None,
            "path": None,
        }
    return {
        "status": "observed",
        "selected_is_latest": latest_ref.artifact_hash == selected_ref.artifact_hash,
        "artifact_hash": latest_ref.artifact_hash,
        "path": latest_ref.path,
    }


def _validate_exact_upstream_set(
    envelope: Mapping[str, Any],
    expected_refs: tuple[ArtifactRef, ...],
) -> None:
    upstreams = envelope.get("inputs", {}).get("upstream_artifacts")
    if not isinstance(upstreams, list) or len(upstreams) != len(expected_refs):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_UPSTREAM_INVALID",
            "artifact_admission",
            "Selected Artifact has an unexpected direct upstream set.",
        )
    observed: dict[str, Mapping[str, Any]] = {}
    for upstream in upstreams:
        if not isinstance(upstream, Mapping):
            raise PatchContinuationError(
                "PATCH_CONTINUATION_UPSTREAM_INVALID",
                "artifact_admission",
                "Selected Artifact direct upstream identity is invalid.",
            )
        artifact_type = upstream.get("artifact_type")
        if type(artifact_type) is not str or artifact_type in observed:
            raise PatchContinuationError(
                "PATCH_CONTINUATION_UPSTREAM_INVALID",
                "artifact_admission",
                "Selected Artifact direct upstream types are invalid.",
            )
        observed[artifact_type] = upstream
    for expected in expected_refs:
        upstream = observed.get(expected.artifact_type)
        if upstream is None or (
            upstream.get("path") != expected.path
            or upstream.get("artifact_hash") != expected.artifact_hash
        ):
            raise PatchContinuationError(
                "PATCH_CONTINUATION_UPSTREAM_MISMATCH",
                "artifact_admission",
                "Selected Artifact ancestry does not match the explicit refs.",
            )


def _validate_selected_plan(
    implementation_plan: Mapping[str, Any],
    evidence_pack: dict[str, Any],
    *,
    feature_id: str,
) -> None:
    if (
        implementation_plan.get("feature_id") != feature_id
        or implementation_plan.get("blocking") is not False
    ):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_PLAN_NOT_ADMITTED",
            "artifact_admission",
            "Selected ImplementationPlan is blocking or has the wrong feature_id.",
        )
    registry = implementation_plan.get("binding_registry")
    proposed_changes = implementation_plan.get("proposed_changes")
    if not isinstance(registry, dict) or not isinstance(proposed_changes, list):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_PLAN_PAYLOAD_INVALID",
            "artifact_admission",
            "Selected ImplementationPlan has no valid binding/change representation.",
        )
    resolver = BindingResolver.from_implementation_plan(
        dict(implementation_plan),
        evidence_pack=evidence_pack,
    )
    binding_validation = resolver.validate_registry_against_evidence()
    if not binding_validation.allowed:
        raise PatchContinuationError(
            "PATCH_CONTINUATION_BINDING_INVALID",
            "binding_admission",
            "Selected ImplementationPlan bindings drifted from EvidencePack.",
        )
    edit_binding_ids = tuple(resolver.edit_bindings)
    if not edit_binding_ids or len(edit_binding_ids) != len(set(edit_binding_ids)):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_BINDING_INVALID",
            "binding_admission",
            "Selected ImplementationPlan edit bindings are empty or duplicated.",
        )
    selected_change_ids: list[str] = []
    selected_binding_ids: list[str] = []
    for change in proposed_changes:
        if not isinstance(change, Mapping):
            raise PatchContinuationError(
                "PATCH_CONTINUATION_PLAN_CHANGE_INVALID",
                "binding_admission",
                "Selected ImplementationPlan change is not an object.",
            )
        change_id = change.get("change_id")
        binding_id = change.get("edit_binding_id")
        if type(change_id) is not str or not change_id or type(binding_id) is not str:
            raise PatchContinuationError(
                "PATCH_CONTINUATION_PLAN_CHANGE_INVALID",
                "binding_admission",
                "Selected ImplementationPlan change identity is incomplete.",
            )
        selected_change_ids.append(change_id)
        selected_binding_ids.append(binding_id)
    if (
        len(selected_change_ids) != len(set(selected_change_ids))
        or len(selected_binding_ids) != len(set(selected_binding_ids))
        or set(selected_binding_ids) != set(edit_binding_ids)
    ):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_PLAN_SELECTION_INVALID",
            "binding_admission",
            "The exact Plan selection must cover every and only its edit bindings once.",
        )


def preflight_patch_continuation(
    request: PatchContinuationInputV1,
    *,
    artifact_store: ArtifactStore | None = None,
) -> _AdmittedPatchInputs:
    """Admit exact selected refs, ancestry, bindings, and current source bytes."""

    _validate_worker_configuration_identity(request.worker_configuration_ref,
        request.worker_profile_sha256, verify_bytes=True)
    if request.qualified_model_routing_ref is not None:
        _read_hashed_repair_file(request.qualified_model_routing_ref)
    if request.patch_upstream_configuration_ref is not None:
        _read_hashed_repair_file(request.patch_upstream_configuration_ref)
    try:
        artifact_root, workspace_root = _resolve_roots(
            request, allow_product_store_subdir=request.reuses_model_output,
        )  # type: ignore[arg-type]
    except PlanningContinuationError as exc:
        raise PatchContinuationError(
            exc.code.replace("PLANNING_", "PATCH_", 1),
            exc.stage,
            exc.reason,
        ) from exc
    if artifact_store is not None and not _same_path(
        artifact_store.root,
        artifact_root,
    ):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_STORE_BINDING_MISMATCH",
            "artifact_admission",
            "Graph ArtifactStore does not match the continuation input root.",
        )
    store = artifact_store or ArtifactStore(artifact_root)
    feature_envelope = _read_selected_exact_revision(
        store,
        request.feature_spec_ref,
        consumer_stage=None,
    )
    evidence_envelope = _read_selected_exact_revision(
        store,
        request.evidence_pack_ref,
        consumer_stage=PATCH_CONSUMER_STAGE,
    )
    plan_envelope = _read_selected_exact_revision(
        store,
        request.implementation_plan_ref,
        consumer_stage=PATCH_CONSUMER_STAGE,
    )
    if (any(envelope.get("status") != "success" for envelope in (feature_envelope, plan_envelope))
            or evidence_envelope.get("status") not in {"success", "blocked"}):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_ARTIFACT_STATUS_INVALID",
            "artifact_admission",
            "Patch continuation requires successful selected Artifacts.",
        )
    _validate_exact_upstream_set(evidence_envelope, (request.feature_spec_ref,))
    _validate_exact_upstream_set(
        plan_envelope,
        (request.feature_spec_ref, request.evidence_pack_ref),
    )
    try:
        feature_spec = strict_feature_spec_payload(feature_envelope["data"])
        evidence_pack = EvidencePack.from_artifact_payload(
            evidence_envelope["data"]
        ).to_artifact_payload()
        implementation_plan = dict(plan_envelope["data"])
        candidate_scope = _admit_compilation_candidate(evidence_pack, feature_spec,
            worker_configuration_ref=request.worker_configuration_ref, require_selected_worker=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise PatchContinuationError(
            "PATCH_CONTINUATION_PAYLOAD_INVALID",
            "artifact_admission",
            "Selected Artifact payload failed strict semantic admission.",
        ) from exc
    if evidence_envelope.get("status") != "success" and candidate_scope is None:
        raise PatchContinuationError("PATCH_CONTINUATION_ARTIFACT_STATUS_INVALID", "artifact_admission",
            "Blocked Evidence requires a current compilation candidate scope.")
    if (
        feature_spec["feature_id"] != request.feature_spec_ref.feature_id
        or evidence_pack["query"] != feature_spec["user_request"]
    ):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_SEMANTIC_IDENTITY_MISMATCH",
            "artifact_admission",
            "Selected FeatureSpec and EvidencePack semantic identities differ.",
        )
    if request.qualified_model_routing_ref is not None:
        from .model_qualification import ModelQualificationError, resolve_qualified_patch_profile
        from .patch_worker_configuration import PatchWorkerConfiguration

        if request.worker_configuration_ref is None:
            raise PatchContinuationError("MODEL_QUALIFICATION_PENDING", "model_qualification", "Qualified routing requires a worker configuration.")
        config = PatchWorkerConfiguration.load(request.worker_configuration_ref["path"])
        try:
            selected = resolve_qualified_patch_profile(
                routing_path=request.qualified_model_routing_ref["path"],
                role="repair" if request.repair_parent_proposal_ref is not None else "initial_patch",
                target_mod=feature_spec.get("target_mod"), worker_policy=config.policy,
            )
        except ModelQualificationError as exc:
            raise PatchContinuationError(exc.code, "model_qualification", str(exc)) from exc
        if selected.routing_ref != request.qualified_model_routing_ref or _canonical_sha256(selected.profile.to_dict()) != request.worker_profile_sha256:
            raise PatchContinuationError("MODEL_QUALIFIED_PROFILE_ARGUMENT_CONFLICT", "model_qualification", "Qualified routing identity differs from the requested profile.")
    _validate_selected_plan(
        implementation_plan,
        evidence_pack,
        feature_id=request.feature_spec_ref.feature_id,
    )
    try:
        source_validation = _validate_source_items(evidence_pack, workspace_root)
    except PlanningContinuationError as exc:
        raise PatchContinuationError(
            exc.code.replace("PLANNING_", "PATCH_", 1),
            exc.stage,
            exc.reason,
        ) from exc
    selected_refs = (
        request.feature_spec_ref,
        request.evidence_pack_ref,
        request.implementation_plan_ref,
    )
    source_tree_snapshot = _patch_request_source_snapshot(request)
    repair_context = _admit_patch_repair_context(
        request=request, store=store, feature_spec=feature_spec,
        evidence_pack=evidence_pack, implementation_plan=implementation_plan,
        workspace_root=workspace_root, source_tree_snapshot=source_tree_snapshot,
    ) if request.repair_parent_proposal_ref is not None else None
    admitted = _AdmittedPatchInputs(
        artifact_store=store,
        feature_spec=feature_spec,
        evidence_pack=evidence_pack,
        implementation_plan=implementation_plan,
        source_validation=source_validation,
        source_tree_snapshot=source_tree_snapshot,
        latest_observations={
            ref.artifact_type: _latest_observation(store, ref)
            for ref in selected_refs
        },
        repair_context=repair_context,
    )
    if request.saved_patch_selection is not None:
        if any(_is_within(workspace_root / row["source_path"], artifact_root)
               for row in source_validation["files"]):
            raise PatchContinuationError("PATCH_SAVED_SOURCE_IN_STORE", "source_validation",
                "Saved Patch target evidence must not resolve inside the ArtifactStore.")
        from src.mod_agent.saved_patch_admission import admit_saved_patch

        try:
            admitted = replace(admitted, saved_patch=admit_saved_patch(request, admitted))
        except PatchContinuationError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise PatchContinuationError("PATCH_SAVED_ADMISSION_REJECTED", "saved_patch_admission",
                "Saved source, verification or budget records failed exact admission.") from exc
    if request.candidate_reverification_selection is not None:
        from src.mod_agent.saved_candidate_reverification import admit_saved_candidate
        admitted = replace(admitted, saved_candidate=admit_saved_candidate(request, admitted))
    return admitted


def _read_hashed_repair_file(ref: Mapping[str, str] | None) -> bytes:
    if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
            or type(ref.get("path")) is not str or not Path(ref["path"]).is_absolute()
            or not _is_sha256(ref.get("sha256"))):
        raise PatchContinuationError("PATCH_REPAIR_FILE_REF_INVALID", "repair_admission", "Repair files require exact absolute path and SHA-256.")
    try:
        payload = Path(ref["path"]).read_bytes()
    except OSError as exc:
        raise PatchContinuationError("PATCH_REPAIR_FILE_MISSING", "repair_admission", "A selected repair input cannot be read.") from exc
    if hashlib.sha256(payload).hexdigest() != ref["sha256"]:
        raise PatchContinuationError("PATCH_REPAIR_FILE_HASH_MISMATCH", "repair_admission", "Selected repair file bytes changed.")
    return payload


def _repair_json(ref: Mapping[str, str] | None) -> dict[str, Any]:
    value = json.loads(_read_hashed_repair_file(ref))
    if not isinstance(value, dict):
        raise ValueError("A bound repair document must be a JSON object.")
    return value


def _patch_request_source_snapshot(request):
    root = Path(request.workspace_root).resolve()
    if getattr(request, "candidate_reverification_selection", None) is not None:
        manifest = _repair_json(request.candidate_reverification_selection["parent_manifest_ref"])
        target = Path(manifest["nuget_isolated_build_profile"]["working_directory"]).resolve()
        if target == root or not target.is_relative_to(root) or target.is_relative_to(Path(request.artifact_store_root).resolve()):
            raise PatchContinuationError("SAVED_CANDIDATE_SOURCE_SCOPE_INVALID", "source_validation", "Retained target must be inside the original workspace.")
        # 为什么要有它：验收会创建新的编译副本，源码漂移比较应覆盖原目标工程，而不是把验收产物当成产品修改。
        root = target
    return _csharp_source_snapshot(root)


def _formal_repair_selection(source: Mapping[str, Any], source_ref: Mapping[str, str]) -> dict[str, Any]:
    artifacts = source.get("artifacts") or {}
    if source.get("proposal_admitted") is True and all(kind in artifacts for kind in ("patch_proposal", "proposal_review")):
        response = source["factory_receipt"]["model_response"]
        return {
            "repair_parent_proposal_ref": ArtifactRef(**artifacts["patch_proposal"]),
            "repair_parent_review_ref": ArtifactRef(**artifacts["proposal_review"]),
            "repair_parent_attempt_ref": dict(source_ref),
            "repair_parent_response_ref": {"path": response["path"], "sha256": response["output_sha256"]},
        }
    context = source.get("repair_context") or {}
    return {
        "repair_parent_proposal_ref": ArtifactRef(**context["parent_proposal"]),
        "repair_parent_review_ref": ArtifactRef(**context["parent_review"]),
        "repair_parent_attempt_ref": dict(context["parent_attempt_ref"]),
        "repair_parent_response_ref": dict(context["parent_response_ref"]),
    }


def _verified_patch_repair_binding(
    *, request: PatchContinuationInputV1, store: ArtifactStore, parent: Mapping[str, Any],
    prepared: Any, generator: Any, inspector: Any, workspace_root: Path,
    implementation_plan: dict[str, Any], workspace_snapshot: str, parent_retry_count: int,
) -> dict[str, Any]:
    """Recheck a separate executed verification against one explicit source output."""
    from src.libs.llm import ChatResponse
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch

    source = _repair_json(request.repair_source_attempt_ref)
    verification = _repair_json(request.repair_verification_ref)
    formal = _formal_repair_selection(source, request.repair_source_attempt_ref)
    if any(getattr(request, key) != value for key, value in formal.items()):
        raise ValueError("The selected formal ancestor differs from the source attempt's explicit lineage.")
    selected = source.get("selected_artifacts")
    expected = {"feature_spec": _artifact_ref_dict(request.feature_spec_ref),
                "evidence_pack": _artifact_ref_dict(request.evidence_pack_ref),
                "implementation_plan": _artifact_ref_dict(request.implementation_plan_ref)}
    if selected != expected or source.get("feature_id") != request.feature_spec_ref.feature_id:
        raise ValueError("The source attempt belongs to different FeatureSpec, Evidence or Plan revisions.")
    if (source.get("generated_against_snapshot_sha256") != workspace_snapshot
            or source.get("current_snapshot_sha256") != workspace_snapshot):
        raise ValueError("The source attempt's baseline is not the current unchanged workspace.")
    factory = source.get("factory_receipt") or {}
    response = factory.get("model_response")
    if (not isinstance(response, dict) or response.get("retained") is not True
            or type(source.get("model_call_count")) is not int or source["model_call_count"] < 1
            or response.get("model") != (factory.get("profile") or {}).get("model")):
        raise ValueError("The source attempt has no retained, model-bound generation response.")
    failed_response_ref = {"path": response["path"], "sha256": response["output_sha256"]}
    response_bytes = _read_hashed_repair_file(failed_response_ref)
    if response.get("output_utf8_bytes") != len(response_bytes):
        raise ValueError("The source response byte count differs from its retained bytes.")
    callback_ref = verification["original_callback_ref"]
    callback = _repair_json(callback_ref)
    if (verification.get("original_model_response") != response or callback.get("model_response") != response
            or type(verification.get("model_calls")) is not int or verification["model_calls"] != 0
            or verification.get("source_unchanged") is not True):
        raise ValueError("Separate verification is not bound to this exact source response and unchanged baseline.")
    previous_response = response_bytes.decode("utf-8")
    generated = generator.materialize_prepared_native(prepared, ChatResponse(content=previous_response, model=response["model"]))
    native_hash = _canonical_sha256(generated.changes)
    if (callback.get("generated_changes") != list(generated.changes)
            or callback.get("generated_changes_sha256") != native_hash
            or callback.get("source_snapshot_sha256") != inspector.source_snapshot_sha256):
        raise ValueError("Verification callback code, native hash, or source snapshot differs from the source output.")
    materialized = materialize_exact_text_patch(workspace_root=workspace_root,
        diff_intent=generated.changes, implementation_plan=implementation_plan)
    current_code = {"baseline_sha256": materialized.baseline_sha256,
                    "code_sha256": materialized.code_sha256, "files": sorted(materialized.files)}
    if callback.get("materialized_code_identity") != current_code:
        raise ValueError("Verification callback materialized bytes differ from the current source output.")
    reported = verification.get("behavior_report")
    if not isinstance(reported, dict):
        raise ValueError("Separate verification omitted the actual behavior report.")
    original_report_ref = reported["report_ref"]
    original_report = _repair_json(original_report_ref)
    if original_report != {key: value for key, value in reported.items() if key != "report_ref"}:
        raise ValueError("Verification result and persisted behavior report disagree.")
    if (original_report.get("execution_origin") != "executed" or original_report.get("callback_invoked") is not True
            or original_report.get("status") not in {"behavior_failed", "needs_evidence"}
            or original_report.get("generated_changes_sha256") != native_hash
            or original_report.get("source_snapshot_sha256") != inspector.source_snapshot_sha256
            or original_report.get("materialized_code_identity") != current_code):
        raise ValueError("A newly bound repair requires an executed failing verification of these exact code bytes.")
    owners = original_report.get("owner_binding_ids")
    if (not isinstance(owners, list) or any(type(binding) is not str for binding in owners)
            or len(set(owners)) != len(owners) or not set(owners).issubset(inspector.available_binding_ids)):
        raise ValueError("Failure owner bindings exceed the unchanged selected Plan.")
    matrix_ref = original_report["verification_ref"]
    matrix = _repair_json(matrix_ref)
    checks = matrix.get("checks")
    if (matrix.get("source_provenance_verified") is not True
            or type(matrix.get("executions")) is not int or matrix["executions"] < 1
            or not isinstance(checks, list) or not checks
            or any(not isinstance(check, dict) or type(check.get("check")) is not str
                   or type(check.get("passed")) is not bool for check in checks)):
        raise ValueError("The bound verification matrix has no completed checks with verified source provenance.")
    failures = [check for check in checks if check["passed"] is False]
    failed_by_id = {check["check"]: check for check in failures}
    observations = original_report.get("observations")
    summary = original_report.get("verification_summary") or {}
    if (len({check["check"] for check in checks}) != len(checks) or not failures
            or matrix.get("failures") != failures
            or summary != {"checks": len(checks), "executions": matrix["executions"], "failed_checks": len(failures)}
            or not isinstance(observations, list) or not observations
            or {item.get("check_id") for item in observations if isinstance(item, dict)} != set(failed_by_id)
            or len(observations) != len(failures)):
        raise ValueError("Failure observations and verification totals do not cover the same failed checks.")
    for observation in observations:
        failure = failed_by_id[observation["check_id"]]
        if (observation.get("expected") != failure.get("expected")
                or observation.get("observed") != failure.get("actual")
                or observation.get("result_ref") != matrix_ref or not isinstance(observation.get("input"), dict)):
            raise ValueError("A reported failure disagrees with its exact matrix result or lacks its inputs.")
    if "source_report_ref" in original_report:
        raw_report = _repair_json(original_report["source_report_ref"])
        if any(raw_report.get(key) != original_report.get(key)
               for key in ("generated_changes_sha256", "source_snapshot_sha256", "status")):
            raise ValueError("The original verifier report identifies different generated code.")
    configuration = _repair_json(verification["configuration"])
    evaluator = configuration.get("behavior_evaluator")
    if isinstance(evaluator, dict):
        if "factory_ref" in evaluator:
            _read_hashed_repair_file(evaluator["factory_ref"])
        kwargs = evaluator.get("kwargs") or {}
        if "plan_ref" in kwargs and kwargs["plan_ref"] != expected["implementation_plan"]:
            raise ValueError("The verifier configuration selects a different Plan.")
        if "source_root" in kwargs and not _same_path(Path(kwargs["source_root"]), workspace_root):
            raise ValueError("The verifier configuration selects a different source tree.")
        if "adapter_ref" in kwargs:
            _read_hashed_repair_file(kwargs["adapter_ref"])
    parent_code = materialize_exact_text_patch(workspace_root=workspace_root,
        diff_intent=parent["data"]["diff_intent"], implementation_plan=implementation_plan)
    formal_code = {"baseline_sha256": parent_code.baseline_sha256,
                   "code_sha256": parent_code.code_sha256, "files": sorted(parent_code.files)}
    if current_code["baseline_sha256"] != formal_code["baseline_sha256"]:
        raise ValueError("Current output and formal ancestor do not share the same original source baseline.")
    source_repair_ref = (source.get("artifacts") or {}).get("repair_plan")
    if source_repair_ref is not None:
        if source_repair_ref.get("feature_id") != request.feature_spec_ref.feature_id:
            raise ValueError("Source RepairPlan belongs to a different feature.")
        repair = _read_selected_exact_revision(store, ArtifactRef(**source_repair_ref), consumer_stage=None)
        count = repair.get("data", {}).get("retry_count")
        if repair.get("status") != "success" or type(count) is not int or count < parent_retry_count:
            raise ValueError("Source RepairPlan retry count is invalid.")
        if source.get("proposal_admitted") is not True:
            upstream = repair.get("inputs", {}).get("upstream_artifacts", [])
            if not any(item.get("path") == request.repair_parent_proposal_ref.path
                       and item.get("artifact_hash") == request.repair_parent_proposal_ref.artifact_hash for item in upstream):
                raise ValueError("Source RepairPlan does not descend from the selected formal Proposal.")
        parent_retry_count = count
    derived = dict(original_report)
    derived.update(execution_origin="unknown", callback_invoked=False,
        code_revision={"comparison_basis": "same_B0_materializer_whole_file_bytes",
            "same_materialized_code": current_code == formal_code,
            "different_code_is_not_proof_of_repair": True},
        derivation={"source_attempt_ref": dict(request.repair_source_attempt_ref),
            "verification_result_ref": dict(request.repair_verification_ref),
            "original_report_ref": dict(original_report_ref), "original_callback_ref": dict(callback_ref),
            "failed_response_ref": failed_response_ref, "configuration_ref": dict(verification["configuration"]),
            "verification_ref": dict(matrix_ref), "formal_parent_ref": _artifact_ref_dict(request.repair_parent_proposal_ref),
            "current_materialized_code": current_code, "formal_parent_materialized_code": formal_code,
            "source_repair_plan_ref": source_repair_ref, "inherited_retry_count": parent_retry_count,
            "original_execution_origin": "executed", "new_model_calls": 0, "new_verifier_executions": 0,
            "approval_inherited": False})
    return {"failure_report": derived, "previous_response": previous_response, "generated": generated,
            "failed_response_ref": failed_response_ref, "parent_retry_count": parent_retry_count}


def bind_patch_repair_verification(
    *, request: PatchContinuationInputV1, source_attempt_ref: Mapping[str, str],
    verification_result_ref: Mapping[str, str],
) -> PatchContinuationInputV1:
    """Bind an explicit verified output to existing Repair inputs without a model or write Apply."""
    if request.attempt_id is None:
        raise PatchContinuationError("PATCH_REPAIR_ATTEMPT_ID_REQUIRED", "repair_binding",
            "Binding requires the caller's explicit attempt identity; it does not start an attempt.")
    try:
        base = replace(request, repair_parent_proposal_ref=None, repair_parent_review_ref=None,
            repair_failure_report_ref=None, repair_parent_response_ref=None, repair_parent_attempt_ref=None,
            repair_source_attempt_ref=None, repair_verification_ref=None, qualified_model_routing_ref=None)
        admitted = preflight_patch_continuation(base)
        source = _repair_json(source_attempt_ref)
        verification = _repair_json(verification_result_ref)
        selected = replace(base, **_formal_repair_selection(source, source_attempt_ref),
            repair_failure_report_ref=dict(verification["behavior_report"]["report_ref"]),
            repair_source_attempt_ref=dict(source_attempt_ref), repair_verification_ref=dict(verification_result_ref),
            qualified_model_routing_ref=request.qualified_model_routing_ref)
        context = _admit_patch_repair_context(request=selected, store=admitted.artifact_store,
            feature_spec=admitted.feature_spec, evidence_pack=admitted.evidence_pack,
            implementation_plan=admitted.implementation_plan, workspace_root=Path(request.workspace_root).resolve(),
            source_tree_snapshot=admitted.source_tree_snapshot, binding_preview=True)
        content = json.dumps(context["failure_report"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False)
        payload = content.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        path = admitted.artifact_store.root / "cr" / f"rv-{digest[:24]}.json"
        try:
            admitted.artifact_store._write_text_exclusive_atomic(path, content)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise PatchContinuationError("PATCH_REPAIR_BINDING_CONFLICT", "repair_binding",
                    "An immutable derived repair file already exists with different bytes.")
        result = replace(selected, repair_failure_report_ref={"path": str(path), "sha256": digest})
        preflight_patch_continuation(result)
        return result
    except (ValueError, KeyError, TypeError, OSError) as exc:
        if isinstance(exc, PatchContinuationError):
            raise
        raise PatchContinuationError("PATCH_REPAIR_BINDING_INVALID", "repair_binding", str(exc)) from exc


def _admit_patch_repair_context(
    *, request: PatchContinuationInputV1, store: ArtifactStore,
    feature_spec: dict[str, Any], evidence_pack: dict[str, Any],
    implementation_plan: dict[str, Any], workspace_root: Path,
    source_tree_snapshot: tuple[tuple[str, str], ...],
    binding_preview: bool = False,
) -> dict[str, Any]:
    """Bind a real post-Proposal observation without rewriting its Review result."""
    from src.libs.llm import ChatResponse
    from src.mod_agent.local_entry import (
        _external_evidence_items,
        _patch_generation_obligation_projection,
    )
    from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator
    from src.mod_agent.proposal.patch_agent_loop import (
        PatchAgentLoopPolicy,
        PlanBoundSourceInspector,
    )

    parent_ref = request.repair_parent_proposal_ref
    review_ref = request.repair_parent_review_ref
    assert parent_ref is not None and review_ref is not None
    parent = _read_selected_exact_revision(store, parent_ref, consumer_stage=None)
    review = _read_selected_exact_revision(store, review_ref, consumer_stage=None)
    expected_upstream = {
        "implementation_plan": request.implementation_plan_ref,
        "evidence_pack": request.evidence_pack_ref,
    }
    actual_upstream = {item.get("artifact_type"): item for item in parent.get("inputs", {}).get("upstream_artifacts", [])}
    for kind, ref in expected_upstream.items():
        bound = actual_upstream.get(kind, {})
        if bound.get("path") != ref.path or bound.get("artifact_hash") != ref.artifact_hash:
            raise PatchContinuationError("PATCH_REPAIR_PARENT_PLAN_MISMATCH", "repair_admission", "Parent Proposal was generated against different Plan or Evidence bytes.")
    parent_retry_count = 0
    if "repair_plan" in actual_upstream:
        recorded = actual_upstream["repair_plan"]
        parent_repair = _read_selected_exact_revision(store, ArtifactRef(
            feature_id=parent_ref.feature_id, artifact_type="repair_plan",
            path=recorded["path"], artifact_hash=recorded["artifact_hash"],
        ), consumer_stage=None)
        parent_retry_count = parent_repair["data"].get("retry_count")
        if type(parent_retry_count) is not int or parent_retry_count < 0:
            raise PatchContinuationError("PATCH_REPAIR_PARENT_RETRY_INVALID", "repair_admission", "Parent repair retry identity is invalid.")
    review_inputs = review.get("inputs", {}).get("upstream_artifacts", [])
    if not any(item.get("artifact_type") == "patch_proposal" and item.get("path") == parent_ref.path and item.get("artifact_hash") == parent_ref.artifact_hash for item in review_inputs):
        raise PatchContinuationError("PATCH_REPAIR_REVIEW_PARENT_MISMATCH", "repair_admission", "Selected Review does not belong to the parent Proposal.")
    if parent.get("status") != "success" or review.get("data", {}).get("allowed_next_stage") != "dry_run_apply":
        raise PatchContinuationError("PATCH_REPAIR_PARENT_REVIEW_INVALID", "repair_admission", "Behavior recovery requires the original successful Proposal and passed deterministic Review.")
    try:
        report = json.loads(_read_hashed_repair_file(request.repair_failure_report_ref))
        attempt = json.loads(_read_hashed_repair_file(request.repair_parent_attempt_ref))
        previous_response = _read_hashed_repair_file(request.repair_parent_response_ref).decode("utf-8")
        if not isinstance(report, dict) or not isinstance(attempt, dict):
            raise ValueError("Repair report and parent receipt must be objects.")
        factory = attempt.get("factory_receipt", {})
        response_identity = factory.get("model_response", {})
        receipt_parent = attempt.get("artifacts", {}).get("patch_proposal", {})
        workspace_snapshot = _source_tree_snapshot_sha256(source_tree_snapshot)
        if (receipt_parent.get("artifact_hash") != parent_ref.artifact_hash
                or receipt_parent.get("path") != parent_ref.path
                or attempt.get("generated_against_snapshot_sha256") != workspace_snapshot
                or response_identity.get("output_sha256") != request.repair_parent_response_ref["sha256"]
                or response_identity.get("path") != request.repair_parent_response_ref["path"]):
            raise ValueError("Parent attempt, response, Proposal, and current baseline do not match.")
        inspector = PlanBoundSourceInspector(workspace_root=workspace_root,
            implementation_plan=implementation_plan, policy=PatchAgentLoopPolicy())
        if report.get("source_snapshot_sha256") != inspector.source_snapshot_sha256:
            raise ValueError("Behavior report source snapshot does not match the current edit sources.")
        obligations = _patch_generation_obligation_projection(
            feature_spec=feature_spec, implementation=implementation_plan,
            planned_changes=list(implementation_plan["proposed_changes"]),
            evidence_pack=evidence_pack,
        )
        generator = LLMExactPatchGenerator(None)  # preparation/materialization never calls this provider
        prepared = generator.prepare_native_contract(
            **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
            user_request=feature_spec["user_request"],
            evidence_items=_external_evidence_items(evidence_pack["items"]),
            binding_registry=implementation_plan["binding_registry"],
            constraints=obligations["side_effect_boundaries"],
            target_behavior=obligations["target_behavior"],
            acceptance_criteria=obligations["acceptance_criteria"],
            runtime_verification_plan=obligations["runtime_verification_plan"],
            required_change_units=obligations["required_change_units"],
            required_validation_by_binding={item["edit_binding_id"]: item["expected_validation"] for item in obligations["required_change_units"]},
        )
        generated = generator.materialize_prepared_native(prepared, ChatResponse(content=previous_response, model=response_identity["model"]))
        native_hash = _canonical_sha256(generated.changes)
        if request.repair_source_attempt_ref is None and native_hash != report.get("generated_changes_sha256"):
            raise ValueError("Behavior report was measured against different generated code.")
        parent_changes = parent["data"]["diff_intent"]
        parent_by_binding = {item["edit_binding_id"]: item for item in parent_changes}
        if len(parent_by_binding) != len(parent_changes) or set(parent_by_binding) != {item["edit_binding_id"] for item in generated.changes}:
            raise ValueError("Parent response binding coverage differs from the stored Proposal.")
        for change in generated.changes:
            stored = parent_by_binding[change["edit_binding_id"]]
            if stored.get("patch") != change["patch"] or stored.get("generation", {}).get("model_response_sha256") != request.repair_parent_response_ref["sha256"]:
                raise ValueError("Parent Proposal code or model response identity differs from its original response.")
        failed_response_ref = dict(request.repair_parent_response_ref)
        binding_sources: dict[str, Any] = {}
        if request.repair_source_attempt_ref is not None:
            binding = _verified_patch_repair_binding(
                request=request, store=store, parent=parent, prepared=prepared, generator=generator,
                inspector=inspector, workspace_root=workspace_root,
                implementation_plan=implementation_plan, workspace_snapshot=workspace_snapshot,
                parent_retry_count=parent_retry_count,
            )
            if not binding_preview and report != binding["failure_report"]:
                raise ValueError("The derived failure report differs from its verified source inputs.")
            report = binding["failure_report"]
            previous_response = binding["previous_response"]
            generated = binding["generated"]
            native_hash = _canonical_sha256(generated.changes)
            failed_response_ref = binding["failed_response_ref"]
            parent_retry_count = binding["parent_retry_count"]
            binding_sources = {"source_attempt_ref": dict(request.repair_source_attempt_ref),
                               "verification_result_ref": dict(request.repair_verification_ref)}
        if report.get("status") not in {"behavior_failed", "needs_evidence", "infrastructure_failed", "identity_failed"}:
            raise ValueError("Repair input is not a failing behavior observation.")
        observations = report.get("observations")
        if not isinstance(observations, list) or not observations or any(not isinstance(item, dict) for item in observations):
            raise ValueError("Repair requires structured observed behavior.")
        if "verification_ref" in report:
            _read_hashed_repair_file(report["verification_ref"])
        for observation in observations:
            if "result_ref" in observation:
                _read_hashed_repair_file(observation["result_ref"])
        owner_ids = report.get("owner_binding_ids")
        if not isinstance(owner_ids, list) or len(set(owner_ids)) != len(owner_ids) or any(type(value) is not str for value in owner_ids):
            raise ValueError("Failure owner binding identities are invalid.")
    except (ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, PatchContinuationError):
            raise
        raise PatchContinuationError("PATCH_REPAIR_FAILURE_IDENTITY_MISMATCH", "repair_admission", str(exc)) from exc
    return {
        "schema_version": "patch_repair_context_v1",
        "failure_source": "behavior_report",
        "parent_proposal": _artifact_ref_dict(parent_ref),
        "parent_review": _artifact_ref_dict(review_ref),
        "parent_review_decision": review["data"]["decision"],
        "parent_review_role": "passed_context_lineage_not_failure_evidence",
        "parent_response_ref": dict(request.repair_parent_response_ref),
        "parent_attempt_ref": dict(request.repair_parent_attempt_ref),
        "previous_response": previous_response,
        "failed_response_ref": failed_response_ref,
        "failed_diff_intent": list(generated.changes),
        "failure_report_ref": dict(request.repair_failure_report_ref),
        "failure_report": report,
        "source_snapshot_sha256": inspector.source_snapshot_sha256,
        "workspace_snapshot_sha256": workspace_snapshot,
        "generated_changes_sha256": native_hash,
        "parent_overlay_changes_sha256": attempt.get("generated_changes_sha256"),
        "allowed_edit_binding_ids": list(inspector.available_binding_ids),
        "base_relationship": "full_replacement_against_unchanged_parent_base",
        "approval_inherited": False,
        "parent_retry_count": parent_retry_count,
        **binding_sources,
    }


def _source_tree_snapshot_sha256(
    snapshot: tuple[tuple[str, str], ...],
) -> str:
    return _canonical_sha256(
        [{"path": path, "sha256": sha256} for path, sha256 in snapshot]
    )


def admit_active_worker_failure(request, receipt, artifact_store):
    """Rebind the latest actually checked code, keeping its formal ancestors intact."""
    from src.libs.llm import ChatResponse
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch
    from src.mod_agent.local_entry import (
        _external_evidence_items,
        _patch_generation_obligation_projection,
    )
    from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator

    admitted = preflight_patch_continuation(request, artifact_store=artifact_store)
    context = admitted.repair_context
    evaluations = (receipt.get("patch_agent_loop") or {}).get("behavior_evaluations") or []
    if context is None or not evaluations:
        return admitted
    try:
        latest = evaluations[-1]
        if (latest.get("admission") != "accepted" or latest.get("execution_origin") != "executed"
                or latest.get("status") not in {"behavior_failed", "needs_evidence"}):
            raise ValueError("Upstream recovery requires an admitted, executed failing observation.")
        report_ref = latest["report_ref"]
        report = _repair_json(report_ref)
        for key in ("status", "failure_owner", "owner_binding_ids", "observations",
                    "source_snapshot_sha256", "generated_changes_sha256", "execution_origin"):
            if latest.get(key) != report.get(key):
                raise ValueError("The latest failure differs from its saved verification result.")
        if report["source_snapshot_sha256"] != context["source_snapshot_sha256"]:
            raise ValueError("The observed failure belongs to different source bytes.")
        response_hash = latest.get("model_response_sha256") or report.get("model_response_sha256")
        if "model_response_sha256" in report and report["model_response_sha256"] != response_hash:
            raise ValueError("The persisted failure names a different model response.")
        identities = [receipt.get("model_response"), *(receipt.get("model_responses") or []),
                      ((receipt.get("patch_agent_loop") or {}).get("behavior_correction") or {}).get("response")]
        identities = [item for item in identities if isinstance(item, Mapping)
                      and item.get("output_sha256") == response_hash]
        if not identities or not response_hash:
            raise ValueError("No retained code response matches the latest checked output.")
        response = identities[-1]
        response_ref = {"path": response["path"], "sha256": response_hash}
        raw = _read_hashed_repair_file(response_ref).decode("utf-8")
        obligations = _patch_generation_obligation_projection(feature_spec=admitted.feature_spec,
            implementation=admitted.implementation_plan,
            planned_changes=list(admitted.implementation_plan["proposed_changes"]), evidence_pack=admitted.evidence_pack)
        generator = LLMExactPatchGenerator(None)
        prepared = generator.prepare_native_contract(user_request=admitted.feature_spec["user_request"],
            **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
            evidence_items=_external_evidence_items(admitted.evidence_pack["items"]),
            binding_registry=admitted.implementation_plan["binding_registry"],
            constraints=obligations["side_effect_boundaries"], target_behavior=obligations["target_behavior"],
            acceptance_criteria=obligations["acceptance_criteria"],
            runtime_verification_plan=obligations["runtime_verification_plan"],
            required_change_units=obligations["required_change_units"],
            required_validation_by_binding={item["edit_binding_id"]: item["expected_validation"]
                for item in obligations["required_change_units"]})
        generated = generator.materialize_prepared_native(prepared, ChatResponse(content=raw, model=response["model"]))
        if _canonical_sha256(generated.changes) != report["generated_changes_sha256"]:
            raise ValueError("Failure observations were measured against different generated changes.")
        materialized = materialize_exact_text_patch(workspace_root=request.workspace_root,
            diff_intent=generated.changes, implementation_plan=admitted.implementation_plan)
        code_identity = {"baseline_sha256": materialized.baseline_sha256,
                         "code_sha256": materialized.code_sha256, "files": sorted(materialized.files)}
        if latest.get("materialized_code_identity") != code_identity:
            raise ValueError("Failure observations do not match the actual materialized code.")
        if "verification_ref" in report:
            _read_hashed_repair_file(report["verification_ref"])
        for observation in report["observations"]:
            if "result_ref" in observation:
                _read_hashed_repair_file(observation["result_ref"])
        current = {**context, "previous_response": raw, "failed_response_ref": response_ref,
            "failed_diff_intent": list(generated.changes), "generated_changes_sha256": _canonical_sha256(generated.changes),
            "failure_report_ref": dict(report_ref), "failure_report": report,
            "previous_failure_binding": {"failed_response_ref": context["failed_response_ref"],
                "failure_report_ref": context["failure_report_ref"],
                "generated_changes_sha256": context["generated_changes_sha256"]},
            "active_failure_evaluation_index": len(evaluations), "approval_inherited": False}
        return replace(admitted, repair_context=current)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise PatchContinuationError("PATCH_UPSTREAM_FAILURE_BINDING_REQUIRED", "upstream_admission", str(exc)) from exc


class PatchContinuationAttemptJournal:
    """Atomic non-Artifact receipt for one external Patch attempt and its gates."""

    @classmethod
    def for_product_execution(
        cls,
        *,
        artifact_root: str | Path,
        run_id: str,
        request_id: str,
        parent_run_id: str,
        feature_spec_ref: ArtifactRef,
        evidence_pack_ref: ArtifactRef,
        implementation_plan_ref: ArtifactRef,
        workspace_root: str | Path,
        max_model_calls: int | None,
        model_call_budget_policy: str = "stage_and_total",
        worker_configuration_ref: Mapping[str, str] | None = None,
        worker_profile_sha256: str | None = None,
    ) -> PatchContinuationAttemptJournal:
        """Bind a Product Graph worker segment without claiming a Fresh continuation.

        This records current owner outputs and source observations. The caller
        retains execution and disclosure admission; no Repair context is inferred.
        """
        if any(type(value) is not str or not value.strip()
               for value in (run_id, request_id, parent_run_id)) or run_id == parent_run_id:
            raise PatchContinuationError(
                "PATCH_PRODUCT_JOURNAL_IDENTITY_INVALID", "patch_attempt_receipt",
                "A worker segment requires explicit request and distinct parent/run identities.",
            )
        if (model_call_budget_policy not in {"stage_and_total", "task_total_only", "meter_only"}
                or (max_model_calls is not None if model_call_budget_policy == "meter_only" else
                    type(max_model_calls) is not int or max_model_calls < 1
                    or model_call_budget_policy == "stage_and_total" and max_model_calls > 5)):
            raise PatchContinuationError(
                "PATCH_CONTINUATION_MODEL_CALL_BUDGET_INVALID", "request",
                "Metered segments have no call quota; bounded segments require their original positive allowance.",
            )
        selected_refs = (feature_spec_ref, evidence_pack_ref, implementation_plan_ref)
        try:
            for ref, kind in zip(selected_refs, ("feature_spec", "evidence_pack", "implementation_plan")):
                _validate_artifact_ref(ref, field_name=kind + "_ref", expected_type=kind)
            if len({ref.feature_id for ref in selected_refs}) != 1:
                raise PatchContinuationError(
                    "PATCH_CONTINUATION_FEATURE_MISMATCH", "artifact_admission",
                    "The Product Graph selected refs from different features.",
                )
            root = Path(artifact_root).resolve(strict=True)
            workspace = Path(workspace_root).resolve(strict=True)
            if not root.is_dir() or not workspace.is_dir():
                raise OSError("Journal roots must be existing directories.")
            # Product execution already stores Artifacts beneath its run root;
            # that workspace may contain the Store, but cannot be inside it.
            if _is_within(workspace, root):
                raise PatchContinuationError(
                    "PATCH_CONTINUATION_ROOTS_OVERLAP", "artifact_admission",
                    "The product workspace cannot be the ArtifactStore or lie inside it.",
                )
            _validate_worker_configuration_identity(
                worker_configuration_ref, worker_profile_sha256, verify_bytes=True,
            )
            store = ArtifactStore(root)
            envelopes = tuple(_read_selected_exact_revision(
                store, ref, consumer_stage=None if index == 0 else PATCH_CONSUMER_STAGE,
            ) for index, ref in enumerate(selected_refs))
            if (any(envelope.get("status") != "success" for envelope in (envelopes[0], envelopes[2]))
                    or envelopes[1].get("status") not in {"success", "blocked"}):
                raise PatchContinuationError(
                    "PATCH_CONTINUATION_ARTIFACT_STATUS_INVALID", "artifact_admission",
                    "A Product Graph worker requires successful selected Artifacts.",
                )
            _validate_exact_upstream_set(envelopes[1], (feature_spec_ref,))
            _validate_exact_upstream_set(envelopes[2], (feature_spec_ref, evidence_pack_ref))
            feature = strict_feature_spec_payload(envelopes[0]["data"])
            evidence = EvidencePack.from_artifact_payload(envelopes[1]["data"]).to_artifact_payload()
            plan = dict(envelopes[2]["data"])
            candidate_scope = _admit_compilation_candidate(evidence, feature,
                worker_configuration_ref=worker_configuration_ref, require_selected_worker=True)
            if envelopes[1].get("status") != "success" and candidate_scope is None:
                raise PatchContinuationError("PATCH_CONTINUATION_ARTIFACT_STATUS_INVALID", "artifact_admission",
                    "Blocked Evidence requires a current compilation candidate scope.")
            if feature["feature_id"] != feature_spec_ref.feature_id or evidence["query"] != feature["user_request"]:
                raise PatchContinuationError(
                    "PATCH_CONTINUATION_SEMANTIC_IDENTITY_MISMATCH", "artifact_admission",
                    "Selected FeatureSpec and EvidencePack semantic identities differ.",
                )
            _validate_selected_plan(plan, evidence, feature_id=feature_spec_ref.feature_id)
            before = _csharp_source_snapshot(workspace)
            source_validation = _validate_source_items(evidence, workspace)
            if any(_is_within(workspace / row["source_path"], root)
                   for row in source_validation["files"]):
                raise PatchContinuationError(
                    "PATCH_PRODUCT_JOURNAL_SOURCE_IN_STORE", "source_validation",
                    "Product target-source evidence must not resolve into the ArtifactStore.",
                )
            after = _csharp_source_snapshot(workspace)
            if before != after:
                raise PatchContinuationError(
                    "PATCH_CONTINUATION_SOURCE_HASH_MISMATCH", "source_validation",
                    "Product source changed while the worker journal was admitted.",
                )
        except PlanningContinuationError as exc:
            raise PatchContinuationError(
                exc.code.replace("PLANNING_", "PATCH_", 1), exc.stage, exc.reason,
            ) from exc
        except (OSError, TypeError, KeyError, ValueError) as exc:
            if isinstance(exc, PatchContinuationError):
                raise
            raise PatchContinuationError(
                "PATCH_PRODUCT_JOURNAL_INPUT_INVALID", "artifact_admission",
                "The Product Graph worker journal inputs failed admission.",
            ) from exc

        selection = {ref.artifact_type: _artifact_ref_dict(ref) for ref in selected_refs}
        snapshot_hash = _source_tree_snapshot_sha256(after)
        identity = {
            "continuation_selection": "product_graph_current_implementation_plan",
            "run_id": run_id, "request_id": request_id, "parent_run_id": parent_run_id,
            "artifact_store_root": str(root), "workspace_root": str(workspace),
            "selected_artifacts": selection, "source_snapshot_sha256": snapshot_hash,
            "max_model_calls": max_model_calls,
            **({"model_call_budget_policy": model_call_budget_policy} if model_call_budget_policy != "stage_and_total" else {}),
            "worker_configuration_ref": (dict(worker_configuration_ref)
                                         if worker_configuration_ref is not None else None),
            "worker_profile_sha256": worker_profile_sha256,
        }
        journal = cls.__new__(cls)
        locator = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]
        journal.path = root / "cr" / f"{locator}.json"
        journal._document = {
            "receipt_version": PATCH_CONTINUATION_RECEIPT_VERSION,
            "status": "admitted", "run_id": run_id, "request_id": request_id,
            "parent_run_id": parent_run_id, "feature_id": feature_spec_ref.feature_id,
            "continuation_selection": identity["continuation_selection"],
            "caller": {"workspace_root": str(workspace), "parent_run_id": parent_run_id},
            "semantic_authority_claimed": False, "selected_artifacts": selection,
            "latest_observations": {ref.artifact_type: _latest_observation(store, ref)
                                    for ref in selected_refs},
            "source_validation": source_validation,
            "generated_against_snapshot_sha256": snapshot_hash,
            "current_snapshot_sha256": snapshot_hash,
            "model_call_count": 0, "proposal_admitted": False, "artifacts": {}, "events": [],
            "attempt_identity_sha256": _canonical_sha256(identity),
            "max_model_calls": max_model_calls,
            **({"model_call_budget_policy": model_call_budget_policy} if model_call_budget_policy == "meter_only" else {}),
        }
        if worker_configuration_ref is not None:
            journal._document["worker_configuration_ref"] = dict(worker_configuration_ref)
            journal._document["worker_profile_sha256"] = worker_profile_sha256
        journal._write(exclusive=True)
        return journal

    def __init__(
        self,
        *,
        artifact_root: Path,
        run_id: str,
        request_id: str,
        request: PatchContinuationInputV1,
        admitted: _AdmittedPatchInputs,
        max_model_calls: int = 1,
    ) -> None:
        receipt_locator = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32]
        self.path = artifact_root / "cr" / f"{receipt_locator}.json"
        self._document: dict[str, Any] = {
            "receipt_version": PATCH_CONTINUATION_RECEIPT_VERSION,
            "status": "admitted",
            "run_id": run_id,
            "request_id": request_id,
            "feature_id": request.feature_spec_ref.feature_id,
            "continuation_selection": "explicit_exact_implementation_plan_ref",
            "semantic_authority_claimed": False,
            "selected_artifacts": {
                "feature_spec": _artifact_ref_dict(request.feature_spec_ref),
                "evidence_pack": _artifact_ref_dict(request.evidence_pack_ref),
                "implementation_plan": _artifact_ref_dict(
                    request.implementation_plan_ref
                ),
            },
            "latest_observations": admitted.latest_observations,
            "generated_against_snapshot_sha256": _source_tree_snapshot_sha256(
                admitted.source_tree_snapshot
            ),
            "current_snapshot_sha256": _source_tree_snapshot_sha256(
                admitted.source_tree_snapshot
            ),
            "model_call_count": 0,
            "proposal_admitted": False,
            "artifacts": {},
            "events": [],
            "attempt_identity_sha256": _patch_attempt_identity(request, max_model_calls),
            "max_model_calls": max_model_calls,
        }
        if request.worker_configuration_ref is not None:
            self._document["worker_configuration_ref"] = dict(request.worker_configuration_ref)
            self._document["worker_profile_sha256"] = request.worker_profile_sha256
        if request.qualified_model_routing_ref is not None:
            self._document["qualified_model_routing_ref"] = dict(request.qualified_model_routing_ref)
        if admitted.repair_context is not None:
            self._document["repair_context"] = {
                key: value for key, value in admitted.repair_context.items()
                if key not in {"previous_response", "failure_report", "failed_diff_intent"}
            }
        if admitted.saved_patch is not None:
            saved = admitted.saved_patch
            self._document.update(
                continuation_selection="saved_generated_patch_reassembly",
                generation_origin="reused_saved_response",
                parent_run_id=saved["parent_run_id"],
                inherited_model_budget=saved["inherited_model_budget"],
                logical_budget=saved["logical_budget"],
                new_provider_sends=0,
                new_compilations=0,
                saved_patch={
                    "source_attempt_ref": saved["source_attempt_ref"],
                    "source_proposal_ref": _artifact_ref_dict(saved["source_proposal_ref"]),
                    "source_review_ref": _artifact_ref_dict(saved["source_review_ref"]),
                    "response_ref": saved["response_ref"],
                    "historical_verification": saved["historical_verification"],
                },
            )
        if admitted.saved_candidate is not None:
            saved = admitted.saved_candidate
            self._document.update(continuation_selection="saved_candidate_reverification",
                generation_origin="retained_live_response", parent_run_id=saved["parent_run_id"],
                inherited_model_budget=saved["inherited_model_budget"], new_provider_sends=0,
                saved_candidate={key: deepcopy(saved[key]) for key in ("source_attempt_ref", "response_ref")})
        self._write(exclusive=True)

    def record(self, event: Mapping[str, Any]) -> None:
        phase = event.get("phase")
        if type(phase) is not str or not phase:
            raise PatchContinuationError(
                "PATCH_CONTINUATION_RECEIPT_EVENT_INVALID",
                "patch_attempt_receipt",
                "Patch attempt receipt event requires a non-empty phase.",
            )
        normalized = json.loads(
            json.dumps(
                dict(event),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        normalized["event_index"] = len(self._document["events"]) + 1
        self._document["events"].append(normalized)
        for key, value in normalized.items():
            if key not in {"phase", "event_index"}:
                if key == "artifacts" and isinstance(value, Mapping):
                    current = self._document.get("artifacts")
                    self._document["artifacts"] = {
                        **(dict(current) if isinstance(current, Mapping) else {}),
                        **dict(value),
                    }
                else:
                    self._document[key] = value
        self._document["status"] = phase
        self._write()

    def record_observation(self, observation: Mapping[str, Any]) -> None:
        """Upsert measured execution state without changing workflow status/budget."""
        normalized = json.loads(json.dumps(dict(observation), allow_nan=False))
        rows = self._document.setdefault("execution_observations", {})
        rows[normalized["id"]] = normalized
        self._write()

    def record_trace_summary(self, summary: Mapping[str, Any]) -> None:
        self._document["execution_trace"] = json.loads(json.dumps(dict(summary), allow_nan=False))
        self._write()

    def artifact_ref(self, artifact_type: str) -> ArtifactRef | None:
        artifacts = self._document.get("artifacts")
        value = artifacts.get(artifact_type) if isinstance(artifacts, Mapping) else None
        if not isinstance(value, Mapping):
            return None
        feature_id = value.get("feature_id")
        observed_type = value.get("artifact_type")
        path = value.get("path")
        artifact_hash = value.get("artifact_hash")
        if (
            type(feature_id) is not str
            or not feature_id
            or observed_type != artifact_type
            or type(path) is not str
            or not Path(path).is_absolute()
            or not _is_sha256(artifact_hash)
        ):
            return None
        return ArtifactRef(
            feature_id=feature_id,
            artifact_type=artifact_type,
            path=path,
            artifact_hash=artifact_hash,
        )

    def projection(self) -> dict[str, Any]:
        payload = self.path.read_bytes()
        return {
            "receipt_version": PATCH_CONTINUATION_RECEIPT_VERSION,
            "path": str(self.path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "status": self._document["status"],
            **({"execution_trace": self._document["execution_trace"]}
               if "execution_trace" in self._document else {}),
            "event_count": len(self._document["events"]),
            "model_call_count": self._document.get("model_call_count", 0),
            "proposal_admitted": self._document.get("proposal_admitted", False),
            "generated_against_snapshot_sha256": self._document.get(
                "generated_against_snapshot_sha256"
            ),
            "current_snapshot_sha256": self._document.get(
                "current_snapshot_sha256"
            ),
        }

    def _write(self, *, exclusive: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        payload = (
            json.dumps(
                self._document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            if exclusive:
                with self.path.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                return
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise PatchContinuationError(
                "PATCH_CONTINUATION_RECEIPT_WRITE_FAILED",
                "patch_attempt_receipt",
                "Patch attempt receipt could not be persisted atomically.",
            ) from exc


def blocked_planning_continuation_result(
    *,
    code: str,
    stage: str,
    duration_seconds: float,
    reason: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
    feature_id: str | None = None,
    feature_spec_ref: ArtifactRef | None = None,
    evidence_pack_ref: ArtifactRef | None = None,
    implementation_plan_ref: ArtifactRef | None = None,
    plan_blocking: bool | None = None,
    artifact_write: ArtifactWriteResult | Mapping[str, Any] | None = None,
    planning_execution_count: int = 0,
    source_validation: Mapping[str, Any] | None = None,
    source_modified_during_run: bool | None = None,
    status: Literal["blocked", "partial"] = "blocked",
) -> PlanningContinuationResultV1:
    return PlanningContinuationResultV1(
        status=status,
        terminal_stage=stage,
        first_failure={"code": code, "stage": stage},
        run_id=run_id or f"planning_continuation_{uuid4().hex}",
        request_id=request_id or f"planning_request_{uuid4().hex}",
        feature_id=feature_id,
        feature_spec_artifact=(
            None if feature_spec_ref is None else _artifact_ref_dict(feature_spec_ref)
        ),
        evidence_pack_artifact=(
            None if evidence_pack_ref is None else _artifact_ref_dict(evidence_pack_ref)
        ),
        implementation_plan_artifact=(
            None if implementation_plan_ref is None else _artifact_ref_dict(implementation_plan_ref)
        ),
        plan_blocking=plan_blocking,
        artifact_write=_artifact_write_projection(artifact_write),
        source_validation=dict(
            source_validation
            or {
                "status": "not_completed",
                "checked_item_count": 0,
                "checked_file_count": 0,
            }
        ),
        planning_execution_count=planning_execution_count,
        source_modified_during_run=source_modified_during_run,
        duration_seconds=duration_seconds,
        reason=reason or code,
    )


def _plan_ref_from_state(
    state: Mapping[str, Any],
    feature_id: str,
) -> ArtifactRef:
    artifacts = state.get("artifacts")
    artifact_hashes = state.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or not isinstance(artifact_hashes, Mapping):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_PLAN_REF_MISSING",
            "implementation_plan",
            "Planning graph returned no Artifact reference maps.",
        )
    path = artifacts.get("implementation_plan")
    artifact_hash = artifact_hashes.get("implementation_plan")
    if type(path) is not str or not path or not _is_sha256(artifact_hash):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_PLAN_REF_MISSING",
            "implementation_plan",
            "Planning graph did not persist an exact ImplementationPlan ref.",
        )
    return ArtifactRef(
        feature_id=feature_id,
        artifact_type="implementation_plan",
        path=path,
        artifact_hash=artifact_hash,
    )


def _plan_ref_from_write_result(
    store: ArtifactStore,
    write_result: ArtifactWriteResult,
) -> ArtifactRef | None:
    if write_result.artifact_ref is not None:
        return write_result.artifact_ref
    if (
        write_result.revision_state != "valid"
        or not _is_sha256(write_result.artifact_hash)
        or type(write_result.revision_path) is not str
        or not write_result.revision_path
    ):
        return None
    ref = ArtifactRef(
        feature_id=write_result.feature_id,
        artifact_type=write_result.artifact_type,
        path=write_result.revision_path,
        artifact_hash=write_result.artifact_hash,
    )
    try:
        store.read(ref)
    except ArtifactStoreError:
        return None
    return ref


def _planning_trace_count(state: Mapping[str, Any]) -> int:
    trace_events = state.get("trace_events")
    if not isinstance(trace_events, list):
        return 0
    return sum(
        1
        for event in trace_events
        if isinstance(event, Mapping)
        and event.get("node_name") == "create_implementation_plan"
        and event.get("event_type") == "node_started"
    )


def _write_result_from_state(state: Mapping[str, Any]) -> ArtifactWriteResult:
    runtime_summary = state.get("runtime_summary")
    raw_result = (
        runtime_summary.get("planning_artifact_write_result")
        if isinstance(runtime_summary, Mapping)
        else None
    )
    if not isinstance(raw_result, Mapping):
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_WRITE_RESULT_MISSING",
            "implementation_plan",
            "Planning graph returned no exact Artifact write result.",
        )
    try:
        return ArtifactWriteResult.from_dict(raw_result)
    except ArtifactStoreError as exc:
        raise PlanningContinuationError(
            "PLANNING_CONTINUATION_WRITE_RESULT_INVALID",
            "implementation_plan",
            "Planning graph Artifact write result is invalid.",
        ) from exc


def _plan_blocking_from_ref(
    store: ArtifactStore,
    ref: ArtifactRef | None,
) -> bool | None:
    if ref is None:
        return None
    try:
        envelope = store.read(ref)
    except ArtifactStoreError:
        return None
    data = envelope.get("data")
    if not isinstance(data, Mapping) or type(data.get("blocking")) is not bool:
        return None
    return data["blocking"]


def _first_plan_failure(state: Mapping[str, Any]) -> dict[str, str] | None:
    blockers = state.get("blockers")
    if not isinstance(blockers, list):
        return {
            "code": "PLANNING_CONTINUATION_BLOCKERS_INVALID",
            "stage": "implementation_plan",
        }
    for blocker in blockers:
        if not isinstance(blocker, Mapping):
            continue
        code = blocker.get("code")
        if isinstance(code, str) and code:
            return {"code": code, "stage": "implementation_plan"}
    if blockers:
        return {
            "code": "IMPLEMENTATION_PLAN_BLOCKED",
            "stage": "implementation_plan",
        }
    return None


def _continuation_state(
    *,
    run_id: str,
    request_id: str,
    request: PlanningContinuationInputV1,
    admitted: _AdmittedPlanningInputs,
):
    from src.mod_agent.workflow.state import create_initial_state

    feature_spec = admitted.feature_spec
    evidence_pack = admitted.evidence_pack
    candidate_scope = compilation_candidate_scope(evidence_pack, feature_spec)
    state = create_initial_state(
        run_id=run_id,
        request_id=request_id,
        user_request=feature_spec["user_request"],
        target_mod=feature_spec["target_mod"],
    )
    state["feature_id"] = request.feature_spec_ref.feature_id
    state["risk_level"] = feature_spec["risk_level"]
    state["current_stage"] = "evidence_retrieval"
    state["next_stage"] = "implementation_plan"
    state["artifacts"] = {
        "feature_spec": request.feature_spec_ref.path,
        "evidence_pack": request.evidence_pack_ref.path,
    }
    state["artifact_hashes"] = {
        "feature_spec": request.feature_spec_ref.artifact_hash,
        "evidence_pack": request.evidence_pack_ref.artifact_hash,
    }
    state["evidence_summary"] = {
        "readiness": evidence_pack["readiness"],
        "can_generate_patch": evidence_pack["can_generate_patch"],
        "num_items": len(evidence_pack["items"]),
        "gaps": list(evidence_pack["gaps"]),
        "retrieval_methods_used": list(evidence_pack["retrieval_methods_used"]),
        "missing_evidence": list(evidence_pack["missing_evidence"]),
        "disabled_modes": dict(evidence_pack["disabled_modes"]),
        "timing": dict(evidence_pack["timing"]),
        **({"compilation_repair_candidate": candidate_scope} if candidate_scope is not None else {}),
    }
    return state


def _source_tree_changed_since(
    request: PlanningContinuationInputV1,
    before: tuple[tuple[str, str], ...],
) -> bool | None:
    try:
        _, workspace_root = _resolve_roots(request)
        return _patch_request_source_snapshot(request) != before
    except PlanningContinuationError:
        return None


def run_fresh_planning_continuation(
    request: PlanningContinuationInputV1,
) -> PlanningContinuationResultV1:
    """Run exact Fresh inputs through Planning once and stop before Patch."""

    started = time.perf_counter()
    run_id = f"planning_continuation_{uuid4().hex}"
    request_id = f"planning_request_{uuid4().hex}"
    feature_id = request.feature_spec_ref.feature_id
    try:
        admitted = preflight_planning_continuation(request)
    except PlanningContinuationError as exc:
        return blocked_planning_continuation_result(
            code=exc.code,
            stage=exc.stage,
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=request.feature_spec_ref,
            evidence_pack_ref=request.evidence_pack_ref,
        )

    from src.mod_agent.workflow.graph import build_graph

    state = _continuation_state(
        run_id=run_id,
        request_id=request_id,
        request=request,
        admitted=admitted,
    )
    try:
        graph = build_graph(
            admitted.artifact_store,
            planning_continuation_mode=request,
            use_default_proposal_git_binding=False,
        )
    except (PlanningContinuationError, RuntimeError, ValueError) as exc:
        code = getattr(exc, "code", "PLANNING_CONTINUATION_GRAPH_INVALID")
        return blocked_planning_continuation_result(
            code=code if isinstance(code, str) and code else "PLANNING_CONTINUATION_GRAPH_INVALID",
            stage=getattr(exc, "stage", "planning_admission"),
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=request.feature_spec_ref,
            evidence_pack_ref=request.evidence_pack_ref,
            source_validation={
                "status": "passed_before_planning",
                "before": admitted.source_validation,
            },
        )

    try:
        graph_result = graph.invoke(state)
    except (ArtifactStoreError, PlanningContinuationError, RuntimeError, ValueError) as exc:
        node_started = getattr(exc, "_mod_agent_graph_node", None) == "create_implementation_plan"
        write_result = (
            exc.write_result if isinstance(exc, PlanningContinuationArtifactWriteError) else None
        )
        plan_ref = (
            _plan_ref_from_write_result(admitted.artifact_store, write_result)
            if write_result is not None
            else None
        )
        if write_result is not None:
            code = (
                "PLANNING_CONTINUATION_PLAN_STORE_PARTIAL"
                if write_result.outcome == "partial"
                else "PLANNING_CONTINUATION_PLAN_STORE_BLOCKED"
            )
            stage = "implementation_plan_store"
        else:
            code = getattr(
                exc,
                "code",
                "PLANNING_CONTINUATION_EXECUTION_FAILED",
            )
            if not isinstance(code, str) or not code:
                code = "PLANNING_CONTINUATION_EXECUTION_FAILED"
            stage = getattr(exc, "stage", "implementation_plan")
        is_partial = write_result is not None and (
            write_result.outcome == "partial" or write_result.commit_started or plan_ref is not None
        )
        return blocked_planning_continuation_result(
            code=code,
            stage=stage,
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=request.feature_spec_ref,
            evidence_pack_ref=request.evidence_pack_ref,
            implementation_plan_ref=plan_ref,
            plan_blocking=_plan_blocking_from_ref(
                admitted.artifact_store,
                plan_ref,
            ),
            artifact_write=write_result,
            planning_execution_count=1 if node_started else 0,
            source_validation={
                "status": (
                    "failed_during_planning"
                    if _source_tree_changed_since(
                        request,
                        admitted.source_tree_snapshot,
                    )
                    else "passed_before_planning"
                ),
                "before": admitted.source_validation,
            },
            source_modified_during_run=_source_tree_changed_since(
                request,
                admitted.source_tree_snapshot,
            ),
            status="partial" if is_partial else "blocked",
        )

    planning_execution_count = _planning_trace_count(graph_result)
    plan_ref: ArtifactRef | None = None
    write_result: ArtifactWriteResult | None = None
    try:
        write_result = _write_result_from_state(graph_result)
        plan_ref = _plan_ref_from_state(graph_result, feature_id)
        if write_result.artifact_ref != plan_ref:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_PLAN_REF_MISMATCH",
                "implementation_plan",
                "Planning state and Artifact write result refs disagree.",
            )
        if planning_execution_count != 1:
            raise PlanningContinuationError(
                "PLANNING_CONTINUATION_EXECUTION_COUNT_INVALID",
                "implementation_plan",
                "Planning graph did not record exactly one Planning start.",
            )
        plan_envelope = admitted.artifact_store.read(plan_ref)
    except (ArtifactStoreError, PlanningContinuationError) as exc:
        code = getattr(
            exc,
            "code",
            "PLANNING_CONTINUATION_PLAN_READBACK_FAILED",
        )
        return blocked_planning_continuation_result(
            code=code
            if isinstance(code, str) and code
            else "PLANNING_CONTINUATION_PLAN_READBACK_FAILED",
            stage=getattr(exc, "stage", "implementation_plan"),
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=request.feature_spec_ref,
            evidence_pack_ref=request.evidence_pack_ref,
            implementation_plan_ref=plan_ref,
            plan_blocking=_plan_blocking_from_ref(
                admitted.artifact_store,
                plan_ref,
            ),
            artifact_write=write_result,
            planning_execution_count=planning_execution_count,
            source_validation={
                "status": "passed_before_planning",
                "before": admitted.source_validation,
            },
            source_modified_during_run=_source_tree_changed_since(
                request,
                admitted.source_tree_snapshot,
            ),
            status="partial" if plan_ref is not None else "blocked",
        )

    plan_data = plan_envelope["data"]
    if not isinstance(plan_data, Mapping) or type(plan_data.get("blocking")) is not bool:
        return blocked_planning_continuation_result(
            code="PLANNING_CONTINUATION_PLAN_PAYLOAD_INVALID",
            stage="implementation_plan",
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=request.feature_spec_ref,
            evidence_pack_ref=request.evidence_pack_ref,
            implementation_plan_ref=plan_ref,
            artifact_write=write_result,
            planning_execution_count=planning_execution_count,
            source_validation={
                "status": "passed_before_planning",
                "before": admitted.source_validation,
            },
        )

    try:
        after = preflight_planning_continuation(request)
    except PlanningContinuationError as exc:
        source_modified: bool | None = None
        try:
            _, workspace_root = _resolve_roots(request)
            source_modified = (
                _csharp_source_snapshot(workspace_root) != admitted.source_tree_snapshot
            )
        except PlanningContinuationError:
            pass
        return blocked_planning_continuation_result(
            code=exc.code,
            stage=exc.stage,
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=request.feature_spec_ref,
            evidence_pack_ref=request.evidence_pack_ref,
            implementation_plan_ref=plan_ref,
            plan_blocking=bool(plan_data["blocking"]),
            artifact_write=write_result,
            planning_execution_count=planning_execution_count,
            source_validation={
                "status": "failed_after_planning",
                "before": admitted.source_validation,
            },
            source_modified_during_run=source_modified,
            status="partial",
        )

    source_modified = after.source_tree_snapshot != admitted.source_tree_snapshot
    if source_modified:
        return blocked_planning_continuation_result(
            code="PLANNING_CONTINUATION_SOURCE_CHANGED",
            stage="source_validation",
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=request.feature_spec_ref,
            evidence_pack_ref=request.evidence_pack_ref,
            implementation_plan_ref=plan_ref,
            plan_blocking=bool(plan_data["blocking"]),
            artifact_write=write_result,
            planning_execution_count=planning_execution_count,
            source_validation={
                "status": "failed_after_planning",
                "before": admitted.source_validation,
                "after": after.source_validation,
            },
            source_modified_during_run=True,
            status="partial",
        )

    plan_blocking = plan_data["blocking"]
    first_failure = _first_plan_failure(graph_result)
    status: Literal["terminal", "blocked"] = "blocked" if plan_blocking else "terminal"
    return PlanningContinuationResultV1(
        status=status,
        terminal_stage="implementation_plan",
        first_failure=first_failure,
        run_id=run_id,
        request_id=request_id,
        feature_id=feature_id,
        feature_spec_artifact=_artifact_ref_dict(request.feature_spec_ref),
        evidence_pack_artifact=_artifact_ref_dict(request.evidence_pack_ref),
        implementation_plan_artifact=_artifact_ref_dict(plan_ref),
        plan_blocking=plan_blocking,
        artifact_write=_artifact_write_projection(write_result),
        source_validation={
            "status": "passed",
            "before": admitted.source_validation,
            "after_snapshot_sha256": after.source_validation["snapshot_sha256"],
            "source_tree_unchanged": True,
        },
        planning_execution_count=planning_execution_count,
        source_modified_during_run=False,
        duration_seconds=time.perf_counter() - started,
        reason=(None if first_failure is None else first_failure["code"]),
    )


def blocked_patch_continuation_result(
    *,
    code: str,
    stage: str,
    duration_seconds: float,
    reason: str | None = None,
    run_id: str | None = None,
    request_id: str | None = None,
    feature_id: str | None = None,
    feature_spec_ref: ArtifactRef | None = None,
    evidence_pack_ref: ArtifactRef | None = None,
    implementation_plan_ref: ArtifactRef | None = None,
    patch_proposal_ref: ArtifactRef | None = None,
    proposal_review_ref: ArtifactRef | None = None,
    dry_run_ref: ArtifactRef | None = None,
    source_validation: Mapping[str, Any] | None = None,
    execution_counts: Mapping[str, int] | None = None,
    attempt_receipt: Mapping[str, Any] | None = None,
    source_modified_during_run: bool | None = None,
    status: Literal["blocked", "partial"] = "blocked",
) -> PatchContinuationResultV1:
    counts = {
        "fresh_coverage": 0,
        "model": 0,
        "planning": 0,
        "patch": 0,
        "review": 0,
        "dry_run": 0,
        "apply": 0,
        "build": 0,
        "runtime": 0,
    }
    counts.update(dict(execution_counts or {}))
    return PatchContinuationResultV1(
        status=status,
        terminal_stage=stage,
        first_failure={"code": code, "stage": stage},
        run_id=run_id or f"patch_continuation_{uuid4().hex}",
        request_id=request_id or f"patch_request_{uuid4().hex}",
        feature_id=feature_id,
        feature_spec_artifact=(
            None if feature_spec_ref is None else _artifact_ref_dict(feature_spec_ref)
        ),
        evidence_pack_artifact=(
            None if evidence_pack_ref is None else _artifact_ref_dict(evidence_pack_ref)
        ),
        implementation_plan_artifact=(
            None
            if implementation_plan_ref is None
            else _artifact_ref_dict(implementation_plan_ref)
        ),
        patch_proposal_artifact=(
            None if patch_proposal_ref is None else _artifact_ref_dict(patch_proposal_ref)
        ),
        proposal_review_artifact=(
            None if proposal_review_ref is None else _artifact_ref_dict(proposal_review_ref)
        ),
        dry_run_artifact=(
            None if dry_run_ref is None else _artifact_ref_dict(dry_run_ref)
        ),
        source_validation=dict(
            source_validation
            or {
                "status": "not_completed",
                "checked_item_count": 0,
                "checked_file_count": 0,
            }
        ),
        execution_counts=counts,
        attempt_receipt=(
            None if attempt_receipt is None else dict(attempt_receipt)
        ),
        source_modified_during_run=source_modified_during_run,
        duration_seconds=duration_seconds,
        reason=reason or code,
    )


def _patch_continuation_state(
    *,
    run_id: str,
    request_id: str,
    request: PatchContinuationInputV1,
    admitted: _AdmittedPatchInputs,
):
    from src.mod_agent.workflow.state import create_initial_state

    state = create_initial_state(
        run_id=run_id,
        request_id=request_id,
        user_request=admitted.feature_spec["user_request"],
        target_mod=admitted.feature_spec["target_mod"],
    )
    state["feature_id"] = request.feature_spec_ref.feature_id
    state["risk_level"] = admitted.feature_spec["risk_level"]
    state["current_stage"] = "implementation_plan"
    state["next_stage"] = "patch_proposal"
    state["artifacts"] = {
        "feature_spec": request.feature_spec_ref.path,
        "evidence_pack": request.evidence_pack_ref.path,
        "implementation_plan": request.implementation_plan_ref.path,
    }
    state["artifact_hashes"] = {
        "feature_spec": request.feature_spec_ref.artifact_hash,
        "evidence_pack": request.evidence_pack_ref.artifact_hash,
        "implementation_plan": request.implementation_plan_ref.artifact_hash,
    }
    if admitted.repair_context is not None:
        state["current_stage"] = "behavior_report"
        state["next_stage"] = "repair_plan"
        state["retry_count"] = admitted.repair_context["parent_retry_count"]
        for ref in (request.repair_parent_proposal_ref, request.repair_parent_review_ref):
            state["artifacts"][ref.artifact_type] = ref.path
            state["artifact_hashes"][ref.artifact_type] = ref.artifact_hash
    return state


def _bind_patch_attempt_recorder(
    factory: PatchContinuationFactory,
    recorder: Callable[[Mapping[str, Any]], None],
) -> None:
    binder = getattr(factory, "bind_attempt_recorder", None)
    if binder is None:
        return
    if not callable(binder):
        raise PatchContinuationError(
            "PATCH_CONTINUATION_FACTORY_RECORDER_INVALID",
            "patch_attempt_receipt",
            "Patch factory recorder binder must be callable.",
        )
    binder(recorder)


def _ref_from_patch_state(
    state: Mapping[str, Any],
    artifact_type: str,
    *,
    feature_id: str,
) -> ArtifactRef | None:
    artifacts = state.get("artifacts")
    hashes = state.get("artifact_hashes")
    if not isinstance(artifacts, Mapping) or not isinstance(hashes, Mapping):
        return None
    path = artifacts.get(artifact_type)
    artifact_hash = hashes.get(artifact_type)
    if type(path) is not str or not path or not _is_sha256(artifact_hash):
        return None
    return ArtifactRef(
        feature_id=feature_id,
        artifact_type=artifact_type,
        path=path,
        artifact_hash=artifact_hash,
    )


def _node_execution_count(state: Mapping[str, Any], node_name: str) -> int:
    trace_events = state.get("trace_events")
    if not isinstance(trace_events, list):
        return 0
    return sum(
        1
        for event in trace_events
        if isinstance(event, Mapping)
        and event.get("node_name") == node_name
        and event.get("event_type") == "node_started"
    )


def _patch_execution_counts(
    state: Mapping[str, Any],
    *,
    model_call_count: int,
) -> dict[str, int]:
    return {
        "fresh_coverage": 0,
        "model": model_call_count,
        "planning": _node_execution_count(state, "create_implementation_plan"),
        "patch": _node_execution_count(state, "generate_patch_proposal"),
        "review": _node_execution_count(state, "review_proposal"),
        "dry_run": _node_execution_count(state, "dry_run_apply"),
        "apply": 0,
        "build": 0,
        "runtime": 0,
    }


def _failed_patch_execution_counts(
    failed_node: str | None,
    *,
    model_call_count: int,
) -> dict[str, int]:
    ordered = (
        "generate_patch_proposal",
        "review_proposal",
        "dry_run_apply",
    )
    reached = ordered.index(failed_node) + 1 if failed_node in ordered else 0
    return {
        "fresh_coverage": 0,
        "model": model_call_count,
        "planning": 0,
        "patch": int(reached >= 1),
        "review": int(reached >= 2),
        "dry_run": int(reached >= 3),
        "apply": 0,
        "build": 0,
        "runtime": 0,
    }


def _patch_first_failure(state: Mapping[str, Any]) -> dict[str, str] | None:
    blockers = state.get("blockers")
    if not isinstance(blockers, list):
        return {
            "code": "PATCH_CONTINUATION_BLOCKERS_INVALID",
            "stage": str(state.get("current_stage") or "workflow"),
        }
    for blocker in blockers:
        if not isinstance(blocker, Mapping):
            continue
        code = blocker.get("code")
        if type(code) is str and code:
            return {
                "code": code,
                "stage": str(blocker.get("stage") or state.get("current_stage") or "workflow"),
            }
    return None


def _proposal_source_snippet_char_limit(admitted: _AdmittedPatchInputs) -> int:
    from src.mod_agent.proposal import PatchProposalBuilder

    return PatchProposalBuilder.required_source_snippet_char_limit(
        admitted.implementation_plan, admitted.evidence_pack,
    )


def _patch_attempt_identity(request: PatchContinuationInputV1, max_model_calls: int) -> str:
    identity_request = asdict(request)
    if request.saved_patch_selection is None:
        identity_request.pop("saved_patch_selection")
    if request.candidate_reverification_selection is None:
        identity_request.pop("candidate_reverification_selection")
    if request.patch_upstream_configuration_ref is None:
        identity_request.pop("patch_upstream_configuration_ref")
    if request.qualified_model_routing_ref is None:
        identity_request.pop("qualified_model_routing_ref")
    if request.worker_configuration_ref is None and request.worker_profile_sha256 is None:
        # Existing attempts predate this optional input; keep their exact identity.
        identity_request.pop("worker_configuration_ref")
        identity_request.pop("worker_profile_sha256")
    if request.repair_source_attempt_ref is None and request.repair_verification_ref is None:
        identity_request.pop("repair_source_attempt_ref")
        identity_request.pop("repair_verification_ref")
    return _canonical_sha256({"request": identity_request, "max_model_calls": max_model_calls})


def _attempt_receipt_projection(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "receipt_version": PATCH_CONTINUATION_RECEIPT_VERSION,
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": document.get("status"), "event_count": len(document.get("events", [])),
        "model_call_count": document.get("model_call_count", 0),
        "proposal_admitted": document.get("proposal_admitted", False),
        "generated_against_snapshot_sha256": document.get("generated_against_snapshot_sha256"),
        "current_snapshot_sha256": document.get("current_snapshot_sha256"),
        **({"execution_trace": document["execution_trace"]} if "execution_trace" in document else {}),
        **({key: document[key] for key in (
            "generation_origin", "saved_patch", "inherited_model_budget", "logical_budget",
            "new_provider_sends", "new_compilations", "parent_run_id")}
           if "saved_patch" in document else {}),
        **({key: document[key] for key in (
            "generation_origin", "saved_candidate", "inherited_model_budget", "new_provider_sends", "parent_run_id")}
           if "saved_candidate" in document else {}),
    }


class _PatchUpstreamExecution:
    """Per-invocation adapters and budget; Graph edges still own every transition."""

    def __init__(self, request, factory, configuration, journal, maximum):
        from src.mod_agent.patch_upstream_configuration import PatchUpstreamConfiguration
        if not isinstance(configuration, PatchUpstreamConfiguration) or configuration.reference != request.patch_upstream_configuration_ref:
            raise PatchContinuationError("PATCH_UPSTREAM_CONFIGURATION_INVALID", "request", "The selected providers require their exact configuration reference.")
        self.request, self.factory, self.configuration, self.journal = request, factory, configuration, journal
        self.maximum, self.used, self.allocation = maximum, 0, maximum
        self.roles: dict[str, int] = {}
        self.segments: list[dict[str, Any]] = []
        self.replan_count = 0
        self.initial_selection = self.selection(request)
        admitted = preflight_patch_continuation(request)
        self.source_snippet_char_limit = _proposal_source_snippet_char_limit(admitted)
        constraint_options = {}
        if admitted.implementation_plan.get("constraint_context") is not None:
            from src.mod_agent.planning.constraint_context import validate_constraint_context

            # Scope: the already admitted Plan selects the explicit experimental representation.
            # 为什么要有它：独立上游重建不能默默退回A；只继承重新核准的来源策略，不读取环境变量或继承编辑批准。
            context = validate_constraint_context(admitted.implementation_plan["constraint_context"],
                feature_spec=admitted.feature_spec, evidence_pack=admitted.evidence_pack,
                constraints=admitted.implementation_plan["side_effect_boundaries"])
            constraint_options["constraint_policy"] = context["policy"]
        self.providers = configuration.assemble(request, recorder=self.record,
            reserve_model_call=self.reserve_model_call, **constraint_options)
        self._attach(factory)

    @staticmethod
    def selection(request):
        return {kind: _artifact_ref_dict(getattr(request, kind + "_ref"))
                for kind in ("feature_spec", "evidence_pack", "implementation_plan")}

    def reserve_model_call(self, role):
        if self.used >= self.maximum:
            raise PatchContinuationError("PATCH_CONTINUATION_MODEL_CALL_BUDGET_EXHAUSTED", "model_admission", "Patch and upstream providers have consumed the shared invocation budget.")
        self.used += 1
        self.roles[role] = self.roles.get(role, 0) + 1
        self.record({"phase": "shared_model_call_reserved", "role": role})

    def record(self, event):
        current = dict(event)
        current["model_call_count"] = self.used
        current["shared_model_budget"] = {"maximum": self.maximum, "consumed": self.used,
            "remaining": self.maximum - self.used, "by_role": dict(self.roles)}
        if "factory_receipt" in current:
            self.segments[-1]["factory_receipt"] = current["factory_receipt"]
            current["patch_execution_segments"] = list(self.segments)
        self.journal.record(current)

    def _attach(self, factory):
        self.segments.append({"segment_index": len(self.segments) + 1,
            "selected_artifacts": self.selection(self.request),
            "allocated_model_calls": self.allocation, "factory_receipt": None})
        factory._shared_model_call_reserver = self.reserve_model_call
        _bind_patch_attempt_recorder(factory, self.record)

    def after_plan(self, state):
        """Continue an explicitly assessed Plan within the same shared budget."""
        if state.get("blockers") or not state["runtime_summary"].get("patch_upstream_return"):
            return state
        from copy import deepcopy

        from src.mod_agent.repair.upstream import prepare_replanned_patch_input
        next_state = deepcopy(state)
        # This invocation accepts one complete upstream cycle. Further recovery
        # retains its own failed output rather than silently granting another run.
        if self.replan_count or self.used >= self.maximum:
            next_state["blockers"].append({"stage": "implementation_plan",
                "code": "PATCH_UPSTREAM_CONTINUATION_BUDGET_EXHAUSTED",
                "message": "No further upstream cycle or model call is available in this invocation."})
            return next_state
        previous_request = self.request
        assessment = state["runtime_summary"].get("planning_upstream_assessment") or {}
        retain_plan = assessment.get("decision") == "retain_current_plan"
        if retain_plan and (
                assessment.get("implementation_plan") != {
                    "path": previous_request.implementation_plan_ref.path,
                    "artifact_hash": previous_request.implementation_plan_ref.artifact_hash,
                }
                or any(state["artifacts"].get(kind) != ref["path"]
                       or state["artifact_hashes"].get(kind) != ref["artifact_hash"]
                       for kind, ref in self.selection(previous_request).items())
                or assessment.get("repair_plan_hash") != state["artifact_hashes"].get("repair_plan")):
            raise PatchContinuationError("PATCH_UPSTREAM_RETAIN_PLAN_IDENTITY_CHANGED", "upstream_admission",
                "Retaining the Plan requires the exact assessed Feature, Evidence and Plan references.")
        retained_request = self.factory.retained_worker_request(complete_upstream=True) if retain_plan else None
        retained_calibration = self.factory.retained_context_calibration() if retain_plan else None
        retained_input_projection = self.factory._repair_input_projection if retain_plan else None
        previous_factory_receipt = self.factory.patch_attempt_receipt()
        previous_patch_calls = previous_factory_receipt["model_call_count"]
        from src.mod_agent.proposal.tool_argument_feedback import receipt_correction_used
        inherited_argument_correction = receipt_correction_used(previous_factory_receipt)
        self.request = prepare_replanned_patch_input(prior_request=previous_request, state=state,
            attempt_id=("retained_" if retain_plan else "replanned_") + _canonical_sha256(self.initial_selection)[:24])
        self.request = replace(self.request, worker_configuration_ref=previous_request.worker_configuration_ref,
            worker_profile_sha256=previous_request.worker_profile_sha256,
            patch_upstream_configuration_ref=previous_request.patch_upstream_configuration_ref)
        newly_admitted = preflight_patch_continuation(self.request)
        self.source_snippet_char_limit = _proposal_source_snippet_char_limit(newly_admitted)
        self.allocation = self.maximum - self.used
        previous = preflight_patch_continuation(previous_request)
        context = admit_active_worker_failure(previous_request, self.factory.patch_attempt_receipt(),
            previous.artifact_store).repair_context
        repair_ref = {"feature_id": state["feature_id"], "artifact_type": "repair_plan",
            "path": state["artifacts"]["repair_plan"], "artifact_hash": state["artifact_hashes"]["repair_plan"]}
        old_policy = self.factory.patch_agent_loop_policy
        consumed_reads, consumed_source_bytes = self.factory._recovery_observation_usage()
        consumed_observation_bytes = consumed_source_bytes + self.factory._recovery_observation_bytes
        remaining_reads = old_policy.max_tool_rounds - consumed_reads
        remaining_bytes = old_policy.max_observation_bytes - consumed_observation_bytes
        if min(remaining_reads, remaining_bytes) < 0 or (
                not retain_plan and remaining_reads == 0 and old_policy.require_source_inspection):
            raise PatchContinuationError("PATCH_UPSTREAM_CONTINUATION_BUDGET_EXHAUSTED", "model_admission",
                "The original observation policy leaves no legal allowance for the next worker.")
        self.factory = self.configuration.make_replanned_adapter(self.request, self.allocation)
        # Scope: both retained and replaced Plan continuations under this original invocation.
        # 为什么要有它：新factory不含旧native history时仍须继承已用机会，但不继承旧源码身份或编辑权限。
        if inherited_argument_correction:
            self.factory.bind_tool_argument_correction_used(True)
        if retain_plan:
            self.factory.patch_agent_loop_policy = replace(self.factory.patch_agent_loop_policy,
                max_total_model_turns=previous_patch_calls + self.allocation,
                max_tool_rounds=old_policy.max_tool_rounds, max_observation_bytes=old_policy.max_observation_bytes)
            self.factory.bind_retained_worker_request(retained_request,
                consumed_model_turns=previous_patch_calls, consumed_tool_rounds=consumed_reads,
                consumed_observation_bytes=consumed_observation_bytes, planning_rationale=assessment["rationale"],
                context_calibration=retained_calibration)
            # Keep the same checked wire view and charged history while the
            # existing same-Plan owner retains the original logical request.
            self.factory._repair_input_projection = retained_input_projection
        else:
            self.factory.patch_agent_loop_policy = replace(self.factory.patch_agent_loop_policy,
                max_tool_rounds=remaining_reads, max_observation_bytes=remaining_bytes)
        observation_budget = {"maximum_reads": old_policy.max_tool_rounds,
            "consumed_reads": consumed_reads, "remaining_reads": remaining_reads,
            "maximum_bytes": old_policy.max_observation_bytes, "consumed_bytes": consumed_observation_bytes,
            "source_bytes": consumed_source_bytes,
            "feedback_bytes": consumed_observation_bytes - consumed_source_bytes,
            "remaining_bytes": remaining_bytes}
        if not retain_plan:
            upstream = previous.artifact_store.semantic_read_immutable(ArtifactRef(**repair_ref))["data"][
                "repair_context"]["upstream_request"]
            self.factory._upstream_continuation_context = {
                "failure_context_role": "diagnostic_history_not_current_plan_authority",
                "upstream_request": {key: deepcopy(upstream[key]) for key in
                    ("origin", "required_owner", "reason", "source_refs")},
                "prior_plan": _artifact_ref_dict(previous_request.implementation_plan_ref),
                "repair_plan": repair_ref,
                "previous_response": context["previous_response"],
                "failure_report": context["failure_report"],
                "failed_response_ref": context["failed_response_ref"],
                "failure_report_ref": context["failure_report_ref"],
                "approval_inherited": False}
        self._attach(self.factory)
        self.replan_count += 1
        # The assessment RepairPlan is history; its empty edit subset cannot govern Patch.
        for kind in ("patch_proposal", "proposal_review", "dry_run_result", "repair_plan"):
            next_state["artifacts"].pop(kind, None)
            next_state["artifact_hashes"].pop(kind, None)
        next_state["approvals"] = {}
        next_state["run_id"] = state["run_id"] + ("_retained_plan" if retain_plan else "_replanned")
        next_state["runtime_summary"].pop("patch_upstream_return", None)
        next_state["runtime_summary"]["replanned_patch_lineage"] = {
            "repair_plan": repair_ref, "prior_plan": _artifact_ref_dict(previous_request.implementation_plan_ref),
            "failed_response_ref": context["failed_response_ref"], "failure_report_ref": context["failure_report_ref"],
            "approval_inherited": False,
            **({"plan_retained": True, "planning_rationale": assessment["rationale"],
                "planning_trace_refs": assessment["trace_refs"]} if retain_plan else {})}
        self.record({"phase": "replanned_patch_prepared", "initial_selected_artifacts": self.initial_selection,
            "selected_artifacts": self.selection(self.request), "replanned_patch_lineage": next_state["runtime_summary"]["replanned_patch_lineage"],
            "effective_worker_model_allowance": self.allocation, "failure": None,
            "shared_observation_budget": observation_budget,
            "proposal_admitted": False})
        return next_state


def run_fresh_patch_continuation(
    request: PatchContinuationInputV1,
    *, patch_change_factory: PatchContinuationFactory | None,
    dry_run_factory: PatchContinuationDryRunFactory,
    max_model_calls: int = 1,
    upstream_configuration: Any = None,
    execution_trace_configuration: Any = None,
) -> PatchContinuationResultV1:
    """Run once, or read the same attempt's durable result without replaying tools.

    An interrupted attempt with no terminal result is stopped for recovery review:
    absence of a recorded response is not proof that the provider did not run.
    """
    saved_mode = request.reuses_model_output
    if saved_mode and (patch_change_factory is not None or max_model_calls != 0
                       or isinstance(max_model_calls, bool) or upstream_configuration is not None):
        return blocked_patch_continuation_result(
            code="PATCH_SAVED_EXECUTION_MODE_CONFLICT", stage="request", duration_seconds=0.0,
            reason="Saved Patch reassembly cannot receive a model factory, model allowance or upstream execution.",
            feature_id=request.feature_spec_ref.feature_id)
    run_id = (
        "patch_continuation_" + hashlib.sha256(request.attempt_id.encode("utf-8")).hexdigest()
        if request.attempt_id is not None else f"patch_continuation_{uuid4().hex}"
    )
    path = Path(request.artifact_store_root).resolve() / "cr" / (
        hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:32] + ".json"
    )
    if path.exists():
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("attempt_identity_sha256") != _patch_attempt_identity(request, max_model_calls):
                raise PatchContinuationError("PATCH_CONTINUATION_ATTEMPT_IDENTITY_CONFLICT", "resume_admission", "An attempt identity cannot be reused with different inputs or a reset budget.")
            preflight_patch_continuation(request)
            cached = document.get("terminal_result")
            if isinstance(cached, dict):
                if document.get("terminal_result_sha256") != _canonical_sha256(cached):
                    raise PatchContinuationError("PATCH_CONTINUATION_RESUME_RESULT_INVALID", "resume_admission", "The saved continuation result hash is invalid.")
                for name, kind in (("feature_spec_artifact", "feature_spec"),
                                   ("evidence_pack_artifact", "evidence_pack"),
                                   ("implementation_plan_artifact", "implementation_plan"),
                                   ("patch_proposal_artifact", "patch_proposal"),
                                   ("proposal_review_artifact", "proposal_review"),
                                   ("dry_run_artifact", "dry_run_result")):
                    if isinstance(cached.get(name), dict):
                        _read_selected_exact_revision(ArtifactStore(Path(request.artifact_store_root)), ArtifactRef(**cached[name]), consumer_stage=None)
                values = {field.name: cached.get(field.name) for field in fields(PatchContinuationResultV1)}
                values["attempt_receipt"] = _attempt_receipt_projection(path, document)
                return PatchContinuationResultV1(**values)
            return blocked_patch_continuation_result(
                code="PATCH_CONTINUATION_ATTEMPT_RECOVERY_REQUIRED", stage="resume_admission",
                duration_seconds=0.0,
                reason="This attempt is already claimed and has no durable terminal result; it will not send another model request or reset its budget.",
                run_id=run_id, request_id=str(document.get("request_id") or ""),
                feature_id=request.feature_spec_ref.feature_id,
                attempt_receipt=_attempt_receipt_projection(path, document),
                execution_counts=_failed_patch_execution_counts(None, model_call_count=int(document.get("model_call_count") or 0)),
            )
        except (PatchContinuationError, ValueError, OSError) as exc:
            return blocked_patch_continuation_result(
                code=getattr(exc, "code", "PATCH_CONTINUATION_RESUME_RECEIPT_INVALID"),
                duration_seconds=0.0,
                stage="resume_admission", reason=str(getattr(exc, "reason", exc)),
                run_id=run_id, request_id="", feature_id=request.feature_spec_ref.feature_id,
            )
    try:
        result = _run_fresh_patch_continuation_once(request,
            patch_change_factory=patch_change_factory, dry_run_factory=dry_run_factory,
            max_model_calls=max_model_calls, run_id=run_id, upstream_configuration=upstream_configuration,
            execution_trace_configuration=execution_trace_configuration)
    except PatchContinuationError as exc:
        return blocked_patch_continuation_result(code=exc.code, stage=exc.stage,
            duration_seconds=0.0,
            reason=exc.reason, run_id=run_id, request_id="", feature_id=request.feature_spec_ref.feature_id)
    if result.attempt_receipt is not None:
        document = json.loads(path.read_text(encoding="utf-8"))
        cached = result.to_dict()
        cached.pop("attempt_receipt", None)
        document["terminal_result"] = cached
        document["terminal_result_sha256"] = _canonical_sha256(cached)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        result = replace(result, attempt_receipt=_attempt_receipt_projection(path, document))
    return result


def _run_fresh_patch_continuation_once(
    request: PatchContinuationInputV1,
    *,
    patch_change_factory: PatchContinuationFactory | None,
    dry_run_factory: PatchContinuationDryRunFactory,
    max_model_calls: int = 1,
    run_id: str,
    upstream_configuration: Any = None,
    execution_trace_configuration: Any = None,
) -> PatchContinuationResultV1:
    """Run an exact Plan through Patch, Review, and no-write Dry-run once."""

    started = time.perf_counter()
    request_id = f"patch_request_{uuid4().hex}"
    feature_id = request.feature_spec_ref.feature_id
    reported_refs = {kind: getattr(request, kind + "_ref")
                     for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
    if (
        isinstance(max_model_calls, bool)
        or not isinstance(max_model_calls, int)
        or (max_model_calls != 0 if request.reuses_model_output
            else not 1 <= max_model_calls <= 5)
    ):
        return blocked_patch_continuation_result(
            code="PATCH_CONTINUATION_MODEL_CALL_BUDGET_INVALID",
            stage="request",
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=reported_refs["feature_spec"],
            evidence_pack_ref=reported_refs["evidence_pack"],
            implementation_plan_ref=reported_refs["implementation_plan"],
        )
    try:
        admitted = preflight_patch_continuation(request)
    except PatchContinuationError as exc:
        return blocked_patch_continuation_result(
            code=exc.code,
            stage=exc.stage,
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=reported_refs["feature_spec"],
            evidence_pack_ref=reported_refs["evidence_pack"],
            implementation_plan_ref=reported_refs["implementation_plan"],
        )

    if admitted.saved_patch is not None:
        request_id = admitted.saved_patch["request_id"]
    if admitted.saved_candidate is not None:
        request_id = admitted.saved_candidate["request_id"]
    journal = PatchContinuationAttemptJournal(
        artifact_root=admitted.artifact_store.root,
        run_id=run_id,
        request_id=request_id,
        request=request,
        admitted=admitted,
        max_model_calls=max_model_calls,
    )
    from src.mod_agent.patch_execution_trace import PatchExecutionTraceConfiguration

    configuration = execution_trace_configuration or PatchExecutionTraceConfiguration()
    trace = configuration.make_trace(run_id=run_id, feature_id=feature_id,
        artifact_store_root=request.artifact_store_root, recorder=journal.record_observation,
        **({"request_id": admitted.saved_patch["request_id"]} if admitted.saved_patch is not None else
           {"request_id": admitted.saved_candidate["request_id"]} if admitted.saved_candidate is not None else {}))
    try:
        with trace.activate():
            trace.update(artifact_refs=[{"artifact_type": kind, "artifact_hash": ref.artifact_hash}
                for kind, ref in reported_refs.items()])
            result = _run_admitted_patch_continuation(request,
                patch_change_factory=patch_change_factory, dry_run_factory=dry_run_factory,
                max_model_calls=max_model_calls, run_id=run_id,
                upstream_configuration=upstream_configuration, started=started,
                request_id=request_id, admitted=admitted, journal=journal)
            trace.update(workflow_status=result.status, terminal_stage=result.terminal_stage,
                failure_code=(result.first_failure or {}).get("code"),
                proposal_admitted=(result.attempt_receipt or {}).get("proposal_admitted", False))
    finally:
        summary = configuration.finish(trace)
        summary.update(trace_id=trace.context.trace_id, session_id=trace.session_id,
                       environment=trace.environment)
        try:
            journal.record_trace_summary(summary)
        except Exception:
            # Telemetry write failure is not a product state transition.
            pass
    return replace(result, attempt_receipt=journal.projection())


def _run_admitted_patch_continuation(request, *, patch_change_factory, dry_run_factory,
        max_model_calls, run_id, upstream_configuration, started, request_id, admitted, journal):
    feature_id = request.feature_spec_ref.feature_id
    reported_refs = {kind: getattr(request, kind + "_ref")
                     for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
    upstream_execution = None
    try:
        if upstream_configuration is not None:
            upstream_execution = _PatchUpstreamExecution(request, patch_change_factory,
                upstream_configuration, journal, max_model_calls)
        elif request.patch_upstream_configuration_ref is not None:
            raise PatchContinuationError("PATCH_UPSTREAM_CONFIGURATION_INVALID", "request", "Configured upstream providers were not supplied.")
        else:
            _bind_patch_attempt_recorder(patch_change_factory, journal.record)
    except PatchContinuationError as exc:
        return blocked_patch_continuation_result(
            code=exc.code,
            stage=exc.stage,
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=reported_refs["feature_spec"],
            evidence_pack_ref=reported_refs["evidence_pack"],
            implementation_plan_ref=reported_refs["implementation_plan"],
            source_validation={"status": "passed_before_patch", "before": admitted.source_validation},
            attempt_receipt=journal.projection(),
            source_modified_during_run=False,
        )

    from src.mod_agent.workflow.graph import build_graph

    state = _patch_continuation_state(
        run_id=run_id,
        request_id=request_id,
        request=request,
        admitted=admitted,
    )
    try:
        graph = build_graph(
            admitted.artifact_store,
            patch_change_factory=patch_change_factory,
            dry_run_factory=dry_run_factory,
            use_default_proposal_git_binding=False,
            proposal_source_snippet_char_limit=_proposal_source_snippet_char_limit(
                admitted
            ),
            patch_continuation_mode=request,
            patch_continuation_max_model_calls=max_model_calls,
            patch_continuation_attempt_recorder=journal.record,
            **({"enable_patch_upstream_replanning": True,
                "patch_replanning_context": upstream_execution,
                **upstream_execution.providers} if upstream_execution is not None else {}),
        )
        graph_result = graph.invoke(state)
        reported_refs = {kind: _ref_from_patch_state(graph_result, kind, feature_id=feature_id) or ref
                         for kind, ref in reported_refs.items()}
        if upstream_execution is not None:
            journal.record({"phase": "current_upstream_artifacts_observed",
                "current_upstream_artifacts": {kind: _artifact_ref_dict(ref) for kind, ref in reported_refs.items()}})
    except Exception as exc:
        failed_node = getattr(exc, "_mod_agent_graph_node", None)
        failure_stage = str(
            getattr(exc, "stage", None) or failed_node or "workflow"
        )
        current_snapshot = _source_tree_snapshot_sha256(
            _patch_request_source_snapshot(request)
        )
        failure_code = str(getattr(exc, "code", type(exc).__name__))
        failure_message = str(exc)[:4096] or failure_code
        journal.record(
            {
                "phase": "continuation_failed",
                "current_snapshot_sha256": current_snapshot,
                "failure": {
                    "code": failure_code,
                    "failure_class": type(exc).__name__,
                    "message": failure_message,
                    "stage": failure_stage,
                },
                "proposal_admitted": False,
            }
        )
        receipt = journal.projection()
        source_modified = (
            current_snapshot
            != _source_tree_snapshot_sha256(admitted.source_tree_snapshot)
        )
        counts = _failed_patch_execution_counts(
            failed_node if isinstance(failed_node, str) else None,
            model_call_count=int(receipt.get("model_call_count") or 0),
        )
        failure_state = getattr(exc, "_mod_agent_graph_state", None)
        if upstream_execution is not None and isinstance(failure_state, Mapping):
            reported_refs = {kind: _ref_from_patch_state(failure_state, kind, feature_id=feature_id) or ref
                             for kind, ref in reported_refs.items()}
            counts = _patch_execution_counts(failure_state, model_call_count=int(receipt.get("model_call_count") or 0))
            journal.record({"phase": "upstream_execution_failed", "execution_counts": counts,
                "trace_events": failure_state.get("trace_events", []), "proposal_admitted": False,
                "current_upstream_artifacts": {kind: _artifact_ref_dict(ref) for kind, ref in reported_refs.items()}})
            receipt = journal.projection()
        patch_ref = journal.artifact_ref("patch_proposal")
        review_ref = journal.artifact_ref("proposal_review")
        return blocked_patch_continuation_result(
            code=failure_code,
            stage=failure_stage,
            reason=failure_message,
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=reported_refs["feature_spec"],
            evidence_pack_ref=reported_refs["evidence_pack"],
            implementation_plan_ref=reported_refs["implementation_plan"],
            patch_proposal_ref=patch_ref,
            proposal_review_ref=review_ref,
            source_validation={
                "status": "failed_during_continuation",
                "before": admitted.source_validation,
                "current_snapshot_sha256": current_snapshot,
            },
            execution_counts=counts,
            attempt_receipt=receipt,
            source_modified_during_run=source_modified,
            status="partial" if counts["model"] else "blocked",
        )

    if upstream_execution is not None:
        request = upstream_execution.request
    patch_ref = _ref_from_patch_state(
        graph_result,
        "patch_proposal",
        feature_id=feature_id,
    )
    review_ref = _ref_from_patch_state(
        graph_result,
        "proposal_review",
        feature_id=feature_id,
    )
    dry_run_ref = _ref_from_patch_state(
        graph_result,
        "dry_run_result",
        feature_id=feature_id,
    )
    if request.repair_parent_proposal_ref is not None:
        if patch_ref == request.repair_parent_proposal_ref:
            patch_ref = None
        if review_ref == request.repair_parent_review_ref:
            review_ref = None
    artifacts = {
        artifact_type: _artifact_ref_dict(ref)
        for artifact_type, ref in (
            ("patch_proposal", patch_ref),
            ("proposal_review", review_ref),
            ("dry_run_result", dry_run_ref),
        )
        if ref is not None
    }
    journal.record(
        {
            "phase": "workflow_artifacts_persisted",
            "artifacts": artifacts,
            "proposal_admitted": patch_ref is not None and not graph_result.get("blockers"),
        }
    )
    try:
        after = preflight_patch_continuation(
            request,
            artifact_store=admitted.artifact_store,
        )
    except PatchContinuationError as exc:
        current_snapshot = _source_tree_snapshot_sha256(
            _patch_request_source_snapshot(request)
        )
        journal.record(
            {
                "phase": "post_workflow_admission_rejected",
                "current_snapshot_sha256": current_snapshot,
                "failure": {
                    "code": exc.code,
                    "failure_class": type(exc).__name__,
                    "stage": exc.stage,
                },
                "proposal_admitted": False,
            }
        )
        counts = _patch_execution_counts(
            graph_result,
            model_call_count=int(journal.projection().get("model_call_count") or 0),
        )
        return blocked_patch_continuation_result(
            code=exc.code,
            stage=exc.stage,
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=reported_refs["feature_spec"],
            evidence_pack_ref=reported_refs["evidence_pack"],
            implementation_plan_ref=reported_refs["implementation_plan"],
            patch_proposal_ref=patch_ref,
            proposal_review_ref=review_ref,
            dry_run_ref=dry_run_ref,
            source_validation={"status": "failed_after_workflow", "before": admitted.source_validation},
            execution_counts=counts,
            attempt_receipt=journal.projection(),
            source_modified_during_run=None,
            status="partial",
        )
    source_modified = after.source_tree_snapshot != admitted.source_tree_snapshot
    if source_modified:
        code = "PATCH_CONTINUATION_SOURCE_CHANGED_DURING_WORKFLOW"
        journal.record(
            {
                "phase": "post_workflow_source_rejected",
                "current_snapshot_sha256": _source_tree_snapshot_sha256(
                    after.source_tree_snapshot
                ),
                "failure": {
                    "code": code,
                    "failure_class": "PatchContinuationError",
                    "stage": "source_validation",
                },
                "proposal_admitted": False,
            }
        )
        return blocked_patch_continuation_result(
            code=code,
            stage="source_validation",
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=reported_refs["feature_spec"],
            evidence_pack_ref=reported_refs["evidence_pack"],
            implementation_plan_ref=reported_refs["implementation_plan"],
            patch_proposal_ref=patch_ref,
            proposal_review_ref=review_ref,
            dry_run_ref=dry_run_ref,
            source_validation={
                "status": "failed_after_workflow",
                "before": admitted.source_validation,
                "after": after.source_validation,
            },
            execution_counts=_patch_execution_counts(
                graph_result,
                model_call_count=int(journal.projection().get("model_call_count") or 0),
            ),
            attempt_receipt=journal.projection(),
            source_modified_during_run=True,
            status="partial",
        )

    first_failure = _patch_first_failure(graph_result)
    pre_terminal_receipt = journal.projection()
    counts = _patch_execution_counts(
        graph_result,
        model_call_count=int(pre_terminal_receipt.get("model_call_count") or 0),
    )
    expected_counts = {
        "fresh_coverage": 0,
        "model": counts["model"],
        "planning": upstream_execution.replan_count if upstream_execution is not None else 0,
        "patch": 1 + (upstream_execution.replan_count if upstream_execution is not None else 0),
        "review": 1,
        "dry_run": 1,
        "apply": 0,
        "build": 0,
        "runtime": 0,
    }
    complete = (
        first_failure is None
        and patch_ref is not None
        and review_ref is not None
        and dry_run_ref is not None
        and graph_result.get("current_stage") == "dry_run_apply"
        and counts == expected_counts
    )
    terminal_phase = (
        "stopped_before_exact_diff_approval"
        if complete
        else "continuation_blocked"
    )
    journal.record(
        {
            "phase": terminal_phase,
            "current_snapshot_sha256": _source_tree_snapshot_sha256(
                after.source_tree_snapshot
            ),
            "proposal_admitted": complete,
            "source_tree_unchanged": True,
        }
    )
    receipt = journal.projection()
    if not complete:
        failure = first_failure or {
            "code": "PATCH_CONTINUATION_INCOMPLETE",
            "stage": str(graph_result.get("current_stage") or "workflow"),
        }
        return blocked_patch_continuation_result(
            code=failure["code"],
            stage=failure["stage"],
            duration_seconds=time.perf_counter() - started,
            run_id=run_id,
            request_id=request_id,
            feature_id=feature_id,
            feature_spec_ref=reported_refs["feature_spec"],
            evidence_pack_ref=reported_refs["evidence_pack"],
            implementation_plan_ref=reported_refs["implementation_plan"],
            patch_proposal_ref=patch_ref,
            proposal_review_ref=review_ref,
            dry_run_ref=dry_run_ref,
            source_validation={
                "status": "passed",
                "before": admitted.source_validation,
                "after_snapshot_sha256": after.source_validation["snapshot_sha256"],
                "source_tree_unchanged": True,
            },
            execution_counts=counts,
            attempt_receipt=receipt,
            source_modified_during_run=False,
            status="partial" if counts["model"] else "blocked",
        )
    return PatchContinuationResultV1(
        status="terminal",
        terminal_stage="dry_run_apply",
        first_failure=None,
        run_id=run_id,
        request_id=request_id,
        feature_id=feature_id,
        feature_spec_artifact=_artifact_ref_dict(reported_refs["feature_spec"]),
        evidence_pack_artifact=_artifact_ref_dict(reported_refs["evidence_pack"]),
        implementation_plan_artifact=_artifact_ref_dict(reported_refs["implementation_plan"]),
        patch_proposal_artifact=_artifact_ref_dict(patch_ref),
        proposal_review_artifact=_artifact_ref_dict(review_ref),
        dry_run_artifact=_artifact_ref_dict(dry_run_ref),
        source_validation={
            "status": "passed",
            "before": admitted.source_validation,
            "after_snapshot_sha256": after.source_validation["snapshot_sha256"],
            "source_tree_unchanged": True,
        },
        execution_counts=counts,
        attempt_receipt=receipt,
        source_modified_during_run=False,
        duration_seconds=time.perf_counter() - started,
    )
