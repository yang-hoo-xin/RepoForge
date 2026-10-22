"""Filesystem-backed artifact store with Phase B artifact contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

from src.mod_agent.artifact_semantics import (
    ArtifactWriteIntent,
    SEMANTIC_CONFLICT_HASH_DOMAIN,
    project_artifact_write_intent,
    semantic_binding_from_payload,
)
from src.mod_agent.action_identity import hash_action_identity

from .envelope import ArtifactEnvelope, ArtifactRef
from .freshness import FreshnessView, StaleArtifactError
from .registry import ArtifactContract, ArtifactRegistryError, ArtifactTypeRegistry
from ..edit_authority_capabilities import (
    InvalidEditAuthorityApprovalIdentityError,
    bind_edit_authority_exact_input_upstreams,
    validate_edit_authority_exact_input_admission,
    validate_edit_authority_request_identity,
    validate_edit_authority_resume_report_against_approval,
)

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ARTIFACT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENVELOPE_HASH_FIELDS = (
    "schema_version",
    "artifact_type",
    "feature_id",
    "producer_stage",
    "status",
    "data",
    "inputs",
    "blockers",
    "trace_refs",
)
_ENVELOPE_FIELDS = frozenset((*_ENVELOPE_HASH_FIELDS, "artifact_hash"))
_PENDING_WRITE_FIELDS = frozenset(
    (
        "schema_version",
        "feature_id",
        "artifact_type",
        "artifact_hash",
        "revision_path",
        "latest_path",
        "status",
    )
)
_PENDING_WRITE_STATUS = "pending_latest_replace"
_SEMANTIC_PENDING_WRITE_FIELDS = frozenset(
    (
        "schema_version",
        "status",
        "feature_id",
        "artifact_type",
        "artifact_hash",
        "revision_path",
        "latest_path",
        "semantic_binding_key",
        "semantic_binding_version",
        "commit_conflict_key",
        "commit_conflict_key_hash",
    )
)
_SEMANTIC_PENDING_WRITE_STATUS = "pending_semantic_commit"
_PHYSICAL_LOCATOR_HEX_LENGTH = 32
_REVISION_DIRECTORY = "r"
_PENDING_DIRECTORY = "p"
_PREFLIGHT_CONTENT = '{"artifact_store_preflight":true}\n'


class ArtifactStoreError(ValueError):
    """Raised when artifact storage validation fails."""


class ArtifactStorePreflightError(ArtifactStoreError):
    """Raised for a recognized atomic-replace path failure during preflight."""

    def __init__(
        self,
        *,
        artifact_type: str,
        path_role: str,
        source_path_length: int,
        target_path_length: int,
        target_parent_exists: bool,
        winerror: int,
    ) -> None:
        super().__init__(
            "ArtifactStore preflight could not address the "
            f"{path_role} path for {artifact_type}."
        )
        self.artifact_type = artifact_type
        self.path_role = path_role
        self.source_path_length = source_path_length
        self.target_path_length = target_path_length
        self.target_parent_exists = target_parent_exists
        self.winerror = winerror


ArtifactWriteOutcome = Literal["committed", "blocked_before_commit", "partial"]
ArtifactRevisionState = Literal[
    "unavailable",
    "absent",
    "valid",
    "invalid",
    "unknown",
]
ArtifactLatestAliasState = Literal[
    "unavailable",
    "absent",
    "matches_identity",
    "different_identity",
    "invalid",
    "unknown",
]
ArtifactPendingMarkerState = Literal[
    "unavailable",
    "absent",
    "present",
    "invalid",
    "unknown",
]
ArtifactWriteFailureStage = Literal[
    "admission_validation",
    "pending_marker_write",
    "immutable_revision_write",
    "latest_alias_write",
    "commit_verification",
    "pending_marker_cleanup",
]


@dataclass(frozen=True)
class ArtifactWriteResult:
    """Authoritative API result for one ArtifactStore write attempt."""

    outcome: ArtifactWriteOutcome
    feature_id: str
    artifact_type: str
    artifact_hash: str | None
    artifact_ref: ArtifactRef | None
    revision_path: str | None
    latest_alias_path: str | None
    pending_marker_path: str | None
    commit_started: bool
    revision_state: ArtifactRevisionState
    latest_alias_state: ArtifactLatestAliasState
    pending_marker_state: ArtifactPendingMarkerState
    failure_stage: ArtifactWriteFailureStage | None
    failure_class: str | None
    failure_message: str | None
    recovery_available: bool
    recovery_authorized: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactWriteResult":
        """Parse the exact public projection without weakening Store semantics."""

        expected_fields = {
            "outcome",
            "feature_id",
            "artifact_type",
            "artifact_hash",
            "artifact_ref",
            "revision_path",
            "latest_alias_path",
            "pending_marker_path",
            "commit_started",
            "revision_state",
            "latest_alias_state",
            "pending_marker_state",
            "failure_stage",
            "failure_class",
            "failure_message",
            "recovery_available",
            "recovery_authorized",
        }
        if set(value) != expected_fields:
            raise ArtifactStoreError("ArtifactWriteResult fields changed.")

        def require_string(field: str) -> str:
            item = value[field]
            if type(item) is not str or not item:
                raise ArtifactStoreError(
                    f"ArtifactWriteResult {field} must be a non-empty string."
                )
            return item

        def optional_string(field: str) -> str | None:
            item = value[field]
            if item is not None and (type(item) is not str or not item):
                raise ArtifactStoreError(
                    f"ArtifactWriteResult {field} must be a non-empty string or null."
                )
            return item

        outcome = value["outcome"]
        if outcome not in {"committed", "blocked_before_commit", "partial"}:
            raise ArtifactStoreError("ArtifactWriteResult outcome is not registered.")
        revision_state = value["revision_state"]
        if revision_state not in {
            "unavailable",
            "absent",
            "valid",
            "invalid",
            "unknown",
        }:
            raise ArtifactStoreError(
                "ArtifactWriteResult revision_state is not registered."
            )
        latest_alias_state = value["latest_alias_state"]
        if latest_alias_state not in {
            "unavailable",
            "absent",
            "matches_identity",
            "different_identity",
            "invalid",
            "unknown",
        }:
            raise ArtifactStoreError(
                "ArtifactWriteResult latest_alias_state is not registered."
            )
        pending_marker_state = value["pending_marker_state"]
        if pending_marker_state not in {
            "unavailable",
            "absent",
            "present",
            "invalid",
            "unknown",
        }:
            raise ArtifactStoreError(
                "ArtifactWriteResult pending_marker_state is not registered."
            )
        failure_stage = value["failure_stage"]
        if failure_stage not in {
            None,
            "admission_validation",
            "pending_marker_write",
            "immutable_revision_write",
            "latest_alias_write",
            "commit_verification",
            "pending_marker_cleanup",
        }:
            raise ArtifactStoreError(
                "ArtifactWriteResult failure_stage is not registered."
            )
        for field in ("commit_started", "recovery_available", "recovery_authorized"):
            if type(value[field]) is not bool:
                raise ArtifactStoreError(f"ArtifactWriteResult {field} must be bool.")

        artifact_ref_value = value["artifact_ref"]
        artifact_ref: ArtifactRef | None = None
        if artifact_ref_value is not None:
            if not isinstance(artifact_ref_value, Mapping) or set(artifact_ref_value) != {
                "feature_id",
                "artifact_type",
                "path",
                "artifact_hash",
            }:
                raise ArtifactStoreError(
                    "ArtifactWriteResult artifact_ref fields changed."
                )
            ref_values = {
                field: artifact_ref_value[field]
                for field in ("feature_id", "artifact_type", "path", "artifact_hash")
            }
            if any(type(item) is not str or not item for item in ref_values.values()):
                raise ArtifactStoreError("ArtifactWriteResult artifact_ref is invalid.")
            artifact_ref = ArtifactRef(**ref_values)

        result = cls(
            outcome=outcome,
            feature_id=require_string("feature_id"),
            artifact_type=require_string("artifact_type"),
            artifact_hash=optional_string("artifact_hash"),
            artifact_ref=artifact_ref,
            revision_path=optional_string("revision_path"),
            latest_alias_path=optional_string("latest_alias_path"),
            pending_marker_path=optional_string("pending_marker_path"),
            commit_started=value["commit_started"],
            revision_state=revision_state,
            latest_alias_state=latest_alias_state,
            pending_marker_state=pending_marker_state,
            failure_stage=failure_stage,
            failure_class=optional_string("failure_class"),
            failure_message=optional_string("failure_message"),
            recovery_available=value["recovery_available"],
            recovery_authorized=value["recovery_authorized"],
        )
        if artifact_ref is not None and (
            artifact_ref.feature_id != result.feature_id
            or artifact_ref.artifact_type != result.artifact_type
            or artifact_ref.artifact_hash != result.artifact_hash
            or artifact_ref.path != result.revision_path
        ):
            raise ArtifactStoreError(
                "ArtifactWriteResult artifact_ref disagrees with its Store identity."
            )
        if result.outcome == "committed":
            if (
                result.artifact_ref is None
                or result.artifact_hash is None
                or result.revision_state != "valid"
                or result.latest_alias_state != "matches_identity"
                or result.pending_marker_state != "absent"
                or result.failure_stage is not None
                or result.failure_class is not None
                or result.failure_message is not None
                or result.recovery_available
                or result.recovery_authorized
            ):
                raise ArtifactStoreError(
                    "Committed ArtifactWriteResult has inconsistent Store facts."
                )
        else:
            if (
                result.artifact_ref is not None
                or result.failure_stage is None
                or result.failure_class is None
                or result.failure_message is None
                or result.recovery_authorized
            ):
                raise ArtifactStoreError(
                    "Non-committed ArtifactWriteResult has inconsistent Store facts."
                )
            if result.outcome == "blocked_before_commit" and (
                result.commit_started or result.recovery_available
            ):
                raise ArtifactStoreError(
                    "Blocked-before-commit ArtifactWriteResult entered commit or recovery."
                )
            if result.recovery_available and (
                result.outcome != "partial"
                or result.pending_marker_state != "present"
                or result.revision_state != "valid"
            ):
                raise ArtifactStoreError(
                    "Recoverable ArtifactWriteResult lacks partial pending revision facts."
                )
        if result.to_dict() != dict(value):
            raise ArtifactStoreError("ArtifactWriteResult projection changed during parsing.")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe projection without changing Artifact identity."""

        artifact_ref = None
        if self.artifact_ref is not None:
            artifact_ref = {
                "feature_id": self.artifact_ref.feature_id,
                "artifact_type": self.artifact_ref.artifact_type,
                "path": self.artifact_ref.path,
                "artifact_hash": self.artifact_ref.artifact_hash,
            }
        return {
            "outcome": self.outcome,
            "feature_id": self.feature_id,
            "artifact_type": self.artifact_type,
            "artifact_hash": self.artifact_hash,
            "artifact_ref": artifact_ref,
            "revision_path": self.revision_path,
            "latest_alias_path": self.latest_alias_path,
            "pending_marker_path": self.pending_marker_path,
            "commit_started": self.commit_started,
            "revision_state": self.revision_state,
            "latest_alias_state": self.latest_alias_state,
            "pending_marker_state": self.pending_marker_state,
            "failure_stage": self.failure_stage,
            "failure_class": self.failure_class,
            "failure_message": self.failure_message,
            "recovery_available": self.recovery_available,
            "recovery_authorized": self.recovery_authorized,
        }


@dataclass(frozen=True)
class _ArtifactWriteAttempt:
    result: ArtifactWriteResult
    error: Exception | None


class _AtomicReplaceFailure(RuntimeError):
    """Carry exact source/target evidence from a preflight replace failure."""

    def __init__(self, source_path: Path, target_path: Path, cause: OSError) -> None:
        super().__init__(str(cause))
        self.source_path = source_path
        self.target_path = target_path
        self.cause = cause


def _requires_conditional_upstream_contract(
    *,
    artifact_type: str,
    payload: Mapping[str, Any],
    status: str,
) -> bool:
    if artifact_type != "repair_plan":
        return False
    return (
        status != "blocked"
        and not payload.get("blockers")
        and payload.get("allowed_next_stage") is not None
    )


class ArtifactStore:
    """Write and read feature-scoped artifacts through one guarded interface."""

    def __init__(
        self,
        root: str | Path = "artifacts",
        registry: ArtifactTypeRegistry | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.registry = registry or ArtifactTypeRegistry()

    def write(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        payload: Mapping[str, Any],
        status: str = "success",
        blockers: list[dict[str, Any]] | None = None,
        trace_refs: list[str] | None = None,
        upstream_refs: list[ArtifactRef] | None = None,
        allow_stale_upstreams: bool = False,
        exact_revision_upstream_types: set[str] | None = None,
        semantic_intent: ArtifactWriteIntent | None = None,
    ) -> ArtifactRef:
        """Write an Artifact and preserve the historical return/error contract."""

        attempt = self._execute_write(
            feature_id=feature_id,
            artifact_type=artifact_type,
            payload=payload,
            status=status,
            blockers=blockers,
            trace_refs=trace_refs,
            upstream_refs=upstream_refs,
            allow_stale_upstreams=allow_stale_upstreams,
            exact_revision_upstream_types=exact_revision_upstream_types,
            semantic_intent=semantic_intent,
        )
        if attempt.result.outcome == "committed":
            if attempt.result.artifact_ref is None:
                raise ArtifactStoreError(
                    "Committed Artifact write result is missing artifact_ref"
                )
            return attempt.result.artifact_ref
        if attempt.error is not None:
            raise attempt.error
        raise ArtifactStoreError(
            "Artifact write did not commit: "
            f"{attempt.result.outcome} at {attempt.result.failure_stage or 'unknown'}"
        )

    def write_with_result(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        payload: Mapping[str, Any],
        status: str = "success",
        blockers: list[dict[str, Any]] | None = None,
        trace_refs: list[str] | None = None,
        upstream_refs: list[ArtifactRef] | None = None,
        allow_stale_upstreams: bool = False,
        exact_revision_upstream_types: set[str] | None = None,
        semantic_intent: ArtifactWriteIntent | None = None,
    ) -> ArtifactWriteResult:
        """Return the Store-owned outcome without retrying or recovering."""

        return self._execute_write(
            feature_id=feature_id,
            artifact_type=artifact_type,
            payload=payload,
            status=status,
            blockers=blockers,
            trace_refs=trace_refs,
            upstream_refs=upstream_refs,
            allow_stale_upstreams=allow_stale_upstreams,
            exact_revision_upstream_types=exact_revision_upstream_types,
            semantic_intent=semantic_intent,
        ).result

    def _execute_write(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        payload: Mapping[str, Any],
        status: str,
        blockers: list[dict[str, Any]] | None,
        trace_refs: list[str] | None,
        upstream_refs: list[ArtifactRef] | None,
        allow_stale_upstreams: bool,
        exact_revision_upstream_types: set[str] | None,
        semantic_intent: ArtifactWriteIntent | None,
    ) -> _ArtifactWriteAttempt:
        failure_stage: ArtifactWriteFailureStage = "admission_validation"
        artifact_hash: str | None = None
        revision_path: Path | None = None
        latest_path: Path | None = None
        pending_path: Path | None = None
        commit_started = False
        pending_write_attempted = False
        validated_semantic_intent: ArtifactWriteIntent | None = None
        try:
            self._validate_feature_id(feature_id)
            self._validate_artifact_type(artifact_type)
            data = dict(payload)
            try:
                contract = self.registry.validate_write(
                    artifact_type=artifact_type,
                    status=status,
                    payload=data,
                )
            except ArtifactRegistryError as exc:
                raise ArtifactStoreError(str(exc)) from exc
            resolved_upstream_refs = list(upstream_refs or [])
            exact_upstream_types = set(exact_revision_upstream_types or set())
            self._validate_upstream_refs(
                feature_id=feature_id,
                artifact_type=artifact_type,
                contract=contract,
                payload=data,
                status=status,
                upstream_refs=resolved_upstream_refs,
                allow_stale_upstreams=allow_stale_upstreams,
                exact_revision_upstream_types=exact_upstream_types,
            )
            try:
                semantic_binding = semantic_binding_from_payload(
                    artifact_type,
                    data,
                )
                if semantic_binding is None:
                    if semantic_intent is not None:
                        raise ValueError(
                            "Artifact write intent was supplied without a semantic binding."
                        )
                else:
                    if semantic_intent is None:
                        raise ValueError(
                            "Registered semantic Artifact writes require an exact intent."
                        )
                    expected_intent = project_artifact_write_intent(
                        reader=self,
                        feature_id=feature_id,
                        artifact_type=artifact_type,
                        payload=data,
                        status=status,
                        upstream_refs=resolved_upstream_refs,
                        exact_revision_upstream_types=exact_upstream_types,
                        blockers=tuple(blockers or ()),
                        trace_refs=tuple(trace_refs or ()),
                        protocol_refs=semantic_intent.protocol_refs,
                        execution_file_bindings=(
                            semantic_intent.execution_file_bindings
                        ),
                    )
                    if semantic_intent != expected_intent:
                        raise ValueError(
                            "Artifact semantic write intent changed after projection."
                        )
                    validated_semantic_intent = expected_intent
            except ValueError as exc:
                raise ArtifactStoreError(str(exc)) from exc
            inputs = {
                "upstream_artifacts": [
                    {
                        "artifact_type": upstream.artifact_type,
                        "path": upstream.path,
                        "artifact_hash": upstream.artifact_hash,
                        "require_latest": (
                            upstream.artifact_type not in exact_upstream_types
                        ),
                    }
                    for upstream in resolved_upstream_refs
                ]
            }
            envelope_content = {
                "schema_version": contract.schema_version,
                "artifact_type": artifact_type,
                "feature_id": feature_id,
                "producer_stage": contract.producer_stage,
                "status": status,
                "data": data,
                "inputs": inputs,
                "blockers": blockers or [],
                "trace_refs": trace_refs or [],
            }
            artifact_hash = self._stable_hash(
                self._envelope_content_for_hash(envelope_content)
            )
            if (
                validated_semantic_intent is not None
                and artifact_hash
                != validated_semantic_intent.expected_artifact_hash
            ):
                raise ArtifactStoreError(
                    "Artifact semantic intent content identity changed."
                )
            envelope = ArtifactEnvelope(
                schema_version=contract.schema_version,
                artifact_type=artifact_type,
                feature_id=feature_id,
                producer_stage=contract.producer_stage,
                status=status,
                data=data,
                artifact_hash=artifact_hash,
                inputs=inputs,
                blockers=blockers or [],
                trace_refs=trace_refs or [],
            )
            serialized_envelope = (
                json.dumps(
                    envelope.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            latest_path = self._artifact_path(feature_id, artifact_type)
            if validated_semantic_intent is None:
                pending_path = self._pending_write_path(
                    feature_id,
                    artifact_type,
                    artifact_hash,
                )
                existing_pending = self._validate_pending_locator(
                    pending_path,
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    artifact_hash=artifact_hash,
                )
            else:
                pending_path = self._semantic_pending_write_path(
                    feature_id,
                    artifact_type,
                    validated_semantic_intent.commit_conflict_key_hash,
                )
                existing_pending = self._validate_semantic_pending_locator(
                    pending_path,
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    semantic_binding_key=(
                        validated_semantic_intent.semantic_binding_key
                    ),
                    semantic_binding_version=(
                        validated_semantic_intent.semantic_binding_version
                    ),
                    commit_conflict_key=(
                        validated_semantic_intent.commit_conflict_key
                    ),
                    commit_conflict_key_hash=(
                        validated_semantic_intent.commit_conflict_key_hash
                    ),
                )
            if existing_pending is not None:
                revision_path = self._revision_artifact_path(
                    feature_id,
                    artifact_type,
                    artifact_hash,
                )
                error = ArtifactStoreError(
                    "Existing pending write requires explicit recovery"
                )
                return _ArtifactWriteAttempt(
                    result=self._artifact_write_result(
                        outcome="partial",
                        feature_id=feature_id,
                        artifact_type=artifact_type,
                        artifact_hash=artifact_hash,
                        revision_path=revision_path,
                        latest_path=latest_path,
                        pending_path=pending_path,
                        commit_started=False,
                        failure_stage="admission_validation",
                        error=error,
                        semantic_intent=validated_semantic_intent,
                    ),
                    error=error,
                )

            existing_revision_path = self._find_existing_revision_path(
                feature_id,
                artifact_type,
                artifact_hash,
            )
            revision_path = existing_revision_path or self._revision_artifact_path(
                feature_id,
                artifact_type,
                artifact_hash,
            )
            initial_state = self._artifact_write_result(
                outcome="blocked_before_commit",
                feature_id=feature_id,
                artifact_type=artifact_type,
                artifact_hash=artifact_hash,
                revision_path=revision_path,
                latest_path=latest_path,
                pending_path=pending_path,
                commit_started=False,
                failure_stage=None,
                error=None,
                semantic_intent=validated_semantic_intent,
            )
            if (
                initial_state.revision_state == "valid"
                and initial_state.latest_alias_state == "matches_identity"
                and initial_state.pending_marker_state == "absent"
            ):
                return _ArtifactWriteAttempt(
                    result=self._artifact_write_result(
                        outcome="committed",
                        feature_id=feature_id,
                        artifact_type=artifact_type,
                        artifact_hash=artifact_hash,
                        revision_path=revision_path,
                        latest_path=latest_path,
                        pending_path=pending_path,
                        commit_started=False,
                        failure_stage=None,
                        error=None,
                        semantic_intent=validated_semantic_intent,
                    ),
                    error=None,
                )

            failure_stage = "pending_marker_write"
            pending_write_attempted = True
            if validated_semantic_intent is None:
                self._write_text_atomic(
                    pending_path,
                    self._serialize_pending_write(
                        feature_id=feature_id,
                        artifact_type=artifact_type,
                        artifact_hash=artifact_hash,
                        revision_path=revision_path,
                        latest_path=latest_path,
                    ),
                )
            else:
                try:
                    self._write_text_exclusive_atomic(
                        pending_path,
                        self._serialize_semantic_pending_write(
                            feature_id=feature_id,
                            artifact_type=artifact_type,
                            artifact_hash=artifact_hash,
                            revision_path=revision_path,
                            latest_path=latest_path,
                            semantic_intent=validated_semantic_intent,
                        ),
                    )
                except FileExistsError:
                    failure_stage = "admission_validation"
                    existing_semantic_pending = (
                        self._validate_semantic_pending_locator(
                            pending_path,
                            feature_id=feature_id,
                            artifact_type=artifact_type,
                            semantic_binding_key=(
                                validated_semantic_intent.semantic_binding_key
                            ),
                            semantic_binding_version=(
                                validated_semantic_intent.semantic_binding_version
                            ),
                            commit_conflict_key=(
                                validated_semantic_intent.commit_conflict_key
                            ),
                            commit_conflict_key_hash=(
                                validated_semantic_intent.commit_conflict_key_hash
                            ),
                        )
                    )
                    if existing_semantic_pending is None:
                        raise ArtifactStoreError(
                            "Semantic reservation disappeared during admission."
                        )
                    error = ArtifactStoreError(
                        "Existing semantic reservation requires a new attempt"
                    )
                    return _ArtifactWriteAttempt(
                        result=self._artifact_write_result(
                            outcome="partial",
                            feature_id=feature_id,
                            artifact_type=artifact_type,
                            artifact_hash=artifact_hash,
                            revision_path=revision_path,
                            latest_path=latest_path,
                            pending_path=pending_path,
                            commit_started=False,
                            failure_stage="admission_validation",
                            error=error,
                            semantic_intent=validated_semantic_intent,
                        ),
                        error=error,
                    )
                installed = self._validate_semantic_pending_locator(
                    pending_path,
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    semantic_binding_key=(
                        validated_semantic_intent.semantic_binding_key
                    ),
                    semantic_binding_version=(
                        validated_semantic_intent.semantic_binding_version
                    ),
                    commit_conflict_key=(
                        validated_semantic_intent.commit_conflict_key
                    ),
                    commit_conflict_key_hash=(
                        validated_semantic_intent.commit_conflict_key_hash
                    ),
                )
                if (
                    installed is None
                    or installed.get("artifact_hash") != artifact_hash
                ):
                    raise ArtifactStoreError(
                        "Semantic reservation readback identity changed."
                    )
            commit_started = True
            if validated_semantic_intent is not None:
                failure_stage = "commit_verification"
                try:
                    second_intent = project_artifact_write_intent(
                        reader=self,
                        feature_id=feature_id,
                        artifact_type=artifact_type,
                        payload=data,
                        status=status,
                        upstream_refs=resolved_upstream_refs,
                        exact_revision_upstream_types=exact_upstream_types,
                        blockers=tuple(blockers or ()),
                        trace_refs=tuple(trace_refs or ()),
                        protocol_refs=validated_semantic_intent.protocol_refs,
                        execution_file_bindings=(
                            validated_semantic_intent.execution_file_bindings
                        ),
                    )
                    if second_intent != validated_semantic_intent:
                        raise ArtifactStoreError(
                            "Semantic write identity changed after reservation."
                        )
                except Exception:
                    try:
                        self._unlink_exact_semantic_pending_marker(
                            pending_path,
                            validated_semantic_intent,
                        )
                    except Exception as cleanup_exc:
                        failure_stage = "pending_marker_cleanup"
                        raise cleanup_exc
                    raise
            failure_stage = "immutable_revision_write"
            if existing_revision_path is None:
                self._write_text_atomic(revision_path, serialized_envelope)
            failure_stage = "latest_alias_write"
            self._write_text_atomic(latest_path, serialized_envelope)
            failure_stage = "commit_verification"
            before_cleanup = self._artifact_write_result(
                outcome="partial",
                feature_id=feature_id,
                artifact_type=artifact_type,
                artifact_hash=artifact_hash,
                revision_path=revision_path,
                latest_path=latest_path,
                pending_path=pending_path,
                commit_started=True,
                failure_stage=None,
                error=None,
                semantic_intent=validated_semantic_intent,
            )
            if (
                before_cleanup.revision_state != "valid"
                or before_cleanup.latest_alias_state != "matches_identity"
                or before_cleanup.pending_marker_state != "present"
            ):
                raise ArtifactStoreError(
                    "Artifact commit verification failed before pending cleanup"
                )
            failure_stage = "pending_marker_cleanup"
            if validated_semantic_intent is None:
                self._unlink_if_exists(pending_path)
            else:
                self._unlink_exact_semantic_pending_marker(
                    pending_path,
                    validated_semantic_intent,
                )
            committed = self._artifact_write_result(
                outcome="committed",
                feature_id=feature_id,
                artifact_type=artifact_type,
                artifact_hash=artifact_hash,
                revision_path=revision_path,
                latest_path=latest_path,
                pending_path=pending_path,
                commit_started=True,
                failure_stage=None,
                error=None,
                semantic_intent=validated_semantic_intent,
            )
            if (
                committed.revision_state != "valid"
                or committed.latest_alias_state != "matches_identity"
                or committed.pending_marker_state != "absent"
            ):
                failure_stage = "commit_verification"
                raise ArtifactStoreError(
                    "Artifact commit verification failed after pending cleanup"
                )
            return _ArtifactWriteAttempt(result=committed, error=None)
        except Exception as exc:
            if (
                commit_started
                and validated_semantic_intent is None
                and revision_path is not None
                and pending_path is not None
                and not revision_path.exists()
            ):
                try:
                    self._unlink_if_exists(pending_path)
                except Exception:
                    pass
            if (
                not commit_started
                and pending_write_attempted
                and artifact_hash is not None
                and pending_path is not None
            ):
                try:
                    if validated_semantic_intent is None:
                        installed_marker = self._validate_pending_locator(
                            pending_path,
                            feature_id=feature_id,
                            artifact_type=artifact_type,
                            artifact_hash=artifact_hash,
                        )
                    else:
                        installed_marker = (
                            self._validate_semantic_pending_locator(
                                pending_path,
                                feature_id=feature_id,
                                artifact_type=artifact_type,
                                semantic_binding_key=(
                                    validated_semantic_intent.semantic_binding_key
                                ),
                                semantic_binding_version=(
                                    validated_semantic_intent.semantic_binding_version
                                ),
                                commit_conflict_key=(
                                    validated_semantic_intent.commit_conflict_key
                                ),
                                commit_conflict_key_hash=(
                                    validated_semantic_intent.commit_conflict_key_hash
                                ),
                            )
                        )
                        if (
                            installed_marker is not None
                            and installed_marker.get("artifact_hash")
                            != artifact_hash
                        ):
                            installed_marker = None
                except (ArtifactStoreError, OSError):
                    commit_started = False
                else:
                    commit_started = installed_marker is not None
            outcome: ArtifactWriteOutcome = (
                "partial" if commit_started else "blocked_before_commit"
            )
            return _ArtifactWriteAttempt(
                result=self._artifact_write_result(
                    outcome=outcome,
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    artifact_hash=artifact_hash,
                    revision_path=revision_path,
                    latest_path=latest_path,
                    pending_path=pending_path,
                    commit_started=commit_started,
                    failure_stage=failure_stage,
                    error=exc,
                    semantic_intent=validated_semantic_intent,
                ),
                error=exc,
            )

    def read(self, ref: ArtifactRef) -> dict[str, Any]:
        """Read an artifact envelope from a guarded artifact reference."""

        path = self._guard_path(Path(ref.path))
        if not path.exists():
            raise ArtifactStoreError(f"Artifact does not exist: {ref.path}")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("artifact_hash") != ref.artifact_hash:
            raise ArtifactStoreError("Artifact hash mismatch")
        self._validate_envelope_fields(envelope)
        if envelope.get("feature_id") != ref.feature_id:
            raise ArtifactStoreError("Artifact feature_id mismatch")
        if envelope.get("artifact_type") != ref.artifact_type:
            raise ArtifactStoreError("Artifact artifact_type mismatch")
        try:
            self.registry.validate_envelope(envelope)
        except ArtifactRegistryError as exc:
            raise ArtifactStoreError(str(exc)) from exc
        content_hash = self._stable_hash(self._envelope_content_for_hash(envelope))
        if content_hash != envelope["artifact_hash"]:
            raise ArtifactStoreError("Artifact content hash mismatch")
        return envelope

    def check_freshness(self, ref: ArtifactRef) -> FreshnessView:
        """Check whether an artifact's upstream hashes still match."""

        return self._check_freshness(ref, recursive=True)

    def check_direct_freshness(self, ref: ArtifactRef) -> FreshnessView:
        """Check only direct upstream hashes for write-time and checkpoint gates."""

        return self._check_freshness(ref, recursive=False)

    def check_exact_revision_freshness(
        self,
        ref: ArtifactRef,
    ) -> FreshnessView:
        """Validate one immutable revision subgraph without latest-pointer drift."""

        return self._check_freshness(
            ref,
            recursive=True,
            require_latest=False,
        )

    def check_freshness_with_exact_revision_types(
        self,
        ref: ArtifactRef,
        *,
        exact_revision_types: set[str],
    ) -> FreshnessView:
        """Check recursive freshness while preserving approved exact roles."""

        return self._check_freshness(
            ref,
            recursive=True,
            exact_revision_types=exact_revision_types,
        )

    def _check_freshness(
        self,
        ref: ArtifactRef,
        *,
        recursive: bool,
        require_latest: bool = True,
        exact_revision_types: set[str] | None = None,
    ) -> FreshnessView:
        envelope = self.read(ref)
        stale_reasons: list[str] = []
        seen = {(str(self._guard_path(Path(ref.path))), ref.artifact_hash)}
        self._collect_stale_reasons(
            envelope,
            stale_reasons,
            seen,
            recursive=recursive,
            require_latest=require_latest,
            exact_revision_types=set(exact_revision_types or set()),
        )
        if stale_reasons:
            return FreshnessView(status="stale", stale_reasons=stale_reasons)
        return FreshnessView(status="fresh", stale_reasons=[])

    def _collect_stale_reasons(
        self,
        envelope: Mapping[str, Any],
        stale_reasons: list[str],
        seen: set[tuple[str, str]],
        *,
        recursive: bool,
        require_latest: bool,
        exact_revision_types: set[str],
    ) -> None:
        for upstream in envelope.get("inputs", {}).get("upstream_artifacts", []):
            upstream_requires_latest = (
                require_latest
                and upstream.get("require_latest", True) is not False
                and upstream.get("artifact_type") not in exact_revision_types
            )
            upstream_ref = ArtifactRef(
                feature_id=envelope["feature_id"],
                artifact_type=upstream["artifact_type"],
                path=upstream["path"],
                artifact_hash=upstream["artifact_hash"],
            )
            upstream_path = self._guard_path(Path(upstream_ref.path))
            if not upstream_path.exists():
                self._append_stale_reason(
                    stale_reasons,
                    f"{upstream_ref.artifact_type} missing at {upstream_ref.path}",
                )
                continue

            try:
                upstream_envelope = self.read(upstream_ref)
            except ArtifactStoreError as exc:
                reason = (
                    f"{upstream_ref.artifact_type} hash changed"
                    if str(exc) == "Artifact hash mismatch"
                    else f"{upstream_ref.artifact_type} invalid: {exc}"
                )
                self._append_stale_reason(stale_reasons, reason)
                continue
            if upstream_requires_latest:
                try:
                    latest_hash = self._latest_artifact_hash(
                        upstream_ref.feature_id,
                        upstream_ref.artifact_type,
                    )
                except ArtifactStoreError as exc:
                    self._append_stale_reason(
                        stale_reasons,
                        f"{upstream_ref.artifact_type} latest invalid: {exc}",
                    )
                    continue
                if (
                    latest_hash is not None
                    and latest_hash != upstream_ref.artifact_hash
                ):
                    self._append_stale_reason(
                        stale_reasons,
                        f"{upstream_ref.artifact_type} hash changed",
                    )
                    continue

            upstream_key = (str(upstream_path), upstream_ref.artifact_hash)
            if upstream_key in seen:
                continue
            seen.add(upstream_key)
            if recursive:
                recursive_requires_latest = (
                    require_latest
                    if upstream_ref.artifact_type in exact_revision_types
                    else upstream_requires_latest
                )
                self._collect_stale_reasons(
                    upstream_envelope,
                    stale_reasons,
                    seen,
                    recursive=recursive,
                    require_latest=recursive_requires_latest,
                    exact_revision_types=exact_revision_types,
                )

    def read_for_stage(self, ref: ArtifactRef, *, consumer_stage: str) -> dict[str, Any]:
        """Read an artifact for a registered consumer, blocking stale inputs."""

        try:
            self.registry.validate_consumer(ref.artifact_type, consumer_stage)
        except ArtifactRegistryError as exc:
            raise ArtifactStoreError(str(exc)) from exc
        freshness = self.check_freshness(ref)
        if not freshness.is_fresh:
            raise StaleArtifactError("; ".join(freshness.stale_reasons))
        return self.read(ref)

    def list_feature_artifacts(self, feature_id: str) -> list[ArtifactRef]:
        """List all artifacts for a feature."""

        self._validate_feature_id(feature_id)
        feature_dir = self._guard_path(self.root / "features" / feature_id)
        if not feature_dir.exists():
            return []

        refs: list[ArtifactRef] = []
        for path in sorted(feature_dir.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            refs.append(
                ArtifactRef(
                    feature_id=envelope["feature_id"],
                    artifact_type=envelope["artifact_type"],
                    path=str(path),
                    artifact_hash=envelope["artifact_hash"],
                )
            )
        return refs

    def list_feature_artifact_revisions(
        self,
        feature_id: str,
        artifact_type: str,
    ) -> list[ArtifactRef]:
        """List every immutable revision for one feature artifact type."""

        self._validate_feature_id(feature_id)
        self._validate_artifact_type(artifact_type)
        revision_directories = (
            self._guard_path(
                self.root
                / "features"
                / feature_id
                / _REVISION_DIRECTORY
                / artifact_type
            ),
            self._guard_path(
                self.root
                / "features"
                / feature_id
                / "revisions"
                / artifact_type
            ),
        )
        refs: list[ArtifactRef] = []
        seen: set[tuple[str, str]] = set()
        for revision_directory in revision_directories:
            if not revision_directory.exists():
                continue
            for path in sorted(revision_directory.glob("*.json")):
                envelope = json.loads(path.read_text(encoding="utf-8"))
                artifact_hash = envelope.get("artifact_hash")
                if not isinstance(artifact_hash, str):
                    raise ArtifactStoreError(
                        "Artifact revision envelope missing artifact_hash"
                    )
                ref = ArtifactRef(
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    path=str(path),
                    artifact_hash=artifact_hash,
                )
                self.read(ref)
                key = (ref.path, ref.artifact_hash)
                if key not in seen:
                    refs.append(ref)
                    seen.add(key)
        return refs

    def list_feature_ids(self) -> list[str]:
        """List feature ids with artifact directories."""

        features_dir = self._guard_path(self.root / "features")
        if not features_dir.exists():
            return []
        return sorted(path.name for path in features_dir.iterdir() if path.is_dir())

    def recover_pending_writes(self) -> list[dict[str, str]]:
        """Complete pending revision-to-latest writes after an interrupted write."""

        features_dir = self._guard_path(self.root / "features")
        if not features_dir.exists():
            return []

        results: list[dict[str, str]] = []
        for marker_path in sorted(features_dir.glob(f"*/{_PENDING_DIRECTORY}/*/*.json")):
            results.append(self._recover_pending_write(marker_path))
        return results

    def preflight_atomic_commit(
        self,
        *,
        feature_id: str,
        artifact_types: tuple[str, ...],
    ) -> None:
        """Probe actual write locations without creating business artifacts."""

        self._validate_feature_id(feature_id)
        if not artifact_types:
            raise ArtifactStoreError("Artifact preflight requires artifact_types")

        synthetic_hash = "0" * 64
        probes: list[tuple[str, str, Path]] = []
        for artifact_type in artifact_types:
            self._validate_artifact_type(artifact_type)
            targets = (
                (
                    "pending_marker",
                    self._pending_write_path(feature_id, artifact_type, synthetic_hash),
                ),
                (
                    "revision",
                    self._revision_artifact_path(feature_id, artifact_type, synthetic_hash),
                ),
                (
                    "latest_projection",
                    self._artifact_path(feature_id, artifact_type),
                ),
            )
            for path_role, business_path in targets:
                probe_path = business_path.with_name(
                    self._preflight_filename(len(business_path.stem))
                )
                probes.append((artifact_type, path_role, self._guard_path(probe_path)))

        created_directories = self._missing_probe_directories(
            path for _, _, path in probes
        )
        try:
            for artifact_type, path_role, probe_path in probes:
                try:
                    self._write_text_atomic(
                        probe_path,
                        _PREFLIGHT_CONTENT,
                        capture_replace_error=True,
                    )
                except _AtomicReplaceFailure as exc:
                    winerror = getattr(exc.cause, "winerror", None)
                    if winerror == 3:
                        raise ArtifactStorePreflightError(
                            artifact_type=artifact_type,
                            path_role=path_role,
                            source_path_length=len(str(exc.source_path)),
                            target_path_length=len(str(exc.target_path)),
                            target_parent_exists=exc.target_path.parent.exists(),
                            winerror=winerror,
                        ) from exc.cause
                    raise exc.cause
                finally:
                    self._unlink_if_exists(probe_path)
        finally:
            for directory in sorted(
                created_directories,
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                try:
                    directory.rmdir()
                except (FileNotFoundError, OSError):
                    pass

    def semantic_read_immutable(self, ref: ArtifactRef) -> Mapping[str, Any]:
        """Read one exact immutable revision for trusted semantic admission."""

        self._validate_immutable_upstream_ref(
            feature_id=ref.feature_id,
            upstream=ref,
        )
        return self.read(ref)

    def project_artifact_hash(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        payload: Mapping[str, Any],
        status: str,
        blockers: Iterable[Mapping[str, Any]],
        trace_refs: Iterable[str],
        upstream_refs: Iterable[ArtifactRef],
        exact_revision_upstream_types: set[str],
    ) -> str:
        """Project the Store-owned envelope content hash without writing."""

        self._validate_feature_id(feature_id)
        self._validate_artifact_type(artifact_type)
        try:
            contract = self.registry.validate_write(
                artifact_type=artifact_type,
                status=status,
                payload=dict(payload),
            )
        except ArtifactRegistryError as exc:
            raise ArtifactStoreError(str(exc)) from exc
        exact_types = set(exact_revision_upstream_types)
        inputs = {
            "upstream_artifacts": [
                {
                    "artifact_type": ref.artifact_type,
                    "path": ref.path,
                    "artifact_hash": ref.artifact_hash,
                    "require_latest": ref.artifact_type not in exact_types,
                }
                for ref in upstream_refs
            ]
        }
        content = {
            "schema_version": contract.schema_version,
            "artifact_type": artifact_type,
            "feature_id": feature_id,
            "producer_stage": contract.producer_stage,
            "status": status,
            "data": dict(payload),
            "inputs": inputs,
            "blockers": [dict(item) for item in blockers],
            "trace_refs": list(trace_refs),
        }
        return self._stable_hash(self._envelope_content_for_hash(content))

    def exists(self, ref: ArtifactRef) -> bool:
        """Return whether a guarded artifact reference exists."""

        try:
            return self._guard_path(Path(ref.path)).exists()
        except ArtifactStoreError:
            return False

    def _artifact_path(self, feature_id: str, artifact_type: str) -> Path:
        return self._guard_path(
            self.root / "features" / feature_id / f"{artifact_type}.json"
        )

    def _revision_artifact_path(
        self,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str,
    ) -> Path:
        return self._guard_path(
            self.root
            / "features"
            / feature_id
            / _REVISION_DIRECTORY
            / artifact_type
            / f"{self._physical_locator(artifact_hash)}.json"
        )

    def _legacy_revision_artifact_path(
        self,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str,
    ) -> Path:
        return self._guard_path(
            self.root
            / "features"
            / feature_id
            / "revisions"
            / artifact_type
            / f"{artifact_hash}.json"
        )

    def _pending_write_path(
        self,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str,
    ) -> Path:
        return self._guard_path(
            self.root
            / "features"
            / feature_id
            / _PENDING_DIRECTORY
            / artifact_type
            / f"{self._physical_locator(artifact_hash)}.json"
        )

    def _semantic_pending_write_path(
        self,
        feature_id: str,
        artifact_type: str,
        commit_conflict_key_hash: str,
    ) -> Path:
        prefix = "sha256:"
        if not commit_conflict_key_hash.startswith(prefix):
            raise ArtifactStoreError("Semantic conflict key hash is invalid")
        digest = commit_conflict_key_hash[len(prefix) :]
        return self._guard_path(
            self.root
            / "features"
            / feature_id
            / _PENDING_DIRECTORY
            / artifact_type
            / f"{self._physical_locator(digest)}.json"
        )

    def _latest_artifact_hash(self, feature_id: str, artifact_type: str) -> str | None:
        latest_path = self._artifact_path(feature_id, artifact_type)
        if not latest_path.exists():
            return None
        envelope = json.loads(latest_path.read_text(encoding="utf-8"))
        artifact_hash = envelope.get("artifact_hash")
        if not isinstance(artifact_hash, str):
            raise ArtifactStoreError("Latest artifact envelope missing artifact_hash")
        latest_ref = ArtifactRef(
            feature_id=feature_id,
            artifact_type=artifact_type,
            path=str(latest_path),
            artifact_hash=artifact_hash,
        )
        return str(self.read(latest_ref)["artifact_hash"])

    def _write_text_atomic(
        self,
        path: Path,
        content: str,
        *,
        capture_replace_error: bool = False,
    ) -> None:
        guarded_path = self._guard_path(path)
        guarded_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._guard_path(
            guarded_path.with_name(f".tmp-{uuid4().hex[:12]}.json")
        )
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temp_path, guarded_path)
            except OSError as exc:
                if capture_replace_error:
                    raise _AtomicReplaceFailure(temp_path, guarded_path, exc) from exc
                raise
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _write_text_exclusive_atomic(self, path: Path, content: str) -> None:
        """Install complete marker bytes without replacing a competing marker."""

        guarded_path = self._guard_path(path)
        guarded_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._guard_path(
            guarded_path.with_name(f".tmp-{uuid4().hex[:12]}.json")
        )
        try:
            with temp_path.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp_path, guarded_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _recover_pending_write(self, marker_path: Path) -> dict[str, str]:
        guarded_marker_path = self._guard_path(marker_path)
        marker = json.loads(guarded_marker_path.read_text(encoding="utf-8"))
        self._validate_pending_write_marker(marker)

        if marker.get("schema_version") == "2.0":
            conflict_key = marker.get("commit_conflict_key")
            if not isinstance(conflict_key, Mapping):
                raise ArtifactStoreError("Semantic pending marker key is invalid")
            conflict_hash = hash_action_identity(
                conflict_key,
                expected_hash_domain=SEMANTIC_CONFLICT_HASH_DOMAIN,
            )
            if marker.get("commit_conflict_key_hash") != conflict_hash:
                raise ArtifactStoreError("Semantic pending marker hash mismatch")
            self._validate_semantic_pending_locator(
                guarded_marker_path,
                feature_id=str(marker["feature_id"]),
                artifact_type=str(marker["artifact_type"]),
                semantic_binding_key=str(marker["semantic_binding_key"]),
                semantic_binding_version=str(marker["semantic_binding_version"]),
                commit_conflict_key=conflict_key,
                commit_conflict_key_hash=conflict_hash,
            )
            return self._pending_write_result(
                marker,
                status="blocked",
                reason="semantic_recovery_not_authorized",
            )

        feature_id = str(marker["feature_id"])
        artifact_type = str(marker["artifact_type"])
        artifact_hash = str(marker["artifact_hash"])
        revision_path = self._guard_path(Path(str(marker["revision_path"])))
        latest_path = self._artifact_path(feature_id, artifact_type)
        expected_marker_path = self._pending_write_path(
            feature_id,
            artifact_type,
            artifact_hash,
        )
        if guarded_marker_path != expected_marker_path:
            raise ArtifactStoreError("Pending write marker path mismatch")
        allowed_revision_paths = {
            self._revision_artifact_path(feature_id, artifact_type, artifact_hash),
            self._legacy_revision_artifact_path(
                feature_id,
                artifact_type,
                artifact_hash,
            ),
        }
        if revision_path not in allowed_revision_paths:
            raise ArtifactStoreError("Pending write revision_path mismatch")
        if str(latest_path) != marker["latest_path"]:
            raise ArtifactStoreError("Pending write latest_path mismatch")

        ref = ArtifactRef(
            feature_id=feature_id,
            artifact_type=artifact_type,
            path=str(revision_path),
            artifact_hash=artifact_hash,
        )
        if not revision_path.exists():
            return self._pending_write_result(marker, status="blocked", reason="revision_missing")
        try:
            self.read(ref)
        except ArtifactStoreError as exc:
            return self._pending_write_result(marker, status="blocked", reason=str(exc))

        latest_needs_restore = True
        if latest_path.exists():
            try:
                latest_hash = self._latest_artifact_hash(feature_id, artifact_type)
            except ArtifactStoreError:
                latest_needs_restore = True
            else:
                latest_needs_restore = latest_hash != artifact_hash

        if latest_needs_restore:
            self._write_text_atomic(
                latest_path,
                revision_path.read_text(encoding="utf-8"),
            )
            status = "recovered"
        else:
            status = "already_current"
        self._unlink_if_exists(guarded_marker_path)
        return self._pending_write_result(marker, status=status, reason="")

    def _serialize_pending_write(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str,
        revision_path: Path,
        latest_path: Path,
    ) -> str:
        marker = {
            "schema_version": "1.0",
            "feature_id": feature_id,
            "artifact_type": artifact_type,
            "artifact_hash": artifact_hash,
            "revision_path": str(revision_path),
            "latest_path": str(latest_path),
            "status": _PENDING_WRITE_STATUS,
        }
        return json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    def _serialize_semantic_pending_write(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str,
        revision_path: Path,
        latest_path: Path,
        semantic_intent: ArtifactWriteIntent,
    ) -> str:
        marker = {
            "schema_version": "2.0",
            "status": _SEMANTIC_PENDING_WRITE_STATUS,
            "feature_id": feature_id,
            "artifact_type": artifact_type,
            "artifact_hash": artifact_hash,
            "revision_path": str(revision_path),
            "latest_path": str(latest_path),
            "semantic_binding_key": semantic_intent.semantic_binding_key,
            "semantic_binding_version": semantic_intent.semantic_binding_version,
            "commit_conflict_key": dict(semantic_intent.commit_conflict_key),
            "commit_conflict_key_hash": (
                semantic_intent.commit_conflict_key_hash
            ),
        }
        return json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _pending_write_result(
        marker: Mapping[str, Any],
        *,
        status: str,
        reason: str,
    ) -> dict[str, str]:
        result = {
            "feature_id": str(marker["feature_id"]),
            "artifact_type": str(marker["artifact_type"]),
            "artifact_hash": str(marker["artifact_hash"]),
            "status": status,
        }
        if reason:
            result["reason"] = reason
        return result

    @staticmethod
    def _validate_pending_write_marker(marker: Mapping[str, Any]) -> None:
        if marker.get("schema_version") == "2.0":
            ArtifactStore._validate_semantic_pending_write_marker(marker)
            return
        fields = set(marker)
        missing = sorted(_PENDING_WRITE_FIELDS - fields)
        if missing:
            raise ArtifactStoreError(f"Pending write marker missing fields: {missing}")
        unexpected = sorted(fields - _PENDING_WRITE_FIELDS)
        if unexpected:
            raise ArtifactStoreError(f"Pending write marker unexpected fields: {unexpected}")
        if marker["schema_version"] != "1.0":
            raise ArtifactStoreError("Pending write marker schema_version mismatch")
        if marker["status"] != _PENDING_WRITE_STATUS:
            raise ArtifactStoreError("Pending write marker status mismatch")

    @staticmethod
    def _validate_semantic_pending_write_marker(marker: Mapping[str, Any]) -> None:
        fields = set(marker)
        missing = sorted(_SEMANTIC_PENDING_WRITE_FIELDS - fields)
        if missing:
            raise ArtifactStoreError(
                f"Semantic pending marker missing fields: {missing}"
            )
        unexpected = sorted(fields - _SEMANTIC_PENDING_WRITE_FIELDS)
        if unexpected:
            raise ArtifactStoreError(
                f"Semantic pending marker unexpected fields: {unexpected}"
            )
        if (
            marker.get("schema_version") != "2.0"
            or marker.get("status") != _SEMANTIC_PENDING_WRITE_STATUS
            or not isinstance(marker.get("semantic_binding_key"), str)
            or not marker.get("semantic_binding_key")
            or not isinstance(marker.get("semantic_binding_version"), str)
            or not marker.get("semantic_binding_version")
            or not isinstance(marker.get("commit_conflict_key"), Mapping)
            or not isinstance(marker.get("commit_conflict_key_hash"), str)
        ):
            raise ArtifactStoreError("Semantic pending marker contract mismatch")

    @staticmethod
    def _unlink_if_exists(path: Path) -> None:
        if path.exists():
            path.unlink()

    def _find_existing_revision_path(
        self,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str,
    ) -> Path | None:
        legacy_path = self._legacy_revision_artifact_path(
            feature_id,
            artifact_type,
            artifact_hash,
        )
        revision_path = self._revision_artifact_path(
            feature_id,
            artifact_type,
            artifact_hash,
        )
        found: list[Path] = []
        for path, is_short_locator in (
            (legacy_path, False),
            (revision_path, True),
        ):
            if not path.exists():
                continue
            ref = ArtifactRef(
                feature_id=feature_id,
                artifact_type=artifact_type,
                path=str(path),
                artifact_hash=artifact_hash,
            )
            try:
                self.read(ref)
            except ArtifactStoreError as exc:
                if is_short_locator and str(exc) == "Artifact hash mismatch":
                    raise ArtifactStoreError(
                        "Artifact revision locator collision"
                    ) from exc
                raise ArtifactStoreError(
                    f"Existing artifact revision is invalid: {exc}"
                ) from exc
            found.append(path)
        if len(found) == 2 and found[0].read_bytes() != found[1].read_bytes():
            raise ArtifactStoreError(
                "Duplicate artifact revisions have inconsistent bytes"
            )
        return found[0] if found else None

    def _validate_pending_locator(
        self,
        path: Path,
        *,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str,
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
            self._validate_pending_write_marker(marker)
        except (ArtifactStoreError, json.JSONDecodeError, OSError) as exc:
            raise ArtifactStoreError("Existing pending write marker is invalid") from exc
        if (
            marker.get("feature_id") != feature_id
            or marker.get("artifact_type") != artifact_type
            or marker.get("artifact_hash") != artifact_hash
        ):
            raise ArtifactStoreError("Artifact pending locator collision")
        try:
            revision_path = self._guard_path(Path(str(marker["revision_path"])))
            latest_path = self._guard_path(Path(str(marker["latest_path"])))
        except (ArtifactStoreError, OSError, ValueError) as exc:
            raise ArtifactStoreError("Existing pending write marker is invalid") from exc
        allowed_revision_paths = {
            self._revision_artifact_path(feature_id, artifact_type, artifact_hash),
            self._legacy_revision_artifact_path(
                feature_id,
                artifact_type,
                artifact_hash,
            ),
        }
        if revision_path not in allowed_revision_paths:
            raise ArtifactStoreError("Existing pending write marker is invalid")
        if latest_path != self._artifact_path(feature_id, artifact_type):
            raise ArtifactStoreError("Existing pending write marker is invalid")
        return dict(marker)

    def _validate_semantic_pending_locator(
        self,
        path: Path,
        *,
        feature_id: str,
        artifact_type: str,
        semantic_binding_key: str,
        semantic_binding_version: str,
        commit_conflict_key: Mapping[str, Any],
        commit_conflict_key_hash: str,
    ) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
            self._validate_semantic_pending_write_marker(marker)
        except (ArtifactStoreError, json.JSONDecodeError, OSError) as exc:
            raise ArtifactStoreError(
                "Existing semantic reservation is invalid"
            ) from exc
        expected_path = self._semantic_pending_write_path(
            feature_id,
            artifact_type,
            commit_conflict_key_hash,
        )
        if path != expected_path:
            raise ArtifactStoreError("Semantic reservation path mismatch")
        if (
            marker.get("feature_id") != feature_id
            or marker.get("artifact_type") != artifact_type
            or marker.get("semantic_binding_key") != semantic_binding_key
            or marker.get("semantic_binding_version")
            != semantic_binding_version
            or commit_conflict_key.get("semantic_binding_key")
            != semantic_binding_key
            or commit_conflict_key.get("semantic_binding_version")
            != semantic_binding_version
            or marker.get("commit_conflict_key_hash")
            != commit_conflict_key_hash
            or marker.get("commit_conflict_key") != dict(commit_conflict_key)
        ):
            raise ArtifactStoreError("Semantic reservation locator collision")
        artifact_hash = marker.get("artifact_hash")
        if not isinstance(artifact_hash, str):
            raise ArtifactStoreError("Semantic reservation artifact hash is invalid")
        revision_path = self._guard_path(Path(str(marker["revision_path"])))
        latest_path = self._guard_path(Path(str(marker["latest_path"])))
        if revision_path not in {
            self._revision_artifact_path(feature_id, artifact_type, artifact_hash),
            self._legacy_revision_artifact_path(
                feature_id,
                artifact_type,
                artifact_hash,
            ),
        }:
            raise ArtifactStoreError("Semantic reservation revision path mismatch")
        if latest_path != self._artifact_path(feature_id, artifact_type):
            raise ArtifactStoreError("Semantic reservation latest path mismatch")
        return dict(marker)

    def _unlink_exact_semantic_pending_marker(
        self,
        path: Path,
        semantic_intent: ArtifactWriteIntent,
    ) -> None:
        marker = self._validate_semantic_pending_locator(
            path,
            feature_id=semantic_intent.feature_id,
            artifact_type=semantic_intent.artifact_type,
            semantic_binding_key=semantic_intent.semantic_binding_key,
            semantic_binding_version=semantic_intent.semantic_binding_version,
            commit_conflict_key=semantic_intent.commit_conflict_key,
            commit_conflict_key_hash=semantic_intent.commit_conflict_key_hash,
        )
        if marker is None:
            raise ArtifactStoreError("Semantic reservation is missing during cleanup")
        path.unlink()

    def _artifact_write_result(
        self,
        *,
        outcome: ArtifactWriteOutcome,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str | None,
        revision_path: Path | None,
        latest_path: Path | None,
        pending_path: Path | None,
        commit_started: bool,
        failure_stage: ArtifactWriteFailureStage | None,
        error: Exception | None,
        semantic_intent: ArtifactWriteIntent | None = None,
    ) -> ArtifactWriteResult:
        revision_state = self._artifact_revision_state(
            feature_id=feature_id,
            artifact_type=artifact_type,
            artifact_hash=artifact_hash,
            revision_path=revision_path,
        )
        latest_alias_state = self._artifact_latest_alias_state(
            feature_id=feature_id,
            artifact_type=artifact_type,
            artifact_hash=artifact_hash,
            latest_path=latest_path,
        )
        pending_marker_state = self._artifact_pending_marker_state(
            feature_id=feature_id,
            artifact_type=artifact_type,
            artifact_hash=artifact_hash,
            pending_path=pending_path,
            semantic_intent=semantic_intent,
        )
        artifact_ref = None
        if (
            outcome == "committed"
            and artifact_hash is not None
            and revision_path is not None
            and revision_state == "valid"
            and latest_alias_state == "matches_identity"
            and pending_marker_state == "absent"
        ):
            artifact_ref = ArtifactRef(
                feature_id=feature_id,
                artifact_type=artifact_type,
                path=str(revision_path),
                artifact_hash=artifact_hash,
            )
        return ArtifactWriteResult(
            outcome=outcome,
            feature_id=feature_id,
            artifact_type=artifact_type,
            artifact_hash=artifact_hash,
            artifact_ref=artifact_ref,
            revision_path=(None if revision_path is None else str(revision_path)),
            latest_alias_path=(None if latest_path is None else str(latest_path)),
            pending_marker_path=(None if pending_path is None else str(pending_path)),
            commit_started=commit_started,
            revision_state=revision_state,
            latest_alias_state=latest_alias_state,
            pending_marker_state=pending_marker_state,
            failure_stage=failure_stage,
            failure_class=None if error is None else type(error).__name__,
            failure_message=None if error is None else str(error),
            recovery_available=(
                semantic_intent is None
                and outcome == "partial"
                and pending_marker_state == "present"
                and revision_state == "valid"
            ),
            recovery_authorized=False,
        )

    def _artifact_revision_state(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str | None,
        revision_path: Path | None,
    ) -> ArtifactRevisionState:
        if artifact_hash is None or revision_path is None:
            return "unavailable"
        try:
            if not revision_path.exists():
                return "absent"
            self.read(
                ArtifactRef(
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    path=str(revision_path),
                    artifact_hash=artifact_hash,
                )
            )
        except ArtifactStoreError:
            return "invalid"
        except OSError:
            return "unknown"
        return "valid"

    def _artifact_latest_alias_state(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str | None,
        latest_path: Path | None,
    ) -> ArtifactLatestAliasState:
        if artifact_hash is None or latest_path is None:
            return "unavailable"
        try:
            if not latest_path.exists():
                return "absent"
            envelope = json.loads(latest_path.read_text(encoding="utf-8"))
            latest_hash = envelope.get("artifact_hash")
            if not isinstance(latest_hash, str):
                return "invalid"
            self.read(
                ArtifactRef(
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    path=str(latest_path),
                    artifact_hash=latest_hash,
                )
            )
        except ArtifactStoreError:
            return "invalid"
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "invalid"
        except OSError:
            return "unknown"
        return (
            "matches_identity"
            if latest_hash == artifact_hash
            else "different_identity"
        )

    def _artifact_pending_marker_state(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        artifact_hash: str | None,
        pending_path: Path | None,
        semantic_intent: ArtifactWriteIntent | None = None,
    ) -> ArtifactPendingMarkerState:
        if artifact_hash is None or pending_path is None:
            return "unavailable"
        try:
            if not pending_path.exists():
                return "absent"
            if semantic_intent is not None:
                marker = self._validate_semantic_pending_locator(
                    pending_path,
                    feature_id=feature_id,
                    artifact_type=artifact_type,
                    semantic_binding_key=semantic_intent.semantic_binding_key,
                    semantic_binding_version=(
                        semantic_intent.semantic_binding_version
                    ),
                    commit_conflict_key=semantic_intent.commit_conflict_key,
                    commit_conflict_key_hash=(
                        semantic_intent.commit_conflict_key_hash
                    ),
                )
                return "present" if marker is not None else "absent"
            marker = json.loads(pending_path.read_text(encoding="utf-8"))
            self._validate_pending_write_marker(marker)
            if (
                marker.get("feature_id") != feature_id
                or marker.get("artifact_type") != artifact_type
                or marker.get("artifact_hash") != artifact_hash
            ):
                return "invalid"
            marker_revision = self._guard_path(Path(str(marker["revision_path"])))
            marker_latest = self._guard_path(Path(str(marker["latest_path"])))
            if marker_revision not in {
                self._revision_artifact_path(
                    feature_id,
                    artifact_type,
                    artifact_hash,
                ),
                self._legacy_revision_artifact_path(
                    feature_id,
                    artifact_type,
                    artifact_hash,
                ),
            }:
                return "invalid"
            if marker_latest != self._artifact_path(feature_id, artifact_type):
                return "invalid"
        except (ArtifactStoreError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return "invalid"
        except OSError:
            return "unknown"
        return "present"

    def _missing_probe_directories(self, paths: Iterable[Path]) -> set[Path]:
        missing: set[Path] = set()
        if not self.root.exists():
            missing.add(self.root)
        for path in paths:
            current = path.parent
            while current != self.root:
                if not current.exists():
                    missing.add(current)
                current = current.parent
        return missing

    @staticmethod
    def _preflight_filename(stem_length: int) -> str:
        digest = hashlib.sha256(uuid4().hex.encode("ascii")).hexdigest()
        if stem_length < 2:
            raise ArtifactStoreError("Artifact preflight target stem is too short")
        return f".{digest[: stem_length - 1]}.json"

    @staticmethod
    def _physical_locator(artifact_hash: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
            raise ArtifactStoreError("Invalid artifact_hash for physical locator")
        return artifact_hash[:_PHYSICAL_LOCATOR_HEX_LENGTH]

    def _validate_edit_authority_request_identity_upstreams(
        self,
        *,
        feature_id: str,
        payload: Mapping[str, Any],
        upstream_refs: list[ArtifactRef],
    ) -> None:
        try:
            validate_edit_authority_request_identity(
                payload,
                allow_legacy=False,
            )
            bindings = bind_edit_authority_exact_input_upstreams(
                payload,
                [
                    {
                        "artifact_type": ref.artifact_type,
                        "artifact_hash": ref.artifact_hash,
                    }
                    for ref in upstream_refs
                ],
            )
        except InvalidEditAuthorityApprovalIdentityError as exc:
            raise ArtifactStoreError(str(exc)) from exc
        refs_by_identity: dict[tuple[str, str], list[ArtifactRef]] = {}
        for ref in upstream_refs:
            refs_by_identity.setdefault(
                (ref.artifact_type, ref.artifact_hash),
                [],
            ).append(ref)
        input_facts: list[dict[str, Any]] = []
        for requirement, _ in bindings:
            key = (
                str(requirement.get("artifact_type")),
                str(requirement.get("artifact_hash")),
            )
            matches = refs_by_identity.get(key)
            if not matches:
                raise ArtifactStoreError(
                    "Edit Authority Request exact input has no immutable upstream."
                )
            ref = matches.pop(0)
            if ref.feature_id != feature_id:
                raise ArtifactStoreError(
                    "Edit Authority Request exact input feature_id mismatch."
                )
            self.read(ref)
            input_facts.append(
                {
                    "role": requirement["role"],
                    "artifact_type": ref.artifact_type,
                    "artifact_hash": ref.artifact_hash,
                    "registered": True,
                    "latest_artifact_hash": self._latest_artifact_hash(
                        feature_id,
                        ref.artifact_type,
                    ),
                    "recursive_fresh": self.check_freshness(ref).is_fresh,
                    "exact_recursive_fresh": (
                        self.check_exact_revision_freshness(ref).is_fresh
                    ),
                }
            )
        try:
            validate_edit_authority_exact_input_admission(payload, input_facts)
        except InvalidEditAuthorityApprovalIdentityError as exc:
            raise ArtifactStoreError(str(exc)) from exc

    def _validate_immutable_upstream_ref(
        self,
        *,
        feature_id: str,
        upstream: ArtifactRef,
    ) -> None:
        """Require one new durable upstream to name a registered revision."""

        if upstream.feature_id != feature_id:
            raise ArtifactStoreError(
                f"Upstream artifact feature_id mismatch for {upstream.artifact_type!r}"
            )
        self._validate_artifact_type(upstream.artifact_type)
        upstream_path = self._guard_path(Path(upstream.path))
        allowed_revision_paths = {
            self._revision_artifact_path(
                feature_id,
                upstream.artifact_type,
                upstream.artifact_hash,
            ),
            self._legacy_revision_artifact_path(
                feature_id,
                upstream.artifact_type,
                upstream.artifact_hash,
            ),
        }
        if upstream_path not in allowed_revision_paths:
            raise ArtifactStoreError(
                "Upstream artifact must reference a registered immutable revision: "
                f"{upstream.artifact_type!r}"
            )
        self.read(upstream)

    def _validate_upstream_refs(
        self,
        *,
        feature_id: str,
        artifact_type: str,
        contract: ArtifactContract,
        payload: Mapping[str, Any],
        status: str,
        upstream_refs: list[ArtifactRef],
        allow_stale_upstreams: bool,
        exact_revision_upstream_types: set[str],
    ) -> None:
        for upstream in upstream_refs:
            self._validate_immutable_upstream_ref(
                feature_id=feature_id,
                upstream=upstream,
            )
        provided_types = {ref.artifact_type for ref in upstream_refs}
        if artifact_type == "edit_authority_request":
            self._validate_edit_authority_request_identity_upstreams(
                feature_id=feature_id,
                payload=payload,
                upstream_refs=upstream_refs,
            )
        if artifact_type == "build_result":
            verification_only = (
                payload.get("input_execution_mode") == "verification_only"
            )
            verification_types = {"feature_spec", "evidence_pack"}
            mutation_types = {"dry_run_result", "isolated_apply_result"}
            if verification_only:
                if (
                    provided_types != verification_types
                    or len(upstream_refs) != len(verification_types)
                ):
                    raise ArtifactStoreError(
                        "Verification-only BuildResult requires only one exact "
                        "FeatureSpec and one exact EvidencePack upstream."
                    )
                lineage = payload.get("verification_only_lineage")
                refs_by_type = {ref.artifact_type: ref for ref in upstream_refs}
                if not isinstance(lineage, Mapping) or any(
                    not isinstance(lineage.get(field), Mapping)
                    or lineage[field].get("path") != refs_by_type[upstream_type].path
                    or lineage[field].get("artifact_hash")
                    != refs_by_type[upstream_type].artifact_hash
                    for field, upstream_type in (
                        ("feature_spec", "feature_spec"),
                        ("evidence_pack", "evidence_pack"),
                    )
                ):
                    raise ArtifactStoreError(
                        "Verification-only BuildResult lineage does not match "
                        "its exact upstreams."
                    )
                if not isinstance(lineage.get("preflight_hash"), str) or not lineage.get(
                    "preflight_hash"
                ):
                    raise ArtifactStoreError(
                        "Verification-only BuildResult is missing its preflight hash."
                    )
            elif (
                "dry_run_result" not in provided_types
                or not provided_types.issubset(mutation_types)
                or len(upstream_refs) != len(provided_types)
                or payload.get("verification_only_lineage") is not None
            ):
                raise ArtifactStoreError(
                    "Mutation BuildResult requires dry_run_result, optional "
                    "isolated_apply_result, and no verification-only roles."
                )
        if (
            artifact_type == "edit_authority_approval"
            and payload.get("authority_source")
            != "edit_authority_request"
        ):
            raise ArtifactStoreError(
                "New EditAuthorityApproval writes must originate from an "
                "exact EditAuthorityRequest and ResumeReport."
            )
        request_payload = payload.get("request_payload")
        if (
            artifact_type == "repair_planning_request"
            and isinstance(request_payload, Mapping)
            and "binding_resolution"
            in contract.required_upstream_artifacts
        ):
            binding_ref = next(
                (
                    ref
                    for ref in upstream_refs
                    if ref.artifact_type == "binding_resolution"
                ),
                None,
            )
            formal_lineage = request_payload.get("formal_lineage")
            binding_identity = (
                formal_lineage.get("binding_resolution")
                if isinstance(formal_lineage, Mapping)
                else None
            )
            if (
                binding_ref is None
                or not isinstance(binding_identity, Mapping)
                or binding_identity.get("path") != binding_ref.path
                or binding_identity.get("artifact_hash")
                != binding_ref.artifact_hash
            ):
                raise ArtifactStoreError(
                    "Repair planning Request does not bind the exact "
                    "BindingResolution upstream."
                )
        if (
            artifact_type == "edit_authority_request"
            and isinstance(request_payload, Mapping)
            and request_payload.get("schema_version")
            in {
                "stagger_hud_parent_observation_authority_request_v1",
                (
                    "stagger_hud_parent_diagnostic_execution_"
                    "authority_request_v2"
                ),
            }
        ):
            observation_authority_types = {
                "feature_spec",
                "runtime_evidence",
                "implementation_plan",
                "patch_proposal",
                "binding_resolution",
                "edit_authority_approval",
                "evidence_pack",
            }
            if (
                observation_authority_types - provided_types
                or observation_authority_types
                - exact_revision_upstream_types
            ):
                raise ArtifactStoreError(
                    "Stagger observation authority Request requires every "
                    "exact technical and authority role."
                )
            refs_by_type = {
                ref.artifact_type: ref
                for ref in upstream_refs
                if ref.artifact_type in observation_authority_types
            }
            if (
                len(refs_by_type) != len(observation_authority_types)
                or sum(
                    1
                    for ref in upstream_refs
                    if ref.artifact_type in observation_authority_types
                )
                != len(observation_authority_types)
            ):
                raise ArtifactStoreError(
                    "Stagger observation authority Request roles are "
                    "duplicated or incomplete."
                )
            proposal = request_payload.get("proposal")
            revision_roles = (
                ("feature_spec_revision", "feature_spec"),
                ("runtime_evidence_revision", "runtime_evidence"),
                (
                    "current_implementation_plan_revision",
                    "implementation_plan",
                ),
                ("patch_proposal_revision", "patch_proposal"),
                (
                    "existing_binding_resolution_revision",
                    "binding_resolution",
                ),
                (
                    "existing_edit_authority_approval_revision",
                    "edit_authority_approval",
                ),
                ("evidence_pack_revision", "evidence_pack"),
            )
            if not isinstance(proposal, Mapping) or any(
                not isinstance(proposal.get(revision_field), Mapping)
                or proposal[revision_field].get("path")
                != refs_by_type[upstream_type].path
                or proposal[revision_field].get("artifact_hash")
                != refs_by_type[upstream_type].artifact_hash
                for revision_field, upstream_type in revision_roles
            ):
                raise ArtifactStoreError(
                    "Stagger observation authority Request revision roles do "
                    "not match exact upstreams."
                )
        if (
            artifact_type == "binding_resolution"
            and payload.get("authority_source")
            == "edit_authority_approval"
            and payload.get("authority_contract_version")
            not in {
                "edit_authority_request_approval_v1",
                "edit_authority_existing_binding_reauthorization_v1",
            }
        ):
            raise ArtifactStoreError(
                "New approved binding_resolution writes must use the "
                "request-bound Edit Authority contract."
            )
        required_upstreams = contract.required_upstream_artifacts
        if artifact_type == "patch_proposal" and payload.get("context_pack", {}).get("task_agent_protocol") == "v2":
            # 为什么要有它：v2产品原件直接依赖需求和已验证交接，不能伪造旧Evidence/Plan来满足存储格式。
            from src.mod_agent.task_agent_v2_handoff import revalidate_handoff_v2
            pack = payload["context_pack"]
            revalidate_handoff_v2(pack["task_agent_handoff_v2"], store=self)
            if len(upstream_refs) != 1 or provided_types != {"feature_spec"} or {
                    "feature_id": upstream_refs[0].feature_id, "artifact_type": upstream_refs[0].artifact_type,
                    "path": upstream_refs[0].path, "artifact_hash": upstream_refs[0].artifact_hash} != pack["feature_spec_ref"]:
                raise ArtifactStoreError("V2 proposal FeatureSpec binding changed.")
            required_upstreams = frozenset({"feature_spec"})
        missing = sorted(required_upstreams - provided_types)
        if missing:
            raise ArtifactStoreError(
                f"Artifact {artifact_type!r} missing required upstream artifacts: {missing}"
            )
        if (
            artifact_type == "binding_resolution"
            and payload.get("authority_contract_version")
            in {
                "edit_authority_request_approval_v1",
                "edit_authority_existing_binding_reauthorization_v1",
            }
        ):
            authority_types = {
                "edit_authority_request",
                "edit_authority_approval",
                "evidence_pack",
            }
            if (
                payload.get("authority_contract_version")
                == "edit_authority_existing_binding_reauthorization_v1"
            ):
                authority_types.update(
                    {
                        "feature_spec",
                        "runtime_evidence",
                        "implementation_plan",
                        "patch_proposal",
                        "binding_resolution",
                    }
                )
            missing_authority = sorted(authority_types - provided_types)
            not_exact_authority = sorted(
                authority_types - exact_revision_upstream_types
            )
            if missing_authority or not_exact_authority:
                raise ArtifactStoreError(
                    "Approved binding_resolution requires exact request, "
                    "approval, and evidence upstreams."
                )
            refs_by_type = {
                ref.artifact_type: ref
                for ref in upstream_refs
                if ref.artifact_type in authority_types
            }
            for revision_field, upstream_type in (
                (
                    "edit_authority_request_revision",
                    "edit_authority_request",
                ),
                (
                    "edit_authority_approval_revision",
                    "edit_authority_approval",
                ),
                ("evidence_pack_revision", "evidence_pack"),
            ):
                revision = payload.get(revision_field)
                upstream = refs_by_type.get(upstream_type)
                if (
                    not isinstance(revision, Mapping)
                    or upstream is None
                    or revision.get("path") != upstream.path
                    or revision.get("artifact_hash")
                    != upstream.artifact_hash
                ):
                    raise ArtifactStoreError(
                        "Approved binding_resolution payload revision does "
                        f"not match exact {upstream_type} upstream."
                    )
            if (
                payload.get("authority_contract_version")
                == "edit_authority_existing_binding_reauthorization_v1"
            ):
                refs_by_type_list: dict[str, list[ArtifactRef]] = {}
                for ref in upstream_refs:
                    refs_by_type_list.setdefault(
                        ref.artifact_type,
                        [],
                    ).append(ref)

                def revision_matches_upstream(
                    revision_field: str,
                    upstream_type: str,
                ) -> bool:
                    revision = payload.get(revision_field)
                    return isinstance(revision, Mapping) and any(
                        revision.get("path") == ref.path
                        and revision.get("artifact_hash")
                        == ref.artifact_hash
                        for ref in refs_by_type_list.get(
                            upstream_type,
                            [],
                        )
                    )

                for revision_field, upstream_type in (
                    ("runtime_evidence_revision", "runtime_evidence"),
                    ("implementation_plan_revision", "implementation_plan"),
                    ("patch_proposal_revision", "patch_proposal"),
                    (
                        "predecessor_implementation_plan_revision",
                        "implementation_plan",
                    ),
                ):
                    if not revision_matches_upstream(
                        revision_field,
                        upstream_type,
                    ):
                        raise ArtifactStoreError(
                            "Reauthorized binding_resolution payload revision "
                            f"does not match exact {revision_field} upstream."
                        )
                previous_revision = payload.get(
                    "previous_binding_resolution_revision"
                )
                previous_ref = refs_by_type.get("binding_resolution")
                if (
                    not isinstance(previous_revision, Mapping)
                    or previous_ref is None
                    or previous_revision.get("path")
                    != previous_ref.path
                    or previous_revision.get("artifact_hash")
                    != previous_ref.artifact_hash
                ):
                    raise ArtifactStoreError(
                        "Reauthorized binding_resolution does not bind the "
                        "exact previous BindingResolution."
                    )
        if (
            artifact_type == "edit_authority_approval"
            and payload.get("authority_source")
            == "edit_authority_request"
        ):
            refs_by_type = {
                ref.artifact_type: ref
                for ref in upstream_refs
                if ref.artifact_type
                in {"edit_authority_request", "resume_report"}
            }
            for revision_field, upstream_type in (
                (
                    "edit_authority_request_revision",
                    "edit_authority_request",
                ),
                ("resume_report_revision", "resume_report"),
            ):
                revision = payload.get(revision_field)
                upstream = refs_by_type.get(upstream_type)
                if (
                    not isinstance(revision, Mapping)
                    or upstream is None
                    or revision.get("path") != upstream.path
                    or revision.get("artifact_hash")
                    != upstream.artifact_hash
                ):
                    raise ArtifactStoreError(
                        "EditAuthorityApproval payload revision does not "
                        f"match exact {upstream_type} upstream."
                    )
            request_ref = refs_by_type.get("edit_authority_request")
            resume_ref = refs_by_type.get("resume_report")
            if request_ref is None or resume_ref is None:
                raise ArtifactStoreError(
                    "EditAuthorityApproval requires its exact Request and "
                    "ResumeReport upstreams."
                )
            request_data = self.read(request_ref).get("data")
            resume_data = self.read(resume_ref).get("data")
            if not isinstance(request_data, Mapping):
                raise ArtifactStoreError(
                    "EditAuthorityApproval Request upstream data is invalid."
                )
            if not isinstance(resume_data, Mapping):
                raise ArtifactStoreError(
                    "EditAuthorityApproval ResumeReport upstream data is invalid."
                )
            try:
                validate_edit_authority_resume_report_against_approval(
                    approval_payload=payload,
                    request_artifact_payload=request_data,
                    resume_report_payload=resume_data,
                )
            except InvalidEditAuthorityApprovalIdentityError as exc:
                raise ArtifactStoreError(str(exc)) from exc
        exact_revision_fields_by_artifact = {
            "repair_planning_approval": (
                (
                    "repair_planning_authority_request_revision",
                    "repair_planning_request",
                ),
                ("resume_report_revision", "resume_report"),
            ),
            "root_cause_evidence": (
                (
                    "repair_planning_authority_request_revision",
                    "repair_planning_request",
                ),
                (
                    "repair_planning_authority_approval_revision",
                    "repair_planning_approval",
                ),
                ("binding_resolution_revision", "binding_resolution"),
            ),
            "repair_hypothesis_set": (
                (
                    "repair_planning_authority_approval_revision",
                    "repair_planning_approval",
                ),
                (
                    "root_cause_evidence_revision",
                    "root_cause_evidence",
                ),
                ("binding_resolution_revision", "binding_resolution"),
            ),
        }
        revision_fields = exact_revision_fields_by_artifact.get(
            artifact_type,
            (),
        )
        if revision_fields:
            refs_by_type = {
                ref.artifact_type: ref
                for ref in upstream_refs
                if ref.artifact_type
                in {upstream_type for _, upstream_type in revision_fields}
            }
            for revision_field, upstream_type in revision_fields:
                revision = payload.get(revision_field)
                upstream = refs_by_type.get(upstream_type)
                if (
                    not isinstance(revision, Mapping)
                    or upstream is None
                    or revision.get("path") != upstream.path
                    or revision.get("artifact_hash")
                    != upstream.artifact_hash
                ):
                    raise ArtifactStoreError(
                        f"{artifact_type} payload revision does not match "
                        f"exact {upstream_type} upstream."
                    )
        conditional_upstreams_required = _requires_conditional_upstream_contract(
            artifact_type=artifact_type,
            payload=payload,
            status=status,
        )
        one_of_upstreams_required = bool(
            contract.required_one_of_upstream_artifact_groups
        ) and (artifact_type != "repair_plan" or conditional_upstreams_required)
        if one_of_upstreams_required:
            for group in contract.required_one_of_upstream_artifact_groups:
                if not provided_types.intersection(group):
                    raise ArtifactStoreError(
                        f"Artifact {artifact_type!r} missing one of required upstream artifacts: {sorted(group)}"
                    )
        exact_upstreams_required = (
            artifact_type != "repair_plan"
            or conditional_upstreams_required
        )
        if exact_upstreams_required:
            missing_exact = sorted(
                contract.required_exact_revision_upstream_artifacts - provided_types
            )
            if missing_exact:
                raise ArtifactStoreError(
                    f"Artifact {artifact_type!r} missing required exact upstream artifacts: {missing_exact}"
                )
            not_exact = sorted(
                artifact_type
                for artifact_type in contract.required_exact_revision_upstream_artifacts
                if artifact_type not in exact_revision_upstream_types
            )
            if not_exact:
                raise ArtifactStoreError(
                    f"Artifact {artifact_type!r} requires exact revision upstream artifacts: {not_exact}"
                )

        for upstream in upstream_refs:
            freshness = self._check_upstream_write_freshness(
                upstream,
                require_latest=upstream.artifact_type not in exact_revision_upstream_types,
                exact_revision_types=exact_revision_upstream_types,
            )
            if not freshness.is_fresh and not allow_stale_upstreams:
                raise StaleArtifactError("; ".join(freshness.stale_reasons))

    def _check_upstream_write_freshness(
        self,
        ref: ArtifactRef,
        *,
        require_latest: bool,
        exact_revision_types: set[str],
    ) -> FreshnessView:
        if require_latest:
            return self._check_freshness(
                ref,
                recursive=False,
                exact_revision_types=exact_revision_types,
            )
        try:
            self.read(ref)
        except ArtifactStoreError as exc:
            reason = (
                f"{ref.artifact_type} hash changed"
                if str(exc) == "Artifact hash mismatch"
                else f"{ref.artifact_type} invalid: {exc}"
            )
            return FreshnessView(status="stale", stale_reasons=[reason])
        return FreshnessView(status="fresh", stale_reasons=[])

    def _guard_path(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactStoreError("Artifact path cannot escape artifact root") from exc
        return resolved

    @staticmethod
    def _stable_hash(value: Mapping[str, Any]) -> str:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _envelope_content_for_hash(envelope: Mapping[str, Any]) -> dict[str, Any]:
        return {field: envelope[field] for field in _ENVELOPE_HASH_FIELDS}

    @staticmethod
    def _validate_envelope_fields(envelope: Mapping[str, Any]) -> None:
        fields = set(envelope)
        missing = sorted(_ENVELOPE_FIELDS - fields)
        if missing:
            raise ArtifactStoreError(f"Artifact envelope missing fields: {missing}")
        unexpected = sorted(fields - _ENVELOPE_FIELDS)
        if unexpected:
            raise ArtifactStoreError(f"Artifact envelope unexpected fields: {unexpected}")

    @staticmethod
    def _append_stale_reason(stale_reasons: list[str], reason: str) -> None:
        if reason not in stale_reasons:
            stale_reasons.append(reason)

    @staticmethod
    def _validate_feature_id(feature_id: str) -> None:
        if not _SEGMENT_RE.match(feature_id) or ".." in feature_id:
            raise ArtifactStoreError(f"Invalid feature_id: {feature_id!r}")

    @staticmethod
    def _validate_artifact_type(artifact_type: str) -> None:
        if not _ARTIFACT_TYPE_RE.match(artifact_type):
            raise ArtifactStoreError(f"Invalid artifact_type: {artifact_type!r}")
