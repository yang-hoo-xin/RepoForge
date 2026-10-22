"""Artifact type registry and contract declarations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..edit_authority_capabilities import (
    InvalidEditAuthorityApprovalIdentityError,
    UnsupportedEditAuthorityCapabilityError,
    validate_edit_authority_approval_contract,
    validate_edit_authority_request_identity,
)


@dataclass(frozen=True)
class ArtifactContract:
    """Contract for a registered artifact type."""

    artifact_type: str
    schema_version: str
    required_fields: frozenset[str]
    producer_stage: str
    consumer_stages: frozenset[str]
    allowed_statuses: frozenset[str]
    required_upstream_artifacts: frozenset[str] = frozenset()
    required_exact_revision_upstream_artifacts: frozenset[str] = frozenset()
    required_one_of_upstream_artifact_groups: tuple[frozenset[str], ...] = ()
    dashboard_projection: tuple[str, ...] = ()
    eval_visibility: bool = True


class ArtifactRegistryError(ValueError):
    """Raised when an artifact contract is missing or violated."""


class ArtifactTypeRegistry:
    """Producer-consumer registry for artifact contracts."""

    def __init__(self, contracts: list[ArtifactContract] | None = None) -> None:
        self._contracts = {
            contract.artifact_type: contract
            for contract in (contracts or default_artifact_contracts())
        }

    def get(self, artifact_type: str) -> ArtifactContract:
        """Return the contract for an artifact type."""

        try:
            return self._contracts[artifact_type]
        except KeyError as exc:
            raise ArtifactRegistryError(f"Unregistered artifact_type: {artifact_type}") from exc

    def validate_write(
        self,
        *,
        artifact_type: str,
        status: str,
        payload: dict[str, Any],
    ) -> ArtifactContract:
        """Validate artifact type, status, and required payload fields."""

        contract = self.get(artifact_type)
        if status not in contract.allowed_statuses:
            raise ArtifactRegistryError(
                f"Status {status!r} is not allowed for {artifact_type!r}"
            )
        missing = sorted(field for field in contract.required_fields if field not in payload)
        if missing:
            raise ArtifactRegistryError(
                f"Artifact {artifact_type!r} missing required fields: {missing}"
            )
        if artifact_type == "edit_authority_request":
            _validate_edit_authority_request(payload, allow_legacy=False)
        if artifact_type == "edit_authority_approval":
            _validate_edit_authority_approval(payload, allow_legacy=False)
        self._validate_payload_contract(artifact_type, payload)
        return contract

    def validate_envelope(self, envelope: dict[str, Any]) -> ArtifactContract:
        """Validate envelope-level contract fields."""

        artifact_type = envelope.get("artifact_type")
        if not isinstance(artifact_type, str):
            raise ArtifactRegistryError("Artifact envelope missing artifact_type")
        contract = self.get(artifact_type)
        if envelope.get("schema_version") != contract.schema_version:
            raise ArtifactRegistryError(
                f"Unsupported schema_version for {artifact_type!r}: "
                f"{envelope.get('schema_version')!r}"
            )
        if envelope.get("producer_stage") != contract.producer_stage:
            raise ArtifactRegistryError(
                f"Invalid producer_stage for {artifact_type!r}: "
                f"{envelope.get('producer_stage')!r}"
            )
        status = envelope.get("status")
        if status not in contract.allowed_statuses:
            raise ArtifactRegistryError(
                f"Status {status!r} is not allowed for {artifact_type!r}"
            )
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise ArtifactRegistryError("Artifact envelope data must be a mapping")
        missing = sorted(field for field in contract.required_fields if field not in data)
        if missing:
            raise ArtifactRegistryError(
                f"Artifact {artifact_type!r} missing required fields: {missing}"
            )
        self._validate_payload_contract(artifact_type, data)
        return contract

    def validate_consumer(self, artifact_type: str, consumer_stage: str) -> None:
        """Validate that a stage may consume an artifact type."""

        contract = self.get(artifact_type)
        if consumer_stage not in contract.consumer_stages:
            raise ArtifactRegistryError(
                f"{consumer_stage!r} is not a registered consumer for {artifact_type!r}"
            )

    def artifact_types(self) -> frozenset[str]:
        """Return registered artifact type names."""

        return frozenset(self._contracts)

    def _validate_payload_contract(
        self,
        artifact_type: str,
        payload: dict[str, Any],
    ) -> None:
        if artifact_type == "isolated_apply_result" and (
            "authorization_origin" in payload or "task_delegation_ref" in payload
            or "task_delegation_resume_ref" in payload
        ):
            from ..execution.task_delegation import has_verified_apply_authorization

            try:
                has_verified_apply_authorization(payload)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise ArtifactRegistryError(f"Invalid isolated Apply authorization: {exc}") from exc
        if artifact_type == "implementation_plan":
            _validate_implementation_plan_binding_registry(payload)
        if artifact_type == "binding_resolution":
            _validate_binding_resolution(payload)
        if artifact_type == "edit_authority_request":
            _validate_edit_authority_request(
                payload,
                allow_legacy=True,
            )
        if artifact_type == "edit_authority_approval":
            _validate_edit_authority_approval(payload, allow_legacy=True)
        if artifact_type == "runtime_evidence":
            _validate_runtime_evidence_consolidation_binding(payload)
        if artifact_type == "evidence_pack":
            from ..evidence.models import EvidencePack

            if EvidencePack.requires_strict_semantic_rehydrate(payload):
                try:
                    EvidencePack.from_artifact_payload(payload)
                except ValueError as exc:
                    raise ArtifactRegistryError(
                        f"Invalid semantic EvidencePack: {exc}"
                    ) from exc
        if artifact_type == "repair_planning_request":
            _validate_repair_planning_authority_request(payload)
        if artifact_type == "repair_planning_approval":
            _validate_repair_planning_authority_approval(payload)
        if artifact_type == "root_cause_evidence":
            _validate_root_cause_evidence(payload)
        if artifact_type == "repair_hypothesis_set":
            _validate_repair_hypothesis_set(payload)
        if artifact_type == "evidence_run_attempt":
            _validate_evidence_run_attempt(payload)
        if artifact_type == "evidence_post_run_validation":
            _validate_evidence_post_run_validation(payload)
        if artifact_type == "task_memory":
            from ..planning.task_memory import validate_task_memory_payload

            try:
                validate_task_memory_payload(payload)
            except ValueError as exc:
                raise ArtifactRegistryError(str(exc)) from exc


def _validate_evidence_run_attempt(payload: dict[str, Any]) -> None:
    if payload.get("run_validity") not in {
        "attempt_started",
        "valid_run",
        "invalid_run",
    }:
        raise ArtifactRegistryError("evidence_run_attempt run_validity is invalid")
    for field_name in (
        "external_model_calls",
        "model_calls",
        "accepted_model_actions",
        "tool_calls_executed",
    ):
        value = payload.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ArtifactRegistryError(
                f"evidence_run_attempt {field_name} must be a non-negative integer"
            )
    if not isinstance(payload.get("provider_turns"), list):
        raise ArtifactRegistryError("evidence_run_attempt provider_turns must be a list")
    if not isinstance(payload.get("failure"), dict):
        raise ArtifactRegistryError("evidence_run_attempt failure must be an object")
    diagnostics = payload.get("failure_diagnostics")
    if diagnostics is not None:
        _validate_evidence_run_failure_diagnostics(diagnostics)

    lineage_schema = payload.get("lineage_schema_version")
    if lineage_schema is not None:
        if lineage_schema != "phase1_source_bound_no_write_v1":
            raise ArtifactRegistryError(
                "evidence_run_attempt lineage_schema_version is invalid"
            )
        for field_name in (
            "request_binding",
            "platform_source_binding",
            "target_source_binding",
            "authorization_boundary",
            "lineage",
        ):
            if not isinstance(payload.get(field_name), dict):
                raise ArtifactRegistryError(
                    f"evidence_run_attempt {field_name} must be an object"
                )


def _validate_binding_resolution(payload: dict[str, Any]) -> None:
    result_type = payload.get("result_type")
    if result_type not in {
        "resolved_repair_target",
        "binding_gap",
        "binding_resolution_blocked",
    }:
        raise ArtifactRegistryError(
            "binding_resolution result_type is invalid"
        )
    resolved_target = payload.get("resolved_repair_target")
    binding_gap = payload.get("binding_gap")
    proposal_request = payload.get("binding_proposal_request")
    blockers = payload.get("blockers")
    if not isinstance(blockers, list):
        raise ArtifactRegistryError(
            "binding_resolution blockers must be a list"
        )
    feature_id = payload.get("feature_id")
    run_id = payload.get("run_id")
    if (
        not isinstance(feature_id, str)
        or not feature_id
        or not isinstance(run_id, str)
        or not run_id
    ):
        raise ArtifactRegistryError(
            "binding_resolution feature_id and run_id must be non-empty"
        )
    for revision_field in (
        "runtime_evidence_revision",
        "implementation_plan_revision",
        "patch_proposal_revision",
    ):
        revision = payload.get(revision_field)
        if (
            not isinstance(revision, dict)
            or revision.get("feature_id") != feature_id
            or not isinstance(revision.get("path"), str)
            or not revision["path"]
            or not isinstance(revision.get("artifact_hash"), str)
            or not revision["artifact_hash"]
        ):
            raise ArtifactRegistryError(
                f"binding_resolution {revision_field} is not bound to "
                "the current feature"
            )
    if result_type == "resolved_repair_target":
        if (
            not isinstance(resolved_target, dict)
            or resolved_target.get("status") != "resolved"
            or resolved_target.get("feature_id") != feature_id
            or resolved_target.get("run_id") != run_id
            or resolved_target.get("implementation_plan_revision")
            != payload["implementation_plan_revision"]
            or not resolved_target.get("binding_id")
            or binding_gap is not None
            or proposal_request is not None
            or blockers
        ):
            raise ArtifactRegistryError(
                "resolved binding_resolution requires one resolved target "
                "and no gap, request, or blockers"
            )
        if payload.get("authority_source") == "edit_authority_approval":
            revision_fields = [
                "edit_authority_approval_revision",
                "evidence_pack_revision",
            ]
            authority_contract_version = payload.get(
                "authority_contract_version"
            )
            if authority_contract_version in {
                "edit_authority_request_approval_v1",
                "edit_authority_existing_binding_reauthorization_v1",
            }:
                revision_fields.append(
                    "edit_authority_request_revision"
                )
            if (
                authority_contract_version
                == "edit_authority_existing_binding_reauthorization_v1"
            ):
                revision_fields.extend(
                    [
                        "previous_binding_resolution_revision",
                        "predecessor_implementation_plan_revision",
                    ]
                )
            for revision_field in revision_fields:
                revision = payload.get(revision_field)
                if (
                    not isinstance(revision, dict)
                    or revision.get("feature_id") != feature_id
                    or not isinstance(revision.get("path"), str)
                    or not revision["path"]
                    or not isinstance(
                        revision.get("artifact_hash"),
                        str,
                    )
                    or not revision["artifact_hash"]
                ):
                    raise ArtifactRegistryError(
                        "approved binding_resolution is missing an exact "
                        f"{revision_field}"
                    )
            if authority_contract_version not in {
                None,
                "edit_authority_request_approval_v1",
                "edit_authority_existing_binding_reauthorization_v1",
            }:
                raise ArtifactRegistryError(
                    "approved binding_resolution authority contract is invalid"
                )
            ownership = payload.get(
                "approved_runtime_failure_ownership"
            )
            if (
                not isinstance(ownership, dict)
                or ownership.get("owner") != "product_behavior"
                or ownership.get("source_repair_allowed") is not True
                or payload.get("allowed_edit_binding_ids")
                != [resolved_target["binding_id"]]
                or payload.get("repair_planner_allowed") is not False
                or payload.get("source_patch_allowed") is not False
            ):
                raise ArtifactRegistryError(
                    "approved binding_resolution authority boundary is invalid"
                )
        return
    if resolved_target is not None or not blockers:
        raise ArtifactRegistryError(
            "blocked binding_resolution requires no resolved target and "
            "at least one blocker"
        )
    if result_type == "binding_gap":
        if (
            not isinstance(binding_gap, dict)
            or binding_gap.get("gap") is not True
            or not isinstance(binding_gap.get("gap_reasons"), list)
            or not binding_gap["gap_reasons"]
            or any(
                not isinstance(reason, str) or not reason
                for reason in binding_gap["gap_reasons"]
            )
            or len(set(binding_gap["gap_reasons"]))
            != len(binding_gap["gap_reasons"])
            or not isinstance(proposal_request, dict)
            or proposal_request.get("status") != "requested"
            or proposal_request.get("approval_required") is not True
            or proposal_request.get("registry_mutation_allowed") is not False
            or proposal_request.get("repair_allowed") is not False
            or proposal_request.get("required_plan_lineage")
            != payload["implementation_plan_revision"]
        ):
            raise ArtifactRegistryError(
                "binding_gap requires a non-authorizing binding proposal request"
            )
        return
    if binding_gap is not None or proposal_request is not None:
        raise ArtifactRegistryError(
            "invalid binding inputs cannot produce a gap or proposal request"
        )


def _validate_evidence_post_run_validation(payload: dict[str, Any]) -> None:
    if payload.get("lineage_schema_version") != "phase1_source_bound_no_write_v1":
        raise ArtifactRegistryError(
            "evidence_post_run_validation lineage_schema_version is invalid"
        )
    checks = payload.get("lineage_checks")
    if not isinstance(checks, dict) or not checks:
        raise ArtifactRegistryError(
            "evidence_post_run_validation lineage_checks must be a non-empty object"
        )
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ArtifactRegistryError(
            "evidence_post_run_validation lineage_checks values must be boolean"
        )
    lineage_passed = payload.get("lineage_passed")
    if not isinstance(lineage_passed, bool) or lineage_passed != all(checks.values()):
        raise ArtifactRegistryError(
            "evidence_post_run_validation lineage_passed must equal all lineage checks"
        )
    if payload.get("affects_runtime_route") is not False:
        raise ArtifactRegistryError(
            "evidence_post_run_validation affects_runtime_route must be false"
        )
    if not isinstance(payload.get("runtime_outcome"), dict):
        raise ArtifactRegistryError(
            "evidence_post_run_validation runtime_outcome must be an object"
        )
    domain_validation = payload.get("domain_validation")
    if domain_validation is not None and not isinstance(domain_validation, dict):
        raise ArtifactRegistryError(
            "evidence_post_run_validation domain_validation must be null or an object"
        )
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) or not item for item in limitations
    ):
        raise ArtifactRegistryError(
            "evidence_post_run_validation limitations must be a list of non-empty strings"
        )


def _validate_evidence_run_failure_diagnostics(value: object) -> None:
    if not isinstance(value, dict):
        raise ArtifactRegistryError(
            "evidence_run_attempt failure_diagnostics must be an object"
        )
    if value.get("schema_version") != "evidence_run_failure_diagnostics_v1":
        raise ArtifactRegistryError(
            "evidence_run_attempt failure_diagnostics schema_version is invalid"
        )
    for field_name in (
        "provider_tool_calls",
        "adapter_accepted_actions",
        "deterministic_validation_passed",
        "dispatch_started",
        "tool_execution_completed",
        "observation_created",
    ):
        field_value = value.get(field_name)
        if (
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
            or field_value < 0
        ):
            raise ArtifactRegistryError(
                f"evidence_run_attempt failure_diagnostics {field_name} "
                "must be a non-negative integer"
            )
    for field_name in ("failure_stage", "symbolic_failure_code"):
        field_value = value.get(field_name)
        if not isinstance(field_value, str) or not field_value:
            raise ArtifactRegistryError(
                f"evidence_run_attempt failure_diagnostics {field_name} "
                "must be a non-empty string"
            )
    traceback_data = value.get("traceback")
    if not isinstance(traceback_data, dict):
        raise ArtifactRegistryError(
            "evidence_run_attempt failure_diagnostics traceback must be an object"
        )
    for field_name in (
        "available_in_process",
        "recoverable_from_artifact",
        "source_lines_persisted",
        "locals_persisted",
        "exception_message_persisted",
    ):
        if not isinstance(traceback_data.get(field_name), bool):
            raise ArtifactRegistryError(
                f"evidence_run_attempt failure_diagnostics traceback {field_name} "
                "must be boolean"
            )
    external_frame_count = traceback_data.get("external_frame_count")
    if (
        not isinstance(external_frame_count, int)
        or isinstance(external_frame_count, bool)
        or external_frame_count < 0
    ):
        raise ArtifactRegistryError(
            "evidence_run_attempt failure_diagnostics traceback "
            "external_frame_count must be a non-negative integer"
        )
    frames = traceback_data.get("project_frames")
    if not isinstance(frames, list):
        raise ArtifactRegistryError(
            "evidence_run_attempt failure_diagnostics traceback project_frames "
            "must be a list"
        )
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"path", "line", "function"}:
            raise ArtifactRegistryError(
                "evidence_run_attempt failure_diagnostics traceback frame is invalid"
            )
        path = frame.get("path")
        line = frame.get("line")
        function = frame.get("function")
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or path.startswith("/")
            or ":" in path
            or ".." in path.split("/")
            or not isinstance(line, int)
            or isinstance(line, bool)
            or line <= 0
            or not isinstance(function, str)
            or not function
        ):
            raise ArtifactRegistryError(
                "evidence_run_attempt failure_diagnostics traceback frame is invalid"
            )


def _validate_implementation_plan_binding_registry(payload: dict[str, Any]) -> None:
    if payload.get("blocking") is True:
        return
    registry = payload.get("binding_registry")
    if not isinstance(registry, dict):
        raise ArtifactRegistryError(
            "Non-blocking implementation_plan requires binding_registry"
        )
    if registry.get("schema_version") != "binding_registry_v1":
        raise ArtifactRegistryError(
            "Non-blocking implementation_plan binding_registry schema_version must be binding_registry_v1"
        )
    edit_bindings = registry.get("edit_bindings")
    if not isinstance(edit_bindings, list) or not edit_bindings:
        raise ArtifactRegistryError(
            "Non-blocking implementation_plan requires at least one edit binding"
        )
    edit_binding_ids = {
        str(binding.get("binding_id"))
        for binding in edit_bindings
        if isinstance(binding, dict) and binding.get("binding_id")
    }
    if not edit_binding_ids:
        raise ArtifactRegistryError(
            "Non-blocking implementation_plan edit bindings require binding_id"
        )
    proposed_changes = payload.get("proposed_changes")
    if not isinstance(proposed_changes, list) or not proposed_changes:
        raise ArtifactRegistryError(
            "Non-blocking implementation_plan requires proposed_changes"
        )
    for change in proposed_changes:
        if not isinstance(change, dict):
            raise ArtifactRegistryError(
                "Non-blocking implementation_plan proposed_changes must be mappings"
            )
        binding_id = change.get("edit_binding_id")
        if not binding_id:
            raise ArtifactRegistryError(
                "Non-blocking implementation_plan proposed_changes require edit_binding_id"
            )
        if str(binding_id) not in edit_binding_ids:
            raise ArtifactRegistryError(
                "Non-blocking implementation_plan proposed_changes edit_binding_id must resolve in binding_registry"
            )
    diagnostic_scope = payload.get("diagnostic_scope")
    if diagnostic_scope is None:
        return
    if (
        isinstance(diagnostic_scope, dict)
        and diagnostic_scope.get("schema_version")
        == "stagger_hud_parent_diagnostic_execution_authority_v2"
    ):
        _validate_stagger_diagnostic_only_plan(
            payload=payload,
            diagnostic_scope=diagnostic_scope,
            edit_bindings=edit_bindings,
            proposed_changes=proposed_changes,
        )
        return
    if (
        not isinstance(diagnostic_scope, dict)
        or diagnostic_scope.get("schema_version")
        != "stagger_hud_parent_observation_authority_v1"
        or diagnostic_scope.get("root_cause_status") != "unresolved"
        or diagnostic_scope.get("observation_only") is not True
        or diagnostic_scope.get("repair_planner_allowed") is not False
        or diagnostic_scope.get("source_patch_allowed") is not False
        or diagnostic_scope.get("product_acceptance_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "Stagger observation diagnostic_scope boundary is invalid"
        )
    diagnostic_binding_ids = diagnostic_scope.get("edit_binding_ids")
    approval_revision = diagnostic_scope.get(
        "edit_authority_approval_revision"
    )
    if (
        not isinstance(diagnostic_binding_ids, list)
        or len(diagnostic_binding_ids) != 2
        or any(
            not isinstance(binding_id, str) or not binding_id
            for binding_id in diagnostic_binding_ids
        )
        or len(set(diagnostic_binding_ids)) != 2
        or not isinstance(approval_revision, dict)
        or approval_revision.get("feature_id") != payload.get("feature_id")
        or not isinstance(approval_revision.get("path"), str)
        or not approval_revision["path"]
        or not isinstance(approval_revision.get("artifact_hash"), str)
        or not approval_revision["artifact_hash"]
    ):
        raise ArtifactRegistryError(
            "Stagger observation diagnostic binding authority is invalid"
        )
    exact_bindings = [
        binding
        for binding in edit_bindings
        if isinstance(binding, dict)
        and binding.get("binding_id") in diagnostic_binding_ids
    ]
    exact_changes = [
        change
        for change in proposed_changes
        if isinstance(change, dict)
        and change.get("edit_binding_id") in diagnostic_binding_ids
    ]
    if (
        len(exact_bindings) != 2
        or len(exact_changes) != 2
        or {
            binding.get("binding_id")
            for binding in exact_bindings
        }
        != set(diagnostic_binding_ids)
        or {
            change.get("edit_binding_id")
            for change in exact_changes
        }
        != set(diagnostic_binding_ids)
    ):
        raise ArtifactRegistryError(
            "Stagger observation Plan does not contain exactly two bound "
            "diagnostic changes"
        )
    allowed_operations = {
        "insert_observation_only_hud_parent_selection_state_after_existing_assignment",
        "insert_observation_only_hud_parent_use_state_before_existing_cast_and_guard",
    }
    if any(
        binding.get("usage") != "observation_only_edit_target_scope"
        or binding.get("edit_authority_approval_ref") != approval_revision
        or binding.get("repair_planner_allowed") is not False
        or binding.get("source_patch_allowed") is not False
        or not isinstance(binding.get("observation_contract"), dict)
        for binding in exact_bindings
    ) or any(
        change.get("change_type")
        != "add_observation_only_runtime_diagnostic"
        or change.get("operation") not in allowed_operations
        for change in exact_changes
    ):
        raise ArtifactRegistryError(
            "Stagger observation binding or change exceeds its approved scope"
        )


def _validate_stagger_diagnostic_only_plan(
    *,
    payload: dict[str, Any],
    diagnostic_scope: dict[str, Any],
    edit_bindings: list[Any],
    proposed_changes: list[Any],
) -> None:
    selected_binding_ids = diagnostic_scope.get("selected_binding_ids")
    selected_change_ids = diagnostic_scope.get("selected_change_ids")
    approval_revision = diagnostic_scope.get(
        "edit_authority_approval_revision"
    )
    if (
        diagnostic_scope.get("root_cause_status") != "unresolved"
        or diagnostic_scope.get("observation_only") is not True
        or diagnostic_scope.get(
            "new_plan_contains_only_selected_bindings"
        )
        is not True
        or diagnostic_scope.get(
            "new_plan_contains_only_selected_changes"
        )
        is not True
        or diagnostic_scope.get("repair_planner_allowed") is not False
        or diagnostic_scope.get("source_patch_allowed") is not False
        or diagnostic_scope.get("product_acceptance_allowed") is not False
        or not isinstance(selected_binding_ids, list)
        or len(selected_binding_ids) != 4
        or len(set(selected_binding_ids)) != 4
        or not isinstance(selected_change_ids, list)
        or len(selected_change_ids) != 4
        or len(set(selected_change_ids)) != 4
        or not isinstance(approval_revision, dict)
        or approval_revision.get("feature_id") != payload.get("feature_id")
    ):
        raise ArtifactRegistryError(
            "Stagger diagnostic-only Plan boundary is invalid"
        )
    exact_bindings = [
        item for item in edit_bindings if isinstance(item, dict)
    ]
    exact_changes = [
        item for item in proposed_changes if isinstance(item, dict)
    ]
    if (
        len(exact_bindings) != 4
        or len(exact_changes) != 4
        or {
            item.get("binding_id") for item in exact_bindings
        }
        != set(selected_binding_ids)
        or {
            item.get("change_id") for item in exact_changes
        }
        != set(selected_change_ids)
        or {
            item.get("edit_binding_id") for item in exact_changes
        }
        != set(selected_binding_ids)
    ):
        raise ArtifactRegistryError(
            "Stagger diagnostic-only Plan contains unselected changes"
        )
    excluded_binding_ids = diagnostic_scope.get(
        "excluded_predecessor_binding_ids"
    )
    excluded_change_ids = diagnostic_scope.get(
        "excluded_predecessor_change_ids"
    )
    if (
        not isinstance(excluded_binding_ids, list)
        or not isinstance(excluded_change_ids, list)
        or set(selected_binding_ids).intersection(excluded_binding_ids)
        or set(selected_change_ids).intersection(excluded_change_ids)
    ):
        raise ArtifactRegistryError(
            "Stagger diagnostic-only Plan predecessor exclusion is invalid"
        )
    allowed_operations = {
        (
            "insert_observation_only_hud_parent_selection_state_"
            "after_existing_assignment"
        ),
        (
            "insert_observation_only_hud_parent_use_state_"
            "before_existing_cast_and_guard"
        ),
        "insert_bounded_observation_state_after_existing_hud_parent_field",
        (
            "insert_observation_identity_helpers_before_existing_"
            "hud_parent_resolver"
        ),
    }
    if {
        item.get("operation") for item in exact_changes
    } != allowed_operations or any(
        binding.get("usage") != "observation_only_edit_target_scope"
        or binding.get("edit_authority_approval_ref") != approval_revision
        or binding.get("repair_planner_allowed") is not False
        or binding.get("source_patch_allowed") is not False
        or not isinstance(binding.get("observation_contract"), dict)
        for binding in exact_bindings
    ) or any(
        change.get("change_type")
        != "add_observation_only_runtime_diagnostic"
        for change in exact_changes
    ):
        raise ArtifactRegistryError(
            "Stagger diagnostic-only binding contract is invalid"
        )
    test_plan = payload.get("test_plan")
    if (
        not isinstance(test_plan, dict)
        or test_plan.get("schema_version")
        != "stagger_hud_parent_diagnostic_test_plan_v1"
        or test_plan.get("product_acceptance_allowed") is not False
        or test_plan.get("repair_planner_allowed") is not False
        or test_plan.get("required_signal")
        != "STAGGER_HUD_PARENT_DIAGNOSTIC"
    ):
        raise ArtifactRegistryError(
            "Stagger diagnostic-only test plan is invalid"
        )


def _validate_edit_authority_request(
    payload: dict[str, Any],
    *,
    allow_legacy: bool,
) -> None:
    if payload.get("status") != "pending_human_approval":
        raise ArtifactRegistryError(
            "edit_authority_request status must be pending_human_approval"
        )
    request_hash = payload.get("request_hash")
    request_payload = payload.get("request_payload")
    if (
        not isinstance(request_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", request_hash) is None
        or not isinstance(request_payload, dict)
    ):
        raise ArtifactRegistryError(
            "edit_authority_request exact request binding is invalid"
        )
    try:
        validate_edit_authority_request_identity(
            payload,
            allow_legacy=allow_legacy,
        )
    except (
        InvalidEditAuthorityApprovalIdentityError,
        UnsupportedEditAuthorityCapabilityError,
    ) as exc:
        raise ArtifactRegistryError(str(exc)) from exc
    if (
        payload.get("authority_granted") is not False
        or payload.get("registry_mutation_allowed") is not False
        or payload.get("repair_planner_allowed") is not False
        or payload.get("source_patch_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "edit_authority_request cannot grant execution authority"
        )


def _validate_repair_planning_authority_request(
    payload: dict[str, Any],
) -> None:
    if payload.get("status") != "pending_human_approval":
        raise ArtifactRegistryError(
            "repair_planning_authority_request status must be "
            "pending_human_approval"
        )
    request_hash = payload.get("request_hash")
    request_payload = payload.get("request_payload")
    if (
        not isinstance(request_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", request_hash) is None
        or not isinstance(request_payload, dict)
    ):
        raise ArtifactRegistryError(
            "repair_planning_authority_request exact request binding is invalid"
        )
    computed_hash = "sha256:" + hashlib.sha256(
        json.dumps(
            request_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if computed_hash != request_hash:
        raise ArtifactRegistryError(
            "repair_planning_authority_request request hash mismatch"
        )
    if (
        request_payload.get("schema_version")
        != "stagger_repair_planning_authority_request_v1"
        or request_payload.get("feature_id") != payload.get("feature_id")
        or request_payload.get("planning_run_id") != payload.get("run_id")
        or not isinstance(request_payload.get("preprojection_hash"), str)
        or not isinstance(request_payload.get("formal_lineage"), dict)
        or not isinstance(
            request_payload.get("root_cause_evidence_core"),
            dict,
        )
        or not isinstance(
            request_payload.get("repair_hypothesis_set_core"),
            dict,
        )
        or request_payload.get("requested_actions")
        != [
            "materialize_root_cause_evidence",
            "materialize_repair_hypothesis_set",
            "invoke_repair_planner_for_semantic_decision",
            "materialize_repair_plan",
        ]
        or request_payload.get("forbidden_actions")
        != [
            "modify_source",
            "build",
            "launch_process",
            "write_game_directory",
            "claim_product_acceptance",
            "claim_audio_audibility",
        ]
    ):
        raise ArtifactRegistryError(
            "repair_planning_authority_request payload boundary is invalid"
        )
    if (
        payload.get("authority_granted") is not False
        or payload.get("artifact_materialization_allowed") is not False
        or payload.get("repair_planner_allowed") is not False
        or payload.get("source_patch_allowed") is not False
        or payload.get("build_allowed") is not False
        or payload.get("runtime_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "repair_planning_authority_request cannot grant execution authority"
        )


def _validate_repair_planning_authority_approval(
    payload: dict[str, Any],
) -> None:
    if payload.get("status") != "approved":
        raise ArtifactRegistryError(
            "repair_planning_authority_approval status must be approved"
        )
    for field_name in (
        "request_hash",
        "approval_hash",
    ):
        value = payload.get(field_name)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
        ):
            raise ArtifactRegistryError(
                "repair_planning_authority_approval exact hash is invalid"
            )
    if payload["approval_hash"] != payload["request_hash"]:
        raise ArtifactRegistryError(
            "repair_planning_authority_approval must approve the exact request hash"
        )
    if (
        not isinstance(payload.get("approved_by"), str)
        or not payload["approved_by"].strip()
        or not isinstance(payload.get("approved_at"), str)
        or not payload["approved_at"].strip()
    ):
        raise ArtifactRegistryError(
            "repair_planning_authority_approval human identity is incomplete"
        )
    for revision_field in (
        "repair_planning_authority_request_revision",
        "resume_report_revision",
    ):
        _validate_exact_revision(
            payload.get(revision_field),
            feature_id=payload.get("feature_id"),
            field_name=revision_field,
        )
    if (
        payload.get("allowed_actions")
        != [
            "materialize_root_cause_evidence",
            "materialize_repair_hypothesis_set",
            "invoke_repair_planner_for_semantic_decision",
            "materialize_repair_plan",
        ]
        or payload.get("forbidden_actions")
        != [
            "modify_source",
            "build",
            "launch_process",
            "write_game_directory",
            "claim_product_acceptance",
            "claim_audio_audibility",
        ]
        or payload.get("artifact_materialization_allowed") is not True
        or payload.get("repair_planner_allowed") is not True
        or payload.get("source_patch_allowed") is not False
        or payload.get("build_allowed") is not False
        or payload.get("runtime_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "repair_planning_authority_approval exceeds its planning boundary"
        )


def _validate_root_cause_evidence(payload: dict[str, Any]) -> None:
    if (
        payload.get("schema_version")
        != "stagger_root_cause_evidence_v1"
        or payload.get("failure_owner") != "product_behavior"
        or payload.get("direct_failure_mechanism") != "confirmed"
        or not isinstance(payload.get("confirmed_mechanism"), dict)
        or payload.get("product_acceptance") != "not_evaluated"
        or payload.get("audio_audibility") != "not_evaluated"
        or payload.get("yellow_x_repaired") is not False
        or payload.get("formal_runtime_evidence") is not False
        or payload.get("source_patch_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "root_cause_evidence claim boundary is invalid"
        )
    if (
        not isinstance(payload.get("failed_feature_spec_criterion"), str)
        or not payload["failed_feature_spec_criterion"]
        or not isinstance(payload.get("evidence_items"), list)
        or not payload["evidence_items"]
        or not isinstance(payload.get("evidence_gaps"), list)
        or not payload["evidence_gaps"]
        or not isinstance(payload.get("immutable_limitations"), list)
        or not payload["immutable_limitations"]
        or not isinstance(payload.get("unresolved"), list)
        or not payload["unresolved"]
    ):
        raise ArtifactRegistryError(
            "root_cause_evidence evidence or limitation set is incomplete"
        )
    if any(
        not isinstance(item, dict)
        or item.get("formal_runtime_evidence") is not False
        or item.get("provenance_class")
        not in {
            "post_run_copy",
            "derived_summary",
            "runner_output_limit_projection",
            "run_local_output",
            "isolated_materialization",
            "isolated_build_output",
            "isolated_runtime_input",
        }
        for item in payload["evidence_items"]
    ):
        raise ArtifactRegistryError(
            "root_cause_evidence provenance classification is invalid"
        )
    if any(
        not isinstance(item, dict)
        or item.get("status") != "missing"
        or item.get("recoverable") is not False
        for item in payload["evidence_gaps"]
    ):
        raise ArtifactRegistryError(
            "root_cause_evidence missing evidence must remain an explicit gap"
        )
    for revision_field in (
        "repair_planning_authority_request_revision",
        "repair_planning_authority_approval_revision",
        "binding_resolution_revision",
    ):
        _validate_exact_revision(
            payload.get(revision_field),
            feature_id=payload.get("feature_id"),
            field_name=revision_field,
        )


def _validate_repair_hypothesis_set(payload: dict[str, Any]) -> None:
    if (
        payload.get("schema_version")
        != "stagger_repair_hypothesis_set_v1"
        or payload.get("selection_status")
        not in {"unresolved", "selected"}
        or payload.get("source_patch_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "repair_hypothesis_set boundary is invalid"
        )
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ArtifactRegistryError(
            "repair_hypothesis_set requires hypotheses"
        )
    required_fields = {
        "hypothesis_id",
        "statement",
        "proposed_edit_location",
        "supporting_evidence",
        "unverified_assumptions",
        "binding_status",
        "required_observation",
        "status",
    }
    identifiers: list[str] = []
    for item in hypotheses:
        if (
            not isinstance(item, dict)
            or not required_fields.issubset(item)
            or item.get("status")
            not in {"unverified", "supported", "rejected"}
            or item.get("binding_status")
            not in {
                "covered_by_current_binding",
                "outside_current_binding",
                "requires_exact_member_evidence",
                "requires_semantic_decision",
            }
            or not isinstance(item.get("supporting_evidence"), list)
            or not isinstance(item.get("unverified_assumptions"), list)
        ):
            raise ArtifactRegistryError(
                "repair_hypothesis_set hypothesis entry is invalid"
            )
        identifiers.append(str(item["hypothesis_id"]))
    if len(identifiers) != len(set(identifiers)):
        raise ArtifactRegistryError(
            "repair_hypothesis_set hypothesis ids must be unique"
        )
    supported_count = sum(
        1
        for item in hypotheses
        if item.get("status") == "supported"
    )
    if (
        payload["selection_status"] == "unresolved"
        and supported_count != 0
    ) or (
        payload["selection_status"] == "selected"
        and supported_count != 1
    ):
        raise ArtifactRegistryError(
            "repair_hypothesis_set selection status does not match "
            "supported hypotheses"
        )
    for revision_field in (
        "repair_planning_authority_approval_revision",
        "root_cause_evidence_revision",
        "binding_resolution_revision",
    ):
        _validate_exact_revision(
            payload.get(revision_field),
            feature_id=payload.get("feature_id"),
            field_name=revision_field,
        )


def _validate_exact_revision(
    revision: Any,
    *,
    feature_id: Any,
    field_name: str,
) -> None:
    if (
        not isinstance(feature_id, str)
        or not feature_id
        or not isinstance(revision, dict)
        or revision.get("feature_id") != feature_id
        or not isinstance(revision.get("path"), str)
        or not revision["path"]
        or not isinstance(revision.get("artifact_hash"), str)
        or not revision["artifact_hash"]
    ):
        raise ArtifactRegistryError(
            f"{field_name} is not an exact revision for the current feature"
        )


def _validate_edit_authority_approval(
    payload: dict[str, Any],
    *,
    allow_legacy: bool,
) -> None:
    if payload.get("status") != "approved":
        raise ArtifactRegistryError(
            "edit_authority_approval status must be approved"
        )
    if payload.get("authority_source") == "edit_authority_request":
        _validate_request_bound_edit_authority_approval(
            payload,
            allow_legacy=allow_legacy,
        )
        return
    if not allow_legacy:
        raise ArtifactRegistryError(
            "New Edit Authority Approval writes require versioned_v1 identity."
        )
    approval_hash = payload.get("approval_hash")
    if (
        not isinstance(approval_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", approval_hash) is None
    ):
        raise ArtifactRegistryError(
            "edit_authority_approval approval_hash is invalid"
        )
    proposal = payload.get("approved_proposal")
    if not isinstance(proposal, dict):
        raise ArtifactRegistryError(
            "edit_authority_approval approved_proposal must be a mapping"
        )
    computed_hash = "sha256:" + hashlib.sha256(
        json.dumps(
            proposal,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if computed_hash != approval_hash:
        raise ArtifactRegistryError(
            "edit_authority_approval proposal hash mismatch"
        )
    if not isinstance(payload.get("approved_by"), str) or not payload[
        "approved_by"
    ].strip():
        raise ArtifactRegistryError(
            "edit_authority_approval approved_by must be non-empty"
        )
    if payload.get("approval_scope") != [
        "runtime_failure_ownership_reclassification",
        "new_exact_edit_binding",
    ]:
        raise ArtifactRegistryError(
            "edit_authority_approval approval_scope is invalid"
        )
    stored_ownership = payload.get("stored_runtime_failure_ownership")
    approved_ownership = payload.get("approved_runtime_failure_ownership")
    if (
        not isinstance(stored_ownership, dict)
        or stored_ownership.get("owner") != "observation"
        or stored_ownership.get("source_repair_allowed") is not False
        or not isinstance(approved_ownership, dict)
        or approved_ownership.get("owner") != "product_behavior"
        or approved_ownership.get("source_repair_allowed") is not True
    ):
        raise ArtifactRegistryError(
            "edit_authority_approval ownership boundary is invalid"
        )
    binding_ids = payload.get("allowed_edit_binding_ids")
    evidence_ids = payload.get("allowed_evidence_ids")
    if (
        not isinstance(binding_ids, list)
        or len(binding_ids) != 1
        or not isinstance(binding_ids[0], str)
        or not binding_ids[0]
        or not isinstance(evidence_ids, list)
        or len(evidence_ids) != 1
        or not isinstance(evidence_ids[0], str)
        or not evidence_ids[0]
    ):
        raise ArtifactRegistryError(
            "edit_authority_approval requires one binding and evidence id"
        )
    if (
        payload.get("runtime_evidence_mutation_allowed") is not False
        or payload.get("old_patch_proposal_mutation_allowed") is not False
        or payload.get("repair_planner_allowed") is not False
        or payload.get("source_patch_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "edit_authority_approval exceeds its mutation boundary"
        )


def _validate_request_bound_edit_authority_approval(
    payload: dict[str, Any],
    *,
    allow_legacy: bool,
) -> None:
    allowed_fields = {
        "node_name",
        "stage",
        "message",
        "status",
        "feature_id",
        "run_id",
        "authority_source",
        "authority_contract_version",
        "approval_identity_mode",
        "approval_identity_schema_version",
        "capability_id",
        "capability_contract_version",
        "approval_hash",
        "request_hash",
        "approved_by",
        "approved_at",
        "resume_id",
        "decision",
        "edit_authority_request_revision",
        "resume_report_revision",
        "allowed_actions",
        "forbidden_actions",
        "repair_planner_allowed",
        "source_patch_allowed",
    }
    unexpected_fields = sorted(set(payload) - allowed_fields)
    if unexpected_fields:
        raise ArtifactRegistryError(
            "edit_authority_approval contains non-authority fields: "
            f"{unexpected_fields}"
        )
    approval_hash = payload.get("approval_hash")
    request_hash = payload.get("request_hash")
    request_revision = payload.get("edit_authority_request_revision")
    resume_revision = payload.get("resume_report_revision")
    if (
        not isinstance(approval_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", approval_hash) is None
        or approval_hash != request_hash
        or not isinstance(request_revision, dict)
        or request_revision.get("feature_id") != payload.get("feature_id")
        or not isinstance(request_revision.get("path"), str)
        or not request_revision["path"]
        or not isinstance(request_revision.get("artifact_hash"), str)
        or not request_revision["artifact_hash"]
        or not isinstance(resume_revision, dict)
        or resume_revision.get("feature_id") != payload.get("feature_id")
        or not isinstance(resume_revision.get("path"), str)
        or not resume_revision["path"]
        or not isinstance(resume_revision.get("artifact_hash"), str)
        or not resume_revision["artifact_hash"]
    ):
        raise ArtifactRegistryError(
            "edit_authority_approval request and resume binding is invalid"
        )
    if (
        payload.get("decision") != "approve"
        or not isinstance(payload.get("resume_id"), str)
        or not payload["resume_id"]
        or not isinstance(payload.get("approved_by"), str)
        or not payload["approved_by"].strip()
        or not isinstance(payload.get("approved_at"), str)
        or not payload["approved_at"].strip()
    ):
        raise ArtifactRegistryError(
            "edit_authority_approval human decision identity is invalid"
        )
    try:
        validate_edit_authority_approval_contract(
            payload,
            allow_legacy=allow_legacy,
        )
    except InvalidEditAuthorityApprovalIdentityError as exc:
        raise ArtifactRegistryError(str(exc)) from exc
    if (
        payload.get("repair_planner_allowed") is not False
        or payload.get("source_patch_allowed") is not False
    ):
        raise ArtifactRegistryError(
            "edit_authority_approval exceeds its approved action boundary"
        )


def _validate_runtime_evidence_consolidation_binding(
    payload: dict[str, Any],
) -> None:
    """Validate only the registered semantic binding discriminator."""

    if payload.get("verification_only_evidence_consolidation") is None:
        return
    from src.mod_agent.artifact_semantics import (
        require_registered_semantic_binding,
    )

    try:
        require_registered_semantic_binding("runtime_evidence", payload)
    except ValueError as exc:
        raise ArtifactRegistryError(str(exc)) from exc


def default_artifact_contracts() -> list[ArtifactContract]:
    """Return the first controlled workflow artifact contracts."""

    statuses = frozenset({"success", "blocked", "failed", "stubbed", "skipped"})
    return [
        ArtifactContract(
            artifact_type="request_planning_report",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "user_request",
                    "route",
                    "risk_level",
                    "requires_hook_evidence",
                    "requires_runtime_evidence",
                    "uncertainties",
                    "blocked",
                }
            ),
            producer_stage="request_planning",
            consumer_stages=frozenset(
                {"planning_brief", "draft_feature_spec", "blocked", "dashboard"}
            ),
            allowed_statuses=statuses,
            dashboard_projection=("message", "route", "risk_level", "blocked"),
        ),
        ArtifactContract(
            artifact_type="mod_planning_brief",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "decision",
                    "user_intent",
                    "request_type",
                    "knowledge_query_plan",
                    "retrieved_experience",
                    "planning_options",
                    "selected_plan",
                    "handoff",
                    "verification_boundary",
                    "next_action",
                    "blocked",
                    "blockers",
                }
            ),
            producer_stage="planning_brief",
            consumer_stages=frozenset(
                {"draft_feature_spec", "retrieve_evidence", "dashboard"}
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"request_planning_report"}),
            dashboard_projection=("message", "decision", "request_type", "next_action"),
        ),
        ArtifactContract(
            artifact_type="feature_spec",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "user_request",
                    "target_behavior",
                    "non_goals",
                    "target_mod",
                    "risk_level",
                    "acceptance_criteria",
                    "evidence_requirements",
                    "runtime_verification_plan",
                    "blocked",
                    "blockers",
                }
            ),
            producer_stage="feature_spec",
            consumer_stages=frozenset(
                {
                    "retrieve_evidence",
                    "create_implementation_plan",
                    "build",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            dashboard_projection=("message", "target_behavior", "risk_level", "blocked"),
        ),
        ArtifactContract(
            artifact_type="evidence_run_attempt",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "run_id",
                    "request_id",
                    "run_validity",
                    "terminal_stage",
                    "model_execution",
                    "model_calls",
                    "external_model_calls",
                    "accepted_model_actions",
                    "tool_calls_executed",
                    "provider_turns",
                    "failure_attribution",
                    "failure",
                }
            ),
            producer_stage="evidence_retrieval",
            consumer_stages=frozenset({"dashboard", "eval"}),
            allowed_statuses=frozenset({"running", "success", "failed"}),
            dashboard_projection=(
                "message",
                "run_id",
                "run_validity",
                "terminal_stage",
                "model_calls",
                "external_model_calls",
                "accepted_model_actions",
                "tool_calls_executed",
                "failure_attribution",
            ),
        ),
        ArtifactContract(
            artifact_type="evidence_post_run_validation",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "run_id",
                    "request_id",
                    "lineage_schema_version",
                    "lineage_checks",
                    "lineage_passed",
                    "runtime_outcome",
                    "domain_validation",
                    "affects_runtime_route",
                    "limitations",
                }
            ),
            producer_stage="post_run_validation",
            consumer_stages=frozenset({"dashboard", "eval"}),
            allowed_statuses=frozenset({"success", "blocked", "failed"}),
            required_upstream_artifacts=frozenset(
                {"evidence_pack", "workflow_trace"}
            ),
            dashboard_projection=(
                "message",
                "run_id",
                "lineage_passed",
            ),
            eval_visibility=True,
        ),
        ArtifactContract(
            artifact_type="evidence_pack",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "query",
                    "readiness",
                    "can_generate_patch",
                    "authority",
                    "authoritative_for_patch",
                    "identity_schema_version",
                    "identity_field_mapping",
                    "items",
                    "gaps",
                    "excluded",
                    "blockers",
                    "retrieval_methods_used",
                    "missing_evidence",
                    "disabled_modes",
                    "timing",
                    "trace_refs",
                }
            ),
            producer_stage="evidence_retrieval",
            consumer_stages=frozenset(
                {
                    "create_implementation_plan",
                    "generate_patch_proposal",
                    "build",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"feature_spec"}),
            dashboard_projection=(
                "message",
                "readiness",
                "can_generate_patch",
                "authority",
                "authoritative_for_patch",
                "budget_summary",
            ),
        ),
        ArtifactContract(
            artifact_type="implementation_plan",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "evidence_source_paths",
                    "target_files",
                    "target_symbols",
                    "evidence_refs",
                    "proposed_changes",
                    "side_effect_boundaries",
                    "test_plan",
                    "rollback_plan",
                    "open_questions",
                    "blocking",
                    "blockers",
                }
            ),
            producer_stage="implementation_plan",
            consumer_stages=frozenset(
                {
                    "generate_patch_proposal",
                    "binding_resolution",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"feature_spec", "evidence_pack"}),
            dashboard_projection=("message", "target_files", "blocking"),
        ),
        ArtifactContract(
            artifact_type="patch_proposal",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "proposal_id",
                    "feature_id",
                    "summary",
                    "files",
                    "diff_intent",
                    "change_units",
                    "context_pack",
                    "evidence_refs",
                    "scope_boundary",
                    "risk_assessment",
                    "expected_tests",
                    "no_op_status",
                    "rollback_hints",
                    "manual_review_notes",
                    "blocked",
                    "blockers",
                }
            ),
            producer_stage="patch_proposal",
            consumer_stages=frozenset(
                {
                    "review_proposal",
                    "dry_run_apply",
                    "binding_resolution",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"implementation_plan", "evidence_pack"}),
            dashboard_projection=("message", "files", "blocked"),
        ),
        ArtifactContract(
            artifact_type="critic_assessment",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "assessor_id",
                    "assessor_kind",
                    "independence_boundary",
                    "subject_type",
                    "subject_artifact_ref",
                    "subject_artifact_hash",
                    "score_scale",
                    "dimension_scores",
                    "total_score",
                    "policy_threshold",
                    "policy_version",
                    "applicability",
                    "aggregation_method",
                    "dimension_weights",
                    "verdict",
                    "findings",
                    "limitations",
                }
            ),
            producer_stage="proposal_review",
            consumer_stages=frozenset({"review_proposal", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {
                    "feature_spec",
                    "evidence_pack",
                    "implementation_plan",
                    "patch_proposal",
                }
            ),
            required_exact_revision_upstream_artifacts=frozenset({"patch_proposal"}),
            dashboard_projection=(
                "message",
                "assessor_kind",
                "total_score",
                "verdict",
            ),
        ),
        ArtifactContract(
            artifact_type="proposal_review",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "decision",
                    "findings",
                    "required_changes",
                    "allowed_next_stage",
                    "checked_rules",
                    "trace_metrics",
                }
            ),
            producer_stage="proposal_review",
            consumer_stages=frozenset({"dry_run_apply", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"patch_proposal"}),
            dashboard_projection=("message", "decision", "allowed_next_stage"),
        ),
        ArtifactContract(
            artifact_type="dry_run_result",
            schema_version="1.0",
            required_fields=frozenset({"node_name", "stage", "message"}),
            producer_stage="dry_run_apply",
            consumer_stages=frozenset({"build", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"patch_proposal", "proposal_review"}),
            dashboard_projection=("message",),
        ),
        ArtifactContract(
            artifact_type="isolated_apply_result",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "execution_mode",
                    "actual_write_performed",
                    "real_patch_applied",
                    "approved_by_review",
                    "approved_by_user",
                    "changed_files",
                }
            ),
            producer_stage="isolated_apply",
            consumer_stages=frozenset({"build", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {"patch_proposal", "proposal_review", "dry_run_result"}
            ),
            dashboard_projection=("message", "real_patch_applied", "changed_files"),
        ),
        ArtifactContract(
            artifact_type="build_result",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "execution_mode",
                    "status",
                    "command",
                    "exit_code",
                    "stdout_path",
                    "stderr_path",
                    "errors",
                    "warnings",
                    "duration_ms",
                    "error_class",
                    "diagnostics",
                    "repair_handoff",
                    "expected_artifacts",
                    "verified_artifacts",
                    "artifact_verification",
                    "real_build_performed",
                    "claims_real_build_success",
                }
            ),
            producer_stage="build",
            consumer_stages=frozenset(
                {"collect_runtime_evidence", "evaluate", "repair_plan", "dashboard"}
            ),
            allowed_statuses=statuses,
            required_one_of_upstream_artifact_groups=(
                frozenset({"dry_run_result", "feature_spec"}),
                frozenset({"dry_run_result", "evidence_pack"}),
            ),
            dashboard_projection=("message", "status", "error_class"),
        ),
        ArtifactContract(
            artifact_type="runtime_evidence",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "run_id",
                    "build_artifact_ref",
                    "build_artifact_hash",
                    "execution_mode",
                    "evidence_source",
                    "plugin_marker_binding",
                    "screenshot_refs",
                    "runtime_trust_binding",
                    "startup_loaded",
                    "gameplay_behavior_verified",
                    "evidence_items",
                    "verification_status",
                    "acceptance_status",
                    "acceptance_verified",
                    "acceptance_contract",
                    "raw_supplied_output",
                    "sources",
                    "non_goals_checked",
                    "non_goal_preservation",
                    "limitations",
                    "real_runtime_performed",
                    "claims_real_runtime_success",
                }
            ),
            producer_stage="runtime_evidence",
            consumer_stages=frozenset(
                {
                    "evaluate",
                    "binding_resolution",
                    "repair_plan",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"build_result"}),
            dashboard_projection=("message",),
        ),
        ArtifactContract(
            artifact_type="edit_authority_request",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "status",
                    "feature_id",
                    "run_id",
                    "request_hash",
                    "request_payload",
                    "authority_granted",
                    "registry_mutation_allowed",
                    "repair_planner_allowed",
                    "source_patch_allowed",
                }
            ),
            producer_stage="edit_authority_request",
            consumer_stages=frozenset(
                {
                    "edit_authority_approval",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {
                    "feature_spec",
                    "runtime_evidence",
                    "implementation_plan",
                    "patch_proposal",
                }
            ),
            required_exact_revision_upstream_artifacts=frozenset(
                {
                    "feature_spec",
                    "runtime_evidence",
                    "implementation_plan",
                    "patch_proposal",
                }
            ),
            dashboard_projection=(
                "message",
                "status",
                "request_hash",
                "authority_granted",
            ),
        ),
        ArtifactContract(
            artifact_type="edit_authority_approval",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "status",
                    "feature_id",
                    "run_id",
                    "approval_hash",
                    "approved_by",
                    "repair_planner_allowed",
                    "source_patch_allowed",
                }
            ),
            producer_stage="edit_authority_approval",
            consumer_stages=frozenset(
                {
                    "evidence_retrieval",
                    "implementation_plan",
                    "binding_resolution",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {
                    "edit_authority_request",
                    "resume_report",
                }
            ),
            required_exact_revision_upstream_artifacts=frozenset(
                {
                    "edit_authority_request",
                    "resume_report",
                }
            ),
            dashboard_projection=(
                "message",
                "status",
                "approval_hash",
                "repair_planner_allowed",
                "source_patch_allowed",
            ),
        ),
        ArtifactContract(
            artifact_type="repair_planning_request",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "status",
                    "feature_id",
                    "run_id",
                    "request_hash",
                    "request_payload",
                    "authority_granted",
                    "artifact_materialization_allowed",
                    "repair_planner_allowed",
                    "source_patch_allowed",
                    "build_allowed",
                    "runtime_allowed",
                }
            ),
            producer_stage="repair_planning_authority_request",
            consumer_stages=frozenset(
                {
                    "repair_planning_approval",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {"binding_resolution"}
            ),
            required_exact_revision_upstream_artifacts=frozenset(
                {"binding_resolution"}
            ),
            dashboard_projection=(
                "message",
                "status",
                "request_hash",
                "authority_granted",
            ),
        ),
        ArtifactContract(
            artifact_type="repair_planning_approval",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "status",
                    "feature_id",
                    "run_id",
                    "request_hash",
                    "approval_hash",
                    "approved_by",
                    "approved_at",
                    "repair_planning_authority_request_revision",
                    "resume_report_revision",
                    "allowed_actions",
                    "forbidden_actions",
                    "artifact_materialization_allowed",
                    "repair_planner_allowed",
                    "source_patch_allowed",
                    "build_allowed",
                    "runtime_allowed",
                }
            ),
            producer_stage="repair_planning_approval",
            consumer_stages=frozenset(
                {
                    "root_cause_evidence",
                    "repair_hypothesis_set",
                    "repair_plan",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {
                    "repair_planning_request",
                    "resume_report",
                }
            ),
            required_exact_revision_upstream_artifacts=frozenset(
                {
                    "repair_planning_request",
                    "resume_report",
                }
            ),
            dashboard_projection=(
                "message",
                "status",
                "approval_hash",
                "repair_planner_allowed",
            ),
        ),
        ArtifactContract(
            artifact_type="root_cause_evidence",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "schema_version",
                    "feature_id",
                    "planning_run_id",
                    "failed_feature_spec_criterion",
                    "failure_owner",
                    "direct_failure_mechanism",
                    "confirmed_mechanism",
                    "excluded_for_observed_event",
                    "unresolved",
                    "product_acceptance",
                    "audio_audibility",
                    "yellow_x_repaired",
                    "formal_runtime_evidence",
                    "evidence_items",
                    "evidence_gaps",
                    "immutable_limitations",
                    "formal_lineage",
                    "product_identity",
                    "diagnostic_identity",
                    "runtime_result_identity",
                    "binding_identity",
                    "preprojection_hash",
                    "repair_planning_authority_request_revision",
                    "repair_planning_authority_approval_revision",
                    "binding_resolution_revision",
                    "source_patch_allowed",
                }
            ),
            producer_stage="root_cause_evidence",
            consumer_stages=frozenset(
                {
                    "repair_hypothesis_set",
                    "repair_plan",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {
                    "repair_planning_request",
                    "repair_planning_approval",
                    "binding_resolution",
                }
            ),
            required_exact_revision_upstream_artifacts=frozenset(
                {
                    "repair_planning_request",
                    "repair_planning_approval",
                    "binding_resolution",
                }
            ),
            dashboard_projection=(
                "message",
                "failure_owner",
                "direct_failure_mechanism",
                "product_acceptance",
            ),
        ),
        ArtifactContract(
            artifact_type="repair_hypothesis_set",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "schema_version",
                    "feature_id",
                    "planning_run_id",
                    "selection_status",
                    "hypotheses",
                    "preprojection_hash",
                    "repair_planning_authority_approval_revision",
                    "root_cause_evidence_revision",
                    "binding_resolution_revision",
                    "source_patch_allowed",
                }
            ),
            producer_stage="repair_hypothesis_set",
            consumer_stages=frozenset(
                {
                    "repair_plan",
                    "dashboard",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {
                    "repair_planning_approval",
                    "root_cause_evidence",
                    "binding_resolution",
                }
            ),
            required_exact_revision_upstream_artifacts=frozenset(
                {
                    "repair_planning_approval",
                    "root_cause_evidence",
                    "binding_resolution",
                }
            ),
            dashboard_projection=(
                "message",
                "selection_status",
            ),
        ),
        ArtifactContract(
            artifact_type="manual_evidence",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "run_id",
                    "build_artifact_ref",
                    "build_artifact_hash",
                    "execution_mode",
                    "evidence_source",
                    "plugin_marker_binding",
                    "screenshot_refs",
                    "runtime_trust_binding",
                    "startup_loaded",
                    "gameplay_behavior_verified",
                    "evidence_items",
                    "verification_status",
                    "acceptance_status",
                    "acceptance_verified",
                    "acceptance_contract",
                    "raw_supplied_output",
                    "sources",
                    "non_goals_checked",
                    "non_goal_preservation",
                    "limitations",
                    "real_runtime_performed",
                    "claims_real_runtime_success",
                }
            ),
            producer_stage="runtime_evidence",
            consumer_stages=frozenset({"evaluate", "repair_plan", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"build_result"}),
            dashboard_projection=("message", "verification_status", "acceptance_status"),
        ),
        ArtifactContract(
            artifact_type="screenshot_registry",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "screenshot_count",
                    "captioned_count",
                    "missing_caption_count",
                    "screenshots",
                    "verification_status",
                    "acceptance_status",
                    "claims_real_runtime_success",
                    "limitations",
                }
            ),
            producer_stage="runtime_evidence",
            consumer_stages=frozenset({"evaluate", "repair_plan", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"build_result"}),
            dashboard_projection=(
                "message",
                "screenshot_count",
                "captioned_count",
                "verification_status",
                "acceptance_status",
            ),
        ),
        ArtifactContract(
            artifact_type="binding_resolution",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "feature_id",
                    "run_id",
                    "result_type",
                    "runtime_evidence_revision",
                    "implementation_plan_revision",
                    "patch_proposal_revision",
                    "allowed_edit_binding_ids",
                    "resolved_repair_target",
                    "binding_gap",
                    "binding_proposal_request",
                    "blockers",
                }
            ),
            producer_stage="binding_resolution",
            consumer_stages=frozenset({"repair_plan", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset(
                {
                    "runtime_evidence",
                    "implementation_plan",
                    "patch_proposal",
                }
            ),
            dashboard_projection=(
                "message",
                "result_type",
                "binding_gap",
                "binding_proposal_request",
            ),
        ),
        ArtifactContract(
            artifact_type="repair_plan",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "repair_id",
                    "feature_id",
                    "failure_source",
                    "failure_class",
                    "failure_evidence",
                    "evidence_refs",
                    "root_cause_hypotheses",
                    "hypothesis",
                    "recommended_fix_strategy",
                    "minimal_changes",
                    "retry_count",
                    "retry_budget",
                    "retry_budget_remaining",
                    "retry_exhausted",
                    "scope_expansion_allowed",
                    "stop_condition",
                    "forbidden_regressions",
                    "requires_new_evidence",
                    "allowed_next_stage",
                    "blockers",
                }
            ),
            producer_stage="repair_plan",
            consumer_stages=frozenset({"generate_patch_proposal", "dashboard"}),
            allowed_statuses=statuses,
            required_exact_revision_upstream_artifacts=frozenset({"patch_proposal"}),
            required_one_of_upstream_artifact_groups=(
                frozenset(
                    {
                        "build_result",
                        "runtime_evidence",
                        "proposal_review",
                        "root_cause_evidence",
                    }
                ),
            ),
            dashboard_projection=(
                "message",
                "failure_source",
                "failure_class",
                "retry_budget_remaining",
            ),
        ),
        ArtifactContract(
            artifact_type="eval_report",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "execution_mode",
                    "workflow_completion_status",
                    "requested_behavior_status",
                    "verdict",
                    "claims_real_success",
                    "input_acceptance_status",
                    "input_acceptance_verified",
                    "acceptance_contract",
                    "evaluation_gate",
                    "v1_acceptance_status",
                    "v1_acceptance_blockers",
                }
            ),
            producer_stage="evaluation",
            consumer_stages=frozenset({"done", "dashboard"}),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"build_result"}),
            dashboard_projection=(
                "message",
                "workflow_completion_status",
                "requested_behavior_status",
            ),
        ),
        # 为什么要有它：策略与执行来源沿用正常ArtifactStore，通用消费者也必须识别其不可变原件。
        ArtifactContract(
            artifact_type="strategy_experience", schema_version="1.0",
            required_fields=frozenset({"kind", "schema_version"}),
            producer_stage="strategy_experience",
            consumer_stages=frozenset({"strategy_experience", "task_agent", "evaluation", "dashboard"}),
            allowed_statuses=frozenset({"success"}), eval_visibility=False,
        ),
        ArtifactContract(
            artifact_type="task_memory",
            schema_version="1.0",
            required_fields=frozenset({
                "node_name", "stage", "message", "schema_version", "authority",
                "lesson_id", "revision", "previous_revision_ref", "request",
                "evidence_refs", "facts", "outcome", "applicability", "unknowns",
                "invalidation_conditions", "expires_at",
            }),
            producer_stage="task_memory",
            consumer_stages=frozenset({
                "task_memory", "planning_brief", "implementation_plan", "dashboard",
            }),
            allowed_statuses=frozenset({"success"}),
            dashboard_projection=("lesson_id", "revision", "authority", "outcome"),
            eval_visibility=False,
        ),
        ArtifactContract(
            artifact_type="workflow_trace",
            schema_version="1.0",
            required_fields=frozenset({"node_name", "stage", "message"}),
            producer_stage="workflow",
            consumer_stages=frozenset({"dashboard", "eval"}),
            allowed_statuses=statuses,
            eval_visibility=True,
        ),
        ArtifactContract(
            artifact_type="workflow_checkpoint",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "checkpoint_id",
                    "run_id",
                    "feature_id",
                    "request_id",
                    "checkpoint_stage",
                    "status",
                    "state_snapshot",
                    "artifact_refs",
                    "artifact_hashes",
                    "resume_binding",
                    "resume_allowed_fields",
                    "interrupt_reason",
                    "checkpoint_policy",
                }
            ),
            producer_stage="checkpoint",
            consumer_stages=frozenset({"resume", "dashboard", "eval"}),
            allowed_statuses=statuses,
            dashboard_projection=(
                "message",
                "checkpoint_id",
                "checkpoint_stage",
                "status",
            ),
            eval_visibility=True,
        ),
        ArtifactContract(
            artifact_type="resume_report",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "resume_id",
                    "checkpoint_id",
                    "run_id",
                    "resume_stage",
                    "decision",
                    "status",
                    "allowed",
                    "reason",
                    "applied_updates",
                    "stale_reasons",
                    "blockers",
                }
            ),
            producer_stage="resume",
            consumer_stages=frozenset(
                {
                    "edit_authority_approval",
                    "repair_planning_approval",
                    "dashboard",
                    "eval",
                }
            ),
            allowed_statuses=statuses,
            required_upstream_artifacts=frozenset({"workflow_checkpoint"}),
            required_exact_revision_upstream_artifacts=frozenset(
                {"workflow_checkpoint"}
            ),
            dashboard_projection=(
                "message",
                "resume_id",
                "resume_stage",
                "allowed",
            ),
            eval_visibility=True,
        ),
        ArtifactContract(
            artifact_type="job_status",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "job_id",
                    "job_type",
                    "status",
                    "created_at",
                    "started_at",
                    "finished_at",
                    "artifact_refs",
                    "trace_refs",
                    "error",
                }
            ),
            producer_stage="mcp_job",
            consumer_stages=frozenset({"dashboard", "eval", "mcp"}),
            allowed_statuses=statuses,
            dashboard_projection=(
                "message",
                "job_id",
                "job_type",
                "status",
                "artifact_refs",
                "budget_summary",
            ),
            eval_visibility=True,
        ),
        ArtifactContract(
            artifact_type="git_change_record",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "record_id",
                    "base_revision",
                    "commit_sha",
                    "changed_files",
                    "diff_hash",
                    "commit_diff_hash",
                    "working_tree",
                    "proposal_binding",
                    "dirty_tree_policy",
                    "replay_command",
                    "rollback_hint",
                }
            ),
            producer_stage="git_audit",
            consumer_stages=frozenset({"dashboard", "eval", "repair_plan"}),
            allowed_statuses=statuses,
            dashboard_projection=(
                "message",
                "base_revision",
                "commit_sha",
                "changed_files",
                "diff_hash",
                "dirty_tree_policy",
                "rollback_hint",
            ),
            eval_visibility=True,
        ),
        ArtifactContract(
            artifact_type="knowledge_index_report",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "source_id",
                    "total_files",
                    "indexed_files",
                    "skipped_files",
                    "failed_files",
                    "symbols_indexed",
                }
            ),
            producer_stage="knowledge_index",
            consumer_stages=frozenset({"retrieve_evidence", "dashboard", "eval"}),
            allowed_statuses=statuses,
            dashboard_projection=(
                "message",
                "source_id",
                "total_files",
                "indexed_files",
                "skipped_files",
                "symbols_indexed",
            ),
        ),
        ArtifactContract(
            artifact_type="workflow_summary",
            schema_version="1.0",
            required_fields=frozenset(
                {
                    "node_name",
                    "stage",
                    "message",
                    "execution_mode",
                    "workflow_completion_status",
                    "requested_behavior_status",
                    "final_status",
                    "claims_real_mod_success",
                    "claims_real_build_success",
                    "claims_real_runtime_success",
                    "v1_acceptance_status",
                    "v1_acceptance_blockers",
                }
            ),
            producer_stage="done",
            consumer_stages=frozenset({"dashboard", "eval"}),
            allowed_statuses=statuses,
            dashboard_projection=(
                "message",
                "workflow_completion_status",
                "requested_behavior_status",
            ),
        ),
        ArtifactContract(
            artifact_type="blocked_report",
            schema_version="1.0",
            required_fields=frozenset({"node_name", "stage", "message"}),
            producer_stage="blocked",
            consumer_stages=frozenset({"dashboard", "eval"}),
            allowed_statuses=statuses,
            dashboard_projection=("message",),
        ),
    ]
