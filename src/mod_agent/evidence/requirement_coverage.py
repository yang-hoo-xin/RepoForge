"""Generic requirement-to-evidence coverage for ProductExecution planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .identity import feature_spec_payload_sha256, request_relation_identity
from .models import (
    DETERMINISTIC_EVIDENCE_AUTHORITY,
    EvidenceItem,
    EvidencePack,
)

LEGACY_REQUIREMENT_COVERAGE_SCHEMA_VERSION = "evidence_requirement_coverage_v1"
PREVIOUS_REQUIREMENT_COVERAGE_SCHEMA_VERSION = "evidence_requirement_coverage_v2"
REQUIREMENT_COVERAGE_SCHEMA_VERSION = "evidence_requirement_coverage_v3"
_SUPPORTED_REQUIREMENT_COVERAGE_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_REQUIREMENT_COVERAGE_SCHEMA_VERSION,
        PREVIOUS_REQUIREMENT_COVERAGE_SCHEMA_VERSION,
        REQUIREMENT_COVERAGE_SCHEMA_VERSION,
    }
)
REQUIREMENT_ROLES = frozenset(
    {"change", "pre_patch_evidence", "preservation", "post_change_acceptance"}
)
REQUIREMENT_RELATIONSHIPS = frozenset({"support", "conflict", "missing"})
REQUIREMENT_COVERAGE_DOES_NOT_GRANT = (
    "human_approval",
    "apply_permission",
    "build_permission",
    "runtime_acceptance",
    "final_correctness",
)
REQUIREMENT_COVERAGE_TRUST_BOUNDARY = (
    "model_originated_semantic_support_conflict_finding",
    "deterministic_requirement_evidence_identity_validation",
    "deterministic_current_source_reference_validation",
    "deterministic_input_accounting_and_coverage_projection",
    "deterministic_sufficiency_edit_target_patch_authority_derivation",
)
LEGACY_REQUIREMENT_COVERAGE_V3_TRUST_BOUNDARY = (
    "model_originated_semantic_support_conflict_and_edit_target_usage_proposal",
    "deterministic_requirement_evidence_identity_validation",
    "deterministic_current_source_reference_validation",
    "deterministic_input_accounting_and_coverage_projection",
    "deterministic_sufficiency_edit_target_patch_authority_derivation",
)
REQUIREMENT_COVERAGE_V3_TRUST_BOUNDARY = (
    "model_originated_semantic_support_conflict_and_edit_target_usage_hint",
    "deterministic_requirement_evidence_identity_validation",
    "deterministic_current_source_reference_validation",
    "deterministic_input_accounting_and_coverage_projection",
    "deterministic_coverage_sufficiency_without_edit_target_authority",
)
_FEATURE_SPEC_DERIVED_FIELDS = frozenset(
    {"feature_spec_sha256", "input_obligations"}
)


class RequirementCoverageProjectionError(ValueError):
    """Legacy-compatible projection rejection with bounded exact diagnostics."""

    def __init__(self, code: str, diagnostic: Mapping[str, Any]) -> None:
        self.code = code
        self.diagnostic = dict(diagnostic)
        super().__init__(code)


def is_exact_edit_target_evidence(item: EvidenceItem) -> bool:
    """Existing physical identity condition for an edit hint, not write authority."""
    span = item.span
    return (
        item.authority == DETERMINISTIC_EVIDENCE_AUTHORITY
        and item.source_type == "target_source"
        and bool(item.source_path)
        and isinstance(span, dict)
        and type(span.get("start_line")) is int
        and type(span.get("end_line")) is int
        and span["start_line"] >= 1
        and span["end_line"] >= span["start_line"]
        and bool(item.source_hash or item.file_hash or item.doc_hash)
    )


@dataclass(frozen=True)
class RequirementCoverageView:
    """Stable consumer view over one strictly validated coverage receipt."""

    feature_id: str
    feature_spec_sha256: str
    change_support_evidence_ids: frozenset[str]
    edit_target_hint_evidence_ids: frozenset[str]
    preservation_requirement_texts: tuple[str, ...]


def requirement_input_obligation_model_view(
    feature_data: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Expose indexed obligations without granting stable identity authority."""

    return [
        {
            "input_obligation_index": index,
            "origin": dict(obligation["origin"]),
            "text": obligation["text"],
            "allowed_roles": list(obligation["allowed_roles"]),
            "allowed_dispositions": list(
                obligation["allowed_dispositions"]
            ),
            "required_before_patch_policy": obligation[
                "required_before_patch"
            ],
        }
        for index, obligation in enumerate(_input_obligations(feature_data), 1)
    ]


def validated_requirement_coverage_view(
    evidence_pack_payload: Mapping[str, Any],
) -> RequirementCoverageView | None:
    """Return consumer facts without exposing raw semantic representation."""

    if "requirement_coverage" not in evidence_pack_payload:
        return None
    pack = EvidencePack.from_artifact_payload(evidence_pack_payload)
    section = pack.requirement_coverage
    if not isinstance(section, dict):
        raise ValueError("REQUIREMENT_COVERAGE_SECTION_REQUIRED")
    requirements = section["requirements"]
    findings = section["findings"]
    edit_target_hint_ids = frozenset(
        str(value) for value in section["edit_target_evidence_ids"]
    )
    change_requirement_ids = {
        str(requirement["requirement_id"])
        for requirement in requirements
        if requirement["role"] == "change"
    }
    change_related_evidence_ids = {
        str(evidence_id)
        for finding in findings
        if finding["relationship"] == "support"
        and change_requirement_ids.intersection(
            str(value) for value in finding["parent_requirement_ids"]
        )
        for evidence_id in finding["evidence_ids"]
    }
    return RequirementCoverageView(
        feature_id=str(section["feature_spec_identity"]["feature_id"]),
        feature_spec_sha256=str(
            section["feature_spec_identity"]["feature_spec_sha256"]
        ),
        change_support_evidence_ids=frozenset(change_related_evidence_ids),
        edit_target_hint_evidence_ids=edit_target_hint_ids,
        preservation_requirement_texts=tuple(
            str(requirement["text"])
            for requirement in requirements
            if requirement["role"] == "preservation"
        ),
    )


def default_feature_requirement_projection(
    feature_data: Mapping[str, Any],
) -> dict[str, object]:
    """Project exact FeatureSpec fields without inferring evidence support."""

    obligations = _input_obligations(feature_data)
    requirements: list[dict[str, object]] = []
    change_indexes = [
        index
        for index, obligation in enumerate(obligations, 1)
        if "change" in obligation["allowed_roles"]
    ]
    requirements.append(
        _projected_requirement(
            role="change",
            text=_required_string(feature_data, "target_behavior"),
            input_obligation_indexes=change_indexes,
        )
    )
    for index, obligation in enumerate(obligations, 1):
        if "change" in obligation["allowed_roles"]:
            continue
        requirements.append(
            _projected_requirement(
                role=str(obligation["allowed_roles"][0]),
                text=str(obligation["text"]),
                input_obligation_indexes=[index],
            )
        )
    if len(requirements) > 16:
        raise ValueError("REQUIREMENT_COVERAGE_REQUIREMENT_BUDGET_EXCEEDED")
    return {
        "requirements": requirements,
        "relations": [
            {
                "parent_requirement_indexes": [index],
                "meaning": f"Evidence disposition for requirement {index}.",
            }
            for index in range(1, len(requirements) + 1)
        ],
        "input_dispositions": [],
    }


def materialize_requirement_coverage_projection(
    *,
    feature_data: Mapping[str, Any],
    request: str,
    projection: Mapping[str, Any],
) -> dict[str, object]:
    """Assign deterministic requirement, relation, and query identities."""

    _reject_feature_spec_derived_fields(feature_data)
    return _materialize_requirement_coverage_projection(
        feature_id=_required_string(feature_data, "feature_id"),
        feature_spec_sha256=feature_spec_payload_sha256(feature_data),
        input_obligations=_input_obligations(feature_data),
        request=request,
        projection=projection,
        schema_version=REQUIREMENT_COVERAGE_SCHEMA_VERSION,
    )


def _materialize_requirement_coverage_projection(
    *,
    feature_id: str,
    feature_spec_sha256: str,
    input_obligations: list[dict[str, object]],
    request: str,
    projection: Mapping[str, Any],
    schema_version: str,
) -> dict[str, object]:
    """Materialize from identity inputs owned by deterministic code."""

    if schema_version not in _SUPPORTED_REQUIREMENT_COVERAGE_SCHEMA_VERSIONS:
        raise ValueError("REQUIREMENT_COVERAGE_SCHEMA_VERSION_INVALID")
    if set(projection) != {"requirements", "relations", "input_dispositions"}:
        raise ValueError("REQUIREMENT_COVERAGE_PROJECTION_SHAPE_INVALID")
    raw_requirements = projection.get("requirements")
    raw_relations = projection.get("relations")
    if (
        not isinstance(raw_requirements, list)
        or not 1 <= len(raw_requirements) <= 16
        or not isinstance(raw_relations, list)
        or not 1 <= len(raw_relations) <= 32
    ):
        raise ValueError("REQUIREMENT_COVERAGE_PROJECTION_BUDGET_INVALID")
    request_sha256 = hashlib.sha256(request.strip().encode("utf-8")).hexdigest()
    normalized_projection = _normalized_projection(
        raw_requirements=raw_requirements,
        raw_relations=raw_relations,
        raw_dispositions=projection.get("input_dispositions"),
        input_obligations=input_obligations,
    )
    projection_sha256 = _json_sha256(normalized_projection)
    requirements: list[dict[str, object]] = []
    queries: list[dict[str, str]] = []
    for index, raw in enumerate(normalized_projection["requirements"], 1):
        requirement_id = _stable_id(
            "requirement",
            feature_spec_sha256,
            projection_sha256,
            str(index),
            str(raw["role"]),
            str(raw["text"]),
            *[str(value) for value in raw["input_obligation_indexes"]],
        )
        obligation_rows = [
            input_obligations[int(value) - 1]
            for value in raw["input_obligation_indexes"]
        ]
        required_before_patch = any(
            value["required_before_patch"] is True for value in obligation_rows
        )
        query_ids: list[str] = []
        for query_text in raw["query_leads"]:
            query_id = _stable_id("query", requirement_id, str(query_text))
            query_ids.append(query_id)
            queries.append(
                {
                    "query_id": query_id,
                    "query": str(query_text),
                    "requirement_id": requirement_id,
                }
            )
        requirements.append(
            {
                "requirement_id": requirement_id,
                "role": raw["role"],
                "text": raw["text"],
                "required_before_patch": required_before_patch,
                "input_obligation_ids": [
                    value["input_obligation_id"] for value in obligation_rows
                ],
                "query_leads": list(raw["query_leads"]),
                "query_ids": query_ids,
            }
        )
    relations: list[dict[str, object]] = []
    seen_relation_keys: set[tuple[tuple[int, ...], str]] = set()
    for index, raw in enumerate(normalized_projection["relations"], 1):
        parent_indexes = tuple(int(value) for value in raw["parent_requirement_indexes"])
        relation_key = (parent_indexes, str(raw["meaning"]))
        if relation_key in seen_relation_keys:
            raise ValueError("REQUIREMENT_COVERAGE_RELATION_DUPLICATE")
        seen_relation_keys.add(relation_key)
        relations.append(
            {
                "relation_id": request_relation_identity(
                    request_sha256=request_sha256,
                    projection_sha256=projection_sha256,
                    relation_index=index,
                ),
                "parent_requirement_ids": [
                    requirements[parent - 1]["requirement_id"]
                    for parent in parent_indexes
                ],
                "meaning": raw["meaning"],
                "provenance": {"relation_index": index},
            }
        )
    covered_requirement_ids = {
        str(parent)
        for relation in relations
        for parent in relation["parent_requirement_ids"]
    }
    if covered_requirement_ids != {
        str(requirement["requirement_id"]) for requirement in requirements
    }:
        raise ValueError("REQUIREMENT_COVERAGE_RELATION_COVERAGE_INVALID")
    return {
        "schema_version": schema_version,
        "feature_spec_identity": {
            "feature_id": feature_id,
            "feature_spec_sha256": feature_spec_sha256,
        },
        "request_identity": {"request_sha256": request_sha256},
        "projection_identity": {"projection_sha256": projection_sha256},
        "input_obligations": input_obligations,
        "input_dispositions": [
            {
                "input_obligation_id": input_obligations[
                    int(value["input_obligation_index"]) - 1
                ]["input_obligation_id"],
                "disposition": value["disposition"],
                "reason": value["reason"],
            }
            for value in normalized_projection["input_dispositions"]
        ],
        "requirements": requirements,
        "relations": relations,
        "queries": queries,
    }


def validate_requirement_coverage_evidence_pack(pack: EvidencePack) -> None:
    """Recompute coverage sufficiency and strictly read legacy semantics."""

    section = pack.requirement_coverage
    if not isinstance(section, dict):
        raise ValueError("REQUIREMENT_COVERAGE_SECTION_REQUIRED")
    materialized = {
        key: section[key]
        for key in (
            "schema_version",
            "feature_spec_identity",
            "request_identity",
            "projection_identity",
            "input_obligations",
            "input_dispositions",
            "requirements",
            "relations",
            "queries",
        )
        if key in section
    }
    relation_index_by_id = {
        str(relation["relation_id"]): int(
            relation["provenance"]["relation_index"]
        )
        for relation in section.get("relations", [])
        if isinstance(relation, dict)
        and isinstance(relation.get("provenance"), dict)
    }
    persisted_findings = section.get("findings")
    schema_version = section.get("schema_version")
    rebuilt_findings: list[dict[str, object]] | None = None
    if isinstance(persisted_findings, list):
        rebuilt_findings = []
        for finding in persisted_findings:
            if not isinstance(finding, dict):
                continue
            rebuilt_finding = {
                "relation_index": relation_index_by_id.get(
                    str(finding.get("relation_id"))
                ),
                "relationship": finding.get("relationship"),
                "evidence_ids": finding.get("evidence_ids"),
                "unresolved_gaps": finding.get("unresolved_gaps"),
            }
            if schema_version == REQUIREMENT_COVERAGE_SCHEMA_VERSION:
                rebuilt_finding["edit_target_evidence_ids"] = finding.get(
                    "edit_target_evidence_ids"
                )
            rebuilt_findings.append(rebuilt_finding)
    findings = {
        "findings": rebuilt_findings
    }
    rebuilt, readiness, can_generate_patch, blockers, gaps = (
        resolve_requirement_coverage(
            query=pack.query,
            materialized_projection=materialized,
            items=pack.items,
            findings=findings,
            persisted_semantic_trust_boundary=(
                tuple(str(value) for value in section["semantic_trust_boundary"])
                if schema_version == REQUIREMENT_COVERAGE_SCHEMA_VERSION
                and isinstance(section.get("semantic_trust_boundary"), list)
                else None
            ),
        )
    )
    if rebuilt != section:
        raise ValueError("REQUIREMENT_COVERAGE_SEMANTIC_DRIFT")
    if (
        pack.readiness != readiness
        or pack.can_generate_patch != can_generate_patch
        or pack.blockers != blockers
        or pack.gaps != gaps
    ):
        raise ValueError("REQUIREMENT_COVERAGE_TOP_LEVEL_PROJECTION_DRIFT")


def resolve_requirement_coverage(
    *,
    query: str,
    materialized_projection: Mapping[str, Any],
    items: tuple[EvidenceItem, ...],
    findings: Mapping[str, Any],
    persisted_semantic_trust_boundary: tuple[str, ...] | None = None,
) -> tuple[
    dict[str, object], str, bool, tuple[dict[str, Any], ...], tuple[str, ...]
]:
    _validate_materialized_projection(query, materialized_projection)
    legacy_gap_semantics = (
        materialized_projection["schema_version"]
        == LEGACY_REQUIREMENT_COVERAGE_SCHEMA_VERSION
    )
    explicit_edit_targets = (
        materialized_projection["schema_version"]
        == REQUIREMENT_COVERAGE_SCHEMA_VERSION
    )
    legacy_v3_semantics = False
    if explicit_edit_targets and persisted_semantic_trust_boundary is not None:
        if (
            persisted_semantic_trust_boundary
            == LEGACY_REQUIREMENT_COVERAGE_V3_TRUST_BOUNDARY
        ):
            legacy_v3_semantics = True
        elif persisted_semantic_trust_boundary != REQUIREMENT_COVERAGE_V3_TRUST_BOUNDARY:
            raise ValueError("REQUIREMENT_COVERAGE_TRUST_BOUNDARY_INVALID")
    if set(findings) != {"findings"} or not isinstance(findings["findings"], list):
        raise ValueError("REQUIREMENT_COVERAGE_FINDINGS_SHAPE_INVALID")
    item_by_id = {item.evidence_id: item for item in items}
    relation_by_index = {
        int(relation["provenance"]["relation_index"]): relation
        for relation in materialized_projection["relations"]
    }
    normalized_findings: list[dict[str, object]] = []
    seen_relation_indexes: set[int] = set()
    for raw in findings["findings"]:
        expected_finding_fields = {
            "relation_index",
            "relationship",
            "evidence_ids",
            "unresolved_gaps",
        }
        if explicit_edit_targets:
            expected_finding_fields.add("edit_target_evidence_ids")
        if not isinstance(raw, dict) or set(raw) != expected_finding_fields:
            raise ValueError("REQUIREMENT_COVERAGE_FINDING_SHAPE_INVALID")
        relation_index = raw.get("relation_index")
        relationship = raw.get("relationship")
        evidence_ids = raw.get("evidence_ids")
        edit_target_ids = (
            raw.get("edit_target_evidence_ids")
            if explicit_edit_targets
            else None
        )
        unresolved_gaps = raw.get("unresolved_gaps")
        if (
            isinstance(relation_index, bool)
            or not isinstance(relation_index, int)
            or relation_index not in relation_by_index
            or relation_index in seen_relation_indexes
            or relationship not in REQUIREMENT_RELATIONSHIPS
            or not isinstance(evidence_ids, list)
            or len(evidence_ids) != len(set(evidence_ids))
            or any(value not in item_by_id for value in evidence_ids)
            or not isinstance(unresolved_gaps, list)
            or any(not isinstance(value, str) or not value for value in unresolved_gaps)
            or (relationship in {"support", "conflict"} and not evidence_ids)
        ):
            raise ValueError("REQUIREMENT_COVERAGE_FINDING_INVALID")
        seen_relation_indexes.add(relation_index)
        relation = relation_by_index[relation_index]
        relation_has_change_requirement = any(
            requirement.get("role") == "change"
            and requirement.get("requirement_id")
            in relation["parent_requirement_ids"]
            for requirement in materialized_projection["requirements"]
        )
        if explicit_edit_targets and (
            not isinstance(edit_target_ids, list)
            or len(edit_target_ids) != len(set(edit_target_ids))
            or any(value not in item_by_id for value in edit_target_ids)
            or not set(edit_target_ids).issubset(set(evidence_ids))
            or (
                bool(edit_target_ids)
                and (
                    relationship != "support"
                    or not relation_has_change_requirement
                    or bool(unresolved_gaps)
                )
            )
        ):
            error = ValueError("REQUIREMENT_COVERAGE_EDIT_TARGET_SELECTION_INVALID")
            # Diagnostic projection only: keep the original rejection and fields unchanged.
            error.diagnostic = {
                "relation_index": relation_index,
                "relationship": relationship,
                "parent_roles": [requirement["role"] for requirement in materialized_projection["requirements"]
                                 if requirement["requirement_id"] in relation["parent_requirement_ids"]],
                "edit_target_evidence_ids": edit_target_ids,
                "evidence_ids": evidence_ids,
                "unresolved_gaps": unresolved_gaps,
                "rule": "Nonempty edit hints require support, a change parent, no unresolved gaps, and a unique subset of cited evidence.",
                "relation": dict(relation),
                "violated_rules": [name for name, violated in (
                    ("edit_targets_must_be_an_array", not isinstance(edit_target_ids, list)),
                    ("edit_targets_must_be_unique", isinstance(edit_target_ids, list) and len(edit_target_ids) != len(set(edit_target_ids))),
                    ("edit_targets_must_be_known_evidence", isinstance(edit_target_ids, list) and any(value not in item_by_id for value in edit_target_ids)),
                    ("edit_targets_must_be_cited_evidence_subset", isinstance(edit_target_ids, list) and not set(edit_target_ids).issubset(evidence_ids)),
                    ("nonempty_edit_targets_require_support", bool(edit_target_ids) and relationship != "support"),
                    ("nonempty_edit_targets_require_change_parent", bool(edit_target_ids) and not relation_has_change_requirement),
                    ("nonempty_edit_targets_require_no_unresolved_gaps", bool(edit_target_ids) and bool(unresolved_gaps)),
                ) if violated],
            }
            raise error
        normalized_edit_target_ids = sorted(
            str(value) for value in (edit_target_ids or [])
        )
        finding_identity_parts = [
            "finding",
            str(relation["relation_id"]),
            str(relationship),
            *sorted(str(value) for value in evidence_ids),
        ]
        if explicit_edit_targets:
            finding_identity_parts.extend(
                ["edit_targets", *normalized_edit_target_ids]
            )
        normalized_findings.append(
            {
                "finding_id": _stable_id(*finding_identity_parts),
                "relation_id": relation["relation_id"],
                "parent_requirement_ids": list(relation["parent_requirement_ids"]),
                "relationship": relationship,
                "evidence_ids": sorted(str(value) for value in evidence_ids),
                **(
                    {"edit_target_evidence_ids": normalized_edit_target_ids}
                    if explicit_edit_targets
                    else {}
                ),
                "unresolved_gaps": list(unresolved_gaps),
            }
        )
    requirements = [dict(value) for value in materialized_projection["requirements"]]
    requirement_by_id = {
        str(value["requirement_id"]): value for value in requirements
    }
    required_ids = {
        str(value["requirement_id"])
        for value in requirements
        if value["required_before_patch"] is True
    }
    authoritative_ids = {
        item.evidence_id
        for item in items
        if item.authority == DETERMINISTIC_EVIDENCE_AUTHORITY
        and bool(item.source_path)
        and bool(item.source_hash or item.file_hash or item.doc_hash)
    }
    reasons: list[str] = []
    for requirement_id in sorted(required_ids):
        requirement_role = requirement_by_id[requirement_id]["role"]
        relevant = [
            finding
            for finding in normalized_findings
            if requirement_id in finding["parent_requirement_ids"]
        ]
        support_ids = {
            str(evidence_id)
            for finding in relevant
            if finding["relationship"] == "support"
            and (
                legacy_gap_semantics and requirement_role == "change"
                or not finding["unresolved_gaps"]
            )
            for evidence_id in finding["evidence_ids"]
            if evidence_id in authoritative_ids
        }
        if not support_ids:
            reasons.append(f"required_requirement_unsupported:{requirement_id}")
        if any(finding["relationship"] == "conflict" for finding in relevant):
            reasons.append(f"required_requirement_conflicting:{requirement_id}")
        if any(finding["relationship"] == "missing" for finding in relevant):
            reasons.append(f"required_requirement_missing:{requirement_id}")
        if any(
            finding["unresolved_gaps"]
            and not (
                legacy_gap_semantics
                and requirement_role == "change"
                and finding["relationship"] == "support"
            )
            for finding in relevant
        ):
            reasons.append(f"required_requirement_unresolved:{requirement_id}")
    change_ids = {
        str(value["requirement_id"])
        for value in requirements
        if value["role"] == "change"
    }
    change_support_evidence_ids = {
        str(evidence_id)
        for finding in normalized_findings
        if finding["relationship"] == "support"
        and change_ids.intersection(finding["parent_requirement_ids"])
        and (legacy_gap_semantics or not finding["unresolved_gaps"])
        for evidence_id in finding["evidence_ids"]
    }
    selected_edit_target_ids = (
        {
            str(evidence_id)
            for finding in normalized_findings
            if finding["relationship"] == "support"
            and change_ids.intersection(finding["parent_requirement_ids"])
            and not finding["unresolved_gaps"]
            for evidence_id in finding["edit_target_evidence_ids"]
        }
        if explicit_edit_targets
        else change_support_evidence_ids
    )
    edit_target_evidence_ids: list[str] = []
    for evidence_id in sorted(selected_edit_target_ids):
        item = item_by_id[evidence_id]
        exact_target_identity = is_exact_edit_target_evidence(item)
        if explicit_edit_targets and not exact_target_identity:
            error = ValueError("REQUIREMENT_COVERAGE_EDIT_TARGET_IDENTITY_INVALID")
            error.diagnostic = {"evidence_id": evidence_id,
                "violated_rules": ["edit_targets_require_exact_target_source_identity"],
                "rule": "Edit hints must satisfy is_exact_edit_target_evidence; API or incomplete source identities cannot be edit targets."}
            raise error
        if exact_target_identity:
            edit_target_evidence_ids.append(evidence_id)
    if legacy_v3_semantics:
        required_change_ids = {
            requirement_id
            for requirement_id in change_ids
            if requirement_by_id[requirement_id]["required_before_patch"] is True
        }
        for requirement_id in sorted(required_change_ids):
            selected_for_requirement = {
                str(evidence_id)
                for finding in normalized_findings
                if requirement_id in finding["parent_requirement_ids"]
                for evidence_id in finding["edit_target_evidence_ids"]
            }
            if not selected_for_requirement:
                reasons.append(
                    f"required_change_edit_target_missing:{requirement_id}"
                )
    reasons = list(dict.fromkeys(reasons))
    can_generate_patch = not reasons
    status = "sufficient" if can_generate_patch else "insufficient"
    blockers = tuple(
        {
            "code": "REQUIREMENT_COVERAGE_INSUFFICIENT",
            "message": reason,
            "needed_evidence": ["explicit authoritative requirement coverage"],
        }
        for reason in reasons
    )
    section = {
        **{key: _copy_json(value) for key, value in materialized_projection.items()},
        "findings": normalized_findings,
        "edit_target_evidence_ids": edit_target_evidence_ids,
        "sufficiency": {"status": status, "reasons": reasons},
        "semantic_trust_boundary": list(
            LEGACY_REQUIREMENT_COVERAGE_V3_TRUST_BOUNDARY
            if legacy_v3_semantics
            else REQUIREMENT_COVERAGE_V3_TRUST_BOUNDARY
            if explicit_edit_targets
            else REQUIREMENT_COVERAGE_TRUST_BOUNDARY
        ),
        "does_not_grant": list(REQUIREMENT_COVERAGE_DOES_NOT_GRANT),
    }
    return section, status, can_generate_patch, blockers, tuple(reasons)


def _validate_materialized_projection(
    query: str,
    value: Mapping[str, Any],
) -> None:
    if set(value) != {
        "schema_version",
        "feature_spec_identity",
        "request_identity",
        "projection_identity",
        "input_obligations",
        "input_dispositions",
        "requirements",
        "relations",
        "queries",
    } or value.get(
        "schema_version"
    ) not in _SUPPORTED_REQUIREMENT_COVERAGE_SCHEMA_VERSIONS:
        raise ValueError("REQUIREMENT_COVERAGE_MATERIALIZED_SHAPE_INVALID")
    feature_identity = value.get("feature_spec_identity")
    request_identity = value.get("request_identity")
    projection_identity = value.get("projection_identity")
    if (
        not isinstance(feature_identity, dict)
        or set(feature_identity) != {"feature_id", "feature_spec_sha256"}
        or not _hash_string(feature_identity.get("feature_spec_sha256"))
        or not isinstance(feature_identity.get("feature_id"), str)
        or not feature_identity["feature_id"]
        or not isinstance(request_identity, dict)
        or set(request_identity) != {"request_sha256"}
        or request_identity.get("request_sha256")
        != hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        or not isinstance(projection_identity, dict)
        or set(projection_identity) != {"projection_sha256"}
        or not _hash_string(projection_identity.get("projection_sha256"))
    ):
        raise ValueError("REQUIREMENT_COVERAGE_IDENTITY_INVALID")
    requirements = value.get("requirements")
    relations = value.get("relations")
    queries = value.get("queries")
    input_obligations = value.get("input_obligations")
    input_dispositions = value.get("input_dispositions")
    if (
        not isinstance(input_obligations, list)
        or not input_obligations
        or not isinstance(input_dispositions, list)
        or not isinstance(requirements, list)
        or not isinstance(relations, list)
        or not isinstance(queries, list)
    ):
        raise ValueError("REQUIREMENT_COVERAGE_COLLECTION_INVALID")
    normalized_projection = {
        "requirements": [
            {
                "role": requirement.get("role"),
                "text": requirement.get("text"),
                "input_obligation_indexes": [
                    next(
                        index
                        for index, obligation in enumerate(input_obligations, 1)
                        if obligation.get("input_obligation_id") == obligation_id
                    )
                    for obligation_id in requirement.get(
                        "input_obligation_ids", []
                    )
                ],
                "query_leads": requirement.get("query_leads"),
            }
            for requirement in requirements
            if isinstance(requirement, dict)
        ],
        "relations": [
            {
                "parent_requirement_indexes": [
                    next(
                        index
                        for index, requirement in enumerate(requirements, 1)
                        if requirement.get("requirement_id") == parent
                    )
                    for parent in relation.get("parent_requirement_ids", [])
                ],
                "meaning": relation.get("meaning"),
            }
            for relation in relations
            if isinstance(relation, dict)
        ],
        "input_dispositions": [
            {
                "input_obligation_index": next(
                    index
                    for index, obligation in enumerate(input_obligations, 1)
                    if obligation.get("input_obligation_id")
                    == disposition.get("input_obligation_id")
                ),
                "disposition": disposition.get("disposition"),
                "reason": disposition.get("reason"),
            }
            for disposition in input_dispositions
            if isinstance(disposition, dict)
        ],
    }
    if _json_sha256(normalized_projection) != projection_identity["projection_sha256"]:
        raise ValueError("REQUIREMENT_COVERAGE_PROJECTION_HASH_INVALID")
    rematerialized = _rematerialize_stored_requirement_coverage(
        feature_identity=feature_identity,
        input_obligations=input_obligations,
        request=query,
        projection=normalized_projection,
        schema_version=str(value["schema_version"]),
    )
    rematerialized["feature_spec_identity"] = dict(feature_identity)
    if rematerialized != dict(value):
        raise ValueError("REQUIREMENT_COVERAGE_IDENTITY_DRIFT")


def _normalized_projection(
    *,
    raw_requirements: list[Any],
    raw_relations: list[Any],
    raw_dispositions: object,
    input_obligations: list[dict[str, object]],
) -> dict[str, object]:
    if not isinstance(raw_dispositions, list):
        raise ValueError("REQUIREMENT_COVERAGE_INPUT_DISPOSITIONS_INVALID")
    requirements: list[dict[str, object]] = []
    accounted_indexes: set[int] = set()
    requirement_locations_by_obligation: dict[
        int, list[dict[str, object]]
    ] = {}
    for requirement_item_index, raw in enumerate(raw_requirements):
        if not isinstance(raw, dict) or set(raw) != {
            "role",
            "text",
            "input_obligation_indexes",
            "query_leads",
        }:
            raise ValueError("REQUIREMENT_COVERAGE_REQUIREMENT_SHAPE_INVALID")
        role = raw.get("role")
        text = raw.get("text")
        obligation_indexes = raw.get("input_obligation_indexes")
        query_leads = raw.get("query_leads")
        if (
            role not in REQUIREMENT_ROLES
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(obligation_indexes, list)
            or not obligation_indexes
            or len(obligation_indexes) != len(set(obligation_indexes))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 1 <= index <= len(input_obligations)
                for index in obligation_indexes
            )
            or not isinstance(query_leads, list)
            or not 1 <= len(query_leads) <= 4
            or any(not isinstance(value, str) or not value.strip() for value in query_leads)
        ):
            raise ValueError("REQUIREMENT_COVERAGE_REQUIREMENT_INVALID")
        role_conflicts = [
            {
                "input_obligation_index": int(index),
                "input_obligation_json_path": (
                    f"$.requirements[{requirement_item_index}]"
                    ".input_obligation_indexes"
                    f"[{input_obligation_item_index}]"
                ),
                "allowed_roles": list(
                    input_obligations[int(index) - 1]["allowed_roles"]
                ),
            }
            for input_obligation_item_index, index in enumerate(
                obligation_indexes
            )
            if role not in input_obligations[int(index) - 1]["allowed_roles"]
        ]
        if role_conflicts:
            raise RequirementCoverageProjectionError(
                "REQUIREMENT_COVERAGE_REQUIREMENT_INVALID",
                {
                    "phase": "semantic_materialization",
                    "failure_category": "requirement_role_not_allowed",
                    "json_path": f"$.requirements[{requirement_item_index}].role",
                    "requirement_item_index": requirement_item_index,
                    "observed_role": role,
                    "role_conflicts": role_conflicts,
                    "allowed_boundary": {
                        "role_must_be_allowed_by_every_referenced_obligation": True,
                        "split_requirement_when_no_single_role_is_shared": True,
                    },
                },
            )
        normalized_query_leads = list(
            dict.fromkeys(value.strip() for value in query_leads)
        )
        accounted_indexes.update(int(index) for index in obligation_indexes)
        for input_obligation_item_index, input_obligation_index in enumerate(
            obligation_indexes
        ):
            requirement_locations_by_obligation.setdefault(
                int(input_obligation_index), []
            ).append(
                {
                    "requirement_item_index": requirement_item_index,
                    "requirement_item_json_path": (
                        f"$.requirements[{requirement_item_index}]"
                    ),
                    "input_obligation_item_index": input_obligation_item_index,
                    "input_obligation_json_path": (
                        f"$.requirements[{requirement_item_index}]"
                        ".input_obligation_indexes"
                        f"[{input_obligation_item_index}]"
                    ),
                }
            )
        requirements.append(
            {
                "role": role,
                "text": text.strip(),
                "input_obligation_indexes": [
                    int(value) for value in obligation_indexes
                ],
                "query_leads": normalized_query_leads,
            }
        )
    dispositions: list[dict[str, object]] = []
    disposition_indexes: dict[int, int] = {}
    for disposition_item_index, raw in enumerate(raw_dispositions):
        if not isinstance(raw, dict) or set(raw) != {
            "input_obligation_index",
            "disposition",
            "reason",
        }:
            raise ValueError("REQUIREMENT_COVERAGE_INPUT_DISPOSITION_SHAPE_INVALID")
        index = raw.get("input_obligation_index")
        disposition = raw.get("disposition")
        reason = raw.get("reason")
        item_path = f"$.input_dispositions[{disposition_item_index}]"
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 1 <= index <= len(input_obligations)
        ):
            raise RequirementCoverageProjectionError(
                "REQUIREMENT_COVERAGE_INPUT_DISPOSITION_INVALID",
                {
                    "phase": "semantic_materialization",
                    "failure_category": "invalid_obligation_index",
                    "json_path": f"{item_path}.input_obligation_index",
                    "input_obligation_count": len(input_obligations),
                },
            )
        if index in disposition_indexes:
            raise RequirementCoverageProjectionError(
                "REQUIREMENT_COVERAGE_INPUT_DISPOSITION_INVALID",
                {
                    "phase": "semantic_materialization",
                    "failure_category": "duplicate_disposition_index",
                    "json_path": f"{item_path}.input_obligation_index",
                    "input_obligation_index": index,
                    "first_disposition_item_index": disposition_indexes[index],
                    "duplicate_disposition_item_index": disposition_item_index,
                },
            )
        if index in accounted_indexes:
            conflicting_requirement_locations = [
                dict(location)
                for location in requirement_locations_by_obligation[index]
            ]
            conflicting_requirement_paths = [
                str(location["input_obligation_json_path"])
                for location in conflicting_requirement_locations
            ]
            allowed_dispositions = list(
                input_obligations[index - 1]["allowed_dispositions"]
            )
            affected_requirements_remain_non_empty = all(
                len(
                    raw_requirements[
                        int(location["requirement_item_index"])
                    ]["input_obligation_indexes"]
                )
                > 1
                for location in conflicting_requirement_locations
            )
            empty_requirement_item_indexes_after_removal = [
                int(location["requirement_item_index"])
                for location in conflicting_requirement_locations
                if len(
                    raw_requirements[
                        int(location["requirement_item_index"])
                    ]["input_obligation_indexes"]
                )
                == 1
            ]
            raw_valid_disposition_indexes = {
                int(value["input_obligation_index"])
                for value in raw_dispositions
                if isinstance(value, dict)
                and type(value.get("input_obligation_index")) is int
                and 1
                <= int(value["input_obligation_index"])
                <= len(input_obligations)
            }
            missing_indexes_after_requirement_removal = sorted(
                set(range(1, len(input_obligations) + 1))
                - (accounted_indexes - {index})
                - raw_valid_disposition_indexes
            )
            observed_disposition_is_allowed = (
                disposition in allowed_dispositions
            )
            disposition_reason_is_non_empty = (
                isinstance(reason, str) and bool(reason.strip())
            )
            all_input_obligations_remain_accounted = not (
                missing_indexes_after_requirement_removal
            )
            raise RequirementCoverageProjectionError(
                "REQUIREMENT_COVERAGE_INPUT_DISPOSITION_INVALID",
                {
                    "phase": "semantic_materialization",
                    "failure_category": (
                        "obligation_already_accounted_and_dispositioned"
                    ),
                    "json_path": f"{item_path}.input_obligation_index",
                    "input_obligation_index": index,
                    "allowed_dispositions": allowed_dispositions,
                    "conflicting_requirement_locations": (
                        conflicting_requirement_locations
                    ),
                    "disposition_item_index": disposition_item_index,
                    "disposition_item_json_path": item_path,
                    "correction_directions": [
                        {
                            "direction": (
                                "keep_requirement_accounting_remove_"
                                "input_disposition"
                            ),
                            "remove_json_paths": [item_path],
                            "retain_json_paths": conflicting_requirement_paths,
                            "conditions": {
                                "listed_requirement_accounting_is_retained": True,
                                "complete_corrected_projection_must_pass_"
                                "existing_validation": True,
                            },
                            "locally_applicable": True,
                        },
                        {
                            "direction": (
                                "keep_allowed_disposition_remove_requirement_"
                                "accounting"
                            ),
                            "remove_json_paths": conflicting_requirement_paths,
                            "retain_json_paths": [item_path],
                            "conditions": {
                                "observed_disposition_is_allowed": (
                                    observed_disposition_is_allowed
                                ),
                                "disposition_reason_is_non_empty_after_strip": (
                                    disposition_reason_is_non_empty
                                ),
                                "affected_requirements_remain_non_empty": (
                                    affected_requirements_remain_non_empty
                                ),
                                "empty_requirement_item_indexes_after_removal": (
                                    empty_requirement_item_indexes_after_removal
                                ),
                                "all_input_obligations_remain_accounted": (
                                    all_input_obligations_remain_accounted
                                ),
                                "missing_input_obligation_indexes_after_removal": (
                                    missing_indexes_after_requirement_removal
                                ),
                                "complete_corrected_projection_must_pass_"
                                "existing_validation": True,
                            },
                            "locally_applicable": (
                                observed_disposition_is_allowed
                                and disposition_reason_is_non_empty
                                and affected_requirements_remain_non_empty
                                and all_input_obligations_remain_accounted
                            ),
                        },
                    ],
                },
            )
        allowed_dispositions = list(
            input_obligations[index - 1]["allowed_dispositions"]
        )
        if disposition not in allowed_dispositions:
            raise RequirementCoverageProjectionError(
                "REQUIREMENT_COVERAGE_INPUT_DISPOSITION_INVALID",
                {
                    "phase": "semantic_materialization",
                    "failure_category": "disallowed_disposition",
                    "json_path": f"{item_path}.disposition",
                    "input_obligation_index": index,
                    "allowed_dispositions": allowed_dispositions,
                    "observed_disposition": disposition,
                },
            )
        if not isinstance(reason, str) or not reason.strip():
            raise RequirementCoverageProjectionError(
                "REQUIREMENT_COVERAGE_INPUT_DISPOSITION_INVALID",
                {
                    "phase": "semantic_materialization",
                    "failure_category": "invalid_reason",
                    "json_path": f"{item_path}.reason",
                    "input_obligation_index": index,
                    "required_boundary": "non_empty_after_strip",
                },
            )
        disposition_indexes[index] = disposition_item_index
        dispositions.append(
            {
                "input_obligation_index": index,
                "disposition": disposition,
                "reason": reason.strip(),
            }
        )
    disposition_index_set = set(disposition_indexes)
    if accounted_indexes | disposition_index_set != set(
        range(1, len(input_obligations) + 1)
    ):
        missing_indexes = sorted(
            set(range(1, len(input_obligations) + 1))
            - accounted_indexes
            - disposition_index_set
        )
        raise RequirementCoverageProjectionError(
            "REQUIREMENT_COVERAGE_INPUT_ACCOUNTING_INCOMPLETE",
            {
                "phase": "semantic_materialization",
                "failure_category": "input_accounting_incomplete",
                "json_path": "$.input_dispositions",
                "missing_input_obligation_indexes": missing_indexes,
                "input_obligation_count": len(input_obligations),
            },
        )
    relations: list[dict[str, object]] = []
    for raw in raw_relations:
        if not isinstance(raw, dict) or set(raw) != {
            "parent_requirement_indexes",
            "meaning",
        }:
            raise ValueError("REQUIREMENT_COVERAGE_RELATION_SHAPE_INVALID")
        parents = raw.get("parent_requirement_indexes")
        meaning = raw.get("meaning")
        if (
            not isinstance(parents, list)
            or not parents
            or len(parents) != len(set(parents))
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 1 <= index <= len(requirements)
                for index in parents
            )
            or not isinstance(meaning, str)
            or not meaning.strip()
        ):
            raise ValueError("REQUIREMENT_COVERAGE_RELATION_INVALID")
        relations.append(
            {
                "parent_requirement_indexes": [int(value) for value in parents],
                "meaning": meaning.strip(),
            }
        )
    return {
        "requirements": requirements,
        "relations": relations,
        "input_dispositions": dispositions,
    }


def _projected_requirement(
    *, role: str, text: str, input_obligation_indexes: list[int]
) -> dict[str, object]:
    return {
        "role": role,
        "text": text,
        "input_obligation_indexes": list(input_obligation_indexes),
        "query_leads": [text],
    }


def _input_obligations(
    feature_data: Mapping[str, Any],
) -> list[dict[str, object]]:
    _reject_feature_spec_derived_fields(feature_data)
    feature_hash = feature_spec_payload_sha256(feature_data)
    interpretation = feature_data.get("request_interpretation")
    decomposable = (isinstance(interpretation, Mapping)
                    and interpretation.get("schema_version") == "request_interpretation_v2")
    # Versioned role expansion preserves historical input-obligation identities.
    # 为什么要有它：修后的完整需求可做语义分解，旧产物仍按原契约读取。
    semantic_roles = ("change", "preservation", "pre_patch_evidence")
    sources: list[
        tuple[str, int, str, tuple[str, ...], bool, tuple[str, ...]]
    ] = [
        (
            "user_request",
            0,
            _required_string(feature_data, "user_request"),
            semantic_roles if decomposable else ("change", "preservation"),
            True,
            (),
        ),
        (
            "target_behavior",
            0,
            _required_string(feature_data, "target_behavior"),
            semantic_roles if decomposable else ("change",),
            True,
            (),
        ),
    ]
    from src.mod_agent.planning.clarified_request import ANCHORED_VERSION, projection_context
    clarified = projection_context(_required_string(feature_data, "user_request"))
    if clarified is not None and clarified["schema_version"] == ANCHORED_VERSION:
        if _required_string(feature_data, "target_behavior") != _required_string(feature_data, "user_request"):
            raise ValueError("CLARIFIED_PRODUCT_TARGET_BEHAVIOR_CHANGED")
        eligible = {anchor["unit_id"] for anchor in clarified["eligible_context_anchors"]}
        # 为什么要有它：逐原文单元承接业务义务；仅已验证路由范围允许显式context处置，不把整wrapper重复当Mod行为。
        sources = [("clarified_user_unit", index, unit["text"], semantic_roles, True,
                    ("scope_boundary",) if unit["unit_id"] in eligible else ())
                   for index, unit in enumerate(clarified["user_units"], 1)]
    sources.extend(
        (
            "evidence_requirements",
            index,
            text,
            ("pre_patch_evidence",),
            True,
            (),
        )
        for index, text in enumerate(
            _string_list(feature_data, "evidence_requirements"), 1
        )
    )
    sources.extend(
        (
            "non_goals",
            index,
            text,
            ("preservation", "pre_patch_evidence"),
            True,
            ("scope_boundary",),
        )
        for index, text in enumerate(_string_list(feature_data, "non_goals"), 1)
    )
    sources.extend(
        (
            "acceptance_criteria",
            index,
            text,
            ("post_change_acceptance",),
            False,
            ("post_change",),
        )
        for index, text in enumerate(
            _string_list(feature_data, "acceptance_criteria"), 1
        )
    )
    sources.extend(
        (
            "runtime_verification_plan",
            index,
            text,
            ("post_change_acceptance",),
            False,
            ("post_change",),
        )
        for index, text in enumerate(
            _string_list(feature_data, "runtime_verification_plan"), 1
        )
    )
    obligations = [
        {
            "input_obligation_id": _stable_id(
                "obligation", feature_hash, field, str(index), text
            ),
            "origin": {"field": field, "index": index},
            "text": text,
            "mandatory": True,
            "allowed_roles": list(roles),
            "allowed_dispositions": list(allowed_dispositions),
            "required_before_patch": required_before_patch,
        }
        for (
            field,
            index,
            text,
            roles,
            required_before_patch,
            allowed_dispositions,
        ) in sources
    ]
    return _validated_input_obligations(
        obligations,
        feature_spec_sha256=feature_hash,
    )


def _validated_input_obligations(
    supplied: object,
    *,
    feature_spec_sha256: str,
) -> list[dict[str, object]]:
    if not isinstance(supplied, list) or not supplied:
        raise ValueError("REQUIREMENT_COVERAGE_INPUT_OBLIGATIONS_INVALID")
    obligations = [_copy_json(value) for value in supplied]
    for obligation in obligations:
        if not isinstance(obligation, dict) or set(obligation) != {
            "input_obligation_id",
            "origin",
            "text",
            "mandatory",
            "allowed_roles",
            "allowed_dispositions",
            "required_before_patch",
        }:
            raise ValueError("REQUIREMENT_COVERAGE_INPUT_OBLIGATION_SHAPE_INVALID")
        origin = obligation.get("origin")
        if (
            not isinstance(origin, dict)
            or set(origin) != {"field", "index"}
            or not isinstance(origin.get("field"), str)
            or isinstance(origin.get("index"), bool)
            or not isinstance(origin.get("index"), int)
            or not isinstance(obligation.get("text"), str)
            or not obligation["text"]
            or obligation.get("mandatory") is not True
            or not isinstance(obligation.get("allowed_roles"), list)
            or not obligation["allowed_roles"]
            or len(obligation["allowed_roles"])
            != len(set(obligation["allowed_roles"]))
            or any(
                role not in REQUIREMENT_ROLES
                for role in obligation["allowed_roles"]
            )
            or not isinstance(obligation.get("allowed_dispositions"), list)
            or len(obligation["allowed_dispositions"])
            != len(set(obligation["allowed_dispositions"]))
            or any(
                disposition not in {"post_change", "scope_boundary"}
                for disposition in obligation["allowed_dispositions"]
            )
            or not isinstance(obligation.get("required_before_patch"), bool)
            or obligation.get("input_obligation_id")
            != _stable_id(
                "obligation",
                feature_spec_sha256,
                str(origin["field"]),
                str(origin["index"]),
                str(obligation["text"]),
            )
        ):
            raise ValueError("REQUIREMENT_COVERAGE_INPUT_OBLIGATION_INVALID")
    return [dict(value) for value in obligations]


def _rematerialize_stored_requirement_coverage(
    *,
    feature_identity: Mapping[str, Any],
    input_obligations: object,
    request: str,
    projection: Mapping[str, Any],
    schema_version: str,
) -> dict[str, object]:
    """Rebuild persisted coverage without pretending it is FeatureSpec data."""

    if (
        set(feature_identity) != {"feature_id", "feature_spec_sha256"}
        or not isinstance(feature_identity.get("feature_id"), str)
        or not feature_identity["feature_id"]
        or not _hash_string(feature_identity.get("feature_spec_sha256"))
    ):
        raise ValueError("REQUIREMENT_COVERAGE_IDENTITY_INVALID")
    feature_spec_sha256 = str(feature_identity["feature_spec_sha256"])
    return _materialize_requirement_coverage_projection(
        feature_id=str(feature_identity["feature_id"]),
        feature_spec_sha256=feature_spec_sha256,
        input_obligations=_validated_input_obligations(
            input_obligations,
            feature_spec_sha256=feature_spec_sha256,
        ),
        request=request,
        projection=projection,
        schema_version=schema_version,
    )


def _reject_feature_spec_derived_fields(feature_data: Mapping[str, Any]) -> None:
    reserved = sorted(_FEATURE_SPEC_DERIVED_FIELDS.intersection(feature_data))
    if reserved:
        raise ValueError(
            "REQUIREMENT_COVERAGE_FEATURE_DATA_RESERVED_FIELD:"
            + ",".join(reserved)
        )


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"REQUIREMENT_COVERAGE_{key.upper()}_INVALID")
    return result.strip()


def _string_list(value: Mapping[str, Any], key: str) -> list[str]:
    raw = value.get(key)
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        raise ValueError(f"REQUIREMENT_COVERAGE_{key.upper()}_INVALID")
    return [str(item).strip() for item in raw]


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _hash_string(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _copy_json(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))
