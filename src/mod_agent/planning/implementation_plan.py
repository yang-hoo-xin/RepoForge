"""Implementation planning from feature specs and evidence packs."""

from __future__ import annotations

from typing import Any

from src.mod_agent.evidence.compilation_candidate import compilation_candidate_scope
from src.mod_agent.evidence.identity import feature_spec_payload_sha256
from src.mod_agent.evidence.models import (
    DETERMINISTIC_EVIDENCE_AUTHORITY,
    EVIDENCE_PACK_AUTHORITY,
    LEGACY_RAG_AUXILIARY_AUTHORITY,
)
from src.mod_agent.evidence.requirement_coverage import (
    validated_requirement_coverage_view,
)
from src.mod_agent.knowledge import IDENTITY_FIELD_MAPPING, IDENTITY_SCHEMA_VERSION

from .models import FeatureSpec, ImplementationPlan

EDITABLE_EVIDENCE_SOURCE_TYPES = frozenset({"target_source"})
HOOK_TARGET_SOURCE_TYPE = "decompiled_api"
HOOK_TARGET_USAGE = "hook_target"
AUTHORITATIVE_EVIDENCE_PACK = EVIDENCE_PACK_AUTHORITY
AUTHORITATIVE_ITEM_AUTHORITY = DETERMINISTIC_EVIDENCE_AUTHORITY
PLATFORM_SIDE_EFFECT_BOUNDARIES = (
    "Do not perform unrelated refactors.",
    # 为什么要有它：实现阶段仍需遵守任务范围，无需把平台政策当成待证明的业务事实。
    "Do not modify files outside evidence-supported scope.",
    # 为什么要有它：Patch仍须遵守现有绑定和审批范围，文本政策本身不授予写权限。
    "Do not claim runtime success without runtime evidence.",
    # 为什么要有它：执行结果报告仍须有对应证据，前置语义记账不承担这一真实性校验。
)
EVIDENCE_ITEM_PROVENANCE_FIELDS = frozenset(
    {"chunk_id", "document_id", "source_ref", "source_hash", "file_hash", "doc_hash"}
)


class ImplementationPlanner:
    """Create an evidence-bound implementation plan without producing a patch."""

    def build(
        self,
        *,
        feature_spec: FeatureSpec,
        evidence_pack: dict[str, Any],
        formal_requirement_coverage_required: bool = False,
        planning_target_evidence_ids: tuple[str, ...] | None = None,
        constraint_policy: str | None = None,
    ) -> ImplementationPlan:
        """Return an implementation plan or a blocking result."""
        from .constraint_context import build_constraint_context, validate_constraint_policy

        validate_constraint_policy(constraint_policy)

        if feature_spec.blocked:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=feature_spec.blockers,
                open_questions=("Feature spec is blocked.",),
            )

        try:
            compilation_scope = compilation_candidate_scope(
                evidence_pack, feature_spec.to_artifact_payload())
        except ValueError as exc:
            return self._blocked(feature_id=feature_spec.feature_id,
                blockers=({"code": "COMPILATION_CANDIDATE_SCOPE_INVALID", "message": str(exc)},),
                open_questions=("Revalidate the source-bound compilation candidate.",))
        # 为什么要有它：编译错误限定的候选可在语义问题未决时进入只读Plan，Evidence的原始不足判断和源码权限仍须保留。
        if compilation_scope is None and (evidence_pack.get("readiness") != "sufficient"
                or not evidence_pack.get("can_generate_patch")):
            blockers = tuple(evidence_pack.get("blockers") or ())
            open_questions = tuple(
                str(gap) for gap in evidence_pack.get("gaps", ["Evidence is not sufficient."])
            )
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=blockers,
                open_questions=open_questions,
            )

        if (
            formal_requirement_coverage_required
            and "requirement_coverage" not in evidence_pack
        ):
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "FORMAL_REQUIREMENT_COVERAGE_REQUIRED",
                        "message": (
                            "Selected ProductExecution requires formal requirement "
                            "coverage before edit-binding derivation."
                        ),
                    },
                ),
                open_questions=(
                    "Rebuild formal requirement coverage before planning edits.",
                ),
            )

        if (
            evidence_pack.get("authority") != AUTHORITATIVE_EVIDENCE_PACK
            or (compilation_scope is None
                and evidence_pack.get("authoritative_for_patch") is not True)
        ):
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "NON_AUTHORITATIVE_EVIDENCE_PACK",
                        "message": (
                            "Only mod_agent deterministic evidence packs can "
                            "advance implementation planning."
                        ),
                    },
                ),
                open_questions=(
                    "Resolve mod_agent deterministic evidence before planning edits.",
                ),
            )

        if (
            evidence_pack.get("identity_schema_version") != IDENTITY_SCHEMA_VERSION
            or evidence_pack.get("identity_field_mapping") != dict(IDENTITY_FIELD_MAPPING)
        ):
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "UNVERIFIED_EVIDENCE_PROVENANCE",
                        "message": (
                            "Authoritative evidence packs must carry the current "
                            "identity schema and field mapping."
                        ),
                    },
                ),
                open_questions=("Resolve provenance-validated evidence before planning edits.",),
            )

        raw_items = list(evidence_pack.get("items") or [])
        items = [item for item in raw_items if isinstance(item, dict)]
        if len(items) != len(raw_items):
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "MALFORMED_EVIDENCE_ITEMS",
                        "message": "Evidence pack items must be mappings.",
                    },
                ),
                open_questions=("Regenerate evidence with structured evidence items.",),
            )
        missing_authority_items = [
            item
            for item in items
            if "authority" not in item
        ]
        if missing_authority_items:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "NON_AUTHORITATIVE_EVIDENCE_ITEM",
                        "message": (
                            "Every evidence item used for implementation planning "
                            "must be mod_agent deterministic evidence."
                        ),
                    },
                ),
                open_questions=("Regenerate evidence with explicit item authority.",),
            )
        unsupported_authority_items = [
            item
            for item in items
            if item.get("authority")
            not in {AUTHORITATIVE_ITEM_AUTHORITY, LEGACY_RAG_AUXILIARY_AUTHORITY}
        ]
        if unsupported_authority_items:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "UNSUPPORTED_EVIDENCE_ITEM_AUTHORITY",
                        "message": "Evidence item authority is not recognized.",
                    },
                ),
                open_questions=("Regenerate evidence with recognized item authority.",),
            )
        planning_items = [
            item for item in items if item.get("authority") == AUTHORITATIVE_ITEM_AUTHORITY
        ]
        if not planning_items:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "MISSING_AUTHORITATIVE_EVIDENCE_ITEM",
                        "message": (
                            "Implementation planning requires at least one "
                            "mod_agent deterministic evidence item."
                        ),
                    },
                ),
                open_questions=("Resolve mod_agent deterministic evidence before planning edits.",),
            )
        unprovenanced_items = [
            item
            for item in planning_items
            if not any(item.get(field) for field in EVIDENCE_ITEM_PROVENANCE_FIELDS)
        ]
        if unprovenanced_items:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "UNVERIFIED_EVIDENCE_ITEM_PROVENANCE",
                        "message": (
                            "Every evidence item used for implementation planning "
                            "must carry source identity metadata."
                        ),
                    },
                ),
                open_questions=("Regenerate evidence with item provenance metadata.",),
            )

        evidence_refs = tuple(
            item["evidence_id"] for item in planning_items if item.get("evidence_id")
        )
        evidence_source_paths = tuple(
            dict.fromkeys(
                item["source_path"] for item in planning_items if item.get("source_path")
            )
        )
        try:
            coverage_view = validated_requirement_coverage_view(evidence_pack)
        except ValueError as exc:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "REQUIREMENT_COVERAGE_INVALID",
                        "message": str(exc),
                    },
                ),
                open_questions=("Rebuild requirement coverage before planning.",),
                evidence_source_paths=evidence_source_paths,
            )
        selected_edit_target_ids: frozenset[str] | None = None
        preservation_boundaries: tuple[str, ...] = ()
        change_related_evidence_ids: frozenset[str] = frozenset()
        if coverage_view is not None:
            expected_feature_hash = feature_spec_payload_sha256(
                feature_spec.to_artifact_payload()
            )
            if (
                coverage_view.feature_id != feature_spec.feature_id
                or coverage_view.feature_spec_sha256 != expected_feature_hash
            ):
                return self._blocked(
                    feature_id=feature_spec.feature_id,
                    blockers=(
                        {
                            "code": "REQUIREMENT_COVERAGE_INVALID",
                            "message": (
                                "Implementation planning requires one valid "
                                "requirement coverage projection."
                            ),
                        },
                    ),
                    open_questions=("Rebuild requirement coverage before planning.",),
                    evidence_source_paths=evidence_source_paths,
                )
            preservation_boundaries = (
                coverage_view.preservation_requirement_texts
            )
            if formal_requirement_coverage_required or compilation_scope is not None:
                selection, selection_blocker = self._formal_target_selection(
                    raw_selection=planning_target_evidence_ids,
                    planning_items=planning_items,
                    change_support_evidence_ids=(
                        frozenset(compilation_scope["evidence_ids"]) if compilation_scope is not None
                        else coverage_view.change_support_evidence_ids
                    ),
                )
                if selection_blocker is not None:
                    return self._blocked(
                        feature_id=feature_spec.feature_id,
                        blockers=(selection_blocker,),
                        open_questions=(selection_blocker["message"],),
                        evidence_source_paths=evidence_source_paths,
                    )
                selected_edit_target_ids = selection
            else:
                selected_edit_target_ids = (
                    coverage_view.edit_target_hint_evidence_ids
                )
            change_related_evidence_ids = (
                coverage_view.change_support_evidence_ids
                - (selected_edit_target_ids or frozenset())
            )
        editable_items = [
            item
            for item in planning_items
            if item.get("evidence_id")
            and item.get("source_path")
            and item.get("source_type") in EDITABLE_EVIDENCE_SOURCE_TYPES
            and (
                selected_edit_target_ids is None
                or str(item["evidence_id"]) in selected_edit_target_ids
            )
        ]
        target_files = tuple(
            dict.fromkeys(item["source_path"] for item in editable_items)
        )
        target_symbols = tuple(
            dict.fromkeys(item["symbol"] for item in editable_items if item.get("symbol"))
        )
        if not evidence_refs:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "MISSING_EVIDENCE_BINDING",
                        "message": "Sufficient readiness was claimed without evidence refs.",
                    },
                ),
                open_questions=("Evidence pack lacks evidence ids.",),
                evidence_source_paths=evidence_source_paths,
            )

        if not target_files:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(
                    {
                        "code": "EDIT_TARGET_UNRESOLVED",
                        "message": (
                            "Evidence sources do not resolve to an editable target source."
                        ),
                        "evidence_source_paths": list(evidence_source_paths),
                    },
                ),
                open_questions=(
                    "Resolve an editable target source before patch proposal generation.",
                ),
                evidence_source_paths=evidence_source_paths,
            )

        edit_units, normalization_blocker = self._normalize_edit_units(editable_items)
        if normalization_blocker is not None:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(normalization_blocker,),
                open_questions=(normalization_blocker["message"],),
                evidence_source_paths=evidence_source_paths,
            )
        edit_bindings = tuple(
            self._edit_binding(index=index, item=item)
            for index, (item, _) in enumerate(edit_units, start=1)
        )
        supporting_bindings = self._supporting_bindings(planning_items)
        binding_registry = self._binding_registry(
            edit_bindings=edit_bindings,
            supporting_bindings=supporting_bindings,
        )
        change_supporting_binding_ids = tuple(
            str(binding["binding_id"])
            for binding in binding_registry["supporting_bindings"]
            if str(binding.get("evidence_id") or "")
            in change_related_evidence_ids
        )
        supporting_binding_ids_by_evidence = {
            str(binding["evidence_id"]): str(binding["binding_id"])
            for binding in binding_registry["supporting_bindings"]
            if binding.get("evidence_id") and binding.get("binding_id")
        }
        hook_target_binding_id, hook_target_blocker = self._select_hook_target_binding(
            feature_spec=feature_spec,
            binding_registry=binding_registry,
        )
        if hook_target_blocker is not None:
            return self._blocked(
                feature_id=feature_spec.feature_id,
                blockers=(hook_target_blocker,),
                open_questions=(hook_target_blocker["message"],),
                evidence_source_paths=evidence_source_paths,
            )
        proposed_changes = tuple(
            self._proposed_change(
                index=index,
                feature_spec=feature_spec,
                hook_target_binding_id=hook_target_binding_id,
                supporting_binding_ids=tuple(
                    dict.fromkeys(
                        (
                            *change_supporting_binding_ids,
                            *(
                                supporting_binding_ids_by_evidence[evidence_id]
                                for evidence_id in merged_evidence_ids
                            ),
                        )
                    )
                ),
            )
            for index, (_, merged_evidence_ids) in enumerate(edit_units, start=1)
        )

        boundaries = tuple(dict.fromkeys((*PLATFORM_SIDE_EFFECT_BOUNDARIES,
                            *feature_spec.non_goals, *preservation_boundaries)))
        # Scope: explicit experiment only; baseline plans emit exactly their previous fields.
        # 为什么要有它：A的语义不能随A*/B实验悄悄改变，派生提示也不能继续被当成用户硬限制。
        constraint_context = (build_constraint_context(feature_spec=feature_spec.to_artifact_payload(),
            evidence_pack=evidence_pack, plan_boundaries=boundaries) if constraint_policy is not None else None)
        return ImplementationPlan(
            feature_id=feature_spec.feature_id,
            target_files=target_files,
            target_symbols=target_symbols,
            evidence_refs=evidence_refs,
            proposed_changes=proposed_changes,
            side_effect_boundaries=tuple(constraint_context["hard_boundaries"]) if constraint_context else boundaries,
            constraint_context=constraint_context,
            test_plan=feature_spec.runtime_verification_plan,
            rollback_plan=tuple(
                f"Rollback {change['change_id']}: {change['rollback_hint']}"
                for change in proposed_changes
            ),
            open_questions=(),
            verification_questions=tuple(compilation_scope["verification_questions"])
                if compilation_scope is not None else (),
            blocking=False,
            evidence_source_paths=evidence_source_paths,
            binding_registry=binding_registry,
        )

    @staticmethod
    def _formal_target_selection(
        *,
        raw_selection: tuple[str, ...] | None,
        planning_items: list[dict[str, Any]],
        change_support_evidence_ids: frozenset[str],
    ) -> tuple[frozenset[str], dict[str, Any] | None]:
        """Admit one explicit Planning-owned target selection."""

        if raw_selection is None or raw_selection == ():
            return frozenset(), {
                "code": "PLANNING_TARGET_SELECTION_REQUIRED",
                "message": (
                    "Formal requirement coverage requires an explicit Planning "
                    "edit-target selection. Coverage edit-target fields are hints only."
                ),
            }
        if not isinstance(raw_selection, tuple) or any(
            type(value) is not str or not value
            for value in raw_selection
        ):
            return frozenset(), {
                "code": "PLANNING_TARGET_SELECTION_INVALID",
                "message": (
                    "Planning edit-target selection must be a non-empty tuple "
                    "of exact evidence IDs."
                ),
                "reason": "malformed_selection",
            }
        if len(raw_selection) != len(set(raw_selection)):
            duplicate_ids = tuple(
                dict.fromkeys(
                    evidence_id
                    for evidence_id in raw_selection
                    if raw_selection.count(evidence_id) > 1
                )
            )
            return frozenset(), {
                "code": "PLANNING_TARGET_SELECTION_INVALID",
                "message": "Planning edit-target selection contains duplicate evidence IDs.",
                "reason": "duplicate_evidence_ids",
                "evidence_ids": list(duplicate_ids),
            }
        item_by_id = {
            str(item["evidence_id"]): item
            for item in planning_items
            if item.get("evidence_id")
        }
        unknown_ids = tuple(
            evidence_id
            for evidence_id in raw_selection
            if evidence_id not in item_by_id
        )
        if unknown_ids:
            return frozenset(), {
                "code": "PLANNING_TARGET_SELECTION_INVALID",
                "message": (
                    "Planning edit-target selection contains evidence IDs that "
                    "are unavailable to authoritative planning."
                ),
                "reason": "unknown_evidence_ids",
                "evidence_ids": list(unknown_ids),
            }
        outside_support_ids = tuple(
            evidence_id
            for evidence_id in raw_selection
            if evidence_id not in change_support_evidence_ids
        )
        if outside_support_ids:
            return frozenset(), {
                "code": "PLANNING_TARGET_SELECTION_INVALID",
                "message": (
                    "Planning edit-target selection must remain within the "
                    "complete change-support evidence pool."
                ),
                "reason": "outside_change_support",
                "evidence_ids": list(outside_support_ids),
            }
        unresolved_ids = tuple(
            evidence_id
            for evidence_id in raw_selection
            if (
                item_by_id[evidence_id].get("source_type")
                not in EDITABLE_EVIDENCE_SOURCE_TYPES
                or not isinstance(
                    item_by_id[evidence_id].get("source_path"), str
                )
                or not item_by_id[evidence_id]["source_path"]
                or ImplementationPlanner._physical_source_revision_hash(
                    item_by_id[evidence_id]
                )
                is None
                or ImplementationPlanner._exact_line_span(
                    item_by_id[evidence_id].get("span")
                )
                is None
            )
        )
        if unresolved_ids:
            return frozenset(), {
                "code": "PLANNING_TARGET_SELECTION_UNRESOLVED",
                "message": (
                    "Planning edit-target selection contains evidence that "
                    "does not resolve to an exact current target_source identity."
                ),
                "evidence_ids": list(unresolved_ids),
            }
        return frozenset(raw_selection), None

    @classmethod
    def _normalize_edit_units(
        cls,
        items: list[dict[str, Any]],
    ) -> tuple[
        tuple[tuple[dict[str, Any], tuple[str, ...]], ...],
        dict[str, Any] | None,
    ]:
        """Collapse exact physical duplicates, then safe nested method spans."""

        original_items = list(enumerate(items))
        evidence_order: dict[str, int] = {}
        for index, item in original_items:
            if item.get("evidence_id"):
                evidence_order.setdefault(str(item["evidence_id"]), index)

        def ordered_evidence_ids(
            units: list[tuple[dict[str, Any], tuple[str, ...]]],
        ) -> tuple[str, ...]:
            values: list[str] = []
            for item, merged_ids in units:
                if item.get("evidence_id"):
                    values.append(str(item["evidence_id"]))
                values.extend(merged_ids)
            return tuple(
                sorted(
                    dict.fromkeys(values),
                    key=lambda evidence_id: evidence_order.get(
                        evidence_id,
                        len(original_items),
                    ),
                )
            )
        hash_alias_conflicts = [
            item
            for _, item in original_items
            if cls._source_revision_hash_aliases_conflict(item)
        ]
        if hash_alias_conflicts:
            return (), cls._normalization_blocker(
                items=hash_alias_conflicts,
                message=(
                    "Editable evidence source hash aliases do not identify "
                    "one exact source revision."
                ),
            )
        missing_physical_revision = [
            item
            for _, item in original_items
            if cls._physical_source_revision_hash(item) is None
        ]
        if missing_physical_revision:
            return (), cls._normalization_blocker(
                items=missing_physical_revision,
                message=(
                    "Editable evidence requires an exact physical source "
                    "revision hash before change-unit normalization."
                ),
            )
        items_by_path: dict[str, list[dict[str, Any]]] = {}
        for _, item in original_items:
            items_by_path.setdefault(
                str(item.get("source_path") or ""),
                [],
            ).append(item)
        for path_items in items_by_path.values():
            revisions = {
                cls._physical_source_revision_hash(item)
                for item in path_items
            }
            if len(revisions) > 1:
                return (), cls._normalization_blocker(
                    items=path_items,
                    message=(
                        "Editable evidence for one source path cannot bind "
                        "multiple physical source revisions."
                    ),
                )
        paths_with_multiple_items = {
            str(item.get("source_path") or "")
            for _, item in original_items
            if sum(
                1
                for _, other in original_items
                if other.get("source_path") == item.get("source_path")
            )
            > 1
        }
        original_spans: dict[int, tuple[int, int]] = {}
        for index, item in original_items:
            path = str(item.get("source_path") or "")
            if path not in paths_with_multiple_items:
                continue
            span = cls._exact_line_span(item.get("span"))
            if span is None:
                return (), cls._normalization_blocker(
                    items=[item],
                    message=(
                        "Multiple editable evidence items for one path require "
                        "valid exact line spans before physical change-unit normalization."
                    ),
                )
            original_spans[index] = span

        duplicate_regions: dict[
            tuple[str, str, int, int],
            list[tuple[int, dict[str, Any]]],
        ] = {}
        for index, item in original_items:
            path = str(item.get("source_path") or "")
            source_hash = cls._physical_source_revision_hash(item)
            span = original_spans.get(index)
            if path and span is not None:
                duplicate_regions.setdefault(
                    (path, str(source_hash), span[0], span[1]),
                    [],
                ).append((index, item))

        consumed_indexes: set[int] = set()
        indexed_units: list[
            tuple[int, dict[str, Any], tuple[str, ...]]
        ] = []
        for region_items in duplicate_regions.values():
            if len(region_items) < 2:
                continue
            ordered_region = sorted(region_items, key=lambda entry: entry[0])
            if cls._duplicate_logical_identity_conflicts(
                [item for _, item in ordered_region]
            ):
                return (), cls._normalization_blocker(
                    items=[item for _, item in ordered_region],
                    message=(
                        "Exact duplicate editable regions contain conflicting "
                        "non-empty logical identities."
                    ),
                )
            complete_entries = [
                entry
                for entry in ordered_region
                if cls._exact_edit_identity(entry[1]) is not None
            ]
            retained_index, retained_item = (
                complete_entries[0]
                if complete_entries
                else ordered_region[0]
            )
            merged_evidence_ids = tuple(
                dict.fromkeys(
                    str(item["evidence_id"])
                    for index, item in ordered_region
                    if index != retained_index and item.get("evidence_id")
                )
            )
            indexed_units.append(
                (
                    min(index for index, _ in ordered_region),
                    retained_item,
                    merged_evidence_ids,
                )
            )
            consumed_indexes.update(index for index, _ in ordered_region)

        indexed_units.extend(
            (index, item, ())
            for index, item in original_items
            if index not in consumed_indexes
        )
        indexed_units.sort(key=lambda entry: entry[0])
        paths_with_multiple_units = {
            str(item.get("source_path") or "")
            for _, item, _ in indexed_units
            if sum(
                1
                for _, other, _ in indexed_units
                if other.get("source_path") == item.get("source_path")
            )
            > 1
        }
        spans: dict[int, tuple[int, int]] = {}
        for index, item, _ in indexed_units:
            if str(item.get("source_path") or "") not in paths_with_multiple_units:
                continue
            span = cls._exact_line_span(item.get("span"))
            if span is None:
                return (), cls._normalization_blocker(
                    items=[item],
                    message=(
                        "Multiple editable evidence items for one path require "
                        "valid exact line spans before physical change-unit normalization."
                    ),
                )
            spans[index] = span

        for position, (left_index, left, left_merged) in enumerate(indexed_units):
            for right_index, right, right_merged in indexed_units[position + 1 :]:
                if left.get("source_path") != right.get("source_path"):
                    continue
                left_span = spans[left_index]
                right_span = spans[right_index]
                if not cls._spans_overlap(left_span, right_span):
                    continue
                left_source_hash = cls._physical_source_revision_hash(left)
                right_source_hash = cls._physical_source_revision_hash(right)
                left_identity = cls._exact_edit_identity(left)
                right_identity = cls._exact_edit_identity(right)
                if (
                    left_source_hash is None
                    or left_source_hash != right_source_hash
                    or left_identity is None
                    or left_identity != right_identity
                ):
                    return (), cls._normalization_blocker(
                        items=[left, right],
                        evidence_ids=ordered_evidence_ids(
                            [
                                (left, left_merged),
                                (right, right_merged),
                            ]
                        ),
                        message=(
                            "Overlapping editable evidence for one path does not "
                            "share one complete exact source and method identity."
                        ),
                    )

        grouped: dict[
            tuple[Any, ...],
            list[tuple[int, dict[str, Any], tuple[str, ...]]],
        ] = {}
        standalone: list[tuple[int, dict[str, Any], tuple[str, ...]]] = []
        for index, item, merged_evidence_ids in indexed_units:
            identity = cls._exact_edit_identity(item)
            if identity is None or str(item.get("source_path") or "") not in (
                paths_with_multiple_units
            ):
                standalone.append((index, item, merged_evidence_ids))
                continue
            grouped.setdefault(identity, []).append(
                (index, item, merged_evidence_ids)
            )

        normalized: list[tuple[int, dict[str, Any], tuple[str, ...]]] = standalone
        for identity_items in grouped.values():
            remaining = list(identity_items)
            while remaining:
                component = [remaining.pop(0)]
                changed = True
                while changed:
                    changed = False
                    for entry in list(remaining):
                        entry_span = spans[entry[0]]
                        if any(
                            cls._spans_overlap(entry_span, spans[member[0]])
                            for member in component
                        ):
                            component.append(entry)
                            remaining.remove(entry)
                            changed = True
                if len(component) == 1:
                    normalized.append(component[0])
                    continue
                outer_entries = [
                    entry
                    for entry in component
                    if all(
                        cls._span_contains(spans[entry[0]], spans[other[0]])
                        for other in component
                    )
                ]
                if not outer_entries:
                    return (), cls._normalization_blocker(
                        items=[item for _, item, _ in component],
                        evidence_ids=ordered_evidence_ids(
                            [
                                (item, merged_ids)
                                for _, item, merged_ids in component
                            ]
                        ),
                        message=(
                            "Overlapping editable spans cannot be represented by "
                            "one independently authorized outer span."
                        ),
                    )
                retained_index, retained_item, _ = min(
                    outer_entries,
                    key=lambda entry: entry[0],
                )
                merged_evidence_ids: list[str] = []
                for index, item, existing_merged_ids in sorted(
                    component,
                    key=lambda entry: entry[0],
                ):
                    if index != retained_index and item.get("evidence_id"):
                        merged_evidence_ids.append(str(item["evidence_id"]))
                    merged_evidence_ids.extend(existing_merged_ids)
                normalized_merged_ids = tuple(
                    dict.fromkeys(merged_evidence_ids)
                )
                normalized.append(
                    (
                        min(index for index, _, _ in component),
                        retained_item,
                        normalized_merged_ids,
                    )
                )

        normalized.sort(key=lambda entry: entry[0])
        return tuple(
            (
                item,
                tuple(
                    sorted(
                        merged,
                        key=lambda evidence_id: evidence_order.get(
                            evidence_id,
                            len(original_items),
                        ),
                    )
                ),
            )
            for _, item, merged in normalized
        ), None

    @staticmethod
    def _physical_source_revision_hash(item: dict[str, Any]) -> str | None:
        value = item.get("source_hash") or item.get("file_hash")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _source_revision_hash_aliases_conflict(item: dict[str, Any]) -> bool:
        values: dict[str, str] = {}
        for field in ("source_hash", "file_hash"):
            value = item.get(field)
            if value in (None, ""):
                continue
            if not isinstance(value, str):
                return True
            values[field] = value
        doc_hash = item.get("doc_hash")
        if doc_hash not in (None, "") and not isinstance(doc_hash, str):
            return True
        return (
            "source_hash" in values
            and "file_hash" in values
            and values["source_hash"] != values["file_hash"]
        )

    @staticmethod
    def _duplicate_logical_identity_conflicts(
        items: list[dict[str, Any]],
    ) -> bool:
        for field in (
            "source_type",
            "authority",
            "symbol",
            "declaring_type",
            "declaring_type_full_name",
            "method_name",
            "return_type",
            "method_kind",
        ):
            values: set[str] = set()
            for item in items:
                value = item.get(field)
                if value in (None, ""):
                    continue
                if not isinstance(value, str):
                    return True
                values.add(value)
            if len(values) > 1:
                return True

        parameter_values: set[tuple[str, ...]] = set()
        for item in items:
            value = item.get("parameter_types")
            if value is None:
                continue
            if not isinstance(value, (list, tuple)) or not all(
                isinstance(parameter, str) for parameter in value
            ):
                return True
            parameter_values.add(tuple(value))
        return len(parameter_values) > 1

    @staticmethod
    def _exact_edit_identity(item: dict[str, Any]) -> tuple[Any, ...] | None:
        source_hash = ImplementationPlanner._physical_source_revision_hash(item)
        required_identity = (
            item.get("source_path"),
            source_hash,
            item.get("symbol"),
            item.get("declaring_type"),
            item.get("declaring_type_full_name"),
            item.get("method_name"),
            item.get("return_type"),
            item.get("method_kind"),
        )
        if any(not isinstance(value, str) or not value for value in required_identity):
            return None
        parameter_types = item.get("parameter_types")
        if parameter_types is None:
            normalized_parameters = None
        elif isinstance(parameter_types, (list, tuple)) and all(
            isinstance(value, str) for value in parameter_types
        ):
            normalized_parameters = tuple(parameter_types)
        else:
            return None
        return (*required_identity, normalized_parameters)

    @staticmethod
    def _exact_line_span(value: Any) -> tuple[int, int] | None:
        if not isinstance(value, dict):
            return None
        start_line = value.get("start_line")
        end_line = value.get("end_line")
        if (
            type(start_line) is not int
            or type(end_line) is not int
            or start_line < 1
            or end_line < start_line
        ):
            return None
        return start_line, end_line

    @staticmethod
    def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
        return left[0] <= right[1] and right[0] <= left[1]

    @staticmethod
    def _span_contains(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
        return outer[0] <= inner[0] and outer[1] >= inner[1]

    @staticmethod
    def _normalization_blocker(
        *,
        items: list[dict[str, Any]],
        message: str,
        evidence_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        return {
            "code": "EDIT_TARGET_NORMALIZATION_UNSAFE",
            "message": message,
            "evidence_ids": (
                list(evidence_ids)
                if evidence_ids is not None
                else [
                    str(item["evidence_id"])
                    for item in items
                    if item.get("evidence_id")
                ]
            ),
        }

    @staticmethod
    def _edit_binding(
        *,
        index: int,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        binding_id = f"edit_binding_{index:03d}"
        return {
            "binding_id": binding_id,
            "evidence_id": item.get("evidence_id"),
            "source_hash": ImplementationPlanner._physical_source_revision_hash(
                item
            ),
            "canonical_target_path": item.get("source_path"),
            "source_type": item.get("source_type"),
            "target_symbol": item.get("symbol"),
            "target_span": item.get("span"),
            "declaring_type": item.get("declaring_type"),
            "method_name": item.get("method_name"),
            "parameter_types": (
                list(item["parameter_types"])
                if isinstance(item.get("parameter_types"), (list, tuple))
                else None
            ),
            "return_type": item.get("return_type"),
            "method_kind": item.get("method_kind"),
            "declared_member_types": (
                dict(item["declared_member_types"])
                if isinstance(item.get("declared_member_types"), dict)
                else None
            ),
            "declaring_type_full_name": item.get("declaring_type_full_name"),
            "declared_members": (
                {
                    str(name): dict(identity)
                    for name, identity in item["declared_members"].items()
                    if isinstance(identity, dict)
                }
                if isinstance(item.get("declared_members"), dict)
                else None
            ),
            "verified_member_accesses": [
                {
                    **dict(access),
                    "edit_binding_id": binding_id,
                }
                for access in item.get("verified_member_accesses") or []
                if isinstance(access, dict)
            ],
            "retrieval_method": item.get("retrieval_method"),
            "claim_supported": item.get("claim_supported"),
            "authority": item.get("authority"),
            "usage": "edit_target_scope",
        }

    @classmethod
    def _supporting_bindings(
        cls,
        items: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        bindings: list[dict[str, Any]] = []
        hook_target_index = 1
        for item in items:
            is_hook_target = (
                item.get("source_type") == HOOK_TARGET_SOURCE_TYPE
                and item.get("usage") == HOOK_TARGET_USAGE
            )
            binding_id = (
                f"hook_target_{hook_target_index:03d}"
                if is_hook_target
                else f"support_binding_{item.get('evidence_id')}"
            )
            if is_hook_target:
                hook_target_index += 1
            bindings.append(cls._supporting_binding(item=item, binding_id=binding_id))
        return tuple(bindings)

    @staticmethod
    def _supporting_binding(
        *,
        item: dict[str, Any],
        binding_id: str,
    ) -> dict[str, Any]:
        return {
            "binding_id": binding_id,
            "evidence_id": item.get("evidence_id"),
            "source_path": item.get("source_path"),
            "source_type": item.get("source_type"),
            "target_symbol": item.get("symbol"),
            "target_span": item.get("span"),
            "declaring_type": item.get("declaring_type"),
            "method_name": item.get("method_name"),
            "parameter_types": (
                list(item["parameter_types"])
                if isinstance(item.get("parameter_types"), (list, tuple))
                else None
            ),
            "return_type": item.get("return_type"),
            "method_kind": item.get("method_kind"),
            "retrieval_method": item.get("retrieval_method"),
            "claim_supported": item.get("claim_supported"),
            "authority": item.get("authority"),
            "usage": (
                "edit_source_evidence"
                if item.get("source_type") in EDITABLE_EVIDENCE_SOURCE_TYPES
                else HOOK_TARGET_USAGE
                if item.get("source_type") == HOOK_TARGET_SOURCE_TYPE
                and item.get("usage") == HOOK_TARGET_USAGE
                else "hook_target_or_supporting_evidence"
            ),
        }

    @staticmethod
    def _binding_registry(
        *,
        edit_bindings: tuple[dict[str, Any], ...],
        supporting_bindings: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        edit_evidence_ids = {
            str(binding["evidence_id"])
            for binding in edit_bindings
            if binding.get("evidence_id")
        }
        return {
            "schema_version": "binding_registry_v1",
            "owner_artifact_type": "implementation_plan",
            "edit_bindings": [dict(binding) for binding in edit_bindings],
            "supporting_bindings": [
                dict(binding)
                for binding in supporting_bindings
                if binding.get("evidence_id")
                and str(binding["evidence_id"]) not in edit_evidence_ids
            ],
        }

    @classmethod
    def _select_hook_target_binding(
        cls,
        *,
        feature_spec: FeatureSpec,
        binding_registry: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any] | None]:
        if feature_spec.hook_requirement.get("required") is not True:
            return None, None
        hook_bindings = [
            binding
            for binding in binding_registry.get("supporting_bindings") or []
            if isinstance(binding, dict)
            and binding.get("usage") == HOOK_TARGET_USAGE
        ]
        complete_bindings = [
            binding for binding in hook_bindings if cls._has_complete_hook_target_identity(binding)
        ]
        if len(complete_bindings) == 1:
            return str(complete_bindings[0]["binding_id"]), None
        if len(complete_bindings) > 1:
            return None, {
                "code": "AMBIGUOUS_HOOK_TARGET",
                "message": (
                    "Hook-required planning found multiple complete hook targets "
                    "and no single selected target."
                ),
                "binding_ids": [
                    str(binding["binding_id"]) for binding in complete_bindings
                ],
            }
        return None, {
            "code": "MISSING_CONCRETE_HOOK_TARGET_EVIDENCE",
            "message": (
                "Hook-required planning requires one selected decompiled API "
                "method with complete target identity."
            ),
        }

    @staticmethod
    def _has_complete_hook_target_identity(binding: dict[str, Any]) -> bool:
        return (
            binding.get("source_type") == HOOK_TARGET_SOURCE_TYPE
            and binding.get("usage") == HOOK_TARGET_USAGE
            and bool(binding.get("target_symbol"))
            and bool(binding.get("target_span"))
            and bool(binding.get("declaring_type"))
            and bool(binding.get("method_name"))
            and isinstance(binding.get("parameter_types"), list)
            and bool(binding.get("return_type"))
            and bool(binding.get("method_kind"))
        )

    @staticmethod
    def _proposed_change(
        *,
        index: int,
        feature_spec: FeatureSpec,
        hook_target_binding_id: str | None,
        supporting_binding_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        change = {
            "change_id": f"chg_{index:03d}",
            "change_type": "modify_behavior",
            "edit_binding_id": f"edit_binding_{index:03d}",
            "supporting_binding_ids": list(supporting_binding_ids),
            "operation": "minimal_evidence_bound_change",
            "rationale": (
                f"Contribute within edit_binding_{index:03d} to the Feature-wide "
                f"target behavior without assigning it to this change: "
                f"{feature_spec.target_behavior}"
            ),
            "risk_tags": [feature_spec.risk_level],
            "expected_validation": [
                "Validate this bounded edit against the Feature-wide test_plan."
            ],
            "rollback_hint": "Revert this change unit or restore previous behavior.",
        }
        if hook_target_binding_id is not None:
            change["hook_target_binding_id"] = hook_target_binding_id
        return change

    @staticmethod
    def _blocked(
        *,
        feature_id: str,
        blockers: tuple[dict[str, Any], ...],
        open_questions: tuple[str, ...],
        evidence_source_paths: tuple[str, ...] = (),
    ) -> ImplementationPlan:
        return ImplementationPlan(
            feature_id=feature_id,
            target_files=(),
            target_symbols=(),
            evidence_refs=(),
            proposed_changes=(),
            side_effect_boundaries=(),
            test_plan=(),
            rollback_plan=(),
            open_questions=open_questions,
            blocking=True,
            blockers=blockers,
            evidence_source_paths=evidence_source_paths,
        )
