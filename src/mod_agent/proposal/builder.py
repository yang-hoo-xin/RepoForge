"""Build reviewable patch proposals from implementation plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.mod_agent.bindings import BindingResolver

from .models import (
    PatchProposal,
    build_change_units,
    build_context_pack,
    build_no_op_status,
    build_rollback_hints,
    stable_proposal_id,
)


def _not_configured_git_binding() -> dict[str, Any]:
    return {
        "status": "not_configured",
        "base_revision": None,
    }


class PatchProposalBuilder:
    """Convert an implementation plan into patch intent without editing files."""

    def build_v2(self, handoff, *, feature_id, feature_spec_ref, store):
        """Build directly from verified code and authority, without a synthetic Plan.

        为什么要有它：v2当前代码与绑定不归技术计划所有，产品必须直接消费同一候选。
        """
        from src.mod_agent.task_agent_v2_handoff import revalidate_handoff_v2
        selected = revalidate_handoff_v2(handoff.to_dict(), store=store)
        changes = selected.changes
        refs = (str(selected.task_contract_ref), str(selected.check_ref))
        return PatchProposal(proposal_id=stable_proposal_id(feature_id, (selected.candidate_source_id,)),
            feature_id=feature_id, summary="Validated v2 candidate awaiting delivery.",
            files=tuple(row["display_file"] for row in changes), diff_intent=changes,
            change_units=build_change_units(changes), context_pack={"task_agent_protocol": "v2",
                "task_agent_handoff_v2": selected.to_dict(), "feature_spec_ref": feature_spec_ref,
                "artifact_hashes": {"feature_spec": feature_spec_ref["artifact_hash"]}},
            evidence_refs=refs, scope_boundary=("Existing files within task authority only.",),
            risk_assessment={"risk_tags": [], "risk_level": "medium", "requires_manual_review": True},
            expected_tests=("Version-bound public candidate acceptance",),
            no_op_status=build_no_op_status(is_no_op=False, reason=None),
            rollback_hints=build_rollback_hints(changes))

    @staticmethod
    def required_source_snippet_char_limit(
        implementation_plan: Mapping[str, Any],
        evidence_pack: Mapping[str, Any],
    ) -> int:
        """Size review excerpts for exactly the Plan's already-admitted refs."""
        evidence_by_id = {
            str(item["evidence_id"]): item
            for item in evidence_pack.get("items") or []
            if isinstance(item, Mapping) and item.get("evidence_id")
        }
        return max(
            1,
            sum(
                len(excerpt)
                for evidence_id in implementation_plan.get("evidence_refs") or []
                if isinstance(
                    excerpt := evidence_by_id.get(str(evidence_id), {}).get("excerpt"),
                    str,
                )
            ),
        )

    def build(
        self,
        implementation_plan: dict[str, Any],
        *,
        evidence_pack: dict[str, Any] | None = None,
        artifact_hashes: dict[str, str] | None = None,
        git_binding: dict[str, Any] | None = None,
        require_git_binding: bool = False,
        source_snippet_char_limit: int = 6000,
    ) -> PatchProposal:
        """Return a patch proposal or a blocked proposal."""

        feature_id = str(implementation_plan.get("feature_id") or "unknown_feature")
        evidence_refs = tuple(implementation_plan.get("evidence_refs") or ())

        if implementation_plan.get("blocking"):
            return self._blocked(
                feature_id=feature_id,
                evidence_refs=evidence_refs,
                blockers=tuple(implementation_plan.get("blockers") or ()),
                note="Implementation plan is blocking.",
            )

        proposed_changes = list(implementation_plan.get("proposed_changes") or [])
        if not proposed_changes:
            return self._blocked(
                feature_id=feature_id,
                evidence_refs=evidence_refs,
                blockers=(
                    {
                        "code": "NO_OP_PROPOSAL",
                        "message": "Implementation plan contains no proposed changes.",
                    },
                ),
                note="No change units were present.",
            )

        unresolved_changes = [
            change
            for change in proposed_changes
            if not change.get("edit_binding_id")
        ]
        resolver = BindingResolver.from_implementation_plan(
            implementation_plan,
            evidence_pack=evidence_pack,
        )
        edit_binding_ids = [
            str(change["edit_binding_id"])
            for change in proposed_changes
            if change.get("edit_binding_id")
        ]
        display_files = tuple(resolver.display_files_for_edit_ids(edit_binding_ids))
        files = display_files
        if not files or unresolved_changes:
            return self._blocked(
                feature_id=feature_id,
                evidence_refs=evidence_refs,
                blockers=(
                    {
                        "code": "EDIT_TARGET_UNRESOLVED",
                        "message": (
                            "Patch proposal requires implementation_plan edit_binding_id "
                            "values resolved before review."
                        ),
                    },
                ),
                note="Implementation plan did not resolve edit targets.",
            )

        expected_tests = tuple(implementation_plan.get("test_plan") or ())
        diff_intent = tuple(self._change_intent(change) for change in proposed_changes)
        change_units = build_change_units(diff_intent)
        scope_boundary = tuple(implementation_plan.get("side_effect_boundaries") or ())
        evidence_items = self._evidence_items_by_id(evidence_pack)
        source_snippets, snippet_budget, snippet_blockers = self._source_snippets(
            evidence_refs=evidence_refs,
            evidence_items=evidence_items,
            char_limit=source_snippet_char_limit,
        )
        symbol_signatures, symbol_blockers = self._target_symbol_signatures(
            implementation_plan=implementation_plan,
        )
        effective_artifact_hashes = dict(artifact_hashes or {})
        omission_blockers = (
            snippet_blockers
            + symbol_blockers
            + self._artifact_hash_blockers(effective_artifact_hashes)
            + self._git_binding_blockers(git_binding, required=require_git_binding)
        )
        context_pack = build_context_pack(
            feature_id=feature_id,
            files=files,
            evidence_refs=evidence_refs,
            evidence_source_paths=tuple(
                implementation_plan.get("evidence_source_paths") or ()
            ),
            scope_boundary=scope_boundary,
            change_units=change_units,
            source_snippets=source_snippets,
            target_symbol_signatures=symbol_signatures,
            binding_registry_ref={
                "artifact_type": "implementation_plan",
                "artifact_hash": effective_artifact_hashes.get("implementation_plan"),
                "schema_version": "binding_registry_v1",
            },
            display_files=display_files,
            allowed_edit_binding_ids=tuple(edit_binding_ids),
            artifact_hashes={
                key: value
                for key, value in effective_artifact_hashes.items()
                if key in {"feature_spec", "evidence_pack", "implementation_plan"}
            },
            implementation_plan_hash=effective_artifact_hashes.get(
                "implementation_plan"
            ),
            git_binding=git_binding or _not_configured_git_binding(),
            git_binding_required=require_git_binding,
            context_budget=snippet_budget,
            omission_blockers=omission_blockers,
        )
        risk_tags = sorted(
            {
                risk
                for change in proposed_changes
                for risk in change.get("risk_tags", [])
                if isinstance(risk, str)
            }
        )

        return PatchProposal(
            proposal_id=stable_proposal_id(feature_id, evidence_refs),
            feature_id=feature_id,
            summary=f"Proposal for {feature_id} with {len(diff_intent)} change unit(s).",
            files=files,
            diff_intent=diff_intent,
            change_units=change_units,
            context_pack=context_pack,
            evidence_refs=evidence_refs,
            scope_boundary=scope_boundary,
            risk_assessment={
                "risk_tags": risk_tags,
                "risk_level": self._risk_level(risk_tags),
                "requires_manual_review": "high" in risk_tags,
            },
            expected_tests=expected_tests,
            no_op_status=build_no_op_status(is_no_op=False, reason=None),
            rollback_hints=build_rollback_hints(diff_intent),
            manual_review_notes=tuple(
                "High-risk proposal requires human review before real apply."
                for _ in [0]
                if "high" in risk_tags
            ),
            blocked=False,
        )

    @staticmethod
    def _change_intent(change: dict[str, Any]) -> dict[str, Any]:
        return {
            "change_id": change.get("change_id"),
            "change_type": change.get("change_type"),
            "edit_binding_id": change.get("edit_binding_id"),
            "hook_target_binding_id": change.get("hook_target_binding_id"),
            "supporting_binding_ids": list(change.get("supporting_binding_ids") or []),
            "operation": change.get("operation"),
            "rationale": change.get("rationale"),
            "evidence_refs": list(change.get("evidence_refs") or []),
            "expected_validation": list(change.get("expected_validation") or []),
            "rollback_hint": change.get("rollback_hint"),
            "patch": dict(change.get("patch") or {}),
            "generation": dict(change.get("generation") or {}),
        }

    @staticmethod
    def _evidence_items_by_id(
        evidence_pack: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(evidence_pack, dict):
            return {}
        return {
            str(item["evidence_id"]): item
            for item in evidence_pack.get("items", [])
            if isinstance(item, dict) and item.get("evidence_id")
        }

    @staticmethod
    def _source_snippets(
        *,
        evidence_refs: tuple[str, ...],
        evidence_items: dict[str, dict[str, Any]],
        char_limit: int,
    ) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], tuple[dict[str, Any], ...]]:
        snippets: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        used_chars = 0
        omitted = 0
        for evidence_ref in evidence_refs:
            item = evidence_items.get(evidence_ref)
            excerpt = item.get("excerpt") if isinstance(item, dict) else None
            if not isinstance(item, dict) or not isinstance(excerpt, str) or not excerpt:
                blockers.append(
                    {
                        "code": "SOURCE_SNIPPET_MISSING",
                        "message": "Proposal context requires a source snippet for every evidence ref.",
                        "evidence_ref": evidence_ref,
                    }
                )
                continue
            remaining = max(char_limit - used_chars, 0)
            if remaining <= 0:
                omitted += 1
                blockers.append(
                    {
                        "code": "SOURCE_SNIPPET_BUDGET_EXCEEDED",
                        "message": "Proposal context source snippet budget was exceeded.",
                        "evidence_ref": evidence_ref,
                    }
                )
                continue
            snippet_text = excerpt[:remaining]
            used_chars += len(snippet_text)
            snippets.append(
                {
                    "evidence_id": evidence_ref,
                    "source_path": item.get("source_path"),
                    "source_type": item.get("source_type"),
                    "symbol": item.get("symbol"),
                    "span": item.get("span"),
                    "retrieval_method": item.get("retrieval_method"),
                    "excerpt": snippet_text,
                    "truncated": len(snippet_text) < len(excerpt),
                    "chunk_id": item.get("chunk_id"),
                    "document_id": item.get("document_id"),
                    "source_ref": item.get("source_ref"),
                    "source_hash": item.get("source_hash"),
                    "file_hash": item.get("file_hash"),
                    "doc_hash": item.get("doc_hash"),
                }
            )
        budget = {
            "source_snippet_count": len(snippets),
            "source_snippet_chars": used_chars,
            "source_snippet_char_limit": char_limit,
            "omitted_source_snippet_count": omitted,
        }
        return tuple(snippets), budget, tuple(blockers)

    @staticmethod
    def _target_symbol_signatures(
        *,
        implementation_plan: dict[str, Any],
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        signatures: list[dict[str, Any]] = []
        registry = implementation_plan.get("binding_registry")
        if not isinstance(registry, dict):
            return (), (
                {
                    "code": "BINDING_REGISTRY_MISSING",
                    "message": "Proposal context requires implementation_plan binding_registry.",
                },
            )
        for binding in registry.get("edit_bindings") or []:
            if not isinstance(binding, dict) or not binding.get("target_symbol"):
                continue
            method_name = binding.get("method_name")
            return_type = binding.get("return_type")
            parameter_types = binding.get("parameter_types")
            if method_name and return_type and isinstance(parameter_types, list):
                signature = (
                    f"{return_type} {method_name}("
                    + ", ".join(str(item) for item in parameter_types)
                    + ")"
                )
            else:
                signature = str(binding["target_symbol"])
            signatures.append(
                {
                    "symbol": str(binding["target_symbol"]),
                    "binding_id": binding.get("binding_id"),
                    "signature": signature,
                    "signature_source": "binding_registry",
                    "evidence_ids": [binding.get("evidence_id")],
                }
            )
        return tuple(signatures), ()

    @staticmethod
    def _artifact_hash_blockers(
        artifact_hashes: dict[str, str],
    ) -> tuple[dict[str, Any], ...]:
        blockers: list[dict[str, Any]] = []
        for artifact_type in ("evidence_pack", "implementation_plan"):
            if not artifact_hashes.get(artifact_type):
                blockers.append(
                    {
                        "code": "CONTEXT_ARTIFACT_HASH_MISSING",
                        "message": "Proposal context requires upstream artifact hashes.",
                        "artifact_type": artifact_type,
                    }
                )
        return tuple(blockers)

    @staticmethod
    def _git_binding_blockers(
        git_binding: dict[str, Any] | None,
        *,
        required: bool,
    ) -> tuple[dict[str, Any], ...]:
        if not required:
            return ()
        if isinstance(git_binding, dict) and git_binding.get("base_revision"):
            return ()
        return (
            {
                "code": "GIT_BASE_REVISION_MISSING",
                "message": "Proposal context requires a proposal-time Git base revision.",
            },
        )

    @staticmethod
    def _risk_level(risk_tags: list[str]) -> str:
        if "high" in risk_tags:
            return "high"
        if "medium" in risk_tags:
            return "medium"
        return "low"

    @staticmethod
    def _blocked(
        *,
        feature_id: str,
        evidence_refs: tuple[str, ...],
        blockers: tuple[dict[str, Any], ...],
        note: str,
        ) -> PatchProposal:
        no_op = note == "No change units were present."
        return PatchProposal(
            proposal_id=stable_proposal_id(feature_id, evidence_refs),
            feature_id=feature_id,
            summary="Patch proposal blocked.",
            files=(),
            diff_intent=(),
            change_units=(),
            context_pack=build_context_pack(
                feature_id=feature_id,
                files=(),
                evidence_refs=evidence_refs,
                evidence_source_paths=(),
                scope_boundary=(),
                change_units=(),
                omission_blockers=(
                    {
                        "code": "BLOCKED_PROPOSAL_CONTEXT",
                        "message": note,
                    },
                ),
            ),
            evidence_refs=evidence_refs,
            scope_boundary=(),
            risk_assessment={"risk_level": "blocked", "requires_manual_review": True},
            expected_tests=(),
            no_op_status=build_no_op_status(is_no_op=no_op, reason=note),
            rollback_hints=(),
            manual_review_notes=(note,),
            blocked=True,
            blockers=blockers,
        )
