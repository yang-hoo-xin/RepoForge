"""Rule-based patch proposal review gate."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from src.mod_agent.bindings import BindingResolver
from src.mod_agent.domain import DomainReviewPolicy

from .critic import (
    CRITIC_AGGREGATION,
    CRITIC_APPLICABILITY,
    CRITIC_DIMENSIONS,
    CRITIC_PASS_THRESHOLD,
    CRITIC_POLICY_VERSION,
)
from .models import ProposalReview, ReviewFinding, SemanticAssuranceV1

CHECKED_RULES = (
    "schema_validity",
    "context_pack",
    "plan_change_coverage",
    "binding_source_locality",
    "evidence_binding",
    "hook_target_binding",
    "scope_boundary",
    "risk_policy",
    "domain_profile",
    "forbidden_paths",
    "permission_level",
    "test_plan_presence",
    "validation_contract",
    "no_op_detection",
    "executable_patch",
    "semantic_assurance",
)


class ProposalReviewer:
    """Evaluate a patch proposal before dry-run apply is allowed."""

    def review_v2(self, proposal, *, store, feature_spec_ref):
        """Review exact admitted v2 content, retaining semantic uncertainty.

        为什么要有它：公开检查不是完整语义证明，Review仍核验候选一致性且不授予Apply权限。
        """
        from src.mod_agent.model_qualification import canonical_hash
        from src.mod_agent.task_agent_v2_handoff import revalidate_handoff_v2
        pack = proposal["context_pack"]
        selected = revalidate_handoff_v2(pack["task_agent_handoff_v2"], store=store)
        if (pack.get("feature_spec_ref") != feature_spec_ref
                or canonical_hash(proposal["diff_intent"]) != canonical_hash(selected.changes)
                or proposal["files"] != [row["display_file"] for row in selected.changes]):
            raise ValueError("V2_PROPOSAL_CANDIDATE_CHANGED")
        return ProposalReview(decision="approve_dry_run", findings=(), required_changes=(), allowed_next_stage="dry_run_apply",
            checked_rules=("v2_source_authority", "v2_model_provenance", "v2_candidate_check", "v2_exact_changes"),
            trace_metrics={"checked_change_count": len(selected.changes)},
            critic_gate={"required": False, "status": "not_requested"},
            human_semantic_review_required=True)

    def review(
        self,
        proposal: dict[str, Any],
        *,
        evidence_pack: dict[str, Any] | None = None,
        repair_plan: dict[str, Any] | None = None,
        implementation_plan: dict[str, Any] | None = None,
        feature_spec: dict[str, Any] | None = None,
        artifact_hashes: dict[str, str] | None = None,
        critic_assessment: dict[str, Any] | None = None,
        require_critic_assessment: bool = False,
        domain_policy: DomainReviewPolicy | None = None,
        permission_level: str = "dry_run_write",
        require_executable_patch: bool = False,
    ) -> ProposalReview:
        """Return a composite review decision."""

        findings: list[ReviewFinding] = []
        diff_intent = list(proposal.get("diff_intent") or [])
        change_units = list(proposal.get("change_units") or [])
        context_pack = (
            proposal.get("context_pack")
            if isinstance(proposal.get("context_pack"), dict)
            else {}
        )
        no_op_status = (
            proposal.get("no_op_status")
            if isinstance(proposal.get("no_op_status"), dict)
            else {}
        )
        evidence_ids = self._evidence_ids(evidence_pack, repair_plan)
        proposal_evidence_refs = set(proposal.get("evidence_refs") or [])
        allowed_files = set(proposal.get("files") or [])
        checked_rules = CHECKED_RULES
        semantic_assurance, semantic_contract_finding = (
            self._semantic_assurance_from_proposal(proposal)
        )
        if semantic_contract_finding is not None:
            findings.append(semantic_contract_finding)
        if semantic_assurance.verdict == "violated":
            findings.append(
                ReviewFinding(
                    rule_id="semantic_assurance_violation",
                    level="blocking",
                    message=(
                        "A bound semantic verifier or human review reported that "
                        "the proposal violates the declared Feature boundary."
                    ),
                    evidence_refs=semantic_assurance.evidence_refs,
                )
            )
        elif semantic_assurance.verdict == "unproven":
            findings.append(
                ReviewFinding(
                    rule_id="semantic_assurance",
                    level="warning",
                    message=(
                        "Feature-to-code semantics are unproven; human semantic "
                        "review of the exact dry-run diff is required before "
                        "isolated apply."
                    ),
                    evidence_refs=semantic_assurance.evidence_refs,
                )
            )

        critic_gate = self._critic_gate(
            critic_assessment=critic_assessment,
            artifact_hashes=artifact_hashes,
            required=require_critic_assessment,
        )
        if critic_gate["state"] != "not_required":
            checked_rules = (*checked_rules, "proposal_critic")
            findings.extend(
                self._review_critic_assessment(
                    critic_assessment=critic_assessment,
                    critic_gate=critic_gate,
                )
            )

        if proposal.get("blocked"):
            findings.append(
                ReviewFinding(
                    rule_id="schema_validity",
                    level="blocking",
                    message="Proposal is already marked blocked.",
                )
            )

        if not diff_intent:
            findings.append(
                ReviewFinding(
                    rule_id="no_op_detection",
                    level="blocking",
                    message="Proposal has no change intent and would be a no-op.",
                )
            )

        if not context_pack:
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal context_pack is missing or malformed.",
                )
            )

        if diff_intent and not change_units:
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal change_units are missing for diff intent.",
                )
            )

        if context_pack and set(context_pack.get("files") or []) != allowed_files:
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal context_pack files do not match proposal files.",
                )
            )

        if context_pack and set(context_pack.get("evidence_refs") or []) != proposal_evidence_refs:
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message=(
                        "Proposal context_pack evidence refs do not match proposal "
                        "evidence refs."
                    ),
                )
            )

        if self._change_unit_ids(diff_intent) != self._declared_change_unit_ids(change_units):
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal change_units do not match diff_intent change ids.",
                )
            )

        if (
            context_pack
            and self._context_change_unit_ids(context_pack)
            != self._declared_change_unit_ids(change_units)
        ):
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message=(
                        "Proposal context_pack change_unit_ids do not match "
                        "proposal change_units."
                    ),
                )
            )

        findings.extend(
            self._review_context_pack_depth(
                proposal=proposal,
                context_pack=context_pack,
                evidence_pack=evidence_pack,
                repair_plan=repair_plan,
                implementation_plan=implementation_plan,
                artifact_hashes=artifact_hashes,
                domain_policy=domain_policy,
            )
        )
        findings.extend(
            self._review_plan_change_coverage(
                proposal=proposal,
                implementation_plan=implementation_plan,
                repair_plan=repair_plan,
            )
        )
        findings.extend(
            self._review_binding_source_locality(
                diff_intent=diff_intent,
                implementation_plan=implementation_plan,
                evidence_pack=evidence_pack,
            )
        )
        findings.extend(
            self._review_hook_target_bindings(
                diff_intent=diff_intent,
                implementation_plan=implementation_plan,
                evidence_pack=evidence_pack,
                feature_spec=feature_spec,
                repair_plan=repair_plan,
            )
        )

        if no_op_status.get("is_no_op") is True and diff_intent:
            findings.append(
                ReviewFinding(
                    rule_id="no_op_detection",
                    level="blocking",
                    message="Proposal declares no-op status while carrying diff intent.",
                )
            )

        if permission_level != "dry_run_write":
            findings.append(
                ReviewFinding(
                    rule_id="permission_level",
                    level="blocking",
                    message="Only dry_run_write proposals may advance to dry-run apply.",
                )
            )

        if not proposal.get("expected_tests"):
            findings.append(
                ReviewFinding(
                    rule_id="test_plan_presence",
                    level="blocking",
                    message="Proposal must declare expected tests or validation steps.",
                )
            )

        findings.extend(
            self._review_validation_contract(
                diff_intent=diff_intent,
                implementation_plan=implementation_plan,
            )
        )

        if require_executable_patch:
            findings.extend(self._review_executable_patch(diff_intent))

        if proposal.get("risk_assessment", {}).get("requires_manual_review"):
            findings.append(
                ReviewFinding(
                    rule_id="risk_policy",
                    level="warning",
                    message="Proposal declares manual review requirement before real apply.",
                )
            )

        if domain_policy is not None and self._requires_game_profile(proposal):
            if not domain_policy.allows_runtime_change():
                findings.append(
                    ReviewFinding(
                        rule_id="domain_profile",
                        level="blocking",
                        message=(
                            "Game-specific runtime changes are blocked without "
                            "an experimental or supported GameProfile."
                        ),
                    )
                )

        for change in diff_intent:
            change_id = change.get("change_id")
            refs = tuple(change.get("evidence_refs") or ())
            target_file = change.get("target_file")
            binding_id = change.get("edit_binding_id")
            if not refs and not binding_id:
                findings.append(
                    ReviewFinding(
                        rule_id="evidence_binding",
                        level="blocking",
                        message="Change unit has no evidence refs or edit binding id.",
                        change_id=change_id,
                    )
                )
            elif refs and not evidence_ids:
                findings.append(
                    ReviewFinding(
                        rule_id="evidence_binding",
                        level="blocking",
                        message=(
                            "Change unit evidence refs cannot be validated without "
                            "an evidence pack or repair plan."
                        ),
                        change_id=change_id,
                        evidence_refs=refs,
                    )
                )
            elif refs and evidence_ids and not set(refs).issubset(evidence_ids):
                findings.append(
                    ReviewFinding(
                        rule_id="evidence_binding",
                        level="blocking",
                        message=(
                            "Change unit references evidence not present in the "
                            "approved evidence source."
                        ),
                        change_id=change_id,
                        evidence_refs=refs,
                    )
                )
            elif refs and not set(refs).issubset(proposal_evidence_refs):
                findings.append(
                    ReviewFinding(
                        rule_id="evidence_binding",
                        level="blocking",
                        message="Change unit evidence refs are not declared at proposal level.",
                        change_id=change_id,
                        evidence_refs=refs,
                    )
                )

            if target_file and target_file not in allowed_files:
                findings.append(
                    ReviewFinding(
                        rule_id="scope_boundary",
                        level="blocking",
                        message="Change unit target file is outside proposal file list.",
                        change_id=change_id,
                    )
                )

            if target_file and self._is_forbidden_path(str(target_file)):
                findings.append(
                    ReviewFinding(
                        rule_id="forbidden_paths",
                        level="blocking",
                        message="Change unit targets a forbidden or unsafe path.",
                        change_id=change_id,
                    )
                )
            if target_file and domain_policy is not None:
                domain_path = domain_policy.review_path(str(target_file))
                if not domain_path.allowed:
                    findings.append(
                        ReviewFinding(
                            rule_id=domain_path.rule_id or "forbidden_paths",
                            level="blocking",
                            message=domain_path.reason,
                            change_id=change_id,
                        )
                    )

        planned_change_count = len(
            (
                implementation_plan.get("proposed_changes")
                if isinstance(implementation_plan, dict)
                else []
            )
            or []
        )
        blocking = [finding for finding in findings if finding.level == "blocking"]
        if blocking:
            decision = self._blocked_decision(blocking)
            return ProposalReview(
                decision=decision,
                findings=tuple(findings),
                required_changes=tuple(finding.message for finding in blocking),
                allowed_next_stage=None,
                checked_rules=checked_rules,
                trace_metrics={
                    "num_changes_checked": len(diff_intent),
                    "num_plan_changes_expected": planned_change_count,
                    "num_blocking_findings": len(blocking),
                },
                critic_gate=critic_gate,
                semantic_assurance=semantic_assurance,
                human_semantic_review_required=(
                    semantic_assurance.verdict == "unproven"
                ),
            )

        return ProposalReview(
            decision="approve_dry_run",
            findings=tuple(findings),
            required_changes=(),
            allowed_next_stage="dry_run_apply",
            checked_rules=checked_rules,
            trace_metrics={
                "num_changes_checked": len(diff_intent),
                "num_plan_changes_expected": planned_change_count,
                "num_blocking_findings": 0,
            },
            critic_gate=critic_gate,
            semantic_assurance=semantic_assurance,
            human_semantic_review_required=(
                semantic_assurance.verdict == "unproven"
            ),
        )

    @staticmethod
    def _semantic_assurance_from_proposal(
        proposal: dict[str, Any],
    ) -> tuple[SemanticAssuranceV1, ReviewFinding | None]:
        if "semantic_assurance" not in proposal:
            return SemanticAssuranceV1.unproven(), None
        value = proposal.get("semantic_assurance")
        if not isinstance(value, dict):
            return (
                SemanticAssuranceV1.unproven(),
                ReviewFinding(
                    rule_id="semantic_assurance_contract",
                    level="blocking",
                    message="Proposal semantic_assurance is missing or malformed.",
                ),
            )
        try:
            assurance = SemanticAssuranceV1.from_dict(value)
        except ValueError as exc:
            return (
                SemanticAssuranceV1.unproven(),
                ReviewFinding(
                    rule_id="semantic_assurance_contract",
                    level="blocking",
                    message=f"Proposal semantic_assurance is invalid: {exc}",
                ),
            )
        if assurance.verdict != "unproven":
            return (
                SemanticAssuranceV1.unproven(),
                ReviewFinding(
                    rule_id="semantic_assurance_untrusted_proposal_claim",
                    level="blocking",
                    message=(
                        "Semantic Assurance V0 has no independently registered "
                        "verifier owner; a Proposal cannot establish a satisfied "
                        "or violated semantic verdict from its own payload."
                    ),
                    evidence_refs=assurance.evidence_refs,
                ),
            )
        return assurance, None

    @staticmethod
    def _review_executable_patch(
        diff_intent: list[dict[str, Any]],
    ) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        for index, change in enumerate(diff_intent, start=1):
            patch = change.get("patch")
            before = patch.get("before_text") if isinstance(patch, dict) else None
            after = patch.get("after_text") if isinstance(patch, dict) else None
            if (
                not isinstance(patch, dict)
                or patch.get("kind") != "exact_text_replace"
                or not isinstance(before, str)
                or not before
                or not isinstance(after, str)
                or not after
                or before == after
                or patch.get("expected_occurrences") != 1
            ):
                findings.append(
                    ReviewFinding(
                        rule_id="executable_patch",
                        level="blocking",
                        message=f"Change {index} has invalid or no-op executable patch data.",
                        change_id=change.get("change_id"),
                    )
                )
        return findings

    @staticmethod
    def _review_validation_contract(
        *,
        diff_intent: list[dict[str, Any]],
        implementation_plan: dict[str, Any] | None,
    ) -> list[ReviewFinding]:
        if not isinstance(implementation_plan, dict):
            return []
        planned_changes = {
            str(change["edit_binding_id"]): change
            for change in implementation_plan.get("proposed_changes") or []
            if isinstance(change, dict) and change.get("edit_binding_id")
        }
        findings: list[ReviewFinding] = []
        for change in diff_intent:
            change_id = str(change.get("change_id") or "")
            binding_id = str(change.get("edit_binding_id") or "")
            planned = planned_changes.get(binding_id)
            if planned is None:
                continue
            required = [str(item) for item in planned.get("expected_validation") or []]
            declared = {
                str(item) for item in change.get("expected_validation") or []
            }
            missing = [item for item in required if item not in declared]
            if missing:
                findings.append(
                    ReviewFinding(
                        rule_id="validation_contract",
                        level="blocking",
                        message=(
                            "Proposal change omits implementation-plan validation "
                            "requirement(s): " + ", ".join(missing)
                        ),
                        change_id=change_id or None,
                    )
                )
        return findings

    @staticmethod
    def _evidence_ids(
        evidence_pack: dict[str, Any] | None,
        repair_plan: dict[str, Any] | None,
    ) -> set[str]:
        evidence_ids = set()
        if evidence_pack is not None:
            evidence_ids.update(
                {
                    item["evidence_id"]
                    for item in evidence_pack.get("items", [])
                    if item.get("evidence_id")
                }
            )
        if repair_plan is not None:
            evidence_ids.update(str(ref) for ref in repair_plan.get("evidence_refs", []))
        return evidence_ids

    @staticmethod
    def _change_unit_ids(diff_intent: list[dict[str, Any]]) -> set[str]:
        return {
            str(change.get("change_id") or f"change_{index:03d}")
            for index, change in enumerate(diff_intent, start=1)
        }

    @staticmethod
    def _declared_change_unit_ids(change_units: list[Any]) -> set[str]:
        return {
            str(unit.get("change_unit_id"))
            for unit in change_units
            if isinstance(unit, dict) and unit.get("change_unit_id")
        }

    @staticmethod
    def _context_change_unit_ids(context_pack: dict[str, Any]) -> set[str]:
        return {str(value) for value in context_pack.get("change_unit_ids") or []}

    @staticmethod
    def _critic_gate(
        *,
        critic_assessment: dict[str, Any] | None,
        artifact_hashes: dict[str, str] | None,
        required: bool,
    ) -> dict[str, Any]:
        base = {
            "requirement": "required" if required else "not_required",
            "policy_version": CRITIC_POLICY_VERSION,
            "applicability": CRITIC_APPLICABILITY,
            "aggregation_method": CRITIC_AGGREGATION,
            "pass_threshold": CRITIC_PASS_THRESHOLD,
            "hard_rule_priority": "proposal_reviewer_hard_rules_first",
        }
        if not required:
            return {**base, "state": "not_required", "route_semantics": "normal_review"}
        if critic_assessment is None or critic_assessment.get("assessor_kind") == "unresolved":
            return {**base, "state": "required_failed", "route_semantics": "blocked"}
        expected_hash = (artifact_hashes or {}).get("patch_proposal")
        scores = critic_assessment.get("dimension_scores")
        values = [
            *(scores.values() if isinstance(scores, dict) else []),
            critic_assessment.get("total_score"),
        ]
        expected_weights = {dimension: 25 for dimension in CRITIC_DIMENSIONS}
        structurally_valid = (
            critic_assessment.get("independence_boundary") == "read_only_no_route_authority"
            and bool(expected_hash)
            and critic_assessment.get("subject_artifact_hash") == expected_hash
            and critic_assessment.get("policy_version") == CRITIC_POLICY_VERSION
            and critic_assessment.get("applicability") == CRITIC_APPLICABILITY
            and critic_assessment.get("aggregation_method") == CRITIC_AGGREGATION
            and critic_assessment.get("policy_threshold") == CRITIC_PASS_THRESHOLD
            and critic_assessment.get("dimension_weights") == expected_weights
            and critic_assessment.get("verdict") in {"pass", "revise", "block"}
            and isinstance(scores, dict)
            and set(scores) == set(CRITIC_DIMENSIONS)
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 100
                for value in values
            )
            and critic_assessment.get("total_score")
            == round(sum(scores.values()) / len(CRITIC_DIMENSIONS))
            and isinstance(critic_assessment.get("findings"), list)
        )
        return {
            **base,
            "state": "required_succeeded" if structurally_valid else "required_failed",
            "route_semantics": "score_policy" if structurally_valid else "blocked",
        }

    @staticmethod
    def _review_critic_assessment(
        *,
        critic_assessment: dict[str, Any] | None,
        critic_gate: dict[str, Any],
    ) -> tuple[ReviewFinding, ...]:
        if critic_gate["state"] == "required_failed":
            return (
                ReviewFinding(
                    rule_id="proposal_critic_unavailable",
                    level="blocking",
                    message="Required ProposalCritic assessment failed or is invalid.",
                ),
            )
        if critic_gate["state"] == "not_required" or critic_assessment is None:
            return ()
        findings: list[ReviewFinding] = []
        critic_findings = critic_assessment.get("findings")
        if any(
            isinstance(item, dict) and item.get("level") == "blocking"
            for item in critic_findings or []
        ):
            findings.append(
                ReviewFinding(
                    rule_id="proposal_critic",
                    level="blocking",
                    message="ProposalCritic reported a blocking finding.",
                )
            )

        verdict = critic_assessment.get("verdict")
        if verdict == "block":
            findings.append(
                ReviewFinding(
                    rule_id="proposal_critic_block",
                    level="blocking",
                    message="ProposalCritic blocked the proposal.",
                )
            )
        elif verdict != "pass":
            findings.append(
                ReviewFinding(
                    rule_id="proposal_critic",
                    level="blocking",
                    message=(
                        "ProposalCritic did not pass the proposal: "
                        f"{critic_assessment.get('verdict')!r}."
                    ),
                )
            )
        total_score = critic_assessment.get("total_score")
        if (
            isinstance(total_score, int)
            and not isinstance(total_score, bool)
            and total_score < CRITIC_PASS_THRESHOLD
        ):
            findings.append(
                ReviewFinding(
                    rule_id="proposal_critic",
                    level="blocking",
                    message=(
                        "ProposalCritic score is below the deterministic "
                        f"threshold of {CRITIC_PASS_THRESHOLD}."
                    ),
                )
            )
        return tuple(findings)

    @staticmethod
    def _review_plan_change_coverage(
        *,
        proposal: dict[str, Any],
        implementation_plan: dict[str, Any] | None,
        repair_plan: dict[str, Any] | None,
    ) -> tuple[ReviewFinding, ...]:
        if repair_plan is not None or not isinstance(implementation_plan, dict):
            return ()
        planned = [
            (
                str(change.get("change_id") or ""),
                str(change.get("edit_binding_id") or ""),
            )
            for change in implementation_plan.get("proposed_changes") or []
            if isinstance(change, dict)
        ]
        proposed = [
            (
                str(change.get("change_id") or ""),
                str(change.get("edit_binding_id") or ""),
            )
            for change in proposal.get("diff_intent") or []
            if isinstance(change, dict)
        ]
        missing = [pair for pair in planned if pair not in proposed]
        unexpected = [pair for pair in proposed if pair not in planned]
        duplicates = sorted({pair for pair in proposed if proposed.count(pair) > 1})
        if (
            not missing
            and not unexpected
            and not duplicates
            and len(planned) == len(proposed)
        ):
            return ()
        details: list[str] = []
        if missing:
            details.append(
                "missing "
                + ", ".join(
                    f"{change_id}/{binding_id}" for change_id, binding_id in missing
                )
            )
        if unexpected:
            details.append(
                "unexpected "
                + ", ".join(
                    f"{change_id}/{binding_id}"
                    for change_id, binding_id in unexpected
                )
            )
        if duplicates:
            details.append(
                "duplicate "
                + ", ".join(
                    f"{change_id}/{binding_id}"
                    for change_id, binding_id in duplicates
                )
            )
        return (
            ReviewFinding(
                rule_id="plan_change_coverage",
                level="blocking",
                message=(
                    "Proposal does not exactly cover implementation_plan change units: "
                    + "; ".join(details)
                    + "."
                ),
            ),
        )

    @staticmethod
    def _review_binding_source_locality(
        *,
        diff_intent: list[dict[str, Any]],
        implementation_plan: dict[str, Any] | None,
        evidence_pack: dict[str, Any] | None,
    ) -> tuple[ReviewFinding, ...]:
        resolver = BindingResolver.from_implementation_plan(
            implementation_plan,
            evidence_pack=evidence_pack,
        )
        evidence_items = {
            str(item["evidence_id"]): item
            for item in (evidence_pack or {}).get("items") or []
            if isinstance(item, dict) and item.get("evidence_id")
        }
        findings: list[ReviewFinding] = []
        for change in diff_intent:
            if not isinstance(change, dict):
                continue
            change_id = (
                str(change["change_id"])
                if change.get("change_id") is not None
                else None
            )
            binding_id = str(change.get("edit_binding_id") or "")
            binding = resolver.edit_bindings.get(binding_id)
            if binding is None:
                continue
            evidence_id = str(binding.get("evidence_id") or "")
            evidence = evidence_items.get(evidence_id)
            excerpt = evidence.get("excerpt") if isinstance(evidence, dict) else None
            patch = change.get("patch")
            before_text = patch.get("before_text") if isinstance(patch, dict) else None
            if not isinstance(before_text, str) or not before_text:
                continue
            if (
                not isinstance(excerpt, str)
                or not excerpt
                or before_text not in excerpt
            ):
                findings.append(
                    ReviewFinding(
                        rule_id="binding_source_locality",
                        level="blocking",
                        message=(
                            "Executable before_text is not contained in the current "
                            "evidence bound to its edit_binding_id."
                        ),
                        change_id=change_id,
                        evidence_refs=((evidence_id,) if evidence_id else ()),
                    )
                )
        return tuple(findings)

    def _review_context_pack_depth(
        self,
        *,
        proposal: dict[str, Any],
        context_pack: dict[str, Any],
        evidence_pack: dict[str, Any] | None,
        repair_plan: dict[str, Any] | None,
        implementation_plan: dict[str, Any] | None,
        artifact_hashes: dict[str, str] | None,
        domain_policy: DomainReviewPolicy | None,
    ) -> tuple[ReviewFinding, ...]:
        if not context_pack or proposal.get("blocked"):
            return ()
        if repair_plan is not None:
            return self._review_repair_context_pack(
                proposal=proposal,
                context_pack=context_pack,
                repair_plan=repair_plan,
                implementation_plan=implementation_plan,
                evidence_pack=evidence_pack,
            )
        findings: list[ReviewFinding] = []
        proposal_evidence_refs = {str(ref) for ref in proposal.get("evidence_refs") or []}
        snippets = [
            item
            for item in context_pack.get("source_snippets") or []
            if isinstance(item, dict)
        ]
        snippet_refs = {
            str(item.get("evidence_id"))
            for item in snippets
            if item.get("evidence_id")
        }
        if not proposal_evidence_refs or not proposal_evidence_refs.issubset(snippet_refs):
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message=(
                        "Proposal context_pack source_snippets do not cover proposal "
                        "evidence refs."
                    ),
                )
            )
        evidence_ids = self._evidence_ids(evidence_pack, None)
        if evidence_ids and not snippet_refs.issubset(evidence_ids):
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal context_pack source_snippets reference unknown evidence.",
                )
            )

        resolver = BindingResolver.from_implementation_plan(
            implementation_plan,
            evidence_pack=evidence_pack,
        )
        for blocker in resolver.validate_registry_against_evidence().blockers:
            findings.append(_binding_finding(blocker))
        for blocker in resolver.validate_proposal_changes(proposal).blockers:
            findings.append(_binding_finding(blocker))
        for binding in resolver.edit_bindings.values():
            canonical_path = binding.get("canonical_target_path")
            if not canonical_path:
                continue
            if self._is_forbidden_path(str(canonical_path)):
                findings.append(
                    ReviewFinding(
                        rule_id="forbidden_paths",
                        level="blocking",
                        message="Edit binding targets a forbidden or unsafe path.",
                    )
                )
            if domain_policy is not None:
                domain_path = domain_policy.review_path(str(canonical_path))
                if not domain_path.allowed:
                    findings.append(
                        ReviewFinding(
                            rule_id=domain_path.rule_id or "forbidden_paths",
                            level="blocking",
                            message=domain_path.reason,
                        )
                    )

        context_artifact_hashes = context_pack.get("artifact_hashes")
        if not isinstance(context_artifact_hashes, dict):
            context_artifact_hashes = {}
        for artifact_type in ("evidence_pack", "implementation_plan"):
            if not context_artifact_hashes.get(artifact_type):
                findings.append(
                    ReviewFinding(
                        rule_id="context_pack",
                        level="blocking",
                        message=(
                            "Proposal context_pack is missing required upstream "
                            f"artifact hash: {artifact_type}."
                        ),
                    )
                )
            elif artifact_hashes is not None and (
                context_artifact_hashes.get(artifact_type)
                != artifact_hashes.get(artifact_type)
            ):
                findings.append(
                    ReviewFinding(
                        rule_id="context_pack",
                        level="blocking",
                        message=(
                            "Proposal context_pack upstream artifact hash drifted: "
                            f"{artifact_type}."
                        ),
                    )
                )
        if not context_pack.get("implementation_plan_hash") or (
            context_pack.get("implementation_plan_hash")
            != context_artifact_hashes.get("implementation_plan")
        ):
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal context_pack implementation_plan_hash is not bound.",
                )
            )

        git_binding = context_pack.get("git_binding")
        if context_pack.get("git_binding_required") is True and (
            not isinstance(git_binding, dict) or not git_binding.get("base_revision")
        ):
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal context_pack is missing Git base revision binding.",
                )
            )

        budget = context_pack.get("context_budget")
        if not isinstance(budget, dict) or not isinstance(
            budget.get("source_snippet_char_limit"), int
        ):
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message="Proposal context_pack is missing context budget metadata.",
                )
            )

        omission_blockers = [
            blocker
            for blocker in context_pack.get("omission_blockers") or []
            if isinstance(blocker, dict)
        ]
        for blocker in omission_blockers:
            findings.append(
                ReviewFinding(
                    rule_id="context_pack",
                    level="blocking",
                    message=str(
                        blocker.get("message")
                        or blocker.get("code")
                        or "Proposal context omission blocker is present."
                    ),
                )
            )
        return tuple(findings)

    def _review_repair_context_pack(
        self,
        *,
        proposal: dict[str, Any],
        context_pack: dict[str, Any],
        repair_plan: dict[str, Any],
        implementation_plan: dict[str, Any] | None,
        evidence_pack: dict[str, Any] | None,
    ) -> tuple[ReviewFinding, ...]:
        findings: list[ReviewFinding] = []
        resolver = BindingResolver.from_implementation_plan(
            implementation_plan,
            evidence_pack=evidence_pack,
        )
        for blocker in resolver.validate_registry_against_evidence().blockers:
            findings.append(_binding_finding(blocker))
        for blocker in resolver.validate_proposal_changes(proposal).blockers:
            findings.append(_binding_finding(blocker))
        for blocker in resolver.validate_repair_subset(
            repair_plan=repair_plan,
            proposal=proposal,
        ).blockers:
            findings.append(_binding_finding(blocker))
        return tuple(findings)

    def _review_hook_target_bindings(
        self,
        *,
        diff_intent: list[dict[str, Any]],
        implementation_plan: dict[str, Any] | None,
        evidence_pack: dict[str, Any] | None,
        feature_spec: dict[str, Any] | None,
        repair_plan: dict[str, Any] | None,
    ) -> tuple[ReviewFinding, ...]:
        if not isinstance(feature_spec, dict):
            return ()
        if not _hook_target_required(feature_spec):
            return ()
        resolver = BindingResolver.from_implementation_plan(
            implementation_plan,
            evidence_pack=evidence_pack,
        )
        planned_changes = _planned_changes_by_id(implementation_plan)
        planned_repairs_by_binding = _planned_changes_by_binding(
            implementation_plan
        )
        findings: list[ReviewFinding] = []
        for change in diff_intent:
            if not isinstance(change, dict):
                continue
            change_id = (
                str(change["change_id"])
                if change.get("change_id") is not None
                else None
            )
            hook_target_binding_id = change.get("hook_target_binding_id")
            planned_repair = planned_repairs_by_binding.get(
                str(change.get("edit_binding_id") or "")
            )
            if (
                repair_plan is not None
                and planned_repair is not None
                and planned_repair.get("hook_target_binding_id") is None
                and hook_target_binding_id is None
            ):
                continue
            if not hook_target_binding_id:
                findings.append(
                    ReviewFinding(
                        rule_id="hook_target_binding",
                        level="blocking",
                        message=(
                            "Hook-required change unit is missing "
                            "hook_target_binding_id."
                        ),
                        change_id=change_id,
                    )
                )
                continue
            binding_id = str(hook_target_binding_id)
            binding = resolver.supporting_bindings.get(binding_id)
            if binding is None:
                findings.append(
                    ReviewFinding(
                        rule_id="hook_target_binding",
                        level="blocking",
                        message="Hook target binding id is not in binding_registry.",
                        change_id=change_id,
                    )
                )
                continue
            if not _is_complete_hook_target_binding(binding):
                findings.append(
                    ReviewFinding(
                        rule_id="hook_target_binding",
                        level="blocking",
                        message="Hook target binding identity is incomplete.",
                        change_id=change_id,
                    )
                )
            planned_change = (
                planned_repair
                if repair_plan is not None and planned_repair is not None
                else planned_changes.get(change_id or "")
            )
            if not planned_change or str(
                planned_change.get("hook_target_binding_id") or ""
            ) != binding_id:
                findings.append(
                    ReviewFinding(
                        rule_id="hook_target_binding",
                        level="blocking",
                        message=(
                            "Hook target binding id was not selected by the "
                            "implementation_plan change unit."
                        ),
                        change_id=change_id,
                    )
                )
        return tuple(findings)

    @staticmethod
    def _is_forbidden_path(path: str) -> bool:
        normalized = path.replace("\\", "/")
        posix = PurePosixPath(normalized)
        parts = set(posix.parts)
        if ".." in parts:
            return True
        forbidden_parts = {".git", ".env", "secrets"}
        return bool(parts & forbidden_parts)

    @staticmethod
    def _requires_game_profile(proposal: dict[str, Any]) -> bool:
        risk = proposal.get("risk_assessment", {})
        if risk.get("requires_game_profile"):
            return True
        risk_tags = set(risk.get("risk_tags") or [])
        if risk.get("risk_level") in {"high", "critical"}:
            return True
        game_specific_tags = {
            "game_hook",
            "runtime_behavior_change",
            "native/interop/reflection-heavy",
            "critical",
            "high",
        }
        return bool(risk_tags & game_specific_tags)

    @staticmethod
    def _blocked_decision(blocking: list[ReviewFinding]) -> str:
        rule_ids = {finding.rule_id for finding in blocking}
        if rule_ids & {"proposal_critic_unavailable", "proposal_critic_block"}:
            return "blocked_requires_human"
        if "semantic_assurance_violation" in rule_ids:
            return "blocked_requires_human"
        if "evidence_binding" in rule_ids:
            return "blocked_insufficient_evidence"
        if "scope_boundary" in rule_ids or "forbidden_paths" in rule_ids:
            return "blocked_scope_risk"
        if "domain_profile" in rule_ids:
            return "blocked_requires_human"
        if "permission_level" in rule_ids or "risk_policy" in rule_ids:
            return "blocked_requires_human"
        return "revise_proposal"


def _binding_finding(blocker: dict[str, Any]) -> ReviewFinding:
    return ReviewFinding(
        rule_id="binding_registry",
        level="blocking",
        message=str(blocker.get("message") or blocker.get("code") or "Binding failed."),
        change_id=(
            str(blocker["change_id"])
            if blocker.get("change_id") is not None
            else None
        ),
        evidence_refs=(
            (str(blocker["evidence_id"]),)
            if blocker.get("evidence_id") is not None
            else ()
        ),
    )


def _hook_target_required(feature_spec: dict[str, Any]) -> bool:
    hook_requirement = feature_spec.get("hook_requirement")
    return isinstance(hook_requirement, dict) and hook_requirement.get("required") is True


def _is_complete_hook_target_binding(binding: dict[str, Any]) -> bool:
    return (
        binding.get("source_type") == "decompiled_api"
        and binding.get("usage") == "hook_target"
        and bool(binding.get("target_symbol"))
        and bool(binding.get("target_span"))
        and bool(binding.get("declaring_type"))
        and bool(binding.get("method_name"))
        and isinstance(binding.get("parameter_types"), list)
        and bool(binding.get("return_type"))
        and bool(binding.get("method_kind"))
    )


def _planned_changes_by_id(
    implementation_plan: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(implementation_plan, dict):
        return {}
    return {
        str(change["change_id"]): change
        for change in implementation_plan.get("proposed_changes") or []
        if isinstance(change, dict) and change.get("change_id")
    }


def _planned_changes_by_binding(
    implementation_plan: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(implementation_plan, dict):
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for change in implementation_plan.get("proposed_changes") or []:
        if not isinstance(change, dict) or not change.get("edit_binding_id"):
            continue
        grouped.setdefault(str(change["edit_binding_id"]), []).append(change)
    return {
        binding_id: changes[0]
        for binding_id, changes in grouped.items()
        if len(changes) == 1
    }
