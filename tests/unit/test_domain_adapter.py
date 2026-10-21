"""Tests for domain adapter and GameProfile contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.mod_agent.domain import (
    DomainAdapterRegistry,
    DomainProfile,
    GTFOAdapter,
    UnsupportedDomainAdapter,
    build_gtfo_profile,
)
from src.mod_agent.proposal import ProposalReviewer


def test_gtfo_profile_exposes_required_game_profile_contract(tmp_path: Path) -> None:
    profile = build_gtfo_profile(tmp_path)
    payload = profile.to_dict()

    assert payload["game_id"] == "gtfo"
    assert payload["display_name"] == "GTFO"
    assert payload["runtime"] == "Unity IL2CPP"
    assert payload["mod_loader"] == "BepInEx"
    assert payload["harmony"] == "HarmonyX"
    assert payload["status"] == "experimental"
    assert "generated_mods" in payload["allowed_roots"]
    assert "BepInEx/core" in payload["forbidden_paths"]
    assert "GameAssembly.dll" in payload["forbidden_paths"]
    assert payload["build_profiles"][0]["profile_id"] == "gtfo_bepinex_il2cpp"
    assert {item["source_type"] for item in payload["evidence_sources"]} >= {
        "target_source",
        "reference_mod",
        "decompiled_api",
        "domain_note",
        "runtime_log",
        "prior_artifact",
    }


def test_gtfo_adapter_classifies_risk_taxonomy() -> None:
    adapter = GTFOAdapter()

    assert adapter.classify_request("change config default").risk_level == "low"
    assert adapter.classify_request("make the HUD marker pink").risk_level == "medium"
    assert adapter.classify_request("patch a Harmony hook").risk_level == "high"
    assert adapter.classify_request("change enemy runtime behavior").risk_level == "high"
    assert adapter.classify_request("use IL2CPP reflection").risk_level == "critical"
    unknown = adapter.classify_request("do the thing")
    assert unknown.risk_level == "unknown"
    assert unknown.blocked is True


def test_unknown_profile_blocks_game_specific_runtime_patch() -> None:
    profile = UnsupportedDomainAdapter().identify_project(".")
    proposal = _runtime_patch_proposal()

    review = ProposalReviewer().review(
        proposal,
        implementation_plan=_implementation_plan(target_file="mods/BetterHitFeedback/Plugin.cs"),
        evidence_pack=_evidence_pack(target_file="mods/BetterHitFeedback/Plugin.cs"),
        feature_spec={"target_behavior": "change runtime behavior"},
        domain_policy=profile.review_policy(),
    )

    assert review.decision == "blocked_requires_human"
    assert any(finding.rule_id == "domain_profile" for finding in review.findings)
    assert review.allowed_next_stage is None


def test_gtfo_domain_policy_blocks_forbidden_paths() -> None:
    profile = build_gtfo_profile(".")
    proposal = _runtime_patch_proposal(target_file="interop/GTFO.API.dll")

    review = ProposalReviewer().review(
        proposal,
        implementation_plan=_implementation_plan(target_file="interop/GTFO.API.dll"),
        evidence_pack=_evidence_pack(target_file="interop/GTFO.API.dll"),
        feature_spec={"target_behavior": "change runtime behavior"},
        domain_policy=profile.review_policy(),
    )

    assert review.decision == "blocked_scope_risk"
    assert any(finding.rule_id == "forbidden_paths" for finding in review.findings)


def test_gtfo_domain_policy_allows_safe_root_for_dry_run_review() -> None:
    profile = build_gtfo_profile(".")
    proposal = _runtime_patch_proposal(target_file="mods/BetterHitFeedback/Plugin.cs")

    review = ProposalReviewer().review(
        proposal,
        implementation_plan=_implementation_plan(target_file="mods/BetterHitFeedback/Plugin.cs"),
        evidence_pack=_evidence_pack(target_file="mods/BetterHitFeedback/Plugin.cs"),
        feature_spec={"target_behavior": "change runtime behavior"},
        domain_policy=profile.review_policy(),
    )

    assert review.decision == "approve_dry_run"
    assert review.allowed_next_stage == "dry_run_apply"


def test_gtfo_domain_policy_binds_project_relative_path_to_exact_target_root(
    tmp_path: Path,
) -> None:
    policy = (
        build_gtfo_profile(tmp_path)
        .review_policy()
        .bind_target_root("generated_mods/BetterHitFeedback")
    )

    allowed = policy.review_path("Patches/HitFeedbackBehaviorPatch.cs")
    other_project = policy.review_path("generated_mods/OtherMod/Plugin.cs")
    forbidden = policy.review_path("interop/GTFO.API.dll")
    absolute = policy.review_path(r"C:\outside\Plugin.cs")

    assert policy.project_root == str(tmp_path.resolve())
    assert policy.target_root == "generated_mods/BetterHitFeedback"
    assert allowed.allowed is True
    assert other_project == type(other_project)(
        allowed=False,
        rule_id="scope_boundary",
        reason="Target path is outside bound domain target root.",
    )
    assert forbidden.allowed is False
    assert forbidden.rule_id == "forbidden_paths"
    assert absolute.allowed is False
    assert absolute.rule_id == "forbidden_paths"


def test_gtfo_domain_policy_rejects_disallowed_target_root() -> None:
    policy = build_gtfo_profile(".").review_policy()

    with pytest.raises(
        ValueError,
        match="Domain target root is not allowed by the domain policy",
    ):
        policy.bind_target_root("workspaces/materialized_source/BetterHitFeedback")
    with pytest.raises(ValueError, match="Domain target root must be relative"):
        policy.bind_target_root(r"C:\generated_mods\BetterHitFeedback")


def test_domain_adapter_registry_accepts_new_profile_without_workflow_change(
    tmp_path: Path,
) -> None:
    adapter = GTFOAdapter()
    registry = DomainAdapterRegistry()

    registry.register(adapter)

    assert registry.names() == ("gtfo_bepinex_il2cpp",)
    assert registry.get("gtfo_bepinex_il2cpp").identify_project(str(tmp_path)).status in {
        "unsupported",
        "experimental",
    }
    with pytest.raises(KeyError, match="Unknown domain adapter"):
        registry.get("missing")


def _runtime_patch_proposal(
    *,
    target_file: str = "mods/BetterHitFeedback/Plugin.cs",
) -> dict[str, object]:
    return {
        "blocked": False,
        "summary": "change runtime behavior",
        "files": [target_file],
        "evidence_refs": ["ev_001"],
        "diff_intent": [
            {
                "change_id": "chg_001",
                "edit_binding_id": "edit_binding_001",
                "rationale": "change runtime behavior",
                "rollback_hint": "Restore previous behavior.",
            }
        ],
        "change_units": [
            {
                "change_unit_id": "chg_001",
                "change_type": None,
                "edit_binding_id": "edit_binding_001",
                "supporting_binding_ids": [],
                "target_file": None,
                "edit_target_path": None,
                "target_symbol": None,
                "evidence_refs": [],
                "expected_validation": [],
                "rollback_hint": "Restore previous behavior.",
            }
        ],
        "context_pack": {
            "feature_id": "feature_001",
            "files": [target_file],
            "evidence_refs": ["ev_001"],
            "evidence_source_paths": [target_file],
            "scope_boundary": [],
            "change_unit_ids": ["chg_001"],
            "source_snippets": [
                {
                    "evidence_id": "ev_001",
                    "source_path": "mods/BetterHitFeedback/Plugin.cs",
                    "source_type": "target_source",
                    "symbol": None,
                    "span": {"start_line": 1, "end_line": 1},
                    "retrieval_method": "symbol_exact",
                    "excerpt": "runtime behavior evidence",
                }
            ],
            "target_symbol_signatures": [],
            "binding_registry_ref": {
                "artifact_type": "implementation_plan",
                "artifact_hash": "sha256:implementation",
                "schema_version": "binding_registry_v1",
            },
            "display_files": [target_file],
            "display_fields_authority": "derived_non_authoritative",
            "allowed_edit_binding_ids": ["edit_binding_001"],
            "artifact_hashes": {
                "evidence_pack": "sha256:evidence",
                "implementation_plan": "sha256:implementation",
            },
            "implementation_plan_hash": "sha256:implementation",
            "git_binding": {
                "status": "not_configured",
                "base_revision": None,
            },
            "git_binding_required": False,
            "context_budget": {
                "source_snippet_count": 1,
                "source_snippet_chars": 25,
                "source_snippet_char_limit": 6000,
                "omitted_source_snippet_count": 0,
            },
            "omission_blockers": [],
        },
        "risk_assessment": {
            "risk_level": "high",
            "risk_tags": ["runtime_behavior_change"],
            "requires_game_profile": True,
            "requires_manual_review": False,
        },
        "expected_tests": ["compile"],
        "no_op_status": {"is_no_op": False, "reason": None},
        "rollback_hints": [
            {
                "change_unit_id": "chg_001",
                "rollback_hint": "Restore previous behavior.",
            }
        ],
    }


def _evidence_pack(
    *,
    target_file: str = "mods/BetterHitFeedback/Plugin.cs",
) -> dict[str, object]:
    return {
        "items": [
            {
                "evidence_id": "ev_001",
                "source_path": target_file,
                "source_type": "target_source",
                "symbol": None,
                "span": {"start_line": 1, "end_line": 1},
                "source_hash": "a" * 64,
            }
        ]
    }


def _implementation_plan(
    *,
    target_file: str,
) -> dict[str, object]:
    return {
        "binding_registry": {
            "schema_version": "binding_registry_v1",
            "owner_artifact_type": "implementation_plan",
            "edit_bindings": [
                {
                    "binding_id": "edit_binding_001",
                    "evidence_id": "ev_001",
                    "canonical_target_path": target_file,
                    "source_type": "target_source",
                    "source_hash": "a" * 64,
                    "target_symbol": None,
                    "target_span": {"start_line": 1, "end_line": 1},
                }
            ],
            "supporting_bindings": [],
        },
        "proposed_changes": [{
            "change_id": "chg_001",
            "edit_binding_id": "edit_binding_001",
            "operation": "minimal_evidence_bound_change",
        }],
    }
