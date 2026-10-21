"""Tests for Phase E deterministic planning."""

from __future__ import annotations

import pytest

from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.knowledge import IDENTITY_FIELD_MAPPING, IDENTITY_SCHEMA_VERSION
from src.mod_agent.planning import (
    FeatureSpecBuilder,
    ImplementationPlanner,
    ModPlanningBriefBuilder,
    RequestPlanner,
)
from src.mod_agent.planning.evidence_requirements import CONFIGURATION_EVIDENCE


def _identity_schema_payload() -> dict[str, object]:
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "identity_field_mapping": dict(IDENTITY_FIELD_MAPPING),
    }


def _authoritative_item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "evidence_id": "ev_001",
        "source_type": "target_source",
        "source_path": "mods/BetterHitFeedback/Plugin.cs",
        "symbol": "Load",
        "span": {"start_line": 10, "end_line": 20},
        "retrieval_method": "symbol_exact",
        "claim_supported": "Symbol Load exists.",
        "authority": "mod_agent_deterministic",
        "chunk_id": "chunk_001",
        "document_id": "doc_001",
        "source_ref": "doc_001",
        "source_hash": "file_hash_001",
        "file_hash": "file_hash_001",
        "doc_hash": "doc_hash_001",
        "declaring_type": "Plugin",
        "method_name": "Load",
        "parameter_types": [],
        "return_type": "void",
        "method_kind": "method",
        "declared_member_types": {
            "ConfigEnabled": "ConfigEntry<bool>",
            "PrimitiveEnabled": "bool",
        },
        "declaring_type_full_name": "BetterHitFeedback.Plugin",
        "declared_members": {
            "ConfigEnabled": {
                "member_name": "ConfigEnabled",
                "member_kind": "property",
                "declared_type": "ConfigEntry<bool>",
                "declaring_type": "BetterHitFeedback.Plugin",
                "declaration_span": {"start_line": 5, "end_line": 5},
                "is_static": True,
            }
        },
        "verified_member_accesses": [
            {
                "source_receiver": {
                    "evidence_id": "ev_001",
                    "source_path": "mods/BetterHitFeedback/Plugin.cs",
                    "source_hash": "file_hash_001",
                    "declaring_type": "BetterHitFeedback.Plugin",
                    "member_name": "ConfigEnabled",
                    "member_kind": "property",
                    "declaration_span": {"start_line": 5, "end_line": 5},
                    "declared_type": "ConfigEntry<bool>",
                    "receiver_closed_type": (
                        "BepInEx.Configuration.ConfigEntry`1<System.Boolean>"
                    ),
                },
                "consumer": {
                    "declaring_type": "BetterHitFeedback.Plugin",
                    "receiver_member": "ConfigEnabled",
                    "member_kind": "property",
                    "is_static": True,
                    "receiver_closed_type": (
                        "BepInEx.Configuration.ConfigEntry`1<System.Boolean>"
                    ),
                },
                "type_reference": {
                    "symbol": "BepInEx.Configuration.ConfigEntry`1",
                },
                "member_access": {
                    "declaring_type": "BepInEx.Configuration.ConfigEntry`1",
                    "member_name": "Value",
                    "member_kind": "property",
                    "access_kind": "instance_property_get",
                    "closed_result_type": "System.Boolean",
                },
            }
        ],
    }
    item.update(overrides)
    return item


def _hook_target_item(**overrides: object) -> dict[str, object]:
    item = _authoritative_item(
        evidence_id="ev_api_001",
        source_type="decompiled_api",
        source_path="decompiled/CrosshairGuiLayer.cs",
        symbol="CrosshairGuiLayer.ShowHitIndicator",
        span={"start_line": 100, "end_line": 120},
        retrieval_method="symbol_exact",
        claim_supported="ShowHitIndicator is the hook target.",
        usage="hook_target",
        declaring_type="CrosshairGuiLayer",
        method_name="ShowHitIndicator",
        parameter_types=[
            "UnityEngine.Vector3",
            "System.Boolean",
            "System.Boolean",
            "System.Boolean",
        ],
        return_type="System.Void",
        method_kind="instance_method",
        chunk_id="chunk_api_001",
        document_id="doc_api_001",
        source_ref="doc_api_001",
        source_hash="file_hash_api_001",
        file_hash="file_hash_api_001",
        doc_hash="doc_hash_api_001",
    )
    item.update(overrides)
    return item


def test_request_planner_blocks_ambiguous_request() -> None:
    plan = RequestPlanner().plan("fix it")

    assert plan.blocked is True
    assert plan.route == "blocked_ambiguous"
    assert plan.risk_level == "blocked"
    assert plan.uncertainties[0].uncertainty_id == "ambiguous_user_request"


def test_request_planner_blocks_ambiguous_chinese_request() -> None:
    plan = RequestPlanner().plan("请帮我处理一下这个东西")

    assert plan.blocked is True
    assert plan.route == "blocked_ambiguous"


def test_request_planner_allows_clear_chinese_runtime_request_to_seek_evidence() -> None:
    plan = RequestPlanner().plan(
        "我想让模组在玩家用枪打到敌人并触发踉跄时，额外显示一个短暂的黄色 X，"
        "并播放一声简短提示音。",
        target_mod="BetterHitFeedback",
    )

    assert plan.blocked is False
    assert plan.route == "standard_feature"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is True
    assert plan.hook_requirement == {
        "required": False,
        "reason": "implementation_mechanism_unresolved",
    }


def test_non_goal_damage_terms_do_not_force_hook_evidence() -> None:
    plan = RequestPlanner().plan(
        "Change the kill marker color to pink without changing damage or sound.",
        target_mod="BetterHitFeedback",
    )

    assert plan.requires_hook_evidence is False
    assert plan.hook_requirement == {"required": False, "reason": "implementation_mechanism_unresolved"}


def test_request_planner_keeps_runtime_acceptance_without_selecting_hook() -> None:
    plan = RequestPlanner().plan(
        "make enemies glow when damaged in BetterHitFeedback"
    )

    assert plan.blocked is False
    assert plan.route == "standard_feature"
    assert plan.risk_level == "medium"
    assert plan.target_mod == "BetterHitFeedback"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is True
    assert plan.hook_requirement == {
        "required": False,
        "reason": "implementation_mechanism_unresolved",
    }


def test_request_planner_does_not_downgrade_mixed_config_gameplay_request() -> None:
    plan = RequestPlanner().plan(
        "WallCrawler JSON config changes enemy movement on walls, slopes, and ceilings.",
        target_mod="WallCrawler",
    )

    assert plan.route == "standard_feature"
    assert plan.risk_level == "medium"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is True


def test_request_planner_classifies_wallcrawler_chinese_request() -> None:
    plan = RequestPlanner().plan(
        "希望让 GTFO 的怪物在配置指定的区域内，能够在原本不支持行走的墙面、"
        "斜面和天花板上移动。区域应能通过面向地图开发者的 JSON 配置；配置需要"
        "与游戏/地图数据库的真实 PersistentID 或准确等价身份建立绑定。",
        target_mod="WallCrawler",
    )

    assert plan.route == "standard_feature"
    assert plan.risk_level == "medium"
    assert plan.target_mod == "WallCrawler"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is True


def test_request_planner_does_not_infer_persistent_id_as_target_mod() -> None:
    plan = RequestPlanner().plan(
        "JSON config binds a region to PersistentID without changing runtime behavior."
    )

    assert plan.target_mod is None


def test_request_planner_separates_runtime_diagnostics_from_hook_work() -> None:
    plan = RequestPlanner().plan(
        "default config Load runtime diagnostics",
        target_mod="BetterHitFeedback",
    )

    assert plan.route == "quick_patch"
    assert plan.risk_level == "high"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is True


def test_request_planner_keeps_plain_config_compile_level() -> None:
    plan = RequestPlanner().plan(
        "default config EnableLogging false",
        target_mod="BetterHitFeedback",
    )

    assert plan.route == "quick_patch"
    assert plan.risk_level == "low"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is False


def test_request_planner_requires_runtime_for_accurate_crosshair_restart() -> None:
    request = (
        "为 AccurateCrosshair 增加 [General Settings] Accurate Spread Enabled，默认 true；"
        "设置为 false 并重启后，只停止 AccurateCrosshair 使用准确扩散数据覆盖游戏传入的"
        "原始准星尺寸；Follow Recoil、First Shot Display Mode、Pop Enabled 保持独立；"
        "true 时保持现有行为。"
    )

    plan = RequestPlanner().plan(request, target_mod="AccurateCrosshair")
    spec = FeatureSpecBuilder().build(plan)

    assert plan.route == "quick_patch"
    assert plan.risk_level == "low"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is True
    assert plan.hook_requirement == {"required": False, "reason": "implementation_mechanism_unresolved"}
    assert spec.runtime_verification_plan == (
        "compile or static validation",
        "startup_loaded",
        "target_path_executed",
        "acceptance_evidence_observed",
        "non_goals_preserved",
    )


@pytest.mark.parametrize("lifecycle_term", ["restart", "reload"])
def test_request_planner_requires_runtime_for_explicit_config_lifecycle_term(
    lifecycle_term: str,
) -> None:
    plan = RequestPlanner().plan(
        f"default config EnableLogging false after {lifecycle_term}",
        target_mod="BetterHitFeedback",
    )

    assert plan.route == "quick_patch"
    assert plan.risk_level == "low"
    assert plan.requires_hook_evidence is False
    assert plan.requires_runtime_evidence is True


def test_feature_spec_builder_creates_verifiable_contract() -> None:
    plan = RequestPlanner().plan(
        "default config EnableLogging false",
        target_mod="BetterHitFeedback",
    )

    spec = FeatureSpecBuilder().build(plan)

    assert spec.blocked is False
    assert spec.feature_id == plan.feature_id
    assert spec.target_behavior == plan.user_request
    assert spec.hook_requirement == {"required": False, "reason": "implementation_mechanism_unresolved"}
    assert spec.evidence_requirements == ()
    assert spec.acceptance_criteria
    assert spec.non_goals == ()


def test_feature_spec_builder_keeps_runtime_proof_in_post_change_handoff() -> None:
    plan = RequestPlanner().plan(
        "make Load enable diagnostics",
        target_mod="BetterHitFeedback",
    )
    brief_payload = ModPlanningBriefBuilder().build(plan).to_artifact_payload()

    spec = FeatureSpecBuilder().build(plan, planning_brief=brief_payload)

    assert "log marker or startup evidence" not in spec.evidence_requirements
    assert "startup_loaded" in spec.runtime_verification_plan
    assert "diagnostic marker observed" in spec.runtime_verification_plan


def test_feature_spec_builder_preserves_blocking_uncertainty() -> None:
    plan = RequestPlanner().plan("test")

    spec = FeatureSpecBuilder().build(plan)

    assert spec.blocked is True
    assert spec.blockers[0]["code"] == "AMBIGUOUS_USER_REQUEST"


def test_mod_planning_brief_classifies_config_only_request() -> None:
    plan = RequestPlanner().plan(
        "default config EnableLogging false",
        target_mod="BetterHitFeedback",
    )

    brief = ModPlanningBriefBuilder().build(plan, target_game="GTFO")
    payload = brief.to_artifact_payload()

    assert payload["decision"] == "ready_for_feature_spec"
    assert payload["request_type"]["value"] == "config-only"
    assert payload["user_intent"]["target_game"] == "GTFO"
    assert payload["user_intent"]["target_mod"] == "BetterHitFeedback"
    assert CONFIGURATION_EVIDENCE in (
        payload["knowledge_query_plan"]["queries"]
    )
    assert payload["handoff"]["feature_spec_guidance"]["evidence_requirements"] == []
    assert payload["next_action"] == "feature_spec"
    assert payload["blocked"] is False


def test_mod_planning_brief_classifies_ui_feedback_request() -> None:
    plan = RequestPlanner().plan(
        "show reload progress in HUD",
        target_mod="BetterHitFeedback",
    )

    payload = ModPlanningBriefBuilder().build(plan).to_artifact_payload()

    assert payload["request_type"]["value"] == "UI feedback"
    assert "game_profile" in payload["knowledge_query_plan"]["source_types"]
    assert "UI or feedback surface evidence" in (
        payload["planning_options"][0]["required_evidence"]
    )
    assert payload["handoff"]["runtime_verification_expectations"] == [
        "compile or static validation",
        "user-visible feedback evidence",
    ]


def test_mod_planning_brief_classifies_hook_required_request() -> None:
    plan = RequestPlanner().plan(
        "hook enemy damage method for BetterHitFeedback"
    )

    payload = ModPlanningBriefBuilder().build(plan).to_artifact_payload()

    assert payload["request_type"]["value"] == "hook-required"
    assert payload["planning_options"][0]["hook_likelihood"] == "high"
    assert "symbol_search" in payload["knowledge_query_plan"]["source_types"]
    assert "target symbol evidence" in payload["planning_options"][0]["required_evidence"]
    assert payload["handoff"]["hook_discovery_guidance"] == (
        "Run hook discovery before implementation planning."
    )


def test_planning_only_brief_stops_hook_request_before_feature_spec() -> None:
    plan = RequestPlanner().plan(
        "Explicitly hook enemy movement methods for WallCrawler.",
        target_mod="WallCrawler",
    )

    payload = ModPlanningBriefBuilder().build_planning_only(
        plan,
        target_game="GTFO",
    ).to_artifact_payload()

    assert payload["decision"] == "needs_evidence"
    assert payload["blocked"] is True
    assert payload["request_type"]["value"] == "hook-required"
    assert payload["selected_plan"]["scope"] == plan.user_request
    assert payload["next_action"] == "hook discovery"
    assert payload["blockers"][0]["code"] == "HOOK_EVIDENCE_REQUIRED"


def test_mod_planning_brief_blocks_unclear_request() -> None:
    plan = RequestPlanner().plan("fix it")

    payload = ModPlanningBriefBuilder().build(plan).to_artifact_payload()

    assert payload["decision"] == "needs_clarification"
    assert payload["request_type"]["value"] == "unknown / needs evidence"
    assert payload["selected_plan"]["scope"] == ""
    assert payload["next_action"] == "ask user"
    assert payload["blocked"] is True
    assert payload["blockers"][0]["code"] == "AMBIGUOUS_USER_REQUEST"


def test_implementation_plan_blocks_when_evidence_is_not_sufficient() -> None:
    plan = RequestPlanner().plan("make enemies glow when damaged")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "partial",
        "can_generate_patch": False,
        "gaps": ["No safe hook found."],
        "blockers": [{"code": "INSUFFICIENT_HOOK_EVIDENCE"}],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.open_questions == ("No safe hook found.",)
    assert implementation.proposed_changes == ()


def test_implementation_plan_binds_every_change_to_evidence() -> None:
    plan = RequestPlanner().plan("make Load enable diagnostics")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [_authoritative_item()],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    assert implementation.evidence_source_paths == ("mods/BetterHitFeedback/Plugin.cs",)
    assert implementation.target_files == ("mods/BetterHitFeedback/Plugin.cs",)
    assert implementation.target_symbols == ("Load",)
    assert implementation.evidence_refs == ("ev_001",)
    assert implementation.proposed_changes[0]["edit_binding_id"] == (
        "edit_binding_001"
    )
    assert implementation.proposed_changes[0]["supporting_binding_ids"] == []
    assert "target_file" not in implementation.proposed_changes[0]
    edit_binding = implementation.binding_registry["edit_bindings"][0]
    assert edit_binding["evidence_id"] == "ev_001"
    assert edit_binding["source_hash"] == "file_hash_001"
    assert edit_binding["canonical_target_path"] == "mods/BetterHitFeedback/Plugin.cs"
    assert edit_binding["target_span"] == {"start_line": 10, "end_line": 20}
    assert edit_binding["declaring_type"] == "Plugin"
    assert edit_binding["method_name"] == "Load"
    assert edit_binding["parameter_types"] == []
    assert edit_binding["return_type"] == "void"
    assert edit_binding["method_kind"] == "method"
    assert edit_binding["declared_member_types"] == {
        "ConfigEnabled": "ConfigEntry<bool>",
        "PrimitiveEnabled": "bool",
    }
    assert edit_binding["declaring_type_full_name"] == "BetterHitFeedback.Plugin"
    assert edit_binding["declared_members"]["ConfigEnabled"]["declaration_span"] == {
        "start_line": 5,
        "end_line": 5,
    }
    assert edit_binding["verified_member_accesses"][0]["edit_binding_id"] == (
        "edit_binding_001"
    )
    assert edit_binding["verified_member_accesses"][0]["member_access"][
        "closed_result_type"
    ] == "System.Boolean"
    assert implementation.binding_registry["supporting_bindings"] == []
    assert implementation.rollback_plan


def test_formal_coverage_admission_is_selected_only_and_legacy_default_remains() -> None:
    plan = RequestPlanner().plan("make Load enable diagnostics")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [_authoritative_item()],
    }

    selected = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
        formal_requirement_coverage_required=True,
    )
    legacy = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert selected.blocking is True
    assert selected.proposed_changes == ()
    assert selected.binding_registry is None
    assert selected.blockers == (
        {
            "code": "FORMAL_REQUIREMENT_COVERAGE_REQUIRED",
            "message": (
                "Selected ProductExecution requires formal requirement coverage "
                "before edit-binding derivation."
            ),
        },
    )
    assert legacy.blocking is False
    assert legacy.proposed_changes[0]["edit_binding_id"] == "edit_binding_001"


def test_implementation_plan_allows_mixed_evidence_without_editing_every_source() -> None:
    plan = RequestPlanner().plan("hook enemy damage method for BetterHitFeedback")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(),
            _hook_target_item(),
            _authoritative_item(
                evidence_id="ev_runtime_001",
                source_type="runtime_evidence",
                source_path="logs/gtfo/BepInEx.log",
                symbol="BetterHitFeedback.Load",
                span={"start_line": 1, "end_line": 4},
                claim_supported="Runtime log proves plugin load marker.",
                chunk_id="chunk_runtime_001",
                document_id="doc_runtime_001",
                source_ref="doc_runtime_001",
                source_hash="file_hash_runtime_001",
                file_hash="file_hash_runtime_001",
                doc_hash="doc_hash_runtime_001",
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    assert implementation.target_files == ("mods/BetterHitFeedback/Plugin.cs",)
    assert implementation.evidence_refs == (
        "ev_001",
        "ev_api_001",
        "ev_runtime_001",
    )
    assert [
        item["evidence_id"]
        for item in implementation.binding_registry["edit_bindings"]
    ] == [
        "ev_001"
    ]
    assert {item["evidence_id"] for item in implementation.binding_registry["supporting_bindings"]} == {
        "ev_api_001",
        "ev_runtime_001",
    }
    assert implementation.proposed_changes[0]["supporting_binding_ids"] == []
    assert implementation.proposed_changes[0]["hook_target_binding_id"] == (
        "hook_target_001"
    )


def test_implementation_plan_blocks_ambiguous_hook_target_selection() -> None:
    plan = RequestPlanner().plan("hook enemy damage method for BetterHitFeedback")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(),
            _hook_target_item(),
            _hook_target_item(
                evidence_id="ev_api_002",
                symbol="EnemyAgent.Damage",
                source_path="decompiled/EnemyAgent.cs",
                declaring_type="EnemyAgent",
                method_name="Damage",
                parameter_types=["System.Single"],
                chunk_id="chunk_api_002",
                document_id="doc_api_002",
                source_ref="doc_api_002",
                source_hash="file_hash_api_002",
                file_hash="file_hash_api_002",
                doc_hash="doc_hash_api_002",
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.blockers[0]["code"] == "AMBIGUOUS_HOOK_TARGET"
    assert implementation.proposed_changes == ()


def test_implementation_plan_selects_complete_hook_target_signature() -> None:
    plan = RequestPlanner().plan("hook enemy damage method for BetterHitFeedback")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(),
            _hook_target_item(),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    hook_binding = implementation.binding_registry["supporting_bindings"][0]
    assert hook_binding["binding_id"] == "hook_target_001"
    assert hook_binding["declaring_type"] == "CrosshairGuiLayer"
    assert hook_binding["method_name"] == "ShowHitIndicator"
    assert hook_binding["parameter_types"] == [
        "UnityEngine.Vector3",
        "System.Boolean",
        "System.Boolean",
        "System.Boolean",
    ]
    assert implementation.proposed_changes[0]["hook_target_binding_id"] == (
        "hook_target_001"
    )


def test_implementation_plan_blocks_when_evidence_source_is_not_editable() -> None:
    plan = RequestPlanner().plan("make Load enable diagnostics")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(
                evidence_id="ev_api_001",
                source_type="decompiled_api",
                source_path="decompiled/EnemyAgent.cs",
                symbol="Load",
            )
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.target_files == ()
    assert implementation.proposed_changes == ()
    assert implementation.evidence_refs == ()
    assert implementation.evidence_source_paths == ("decompiled/EnemyAgent.cs",)
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_UNRESOLVED"


def test_planning_payloads_validate_against_artifact_contract(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    plan = RequestPlanner().plan("make Load enable diagnostics")
    brief = ModPlanningBriefBuilder().build(plan, target_game="GTFO")
    spec = FeatureSpecBuilder().build(
        plan,
        planning_brief=brief.to_artifact_payload(),
    )
    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack={
            "readiness": "sufficient",
            "can_generate_patch": True,
            "authority": "mod_agent_deterministic_evidence",
            "authoritative_for_patch": True,
            **_identity_schema_payload(),
            "items": [_authoritative_item()],
        },
    )

    request_ref = store.write(
        feature_id=plan.feature_id,
        artifact_type="request_planning_report",
        payload=plan.to_artifact_payload(),
    )
    brief_ref = store.write(
        feature_id=plan.feature_id,
        artifact_type="mod_planning_brief",
        payload=brief.to_artifact_payload(),
        upstream_refs=[request_ref],
    )
    feature_ref = store.write(
        feature_id=plan.feature_id,
        artifact_type="feature_spec",
        payload=spec.to_artifact_payload(),
        upstream_refs=[request_ref, brief_ref],
    )
    evidence_ref = store.write(
        feature_id=plan.feature_id,
        artifact_type="evidence_pack",
        payload={
            "node_name": "retrieve_evidence",
            "stage": "evidence_retrieval",
            "message": "Evidence is sufficient.",
            "query": "make Load enable diagnostics",
            "readiness": "sufficient",
            "can_generate_patch": True,
            "authority": "mod_agent_deterministic_evidence",
            "authoritative_for_patch": True,
            **_identity_schema_payload(),
            "items": [_authoritative_item()],
            "gaps": [],
            "excluded": [],
            "blockers": [],
            "retrieval_methods_used": ["symbol_exact"],
            "missing_evidence": [],
            "disabled_modes": {},
            "timing": {},
            "trace_refs": [],
        },
        upstream_refs=[feature_ref],
    )
    implementation_ref = store.write(
        feature_id=plan.feature_id,
        artifact_type="implementation_plan",
        payload=implementation.to_artifact_payload(),
        upstream_refs=[feature_ref, evidence_ref],
    )

    assert store.read(request_ref)["data"]["route"] == plan.route
    assert store.read(brief_ref)["data"]["request_type"]["value"] == "runtime diagnostics"
    feature_inputs = store.read(feature_ref)["inputs"]["upstream_artifacts"]
    assert {item["artifact_type"] for item in feature_inputs} == {
        "request_planning_report",
        "mod_planning_brief",
    }
    assert store.read(feature_ref)["data"]["target_behavior"] == plan.user_request
    assert store.read(implementation_ref)["data"]["evidence_refs"] == ["ev_001"]


def test_implementation_plan_blocks_non_authoritative_evidence_pack() -> None:
    plan = RequestPlanner().plan("make Load enable diagnostics")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "legacy_rag_auxiliary",
        "authoritative_for_patch": False,
        **_identity_schema_payload(),
        "items": [
            {
                "evidence_id": "legacy_001",
                "source_type": "target_source",
                "source_path": "mods/BetterHitFeedback/Plugin.cs",
                "symbol": "Load",
                "authority": "legacy_rag_auxiliary",
            }
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "NON_AUTHORITATIVE_EVIDENCE_PACK"


def test_implementation_plan_blocks_item_with_missing_authority() -> None:
    plan = RequestPlanner().plan("make Load enable diagnostics")
    spec = FeatureSpecBuilder().build(plan)
    item = _authoritative_item()
    item.pop("authority")
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [item],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.blockers[0]["code"] == "NON_AUTHORITATIVE_EVIDENCE_ITEM"


def test_implementation_plan_blocks_unprovenanced_raw_evidence_pack() -> None:
    plan = RequestPlanner().plan("make Load enable diagnostics")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        "items": [_authoritative_item()],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.blockers[0]["code"] == "UNVERIFIED_EVIDENCE_PROVENANCE"


def test_implementation_plan_ignores_auxiliary_context_items() -> None:
    plan = RequestPlanner().plan("make Load enable diagnostics")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(),
            {
                "evidence_id": "legacy_001",
                "source_type": "target_source",
                "source_path": "legacy/Plugin.cs",
                "symbol": "Load",
                "authority": "legacy_rag_auxiliary",
            },
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    assert implementation.evidence_refs == ("ev_001",)
    assert implementation.evidence_source_paths == ("mods/BetterHitFeedback/Plugin.cs",)
    assert implementation.target_files == ("mods/BetterHitFeedback/Plugin.cs",)


def test_implementation_plan_normalizes_actual15_nested_edit_evidence() -> None:
    plan = RequestPlanner().plan(
        "add an accurate spread enabled setting",
        target_mod="AccurateCrosshair",
    )
    spec = FeatureSpecBuilder().build(plan)
    configuration = _authoritative_item(
        evidence_id="rg_live_configuration",
        source_path="generated_mods/AccurateCrosshair/Configuration.cs",
        symbol="OnFileChanged",
        span={"start_line": 35, "end_line": 55},
        source_hash="configuration_hash",
        file_hash="configuration_hash",
        declaring_type="Configuration",
        declaring_type_full_name="AccurateCrosshair.Configuration",
        method_name="OnFileChanged",
        parameter_types=None,
        return_type="void",
    )
    spread_inner = _authoritative_item(
        evidence_id="rg_live_spread_inner",
        source_path=(
            "generated_mods/AccurateCrosshair/CrosshairPatches/SpreadPatches.cs"
        ),
        symbol="OverrideCrosshairSize",
        span={"start_line": 115, "end_line": 155},
        source_hash="spread_hash",
        file_hash="spread_hash",
        declaring_type="SpreadPatches",
        declaring_type_full_name=(
            "AccurateCrosshair.CrosshairPatches.SpreadPatches"
        ),
        method_name="OverrideCrosshairSize",
        parameter_types=None,
        return_type="bool",
    )
    spread_outer = _authoritative_item(
        **{
            **spread_inner,
            "evidence_id": "rg_live_spread_outer",
            "span": {"start_line": 115, "end_line": 160},
        }
    )
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [configuration, spread_inner, spread_outer],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    assert len(implementation.proposed_changes) == 2
    assert len(implementation.binding_registry["edit_bindings"]) == 2
    assert implementation.evidence_refs == (
        "rg_live_configuration",
        "rg_live_spread_inner",
        "rg_live_spread_outer",
    )
    spread_binding = implementation.binding_registry["edit_bindings"][1]
    assert spread_binding["binding_id"] == "edit_binding_002"
    assert spread_binding["evidence_id"] == "rg_live_spread_outer"
    assert spread_binding["target_span"] == {
        "start_line": 115,
        "end_line": 160,
    }
    merged_support = next(
        binding
        for binding in implementation.binding_registry["supporting_bindings"]
        if binding["evidence_id"] == "rg_live_spread_inner"
    )
    assert merged_support["target_span"] == {
        "start_line": 115,
        "end_line": 155,
    }
    assert implementation.proposed_changes[1]["supporting_binding_ids"] == [
        merged_support["binding_id"]
    ]


def test_implementation_plan_collapses_exact_duplicate_physical_region() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    symbol_exact = _authoritative_item(
        evidence_id="ev_symbol_exact",
        retrieval_method="symbol_exact",
    )
    text_match = _authoritative_item(
        evidence_id="ev_text_match",
        retrieval_method="text_match",
        symbol=None,
        declaring_type=None,
        declaring_type_full_name=None,
        method_name=None,
        parameter_types=None,
        return_type=None,
        method_kind=None,
    )
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [symbol_exact, text_match],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    assert implementation.evidence_refs == (
        "ev_symbol_exact",
        "ev_text_match",
    )
    assert len(implementation.binding_registry["edit_bindings"]) == 1
    assert len(implementation.proposed_changes) == 1
    assert implementation.binding_registry["edit_bindings"][0]["evidence_id"] == (
        "ev_symbol_exact"
    )
    support = next(
        binding
        for binding in implementation.binding_registry["supporting_bindings"]
        if binding["evidence_id"] == "ev_text_match"
    )
    assert implementation.proposed_changes[0]["supporting_binding_ids"] == [
        support["binding_id"]
    ]


def test_implementation_plan_fails_closed_for_exact_span_different_hashes() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(evidence_id="ev_first"),
            _authoritative_item(
                evidence_id="ev_second",
                source_hash="file_hash_002",
                file_hash="file_hash_002",
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == ["ev_first", "ev_second"]


def test_implementation_plan_fails_closed_for_conflicting_source_hash_aliases() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(
                evidence_id="ev_conflicting_aliases",
                file_hash="file_hash_002",
            ),
            _authoritative_item(evidence_id="ev_other"),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == [
        "ev_conflicting_aliases"
    ]


def test_overlap_does_not_treat_doc_hash_as_physical_source_revision() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(
                evidence_id="ev_doc_only_first",
                source_hash=None,
                file_hash=None,
                doc_hash="canonical_text_hash",
            ),
            _authoritative_item(
                evidence_id="ev_doc_only_second",
                source_hash=None,
                file_hash=None,
                doc_hash="canonical_text_hash",
                retrieval_method="text_match",
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == [
        "ev_doc_only_first",
        "ev_doc_only_second",
    ]


def test_single_doc_hash_only_item_cannot_grant_edit_authority() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(
                evidence_id="ev_doc_only",
                source_hash=None,
                file_hash=None,
                doc_hash="canonical_text_hash",
            )
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.binding_registry is None
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == ["ev_doc_only"]


def test_implementation_plan_fails_closed_for_duplicate_region_identity_conflict() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(evidence_id="ev_load"),
            _authoritative_item(
                evidence_id="ev_save",
                symbol="Save",
                method_name="Save",
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == ["ev_load", "ev_save"]


def test_exact_duplicate_inner_keeps_all_lineage_when_outer_is_retained() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    inner = _authoritative_item(
        evidence_id="ev_inner",
        span={"start_line": 10, "end_line": 20},
    )
    inner_duplicate = _authoritative_item(
        evidence_id="ev_inner_text_match",
        retrieval_method="text_match",
        span={"start_line": 10, "end_line": 20},
        symbol=None,
        declaring_type=None,
        declaring_type_full_name=None,
        method_name=None,
        parameter_types=None,
        return_type=None,
        method_kind=None,
    )
    outer = _authoritative_item(
        evidence_id="ev_outer",
        span={"start_line": 5, "end_line": 25},
    )
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [inner, inner_duplicate, outer],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    assert len(implementation.binding_registry["edit_bindings"]) == 1
    edit_binding = implementation.binding_registry["edit_bindings"][0]
    assert edit_binding["evidence_id"] == "ev_outer"
    assert edit_binding["target_span"] == {"start_line": 5, "end_line": 25}
    support_by_evidence = {
        binding["evidence_id"]: binding["binding_id"]
        for binding in implementation.binding_registry["supporting_bindings"]
    }
    assert implementation.proposed_changes[0]["supporting_binding_ids"] == [
        support_by_evidence["ev_inner"],
        support_by_evidence["ev_inner_text_match"],
    ]


def test_nested_merge_preserves_original_supporting_evidence_order() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    outer = _authoritative_item(
        evidence_id="ev_outer",
        span={"start_line": 5, "end_line": 25},
    )
    inner = _authoritative_item(
        evidence_id="ev_inner",
        span={"start_line": 10, "end_line": 20},
    )
    outer_duplicate = _authoritative_item(
        evidence_id="ev_outer_text_match",
        retrieval_method="text_match",
        span={"start_line": 5, "end_line": 25},
        symbol=None,
        declaring_type=None,
        declaring_type_full_name=None,
        method_name=None,
        parameter_types=None,
        return_type=None,
        method_kind=None,
    )
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [outer, inner, outer_duplicate],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    support_by_evidence = {
        binding["evidence_id"]: binding["binding_id"]
        for binding in implementation.binding_registry["supporting_bindings"]
    }
    assert implementation.proposed_changes[0]["supporting_binding_ids"] == [
        support_by_evidence["ev_inner"],
        support_by_evidence["ev_outer_text_match"],
    ]


def test_duplicate_lineage_is_preserved_when_later_overlap_blocks() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    first = _authoritative_item(
        evidence_id="ev_first",
        span={"start_line": 10, "end_line": 20},
    )
    duplicate = _authoritative_item(
        evidence_id="ev_duplicate",
        retrieval_method="text_match",
        span={"start_line": 10, "end_line": 20},
        symbol=None,
        declaring_type=None,
        declaring_type_full_name=None,
        method_name=None,
        parameter_types=None,
        return_type=None,
        method_kind=None,
    )
    partial_overlap = _authoritative_item(
        evidence_id="ev_partial_overlap",
        span={"start_line": 15, "end_line": 25},
    )
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [first, duplicate, partial_overlap],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == [
        "ev_first",
        "ev_duplicate",
        "ev_partial_overlap",
    ]


def test_implementation_plan_keeps_distinct_nonoverlapping_edit_units() -> None:
    plan = RequestPlanner().plan("change two methods", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    first = _authoritative_item(
        evidence_id="ev_first",
        symbol="First",
        method_name="First",
        span={"start_line": 10, "end_line": 20},
    )
    different_symbol = _authoritative_item(
        evidence_id="ev_second",
        symbol="Second",
        method_name="Second",
        span={"start_line": 30, "end_line": 40},
    )
    different_hash = _authoritative_item(
        evidence_id="ev_third",
        source_path="mods/BetterHitFeedback/Other.cs",
        symbol="Third",
        method_name="Third",
        span={"start_line": 50, "end_line": 60},
        source_hash="file_hash_002",
        file_hash="file_hash_002",
    )
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [first, different_symbol, different_hash],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is False
    assert len(implementation.binding_registry["edit_bindings"]) == 3
    assert len(implementation.proposed_changes) == 3
    assert implementation.binding_registry["supporting_bindings"] == []


def test_nonoverlapping_same_path_different_revisions_fail_closed() -> None:
    plan = RequestPlanner().plan("change two methods", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(
                evidence_id="ev_first",
                symbol="First",
                method_name="First",
                span={"start_line": 10, "end_line": 20},
            ),
            _authoritative_item(
                evidence_id="ev_second",
                symbol="Second",
                method_name="Second",
                span={"start_line": 30, "end_line": 40},
                source_hash="file_hash_002",
                file_hash="file_hash_002",
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == ["ev_first", "ev_second"]


def test_implementation_plan_fails_closed_for_partially_overlapping_edit_units() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(
                evidence_id="ev_first",
                span={"start_line": 10, "end_line": 20},
            ),
            _authoritative_item(
                evidence_id="ev_second",
                span={"start_line": 15, "end_line": 25},
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == ["ev_first", "ev_second"]


def test_implementation_plan_fails_closed_for_nested_different_source_hashes() -> None:
    plan = RequestPlanner().plan("change Load", target_mod="Example")
    spec = FeatureSpecBuilder().build(plan)
    evidence_pack = {
        "readiness": "sufficient",
        "can_generate_patch": True,
        "authority": "mod_agent_deterministic_evidence",
        "authoritative_for_patch": True,
        **_identity_schema_payload(),
        "items": [
            _authoritative_item(
                evidence_id="ev_inner",
                span={"start_line": 10, "end_line": 20},
            ),
            _authoritative_item(
                evidence_id="ev_outer",
                span={"start_line": 5, "end_line": 25},
                source_hash="file_hash_002",
                file_hash="file_hash_002",
            ),
        ],
    }

    implementation = ImplementationPlanner().build(
        feature_spec=spec,
        evidence_pack=evidence_pack,
    )

    assert implementation.blocking is True
    assert implementation.proposed_changes == ()
    assert implementation.blockers[0]["code"] == "EDIT_TARGET_NORMALIZATION_UNSAFE"
    assert implementation.blockers[0]["evidence_ids"] == ["ev_inner", "ev_outer"]
