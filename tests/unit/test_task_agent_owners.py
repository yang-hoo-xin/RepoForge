"""Actual upstream assembly and leaf APIs with controlled native model responses.

为什么要有它：验证真实搜索/重评/规划调用链与共享计费，不以unsupported占位或假成功替代；没有真实模型或Build。
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.libs.llm.agent_turn import AgentTurnResponse
from src.mod_agent import patch_upstream_configuration as upstream_module
from src.mod_agent.local_entry import LOCAL_OLLAMA_PATCH_PROFILE_VERSION
from src.mod_agent.patch_upstream_configuration import PatchUpstreamConfiguration
from src.mod_agent.task_agent_owners import make_task_agent_owners
from tests.unit import test_task_agent_tools as tool_fixtures
from tests.unit.test_task_agent_loop import call, response

setup = tool_fixtures.setup


@pytest.fixture
def assembled(setup, monkeypatch):
    inputs, session, evaluator, budget = setup
    settings = {"profile_version": LOCAL_OLLAMA_PATCH_PROFILE_VERSION, "base_url": "http://127.0.0.1:11434",
                "model": "controlled", "context_window": 8192, "output_budget": 1024, "timeout_seconds": 30.0}
    path = inputs.store.root / "upstream-config.json"
    path.write_text(json.dumps({"schema_version": "patch_upstream_configuration_v1", "planning_profile": settings,
        "evidence": {"profile": settings, "search_root": "generated_mods/SyntheticMod", "max_additional_queries": 2}}))
    identity = {"name": "controlled", "model": "controlled", "digest": "a" * 64,
        "modified_at": "2026-09-04T00:00:00Z", "size": 1, "details": {"context_length": 8192},
        "capabilities": ["completion", "tools"], "base_url": settings["base_url"]}
    monkeypatch.setattr(upstream_module, "read_ollama_model_identity", lambda *_: deepcopy(identity))
    requests, roles, events = [], [], []
    mode = {"planning": "select_targets", "invalid_selection": False, "findings_missing": False}

    class Scripted:
        provider, model = "ollama_chat_local", "controlled"

        def complete_turn(self, request):
            requests.append(request)
            payload = request.messages[1].content
            name = request.response_format.name
            if name == "planning_target_selection":
                question_key = "verification_questions" if "verification_questions" in request.response_format.schema["properties"] else "missing_evidence"
                selected = ([] if mode["planning"] in {"retain_current_plan", "needs_evidence"} else
                    ["not_registered"] if mode["invalid_selection"] else payload["allowed_edit_evidence_ids"])
                result = {"decision": mode["planning"], "selected_evidence_ids": selected,
                          "rationale": "Controlled actual planning response",
                          question_key: ["Controlled missing fact"] if mode["planning"] == "needs_evidence" else []}
            elif name == "upstream_evidence_queries":
                result = {"queries": [{"query": "ChangeMarker", "requirement_id": payload["requirements"][0]["requirement_id"]}]}
            elif name == "requirement_findings":
                indexes = request.response_format.schema["properties"]["findings"]["items"]["properties"]["relation_index"]["enum"]
                old = session.documents.evidence_pack["requirement_coverage"]["findings"]
                result = {"findings": [{"relation_index": index,
                    **{key: deepcopy(old[index - 1][key]) for key in
                       ("relationship", "evidence_ids", "edit_target_evidence_ids", "unresolved_gaps")}} for index in indexes]}
                if mode["findings_missing"]:
                    result["findings"][0].update(relationship="missing", edit_target_evidence_ids=[],
                                                unresolved_gaps=["Controlled missing API fact"])
            else:
                pytest.fail("Unexpected semantic call: " + name)
            return AgentTurnResponse(self.provider, self.model, json.dumps(result))

    monkeypatch.setattr(upstream_module.LocalOllamaInitialPatchAdapter, "_default_agent_turn_client",
                        staticmethod(lambda _profile: Scripted()))

    def reserve(role):
        roles.append(role)
        budget.reserve()

    owners = make_task_agent_owners(store=inputs.store, workspace_root=inputs.workspace,
        configuration=PatchUpstreamConfiguration.load(path), initial_context=session.context,
        failure_context_provider=session.upstream_failure_context, reserve_model_call=reserve,
        budget_snapshot=budget.snapshot, record=events.append, output_root=inputs.store.root / "actual-owners")
    session.source_search_owner, session.upstream_owner = owners.source_search, owners.upstream
    return inputs, session, owners, requests, roles, events, mode


def invoke_upstream(session, owner="planning"):
    arguments = {"context_id": session.context_id, "request": tool_fixtures.upstream_request(session, owner)}
    session.response_admission(response(call("upstream", arguments, "actual-upstream")))
    action = tool_fixtures.action(session, "upstream")
    return action.execute(action.prepare(arguments).value)


def test_factory_uses_real_source_provider_and_zero_model_search(assembled):
    _, session, owners, requests, roles, _, _ = assembled
    result = owners.source_search(query="ChangeMarker", context=session.documents)
    assert result.payload["items"] and result.source_refs
    assert all(Path(row["path"]).is_file() for row in result.source_refs)
    assert json.loads(Path(result.sidecar_ref["path"]).read_text())["semantic_evidence_or_plan_updated"] is False
    assert not requests and not roles


def current_owners(assembled):
    inputs, session, owners, *_ = assembled
    session.candidate_mode = "current"
    owners.current_code_view_provider = session.current_code_view
    owners.validate_context_transfer = session.validate_context_transfer
    candidate_id = tool_fixtures.submit(session, correct=False)
    return inputs, session, owners, candidate_id


def test_current_candidate_search_observes_changed_bytes_with_real_scoped_refs(assembled):
    inputs, session, owners, candidate_id = current_owners(assembled)
    before = inputs.behavior.read_bytes()
    selected = tool_fixtures.action(session, "source_search")
    result = selected.execute(selected.prepare({"query": "RecordMissingMarker"}).value)
    assert result["items"]
    result = owners.source_search(query="RecordMissingMarker", context=session.documents)
    assert result.payload["current_code_view"]["candidate_ref"] == session.candidates[candidate_id]["ref"]
    for item in result.payload["items"]:
        assert item["logical_source_path"] == "generated_mods/SyntheticMod/Behavior.cs"
        assert "RecordMissingMarker" in item["excerpt"]
        assert Path(item["source_path"]).is_relative_to(inputs.store.root)
        assert item["source_ref"] in result.source_refs
        assert "RecordMissingMarker" in Path(item["source_path"]).read_text()
    assert inputs.behavior.read_bytes() == before
    assert not assembled[3] and not assembled[4]


@pytest.mark.parametrize("owner", ["planning", "evidence"])
def test_real_upstream_models_observe_current_code_and_preserve_original_apply_preimage(assembled, owner):
    inputs, session, _, first = current_owners(assembled)
    original, before = session.context, inputs.behavior.read_bytes()
    result = invoke_upstream(session, owner)
    assert result["upstream_outcome"] == "replanned", result
    assert result["current_code_preserved"] and result["new_check_required"]
    assert session.current_check is None and session.current_candidate != first
    current = session.current_code_view()
    assert current["materialized_code_identity"] == session.candidates[first]["data"]["materialized_code_identity"]
    requests = assembled[3]
    planning = requests[-1].messages[1].content["upstream_assessment"]
    assert "RecordMissingMarker" in planning["current_code_view"]["files"][0]["text"]
    assert "immutable Apply preimage" in planning["current_code_view"]["source_semantics"]
    if owner == "evidence":
        query = requests[0].messages[1].content
        assert "RecordMissingMarker" in query["failure_feedback"]["current_code_view"]["files"][0]["text"]
        assert planning["current_source_observations"][0]["execution"]["all_queries_executed"] is True
        assert any("RecordMissingMarker" in item["excerpt"]
            for item in planning["current_source_observations"][0]["items"])
    assert session.context != original and inputs.behavior.read_bytes() == before
    assert all(item["source_hash"] != current["files"][0]["source_hash"]
        for item in session.documents.evidence_pack["items"] if item["source_path"].endswith("Behavior.cs"))


def test_unmappable_replan_is_preflighted_before_publishing_and_keeps_current_code(assembled):
    inputs, session, owners, candidate_id = current_owners(assembled)
    original = session.context

    def reject(**_payload):
        assert inputs.store.check_freshness(original.evidence_pack_ref).is_fresh
        assert inputs.store.check_freshness(original.implementation_plan_ref).is_fresh
        raise ValueError("TASK_CURRENT_TRANSFER_SCOPE_NOT_COVERED")

    owners.validate_context_transfer = reject
    result = invoke_upstream(session, "evidence")
    assert result["upstream_outcome"] == "retained_context"
    assert result["model_feedback"]["proposed_context_published"] is False
    assert result["candidate_submission_allowed"] is True
    assert session.context == original and session.current_candidate == candidate_id
    assert inputs.store.check_freshness(original.evidence_pack_ref).is_fresh
    assert inputs.store.check_freshness(original.implementation_plan_ref).is_fresh


def test_second_real_replan_receives_versioned_historical_failure_not_current_acceptance(assembled):
    inputs, session, _, candidate_id = current_owners(assembled)
    check = tool_fixtures.action(session, "candidate_check")
    failed = check.execute(check.prepare({"candidate_id": candidate_id}).value)
    assert failed["status"] == "behavior_failed"
    original_check = deepcopy(session.current_check)
    first_context = session.context_id
    invoke_upstream(session)
    assert session.current_check is None
    invoke_upstream(session)
    assessment = assembled[3][-1].messages[1].content["upstream_assessment"]
    assert assessment["diagnostic_failure_available"] is True
    assert assessment["current_failure_available"] is False
    assert assessment["failure_version"]["origin"] == "historical_checked_candidate_not_current_acceptance"
    assert assessment["failure_version"]["context_id"] == first_context
    assert assessment["failure_version"]["failure_check_ref"] == {"sha256": original_check["sha256"]}
    assert assessment["failure_version"]["eligible_for_current_completion"] is False
    assert assessment["current_code_view"]["last_checked_failure"]["candidate_id"] == candidate_id
    assert session.current_check is None
    repairs = [inputs.store.read(reference)["data"] for reference in
        inputs.store.list_feature_artifact_revisions(inputs.feature_ref.feature_id, "repair_plan")]
    diagnostic = next(row["repair_context"] for row in repairs if row["retry_count"] == 1)
    assert diagnostic["diagnostic_origin"] == "historical_checked_candidate_not_current_acceptance"
    assert diagnostic["failed_check_ref"] == original_check
    assert diagnostic["failed_context_id"] == first_context
    assert diagnostic["eligible_for_current_completion"] is False


@pytest.mark.parametrize("decision", ["select_targets", "retain_current_plan"])
def test_real_planning_owner_selects_or_retains_with_actual_provider_record_and_budget(assembled, decision):
    inputs, session, owners, requests, roles, events, mode = assembled
    mode["planning"] = decision
    original = session.context
    result = invoke_upstream(session)
    assert result["upstream_outcome"] == ("replanned" if decision == "select_targets" else "retained_context")
    assert session.context.feature_spec_ref == original.feature_spec_ref
    assert session.context.evidence_pack_ref == original.evidence_pack_ref
    assert len(requests) == 1 and roles == ["planning"]
    model_assessment = requests[0].messages[1].content["upstream_assessment"]
    original_plan = inputs.store.read(original.implementation_plan_ref)["data"]
    assert model_assessment["selected_artifact_hashes"] == {
        name: getattr(original, name + "_ref").artifact_hash for name in ("feature_spec", "evidence_pack", "implementation_plan")}
    assert model_assessment["current_plan"]["side_effect_boundaries"] == original_plan["side_effect_boundaries"]
    assert [row["edit_binding_id"] for row in model_assessment["current_plan"]["proposed_changes"]] == [
        row["edit_binding_id"] for row in original_plan["proposed_changes"]]
    assert any(row.get("phase") == "patch_upstream_provider_turn" and row.get("status") == "responded" for row in events)
    repair = inputs.store.list_feature_artifact_revisions(original.feature_spec_ref.feature_id, "repair_plan")[-1]
    payload = inputs.store.read(repair)["data"]
    assert payload["source_proposal"] == {} and payload["failure_source"] == "patch_upstream_request"
    assert payload["allowed_next_stage"] is None and not payload["scope_expansion_allowed"]
    assert owners.assessment_count == 1


def test_real_evidence_reassessment_and_planning_both_use_common_budget(assembled):
    _, session, _, requests, roles, events, _ = assembled
    result = invoke_upstream(session, "evidence")
    assert result["upstream_outcome"] == "replanned", result
    assert roles == ["evidence_query", "evidence_findings", "planning"]
    assert len(requests) == 3
    assert [row["new_model_calls"] for row in events if row.get("phase") == "task_upstream_completed"] == [3]
    assert session.documents.evidence_pack["budget_summary"]["reassessment"]["all_queries_executed"] is True


def test_planning_cannot_return_an_id_outside_actual_allowed_evidence(assembled):
    _, session, _, requests, roles, _, mode = assembled
    original = session.context
    mode["invalid_selection"] = True
    with pytest.raises(ValueError, match="outside"):
        invoke_upstream(session)
    assert session.context == original and len(requests) == 1 and roles == ["planning"]


def test_upstream_cannot_replace_actual_model_request_with_claimed_success(assembled):
    _, session, owners, requests, roles, _, _ = assembled
    owners.failure_context_provider = lambda: {"status": "success"}
    with pytest.raises(ValueError, match="REQUEST_NOT_BOUND"):
        owners.upstream(request=tool_fixtures.upstream_request(session), context=session.documents)
    assert not requests and not roles


@pytest.mark.parametrize("block_at", ["evidence", "planning"])
def test_known_upstream_block_does_not_move_evidence_head_then_claim_old_plan_fresh(assembled, block_at):
    inputs, session, _, _, roles, _, mode = assembled
    original = session.context
    mode["findings_missing"] = block_at == "evidence"
    mode["planning"] = "needs_evidence" if block_at == "planning" else "select_targets"
    result = invoke_upstream(session, "evidence")
    assert result["upstream_outcome"] == "blocked" and result["candidate_submission_allowed"] is False
    assert session.context == original
    assert inputs.store.check_freshness(original.evidence_pack_ref).is_fresh
    assert inputs.store.check_freshness(original.implementation_plan_ref).is_fresh
    assert roles == ["evidence_query", "evidence_findings"] + (["planning"] if block_at == "planning" else [])
    with pytest.raises(ValueError, match="REASSESSMENT_UNRESOLVED"):
        tool_fixtures.action(session, "candidate_submit").prepare(tool_fixtures.patch(session))


def test_explicit_product_providers_keep_original_augmentation_and_charge_once(assembled):
    inputs, session, owners, requests, roles, events, _ = assembled
    def augmentation(*_):
        return None

    def documents(*_):
        return None

    owners.evidence_factory.source_augmentation_factory = augmentation
    owners.evidence_factory.document_augmentation_factory = documents
    injected = make_task_agent_owners(store=inputs.store, workspace_root=inputs.workspace,
        providers=owners.providers, validate_providers=owners.validate_configuration,
        initial_context=session.context, failure_context_provider=session.upstream_failure_context,
        reserve_model_call=lambda _role: pytest.fail("Already assembled providers must not be wrapped again"),
        budget_snapshot=session.budget_snapshot, record=events.append, output_root=inputs.store.root / "injected-owners")
    assert injected.evidence_factory is owners.evidence_factory
    assert injected.evidence_factory.source_augmentation_factory is augmentation
    assert injected.evidence_factory.document_augmentation_factory is documents
    session.source_search_owner, session.upstream_owner = injected.source_search, injected.upstream
    invoke_upstream(session)
    assert roles == ["planning"] and len(requests) == 1
