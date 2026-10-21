"""Controlled selection through real Planning/Patch owners; no model or Build.

Scope: two permitted source units, only one selected for modification.
为什么要有它：选择权限范围的子集须真实形成单项Patch合同，保留文件不应被迫伪造修改。
"""
from dataclasses import replace
import json
import socket

import pytest

from src.libs.llm import ChatResponse
from src.mod_agent.evidence import (
    RgLiveEvidenceProvider,
    build_requirement_coverage_pack,
    materialize_requirement_coverage_projection,
)
from src.mod_agent.execution.patch_apply import materialize_exact_text_patch
from src.mod_agent.local_entry import _patch_generation_obligation_projection
from src.mod_agent.planning.model_selection import ModelPlanningTargetSelector
from src.mod_agent.proposal.code_generator import (
    LLMExactPatchGenerator,
    PatchGenerationError,
    PatchSubmissionValidationError,
)
from src.mod_agent.workflow.graph import _run_create_implementation_plan
from tests.unit.test_fresh_planning_continuation import (
    _direct_graph_state,
    _inputs,
    _positive_findings,
    _projection,
    _source_hashes,
    _write_evidence_revision,
)
from tests.unit.test_model_planning_selection import _Client, _stored_plan


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Minimal candidate-scope checks are offline.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def _prepared_scope(tmp_path, *, select_both=False):
    inputs = _inputs(tmp_path)
    materialized = materialize_requirement_coverage_projection(
        feature_data=inputs.feature_payload,
        request=inputs.feature_payload["user_request"],
        projection=_projection(),
    )
    retrieved = RgLiveEvidenceProvider(inputs.workspace).collect_query_set(
        queries=tuple(dict(row) for row in materialized["queries"]),
        search_root="generated_mods/SyntheticMod",
        max_items=len(materialized["queries"]),
    )
    findings = _positive_findings(materialized, retrieved)
    needed_id = findings["findings"][0]["edit_target_evidence_ids"][0]
    preserved = next(item for item in retrieved.items if item.source_path.endswith("/Preservation.cs"))
    # Scope: permission/selection distinction. 为什么要有它：共同相关源码可供选择，不等于每处都需实改。
    findings["findings"][0]["evidence_ids"] = [needed_id, preserved.evidence_id]
    evidence = build_requirement_coverage_pack(
        query=inputs.feature_payload["user_request"],
        materialized_projection=materialized,
        retrieved_pack=retrieved,
        findings=findings,
    ).to_artifact_payload()
    inputs = replace(inputs, evidence_ref=_write_evidence_revision(inputs, payload=evidence))
    selected = [needed_id, preserved.evidence_id] if select_both else [needed_id]
    client = _Client(override=lambda _payload: {
        "decision": "select_targets", "selected_evidence_ids": selected,
        "rationale": "Controlled fixture selection; it is not a live-model judgment.",
        "verification_questions": [],
    })
    selector = ModelPlanningTargetSelector(
        client=client, workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "selection.json",
        behavior_evaluator_configured=True,
    )
    before = _source_hashes(inputs.workspace)
    state = _run_create_implementation_plan(
        _direct_graph_state(inputs), inputs.store,
        formal_requirement_coverage_required=True,
        planning_target_selection_factory=selector,
    )
    envelope = _stored_plan(inputs, state)
    assert envelope["status"] == "success"
    plan = envelope["data"]
    assert len(client.calls) == 1
    payload = client.calls[0].messages[1].content
    assert set(payload["allowed_edit_evidence_ids"]) == {needed_id, preserved.evidence_id}
    assert [binding["evidence_id"] for binding in plan["binding_registry"]["edit_bindings"]] == selected
    obligations = _patch_generation_obligation_projection(
        feature_spec=inputs.feature_payload, implementation=plan,
        planned_changes=plan["proposed_changes"],
    )
    generator = LLMExactPatchGenerator(None)
    prepared = generator.prepare_native_contract(
        user_request=inputs.feature_payload["user_request"],
        evidence_items=evidence["items"], binding_registry=plan["binding_registry"],
        constraints=obligations["side_effect_boundaries"],
        target_behavior=obligations["target_behavior"],
        acceptance_criteria=obligations["acceptance_criteria"],
        runtime_verification_plan=obligations["runtime_verification_plan"],
        required_change_units=obligations["required_change_units"],
        required_validation_by_binding={
            row["edit_binding_id"]: row["expected_validation"]
            for row in obligations["required_change_units"]
        },
    )
    return inputs, client, plan, obligations, generator, prepared, before, preserved


def _change(prepared, binding_id, *, noop=False):
    before = prepared.materialization_context["binding_evidence_excerpts"][binding_id]
    after = before if noop else before.replace(
        "KeepExistingFeedback();", 'KeepExistingFeedback();\n        SetMarkerColor("pink");',
    )
    assert noop or before != after
    return {"edit_binding_id": binding_id, "after_text": after,
            "rationale": "Implement the selected marker change.",
            "rollback_hint": "Restore the exact original source."}


def test_two_allowed_sources_select_one_required_edit_and_preserve_other(tmp_path):
    inputs, client, plan, obligations, generator, prepared, before, preserved = _prepared_scope(tmp_path)
    units = obligations["required_change_units"]
    assert len(units) == 1
    binding_id = units[0]["edit_binding_id"]
    schema = prepared.response_schema["properties"]["changes"]
    assert schema["minItems"] == schema["maxItems"] == 1
    patch = generator.materialize_prepared_native(
        prepared, ChatResponse(content=json.dumps({"changes": [_change(prepared, binding_id)]}),
                               model="controlled-offline-response"),
    )
    assert len(patch.changes) == 1
    materialized = materialize_exact_text_patch(
        workspace_root=inputs.workspace, diff_intent=patch.changes, implementation_plan=plan,
    )
    behavior_path = inputs.behavior.relative_to(inputs.workspace).as_posix()
    assert set(materialized.files) == {behavior_path}
    assert b'SetMarkerColor("pink");' in materialized.files[behavior_path]
    assert preserved.source_path not in materialized.files
    assert _source_hashes(inputs.workspace) == before
    assert obligations["acceptance_criteria"] == inputs.feature_payload["acceptance_criteria"]
    assert "PreserveFeedback must remain unchanged." in obligations["side_effect_boundaries"]
    # Scope: delivered selector input. 为什么要有它：端到端合同成立之外，模型还必须看到选中即必改的真实后果。
    rule = client.calls[0].messages[1].content["candidate_scope_contract"]["selected_targets_rule"]
    assert rule["effect"] == "required_non_noop_edit_per_selected_source_unit"
    assert rule["scope"] and rule["why"].startswith("为什么要有它：")


def test_selected_unit_still_rejects_noop_replacement(tmp_path):
    _, _, _, obligations, generator, prepared, _, _ = _prepared_scope(tmp_path)
    binding = obligations["required_change_units"][0]["edit_binding_id"]
    response = ChatResponse(content=json.dumps({"changes": [_change(prepared, binding, noop=True)]}),
                            model="controlled-offline-response")
    with pytest.raises(PatchSubmissionValidationError) as caught:
        generator.materialize_prepared_native(prepared, response)
    assert caught.value.diagnostics["failure_classification"] == "native_after_text_not_replacement"


def test_selecting_two_units_still_rejects_incomplete_required_coverage(tmp_path):
    _, _, _, obligations, generator, prepared, _, _ = _prepared_scope(tmp_path, select_both=True)
    assert len(obligations["required_change_units"]) == 2
    assert prepared.response_schema["properties"]["changes"]["minItems"] == 2
    assert prepared.response_schema["properties"]["changes"]["maxItems"] == 2
    binding = obligations["required_change_units"][0]["edit_binding_id"]
    with pytest.raises(PatchGenerationError):
        generator.materialize_prepared_native(
            prepared, ChatResponse(content=json.dumps({"changes": [_change(prepared, binding)]}),
                                   model="controlled-offline-response"),
        )
