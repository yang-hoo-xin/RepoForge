"""Real Graph/adapter/restore callers with scripted responses; no live model or Build."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.fresh_planning_continuation import (
    PatchContinuationInputV1,
    _PatchUpstreamExecution,
    run_fresh_patch_continuation,
)
from src.mod_agent.interrupted_patch_admission import prepare_retained_generation
from src.mod_agent.local_entry import (
    LocalEntryError,
    _patch_generation_obligation_projection,
    make_exact_no_write_dry_run_factory,
)
from src.mod_agent.patch_upstream_configuration import PatchUpstreamConfiguration
from src.mod_agent.planning.model_selection import ModelPlanningTargetSelector
from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator
from src.mod_agent.workflow.graph import _run_create_implementation_plan
from tests.unit import test_fresh_planning_continuation as fixture
from tests.unit.test_model_planning_selection import _Client
from tests.unit.test_patch_behavior_feedback import _adapter

HINT = "FeatureBehavior must remain unchanged even when the requested marker changes."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    # Scope: all providers in these checks are local scripted objects.
    # 为什么要有它：真实入口验证不能意外调用外部模型，也不能把受控响应计成实际产品实验。
    def forbidden(*args, **kwargs):
        raise AssertionError("No network or actual model is allowed")
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


class Client(fixture._PatchAgentTurnClient):
    def complete_turn(self, request):
        response = super().complete_turn(request)
        self.last_response = response
        return response


def planned_case(tmp_path, monkeypatch, policy):
    original = fixture._projection()
    preservation = [row for row in original["requirements"] if row["role"] == "preservation"]
    assert preservation
    preservation[0]["text"] = HINT
    with monkeypatch.context() as temporary:
        temporary.setattr(fixture, "_projection", lambda: deepcopy(original))
        inputs = fixture._inputs(tmp_path)
    selector = ModelPlanningTargetSelector(client=_Client(), workspace_root=inputs.workspace,
        receipt_path=inputs.store.root/"controlled-planning.json", constraint_policy=policy)
    state = _run_create_implementation_plan(fixture._direct_graph_state(inputs), inputs.store,
        formal_requirement_coverage_required=True, planning_target_selection_factory=selector)
    assert not state["blockers"]
    ref = ArtifactRef(feature_id=inputs.feature_ref.feature_id, artifact_type="implementation_plan",
        path=state["artifacts"]["implementation_plan"], artifact_hash=state["artifact_hashes"]["implementation_plan"])
    plan = inputs.store.read(ref)["data"]
    request = PatchContinuationInputV1(contract_version=fixture.PATCH_CONTINUATION_CONTRACT_VERSION, artifact_store_root=str(inputs.store.root),
        workspace_root=str(inputs.workspace), feature_spec_ref=inputs.feature_ref,
        evidence_pack_ref=inputs.evidence_ref, implementation_plan_ref=ref)
    return inputs, plan, request


def run_patch(inputs, request):
    before = fixture._source_hashes(inputs.workspace)
    client = Client()
    adapter = _adapter(inputs, client, None)
    result = run_fresh_patch_continuation(request, patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
        max_model_calls=adapter.patch_agent_loop_policy.max_total_model_turns)
    assert result.status == "terminal", result
    assert fixture._source_hashes(inputs.workspace) == before
    assert len(client.requests) == 2
    return client, adapter, result


@pytest.mark.parametrize("policy", [None, "source_priority_v1"], ids=["baseline-a", "source-priority"])
def test_graph_plan_and_normal_local_adapter_preserve_selected_constraint_policy(tmp_path, monkeypatch, policy):
    inputs, plan, request = planned_case(tmp_path, monkeypatch, policy)
    client, adapter, result = run_patch(inputs, request)
    prompt = json.loads(client.requests[0].messages[1].content)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    obligations = _patch_generation_obligation_projection(feature_spec=inputs.feature_payload,
        implementation=plan, planned_changes=plan["proposed_changes"], evidence_pack=evidence)
    assert prompt["request"] == inputs.feature_payload["user_request"]
    assert prompt["acceptance_criteria"] == inputs.feature_payload["acceptance_criteria"]
    assert prompt["runtime_verification_plan"] == inputs.feature_payload["runtime_verification_plan"]
    if policy is None:
        assert "constraint_context" not in plan and "constraint_context" not in prompt and "constraint_context" not in obligations
        assert HINT in prompt["constraints"]
    else:
        assert HINT not in prompt["constraints"]
        assert prompt["constraint_context"]["projection_version"] == "source_priority_model_view_v2"
        assert prompt["constraint_context"]["omitted_model_preservation_hints"]
        assert all("text" not in row for row in prompt["constraint_context"]["omitted_model_preservation_hints"])
        assert HINT not in json.dumps([asdict(item) for item in client.requests])
        assert plan["constraint_context"]["model_preservation_hints"][0]["text"] == HINT
        assert obligations["constraint_context"] == plan["constraint_context"]
        assert all(text in prompt["constraints"] for text in inputs.feature_payload["non_goals"])
    assert adapter.patch_attempt_receipt()["model_call_count"] == 2
    assert result.dry_run_artifact is not None


def test_restore_prepares_and_materializes_with_the_complete_retained_context(tmp_path, monkeypatch):
    inputs, plan, request = planned_case(tmp_path, monkeypatch, "source_priority_v1")
    client, _, _ = run_patch(inputs, request)
    observed = []
    original = LLMExactPatchGenerator.prepare_native_contract
    def record(self, **kwargs):
        observed.append(deepcopy(kwargs.get("constraint_context")))
        return original(self, **kwargs)
    monkeypatch.setattr(LLMExactPatchGenerator, "prepare_native_contract", record)
    response = client.last_response
    _, prepared, generated = prepare_retained_generation(inputs.feature_payload,
        inputs.store.read(inputs.evidence_ref)["data"], plan,
        {"content": response.content, "model": response.model, "usage": None, "raw_response": response.raw_response})
    assert observed == [plan["constraint_context"]]
    assert generated.changes
    restored = json.loads(prepared.messages[1].content)
    original_prompt = json.loads(client.requests[0].messages[1].content)
    assert restored["constraint_context"] == original_prompt["constraint_context"]
    assert restored["constraints"] == original_prompt["constraints"]
    assert HINT not in json.dumps([asdict(message) for message in prepared.messages])
    assert plan["constraint_context"]["model_preservation_hints"][0]["text"] == HINT


@pytest.mark.parametrize("change", ["evidence", "user_request", "acceptance", "hard_boundary", "missing_evidence"])
def test_normal_obligation_projection_rejects_changed_sources_before_native_prepare(tmp_path, monkeypatch, change):
    inputs, plan, _ = planned_case(tmp_path, monkeypatch, "source_priority_v1")
    feature = deepcopy(inputs.feature_payload)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    if change == "evidence":
        evidence["requirement_coverage"]["requirements"][0]["text"] = "changed origin"
    elif change == "user_request":
        feature["user_request"] += " changed"
    elif change == "acceptance":
        feature["acceptance_criteria"] = ["weaker acceptance"]
    elif change == "hard_boundary":
        plan["side_effect_boundaries"] = []
    else:
        evidence = None
    with pytest.raises(LocalEntryError):
        _patch_generation_obligation_projection(feature_spec=feature, implementation=plan,
            planned_changes=plan["proposed_changes"], evidence_pack=evidence)


@pytest.mark.parametrize("policy", [None, "source_priority_v1"])
def test_actual_fresh_upstream_assembly_preserves_the_admitted_plan_policy_without_calls(tmp_path, monkeypatch, policy):
    inputs, plan, request = planned_case(tmp_path, monkeypatch, policy)
    adapter = _adapter(inputs, Client(), None)
    path = tmp_path/"upstream.json"
    path.write_text(json.dumps({"schema_version":"patch_upstream_configuration_v1",
        "planning_profile":asdict(adapter.profile),"evidence":None}), encoding="utf-8")
    configuration = PatchUpstreamConfiguration.load(path)
    request = replace(request, patch_upstream_configuration_ref=configuration.reference)
    events = []
    execution = _PatchUpstreamExecution(request, adapter, configuration, SimpleNamespace(record=events.append), maximum=5)
    selector = execution.providers["planning_target_selection_factory"]
    assert execution.used == 0 and selector.client.calls == 0
    assert selector.constraint_policy == policy
    model_input = selector.prepare(inputs.feature_payload, inputs.store.read(inputs.evidence_ref)["data"]).messages[1].content
    assert ("constraint_context" in model_input) is (policy is not None)
    if policy is not None:
        assert selector.constraint_policy == plan["constraint_context"]["policy"]
        assert HINT not in json.dumps(model_input)
        assert model_input["feature"] == inputs.feature_payload
        assert model_input["constraint_context"]["projection_version"] == "source_priority_model_view_v2"


def test_initial_planning_model_view_omits_only_derived_preservation_text(tmp_path, monkeypatch):
    inputs, plan, _request = planned_case(tmp_path, monkeypatch, "source_priority_v1")
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    before = deepcopy((inputs.feature_payload, evidence, plan))
    selector = ModelPlanningTargetSelector(client=_Client(), workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "new-planning.json", constraint_policy="source_priority_v1")
    request = selector.prepare(inputs.feature_payload, evidence)
    payload = request.messages[1].content
    assert HINT not in json.dumps(asdict(request))
    assert payload["feature"] == inputs.feature_payload
    original_rows = evidence["requirement_coverage"]["requirements"]
    for original, selected in zip(original_rows, payload["requirements"], strict=True):
        if original["role"] == "preservation":
            assert "text" not in selected
            assert {key: value for key, value in selected.items() if key != "text_projection"} == {
                key: value for key, value in original.items() if key != "text"}
        else:
            assert selected == original
    assert (inputs.feature_payload, evidence, plan) == before


def test_upstream_planning_model_view_does_not_reintroduce_raw_context(tmp_path, monkeypatch):
    inputs, plan, _request = planned_case(tmp_path, monkeypatch, "source_priority_v1")
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    selector = ModelPlanningTargetSelector(client=_Client(), workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "upstream-selection.json", constraint_policy="source_priority_v1")
    assessment = {"current_plan": deepcopy(plan), "retain_current_plan_allowed": True,
        "current_failure_available": False}
    selector.upstream_assessment = deepcopy(assessment)
    requested = selector.prepare(inputs.feature_payload, evidence)
    visible = requested.messages[1].content["upstream_assessment"]
    assert HINT not in json.dumps(asdict(requested))
    assert visible["current_plan"]["constraint_context"]["projection_version"] == "source_priority_model_view_v2"
    baseline = ModelPlanningTargetSelector(client=_Client(), workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "unchanged-upstream-projection.json")
    baseline.upstream_assessment = deepcopy(assessment)
    expected = baseline.prepare(inputs.feature_payload, evidence).messages[1].content["upstream_assessment"]["current_plan"]
    assert {key: value for key, value in visible["current_plan"].items() if key != "constraint_context"} == {
        key: value for key, value in expected.items() if key != "constraint_context"}
    assert selector.upstream_assessment == assessment
    assert plan["constraint_context"]["model_preservation_hints"][0]["text"] == HINT
