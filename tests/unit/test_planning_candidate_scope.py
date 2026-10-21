"""Candidate edit scope defers behavioral uncertainty to the real Patch evaluator."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.fresh_planning_continuation import (
    PATCH_CONTINUATION_CONTRACT_VERSION,
    PatchContinuationInputV1,
    run_fresh_patch_continuation,
)
from src.mod_agent.local_entry import make_exact_no_write_dry_run_factory
from src.mod_agent.planning.model_selection import (
    ModelPlanningTargetSelector,
    PlanningTargetSelectionError,
)
from src.mod_agent.workflow.graph import _run_create_implementation_plan
from src.mod_agent.workflow.transitions import route_after_implementation_plan
from tests.unit.test_fresh_planning_continuation import _direct_graph_state, _inputs, _source_hashes
from tests.unit.test_model_planning_selection import _Client, _stored_plan
from tests.unit.test_patch_behavior_feedback import _adapter, _BehaviorClient, _BehaviorOracle
from tests.unit.test_planning_target_correction import _Client as _CorrectionClient
from tests.unit.test_planning_target_correction import _overlap_inputs, _selection

# Observed dev03_flow questions, not answers or recommended edit targets.
QUESTIONS = [
    "Information about how BepInEx ConfigFile handles default values vs. missing entries",
    "Details on how to detect if a configuration key has been explicitly set by user vs. using default",
    "Evidence showing current initialization flow of Configuration class and how defaults are applied",
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Candidate scope tests must not call a model or network.")
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


def _candidate_response(payload):
    return {key: value for key, value in payload.items() if key != "missing_evidence"} | {
        "verification_questions": list(QUESTIONS),
    }


def _run(inputs, client, *, candidate=False):
    selector = ModelPlanningTargetSelector(client=client, workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "selection.json",
        **({"behavior_evaluator_configured": True} if candidate else {}))
    state = _run_create_implementation_plan(_direct_graph_state(inputs), inputs.store,
        planning_target_selection_factory=selector)
    return selector, state, _stored_plan(inputs, state)


def test_observed_api_questions_block_legacy_but_not_candidate_scope(tmp_path):
    legacy = _inputs(tmp_path / "legacy")
    legacy_client = _Client(override=lambda payload: {**payload, "decision": "needs_evidence",
        "selected_evidence_ids": [], "missing_evidence": list(QUESTIONS)})
    _, old_state, old_plan = _run(legacy, legacy_client)
    assert old_plan["status"] == "blocked"
    assert old_state["blockers"][0]["code"] == "PLANNING_MODEL_NEEDS_EVIDENCE"
    assert "verification_questions" not in old_plan["data"]

    inputs = _inputs(tmp_path / "candidate")
    client = _Client(override=_candidate_response)
    selector, state, plan = _run(inputs, client, candidate=True)
    assert len(client.calls) == 1
    assert plan["status"] == "success"
    assert route_after_implementation_plan(state) == "continue"
    assert plan["data"]["verification_questions"] == QUESTIONS
    assert not set(QUESTIONS).intersection(plan["data"]["side_effect_boundaries"])
    assert plan["data"]["open_questions"] == []
    assert selector.verification_questions == tuple(QUESTIONS)
    schema = client.calls[0].response_format.schema
    assert schema["properties"]["decision"]["enum"] == ["select_targets"]
    assert "missing_evidence" not in schema["properties"]
    assert schema["properties"]["selected_evidence_ids"]["minItems"] == 1
    capability = client.calls[0].messages[1].content["candidate_scope_contract"]
    assert capability["behavior_evaluator"] == {"configured": True, "execution_status": "not_established_by_planning"}
    assert "factory" not in json.dumps(capability)


@pytest.mark.parametrize("fault", ["empty", "outside", "duplicate", "needs_evidence", "question_shape"])
def test_candidate_scope_still_rejects_invalid_selection(tmp_path, fault):
    inputs = _inputs(tmp_path)

    def response(payload):
        result = _candidate_response(payload)
        if fault == "empty":
            result["selected_evidence_ids"] = []
        elif fault == "outside":
            result["selected_evidence_ids"] = ["outside_current_source"]
        elif fault == "duplicate":
            result["selected_evidence_ids"] *= 2
        elif fault == "needs_evidence":
            result["decision"] = "needs_evidence"
            result["selected_evidence_ids"] = []
        else:
            result["verification_questions"] = [False]
        return result

    client = _Client(override=response)
    selector, state, plan = _run(inputs, client, candidate=True)
    assert len(client.calls) == 1
    assert plan["status"] == "blocked"
    assert plan["data"]["proposed_changes"] == []
    assert "verification_questions" not in plan["data"]
    assert route_after_implementation_plan(state) == "blocked"
    assert selector.verification_questions == ()


@pytest.mark.parametrize("fault", ["missing_source", "drift_before", "drift_after", "outside_root"])
def test_candidate_contract_does_not_admit_missing_or_changed_source(tmp_path, fault):
    inputs = _inputs(tmp_path)

    def mutate():
        inputs.behavior.write_text(inputs.behavior.read_text() + "// drift\n")

    client = _Client(override=_candidate_response, mutate=mutate if fault == "drift_after" else None)
    if fault == "missing_source":
        inputs.behavior.unlink()
    elif fault == "drift_before":
        mutate()
    elif fault == "outside_root":
        evidence = inputs.store.read(inputs.evidence_ref)["data"]
        evidence["items"][0]["source_path"] = "../outside.cs"
        selector = ModelPlanningTargetSelector(client=client, workspace_root=inputs.workspace,
            receipt_path=inputs.store.root / "selection.json", behavior_evaluator_configured=True)
        with pytest.raises((PlanningTargetSelectionError, ValueError)):
            selector(inputs.feature_payload, evidence)
        assert client.calls == []
        return
    _, _, plan = _run(inputs, client, candidate=True)
    assert plan["status"] == "blocked"
    assert len(client.calls) == int(fault == "drift_after")


def test_candidate_requires_current_formal_evidence_before_model(tmp_path):
    inputs = _inputs(tmp_path)
    client = _Client(override=_candidate_response)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    evidence["can_generate_patch"] = False
    selector = ModelPlanningTargetSelector(client=client, workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "selection.json", behavior_evaluator_configured=True)
    with pytest.raises(PlanningTargetSelectionError) as raised:
        selector(inputs.feature_payload, evidence)
    # Coverage rejects an inconsistent readiness flag before the selector's own check.
    assert raised.value.code in {"PLANNING_MODEL_EVIDENCE_INVALID", "PLANNING_MODEL_SELECTION_FAILED"}
    assert client.calls == []


def test_compiler_candidate_planning_preserves_insufficient_findings_and_exact_scope(tmp_path):
    from tests.unit.test_compilation_candidate_continuation import _candidate_inputs

    inputs, _ = _candidate_inputs(tmp_path)
    original = inputs.store.read(inputs.evidence_ref)
    scope = original["data"]["budget_summary"]["compilation_repair_candidate"]
    client = _Client(override=_candidate_response)
    selector, state, plan = _run(inputs, client, candidate=True)

    assert plan["status"] == "success"
    assert route_after_implementation_plan(state) == "continue"
    assert len(client.calls) == 1
    payload = client.calls[0].messages[1].content
    assert payload["allowed_edit_evidence_ids"] == scope["evidence_ids"]
    assert payload["compilation_candidate_scope"]["semantic_coverage_established"] is False
    assert "worker_configuration_ref" not in payload["compilation_candidate_scope"]
    assert {row["evidence_id"] for row in plan["data"]["binding_registry"]["edit_bindings"]} == set(scope["evidence_ids"])
    assert plan["data"]["verification_questions"] == list(dict.fromkeys([
        *scope["verification_questions"], *QUESTIONS]))
    assert selector.verification_questions == tuple(plan["data"]["verification_questions"])
    assert inputs.store.read(inputs.evidence_ref) == original
    assert original["status"] == "blocked"
    assert original["data"]["readiness"] == "insufficient"
    assert original["data"]["can_generate_patch"] is False
    assert original["data"]["authoritative_for_patch"] is False


def test_compiler_candidate_cannot_select_unrelated_observed_source(tmp_path):
    from tests.unit.test_compilation_candidate_continuation import _candidate_inputs

    inputs, _ = _candidate_inputs(tmp_path)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    scope = evidence["budget_summary"]["compilation_repair_candidate"]
    unrelated = next(row["evidence_id"] for row in evidence["items"]
        if row["source_type"] == "target_source" and row["evidence_id"] not in scope["evidence_ids"])
    client = _Client(override=lambda result: {
        **_candidate_response(result), "selected_evidence_ids": [unrelated]})
    _, _, plan = _run(inputs, client, candidate=True)
    assert len(client.calls) == 1
    assert plan["status"] == "blocked"
    assert plan["data"]["proposed_changes"] == []


@pytest.mark.parametrize("evaluator_enabled", [False, True])
def test_compiler_candidate_requires_real_evaluator_configuration(tmp_path, evaluator_enabled):
    from tests.unit.test_compilation_candidate_continuation import _candidate_inputs

    inputs, _ = _candidate_inputs(tmp_path, evaluator=evaluator_enabled)
    client = _Client(override=_candidate_response)
    # A real evaluator without the selector switch, or the switch without an
    # evaluator, cannot admit a candidate based only on insufficient Findings.
    _, _, plan = _run(inputs, client, candidate=not evaluator_enabled)
    assert client.calls == []
    assert plan["status"] == "blocked"


@pytest.mark.parametrize("damage", ["source", "scope", "worker_configuration"])
def test_compiler_candidate_rechecks_bound_inputs_after_model(tmp_path, damage):
    from pathlib import Path

    from tests.unit.test_compilation_candidate_continuation import _candidate_inputs

    inputs, configuration = _candidate_inputs(tmp_path)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]

    def mutate():
        if damage == "source":
            inputs.behavior.write_text(inputs.behavior.read_text() + "// changed\n")
        elif damage == "scope":
            evidence["budget_summary"]["compilation_repair_candidate"]["evidence_ids"] = ["unbound"]
        else:
            path = Path(configuration.reference["path"])
            path.write_bytes(path.read_bytes() + b" ")

    client = _Client(override=_candidate_response, mutate=mutate)
    selector = ModelPlanningTargetSelector(client=client, workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "selection.json", behavior_evaluator_configured=True)
    with pytest.raises(PlanningTargetSelectionError):
        selector(inputs.feature_payload, evidence)
    assert len(client.calls) == 1
    receipt = json.loads(selector.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"


def test_compiler_candidate_correction_rechecks_evaluator_before_another_model_call(tmp_path):
    from pathlib import Path

    from tests.unit.test_compilation_candidate_continuation import _candidate_inputs

    inputs, configuration = _candidate_inputs(tmp_path)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    client = _Client(override=_candidate_response)
    selector = ModelPlanningTargetSelector(client=client, workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "selection.json", behavior_evaluator_configured=True)
    selector(inputs.feature_payload, evidence)
    child = selector.for_target_correction({"code": "EDIT_TARGET_NORMALIZATION_UNSAFE"})
    path = Path(configuration.reference["path"])
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(PlanningTargetSelectionError, match="COMPILATION_CANDIDATE_WORKER_INVALID"):
        child(inputs.feature_payload, evidence)
    assert len(client.calls) == 1


def test_candidate_correction_uses_child_questions_and_preserves_parent_receipt(tmp_path):
    inputs, _, ids = _overlap_inputs(tmp_path)
    child_questions = ["Check the current caller behavior in candidate verification."]
    first = _candidate_response(_selection(ids))
    second = {**_candidate_response(_selection(ids[:1])), "verification_questions": child_questions}
    client = _CorrectionClient([first, second])
    selector, _, plan = _run(inputs, client, candidate=True)
    assert plan["status"] == "success"
    assert len(client.calls) == 2
    assert selector.selection_result == first
    assert selector.verification_questions == tuple(child_questions)
    assert plan["data"]["verification_questions"] == child_questions
    assert selector._target_correction.behavior_evaluator_configured is True
    assert len(plan["trace_refs"]) == 2
    assert "needs_evidence" not in client.calls[1].messages[1].content["correction_context"]["instruction"]
    parent = json.loads(selector.receipt_path.read_text(encoding="utf-8"))
    assert json.loads(parent["response_content"]) == first


@pytest.mark.parametrize("new_questions", [False, True])
def test_candidate_upstream_child_retention_never_swallows_new_questions(tmp_path, new_questions):
    inputs = _inputs(tmp_path)
    client = _Client(override=lambda payload: {**_candidate_response(payload),
        "decision": "retain_current_plan", "selected_evidence_ids": [],
        "verification_questions": list(QUESTIONS) if new_questions else []})
    parent = ModelPlanningTargetSelector(client=client, workspace_root=inputs.workspace,
        receipt_path=inputs.store.root / "selection.json", behavior_evaluator_configured=True)
    child = parent.for_product_assessment("a" * 64)
    child.upstream_assessment = {"retain_current_plan_allowed": True}
    assert child.behavior_evaluator_configured is True
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    if new_questions:
        with pytest.raises(PlanningTargetSelectionError) as raised:
            child(inputs.feature_payload, evidence)
        assert raised.value.code == "PLANNING_MODEL_RETAIN_PLAN_NOT_ADMITTED"
    else:
        assert child(inputs.feature_payload, evidence) == ()
        assert child.selection_result["decision"] == "retain_current_plan"
    assert client.calls[0].response_format.schema["properties"]["decision"]["enum"] == [
        "select_targets", "retain_current_plan"]


@pytest.mark.parametrize("passes", [True, False])
def test_candidate_questions_reach_actual_patch_worker_and_evaluator_failure_still_blocks(tmp_path, passes):
    inputs = _inputs(tmp_path)
    original = _source_hashes(inputs.workspace)
    _, state, plan = _run(inputs, _Client(override=_candidate_response), candidate=True)
    assert plan["status"] == "success"
    client = _BehaviorClient(fixes=passes)
    oracle = _BehaviorOracle(tmp_path / "behavior")
    adapter = _adapter(inputs, client, oracle)
    request = PatchContinuationInputV1(contract_version=PATCH_CONTINUATION_CONTRACT_VERSION,
        artifact_store_root=str(inputs.store.root), workspace_root=str(inputs.workspace),
        feature_spec_ref=inputs.feature_ref, evidence_pack_ref=inputs.evidence_ref,
        implementation_plan_ref=ArtifactRef(feature_id=inputs.feature_ref.feature_id,
            artifact_type="implementation_plan", path=state["artifacts"]["implementation_plan"],
            artifact_hash=state["artifact_hashes"]["implementation_plan"]))
    result = run_fresh_patch_continuation(request, patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
        max_model_calls=adapter.patch_agent_loop_policy.max_total_model_turns)
    assert client.requests
    actual_worker_payload = json.loads(client.requests[0].messages[1].content)
    assert actual_worker_payload["verification_questions"] == QUESTIONS
    assert actual_worker_payload["acceptance_criteria"] == inputs.feature_payload["acceptance_criteria"]
    assert not set(QUESTIONS).intersection(actual_worker_payload["constraints"])
    assert all(context["implementation_plan"]["verification_questions"] == QUESTIONS for context in oracle.contexts)
    assert len(oracle.contexts) == 2
    assert _source_hashes(inputs.workspace) == original
    if passes:
        assert result.status == "terminal"
        assert result.patch_proposal_artifact is not None
        assert oracle.reports[-1]["status"] == "passed"
    else:
        assert result.status != "terminal"
        assert result.patch_proposal_artifact is None
        assert result.proposal_review_artifact is None
        assert result.dry_run_artifact is None
        assert all(report["status"] == "behavior_failed" for report in oracle.reports)


def test_absent_questions_preserve_legacy_plan_payload(tmp_path):
    inputs = _inputs(tmp_path)
    _, _, plan = _run(inputs, _Client())
    assert "verification_questions" not in plan["data"]
    assert plan["data"]["test_plan"] == inputs.feature_payload["runtime_verification_plan"]


def test_normal_entry_and_same_run_upstream_enable_only_configured_candidate_contract(tmp_path, monkeypatch, capsys):
    from src.mod_agent import local_entry
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from tests.unit.test_mod_agent_cli import _public_patch_upstream_scenario
    from tests.unit.test_patch_worker_configuration import _write_configuration

    factory_source = '''
import hashlib
import json
from pathlib import Path
def make_evaluator(*, output):
    def evaluate(context):
        passed = all("PRODUCT_TEST_CHANGE" in row["patch"]["after_text"]
                     for row in context["generated_changes"])
        report = {
            "status": "passed" if passed else "behavior_failed",
            "failure_owner": None if passed else "patch_implementation",
            "owner_binding_ids": [] if passed else [row["edit_binding_id"] for row in context["generated_changes"]],
            "observations": [{"case_id": "synthetic_marker", "passed": passed}],
            "generated_changes_sha256": context["generated_changes_sha256"],
            "source_snapshot_sha256": context["source_snapshot_sha256"],
            "execution_origin": "executed",
        }
        raw = json.dumps(report).encode()
        path = Path(output)
        path.write_bytes(raw)
        return {**report, "report_ref": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}}
    return evaluate
'''
    path, _, value = _write_configuration(tmp_path, factory_source,
        kwargs={"output": str(tmp_path / "observed.json")})
    value["policy"] = {"require_source_inspection": False, "finalize_after_first_observation": False,
        "enable_supporting_reads": True, "allow_upstream_request": True}
    path.write_text(json.dumps(value))
    configuration = PatchWorkerConfiguration.load(path)
    actual_run = local_entry.run_local_request

    def configured_run(**kwargs):
        return actual_run(**{**kwargs, "patch_worker_configuration": configuration})

    monkeypatch.setattr(local_entry, "run_local_request", configured_run)
    configured_flags = []
    ordinary_from_settings = ModelPlanningTargetSelector.from_settings.__func__

    def from_settings(cls, **kwargs):
        configured_flags.append(kwargs.get("behavior_evaluator_configured", False))
        selector = ordinary_from_settings(cls, **kwargs)
        ordinary_complete = selector.client.complete_turn

        def controlled_response(request):
            response = ordinary_complete(request)
            # Adapt the existing fake transport fixture to this new response contract.
            return replace(response, content=json.dumps(_candidate_response(json.loads(response.content))))

        selector.client.complete_turn = controlled_response
        return selector

    monkeypatch.setattr(ModelPlanningTargetSelector, "from_settings", classmethod(from_settings))
    observed = _public_patch_upstream_scenario(tmp_path, monkeypatch, capsys, owner="planning")
    assert configured_flags == [True, True]
    assert observed.result.get("status") == "awaiting_approval", observed.result
    assert len(observed.planning) == len(observed.patch_contexts) == 2
    assert all(row["candidate_scope_contract"]["behavior_evaluator"]["configured"] for row in observed.planning)
    assert all(row["verification_questions"] == QUESTIONS for row in observed.patch_contexts)
    assert observed.result["actual_write_performed"] is False
    assert observed.result["approval_recorded"] is False
    assert observed.result["builds_performed"] == observed.result["runtime_executions_performed"] == 0
    assert json.loads((tmp_path / "observed.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("configured", [False, True])
def test_restored_upstream_assembly_inherits_actual_worker_configuration(tmp_path, configured):
    from src.mod_agent.patch_upstream_configuration import PatchUpstreamConfiguration
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
    from tests.unit.test_patch_worker_configuration import _write_configuration

    inputs = _inputs(tmp_path)
    path = tmp_path / "upstream.json"
    path.write_text(json.dumps({"schema_version": "patch_upstream_configuration_v1", "evidence": None,
        "planning_profile": {"profile_version": "local_ollama_initial_patch_profile_v1",
            "base_url": "http://127.0.0.1:11434", "model": "qwen3:4b-instruct",
            "context_window": 49152, "output_budget": 4096, "timeout_seconds": 30.0}}))
    configuration = PatchUpstreamConfiguration.load(path)
    if configured:
        worker_path, _, value = _write_configuration(tmp_path,
            "def make_evaluator():\n    raise AssertionError('assembly must not construct evaluator')\n")
        value["policy"] = {"allow_upstream_request": True}
        worker_path.write_text(json.dumps(value))
        worker = PatchWorkerConfiguration.load(worker_path)
        configuration = configuration.with_worker(profile=configuration.planning_profile, worker_configuration=worker)
    providers = configuration.assemble(inputs.request)
    selector = providers["planning_target_selection_factory"]
    assert selector.behavior_evaluator_configured is configured
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    prepared = selector.prepare(inputs.feature_payload, evidence)
    assert ("candidate_scope_contract" in prepared.messages[1].content) is configured
    assert ("needs_evidence" in prepared.response_format.schema["properties"]["decision"]["enum"]) is not configured
    assert not selector.receipt_path.exists()
