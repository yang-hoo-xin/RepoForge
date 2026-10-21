"""Focused Product integration checks with explicit scripted owners; no live models/Build."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.product_execution import (
    ProductExecutionContractError,
    ProductExecutionEffectEventV1,
    ProductExecutionInputBindingV1,
    _validate_store_trace,
    _validated_task_agent_sequences,
)
from src.mod_agent.product_executor import _ProductStoreRecorder
from src.mod_agent.task_agent_execution import TaskAgentExecution, TaskAgentPatchOutcome, ValidatedTaskAgentHandoff
from src.mod_agent.task_agent_tools import TaskAgentContext
from src.mod_agent.workflow import graph
from src.mod_agent.workflow.state import create_initial_state
from src.observability.evaluation.unified import save
from tests.unit.test_product_execution_projection_v2 import FEATURE_ID, _artifact_ref, _committed_result


def setup(tmp_path):
    recorder = _ProductStoreRecorder(ArtifactStore(tmp_path / "store"))
    for role in ("request_planning_report", "mod_planning_brief", "feature_spec", "evidence_pack", "implementation_plan"):
        recorder._record(_committed_result(_artifact_ref(role), commit_started=True), payload={})
    context = TaskAgentContext(**{role + "_ref": _artifact_ref(role)
        for role in ("feature_spec", "evidence_pack", "implementation_plan")})
    state = create_initial_state(run_id="task-run", request_id="task-request", user_request="fix behavior",
        target_domain="gtfo", target_mod="Example")
    state.update(feature_id=FEATURE_ID, current_stage="implementation_plan",
        artifacts={role: getattr(context, role + "_ref").path for role in ("feature_spec", "evidence_pack", "implementation_plan")},
        artifact_hashes={role: getattr(context, role + "_ref").artifact_hash for role in ("feature_spec", "evidence_pack", "implementation_plan")})
    owner = object.__new__(TaskAgentExecution)
    owner._receipt = None
    return recorder, context, state, owner


def outcome(context, *, status="blocked", execution_ref=None, claim_ref=None):
    return TaskAgentPatchOutcome(status, {"code": "TEST_BLOCKED"}, context, context, execution_ref,
        None, claim_ref, None, None, None, {}, {}, False, False)


def recorded_scope(tmp_path, *, partial_evidence=False):
    recorder, context, state, owner = setup(tmp_path)
    claim = save(tmp_path / "claim.json", {"run_id": state["run_id"], "request_id": state["request_id"],
        "initial_context": context.to_dict()})
    with recorder.task_agent_scope(owner, state):
        recorder._record(_committed_result(_artifact_ref("repair_plan"), commit_started=True), payload={"retry_count": 99})
        if partial_evidence:
            new_ref = replace(_artifact_ref("evidence_pack"), artifact_hash="b" * 64, path=str(tmp_path / "new-evidence.json"))
            recorder._record(_committed_result(new_ref, commit_started=True), payload={})
        owner._receipt = outcome(context, claim_ref=claim)
        evidence = {**{key: value for key, value in owner._receipt.to_dict().items() if key != "execution_ref"},
            "run_id": state["run_id"], "request_id": state["request_id"],
            "product_store_attempts": recorder.task_agent_store_attempts(owner)}
        owner._receipt = replace(owner._receipt, execution_ref=save(tmp_path / "execution.json", evidence))
    facts = SimpleNamespace(request=SimpleNamespace(run_id=state["run_id"], request_id=state["request_id"]),
        feature_id=FEATURE_ID, store_attempts=tuple(recorder.attempts), outputs=tuple(recorder.outputs),
        task_agent_executions=tuple(recorder.task_agent_executions), failure_stage=None,
        effect_events=(ProductExecutionEffectEventV1(contract_version="1", sequence=0, phase="proposal",
            repair_attempt=0, effect_role="patch_generation", event_status="started", failure_class=None, failure_message=None),))
    return recorder, facts


def test_actual_task_store_attempts_keep_outer_retry_and_declared_phase(tmp_path):
    recorder, facts = recorded_scope(tmp_path)
    assert recorder.current_repair_attempt == 0
    last = recorder.attempts[-1]
    assert (last.node_name, last.phase, last.repair_attempt) == ("generate_patch_proposal", "proposal", 0)
    admitted = _validated_task_agent_sequences(facts)
    assert admitted == {last.sequence}
    _validate_store_trace(tuple(recorder.attempts), tuple(recorder.outputs), task_agent_sequences=admitted)
    with pytest.raises(ProductExecutionContractError, match="node/role"):
        _validate_store_trace(tuple(recorder.attempts), tuple(recorder.outputs))


def test_partial_internal_evidence_write_is_preserved_without_final_plan(tmp_path):
    recorder, facts = recorded_scope(tmp_path, partial_evidence=True)
    assert _validated_task_agent_sequences(facts) == {5, 6}
    assert recorder.outputs[-1].artifact_role == "evidence_pack"
    assert facts.task_agent_executions[0].outcome["status"] == "blocked"


@pytest.mark.parametrize("change", ["sequence", "owner", "status", "outcome", "initial", "phase"])
def test_task_store_sequence_exemption_requires_exact_execution_binding(tmp_path, change):
    recorder, facts = recorded_scope(tmp_path)
    execution = facts.task_agent_executions[0]
    if change == "sequence":
        execution = replace(execution, store_sequences=(1,))
    elif change == "owner":
        execution = replace(execution, run_id="another-run")
    elif change == "status":
        execution = replace(execution, outcome={**execution.outcome, "whole_task_delivered": True})
    elif change == "outcome":
        execution = replace(execution, outcome={**execution.outcome, "termination": {"code": "changed"}})
    elif change == "initial":
        execution = replace(execution, initial_context={})
    else:
        facts.store_attempts = (*facts.store_attempts[:-1], replace(facts.store_attempts[-1], phase="repair"))
    facts.task_agent_executions = (execution,)
    with pytest.raises(ProductExecutionContractError):
        _validated_task_agent_sequences(facts)


def test_task_scope_cannot_write_formal_delivery_artifact(tmp_path):
    recorder, context, state, owner = setup(tmp_path)
    with recorder.task_agent_scope(owner, state):
        with pytest.raises(ProductExecutionContractError, match="delivery artifacts"):
            recorder.write(feature_id=FEATURE_ID, artifact_type="build_result", payload={})
    assert len(recorder.attempts) == 5


def test_unretained_failure_keeps_attempt_but_cannot_be_success(tmp_path):
    recorder, facts = recorded_scope(tmp_path)
    execution = facts.task_agent_executions[0]
    failed = {**execution.outcome, "execution_ref": None, "status": "failed"}
    facts.task_agent_executions = (replace(execution, outcome=failed),)
    facts.failure_stage = "generate_patch_proposal_execution"
    assert _validated_task_agent_sequences(facts) == {5}
    facts.task_agent_executions = (replace(execution, outcome={**failed, "status": "candidate_ready"}),)
    with pytest.raises(ProductExecutionContractError):
        _validated_task_agent_sequences(facts)


def test_failed_task_technical_records_survive_strict_result_roundtrip(tmp_path):
    from src.mod_agent.product_execution import (
        ProductBuildCompletionV1, ProductExecutionBlockerV1, ProductExecutionFactsV1,
        ProductExecutionResultV1, ProductExecutionValidationDetailV1, ProductExecutionValidationFactsV1,
        _ProductExecutionResultIdentityV1, project_product_execution_result,
    )
    from tests.unit.test_product_execution_projection_v2 import _readback_envelope

    recorder, raw = recorded_scope(tmp_path)
    raw.request = _ProductExecutionResultIdentityV1(run_id="task-run", request_id="task-request", requested_completion_scope="compile_only")
    raw.terminal_node = None
    raw.failure_stage = "generate_patch_proposal_execution"
    raw.variant_result = ProductBuildCompletionV1(contract_version="1", build_result_ref=None,
        workflow_checkpoint_ref=None, build_status="not_reached", claims_real_build_success=None)
    raw.blockers = (ProductExecutionBlockerV1(contract_version="1", owner="platform",
        stage=raw.failure_stage, code="SCRIPTED_EXECUTION_STOP", message="Scripted bounded execution stopped.",
        source_node="generate_patch_proposal"),)
    raw.effect_events = (*raw.effect_events, replace(raw.effect_events[0], sequence=1, event_status="failed",
        failure_class="ScriptedFailure", failure_message="Stopped"))
    raw.validation_facts = ProductExecutionValidationFactsV1(contract_version="1", output_readback_status="passed",
        canonical_source_status="not_applicable", compiled_artifact_status="not_applicable",
        source_change_status="not_applicable", completion_classification_status="not_applicable",
        failure_class=None, failure_message=None,
        details=tuple(ProductExecutionValidationDetailV1(contract_version="1", detail_kind=kind,
            payload=[_readback_envelope(row) for row in recorder.outputs] if kind == "output_readback" else None)
            for kind in ("output_readback", "canonical_source", "compiled_artifact", "source_change", "completion_classification")))
    result = project_product_execution_result(ProductExecutionFactsV1(**vars(raw)), validate_request=False)
    assert result.platform_status == "partial"
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result
    assert result.task_agent_executions[0].store_sequences == (5,)


def test_typed_blocked_outcome_creates_no_proposal(tmp_path, monkeypatch):
    _, context, state, owner = setup(tmp_path)
    monkeypatch.setattr(graph, "_read_artifact_data", lambda *args: {})
    result = graph._run_generate_patch_proposal(state, SimpleNamespace(),
        patch_change_factory=lambda *args: outcome(context), task_agent_execution=owner)
    assert result["blockers"][-1]["code"] == "TEST_BLOCKED"
    assert "patch_proposal" not in result["artifacts"]
    assert state["blockers"] == []


def test_validated_handoff_builds_proposal_against_final_context(tmp_path, monkeypatch):
    _, context, state, owner = setup(tmp_path)
    final = replace(context, implementation_plan_ref=replace(context.implementation_plan_ref,
        artifact_hash="b" * 64, path=str(tmp_path / "new-plan.json")))
    change = {"change_id": "actual-candidate-change", "generation": {"original": True}}
    current_outcome = replace(outcome(context, status="candidate_ready"), final_context=final,
        execution_ref={"path": "execution", "sha256": "a" * 64})
    handoff = ValidatedTaskAgentHandoff(final, (change,), {}, {}, current_outcome.execution_ref)
    monkeypatch.setattr(owner, "validate_handoff", lambda supplied, expected_initial_context:
        handoff if supplied == current_outcome and expected_initial_context == context else None)
    reads, built, saved = [], [], []
    def read(state, store, role):
        reads.append((role, state["artifact_hashes"][role]))
        return {"proposed_changes": []} if role == "implementation_plan" else {}
    monkeypatch.setattr(graph, "_read_artifact_data", read)
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Proposal:
        feature_id: str = FEATURE_ID
        context_pack: dict = None
        blocked: bool = False
        def to_artifact_payload(self):
            return {"context_pack": self.context_pack}
    class Builder:
        def build(self, implementation, **kwargs):
            built.append((deepcopy(implementation), kwargs))
            return Proposal(context_pack={})
    monkeypatch.setattr(graph, "PatchProposalBuilder", Builder)
    def write(**kwargs):
        saved.append(kwargs)
        return _artifact_ref("patch_proposal")
    result = graph._run_generate_patch_proposal(state, SimpleNamespace(write=write),
        patch_change_factory=lambda *args: current_outcome, task_agent_execution=owner)
    assert built[0][0]["proposed_changes"] == [change]
    assert built[0][1]["artifact_hashes"]["implementation_plan"] == "b" * 64
    assert ("implementation_plan", "b" * 64) in reads
    assert final.implementation_plan_ref in saved[0]["upstream_refs"]
    assert result["artifact_hashes"]["implementation_plan"] == "b" * 64
    assert saved[0]["payload"]["context_pack"]["task_agent_lineage"]["approval_inherited"] is False


@pytest.mark.parametrize("mismatch", [None, "workspace", "store", "model", "digest", "endpoint", "read_root", "edit_path"])
def test_task_owner_admission_is_bound_before_any_effect(tmp_path, mismatch):
    from tests.unit.test_product_execution import _request, _blocked_evidence_adapters
    from src.mod_agent.model_admission import local_model_admission_record

    request, store = _request(tmp_path)
    identity = {"name": "qualified-model", "model": "qualified-model", "digest": "1" * 64,
        "base_url": "http://127.0.0.1:11434", "details": {"context_length": 65536}, "capabilities": ["tools"]}
    binding = ProductExecutionInputBindingV1.create("model_admission", local_model_admission_record(
        model_identity=identity, provider_adapter="scripted_test"))
    request = replace(request, input_bindings=tuple(binding if row.binding_kind == "model_admission" else row
        for row in request.input_bindings))
    owner = object.__new__(TaskAgentExecution)
    owner.workspace_root, owner.store = Path(request.isolated_workspace_root), store
    owner.profile = SimpleNamespace(base_url=identity["base_url"], model=identity["name"], context_window=65536)
    owner.expected_model_identity = deepcopy(identity)
    owner.allowed_read_roots = (Path(request.isolated_target_root),)
    owner._allowed_edit_paths = ("generated_mods/Example/Plugin.cs",)
    calls = []
    owner.identity_reader = lambda *_: calls.append("identity")
    owner.client_factory = lambda *_: calls.append("model")
    owner.evaluator = lambda *_: calls.append("Build")
    if mismatch == "workspace":
        owner.workspace_root = Path(request.source_root)
    elif mismatch == "store":
        owner.store = ArtifactStore(tmp_path / "another-store")
    elif mismatch == "model":
        owner.profile.model = "other-model"
    elif mismatch == "digest":
        owner.expected_model_identity["digest"] = "2" * 64
    elif mismatch == "endpoint":
        owner.profile.base_url = "http://127.0.0.1:11435"
    elif mismatch == "read_root":
        owner.allowed_read_roots = (Path(request.source_root),)
    elif mismatch == "edit_path":
        owner._allowed_edit_paths = ("../source/Plugin.cs",)
    adapters = replace(_blocked_evidence_adapters(), patch_change_factory=owner, task_agent_execution=owner)
    if mismatch is None:
        adapters.validate_for(request)
    else:
        with pytest.raises(ProductExecutionContractError, match="Task execution input admission") as captured:
            adapters.validate_for(request)
        assert captured.value.side_effects_may_have_started is False
    assert calls == []
