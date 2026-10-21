from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.mod_agent import fresh_planning_continuation as continuation
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.workflow.graph import _run_create_implementation_plan
from src.observability.patch_execution_index import project_patch_executions
from tests.unit.test_fresh_planning_continuation import (
    _direct_graph_state,
    _inputs,
    _patch_request,
)


@pytest.fixture
def graph_inputs(tmp_path):
    inputs = _inputs(tmp_path)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    evidence["budget_summary"].pop("fresh_coverage_model_execution")
    evidence_ref = inputs.store.write(
        feature_id=inputs.feature_ref.feature_id,
        artifact_type="evidence_pack", payload=evidence, status="success",
        upstream_refs=[inputs.feature_ref],
    )
    inputs = replace(inputs, evidence_ref=evidence_ref)
    state = _run_create_implementation_plan(
        _direct_graph_state(inputs), inputs.store,
        formal_requirement_coverage_required=True,
        planning_target_selection_factory=lambda _feature, _evidence: (
            inputs.request.planning_target_evidence_ids
        ),
    )
    assert not state["blockers"]
    refs = {kind + "_ref": ArtifactRef(
        feature_id=state["feature_id"], artifact_type=kind,
        path=state["artifacts"][kind], artifact_hash=state["artifact_hashes"][kind],
    ) for kind in ("feature_spec", "evidence_pack", "implementation_plan")}
    return inputs, {
        "artifact_root": inputs.store.root, "workspace_root": inputs.workspace,
        "run_id": "product_worker_001", "request_id": "product_request_001",
        "parent_run_id": "product_run_001", "max_model_calls": 2, **refs,
    }


def test_product_graph_outputs_create_original_queryable_journal_without_fresh_or_repair(
    graph_inputs, monkeypatch,
):
    inputs, arguments = graph_inputs

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A Product journal must not fabricate Fresh admission.")

    monkeypatch.setattr(continuation, "PatchContinuationInputV1", forbidden)
    monkeypatch.setattr(continuation, "preflight_patch_continuation", forbidden)
    journal = continuation.PatchContinuationAttemptJournal.for_product_execution(**arguments)
    document = json.loads(journal.path.read_bytes())
    assert document["receipt_version"] == continuation.PATCH_CONTINUATION_RECEIPT_VERSION
    assert document["continuation_selection"] == "product_graph_current_implementation_plan"
    assert document["caller"] == {
        "workspace_root": str(inputs.workspace.resolve()), "parent_run_id": "product_run_001",
    }
    assert "repair_context" not in document
    assert document["model_call_count"] == 0
    assert document["source_validation"]["status"] == "passed"
    snapshot = continuation._csharp_source_snapshot(inputs.workspace)
    assert document["generated_against_snapshot_sha256"] == (
        continuation._source_tree_snapshot_sha256(snapshot)
    )
    journal.record({"phase": "patch_generation_started", "model_call_count": 1})
    journal.record_observation({"id": "observation-001", "name": "patch.execution"})
    journal.record_trace_summary({"mode": "local_only"})
    projection = journal.projection()
    assert projection["status"] == "patch_generation_started"
    assert projection["model_call_count"] == 1
    assert projection["event_count"] == 1
    assert projection["sha256"] == hashlib.sha256(journal.path.read_bytes()).hexdigest()
    before = journal.path.read_bytes()
    queried = project_patch_executions(inputs.store.root, run_id=arguments["run_id"])
    assert queried["match_count"] == 1
    assert journal.path.read_bytes() == before


def test_meter_only_product_journal_records_null_quota_and_real_counts(graph_inputs):
    _, arguments = graph_inputs
    journal = continuation.PatchContinuationAttemptJournal.for_product_execution(
        **dict(arguments, max_model_calls=None, model_call_budget_policy="meter_only"))
    journal.record({"phase": "patch_generation_started", "model_call_count": 31})
    document = json.loads(journal.path.read_bytes())
    assert document["max_model_calls"] is None
    assert document["model_call_budget_policy"] == "meter_only"
    assert document["model_call_count"] == 31 and document["source_validation"]["status"] == "passed"


def test_meter_only_request_worker_assembles_serializable_null_quota(graph_inputs):
    from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
    from src.mod_agent.request_patch_execution import RequestPatchExecution
    from tests.unit.test_fresh_planning_continuation import _local_patch_adapter

    inputs, arguments = graph_inputs
    adapter, _ = _local_patch_adapter(inputs)
    refs = [arguments[kind + "_ref"] for kind in ("feature_spec", "evidence_pack", "implementation_plan")]
    state = {"run_id": "metered-run", "request_id": "metered-request", "feature_id": refs[0].feature_id,
        "retry_count": 0, "current_stage": "implementation_plan",
        "artifacts": {ref.artifact_type: ref.path for ref in refs},
        "artifact_hashes": {ref.artifact_type: ref.artifact_hash for ref in refs}}
    observed = []

    class Worker:
        _recovery_observation_bytes = 0

        def __init__(self, **kwargs):
            observed.append(kwargs["patch_agent_loop_policy"])

        def bind_attempt_recorder(self, _recorder):
            pass

        def bind_upstream_effects(self, _effects):
            pass

        def patch_attempt_receipt(self):
            return {}

        def __call__(self, *_args):
            return []

    budget = {"policy": "meter_only", "maximum": None, "remaining": None, "consumed": 31,
        "by_stage": {"initial_patch": {"maximum": None, "remaining": None, "consumed": 31}},
        "count_basis": "reserved_provider_invocations_not_transport_proof"}
    execution = RequestPatchExecution(artifact_store=inputs.store, workspace_root=inputs.workspace,
        profile=adapter.profile, identity_reader=adapter.identity_reader,
        policy=PatchAgentLoopPolicy(model_call_budget_policy="meter_only"),
        reserve_model_call=lambda _role: pytest.fail("No model should run"), budget_snapshot=lambda: deepcopy(budget),
        adapter_factory=Worker)
    assert execution(state, inputs.store.read(refs[2])["data"], inputs.store.read(refs[1])["data"]) == []
    assert observed[0].max_total_model_turns is observed[0].max_tool_rounds is observed[0].max_observation_bytes is None
    document = json.loads(execution.journals[0].path.read_bytes())
    assert document["max_model_calls"] is None and document["model_call_budget_policy"] == "meter_only"
    json.dumps(document, allow_nan=False)


@pytest.mark.parametrize("change", ["hash", "foreign_feature", "source"])
@pytest.mark.parametrize("metered", [False, True])
def test_product_journal_rejects_changed_inputs_before_receipt_write(graph_inputs, change, metered):
    inputs, arguments = graph_inputs
    if metered:
        arguments.update(max_model_calls=None, model_call_budget_policy="meter_only")
    if change == "hash":
        arguments["implementation_plan_ref"] = replace(
            arguments["implementation_plan_ref"], artifact_hash="0" * 64,
        )
        code = "PATCH_CONTINUATION_EXACT_REVISION_REQUIRED"
    elif change == "foreign_feature":
        arguments["evidence_pack_ref"] = replace(
            arguments["evidence_pack_ref"], feature_id="another_feature",
        )
        code = "PATCH_CONTINUATION_FEATURE_MISMATCH"
    else:
        inputs.behavior.write_text(inputs.behavior.read_text(encoding="utf-8") + "// changed\n", encoding="utf-8")
        code = "PATCH_CONTINUATION_SOURCE_HASH_MISMATCH"
    with pytest.raises(continuation.PatchContinuationError, match=code):
        continuation.PatchContinuationAttemptJournal.for_product_execution(**arguments)
    assert not (inputs.store.root / "cr").exists()


def test_product_journal_detects_source_race_during_validation(graph_inputs, monkeypatch):
    inputs, arguments = graph_inputs
    original = continuation._validate_source_items

    def mutate_after_read(evidence, workspace):
        result = original(evidence, workspace)
        inputs.behavior.write_text(inputs.behavior.read_text(encoding="utf-8") + "// race\n", encoding="utf-8")
        return result

    monkeypatch.setattr(continuation, "_validate_source_items", mutate_after_read)
    with pytest.raises(continuation.PatchContinuationError, match="PATCH_CONTINUATION_SOURCE_HASH_MISMATCH"):
        continuation.PatchContinuationAttemptJournal.for_product_execution(**arguments)
    assert not (inputs.store.root / "cr").exists()


def test_product_journal_binds_optional_worker_configuration_and_rejects_drift(graph_inputs, tmp_path):
    inputs, arguments = graph_inputs
    config = tmp_path / "worker.json"
    config.write_text('{"worker":"test"}\n', encoding="utf-8")
    arguments.update(
        worker_configuration_ref={"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        worker_profile_sha256="1" * 64,
    )
    journal = continuation.PatchContinuationAttemptJournal.for_product_execution(**arguments)
    document = json.loads(journal.path.read_bytes())
    assert document["worker_configuration_ref"] == arguments["worker_configuration_ref"]
    assert document["worker_profile_sha256"] == "1" * 64
    config.write_text('{}\n', encoding="utf-8")
    arguments["run_id"] = "product_worker_002"
    with pytest.raises(continuation.PatchContinuationError, match="PATCH_WORKER_CONFIGURATION_IDENTITY_INVALID"):
        continuation.PatchContinuationAttemptJournal.for_product_execution(**arguments)
    assert len(list((inputs.store.root / "cr").glob("*.json"))) == 1


def test_existing_fresh_constructor_keeps_original_identity_and_selection(tmp_path):
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    admitted = continuation.preflight_patch_continuation(request)
    journal = continuation.PatchContinuationAttemptJournal(
        artifact_root=inputs.store.root, run_id="fresh_worker_001", request_id="fresh_request_001",
        request=request, admitted=admitted, max_model_calls=2,
    )
    document = json.loads(journal.path.read_bytes())
    assert document["continuation_selection"] == "explicit_exact_implementation_plan_ref"
    assert document["attempt_identity_sha256"] == continuation._patch_attempt_identity(request, 2)
    assert "caller" not in document and "parent_run_id" not in document
    assert "repair_context" not in document


def _rebind_store(arguments, root):
    original = ArtifactStore(arguments["artifact_root"])
    store = ArtifactStore(root)
    selected = dict(arguments, artifact_root=store.root)
    upstream = []
    for kind in ("feature_spec", "evidence_pack", "implementation_plan"):
        envelope = original.semantic_read_immutable(arguments[kind + "_ref"])
        ref = store.write(
            feature_id=arguments["feature_spec_ref"].feature_id,
            artifact_type=kind, payload=envelope["data"], status="success",
            upstream_refs=list(upstream),
        )
        selected[kind + "_ref"] = ref
        upstream.append(ref)
    return selected


def test_product_run_root_layout_keeps_store_inside_workspace(graph_inputs):
    inputs, arguments = graph_inputs
    selected = _rebind_store(arguments, inputs.workspace / "a")
    journal = continuation.PatchContinuationAttemptJournal.for_product_execution(**selected)
    document = json.loads(journal.path.read_bytes())
    assert journal.path.parent == inputs.workspace / "a" / "cr"
    assert document["caller"]["workspace_root"] == str(inputs.workspace.resolve())
    assert document["source_validation"]["status"] == "passed"
    assert document["generated_against_snapshot_sha256"] == continuation._source_tree_snapshot_sha256(
        continuation._csharp_source_snapshot(inputs.workspace)
    )


def test_product_run_root_layout_rejects_evidence_source_inside_store(graph_inputs):
    inputs, arguments = graph_inputs
    # The unchanged Evidence paths resolve into this Store root, so this is a
    # real containment conflict rather than a malformed source-ref fixture.
    selected = _rebind_store(arguments, inputs.workspace / "generated_mods")
    with pytest.raises(continuation.PatchContinuationError, match="PATCH_PRODUCT_JOURNAL_SOURCE_IN_STORE"):
        continuation.PatchContinuationAttemptJournal.for_product_execution(**selected)
    assert not (selected["artifact_root"] / "cr").exists()


@pytest.mark.parametrize("child", ["", "workspace"])
def test_product_journal_rejects_workspace_inside_store(graph_inputs, child):
    inputs, arguments = graph_inputs
    workspace = inputs.store.root / child
    workspace.mkdir(exist_ok=True)
    with pytest.raises(continuation.PatchContinuationError, match="PATCH_CONTINUATION_ROOTS_OVERLAP"):
        continuation.PatchContinuationAttemptJournal.for_product_execution(
            **dict(arguments, workspace_root=workspace),
        )
    assert not (inputs.store.root / "cr").exists()


@pytest.mark.parametrize("provider,owner,code", [
    ("patch_upstream_evidence_pack_factory", "evidence", "PATCH_UPSTREAM_EVIDENCE_PROVIDER_REQUIRED"),
    ("patch_upstream_planning_target_selection_factory", "planning", "PATCH_UPSTREAM_PLANNING_PROVIDER_REQUIRED"),
])
def test_product_upstream_missing_provider_stops_current_public_request(
    tmp_path, monkeypatch, capsys, provider, owner, code,
):
    """Deterministic transport; the real Product Graph receives a missing provider."""
    from src.mod_agent import product_executor
    from tests.unit.test_mod_agent_cli import _public_patch_upstream_scenario

    ordinary_build = product_executor.build_graph

    def build(*args, **kwargs):
        kwargs[provider] = None
        return ordinary_build(*args, **kwargs)

    monkeypatch.setattr(product_executor, "build_graph", build)
    observed = _public_patch_upstream_scenario(tmp_path, monkeypatch, capsys, owner=owner)
    assert code in json.dumps(observed.result), observed.result
    assert observed.roles == ["product_requirement_projection", "product_requirement_findings", "planning", "patch"]
    assert observed.result["actual_write_performed"] is False
    assert observed.result["model_call_budget"]["consumed"] == 4
    loop_state = next(item for item in observed.worker_requests[0]
        if item.get("message_type") == "patch_agent_loop_state")
    assert loop_state["upstream_effects"][owner] == "provider_unavailable_stop"
    assert len(observed.searches) == len(observed.planning) == 1
    assert observed.evaluations == []


def test_product_upstream_second_request_stops_without_new_assessment(tmp_path, monkeypatch, capsys):
    """The current logical task keeps its consumed budget and one-cycle limit."""
    from tests.unit.test_mod_agent_cli import _public_patch_upstream_scenario

    observed = _public_patch_upstream_scenario(tmp_path, monkeypatch, capsys,
        owner="planning", repeat_upstream=True, model_call_budget_policy="stage_and_total")
    assert "PATCH_UPSTREAM_CONTINUATION_BUDGET_EXHAUSTED" in json.dumps(observed.result), observed.result
    assert observed.roles == ["product_requirement_projection", "product_requirement_findings",
        "planning", "patch", "planning", "patch"]
    assert observed.result["model_call_budget"]["consumed"] == 6
    assert observed.result["actual_write_performed"] is False
    assert len(observed.searches) == 1 and len(observed.planning) == 2
    assert observed.evaluations == []
    assert observed.result["model_call_budget"]["by_stage"]["initial_patch"] == {
        "maximum": 2, "consumed": 2, "remaining": 0}


@pytest.mark.parametrize("mismatch", ["failed_response", "failure_feedback", "product_task"])
def test_product_upstream_rejects_mismatched_current_failure_before_planning(
    tmp_path, monkeypatch, capsys, mismatch,
):
    """Real worker/evaluator fixtures; mutate one retained identity at admission."""
    from src.mod_agent.request_patch_execution import RequestPatchExecution
    from tests.unit.test_mod_agent_cli import _public_patch_upstream_scenario

    ordinary_validate = RequestPatchExecution.validate_upstream
    bound_contexts = []

    def validate(execution, state, artifact_store):
        # Prove the unmodified failure is valid against this worker's actual
        # source, generated response, persisted report and current Product task.
        context, _ = ordinary_validate(execution, state, artifact_store)
        bound_contexts.append(deepcopy(context))
        receipt = deepcopy(execution.patch_attempt_receipt())
        if mismatch == "failed_response":
            receipt["patch_agent_loop"]["behavior_evaluations"][-1]["model_response_sha256"] = "f" * 64
        elif mismatch == "failure_feedback":
            receipt["patch_agent_loop"]["behavior_correction"]["validator"]["generated_changes_sha256"] = "f" * 64
        else:
            execution.last_state["request_id"] = "unrelated_product_task"
        execution.last_worker._receipt = deepcopy(receipt)
        execution.receipts[-1] = deepcopy(receipt)
        execution.journals[-1].record({"phase": "test_mismatched_identity", "factory_receipt": receipt})
        return ordinary_validate(execution, state, artifact_store)

    monkeypatch.setattr(RequestPatchExecution, "validate_upstream", validate)
    observed = _public_patch_upstream_scenario(tmp_path, monkeypatch, capsys,
        owner="planning", fail_first=True, retain_plan=True, task_limit=10)
    expected = ("PATCH_UPSTREAM_ATTEMPT_MISMATCH" if mismatch == "product_task"
                else "PATCH_UPSTREAM_FAILURE_BINDING_REQUIRED")
    assert expected in json.dumps(observed.result), observed.result
    assert observed.roles == ["product_requirement_projection", "product_requirement_findings",
        "planning", "patch", "patch"]
    assert len(observed.evaluations) == 1
    assert observed.result["actual_write_performed"] is False
    assert len(bound_contexts) == 1 and bound_contexts[0]["current_failure_available"] is True
    current = bound_contexts[0]
    evaluated, report = observed.evaluations[0]
    assert current["generated_changes_sha256"] == evaluated["generated_changes_sha256"]
    assert current["source_snapshot_sha256"] == evaluated["source_snapshot_sha256"]
    assert current["failed_response_ref"]["sha256"] == evaluated["model_response"]["output_sha256"]
    assert current["failure_report_ref"] == report["report_ref"]
    for ref in (current["failed_response_ref"], current["failure_report_ref"]):
        assert hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
    assert current["failure_report"]["owner_binding_ids"] == [
        unit["edit_binding_id"] for unit in observed.patch_contexts[0]["required_change_units"]]
    assert len(observed.planning) == len(observed.searches) == 1
