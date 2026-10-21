"""Real source/materializer/ArtifactStore bindings with a controlled Python oracle.

为什么要有它：验证实际工具装配和负控，不运行真实模型、C# Build或游戏，也不把受控oracle当C5实测。
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import pytest

from src.libs.llm import ChatResponse
from src.libs.llm.agent_turn import AgentMessage, AgentTurnRequest
from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.candidate_execution import materialize_candidate, materialized_code_identity
from src.mod_agent.comparison_checkpoint import CheckedComparisonCheckpoint
from src.mod_agent.evidence.rg_live import RgLiveEvidenceProvider
from src.mod_agent.proposal.code_generator import LLMExactPatchGenerator
from src.mod_agent.proposal.failure_state import failure_signature
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
from src.mod_agent.task_agent_loop import TaskActionInputError
from src.mod_agent.task_agent_tools import (
    SourceSearchResult,
    TaskAgentContext,
    TaskAgentSession,
    TaskUpstreamAssessment,
)
from tests.unit.test_fresh_planning_continuation import _inputs, _patch_request, _source_hashes
from tests.unit.test_task_agent_loop import Budget, call, response


def save(root, value):
    root.mkdir(parents=True, exist_ok=True)
    path = root / (uuid4().hex + ".json")
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class QualifiedPythonOracle:
    def __init__(self, root):
        self.root, self.calls = root, []
        self.identity = {"validator_sha256": "1" * 64, "inputs_sha256": "2" * 64,
            "environment_sha256": "3" * 64, "deterministic_reuse_allowed": False}
        self.after_call = None

    def verification_identity(self):
        return deepcopy(self.identity)

    def __call__(self, context):
        self.calls.append(deepcopy(context))
        changed = "\n".join(row["patch"]["after_text"] for row in context["generated_changes"])
        passed = "RecordRuntimeMarker();" in changed
        report = {"status": "passed" if passed else "behavior_failed",
            "failure_owner": None if passed else "patch_implementation",
            "owner_binding_ids": [] if passed else [row["edit_binding_id"] for row in context["generated_changes"]],
            "observations": [] if passed else [{"check_id": "synthetic_python_marker", "expected": True, "observed": False}],
            "execution_origin": "executed", "verification_scope": "synthetic_python_not_build_or_game",
            **{key: deepcopy(context[key]) for key in ("generated_changes_sha256", "source_snapshot_sha256",
                "materialized_code_identity", "implementation_plan_ref", "verification_identity")},
            "model_response_sha256": context["model_response"]["output_sha256"]}
        report["report_ref"] = save(self.root, report)
        if self.after_call:
            self.after_call()
        return report


@pytest.fixture
def setup(short_workspace):
    inputs = _inputs(short_workspace)
    request = _patch_request(inputs)
    context = TaskAgentContext(inputs.feature_ref, inputs.evidence_ref, request.implementation_plan_ref)
    budget = Budget(maximum=30, max_reads=10, max_bytes=200_000)
    generator = LLMExactPatchGenerator(None)
    evaluator = QualifiedPythonOracle(inputs.store.root / "oracle")

    def contract(documents):
        feature, plan, evidence = documents.feature_spec, documents.implementation_plan, documents.evidence_pack
        return generator.prepare_native_contract(user_request=feature["user_request"],
            evidence_items=evidence["items"], binding_registry=plan["binding_registry"],
            constraints=plan["side_effect_boundaries"], target_behavior=feature["target_behavior"],
            acceptance_criteria=feature["acceptance_criteria"], runtime_verification_plan=feature["runtime_verification_plan"],
            required_change_units=plan["proposed_changes"], required_validation_by_binding={
                row["edit_binding_id"]: row["expected_validation"] for row in plan["proposed_changes"]})

    def search(*, query, context):
        pack = RgLiveEvidenceProvider(inputs.workspace).collect(query=query,
            search_root="generated_mods/SyntheticMod", max_items=2)
        payload = pack.to_artifact_payload()
        sources = {}
        for item in pack.items:
            path = (inputs.workspace / item.source_path).resolve()
            sources[str(path)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        refs = tuple(sources.values())
        ref = save(inputs.store.root / "search", {"query": query, "payload": payload, "source_refs": list(refs)})
        return SourceSearchResult(query, payload, refs, ref)

    def upstream(*, request, context, source_observations=()):
        before = budget.snapshot()
        budget.reserve()  # Controlled nested provider cost, not a real provider invocation.
        plan = deepcopy(context.implementation_plan)
        plan["proposed_changes"][0]["rationale"] = "Controlled revised technical hypothesis"
        plan_ref = inputs.store.write(feature_id=inputs.feature_ref.feature_id, artifact_type="implementation_plan",
            payload=plan, upstream_refs=[context.context.feature_spec_ref, context.context.evidence_pack_ref])
        selected = replace(context.context, implementation_plan_ref=plan_ref)
        owner_result = save(inputs.store.root / "upstream", {"controlled_owner": "test_python_assessment",
            "plan": asdict(plan_ref), "model_execution": "synthetic_budget_only",
            "source_observations": list(source_observations)})
        assessment = save(inputs.store.root / "upstream", {"input_context": context.context.to_dict(),
            "output_context": selected.to_dict(), "request_sha256": digest(request),
            "budget_before": before, "budget_after": budget.snapshot(), "owner_result_refs": [owner_result]})
        return TaskUpstreamAssessment(selected, assessment)

    session = TaskAgentSession(store=inputs.store, workspace_root=inputs.workspace, context=context,
        policy=PatchAgentLoopPolicy(enable_supporting_reads=True, finalize_after_first_observation=False),
        generator=generator, evaluator=evaluator, allowed_edit_paths=("generated_mods/SyntheticMod/Behavior.cs",),
        allowed_read_roots=(inputs.workspace / "generated_mods/SyntheticMod",),
        output_root=inputs.store.root / "task-session", prepare_contract=contract,
        source_search=search, upstream=upstream, reserve_model_call=budget.reserve,
        budget_snapshot=budget.snapshot, charge_tool_budget=budget.charge,
        model_response_admission=lambda observed: None)
    return inputs, session, evaluator, budget


def patch(session, *, correct=True):
    prompt = json.loads(session.contract.messages[1].content)
    changes = []
    for unit in prompt["required_change_units"]:
        binding = unit["edit_binding_id"]
        excerpt = prompt["edit_binding_contexts"][binding]["source_excerpt"]
        changed = excerpt.replace("KeepExistingFeedback();", "KeepExistingFeedback();\n        "
            + ("RecordRuntimeMarker();" if correct else "RecordMissingMarker();"))
        changes.append({"edit_binding_id": binding, "after_text": changed,
            "rationale": "Controlled candidate", "rollback_hint": "Restore original source"})
    return {"context_id": session.context_id, "patch_json": json.dumps({"changes": changes})}


def action(session, name):
    return next(item for item in session.actions() if item.definition.name == name)


def upstream_request(session, owner="planning"):
    return {"required_owner": owner, "reason": "Reassess the observed failure",
            "source_refs": ["binding://" + session.inspector.available_read_binding_ids[0]]}


def submit(session, *, correct=True):
    arguments = patch(session, correct=correct)
    session.response_admission(response(call("candidate_submit", arguments, "submit-" + uuid4().hex)))
    selected = action(session, "candidate_submit")
    return selected.execute(selected.prepare(arguments).value)["candidate_id"]


def test_real_tool_owners_keep_model_provenance_and_source_bytes_through_replanning(setup):
    inputs, session, evaluator, budget = setup
    before = _source_hashes(inputs.workspace)
    history = []
    binding = session.inspector.available_binding_ids[0]
    steps = [lambda: call("source_search", {"query": "ChangeMarker"}, "search"),
        lambda: call("source_read", {"binding_id": binding}, "read"),
        lambda: call("candidate_submit", patch(session, correct=False), "bad-submit"),
        lambda: call("candidate_check", {"candidate_id": session.current_candidate}, "bad-check"),
        lambda: call("upstream", {"context_id": session.context_id, "request": upstream_request(session)}, "upstream"),
        lambda: call("candidate_submit", patch(session), "good-submit"),
        lambda: call("candidate_check", {"candidate_id": session.current_candidate}, "good-check"),
        lambda: call("finish", {"candidate_id": session.current_candidate}, "finish")]

    class Scripted:
        def complete_turn(self, request):
            history.append(request)
            return response(steps.pop(0)())

    result = session.run(client=Scripted())
    assert result.status == "completed", result.termination
    assert len(evaluator.calls) == 2 and len(session.candidates) == 2
    assert result.model_budget["consumed"] == 9 and result.model_turns == 8
    assert result.read_units == budget.reads == 2
    assert len(history[5].historical_tool_exchanges) == 4
    assert _source_hashes(inputs.workspace) == before
    assert result.completion["whole_task_delivered"] is False
    assert "continuous session" in history[0].messages[0].content
    assert any(isinstance(message.content, dict) and "candidate_submission_contract_instructions" in message.content
               for message in history[0].messages)
    for item in session.candidates.values():
        candidate = item["data"]
        origin = candidate["model_origin"]
        assert origin["kind"] == "derived_from_actual_tool_call_arguments"
        raw_response = json.loads(Path(origin["response_ref"]["path"]).read_text())
        assert raw_response["content"] == ""
        raw_args = json.loads(Path(origin["arguments_ref"]["path"]).read_text())
        assert raw_args["patch_json"].encode() == Path(origin["derived_content_ref"]["path"]).read_bytes()
    assert all(set(next(tool.parameters for tool in request.tools if tool.name == "candidate_submit")["properties"])
        == {"context_id", "patch_json"} for request in history)


def test_search_result_requires_explicit_evidence_upstream_before_replanning(setup):
    inputs, session, _evaluator, _budget = setup
    configuration = "generated_mods/SyntheticMod/Configuration.cs"
    session.allowed_edit_paths = frozenset((*session.allowed_edit_paths, configuration))
    session.edit_baselines[configuration] = hashlib.sha256(
        (inputs.workspace / configuration).read_bytes()).hexdigest()
    search = action(session, "source_search")
    search_args = {"query": "ConfigKey", "context_id": session.context_id}
    session.response_admission(response(call("source_search", search_args, "search-config")))
    observed = search.execute(search.prepare(search_args).value)
    selected = next(row for row in observed["items"] if row["source_path"].endswith("Configuration.cs"))
    reference = selected["reassessment_ref"]
    upstream = action(session, "upstream")
    planning = {"context_id": session.context_id, "request": {
        "required_owner": "planning", "reason": "Select the searched configuration source",
        "source_refs": [reference]}}
    session.response_admission(response(call("upstream", planning, "planning-search-ref")))
    with pytest.raises(ValueError, match="REQUEST_NOT_AUTHORIZED"):
        upstream.prepare(planning)

    captured = {}
    original = session.upstream_owner
    def owner(*, request, context, source_observations=()):
        captured["observations"] = deepcopy(source_observations)
        return original(request=request, context=context, source_observations=source_observations)
    session.upstream_owner = owner
    evidence = {"context_id": session.context_id, "request": {
        "required_owner": "evidence", "reason": "Admit the searched configuration source before replanning",
        "source_refs": [reference]}}
    session.response_admission(response(call("upstream", evidence, "evidence-search-ref")))
    result = upstream.execute(upstream.prepare(evidence).value)
    assert result["upstream_outcome"] == "replanned"
    assert captured["observations"] == [selected]
    assert result["old_candidate_invalidated"] is True


def test_initial_task_instructions_and_tool_descriptions_disclose_batch_policy(setup):
    _inputs_value, session, _evaluator, _budget = setup
    instructions = session.initial_request().messages[0].content
    assert "Batch only source_read and source_search" in instructions
    assert "upstream, candidate_submit, candidate_check and finish one call at a time" in instructions
    assert "为什么要有它" in instructions
    for action_value in session.actions():
        description = action_value.definition.description
        assert ("May be batched only" if action_value.role in {"source_read", "source_search"} else "Invoke alone") in description
        assert "为什么要有它" in description


def test_candidate_check_reuses_lossless_feedback_projection_without_changing_raw_evidence(setup):
    from src.mod_agent.proposal.observation_value_refs import GUIDE, restore_observation_values

    inputs, session, evaluator, _budget = setup
    state = {"packet": {"session": 101, "sequence": 1}, "before": {"seen": 1}, "after": {"seen": 2},
        "events": [{"action": "receive", "value": "unchanged observation"} for _ in range(12)]}

    def detailed(context):
        report = evaluator(context)
        report["observations"] = [{"check_id": f"synthetic-{index}", "expected": 1,
            "observed": {"actual": 2, "diagnostic": deepcopy(state)}} for index in range(4)]
        report.pop("report_ref")
        report["report_ref"] = save(inputs.store.root / "detailed-oracle", report)
        return report

    detailed.verification_identity = evaluator.verification_identity
    session.evaluator = detailed
    candidate = submit(session, correct=False)
    check = action(session, "candidate_check")
    visible = check.execute(check.prepare({"candidate_id": candidate}).value)
    assert GUIDE in visible
    raw_bytes = Path(session.current_check["path"]).read_bytes()
    raw = json.loads(raw_bytes)["report"]
    restored = restore_observation_values(visible)
    assert restored["observations"] == raw["observations"]
    assert restored["candidate_id"] == candidate and restored["status"] == "behavior_failed"
    assert len(json.dumps(visible)) < len(json.dumps(restored))
    assert Path(session.current_check["path"]).read_bytes() == raw_bytes
    assert GUIDE not in raw


def test_candidate_submit_cannot_invent_provider_content_or_use_old_context(setup):
    _, session, _, _ = setup
    arguments = patch(session)
    with pytest.raises(ValueError, match="ACTUAL_MODEL_CALL_REQUIRED"):
        action(session, "candidate_submit").prepare(arguments)
    session.response_admission(response(call("candidate_submit", arguments, "actual")))
    with pytest.raises(ValueError, match="CONTEXT_VERSION_CHANGED"):
        action(session, "candidate_submit").prepare({**arguments, "context_id": "old"})


def test_read_keeps_exact_model_view_and_does_not_soften_permission_errors(setup):
    _, session, _, _ = setup
    read = action(session, "source_read")
    with pytest.raises(PermissionError):
        read.prepare({"path": "../private/checker.py"})
    with pytest.raises(TaskActionInputError):
        read.prepare({})
    binding = session.inspector.available_binding_ids[0]
    prepared = read.prepare({"binding_id": binding})
    observed = read.execute(prepared.value)
    original = session.inspector.inspect(binding_id=binding, observation_id=observed["observation_id"]).model_view()
    assert observed == original and prepared.read_cost == 1


@pytest.mark.parametrize("owner", ["source_search_owner", "upstream_owner"])
def test_missing_actual_owner_blocks_qualification_before_model(setup, owner):
    _, session, _, _ = setup
    setattr(session, owner, None)
    assert not session.qualification()["ready"]
    with pytest.raises(ValueError, match="OWNERS_NOT_QUALIFIED"):
        session.run(client=object())


@pytest.mark.parametrize("fault", ["claimed_success", "outside_edit", "wrong_requirement", "changed_result"])
def test_upstream_requires_real_bound_context_and_preserves_authority(setup, fault):
    inputs, session, _, budget = setup
    original = session.context
    old = session.upstream_owner

    def invalid(*, request, context):
        if fault == "claimed_success":
            return {"status": "success"}
        result = old(request=request, context=context)
        if fault == "wrong_requirement":
            return replace(result, context=replace(result.context,
                feature_spec_ref=ArtifactRef(inputs.feature_ref.feature_id, "feature_spec", inputs.feature_ref.path, "0" * 64)))
        if fault == "changed_result":
            Path(result.assessment_ref["path"]).write_text("{}")
            return result
        plan = deepcopy(context.implementation_plan)
        binding = plan["binding_registry"]["edit_bindings"][0]
        binding["canonical_target_path"] = "generated_mods/SyntheticMod/Configuration.cs"
        binding["source_hash"] = hashlib.sha256((inputs.workspace / binding["canonical_target_path"]).read_bytes()).hexdigest()
        changed = inputs.store.write(feature_id=inputs.feature_ref.feature_id, artifact_type="implementation_plan",
            payload=plan, upstream_refs=[inputs.feature_ref, inputs.evidence_ref])
        selected = replace(result.context, implementation_plan_ref=changed)
        saved = json.loads(Path(result.assessment_ref["path"]).read_text())
        saved["output_context"] = selected.to_dict()
        return TaskUpstreamAssessment(selected, save(inputs.store.root / "upstream", saved))

    session.upstream_owner = invalid
    selected = action(session, "upstream")
    with pytest.raises(ValueError):
        selected.execute(selected.prepare({"context_id": session.context_id, "request": upstream_request(session)}).value)
    assert session.context == original
    assert budget.consumed == (0 if fault == "claimed_success" else 1)


def test_upstream_invalidates_old_candidate_and_does_not_reuse_old_pass(setup):
    _, session, _, _ = setup
    candidate = submit(session)
    checker = action(session, "candidate_check")
    checker.execute(checker.prepare({"candidate_id": candidate}).value)
    selected = action(session, "upstream")
    selected.execute(selected.prepare({"context_id": session.context_id, "request": upstream_request(session)}).value)
    assert candidate in session.candidates and session.current_candidate is None and session.current_check is None
    with pytest.raises(ValueError, match="CANDIDATE_NOT_CURRENT"):
        action(session, "finish").prepare({"candidate_id": candidate})


@pytest.mark.parametrize("fault", ["source_drift", "verifier_drift", "callback_failure"])
def test_checker_guards_survive_failure_and_do_not_leave_old_pass_deliverable(setup, fault):
    inputs, session, evaluator, _ = setup
    candidate = submit(session)
    checker = action(session, "candidate_check")
    checker.execute(checker.prepare({"candidate_id": candidate}).value)
    assert session.current_check is not None
    if fault == "source_drift":
        evaluator.after_call = lambda: inputs.behavior.write_text("changed during evaluator")
    elif fault == "verifier_drift":
        evaluator.after_call = lambda: evaluator.identity.update(validator_sha256="4" * 64)
    else:
        def fail():
            raise OSError("controlled unknown completion")
        evaluator.after_call = fail
    with pytest.raises((ValueError, RuntimeError)):
        checker.execute(checker.prepare({"candidate_id": candidate}).value)
    assert session.current_check is None
    assert "unresolved_candidate_execution" in session.qualification()["missing_owners"]


def test_tampered_model_output_and_result_are_not_accepted_at_finish(setup):
    _, session, _, _ = setup
    candidate = submit(session)
    checker = action(session, "candidate_check")
    checker.execute(checker.prepare({"candidate_id": candidate}).value)
    checked = json.loads(Path(session.current_check["path"]).read_text())
    Path(checked["report"]["report_ref"]["path"]).write_text("{}")
    finish = action(session, "finish")
    prepared = finish.prepare({"candidate_id": candidate})
    with pytest.raises(ValueError, match="FILE_BINDING_CHANGED"):
        session.completion_check(prepared=prepared, output=finish.execute(prepared.value))


def test_rechecking_failed_candidate_keeps_finish_rejected_with_repair_reason(setup):
    _, session, _, _ = setup
    candidate = submit(session, correct=False)
    checker, finish = action(session, "candidate_check"), action(session, "finish")
    for _ in range(2):
        checker.execute(checker.prepare({"candidate_id": candidate}).value)
        prepared = finish.prepare({"candidate_id": candidate})
        result = session.completion_check(prepared=prepared, output=finish.execute(prepared.value))
        assert result["accepted"] is False
        assert result["reason"] == "current_candidate_check_failed"
        assert result["check_status"] == "behavior_failed"
        assert "revise" in result["next_step"]
        assert result["whole_task_delivered"] is False
    assert session.current_candidate == candidate and len(session.candidates) == 1


def test_finish_instructions_require_a_passed_check_not_merely_a_new_check(setup):
    _, session, _, _ = setup
    description = action(session, "finish").definition.description
    instruction = session.initial_request().messages[0].content
    assert "status=passed" in description and "status=passed" in instruction
    assert "A rejected finish may be requested again after a real new check" not in description
    assert "then finish to request independent completion checking" not in instruction


@pytest.mark.parametrize("fault", ["claimed_success", "outside_root", "changed_hash"])
def test_search_cannot_turn_arbitrary_success_or_private_source_into_observation(setup, fault):
    inputs, session, _, _ = setup
    original = session.source_search_owner

    def invalid(*, query, context):
        if fault == "claimed_success":
            return {"status": "success"}
        result = original(query=query, context=context)
        assert result.source_refs
        refs = [dict(row) for row in result.source_refs]
        payload = deepcopy(result.payload)
        if fault == "outside_root":
            private = inputs.store.root / "private-checker.py"
            private.write_text("controlled private verifier fixture")
            refs[0] = {"path": str(private), "sha256": hashlib.sha256(private.read_bytes()).hexdigest()}
            payload["items"][0].update(source_path=str(private), source_hash=refs[0]["sha256"])
        else:
            refs[0]["sha256"] = "0" * 64
        ref = save(inputs.store.root / "search", {"query": query, "payload": payload, "source_refs": refs})
        return SourceSearchResult(query, payload, tuple(refs), ref)

    session.source_search_owner = invalid
    selected = action(session, "source_search")
    with pytest.raises(ValueError):
        selected.execute(selected.prepare({"query": "ChangeMarker"}).value)


def test_owner_cannot_return_a_pass_for_wrong_materialized_candidate(setup):
    inputs, session, evaluator, _ = setup
    candidate = submit(session)

    def wrong(context):
        report = evaluator(context)
        report["materialized_code_identity"] = {"code_sha256": "wrong"}
        body = {key: value for key, value in report.items() if key != "report_ref"}
        report["report_ref"] = save(inputs.store.root / "wrong-result", body)
        return report

    wrong.verification_identity = evaluator.verification_identity
    session.evaluator = wrong
    checker = action(session, "candidate_check")
    with pytest.raises(RuntimeError, match="CANDIDATE_REPORT_BINDING_CHANGED"):
        checker.execute(checker.prepare({"candidate_id": candidate}).value)
    assert session.current_check is None


@pytest.mark.parametrize("enabled", [False, True])
def test_source_reuse_obeys_selected_policy_through_normal_session(setup, enabled):
    _, session, _, budget = setup
    session.policy = replace(session.policy, reuse_covered_source_observations=enabled)
    session.documents, session.inspector, session.contract = session._admit_context(session.context)
    budget.maximum = 2
    selected = session.inspector.available_binding_ids[0]
    requests = []

    class Scripted:
        def complete_turn(self, request):
            requests.append(request)
            return response(call("source_read", {"binding_id": selected}, "same-read-id"))

    result = session.run(client=Scripted())
    assert len(requests) == 2
    assert result.read_units == budget.reads == 1
    assert len(list(session.output_root.glob("source-read-*.json"))) == 1
    if enabled:
        assert result.termination["code"] == "TASK_AGENT_MODEL_BUDGET_EXHAUSTED"
        assert result.request.tool_results[0].output["execution"] == "reused"
        assert result.request.tool_results[0].output["reuse_confirmation"]["evidence"]
    else:
        assert result.termination["code"] == "TASK_AGENT_BATCH_NOT_ADMITTED"
        assert result.request.tool_results[0].output["execution"] == "executed"
        assert not any(event.get("mode") == "reuse" for event in result.events)


def checked_history_fixture(inputs, session, evaluator, budget):
    """Controlled prefix admission callbacks; not the real C5 extractor qualification."""
    content = patch(session, correct=False)["patch_json"]
    generated = session.generator.materialize_prepared_native(session.contract, ChatResponse(content=content, model="controlled"))
    materialized = materialize_candidate(workspace_root=inputs.workspace,
        implementation_plan=session.documents.implementation_plan, changes=generated.changes)
    model_path = inputs.store.root / "historical-output.txt"
    model_path.write_text(content, encoding="utf-8")
    model_ref = {"path": str(model_path), "output_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    context = {"generated_changes": list(generated.changes), "generated_changes_sha256": digest(generated.changes),
        "source_snapshot_sha256": session.inspector.source_snapshot_sha256,
        "materialized_code_identity": materialized_code_identity(materialized),
        "implementation_plan_ref": asdict(session.context.implementation_plan_ref),
        "verification_identity": evaluator.verification_identity(), "model_response": model_ref}
    historical = evaluator(context)
    worker_ref = save(inputs.store.root / "controlled-worker", {"behavior_evaluator": {"kwargs": {"compilation_kwargs": {
        "workspace_root": str(inputs.workspace), "artifact_store_root": str(inputs.store.root)}}}})
    request = AgentTurnRequest((AgentMessage("assistant", content), AgentMessage("user", {
        "message_type": "patch_behavior_correction", "validator_feedback": {"observations": historical["observations"]}})))
    checked = CheckedComparisonCheckpoint(trial_id="controlled-unit-trial", request=request, history={},
        known_failure=historical, known_code=materialized_code_identity(materialized),
        consumed_model_turns=2, consumed_tool_rounds=0, consumed_observation_bytes=0,
        product_consumed=5, product_remaining=13, routing_calls=1,
        no_progress_seed={failure_signature(historical, ()): 1}, source_facts=(), wall_seconds=255,
        expected_model_identity={"model": "controlled"}, worker_policy=session.policy,
        worker_configuration_ref=worker_ref, source_refs=(historical["report_ref"], worker_ref),
        provenance={"current_artifacts": session._selected_artifacts(), "fixture_scope": "controlled_unit_only"})
    projection = {key: deepcopy(historical[key]) for key in ("status", "failure_owner", "owner_binding_ids",
        "observations", "materialized_code_identity", "verification_identity")}
    admission = {"verification_identity": evaluator.verification_identity(), "known_failure_projection": projection,
        "no_progress_seed": {failure_signature(projection, ()): 1}, "provenance": {"scope": "controlled_unit_only"}}
    budget.maximum, budget.consumed = 18, 5
    return checked, admission


@pytest.mark.parametrize("new_source_fact", [False, True])
def test_checked_failure_seed_stops_after_three_equal_states_without_reset_or_extra_send(setup, new_source_fact):
    inputs, session, evaluator, budget = setup
    checked, admission = checked_history_fixture(inputs, session, evaluator, budget)
    revalidated = []

    def revalidate():
        revalidated.append(True)
        return deepcopy(checked)

    session.bind_comparison_checkpoint(checked, revalidate=revalidate, admit_verifier=lambda *_: deepcopy(admission))
    assert revalidated == [True] and session.current_candidate is None and session.current_check is None
    assert not session.candidates and len(evaluator.calls) == 1
    initial = session.initial_request()
    assert any(message.role == "assistant" and message.content == checked.request.messages[0].content for message in initial.messages)
    sent = []
    binding = session.inspector.available_read_binding_ids[0]

    def submit_call(number):
        arguments = patch(session, correct=False)
        value = json.loads(arguments["patch_json"])
        value["changes"][0]["rationale"] = "Same code, controlled explanation " + str(number)
        arguments["patch_json"] = json.dumps(value)
        return call("candidate_submit", arguments, "submit" + str(number))

    steps = ([lambda: call("source_read", {"binding_id": binding}, "new-read")] if new_source_fact else []) + [
        lambda: submit_call(1), lambda: call("candidate_check", {"candidate_id": session.current_candidate}, "check1"),
        lambda: submit_call(2), lambda: call("candidate_check", {"candidate_id": session.current_candidate}, "check2")]
    if new_source_fact:
        budget.maximum = budget.consumed + len(steps)

    class Scripted:
        def complete_turn(self, request):
            sent.append(request)
            return response(steps.pop(0)())

    result = session.run(client=Scripted())
    assert not result.unknown_effects and len(evaluator.calls) == 3 and not steps
    assert result.model_turns == len(sent) == (5 if new_source_fact else 4)
    assert result.model_budget["consumed"] == 5 + len(sent)
    if new_source_fact:
        assert result.termination["code"] == "TASK_AGENT_MODEL_BUDGET_EXHAUSTED"
        assert sorted(session._failure_counts.values()) == [1, 2]
    else:
        assert result.termination["code"] == "PATCH_BEHAVIOR_NO_PROGRESS"
        assert result.termination["same_state_observations"] == 3
        assert result.call_records["check2"]["status"] == "completed"


@pytest.mark.parametrize("fault", ["changed_checkpoint", "changed_verifier", "changed_report"])
def test_checkpoint_binding_does_not_trust_dataclass_or_unproven_verifier_claim(setup, fault):
    inputs, session, evaluator, budget = setup
    checked, admission = checked_history_fixture(inputs, session, evaluator, budget)
    current = replace(checked, trial_id="other") if fault == "changed_checkpoint" else checked
    if fault == "changed_verifier":
        admission["verification_identity"]["validator_sha256"] = "9" * 64
    if fault == "changed_report":
        Path(checked.known_failure["report_ref"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        session.bind_comparison_checkpoint(checked, revalidate=lambda: current, admit_verifier=lambda *_: admission)
    assert session._checkpoint is None and not session._failure_counts and not session.candidates
