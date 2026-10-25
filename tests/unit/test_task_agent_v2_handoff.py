"""Controlled provenance and product consumption, not model-success evidence."""
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.mod_agent.execution.patch_apply import ExactTextPatchApplier
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.proposal.builder import PatchProposalBuilder
from src.mod_agent.proposal.reviewer import ProposalReviewer
from src.mod_agent.task_agent_execution import _worker_binding
from src.mod_agent.task_agent_v2_handoff import (
    TaskAgentPatchOutcomeV2,
    validate_task_agent_handoff_v2,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact


def save(root, name, data):
    path = root / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture(params=["binding_id", "path"])
def handoff_case(tmp_path, monkeypatch, request):
    root = tmp_path / "store"
    root.mkdir()
    target = tmp_path / "source"
    target.mkdir()
    (target / "A.cs").write_text("class A { int Value = 1; }", encoding="utf-8")
    authority = save_v2_artifact(root, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": ["A.cs"], "read_roots": {"target": str(target)}})
    task = TaskAgentV2State.create(root, "controlled", target, allowed_edit_paths=["A.cs"], authority_ref=authority,
        contract_fields={"public_acceptance_ref": "controlled-acceptance", "feature_spec_ref": {
            "feature_id": "controlled", "artifact_type": "feature_spec", "path": "feature.json", "artifact_hash": "abc"}})
    parent = task.current_revision_ref
    view = task.read_source(parent, "A.cs")
    target_arg = {"path": "A.cs"}
    if request.param == "binding_id":
        target_arg = {"binding_id": task.add_binding(task.registry_ref, view["source_view_ref"], "A.cs")["binding_id"]}
    args = {"parent_revision_ref": parent, "edits": [{**target_arg,
        "source_view_ref": view["source_view_ref"], "before_text": "Value = 1", "after_text": "Value = 2",
        "rationale": "Controlled replacement."}]}
    materialized_args = deepcopy(args)
    if request.param == "path_mismatch":
        args["edits"][0]["path"] = "B.cs"
    origin = {"call_id": "candidate_submit0", "arguments_ref": save(root, "candidate_submit0-args.json", args),
        "response_ref": save(root, "candidate_submit0-response.json", {"provider": "ollama_chat_local",
            "model": "controlled", "provider_state": {}, "stop_reason": "tool_calls",
            "tool_calls": [{"name": "candidate_submit", "call_id": "candidate_submit0", "arguments": args}]})}
    candidate = task.submit(**materialized_args, operation_id="submit-1", model_origin=origin)
    profile = {"model": "controlled"}
    records, calls, events = [], {}, []

    def call(name, arguments, value):
        call_id = name + str(len(records))
        response = {"provider": "ollama_chat_local", "model": "controlled", "provider_state": {},
            "stop_reason": "tool_calls", "tool_calls": [{"name": name, "call_id": call_id, "arguments": arguments}]}
        response_ref = save(root, call_id + "-response.json", response)
        output = {"execution": "executed", "value": value}
        event = {"event": "tool_dispatch_completed", "call_id": call_id, "output_sha256": canonical_hash(output)}
        calls[call_id] = {"status": "completed", "name": name, "arguments_sha256": canonical_hash(arguments),
            "result": {"output": output}}
        records.append({"name": name, "call_id": call_id, "arguments_ref": save(root, call_id + "-args.json", arguments),
            "response_ref": response_ref, "result_ref": save(root, call_id + "-result.json", output), "event": event})
        events.append(event)
        return response_ref

    submit_response = call("candidate_submit", args, candidate)
    identity = {"name": "controlled-python-checker"}
    request = {"task_contract_ref": task.contract_ref, "binding_registry_ref": task.registry_ref,
        "candidate_ref": candidate["candidate_ref"], "candidate_source_id": candidate["candidate_source_id"],
        "verification_identity": identity,
        "model_response": {"path": submit_response["path"], "output_sha256": submit_response["sha256"]}}
    report = {"status": "passed", "verification_identity": identity}
    check_ref = task.save("check", {**request, "request_ref": save(root, "check-request.json", request), "report": report})
    call("candidate_check", {"candidate_ref": candidate["candidate_ref"]}, {"check_ref": check_ref})
    finish_ref = call("finish", {"candidate_ref": candidate["candidate_ref"], "check_ref": check_ref}, {})
    factory = save(root, "factory.json", {})
    worker = save(root, "worker.json", {"behavior_evaluator": {"factory_ref": factory, "factory_name": "controlled", "kwargs": {}}})
    def budget(count):
        return {"policy": "meter_only", "maximum": None, "remaining": None, "consumed": count,
            "by_stage": {"initial_patch": {"consumed": count, "maximum": None, "remaining": None}},
            "count_basis": "reserved_provider_invocations_not_transport_proof"}
    before, after = budget(0), budget(3)
    termination = {"code": "TASK_AGENT_COMPLETION_ACCEPTED"}
    loop = {"status": "completed", "unknown_effects": False, "model_budget": after, "termination": termination,
        "events": events, "call_records": calls, "model_turns": 3}
    feature = {"feature_id": "controlled", "artifact_type": "feature_spec", "path": "feature.json", "artifact_hash": "abc"}
    context = {"feature_spec_ref": feature}
    outcome = TaskAgentPatchOutcomeV2("candidate_ready", termination, context, context, task.contract_ref,
        task.registry_ref, candidate["candidate_ref"], check_ref, None, save(root, "loop.json", loop),
        finish_ref, before, after)
    execution = {**outcome.to_dict(), "workspace_root": str(tmp_path), "allowed_edit_paths": ["source/A.cs"],
        "worker_binding": _worker_binding(worker), "expected_verification_identity": identity,
        "model_profile": profile, "completed_tool_records": records}
    outcome = replace(outcome, execution_ref=save(root, "execution.json", execution))
    checked = []
    def validate(report, *, context, **kwargs):
        assert report["status"] == "passed"
        assert context == request
        checked.append(context)
    monkeypatch.setattr("src.mod_agent.domain.csharp_repository_backend.validate_candidate_report_v2", validate)
    return SimpleNamespace(root=root, target=target, task=task, outcome=outcome, execution=execution, loop=loop,
        records=records, checked=checked, feature=feature, kwargs={"store": SimpleNamespace(root=root),
            "workspace_root": tmp_path, "expected_initial_context": context, "allowed_edit_paths": ["source/A.cs"],
            "worker_configuration_ref": worker, "expected_verification_identity": identity})


def test_controlled_candidate_reaches_proposal_review_and_no_write_dryrun(handoff_case):
    c = handoff_case
    handoff = validate_task_agent_handoff_v2(c.outcome, **c.kwargs)
    proposal = PatchProposalBuilder().build_v2(handoff, feature_id="controlled", feature_spec_ref=c.feature,
        store=c.kwargs["store"]).to_artifact_payload()
    review = ProposalReviewer().review_v2(proposal, store=c.kwargs["store"], feature_spec_ref=c.feature).to_artifact_payload()
    dry = ExactTextPatchApplier().dry_run_v2(proposal=proposal, review=review, store=c.kwargs["store"])
    assert dry["candidate_ref"] == c.outcome.candidate_ref
    assert dry["delivery_status"] == "pending" and dry["actual_write_performed"] is False
    assert "Value = 1" in (c.target / "A.cs").read_text()
    assert "Value = 2" in dry["unified_diff"]
    assert len(c.checked) >= 4
    assert "implementation_plan_hash" not in proposal["context_pack"]


@pytest.mark.parametrize("handoff_case", ["path_mismatch"], indirect=True)
def test_platform_cannot_materialize_a_different_path_than_the_retained_model_submission(handoff_case):
    with pytest.raises(ValueError, match="V2_SUBMITTED_BYTES_CHANGED"):
        validate_task_agent_handoff_v2(handoff_case.outcome, **handoff_case.kwargs)


@pytest.mark.parametrize("tamper", ["missing_submit", "wrong_result", "wrong_finish", "authority"])
def test_tampered_execution_cannot_cross_product_boundary(handoff_case, tamper):
    c = handoff_case
    execution = deepcopy(c.execution)
    if tamper == "missing_submit":
        execution["completed_tool_records"] = execution["completed_tool_records"][1:]
    elif tamper == "wrong_result":
        execution["completed_tool_records"][0]["result_ref"] = save(c.root, "wrong.json", {"execution": "executed", "value": {}})
    elif tamper == "wrong_finish":
        execution["completed_tool_records"][-1]["response_ref"] = execution["completed_tool_records"][0]["response_ref"]
    else:
        execution["allowed_edit_paths"] = ["source/B.cs"]
    outcome = replace(c.outcome, execution_ref=save(c.root, "tampered-execution.json", execution))
    with pytest.raises(ValueError):
        validate_task_agent_handoff_v2(outcome, **c.kwargs)


def test_proposal_bytes_and_current_preimage_are_rechecked(handoff_case):
    c = handoff_case
    handoff = validate_task_agent_handoff_v2(c.outcome, **c.kwargs)
    proposal = PatchProposalBuilder().build_v2(handoff, feature_id="controlled", feature_spec_ref=c.feature,
        store=c.kwargs["store"]).to_artifact_payload()
    tampered = deepcopy(proposal)
    tampered["diff_intent"][0]["patch"]["after"] = "unrelated"
    with pytest.raises(ValueError, match="V2_PROPOSAL_CANDIDATE_CHANGED"):
        ProposalReviewer().review_v2(tampered, store=c.kwargs["store"], feature_spec_ref=c.feature)
    review = ProposalReviewer().review_v2(proposal, store=c.kwargs["store"], feature_spec_ref=c.feature).to_artifact_payload()
    (c.target / "A.cs").write_text("changed concurrently", encoding="utf-8")
    with pytest.raises(RuntimeError, match="V2_SOURCE_PREIMAGE_CHANGED"):
        ExactTextPatchApplier().dry_run_v2(proposal=proposal, review=review, store=c.kwargs["store"])
