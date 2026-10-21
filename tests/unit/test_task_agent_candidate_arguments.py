"""Candidate form errors use the existing single correction, not new authority.

为什么要有它：真实首候选的空父字段不能终止整个任务，反馈也不能放松版本、来源或修改范围。
"""
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.libs.llm.agent_turn import AgentTurnProviderError
from src.mod_agent.comparison_checkpoint import _request
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopError
from src.mod_agent.task_agent_loop import TaskCandidateRevisionInputError
from tests.unit.test_current_candidate import current_setup, revision  # noqa: F401
from tests.unit.test_fresh_planning_continuation import _source_hashes
from tests.unit.test_task_agent_loop import call, response
from tests.unit.test_task_agent_tools import (  # noqa: F401
    action,
    patch,
    setup,
    submit,
    upstream_request,
)


def malformed(session, fault):
    args = patch(session)
    if fault == "null_parent":
        args.update(parent_candidate_id=None, parent_code_sha256=None)
    elif fault == "patch_type":
        args["patch_json"] = {}
    elif fault == "json":
        args["patch_json"] = '{"changes":'
    elif fault == "missing_context":
        args.pop("context_id")
    else:
        payload = json.loads(args["patch_json"])
        payload["changes"][0]["invented_field"] = "unsupported"
        args["patch_json"] = json.dumps(payload)
        if fault == "outer_and_nested":
            args.update(parent_candidate_id=None, parent_code_sha256=None)
    return args


@pytest.mark.parametrize("fault", ["null_parent", "patch_type", "json", "missing_context", "nested", "outer_and_nested"])
def test_first_candidate_format_feedback_then_real_submit_and_check(current_setup, fault):  # noqa: F811
    inputs, session, oracle, budget = current_setup
    before = _source_hashes(inputs.workspace)
    turns = 0

    class Scripted:
        def complete_turn(self, request):
            nonlocal turns
            turns += 1
            if turns == 1:
                return response(call("candidate_submit", malformed(session, fault), "malformed"))
            if turns == 2:
                feedback = request.tool_results[0].output
                assert feedback["status"] == "rejected_not_executed"
                assert feedback["code"] == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"
                assert feedback["argument_errors"]
                if fault == "outer_and_nested":
                    assert len(feedback["argument_errors"]) >= 2
                assert session.current_candidate is None and not session.candidates and not oracle.calls
                assert not list(session.output_root.glob("candidate-tool-content-*"))
                assert budget.bytes > 0 and budget.reads == 0
                return response(call("candidate_submit", patch(session), "corrected"))
            if turns == 3:
                return response(call("candidate_check", {"candidate_id": session.current_candidate}, "check"))
            return response(call("finish", {"candidate_id": session.current_candidate}, "finish"))

    result = session.run(client=Scripted())
    assert result.status == "completed", result.termination
    assert result.tool_argument_correction_used and result.model_turns == 4
    assert len(session.candidates) == 1 and len(oracle.calls) == 1
    assert result.call_records["malformed"]["status"] == "not_dispatched"
    assert _source_hashes(inputs.workspace) == before


def test_candidate_schema_exposes_only_the_current_form(current_setup):  # noqa: F811
    _inputs, session, _oracle, _budget = current_setup
    schema = action(session, "candidate_submit").definition.parameters
    assert schema["properties"]["context_id"]["enum"] == [session.context_id]
    validator = Draft202012Validator(schema)
    first = patch(session)
    assert validator.is_valid(first)
    assert not validator.is_valid({**first, "parent_candidate_id": None, "parent_code_sha256": None})
    assert not validator.is_valid({"context_id": session.context_id})
    submit(session, correct=False)
    updated = revision(session)
    assert not validator.is_valid(updated)
    revised_schema = action(session, "candidate_submit").definition.parameters
    validator = Draft202012Validator(revised_schema)
    assert set(revised_schema["properties"]) == {"context_id", "parent_candidate_id", "parent_code_sha256", "edits"}
    assert revised_schema["properties"]["parent_candidate_id"]["enum"] == [updated["parent_candidate_id"]]
    assert revised_schema["properties"]["parent_code_sha256"]["enum"] == [updated["parent_code_sha256"]]
    assert "oneOf" not in revised_schema
    assert not validator.is_valid(first)
    assert validator.is_valid(updated)
    for field in ("parent_candidate_id", "parent_code_sha256", "edits"):
        invalid = deepcopy(updated)
        invalid.pop(field)
        assert not validator.is_valid(invalid)
    assert not validator.is_valid({**updated, "patch_json": first["patch_json"]})
    assert not validator.is_valid({**updated, "parent_code_sha256": None})


@pytest.mark.parametrize("resume", [False, True])
def test_loop_refreshes_candidate_form_across_submit_replan_and_restore(current_setup, resume):  # noqa: F811
    inputs, session, oracle, budget = current_setup
    original_sources = _source_hashes(inputs.workspace)
    requests, saved_schemas = [], []
    steps = [
        lambda: response(call("candidate_submit", patch(session, correct=False), "first")),
        lambda: response(call("candidate_check", {"candidate_id": session.current_candidate}, "failed-check")),
        lambda: response(call("upstream", {"context_id": session.context_id,
            "request": upstream_request(session)}, "replan")),
        lambda: response(call("candidate_read", {"context_id": session.context_id, "version": session.current_candidate,
            "path": sorted(session.allowed_edit_paths)[0]}, "read-current")),
        lambda: response(call("candidate_submit", revision(session), "revise")),
        lambda: response(call("candidate_check", {"candidate_id": session.current_candidate}, "passed-check")),
        lambda: response(call("finish", {"candidate_id": session.current_candidate}, "finish")),
    ]
    interrupted = False

    class Scripted:
        def complete_turn(self, request):
            nonlocal interrupted
            schema = next(tool.parameters for tool in request.tools if tool.name == "candidate_submit")
            expected = {"context_id", "patch_json"} if session.current_candidate is None else {
                "context_id", "parent_candidate_id", "parent_code_sha256", "edits"}
            assert set(schema["properties"]) == set(schema["required"]) == expected
            assert "oneOf" not in schema
            assert schema["properties"]["context_id"]["enum"] == [session.context_id]
            if session.current_candidate is not None:
                assert schema["properties"]["parent_candidate_id"]["enum"] == [session.current_candidate]
                assert schema["properties"]["parent_code_sha256"]["enum"] == [
                    session.candidates[session.current_candidate]["data"]["materialized_code_identity"]["code_sha256"]]
            requests.append(request)
            saved_schemas.append(deepcopy(schema))
            if resume and len(steps) == 4 and not interrupted:
                interrupted = True
                raise AgentTurnProviderError("controlled interruption after replan")
            return steps.pop(0)()

    first = session.run(client=Scripted())
    if resume:
        assert first.termination["code"] == "TASK_AGENT_PROVIDER_TURN_FAILED"
        result = session.run(client=Scripted(), initial_request=_request(asdict(first.request)),
            prior_call_records=first.call_records)
    else:
        result = first
    assert result.status == "completed", result.termination
    assert len(oracle.calls) == 2 and len(session.candidates) == 3
    assert budget.consumed == 8 + int(resume)
    assert all(next(tool.parameters for tool in request.tools if tool.name == "candidate_submit") == schema
        for request, schema in zip(requests, saved_schemas))
    assert _source_hashes(inputs.workspace) == original_sources


@pytest.mark.parametrize("fault", ["context", "foreign_parent", "foreign_hash", "binding", "changed_source"])
def test_format_errors_do_not_hide_identity_or_authority_errors(current_setup, fault):  # noqa: F811
    inputs, session, _oracle, _budget = current_setup
    args = malformed(session, "null_parent")
    if fault == "context":
        args["context_id"] = "old-context"
    elif fault == "foreign_parent":
        args["parent_candidate_id"] = "foreign-candidate"
    elif fault == "foreign_hash":
        args["parent_code_sha256"] = "f" * 64
    elif fault == "binding":
        payload = json.loads(args["patch_json"])
        payload["changes"].append({"edit_binding_id": "not-authorized", "wrong": True})
        args["patch_json"] = json.dumps(payload)
    else:
        source = inputs.workspace / sorted(session.allowed_edit_paths)[0]
        source.write_text(source.read_text(encoding="utf-8") + "\n// changed\n", encoding="utf-8")
    session.response_admission(response(call("candidate_submit", args, "invalid")))
    with pytest.raises((ValueError, PermissionError, PatchAgentLoopError)) as observed:
        action(session, "candidate_submit").prepare(args)
    assert not isinstance(observed.value, TaskCandidateRevisionInputError)
    assert not session.candidates and session.current_candidate is None


def test_format_feedback_requires_actual_admitted_call(current_setup):  # noqa: F811
    _inputs, session, _oracle, _budget = current_setup
    with pytest.raises(ValueError, match="ACTUAL_MODEL_CALL_REQUIRED"):
        action(session, "candidate_submit").prepare(malformed(session, "null_parent"))


@pytest.mark.parametrize("fault", ["arguments_ref", "response_ref", "parent_ref", "duplicate_actual"])
def test_malformed_arguments_do_not_soften_tampered_call_or_parent_evidence(current_setup, fault):  # noqa: F811
    _inputs, session, _oracle, _budget = current_setup
    if fault == "parent_ref":
        submit(session, correct=False)
        args = revision(session)
        args["unexpected"] = True
    else:
        args = malformed(session, "null_parent")
    session.response_admission(response(call("candidate_submit", args, "must-stay-hard")))
    if fault == "duplicate_actual":
        session._actual_calls.append(deepcopy(session._actual_calls[0]))
    else:
        reference = (session.candidates[session.current_candidate]["ref"] if fault == "parent_ref"
            else session._actual_calls[0][fault])
        path = Path(reference["path"])
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError) as observed:
        action(session, "candidate_submit").prepare(args)
    assert not isinstance(observed.value, TaskCandidateRevisionInputError)
    assert len(session.candidates) == (1 if fault == "parent_ref" else 0)


@pytest.mark.parametrize("fault", ["missing_edits", "edit_type", "extra_field", "missing_rationale"])
def test_revision_format_errors_are_correctable_without_creating_candidate(current_setup, fault):  # noqa: F811
    _inputs, session, _oracle, _budget = current_setup
    submit(session, correct=False)
    prior = session.current_candidate
    args = revision(session)
    if fault == "missing_edits":
        args.pop("edits")
    elif fault == "edit_type":
        args["edits"] = "not a list"
    elif fault == "missing_rationale":
        args["edits"][0].pop("rationale")
    else:
        args["patch_json"] = patch(session)["patch_json"]
    session.response_admission(response(call("candidate_submit", args, "bad-revision")))
    with pytest.raises(TaskCandidateRevisionInputError):
        action(session, "candidate_submit").prepare(args)
    assert session.current_candidate == prior and len(session.candidates) == 1


@pytest.mark.parametrize("fault", ["parent", "hash", "binding"])
def test_revision_authority_errors_precede_format_feedback(current_setup, fault):  # noqa: F811
    _inputs, session, _oracle, _budget = current_setup
    submit(session, correct=False)
    args = revision(session)
    args["unexpected"] = True
    if fault == "parent":
        args["parent_candidate_id"] = "foreign"
    elif fault == "hash":
        args["parent_code_sha256"] = "foreign"
    else:
        args["edits"].append({"edit_binding_id": "unauthorized", "bad": True})
    session.response_admission(response(call("candidate_submit", args, "mixed-invalid")))
    with pytest.raises((ValueError, PermissionError)) as observed:
        action(session, "candidate_submit").prepare(args)
    assert not isinstance(observed.value, TaskCandidateRevisionInputError)
    assert len(session.candidates) == 1


def test_second_candidate_input_error_does_not_gain_another_correction(current_setup):  # noqa: F811
    _inputs, session, _oracle, _budget = current_setup
    turns = 0
    class Scripted:
        def complete_turn(self, request):
            nonlocal turns
            turns += 1
            return response(call("candidate_submit", malformed(session, "null_parent"), f"wrong-{turns}"))
    result = session.run(client=Scripted())
    assert result.termination["code"] == "TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED"
    assert result.model_turns == 2 and result.tool_argument_correction_used
    assert not session.candidates


@pytest.mark.parametrize("limit", ["calls", "feedback"])
def test_first_candidate_format_error_keeps_existing_resource_caps(current_setup, limit):  # noqa: F811
    _inputs, session, _oracle, budget = current_setup
    if limit == "calls":
        budget.maximum = 1
    else:
        budget.max_bytes = 0
    class Scripted:
        def complete_turn(self, request):
            return response(call("candidate_submit", malformed(session, "null_parent"), "bad"))
    result = session.run(client=Scripted())
    assert result.termination["code"] == ("TASK_AGENT_MODEL_BUDGET_EXHAUSTED" if limit == "calls"
        else "TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED")
    assert result.model_turns == 1 and not result.tool_argument_correction_used
    assert not session.candidates
