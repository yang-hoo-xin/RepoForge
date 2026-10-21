"""Current-code revisions reuse immutable snapshots and the existing checker.

Scope: scripted model plus controlled oracle, not real model or C# delivery.
为什么要有它：先验证修改、失败、读取和再次修改的版本闭环，不把脚本修复当模型能力提升。
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.current_candidate import (
    context_transfer_content,
    derive_current_revision,
    validate_current_candidate_contract,
)
from src.mod_agent.task_agent_loop import TaskActionInputError
from src.mod_agent.task_agent_tools import TaskAgentSession
from tests.unit.test_fresh_planning_continuation import _source_hashes
from tests.unit.test_task_agent_loop import call, response
from tests.unit.test_task_agent_tools import action, patch, setup, submit  # noqa: F401


@pytest.fixture
def current_setup(setup):  # noqa: F811
    inputs, previous, oracle, budget = setup
    session = TaskAgentSession(store=previous.store, workspace_root=previous.workspace_root,
        context=previous.context, policy=previous.policy, generator=previous.generator, evaluator=oracle,
        allowed_edit_paths=tuple(previous.allowed_edit_paths), allowed_read_roots=previous.allowed_read_roots,
        output_root=previous.store.root / "current-session", prepare_contract=previous.prepare_contract,
        source_search=previous.source_search_owner, upstream=previous.upstream_owner,
        reserve_model_call=budget.reserve, budget_snapshot=budget.snapshot, charge_tool_budget=budget.charge,
        model_response_admission=lambda observed: None, candidate_mode="current")
    return inputs, session, oracle, budget


def revision(session, before="RecordMissingMarker();", after="RecordRuntimeMarker();"):
    current = session._candidate(session.current_candidate)
    return {"context_id": session.context_id, "parent_candidate_id": current["candidate_id"],
        "parent_code_sha256": current["materialized_code_identity"]["code_sha256"],
        "edits": [{"edit_binding_id": current["generated_changes"][0]["edit_binding_id"],
            "before_text": before, "after_text": after, "rationale": "Fix the observed missing marker"}]}


def revise(session, arguments=None, call_id="revision"):
    arguments = revision(session) if arguments is None else arguments
    session.response_admission(response(call("candidate_submit", arguments, call_id)))
    selected = action(session, "candidate_submit")
    return selected.execute(selected.prepare(arguments).value)


def read_current(session, **overrides):
    arguments = {"context_id": session.context_id, "version": session.current_candidate,
        "path": sorted(session.allowed_edit_paths)[0], **overrides}
    selected = action(session, "candidate_read")
    prepared = selected.prepare(arguments)
    assert prepared.read_cost == 1 and selected.role == "source_read"
    return selected.execute(prepared.value)


def test_scripted_modify_fail_read_revise_pass_keeps_origin_and_source(current_setup):
    inputs, session, oracle, budget = current_setup
    before = _source_hashes(inputs.workspace)
    turns = [lambda: call("candidate_submit", patch(session, correct=False), "first"),
        lambda: call("candidate_check", {"candidate_id": session.current_candidate}, "fail"),
        lambda: call("candidate_read", {"context_id": session.context_id, "version": session.current_candidate,
            "path": sorted(session.allowed_edit_paths)[0], "compare_to": "baseline"}, "read"),
        lambda: call("candidate_submit", revision(session), "revise"),
        lambda: call("candidate_check", {"candidate_id": session.current_candidate}, "pass"),
        lambda: call("finish", {"candidate_id": session.current_candidate}, "finish")]

    class Scripted:
        def complete_turn(self, request):
            return response(turns.pop(0)())

    result = session.run(client=Scripted())
    assert result.status == "completed", result.termination
    assert result.model_turns == 6 and budget.reads == 1 and budget.bytes > 0
    assert len(oracle.calls) == 2 and len(session.candidates) == 2
    assert _source_hashes(inputs.workspace) == before
    assert result.completion["whole_task_delivered"] is False
    current = session._candidate(session.current_candidate)
    origin = current["model_origin"]
    assert origin["kind"] == "derived_from_current_candidate_revision"
    actual = json.loads(Path(origin["arguments_ref"]["path"]).read_text())
    assert "patch_json" not in actual and "edits" in actual
    parent = json.loads(Path(origin["parent_candidate_ref"]["path"]).read_text())
    content = derive_current_revision(parent, arguments=actual, prepared_contract=session.contract)
    assert content.encode() == Path(origin["derived_content_ref"]["path"]).read_bytes()
    assert "RecordRuntimeMarker();" in read_current(session)["text"]


def test_read_previous_current_and_preserve_multiple_revisions(current_setup):
    _inputs, session, oracle, _budget = current_setup
    first = submit(session, correct=False)
    first_read = read_current(session, compare_to="baseline")
    assert "RecordMissingMarker();" in first_read["text"]
    assert "+" in first_read["comparison"]["unified_diff"]
    result = revise(session)
    second = result["candidate_id"]
    assert result["parent_candidate_id"] == first and result["code_sha256"]
    assert "RecordRuntimeMarker();" in read_current(session)["text"]
    assert "RecordMissingMarker();" in read_current(session, version=first)["text"]
    assert "RecordMissingMarker();" not in read_current(session, version="baseline")["text"]
    revise(session, revision(session, "RecordRuntimeMarker();",
        "RecordRuntimeMarker();\n        AnotherMarker();"), "second-revision")
    selected = read_current(session, compare_to=second)
    assert "RecordRuntimeMarker();" in selected["text"] and "AnotherMarker();" in selected["text"]
    assert "+        AnotherMarker();" in selected["comparison"]["unified_diff"]
    assert oracle.calls == []


def test_revision_invalidates_old_check_and_old_candidate_cannot_finish(current_setup):
    _inputs, session, _oracle, _budget = current_setup
    first = submit(session)
    check = action(session, "candidate_check")
    check.execute(check.prepare({"candidate_id": first}).value)
    assert session.current_check
    revised = revise(session, revision(session, "RecordRuntimeMarker();",
        "RecordRuntimeMarker();\n        AnotherMarker();"))
    assert session.current_check is None
    with pytest.raises(ValueError, match="NOT_CURRENT"):
        action(session, "finish").prepare({"candidate_id": first})
    prepared = action(session, "finish").prepare({"candidate_id": revised["candidate_id"]})
    assert not session.completion_check(prepared=prepared,
        output={"candidate_id": revised["candidate_id"]})["accepted"]


def test_local_revert_is_not_append_only_and_keeps_other_changes(current_setup):
    _inputs, session, _oracle, _budget = current_setup
    submit(session, correct=False)
    revise(session, revision(session, "RecordMissingMarker();",
        "RecordRuntimeMarker();\n        UnnecessaryMarker();"))
    revise(session, revision(session, "\n        UnnecessaryMarker();", ""), "remove-local-mistake")
    text = read_current(session)["text"]
    assert "RecordRuntimeMarker();" in text and "UnnecessaryMarker();" not in text


def test_explicit_snapshot_ids_change_read_arguments_and_never_overwrite_prior_read(current_setup):
    _inputs, session, _oracle, _budget = current_setup
    first = submit(session, correct=False)
    read_first = read_current(session, version=first)
    second = revise(session)["candidate_id"]
    read_second = read_current(session, version=second, compare_to=first)
    assert first != second and read_first["file_sha256"] != read_second["file_sha256"]
    assert read_current(session, version=first) == read_first
    with pytest.raises(ValueError, match="EXPLICIT_CANDIDATE_ID_REQUIRED"):
        read_current(session, version="current")


def test_candidate_read_range_is_explicit_and_does_not_expand_authority(current_setup):
    _inputs, session, _oracle, _budget = current_setup
    submit(session, correct=False)
    full = read_current(session)
    part = read_current(session, start_line=2, end_line=3)
    assert part["text"] == "".join(full["text"].splitlines(keepends=True)[1:3])
    assert part["total_utf8_bytes"] == full["total_utf8_bytes"] > len(part["text"].encode())
    assert part["total_lines"] == full["total_lines"] and part["start_line"] == 2 and part["end_line"] == 3


@pytest.mark.parametrize("fault", ["parent", "hash", "binding", "before", "reset"])
def test_revision_rejects_stale_unauthorized_or_reset_inputs(current_setup, fault):
    _inputs, session, _oracle, _budget = current_setup
    first = submit(session, correct=False)
    args = revision(session)
    if fault == "parent":
        args["parent_candidate_id"] = "foreign-candidate"
    elif fault == "hash":
        args["parent_code_sha256"] = "0" * 64
    elif fault == "binding":
        args["edits"][0]["edit_binding_id"] = "hidden-test"
    elif fault == "before":
        args["edits"][0]["before_text"] = "not in current code"
    else:
        args = patch(session)
    with pytest.raises((ValueError, PermissionError)):
        revise(session, args)
    assert session.current_candidate == first and len(session.candidates) == 1


@pytest.mark.parametrize("fault", [{"path": "../private/test.cs"}, {"version": "foreign-candidate"},
    {"start_line": True}, {"start_line": 999999}, {"context_id": "old"}])
def test_candidate_reads_reject_scope_range_version_errors(current_setup, fault):
    _inputs, session, _oracle, _budget = current_setup
    submit(session, correct=False)
    with pytest.raises((ValueError, PermissionError, TaskActionInputError)):
        read_current(session, **fault)


def test_candidate_read_past_eof_is_explicit_and_bad_start_never_dispatches(current_setup):
    _inputs, session, _oracle, _budget = current_setup
    submit(session, correct=False)
    full = read_current(session)
    clamped = read_current(session, end_line=200)
    assert clamped["text"] == full["text"] and clamped["end_line"] == full["total_lines"]
    assert clamped["requested_end_line"] == 200 and clamped["end_clamped_to_eof"] is True
    with pytest.raises(TaskActionInputError):
        action(session, "candidate_read").prepare({"context_id": session.context_id,
            "version": session.current_candidate, "path": sorted(session.allowed_edit_paths)[0], "start_line": 200})


def test_identical_snapshot_read_uses_existing_policy_without_extra_read_charge(current_setup):
    _inputs, session, _oracle, budget = current_setup
    session.policy = replace(session.policy, reuse_covered_source_observations=True)
    requests = []

    def read_call():
        return call("candidate_read", {"context_id": session.context_id, "version": session.current_candidate,
            "path": sorted(session.allowed_edit_paths)[0]}, "same-native-call-id")

    turns = [lambda: call("candidate_submit", patch(session, correct=False), "first"),
        read_call, read_call, lambda: call("candidate_submit", revision(session), "revise"),
        lambda: call("candidate_check", {"candidate_id": session.current_candidate}, "pass"),
        lambda: call("finish", {"candidate_id": session.current_candidate}, "finish")]

    class Scripted:
        def complete_turn(self, request):
            requests.append(request)
            return response(turns.pop(0)())

    result = session.run(client=Scripted())
    assert result.status == "completed", result.termination
    assert budget.reads == 1
    saved = result.call_records["same-native-call-id"]["result"]["output"]
    assert saved["execution"] == "reused"
    assert "historical" in saved["reuse_confirmation"]["evidence"]["budget_scope"]


def test_upstream_preserves_code_and_requires_new_check(current_setup):
    from tests.unit.test_task_agent_tools import upstream_request

    _inputs, session, _oracle, _budget = current_setup
    first = submit(session, correct=False)
    original = session._candidate(first)
    selected_check = action(session, "candidate_check")
    selected_check.execute(selected_check.prepare({"candidate_id": first}).value)
    selected = action(session, "upstream")
    selected.execute(selected.prepare({"context_id": session.context_id,
        "request": upstream_request(session)}).value)
    assert session.current_candidate is not None
    assert session.current_check is None
    assert session._candidate(session.current_candidate)["materialized_code_identity"] == original["materialized_code_identity"]
    assert "RecordMissingMarker();" in read_current(session, version=first)["text"]
    assert "RecordMissingMarker();" in read_current(session)["text"]
    revise(session)
    selected_check.execute(selected_check.prepare({"candidate_id": session.current_candidate}).value)
    finish = action(session, "finish").prepare({"candidate_id": session.current_candidate})
    assert session.completion_check(prepared=finish, output={"candidate_id": session.current_candidate})["accepted"]


def test_replan_changes_binding_identity_and_retains_readable_parent(current_setup):
    from src.mod_agent.task_agent_tools import TaskUpstreamAssessment
    from tests.unit.test_task_agent_tools import save, upstream_request

    inputs, session, _oracle, budget = current_setup
    first = submit(session, correct=False)
    old_binding = session.inspector.available_binding_ids[0]
    old_context = session.context_id
    owner = session.upstream_owner

    def replan(**kwargs):
        selected = owner(**kwargs)
        data = inputs.store.read(selected.context.implementation_plan_ref)["data"]
        plan = json.loads(json.dumps(data).replace(old_binding, "new-technical-region"))
        session.validate_context_transfer(feature_spec=session.documents.feature_spec,
            evidence_pack=session.documents.evidence_pack, implementation_plan=plan)
        plan_ref = inputs.store.write(feature_id=inputs.feature_ref.feature_id, artifact_type="implementation_plan",
            payload=plan, upstream_refs=[selected.context.feature_spec_ref, selected.context.evidence_pack_ref])
        context = replace(selected.context, implementation_plan_ref=plan_ref)
        assessment = json.loads(Path(selected.assessment_ref["path"]).read_text())
        assessment["output_context"] = context.to_dict()
        return TaskUpstreamAssessment(context, save(inputs.store.root / "upstream", assessment))

    session.upstream_owner = replan
    action(session, "upstream").execute(upstream_request(session))
    assert session.context_id != old_context
    assert old_binding not in session.inspector.available_binding_ids
    assert session.current_candidate != first
    assert budget.snapshot()["consumed"] == 1
    assert "RecordMissingMarker();" in read_current(session, version=first)["text"]
    origin = session._candidate(session.current_candidate)["model_origin"]
    assert origin["kind"] == "derived_from_context_transition" and "response_ref" not in origin
    revise(session)
    assert "RecordRuntimeMarker();" in read_current(session)["text"]
    assert session._candidate(session.current_candidate)["generated_changes"][0]["edit_binding_id"] == "new-technical-region"
    second = session.current_candidate
    action(session, "upstream").execute(upstream_request(session))
    assert budget.snapshot()["consumed"] == 2
    assert "RecordRuntimeMarker();" in read_current(session, version=second)["text"]
    assert "RecordMissingMarker();" in read_current(session, version=first)["text"]
    assert "RecordRuntimeMarker();" in read_current(session)["text"]


@pytest.mark.parametrize("outcome", ["retained_context", "blocked"])
def test_retained_or_blocked_reassessment_preserves_current_code(current_setup, outcome):
    from src.mod_agent.task_agent_tools import TaskUpstreamAssessment
    from tests.unit.test_task_agent_tools import digest, save, upstream_request

    inputs, session, _oracle, budget = current_setup
    first = submit(session)
    original_context = session.context_id

    def retained(*, request, context):
        owner_ref = save(inputs.store.root / "upstream", {"status": outcome})
        reference = save(inputs.store.root / "upstream", {"input_context": context.context.to_dict(),
            "output_context": context.context.to_dict(), "request_sha256": digest(request),
            "budget_before": budget.snapshot(), "budget_after": budget.snapshot(), "owner_result_refs": [owner_ref],
            "outcome": outcome})
        return TaskUpstreamAssessment(context.context, reference)

    session.upstream_owner = retained
    action(session, "upstream").execute(upstream_request(session))
    assert session.current_candidate == first and session.context_id == original_context
    assert "RecordRuntimeMarker();" in read_current(session)["text"]
    check = action(session, "candidate_check")
    check.execute(check.prepare({"candidate_id": first}).value)
    finish = action(session, "finish").prepare({"candidate_id": first})
    assert session.completion_check(prepared=finish, output={"candidate_id": first})["accepted"] is (outcome != "blocked")


def test_current_source_read_and_reuse_change_with_candidate(current_setup):
    _inputs, session, _oracle, _budget = current_setup
    session.policy = replace(session.policy, reuse_covered_source_observations=True)
    binding = session.inspector.available_binding_ids[0]
    tool = action(session, "source_read")
    original = tool.prepare({"binding_id": binding})
    submit(session, correct=False)
    selected = tool.prepare({"binding_id": binding})
    result = tool.execute(selected.value)
    assert selected.version_identity != original.version_identity
    assert "RecordMissingMarker();" in result["excerpt"]
    assert result["source_hash"] == read_current(session)["file_sha256"]
    assert result["candidate_id"] == session.current_candidate and result["baseline_source_hash"]
    revise(session)
    assert session._confirm_reuse("source_read", selected, {}) is None
    assert "RecordRuntimeMarker();" in tool.execute(tool.prepare({"binding_id": binding}).value)["excerpt"]


def test_two_replans_keep_latest_failure_not_older_checkpoint_or_completion(current_setup):
    from tests.unit.test_task_agent_tools import upstream_request

    _inputs, session, oracle, _budget = current_setup
    first = submit(session, correct=False)
    checker = action(session, "candidate_check")
    checker.execute(checker.prepare({"candidate_id": first}).value)
    oldest_check = json.loads(Path(session.current_check["path"]).read_text())
    oldest_candidate = session._candidate(first)
    # A previously admitted seed is older diagnostic history, never a new check.
    # 为什么要有它：有checkpoint的恢复入口不能覆盖后来实际产生的失败。
    session._checkpoint = SimpleNamespace(known_failure={"report_ref": oldest_check["report"]["report_ref"]})
    session._checkpoint_content_ref = oldest_candidate["model_origin"]["derived_content_ref"]
    latest = revise(session, revision(session, "RecordMissingMarker();", "RecordAnotherMissingMarker();"))["candidate_id"]
    checker.execute(checker.prepare({"candidate_id": latest}).value)
    latest_check_ref = deepcopy(session.current_check)
    latest_check = json.loads(Path(latest_check_ref["path"]).read_text())
    captured = []
    previous_owner = session.upstream_owner

    def capture(**kwargs):
        captured.append(session.upstream_failure_context())
        return previous_owner(**kwargs)

    session.upstream_owner = capture
    for index in range(2):
        arguments = {"context_id": session.context_id, "request": upstream_request(session)}
        session.response_admission(response(call("upstream", arguments, f"replan-{index}")))
        upstream = action(session, "upstream")
        upstream.execute(upstream.prepare(arguments).value)
        assert session.current_check is None
        finish = action(session, "finish").prepare({"candidate_id": session.current_candidate})
        assert not session.completion_check(prepared=finish, output={"candidate_id": session.current_candidate})["accepted"]
    assert captured[0]["origin"] == "current_checked_candidate"
    assert captured[1]["origin"] == "historical_checked_candidate_not_current_acceptance"
    assert captured[1]["failure_report_ref"] == latest_check["report"]["report_ref"]
    assert captured[1]["failure_check_ref"] == latest_check_ref
    assert captured[1]["failure_candidate_ref"] == session.candidates[latest]["ref"]
    assert captured[1]["failure_is_current_acceptance"] is False
    assert "RecordAnotherMissingMarker" in captured[1]["previous_response"]
    view = session.current_code_view()["last_checked_failure"]
    assert view["candidate_id"] == latest and view["check_ref"] == latest_check_ref
    assert view["eligible_for_current_completion"] is False
    assert len(oracle.calls) == 2
    checker.execute(checker.prepare({"candidate_id": session.current_candidate}).value)
    assert len(oracle.calls) == 3 and session.current_check != latest_check_ref
    assert session.current_code_view()["last_checked_failure"]["candidate_id"] == session.current_candidate


@pytest.mark.parametrize("field", ["candidate_id", "context_id", "materialized_code_identity"])
def test_historical_failure_does_not_accept_mismatched_version(current_setup, field):
    _inputs, session, _oracle, _budget = current_setup
    first = submit(session, correct=False)
    checker = action(session, "candidate_check")
    checker.execute(checker.prepare({"candidate_id": first}).value)
    session.last_checked_failure[field] = "different-version"
    with pytest.raises(ValueError, match="HISTORICAL_FAILURE_CHANGED"):
        session.current_code_view()


def test_context_transfer_maps_new_regions_without_fabricating_unchanged_units():
    from src.mod_agent.proposal.code_generator import PreparedNativePatchContract

    baseline = {"x.cs": b"first\nsecond\nthird\n"}
    current = {"x.cs": b"first-new\nsecond\nthird-new\n"}
    bindings = [{"binding_id": key, "canonical_target_path": "x.cs", "target_span": {"start_line": line, "end_line": line}}
        for key, line in (("new-a", 1), ("new-b", 2), ("new-c", 3))]
    context = {"required_change_units": [{"edit_binding_id": row["binding_id"]} for row in bindings]}
    contract = PreparedNativePatchContract((), {}, context, "test", "3", "hash")
    content, projection = context_transfer_content(baseline_files=baseline, current_files=current,
        implementation_plan={"binding_registry": {"edit_bindings": bindings}}, prepared_contract=contract)
    assert [row["edit_binding_id"] for row in json.loads(content)["changes"]] == ["new-a", "new-c"]
    assert len(projection.materialization_context["required_change_units"]) == 2
    assert len(contract.materialization_context["required_change_units"]) == 3
    limited = replace(contract, materialization_context={"required_change_units": context["required_change_units"][:2]})
    with pytest.raises(ValueError, match="SCOPE_NOT_COVERED"):
        context_transfer_content(baseline_files=baseline, current_files=current,
            implementation_plan={"binding_registry": {"edit_bindings": bindings}}, prepared_contract=limited)


def test_failed_transfer_preflight_keeps_original_candidate_context_and_bytes(current_setup):
    _inputs, session, _oracle, _budget = current_setup
    first = submit(session, correct=False)
    original_context = session.context_id
    plan = deepcopy(session.documents.implementation_plan)
    binding = plan["binding_registry"]["edit_bindings"][0]
    binding["target_span"] = {"start_line": 1, "end_line": 1}
    evidence = deepcopy(session.documents.evidence_pack)
    item = next(row for row in evidence["items"] if row["evidence_id"] == binding["evidence_id"])
    item["span"] = dict(binding["target_span"])
    item["excerpt"] = session._edit_path(binding["canonical_target_path"]).read_bytes().decode("utf-8").splitlines(keepends=True)[0]
    with pytest.raises(ValueError, match="SCOPE_NOT_COVERED"):
        session.validate_context_transfer(feature_spec=session.documents.feature_spec,
            evidence_pack=evidence, implementation_plan=plan)
    assert session.current_candidate == first and session.context_id == original_context
    assert "RecordMissingMarker();" in read_current(session)["text"]
    revise(session)


def test_reconstruction_preserves_other_files_and_disjoint_binding_regions():
    contract = SimpleNamespace(materialization_context={"binding_evidence_excerpts": {
        "a": "\ufeffA\r\n", "b": "B\r\n", "c": "C\n"}, "required_change_units": [
        {"edit_binding_id": key} for key in ("a", "b", "c")]})
    parent = {"context_id": "ctx", "candidate_id": "old",
        "materialized_code_identity": {"code_sha256": "hash"},
        "generated_changes": [{"edit_binding_id": key, "rationale": "prior", "rollback_hint": "restore",
            "patch": {"before_text": text, "after_text": text.replace(key.upper(), key.upper() + "1")}}
            for key, text in contract.materialization_context["binding_evidence_excerpts"].items()]}
    args = {"context_id": "ctx", "parent_candidate_id": "old", "parent_code_sha256": "hash", "edits": [
        {"edit_binding_id": "b", "before_text": "B1", "after_text": "B2", "rationale": "correct"}]}
    result = json.loads(derive_current_revision(parent, arguments=args, prepared_contract=contract))["changes"]
    assert [row["after_text"] for row in result] == ["\ufeffA1\r\n", "B2\r\n", "C1\n"]


def test_overlapping_binding_contract_is_rejected_before_model():
    contract = SimpleNamespace(contract_version="3", materialization_context={"required_change_units": [
        {"edit_binding_id": "a"}, {"edit_binding_id": "b"}]})
    plan = {"binding_registry": {"edit_bindings": [{"binding_id": "a", "canonical_target_path": "x.cs",
        "target_span": {"start_line": 1, "end_line": 10}}, {"binding_id": "b", "canonical_target_path": "x.cs",
        "target_span": {"start_line": 10, "end_line": 20}}]}}
    with pytest.raises(ValueError, match="OVERLAPPING_BINDINGS"):
        validate_current_candidate_contract(plan, contract)
    plan["binding_registry"]["edit_bindings"][1]["target_span"]["start_line"] = 11
    validate_current_candidate_contract(plan, contract)


def test_multiline_edit_from_actual_crlf_candidate_preserves_bom(short_workspace, monkeypatch):
    import tests.unit.test_fresh_planning_continuation as inputs_module

    original_write = inputs_module._write_sources

    def write_bom_crlf(workspace):
        path = original_write(workspace)
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        path.write_bytes(b"\xef\xbb\xbf" + text.replace("\n", "\r\n").encode("utf-8"))
        return path

    monkeypatch.setattr(inputs_module, "_write_sources", write_bom_crlf)
    inputs, session, _oracle, _budget = current_setup.__wrapped__(setup.__wrapped__(short_workspace))
    initial = _source_hashes(inputs.workspace)
    args = patch(session, correct=False)
    payload = json.loads(args["patch_json"])
    for change in payload["changes"]:
        change["after_text"] = change["after_text"].replace("\r\n", "\n").replace("\n", "\r\n")
    args["patch_json"] = json.dumps(payload)
    session.response_admission(response(call("candidate_submit", args, "crlf-first")))
    selected = action(session, "candidate_submit")
    selected.execute(selected.prepare(args).value)
    text = read_current(session)["text"]
    assert text.startswith("\ufeff") and "\r\n" in text
    start = text.index("KeepExistingFeedback();")
    end = text.index("RecordMissingMarker();") + len("RecordMissingMarker();")
    actual_current_excerpt = text[start:end]
    assert "\r\n" in actual_current_excerpt
    revise(session, revision(session, actual_current_excerpt,
        actual_current_excerpt.replace("RecordMissingMarker();", "RecordRuntimeMarker();")))
    revised = read_current(session)["text"]
    assert revised.startswith("\ufeff") and "\r\n" in revised
    assert "RecordRuntimeMarker();" in revised and "RecordMissingMarker();" not in revised
    assert "\n" not in revised.replace("\r\n", "")
    assert _source_hashes(inputs.workspace) == initial
