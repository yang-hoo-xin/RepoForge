"""Thin Product task execution with real source/Store and a Python-only oracle."""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.libs.llm.agent_turn import AgentToolCall, AgentTurnResponse
from src.mod_agent.evidence import RgLiveEvidenceProvider
from src.mod_agent.evidence.reassessment import RequirementCoverageReassessmentFactory
from src.mod_agent.local_entry import (
    LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
    LocalOllamaInitialPatchProfileV1,
)
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
from src.mod_agent.task_agent_execution import (
    TaskAgentExecution,
    TaskAgentPatchOutcome,
    load_task_agent_outcome,
)
from src.mod_agent.task_agent_tools import TaskAgentContext
from tests.unit.test_fresh_planning_continuation import _inputs, _patch_request
from tests.unit.test_model_planning_selection import _Client, _selector
from tests.unit.test_task_agent_tools import QualifiedPythonOracle


def file_ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class SharedBudget:
    def __init__(self):
        self.counts = {"initial_patch": 0, "repair": 0, "planning": 1, "requirement_projection": 1, "requirement_findings": 1}

    def snapshot(self):
        used = sum(self.counts.values())
        return {"policy": "task_total_only", "maximum": 18, "consumed": used, "remaining": 18 - used,
            "by_stage": {role: {"consumed": count} for role, count in self.counts.items()},
            "count_basis": "reserved_provider_invocations_not_transport_proof"}

    def reserve(self, role):
        assert sum(self.counts.values()) < 18
        self.counts[role] += 1


def use_meter_only(case):
    def snapshot():
        return {"policy": "meter_only", "maximum": None, "remaining": None,
            "consumed": sum(case.budget.counts.values()),
            "by_stage": {role: {"consumed": count, "maximum": None, "remaining": None}
                for role, count in case.budget.counts.items()},
            "count_basis": "reserved_provider_invocations_not_transport_proof"}

    def reserve(role):
        case.budget.counts[role] += 1

    case.budget.snapshot, case.budget.reserve = snapshot, reserve
    case.kwargs.update(budget_snapshot=snapshot, reserve_model_call=reserve,
        policy=replace(case.kwargs["policy"], model_call_budget_policy="meter_only"))


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("No actual network or model is allowed")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


@pytest.fixture
def case(short_workspace):
    inputs = _inputs(short_workspace)
    request = _patch_request(inputs)
    context = TaskAgentContext(inputs.feature_ref, inputs.evidence_ref, request.implementation_plan_ref)
    feature = inputs.store.read(inputs.feature_ref)["data"]
    plan = inputs.store.read(request.implementation_plan_ref)["data"]
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    state = {"run_id": "bounded-task-execution", "request_id": "same-original-request", "feature_id": feature["feature_id"],
        "artifacts": {kind: getattr(context, kind + "_ref").path for kind in ("feature_spec", "evidence_pack", "implementation_plan")},
        "artifact_hashes": {kind: getattr(context, kind + "_ref").artifact_hash for kind in ("feature_spec", "evidence_pack", "implementation_plan")}}
    evaluator = QualifiedPythonOracle(inputs.store.root / "oracle")
    factory = inputs.store.root / "controlled-owner.py"
    factory.write_text("# Bound identity only; test directly supplies a Python oracle.\n", encoding="utf-8")
    worker = {"behavior_evaluator": {"factory_ref": file_ref(factory), "factory_name": "controlled_python_oracle", "kwargs": {}}}
    worker_path = inputs.store.root / "worker.json"
    worker_path.write_text(json.dumps(worker), encoding="utf-8")
    worker_ref = file_ref(worker_path)
    evaluator.configuration_receipt = lambda: {"configuration_ref": worker_ref, **worker["behavior_evaluator"],
        "kwargs_sha256": canonical_hash({})}
    binding = evaluator.configuration_receipt()
    binding.pop("kwargs")
    evaluator.configuration_receipt = lambda: deepcopy(binding)
    profile = LocalOllamaInitialPatchProfileV1(profile_version=LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
        base_url="http://127.0.0.1:11434", model="controlled", context_window=8192, output_budget=1024, timeout_seconds=30)
    identity = {"name": "controlled", "model": "controlled", "digest": "a" * 64, "modified_at": "2026-10-18T00:00:00Z",
        "size": 1, "details": {"context_length": 8192}, "capabilities": ["completion", "tools"], "base_url": profile.base_url}
    providers = {"planning_target_selection_factory": _selector(inputs, _Client()),
        "evidence_pack_factory": RequirementCoverageReassessmentFactory(artifact_store=inputs.store, workspace_root=inputs.workspace,
            search_root="generated_mods/SyntheticMod", source_provider=RgLiveEvidenceProvider(inputs.workspace),
            semantic_provider=object(), query_selector=lambda *_: [], max_additional_queries=3)}
    budget, requests, mode = SharedBudget(), [], {"finish": True, "provider_error": None}

    class Scripted:
        provider, model = "ollama_chat_local", "controlled"

        def complete_turn(self, current):
            requests.append(current)
            if mode["provider_error"]:
                raise mode["provider_error"]
            turn = len(requests)
            if mode.get("upstream"):
                view = next(message.content for message in current.messages
                    if isinstance(message.content, dict) and "context_id" in message.content)
                name, arguments = "upstream", {"context_id": view["context_id"], "request": {
                    "required_owner": "evidence", "reason": "Inspect the current failure", "source_refs": [
                        "binding://" + view["allowed_read_binding_ids"][0]]}}
            elif turn == 1:
                payloads = [json.loads(message.content) for message in current.messages
                    if isinstance(message.content, str) and message.content.startswith("{")]
                payload = next(row for row in payloads if "required_change_units" in row)
                context_id = next(message.content["context_id"] for message in current.messages
                    if isinstance(message.content, dict) and "context_id" in message.content)
                changes = [{"edit_binding_id": unit["edit_binding_id"], "after_text":
                    payload["edit_binding_contexts"][unit["edit_binding_id"]]["source_excerpt"].replace(
                        "KeepExistingFeedback();", "KeepExistingFeedback();\n        RecordRuntimeMarker();"),
                    "rationale": "Controlled Python test", "rollback_hint": "Restore original source"}
                    for unit in payload["required_change_units"]]
                name, arguments = "candidate_submit", {"context_id": context_id, "patch_json": json.dumps({"changes": changes})}
            elif turn == 2:
                name, arguments = "candidate_check", {"candidate_id": current.tool_results[-1].output["value"]["candidate_id"]}
            elif mode["finish"]:
                name, arguments = "finish", {"candidate_id": current.tool_results[-1].output["value"]["candidate_id"]}
            else:
                return AgentTurnResponse(self.provider, self.model, "passed", provider_state={"content": "passed"})
            call = AgentToolCall("controlled-" + str(turn), name, arguments)
            return AgentTurnResponse(self.provider, self.model, "", (call,), provider_state={"calls": [asdict(call)]})

    kwargs = dict(artifact_store=inputs.store, workspace_root=inputs.workspace, profile=profile,
        policy=PatchAgentLoopPolicy(max_total_model_turns=18, max_tool_rounds=6, max_observation_bytes=32768,
            enable_supporting_reads=True, finalize_after_first_observation=False, model_call_budget_policy="task_total_only"),
        behavior_evaluator=evaluator, worker_configuration_ref=worker_ref, expected_model_identity=identity,
        identity_reader=lambda *_: deepcopy(identity), providers=providers, validate_providers=lambda: None,
        reserve_model_call=budget.reserve, budget_snapshot=budget.snapshot,
        allowed_edit_paths=("generated_mods/SyntheticMod/Behavior.cs",),
        allowed_read_roots=(inputs.workspace / "generated_mods/SyntheticMod",),
        output_root=inputs.store.root / "task-execution", client_factory=lambda _: Scripted())
    return SimpleNamespace(inputs=inputs, context=context, state=state, plan=plan, evidence=evidence,
        evaluator=evaluator, budget=budget, requests=requests, mode=mode, kwargs=kwargs)


def execute(case):
    owner = TaskAgentExecution(**case.kwargs)
    return owner, owner(case.state, case.plan, case.evidence)


def execute_current_revision(case, *, rejected_finish=False):
    """Exercise real tool dispatch and two oracle checks, without model or compiler."""
    first_client = case.kwargs["client_factory"](case.kwargs["profile"])
    evidence = {}

    class RepairingClient:
        provider, model = "ollama_chat_local", "controlled"

        def complete_turn(self, request):
            if not case.requests:
                first = first_client.complete_turn(request)
                call = first.tool_calls[0]
                arguments = {**call.arguments, "patch_json": call.arguments["patch_json"].replace(
                    "RecordRuntimeMarker();", "RecordWrongMarker();")}
                evidence["context_id"] = arguments["context_id"]
                evidence["binding_id"] = json.loads(arguments["patch_json"])["changes"][0]["edit_binding_id"]
                name = "candidate_submit"
            else:
                case.requests.append(request)
                turn = len(case.requests)
                observed = request.tool_results[-1].output["value"]
                if rejected_finish and turn == 3:
                    evidence["first_status"] = observed["status"]
                    call = AgentToolCall("current-revision-" + str(turn), "finish",
                        {"candidate_id": evidence["first_id"]})
                    return AgentTurnResponse(self.provider, self.model, "", (call,), provider_state={"calls": [asdict(call)]})
                if rejected_finish and turn > 3:
                    if turn == 4:
                        evidence["rejected_finish"] = deepcopy(request.tool_results[-1].output["completion_check"])
                    turn -= 1
                if turn == 2:
                    evidence["first_id"] = observed["candidate_id"]
                    name, arguments = "candidate_check", {"candidate_id": observed["candidate_id"]}
                elif turn == 3:
                    if "status" in observed:
                        evidence["first_status"] = observed["status"]
                    name, arguments = "candidate_read", {"context_id": evidence["context_id"], "version": evidence["first_id"],
                        "path": "generated_mods/SyntheticMod/Behavior.cs", "compare_to": "baseline"}
                elif turn == 4:
                    evidence["read"] = observed
                    name, arguments = "candidate_submit", {"context_id": evidence["context_id"],
                        "parent_candidate_id": observed["version"], "parent_code_sha256": observed["code_sha256"],
                        "edits": [{"edit_binding_id": evidence["binding_id"], "before_text": "RecordWrongMarker();",
                            "after_text": "RecordRuntimeMarker();", "rationale": "Correct the observed marker failure"}]}
                elif turn == 5:
                    evidence["second_id"] = observed["candidate_id"]
                    name, arguments = "candidate_check", {"candidate_id": observed["candidate_id"]}
                else:
                    evidence["second_status"] = observed["status"]
                    name, arguments = "finish", {"candidate_id": observed["candidate_id"]}
            call = AgentToolCall("current-revision-" + str(len(case.requests)), name, arguments)
            return AgentTurnResponse(self.provider, self.model, "", (call,), provider_state={"calls": [asdict(call)]})

    case.kwargs.update(candidate_mode="current", client_factory=lambda _: RepairingClient())
    owner, outcome = execute(case)
    return owner, outcome, evidence


def execute_context_transfer(case, *, revise=True):
    """Normal owner and session path with byte-preserving replan, then a fresh check."""
    first_client = case.kwargs["client_factory"](case.kwargs["profile"])
    observed, planning_requests = {}, []

    class PlanningClient:
        provider, model = "local_test", "planning-test-model"

        def complete_turn(self, request):
            case.budget.reserve("planning")
            planning_requests.append(request)
            payload = request.messages[1].content
            question = ("verification_questions" if "verification_questions" in request.response_format.schema["properties"]
                        else "missing_evidence")
            return AgentTurnResponse(self.provider, self.model, json.dumps({"decision": "select_targets",
                "selected_evidence_ids": payload["allowed_edit_evidence_ids"], "rationale": "Reassess current code", question: []}))

    class Client:
        provider, model = "ollama_chat_local", "controlled"

        def complete_turn(self, request):
            if not case.requests:
                first = first_client.complete_turn(request)
                arguments = deepcopy(first.tool_calls[0].arguments)
                if revise:
                    arguments["patch_json"] = arguments["patch_json"].replace("RecordRuntimeMarker();", "RecordWrongMarker();")
                observed["initial_context_id"] = arguments["context_id"]
                name = "candidate_submit"
            else:
                case.requests.append(request)
                turn = len(case.requests)
                value = request.tool_results[-1].output["value"]
                if turn == 2:
                    observed["first_id"] = value["candidate_id"]
                    name, arguments = "candidate_check", {"candidate_id": value["candidate_id"]}
                elif turn == 3:
                    observed["first_status"] = value["status"]
                    initial_view = next(message.content for message in request.messages
                        if isinstance(message.content, dict) and "allowed_read_binding_ids" in message.content)
                    name, arguments = "upstream", {"context_id": observed["initial_context_id"], "request": {
                        "required_owner": "planning", "reason": "Review the actual checked candidate and plan",
                        "source_refs": ["binding://" + initial_view["allowed_read_binding_ids"][0]]}}
                elif turn == 4:
                    observed["transfer"] = deepcopy(value)
                    assert value["current_code_preserved"] and value["new_check_required"]
                    if revise:
                        name, arguments = "candidate_read", {"context_id": value["context_id"],
                            "version": value["current_candidate_id"], "path": "generated_mods/SyntheticMod/Behavior.cs",
                            "compare_to": observed["first_id"]}
                    else:
                        name, arguments = "candidate_check", {"candidate_id": value["current_candidate_id"]}
                elif revise and turn == 5:
                    observed["read"] = deepcopy(value)
                    name, arguments = "candidate_submit", {"context_id": observed["transfer"]["context_id"],
                        "parent_candidate_id": value["version"], "parent_code_sha256": value["code_sha256"],
                        "edits": [{"edit_binding_id": observed["transfer"]["allowed_edit_binding_ids"][0],
                            "before_text": "RecordWrongMarker();", "after_text": "RecordRuntimeMarker();",
                            "rationale": "Correct the observed marker failure on current code"}]}
                elif revise and turn == 6:
                    name, arguments = "candidate_check", {"candidate_id": value["candidate_id"]}
                else:
                    observed["final_status"] = value["status"]
                    name, arguments = "finish", {"candidate_id": value["candidate_id"]}
            call = AgentToolCall("transfer-" + str(len(case.requests)), name, arguments)
            return AgentTurnResponse(self.provider, self.model, "", (call,), provider_state={"calls": [asdict(call)]})

    case.kwargs["providers"]["planning_target_selection_factory"] = _selector(case.inputs, PlanningClient())
    case.kwargs.update(candidate_mode="current", client_factory=lambda _: Client())
    owner, outcome = execute(case)
    return owner, outcome, observed, planning_requests


@pytest.mark.parametrize("revise", [True, False])
def test_normal_execution_replans_current_code_rechecks_and_handoffs(case, revise):
    before = case.inputs.behavior.read_bytes()
    owner, outcome, observed, planning = execute_context_transfer(case, revise=revise)
    assert outcome.status == "candidate_ready", outcome.termination
    handoff = owner.validate_handoff(outcome, case.context)
    assert outcome.final_context != outcome.initial_context
    assert observed["final_status"] == "passed" and len(case.evaluator.calls) == 2
    assert observed["first_status"] == ("behavior_failed" if revise else "passed")
    assert handoff.candidate["model_origin"]["kind"] == (
        "derived_from_current_candidate_revision" if revise else "derived_from_context_transition")
    assert len(planning) == 1 and case.budget.counts["planning"] == 2
    assert "RecordWrongMarker" in planning[0].messages[1].content["upstream_assessment"]["current_code_view"]["files"][0]["text"] if revise else True
    assert case.inputs.behavior.read_bytes() == before
    if revise:
        assert observed["read"]["comparison"]["unified_diff"] == ""
        assert observed["read"]["version"] != observed["first_id"]


@pytest.mark.parametrize("tamper", ["remove_transfer", "transfer_order", "assessment_bytes"])
def test_handoff_requires_actual_host_transfer_and_original_assessment(case, tamper):
    owner, outcome, _, _ = execute_context_transfer(case)
    assert outcome.status == "candidate_ready", outcome.termination
    owner.validate_handoff(outcome, case.context)
    document = json.loads(Path(outcome.execution_ref["path"]).read_bytes())
    transfer = next(row for row in document["completed_tool_records"] if row["role"] == "upstream")
    if tamper == "assessment_bytes":
        candidate = json.loads(Path(transfer["candidate_ref"]["path"]).read_bytes())
        reference = candidate["model_origin"]["assessment_ref"]
        Path(reference["path"]).write_text("{}", encoding="utf-8")
    else:
        document["completed_tool_records"].remove(transfer)
        if tamper == "transfer_order":
            document["completed_tool_records"].insert(0, transfer)
        path = Path(outcome.execution_ref["path"]).with_name("changed-transfer-order.json")
        path.write_text(json.dumps(document), encoding="utf-8")
        outcome = replace(outcome, execution_ref=file_ref(path))
    with pytest.raises(ValueError):
        owner.validate_handoff(outcome, case.context)


def test_current_code_failure_read_revision_check_is_rebuilt_for_handoff(case):
    before = case.inputs.behavior.read_bytes()
    owner, outcome, observed = execute_current_revision(case)
    assert outcome.status == "candidate_ready", outcome.termination
    handoff = owner.validate_handoff(outcome, case.context)
    assert observed["first_status"] == "behavior_failed" and observed["second_status"] == "passed"
    assert "RecordWrongMarker();" in observed["read"]["text"]
    assert "RecordWrongMarker();" in observed["read"]["comparison"]["unified_diff"]
    assert observed["first_id"] != observed["second_id"] == handoff.candidate["candidate_id"]
    assert len(case.evaluator.calls) == 2 and len(case.requests) == 6
    assert case.inputs.behavior.read_bytes() == before and not outcome.whole_task_delivered
    origin = handoff.candidate["model_origin"]
    assert origin["kind"] == "derived_from_current_candidate_revision"
    arguments = json.loads(Path(origin["arguments_ref"]["path"]).read_bytes())
    assert "patch_json" not in arguments
    assert arguments["parent_candidate_id"] == observed["first_id"]
    assert "RecordRuntimeMarker();" in handoff.changes[0]["patch"]["after_text"]
    assert "RecordWrongMarker();" not in handoff.changes[0]["patch"]["after_text"]


def test_rejected_finish_does_not_prevent_later_valid_current_candidate_handoff(case):
    owner, outcome, observed = execute_current_revision(case, rejected_finish=True)
    assert outcome.status == "candidate_ready", outcome.termination
    assert observed["rejected_finish"]["accepted"] is False
    handoff = owner.validate_handoff(outcome, case.context)
    assert handoff.candidate["candidate_id"] == observed["second_id"]
    assert len(case.requests) == 7 and len(case.evaluator.calls) == 2
    execution = json.loads(Path(outcome.execution_ref["path"]).read_bytes())
    finishes = [row for row in execution["completed_tool_records"] if row["role"] == "finish"]
    assert len(finishes) == 2
    assert outcome.finish_response_ref == finishes[-1]["response_ref"]
    with pytest.raises(ValueError, match="OUTCOME_RECORD_CHANGED|FINISH_CANDIDATE_CHANGED"):
        owner.validate_handoff(replace(outcome, finish_response_ref=finishes[0]["response_ref"]), case.context)


@pytest.mark.parametrize("damage", ["missing_first_submit", "submission_order", "old_report", "mode", "derived_bytes"])
def test_current_handoff_rejects_history_or_derived_origin_substitution(case, monkeypatch, damage):
    owner, outcome, _ = execute_current_revision(case)
    assert outcome.status == "candidate_ready", outcome.termination
    document = json.loads(Path(outcome.execution_ref["path"]).read_bytes())
    if damage == "derived_bytes":
        monkeypatch.setattr("src.mod_agent.current_candidate.derive_current_revision", lambda *_args, **_kwargs: "{}")
    elif damage == "mode":
        owner.candidate_mode = "baseline"
    else:
        records = document["completed_tool_records"]
        if damage == "missing_first_submit":
            first = next(row for row in records if row["role"] == "candidate_submit")
            records.remove(first)
        elif damage == "submission_order":
            positions = [index for index, row in enumerate(records) if row["role"] == "candidate_submit"]
            records[positions[0]], records[positions[1]] = records[positions[1]], records[positions[0]]
        else:
            old_check = next(row["check_ref"] for row in records if row["role"] == "candidate_check")
            document["check_ref"] = old_check
            outcome = replace(outcome, check_ref=old_check)
        changed = Path(outcome.execution_ref["path"]).with_name("substituted-execution.json")
        changed.write_text(json.dumps(document), encoding="utf-8")
        outcome = replace(outcome, execution_ref=file_ref(changed))
    with pytest.raises(ValueError):
        owner.validate_handoff(outcome, case.context)


def test_real_finish_and_exact_native_tool_candidate_are_handed_off_without_delivery_claim(case):
    owner, outcome = execute(case)
    assert isinstance(outcome, TaskAgentPatchOutcome) and outcome.status == "candidate_ready", outcome.termination
    handoff = owner.validate_handoff(outcome, case.context)
    assert handoff.final_context == case.context and handoff.changes
    assert len(case.requests) == 3 and len(case.evaluator.calls) == 1
    assert outcome.budget_before["consumed"] == 3 and outcome.budget_after["consumed"] == 6
    assert not outcome.whole_task_delivered and not outcome.unknown_effects
    assert all("patch_agent_loop" not in row.get("generation", {}) for row in handoff.changes)
    record = json.loads(Path(outcome.execution_ref["path"]).read_bytes())
    assert record["source_before"] == record["source_after"]
    assert record["run_id"] == case.state["run_id"] and record["request_id"] == case.state["request_id"]
    assert record["worker_binding"] == case.evaluator.configuration_receipt()
    assert record["product_store_attempts"] == []
    assert len(record["provider_turns"]) == 3 and all(row["status"] == "responded" for row in record["provider_turns"])
    assert owner.execution_receipt() == outcome
    restored = load_task_agent_outcome(outcome.execution_ref, artifact_store_root=case.inputs.store.root)
    assert restored == outcome and owner.validate_handoff(restored, case.context) == handoff
    assert TaskAgentPatchOutcome.from_dict(outcome.to_dict()) == outcome
    with pytest.raises(ValueError, match="SINGLE_INVOCATION"):
        owner(case.state, case.plan, case.evidence)


def test_product_trace_reuses_transport_observer_without_extra_calls(case):
    from src.core.trace.execution_trace import ExecutionTrace

    trace = ExecutionTrace("task-execution-test", "same-request", "local")
    with trace.activate():
        owner, outcome = execute(case)
    assert outcome.status == "candidate_ready", outcome.termination
    assert len(case.requests) == 3 and len(case.evaluator.calls) == 1
    assert owner.validate_handoff(outcome, case.context)
    assert trace.context.stages


def test_initial_task_model_view_shows_the_real_read_allowance(case):
    owner, outcome = execute(case)
    assert outcome.status == "candidate_ready", outcome.termination
    context = next(message.content for message in case.requests[0].messages
        if isinstance(message.content, dict) and "context_id" in message.content)
    assert context["read_budget"] == {"used": 0, "limit": 6, "remaining": 6,
        "count_unit": "one_source_read_or_source_search_call"}
    assert owner.read_budget_snapshot() == context["read_budget"]


def test_meter_only_current_repair_replans_and_handoffs_with_cumulative_counts(case):
    use_meter_only(case)
    case.budget.counts["initial_patch"] = 30
    owner, outcome, observed, planning = execute_context_transfer(case)
    assert outcome.status == "candidate_ready", outcome.termination
    assert observed["first_status"] == "behavior_failed" and observed["final_status"] == "passed"
    assert len(planning) == 1 and len(case.requests) == 7 and len(case.evaluator.calls) == 2
    assert outcome.budget_before["consumed"] == 33 and outcome.budget_after["consumed"] == 41
    assert outcome.budget_after["maximum"] is None and outcome.budget_after["remaining"] is None
    assert observed["read"]["read_budget"] == {"policy": "meter_only", "used": 1,
        "limit": None, "remaining": None, "count_unit": "one_source_read_or_source_search_call"}
    assert owner.validate_handoff(outcome, case.context).changes
    assert "no experimental quota" in case.requests[0].messages[0].content
    json.dumps(outcome.to_dict(), allow_nan=False)


def test_meter_only_tool_counts_exceed_historical_limits_without_reset(case):
    use_meter_only(case)
    owner = TaskAgentExecution(**case.kwargs)
    owner._charge(reads=100, observation_bytes=1_000_000)
    outcome = owner(case.state, case.plan, case.evidence)
    assert outcome.status == "candidate_ready", outcome.termination
    view = next(message.content["read_budget"] for message in case.requests[0].messages
        if isinstance(message.content, dict) and "read_budget" in message.content)
    assert view["used"] == 100 and view["limit"] is view["remaining"] is None
    record = json.loads(Path(outcome.execution_ref["path"]).read_bytes())
    assert record["tool_budget"]["reads"] == 100
    assert record["tool_budget"]["observation_bytes"] > 1_000_000
    owner.validate_handoff(outcome, case.context)


def test_plain_success_text_and_passed_check_without_finish_cannot_be_handed_off(case):
    case.mode["finish"] = False
    owner, outcome = execute(case)
    assert outcome.status == "blocked" and outcome.termination["code"] == "TASK_AGENT_FINISH_TOOL_REQUIRED"
    assert outcome.candidate_ref and outcome.check_ref and outcome.finish_response_ref is None
    with pytest.raises(ValueError, match="HANDOFF_NOT_READY"):
        owner.validate_handoff(outcome, case.context)


@pytest.mark.parametrize("error", [TimeoutError("controlled timeout"), RuntimeError("controlled provider failure")])
def test_provider_failure_retains_reservation_and_saved_failure_without_delivery(case, error):
    case.mode["provider_error"] = error
    owner, outcome = execute(case)
    assert outcome.status == "blocked" and outcome.termination["code"] == "TASK_AGENT_PROVIDER_TURN_FAILED"
    assert outcome.execution_ref and outcome.loop_result_ref and outcome.claim_ref
    assert outcome.budget_after["consumed"] == 4 and len(case.requests) == 1
    assert not case.evaluator.calls and outcome.candidate_ref is None
    with pytest.raises(ValueError, match="HANDOFF_NOT_READY"):
        owner.validate_handoff(outcome, case.context)


@pytest.mark.parametrize("metered", [False, True])
def test_expired_deadline_stops_before_new_reservation(case, metered):
    if metered:
        use_meter_only(case)
    case.kwargs["deadline"] = time.monotonic() - 1
    _, outcome = execute(case)
    assert outcome.status == "blocked" and outcome.termination["code"] == "TASK_AGENT_WALL_BUDGET_EXHAUSTED"
    assert outcome.budget_after == outcome.budget_before and not case.requests


@pytest.mark.parametrize("metered", [False, True])
def test_unknown_checker_outcome_is_retained_and_never_handed_off(case, monkeypatch, metered):
    if metered:
        use_meter_only(case)
    def broken(_self, _context):
        raise OSError("controlled checker failed after dispatch")

    monkeypatch.setattr(QualifiedPythonOracle, "__call__", broken)
    owner, outcome = execute(case)
    assert outcome.status == "unknown" and outcome.unknown_effects
    assert outcome.termination["code"] == "TASK_AGENT_ACTION_OUTCOME_UNKNOWN"
    assert outcome.candidate_ref and outcome.check_ref is None and outcome.execution_ref
    with pytest.raises(ValueError, match="HANDOFF_NOT_READY"):
        owner.validate_handoff(outcome, case.context)


def test_unknown_upstream_effect_survives_a_later_execution_recording_failure(case, monkeypatch):
    case.mode["upstream"] = True

    def broken_query(*_args):
        raise OSError("controlled failure after RepairPlan commit")

    case.kwargs["providers"]["evidence_pack_factory"].query_selector = broken_query
    owner = TaskAgentExecution(**case.kwargs)
    original_save = owner._save
    failures = []

    def save(name, value):
        if name == "execution.json" and not failures:
            failures.append(name)
            raise OSError("controlled first closeout write failure")
        return original_save(name, value)

    monkeypatch.setattr(owner, "_save", save)
    outcome = owner(case.state, case.plan, case.evidence)
    assert outcome.status == "unknown" and outcome.unknown_effects and outcome.execution_ref
    assert outcome.termination["prior_loop_termination"]["code"] == "TASK_AGENT_ACTION_OUTCOME_UNKNOWN"
    assert json.loads(Path(outcome.loop_result_ref["path"]).read_bytes())["unknown_effects"] is True
    assert len(case.inputs.store.list_feature_artifact_revisions(case.context.feature_spec_ref.feature_id, "repair_plan")) == 1
    assert len(case.requests) == 1 and not case.evaluator.calls
    with pytest.raises(ValueError, match="HANDOFF_NOT_READY"):
        owner.validate_handoff(outcome, case.context)


@pytest.mark.parametrize("changed", ["old_plan", "candidate", "check", "finish", "budget", "scope", "worker", "source"])
def test_handoff_rejects_changed_context_candidate_checker_response_or_authority(case, changed):
    owner, outcome = execute(case)
    assert outcome.status == "candidate_ready", outcome.termination
    expected = case.context
    if changed == "old_plan":
        expected = replace(expected, implementation_plan_ref=replace(expected.implementation_plan_ref, artifact_hash="f" * 32))
    elif changed in {"candidate", "check", "finish"}:
        reference = getattr(outcome, {"candidate": "candidate_ref", "check": "check_ref", "finish": "finish_response_ref"}[changed])
        Path(reference["path"]).write_text('{"status":"passed"}', encoding="utf-8")
    elif changed == "budget":
        outcome = replace(outcome, budget_after={**outcome.budget_after, "consumed": 3})
    elif changed == "scope":
        owner._allowed_edit_paths = ("other.cs",)
    elif changed == "worker":
        Path(owner.worker_configuration_ref["path"]).write_text("{}", encoding="utf-8")
    else:
        (case.inputs.workspace / "generated_mods/SyntheticMod/Behavior.cs").write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError):
        owner.validate_handoff(outcome, expected)


def test_store_binding_is_once_same_root_and_before_execution(case):
    owner = TaskAgentExecution(**case.kwargs)
    store = case.inputs.store

    class Recorder:
        root = store.root

        def __getattr__(self, name):
            return getattr(store, name)

        def task_agent_store_attempts(self, selected):
            assert selected is owner
            return []

    with pytest.raises(ValueError, match="STORE_BINDING"):
        owner.bind_product_store(SimpleNamespace(root=store.root / "other"))
    owner.bind_product_store(Recorder())
    with pytest.raises(ValueError, match="STORE_BINDING"):
        owner.bind_product_store(Recorder())
    outcome = owner(case.state, case.plan, case.evidence)
    assert outcome.status == "candidate_ready", outcome.termination
    with pytest.raises(ValueError, match="STORE_BINDING"):
        owner.bind_product_store(Recorder())


@pytest.mark.parametrize("changed", ["missing_check_call", "wrong_claim_request", "missing_claim"])
def test_self_consistent_saved_result_still_requires_actual_check_call_and_bound_claim(case, changed):
    owner, outcome = execute(case)
    assert outcome.status == "candidate_ready", outcome.termination
    document = json.loads(Path(outcome.execution_ref["path"]).read_bytes())
    if changed == "missing_check_call":
        document["completed_tool_records"] = [row for row in document["completed_tool_records"] if row["role"] != "candidate_check"]
    elif changed == "wrong_claim_request":
        claim = json.loads(Path(outcome.claim_ref["path"]).read_bytes())
        claim["request_id"] = "another-request"
        path = Path(outcome.claim_ref["path"]).with_name("changed-claim.json")
        path.write_text(json.dumps(claim), encoding="utf-8")
        document["claim_ref"] = file_ref(path)
        outcome = replace(outcome, claim_ref=document["claim_ref"])
    else:
        document["claim_ref"] = {"path": str(Path(outcome.claim_ref["path"]).with_name("missing-claim.json")), "sha256": "a" * 64}
        outcome = replace(outcome, claim_ref=document["claim_ref"])
    selected = Path(outcome.execution_ref["path"]).with_name("changed-execution.json")
    selected.write_text(json.dumps(document), encoding="utf-8")
    outcome = replace(outcome, execution_ref=file_ref(selected))
    with pytest.raises((ValueError, OSError)):
        owner.validate_handoff(outcome, case.context)


def test_nonlocal_provider_and_payload_drift_fail_closed(case):
    with pytest.raises(ValueError, match="LOCAL_OLLAMA_REQUIRED"):
        TaskAgentExecution(**{**case.kwargs, "profile": SimpleNamespace(provider="external")})
    owner = TaskAgentExecution(**case.kwargs)
    result = owner(case.state, {**case.plan, "invented": True}, case.evidence)
    assert result.status == "failed" and result.execution_ref and not case.requests
    saved = json.loads(Path(result.execution_ref["path"]).read_bytes())
    assert saved["run_id"] == case.state["run_id"] and saved["request_id"] == case.state["request_id"]
    with pytest.raises(ValueError, match="HANDOFF_NOT_READY"):
        owner.validate_handoff(result, case.context)


def test_existing_execution_directory_is_not_overwritten_and_failure_receipt_is_available(case):
    root = case.kwargs["output_root"]
    root.mkdir()
    retained = root / "execution.json"
    retained.write_text("old retained attempt", encoding="utf-8")
    owner = TaskAgentExecution(**case.kwargs)
    with pytest.raises(FileExistsError):
        owner(case.state, case.plan, case.evidence)
    assert retained.read_text(encoding="utf-8") == "old retained attempt"
    receipt = owner.execution_receipt()
    assert receipt.status == "failed" and receipt.execution_ref is None and receipt.claim_ref is None
    assert not case.requests and not case.evaluator.calls
