"""Normal adapter/owner calls over a controlled admitted prefix, without a provider.

Checkpoint/verifier provenance validation has separate tests. Their admitted
records are substituted; source inspection, generation, verification and budgets run.
"""
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.libs.llm.agent_turn import AgentMessage, AgentToolCall, AgentTurnProviderError
from src.mod_agent import comparison_checkpoint
from src.mod_agent.local_entry import LocalEntryError
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.proposal.failure_state import failure_signature, source_fact_hashes
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
from src.mod_agent.request_patch_execution import RequestPatchExecution
from tests.unit import test_fresh_planning_continuation as fixture
from tests.unit.test_patch_behavior_feedback import _adapter
from tests.unit.test_patch_recovery_worker import _DeclaredOracle


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    # Scope: this caller test uses only scripted responses and a synthetic checker.
    # 为什么要有它：调用方验证不能触发模型、Build或把受控响应算成真实实验。
    def forbidden(*args, **kwargs):
        raise AssertionError("No network or real provider is permitted")
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


class PrefixClient(fixture._PatchAgentTurnClient):
    def complete_turn(self, request):
        if len(self.requests) == 2:
            self.pending = request
            raise AgentTurnProviderError("Controlled stop before the third response")
        response = super().complete_turn(request)
        if response.content:
            self.failed_response = response
        return response


class SuffixClient(fixture._PatchAgentTurnClient):
    def __init__(self, response, *, passes):
        super().__init__()
        self.response = replace(response, content=response.content.replace(
            "RecordRuntimeMarker", "RecordRepairMarker")) if passes else response

    def complete_turn(self, request):
        self.requests.append(request)
        return self.response


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    inputs = fixture._inputs(tmp_path)
    selected = fixture._patch_request(inputs)
    state = fixture._direct_graph_state(inputs)
    state.update(current_stage="implementation_plan", next_stage="patch_proposal")
    state["artifacts"]["implementation_plan"] = selected.implementation_plan_ref.path
    state["artifact_hashes"]["implementation_plan"] = selected.implementation_plan_ref.artifact_hash
    plan = inputs.store.read(selected.implementation_plan_ref)["data"]
    evidence = inputs.store.read(selected.evidence_pack_ref)["data"]
    oracle = _DeclaredOracle(tmp_path / "checker")
    client = PrefixClient()
    adapter = _adapter(inputs, client, oracle)
    policy = PatchAgentLoopPolicy(max_total_model_turns=15, max_tool_rounds=6,
        max_observation_bytes=32768, require_source_inspection=False,
        finalize_after_first_observation=False, enable_supporting_reads=True,
        allow_upstream_request=True,
        model_call_budget_policy="task_total_only")
    adapter.patch_agent_loop_policy = policy
    adapter.enable_recovery_loop = True
    events = []
    adapter.bind_attempt_recorder(events.append)
    with pytest.raises(LocalEntryError) as stopped:
        adapter(state, plan, evidence)
    receipt = adapter.patch_attempt_receipt()
    assert len(client.requests) == 2, str(stopped.value)
    assert len(receipt["patch_agent_loop"]["behavior_evaluations"]) == 1
    failure = receipt["patch_agent_loop"]["behavior_evaluations"][0]
    facts = tuple(source_fact_hashes(receipt["patch_agent_loop"]["observation_trace"]))
    pending = client.pending
    assert pending.messages[-1].content["message_type"] == "patch_agent_loop_state"
    retained = replace(pending, messages=(
        AgentMessage("system", "Controlled historical system contract"),
        AgentMessage("user", "Controlled historical user contract"),
        *pending.messages[2:-1]))
    # Scope: 2/4/16214 are admitted controlled offsets, not claimed measurements of
    # this one-read synthetic prefix. Exact C5 prefix admission is tested separately.
    # 为什么要有它：消费者必须保留核准偏移，不能把本测试夹具伪报为C5原运行。
    checked = SimpleNamespace(request=retained, worker_policy=policy, history=deepcopy(receipt),
        consumed_model_turns=2, consumed_tool_rounds=4, consumed_observation_bytes=16214,
        product_consumed=5, product_remaining=13,
        known_code=deepcopy(failure["materialized_code_identity"]), known_failure=deepcopy(failure),
        no_progress_seed={failure_signature(failure, facts): 1}, source_facts=facts,
        expected_model_identity=receipt["model_identity"],
        provenance={"new_trial_not_original_resume": True, "approval_inherited": False,
            "current_artifacts": {kind: asdict(getattr(selected, kind + "_ref"))
                for kind in ("feature_spec", "evidence_pack", "implementation_plan")}})
    calls = []
    def admit(**kwargs):
        calls.append(kwargs)
        return checked
    monkeypatch.setattr(comparison_checkpoint, "admit_comparison_checkpoint", admit)
    def admit_verifier(selected, evaluator):
        assert selected is checked
        if evaluator.verification_identity() != checked.known_failure["verification_identity"]:
            raise comparison_checkpoint.ComparisonCheckpointError("CONTROLLED_VERIFIER_CHANGED")
        return {"verification_identity": evaluator.verification_identity(),
            "no_progress_seed": checked.no_progress_seed, "known_failure_projection": checked.known_failure,
            "provenance": {"scope": "controlled verifier admission, not C5 component qualification"}}
    monkeypatch.setattr(comparison_checkpoint, "admit_checkpoint_verifier", admit_verifier)
    arguments = dict(fixture_ref={"path": "controlled-admission", "sha256": "a" * 64},
        trial_id="controlled_new_trial", feature_spec_ref=selected.feature_spec_ref,
        evidence_pack_ref=selected.evidence_pack_ref, implementation_plan_ref=selected.implementation_plan_ref,
        worker_configuration_ref={"path": "controlled-worker", "sha256": "b" * 64})
    state["run_id"] += "_new_trial"
    return SimpleNamespace(inputs=inputs, state=state, plan=plan, evidence=evidence,
        checked=checked, arguments=arguments, oracle=oracle, response=client.failed_response,
        admission_calls=calls, before=fixture._source_hashes(inputs.workspace))


def owner_for(case, *, passes):
    client = SuffixClient(case.response, passes=passes)
    workers = []
    budget = {"policy": "task_total_only", "maximum": 18, "consumed": 5, "remaining": 13,
        "count_basis": "reserved_provider_invocations_not_transport_proof",
        "by_stage": {"planning": {"consumed": 3}, "initial_patch": {"consumed": 2},
                     "repair": {"consumed": 0}}}
    template = _adapter(case.inputs, client, case.oracle)
    def reserve(role):
        assert budget["remaining"] > 0
        budget["consumed"] += 1
        budget["remaining"] -= 1
        budget["by_stage"][role]["consumed"] += 1
    def factory(**kwargs):
        worker = _adapter(case.inputs, client, case.oracle)
        worker.patch_agent_loop_policy = case.checked.worker_policy
        worker.enable_recovery_loop = True
        worker.bind_comparison_checkpoint(**case.arguments)
        workers.append(worker)
        return worker
    owner = RequestPatchExecution(artifact_store=case.inputs.store,
        workspace_root=case.inputs.workspace, profile=template.profile,
        identity_reader=template.identity_reader, policy=case.checked.worker_policy,
        reserve_model_call=reserve, budget_snapshot=lambda: deepcopy(budget),
        behavior_evaluator=case.oracle, adapter_factory=factory,
        invocation_namespace="controlled_checkpoint")
    # Scope: trusted experiment harness; production's default remains zero.
    # 为什么要有它：后续产品Graph上游恢复只累计新增观察，初始核准成本须在唯一owner登记一次。
    owner._consumed_reads = case.checked.consumed_tool_rounds
    owner._consumed_bytes = case.checked.consumed_observation_bytes
    return owner, client, workers, budget


def test_normal_product_owner_keeps_prefix_budget_and_does_not_repeat_feedback(admitted):
    case = admitted
    owner, client, workers, budget = owner_for(case, passes=True)
    changes = owner(case.state, case.plan, case.evidence)
    worker = workers[0]
    receipt = worker.patch_attempt_receipt()
    assert changes and len(client.requests) == receipt["model_call_count"] == 1
    assert len(case.admission_calls) == 2  # binding and actual invocation recheck
    assert worker._prior_worker_budget == {"consumed_model_turns": 2,
        "consumed_tool_rounds": 4, "consumed_observation_bytes": 16214}
    assert worker._recovery_observation_bytes == 0
    assert (owner._consumed_reads, owner._consumed_bytes) == (4, 16214)
    assert (budget["consumed"], budget["remaining"]) == (6, 12)
    actual = client.requests[0]
    assert actual.messages[:2] != case.checked.request.messages[:2]
    assert actual.messages[2:-1] == case.checked.request.messages[2:]
    assert actual.tool_results == case.checked.request.tool_results
    assert actual.historical_tool_exchanges == case.checked.request.historical_tool_exchanges
    content = [row.content for row in actual.messages if isinstance(row.content, dict)]
    assert sum(row.get("message_type") == "patch_behavior_correction" for row in content) == 1
    assert not any(row.get("message_type") == "patch_same_plan_assessment" for row in content)
    assert content[-1]["budget_state"]["model_turns_used"] == 2
    assert content[-1]["budget_state"]["tool_rounds_used"] == 4
    assert content[-1]["budget_state"]["observation_bytes_used"] == 16214
    assert fixture._source_hashes(case.inputs.workspace) == case.before


def test_inherited_failure_and_source_facts_stop_after_two_new_identical_failures(admitted):
    case = admitted
    owner, client, workers, budget = owner_for(case, passes=False)
    with pytest.raises(LocalEntryError, match="PATCH_BEHAVIOR_NO_PROGRESS"):
        owner(case.state, case.plan, case.evidence)
    receipt = workers[0].patch_attempt_receipt()
    termination = receipt["patch_agent_loop"]["behavior_repair_termination"]
    assert termination["same_state_observations"] == 3
    assert termination["failure_signature_sha256"] in case.checked.no_progress_seed
    assert len(client.requests) == receipt["model_call_count"] == 2
    assert len(receipt["patch_agent_loop"]["behavior_evaluations"]) == 2
    assert (budget["consumed"], budget["remaining"]) == (7, 11)
    assert owner._consumed_reads == 4 and owner._consumed_bytes > 16214
    assert fixture._source_hashes(case.inputs.workspace) == case.before


@pytest.mark.parametrize("change", ["model", "verifier"])
def test_actual_current_identity_must_match_admitted_checkpoint_before_new_send(admitted, change):
    case = admitted
    if change == "model":
        case.checked.expected_model_identity = {**case.checked.expected_model_identity, "digest": "f" * 64}
    else:
        case.oracle.environment += 1
    owner, client, workers, budget = owner_for(case, passes=True)
    with pytest.raises((LocalEntryError, comparison_checkpoint.ComparisonCheckpointError)):
        owner(case.state, case.plan, case.evidence)
    assert not client.requests
    assert budget["consumed"] == 5
    assert workers[0].patch_attempt_receipt()["model_call_count"] == 0
    assert fixture._source_hashes(case.inputs.workspace) == case.before


@pytest.mark.parametrize("remove_feedback", [False, True])
def test_immediate_upstream_keeps_admitted_old_failure_without_new_verification(admitted, remove_feedback):
    case = admitted
    owner, client, workers, budget = owner_for(case, passes=True)
    arguments = {"required_owner": "planning", "reason": "Reassess the observed failure against the original scope.",
                 "source_refs": ["binding://edit_binding_001"]}
    client.response = replace(case.response, content=None,
        tool_calls=(AgentToolCall("diagnostic-upstream", "request_patch_upstream", arguments),),
        provider_state={"content": None, "tool_calls": [{"id": "diagnostic-upstream"}]})
    owner.bind_upstream_effects({"planning": "assess_and_continue", "evidence": "assess_and_continue"})
    with pytest.raises(LocalEntryError, match="PATCH_AGENT_UPSTREAM_REQUESTED"):
        owner(case.state, case.plan, case.evidence)
    prior_checks = len(case.oracle.contexts)
    if remove_feedback:
        # Scope: only this test's new request/journal, never the retained prefix.
        # 为什么要有它：即使新请求及计数字段内部自洽，缺失实际F1也不能借旧报告通过上游准入。
        worker = workers[0]
        turn = worker._receipt["worker_turns"][-1]
        path = Path(turn["request_ref"]["path"])
        request = json.loads(path.read_bytes())
        request["messages"] = [row for row in request["messages"] if not (
            isinstance(row["content"], dict) and row["content"].get("message_type") == "patch_behavior_correction")]
        raw = json.dumps(request).encode()
        path.write_bytes(raw)
        turn["request_ref"]["sha256"] = hashlib.sha256(raw).hexdigest()
        turn["request_sha256"] = canonical_hash(request)
        owner.receipts[-1] = worker.patch_attempt_receipt()
        journal_path = Path(owner.journals[-1].path)
        journal = json.loads(journal_path.read_bytes())
        journal["factory_receipt"] = owner.receipts[-1]
        journal_path.write_text(json.dumps(journal), encoding="utf-8")
        from src.mod_agent.fresh_planning_continuation import PatchContinuationError
        with pytest.raises(PatchContinuationError):
            owner.validate_upstream(case.state, case.inputs.store)
        assert len(case.oracle.contexts) == prior_checks
        return
    context, upstream = owner.validate_upstream(case.state, case.inputs.store)
    assert context["previous_response"] == case.response.content
    assert context["failure_report_ref"] == case.checked.known_failure["report_ref"]
    assert context["generated_changes_sha256"] == case.checked.known_failure["generated_changes_sha256"]
    assert context["current_failure_available"] is True
    assert context["current_segment_failure_executed"] is False
    assert context["failure_origin"] == "verified_historical_comparison_prefix"
    assert context["approval_inherited"] is False
    assert upstream["failure_report_ref"] == context["failure_report_ref"]
    assert len(case.oracle.contexts) == prior_checks
    assert not workers[0].patch_attempt_receipt()["patch_agent_loop"].get("behavior_evaluations")
    assert (budget["consumed"], budget["remaining"]) == (6, 12)
