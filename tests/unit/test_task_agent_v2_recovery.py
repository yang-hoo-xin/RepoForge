"""Resume real persisted owners and the shared loop without redoing completed tools.

为什么要有它：恢复测试要跨新owner与磁盘原件，不能只证明字典能序列化。
"""
import json
import time
from copy import deepcopy
from pathlib import Path

import pytest

from src.libs.llm.agent_turn import AgentMessage, AgentTurnRequest
from src.mod_agent.interrupted_patch_admission import retained_request_from_document
from src.mod_agent.task_agent_v2_recovery import (
    TaskAgentRecoveryJournal,
    load_task_agent_checkpoint,
    resume_v2_task_agent,
    single_task_lock,
)
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)
from tests.unit.test_task_agent_loop import Budget, Client, Owners, call, response, run
from tests.unit.test_task_agent_v2_delivery import delivery_case as delivery_case


@pytest.mark.parametrize("phase", ["response_ready", "dispatch_ready", "observations_charged"])
def test_loop_resumes_received_response_and_completed_tool_without_new_cost(phase):
    owners = Owners(Budget())
    retained = []
    initial = AgentTurnRequest((AgentMessage("system", "Tools are data"), AgentMessage("user", "Task")))

    def interrupt(value):
        if value["phase"] == phase and value["model_turns"] == 1:
            if phase == "dispatch_ready" and not value["call_records"].get("submit", {}).get("status") == "completed":
                return
            retained.append(deepcopy(value))
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(Client([response(call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "submit"))]),
            owners, initial_request=initial, checkpoint_callback=interrupt)
    snapshot = json.loads(json.dumps(retained[-1]))
    result = run(Client([response(call("candidate_check", {"candidate_id": "c1"}, "check")),
        response(call("finish", {"candidate_id": "c1"}, "finish"))]), owners,
        initial_request=retained_request_from_document(snapshot["request"]), resume_checkpoint=snapshot)
    assert result.status == "completed", result.termination
    assert owners.budget.consumed == result.model_turns == 3
    assert [role for role, _args in owners.effects] == ["candidate_submit", "candidate_check", "finish"]
    assert result.observation_bytes == owners.budget.bytes


def test_loop_unknown_pending_is_not_reissued():
    owners = Owners(Budget())
    retained = []
    initial = AgentTurnRequest((AgentMessage("system", "Tools are data"), AgentMessage("user", "Task")))

    def interrupt(value):
        if value["phase"] == "tool_pending":
            retained.append(deepcopy(value))
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(Client([response(call("candidate_submit", {"candidate_id": "c1", "fixed": True}, "submit"))]),
            owners, initial_request=initial, checkpoint_callback=interrupt)
    snapshot = retained[-1]
    client = Client([])
    result = run(client, owners, initial_request=retained_request_from_document(snapshot["request"]), resume_checkpoint=snapshot)
    assert result.status == "blocked" and not client.requests and not owners.effects
    assert owners.budget.consumed == 1


def test_normal_cli_semantic_provider_receipt_recovery_rejects_forgery_and_charges_once(
    repository_workspace, monkeypatch, capsys,
):
    from src.mod_agent import cli
    from src.mod_agent.model_qualification import canonical_hash
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
    from tests.unit.test_continuous_repair_product_entry import (
        test_normal_cli_no_plan_or_binding_gate_retains_observation_and_repairs,
    )

    observed_turns, retained = [], []

    def semantic_observer(*, arguments, current, turn, **_kwargs):
        observed_turns.append(turn)
        selected = next(row for row in current["semantic_evidence_catalog"] if row["kind"] == "task_requirements")
        if turn > 1:
            assert current["semantic_work_state"]["latest_update"]["status"] in {"accepted", "reaffirmed"}
        evidence = [{"id": selected["id"], "quote": "Arithmetic.Double"}]
        return {**arguments, "working_state": {
            "observations": [{"statement": "The original task describes Arithmetic.Double", "evidence": evidence}],
            "hypotheses": [{"id": "h1", "statement": "The implementation needs checking", "status": "open", "evidence": []}],
            "open_questions": ["Which observation will test this hypothesis?"],
            "next_action": {"purpose": "Continue the recorded task", "expected_observation": "A current source or public check result",
                "decision_if_observed": "Choose the next action from the actual result"}}}

    original_save = TaskAgentExecutionV2._save

    def interrupt(owner, name, payload):
        reference = original_save(owner, name, payload)
        if name.startswith("provider-outcome-") and payload.get("status") == "responded" and payload["turn"] == 3 and not retained:
            retained.append((owner.store.root, owner.v2_checkpoint_ref, owner.budget_snapshot()["consumed"]))
            raise KeyboardInterrupt()
        return reference

    monkeypatch.setattr(TaskAgentExecutionV2, "_save", interrupt)
    with pytest.raises(KeyboardInterrupt):
        test_normal_cli_no_plan_or_binding_gate_retains_observation_and_repairs(
            repository_workspace, monkeypatch, capsys, semantic_observer=semantic_observer)
    capsys.readouterr()
    monkeypatch.setattr(TaskAgentExecutionV2, "_save", original_save)
    store, reference, consumed = retained[0]
    assert consumed == 3 and observed_turns == [1, 2, 3]

    # 为什么要有它：已收到的第三响应只重建一次；有真词句的伪造状态也不能借恢复进入实际请求。
    loaded = load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=reference, reconcile=True)
    good_ref, checkpoint, task = loaded["checkpoint_ref"], loaded["checkpoint"], loaded["state"]
    assert loaded["host"]["semantic_state_enabled"] is True
    semantic = checkpoint["owner"]["working_state"]["semantic_state"]
    assert len(semantic["events"]) == 3 and len(semantic["snapshots"]) == 1
    assert len(semantic["snapshots"][0]["occurrences"]) == 3
    assert observed_turns == [1, 2, 3]
    forged = deepcopy(checkpoint)
    forged["owner"]["working_state"]["semantic_state"]["snapshots"][0]["state"]["hypotheses"][0]["status"] = "refuted"
    forged_ref = task.save("session-checkpoint", forged)
    pointer = task.root / "session-checkpoints" / (canonical_hash(task.contract_ref) + ".json")
    original_pointer = pointer.read_bytes()
    pointer.write_text(json.dumps({"checkpoint_ref": forged_ref, "host_config_ref": checkpoint["host_config_ref"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="SEMANTIC_STATE_NOT_OBSERVED"):
        load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=forged_ref)
    pointer.write_bytes(original_pointer)
    assert observed_turns == [1, 2, 3]

    code = cli.main(["run-request", "--resume-task-agent", good_ref, "--artifact-store-root", str(store)])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["status"] == "awaiting_delivery", result
    assert result["new_task_created"] is False
    assert observed_turns == list(range(1, 11))
    assert result["cumulative_resources"]["model_calls"] == 10
    final = load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=result["v2_checkpoint_ref"])
    assert len(final["checkpoint"]["owner"]["working_state"]["semantic_state"]["events"]) == 10


def test_os_lock_excludes_concurrent_owner_but_releases_without_deleting_file(tmp_path):
    with single_task_lock(tmp_path, "contract"):
        with pytest.raises(ValueError, match="SESSION_ALREADY_RUNNING"):
            with single_task_lock(tmp_path, "contract"):
                pytest.fail("second owner acquired active lock")
    with single_task_lock(tmp_path, "contract"):
        assert list((tmp_path / "v2/session-locks").glob("*.lock"))


@pytest.mark.parametrize("boundary", ["checkpoint_completed", "tool_receipt", "plan_receipt", "provider_receipt", "read_charge", "observation_charge", "resource_ledger", "history_receipt", "projected_provider_receipt", "candidate_not_ready", "multiple_safe_corrections", "stable_id_correction"])
def test_normal_entry_interruption_reconstructs_owner_and_finishes_same_product(repository_workspace, monkeypatch, capsys, boundary,
                                                                               candidate_boundary=None):
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
    from src.mod_agent.task_agent_v2_state import load_v2_artifact
    from tests.unit.test_task_agent_v2_product_entry import (
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff,
    )

    original_save = TaskAgentRecoveryJournal.save
    retained = []
    state_views = []
    original_prepare = TaskAgentExecutionV2.prepare_decision_request

    def inspect_current_state(owner, request, session):
        prepared = original_prepare(owner, request, session)
        facts = next(message.content["current_task_state"] for message in prepared.messages
            if isinstance(message.content, dict) and "current_task_state" in message.content)
        assert facts["current_revision_ref"] == session.state.current_revision_ref
        assert facts["current_candidate_ref"] == session.state.current_candidate_ref
        assert facts["binding_registry_ref"] == session.state.registry_ref
        assert facts["current_check_ref"] == session.current_check
        definition = next(tool for tool in prepared.tools if tool.name == "candidate_submit")
        assert definition.parameters["properties"]["parent_revision_ref"]["enum"] == [session.state.current_revision_ref]
        state_views.append(deepcopy(facts))
        return prepared

    monkeypatch.setattr(TaskAgentExecutionV2, "prepare_decision_request", inspect_current_state)

    def interrupt(journal, value):
        ref = original_save(journal, value)
        if boundary == "multiple_safe_corrections" and value["phase"] == "ready" and not retained:
            corrections = [row for row in value["call_records"].values()
                if row.get("result", {}).get("output", {}).get("code") == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"]
            if len(corrections) == 2:
                assert all(row["status"] == "not_dispatched" for row in corrections)
                retained.append((journal.state.store_root, ref, value["model_budget"]["consumed"]))
                raise KeyboardInterrupt()
        if boundary == "stable_id_correction" and value["phase"] == "ready" and not retained:
            records = value["call_records"]
            if all(records.get(call_id, {}).get("status") == "completed"
                    for call_id in ("stable-plan", "stable-read")):
                retained.append((journal.state.store_root, ref, value["model_budget"]["consumed"]))
                raise KeyboardInterrupt()
        if boundary == "candidate_not_ready" and value["phase"] == "ready" and not retained:
            record = value["call_records"].get("not-ready-candidate_check", {})
            if record.get("result", {}).get("output", {}).get("status") == "not_ready":
                assert value["tool_argument_correction_used"] is True
                assert not journal.session.public_checks
                retained.append((journal.state.store_root, ref, value["model_budget"]["consumed"]))
                raise KeyboardInterrupt()
        if boundary == "checkpoint_completed" and value["phase"] == "dispatch_ready" and journal.session.current_check and not retained:
            retained.append((journal.state.store_root, ref, value["model_budget"]["consumed"]))
            raise KeyboardInterrupt()
        return ref

    monkeypatch.setattr(TaskAgentRecoveryJournal, "save", interrupt)
    original_tool = TaskAgentExecutionV2.record_v2_tool_receipt
    original_output = TaskAgentExecutionV2._save

    def interrupt_tool(owner, origin, output, session):
        reference = original_tool(owner, origin, output, session)
        if ((boundary == "tool_receipt" and origin["name"] == "candidate_check"
                or boundary == "plan_receipt" and origin["name"] == "plan_update"
                or boundary == "history_receipt" and origin["name"] == "history_read") and not retained):
            retained.append((owner.store.root, owner.v2_checkpoint_ref, owner.budget_snapshot()["consumed"]))
            raise KeyboardInterrupt()
        return reference

    def interrupt_provider(owner, name, payload):
        ref = original_output(owner, name, payload)
        if (boundary == "projected_provider_receipt" and name.startswith("provider-outcome-")
                and payload.get("status") == "responded" and not retained):
            from src.mod_agent.model_qualification import read_bound_json
            actual = read_bound_json(payload["request_ref"])
            current = next(row["content"]["current_task_state"] for row in actual["messages"]
                if isinstance(row.get("content"), dict) and "current_task_state" in row["content"])
            if (current.get("history_projection") or {}).get("history_ref"):
                retained.append((owner.store.root, owner.v2_checkpoint_ref, owner.budget_snapshot()["consumed"]))
                raise KeyboardInterrupt()
        if boundary == "provider_receipt" and name.startswith("provider-outcome-") and payload["turn"] == 5 and not retained:
            retained.append((owner.store.root, owner.v2_checkpoint_ref, owner.budget_snapshot()["consumed"]))
            raise KeyboardInterrupt()
        if boundary in {"read_charge", "observation_charge"} and name.startswith("resource-tools-") and not retained:
            if payload["reservation"]["turn"] == 5 and bool(payload["usage"]["observation_bytes"]) == (boundary == "observation_charge"):
                retained.append((owner.store.root, owner.v2_checkpoint_ref, owner.budget_snapshot()["consumed"]))
                raise KeyboardInterrupt()
        return ref

    monkeypatch.setattr(TaskAgentExecutionV2, "record_v2_tool_receipt", interrupt_tool)
    monkeypatch.setattr(TaskAgentExecutionV2, "_save", interrupt_provider)
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State
    original_resource = TaskAgentV2State.record_resources

    def interrupt_resource(task, usage, operation_id):
        result = original_resource(task, usage, operation_id)
        if boundary == "resource_ledger" and ":tools:" in operation_id and not retained:
            reservations = sum(key.startswith("model-reservation:") for key in task.head["operations"])
            if reservations == 5:
                latest = task.root / "session-checkpoints" / (recovery_hash(task.contract_ref) + ".json")
                retained.append((task.store_root, json.loads(latest.read_bytes())["checkpoint_ref"], 5))
                raise KeyboardInterrupt()
        return result

    from src.mod_agent.model_qualification import canonical_hash as recovery_hash
    monkeypatch.setattr(TaskAgentV2State, "record_resources", interrupt_resource)
    with pytest.raises(KeyboardInterrupt):
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff(repository_workspace, monkeypatch, capsys, False,
            exercise_paging=boundary in {"history_receipt", "projected_provider_receipt", "multiple_safe_corrections"},
            exercise_history=boundary in {"history_receipt", "projected_provider_receipt"},
            exercise_candidate_readiness=boundary == "candidate_not_ready",
            exercise_multiple_corrections=boundary == "multiple_safe_corrections",
            exercise_stable_call_ids=boundary == "stable_id_correction")
    capsys.readouterr()
    monkeypatch.setattr(TaskAgentRecoveryJournal, "save", original_save)
    monkeypatch.setattr(TaskAgentExecutionV2, "record_v2_tool_receipt", original_tool)
    monkeypatch.setattr(TaskAgentExecutionV2, "_save", original_output)
    monkeypatch.setattr(TaskAgentV2State, "record_resources", original_resource)
    store, reference, before = retained[0]
    saved = load_v2_artifact(store, reference)
    host = load_v2_artifact(store, saved["host_config_ref"])
    if candidate_boundary is not None:
        import hashlib

        from src.mod_agent.model_qualification import read_bound_json

        outcome_path = Path(saved["owner"]["output_root"]) / (
            "provider-outcome-" + saved["owner"]["reservation"]["reservation_id"] + ".json")
        outcome = json.loads(outcome_path.read_bytes())
        actual = read_bound_json(outcome["request_ref"])
        current = next(row["content"]["current_task_state"] for row in actual["messages"]
            if isinstance(row.get("content"), dict) and "current_task_state" in row["content"])
        assert current["current_candidate_ref"] == current["current_revision_ref"]
        assert current["current_candidate_ref"] != current["baseline_revision_ref"]
        if candidate_boundary == "legacy":
            del current["current_candidate_ref"]
        else:
            assert candidate_boundary == "forged"
            current["current_candidate_ref"] = current["baseline_revision_ref"]
        actual_path = Path(outcome["request_ref"]["path"])
        actual_path.write_text(json.dumps(actual, ensure_ascii=False), encoding="utf-8")
        outcome["request_ref"]["sha256"] = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        outcome_path.write_text(json.dumps(outcome, ensure_ascii=False), encoding="utf-8")
        if candidate_boundary == "forged":
            with pytest.raises(ValueError, match="PROVIDER_CANDIDATE_BOUNDARY_CHANGED"):
                load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=reference, reconcile=True)
            return
    if boundary == "checkpoint_completed":
        assert saved["session"]["public_checks"][0]["status"] == "behavior_failed"
        import subprocess
        import sys
        child = subprocess.run([sys.executable, "-B", "-c",
            "import json,sys; from src.mod_agent.task_agent_v2_recovery import load_task_agent_checkpoint; "
            "x=load_task_agent_checkpoint(artifact_store_root=sys.argv[1],checkpoint_ref=sys.argv[2]); "
            "print(json.dumps({'turns':x['checkpoint']['loop']['model_turns'],'contract':x['state'].contract_ref}))",
            str(store), reference], check=True, capture_output=True, text=True)
        assert json.loads(child.stdout) == {"turns": 5, "contract": saved["task_contract_ref"]}
    from src.mod_agent import cli
    manifest_path = store.parent / "run_manifest.json"
    original_manifest = json.loads(manifest_path.read_bytes())
    from src.mod_agent import task_agent_v2_recovery as recovery_owner
    original_reconcile = recovery_owner._reconcile_checkpoint

    def inspect_reconciled_working_boundary(task, checkpoint_ref, checkpoint, latest_path):
        ref, value = original_reconcile(task, checkpoint_ref, checkpoint, latest_path)
        if checkpoint["loop"]["phase"] == "provider_pending":
            from src.mod_agent.model_qualification import read_bound_json
            actual = read_bound_json(value["owner"]["reservation"]["request_ref"])
            current = next(row["content"]["current_task_state"] for row in actual["messages"]
                if isinstance(row.get("content"), dict) and "current_task_state" in row["content"])
            assert value["owner"]["working_state"]["boundary"] == {key: current[key] for key in (
                "task_contract_ref", "current_revision_ref", "binding_registry_ref", "current_check_ref")}
        return ref, value

    monkeypatch.setattr(recovery_owner, "_reconcile_checkpoint", inspect_reconciled_working_boundary)
    exit_code = cli.main(["run-request", "--resume-task-agent", reference, "--artifact-store-root", str(store)])
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0, result
    assert result["status"] == "awaiting_delivery", result.get("outcome", {}).get("termination", result)
    assert result["new_task_created"] is False
    from src.mod_agent.model_qualification import read_bound_json
    recovery_execution = read_bound_json(result["recovery_execution_ref"])
    assert recovery_execution["run_id"] == result["run_id"]
    assert recovery_execution["request_id"] == result["request_id"]
    assert recovery_execution["resumed_from_checkpoint_ref"] == result["resumed_from_checkpoint_ref"]
    assert recovery_execution["execution_trace"]["logical_request_id"] == result["request_id"]
    assert recovery_execution["execution_observations"]
    after_manifest = json.loads(manifest_path.read_bytes())
    for key in ("execution_trace", "execution_observations"):
        assert after_manifest[key] == original_manifest[key]
    resumed_native = read_bound_json(result["task_agent_outcome"]["execution_ref"])
    new_turns = [turn for turn in resumed_native["provider_turns"]
        if Path(turn["request_ref"]["path"]).parent == Path(result["task_agent_outcome"]["execution_ref"]["path"]).parent]
    assert new_turns
    assert recovery_execution["model_calls_performed"] == len(new_turns)
    for turn in new_turns:
        read_bound_json(turn["provider_request_ref"])
        read_bound_json(turn["response_ref"])
    assert result["request_id"] == host["graph_state"]["request_id"]
    assert result["run_id"] == host["graph_state"]["run_id"]
    expected_turns = (22 if boundary == "multiple_safe_corrections"
        else 14 if boundary == "stable_id_correction"
        else 20 if boundary in {"history_receipt", "projected_provider_receipt"}
        else 16 if boundary == "candidate_not_ready" else 13)
    assert result["cumulative_resources"]["model_calls"] == expected_turns
    if boundary == "multiple_safe_corrections":
        assert before == 8
    elif boundary == "stable_id_correction":
        assert before == 3
    else:
        assert before > 0 if boundary in {"history_receipt", "projected_provider_receipt"} else before == (1 if boundary == "plan_receipt" else 5)
    final = load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=result["v2_checkpoint_ref"])
    assert len(final["checkpoint"]["session"]["public_checks"]) == 2
    if boundary == "candidate_not_ready":
        # 为什么要有它：恢复须保留已消耗的参数纠错及未执行反馈，不能补做检查、清零成本或领取新纠错额度。
        assert final["checkpoint"]["loop"]["tool_argument_correction_used"] is True
        assert sum(row["name"] == "candidate_check" for row in final["checkpoint"]["details"]["completed_tool_records"]) == 2
        assert final["checkpoint"]["loop"]["call_records"]["not-ready-candidate_check"]["status"] == "not_dispatched"
    if boundary == "multiple_safe_corrections":
        loop = final["checkpoint"]["loop"]
        batches = [exchange["tool_results"] for exchange in loop["request"]["historical_tool_exchanges"]]
        batches.append(loop["request"]["tool_results"])
        corrections = [result for batch in batches for result in batch
            if result["output"].get("code") == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"]
        assert len(corrections) == 2 and all(result["output"]["status"] == "rejected_not_executed"
            for result in corrections)
        records = loop["call_records"]
        assert all(records[result["call_id"]]["status"] == "not_dispatched" for result in corrections)
        submitted = [row for row in final["checkpoint"]["session"]["actual_calls"]
            if row["name"] == "candidate_submit"]
        assert "source_view_ref" not in submitted[0]["arguments"]["edits"][0]
        assert submitted[1]["arguments"]["edits"][0]["source_view_ref"]
        completed_ids = {row["call_id"] for row in final["checkpoint"]["details"]["completed_tool_records"]}
        assert all(result["call_id"] not in completed_ids for result in corrections)
        forged = deepcopy(final["checkpoint"])
        first_rejected = next(row for row in forged["session"]["actual_calls"]
            if row["call_id"] == corrections[0]["call_id"])
        first_rejected["arguments"]["edits"][0]["source_view_ref"] = "forged-source-view"
        forged_ref = final["state"].save("session-checkpoint", forged)
        latest = final["state"].root / "session-checkpoints" / (recovery_hash(final["state"].contract_ref) + ".json")
        original_pointer = latest.read_bytes()
        latest.write_text(json.dumps({"checkpoint_ref": forged_ref,
            "host_config_ref": forged["host_config_ref"]}), encoding="utf-8")
        with pytest.raises(ValueError, match="CORRECTION_ARGUMENTS_CHANGED"):
            load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=forged_ref)
        latest.write_bytes(original_pointer)
    if boundary == "stable_id_correction":
        loop = final["checkpoint"]["loop"]
        batches = [exchange["tool_results"] for exchange in loop["request"]["historical_tool_exchanges"]]
        batches.append(loop["request"]["tool_results"])
        corrections = [result for batch in batches for result in batch
            if result["output"].get("code") == "TASK_AGENT_MIXED_TOOL_BATCH"]
        assert {result["call_id"] for result in corrections} == {"stable-plan", "stable-read"}
        assert all(loop["call_records"][result["call_id"]]["status"] == "completed" for result in corrections)
        origins = [row for row in final["checkpoint"]["session"]["actual_calls"]
            if row["call_id"] in {"stable-plan", "stable-read"}]
        assert {call_id: sum(row["call_id"] == call_id for row in origins)
            for call_id in ("stable-plan", "stable-read")} == {"stable-plan": 2, "stable-read": 2}
        completed = [row for row in final["checkpoint"]["details"]["completed_tool_records"]
            if row["call_id"] in {"stable-plan", "stable-read"}]
        assert {row["call_id"] for row in completed} == {"stable-plan", "stable-read"}
        assert len(completed) == 2
    assert final["checkpoint"]["loop"]["model_turns"] == expected_turns
    assert final["state"].contract_ref == saved["task_contract_ref"]
    assert len(state_views) == expected_turns
    assert state_views[0]["current_revision_ref"] == final["state"].contract["baseline_manifest_ref"]
    assert state_views[-1]["current_revision_ref"] == final["state"].current_revision_ref


@pytest.mark.parametrize("candidate_boundary", ["legacy", "forged"])
def test_provider_recovery_validates_optional_candidate_boundary(repository_workspace, monkeypatch, capsys, candidate_boundary):
    test_normal_entry_interruption_reconstructs_owner_and_finishes_same_product(
        repository_workspace, monkeypatch, capsys, "provider_receipt", candidate_boundary=candidate_boundary)


@pytest.mark.parametrize("storage_fails", [False, True])
def test_recovery_trace_preserves_original_exception_and_unknown_response(tmp_path, storage_fails):
    from types import SimpleNamespace

    from src.core.trace.execution_trace import current_trace
    from src.mod_agent.domain.full_project_compilation import _save
    from src.mod_agent.model_qualification import read_bound_json
    from src.mod_agent.task_agent_v2_recovery import _recovery_execution_trace

    def save(name, value):
        if storage_fails:
            raise OSError("controlled storage failure")
        return _save(tmp_path / name, value)

    owner = SimpleNamespace(store=SimpleNamespace(root=tmp_path), output_root=tmp_path,
        v2_checkpoint_ref="checkpoint-after", _receipt=None, _save=save)
    host = {"graph_state": {"run_id": "run", "request_id": "request", "feature_id": "feature"}}
    error = RuntimeError("controlled transport failure")
    with pytest.raises(RuntimeError) as caught:
        with _recovery_execution_trace(owner, host, "checkpoint-before") as projection:
            with current_trace().model_call({"provider": "controlled", "usage_available": False}) as call:
                call.transport_started()
                raise error
    assert caught.value is error
    if storage_fails:
        assert projection["recovery_execution_status"] == "unavailable"
        assert not hasattr(error, "recovery_execution_ref")
    else:
        record = read_bound_json(error.recovery_execution_ref)
        assert record["status"] == "failed" and record["error_type"] == "RuntimeError"
        assert record["model_calls_performed"] == 1
        generations = [row for row in record["execution_observations"].values() if row["kind"] == "generation"]
        assert len(generations) == 1 and generations[0]["status"] == "failed"
        assert "usage" not in generations[0]["attributes"]
    assert current_trace() is None


@pytest.mark.parametrize("drift", ["source", "worker", "deadline", "learning_helper"])
def test_saved_checkpoint_rejects_source_drift_before_model_send(repository_workspace, monkeypatch, capsys, drift):
    from src.mod_agent import task_agent_v2_recovery as recovery
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
    from tests.unit.test_task_agent_v2_product_entry import (
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff,
    )

    if drift == "deadline":
        original_init = TaskAgentExecutionV2.__init__
        def bounded(owner, **kwargs):
            original_init(owner, **kwargs)
            owner.deadline = recovery.time.monotonic() + 300
        monkeypatch.setattr(TaskAgentExecutionV2, "__init__", bounded)

    original = TaskAgentRecoveryJournal.save
    retained = []

    def interrupt(journal, value):
        ref = original(journal, value)
        if value["phase"] == "ready" and value["model_turns"] == 1:
            retained.append((journal.state.store_root, ref, journal.state))
            raise KeyboardInterrupt()
        return ref

    monkeypatch.setattr(TaskAgentRecoveryJournal, "save", interrupt)
    with pytest.raises(KeyboardInterrupt):
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff(repository_workspace, monkeypatch, capsys, False)
    capsys.readouterr()
    store, ref, state = retained[0]
    host = state.load(state.load(ref)["host_config_ref"])
    if drift == "source":
        source = state.load(state.contract["baseline_manifest_ref"])["source_root"]
        Path(source, "Library/Arithmetic.cs").write_bytes(b"changed after interruption")
    elif drift == "worker":
        worker = Path(host["worker_configuration_ref"]["path"])
        worker.write_bytes(worker.read_bytes() + b"\n")
    elif drift == "learning_helper":
        from src.mod_agent.planning import learning_request
        expected = Path(learning_request.__file__).resolve()
        assert any(Path(row["path"]).resolve() == expected for row in host["owner_refs"])
        original_ref = recovery._ref

        def changed_helper(path):
            value = original_ref(path)
            return {**value, "sha256": "0" * 64} if Path(path).resolve() == expected else value

        monkeypatch.setattr(recovery, "_ref", changed_helper)
    else:
        monkeypatch.setattr(recovery.time, "time", lambda: host["deadline_utc"] + 1)
    with pytest.raises(ValueError):
        resume_v2_task_agent(artifact_store_root=store, checkpoint_ref=ref)


def test_repeated_recovery_retains_earliest_deadline_without_new_reservations(repository_workspace, monkeypatch, capsys):
    from src.mod_agent.task_agent_v2_state import load_v2_artifact
    from tests.unit.test_task_agent_v2_product_entry import (
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff,
    )

    deadline = time.time() + 1800
    monkeypatch.setenv("REPOFORGE_TASK_DEADLINE_UTC", str(deadline))
    original = TaskAgentRecoveryJournal.save
    retained = []

    def interrupt(journal, value):
        ref = original(journal, value)
        if value["phase"] == "ready" and value["model_turns"] == 1:
            retained.append((journal.state.store_root, ref))
            raise KeyboardInterrupt()
        return ref

    monkeypatch.setattr(TaskAgentRecoveryJournal, "save", interrupt)
    with pytest.raises(KeyboardInterrupt):
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff(repository_workspace, monkeypatch, capsys, False)
    capsys.readouterr()
    for inherited, expected in ((deadline + 300, deadline), (deadline - 300, deadline - 300), (None, deadline - 300)):
        if inherited is None:
            monkeypatch.delenv("REPOFORGE_TASK_DEADLINE_UTC")
        else:
            monkeypatch.setenv("REPOFORGE_TASK_DEADLINE_UTC", str(inherited))
        store, reference = retained[-1]
        with pytest.raises(KeyboardInterrupt):
            resume_v2_task_agent(artifact_store_root=store, checkpoint_ref=reference)
        latest = load_v2_artifact(*retained[-1])
        assert latest["owner"]["deadline_utc"] == expected
        assert latest["loop"]["model_budget"]["consumed"] == 1
        assert latest["task_head"]["resources"]["model_calls"] == 1
    monkeypatch.setenv("REPOFORGE_TASK_DEADLINE_UTC", str(time.time() - 1))
    with pytest.raises(ValueError, match="DEADLINE_EXPIRED"):
        resume_v2_task_agent(artifact_store_root=retained[-1][0], checkpoint_ref=retained[-1][1])
    assert load_v2_artifact(*retained[-1])["task_head"]["resources"]["model_calls"] == 1


def test_terminal_learning_refreshes_total_resources_without_rewriting_execution(repository_workspace, monkeypatch, capsys):
    from src.mod_agent.model_qualification import read_bound_json
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
    from tests.unit.test_task_agent_v2_product_entry import (
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff,
    )

    owners = []
    original = TaskAgentExecutionV2.complete_terminal_learning

    def capture(owner):
        owners.append(owner)
        return original(owner)

    monkeypatch.setattr(TaskAgentExecutionV2, "complete_terminal_learning", capture)
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(repository_workspace, monkeypatch, capsys, False)
    owner = owners[0]
    before = owner.execution_receipt()
    immutable_execution = read_bound_json(before.execution_ref)
    owner.reserve_model_call("initial_patch")
    owner._v2_state.record_resources({"model_calls": 1, "input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        operation_id="controlled-learning-usage")
    receipt = {"status": "controlled_fixture", "budget_before": before.budget_after, "model_calls": 1,
        "budget_after": owner.budget_snapshot(), "original_execution_ref": before.execution_ref}
    receipt["receipt_ref"] = owner._save("controlled-learning.json", receipt.copy())
    checkpoint_ref = owner.refresh_recovery_checkpoint_after_learning(receipt)
    loaded = load_task_agent_checkpoint(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref)
    assert loaded["checkpoint"]["loop"]["model_turns"] == 13
    assert loaded["checkpoint"]["loop"]["model_budget"]["consumed"] == 14
    assert loaded["checkpoint"]["loop"]["post_loop_learning"]["loop_budget_before_learning"]["consumed"] == 13
    assert loaded["state"].head["resources"]["model_calls"] == 14
    assert read_bound_json(before.execution_ref) == immutable_execution


def test_independent_receipts_reject_self_consistent_forgery_and_finalization_holds_lock(repository_workspace, monkeypatch, capsys):
    from types import SimpleNamespace

    from src.mod_agent.model_qualification import canonical_hash, read_bound_json
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
    from tests.unit.test_task_agent_v2_product_entry import (
        test_v2_normal_cli_empty_registry_retarget_and_product_handoff,
    )

    owners, locked = [], []
    original_save = TaskAgentExecutionV2._save
    original_learning = TaskAgentExecutionV2.complete_terminal_learning

    def inspect_save(owner, name, payload):
        if name in {"loop-result.json", "execution.json"}:
            with pytest.raises(ValueError, match="SESSION_ALREADY_RUNNING"):
                with single_task_lock(owner.store.root, owner._recovery_journal.lock_key):
                    pytest.fail("resume acquired lock during finalization")
            locked.append(name)
        return original_save(owner, name, payload)

    def capture(owner):
        owners.append(owner)
        return original_learning(owner)

    monkeypatch.setattr(TaskAgentExecutionV2, "_save", inspect_save)
    monkeypatch.setattr(TaskAgentExecutionV2, "complete_terminal_learning", capture)
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(repository_workspace, monkeypatch, capsys, False)
    owner = owners[0]
    assert locked == ["loop-result.json", "execution.json"]

    def controlled_learning():
        with pytest.raises(ValueError, match="SESSION_ALREADY_RUNNING"):
            with single_task_lock(owner.store.root, owner._recovery_journal.lock_key):
                pytest.fail("resume acquired lock during terminal learning")
        receipt = {"status": "controlled_lock_check", "budget_before": owner.budget_snapshot(),
            "budget_after": owner.budget_snapshot(), "model_calls": 0,
            "original_execution_ref": owner.execution_receipt().execution_ref}
        receipt["receipt_ref"] = owner._save("controlled-lock-learning.json", receipt.copy())
        owner._experience._terminal_receipt = receipt
        return receipt

    owner._experience = SimpleNamespace(complete_terminal_learning=controlled_learning, _terminal_receipt=None)
    original_learning(owner)
    good_ref = owner.v2_checkpoint_ref
    state = owner._v2_state
    good = state.load(good_ref)
    latest = state.root / "session-checkpoints" / (canonical_hash(state.contract_ref) + ".json")
    failures = []
    for case in ("output", "request", "budget", "host", "finish", "learning"):
        changed = deepcopy(good)
        if case == "output":
            call_id, record = next((key, value) for key, value in changed["loop"]["call_records"].items() if value["status"] == "completed")
            record["result"]["output"]["value"] = {"forged_observation": True}
            for event in changed["loop"]["events"]:
                if event.get("event") == "tool_dispatch_completed" and event.get("call_id") == call_id:
                    event["output_sha256"] = canonical_hash(record["result"]["output"])
            code = "COMPLETED_EXECUTION_RECEIPT_CHANGED"
        elif case == "request":
            changed["loop"]["request"]["messages"][1]["content"] = {"forged_instruction": "different task"}
            code = "INITIAL_REQUEST_CHANGED"
        elif case == "budget":
            changed["loop"]["model_budget"]["consumed"] -= 1
            changed["loop"]["model_budget"]["by_stage"]["initial_patch"]["consumed"] -= 1
            code = "MODEL_BUDGET_CHANGED"
        elif case == "host":
            host = state.load(changed["host_config_ref"])
            host["model_profile"]["model"] = "unapproved-model"
            changed["host_config_ref"] = state.save("host-recovery-config", host)
            code = "HOST_CONFIG_NOT_REGISTERED"
        elif case == "finish":
            next(row for row in changed["session"]["actual_calls"] if row["name"] == "finish").pop("execution_receipt_ref")
            code = "COMPLETED_EXECUTION_RECEIPT_REQUIRED"
        else:
            receipt = read_bound_json(changed["loop"]["post_loop_learning"]["receipt_ref"])
            receipt["budget_after"]["consumed"] += 1
            receipt["budget_after"]["by_stage"]["initial_patch"]["consumed"] += 1
            changed["loop"]["post_loop_learning"]["receipt_ref"] = owner._save("forged-learning.json", receipt)
            code = "LEARNING_BUDGET_RECEIPT_CHANGED"
        forged_ref = state.save("session-checkpoint", changed)
        latest.write_text(json.dumps({"checkpoint_ref": forged_ref, "host_config_ref": changed["host_config_ref"]}), encoding="utf-8")
        with pytest.raises(ValueError, match=code):
            load_task_agent_checkpoint(artifact_store_root=state.store_root, checkpoint_ref=forged_ref)
        failures.append(case)
        latest.write_text(json.dumps({"checkpoint_ref": good_ref, "host_config_ref": good["host_config_ref"]}), encoding="utf-8")
    assert len(failures) == 6


def test_cli_resumes_applied_public_failure_from_actual_source_and_requires_new_approval(delivery_case, monkeypatch, capsys):
    import httpx

    from src.mod_agent import cli
    from src.mod_agent.build.adapter import BuildCommandResult
    from src.mod_agent.domain import csharp_repository_backend as backend
    from src.mod_agent.execution.patch_apply import PatchApplyError
    from src.mod_agent.model_qualification import canonical_hash
    from src.mod_agent.task_agent_v2_delivery import (
        apply_v2_delivery,
        build_v2_delivery,
    )
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State, load_v2_artifact
    from tests.unit.test_task_agent_v2_delivery import approve

    store, checkpoint, isolated, prepared = delivery_case
    old_checkpoint = load_v2_artifact(store, checkpoint)
    old_contract = old_checkpoint["handoff"]["task_contract_ref"]
    applied = apply_v2_delivery(artifact_store_root=store, displayed_binding=prepared["displayed_binding"],
        human_semantic_decision=approve(prepared))
    original_verify = backend.nuget._run_nuget_isolated_verification_command

    def formal_failure(profile, command):
        original_verify(profile, command)
        report = Path(command[-1])
        payload = json.loads(report.read_text())
        payload["tests"][0].update(passed=False, actual="controlled formal failure")
        report.write_text(json.dumps(payload), encoding="utf-8")
        return BuildCommandResult(1, "controlled formal failure", "", 1)

    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", formal_failure)
    failed = build_v2_delivery(artifact_store_root=store, apply_ref=applied["receipt_ref"], operation_id="formal-failure", allow_build=True)
    assert failed["status"] == "build_failed"
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", original_verify)
    original_transport = httpx.HTTPTransport.handle_request
    requests, view = [], {}

    def transport(selected, request):
        if request.method == "GET":
            return original_transport(selected, request)
        payload = json.loads(request.content)
        requests.append(payload)
        facts = next(json.loads(row["content"])["current_task_state"] for row in payload["messages"]
            if row["role"] == "user" and '"current_task_state"' in row["content"])
        observations = [json.loads(row["content"])["value"] for row in payload["messages"] if row["role"] == "tool"]
        last = observations[-1]
        if len(requests) == 1:
            transition = next(json.loads(row["content"])["source_context_transition"] for row in payload["messages"]
                if row["role"] == "user" and '"source_context_transition"' in row["content"])
            assert transition["new_apply_approval_granted"] is False and facts["current_check_ref"] is None
            assert facts["task_contract_ref"] != old_contract
            name, args = "source_read", {"parent_revision_ref": facts["current_revision_ref"], "path": "Library/Arithmetic.cs"}
        elif len(requests) == 2:
            assert "value * Factor.Multiplier" in last["text"]
            view.update(last)
            name, args = "binding_add", {"expected_registry_ref": facts["binding_registry_ref"], "source_view_ref": last["source_view_ref"],
                "target_selector": "Library/Arithmetic.cs", "rationale": "Controlled post-Apply revision", "expected_validation": "Same public tests"}
        elif len(requests) == 3:
            name, args = "candidate_submit", {"parent_revision_ref": facts["current_revision_ref"], "edits": [{
                "binding_id": last["binding_id"], "source_view_ref": view["source_view_ref"],
                "before_text": "=> value * Factor.Multiplier;", "after_text": "=> (value * Factor.Multiplier);",
                "rationale": "Controlled equivalent edit to test post-Apply continuity"}]}
        elif len(requests) == 4:
            name, args = "candidate_check", {"candidate_ref": facts["current_revision_ref"]}
        else:
            assert len(requests) == 5 and last["status"] == "passed"
            name, args = "finish", {"candidate_ref": facts["current_revision_ref"], "check_ref": facts["current_check_ref"]}
        return httpx.Response(200, request=request, json={"model": "qwen3:4b-instruct", "done": True, "done_reason": "stop",
            "prompt_eval_count": 10, "eval_count": 20, "message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": name, "arguments": args}}]}})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    from src.mod_agent.evaluation.task_measurement import measure_product_task
    initial_measurement = measure_product_task(allowed_root=store.parent.parent, run_root=store.parent,
        run_id=old_checkpoint["run_id"])
    code = cli.main(["run-request", "--resume-task-agent", failed["receipt_ref"], "--artifact-store-root", str(store)])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["status"] == "awaiting_delivery", result
    assert len(requests) == 5 and result["cumulative_resources"]["model_calls"] == 18
    from src.mod_agent.model_qualification import read_bound_json
    recovery_execution = read_bound_json(result["recovery_execution_ref"])
    assert recovery_execution["model_calls_performed"] == 5
    assert recovery_execution["native_execution_ref"] == result["task_agent_outcome"]["execution_ref"]
    recovered_measurement = measure_product_task(allowed_root=store.parent.parent, run_root=store.parent,
        run_id=old_checkpoint["run_id"], execution_refs=[result["recovery_execution_ref"]])
    assert recovered_measurement["model"]["generation_records"] == initial_measurement["model"]["generation_records"] + 5
    assert len(recovered_measurement["model"]["calls"]) == recovered_measurement["model"]["generation_records"]
    assert all(call["local_evidence"]["provider_request"]["integrity"] == "verified_bytes"
        for call in recovered_measurement["model"]["calls"])
    assert result["run_id"] == old_checkpoint["run_id"] and result["request_id"] == old_checkpoint["request_id"]
    checkpoint_ref = result["v2_delivery_checkpoint_ref"]
    new_delivery = load_v2_artifact(store, checkpoint_ref)
    from src.mod_agent.evaluation.native_execution_lineage import validate_native_execution_lineage
    from src.mod_agent.evaluation.product_trial_batch import _verify_v2_live_submission
    lineage = validate_native_execution_lineage(artifact_store_root=store,
        native_execution_ref=result["task_agent_outcome"]["execution_ref"],
        initial_native_ref=old_checkpoint["handoff"]["execution_ref"])
    live = _verify_v2_live_submission(store.parent, lineage["final_native"], {
        "candidate_ref": new_delivery["handoff"]["candidate_ref"]}, recovered_measurement)
    assert live["call_id"] == lineage["final_state"].validate_candidate(
        new_delivery["handoff"]["candidate_ref"])["model_origin"]["call_id"]
    new_state = TaskAgentV2State(store, new_delivery["handoff"]["task_contract_ref"])
    assert new_state.load(new_state.contract["baseline_manifest_ref"])["source_root"] == applied["applied_source_root"]
    assert new_delivery["resource_record"]["delivery_applies"] == new_delivery["resource_record"]["delivery_builds"] == 1
    assert new_delivery["approval_granted"] is False
    original_state = TaskAgentV2State(store, old_contract)
    old_latest = original_state.root / "session-checkpoints" / (canonical_hash(old_contract) + ".json")
    with pytest.raises(ValueError, match="TASK_CONTRACT_SUPERSEDED"):
        load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=json.loads(old_latest.read_bytes())["checkpoint_ref"])
    next_isolated = isolated.parent / "next-delivery"
    next_isolated.mkdir()
    code = cli.main(["apply-patch", "--artifact-store-root", str(store), "--v2-delivery-checkpoint", checkpoint_ref,
        "--v2-delivery-action", "prepare", "--isolated-root", str(next_isolated), "--operation-id", "new-diff"])
    second = json.loads(capsys.readouterr().out)
    assert code == 0 and second["status"] == "awaiting_approval", second
    with pytest.raises(PatchApplyError):
        apply_v2_delivery(artifact_store_root=store, displayed_binding=second["displayed_binding"], human_semantic_decision=approve(prepared))
    assert not Path(second["displayed_binding"]["isolated_workspace"]).exists()
    binding_path = isolated.parent / "controlled-new-binding.json"
    decision_path = isolated.parent / "controlled-new-approval.json"
    binding_path.write_text(json.dumps(second["displayed_binding"]), encoding="utf-8")
    decision_path.write_text(json.dumps(approve(second)), encoding="utf-8")
    code = cli.main(["apply-patch", "--artifact-store-root", str(store), "--v2-delivery-checkpoint", checkpoint_ref,
        "--v2-delivery-action", "apply", "--confirm", "--v2-displayed-binding", str(binding_path),
        "--human-semantic-decision", str(decision_path)])
    applied_again = json.loads(capsys.readouterr().out)
    assert code == 0 and applied_again["status"] == "applied_in_isolation", applied_again
    code = cli.main(["apply-patch", "--artifact-store-root", str(store), "--v2-delivery-checkpoint", applied_again["receipt_ref"],
        "--v2-delivery-action", "build", "--operation-id", "new-build", "--allow-build"])
    built = json.loads(capsys.readouterr().out)
    assert code == 0 and built["status"] == "built", built
    assert built["candidate_ref"] == new_delivery["handoff"]["candidate_ref"]
    assert built["candidate_source_id"] == new_delivery["candidate_source_id"]
    assert built["resource_record"]["delivery_applies"] == built["resource_record"]["delivery_builds"] == 2
    assert built["resource_record"]["model_budget"]["consumed"] == 18
    assert built["delivered_artifact_refs"]
    from src.mod_agent.task_agent_v2_delivery import read_completed_v2_delivery
    delivered = read_completed_v2_delivery(artifact_store_root=store, receipt_ref=built["receipt_ref"],
        native_execution_ref=result["task_agent_outcome"]["execution_ref"])
    assert delivered["candidate_source_id"] == built["candidate_source_id"]
    import hashlib
    for artifact in built["delivered_artifact_refs"]:
        assert Path(artifact["path"]).resolve().is_relative_to(Path(built["applied_source_root"]).resolve())
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
    assert "=> (value * Factor.Multiplier);" in Path(built["applied_source_root"], "Library/Arithmetic.cs").read_text()
    code = cli.main(["run-request", "--resume-task-agent", failed["receipt_ref"], "--artifact-store-root", str(store)])
    replay = json.loads(capsys.readouterr().out)
    assert code == 0 and replay["cumulative_resources"]["model_calls"] == 18 and len(requests) == 5, replay
    assert read_bound_json(replay["recovery_execution_ref"])["model_calls_performed"] == 0
    replay_measurement = measure_product_task(allowed_root=store.parent.parent, run_root=store.parent,
        run_id=old_checkpoint["run_id"], execution_refs=[result["recovery_execution_ref"],
            result["recovery_execution_ref"], replay["recovery_execution_ref"]])
    assert replay_measurement["model"]["generation_records"] == recovered_measurement["model"]["generation_records"]
