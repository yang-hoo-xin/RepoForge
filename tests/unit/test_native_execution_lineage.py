"""Controlled ancestry counterexamples; not autonomous task evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.evaluation import native_execution_lineage as lineage
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_v2_state import save_v2_artifact


def _case(tmp_path, monkeypatch):
    root = tmp_path / "a"
    root.mkdir()
    docs, states = {}, {}
    original = {"schema_version": "task-agent-product-execution.v2", "run_id": "run", "request_id": "request",
        "task_contract_ref": "old", "provider_turns": [{"reservation_id": "one", "status": "responded"}],
        "completed_tool_records": [{"call_id": "first"}]}
    initial_ref = _save(root / "initial/execution.json", original)
    docs["host-old"] = {"original_output_root": str(root / "initial")}
    states["old"] = SimpleNamespace(contract_ref="old", contract={},
        head={"operations": {"host-recovery-config": {"result": {"host_config_ref": "host-old"}}}}, load=lambda ref: docs[ref])
    monkeypatch.setattr(lineage, "TaskAgentV2State", lambda _root, ref: states[ref])
    return root, docs, states, original, initial_ref


@pytest.mark.parametrize("damage", [None, "unrelated", "prefix", "checkpoint_chain", "missing_receipt", "changed_output"])
def test_same_contract_resume_requires_bound_checkpoint_history(tmp_path, monkeypatch, damage):
    root, docs, _states, initial, initial_ref = _case(tmp_path, monkeypatch)
    before = {"schema": "task-agent-session-checkpoint.v2", "task_contract_ref": "old", "host_config_ref": "host-old",
        "owner": {"output_root": str(root / "initial")}, "details": initial, "previous_checkpoint_ref": None}
    before_ref = save_v2_artifact(root, "session-checkpoint", before)
    final = deepcopy(initial)
    final["provider_turns"].append({"reservation_id": "two", "status": "responded"})
    final["completed_tool_records"].append({"call_id": "second"})
    if damage == "prefix":
        final["completed_tool_records"][0]["call_id"] = "forged"
    after = {**before, "owner": {"output_root": str(root / ("wrong" if damage == "changed_output" else "resume"))},
        "details": final, "previous_checkpoint_ref": None if damage == "checkpoint_chain" else before_ref}
    after_ref = save_v2_artifact(root, "session-checkpoint", after)
    final_ref = _save(root / "resume/execution.json", final)
    if damage != "missing_receipt":
        _save(root / "resume/recovery-execution.json", {"run_id": "run", "request_id": "request", "native_execution_ref": final_ref,
            "resumed_from_checkpoint_ref": before_ref, "v2_checkpoint_ref": after_ref})
    if damage == "unrelated":
        initial_ref = _save(root / "elsewhere/execution.json", initial)
    kwargs = {"artifact_store_root": root, "native_execution_ref": final_ref, "initial_native_ref": initial_ref}
    if damage:
        with pytest.raises(ValueError, match="NATIVE_LINEAGE_"):
            lineage.validate_native_execution_lineage(**kwargs)
    else:
        result = lineage.validate_native_execution_lineage(**kwargs)
        assert result["initial_native_ref"] == initial_ref
        assert result["final_native"] == final
        assert result["chain"][0]["kind"] == "checkpoint_resume"
        assert lineage.validate_native_execution_lineage(artifact_store_root=root,
            native_execution_ref=final_ref)["initial_host"] == docs["host-old"]


@pytest.mark.parametrize("damage", [None, "origin_native", "receipt_native", "historical_native", "prior_contract", "task", "authority"])
def test_rebase_uses_owner_validation_and_exact_prior_delivery(tmp_path, monkeypatch, damage):
    root, docs, states, initial, initial_ref = _case(tmp_path, monkeypatch)
    wrong_ref = _save(root / "unrelated/execution.json", initial)
    checkpoint = {"run_id": "run", "request_id": "request", "handoff": {
        "execution_ref": wrong_ref if damage == "receipt_native" else initial_ref, "task_contract_ref": "old"}}
    checkpoint_ref = save_v2_artifact(root, "delivery-checkpoint", checkpoint)
    receipt_ref = save_v2_artifact(root, "delivery-receipt", {"checkpoint_ref": checkpoint_ref})
    docs["origin"] = {"prior_execution_ref": wrong_ref if damage == "origin_native" else initial_ref,
        "run_id": "run", "request_id": "request"}
    docs["host-new"] = {"original_output_root": str(root / "initial")}
    states["new"] = SimpleNamespace(contract_ref="new", contract={"historical_task_contract_ref": "old",
        "historical_execution_ref": initial_ref}, load=lambda ref: docs[ref],
        head={"operations": {"host-recovery-config": {"result": {"host_config_ref": "host-new"}}}})
    record = {"new_task_contract_ref": "new", "prior_task_contract_ref": "wrong" if damage == "prior_contract" else "old",
        "historical_execution_ref": initial_ref, "rebase_origin_ref": "origin", "receipt_ref": receipt_ref,
        "run_id": "wrong" if damage == "task" else "run", "request_id": "request"}
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        if damage == "authority":
            raise ValueError("V2_DELIVERY_REBASE_AUTHORITY_CHANGED")
        return record
    monkeypatch.setattr(lineage, "validate_v2_delivery_rebase", validate)
    final = {**initial, "task_contract_ref": "new", "rebase_ref": "rebase",
        "historical_execution_ref": wrong_ref if damage == "historical_native" else initial_ref}
    final_ref = _save(root / "resume/execution.json", final)
    kwargs = {"artifact_store_root": root, "native_execution_ref": final_ref, "initial_native_ref": initial_ref}
    if damage:
        with pytest.raises(ValueError):
            lineage.validate_native_execution_lineage(**kwargs)
    else:
        result = lineage.validate_native_execution_lineage(**kwargs)
        assert result["initial_state"] is states["old"] and result["final_state"] is states["new"]
        assert result["chain"] == [{"kind": "delivery_rebase", "native_execution_ref": final_ref,
            "prior_native_execution_ref": initial_ref, "rebase_ref": "rebase"}]
    assert calls == [{"artifact_store_root": Path(root), "rebase_ref": "rebase"}]


@pytest.mark.parametrize("damage", [False, True])
def test_reexported_tool_result_must_retain_event_bound_bytes(tmp_path, damage):
    output = {"execution": "executed", "value": 3}
    first = _save(tmp_path / "first.json", output)
    second = _save(tmp_path / "second.json", {**output, "value": 4} if damage else output)
    row = {"call_id": "one", "event": {"output_sha256": canonical_hash(output)}}
    prior = {"completed_tool_records": [{**row, "result_ref": first}]}
    final = {"completed_tool_records": [{**row, "result_ref": second}]}
    if damage:
        with pytest.raises(ValueError, match="TOOL_RESULT_CHANGED"):
            lineage._prefix(prior, final, tmp_path)
    else:
        lineage._prefix(prior, final, tmp_path)
        lineage._prefix(prior, {"completed_tool_records": [row]}, tmp_path)
        lineage._prefix({"completed_tool_records": [row]}, final, tmp_path)
