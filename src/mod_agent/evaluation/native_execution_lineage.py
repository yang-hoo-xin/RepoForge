"""Read-only identity links between native executions of one logical task.

为什么要有它：恢复可以产生新执行与新合同，但不能替换原任务或重新定义比较起点。
"""
from pathlib import Path

from src.mod_agent.domain.full_project_compilation import _ref
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_execution import _read
from src.mod_agent.task_agent_v2_delivery import validate_v2_delivery_rebase
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, load_v2_artifact


def _require(condition, code):
    if not condition:
        raise ValueError("NATIVE_LINEAGE_" + code)


def _host(state):
    anchor = state.head["operations"].get("host-recovery-config", {}).get("result", {}).get("host_config_ref")
    return state.load(anchor) if anchor else None


def _prefix(prior, current, root):
    for key in ("completed_tool_records", "provider_turns"):
        before, after = prior.get(key, []), current.get(key, [])
        _require(len(after) >= len(before), "HISTORY_SHORTENED")
        for index, row in enumerate(before):
            observed = after[index]
            if row == observed:
                continue
            if key == "completed_tool_records" and {k: v for k, v in row.items() if k != "result_ref"} == {
                    k: v for k, v in observed.items() if k != "result_ref"}:
                # 为什么要有它：owner会重新保存同一工具结果；等价原件引用允许变化，实际观察及事件hash不得变化。
                result = _read(observed.get("result_ref") or row["result_ref"], root)
                _require(canonical_hash(result) == row["event"]["output_sha256"]
                    and (not row.get("result_ref") or _read(row["result_ref"], root) == result), "TOOL_RESULT_CHANGED")
                continue
            # 为什么要有它：已发送但尚无响应的最后请求可在恢复时完成，已完成历史不得改写。
            _require(key == "provider_turns" and index == len(before) - 1 and row.get("status") == "requested"
                and all(observed.get(k) == v for k, v in row.items() if k != "status"),
                "HISTORY_CHANGED:" + key + ":" + str(index) + ":" + ",".join(k for k in set(row) | set(observed)
                    if row.get(k) != observed.get(k)))


def _checkpoint_predecessor(root, reference, native, recovery_ref):
    receipt = _read(recovery_ref, root)
    _require(receipt.get("native_execution_ref") == reference
        and all(receipt.get(key) == native.get(key) for key in ("run_id", "request_id")), "RECOVERY_EXECUTION_CHANGED")
    before = load_v2_artifact(root, receipt["resumed_from_checkpoint_ref"])
    after = load_v2_artifact(root, receipt["v2_checkpoint_ref"])
    _require(before.get("schema") == after.get("schema") == "task-agent-session-checkpoint.v2"
        and before["task_contract_ref"] == after["task_contract_ref"] == native["task_contract_ref"]
        and before["host_config_ref"] == after["host_config_ref"]
        and Path(after["owner"]["output_root"]).resolve() == Path(reference["path"]).parent.resolve(),
        "RECOVERY_CHECKPOINT_CHANGED")
    selected, seen = receipt["v2_checkpoint_ref"], set()
    while selected != receipt["resumed_from_checkpoint_ref"]:
        _require(selected is not None and selected not in seen, "CHECKPOINT_CHAIN_BROKEN")
        seen.add(selected)
        item = load_v2_artifact(root, selected)
        _require(item["task_contract_ref"] == native["task_contract_ref"], "CHECKPOINT_CONTRACT_CHANGED")
        selected = item.get("previous_checkpoint_ref")
    prior_path = Path(before["owner"]["output_root"]) / "execution.json"
    _require(prior_path.resolve() != Path(reference["path"]).resolve(), "RECOVERY_SELF_REFERENCE")
    _prefix(before["details"], after["details"], root)
    _prefix(after["details"], native, root)
    if not prior_path.exists():
        return None, before
    prior_ref = _ref(prior_path)
    prior = _read(prior_ref, root)
    _prefix(prior, before["details"], root)
    return prior_ref, before


def validate_native_execution_lineage(*, artifact_store_root, native_execution_ref,
        initial_native_ref=None, recovery_execution_ref=None):
    """Return validated initial conditions and current execution, without mutation.

    为什么要有它：批次和导出共用实际原件关联；恢复后的候选和冻结前的初始条件不能混用。
    """
    root = Path(artifact_store_root).resolve(strict=True)
    selected, chain, seen = native_execution_ref, [], set()
    initial_checkpoint_ref = None
    final_native = final_state = None
    while True:
        token = (selected["path"], selected["sha256"])
        _require(token not in seen, "EXECUTION_CYCLE")
        seen.add(token)
        native = _read(selected, root)
        _require(native.get("schema_version") == "task-agent-product-execution.v2"
            and Path(selected["path"]).name == "execution.json", "NATIVE_EXECUTION_REQUIRED")
        state = TaskAgentV2State(root, native["task_contract_ref"])
        host = _host(state)
        if final_native is None:
            final_native, final_state = native, state
        _require(all(native.get(key) == final_native.get(key) for key in ("run_id", "request_id")), "TASK_CHANGED")
        if initial_native_ref is not None and selected == initial_native_ref:
            break
        recovery_path = Path(selected["path"]).parent / "recovery-execution.json"
        recovery_ref = (recovery_execution_ref if selected == native_execution_ref and recovery_execution_ref is not None
            else _ref(recovery_path) if recovery_path.is_file() else None)
        checkpoint_prior, origin_checkpoint = (_checkpoint_predecessor(root, selected, native, recovery_ref)
            if recovery_ref is not None else (None, None))
        if origin_checkpoint is not None and checkpoint_prior is None:
            receipt = _read(recovery_ref, root)
            _require(initial_native_ref is None and not native.get("rebase_ref") and host is not None
                and origin_checkpoint["task_contract_ref"] == state.contract_ref
                and Path(origin_checkpoint["owner"]["output_root"]).resolve() == Path(host["original_output_root"]).resolve(),
                "INITIAL_EXECUTION_NOT_REACHED")
            from src.mod_agent.task_agent_v2_recovery import (
                _validate_completed,
                _validate_host_binding,
            )

            _validate_host_binding(state, origin_checkpoint, host)
            _validate_completed(origin_checkpoint, root, host)
            initial_checkpoint_ref = receipt["resumed_from_checkpoint_ref"]
            chain.append({"kind": "checkpoint_only_origin", "native_execution_ref": selected,
                "initial_checkpoint_ref": initial_checkpoint_ref, "recovery_execution_ref": recovery_ref})
            selected = None
            break
        if checkpoint_prior is not None and _read(checkpoint_prior, root)["task_contract_ref"] == state.contract_ref:
            chain.append({"kind": "checkpoint_resume", "native_execution_ref": selected,
                "prior_native_execution_ref": checkpoint_prior, "recovery_execution_ref": recovery_ref})
            selected = checkpoint_prior
            continue
        rebase_ref = native.get("rebase_ref")
        if rebase_ref:
            record = validate_v2_delivery_rebase(artifact_store_root=root, rebase_ref=rebase_ref)
            origin = state.load(record["rebase_origin_ref"])
            receipt = load_v2_artifact(root, record["receipt_ref"])
            checkpoint = load_v2_artifact(root, receipt["checkpoint_ref"])
            prior_ref = record["historical_execution_ref"]
            _require(record["new_task_contract_ref"] == state.contract_ref
                and (checkpoint_prior is None or checkpoint_prior == prior_ref)
                and state.contract["historical_task_contract_ref"] == record["prior_task_contract_ref"]
                and prior_ref == native["historical_execution_ref"] == state.contract["historical_execution_ref"]
                == origin["prior_execution_ref"] == checkpoint["handoff"]["execution_ref"]
                and checkpoint["handoff"]["task_contract_ref"] == record["prior_task_contract_ref"]
                and all(record[key] == origin[key] == checkpoint[key] == native[key] for key in ("run_id", "request_id")),
                "REBASE_EXECUTION_CHANGED")
            prior = _read(prior_ref, root)
            _require(prior["task_contract_ref"] == record["prior_task_contract_ref"], "REBASE_PRIOR_CONTRACT_CHANGED")
            _prefix(prior, native, root)
            chain.append({"kind": "delivery_rebase", "native_execution_ref": selected,
                "prior_native_execution_ref": prior_ref, "rebase_ref": rebase_ref,
                **({"recovery_execution_ref": recovery_ref} if recovery_ref is not None else {})})
            selected = prior_ref
            continue
        _require(checkpoint_prior is None, "RECOVERY_PRIOR_CONTRACT_CHANGED")
        _require(initial_native_ref is None, "INITIAL_EXECUTION_NOT_REACHED")
        # 为什么要有它：恢复目录缺失回执不能被误当成另一份初始执行。
        _require(host is None or Path(selected["path"]).parent.resolve() == Path(host["original_output_root"]).resolve(),
            "RECOVERY_RECEIPT_MISSING")
        break
    return {"initial_native_ref": selected, "initial_checkpoint_ref": initial_checkpoint_ref,
        "initial_native": native, "initial_state": state, "initial_host": host,
        "final_native": final_native, "final_state": final_state, "chain": chain}
