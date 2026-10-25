"""Same-turn semantic declarations through the normal continuous-repair CLI.

为什么要有它：模型声明须进入实际下一请求且保留原始调用；错误声明不能阻止已授权候选或完成交接。
"""
import json
from copy import deepcopy
from pathlib import Path

from src.mod_agent.model_qualification import canonical_hash
from tests.unit.test_continuous_repair_product_entry import (
    repository_workspace as repository_workspace,
)
from tests.unit.test_continuous_repair_product_entry import (
    test_normal_cli_no_plan_or_binding_gate_retains_observation_and_repairs as run_repair,
)


def _value(document, locator):
    for part in locator:
        document = document[part]
    return document


def _visible_quote(current, workspace, turn, quote):
    # 为什么要有它：受控响应仅引用本轮已发送请求中的内容，不从源文件或旧模型结果补写可见证据。
    requests = list(workspace.glob(f"runs/*/a/task-agent-execution/request-{turn:03d}.json"))
    assert len(requests) == 1
    document = json.loads(requests[0].read_bytes())
    for entry in current["semantic_evidence_catalog"]:
        value = _value(document, entry["locator"])
        assert canonical_hash(value) == entry["value_sha256"]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        if quote in text:
            return {"id": entry["id"], "quote": quote}
    raise AssertionError("The controlled quotation must occur in the actually sent evidence catalogue")


def _state(evidence=(), *, refuted=False):
    return {
        "observations": ([{"statement": "The public check returned the recorded failure.", "evidence": list(evidence)}]
            if refuted else [{"statement": "The method currently returns its input.", "evidence": list(evidence)}]
            if evidence else []),
        "hypotheses": [{"id": "h1", "statement": "Multiplying by three might satisfy the requested doubling behavior.",
            "status": "refuted" if refuted else "open", "evidence": list(evidence)}],
        "open_questions": [] if refuted else ["Which multiplier satisfies the public behavior?"],
        "next_action": {"purpose": "Inspect the current source before the next permitted edit.",
            "expected_observation": "Current source bytes or public behavior can distinguish the hypothesis.",
            "decision_if_observed": "Use the observed counterexample to revise the proposed multiplier."},
    }


def test_normal_cli_semantic_state_and_invalid_metadata_preserve_real_action_chain(
    repository_workspace, monkeypatch, capsys, request,
):
    observed, supplied = {}, {}

    def observer(*, name, arguments, current, last, request, turn, workspace):
        observed[turn] = deepcopy(current["semantic_work_state"])
        assert current["semantic_work_state"]["authority"] == "advisory_only"
        assert current["semantic_state_contract"]
        assert all("working_state" in tool["function"]["parameters"]["properties"] for tool in request["tools"])
        if turn == 1:
            update = _state()
        elif turn in {2, 5}:
            quote = _visible_quote(current, workspace, turn, "=> value;")
            update = _state([quote])
        elif turn in {3, 8}:
            update = "invalid model state does not revoke the action"
        elif turn == 7:
            assert name == "source_read" and last["status"] == "behavior_failed"
            quote = _visible_quote(current, workspace, turn, "found 9")
            update = _state([quote], refuted=True)
        else:
            update = {}
        supplied[turn] = deepcopy(update)
        return {**arguments, "working_state": update}

    run_repair(repository_workspace, monkeypatch, capsys, semantic_observer=observer)
    assert observed[1]["status"] == "not_expressed"
    assert observed[2]["state"] == supplied[1]
    assert observed[4]["state"] == supplied[2]
    assert observed[4]["latest_update"]["status"] == "invalid"
    assert observed[6]["state"] == supplied[5]
    assert observed[6]["freshness"] == "historical", "An accepted edit must not silently refresh old evidence"
    assert observed[8]["state"] == supplied[7]
    assert observed[8]["state"]["hypotheses"][0]["status"] == "refuted"
    assert observed[9]["state"] == supplied[7]
    assert observed[9]["latest_update"]["status"] == "invalid"
    assert observed[9]["freshness"] == "historical"

    result = json.loads((repository_workspace / "continuous-repair-result.json").read_bytes())
    requests = json.loads((repository_workspace / "continuous-repair-requests.json").read_bytes())
    assert len(requests) == result["model_call_budget"]["consumed"] == 10
    assert result["status"] == "awaiting_delivery" and result["v2_delivery_checkpoint_ref"]
    # 为什么要有它：状态元数据虽不进入业务执行参数，后续真实工具历史必须仍展示原始模型调用。
    native_calls = [call for row in requests[-1]["messages"] if row["role"] == "assistant"
                    for call in row.get("tool_calls", [])]
    by_id = {row["id"]: row["function"]["arguments"] for row in native_calls}
    assert by_id["continuous-3"]["working_state"] == supplied[3]
    assert by_id["continuous-8"]["working_state"] == supplied[8]
    proposal = json.loads(Path(result["artifacts"]["patch_proposal"]).read_bytes())["data"]
    assert proposal["context_pack"]["task_agent_handoff_v2"]["candidate_ref"]
    # 为什么要有它：未配置提炼也须保留结构化声明，后续原策略消费者不能把它们归类为未表达。
    from src.mod_agent.artifacts import ArtifactStore
    from src.mod_agent.model_qualification import read_bound_json
    from src.mod_agent.planning.strategy_experience import StrategyExperience, _episode_observations
    from src.mod_agent.planning.task_memory import _select

    assert result["strategy_learning"]["status"] == "not_configured"
    episode = read_bound_json(result["strategy_learning"]["episode_ref"])
    memory = StrategyExperience(ArtifactStore(repository_workspace / "semantic-episode-consumer"))
    documents = memory._public_sources(episode["evidence_refs"])
    experience = _episode_observations(episode, documents)["experience"]
    assert len(experience["actions"]) == 10
    for action in experience["actions"]:
        explicit = next(row for row in action["model_statement"] if row["locator"][-1] == "working_state")
        assert explicit["value"] == supplied[action["action_index"] + 1]
        assert explicit["value"] == _select(documents[explicit["evidence_id"]], explicit["locator"])
    assert experience["actions"][0]["model_statement_status"] == "explicit_structured_statement_unassessed"
    assert experience["actions"][6]["model_statement_status"] == "explicit_structured_statement_unassessed"
    assert experience["actions"][8]["model_statement_status"] == "not_explicitly_expressed"
    assert isinstance(experience["actions"][0]["model_statement"][-1]["value"], dict)
    validation = documents["semantic_validation"]
    # 为什么要有它：验收原件从本次测试结果直接定位，后续核对不依赖临时目录时间或“最新”选择。
    request.node.user_properties.append(("public_episode_ref", json.dumps(result["strategy_learning"]["episode_ref"])))
    validation_ref = next(row for row in episode["evidence_refs"] if row["evidence_id"] == "semantic_validation")
    request.node.user_properties.append(("semantic_validation_ref", json.dumps({key: validation_ref[key] for key in ("path", "sha256")})))
    events = next(row["value"] for row in experience["technical_outcome"]
        if row["evidence_id"] == "semantic_validation" and row["locator"] == ["events"])
    assert events == validation["events"] and len(events) == 10
    assert events[2]["status"] == events[7]["status"] == "invalid"
    assert events[3]["status"] == events[5]["status"] == events[8]["status"] == events[9]["status"] == "empty"
    assert events[6]["status"] == "accepted" and events[6]["call_ids"] == ["continuous-7"]
    assert [event["action_indices"] for event in events] == [[index] for index in range(10)]
    assert all(event["outside_completed_prefix_call_ids"] == [] for event in events)
    assert "does not mean semantic truth" in validation["validation_scope"]
    assert validation["authority_changed"] is False and validation["model_calls"] == 0
    checkpoint = read_bound_json(validation["source_checkpoint_ref"])
    assert checkpoint["task_contract_ref"] == episode["task_contract_ref"]
    first = checkpoint["details"]["provider_turns"][0]
    first_request = read_bound_json(first["request_ref"])
    first_system = next(row for row in first_request["messages"] if row["role"] == "system" and row["content"])
    assert "You are the technical agent in one continuous repository task." in first_system["content"]
    assert "Include working_state with your next normal tool action." in first_system["content"]
    assert sum(row["role"] == "system" for row in first_request["messages"]) == 1
    store_root = Path(first["request_ref"]["path"]).parents[1]
    host_path = store_root / checkpoint["host_config_ref"]
    host_ref = {"path": str(host_path), "sha256": host_path.stem}
    host = read_bound_json(host_ref)
    contract_path = store_root / checkpoint["task_contract_ref"]
    contract_ref = {"path": str(contract_path), "sha256": contract_path.stem}
    contract = read_bound_json(contract_ref)
    assert host["semantic_state_enabled"] is True
    # 为什么要有它：后续原生协议检查复用明确的受控入口请求及原任务边界，不能用真实业务轨迹或“最新请求”替换。
    properties = {
        "first_request_ref": first["request_ref"],
        "first_provider_request_ref": first["provider_request_ref"],
        "host_config_ref": host_ref,
        "model_profile_source": {"source_ref": host_ref, "locator": ["model_profile"]},
        "repository_profile_ref": contract["execution_profile_ref"],
        "fixture_original_boundary": {"task_contract_ref": contract_ref,
            "task_requirements": contract["task_requirements"], "allowed_edit_paths": contract["allowed_edit_paths"],
            "baseline_manifest_ref": contract["baseline_manifest_ref"],
            "scope": "Scripted model/build normal-entry fixture; not a real business solver result. Native protocol reuse is input-only.",
            "observed_boundary": "Actual client request; live server rendered token prompt was not captured."},
    }
    for key, value in properties.items():
        request.node.user_properties.append((key, json.dumps(value)))
    for index, event in enumerate(events, 1):
        actual_response = read_bound_json(event["response_ref"])
        actual_request = read_bound_json(event["request_ref"])
        assert actual_response["tool_calls"][0]["arguments"]["working_state"] == supplied[index]
        assert any(isinstance(row.get("content"), dict) and "current_task_state" in row["content"]
            for row in actual_request["messages"])
    assert "not delivered" in experience["fact_scope"]["technical_outcome"]
