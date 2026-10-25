"""Explicit catalogs receive normal and stopped public runs before controlled extraction."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.planning.learning_context import expand_learning_content
from src.mod_agent.planning.strategy_experience import StrategyExperience
from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
from src.mod_agent.task_agent_v2_experience import TaskAgentV2Experience, seal_saved_public_episode
from src.mod_agent.task_agent_v2_recovery import load_task_agent_checkpoint, resume_v2_task_agent
from tests.unit.test_csharp_repository_backend import _fake_execution, _profile
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)
from tests.unit.test_strategy_learning_request import question


@pytest.mark.parametrize("change,error", [
    ({"episode_catalog": {}}, "CATALOG_CONFIG_INVALID"),
    ({"episode_catalog": {"catalog_id": "all", "select_latest": True}}, "CATALOG_CONFIG_INVALID"),
    ({"extraction_episode_set_ref": {"fake": "frozen"}}, "CATALOG_AND_FROZEN_SET_CONFLICT"),
    ({"extract_on_terminal": True}, "CATALOG_LEARNING_PURPOSE_REQUIRED"),
    ({"episode_catalog": {"catalog_id": "all", "selection": {"only_success": True}}}, "CATALOG_SELECTION_INVALID"),
])
def test_catalog_mode_requires_explicit_selection_and_learning_purpose(tmp_path, change, error):
    value = {"schema_version": "task_agent_experience_config_v1", "store_root": str(tmp_path / "memory"),
        "snapshot_ref": None, "extract_on_terminal": False, "episode_catalog": {"catalog_id": "all"}, **change}
    with pytest.raises(ValueError, match=error):
        TaskAgentV2Experience(SimpleNamespace(), _save(tmp_path / "config.json", value))


def test_normal_no_candidate_runs_join_next_actual_extraction_and_stop_replays(
        repository_workspace, monkeypatch, capsys, record_property):
    root = repository_workspace
    memory = StrategyExperience(ArtifactStore(root / "experience"))
    learning_ref = question(root)
    scope = {"runtime": "dotnet", "task_phase": "investigation", "has_candidate": False}
    owners, decision_requests, learning_requests = [], [], []
    active = {}
    original_init = TaskAgentExecutionV2.__init__

    def capture(owner, **kwargs):
        original_init(owner, **kwargs)
        owners.append(owner)

    monkeypatch.setattr(TaskAgentExecutionV2, "__init__", capture)
    _fake_execution(monkeypatch)
    fake_build = backend.nuget._run_nuget_isolated_build

    def build(selected):
        result = fake_build(selected)
        artifact = selected.working_directory / "Library/bin/Release/net8.0/Library.dll"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"CONTROLLED NOT A REAL DLL")
        return result

    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
    model = "qwen3:4b-instruct"

    def transport(_transport, request):
        if request.method == "GET":
            payload = {"version": "controlled-catalog"} if request.url.path == "/api/version" else {
                "models": [{"name": model, "model": model, "digest": "1" * 64,
                    "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}]}
            return httpx.Response(200, request=request, json=payload)
        payload = json.loads(request.content)
        if not payload.get("tools"):
            learning_requests.append(payload)
            content = json.dumps({"status": "insufficient_information",
                "reason": "Controlled process observations do not establish strategy transfer.",
                "evidence_gaps": [{"status": "not_observed", "description": "Independent transfer was not measured."}]})
            message, reason = {"role": "assistant", "content": content}, "stop"
        else:
            decision_requests.append(payload)
            active["turn"] += 1
            if active["turn"] == 1:
                context = next(json.loads(row["content"]) for row in payload["messages"]
                    if row["role"] == "user" and '"task_agent_v2"' in row["content"])
                message = {"role": "assistant", "content": "I will inspect the source before choosing an experiment.",
                    "tool_calls": [{"id": "read-first", "function": {"name": "source_read", "arguments": {
                        "parent_revision_ref": context["parent_revision_ref"], "path": "Library/Arithmetic.cs"}}}]}
                reason = "stop"
            else:
                assert active["turn"] == 2
                message = {"role": "assistant", "content": "A stale binding attempt ends this controlled run.",
                    "tool_calls": [{"id": "stale-binding", "function": {"name": "binding_add", "arguments": {
                        "expected_registry_ref": "controlled-stale-registry", "source_view_ref": "controlled-missing-view",
                        "target_selector": "Library/Arithmetic.cs", "rationale": "Controlled public admission rejection.",
                        "expected_validation": "Double public test"}}}]}
                reason = "stop"
        return httpx.Response(200, request=request, json={"model": model, "message": message,
            "done": True, "done_reason": reason, "prompt_eval_count": 10, "eval_count": 20})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)

    def run(name, *, extract=False, frozen=None, configured=True):
        case = root / name
        profile, source = _profile(case)
        configuration = {"schema_version": "task_agent_experience_config_v1", "store_root": str(memory.store.root),
            "snapshot_ref": None, "extract_on_terminal": extract}
        if frozen is not None:
            configuration["extraction_episode_set_ref"] = asdict(frozen)
        else:
            configuration["episode_catalog"] = {"catalog_id": "all-public-runs"}
        if extract:
            configuration.update(learning_request_ref=learning_ref, requested_strategy_scope=scope)
        config = _save(case / "experience.json", configuration)
        active["turn"] = 0
        arguments = ["run-request", "--target-project", str(source), "--repository-profile", profile.reference["path"],
            "--workspace-root", str(case / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
            "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536",
            "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1",
            "--technical-executor", "task_agent", "--task-agent-protocol", "v2", "--defer-approval",
            "--confirm-isolated-write", "--allow-build"]
        if configured:
            arguments += ["--task-agent-experience-config", config["path"]]
        cli.main([*arguments, "--", "Double the integer input, preserving the public signature."])
        result = json.loads(capsys.readouterr().out)
        assert owners and result.get("status") != "awaiting_delivery", result
        owner = owners[-1]
        receipt = owner.complete_terminal_learning()
        episode = read_bound_json(receipt["episode_ref"])
        assert episode["applicability"]["has_candidate"] is False
        assert episode["information_gain"] == ["public_process_observation"]
        assert not episode["public_counterexamples"]
        assert [row["name"] for row in owner._v2_session.actual_calls] == ["source_read", "binding_add"]
        assert owner._v2_state.current_revision_ref == owner._v2_state.contract["baseline_manifest_ref"]
        return owner, receipt

    first, first_receipt = run("first")
    assert first_receipt["status"] == "not_enabled" and first_receipt["model_calls"] == 0
    assert first_receipt["catalog_registration"]["status"] == "registered" and not learning_requests
    first_frozen = memory.freeze_catalog("all-public-runs", applicability=scope, learning_request_ref=learning_ref)
    original_frozen = memory.read(first_frozen)
    assert [row["episode_ref"] for row in original_frozen["episodes"]] == [first_receipt["episode_ref"]]
    original_budget = first.budget_snapshot()

    # 为什么要有它：停止收尾必须使用原宿主配置、保留未知尾部并可重复登记，不能为收集经历恢复模型执行。
    checkpoint_ref = first.v2_checkpoint_ref
    while True:
        checkpoint = first._v2_state.load(checkpoint_ref)
        if checkpoint["loop"]["phase"] == "provider_pending" and len(checkpoint["details"]["completed_tool_records"]) == 1:
            break
        checkpoint_ref = checkpoint["previous_checkpoint_ref"]
        assert checkpoint_ref is not None
    host = first._v2_state.load(first._recovery_journal.host_config_ref)
    assert {"episode_catalog.py", "learning_context.py"} <= {Path(row["path"]).name for row in host["owner_refs"]}
    source = _save(root / "stopped-process.json", {"status": "user_stopped", "exit_code": 130,
        "run_id": checkpoint["details"]["run_id"], "request_id": checkpoint["details"]["request_id"]})
    stop = _save(root / "stop.json", {"schema": "task-agent-process-stop.v1", "checkpoint_ref": checkpoint_ref,
        "task_contract_ref": checkpoint["task_contract_ref"], "run_id": checkpoint["details"]["run_id"],
        "request_id": checkpoint["details"]["request_id"], "stop_kind": "user_stopped", "source_ref": source, "exit_code": 130})
    stopped = seal_saved_public_episode(artifact_store_root=first.store.root, checkpoint_ref=checkpoint_ref, stop_ref=stop)
    assert stopped == seal_saved_public_episode(artifact_store_root=first.store.root, checkpoint_ref=checkpoint_ref, stop_ref=stop)
    observed = read_bound_json(read_bound_json(stopped)["execution_ref"])
    assert observed["unknown_effects"] == ["provider_response_unknown"]
    assert {"episode_catalog.py", "learning_context.py"} <= {Path(row["path"]).name for row in observed["collector_source_refs"]}
    assert first.budget_snapshot() == original_budget and not learning_requests
    registrations = [json.loads(path.read_text(encoding="utf-8")) for path in
        (Path(first.store.root) / "public-learning").glob("public-episode-registration-*.json")]
    assert len(registrations) == 1 and registrations[0]["episode_ref"] == stopped
    assert registrations[0]["status"] == "registered"

    second, receipt = run("second", extract=True)
    assert receipt["status"] == "insufficient_information", receipt
    assert receipt["model_calls"] == 1 and len(learning_requests) == 1
    actual = expand_learning_content(json.loads(learning_requests[0]["messages"][-1]["content"]))
    expected = {read_bound_json(row)["run_id"] for row in (first_receipt["episode_ref"], receipt["episode_ref"])}
    assert {row["run_id"] for row in actual["episodes"]} == expected
    assert len(actual["episodes"]) == 2
    assert all(row["source_applicability"]["has_candidate"] is False for row in actual["episodes"])
    frozen = memory.read(receipt["episode_set_ref"])
    assert receipt["catalog_selection_ref"] == frozen["catalog_selection_ref"]
    assert receipt["episode_ref"] in [row["episode_ref"] for row in frozen["episodes"]]
    assert second.complete_terminal_learning() == receipt and len(learning_requests) == 1
    loaded = load_task_agent_checkpoint(artifact_store_root=second.store.root, checkpoint_ref=second.v2_checkpoint_ref)
    assert loaded["checkpoint"]["terminal_learning_receipt"] == receipt
    original_requests = len(decision_requests)
    resumed = resume_v2_task_agent(artifact_store_root=second.store.root, checkpoint_ref=second.v2_checkpoint_ref)
    assert resumed["strategy_learning"] == receipt
    assert len(decision_requests) == original_requests and len(learning_requests) == 1
    assert memory.read(first_frozen) == original_frozen

    third, legacy = run("legacy", extract=True, frozen=first_frozen)
    assert legacy["status"] == "insufficient_information", legacy
    legacy_request = expand_learning_content(json.loads(learning_requests[-1]["messages"][-1]["content"]))
    assert len(legacy_request["episodes"]) == 1
    assert legacy["episode_set_ref"] == asdict(first_frozen)
    assert "catalog_registration" not in legacy
    assert third.complete_terminal_learning() == legacy and len(learning_requests) == 2

    _plain, plain_receipt = run("unconfigured", configured=False)
    assert plain_receipt["status"] == "not_configured" and len(learning_requests) == 2
    assert memory.read(first_frozen) == original_frozen
    record_property("catalog_normal_entry", json.dumps({"first_episode_ref": first_receipt["episode_ref"],
        "stopped_episode_ref": stopped, "current_episode_ref": receipt["episode_ref"],
        "episode_set_ref": receipt["episode_set_ref"], "catalog_selection_ref": receipt["catalog_selection_ref"],
        "learning_receipt_ref": receipt["receipt_ref"], "legacy_receipt_ref": legacy["receipt_ref"],
        "actual_controlled_wire_ref": _save(root / "actual-learning-wire.json", learning_requests[0]),
        "scope": "normal_CLI_controlled_provider_and_SDK_not_business_or_transfer_result"}))
