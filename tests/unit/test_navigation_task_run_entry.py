"""Controlled external providers through the real Task Run and terminal owners.

为什么要有它：预检通过不代表运行采用了同一预算；零模型入口反例必须覆盖真实终态写入而非手造结果。
"""
import json
from pathlib import Path

import httpx
import pytest

from scripts import dsh_run_request as bridge
from scripts import run_repository_product_pilot as pilot
from src.mod_agent import local_entry
from src.mod_agent.build.adapter import BuildCommandResult
from src.mod_agent.build.nuget import NuGetIsolatedBuildExecution
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref
from src.mod_agent.model_qualification import read_bound_json
from tests.unit.test_csharp_repository_backend import _profile
from tests.unit.test_csharp_repository_product_entry import repository_workspace as repository_workspace
from tests.unit.test_navigation_experiment_admission import requirement


def controlled_entry(root, monkeypatch, *, navigation=False, model_calls=1, reads=None):
    profile, source = _profile(root)
    requests, saved = [], {}
    original = (source / "Library/Arithmetic.cs").read_bytes()

    def build(selected):
        directory = selected.working_directory
        artifact = directory / "Library/bin/Release/net8.0/Library.dll"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"CONTROLLED BUILD BOUNDARY; NOT A REAL DLL")
        capture = next((part.split("=", 1)[1] for part in selected.build_command
                        if "RepositoryCompileInputs=" in part), None)
        if capture:
            Path(capture).write_text(f"{directory / 'Library/Library.csproj'}|{directory / 'Library/Arithmetic.cs'}\n", encoding="utf-8")
        return NuGetIsolatedBuildExecution(0, 0, "", "", "", "", 1)

    def verify(selected, command):
        text = (selected.working_directory / "Library/Arithmetic.cs").read_text(encoding="utf-8")
        passed = "value * 2" in text
        Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1", "tests": [
            {"name": "double", "passed": passed, "expected": 6, "actual": 6 if passed else 3}]}), encoding="utf-8")
        return BuildCommandResult(0, "controlled check", "", 1)

    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)
    model = "qwen3:4b-instruct"

    def transport(_transport, request):
        if request.method == "GET":
            value = {"version": "controlled"} if request.url.path == "/api/version" else {"models": [
                {"name": model, "model": model, "digest": "1" * 64,
                 "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}]}
            return httpx.Response(200, request=request, json=value)
        payload = json.loads(request.content)
        requests.append(payload)
        current = next(json.loads(row["content"])["current_task_state"] for row in reversed(payload["messages"])
            if row["role"] == "user" and '"current_task_state"' in row["content"])
        outputs = [json.loads(row["content"]) for row in payload["messages"] if row["role"] == "tool"]
        last = outputs[-1].get("value", outputs[-1]) if outputs else None
        turn, revision = len(requests), current["current_revision_ref"]
        if navigation and turn <= 8:
            name, args = "source_search", {"parent_revision_ref": revision, "query": f"controlled_missing_{turn}"}
        elif not navigation or turn == 9:
            if navigation:
                assert current["working_state"]["stagnation"]["navigation_only"] is True
            name, args = "source_read", {"parent_revision_ref": revision, "path": "Library/Arithmetic.cs"}
        elif turn == 10:
            saved["view"] = last["source_view_ref"]
            name, args = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": saved["view"],
                "before_text": "=> value;", "after_text": "=> value * 2;"}]}
        elif turn == 11:
            assert last["candidate_ref"] == revision
            name, args = "candidate_check", {"candidate_ref": revision}
        else:
            assert turn == 12 and last["status"] == "passed"
            name, args = "finish", {"candidate_ref": revision, "check_ref": last["check_ref"]}
        return httpx.Response(200, request=request, json={"model": model, "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"controlled-{turn}", "function": {"name": name, "arguments": args}}]},
            "prompt_eval_count": 10, "eval_count": 20})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    arguments = ["--target-project", str(source), "--repository-profile", profile.reference["path"],
        "--workspace-root", str(root / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536",
        "--model-call-budget-policy", "task_total_only", "--task-model-call-limit", "16" if navigation else str(model_calls),
        "--constraint-policy", "source_priority_v1", "--technical-executor", "task_agent",
        "--task-agent-protocol", "v2", "--task-agent-semantic-state", "enabled",
        "--source-retrieval", "rg_live", "--defer-approval", "--confirm-isolated-write", "--allow-build"]
    if navigation or reads is not None:
        worker = root / "policy.json"
        worker.write_text(json.dumps({"schema_version": "patch_worker_configuration_v1",
            "enable_recovery_loop": False, "behavior_evaluator": None, "policy": {
                "model_call_budget_policy": "task_total_only", "max_total_model_turns": 16 if navigation else model_calls,
                "max_tool_rounds": 12 if navigation else reads, "max_observation_bytes": 131072}}), encoding="utf-8")
        arguments += ["--patch-worker-config", str(worker)]
    configuration = root / "execution.json"
    configuration.write_text(json.dumps({"arguments": arguments}), encoding="utf-8")
    command = bridge.OpenTaskRun(execution_config_ref=_ref(configuration), intent="Make Double return twice its input.")
    bound = command.bind()
    if navigation:
        pilot.validate_navigation_experiment(bound.execution_config_ref, requirement(), wall_seconds=900)
    result = bridge.execute_task_run(bridge.OpenTaskRun(bound.execution_config_ref, command.intent))
    assert (source / "Library/Arithmetic.cs").read_bytes() == original
    return result, requests


def test_admitted_budget_reaches_feedback_candidate_and_check(repository_workspace, monkeypatch):
    result, requests = controlled_entry(repository_workspace, monkeypatch, navigation=True)
    assert isinstance(result, bridge.TaskRunCompleted)
    assert result.status == "awaiting_delivery"
    assert len(requests) == 12


@pytest.mark.parametrize("model_calls,reads", [(1, None), (3, 1)])
def test_budget_stop_without_candidate_returns_blocked_evidence(repository_workspace, monkeypatch, model_calls, reads):
    result, requests = controlled_entry(repository_workspace, monkeypatch, model_calls=model_calls, reads=reads)
    assert isinstance(result, bridge.TaskRunCompleted)
    assert result.status == "blocked"
    assert len(requests) == model_calls
    manifest = read_bound_json(result.evidence_bundle_ref)
    terminal = read_bound_json(manifest["terminal_record"])
    outcome = read_bound_json(result.run_outcome_ref)
    assert terminal["status"] == terminal["current_stage"] == "blocked"
    assert outcome["run_id"] == result.run_id and outcome["request_id"] == result.request_id
    assert terminal["actual_write_performed"] is False
    assert terminal["model_call_budget"]["consumed"] == model_calls
    assert terminal["task_agent_outcome"]["candidate_ref"] is None
    assert terminal["task_agent_outcome"]["check_ref"] is None
    assert any("BUDGET" in row["code"] for row in terminal["blockers"])
    loop = read_bound_json(terminal["task_agent_outcome"]["loop_result_ref"])
    assert loop["read_units"] == 1
    assert sum(row["status"] == "completed" for row in loop["call_records"].values()) == 1
    if reads is not None:
        rejected = [row for row in loop["call_records"].values() if row["status"] == "not_dispatched"]
        assert len(rejected) == 1
        assert rejected[0]["result"]["output"]["code"] == "TASK_AGENT_READ_BUDGET_EXCEEDED"
