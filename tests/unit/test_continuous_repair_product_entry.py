"""Continuous repair through the public CLI with controlled provider decisions.

为什么要有它：实际请求、原子编辑和公开失败续修须在同一正常入口相遇；受控模型不计自主成功。
"""
import json
from pathlib import Path

import httpx

from src.mod_agent import cli, local_entry
from src.mod_agent.build.adapter import BuildCommandResult
from src.mod_agent.build.nuget import NuGetIsolatedBuildExecution
from src.mod_agent.domain import csharp_repository_backend as backend
from tests.unit.test_csharp_repository_backend import _profile
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def test_normal_cli_no_plan_or_binding_gate_retains_observation_and_repairs(
    repository_workspace, monkeypatch, capsys, semantic_observer=None, distinct_navigation=False,
):
    profile, source = _profile(repository_workspace)
    baseline = (source / "Library/Arithmetic.cs").read_bytes()
    requests, actions, checks = [], [], []
    memory = {"reads": 0, "phase": "investigate"}
    statement = "Initial hypothesis: the current arithmetic expression may explain the public failure."
    model = "qwen3:4b-instruct"

    def build(selected):
        root = selected.working_directory
        dll = root / "Library/bin/Release/net8.0/Library.dll"
        dll.parent.mkdir(parents=True, exist_ok=True)
        dll.write_bytes(b"CONTROLLED NOT A REAL DLL")
        capture = next((p.split("=", 1)[1] for p in selected.build_command
                        if "RepositoryCompileInputs=" in p), None)
        if capture:
            Path(capture).write_text(
                f"{root / 'Library/Library.csproj'}|{root / 'Library/Arithmetic.cs'}\n",
                encoding="utf-8",
            )
        return NuGetIsolatedBuildExecution(0, 0, "", "", "", "", 1)

    def verify(selected, command):
        code = (selected.working_directory / "Library/Arithmetic.cs").read_text()
        actual = 6 if "value * 2" in code else 9 if "value * 3" in code else 3
        checks.append(actual)
        Path(command[-1]).write_text(json.dumps({
            "schema_version": "csharp-test-report.v1",
            "tests": [{"name": "double", "passed": actual == 6,
                       "expected": 6, "actual": actual,
                       "details": f"Expected Double(3) to equal 6, but found {actual}."}],
        }), encoding="utf-8")
        return BuildCommandResult(0, "controlled public check", "", 1)

    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)

    def transport(_transport, request):
        if request.method == "GET":
            payload = {"version": "controlled-continuous-repair"} if request.url.path == "/api/version" else {
                "models": [{"name": model, "model": model, "digest": "1" * 64,
                            "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}],
            }
            return httpx.Response(200, request=request, json=payload)
        payload = json.loads(request.content)
        requests.append(payload)
        current = next(json.loads(row["content"])["current_task_state"]
                       for row in reversed(payload["messages"])
                       if row["role"] == "user" and '"current_task_state"' in row["content"])
        outputs = [json.loads(row["content"]) for row in payload["messages"] if row["role"] == "tool"]
        last = outputs[-1].get("value", outputs[-1]) if outputs else None
        revision = current["current_revision_ref"]
        phase = memory["phase"]
        content = ""
        read_count = 8 if distinct_navigation else 4
        if phase == "investigate" and memory["reads"] < read_count:
            content = statement if memory["reads"] == 0 else "Re-read the same current source view."
            name, arguments = "source_read", {"parent_revision_ref": revision, "path": "Library/Arithmetic.cs"}
            if distinct_navigation and memory["reads"] < 7:
                arguments["limit"] = memory["reads"] + 1
            if distinct_navigation:
                assert current["working_state"]["stagnation"]["navigation_only"] is False
            memory["reads"] += 1
        elif phase == "investigate":
            assert statement in json.dumps(payload), "The actual request must retain the observation without duplicating its audit ledger"
            assert current["investigation_observations"]["repeated_view_returns"] == (0 if distinct_navigation else 3)
            if distinct_navigation:
                signal = current["working_state"]["stagnation"]
                assert signal["navigation_only"] is True
                assert signal["consecutive_navigation_returns"] == 8
                assert "eight consecutive" in signal["guidance"]
            assert not current["valid_bindings"]
            name, arguments = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": last["source_view_ref"],
                "before_text": "=> value;", "after_text": "=> value * 3;",
            }]}
            content = "Test an intentionally incorrect controlled multiplier hypothesis."
            memory["phase"] = "check_first"
        elif phase == "check_first":
            if distinct_navigation:
                assert current["working_state"]["stagnation"]["navigation_only"] is False
            assert last["candidate_ref"] == revision
            assert current["valid_bindings"], "The platform must have created the edit binding"
            memory["failed_candidate"] = revision
            name, arguments = "candidate_check", {"candidate_ref": revision}
            memory["phase"] = "read_revision"
        elif phase == "read_revision":
            assert last["status"] == "behavior_failed"
            assert last["candidate_ref"] == memory["failed_candidate"]
            assert "found 9" in json.dumps(last)
            memory["failed_check"] = last["check_ref"]
            content = "The public result refutes the multiplier-three hypothesis; inspect the current preimage."
            name, arguments = "source_read", {"parent_revision_ref": revision, "path": "Library/Arithmetic.cs"}
            memory["phase"] = "repair"
        elif phase == "repair":
            assert current["current_check_ref"] == memory["failed_check"]
            assert "value * 3" in last["text"]
            name, arguments = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": last["source_view_ref"],
                "before_text": "value * 3", "after_text": "value * 2",
            }]}
            memory["phase"] = "check_repair"
        elif phase == "check_repair":
            assert revision != memory["failed_candidate"]
            assert current["current_check_ref"] is None
            name, arguments = "candidate_check", {"candidate_ref": revision}
            memory["phase"] = "finish"
        else:
            assert phase == "finish" and last["status"] == "passed"
            name, arguments = "finish", {"candidate_ref": revision, "check_ref": last["check_ref"]}
        if semantic_observer is not None:
            # 为什么要有它：同一正常入口可检验语义声明与既有修复轨迹共同运行，避免复制或替换执行器。
            arguments = semantic_observer(name=name, arguments=arguments, current=current,
                last=last, request=payload, turn=len(requests), workspace=repository_workspace)
        actions.append(name)
        return httpx.Response(200, request=request, json={
            "model": model, "message": {"role": "assistant", "content": content,
                "tool_calls": [{"id": f"continuous-{len(requests)}", "function": {"name": name, "arguments": arguments}}]},
            "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20,
        })

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    exit_code = cli.main([
        "run-request", "--target-project", str(source), "--repository-profile", profile.reference["path"],
        "--workspace-root", str(repository_workspace / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536",
        "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
        "--defer-approval", "--confirm-isolated-write", "--allow-build", "--",
        "Make Arithmetic.Double return twice its input. Preserve the public signatures and tests.",
    ])
    result = json.loads(capsys.readouterr().out)
    (repository_workspace / "continuous-repair-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (repository_workspace / "continuous-repair-requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    assert exit_code == 0 and result["status"] == "awaiting_delivery", result.get("task_agent_outcome", result)
    assert "plan_update" not in actions and "binding_add" not in actions
    assert actions.count("candidate_submit") == actions.count("candidate_check") == 2
    assert checks[-2:] == [9, 6]
    assert (source / "Library/Arithmetic.cs").read_bytes() == baseline
    assert result["model_call_budget"]["consumed"] == len(requests)
    assert result["v2_delivery_checkpoint_ref"]
    learning = result["strategy_learning"]
    assert learning["status"] == "not_configured" and learning["episode_status"] == "sealed"
    episode = json.loads(Path(learning["episode_ref"]["path"]).read_text(encoding="utf-8"))
    assert "public_counterexample" in episode["information_gain"]
    assert "effective_revision" in episode["information_gain"]
    assert episode["visibility"] == "public" and episode["data_use"] == "strategy_learning"
    proposal = json.loads(Path(result["artifacts"]["patch_proposal"]).read_text())["data"]
    handoff = proposal["context_pack"]["task_agent_handoff_v2"]
    assert handoff["candidate_ref"] != memory["failed_candidate"]
    assert "Arithmetic.cs" in json.dumps(proposal["diff_intent"])


def test_normal_entry_sends_navigation_feedback_then_accepts_model_chosen_candidate(
        repository_workspace, monkeypatch, capsys):
    # 为什么要有它：反馈须进入实际 provider 请求，正常候选权限与检查行为也须继续成立。
    test_normal_cli_no_plan_or_binding_gate_retains_observation_and_repairs(
        repository_workspace, monkeypatch, capsys, distinct_navigation=True)


def test_navigation_receipt_recovery_sends_feedback_without_repeating_eighth_read(
        repository_workspace, monkeypatch, capsys):
    import pytest
    from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2

    original = TaskAgentExecutionV2.record_v2_tool_receipt
    retained, read_calls = [], []

    def interrupt(owner, origin, output, session):
        reference = original(owner, origin, output, session)
        if origin["name"] == "source_read":
            read_calls.append(origin["call_id"])
        if origin["turn"] == 8 and not retained:
            retained.append((str(owner.store.root), owner.v2_checkpoint_ref))
            raise KeyboardInterrupt()
        return reference

    # 为什么要有它：第八次读取回执后的真实恢复必须送达同一导航反馈，且不能重复读取或重置预算。
    monkeypatch.setattr(TaskAgentExecutionV2, "record_v2_tool_receipt", interrupt)
    with pytest.raises(KeyboardInterrupt):
        test_normal_cli_no_plan_or_binding_gate_retains_observation_and_repairs(
            repository_workspace, monkeypatch, capsys, distinct_navigation=True)
    capsys.readouterr()
    assert len(read_calls) == 8 and len(set(read_calls)) == 8
    store, checkpoint = retained[0]
    code = cli.main(["run-request", "--resume-task-agent", checkpoint, "--artifact-store-root", store])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["status"] == "awaiting_delivery"
    assert result["new_task_created"] is False
    assert result["cumulative_resources"]["model_calls"] == 14
    assert len(read_calls) == 9 and len(set(read_calls)) == 9
