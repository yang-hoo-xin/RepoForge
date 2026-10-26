"""Search navigation, observed edit windows and legacy-call recovery through normal CLI.

为什么要有它：行号与字符坐标必须在真实请求和返回中可区分，定位改善不能扩张编辑前像或源码权限。
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.build.adapter import BuildCommandResult
from src.mod_agent.build.nuget import NuGetIsolatedBuildExecution
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
from tests.unit.test_csharp_repository_backend import _profile
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def test_search_read_coordinates_candidate_and_receipt_recovery(repository_workspace, monkeypatch, capsys, record_property):
    root = repository_workspace
    profile, source = _profile(root)
    text = "".join(f"// 前置🙂说明 {index:03d}，字符坐标不是行号。\r\n" for index in range(1, 319))
    text += "public static class Arithmetic {\r\n  public static int Double(int value) => value;\r\n}\r\n"
    text += "".join(f"// 后续位置 {index:03d}\r\n" for index in range(322, 422))
    baseline = text.encode("utf-8")
    (source / "Library/Arithmetic.cs").write_bytes(baseline)
    assert "public static int Double" in text.splitlines()[319]
    owners, requests, retained, executed_reads, checks = [], [], [], [], []
    saved = {}
    original_init = TaskAgentExecutionV2.__init__
    original_receipt = TaskAgentExecutionV2.record_v2_tool_receipt

    def capture(owner, **kwargs):
        original_init(owner, **kwargs)
        owners.append(owner)

    def interrupt_receipt(owner, origin, output, session):
        reference = original_receipt(owner, origin, output, session)
        if origin["name"] in {"source_read", "candidate_read"}:
            executed_reads.append({"origin": deepcopy(origin), "receipt_ref": reference})
        if not retained and origin["name"] == "source_read" and origin["arguments"].get("offset") == 320:
            retained.append((owner.store.root, owner.v2_checkpoint_ref))
            saved["legacy_receipt_ref"] = reference
            raise KeyboardInterrupt()
        return reference

    monkeypatch.setattr(TaskAgentExecutionV2, "__init__", capture)
    monkeypatch.setattr(TaskAgentExecutionV2, "record_v2_tool_receipt", interrupt_receipt)

    def build(selected):
        directory = selected.working_directory
        artifact = directory / "Library/bin/Release/net8.0/Library.dll"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"CONTROLLED NOT A REAL DLL")
        capture_path = next((part.split("=", 1)[1] for part in selected.build_command
            if "RepositoryCompileInputs=" in part), None)
        if capture_path:
            Path(capture_path).write_text(f"{directory / 'Library/Library.csproj'}|{directory / 'Library/Arithmetic.cs'}\n", encoding="utf-8")
        return NuGetIsolatedBuildExecution(0, 0, "", "", "", "", 1)

    def verify(selected, command):
        actual_text = (selected.working_directory / "Library/Arithmetic.cs").read_bytes().decode("utf-8")
        passed = "value * 2" in actual_text
        checks.append(passed)
        Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1", "tests": [
            {"name": "double", "passed": passed, "expected": 6, "actual": 6 if passed else 3}]}), encoding="utf-8")
        return BuildCommandResult(0, "controlled public verification", "", 1)

    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)
    model = "qwen3:4b-instruct"

    def transport(_transport, request):
        if request.method == "GET":
            value = {"version": "controlled-source-navigation"} if request.url.path == "/api/version" else {
                "models": [{"name": model, "model": model, "digest": "1" * 64,
                    "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}]}
            return httpx.Response(200, request=request, json=value)
        payload = json.loads(request.content)
        requests.append(payload)
        current = next(json.loads(row["content"])["current_task_state"] for row in reversed(payload["messages"])
            if row["role"] == "user" and '"current_task_state"' in row["content"])
        outputs = [json.loads(row["content"]) for row in payload["messages"] if row["role"] == "tool"]
        last = outputs[-1].get("value", outputs[-1]) if outputs else None
        revision, turn = current["current_revision_ref"], len(requests)
        if turn == 1:
            saved["baseline_revision"] = revision
            # 为什么要有它：新schema不宣传旧参数，但旧客户端/已存响应仍须按原字符语义执行并恢复。
            schema = next(row["function"]["parameters"] for row in payload["tools"] if row["function"]["name"] == "source_read")
            assert "offset" not in schema["properties"] and "offset_chars" in schema["properties"]
            name, arguments = "source_read", {"parent_revision_ref": revision, "path": "Library/Arithmetic.cs", "offset": 320, "limit": 60}
            saved["legacy_arguments"] = deepcopy(arguments)
        elif turn == 2:
            assert last["text"] == text[320:380]
            assert last["line_window"]["start_line"] < 20 and "public static int Double" not in last["text"]
            assert last["read_window_chars"] == {"offset_chars": 320, "end_chars": 380}
            saved["earlier_text"] = last["text"]
            saved["next_read"] = deepcopy(last["next_read"])
            assert saved["next_read"]["offset_chars"] == 380 and "offset" not in saved["next_read"]
            name, arguments = "source_read", deepcopy(last["next_read"])
        elif turn == 3:
            assert last["read_window_chars"]["offset_chars"] == 380
            name, arguments = "source_search", {"parent_revision_ref": revision, "query": "public static int Double"}
        elif turn == 4:
            assert len(last["matches"]) == 1
            match = last["matches"][0]
            assert match["line"] == 320 and match["read_args"]["start_line"] > 300
            saved["search_result"] = deepcopy(last)
            name, arguments = "source_read", deepcopy(match["read_args"])
            saved["search_read_args"] = deepcopy(arguments)
        elif turn == 5:
            assert last["line_window"]["start_line"] > 300 and "\r\n" in last["text"]
            assert last["read_window_chars"]["offset_chars"] > 320
            assert saved["earlier_text"] not in last["text"]
            saved["target_read"] = deepcopy(last)
            saved["before_text"] = next(line for line in last["text"].splitlines(keepends=True) if "public static int Double" in line)
            name, arguments = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": last["source_view_ref"],
                "before_text": saved["earlier_text"], "after_text": saved["earlier_text"] + "// wrong view"}]}
        elif turn == 6:
            assert "before_text_outside_read_window" in json.dumps(payload)
            assert not current["valid_bindings"] and revision == saved["baseline_revision"]
            name, arguments = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": saved["target_read"]["source_view_ref"],
                "before_text": saved["before_text"], "after_text": saved["before_text"].replace("=> value;", "=> value * 2;")}]}
        elif turn == 7:
            assert last["candidate_ref"] == revision and revision != saved["baseline_revision"]
            assert len(current["valid_bindings"]) == 1
            name, arguments = "candidate_check", {"candidate_ref": revision}
        else:
            assert turn == 8 and last["status"] == "passed"
            name, arguments = "finish", {"candidate_ref": revision, "check_ref": last["check_ref"]}
        return httpx.Response(200, request=request, json={"model": model,
            "message": {"role": "assistant", "content": "Controlled navigation decision.", "tool_calls": [
                {"id": "navigation-" + str(turn), "function": {"name": name, "arguments": arguments}}]},
            "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    arguments = ["run-request", "--target-project", str(source), "--repository-profile", profile.reference["path"],
        "--workspace-root", str(root / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536", "--model-call-budget-policy", "meter_only",
        "--constraint-policy", "source_priority_v1", "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
        "--defer-approval", "--confirm-isolated-write", "--allow-build", "--", "Make Arithmetic.Double return twice its input. Preserve signatures and tests."]
    with pytest.raises(KeyboardInterrupt):
        cli.main(arguments)
    capsys.readouterr()
    assert len(executed_reads) == len(requests) == 1
    store, checkpoint_ref = retained[0]
    exit_code = cli.main(["run-request", "--resume-task-agent", checkpoint_ref, "--artifact-store-root", str(store)])
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0 and result["status"] == "awaiting_delivery", result
    assert len(requests) == 8 and len(executed_reads) == 3 and checks == [True]
    owner = owners[-1]
    assert owner.budget_snapshot()["consumed"] == 8
    legacy = read_bound_json(saved["legacy_receipt_ref"])
    assert legacy["origin"]["arguments"] == saved["legacy_arguments"]
    assert read_bound_json(legacy["origin"]["arguments_ref"]) == saved["legacy_arguments"]
    assert (source / "Library/Arithmetic.cs").read_bytes() == baseline
    candidate = owner._v2_state.materialize(owner._v2_state.current_revision_ref)["Library/Arithmetic.cs"]
    assert candidate == text.replace(saved["before_text"], saved["before_text"].replace("=> value;", "=> value * 2;"), 1)
    checkpoint = owner._v2_state.load(owner.v2_checkpoint_ref)
    executed = [row["name"] for row in checkpoint["details"]["completed_tool_records"]]
    assert executed.count("candidate_submit") == 1
    assert any(row["name"] == "candidate_submit" and row["status"] == "not_dispatched"
        for row in checkpoint["loop"]["call_records"].values())
    assert result["v2_delivery_checkpoint_ref"]
    evidence = {"scope": "controlled_model_and_SDK_not_autonomous_repair", "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
        "original_pending_checkpoint_ref": checkpoint_ref, "final_checkpoint_ref": owner.v2_checkpoint_ref,
        "actual_request_refs": [row["request_ref"] for row in checkpoint["details"]["provider_turns"]],
        "read_receipts": executed_reads, "legacy_arguments_ref": legacy["origin"]["arguments_ref"],
        "search_read_args": saved["search_read_args"], "source_view_ref": saved["target_read"]["source_view_ref"],
        "candidate_ref": owner._v2_state.current_revision_ref, "delivery_checkpoint_ref": result["v2_delivery_checkpoint_ref"],
        "actual_controlled_wire_ref": _save(root / "navigation-actual-requests.json", requests),
        "product_result_ref": _save(root / "navigation-product-result.json", result)}
    record_property("navigation_entry", json.dumps(evidence))
