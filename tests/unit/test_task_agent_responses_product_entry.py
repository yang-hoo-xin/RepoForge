"""Normal Responses task entry and recovery with controlled HTTP and SDK.

为什么要有它：新提供方必须经过同一源码授权、公开反例和恢复入口；脚本化模型不能计为自主修复。
"""
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from src.core.settings import load_settings
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


@pytest.mark.parametrize("boundary", ["provider_receipt", "tool_receipt"])
def test_responses_normal_cli_failure_revision_and_receipt_recovery(
    repository_workspace, monkeypatch, capsys, record_property, boundary, registered_entry=False, omit_host_ids=False,
):
    root = repository_workspace
    profile, source = _profile(root)
    baseline = (source / "Library/Arithmetic.cs").read_bytes()
    model = "controlled-reference-model"
    settings = load_settings("config/settings.yaml")
    settings = replace(settings, llm=replace(settings.llm, provider="openai", model=model,
        base_url="https://api.openai.com/v1", max_tokens=4096))
    monkeypatch.setattr(local_entry, "load_settings", lambda _path: settings)
    monkeypatch.setenv("OPENAI_API_KEY", "controlled-test-credential")
    monkeypatch.delenv("REPOFORGE_TEST_BASE_URL", raising=False)
    external_path = root / "responses-profile.json"
    external_path.write_text(json.dumps({
        "profile_version": "external_openai_responses_profile_v1",
        "base_url": "https://api.openai.com/v1", "model": model,
        "model_catalog_url": "https://api.openai.com/v1/models",
        "context_window": 65536, "output_budget": 4096, "timeout_seconds": 600,
        "credential_env": "OPENAI_API_KEY", "reasoning_effort": "medium",
    }), encoding="utf-8")
    owners, requests, returned, receipts, checks, interruptions = [], [], [], [], [], []
    saved = {}
    original_init = TaskAgentExecutionV2.__init__
    original_receipt = TaskAgentExecutionV2.record_v2_tool_receipt
    original_save = TaskAgentExecutionV2._save

    def capture(owner, **kwargs):
        original_init(owner, **kwargs)
        owners.append(owner)

    def interrupt_tool(owner, origin, output, session):
        reference = original_receipt(owner, origin, output, session)
        receipts.append({"origin": deepcopy(origin), "receipt_ref": reference})
        if boundary == "tool_receipt" and origin["name"] == "source_read" and not interruptions:
            interruptions.append((owner.store.root, owner.v2_checkpoint_ref))
            raise KeyboardInterrupt()
        return reference

    def interrupt_provider(owner, name, payload):
        reference = original_save(owner, name, payload)
        if (boundary == "provider_receipt" and name.startswith("provider-outcome-")
                and payload.get("status") == "responded" and payload.get("turn") == 1 and not interruptions):
            interruptions.append((owner.store.root, owner.v2_checkpoint_ref))
            raise KeyboardInterrupt()
        return reference

    monkeypatch.setattr(TaskAgentExecutionV2, "__init__", capture)
    monkeypatch.setattr(TaskAgentExecutionV2, "record_v2_tool_receipt", interrupt_tool)
    monkeypatch.setattr(TaskAgentExecutionV2, "_save", interrupt_provider)

    def build(selected):
        directory = selected.working_directory
        artifact = directory / "Library/bin/Release/net8.0/Library.dll"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"CONTROLLED NOT A REAL DLL")
        capture_path = next((part.split("=", 1)[1] for part in selected.build_command
            if "RepositoryCompileInputs=" in part), None)
        if capture_path:
            Path(capture_path).write_text(
                f"{directory / 'Library/Library.csproj'}|{directory / 'Library/Arithmetic.cs'}\n", encoding="utf-8")
        return NuGetIsolatedBuildExecution(0, 0, "", "", "", "", 1)

    def verify(selected, command):
        actual = (selected.working_directory / "Library/Arithmetic.cs").read_text(encoding="utf-8")
        passed = "value * 2" in actual
        checks.append(passed)
        Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1", "tests": [
            {"name": "double", "passed": passed, "expected": 6, "actual": 6 if passed else 4}]}), encoding="utf-8")
        return BuildCommandResult(0, "Controlled public business assertion", "", 1)

    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)

    def transport(_transport, request):
        assert request.url.host == "api.openai.com"
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            return httpx.Response(200, request=request, json={"data": [{"id": model}]})
        assert request.method == "POST" and request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        requests.append({"body": payload, "sha256": hashlib.sha256(request.content).hexdigest()})
        assert payload["model"] == model and payload["parallel_tool_calls"] is False
        assert payload["store"] is False and payload["truncation"] == "disabled"
        assert "reasoning.encrypted_content" in payload["include"]
        assert all(tool["strict"] is False for tool in payload["tools"])
        if omit_host_ids:
            assert all("working_state" not in tool["parameters"].get("required", []) for tool in payload["tools"])
        assert payload.get("reasoning", {}).get("effort") == "medium"
        items = payload["input"]
        current = next(json.loads(row["content"])["current_task_state"] for row in reversed(items)
            if row.get("role") == "user" and '"current_task_state"' in row["content"])
        calls = {row["call_id"]: row for row in items if row.get("type") == "function_call"}
        outputs = [row for row in items if row.get("type") == "function_call_output"]
        for row in outputs:
            assert row["call_id"] in calls
            assert calls[row["call_id"]] in returned
            sequence = int(row["call_id"].split("-")[-1])
            assert any(item.get("type") == "reasoning" and item.get("encrypted_content") == f"controlled-opaque-{sequence}"
                for item in items)
        last = json.loads(outputs[-1]["output"]) if outputs else None
        last = last.get("value", last) if isinstance(last, dict) else last
        revision, turn = current["current_revision_ref"], len(requests)
        if turn == 1:
            saved["baseline_revision"] = revision
            name, arguments = "source_read", {"parent_revision_ref": revision, "path": "Library/Arithmetic.cs"}
        elif turn == 2:
            saved["first_view"] = deepcopy(last)
            name, arguments = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": last["source_view_ref"],
                "before_text": "text never returned by the source reader", "after_text": "untrusted change"}]}
        elif turn == 3:
            assert revision == saved["baseline_revision"] and not current["valid_bindings"]
            before = saved["first_view"]["text"]
            name, arguments = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": saved["first_view"]["source_view_ref"],
                "before_text": before, "after_text": before.replace("=> value;", "=> value + 1;")}]}
        elif turn == 4:
            assert revision != saved["baseline_revision"]
            saved["wrong_candidate"] = revision
            name, arguments = "candidate_check", {"candidate_ref": revision}
        elif turn == 5:
            assert last["status"] == "behavior_failed" and revision == saved["wrong_candidate"]
            saved["public_failure"] = deepcopy(last)
            name, arguments = "candidate_read", {"parent_revision_ref": revision, "path": "Library/Arithmetic.cs"}
        elif turn == 6:
            before = last["text"]
            assert "=> value + 1;" in before
            name, arguments = "candidate_submit", {"parent_revision_ref": revision, "edits": [{
                "path": "Library/Arithmetic.cs", "source_view_ref": last["source_view_ref"],
                "before_text": before, "after_text": before.replace("=> value + 1;", "=> value * 2;")}]}
        elif turn == 7:
            assert revision != saved["wrong_candidate"]
            name, arguments = "candidate_check", {"candidate_ref": revision}
        else:
            assert turn == 8 and last["status"] == "passed"
            name, arguments = "finish", {"candidate_ref": revision, "check_ref": last["check_ref"]}
        if omit_host_ids:
            arguments = {key: value for key, value in arguments.items()
                if key not in {"parent_revision_ref", "candidate_ref", "check_ref", "expected_registry_ref"}}
        else:
            arguments["working_state"] = {}
        call = {"type": "function_call", "id": f"item-{turn}", "call_id": f"call-{turn}",
            "status": "completed", "name": name, "arguments": json.dumps(arguments)}
        returned.append(call)
        return httpx.Response(200, request=request, json={"id": f"response-{turn}", "model": model,
            "status": "completed", "output": [
                {"type": "reasoning", "id": f"reasoning-{turn}", "summary": [],
                    "encrypted_content": f"controlled-opaque-{turn}"}, call],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 4}, "output_tokens_details": {"reasoning_tokens": 5}}})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    arguments = ["run-request", "--target-project", str(source), "--repository-profile", profile.reference["path"],
        "--workspace-root", str(root / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--external-patch-profile", str(external_path), "--api-key-env", "OPENAI_API_KEY",
        "--base-url-env", "REPOFORGE_TEST_BASE_URL", "--allow-source-disclosure",
        "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
        "--defer-approval", "--confirm-isolated-write", "--allow-build", "--",
        "Make Arithmetic.Double return twice its input. Preserve signatures and tests."]
    registered = None
    if registered_entry:
        import yaml

        from scripts import dsh_run_request as bridge

        # 为什么要有它：注册入口校验磁盘上的实际配置，不能依赖测试进程内替换settings掩盖提供方不一致。
        document = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
        def clear_credentials(value):
            # 为什么要有它：受控配置不能复制本机其他provider的凭据；模型只用本测试显式注入的虚构ENV值。
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key).casefold() in {"api_key", "secret_key", "access_token", "refresh_token", "password", "authorization", "client_secret"}:
                        value[key] = None
                    else:
                        clear_credentials(item)
            elif isinstance(value, list):
                for item in value:
                    clear_credentials(item)
        clear_credentials(document)
        document["llm"].update(provider="openai", model=model, base_url="https://api.openai.com/v1", max_tokens=4096)
        settings_path = root / "responses-settings.yaml"
        settings_path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
        arguments[arguments.index("--settings") + 1] = str(settings_path)
        configuration = root / "registered-input.json"
        configuration.write_text(json.dumps({"arguments": arguments[1:arguments.index("--")]}), encoding="utf-8")
        description = arguments[-1]
        registered = bridge.prepare_new_task(execution_config=str(configuration), description=description)
        arguments = bridge.build_product_argv(execution_config=registered, description=description)
    with pytest.raises(KeyboardInterrupt):
        if registered is None:
            cli.main(arguments)
        else:
            bridge._dispatch_product(arguments, execution_config=registered)
    capsys.readouterr()
    assert len(requests) == 1
    store, checkpoint_ref = interruptions[0]
    code = cli.main(["run-request", "--resume-task-agent", checkpoint_ref, "--artifact-store-root", str(store)])
    result = json.loads(capsys.readouterr().out)
    assert code == 0 and result["status"] == "awaiting_delivery", result
    assert len(requests) == 8 and checks == [False, True]
    assert sum(row["origin"]["name"] == "source_read" for row in receipts) == 1
    owner = owners[-1]
    assert owner.budget_snapshot()["consumed"] == 8
    assert (source / "Library/Arithmetic.cs").read_bytes() == baseline
    assert "value * 2" in owner._v2_state.materialize(owner._v2_state.current_revision_ref)["Library/Arithmetic.cs"]
    checkpoint = owner._v2_state.load(owner.v2_checkpoint_ref)
    turns = checkpoint["details"]["provider_turns"]
    assert len(turns) == 8
    responses = [read_bound_json(turn["response_ref"]) for turn in turns]
    assert all(row["provider"] == "openai_responses" and row["model"] == model for row in responses)
    assert sum(row["usage"]["input_tokens"] + row["usage"]["output_tokens"] for row in responses) == 240
    assert all(row["usage"]["reasoning_tokens"] == 5 and row["usage"]["cached_input_tokens"] == 4 for row in responses)
    if omit_host_ids:
        from src.mod_agent.task_agent_v2_tools import resolved_origin_arguments

        for origin in checkpoint["session"]["actual_calls"]:
            assert origin["arguments"] == read_bound_json(origin["arguments_ref"])
            assert not {"parent_revision_ref", "candidate_ref", "check_ref", "working_state"}.intersection(origin["arguments"])
            effective = resolved_origin_arguments(origin)
            assert effective == origin["effective_arguments"] and origin["argument_resolution"]["defaulted_fields"]
    for response, request in zip(responses, requests, strict=True):
        diagnostic = response["provider_diagnostic"]
        assert diagnostic["usage_available"] is True
        assert diagnostic["request_identity"]["provider_body_sha256"] == request["sha256"]
    assert result["v2_delivery_checkpoint_ref"]
    registered_evidence = None
    if registered is not None:
        # 为什么要有它：同一registered配置必须能查回原生证据且不触发生成；恢复后的产品结果不能被桥接器补写为成功。
        run_id = result["run_id"]
        assert Path(bridge.task_execution_config(execution_config=registered, run_id=run_id)).resolve() == Path(registered).resolve()
        assert bridge.main(["--execution-config", registered, "--inspect-run", run_id,
            "--view", "evidence", "--format", "json"]) == 0
        observed = json.loads(capsys.readouterr().out)
        assert observed["run_id"] == run_id and observed["read_only"] is True
        assert len(requests) == 8
        registered_evidence = _save(root / "responses-registered-evidence.json", observed)
    record_property("responses_entry", json.dumps({
        "scope": "controlled_HTTP_and_SDK_not_autonomous_repair", "interruption": boundary, "omitted_host_identifiers": omit_host_ids,
        "actual_request_refs": [turn["request_ref"] for turn in turns],
        "initial_checkpoint_ref": checkpoint_ref, "final_checkpoint_ref": owner.v2_checkpoint_ref,
        "public_failure": saved["public_failure"], "candidate_ref": owner._v2_state.current_revision_ref,
        "delivery_checkpoint_ref": result["v2_delivery_checkpoint_ref"], "calls": 8,
        "registered_configuration": registered, "registered_evidence_ref": registered_evidence,
        "wire_ref": _save(root / "responses-controlled-wire.json", requests),
        "result_ref": _save(root / "responses-product-result.json", result)}))


def test_registered_responses_entry_and_read_only_evidence(
    repository_workspace, monkeypatch, capsys, record_property,
):
    test_responses_normal_cli_failure_revision_and_receipt_recovery(
        repository_workspace, monkeypatch, capsys, record_property, "tool_receipt", registered_entry=True)


@pytest.mark.parametrize("boundary", ["provider_receipt", "tool_receipt"])
def test_registered_work_view_uses_bound_defaults_and_recovers(
    repository_workspace, monkeypatch, capsys, record_property, boundary,
):
    test_responses_normal_cli_failure_revision_and_receipt_recovery(
        repository_workspace, monkeypatch, capsys, record_property, boundary,
        registered_entry=True, omit_host_ids=True)
