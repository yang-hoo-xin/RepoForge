"""Scripted decisions through the normal v2 CLI, not autonomous model success.

为什么要有它：受控轨迹只证明调查、目标调整和交接能执行，真实B是否选择这些动作必须另行验证。
"""
import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.build.adapter import BuildCommandResult
from src.mod_agent.build.nuget import NuGetIsolatedBuildExecution
from src.mod_agent.domain import csharp_repository_backend as backend
from tests.unit.test_csharp_repository_backend import _profile
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


@pytest.mark.parametrize("real_sdk", [False, True], ids=["controlled-build", "real-sdk"])
def test_v2_normal_cli_empty_registry_retarget_and_product_handoff(repository_workspace, monkeypatch, capsys, real_sdk,
                                                                 exercise_experiments=False, exercise_discovery=False,
                                                                 exercise_correction=False):
    short_workspace = repository_workspace
    if real_sdk and os.environ.get("REPOFORGE_REAL_REPOSITORY_CLI_INTEGRATION") != "1":
        pytest.skip("Explicit SDK integration; decisions remain scripted.")
    profile, source = _profile(short_workspace, real=real_sdk)
    (source / "Library/Unused.cs").write_bytes(b"public static class Unused { public const int Marker = 0; }\n")
    (source / "Library/Factor.cs").write_bytes(b"public static class Factor { public const int Multiplier = 1; }\n")
    profile_path = Path(profile.reference["path"])
    data = json.loads(profile_path.read_text())
    data["allowed_edit_paths"] = ["Library/Arithmetic.cs", "Library/Factor.cs", "Library/Unused.cs"]
    profile_path.write_text(json.dumps(data))
    baseline = {p.name: p.read_bytes() for p in (source / "Library").glob("*.cs")}
    builds = []
    if not real_sdk:
        def build(selected):
            root = selected.working_directory
            builds.append(str(root))
            dll = root / "Library/bin/Release/net8.0/Library.dll"
            dll.parent.mkdir(parents=True, exist_ok=True)
            dll.write_bytes(b"CONTROLLED NOT A REAL DLL")
            capture = next((p.split("=", 1)[1] for p in selected.build_command if "RepositoryCompileInputs=" in p), None)
            if capture:
                Path(capture).write_text("".join(f"{root / 'Library/Library.csproj'}|{p}\n" for p in (root / "Library").glob("*.cs")))
            return NuGetIsolatedBuildExecution(0, 0, "", "", "", "", 1)

        def verify(selected, command):
            root = selected.working_directory
            arithmetic = (root / "Library/Arithmetic.cs").read_text()
            factor = (root / "Library/Factor.cs").read_text()
            passed = "value * Factor.Multiplier" in arithmetic and "Multiplier = 2" in factor
            Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1",
                "tests": [{"name": "double", "passed": passed, "expected": 6, "actual": 6 if passed else 3}]}))
            return BuildCommandResult(0, "controlled test", "", 1)

        monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
        monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
        monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)
    memory = {}
    requests = []
    model = "qwen3:4b-instruct"

    def transport(_transport, request):
        if request.method == "GET":
            payload = {"version": "scripted-v2"} if request.url.path == "/api/version" else {
                "models": [{"name": model, "model": model, "digest": "1" * 64,
                    "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}]}
            return httpx.Response(200, request=request, json=payload)
        payload = json.loads(request.content)
        requests.append(payload)
        tools = {x["function"]["name"] for x in payload.get("tools", [])}
        assert "binding_add" in tools, "v2 must reach investigation without pre-model planning"
        context = next(json.loads(x["content"]) for x in payload["messages"]
            if x["role"] == "user" and '"task_agent_v2"' in x["content"])
        tool_outputs = [json.loads(x["content"]) for x in payload["messages"] if x["role"] == "tool"]
        observed = [row.get("value", row) for row in tool_outputs]
        last = observed[-1] if observed else None
        if exercise_discovery and len(requests) == 1:
            return httpx.Response(200, request=request, json={"model": model,
                "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "test_catalog", "arguments": {"execute_discovery": True,
                        "source_revision_ref": context["parent_revision_ref"]}}}]}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
        if exercise_discovery and len(requests) == 2:
            receipt = last["discovery_receipt"]
            assert receipt["executed"] is True and receipt["status"] == "failed"
            assert last["selection_available"] is False
            assert receipt["observations"]["tests_executed"] == 1
            assert last["grants_acceptance"] is False
        if exercise_experiments and len(requests) <= 3:
            if len(requests) == 1:
                name, args = "probe_submit", {"source_revision_ref": context["parent_revision_ref"],
                    "files": [{"relative_path": "probe/Program.cs", "content": "System.Console.WriteLine(1);"}],
                    "entrypoint_id": "csharp_console", "rationale": "Controlled probe submission",
                    "expected_observation": "One console line"}
            elif len(requests) == 2:
                assert last["executed"] is False
                name, args = "experiment_run", {"source_revision_ref": context["parent_revision_ref"],
                    "kind": "temporary_probe", "probe_artifact_ref": last["probe_artifact_ref"],
                    "hypothesis": "Controlled probe", "expected_observation": "One console line"}
            else:
                assert last["status"] == "capability_unavailable" and last["executed"] is False
                name, args = "test_catalog", {}
            return httpx.Response(200, request=request, json={"model": model,
                "message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": name, "arguments": args}}]}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
        turn = (len(requests) - (3 if exercise_experiments else 0) - int(exercise_discovery)
                - int(exercise_correction and len(requests) > 4))
        if turn == 1:
            memory.update(parent=context["parent_revision_ref"], registry=context["binding_registry_ref"])
            name, args = "plan_update", {"plan": {"hypotheses": ["Investigate marker state"], "intended_targets": ["Library/Unused.cs"]}}
        elif turn in {2, 6, 8, 10}:
            path = {2: "Unused.cs", 6: "Arithmetic.cs", 8: "Factor.cs", 10: "Unused.cs"}[turn]
            name, args = "source_read", {"parent_revision_ref": memory["parent"], "path": "Library/" + path}
            memory["reading"] = path
        elif turn in {3, 7, 9}:
            memory[memory["reading"]] = last
            name, args = "binding_add", {"expected_registry_ref": memory["registry"],
                "source_view_ref": last["source_view_ref"], "target_selector": "Library/" + memory["reading"],
                "rationale": "Controlled target selection", "expected_validation": "Public double behavior"}
        elif turn == 4:
            if exercise_correction and len(requests) == 5:
                assert last["status"] == "rejected_not_executed"
                assert last["code"] == "TASK_AGENT_CANDIDATE_ARGUMENTS_INVALID"
                assert last["read_cost"] == 0
                assert "before_text" in str(last["argument_errors"])
                current = next(json.loads(row["content"])["current_task_state"] for row in reversed(payload["messages"])
                    if row["role"] == "user" and '"current_task_state"' in row["content"])
                assert current["current_revision_ref"] == memory["parent"]
                assert current["binding_registry_ref"] == memory["registry"]
            else:
                memory.update(registry=last["registry_ref"], unused_binding=last["binding_id"])
            name, args = "candidate_submit", {"parent_revision_ref": memory["parent"], "edits": [{
                "binding_id": memory["unused_binding"], "source_view_ref": memory["Unused.cs"]["source_view_ref"],
                "before_text": "Marker = 0", "after_text": "Marker = 1", "rationale": "Intentionally wrong controlled hypothesis"}]}
            if exercise_correction and len(requests) == 4:
                del args["edits"][0]["before_text"]
        elif turn in {5, 12}:
            memory["parent"] = last["candidate_ref"]
            name, args = "candidate_check", {"candidate_ref": memory["parent"]}
        elif turn == 11:
            memory["Unused.cs"] = last
            name, args = "candidate_submit", {"parent_revision_ref": memory["parent"], "edits": [
                {"binding_id": memory["unused_binding"], "source_view_ref": last["source_view_ref"],
                 "before_text": "Marker = 1", "after_text": "Marker = 0", "rationale": "Undo wrong edit"},
                {"binding_id": memory["Arithmetic.cs_binding"], "source_view_ref": memory["Arithmetic.cs"]["source_view_ref"],
                 "before_text": "=> value;", "after_text": "=> value * Factor.Multiplier;", "rationale": "Use multiplier"},
                {"binding_id": memory["Factor.cs_binding"], "source_view_ref": memory["Factor.cs"]["source_view_ref"],
                 "before_text": "Multiplier = 1", "after_text": "Multiplier = 2", "rationale": "Correct multiplier"}]}
        else:
            assert turn == 13 and last["status"] == "passed"
            name, args = "finish", {"candidate_ref": memory["parent"], "check_ref": last["check_ref"]}
        if turn in {8, 10}:
            path = "Arithmetic.cs" if turn == 8 else "Factor.cs"
            memory.update(registry=last["registry_ref"])
            memory[path + "_binding"] = last["binding_id"]
        if turn == 6:
            assert last["status"] == "behavior_failed"
            assert last["verification_summary"]["test_command_executed"]
        response = {"model": model, "message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": name, "arguments": args}}]}, "done": True,
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20}
        return httpx.Response(200, request=request, json=response)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    code = cli.main(["run-request", "--target-project", str(source), "--repository-profile", str(profile_path),
        "--workspace-root", str(short_workspace / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536",
        "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
        "--defer-approval", "--confirm-isolated-write", "--allow-build", "--",
        "Make Arithmetic.Double return twice its integer input. Preserve the public signatures and public tests."])
    pending = json.loads(capsys.readouterr().out)
    (short_workspace / "v2-entry-result.json").write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    (short_workspace / "scripted-requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding="utf-8")
    assert code == 0 and pending["status"] == "awaiting_delivery", {
        "result_file": str(short_workspace / "v2-entry-result.json"), "status": pending.get("status"),
        "reason": pending.get("reason"), "outcome": pending.get("task_agent_outcome")}
    assert len(requests) == 13 + (3 if exercise_experiments else 0) + int(exercise_discovery) + int(exercise_correction)
    if exercise_correction:
        assert pending["model_call_budget"]["consumed"] == 14
    assert {p.name: p.read_bytes() for p in (source / "Library").glob("*.cs")} == baseline
    root = Path(pending["run_root"])
    assert not (root / "a/features" / pending["feature_id"] / "implementation_plan.json").exists()
    proposal = json.loads(Path(pending["artifacts"]["patch_proposal"]).read_text())["data"]
    assert "Unused.cs" not in json.dumps(proposal["diff_intent"])
    assert "Arithmetic.cs" in json.dumps(proposal["diff_intent"])
    assert "Factor.cs" in json.dumps(proposal["diff_intent"])
    assert pending.get("actual_write_performed") is False
    if real_sdk:
        # 为什么要有它：真实SDK验证必须经同一公开CLI完成精确授权、Apply和Build；测试批准不能冒充用户审阅。
        delivery_root = short_workspace / "delivery-sdk"
        delivery_root.mkdir()
        checkpoint = pending["v2_delivery_checkpoint_ref"]
        common = ["apply-patch", "--artifact-store-root", str(root / "a")]

        def invoke_delivery(label, arguments):
            code = cli.main([*common, *arguments])
            result = json.loads(capsys.readouterr().out)
            (short_workspace / ("v2-delivery-" + label + ".json")).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return code, result

        prepare_code, prepared = invoke_delivery("prepare", ["--v2-delivery-checkpoint", checkpoint,
            "--v2-delivery-action", "prepare", "--isolated-root", str(delivery_root),
            "--operation-id", "scripted-sdk-apply"])
        assert prepare_code == 0 and prepared["status"] == "awaiting_approval", prepared
        binding_file = short_workspace / "v2-displayed-binding.json"
        binding_file.write_text(json.dumps(prepared["displayed_binding"]), encoding="utf-8")
        human = {**prepared["human_semantic_decision_binding"], "decision": "approve",
            "reviewer_identity": "scripted-test-not-human-user"}
        decision_file = short_workspace / "v2-test-approval.json"
        decision_file.write_text(json.dumps({**human, "diff_hash": "0" * 64}), encoding="utf-8")
        apply_arguments = ["--v2-delivery-checkpoint", checkpoint, "--v2-delivery-action", "apply", "--confirm",
            "--v2-displayed-binding", str(binding_file), "--human-semantic-decision", str(decision_file)]
        denied_code, denied = invoke_delivery("wrong-approval", apply_arguments)
        assert denied_code == 2 and denied["status"] == "blocked" and "diff_hash" in denied["reason"], denied
        assert not Path(prepared["displayed_binding"]["isolated_workspace"]).exists()
        decision_file.write_text(json.dumps(human), encoding="utf-8")
        apply_code, applied = invoke_delivery("apply", apply_arguments)
        assert apply_code == 0 and applied["status"] == "applied_in_isolation", applied
        handoff = proposal["context_pack"]["task_agent_handoff_v2"]
        assert applied["candidate_ref"] == handoff["candidate_ref"]
        assert applied["candidate_source_id"] == handoff["candidate_source_id"]
        assert applied["approval"]["decision"]["reviewer_identity"] == "scripted-test-not-human-user"
        build_code, built = invoke_delivery("build", ["--v2-delivery-checkpoint", applied["receipt_ref"],
            "--v2-delivery-action", "build", "--operation-id", "scripted-sdk-build", "--allow-build"])
        assert build_code == 0 and built["status"] == "built", built
        assert built["whole_task_delivered"] is True
        assert built["delivery_scope"] == "approved_isolated_source_and_bound_public_build_tests"
        assert built["candidate_ref"] == handoff["candidate_ref"]
        assert built["candidate_source_id"] == handoff["candidate_source_id"]
        assert built["build_result"]["claims_real_build_success"] is True
        verification = built["verification_ref"]
        raw = Path(verification["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == verification["sha256"]
        report = json.loads(raw)
        assert report["execution_origin"] == "executed" and report["test_execution"]["status"] == "executed"
        assert report["checks"] and all(row["passed"] for row in report["checks"])
        assert built["delivered_artifact_refs"]
        for reference in built["delivered_artifact_refs"]:
            artifact = Path(reference["path"])
            assert artifact.resolve().is_relative_to(Path(applied["applied_source_root"]).resolve())
            content = artifact.read_bytes()
            assert hashlib.sha256(content).hexdigest() == reference["sha256"]
            if artifact.suffix == ".dll":
                assert content.startswith(b"MZ"), "SDK integration must produce a real PE assembly"
        assert {p.name: p.read_bytes() for p in (source / "Library").glob("*.cs")} == baseline
        assert len(requests) == 13 + (3 if exercise_experiments else 0), "Delivery must not generate additional model calls"


def test_normal_cli_probe_unavailable_preserves_task_path(repository_workspace, monkeypatch, capsys):
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(
        repository_workspace, monkeypatch, capsys, False, exercise_experiments=True)


def test_normal_cli_discovery_before_edit_preserves_actual_failed_execution(repository_workspace, monkeypatch, capsys):
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(
        repository_workspace, monkeypatch, capsys, False, exercise_discovery=True)


def test_normal_cli_correctable_missing_preimage_continues_same_task_and_budget(repository_workspace, monkeypatch, capsys):
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(
        repository_workspace, monkeypatch, capsys, False, exercise_correction=True)
