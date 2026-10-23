"""Scripted decisions through the normal v2 CLI, not autonomous model success.

为什么要有它：受控轨迹只证明调查、目标调整和交接能执行，真实B是否选择这些动作必须另行验证。
"""
import hashlib
import json
import os
import sys
import time
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
                                                                 exercise_correction=False, exercise_stale_view=False,
                                                                 exercise_feedback=False, expect_postcheck_failure=False):
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
            test = {"name": "double", "passed": passed, "expected": 6, "actual": 6 if passed else 3}
            if exercise_feedback and not passed:
                test["details"] = ("Expected Double(3) to equal 6, but found 3.\n"
                    f"   at PublicExample in {root / 'Library/Arithmetic.cs'}:line 1\n"
                    + "Recorded public assertion context. " * 100 + "END OF ORIGINAL PUBLIC FAILURE")
                memory["failure_details"] = test["details"]
            Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1", "tests": [test]}))
            if expect_postcheck_failure:
                memory["test_returned"] = True
                memory["test_report_path"] = Path(command[-1])
                # 为什么要有它：正常CLI按已绑定源码装载独立评估器模块，故障须注入实际执行者的后置核验。
                loaded = [module for name, module in tuple(sys.modules.items())
                          if name.startswith("_mod_agent_patch_factory_")
                          and getattr(module, "__file__", None) == backend.__file__]
                assert loaded
                for module in loaded:
                    monkeypatch.setattr(module, "_snapshot", snapshot_after_execution)
            return BuildCommandResult(0, "controlled test", "", 1)

        monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
        monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
        monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)
    memory = {}
    if expect_postcheck_failure:
        original_snapshot = backend._snapshot

        def snapshot_after_execution(path):
            if memory.pop("test_returned", False):
                raise OSError("controlled post-execution source integrity failure")
            return original_snapshot(path)

    requests = []
    admission_clocks = []
    original_selection = local_entry._request_technical_execution_selection

    def select_execution(**kwargs):
        admission_clocks.append(time.monotonic())
        return original_selection(**kwargs)

    monkeypatch.setattr(local_entry, "_request_technical_execution_selection", select_execution)
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
        capabilities = next(json.loads(x["content"])["current_task_state"]["experiment_capabilities"]
            for x in reversed(payload["messages"]) if x["role"] == "user" and '"current_task_state"' in x["content"])
        assert capabilities["registered_tests"]["status"] == "configured"
        assert capabilities["probe_execution"]["status"] == "unavailable"
        assert capabilities["probe_submission"]["executes_code"] is False
        tool_outputs = [json.loads(x["content"]) for x in payload["messages"] if x["role"] == "tool"]
        observed = [row.get("value", row) for row in tool_outputs]
        last = observed[-1] if observed else None
        if exercise_feedback and len(requests) == 6:
            assert "check_feedback_read" in tools
            assert last["candidate_ref"] == memory["parent"]
            assert last["candidate_source_id"] == memory["failed_source_id"]
            assert last["check_ref"] and last["feedback_ref"]
            failure = last["observations"][0]
            assert failure["test_name"] == "double"
            assert failure["details"].startswith("Expected Double(3) to equal 6, but found 3.")
            assert failure["details_truncated"] is True
            assert failure["details_total_chars"] == len(memory["failure_details"])
            assert failure["source_locations"] == [{"path": "Library/Arithmetic.cs", "line": 1,
                "origin": "reported_public_stack_trace"}]
            assert failure["raw_report_ref"]["sha256"]
            memory["failed_check"] = last
            return httpx.Response(200, request=request, json={"model": model,
                "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "check_feedback_read", "arguments": {"feedback_ref": last["feedback_ref"],
                        "offset": 0, "limit": 8000}}}]}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
        if exercise_feedback and len(requests) == 7:
            assert last["candidate_ref"] == memory["parent"]
            assert last["candidate_source_id"] == memory["failed_source_id"]
            assert last["check_ref"] == memory["failed_check"]["check_ref"]
            assert last["offset"] == 0 and last["truncated"] is False
            assert last["next_offset"] is None
            full_failure = json.loads(last["text"])[0]
            assert full_failure["details"] == memory["failure_details"]
            assert full_failure["name"] == "double" and full_failure["actual"] == 3
            assert last["total_chars"] == len(last["text"])
            last = memory["failed_check"]
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
        if exercise_stale_view and len(requests) == 12:
            assert last["status"] == "rejected_not_executed"
            assert "historical revision" in str(last["argument_errors"])
            assert memory["parent"] in str(last["argument_errors"])
            return httpx.Response(200, request=request, json={"model": model,
                "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {
                    "name": "candidate_read", "arguments": {"parent_revision_ref": memory["parent"],
                        "path": "Library/Unused.cs"}}}]}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
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
                - int(exercise_correction and len(requests) > 4)
                - (2 if exercise_stale_view and len(requests) > 12 else 0)
                - int(exercise_feedback and len(requests) > 6))
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
                memory["original_unused_view"] = memory["Unused.cs"]["source_view_ref"]
            name, args = "candidate_submit", {"parent_revision_ref": memory["parent"], "edits": [{
                "binding_id": memory["unused_binding"], "source_view_ref": memory["Unused.cs"]["source_view_ref"],
                "before_text": "Marker = 0", "after_text": "Marker = 1", "rationale": "Intentionally wrong controlled hypothesis"}]}
            del args["edits"][0]["rationale"]
            if exercise_correction and len(requests) == 4:
                del args["edits"][0]["before_text"]
        elif turn in {5, 12}:
            if turn == 5:
                assert last["edit_explanations"] == [{"binding_id": memory["unused_binding"], "status": "not_expressed", "rationale": None}]
                if exercise_feedback:
                    memory["failed_source_id"] = last["candidate_source_id"]
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
            if exercise_stale_view and len(requests) == 11:
                args["edits"][0]["source_view_ref"] = memory["original_unused_view"]
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
    if expect_postcheck_failure:
        assert code != 0 and pending["current_stage"] == "blocked", pending
        assert pending.get("status") != "awaiting_delivery"
        assert pending["task_agent_outcome"]["status"] == "unknown", pending
        assert pending["task_agent_outcome"]["check_ref"] is None
        assert pending["task_agent_outcome"]["whole_task_delivered"] is False
        assert "patch_proposal" not in pending["artifacts"]
        assert len(requests) == 5, "Unconfirmed business observations must not reach another model request"
        assert pending.get("actual_write_performed") is not True
        assert not pending.get("v2_delivery_checkpoint_ref")
        assert {p.name: p.read_bytes() for p in (source / "Library").glob("*.cs")} == baseline
        errors = [parent / "integrity-error.json" for parent in memory["test_report_path"].parents
                  if (parent / "integrity-error.json").is_file()]
        assert len(errors) == 1
        integrity = json.loads(errors[0].read_text(encoding="utf-8"))
        assert integrity["status"] == "unknown" and integrity["error_type"] == "OSError"
        assert "controlled post-execution source integrity failure" in integrity["error"]
        assert integrity["context"]["candidate_ref"] == memory["parent"]
        assert set(integrity["stage_receipts"]) == {"build", "test_build", "test_execution", "raw_test_result"}
        facts = {}
        for stage, reference in integrity["stage_receipts"].items():
            raw = Path(reference["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
            facts[stage] = json.loads(raw)
            assert facts[stage]["context"] == integrity["context"]
            for log_ref in facts[stage]["logs"].values():
                assert hashlib.sha256(Path(log_ref["path"]).read_bytes()).hexdigest() == log_ref["sha256"]
        assert facts["build"]["execution"]["build_exit_code"] == 0
        assert facts["build"]["commands"]["build"]
        assert facts["test_execution"]["execution"]["exit_code"] == 0
        assert facts["test_execution"]["execution"]["stdout"] == "controlled test"
        assert facts["test_execution"]["commands"]["test"]
        report_ref = facts["raw_test_result"]["raw_report_ref"]
        report_raw = Path(report_ref["path"]).read_bytes()
        assert hashlib.sha256(report_raw).hexdigest() == report_ref["sha256"]
        assert json.loads(report_raw)["tests"][0]["passed"] is False
        assert not (errors[0].parent / "report.json").exists()
        assert not (errors[0].parent / "integrity.json").exists()
        return
    assert code == 0 and pending["status"] == "awaiting_delivery", {
        "result_file": str(short_workspace / "v2-entry-result.json"), "status": pending.get("status"),
        "reason": pending.get("reason"), "outcome": pending.get("task_agent_outcome")}
    assert len(requests) == (13 + (3 if exercise_experiments else 0) + int(exercise_discovery)
                             + int(exercise_correction) + (2 if exercise_stale_view else 0) + int(exercise_feedback))
    if exercise_feedback:
        assert pending["model_call_budget"]["consumed"] == 14
    if exercise_correction:
        assert pending["model_call_budget"]["consumed"] == 14
    if exercise_stale_view:
        assert pending["model_call_budget"]["consumed"] == 15
    assert {p.name: p.read_bytes() for p in (source / "Library").glob("*.cs")} == baseline
    root = Path(pending["run_root"])
    timing_origin = json.loads((root / "entry-timing.json").read_text(encoding="utf-8"))
    assert timing_origin["entry_clock"]["monotonic"] <= admission_clocks[0]
    timing = json.loads(Path(pending["task_timing_ref"]["path"]).read_text(encoding="utf-8"))
    assert timing["event"] == "product_return" and timing["status"] == "awaiting_delivery"
    assert Path(timing["origin_ref"]["path"]) == root / "entry-timing.json"
    assert timing["clock_basis"] == "same_process_monotonic" and timing["timing_status"] == "observed"
    assert timing["duration_ms"] == int((timing["observed_clock"]["monotonic"] - timing_origin["entry_clock"]["monotonic"]) * 1000)
    assert timing["whole_task_finished"] is False
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


def test_normal_cli_historical_view_requires_refresh_before_revision(repository_workspace, monkeypatch, capsys):
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(
        repository_workspace, monkeypatch, capsys, False, exercise_stale_view=True)


def test_normal_cli_public_failure_details_reach_next_request_and_revision(repository_workspace, monkeypatch, capsys):
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(
        repository_workspace, monkeypatch, capsys, False, exercise_feedback=True)
    report = _inspect_retained_entry(repository_workspace, monkeypatch, capsys)
    chain = report["executions"][0]
    assert len(chain["turns"]) == 14
    assert all(row["status"] == "executed_bound" for row in chain["actions"])
    feedback = next(row for row in chain["feedback_links"] if row["turn"] == 6)
    assert feedback["matches_recorded_tool_result"] and feedback["included_in_provider_request"]
    assert feedback["candidate_ref"] and feedback["candidate_source_id"] and feedback["check_ref"]
    assert any(row["schema"] == "task-agent-candidate.v2" for row in report["artifacts"])
    assert any(link["integrity"] == "verified_artifact_semantics"
        for link in report["links"] if link["relation"] == "artifact_semantic_reference")


def test_normal_cli_postcheck_failure_retains_stage_facts_without_delivery(repository_workspace, monkeypatch, capsys):
    test_v2_normal_cli_empty_registry_retarget_and_product_handoff(
        repository_workspace, monkeypatch, capsys, False, expect_postcheck_failure=True)
    report = _inspect_retained_entry(repository_workspace, monkeypatch, capsys)
    assert report["terminal"]["current_stage"] == "blocked"
    assert report["executions"][0]["status"] == "unknown"
    assert len(report["executions"][0]["turns"]) == 5
    assert not report["executions"][0]["feedback_links"]
    assert any(row["stage"] == "test_execution" for row in report["stage_facts"])
    assert any(row["status"] == "unknown" for row in report["stage_facts"])


def _inspect_retained_entry(workspace, monkeypatch, capsys):
    from scripts import dsh_run_request

    pending = json.loads((workspace / "v2-entry-result.json").read_text(encoding="utf-8"))
    config = workspace / "inspect-config.json"
    config.write_text(json.dumps({"arguments": ["--workspace-root", str(workspace / "runs")]}), encoding="utf-8")
    root = workspace / "runs" / pending["run_id"]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(dsh_run_request.cli, "main", lambda *_: pytest.fail("Evidence inspection dispatched execution"))
    assert dsh_run_request.main(["--execution-config", str(config), "--inspect-run", pending["run_id"],
        "--view", "evidence", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["query_effects"] == {"model_calls": 0, "builds": 0, "tests": 0, "recoveries": 0, "writes": 0}
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
    return report
