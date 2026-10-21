"""Normal repository CLI and public recovery, with scripted model/approval and real SDK.

为什么要有它：模型替身不代表自主修复成绩，但实际代码、编译、测试与审批恢复必须穿过正常产品入口。
"""
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.proposal.observation_value_refs import restore_observation_values
from src.mod_agent.workflow.interrupts import ResumePayload
from tests.unit.test_csharp_repository_backend import _profile
from tests.unit.test_mod_agent_cli import _formal_coverage_model_response


def _json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


@pytest.fixture
def repository_workspace():
    # 为什么要有它：真实CLI验收的失败和运行原件需要复核，不能随普通单元测试临时目录自动清理。
    return Path(tempfile.mkdtemp(prefix="rfcscli-")).resolve()


@pytest.mark.skipif(os.environ.get("REPOFORGE_REAL_REPOSITORY_CLI_INTEGRATION") != "1",
    reason="Explicit real local SDK integration; model and human approval remain scripted fixtures.")
@pytest.mark.parametrize("technical_executor", ["task_agent", "workflow"])
def test_normal_cli_real_repository_repair_and_public_apply_build(repository_workspace, monkeypatch, capsys, technical_executor):
    short_workspace = repository_workspace
    profile, source = _profile(short_workspace, real=True)
    original = (source / "Library/Arithmetic.cs").read_bytes()
    responses = []
    state = {}
    model = "qwen3:4b-instruct"

    def reply(payload, *, content=None, tool=None, arguments=None):
        message = {"role": "assistant", "content": json.dumps(content) if content is not None else ""}
        if tool is not None:
            message["tool_calls"] = [{"id": f"scripted-repository-{len(responses)}", "type": "function",
                "function": {"name": tool, "arguments": arguments}}]
        return {"model": payload["model"], "message": message, "done": True, "done_reason": "stop",
            "prompt_eval_count": 10, "eval_count": 20}

    def transport(_transport, request):
        if request.method == "GET":
            value = {"version": "scripted-no-real-model"} if request.url.path == "/api/version" else {
                "models": [{"name": model, "model": model, "digest": "1" * 64,
                    "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}]}
            return httpx.Response(200, request=request, json=value)
        assert request.url.path == "/api/chat", request.url
        payload = json.loads(request.content)
        responses.append(payload)
        contexts = []
        for message in payload["messages"]:
            if message["role"] == "user" and message["content"].startswith("{"):
                context = json.loads(message["content"])
                if "patch_agent_context_encoding" in context:
                    context = local_entry._restore_patch_agent_context(context)
                contexts.append(context)
        tools = {item["function"]["name"] for item in payload.get("tools", [])}
        if "candidate_submit" not in tools:
            context = next(value for value in contexts if value.get("message_type") in {
                "product_requirement_projection", "product_requirement_findings"} or "allowed_edit_evidence_ids" in value
                or "required_change_units" in value)
            content = _formal_coverage_model_response(context)
            if context.get("message_type") == "product_requirement_projection":
                content["requirements"][0]["text"] = "Make Arithmetic.Double return twice its input."
                for row in content["requirements"]:
                    row["query_leads"] = ["Arithmetic", "Double"]
            elif "allowed_edit_evidence_ids" in context:
                chosen = [row["evidence_id"] for row in context["source_observations"]
                    if "class Arithmetic" in row.get("excerpt", "") and row["evidence_id"] in context["allowed_edit_evidence_ids"]]
                assert chosen
                content = {"decision": "select_targets", "selected_evidence_ids": chosen,
                    "rationale": "Scripted test selects the public library source.", "verification_questions": []}
            elif "required_change_units" in context:
                assert technical_executor == "workflow"
                attempt = state.get("workflow_patch_turn", 0) + 1
                state["workflow_patch_turn"] = attempt
                assert attempt <= 2
                changes = []
                for unit in context["required_change_units"]:
                    binding = unit["edit_binding_id"]
                    before = context["edit_binding_contexts"][binding]["source_excerpt"]
                    changes.append({"edit_binding_id": binding,
                        "after_text": before.replace("=> value;", "=> value + 1;" if attempt == 1 else "=> value * 2;"),
                        "rationale": "Scripted workflow repair on the original source after actual test feedback.",
                        "rollback_hint": "Restore the original source."})
                content = {"changes": changes}
            return httpx.Response(200, request=request, json=reply(payload, content=content))
        turn = state.setdefault("turn", 0) + 1
        state["turn"] = turn
        if turn == 1:
            contract = next(row for row in contexts if "required_change_units" in row)
            current = next(row for row in contexts if "context_id" in row)
            unit = contract["required_change_units"][0]
            binding = unit["edit_binding_id"]
            before = contract["edit_binding_contexts"][binding]["source_excerpt"]
            state.update(binding=binding, context_id=current["context_id"])
            arguments = {"context_id": current["context_id"], "patch_json": json.dumps({"changes": [{
                "edit_binding_id": binding, "after_text": before.replace("=> value;", "=> value + 1;"),
                "rationale": "Scripted first candidate intentionally fails the public test.", "rollback_hint": "Restore original bytes."}]})}
            tool = "candidate_submit"
        else:
            last = json.loads(next(row["content"] for row in reversed(payload["messages"]) if row["role"] == "tool"))
            value = restore_observation_values(last["value"])
            if turn in (2, 5):
                tool, arguments = "candidate_check", {"candidate_id": value["candidate_id"]}
            elif turn == 3:
                assert value["status"] == "behavior_failed", value
                assert any(row["observed"]["actual"] == 4 and row["expected"] == 6 for row in value["observations"])
                state["failed_candidate"] = value["candidate_id"]
                tool, arguments = "candidate_read", {"context_id": state["context_id"],
                    "version": value["candidate_id"], "path": "generated_mods/source/Library/Arithmetic.cs"}
            elif turn == 4:
                assert "value + 1" in value["text"]
                tool, arguments = "candidate_submit", {"context_id": state["context_id"],
                    "parent_candidate_id": value["version"], "parent_code_sha256": value["code_sha256"],
                    "edits": [{"edit_binding_id": state["binding"], "before_text": "=> value + 1;",
                        "after_text": "=> value * 2;", "rationale": "Scripted correction using the actual failed result."}]}
            else:
                assert turn == 6 and value["status"] == "passed", value
                tool, arguments = "finish", {"candidate_id": value["candidate_id"]}
        return httpx.Response(200, request=request, json=reply(payload, tool=tool, arguments=arguments))

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    executor_options = (["--technical-executor", "task_agent", "--task-agent-candidate-mode", "current"]
        if technical_executor == "task_agent" else [])
    code = cli.main(["run-request", "--target-project", str(source), "--repository-profile", profile.reference["path"],
        "--workspace-root", str(short_workspace / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536",
        "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1", *executor_options,
        "--defer-approval", "--confirm-isolated-write", "--allow-build",
        "--", "Make Arithmetic.Double return twice its integer input. Preserve its public signature and the public tests."])
    pending = json.loads(capsys.readouterr().out)
    _json_file(short_workspace / "integration-pending.json", pending)
    _json_file(short_workspace / "scripted-provider-requests.json", responses)
    assert code == 0 and pending["status"] == "awaiting_approval", json.dumps(pending, ensure_ascii=False)
    assert state.get("turn", 0) == (6 if technical_executor == "task_agent" else 0)
    assert state.get("workflow_patch_turn", 0) == (2 if technical_executor == "workflow" else 0)
    assert (source / "Library/Arithmetic.cs").read_bytes() == original
    root = Path(pending["run_root"])
    actual_checks = [json.loads(path.read_bytes()) for path in (root / "repository_candidates").rglob("report.json")]
    assert any(row["status"] == "behavior_failed" and row["checks"][0]["actual"] == 4 for row in actual_checks)
    assert any(row["status"] == "passed" and row["checks"][0]["actual"] == 6 for row in actual_checks)
    from src.mod_agent.evaluation.isolated_delivery_adapter import prepare_delivery
    prepared = prepare_delivery({"run_root": str(root), "trial": {"delivery": {
        "policy_id": "scripted-integration-fixture-not-human", "verification_selection": "current_proposal",
        "verifier_selection": "bound_worker_execution"}}})
    _json_file(short_workspace / "delivery-preparation.json", {
        "verifier_identity_binding": prepared["verifier_identity_binding"], "required_check_ids": prepared["required_check_ids"]})
    expected_origin = ("normal_product_repository_binding_and_native_candidate" if technical_executor == "task_agent"
        else "normal_product_repository_binding_and_workflow_candidate")
    assert prepared["verifier_identity_binding"]["source"] == expected_origin
    assert prepared["required_check_ids"] == ["double"]
    store = ArtifactStore(root / "a")
    checkpoint = ArtifactRef(**pending["workflow_checkpoint"])
    saved = store.semantic_read_immutable(checkpoint)["data"]
    declared = json.loads(saved["state_snapshot"]["runtime_summary"]["deferred_product_request_json"])
    assert declared["target_domain"] == "csharp_repository"
    assert declared["input_bindings"][-1]["binding_kind"] == "repository_build_and_test"
    approval = ResumePayload(resume_id="scripted-approval", run_id=pending["run_id"], request_id=pending["request_id"],
        feature_id=pending["feature_id"], stage="dry_run_apply", decision="approve",
        approved_by="scripted-test-fixture-not-human", approved_at="2026-10-20T00:00:00Z",
        artifact_refs=tuple(saved["artifact_refs"]), diff_hash=pending["diff_hash"])
    decision = {**pending["human_semantic_decision_binding"], "decision": "approve",
        "reviewer_identity": "scripted-test-fixture-not-human"}
    resume_path = _json_file(short_workspace / "approval.json", approval.to_dict())
    decision_path = _json_file(short_workspace / "decision.json", decision)
    assert cli.main(["apply-patch", "--resume-approval", "--artifact-store-root", str(store.root),
        "--feature-id", checkpoint.feature_id, "--checkpoint-path", checkpoint.path,
        "--checkpoint-hash", checkpoint.artifact_hash, "--resume-payload", resume_path,
        "--human-semantic-decision", decision_path]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["status"] == "applied_in_isolation", applied
    next_checkpoint = ArtifactRef(**applied["workflow_checkpoint"])
    after_apply = store.semantic_read_immutable(next_checkpoint)["data"]
    assert "repository_build_and_test" in after_apply["state_snapshot"]["runtime_summary"]["reviewed_apply"]
    build_payload = replace(approval, resume_id="scripted-build", stage="isolated_apply",
        artifact_refs=tuple(after_apply["artifact_refs"]))
    build_path = _json_file(short_workspace / "build.json", build_payload.to_dict())
    assert cli.main(["apply-patch", "--continue-build", "--allow-build", "--artifact-store-root", str(store.root),
        "--feature-id", next_checkpoint.feature_id, "--checkpoint-path", next_checkpoint.path,
        "--checkpoint-hash", next_checkpoint.artifact_hash, "--resume-payload", build_path]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["status"] == "built", built
    result = store.semantic_read_immutable(ArtifactRef(**built["build_result"]))["data"]
    assert result["claims_real_build_success"] is True
    verification = result["artifact_verification"]
    report = backend.read_bound_json(verification["repository_validation_ref"])
    assert report["status"] == "passed" and report["test_execution"]["status"] == "executed"
    assert report["checks"] and report["checks"][0]["actual"] == 6
    assert (source / "Library/Arithmetic.cs").read_bytes() == original
    _json_file(short_workspace / "integration-result.json", {"status": "passed",
        "scope": "real local SDK/build/tests and public CLI/Apply/Build; scripted model and scripted approval",
        "technical_executor": technical_executor,
        "pending": pending, "applied": applied, "built": built, "formal_test_report_ref": verification["repository_validation_ref"]})
