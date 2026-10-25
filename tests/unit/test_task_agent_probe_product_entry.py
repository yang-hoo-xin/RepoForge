"""Normal entry with scripted decisions and controlled or explicitly enabled container probes.

为什么要有它：真实实验反馈、恢复和学习消费要穿过产品入口；脚本化选择不能计为模型自主收益。
"""
import json
import os
from copy import deepcopy
from pathlib import Path

import httpx
import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain import repository_probe_runner as probe_module
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.domain.repository_probe_runner import (
    RepositoryProbeRunner,
    validate_probe_runner_config,
)
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.planning.learning_context import expand_learning_content
from src.mod_agent.planning.strategy_experience import StrategyExperience
from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
from src.mod_agent.task_agent_v2_experiments import _source
from src.mod_agent.task_agent_v2_recovery import load_task_agent_checkpoint, resume_v2_task_agent
from tests.unit.test_csharp_repository_backend import _fake_execution, _profile
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)
from tests.unit.test_strategy_learning_request import question

PROBE = '''using System;
using System.Text.Json;
class Program {
  static int Main() {
    var actual = Arithmetic.Double(3);
    Console.WriteLine(JsonSerializer.Serialize(new {input = 3, expected = 6, actual}));
    return actual == 6 ? 0 : 1;
  }
}
'''


def test_probe_actual_feedback_recovery_and_episode(repository_workspace, monkeypatch, capsys, record_property,
                                                   interrupt_name="experiment_run"):
    real = os.environ.get("REPOFORGE_REAL_PROBE_ENTRY") == "1"
    requested_root = os.environ.get("REPOFORGE_PROBE_WORKSPACE")
    root = Path(requested_root).resolve() if requested_root else repository_workspace
    if requested_root:
        root.mkdir(parents=True, exist_ok=False)
    if real:
        config_path = Path(os.environ["REPOFORGE_PROBE_RUNNER_CONFIG"]).resolve(strict=True)
        runner_config = validate_probe_runner_config(json.loads(config_path.read_text(encoding="utf-8")))
        assert runner_config["target_framework"] == "net8.0"
        assert runner_config["project_path"] == "Library/Library.csproj"
    else:
        runner_config = {"kind": "docker_sdk_project_v1", "image": "mcr.microsoft.com/dotnet/sdk@sha256:" + "a" * 64,
            "sdk_version": "8.0.100", "target_framework": "net8.0", "project_path": "Library/Library.csproj"}
    profile, source = _profile(root, real=real)
    profile_ref = _save(root / "repository-probe-profile.json", {**profile.data, "probe_runner": runner_config})
    before = {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    memory = StrategyExperience(ArtifactStore(root / "experience"))
    config = _save(root / "experience-config.json", {"schema_version": "task_agent_experience_config_v1",
        "store_root": str(memory.store.root), "snapshot_ref": None, "extract_on_terminal": True,
        "episode_catalog": {"catalog_id": "probe-feedback"}, "learning_request_ref": question(root),
        "requested_strategy_scope": {"runtime": "dotnet", "has_candidate": False}})
    owners, requests, learning, retained, probe_calls, container_calls, qualifications = [], [], [], [], [], [], []
    saved = {}
    original_init = TaskAgentExecutionV2.__init__

    def capture(owner, **kwargs):
        original_init(owner, **kwargs)
        owners.append(owner)

    monkeypatch.setattr(TaskAgentExecutionV2, "__init__", capture)
    if not real:
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

        def qualify(runner, profile_reference):
            capability = {"status": "qualified", "profile_ref": profile_reference, "execution_kind": "controlled",
                "runner_identity": {"kind": "controlled_fixture_not_container"},
                "boundaries": {"filesystem": "isolated", "network": "denied", "process": "isolated"}}
            return {**capability, "evidence_refs": [runner.state.save("probe-qualification", capability)]}

        def controlled_probe(runner, *, state, source_revision_ref, capability, operation_id, **_kwargs):
            ref, identity, _files = _source(state, source_revision_ref)
            observed = {"compiled": True, "stdout": json.dumps({"input": 3, "expected": 6, "actual": 3}), "stderr": ""}
            return {"execution_origin": "executed", "execution_kind": "controlled", "source_revision_ref": ref,
                "source_identity": identity, "runner_identity": capability["runner_identity"], "exit_code": 1,
                "passed": False, "observations": observed,
                "logs_ref": state.save("probe-execution", {"operation_id": operation_id, "observations": observed,
                    "scope": "controlled_fixture_no_process_or_container"})}

        monkeypatch.setattr(RepositoryProbeRunner, "verify_capability", qualify)
        monkeypatch.setattr(RepositoryProbeRunner, "run_probe", controlled_probe)
    original_probe = RepositoryProbeRunner.run_probe

    def count_probe(runner, **kwargs):
        probe_calls.append(kwargs["operation_id"])
        return original_probe(runner, **kwargs)

    monkeypatch.setattr(RepositoryProbeRunner, "run_probe", count_probe)
    original_qualify = RepositoryProbeRunner.verify_capability
    original_container = probe_module.execute_sdk_container

    def count_qualification(runner, profile_reference):
        qualifications.append(profile_reference)
        return original_qualify(runner, profile_reference)

    def count_container(**kwargs):
        container_calls.append({"root": str(kwargs["root"]), "runtime": kwargs["runtime"]})
        return original_container(**kwargs)

    monkeypatch.setattr(RepositoryProbeRunner, "verify_capability", count_qualification)
    monkeypatch.setattr(probe_module, "execute_sdk_container", count_container)
    original_receipt = TaskAgentExecutionV2.record_v2_tool_receipt

    def interrupt_after_receipt(owner, origin, output, session):
        reference = original_receipt(owner, origin, output, session)
        if not retained and origin["name"] == interrupt_name:
            retained.append((owner.store.root, owner.v2_checkpoint_ref, owner.budget_snapshot()["consumed"]))
            saved["interrupted_tool_receipt_ref"] = reference
            raise KeyboardInterrupt()
        return reference

    monkeypatch.setattr(TaskAgentExecutionV2, "record_v2_tool_receipt", interrupt_after_receipt)
    model = "qwen3:4b-instruct"

    def transport(_transport, request):
        if request.method == "GET":
            payload = {"version": "controlled-probe-entry"} if request.url.path == "/api/version" else {
                "models": [{"name": model, "model": model, "digest": "1" * 64,
                    "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}]}
            return httpx.Response(200, request=request, json=payload)
        payload = json.loads(request.content)
        if not payload.get("tools"):
            learning.append(payload)
            content = {"status": "insufficient_information", "reason": "A scripted probe does not prove autonomous transfer.",
                "evidence_gaps": [{"status": "not_observed", "description": "No independent transfer comparison."}]}
            message = {"role": "assistant", "content": json.dumps(content)}
        else:
            requests.append(payload)
            context = next(json.loads(row["content"]) for row in payload["messages"]
                if row["role"] == "user" and '"task_agent_v2"' in row["content"])
            outputs = [json.loads(row["content"]) for row in payload["messages"] if row["role"] == "tool"]
            last = outputs[-1].get("value", outputs[-1]) if outputs else None
            turn = len(requests)
            if turn == 1:
                saved["revision"] = context["parent_revision_ref"]
                name, args = "source_read", {"parent_revision_ref": saved["revision"], "path": "Library/Arithmetic.cs"}
            elif turn == 2:
                name, args = "probe_submit", {"source_revision_ref": saved["revision"],
                    "files": [{"relative_path": "Program.cs", "content": PROBE}], "entrypoint_id": "csharp_console",
                    "rationale": "Measure the original Double result without modifying it.",
                    "expected_observation": "Input 3 should yield 6; retain the actual value."}
            elif turn == 3:
                assert last["executed"] is False and last["probe_artifact_ref"]
                assert last["path_mapping"] == [{"submitted_path": "Program.cs", "artifact_path": "probe/Program.cs"}]
                saved["probe_ref"] = last["probe_artifact_ref"]
                name, args = "experiment_run", {"source_revision_ref": saved["revision"], "kind": "temporary_probe",
                    "probe_artifact_ref": last["probe_artifact_ref"], "hypothesis": "The baseline returns 3 instead of 6.",
                    "expected_observation": "The probe prints actual=3 and exits 1."}
            else:
                assert turn == 4
                saved["feedback"] = deepcopy(last)
                assert last["executed"] is True and last["status"] == "failed", last
                assert last["grants_acceptance"] is False and last["verification_scope"] == "exploration_only"
                assert last["observations"]["execution_kind"] == ("isolated" if real else "controlled")
                observed = last["observations"]["observations"]
                assert observed["compiled"] is True and json.loads(observed["stdout"]) == {"input": 3, "expected": 6, "actual": 3}
                name, args = "binding_add", {"expected_registry_ref": "controlled-stale-registry",
                    "source_view_ref": "controlled-missing-view", "target_selector": "Library/Arithmetic.cs",
                    "rationale": "End the scripted observation task without editing.", "expected_validation": "No candidate produced."}
            message = {"role": "assistant", "content": "Scripted product-entry verification, not autonomous decision evidence.",
                "tool_calls": [{"id": "probe-entry-" + str(turn), "function": {"name": name, "arguments": args}}]}
        return httpx.Response(200, request=request, json={"model": model, "message": message, "done": True,
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    arguments = ["run-request", "--target-project", str(source), "--repository-profile", profile_ref["path"],
        "--workspace-root", str(root / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536",
        "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2", "--defer-approval",
        "--task-agent-experience-config", config["path"], "--confirm-isolated-write", "--allow-build", "--",
        "Observe Arithmetic.Double(3) with a temporary probe; retain its actual result without changing product source."]
    with pytest.raises(KeyboardInterrupt):
        cli.main(arguments)
    capsys.readouterr()
    expected_turns = 3 if interrupt_name == "experiment_run" else 2
    assert len(probe_calls) == (1 if interrupt_name == "experiment_run" else 0)
    assert len(requests) == expected_turns and not learning
    container_count, qualification_count = len(container_calls), len(qualifications)
    if real:
        assert container_count == 4
    monkeypatch.setattr(TaskAgentExecutionV2, "record_v2_tool_receipt", original_receipt)
    store, checkpoint_ref, budget = retained[0]
    assert budget == expected_turns
    receipt_path = Path(saved["interrupted_tool_receipt_ref"]["path"])
    original_receipt_bytes = receipt_path.read_bytes()
    moved = receipt_path.with_suffix(".unconfirmed-fixture")
    assert receipt_path.resolve().is_relative_to(root) and moved.resolve().is_relative_to(root)
    receipt_path.rename(moved)
    try:
        with pytest.raises(ValueError, match="TOOL_OUTCOME_UNCONFIRMED"):
            load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=checkpoint_ref, reconcile=True)
    finally:
        moved.rename(receipt_path)
    original_count = len(probe_calls)
    changed = json.loads(original_receipt_bytes)
    changed["output"]["source_identity"] = "forged" if interrupt_name == "probe_submit" else "not-an-experiment-field"
    receipt_path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="PROBE_SUBMISSION_RECEIPT_CHANGED|EXPERIMENT_RECEIPT_CHANGED"):
            load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=checkpoint_ref, reconcile=True)
    finally:
        receipt_path.write_bytes(original_receipt_bytes)
    if interrupt_name == "experiment_run":
        original_output = json.loads(original_receipt_bytes)["output"]
        probe = owners[-1]._v2_state.load(saved["probe_ref"])
        for reference in (saved["probe_ref"], probe["profile_ref"], original_output["observations"]["logs_ref"]):
            artifact_path = Path(store) / reference
            original_artifact = artifact_path.read_bytes()
            moved_artifact = artifact_path.with_suffix(".missing-fixture")
            assert artifact_path.resolve().is_relative_to(root) and moved_artifact.resolve().is_relative_to(root)
            artifact_path.rename(moved_artifact)
            try:
                with pytest.raises((ValueError, FileNotFoundError)):
                    load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=checkpoint_ref, reconcile=True)
            finally:
                moved_artifact.rename(artifact_path)
            artifact_path.write_text("{}", encoding="utf-8")
            try:
                with pytest.raises(ValueError, match="artifact_hash_mismatch"):
                    load_task_agent_checkpoint(artifact_store_root=store, checkpoint_ref=checkpoint_ref, reconcile=True)
            finally:
                artifact_path.write_bytes(original_artifact)
    assert len(probe_calls) == original_count and len(requests) == expected_turns and not learning
    resumed = resume_v2_task_agent(artifact_store_root=store, checkpoint_ref=checkpoint_ref)
    owner = owners[-1]
    assert len(probe_calls) == 1 and len(requests) == 4 and len(learning) == 1
    if interrupt_name == "experiment_run":
        assert len(container_calls) == container_count and len(qualifications) == qualification_count
    receipt = resumed["strategy_learning"]
    assert receipt["status"] == "insufficient_information", receipt
    assert receipt["catalog_registration"]["status"] == "registered"
    assert owner._v2_state.current_revision_ref == owner._v2_state.contract["baseline_manifest_ref"]
    assert before == {path.relative_to(source).as_posix(): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert not owner._v2_session.public_checks
    episode = read_bound_json(receipt["episode_ref"])
    assert episode["applicability"]["has_candidate"] is False and not episode["public_counterexamples"]
    sent = expand_learning_content(json.loads(learning[0]["messages"][-1]["content"]))
    experiments = sent["episodes"][0]["experience"]["public_experiment"]
    assert any(row["value"] == saved["feedback"]["observations"] for row in experiments)
    repeated = resume_v2_task_agent(artifact_store_root=store, checkpoint_ref=owner.v2_checkpoint_ref)
    assert repeated["strategy_learning"] == receipt
    assert len(probe_calls) == 1 and len(requests) == 4 and len(learning) == 1
    assert resumed["cumulative_resources"]["model_calls"] == repeated["cumulative_resources"]["model_calls"] == 5
    result = {"scope": "scripted_agent_decisions_with_real_container_probe" if real else "controlled_no_container_or_SDK",
        "autonomous_model_gain_established": False, "business_acceptance_established": False,
        "probe_invocations": len(probe_calls), "scripted_decision_calls": len(requests), "scripted_learning_calls": len(learning),
        "container_invocations": container_calls, "qualification_checks": len(qualifications),
        "source_unchanged": True, "candidate_produced": False, "profile_ref": profile_ref,
        "original_interrupted_checkpoint_ref": checkpoint_ref, "artifact_store_root": str(store),
        "resumed_checkpoint_ref": repeated["v2_checkpoint_ref"], "episode_ref": receipt["episode_ref"],
        "catalog_selection_ref": receipt["catalog_selection_ref"], "learning_receipt_ref": receipt["receipt_ref"],
        "experiment_result_ref": saved["feedback"]["result_ref"],
        "actual_next_request_ref": _save(root / "actual-next-request.json", requests[3]),
        "actual_learning_request_ref": _save(root / "actual-learning-request.json", learning[0])}
    result_ref = _save(root / "probe-entry-result.json", result)
    record_property("probe_entry_result", json.dumps(result_ref))


def test_probe_submission_receipt_recovery_without_runner(repository_workspace, monkeypatch, capsys, record_property):
    if os.environ.get("REPOFORGE_REAL_PROBE_ENTRY") == "1":
        pytest.skip("The one selected real-container run uses the experiment receipt interruption.")
    test_probe_actual_feedback_recovery_and_episode(repository_workspace, monkeypatch, capsys, record_property,
        interrupt_name="probe_submit")
