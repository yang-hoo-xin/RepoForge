"""Experience reaches the actual HTTP adapter on the normal repository CLI."""
import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from src.mod_agent import cli, local_entry
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.build.adapter import BuildCommandResult
from src.mod_agent.build.nuget import NuGetIsolatedBuildExecution
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.evaluation.strategy_adoption import record_strategy_withdrawal
from src.mod_agent.planning.strategy_experience import StrategyExperience
from src.mod_agent.task_agent_v2_execution import TaskAgentExecutionV2
from tests.unit.test_csharp_repository_backend import _fake_execution, _profile
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def _configuration(root, *, consume, extract):
    memory = StrategyExperience(ArtifactStore(root / "experience"))
    source = _save(root / "public-experience-source.json", {"observation": "A failed behavior was reproduced before repair."})
    strategy = {"applicability": {"runtime": "dotnet", "target_domain": "csharp_repository"},
        "suggested_action": "Read the implementation and compare the public counterexample before editing.",
        "expected_observation": "The selected statement explains the public behavior mismatch.",
        "counterevidence": ["The implementation already produces the expected public behavior."]}
    proposed = memory.record_strategy("public-reproduction", strategy, evidence_refs=[{
        "evidence_id": "public", "kind": "json_file", **source,
        "visibility": "public", "allowed_uses": ["strategy_learning"]}], origin={"kind": "human", "actor": "test"})
    policy = _save(root / "trial-policy.json", {"schema_version": "strategy_trial_policy_v1",
        "allowed_origin_kinds": ["human"], "allowed_domains": {"runtime": ["dotnet"]}})
    snapshot = memory.admit_trial(proposed, policy_ref=policy)
    ref = _save(root / "experience-config.json", {"schema_version": "task_agent_experience_config_v1",
        "store_root": str(memory.store.root), "snapshot_ref": asdict(snapshot) if consume else None,
        "extract_on_terminal": extract})
    return memory, ref


@pytest.mark.parametrize("mode", ["consume", "withdraw", "extract", "compile_extract", "rejected_extract",
    "discovery_extract", "discovery_tamper_extract", "discovery_environment_extract", "recheck_extract"])
def test_experience_through_normal_cli_actual_requests_and_terminal(repository_workspace, monkeypatch, capsys, mode):
    root = repository_workspace
    profile, source = _profile(root)
    extract_mode = mode.endswith("extract")
    memory, config = _configuration(root, consume=not extract_mode, extract=extract_mode)
    _fake_execution(monkeypatch)
    fake_build = backend.nuget._run_nuget_isolated_build
    def build(selected):
        if mode == "discovery_environment_extract" and "experiments" in selected.working_directory.parts:
            return NuGetIsolatedBuildExecution(1, None, "", "controlled restore unavailable", "", "", 1)
        source_file = selected.working_directory / "Library/Arithmetic.cs"
        if source_file.is_file() and "missingSymbol" in source_file.read_text():
            return NuGetIsolatedBuildExecution(0, 1, "restored", "", "error CS0103: missingSymbol is not defined", "", 1)
        result = fake_build(selected)
        artifact = selected.working_directory / "Library/bin/Release/net8.0/Library.dll"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"CONTROLLED NOT A REAL DLL")
        return result
    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_build", build)
    repeated_wrong_code_checks = 0
    def verify(selected, command):
        nonlocal repeated_wrong_code_checks
        text = (selected.working_directory / "Library/Arithmetic.cs").read_text()
        actual = 6 if "value * 2" in text else 4 if "value + 1" in text else 3
        if mode == "recheck_extract" and "value + 1" in text:
            repeated_wrong_code_checks += 1
            if repeated_wrong_code_checks > 1:
                actual = 6  # Controlled flaky oracle; this must not count as a code revision.
        Path(command[-1]).write_text(json.dumps({"schema_version": "csharp-test-report.v1",
            "tests": [{"name": "double", "passed": actual == 6, "expected": 6, "actual": actual}]}), encoding="utf-8")
        return BuildCommandResult(0, "controlled public test", "", 1)
    monkeypatch.setattr(backend.nuget, "_run_nuget_isolated_verification_command", verify)
    owners = []
    original_init = TaskAgentExecutionV2.__init__
    def configured(self, **kwargs):
        assert kwargs["experience_configuration_ref"] == config
        original_init(self, **kwargs)
        owners.append(self)
    monkeypatch.setattr(TaskAgentExecutionV2, "__init__", configured)
    if mode == "discovery_tamper_extract":
        original_learning = TaskAgentExecutionV2.complete_terminal_learning
        def tamper_experiment(owner):
            if not getattr(owner, "_test_experiment_tampered", False):
                call = next(row for row in owner._v2_session.actual_calls if row["name"] == "test_catalog")
                record = json.loads(Path(call["execution_receipt_ref"]["path"]).read_text(encoding="utf-8"))
                record["output"]["discovery_receipt"]["observations"]["observations"][0]["observed"]["actual"] = "fabricated observation"
                call["execution_receipt_ref"] = owner._save("controlled-tampered-experiment-receipt.json", record)
                owner._test_experiment_tampered = True
            return original_learning(owner)
        monkeypatch.setattr(TaskAgentExecutionV2, "complete_terminal_learning", tamper_experiment)
    model, requests, learning, saved = "qwen3:4b-instruct", [], [], {}
    def transport(_transport, request):
        if request.method == "GET":
            payload = {"version": "controlled-experience"} if request.url.path == "/api/version" else {
                "models": [{"name": model, "model": model, "digest": "1" * 64,
                    "details": {"context_length": 65536}, "capabilities": ["completion", "tools"]}]}
            return httpx.Response(200, request=request, json=payload)
        payload = json.loads(request.content)
        if not payload.get("tools"):
            learning.append(payload)
            content = json.dumps({"suggested_action": "Reproduce the public behavior before changing the expression.",
                "expected_observation": "The expression predicts the observed wrong value.",
                "counterevidence": ["The public reproduction passes before modification."]})
            return httpx.Response(200, request=request, json={"model": model, "message": {"role": "assistant", "content": content},
                "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
        requests.append(payload)
        views = [json.loads(row["content"])["strategy_experience"] for row in payload["messages"]
            if row["role"] == "user" and '"strategy_experience"' in row["content"]]
        if not extract_mode:
            assert len(views) == 1 and views[0]["strategies"][0]["suggested_action"].startswith("Read the implementation")
        else:
            assert not views
        context = next(json.loads(row["content"]) for row in payload["messages"]
            if row["role"] == "user" and '"task_agent_v2"' in row["content"])
        observations = [json.loads(row["content"])["value"] for row in payload["messages"] if row["role"] == "tool"]
        last = observations[-1] if observations else None
        turn = len(requests)
        if mode.startswith("discovery_"):
            if turn == 1:
                saved["parent"] = context["parent_revision_ref"]
                name, args = "test_catalog", {"source_revision_ref": saved["parent"], "execute_discovery": True}
            elif turn == 2:
                assert last.get("discovery_receipt", last)["executed"] is True
                name, args = "plan_update", {"plan": {"hypothesis": "The current expression may preserve the input.",
                    "next_experiment": "Compare public Double assertions against the source before editing."}}
            else:
                assert turn == 3
                name, args = "binding_add", {"expected_registry_ref": "controlled-stale-registry",
                    "source_view_ref": "controlled-missing-view", "target_selector": "Library/Arithmetic.cs",
                    "rationale": "Controlled terminal admission rejection after retained experiment.", "expected_validation": "Double test"}
        elif turn == 1:
            saved["parent"] = context["parent_revision_ref"]
            name, args = "source_read", {"parent_revision_ref": saved["parent"], "path": "Library/Arithmetic.cs"}
            if mode == "withdraw":
                withdrawal = _save(root / "withdrawal.json", {"schema_version": "strategy_withdrawal_v1",
                    "strategy_id": "public-reproduction", "actor": "user", "reason": "Controlled withdrawal", "target_state": "disabled"})
                current = memory.current_strategy("public-reproduction")
                memory.transition(current, record_strategy_withdrawal(memory, current, withdrawal_ref=withdrawal))
        elif turn == 2:
            saved["view"] = last["source_view_ref"]
            name, args = "binding_add", {"expected_registry_ref": context["binding_registry_ref"],
                "source_view_ref": saved["view"], "target_selector": "Library/Arithmetic.cs",
                "rationale": "Read public implementation", "expected_validation": "Double test"}
        elif mode == "recheck_extract" and turn in {5, 6}:
            if turn == 5:
                assert last["status"] == "behavior_failed"
                name, args = "candidate_check", {"candidate_ref": saved["parent"]}
            else:
                assert last["status"] == "passed"
                name, args = "finish", {"candidate_ref": saved["parent"], "check_ref": last["check_ref"]}
        elif turn in {3, 6}:
            if turn == 3:
                saved["binding"] = last["binding_id"]
            else:
                saved["view"] = last["source_view_ref"]
            before = "=> value;" if turn == 3 else "=> value + 1;"
            after = "=> value + 1;" if turn == 3 and mode in {"extract", "recheck_extract"} else "=> value * 2;"
            if mode == "compile_extract":
                after = "=> missingSymbol;"
            name, args = "candidate_submit", {"parent_revision_ref": saved["parent"], "edits": [{
                "binding_id": saved["binding"], "source_view_ref": saved["view"], "before_text": before,
                "after_text": after, "rationale": "Controlled repair choice"}]}
            if mode == "rejected_extract":
                args["parent_revision_ref"] = "not-the-current-parent"
        elif turn in {4, 7}:
            saved["parent"] = last["candidate_ref"]
            name, args = "candidate_check", {"candidate_ref": saved["parent"]}
        elif turn == 5 and mode == "extract":
            assert last["status"] == "behavior_failed"
            name, args = "candidate_read", {"parent_revision_ref": saved["parent"], "path": "Library/Arithmetic.cs"}
        elif turn == 5 and mode == "compile_extract":
            assert last["status"] == "behavior_failed" and not last["verification_summary"]["test_command_executed"]
            name, args = "candidate_submit", {"parent_revision_ref": "not-the-current-parent", "edits": [{
                "binding_id": saved["binding"], "source_view_ref": saved["view"], "before_text": "=> missingSymbol;",
                "after_text": "=> value * 2;", "rationale": "Controlled stale-parent rejection"}]}
        else:
            assert last["status"] == "passed"
            name, args = "finish", {"candidate_ref": saved["parent"], "check_ref": last["check_ref"]}
        return httpx.Response(200, request=request, json={"model": model, "message": {"role": "assistant", "content": "",
            "tool_calls": [{**({"id": f"controlled-recheck-{turn}"} if mode == "recheck_extract" else {}),
                "function": {"name": name, "arguments": args}}]}, "done": True,
            "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    code = cli.main(["run-request", "--target-project", str(source), "--repository-profile", profile.reference["path"],
        "--workspace-root", str(root / "runs"), "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", model, "--ollama-context-window", "65536",
        "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2", "--task-agent-experience-config", config["path"], "--defer-approval",
        "--confirm-isolated-write", "--allow-build", "--", "Double the integer input, preserving the public signature."])
    result = json.loads(capsys.readouterr().out)
    assert owners, result
    owner = owners[0]
    execution_ref = owner.execution_receipt().execution_ref
    original = Path(execution_ref["path"]).read_bytes()
    if mode == "withdraw":
        assert len(requests) == 1
        assert owner._v2_state.load(owner._v2_state.registry_ref)["bindings"] == {}
        assert not owner._v2_session.public_checks
        assert result.get("status") != "awaiting_delivery"
        return
    if mode not in {"compile_extract", "rejected_extract"} and not mode.startswith("discovery_"):
        assert code == 0 and result["status"] == "awaiting_delivery", result
    receipt = owner.complete_terminal_learning()
    assert Path(execution_ref["path"]).read_bytes() == original
    if mode in {"extract", "compile_extract", "discovery_extract", "recheck_extract"}:
        assert receipt["status"] == "completed", receipt
        assert len(learning) == 1
        assert receipt["budget_after"]["consumed"] == owner.execution_receipt().budget_after["consumed"] + 1
        assert receipt["known_usage"][0]["total_tokens"] == 30
        assert memory.read(receipt["strategy_ref"])["origin"]["kind"] == "model"
        assert memory.read(receipt["strategy_ref"])["state"] == "proposed"
        assert owner.complete_terminal_learning() == receipt and len(learning) == 1
        expected_kind = "public_compilation_counterexample" if mode == "compile_extract" else "public_business_counterexample"
        learned_strategy = memory.read(receipt["strategy_ref"])["strategy"]
        assert learned_strategy["applicability"]["counterexample_kind"] == expected_kind
        trial_policy = _save(root / "automatic-trial-policy.json", {"schema_version": "strategy_trial_policy_v1",
            "allowed_origin_kinds": ["model"], "allowed_domains": {"counterexample_kind": [expected_kind]}})
        snapshot = memory.admit_trial(receipt["strategy_ref"], policy_ref=trial_policy)
        facts = owner._experience.context(owner._v2_session)
        assert memory.select(snapshot, facts)["selected"]
        other_kind = "public_business_counterexample" if mode == "compile_extract" else "public_compilation_counterexample"
        assert not memory.select(snapshot, {**facts, "counterexample_kind": other_kind})["selected"]
        assert not memory.select(snapshot, {key: value for key, value in facts.items() if key != "counterexample_kind"})["selected"]
        if mode == "compile_extract":
            episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
            assert episode["public_counterexamples"][0]["kind"] == "public_compilation_counterexample"
            assert episode["information_gain"] == ["public_counterexample"]
            assert episode["applicability"]["has_public_counterexample"] is True
            assert not owner._v2_session.public_checks[0]["report"]["verification_summary"]["test_command_executed"]
        if mode == "discovery_extract":
            assert not owner._v2_session.public_checks
            episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
            assert episode["information_gain"] == ["public_counterexample"]
            assert episode["public_counterexamples"][0]["scope"] == "exploration_only"
            learning_payload = json.loads(learning[0]["messages"][-1]["content"])
            observed = learning_payload["observations"]
            assert any(row["value"] == "test_catalog" for row in observed)
            assert any(isinstance(row["value"], dict) and row["value"].get("execute_discovery") is True for row in observed)
            assert any(isinstance(row["value"], dict) and "plan" in row["value"] for row in observed)
            assert any(row["value"] == "exploration_only" for row in observed)
            assert any(row["locator"] == ["grants_acceptance"] and row["value"] is False for row in observed)
            assert all(source["visibility"] == "public" for source in episode["evidence_refs"])
        if mode in {"extract", "recheck_extract"}:
            episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
            assert ("effective_revision" in episode["information_gain"]) is (mode == "extract")
            if mode == "recheck_extract":
                assert owner._v2_session.public_checks[0]["report"]["candidate_source_id"] == owner._v2_session.public_checks[1]["report"]["candidate_source_id"]
    elif mode == "rejected_extract":
        assert receipt["status"] == "no_information_gain" and not learning
        assert not owner._v2_session.public_checks
        assert owner._experience.context(owner._v2_session)["has_public_counterexample"] is False
        assert owner._experience.context(owner._v2_session)["counterexample_kind"] == "none"
    elif mode == "discovery_tamper_extract":
        assert receipt["status"] == "failed" and "EXPERIMENT_RESULT_CHANGED" in receipt["message"]
        assert not learning and receipt["model_calls"] == 0
    elif mode == "discovery_environment_extract":
        assert receipt["status"] == "no_information_gain" and not learning
        assert owner._experience.context(owner._v2_session)["counterexample_kind"] == "none"
    else:
        assert receipt["status"] == "not_enabled"
        assert len(list(owner.output_root.glob("experience-delivery-*.json"))) == len(requests)
