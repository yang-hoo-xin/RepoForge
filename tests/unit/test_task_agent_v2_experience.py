"""Experience reaches the actual HTTP adapter on the normal repository CLI."""
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

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
from src.mod_agent.task_agent_v2_experience import TaskAgentV2Experience
from tests.unit.test_csharp_repository_backend import _fake_execution, _profile
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def _configuration(root, *, consume, extract, trial=False, strategy_conditions=None):
    memory = StrategyExperience(ArtifactStore(root / "experience"))
    source = _save(root / "public-experience-source.json", {"observation": "A failed behavior was reproduced before repair."})
    strategy = {"applicability": {"runtime": "dotnet", "target_domain": "csharp_repository", **(strategy_conditions or {})},
        "suggested_action": "Read the implementation and compare the public counterexample before editing.",
        "expected_observation": "The selected statement explains the public behavior mismatch.",
        "counterevidence": ["The implementation already produces the expected public behavior."]}
    proposed = memory.record_strategy("public-reproduction", strategy, evidence_refs=[{
        "evidence_id": "public", "kind": "json_file", **source,
        "visibility": "public", "allowed_uses": ["strategy_learning"]}], origin={"kind": "human", "actor": "test"})
    policy = _save(root / "trial-policy.json", {"schema_version": "strategy_trial_policy_v1",
        "allowed_origin_kinds": ["human"], "allowed_domains": {"runtime": ["dotnet"]}})
    snapshot = memory.admit_trial(proposed, policy_ref=policy)
    config = {"schema_version": "task_agent_experience_config_v1",
        "store_root": str(memory.store.root), "snapshot_ref": asdict(snapshot) if consume else None,
        "extract_on_terminal": extract}
    if trial:
        from src.mod_agent.evaluation.memory_comparison import INVARIANTS
        from src.mod_agent.model_qualification import canonical_hash

        task_source = _save(root / "trial-source.json", {"task_id": "double", "task_family": "controlled-arithmetic",
            "source_version": "controlled-v1"})
        exposure = _save(root / "trial-exposure.json", {"task_id": "double", "source_ref": task_source,
            "attempt": 1, "exposure": "exposed"})
        evaluator = _save(root / "trial-evaluator.json", {"kind": "controlled-placeholder-not-qualified"})
        invariants = {key: {"fixture": key} for key in INVARIANTS | {"feedback"}}
        invariants["judge"] = {"evaluator_ref": evaluator}
        protocol = _save(root / "trial-protocol.json", {"schema_version": "strategy_trial_protocol_v1",
            "batch_complete": True, "snapshot_ref": asdict(snapshot), "invariants": invariants,
            "invariants_sha256": canonical_hash(invariants), "task_families": ["controlled-arithmetic"],
            "planned_pairs": [{"task_id": "double", "task_family": "controlled-arithmetic", "attempt": 1,
                "source_ref": task_source, "exposure_ref": exposure,
                "current_context": {"runtime": "dotnet", "target_domain": "csharp_repository"}}]})
        config["trial_binding_ref"] = _save(root / "trial-binding.json", {
            "schema_version": "strategy_trial_run_binding_v1", "protocol_ref": protocol,
            "task_id": "double", "attempt": 1, "arm": "B+" if consume else "B"})
    ref = _save(root / "experience-config.json", config)
    return memory, ref


def test_trial_arm_cannot_relabel_a_consuming_configuration_as_baseline(tmp_path):
    _memory, reference = _configuration(tmp_path, consume=True, extract=False, trial=True)
    config = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    binding = json.loads(Path(config["trial_binding_ref"]["path"]).read_text(encoding="utf-8"))
    config["trial_binding_ref"] = _save(tmp_path / "wrong-arm.json", {**binding, "arm": "B"})
    wrong = _save(tmp_path / "wrong-config.json", config)
    with pytest.raises(ValueError, match="TASK_V2_TRIAL_ARM_SNAPSHOT_MISMATCH"):
        TaskAgentV2Experience(SimpleNamespace(), wrong)


@pytest.mark.parametrize("mode", ["consume", "withdraw", "extract", "compile_extract", "rejected_extract",
    "discovery_extract", "discovery_tamper_extract", "discovery_environment_extract", "recheck_extract",
    "trial_consume", "trial_off", "multi_extract", "context_trial_consume", "context_trial_off"])
def test_experience_through_normal_cli_actual_requests_and_terminal(repository_workspace, monkeypatch, capsys, mode,
                                                                expire_learning_send=False, capture_projection=None,
                                                                capture_owner=None, learning_question=False,
                                                                learning_outcome="completed"):

    root = repository_workspace
    profile, source = _profile(root)
    extract_mode = mode.endswith("extract")
    consume = not extract_mode and mode not in {"trial_off", "context_trial_off"}
    contextual = mode.startswith("context_trial_")
    conditions = {"has_candidate": False, "task_phase": "investigation", "stagnation_triggered": True} if contextual else None
    memory, config = _configuration(root, consume=consume, extract=extract_mode,
        trial=mode.startswith("trial_") or contextual, strategy_conditions=conditions)
    learning_ref = None
    if learning_question:
        from tests.unit.test_strategy_learning_request import question
        learning_ref = question(root)
        configuration = json.loads(Path(config["path"]).read_text(encoding="utf-8"))
        config = _save(root / "experience-learning-question.json", {**configuration,
            "learning_request_ref": learning_ref,
            "requested_strategy_scope": {"runtime": "dotnet", "task_phase": "investigation", "has_candidate": False}})
    if mode == "multi_extract":
        episodes = []
        for run_id in ("retained-one", "retained-two"):
            observation = _save(root / (run_id + "-observation.json"), {"status": "timed_out", "candidate_ref": None})
            episodes.append(_save(root / (run_id + "-episode.json"), {
                "schema_version": "public_strategy_episode_v1", "visibility": "public", "data_use": "strategy_learning",
                "run_id": run_id, "task_contract_ref": run_id + "-contract", "projection_version": "ordered_public_actions_v1",
                "information_gain": ["public_process_observation"], "applicability": {"runtime": "dotnet"},
                "evidence_refs": [{"kind": "json_file", "evidence_id": "outcome", **observation,
                    "visibility": "public", "allowed_uses": ["strategy_learning"]}],
                "selections": [{"evidence_id": "outcome", "locator": ["status"], "role": "technical_outcome"}]}))
        frozen = memory.freeze_episode_set(episodes, applicability={"runtime": "dotnet"},
            source_roles=["context", "opposing"])
        value = json.loads(Path(config["path"]).read_text(encoding="utf-8"))
        config = _save(root / "multi-experience-config.json", {**value, "extraction_episode_set_ref": asdict(frozen)})
    if expire_learning_send:
        original_window = TaskAgentExecutionV2.require_model_send_window

        def expire_before_learning(owner, purpose):
            if purpose == "strategy_learning":
                owner.deadline = 0
            return original_window(owner, purpose)

        monkeypatch.setattr(TaskAgentExecutionV2, "require_model_send_window", expire_before_learning)
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
            proposed = {"suggested_action": "Reproduce the public behavior before changing the expression.",
                "expected_observation": "The expression predicts the observed wrong value.",
                "counterevidence": ["The public reproduction passes before modification."]}
            if learning_question:
                sent = json.loads(payload["messages"][-1]["content"])
                row = sent["episodes"][0]
                assert sent["learning_question"]["reference"] == learning_ref
                assert sent["requested_strategy_scope"]["task_phase"] == "investigation"
                assert row["source_applicability"]["has_candidate"] is True
                assert row["source_task_context"]["fields"]["original_request"] == "Double the integer input, preserving the public signature."
                assessment = {"episode_id": row["episode_id"], "relation": "insufficient",
                    "reason": "A source task outcome does not establish transfer.",
                    "evidence": [row["source_task_context"]["field_sources"]["original_request"]]}
                if learning_outcome == "insufficient_information":
                    proposed = {"status": "insufficient_information", "reason": "No independent transfer comparison was observed."}
                proposed.update(source_assessments=[assessment], evidence_gaps=[{
                    "status": "not_observed", "description": "Independent transfer evidence is missing.", "evidence": []}])
            content = json.dumps(proposed)
            return httpx.Response(200, request=request, json={"model": model, "message": {"role": "assistant", "content": content},
                "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20})
        requests.append(payload)
        views = [json.loads(row["content"])["strategy_experience"] for row in payload["messages"]
            if row["role"] == "user" and '"strategy_experience"' in row["content"]]
        if consume:
            if contextual:
                assert len(views) == 1
                current = next(json.loads(row["content"])["current_task_state"] for row in payload["messages"]
                    if row["role"] == "user" and '"current_task_state"' in row["content"])
                saved.setdefault("strategy_selections", []).append({"view": views[0], "current": current})
            else:
                assert len(views) == 1 and views[0]["strategies"][0]["suggested_action"].startswith("Read the implementation")
        else:
            assert not views
        context = next(json.loads(row["content"]) for row in payload["messages"]
            if row["role"] == "user" and '"task_agent_v2"' in row["content"])
        tool_contents = [json.loads(row["content"]) for row in payload["messages"] if row["role"] == "tool"]
        observations = [value.get("value", value) for value in tool_contents]
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
        elif contextual:
            if turn == 3:
                saved["binding"] = last["binding_id"]
            if turn in {3, 4, 5}:
                name, args = "source_read", {"parent_revision_ref": saved["parent"], "path": "Library/Arithmetic.cs"}
            elif turn == 6:
                saved["view"] = last["source_view_ref"]
                name, args = "candidate_submit", {"parent_revision_ref": saved["parent"], "edits": [{
                    "binding_id": saved["binding"], "source_view_ref": saved["view"],
                    "before_text": "=> value;", "after_text": "=> value * 2;"}]}
            elif turn == 7:
                saved["parent"] = last["candidate_ref"]
                name, args = "candidate_check", {"candidate_ref": saved["parent"]}
            else:
                assert turn == 8 and last["status"] == "passed"
                name, args = "finish", {"candidate_ref": saved["parent"], "check_ref": last["check_ref"]}
        elif mode == "argument_extract" and turn >= 5:
            if turn in {5, 6}:
                assert last["status"] == ("behavior_failed" if turn == 5 else "rejected_not_executed")
                name, args = "candidate_read", {"parent_revision_ref": saved["parent"]}
                if turn == 6:
                    args["path"] = "Library/Arithmetic.cs"
            elif turn == 7:
                saved["view"] = last["source_view_ref"]
                name, args = "candidate_submit", {"parent_revision_ref": saved["parent"], "edits": [{
                    "binding_id": saved["binding"], "source_view_ref": saved["view"],
                    "before_text": "=> value + 1;", "after_text": "=> value * 2;"}]}
            elif turn == 8:
                saved["parent"] = last["candidate_ref"]
                name, args = "candidate_check", {"candidate_ref": saved["parent"]}
            else:
                assert turn == 9 and last["status"] == "passed"
                name, args = "finish", {"candidate_ref": saved["parent"], "check_ref": last["check_ref"]}
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
            after = "=> value + 1;" if turn == 3 and mode in {"extract", "recheck_extract", "argument_extract"} else "=> value * 2;"
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
            "tool_calls": [{**({"id": f"controlled-recheck-{turn}"} if mode == "recheck_extract" or contextual else {}),
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
        assert code == 0 and result["status"] == "awaiting_delivery", json.dumps({
            "root": str(root), "code": code, "product_status": result.get("status"),
            "termination": json.loads(original).get("termination")})
    receipt = owner.complete_terminal_learning()
    assert Path(execution_ref["path"]).read_bytes() == original
    if capture_owner is not None:
        capture_owner(owner)
    if contextual:
        assert len(requests) == 8 and not learning
        if consume:
            selected_turns = [index + 1 for index, row in enumerate(saved["strategy_selections"]) if row["view"]["strategies"]]
            assert selected_turns == [6]
            assert saved["strategy_selections"][5]["current"]["working_state"]["stagnation"]["triggered"] is True
            assert saved["strategy_selections"][6]["current"]["current_revision_ref"] != saved["strategy_selections"][6]["current"]["baseline_revision_ref"]
            receipts = [json.loads(path.read_text(encoding="utf-8")) for path in owner.output_root.glob("experience-selection-*.json")]
            assert len(receipts) == 8
            chosen = next(row for row in receipts if row["selection"]["selected"])
            sources = chosen["context_sources"]
            actual = saved["strategy_selections"][5]["current"]
            assert all(sources[key] == actual[key] for key in (
                "task_contract_ref", "current_revision_ref", "working_state_ref", "current_check_ref"))
            assert chosen["selection"]["current_context"]["has_candidate"] is False
            assert chosen["selection"]["current_context"]["task_phase"] == "investigation"
            assert sources["stagnation_scope"] == "exact_repeated_tool_returns_only_not_semantic_progress"
        else:
            assert not list(owner.output_root.glob("experience-selection-*.json"))
        episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
        assert episode["applicability"]["has_candidate"] is True
        assert episode["applicability"]["task_phase"] == "public_check_passed"
        assert episode["applicability_sources"]["current_revision_ref"] == saved["parent"]
        assert "hypothesis_refuted" not in episode["information_gain"]
        return
    if expire_learning_send:
        assert receipt["status"] == "failed" and not learning
        assert receipt["model_calls"] == 1 and receipt["unknown_usage_calls"] == 1
        assert receipt["budget_after"]["consumed"] == owner.execution_receipt().budget_after["consumed"] + 1
        raw = next(owner.output_root.glob("model-not-sent-*.json"))
        assert json.loads(raw.read_bytes())["purpose"] == "strategy_learning"
        assert json.loads(raw.read_bytes())["transport_attempted"] is False
        return
    if learning_question:
        assert receipt["status"] == learning_outcome, receipt
        assert len(learning) == 1 and receipt["model_calls"] == 1
        assert receipt["known_usage"][0]["total_tokens"] == 30
        assert receipt["learning_request_ref"] == learning_ref
        assert receipt["assessment_kind"] == "model_interpretation_not_verified_fact"
        assert receipt["evidence_gaps"][0]["status"] == "not_observed"
        assert owner.complete_terminal_learning() == receipt and len(learning) == 1
        sent = json.loads(learning[0]["messages"][-1]["content"])
        assert sent["episodes"][0]["source_task_context"]["fields"] == owner._v2_state.contract["task_requirements"]
        if learning_outcome == "completed":
            assert memory.read(receipt["strategy_ref"])["state"] == "proposed"
        else:
            assert receipt["strategy_ref"] is None
        if capture_projection is not None:
            capture_projection({"episode_ref": receipt["episode_ref"], "extraction_ref": receipt["extraction_ref"],
                "controlled_wire_ref": _save(root / "controlled-learning-contract-wire.json", learning[0])})
        return
    if mode == "multi_extract":
        assert receipt["status"] == "completed" and len(learning) == 1
        assert receipt["episode_set_ref"] == asdict(frozen)
        sent = json.loads(learning[0]["messages"][-1]["content"])
        assert [row["run_id"] for row in sent["episodes"]] == ["retained-one", "retained-two"]
        assert [row["source_role"] for row in sent["episodes"]] == ["context", "opposing"]
        assert receipt["known_usage"][0]["total_tokens"] == 30
        assert memory.read(receipt["strategy_ref"])["state"] == "proposed"
        assert owner.complete_terminal_learning() == receipt and len(learning) == 1
    elif mode in {"extract", "compile_extract", "discovery_extract", "recheck_extract", "argument_extract"}:
        assert receipt["status"] == "completed", receipt
        assert len(learning) == 1
        assert receipt["budget_after"]["consumed"] == owner.execution_receipt().budget_after["consumed"] + 1
        assert receipt["known_usage"][0]["total_tokens"] == 30
        assert memory.read(receipt["strategy_ref"])["origin"]["kind"] == "model"
        assert memory.read(receipt["strategy_ref"])["state"] == "proposed"
        assert owner.complete_terminal_learning() == receipt and len(learning) == 1
        learning_payload = json.loads(learning[0]["messages"][-1]["content"])
        experience = learning_payload["experience"]
        assert "observations" not in learning_payload  # No duplicate flat code payload.
        assert experience["projection_version"] == "ordered_public_actions_v1"
        actions = experience["actions"]
        native = json.loads(original)
        assert [a["action_index"] for a in actions] == list(range(len(native["completed_tool_records"])))
        assert [a["action_request"][0]["value"] for a in actions] == [
            row["name"] for row in native["completed_tool_records"]]
        assert all(a["model_statement_status"] == "not_explicitly_expressed" for a in actions)
        assert all(not a["model_statement"][0]["value"] for a in actions)
        assert {row["locator"][0]: row["value"] for row in experience["technical_outcome"]} == {
            key: native[key] for key in ("status", "candidate_ref", "check_ref", "unknown_effects")}
        assert "not delivered" in experience["fact_scope"]["technical_outcome"]
        assert "subsequent request" in experience["fact_scope"]["action_return"]
        episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
        # Verify every projected value against its immutable original locator, not an expected narrative.
        from src.mod_agent.model_qualification import read_bound_json
        from src.mod_agent.planning.task_memory import _select
        documents = {row["evidence_id"]: read_bound_json({key: row[key] for key in ("path", "sha256")})
            for row in episode["evidence_refs"]}
        projected = [value for action in actions for key in (
            "action_request", "action_return", "model_statement", "source_state") for value in action[key]]
        projected += [value for key in ("public_check", "public_experiment", "technical_outcome", "nonexecuted_attempt")
            for value in experience[key]]
        assert len(projected) == len(episode["selections"])
        for value in projected:
            assert value["value"] == _select(documents[value["evidence_id"]], value["locator"])
        if capture_projection is not None:
            capture_projection({"episode_ref": receipt["episode_ref"], "execution_ref": execution_ref,
                "extraction_ref": receipt["extraction_ref"], "strategy_ref": receipt["strategy_ref"],
                "controlled_wire_ref": _save(root / "controlled-learning-wire.json", learning[0]),
                "scope": "controlled_HTTP_not_real_model_or_business_success"})
        if mode == "argument_extract":
            assert any(value["value"] == "not_dispatched" for value in experience["nonexecuted_attempt"])
            assert any("argument_errors" in value["value"].get("output", {})
                for value in experience["nonexecuted_attempt"] if isinstance(value["value"], dict))
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
            observed = projected
            assert any(row["value"] == "test_catalog" for row in observed)
            assert any(isinstance(row["value"], dict) and row["value"].get("execute_discovery") is True for row in observed)
            assert any(isinstance(row["value"], dict) and "plan" in row["value"] for row in observed)
            assert any(row["value"] == "exploration_only" for row in observed)
            assert any(row["locator"] == ["grants_acceptance"] and row["value"] is False for row in observed)
            assert all(source["visibility"] == "public" for source in episode["evidence_refs"])
        if mode in {"extract", "recheck_extract", "argument_extract"}:
            episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
            assert ("effective_revision" in episode["information_gain"]) is (mode != "recheck_extract")
            if mode == "recheck_extract":
                assert owner._v2_session.public_checks[0]["report"]["candidate_source_id"] == owner._v2_session.public_checks[1]["report"]["candidate_source_id"]
    elif mode == "rejected_extract":
        assert receipt["status"] == "completed" and len(learning) == 1
        episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
        assert episode["information_gain"] == ["public_process_observation"]
        assert not episode["public_counterexamples"]
        assert not owner._v2_session.public_checks
        assert owner._experience.context(owner._v2_session)["has_public_counterexample"] is False
        assert owner._experience.context(owner._v2_session)["counterexample_kind"] == "none"
    elif mode == "discovery_tamper_extract":
        assert receipt["status"] == "failed" and "EXPERIMENT_RESULT_CHANGED" in receipt["message"]
        assert not learning and receipt["model_calls"] == 0
    elif mode == "discovery_environment_extract":
        assert receipt["status"] == "completed" and len(learning) == 1
        episode = json.loads(Path(receipt["episode_ref"]["path"]).read_text(encoding="utf-8"))
        assert episode["information_gain"] == ["public_process_observation"]
        assert not episode["public_counterexamples"]
        assert owner._experience.context(owner._v2_session)["counterexample_kind"] == "none"
    else:
        assert receipt["status"] == "not_enabled"
        assert len(list(owner.output_root.glob("experience-delivery-*.json"))) == (len(requests) if consume else 0)
        if mode.startswith("trial_"):
            journal = owner._recovery_journal
            host = owner._v2_state.load(journal.host_config_ref)
            assert host["experience_configuration_ref"] == config
            assert owner._experience.trial_binding["binding"]["arm"] == ("B+" if consume else "B")
            protocol = owner._experience.trial_binding["protocol"]
            assert protocol["planned_pairs"][0]["task_id"] == "double"
            # 为什么要有它：首个派发前的宿主配置必须已包含冻结引用，导出不能事后为旧运行补造比较条件。
            first_checkpoint = owner._v2_state.load(owner._recovery_journal.latest_ref)
            while first_checkpoint["previous_checkpoint_ref"] is not None:
                first_checkpoint = owner._v2_state.load(first_checkpoint["previous_checkpoint_ref"])
            assert first_checkpoint["host_config_ref"] == journal.host_config_ref
            assert first_checkpoint["loop"]["model_budget"]["consumed"] == 0


def test_learning_preparation_expiry_preserves_reservation_without_model_send(repository_workspace, monkeypatch, capsys):
    test_experience_through_normal_cli_actual_requests_and_terminal(
        repository_workspace, monkeypatch, capsys, "extract", expire_learning_send=True)


def test_ordered_episode_keeps_parameter_correction_and_original_receipts(
        repository_workspace, monkeypatch, capsys, record_property):
    test_experience_through_normal_cli_actual_requests_and_terminal(
        repository_workspace, monkeypatch, capsys, "argument_extract",
        capture_projection=lambda refs: record_property("projection_receipts", json.dumps(refs)))


def test_stopped_pending_prefix_seals_without_resume_or_model_and_binds_stop_source(
        repository_workspace, monkeypatch, capsys):
    from src.mod_agent.model_qualification import read_bound_json
    from src.mod_agent.task_agent_v2_experience import (
        seal_current_public_episode,
        seal_saved_public_episode,
    )

    owners = []
    test_experience_through_normal_cli_actual_requests_and_terminal(
        repository_workspace, monkeypatch, capsys, "consume", capture_owner=owners.append)
    owner = owners[0]
    original_calls = owner.budget_snapshot()["consumed"]
    normal = seal_current_public_episode(owner)
    assert normal == seal_current_public_episode(owner)
    checkpoint_ref = owner.v2_checkpoint_ref
    while True:
        checkpoint = owner._v2_state.load(checkpoint_ref)
        if checkpoint["loop"]["phase"] == "provider_pending" and len(checkpoint["details"]["completed_tool_records"]) == 1:
            break
        checkpoint_ref = checkpoint["previous_checkpoint_ref"]
        assert checkpoint_ref is not None
    checkpoint_path = Path(owner.store.root) / checkpoint_ref
    original_checkpoint = checkpoint_path.read_bytes()
    source = _save(repository_workspace / "stopped-process.json", {"status": "user_stopped", "exit_code": 130,
        "run_id": checkpoint["details"]["run_id"], "request_id": checkpoint["details"]["request_id"]})
    stop_value = {"schema": "task-agent-process-stop.v1", "checkpoint_ref": checkpoint_ref,
        "task_contract_ref": checkpoint["task_contract_ref"], "run_id": checkpoint["details"]["run_id"],
        "request_id": checkpoint["details"]["request_id"], "stop_kind": "user_stopped", "source_ref": source,
        "exit_code": 130}
    stop = _save(repository_workspace / "stop.json", stop_value)
    sealed = seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=stop)
    assert sealed == seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=stop)
    episode = read_bound_json(sealed)
    assert episode["information_gain"] == ["public_process_observation"]
    assert not episode["public_counterexamples"]
    assert episode["applicability"]["has_candidate"] is False
    assert episode["applicability"]["task_phase"] == "investigation"
    assert episode["applicability_sources"]["current_revision_ref"] == checkpoint["task_head"]["current_revision_ref"]
    assert episode["applicability_sources"]["working_state_locator"] == ["owner", "working_state"]
    observed = read_bound_json(episode["execution_ref"])
    assert observed["status"] == "user_stopped" and observed["native_terminal"] is False
    assert observed["business_result"] == "not_established"
    assert observed["unknown_effects"] == ["provider_response_unknown"]
    assert owner.budget_snapshot()["consumed"] == original_calls
    assert checkpoint_path.read_bytes() == original_checkpoint
    wrong_stop = _save(repository_workspace / "wrong-stop.json", {**stop_value, "run_id": "another-run"})
    with pytest.raises(ValueError, match="PROCESS_STOP_TASK_CHANGED"):
        seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=wrong_stop)
    wrong_timeout = _save(repository_workspace / "wrong-timeout.json", {**stop_value, "stop_kind": "timed_out"})
    with pytest.raises(ValueError, match="PROCESS_STOP_SOURCE_CHANGED"):
        seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=wrong_timeout)
    normal_source = _save(repository_workspace / "normal-process.json", {"status": "exited", "exit_code": 0,
        "run_id": stop_value["run_id"], "request_id": stop_value["request_id"]})
    mislabeled_stop = _save(repository_workspace / "mislabeled-stop.json", {
        **stop_value, "source_ref": normal_source, "exit_code": 0})
    with pytest.raises(ValueError, match="PROCESS_STOP_SOURCE_CHANGED"):
        seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=mislabeled_stop)
    arm_root = repository_workspace / "owned-arm"
    argv = ["python", "-m", "controlled_public_runner"]
    planned_source = _save(arm_root / "execution/process.json", {"status": "timed_out", "timeout": True,
        "exit_code": 124, "argv": argv, "cwd": str(repository_workspace)})
    plan_value = {"run_id": stop_value["run_id"], "run_root": str(Path(owner.store.root).parent),
        "root": str(arm_root), "workspace": str(repository_workspace), "command": argv}
    plan = _save(repository_workspace / "frozen-plan.json", plan_value)
    planned_stop = {**stop_value, "stop_kind": "timed_out", "exit_code": 124,
        "source_ref": planned_source, "plan_ref": plan}
    plan_stop_ref = _save(repository_workspace / "planned-stop.json", planned_stop)
    planned_episode = seal_saved_public_episode(artifact_store_root=owner.store.root,
        checkpoint_ref=checkpoint_ref, stop_ref=plan_stop_ref)
    planned_outcome = read_bound_json(read_bound_json(planned_episode)["execution_ref"])
    assert planned_outcome["collector_scope"] == "post_process_public_collector"
    assert planned_outcome["collector_source_refs"] and planned_outcome["plan_ref"] == plan
    bad_plan = _save(repository_workspace / "bad-frozen-plan.json", {**plan_value, "command": ["other-run"]})
    bad_plan_stop = _save(repository_workspace / "bad-plan-stop.json", {**planned_stop, "plan_ref": bad_plan})
    with pytest.raises(ValueError, match="PROCESS_PLAN_BINDING_CHANGED"):
        seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=bad_plan_stop)
    unbound_stop = _save(repository_workspace / "unbound-process-stop.json", {
        key: value for key, value in planned_stop.items() if key != "plan_ref"})
    with pytest.raises(ValueError, match="PROCESS_STOP_SOURCE_TASK_CHANGED"):
        seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=unbound_stop)
    Path(source["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        seal_saved_public_episode(artifact_store_root=owner.store.root, checkpoint_ref=checkpoint_ref, stop_ref=stop)
    memory = StrategyExperience(ArtifactStore(repository_workspace / "sealed-learning"))
    with pytest.raises(ValueError):
        memory.freeze_episode_set([sealed], applicability={"runtime": "dotnet"})


@pytest.mark.parametrize("outcome", ["completed", "insufficient_information"])
def test_explicit_learning_contract_through_normal_terminal_actual_http(repository_workspace, monkeypatch, capsys, outcome):
    test_experience_through_normal_cli_actual_requests_and_terminal(repository_workspace, monkeypatch, capsys,
        "extract", learning_question=True, learning_outcome=outcome)
