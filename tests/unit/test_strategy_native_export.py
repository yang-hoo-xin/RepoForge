"""Normal CLI native artifacts feed the offline evaluation export entry."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from src.mod_agent import cli
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.strategy_adoption import decide_strategy_adoption
from src.mod_agent.evaluation.strategy_native_export import (
    _learning,
    export_strategy_execution,
    validate_native_strategy_execution,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.planning.strategy_experience import StrategyExperience
from src.mod_agent.task_agent_v2_state import TaskAgentV2State
from tests.unit import test_strategy_experience as memory_fixtures
from tests.unit import test_task_agent_v2_experience as entry_fixtures
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def _native(root):
    paths = list((root / "runs").glob("*/a/task-agent-execution/execution.json"))
    assert len(paths) == 1
    reference = _ref(paths[0])
    value = read_bound_json(reference)
    state = TaskAgentV2State(paths[0].parent.parent, value["task_contract_ref"])
    call_id = next(row["call_id"] for row in value["completed_tool_records"] if row["name"] == "candidate_submit")
    return reference, value, state, call_id


@pytest.mark.parametrize("mode,arm", [("trial_consume", "B+"), ("trial_off", "B")])
def test_normal_cli_exports_actual_input_action_cost_and_preregistered_binding(repository_workspace, monkeypatch, capsys, mode, arm):
    root = repository_workspace
    original_save = entry_fixtures._save
    def save(path, payload):
        if path.name == "trial-source.json":
            source = root / "source"
            payload = {**payload, "source_files_sha256": canonical_hash({p.relative_to(source).as_posix(): _ref(p)["sha256"]
                for p in source.rglob("*") if p.is_file()}),
                "original_request_sha256": canonical_hash("Double the integer input, preserving the public signature.")}
        return original_save(path, payload)
    monkeypatch.setattr(entry_fixtures, "_save", save)
    entry_fixtures.test_experience_through_normal_cli_actual_requests_and_terminal(root, monkeypatch, capsys, mode)
    native_ref, native, state, call_id = _native(root)
    before = Path(native_ref["path"]).read_bytes()
    code = cli.main(["export-strategy-execution", "--execution-record", native_ref["path"],
        "--call-id", call_id, "--output-root", str(root / "export")])
    import json
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    exported = read_bound_json(result["execution_ref"])
    validated = validate_native_strategy_execution(result["execution_ref"])
    assert validated["status"] == "native_execution_verified"
    assert validated["freeze_binding"]["status"] == "protocol_bound_before_model"
    assert validated["freeze_binding"]["host_event_sequence"] < validated["freeze_binding"]["first_model_reservation_sequence"]
    assert exported["planned_source_binding"] == "source_and_request_verified"
    assert exported["arm"] == arm and exported["native_call_id"] == call_id
    assert exported["known_cost"]["model_calls"] == native["budget_after"]["consumed"]
    assert exported["known_cost"]["input_tokens"] == sum(row["usage"]["input_tokens"] for row in native["provider_turns"])
    assert exported["known_cost"]["duration_ms"] >= 0 and exported["human_technical_interventions"] is None
    assert exported["resource_evidence"]["duration_source"]["boundary"] == "product_return"
    assert exported["resource_evidence"]["human_intervention_source"]["closed_loop_human_technical_interventions"] == 0
    assert exported["resource_evidence"]["human_intervention_source"]["outside_scope_human_technical_interventions"] is None
    action = read_bound_json(exported["action_ref"])
    response = read_bound_json(action["response_ref"])
    assert action["action"] == {key: next(row for row in response["tool_calls"] if row["call_id"] == call_id)[key]
        for key in ("name", "arguments")}
    assert read_bound_json(action["output_ref"])["call_id"] == call_id
    assert Path(native_ref["path"]).read_bytes() == before
    with pytest.raises(ValueError, match="EXPORTED_NATIVE_FACTS_CHANGED"):
        validate_native_strategy_execution(_save(root / "wrong-version.json", {**exported, "candidate_source_id": "different"}))
    with pytest.raises(ValueError, match="EXPORTED_NATIVE_FACTS_CHANGED"):
        validate_native_strategy_execution(_save(root / "invented-cost.json", {**exported, "known_cost": {"model_calls": 0}}))
    with pytest.raises(ValueError, match="PREREGISTERED_TRIAL_CHANGED"):
        export_strategy_execution(native_execution_ref=native_ref, call_id=call_id, output_root=root / "wrong-arm",
            arm="B" if arm == "B+" else "B+")
    state.record_resources({"model_calls": 1}, operation_id="controlled-unanswered-reservation")
    tail_ref = export_strategy_execution(native_execution_ref=native_ref, call_id=call_id, output_root=root / "ledger-tail-export")
    tail = read_bound_json(tail_ref)
    assert tail["known_cost"]["model_calls"] == native["budget_after"]["consumed"] + 1
    assert tail["known_cost"]["input_tokens"] is None
    assert tail["known_cost"]["duration_ms"] is None
    assert tail["resource_evidence"]["duration_source"]["known_observed_elapsed_ms"] == exported["known_cost"]["duration_ms"]
    assert tail["resource_evidence"]["unknown_usage_calls"] == 1
    assert tail["resource_evidence"]["unreconciled_calls_after_native_and_learning"] == 1
    assert tail["resource_evidence"]["budget_after"] == native["budget_after"]
    state.record_resources({"model_calls": 1}, operation_id="controlled-later-reservation")
    assert validate_native_strategy_execution(tail_ref)["unknown_usage_calls"] == 1
    assert validate_native_strategy_execution(result["execution_ref"])["unknown_usage_calls"] == 0
    if arm == "B+":
        _adoption_consumes_actual_export_but_does_not_trust_handwritten_baseline(root, result["execution_ref"])


def _adoption_consumes_actual_export_but_does_not_trust_handwritten_baseline(root, exported_ref):
    exported = read_bound_json(exported_ref)
    request = read_bound_json(exported["native_export_request_ref"])
    protocol_ref = request["protocol_ref"]
    protocol = read_bound_json(protocol_ref)
    task = protocol["planned_pairs"][0]
    response = read_bound_json(exported["model_request_ref"])
    response["messages"] = [message for message in response["messages"] if not isinstance(message.get("content"), dict)
        or "strategy_experience" not in message["content"]]
    baseline_request = _save(root / "handwritten-baseline-request.json", response)
    action = read_bound_json(exported["action_ref"])
    baseline_action = _save(root / "handwritten-baseline-action.json", {**action, "request_ref": baseline_request})
    baseline = {key: value for key, value in exported.items() if key not in {"native_execution_ref", "native_export_request_ref"}}
    baseline.update(arm="B", model_request_ref=baseline_request, action_ref=baseline_action, experience_snapshot_ref=None)
    baseline_ref = _save(root / "handwritten-baseline.json", baseline)
    arms = {}
    for arm, reference in (("B", baseline_ref), ("B+", exported_ref)):
        verdict = _save(root / ("unqualified-verdict-" + arm.replace("+", "plus") + ".json"), {
            "schema_version": "independent_strategy_verdict_v1", "independent": True, "batch_complete": True,
            "task_id": task["task_id"], "execution_ref": reference, "evaluator_ref": protocol["invariants"]["judge"]["evaluator_ref"],
            "candidate_source_id": exported["candidate_source_id"], "task_success": False, "preservation_passed": False})
        arms[arm] = {"execution_ref": reference, "evaluation_ref": verdict}
    paired = _save(root / "diagnostic-pair.json", {"schema_version": "strategy_pair_result_v1", "protocol_ref": protocol_ref,
        **task, "exposure": read_bound_json(task["exposure_ref"])["exposure"], "arms": arms})
    policy = _save(root / "adoption-policy.json", {"schema_version": "strategy_adoption_policy_v1",
        "allow_limited_activation": True, "minimum_unseen_pairs": 1, "cost_policy": {"mode": "report_only"}})
    memory = StrategyExperience(ArtifactStore(root / "experience"))
    strategy = memory.read(protocol["snapshot_ref"])["strategy_refs"][0]
    decision = memory.read(decide_strategy_adoption(memory, strategy, protocol_ref=protocol_ref, result_refs=[paired], policy_ref=policy))
    assert decision["target_state"] == "trial"
    assert decision["results"][0]["native_evidence"]["B+"]["status"] == "native_execution_verified"
    assert decision["results"][0]["native_evidence"]["B"]["status"] == "native_execution_evidence_missing"


def test_posthoc_protocol_is_explicitly_unproven_and_wrapper_not_native(repository_workspace, monkeypatch, capsys):
    root = repository_workspace
    entry_fixtures.test_experience_through_normal_cli_actual_requests_and_terminal(root, monkeypatch, capsys, "consume")
    native_ref, _native_value, state, call_id = _native(root)
    config = read_bound_json(_ref(root / "experience-config.json"))
    baseline = state.load(state.contract["baseline_manifest_ref"])
    source = _save(root / "posthoc-source.json", {"task_id": "diagnostic", "task_family": "arithmetic", "source_version": "observed-after-run",
        "source_files_sha256": canonical_hash({p: row["sha256"] for p, row in baseline["files"].items()}),
        "original_request_sha256": canonical_hash(state.load(state.contract["original_request_ref"])["original_request"])})
    exposure = _save(root / "posthoc-exposure.json", {"task_id": "diagnostic", "source_ref": source, "attempt": 1, "exposure": "exposed"})
    protocol = _save(root / "posthoc-protocol.json", {"schema_version": "strategy_trial_protocol_v1", "batch_complete": True,
        "snapshot_ref": config["snapshot_ref"], "invariants": {}, "invariants_sha256": canonical_hash({}),
        "task_families": ["arithmetic"], "planned_pairs": [{"task_id": "diagnostic", "task_family": "arithmetic", "attempt": 1,
            "source_ref": source, "exposure_ref": exposure, "current_context": {"runtime": "dotnet", "target_domain": "csharp_repository"}}]})
    code = cli.main(["export-strategy-execution", "--execution-record", native_ref["path"], "--call-id", call_id,
        "--output-root", str(root / "posthoc-export"), "--protocol-record", protocol["path"],
        "--task-id", "diagnostic", "--attempt", "1", "--arm", "B+"])
    import json
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    assert validate_native_strategy_execution(result["execution_ref"])["freeze_binding"]["status"] == "freeze_binding_unproven"
    with pytest.raises(ValueError, match="NATIVE_EXECUTION_REQUIRED"):
        export_strategy_execution(native_execution_ref=result["execution_ref"], call_id=call_id, output_root=root / "fake")


def test_unknown_learning_retains_unknown_usage_from_actual_extraction_receipt(tmp_path):
    setup = memory_fixtures.setup.__wrapped__(tmp_path)
    memory, _, strategy, _ = setup
    episode, admission = memory_fixtures.extraction_inputs(tmp_path, setup)
    class Failing(memory_fixtures.Provider):
        def complete_turn(self, request):
            raise TimeoutError("controlled unknown response")
    receipts = []
    with pytest.raises(TimeoutError):
        memory.propose_from_episode(Failing(strategy), episode, admission_ref=admission,
            before_call=lambda ref: True, after_call=receipts.append)
    budget = {"policy": "meter_only", "maximum": None, "consumed": 2, "remaining": None,
        "by_stage": {"initial_patch": {"consumed": 2, "maximum": None, "remaining": None}},
        "count_basis": "reserved_provider_invocations_not_transport_proof"}
    native = {"budget_after": budget}
    native_ref = _save(tmp_path / "native-reference.json", native)
    after = deepcopy(budget)
    after.update(consumed=3)
    after["by_stage"]["initial_patch"]["consumed"] = 3
    config = _save(tmp_path / "learning-config.json", {"store_root": str(memory.store.root)})
    learning = _save(tmp_path / "unknown-learning.json", {"original_execution_ref": native_ref, "configuration_ref": config,
        "budget_before": budget, "budget_after": after, "model_calls": 1, "known_usage": [], "unknown_usage_calls": 1,
        "call_receipts": receipts})
    actual_after, known, unknown, _ = _learning(native_ref, native, learning)
    assert actual_after["consumed"] == 3 and known == [] and unknown == 1


def test_unknown_learning_response_is_not_reclassified_as_known_zero_usage(tmp_path):
    memory = StrategyExperience(ArtifactStore(tmp_path / "experience"))
    response = memory._write("unknown-response", "response", {"request_ref": {}, "response": {
        "provider": "openai_responses", "provider_diagnostic": {"measured_usage": None},
        "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
            "cached_input_tokens": None}}})
    budget = {"policy": "meter_only", "maximum": None, "consumed": 2, "remaining": None,
        "by_stage": {"initial_patch": {"consumed": 2, "maximum": None, "remaining": None}},
        "count_basis": "reserved_provider_invocations_not_transport_proof"}
    native = {"budget_after": budget}
    native_ref = _save(tmp_path / "native-response-reference.json", native)
    after = deepcopy(budget)
    after.update(consumed=3)
    after["by_stage"]["initial_patch"]["consumed"] = 3
    config = _save(tmp_path / "response-learning-config.json", {"store_root": str(memory.store.root)})
    learning = _save(tmp_path / "unknown-response-learning.json", {"original_execution_ref": native_ref,
        "configuration_ref": config, "budget_before": budget, "budget_after": after, "model_calls": 1,
        "known_usage": [], "unknown_usage_calls": 1,
        "call_receipts": [{"response_ref": asdict(response), "usage": None}]})

    actual_after, known, unknown, _ = _learning(native_ref, native, learning)

    assert actual_after["consumed"] == 3 and known == [] and unknown == 1
