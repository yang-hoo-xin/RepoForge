"""No-model tests of serial trial orchestration and full-delivery comparisons."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.evaluation.product_trial_batch import (
    run_product_trial_batch,
    summarize_trials,
    verify_product_model_proposal,
)


@pytest.fixture(autouse=True)
def isolate_owned_model_origin_check(monkeypatch):
    # Orchestration cases use synthetic files; the origin reader is checked separately.
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.verify_product_model_proposal",
        lambda *_args: {"verification": "controlled_origin_fixture"})


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _spec(tmp_path, *, repeats=1):
    launcher = Path(__file__).parents[2] / "scripts/run_frozen_product.py"
    judge = _save(tmp_path / "judge.json", {"identity": "same selected judge"})
    trials = []
    for repeat in range(1, repeats + 1):
        for arm in ("baseline", "candidate"):
            run_id = f"{arm}_{repeat}"
            worker = _save(tmp_path / "inputs" / (run_id + ".json"), {"policy": {
                "max_tool_rounds": 6, "max_observation_bytes": 32768, "max_total_model_turns": 8},
                "output_root": str(tmp_path / "runs" / run_id)})
            trials.append({"task_id": "one-task", "arm": arm, "repeat": repeat, "run_id": run_id,
                "bound_inputs": [worker], "process_deadline_seconds": 1800,
                "allowed_root": str(tmp_path / "runs"), "run_root": str(tmp_path / "runs" / run_id),
                "snapshot_root": str(tmp_path / "snapshots" / run_id), "launcher": str(launcher),
                "business_argv": ["run-request", "caller original request", "--run-id", run_id,
                    "--workspace-root", str(tmp_path / "runs"), "--patch-worker-config", worker["path"]]})
    return {"judge_ref": judge, "trials": trials}


def _measurement(**kwargs):
    return {"model": {"generation_records": 2, "response_origins": {"live_provider_response": 2},
        "usage": {"input_tokens": {"total": 100},
        "output_tokens": {"total": 10}, "generation_elapsed_ms_sum": 50}},
        "tools": {"observations": [{"trace_id": "initial", "observation_id": "source"}], "selected_record_total": 1}}


def _runner(log, *, failed_arm=None, missing_run=False, changed_budget=False):
    def run(command, **kwargs):
        root = Path(command[command.index("--run-root") + 1])
        snapshot = Path(command[command.index("--snapshot-root") + 1])
        log.append(("normal", root.name))
        if missing_run:
            return SimpleNamespace(returncode=2, stderr="startup failed")
        root.mkdir(parents=True)
        manifest = {"run_id": root.name, "request_id": "request-" + root.name,
            "request": "same task", "source_snapshot_sha256": "s" * 64,
            "ollama_patch_generation_profile": {"model": "model", "context_window": 32768, "output_budget": 4096, "timeout_seconds": 180},
            "nuget_isolated_build_profile": {"timeout_seconds": 120},
            "model_admission": {"model": "model", "model_digest": "m" * 64, "provider": "local",
                "execution_locality": "machine_local"}, "source_disclosure": {}, "runtime_admission": {}}
        result = {"run_id": root.name, "status": "blocked" if root.name.startswith(str(failed_arm)) else "awaiting_approval",
            "model_call_budget": {"maximum": 10 if changed_budget and root.name.startswith("candidate") else 12,
                "policy": "task_total_only"}}
        _save(root / "run_manifest.json", manifest)
        _save(root / "run_result.json", result)
        argv = command[command.index("--") + 1:]
        worker_path = Path(argv[argv.index("--patch-worker-config") + 1])
        raw = worker_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        retained = "external/" + digest + "/" + worker_path.name
        _save(snapshot / retained, json.loads(raw))
        _save(snapshot / "execution_snapshot.json", {"files": {"src/owner.py": {"sha256": "1" * 64},
            retained: {"sha256": digest, "size": len(raw)}}, "business_argv": argv,
            "external_refs": [{"role": "bound_input", "path": str(worker_path), "sha256": digest, "retained_path": retained}]})
        _save(snapshot / "child_execution.json", {"status": "verified"})
        return SimpleNamespace(returncode=2 if result["status"] == "blocked" else 0)
    return run


def _delivery(log, *, origin="product_model_proposal", model_calls=0, missing_observations=False):
    def deliver(context):
        root = Path(context["run_root"])
        log.append(("delivery", root.name))
        evidence = _save(root / "delivery/evidence.json", {"actual": "passed"})
        public = {"run_id": root.name, "status": "built", "model_calls_performed": model_calls,
            "execution_trace": {"trace_id": "delivery"}, "execution_observations": {
                "build": {"id": "build", "kind": "tool"}}}
        if missing_observations:
            public.pop("execution_observations")
        reference = _save(root / "delivery/build.json", public)
        return {"status": "delivered", "source_origin": origin, "delivery_complete": True,
            "acceptance_passed": True, "evidence_refs": [evidence], "execution_refs": [reference]}
    return deliver


def test_prepare_preserves_original_argv_and_performs_no_execution(tmp_path):
    spec = _spec(tmp_path)
    output = run_product_trial_batch(spec)
    assert not output["executed"]
    assert output["trials"][0]["command"][-len(spec["trials"][0]["business_argv"]):] == spec["trials"][0]["business_argv"]
    assert not Path(spec["trials"][0]["run_root"]).exists()
    assert output["summary"]["distinct_task_count"] == 1
    assert output["summary"]["by_arm"]["baseline"]["completion_fraction"] is None


@pytest.mark.parametrize("damage", [None, "receipt", "candidate", "judge", "native", "unqualified", "resume", "unlinked_resume"])
def test_v2_pending_delivery_reaches_native_receipt_and_independent_verdict(tmp_path, monkeypatch, damage):
    from src.mod_agent.evaluation import product_trial_batch as batch

    spec, log, verified = _spec(tmp_path), [], []
    ordinary_runner, ordinary_delivery = _runner(log), _delivery(log)
    monkeypatch.setattr(batch, "measure_product_task", _measurement)
    monkeypatch.setattr(batch, "verify_product_model_proposal", lambda *_: pytest.fail("v2 must not fabricate a v1 Plan"))
    monkeypatch.setattr(batch, "_verify_v2_live_submission", lambda *_: {"scope": "controlled_origin_fixture"})
    monkeypatch.setattr("src.mod_agent.evaluation.strategy_native_export.validate_native_strategy_execution",
        lambda _: {"status": "native_execution_verified"})
    lineage_calls = []

    def lineage(**kwargs):
        lineage_calls.append(kwargs)
        assert kwargs["native_execution_ref"] != kwargs["initial_native_ref"]
        if damage == "unlinked_resume":
            raise ValueError("controlled unrelated recovery")
        return {"chain": [{"kind": "checkpoint_resume"}]}

    monkeypatch.setattr("src.mod_agent.evaluation.native_execution_lineage.validate_native_execution_lineage", lineage)

    def run(command, **kwargs):
        result = ordinary_runner(command, **kwargs)
        root = Path(command[command.index("--run-root") + 1])
        original = json.loads((root / "run_result.json").read_bytes())
        native_ref = _save(root / "a/task-agent-execution/execution.json", {
            "schema_version": "task-agent-product-execution.v2", "run_id": root.name})
        _save(root / "run_result.json", {**original, "status": "awaiting_delivery", "protocol": "v2",
            "task_agent_execution_ref": native_ref})
        return result

    def completed(**kwargs):
        verified.append(kwargs)
        assert kwargs["receipt_ref"] == "v2/receipt"
        assert kwargs["artifact_store_root"].name == "a"
        return {"candidate_source_id": "source", "candidate_ref": "candidate", "status": "delivered"}

    monkeypatch.setattr("src.mod_agent.task_agent_v2_delivery.read_completed_v2_delivery", completed)
    monkeypatch.setattr("src.mod_agent.evaluation.strategy_adoption._evaluator_evidence", lambda _: {
        "status": "missing_original_evaluator_evidence" if damage == "unqualified" else "verified_repository_csharp"})

    def deliver(context):
        result = ordinary_delivery(context)
        root = Path(context["run_root"])
        native = context["initial_result"]["task_agent_execution_ref"]
        if damage in {"resume", "unlinked_resume"}:
            native = _save(root / "a/task-agent-execution/resume/execution.json", {
                "schema_version": "task-agent-product-execution.v2", "run_id": root.name})
        execution = _save(root / "delivery/export.json", {"native_execution_ref": {} if damage == "native" else native,
            "candidate_source_id": "source", "candidate_state": {"candidate_ref": "candidate"}})
        verdict = _save(root / "delivery/verdict.json", {"schema_version": "independent_strategy_verdict_v1",
            "independent": True, "batch_complete": True, "execution_ref": execution,
            "candidate_source_id": "other" if damage == "candidate" else "source",
            "evaluator_ref": {} if damage == "judge" else spec["judge_ref"],
            "task_success": True, "preservation_passed": True})
        return {**result, "v2_delivery_receipt_ref": None if damage == "receipt" else "v2/receipt",
            "verification_report_ref": verdict}

    report = run_product_trial_batch(spec, execute=True, command_runner=run, delivery_adapter=deliver)
    assert [kind for kind, _ in log] == ["normal", "delivery"] * 2
    assert all(row["status"] == ("delivered" if damage in {None, "resume"} else "infrastructure_failure") for row in report["trials"])
    if damage in {None, "resume"}:
        assert len(verified) == 2
        assert all(row["delivery"]["proposal_origin"]["verification"] == "native_v2_delivery_and_qualified_independent_verdict"
            for row in report["trials"])
        if damage == "resume":
            assert len(lineage_calls) == 2
            assert all(row["delivery"]["proposal_origin"]["recovery_chain"] for row in report["trials"])
    else:
        assert not any(row["success"] for row in report["trials"])


@pytest.mark.parametrize("adapter_fails", [False, True])
def test_delivery_recovery_cost_is_remeasured_even_when_adapter_fails(tmp_path, monkeypatch, adapter_fails):
    from src.mod_agent.evaluation import product_trial_batch as batch

    log, measurements = [], []

    def measurement(**kwargs):
        refs = kwargs.get("execution_refs", [])
        measurements.append(refs)
        value = _measurement(**kwargs)
        if refs:
            value["model"]["usage"]["input_tokens"]["total"] = 175
            value["model"]["usage"]["output_tokens"]["total"] = 17
            value["model"]["generation_records"] = 3
        return value

    monkeypatch.setattr(batch, "measure_product_task", measurement)

    def delivery(context):
        root = Path(context["run_root"])
        reference = _save(root / "a/resume/recovery-execution.json", {"controlled": True})
        boundary = _save(root / "run-continuations/recovery.json", {"schema": "run_outcome_boundary_v1",
            "run_id": root.name, "request_id": context["initial_manifest"]["request_id"],
            "result": {"recovery_execution_ref": reference}})
        manifest = json.loads((root / "run_manifest.json").read_bytes())
        _save(root / "run_manifest.json", {**manifest, "outcome_boundaries": [boundary, boundary]})
        if adapter_fails:
            raise RuntimeError("controlled adapter failure after recovery")
        return {"status": "failed", "reason": "controlled independent rejection"}

    report = run_product_trial_batch(_spec(tmp_path), execute=True, command_runner=_runner(log), delivery_adapter=delivery)
    for row in report["trials"]:
        assert len(row["recovery_execution_refs"]) == 1
        if adapter_fails:
            assert row["failure"]["message"] == "controlled adapter failure after recovery"
            assert row["metrics"]["input_tokens"] == 175
            assert row["metrics"]["output_tokens"] == 17
        else:
            assert row["status"] == "failed"
            # No public delivery execution receipt: unmeasured extra cost is unknown.
            assert row["metrics"]["input_tokens"] is None
    assert sum(bool(refs) for refs in measurements) == 2


def test_missing_recovery_cost_does_not_invent_zero_or_erase_verified_delivery(tmp_path, monkeypatch):
    from src.mod_agent.evaluation import product_trial_batch as batch

    log = []
    monkeypatch.setattr(batch, "measure_product_task", _measurement)
    deliver = _delivery(log)

    def adapter(context):
        root = Path(context["run_root"])
        manifest = json.loads((root / "run_manifest.json").read_bytes())
        boundary = _save(root / "run-continuations/missing-observations.json", {
            "schema": "run_outcome_boundary_v1", "run_id": root.name,
            "request_id": manifest["request_id"], "event": "recovery_return", "result": {}})
        _save(root / "run_manifest.json", {**manifest, "outcome_boundaries": [boundary]})
        return deliver(context)

    report = run_product_trial_batch(_spec(tmp_path), execute=True, command_runner=_runner(log), delivery_adapter=adapter)
    for row in report["trials"]:
        assert row["success"] is True
        assert "unavailable" in row["delivery_measurement_failure"]["message"]
        assert row["metrics"]["input_tokens"] is None
        assert row["metrics"]["full_delivery_cost"] is False


def test_already_measured_recovery_receipt_does_not_erase_known_cost(tmp_path, monkeypatch):
    from src.mod_agent.evaluation import product_trial_batch as batch

    log = []

    def measurement(**kwargs):
        value = _measurement(**kwargs)
        value["selected_execution_refs"] = kwargs.get("execution_refs", [])
        if value["selected_execution_refs"]:
            value["model"]["usage"]["input_tokens"]["total"] = 175
            value["model"]["generation_records"] = 3
        return value

    monkeypatch.setattr(batch, "measure_product_task", measurement)
    deliver = _delivery(log)

    def adapter(context):
        root = Path(context["run_root"])
        reference = _save(root / "a/resume/recovery-execution.json", {"run_id": root.name,
            "model_calls_performed": 1, "execution_trace": {"trace_id": "recovery"},
            "execution_observations": {"new-model": {"id": "new-model", "kind": "generation"}}})
        manifest = json.loads((root / "run_manifest.json").read_bytes())
        boundary = _save(root / "run-continuations/recovery.json", {"schema": "run_outcome_boundary_v1",
            "run_id": root.name, "request_id": manifest["request_id"], "event": "recovery_return",
            "result": {"recovery_execution_ref": reference}})
        _save(root / "run_manifest.json", {**manifest, "outcome_boundaries": [boundary]})
        result = deliver(context)
        result["execution_refs"].append(reference)
        return result

    report = run_product_trial_batch(_spec(tmp_path), execute=True, command_runner=_runner(log), delivery_adapter=adapter)
    for row in report["trials"]:
        assert row["success"] is True and row["metrics"]["full_delivery_cost"] is True
        assert row["metrics"]["input_tokens"] == 175
        assert row["metrics"]["model_generation_records"] == 3


def test_meter_only_identity_is_explicit_not_missing_or_invented():
    from src.mod_agent.evaluation.product_trial_batch import _comparison_identity

    manifest = {"request": "task", "source_snapshot_sha256": "source", "model_admission": {"model": "m", "model_digest": "digest"},
        "ollama_patch_generation_profile": {"model": "m"}}
    result = {"model_call_budget": {"policy": "meter_only", "maximum": None}}
    assert _comparison_identity(manifest, result, {"sha256": "judge"})["budget_maximum"] is None
    for budget in ({"policy": "meter_only"}, {"policy": "task_total_only", "maximum": None}):
        assert _comparison_identity(manifest, {"model_call_budget": budget}, {"sha256": "judge"}) is None


def test_normal_batch_cli_continues_v2_pending_delivery_and_keeps_failure(tmp_path, monkeypatch, capsys):
    from scripts.run_product_trial_batch import main
    from src.mod_agent.evaluation import product_trial_batch as batch

    spec, log = _spec(tmp_path), []
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def deliver(context):\n    return {'status': 'failed', 'reason': 'controlled independent rejection'}\n", encoding="utf-8")
    spec["delivery_adapter_ref"] = {"path": str(adapter), "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()}
    original_runner = _runner(log)

    def run(command, **kwargs):
        output = original_runner(command, **kwargs)
        root = Path(command[command.index("--run-root") + 1])
        result = json.loads((root / "run_result.json").read_bytes())
        _save(root / "run_result.json", {**result, "status": "awaiting_delivery", "protocol": "v2"})
        return output

    monkeypatch.setattr(batch, "_run_command", run)
    monkeypatch.setattr(batch, "measure_product_task", _measurement)
    selection = _save(tmp_path / "spec.json", spec)
    result_path = tmp_path / "batch.json"
    assert main(["--spec", selection["path"], "--output", str(result_path), "--execute"]) == 0
    assert json.loads(capsys.readouterr().out)["executed"] is True
    result = json.loads(result_path.read_bytes())
    assert len(result["trials"]) == 2
    assert all(row["status"] == "failed" and row["delivery"]["reason"] == "controlled independent rejection"
        and row["success"] is False for row in result["trials"])


@pytest.mark.parametrize("damage", [None, "different_response", "different_request", "nonlive", "incomplete"])
def test_v2_submit_must_match_the_exact_live_request_response(tmp_path, damage):
    from src.mod_agent.evaluation.product_trial_batch import _verify_v2_live_submission
    from src.mod_agent.model_qualification import canonical_hash
    from src.mod_agent.task_agent_v2_state import save_v2_artifact

    response = {"tool_calls": [{"call_id": "submit", "name": "candidate_submit", "arguments": {}}]}
    response_ref = _save(tmp_path / "response.json", response)
    request_ref = _save(tmp_path / "request.json", {"messages": ["actual"]})
    provider_ref = _save(tmp_path / "provider.json", {"messages": ["actual wire request"]})
    candidate = save_v2_artifact(tmp_path / "a", "candidate", {"model_origin": {"call_id": "submit", "response_ref": response_ref}})
    call = {"generation_span_id": "generation", "trace_id": "trace", "response_sha256": canonical_hash(response),
        "transport": {"response_origin": "live_provider_response", "response_received": True, "response_complete": True},
        "local_evidence": {role: {"integrity": "verified_bytes", "sha256": ref["sha256"]}
            for role, ref in (("request", request_ref), ("provider_request", provider_ref))}}
    if damage == "different_response":
        call["response_sha256"] = "unrelated-live-response"
    elif damage == "different_request":
        call["local_evidence"]["request"]["sha256"] = "other-request"
    elif damage == "nonlive":
        call["transport"]["response_origin"] = "controlled"
    elif damage == "incomplete":
        call["transport"]["response_complete"] = False
    args = (tmp_path, {"provider_turns": [{"response_ref": response_ref, "request_ref": request_ref,
        "provider_request_ref": provider_ref}]}, {"candidate_ref": candidate}, {"model": {"calls": [call]}})
    if damage is None:
        assert _verify_v2_live_submission(*args)["call_id"] == "submit"
    else:
        with pytest.raises(ValueError, match="retained live model response"):
            _verify_v2_live_submission(*args)


def test_serial_full_delivery_collects_usage_and_repeats_do_not_add_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", _measurement)
    log, checkpoints = [], []
    report = run_product_trial_batch(_spec(tmp_path, repeats=2), execute=True,
        command_runner=_runner(log), delivery_adapter=_delivery(log), checkpoint=checkpoints.append)
    assert [kind for kind, _ in log] == ["normal", "delivery"] * 4
    assert report["summary"]["distinct_task_count"] == 1 and report["summary"]["trial_count"] == 4
    assert report["summary"]["independent_task_count"] is None
    assert report["summary"]["complete_paired_delivery_resource_comparison"]
    assert report["summary"]["input_token_reduction"] == 0
    assert report["summary"]["by_arm"]["baseline"]["input_tokens"] == 200
    assert report["summary"]["by_arm"]["baseline"]["tokens_per_successful_delivery"] == 110
    assert report["summary"]["by_arm"]["baseline"]["observed_tool_calls"] == 4
    assert checkpoints[0]["trials"][0]["status"] == "running"
    assert checkpoints[0]["trials"][1]["status"] == "pending"
    assert all(row["execution_seconds"] >= row["normal_entry_seconds"] for row in report["trials"])


@pytest.mark.parametrize("failed_output,expected", [(30, 200), (None, None)])
def test_cost_per_success_includes_failed_output_and_preserves_unknown(failed_output, expected):
    rows = [{"task_id": "task", "repeat": 1, "arm": "candidate", "status": "delivered", "success": True,
             "execution_seconds": 1, "metrics": {"input_tokens": 100, "output_tokens": 20}},
            {"task_id": "task", "repeat": 2, "arm": "candidate", "status": "product_failed", "success": False,
             "execution_seconds": 1, "metrics": {"input_tokens": 50, "output_tokens": failed_output}}]
    assert summarize_trials(rows)["by_arm"]["candidate"]["tokens_per_successful_delivery"] == expected


def test_failure_is_retained_and_never_becomes_a_resource_saving(tmp_path, monkeypatch):
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", _measurement)
    log = []
    report = run_product_trial_batch(_spec(tmp_path), execute=True,
        command_runner=_runner(log, failed_arm="baseline"), delivery_adapter=_delivery(log))
    assert report["trials"][0]["status"] == "product_failed"
    assert log == [("normal", "baseline_1"), ("normal", "candidate_1"), ("delivery", "candidate_1")]
    assert report["summary"]["by_arm"]["baseline"]["completion_fraction"] == 0
    assert report["summary"]["input_token_reduction"] is None


@pytest.mark.parametrize("failure", ["startup", "delivery"])
def test_infrastructure_failure_does_not_drop_selected_trials(tmp_path, monkeypatch, failure):
    log = []
    if failure != "startup":
        monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", _measurement)
    def fail_delivery(_context):
        raise RuntimeError("judge infrastructure unavailable")
    report = run_product_trial_batch(_spec(tmp_path), execute=True, command_runner=_runner(log, missing_run=failure == "startup"),
        delivery_adapter=fail_delivery)
    assert len(report["trials"]) == 2
    assert all(row["status"] == "infrastructure_failure" for row in report["trials"])
    assert report["summary"]["input_token_reduction"] is None


@pytest.mark.parametrize("structured", [True, False])
def test_missing_terminal_keeps_launcher_failure_before_measurement_error(tmp_path, monkeypatch, structured):
    reason = "PRODUCT_MODEL_CALL_BUDGET_EXHAUSTED"
    stdout = json.dumps({"status": "CONTRACT_ERROR", "reason": reason}) if structured else "progress before failure\n"
    stderr = "" if structured else "原始失败\n"

    def run(_command, **_kwargs):
        return subprocess.CompletedProcess(_command, 2, stdout, stderr)

    def missing_terminal(**_kwargs):
        raise FileNotFoundError("run_result.json")

    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", missing_terminal)
    checkpoints = []
    report = run_product_trial_batch(_spec(tmp_path), execute=True, command_runner=run,
        delivery_adapter=lambda _: pytest.fail("No terminal record permits delivery"), checkpoint=checkpoints.append)
    for row in report["trials"]:
        assert row["launcher_stdout"] == stdout
        assert row["launcher_stderr"] == stderr
        assert row["failure"]["type"] == "LauncherExitError"
        assert row["failure"]["exit_code"] == 2
        assert row["terminal_record_failure"]["type"] == "FileNotFoundError"
        assert row["success"] is False and row["metrics"] is None
        if structured:
            assert row["failure"]["message"] == reason
            assert row["failure"]["reported_status"] == "CONTRACT_ERROR"
    assert checkpoints[-1]["trials"] == report["trials"]


def test_zero_exit_missing_terminal_retains_output_without_claiming_delivery(tmp_path, monkeypatch):
    stdout = '{"status":"completed"}'
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("run_result.json")))
    report = run_product_trial_batch(_spec(tmp_path), execute=True,
        command_runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout, ""),
        delivery_adapter=lambda _: pytest.fail("Unbound stdout cannot authorize delivery"))
    assert all(row["launcher_stdout"] == stdout and row["success"] is False for row in report["trials"])
    assert all(row["failure"]["type"] == "FileNotFoundError" for row in report["trials"])


@pytest.mark.parametrize("change", ["developer", "additional_models", "missing_tools", "different_budget"])
def test_incomplete_or_unfair_cost_scope_cannot_claim_resource_benefit(tmp_path, monkeypatch, change):
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", _measurement)
    log = []
    report = run_product_trial_batch(_spec(tmp_path), execute=True,
        command_runner=_runner(log, changed_budget=change == "different_budget"),
        delivery_adapter=_delivery(log, origin="developer_control" if change == "developer" else "product_model_proposal",
            model_calls=1 if change == "additional_models" else 0, missing_observations=change == "missing_tools"))
    assert not report["summary"]["complete_paired_delivery_resource_comparison"]
    assert report["summary"]["input_token_reduction"] is None
    if change == "developer":
        assert not any(row["success"] for row in report["trials"])


@pytest.mark.parametrize("change", ["duplicate", "escape", "argv", "snapshot"])
def test_invalid_trial_selection_is_rejected_before_execution(tmp_path, change):
    spec = _spec(tmp_path)
    if change == "duplicate":
        spec["trials"][1] = deepcopy(spec["trials"][0])
    elif change == "escape":
        spec["trials"][0]["allowed_root"] = str(tmp_path / "elsewhere")
    elif change == "snapshot":
        spec["trials"][0]["snapshot_root"] = spec["trials"][0]["run_root"]
    else:
        argv = spec["trials"][0]["business_argv"]
        argv[argv.index("--workspace-root") + 1] = str(tmp_path / "different")
    with pytest.raises(ValueError):
        run_product_trial_batch(spec)


def test_existing_run_is_not_overwritten_and_other_trial_still_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", _measurement)
    spec = _spec(tmp_path)
    original = Path(spec["trials"][0]["run_root"])
    original.mkdir(parents=True)
    marker = original / "history.txt"
    marker.write_text("immutable history")
    log = []
    report = run_product_trial_batch(spec, execute=True, command_runner=_runner(log), delivery_adapter=_delivery(log))
    assert report["trials"][0]["status"] == "infrastructure_failure"
    assert marker.read_text() == "immutable history"
    assert report["trials"][1]["success"] is True


def test_summary_retains_unknown_metrics_instead_of_zero():
    rows = [{"task_id": "t", "arm": "baseline", "repeat": 1, "status": "failed", "success": False,
        "metrics": {"input_tokens": None}}]
    result = summarize_trials(rows)
    assert result["by_arm"]["baseline"]["input_tokens"] is None
    assert result["by_arm"]["baseline"]["tokens_per_successful_delivery"] is None


def test_different_platform_versions_within_one_arm_are_not_pooled_as_a_fixed_strategy(tmp_path, monkeypatch):
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", _measurement)
    log = []
    report = run_product_trial_batch(_spec(tmp_path, repeats=2), execute=True,
        command_runner=_runner(log), delivery_adapter=_delivery(log))
    report["trials"][-1]["platform_execution"]["platform_files_sha256"] = "other-version"
    summary = summarize_trials(report["trials"])
    assert not summary["fixed_platform_per_arm"]
    assert summary["input_token_reduction"] is None


def test_judge_drift_during_delivery_is_recorded_without_claiming_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", _measurement)
    spec, log = _spec(tmp_path), []
    ordinary = _delivery(log)
    def delivery(context):
        result = ordinary(context)
        Path(context["judge_ref"]["path"]).write_text("changed judge")
        return result
    report = run_product_trial_batch(spec, execute=True, command_runner=_runner(log), delivery_adapter=delivery)
    assert all(row["status"] == "infrastructure_failure" for row in report["trials"])
    assert report["summary"]["input_token_reduction"] is None
    assert len(log) == 2  # No second model command under a changed judge.


def test_cli_prepare_does_not_import_delivery_adapter_or_launch_process(tmp_path, monkeypatch, capsys):
    from scripts.run_product_trial_batch import main

    spec = _spec(tmp_path)
    adapter = tmp_path / "adapter.py"
    adapter.write_text("raise AssertionError('prepare must not import me')")
    spec["delivery_adapter_ref"] = {"path": str(adapter), "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()}
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(spec))
    output = tmp_path / "batch.json"
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("prepare must not launch"))
    assert main(["--spec", str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["executed"] is False and report["summary"]["trial_count"] == 2
    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    with pytest.raises(ValueError, match="earlier records"):
        main(["--spec", str(source), "--output", str(output)])


def test_interrupted_trial_and_pending_rows_survive_checkpoint(tmp_path):
    checkpoints = []
    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run_product_trial_batch(_spec(tmp_path), execute=True, command_runner=interrupted,
            delivery_adapter=lambda _: {}, checkpoint=checkpoints.append)
    assert [row["status"] for row in checkpoints[-1]["trials"]] == ["interrupted", "pending"]
    assert checkpoints[-1]["summary"]["trial_count"] == 2


def test_owned_process_deadline_stops_launcher_tree_and_records_timeout(monkeypatch):
    from src.mod_agent.evaluation.product_trial_batch import _run_command
    waits, stops = [], []
    class Process:
        pid = 42317
        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise subprocess.TimeoutExpired("controlled-launcher", timeout)
            return 124
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: stops.append(command))
    monkeypatch.setattr(os, "killpg", lambda *args: stops.append(args), raising=False)
    result = _run_command(["controlled-launcher"], timeout=7)
    assert result.deadline_exceeded is True and result.returncode == 124
    assert 0 < waits[0] <= 7 and waits[1:] == [30] and len(stops) == 1
    if os.name == "nt":
        assert stops[0] == ["taskkill", "/PID", "42317", "/T", "/F"]


@pytest.mark.parametrize('inherited,timeout,expected', [(None, 200, 1200), ('1050', 200, 1050),
    ('1400', 200, 1200), ('1050', None, 1050), (None, None, None)])
def test_trial_launcher_forwards_effective_deadline(monkeypatch, inherited, timeout, expected):
    from src.mod_agent.evaluation import product_trial_batch as batch

    if inherited is None:
        monkeypatch.delenv('REPOFORGE_TASK_DEADLINE_UTC', raising=False)
    else:
        monkeypatch.setenv('REPOFORGE_TASK_DEADLINE_UTC', inherited)
    monkeypatch.setattr(batch.time, 'time', lambda: 1000)
    monkeypatch.setattr(batch.time, 'monotonic', lambda: 100)
    received = {}

    def start(_command, **kwargs):
        received.update(kwargs)
        def wait(timeout):
            received['timeout'] = timeout
            return 0
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(subprocess, 'Popen', start)
    result = batch._run_command(['controlled'], timeout=timeout)
    assert result.deadline_utc == expected
    actual = received['env'].get('REPOFORGE_TASK_DEADLINE_UTC')
    assert (float(actual) if actual is not None else None) == expected
    assert received['timeout'] == (expected - 1000 if expected is not None else None)


def test_expired_trial_window_does_not_launch(monkeypatch):
    from src.mod_agent.evaluation.product_trial_batch import _run_command

    monkeypatch.setenv('REPOFORGE_TASK_DEADLINE_UTC', '0')
    monkeypatch.setattr(subprocess, 'Popen', lambda *_a, **_k: pytest.fail('Expired window launched a process'))
    result = _run_command(['controlled'], timeout=200)
    assert result.deadline_exceeded and result.returncode == 124


def test_cli_adapter_load_failure_is_retained_without_launching_models(tmp_path, monkeypatch):
    from scripts.run_product_trial_batch import main

    spec = _spec(tmp_path)
    adapter = tmp_path / "broken.py"
    adapter.write_text("raise RuntimeError('controlled adapter load failure')")
    spec["delivery_adapter_ref"] = {"path": str(adapter), "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()}
    path, output = tmp_path / "spec.json", tmp_path / "batch.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: pytest.fail("No model process under a broken adapter"))
    assert main(["--spec", str(path), "--output", str(output), "--execute"]) == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["trial_count"] == 2
    assert all(row["failure"]["stage"] == "adapter_load" for row in report["trials"])


def test_no_live_product_generation_cannot_be_counted_as_product_success(tmp_path, monkeypatch):
    def preparation(**kwargs):
        result = _measurement(**kwargs)
        result["model"]["response_origins"] = {"controlled": 2}
        return result
    monkeypatch.setattr("src.mod_agent.evaluation.product_trial_batch.measure_product_task", preparation)
    log = []
    report = run_product_trial_batch(_spec(tmp_path), execute=True, command_runner=_runner(log), delivery_adapter=_delivery(log))
    assert not any(row["success"] for row in report["trials"])
    assert report["summary"]["input_token_reduction"] is None


@pytest.mark.parametrize("damage", [None, "native_content", "replay", "plan", "proposal", "escape"])
def test_origin_reader_joins_bound_judge_proposal_and_retained_native_response(tmp_path, monkeypatch, damage):
    documents = {}
    paths = {}
    for kind in ("feature_spec", "evidence_pack", "implementation_plan", "patch_proposal"):
        path = tmp_path / "a" / (kind + ".json")
        _save(path, {"artifact_hash": "a" * 64})
        documents[kind] = {"owned": kind}
        paths[kind] = str(path)
    raw_content = '{"changes":[]}'
    response_hash = hashlib.sha256(raw_content.encode()).hexdigest()
    documents["patch_proposal"] = {"diff_intent": [{"generation": {"model_response_sha256": response_hash}}]}
    report = {"status": "passed", "execution_origin": "executed", "model_response_sha256": response_hash,
        "generated_changes_sha256": "g" * 64, "implementation_plan_ref": {
            "feature_id": "f", "artifact_type": "implementation_plan", "path": paths["implementation_plan"],
            "artifact_hash": "a" * 64}}
    if damage == "plan":
        report["implementation_plan_ref"]["artifact_hash"] = "b" * 64
    if damage == "proposal":
        documents["patch_proposal"]["diff_intent"][0]["generation"]["model_response_sha256"] = "0" * 64
    if damage == "escape":
        paths["patch_proposal"] = str(tmp_path.parent / "outside.json")
    response = _save(tmp_path / "native-response.json", {"content": raw_content if damage != "native_content" else "changed"})
    measurement = {"model": {"calls": [{"transport": {"response_origin": "replay" if damage == "replay" else "live_provider_response"},
        "local_evidence": {"response": {"file": "native-response.json", "sha256": response["sha256"], "integrity": "verified_bytes"}}}]}}
    monkeypatch.setattr("src.mod_agent.artifacts.ArtifactStore.semantic_read_immutable", lambda self, ref: {"data": documents[ref.artifact_type]})
    calls = []
    monkeypatch.setattr("src.mod_agent.model_qualification._verify_materialized_response", lambda **kwargs: calls.append(kwargs))
    arguments = (tmp_path, {"feature_id": "f", "artifacts": paths}, measurement, _save(tmp_path / "verification.json", report))
    if damage:
        with pytest.raises(ValueError):
            verify_product_model_proposal(*arguments)
        assert calls == []
    else:
        result = verify_product_model_proposal(*arguments)
        assert result["model_response_sha256"] == response_hash
        assert calls[0]["workspace_root"] is None
        assert calls[0]["parsed_output"] == {"changes": []}
        assert calls[0]["expected_changes_hash"] == report["generated_changes_sha256"]
