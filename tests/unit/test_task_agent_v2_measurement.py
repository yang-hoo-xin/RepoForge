"""Entry wait and closed-loop input provenance across interruption and recovery."""
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from src.mod_agent import cli
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_execution import (
    capture_clock,
    classify_v2_technical_inputs,
    task_timing_snapshot,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2State
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)
from tests.unit.test_task_agent_v2_recovery import (
    test_normal_entry_interruption_reconstructs_owner_and_finishes_same_product as run_interrupted_recovery,
)


def test_normal_interrupted_resume_keeps_entry_origin_and_exports_total_observed_wait(repository_workspace, monkeypatch, capsys):
    root = repository_workspace
    run_interrupted_recovery(root, monkeypatch, capsys, "checkpoint_completed")
    paths = list((root / "runs").glob("*/a/task-agent-execution/resume-*/execution.json"))
    assert len(paths) == 1
    native_ref = _ref(paths[0])
    native = read_bound_json(native_ref)
    assert not (paths[0].parent.parent / "execution.json").exists()
    origin_ref = native["task_timing_origin_ref"]
    origin = read_bound_json(origin_ref)
    assert origin_ref == _ref(Path(native["workspace_root"]) / "entry-timing.json")
    store = paths[0].parent.parent.parent
    state = TaskAgentV2State(store, native["task_contract_ref"])
    checkpoints = [json.loads(path.read_bytes()) for path in (store / "v2/session-checkpoint").glob("*.json")]
    interrupted = [row for row in checkpoints if row["loop"]["model_turns"] == 5 and row["loop"]["phase"] == "dispatch_ready"]
    assert interrupted and all(row["task_timing_observation"]["origin_ref"] == origin_ref for row in interrupted)
    assert all(row["task_timing_observation"]["whole_task_finished"] is False for row in interrupted)
    returned = [json.loads(path.read_bytes()) for path in paths[0].parent.glob("task-observation-*.json")]
    final = next(row for row in returned if row["event"] == "resumed_product_return")
    assert final["origin_ref"] == origin_ref and final["duration_ms"] >= min(row["task_timing_observation"]["duration_ms"] for row in interrupted)
    assert datetime.fromisoformat(final["observed_clock"]["received_at"]) >= datetime.fromisoformat(origin["entry_clock"]["received_at"])
    baseline = state.load(state.contract["baseline_manifest_ref"])
    source = _save(root / "source-facts.json", {"task_id": "recovered", "task_family": "arithmetic", "source_version": "controlled",
        "source_files_sha256": canonical_hash({p: row["sha256"] for p, row in baseline["files"].items()}),
        "original_request_sha256": canonical_hash(state.load(state.contract["original_request_ref"])["original_request"])})
    exposure = _save(root / "exposure.json", {"task_id": "recovered", "attempt": 1, "source_ref": source, "exposure": "exposed"})
    protocol = _save(root / "protocol.json", {"schema_version": "strategy_trial_protocol_v1", "invariants": {},
        "invariants_sha256": canonical_hash({}), "task_families": ["arithmetic"], "snapshot_ref": None,
        "planned_pairs": [{"task_id": "recovered", "task_family": "arithmetic", "attempt": 1,
            "source_ref": source, "exposure_ref": exposure, "current_context": {}}]})
    call = next(row for row in native["completed_tool_records"] if row["name"] == "candidate_submit")
    code = cli.main(["export-strategy-execution", "--execution-record", native_ref["path"], "--call-id", call["call_id"],
        "--protocol-record", protocol["path"], "--task-id", "recovered", "--arm", "B", "--attempt", "1",
        "--output-root", str(root / "export")])
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    exported = read_bound_json(result["execution_ref"])
    assert exported["known_cost"]["duration_ms"] == final["duration_ms"]
    assert exported["resource_evidence"]["duration_source"]["boundary"] == "resumed_product_return"
    assert exported["resource_evidence"]["human_intervention_source"]["closed_loop_human_technical_interventions"] == 0
    assert exported["human_technical_interventions"] is None


def test_clock_cross_process_and_regression_never_invent_task_end(tmp_path):
    clock = capture_clock()
    origin = _save(tmp_path / "origin.json", {"schema": "task-agent-entry-timing.v2", "run_id": "test", "request_id": "test",
        "entry_clock": clock})
    child = subprocess.run([sys.executable, "-B", "-c",
        "import json,sys;from src.mod_agent.task_agent_v2_execution import task_timing_snapshot;"
        "print(json.dumps(task_timing_snapshot(json.loads(sys.argv[1]),'checkpoint_observed','pending')))", json.dumps(origin)],
        capture_output=True, text=True, check=True)
    observed = json.loads(child.stdout)
    assert observed["clock_basis"] == "cross_process_utc_wall_clock" and observed["duration_ms"] >= 0
    assert observed["whole_task_finished"] is False
    forked = task_timing_snapshot(origin, "checkpoint_observed", "pending",
        observed_clock={**capture_clock(), "process_id": clock["process_id"] + 1})
    assert forked["clock_basis"] == "cross_process_utc_wall_clock"
    earlier = {**clock, "process_instance": "other", "received_at": (datetime.fromisoformat(clock["received_at"]) - timedelta(seconds=1)).isoformat()}
    regressed = task_timing_snapshot(origin, "checkpoint_observed", "unknown", observed_clock=earlier)
    assert regressed["duration_ms"] is None and regressed["timing_status"] == "clock_regressed"
    assert task_timing_snapshot(None, "legacy", "unknown")["duration_ms"] is None


def test_closed_input_scope_does_not_label_unattributed_advice_as_zero_intervention():
    initial = {"messages": [{"role": "system", "content": "task protocol"}, {"role": "user", "content": {"task": "fix"}}]}
    prepared = deepcopy(initial)
    prepared["messages"].append({"role": "user", "content": {"current_task_state": {"origin": "execution_host"}}})
    closed = classify_v2_technical_inputs(initial, prepared, initial, [], [])
    assert closed["closed_loop_human_technical_interventions"] == 0
    assert closed["outside_scope_human_technical_interventions"] is None
    injected = deepcopy(initial)
    injected["messages"].append({"role": "user", "content": "Developer: replace this method with my patch"})
    unknown = classify_v2_technical_inputs(injected, injected, initial, [], [])
    assert unknown["closed_loop_human_technical_interventions"] is None and unknown["unknown_input_sources"]
