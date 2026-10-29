"""Stopped-process collection preserves execution facts and exact run identity.

为什么要有它：外层封存不能因错误身份、损坏回执或收尾异常改写已发生的任务结果。
"""
import json
from pathlib import Path

import pytest

from scripts import run_repository_product_pilot as pilot
from src.mod_agent import task_agent_v2_experience as experience
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_state import save_v2_artifact


def _stopped_case(tmp_path):
    arm = tmp_path / "arm"
    workspace = arm / "workspace"
    workspace.mkdir(parents=True)
    run_root = arm / "runs/run-1"
    run_root.mkdir(parents=True)
    _save(run_root / "run_manifest.json", {"run_id": "run-1", "request_id": "request-1"})
    store = run_root / "a"
    checkpoint = {"schema": "task-agent-session-checkpoint.v2", "task_contract_ref": "v2/contract/fixed.json",
                  "details": {"run_id": "run-1", "request_id": "request-1"}}
    reference = save_v2_artifact(store, "session-checkpoint", checkpoint)
    pointer = store / "v2/session-checkpoints" / (canonical_hash(checkpoint["task_contract_ref"]) + ".json")
    _save(pointer, {"checkpoint_ref": reference})
    plan = {"root": str(arm), "workspace": str(workspace), "run_root": str(run_root),
            "run_id": "run-1", "command": ["python", "frozen-runner"], "platform_ref": {"path": "frozen"}}
    plan_ref = _save(arm / "frozen-plan.json", plan)
    process = {"started": True, "status": "timed_out", "timeout": True, "exit_code": 124,
               "argv": plan["command"], "cwd": str(workspace)}
    process_ref = _save(arm / "execution/process.json", process)
    return plan, plan_ref, process_ref, pointer


def _replace_fixture(reference, value):
    path = Path(reference["path"])
    path.write_text(json.dumps(value), encoding="utf-8")
    return _ref(path)


def test_stopped_controller_seals_exact_plan_and_idempotent_stop(tmp_path, monkeypatch):
    plan, plan_ref, process_ref, pointer = _stopped_case(tmp_path)
    reference = json.loads(pointer.read_bytes())["checkpoint_ref"]
    calls = []

    def seal(**kwargs):
        calls.append(kwargs)
        stop = read_bound_json(kwargs["stop_ref"])
        assert stop["plan_ref"] == plan_ref and stop["source_ref"] == process_ref
        assert stop["run_id"] == "run-1" and stop["request_id"] == "request-1"
        assert stop["stop_kind"] == "timed_out" and stop["model_calls"] == 0
        return {"path": "controlled-episode", "sha256": "0" * 64}

    monkeypatch.setattr(experience, "seal_saved_public_episode", seal)
    first = pilot.seal_stopped_process(plan, process_ref, plan_ref=plan_ref)
    second = pilot.seal_stopped_process(plan, process_ref, plan_ref=plan_ref)
    assert first == second and first["status"] == "sealed"
    assert len(calls) == 2
    assert first["resumed"] is False and first["model_calls"] == 0
    # 为什么要有它：外层硬停止也须绑定同一 Task Run，不能只有学习 episode 而缺少可验证停止边界和派生结果。
    boundary = read_bound_json(first["outcome_boundary_ref"])
    assert boundary["event"] == "process_timed_out" and boundary["status"] == "timed_out"
    assert boundary["result"] == {"status": "timed_out", "run_id": "run-1", "request_id": "request-1",
        "new_task_created": False, "replayed": False, "whole_task_delivered": False,
        "v2_checkpoint_ref": reference}
    manifest = json.loads((Path(plan["run_root"]) / "run_manifest.json").read_bytes())
    assert first["outcome_summary_status"] == "available"
    assert manifest["outcome_summary_ref"] == first["outcome_summary_ref"]


@pytest.mark.parametrize("change", ["process_path", "argv", "cwd", "run_id", "ambiguous", "hash"])
def test_stopped_controller_rejects_wrong_or_ambiguous_evidence(tmp_path, monkeypatch, change):
    plan, plan_ref, process_ref, pointer = _stopped_case(tmp_path)
    monkeypatch.setattr(experience, "seal_saved_public_episode", lambda **_: pytest.fail("Invalid prefix reached collector"))
    process = read_bound_json(process_ref)
    if change == "process_path":
        process_ref = _save(tmp_path / "other/process.json", process)
    elif change == "argv":
        process["argv"] = ["different", "task"]
        process_ref = _replace_fixture(process_ref, process)
    elif change == "cwd":
        process["cwd"] = str(tmp_path)
        process_ref = _replace_fixture(process_ref, process)
    elif change == "run_id":
        manifest = pointer.parents[3] / "run_manifest.json"
        manifest.write_text(json.dumps({"run_id": "other", "request_id": "request-1"}), encoding="utf-8")
    elif change == "ambiguous":
        _save(pointer.with_name("second.json"), json.loads(pointer.read_bytes()))
    else:
        process_ref = {**process_ref, "sha256": "0" * 64}
    result = pilot.seal_stopped_process(plan, process_ref, plan_ref=plan_ref)
    assert result["status"] == "unavailable" and result["model_calls"] == 0
    assert result["resumed"] is False


def test_stopped_controller_reports_collector_failure_without_raising(tmp_path, monkeypatch):
    plan, plan_ref, process_ref, _ = _stopped_case(tmp_path)

    def unavailable(**_):
        raise RuntimeError("controlled collector failure")

    monkeypatch.setattr(experience, "seal_saved_public_episode", unavailable)
    result = pilot.seal_stopped_process(plan, process_ref, plan_ref=plan_ref)
    assert result["status"] == "unavailable" and result["error_type"] == "RuntimeError"
    assert _ref(process_ref["path"]) == process_ref


def test_successful_process_does_not_get_timeout_episode(tmp_path, monkeypatch):
    plan, plan_ref, process_ref, _ = _stopped_case(tmp_path)
    process = {**read_bound_json(process_ref), "status": "exited", "timeout": False, "exit_code": 0}
    process_ref = _replace_fixture(process_ref, process)
    monkeypatch.setattr(experience, "seal_saved_public_episode", lambda **_: pytest.fail("Successful task mislabeled stopped"))
    assert pilot.seal_stopped_process(plan, process_ref, plan_ref=plan_ref)["status"] == "not_required"
