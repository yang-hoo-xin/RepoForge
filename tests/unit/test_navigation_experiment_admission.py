"""Experiment admission uses the same effective policy as repository execution.

为什么要有它：模型总次数不能代替读取与观察预算，配置不可达时应在派发前拒绝。
"""
import json

import pytest

from scripts import run_repository_product_pilot as pilot
from src.mod_agent.domain.full_project_compilation import _ref


def requirement():
    return {"evaluation_version": "controlled-navigation-v2", "threshold": 8,
        "navigation_observation_bytes": 65536, "navigation_seconds": 240,
        "post_trigger_reserve": {"reads": 2, "model_calls": 3, "observation_bytes": 16384, "seconds": 60}}


def configuration(tmp_path, *, reads=None, bytes_limit=81920, calls=16):
    arguments = ["--repository-profile", str(tmp_path / "profile.json"),
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
        "--model-call-budget-policy", "task_total_only", "--task-model-call-limit", str(calls)]
    if reads is not None:
        worker = tmp_path / "worker.json"
        worker.write_text(json.dumps({"schema_version": "patch_worker_configuration_v1",
            "enable_recovery_loop": False, "behavior_evaluator": None,
            "policy": {"model_call_budget_policy": "task_total_only", "max_total_model_turns": calls,
                "max_tool_rounds": reads, "max_observation_bytes": bytes_limit}}), encoding="utf-8")
        arguments += ["--patch-worker-config", str(worker)]
    path = tmp_path / "execution.json"
    path.write_text(json.dumps({"arguments": arguments}), encoding="utf-8")
    return _ref(path)


def test_default_three_reads_rejects_eight_navigation_experiment(tmp_path):
    reference = configuration(tmp_path)
    with pytest.raises(ValueError, match="NAVIGATION_EXPERIMENT_UNREACHABLE.*reads"):
        pilot.validate_navigation_experiment(reference, requirement(), wall_seconds=900)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("reads,byte_limit,calls,seconds,insufficient", [
    (9, 81920, 16, 900, "reads"),
    (10, 81919, 16, 900, "observation_bytes"),
    (10, 81920, 10, 900, "model_calls"),
    (10, 81920, 16, 299, "seconds"),
])
def test_each_resource_must_cover_navigation_and_followup(tmp_path, reads, byte_limit, calls, seconds, insufficient):
    reference = configuration(tmp_path, reads=reads, bytes_limit=byte_limit, calls=calls)
    with pytest.raises(ValueError, match="NAVIGATION_EXPERIMENT_UNREACHABLE.*" + insufficient):
        pilot.validate_navigation_experiment(reference, requirement(), wall_seconds=seconds)


def test_explicit_capacity_admitted_without_changing_threshold_or_policy(tmp_path):
    reference = configuration(tmp_path, reads=10)
    before = (tmp_path / "worker.json").read_bytes()
    result = pilot.validate_navigation_experiment(reference, requirement(), wall_seconds=900)
    assert result["status"] == "admitted"
    assert result["threshold"] == 8
    assert result["capacity"] == {"reads": 10, "model_calls": 16, "observation_bytes": 81920, "seconds": 900}
    assert result["required"] == {"reads": 10, "model_calls": 11, "observation_bytes": 81920, "seconds": 300}
    assert (tmp_path / "worker.json").read_bytes() == before


def test_lowering_threshold_does_not_admit_the_failed_experiment(tmp_path):
    requirements = {**requirement(), "threshold": 3}
    with pytest.raises(ValueError, match="NAVIGATION_EXPERIMENT_REQUIREMENTS_INVALID"):
        pilot.validate_navigation_experiment(configuration(tmp_path), requirements, wall_seconds=900)


@pytest.mark.parametrize("flag,value", [("--technical-executor", "workflow"), ("--task-agent-protocol", "v1")])
def test_non_task_agent_v2_entry_cannot_admit_navigation_experiment(tmp_path, flag, value):
    reference = configuration(tmp_path, reads=10)
    path = tmp_path / "execution.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["arguments"][document["arguments"].index(flag) + 1] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="NAVIGATION_EXPERIMENT_REQUIRES_TASK_AGENT_V2"):
        pilot.validate_navigation_experiment(_ref(path), requirement(), wall_seconds=900)


@pytest.mark.parametrize("mode", ["native_reference", "strategy_pair"])
def test_freeze_cannot_ignore_navigation_requirements_on_other_modes(tmp_path, mode):
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({"mode": mode, "navigation_feedback_experiment": requirement()}), encoding="utf-8")
    with pytest.raises(ValueError, match="NAVIGATION_EXPERIMENT_MODE_UNSUPPORTED"):
        pilot.freeze(_ref(path))
