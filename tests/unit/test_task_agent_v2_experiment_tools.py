"""Discovery feedback remains executed evidence even when catalog admission fails.

为什么要有它：测试发现是模型可选执行动作，失败不能丢失观察或继续沿用同版本旧验收。
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.task_agent_v2_experiment_tools import _discover_catalog
from src.mod_agent.task_agent_v2_experiments import ExperimentRejectedError, TaskAgentExperiments
from tests.unit.test_task_agent_v2_state import edit
from tests.unit.test_task_agent_v2_state import state as state


@pytest.mark.parametrize("mode", ["compile_failed", "test_failed", "catalog_invalid", "passed"])
def test_discovery_preserves_execution_and_invalidates_only_current_failure(state, tmp_path, mode):
    candidate = edit(state, "A.cs", "1", "2")["candidate_ref"]
    baseline = state.load(state.contract["baseline_manifest_ref"])
    calls = []

    def execute(scratch, *_args):
        calls.append(scratch)
        assert (scratch / "A.cs").read_text() == state.materialize(candidate)["A.cs"]
        return {"report_ref": {"path": "controlled-report"},
            "build_execution": {"exit_code": 1 if mode == "compile_failed" else 0},
            "test_execution": {"status": "not_executed"} if mode == "compile_failed" else {"exit_code": 0 if mode == "passed" else 1},
            "verification_summary": {"checks": 0 if mode == "compile_failed" else 1},
            "status": "passed" if mode in {"passed", "catalog_invalid"} else "failed", "observations": [mode]}

    backend = SimpleNamespace(source_root=Path(baseline["source_root"]), workspace_root=tmp_path,
        profile=SimpleNamespace(task_work_root=lambda _root: tmp_path / "scratch"), run_build_test=execute)
    runner = SimpleNamespace(backend=backend, identity=lambda: {"runner": "controlled"})
    experiments = TaskAgentExperiments(state=state, test_runner=runner)

    def catalog(_ref):
        if mode in {"compile_failed", "catalog_invalid"}:
            raise ExperimentRejectedError("CATALOG_NOT_EXECUTED_OR_INVALID")
        return {"catalog_ref": "controlled-catalog", "entries": ["a"]}

    experiments.test_catalog = catalog
    session = SimpleNamespace(state=state, feature_id="fixture", current_check="old-passed-check")
    result = _discover_catalog(session, experiments, candidate, "discover", {"call_id": "discover"})
    receipt = result.get("discovery_receipt", result)
    assert receipt["executed"] is True and receipt["grants_acceptance"] is False
    assert receipt["observations"]["tests_executed"] == (0 if mode == "compile_failed" else 1)
    if mode in {"compile_failed", "test_failed"}:
        assert session.current_check is None
        assert receipt["status"] == "failed"
    else:
        assert session.current_check == "old-passed-check"
    if mode in {"compile_failed", "catalog_invalid"}:
        assert result["catalog_status"] == "unavailable"
    assert len(calls) == 1
    _discover_catalog(session, experiments, candidate, "discover", {"call_id": "discover"})
    assert len(calls) == 1


def test_old_revision_failure_does_not_invalidate_current_candidate(state):
    from src.mod_agent.task_agent_v2_experiment_tools import _invalidate_check

    edit(state, "A.cs", "1", "2")
    session = SimpleNamespace(state=state, current_check="current-passed-check")
    _invalidate_check(session, "baseline", {"executed": True, "status": "failed", "result_ref": "old-observation"})
    assert session.current_check == "current-passed-check"


def test_normal_experiment_factory_connects_optional_sdk_runner_without_executing_it(state, monkeypatch):
    from src.mod_agent import task_agent_v2_experiment_tools as tools
    from src.mod_agent.domain.repository_probe_runner import RepositoryProbeRunner

    config = {"kind": "docker_sdk_project_v1", "image": "mcr.microsoft.com/dotnet/sdk@sha256:" + "a" * 64,
        "sdk_version": "8.0.100", "target_framework": "net8.0", "project_path": "Library.csproj"}
    backend = SimpleNamespace(profile=SimpleNamespace(data={"probe_runner": config}))
    monkeypatch.setattr(tools, "read_bound_json", lambda _ref: {"behavior_evaluator": {"kwargs": {}}})
    monkeypatch.setattr(tools, "CSharpRepositoryBackend", lambda **_kwargs: backend)
    monkeypatch.setattr(RepositoryProbeRunner, "_environment", lambda *_args: pytest.fail("Constructing tools must not execute qualification"))
    session = SimpleNamespace(state=state, owner=SimpleNamespace(worker_configuration_ref="controlled"))
    experiments = tools._experiments(session)
    assert isinstance(experiments.probe_runner, RepositoryProbeRunner)
    assert experiments.probe_runner.backend is backend
    assert tools._experiments(session) is experiments
