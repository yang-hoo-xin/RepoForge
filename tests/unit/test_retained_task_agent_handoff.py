"""Native recovery through real Proposal/Review/DryRun; no compiler or provider."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent import retained_task_agent_handoff as recovery
from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256
from tests.unit import test_task_agent_delivery_binding as delivery_fixture
from tests.unit import test_task_agent_execution as execution_fixture


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return recovery._file(path)[0]


@pytest.fixture
def retained(short_workspace, monkeypatch):
    execute = execution_fixture.execute

    def owned(case):
        case.state["run_id"] = case.inputs.workspace.name
        case.kwargs["output_root"] = case.inputs.store.root / "task-agent-execution"
        return execute(case)

    monkeypatch.setattr(execution_fixture, "execute", owned)
    case, values = delivery_fixture.delivery.__wrapped__(short_workspace, monkeypatch,
        SimpleNamespace(param="baseline"))
    root = values["root"]
    ready_ref = values["initial"]["task_agent_execution_ref"]
    ready = recovery._checked(ready_ref)
    error = deepcopy(ready)
    error["status"] = "failed"
    error["termination"] = {"code": "TASK_AGENT_EXECUTION_FAILED", "error_type": "ValueError",
        "reason": "TASK_AGENT_EXECUTION_REFERENCE_OUTSIDE_STORE",
        "prior_loop_termination": {"code": "TASK_AGENT_COMPLETION_ACCEPTED"}}
    error_ref = save(root / "a/task-agent-execution/execution-error.json", error)
    old = deepcopy(values["initial"]["task_agent_outcome"])
    old.update(status="failed", termination=error["termination"], execution_ref=error_ref)
    target = root / "generated_mods/SyntheticMod"
    initial = {"run_id": root.name, "current_stage": "blocked", "actual_write_performed": False,
        "canonical_source_unchanged": None, "patch_execution_receipts": [], "task_agent_outcome": old,
        "task_agent_execution_ref": error_ref, "model_call_budget": old["budget_after"],
        "model_call_count": old["budget_after"]["consumed"], "technical_execution": values["manifest"]["technical_execution"],
        "initial_isolated_source_snapshot": source_snapshot(target), "final_isolated_source_snapshot": None,
        "completion_classification": {}}
    original_ref = save(root / "run_result.json", initial)
    from src.mod_agent.model_admission import local_model_admission_record

    manifest = {**values["manifest"], "run_id": root.name, "status": "blocked", "terminal_record": original_ref,
        "local_model_identity": ready["expected_model_identity"], "model_call_count": initial["model_call_count"],
        "target_source": str(target), "source_snapshot_sha256": source_snapshot_sha256(source_snapshot(target)),
        "request": case.inputs.store.read(case.context.feature_spec_ref)["data"]["user_request"],
        "source_revision_binding": {}, "source_disclosure": {"scope": "local_controlled_test"},
        "model_admission": local_model_admission_record(model_identity=ready["expected_model_identity"],
            provider_adapter="scripted_test"), "runtime_admission": {"requested": False}}
    save(root / "run_manifest.json", manifest)
    config = root.parent / ".dsh-inputs" / root.name / "execution.json"
    config_ref = save(config, {"test_only": "authority independently tested below"})
    monkeypatch.setattr(recovery, "_retained_authority", lambda *_: config_ref)
    return SimpleNamespace(root=root, config=config, case=case, original=initial, manifest=manifest,
        ready=ready, ready_ref=ready_ref, kwargs={"execution_config": config, "run_root": root,
            "run_id": root.name, "ready_execution_ref": ready_ref, "attempt_id": "recover-original"})


def test_normal_graph_restores_exact_approval_without_model_build_or_source_write(retained, monkeypatch):
    from src.mod_agent.task_agent_execution import TaskAgentExecution

    monkeypatch.setattr(TaskAgentExecution, "__call__", lambda *_: pytest.fail("No new model execution"))
    before = source_snapshot(retained.root / "generated_mods/SyntheticMod")
    original_refs = [recovery._file(retained.root / name)[0] for name in ("run_manifest.json", "run_result.json")]
    result = recovery.resume_retained_task_agent_handoff(**retained.kwargs)
    assert result["status"] == "awaiting_approval", result
    assert result["run_id"] == retained.root.name
    assert result["model_call_budget"] == retained.original["model_call_budget"]
    assert result["patch_execution_receipts"] == []
    assert result["new_provider_sends"] == result["new_builds"] == 0
    assert result["approval_recorded"] is False
    assert result["actual_write_performed"] is False
    assert source_snapshot(retained.root / "generated_mods/SyntheticMod") == before
    assert [recovery._file(retained.root / name)[0] for name in ("run_manifest.json", "run_result.json")] == original_refs
    assert recovery.resume_retained_task_agent_handoff(**retained.kwargs) == result
    with pytest.raises(ValueError, match="OPERATION_CONFLICT"):
        recovery.resume_retained_task_agent_handoff(**{**retained.kwargs, "attempt_id": "another"})


@pytest.mark.parametrize("change", ["unknown", "non_finish", "candidate", "budget", "source", "worker", "apply"])
def test_changed_origin_rejected_before_continuation_write(retained, change):
    root = retained.root
    original = deepcopy(retained.original)
    if change == "unknown":
        original["task_agent_outcome"]["unknown_effects"] = True
    elif change == "non_finish":
        original["task_agent_outcome"]["termination"]["reason"] = "another failure"
    elif change == "candidate":
        original["task_agent_outcome"]["candidate_ref"]["sha256"] = "f" * 64
    elif change == "budget":
        original["model_call_count"] = 0
    elif change == "source":
        retained.case.inputs.behavior.write_text("changed", encoding="utf-8")
    elif change == "worker":
        Path(retained.ready["worker_binding"]["configuration_ref"]["path"]).write_text("{}", encoding="utf-8")
    else:
        original["actual_write_performed"] = True
    reference = save(root / "run_result.json", original)
    save(root / "run_manifest.json", {**retained.manifest, "terminal_record": reference})
    with pytest.raises((ValueError, KeyError)):
        recovery.resume_retained_task_agent_handoff(**retained.kwargs)
    assert not (root / "continuations").exists()


def test_incomplete_claim_never_reexecutes_graph(retained, monkeypatch):
    from src.mod_agent.workflow import graph

    monkeypatch.setattr(graph, "build_retained_task_agent_handoff_graph",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")))
    with pytest.raises(RuntimeError, match="interrupted"):
        recovery.resume_retained_task_agent_handoff(**retained.kwargs)
    result = recovery.resume_retained_task_agent_handoff(**retained.kwargs)
    assert result["reason"] == "RETAINED_TASK_AGENT_INCOMPLETE_REVIEW_REQUIRED"


@pytest.mark.parametrize("status,exit_code", [("awaiting_approval", 0), ("delivered", 0), ("blocked", 2)])
def test_cli_routes_only_to_explicit_recovery(monkeypatch, capsys, status, exit_code):
    from src.mod_agent import cli, local_entry

    calls = []
    def recover(**kwargs):
        calls.append(kwargs)
        return {"status": status, "new_provider_sends": 0}
    monkeypatch.setattr(local_entry, "resume_retained_task_agent_handoff", recover)
    assert cli.main(["resume-retained-task-agent-handoff", "--execution-config", "config",
        "--run-root", "root", "--run-id", "run", "--execution-path", "execution",
        "--execution-sha256", "a" * 64, "--attempt-id", "resume-1"]) == exit_code
    assert calls == [{"execution_config": "config", "run_root": "root", "run_id": "run",
        "ready_execution_ref": {"path": str(Path("execution").resolve()), "sha256": "a" * 64}, "attempt_id": "resume-1"}]
    assert json.loads(capsys.readouterr().out)["new_provider_sends"] == 0


def test_cli_normalizes_forward_slash_execution_path_without_changing_hash(tmp_path, monkeypatch, capsys):
    from src.mod_agent import cli, local_entry

    execution = tmp_path / "run/a/task-agent-execution/execution.json"
    reference = save(execution, {"candidate": "retained"})
    calls = []
    def recover(**kwargs):
        calls.append(kwargs)
        assert kwargs["ready_execution_ref"] == reference
        return {"status": "awaiting_approval", "new_provider_sends": 0}
    monkeypatch.setattr(local_entry, "resume_retained_task_agent_handoff", recover)
    assert cli.main(["resume-retained-task-agent-handoff", "--execution-config", "config",
        "--run-root", "root", "--run-id", "run", "--execution-path", execution.as_posix(),
        "--execution-sha256", reference["sha256"], "--attempt-id", "resume-1"]) == 0
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["new_provider_sends"] == 0


def test_started_delivery_without_completed_continuation_cannot_restart(retained):
    (retained.root / "delivery").mkdir()
    with pytest.raises(ValueError, match="DELIVERY_ALREADY_STARTED"):
        recovery.resume_retained_task_agent_handoff(**retained.kwargs)
    assert not (retained.root / "continuations").exists()


@pytest.mark.parametrize("change", [None, "model_budget", "worker", "scope", "approval", "identity", "outside"])
def test_allocated_authority_checks_exact_original_settings(tmp_path, monkeypatch, change):
    from scripts import dsh_run_request

    root = tmp_path / "runs/task"
    root.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    worker = save(root.parent / ".dsh-inputs/task/worker.json", {"test": 1})
    config = root.parent / ".dsh-inputs/task/execution.json"
    save(config, {"arguments": []})
    selection = {"technical_executor": "task_agent", "constraint_policy": "source_priority_v1",
        "candidate_mode": "current", "allowed_edit_paths_relative_to_project": ["Plugin.cs"]}
    manifest = {"run_id": root.name, "request_id": "request", "request": "fix", "target_source": str(source),
        "technical_execution": selection, "runtime_admission": {"requested": False}}
    original = {"model_call_budget": {"maximum": 18, "policy": "task_total_only"}}
    parsed = SimpleNamespace(run_id="task", request_id="request", workspace_root=str(root.parent),
        target_project=str(source), confirm_isolated_write=True, allow_build=True, defer_approval=True,
        task_model_call_limit=18, model_call_budget_policy="task_total_only", patch_worker_config=worker["path"],
        technical_executor="task_agent", constraint_policy="source_priority_v1", task_agent_candidate_mode="current",
        task_agent_edit_paths='["Plugin.cs"]')
    if change == "model_budget":
        parsed.task_model_call_limit = 19
    elif change == "worker":
        parsed.patch_worker_config = save(tmp_path / "other-worker.json", {"test": 1})["path"]
    elif change == "scope":
        parsed.task_agent_edit_paths = '["Other.cs"]'
    elif change == "approval":
        parsed.defer_approval = False
    elif change == "identity":
        parsed.request_id = "other"
    elif change == "outside":
        config = tmp_path / "elsewhere.json"
        save(config, {"arguments": []})
    monkeypatch.setattr(dsh_run_request, "validate_product_profile", lambda **_: (None, parsed))
    args = (config, root, manifest, original, {"worker_binding": {"configuration_ref": worker}})
    if change is None:
        assert recovery._retained_authority(*args) == recovery._file(config)[0]
    else:
        with pytest.raises(ValueError):
            recovery._retained_authority(*args)
