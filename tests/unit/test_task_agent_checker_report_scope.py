"""Real run/v report layout with a scripted model and Python-only verifier."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_execution import _read_checker_report
from tests.unit.test_task_agent_execution import case as case
from tests.unit.test_task_agent_execution import execute_current_revision, file_ref


def bind_checker_output(case):
    workspace, store = case.inputs.workspace, case.inputs.store.root
    output = workspace / "v" / "bounded-checker"
    output.mkdir(parents=True)
    kwargs = {"output_root": str(output), "compilation_kwargs": {
        "workspace_root": str(workspace), "artifact_store_root": str(store)}}
    worker_path = Path(case.kwargs["worker_configuration_ref"]["path"])
    worker = json.loads(worker_path.read_bytes())
    worker["behavior_evaluator"]["kwargs"] = kwargs
    worker_path.write_text(json.dumps(worker), encoding="utf-8")
    worker_ref = file_ref(worker_path)
    case.kwargs["worker_configuration_ref"] = worker_ref
    binding = {"configuration_ref": worker_ref, "factory_ref": worker["behavior_evaluator"]["factory_ref"],
        "factory_name": worker["behavior_evaluator"]["factory_name"], "kwargs_sha256": canonical_hash(kwargs)}
    case.evaluator.configuration_receipt = lambda: deepcopy(binding)
    case.evaluator.root = output
    return output, worker_path


def test_current_revision_handoff_accepts_exact_configured_run_verifier_output(case):
    output, _ = bind_checker_output(case)
    owner, outcome, _ = execute_current_revision(case)
    assert outcome.status == "candidate_ready", outcome.termination
    verified = owner.validate_handoff(outcome, case.context)
    report_path = Path(verified.check["report"]["report_ref"]["path"])
    assert report_path.is_relative_to(output)
    assert not report_path.is_relative_to(case.inputs.store.root)
    assert len(case.requests) == 6 and len(case.evaluator.calls) == 2
    assert not outcome.whole_task_delivered


@pytest.mark.parametrize("damage", ["other_run", "undeclared_output", "broad_root", "wrong_workspace", "wrong_store", "changed_sha", "wrong_worker_sha"])
def test_checker_report_reader_rejects_unbound_paths_and_changed_content(case, damage):
    output, worker_path = bind_checker_output(case)
    report = output / "report.json"
    report.write_text('{"status":"passed"}', encoding="utf-8")
    reference = file_ref(report)
    worker = json.loads(worker_path.read_bytes())
    kwargs = worker["behavior_evaluator"]["kwargs"]
    worker_ref = case.kwargs["worker_configuration_ref"]
    if damage == "other_run":
        other = case.inputs.workspace.parent / "another-run" / "v"
        other.mkdir(parents=True)
        report = other / "report.json"
        report.write_text('{"status":"passed"}', encoding="utf-8")
        reference = file_ref(report)
    elif damage == "undeclared_output":
        kwargs.pop("output_root")
    elif damage == "broad_root":
        kwargs["output_root"] = str(case.inputs.workspace)
    elif damage == "wrong_workspace":
        kwargs["compilation_kwargs"]["workspace_root"] = str(case.inputs.workspace.parent)
    elif damage == "wrong_store":
        kwargs["compilation_kwargs"]["artifact_store_root"] = str(case.inputs.workspace / "other-store")
    elif damage == "changed_sha":
        report.write_text('{"status":"changed"}', encoding="utf-8")
    else:
        kwargs["output_root"] = str(case.inputs.workspace)
    if damage in {"undeclared_output", "broad_root", "wrong_workspace", "wrong_store", "wrong_worker_sha"}:
        worker_path.write_text(json.dumps(worker), encoding="utf-8")
        if damage != "wrong_worker_sha":
            worker_ref = file_ref(worker_path)
    with pytest.raises((ValueError, OSError)):
        _read_checker_report(reference, store_root=case.inputs.store.root,
            workspace_root=case.inputs.workspace, worker_configuration_ref=worker_ref)
