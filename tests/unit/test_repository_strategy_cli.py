"""Normal offline CLI consumes native backend reports; transport is controlled."""
import json
import os
import shutil

import pytest

from src.mod_agent import cli
from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.strategy_adoption import _evaluator_evidence
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.task_agent_execution import _worker_binding
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)
from tests.unit.test_strategy_repository_evaluator import evidence as evidence
from tests.unit.test_task_agent_v2_state import edit


def test_normal_qualification_and_verdict_route_to_adoption_validator(evidence, tmp_path, capsys):
    kwargs, source = evidence
    document = {key: value for key, value in kwargs.items() if key != "output_root"}
    document["schema_version"] = "repository_strategy_qualification_input_v1"
    input_ref = _save(tmp_path / "input.json", document)
    assert cli.main(["qualify-repository-evaluator", "--input-record", input_ref["path"],
        "--output-root", str(tmp_path / "qualified")]) == 0
    evaluator_ref = json.loads(capsys.readouterr().out)["evaluator_ref"]
    store = tmp_path / "store"
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": ["Library/Arithmetic.cs"], "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "task", source, {}, ["Library/Arithmetic.cs"], authority)
    candidate = edit(state, "Library/Arithmetic.cs", "=> value;", "=> value * 2;")
    profile = backend.CSharpRepositoryProfile(kwargs["profile_ref"])
    worker_ref = _save(tmp_path / "worker.json", {"behavior_evaluator": {
        "factory_ref": _ref(backend.__file__), "factory_name": "make_csharp_repository_evaluator",
        "kwargs": {"profile_ref": kwargs["profile_ref"], "expected_fixed_inputs": profile.bind(source)}}})
    native_ref = _save(tmp_path / "native.json", {"worker_binding": _worker_binding(worker_ref)})
    execution = _save(tmp_path / "execution.json", {"task_id": "task", "candidate_source_id": candidate["candidate_source_id"],
        "native_execution_ref": native_ref,
        "candidate_state": {"artifact_store_root": str(store), "task_contract_ref": state.contract_ref,
            "candidate_ref": candidate["candidate_ref"]}})
    assert cli.main(["evaluate-repository-strategy", "--evaluator-record", evaluator_ref["path"],
        "--execution-record", execution["path"], "--evidence-record", kwargs["controls"]["correct"]["path"],
        "--output", str(tmp_path / "verdict.json")]) == 0
    result = json.loads(capsys.readouterr().out)
    verdict = read_bound_json(result["verdict_ref"])
    assert _evaluator_evidence(verdict)["status"] == "verified_repository_csharp"
    assert verdict["task_success"] is True and verdict["preservation_passed"] is True
    assert "checks" not in verdict and "observations" not in verdict
    missing = {key: value for key, value in verdict.items() if key != "evaluator_result_ref"}
    assert _evaluator_evidence(missing)["status"] == "missing_original_evaluator_evidence"
    assert cli.main(["evaluate-repository-strategy", "--evaluator-record", evaluator_ref["path"],
        "--execution-record", execution["path"], "--evidence-record", kwargs["controls"]["wrong"]["path"],
        "--output", str(tmp_path / "mismatched.json")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not (tmp_path / "mismatched.json").exists()


def test_qualification_cli_rejects_summary_only_input(tmp_path, capsys):
    reference = _save(tmp_path / "summary.json", {"schema_version": "repository_strategy_qualification_input_v1", "passed": True})
    assert cli.main(["qualify-repository-evaluator", "--input-record", reference["path"],
        "--output-root", str(tmp_path / "output")]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "REPOSITORY_QUALIFICATION_INPUT_INVALID"
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(os.environ.get("REPOFORGE_REAL_REPOSITORY_CLI_INTEGRATION") != "1", reason="Explicit real SDK integration")
def test_real_sdk_control_reports_qualify_through_normal_cli(repository_workspace, capsys):
    from tests.unit.test_csharp_repository_backend import _backend, _profile

    root = repository_workspace
    profile, source = _profile(root, real=True)
    (source / "tests/Program.cs").write_text('''
using System.IO;
using System.Text.Json;
var actual = Arithmetic.Double(3);
var zero = Arithmetic.Double(0);
File.WriteAllText(args[0], JsonSerializer.Serialize(new {
    schema_version = "csharp-test-report.v1",
    tests = new[] {
        new { name = "target", passed = actual == 6, expected = 6, actual = actual },
        new { name = "preserve", passed = zero == 0, expected = 0, actual = zero }
    }
}));
''', encoding="utf-8")
    controls = {}
    for role, expression in (("correct", "value * 2"), ("equivalent", "value + value"), ("wrong", "value")):
        copied = root / role
        shutil.copytree(source, copied)
        (copied / "Library/Arithmetic.cs").write_text(
            "public static class Arithmetic { public static int Double(int value) => " + expression + "; }", encoding="utf-8")
        controls[role] = _backend(root, profile, copied).run_build_test(copied, "sdk-qualification", "isolated")["report_ref"]
    request = _save(root / "qualification-input.json", {"schema_version": "repository_strategy_qualification_input_v1",
        "profile_ref": profile.reference, "controls": controls,
        "production_paths": ["Library/Arithmetic.cs", "Library/Library.csproj", "nuget.config"],
        "target_test_ids": ["target"], "preservation_test_ids": ["preserve"]})
    assert cli.main(["qualify-repository-evaluator", "--input-record", request["path"],
        "--output-root", str(root / "qualification")]) == 0
    output = json.loads(capsys.readouterr().out)
    _save(root / "real-sdk-qualification-result.json", {"status": output["status"],
        "evaluator_ref": output["evaluator_ref"], "controls": controls, "model_calls": 0,
        "scope": "real_sdk_control_qualification_not_autonomous_repair"})
    assert read_bound_json(output["evaluator_ref"])["execution_scope"] == "host_workspace_execution_no_os_isolation"
