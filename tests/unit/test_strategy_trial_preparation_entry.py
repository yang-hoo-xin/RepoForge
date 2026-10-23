"""Prepared B/B+ configurations reach the actual normal HTTP request path.

为什么要有它：准备清单不等于经验送达，受控响应只验证接线而不计自主成绩。
"""
import json
from pathlib import Path

from scripts import dsh_run_request, prepare_memory_trials
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.strategy_native_export import (
    export_strategy_execution,
    validate_native_strategy_execution,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from tests.unit import test_task_agent_v2_experience as entry
from tests.unit.test_csharp_repository_product_entry import (
    repository_workspace as repository_workspace,
)


def test_prepared_pair_consumed_by_normal_cli_and_exported_without_condition_drift(repository_workspace, monkeypatch, capsys):
    root = repository_workspace
    original_configuration, original_cli = entry._configuration, entry.cli.main
    saved = {}
    description = "Double the integer input, preserving the public signature."

    def configuration(workspace, *, consume, extract, trial=False):
        if not saved:
            memory, old_ref = original_configuration(workspace, consume=consume, extract=extract, trial=True)
            old = read_bound_json(old_ref)
            protocol = read_bound_json(read_bound_json(old["trial_binding_ref"])["protocol_ref"])
            source = workspace / "source"
            task = _save(workspace / "prepared-source.json", {"task_id": "double", "task_family": "controlled-arithmetic",
                "source_version": "controlled-v1", "source_files_sha256": canonical_hash({
                    p.relative_to(source).as_posix(): _ref(p)["sha256"] for p in source.rglob("*") if p.is_file()}),
                "original_request_sha256": canonical_hash(description)})
            exposure = _save(workspace / "prepared-exposure.json", {"task_id": "double", "source_ref": task,
                "attempt": 1, "exposure": "exposed"})
            protocol["planned_pairs"][0].update(source_ref=task, exposure_ref=exposure)
            protocol_ref = _save(workspace / "prepared-protocol.json", protocol)
            request = _save(workspace / "public-request.json", {"schema_version": "public_development_request_v1",
                "visibility": "public", "description": description})
            config = _save(workspace / "base-execution.json", {"arguments": [
                "--target-project", str(source), "--repository-profile", str(workspace / "repository-profile.json"),
                "--workspace-root", str(workspace / "original-runs"), "--settings", str(Path("config/settings.yaml").resolve()),
                "--local-ollama", "--ollama-model", "qwen3:4b-instruct", "--ollama-context-window", "65536",
                "--model-call-budget-policy", "meter_only", "--constraint-policy", "source_priority_v1",
                "--technical-executor", "task_agent", "--task-agent-protocol", "v2",
                "--defer-approval", "--confirm-isolated-write", "--allow-build"]})
            spec = _save(workspace / "preparation-input.json", {"schema_version": "strategy_trial_preparation_input_v1",
                "protocol_ref": protocol_ref, "store_root": str(memory.store.root), "extract_on_terminal": False,
                "tasks": [{"task_id": "double", "attempt": 1, "execution_config_ref": config, "request_ref": request}],
                "order": [{"task_id": "double", "attempt": 1, "arm": arm} for arm in ("B", "B+")]})
            assert prepare_memory_trials.main(["prepare-v2", "--input-record", spec["path"],
                "--output-root", str(workspace / "prepared")]) == 0
            prepared_ref = json.loads(capsys.readouterr().out)
            saved.update(memory=memory, prepared_ref=prepared_ref, prepared=read_bound_json(prepared_ref))
        row = next(row for row in saved["prepared"]["trials"] if row["arm"] == ("B+" if consume else "B"))
        saved["current"] = row
        return saved["memory"], row["experience_config_ref"]

    def run_prepared(_argv):
        row = saved["current"]
        return original_cli(dsh_run_request.build_product_argv(execution_config=row["execution_config_ref"]["path"],
            description=description))

    exports = []
    for mode in ("trial_off", "trial_consume"):
        with monkeypatch.context() as patch:
            patch.setattr(entry, "_configuration", configuration)
            patch.setattr(entry.cli, "main", run_prepared)
            entry.test_experience_through_normal_cli_actual_requests_and_terminal(root, patch, capsys, mode)
        row = saved["current"]
        path = Path(row["run_root"]) / "a/task-agent-execution/execution.json"
        native = read_bound_json(_ref(path))
        call = next(c for c in reversed(native["completed_tool_records"]) if c["name"] == "candidate_submit")
        ref = export_strategy_execution(native_execution_ref=_ref(path), call_id=call["call_id"],
            output_root=root / ("export-" + row["arm"].replace("+", "plus")))
        checked = validate_native_strategy_execution(ref)
        assert checked["freeze_binding"]["status"] == "protocol_bound_before_model"
        assert checked["planned_source_binding"] == "source_and_request_verified"
        exports.append(checked)
    assert exports[0]["actual_common_conditions_sha256"] == exports[1]["actual_common_conditions_sha256"]
    assert prepare_memory_trials.main(["verify-v2", "--prepared-record", saved["prepared_ref"]["path"]]) == 0
