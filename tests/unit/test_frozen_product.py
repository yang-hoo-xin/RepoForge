"""Fresh-process provenance checks with a controlled CLI, no model or Build."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.mod_agent.platform_execution_snapshot import prepare_snapshot

REPOSITORY = Path(__file__).resolve().parents[2]
CLI = '''import argparse
import hashlib
import json
from pathlib import Path
from src.mod_agent.patch_execution_trace import product_execution_trace, product_trace_identity, preserve_product_trace_projection

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run-request"])
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--request-id", default="request-one")
    parser.add_argument("--target-project")
    parser.add_argument("--load-file", action="append", default=[])
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--exit-code", type=int, default=7)
    return parser

def main(argv):
    args = build_parser().parse_args(argv)
    root = Path(args.workspace_root) / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "run_manifest.json"
    identity = {"run_id":args.run_id, "request_id":args.request_id, "status":"controlled"}
    manifest.write_text(json.dumps(identity), encoding="utf-8")
    with product_execution_trace(run_root=root, run_id=args.run_id, request_id=args.request_id, artifact_store_root=root / "a") as trace:
        (root / "initial_seen.json").write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
        for filename in args.load_file:
            raw = Path(filename).read_bytes()
            exec(compile(raw, filename, "exec", dont_inherit=True), {})
        if args.drift:
            target = Path(__file__).resolve().parents[2] / "config/required.json"
            target.write_text('{"changed":true}', encoding="utf-8")
        result = {**identity, "arguments":argv, "execution_trace":product_trace_identity(trace, root)}
        raw = json.dumps(result).encode("utf-8")
        (root / "run_result.json").write_bytes(raw)
        refreshed = {**identity, "status":"blocked", "terminal_stage":"blocked", "terminal_record":{"path":str(root / "run_result.json"), "sha256":hashlib.sha256(raw).hexdigest()}}
        manifest.write_text(json.dumps(preserve_product_trace_projection(refreshed, root)), encoding="utf-8")
    print(json.dumps({"controlled":True,"argv":argv}))
    return args.exit_code
'''


def _source(tmp_path, *, preamble=""):
    root = tmp_path / "source"
    for relative in ("src/__init__.py", "src/mod_agent/__init__.py", "src/core/__init__.py",
                     "src/core/trace/__init__.py", "scripts/__init__.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    for relative in ("src/mod_agent/platform_execution_snapshot.py", "src/mod_agent/patch_execution_trace.py",
                     "src/core/trace/execution_trace.py", "src/core/trace/trace_context.py",
                     "scripts/run_frozen_product.py"):
        shutil.copyfile(REPOSITORY / relative, root / relative)
    (root / "src/mod_agent/cli.py").write_text(preamble + CLI, encoding="utf-8")
    resource = root / "config/required.json"
    resource.parent.mkdir()
    resource.write_text('{"fixture":true}', encoding="utf-8")
    cache = root / "src/__pycache__/old.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"not loaded")
    return root


def _prepare(tmp_path, *, argv=None, selected=None, source=None, extras=(), inputs=()):
    source = source or _source(tmp_path)
    selected = selected or tmp_path / "runs/run-one"
    argv = argv or ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name]
    reference = prepare_snapshot(source_root=source, snapshot_root=tmp_path / "frozen",
        business_argv=argv, run_root=selected, extra_load_refs=extras, bound_inputs=inputs)
    return reference, selected, argv


def _child(reference, *, environment=None):
    root = Path(reference["path"]).parent
    return subprocess.run([sys.executable, "-I", "-B", str(root / "scripts/run_frozen_product.py"),
        "--_child", reference["path"], reference["sha256"]], capture_output=True,
        text=True, encoding="utf-8", timeout=30, check=False, env=environment)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_new_process_preserves_arguments_exit_and_initial_terminal_version_binding(tmp_path):
    reference, selected, argv = _prepare(tmp_path)
    result = _child(reference, environment={**os.environ, "PYTHONPATH": str(tmp_path / "source")})
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["argv"] == argv
    snapshot = _read(Path(reference["path"]))
    assert "config/required.json" in snapshot["files"]
    assert not list((tmp_path / "frozen").rglob("*.pyc"))
    with zipfile.ZipFile(snapshot["archive_ref"]["path"]) as archive:
        assert archive.read("config/required.json") == b'{"fixture":true}'
    initial = _read(selected / "initial_seen.json")
    terminal = _read(selected / "run_result.json")
    final = _read(selected / "run_manifest.json")
    assert initial["execution_version_ref"] == reference == final["execution_version_ref"]
    assert terminal["execution_trace"]["execution_version_ref"] == reference
    assert final["execution_version_status"] == "verified"
    receipt = _read(Path(final["execution_version_check_ref"]["path"]))
    assert receipt["phase"] == "finished" and receipt["process_id"] != os.getpid()
    assert receipt["loaded_project_modules"] and receipt["executed_source_files"]
    assert all(Path(row["path"]).is_relative_to(tmp_path / "frozen") for row in receipt["executed_source_files"])


def test_help_runs_original_parser_without_creating_product_run(tmp_path):
    reference, selected, _ = _prepare(tmp_path, argv=["--help"])
    result = _child(reference)
    assert result.returncode == 0 and "usage:" in result.stdout
    assert not selected.exists()
    assert _read(tmp_path / "frozen/child_execution.json")["status"] == "verified"


def test_snapshot_file_drift_is_rejected_before_cli_execution(tmp_path):
    reference, selected, _ = _prepare(tmp_path)
    (tmp_path / "frozen/config/required.json").write_text("changed", encoding="utf-8")
    result = _child(reference)
    assert result.returncode == 2 and "SOURCE_CHANGED" in result.stderr
    assert not selected.exists()


def test_drift_during_business_retains_exit_code_but_marks_version_unknown(tmp_path):
    selected = tmp_path / "runs/run-one"
    argv = ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name, "--drift"]
    reference, selected, _ = _prepare(tmp_path, argv=argv)
    result = _child(reference)
    assert result.returncode == 7, result.stderr
    final = _read(selected / "run_manifest.json")
    assert final["execution_version_ref"] == reference
    assert final["execution_version_status"] == "unknown"
    assert _read(tmp_path / "frozen/child_execution.json")["status"] == "unknown"


def test_unselected_existing_run_is_not_touched(tmp_path):
    selected, other = tmp_path / "runs/selected", tmp_path / "runs/old-run"
    other.mkdir(parents=True)
    (other / "run_manifest.json").write_bytes(b"immutable-history")
    argv = ["run-request", "--workspace-root", str(other.parent), "--run-id", other.name]
    reference, _, _ = _prepare(tmp_path, argv=argv, selected=selected)
    result = _child(reference)
    assert result.returncode == 2 and "UNSELECTED_RUN_ROOT" in result.stderr
    assert (other / "run_manifest.json").read_bytes() == b"immutable-history"
    assert not selected.exists()


def test_execute_original_tree_module_is_rejected_not_called_frozen(tmp_path):
    source = _source(tmp_path)
    outside = source / "src/outside.py"
    marker = tmp_path / "outside-ran"
    outside.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n", encoding="utf-8")
    preamble = "exec(compile(open(" + repr(str(outside)) + ", 'rb').read(), " + repr(str(outside)) + ", 'exec'), {})\n"
    (source / "src/mod_agent/cli.py").write_text(preamble + CLI, encoding="utf-8")
    reference, selected, _ = _prepare(tmp_path, source=source)
    result = _child(reference)
    assert result.returncode == 2 and "LOADED_OUTSIDE_SNAPSHOT" in result.stderr
    assert not marker.exists() and not selected.exists()
    assert _read(tmp_path / "frozen/child_execution.json")["status"] == "unknown"


def test_already_loaded_outside_module_fails_before_business(tmp_path):
    outside = tmp_path / "other.py"
    outside.write_text("VALUE = 1", encoding="utf-8")
    preamble = ("import sys, types\nmodule=types.ModuleType('src.foreign')\nmodule.__file__="
                + repr(str(outside)) + "\nsys.modules['src.foreign']=module\n")
    source = _source(tmp_path, preamble=preamble)
    reference, selected, _ = _prepare(tmp_path, source=source)
    result = _child(reference)
    assert result.returncode == 2 and "LOADED_OUTSIDE_SNAPSHOT" in result.stderr
    assert not selected.exists()


def test_platform_snapshot_cannot_be_selected_as_model_target_source(tmp_path):
    selected = tmp_path / "runs/run-one"
    argv = ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name,
            "--target-project", str(tmp_path / "frozen")]
    reference, selected, _ = _prepare(tmp_path, argv=argv)
    result = _child(reference)
    assert result.returncode == 2 and "PLATFORM_IS_NOT_TASK_SOURCE" in result.stderr
    assert not selected.exists()


def test_two_explicit_dynamic_factories_are_retained_and_observed_without_module_registry(tmp_path):
    source = _source(tmp_path)
    external = source / "tests/factories"
    external.mkdir(parents=True)
    paths = [external / "adapter.py", external / "oracle.py"]
    for index, path in enumerate(paths):
        path.write_text(f"FACTORY = {index}\n", encoding="utf-8")
    refs = [(str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    selected = tmp_path / "runs/run-one"
    argv = ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name]
    for path in paths:
        argv.extend(["--load-file", str(path)])
    reference, _, _ = _prepare(tmp_path, source=source, argv=argv, extras=refs)
    result = _child(reference)
    assert result.returncode == 7, result.stderr
    receipt = _read(tmp_path / "frozen/child_execution.json")
    assert all(row["execution_observed"] for row in receipt["external_bindings"])
    assert all(Path(row["retained_path"]).read_bytes() == Path(row["path"]).read_bytes()
               for row in receipt["external_bindings"])


def test_external_factory_drift_rejected_before_product(tmp_path):
    source = _source(tmp_path)
    external = tmp_path / "factory.py"
    external.write_text("VALUE = 1\n", encoding="utf-8")
    reference, selected, _ = _prepare(tmp_path, source=source,
        extras=[(str(external), hashlib.sha256(external.read_bytes()).hexdigest())])
    external.write_text("VALUE = 2\n", encoding="utf-8")
    result = _child(reference)
    assert result.returncode == 2 and "EXTERNAL_REFERENCE_CHANGED" in result.stderr
    assert not selected.exists()


def test_existing_selected_run_rejected_before_snapshot_write(tmp_path):
    source = _source(tmp_path)
    existing = tmp_path / "old-run"
    existing.mkdir()
    with pytest.raises(ValueError, match="SELECTED_RUN_ALREADY_EXISTS"):
        prepare_snapshot(source_root=source, snapshot_root=tmp_path / "frozen",
            business_argv=["--help"], run_root=existing)
    assert not (tmp_path / "frozen").exists()


def _configuration_source(tmp_path):
    source = _source(tmp_path)
    shutil.copyfile(REPOSITORY / "src/core/settings.py", source / "src/core/settings.py")
    shutil.copyfile(REPOSITORY / "config/settings.yaml", source / "config/settings.yaml")
    (source / "src/mod_agent/cli.py").write_text('''import argparse
import json
from pathlib import Path
from src.core.settings import load_settings

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--workspace-root")
    parser.add_argument("--run-id")
    parser.add_argument("--settings", default="config/settings.yaml")
    for flag in ("--patch-worker-config", "--stage-model-call-limits", "--game-folder-build-binding", "--external-patch-profile", "--repository-profile", "--task-agent-experience-config"):
        parser.add_argument(flag)
    return parser

def main(argv):
    args = build_parser().parse_args(argv)
    observed = {"model":load_settings(args.settings).llm.model}
    for field in ("patch_worker_config", "stage_model_call_limits", "game_folder_build_binding", "external_patch_profile", "repository_profile", "task_agent_experience_config"):
        value = getattr(args, field)
        if value:
            observed[field] = json.loads(Path(value).read_text(encoding="utf-8"))
    print(json.dumps(observed))
    return 7
''', encoding="utf-8")
    return source


@pytest.mark.parametrize("field", ["settings", "patch-worker-config", "stage-model-call-limits",
                                    "game-folder-build-binding", "external-patch-profile", "repository-profile", "task-agent-experience-config"])
def test_unbound_external_configuration_rejected_before_actual_consumption(tmp_path, field):
    import yaml

    source = _configuration_source(tmp_path)
    external = source / "config/settings.yaml" if field == "settings" else tmp_path / "outside.json"
    if field != "settings":
        external.write_text('{"original":true}', encoding="utf-8")
    selected = tmp_path / "runs/run-one"
    argv = ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name,
            "--" + field, str(external)]
    reference, _, _ = _prepare(tmp_path, source=source, argv=argv)
    if field == "settings":
        document = yaml.safe_load(external.read_text(encoding="utf-8"))
        document["llm"]["model"] = "unbound-live-mutated"
        external.write_text(yaml.safe_dump(document), encoding="utf-8")
    else:
        external.write_text('{"unbound_live_mutated":true}', encoding="utf-8")
    result = _child(reference)
    assert result.returncode == 2, result.stdout
    assert "UNBOUND_CLI_INPUT" in result.stderr
    assert "unbound-live-mutated" not in result.stdout
    assert not selected.exists()


def test_project_relative_dynamic_exec_is_not_claimed_as_verified_source(tmp_path):
    source = _source(tmp_path)
    dynamic = source / "src/legacy_helper.py"
    marker = tmp_path / "dynamic-ran"
    dynamic.write_text("VALUE = 1\n", encoding="utf-8")
    preamble = "exec(compile(open(" + repr(str(dynamic)) + ", 'rb').read(), 'legacy_helper.py', 'exec'), {})\n"
    (source / "src/mod_agent/cli.py").write_text(preamble + CLI, encoding="utf-8")
    reference, _, _ = _prepare(tmp_path, source=source)
    dynamic.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n", encoding="utf-8")
    result = _child(reference)
    assert result.returncode == 7 and marker.exists()
    receipt = _read(tmp_path / "frozen/child_execution.json")
    assert receipt["status"] == "unknown"
    assert any(row["reason"] == "unbound_dynamic_exec_from_project" for row in receipt["failures"])


def test_default_relative_settings_resolves_to_frozen_resource_and_records_actual_read(tmp_path):
    source = _configuration_source(tmp_path)
    reference, _, _ = _prepare(tmp_path, source=source)
    result = _child(reference)
    assert result.returncode == 7, result.stderr
    receipt = _read(tmp_path / "frozen/child_execution.json")
    assert receipt["status"] == "verified"
    selected = receipt["selected_cli_inputs"]
    assert len(selected) == 1 and selected[0]["field"] == "settings"
    assert Path(selected[0]["path"]) == tmp_path / "frozen/config/settings.yaml"
    assert selected[0]["scope"] == "frozen_resource" and selected[0]["read_open_count"] == 1


@pytest.mark.parametrize("field", ["settings", "patch-worker-config", "stage-model-call-limits",
                                    "game-folder-build-binding", "external-patch-profile", "repository-profile", "task-agent-experience-config"])
def test_declared_external_input_is_hash_checked_and_retained_without_argv_change(tmp_path, field):
    source = _configuration_source(tmp_path)
    external = source / "config/settings.yaml" if field == "settings" else tmp_path / "outside.json"
    if field != "settings":
        external.write_text('{"same":true}', encoding="utf-8")
    digest = hashlib.sha256(external.read_bytes()).hexdigest()
    selected = tmp_path / "runs/run-one"
    argv = ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name,
            "--" + field, str(external)]
    reference, _, _ = _prepare(tmp_path, source=source, argv=argv, inputs=[(str(external), digest)])
    result = _child(reference)
    assert result.returncode == 7, result.stderr
    receipt = _read(tmp_path / "frozen/child_execution.json")
    assert receipt["status"] == "verified"
    entry = next(row for row in receipt["selected_cli_inputs"] if row["field"] == field.replace("-", "_"))
    assert entry["read_open_count"] == 1 and entry["sha256"] == digest
    assert Path(entry["retained_path"]).read_bytes() == external.read_bytes()
    assert _read(Path(reference["path"]))["business_argv"] == argv


def test_bound_input_changed_after_consumption_is_unknown_without_changing_business_exit(tmp_path):
    source = _configuration_source(tmp_path)
    external = tmp_path / "limits.json"
    external.write_text('{"limit":12}', encoding="utf-8")
    cli = source / "src/mod_agent/cli.py"
    cli.write_text(cli.read_text(encoding="utf-8").replace(
        "    return 7\n", "    Path(args.stage_model_call_limits).write_text('{\"limit\":99}', encoding='utf-8')\n    return 7\n"),
        encoding="utf-8")
    selected = tmp_path / "runs/run-one"
    argv = ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name,
            "--stage-model-call-limits", str(external)]
    reference, _, _ = _prepare(tmp_path, source=source, argv=argv,
        inputs=[(str(external), hashlib.sha256(external.read_bytes()).hexdigest())])
    result = _child(reference)
    assert result.returncode == 7, result.stderr
    receipt = _read(tmp_path / "frozen/child_execution.json")
    assert receipt["status"] == "unknown" and "EXTERNAL_REFERENCE_CHANGED" in receipt["reason"]
    assert json.loads(result.stdout)["stage_model_call_limits"] == {"limit": 12}


def test_bound_input_changed_after_startup_is_rejected_before_read(tmp_path):
    source = _configuration_source(tmp_path)
    external = tmp_path / "limits.json"
    external.write_text('{"limit":12}', encoding="utf-8")
    cli = source / "src/mod_agent/cli.py"
    cli.write_text(cli.read_text(encoding="utf-8").replace(
        "        if value:\n", "        if value:\n            Path(value).write_text('{\"limit\":99}', encoding='utf-8')\n"),
        encoding="utf-8")
    selected = tmp_path / "runs/run-one"
    argv = ["run-request", "--workspace-root", str(selected.parent), "--run-id", selected.name,
            "--stage-model-call-limits", str(external)]
    reference, _, _ = _prepare(tmp_path, source=source, argv=argv,
        inputs=[(str(external), hashlib.sha256(external.read_bytes()).hexdigest())])
    result = _child(reference)
    assert result.returncode == 1 and "CLI_INPUT_CHANGED_BEFORE_READ" in result.stderr
    assert result.stdout == ""
    receipt = _read(tmp_path / "frozen/child_execution.json")
    assert receipt["status"] == "unknown"
    assert any(row["event"] == "input_open" for row in receipt["failures"])
