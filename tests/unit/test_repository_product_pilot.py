"""Pure controller checks, not real model or business success evidence.

为什么要有它：先检验控制配置和阻断条件，不用模型运行替控制脚本排错。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_repository_product_pilot as pilot
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.native_backend_admission import freeze_native_backend_admission


def _filesystem(command):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    value = next((value for value in command if value.startswith("permissions.benchmark.filesystem=")), None)
    if value is None:
        name = command[command.index("--profile") + 1]
        value = (Path(pilot.os.environ["CODEX_HOME"]) / (name + ".config.toml")).read_text(encoding="utf-8")
    return tomllib.loads(value)["permissions"]["benchmark"]["filesystem"]


def test_absolute_real_b_does_not_require_external_product_readiness():
    assert "codex_executable_ref" not in pilot.configuration_keys({"mode": "absolute_real_B"})
    assert "codex_executable_ref" in pilot.configuration_keys({"mode": "matched_product_pair"})


def test_frozen_task_run_adapter_calls_the_shared_interface(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    assert not hasattr(pilot, "prepare_new_task") and not hasattr(pilot, "build_product_argv")
    arguments = ["--run-id", "dsh-controlled", "--request-id", "request-controlled",
        "--workspace-root", str(tmp_path / "runs")]
    execution_ref = _save(tmp_path / "execution.json", {"arguments": arguments})
    platform_ref = _save(tmp_path / "platform.json", {"controlled": True})
    document = {"business_argv": ["run-request", *arguments, "--", "Controlled intent"],
        "external_refs": [{"role": "bound_input", **execution_ref,
            "retained_path": "external/" + execution_ref["sha256"] + "/execution.json"}]}
    monkeypatch.setattr(pilot, "FrozenExecution", lambda _ref: SimpleNamespace(
        document=document, verify_files=lambda: None))
    observed = []
    expected = pilot.TaskRunCompleted(run_ref=execution_ref, run_id="dsh-controlled",
        request_id="request-controlled", status="awaiting_delivery",
        run_outcome_ref=_save(tmp_path / "outcome.json", {"run_id": "dsh-controlled"}),
        evidence_bundle_ref=_save(tmp_path / "manifest.json", {"run_id": "dsh-controlled"}))
    monkeypatch.setattr(pilot, "execute_task_run", lambda command: observed.append(command) or expected)

    assert pilot.run_task_run_snapshot(platform_ref) == 0
    assert observed == [pilot.OpenTaskRun(execution_config_ref=execution_ref, intent="Controlled intent")]
    raw = capsys.readouterr().out
    projected = json.loads(raw)
    assert projected == pilot._task_run_projection(expected)
    stdout = tmp_path / "stdout.json"
    stdout.write_text(raw, encoding="utf-8")
    process = {"stdout_ref": _ref(stdout)}
    plan = {"execution_config_ref": execution_ref, "run_id": "dsh-controlled"}
    assert pilot._task_run_process_result(process, plan) == projected
    stdout.write_text(json.dumps({**projected, "request_id": "forged-request"}), encoding="utf-8")
    process = {"stdout_ref": _ref(stdout)}
    with pytest.raises(ValueError, match="RESULT_IDENTITY_CHANGED"):
        pilot._task_run_process_result(process, plan)


def test_frozen_task_run_bootstrap_rejects_changed_source_before_import(tmp_path):
    root = tmp_path / "snapshot"
    script = root / "scripts/run_repository_product_pilot.py"
    script.parent.mkdir(parents=True)
    marker = tmp_path / "imported.txt"
    script.write_text("from pathlib import Path\nPath(" + repr(str(marker))
        + ").write_text('imported')\ndef run_task_run_snapshot(_ref): return 0\n", encoding="utf-8")
    manifest = root / "execution_snapshot.json"
    document = {"snapshot_root": str(root), "files": {
        "scripts/run_repository_product_pilot.py": {"sha256": "0" * 64}}}
    manifest.write_text(json.dumps(document), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    result = subprocess.run([sys.executable, "-I", "-B", "-c", pilot.TASK_RUN_CHILD_CODE,
        str(root), str(manifest), digest], capture_output=True, text=True)

    assert result.returncode != 0
    assert not marker.exists()


def test_native_reference_requires_own_profile_and_no_api_settings():
    keys = pilot.configuration_keys({"mode": "native_reference"})
    assert "native_reference_profile_ref" in keys
    assert "codex_model_availability_ref" in keys
    assert "codex_executable_ref" in keys
    assert "native_user_config_ref" in keys and "native_shell_ref" in keys
    assert "settings_ref" not in keys


@pytest.fixture
def native_reference(tmp_path, monkeypatch):
    from tests.unit.test_csharp_repository_backend import _profile

    repository, source = _profile(tmp_path)
    fake_home = tmp_path / "controlled-codex-home"
    fake_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(fake_home))
    user_config = fake_home / "config.toml"
    user_config.write_text('model_context_window=1000000\nmodel_auto_compact_token_limit=900000\n'
        'personality="pragmatic"\nservice_tier="priority"\nnotify=["CONTROLLED_PRIVATE_NOTIFY_ARGUMENT"]\n'
        '[mcp_servers.openaiDeveloperDocs]\nenabled=true\n[mcp_servers.node_repl]\nenabled=true\n', encoding="utf-8")
    shell = tmp_path / "native-powershell/pwsh.exe"
    shell.parent.mkdir()
    shell.write_bytes(b"CONTROLLED SHELL IDENTITY; NEVER EXECUTED")
    allowed = source / "src/Example/Allowed.cs"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("public class Allowed {}", encoding="utf-8")
    document = {**repository.data, "allowed_edit_paths": ["src/Example/Allowed.cs"]}
    repository_ref = _save(tmp_path / "native-public-profile.json", document)
    executable = tmp_path / "codex.exe"
    executable.write_text("CONTROLLED BINARY IDENTITY; NEVER EXECUTED", encoding="utf-8")
    problem, contract = tmp_path / "problem.txt", tmp_path / "contract.txt"
    problem.write_text("Controlled new public task.", encoding="utf-8")
    contract.write_text("Modify only the registered public source.", encoding="utf-8")
    profile = {"schema_version": "codex_native_reference_profile_v1", "model": "gpt-5.6-sol",
        "reasoning_effort": "high", "authentication": "chatgpt",
        "codex_home": str(Path(pilot.os.environ.get("CODEX_HOME") or Path.home() / ".codex").resolve())}
    config = {"mode": "native_reference", "root": str(tmp_path / "pilot"), "public_source_root": str(source),
        "intent_ref": _save(tmp_path / "intent.json", {"scope": "controlled native reference"}),
        "public_problem_ref": _ref(problem), "public_contract_ref": _ref(contract), "public_profile_ref": repository_ref,
        "qualification_ref": _save(tmp_path / "qualification.json", {"qualified": True,
            "evaluator_ref": _save(tmp_path / "private/evaluator.json", {"controlled": True})}),
        "codex_executable_ref": _ref(executable), "environment": {},
        "native_user_config_ref": _ref(user_config), "native_shell_ref": _ref(shell),
        "native_reference_profile_ref": _save(tmp_path / "native-profile.json", profile),
        "codex_model_availability_ref": _save(tmp_path / "availability.json", {
            "status": "listed", "binary_ref": _ref(executable), "model_generations": 0, "threads_created": 0,
            "models": [{"model": "gpt-5.6-sol", "hidden": False, "supportedReasoningEfforts": [{"reasoningEffort": "high"}]}]})}
    monkeypatch.setattr(pilot, "validate_frozen_evaluator", lambda _qualification: None)
    monkeypatch.setattr(pilot, "read_ollama_model_identity", lambda *_args: pytest.fail("Native reference cannot contact Ollama"))
    monkeypatch.setattr(pilot, "external_pilot_identity", lambda *_args: pytest.fail("Native reference cannot contact API"))
    return config


def _native_admission(tmp_path, plan_ref, plan, first_outputs=(1.0, 2.0, 3.0)):
    matrix_ref = _save(tmp_path / "native-matrix.json", {"passed": True})
    trials = []
    for sequence, seconds in enumerate(first_outputs, 1):
        process_ref = _save(tmp_path / f"native-process-{sequence}.json", {
            "sequence": sequence, "first_output_seconds": seconds, "exit_code": 0, "timeout": False})
        trials.append({"sequence": sequence, "configuration_identity": plan_ref["sha256"],
            "command_kind": "cmd", "first_output_seconds": seconds, "exit_code": 0,
            "timeout": False, "process_ref": process_ref})
    measurement = {
        "schema_version": "native_backend_admission_measurement_v1",
        "configuration_identity": plan_ref["sha256"], "plan_ref": plan_ref,
        "permission_profile_ref": plan["native_permission_profile_ref"],
        "thresholds": {"consecutive_commands": 3, "first_output_median_seconds_max": 15.0,
            "first_output_single_seconds_max": 30.0},
        "admission_trials": trials,
        "permission_matrix": {"passed": True, "required_checks": ["allow", "deny"],
            "checks": [{"name": "allow", "passed": True}, {"name": "deny", "passed": True}],
            "evidence_ref": matrix_ref},
        "comparison_axes": {name: {"status": "measured", "evidence_refs": []}
            for name in ("permission_scale", "cold_vs_repeated", "cmd_vs_powershell")},
    }
    measurement_ref = _save(tmp_path / "native-measurement.json", measurement)
    return freeze_native_backend_admission(measurement_ref, tmp_path / "native-admission.json")


def test_native_reference_cli_freeze_run_records_native_facts_and_child_only_auth_route(tmp_path, monkeypatch, capsys, native_reference):
    config = native_reference
    frozen_ref = _cli(monkeypatch, capsys, "freeze", _save(tmp_path / "config.json", config))["pilot_ref"]
    frozen = pilot.read_bound_json(frozen_ref)
    assert frozen["mode"] == "native_reference" and frozen["order"] == [pilot.NATIVE_ARM]
    assert frozen["model_calls"] == 0
    plan = pilot.read_bound_json(frozen["plans"][pilot.NATIVE_ARM])
    admission_ref = _native_admission(tmp_path, frozen["plans"][pilot.NATIVE_ARM], plan)
    assert "platform_ref" not in plan and "execution_config_ref" not in plan and "run_id" not in plan
    assert plan["product_kind"] == "native_codex_reference"
    assert plan["evaluator_ref"] == pilot.read_bound_json(config["qualification_ref"])["evaluator_ref"]
    assert Path(plan["prompt_path"]).read_text(encoding="utf-8") == "Controlled new public task.\n\nModify only the registered public source."
    command = plan["command"]
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="high"' in command and 'forced_login_method="chatgpt"' in command
    assert "--ephemeral" in command and "--ignore-user-config" not in command and "--ignore-rules" in command
    assert "mcp_servers.openaiDeveloperDocs.enabled=false" in command and "mcp_servers.node_repl.enabled=false" in command
    assert "notify=[]" in command
    assert plan["native_user_configuration"]["retained_settings"] == {
        "model_context_window": 1000000, "model_auto_compact_token_limit": 900000,
        "personality": "pragmatic", "service_tier": "priority"}
    assert plan["native_user_configuration"]["user_config_ref"] == config["native_user_config_ref"]
    assert "CONTROLLED_PRIVATE_NOTIFY_ARGUMENT" not in json.dumps(plan)
    assert command[command.index("--profile") + 1].startswith("repoforge-native-")
    assert pilot.read_bound_json(config["native_reference_profile_ref"])["codex_home"] == str(Path(plan["native_permission_profile_ref"]["path"]).parent)
    assert Path(plan["native_permission_profile_ref"]["path"]).read_bytes() == Path(plan["native_permission_profile_archive_ref"]["path"]).read_bytes()
    assert not any(value.startswith("permissions.benchmark.filesystem=") for value in command)
    assert "--oss" not in command and "--local-provider" not in command
    assert 'windows.sandbox="mxc"' in command and 'windows.sandbox="elevated"' not in command
    assert 'web_search="disabled"' in command and 'mcp_servers={}' in command
    for feature in ("apps", "plugins", "hooks", "memories", "multi_agent", "browser_use", "computer_use"):
        assert command[command.index(feature) - 1] == "--disable"
    filesystem = _filesystem(command)
    assert filesystem[":root"] == "deny" and filesystem[":minimal"] == "read"
    assert filesystem[":tmpdir"] == "write" and filesystem[":slash_tmp"] == "deny"
    assert filesystem[(Path(plan["workspace"]) / ".build/temp").as_posix()] == "write"
    for name in ("TEMP", "TMP"):
        if pilot.os.environ.get(name):
            assert filesystem[Path(pilot.os.environ[name]).resolve().as_posix()] == "deny"
    assert filesystem[(Path(plan["source_root"]) / "src/Example").as_posix()]["Allowed.cs"] == "write"
    assert filesystem[(Path(plan["workspace"]) / "materials").as_posix()] == "read"
    assert filesystem[Path(config["qualification_ref"]["path"]).parent.as_posix()] == "deny"
    assert filesystem[Path(config["native_shell_ref"]["path"]).parent.as_posix()] == {
        ".": "read", "profile.ps1": "deny", "Microsoft.PowerShell_profile.ps1": "deny"}
    assert 'permissions.benchmark.network.enabled=false' in command
    assert any(value.startswith("shell_environment_policy.set=") and "NUGET_PACKAGES" in value for value in command)
    monkeypatch.setenv("OPENAI_API_KEY", "CONTROLLED-DO-NOT-COPY")
    monkeypatch.setenv("CODEX_API_KEY", "CONTROLLED-DO-NOT-COPY")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "CONTROLLED-DO-NOT-COPY")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://controlled.invalid")
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        assert not {key.casefold() for key in kwargs["environment"]} & pilot._NATIVE_CONFLICT_ENV
        assert kwargs["environment"]["CODEX_HOME"] == pilot.read_bound_json(config["native_reference_profile_ref"])["codex_home"]
        assert kwargs["environment"]["TEMP"] == kwargs["environment"]["TMP"] == str(Path(plan["workspace"]) / ".build/temp")
        assert kwargs["stdin_path"] == plan["prompt_path"]
        output = kwargs["output"]
        output.mkdir()
        stdout = output / "stdout.bin"
        stdout.write_text('\n'.join(json.dumps(event) for event in [
            {"type": "thread.started", "thread_id": "controlled-ephemeral-native-session"},
            {"type": "item.completed", "item": {"type": "file_change", "changes": []}},
            {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 50}}]), encoding="utf-8")
        (Path(plan["source_root"]) / "src/Example/Allowed.cs").write_text("public class Allowed { public int Value => 1; }", encoding="utf-8")
        receipt = {"started": True, "timeout": False, "exit_code": 0, "stdout_ref": _ref(stdout)}
        _save(output / "process.json", receipt)
        return receipt

    monkeypatch.setattr(pilot, "execute", execute)
    result_ref = _cli(monkeypatch, capsys, "run", frozen_ref, pilot.NATIVE_ARM, admission_ref)["result_ref"]
    result = pilot.read_bound_json(result_ref)
    assert len(calls) == 1 and result["changed_source_paths"] == ["src/Example/Allowed.cs"]
    assert result["forbidden_source_changes"] == []
    assert result["task_agent_success"] is None and result["independent_business_result"] is None
    assert result["experience_comparison"] is None and result["delivery_process"] is None
    events = pilot.read_bound_json(result["native_events_ref"])
    assert events["usage_totals"] == {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 50}
    assert events["provider_send_count"] is None
    assert pilot.os.environ["OPENAI_API_KEY"] == "CONTROLLED-DO-NOT-COPY"
    assert "CONTROLLED-DO-NOT-COPY" not in Path(frozen["plans"][pilot.NATIVE_ARM]["path"]).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="ALREADY_STARTED"):
        _cli(monkeypatch, capsys, "run", frozen_ref, pilot.NATIVE_ARM, admission_ref)
    assert len(calls) == 1


def test_native_reference_not_admitted_stops_before_started_record_or_model(
    tmp_path, monkeypatch, capsys, native_reference,
):
    frozen_ref = _cli(monkeypatch, capsys, "freeze", _save(tmp_path / "config.json", native_reference))["pilot_ref"]
    frozen = pilot.read_bound_json(frozen_ref)
    plan_ref = frozen["plans"][pilot.NATIVE_ARM]
    plan = pilot.read_bound_json(plan_ref)
    admission_ref = _native_admission(tmp_path, plan_ref, plan, (12.0, 14.0, 31.0))
    monkeypatch.setattr(pilot, "execute", lambda *_args, **_kwargs: pytest.fail("Not-admitted route cannot execute"))

    with pytest.raises(ValueError, match="NATIVE_BACKEND_NOT_ADMITTED"):
        _cli(monkeypatch, capsys, "run", frozen_ref, pilot.NATIVE_ARM, admission_ref)

    assert not (Path(plan["root"]) / "started.json").exists()


def test_native_reference_freeze_reuses_public_snapshot_preparation_and_protects_copies(tmp_path, monkeypatch, capsys, native_reference):
    config = native_reference
    source = Path(config["public_source_root"])
    expected = source / "tests/Expectations/nested/example.verified.txt"
    expected.parent.mkdir(parents=True)
    expected.write_text("frozen public expectation", encoding="utf-8")
    profile = pilot.read_bound_json(config["public_profile_ref"])
    profile["diagnostics"] = {"kind": "verify_snapshots_v1", "expectation_root": "tests/Expectations",
        "compiler_output_roots": ["Library/bin", "Library/obj", "tests/bin", "tests/obj"],
        "received_markers": [".received.", ".DotNet8_0.received."]}
    config["public_profile_ref"] = _save(tmp_path / "diagnostic-profile.json", profile)
    frozen_ref = _cli(monkeypatch, capsys, "freeze", _save(tmp_path / "config.json", config))["pilot_ref"]
    frozen = pilot.read_bound_json(frozen_ref)
    plan = pilot.read_bound_json(frozen["plans"][pilot.NATIVE_ARM])
    assert len(plan["native_expected_refs"]) == 1
    copy_ref = plan["native_expected_refs"][0]
    assert copy_ref["sha256"] == _ref(expected)["sha256"]
    assert Path(copy_ref["path"]).read_text(encoding="utf-8") == "frozen public expectation"
    assert _ref(expected)["sha256"] == copy_ref["sha256"]
    assert pilot.source_snapshot(source) == plan["source_before"] == pilot.source_snapshot(plan["source_root"])
    received = [Path(reference["path"]) for reference in plan["native_received_seed_refs"]]
    assert {path.name for path in received} == {"example.received.txt", "example.DotNet8_0.received.txt"}
    assert all(path.read_bytes() == b"" and path.parent.name == "nested" for path in received)
    assert "known AccessDenied" in plan["native_public_check_environment"]["new_process_recreate_after_delete"]
    assert all((Path(plan["source_root"]) / "src/Example" / output).is_dir() for output in ("bin", "obj", "TestResults"))
    filesystem = _filesystem(plan["command"])
    assert filesystem[(Path(plan["workspace"]) / ".build/diagnostics").as_posix()] == {
        ".": "read", "nested": "read", "nested/example.received.txt": "write",
        "nested/example.DotNet8_0.received.txt": "write"}
    Path(copy_ref["path"]).write_text("tampered expected output", encoding="utf-8")
    monkeypatch.setattr(pilot, "execute", lambda *_args, **_kwargs: pytest.fail("Changed snapshot cannot run"))
    with pytest.raises(ValueError, match="BOUND_INPUT_CHANGED"):
        _cli(monkeypatch, capsys, "run", frozen_ref, pilot.NATIVE_ARM)


@pytest.mark.parametrize("failure", [None, "package", "nonpackage"])
def test_native_reference_readable_dependency_directory_is_manifest_bound(tmp_path, native_reference, failure):
    from src.mod_agent.domain.repository_probe_runner import offline_package_manifest

    config = native_reference
    profile = pilot.read_bound_json(config["public_profile_ref"])
    feed = tmp_path / "public-packages"
    feed.mkdir()
    package = feed / "controlled.1.0.0.nupkg"
    package.write_bytes(b"CONTROLLED PACKAGE; NO SDK EXECUTION")
    profile["probe_runner"] = {"offline_packages": {"path": str(feed), "manifest_sha256": pilot.canonical_hash(offline_package_manifest(feed))}}
    if failure == "package":
        package.write_bytes(b"changed")
    elif failure == "nonpackage":
        (feed / "unregistered.txt").write_text("NONSECRET CANARY", encoding="utf-8")
    args = (config["codex_executable_ref"]["path"], tmp_path / "workspace", tmp_path / "last.txt", [],
        profile, pilot.read_bound_json(config["native_reference_profile_ref"]))
    if failure:
        with pytest.raises(ValueError, match="NATIVE_REFERENCE_"):
            pilot.native_codex_command(*args, native_user_config_ref=config["native_user_config_ref"], native_shell_ref=config["native_shell_ref"])
    else:
        command = pilot.native_codex_command(*args, native_user_config_ref=config["native_user_config_ref"], native_shell_ref=config["native_shell_ref"])
        assert _filesystem(command)[feed.as_posix()] == "read"


def test_native_permission_literal_encoding_roundtrips_quote_and_control_paths(tmp_path):
    denied = [tmp_path / "literal", tmp_path / "apostrophe'name", tmp_path / "control\nname"]
    command = pilot.local_codex_command("codex.exe", tmp_path / "workspace", tmp_path / "last.txt", denied,
        tmp_path / "sdk/dotnet.exe", ["src/Example/Allowed.cs"], native=True)
    filesystem = _filesystem(command)
    for path in denied:
        assert filesystem[path.resolve().as_posix()] == "deny"


@pytest.mark.parametrize("failure", ["binary", "availability", "profile", "user_config", "shell", "public_material", "permissions", "permission_archive", "command"])
def test_native_reference_changed_input_never_starts(tmp_path, monkeypatch, capsys, native_reference, failure):
    config = native_reference
    frozen_ref = _cli(monkeypatch, capsys, "freeze", _save(tmp_path / "config.json", config))["pilot_ref"]
    frozen = pilot.read_bound_json(frozen_ref)
    plan = pilot.read_bound_json(frozen["plans"][pilot.NATIVE_ARM])
    if failure == "command":
        plan["command"] += ["--dangerously-bypass-approvals-and-sandbox"]
        frozen["plans"][pilot.NATIVE_ARM] = _save(tmp_path / "changed-plan.json", plan)
        frozen_ref = _save(tmp_path / "changed-pilot.json", frozen)
    else:
        reference = (plan["public_material_refs"][0] if failure == "public_material" else
            plan["native_permission_profile_archive_ref"] if failure == "permission_archive" else
            plan["native_permission_profile_ref"] if failure == "permissions" else config[{
            "binary": "codex_executable_ref", "availability": "codex_model_availability_ref", "profile": "native_reference_profile_ref",
            "user_config": "native_user_config_ref", "shell": "native_shell_ref"}[failure]])
        Path(reference["path"]).write_text("changed", encoding="utf-8")
    monkeypatch.setattr(pilot, "execute", lambda *_args, **_kwargs: pytest.fail("Changed input cannot execute"))
    with pytest.raises(ValueError):
        _cli(monkeypatch, capsys, "run", frozen_ref, pilot.NATIVE_ARM)
    assert not (Path(plan["root"]) / "started.json").exists()


@pytest.mark.parametrize("failure", ["mode", "model", "effort", "authentication", "home", "metadata", "api_settings", "credential_environment"])
def test_native_reference_invalid_route_rejects_before_copy(tmp_path, monkeypatch, native_reference, failure):
    config = native_reference
    profile = pilot.read_bound_json(config["native_reference_profile_ref"])
    if failure == "mode":
        config["mode"] = "matched_product_pair"
    elif failure in {"model", "effort", "authentication", "home"}:
        profile[{"model": "model", "effort": "reasoning_effort", "authentication": "authentication", "home": "codex_home"}[failure]] = (
            str(tmp_path / "other-auth") if failure == "home" else "unexpected")
        config["native_reference_profile_ref"] = _save(tmp_path / "wrong-profile.json", profile)
    elif failure == "metadata":
        observed = pilot.read_bound_json(config["codex_model_availability_ref"])
        observed["models"][0]["supportedReasoningEfforts"] = []
        config["codex_model_availability_ref"] = _save(tmp_path / "wrong-availability.json", observed)
    elif failure == "api_settings":
        config["settings_ref"] = {"path": "not-read", "sha256": "not-read"}
    else:
        config["environment"]["oPeNaI_aPi_KeY"] = "CONTROLLED-NEVER-FROZEN"
    monkeypatch.setattr(pilot, "validate_frozen_evaluator", lambda _value: pytest.fail("Reject before preparation"))
    with pytest.raises(ValueError):
        pilot.freeze(_save(tmp_path / "invalid-native-config.json", config))
    assert not Path(config["root"]).exists()


@pytest.mark.parametrize("override", [
    'developer_instructions="UNREGISTERED_PROMPT"\n',
    '[model_providers.unregistered]\nname="UNREGISTERED_PROVIDER"\n',
    '[mcp_servers.unregistered]\nenabled=true\n',
    '[shell_environment_policy.set]\nOPENAI_API_KEY="CONTROLLED_NEVER_RECORDED"\n'])
def test_native_user_configuration_rejects_unregistered_overrides_even_with_new_hash(tmp_path, native_reference, override):
    config = native_reference
    path = Path(config["native_user_config_ref"]["path"])
    path.write_text(override, encoding="utf-8")
    config["native_user_config_ref"] = _ref(path)
    with pytest.raises(ValueError, match="UNREGISTERED_USER_CONFIG_OVERRIDE"):
        pilot.freeze(_save(tmp_path / "invalid-config.json", config))
    assert not Path(config["root"]).exists()


def test_equivalent_absolute_reference_spelling_preserves_content_binding(tmp_path):
    reference = _save(tmp_path / "reference.json", {"value": "original"})
    (tmp_path / "detour").mkdir()
    alias = {**reference, "path": (tmp_path / "detour" / ".." / "reference.json").as_posix()}
    pilot._verify(alias)
    pilot._verify({**reference, "path": Path(reference["path"]).as_posix()})
    Path(reference["path"]).write_text('{"value":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="BOUND_INPUT_CHANGED"):
        pilot._verify(alias)


def test_reference_path_compatibility_does_not_accept_extra_fields_or_relative_paths(tmp_path):
    reference = _save(tmp_path / "reference.json", {"value": "original"})
    for changed in ({**reference, "extra": True}, {**reference, "path": "reference.json"}):
        with pytest.raises(ValueError, match="BOUND_INPUT_CHANGED"):
            pilot._verify(changed)


@pytest.mark.parametrize("mode", ["absolute_real_B", "matched_product_pair"])
@pytest.mark.parametrize("selected,with_experience", [(None, False), ("enabled", False), ("disabled", False), ("enabled", True)],
    ids=["default", "enabled", "disabled", "catalog"])
def test_semantic_state_reaches_generated_configuration_and_product_argv(tmp_path, monkeypatch, capsys, mode, selected, with_experience, external=False):
    from tests.unit.test_csharp_repository_backend import _profile

    profile, source = _profile(tmp_path)
    problem, contract, settings, executable = (tmp_path / name for name in ("problem.txt", "contract.txt", "settings.yaml", "codex.exe"))
    problem.write_text("Double the input while preserving public signatures.", encoding="utf-8")
    contract.write_text("Use the registered public checks.", encoding="utf-8")
    settings.write_text("llm: {}\n", encoding="utf-8")
    executable.write_text("CONTROLLED TEST DOUBLE; NOT EXECUTED", encoding="utf-8")
    config = {"mode": mode, "root": str(tmp_path / "pilot"), "public_source_root": str(source),
        "intent_ref": _save(tmp_path / "intent.json", {"kind": "controlled_pilot_freeze"}),
        "public_profile_ref": profile.reference, "public_problem_ref": _ref(problem), "public_contract_ref": _ref(contract),
        "qualification_ref": _save(tmp_path / "qualification.json", {"qualified": True,
            "evaluator_ref": _save(tmp_path / "evaluator.json", {"controlled": True})}),
        "settings_ref": _ref(settings), "environment": {}}
    if mode == "matched_product_pair":
        config["codex_executable_ref"] = _ref(executable)
    if selected is not None:
        config["task_agent_semantic_state"] = selected
    if external:
        config["external_profile_ref"] = _save(tmp_path / "openai-profile.json", {
            "profile_version": "external_openai_responses_profile_v1", "base_url": "https://api.openai.com/v1",
            "model": "controlled-reference", "model_catalog_url": "https://api.openai.com/v1/models",
            "context_window": 65536, "output_budget": 4096, "timeout_seconds": 600,
            "credential_env": "CONTROLLED_REFERENCE_KEY", "reasoning_effort": "medium"})
        monkeypatch.setattr(pilot, "external_pilot_identity", lambda profile, expected=None: {
            "provider": "openai_responses", "model": profile.model, "model_digest": None})
    if with_experience:
        experience = {"schema_version": "task_agent_experience_config_v1", "store_root": str(tmp_path / "experience"),
            "snapshot_ref": None, "extract_on_terminal": False, "episode_catalog": {"catalog_id": "real-runs"}}
        config["experience_config_ref"] = _save(tmp_path / "experience-config.json", experience)
    captured = []

    def snapshot(**kwargs):
        captured.append(kwargs)
        return _save(Path(kwargs["snapshot_root"]) / "execution_snapshot.json", {
            "argv": kwargs["business_argv"], "controlled": True})

    monkeypatch.setattr(pilot, "prepare_snapshot", snapshot)
    monkeypatch.setattr(pilot, "validate_frozen_evaluator", lambda _qualification: None)
    monkeypatch.setattr(pilot, "read_ollama_model_identity", lambda _url, model: {"model": model, "digest": pilot.DIGEST})
    monkeypatch.setattr(pilot, "local_codex_command", lambda *_args: ["controlled-codex-not-executed"])
    configuration = _save(tmp_path / "pilot-config.json", config)
    pilot.freeze(configuration)
    pilot_ref = json.loads(capsys.readouterr().out)["pilot_ref"]
    frozen = pilot.read_bound_json(pilot_ref)
    plan = pilot.read_bound_json(frozen["plans"]["RepoForge_B"])
    execution = pilot.read_bound_json(plan["execution_config_ref"])
    expected = selected or "enabled"
    for arguments in (execution["arguments"], captured[0]["business_argv"]):
        assert arguments.count("--task-agent-semantic-state") == 1
        assert arguments[arguments.index("--task-agent-semantic-state") + 1] == expected
        if with_experience:
            assert arguments.count("--task-agent-experience-config") == 1
            assert arguments[arguments.index("--task-agent-experience-config") + 1] == config["experience_config_ref"]["path"]
        else:
            assert "--task-agent-experience-config" not in arguments
    assert plan["task_agent_semantic_state"] == expected
    assert plan["configuration_ref"] == configuration
    assert configuration in plan["bound_inputs"]
    assert plan["command"] == pilot.frozen_task_run_command(plan["platform_ref"])
    assert not any("run_frozen_product.py" in part for part in plan["command"])
    assert frozen["model_calls"] == 0 and not (Path(plan["root"]) / "started.json").exists()
    if external:
        from types import SimpleNamespace

        reference = config["external_profile_ref"]
        assert plan["external_profile_ref"] == reference and reference in plan["bound_inputs"]
        assert (reference["path"], reference["sha256"]) in captured[0]["bound_inputs"]
        assert execution["external_profile_binding"]["configuration_ref"] == reference
        arguments = captured[0]["business_argv"]
        assert "--local-ollama" not in arguments and "--allow-source-disclosure" in arguments
        assert arguments[arguments.index("--external-patch-profile") + 1] == reference["path"]
        assert arguments[arguments.index("--api-key-env") + 1] == "CONTROLLED_REFERENCE_KEY"
        monkeypatch.setattr(pilot, "FrozenExecution", lambda _reference: SimpleNamespace(verify_files=lambda: None))
        monkeypatch.setattr(pilot, "external_pilot_identity", lambda *_args: (_ for _ in ()).throw(ValueError("MODEL_CHANGED")))
        monkeypatch.setattr(pilot, "execute", lambda *_args, **_kwargs: pytest.fail("Changed model cannot start a process"))
        with pytest.raises(ValueError, match="MODEL_CHANGED"):
            pilot.run(pilot_ref, "RepoForge_B")
        assert not (Path(plan["root"]) / "started.json").exists()
        Path(reference["path"]).write_text("{}", encoding="utf-8")
        monkeypatch.setattr(pilot, "external_pilot_identity", lambda *_args: pytest.fail("Changed profile must reject before catalogue"))
        with pytest.raises(ValueError, match="BOUND_INPUT_CHANGED"):
            pilot.run(pilot_ref, "RepoForge_B")
    if with_experience:
        reference = config["experience_config_ref"]
        assert plan["experience_config_ref"] == reference and reference in plan["bound_inputs"]
        assert (reference["path"], reference["sha256"]) in captured[0]["bound_inputs"]
        assert plan["experience_policy"] == {key: experience[key] for key in ("snapshot_ref", "extract_on_terminal", "episode_catalog")}
        assert "no injected strategy or terminal extraction configured" not in plan["configuration_difference"]
        Path(reference["path"]).write_text("{}", encoding="utf-8")
        monkeypatch.setattr(pilot, "read_ollama_model_identity", lambda *_args: pytest.fail("Changed input must reject before provider admission"))
        with pytest.raises(ValueError, match="BOUND_INPUT_CHANGED"):
            pilot.run(pilot_ref, "RepoForge_B")
        assert not (Path(plan["root"]) / "started.json").exists()
    else:
        assert "experience_config_ref" not in plan
        assert "no injected strategy or terminal extraction configured" in plan["configuration_difference"]


def test_external_absolute_pilot_uses_registered_profile_and_existing_frozen_runner(tmp_path, monkeypatch, capsys):
    test_semantic_state_reaches_generated_configuration_and_product_argv(
        tmp_path, monkeypatch, capsys, "absolute_real_B", "enabled", False, external=True)


@pytest.mark.parametrize("mode", ["strategy_pair", "matched_product_pair"])
def test_external_profile_cannot_relabel_a_different_model_as_a_common_model_pair(tmp_path, mode):
    with pytest.raises(ValueError, match="EXTERNAL_PROFILE_REQUIRES_ABSOLUTE_TASK_AGENT"):
        pilot.freeze(_save(tmp_path / "invalid-pair.json", {"mode": mode, "external_profile_ref": None}))


def test_external_settings_credentials_reject_before_metadata_or_output(tmp_path, monkeypatch):
    reference = _save(tmp_path / "profile.json", {
        "profile_version": "external_openai_responses_profile_v1", "base_url": "https://api.openai.com/v1",
        "model": "controlled-reference", "model_catalog_url": "https://api.openai.com/v1/models",
        "context_window": 65536, "output_budget": 4096, "timeout_seconds": 600})
    settings = tmp_path / "settings.yaml"
    settings.write_text('llm: {}\nembedding:\n  api_key: "CONTROLLED-FAKE-SECRET"\n', encoding="utf-8")
    before = settings.read_bytes()
    root = tmp_path / "pilot"
    monkeypatch.setattr(pilot, "external_pilot_identity", lambda *_args: pytest.fail("No catalogue call"))
    with pytest.raises(ValueError, match="EMBEDDED_CREDENTIAL_DENIED"):
        pilot.freeze(_save(tmp_path / "config.json", {"mode": "absolute_real_B", "root": str(root),
            "external_profile_ref": reference, "settings_ref": _ref(settings)}))
    assert not root.exists() and settings.read_bytes() == before


@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "openai_api_key"])
def test_pilot_never_freezes_credential_environment_values(tmp_path, name):
    reference = _save(tmp_path / "profile.json", {
        "profile_version": "external_openai_responses_profile_v1", "base_url": "https://api.openai.com/v1",
        "model": "controlled-reference", "model_catalog_url": "https://api.openai.com/v1/models",
        "context_window": 65536, "output_budget": 4096, "timeout_seconds": 600})
    with pytest.raises(ValueError, match="PILOT_CREDENTIAL_VALUE_MUST_NOT_BE_FROZEN"):
        pilot.external_pilot_profile({"mode": "absolute_real_B", "external_profile_ref": reference,
            "environment": {name: "CONTROLLED-FAKE-VALUE"}})


@pytest.mark.parametrize("value", [None, "not-a-reference", {"path": "relative.json", "sha256": "a" * 64}])
def test_supplied_invalid_experience_ref_is_not_ignored(tmp_path, monkeypatch, value):
    monkeypatch.setattr(pilot, "validate_frozen_evaluator", lambda *_args: pytest.fail("No evaluator preparation"))
    monkeypatch.setattr(pilot, "read_ollama_model_identity", lambda *_args: pytest.fail("No provider admission"))
    config = _save(tmp_path / "invalid-experience.json", {"mode": "absolute_real_B", "experience_config_ref": value})
    with pytest.raises(ValueError):
        pilot.freeze(config)


@pytest.mark.parametrize("mode", ["absolute_real_B", "matched_product_pair"])
@pytest.mark.parametrize("value", [None, True, "on", {}], ids=["null", "boolean", "unknown", "object"])
def test_invalid_semantic_state_is_rejected_before_freeze_preparation(tmp_path, monkeypatch, mode, value):
    monkeypatch.setattr(pilot, "validate_frozen_evaluator", lambda _value: pytest.fail("No evaluator preparation"))
    monkeypatch.setattr(pilot, "read_ollama_model_identity", lambda *_args: pytest.fail("No provider admission"))
    configuration = _save(tmp_path / "invalid.json", {"mode": mode, "task_agent_semantic_state": value})
    with pytest.raises(ValueError, match="SEMANTIC_STATE_POLICY_INVALID"):
        pilot.freeze(configuration)


def test_local_command_preserves_native_provider_and_private_boundaries(tmp_path):
    workspace = tmp_path / "workspace"
    project = workspace / "source/test/Example"
    project.mkdir(parents=True)
    (project / "Example.csproj").write_text("<Project/>", encoding="utf-8")
    denied = tmp_path / "private"
    command = pilot.local_codex_command("codex.exe", workspace, tmp_path / "last.txt", [denied],
        tmp_path / "sdk/dotnet.exe", ["src/Example/Allowed.cs"])
    assert command[command.index("--model") + 1] == pilot.MODEL
    assert command[command.index("--local-provider") + 1] == "ollama"
    assert "--oss" in command and "unbounded_connection_retries" in command
    assert 'windows.sandbox="elevated"' in command and 'windows.sandbox="mxc"' not in command
    assert not any("model_reasoning_effort" in part for part in command)
    filesystem = next(part for part in command if part.startswith("permissions.benchmark.filesystem="))
    assert json.dumps(denied.resolve().as_posix()) + '="deny"' in filesystem
    assert json.dumps((workspace / "materials").resolve().as_posix()) + '="read"' in filesystem
    assert json.dumps((workspace / "source").resolve().as_posix()) + '="read"' in filesystem
    assert '"Allowed.cs"="write"' in filesystem
    assert '"."="write"' not in filesystem
    assert json.dumps(project.resolve().as_posix() + "/TestResults") + '="write"' in filesystem
    assert '"C:/tmp"="deny"' in filesystem
    assert 'permissions.benchmark.network.enabled=false' in command
    assert "--yolo" not in command


@pytest.mark.parametrize("relative", ["../escape.cs", "src/Example/../escape.cs", "root.cs"])
def test_permissions_reject_noncanonical_source_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        pilot.local_codex_command("codex.exe", tmp_path, tmp_path / "last.txt", [],
            tmp_path / "sdk/dotnet.exe", [relative])


def test_changed_reference_is_rejected(tmp_path):
    reference = _save(tmp_path / "bound.json", {"value": 1})
    (tmp_path / "bound.json").write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="BOUND_INPUT_CHANGED"):
        pilot._verify(reference)


@pytest.mark.parametrize("failure", [None, "adapter_path", "qualification"])
@pytest.mark.parametrize("snapshot_key", ["snapshot_ref", "platform_ref"])
def test_frozen_qualification_uses_exact_original_adapter(tmp_path, monkeypatch, failure, snapshot_key):
    root = tmp_path / "frozen"
    adapter = root / "src/mod_agent/evaluation/strategy_repository_evaluator.py"
    adapter.parent.mkdir(parents=True)
    for package in (root / "src", root / "src/mod_agent", adapter.parent):
        (package / "__init__.py").write_text("", encoding="utf-8")
    adapter.write_text("def _configuration(reference):\n"
        "    import json\n"
        "    from pathlib import Path\n"
        "    value = json.loads(Path(reference['path']).read_text())\n"
        "    assert value['adoption_adapter_ref']['path'] == str(Path(__file__).resolve())\n"
        "    assert value['qualified']\n", encoding="utf-8")
    calls = []
    timeouts = []
    original_run = pilot.subprocess.run

    def run(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(pilot.subprocess, "run", run)

    class Frozen:
        def __init__(self, reference):
            assert reference == {"frozen": True}
            self.root = root

        def verify_files(self):
            calls.append("verified")

    monkeypatch.setattr(pilot, "FrozenExecution", Frozen)
    adapter_ref = _ref(adapter)
    if failure == "adapter_path":
        adapter_ref["path"] = str(tmp_path / "mutable/adapter.py")
    evaluator_ref = _save(tmp_path / "evaluator.json", {
        "adoption_adapter_ref": adapter_ref, "qualified": failure != "qualification"})
    qualification = {"evaluator_ref": evaluator_ref, snapshot_key: {"frozen": True}}
    if failure == "adapter_path":
        with pytest.raises(ValueError, match="QUALIFIED_ADAPTER_OUTSIDE_FROZEN_PLATFORM"):
            pilot.validate_frozen_evaluator(qualification)
    elif failure == "qualification":
        with pytest.raises(pilot.subprocess.CalledProcessError):
            pilot.validate_frozen_evaluator(qualification)
    else:
        pilot.validate_frozen_evaluator(qualification)
        assert calls == ["verified", "verified"]
    assert not list(root.rglob("*.pyc"))
    assert timeouts == ([] if failure == "adapter_path" else [300])


def test_delivery_without_native_result_does_not_apply_or_build(tmp_path, monkeypatch):
    class Frozen:
        def __init__(self, _reference):
            pass

        def verify_files(self):
            pass

    monkeypatch.setattr(pilot, "FrozenExecution", Frozen)
    reference = _save(tmp_path / "plan.json", {"root": str(tmp_path), "run_root": str(tmp_path / "missing"),
        "platform_ref": {}})
    assert pilot.deliver(reference) == {"status": "native_result_missing", "delivered": False}


def test_delivery_does_not_infer_handoff_from_candidate_or_exit(tmp_path, monkeypatch):
    class Frozen:
        def __init__(self, _reference):
            pass

        def verify_files(self):
            pass

    monkeypatch.setattr(pilot, "FrozenExecution", Frozen)
    run_root = tmp_path / "run"
    run_root.mkdir()
    _save(run_root / "run_result.json", {"status": "passed", "candidate_ref": "candidate"})
    reference = _save(tmp_path / "plan.json", {"root": str(tmp_path), "run_root": str(run_root), "platform_ref": {}})
    assert pilot.deliver(reference) == {"status": "no_product_handoff", "delivered": False}
    assert not (tmp_path / "delivery").exists()


def test_source_audit_uses_the_repository_backend_generated_output_boundary(tmp_path):
    (tmp_path / "source.cs").write_text("original", encoding="utf-8")
    before = pilot.source_snapshot(tmp_path)
    (tmp_path / "TestResults").mkdir()
    (tmp_path / "TestResults/received.txt").write_text("diagnostic", encoding="utf-8")
    assert pilot.source_snapshot(tmp_path) == before
    (tmp_path / "unapproved.cs").write_text("edit", encoding="utf-8")
    assert "unapproved.cs" in pilot.source_snapshot(tmp_path)


def test_successful_apply_continues_to_existing_build_and_delivery_reader(tmp_path, monkeypatch):
    from src.mod_agent import reviewed_patch_continuation as continuation
    from src.mod_agent import task_agent_v2_delivery as delivery

    class Frozen:
        def __init__(self, _reference):
            pass

        def verify_files(self):
            pass

    monkeypatch.setattr(pilot, "FrozenExecution", Frozen)
    run_root = tmp_path / "run"
    run_root.mkdir()
    _save(run_root / "run_result.json", {"v2_delivery_checkpoint_ref": "checkpoint"})
    store = run_root / "a/task-agent-execution"
    store.mkdir(parents=True)
    _save(store / "execution.json", {"status": "passed"})
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return {"prepare": {"displayed_binding": {"exact": True}},
                "apply": {"status": "applied_in_isolation", "receipt_ref": "apply"},
                "build": {"status": "built", "receipt_ref": "build"}}[kwargs["action"]]

    monkeypatch.setattr(continuation, "continue_v2_delivery", execute)
    monkeypatch.setattr(delivery, "prepare_v2_task_delegation", lambda **kwargs: "delegation")
    monkeypatch.setattr(delivery, "read_completed_v2_delivery", lambda **kwargs: {"status": "delivered"})
    reference = _save(tmp_path / "plan.json", {"root": str(tmp_path), "run_root": str(run_root),
        "platform_ref": {}, "policy_id": "policy", "authorization_ref": {}})
    assert pilot.deliver(reference) == {"status": "delivered"}
    assert [row["action"] for row in calls] == ["prepare", "apply", "build"]
    assert calls[-1]["checkpoint_ref"] == "apply"
    assert calls[-1]["allow_build"] is True


@pytest.fixture
def strategy_pilot(tmp_path, monkeypatch):
    from src.mod_agent.evaluation.strategy_trial_preparation import prepare_strategy_trials
    from src.mod_agent.model_qualification import canonical_hash, read_bound_json
    from tests.unit.test_strategy_trial_preparation import trial_input

    input_ref, _, source = trial_input.__wrapped__(tmp_path)
    document = read_bound_json(input_ref)
    protocol = read_bound_json(document["protocol_ref"])
    protocol["invariants"]["model"] = {"name": "controlled-model", "base_url": "http://127.0.0.1:11434",
        "digest": "controlled-digest", "context_window": 32768, "output_tokens": 2048}
    protocol["invariants_sha256"] = canonical_hash(protocol["invariants"])
    document["protocol_ref"] = _save(tmp_path / "registered-protocol.json", protocol)
    input_ref = _save(tmp_path / "registered-input.json", document)
    prepared_ref = prepare_strategy_trials(input_ref, tmp_path / "prepared")
    prepared = read_bound_json(prepared_ref)
    profile_ref = _ref(tmp_path / "repository/repository-profile.json")
    qualification = _save(tmp_path / "controlled-qualification.json", {
        "qualified": True, "profile_ref": profile_ref, "public_source_root": str(source),
        "evaluator_ref": _save(tmp_path / "controlled-evaluator.json", {"controlled": True})})
    intent = _save(tmp_path / "intent.json", {"wall_seconds": 1800})
    config = {"mode": "strategy_pair", "root": str(tmp_path / "pilot"), "prepared_ref": prepared_ref,
        "intent_ref": intent, "qualification_ref": qualification, "public_profile_ref": profile_ref,
        "wall_seconds": 1800, "environment": {}}
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return _save(Path(kwargs["snapshot_root"]) / "execution_snapshot.json", {
            "argv": kwargs["business_argv"], "controlled": True, "files": {"controlled": "same"}})

    class Frozen:
        def __init__(self, reference):
            self.reference = reference

        def verify_files(self):
            pilot._verify(self.reference)

    monkeypatch.setattr(pilot, "prepare_snapshot", snapshot)
    monkeypatch.setattr(pilot, "FrozenExecution", Frozen)
    monkeypatch.setattr(pilot, "validate_frozen_evaluator", lambda qualification: None)
    monkeypatch.setattr(pilot, "read_ollama_model_identity", lambda url, model: {
        "model": model, "digest": "controlled-digest"})
    return config, prepared, calls


def _cli(monkeypatch, capsys, mode, reference, arm=None, native_admission=None):
    monkeypatch.setattr(pilot.sys, "argv", ["pilot", mode, "--reference", reference["path"],
        "--sha256", reference["sha256"], *(["--arm", arm] if arm else []),
        *(["--native-admission-reference", native_admission["path"],
            "--native-admission-sha256", native_admission["sha256"]] if native_admission else [])])
    pilot.main()
    return json.loads(capsys.readouterr().out)


def test_strategy_pair_cli_preserves_prepared_inputs_and_explicit_order(tmp_path, monkeypatch, capsys, strategy_pilot):
    config, prepared, calls = strategy_pilot
    reference = _save(tmp_path / "pilot-config.json", config)
    result = _cli(monkeypatch, capsys, "freeze", reference)
    frozen = pilot.read_bound_json(result["pilot_ref"])
    assert frozen["mode"] == "strategy_pair" and frozen["order"] == ["B+", "B"]
    assert result["arms_started"] == 0
    assert [call["business_argv"] for call in calls] == [t["argv"] for t in prepared["trials"]]
    for trial in prepared["trials"]:
        plan = pilot.read_bound_json(frozen["plans"][trial["arm"]])
        assert plan["run_id"] == trial["run_id"] and plan["run_root"] == trial["run_root"]
        assert Path(plan["run_root"]) == Path(plan["root"]) / "runs" / plan["run_id"]
        assert plan["execution_config_ref"] == trial["execution_config_ref"]
        assert not Path(plan["run_root"]).exists()
        assert not (Path(plan["root"]) / "started.json").exists()
        experience = pilot.read_bound_json(trial["experience_config_ref"])
        assert bool(experience["snapshot_ref"]) == (trial["arm"] == "B+")
    with pytest.raises(ValueError, match="SERIAL_ARM_ORDER_REQUIRED"):
        _cli(monkeypatch, capsys, "run", result["pilot_ref"], "B")


@pytest.mark.parametrize("failure", ["qualification", "profile", "source", "budget", "model", "experience"])
def test_strategy_pair_invalid_preparation_never_freezes(tmp_path, strategy_pilot, failure):
    config, prepared, calls = strategy_pilot
    if failure == "qualification":
        qualification = pilot.read_bound_json(config["qualification_ref"])
        qualification["qualified"] = False
        config["qualification_ref"] = _save(tmp_path / "unqualified.json", qualification)
    elif failure == "profile":
        qualification = pilot.read_bound_json(config["qualification_ref"])
        qualification["profile_ref"] = {"path": "other", "sha256": "changed"}
        config["qualification_ref"] = _save(tmp_path / "wrong-profile.json", qualification)
    elif failure == "budget":
        config["wall_seconds"] = 3600
    elif failure == "source":
        wrong = tmp_path / "wrong-source"
        wrong.mkdir()
        (wrong / "other.cs").write_text("unrelated source", encoding="utf-8")
        qualification = pilot.read_bound_json(config["qualification_ref"])
        qualification["public_source_root"] = str(wrong)
        config["qualification_ref"] = _save(tmp_path / "wrong-source-qualification.json", qualification)
    elif failure == "model":
        trial = prepared["trials"][0]
        path = Path(trial["execution_config_ref"]["path"])
        value = json.loads(path.read_bytes())
        value["arguments"][value["arguments"].index("--ollama-model") + 1] = "different"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        Path(prepared["trials"][0]["experience_config_ref"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        pilot.freeze(_save(tmp_path / "pilot-config.json", config))
    assert calls == [] and not Path(config["root"]).exists()


def test_strategy_pair_cli_uses_one_deadline_for_execution_and_delivery(tmp_path, monkeypatch, capsys, strategy_pilot):
    config, prepared, _ = strategy_pilot
    reference = _save(tmp_path / "pilot-config.json", config)
    frozen_ref = _cli(monkeypatch, capsys, "freeze", reference)["pilot_ref"]
    calls = []
    now = {"clock": 10.0, "utc": 100.0}
    monkeypatch.setattr(pilot.time, "monotonic", lambda: now["clock"])
    monkeypatch.setattr(pilot.time, "time", lambda: now["utc"])
    monkeypatch.setenv("REPOFORGE_TASK_DEADLINE_UTC", "1000")
    frozen = pilot.read_bound_json(frozen_ref)
    plans = {arm: pilot.read_bound_json(plan_ref) for arm, plan_ref in frozen["plans"].items()}

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        now["clock"] += 25
        kwargs["output"].mkdir(parents=True, exist_ok=True)
        receipt = {"started": True, "timeout": False, "exit_code": 0}
        selected = next((plan for plan in plans.values() if plan["command"] == argv), None)
        if selected is not None:
            _, options = pilot._configuration_arguments(selected["execution_config_ref"]["path"])
            outcome = _save(kwargs["output"] / "outcome.json", {"run_id": selected["run_id"]})
            manifest = _save(kwargs["output"] / "manifest.json", {"run_id": selected["run_id"]})
            stdout = kwargs["output"] / "stdout.bin"
            stdout.write_text(json.dumps({"run_ref": selected["execution_config_ref"],
                "run_id": selected["run_id"], "request_id": options["--request-id"],
                "status": "awaiting_delivery", "run_outcome_ref": outcome,
                "evidence_bundle_ref": manifest}), encoding="utf-8")
            receipt["stdout_ref"] = _ref(stdout)
        _save(kwargs["output"] / "process.json", receipt)
        return receipt

    monkeypatch.setattr(pilot, "execute", execute)
    _cli(monkeypatch, capsys, "run", frozen_ref, "B+")
    assert len(calls) == 2
    assert calls[0][1]["deadline"] == calls[1][1]["deadline"] == 910
    assert calls[0][1]["environment"]["REPOFORGE_TASK_DEADLINE_UTC"] == "1000.0"
    assert "deliver" in calls[1][0]
    first = prepared["trials"][0]
    started = json.loads((Path(first["run_root"]).parent.parent / "started.json").read_bytes())
    assert started["run_id"] == first["run_id"] and started["deadline_utc"] == 1000
    with pytest.raises(ValueError, match="ALREADY_STARTED"):
        _cli(monkeypatch, capsys, "run", frozen_ref, "B+")
    monkeypatch.delenv("REPOFORGE_TASK_DEADLINE_UTC")
    now.update(clock=2000, utc=2100)
    _cli(monkeypatch, capsys, "run", frozen_ref, "B")
    assert len(calls) == 4
    assert calls[2][1]["deadline"] == calls[3][1]["deadline"] == 3800


def test_qualification_conflicting_snapshot_aliases_rejected_before_loading():
    with pytest.raises(ValueError, match="QUALIFIED_PLATFORM_IDENTITY_REQUIRED"):
        pilot.validate_frozen_evaluator({"evaluator_ref": {}, "snapshot_ref": {"path": "a"},
            "platform_ref": {"path": "b"}})


@pytest.mark.parametrize("change_platform", [False, True])
def test_strategy_pair_cli_real_snapshot_distinguishes_bound_inputs_from_platform(
        tmp_path, monkeypatch, capsys, strategy_pilot, change_platform):
    from src.mod_agent.evaluation.trial_comparison_identity import platform_identity
    from src.mod_agent.platform_execution_snapshot import FrozenExecution, prepare_snapshot

    config, prepared, _ = strategy_pilot
    miniature = tmp_path / "mini-platform"
    for directory in ("src", "scripts", "config"):
        target = miniature / directory / "retained.txt"
        target.parent.mkdir(parents=True)
        target.write_text("fixed platform", encoding="utf-8")
    monkeypatch.setattr(pilot, "REPO", miniature)
    snapshots = []

    def real_snapshot(**kwargs):
        if snapshots and change_platform:
            (miniature / "src/retained.txt").write_text("changed platform", encoding="utf-8")
        result = prepare_snapshot(**kwargs)
        FrozenExecution(result).verify_files()
        snapshots.append(result)
        return result

    monkeypatch.setattr(pilot, "prepare_snapshot", real_snapshot)
    reference = _save(tmp_path / "pilot-config.json", config)
    if change_platform:
        with pytest.raises(ValueError, match="STRATEGY_PAIR_PLATFORM_CHANGED_DURING_FREEZE"):
            _cli(monkeypatch, capsys, "freeze", reference)
        assert not (Path(config["root"]) / "frozen-pilot.json").exists()
    else:
        frozen = _cli(monkeypatch, capsys, "freeze", reference)
        assert frozen["arms_started"] == 0
    assert len(snapshots) == 2
    documents = [pilot.read_bound_json(item) for item in snapshots]
    assert documents[0]["files"] != documents[1]["files"]
    assert documents[0]["business_argv"] == prepared["trials"][0]["argv"]
    assert documents[1]["business_argv"] == prepared["trials"][1]["argv"]
    identity = [platform_identity(item) for item in documents]
    assert identity[0]["bound_input_refs"] != identity[1]["bound_input_refs"]
    assert (identity[0]["platform_files_sha256"] == identity[1]["platform_files_sha256"]) is not change_platform
