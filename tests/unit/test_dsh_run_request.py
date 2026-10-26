"""DSH starts the real product owner and stops at its original approval boundary."""

from __future__ import annotations

import hashlib
import io
import json
import socket
import stat
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import dsh_run_request as bridge
from src.mod_agent import cli, local_entry
from tests.unit.test_mod_agent_cli import _formal_coverage_model_response


@pytest.fixture
def execution_config(tmp_path):
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({
        "requirement_projection": 2, "requirement_findings": 2,
        "planning": 1, "initial_patch": 2, "repair": 2,
    }), encoding="utf-8")
    arguments = [
        "--target-project", str(tmp_path / "Example"),
        "--workspace-root", str(tmp_path / "runs"),
        "--settings", str(Path("config/settings.yaml").resolve()),
        "--local-ollama", "--ollama-model", "qwen3:4b-instruct",
        "--ollama-context-window", "32768", "--ollama-timeout-seconds", "600",
        "--task-model-call-limit", "9", "--stage-model-call-limits", str(limits),
        "--model-call-budget-policy", "stage_and_total",
        "--defer-approval", "--confirm-isolated-write", "--allow-build",
    ]
    config = tmp_path / "execution.json"
    config.write_text(json.dumps({"arguments": arguments}), encoding="utf-8")
    return config


def _arguments(config, transform):
    value = json.loads(config.read_text(encoding="utf-8"))
    value["arguments"] = transform(value["arguments"])
    config.write_text(json.dumps(value), encoding="utf-8")


def _call(config, description, monkeypatch):
    monkeypatch.setattr(bridge.sys, "stdin", io.TextIOWrapper(io.BytesIO(description.encode("utf-8"))))
    return bridge.main(["--execution-config", str(config)])


def _dependency_api_config(execution_config):
    from tests.unit.test_dependency_api_config import configuration
    from tests.unit.test_local_entry import _game_folder_binding, _game_folder_dependency

    path, inspector = configuration(execution_config.parent)
    dependency = _game_folder_dependency(execution_config.parent)
    binding = execution_config.parent / "dependency.json"
    binding.write_text(json.dumps(_game_folder_binding(dependency).to_dict()), encoding="utf-8")
    _arguments(execution_config, lambda values: [*values,
        "--game-folder-build-binding", str(binding), "--dependency-api-config", str(path)])
    return path, inspector


@pytest.mark.parametrize("changed", ["config", "inspector"])
def test_prepared_dependency_api_binding_rejects_drift(execution_config, changed):
    path, inspector = _dependency_api_config(execution_config)
    selected = bridge.prepare_new_task(execution_config=str(execution_config), description="change Load")
    document = json.loads(Path(selected).read_text(encoding="utf-8"))
    assert document["dependency_api_binding"]["configuration_ref"] == bridge._file_ref(path)
    assert bridge.build_product_argv(execution_config=selected, description="change Load")
    if changed == "config":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["max_method_inspections"] = 2
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        inspector.write_bytes(b"changed")
    with pytest.raises(ValueError, match="CHANGED"):
        bridge.build_product_argv(execution_config=selected, description="change Load")


def test_dependency_api_context_detects_valid_config_drift(execution_config):
    from scripts.dsh_request_routing import routing_context

    path, _ = _dependency_api_config(execution_config)
    before = routing_context(execution_config=str(execution_config))
    value = json.loads(path.read_text(encoding="utf-8"))
    value["max_method_inspections"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")
    after = routing_context(execution_config=str(execution_config))
    assert before[0] == after[0]
    assert before[1]["dependency_api"] != after[1]["dependency_api"]


def _repository_config(execution_config):
    from tests.unit.test_csharp_repository_backend import _profile

    profile, source = _profile(execution_config.parent / "repository")
    def select(arguments):
        arguments[arguments.index("--target-project") + 1] = str(source)
        return [*arguments, "--repository-profile", profile.reference["path"]]
    _arguments(execution_config, select)
    return profile


@pytest.mark.parametrize("changed", ["configuration", "sdk"])
def test_prepared_repository_profile_is_forwarded_and_rejects_drift(execution_config, changed):
    profile = _repository_config(execution_config)
    prepared = bridge.prepare_new_task(execution_config=str(execution_config), description="Correct the library behavior.")
    document = json.loads(Path(prepared).read_text(encoding="utf-8"))
    assert document["repository_profile_binding"]["configuration_ref"] == profile.reference
    argv = bridge.build_product_argv(execution_config=prepared, description="Correct the library behavior.")
    assert cli.build_parser().parse_args(argv).repository_profile == profile.reference["path"]
    if changed == "configuration":
        path = Path(profile.reference["path"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value["configuration"] = "Debug"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        Path(profile.data["dotnet_ref"]["path"]).write_bytes(b"changed sdk")
    with pytest.raises(ValueError, match="CHANGED"):
        bridge.build_product_argv(execution_config=prepared, description="Correct the library behavior.")


def test_repository_routing_tracks_profile_but_never_initializes_a_game_project(execution_config):
    from scripts.dsh_request_routing import routing_context

    profile = _repository_config(execution_config)
    before = routing_context(execution_config=str(execution_config))
    path = Path(profile.reference["path"])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["configuration"] = "Debug"
    path.write_text(json.dumps(value), encoding="utf-8")
    after = routing_context(execution_config=str(execution_config))
    assert before[0] == after[0]
    assert before[1]["repository_profile"] != after[1]["repository_profile"]
    with pytest.raises(ValueError, match="REQUIRES_EXISTING_REPOSITORY"):
        bridge.prepare_new_task(execution_config=str(execution_config), description="A new game plugin.", project_name="Plugin")
    assert not (execution_config.parent / "runs" / ".dsh-inputs").exists()


def test_normal_entry_dependency_api_reaches_reassessment_owner(execution_config, monkeypatch, capsys):
    _install_execution_boundaries(execution_config.parent, monkeypatch)
    _dependency_api_config(execution_config)
    original = local_entry._make_request_upstream_evidence_factory
    received = []
    def assemble(**kwargs):
        augmentation = kwargs["source_augmentation_factory"]
        assert callable(augmentation)
        assert augmentation.query_capabilities["available"] is True
        assert augmentation.query_capabilities["tools"]["inspect_method_il"]["target_code_execution_allowed"] is False
        received.append(augmentation)
        return original(**kwargs)
    monkeypatch.setattr(local_entry, "_make_request_upstream_evidence_factory", assemble)
    assert _call(execution_config, "make Load record deterministic product diagnostics", monkeypatch) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "awaiting_approval"
    assert len(received) == 1
    manifest = json.loads((Path(result["run_root"]) / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependency_api"]["status"] == "configured"
    assert manifest["dependency_api"]["source_snapshot_sha256"] == manifest["build_dependency"]["game_folder"]["copy_snapshot_sha256"]


def test_literal_stdin_cannot_replace_options_and_each_invocation_is_new(execution_config):
    description = '--allow-source-disclosure\n$(touch stolen); --target-project elsewhere & 不改需求'
    first = cli.build_parser().parse_args(bridge.build_product_argv(
        execution_config=str(execution_config), description=description))
    second = cli.build_parser().parse_args(bridge.build_product_argv(
        execution_config=str(execution_config), description=description))
    assert first.description == second.description == description
    assert first.target_project == str(execution_config.parent / "Example")
    assert first.allow_source_disclosure is False
    assert first.run_id != second.run_id and first.request_id != second.request_id
    assert first.defer_approval and first.local_ollama
    assert first.confirm_isolated_write and first.allow_build


@pytest.mark.parametrize("option", [
    "--run-gtfo-runtime", "--allow-game-launch", "--allow-game-directory-write",
    "--runtime-plugin-destination", "--allow-source-disclosure", "--external-patch-profile",
    "--langfuse-metadata", "--planning-target-evidence-id",
    "--local", "--help", "--", "run-request", "-m",
])
def test_profile_cannot_expand_execution_boundary(execution_config, option):
    _arguments(execution_config, lambda values: [*values, option])
    # 为什么要有它：新增外部选项只属于显式OpenAI v2分支，旧local配置仍必须拒绝混用或缺值。
    expected = {"--allow-source-disclosure": "DSH_RUN_CONFIGURATION_PROVIDER_CONFLICT",
        "--external-patch-profile": "DSH_RUN_CONFIGURATION_VALUE_MISSING"}.get(option, "DSH_RUN_CONFIGURATION_OPTION_NOT_ALLOWED")
    with pytest.raises(ValueError, match=expected):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")


@pytest.mark.parametrize("option", [
    "--defer-approval", "--confirm-isolated-write", "--allow-build", "--local-ollama",
    "--task-model-call-limit", "--stage-model-call-limits",
])
def test_execution_and_budget_options_must_be_explicit(execution_config, option):
    def remove(values):
        offset = values.index(option)
        del values[offset:offset + (1 if option in bridge._SWITCHES else 2)]
        return values
    _arguments(execution_config, remove)
    with pytest.raises(ValueError, match="REQUIRED_OPTIONS_MISSING"):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")


def test_remote_ollama_and_duplicate_options_are_rejected(execution_config):
    _arguments(execution_config, lambda values: [*values, "--ollama-base-url=https://example.com"])
    with pytest.raises(ValueError):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")
    _arguments(execution_config, lambda values: [*values, "--local-ollama"])
    with pytest.raises(ValueError, match="OPTION_NOT_ALLOWED"):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")


@pytest.mark.parametrize("description", ["", "\0", "界" * 10923], ids=["empty", "nul", "utf8-limit"])
def test_request_validation_does_not_enter_product(execution_config, description, monkeypatch, capsys):
    monkeypatch.setattr(cli, "main", lambda *_: pytest.fail("Invalid input reached the product"))
    assert _call(execution_config, description, monkeypatch) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "DSH_INPUT_ERROR"


def test_invalid_stage_budget_is_rejected_before_model_or_build(execution_config):
    (execution_config.parent / "limits.json").write_text(
        json.dumps({"planning": 9}), encoding="utf-8")
    with pytest.raises(ValueError, match="STAGE_LIMITS_INVALID"):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")


def test_fixed_worker_configuration_is_forwarded_to_its_original_owner(execution_config):
    worker = execution_config.parent / "worker.json"
    worker.write_text("{}", encoding="utf-8")
    _arguments(execution_config, lambda values: [*values, "--patch-worker-config", str(worker)])
    parsed = cli.build_parser().parse_args(bridge.build_product_argv(
        execution_config=str(execution_config), description="change Load"))
    assert parsed.patch_worker_config == str(worker)
    worker.unlink()
    with pytest.raises(ValueError, match="WORKER_FILE_MISSING"):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")


def test_explicit_run_layout_is_single_use_and_requires_complete_safe_identity(execution_config):
    _arguments(execution_config, lambda values: [*values, "--run-id", "dsh_bound"])
    with pytest.raises(ValueError, match="IDENTITY_PAIR_REQUIRED"):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")
    _arguments(execution_config, lambda values: [*values, "--request-id", "request_bound"])
    parsed = cli.build_parser().parse_args(bridge.build_product_argv(
        execution_config=str(execution_config), description="change Load"))
    assert parsed.run_id == "dsh_bound" and parsed.request_id == "request_bound"
    (execution_config.parent / "runs" / "dsh_bound").mkdir(parents=True)
    with pytest.raises(ValueError, match="RUN_ALREADY_EXISTS"):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")
    _arguments(execution_config, lambda values: ["../escape" if value == "dsh_bound" else value
                                                 for value in values])
    with pytest.raises(ValueError, match="IDENTITY_INVALID"):
        bridge.build_product_argv(execution_config=str(execution_config), description="change Load")


def test_new_task_allocates_independent_inputs_and_relocates_only_declared_paths(execution_config):
    worker_path = execution_config.parent / "worker-template.json"
    worker = {"behavior_evaluator": {"factory_ref": {"path": "unchanged.py", "sha256": "a" * 64},
              "kwargs": {"source_root": "old/source", "output_root": "old/checks", "preserve": "old/source"}}}
    worker_path.write_text(json.dumps(worker), encoding="utf-8")
    document = json.loads(execution_config.read_text())
    document["arguments"] += ["--run-id", "old", "--request-id", "old_request", "--patch-worker-config", str(worker_path)]
    document["worker_path_bindings"] = [
        {"path": ["behavior_evaluator", "kwargs", "source_root"], "relative_path": "generated_mods/Example"},
        {"path": ["behavior_evaluator", "kwargs", "output_root"], "relative_path": "checks"},
    ]
    execution_config.write_text(json.dumps(document), encoding="utf-8")
    original = execution_config.read_bytes(), worker_path.read_bytes()
    selected = [bridge.prepare_new_task(execution_config=str(execution_config), description="new task") for _ in range(2)]
    parsed = [cli.build_parser().parse_args(bridge.build_product_argv(execution_config=value, description="new task")) for value in selected]
    assert parsed[0].run_id != parsed[1].run_id
    assert parsed[0].request_id != parsed[1].request_id
    for path, args in zip(selected, parsed):
        run_root = Path(args.workspace_root) / args.run_id
        assert not run_root.exists()  # Public owner still performs exclusive mkdir.
        allocated = json.loads(Path(args.patch_worker_config).read_text())
        assert allocated["behavior_evaluator"]["kwargs"]["source_root"] == str(run_root / "generated_mods" / "Example")
        assert allocated["behavior_evaluator"]["kwargs"]["preserve"] == "old/source"
        assert args.task_model_call_limit == 9
        assert bridge.task_execution_config(execution_config=str(execution_config), run_id=args.run_id) == path
    assert (execution_config.read_bytes(), worker_path.read_bytes()) == original


def test_new_task_does_not_guess_evaluator_paths(execution_config):
    worker = execution_config.parent / "worker.json"
    worker.write_text(json.dumps({"behavior_evaluator": {"kwargs": {"source_root": "old"}}}))
    _arguments(execution_config, lambda values: [*values, "--patch-worker-config", str(worker)])
    with pytest.raises(ValueError, match="WORKER_PATH_BINDINGS_REQUIRED"):
        bridge.prepare_new_task(execution_config=str(execution_config), description="new task")


def _install_execution_boundaries(tmp_path, monkeypatch):
    """Replace only provider HTTP and baseline compiler; retain public owner/Graph."""
    target = tmp_path / "Example"
    target.mkdir()
    dependencies = tmp_path / "BepInEx"
    dependencies.mkdir()
    (target / "Example.csproj").write_text(
        "<Project><PropertyGroup><AssemblyName>Example</AssemblyName>"
        f"<BepInExDir>{dependencies}</BepInExDir></PropertyGroup></Project>", encoding="utf-8")
    (target / "Plugin.cs").write_text(
        "namespace Example; public sealed class Plugin { public void Load() {} }\n", encoding="utf-8")
    (target / "nuget.config").write_text("<configuration />", encoding="utf-8")
    roles = []

    def transport(_transport, request):
        assert str(request.url).startswith("http://127.0.0.1:11434/api/")
        if request.method == "GET":
            value = ({"version": "dsh-test"} if request.url.path == "/api/version" else {
                "models": [{"name": "qwen3:4b-instruct", "model": "qwen3:4b-instruct",
                            "digest": "1" * 64, "details": {"context_length": 40960},
                            "capabilities": ["completion", "tools"]}]})
            return httpx.Response(200, request=request, json=value)
        assert request.method == "POST" and request.url.path == "/api/chat"
        payload = json.loads(request.content)
        contexts = []
        for message in payload["messages"]:
            if message["role"] != "user":
                continue
            try:
                context = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if "patch_agent_context_encoding" in context:
                context = local_entry._restore_patch_agent_context(context)
            contexts.append(context)
        context = next(value for value in contexts if value.get("message_type") in {
            "product_requirement_projection", "product_requirement_findings",
        } or "allowed_edit_evidence_ids" in value or "required_change_units" in value)
        content = _formal_coverage_model_response(context)
        if content is not None:
            roles.append(context["message_type"])
        elif "allowed_edit_evidence_ids" in context:
            roles.append("planning")
            content = {"decision": "select_targets",
                       "selected_evidence_ids": context["allowed_edit_evidence_ids"],
                       "rationale": "DSH transport fixture selected the available source evidence.",
                       "missing_evidence": []}
        else:
            roles.append("patch")
            changes = []
            for unit in context["required_change_units"]:
                binding_id = unit["edit_binding_id"]
                before = context["edit_binding_contexts"][binding_id]["source_excerpt"]
                changes.append({"edit_binding_id": binding_id, "rationale": "Test-only edit.",
                                "rollback_hint": "Restore prior bytes.",
                                "after_text": before.replace("{}", "{ /* DSH_TEST_CHANGE */ }", 1)})
            content = {"changes": changes}
        return httpx.Response(200, request=request, json={
            "model": payload["model"], "message": {"role": "assistant", "content": json.dumps(content)},
            "done": True, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 20,
        })

    builds = []

    def build(profile):
        builds.append(profile)
        artifact = profile.working_directory / "bin" / "Debug" / "Example.dll"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"deterministic baseline compilation")
        return local_entry.NuGetIsolatedBuildExecution(
            restore_exit_code=0, build_exit_code=0, restore_stdout="restore ok", restore_stderr="",
            build_stdout="build ok", build_stderr="", duration_ms=1)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", transport)
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("No actual model/network"))
    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", build)
    monkeypatch.setattr(cli, "_read_exact_diff_approval_from_stdin",
                        lambda *_: pytest.fail("DSH must not request or consume an approval"))
    return target, roles, builds


def test_real_cli_owner_graph_reaches_patch_review_and_deferred_approval(
    execution_config, monkeypatch, capsys,
):
    target, roles, builds = _install_execution_boundaries(execution_config.parent, monkeypatch)
    original = (target / "Plugin.cs").read_bytes()
    exit_code = _call(execution_config, "make Load record deterministic product diagnostics", monkeypatch)
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0, result.get("blockers")
    assert result["status"] == "awaiting_approval"
    assert result["graph_invoked"] is True
    assert result["patch_payload_parser_entered"] is True
    assert result["actual_write_performed"] is False
    assert result["canonical_source_unchanged"] is True
    assert roles == ["product_requirement_projection", "product_requirement_findings", "planning", "patch"]
    assert len(builds) == 1  # Baseline compiler only, no approved post-Apply Build.
    assert (target / "Plugin.cs").read_bytes() == original
    assert result["model_call_budget"]["maximum"] == 9
    assert result["model_call_count"] == 4
    envelopes = [json.loads(path.read_text(encoding="utf-8"))
                 for path in (Path(result["run_root"]) / "a").rglob("*.json")]
    types = {value.get("artifact_type") for value in envelopes}
    assert {"implementation_plan", "patch_proposal", "proposal_review", "dry_run_result"} <= types
    assert "isolated_apply_result" not in types
    assert "build_result" not in types
    feature = next(value["data"] for value in envelopes if value.get("artifact_type") == "feature_spec")
    assert "log marker or startup evidence" not in feature["evidence_requirements"]
    assert {"startup_loaded", "diagnostic marker observed"} <= set(feature["runtime_verification_plan"])
    pack = next(value["data"] for value in envelopes if value.get("artifact_type") == "evidence_pack")
    assert pack["can_generate_patch"] is True
    assert pack["requirement_coverage"]["sufficiency"] == {"status": "sufficient", "reasons": []}
    runtime = [row for row in pack["requirement_coverage"]["input_obligations"]
               if row["origin"]["field"] == "runtime_verification_plan"]
    assert runtime and all(row["required_before_patch"] is False for row in runtime)
    plan = next(value["data"] for value in envelopes if value.get("artifact_type") == "implementation_plan")
    assert "diagnostic marker observed" in plan["test_plan"]


def test_real_product_blocked_result_is_not_replaced_with_bridge_success(
    execution_config, monkeypatch, capsys,
):
    _install_execution_boundaries(execution_config.parent, monkeypatch)
    exit_code = _call(execution_config, "test", monkeypatch)
    result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert result["product_execution_result"]["execution_disposition"] == "domain_blocked"
    assert result["product_execution_result"]["terminal_node"] == "blocked"
    assert result.get("status") != "DSH_INPUT_ERROR"


def test_two_new_tasks_use_real_public_graph_and_remain_queryable(execution_config, monkeypatch, capsys):
    target, roles, builds = _install_execution_boundaries(execution_config.parent, monkeypatch)
    results = []
    before = (target / "Plugin.cs").read_bytes()
    for _ in range(2):
        monkeypatch.setattr(bridge.sys, "stdin", io.TextIOWrapper(io.BytesIO(
            b"make Load record deterministic product diagnostics")))
        assert bridge.main(["--execution-config", str(execution_config), "--new-task"]) == 0
        results.append(json.loads(capsys.readouterr().out))
    assert results[0]["run_id"] != results[1]["run_id"]
    assert results[0]["request_id"] != results[1]["request_id"]
    for result in results:
        assert result["status"] == "awaiting_approval" and result["graph_invoked"] is True
        assert result["model_call_budget"]["maximum"] == 9
        record = bridge.inspect_run(execution_config=str(execution_config), run_id=result["run_id"])
        assert Path(record["execution_config_path"]).is_file()
        assert record["result"]["workflow_checkpoint"] == result["workflow_checkpoint"]
    assert {row["run_id"] for row in bridge.list_runs(execution_config=str(execution_config))["records"]} == {
        result["run_id"] for result in results}
    assert (target / "Plugin.cs").read_bytes() == before


def _record_run(config, *, run_id="dsh_recorded", manifest=None, result=None):
    run_root = config.parent / "runs" / run_id
    run_root.mkdir(parents=True)
    for name, content in (("run_manifest.json", manifest), ("run_result.json", result)):
        if content is not None:
            (run_root / name).write_text(json.dumps(content), encoding="utf-8")
    return run_root


def test_inspect_preserves_public_upstream_failure_without_patch_or_new_execution(
    execution_config, monkeypatch, capsys,
):
    run_id = "dsh_recorded"
    manifest = {"run_id": run_id, "model_call_count": 3, "status": "partial"}
    result = {"run_id": run_id, "patch_execution_receipts": [], "blockers": [
        {"code": "SEMANTIC_MISMATCH", "owner": "platform"}], "actual_write_performed": False}
    run_root = _record_run(execution_config, manifest=manifest, result=result)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in run_root.iterdir()}
    _arguments(execution_config, lambda values: [*values, "--run-id", run_id,
                                                "--request-id", "request_recorded"])
    # Execution dependencies may no longer exist; inspection must still work.
    (execution_config.parent / "limits.json").unlink()
    monkeypatch.setattr(bridge.sys, "stdin", object())
    monkeypatch.setattr(cli, "main", lambda *_: pytest.fail("Inspection ran CLI execution"))
    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build",
                        lambda *_: pytest.fail("Inspection ran Build"))
    monkeypatch.setattr(socket.socket, "connect", lambda *_: pytest.fail("Inspection used network"))
    assert bridge.main(["--execution-config", str(execution_config), "--inspect-run", run_id]) == 0
    wrapper = json.loads(capsys.readouterr().out)
    assert wrapper == {"entry": "run-request-status", "read_only": True,
                       "run_id": run_id, "run_root": str(run_root.resolve()),
                       "manifest": manifest, "result": result, "status": "recorded",
                       "model_calls_performed": 0, "execution_config_path": str(execution_config)}
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in run_root.iterdir()} == before
    assert not (run_root / "a").exists()


def test_inspect_distinguishes_not_found_from_recorded_but_unfinished(execution_config):
    missing = bridge.inspect_run(execution_config=str(execution_config), run_id="absent")
    assert missing["status"] == "not_found"
    assert missing["result"] is None and missing["manifest"] is None
    assert not (execution_config.parent / "runs").exists()
    _record_run(execution_config, manifest={"run_id": "dsh_recorded", "status": "admitted"})
    unfinished = bridge.inspect_run(execution_config=str(execution_config), run_id="dsh_recorded")
    assert unfinished["status"] == "recorded"
    assert unfinished["result"] is None
    assert unfinished["manifest"]["status"] == "admitted"


@pytest.mark.parametrize("run_id", ["../escape", "..", "/escape", r"C:\escape", r"nested\escape",
                                    "nested/escape", "name:stream", "--workspace-root=x"])
def test_inspect_rejects_path_syntax(execution_config, run_id):
    with pytest.raises(ValueError, match="INSPECTION_IDENTITY_INVALID"):
        bridge.inspect_run(execution_config=str(execution_config), run_id=run_id)


@pytest.mark.parametrize("link_location", ["run", "record"])
def test_inspect_rejects_directory_and_file_links(execution_config, link_location, monkeypatch):
    outside = execution_config.parent / "outside"
    outside.mkdir()
    (outside / "run_manifest.json").write_text('{"run_id":"dsh_recorded"}', encoding="utf-8")
    workspace = execution_config.parent / "runs"
    workspace.mkdir()
    run_root = workspace / "dsh_recorded"
    link = run_root if link_location == "run" else run_root / "run_manifest.json"
    try:
        if link_location == "run":
            run_root.symlink_to(outside, target_is_directory=True)
        else:
            run_root.mkdir()
            (run_root / "run_manifest.json").symlink_to(outside / "run_manifest.json")
    except OSError:
        # Windows without symlink privilege still exercises the actual reparse
        # guard with an injected lstat result; no outside target is ever opened.
        original_lstat = Path.lstat

        def lstat(path):
            if path == link:
                return SimpleNamespace(st_mode=stat.S_IFDIR if link_location == "run" else stat.S_IFREG,
                                       st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return original_lstat(path)

        monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="LINK_NOT_ALLOWED"):
        bridge.inspect_run(execution_config=str(execution_config), run_id="dsh_recorded")


def test_inspect_rejects_oversized_or_wrong_identity_records(execution_config, monkeypatch):
    run_root = _record_run(execution_config, manifest={"run_id": "different"})
    with pytest.raises(ValueError, match="RECORD_IDENTITY_INVALID"):
        bridge.inspect_run(execution_config=str(execution_config), run_id="dsh_recorded")
    path = run_root / "run_manifest.json"
    path.write_bytes(b"x" * 101)
    monkeypatch.setattr(bridge, "MAX_RUN_RECORD_BYTES", 100)
    with pytest.raises(ValueError, match="RECORD_SIZE_OR_TYPE_INVALID"):
        bridge.inspect_run(execution_config=str(execution_config), run_id="dsh_recorded")
