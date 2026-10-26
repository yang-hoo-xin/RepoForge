"""Registered external v2 inputs bind only declarations, never credential values.

为什么要有它：正常DSH入口必须保存并重核提供方合同，不能靠旁路CLI或修改旧配置完成实验。
"""
import json
from pathlib import Path

import pytest
import yaml

from scripts import dsh_run_request as bridge
from src.mod_agent import cli
from tests.unit.test_csharp_repository_backend import _profile


@pytest.fixture
def external_configuration(tmp_path):
    repository, source = _profile(tmp_path / "repository")
    profile = tmp_path / "provider.json"
    profile.write_text(json.dumps({"profile_version": "external_openai_responses_profile_v1",
        "base_url": "https://api.openai.com/v1", "model": "controlled-reference", "model_catalog_url": "https://api.openai.com/v1/models",
        "context_window": 65536, "output_budget": 4096, "timeout_seconds": 600,
        "credential_env": "REPOFORGE_TEST_REFERENCE_KEY", "reasoning_effort": "medium"}), encoding="utf-8")
    settings = tmp_path / "settings.yaml"
    settings.write_text(yaml.safe_dump({"llm": {"provider": "openai", "model": "controlled-reference",
        "base_url": "https://api.openai.com/v1", "max_tokens": 4096}}), encoding="utf-8")
    path = tmp_path / "execution.json"
    path.write_text(json.dumps({"arguments": ["--target-project", str(source), "--workspace-root", str(tmp_path / "runs"),
        "--settings", str(settings), "--repository-profile", repository.reference["path"], "--external-patch-profile", str(profile),
        "--api-key-env", "REPOFORGE_TEST_REFERENCE_KEY", "--base-url-env", "REPOFORGE_TEST_REFERENCE_URL",
        "--allow-source-disclosure", "--defer-approval", "--confirm-isolated-write", "--allow-build",
        "--technical-executor", "task_agent", "--task-agent-protocol", "v2", "--constraint-policy", "source_priority_v1",
        "--model-call-budget-policy", "meter_only", "--source-retrieval", "rg_live"]}), encoding="utf-8")
    return path, profile, settings


def test_external_configuration_allocates_real_registered_inputs_without_reading_key(external_configuration, monkeypatch):
    path, profile, settings = external_configuration
    import src.mod_agent.external_patch as external

    original_get = external.os.environ.get

    def environment(name, *args):
        if name in {"REPOFORGE_TEST_REFERENCE_KEY", "REPOFORGE_TEST_REFERENCE_URL", "OPENAI_API_KEY"}:
            pytest.fail("No credential or endpoint environment values are read during registration")
        return original_get(name, *args)

    monkeypatch.setattr(external.os.environ, "get", environment)
    first = bridge.prepare_new_task(execution_config=str(path), description="Repair the public behavior.")
    second = bridge.prepare_new_task(execution_config=str(path), description="Repair the public behavior.")
    document = json.loads(Path(first).read_bytes())
    bound = document["external_profile_binding"]
    assert bound["configuration_ref"] == bridge._file_ref(profile)
    assert bound["settings_ref"] == bridge._file_ref(settings)
    assert bound["credential_values_recorded"] is False
    assert bound["api_key_env"] == "REPOFORGE_TEST_REFERENCE_KEY"
    argv = bridge.build_product_argv(execution_config=first, description="Repair the public behavior.")
    parsed = cli.build_parser().parse_args(argv)
    assert not parsed.local_ollama and parsed.allow_source_disclosure and parsed.task_agent_protocol == "v2"
    assert parsed.external_patch_profile == str(profile)
    assert first != second and not (Path(parsed.workspace_root) / parsed.run_id).exists()
    assert bridge.task_execution_config(execution_config=str(path), run_id=parsed.run_id) == first


@pytest.mark.parametrize("changed", ["profile", "settings", "env_name"])
def test_registered_external_binding_rejects_valid_declaration_drift(external_configuration, changed):
    path, profile, settings = external_configuration
    registered = Path(bridge.prepare_new_task(execution_config=str(path), description="Repair behavior."))
    if changed == "profile":
        data = json.loads(profile.read_bytes())
        data["output_budget"] = 2048
        profile.write_text(json.dumps(data), encoding="utf-8")
    elif changed == "settings":
        data = yaml.safe_load(settings.read_text())
        data["llm"]["max_tokens"] = 2048
        settings.write_text(yaml.safe_dump(data), encoding="utf-8")
    else:
        data = json.loads(registered.read_bytes())
        args = data["arguments"]
        args[args.index("--base-url-env") + 1] = "OTHER_DECLARED_URL"
        registered.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="EXTERNAL_PROFILE_BINDING_CHANGED"):
        bridge._configuration_arguments(str(registered))


@pytest.mark.parametrize("identity_fields", ["both", "--run-id", "--request-id"])
def test_allocated_external_identity_cannot_drop_its_profile_binding(external_configuration, identity_fields):
    path, _profile, _settings = external_configuration
    registered = Path(bridge.prepare_new_task(execution_config=str(path), description="Repair behavior."))
    document = json.loads(registered.read_bytes())
    del document["external_profile_binding"]
    if identity_fields != "both":
        omitted = "--request-id" if identity_fields == "--run-id" else "--run-id"
        offset = document["arguments"].index(omitted)
        del document["arguments"][offset:offset + 2]
    registered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="DSH_EXTERNAL_PROFILE_BINDING_REQUIRED"):
        bridge._configuration_arguments(str(registered))


@pytest.mark.parametrize("changed,error", [("local", "PROVIDER_CONFLICT"), ("disclosure", "REQUIRED_OPTIONS_MISSING"),
    ("provider", "SETTINGS_PROFILE_MISMATCH"), ("endpoint", "EXTERNAL_PROFILE_INVALID"),
    ("credential_name", "CREDENTIAL_ENV_MISMATCH"), ("secret", "EMBEDDED_CREDENTIAL_DENIED")])
def test_external_configuration_rejects_unsupported_or_unbound_inputs(external_configuration, changed, error):
    path, profile, settings = external_configuration
    if changed in {"local", "disclosure", "credential_name"}:
        value = json.loads(path.read_bytes())
        args = value["arguments"]
        if changed == "local":
            args.append("--local-ollama")
        elif changed == "disclosure":
            args.remove("--allow-source-disclosure")
        else:
            args[args.index("--api-key-env") + 1] = "UNBOUND_KEY_ENV"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif changed == "endpoint":
        value = json.loads(profile.read_bytes())
        value["base_url"] = "https://example.invalid/v1"
        profile.write_text(json.dumps(value), encoding="utf-8")
    else:
        value = yaml.safe_load(settings.read_text())
        value["llm"]["provider" if changed == "provider" else "api_key"] = "controlled-secret-not-real"
        settings.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match=error) as failure:
        bridge.prepare_new_task(execution_config=str(path), description="Repair behavior.")
    assert "controlled-secret-not-real" not in str(failure.value)
