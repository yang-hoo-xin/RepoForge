"""Neutral project preparation reaches the existing owner without business scaffolding."""
from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

import pytest

from src.mod_agent import new_mod_project as owner
from src.mod_agent.local_entry import _binary_tree_snapshot, _binary_tree_snapshot_sha256


def _ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _template_line_endings(tmp_path, monkeypatch, newline):
    templates = tmp_path / "line-ending-templates"
    shutil.copytree(owner.TEMPLATE_ROOT, templates)
    template = templates / "Plugin.cs.template"
    template_bytes = template.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline)
    template.write_bytes(template_bytes)
    monkeypatch.setattr(owner, "TEMPLATE_ROOT", templates)
    return template_bytes


@pytest.fixture
def inputs(tmp_path):
    dependency = tmp_path / "dependencies"
    for name in owner._REQUIRED_DEPENDENCIES:
        path = dependency / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"metadata-fixture-" + name.encode("utf-8"))
    snapshot = _binary_tree_snapshot(dependency)
    binding = tmp_path / "dependency.json"
    binding.write_text(json.dumps({
        "binding_version": "run_local_game_folder_binding_v1", "source_root": str(dependency),
        "snapshot_sha256": _binary_tree_snapshot_sha256(snapshot), "file_count": len(snapshot),
        "property_name": "GameFolder",
    }), encoding="utf-8")
    return {"workspace_root": tmp_path / "inputs", "project_id": "dsh_one",
            "display_name": 'Neutral "Plugin" 测试', "dependency_binding_ref": _ref(binding)}


def test_read_only_preparation_then_exclusive_initialization_and_restart(inputs):
    prepared = owner.prepare_new_mod_project(**inputs)
    assert not inputs["workspace_root"].exists()
    receipt = owner.initialize_new_mod_project(prepared)
    target = Path(receipt["target_project"])
    assert target.is_relative_to(inputs["workspace_root"])
    assert len(list(target.glob("*.csproj"))) == 1
    project = ElementTree.parse(receipt["project_file"]).getroot()
    assert project.findtext("PropertyGroup/AssemblyName") == receipt["identity"]["assembly_name"]
    assert project.findtext("PropertyGroup/Version") == "0.1.0"
    assert receipt["identity"]["config_file_name"] == receipt["identity"]["plugin_guid"] + ".cfg"
    assert receipt["config_identity_status"] == "reserved_not_written"
    assert receipt["model_calls_performed"] == receipt["builds_performed"] == 0
    assert receipt["business_logic_generated"] is False
    before = {p: p.read_bytes() for p in target.parent.rglob("*") if p.is_file()}
    assert owner.initialize_new_mod_project(prepared) == receipt
    assert owner.validate_new_mod_project_receipt(_ref(Path(receipt["receipt_path"])),
        workspace_root=inputs["workspace_root"], expected_scope_hash=receipt["scope_hash"]) == receipt
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_empty_plugin_contains_only_baseplugin_entry_and_unique_identities(inputs, tmp_path, monkeypatch, newline):
    template_bytes = _template_line_endings(tmp_path, monkeypatch, newline)
    first = owner.prepare_new_mod_project(**inputs)
    second = owner.prepare_new_mod_project(**{**inputs, "project_id": "dsh_two"})
    for key in ("assembly_name", "root_namespace", "plugin_guid", "config_file_name"):
        assert first["identity"][key] != second["identity"][key]
    for plan in (first, second):
        plugin = next(x["content"] for x in plan["source_files"] if x["path"] == "Plugin.cs")
        # 为什么要有它：空入口是语义要求；模板原始换行仍须保留，不能为了断言统一生成源码字节。
        assert plugin[plugin.index("public override void Load()"):].split("}", 1)[0].strip().splitlines() == [
            "public override void Load()", "        {"]
        assert plugin.encode("utf-8").count(b"\r\n") == template_bytes.count(b"\r\n")
        assert plugin.encode("utf-8").count(b"\n") == template_bytes.count(b"\n")
        assert plugin.count("(") == 2  # Plugin identity attribute and the empty entry method.
        assert len(plan["source_files"]) == 4
        assert not any(row["path"].endswith(".cfg") for row in plan["source_files"])


@pytest.mark.parametrize("project_id", ["../escape", "CON", "aux.json", "x:y", "foo/Bar", "end."])
def test_invalid_identity_has_no_writes(inputs, project_id):
    with pytest.raises(ValueError, match="NEW_MOD_PROJECT_ID_INVALID"):
        owner.prepare_new_mod_project(**{**inputs, "project_id": project_id})
    assert not inputs["workspace_root"].exists()


def test_dependency_and_template_drift_stop_before_materialization(inputs, monkeypatch, tmp_path):
    prepared = owner.prepare_new_mod_project(**inputs)
    root = Path(json.loads(Path(inputs["dependency_binding_ref"]["path"]).read_text())["source_root"])
    path = root / owner._REQUIRED_DEPENDENCIES[0]
    original = path.read_bytes()
    path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="DEPENDENCY_SNAPSHOT_CHANGED"):
        owner.initialize_new_mod_project(prepared)
    path.write_bytes(original)
    templates = tmp_path / "templates"
    templates.mkdir()
    for template in owner.TEMPLATE_ROOT.iterdir():
        (templates / template.name).write_bytes(template.read_bytes())
    monkeypatch.setattr(owner, "TEMPLATE_ROOT", templates)
    with pytest.raises(ValueError, match="PREPARATION_CHANGED"):
        owner.initialize_new_mod_project(prepared)
    assert not inputs["workspace_root"].exists()


def test_recovery_rejects_source_drift_extra_project_and_changed_receipt(inputs):
    prepared = owner.prepare_new_mod_project(**inputs)
    receipt = owner.initialize_new_mod_project(prepared)
    path = Path(receipt["receipt_path"])
    ref = _ref(path)
    target = Path(receipt["target_project"])
    extra = target / "Other.csproj"
    extra.write_text("<Project />", encoding="utf-8")
    with pytest.raises(ValueError, match="NEW_MOD_SOURCE_CHANGED"):
        owner.validate_new_mod_project_receipt(ref, workspace_root=inputs["workspace_root"],
                                              expected_scope_hash=prepared["scope_hash"])

    extra.unlink()
    plugin = target / "Plugin.cs"
    plugin.write_bytes(plugin.read_bytes() + b"// edited\n")
    with pytest.raises(ValueError, match="NEW_MOD_SOURCE_CHANGED"):
        owner.initialize_new_mod_project(prepared)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="NEW_MOD_REFERENCE_CHANGED"):
        owner.validate_new_mod_project_receipt(ref, workspace_root=inputs["workspace_root"],
                                              expected_scope_hash=prepared["scope_hash"])


@pytest.mark.parametrize("damage", [None, "historical_template", "source"])
def test_receipt_recovery_binds_historical_templates_after_owner_moves(inputs, monkeypatch, tmp_path, damage):
    historical = tmp_path / "frozen_product" / "neutral"
    relocated = tmp_path / "current_product" / "neutral"
    shutil.copytree(owner.TEMPLATE_ROOT, historical)
    shutil.copytree(owner.TEMPLATE_ROOT, relocated)
    monkeypatch.setattr(owner, "TEMPLATE_ROOT", historical)
    prepared = owner.prepare_new_mod_project(**inputs)
    receipt = owner.initialize_new_mod_project(prepared)
    ref = _ref(Path(receipt["receipt_path"]))
    monkeypatch.setattr(owner, "TEMPLATE_ROOT", relocated)
    if damage == "historical_template":
        (historical / "Plugin.cs.template").write_bytes(b"changed historical template")
    elif damage == "source":
        (Path(receipt["target_project"]) / "Plugin.cs").write_bytes(b"changed initialized source")
    if damage is not None:
        with pytest.raises(ValueError, match="TEMPLATE_REFERENCE_CHANGED" if damage == "historical_template" else "SOURCE_CHANGED"):
            owner.validate_new_mod_project_receipt(ref, workspace_root=inputs["workspace_root"],
                                                  expected_scope_hash=prepared["scope_hash"])
    else:
        recovered = owner.validate_new_mod_project_receipt(ref, workspace_root=inputs["workspace_root"],
                                                          expected_scope_hash=prepared["scope_hash"])
        assert recovered == receipt
        assert all(Path(row["path"]).parent == historical for row in recovered["template_refs"])
        new_plan = owner.prepare_new_mod_project(**{**inputs, "project_id": "after_owner_move"})
        assert all(Path(row["path"]).parent == relocated for row in new_plan["template_refs"])


def test_initialization_never_overwrites_unowned_or_partial_directory(inputs):
    plan = owner.prepare_new_mod_project(**inputs)
    container = Path(plan["target_project"]).parent
    container.mkdir(parents=True)
    marker = container / "user.txt"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        owner.initialize_new_mod_project(plan)
    assert marker.read_bytes() == b"preserve"
    assert not Path(plan["target_project"]).exists()


def test_forged_plan_cannot_inject_code_or_destination(inputs):
    plan = owner.prepare_new_mod_project(**inputs)
    forged = deepcopy(plan)
    forged["source_files"][0]["content"] = "execute something else"
    forged["scope_hash"] = owner._digest({k: v for k, v in forged.items() if k != "scope_hash"})
    with pytest.raises(ValueError, match="PREPARATION_CHANGED"):
        owner.initialize_new_mod_project(forged)
    assert not inputs["workspace_root"].exists()


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_initialized_project_enters_original_cli_graph_and_stops_at_diff_review(
    inputs, tmp_path, monkeypatch, capsys, newline,
):
    import httpx

    from src.mod_agent import cli, local_entry
    from tests.unit.test_dsh_run_request import _install_execution_boundaries

    _template_line_endings(tmp_path, monkeypatch, newline)
    # The existing-project fixture stays present and byte-identical throughout.
    existing, roles, _ = _install_execution_boundaries(tmp_path, monkeypatch)
    old_bytes = {p: p.read_bytes() for p in existing.iterdir()}
    plan = owner.prepare_new_mod_project(**inputs)
    receipt = owner.initialize_new_mod_project(plan)
    target = Path(receipt["target_project"])
    original = (target / "Plugin.cs").read_bytes()
    transport = httpx.HTTPTransport.handle_request

    def fixture_transport(client, request):
        response = transport(client, request)
        if request.method == "POST" and roles[-1] == "patch":
            native = json.loads(request.content)
            for message in native["messages"]:
                if message["role"] != "user":
                    continue
                try:
                    context = json.loads(message["content"])
                except ValueError:
                    continue
                if "patch_agent_context_encoding" in context:
                    context = local_entry._restore_patch_agent_context(context)
                if "required_change_units" not in context:
                    continue
                body = response.json()
                generated = json.loads(body["message"]["content"])
                for change in generated["changes"]:
                    before = context["edit_binding_contexts"][change["edit_binding_id"]]["source_excerpt"]
                    # 为什么要有它：受控补丁必须实际改变当前前像，并保留它的换行，不能在CRLF工程返回无效原文。
                    newline = "\r\n" if "\r\n" in before else "\n"
                    empty_body = "{" + newline + "        }"
                    assert empty_body in before
                    change["after_text"] = before.replace(empty_body,
                        "{" + newline + "            // FIXTURE_ONLY" + newline + "        }", 1)
                    assert change["after_text"] != before
                body["message"]["content"] = json.dumps(generated)
                return httpx.Response(200, request=request, json=body)
        return response

    def baseline(profile):
        # No compiler is launched; this verifies source preparation and original routes only.
        artifact = profile.working_directory / "bin" / "Debug" / "net6.0" / (
            receipt["identity"]["assembly_name"] + ".dll")
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"offline-baseline-fixture")
        return local_entry.NuGetIsolatedBuildExecution(
            restore_exit_code=0, build_exit_code=0, restore_stdout="fixture", restore_stderr="",
            build_stdout="fixture", build_stderr="", duration_ms=0)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fixture_transport)
    monkeypatch.setattr(local_entry, "_run_nuget_isolated_build", baseline)
    code = cli.main(["run-request", "make Load record deterministic product diagnostics",
        "--target-project", str(target), "--workspace-root", str(tmp_path / "runs"),
        "--settings", str(Path("config/settings.yaml").resolve()),
        "--game-folder-build-binding", inputs["dependency_binding_ref"]["path"],
        "--local-ollama", "--ollama-model", "qwen3:4b-instruct", "--ollama-context-window", "32768",
        "--task-model-call-limit", "9", "--defer-approval", "--confirm-isolated-write", "--allow-build"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    assert result["status"] == "awaiting_approval"
    assert result["graph_invoked"] is True
    assert result["actual_write_performed"] is False
    assert roles == ["product_requirement_projection", "product_requirement_findings", "planning", "patch"]
    assert (target / "Plugin.cs").read_bytes() == original
    assert all(path.read_bytes() == raw for path, raw in old_bytes.items())
