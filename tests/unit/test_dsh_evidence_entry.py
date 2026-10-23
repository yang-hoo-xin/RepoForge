"""The evidence CLI resolves business identities without dispatching execution."""

import io
import json
import sys
from types import ModuleType

import pytest

from scripts import dsh_run_request as bridge
from src.observability.dashboard.services.trace_service import TraceService


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "execution.json"
    path.write_text(json.dumps({"arguments": ["--workspace-root", str(tmp_path / "runs")]}), encoding="utf-8")
    return path


def test_evidence_json_uses_shared_service_and_writes_nothing(config, monkeypatch, capsys):
    received = []
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    def query(self, **kwargs):
        received.append(kwargs)
        return {"run_id": "business_1", "status": "incomplete", "read_only": True, "gaps": ["missing response"]}
    monkeypatch.setattr(TraceService, "query_run_evidence", query, raising=False)
    monkeypatch.setattr(bridge.cli, "main", lambda *a: pytest.fail("execution dispatched"))
    monkeypatch.setattr(bridge, "advance_task", lambda **k: pytest.fail("recovery dispatched"))
    monkeypatch.setattr(bridge.sys, "stdin", None)
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "business_1", "--view", "evidence"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "incomplete"
    assert received == [{"allowed_roots": [str(config.parent / "runs")], "run_id": "business_1"}]
    assert result["inspection"]["argv"][-6:] == ["--inspect-run", "business_1", "--view", "evidence", "--format", "json"]
    assert before == {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}


def test_evidence_markdown_uses_shared_renderer(config, monkeypatch, capsys):
    monkeypatch.setattr(TraceService, "query_run_evidence", lambda *a, **k: {"run_id": "one"}, raising=False)
    renderer = ModuleType("src.observability.dashboard.services.run_evidence")
    renderer.render_evidence_markdown = lambda report: "# " + report["run_id"]
    monkeypatch.setitem(sys.modules, renderer.__name__, renderer)
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence", "--format", "markdown"]) == 0
    assert capsys.readouterr().out == "# one\n"


@pytest.mark.parametrize("extra", [
    ["--new-task"], ["--route-request"], ["--route-preview"], ["--advance-task", "one"],
    ["--start-project", "one"], ["--preview-project", "one"], ["--initialize-project-name", "one"],
    ["--prepare-findings-continuation", "one"], ["--execute-findings-continuation", "one"],
    ["--findings-allocation-config", "one"], ["--decision-stdin"], ["--list-runs"],
    ["--active-run-id", "one"], ["--conversation-id", "one"], ["--clarification-for", "one"],
    ["--routing-turn-id", "one"], ["--followup-delivery-root", "one"], ["--parent-execution-config", "one"],
])
def test_evidence_rejects_mixed_modes_before_config_or_dispatch(extra, monkeypatch, capsys):
    monkeypatch.setattr(bridge, "_configuration_arguments", lambda *a: pytest.fail("config read before conflict"))
    monkeypatch.setattr(bridge.cli, "main", lambda *a: pytest.fail("execution dispatched"))
    assert bridge.main(["--execution-config", "absent", "--inspect-run", "one", "--view", "evidence", *extra]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "DSH_RUN_INSPECTION_OPERATION_CONFLICT"


@pytest.mark.parametrize("extra", [["--view", "evidence"], ["--format", "json"], ["--inspect-run", "one", "--format", "markdown"]])
def test_new_format_and_view_require_evidence_inspection(config, extra, capsys):
    assert bridge.main(["--execution-config", str(config), *extra]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "DSH_RUN_EVIDENCE_INSPECTION_REQUIRED"


@pytest.mark.parametrize("identity", ["../escape", "other"])
def test_evidence_rejects_invalid_or_conflicting_identity(config, identity, monkeypatch, capsys):
    config.write_text(json.dumps({"arguments": ["--workspace-root", str(config.parent), "--run-id", "bound"]}), encoding="utf-8")
    monkeypatch.setattr(TraceService, "query_run_evidence", lambda *a, **k: pytest.fail("query dispatched"), raising=False)
    assert bridge.main(["--execution-config", str(config), "--inspect-run", identity, "--view", "evidence"]) == 2
    assert "IDENTITY" in json.loads(capsys.readouterr().out)["reason"]


def test_evidence_requires_absolute_config(capsys):
    assert bridge.main(["--execution-config", "relative.json", "--inspect-run", "one", "--view", "evidence"]) == 2
    assert "MUST_BE_ABSOLUTE" in json.loads(capsys.readouterr().out)["reason"]


def test_new_run_output_preserves_product_outcome_and_returns_inspection(config, monkeypatch, capsys):
    def execute(argv):
        print(json.dumps({"run_id": "new_run", "status": "unknown", "cost": 123}))
        return 2
    monkeypatch.setattr(bridge.cli, "main", execute)
    monkeypatch.setattr(bridge, "build_product_argv", lambda **k: ["run-request"])
    monkeypatch.setattr(bridge.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"request")))
    assert bridge.main(["--execution-config", str(config)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unknown" and result["cost"] == 123
    assert result["inspection"]["run_id"] == "new_run"
    assert result["inspection"]["execution_config_path"] == str(config)


def test_routed_child_and_advance_locator_use_registered_child_config(config):
    workspace = config.parent / "runs"
    target = config.parent / "project"
    config.write_text(json.dumps({"arguments": ["--workspace-root", str(workspace), "--target-project", str(target),
        "--run-id", "parent", "--request-id", "request-parent"]}), encoding="utf-8")
    child = workspace / ".dsh-inputs" / "child" / "execution.json"
    child.parent.mkdir(parents=True)
    child.write_text(json.dumps({"arguments": ["--workspace-root", str(workspace), "--target-project", str(target),
        "--run-id", "child", "--request-id", "request-child"]}), encoding="utf-8")
    before = {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}
    direct = bridge._with_inspection({"run_id": "child", "status": "unknown"}, str(config))
    routed = bridge._with_inspection({"status": "unknown", "product_result": {"run_id": "child", "status": "unknown"}}, str(config))
    assert direct["inspection"]["execution_config_path"] == str(child)
    assert routed["product_result"]["inspection"] == direct["inspection"]
    assert direct["status"] == routed["status"] == "unknown"
    assert before == {p: p.read_bytes() for p in config.parent.rglob("*") if p.is_file()}


def test_conflicting_missing_child_config_never_emits_unusable_command(config):
    config.write_text(json.dumps({"arguments": ["--workspace-root", str(config.parent / "runs"), "--run-id", "parent"]}), encoding="utf-8")
    result = bridge._with_inspection({"run_id": "child", "status": "delivered"}, str(config))
    assert result["status"] == "delivered"
    assert result["inspection"]["status"] == "unavailable" and "argv" not in result["inspection"]
