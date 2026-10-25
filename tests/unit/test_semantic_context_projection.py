"""Semantic citations describe the final sent projection, with unchanged action routing.

为什么要有它：目录必须跟随容量裁剪变化，且正常配置入口须真实选择受测机制而非依赖隐藏环境开关。
"""
import json
from dataclasses import asdict

import pytest

from scripts import dsh_run_request
from src.mod_agent import cli
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_v2_context import project_history
from src.mod_agent.task_agent_v2_semantic_state import with_semantic_evidence
from tests.unit.test_dsh_run_request import execution_config as execution_config  # noqa: F401
from tests.unit.test_task_agent_v2_context import configured_request, current_state
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def _value(document, locator):
    for part in locator:
        document = document[part]
    return document


def test_catalog_is_rebuilt_after_capacity_removes_an_old_source(state):
    measurements = []

    class Provider:
        model = "controlled"
        context_window = 65536
        max_output_tokens = 4096
        seed = 0

        def _context_admission(self, payload):
            host = next(json.loads(row["content"])["current_task_state"]
                for row in payload["messages"] if row["role"] == "user"
                and '"current_task_state"' in row["content"])
            assert "semantic_evidence_catalog" in host
            measurements.append(host["semantic_evidence_catalog"])
            return {"admitted": not any("o" * 1000 in row["content"] for row in payload["messages"]),
                "mode": "controlled_projection_pressure"}

    original = configured_request()
    projected, facts = project_history(original, state, provider=Provider(), decorate=with_semantic_evidence)
    assert facts["archived_exchange_count"] == 1 and len(measurements) >= 3
    catalog = current_state(projected)["semantic_evidence_catalog"]
    assert catalog == current_state(with_semantic_evidence(projected))["semantic_evidence_catalog"]
    assert all(row["label"] != "source_read:large-old" for row in catalog)
    assert any(row["label"] == "source_read:active" for row in catalog)
    document = asdict(projected)
    assert all(canonical_hash(_value(document, row["locator"])) == row["value_sha256"] for row in catalog)
    assert projected.tool_results == original.tool_results
    assert projected.provider_state == original.provider_state


def test_byte_budget_test_hook_also_describes_only_retained_values(state):
    projected, facts = project_history(configured_request(), state, history_bytes=100,
        decorate=with_semantic_evidence)
    assert facts["archived_exchange_count"] == 1
    assert all(row["label"] != "source_read:large-old"
        for row in current_state(projected)["semantic_evidence_catalog"])


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_normal_cli_passes_the_declared_semantic_mode(monkeypatch, tmp_path, capsys, mode):
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return {"status": "awaiting_approval"}

    monkeypatch.setattr(cli, "run_local_request", capture)
    result = cli.main(["run-request", "repair the behavior", "--target-project", str(tmp_path / "source"),
        "--workspace-root", str(tmp_path / "runs"), "--technical-executor", "task_agent",
        "--task-agent-protocol", "v2", "--task-agent-semantic-state", mode,
        "--task-model-call-limit", "5"])
    assert result == 0 and calls[0]["task_agent_semantic_state"] == mode
    capsys.readouterr()


def test_registered_dsh_configuration_preserves_the_semantic_mode(execution_config):
    document = json.loads(execution_config.read_text(encoding="utf-8"))
    document["arguments"] += ["--task-agent-protocol", "v2", "--task-agent-semantic-state", "disabled"]
    execution_config.write_text(json.dumps(document), encoding="utf-8")
    arguments, _bindings = dsh_run_request._configuration_arguments(str(execution_config))
    assert arguments[arguments.index("--task-agent-semantic-state") + 1] == "disabled"
