"""Pure controller checks, not real model or business success evidence.

为什么要有它：先检验控制配置和阻断条件，不用模型运行替控制脚本排错。
"""
from __future__ import annotations

import json

import pytest

from scripts import run_repository_product_pilot as pilot
from src.mod_agent.domain.full_project_compilation import _ref, _save


def test_absolute_real_b_does_not_require_external_product_readiness():
    assert "codex_executable_ref" not in pilot.configuration_keys({"mode": "absolute_real_B"})
    assert "codex_executable_ref" in pilot.configuration_keys({"mode": "matched_product_pair"})


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
def test_frozen_qualification_uses_exact_original_adapter(tmp_path, monkeypatch, failure):
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
    qualification = {"evaluator_ref": evaluator_ref, "snapshot_ref": {"frozen": True}}
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
