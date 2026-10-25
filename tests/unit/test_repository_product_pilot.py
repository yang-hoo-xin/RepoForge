"""Pure controller checks, not real model or business success evidence.

为什么要有它：先检验控制配置和阻断条件，不用模型运行替控制脚本排错。
"""
from __future__ import annotations

import json
from pathlib import Path

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


def _cli(monkeypatch, capsys, mode, reference, arm=None):
    monkeypatch.setattr(pilot.sys, "argv", ["pilot", mode, "--reference", reference["path"],
        "--sha256", reference["sha256"], *(["--arm", arm] if arm else [])])
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

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        now["clock"] += 25
        receipt = {"started": True, "timeout": False, "exit_code": 0}
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
