"""Session-produced candidate -> real compiler checker, with only compiler subprocesses replaced.

No real model, SDK process, game or successful real compilation is claimed here.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.build.nuget import _run_nuget_isolated_build
from src.mod_agent.domain.full_project_compilation import FullProjectCompilationChecker, _ref
from src.mod_agent.model_qualification import canonical_hash
from tests.unit.test_task_agent_loop import call, response
from tests.unit.test_task_agent_tools import action, patch, save
from tests.unit.test_task_agent_tools import setup as setup


def submitted(session):
    arguments = patch(session)
    value = json.loads(arguments["patch_json"])
    value["changes"][0]["rationale"] = "原始工具参数中的中文与🙂，不是伪造provider正文"
    arguments["patch_json"] = json.dumps(value, ensure_ascii=False)
    session.response_admission(response(call("candidate_submit", arguments, "unicode-submit")))
    owner = action(session, "candidate_submit")
    result = owner.execute(owner.prepare(arguments).value)
    return result["candidate_id"], arguments


def compiler_checker(inputs, tmp_path):
    project_root = inputs.workspace / "generated_mods/SyntheticMod"
    project = project_root / "SyntheticMod.csproj"
    project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net7.0</TargetFramework></PropertyGroup></Project>', encoding="utf-8")
    nuget = project_root / "NuGet.Config"
    nuget.write_text("<configuration><packageSources><clear /></packageSources></configuration>", encoding="utf-8")
    commands = []

    def compiler_subprocess(argv, **kwargs):
        # Scope: only the OS restore/build command is simulated; all checker admission/copy/readback is production code.
        # 为什么要有它：旧Python oracle没有消费真实编译器所需的响应字节元数据，必须覆盖真正调用链而非再次替换验收器。
        commands.append((tuple(argv), {"cwd": str(kwargs["cwd"]), "timeout": kwargs.get("timeout")}))
        if "build" in argv:
            manifest = next(value.split("=", 1)[1] for value in argv if value.startswith("-p:ControlledCompileInputsFile="))
            current = Path(kwargs["cwd"])
            Path(manifest).write_text("\n".join(str(path) for path in current.rglob("*.cs")), encoding="utf-8")
            assembly = current / "bin/Debug/net7.0/SyntheticMod.dll"
            assembly.parent.mkdir(parents=True)
            assembly.write_bytes(b"controlled compiler subprocess output, not a real DLL")
        return SimpleNamespace(returncode=0, stdout="controlled subprocess", stderr="")

    checker = FullProjectCompilationChecker(source_root=project_root, workspace_root=inputs.workspace,
        project_relative_path=project.name, artifact_store_root=inputs.store.root, dependency_binding=None,
        nuget_config_ref=_ref(nuget), output_root=tmp_path / "checker", expected_artifact="bin/Debug/net7.0/SyntheticMod.dll",
        build_runner=lambda profile: _run_nuget_isolated_build(profile, command_runner=compiler_subprocess))
    return checker, commands


def test_session_metadata_reaches_real_full_project_checker_and_preserves_tool_origin(setup, tmp_path):
    inputs, session, _oracle, _budget = setup
    checker, commands = compiler_checker(inputs, tmp_path)
    candidate_id, arguments = submitted(session)
    candidate = session._candidate(candidate_id)
    context = {key: deepcopy(candidate[key]) for key in ("generated_changes", "generated_changes_sha256", "model_response",
        "source_snapshot_sha256", "materialized_code_identity")}
    context.update(implementation_plan=deepcopy(session.documents.implementation_plan),
        implementation_plan_ref=asdict(session.context.implementation_plan_ref))
    result = checker(context)
    assert result["status"] == "passed" and result["changed_compilation_performed"] is True
    assert len(commands) == 4  # baseline restore/build + changed restore/build, all simulated subprocesses.
    assert result["baseline"]["all_required_files_compiled"] is True
    assert result["changed"]["all_required_files_compiled"] is True
    raw = Path(candidate["model_origin"]["derived_content_ref"]["path"]).read_bytes()
    assert candidate["model_response"]["output_utf8_bytes"] == len(raw) > len(arguments["patch_json"])
    assert candidate["model_response"]["output_sha256"] == hashlib.sha256(raw).hexdigest()
    assert raw == arguments["patch_json"].encode("utf-8")
    source_response = json.loads(Path(candidate["model_origin"]["response_ref"]["path"]).read_bytes())
    assert source_response["content"] == ""
    assert source_response["tool_calls"][0]["arguments"] == arguments


@pytest.mark.parametrize("changed", ["missing", "characters", "wrong", "string", "path", "hash"])
def test_session_rejects_self_consistent_candidate_with_wrong_response_byte_metadata_before_check(setup, changed):
    inputs, session, evaluator, _budget = setup
    candidate_id, arguments = submitted(session)
    candidate = deepcopy(session.candidates[candidate_id]["data"])
    if changed == "missing":
        candidate["model_response"].pop("output_utf8_bytes", None)
    elif changed == "characters":
        candidate["model_response"]["output_utf8_bytes"] = len(arguments["patch_json"])
    elif changed == "wrong":
        candidate["model_response"]["output_utf8_bytes"] = 1
    elif changed == "string":
        candidate["model_response"]["output_utf8_bytes"] = str(len(arguments["patch_json"].encode("utf-8")))
    elif changed == "path":
        candidate["model_response"]["path"] = str(inputs.workspace / "unrelated.json")
    else:
        candidate["model_response"]["output_sha256"] = "0" * 64
    candidate.pop("candidate_id")
    changed_id = "candidate-" + canonical_hash(candidate)[:20]
    candidate["candidate_id"] = changed_id
    session.candidates[changed_id] = {"data": candidate, "ref": save(inputs.store.root / "changed-candidate", candidate)}
    session.current_candidate = changed_id
    with pytest.raises(ValueError, match="MODEL_OUTPUT_BINDING_CHANGED"):
        action(session, "candidate_check").prepare({"candidate_id": changed_id})
    assert evaluator.calls == []
