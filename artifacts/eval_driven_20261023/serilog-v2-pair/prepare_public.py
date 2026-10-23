"""Prepare an exposed public Serilog development task; does not run a model.

为什么要有它：公开反例来自题面，平台与经验改动不改变题目或夹带隐藏修法。
"""
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.mod_agent.domain.csharp_repository_backend import (  # noqa: E402
    _IGNORED,
    CSharpRepositoryProfile,
    _snapshot,
)
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402

ROOT = Path("C:/tmp/rf-serilog-v2-pair-main")
OLD = Path("C:/Users/10160/AppData/Local/Temp/rfext1021/controller-v6/protocol.json")


def main():
    old_ref = _ref(OLD)
    old = read_bound_json(old_ref)
    original = Path(old["source_root"])
    problem = Path(old["problem_ref"]["path"])
    if _ref(problem) != old["problem_ref"]:
        raise ValueError("PUBLIC_PROBLEM_CHANGED")
    source_result = read_bound_json(old["source_result_ref"])
    identity_ref = old["source_identities_ref"]
    if "bytes" in identity_ref and Path(identity_ref["path"]).stat().st_size != identity_ref["bytes"]:
        raise ValueError("SOURCE_IDENTITY_SIZE_CHANGED")
    identities = read_bound_json({key: identity_ref[key] for key in ("path", "sha256")})
    if identities["base_commit"] != "3f6d855a9a3b3cfab8f756b739d8120f9f398e04":
        raise ValueError("REGISTERED_BASE_CHANGED")
    for row in identities["files"]:
        if _ref(original / row["path"])["sha256"] != row["sha256"]:
            raise ValueError("REGISTERED_SOURCE_CHANGED:" + row["path"])
    ROOT.mkdir(parents=True, exist_ok=False)
    source = ROOT / "public-source"
    shutil.copytree(original, source, ignore=shutil.ignore_patterns(*_IGNORED))
    public_test = source / "test/Serilog.Tests/ReportedIssue2103Tests.cs"
    shutil.copyfile(Path(__file__).with_name(public_test.name), public_test)
    request = _save(ROOT / "public-request.json", {"schema_version": "public_development_request_v1", "visibility": "public",
        "description": problem.read_text(encoding="utf-8") + "\nFix the reported behavior while preserving applicable existing behavior. Public reproduction tests are included in the test project."})
    profile = _save(ROOT / "repository-profile.json", {"profile_version": "csharp_repository_profile_v1",
        "dotnet_ref": old["dotnet_ref"], "project_path": "Serilog.sln",
        "test_project_path": "test/Serilog.Tests/Serilog.Tests.csproj", "nuget_config_path": "NuGet.config",
        "configuration": "Debug", "allowed_edit_paths": old["allowed_edit_paths"],
        "expected_artifacts": ["src/Serilog/bin/Debug/net8.0/Serilog.dll"],
        "test_command": ["test", "{test_project}", "--framework", "net8.0", "--configuration", "{configuration}",
            "--logger", "trx;LogFileName={report_path}"],
        "test_report": {"format": "trx", "path": "public-net8.trx"}, "timeout_seconds": 600})
    settings = yaml.safe_load((REPO / "config/settings.yaml").read_text(encoding="utf-8"))
    settings["task_memory"] = {"enabled": False, "store_root": None, "max_context_chars": 3000}
    settings["llm"].update(provider="ollama", model="qwen3-coder:30b", base_url="http://127.0.0.1:11434", max_tokens=4096)
    with (ROOT / "settings.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(settings, stream, allow_unicode=True, sort_keys=False)
    fixed = CSharpRepositoryProfile(profile).bind(source)
    result = _save(ROOT / "public-preparation.json", {"schema_version": "exposed_public_task_preparation_v1",
        "task_id": "serilog__serilog-2103-v2-development", "exposure": "exposed", "old_protocol_ref": old_ref,
        "registered_source_result_ref": old["source_result_ref"], "original_problem_ref": old["problem_ref"],
        "public_test_ref": _ref(public_test), "request_ref": request, "profile_ref": profile,
        "source_root": str(source), "source_inputs": _snapshot(source), "fixed_inputs": fixed,
        "settings_ref": _ref(ROOT / "settings.yaml"), "model_calls": 0,
        "source_result_schema": source_result.get("schema_version"),
        "new_question": "Under v2 and diagnostic public feedback, does a conditional investigation strategy alter autonomous decisions and repair compared with the same B without it?",
        "scope": "preparation_only_not_qualification_or_model_success"})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
