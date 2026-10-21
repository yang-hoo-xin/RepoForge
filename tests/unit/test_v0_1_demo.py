"""Release-demo contract checks without starting .NET or a model."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_v0_1_demo as demo


def test_resolve_dotnet_prefers_explicit_path(tmp_path, monkeypatch):
    selected = tmp_path / "dotnet"
    selected.write_text("fixture", encoding="utf-8")
    monkeypatch.setenv("REPOFORGE_DOTNET", str(tmp_path / "other"))
    assert demo._resolve_dotnet(str(selected)) == selected.resolve()


def test_resolve_dotnet_rejects_missing_path(tmp_path, monkeypatch):
    monkeypatch.delenv("REPOFORGE_DOTNET", raising=False)
    monkeypatch.setattr(demo.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError, match="REPOFORGE_DOTNET_REQUIRED"):
        demo._resolve_dotnet(None)


def test_run_demo_requires_two_real_integration_results(tmp_path, monkeypatch):
    dotnet = tmp_path / "dotnet"
    dotnet.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(demo, "_git_state", lambda: {"commit_sha": "a" * 40, "clean": True, "porcelain": []})

    def run(*argv, env=None):
        if argv[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="8.0.100\n", stderr="")
        output = Path(argv[argv.index("--junitxml") + 1]).parent
        junit = output / "junit.xml"
        junit.write_text('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0" /></testsuites>', encoding="utf-8")
        workspace = Path(env["TMP"])
        for index in range(2):
            path = workspace / f"rfcscli-{index}" / "integration-result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="2 passed", stderr="")

    monkeypatch.setattr(demo, "_run", run)
    result = demo.run_demo(
        output=tmp_path / "output",
        dotnet=dotnet,
        require_clean=True,
        workspace_parent=tmp_path,
    )
    assert result["status"] == "passed"
    assert result["junit"]["tests"] == 2
    assert len(result["integration_results"]) == 2
    assert (tmp_path / "output" / "result.json").is_file()


def test_run_demo_blocks_dirty_release_snapshot_before_pytest(tmp_path, monkeypatch):
    dotnet = tmp_path / "dotnet"
    dotnet.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(demo, "_git_state", lambda: {"commit_sha": "a" * 40, "clean": False, "porcelain": [" M file"]})
    monkeypatch.setattr(demo, "_run", lambda *argv, env=None: SimpleNamespace(returncode=0, stdout="8.0.100\n", stderr=""))
    with pytest.raises(ValueError, match="RELEASE_DEMO_REQUIRES_CLEAN_GIT_STATE"):
        demo.run_demo(
            output=tmp_path / "output",
            dotnet=dotnet,
            require_clean=True,
            workspace_parent=tmp_path,
        )
