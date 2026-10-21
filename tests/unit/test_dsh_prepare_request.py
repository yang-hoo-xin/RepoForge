"""DSH intake uses the original planning result and Artifact lineage."""

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dsh_prepare_request import prepare_request
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.cli import plan_request


@pytest.fixture
def configuration(tmp_path):
    workspace = tmp_path / "source"
    workspace.mkdir()
    (workspace / "keep.cs").write_text("// unrelated source\n", encoding="utf-8")
    store = tmp_path / "store"
    store.mkdir()
    return {"workspace_root": str(workspace), "artifact_store_root": str(store),
            "target_mod": "OwnedConfigService"}


def test_original_planning_results_are_persisted_with_existing_lineage(configuration, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("No network in intake"))
    description = "Validate configuration maxRetries values from 0 through 5 and preserve unrelated keys."
    expected = plan_request(description, target_mod=configuration["target_mod"])
    before = (Path(configuration["workspace_root"]) / "keep.cs").read_bytes()
    result = prepare_request(description=description, **configuration)
    assert result["request_planning_report"] == expected.request_planning_report
    assert result["mod_planning_brief"] == expected.mod_planning_brief
    assert result["stopped_before"] == "feature_spec"
    assert result["model_calls_performed"] == 0
    assert result["implementation_plan_generated"] is False
    assert (Path(configuration["workspace_root"]) / "keep.cs").read_bytes() == before
    store = ArtifactStore(configuration["artifact_store_root"])
    request_ref = ArtifactRef(**result["artifacts"]["request_planning_report"])
    brief_ref = ArtifactRef(**result["artifacts"]["mod_planning_brief"])
    assert store.read(request_ref)["data"] == expected.request_planning_report
    brief = store.read(brief_ref)
    assert brief["inputs"]["upstream_artifacts"][0]["artifact_hash"] == request_ref.artifact_hash
    assert not list(Path(configuration["artifact_store_root"]).rglob("feature_spec.json"))


def test_repeated_submission_reuses_existing_artifact_identity(configuration):
    description = "Change configuration retry default and keep every unrelated setting."
    first = prepare_request(description=description, **configuration)
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in Path(configuration["artifact_store_root"]).rglob("*.json")}
    second = prepare_request(description=description, **configuration)
    assert first == second
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes} == hashes


@pytest.mark.parametrize(("description", "must_have_gap"), [
    ("help", True),
    ("Change enemy damage through a gameplay hook.", False),
])
def test_ambiguity_and_evidence_gaps_remain_original_owner_results(
    configuration, description, must_have_gap,
):
    result = prepare_request(description=description, **configuration)
    expected = plan_request(description, target_mod=configuration["target_mod"])
    assert result["status"] == expected.status
    assert result["mod_planning_brief"]["blockers"] == expected.mod_planning_brief["blockers"]
    assert result["request_planning_report"]["uncertainties"] == expected.request_planning_report["uncertainties"]
    has_gap = bool(result["mod_planning_brief"]["blockers"] or result["request_planning_report"]["uncertainties"])
    assert has_gap is must_have_gap


@pytest.mark.parametrize("description", ["", "\0", "x" * 32769], ids=["empty", "nul", "oversized"])
def test_invalid_request_writes_nothing(configuration, description):
    with pytest.raises(ValueError, match="SUBMIT_REQUEST_INVALID"):
        prepare_request(description=description, **configuration)
    assert not list(Path(configuration["artifact_store_root"]).iterdir())


def test_real_process_treats_request_flags_and_shell_text_as_plain_stdin(configuration):
    description = '--artifact-store-root=elsewhere\nChange configuration; $(ignore-me) & preserve JSON keys.'
    process = subprocess.run(
        [sys.executable, "-B", "-m", "scripts.dsh_prepare_request",
         "--workspace-root=" + configuration["workspace_root"],
         "--artifact-store-root=" + configuration["artifact_store_root"],
         "--target-mod=" + configuration["target_mod"]],
        input=description.encode("utf-8"), capture_output=True, timeout=30,
        cwd=Path(__file__).resolve().parents[2], check=True,
    )
    result = json.loads(process.stdout)
    assert result["request_planning_report"]["user_request"] == description
    assert all(Path(ref["path"]).is_relative_to(configuration["artifact_store_root"])
               for ref in result["artifacts"].values())
