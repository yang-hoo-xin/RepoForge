from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.engineering_cognition import (
    _fresh_api_augmentation_wiring,
    build_projection,
    project_current,
    project_ownership,
    project_product_api,
    project_surfaces,
    project_workflow,
    query_entity,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_fresh_api_wiring_requires_forwarding_not_names_in_comments(tmp_path):
    modules = {
        "src/mod_agent/cli.py": "# make_fresh_api_source_augmentation_factory\n# make_fresh_assisted_api_source_augmentation_factory\n",
        "src/mod_agent/fresh_coverage.py": "def run_fresh_requirement_coverage(): pass\n",
        "src/mod_agent/evidence/requirement_coverage_provider.py": "# source_augmentation_factory\n",
        "src/mod_agent/intelligent_evidence_only.py": "def make_fresh_api_source_augmentation_factory(): pass\ndef make_fresh_assisted_api_source_augmentation_factory(): pass\n",
    }
    for name, source in modules.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    result = _fresh_api_augmentation_wiring(tmp_path)
    assert result["connected"] is False
    assert result["api_factory_names"] == []


def test_summary_and_detail_agree_on_optional_fresh_api_wiring():
    summary = build_projection("summary", root=REPOSITORY_ROOT)
    assert summary["data"]["product_api"]["fresh_status"] == project_product_api(REPOSITORY_ROOT)["fresh_coverage"]["status"]


def _run(*args: str, cwd: Path = REPOSITORY_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-m", "scripts.engineering_cognition", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="strict",
        text=True,
    )


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        check=True,
        encoding="utf-8",
        errors="strict",
        text=True,
    )
    return completed.stdout.strip()


def test_current_marks_mismatched_task_packet_as_context_only(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "engineering-cognition@example.invalid")
    _git(tmp_path, "config", "user.name", "Engineering Cognition Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")
    current_head = _git(tmp_path, "rev-parse", "HEAD")

    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "task-packet.json").write_text(
        json.dumps(
            {
                "task_id": "old-task",
                "base_sha": "1" * 40,
                "git_snapshot": {"head_sha": "2" * 40},
            }
        ),
        encoding="utf-8",
    )
    (codex / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PreCompact": [
                        {
                            "matcher": "^(manual|auto)$",
                            "hooks": [{"type": "command", "commandWindows": "python hook.py"}],
                        }
                    ],
                    "SessionStart": [
                        {
                            "matcher": "^(compact|resume)$",
                            "hooks": [{"type": "command", "commandWindows": "python hook.py"}],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    for index in range(105):
        (tmp_path / f"untracked-{index:03d}.txt").write_text("x\n", encoding="utf-8")

    result = project_current(tmp_path)

    assert result["git"]["head_sha"] == current_head
    assert result["task_packet"]["freshness"] == "stale"
    assert result["task_packet"]["reason"] == "head_changed"
    assert result["task_packet"]["context_role"] == "context_only"
    assert result["hooks"]["configuration"]["configured"] is True
    assert result["hooks"]["configuration"]["shape_valid"] is True
    assert result["hooks"]["runtime_snapshot"]["observed"] is False
    assert result["hooks"]["lifecycle_acceptance"]["status"] == "not_observed"
    assert result["git"]["status_count"] > 100
    assert len(result["git"]["status_short"]) == 100
    assert result["git"]["status_truncated"] is True


def test_surface_projection_keeps_mcp_as_two_bounded_surfaces() -> None:
    result = project_surfaces(REPOSITORY_ROOT)
    surfaces = {item["surface"]: item for item in result["surfaces"]}

    assert result["scope"] == "registered_entry_surfaces_not_all_internal_apis"
    assert {entry["name"] for entry in surfaces["standard_rag_mcp"]["entries"]} == {
        "get_document_summary",
        "list_collections",
        "query_knowledge_hub",
    }
    assert {entry["name"] for entry in surfaces["controlled_agent_mcp"]["entries"]} == {
        "get_artifact",
        "get_eval_report",
        "get_feature_status",
        "get_job_status",
        "get_proposal_review",
        "list_features",
        "preview_patch_proposal",
        "preview_retrieval",
    }
    assert {entry["path"] for entry in surfaces["http_get"]["entries"]} == {
        "/features",
        "/features/{feature_id}/artifacts",
        "/features/{feature_id}/dashboard",
        "/features/{feature_id}/status",
        "/health",
    }
    assert surfaces["product_cli"]["entry_count"] == 24
    assert "continue-fresh-patch" in {entry["name"] for entry in surfaces["product_cli"]["entries"]}
    assert surfaces["product_cli"]["entrypoint_alignment"] == "match"
    assert surfaces["standard_rag_mcp"]["package_entrypoint"] == "src.mcp_server.server:main"
    assert surfaces["standard_rag_mcp"]["implementation_entrypoint"] == (
        "src.mcp_server.server:main"
    )
    assert surfaces["standard_rag_mcp"]["entrypoint_alignment"] == "match"
    assert surfaces["standard_rag_mcp"]["installed_entry_health"] == "not_inferred"
    assert surfaces["dashboard_pages"]["entry_count"] == 7


def test_workflow_and_artifact_entity_are_registry_backed() -> None:
    workflow = project_workflow(REPOSITORY_ROOT)
    routes = [route for route in workflow["routes"] if route["source"] == "retrieve_evidence"]
    fields = {field["field"]: field for field in workflow["state_fields"]}

    assert {route["label"]: route["destination"] for route in routes} == {
        "blocked": "blocked",
        "continue": "create_implementation_plan",
    }
    assert fields["artifact_hashes"]["writers"] == ["artifact_store"]
    assert fields["runtime_summary"]["writers"] == [
        "blocked",
        "collect_runtime_evidence",
        "done",
        "evaluate",
    ]

    entity = query_entity(REPOSITORY_ROOT, "artifact", "implementation_plan")
    contract = entity["result"]["contract"]
    assert contract["producer_stage"] == "implementation_plan"
    assert contract["required_upstream_artifacts"] == ["evidence_pack", "feature_spec"]
    assert "generate_patch_proposal" in contract["consumer_stages"]
    assert entity["result"]["latest_actual_evidence"]["binding"] == "not_available"
    assert entity["result"]["latest_actual_evidence"]["status"] == "unknown"
    assert entity["result"]["semantic_owner"] == "not_inferred"
    assert entity["result"]["final_authority"] == "not_inferred"
    assert entity["result"]["test_reference_semantics"] == ("exact_lexical_reference_only")
    assert entity["result"]["test_execution"] == "not_performed_by_projection"
    assert entity["result"]["validator_scope"] == "generic_registry_entrypoints_only"
    assert entity["result"]["artifact_specific_validator"] == "not_inferred"

    build_contract = query_entity(REPOSITORY_ROOT, "artifact", "build_result")["result"]["contract"]
    assert build_contract["required_one_of_upstream_artifact_groups"] == [
        ["dry_run_result", "evidence_pack"],
        ["dry_run_result", "feature_spec"],
    ]


def test_ownership_anchor_is_not_observed_without_source_refs() -> None:
    with patch("scripts.engineering_cognition._text_refs_in_paths", return_value=[]):
        result = project_ownership(REPOSITORY_ROOT)

    anchor = result["coverage_planning_binding_drift_anchor"]
    assert anchor["source_refs"] == []
    assert anchor["status"] == "not_observed"


def test_product_api_reports_optional_fresh_wiring_without_claiming_semantic_success() -> None:
    result = project_product_api(REPOSITORY_ROOT)
    components = {item["name"]: item for item in result["components"]}

    assert components["EvidenceDiscoverySpecialist"]["status"] == (
        "source_and_test_references_observed"
    )
    assert components["LocalAssemblyEvidenceToolExecutor"]["status"] == (
        "source_and_test_references_observed"
    )
    assert components["RunLocalMemberAccessResolver"]["status"] == (
        "source_and_test_references_observed"
    )
    assert components["verified_member_accesses"]["status"] == (
        "source_and_test_references_observed"
    )
    assert all(
        item["verification_status"] == "not_executed_by_projection" for item in components.values()
    )
    assert result["fresh_coverage"]["evidence_provider"] == "RgLiveEvidenceProvider"
    assert result["fresh_coverage"]["source_ref"].startswith("src/mod_agent/fresh_coverage.py:")
    assert result["fresh_coverage"]["provider_scope"] == "RgLive_plus_optional_API_augmentation"
    assert result["fresh_coverage"]["status"] == "partially_connected"
    assert result["fresh_coverage"]["verification_status"] == "source_wiring_not_executed_by_projection"
    assert result["fresh_coverage"]["api_factory_names"] == [
        "make_fresh_api_source_augmentation_factory",
        "make_fresh_assisted_api_source_augmentation_factory",
    ]
    assert len(result["connection_boundaries"]) == 3
    assert result["general_documentation_gap"]["status"] == "not_connected_to_fresh"
    assert result["general_product_api_semantics"]["status"] == ("general_semantics_unproven")
    assert set(result["api_semantic_questions"]) >= {
        "read",
        "write",
        "subscription",
        "Harmony target",
        "return propagation",
        "legacy compatibility",
    }


def test_summary_is_bounded_and_all_eight_views_are_explicit() -> None:
    payload = build_projection(root=REPOSITORY_ROOT)
    data = payload["data"]

    assert payload["read_only"] is True
    assert payload["persistence"] == "none"
    assert [item["name"] for item in data["eight_views"]] == [
        "Entry/Surface",
        "Workflow/Route",
        "Artifact DAG",
        "Ownership/Authority",
        "Source/Dependency",
        "API Usage",
        "Behavior/Dataflow",
        "Verification/Claim",
    ]
    assert "contracts" not in data
    assert "routes" not in data
    assert len(json.dumps(payload, ensure_ascii=False, sort_keys=True)) < 12_000


def test_cli_emits_stably_sorted_json_and_exact_entity_query() -> None:
    first = _run("--kind", "state-field", "--name", "artifacts")
    second = _run("--kind", "state-field", "--name", "artifacts")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["view"] == "entity"
    assert payload["data"]["exact_match_only"] is True
    assert payload["data"]["result"]["field"]["writers"] == ["artifact_producing_node"]


def test_cli_rejects_half_entity_query_without_writing_state() -> None:
    before = sorted(path.as_posix() for path in REPOSITORY_ROOT.glob(".codex/.runtime/**/*"))
    result = _run("--kind", "route")
    after = sorted(path.as_posix() for path in REPOSITORY_ROOT.glob(".codex/.runtime/**/*"))

    assert result.returncode == 2
    assert "--kind and --name must be provided together" in result.stderr
    assert before == after


def test_symbol_query_does_not_treat_docstring_text_as_a_definition(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "example.py").write_text(
        '"""Example text.\n\ndef Phantom():\n    pass\n"""\n',
        encoding="utf-8",
    )

    result = query_entity(tmp_path, "symbol", "Phantom")

    assert result["result"]["definition_refs"] == []
    assert result["result"]["status"] == "not_found_exact"


def test_entity_query_does_not_bind_actual_from_free_text(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "example.py").write_text("class ExactThing:\n    pass\n", encoding="utf-8")
    journal = tmp_path / "artifacts/run/actual_journal.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "actual_attempt_id": "actual_1",
                "journal_version": "actual_run_journal_v1",
                "status": "success",
                "terminal": {"note": "ExactThing", "status": "success"},
                "terminal_summary_complete": True,
            }
        ),
        encoding="utf-8",
    )

    result = query_entity(tmp_path, "symbol", "ExactThing")["result"]

    assert result["status"] == "found"
    assert result["latest_actual_evidence"]["binding"] == "not_available"
    assert result["latest_actual_evidence"]["status"] == "unknown"
