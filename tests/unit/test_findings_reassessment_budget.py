"""Production reassessment queries must not spend an unusable stage allowance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.mod_agent import local_entry as owner
from src.mod_agent.evidence.discovery_contract import EvidenceDiscoverySource
from src.mod_agent.evidence.discovery_tools import LocalAssemblyEvidenceToolExecutor
from src.mod_agent.evidence.reassessment import make_reassessment_api_augmentation
from tests.unit.test_intelligent_evidence_discovery import _ILRunner, _MetadataRunner
from tests.unit.test_requirement_coverage import _initial_findings_reassessment_case


def _budget(findings=3):
    limits = {"requirement_projection": 1, "requirement_findings": findings,
              "planning": 1, "initial_patch": 5 - findings, "repair": 4}
    used = {"requirement_projection": 1, "requirement_findings": 2}
    return {"maximum": 11, "consumed": 3, "remaining": 8,
            "count_basis": "reserved_provider_invocations_not_transport_proof",
            "by_stage": {key: {"maximum": value, "consumed": used.get(key, 0),
                               "remaining": value - used.get(key, 0)} for key, value in limits.items()}}


def _factory(case, root, budget, calls, *, augmentation=None, require_api=False, api_edit=False,
             cycles=1, first_findings_missing=False, cycle_observer=None,
             first_query="ResolveCoverageGap", native=False):
    query_calls = 0
    findings_calls = 0
    def complete(role, payload, _schema):
        nonlocal query_calls, findings_calls
        updated = owner._reserve_request_model_call(budget, owner._request_model_budget_stage(role))
        budget.clear()
        budget.update(updated)
        calls.append((role, deepcopy(payload)))
        if role == "initial_findings_queries":
            query_calls += 1
            return {"queries": [{"query": first_query if query_calls == 1 else "ChangeMarker",
                "requirement_id": payload["requirements"][0]["requirement_id"],
                "gap_ids": [payload["gap_investigation_targets"][0]["gap_id"]],
                "current_fact_needed": "Existing source and API ownership for this gap.",
                "modification_impact": "Planning cannot bind the change without that current fact.",
                "authorized_source": "dependency_api" if require_api else "target_source",
                "metadata_queries": ["Move"] if require_api else []}]}
        findings_calls += 1
        source = next((item["evidence_id"] for item in payload["source_observations"]
                       if item["source_type"] == "target_source" and "ResolveCoverageGap" in item["excerpt"]),
                      next(item["evidence_id"] for item in payload["source_observations"]
                           if item["source_type"] == "target_source"))
        api = [item["evidence_id"] for item in payload["source_observations"] if item["source_type"] == "decompiled_api"]
        found = (not require_api or bool(api)) and not (
            first_findings_missing and findings_calls == 1)
        if payload.get("native_findings_contract") == "planning_readiness_v1":
            return {"findings": [{"relation_index": item["relation_index"],
                "planning_readiness": "ready" if found else "needs_current_evidence",
                "current_evidence_ids": [source, *api] if found else [],
                "current_basis": "Fixture current implementation and API evidence." if found else "Current dependency contract is not observed.",
                "requested_behavior_status": "not_yet_implemented", "edit_target_evidence_ids": api if api_edit else [],
                "blocking_current_facts": [] if found else [{"fact_needed": "Current dependency API contract",
                    "planning_impact": "The target implementation cannot be planned without that API fact."}]}
                for item in payload["relations"]]}
        return {"findings": [{"relation_index": item["relation_index"],
            "relationship": "support" if found else "missing", "evidence_ids": [source, *api] if found else [],
            "edit_target_evidence_ids": api if api_edit else [],
            "unresolved_gaps": [] if found else ["Current dependency evidence is still unavailable."]}
            for item in payload["relations"]]}
    return owner._make_request_upstream_evidence_factory(artifact_store=case.store, workspace_root=root,
        source_provider=case.source_provider, upstream_search_root="generated_mods/SyntheticMod",
        complete_requirement_coverage_json=complete, budget_snapshot=lambda: deepcopy(budget),
        source_augmentation_factory=augmentation, max_initial_reassessment_cycles=cycles,
        cycle_observer=cycle_observer, native_findings_contract="planning_readiness_v1" if native else None)


@pytest.mark.parametrize("require_api", [False, True])
def test_native_planning_readiness_uses_same_findings_call_and_preserves_unknown_api_blocker(tmp_path, require_api):
    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls = _budget(), []
    result = _factory(case, tmp_path, budget, calls, native=True, require_api=require_api).reassess_initial(case.state, case.feature)
    assert [role for role, _ in calls] == ["initial_findings_queries", "requirement_findings"]
    assert budget["consumed"] == 5 and budget["remaining"] == 6
    assert calls[1][1]["native_findings_contract"] == "planning_readiness_v1"
    assert result.can_generate_patch is (not require_api)
    audit = result.budget_summary["reassessment"]["findings_attempt_audit"][-1]
    native_result = audit["native_findings_provenance"]["native_response"]
    assert all(row["requested_behavior_status"] == "not_yet_implemented" for row in native_result["findings"])


def test_meter_only_reassessment_counts_calls_and_keeps_no_new_information_stop(tmp_path):
    from src.mod_agent.model_call_budget import as_meter_only

    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls = as_meter_only(_budget()), []
    result = _factory(case, tmp_path, budget, calls,
        first_findings_missing=True, first_query="ChangeMarker").reassess_initial(case.state, case.feature)
    assert budget["policy"] == "meter_only" and budget["maximum"] is budget["remaining"] is None
    assert budget["consumed"] == 3 + len(calls)
    assert len(calls) >= 2
    if not result.can_generate_patch:
        assert result.budget_summary["reassessment_stop"]["code"] == "INITIAL_FINDINGS_NO_NEW_INFORMATION_STOP"
    assert result.budget_summary["initial_findings_reassessment"]["model_call_budget_policy"] == "meter_only"


@pytest.mark.parametrize("shortage", ["stage", "total"])
def test_production_factory_preflights_before_query_or_retrieval(tmp_path, monkeypatch, shortage):
    case = _initial_findings_reassessment_case(tmp_path)
    budget = _budget(2)
    if shortage == "total":
        for row in budget["by_stage"].values():
            row.update(consumed=row["maximum"], remaining=0)
        budget.update(consumed=11, remaining=0)
    original = deepcopy(budget)
    calls = []
    monkeypatch.setattr(case.source_provider, "collect_query_set", lambda **_: pytest.fail("Futile retrieval started"))
    with pytest.raises(owner.LocalEntryError) as caught:
        _factory(case, tmp_path, budget, calls).reassess_initial(case.state, case.feature)
    diagnostic = caught.value.model_call_boundary["budget_preflight"]
    assert diagnostic["blockers"][0]["scope"] == shortage
    assert all(row["sequence_execution_allowed"] is False and row["reservation_performed"] is False
               for row in diagnostic["calls"])
    assert str(caught.value) == ("PRODUCT_STAGE_MODEL_CALL_BUDGET_EXHAUSTED" if shortage == "stage"
                                 else "PRODUCT_MODEL_CALL_BUDGET_EXHAUSTED")
    assert budget == original and calls == []
    assert caught.value.model_call_boundary["model_request_invoked"] is False


def test_sufficient_missing_reassesses_real_source_and_preserves_prior_materials(tmp_path):
    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls = _budget(), []
    original = Path(case.evidence_ref.path).read_bytes()
    result = _factory(case, tmp_path, budget, calls).reassess_initial(case.state, case.feature)
    assert [role for role, _ in calls] == ["initial_findings_queries", "requirement_findings"]
    instruction = calls[0][1]["instruction"]
    assert "do not query whether the requested new behavior is already implemented" in instruction
    assert "current source or dependency API facts" in instruction
    assert "case-insensitive literal substring matching" in instruction
    assert "compact type or member identifier substrings in metadata_queries" in calls[0][1]["instruction"]
    assert "Every metadata_queries item is 1-128 characters" in calls[0][1]["instruction"]
    target = calls[0][1]["gap_investigation_targets"][0]
    assert set(target) == {"gap_id", "relation_index", "gap_index"}
    query = calls[0][1]
    assert "prior_query_outcomes" not in query
    assert "must not choose a Hook" in instruction
    assert budget["consumed"] == 5 and budget["remaining"] == 6
    assert result.can_generate_patch
    assert {item.evidence_id for item in case.pack.items} < {item.evidence_id for item in result.items}
    assert result.budget_summary["dependency_api_supplement"]["status"] == "not_configured"
    assert Path(case.evidence_ref.path).read_bytes() == original
    assert not case.state["approvals"]


def test_structured_metadata_queries_reject_questions_before_tool_dispatch(tmp_path, monkeypatch):
    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls = _budget(), []
    factory = _factory(case, tmp_path, budget, calls)
    factory.query_selector = lambda feature, projection, old, feedback: [{
        "query": "ResolveCoverageGap",
        "requirement_id": projection["requirements"][0]["requirement_id"],
        "metadata_queries": ["Which input API?"],
    }]
    monkeypatch.setattr(case.source_provider, "collect_query_set",
                        lambda **_: pytest.fail("Invalid metadata query reached retrieval"))

    with pytest.raises(ValueError, match="PATCH_REASSESSMENT_QUERY_INVALID"):
        factory.reassess_initial(case.state, case.feature)


def test_preview_uses_same_production_role_mapping_without_reservation():
    budget = _budget()
    # Before the first correction, not after the fixture's accepted missing turn.
    budget["by_stage"]["requirement_findings"].update(consumed=1, remaining=2)
    budget.update(consumed=2, remaining=9)
    before = deepcopy(budget)
    result = owner.preflight_initial_findings_model_calls(budget, include_correction=True)
    assert result["admitted"] and budget == before
    assert [row["budget_stage"] for row in result["calls"]] == ["requirement_findings", "repair", "requirement_findings"]
    assert result["required_by_stage"] == {"requirement_findings": 2, "repair": 1}
    assert result["minimum_stage_limits"] == {"requirement_findings": 3, "repair": 1}
    assert result["projected_budget"]["consumed"] == 5
    assert result["model_calls_performed"] == 0 and result["budget_changed"] is False


def test_bounded_query_roles_use_smaller_output_reservation_without_changing_other_roles():
    assert owner._request_model_output_budget("initial_findings_queries", 4096) == 768
    assert owner._request_model_output_budget("upstream_evidence_queries", 4096) == 768
    assert owner._request_model_output_budget("requirement_findings", 4096) == 4096
    assert owner._request_model_output_budget("requirement_projection", 4096) == 4096


def test_two_cycle_preflight_covers_query_findings_planning_and_patch():
    budget = {"maximum": 17, "consumed": 11, "remaining": 6,
        "count_basis": "reserved_provider_invocations_not_transport_proof",
        "by_stage": {
            "requirement_projection": {"maximum": 1, "consumed": 1, "remaining": 0},
            "requirement_findings": {"maximum": 8, "consumed": 6, "remaining": 2},
            "planning": {"maximum": 1, "consumed": 0, "remaining": 1},
            "initial_patch": {"maximum": 1, "consumed": 0, "remaining": 1},
            "repair": {"maximum": 6, "consumed": 4, "remaining": 2},
        }}

    result = owner.preflight_initial_findings_model_calls(
        budget, include_correction=False, max_cycles=2, include_downstream=True)

    assert result["admitted"] is True
    assert [row["budget_stage"] for row in result["calls"]] == [
        "repair", "requirement_findings", "repair", "requirement_findings",
        "planning", "initial_patch"]
    assert result["projected_budget"]["remaining"] == 0
    assert result["model_calls_performed"] == 0


def test_two_cycle_reassessment_changes_query_after_new_observation(tmp_path):
    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls = _budget(4), []
    observed_cycles = []

    result = _factory(case, tmp_path, budget, calls, cycles=2,
                      first_findings_missing=True,
                      cycle_observer=lambda pack, cycle: observed_cycles.append(
                          (cycle, pack.requirement_coverage["sufficiency"]["status"]))).reassess_initial(
                              case.state, case.feature)

    assert [role for role, _ in calls] == [
        "initial_findings_queries", "requirement_findings",
        "initial_findings_queries", "requirement_findings"]
    assert calls[0][1]["message_type"] == calls[2][1]["message_type"]
    assert "prior_query_outcomes" not in calls[0][1]
    assert calls[2][1]["prior_query_outcomes"]["new_evidence_ids"]
    assert result.can_generate_patch is True
    assert result.budget_summary["reassessment_cycle_history"][0]["cycle"] == 1
    assert observed_cycles == [(1, "insufficient"), (2, "sufficient")]


def test_two_cycle_reassessment_records_explicit_stop_when_first_cycle_adds_no_information(tmp_path):
    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls, observed = _budget(4), [], []

    result = _factory(case, tmp_path, budget, calls, cycles=2,
        first_findings_missing=True, first_query="ChangeMarker",
        cycle_observer=lambda pack, cycle: observed.append(
            deepcopy(pack.budget_summary.get("reassessment_stop")))).reassess_initial(
                case.state, case.feature)

    assert [role for role, _ in calls] == ["initial_findings_queries", "requirement_findings"]
    assert result.can_generate_patch is False
    assert result.budget_summary["reassessment_stop"] == {
        "code": "INITIAL_FINDINGS_NO_NEW_INFORMATION_STOP",
        "cycle": 1,
        "prior_item_count": len(case.pack.items),
        "new_item_count": 0,
        "new_evidence_ids": [],
        "sufficiency_status": "insufficient",
        "not_started": ["initial_findings_queries_cycle_2", "requirement_findings_cycle_2",
                        "planning", "initial_patch"],
        "fact_boundary": "deterministic counts and stage reachability only",
    }
    assert observed == [result.budget_summary["reassessment_stop"]]


def test_zero_new_source_only_history_requires_one_dependency_query_before_tool_dispatch(
    tmp_path, monkeypatch,
):
    from src.mod_agent.evidence.reassessment import (
        RequirementCoverageReassessmentFactory,
        dependency_source_diversification_required,
    )
    from src.mod_agent.evidence.requirement_coverage_provider import (
        StructuredModelRequirementCoverageProvider,
    )

    case = _initial_findings_reassessment_case(tmp_path)
    coverage = deepcopy(case.pack.requirement_coverage)
    api_requirement = next(row for row in coverage["requirements"]
                           if row["role"] == "pre_patch_evidence")
    api_requirement.update(text="current dependency API evidence",
                           query_leads=["current dependency API evidence"])
    old = replace(case.pack, requirement_coverage=coverage,
        budget_summary={**case.pack.budget_summary,
        "reassessment": {"new_item_count": 0, "queries": [{
            "query": "ChangeMarker", "authorized_source": "target_source",
            "metadata_queries": []}]},
        "dependency_api_supplement": {"status": "not_requested"}})
    assert dependency_source_diversification_required(old) is True
    contract = owner._build_request_upstream_query_contract(
        feature=case.feature, projection=coverage, old=old,
        feedback={"initial_findings_reassessment": {}},
        upstream_search_root="generated_mods/SyntheticMod")
    assert contract["payload"]["source_diversification_required"] is True
    assert contract["payload"]["source_diversification_mode"] == "deterministic_bounded_source_policy"
    assert contract["payload"]["source_diversification_requirement_ids"]
    query = {"query": "ChangeMarker",
        "requirement_id": case.materialized["requirements"][0]["requirement_id"],
        "gap_ids": [contract["payload"]["gap_investigation_targets"][0]["gap_id"]],
        "current_fact_needed": "Current dependency input API.",
        "modification_impact": "Planning needs a current dependency fact.",
        "authorized_source": "target_source", "metadata_queries": []}
    monkeypatch.setattr(case.source_provider, "collect_query_set",
                        lambda **_: pytest.fail("Source-diversification failure reached tool dispatch"))
    factory = RequirementCoverageReassessmentFactory(
        artifact_store=case.store, workspace_root=tmp_path,
        search_root="generated_mods/SyntheticMod", source_provider=case.source_provider,
        semantic_provider=StructuredModelRequirementCoverageProvider(
            lambda *_args, **_kwargs: pytest.fail("Invalid query reached Findings")),
        query_selector=lambda *_args: [query])

    with pytest.raises(ValueError, match="INITIAL_FINDINGS_DEPENDENCY_SOURCE_DIVERSIFICATION_REQUIRED"):
        factory._reassess(case.state, case.feature, old,
                          {"initial_findings_reassessment": {}}, initial_lineage={"cycle": 1})


def _api(root, *, authorized=True, unreadable=False):
    folder = root / "api"
    folder.mkdir()
    (folder / "Reference.dll").write_bytes(b"controlled current assembly bytes")
    source = EvidenceDiscoverySource(source_id="dependency-fixture", source_type="decompiled_api",
        kind="managed_assembly", evidence_class="metadata_il", source_authority="supporting",
        description="Public tool-boundary fixture", root_path=folder,
        allowed_tools=("search_dotnet_metadata", "inspect_method_il") if authorized else (),
        assembly_relative_paths=("Reference.dll",))
    metadata = _MetadataRunner()
    if unreadable:
        def absent(*_args, **_kwargs):
            raise OSError("controlled unavailable assembly")
        metadata.search = absent
    executor = LocalAssemblyEvidenceToolExecutor(metadata_runner=metadata, il_runner_factory=_ILRunner)
    return make_reassessment_api_augmentation(source=source, tool_executor=executor,
        run_id="public-reassessment-fixture", max_method_inspections=1)


def test_run_dependency_binding_constructs_the_same_existing_tool_path(tmp_path):
    run = tmp_path / "run"
    api = run / "build_dependencies" / "GameFolder"
    api.mkdir(parents=True)
    (api / "Reference.dll").write_bytes(b"controlled current assembly bytes")
    snapshot = owner._binary_tree_snapshot(api)
    digest = owner._binary_tree_snapshot_sha256(snapshot)
    inspector = tmp_path / "inspector.dll"
    inspector.write_bytes(b"controlled inspector identity")
    executor = LocalAssemblyEvidenceToolExecutor(metadata_runner=_MetadataRunner(), il_runner_factory=_ILRunner)
    factory, binding = owner.make_run_dependency_api_reassessment_augmentation(run_root=run,
        run_id="run-dependency-fixture", game_folder_dependency={
            "binding_version": "run_local_game_folder_binding_v1", "source_snapshot_sha256": digest,
            "source_file_count": 1, "isolated_copy": str(api), "copy_snapshot_sha256": digest,
            "copy_file_count": 1, "build_uses_isolated_copy": True}, inspector_path=inspector,
        inspector_sha256=__import__("hashlib").sha256(inspector.read_bytes()).hexdigest(),
        max_method_inspections=1, tool_executor=executor)
    assert callable(factory)
    assert binding["source_snapshot_sha256"] == digest and binding["assembly_count"] == 1
    assert binding["source"]["source_type"] == "decompiled_api"
    assert binding["model_calls_performed"] == 0 and binding["target_code_execution_allowed"] is False


def test_run_dependency_binding_reports_missing_inspector_without_building(tmp_path):
    run = tmp_path / "run"
    api = run / "build_dependencies" / "GameFolder"
    api.mkdir(parents=True)
    (api / "Reference.dll").write_bytes(b"controlled current assembly bytes")
    snapshot = owner._binary_tree_snapshot(api)
    digest = owner._binary_tree_snapshot_sha256(snapshot)
    with pytest.raises(owner.LocalEntryError, match="DEPENDENCY_API_INSPECTOR_UNAVAILABLE"):
        owner.make_run_dependency_api_reassessment_augmentation(run_root=run, run_id="run",
            game_folder_dependency={"binding_version": "run_local_game_folder_binding_v1",
                "source_snapshot_sha256": digest, "source_file_count": 1,
                "isolated_copy": str(api), "copy_snapshot_sha256": digest, "copy_file_count": 1,
                "build_uses_isolated_copy": True}, inspector_path=tmp_path / "missing-inspector.dll",
            inspector_sha256="0" * 64)


@pytest.mark.parametrize("mode", ["current", "not_authorized", "unavailable"])
def test_dependency_gap_uses_original_tools_and_never_invents_edit_authority(tmp_path, mode):
    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls = _budget(), []
    augment = _api(tmp_path, authorized=mode != "not_authorized", unreadable=mode == "unavailable")
    result = _factory(case, tmp_path, budget, calls, augmentation=augment, require_api=True).reassess_initial(case.state, case.feature)
    api = [item for item in result.items if item.source_type == "decompiled_api"]
    assert {item.evidence_id for item in case.pack.items}.issubset({item.evidence_id for item in result.items})
    assert result.can_generate_patch is (mode == "current")
    assert bool(api) is (mode == "current")
    expected_status = {"current": "observed", "not_authorized": "not_authorized",
                       "unavailable": "unavailable"}[mode]
    assert result.budget_summary["dependency_api_supplement"]["status"] == expected_status
    assert result.budget_summary["dependency_api_supplement"]["query_source"] == "metadata_queries"
    assert result.budget_summary["dependency_api_supplement"]["metadata_query_count"] == 1
    visible = result.budget_summary["dependency_api_supplement"]["visible_metadata"]
    if mode == "current":
        assert visible["methods"] and visible["types"] and visible["fields"]
    else:
        assert visible == {"methods": [], "types": [], "fields": []}
    if mode == "not_authorized":
        assert result.budget_summary["dependency_api_supplement"]["reason"] == "DEPENDENCY_API_TOOLS_NOT_AUTHORIZED"
    elif mode == "unavailable":
        assert result.budget_summary["dependency_api_supplement"]["reason"] == "DEPENDENCY_API_SOURCE_UNAVAILABLE"
    if api:
        assert api[0].source_hash == api[0].file_hash and api[0].metadata_token == "0x06000002"
        assert api[0].usage == "hook_evidence"
        assert result.budget_summary["source_augmentation"]["provenance_validation"]["target_code_executed"] is False
    assert all(not row["edit_target_evidence_ids"] for row in result.requirement_coverage["findings"])
    assert not case.state["approvals"]
    assert budget["consumed"] == 5


def test_api_material_does_not_become_a_writable_target(tmp_path):
    case = _initial_findings_reassessment_case(tmp_path)
    budget, calls = _budget(4), []
    with pytest.raises(ValueError, match="REQUIREMENT_COVERAGE"):
        _factory(case, tmp_path, budget, calls, augmentation=_api(tmp_path), require_api=True,
                 api_edit=True).reassess_initial(case.state, case.feature)
    assert not case.state["approvals"]
