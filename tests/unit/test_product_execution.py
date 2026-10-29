from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import copy
import hashlib
import inspect
import json
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.mod_agent import product_executor as product_executor_module
from src.mod_agent.artifacts import ArtifactStore, ArtifactWriteResult
from src.mod_agent.build import BuildAdapter, BuildCommandResult, BuildProfile
from src.mod_agent.evidence import (
    EvidenceItem,
    EvidencePack,
    RgLiveEvidenceProvider,
    build_requirement_coverage_pack,
    default_feature_requirement_projection,
    make_requirement_coverage_evidence_provider,
    materialize_requirement_coverage_projection,
    validated_requirement_coverage_view,
)
from src.mod_agent.evidence.models import DETERMINISTIC_EVIDENCE_AUTHORITY
from src.mod_agent.execution import (
    ExactDiffApprovalRejectedError,
    ExactTextPatchApplier,
)
from src.mod_agent.git.admission import GitExecutableIdentityV1, StrictGitAdmission
from src.mod_agent.runtime import (
    RuntimeEvidenceItem,
    build_real_runtime_evidence_payload,
)
from src.mod_agent.product_execution import (
    PRODUCT_EXECUTION_DECLARATION,
    ProductExecutionContractError,
    ProductExecutionInputBindingV1,
    ProductExecutionOutputV1,
    ProductExecutionRequestV1,
    ProductExecutionResultV1,
)
from src.mod_agent.product_executor import (
    LocalEntryError,
    ProductExecutionAdaptersV1,
    SOURCE_DISCLOSURE_DATA_CLASSES,
    _bind_owner_validated_runtime_evidence_factory,
    _external_model_admission_record,
    _local_model_admission_record,
    _local_no_external_disclosure_record,
    _runtime_admission_record,
    _source_disclosure_record,
    _verify_source_revision_binding,
    execute_product_execution,
    preflight_product_execution,
    source_snapshot,
    source_snapshot_sha256,
)
from src.mod_agent.local_entry import (
    _persist_graph_terminal_result,
    _verify_patch_model_input_disclosure,
)
from src.non_authoritative_projection import (
    EdgeReporterV1,
    ExplicitEdgeV1,
    OwnerFactV1,
    ProjectionViewV1,
)


@pytest.mark.parametrize("mismatch", ["prior_evidence_ref", "trace_ref", "projection"])
def test_initial_findings_reassessment_readback_rejects_changed_lineage(
    tmp_path, monkeypatch, capsys, mismatch,
):
    """Replay an actual public-entry deterministic stop; no provider is called on replay."""
    from tests.unit.test_mod_agent_cli import _public_patch_upstream_scenario

    observed = _public_patch_upstream_scenario(tmp_path, monkeypatch, capsys,
        initial_findings_gap=True, findings_still_missing=True)
    projection = copy.deepcopy(observed.result["product_execution_result"])
    ProductExecutionResultV1.from_dict(projection)
    calls_before = len(observed.requests)
    readback = next(row["payload"] for row in projection["validation_facts"]["details"]
                    if row["detail_kind"] == "output_readback")
    evidence = [row for row in readback if row["artifact_type"] == "evidence_pack"]
    assert len(evidence) == 2
    prior, reassessed = evidence
    assert prior["status"] == reassessed["status"] == "blocked"
    assert prior["data"]["requirement_coverage"]["projection_identity"] == (
        reassessed["data"]["requirement_coverage"]["projection_identity"])
    if mismatch == "prior_evidence_ref":
        reassessed["data"]["budget_summary"]["initial_findings_reassessment"]["prior_evidence_ref"]["artifact_hash"] = "0" * 64
    elif mismatch == "trace_ref":
        reassessed["trace_refs"] = []
    else:
        reassessed["data"]["requirement_coverage"]["requirements"][0]["text"] = "Changed requirement"
    with pytest.raises(ProductExecutionContractError, match="Initial Findings reassessment"):
        ProductExecutionResultV1.from_dict(projection)
    assert len(observed.requests) == calls_before
    assert observed.planning == observed.patch_contexts == []


def _sha256_json(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _human_semantic_approve(
    proposal: dict[str, object],
    review: dict[str, object],
    diff_hash: str,
) -> dict[str, object]:
    context_pack = proposal["context_pack"]
    assert isinstance(context_pack, dict)
    artifact_hashes = context_pack["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    assurance = proposal["semantic_assurance"]
    return {
        "schema_version": "human_semantic_decision_v1",
        "feature_id": proposal["feature_id"],
        "feature_spec_artifact_hash": artifact_hashes["feature_spec"],
        "proposal_id": proposal["proposal_id"],
        "proposal_payload_sha256": _sha256_json(proposal),
        "proposal_review_payload_sha256": _sha256_json(review),
        "semantic_assurance_payload_sha256": _sha256_json(assurance),
        "diff_hash": diff_hash,
        "decision": "approve",
        "reviewer_identity": "human:product-execution-test",
    }


def _human_semantic_approve_from_dry_run(
    dry_run: dict[str, object],
) -> dict[str, object]:
    binding = dry_run["human_semantic_decision_binding"]
    assert isinstance(binding, dict)
    return {
        **binding,
        "decision": "approve",
        "reviewer_identity": "human:product-execution-test",
    }


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _nested_source_binding(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    repository = tmp_path / "repo"
    project = repository / "App"
    project.mkdir(parents=True)
    (project / "App.csproj").write_text(
        '<Project Sdk="Microsoft.NET.Sdk"></Project>\n', encoding="utf-8"
    )
    loader = project / "Loader.cs"
    loader.write_text("sealed class Loader {}\n", encoding="utf-8")
    generated = project / "obj" / "Generated.cs"
    generated.parent.mkdir()
    generated.write_text("sealed class Generated {}\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.name", "Product Binding Test")
    _git(repository, "config", "user.email", "product-binding@test.local")
    _git(repository, "config", "core.autocrlf", "false")
    _git(repository, "add", "--all")
    _git(repository, "commit", "-m", "baseline")
    snapshot = source_snapshot(project)
    loader_entry = next(item for item in snapshot if item["path"] == "Loader.cs")
    binding: dict[str, object] = {
        "binding_version": "product_source_revision_binding_v2",
        "complete_snapshot_sha256": source_snapshot_sha256(snapshot),
        "repository_head": _git(repository, "rev-parse", "HEAD"),
        "repository_tree": _git(repository, "rev-parse", "HEAD^{tree}"),
        "project_root": "App",
        "project_file": "App/App.csproj",
        "project_tree": _git(repository, "rev-parse", "HEAD:App"),
        "primary_sources": [
            {
                "path": "App/Loader.cs",
                "sha256": loader_entry["sha256"],
            }
        ],
    }
    return project, binding


@pytest.mark.parametrize("retain_plan", [True, False])
def test_product_upstream_retains_current_failed_plan_and_original_checks(tmp_path, monkeypatch, capsys, retain_plan):
    """Public run-request; deterministic transport/evaluator, no product Apply."""
    from tests.unit.test_mod_agent_cli import _public_patch_upstream_scenario

    observed = _public_patch_upstream_scenario(tmp_path, monkeypatch, capsys,
        owner="planning" if retain_plan else "evidence", fail_first=True,
        retain_plan=retain_plan, task_limit=10)
    assert observed.exit_code == 0 and observed.result["status"] == "awaiting_approval", observed.result
    assert observed.roles == ["product_requirement_projection", "product_requirement_findings",
        "planning", "patch", "patch",
        *([] if retain_plan else ["patch_upstream_evidence_queries", "product_requirement_findings"]),
        "planning", "patch"]
    assert len(observed.evaluations) == 2
    initial_context, failed = observed.evaluations[0]
    final_context, passed = observed.evaluations[-1]
    assert failed["status"] == "behavior_failed" and passed["status"] == "passed"
    assessment = observed.planning[-1]["upstream_assessment"]
    assert assessment["retain_current_plan_allowed"] is retain_plan
    assert assessment["failure_report"]["generated_changes_sha256"] == failed["generated_changes_sha256"]
    assert assessment["failure_report"]["source_snapshot_sha256"] == failed["source_snapshot_sha256"]
    assert assessment["failure_report"]["observations"] == failed["observations"]
    assert assessment["previous_response"] == observed.generated_responses[0]
    assert assessment["failed_response_sha256"] == hashlib.sha256(
        observed.generated_responses[0].encode("utf-8")).hexdigest()
    assert assessment["failure_report_sha256"] == failed["report_ref"]["sha256"]
    assert (initial_context["implementation_plan_ref"] == final_context["implementation_plan_ref"]) is retain_plan
    assert initial_context["generated_changes_sha256"] != final_context["generated_changes_sha256"]
    assert failed["observations"][0]["check_id"] == passed["observations"][0]["check_id"]
    if retain_plan:
        assert initial_context["source_snapshot_sha256"] == final_context["source_snapshot_sha256"]
        carried = next(row for row in observed.worker_requests[-1]
            if row.get("message_type") == "patch_same_plan_assessment")
        assert carried["approval_inherited"] is False
    else:
        assert len(final_context["generated_changes"]) == 2
        assert "RefreshSharedState" in json.dumps(observed.patch_contexts[-1])
    feedback = next(row for row in observed.worker_requests[-1]
        if row.get("message_type") == ("patch_behavior_correction" if retain_plan else "patch_replanned_context"))
    assert feedback["validator_feedback"]["observations"] == failed["observations"]
    assert feedback["validator_feedback"]["generated_changes_sha256"] == failed["generated_changes_sha256"]
    assert observed.result["approval_recorded"] is False
    assert observed.result["actual_write_performed"] is False
    assert observed.result["model_call_budget"]["consumed"] == (7 if retain_plan else 9)


def test_legacy_root_plugin_source_binding_remains_compatible(tmp_path: Path) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    plugin = source / "Plugin.cs"
    plugin.write_text("sealed class Plugin {}\n", encoding="utf-8")
    snapshot = source_snapshot(source)

    _verify_source_revision_binding(
        source=source,
        snapshot=snapshot,
        snapshot_sha256=source_snapshot_sha256(snapshot),
        binding={
            "complete_snapshot_sha256": source_snapshot_sha256(snapshot),
            "plugin_cs_sha256": hashlib.sha256(plugin.read_bytes()).hexdigest(),
            "repository_head": "legacy-head",
            "subtree_tree": "legacy-tree",
        },
    )


def test_nested_project_and_non_plugin_primary_source_bind_and_roundtrip(
    tmp_path: Path,
) -> None:
    project, binding = _nested_source_binding(tmp_path)
    snapshot = source_snapshot(project)

    _verify_source_revision_binding(
        source=project,
        snapshot=snapshot,
        snapshot_sha256=source_snapshot_sha256(snapshot),
        binding=binding,
    )
    artifact_binding = ProductExecutionInputBindingV1.create(
        "source_revision_binding", binding
    )

    assert ProductExecutionInputBindingV1.from_dict(
        artifact_binding.to_dict()
    ) == artifact_binding
    assert binding["primary_sources"][0]["path"] == "App/Loader.cs"


def test_strict_source_binding_does_not_fall_back_to_path_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, binding = _nested_source_binding(tmp_path)
    repository = project.parent
    executable = shutil.which("git")
    assert executable is not None
    strict = StrictGitAdmission(
        executable_identity=GitExecutableIdentityV1.read(executable),
        source_declared_root=repository.as_posix(),
        source_canonical_root=repository.resolve().as_posix(),
    )
    monkeypatch.setattr(
        "src.mod_agent.product_executor._git_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("PATH Git fallback must not run")
        ),
    )
    snapshot = source_snapshot(project)

    _verify_source_revision_binding(
        source=project,
        snapshot=snapshot,
        snapshot_sha256=source_snapshot_sha256(snapshot),
        binding=binding,
        strict_git_admission=strict,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("project_missing", "project_file is absent from snapshot"),
        ("project_escape", "project_root escapes the repository"),
        ("source_escape", "primary_sources.path escapes the repository"),
        ("source_missing", "primary source is absent from snapshot"),
        ("source_hash", "primary source SHA256 changed"),
        ("head", "repository head drifted"),
        ("repository_tree", "repository tree drifted"),
        ("project_tree", "project tree drifted"),
        ("snapshot", "snapshot SHA256 does not match"),
        ("duplicate_source", "primary source path is duplicated"),
    ],
)
def test_generic_source_binding_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    project, original = _nested_source_binding(tmp_path)
    binding = copy.deepcopy(original)
    if mutation == "project_missing":
        binding["project_file"] = "App/Missing.csproj"
    elif mutation == "project_escape":
        binding["project_root"] = "../App"
    elif mutation == "source_escape":
        binding["primary_sources"][0]["path"] = "../Loader.cs"
    elif mutation == "source_missing":
        binding["primary_sources"][0] = {
            "path": "App/obj/Generated.cs",
            "sha256": hashlib.sha256(
                (project / "obj" / "Generated.cs").read_bytes()
            ).hexdigest(),
        }
    elif mutation == "source_hash":
        binding["primary_sources"][0]["sha256"] = "0" * 64
    elif mutation == "head":
        binding["repository_head"] = "0" * 40
    elif mutation == "repository_tree":
        binding["repository_tree"] = "0" * 40
    elif mutation == "project_tree":
        binding["project_tree"] = "0" * 40
    elif mutation == "snapshot":
        binding["complete_snapshot_sha256"] = "0" * 64
    else:
        binding["primary_sources"].append(
            copy.deepcopy(binding["primary_sources"][0])
        )
    snapshot = source_snapshot(project)

    with pytest.raises(LocalEntryError, match=message):
        _verify_source_revision_binding(
            source=project,
            snapshot=snapshot,
            snapshot_sha256=source_snapshot_sha256(snapshot),
            binding=binding,
        )


def test_multiple_projects_require_explicit_binding(tmp_path: Path) -> None:
    source = tmp_path / "multi"
    (source / "One").mkdir(parents=True)
    (source / "Two").mkdir()
    (source / "One" / "One.csproj").write_text("<Project />\n", encoding="utf-8")
    (source / "Two" / "Two.csproj").write_text("<Project />\n", encoding="utf-8")
    snapshot = source_snapshot(source)

    with pytest.raises(
        LocalEntryError,
        match="Multiple source projects require an explicit source revision binding",
    ):
        _verify_source_revision_binding(
            source=source,
            snapshot=snapshot,
            snapshot_sha256=source_snapshot_sha256(snapshot),
            binding=None,
        )


def test_generic_source_binding_rejects_live_revision_drift(tmp_path: Path) -> None:
    project, binding = _nested_source_binding(tmp_path)
    (project / "Loader.cs").write_text("sealed class Changed {}\n", encoding="utf-8")
    snapshot = source_snapshot(project)

    with pytest.raises(LocalEntryError, match="snapshot SHA256 does not match"):
        _verify_source_revision_binding(
            source=project,
            snapshot=snapshot,
            snapshot_sha256=source_snapshot_sha256(snapshot),
            binding=binding,
        )


def _request(tmp_path: Path) -> tuple[ProductExecutionRequestV1, ArtifactStore]:
    source = (tmp_path / "source").resolve()
    isolated = (tmp_path / "run" / "generated_mods" / "Example").resolve()
    source.mkdir(parents=True)
    isolated.mkdir(parents=True)
    (source / "Plugin.cs").write_text("sealed class Plugin {}\n", encoding="utf-8")
    (isolated / "Plugin.cs").write_text("sealed class Plugin {}\n", encoding="utf-8")
    store = ArtifactStore((tmp_path / "run" / "a").resolve())
    return (
        ProductExecutionRequestV1(
            contract_version="1",
            run_id="run_product_001",
            request_id="request_product_001",
            user_request="make the example behavior explicit",
            target_domain="gtfo",
            target_mod="Example",
            requested_completion_scope="compile_only",
            repair_budget=2,
            isolated_workspace_root=str((tmp_path / "run").resolve()),
            isolated_target_root=str(isolated),
            source_root=str(source),
            artifact_store_root=str(store.root),
            source_snapshot_sha256=source_snapshot_sha256(source_snapshot(source)),
            isolated_source_snapshot_sha256=source_snapshot_sha256(
                source_snapshot(isolated)
            ),
            input_bindings=tuple(
                ProductExecutionInputBindingV1.create(kind, payload)
                for kind, payload in (
                    ("source_revision_binding", None),
                    ("source_disclosure", _local_no_external_disclosure_record()),
                    (
                        "model_admission",
                        _local_model_admission_record(
                            model_identity={
                                "name": "deterministic-test",
                                "digest": "sha256:" + "1" * 64,
                                "base_url": "http://127.0.0.1:11434",
                            },
                            provider_adapter="deterministic_test",
                        ),
                    ),
                    (
                        "runtime_admission",
                        _runtime_admission_record(
                            object(),
                            run_gtfo_runtime=False,
                            allow_game_launch=False,
                            runtime_plugin_destination=None,
                            allow_game_directory_write=False,
                        ),
                    ),
                )
            ),
            completion_classification=(),
            confirmed_isolated_write=True,
            allow_build=True,
        ),
        store,
    )


def _blocked_evidence_adapters() -> ProductExecutionAdaptersV1:
    return ProductExecutionAdaptersV1(
        evidence_pack_factory=lambda _state, _feature: EvidencePack(
            query="missing symbol",
            readiness="insufficient",
            can_generate_patch=False,
            blockers=(
                {
                    "code": "PRODUCT_EVIDENCE_MISSING",
                    "message": "Required source evidence is missing.",
                },
            ),
        ),
        patch_change_factory=lambda *_: (),
        dry_run_factory=lambda *_: {},
        isolated_apply_factory=lambda *_: {},
        build_result_factory=lambda *_: (_ for _ in ()).throw(
            AssertionError("Build must not run on the early blocked route")
        ),
        runtime_evidence_factory=None,
        compiled_artifact_verifier=None,
        source_change_verifier=None,
    )


def test_external_owner_records_bind_complete_feature_wide_disclosure() -> None:
    disclosure = _source_disclosure_record(
        provider="openai",
        model="qwen3.6-plus",
        base_url="https://example.invalid/compatible-mode/v1",
        approved_at="2026-08-31T00:00:00+00:00",
        approval_id="approval_feature_wide_001",
        provider_adapter="deterministic_external_test",
        destination_url=(
            "https://example.invalid/compatible-mode/v1/chat/completions"
        ),
    )

    assert disclosure["requested_data_classes"] == list(
        SOURCE_DISCLOSURE_DATA_CLASSES
    )
    assert disclosure["approved_data_classes"] == list(
        SOURCE_DISCLOSURE_DATA_CLASSES
    )
    admission = _external_model_admission_record(disclosure)
    assert admission["requested_data_classes"] == list(
        SOURCE_DISCLOSURE_DATA_CLASSES
    )
    assert admission["approved_data_classes"] == list(
        SOURCE_DISCLOSURE_DATA_CLASSES
    )


@pytest.mark.parametrize(
    "missing_data_class",
    [
        "feature_target_behavior",
        "feature_acceptance_criteria",
        "feature_runtime_verification_plan",
    ],
)
def test_external_product_execution_missing_feature_consent_has_zero_patch_calls(
    tmp_path: Path,
    missing_data_class: str,
) -> None:
    request, store = _request(tmp_path)
    approved_data_classes = tuple(
        data_class
        for data_class in SOURCE_DISCLOSURE_DATA_CLASSES
        if data_class != missing_data_class
    )
    disclosure = _source_disclosure_record(
        provider="openai",
        model="qwen3.6-plus",
        base_url="https://example.invalid/compatible-mode/v1",
        approved_at="2026-08-31T00:00:00+00:00",
        approval_id="approval_feature_wide_subset_001",
        provider_adapter="deterministic_external_test",
        destination_url=(
            "https://example.invalid/compatible-mode/v1/chat/completions"
        ),
        requested_data_classes=SOURCE_DISCLOSURE_DATA_CLASSES,
        approved_data_classes=approved_data_classes,
    )
    values = request.to_dict()
    values["input_bindings"][1] = ProductExecutionInputBindingV1.create(
        "source_disclosure", disclosure
    ).to_dict()
    values["input_bindings"][2] = ProductExecutionInputBindingV1.create(
        "model_admission", _external_model_admission_record(disclosure)
    ).to_dict()
    request = ProductExecutionRequestV1.from_dict(values)
    original_adapters = _successful_compile_adapters(request)
    patch_provider_calls: list[dict[str, object]] = []

    def patch_change_factory(state, implementation, evidence_pack):
        model_input = {
            "request": state["user_request"],
            "evidence": list(evidence_pack.get("items") or []),
            "binding_registry": dict(
                implementation.get("binding_registry") or {}
            ),
            "required_change_units": list(
                implementation.get("proposed_changes") or []
            ),
            "side_effect_boundaries": list(
                implementation.get("side_effect_boundaries") or []
            ),
            "target_behavior": "Implement the Feature-wide behavior.",
            "acceptance_criteria": ["Observe the requested behavior."],
            "runtime_verification_plan": [
                "Restart and observe the requested behavior."
            ],
        }
        _verify_patch_model_input_disclosure(
            disclosure,
            model_input=model_input,
        )
        patch_provider_calls.append({"chat": "called"})
        return original_adapters.patch_change_factory(
            state,
            implementation,
            evidence_pack,
        )

    adapters = replace(
        original_adapters,
        patch_change_factory=patch_change_factory,
    )
    preflight = preflight_product_execution(request, adapters, store)

    assert preflight.preflight_status == "ready"
    result = execute_product_execution(preflight, adapters, store)
    assert patch_provider_calls == []
    assert any(missing_data_class in item.message for item in result.blockers)


def _formalize_product_test_pack(
    *,
    query: str,
    feature_data: dict[str, object],
    retrieved_pack: EvidencePack,
) -> EvidencePack:
    assert retrieved_pack.items
    materialized = materialize_requirement_coverage_projection(
        feature_data=feature_data,
        request=query,
        projection=default_feature_requirement_projection(feature_data),
    )
    evidence_id = retrieved_pack.items[0].evidence_id
    per_query_execution = [
        {
            "query_id": row["query_id"],
            "requirement_id": row["requirement_id"],
            "query_sha256": hashlib.sha256(
                str(row["query"]).encode("utf-8")
            ).hexdigest(),
            "evidence_ids": [evidence_id],
        }
        for row in materialized["queries"]
    ]
    retrieved_pack = replace(
        retrieved_pack,
        budget_summary={
            **dict(retrieved_pack.budget_summary),
            "query_count": len(per_query_execution),
            "max_items": len(per_query_execution),
            "per_query_execution": per_query_execution,
            "all_queries_executed": True,
            "dense_enabled": False,
        },
    )
    required_by_id = {
        str(requirement["requirement_id"]): bool(
            requirement["required_before_patch"]
        )
        for requirement in materialized["requirements"]
    }
    change_requirement_ids = {
        str(requirement["requirement_id"])
        for requirement in materialized["requirements"]
        if requirement["role"] == "change"
    }
    findings = {
        "findings": [
            {
                "relation_index": int(
                    relation["provenance"]["relation_index"]
                ),
                "relationship": (
                    "support"
                    if any(
                        required_by_id[str(parent)]
                        for parent in relation["parent_requirement_ids"]
                    )
                    else "missing"
                ),
                "evidence_ids": (
                    [evidence_id]
                    if any(
                        required_by_id[str(parent)]
                        for parent in relation["parent_requirement_ids"]
                    )
                    else []
                ),
                "edit_target_evidence_ids": (
                    [evidence_id]
                    if change_requirement_ids.intersection(
                        str(parent) for parent in relation["parent_requirement_ids"]
                    )
                    else []
                ),
                "unresolved_gaps": (
                    []
                    if any(
                        required_by_id[str(parent)]
                        for parent in relation["parent_requirement_ids"]
                    )
                    else ["Observed only after patching."]
                ),
            }
            for relation in materialized["relations"]
        ]
    }
    return build_requirement_coverage_pack(
        query=query,
        materialized_projection=materialized,
        retrieved_pack=retrieved_pack,
        findings=findings,
    )


def _successful_compile_adapters(
    request: ProductExecutionRequestV1,
    *,
    build_exit_codes: list[int] | None = None,
    physical_apply: bool = False,
) -> ProductExecutionAdaptersV1:
    source_path = "generated_mods/Example/Plugin.cs"
    diff_hash = "sha256:" + "d" * 64
    remaining_build_exit_codes = list(build_exit_codes or [0])
    target = Path(request.isolated_workspace_root) / source_path
    applier = ExactTextPatchApplier()
    observed_dry_run = None
    patch_calls = 0

    def evidence_pack_factory(_state, feature_data):
        retrieved_pack = EvidencePack(
            query=request.user_request,
            readiness="sufficient",
            can_generate_patch=True,
            items=(
                EvidenceItem(
                    evidence_id="evidence_product_compile_001",
                    source_type="target_source",
                    source_path=source_path,
                    symbol="Load",
                    span={"start_line": 1, "end_line": 1},
                    retrieval_method="symbol_exact",
                    score=1.0,
                    claim_supported="Load is the exact selected source symbol.",
                    excerpt=target.read_bytes().decode("utf-8") if physical_apply else "public void Load() {}",
                    source_ref=str(target) if physical_apply else None,
                    source_hash=hashlib.sha256(target.read_bytes()).hexdigest() if physical_apply else "a" * 64,
                    file_hash=hashlib.sha256(target.read_bytes()).hexdigest() if physical_apply else None,
                    doc_hash=hashlib.sha256(target.read_bytes()).hexdigest() if physical_apply else None,
                    authority=DETERMINISTIC_EVIDENCE_AUTHORITY,
                ),
            ),
            retrieval_methods_used=("symbol_exact",),
        )
        return _formalize_product_test_pack(
            query=request.user_request,
            feature_data=feature_data,
            retrieved_pack=retrieved_pack,
        )

    def patch_change_factory(_state, implementation, _evidence_pack):
        nonlocal patch_calls
        patch_calls += 1
        changes = copy.deepcopy(implementation["proposed_changes"])
        assert len(changes) == 1
        changes[0]["patch"] = {
            "kind": "exact_text_replace",
            "before_text": "public void Load() {}",
            "after_text": "public void Load() { /* product */ }",
            "expected_occurrences": 1,
        }
        if physical_apply:
            changes[0]["patch"].update(before_text=target.read_bytes().decode("utf-8"),
                after_text=f"public void Load() {{ /* product revision {patch_calls} */ }}\n")
        return tuple(changes)

    def dry_run_factory(_proposal, _implementation, _review):
        nonlocal observed_dry_run
        if physical_apply:
            observed_dry_run = applier.run(workspace_root=request.isolated_workspace_root,
                proposal=_proposal, implementation_plan=_implementation, review=_review, allow_write=False)
            return observed_dry_run
        return {
            "execution_mode": "dry_run",
            "diff_hash": diff_hash,
            "unified_diff": "--- Plugin.cs\n+++ Plugin.cs\n",
            "actual_write_performed": False,
            "real_patch_applied": False,
            "changed_files": [source_path],
        }

    def isolated_apply_factory(proposal, _implementation, review):
        if physical_apply:
            assert observed_dry_run is not None
            return applier.run(workspace_root=request.isolated_workspace_root,
                proposal=proposal, implementation_plan=_implementation, review=review, allow_write=True,
                approved_diff_hash=observed_dry_run["diff_hash"],
                human_semantic_decision=_human_semantic_approve_from_dry_run(observed_dry_run))
        human_decision = _human_semantic_approve(
            proposal,
            review,
            diff_hash,
        )
        return {
            "execution_mode": "isolated_apply",
            "actual_write_performed": True,
            "real_patch_applied": True,
            "approved_by_review": True,
            "approved_by_user": True,
            "changed_files": [source_path],
            "diff_hash": diff_hash,
            "approved_diff_hash": diff_hash,
            "human_semantic_review_completed": True,
            "human_semantic_review_result": "approved",
            "human_semantic_decision": human_decision,
        }

    def build_result_factory(feature_id, input_execution_mode):
        build_root = Path(request.isolated_target_root)
        expected = build_root / "bin" / "Debug" / "Example.dll"

        def runner(_profile):
            exit_code = remaining_build_exit_codes.pop(0)
            if exit_code == 0:
                expected.parent.mkdir(parents=True, exist_ok=True)
                expected.write_bytes(b"deterministic product build")
            return BuildCommandResult(
                exit_code=exit_code,
                stdout="Build succeeded" if exit_code == 0 else "Build failed",
                stderr=(
                    ""
                    if exit_code == 0
                    else (
                        f"{source_path}(1,1): error CS0246: "
                        "The type or namespace name 'Missing' could not be found"
                    )
                ),
                duration_ms=3,
            )

        return BuildAdapter().run_local(
            feature_id=feature_id,
            profile=BuildProfile(
                profile_id="product_compile_test",
                build_system="dotnet",
                working_directory=str(build_root),
                command=("dotnet", "build"),
                timeout_seconds=30,
                expected_artifacts=("bin/Debug/Example.dll",),
            ),
            command_runner=runner,
            allow_execution=True,
            input_execution_mode=input_execution_mode,
        )

    return ProductExecutionAdaptersV1(
        evidence_pack_factory=evidence_pack_factory,
        planning_target_selection_factory=(
            lambda _feature, _evidence: (
                "evidence_product_compile_001",
            )
        ),
        patch_change_factory=patch_change_factory,
        dry_run_factory=dry_run_factory,
        isolated_apply_factory=isolated_apply_factory,
        build_result_factory=build_result_factory,
        runtime_evidence_factory=None,
        compiled_artifact_verifier=None,
        source_change_verifier=None,
    )


def _formal_coverage_vertical_scenario(
    tmp_path: Path,
    *,
    search_root: str = "generated_mods/Example",
    remove_requirement_coverage: bool = False,
    requirement_coverage_completion=None,
    change_unresolved_gaps: tuple[str, ...] = (),
) -> SimpleNamespace:
    request, store = _request(tmp_path)
    workspace_root = Path(request.isolated_workspace_root)
    target = workspace_root / search_root / "Behavior.cs"
    target.parent.mkdir(parents=True, exist_ok=True)
    source_before = (
        "public sealed class Behavior\n"
        "{\n"
        "    public void ChangeMarker()\n"
        "    {\n"
        "        // preserve normal feedback target source API evidence config text\n"
        "        // runtime verification acceptance\n"
        "    }\n"
        "}\n"
    ).encode("utf-8")
    source_after = source_before.replace(
        b"public void ChangeMarker()",
        b"public void ChangeMarkerWithCoverage()",
    )
    target.write_bytes(source_before)
    canonical = Path(request.source_root) / "Plugin.cs"
    canonical_before = canonical.read_bytes()
    values = request.to_dict()
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(values)
    trace: dict[str, object] = {
        "completion_roles": [],
        "patch_calls": 0,
        "dry_run_calls": 0,
        "apply_calls": 0,
        "build_calls": 0,
        "build_observed_sha256": None,
        "dry_run_diff_hash": None,
        "coverage_edit_target_ids": (),
        "patch_edit_binding_id": None,
        "patch_binding_evidence_id": None,
    }

    def complete_json(role, payload, _schema):
        trace["completion_roles"].append(role)
        if requirement_coverage_completion is not None:
            return requirement_coverage_completion(role, payload, _schema)
        if role == "requirement_projection":
            obligations = payload["input_obligations"]
            change_indexes = [
                row["input_obligation_index"]
                for row in obligations
                if "change" in row["allowed_roles"]
            ]
            requirements = [
                {
                    "role": "change",
                    "text": "Change the marker behavior.",
                    "input_obligation_indexes": change_indexes,
                    "query_leads": ["ChangeMarker"],
                },
                {
                    "role": "preservation",
                    "text": "Preserve normal feedback requested by the user.",
                    "input_obligation_indexes": [1],
                    "query_leads": ["ChangeMarker"],
                },
            ]
            dispositions = []
            for row in obligations:
                index = row["input_obligation_index"]
                if index in change_indexes or index == 1:
                    continue
                if "scope_boundary" in row["allowed_dispositions"]:
                    dispositions.append(
                        {
                            "input_obligation_index": index,
                            "disposition": "scope_boundary",
                            "reason": (
                                "The non-goal is retained as a planning scope "
                                "boundary."
                            ),
                        }
                    )
                elif "post_change" in row["allowed_dispositions"]:
                    dispositions.append(
                        {
                            "input_obligation_index": index,
                            "disposition": "post_change",
                            "reason": "This observation occurs after patching.",
                        }
                    )
                else:
                    requirements.append(
                        {
                            "role": row["allowed_roles"][0],
                            "text": row["text"],
                            "input_obligation_indexes": [index],
                            "query_leads": ["evidence"],
                        }
                    )
            return {
                "requirements": requirements,
                "input_dispositions": dispositions,
            }
        if role == "requirement_findings":
            evidence_id = payload["source_observations"][0]["evidence_id"]
            return {
                "findings": [
                    {
                        "relation_index": relation["relation_index"],
                        "relationship": "support",
                        "evidence_ids": [evidence_id],
                        "edit_target_evidence_ids": (
                            [evidence_id]
                            if any(
                                parent["role"] == "change"
                                for parent in relation["parent_requirements"]
                            )
                            and not change_unresolved_gaps
                            else []
                        ),
                        "unresolved_gaps": (
                            list(change_unresolved_gaps)
                            if any(
                                parent["role"] == "change"
                                for parent in relation["parent_requirements"]
                            )
                            else []
                        ),
                    }
                    for relation in payload["relations"]
                ]
            }
        raise AssertionError(role)

    coverage_provider = make_requirement_coverage_evidence_provider(
        RgLiveEvidenceProvider(workspace_root),
        complete_json=complete_json,
    )

    def evidence_pack_factory(state, feature_data):
        pack = coverage_provider.collect(
            query=str(state["user_request"]),
            search_root=search_root,
            feature_data=feature_data,
        )
        coverage = validated_requirement_coverage_view(pack.to_artifact_payload())
        assert coverage is not None
        trace["coverage_edit_target_ids"] = tuple(
            sorted(coverage.edit_target_hint_evidence_ids)
        )
        return (
            replace(pack, requirement_coverage=None)
            if remove_requirement_coverage
            else pack
        )

    def patch_change_factory(_state, implementation, evidence_pack):
        trace["patch_calls"] = int(trace["patch_calls"]) + 1
        changes = copy.deepcopy(implementation["proposed_changes"])
        assert len(changes) == 1
        binding_id = changes[0]["edit_binding_id"]
        bindings = {
            binding["binding_id"]: binding
            for binding in implementation["binding_registry"]["edit_bindings"]
        }
        binding = bindings[binding_id]
        coverage = validated_requirement_coverage_view(evidence_pack)
        if coverage is not None:
            assert binding["evidence_id"] in coverage.change_support_evidence_ids
        trace["patch_edit_binding_id"] = binding_id
        trace["patch_binding_evidence_id"] = binding["evidence_id"]
        assert "target_file" not in changes[0]
        assert "canonical_target_path" not in changes[0]
        changes[0]["patch"] = {
            "kind": "exact_text_replace",
            "match_mode": "bound_span_exact",
            "before_text": "public void ChangeMarker()",
            "after_text": "public void ChangeMarkerWithCoverage()",
            "expected_occurrences": 1,
        }
        return tuple(changes)

    applier = ExactTextPatchApplier()

    def dry_run_factory(proposal, implementation, review):
        trace["dry_run_calls"] = int(trace["dry_run_calls"]) + 1
        before = target.read_bytes()
        result = applier.run(
            workspace_root=workspace_root,
            proposal=proposal,
            implementation_plan=implementation,
            review=review,
            allow_write=False,
        )
        assert target.read_bytes() == before == source_before
        trace["dry_run_diff_hash"] = result["diff_hash"]
        trace["dry_run_result"] = result
        return result

    def isolated_apply_factory(proposal, implementation, review):
        trace["apply_calls"] = int(trace["apply_calls"]) + 1
        approved_diff_hash = trace["dry_run_diff_hash"]
        assert isinstance(approved_diff_hash, str)
        dry_run = trace["dry_run_result"]
        assert isinstance(dry_run, dict)
        return applier.run(
            workspace_root=workspace_root,
            proposal=proposal,
            implementation_plan=implementation,
            review=review,
            allow_write=True,
            approved_diff_hash=approved_diff_hash,
            human_semantic_decision=_human_semantic_approve_from_dry_run(
                dry_run
            ),
        )

    def build_result_factory(feature_id, input_execution_mode):
        trace["build_calls"] = int(trace["build_calls"]) + 1
        expected_artifact = (
            Path(request.isolated_target_root)
            / "bin"
            / "Debug"
            / "Example.dll"
        )

        def runner(_profile):
            observed = target.read_bytes()
            trace["build_observed_sha256"] = hashlib.sha256(observed).hexdigest()
            if observed != source_after:
                return BuildCommandResult(
                    exit_code=1,
                    stderr=(
                        f"{target.as_posix()}(1,1): error CS1002: "
                        "Applied bytes do not match the deterministic expectation"
                    ),
                )
            expected_artifact.parent.mkdir(parents=True, exist_ok=True)
            expected_artifact.write_bytes(hashlib.sha256(observed).digest())
            return BuildCommandResult(
                exit_code=0,
                stdout="Deterministic byte-consuming Build substitute passed",
            )

        return BuildAdapter().run_local(
            feature_id=feature_id,
            profile=BuildProfile(
                profile_id="coverage_vertical_byte_consumer",
                build_system="deterministic_byte_check",
                working_directory=str(request.isolated_target_root),
                command=("deterministic-byte-check",),
                timeout_seconds=1,
                expected_artifacts=("bin/Debug/Example.dll",),
            ),
            command_runner=runner,
            allow_execution=True,
            input_execution_mode=input_execution_mode,
        )

    adapters = ProductExecutionAdaptersV1(
        evidence_pack_factory=evidence_pack_factory,
        planning_target_selection_factory=(
            lambda _feature, evidence: tuple(
                str(item["evidence_id"])
                for item in evidence["items"]
                if item.get("source_type") == "target_source"
            )
        ),
        patch_change_factory=patch_change_factory,
        dry_run_factory=dry_run_factory,
        isolated_apply_factory=isolated_apply_factory,
        build_result_factory=build_result_factory,
        runtime_evidence_factory=None,
        compiled_artifact_verifier=None,
        source_change_verifier=None,
    )
    return SimpleNamespace(
        request=request,
        store=store,
        adapters=adapters,
        trace=trace,
        target=target,
        source_before=source_before,
        source_after=source_after,
        canonical=canonical,
        canonical_before=canonical_before,
    )


def test_product_execution_without_planning_selection_blocks_before_patch(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    adapters = replace(
        _successful_compile_adapters(request),
        planning_target_selection_factory=None,
    )

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store),
        adapters,
        store,
    )

    assert result.execution_disposition == "domain_blocked"
    assert any(
        blocker.code == "PLANNING_TARGET_SELECTION_REQUIRED"
        for blocker in result.blockers
    )
    assert all(
        event.effect_role != "patch_generation"
        for event in result.effect_events
    )


def test_planning_selector_receives_defensive_payload_copies(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)

    def mutating_selector(feature, evidence):
        selected = str(evidence["items"][0]["evidence_id"])
        feature["target_behavior"] = "mutated outside Planning input"
        evidence["items"].clear()
        return (selected,)

    adapters = replace(
        _successful_compile_adapters(request),
        planning_target_selection_factory=mutating_selector,
    )
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store),
        adapters,
        store,
    )

    assert result.execution_disposition == "succeeded"
    evidence_output = next(
        item for item in result.outputs if item.artifact_role == "evidence_pack"
    )
    stored = store.read(evidence_output.artifact_ref)["data"]
    assert stored["items"][0]["evidence_id"] == "evidence_product_compile_001"


def _install_actual_store_failure(
    monkeypatch: pytest.MonkeyPatch,
    store: ArtifactStore,
    *,
    artifact_role: str,
    outcome: str,
    occurrence: int = 1,
) -> None:
    original_write_with_result = store.write_with_result
    original_write_text = store._write_text_atomic
    state: dict[str, object] = {
        "active": False,
        "count": 0,
        "latest_path": None,
    }

    def fail_commit_path(
        path: Path,
        content: str,
        *,
        capture_replace_error: bool = False,
    ) -> None:
        if state["active"] is True:
            if outcome == "blocked_before_commit" and path.parent.parent.name == "p":
                raise OSError(f"injected {artifact_role} pending failure")
            if outcome == "partial" and path == state["latest_path"]:
                raise OSError(f"injected {artifact_role} latest failure")
        original_write_text(
            path,
            content,
            capture_replace_error=capture_replace_error,
        )

    def write_with_result(**kwargs):
        selected = kwargs["artifact_type"] == artifact_role
        if selected:
            state["count"] = int(state["count"]) + 1
        active = selected and state["count"] == occurrence
        state["active"] = active
        state["latest_path"] = (
            store._artifact_path(kwargs["feature_id"], artifact_role)
            if active
            else None
        )
        try:
            return original_write_with_result(**kwargs)
        finally:
            state["active"] = False
            state["latest_path"] = None

    monkeypatch.setattr(store, "_write_text_atomic", fail_commit_path)
    monkeypatch.setattr(store, "write_with_result", write_with_result)


def _runtime_owner_binding(
    tmp_path: Path,
) -> tuple[dict[str, object], SimpleNamespace, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable = tmp_path / "GTFO.exe"
    executable.write_bytes(b"test executable")
    config = tmp_path / "BepInEx" / "config" / "BepInEx.cfg"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[Logging.Disk]\nAppendLog = false\n", encoding="utf-8")
    destination = tmp_path / "BepInEx" / "plugins" / "Example.dll"
    destination.parent.mkdir(parents=True, exist_ok=True)
    settings = SimpleNamespace(
        runtime=SimpleNamespace(
            gtfo=SimpleNamespace(
                enabled=True,
                game_executable=str(executable),
                working_directory=str(tmp_path),
                launch_args=["-test"],
                log_paths=["BepInEx/LogOutput.log"],
                screenshot_directory=str(tmp_path / "screenshots"),
                screenshot_timeout_seconds=15,
                startup_markers=["Chainloader startup complete"],
                acceptance_markers={"behavior works": "BEHAVIOR_OK"},
                timeout_seconds=120,
                poll_interval_seconds=1.0,
                output_limit_chars=20_000,
                terminate_after_capture=True,
            )
        )
    )
    record = _runtime_admission_record(
        settings,
        run_gtfo_runtime=True,
        allow_game_launch=True,
        runtime_plugin_destination=destination,
        allow_game_directory_write=True,
    )
    return record, settings, destination


def _runtime_request_and_adapters(
    tmp_path: Path,
    *,
    diagnosed_binding_gap: bool = False,
    terminal_runtime_repair: bool = False,
    exhausted_runtime_repair: bool = False,
    build_exit_codes: list[int] | None = None,
) -> tuple[ProductExecutionRequestV1, ArtifactStore, ProductExecutionAdaptersV1]:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    values = request.to_dict()
    values["requested_completion_scope"] = "runtime_workflow"
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    runtime_admission, runtime_settings, runtime_destination = (
        _runtime_owner_binding(tmp_path / "runtime")
    )
    values["input_bindings"][3] = ProductExecutionInputBindingV1.create(
        "runtime_admission", runtime_admission
    ).to_dict()
    request = ProductExecutionRequestV1.from_dict(values)
    base = _successful_compile_adapters(
        request,
        build_exit_codes=build_exit_codes,
        physical_apply=bool(build_exit_codes and any(code != 0 for code in build_exit_codes)),
    )

    def runtime_evidence_factory(feature_id, build, feature_spec):
        criteria = tuple(str(item) for item in feature_spec["acceptance_criteria"])
        if terminal_runtime_repair or exhausted_runtime_repair:
            return build_real_runtime_evidence_payload(
                feature_id=feature_id,
                input_execution_mode=str(build["execution_mode"]),
                evidence_items=(),
                startup_loaded=True,
                gameplay_behavior_verified=False,
                verification_status="failed",
                raw_runtime_output="Runtime observation requires new evidence.",
                acceptance_criteria=criteria,
                runtime_verification_plan=tuple(
                    str(item) for item in feature_spec["runtime_verification_plan"]
                ),
                failure_ownership=(
                    {
                        "owner": "product_behavior",
                        "failure_class": "product_behavior",
                        "source_repair_allowed": True,
                        "required_action": "repair_product_behavior",
                        "reason": "A bound product criterion is false.",
                    }
                    if exhausted_runtime_repair
                    else {
                        "owner": "observation",
                        "failure_class": "runtime_observation_incomplete",
                        "source_repair_allowed": False,
                        "required_action": "collect_runtime_evidence",
                        "reason": "The observation does not authorize source repair.",
                    }
                ),
            )
        if diagnosed_binding_gap:
            payload = build_real_runtime_evidence_payload(
                feature_id=feature_id,
                input_execution_mode=str(build["execution_mode"]),
                evidence_items=(),
                startup_loaded=True,
                gameplay_behavior_verified=False,
                verification_status="failed",
                raw_runtime_output=(
                    "signal=PRODUCT_RUNTIME_FAILED "
                    "reason=unresolved_runtime_binding"
                ),
                acceptance_criteria=criteria,
                runtime_verification_plan=tuple(
                    str(item) for item in feature_spec["runtime_verification_plan"]
                ),
                failure_ownership={
                    "owner": "product_behavior",
                    "failure_class": "product_behavior",
                    "source_repair_allowed": True,
                    "required_action": "repair_product_behavior",
                    "reason": "A bound product criterion is false.",
                },
            )
            payload["runtime_failure_diagnosis"] = {
                "schema_version": "stagger_runtime_failure_diagnosis_v1",
                "status": "diagnosed",
                "feature_id": feature_id,
                "failure_class": "product_behavior",
                "evidence_refs": [
                    {
                        "source": "deterministic_product_runtime",
                        "locator": "runtime://binding-gap",
                    }
                ],
                "responsibility_anchors": {
                    "component": "ui_attachment",
                    "symbol": "Example.Plugin.Load",
                    "dependency": "runtime binding",
                    "repair_scope": ["runtime_binding"],
                    "source_hash": None,
                    "source_span": None,
                },
                "root_cause_status": "unresolved",
                "unresolved_root_causes": ["runtime_binding_unresolved"],
            }
            return payload
        return build_real_runtime_evidence_payload(
            feature_id=feature_id,
            input_execution_mode=str(build["execution_mode"]),
            evidence_items=(
                RuntimeEvidenceItem(
                    item_type="runtime_log",
                    source="deterministic_product_runtime",
                    locator="runtime://product-001",
                    supports_acceptance=criteria,
                    confidence=1.0,
                    raw_excerpt="Runtime observed the requested behavior.",
                ),
            ),
            startup_loaded=True,
            gameplay_behavior_verified=True,
            verification_status="verified",
            raw_runtime_output="Runtime observed the requested behavior.",
            acceptance_criteria=criteria,
            runtime_verification_plan=tuple(
                str(item) for item in feature_spec["runtime_verification_plan"]
            ),
            plugin_marker_binding={
                "marker_source": "deterministic_product_runtime",
                "markers": [
                    {
                        "acceptance_criterion": criterion,
                        "marker": f"MARKER_{index}",
                        "observed": True,
                    }
                    for index, criterion in enumerate(criteria, start=1)
                ],
            },
        )

    adapters = replace(
        base,
        runtime_evidence_factory=_bind_owner_validated_runtime_evidence_factory(
            runtime_evidence_factory,
            settings=runtime_settings,
            allow_game_launch=True,
            runtime_plugin_destination=runtime_destination,
            allow_game_directory_write=True,
            expected_runtime_admission=runtime_admission,
        ),
    )
    return request, store, adapters


def _execute_store_role_scenario(
    tmp_path: Path,
    *,
    scenario: str,
    monkeypatch: pytest.MonkeyPatch | None = None,
    failure_role: str | None = None,
    failure_outcome: str | None = None,
    failure_occurrence: int = 1,
) -> ProductExecutionResultV1:
    if scenario in {
        "runtime",
        "runtime_terminal",
        "runtime_terminal_max",
        "runtime_max",
        "binding",
    }:
        request, store, adapters = _runtime_request_and_adapters(
            tmp_path,
            diagnosed_binding_gap=scenario == "binding",
            terminal_runtime_repair=scenario
            in {"runtime_terminal", "runtime_terminal_max"},
            exhausted_runtime_repair=scenario == "runtime_max",
            build_exit_codes=(
                [1, 1, 0]
                if scenario in {"runtime_terminal_max", "runtime_max"}
                else None
            ),
        )
    else:
        request, store = _request(tmp_path)
        if scenario == "blocked":
            adapters = _blocked_evidence_adapters()
        else:
            Path(request.isolated_target_root, "Plugin.cs").write_text(
                "public void Load() {}\n", encoding="utf-8"
            )
            values = request.to_dict()
            values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
                source_snapshot(Path(request.isolated_target_root))
            )
            request = ProductExecutionRequestV1.from_dict(values)
            adapters = _successful_compile_adapters(
                request,
                build_exit_codes=(
                    [1, 1]
                    if scenario == "review_max"
                    else [1, 1, 1]
                    if scenario == "repair"
                    else [0]
                ),
                physical_apply=scenario in {"review_max", "repair"},
            )
            if scenario in {"review", "review_max"}:
                original_patch_factory = adapters.patch_change_factory

                def invalid_review_patch(state, *args):
                    items = original_patch_factory(state, *args)
                    should_block = scenario == "review" or state["retry_count"] == 2
                    return tuple(
                        {**item, "expected_validation": []}
                        if should_block
                        else item
                        for item in items
                    )

                adapters = replace(
                    adapters,
                    patch_change_factory=invalid_review_patch,
                )
    if failure_role is not None:
        assert monkeypatch is not None and failure_outcome is not None
        _install_actual_store_failure(
            monkeypatch,
            store,
            artifact_role=failure_role,
            outcome=failure_outcome,
            occurrence=failure_occurrence,
        )
    return execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )


def _canonical_product_result_bytes(result: ProductExecutionResultV1) -> bytes:
    return json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _product_result_revision_set(
    result: ProductExecutionResultV1,
) -> frozenset[tuple[str, str, str]]:
    return frozenset(
        (
            output.artifact_role,
            output.artifact_ref.path,
            output.artifact_ref.artifact_hash,
        )
        for output in result.outputs
    )


class _DuplicateFieldMapping(Mapping[str, object]):
    def __init__(self, value: dict[str, object], duplicate_field: str) -> None:
        self._value = value
        self._duplicate_field = duplicate_field

    def __getitem__(self, key: str) -> object:
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return iter((*self._value, self._duplicate_field))

    def __len__(self) -> int:
        return len(self._value) + 1


def test_graph_terminal_persistence_preserves_owner_result_strict_roundtrip(
    tmp_path: Path,
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario="blocked")
    owner_projection = result.to_dict()
    assert list(owner_projection) == [
        "contract_version",
        "run_id",
        "request_id",
        "feature_id",
        "requested_completion_scope",
        "platform_status",
        "execution_disposition",
        "failure_stage",
        "side_effects_may_have_started",
        "terminal_node",
        "store_attempts",
        "outputs",
        "effect_events",
        "validation_facts",
        "blockers",
        "variant_result",
    ]
    assert result.store_attempts
    assert result.outputs
    assert result.effect_events
    assert result.validation_facts.details
    assert result.blockers
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"run_id": result.run_id, "request_id": result.request_id}),
        encoding="utf-8",
    )
    summary = {
        "run_id": result.run_id,
        "current_stage": result.terminal_node,
        "z_outer": {"z": 1, "a": {"z": 2, "a": 3}},
        "product_execution_result": owner_projection,
        "a_outer": {"z": 4, "a": 5},
    }

    _persist_graph_terminal_result(
        run_root=tmp_path,
        run_id=result.run_id,
        request_id=result.request_id,
        summary=summary,
        model_call_count=0,
        patch_payload_parser_entered=False,
        actual_write_performed=False,
    )

    result_path = tmp_path / "run_result.json"
    result_bytes = result_path.read_bytes()
    persisted = json.loads(result_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    persisted_owner = persisted["product_execution_result"]
    assert list(persisted) == sorted([*summary, "status"])
    assert persisted["status"] == result.terminal_node
    assert list(persisted["z_outer"]) == ["a", "z"]
    assert list(persisted["z_outer"]["a"]) == ["a", "z"]
    assert list(persisted_owner) == list(owner_projection)
    assert ProductExecutionResultV1.from_dict(persisted_owner) == result
    assert manifest["terminal_record"] == {
        "path": str(result_path),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }


@pytest.mark.parametrize("mutation", ("missing", "unknown", "reordered"))
def test_strict_result_replay_rejects_changed_owner_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario="blocked")
    projection = result.to_dict()
    if mutation == "missing":
        projection.pop("run_id")
    elif mutation == "unknown":
        projection["unexpected_owner_field"] = None
    else:
        projection = {key: projection[key] for key in reversed(projection)}

    with pytest.raises(
        ProductExecutionContractError,
        match="ProductExecutionResultV1 fields changed",
    ):
        ProductExecutionResultV1.from_dict(projection)


def test_strict_result_mapping_boundary_rejects_duplicate_owner_field(
    tmp_path: Path,
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario="blocked")
    duplicate_mapping = _DuplicateFieldMapping(result.to_dict(), "run_id")

    with pytest.raises(
        ProductExecutionContractError,
        match="ProductExecutionResultV1 fields changed",
    ):
        ProductExecutionResultV1.from_dict(duplicate_mapping)


def test_public_product_executor_runs_real_graph_and_strict_store_chain(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    adapters = _blocked_evidence_adapters()

    preflight = preflight_product_execution(request, adapters, store)
    result = execute_product_execution(preflight, adapters, store)

    assert preflight.preflight_status == "ready"
    assert result.platform_status == "completed"
    assert result.execution_disposition == "domain_blocked"
    assert result.terminal_node == "blocked"
    assert [item.artifact_role for item in result.store_attempts] == [
        "request_planning_report",
        "mod_planning_brief",
        "feature_spec",
        "evidence_pack",
        "implementation_plan",
        "blocked_report",
    ]
    assert all(
        item.store_result.outcome == "committed" for item in result.store_attempts
    )
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_external_post_return_projection_rebuild_preserves_product_and_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_build_graph = product_executor_module.build_graph
    graph_invoke_count = 0

    def tracked_build_graph(*args, **kwargs):
        graph = original_build_graph(*args, **kwargs)

        class TrackedGraph:
            def invoke(self, state):
                nonlocal graph_invoke_count
                graph_invoke_count += 1
                return graph.invoke(state)

        return TrackedGraph()

    monkeypatch.setattr(product_executor_module, "build_graph", tracked_build_graph)
    request, store = _request(tmp_path)
    adapters = _blocked_evidence_adapters()
    store_counts = {"read": 0, "write": 0}
    original_read = store.read
    original_write = store.write_with_result

    def tracked_read(ref):
        store_counts["read"] += 1
        return original_read(ref)

    def tracked_write(**kwargs):
        store_counts["write"] += 1
        return original_write(**kwargs)

    monkeypatch.setattr(store, "read", tracked_read)
    monkeypatch.setattr(store, "write_with_result", tracked_write)
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store),
        adapters,
        store,
    )
    assert graph_invoke_count == 1
    assert ProductExecutionResultV1.from_dict(
        copy.deepcopy(result.to_dict())
    ) == result

    readback = result.validation_facts.details[0].payload
    assert isinstance(readback, list)
    envelope_by_hash = {item["artifact_hash"]: item for item in readback}

    def product_fact(output: ProductExecutionOutputV1) -> OwnerFactV1:
        envelope = envelope_by_hash[output.artifact_ref.artifact_hash]
        assert envelope["artifact_type"] == output.artifact_ref.artifact_type
        assert envelope["feature_id"] == output.artifact_ref.feature_id
        return OwnerFactV1(
            schema_version="1",
            owner="product_execution",
            fact_type="product_execution_output_v1",
            fact_ref=output.artifact_ref.path,
            payload_sha256=output.artifact_ref.artifact_hash,
            payload_schema_version=envelope["schema_version"],
            owner_sequence=output.sequence,
        )

    def artifact_fact(output: ProductExecutionOutputV1) -> OwnerFactV1:
        envelope = envelope_by_hash[output.artifact_ref.artifact_hash]
        assert envelope["artifact_type"] == output.artifact_ref.artifact_type
        assert envelope["feature_id"] == output.artifact_ref.feature_id
        return OwnerFactV1(
            schema_version="1",
            owner="artifact_store",
            fact_type="artifact_revision_v1",
            fact_ref=output.artifact_ref.path,
            payload_sha256=envelope["artifact_hash"],
            payload_schema_version=envelope["schema_version"],
            owner_sequence=None,
        )

    source_fact = product_fact(result.outputs[0])
    target_fact = product_fact(result.outputs[1])
    store_fact = artifact_fact(result.outputs[2])

    def edge(relation: str, target: OwnerFactV1) -> ExplicitEdgeV1:
        assert source_fact.owner_sequence is not None
        return ExplicitEdgeV1(
            schema_version="1",
            relation=relation,
            source=source_fact.ref,
            target=target.ref,
            reporter=EdgeReporterV1(
                fact=source_fact.ref,
                payload_sha256=source_fact.payload_sha256,
                owner_sequence=source_fact.owner_sequence,
            ),
        )

    def snapshot() -> dict[str, object]:
        immutable_revisions = tuple(
            (path.as_posix(), path.read_bytes())
            for path in sorted(store.root.rglob("*.json"))
            if "r" in path.relative_to(store.root).parts
        )
        return {
            "canonical_result": _canonical_product_result_bytes(result),
            "terminal": result.terminal_node,
            "blockers": tuple(item.to_dict() for item in result.blockers),
            "store_attempts": tuple(
                item.to_dict() for item in result.store_attempts
            ),
            "effect_events": tuple(item.to_dict() for item in result.effect_events),
            "artifact_refs_and_hashes": _product_result_revision_set(result),
            "graph_invoke_count": graph_invoke_count,
            "store_counts": dict(store_counts),
            "immutable_revision_paths_and_bytes": immutable_revisions,
        }

    baseline = snapshot()
    normal = ProjectionViewV1.rebuild(
        (source_fact, target_fact),
        (edge("references", target_fact),),
    )
    assert normal.diagnostics == ()
    assert snapshot() == baseline

    missing = ProjectionViewV1.rebuild(
        (source_fact,),
        (edge("references", target_fact),),
    )
    assert [item.code for item in missing.diagnostics] == ["missing_target_fact"]
    assert snapshot() == baseline

    conflict = ProjectionViewV1.rebuild(
        (source_fact, store_fact),
        (edge("conflicts_with", store_fact),),
    )
    assert [item.code for item in conflict.diagnostics] == ["explicit_conflict"]
    assert {item.owner for item in conflict.facts} == {
        "artifact_store",
        "product_execution",
    }
    assert snapshot() == baseline


def test_product_executor_has_no_projection_callback_or_reverse_dependency() -> None:
    signature = inspect.signature(execute_product_execution)
    source = inspect.getsource(product_executor_module)

    assert "result_observer" not in signature.parameters
    assert "non_authoritative_projection" not in source


def test_preflight_and_execute_repeat_the_same_strict_git_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, store = _request(tmp_path)
    adapters = _blocked_evidence_adapters()
    sentinel = SimpleNamespace(name="strict-git-admission")
    observed: list[object] = []
    original = _verify_source_revision_binding

    def verify(**kwargs):
        observed.append(kwargs.get("strict_git_admission"))
        return original(**kwargs)

    monkeypatch.setattr(
        "src.mod_agent.product_executor._verify_source_revision_binding",
        verify,
    )

    preflight = preflight_product_execution(
        request,
        adapters,
        store,
        strict_git_admission=sentinel,
    )
    result = execute_product_execution(
        preflight,
        adapters,
        store,
        strict_git_admission=sentinel,
    )

    assert preflight.preflight_status == "ready"
    assert result.execution_disposition == "domain_blocked"
    assert observed == [sentinel, sentinel]


def test_product_preflight_rejects_snapshot_drift_without_graph_or_store(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "sealed class Changed {}\n", encoding="utf-8"
    )

    preflight = preflight_product_execution(
        request, _blocked_evidence_adapters(), store
    )

    assert preflight.preflight_status == "blocked"
    assert [item.code for item in preflight.blockers] == [
        "PRODUCT_ISOLATED_SNAPSHOT_DRIFT"
    ]
    assert not store.root.exists()


def test_compile_preflight_accepts_owner_runtime_record_with_enabled_settings(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    values = request.to_dict()
    settings = SimpleNamespace(
        runtime=SimpleNamespace(gtfo=SimpleNamespace(enabled=True))
    )
    runtime_admission = _runtime_admission_record(
        settings,
        run_gtfo_runtime=False,
        allow_game_launch=False,
        runtime_plugin_destination=None,
        allow_game_directory_write=False,
    )
    values["input_bindings"][3] = ProductExecutionInputBindingV1.create(
        "runtime_admission", runtime_admission
    ).to_dict()
    request = ProductExecutionRequestV1.from_dict(values)

    preflight = preflight_product_execution(
        request, _blocked_evidence_adapters(), store
    )

    assert preflight.preflight_status == "ready"
    assert not store.root.exists()


def test_product_preflight_returns_structured_blocked_for_binding_drift(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    values = request.to_dict()
    disclosure = copy.deepcopy(values["input_bindings"][1]["canonical_payload"])
    disclosure["reason"] = "caller_selected_reason"
    values["input_bindings"][1] = ProductExecutionInputBindingV1.create(
        "source_disclosure", disclosure
    ).to_dict()
    request = ProductExecutionRequestV1.from_dict(values)

    preflight = preflight_product_execution(
        request, _blocked_evidence_adapters(), store
    )

    assert preflight.preflight_status == "blocked"
    assert preflight.failure_stage == "admission"
    assert [item.code for item in preflight.blockers] == [
        "PRODUCT_INPUT_BINDING_ADMISSION_FAILED"
    ]
    assert not store.root.exists()


def test_product_preflight_rejects_unbound_runtime_adapter(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    values = request.to_dict()
    values["requested_completion_scope"] = "runtime_workflow"
    values["input_bindings"][3] = ProductExecutionInputBindingV1.create(
        "runtime_admission", {"requested": True, "caller_asserted": True}
    ).to_dict()
    request = ProductExecutionRequestV1.from_dict(values)
    adapters = replace(
        _blocked_evidence_adapters(), runtime_evidence_factory=lambda *_: None
    )

    preflight = preflight_product_execution(request, adapters, store)

    assert preflight.preflight_status == "blocked"
    assert preflight.failure_stage == "admission"
    assert not store.root.exists()


def test_public_product_executor_runs_exact_mutation_build_chain(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    request_values = request.to_dict()
    request_values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(request_values)
    adapters = _successful_compile_adapters(request)

    preflight = preflight_product_execution(request, adapters, store)
    result = execute_product_execution(preflight, adapters, store)

    assert result.platform_status == "completed"
    assert result.execution_disposition == "succeeded"
    assert result.terminal_node == "build"
    roles = tuple(item.artifact_role for item in result.store_attempts)
    assert roles[-4:] == (
        "dry_run_result",
        "isolated_apply_result",
        "build_result",
        "workflow_checkpoint",
    )
    assert all(
        item.store_result.outcome == "committed" for item in result.store_attempts
    )
    assert tuple(item.artifact_role for item in result.outputs) == roles
    assert result.variant_result.build_status == "success"
    assert result.variant_result.workflow_checkpoint_ref is not None
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_product_execution_formal_coverage_vertical_reaches_changed_bytes_and_build(
    tmp_path: Path,
) -> None:
    scenario = _formal_coverage_vertical_scenario(tmp_path)

    preflight = preflight_product_execution(
        scenario.request,
        scenario.adapters,
        scenario.store,
    )
    result = execute_product_execution(
        preflight,
        scenario.adapters,
        scenario.store,
    )

    assert preflight.preflight_status == "ready"
    assert result.platform_status == "completed"
    assert result.execution_disposition == "succeeded"
    assert result.terminal_node == "build"
    assert scenario.trace["completion_roles"] == [
        "requirement_projection",
        "requirement_findings",
    ]
    assert (
        scenario.trace["patch_calls"],
        scenario.trace["dry_run_calls"],
        scenario.trace["apply_calls"],
        scenario.trace["build_calls"],
    ) == (1, 1, 1, 1)
    assert scenario.trace["patch_binding_evidence_id"] in (
        scenario.trace["coverage_edit_target_ids"]
    )
    assert scenario.target.read_bytes() == scenario.source_after
    assert scenario.canonical.read_bytes() == scenario.canonical_before
    assert scenario.trace["build_observed_sha256"] == hashlib.sha256(
        scenario.source_after
    ).hexdigest()

    roles = tuple(item.artifact_role for item in result.store_attempts)
    assert roles == (
        "request_planning_report",
        "mod_planning_brief",
        "feature_spec",
        "evidence_pack",
        "implementation_plan",
        "patch_proposal",
        "proposal_review",
        "dry_run_result",
        "isolated_apply_result",
        "build_result",
        "workflow_checkpoint",
    )
    assert tuple(item.artifact_role for item in result.outputs) == roles
    assert tuple(
        (event.effect_role, event.event_status) for event in result.effect_events
    ) == (
        ("evidence_retrieval", "started"),
        ("evidence_retrieval", "completed"),
        ("patch_generation", "started"),
        ("patch_generation", "completed"),
        ("dry_run", "started"),
        ("dry_run", "completed"),
        ("isolated_apply", "started"),
        ("isolated_apply", "completed"),
        ("build", "started"),
        ("build", "completed"),
    )
    assert all(
        item.store_result.outcome == "committed" for item in result.store_attempts
    )

    refs = {item.artifact_role: item.artifact_ref for item in result.outputs}

    def data(role: str) -> dict[str, object]:
        return scenario.store.read(refs[role])["data"]

    def upstream_roles(role: str) -> tuple[str, ...]:
        return tuple(
            item["artifact_type"]
            for item in scenario.store.read(refs[role])["inputs"][
                "upstream_artifacts"
            ]
        )

    assert upstream_roles("evidence_pack") == ("feature_spec",)
    assert upstream_roles("implementation_plan") == (
        "feature_spec",
        "evidence_pack",
    )
    assert upstream_roles("patch_proposal") == (
        "implementation_plan",
        "evidence_pack",
    )
    assert upstream_roles("proposal_review") == ("patch_proposal",)
    assert upstream_roles("dry_run_result") == (
        "patch_proposal",
        "proposal_review",
    )
    assert upstream_roles("isolated_apply_result") == (
        "patch_proposal",
        "proposal_review",
        "dry_run_result",
    )
    assert upstream_roles("build_result") == (
        "dry_run_result",
        "isolated_apply_result",
    )
    assert "build_result" in upstream_roles("workflow_checkpoint")

    evidence = data("evidence_pack")
    implementation = data("implementation_plan")
    proposal = data("patch_proposal")
    review = data("proposal_review")
    dry_run = data("dry_run_result")
    apply_result = data("isolated_apply_result")
    build = data("build_result")
    coverage = validated_requirement_coverage_view(evidence)
    assert coverage is not None
    edit_binding = implementation["binding_registry"]["edit_bindings"][0]
    assert edit_binding["evidence_id"] in coverage.change_support_evidence_ids
    assert proposal["diff_intent"][0]["edit_binding_id"] == edit_binding["binding_id"]
    assert proposal["files"] == ["generated_mods/Example/Behavior.cs"]
    assert review["decision"] == "approve_dry_run"
    assert dry_run["actual_write_performed"] is False
    assert dry_run["real_patch_applied"] is False
    assert dry_run["approved_by_user"] is False
    assert dry_run["diff_hash"] == scenario.trace["dry_run_diff_hash"]
    assert dry_run["unified_diff_sha256"] == hashlib.sha256(
        dry_run["unified_diff"].encode("utf-8")
    ).hexdigest()
    assert apply_result["actual_write_performed"] is True
    assert apply_result["real_patch_applied"] is True
    assert apply_result["approved_diff_hash"] == dry_run["diff_hash"]
    assert apply_result["exact_diff_approved"] is True
    assert build["execution_mode"] == "local"
    assert build["build_profile"]["build_system"] == "deterministic_byte_check"
    assert build["input_execution_mode"] == "isolated_apply"
    assert build["status"] == "success"
    assert build["verified_artifacts"][0]["path"] == "bin/Debug/Example.dll"
    assert "runtime_evidence" not in refs
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_product_execution_blocks_unresolved_change_authority_before_patch(
    tmp_path: Path,
) -> None:
    authority_gap = (
        "A required physical edit site is not covered by exact current-source "
        "edit authority."
    )
    scenario = _formal_coverage_vertical_scenario(
        tmp_path,
        change_unresolved_gaps=(authority_gap,),
    )

    result = execute_product_execution(
        preflight_product_execution(
            scenario.request,
            scenario.adapters,
            scenario.store,
        ),
        scenario.adapters,
        scenario.store,
    )

    assert result.platform_status == "completed"
    assert result.execution_disposition == "domain_blocked"
    assert result.terminal_node == "blocked"
    assert scenario.trace["completion_roles"] == [
        "requirement_projection",
        "requirement_findings",
    ]
    assert (
        scenario.trace["patch_calls"],
        scenario.trace["dry_run_calls"],
        scenario.trace["apply_calls"],
        scenario.trace["build_calls"],
    ) == (0, 0, 0, 0)
    assert scenario.trace["coverage_edit_target_ids"] == ()

    refs = {item.artifact_role: item.artifact_ref for item in result.outputs}
    evidence = scenario.store.read(refs["evidence_pack"])["data"]
    implementation = scenario.store.read(refs["implementation_plan"])["data"]
    change_requirement_id = next(
        requirement["requirement_id"]
        for requirement in evidence["requirement_coverage"]["requirements"]
        if requirement["role"] == "change"
    )
    assert evidence["readiness"] == "insufficient"
    assert evidence["can_generate_patch"] is False
    assert evidence["requirement_coverage"]["edit_target_evidence_ids"] == []
    assert authority_gap in str(evidence["requirement_coverage"]["findings"])
    assert f"required_requirement_unresolved:{change_requirement_id}" in (
        evidence["requirement_coverage"]["sufficiency"]["reasons"]
    )
    assert implementation["blocking"] is True
    assert implementation["proposed_changes"] == []
    assert tuple(item.artifact_role for item in result.outputs) == (
        "request_planning_report",
        "mod_planning_brief",
        "feature_spec",
        "evidence_pack",
        "implementation_plan",
        "blocked_report",
    )


def test_product_execution_actual8_projection_correction_no_progress_fails_closed(
    tmp_path: Path,
) -> None:
    raw_outputs: list[dict[str, object]] = []
    response_schemas: list[dict[str, object]] = []
    raw_markers: list[str] = []
    overlap_details: dict[str, object] = {}

    def overlap_completion(role, payload, schema):
        assert role in {
            "requirement_projection",
            "requirement_projection_correction",
        }
        obligations = payload["input_obligations"]
        assert [row["input_obligation_index"] for row in obligations] == list(
            range(1, len(obligations) + 1)
        )
        grouped: dict[str, dict[str, object]] = {}
        for obligation in obligations:
            requirement_role = obligation["allowed_roles"][0]
            group = grouped.setdefault(
                requirement_role,
                {
                    "role": requirement_role,
                    "text": obligation["text"],
                    "input_obligation_indexes": [],
                    "query_leads": [obligation["text"]],
                },
            )
            group["input_obligation_indexes"].append(
                obligation["input_obligation_index"]
            )
        requirements = list(grouped.values())
        # 为什么要有它：反例要制造真实义务的重复归类，不依赖已移出用户需求的旧平台非目标序号。
        overlap_obligation = next(row for row in obligations if row["allowed_dispositions"]
            and len(grouped[row["allowed_roles"][0]]["input_obligation_indexes"]) > 1)
        overlap_index = overlap_obligation["input_obligation_index"]
        group_index = next(index for index, row in enumerate(requirements)
            if overlap_index in row["input_obligation_indexes"])
        item_index = requirements[group_index]["input_obligation_indexes"].index(overlap_index)
        overlap_details.update(index=overlap_index, group_index=group_index, item_index=item_index,
            allowed_dispositions=overlap_obligation["allowed_dispositions"], requirement_count=len(requirements))
        raw_marker = "actual8-overlap-output-must-not-persist"
        projection = {
            "requirements": requirements,
            "input_dispositions": [
                {
                    "input_obligation_index": overlap_index,
                    "disposition": overlap_obligation["allowed_dispositions"][0],
                    "reason": raw_marker,
                }
            ],
        }
        raw_outputs.append(copy.deepcopy(projection))
        response_schemas.append(copy.deepcopy(schema))
        raw_markers.append(raw_marker)
        return projection

    scenario = _formal_coverage_vertical_scenario(
        tmp_path,
        requirement_coverage_completion=overlap_completion,
    )
    isolated_snapshot_before = source_snapshot_sha256(
        source_snapshot(Path(scenario.request.isolated_target_root))
    )
    canonical_snapshot_before = source_snapshot_sha256(
        source_snapshot(Path(scenario.request.source_root))
    )

    result = execute_product_execution(
        preflight_product_execution(
            scenario.request,
            scenario.adapters,
            scenario.store,
        ),
        scenario.adapters,
        scenario.store,
    )

    assert scenario.trace["completion_roles"] == [
        "requirement_projection",
        "requirement_projection_correction",
    ]
    assert len(raw_outputs) == 2
    assert response_schemas[0] == response_schemas[1]
    assert (
        scenario.trace["patch_calls"],
        scenario.trace["dry_run_calls"],
        scenario.trace["apply_calls"],
        scenario.trace["build_calls"],
    ) == (0, 0, 0, 0)
    assert scenario.target.read_bytes() == scenario.source_before
    assert scenario.canonical.read_bytes() == scenario.canonical_before
    assert source_snapshot_sha256(
        source_snapshot(Path(scenario.request.isolated_target_root))
    ) == isolated_snapshot_before
    assert source_snapshot_sha256(
        source_snapshot(Path(scenario.request.source_root))
    ) == canonical_snapshot_before

    roles = tuple(item.artifact_role for item in result.store_attempts)
    assert result.platform_status == "completed"
    assert result.execution_disposition == "domain_blocked"
    assert result.terminal_node == "blocked"
    assert roles == (
        "request_planning_report",
        "mod_planning_brief",
        "feature_spec",
        "evidence_pack",
        "implementation_plan",
        "blocked_report",
    )
    assert tuple(item.artifact_role for item in result.outputs) == roles
    assert all(item.repair_attempt == 0 for item in result.store_attempts)
    assert tuple(
        (event.effect_role, event.event_status, event.repair_attempt)
        for event in result.effect_events
    ) == (
        ("evidence_retrieval", "started", 0),
        ("evidence_retrieval", "completed", 0),
    )

    refs = {item.artifact_role: item.artifact_ref for item in result.outputs}
    evidence = scenario.store.read(refs["evidence_pack"])["data"]
    implementation = scenario.store.read(refs["implementation_plan"])["data"]
    attempts = evidence["budget_summary"][
        "requirement_projection_attempt_audit"
    ]
    attempt_audit_fields = {
        "attempt_index",
        "structured_output_canonical_sha256",
        "structured_output_hash_basis",
        "schema_sha256",
        "status",
        "error_code",
        "response_schema_admitted",
        "semantic_materialization_admitted",
        "requirement_count",
        "relation_count",
        "input_disposition_count",
        "rejection_diagnostic",
    }
    assert [set(attempt) for attempt in attempts] == [
        attempt_audit_fields,
        attempt_audit_fields,
    ]
    assert [attempt["response_schema_admitted"] for attempt in attempts] == [
        True,
        True,
    ]
    assert [
        attempt["semantic_materialization_admitted"] for attempt in attempts
    ] == [False, None]
    raw_output_hashes = [
        hashlib.sha256(
            json.dumps(
                output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for output in raw_outputs
    ]
    assert raw_outputs[0] == raw_outputs[1]
    assert len(set(raw_output_hashes)) == 1

    def expected_overlap_diagnostic(remaining: int) -> dict[str, object]:
        group_index, item_index = overlap_details["group_index"], overlap_details["item_index"]
        requirement_path = f"$.requirements[{group_index}].input_obligation_indexes[{item_index}]"
        disposition_path = "$.input_dispositions[0]"
        return {
            "error_code": "REQUIREMENT_COVERAGE_INPUT_DISPOSITION_INVALID",
            "phase": "semantic_materialization",
            "failure_category": (
                "obligation_already_accounted_and_dispositioned"
            ),
            "json_path": f"{disposition_path}.input_obligation_index",
            "input_obligation_index": overlap_details["index"],
            "allowed_dispositions": overlap_details["allowed_dispositions"],
            "conflicting_requirement_locations": [
                {
                    "requirement_item_index": group_index,
                    "requirement_item_json_path": f"$.requirements[{group_index}]",
                    "input_obligation_item_index": item_index,
                    "input_obligation_json_path": requirement_path,
                }
            ],
            "disposition_item_index": 0,
            "disposition_item_json_path": disposition_path,
            "correction_directions": [
                {
                    "direction": (
                        "keep_requirement_accounting_remove_input_disposition"
                    ),
                    "remove_json_paths": [disposition_path],
                    "retain_json_paths": [requirement_path],
                    "conditions": {
                        "listed_requirement_accounting_is_retained": True,
                        "complete_corrected_projection_must_pass_"
                        "existing_validation": True,
                    },
                    "locally_applicable": True,
                },
                {
                    "direction": (
                        "keep_allowed_disposition_remove_requirement_accounting"
                    ),
                    "remove_json_paths": [requirement_path],
                    "retain_json_paths": [disposition_path],
                    "conditions": {
                        "observed_disposition_is_allowed": True,
                        "disposition_reason_is_non_empty_after_strip": True,
                        "affected_requirements_remain_non_empty": True,
                        "empty_requirement_item_indexes_after_removal": [],
                        "all_input_obligations_remain_accounted": True,
                        "missing_input_obligation_indexes_after_removal": [],
                        "complete_corrected_projection_must_pass_"
                        "existing_validation": True,
                    },
                    "locally_applicable": True,
                },
            ],
            "bounded_correction_attempts_remaining": remaining,
        }

    assert [
        attempt["rejection_diagnostic"] for attempt in attempts
    ] == [
        expected_overlap_diagnostic(1),
        {
            "error_code": (
                "REQUIREMENT_COVERAGE_PROJECTION_CORRECTION_NO_PROGRESS"
            ),
            "phase": "correction_progress_validation",
            "failure_category": "canonical_projection_unchanged",
            "json_path": "$",
            "first_rejected_projection_canonical_sha256": raw_output_hashes[0],
            "correction_projection_canonical_sha256": raw_output_hashes[1],
            "structured_output_hash_basis": "canonical_json_utf8",
            "bounded_correction_attempts_remaining": 0,
        },
    ]
    assert [
        (
            attempt["requirement_count"],
            attempt["relation_count"],
            attempt["input_disposition_count"],
        )
        for attempt in attempts
    ] == [(overlap_details["requirement_count"], None, 1)] * 2
    assert [
        attempt["structured_output_canonical_sha256"] for attempt in attempts
    ] == raw_output_hashes
    assert evidence["budget_summary"]["semantic_provider_failure_codes"] == [
        "REQUIREMENT_COVERAGE_PROJECTION_CORRECTION_NO_PROGRESS"
    ]
    assert evidence["budget_summary"]["semantic_provider_failed_closed"] is True
    assert evidence["budget_summary"][
        "semantic_provider_failure_diagnostics"
    ] == [attempts[1]["rejection_diagnostic"]]

    assert evidence["readiness"] == "insufficient"
    assert evidence["authoritative_for_patch"] is False
    assert evidence["can_generate_patch"] is False
    assert evidence["verified_findings"] == []
    assert evidence["requirement_coverage"]["findings"] == []
    coverage = validated_requirement_coverage_view(evidence)
    assert coverage is not None
    assert coverage.edit_target_hint_evidence_ids == frozenset()
    assert implementation["blocking"] is True
    assert implementation["proposed_changes"] == []
    assert implementation["binding_registry"] == {}

    persisted_payloads = [scenario.store.read(ref) for ref in refs.values()]
    persisted_json = json.dumps(
        persisted_payloads,
        ensure_ascii=False,
        sort_keys=True,
    )
    rehydrated_evidence = EvidencePack.from_artifact_payload(
        evidence
    ).to_artifact_payload()
    assert rehydrated_evidence == evidence
    assert "previous_projection" not in json.dumps(
        rehydrated_evidence,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert "previous_projection" not in persisted_json
    assert all(raw_marker not in persisted_json for raw_marker in raw_markers)
    assert not {
        "patch_proposal",
        "proposal_review",
        "dry_run_result",
        "isolated_apply_result",
        "build_result",
        "repair_plan",
        "workflow_checkpoint",
        "workflow_summary",
    }.intersection(roles)
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_product_execution_blocks_removed_formal_coverage_before_patch_effect(
    tmp_path: Path,
) -> None:
    scenario = _formal_coverage_vertical_scenario(
        tmp_path,
        remove_requirement_coverage=True,
    )

    result = execute_product_execution(
        preflight_product_execution(
            scenario.request,
            scenario.adapters,
            scenario.store,
        ),
        scenario.adapters,
        scenario.store,
    )

    roles = tuple(item.artifact_role for item in result.store_attempts)
    assert (
        (
            scenario.trace["patch_calls"],
            scenario.trace["dry_run_calls"],
            scenario.trace["apply_calls"],
            scenario.trace["build_calls"],
        ),
        scenario.target.read_bytes(),
        scenario.canonical.read_bytes(),
    ) == (
        (0, 0, 0, 0),
        scenario.source_before,
        scenario.canonical_before,
    )
    assert result.execution_disposition == "domain_blocked"
    assert result.terminal_node == "blocked"
    assert roles == (
        "request_planning_report",
        "mod_planning_brief",
        "feature_spec",
        "evidence_pack",
        "implementation_plan",
        "blocked_report",
    )
    assert not {
        "patch_proposal",
        "proposal_review",
        "dry_run_result",
        "isolated_apply_result",
        "build_result",
        "workflow_checkpoint",
        "workflow_summary",
    }.intersection(roles)
    assert tuple(
        (event.effect_role, event.event_status) for event in result.effect_events
    ) == (
        ("evidence_retrieval", "started"),
        ("evidence_retrieval", "completed"),
    )


def test_product_execution_blocks_coverage_derived_forbidden_path_before_apply(
    tmp_path: Path,
) -> None:
    scenario = _formal_coverage_vertical_scenario(
        tmp_path,
        search_root="BepInEx/core",
    )

    result = execute_product_execution(
        preflight_product_execution(
            scenario.request,
            scenario.adapters,
            scenario.store,
        ),
        scenario.adapters,
        scenario.store,
    )

    roles = tuple(item.artifact_role for item in result.store_attempts)
    assert (
        scenario.trace["patch_calls"],
        scenario.trace["dry_run_calls"],
        scenario.trace["apply_calls"],
        scenario.trace["build_calls"],
    ) == (1, 0, 0, 0)
    assert scenario.target.read_bytes() == scenario.source_before
    assert scenario.canonical.read_bytes() == scenario.canonical_before
    assert result.execution_disposition == "domain_blocked"
    assert result.terminal_node == "blocked"
    assert any(blocker.code == "FORBIDDEN_PATHS" for blocker in result.blockers)
    assert "patch_proposal" in roles
    assert "proposal_review" in roles
    assert roles[-2:] == ("proposal_review", "blocked_report")
    assert not {
        "dry_run_result",
        "isolated_apply_result",
        "build_result",
        "workflow_checkpoint",
        "workflow_summary",
    }.intersection(roles)
    assert not {
        "dry_run",
        "isolated_apply",
        "build",
    }.intersection(event.effect_role for event in result.effect_events)


def test_public_product_executor_runs_exact_runtime_terminal_chain(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    source = Path(request.isolated_target_root, "Plugin.cs")
    source.write_text("public void Load() {}\n", encoding="utf-8")
    values = request.to_dict()
    values["requested_completion_scope"] = "runtime_workflow"
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    runtime_admission, runtime_settings, runtime_destination = (
        _runtime_owner_binding(tmp_path / "runtime")
    )
    values["input_bindings"][3] = ProductExecutionInputBindingV1.create(
        "runtime_admission", runtime_admission
    ).to_dict()
    request = ProductExecutionRequestV1.from_dict(values)
    base = _successful_compile_adapters(request)

    def runtime_evidence_factory(feature_id, build, feature_spec):
        criteria = tuple(str(item) for item in feature_spec["acceptance_criteria"])
        return build_real_runtime_evidence_payload(
            feature_id=feature_id,
            input_execution_mode=str(build["execution_mode"]),
            evidence_items=(
                RuntimeEvidenceItem(
                    item_type="runtime_log",
                    source="deterministic_product_runtime",
                    locator="runtime://product-001",
                    supports_acceptance=criteria,
                    confidence=1.0,
                    raw_excerpt="Runtime observed the requested behavior.",
                ),
            ),
            startup_loaded=True,
            gameplay_behavior_verified=True,
            verification_status="verified",
            raw_runtime_output="Runtime observed the requested behavior.",
            acceptance_criteria=criteria,
            runtime_verification_plan=tuple(
                str(item) for item in feature_spec["runtime_verification_plan"]
            ),
            plugin_marker_binding={
                "marker_source": "deterministic_product_runtime",
                "markers": [
                    {
                        "acceptance_criterion": criterion,
                        "marker": f"MARKER_{index}",
                        "observed": True,
                    }
                    for index, criterion in enumerate(criteria, start=1)
                ],
            },
        )

    adapters = replace(
        base,
        runtime_evidence_factory=_bind_owner_validated_runtime_evidence_factory(
            runtime_evidence_factory,
            settings=runtime_settings,
            allow_game_launch=True,
            runtime_plugin_destination=runtime_destination,
            allow_game_directory_write=True,
            expected_runtime_admission=runtime_admission,
        ),
    )
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "completed"
    assert result.execution_disposition == "succeeded"
    assert result.terminal_node == "done"
    assert tuple(item.artifact_role for item in result.store_attempts[-4:]) == (
        "build_result",
        "runtime_evidence",
        "eval_report",
        "workflow_summary",
    )
    assert all(
        item.store_result.outcome == "committed"
        for item in result.store_attempts
    )
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_public_product_executor_reaches_runtime_binding_gap_store_role(
    tmp_path: Path,
) -> None:
    request, store, adapters = _runtime_request_and_adapters(
        tmp_path, diagnosed_binding_gap=True
    )

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.execution_disposition == "domain_blocked"
    assert "binding_resolution" in tuple(
        item.artifact_role for item in result.store_attempts
    )
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_product_request_rejects_binding_order_drift(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    values = request.to_dict()
    values["input_bindings"] = list(reversed(values["input_bindings"]))

    with pytest.raises(ProductExecutionContractError, match="exact ordered"):
        ProductExecutionRequestV1.from_dict(values)


def test_product_preflight_delegates_owner_records_and_rejects_scope_drift(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    values = request.to_dict()
    drifted = dict(values["input_bindings"][1]["canonical_payload"])
    drifted["scope"] = "another_run"
    values["input_bindings"][1] = ProductExecutionInputBindingV1.create(
        "source_disclosure", drifted
    ).to_dict()
    drifted_request = ProductExecutionRequestV1.from_dict(values)

    preflight = preflight_product_execution(
        drifted_request,
        _blocked_evidence_adapters(),
        store,
    )
    assert preflight.preflight_status == "blocked"
    assert preflight.failure_stage == "admission"
    assert "Source disclosure record changed" in preflight.blockers[0].message
    assert not store.root.exists()


def test_domain_blocked_result_is_not_overridden_by_completion_classification(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    values = request.to_dict()
    values["completion_classification"] = [["requested_behavior", True]]
    request = ProductExecutionRequestV1.from_dict(values)
    adapters = _blocked_evidence_adapters()

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "completed"
    assert result.execution_disposition == "domain_blocked"
    assert result.failure_stage is None
    assert result.validation_facts.completion_classification_status == "not_applicable"


def test_product_store_first_noncommitted_attempt_stops_real_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, store = _request(tmp_path)
    original = store.write_with_result

    def write_with_result(**kwargs):
        if kwargs["artifact_type"] == "patch_proposal":
            return ArtifactWriteResult(
                outcome="blocked_before_commit",
                feature_id=kwargs["feature_id"],
                artifact_type="patch_proposal",
                artifact_hash=None,
                artifact_ref=None,
                revision_path=None,
                latest_alias_path=None,
                pending_marker_path=None,
                commit_started=False,
                revision_state="absent",
                latest_alias_state="absent",
                pending_marker_state="absent",
                failure_stage="admission_validation",
                failure_class="ArtifactStoreError",
                failure_message="injected precommit failure",
                recovery_available=False,
                recovery_authorized=False,
            )
        return original(**kwargs)

    monkeypatch.setattr(store, "write_with_result", write_with_result)
    adapters = _successful_compile_adapters(request)

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "partial"
    assert result.execution_disposition == "partial"
    assert result.failure_stage == "patch_proposal_store"
    assert result.store_attempts[-1].artifact_role == "patch_proposal"
    assert result.store_attempts[-1].store_result.outcome == "blocked_before_commit"
    assert all(
        item.artifact_role not in {
            "proposal_review",
            "dry_run_result",
            "isolated_apply_result",
            "build_result",
            "workflow_checkpoint",
        }
        for item in result.store_attempts
    )
    assert tuple(item.artifact_role for item in result.outputs) == tuple(
        item.artifact_role for item in result.store_attempts[:-1]
    )


def test_store_failure_is_not_overridden_by_completion_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, store = _request(tmp_path)
    values = request.to_dict()
    values["completion_classification"] = [["requested_behavior", True]]
    request = ProductExecutionRequestV1.from_dict(values)
    original = store.write_with_result

    def write_with_result(**kwargs):
        if kwargs["artifact_type"] == "patch_proposal":
            return ArtifactWriteResult(
                outcome="blocked_before_commit",
                feature_id=kwargs["feature_id"],
                artifact_type="patch_proposal",
                artifact_hash=None,
                artifact_ref=None,
                revision_path=None,
                latest_alias_path=None,
                pending_marker_path=None,
                commit_started=False,
                revision_state="absent",
                latest_alias_state="absent",
                pending_marker_state="absent",
                failure_stage="admission_validation",
                failure_class="ArtifactStoreError",
                failure_message="injected precommit failure",
                recovery_available=False,
                recovery_authorized=False,
            )
        return original(**kwargs)

    monkeypatch.setattr(store, "write_with_result", write_with_result)
    adapters = _successful_compile_adapters(request)

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.failure_stage == "patch_proposal_store"
    assert result.execution_disposition == "partial"
    assert result.validation_facts.completion_classification_status == "not_applicable"


def test_domain_blocked_output_readback_failure_is_platform_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, store = _request(tmp_path)
    original = store.read

    def read(ref):
        if ref.artifact_type == "blocked_report" and any(
            frame.function == "validate_product_execution_outputs"
            for frame in inspect.stack()
        ):
            raise OSError("injected committed BlockedReport readback failure")
        return original(ref)

    monkeypatch.setattr(store, "read", read)
    adapters = _blocked_evidence_adapters()
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "partial"
    assert result.execution_disposition == "partial"
    assert result.failure_stage == "post_execution_validation"
    assert result.terminal_node == "blocked"
    assert result.validation_facts.output_readback_status == "failed"
    assert result.outputs[-1].artifact_role == "blocked_report"
    assert all(
        item.store_result.outcome == "committed"
        for item in result.store_attempts
    )


def test_graph_node_exception_uses_exact_current_node_after_completed_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, store = _request(tmp_path)
    adapters = _successful_compile_adapters(request)

    def fail_create_implementation_plan(*_args, **_kwargs):
        raise RuntimeError("injected plan construction failure")

    monkeypatch.setattr(
        "src.mod_agent.workflow.graph._run_create_implementation_plan",
        fail_create_implementation_plan,
    )
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.failure_stage == "create_implementation_plan_execution"
    assert result.execution_disposition == "partial"
    assert result.effect_events[-1].effect_role == "evidence_retrieval"
    assert result.effect_events[-1].event_status == "completed"
    assert all(
        item.artifact_role != "implementation_plan"
        for item in result.store_attempts
    )


def test_public_product_executor_preserves_repair_attempt_prefixes(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    request_values = request.to_dict()
    request_values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(request_values)
    adapters = _successful_compile_adapters(request, build_exit_codes=[1, 0], physical_apply=True)

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.execution_disposition == "succeeded", (result.failure_stage, [item.message for item in result.blockers])
    build_attempts = [
        item for item in result.store_attempts if item.artifact_role == "build_result"
    ]
    repair_attempts = [
        item for item in result.store_attempts if item.artifact_role == "repair_plan"
    ]
    assert [item.repair_attempt for item in build_attempts] == [0, 1]
    assert [item.repair_attempt for item in repair_attempts] == [1]
    assert result.store_attempts[-1].artifact_role == "workflow_checkpoint"
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_product_repair_review_uses_ephemeral_review_context_only_on_repair(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    request_values = request.to_dict()
    request_values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(request_values)
    adapters = _successful_compile_adapters(request, build_exit_codes=[1, 0], physical_apply=True)
    review_context_calls: list[str] = []

    def proposal_review_context_factory(state, implementation, evidence_pack):
        review_context_calls.append(str(state["current_stage"]))
        return implementation, evidence_pack

    adapters = replace(
        adapters,
        proposal_review_context_factory=proposal_review_context_factory,
    )

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.execution_disposition == "succeeded", (result.failure_stage, [item.message for item in result.blockers])
    assert review_context_calls == ["patch_proposal"]
    assert [
        item.repair_attempt
        for item in result.store_attempts
        if item.artifact_role == "proposal_review"
    ] == [0, 1]


def test_public_product_executor_returns_repair_budget_exhaustion_result(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    request_values = request.to_dict()
    request_values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(request_values)
    adapters = _successful_compile_adapters(request, build_exit_codes=[1, 1, 1], physical_apply=True)

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "completed", (result.failure_stage, [item.message for item in result.blockers])
    assert result.execution_disposition == "domain_blocked"
    assert result.failure_stage is None
    selected = [
        (item.node_name, item.repair_attempt, item.artifact_role)
        for item in result.store_attempts
        if item.node_name in {"build", "create_repair_plan", "blocked"}
    ]
    assert selected == [
        ("build", 0, "build_result"),
        ("create_repair_plan", 1, "repair_plan"),
        ("create_repair_plan", 1, "evidence_pack"),
        ("create_repair_plan", 1, "implementation_plan"),
        ("build", 1, "build_result"),
        ("create_repair_plan", 2, "repair_plan"),
        ("create_repair_plan", 2, "evidence_pack"),
        ("create_repair_plan", 2, "implementation_plan"),
        ("build", 2, "build_result"),
        ("create_repair_plan", 2, "repair_plan"),
        ("blocked", 2, "blocked_report"),
    ]
    assert all(
        item.store_result.outcome == "committed" for item in result.store_attempts
    )
    assert result.blockers
    assert all(item.owner == "domain" for item in result.blockers)
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("scenario", "expected_tail"),
    (
        (
            "review",
            (
                ("review_proposal", 0, "proposal_review"),
                ("create_repair_plan", 1, "repair_plan"),
                ("blocked", 1, "blocked_report"),
            ),
        ),
        (
            "review_max",
            (
                ("review_proposal", 2, "proposal_review"),
                ("create_repair_plan", 2, "repair_plan"),
                ("blocked", 2, "blocked_report"),
            ),
        ),
        (
            "runtime_terminal",
            (
                ("collect_runtime_evidence", 0, "runtime_evidence"),
                ("create_repair_plan", 0, "repair_plan"),
                ("blocked", 0, "blocked_report"),
            ),
        ),
        (
            "runtime_max",
            (
                ("collect_runtime_evidence", 2, "runtime_evidence"),
                ("create_repair_plan", 2, "repair_plan"),
                ("blocked", 2, "blocked_report"),
            ),
        ),
        (
            "runtime_terminal_max",
            (
                ("collect_runtime_evidence", 2, "runtime_evidence"),
                ("create_repair_plan", 2, "repair_plan"),
                ("blocked", 2, "blocked_report"),
            ),
        ),
    ),
)
def test_owner_produced_terminal_repair_shapes_use_exact_attempt_transition(
    tmp_path: Path,
    scenario: str,
    expected_tail: tuple[tuple[str, int, str], ...],
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario=scenario)

    assert result.platform_status == "completed"
    assert result.execution_disposition == "domain_blocked"
    assert result.failure_stage is None
    selected = tuple(
        (item.node_name, item.repair_attempt, item.artifact_role)
        for item in result.store_attempts
        if item.node_name in {
            "review_proposal",
            "collect_runtime_evidence",
            "create_repair_plan",
            "blocked",
        }
    )
    assert selected[-3:] == expected_tail
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "success"),
        ("retry_exhausted", False),
        ("retry_count", 1),
        ("retry_budget", 3),
        ("retry_budget_remaining", 1),
        ("allowed_next_stage", "generate_patch_proposal"),
        ("blockers", []),
    ),
)
def test_strict_replay_rejects_tampered_exhausted_repair_plan(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    values = request.to_dict()
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(values)
    adapters = _successful_compile_adapters(request, build_exit_codes=[1, 1, 1], physical_apply=True)
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )
    projection = copy.deepcopy(result.to_dict())
    readback = projection["validation_facts"]["details"][0]["payload"]
    plan = next(
        item
        for item in reversed(readback)
        if item["artifact_type"] == "repair_plan"
    )
    if field == "status":
        plan["status"] = value
    else:
        plan["data"][field] = value

    with pytest.raises(ProductExecutionContractError, match="lifecycle predicate"):
        ProductExecutionResultV1.from_dict(projection)


@pytest.mark.parametrize(
    ("scenario", "replacement_attempt"),
    (
        ("review", 0),
        ("review_max", 1),
        ("runtime_terminal", 1),
        ("runtime_max", 1),
        ("repair", 1),
    ),
)
def test_strict_replay_rejects_terminal_repair_attempt_drift_by_source(
    tmp_path: Path,
    scenario: str,
    replacement_attempt: int,
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario=scenario)
    projection = copy.deepcopy(result.to_dict())
    terminal_nodes = {"create_repair_plan", "blocked"}
    for collection in (projection["store_attempts"], projection["outputs"]):
        last_plan = max(
            index
            for index, item in enumerate(collection)
            if item["node_name"] == "create_repair_plan"
        )
        assert collection[last_plan]["repair_attempt"] != replacement_attempt
        collection[last_plan]["repair_attempt"] = replacement_attempt
        if collection[-1]["node_name"] == "blocked":
            collection[-1]["repair_attempt"] = replacement_attempt
    assert projection["store_attempts"][-1]["node_name"] in terminal_nodes

    with pytest.raises(ProductExecutionContractError):
        ProductExecutionResultV1.from_dict(projection)


@pytest.mark.parametrize("outcome", ["blocked_before_commit", "partial"])
def test_repair_source_blocked_report_write_failure_retains_cause_without_fabricating_output(tmp_path, monkeypatch, outcome):
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_bytes(b"public void Load() {}\n")
    values = request.to_dict()
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(source_snapshot(Path(request.isolated_target_root)))
    request = ProductExecutionRequestV1.from_dict(values)
    adapters = _successful_compile_adapters(request, build_exit_codes=[1], physical_apply=True)
    original_apply = adapters.isolated_apply_factory

    def incomplete_apply_receipt(*args):
        result = original_apply(*args)
        result.pop("changes")
        return result

    adapters = replace(adapters, isolated_apply_factory=incomplete_apply_receipt)
    _install_actual_store_failure(monkeypatch, store, artifact_role="blocked_report", outcome=outcome)
    result = execute_product_execution(preflight_product_execution(request, adapters, store), adapters, store)
    assert result.platform_status == "partial"
    assert result.failure_stage == "blocked_report_store"
    assert result.store_attempts[-1].store_result.outcome == outcome
    assert result.store_attempts[-1].artifact_role == "blocked_report"
    assert not any(output.artifact_role == "blocked_report" for output in result.outputs)
    assert {blocker.code for blocker in result.blockers} == {
        "PRODUCT_STORE_NOT_COMMITTED", "PRODUCT_REPAIR_SOURCE_REANCHORING_REQUIRED"}
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result
    without_cause = result.to_dict()
    without_cause["blockers"] = [blocker for blocker in without_cause["blockers"]
                                if blocker["code"] != "PRODUCT_REPAIR_SOURCE_REANCHORING_REQUIRED"]
    with pytest.raises(ProductExecutionContractError, match="lifecycle predicate"):
        ProductExecutionResultV1.from_dict(without_cause)


@pytest.mark.parametrize("mutation", ("missing", "wrong_attempt"))
def test_strict_replay_rejects_missing_or_wrong_attempt_blocked_report(
    tmp_path: Path,
    mutation: str,
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario="review")
    projection = copy.deepcopy(result.to_dict())
    readback = projection["validation_facts"]["details"][0]["payload"]
    if mutation == "missing":
        projection["store_attempts"].pop()
        projection["outputs"].pop()
        readback.pop()
    else:
        projection["store_attempts"][-1]["repair_attempt"] = 0
        projection["outputs"][-1]["repair_attempt"] = 0

    with pytest.raises(ProductExecutionContractError):
        ProductExecutionResultV1.from_dict(projection)


def test_strict_replay_rejects_unreadable_terminal_repair_without_blocked_report(
    tmp_path: Path,
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario="review")
    projection = copy.deepcopy(result.to_dict())
    projection["store_attempts"].pop()
    projection["outputs"].pop()
    validation = projection["validation_facts"]
    validation["output_readback_status"] = "failed"
    validation["failure_class"] = "OSError"
    validation["failure_message"] = "injected terminal readback failure"
    validation["details"][0]["payload"] = None
    projection["platform_status"] = "partial"
    projection["execution_disposition"] = "partial"
    projection["failure_stage"] = "post_execution_validation"
    projection["blockers"].append(
        {
            "contract_version": "1",
            "owner": "platform",
            "stage": "post_execution_validation",
            "code": "PRODUCT_OUTPUT_READBACK_FAILED",
            "message": "Committed output readback failed.",
            "source_node": "product_executor",
        }
    )

    with pytest.raises(
        ProductExecutionContractError,
        match="lacks its committed BlockedReport",
    ):
        ProductExecutionResultV1.from_dict(projection)


def test_strict_replay_rejects_store_after_blocked_terminal(tmp_path: Path) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario="review")
    projection = copy.deepcopy(result.to_dict())
    extra_attempt = copy.deepcopy(projection["store_attempts"][-1])
    extra_output = copy.deepcopy(projection["outputs"][-1])
    extra_envelope = copy.deepcopy(
        projection["validation_facts"]["details"][0]["payload"][-1]
    )
    extra_attempt["sequence"] = len(projection["store_attempts"])
    extra_output["sequence"] = len(projection["outputs"])
    projection["store_attempts"].append(extra_attempt)
    projection["outputs"].append(extra_output)
    projection["validation_facts"]["details"][0]["payload"].append(
        extra_envelope
    )

    with pytest.raises(ProductExecutionContractError):
        ProductExecutionResultV1.from_dict(projection)


def test_strict_replay_rejects_higher_attempt_effect_after_blocked_terminal(
    tmp_path: Path,
) -> None:
    result = _execute_store_role_scenario(tmp_path, scenario="review")
    projection = copy.deepcopy(result.to_dict())
    next_sequence = len(projection["effect_events"])
    for offset, status in enumerate(("started", "completed")):
        projection["effect_events"].append(
            {
                "contract_version": "1",
                "sequence": next_sequence + offset,
                "phase": "repair",
                "repair_attempt": 2,
                "effect_role": "build",
                "event_status": status,
                "failure_class": None,
                "failure_message": None,
            }
        )

    with pytest.raises(
        ProductExecutionContractError,
        match="continues after its blocked repair terminal",
    ):
        ProductExecutionResultV1.from_dict(projection)


@pytest.mark.parametrize("outcome", ("blocked_before_commit", "partial"))
def test_repair_plan_store_failure_keeps_c5_truth_before_payload_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    values = request.to_dict()
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(values)
    _install_actual_store_failure(
        monkeypatch,
        store,
        artifact_role="repair_plan",
        outcome=outcome,
        occurrence=3,
    )
    adapters = _successful_compile_adapters(request, build_exit_codes=[1, 1, 1], physical_apply=True)

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "partial"
    assert result.execution_disposition == "partial"
    assert result.failure_stage == "repair_plan_store"
    assert result.store_attempts[-1].artifact_role == "repair_plan"
    assert result.store_attempts[-1].store_result.outcome == outcome
    assert tuple(item.artifact_role for item in result.outputs) == tuple(
        item.artifact_role for item in result.store_attempts[:-1]
    )
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("outcome", ("blocked_before_commit", "partial"))
def test_blocked_report_store_failure_validates_committed_repair_plan_then_keeps_c5_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    values = request.to_dict()
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(values)
    _install_actual_store_failure(
        monkeypatch,
        store,
        artifact_role="blocked_report",
        outcome=outcome,
    )
    adapters = _successful_compile_adapters(request, build_exit_codes=[1, 1, 1], physical_apply=True)
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "partial"
    assert result.execution_disposition == "partial"
    assert result.failure_stage == "blocked_report_store"
    assert result.store_attempts[-1].store_result.outcome == outcome
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result

    projection = copy.deepcopy(result.to_dict())
    readback = projection["validation_facts"]["details"][0]["payload"]
    plan = next(
        item
        for item in reversed(readback)
        if item["artifact_type"] == "repair_plan"
    )
    plan["data"]["retry_exhausted"] = False
    with pytest.raises(ProductExecutionContractError, match="lifecycle predicate"):
        ProductExecutionResultV1.from_dict(projection)


def test_committed_terminal_repair_plan_readback_failure_is_post_validation_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, store = _request(tmp_path)
    Path(request.isolated_target_root, "Plugin.cs").write_text(
        "public void Load() {}\n", encoding="utf-8"
    )
    values = request.to_dict()
    values["isolated_source_snapshot_sha256"] = source_snapshot_sha256(
        source_snapshot(Path(request.isolated_target_root))
    )
    request = ProductExecutionRequestV1.from_dict(values)
    original_read = store.read
    state = {"repair_reads": 0}

    def fail_final_repair_readback(ref):
        if ref.artifact_type == "repair_plan" and any(
            frame.function == "validate_product_execution_outputs"
            for frame in inspect.stack()
        ):
            state["repair_reads"] += 1
            if state["repair_reads"] == 3:
                raise OSError("injected terminal RepairPlan readback failure")
        return original_read(ref)

    monkeypatch.setattr(store, "read", fail_final_repair_readback)
    adapters = _successful_compile_adapters(request, build_exit_codes=[1, 1, 1], physical_apply=True)

    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "partial"
    assert result.execution_disposition == "partial"
    assert result.failure_stage == "post_execution_validation"
    assert result.validation_facts.output_readback_status == "failed"
    assert result.store_attempts[-2].artifact_role == "repair_plan"
    assert result.store_attempts[-1].artifact_role == "blocked_report"
    assert all(
        item.store_result.outcome == "committed" for item in result.store_attempts
    )
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


@pytest.mark.parametrize("outcome", ("blocked_before_commit", "partial"))
@pytest.mark.parametrize(
    ("scenario", "artifact_role", "occurrence"),
    (
        ("blocked", "request_planning_report", 1),
        ("blocked", "mod_planning_brief", 1),
        ("blocked", "feature_spec", 1),
        ("blocked", "evidence_pack", 1),
        ("compile", "implementation_plan", 1),
        ("compile", "patch_proposal", 1),
        ("compile", "proposal_review", 1),
        ("compile", "dry_run_result", 1),
        ("compile", "isolated_apply_result", 1),
        ("compile", "build_result", 1),
        ("compile", "workflow_checkpoint", 1),
        ("repair", "repair_plan", 3),
        ("binding", "binding_resolution", 1),
        ("runtime", "runtime_evidence", 1),
        ("runtime", "eval_report", 1),
        ("runtime", "workflow_summary", 1),
        ("repair", "blocked_report", 1),
    ),
)
def test_each_active_public_store_role_preserves_real_c5_failure_and_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    artifact_role: str,
    occurrence: int,
    outcome: str,
) -> None:
    declared_roles = tuple(
        role for _node, role in PRODUCT_EXECUTION_DECLARATION.node_store_roles
    )
    assert artifact_role in declared_roles
    baseline = _execute_store_role_scenario(
        tmp_path / "baseline",
        scenario=scenario,
    )
    matching = [
        index
        for index, item in enumerate(baseline.store_attempts)
        if item.artifact_role == artifact_role
    ]
    assert len(matching) >= occurrence
    stop_index = matching[occurrence - 1]

    result = _execute_store_role_scenario(
        tmp_path / "failure",
        scenario=scenario,
        monkeypatch=monkeypatch,
        failure_role=artifact_role,
        failure_outcome=outcome,
        failure_occurrence=occurrence,
    )

    attempt_signature = lambda item: (
        item.node_name,
        item.artifact_role,
        item.repair_attempt,
    )
    output_signature = lambda item: (
        item.node_name,
        item.artifact_role,
        item.repair_attempt,
    )
    assert tuple(map(attempt_signature, result.store_attempts)) == tuple(
        map(attempt_signature, baseline.store_attempts[: stop_index + 1])
    )
    assert result.store_attempts[-1].artifact_role == artifact_role
    assert result.store_attempts[-1].store_result.outcome == outcome
    assert tuple(map(output_signature, result.outputs)) == tuple(
        map(attempt_signature, baseline.store_attempts[:stop_index])
    )
    assert tuple(item.to_dict() for item in result.effect_events) == tuple(
        item.to_dict()
        for item in baseline.effect_events[: len(result.effect_events)]
    )
    assert result.failure_stage == f"{artifact_role}_store"
    expected_partial = bool(result.outputs) or any(
        item.event_status == "started" for item in result.effect_events
    ) or outcome == "partial"
    assert result.platform_status == ("partial" if expected_partial else "blocked")
    assert result.execution_disposition == (
        "partial" if expected_partial else "platform_blocked"
    )
    assert ProductExecutionResultV1.from_dict(result.to_dict()) == result


def test_product_store_role_matrix_excludes_only_unreachable_optional_critic(
    tmp_path: Path,
) -> None:
    assert ("review_proposal", "critic_assessment") in (
        PRODUCT_EXECUTION_DECLARATION.node_store_roles
    )
    assert "proposal_critic" not in ProductExecutionAdaptersV1.__dataclass_fields__
    request, _ = _request(tmp_path)
    adapters = _successful_compile_adapters(request)
    assert ("planning_target_selection_factory", True) in adapters.topology()


def test_exact_diff_rejection_is_domain_terminal_without_apply_effect(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    source = Path(request.isolated_target_root, "Plugin.cs")
    before = source.read_bytes()
    base = _successful_compile_adapters(request)

    def reject_exact_diff(*_args):
        raise ExactDiffApprovalRejectedError("operator rejected exact diff")

    adapters = replace(base, isolated_apply_factory=reject_exact_diff)
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )

    assert result.platform_status == "completed"
    assert result.execution_disposition == "domain_blocked"
    assert source.read_bytes() == before
    assert tuple(item.artifact_role for item in result.store_attempts[-2:]) == (
        "isolated_apply_result",
        "blocked_report",
    )
    assert all(
        item.store_result.outcome == "committed"
        for item in result.store_attempts[-2:]
    )
    assert all(
        item.effect_role not in {"isolated_apply", "build"}
        for item in result.effect_events
    )


def test_strict_result_replay_rejects_store_order_drift(tmp_path: Path) -> None:
    request, store = _request(tmp_path)
    result = execute_product_execution(
        preflight_product_execution(request, _blocked_evidence_adapters(), store),
        _blocked_evidence_adapters(),
        store,
    )
    projection = result.to_dict()
    projection["store_attempts"][0], projection["store_attempts"][1] = (
        projection["store_attempts"][1],
        projection["store_attempts"][0],
    )

    with pytest.raises(ProductExecutionContractError):
        ProductExecutionResultV1.from_dict(projection)


def test_strict_result_replay_rejects_terminal_stage_drift(tmp_path: Path) -> None:
    request, store = _request(tmp_path)
    adapters = _blocked_evidence_adapters()
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )
    projection = result.to_dict()
    projection["terminal_node"] = "build"

    with pytest.raises(ProductExecutionContractError):
        ProductExecutionResultV1.from_dict(projection)


def test_strict_result_replay_rejects_undeclared_failure_stage(
    tmp_path: Path,
) -> None:
    request, store = _request(tmp_path)
    adapters = _blocked_evidence_adapters()
    result = execute_product_execution(
        preflight_product_execution(request, adapters, store), adapters, store
    )
    projection = result.to_dict()
    projection["failure_stage"] = "caller_selected_failure"

    with pytest.raises(ProductExecutionContractError):
        ProductExecutionResultV1.from_dict(projection)
