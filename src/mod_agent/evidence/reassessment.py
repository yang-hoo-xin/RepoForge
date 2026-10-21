"""Reassess unchanged requirements over old current plus newly retrieved Evidence."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.evidence.identity import feature_spec_payload_sha256
from src.mod_agent.evidence.models import EvidencePack
from src.mod_agent.evidence.requirement_coverage import validated_requirement_coverage_view
from src.mod_agent.evidence.requirement_coverage_provider import (
    _merge_source_augmentation,
    _strict_source_augmentation,
    _valid_source_augmentation_item,
    build_requirement_coverage_pack,
)
from src.mod_agent.fresh_planning_continuation import _validate_source_items
from src.mod_agent.proposal.behavior_feedback_view import project_execution_reference_paths


def gap_investigation_targets(projection, old):
    """Use the accepted relation to bind gap identity to requirement ownership."""
    relations = {row["relation_id"]: row for row in projection["relations"]}
    targets = []
    for finding in old.requirement_coverage["findings"]:
        relation = relations[finding["relation_id"]]
        parents = list(finding["parent_requirement_ids"])
        if set(parents) != set(relation["parent_requirement_ids"]):
            raise ValueError("PATCH_REASSESSMENT_GAP_RELATION_CHANGED")
        for gap_index, gap in enumerate(finding["unresolved_gaps"]):
            identity = {"relation_id": finding["relation_id"], "gap": gap,
                        "parent_requirement_ids": parents}
            digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            targets.append({"gap_id": "gap_" + digest[:20],
                "relation_index": relation["provenance"]["relation_index"],
                "gap_index": gap_index})
    return targets


def gap_requirement_bindings(projection, old):
    """Derive the allowed parents; no query text or model IDs grant ownership."""
    parents = {row["provenance"]["relation_index"]: tuple(row["parent_requirement_ids"])
               for row in projection["relations"]}
    return {row["gap_id"]: parents[row["relation_index"]]
            for row in gap_investigation_targets(projection, old)}


def merge_reobserved_source_item(prior, current):
    """Keep one physical source observation despite query-dependent lexical labels."""
    left, right = prior.to_dict(), current.to_dict()
    if left == right:
        return prior
    # Lexical window labels are not declarations; only the displayed search label may vary.
    # Why: the existing rg identity binds path/hash/span, while _best_symbol depends on query terms.
    declaration_fields = ("declaring_type", "declaring_type_full_name", "method_name", "method_kind",
                          "parameter_types", "return_type", "declared_members", "declared_member_types")
    for row in (left, right):
        if (row["source_type"] != "target_source" or row["retrieval_method"] != "rg_live_read"
                or not row.get("source_hash") or not row.get("source_ref") or not row.get("excerpt")
                or any(row.get(key) is not None for key in declaration_fields)
                or row.get("verified_member_accesses")
                or row["claim_supported"] != (
                    f"Live source contains request terms near {row['symbol'] or row['source_path']}.")):
            raise ValueError("PATCH_REASSESSMENT_EVIDENCE_ID_CONFLICT")
    if {k: v for k, v in left.items() if k not in {"symbol", "claim_supported"}} != {
            k: v for k, v in right.items() if k not in {"symbol", "claim_supported"}}:
        raise ValueError("PATCH_REASSESSMENT_EVIDENCE_ID_CONFLICT")
    return prior


def _successful_source_queries(pack: EvidencePack) -> bool:
    rows = pack.budget_summary.get("per_query_execution")
    return (
        pack.budget_summary.get("all_queries_executed") is True
        and isinstance(rows, list) and bool(rows)
        and all(isinstance(row, dict) and row.get("retrieval_status") in {
            "source_observations_returned", "no_matches", "requested_source_no_matches",
        } for row in rows)
    )


def _initial_source_queries_closed(pack: EvidencePack) -> bool:
    """Admit bounded initial no-term outcomes so reassessment can choose better queries."""
    rows = pack.budget_summary.get("per_query_execution")
    return (
        isinstance(rows, list) and bool(rows)
        and all(isinstance(row, dict) and row.get("retrieval_status") in {
            "source_observations_returned", "no_matches", "requested_source_no_matches",
            "query_terms_missing",
        } for row in rows)
    )


def make_reassessment_api_augmentation(*, source, tool_executor, run_id, max_method_inspections=3):
    """Adapt already authorized metadata/IL tools, with no independent model budget.

    Query text comes from the existing reassessment query turn. Method identities
    come only from that tool search; no application Hook or method is supplied here.
    The caller must bind the source registry and inspector in its approved inputs.
    """
    if type(max_method_inspections) is not int or not 1 <= max_method_inspections <= 3:
        raise ValueError("DEPENDENCY_API_INSPECTION_LIMIT_INVALID")
    if source.source_type != "decompiled_api" or source.source_authority not in {"authoritative", "supporting"}:
        raise ValueError("DEPENDENCY_API_SOURCE_INVALID")

    def augment(_projection, retrieved):
        status, observations, items = "unavailable", [], []
        reason = None
        visible_metadata = {"methods": [], "types": [], "fields": []}
        query_rows = retrieved.budget_summary["reassessment"]["queries"]
        structured_metadata_queries = any("metadata_queries" in row for row in query_rows)
        explicit_metadata_queries = tuple(dict.fromkeys(
            query.strip()
            for row in query_rows
            for query in row.get("metadata_queries", ())
            if isinstance(query, str) and query.strip()
        ))
        queries = (explicit_metadata_queries if structured_metadata_queries else
                   tuple(row["query"] for row in query_rows))
        if not {"search_dotnet_metadata", "inspect_method_il"}.issubset(source.allowed_tools):
            status, reason = "not_authorized", "DEPENDENCY_API_TOOLS_NOT_AUTHORIZED"
        elif not queries:
            status = "not_requested" if structured_metadata_queries else "unavailable"
            reason = ("DEPENDENCY_API_METADATA_QUERY_NOT_REQUESTED"
                      if structured_metadata_queries else "DEPENDENCY_API_QUERY_REQUIRED")
        else:
            tool_executor.begin_run(run_id=run_id)
            prefix = "reassessment_api_" + uuid4().hex
            try:
                search = tool_executor.execute(source=source, observation_id=prefix + "_search", action={
                    "tool": "search_dotnet_metadata", "arguments": {
                        "queries": list(queries), "result_kinds": ["methods", "types", "fields"],
                        "limit_per_kind": max_method_inspections}})
                observations.append(search)
                visible_metadata = {kind: [{key: deepcopy(row.get(key)) for key in (
                    "symbol", "metadata_token", "matched_queries", "assembly_id")}
                    for row in search.data[kind]] for kind in visible_metadata}
                for index, method in enumerate(search.data["methods"][:max_method_inspections]):
                    observed = tool_executor.execute(source=source, observation_id=f"{prefix}_inspect_{index}",
                        action={"tool": "inspect_method_il", "arguments": {
                            "candidate_ref": method["candidate_ref"]}})
                    observations.append(observed)
                    items.extend(observed.evidence_items)
                status = ("observed" if items else "observed_metadata_only"
                          if any(visible_metadata.values()) else "unavailable")
                if not items:
                    reason = ("DEPENDENCY_API_METHOD_IL_EVIDENCE_UNAVAILABLE"
                              if any(visible_metadata.values())
                              else "DEPENDENCY_API_NO_MATCHING_METADATA_EVIDENCE")
            except (OSError, RuntimeError, ValueError) as error:
                status = "partial" if items else "unavailable"
                reason = ("DEPENDENCY_API_SOURCE_UNAVAILABLE" if isinstance(error, OSError)
                          else "DEPENDENCY_API_INSPECTION_FAILED")
        availability = {"status": status, "source_id": source.source_id,
            "reason": reason, "tool_calls_completed": len(observations), "model_calls_performed": 0,
            "scope": "bounded_observed_methods_not_complete_dependency_coverage",
            "query_source": ("metadata_queries" if structured_metadata_queries else "legacy_source_queries"),
            "metadata_query_count": len(explicit_metadata_queries),
            "visible_metadata": visible_metadata}
        return EvidencePack(query=retrieved.query, readiness="partial" if items else "insufficient",
            can_generate_patch=False, items=tuple(items),
            retrieval_methods_used=tuple(dict.fromkeys(obs.tool for obs in observations)),
            budget_summary={"dependency_api_supplement": availability},
            missing_capabilities=(reason,) if reason else (),
            trace_refs=tuple(obs.observation_id for obs in observations),
            tool_trace_summary=tuple({"observation_id": obs.observation_id, "tool": obs.tool,
                "source_id": obs.source_id, "summary": obs.summary} for obs in observations),
            provenance_validation={"target_code_executed": False},
            model_identity={"execution_kind": "not_invoked", "provider": None, "model": None})

    augment.query_capabilities = {
        "source": source.model_view(),
        "available": {"search_dotnet_metadata", "inspect_method_il"}.issubset(source.allowed_tools),
        "tools": {
            "search_dotnet_metadata": {
                "input_field": "metadata_queries",
                "matching": "case-insensitive literal substring matching of compact type/member identifiers",
                "result_kinds": ["methods", "types", "fields"],
                "limit_per_kind": max_method_inspections,
                "assembly_scope": "assembly_relative_paths only; if empty, *.dll recursively within the bound source root",
            },
            "inspect_method_il": {
                "selection": "only exact candidate_ref returned by this metadata search",
                "max_method_inspections": max_method_inspections,
                "target_code_execution_allowed": False,
            },
        },
    }
    return augment


def _admit_api_supplement(value, *, query):
    if not isinstance(value, EvidencePack):
        raise ValueError("SOURCE_AUGMENTATION_PACK_INVALID")
    if (value.hook_discovery is not None or value.agent_discovery is not None
            or value.provenance_validation.get("api_selection", {}).get("mode") == "assisted_explicit"):
        try:
            return _strict_source_augmentation(value, expected_query=query)
        except ValueError as error:
            if str(error) != "SOURCE_AUGMENTATION_API_EVIDENCE_MISSING":
                raise
            return EvidencePack.from_artifact_payload(value.to_artifact_payload())
    # Tool-only evidence has no model semantic section to rehydrate. Preserve
    # that absence; validate the existing item boundary instead of inventing a
    # Hook/Agent judgment just to pass a semantic-pack parser.
    if value.query != query or value.requirement_coverage is not None:
        raise ValueError("SOURCE_AUGMENTATION_QUERY_OR_COVERAGE_INVALID")
    if value.can_generate_patch or value.provenance_validation.get("target_code_executed") is not False:
        raise ValueError("SOURCE_AUGMENTATION_TARGET_EXECUTION_INVALID")
    if any(item.source_type != "decompiled_api" or not _valid_source_augmentation_item(item) for item in value.items):
        raise ValueError("SOURCE_AUGMENTATION_ITEM_BOUNDARY_INVALID")
    return deepcopy(value)


def initial_findings_reassessment_allowed(payload, feature_data) -> bool:
    """Recognize an accepted, complete semantic gap before initial Planning.

    A provider/schema failure may also retain an insufficient EvidencePack. Its
    fallback is not an accepted Findings response and cannot activate this path.
    """
    try:
        old = EvidencePack.from_artifact_payload(payload)
        coverage = validated_requirement_coverage_view(payload)
        audit = old.budget_summary.get("requirement_findings_attempt_audit")
        section = old.requirement_coverage
        return (
            coverage is not None
            and coverage.feature_id == feature_data["feature_id"]
            and coverage.feature_spec_sha256 == feature_spec_payload_sha256(feature_data)
            and old.query == feature_data["user_request"]
            and old.readiness == "insufficient" and old.can_generate_patch is False
            and bool(old.blockers)
            and all(item["code"] == "REQUIREMENT_COVERAGE_INSUFFICIENT" for item in old.blockers)
            and isinstance(audit, list) and bool(audit)
            and audit[-1]["status"] == "accepted" and audit[-1]["error_code"] is None
            and not old.budget_summary.get("semantic_provider_failed_closed")
            and not old.budget_summary.get("semantic_provider_failure_codes")
            and not old.budget_summary.get("semantic_provider_failure_diagnostics")
            and _initial_source_queries_closed(old)
            and "initial_findings_reassessment" not in old.budget_summary
            and len(section["findings"]) == len(section["relations"]) > 0
            and {item["relation_id"] for item in section["findings"]}
                == {item["relation_id"] for item in section["relations"]}
        )
    except (KeyError, TypeError, ValueError):
        return False


def _reassessment_has_new_information(pack: EvidencePack) -> bool:
    reassessment = pack.budget_summary.get("reassessment")
    dependency = pack.budget_summary.get("dependency_api_supplement")
    return bool(isinstance(reassessment, dict) and reassessment.get("new_item_count", 0) > 0) or (
        isinstance(dependency, dict)
        and dependency.get("status") in {"observed", "observed_metadata_only"}
        and (bool(dependency.get("visible_metadata", {}).get("methods"))
             or bool(dependency.get("visible_metadata", {}).get("types"))
             or bool(dependency.get("visible_metadata", {}).get("fields")))
    )


def validate_unchanged_source_reassessment(pack, *, state, artifact_store, workspace_root, feature_data):
    """Admit reuse of existing judgments only over the exact current Product inputs.

    为什么要有它：真实重查没有新事实时可复用原判断，但源码、需求、边界或失败材料变化不能借用旧判断。
    """
    summary = pack.budget_summary.get("reassessment", {})
    if summary.get("reuse_current_evidence") is not True or not workspace_root:
        return None
    try:
        repair_ref = ArtifactRef(feature_id=state["feature_id"], artifact_type="repair_plan",
            path=state["artifacts"]["repair_plan"], artifact_hash=state["artifact_hashes"]["repair_plan"])
        repair = artifact_store.semantic_read_immutable(repair_ref)["data"]
        context, upstream = repair["repair_context"], repair["repair_context"]["upstream_request"]
        selected = context["selected_artifacts"]
        if (not context.get("product_context_sha256") or context.get("current_failure_available") is not False
                or context.get("failure_report_ref") is not None or repair.get("failure_findings")
                or upstream["required_owner"] != "evidence"
                or summary["repair_plan_hash"] != repair_ref.artifact_hash
                or summary["old_evidence_hash"] != selected["evidence_pack"]["artifact_hash"]
                or any(state["artifacts"].get(kind) != selected[kind]["path"]
                    or state["artifact_hashes"].get(kind) != selected[kind]["artifact_hash"]
                    for kind in ("feature_spec", "evidence_pack", "implementation_plan"))):
            return None
        old_payload = artifact_store.semantic_read_immutable(ArtifactRef(**selected["evidence_pack"]))["data"]
        old = EvidencePack.from_artifact_payload(old_payload)
        feature = artifact_store.semantic_read_immutable(ArtifactRef(**selected["feature_spec"]))["data"]
        plan = artifact_store.semantic_read_immutable(ArtifactRef(**selected["implementation_plan"]))["data"]
        coverage = validated_requirement_coverage_view(old_payload)
        if (feature != feature_data or state["user_request"] != feature_data["user_request"]
                or state["target_mod"] != feature_data["target_mod"]
                or coverage is None or coverage.feature_spec_sha256 != feature_spec_payload_sha256(feature_data)
                or old.query != feature_data["user_request"] or old.readiness != "sufficient" or not old.can_generate_patch
                or old.blockers or old.gaps or old.missing_evidence
                or any(row["unresolved_gaps"] for row in old.requirement_coverage["findings"])
                or replace(pack, budget_summary=old.budget_summary).to_artifact_payload() != old_payload
                or upstream["preserved_plan_boundaries"] != plan["side_effect_boundaries"]):
            return None
        from src.mod_agent.fresh_planning_continuation import _validate_selected_plan

        _validate_selected_plan(plan, old_payload, feature_id=state["feature_id"])
        refs = {"binding://" + row["binding_id"] for group in ("edit_bindings", "supporting_bindings")
            for row in plan["binding_registry"].get(group, [])}
        if not upstream["source_refs"] or not set(upstream["source_refs"]) <= refs:
            return None
        queries, executions = summary["queries"], summary["per_query_execution"]
        if (not queries or summary["new_item_count"] != 0 or summary["new_evidence_ids"]
                or summary["all_queries_executed"] is not True
                or summary["findings_provider_invoked"] is not False
                or pack.budget_summary["dependency_api_supplement"]["status"] != "not_configured"
                or any(row.get("authorized_source") != "target_source" or row.get("gap_ids")
                    or row.get("metadata_queries") for row in queries)
                or not executions or len(executions) != len(queries)
                or any(row.get("retrieval_status") != "source_observations_returned" for row in executions)):
            return None
        evidence_ids = {item.evidence_id for item in old.items}
        if any(row.get("query_id") != query["query_id"] or row.get("requirement_id") != query["requirement_id"]
                or row.get("query_sha256") != hashlib.sha256(query["query"].encode("utf-8")).hexdigest()
                or not row.get("evidence_ids") or not set(row["evidence_ids"]) <= evidence_ids
                for query, row in zip(queries, executions)):
            return None
        _validate_source_items(old_payload, Path(workspace_root))
        if not old.items:
            return None
        for item in old.items:
            if item.source_type != "target_source" or item.retrieval_method not in {"rg_live_read", "indexed_source"}:
                return None
            raw = (Path(workspace_root) / item.source_path).read_bytes()
            text = raw.decode("utf-8")
            if (hashlib.sha256(raw).hexdigest() != item.source_hash or item.excerpt != text
                    or item.span != {"start_line": 1, "end_line": len(text.splitlines())}):
                return None
        return {"origin": "executed_source_recheck_reusing_unchanged_judgments",
            "selected_artifact_hashes": {kind: selected[kind]["artifact_hash"] for kind in selected},
            "repair_plan_hash": repair_ref.artifact_hash, "query_count": len(queries),
            "queries": deepcopy(queries), "per_query_execution": deepcopy(executions),
            "source_evidence_ids": [item.evidence_id for item in old.items],
            "source_hashes": {item.source_path: item.source_hash for item in old.items},
            "new_source_facts": False, "findings_recomputed": False, "planning_recomputed": False,
            "semantic_repair_proven": False, "grants_new_authority": False}
    except (KeyError, TypeError, ValueError, OSError):
        return None


def dependency_source_diversification_required(pack: EvidencePack) -> bool:
    """Require one dependency query only after a source-only zero-new cycle."""
    reassessment = pack.budget_summary.get("reassessment")
    dependency = pack.budget_summary.get("dependency_api_supplement")
    if (not isinstance(reassessment, dict) or reassessment.get("new_item_count") != 0
            or not isinstance(dependency, dict) or dependency.get("status") != "not_requested"):
        return False
    queries = reassessment.get("queries")
    source_only_zero_new = isinstance(queries, list) and bool(queries) and all(
        isinstance(row, dict) and row.get("authorized_source", "target_source") == "target_source"
        and not row.get("metadata_queries") for row in queries)
    requirements = pack.requirement_coverage.get("requirements", [])
    dependency_fact_required = any(isinstance(row, dict)
        and row.get("role") == "pre_patch_evidence"
        and ("API" in str(row.get("text", ""))
             or any("API" in str(value) for value in row.get("query_leads", [])))
        for row in requirements)
    return source_only_zero_new and dependency_fact_required


def prepare_initial_findings_reassessment_revision(payload, feature_data, *, revision_number=1):
    """Project one accepted prior reassessment into a new bounded initial input."""
    try:
        if revision_number not in {1, 2, 3, 4}:
            raise ValueError("INITIAL_FINDINGS_REASSESSMENT_REVISION_INVALID")
        old = EvidencePack.from_artifact_payload(payload)
        summary = deepcopy(old.budget_summary)
        lineage = summary.get("initial_findings_reassessment")
        previous_revision_number = (lineage.get("revision_number", 0)
                                    if isinstance(lineage, dict) else None)
        reassessment = summary.get("reassessment")
        audit = reassessment.get("findings_attempt_audit")
        rows = reassessment.get("per_query_execution")
        if (not isinstance(lineage, dict)
                or "prior_initial_findings_reassessment" in summary
                or previous_revision_number not in (
                    {0} if revision_number == 1 else {0, 1} if revision_number == 2
                    else {2} if revision_number == 3 else {3})
                or not isinstance(reassessment, dict)
                or not isinstance(audit, list) or not audit
                or audit[-1].get("status") != "accepted"
                or audit[-1].get("error_code") is not None
                or not isinstance(rows, list) or not rows
                or not all(isinstance(row, dict) and row.get("retrieval_status") in {
                    "source_observations_returned", "no_matches", "requested_source_no_matches"}
                           for row in rows)):
            raise ValueError("INITIAL_FINDINGS_REASSESSMENT_REVISION_INVALID")
        projected = deepcopy(payload)
        projected_summary = projected["budget_summary"]
        projected_summary["prior_initial_findings_reassessment"] = projected_summary.pop(
            "initial_findings_reassessment")
        projected_summary["requirement_findings_attempt_audit"] = deepcopy(audit)
        projected_summary["per_query_execution"] = deepcopy(rows)
        projected_summary["all_queries_executed"] = True
        projected_summary["reassessment_revision_admission"] = {
            "prior_evidence_hash": reassessment.get("old_evidence_hash"),
            "prior_query_count": reassessment.get("query_count"),
            "prior_model_receipt_available": reassessment.get("model_receipt_available"),
            "revision_number": revision_number,
        }
        projected_summary["reassessment_revision_number"] = revision_number
        if not initial_findings_reassessment_allowed(projected, feature_data):
            raise ValueError("INITIAL_FINDINGS_REASSESSMENT_REVISION_INVALID")
        return projected
    except (KeyError, TypeError, ValueError) as error:
        if str(error) == "INITIAL_FINDINGS_REASSESSMENT_REVISION_INVALID":
            raise
        raise ValueError("INITIAL_FINDINGS_REASSESSMENT_REVISION_INVALID") from error


class RequirementCoverageReassessmentFactory:
    """Existing Evidence node adapter; semantic queries and Findings remain provider-owned."""

    def __init__(self, *, artifact_store, workspace_root, search_root, source_provider,
            semantic_provider, query_selector, max_additional_queries: int = 3,
            model_call_preflight=None, source_augmentation_factory=None,
            document_augmentation_factory=None,
            max_initial_reassessment_cycles: int = 1, cycle_observer=None, budget_snapshot=None):
        if type(max_additional_queries) is not int or not 1 <= max_additional_queries <= 64:
            raise ValueError("A bounded positive query budget is required.")
        self.artifact_store = artifact_store
        self.workspace_root = Path(workspace_root).resolve()
        self.search_root = search_root
        self.source_provider = source_provider
        self.semantic_provider = semantic_provider
        self.query_selector = query_selector
        self.max_additional_queries = max_additional_queries
        self._initial_reassessment_used = False
        self.model_call_preflight = model_call_preflight
        self.source_augmentation_factory = source_augmentation_factory
        self.document_augmentation_factory = document_augmentation_factory
        if max_initial_reassessment_cycles not in {1, 2}:
            raise ValueError("INITIAL_FINDINGS_REASSESSMENT_CYCLE_LIMIT_INVALID")
        self.max_initial_reassessment_cycles = max_initial_reassessment_cycles
        if cycle_observer is not None and not callable(cycle_observer):
            raise ValueError("INITIAL_FINDINGS_CYCLE_OBSERVER_INVALID")
        self.cycle_observer = cycle_observer
        self.budget_snapshot = budget_snapshot

    def reassess_initial(self, state: dict[str, Any], feature_data: dict[str, Any]) -> EvidencePack:
        """Recheck accepted initial Findings without manufacturing a RepairPlan."""
        if self._initial_reassessment_used or any(kind in state["artifacts"] for kind in (
                "implementation_plan", "patch_proposal", "repair_plan")):
            raise ValueError("INITIAL_FINDINGS_REASSESSMENT_NOT_ADMITTED")
        ref = ArtifactRef(feature_id=state["feature_id"], artifact_type="evidence_pack",
            path=state["artifacts"]["evidence_pack"], artifact_hash=state["artifact_hashes"]["evidence_pack"])
        payload = self.artifact_store.semantic_read_immutable(ref)["data"]
        if not initial_findings_reassessment_allowed(payload, feature_data):
            raise ValueError("INITIAL_FINDINGS_REASSESSMENT_NOT_ADMITTED")
        old = EvidencePack.from_artifact_payload(payload)
        if any(item.source_type == "target_source" for item in old.items):
            _validate_source_items(payload, self.workspace_root)
        feedback = {"initial_findings_reassessment": {
            "prior_evidence_hash": ref.artifact_hash,
            "findings": deepcopy(old.requirement_coverage["findings"]),
            "sufficiency": deepcopy(old.requirement_coverage["sufficiency"]),
        }, "interpretation_is_not_observation": True}
        self._initial_reassessment_used = True
        from src.mod_agent.model_call_budget import is_meter_only, total_only
        if self.budget_snapshot is not None and (total_only(self.budget_snapshot()) or is_meter_only(self.budget_snapshot())):
            return self._reassess_task_total(state, feature_data, old, ref, feedback)
        revision_number = old.budget_summary.get("reassessment_revision_number")
        first = self._reassess(state, feature_data, old, feedback, initial_lineage={
            "prior_evidence_ref": asdict(ref),
            "projection_identity": deepcopy(old.requirement_coverage["projection_identity"]),
            "max_cycles": self.max_initial_reassessment_cycles, "cycle": 1,
            **({"revision_number": revision_number} if revision_number is not None else {}),
        })
        no_new_information = not _reassessment_has_new_information(first)
        if (self.max_initial_reassessment_cycles == 2 and not first.can_generate_patch
                and no_new_information):
            reassessment = first.budget_summary.get("reassessment", {})
            first = replace(first, budget_summary={**first.budget_summary,
                "reassessment_stop": {
                    "code": "INITIAL_FINDINGS_NO_NEW_INFORMATION_STOP",
                    "cycle": 1,
                    "prior_item_count": reassessment.get("prior_item_count"),
                    "new_item_count": reassessment.get("new_item_count"),
                    "new_evidence_ids": deepcopy(reassessment.get("new_evidence_ids", [])),
                    "sufficiency_status": first.requirement_coverage["sufficiency"]["status"],
                    "not_started": ["initial_findings_queries_cycle_2",
                                    "requirement_findings_cycle_2", "planning", "initial_patch"],
                    "fact_boundary": "deterministic counts and stage reachability only",
                }})
        if self.cycle_observer is not None:
            self.cycle_observer(first, 1)
        if (self.max_initial_reassessment_cycles == 1 or first.can_generate_patch
                or no_new_information):
            return first
        first_cycle = deepcopy(first.budget_summary.get("reassessment", {}))
        second_feedback = {"initial_findings_reassessment": {
            "prior_evidence_hash": ref.artifact_hash,
            "findings": deepcopy(first.requirement_coverage["findings"]),
            "sufficiency": deepcopy(first.requirement_coverage["sufficiency"]),
        }, "prior_cycle": {
            "queries": deepcopy(first_cycle.get("queries", [])),
            "per_query_execution": deepcopy(first_cycle.get("per_query_execution", [])),
            "new_evidence_ids": deepcopy(first_cycle.get("new_evidence_ids", [])),
            "dependency_api_supplement": deepcopy(
                first.budget_summary.get("dependency_api_supplement")),
        }, "interpretation_is_not_observation": True}
        second = self._reassess(state, feature_data, first, second_feedback, initial_lineage={
            "prior_evidence_ref": asdict(ref),
            "projection_identity": deepcopy(old.requirement_coverage["projection_identity"]),
            "max_cycles": 2, "cycle": 2,
            **({"revision_number": revision_number} if revision_number is not None else {}),
        })
        if self.cycle_observer is not None:
            self.cycle_observer(second, 2)
        return replace(second, budget_summary={**second.budget_summary,
            "reassessment_cycle_history": [{
                "cycle": 1,
                "queries": deepcopy(first_cycle.get("queries", [])),
                "per_query_execution": deepcopy(first_cycle.get("per_query_execution", [])),
                "new_evidence_ids": deepcopy(first_cycle.get("new_evidence_ids", [])),
                "dependency_api_supplement": deepcopy(
                    first.budget_summary.get("dependency_api_supplement")),
                "sufficiency": deepcopy(first.requirement_coverage["sufficiency"]),
            }]})

    def _reassess_task_total(self, state, feature_data, old, ref, feedback):
        """Continue useful Evidence cycles under the same reserved task ledger."""
        from src.mod_agent.model_call_budget import is_meter_only

        cycle, history = 0, []
        while True:
            before = self.budget_snapshot()
            cycle += 1
            current = self._reassess(state, feature_data, old, feedback, initial_lineage={
                "prior_evidence_ref": asdict(ref),
                "projection_identity": deepcopy(old.requirement_coverage["projection_identity"]),
                "max_cycles": None, "cycle": cycle, "model_call_budget_policy": before["policy"]})
            after = self.budget_snapshot()
            if after["consumed"] < before["consumed"] + 2:
                raise ValueError("INITIAL_FINDINGS_MODEL_ACCOUNTING_NOT_ADVANCED")
            no_new = not _reassessment_has_new_information(current)
            stop = None
            if not current.can_generate_patch and (no_new or not is_meter_only(after) and after["remaining"] < 2):
                stop = {"code": "INITIAL_FINDINGS_NO_NEW_INFORMATION_STOP" if no_new else "PRODUCT_MODEL_CALL_BUDGET_EXHAUSTED",
                        "cycle": cycle, "remaining_model_calls": after["remaining"], "next_cycle_requires": 2}
            current = replace(current, budget_summary={**current.budget_summary,
                "reassessment_cycle_history": deepcopy(history), **({"reassessment_stop": stop} if stop else {})})
            if self.cycle_observer is not None:
                self.cycle_observer(current, cycle)
            if current.can_generate_patch or stop:
                return current
            observed = current.budget_summary["reassessment"]
            prior_cycle = {key: deepcopy(observed.get(key, [])) for key in (
                "queries", "per_query_execution", "new_evidence_ids")}
            prior_cycle["dependency_api_supplement"] = deepcopy(current.budget_summary.get("dependency_api_supplement"))
            history.append({"cycle": cycle, **prior_cycle, "sufficiency": deepcopy(current.requirement_coverage["sufficiency"])})
            feedback = {"initial_findings_reassessment": {"prior_evidence_hash": ref.artifact_hash,
                "findings": deepcopy(current.requirement_coverage["findings"]),
                "sufficiency": deepcopy(current.requirement_coverage["sufficiency"])},
                "prior_cycle": prior_cycle, "interpretation_is_not_observation": True}
            old = current

    def __call__(self, state: dict[str, Any], feature_data: dict[str, Any]) -> EvidencePack:
        def read(kind):
            ref = ArtifactRef(feature_id=state["feature_id"], artifact_type=kind,
                path=state["artifacts"][kind], artifact_hash=state["artifact_hashes"][kind])
            return self.artifact_store.semantic_read_immutable(ref)["data"]
        old_payload = read("evidence_pack")
        old = EvidencePack.from_artifact_payload(old_payload)
        coverage = validated_requirement_coverage_view(old_payload)
        if coverage is None or coverage.feature_id != feature_data["feature_id"]:
            raise ValueError("PATCH_REASSESSMENT_COVERAGE_INVALID")
        _validate_source_items(old_payload, self.workspace_root)
        repair = read("repair_plan")
        upstream = repair["repair_context"]["upstream_request"]
        recommendation = {key: deepcopy(upstream[key]) for key in (
            "required_owner", "reason", "source_refs", "preserved_plan_boundaries")}
        for key in ("source_worker_response_ref", "source_attempt_ref", "failure_report_ref"):
            if key in upstream:
                ref = upstream[key]
                recommendation[key] = {"sha256": ref["sha256"]} if ref is not None else None
        observations = project_execution_reference_paths(repair["failure_findings"])
        feedback = {"recommendation": recommendation, "observed_failure": observations,
            "interpretation_is_not_observation": True}
        return self._reassess(state, feature_data, old, feedback)

    def _reassess(self, state, feature_data, old, feedback, *, initial_lineage=None):
        budget_preflight = (self.model_call_preflight(initial_lineage is not None)
                            if self.model_call_preflight is not None else None)
        materialized = {key: deepcopy(old.requirement_coverage[key]) for key in (
            "schema_version", "feature_spec_identity", "request_identity", "projection_identity",
            "input_obligations", "input_dispositions", "requirements", "relations", "queries")}
        selected = self.query_selector(deepcopy(feature_data), deepcopy(materialized), deepcopy(old), deepcopy(feedback))
        requirement_ids = {item["requirement_id"] for item in materialized["requirements"]}
        if not isinstance(selected, (tuple, list)) or len(selected) > self.max_additional_queries:
            raise ValueError("PATCH_REASSESSMENT_QUERY_BUDGET_INVALID")
        gap_parents = gap_requirement_bindings(materialized, old)
        queries = []
        for index, item in enumerate(selected, 1):
            allowed_fields = {"query", "requirement_id"}
            if isinstance(item, dict) and "metadata_queries" in item:
                allowed_fields.add("metadata_queries")
            structured_gap_fields = {"gap_ids", "current_fact_needed", "modification_impact",
                                     "authorized_source"}
            if isinstance(item, dict) and set(item).intersection(structured_gap_fields):
                allowed_fields.update(structured_gap_fields)
            metadata_queries = item.get("metadata_queries", ()) if isinstance(item, dict) else ()
            gap_ids = item.get("gap_ids", ()) if isinstance(item, dict) else ()
            if (not isinstance(item, dict) or set(item) != allowed_fields
                    or type(item["query"]) is not str or not item["query"].strip()
                    or len(item["query"]) > 256
                    or item["requirement_id"] not in requirement_ids
                    or not isinstance(metadata_queries, (tuple, list))
                    or len(metadata_queries) > 3
                    or any(type(query) is not str or not query.strip()
                           or len(query) > 128 or any(character.isspace() for character in query)
                           or "?" in query
                           for query in metadata_queries)
                    or (bool(set(item).intersection(structured_gap_fields))
                        and (not isinstance(gap_ids, (tuple, list))
                             or (initial_lineage is not None and not gap_ids)
                             or len(set(gap_ids)) != len(gap_ids)
                             or any(type(value) is not str or not value for value in gap_ids)
                             or type(item.get("current_fact_needed")) is not str
                             or not item["current_fact_needed"].strip()
                             or len(item["current_fact_needed"]) > 256
                             or type(item.get("modification_impact")) is not str
                             or not item["modification_impact"].strip()
                             or len(item["modification_impact"]) > 256
                             or item.get("authorized_source") not in {
                                 "target_source", "dependency_api"}))):
                raise ValueError("PATCH_REASSESSMENT_QUERY_INVALID")
            # 为什么要有它：独立合法的 ID 不足以证明 gap 属于所选需求；工具前必须复核关联。
            if gap_ids and any(item["requirement_id"] not in gap_parents.get(gap, ()) for gap in gap_ids):
                raise ValueError("PATCH_REASSESSMENT_QUERY_GAP_REQUIREMENT_MISMATCH")
            queries.append({"query_id": f"upstream_query_{index:03d}", **item})
        if len({"metadata_queries" in item for item in queries}) > 1:
            raise ValueError("PATCH_REASSESSMENT_QUERY_INVALID")
        if initial_lineage is not None and not queries:
            raise ValueError("INITIAL_FINDINGS_QUERY_REQUIRED")
        if (dependency_source_diversification_required(old)
                and not any(row.get("authorized_source") == "dependency_api"
                            and bool(row.get("metadata_queries")) for row in queries)):
            raise ValueError("INITIAL_FINDINGS_DEPENDENCY_SOURCE_DIVERSIFICATION_REQUIRED")
        prior_reassessment = old.budget_summary.get("reassessment")
        if (isinstance(prior_reassessment, dict)
                and prior_reassessment.get("new_item_count") == 0):
            prior_query_keys = {(row.get("query"), tuple(row.get("metadata_queries", ())))
                                for row in prior_reassessment.get("queries", ())
                                if isinstance(row, dict)}
            if any((row["query"], tuple(row.get("metadata_queries", ()))) in prior_query_keys
                   for row in queries):
                raise ValueError("PATCH_REASSESSMENT_QUERY_REPEATS_NO_NEW_INFORMATION")
        source_queries = tuple({key: item[key] for key in ("query_id", "query", "requirement_id")}
                               for item in queries)
        added = (self.source_provider.collect_query_set(queries=source_queries, search_root=self.search_root,
            max_items=len(queries)) if queries else EvidencePack(query=old.query, readiness="partial", can_generate_patch=False))
        if initial_lineage is not None and queries and not _successful_source_queries(added):
            raise ValueError("INITIAL_FINDINGS_RETRIEVAL_FAILED")
        by_id = {item.evidence_id: item for item in old.items}
        for item in added.items:
            by_id[item.evidence_id] = (merge_reobserved_source_item(by_id[item.evidence_id], item)
                if item.evidence_id in by_id else item)
        new_evidence_ids = [value for value in by_id if value not in {
            item.evidence_id for item in old.items}]
        combined = EvidencePack(query=old.query, readiness="partial", can_generate_patch=False,
            items=tuple(by_id.values()), retrieval_methods_used=tuple(dict.fromkeys((*old.retrieval_methods_used, *added.retrieval_methods_used))),
            timing=dict(added.timing), budget_summary={
                **({"baseline_compilation_failure": deepcopy(old.budget_summary["baseline_compilation_failure"])}
                   if old.budget_summary.get("baseline_compilation_failure") else {}),
                "reassessment": {
                "old_evidence_hash": state["artifact_hashes"]["evidence_pack"],
                **({"repair_plan_hash": state["artifact_hashes"]["repair_plan"]} if initial_lineage is None else {}),
                "prior_item_count": len(old.items), "new_item_count": len(by_id) - len(old.items),
                "new_evidence_ids": new_evidence_ids,
                "query_count": len(queries), "queries": deepcopy(queries), "fresh_execution": False,
                **({"budget_preflight": budget_preflight} if budget_preflight is not None else {}),
                "per_query_execution": deepcopy(added.budget_summary.get("per_query_execution", [])),
                "all_queries_executed": added.budget_summary.get("all_queries_executed") is True,
                "old_execution_receipts_reused_as_new": False},
                **({"initial_findings_reassessment": deepcopy(initial_lineage)} if initial_lineage is not None else {})},
            trace_refs=tuple(added.trace_refs))
        api_status = {"status": "not_configured", "model_calls_performed": 0,
                      "reason": "DEPENDENCY_API_SUPPLEMENT_NOT_CONFIGURED"}
        if self.source_augmentation_factory is not None:
            augmentation = self.source_augmentation_factory(deepcopy(materialized), deepcopy(combined))
            augmentation = _admit_api_supplement(augmentation, query=old.query)
            combined = _merge_source_augmentation(retrieved_pack=combined, augmentation_pack=augmentation)
            api_status = deepcopy(augmentation.budget_summary.get("dependency_api_supplement", {
                "status": "observed" if augmentation.items else "unavailable",
                "scope": "configured_source_augmentation"}))
        if self.document_augmentation_factory is not None:
            from .document_knowledge import merge_document_augmentation
            documents = self.document_augmentation_factory(deepcopy(materialized), deepcopy(combined))
            combined = merge_document_augmentation(combined, documents)
        combined = replace(combined, budget_summary={**combined.budget_summary,
            "dependency_api_supplement": api_status})
        feedback = {**feedback, "dependency_api_supplement": deepcopy(api_status)}
        if initial_lineage is None or any(item.source_type == "target_source" for item in combined.items):
            _validate_source_items(combined.to_artifact_payload(), self.workspace_root)
        if initial_lineage is None and combined.items == old.items:
            summary = {**combined.budget_summary["reassessment"],
                "reuse_current_evidence": True, "findings_provider_invoked": False,
                "model_receipt_available": False}
            candidate = replace(old, budget_summary={**old.budget_summary,
                "reassessment": summary, "dependency_api_supplement": deepcopy(api_status)})
            if validate_unchanged_source_reassessment(candidate, state=state,
                    artifact_store=self.artifact_store, workspace_root=self.workspace_root, feature_data=feature_data) is not None:
                return candidate
        findings = self.semantic_provider.findings(materialized_projection=deepcopy(materialized),
            retrieved_pack=combined, request=old.query, semantic_feedback=feedback)
        audit = deepcopy(list(getattr(self.semantic_provider, "findings_attempt_audit", ())))
        combined = replace(combined, budget_summary={**combined.budget_summary, "reassessment": {
            **combined.budget_summary["reassessment"], "findings_provider_invoked": True,
            "findings_attempt_audit": audit, "model_receipt_available": bool(audit)}})
        return build_requirement_coverage_pack(query=old.query, materialized_projection=materialized,
            retrieved_pack=combined, findings=findings)
