"""Read-only admission of the retained C5 first-failure prefix into a NEW trial.

This is not interrupted-attempt recovery or a reservation/locking mechanism.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentResponseFormat,
    AgentToolDefinition,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnRequest,
)
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
from src.mod_agent.planning.constraint_context import build_constraint_context
from src.mod_agent.proposal.failure_state import failure_signature, source_fact_hashes

CONTRACT = "c5-first-failure-new-trial.v1"
_SUBDIRECTORY = Path("generated_mods/SquadSignal")
_VERIFIER_COMPONENTS = {
    "checker": "src/mod_agent/domain/full_project_compilation.py",
    "build_adapter": "src/mod_agent/build/adapter.py",
    "baseline_failure_classifier": "src/mod_agent/build/baseline_repair.py",
    "build_execution_owner": "src/mod_agent/build/nuget.py",
    "dependency_copy_owner": "src/mod_agent/build/dependencies.py",
    "feedback_owner": "src/mod_agent/domain/compilation_feedback.py",
    "documentation_owner": "src/mod_agent/proposal/dependency_documentation.py",
    "probe_owner": "src/mod_agent/domain/compiled_probe_evaluator.py",
}


class ComparisonCheckpointError(ValueError):
    def __init__(self, code):
        self.code = "COMPARISON_CHECKPOINT_" + code
        super().__init__(self.code)


def _require(value, code):
    if not value:
        raise ComparisonCheckpointError(code)


def _equal(left, right):
    return canonical_hash(left) == canonical_hash(right)


def _file(reference):
    _require(isinstance(reference, Mapping) and isinstance(reference.get("path"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(reference.get("sha256"))) is not None, "FILE_REFERENCE_INVALID")
    path = Path(reference["path"])
    _require(path.is_absolute() and not str(path).startswith(("\\\\", "//")), "LOCAL_REFERENCE_REQUIRED")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "SOURCE_BYTES_CHANGED")
    return raw


def _document(reference):
    return read_bound_json({key: reference[key] for key in ("path", "sha256")})


def _request(document):
    _require(set(document) == {field.name for field in fields(AgentTurnRequest)}, "REQUEST_SHAPE_INVALID")
    request = AgentTurnRequest(**{**deepcopy(document),
        "messages": tuple(AgentMessage(**item) for item in document["messages"]),
        "tools": tuple(AgentToolDefinition(**item) for item in document["tools"]),
        "tool_results": tuple(AgentToolResult(**item) for item in document["tool_results"]),
        "historical_tool_exchanges": tuple(AgentToolExchange(**{**item,
            "tool_results": tuple(AgentToolResult(**row) for row in item["tool_results"])})
            for item in document["historical_tool_exchanges"]),
        "response_format": AgentResponseFormat(**document["response_format"]) if document["response_format"] else None})
    _require(_equal(asdict(request), document), "NATIVE_HISTORY_ROUNDTRIP_CHANGED")
    _require(all(isinstance(row.output, Mapping) for row in request.tool_results), "NATIVE_TOOL_OUTPUT_MAPPING_REQUIRED")
    return request


@dataclass(frozen=True)
class CheckedComparisonCheckpoint:
    trial_id: str
    request: AgentTurnRequest
    history: Mapping[str, Any]
    known_failure: Mapping[str, Any]
    known_code: Mapping[str, Any]
    consumed_model_turns: int
    consumed_tool_rounds: int
    consumed_observation_bytes: int
    product_consumed: int
    product_remaining: int
    routing_calls: int
    no_progress_seed: Mapping[str, int]
    source_facts: tuple[str, ...]
    wall_seconds: int
    expected_model_identity: Mapping[str, Any]
    worker_policy: Any
    worker_configuration_ref: Mapping[str, str]
    source_refs: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]


def _current_worker(reference, historical, workspace, artifact_store):
    current = _document(reference)
    loaded = PatchWorkerConfiguration.load(reference["path"])
    _require(_equal(loaded.reference, reference) and loaded.enable_recovery_loop, "CURRENT_WORKER_CHANGED")
    original = PatchWorkerConfiguration.load(historical["path"])
    expected = replace(original.policy, max_total_model_turns=15)
    _require(loaded.policy.max_total_model_turns in (15, 18)
        and replace(loaded.policy, max_total_model_turns=15) == expected, "WORKER_CAPABILITY_CHANGED")
    old = _document(historical)["behavior_evaluator"]
    selected = current["behavior_evaluator"]
    _require(selected["factory_name"] == old["factory_name"]
        and selected["factory_ref"]["sha256"] == old["factory_ref"]["sha256"], "EVALUATOR_CAPABILITY_CHANGED")
    _file(selected["factory_ref"])
    old_kwargs, new_kwargs = deepcopy(old["kwargs"]), deepcopy(selected["kwargs"])
    old_compilation = old_kwargs["compilation_kwargs"]
    new_compilation = new_kwargs["compilation_kwargs"]
    _require(Path(new_compilation["workspace_root"]).resolve() == workspace
        and Path(new_compilation["source_root"]).resolve() == workspace / _SUBDIRECTORY
        and Path(new_compilation["artifact_store_root"]).resolve() == artifact_store.root.resolve(), "WORKER_ROOT_BINDING_CHANGED")
    # Scope: explicit local relocation, never new evaluator behavior. 为什么要有它：隔离试验须换输出目录，不能要求新worker字节等于历史worker，也不能借路径重绑改变验收器。
    for key in ("output_root",):
        _require(Path(new_kwargs[key]).resolve().is_relative_to(workspace), "EVALUATOR_OUTPUT_OUTSIDE_TRIAL")
        new_kwargs[key] = old_kwargs[key]
    for key in ("source_root", "workspace_root", "artifact_store_root", "output_root"):
        if key == "output_root":
            _require(Path(new_compilation[key]).resolve().is_relative_to(workspace), "COMPILATION_OUTPUT_OUTSIDE_TRIAL")
        new_compilation[key] = old_compilation[key]
    for key in ("probe_configuration_ref",):
        _file(new_kwargs[key])
        _require(new_kwargs[key]["sha256"] == old_kwargs[key]["sha256"], "PROBE_CHANGED")
        new_kwargs[key] = old_kwargs[key]
    _file(new_compilation["nuget_config_ref"])
    _require(new_compilation["nuget_config_ref"]["sha256"] == old_compilation["nuget_config_ref"]["sha256"], "NUGET_CONFIG_CHANGED")
    new_compilation["nuget_config_ref"] = old_compilation["nuget_config_ref"]
    new_compilation["dependency_binding"]["source_root"] = old_compilation["dependency_binding"]["source_root"]
    _require(_equal(new_kwargs, old_kwargs), "VERIFIER_ARGUMENTS_CHANGED")
    return expected


def relocate_checkpoint_source_items(evidence, *, fixture, historical_workspace, workspace):
    """Derive only target-source addresses after checking both copies against B0."""
    old_root, new_root = Path(historical_workspace).resolve(), Path(workspace).resolve()
    _require(new_root != old_root and not new_root.is_relative_to(old_root), "ISOLATED_NEW_WORKSPACE_REQUIRED")
    baseline = [row for row in fixture["baseline_refs"] if row["location"] == "executed_workspace"]
    _require(len(baseline) == 7 and len({row["relative_path"] for row in baseline}) == 7, "BASELINE_MANIFEST_INVALID")
    by_path = {}
    for row in baseline:
        relative = PurePosixPath(row["relative_path"])
        _require(len(relative.parts) == 1 and relative.name not in {"", ".", ".."}, "BASELINE_PATH_ESCAPE")
        source_path = (_SUBDIRECTORY / relative.name).as_posix()
        old_path = (old_root / source_path).resolve()
        _require(Path(row["ref"]["path"]).resolve() == old_path, "BASELINE_SOURCE_ADDRESS_CHANGED")
        _file(row["ref"])
        new_path = (new_root / source_path).resolve()
        _require(new_path.is_relative_to(new_root / _SUBDIRECTORY), "BASELINE_PATH_ESCAPE")
        _file({"path": str(new_path), "sha256": row["ref"]["sha256"]})
        by_path[source_path] = (old_path, new_path, row["ref"]["sha256"])
    result, changes = deepcopy(evidence), []
    _require(isinstance(result.get("items"), list), "SOURCE_ITEMS_REQUIRED")
    # Scope: immutable EvidencePack target_source addresses only; never text or Trace.
    # 为什么要有它：新副本需要当前源码地址，但源码内容、原观察及模型历史不能随路径迁移被重写。
    for index, item in enumerate(result["items"]):
        if not isinstance(item, dict) or item.get("source_type") != "target_source":
            continue
        relative = item.get("source_path")
        _require(isinstance(relative, str) and relative in by_path, "SOURCE_ITEM_OUTSIDE_BASELINE")
        old_path, new_path, digest = by_path[relative]
        original_ref = item.get("source_ref")
        _require(isinstance(original_ref, str) and Path(original_ref).is_absolute()
            and Path(original_ref).resolve() == old_path, "SOURCE_ITEM_ADDRESS_CHANGED")
        _require(item.get("source_hash") == digest
            and all(item[name] == digest for name in ("file_hash", "doc_hash") if name in item), "SOURCE_ITEM_HASH_CHANGED")
        item["source_ref"] = str(new_path)
        changes.append({"json_pointer": f"/data/items/{index}/source_ref", "source_path": relative,
            "previous": original_ref, "current": str(new_path), "sha256": digest})
    _require(bool(changes), "TARGET_SOURCE_ITEMS_REQUIRED")
    return result, changes


def _admit_artifacts(fixture_ref, fixture, journal, references, store, derived_inputs_ref, workspace):
    originals = {kind: ArtifactRef(**journal["selected_artifacts"][kind]) for kind in references}
    first = originals["implementation_plan"]
    source_root = Path(first.path).resolve().parents[4]
    _require(source_root.is_dir() and (source_root / "features" / first.feature_id / "r" / first.artifact_type)
        == Path(first.path).resolve().parent, "SOURCE_STORE_LAYOUT_INVALID")
    _require(store.root.resolve() != source_root and not store.root.resolve().is_relative_to(source_root), "NEW_TRIAL_STORE_REQUIRED")
    historical_store = ArtifactStore(source_root)
    for kind, original in originals.items():
        _require(_equal(asdict(original), fixture["artifacts"][kind]["artifact_ref"])
            and fixture["artifacts"][kind]["file_ref"]["path"] == original.path, "SOURCE_ARTIFACT_CHANGED")
        historical = historical_store.semantic_read_immutable(original)
        _require(_equal(historical, _document(fixture["artifacts"][kind]["file_ref"])), "SOURCE_ARTIFACT_CONTENT_CHANGED")
    if derived_inputs_ref is None:
        for kind, reference in references.items():
            original = originals[kind]
            _require(reference.artifact_hash == original.artifact_hash and reference.feature_id == original.feature_id
                and reference.artifact_type == kind, "DERIVED_INPUTS_REQUIRED")
            _require(_equal(store.semantic_read_immutable(reference), historical_store.semantic_read_immutable(original))
                and store.check_freshness(reference).is_fresh, "CURRENT_ARTIFACT_STALE_OR_CHANGED")
        return {"mapping_ref": None, "strategy": "exact_identity", "data_equivalence_verified": True}
    mapping = _document(derived_inputs_ref)
    strategy = mapping.get("strategy")
    _require(mapping.get("schema_version") == "comparison-derived-inputs.v1"
        and strategy in {"identity_v1", "source_priority_v1"} and _equal(mapping.get("fixture_ref"), fixture_ref), "DERIVATION_POLICY_UNSUPPORTED")
    relocation_policy = mapping.get("source_relocation_policy")
    _require(relocation_policy in {None, "target_source_baseline_v1"}, "SOURCE_RELOCATION_POLICY_UNSUPPORTED")
    expected_evidence_data, relocation_changes = None, []
    if relocation_policy is not None:
        old_worker = _document(journal["worker_configuration_ref"])
        expected_evidence_data, relocation_changes = relocate_checkpoint_source_items(
            historical_store.semantic_read_immutable(originals["evidence_pack"])["data"], fixture=fixture,
            historical_workspace=old_worker["behavior_evaluator"]["kwargs"]["compilation_kwargs"]["workspace_root"],
            workspace=workspace)
    expected_plan_data = None
    if strategy == "source_priority_v1":
        source_feature = historical_store.semantic_read_immutable(originals["feature_spec"])["data"]
        source_evidence = historical_store.semantic_read_immutable(originals["evidence_pack"])["data"]
        source_plan = historical_store.semantic_read_immutable(originals["implementation_plan"])["data"]
        context = build_constraint_context(feature_spec=source_feature, evidence_pack=source_evidence,
            plan_boundaries=source_plan["side_effect_boundaries"])
        expected_plan_data = {**source_plan, "side_effect_boundaries": context["hard_boundaries"], "constraint_context": context}
    rows = mapping.get("artifacts")
    _require(isinstance(rows, list) and 3 <= len(rows) <= 64, "DERIVATION_CLOSURE_INVALID")
    pairs, derived_keys = {}, set()
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"source_ref", "derived_ref"}, "DERIVATION_ROW_INVALID")
        source, target = ArtifactRef(**row["source_ref"]), ArtifactRef(**row["derived_ref"])
        key, target_key = canonical_hash(asdict(source)), canonical_hash(asdict(target))
        _require(key not in pairs and target_key not in derived_keys and source.feature_id == first.feature_id
            and target.feature_id == first.feature_id and target.artifact_type == source.artifact_type, "DERIVATION_NOT_BIJECTIVE")
        source_doc = historical_store.semantic_read_immutable(source)
        target_doc = store.semantic_read_immutable(target)
        _require(store.check_freshness(target).is_fresh, "CURRENT_ARTIFACT_STALE")
        expected_doc = deepcopy(source_doc)
        if source == originals["evidence_pack"] and expected_evidence_data is not None:
            expected_doc["data"] = expected_evidence_data
        if source == originals["implementation_plan"] and expected_plan_data is not None:
            expected_doc["data"] = expected_plan_data
        _require(_equal({k: v for k, v in expected_doc.items() if k not in {"inputs", "artifact_hash"}},
                        {k: v for k, v in target_doc.items() if k not in {"inputs", "artifact_hash"}}), "DERIVED_DATA_CHANGED")
        pairs[key] = (source, target, source_doc, target_doc)
        derived_keys.add(target_key)
    # Scope: exact transitive relocation of A's inputs, not a semantic Plan rewrite.
    # 为什么要有它：上游绝对路径使新envelope哈希变化，必须逐边核对映射与数据全等，不能把任意新Plan声明成旧身份。
    edges = {}
    for key, (source, _target, source_doc, target_doc) in pairs.items():
        expected = deepcopy(source_doc["inputs"])
        upstream = expected.get("upstream_artifacts", [])
        _require(isinstance(upstream, list), "SOURCE_UPSTREAM_INVALID")
        edges[key] = []
        for item in upstream:
            source_upstream = ArtifactRef(feature_id=source.feature_id,
                **{name: item[name] for name in ("artifact_type", "artifact_hash", "path")})
            parent_key = canonical_hash(asdict(source_upstream))
            _require(parent_key in pairs, "DERIVATION_CLOSURE_INCOMPLETE")
            edges[key].append(parent_key)
            replacement = pairs[parent_key][1]
            item.update(artifact_hash=replacement.artifact_hash, path=replacement.path, artifact_type=replacement.artifact_type)
        _require(_equal(expected, target_doc["inputs"]), "DERIVED_UPSTREAM_MAPPING_CHANGED")
    visited, active = set(), set()

    def visit(key):
        _require(key not in active, "DERIVATION_CYCLE")
        if key in visited:
            return
        active.add(key)
        for parent in edges[key]:
            visit(parent)
        active.remove(key)
        visited.add(key)

    for kind, source in originals.items():
        key = canonical_hash(asdict(source))
        _require(key in pairs and pairs[key][1] == references[kind], "SELECTED_DERIVATION_CHANGED")
        visit(key)
    _require(visited == set(pairs), "UNRELATED_DERIVATION_ARTIFACT")
    return {"mapping_ref": deepcopy(derived_inputs_ref), "strategy": strategy,
        "data_equivalence_verified": strategy == "identity_v1" and relocation_policy is None,
        "data_equivalence_except_verified_source_addresses": strategy == "identity_v1" and relocation_policy is not None,
        "selected_strategy_recomputed": True,
        "constraint_context": deepcopy(expected_plan_data["constraint_context"]) if expected_plan_data is not None else None,
        "source_relocation_policy": relocation_policy, "source_address_changes": relocation_changes,
        "source_to_current_refs": deepcopy(rows), "semantic_identity_reused": False}


def admit_comparison_checkpoint(fixture_ref, *, trial_id, workspace_root, artifact_store,
        feature_spec_ref, evidence_pack_ref, implementation_plan_ref, worker_configuration_ref, model_profile,
        derived_inputs_ref=None):
    """Verify a C5 v3 prefix without provider sends, imports of evaluators or writes.

    The caller must atomically claim the new trial and enforce its deadline.
    Artifact relocation and optional source_priority_v1 are recomputed here.
    """
    fixture = _document(fixture_ref)
    _require(fixture.get("status") == "prefix_extracted_runner_adapter_required"
        and fixture.get("model_input_boundary", "").startswith("Only branch-request"), "FIXTURE_V3_REQUIRED")
    for reference in fixture["source_refs"]:
        _file(reference)
    source_results = [reference for reference in fixture["source_refs"]
        if str(reference["path"]).replace("\\", "/").endswith("/c5-scope-regression/result.json")]
    _require(len(source_results) == 1, "ORIGINAL_SCOPE_RESULT_REQUIRED")
    source_result = _document(source_results[0])
    _require(source_result.get("schema") == "c5-scope-regression-retention.v1", "SOURCE_CASE_VERSION_CHANGED")
    preparation = _document(source_result["preparation_ref"])
    process = _document(source_result["process_ref"])
    saved = {name: _document(reference) for name, reference in fixture["files"].items()}
    _require(set(saved) == {"pending-request.json", "branch-request.json", "prefix-event-20.json",
        "first-failure.json", "first-checks.json", "candidate-response.json"}, "FIXTURE_FILES_CHANGED")
    origin = fixture["checkpoint_origin"]
    _require(origin["event_index"] == 20 and origin["next_event_index"] == 21
        and origin["future_model_responses_or_failures_in_model_input"] is False, "CHECKPOINT_BOUNDARY_CHANGED")
    journal = _document(origin["receipt_ref"])
    _require(_equal(origin["receipt_ref"], source_result["receipt_ref"]), "SOURCE_ATTEMPT_CHANGED")
    first_evaluation = source_result["behavior_evaluations"][0]
    first_feedback = source_result["feedback_consumption"][0]
    _require(first_evaluation["evaluation_index"] == 1 and first_feedback["revision_index"] == 1
        and first_feedback["patch_turn_index"] == 3
        and _equal(saved["pending-request.json"], _document(first_feedback["request_ref"]))
        and _equal(saved["first-checks.json"], _document(first_evaluation["checks_ref"]))
        and _equal(saved["first-failure.json"], _document(first_evaluation["report_ref"])), "ORIGINAL_FIRST_FAILURE_CHANGED")
    events = journal["events"]
    selected = [item for item in events if item["event_index"] == 20]
    following = [item for item in events if item["event_index"] == 21]
    _require(len(selected) == len(following) == 1, "PREFIX_EVENT_NOT_UNIQUE")
    event, next_event = selected[0], following[0]
    _require(_equal(event, saved["prefix-event-20.json"])
        and event["phase"] == "local_patch_post_response_model_admitted"
        and next_event["phase"] == "local_patch_worker_turn_requested", "PREFIX_EVENT_CHANGED")
    _require(isinstance(trial_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", trial_id)
        and trial_id not in {journal["run_id"], journal["request_id"], journal.get("parent_run_id")}, "NEW_TRIAL_ID_REQUIRED")
    shared = event["shared_model_budget"]
    expected_stages = {name: {"consumed": count} for name, count in {
        "initial_patch": 2, "planning": 1, "requirement_findings": 1, "requirement_projection": 1, "repair": 0}.items()}
    _require(_equal(shared["by_stage"], expected_stages)
        and all(type(shared[name]) is int and shared[name] == expected for name, expected in
                (("maximum", 18), ("consumed", 5), ("remaining", 13)))
        and event["model_call_count"] == 2 and next_event["model_call_count"] == 3
        and next_event["shared_model_budget"]["consumed"] == 6, "HISTORICAL_BUDGET_CHANGED")
    factory = event["factory_receipt"]
    loop = factory["patch_agent_loop"]
    _require([row["turn_index"] for row in factory["worker_turns"]] == [1, 2]
        and loop["policy"]["max_total_model_turns"] == 15 and loop["tool_rounds_used"] == 4
        and loop["total_observation_bytes"] == 6182
        and loop["behavior_feedback_budget"]["combined_utf8_bytes"] == 16214
        and loop["behavior_feedback_budget"]["feedback_utf8_bytes"] == 10032, "PREFIX_COST_CHANGED")
    responses = [_document(row["response_ref"]) for row in factory["worker_turns"]]
    _require(_equal(responses[1], saved["candidate-response.json"]), "CANDIDATE_RESPONSE_CHANGED")
    pending = deepcopy(saved["pending-request.json"])
    states = [i for i, row in enumerate(pending["messages"]) if isinstance(row.get("content"), Mapping)
        and row["content"].get("message_type") == "patch_agent_loop_state"]
    _require(states == [len(pending["messages"]) - 1], "PENDING_STATE_NOT_UNIQUE")
    state = pending["messages"].pop()["content"]
    _require(_equal(pending, saved["branch-request.json"]), "BRANCH_CHANGED_MORE_THAN_PENDING_STATE")
    _require(_equal(state["budget_state"], fixture["budget"]["pending_state"])
        and state["budget_state"]["model_turns_used"] == 2 and state["budget_state"]["remaining_model_turns"] == 13
        and state["budget_state"]["tool_rounds_used"] == 4 and state["budget_state"]["observation_bytes_used"] == 16214,
        "PENDING_BUDGET_CHANGED")
    request = _request(pending)
    _require(len(request.tool_results) == 4 and not request.historical_tool_exchanges
        and [row.content for row in request.messages if row.role == "assistant"] == [responses[1]["content"]], "NATIVE_HISTORY_CHANGED")
    tool_ids = [row["call_id"] for row in responses[0]["tool_calls"]]
    _require(len(set(tool_ids)) == 4 and [row.call_id for row in request.tool_results] == tool_ids
        and _equal(request.provider_state, responses[0]["provider_state"]), "NATIVE_TOOL_EXCHANGE_CHANGED")
    raw_outputs = [dict(row.output) for row in request.tool_results]
    _require(all(row.name == "inspect_plan_binding_source" and row.output.get("truncated") is False
        for row in request.tool_results), "SOURCE_TOOL_SCOPE_CHANGED")
    facts = source_fact_hashes([{**row, "admitted_to_model": True} for row in raw_outputs])
    _require(facts == source_fact_hashes(loop["observation_trace"]) and len(facts) == 4, "SOURCE_FACTS_CHANGED")
    failures = loop["behavior_evaluations"]
    _require(len(failures) == 1, "FUTURE_FAILURES_IN_PREFIX")
    failure = deepcopy(failures[0])
    persisted = _document(failure["report_ref"])
    _require(_equal(persisted, saved["first-failure.json"]) and failure["admission"] == "accepted"
        and failure["execution_origin"] == "executed" and failure["status"] == "behavior_failed"
        and failure["failure_owner"] == "patch_implementation" and len(failure["observations"]) == 17,
        "FIRST_FAILURE_NOT_ADMITTED")
    for name in ("status", "failure_owner", "observations", "generated_changes_sha256", "source_snapshot_sha256"):
        _require(_equal(failure[name], persisted[name]), "FIRST_FAILURE_BINDING_CHANGED")
    corrections = [row.content for row in request.messages if isinstance(row.content, Mapping)
        and row.content.get("message_type") == "patch_behavior_correction"]
    _require(len(corrections) == 1 and corrections[0]["validator_feedback"]["report_sha256"] == failure["report_ref"]["sha256"]
        and len(corrections[0]["validator_feedback"]["observations"]) == 17, "FEEDBACK_CHANGED")
    checks = saved["first-checks.json"]
    _require(len(checks["checks"]) == 26 and sum(row["passed"] is True for row in checks["checks"]) == 9
        and _equal(checks["materialized_code_identity"], failure["materialized_code_identity"]), "CHECKS_OR_CODE_CHANGED")
    known_code = failure["materialized_code_identity"]
    candidate = json.loads(responses[1]["content"])
    _require(len(candidate["changes"]) == 1 and candidate["changes"][0]["edit_binding_id"] == "edit_binding_001"
        and hashlib.sha256(candidate["changes"][0]["after_text"].encode()).hexdigest() == fixture["candidate"]["after_text_sha256"]
        and _equal(known_code, fixture["candidate"]["identity"]), "FAILED_CODE_CHANGED")
    workspace = Path(workspace_root).resolve()
    old_worker = _document(journal["worker_configuration_ref"])
    old_workspace = Path(old_worker["behavior_evaluator"]["kwargs"]["compilation_kwargs"]["workspace_root"]).resolve()
    _require(workspace.is_dir() and workspace != old_workspace and not workspace.is_relative_to(old_workspace), "ISOLATED_NEW_WORKSPACE_REQUIRED")
    baseline = [row for row in fixture["baseline_refs"] if row["location"] == "executed_workspace"]
    _require(len(baseline) == 7 and len({row["relative_path"] for row in baseline}) == 7, "BASELINE_MANIFEST_INVALID")
    _require({row["relative_path"]: row["ref"]["sha256"] for row in baseline} == preparation["source_snapshot"],
        "ORIGINAL_BASELINE_MANIFEST_CHANGED")
    for row in baseline:
        current = (workspace / _SUBDIRECTORY / row["relative_path"]).resolve()
        _require(current.is_relative_to(workspace / _SUBDIRECTORY), "BASELINE_PATH_ESCAPE")
        _file({"path": str(current), "sha256": row["ref"]["sha256"]})
    refs = {"feature_spec": feature_spec_ref, "evidence_pack": evidence_pack_ref, "implementation_plan": implementation_plan_ref}
    # Scope: A's original semantic identities only. 为什么要有它：A*的新Plan必须经独立推导审核，不能把任意当前Plan当历史许可或忽略正常ArtifactStore新鲜度。
    for reference in refs.values():
        _require(isinstance(reference, ArtifactRef), "ARTIFACT_REF_REQUIRED")
    derivation = _admit_artifacts(fixture_ref, fixture, journal, refs, artifact_store, derived_inputs_ref, workspace)
    profile = model_profile.to_dict() if hasattr(model_profile, "to_dict") else dict(model_profile)
    _require(_equal(profile, factory["profile"]) and _equal(profile, fixture["model"]["profile"]), "MODEL_PROFILE_CHANGED")
    policy = _current_worker(worker_configuration_ref, journal["worker_configuration_ref"], workspace, artifact_store)
    model_identity = fixture["model"]["identity"]
    _require(_equal(model_identity, _document(fixture["model"]["admission_ref"])["actual_model_identity"])
        and model_identity["digest"] == factory["model_identity"]["digest"], "MODEL_ADMISSION_CHANGED")
    timeline = fixture["wall"]
    first_span = journal["execution_observations"]["a333697e623343de"]
    elapsed = (datetime.fromisoformat(first_span["started_at"]) - datetime.fromisoformat(timeline["original_process_started_at"])).total_seconds()
    _require(timeline["original_process_started_at"] == process["started_at"]
        and preparation["wall_seconds"] == 600 and preparation["product_calls"] == 18
        and timeline["original_limit_seconds"] == 600 and math.floor(600 - elapsed) == 255
        and timeline["admitted_branch_seconds"] == 255, "WALL_BUDGET_CHANGED")
    _require(fixture["budget"]["routing_calls_outside_product_limit"] == 1
        and source_result["historical_usage"]["routing_transport_calls"] == 1, "ROUTING_COST_CHANGED")
    provenance = {"contract": CONTRACT, "trial_id": trial_id, "fixture_ref": deepcopy(fixture_ref),
        "source_attempt_ref": deepcopy(origin["receipt_ref"]), "source_run_id": journal["run_id"],
        "source_event_index": 20, "new_trial_not_original_resume": True, "reservation_performed": False,
        "approval_inherited": False, "original_worker_configuration_ref": deepcopy(journal["worker_configuration_ref"]),
        "current_worker_configuration_ref": deepcopy(worker_configuration_ref),
        "original_platform_code_identity_is_historical_only": True,
        "current_provider_identity_and_dependency_preflight_required_before_send": True,
        "deadline_starts_before_new_trial_setup": True, "semantic_plan_derivation_supported": ["source_priority_v1"],
        "derived_inputs": derivation, "current_artifacts": {kind: asdict(reference) for kind, reference in refs.items()}}
    return CheckedComparisonCheckpoint(trial_id, request, deepcopy(factory), failure, deepcopy(known_code),
        2, 4, 16214, 5, 13, 1, {failure_signature(failure, facts): 1}, tuple(facts), 255,
        deepcopy(model_identity), policy, deepcopy(worker_configuration_ref),
        tuple(deepcopy(fixture["source_refs"])), provenance)


def _observed_reference(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _validator_layers(components, checker_identity, probe_ref, factory_ref, probe_files, final_identity):
    checker = {**checker_identity, "validator_sha256": canonical_hash({
        name: components[name] for name in ("checker", "build_adapter", "baseline_failure_classifier",
                                           "build_execution_owner", "dependency_copy_owner")})}
    feedback = {**checker, "validator_sha256": canonical_hash({"owner": components["feedback_owner"],
        "checker": checker, "documentation_owner": components["documentation_owner"]}), "deterministic_reuse_allowed": False}
    final = {**final_identity, "validator_sha256": canonical_hash({"owner": components["probe_owner"],
        "protocol": [factory_ref], "compilation": feedback, "probe": probe_ref, "files": probe_files})}
    return {"checker": checker, "compilation_feedback": feedback, "probe": final}


def admit_checkpoint_verifier(checked: CheckedComparisonCheckpoint, evaluator) -> dict[str, Any]:
    """Admit only eight-owner address relocation for failure-state comparison.

    This never rewrites a report, authorizes result reuse, or checks permission
    for a later Plan. The normal Graph still owns current-stage admission.
    """
    _require(isinstance(checked, CheckedComparisonCheckpoint), "CHECKED_PREFIX_REQUIRED")
    fixture = _document(checked.provenance["fixture_ref"])
    journal = _document(fixture["checkpoint_origin"]["receipt_ref"])
    events = [row for row in journal["events"] if row["event_index"] == 20]
    _require(len(events) == 1, "VERIFIER_SOURCE_EVENT_CHANGED")
    factory = events[0]["factory_receipt"]
    failure = factory["patch_agent_loop"]["behavior_evaluations"][0]
    facts = source_fact_hashes(factory["patch_agent_loop"]["observation_trace"])
    _require(_equal(checked.history, factory) and _equal(checked.known_failure, failure)
        and tuple(facts) == checked.source_facts
        and _equal(checked.no_progress_seed, {failure_signature(failure, facts): 1}), "RETAINED_FAILURE_OR_SEED_CHANGED")
    _require(_equal(_document(failure["report_ref"]), _document(fixture["files"]["first-failure.json"])), "SOURCE_REPORT_CHANGED")
    historical_worker = journal["worker_configuration_ref"]
    current_worker = _document(checked.worker_configuration_ref)
    current_kwargs = current_worker["behavior_evaluator"]["kwargs"]
    workspace = Path(current_kwargs["compilation_kwargs"]["workspace_root"]).resolve()
    store = ArtifactStore(current_kwargs["compilation_kwargs"]["artifact_store_root"])
    _current_worker(checked.worker_configuration_ref, historical_worker, workspace, store)
    for row in fixture["baseline_refs"]:
        if row["location"] == "executed_workspace":
            _file({"path": str((workspace / _SUBDIRECTORY / row["relative_path"]).resolve()), "sha256": row["ref"]["sha256"]})
    original_evaluator = _document(historical_worker)["behavior_evaluator"]
    selected = current_worker["behavior_evaluator"]
    original_kwargs = original_evaluator["kwargs"]
    _require(_equal(selected["factory_ref"], original_evaluator["factory_ref"])
        and _equal(current_kwargs["probe_configuration_ref"], original_kwargs["probe_configuration_ref"])
        and _equal(current_kwargs["compilation_kwargs"]["nuget_config_ref"], original_kwargs["compilation_kwargs"]["nuget_config_ref"])
        and _equal(current_kwargs["compilation_kwargs"]["dependency_binding"], original_kwargs["compilation_kwargs"]["dependency_binding"]),
        "ONLY_EIGHT_OWNER_ADDRESSES_MAY_CHANGE")
    expected_configuration = {"configuration_ref": dict(checked.worker_configuration_ref),
        "factory_ref": selected["factory_ref"], "factory_name": selected["factory_name"],
        "kwargs_sha256": canonical_hash(selected["kwargs"])}
    _require(callable(getattr(evaluator, "configuration_receipt", None))
        and _equal(evaluator.configuration_receipt(), expected_configuration), "CURRENT_EVALUATOR_BINDING_CHANGED")
    observed = evaluator.verification_identity()
    _require(_equal(evaluator.configuration_receipt(), expected_configuration), "CURRENT_EVALUATOR_BINDING_CHANGED")
    source_refs = [row for row in checked.source_refs
        if str(row["path"]).replace("\\", "/").endswith("/c5-scope-regression/result.json")]
    _require(len(source_refs) == 1, "VERIFIER_SOURCE_RESULT_REQUIRED")
    scope = _document(source_refs[0])
    snapshot_ref = scope["actual_source_evidence"]["snapshot_ref"]
    _document(snapshot_ref)
    original_root = Path(snapshot_ref["path"]).resolve().parent
    current_root = Path(__file__).resolve().parents[2]
    old_components, new_components, component_evidence = {}, {}, []
    for name, relative in _VERIFIER_COMPONENTS.items():
        original = _observed_reference(original_root / relative)
        current = _observed_reference(current_root / relative)
        _require(original["sha256"] == current["sha256"], "VERIFIER_COMPONENT_BYTES_CHANGED")
        old_components[name], new_components[name] = original, current
        component_evidence.append({"component": name, "original_ref": original, "current_ref": current})
    compilation = _document(fixture["candidate"]["compilation_ref"])
    feedback = _document(failure["compilation_feedback_ref"])
    probe_ref = selected["kwargs"]["probe_configuration_ref"]
    probe = _document(probe_ref)
    for reference in (*probe["files"], probe["dotnet"]):
        _file(reference)
    old_identity = failure["verification_identity"]
    _require(set(old_identity) == {"validator_sha256", "inputs_sha256", "environment_sha256", "deterministic_reuse_allowed"}
        and old_identity["deterministic_reuse_allowed"] is False, "VERIFIER_IDENTITY_CONTRACT_CHANGED")
    old_layers = _validator_layers(old_components, compilation["verification_identity"], probe_ref,
        selected["factory_ref"], probe["files"], old_identity)
    new_layers = _validator_layers(new_components, compilation["verification_identity"], probe_ref,
        selected["factory_ref"], probe["files"], old_identity)
    _require(_equal(old_layers["checker"], compilation["verification_identity"])
        and _equal(old_layers["compilation_feedback"], feedback["verification_identity"])
        and _equal(old_layers["probe"], old_identity), "HISTORICAL_VERIFIER_RECONSTRUCTION_FAILED")
    _require(_equal(observed, new_layers["probe"]), "CURRENT_VERIFIER_NOT_ADDRESS_EQUIVALENT")
    # Scope: derived comparison facts only, never a new execution report.
    # 为什么要有它：逐层重算证明仅地址影响validator身份，旧F1仍属原执行；新候选必须继续真实验收，不能伪造callback或复用成功结果。
    projection = {name: deepcopy(failure[name]) for name in ("status", "failure_owner", "owner_binding_ids",
        "observations", "materialized_code_identity", "failure_state_observations") if name in failure}
    projection.update(verification_identity=deepcopy(observed),
        comparison_projection_origin="retained_failure_with_verified_owner_address_relocation", new_execution=False,
        source_report_ref=deepcopy(failure["report_ref"]))
    seed = {failure_signature(projection, facts): 1}
    return {"verification_identity": deepcopy(observed), "known_failure_projection": projection,
        "no_progress_seed": seed, "provenance": {"contract": "c5-verifier-owner-address-equivalence.v1",
            "scope": "failure_state_comparison_only_not_result_reuse_or_current_plan_authority",
            "source_attempt_ref": deepcopy(fixture["checkpoint_origin"]["receipt_ref"]),
            "source_report_ref": deepcopy(failure["report_ref"]), "source_identity": deepcopy(old_identity),
            "current_identity": deepcopy(observed), "component_evidence": component_evidence,
            "historical_layers_recomputed": old_layers, "current_layers_recomputed": new_layers,
            "original_no_progress_seed": dict(checked.no_progress_seed), "current_no_progress_seed": seed,
            "historical_source_facts": facts, "original_report_modified": False, "model_calls": 0, "build_calls": 0}}
