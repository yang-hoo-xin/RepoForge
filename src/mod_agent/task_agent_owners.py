"""Concrete source and upstream owners for the continuous task session.

Scope: existing configured providers and deterministic leaf APIs, not a second Graph.
为什么要有它：B必须能真正请求A已有的Evidence/Planning能力，所有内部模型调用继续由同一预算owner预留和记录。
"""
from __future__ import annotations

import hashlib
import json
from copy import copy, deepcopy
from dataclasses import asdict, fields, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from uuid import uuid4

from src.core.trace.execution_trace import trace_operation
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.evidence import RgLiveEvidenceProvider
from src.mod_agent.evidence.compilation_candidate import compilation_candidate_scope
from src.mod_agent.evidence.reassessment import (
    RequirementCoverageReassessmentFactory,
    validate_unchanged_source_reassessment,
)
from src.mod_agent.model_call_budget import validate_budget
from src.mod_agent.patch_upstream_configuration import PatchUpstreamConfiguration
from src.mod_agent.planning.implementation_plan import ImplementationPlanner
from src.mod_agent.planning.model_selection import (
    ModelPlanningTargetSelector,
    PlanningTargetSelectionError,
)
from src.mod_agent.planning.models import FeatureSpec
from src.mod_agent.product_executor import source_snapshot, source_snapshot_sha256
from src.mod_agent.repair.planner import RepairPlanner
from src.mod_agent.task_agent_tools import (
    SourceSearchResult,
    TaskAgentContext,
    TaskContextDocuments,
    TaskUpstreamAssessment,
)


def _raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_raw(value)).hexdigest()


def _ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(reference):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"} or _ref(reference["path"]) != reference:
        raise ValueError("TASK_UPSTREAM_FILE_CHANGED")
    return json.loads(Path(reference["path"]).read_text(encoding="utf-8-sig"))


class TaskAgentOwners:
    def __init__(self, *, store: ArtifactStore, workspace_root, configuration: PatchUpstreamConfiguration | None = None,
                 initial_context: TaskAgentContext, failure_context_provider, reserve_model_call,
                 budget_snapshot, record, output_root, constraint_policy=None, providers=None, validate_providers=None,
                 current_code_view_provider=None, validate_context_transfer=None):
        self.store, self.workspace = store, Path(workspace_root).resolve(strict=True)
        self.configuration, self.initial_context = configuration, initial_context
        self.failure_context_provider, self.budget_snapshot, self.record = failure_context_provider, budget_snapshot, record
        self.output_root = Path(output_root).resolve()
        if (current_code_view_provider is not None and not callable(current_code_view_provider)
                or validate_context_transfer is not None and not callable(validate_context_transfer)):
            raise ValueError("TASK_UPSTREAM_CURRENT_CODE_OWNER_INVALID")
        self.current_code_view_provider = current_code_view_provider
        self.validate_context_transfer = validate_context_transfer
        if (not callable(failure_context_provider)
                or self.output_root == store.root or not self.output_root.is_relative_to(store.root)):
            raise ValueError("TASK_UPSTREAM_OWNER_CONFIGURATION_INVALID")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.events = []
        if providers is not None:
            # Scope: use the original Product leaves as-is, including metadata/document augmentation.
            # 为什么要有它：没有历史Fresh配置时不能伪造配置或另给B替换能力；原provider已统一计费，不再包第二层客户端。
            if configuration is not None or not callable(validate_providers):
                raise ValueError("TASK_UPSTREAM_EXPLICIT_PROVIDER_VALIDATION_REQUIRED")
            self.validate_configuration = validate_providers
            self.validate_configuration()
            self.providers = dict(providers)
        else:
            if not isinstance(configuration, PatchUpstreamConfiguration):
                raise ValueError("TASK_UPSTREAM_CONFIGURATION_REQUIRED")
            self.validate_configuration = configuration.validate_current
            self.providers = configuration.assemble(SimpleNamespace(workspace_root=str(self.workspace),
                artifact_store_root=str(store.root)), recorder=self._record, reserve_model_call=reserve_model_call,
                constraint_policy=constraint_policy)
        self.evidence_factory = self.providers["evidence_pack_factory"]
        self.selector = self.providers["planning_target_selection_factory"]
        if (not isinstance(self.evidence_factory, RequirementCoverageReassessmentFactory)
                or not isinstance(self.selector, ModelPlanningTargetSelector)
                or not isinstance(self.evidence_factory.source_provider, RgLiveEvidenceProvider)):
            raise ValueError("TASK_UPSTREAM_REAL_OWNERS_REQUIRED")
        self.source_provider = self.evidence_factory.source_provider
        self.search_root = self.evidence_factory.search_root
        self.source_root = (self.workspace / self.search_root).resolve(strict=True)
        if not self.source_root.is_relative_to(self.workspace):
            raise ValueError("TASK_UPSTREAM_SOURCE_SCOPE_INVALID")
        self.assessment_count = 0

    def _code_view(self, context):
        view = self.current_code_view_provider() if self.current_code_view_provider is not None else None
        if view is None:
            return None
        if (not isinstance(view, dict) or view.get("context") != context.context.to_dict()
                or not isinstance(view.get("files"), list) or not view["files"]
                or not isinstance(view.get("candidate_ref"), dict)
                or not Path(view["candidate_ref"]["path"]).resolve().is_relative_to(self.store.root.resolve())):
            raise ValueError("TASK_UPSTREAM_CURRENT_CODE_VIEW_INVALID")
        candidate = _read(view["candidate_ref"])
        if any(candidate.get(key) != view.get(key) for key in (
                "candidate_id", "context_id", "context", "materialized_code_identity")):
            raise ValueError("TASK_UPSTREAM_CURRENT_CODE_VIEW_CHANGED")
        paths = set()
        for item in view["files"]:
            path = PurePosixPath(item["path"])
            selected = (self.workspace / path).resolve(strict=True)
            if (path.is_absolute() or ".." in path.parts or "\\" in item["path"] or ":" in item["path"]
                    or not selected.is_relative_to(self.source_root) or not selected.is_file()
                    or item["path"] in paths or type(item.get("text")) is not str
                    or hashlib.sha256(item["text"].encode("utf-8")).hexdigest() != item.get("source_hash")):
                raise ValueError("TASK_UPSTREAM_CURRENT_CODE_SOURCE_INVALID")
            paths.add(item["path"])
        return deepcopy(view)

    def _current_source_provider(self, view, baseline):
        """Materialize a private immutable source view without changing the Apply preimage.

        为什么要有它：搜索必须观察当前候选；文件引用仍指向真实字节，不把候选哈希伪挂到原始文件上。
        """
        key = _sha({"baseline": baseline, "code": view["materialized_code_identity"]})
        workspace = self.output_root / ("current-source-" + key[:24])
        target = workspace / self.search_root
        if (not workspace.resolve().is_relative_to(self.output_root)
                or not target.resolve().is_relative_to(workspace.resolve())):
            raise ValueError("TASK_UPSTREAM_CURRENT_VIEW_OUTSIDE_SCOPE")
        files = {}
        for row in baseline:
            original = (self.source_root / row["path"]).resolve(strict=True)
            if not original.is_relative_to(self.source_root):
                raise ValueError("TASK_UPSTREAM_SOURCE_OUTSIDE_SCOPE")
            raw = original.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("TASK_UPSTREAM_SOURCE_CHANGED")
            files[row["path"]] = raw
        for item in view["files"]:
            relative = (self.workspace / item["path"]).resolve(strict=True).relative_to(self.source_root).as_posix()
            if relative not in files:
                raise ValueError("TASK_UPSTREAM_CURRENT_CODE_OUTSIDE_BASELINE")
            files[relative] = item["text"].encode("utf-8")
        expected = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        if not workspace.exists():
            workspace.mkdir(parents=True, exist_ok=False)
            for name, raw in files.items():
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(raw)
        if {row["path"]: row["sha256"] for row in source_snapshot(target)} != expected:
            raise ValueError("TASK_UPSTREAM_CURRENT_VIEW_CHANGED")
        provider = RgLiveEvidenceProvider(workspace, rg_executable=self.source_provider.rg_executable,
            small_file_max_bytes=self.source_provider.small_file_max_bytes,
            small_project_context_policy=self.source_provider.small_project_context_policy,
            reuse_query_observations=self.source_provider.reuse_query_observations)
        return workspace, provider

    @staticmethod
    def _model_code_view(view):
        if view is None:
            return None
        history = view.get("last_checked_failure")
        history = ({key: deepcopy(value) for key, value in history.items() if not key.endswith("_ref")}
                   | {key: {"sha256": value["sha256"]} for key, value in history.items()
                      if key.endswith("_ref") and isinstance(value, dict)}) if history else None
        return {"candidate_id": view["candidate_id"], "context_id": view["context_id"],
            "candidate_sha256": view["candidate_ref"]["sha256"],
            "materialized_code_identity": deepcopy(view["materialized_code_identity"]),
            "files": deepcopy(view["files"]),
            "last_checked_failure": history,
            "source_semantics": "These are current candidate bytes. Formal Evidence/Plan source items remain the immutable Apply preimage and binding scope, not the current candidate."}

    def _current_reassessment(self, view, baseline, observations, failure_version):
        """Keep formal edit preimages while investigating the actual current code.

        为什么要有它：候选搜索结果不能直接充当原工程的修改前像；同一查询的当前观察单独交给原语义模型，不增加模型角色或预算。
        """
        factory = copy(self.evidence_factory)
        _workspace, current = self._current_source_provider(view, baseline)
        original = self.source_provider
        query_selector, semantic = factory.query_selector, factory.semantic_provider
        current_view = self._model_code_view(view)

        class SourceProvider:
            def collect_query_set(self, **kwargs):
                observed = current.collect_query_set(**kwargs).to_artifact_payload()
                observations.append({"queries": deepcopy(kwargs["queries"]),
                    "candidate_sha256": view["candidate_ref"]["sha256"],
                    "materialized_code_identity": deepcopy(view["materialized_code_identity"]),
                    "items": [{key: deepcopy(item[key]) for key in
                        ("evidence_id", "source_path", "source_hash", "excerpt", "span") if key in item}
                        for item in observed["items"]],
                    "execution": deepcopy(observed.get("budget_summary", {})),
                    "scope": "current_candidate_observation_not_original_edit_binding"})
                return original.collect_query_set(**kwargs)

        class SemanticProvider:
            def __getattr__(self, name):
                return getattr(semantic, name)

            def findings(self, **kwargs):
                kwargs["semantic_feedback"] = {**kwargs.get("semantic_feedback", {}),
                    "current_code_view": deepcopy(current_view),
                    "failure_version": deepcopy(failure_version),
                    "current_source_observations": deepcopy(observations)}
                return semantic.findings(**kwargs)

        factory.source_provider = SourceProvider()
        factory.query_selector = lambda feature, projection, old, feedback: query_selector(
            feature, projection, old, {**feedback, "current_code_view": deepcopy(current_view),
                                      "failure_version": deepcopy(failure_version)})
        factory.semantic_provider = SemanticProvider()
        return factory

    def _record(self, event):
        self.events.append(deepcopy(event))
        if self.record is not None:
            self.record(deepcopy(event))

    def _save(self, kind, value):
        path = self.output_root / (kind + "-" + uuid4().hex + ".json")
        with path.open("xb") as stream:
            stream.write(_raw(value))
        return _ref(path)

    def _retain_planning_receipt(self, selector):
        # Scope: an immutable copy of the actual configured planner's receipt, not expanded path authority.
        # 为什么要有它：正常入口可将规划回执写在run目录；交付仍从任务Store核对副本，不因目录差异拒绝合法候选。
        reference = _ref(selector.receipt_path)
        return self._save("planning-owner-result", {"source_receipt_ref": reference, "receipt": _read(reference)})

    def _context(self, context):
        self.validate_configuration()
        if not isinstance(context, TaskContextDocuments) or context.context.feature_spec_ref != self.initial_context.feature_spec_ref:
            raise ValueError("TASK_UPSTREAM_ORIGINAL_REQUIREMENTS_CHANGED")
        for name in ("feature_spec", "evidence_pack", "implementation_plan"):
            ref = getattr(context.context, name + "_ref")
            if self.store.semantic_read_immutable(ref)["data"] != getattr(context, name) or not self.store.check_exact_revision_freshness(ref).is_fresh:
                raise ValueError("TASK_UPSTREAM_CONTEXT_CHANGED")
        return source_snapshot(self.source_root)

    def source_search(self, *, query, context):
        before = self._context(context)
        view = self._code_view(context)
        selected_workspace, provider = (self._current_source_provider(view, before) if view is not None
                                        else (self.workspace, self.source_provider))
        with trace_operation("task.source.search", kind="tool", attributes={"query_sha256": hashlib.sha256(query.encode()).hexdigest()}):
            pack = provider.collect(query=query, search_root=self.search_root,
                max_items=self.evidence_factory.max_additional_queries)
        if source_snapshot(self.source_root) != before or self._code_view(context) != view:
            raise ValueError("TASK_UPSTREAM_SOURCE_CHANGED")
        payload = pack.to_artifact_payload()
        current_paths = {item["path"] for item in view["files"]} if view is not None else set()
        sources = {}
        for item in payload["items"]:
            logical = item["source_path"]
            current = logical in current_paths
            selected = ((selected_workspace if current else self.workspace) / logical).resolve(strict=True)
            if not selected.is_relative_to((selected_workspace / self.search_root).resolve() if current else self.source_root):
                raise ValueError("TASK_UPSTREAM_SEARCH_OUTSIDE_SCOPE")
            ref = _ref(selected)
            if ref["sha256"] != item["source_hash"]:
                raise ValueError("TASK_UPSTREAM_SEARCH_SOURCE_CHANGED")
            item["source_ref"] = ref
            if current:
                item.update(source_path=str(selected), logical_source_path=logical)
            sources[ref["path"]] = ref
        if view is not None:
            payload["current_code_view"] = {"candidate_ref": view["candidate_ref"],
                                             "materialized_code_identity": view["materialized_code_identity"]}
        refs = tuple(sources.values())
        reference = self._save("source-search", {"query": query, "payload": payload, "source_refs": list(refs),
            "source_snapshot_sha256": source_snapshot_sha256(before), "source_provider": "RgLiveEvidenceProvider.collect",
            "semantic_evidence_or_plan_updated": False})
        return SourceSearchResult(query, payload, refs, reference)

    def upstream(self, *, request, context):
        source_before = self._context(context)
        current_view = self._code_view(context)
        current_observations = []
        budget_before = validate_budget(self.budget_snapshot())
        failure_context = deepcopy(self.failure_context_provider())
        selected = {name: asdict(getattr(context.context, name + "_ref")) for name in
                    ("feature_spec", "evidence_pack", "implementation_plan")}
        actual = failure_context.get("actual_upstream_arguments", {})
        if (set(request) != {"required_owner", "reason", "source_refs"}
                or request["required_owner"] not in {"evidence", "planning"}
                or type(request["reason"]) is not str or not 0 < len(request["reason"].strip()) <= 2000
                or not isinstance(request["source_refs"], list) or not request["source_refs"]
                or any(type(ref) is not str or ref not in failure_context.get("allowed_source_refs", []) for ref in request["source_refs"])
                or len(set(request["source_refs"])) != len(request["source_refs"])
                or actual.get("request") != request or failure_context.get("selected_artifacts") != selected
                or failure_context.get("implementation_plan") != context.implementation_plan):
            raise ValueError("TASK_UPSTREAM_REQUEST_NOT_BOUND")
        worker_ref = failure_context["source_worker_response_ref"]
        worker = _read(worker_ref)
        if not any(row["name"] == "upstream" and row["call_id"] == failure_context["actual_upstream_call_id"]
                   and row["arguments"] == actual for row in worker["tool_calls"]):
            raise ValueError("TASK_UPSTREAM_ACTUAL_MODEL_REQUEST_REQUIRED")
        report = failure_context["failure_report"]
        if report is not None and _read(failure_context["failure_report_ref"]) != report:
            raise ValueError("TASK_UPSTREAM_FAILURE_REPORT_CHANGED")
        failure_version = {"origin": failure_context["origin"],
            "context_id": failure_context.get("failure_context_id"),
            "eligible_for_current_completion": False,
            **{name: {"sha256": failure_context[name]["sha256"]} if failure_context.get(name) else None
               for name in ("failure_candidate_ref", "failure_check_ref", "failure_report_ref")}}
        context_payload = {**failure_context, "workspace_snapshot_sha256": source_snapshot_sha256(source_before)}
        upstream = {**deepcopy(request), "origin": "model", "source_worker_response_ref": worker_ref,
            "failure_report_ref": failure_context["failure_report_ref"],
            "source_attempt_ref": None, "preserved_plan_boundaries": deepcopy(context.implementation_plan["side_effect_boundaries"])}
        self.assessment_count += 1
        repair = RepairPlanner().from_product_patch_upstream_request(context=context_payload, upstream=upstream,
            feature_id=context.context.feature_spec_ref.feature_id, target_behavior=context.feature_spec["target_behavior"],
            retry_count=self.assessment_count - 1)
        # Scope: annotate this task's diagnostic provenance without changing the old workflow contract.
        # 为什么要有它：历史检查可帮助再次调查，但新写记录不能让它看起来像当前候选已经重新验收。
        repair = replace(repair, repair_context={**repair.repair_context,
            "diagnostic_origin": failure_context["origin"],
            "failed_candidate_ref": deepcopy(failure_context.get("failure_candidate_ref")),
            "failed_check_ref": deepcopy(failure_context.get("failure_check_ref")),
            "failed_context_id": failure_context.get("failure_context_id"),
            "eligible_for_current_completion": False,
            "current_failure_available_scope": "legacy field denotes diagnostic report presence, not current candidate acceptance"})
        repair_ref = self.store.write(feature_id=context.context.feature_spec_ref.feature_id, artifact_type="repair_plan",
            payload=repair.to_artifact_payload(), upstream_refs=[getattr(context.context, name + "_ref") for name in selected])
        state = {"feature_id": context.context.feature_spec_ref.feature_id, "user_request": context.feature_spec["user_request"],
            "target_mod": context.feature_spec["target_mod"],
            "artifacts": {**{name: ref["path"] for name, ref in selected.items()}, "repair_plan": repair_ref.path},
            "artifact_hashes": {**{name: ref["artifact_hash"] for name, ref in selected.items()}, "repair_plan": repair_ref.artifact_hash}}
        owner_refs = [_ref(repair_ref.path)]
        evidence_ref, evidence = context.context.evidence_pack_ref, deepcopy(context.evidence_pack)
        pending_evidence = None
        outcome, feedback = "retained_context", {}
        if request["required_owner"] == "evidence":
            factory = (self._current_reassessment(current_view, source_before, current_observations, failure_version)
                       if current_view is not None else self.evidence_factory)
            pack = factory(state, deepcopy(context.feature_spec))
            if current_view is not None:
                owner_refs.append(self._save("current-source-observations", {
                    "current_code_view": self._model_code_view(current_view), "observations": current_observations}))
            owner_refs.append(self._save("evidence-owner-result", pack.to_artifact_payload()))
            reuse = validate_unchanged_source_reassessment(pack, state=state, artifact_store=self.store,
                workspace_root=self.workspace, feature_data=context.feature_spec)
            if reuse is not None:
                feedback = {"decision": "retain_current_plan", "reason": "Executed source recheck returned no new facts; existing judgments remain bound.",
                            "semantic_repair_proven": False}
                if current_view is not None:
                    feedback.update(reason="Original edit-binding evidence did not change. Current-candidate search observations are separate diagnostic evidence.",
                        current_source_observations=deepcopy(current_observations), failure_version=failure_version)
                return self._finish(request, context, context.context, budget_before, source_before, owner_refs,
                                    outcome, feedback)
            evidence = pack.to_artifact_payload()
            if not pack.can_generate_patch and compilation_candidate_scope(evidence, context.feature_spec) is None:
                return self._finish(request, context, context.context, budget_before, source_before, owner_refs,
                    "blocked", {"decision": "needs_evidence", "gaps": list(pack.gaps), "blockers": list(pack.blockers)})
            pending_evidence = (evidence, "success" if pack.can_generate_patch else "blocked")
        selector = self.selector.for_product_assessment(repair_ref.artifact_hash)
        same_evidence = pending_evidence is None
        from src.mod_agent.proposal.behavior_feedback_view import project_execution_reference_paths

        model_report = project_execution_reference_paths(report)
        if model_report is not None:
            # Only task failure observations go to the model, never verifier implementation material.
            model_report = {key: model_report.get(key) for key in ("status", "failure_owner", "owner_binding_ids",
                "observations", "generated_changes_sha256", "source_snapshot_sha256", "materialized_code_identity")}
        selector.semantic_feedback = (*selector.semantic_feedback, *tuple(context.implementation_plan["side_effect_boundaries"]),
            json.dumps({"recommendation": deepcopy(request), "observed_failure": model_report,
                        "recommendation_is_not_observation": True}, ensure_ascii=False, sort_keys=True))
        selector.upstream_assessment = {"current_plan": deepcopy(context.implementation_plan),
            "selected_artifact_hashes": {name: ref["artifact_hash"] for name, ref in selected.items()},
            "repair_plan_hash": repair_ref.artifact_hash, "previous_response": failure_context["previous_response"],
            "failed_response_sha256": (failure_context["failed_response_ref"] or {}).get("sha256"),
            "failure_report": model_report, "failure_report_sha256": (failure_context["failure_report_ref"] or {}).get("sha256"),
            "generated_changes_sha256": failure_context["generated_changes_sha256"],
            "source_snapshot_sha256": failure_context["source_snapshot_sha256"],
            "current_failure_available": failure_context["origin"] == "current_checked_candidate",
            "diagnostic_failure_available": report is not None, "failure_version": failure_version,
            "retain_current_plan_allowed": same_evidence, "approval_inherited": False}
        if current_view is not None:
            selector.upstream_assessment.update(current_code_view=self._model_code_view(current_view),
                current_source_observations=deepcopy(current_observations))
        try:
            targets = selector(deepcopy(context.feature_spec), deepcopy(evidence))
        except PlanningTargetSelectionError as error:
            if error.code != "PLANNING_MODEL_NEEDS_EVIDENCE":
                raise
            owner_refs.append(self._retain_planning_receipt(selector))
            return self._finish(request, context, context.context, budget_before, source_before, owner_refs,
                "blocked", {"decision": "needs_evidence", "blockers": [error.as_blocker()]})
        owner_refs.append(self._retain_planning_receipt(selector))
        if selector.selection_result["decision"] == "retain_current_plan":
            if not same_evidence:
                raise ValueError("TASK_UPSTREAM_RETAINED_PLAN_WITH_CHANGED_EVIDENCE")
            return self._finish(request, context, context.context, budget_before, source_before, owner_refs,
                "retained_context", selector.selection_result)
        feature = FeatureSpec(**{field.name: deepcopy(context.feature_spec[field.name]) for field in fields(FeatureSpec)
                                 if field.name in context.feature_spec})
        plan = ImplementationPlanner().build(feature_spec=feature, evidence_pack=evidence,
            formal_requirement_coverage_required=True, planning_target_evidence_ids=targets,
            constraint_policy=selector.constraint_policy)
        if plan.blocking:
            owner_refs.append(self._save("blocked-plan", plan.to_artifact_payload()))
            return self._finish(request, context, context.context, budget_before, source_before, owner_refs,
                "blocked", {"decision": "planning_blocked", "blockers": list(plan.blockers)})
        boundaries = tuple(dict.fromkeys((*plan.side_effect_boundaries, *context.implementation_plan["side_effect_boundaries"])))
        plan = replace(plan, verification_questions=selector.verification_questions, side_effect_boundaries=boundaries)
        if plan.constraint_context is not None:
            from src.mod_agent.planning.constraint_context import build_constraint_context

            prior = plan.constraint_context
            constraints = build_constraint_context(feature_spec=context.feature_spec, evidence_pack=evidence,
                plan_boundaries=prior["source_plan_boundaries"],
                retained_hard_boundaries=(*prior["retained_hard_boundaries"], *context.implementation_plan["side_effect_boundaries"]))
            plan = replace(plan, constraint_context=constraints, side_effect_boundaries=tuple(constraints["hard_boundaries"]))
        if self._code_view(context) != current_view:
            raise ValueError("TASK_UPSTREAM_CURRENT_CODE_VIEW_CHANGED")
        if current_view is not None:
            if self.validate_context_transfer is None:
                raise ValueError("TASK_UPSTREAM_CONTEXT_TRANSFER_PREFLIGHT_REQUIRED")
            try:
                self.validate_context_transfer(feature_spec=deepcopy(context.feature_spec),
                    evidence_pack=deepcopy(evidence), implementation_plan=plan.to_artifact_payload())
            except ValueError as error:
                if str(error) not in {"TASK_CURRENT_TRANSFER_SCOPE_NOT_COVERED", "TASK_CURRENT_TRANSFER_AMBIGUOUS_SCOPE"}:
                    raise
                owner_refs.append(self._save("unmapped-plan", plan.to_artifact_payload()))
                return self._finish(request, context, context.context, budget_before, source_before, owner_refs,
                    "retained_context", {"decision": "proposed_plan_cannot_preserve_current_code",
                        "reason": str(error), "current_code_preserved": True, "proposed_context_published": False,
                        "next_action": "Continue investigating or propose a plan covering the existing modifications."})
        # Scope: publish a new Evidence head only once its matching Plan is ready.
        # 为什么要有它：不足或规划阻断先留真实owner sidecar，不能先移动Evidence head再把已过期旧Plan说成有效保留。
        if pending_evidence is not None:
            evidence_ref = self.store.write(feature_id=state["feature_id"], artifact_type="evidence_pack",
                payload=pending_evidence[0], status=pending_evidence[1],
                upstream_refs=[context.context.feature_spec_ref], trace_refs=[repair_ref.path])
            owner_refs.append(_ref(evidence_ref.path))
        plan_ref = self.store.write(feature_id=state["feature_id"], artifact_type="implementation_plan",
            payload=plan.to_artifact_payload(), upstream_refs=[context.context.feature_spec_ref, evidence_ref],
            trace_refs=selector.planning_trace_refs())
        owner_refs.append(_ref(plan_ref.path))
        updated = TaskAgentContext(context.context.feature_spec_ref, evidence_ref, plan_ref)
        return self._finish(request, context, updated, budget_before, source_before, owner_refs,
                            "replanned", selector.selection_result)

    def _finish(self, request, context, updated, budget_before, source_before, owner_refs, outcome, feedback):
        self.validate_configuration()
        if source_snapshot(self.source_root) != source_before:
            raise ValueError("TASK_UPSTREAM_SOURCE_CHANGED")
        budget_after = validate_budget(self.budget_snapshot())
        result = {"input_context": context.context.to_dict(), "output_context": updated.to_dict(),
            "request_sha256": _sha(request), "budget_before": budget_before, "budget_after": budget_after,
            "owner_result_refs": owner_refs, "outcome": outcome, "model_feedback": deepcopy(feedback),
            "source_snapshot_sha256": source_snapshot_sha256(source_before),
            "source_modified": False, "approval_inherited": False, "model_history_reset": False}
        reference = self._save("upstream-assessment", result)
        self._record({"phase": "task_upstream_completed", "outcome": outcome, "assessment_ref": reference,
                      "new_model_calls": budget_after["consumed"] - budget_before["consumed"]})
        return TaskUpstreamAssessment(updated, reference)


def make_task_agent_owners(**kwargs):
    return TaskAgentOwners(**kwargs)
