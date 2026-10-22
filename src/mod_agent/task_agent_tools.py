"""Task-session owners over existing source, patch and qualified evaluator APIs.

Scope: isolated candidate preparation/checks; no source Apply or game authority.
为什么要有它：连续会话必须消费真实owner产物，不能把模型计划、任意success字典或检查器文本当执行权限和验收。
"""
from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from src.core.trace.execution_trace import trace_operation
from src.libs.llm import ChatResponse
from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentToolDefinition,
    AgentToolExchange,
    AgentTurnRequest,
    AgentTurnResponse,
)
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.candidate_execution import (
    CandidateExecutionError,
    invoke_candidate_evaluator,
    materialize_candidate,
    materialized_code_identity,
    validate_candidate_report,
    verification_identity,
)
from src.mod_agent.comparison_checkpoint import CheckedComparisonCheckpoint
from src.mod_agent.current_candidate import (
    candidate_projection_contract,
    context_transfer_content,
    derive_current_revision,
    validate_current_candidate_contract,
)
from src.mod_agent.evidence.compilation_candidate import compilation_candidate_scope
from src.mod_agent.model_call_budget import is_meter_only, validate_budget
from src.mod_agent.proposal.code_generator import (
    LLMExactPatchGenerator,
    PatchGenerationError,
    PreparedNativePatchContract,
    validate_patch_submission,
)
from src.mod_agent.proposal.failure_state import failure_signature, source_fact_hashes
from src.mod_agent.proposal.observation_value_refs import project_observation_values
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy, PlanBoundSourceInspector
from src.mod_agent.task_agent_loop import (
    PreparedTaskAction,
    TaskActionInputError,
    TaskAgentAction,
    TaskCandidateRevisionInputError,
    run_task_agent_loop,
)


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _ref(path):
    selected = Path(path).resolve(strict=True)
    return {"path": str(selected), "sha256": hashlib.sha256(selected.read_bytes()).hexdigest()}


def _read(reference):
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"} or _ref(reference["path"]) != dict(reference):
        raise ValueError("TASK_TOOL_FILE_BINDING_CHANGED")
    return json.loads(Path(reference["path"]).read_text(encoding="utf-8-sig"))


def _candidate_submit_forms():
    """Use the same exact forms for model disclosure and pre-dispatch validation.

    为什么要有它：首候选和当前候选修订互斥，不能公开只要求context_id的宽松并集却在执行时拒绝其合法形状。
    """
    text = {"type": "string", "minLength": 1}
    initial = {"context_id": text, "patch_json": text}
    edit = {"type": "object", "additionalProperties": False,
        "properties": {"edit_binding_id": text, "before_text": text, "after_text": {"type": "string"}, "rationale": text},
        "required": ["edit_binding_id", "before_text", "after_text", "rationale"]}
    revision = {"context_id": text, "parent_candidate_id": text, "parent_code_sha256": text,
        "edits": {"type": "array", "minItems": 1, "items": edit}}
    return tuple({"type": "object", "additionalProperties": False,
        "properties": properties, "required": list(properties)} for properties in (initial, revision))


def _submission_format_error(content, schema, *, field):
    """Return schema-only diagnostics; identity and semantic failures are not caught."""
    try:
        validate_patch_submission(content, schema)
    except PatchGenerationError as error:
        expected = schema
        for part in error.diagnostics.get("schema_path", []):
            expected = expected.get(part) if isinstance(expected, Mapping) else None
        constraint = error.diagnostics.get("schema_constraint")
        hint = (f" Expected {constraint}: {json.dumps(expected)}."
            if constraint in {"type", "required", "minItems", "maxItems", "minLength", "additionalProperties"}
            and expected is not None else "")
        return f"{field}: {error}{hint} Allowed fields: {', '.join(schema['properties'])}."
    return None


@dataclass(frozen=True)
class TaskAgentContext:
    feature_spec_ref: ArtifactRef
    evidence_pack_ref: ArtifactRef
    implementation_plan_ref: ArtifactRef

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TaskContextDocuments:
    context: TaskAgentContext
    feature_spec: dict[str, Any]
    evidence_pack: dict[str, Any]
    implementation_plan: dict[str, Any]


@dataclass(frozen=True)
class SourceSearchResult:
    query: str
    payload: dict[str, Any]
    source_refs: tuple[dict[str, str], ...]
    sidecar_ref: dict[str, str]


def _model_search_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Project one verified search item into the technical model conversation.

    为什么要有它：完整检索原件需要保留以便归因和重规划，但重复的本机路径、索引元数据和声明清单不应挤掉模型继续决策所需的上下文。
    """
    values = {
        "evidence_id": item.get("evidence_id"),
        "source_type": item.get("source_type"),
        "source_path": item.get("logical_source_path", item.get("source_path")),
        "source_hash": item.get("source_hash"),
        "symbol": item.get("symbol"),
        "span": item.get("span"),
        "excerpt": item.get("excerpt"),
        "truncated": item.get("truncated"),
        "retrieval_method": item.get("retrieval_method"),
        "score": item.get("score"),
        "claim_supported": item.get("claim_supported"),
        "reassessment_ref": item.get("reassessment_ref"),
    }
    return {key: deepcopy(value) for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class TaskUpstreamAssessment:
    context: TaskAgentContext
    assessment_ref: dict[str, str]


class TaskAgentSession:
    """Own versioned task facts; the core retains the single native conversation."""

    def __init__(self, *, store: ArtifactStore, workspace_root: str | Path, context: TaskAgentContext,
                 policy: PatchAgentLoopPolicy, generator: LLMExactPatchGenerator, evaluator,
                 allowed_edit_paths: tuple[str, ...], allowed_read_roots: tuple[str | Path, ...],
                 output_root: str | Path, prepare_contract: Callable[[TaskContextDocuments], PreparedNativePatchContract],
                 source_search: Callable[..., SourceSearchResult] | None,
                 upstream: Callable[..., TaskUpstreamAssessment] | None,
                 reserve_model_call: Callable[[], Any], budget_snapshot: Callable[[], dict[str, Any]],
                 charge_tool_budget: Callable[..., Any],
                 model_response_admission: Callable[[AgentTurnResponse], None] | None,
                 record_event: Callable[[dict[str, Any]], None] | None = None,
                 read_budget_snapshot: Callable[[], dict[str, Any]] | None = None,
                 candidate_mode: str = "baseline"):
        self.store, self.workspace_root = store, Path(workspace_root).resolve(strict=True)
        self.output_root = Path(output_root).resolve()
        if self.output_root == store.root or not self.output_root.is_relative_to(store.root):
            raise ValueError("TASK_TOOL_OUTPUT_OUTSIDE_STORE")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.policy, self.generator, self.evaluator = policy, generator, evaluator
        if candidate_mode not in {"baseline", "current"}:
            raise ValueError("TASK_TOOL_CANDIDATE_MODE_INVALID")
        self.candidate_mode = candidate_mode
        self.prepare_contract = prepare_contract
        self.source_search_owner, self.upstream_owner = source_search, upstream
        self.reserve_model_call, self.budget_snapshot, self.charge_tool_budget = reserve_model_call, budget_snapshot, charge_tool_budget
        self.model_response_gate, self.record_event_owner = model_response_admission, record_event
        if read_budget_snapshot is not None and not callable(read_budget_snapshot):
            raise ValueError("TASK_TOOL_READ_BUDGET_OWNER_REQUIRED")
        self.read_budget_snapshot = read_budget_snapshot
        self.allowed_read_roots = tuple(Path(root).resolve(strict=True) for root in allowed_read_roots)
        if not self.allowed_read_roots or any(not root.is_dir() for root in self.allowed_read_roots):
            raise ValueError("TASK_TOOL_READ_SCOPE_INVALID")
        self.allowed_edit_paths = frozenset(allowed_edit_paths)
        if not self.allowed_edit_paths:
            raise ValueError("TASK_TOOL_EDIT_SCOPE_EMPTY")
        self.edit_baselines = {path: _ref(self._edit_path(path))["sha256"] for path in self.allowed_edit_paths}
        self.initial_feature_ref = context.feature_spec_ref
        self.evaluator_identity = verification_identity(evaluator)
        self.epoch = 0
        self.documents, self.inspector, self.contract = self._admit_context(context)
        if self.candidate_mode == "current":
            validate_current_candidate_contract(self.documents.implementation_plan, self.contract)
        self.context = context
        self._candidate_contexts = {self.context_id: (context, self.documents, self.inspector, self.contract)}
        self.candidates: dict[str, dict[str, Any]] = {}
        self.current_candidate: str | None = None
        self.current_check: dict[str, Any] | None = None
        self.last_checked_failure: dict[str, Any] | None = None
        self.check_count = 0
        self._check_inflight = False
        self._actual_calls: list[dict[str, Any]] = []
        self._checkpoint = None
        self._checkpoint_failure = None
        self._checkpoint_content_ref = None
        self._failure_counts: dict[str, int] = {}
        self._source_facts: set[str] = set()
        self._pending_source_observations: list[dict[str, Any]] = []
        self._search_reassessment_observations: dict[str, dict[str, Any]] = {}
        self._known_stop = None
        self._upstream_blocked = False

    def current_code_view(self):
        """Trusted read-only overlay; original paths remain Apply preimages.

        为什么要有它：调查与重规划必须能看到当前代码，同时不能把候选副本当成原工程已修改。
        """
        if self.candidate_mode != "current" or self.current_candidate is None:
            return None
        candidate = self._candidate(self.current_candidate)
        _context, documents, _inspector, _contract = self._candidate_contexts[candidate["context_id"]]
        materialized = materialize_candidate(workspace_root=self.workspace_root,
            implementation_plan=documents.implementation_plan, changes=candidate["generated_changes"])
        return {"candidate_id": candidate["candidate_id"], "candidate_ref": deepcopy(self.candidates[self.current_candidate]["ref"]),
            "context_id": candidate["context_id"], "context": deepcopy(candidate["context"]),
            "materialized_code_identity": deepcopy(candidate["materialized_code_identity"]),
            "last_checked_failure": self._failure_history()[0],
            "files": [{"path": path, "source_hash": hashlib.sha256(raw).hexdigest(), "text": raw.decode("utf-8")}
                for path, raw in sorted(materialized.files.items())]}

    def _failure_history(self):
        """Authenticate historical diagnosis without restoring current acceptance.

        为什么要有它：重规划清除旧检查的交付资格，不应丢掉最近失败；历史候选、报告和版本仍须可逐项核验。
        """
        if self.last_checked_failure is None:
            return None, None, None, None
        history = deepcopy(self.last_checked_failure)
        checked = _read(history["check_ref"])
        candidate = _read(history["candidate_ref"])
        stored = self.candidates.get(history["candidate_id"])
        version = self._candidate_contexts.get(history["context_id"])
        if (stored is None or version is None or stored["ref"] != history["candidate_ref"] or stored["data"] != candidate
                or candidate["candidate_id"] != history["candidate_id"] or candidate["context_id"] != history["context_id"]
                or candidate["context"] != version[0].to_dict()
                or checked["candidate_id"] != history["candidate_id"] or checked["context_id"] != history["context_id"]
                or checked["candidate_ref"] != history["candidate_ref"]
                or checked["report"]["report_ref"] != history["report_ref"]
                or history["materialized_code_identity"] != candidate["materialized_code_identity"]):
            raise ValueError("TASK_TOOL_HISTORICAL_FAILURE_CHANGED")
        version[2].validate_snapshot()
        raw = _read(history["report_ref"])
        validate_candidate_report({**raw, "report_ref": history["report_ref"]},
            changes_sha256=candidate["generated_changes_sha256"],
            source_snapshot_sha256=version[2].source_snapshot_sha256, allowed_binding_ids=version[2].available_binding_ids)
        if raw.get("status") not in {"behavior_failed", "needs_evidence", "infrastructure_failed", "identity_failed"}:
            raise ValueError("TASK_TOOL_HISTORICAL_FAILURE_CHANGED")
        history["eligible_for_current_completion"] = False
        return history, checked, candidate, raw

    def validate_context_transfer(self, *, feature_spec, evidence_pack, implementation_plan):
        """Preflight before an owner publishes new heads; never writes or checks.

        为什么要有它：无法承接旧修改的计划不能先发布再让原上下文失效，阻断后仍应保留可继续调查的当前代码。
        """
        if self.candidate_mode != "current" or self.current_candidate is None:
            return
        if feature_spec != self.documents.feature_spec:
            raise ValueError("TASK_TOOL_REQUIREMENTS_CHANGED")
        for binding in implementation_plan.get("binding_registry", {}).get("edit_bindings", []):
            path = binding.get("canonical_target_path")
            if path not in self.allowed_edit_paths or binding.get("source_hash") != self.edit_baselines[path]:
                raise ValueError("TASK_TOOL_EDIT_AUTHORITY_CHANGED")
        documents = TaskContextDocuments(self.context, deepcopy(feature_spec), deepcopy(evidence_pack), deepcopy(implementation_plan))
        contract = self.prepare_contract(documents)
        self._transfer_materialization(documents, contract)

    def _transfer_materialization(self, documents, contract):
        parent = self._candidate(self.current_candidate)
        _context, old_documents, _inspector, _contract = self._candidate_contexts[parent["context_id"]]
        old = materialize_candidate(workspace_root=self.workspace_root,
            implementation_plan=old_documents.implementation_plan, changes=parent["generated_changes"])
        content, projection = context_transfer_content(baseline_files=old.baseline_files, current_files=old.files,
            implementation_plan=documents.implementation_plan, prepared_contract=contract)
        generated = self.generator.materialize_prepared_native(projection,
            ChatResponse(content=content, model=parent["model_response"]["model"]))
        materialized = materialize_candidate(workspace_root=self.workspace_root,
            implementation_plan=documents.implementation_plan, changes=generated.changes)
        if materialized.files != old.files or materialized.baseline_files != old.baseline_files:
            raise ValueError("TASK_CURRENT_TRANSFER_CODE_CHANGED")
        return parent, content, generated, materialized

    def _selected_artifacts(self):
        return {name: asdict(getattr(self.context, name + "_ref")) for name in
                ("feature_spec", "evidence_pack", "implementation_plan")}

    def bind_comparison_checkpoint(self, checked, *, revalidate, admit_verifier):
        """Import checked diagnostic history and its one failure, never a new candidate.

        Scope: the normal checkpoint helper and explicit verifier-equivalence owner are callbacks.
        为什么要有它：只读重核输入和当前检查器后才能继承失败计数；原C1/F1不能伪造为新toolcall或新通过结果。
        """
        if self._checkpoint is not None or self.candidates or self._actual_calls:
            raise ValueError("TASK_TOOL_CHECKPOINT_ALREADY_USED")
        current = revalidate()
        if (not isinstance(checked, CheckedComparisonCheckpoint) or not isinstance(current, CheckedComparisonCheckpoint)
                or _sha(asdict(current)) != _sha(asdict(checked))
                or current.provenance.get("current_artifacts") != self._selected_artifacts()):
            raise ValueError("TASK_TOOL_CHECKPOINT_IDENTITY_CHANGED")
        worker = _read(current.worker_configuration_ref)
        compilation = worker["behavior_evaluator"]["kwargs"]["compilation_kwargs"]
        if (Path(compilation["workspace_root"]).resolve() != self.workspace_root
                or Path(compilation["artifact_store_root"]).resolve() != self.store.root
                or replace(self.policy, max_total_model_turns=current.worker_policy.max_total_model_turns) != current.worker_policy):
            raise ValueError("TASK_TOOL_CHECKPOINT_OWNER_CHANGED")
        self.inspector.validate_snapshot()
        admitted = admit_verifier(current, self.evaluator)
        identity = verification_identity(self.evaluator)
        projection = admitted.get("known_failure_projection") if isinstance(admitted, Mapping) else None
        if (identity is None or admitted.get("verification_identity") != identity or not isinstance(projection, Mapping)
                or projection.get("verification_identity") != identity
                or not isinstance(admitted.get("provenance"), Mapping) or not admitted["provenance"]):
            raise ValueError("TASK_TOOL_CHECKPOINT_VERIFIER_NOT_EQUIVALENT")
        signature = failure_signature(projection, current.source_facts)
        if admitted.get("no_progress_seed") != {signature: 1}:
            raise ValueError("TASK_TOOL_CHECKPOINT_FAILURE_SEED_CHANGED")
        original = _read(current.known_failure["report_ref"])
        for key in ("status", "failure_owner", "owner_binding_ids", "observations", "generated_changes_sha256", "source_snapshot_sha256"):
            if original.get(key) != current.known_failure.get(key):
                raise ValueError("TASK_TOOL_CHECKPOINT_FAILURE_CHANGED")
        for key in ("status", "failure_owner", "owner_binding_ids", "observations", "materialized_code_identity", "failure_state_observations"):
            if key in projection and projection[key] != current.known_failure.get(key):
                raise ValueError("TASK_TOOL_CHECKPOINT_COMPARISON_CHANGED")
        assistants = [message.content for message in current.request.messages if message.role == "assistant"]
        if len(assistants) != 1 or not isinstance(assistants[0], str):
            raise ValueError("TASK_TOOL_CHECKPOINT_CANDIDATE_HISTORY_INVALID")
        path = self.output_root / ("historical-assistant-content-" + uuid4().hex + ".txt")
        with path.open("xb") as stream:
            stream.write(assistants[0].encode())
        self._checkpoint_content_ref = _ref(path)
        self._checkpoint = deepcopy(current)
        self._checkpoint_failure = deepcopy(dict(projection))
        self._failure_counts, self._source_facts = dict(admitted["no_progress_seed"]), set(current.source_facts)
        self._save("checkpoint-history", {"origin": "checked_historical_assistant_not_new_tool_call",
            "provenance": current.provenance, "verifier_equivalence": admitted["provenance"],
            "historical_content_ref": self._checkpoint_content_ref, "failure_report_ref": current.known_failure["report_ref"],
            "no_progress_seed": self._failure_counts, "source_facts": sorted(self._source_facts),
            "current_candidate_created": False, "new_model_calls": 0, "new_builds": 0})

    def stop_check(self):
        return deepcopy(self._known_stop)

    def upstream_failure_context(self):
        """Read the current or checked historical failure and the actual upstream call."""
        self.inspector.validate_snapshot()
        calls = [row for row in self._actual_calls if row["name"] == "upstream"]
        if len(calls) != 1:
            raise ValueError("TASK_TOOL_ACTUAL_UPSTREAM_CALL_REQUIRED")
        actual = calls[0]
        if _read(actual["arguments_ref"]) != actual["arguments"]:
            raise ValueError("TASK_TOOL_UPSTREAM_CALL_CHANGED")
        response = _read(actual["response_ref"])
        if not any(row["call_id"] == actual["call_id"] and row["name"] == "upstream"
                   and row["arguments"] == actual["arguments"] for row in response["tool_calls"]):
            raise ValueError("TASK_TOOL_UPSTREAM_CALL_CHANGED")
        report, report_ref, failed_response_ref, previous = None, None, None, None
        failure_candidate_ref, failure_check_ref, failure_context_id = None, None, None
        origin = "no_checked_failure"
        if self.current_candidate is not None and self.current_check is not None:
            candidate = self._candidate(self.current_candidate)
            checked = _read(self.current_check)
            raw = _read(checked["report"]["report_ref"])
            if raw.get("status") in {"behavior_failed", "needs_evidence", "infrastructure_failed", "identity_failed"}:
                report, report_ref = raw, checked["report"]["report_ref"]
                failed_response_ref = candidate["model_origin"]["derived_content_ref"]
                previous = Path(failed_response_ref["path"]).read_text(encoding="utf-8")
                origin = "current_checked_candidate"
                failure_candidate_ref = deepcopy(checked["candidate_ref"])
                failure_check_ref = deepcopy(self.current_check)
                failure_context_id = checked["context_id"]
        if report is None and self.last_checked_failure is not None:
            history, _checked, candidate, report = self._failure_history()
            report_ref = history["report_ref"]
            failure_candidate_ref, failure_check_ref = history["candidate_ref"], history["check_ref"]
            failure_context_id = history["context_id"]
            failed_response_ref = deepcopy(candidate["model_origin"]["derived_content_ref"])
            if _ref(failed_response_ref["path"]) != failed_response_ref:
                raise ValueError("TASK_TOOL_HISTORICAL_FAILURE_CHANGED")
            previous = Path(failed_response_ref["path"]).read_text(encoding="utf-8")
            origin = "historical_checked_candidate_not_current_acceptance"
        if report is None and self._checkpoint is not None:
            report_ref = deepcopy(self._checkpoint.known_failure["report_ref"])
            report = _read(report_ref)
            failed_response_ref = deepcopy(self._checkpoint_content_ref)
            if _ref(failed_response_ref["path"]) != failed_response_ref:
                raise ValueError("TASK_TOOL_CHECKPOINT_CONTENT_CHANGED")
            previous = Path(failed_response_ref["path"]).read_text(encoding="utf-8")
            origin = "checked_historical_failure_not_new_candidate"
        return {"origin": origin, "selected_artifacts": self._selected_artifacts(),
            "implementation_plan": deepcopy(self.documents.implementation_plan),
            "failure_report": report, "failure_report_ref": report_ref, "failed_response_ref": failed_response_ref,
            "failure_candidate_ref": failure_candidate_ref, "failure_check_ref": failure_check_ref,
            "failure_context_id": failure_context_id, "failure_is_current_acceptance": False,
            "previous_response": previous, "generated_changes_sha256": report.get("generated_changes_sha256") if report else None,
            "source_snapshot_sha256": self.inspector.source_snapshot_sha256,
            "source_worker_response_ref": deepcopy(actual["response_ref"]),
            "actual_upstream_arguments": deepcopy(actual["arguments"]), "actual_upstream_call_id": actual["call_id"],
            "allowed_source_refs": ["binding://" + value for value in self.inspector.available_read_binding_ids],
            "approval_inherited": False}

    def _save(self, kind, value):
        path = self.output_root / (kind + "-" + uuid4().hex + ".json")
        with path.open("xb") as stream:
            stream.write(_bytes(value))
        return _ref(path)

    def _edit_path(self, relative):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative or Path(relative).is_absolute():
            raise ValueError("TASK_TOOL_EDIT_PATH_INVALID")
        selected = (self.workspace_root / relative).resolve(strict=True)
        if not selected.is_file() or not selected.is_relative_to(self.workspace_root):
            raise ValueError("TASK_TOOL_EDIT_PATH_INVALID")
        return selected

    def _allowed_source(self, path):
        selected = Path(path).resolve(strict=True)
        if not selected.is_file() or not any(selected.is_relative_to(root) for root in self.allowed_read_roots):
            raise ValueError("TASK_TOOL_SOURCE_OUTSIDE_AUTHORIZED_ROOTS")
        return selected

    def _admit_context(self, context):
        if not isinstance(context, TaskAgentContext) or context.feature_spec_ref != self.initial_feature_ref:
            raise ValueError("TASK_TOOL_REQUIREMENTS_CHANGED")
        refs = (context.feature_spec_ref, context.evidence_pack_ref, context.implementation_plan_ref)
        kinds = ("feature_spec", "evidence_pack", "implementation_plan")
        envelopes = []
        for ref, kind in zip(refs, kinds):
            if ref.artifact_type != kind or ref.feature_id != self.initial_feature_ref.feature_id:
                raise ValueError("TASK_TOOL_CONTEXT_IDENTITY_CHANGED")
            envelope = self.store.read(ref)
            if not self.store.check_exact_revision_freshness(ref).is_fresh:
                raise ValueError("TASK_TOOL_CONTEXT_STALE")
            if envelope["status"] not in ({"success", "blocked"} if kind == "evidence_pack" else {"success"}):
                raise ValueError("TASK_TOOL_CONTEXT_NOT_ACCEPTED")
            envelopes.append(envelope)
        for envelope, expected in ((envelopes[1], refs[:1]), (envelopes[2], refs[:2])):
            actual = envelope.get("inputs", {}).get("upstream_artifacts", [])
            # Scope: Store upstream entries use type/path/hash plus their own revision policy.
            # 为什么要有它：上游没有重复feature_id，不能错误拒绝正常Artifact；feature已逐项核准，依赖三身份仍须精确相同。
            if sorted((row["artifact_type"], row["path"], row["artifact_hash"]) for row in actual) != sorted(
                    (ref.artifact_type, ref.path, ref.artifact_hash) for ref in expected):
                raise ValueError("TASK_TOOL_CONTEXT_LINEAGE_CHANGED")
        documents = TaskContextDocuments(context, *(deepcopy(envelope["data"]) for envelope in envelopes))
        if envelopes[1]["status"] == "blocked":
            scope = compilation_candidate_scope(documents.evidence_pack, documents.feature_spec)
            receipt = getattr(self.evaluator, "configuration_receipt", None)
            if (scope is None or not callable(receipt)
                    or receipt().get("configuration_ref") != scope["worker_configuration_ref"]):
                raise ValueError("TASK_TOOL_COMPILATION_CANDIDATE_OWNER_MISMATCH")
        plan = documents.implementation_plan
        for binding in plan.get("binding_registry", {}).get("edit_bindings", []):
            relative = binding.get("canonical_target_path")
            if (relative not in self.allowed_edit_paths or binding.get("source_hash") != self.edit_baselines[relative]
                    or _ref(self._edit_path(relative))["sha256"] != self.edit_baselines[relative]):
                raise ValueError("TASK_TOOL_EDIT_AUTHORITY_CHANGED")
        inspector = PlanBoundSourceInspector(workspace_root=self.workspace_root,
            implementation_plan=plan, policy=self.policy, evidence_pack=documents.evidence_pack)
        # Scope: live registered source reads, not evaluator scripts, private tests or reference patches.
        # 为什么要有它：ArtifactStore正确绑定不自动授予新资料披露权限；上游新选内容仍须处于原读取范围。
        for item in documents.evidence_pack.get("items", []):
            selected = Path(item["source_path"])
            selected = selected if selected.is_absolute() else self.workspace_root / selected
            self._allowed_source(selected)
            if item.get("source_hash") is not None and _ref(selected)["sha256"] != item["source_hash"]:
                raise ValueError("TASK_TOOL_EVIDENCE_SOURCE_CHANGED")
        contract = self.prepare_contract(deepcopy(documents))
        if not isinstance(contract, PreparedNativePatchContract):
            raise ValueError("TASK_TOOL_PATCH_CONTRACT_REQUIRED")
        return documents, inspector, contract

    @property
    def context_id(self):
        return "context-" + _sha({"refs": self.context.to_dict(), "epoch": self.epoch})[:16]

    def qualification(self):
        missing = [name for name, owner in (("source_search", self.source_search_owner),
            ("upstream", self.upstream_owner), ("model_response_admission", self.model_response_gate)) if not callable(owner)]
        if self.evaluator_identity is None:
            missing.append("qualified_evaluator_identity")
        if self._check_inflight:
            missing.append("unresolved_candidate_execution")
        return {"ready": not missing, "missing_owners": missing,
                "scope": "registered_owner_interfaces_not_real_model_or_whole_task_acceptance"}

    def _read_budget_model_view(self):
        if self.read_budget_snapshot is None:
            return {}
        from src.mod_agent.task_agent_loop import read_budget_view

        value = deepcopy(self.read_budget_snapshot())
        limit = None if is_meter_only(self.budget_snapshot()) else self.policy.max_tool_rounds
        if (not isinstance(value, dict) or value != read_budget_view(used=value.get("used"), limit=value.get("limit"))
                or value["limit"] != limit):
            raise ValueError("TASK_TOOL_READ_BUDGET_SNAPSHOT_INVALID")
        return {"read_budget": value}

    def model_context(self):
        result = {"context_id": self.context_id, "allowed_read_binding_ids": list(self.inspector.available_read_binding_ids),
            "allowed_edit_binding_ids": list(self.inspector.available_binding_ids),
            "candidate_submission_schema": deepcopy(self.contract.response_schema),
            "source_read_semantics": "B0 is unchanged; checked candidate bytes are separate from source reads.",
            "context_id_scope": "Candidate version precondition only; it grants no read, edit or execution authority.",
            "why": "为什么要有它：context_id防止按旧Plan提交，不产生新的授权；技术计划可修订，真实用户要求和权限不随之改变。",
            **self._read_budget_model_view()}
        if self.candidate_mode == "current":
            result.update(candidate_mode="current", current_candidate_id=self.current_candidate,
                candidate_read_paths=sorted(self.allowed_edit_paths),
                source_read_semantics="source_read shows the current candidate overlay for changed files, otherwise the unchanged source. Plan bindings and final Apply preimages remain B0. candidate_read selects baseline or an explicit immutable candidate_id.",
                candidate_revision_semantics="First candidate_submit uses patch_json against B0. Subsequent submissions use parent_candidate_id, parent_code_sha256 and exact edits against current code; prior edits are retained.")
        return result

    def initial_request(self):
        # Scope: task orchestration instructions; the old Patch format applies only inside patch_json.
        # 为什么要有它：旧system要求直接返回Patch JSON，不能与B的连续工具选择冲突；原任务内容和候选格式仍逐字保留为局部合同。
        instructions = AgentMessage("system", "Work on the provided original task using the registered tools in one continuous session. "
            "Choose investigation, source reads, upstream reassessment, candidate submission or checking according to actual evidence. "
            "Technical assumptions may change; original user requirements, preservation obligations and host permissions do not. "
            "Source reads show unchanged B0. Submit a complete replacement against B0 using candidate_submit.patch_json and the current context_id. "
            "Candidate contract instructions below govern only patch_json, not whether you may use tools. "
            "In historical or local contract text, inspect_plan_binding_source maps to the current source_read tool, and request_patch_upstream maps to the current upstream tool. "
            "The context_id is a version precondition, never authorization. After upstream returns a new context, an old candidate cannot be delivered. "
            "Batch only source_read and source_search calls. Send upstream, candidate_submit, candidate_check and finish one call at a time. "
            "为什么要有它：只读观察可以一起获取；改变任务上下文、提交或验证候选和结束任务必须逐步返回结果，避免同批依赖过期状态。 "
            "Each source_read or source_search costs one read; when read_budget is supplied, use its latest remaining allowance. "
            "为什么要有它：工具调用数量受原任务额度约束，读取失败或调整计划都不会增加额度。 "
            "Use candidate_check for real owner observations. Only request finish when the latest current "
            "candidate_check has status=passed. After a failed check, use its counterexamples to inspect and revise "
            "the current code, or request upstream evidence/planning if needed. A repeated failing check does not "
            "restore completion eligibility. Do not hand repair back to the user merely because a check failed. "
            "为什么要有它：检查执行结束不等于代码通过，失败反馈应引向调查和修订，而不是反复重查后请求结束。 "
            "Text, a changed patch, compilation alone or a failed/stale report is not delivery. Do not modify verifier inputs, tests or permissions.")
        if self.candidate_mode == "current":
            instructions = AgentMessage("system", instructions.content.replace(
                "Source reads show unchanged B0. Submit a complete replacement against B0 using candidate_submit.patch_json and the current context_id. ",
                "source_read shows current candidate code for changed files, with explicit baseline and candidate identities. Use candidate_read with the explicit returned candidate_id to inspect current or historical code and differences. "
                "The first candidate_submit uses patch_json against B0. Then revise the current candidate using parent_candidate_id, "
                "parent_code_sha256 and edits with exact before_text/after_text inside the selected edit binding; previous changes remain. "
                "为什么要有它：修订针对当前代码，历史候选保持不可变，初始源码和最终Apply权限不变。 ").replace(
                "Batch only source_read and source_search calls.", "Batch only source_read, candidate_read and source_search calls.").replace(
                "Each source_read or source_search costs one read;", "Each source_read, candidate_read or source_search costs one read;"))
        if is_meter_only(self.budget_snapshot()):
            instructions = AgentMessage("system", instructions.content.replace(
                "when read_budget is supplied, use its latest remaining allowance. "
                "为什么要有它：工具调用数量受原任务额度约束，读取失败或调整计划都不会增加额度。 ",
                "resource counts are measurements only; read_budget.limit and remaining are null, meaning no experimental quota. "
                "Continue until the task is completed, a real safety/execution failure occurs, or repeated actions make no progress. "
                "为什么要有它：资源计数不应阻断修复；实际权限、版本校验、供应商容量和无进展处理仍然有效。 "))
        messages = [instructions]
        for message in self.contract.messages:
            messages.append(AgentMessage("user", {"candidate_submission_contract_instructions": message.content})
                            if message.role == "system" else AgentMessage(message.role, message.content))
        messages.append(AgentMessage("user", self.model_context()))
        history = ()
        if self._checkpoint is not None:
            checkpoint_request = self._checkpoint.request
            index = len(messages)
            history = (*checkpoint_request.historical_tool_exchanges,
                *((AgentToolExchange(checkpoint_request.tool_results, checkpoint_request.provider_state, index),)
                  if checkpoint_request.tool_results else ()))
            messages.extend(message for message in checkpoint_request.messages if message.role == "assistant"
                or message.role == "user" and isinstance(message.content, Mapping)
                and message.content.get("message_type") == "patch_behavior_correction")
        return AgentTurnRequest(tuple(messages), historical_tool_exchanges=history)

    def response_admission(self, response):
        if not callable(self.model_response_gate):
            raise ValueError("TASK_TOOL_MODEL_ADMISSION_OWNER_MISSING")
        self.model_response_gate(response)
        self._source_facts.update(source_fact_hashes([{**row, "admitted_to_model": True}
            for row in self._pending_source_observations]))
        self._pending_source_observations.clear()
        response_ref = self._save("model-response", asdict(response))
        self._actual_calls = []
        if response.stop_reason in {"length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal"}:
            return
        for call in response.tool_calls:
            args_ref = self._save("model-tool-arguments", call.arguments)
            self._actual_calls.append({"name": call.name, "call_id": call.call_id,
                "arguments": deepcopy(call.arguments), "arguments_ref": args_ref,
                "response_ref": response_ref, "model": response.model})

    def _context_precondition(self, arguments, *, required=False):
        supplied = arguments.get("context_id")
        if required and supplied is None or supplied is not None and supplied != self.context_id:
            raise ValueError("TASK_TOOL_CONTEXT_VERSION_CHANGED")
        self.inspector.validate_snapshot()

    def _prepare_read(self, arguments):
        if set(arguments) - {"binding_id", "context_id", "comment", "reason"}:
            raise PermissionError("TASK_TOOL_SOURCE_ARGUMENTS_OUTSIDE_CONTRACT")
        self._context_precondition(arguments)
        if any(key in arguments and type(arguments[key]) is not str for key in ("comment", "reason")):
            raise PermissionError("TASK_TOOL_SOURCE_ANNOTATION_INVALID")
        if "binding_id" not in arguments or type(arguments["binding_id"]) is not str or not arguments["binding_id"]:
            raise TaskActionInputError(("binding_id requires a nonempty string",))
        if arguments["binding_id"] not in self.inspector.available_read_binding_ids:
            raise PermissionError("TASK_TOOL_SOURCE_BINDING_NOT_AUTHORIZED")
        return PreparedTaskAction(arguments["binding_id"], 1, self._read_version())

    def _read_version(self):
        if self.candidate_mode == "current" and self.current_candidate is not None:
            self._candidate(self.current_candidate)
            return _sha({"baseline": self.inspector.read_source_snapshot_sha256, "candidate": self.current_candidate})
        return self.inspector.read_source_snapshot_sha256

    def _read_source(self, binding_id):
        observation = self.inspector.inspect(binding_id=binding_id, observation_id="task-read-" + uuid4().hex)
        payload = observation.model_view()
        view = self.current_code_view()
        if view:
            registry = self.documents.implementation_plan["binding_registry"]
            binding = next((row for row in (*registry["edit_bindings"], *registry.get("supporting_bindings", []))
                if row["binding_id"] == binding_id), None)
            path = binding.get("canonical_target_path", binding.get("source_path")) if binding else None
            changed = next((row for row in view["files"] if row["path"] == path), None)
            if changed and payload.get("excerpt_span"):
                before = self._edit_path(path).read_bytes().decode("utf-8").splitlines(keepends=True)
                after = changed["text"].splitlines(keepends=True)
                start, end = payload["excerpt_span"]["start_line"] - 1, payload["excerpt_span"]["end_line"]
                intervals = []
                for tag, a, b, c, d in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
                    if tag == "equal" and max(a, start) < min(b, end):
                        intervals.append((c + max(a, start) - a, c + min(b, end) - a))
                    elif tag != "equal" and (max(a, start) < min(b, end) or a == b and start <= a < end):
                        intervals.append((c, d))
                first = min((a for a, _ in intervals), default=0)
                last = max((b for _, b in intervals), default=first)
                payload.update(baseline_source_hash=payload["source_hash"], baseline_excerpt_span=payload["excerpt_span"],
                    target_span_scope="initial_source", source_hash=changed["source_hash"],
                    source_snapshot_sha256=self._read_version(), candidate_id=view["candidate_id"],
                    candidate_ref=view["candidate_ref"], code_sha256=view["materialized_code_identity"]["code_sha256"],
                    source_origin="current_candidate_overlay", excerpt="".join(after[first:last]),
                    excerpt_span={"start_line": first + 1, "end_line": last}, truncated=first > 0 or last < len(after))
                payload.pop("source_identity_sha256", None)
                payload.pop("observation_sha256", None)
                payload["observation_sha256"] = _sha(payload)
        self._pending_source_observations.append(deepcopy(payload))
        self._save("source-read", {"context": self.context.to_dict(), "read_cost": 1, "observation": payload})
        return {**payload, **self._read_budget_model_view()}

    def _candidate_version(self, selected):
        if selected == "current":
            raise ValueError("TASK_TOOL_EXPLICIT_CANDIDATE_ID_REQUIRED")
        if selected != "baseline":
            self._load_candidate(selected)
        return selected

    def _prepare_candidate_read(self, arguments):
        if (self.candidate_mode != "current" or set(arguments) - {
                "context_id", "version", "path", "start_line", "end_line", "compare_to"}
                or any(type(arguments.get(key)) is not str for key in ("version", "path"))):
            raise ValueError("TASK_TOOL_CANDIDATE_READ_ARGUMENTS_INVALID")
        self._context_precondition(arguments, required=True)
        if arguments["path"] not in self.allowed_edit_paths:
            raise PermissionError("TASK_TOOL_CANDIDATE_READ_PATH_NOT_AUTHORIZED")
        self._allowed_source(self._edit_path(arguments["path"]))
        for key in ("start_line", "end_line"):
            if key in arguments and (type(arguments[key]) is not int or arguments[key] < 1):
                raise TaskActionInputError((key + " requires a positive integer",))
        if "end_line" in arguments and arguments["end_line"] < arguments.get("start_line", 1):
            raise TaskActionInputError(("end_line must not precede start_line",))
        selected = {**deepcopy(arguments), "version": self._candidate_version(arguments["version"])}
        if "compare_to" in arguments:
            if type(arguments["compare_to"]) is not str:
                raise ValueError("TASK_TOOL_CANDIDATE_READ_ARGUMENTS_INVALID")
            selected["compare_to"] = self._candidate_version(arguments["compare_to"])
        raw, _code, _origin = self._version_file(selected["version"], selected["path"])
        line_count = len(raw.decode("utf-8").splitlines(keepends=True))
        if selected.get("start_line", 1) > max(1, line_count):
            raise TaskActionInputError((f"start_line exceeds file length {line_count}",))
        if selected.get("end_line", line_count) > line_count:
            selected["requested_end_line"] = selected["end_line"]
            selected["end_line"] = line_count
        return PreparedTaskAction(selected, 1, _sha(selected))

    def _version_file(self, version, path):
        baseline = self._edit_path(path).read_bytes()
        if hashlib.sha256(baseline).hexdigest() != self.edit_baselines[path]:
            raise ValueError("TASK_TOOL_EDIT_AUTHORITY_CHANGED")
        if version == "baseline":
            return baseline, None, "initial_source"
        candidate = self._load_candidate(version)
        _context, documents, _inspector, _contract = self._candidate_contexts[candidate["context_id"]]
        materialized = materialize_candidate(workspace_root=self.workspace_root,
            implementation_plan=documents.implementation_plan, changes=candidate["generated_changes"])
        return (materialized.files.get(path, baseline), candidate["materialized_code_identity"]["code_sha256"],
            "candidate_snapshot" if path in materialized.files else "unchanged_initial_source")

    def _read_candidate(self, selected):
        self._context_precondition(selected, required=True)
        raw, code_sha, origin = self._version_file(selected["version"], selected["path"])
        text = raw.decode("utf-8")
        lines = text.splitlines(keepends=True)
        start, end = selected.get("start_line", 1), selected.get("end_line", len(lines))
        if start > max(1, len(lines)) or end > len(lines):
            raise ValueError("TASK_TOOL_CANDIDATE_READ_RANGE_OUTSIDE_FILE")
        result = {"context_id": self.context_id, "version": selected["version"], "path": selected["path"],
            "file_sha256": hashlib.sha256(raw).hexdigest(), "code_sha256": code_sha, "source_origin": origin,
            "start_line": start, "end_line": end, "total_lines": len(lines), "total_utf8_bytes": len(raw),
            "text": "".join(lines[start - 1:end])}
        if "requested_end_line" in selected:
            result.update(requested_end_line=selected["requested_end_line"], end_clamped_to_eof=True)
        if "compare_to" in selected:
            previous, previous_code, previous_origin = self._version_file(selected["compare_to"], selected["path"])
            result["comparison"] = {"from_version": selected["compare_to"], "to_version": selected["version"],
                "from_file_sha256": hashlib.sha256(previous).hexdigest(), "from_code_sha256": previous_code,
                "from_source_origin": previous_origin, "scope": "complete_selected_file",
                "unified_diff": "".join(difflib.unified_diff(previous.decode("utf-8").splitlines(keepends=True),
                    lines, fromfile=selected["compare_to"] + "/" + selected["path"],
                    tofile=selected["version"] + "/" + selected["path"]))}
        self._save("candidate-read", {"context": self.context.to_dict(), "read_cost": 1, "observation": result})
        return {**result, **self._read_budget_model_view()}

    def _prepare_search(self, arguments):
        if set(arguments) - {"query", "context_id"} or type(arguments.get("query")) is not str or not arguments["query"].strip():
            raise ValueError("TASK_TOOL_SEARCH_ARGUMENTS_INVALID")
        self._context_precondition(arguments)
        return PreparedTaskAction(arguments["query"], 1, _sha({"context": self.context_id, "source": self._read_version()}))

    def _search(self, query):
        result = self.source_search_owner(query=query, context=deepcopy(self.documents))
        if not isinstance(result, SourceSearchResult) or result.query != query:
            raise ValueError("TASK_TOOL_SEARCH_OWNER_RESULT_INVALID")
        saved = _read(result.sidecar_ref)
        expected = {"query": result.query, "payload": result.payload, "source_refs": list(result.source_refs)}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("TASK_TOOL_SEARCH_RESULT_BINDING_CHANGED")
        view = self.current_code_view()
        proof = result.payload.get("current_code_view")
        overlay = {row["path"]: row for row in view["files"]} if view else {}
        if proof is not None and (view is None or proof != {"candidate_ref": view["candidate_ref"],
                "materialized_code_identity": view["materialized_code_identity"]}):
            raise ValueError("TASK_TOOL_SEARCH_CANDIDATE_CHANGED")
        overlay_sources = {str(Path(item["source_path"]).resolve()): item.get("logical_source_path")
            for item in result.payload.get("items", []) if item.get("logical_source_path") in overlay and proof is not None}
        for reference in result.source_refs:
            if reference["path"] in overlay_sources:
                selected = overlay[overlay_sources[reference["path"]]]
                if (not Path(reference["path"]).resolve(strict=True).is_relative_to(self.store.root)
                        or reference["sha256"] != selected["source_hash"]
                        or Path(reference["path"]).read_bytes() != selected["text"].encode("utf-8")):
                    raise ValueError("TASK_TOOL_SEARCH_CANDIDATE_CHANGED")
            else:
                self._allowed_source(reference["path"])
            if _ref(reference["path"]) != reference:
                raise ValueError("TASK_TOOL_SEARCH_SOURCE_CHANGED")
        sources = {row["path"]: row["sha256"] for row in result.source_refs}
        for item in result.payload.get("items", []):
            selected = Path(item["source_path"])
            selected = selected if selected.is_absolute() else self.workspace_root / selected
            selected = selected.resolve(strict=True) if str(selected.resolve()) in overlay_sources else self._allowed_source(selected)
            if sources.get(str(selected)) != item.get("source_hash"):
                raise ValueError("TASK_TOOL_SEARCH_ITEM_NOT_SOURCE_BOUND")
        self.inspector.validate_snapshot()
        items = deepcopy(result.payload.get("items", []))
        for item in items:
            logical_path = item.get("logical_source_path", item.get("source_path"))
            evidence_id = item.get("evidence_id")
            if (type(evidence_id) is not str or not evidence_id
                    or type(logical_path) is not str or logical_path not in self.allowed_edit_paths
                    or item.get("source_type") != "target_source"):
                continue
            reference = "search://" + evidence_id
            item["reassessment_ref"] = reference
            self._search_reassessment_observations[reference] = {
                "context_id": self.context_id,
                "source_result_sha256": result.sidecar_ref["sha256"],
                "item": deepcopy(item),
            }
        self._pending_source_observations.extend(deepcopy(items))
        self._save("source-search", {"context": self.context.to_dict(), "owner_result_ref": result.sidecar_ref, "read_cost": 1})
        model_items = [_model_search_item(item) for item in items]
        return {"query": query, "items": model_items,
                "context_id": self.context_id, "plan_updated": False,
                "source_view": "current_candidate_overlay" if proof else "initial_source",
                "current_code_view": deepcopy(proof),
                "source_result_sha256": result.sidecar_ref["sha256"],
                "model_projection": {"schema_version": "task_agent_search_projection_v1",
                    "full_observation_count": len(items), "full_result_retained": True},
                **self._read_budget_model_view()}

    def _prepare_submit(self, arguments):
        if self._upstream_blocked:
            raise ValueError("TASK_TOOL_UPSTREAM_REASSESSMENT_UNRESOLVED")
        if "context_id" in arguments and arguments["context_id"] != self.context_id:
            raise ValueError("TASK_TOOL_CONTEXT_VERSION_CHANGED")
        self._context_precondition(arguments)
        matches = [row for row in self._actual_calls if row["name"] == "candidate_submit" and row["arguments"] == arguments]
        if len(matches) != 1 or _read(matches[0]["arguments_ref"]) != arguments:
            raise ValueError("TASK_TOOL_ACTUAL_MODEL_CALL_REQUIRED")
        _read(matches[0]["response_ref"])
        parent = self._candidate(self.current_candidate) if self.candidate_mode == "current" and self.current_candidate else None
        # Scope: supplied version and binding identities retain hard rejection before format feedback.
        # 为什么要有它：空的多余字段可纠正，但失效父版本或越权目标不能借另一处格式错误降级为重试。
        for field, expected in (("parent_candidate_id", parent["candidate_id"] if parent else None),
                ("parent_code_sha256", parent["materialized_code_identity"]["code_sha256"] if parent else None)):
            if field in arguments and arguments[field] != expected:
                raise ValueError("TASK_CURRENT_PARENT_VERSION_CHANGED")
        if _sha(self.contract.response_schema) != self.contract.schema_sha256:
            raise ValueError("TASK_TOOL_CANDIDATE_CONTRACT_CHANGED")
        raw_patch = None
        if type(arguments.get("patch_json")) is str:
            try:
                raw_patch = json.loads(arguments["patch_json"])
            except (ValueError, TypeError):
                pass
        rows = []
        for selected in (raw_patch.get("changes") if isinstance(raw_patch, Mapping) else None, arguments.get("edits")):
            if isinstance(selected, list):
                rows.extend(selected)
        allowed = set(self.contract.materialization_context["allowed_bindings"])
        for row in rows:
            if isinstance(row, Mapping) and type(row.get("edit_binding_id")) is str and row["edit_binding_id"]:
                if row["edit_binding_id"] not in allowed:
                    raise PermissionError("TASK_CURRENT_EDIT_BINDING_NOT_AUTHORIZED")
        forms = _candidate_submit_forms()
        errors = []
        outer = _submission_format_error(json.dumps(arguments), forms[1 if parent else 0], field="candidate_submit")
        if outer:
            errors.append(outer)
        if type(arguments.get("patch_json")) is str:
            inner = _submission_format_error(arguments["patch_json"], self.contract.response_schema, field="patch_json")
            if inner:
                item_fields = self.contract.response_schema["properties"]["changes"]["items"]["properties"]
                errors.append(inner + f" Each changes item allows: {', '.join(item_fields)}.")
        if errors:
            raise TaskCandidateRevisionInputError(errors)
        if parent is not None:
            derive_current_revision(parent, arguments=arguments, prepared_contract=self.contract)
        return PreparedTaskAction(deepcopy(matches[0]), 0,
            _sha({"context_id": self.context_id, "current_candidate": self.current_candidate})
            if self.candidate_mode == "current" else self.context_id)

    def _submit(self, actual):
        self.inspector.validate_snapshot()
        arguments = actual["arguments"]
        parent = self._candidate(self.current_candidate) if self.candidate_mode == "current" and self.current_candidate else None
        content = derive_current_revision(parent, arguments=arguments, prepared_contract=self.contract) if parent else arguments["patch_json"]
        content_bytes = content.encode("utf-8")
        output = self.output_root / ("candidate-tool-content-" + uuid4().hex + ".json")
        with output.open("xb") as stream:
            stream.write(content_bytes)
        output_ref = _ref(output)
        origin = {"kind": "derived_from_actual_tool_call_arguments", "response_ref": actual["response_ref"],
                  "call_id": actual["call_id"], "arguments_ref": actual["arguments_ref"], "field": "patch_json",
                  "derived_content_ref": output_ref}
        if parent is not None:
            origin = {"kind": "derived_from_current_candidate_revision", "response_ref": actual["response_ref"],
                "call_id": actual["call_id"], "arguments_ref": actual["arguments_ref"], "derived_content_ref": output_ref,
                "parent_candidate_ref": deepcopy(self.candidates[self.current_candidate]["ref"]),
                "derivation": "binding_exact_revision_v1"}
        projection = candidate_projection_contract(self.contract, content) if parent else self.contract
        generated = self.generator.materialize_prepared_native(projection,
            ChatResponse(content=content, model=actual["model"]))
        materialized = materialize_candidate(workspace_root=self.workspace_root,
            implementation_plan=self.documents.implementation_plan, changes=generated.changes)
        code = materialized_code_identity(materialized)
        if any(path not in self.allowed_edit_paths for path in code["files"]):
            raise ValueError("TASK_TOOL_CANDIDATE_OUTSIDE_EDIT_AUTHORITY")
        self.inspector.validate_snapshot()
        candidate = {"context_id": self.context_id, "context": self.context.to_dict(),
            "generated_changes": list(generated.changes), "generated_changes_sha256": _sha(generated.changes),
            "materialized_code_identity": code, "model_origin": origin,
            "model_response": {"path": output_ref["path"], "output_sha256": output_ref["sha256"],
                               "output_utf8_bytes": len(content_bytes), "model": actual["model"], "origin": origin["kind"]},
            "source_snapshot_sha256": self.inspector.source_snapshot_sha256}
        candidate_id = "candidate-" + _sha(candidate)[:20]
        candidate["candidate_id"] = candidate_id
        reference = self._save("candidate", candidate)
        self.candidates[candidate_id] = {"data": candidate, "ref": reference}
        self.current_candidate, self.current_check = candidate_id, None
        result = {"candidate_id": candidate_id, "context_id": self.context_id,
                "files": code["files"], "validation_status": "not_checked", "source_apply_performed": False}
        if self.candidate_mode == "current":
            result.update(code_sha256=code["code_sha256"],
                parent_candidate_id=parent["candidate_id"] if parent else None)
        return result

    def _candidate(self, candidate_id):
        if self._check_inflight:
            raise ValueError("TASK_TOOL_CANDIDATE_EXECUTION_UNRESOLVED")
        if candidate_id != self.current_candidate or candidate_id not in self.candidates:
            raise ValueError("TASK_TOOL_CANDIDATE_NOT_CURRENT")
        return self._load_candidate(candidate_id)

    def _load_candidate(self, candidate_id, seen=None):
        seen = set() if seen is None else set(seen)
        if candidate_id in seen or candidate_id not in self.candidates:
            raise ValueError("TASK_TOOL_CANDIDATE_LINEAGE_INVALID")
        seen.add(candidate_id)
        row = self.candidates[candidate_id]
        candidate = _read(row["ref"])
        if candidate != row["data"] or candidate["context_id"] not in self._candidate_contexts:
            raise ValueError("TASK_TOOL_CANDIDATE_BINDING_CHANGED")
        context, documents, inspector, contract = self._candidate_contexts[candidate["context_id"]]
        if (candidate["context"] != context.to_dict()
                or self.candidate_mode != "current" and candidate["context_id"] != self.context_id):
            raise ValueError("TASK_TOOL_CANDIDATE_BINDING_CHANGED")
        for reference in (context.feature_spec_ref, context.evidence_pack_ref, context.implementation_plan_ref):
            self.store.read(reference)
        inspector.validate_snapshot()
        origin = candidate["model_origin"]
        if origin["kind"] == "derived_from_context_transition":
            if self.candidate_mode != "current" or origin.get("derivation") != "exact_code_context_transfer_v1":
                raise ValueError("TASK_TOOL_CANDIDATE_MODE_CHANGED")
            parent_record = _read(origin["parent_candidate_ref"])
            parent = self._load_candidate(parent_record["candidate_id"], seen)
            assessment = _read(origin["assessment_ref"])
            if (parent != parent_record or self.candidates[parent["candidate_id"]]["ref"] != origin["parent_candidate_ref"]
                    or assessment.get("input_context") != parent["context"]
                    or assessment.get("output_context") != candidate["context"]):
                raise ValueError("TASK_TOOL_CANDIDATE_PARENT_CHANGED")
            old_documents = self._candidate_contexts[parent["context_id"]][1]
            old = materialize_candidate(workspace_root=self.workspace_root,
                implementation_plan=old_documents.implementation_plan, changes=parent["generated_changes"])
            content, projection = context_transfer_content(baseline_files=old.baseline_files, current_files=old.files,
                implementation_plan=documents.implementation_plan, prepared_contract=contract)
            if _ref(origin["derived_content_ref"]["path"]) != origin["derived_content_ref"] or Path(origin["derived_content_ref"]["path"]).read_bytes() != content.encode("utf-8"):
                raise ValueError("TASK_TOOL_MODEL_OUTPUT_BINDING_CHANGED")
            generated = self.generator.materialize_prepared_native(projection,
                ChatResponse(content=content, model=parent["model_response"]["model"]))
            materialized = materialize_candidate(workspace_root=self.workspace_root,
                implementation_plan=documents.implementation_plan, changes=generated.changes)
            expected_response = {"path": origin["derived_content_ref"]["path"], "output_sha256": origin["derived_content_ref"]["sha256"],
                "output_utf8_bytes": len(content.encode("utf-8")), "model": parent["model_response"]["model"], "origin": origin["kind"]}
            if (list(generated.changes) != candidate["generated_changes"] or _sha(generated.changes) != candidate["generated_changes_sha256"]
                    or materialized_code_identity(materialized) != parent["materialized_code_identity"]
                    or candidate["materialized_code_identity"] != parent["materialized_code_identity"]
                    or candidate["model_response"] != expected_response):
                raise ValueError("TASK_TOOL_CANDIDATE_CONTENT_CHANGED")
            return candidate
        arguments = _read(origin["arguments_ref"])
        response = _read(origin["response_ref"])
        content_bytes = Path(origin["derived_content_ref"]["path"]).read_bytes()
        if origin["kind"] == "derived_from_current_candidate_revision":
            if self.candidate_mode != "current" or origin.get("derivation") != "binding_exact_revision_v1":
                raise ValueError("TASK_TOOL_CANDIDATE_MODE_CHANGED")
            parent_record = _read(origin["parent_candidate_ref"])
            parent = self._load_candidate(parent_record["candidate_id"], seen)
            if parent != parent_record or self.candidates[parent["candidate_id"]]["ref"] != origin["parent_candidate_ref"]:
                raise ValueError("TASK_TOOL_CANDIDATE_PARENT_CHANGED")
            expected_content = derive_current_revision(parent, arguments=arguments, prepared_contract=contract).encode("utf-8")
        elif origin["kind"] == "derived_from_actual_tool_call_arguments" and origin.get("field") == "patch_json":
            expected_content = arguments.get("patch_json", "").encode("utf-8")
        else:
            raise ValueError("TASK_TOOL_MODEL_ORIGIN_INVALID")
        # Scope: compiler metadata describes the actual selected patch_json bytes, not provider content or character count.
        # 为什么要有它：真实编译验收需要长度和哈希共同绑定模型候选；派生工具内容仍保留原响应与call_id来源。
        expected_response = {"path": origin["derived_content_ref"]["path"], "output_sha256": origin["derived_content_ref"]["sha256"],
            "output_utf8_bytes": len(content_bytes), "model": response["model"], "origin": origin["kind"]}
        if (arguments.get("context_id") != candidate["context_id"] or expected_content != content_bytes
                or _ref(origin["derived_content_ref"]["path"]) != origin["derived_content_ref"]
                or candidate.get("model_response") != expected_response
                or not any(call["call_id"] == origin["call_id"] and call["name"] == "candidate_submit"
                           and call["arguments"] == arguments for call in response["tool_calls"])):
            raise ValueError("TASK_TOOL_MODEL_OUTPUT_BINDING_CHANGED")
        if self.candidate_mode == "current":
            projection = candidate_projection_contract(contract, content_bytes.decode("utf-8")) if origin["kind"] == "derived_from_current_candidate_revision" else contract
            generated = self.generator.materialize_prepared_native(projection,
                ChatResponse(content=content_bytes.decode("utf-8"), model=response["model"]))
            materialized = materialize_candidate(workspace_root=self.workspace_root,
                implementation_plan=documents.implementation_plan, changes=generated.changes)
            if (list(generated.changes) != candidate["generated_changes"] or _sha(generated.changes) != candidate["generated_changes_sha256"]
                    or materialized_code_identity(materialized) != candidate["materialized_code_identity"]):
                raise ValueError("TASK_TOOL_CANDIDATE_CONTENT_CHANGED")
        self.inspector.validate_snapshot()
        return candidate

    def _prepare_candidate_action(self, arguments):
        if set(arguments) != {"candidate_id"} or type(arguments["candidate_id"]) is not str:
            raise ValueError("TASK_TOOL_CANDIDATE_SELECTION_INVALID")
        self._candidate(arguments["candidate_id"])
        version = _sha({"candidate": arguments["candidate_id"], "context_id": self.context_id,
                        "check": self.current_check})
        return PreparedTaskAction(arguments["candidate_id"], 0, version)

    def _check(self, candidate_id):
        candidate = self._candidate(candidate_id)
        identity = verification_identity(self.evaluator)
        if identity is None or identity != self.evaluator_identity:
            raise ValueError("TASK_TOOL_VERIFICATION_IDENTITY_CHANGED")
        self.check_count += 1
        self.current_check = None
        self._check_inflight = True
        context = {key: deepcopy(candidate[key]) for key in ("generated_changes", "generated_changes_sha256",
            "model_response", "source_snapshot_sha256", "materialized_code_identity")}
        context.update(evaluation_index=self.check_count, implementation_plan=deepcopy(self.documents.implementation_plan),
            implementation_plan_ref=asdict(self.context.implementation_plan_ref), verification_identity=identity)
        with trace_operation("task.candidate.check", attributes={"candidate_id": candidate_id}):
            try:
                report = invoke_candidate_evaluator(self.evaluator, context)
            except CandidateExecutionError as error:
                self._save("candidate-check-rejected", {"candidate_id": candidate_id,
                    "code": error.code, "report": error.rejected_report, "dispatch_started": error.dispatch_started})
                raise
            finally:
                self.inspector.validate_snapshot()
            if verification_identity(self.evaluator) != identity:
                self._save("candidate-check-rejected", {"candidate_id": candidate_id,
                    "code": "TASK_TOOL_VERIFICATION_IDENTITY_CHANGED", "report": report})
                raise ValueError("TASK_TOOL_VERIFICATION_IDENTITY_CHANGED")
            try:
                admitted, _ = validate_candidate_report(report, changes_sha256=candidate["generated_changes_sha256"],
                    source_snapshot_sha256=self.inspector.source_snapshot_sha256,
                    allowed_binding_ids=self.inspector.available_binding_ids)
                expected = {"materialized_code_identity": candidate["materialized_code_identity"],
                    "implementation_plan_ref": asdict(self.context.implementation_plan_ref),
                    "model_response_sha256": candidate["model_response"]["output_sha256"],
                    "verification_identity": identity}
                if any((key in admitted or admitted["status"] == "passed") and admitted.get(key) != value
                       for key, value in expected.items()):
                    raise CandidateExecutionError("TASK_TOOL_CANDIDATE_REPORT_BINDING_CHANGED",
                        rejected_report={**admitted, "admission": "rejected"})
            except CandidateExecutionError as error:
                self._save("candidate-check-rejected", {"candidate_id": candidate_id, "code": error.code,
                    "report": error.rejected_report})
                raise
        record = {"candidate_id": candidate_id, "context_id": self.context_id, "candidate_ref": self.candidates[candidate_id]["ref"],
            "verification_identity": identity, "report": admitted}
        self.current_check = self._save("candidate-check", record)
        if admitted["status"] in {"behavior_failed", "needs_evidence", "infrastructure_failed", "identity_failed"}:
            self.last_checked_failure = {"check_ref": deepcopy(self.current_check),
                "candidate_id": candidate_id, "candidate_ref": deepcopy(self.candidates[candidate_id]["ref"]),
                "context_id": self.context_id, "materialized_code_identity": deepcopy(candidate["materialized_code_identity"]),
                "report_ref": deepcopy(admitted["report_ref"])}
        self._check_inflight = False
        if admitted["status"] in {"behavior_failed", "needs_evidence"}:
            failure = {**admitted, "materialized_code_identity": candidate["materialized_code_identity"],
                       "verification_identity": identity}
            signature = failure_signature(failure, self._source_facts)
            self._failure_counts[signature] = self._failure_counts.get(signature, 0) + 1
            if self._failure_counts[signature] >= 3:
                self._known_stop = {"code": "PATCH_BEHAVIOR_NO_PROGRESS", "same_state_observations": self._failure_counts[signature],
                    "failure_signature_sha256": signature, "report_ref": deepcopy(admitted["report_ref"])}
        observations = deepcopy(admitted["observations"])
        for item in observations:
            if isinstance(item, dict) and isinstance(item.get("result_ref"), dict):
                item["result_ref"] = {"sha256": item["result_ref"].get("sha256")}
        # Scope: model display only; the bound check and failure signature retain the full original report.
        # 为什么要有它：连续修订复用原工作流的无损观察去重，重复状态不应拖慢反馈，所有反例仍可直接从同一返回值还原。
        return project_observation_values({"candidate_id": candidate_id, "status": admitted["status"], "failure_owner": admitted["failure_owner"],
            "observations": observations, "execution_origin": admitted["execution_origin"],
            "diagnosis_is_not_authority": True, "verification_scope": admitted.get("verification_scope", "candidate_only"),
            "whole_task_delivered": False})

    def completion_check(self, *, prepared, output):
        if self._upstream_blocked:
            return {"accepted": False, "reason": "upstream_reassessment_unresolved", "whole_task_delivered": False}
        candidate = self._candidate(prepared.value)
        if output != {"candidate_id": prepared.value} or self.current_check is None:
            return {"accepted": False, "reason": "current_candidate_has_no_checked_report"}
        current = self._prepare_candidate_action({"candidate_id": prepared.value})
        if prepared.version_identity != current.version_identity:
            raise ValueError("TASK_TOOL_FINISH_STATE_CHANGED")
        checked = _read(self.current_check)
        if (checked["candidate_id"] != prepared.value or checked["context_id"] != self.context_id
                or checked["candidate_ref"] != self.candidates[prepared.value]["ref"]
                or verification_identity(self.evaluator) != checked["verification_identity"]):
            raise ValueError("TASK_TOOL_CHECKED_REPORT_NOT_CURRENT")
        raw = _read(checked["report"]["report_ref"])
        report, _ = validate_candidate_report({**raw, "report_ref": checked["report"]["report_ref"]},
            changes_sha256=candidate["generated_changes_sha256"], source_snapshot_sha256=self.inspector.source_snapshot_sha256,
            allowed_binding_ids=self.inspector.available_binding_ids)
        accepted = (report["status"] == "passed" and report["execution_origin"] == "executed"
            and raw.get("materialized_code_identity") == candidate["materialized_code_identity"]
            and raw.get("implementation_plan_ref") == asdict(self.context.implementation_plan_ref)
            and raw.get("model_response_sha256") == candidate["model_response"]["output_sha256"]
            and raw.get("verification_identity") == checked["verification_identity"])
        result = {"accepted": accepted, "candidate_id": prepared.value,
            "checked_report_ref": deepcopy(self.current_check), "scope": "qualified_candidate_validation_only",
            "whole_task_delivered": False}
        if not accepted:
            result.update(check_status=report["status"],
                reason="current_candidate_check_failed" if report["status"] != "passed" else "current_report_not_eligible",
                next_step="Use the failed observations to inspect current code and revise it, or request upstream "
                    "evidence/planning for a concrete gap. Rechecking unchanged code with the same failure does not "
                    "qualify for finish; a current status=passed check and unchanged identities are required.")
        return result

    def _prepare_upstream(self, arguments):
        if (set(arguments) != {"context_id", "request"} or not isinstance(arguments["request"], dict)
                or not arguments["request"]):
            raise ValueError("TASK_TOOL_UPSTREAM_ARGUMENTS_INVALID")
        self._context_precondition(arguments, required=True)
        request = arguments["request"]
        allowed = {"binding://" + value for value in self.inspector.available_read_binding_ids}
        allowed.update(reference for reference, observation in self._search_reassessment_observations.items()
                       if observation["context_id"] == self.context_id)
        search_refs = [value for value in request.get("source_refs", [])
                       if isinstance(value, str) and value.startswith("search://")]
        if (set(request) != {"required_owner", "reason", "source_refs"}
                or request["required_owner"] not in {"evidence", "planning"}
                or search_refs and request["required_owner"] != "evidence"
                or type(request["reason"]) is not str or not 0 < len(request["reason"].strip()) <= 2000
                or not isinstance(request["source_refs"], list) or not request["source_refs"]
                or any(type(value) is not str or value not in allowed for value in request["source_refs"])
                or len(set(request["source_refs"])) != len(request["source_refs"])):
            raise ValueError("TASK_TOOL_UPSTREAM_REQUEST_NOT_AUTHORIZED")
        return PreparedTaskAction(deepcopy(arguments["request"]), 0, self.context_id)

    def _upstream(self, request):
        before = validate_budget(self.budget_snapshot())
        observations = [deepcopy(self._search_reassessment_observations[reference]["item"])
                        for reference in request["source_refs"] if reference.startswith("search://")]
        result = self.upstream_owner(request=deepcopy(request), context=deepcopy(self.documents),
            **({"source_observations": observations} if observations else {}))
        after = validate_budget(self.budget_snapshot())
        if not isinstance(result, TaskUpstreamAssessment):
            raise ValueError("TASK_TOOL_UPSTREAM_CONTEXT_REQUIRED")
        assessment = _read(result.assessment_ref)
        if (assessment.get("input_context") != self.context.to_dict()
                or assessment.get("output_context") != result.context.to_dict()
                or assessment.get("request_sha256") != _sha(request)
                or assessment.get("budget_before") != before or assessment.get("budget_after") != after
                or after["maximum"] != before["maximum"] or after["consumed"] < before["consumed"]
                or after.get("policy", "stage_and_total") != before.get("policy", "stage_and_total")
                or not isinstance(assessment.get("owner_result_refs"), list) or not assessment["owner_result_refs"]):
            raise ValueError("TASK_TOOL_UPSTREAM_ASSESSMENT_NOT_BOUND")
        for reference in assessment["owner_result_refs"]:
            _read(reference)
        outcome = assessment.get("outcome", "retained_context" if result.context == self.context else "replanned")
        feedback = assessment.get("model_feedback", {})
        if (outcome not in {"retained_context", "replanned", "blocked"} or not isinstance(feedback, Mapping)
                or outcome == "blocked" and result.context != self.context):
            raise ValueError("TASK_TOOL_UPSTREAM_OUTCOME_INVALID")
        documents, inspector, contract = self._admit_context(result.context)
        if self.candidate_mode == "current":
            validate_current_candidate_contract(documents.implementation_plan, contract)
        if verification_identity(self.evaluator) != self.evaluator_identity:
            raise ValueError("TASK_TOOL_UPSTREAM_VERIFIER_CHANGED")
        previous = self.context
        transfer = None
        if self.candidate_mode == "current" and self.current_candidate is not None and result.context != previous:
            # Validate before committing session state. Owners preflight before
            # publishing too. 为什么要有它：失败迁移不能清空正在修复的代码或切到不可用的上下文。
            transfer = self._transfer_materialization(documents, contract)
        self.context, self.documents, self.inspector, self.contract = result.context, documents, inspector, contract
        if self.candidate_mode != "current" or result.context != previous:
            self.epoch += 1
        self._candidate_contexts[self.context_id] = (self.context, documents, inspector, contract)
        self._upstream_blocked = outcome == "blocked"
        self.current_check = None
        if self.candidate_mode != "current":
            self.current_candidate = None
        if transfer is not None:
            parent, content, generated, materialized = transfer
            output = self.output_root / ("context-transfer-content-" + uuid4().hex + ".json")
            with output.open("xb") as stream:
                stream.write(content.encode("utf-8"))
            output_ref = _ref(output)
            origin = {"kind": "derived_from_context_transition", "derivation": "exact_code_context_transfer_v1",
                "parent_candidate_ref": deepcopy(self.candidates[parent["candidate_id"]]["ref"]),
                "assessment_ref": deepcopy(result.assessment_ref), "derived_content_ref": output_ref}
            candidate = {"context_id": self.context_id, "context": self.context.to_dict(),
                "generated_changes": list(generated.changes), "generated_changes_sha256": _sha(generated.changes),
                "materialized_code_identity": materialized_code_identity(materialized), "model_origin": origin,
                "model_response": {"path": output_ref["path"], "output_sha256": output_ref["sha256"],
                    "output_utf8_bytes": len(content.encode("utf-8")), "model": parent["model_response"]["model"], "origin": origin["kind"]},
                "source_snapshot_sha256": inspector.source_snapshot_sha256}
            candidate["candidate_id"] = "candidate-" + _sha(candidate)[:20]
            reference = self._save("candidate-context-transfer", candidate)
            self.candidates[candidate["candidate_id"]] = {"data": candidate, "ref": reference}
            self.current_candidate = candidate["candidate_id"]
        self._save("upstream-context", {"previous": previous.to_dict(), "current": self.context.to_dict(),
            "assessment_ref": result.assessment_ref, "context_id": self.context_id, "old_candidates_deliverable": False,
            "current_candidate_ref": deepcopy(self.candidates[self.current_candidate]["ref"]) if self.current_candidate else None})
        # Scope: return the newly admitted Patch contract in the same upstream observation.
        # 为什么要有它：连续会话保留旧历史；模型必须同时看到新binding的源码前像，才能按新Plan合法修订而不是猜测ID或路径。
        contract_messages = [{"role": message.role, "content": deepcopy(message.content)}
                             for message in self.contract.messages]
        return {**self.model_context(), "candidate_submission_contract_messages": contract_messages,
                "old_candidate_invalidated": self.candidate_mode != "current",
                "current_code_preserved": self.current_candidate is not None, "new_check_required": True,
                "assessment_sha256": result.assessment_ref["sha256"], "upstream_outcome": outcome,
                "model_feedback": deepcopy(dict(feedback)), "candidate_submission_allowed": not self._upstream_blocked}

    def _confirm_reuse(self, role, prepared, previous):
        self.inspector.validate_snapshot()
        if role == "candidate_read":
            if not self.policy.reuse_covered_source_observations:
                return None
            selected = prepared.value
            self._context_precondition(selected, required=True)
            raw, code_sha, _origin = self._version_file(selected["version"], selected["path"])
            if prepared.version_identity != _sha(selected):
                return None
            output = previous.get("result", {}).get("output", {})
            value = output.get("value", output)
            if (value.get("version") != selected["version"] or value.get("path") != selected["path"]
                    or value.get("file_sha256") != hashlib.sha256(raw).hexdigest()
                    or value.get("code_sha256") != code_sha or value.get("context_id") != self.context_id):
                return None
            return {"mode": "reuse", "version_identity": prepared.version_identity, "evidence": {
                "candidate_version": selected["version"], "file_sha256": hashlib.sha256(raw).hexdigest(),
                "context_id": self.context_id, "budget_scope": "value.read_budget is historical; pre_batch_read_budget is before other reads in this response",
                "pre_batch_read_budget": self._read_budget_model_view().get("read_budget")}}
        if role == "source_read":
            # Scope: same-version read reuse only under the selected existing policy.
            # 为什么要有它：共同对照不能给B暗加免费读取；关闭时保留core既有call_id拒绝，不伪造新调用或放宽源码范围。
            if not self.policy.reuse_covered_source_observations:
                return None
            if prepared.version_identity != self._read_version() or prepared.value not in self.inspector.available_read_binding_ids:
                return None
            return {"mode": "reuse", "version_identity": prepared.version_identity,
                    "evidence": {"source_snapshot_sha256": self.inspector.read_source_snapshot_sha256}}
        if role == "finish":
            current = self._prepare_candidate_action({"candidate_id": prepared.value})
            if current.version_identity != prepared.version_identity:
                return None
            return {"mode": "refresh", "version_identity": current.version_identity,
                    "evidence": {"candidate_id": prepared.value, "context_id": self.context_id,
                                 "checked_report_ref": self.current_check}}
        return None

    def actions(self):
        text = {"type": "string", "minLength": 1}
        descriptions = {
            "source_read": "Read one registered current-context binding from unchanged B0; no source writes. Use returned context_id when referring to a new context.",
            "source_search": "Search the authorized source using the real bounded retriever. Results are observations, not a revised Plan or new edit permission.",
            "candidate_submit": "Submit current context_id plus patch_json containing the exact native Patch JSON. Materializes a candidate without source Apply; it does not run tests.",
            "candidate_check": "Run the configured complete candidate checker, including its registered compilation/behavior checks. Uses the exact current candidate; keeps real failure observations.",
            "upstream": "Ask the registered Evidence/Planning owner to reassess a concrete gap or technical assumption. A returned context invalidates old candidate completion, but never expands authorization.",
            "finish": "Request independent completion only when this exact current candidate has a current candidate_check "
                "with status=passed. A failed or stale check requires investigation/revision and a fresh passed check, "
                "not just another check or another finish request. Text is not completion.",
            "candidate_read": "Read an authorized candidate file or inclusive line range; end_line beyond EOF is explicitly clamped. version is baseline or an explicit returned candidate_id from this context; use the latest ID for current code, never the ambiguous word current. Optional compare_to returns the complete selected-file diff. No tests, hidden answers or source writes. Costs one read plus all returned observation bytes. A reused value contains historical budget, not a new current allowance; see reuse_confirmation. 为什么要有它：明确版本避免相同调用ID误复用旧代码，读取继续计入原额度。",
        }
        if self.candidate_mode == "current":
            descriptions["source_read"] = "Read a registered binding from current candidate code when changed, otherwise unchanged source; returns explicit baseline/candidate identities. Original Plan and Apply preimages stay unchanged."
            descriptions["candidate_submit"] = (("No current candidate exists. Submit only context_id and patch_json against B0. "
                "Do not invent a parent candidate or use a source-file hash as parent_code_sha256. "
                "Creates an immutable candidate without source Apply or tests.")
                if self.current_candidate is None else
                ("Revise the existing current candidate. Submit context_id, parent_candidate_id, parent_code_sha256 "
                "and edits[{edit_binding_id,before_text,after_text,rationale}] using its returned identity. "
                "Each exact edit applies sequentially to CURRENT candidate text inside its existing binding; all previous edits remain. "
                "No patch_json or B0 reset. Creates an immutable snapshot and invalidates the prior check, never source Apply."))
            descriptions["source_search"] = (
                "Search authorized source with the current candidate overlay when configured; returned version proof identifies candidate versus unchanged source. "
                "Use candidate_read with an explicit candidate_id for complete current code and differences. "
                "Results do not revise the Plan or grant new edit permission. "
                "An authorized target_source item can include reassessment_ref; pass that exact ref to an Evidence upstream.source_refs "
                "request when it should be considered for a new edit binding. "
                "为什么要有它：搜索和读取必须明确对应当前代码，同时观察不能授予新的修改权限。")

        def definition(name, properties, required):
            reads = "source_read/candidate_read/source_search" if self.candidate_mode == "current" else "source_read/source_search"
            batching = (" May be batched only with " + reads + ". 为什么要有它：只读批量不能夹带改变上下文或候选的动作。"
                if name in {"source_read", "source_search", "candidate_read"} else
                " Invoke alone, without any other tool call in the same response. 为什么要有它：该动作需要先返回结果，下一步才能使用正确的上下文和候选版本。")
            return AgentToolDefinition(name, descriptions[name] + batching, {"type": "object", "additionalProperties": False,
                "properties": properties, "required": required})

        result = [TaskAgentAction(definition("source_read", {"binding_id": text, "context_id": text,
                "comment": {"type": "string"}, "reason": {"type": "string"}}, ["binding_id"]), "source_read",
                self._prepare_read, self._read_source, True,
                lambda prepared, previous: self._confirm_reuse("source_read", prepared, previous)),
            TaskAgentAction(definition("candidate_submit", {"context_id": text, "patch_json": text}, ["context_id", "patch_json"]),
                "candidate_submit", self._prepare_submit, self._submit, False),
            TaskAgentAction(definition("candidate_check", {"candidate_id": text}, ["candidate_id"]), "candidate_check",
                self._prepare_candidate_action, self._check, False),
            TaskAgentAction(definition("finish", {"candidate_id": text}, ["candidate_id"]), "finish",
                self._prepare_candidate_action, lambda candidate_id: {"candidate_id": candidate_id}, True,
                lambda prepared, previous: self._confirm_reuse("finish", prepared, previous))]
        if self.candidate_mode == "current":
            form = _candidate_submit_forms()[int(self.current_candidate is not None)]
            # Scope: disclose exact current preconditions; execution still verifies the submitted values.
            # 为什么要有它：宿主已知的版本标识不应要求模型猜测，单值枚举不会替模型改参或放松原件核验。
            form["properties"]["context_id"] = {**form["properties"]["context_id"], "enum": [self.context_id]}
            if self.current_candidate is not None:
                current = self.candidates[self.current_candidate]["data"]
                for field, value in (("parent_candidate_id", self.current_candidate),
                        ("parent_code_sha256", current["materialized_code_identity"]["code_sha256"])):
                    form["properties"][field] = {**form["properties"][field], "enum": [value]}
            submit_definition = definition("candidate_submit", form["properties"], form["required"])
            result[1] = TaskAgentAction(submit_definition,
                "candidate_submit", self._prepare_submit, self._submit, False)
            result.insert(1, TaskAgentAction(definition("candidate_read", {"context_id": text,
                "version": text, "path": text, "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1}, "compare_to": text},
                ["context_id", "version", "path"]), "source_read", self._prepare_candidate_read, self._read_candidate, True,
                lambda prepared, previous: self._confirm_reuse("candidate_read", prepared, previous)))
        if callable(self.source_search_owner):
            result.append(TaskAgentAction(definition("source_search", {"query": text, "context_id": text}, ["query"]),
                "source_search", self._prepare_search, self._search, True))
        if callable(self.upstream_owner):
            upstream_schema = {"type": "object", "additionalProperties": False,
                "required": ["required_owner", "reason", "source_refs"], "properties": {
                    "required_owner": {"type": "string", "enum": ["evidence", "planning"]},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "source_refs": {"type": "array", "minItems": 1, "uniqueItems": True,
                                    "items": {"type": "string", "minLength": 1}}}}
            result.append(TaskAgentAction(definition("upstream", {"context_id": text, "request": upstream_schema},
                ["context_id", "request"]), "upstream", self._prepare_upstream, self._upstream, False))
        return result

    def record_event(self, event):
        self._save("loop-event", event)
        if self.record_event_owner is not None:
            self.record_event_owner(deepcopy(event))

    def run(self, *, client, initial_request=None, **options):
        if not self.qualification()["ready"]:
            raise ValueError("TASK_TOOL_OWNERS_NOT_QUALIFIED")
        return run_task_agent_loop(client=client, initial_request=initial_request or self.initial_request(),
            actions=self.actions(), reserve_model_call=self.reserve_model_call, budget_snapshot=self.budget_snapshot,
            charge_tool_budget=self.charge_tool_budget, completion_check=self.completion_check,
            response_admission=self.response_admission, record_event=self.record_event, stop_check=self.stop_check,
            tool_definitions=lambda: tuple(action.definition for action in self.actions()), **options)
