"""Bounded Plan-bound source inspection loop for Patch generation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from itertools import count
from math import inf
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from src.core.trace.execution_trace import trace_operation
from src.libs.llm.agent_turn import (
    AgentMessage,
    AgentResponseFormat,
    AgentToolCall,
    AgentToolDefinition,
    AgentToolExchange,
    AgentToolResult,
    AgentTurnClient,
    AgentTurnProviderError,
    AgentTurnRequest,
    AgentTurnResponse,
)
from src.mod_agent.proposal.code_generator import (
    PatchGenerationError,
    PatchSubmissionValidationError,
    validate_patch_submission,
    validate_patch_submission_schema,
)
from src.mod_agent.proposal.noop_submission_feedback import (
    noop_code_identity,
    project_whole_noop_response,
)
from src.mod_agent.proposal.tool_argument_feedback import (
    argument_feedback,
    classify_source_arguments,
    correction_used,
    retained_correction_state,
)
from src.mod_agent.proposal.tool_batch_feedback import reject_nonread_batch

PATCH_SOURCE_INSPECTION_TOOL_NAME = "inspect_plan_binding_source"
PATCH_UPSTREAM_REQUEST_TOOL_NAME = "request_patch_upstream"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def iter_patch_recovery_revisions(loop: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Project recorded segments once, retaining compatibility with old receipts."""
    if "recovery_revisions" in loop:
        revisions = loop["recovery_revisions"]
        if not isinstance(revisions, list) or any(not isinstance(item, Mapping) for item in revisions):
            raise ValueError("Patch recovery revisions must be a list of recorded segments.")
        return tuple(revisions)
    return tuple(loop[name] for name in ("validator_correction", "behavior_correction")
        if isinstance(loop.get(name), Mapping))


def normalize_upstream_effects(value: Mapping[str, str] | None) -> dict[str, str]:
    """Describe caller-owned execution facts; this does not enable a route."""
    if value is None:
        return {"evidence": "record_and_stop", "planning": "record_and_stop"}
    choices = {"record_and_stop", "assess_then_stop", "assess_and_continue", "provider_unavailable_stop"}
    if (not isinstance(value, Mapping) or set(value) != {"evidence", "planning"}
            or any(not isinstance(item, str) or item not in choices for item in value.values())):
        raise PatchAgentLoopError("PATCH_AGENT_UPSTREAM_EFFECTS_INVALID", stage="loop_admission",
            reason="Upstream effects must describe both existing owners using supported execution modes.")
    return dict(value)


class PatchAgentLoopError(RuntimeError):
    """Reject an invalid Patch loop input or a drifted source boundary."""

    def __init__(self, code: str, *, stage: str, reason: str) -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage
        self.reason = reason


@dataclass(frozen=True)
class PatchAgentLoopPolicy:
    """Shared model and binding-read limits plus source-observation controls."""

    max_observation_bytes: int | None = 32_768
    context_radius_lines: int = 64
    finalize_after_first_observation: bool = True
    enable_supporting_reads: bool = False
    require_source_inspection: bool = True
    allow_upstream_request: bool = False
    allow_batched_reads: bool = False
    deduplicate_source_excerpts: bool = False
    reuse_covered_source_observations: bool = False
    diagnose_behavior_failures: bool = False
    # One dispatched binding read consumes one unit, including in a batch.
    max_tool_rounds: int | None = 3
    max_total_model_turns: int | None = 5
    model_call_budget_policy: str = "stage_and_total"

    def __post_init__(self) -> None:
        if any(type(getattr(self, name)) is not bool for name in (
            "finalize_after_first_observation", "enable_supporting_reads",
            "require_source_inspection", "allow_upstream_request", "allow_batched_reads",
            "deduplicate_source_excerpts", "reuse_covered_source_observations",
            "diagnose_behavior_failures",
        )):
            raise PatchAgentLoopError(
                "PATCH_AGENT_POLICY_INVALID",
                stage="policy",
                reason="Patch strategy switches must be boolean values.",
            )
        if self.model_call_budget_policy not in {"stage_and_total", "task_total_only", "meter_only"}:
            raise PatchAgentLoopError("PATCH_AGENT_POLICY_INVALID", stage="policy",
                reason="Unknown model-call budget policy.")
        metered = self.model_call_budget_policy == "meter_only"
        if metered:
            for name in ("max_total_model_turns", "max_tool_rounds", "max_observation_bytes"):
                object.__setattr__(self, name, None)
        if not metered and (type(self.max_total_model_turns) is not int or self.max_total_model_turns < 1
                or self.model_call_budget_policy == "stage_and_total" and self.max_total_model_turns > 5):
            raise PatchAgentLoopError("PATCH_AGENT_POLICY_INVALID", stage="policy",
                reason="The selected model-turn limit must be positive; legacy workers retain their five-turn bound.")
        if not metered and (type(self.max_tool_rounds) is not int or self.max_tool_rounds < 0
                or (self.max_tool_rounds == 0 and self.require_source_inspection)):
            raise PatchAgentLoopError("PATCH_AGENT_POLICY_INVALID", stage="policy",
                reason="The binding-read limit must be non-negative, and positive when source inspection is required.")
        for name in ("max_observation_bytes", "context_radius_lines"):
            if name == "max_observation_bytes" and metered:
                continue
            value = getattr(self, name)
            minimum = 1 if name == "context_radius_lines" else 0
            if type(value) is not int or value < minimum:
                raise PatchAgentLoopError(
                    "PATCH_AGENT_POLICY_INVALID",
                    stage="policy",
                    reason=f"{name} must be an integer greater than or equal to {minimum}.",
                )


@dataclass(frozen=True)
class PatchSourceObservation:
    """One path-hidden source excerpt bound to an admitted Plan edit binding."""

    observation_id: str
    binding_id: str
    source_ref: str
    source_hash: str
    source_snapshot_sha256: str
    target_span: dict[str, int] | None
    excerpt_span: dict[str, int] | None
    excerpt: str
    truncated: bool
    payload_sha256: str
    model_payload_bytes: int
    registered_evidence_identity: Mapping[str, Any] | None = None
    source_identity_sha256: str | None = None

    def model_view(self) -> dict[str, object]:
        """Return the bounded ref-first payload visible to the model."""

        return {
            "observation_id": self.observation_id,
            "binding_id": self.binding_id,
            "source_ref": self.source_ref,
            "source_hash": self.source_hash,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "target_span": dict(self.target_span) if self.target_span is not None else None,
            "excerpt_span": dict(self.excerpt_span) if self.excerpt_span is not None else None,
            "excerpt": self.excerpt,
            "truncated": self.truncated,
            "observation_sha256": self.payload_sha256,
            **({"source_identity_sha256": self.source_identity_sha256} if self.source_identity_sha256 else {}),
            **(deepcopy(dict(self.registered_evidence_identity)) if self.registered_evidence_identity is not None else {}),
        }


def _excerpt_reference_error(reason: str) -> PatchAgentLoopError:
    return PatchAgentLoopError("PATCH_AGENT_EXCERPT_REFERENCE_INVALID", stage="observation_admission", reason=reason)


def _local_excerpt_key(payload: Mapping[str, Any]) -> tuple[object, ...] | None:
    if payload.get("observation_kind") == "registered_api_evidence":
        return None
    span, excerpt = payload.get("excerpt_span"), payload.get("excerpt")
    if not isinstance(span, Mapping) or not isinstance(excerpt, str):
        return None
    return (payload.get("source_snapshot_sha256"), payload.get("source_hash"), payload.get("source_identity_sha256"),
            span.get("start_line"), span.get("end_line"), excerpt)


def _covered_excerpt(earlier: Mapping[str, Any], current: Mapping[str, Any]) -> str | None:
    """Return the exact complete requested window; never merge overlapping windows."""
    if not current.get("source_identity_sha256") or any(current.get(key) != earlier.get(key)
            for key in ("source_snapshot_sha256", "source_hash", "source_identity_sha256")):
        return None
    before, after = earlier.get("excerpt_span"), current.get("excerpt_span")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or not isinstance(earlier.get("excerpt"), str):
        return None
    values = [span.get(key) for span in (before, after) for key in ("start_line", "end_line")]
    if any(type(value) is not int for value in values):
        return None
    start, end, requested_start, requested_end = values
    if not 1 <= start <= requested_start <= requested_end <= end:
        return None
    lines = earlier["excerpt"].splitlines(keepends=True)
    if len(lines) != end - start + 1:
        return None
    return "".join(lines[requested_start - start:requested_end - start + 1])


def _overlapping_excerpt_parts(earlier: Mapping[str, Any], current: Mapping[str, Any]) -> list[Any] | None:
    """Keep new lines literal and reference only a preceding complete body."""
    if not current.get("source_identity_sha256") or any(current.get(key) != earlier.get(key)
            for key in ("source_snapshot_sha256", "source_hash", "source_identity_sha256")):
        return None
    before, after = earlier.get("excerpt_span"), current.get("excerpt_span")
    if (not isinstance(before, Mapping) or not isinstance(after, Mapping)
            or not isinstance(earlier.get("excerpt"), str) or not isinstance(current.get("excerpt"), str)):
        return None
    values = [span.get(key) for span in (before, after) for key in ("start_line", "end_line")]
    if any(type(value) is not int for value in values):
        return None
    start, end = max(values[0], values[2]), min(values[1], values[3])
    old_lines, new_lines = earlier["excerpt"].splitlines(keepends=True), current["excerpt"].splitlines(keepends=True)
    if (start > end or len(old_lines) != values[1] - values[0] + 1
            or len(new_lines) != values[3] - values[2] + 1):
        return None
    old_text = "".join(old_lines[start - values[0]:end - values[0] + 1])
    if old_text != "".join(new_lines[start - values[2]:end - values[2] + 1]):
        return None
    prefix = "".join(new_lines[:start - values[2]])
    suffix = "".join(new_lines[end - values[2] + 1:])
    if not prefix and not suffix:
        return None
    return [*([prefix] if prefix else []), {
        "observation_id": earlier["observation_id"],
        "observation_sha256": earlier["observation_sha256"],
        "excerpt_sha256": hashlib.sha256(earlier["excerpt"].encode("utf-8")).hexdigest(),
        "source_excerpt_span": {"start_line": start, "end_line": end},
    }, *([suffix] if suffix else [])]


def _expand_excerpt_parts(parts, current, full_by_id):
    if (not isinstance(parts, list) or not 2 <= len(parts) <= 3
            or sum(isinstance(part, Mapping) for part in parts) != 1
            or current.get("observation_kind") == "registered_api_evidence"):
        raise _excerpt_reference_error("Partial excerpts require literal text and one preceding full-body reference.")
    span = current.get("excerpt_span")
    if not isinstance(span, Mapping) or any(type(span.get(key)) is not int for key in ("start_line", "end_line")):
        raise _excerpt_reference_error("The partial excerpt line interval is invalid.")
    cursor, text = span["start_line"], []
    for part in parts:
        if isinstance(part, str) and part:
            value = part
        elif isinstance(part, Mapping) and set(part) == {
                "observation_id", "observation_sha256", "excerpt_sha256", "source_excerpt_span"}:
            earlier = full_by_id.get(part["observation_id"]) if isinstance(part["observation_id"], str) else None
            selected = part["source_excerpt_span"]
            if (earlier is None or _local_excerpt_key(earlier) is None
                    or not current.get("source_identity_sha256")
                    or any(current.get(key) != earlier.get(key) for key in
                        ("source_snapshot_sha256", "source_hash", "source_identity_sha256"))
                    or part["observation_sha256"] != earlier["observation_sha256"]
                    or part["excerpt_sha256"] != hashlib.sha256(earlier["excerpt"].encode("utf-8")).hexdigest()
                    or not isinstance(selected, Mapping) or set(selected) != {"start_line", "end_line"}
                    or any(type(selected[key]) is not int for key in selected)
                    or selected["start_line"] != cursor
                    or not earlier["excerpt_span"]["start_line"] <= cursor <= selected["end_line"] <= earlier["excerpt_span"]["end_line"]):
                raise _excerpt_reference_error("A partial excerpt reference is missing, mismatched or noncontiguous.")
            lines = earlier["excerpt"].splitlines(keepends=True)
            value = "".join(lines[cursor - earlier["excerpt_span"]["start_line"]:
                selected["end_line"] - earlier["excerpt_span"]["start_line"] + 1])
        else:
            raise _excerpt_reference_error("A partial excerpt component is invalid.")
        text.append(value)
        cursor += len(value.splitlines())
    if cursor != span["end_line"] + 1:
        raise _excerpt_reference_error("Partial excerpt components do not cover the requested line interval.")
    return "".join(text)


def expand_patch_source_observations(
    ordered_payloads: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Pure expansion of full observations and exact prior-full-excerpt references.

    Each restored observation keeps its original semantic hash. Only preceding
    complete local-source observations may supply excerpt bytes; API Evidence
    and reference chains are not excerpt providers.
    """
    if not isinstance(ordered_payloads, Sequence) or isinstance(ordered_payloads, (str, bytes)):
        raise _excerpt_reference_error("Observation history must be an ordered sequence.")
    full_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    expanded: list[dict[str, Any]] = []
    for payload in ordered_payloads:
        if not isinstance(payload, Mapping):
            raise _excerpt_reference_error("An observation must be a mapping.")
        restored = deepcopy(dict(payload))
        observation_id = restored.get("observation_id")
        if type(observation_id) is not str or not observation_id or observation_id in seen_ids:
            raise _excerpt_reference_error("Observation identities must be non-empty and unique in ordered history.")
        for key in ("observation_sha256", "source_snapshot_sha256"):
            if type(restored.get(key)) is not str or _SHA256_RE.fullmatch(restored[key]) is None:
                raise _excerpt_reference_error("An observation is missing its exact hash identity.")
        if (type(restored.get("source_hash")) is not str
                or re.fullmatch(r"[0-9a-fA-F]{64}", restored["source_hash"]) is None):
            raise _excerpt_reference_error("The original source hash is missing.")
        if "source_identity_sha256" in restored and (type(restored["source_identity_sha256"]) is not str
                or _SHA256_RE.fullmatch(restored["source_identity_sha256"]) is None):
            raise _excerpt_reference_error("The source identity is invalid.")
        has_reference = "excerpt_ref" in restored
        has_parts = "excerpt_parts" in restored
        reference = restored.pop("excerpt_ref", None)
        parts = restored.pop("excerpt_parts", None)
        if has_parts:
            if has_reference or "excerpt" in restored:
                raise _excerpt_reference_error("Partial excerpt fields conflict with a literal/full reference.")
            restored["excerpt"] = _expand_excerpt_parts(parts, restored, full_by_id)
        if has_reference:
            if ("excerpt" in restored or restored.get("observation_kind") == "registered_api_evidence"
                    or not isinstance(reference, Mapping)
                    or set(reference) not in ({"observation_id", "observation_sha256", "excerpt_sha256"},
                        {"observation_id", "observation_sha256", "excerpt_sha256", "source_excerpt_span"})
                    or type(reference.get("observation_id")) is not str):
                raise _excerpt_reference_error("The excerpt reference shape is invalid.")
            earlier = full_by_id.get(reference["observation_id"])
            if earlier is None or _local_excerpt_key(earlier) is None:
                raise _excerpt_reference_error("The referenced preceding complete source observation is unavailable.")
            if (reference["observation_sha256"] != earlier["observation_sha256"]
                    or reference["excerpt_sha256"] != hashlib.sha256(earlier["excerpt"].encode("utf-8")).hexdigest()
                    or any(restored.get(key) != earlier.get(key)
                           for key in ("source_snapshot_sha256", "source_hash", "source_identity_sha256"))):
                raise _excerpt_reference_error("The excerpt reference disagrees with its source, span, or bytes.")
            if "source_excerpt_span" in reference:
                excerpt = _covered_excerpt(earlier, restored)
                if reference["source_excerpt_span"] != earlier["excerpt_span"] or excerpt is None:
                    raise _excerpt_reference_error("The complete requested window is not covered by the referenced source.")
                restored["excerpt"] = excerpt
            else:
                if restored.get("excerpt_span") != earlier.get("excerpt_span"):
                    raise _excerpt_reference_error("The exact excerpt reference has a different span.")
                restored["excerpt"] = earlier["excerpt"]
        if type(restored.get("excerpt")) is not str:
            raise _excerpt_reference_error("The complete source excerpt is unavailable.")
        identity_payload = {key: value for key, value in restored.items() if key != "observation_sha256"}
        if _canonical_sha256(identity_payload) != restored["observation_sha256"]:
            raise _excerpt_reference_error("Expanded observation bytes do not match the original observation identity.")
        seen_ids.add(observation_id)
        if not has_reference and not has_parts:
            full_by_id[observation_id] = restored
        expanded.append(restored)
    return tuple(expanded)


def _encode_source_excerpt(
    observation: PatchSourceObservation,
    full_excerpts: dict[tuple[object, ...], dict[str, Any]],
    *, reuse_covered: bool = False,
) -> dict[str, Any]:
    full = observation.model_view()
    key = _local_excerpt_key(full)
    earlier = full_excerpts.get(key) if key is not None else None
    if earlier is not None and "excerpt_parts" in earlier:
        # Reuse the visible literal-plus-full-reference recipe, not a reference
        # chain to a body which was never sent as a full excerpt.
        return {**{name: value for name, value in full.items() if name != "excerpt"},
                "excerpt_parts": deepcopy(earlier["excerpt_parts"])}
    if earlier is None and reuse_covered and key is not None:
        earlier = next((item for item in full_excerpts.values() if _covered_excerpt(item, full) == full["excerpt"]), None)
    if earlier is not None:
        encoded = {name: value for name, value in full.items() if name != "excerpt"}
        encoded["excerpt_ref"] = {
            "observation_id": earlier["observation_id"],
            "observation_sha256": earlier["observation_sha256"],
            "excerpt_sha256": hashlib.sha256(earlier["excerpt"].encode("utf-8")).hexdigest(),
        }
        if earlier["excerpt_span"] != full["excerpt_span"]:
            encoded["excerpt_ref"]["source_excerpt_span"] = dict(earlier["excerpt_span"])
        if reuse_covered or len(_canonical_json(encoded).encode("utf-8")) < observation.model_payload_bytes:
            return encoded
    if reuse_covered and key is not None:
        choices = []
        for anchor in full_excerpts.values():
            parts = _overlapping_excerpt_parts(anchor, full)
            if parts is not None:
                encoded = {**{name: value for name, value in full.items() if name != "excerpt"}, "excerpt_parts": parts}
                choices.append(encoded)
        if choices:
            encoded = min(choices, key=lambda item: len(_canonical_json(item).encode("utf-8")))
            if len(_canonical_json(encoded).encode("utf-8")) < observation.model_payload_bytes:
                full_excerpts.setdefault(key, encoded)
                return encoded
    if key is not None:
        full_excerpts.setdefault(key, full)
    return full


@dataclass(frozen=True)
class PatchAgentLoopResult:
    """Complete in-process Patch loop result and bounded execution receipt."""

    status: Literal["completed", "blocked"]
    final_content: str | None
    provider: str
    model: str
    model_turns_used: int
    tool_rounds_used: int
    total_observation_bytes: int
    usage_by_turn: tuple[dict[str, object], ...]
    tool_call_trace: tuple[dict[str, object], ...]
    observation_trace: tuple[dict[str, object], ...]
    termination: dict[str, object]
    # In-process context for a subsequent correction; deliberately excluded from receipts.
    final_request: AgentTurnRequest | None = field(default=None, repr=False, compare=False)
    tool_availability_trace: tuple[dict[str, object], ...] = ()
    read_accounting_version: str = "binding_dispatch_v1"
    dispatched_binding_reads: int = 0
    reused_source_observations: int = 0
    turn_progress_trace: tuple[dict[str, object], ...] = ()
    submission_trace: tuple[dict[str, object], ...] = ()
    tool_argument_correction_used: bool = False

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, object]:
        final_bytes = (
            None
            if self.final_content is None
            else self.final_content.encode("utf-8")
        )
        return {
            "status": self.status,
            "final_content_sha256": (
                None if final_bytes is None else hashlib.sha256(final_bytes).hexdigest()
            ),
            "final_content_utf8_bytes": (
                None if final_bytes is None else len(final_bytes)
            ),
            "provider": self.provider,
            "model": self.model,
            "model_turns_used": self.model_turns_used,
            "tool_rounds_used": self.tool_rounds_used,
            "read_accounting_version": self.read_accounting_version,
            "dispatched_binding_reads": self.dispatched_binding_reads,
            "reused_source_observations": self.reused_source_observations,
            "turn_progress_trace": [deepcopy(item) for item in self.turn_progress_trace],
            "total_observation_bytes": self.total_observation_bytes,
            "usage_by_turn": [deepcopy(item) for item in self.usage_by_turn],
            "tool_call_trace": [deepcopy(item) for item in self.tool_call_trace],
            "observation_trace": [
                deepcopy(item) for item in self.observation_trace
            ],
            "termination": deepcopy(self.termination),
            "tool_availability_trace": [deepcopy(item) for item in self.tool_availability_trace],
            "tool_argument_correction_used": self.tool_argument_correction_used,
            **({"submission_trace": [deepcopy(item) for item in self.submission_trace]}
               if self.submission_trace else {}),
        }


@dataclass(frozen=True)
class _BoundSource:
    binding_id: str
    path: Path
    relative_path: str
    source_hash: str
    target_span: dict[str, int]


@dataclass(frozen=True)
class _RegisteredApiEvidence:
    """Immutable registered Evidence content; its source file is never opened here."""

    binding_id: str
    source_hash: str
    excerpt: str
    evidence_id: str
    evidence_item_sha256: str
    excerpt_sha256: str
    source_authority: str
    evidence_source_ref: str
    risk_notes: tuple[str, ...] | None


class PlanBoundSourceInspector:
    """Read current registered source without promoting read scope into edit authority."""

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        implementation_plan: Mapping[str, Any],
        policy: PatchAgentLoopPolicy,
        evidence_pack: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(policy, PatchAgentLoopPolicy):
            raise PatchAgentLoopError(
                "PATCH_AGENT_POLICY_INVALID",
                stage="source_admission",
                reason="A PatchAgentLoopPolicy is required.",
            )
        try:
            root = Path(workspace_root).resolve(strict=True)
        except OSError as exc:
            raise PatchAgentLoopError(
                "PATCH_AGENT_WORKSPACE_INVALID",
                stage="source_admission",
                reason="The workspace root is unavailable.",
            ) from exc
        if not root.is_dir():
            raise PatchAgentLoopError(
                "PATCH_AGENT_WORKSPACE_INVALID",
                stage="source_admission",
                reason="The workspace root must be a directory.",
            )
        registry = implementation_plan.get("binding_registry")
        raw_bindings = registry.get("edit_bindings") if isinstance(registry, Mapping) else None
        if not isinstance(raw_bindings, (list, tuple)) or not raw_bindings:
            raise PatchAgentLoopError(
                "PATCH_AGENT_PLAN_BINDINGS_INVALID",
                stage="source_admission",
                reason="The ImplementationPlan has no edit bindings.",
            )
        raw_changes = implementation_plan.get("proposed_changes")
        if not isinstance(raw_changes, (list, tuple)) or not raw_changes:
            raise PatchAgentLoopError(
                "PATCH_AGENT_PLAN_CHANGES_INVALID",
                stage="source_admission",
                reason="The ImplementationPlan has no selected changes.",
            )
        selected_binding_ids: list[str] = []
        for raw_change in raw_changes:
            binding_id = (
                raw_change.get("edit_binding_id")
                if isinstance(raw_change, Mapping)
                else None
            )
            if type(binding_id) is not str or not binding_id:
                raise PatchAgentLoopError(
                    "PATCH_AGENT_PLAN_CHANGES_INVALID",
                    stage="source_admission",
                    reason="Every selected Plan change must name one edit binding id.",
                )
            selected_binding_ids.append(binding_id)
        if len(selected_binding_ids) != len(set(selected_binding_ids)):
            raise PatchAgentLoopError(
                "PATCH_AGENT_PLAN_CHANGES_INVALID",
                stage="source_admission",
                reason="Selected Plan change binding ids must be unique.",
            )
        selected_binding_set = set(selected_binding_ids)

        bindings: dict[str, _BoundSource] = {}
        binding_order: list[str] = []
        source_hashes: dict[str, str] = {}
        source_paths: dict[str, Path] = {}
        for raw_binding in raw_bindings:
            if not isinstance(raw_binding, Mapping):
                raise PatchAgentLoopError(
                    "PATCH_AGENT_PLAN_BINDINGS_INVALID",
                    stage="source_admission",
                    reason="Every edit binding must be an object.",
                )
            if raw_binding.get("binding_id") not in selected_binding_set:
                continue
            binding = self._admit_binding(root, raw_binding)
            if binding.binding_id in bindings:
                raise PatchAgentLoopError(
                    "PATCH_AGENT_PLAN_BINDINGS_INVALID",
                    stage="source_admission",
                    reason="Edit binding ids must be unique.",
                )
            previous_hash = source_hashes.get(binding.relative_path)
            if previous_hash is not None and previous_hash != binding.source_hash:
                raise PatchAgentLoopError(
                    "PATCH_AGENT_PLAN_SOURCE_IDENTITY_CONFLICT",
                    stage="source_admission",
                    reason="One source path has conflicting Plan hashes.",
                )
            bindings[binding.binding_id] = binding
            binding_order.append(binding.binding_id)
            source_hashes[binding.relative_path] = binding.source_hash
            source_paths[binding.relative_path] = binding.path
        if set(bindings) != selected_binding_set:
            raise PatchAgentLoopError(
                "PATCH_AGENT_PLAN_CHANGES_INVALID",
                stage="source_admission",
                reason="A selected Plan change does not resolve to one edit binding.",
            )
        binding_order = selected_binding_ids

        edit_source_hashes = dict(source_hashes)
        read_binding_order = list(binding_order)
        registered_api: dict[str, _RegisteredApiEvidence] = {}
        if policy.enable_supporting_reads:
            raw_supporting = registry.get("supporting_bindings", [])
            if not isinstance(raw_supporting, (list, tuple)):
                raise PatchAgentLoopError("PATCH_AGENT_PLAN_BINDINGS_INVALID",
                    stage="source_admission", reason="Supporting bindings must be a list.")
            raw_items = evidence_pack.get("items") if isinstance(evidence_pack, Mapping) else None
            if raw_supporting and not isinstance(raw_items, (list, tuple)):
                raise PatchAgentLoopError("PATCH_AGENT_SUPPORTING_EVIDENCE_REQUIRED",
                    stage="source_admission", reason="Supporting reads require their EvidencePack.")
            evidence_by_id: dict[str, Mapping[str, Any]] = {}
            for item in raw_items or ():
                evidence_id = item.get("evidence_id") if isinstance(item, Mapping) else None
                if type(evidence_id) is not str or not evidence_id or evidence_id in evidence_by_id:
                    raise PatchAgentLoopError("PATCH_AGENT_SUPPORTING_EVIDENCE_MISMATCH",
                        stage="source_admission", reason="Evidence identities must be present and unique.")
                evidence_by_id[evidence_id] = item
            for raw_support in raw_supporting:
                if not isinstance(raw_support, Mapping):
                    raise PatchAgentLoopError("PATCH_AGENT_PLAN_BINDINGS_INVALID",
                        stage="source_admission", reason="Every supporting binding must be an object.")
                if raw_support.get("source_type") not in {"target_source", "decompiled_api"}:
                    continue
                item = evidence_by_id.get(raw_support.get("evidence_id"))
                if item is None or any(raw_support.get(plan_key) != item.get(evidence_key)
                    for plan_key, evidence_key in (("source_path", "source_path"),
                        ("source_type", "source_type"), ("target_span", "span"),
                        ("target_symbol", "symbol"))):
                    raise PatchAgentLoopError("PATCH_AGENT_SUPPORTING_EVIDENCE_MISMATCH",
                        stage="source_admission", reason="Supporting source identity differs from its exact Evidence item.")
                if item["source_type"] == "decompiled_api":
                    binding_id = raw_support.get("binding_id")
                    source_hash = item.get("source_hash")
                    excerpt = item.get("excerpt")
                    source_authority = item.get("source_authority")
                    evidence_source_ref = item.get("source_ref")
                    risk_notes = item.get("risk_notes")
                    if (type(binding_id) is not str or not binding_id or binding_id in read_binding_order
                        or type(source_hash) is not str or re.fullmatch(r"[0-9a-fA-F]{64}", source_hash) is None
                        or type(excerpt) is not str or not excerpt
                        or (item.get("file_hash") is not None and item["file_hash"] != source_hash)):
                        raise PatchAgentLoopError("PATCH_AGENT_REGISTERED_API_EVIDENCE_INVALID", stage="source_admission",
                            reason="Registered API content requires an exact source identity and nonempty excerpt.")
                    if (any(value is not None and (type(value) is not str or not value)
                            for value in (source_authority, evidence_source_ref))
                        or (risk_notes is not None and (not isinstance(risk_notes, (list, tuple))
                            or any(type(note) is not str for note in risk_notes)))):
                        raise PatchAgentLoopError("PATCH_AGENT_REGISTERED_API_EVIDENCE_INVALID", stage="source_admission",
                            reason="Optional registered API provenance must retain its source strings and risk-note list.")
                    registered_api[binding_id] = _RegisteredApiEvidence(binding_id=binding_id,
                        source_hash=source_hash, excerpt=excerpt, evidence_id=item["evidence_id"],
                        evidence_item_sha256=_canonical_sha256(item),
                        excerpt_sha256=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                        source_authority=source_authority if source_authority is not None else "unknown",
                        evidence_source_ref=evidence_source_ref if evidence_source_ref is not None else "unknown",
                        risk_notes=tuple(risk_notes) if risk_notes is not None else None)
                    read_binding_order.append(binding_id)
                    continue
                binding = self._admit_binding(root, {
                    "binding_id": raw_support.get("binding_id"),
                    "canonical_target_path": item.get("source_path"),
                    "source_hash": item.get("source_hash"), "target_span": item.get("span"),
                })
                if binding.binding_id in read_binding_order:
                    raise PatchAgentLoopError("PATCH_AGENT_PLAN_BINDINGS_INVALID",
                        stage="source_admission", reason="Read binding ids must be unique across edit and supporting scopes.")
                if binding.relative_path in source_hashes and source_hashes[binding.relative_path] != binding.source_hash:
                    raise PatchAgentLoopError("PATCH_AGENT_PLAN_SOURCE_IDENTITY_CONFLICT",
                        stage="source_admission", reason="Read and edit source identities conflict.")
                bindings[binding.binding_id] = binding
                read_binding_order.append(binding.binding_id)
                source_hashes[binding.relative_path] = binding.source_hash
                source_paths[binding.relative_path] = binding.path

        self.workspace_root = root
        self.policy = policy
        self._bindings = bindings
        self._registered_api = registered_api
        self._binding_order = tuple(binding_order)
        self._read_binding_order = tuple(read_binding_order)
        self._source_hashes = source_hashes
        self._source_paths = source_paths
        self._assert_snapshot_current(code="PATCH_AGENT_SOURCE_HASH_MISMATCH")
        self.source_snapshot_sha256 = _canonical_sha256(
            [
                {"source_path": relative_path, "source_hash": edit_source_hashes[relative_path]}
                for relative_path in sorted(edit_source_hashes)
            ]
        )
        self.read_source_snapshot_sha256 = _canonical_sha256([
            {"source_path": path, "source_hash": source_hashes[path]} for path in sorted(source_hashes)
        ])

    @property
    def available_binding_ids(self) -> tuple[str, ...]:
        return self._binding_order

    @property
    def available_read_binding_ids(self) -> tuple[str, ...]:
        return self._read_binding_order

    def validate_snapshot(self) -> str:
        """Revalidate every Plan-bound source and return the frozen snapshot hash."""

        self._assert_snapshot_current(code="PATCH_AGENT_SOURCE_SNAPSHOT_DRIFT")
        return self.source_snapshot_sha256

    def inspect(
        self,
        *,
        binding_id: str,
        observation_id: str,
    ) -> PatchSourceObservation:
        if type(binding_id) is not str or binding_id not in self._read_binding_order:
            raise PatchAgentLoopError(
                "PATCH_AGENT_BINDING_UNKNOWN",
                stage="tool_admission",
                reason="The requested binding id is not in the admitted Plan read scope.",
            )
        if type(observation_id) is not str or not observation_id:
            raise PatchAgentLoopError(
                "PATCH_AGENT_OBSERVATION_ID_INVALID",
                stage="tool_admission",
                reason="The tool observation id must be non-empty text.",
            )
        self.validate_snapshot()
        if binding_id in self._registered_api:
            return self._inspect_registered_api(binding_id=binding_id, observation_id=observation_id)
        binding = self._bindings[binding_id]
        try:
            source_bytes = binding.path.read_bytes()
            source_text = source_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_READ_FAILED",
                stage="tool_execution",
                reason="The bound source could not be read as UTF-8.",
            ) from exc
        if hashlib.sha256(source_bytes).hexdigest() != binding.source_hash:
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_SNAPSHOT_DRIFT",
                stage="tool_execution",
                reason="The bound source changed during inspection.",
            )

        lines = source_text.splitlines(keepends=True)
        start_line = binding.target_span["start_line"]
        end_line = binding.target_span["end_line"]
        excerpt_start = max(1, start_line - self.policy.context_radius_lines)
        excerpt_end = min(len(lines), end_line + self.policy.context_radius_lines)
        excerpt = "".join(lines[excerpt_start - 1 : excerpt_end])
        base_payload: dict[str, object] = {
            "observation_id": observation_id,
            "binding_id": binding_id,
            "source_ref": f"binding://{binding_id}",
            "source_hash": binding.source_hash,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "target_span": dict(binding.target_span),
            "excerpt_span": {
                "start_line": excerpt_start,
                "end_line": excerpt_end,
            },
            "excerpt": excerpt,
            "truncated": excerpt_start > 1 or excerpt_end < len(lines),
        }
        source_identity = (_canonical_sha256({"relative_path": binding.relative_path, "source_hash": binding.source_hash})
            if self.policy.reuse_covered_source_observations else None)
        if source_identity is not None:
            base_payload["source_identity_sha256"] = source_identity
        payload_sha256 = _canonical_sha256(base_payload)
        model_view = {**base_payload, "observation_sha256": payload_sha256}
        model_payload_bytes = len(_canonical_json(model_view).encode("utf-8"))
        return PatchSourceObservation(
            observation_id=observation_id,
            binding_id=binding_id,
            source_ref=str(base_payload["source_ref"]),
            source_hash=binding.source_hash,
            source_snapshot_sha256=self.source_snapshot_sha256,
            target_span=dict(binding.target_span),
            excerpt_span={"start_line": excerpt_start, "end_line": excerpt_end},
            excerpt=excerpt,
            truncated=bool(base_payload["truncated"]),
            payload_sha256=payload_sha256,
            model_payload_bytes=model_payload_bytes,
            source_identity_sha256=source_identity,
        )

    def _inspect_registered_api(self, *, binding_id: str, observation_id: str) -> PatchSourceObservation:
        item = self._registered_api[binding_id]
        identity = {
            "observation_kind": "registered_api_evidence",
            "source_identity_status": "registered_not_live_revalidated",
            "evidence_id": item.evidence_id,
            "evidence_item_sha256": item.evidence_item_sha256,
            "excerpt_sha256": item.excerpt_sha256,
            # source_ref identifies this tool binding; the registered origin is
            # kept separately without promoting it into current live Evidence.
            "source_authority": item.source_authority,
            "evidence_source_ref": item.evidence_source_ref,
            "risk_notes": list(item.risk_notes) if item.risk_notes is not None else None,
        }
        base_payload = {
            "observation_id": observation_id, "binding_id": binding_id,
            "source_ref": f"binding://{binding_id}", "source_hash": item.source_hash,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "target_span": None, "excerpt_span": None,
            "excerpt": item.excerpt, "truncated": False, **identity,
        }
        payload_hash = _canonical_sha256(base_payload)
        return PatchSourceObservation(observation_id=observation_id, binding_id=binding_id,
            source_ref=f"binding://{binding_id}", source_hash=item.source_hash,
            source_snapshot_sha256=self.source_snapshot_sha256, target_span=None, excerpt_span=None,
            excerpt=item.excerpt, truncated=False, payload_sha256=payload_hash,
            model_payload_bytes=len(_canonical_json({**base_payload, "observation_sha256": payload_hash}).encode("utf-8")),
            registered_evidence_identity=identity)

    @staticmethod
    def _admit_binding(root: Path, raw_binding: Mapping[str, Any]) -> _BoundSource:
        binding_id = raw_binding.get("binding_id")
        relative_path = raw_binding.get("canonical_target_path")
        source_hash = raw_binding.get("source_hash")
        target_span = raw_binding.get("target_span")
        if type(binding_id) is not str or not binding_id:
            raise PatchAgentLoopError(
                "PATCH_AGENT_PLAN_BINDINGS_INVALID",
                stage="source_admission",
                reason="An edit binding id is missing.",
            )
        if type(relative_path) is not str or not relative_path:
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_PATH_INVALID",
                stage="source_admission",
                reason="An edit binding source path is missing.",
            )
        pure_path = PurePosixPath(relative_path)
        if (
            pure_path.is_absolute()
            or Path(relative_path).is_absolute()
            or ".." in pure_path.parts
            or "\\" in relative_path
        ):
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_PATH_ESCAPE",
                stage="source_admission",
                reason="An edit binding source path escapes the workspace.",
            )
        try:
            source_path = (root / Path(*pure_path.parts)).resolve(strict=True)
        except OSError as exc:
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_UNAVAILABLE",
                stage="source_admission",
                reason="A bound source file is unavailable.",
            ) from exc
        if not source_path.is_file() or not source_path.is_relative_to(root):
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_PATH_ESCAPE",
                stage="source_admission",
                reason="A bound source file is outside the workspace.",
            )
        if type(source_hash) is not str or _SHA256_RE.fullmatch(source_hash) is None:
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_HASH_INVALID",
                stage="source_admission",
                reason="An edit binding source hash is invalid.",
            )
        if not isinstance(target_span, Mapping):
            raise PatchAgentLoopError(
                "PATCH_AGENT_TARGET_SPAN_INVALID",
                stage="source_admission",
                reason="An edit binding target span is missing.",
            )
        start_line = target_span.get("start_line")
        end_line = target_span.get("end_line")
        if (
            type(start_line) is not int
            or type(end_line) is not int
            or start_line < 1
            or end_line < start_line
        ):
            raise PatchAgentLoopError(
                "PATCH_AGENT_TARGET_SPAN_INVALID",
                stage="source_admission",
                reason="An edit binding target span is invalid.",
            )
        try:
            source_bytes = source_path.read_bytes()
            line_count = len(source_bytes.decode("utf-8").splitlines())
        except (OSError, UnicodeDecodeError) as exc:
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_READ_FAILED",
                stage="source_admission",
                reason="A bound source could not be read as UTF-8.",
            ) from exc
        if hashlib.sha256(source_bytes).hexdigest() != source_hash:
            raise PatchAgentLoopError(
                "PATCH_AGENT_SOURCE_HASH_MISMATCH",
                stage="source_admission",
                reason="A Plan binding source hash does not match current source.",
            )
        if end_line > line_count:
            raise PatchAgentLoopError(
                "PATCH_AGENT_TARGET_SPAN_INVALID",
                stage="source_admission",
                reason="An edit binding target span exceeds current source.",
            )
        return _BoundSource(
            binding_id=binding_id,
            path=source_path,
            relative_path=pure_path.as_posix(),
            source_hash=source_hash,
            target_span={"start_line": start_line, "end_line": end_line},
        )

    def _assert_snapshot_current(self, *, code: str) -> None:
        for relative_path, expected_hash in self._source_hashes.items():
            source_path = self._source_paths[relative_path]
            try:
                observed_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            except OSError as exc:
                raise PatchAgentLoopError(
                    code,
                    stage="source_freshness",
                    reason="A Plan-bound source is unavailable.",
                ) from exc
            if observed_hash != expected_hash:
                raise PatchAgentLoopError(
                    code,
                    stage="source_freshness",
                    reason="A Plan-bound source changed after admission.",
                )


def _visible_source_payloads(request: AgentTurnRequest) -> list[Mapping[str, Any]]:
    results = [result for exchange in request.historical_tool_exchanges for result in exchange.tool_results]
    results.extend(request.tool_results)
    return [result.output for result in results if result.name == PATCH_SOURCE_INSPECTION_TOOL_NAME
        and isinstance(result.output, Mapping) and ("observation_sha256" in result.output or "excerpt_ref" in result.output)]


def _validated_visible_excerpts(source_inspector: PlanBoundSourceInspector,
        payloads: Sequence[Mapping[str, Any]]) -> dict[tuple[object, ...], dict[str, Any]]:
    anchors: dict[tuple[object, ...], dict[str, Any]] = {}
    for encoded, restored in zip(payloads, expand_patch_source_observations(payloads)):
        expected = source_inspector.inspect(binding_id=restored.get("binding_id"),
            observation_id=restored["observation_id"]).model_view()
        if restored != expected:
            raise _excerpt_reference_error("Retained observation does not match the current registered source identity.")
        key = _local_excerpt_key(restored)
        if key is not None and "excerpt_ref" not in encoded:
            anchors.setdefault(key, deepcopy(encoded) if "excerpt_parts" in encoded else restored)
    return anchors


def _plan_source_reads(source_inspector: PlanBoundSourceInspector, binding_ids: Sequence[str],
        full_excerpts: Mapping[tuple[object, ...], dict[str, Any]], observation_sequence: int):
    anchors = deepcopy(dict(full_excerpts))
    rows = []
    for index, binding_id in enumerate(binding_ids, observation_sequence + 1):
        observation = source_inspector.inspect(binding_id=binding_id, observation_id=f"patch-observation-{index:03d}")
        already_visible = _local_excerpt_key(observation.model_view()) in anchors
        encoded = _encode_source_excerpt(observation, anchors, reuse_covered=True)
        cost = 0 if "excerpt_ref" in encoded or already_visible else 1
        rows.append((observation, encoded, cost))
    return rows


def estimate_patch_source_reads(*, source_inspector: PlanBoundSourceInspector,
        prior_request: AgentTurnRequest, binding_ids: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Plan logical new-window and wire-payload costs; does not record tool execution."""
    payloads = _visible_source_payloads(prior_request)
    anchors = _validated_visible_excerpts(source_inspector, payloads)
    sequence = max([len(payloads), *(int(match.group(1)) for item in payloads
        if (match := re.fullmatch(r"patch-observation-(\d+)", str(item.get("observation_id")))))])
    rows = _plan_source_reads(source_inspector, binding_ids, anchors, sequence)
    return tuple({"binding_id": observation.binding_id, "read_cost": cost,
        "encoded_payload_bytes": len(_canonical_json(encoded).encode("utf-8")),
        "observation_origin": "prior_visible_source_reuse" if cost == 0 else "current_source_read",
        **({"excerpt_ref": deepcopy(encoded["excerpt_ref"])} if "excerpt_ref" in encoded else {})}
        for observation, encoded, cost in rows)


def run_patch_agent_loop(
    *,
    client: AgentTurnClient,
    initial_messages: tuple[AgentMessage, ...],
    final_response_format: AgentResponseFormat,
    source_inspector: PlanBoundSourceInspector,
    policy: PatchAgentLoopPolicy | None = None,
    resume_request: AgentTurnRequest | None = None,
    consumed_model_turns: int = 0,
    consumed_tool_rounds: int = 0,
    consumed_observation_bytes: int = 0,
    reserve_model_turns: int = 0,
    reserve_tool_reads: int = 0,
    reserve_observation_bytes: int = 0,
    upstream_effects: Mapping[str, str] | None = None,
    submission_validator: Callable[[AgentTurnResponse], None] | None = None,
    response_admission: Callable[[AgentTurnResponse], None] | None = None,
    request_projection: Callable[[AgentTurnRequest], AgentTurnRequest] | None = None,
    diagnosis_request: AgentTurnRequest | None = None,
    tool_argument_correction_used: bool = False,
) -> PatchAgentLoopResult:
    """Run bounded source inspection turns and return final Patch content or a receipt."""

    selected_policy = policy or source_inspector.policy
    metered = selected_policy.model_call_budget_policy == "meter_only"
    argument_used = correction_used(tool_argument_correction_used)
    argument_resume_request: AgentTurnRequest | None = None
    actual_upstream_effects = normalize_upstream_effects(upstream_effects)
    reuse_covered = selected_policy.reuse_covered_source_observations
    encode_excerpts = selected_policy.deduplicate_source_excerpts or reuse_covered
    for value in (consumed_model_turns, consumed_tool_rounds, consumed_observation_bytes,
                  reserve_model_turns, reserve_tool_reads, reserve_observation_bytes):
        if type(value) is not int or value < 0:
            raise PatchAgentLoopError("PATCH_AGENT_LOOP_BUDGET_INVALID", stage="loop_admission",
                reason="Consumed and reserved budget counters must be non-negative integers.")
    if not metered and (consumed_model_turns > selected_policy.max_total_model_turns or consumed_tool_rounds > selected_policy.max_tool_rounds):
        raise PatchAgentLoopError("PATCH_AGENT_LOOP_BUDGET_INVALID", stage="loop_admission",
            reason="Consumed counters exceed the shared policy.")
    if not metered and (reserve_model_turns > selected_policy.max_total_model_turns
            or reserve_tool_reads > selected_policy.max_tool_rounds
            or reserve_observation_bytes > selected_policy.max_observation_bytes):
        raise PatchAgentLoopError("PATCH_AGENT_LOOP_BUDGET_INVALID", stage="loop_admission",
            reason="Reserved counters exceed the shared policy.")
    if resume_request is not None and not isinstance(resume_request, AgentTurnRequest):
        raise PatchAgentLoopError("PATCH_AGENT_LOOP_INPUT_INVALID", stage="loop_admission",
            reason="Resuming requires an AgentTurnRequest.")
    if diagnosis_request is not None and not isinstance(diagnosis_request, AgentTurnRequest):
        raise PatchAgentLoopError("PATCH_AGENT_LOOP_INPUT_INVALID", stage="loop_admission",
            reason="An optional diagnosis must be an AgentTurnRequest.")
    if selected_policy != source_inspector.policy:
        raise PatchAgentLoopError(
            "PATCH_AGENT_POLICY_MISMATCH",
            stage="loop_admission",
            reason="The loop and source inspector must use one policy instance value.",
        )
    if (
        not initial_messages
        or any(not isinstance(message, AgentMessage) for message in initial_messages)
        or not isinstance(final_response_format, AgentResponseFormat)
    ):
        raise PatchAgentLoopError(
            "PATCH_AGENT_LOOP_INPUT_INVALID",
            stage="loop_admission",
            reason="Initial messages and a final response format are required.",
        )
    try:
        validate_patch_submission_schema(final_response_format.schema)
    except ValueError as exc:
        raise PatchAgentLoopError("PATCH_AGENT_SUBMISSION_CONTRACT_INVALID",
            stage="loop_admission", reason=str(exc)) from exc

    provider = str(getattr(client, "provider", "") or "unknown")
    model = str(getattr(client, "model", "") or "unknown")
    usage_by_turn: list[dict[str, object]] = []
    tool_call_trace: list[dict[str, object]] = []
    observation_trace: list[dict[str, object]] = []
    observations: list[PatchSourceObservation | Mapping[str, object]] = []
    submission_trace: list[dict[str, object]] = []
    tool_availability_trace: list[dict[str, object]] = []
    response_actions: dict[int, dict[str, object]] = {}
    inspected_binding_ids: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    seen_call_bindings: dict[str, str] = {}
    last_reused_only_batch: str | None = None
    rejected_read_batches: set[str] = set()
    total_observation_bytes = consumed_observation_bytes
    tool_rounds_used = consumed_tool_rounds
    allowed_model_end = inf if metered else selected_policy.max_total_model_turns - reserve_model_turns
    allowed_read_end = inf if metered else selected_policy.max_tool_rounds - reserve_tool_reads
    allowed_observation_end = inf if metered else selected_policy.max_observation_bytes - reserve_observation_bytes
    base_request = deepcopy(resume_request) if resume_request is not None else AgentTurnRequest(messages=initial_messages)
    old_states = [message.content for message in base_request.messages
        if _is_loop_state(message)]
    retained_used, rejected_argument_ids, argument_model_floor, argument_byte_floor, retryable_argument_calls = retained_correction_state(
        base_request, source_tool=PATCH_SOURCE_INSPECTION_TOOL_NAME)
    argument_used = argument_used or retained_used
    for state in old_states:
        argument_used = correction_used(state.get("tool_argument_correction_used", False)) or argument_used
        for observation in state.get("prior_observations", []):
            if isinstance(observation, Mapping):
                observations.append(deepcopy(observation))
    retained_exchanges = list(base_request.historical_tool_exchanges)
    if base_request.tool_results:
        retained_exchanges.append(AgentToolExchange(base_request.tool_results, base_request.provider_state,
            base_request.tool_result_insert_index if base_request.tool_result_insert_index is not None else len(base_request.messages)))
    full_excerpts: dict[tuple[object, ...], dict[str, Any]] = {}
    retained_payloads = _visible_source_payloads(base_request)
    if encode_excerpts:
        full_excerpts = _validated_visible_excerpts(source_inspector, retained_payloads)
    observation_sequence = max([len(retained_payloads), *(int(match.group(1)) for item in retained_payloads
        if (match := re.fullmatch(r"patch-observation-(\d+)", str(item.get("observation_id")))))])
    for exchange in retained_exchanges:
        rejected_bindings = [result.output.get("binding_id") for result in exchange.tool_results
            if isinstance(result.output, Mapping)
            and result.output.get("error_code") == "PATCH_AGENT_TOOL_BATCH_BUDGET_EXCEEDED"]
        if rejected_bindings and all(isinstance(binding, str) for binding in rejected_bindings):
            rejected_read_batches.add(_canonical_sha256(sorted(rejected_bindings)))
        for tool_result in exchange.tool_results:
            if (tool_result.name == PATCH_SOURCE_INSPECTION_TOOL_NAME and isinstance(tool_result.output, Mapping)
                and isinstance(tool_result.output.get("observation_sha256"), str)):
                seen_tool_call_ids.add(tool_result.call_id)
                binding_id = tool_result.output.get("binding_id")
                if isinstance(binding_id, str):
                    inspected_binding_ids.add(binding_id)
                    if tool_result.call_id in seen_call_bindings and seen_call_bindings[tool_result.call_id] != binding_id:
                        raise PatchAgentLoopError("PATCH_AGENT_TOOL_CALL_ID_REUSED", stage="loop_admission",
                            reason="A retained provider call id refers to different bindings.")
                    seen_call_bindings[tool_result.call_id] = binding_id
    for observation in observations:
        binding_id = observation.get("binding_id")
        if isinstance(binding_id, str):
            inspected_binding_ids.add(binding_id)
    if resume_request is not None:
        prior_model_floor = max((state.get("budget_state", {}).get("model_turns_used", 0) + 1
            for state in old_states), default=0)
        prior_read_floor = (len({(item.get("source_snapshot_sha256"), item.get("source_hash"),
            item.get("source_identity_sha256"), _canonical_sha256(item.get("excerpt_span")))
            for item in retained_payloads if "excerpt_ref" not in item})
            if reuse_covered else len(inspected_binding_ids))
        prior_read_floor = max(prior_read_floor, max(
            (state.get("budget_state", {}).get("tool_rounds_used", 0) for state in old_states), default=0))
        prior_byte_floor = max((state.get("budget_state", {}).get("observation_bytes_used", 0)
            for state in old_states), default=0)
        prior_model_floor = max(prior_model_floor, argument_model_floor)
        prior_byte_floor = max(prior_byte_floor, argument_byte_floor)
        if reuse_covered and any(state.get("read_execution", {}).get("accounting_version") != "new_source_window_v2"
                for state in old_states):
            raise PatchAgentLoopError("PATCH_AGENT_LOOP_BUDGET_INVALID", stage="loop_admission",
                reason="A resumed request must retain its original read-accounting version.")
        if (consumed_model_turns < prior_model_floor or consumed_tool_rounds < prior_read_floor
                or consumed_observation_bytes < prior_byte_floor):
            raise PatchAgentLoopError("PATCH_AGENT_LOOP_BUDGET_INVALID", stage="loop_admission",
                reason="Resume counters cannot discard calls, inspections or observation bytes recorded in the prior request.")
    base_request = _without_loop_states(base_request)
    rejected_submissions = {
        message.content["response_sha256"] for message in base_request.messages
        if message.role == "user" and isinstance(message.content, Mapping)
        and message.content.get("message_type") == "patch_submission_feedback"
        and isinstance(message.content.get("response_sha256"), str)
    }
    rejected_noop_codes = {identity for message in base_request.messages
        if message.role == "user" and isinstance(message.content, Mapping)
        and message.content.get("message_type") == "patch_submission_feedback"
        and (identity := noop_code_identity(message.content.get("diagnostics"))) is not None}
    format_correction_pending = False

    def result_for_segment(**kwargs: Any) -> PatchAgentLoopResult:
        if argument_used and kwargs.get("final_request") is None and argument_resume_request is not None:
            kwargs["final_request"] = argument_resume_request
        kwargs["model_turns_used"] -= consumed_model_turns
        kwargs["tool_rounds_used"] -= consumed_tool_rounds
        kwargs["total_observation_bytes"] -= consumed_observation_bytes
        progress = []
        for usage in usage_by_turn:
            turn = usage["turn_index"]
            calls = [item for item in tool_call_trace if item["turn_index"] == turn and item.get("dispatch_started")]
            observations_in_turn = [item for item in observation_trace if item["turn_index"] == turn]
            new_count = sum(item.get("read_cost", 1) for item in calls)
            progress.append({"turn_index": turn, **response_actions.get(turn, {}),
                "dispatched_binding_reads": len(calls), "new_source_windows": new_count if reuse_covered else None,
                "reused_source_observations": sum(item.get("dispatch_started") and item.get("read_cost") == 0 for item in calls),
                "read_cost": new_count, "no_new_source_information": new_count == 0 if reuse_covered else None,
                "observation_bytes_added": sum(item.get("encoded_payload_bytes", item["model_payload_bytes"])
                    for item in observations_in_turn) + sum(item.get("feedback_payload_bytes", 0)
                    for item in tool_call_trace if item["turn_index"] == turn)})
            progress[-1]["observation_bytes_added"] += sum(
                int(item.get("feedback_payload_bytes", 0)) for item in submission_trace
                if item["turn_index"] == turn and item.get("feedback_queued"))
            progress[-1]["observation_bytes_added"] += int(usage.get("diagnosis_context_bytes", 0))
        # Scope: aggregate only dispatched reads as reused source observations.
        # 为什么要有它：未执行的参数反馈read_cost为0，但它不是成功读取或已验证观察的复用。
        return replace(_loop_result(**kwargs), tool_availability_trace=tuple(deepcopy(tool_availability_trace)),
            read_accounting_version="new_source_window_v2" if reuse_covered else "binding_dispatch_v1",
            dispatched_binding_reads=sum(bool(item.get("dispatch_started")) for item in tool_call_trace),
            reused_source_observations=sum(bool(item.get("dispatch_started")) and item.get("read_cost") == 0 for item in tool_call_trace),
            turn_progress_trace=tuple(progress), submission_trace=tuple(deepcopy(submission_trace)),
            tool_argument_correction_used=argument_used)
    pending_response: AgentTurnResponse | None = None
    pending_observation: PatchSourceObservation | None = None
    pending_tool_results: tuple[AgentToolResult, ...] = ()
    force_final = False

    if consumed_model_turns >= allowed_model_end or consumed_observation_bytes > allowed_observation_end:
        return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
            model_turns_used=consumed_model_turns, tool_rounds_used=tool_rounds_used,
            total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
            tool_call_trace=tool_call_trace, observation_trace=observation_trace,
            termination=_termination("PATCH_AGENT_MODEL_TURN_BUDGET_EXHAUSTED" if consumed_model_turns >= allowed_model_end
                else "PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED", "No shared budget remains for this segment.", consumed_model_turns))

    turn_indices = count(consumed_model_turns + 1) if metered else range(
        consumed_model_turns + 1, selected_policy.max_total_model_turns + 1)
    for turn_index in turn_indices:
        if turn_index > allowed_model_end:
            return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                model_turns_used=turn_index - 1, tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                termination=_termination("PATCH_AGENT_MODEL_TURN_BUDGET_EXHAUSTED",
                    "No model turn remains in this segment.", turn_index - 1))
        try:
            source_inspector.validate_snapshot()
        except PatchAgentLoopError as exc:
            return result_for_segment(
                status="blocked",
                final_content=None,
                provider=provider,
                model=model,
                model_turns_used=turn_index - 1,
                tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes,
                usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace,
                observation_trace=observation_trace,
                termination=_termination(exc.code, exc.reason, turn_index - 1),
            )

        remaining_binding_ids = tuple(
            binding_id
            for binding_id in source_inspector.available_read_binding_ids
            if reuse_covered or binding_id not in inspected_binding_ids
        )
        if reuse_covered and tool_rounds_used >= allowed_read_end:
            # Exhausted new-window allowance still permits references to visible source.
            remaining_binding_ids = tuple(binding_id for binding_id in remaining_binding_ids
                if _plan_source_reads(source_inspector, [binding_id], full_excerpts, observation_sequence)[0][2] == 0)
        must_finish = (
            force_final
            or turn_index >= allowed_model_end
            or (tool_rounds_used >= allowed_read_end and not reuse_covered)
            or total_observation_bytes >= allowed_observation_end
            or (
                selected_policy.finalize_after_first_observation
                and tool_rounds_used >= 1
            )
            or not remaining_binding_ids
        )
        native_observation_ids = {
            result.output.get("observation_id")
            for exchange in base_request.historical_tool_exchanges
            for result in exchange.tool_results
            if isinstance(result.output, Mapping)
        }
        native_observation_ids.update(result.output.get("observation_id")
            for result in base_request.tool_results if isinstance(result.output, Mapping))
        if pending_observation is not None:
            native_observation_ids.add(pending_observation.observation_id)
        native_observation_ids.update(result.output.get("observation_id") for result in pending_tool_results
            if isinstance(result.output, Mapping))
        # Keep older non-native observations, but do not duplicate retained native payloads.
        prior_observations = [observation for observation in observations
            if (observation.observation_id if isinstance(observation, PatchSourceObservation)
                else observation.get("observation_id")) not in native_observation_ids]
        state_message = _loop_state_message(
            policy=selected_policy,
            source_inspector=source_inspector,
            prior_observations=prior_observations,
            model_turns_used=turn_index - 1,
            tool_rounds_used=tool_rounds_used,
            force_final=must_finish,
            final_response_format=final_response_format,
            reserve_model_turns=reserve_model_turns,
            reserve_tool_reads=reserve_tool_reads,
            total_observation_bytes=total_observation_bytes,
            reserve_observation_bytes=reserve_observation_bytes,
            upstream_effects=actual_upstream_effects,
            tool_argument_correction_used=argument_used,
        )
        tool_results = pending_tool_results or (
            (
                AgentToolResult(
                    call_id=pending_response.tool_calls[0].call_id,
                    name=pending_response.tool_calls[0].name,
                    output=pending_observation.model_view(),
                ),
            )
            if pending_response is not None and pending_observation is not None
            else ()
        )
        history = base_request.historical_tool_exchanges
        if tool_results and base_request.tool_results:
            history = (*history, AgentToolExchange(base_request.tool_results, base_request.provider_state,
                base_request.tool_result_insert_index if base_request.tool_result_insert_index is not None else len(base_request.messages)))
        available_tools = (
            () if must_finish else (_source_tool_definition(source_inspector, binding_ids=remaining_binding_ids),)
        ) + ((_upstream_tool_definition(source_inspector, upstream_effects=actual_upstream_effects),)
             if selected_policy.allow_upstream_request else ())
        request = AgentTurnRequest(
            messages=(*base_request.messages, state_message),
            tools=available_tools,
            tool_choice=(
                "none"
                if not available_tools
                else "required"
                if selected_policy.require_source_inspection and tool_rounds_used == 0
                else "auto"
            ),
            parallel_tool_calls=False,
            response_format=final_response_format if must_finish or format_correction_pending else None,
            tool_results=tool_results or base_request.tool_results,
            tool_result_insert_index=(len(base_request.messages) if tool_results else base_request.tool_result_insert_index),
            provider_state=(pending_response.provider_state if tool_results else base_request.provider_state),
            temperature=0.0,
            historical_tool_exchanges=history,
        )
        diagnosing = (diagnosis_request is not None and turn_index == consumed_model_turns + 1
                      and turn_index < allowed_model_end)
        if argument_used:
            # Scope: retain the actual source-tool exchange even when a diagnosis has no tools.
            # 为什么要有它：诊断请求失败不能丢掉已消耗纠错机会的原反馈和读取上下文。
            argument_resume_request = request
        if diagnosing:
            # 为什么要有它：诊断占用同一真实调用额度且不执行工具，保留至少一次提交机会；模型猜想不成为验收门禁。
            request = replace(diagnosis_request, tools=(), tool_choice="none", response_format=None,
                tool_results=(), historical_tool_exchanges=(), provider_state=None,
                tool_result_insert_index=None, parallel_tool_calls=False)
        availability: dict[str, object] = {"turn_index": turn_index,
            "allowed_tools": [tool.name for tool in request.tools],
            "budget_state": deepcopy(state_message.content["budget_state"]),
            "selection": "forbidden" if not request.tools else "allowed_not_selected"}
        tool_availability_trace.append(availability)
        feedback_in_request = {
            message.content.get("response_sha256") for message in request.messages
            if message.role == "user" and isinstance(message.content, Mapping)
            and message.content.get("message_type") == "patch_submission_feedback"
        }
        for entry in submission_trace:
            if entry.get("feedback_queued") and entry["response_sha256"] in feedback_in_request:
                entry["feedback_in_request"] = True
        sent_request = request
        if request_projection is not None and not diagnosing:
            try:
                sent_request = request_projection(request)
            except Exception as exc:
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index - 1, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                    termination=_termination(getattr(exc, "code", type(exc).__name__), str(exc), turn_index - 1))
        sent_argument_ids = retained_correction_state(sent_request, source_tool=PATCH_SOURCE_INSPECTION_TOOL_NAME)[1]
        for entry in tool_call_trace:
            if entry.get("argument_correction") and entry["tool_call_id"] in sent_argument_ids:
                entry.update(feedback_in_request=True, feedback_request_attempted=True,
                             feedback_request_turn=turn_index)
        try:
            # Retain logical history for source-cache validation and same-Plan
            # continuation; the recorder/provider receives the checked view.
            response = client.complete_turn(sent_request)
        except AgentTurnProviderError as exc:
            usage_by_turn.append(
                {
                    "turn_index": turn_index,
                    "provider": provider,
                    "model": model,
                    "request_id": None,
                    "stop_reason": None,
                    "usage": None,
                    "usage_available": False,
                }
            )
            return result_for_segment(
                status="blocked",
                final_content=None,
                provider=provider,
                model=model,
                model_turns_used=turn_index,
                tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes,
                usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace,
                observation_trace=observation_trace,
                termination=_termination(
                    "PATCH_AGENT_PROVIDER_TURN_FAILED",
                    str(exc) or "The Patch Agent provider turn failed.",
                    turn_index,
                    provider_diagnostic=exc.diagnostic,
                ),
            )
        if not isinstance(response, AgentTurnResponse):
            return result_for_segment(
                status="blocked",
                final_content=None,
                provider=provider,
                model=model,
                model_turns_used=turn_index,
                tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes,
                usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace,
                observation_trace=observation_trace,
                termination=_termination(
                    "PATCH_AGENT_RESPONSE_INVALID",
                    "The Patch Agent response type is invalid.",
                    turn_index,
                ),
            )
        provider = response.provider
        model = response.model
        for entry in tool_call_trace:
            if entry.get("argument_correction") and entry["tool_call_id"] in sent_argument_ids:
                entry["followup_response_observed"] = True
                entry.setdefault("consumed_in_model_turn", turn_index)
        for entry in submission_trace:
            if entry.get("feedback_queued") and entry["response_sha256"] in feedback_in_request:
                entry["followup_response_observed"] = True
                entry.setdefault("consumed_in_model_turn", turn_index)
        if response.tool_calls:
            availability["selection"] = "selected" if request.tools else "forbidden_attempt"
        usage_by_turn.append(_usage_record(turn_index, response))
        response_actions[turn_index] = {"submitted_patch": False,
            "text_response_received": bool(response.content and not response.tool_calls),
            "requested_source_reads": sum(call.name == PATCH_SOURCE_INSPECTION_TOOL_NAME for call in response.tool_calls)}
        pending_response = None
        pending_observation = None
        pending_tool_results = ()

        if response_admission is not None:
            try:
                response_admission(response)
            except Exception as exc:
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                    termination={**_termination(getattr(exc, "code", type(exc).__name__), str(exc), turn_index),
                        "failure_class": type(exc).__name__, "diagnostics": deepcopy(getattr(exc, "diagnostics", {}))})

        if diagnosing:
            usable = bool(response.content and not response.tool_calls and response.stop_reason not in {
                "length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal"})
            usage_by_turn[-1]["purpose"] = "failure_diagnosis"
            response_actions[turn_index]["diagnosis_usable"] = usable
            availability["selection"] = "diagnosis_no_tools_executed"
            for call in response.tool_calls:
                tool_call_trace.append(_tool_trace(call, turn_index=turn_index,
                    status="rejected_not_executed", code="PATCH_DIAGNOSIS_HAS_NO_TOOL_AUTHORITY",
                    dispatch_started=False))
            additions = ((AgentMessage("assistant", response.content),) if usable else ()) + (
                AgentMessage("user", {"message_type": "patch_failure_diagnosis",
                    "analysis_usable": usable, "authority": "model_hypothesis_only",
                    "required_action": "Implement a corrected candidate using the original requirements, current checked code and actual failure observations. Any preceding diagnosis may be wrong; measured observations remain the facts. Preserve passing behavior and the existing Patch schema, source bindings and permissions."}),)
            context_bytes = len(_canonical_json([asdict(item) for item in additions]).encode("utf-8"))
            if total_observation_bytes + context_bytes > allowed_observation_end:
                additions, context_bytes = (), 0
                response_actions[turn_index]["diagnosis_usable"] = False
                availability["selection"] = "diagnosis_context_budget_not_admitted"
            # 为什么要有它：追加的诊断上下文沿用现有反馈字节计费，装不下时保留调用成本并继续原修订。
            total_observation_bytes += context_bytes
            usage_by_turn[-1]["diagnosis_context_bytes"] = context_bytes
            base_request = replace(base_request, messages=(*base_request.messages, *additions))
            continue

        if response.stop_reason in {"length", "max_tokens", "max_output_tokens", "incomplete", "content_filter", "refusal"}:
            incomplete = response.stop_reason in {"length", "max_tokens", "max_output_tokens", "incomplete"}
            code = "PATCH_AGENT_RESPONSE_INCOMPLETE" if incomplete else "PATCH_AGENT_PROVIDER_STOPPED"
            availability["selection"] = "incomplete_not_executed" if incomplete else "provider_stopped_not_executed"
            submission_trace.append({"turn_index": turn_index, "status": "incomplete" if incomplete else "provider_stopped",
                "response_sha256": hashlib.sha256((response.content if isinstance(response.content, str) else "").encode("utf-8")).hexdigest(),
                "schema_sha256": _canonical_sha256(final_response_format.schema), "stop_reason": response.stop_reason,
                "code_validation_performed": False, "feedback_queued": False})
            return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                termination=_termination(code, "Provider output is incomplete or stopped; it is not a complete submission to format-correct.", turn_index))

        arguments = classify_source_arguments(response.tool_calls, source_tool=PATCH_SOURCE_INSPECTION_TOOL_NAME,
            available_bindings=source_inspector.available_read_binding_ids, inspected_bindings=inspected_binding_ids,
            executed_call_ids=seen_tool_call_ids, rejected_argument_call_ids=rejected_argument_ids,
            reusable_call_bindings=seen_call_bindings if reuse_covered else None,
            retryable_argument_calls=retryable_argument_calls,
            tool_available=any(tool.name == PATCH_SOURCE_INSPECTION_TOOL_NAME for tool in request.tools),
            allow_batches=selected_policy.allow_batched_reads)
        if arguments is not None:
            code, reason = arguments.code, arguments.reason
            try:
                source_inspector.validate_snapshot()
            except PatchAgentLoopError as exc:
                code, reason = exc.code, exc.reason
                arguments = replace(arguments, issues=())
            feedback = ()
            if arguments.issues:
                if argument_used:
                    code, reason = "PATCH_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED", "The shared argument-correction opportunity was already used."
                elif turn_index >= allowed_model_end:
                    code, reason = "PATCH_AGENT_MODEL_TURN_BUDGET_EXHAUSTED", "No model turn remains for argument correction."
                elif tool_rounds_used >= allowed_read_end:
                    code, reason = "PATCH_AGENT_TOOL_ROUND_BUDGET_EXHAUSTED", "Malformed arguments cannot establish a free cached-read identity."
                else:
                    schema = next(tool.parameters for tool in request.tools if tool.name == PATCH_SOURCE_INSPECTION_TOOL_NAME)
                    feedback = argument_feedback(response.tool_calls, arguments, allowed_format=deepcopy(schema),
                        model_turns_used=turn_index, observation_bytes_before=total_observation_bytes)
            feedback_bytes = sum(len(_canonical_json(item.output).encode("utf-8")) for item in feedback)
            if feedback and total_observation_bytes + feedback_bytes > allowed_observation_end:
                code, reason = "PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED", "The complete argument feedback does not fit the original observation budget."
                feedback = ()
            for index, call in enumerate(response.tool_calls):
                entry = _tool_trace(call, turn_index=turn_index,
                    status="rejected_not_executed" if feedback or len(response.tool_calls) > 1 else "rejected",
                    code=code, dispatch_started=False)
                entry.update(read_cost=0, argument_correction=bool(feedback), feedback_queued=bool(feedback),
                    feedback_in_request=False, followup_response_observed=False,
                    feedback_payload_bytes=len(_canonical_json(feedback[index].output).encode("utf-8")) if feedback else 0)
                tool_call_trace.append(entry)
            availability["selection"] = "rejected_not_executed"
            if not feedback:
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                    termination=_termination(code, reason, turn_index))
            # Scope: exactly one correction in the original budgets, retained even before its follow-up succeeds.
            # 为什么要有它：无读取就不扣read；反馈字节与模型回合仍计费，provider失败或续接不能刷新机会。
            argument_used = True
            rejected_argument_ids.update(call.call_id for call in response.tool_calls)
            retryable_argument_calls.update({item.call_id: item.output["validated_retry_arguments"]["binding_id"]
                for item in feedback if "validated_retry_arguments" in item.output})
            total_observation_bytes += feedback_bytes
            retained = _without_loop_states(request)
            history = retained.historical_tool_exchanges
            if retained.tool_results:
                history = (*history, AgentToolExchange(retained.tool_results, retained.provider_state,
                    retained.tool_result_insert_index if retained.tool_result_insert_index is not None else len(retained.messages)))
            content = deepcopy(state_message.content)
            content["tool_argument_correction_used"] = True
            content["budget_state"]["observation_bytes_used"] = total_observation_bytes
            argument_resume_request = replace(retained, messages=(*retained.messages, replace(state_message, content=content)),
                tool_results=feedback, provider_state=response.provider_state,
                tool_result_insert_index=len(retained.messages), historical_tool_exchanges=history)
            base_request = _without_loop_states(argument_resume_request)
            continue

        rejection = reject_nonread_batch(response.tool_calls,
            source_tool_name=PATCH_SOURCE_INSPECTION_TOOL_NAME,
            upstream_tool_name=PATCH_UPSTREAM_REQUEST_TOOL_NAME,
            executed_call_ids=seen_tool_call_ids)
        if rejection is not None:
            feedback_bytes = sum(len(_canonical_json(item.output).encode("utf-8"))
                for item in rejection.results)
            fits = total_observation_bytes + feedback_bytes <= allowed_observation_end
            queued = bool(rejection.results) and fits and turn_index < allowed_model_end
            for call, output in zip(response.tool_calls, rejection.results or (None,) * len(response.tool_calls)):
                entry = _tool_trace(call, turn_index=turn_index, status="rejected_not_executed",
                    code=rejection.code, dispatch_started=False)
                entry.update(read_cost=0, upstream_request_registered=False,
                    feedback_queued=queued, feedback_payload_bytes=(
                        len(_canonical_json(output.output).encode("utf-8")) if queued else 0))
                tool_call_trace.append(entry)
            availability["selection"] = "rejected_not_executed"
            if not queued:
                code = (rejection.code if not rejection.results else
                        "PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED" if not fits else
                        "PATCH_AGENT_MODEL_TURN_BUDGET_EXHAUSTED")
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                    termination=_termination(code, "The rejected batch has no safely deliverable follow-up: "
                        + rejection.reason, turn_index))
            # 为什么要有它：拒绝结果消耗原观察预算，但不伪造源码读取、不预留未执行ID，也不另开模型预算。
            total_observation_bytes += feedback_bytes
            pending_response, pending_tool_results = response, rejection.results
            base_request = _without_loop_states(request)
            continue

        if len(response.tool_calls) > 1 and not selected_policy.allow_batched_reads:
            code = "PATCH_AGENT_PARALLEL_TOOL_CALLS_FORBIDDEN"
            for call in response.tool_calls:
                tool_call_trace.append(
                    _tool_trace(call, turn_index=turn_index, status="rejected", code=code)
                )
            return result_for_segment(
                status="blocked",
                final_content=None,
                provider=provider,
                model=model,
                model_turns_used=turn_index,
                tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes,
                usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace,
                observation_trace=observation_trace,
                termination=_termination(
                    code,
                    "Only one source-inspection tool call is allowed per turn.",
                    turn_index,
                ),
            )
        if encode_excerpts:
            delivered_ids = {result.output.get("observation_id") for result in request.tool_results
                if isinstance(result.output, Mapping)}
            delivered_ids.update(result.output.get("observation_id")
                for exchange in request.historical_tool_exchanges for result in exchange.tool_results
                if isinstance(result.output, Mapping))
            for entry in observation_trace:
                if entry["observation_id"] in delivered_ids and entry.get("payload_admitted"):
                    entry["admitted_to_model"] = True
                    entry.setdefault("consumed_in_model_turn", turn_index)

        if len(response.tool_calls) > 1 or (reuse_covered and len(response.tool_calls) == 1
                and response.tool_calls[0].name == PATCH_SOURCE_INSPECTION_TOOL_NAME):
            calls = response.tool_calls
            code, reason = _validate_read_batch(calls, source_inspector,
                inspected_binding_ids=inspected_binding_ids, seen_tool_call_ids=seen_tool_call_ids,
                reusable_call_bindings=seen_call_bindings if reuse_covered else None)
            if code is None and not any(tool.name == PATCH_SOURCE_INSPECTION_TOOL_NAME for tool in request.tools):
                code, reason = "PATCH_AGENT_TOOL_FORBIDDEN", "Source inspection is disabled for this turn."
            if code is not None:
                for call in calls:
                    tool_call_trace.append(_tool_trace(call, turn_index=turn_index,
                        status="rejected_not_executed", code=code, dispatch_started=False))
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                    termination=_termination(code, reason, turn_index))
            remaining_reads = allowed_read_end - tool_rounds_used
            planned_reads = None
            if reuse_covered:
                try:
                    with trace_operation("source.prepare", attributes={
                        "turn_index": turn_index, "tool_call_ids": [call.call_id for call in calls],
                        "provider_response_id": response.request_id, "preparation_only": True,
                        "tool_dispatch_performed": False,
                    }) as preparation:
                        planned_reads = _plan_source_reads(source_inspector,
                            [call.arguments["binding_id"] for call in calls], full_excerpts, observation_sequence)
                        preparation.update(prepared_binding_count=len(planned_reads),
                            prepared_observation_bytes=sum(len(_canonical_json(row[1]).encode("utf-8")) for row in planned_reads),
                            physical_source_preparation_performed=True)
                except PatchAgentLoopError as exc:
                    for call in calls:
                        tool_call_trace.append(_tool_trace(call, turn_index=turn_index,
                            status="rejected_not_executed", code=exc.code, dispatch_started=False))
                    return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                        model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                        total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                        tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                        termination=_termination(exc.code, exc.reason, turn_index))
            requested_cost = sum(row[2] for row in planned_reads) if planned_reads is not None else len(calls)
            if reuse_covered:
                repeat_identity = _canonical_sha256(sorted(call.arguments["binding_id"] for call in calls))
                invalid_repeat = any(call.arguments["binding_id"] in inspected_binding_ids and row[2] != 0
                    for call, row in zip(calls, planned_reads))
                if invalid_repeat or (requested_cost == 0 and repeat_identity == last_reused_only_batch):
                    reason = ("Previously inspected content is not eligible for complete visible-source reuse."
                        if invalid_repeat else "The same reference-only request repeated without any intervening new source observation.")
                    for call in calls:
                        tool_call_trace.append(_tool_trace(call, turn_index=turn_index,
                            status="rejected_not_executed", code="PATCH_AGENT_REPEATED_TOOL_CALL", dispatch_started=False))
                    return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                        model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                        total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                        tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                        termination=_termination("PATCH_AGENT_REPEATED_TOOL_CALL", reason, turn_index))
            if requested_cost > remaining_reads:
                code = "PATCH_AGENT_TOOL_BATCH_BUDGET_EXCEEDED"
                batch_identity = _canonical_sha256(sorted(call.arguments["binding_id"] for call in calls))
                if rejected_read_batches:
                    code = ("PATCH_AGENT_REPEATED_TOOL_BATCH" if batch_identity in rejected_read_batches
                        else "PATCH_AGENT_TOOL_BATCH_BUDGET_RETRY_EXHAUSTED")
                    for call in calls:
                        tool_call_trace.append(_tool_trace(call, turn_index=turn_index,
                            status="rejected_not_executed", code=code, dispatch_started=False))
                    availability["selection"] = "rejected_not_executed"
                    return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                        model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                        total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                        tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                        termination=_termination(code, "One bounded batch-budget correction opportunity was already used.", turn_index))
                rejected_read_batches.add(batch_identity)
                results = tuple(AgentToolResult(call_id=call.call_id, name=call.name, output={
                    "status": "not_executed", "error_code": code, "reason": "budget_exceeded",
                    "binding_id": call.arguments["binding_id"], "requested_read_count": len(calls),
                    **({"requested_new_source_windows": requested_cost, "reused_source_observations": len(calls) - requested_cost,
                        "read_accounting_version": "new_source_window_v2"} if reuse_covered else {}),
                    "remaining_read_count": remaining_reads,
                    "total_remaining_read_count": None if metered else selected_policy.max_tool_rounds - tool_rounds_used,
                    "reserved_read_count": reserve_tool_reads,
                    "required_action": ("Select a batch whose new source windows fit the remaining read count; visible-source references cost zero."
                        if reuse_covered else "Select no more than the remaining read count, or finish from existing context."),
                    "source_snapshot_sha256": source_inspector.source_snapshot_sha256,
                }) for call in calls)
                for call, result in zip(calls, results):
                    feedback_bytes = len(_canonical_json(result.output).encode("utf-8"))
                    total_observation_bytes += feedback_bytes
                    trace = _tool_trace(call, turn_index=turn_index, status="rejected_not_executed",
                        code=code, dispatch_started=False)
                    trace["feedback_payload_bytes"] = feedback_bytes
                    tool_call_trace.append(trace)
                availability["selection"] = "rejected_not_executed"
                if total_observation_bytes > allowed_observation_end:
                    return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                        model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                        total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                        tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                        termination=_termination("PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED",
                            "The bounded batch rejection feedback exceeds the observation budget.", turn_index))
                # A denied batch did not execute. Provider-derived ids may recur when
                # the model selects a smaller subset; only executed ids are reserved.
                pending_response, pending_tool_results = response, results
                base_request = _without_loop_states(request)
                continue
            if planned_reads is not None and total_observation_bytes + sum(
                    len(_canonical_json(row[1]).encode("utf-8")) for row in planned_reads) > allowed_observation_end:
                # The entire read-only exchange is checked before dispatch. A
                # budget denial still answers every tool call; it does not
                # pretend that planned bodies were delivered or refund history.
                code = "PATCH_AGENT_TOOL_RESULT_BUDGET_EXCEEDED"
                results = tuple(AgentToolResult(call_id=call.call_id, name=call.name, output={
                    "status": "not_executed", "error_code": code, "binding_id": call.arguments["binding_id"],
                    "required_action": "Finish using visible context or request upstream. Repeating this fixed window cannot shrink it.",
                }) for call in calls)
                feedback_bytes = sum(len(_canonical_json(item.output).encode("utf-8")) for item in results)
                feedback_fits = total_observation_bytes + feedback_bytes <= allowed_observation_end
                for call, output in zip(calls, results):
                    entry = _tool_trace(call, turn_index=turn_index, status="rejected_not_executed",
                        code=code, dispatch_started=False)
                    entry.update(feedback_payload_bytes=(len(_canonical_json(output.output).encode("utf-8"))
                        if feedback_fits else 0), feedback_queued=feedback_fits, read_cost=0)
                    tool_call_trace.append(entry)
                availability["selection"] = "rejected_not_executed"
                if not feedback_fits:
                    return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                        model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                        total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                        tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                        termination=_termination("PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED",
                            "Neither this complete source batch nor its per-call budget results fit the remaining allowance.", turn_index))
                total_observation_bytes += feedback_bytes
                pending_response, pending_tool_results = response, results
                base_request = _without_loop_states(request)
                force_final = True
                continue
            batch_results: list[AgentToolResult] = []
            for index, call in enumerate(calls):
                read_cost = planned_reads[index][2] if planned_reads is not None else 1
                tool_rounds_used += read_cost
                observation_sequence += 1
                seen_tool_call_ids.add(call.call_id)
                seen_call_bindings[call.call_id] = call.arguments["binding_id"]
                try:
                    with trace_operation("source.inspect", kind="tool", attributes={
                        **_source_trace_attributes(call, response, turn_index), "read_cost": read_cost,
                        "reused": read_cost == 0, "prepared_before_dispatch": planned_reads is not None,
                    }) as inspection:
                        if planned_reads is not None:
                            source_inspector.validate_snapshot()
                            observation, encoded, _ = planned_reads[index]
                            if "excerpt_ref" not in encoded and (key := _local_excerpt_key(encoded)) is not None:
                                full_excerpts.setdefault(key, encoded)
                            elif "excerpt_parts" in encoded:
                                full_excerpts.setdefault(_local_excerpt_key(observation.model_view()), encoded)
                        else:
                            observation = source_inspector.inspect(binding_id=call.arguments["binding_id"],
                                observation_id=f"patch-observation-{tool_rounds_used:03d}")
                            encoded = (_encode_source_excerpt(observation, full_excerpts)
                                if encode_excerpts else observation.model_view())
                        total_observation_bytes += len(_canonical_json(encoded).encode("utf-8"))
                        admitted = total_observation_bytes <= allowed_observation_end
                        inspection.update(**_source_result_trace_attributes(observation, encoded, admitted))
                    observation_trace.append(_observation_trace(observation, turn_index=turn_index, admitted_to_model=admitted,
                        **({"encoded_payload": encoded} if encode_excerpts else {}),
                        **({"read_cost": read_cost} if reuse_covered else {})))
                    if not admitted:
                        raise PatchAgentLoopError("PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED", stage="tool_execution",
                            reason="The cumulative source-observation byte budget was exhausted.")
                except PatchAgentLoopError as exc:
                    availability["selection"] = "execution_failed"
                    tool_call_trace.append(_tool_trace(call, turn_index=turn_index, status="rejected",
                        code=exc.code, dispatch_started=True))
                    if reuse_covered:
                        tool_call_trace[-1].update(read_cost=read_cost,
                            observation_origin="prior_visible_source_reuse" if read_cost == 0 else "current_source_read")
                    for remaining_call in calls[index + 1:]:
                        tool_call_trace.append(_tool_trace(remaining_call, turn_index=turn_index,
                            status="rejected_not_executed", code=exc.code, dispatch_started=False))
                    return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                        model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                        total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                        tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                        termination=_termination(exc.code, exc.reason, turn_index))
                inspected_binding_ids.add(call.arguments["binding_id"])
                observations.append(observation)
                batch_results.append(AgentToolResult(call_id=call.call_id, name=call.name, output=encoded))
                tool_call_trace.append(_tool_trace(call, turn_index=turn_index, status="accepted", dispatch_started=True))
                if reuse_covered:
                    tool_call_trace[-1].update(read_cost=read_cost,
                        observation_origin="prior_visible_source_reuse" if read_cost == 0 else "current_source_read",
                        observation_id=observation.observation_id,
                        **({"excerpt_ref": deepcopy(encoded["excerpt_ref"])} if "excerpt_ref" in encoded else {}))
            if reuse_covered:
                last_reused_only_batch = repeat_identity if requested_cost == 0 else None
            pending_response, pending_tool_results = response, tuple(batch_results)
            base_request = _without_loop_states(request)
            continue

        if response.tool_calls:
            call = response.tool_calls[0]
            if call.name == PATCH_UPSTREAM_REQUEST_TOOL_NAME:
                if isinstance(call.call_id, str) and call.call_id in seen_tool_call_ids:
                    code = "PATCH_AGENT_TOOL_CALL_ID_REUSED"
                    tool_call_trace.append(_tool_trace(call, turn_index=turn_index,
                        status="rejected_not_executed", code=code, dispatch_started=False))
                    return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                        model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                        total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                        tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                        termination=_termination(code, "An executed tool call ID cannot request a new upstream effect.", turn_index))
                with trace_operation("patch.upstream_request", kind="tool", attributes={
                    "turn_index": turn_index, "tool_call_id": call.call_id,
                    "provider_response_id": response.request_id, "upstream_execution_performed": False,
                }) as upstream_observation:
                    code, upstream = _validate_upstream_request(call, source_inspector, selected_policy,
                        tool_available=any(tool.name == PATCH_UPSTREAM_REQUEST_TOOL_NAME for tool in request.tools))
                    upstream_observation.update(request_registered=upstream is not None, result_code=code)
                tool_call_trace.append(_tool_trace(call, turn_index=turn_index,
                    status="accepted" if upstream is not None else "rejected", code=code,
                    dispatch_started=False))
                termination = _termination(code, "The model requested upstream assessment; no route or authority was granted.", turn_index)
                if upstream is not None:
                    termination["upstream_request"] = upstream
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace, termination=termination,
                    final_request=request)
            code: str | None = None
            reason = ""
            if type(call.call_id) is not str or not call.call_id:
                code = "PATCH_AGENT_TOOL_CALL_ID_INVALID"
                reason = "The provider tool call id is missing."
            elif call.name != PATCH_SOURCE_INSPECTION_TOOL_NAME:
                code = "PATCH_AGENT_UNKNOWN_TOOL"
                reason = "The model requested an unregistered Patch tool."
            elif tool_rounds_used >= allowed_read_end:
                code = "PATCH_AGENT_TOOL_ROUND_BUDGET_EXHAUSTED"
                reason = "The binding-read budget available to this segment was exhausted."
            elif total_observation_bytes >= allowed_observation_end:
                code = "PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED"
                reason = "No observation bytes remain available to this segment."
            elif turn_index >= allowed_model_end:
                code = "PATCH_AGENT_MODEL_TURN_BUDGET_EXHAUSTED"
                reason = "No model turn remains to consume another tool observation."
            elif (
                set(call.arguments) != {"binding_id"}
                or type(call.arguments.get("binding_id")) is not str
                or not call.arguments["binding_id"]
            ):
                code = "PATCH_AGENT_TOOL_ARGUMENTS_INVALID"
                reason = "The source-inspection tool accepts only one binding_id."
            else:
                binding_id = str(call.arguments["binding_id"])
                if binding_id in inspected_binding_ids:
                    code = "PATCH_AGENT_REPEATED_TOOL_CALL"
                    reason = "The same Plan binding cannot be inspected twice."
                elif call.call_id in seen_tool_call_ids:
                    code = "PATCH_AGENT_TOOL_CALL_ID_REUSED"
                    reason = "The provider reused a tool call id."
                elif binding_id not in source_inspector.available_read_binding_ids:
                    code = "PATCH_AGENT_BINDING_UNKNOWN"
                    reason = "The requested binding id is not selected by the Plan."
                elif not any(tool.name == PATCH_SOURCE_INSPECTION_TOOL_NAME for tool in request.tools):
                    code = "PATCH_AGENT_TOOL_FORBIDDEN"
                    reason = "Source inspection is disabled for this turn."
            if code is not None:
                tool_call_trace.append(
                    _tool_trace(call, turn_index=turn_index, status="rejected", code=code)
                )
                return result_for_segment(
                    status="blocked",
                    final_content=None,
                    provider=provider,
                    model=model,
                    model_turns_used=turn_index,
                    tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes,
                    usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace,
                    observation_trace=observation_trace,
                    termination=_termination(code, reason, turn_index),
                )

            binding_id = str(call.arguments["binding_id"])
            seen_tool_call_ids.add(call.call_id)
            tool_rounds_used += 1
            try:
                with trace_operation("source.inspect", kind="tool", attributes={
                    **_source_trace_attributes(call, response, turn_index), "read_cost": 1,
                    "reused": False, "prepared_before_dispatch": False,
                }) as inspection:
                    observation = source_inspector.inspect(
                        binding_id=binding_id,
                        observation_id=f"patch-observation-{tool_rounds_used:03d}",
                    )
                    encoded = (_encode_source_excerpt(observation, full_excerpts)
                        if selected_policy.deduplicate_source_excerpts else observation.model_view())
                    total_observation_bytes += len(_canonical_json(encoded).encode("utf-8"))
                    admitted_to_model = total_observation_bytes <= allowed_observation_end
                    inspection.update(**_source_result_trace_attributes(observation, encoded, admitted_to_model))
            except PatchAgentLoopError as exc:
                availability["selection"] = "execution_failed"
                tool_call_trace.append(
                    _tool_trace(
                        call,
                        turn_index=turn_index,
                        status="rejected",
                        code=exc.code,
                        dispatch_started=True,
                    )
                )
                return result_for_segment(
                    status="blocked",
                    final_content=None,
                    provider=provider,
                    model=model,
                    model_turns_used=turn_index,
                    tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes,
                    usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace,
                    observation_trace=observation_trace,
                    termination=_termination(exc.code, exc.reason, turn_index),
                )
            observation_trace.append(
                _observation_trace(
                    observation,
                    turn_index=turn_index,
                    admitted_to_model=admitted_to_model,
                    **({"encoded_payload": encoded} if selected_policy.deduplicate_source_excerpts else {}),
                )
            )
            if not admitted_to_model:
                code = "PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED"
                tool_call_trace.append(
                    _tool_trace(
                        call,
                        turn_index=turn_index,
                        status="rejected",
                        code=code,
                        dispatch_started=True,
                    )
                )
                return result_for_segment(
                    status="blocked",
                    final_content=None,
                    provider=provider,
                    model=model,
                    model_turns_used=turn_index,
                    tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes,
                    usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace,
                    observation_trace=observation_trace,
                    termination=_termination(
                        code,
                        "The cumulative source-observation byte budget was exhausted.",
                        turn_index,
                    ),
                )
            tool_call_trace.append(
                _tool_trace(
                    call,
                    turn_index=turn_index,
                    status="accepted",
                    dispatch_started=True,
                )
            )
            inspected_binding_ids.add(binding_id)
            observations.append(observation)
            pending_response = response
            if selected_policy.deduplicate_source_excerpts:
                pending_tool_results = (AgentToolResult(call_id=call.call_id, name=call.name, output=encoded),)
            else:
                pending_observation = observation
            base_request = _without_loop_states(request)
            continue

        content = response.content
        if selected_policy.require_source_inspection and tool_rounds_used == 0:
            return result_for_segment(
                status="blocked",
                final_content=None,
                provider=provider,
                model=model,
                model_turns_used=turn_index,
                tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes,
                usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace,
                observation_trace=observation_trace,
                termination=_termination(
                    "PATCH_AGENT_SOURCE_INSPECTION_REQUIRED",
                    "The first Patch Agent turn must inspect one Plan binding.",
                    turn_index,
                ),
            )
        if content is not None and not isinstance(content, str):
            return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                termination=_termination("PATCH_AGENT_RESPONSE_INVALID", "Normalized response content must be text or absent.", turn_index))
        raw_content = content or ""
        submission = {"turn_index": turn_index, "response_sha256": hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
            "schema_sha256": _canonical_sha256(final_response_format.schema), "status": "rejected",
            "content_present": content is not None, "code_validation_performed": False,
            "feedback_queued": False, "feedback_in_request": False, "followup_response_observed": False}
        submission_trace.append(submission)

        def hard_submission_failure(error: Exception) -> PatchAgentLoopResult:
            submission.update(status="validation_failed", failure_class=type(error).__name__,
                diagnostics=deepcopy(getattr(error, "diagnostics", {})))
            return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                termination={**_termination(getattr(error, "code", type(error).__name__), str(error), turn_index),
                    "failure_class": type(error).__name__, "diagnostics": deepcopy(submission["diagnostics"])})

        try:
            with trace_operation("patch.submission", attributes={
                "turn_index": turn_index, "provider_response_id": response.request_id,
                "response_content_sha256": submission["response_sha256"],
                "schema_sha256": submission["schema_sha256"], "validation_scope": "strict_json_and_submission_schema",
                "behavior_validation_performed": False, "submission_admitted": False,
            }) as submission_observation:
                validate_patch_submission(raw_content, final_response_format.schema)
                if submission_validator is not None:
                    # Identity/authority failures are not model-correctable. Only
                    # the explicitly typed, read-only submission check may feed back.
                    source_inspector.validate_snapshot()
                    submission["code_validation_performed"] = True
                    submission_observation.update(validation_scope="submission_and_bound_patch_static_checks")
                    try:
                        submission_validator(response)
                    except PatchSubmissionValidationError:
                        raise
                    except Exception:
                        submission["status"] = "validation_failed"
                        raise
                submission_observation.update(submission_admitted=True)
        except ValueError as exc:
            if submission["code_validation_performed"]:
                return hard_submission_failure(exc)
            submission["status"] = "contract_invalid"
            return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                termination=_termination("PATCH_AGENT_SUBMISSION_CONTRACT_INVALID", str(exc), turn_index))
        except PatchGenerationError as exc:
            if submission["code_validation_performed"] and not isinstance(exc, PatchSubmissionValidationError):
                return hard_submission_failure(exc)
            if isinstance(exc, PatchSubmissionValidationError):
                # A complete, bound Patch with a repairable text defect has
                # reached recovery. Spend its existing reserve, never new credit.
                submission["released_recovery_reservation"] = {
                    "model_turns": 0 if metered else selected_policy.max_total_model_turns - allowed_model_end,
                    "tool_reads": 0 if metered else selected_policy.max_tool_rounds - allowed_read_end,
                    "observation_bytes": 0 if metered else selected_policy.max_observation_bytes - allowed_observation_end,
                }
                allowed_model_end = inf if metered else selected_policy.max_total_model_turns
                allowed_read_end = inf if metered else selected_policy.max_tool_rounds
                allowed_observation_end = inf if metered else selected_policy.max_observation_bytes
            submission["diagnostics"] = deepcopy(exc.diagnostics)
            noop_view = project_whole_noop_response(raw_content, exc.diagnostics)
            noop_identity = noop_code_identity(exc.diagnostics) if noop_view is not None else None
            if noop_identity is not None:
                submission["effective_code_sha256"] = noop_identity
            if submission["response_sha256"] in rejected_submissions or (noop_identity is not None and noop_identity in rejected_noop_codes):
                stop_code, reason = "PATCH_AGENT_REPEATED_INVALID_SUBMISSION", "The same unadmitted submission was returned after submission feedback."
            elif turn_index >= allowed_model_end:
                stop_code, reason = "PATCH_AGENT_SUBMISSION_CORRECTION_BUDGET_EXHAUSTED", "No model turn remains in this segment to correct the submission; reserved behavior turns are unchanged."
            else:
                stop_code, reason = None, ""
            if stop_code:
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                    termination=_termination(stop_code, reason, turn_index))
            feedback = {"message_type": "patch_submission_feedback", "response_sha256": submission["response_sha256"],
                "schema_sha256": submission["schema_sha256"], "diagnostics": deepcopy(exc.diagnostics),
                "error": str(exc), "response_schema": deepcopy(final_response_format.schema),
                "code_validation_performed": submission["code_validation_performed"],
                "required_action": "The preceding output was not admitted as a Patch. Use only the actions currently listed. "
                    "If submitting, return one complete JSON object matching response_schema, without surrounding explanation. "
                    "Correct the reported submission defects and preserve already-correct behavior. "
                    "Keep the original requirements, bindings and unchanged source baseline; no code has been accepted or applied."}
            original_output = (AgentMessage("assistant", noop_view or raw_content),) if raw_content else ()
            additions = (*original_output, AgentMessage("user", feedback))
            feedback_bytes = len(_canonical_json([{"role": item.role, "content": item.content} for item in additions]).encode("utf-8"))
            if total_observation_bytes + feedback_bytes > allowed_observation_end:
                return result_for_segment(status="blocked", final_content=None, provider=provider, model=model,
                    model_turns_used=turn_index, tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes, usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace, observation_trace=observation_trace,
                    termination=_termination("PATCH_AGENT_OBSERVATION_BUDGET_EXHAUSTED", "Submission feedback does not fit the existing observation allowance.", turn_index))
            total_observation_bytes += feedback_bytes
            submission.update(feedback_queued=True, feedback_payload_bytes=feedback_bytes)
            if noop_identity is not None:
                rejected_noop_codes.add(noop_identity)
                submission["feedback_code_view"] = "whole_noop_code_omitted"
            rejected_submissions.add(submission["response_sha256"])
            base_request = _without_loop_states(request)
            base_request = replace(base_request, messages=(*base_request.messages, *additions))
            format_correction_pending = True
            last_reused_only_batch = None
            continue
        except Exception as exc:
            # Preserve already-spent calls/reads even when identity, environment
            # or authority rejects submission. This remains a terminal failure.
            return hard_submission_failure(exc)
        else:
            submission["status"] = "accepted"
            response_actions[turn_index]["submitted_patch"] = True
            try:
                source_inspector.validate_snapshot()
            except PatchAgentLoopError as exc:
                return result_for_segment(
                    status="blocked",
                    final_content=None,
                    provider=provider,
                    model=model,
                    model_turns_used=turn_index,
                    tool_rounds_used=tool_rounds_used,
                    total_observation_bytes=total_observation_bytes,
                    usage_by_turn=usage_by_turn,
                    tool_call_trace=tool_call_trace,
                    observation_trace=observation_trace,
                    termination=_termination(exc.code, exc.reason, turn_index),
                )
            return result_for_segment(
                status="completed",
                final_content=content,
                final_request=request,
                provider=provider,
                model=model,
                model_turns_used=turn_index,
                tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes,
                usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace,
                observation_trace=observation_trace,
                termination=_termination(
                    "PATCH_AGENT_MODEL_FINISHED",
                    "The model returned final Patch content.",
                    turn_index,
                    status="completed",
                ),
            )

        if turn_index >= allowed_model_end:
            return result_for_segment(
                status="blocked",
                final_content=None,
                provider=provider,
                model=model,
                model_turns_used=turn_index,
                tool_rounds_used=tool_rounds_used,
                total_observation_bytes=total_observation_bytes,
                usage_by_turn=usage_by_turn,
                tool_call_trace=tool_call_trace,
                observation_trace=observation_trace,
                termination=_termination(
                    "PATCH_AGENT_MODEL_TURN_BUDGET_EXHAUSTED",
                    "The model-turn budget ended without final Patch content.",
                    turn_index,
                ),
            )
        force_final = True

    raise AssertionError("Patch Agent loop exceeded its fixed model-turn bound.")


def _validate_read_batch(calls: tuple[AgentToolCall, ...], inspector: PlanBoundSourceInspector,
    *, inspected_binding_ids: set[str], seen_tool_call_ids: set[str],
    reusable_call_bindings: Mapping[str, str] | None = None) -> tuple[str | None, str]:
    if any(call.name == PATCH_UPSTREAM_REQUEST_TOOL_NAME for call in calls):
        return "PATCH_AGENT_MIXED_TOOL_BATCH_FORBIDDEN", "Upstream requests must be separate from source reads."
    if any(call.name != PATCH_SOURCE_INSPECTION_TOOL_NAME for call in calls):
        return "PATCH_AGENT_UNKNOWN_TOOL", "The batch contains an unregistered Patch tool."
    selected: set[str] = set()
    call_ids: set[str] = set()
    for call in calls:
        if type(call.call_id) is not str or not call.call_id:
            return "PATCH_AGENT_TOOL_CALL_ID_INVALID", "A provider tool call id is missing."
        if call.call_id in call_ids or (call.call_id in seen_tool_call_ids
                and (reusable_call_bindings is None or reusable_call_bindings.get(call.call_id) != call.arguments.get("binding_id"))):
            return "PATCH_AGENT_TOOL_CALL_ID_REUSED", "The provider reused a tool call id."
        if (set(call.arguments) != {"binding_id"} or type(call.arguments.get("binding_id")) is not str
            or not call.arguments["binding_id"]):
            return "PATCH_AGENT_TOOL_ARGUMENTS_INVALID", "Each source read requires exactly one binding_id."
        binding_id = call.arguments["binding_id"]
        if binding_id in selected or (binding_id in inspected_binding_ids and reusable_call_bindings is None):
            return "PATCH_AGENT_REPEATED_TOOL_CALL", "The same Plan binding cannot be inspected twice."
        if binding_id not in inspector.available_read_binding_ids:
            return "PATCH_AGENT_BINDING_UNKNOWN", "A batch binding is outside the admitted read scope."
        call_ids.add(call.call_id)
        selected.add(binding_id)
    return None, ""


def _source_tool_definition(
    inspector: PlanBoundSourceInspector,
    *,
    binding_ids: tuple[str, ...] | None = None,
) -> AgentToolDefinition:
    allowed_binding_ids = inspector.available_read_binding_ids if binding_ids is None else binding_ids
    return AgentToolDefinition(
        name=PATCH_SOURCE_INSPECTION_TOOL_NAME,
        description=(
            "Read registered content; no Evidence/Coverage or Plan reassessment. "
            "Read validated Plan-bound B0 (no generated edits) via listed binding_id; no paths. "
            "Checked code: assistant Patch bound by latest failure feedback. "
            "Supporting reads grant no edits. API bindings return full saved "
            "Evidence, not live inspection. Sequential calls. "
            + ("New windows cost 1 read; visible refs cost 0 but consume bytes. "
                if inspector.policy.reuse_covered_source_observations else
                "Each dispatched binding read costs one read, including in a batch. ")
            + ("excerpt_ref reuses full requested text verbatim from a prior observation or earlier same-batch item. "
               if inspector.policy.deduplicate_source_excerpts or inspector.policy.reuse_covered_source_observations else "")
            + ("source_excerpt_span gives prior body lines containing the full requested excerpt_span. "
               if inspector.policy.reuse_covered_source_observations else "")
            + ("excerpt_parts joins literal strings and referenced source line spans in order. "
               if inspector.policy.reuse_covered_source_observations else "")
            + "Use current binding target_span; keep full text."
        ),
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["binding_id"],
            "properties": {
                "binding_id": {
                    "type": "string",
                    "enum": list(allowed_binding_ids),
                }
            },
        },
        strict=True,
    )


def _loop_state_message(
    *,
    policy: PatchAgentLoopPolicy,
    source_inspector: PlanBoundSourceInspector,
    prior_observations: list[PatchSourceObservation | Mapping[str, object]],
    model_turns_used: int,
    tool_rounds_used: int,
    force_final: bool,
    final_response_format: AgentResponseFormat,
    reserve_model_turns: int = 0,
    reserve_tool_reads: int = 0,
    total_observation_bytes: int = 0,
    reserve_observation_bytes: int = 0,
    upstream_effects: Mapping[str, str] | None = None,
    tool_argument_correction_used: bool = False,
) -> AgentMessage:
    return AgentMessage(
        role="user",
        content={
            "message_type": "patch_agent_loop_state",
            "tool_argument_correction_used": tool_argument_correction_used,
            "allowed_binding_ids": list(source_inspector.available_binding_ids),
            "allowed_read_binding_ids": list(source_inspector.available_read_binding_ids),
            "source_snapshot_sha256": source_inspector.source_snapshot_sha256,
            "read_source_snapshot_sha256": source_inspector.read_source_snapshot_sha256,
            **({"source_read_semantics": (
                "Tools read B0; checked changes are in the assistant Patch referenced by feedback. "
                "API excerpts are saved Evidence."
            )} if policy.reuse_covered_source_observations else {}),
            **({"upstream_effects": normalize_upstream_effects(upstream_effects)} if policy.allow_upstream_request else {}),
            "read_execution": {"allow_batched_reads": policy.allow_batched_reads,
                "execution": "sequential", "count_unit": "one_new_complete_source_window" if policy.reuse_covered_source_observations else "one_dispatched_binding_read",
                "batch_must_fit_remaining_reads": policy.model_call_budget_policy != "meter_only",
                **({"accounting_version": "new_source_window_v2", "visible_source_reference_read_cost": 0}
                    if policy.reuse_covered_source_observations else {})},
            "prior_observations": [
                observation.model_view() if isinstance(observation, PatchSourceObservation) else deepcopy(dict(observation))
                for observation in prior_observations
            ],
            "budget_state": {
                "policy": policy.model_call_budget_policy,
                "model_turns_used": model_turns_used,
                "total_remaining_model_turns": None if policy.model_call_budget_policy == "meter_only" else policy.max_total_model_turns - model_turns_used,
                "remaining_model_turns": None if policy.model_call_budget_policy == "meter_only" else max(0, policy.max_total_model_turns
                - model_turns_used - reserve_model_turns),
                "reserved_model_turns": reserve_model_turns,
                "tool_rounds_used": tool_rounds_used,
                "total_remaining_tool_rounds": None if policy.model_call_budget_policy == "meter_only" else policy.max_tool_rounds - tool_rounds_used,
                "remaining_tool_rounds": None if policy.model_call_budget_policy == "meter_only" else max(0, policy.max_tool_rounds - tool_rounds_used - reserve_tool_reads),
                "reserved_tool_reads": reserve_tool_reads,
                "observation_bytes_used": total_observation_bytes,
                "total_remaining_observation_bytes": None if policy.model_call_budget_policy == "meter_only" else max(0, policy.max_observation_bytes - total_observation_bytes),
                "remaining_observation_bytes": None if policy.model_call_budget_policy == "meter_only" else max(0, policy.max_observation_bytes
                - total_observation_bytes - reserve_observation_bytes),
                "reserved_observation_bytes": reserve_observation_bytes,
            },
            "required_next_action": (
                "finish_or_request_upstream" if force_final and policy.allow_upstream_request
                else "finish" if force_final else "inspect_or_finish_or_request_upstream"
                if policy.allow_upstream_request else "inspect_or_finish"
            ),
            "final_response_contract": {
                "name": final_response_format.name,
                "description": final_response_format.description,
                "schema_sha256": _canonical_sha256(
                    final_response_format.schema
                ),
            },
        },
    )


def _is_loop_state(message: AgentMessage) -> bool:
    return (message.role == "user" and isinstance(message.content, Mapping)
        and message.content.get("message_type") == "patch_agent_loop_state")


def _without_loop_states(request: AgentTurnRequest) -> AgentTurnRequest:
    """Replace dynamic loop state while preserving native exchange message boundaries."""

    kept_indexes = [index for index, message in enumerate(request.messages) if not _is_loop_state(message)]

    def remap(index: int | None) -> int | None:
        if index is None:
            return None
        if type(index) is not int or not 0 <= index <= len(request.messages):
            raise PatchAgentLoopError("PATCH_AGENT_LOOP_INPUT_INVALID", stage="loop_admission",
                reason="A historical tool exchange has an invalid insertion boundary.")
        return sum(old_index < index for old_index in kept_indexes)

    return replace(request, messages=tuple(request.messages[index] for index in kept_indexes),
        tool_result_insert_index=remap(request.tool_result_insert_index),
        historical_tool_exchanges=tuple(replace(exchange, insert_index=remap(exchange.insert_index))
            for exchange in request.historical_tool_exchanges))


def _upstream_tool_definition(inspector: PlanBoundSourceInspector, *,
        upstream_effects: Mapping[str, str] | None = None) -> AgentToolDefinition:
    effects = normalize_upstream_effects(upstream_effects)
    descriptions = {
        "record_and_stop": "save request and stop this run; no automatic reassessment",
        "assess_then_stop": "Graph checks and attempts reassessment, then stops at the Plan",
        "assess_and_continue": "Graph checks and may reassess then resume Patch within remaining budget",
        "provider_unavailable_stop": "provider unavailable; save request and stop without reassessment",
    }
    return AgentToolDefinition(name=PATCH_UPSTREAM_REQUEST_TOOL_NAME,
        description=("Request reassessment, not a source read: Evidence support/conflicts/coverage or Planning targets/scope. "
            "Explain which judgment needs review; known refs may justify reassessment. No edit authority is granted. "
            f"Current evidence effect: {descriptions[effects['evidence']]}. "
            f"Current planning effect: {descriptions[effects['planning']]}.") ,
        parameters={"type": "object", "additionalProperties": False,
            "required": ["required_owner", "reason", "source_refs"], "properties": {
                "required_owner": {"type": "string", "enum": ["evidence", "planning"]},
                "reason": {"type": "string", "minLength": 1, "maxLength": 2000},
                "source_refs": {"type": "array", "minItems": 1, "uniqueItems": True,
                    "items": {"type": "string", "enum": [f"binding://{binding}" for binding in inspector.available_read_binding_ids]}}
            }})


def _validate_upstream_request(call: AgentToolCall, inspector: PlanBoundSourceInspector,
    policy: PatchAgentLoopPolicy, *, tool_available: bool) -> tuple[str, dict[str, object] | None]:
    if not policy.allow_upstream_request or not tool_available:
        return "PATCH_AGENT_TOOL_FORBIDDEN", None
    arguments = call.arguments
    refs = arguments.get("source_refs")
    reason = arguments.get("reason")
    if (not isinstance(call.call_id, str) or not call.call_id
        or set(arguments) != {"required_owner", "reason", "source_refs"}
        or arguments.get("required_owner") not in ("evidence", "planning")
        or not isinstance(reason, str) or not reason.strip() or len(reason) > 2000
        or not isinstance(refs, list) or not refs
        or any(not isinstance(ref, str) for ref in refs)
        or len(refs) != len(set(refs))
        or not set(refs) <= {f"binding://{binding}" for binding in inspector.available_read_binding_ids}):
        return "PATCH_AGENT_UPSTREAM_REQUEST_INVALID", None
    try:
        inspector.validate_snapshot()
    except PatchAgentLoopError as exc:
        return exc.code, None
    return "PATCH_AGENT_UPSTREAM_REQUESTED", {**deepcopy(arguments),
        "source_snapshot_sha256": inspector.source_snapshot_sha256,
        "read_source_snapshot_sha256": inspector.read_source_snapshot_sha256,
        "assessment_required": True}


def _usage_record(turn_index: int, response: AgentTurnResponse) -> dict[str, object]:
    return {
        "turn_index": turn_index,
        "provider": response.provider,
        "model": response.model,
        "request_id": response.request_id,
        "stop_reason": response.stop_reason,
        "usage": response.usage.to_dict(),
        **(
            {"provider_diagnostic": deepcopy(response.provider_diagnostic)}
            if response.provider_diagnostic is not None else {}
        ),
    }


def _source_trace_attributes(call, response, turn_index):
    return {"turn_index": turn_index, "tool_call_id": call.call_id,
            "binding_id": call.arguments["binding_id"], "tool_dispatch_performed": True,
            "provider_response_id": response.request_id,
            "response_content_sha256": hashlib.sha256(
                (response.content if isinstance(response.content, str) else "").encode("utf-8")).hexdigest()}


def _source_result_trace_attributes(observation, encoded, queued):
    return {"source_observation_id": observation.observation_id,
            "source_snapshot_sha256": observation.source_snapshot_sha256,
            "encoded_payload_sha256": _canonical_sha256(encoded),
            "encoded_payload_bytes": len(_canonical_json(encoded).encode("utf-8")),
            "observation_queued": queued, "model_consumption_proven": False}


def _tool_trace(
    call: AgentToolCall,
    *,
    turn_index: int,
    status: Literal["accepted", "rejected", "rejected_not_executed"],
    code: str | None = None,
    dispatch_started: bool | None = None,
) -> dict[str, object]:
    trace: dict[str, object] = {
        "turn_index": turn_index,
        "tool_call_id": call.call_id,
        "tool": call.name,
        "binding_id": (
            call.arguments.get("binding_id")
            if isinstance(call.arguments, Mapping) and isinstance(call.arguments.get("binding_id"), str)
            else None
        ),
        "arguments_sha256": _canonical_sha256(call.arguments),
        "status": status,
    }
    if dispatch_started is not None:
        trace["dispatch_started"] = dispatch_started
    if code is not None:
        trace["error_code"] = code
    with trace_operation("tool.admission", kind="event", attributes={
        "turn_index": turn_index, "tool_call_id": call.call_id,
        "arguments_sha256": trace["arguments_sha256"], "admission_result": status,
        "dispatch_started": dispatch_started is True, "result_code": code,
        "timing_scope": "decision_record_only_not_validation_or_source_read_duration",
    }):
        pass
    return trace


def _observation_trace(
    observation: PatchSourceObservation,
    *,
    turn_index: int,
    admitted_to_model: bool,
    encoded_payload: Mapping[str, Any] | None = None,
    read_cost: int | None = None,
) -> dict[str, object]:
    return {
        "turn_index": turn_index,
        "observation_id": observation.observation_id,
        "binding_id": observation.binding_id,
        "source_ref": observation.source_ref,
        "source_hash": observation.source_hash,
        "source_snapshot_sha256": observation.source_snapshot_sha256,
        "observation_sha256": observation.payload_sha256,
        **({"read_cost": read_cost, "observation_origin": "prior_visible_source_reuse" if read_cost == 0 else "current_source_read"}
            if read_cost is not None else {}),
        "model_payload_bytes": observation.model_payload_bytes,
        "admitted_to_model": admitted_to_model if encoded_payload is None else False,
        **({"encoded_payload_sha256": _canonical_sha256(encoded_payload),
            "encoded_payload_bytes": len(_canonical_json(encoded_payload).encode("utf-8")),
            "expanded_payload_bytes": observation.model_payload_bytes,
            "payload_admitted": admitted_to_model,
            **({"excerpt_ref": deepcopy(encoded_payload["excerpt_ref"])} if "excerpt_ref" in encoded_payload else {}),
        } if encoded_payload is not None else {}),
        **(dict(observation.registered_evidence_identity) if observation.registered_evidence_identity is not None else {}),
    }


def _termination(
    code: str,
    reason: str,
    turn_index: int,
    *,
    status: Literal["completed", "blocked"] = "blocked",
    provider_diagnostic: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": status,
        "code": code,
        "reason": reason,
        "turn_index": turn_index,
    }
    if provider_diagnostic:
        result["provider_diagnostic"] = deepcopy(dict(provider_diagnostic))
    return result


def _loop_result(
    *,
    status: Literal["completed", "blocked"],
    final_content: str | None,
    provider: str,
    model: str,
    model_turns_used: int,
    tool_rounds_used: int,
    total_observation_bytes: int,
    usage_by_turn: list[dict[str, object]],
    tool_call_trace: list[dict[str, object]],
    observation_trace: list[dict[str, object]],
    termination: dict[str, object],
    final_request: AgentTurnRequest | None = None,
) -> PatchAgentLoopResult:
    return PatchAgentLoopResult(
        status=status,
        final_content=final_content,
        provider=provider,
        model=model,
        model_turns_used=model_turns_used,
        tool_rounds_used=tool_rounds_used,
        total_observation_bytes=total_observation_bytes,
        usage_by_turn=tuple(deepcopy(item) for item in usage_by_turn),
        tool_call_trace=tuple(deepcopy(item) for item in tool_call_trace),
        observation_trace=tuple(deepcopy(item) for item in observation_trace),
        termination=deepcopy(termination),
        final_request=deepcopy(final_request),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
