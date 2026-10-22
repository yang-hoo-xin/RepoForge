"""Fail-closed exact-text patch validation and isolated application."""

from __future__ import annotations

import codecs
import difflib
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from src.mod_agent.bindings import BindingResolver
from src.mod_agent.proposal.models import SemanticAssuranceV1


class PatchApplyError(RuntimeError):
    """Raised when an executable patch is unsafe, stale, or ambiguous."""


def patch_apply_failure_details(error: BaseException) -> dict[str, Any]:
    """Read executor-owned facts; an uninstrumented factory failure is unknown."""
    facts = getattr(error, "patch_apply_failure", None)
    if isinstance(facts, dict) and (
        type(facts.get("actual_write_performed")) is bool
        or facts.get("actual_write_performed") is None
    ) and facts.get("write_completion") in {"not_started", "partial", "complete", "unknown"}:
        return deepcopy(facts)
    return {"actual_write_performed": None, "real_patch_applied": False,
        "changed_files": [], "write_completion": "unknown",
        "write_failure": {"operation": "unreported_factory_failure", "file_observations": []}}


def attach_patch_apply_failure(error: BaseException, facts: Mapping[str, Any]) -> None:
    """Keep the original exception while preserving measured target effects."""
    try:
        error.patch_apply_failure = deepcopy(dict(facts))
    except Exception:
        pass


def _write_failure_details(root, staged, originals, replacements, operation, cleanup_errors):
    observations, confirmed = [], []
    for target, expected in staged.items():
        path = target.relative_to(root).as_posix()
        before_hash, expected_hash = _sha256_bytes(originals[target]), _sha256_bytes(expected)
        replacement = replacements[target]
        row = {"path": path, "before_sha256": before_hash, "expected_after_sha256": expected_hash,
               "replacement_status": replacement, "observed_sha256": None, "target_state": "unreadable"}
        if replacement == "confirmed":
            confirmed.append(path)
        try:
            observed = _sha256_bytes(target.read_bytes())
            row.update(observed_sha256=observed,
                       target_state="expected" if observed == expected_hash else "baseline" if observed == before_hash else "other")
        except Exception as read_error:
            row["observation_error_type"] = type(read_error).__name__
        observations.append(row)
    uncertain = any(value == "uncertain" for value in replacements.values())
    complete = bool(observations) and all(
        row["replacement_status"] == "confirmed" and row["target_state"] == "expected" for row in observations)
    return {"actual_write_performed": True if confirmed else None if uncertain else False,
        "real_patch_applied": complete, "changed_files": sorted(confirmed),
        "write_completion": "complete" if complete else "partial" if confirmed else "unknown" if uncertain else "not_started",
        "write_failure": {"operation": operation, "file_observations": observations,
                          "temporary_cleanup_errors": cleanup_errors}}


class ExactDiffApprovalRejectedError(RuntimeError):
    """Raised when the user explicitly rejects the displayed exact diff."""

    def __init__(
        self,
        message: str,
        *,
        human_semantic_decision: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.human_semantic_decision = (
            dict(human_semantic_decision)
            if human_semantic_decision is not None
            else None
        )


_HUMAN_SEMANTIC_DECISION_SCHEMA_VERSION = "human_semantic_decision_v1"
_HUMAN_SEMANTIC_DECISION_BINDING_FIELDS = {
    "schema_version",
    "feature_id",
    "feature_spec_artifact_hash",
    "proposal_id",
    "proposal_payload_sha256",
    "proposal_review_payload_sha256",
    "semantic_assurance_payload_sha256",
    "diff_hash",
}
_HUMAN_SEMANTIC_DECISION_FIELDS = (
    _HUMAN_SEMANTIC_DECISION_BINDING_FIELDS
    | {"decision", "reviewer_identity"}
)


@dataclass(frozen=True)
class MaterializedPatchBytes:
    """In-memory exact output bytes, with no review, approval, or behavior verdict.

    Only touched files are included. Compare outputs only with the same baseline
    and file set. File identities exclude rationale, change ordering, and JSON
    formatting; differing code is not evidence that a repair succeeded.
    """

    baseline_files: Mapping[str, bytes]
    files: Mapping[str, bytes]
    _file_results: tuple[dict[str, Any], ...] = dataclass_field(repr=False, compare=False)
    _diff_preview: tuple[dict[str, Any], ...] = dataclass_field(repr=False, compare=False)
    _binding_span_checks: tuple[dict[str, Any], ...] = dataclass_field(repr=False, compare=False)

    @property
    def baseline_sha256(self) -> str:
        return _materialized_files_sha256(self.baseline_files)

    @property
    def code_sha256(self) -> str:
        return _materialized_files_sha256(self.files)


def _materialized_files_sha256(files: Mapping[str, bytes]) -> str:
    return _sha256_json([
        {"path": path, "raw_bytes_sha256": _sha256_bytes(files[path])}
        for path in sorted(files)
    ])


def materialize_exact_text_patch(
    *,
    workspace_root: str | Path,
    diff_intent: Sequence[Mapping[str, Any]],
    implementation_plan: dict[str, Any],
) -> MaterializedPatchBytes:
    """Resolve and materialize existing exact replacements without writing.

    This is the byte-materialization portion of the formal applier, not a
    replacement Review or Dry-run. It enforces path, provided source hashes,
    spans, occurrence count, encoding, and BOM rules. Formal static checks and
    all write/semantic approval gates remain in ``ExactTextPatchApplier.run``.
    Legacy plans without source_hash retain their existing exact-text checks;
    current production callers provide source-hashed bindings.
    """
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise PatchApplyError("Patch workspace does not exist or is not a directory.")

    resolver = BindingResolver.from_implementation_plan(implementation_plan)
    staged: dict[Path, bytes] = {}
    original_by_target: dict[Path, bytes] = {}
    file_results: list[dict[str, Any]] = []
    diff_preview: list[dict[str, Any]] = []
    binding_span_checks: list[dict[str, Any]] = []
    for index, change in enumerate(diff_intent, start=1):
        binding_id = str(change.get("edit_binding_id") or "")
        paths = resolver.display_files_for_edit_ids([binding_id])
        if len(paths) != 1:
            raise PatchApplyError(f"Change {index} does not resolve to one edit path.")
        relative = _safe_relative_path(paths[0])
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PatchApplyError(f"Change {index} escapes the isolated workspace.") from exc
        if not target.is_file():
            raise PatchApplyError(f"Change {index} target file is missing: {relative}")

        patch = change.get("patch")
        if not isinstance(patch, dict) or patch.get("kind") != "exact_text_replace":
            raise PatchApplyError(f"Change {index} has no executable exact-text patch.")
        before = patch.get("before_text")
        after = patch.get("after_text")
        if not isinstance(before, str) or not isinstance(after, str) or before == after:
            raise PatchApplyError(f"Change {index} patch text is invalid or a no-op.")

        original = original_by_target.get(target)
        if original is None:
            original = target.read_bytes()
            _validate_utf8(original, change_index=index)
            original_by_target[target] = original
        source_hash = (resolver.edit_bindings.get(binding_id) or {}).get("source_hash")
        if source_hash is not None and source_hash != _sha256_bytes(original):
            raise PatchApplyError(
                f"Change {index} source_hash does not match the live source baseline."
            )
        current = staged.get(target, original)
        try:
            before_bytes = before.encode("utf-8")
            after_bytes = after.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PatchApplyError(
                f"Change {index} patch text is not valid UTF-8 text."
            ) from exc
        match_mode = patch.get("match_mode", "unique")
        if match_mode == "bound_span_exact":
            if current != original:
                raise PatchApplyError(
                    f"Change {index} bound-span match must be the first "
                    "change for its file."
                )
            bound_position = _bound_span_match_position(
                original=original,
                before=before_bytes,
                binding=resolver.edit_bindings.get(binding_id) or {},
                change_index=index,
            )
            binding_span_checks.append(
                _binding_span_check(
                    original=original,
                    before=before_bytes,
                    binding=resolver.edit_bindings.get(binding_id) or {},
                    position=bound_position,
                    change_index=index,
                    binding_id=binding_id,
                )
            )
            count = 1
            updated = (
                current[:bound_position]
                + after_bytes
                + current[bound_position + len(before_bytes) :]
            )
        elif match_mode == "unique":
            binding_span_check = _validate_bound_source_span(
                original=original,
                before=before_bytes,
                binding=resolver.edit_bindings.get(binding_id) or {},
                change_index=index,
                binding_id=binding_id,
            )
            if binding_span_check is not None:
                binding_span_checks.append(binding_span_check)
            count = current.count(before_bytes)
            if patch.get("expected_occurrences") != 1 or count != 1:
                raise PatchApplyError(
                    f"Change {index} expected one live source match but "
                    f"found {count}."
                )
            updated = current.replace(before_bytes, after_bytes, 1)
        else:
            raise PatchApplyError(
                f"Change {index} patch match_mode is unsupported."
            )
        if patch.get("expected_occurrences") != 1:
            raise PatchApplyError(
                f"Change {index} must expect exactly one occurrence."
            )
        if current.startswith(codecs.BOM_UTF8) != updated.startswith(codecs.BOM_UTF8):
            raise PatchApplyError(f"Change {index} must preserve the UTF-8 BOM.")
        staged[target] = updated
        before_sha256 = _sha256_bytes(current)
        after_sha256 = _sha256_bytes(updated)
        encoding = "utf-8-bom" if current.startswith(codecs.BOM_UTF8) else "utf-8"
        file_results.append(
            {
                "change_id": change.get("change_id"),
                "edit_binding_id": binding_id,
                "path": relative.as_posix(),
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "raw_before_sha256": before_sha256,
                "raw_after_sha256": after_sha256,
                "hash_mode": "raw_bytes",
                "encoding": encoding,
                "bom_preserved": current.startswith(codecs.BOM_UTF8)
                == updated.startswith(codecs.BOM_UTF8),
                "matched_occurrences": count,
            }
        )
        preview = {
            "change_id": change.get("change_id"),
            "edit_binding_id": binding_id,
            "path": relative.as_posix(),
            "before_text": before,
            "after_text": after,
            "expected_occurrences": 1,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "raw_before_sha256": before_sha256,
            "raw_after_sha256": after_sha256,
            "hash_mode": "raw_bytes",
            "encoding": encoding,
            "bom_preserved": current.startswith(codecs.BOM_UTF8)
            == updated.startswith(codecs.BOM_UTF8),
        }
        if "match_mode" in patch:
            preview["match_mode"] = match_mode
        diff_preview.append(preview)

    if not staged:
        raise PatchApplyError("Proposal contains no executable changes.")
    return MaterializedPatchBytes(
        baseline_files=MappingProxyType({
            target.relative_to(root).as_posix(): content
            for target, content in original_by_target.items()
        }),
        files=MappingProxyType({
            target.relative_to(root).as_posix(): content
            for target, content in staged.items()
        }),
        _file_results=tuple(file_results),
        _diff_preview=tuple(diff_preview),
        _binding_span_checks=tuple(binding_span_checks),
    )


def validate_materialized_patch_submission(materialized: MaterializedPatchBytes) -> dict[str, int]:
    """Return advisory formatting observations for an exactly materialized patch.

    为什么要有它：tb1 的候选在行为验收前因六行尾部空白被拒；通用格式统计不授予或撤销修改权限，原字节交给真实验收。
    The compatibility name remains callable; source authority is checked by
    materialize_exact_text_patch. No whitespace is trimmed or rewritten here.
    """
    return {"added_trailing_whitespace_count": _added_trailing_whitespace_count(
        original_by_target=materialized.baseline_files, staged=materialized.files,
    )}


class ExactTextPatchApplier:
    """Validate exact replacements and optionally write them inside one workspace."""

    def dry_run_v2(self, *, proposal, review, store):
        """Validate final v2 bytes against the actual baseline without applying.

        为什么要有它：U1接管只形成待交付记录，精确源码前像与候选身份仍需在Dry-run再次核验。
        """
        from src.mod_agent.task_agent_v2_handoff import revalidate_handoff_v2
        from src.mod_agent.task_agent_v2_state import TaskAgentV2State
        from src.mod_agent.proposal.reviewer import ProposalReviewer
        from src.mod_agent.model_qualification import canonical_hash
        pack = proposal["context_pack"]
        handoff = revalidate_handoff_v2(pack["task_agent_handoff_v2"], store=store)
        expected_review = ProposalReviewer().review_v2(proposal, store=store,
            feature_spec_ref=pack["feature_spec_ref"]).to_artifact_payload()
        if review != expected_review:
            raise PatchApplyError("V2_REVIEW_CHANGED")
        state = TaskAgentV2State(store.root, handoff.task_contract_ref)
        baseline = state.load(state.contract["baseline_manifest_ref"])
        root = Path(baseline["source_root"]).resolve(strict=True)
        originals, staged = {}, {}
        for change in handoff.changes:
            target = (root / _safe_relative_path(change["display_file"])).resolve(strict=True)
            if not target.is_relative_to(root) or target.is_symlink():
                raise PatchApplyError("V2_SOURCE_OUTSIDE_ROOT")
            raw = target.read_bytes()
            if hashlib.sha256(raw).hexdigest() != change["baseline_file_sha256"]:
                raise PatchApplyError("V2_SOURCE_PREIMAGE_CHANGED")
            originals[target], staged[target] = raw, change["patch"]["after"].encode("utf-8")
        checks = _run_low_cost_static_checks(staged)
        unified = _build_unified_diff(root=root, original_by_target=originals, staged=staged)
        return {"execution_mode": "dry_run", "actual_write_performed": False, "real_patch_applied": False,
            "would_write": True, "changed_files": list(proposal["files"]), "diff_stats": _unified_diff_stats(unified),
            "unified_diff": unified, "diff_hash": canonical_hash({"handoff": handoff.to_dict(), "diff": unified}),
            "structural_checks": checks, "task_agent_protocol": "v2", "candidate_source_id": handoff.candidate_source_id,
            "candidate_ref": handoff.candidate_ref, "check_ref": handoff.check_ref,
            "task_contract_ref": handoff.task_contract_ref, "binding_registry_ref": handoff.binding_registry_ref,
            "delivery_status": "pending", "whole_task_delivered": False}

    def run(
        self, *, workspace_root: str | Path, proposal: dict[str, Any],
        implementation_plan: dict[str, Any], review: dict[str, Any], allow_write: bool,
        approved_diff_hash: str | None = None,
        human_semantic_decision: Mapping[str, Any] | None = None,
        task_delegation_ref: Mapping[str, str] | None = None,
        task_delegation_resume_ref: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            return self._run(workspace_root=workspace_root, proposal=proposal,
                implementation_plan=implementation_plan, review=review, allow_write=allow_write,
                approved_diff_hash=approved_diff_hash, human_semantic_decision=human_semantic_decision,
                task_delegation_ref=task_delegation_ref, task_delegation_resume_ref=task_delegation_resume_ref)
        except BaseException as error:
            if not hasattr(error, "patch_apply_failure"):
                # Every physical target replacement is covered by the inner
                # handler. Errors before that point have confirmed zero writes.
                attach_patch_apply_failure(error, {"actual_write_performed": False,
                    "real_patch_applied": False, "changed_files": [], "write_completion": "not_started",
                    "write_failure": {"operation": "validation", "file_observations": []}})
            raise

    def _run(
        self,
        *,
        workspace_root: str | Path,
        proposal: dict[str, Any],
        implementation_plan: dict[str, Any],
        review: dict[str, Any],
        allow_write: bool,
        approved_diff_hash: str | None = None,
        human_semantic_decision: Mapping[str, Any] | None = None,
        task_delegation_ref: Mapping[str, str] | None = None,
        task_delegation_resume_ref: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if review.get("allowed_next_stage") != "dry_run_apply":
            raise PatchApplyError("Proposal review did not allow dry-run apply.")
        semantic_assurance, semantic_contract_present = (
            _validated_semantic_assurance(proposal, review)
        )
        human_semantic_review_required = (
            semantic_contract_present
            and semantic_assurance.verdict == "unproven"
        )
        materialized = materialize_exact_text_patch(
            workspace_root=workspace_root,
            diff_intent=proposal.get("diff_intent") or [],
            implementation_plan=implementation_plan,
        )
        root = Path(workspace_root).resolve()
        staged = {root / relative: content for relative, content in materialized.files.items()}
        original_by_target = {
            root / relative: content
            for relative, content in materialized.baseline_files.items()
        }
        file_results = list(materialized._file_results)
        diff_preview = list(materialized._diff_preview)
        binding_span_checks = list(materialized._binding_span_checks)
        structural_checks = _run_low_cost_static_checks(staged)
        unified_diff = _build_unified_diff(
            root=root,
            original_by_target=original_by_target,
            staged=staged,
        )
        format_diagnostics = validate_materialized_patch_submission(materialized)
        unified_diff_bytes = unified_diff.encode("utf-8")
        diff_stats = _unified_diff_stats(unified_diff)
        context_hashes = {
            "proposal_payload_sha256": _sha256_json(proposal),
            "proposal_review_payload_sha256": _sha256_json(review),
            "semantic_assurance_payload_sha256": _sha256_json(
                semantic_assurance.to_dict()
            ),
            "implementation_plan_payload_sha256": _sha256_json(implementation_plan),
            "evidence_artifact_hash": _evidence_artifact_hash(proposal),
            "source_snapshot_sha256": _sha256_json(
                [
                    {
                        "path": item["path"],
                        "before_sha256": item["before_sha256"],
                    }
                    for item in diff_preview
                ]
            ),
        }
        diff_hash = _diff_hash(
            proposal_id=str(proposal.get("proposal_id") or ""),
            diff_preview=diff_preview,
            context_hashes=context_hashes,
        )
        semantic_decision_binding = (
            _human_semantic_decision_binding(
                proposal=proposal,
                diff_hash=diff_hash,
                context_hashes=context_hashes,
            )
            if human_semantic_review_required
            else None
        )
        validated_human_decision: dict[str, Any] | None = None
        delegation_fields = {}
        if task_delegation_ref is not None:
            from .task_delegation import delegation_result_fields, validate_task_delegation

            if not allow_write or human_semantic_decision is not None:
                raise PatchApplyError("Delegation is exclusive to a non-human isolated Apply.")
            validate_task_delegation(task_delegation_ref, workspace_root=root, diff_hash=diff_hash,
                context_hashes=context_hashes, materialized_code_identity={
                    "baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256,
                    "files": sorted(materialized.files)}, before_apply=True,
                resume_report_ref=task_delegation_resume_ref)
            delegation_fields = delegation_result_fields(task_delegation_ref, task_delegation_resume_ref)
        elif task_delegation_resume_ref is not None:
            raise PatchApplyError("A delegated resume receipt requires its exact decision reference.")
        if allow_write:
            if approved_diff_hash is None:
                raise PatchApplyError("Exact diff hash approval is required before writing.")
            if approved_diff_hash != diff_hash:
                raise PatchApplyError(
                    "Approved diff hash does not match the current exact patch."
                )
            if human_semantic_review_required and not delegation_fields:
                validated_human_decision = _validate_human_semantic_decision(
                    human_semantic_decision,
                    expected_binding=semantic_decision_binding or {},
                )
                if validated_human_decision["decision"] == "reject":
                    raise ExactDiffApprovalRejectedError(
                        "User rejected the displayed exact diff after semantic review.",
                        human_semantic_decision=validated_human_decision,
                    )
            replacements = {target: "not_attempted" for target in staged}
            operation = "temporary_write"
            cleanup_errors = []
            try:
                for target, content in staged.items():
                    temporary = target.with_name(f".{target.name}.v2-codegen.tmp")
                    original_error = None
                    try:
                        operation = "temporary_write"
                        temporary.write_bytes(content)
                        operation = "temporary_readback"
                        if temporary.read_bytes() != content:
                            raise PatchApplyError("Temporary patch output does not match dry-run bytes.")
                        operation = "replace"
                        replacements[target] = "uncertain"
                        os.replace(temporary, target)
                        replacements[target] = "confirmed"
                        operation = "target_readback"
                        if target.read_bytes() != content:
                            raise PatchApplyError("Applied patch bytes do not match dry-run bytes.")
                    except BaseException as error:
                        original_error = error
                        raise
                    finally:
                        try:
                            if temporary.exists():
                                temporary.unlink()
                        except BaseException as cleanup_error:
                            if original_error is None:
                                operation = "temporary_cleanup"
                                raise
                            cleanup_errors.append({"path": target.relative_to(root).as_posix(),
                                                   "error_type": type(cleanup_error).__name__})
            except BaseException as error:
                facts = _write_failure_details(root, staged, original_by_target, replacements, operation, cleanup_errors)
                facts.update(diff_hash=diff_hash, approved_diff_hash=approved_diff_hash,
                    approved_by_user=not bool(delegation_fields), context_hashes=context_hashes,
                    human_semantic_decision=validated_human_decision,
                    human_semantic_review_result="approved" if validated_human_decision is not None else "not_required_for_legacy_non_semantic_contract")
                facts.update(delegation_fields)
                attach_patch_apply_failure(error, facts)
                raise

        return {
            "execution_mode": "isolated_apply" if allow_write else "dry_run",
            "actual_write_performed": allow_write,
            "real_patch_applied": allow_write,
            "approved_by_review": True,
            "review_decision": review.get("decision"),
            "proposal_id": proposal.get("proposal_id"),
            "diff_hash": diff_hash,
            "approved_diff_hash": approved_diff_hash,
            "approved_by_user": allow_write and approved_diff_hash == diff_hash,
            "semantic_assurance": semantic_assurance.to_dict(),
            "semantic_assurance_contract": (
                "semantic_assurance_v1"
                if semantic_contract_present
                else "legacy_non_semantic_exact_diff_v0"
            ),
            "human_semantic_review_required": human_semantic_review_required,
            "human_semantic_review_completed": (
                validated_human_decision is not None
                and validated_human_decision["decision"] == "approve"
            ),
            "human_semantic_review_result": (
                "approved"
                if validated_human_decision is not None
                else (
                    "pending"
                    if human_semantic_review_required
                    else "not_required_for_legacy_non_semantic_contract"
                )
            ),
            "human_semantic_decision_binding": semantic_decision_binding,
            "human_semantic_decision": validated_human_decision,
            "context_hashes": context_hashes,
            "diff_preview": diff_preview,
            "unified_diff": unified_diff,
            "unified_diff_sha256": hashlib.sha256(unified_diff_bytes).hexdigest(),
            "unified_diff_bytes": len(unified_diff_bytes),
            "format_diagnostics": format_diagnostics,
            "diff_stats": {
                "files": len(staged),
                "change_units": len(diff_preview),
                **diff_stats,
            },
            "static_checks": [
                *binding_span_checks,
                *structural_checks,
            ],
            "changed_files": sorted({item["path"] for item in file_results}),
            "changes": file_results,
            **delegation_fields,
        }


def _validated_semantic_assurance(
    proposal: dict[str, Any],
    review: dict[str, Any],
) -> tuple[SemanticAssuranceV1, bool]:
    proposal_has_contract = "semantic_assurance" in proposal
    review_has_contract = "semantic_assurance" in review
    proposal_value = proposal.get("semantic_assurance")
    review_value = review.get("semantic_assurance")
    try:
        proposal_assurance = (
            SemanticAssuranceV1.from_dict(proposal_value)
            if proposal_has_contract and isinstance(proposal_value, Mapping)
            else SemanticAssuranceV1.unproven()
        )
        review_assurance = (
            SemanticAssuranceV1.from_dict(review_value)
            if review_has_contract and isinstance(review_value, Mapping)
            else SemanticAssuranceV1.unproven()
        )
    except ValueError as exc:
        raise PatchApplyError(f"Semantic assurance is invalid: {exc}") from exc
    if proposal_has_contract and not isinstance(proposal_value, Mapping):
        raise PatchApplyError("Proposal semantic assurance is malformed.")
    if review_has_contract and not isinstance(review_value, Mapping):
        raise PatchApplyError("Review semantic assurance is malformed.")
    if proposal_assurance != review_assurance:
        raise PatchApplyError(
            "Proposal and review semantic assurance results do not match."
        )
    semantic_contract_present = proposal_has_contract or review_has_contract
    required = review.get("human_semantic_review_required")
    if required is not None and not isinstance(required, bool):
        raise PatchApplyError(
            "Review human_semantic_review_required must be a boolean."
        )
    if semantic_contract_present and proposal_assurance.verdict != "unproven":
        raise PatchApplyError(
            "Semantic Assurance V0 accepts only the unproven machine verdict."
        )
    if (
        semantic_contract_present
        and proposal_assurance.verdict == "unproven"
        and required is not True
    ):
        raise PatchApplyError(
            "Unproven semantic assurance requires an explicit exact-diff human "
            "semantic review obligation."
        )
    return proposal_assurance, semantic_contract_present


def _human_semantic_decision_binding(
    *,
    proposal: Mapping[str, Any],
    diff_hash: str,
    context_hashes: Mapping[str, Any],
) -> dict[str, str]:
    feature_id = proposal.get("feature_id")
    proposal_id = proposal.get("proposal_id")
    context_pack = proposal.get("context_pack")
    artifact_hashes = (
        context_pack.get("artifact_hashes")
        if isinstance(context_pack, Mapping)
        else None
    )
    feature_spec_hash = (
        artifact_hashes.get("feature_spec")
        if isinstance(artifact_hashes, Mapping)
        else None
    )
    exact_values = {
        "feature_id": feature_id,
        "feature_spec_artifact_hash": feature_spec_hash,
        "proposal_id": proposal_id,
        "proposal_payload_sha256": context_hashes.get("proposal_payload_sha256"),
        "proposal_review_payload_sha256": context_hashes.get(
            "proposal_review_payload_sha256"
        ),
        "semantic_assurance_payload_sha256": context_hashes.get(
            "semantic_assurance_payload_sha256"
        ),
        "diff_hash": diff_hash,
    }
    if any(not isinstance(value, str) or not value for value in exact_values.values()):
        raise PatchApplyError(
            "Human semantic decision binding requires exact non-empty Feature, "
            "Proposal, assurance, review, and diff identities."
        )
    return {
        "schema_version": _HUMAN_SEMANTIC_DECISION_SCHEMA_VERSION,
        **exact_values,
    }


def _validate_human_semantic_decision(
    value: Mapping[str, Any] | None,
    *,
    expected_binding: Mapping[str, str],
) -> dict[str, Any]:
    if value is None:
        raise PatchApplyError(
            "A structured human semantic decision is required before writing."
        )
    if not isinstance(value, Mapping) or set(value) != _HUMAN_SEMANTIC_DECISION_FIELDS:
        raise PatchApplyError(
            "The human semantic decision fields must exactly match the V1 contract."
        )
    decision = value.get("decision")
    if decision not in {"approve", "reject"}:
        raise PatchApplyError(
            "Human semantic decision must be exactly approve or reject."
        )
    reviewer_identity = value.get("reviewer_identity")
    if not isinstance(reviewer_identity, str) or not reviewer_identity.strip():
        raise PatchApplyError(
            "Human semantic decision reviewer_identity must be a non-empty string."
        )
    for field, expected in expected_binding.items():
        if value.get(field) != expected:
            raise PatchApplyError(
                f"Human semantic decision {field} does not match the current dry-run binding."
            )
    return dict(value)


def _safe_relative_path(raw: str) -> Path:
    normalized = PurePosixPath(raw.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise PatchApplyError("Edit binding path must be workspace-relative.")
    return Path(*normalized.parts)


def _validate_utf8(value: bytes, *, change_index: int) -> None:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchApplyError(
            f"Change {change_index} target is not valid UTF-8; exact bytes were not changed."
        ) from exc


def _validate_bound_source_span(
    *,
    original: bytes,
    before: bytes,
    binding: dict[str, Any],
    change_index: int,
    binding_id: str,
) -> dict[str, Any] | None:
    span = binding.get("target_span")
    if not isinstance(span, dict):
        return None
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    if (
        not isinstance(start_line, int)
        or isinstance(start_line, bool)
        or not isinstance(end_line, int)
        or isinstance(end_line, bool)
        or start_line <= 0
        or end_line < start_line
    ):
        raise PatchApplyError(
            f"Change {change_index} edit binding has an invalid target_span."
        )
    positions: list[int] = []
    offset = 0
    while True:
        position = original.find(before, offset)
        if position < 0:
            break
        positions.append(position)
        offset = position + max(len(before), 1)
    if len(positions) != 1:
        raise PatchApplyError(
            f"Change {change_index} before_text is not unique in the bound source."
        )
    observed_start = original[: positions[0]].count(b"\n") + 1
    content = before.rstrip(b"\r\n")
    observed_end = observed_start + content.count(b"\n")
    if observed_start < start_line or observed_end > end_line:
        raise PatchApplyError(
            f"Change {change_index} before_text is outside edit binding target_span."
        )
    return {
        "check_id": "binding_target_span",
        "status": "passed",
        "change_index": change_index,
        "binding_id": binding_id,
        "bound_start_line": start_line,
        "bound_end_line": end_line,
        "observed_start_line": observed_start,
        "observed_end_line": observed_end,
    }


def _bound_span_match_position(
    *,
    original: bytes,
    before: bytes,
    binding: dict[str, Any],
    change_index: int,
) -> int:
    span = binding.get("target_span")
    if not isinstance(span, dict):
        raise PatchApplyError(
            f"Change {change_index} bound-span match requires target_span."
        )
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    if (
        not isinstance(start_line, int)
        or isinstance(start_line, bool)
        or not isinstance(end_line, int)
        or isinstance(end_line, bool)
        or start_line <= 0
        or end_line < start_line
    ):
        raise PatchApplyError(
            f"Change {change_index} edit binding has an invalid target_span."
        )
    lines = original.splitlines(keepends=True)
    if end_line > len(lines):
        raise PatchApplyError(
            f"Change {change_index} target_span exceeds the live source."
        )
    span_start = sum(len(line) for line in lines[: start_line - 1])
    span_end = sum(len(line) for line in lines[:end_line])
    bound_bytes = original[span_start:span_end]
    if bound_bytes.count(before) != 1:
        raise PatchApplyError(
            f"Change {change_index} before_text does not match exactly once "
            "inside target_span."
        )
    return span_start + bound_bytes.index(before)


def _binding_span_check(
    *,
    original: bytes,
    before: bytes,
    binding: dict[str, Any],
    position: int,
    change_index: int,
    binding_id: str,
) -> dict[str, Any]:
    span = binding["target_span"]
    observed_start = original[:position].count(b"\n") + 1
    observed_end = observed_start + before.rstrip(b"\r\n").count(b"\n")
    if (
        observed_start < span["start_line"]
        or observed_end > span["end_line"]
    ):
        raise PatchApplyError(
            f"Change {change_index} before_text is outside edit binding "
            "target_span."
        )
    return {
        "check_id": "binding_target_span",
        "status": "passed",
        "change_index": change_index,
        "binding_id": binding_id,
        "bound_start_line": span["start_line"],
        "bound_end_line": span["end_line"],
        "observed_start_line": observed_start,
        "observed_end_line": observed_end,
        "match_mode": "bound_span_exact",
    }


def _run_low_cost_static_checks(staged: dict[Path, bytes]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for target, content in staged.items():
        if target.suffix.lower() != ".cs":
            continue
        text = content.decode("utf-8-sig")
        error = _csharp_delimiter_error(text)
        if error is not None:
            raise PatchApplyError(
                f"C# low-cost structural check failed for {target.name}: {error}"
            )
        checks.append(
            {
                "check_id": "csharp_delimiter_balance",
                "status": "passed",
                "path": target.name,
                "limitations": (
                    "This lexical check does not resolve symbols, types, overloads, "
                    "or all control-flow errors."
                ),
            }
        )
    return checks


def _csharp_delimiter_error(text: str) -> str | None:
    stack: list[tuple[str, int]] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    state = "code"
    index = 0
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "line_comment":
            if current == "\n":
                state = "code"
            index += 1
            continue
        if state == "block_comment":
            if current == "*" and following == "/":
                state = "code"
                index += 2
            else:
                index += 1
            continue
        if state == "string":
            if current == "\\":
                index += 2
            elif current == '"':
                state = "code"
                index += 1
            else:
                index += 1
            continue
        if state == "verbatim_string":
            if current == '"' and following == '"':
                index += 2
            elif current == '"':
                state = "code"
                index += 1
            else:
                index += 1
            continue
        if state == "char":
            if current == "\\":
                index += 2
            elif current == "'":
                state = "code"
                index += 1
            else:
                index += 1
            continue
        if current == "/" and following == "/":
            state = "line_comment"
            index += 2
            continue
        if current == "/" and following == "*":
            state = "block_comment"
            index += 2
            continue
        if text.startswith('$@"', index) or text.startswith('@$"', index):
            state = "verbatim_string"
            index += 3
            continue
        if current == "@" and following == '"':
            state = "verbatim_string"
            index += 2
            continue
        if current == "$" and following == '"':
            state = "string"
            index += 2
            continue
        if current == '"':
            state = "string"
            index += 1
            continue
        if current == "'":
            state = "char"
            index += 1
            continue
        if current in "([{":
            stack.append((current, index))
        elif current in ")]}":
            if not stack or stack[-1][0] != pairs[current]:
                return f"unmatched {current!r} at character {index}"
            stack.pop()
        index += 1
    if state in {"block_comment", "string", "verbatim_string", "char"}:
        return f"unterminated {state.replace('_', ' ')}"
    if stack:
        delimiter, position = stack[-1]
        return f"unclosed {delimiter!r} at character {position}"
    return None


def _build_unified_diff(
    *,
    root: Path,
    original_by_target: dict[Path, bytes],
    staged: dict[Path, bytes],
) -> str:
    chunks: list[str] = []
    for target in sorted(staged, key=lambda item: item.as_posix()):
        relative = target.relative_to(root).as_posix()
        before_lines = original_by_target[target].decode("utf-8-sig").splitlines()
        after_lines = staged[target].decode("utf-8-sig").splitlines()
        chunks.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
                lineterm="",
            )
        )
    return "\n".join(chunks) + ("\n" if chunks else "")


def _unified_diff_stats(unified_diff: str) -> dict[str, int]:
    lines = unified_diff.splitlines()
    return {
        "insertions": sum(
            1 for line in lines if line.startswith("+") and not line.startswith("+++")
        ),
        "deletions": sum(
            1 for line in lines if line.startswith("-") and not line.startswith("---")
        ),
    }


def _added_trailing_whitespace_count(
    *,
    original_by_target: Mapping[Path | str, bytes],
    staged: Mapping[Path | str, bytes],
) -> int:
    count = 0
    for target, after_bytes in staged.items():
        before_lines = original_by_target[target].decode("utf-8-sig").splitlines()
        after_lines = after_bytes.decode("utf-8-sig").splitlines()
        for tag, _, _, after_start, after_end in difflib.SequenceMatcher(
            None,
            before_lines,
            after_lines,
            autojunk=False,
        ).get_opcodes():
            if tag not in {"insert", "replace"}:
                continue
            count += sum(
                line.endswith((" ", "\t"))
                for line in after_lines[after_start:after_end]
            )
    return count


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _diff_hash(
    *,
    proposal_id: str,
    diff_preview: list[dict[str, Any]],
    context_hashes: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "proposal_id": proposal_id,
            "context_hashes": context_hashes,
            "changes": diff_preview,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence_artifact_hash(proposal: dict[str, Any]) -> str | None:
    context_pack = proposal.get("context_pack")
    if not isinstance(context_pack, dict):
        return None
    artifact_hashes = context_pack.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        return None
    value = artifact_hashes.get("evidence_pack")
    return str(value) if value else None
