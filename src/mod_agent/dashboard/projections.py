"""Sanitized controlled-agent dashboard projections."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.mod_agent.artifacts import (
    ArtifactRef,
    ArtifactRegistryError,
    ArtifactStore,
    ArtifactStoreError,
)

EXPECTED_ARTIFACT_TYPES = (
    "request_planning_report",
    "mod_planning_brief",
    "feature_spec",
    "evidence_pack",
    "implementation_plan",
    "patch_proposal",
    "proposal_review",
    "dry_run_result",
    "build_result",
    "runtime_evidence",
    "repair_plan",
    "eval_report",
    "git_change_record",
    "workflow_summary",
    "workflow_checkpoint",
    "resume_report",
)

_MAX_STRING_CHARS = 240
_REDACTED = "[redacted]"
_TRUNCATED_SUFFIX = "...[truncated]"
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_POSIX_ABSOLUTE_PATH = re.compile(r"^/(?:[^/\0]+)(?:/[^/\0]+)*$")
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)
_PATH_KEY_PARTS = (
    "file",
    "files",
    "path",
    "paths",
    "source",
    "source_path",
    "target_files",
)


class DashboardProjectionBuilder:
    """Build read-only, sanitized summaries from feature artifacts."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.registry = artifact_store.registry

    def build_feature_projection(self, feature_id: str) -> dict[str, Any]:
        """Return a dashboard-safe projection for one feature."""

        refs = self.artifact_store.list_feature_artifacts(feature_id)
        if not refs:
            return {
                "feature_id": feature_id,
                "status": "empty",
                "workflow": self._empty_section("No controlled-agent artifacts found."),
                "evidence": self._empty_section("No evidence_pack artifact found."),
                "proposal": self._empty_section("No proposal artifacts found."),
                "build": self._empty_section("No build_result artifact found."),
                "runtime": self._empty_section("No runtime_evidence artifact found."),
                "repair": self._empty_section("No repair_plan artifact found."),
                "evaluation": self._empty_section("No eval_report artifact found."),
                "git": self._empty_section("No git_change_record artifact found."),
                "artifacts": [],
                "artifact_dag": {"nodes": [], "edges": []},
                "missing_artifacts": list(EXPECTED_ARTIFACT_TYPES),
                "blockers": [],
            }

        envelopes = self._read_envelopes(refs)
        artifact_by_type = {
            envelope["artifact_type"]: envelope
            for envelope in envelopes
            if envelope.get("artifact_type") in EXPECTED_ARTIFACT_TYPES
        }
        artifact_summaries = [
            self._artifact_summary(envelope)
            for envelope in sorted(
                envelopes,
                key=lambda item: (
                    EXPECTED_ARTIFACT_TYPES.index(item["artifact_type"])
                    if item.get("artifact_type") in EXPECTED_ARTIFACT_TYPES
                    else len(EXPECTED_ARTIFACT_TYPES),
                    item.get("artifact_type", ""),
                ),
            )
        ]
        missing = [
            artifact_type
            for artifact_type in EXPECTED_ARTIFACT_TYPES
            if artifact_type not in artifact_by_type
        ]
        workflow = self._workflow_summary(artifact_by_type)
        evaluation = self._evaluation_summary(artifact_by_type)
        active, historical, terminal_blockers = self._resolve_current_terminal(
            envelopes, artifact_by_type, artifact_summaries, workflow, evaluation)
        blockers = self._collect_blockers(active) + terminal_blockers
        return {
            "feature_id": feature_id,
            "status": "blocked" if blockers else "available",
            "workflow": workflow,
            "evidence": self._evidence_summary(artifact_by_type),
            "proposal": self._proposal_summary(artifact_by_type),
            "build": self._build_summary(artifact_by_type),
            "runtime": self._runtime_summary(artifact_by_type),
            "repair": self._repair_summary(artifact_by_type),
            "evaluation": evaluation,
            "git": self._git_summary(artifact_by_type),
            "artifacts": artifact_summaries,
            "artifact_dag": self._artifact_dag(envelopes),
            "missing_artifacts": missing,
            "blockers": blockers,
            "historical_blockers": self._collect_blockers(historical),
        }

    def _resolve_current_terminal(self, envelopes, by_type, summaries, workflow, evaluation):
        """Keep historical failures visible without overriding a valid current result."""
        freshness = {item["artifact_type"]: item["freshness"] for item in summaries}
        summary = by_type.get("workflow_summary")
        report = by_type.get("eval_report")
        summary_data, report_data = _data(summary), _data(report)
        summary_fresh = freshness.get("workflow_summary", {}).get("status") == "fresh"
        report_fresh = freshness.get("eval_report", {}).get("status") == "fresh"
        summary_eval_inputs = [item for item in (summary or {}).get("inputs", {}).get("upstream_artifacts", [])
                               if isinstance(item, dict) and item.get("artifact_type") == "eval_report"]
        current_accepted = bool(summary and report and summary_fresh and report_fresh
            and summary.get("status") == report.get("status") == "success"
            and summary_data.get("final_status") == "accepted" and summary_data.get("claims_real_mod_success") is True
            and report_data.get("verdict") == "accepted" and report_data.get("claims_real_success") is True
            and (report_data.get("evaluation_gate") or {}).get("status") == "passed"
            and len(summary_eval_inputs) == 1
            and summary_eval_inputs[0].get("artifact_hash") == report.get("artifact_hash"))
        terminal_blockers = []
        if report:
            evaluation["freshness"] = freshness.get("eval_report")
            if not report_fresh:
                evaluation.update(status="stale", claims_real_success=False)
        if summary:
            workflow["freshness"] = freshness.get("workflow_summary")
            claims_accepted = summary_data.get("final_status") == "accepted" or summary_data.get("claims_real_mod_success") is True
            if claims_accepted and not current_accepted:
                stale = not summary_fresh or not report_fresh
                workflow.update(status="stale" if stale else "blocked",
                    recorded_final_status=summary_data.get("final_status"),
                    final_status="stale" if stale else "unverified", claims_real_mod_success=False)
                terminal_blockers.append({"artifact_type": "workflow_summary", "stage": "dashboard",
                    "code": "CURRENT_TERMINAL_STALE" if stale else "CURRENT_TERMINAL_LINEAGE_INVALID",
                    "message": "Stored success no longer has fresh inputs." if stale
                    else "Stored success does not reference the current accepted evaluation."})
        if report_fresh and report_data.get("verdict") == "blocked":
            workflow.update(status="blocked", final_status="blocked", claims_real_mod_success=False,
                workflow_completion_status=report_data.get("workflow_completion_status"),
                requested_behavior_status=report_data.get("requested_behavior_status"),
                terminal_authority="eval_report")
            if not (report.get("blockers") or report_data.get("blockers")):
                terminal_blockers.append({"artifact_type": "eval_report", "stage": "evaluation",
                    "code": "CURRENT_EVALUATION_BLOCKED", "message": "The current evaluation is blocked."})
        active, historical = [], []
        for envelope in envelopes:
            inputs = envelope.get("inputs", {}).get("upstream_artifacts", [])
            old_evals = [item for item in inputs if isinstance(item, dict) and item.get("artifact_type") == "eval_report"]
            # Scope: display only; never edit or delete historical business artifacts.
            # 为什么要有它：只有新且有效的最终验收可替代旧失败显示，当前失败、无效引用或输入变化不能被隐藏。
            superseded = (current_accepted and envelope.get("artifact_type") == "blocked_report"
                and freshness.get("blocked_report", {}).get("status") == "stale"
                and len(old_evals) == 1 and old_evals[0].get("artifact_hash") != report.get("artifact_hash"))
            (historical if superseded else active).append(envelope)
        if any(item.get("artifact_type") == "blocked_report" and
               (item.get("blockers") or _data(item).get("blockers")) for item in active):
            workflow.update(status="blocked", recorded_final_status=summary_data.get("final_status"),
                            final_status="blocked", claims_real_mod_success=False, terminal_authority="blocked_report")
        return active, historical, terminal_blockers

    def _read_envelopes(self, refs: list[ArtifactRef]) -> list[dict[str, Any]]:
        envelopes = []
        for ref in refs:
            try:
                envelopes.append(self.artifact_store.read(ref))
            except ArtifactStoreError as exc:
                raw = self._raw_envelope_metadata(ref)
                envelopes.append(
                    {
                        "schema_version": raw.get("schema_version"),
                        "artifact_type": ref.artifact_type,
                        "feature_id": ref.feature_id,
                        "producer_stage": raw.get("producer_stage"),
                        "status": raw.get("status", "unknown"),
                        "data": {
                            "message": f"Artifact read failed: {exc}",
                        },
                        "artifact_hash": ref.artifact_hash,
                        "inputs": {"upstream_artifacts": []},
                        "blockers": [
                            {
                                "stage": "dashboard",
                                "code": "ARTIFACT_READ_FAILED",
                                "message": str(exc),
                            }
                        ],
                        "trace_refs": [],
                    }
                )
        return envelopes

    def _raw_envelope_metadata(self, ref: ArtifactRef) -> dict[str, Any]:
        try:
            path = self.artifact_store._guard_path(Path(ref.path))
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (ArtifactStoreError, OSError, json.JSONDecodeError):
            return {}
        return {
            "schema_version": envelope.get("schema_version"),
            "producer_stage": envelope.get("producer_stage"),
            "status": envelope.get("status"),
        }

    def _artifact_summary(self, envelope: dict[str, Any]) -> dict[str, Any]:
        artifact_type = str(envelope["artifact_type"])
        data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}
        try:
            contract = self.registry.get(artifact_type)
        except ArtifactRegistryError as exc:
            return {
                "artifact_type": artifact_type,
                "schema_version": envelope.get("schema_version"),
                "producer_stage": envelope.get("producer_stage"),
                "status": envelope.get("status"),
                "artifact_status": envelope.get("status"),
                "projection_status": "blocked",
                "artifact_hash": envelope.get("artifact_hash"),
                "artifact_file": Path(self._artifact_path_for(envelope)).name,
                "freshness": {
                    "status": "blocked",
                    "stale_reasons": [str(exc)],
                },
                "dashboard_projection": {},
                "upstream_artifacts": [],
                "blocker_count": 1,
                "trace_ref_count": len(envelope.get("trace_refs") or []),
            }
        projected_fields = {
            field: _sanitize_value(data[field], key_path=(field,))
            for field in contract.dashboard_projection
            if field in data
        }
        ref = ArtifactRef(
            feature_id=str(envelope["feature_id"]),
            artifact_type=artifact_type,
            path=self._artifact_path_for(envelope),
            artifact_hash=str(envelope["artifact_hash"]),
        )
        return {
            "artifact_type": artifact_type,
            "schema_version": envelope.get("schema_version"),
            "producer_stage": envelope.get("producer_stage"),
            "status": envelope.get("status"),
            "artifact_status": envelope.get("status"),
            "projection_status": "available",
            "artifact_hash": envelope.get("artifact_hash"),
            "artifact_file": Path(ref.path).name,
            "freshness": self._freshness_summary(ref),
            "dashboard_projection": projected_fields,
            "upstream_artifacts": [
                {
                    "artifact_type": upstream.get("artifact_type"),
                    "artifact_hash": upstream.get("artifact_hash"),
                    "artifact_file": Path(str(upstream.get("path", ""))).name,
                }
                for upstream in envelope.get("inputs", {}).get("upstream_artifacts", [])
                if isinstance(upstream, dict)
            ],
            "blocker_count": len(envelope.get("blockers") or []),
            "trace_ref_count": len(envelope.get("trace_refs") or []),
        }

    def _artifact_path_for(self, envelope: dict[str, Any]) -> str:
        return str(
            self.artifact_store.root
            / "features"
            / str(envelope["feature_id"])
            / f"{envelope['artifact_type']}.json"
        )

    def _freshness_summary(self, ref: ArtifactRef) -> dict[str, Any]:
        try:
            freshness = self.artifact_store.check_freshness(ref)
            return {
                "status": freshness.status,
                "stale_reasons": list(freshness.stale_reasons),
            }
        except ArtifactStoreError as exc:
            return {
                "status": "blocked",
                "stale_reasons": [str(exc)],
            }

    @staticmethod
    def _workflow_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        summary = artifact_by_type.get("workflow_summary")
        checkpoint = artifact_by_type.get("workflow_checkpoint")
        resume = artifact_by_type.get("resume_report")
        if summary is None and checkpoint is None and resume is None:
            return DashboardProjectionBuilder._empty_section(
                "No workflow summary, checkpoint, or resume report found."
            )
        summary_data = _data(summary)
        checkpoint_data = _data(checkpoint)
        resume_data = _data(resume)
        return {
            "status": "available",
            "message": _sanitize_value(summary_data.get("message"), key_path=("message",)),
            "execution_mode": summary_data.get("execution_mode"),
            "workflow_completion_status": summary_data.get("workflow_completion_status"),
            "requested_behavior_status": summary_data.get("requested_behavior_status"),
            "final_status": summary_data.get("final_status"),
            "claims_real_mod_success": summary_data.get("claims_real_mod_success"),
            "claims_real_build_success": summary_data.get("claims_real_build_success"),
            "claims_real_runtime_success": summary_data.get("claims_real_runtime_success"),
            "checkpoint": {
                "checkpoint_id": checkpoint_data.get("checkpoint_id"),
                "checkpoint_stage": checkpoint_data.get("checkpoint_stage"),
                "status": checkpoint_data.get("status"),
            },
            "resume": {
                "resume_id": resume_data.get("resume_id"),
                "resume_stage": resume_data.get("resume_stage"),
                "allowed": resume_data.get("allowed"),
                "status": resume_data.get("status"),
            },
        }

    @staticmethod
    def _evidence_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        evidence = artifact_by_type.get("evidence_pack")
        if evidence is None:
            return DashboardProjectionBuilder._empty_section("No evidence_pack artifact found.")
        data = _data(evidence)
        return {
            "projection_status": (
                "available" if data.get("can_generate_patch") else "blocked"
            ),
            "readiness": data.get("readiness"),
            "can_generate_patch": data.get("can_generate_patch"),
            "authority": data.get("authority"),
            "authoritative_for_patch": data.get("authoritative_for_patch"),
            "item_count": len(data.get("items") or []),
            "retrieval_methods_used": _sanitize_value(
                data.get("retrieval_methods_used") or [],
                key_path=("retrieval_methods_used",),
            ),
            "missing_evidence_count": len(data.get("missing_evidence") or []),
            "blocker_count": len(data.get("blockers") or []),
        }

    @staticmethod
    def _proposal_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        proposal = artifact_by_type.get("patch_proposal")
        review = artifact_by_type.get("proposal_review")
        if proposal is None and review is None:
            return DashboardProjectionBuilder._empty_section("No proposal artifacts found.")
        proposal_data = _data(proposal)
        review_data = _data(review)
        return {
            "status": "blocked" if proposal_data.get("blocked") else "available",
            "proposal_id": proposal_data.get("proposal_id"),
            "summary": _sanitize_value(proposal_data.get("summary"), key_path=("summary",)),
            "file_count": len(proposal_data.get("files") or []),
            "change_count": len(proposal_data.get("diff_intent") or []),
            "change_unit_count": len(proposal_data.get("change_units") or []),
            "evidence_ref_count": len(proposal_data.get("evidence_refs") or []),
            "rollback_hint_count": len(proposal_data.get("rollback_hints") or []),
            "blocked": proposal_data.get("blocked"),
            "no_op": _dict_value(proposal_data.get("no_op_status"), "is_no_op"),
            "blocker_count": len(proposal_data.get("blockers") or []),
            "review_decision": review_data.get("decision"),
            "allowed_next_stage": review_data.get("allowed_next_stage"),
            "finding_count": len(review_data.get("findings") or []),
            "required_change_count": len(review_data.get("required_changes") or []),
        }

    @staticmethod
    def _build_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        build = artifact_by_type.get("build_result")
        if build is None:
            return DashboardProjectionBuilder._empty_section("No build_result artifact found.")
        data = _data(build)
        repair_handoff = data.get("repair_handoff") if isinstance(data.get("repair_handoff"), dict) else {}
        return {
            "status": data.get("status"),
            "build_status": data.get("build_status"),
            "execution_mode": data.get("execution_mode"),
            "error_class": data.get("error_class"),
            "exit_code": data.get("exit_code"),
            "duration_ms": data.get("duration_ms"),
            "error_count": len(data.get("errors") or []),
            "warning_count": len(data.get("warnings") or []),
            "diagnostic_count": len(data.get("diagnostics") or []),
            "expected_artifact_count": len(data.get("expected_artifacts") or []),
            "verified_artifact_count": len(data.get("verified_artifacts") or []),
            "artifact_verification_status": _dict_value(
                data.get("artifact_verification"),
                "status",
            ),
            "repair_handoff_allowed": repair_handoff.get("allowed"),
            "real_build_performed": data.get("real_build_performed"),
            "claims_real_build_success": data.get("claims_real_build_success"),
        }

    @staticmethod
    def _runtime_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        runtime = artifact_by_type.get("runtime_evidence")
        manual = artifact_by_type.get("manual_evidence")
        screenshot = artifact_by_type.get("screenshot_registry")
        if runtime is None and manual is None and screenshot is None:
            return DashboardProjectionBuilder._empty_section(
                "No runtime evidence artifacts found."
            )
        data = _data(runtime or manual)
        screenshot_data = _data(screenshot)
        return {
            "status": data.get("verification_status") or screenshot_data.get("verification_status"),
            "execution_mode": data.get("execution_mode"),
            "evidence_source": data.get("evidence_source"),
            "startup_loaded": data.get("startup_loaded"),
            "gameplay_behavior_verified": data.get("gameplay_behavior_verified"),
            "verification_status": data.get("verification_status"),
            "acceptance_status": data.get("acceptance_status")
            or screenshot_data.get("acceptance_status"),
            "evidence_item_count": len(data.get("evidence_items") or []),
            "source_count": len(data.get("sources") or []),
            "limitation_count": len(data.get("limitations") or []),
            "screenshot_count": screenshot_data.get("screenshot_count"),
            "captioned_count": screenshot_data.get("captioned_count"),
            "claims_real_runtime_success": data.get("claims_real_runtime_success")
            if data
            else screenshot_data.get("claims_real_runtime_success"),
        }

    @staticmethod
    def _repair_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        repair = artifact_by_type.get("repair_plan")
        if repair is None:
            return DashboardProjectionBuilder._empty_section("No repair_plan artifact found.")
        data = _data(repair)
        return {
            "status": "blocked" if data.get("blockers") else "available",
            "repair_id": data.get("repair_id"),
            "failure_source": data.get("failure_source"),
            "failure_class": data.get("failure_class"),
            "evidence_ref_count": len(data.get("evidence_refs") or []),
            "hypothesis_count": len(data.get("root_cause_hypotheses") or []),
            "minimal_change_count": len(data.get("minimal_changes") or []),
            "retry_count": data.get("retry_count"),
            "retry_budget": data.get("retry_budget"),
            "retry_budget_remaining": data.get("retry_budget_remaining"),
            "retry_exhausted": data.get("retry_exhausted"),
            "requires_new_evidence": data.get("requires_new_evidence"),
            "allowed_next_stage": data.get("allowed_next_stage"),
            "blocker_count": len(data.get("blockers") or []),
        }

    @staticmethod
    def _evaluation_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        evaluation = artifact_by_type.get("eval_report")
        if evaluation is None:
            return DashboardProjectionBuilder._empty_section("No eval_report artifact found.")
        data = _data(evaluation)
        evaluation_gate = (
            data.get("evaluation_gate")
            if isinstance(data.get("evaluation_gate"), dict)
            else {}
        )
        gate_summary = (
            evaluation_gate.get("summary")
            if isinstance(evaluation_gate.get("summary"), dict)
            else {}
        )
        return {
            "status": data.get("verdict"),
            "execution_mode": data.get("execution_mode"),
            "workflow_completion_status": data.get("workflow_completion_status"),
            "requested_behavior_status": data.get("requested_behavior_status"),
            "verdict": data.get("verdict"),
            "claims_real_success": data.get("claims_real_success"),
            "evaluation_gate_status": evaluation_gate.get("status"),
            "evaluation_gate_failed": gate_summary.get("failed"),
            "check_count": len(data.get("checks") or []),
            "input_runtime_status": data.get("input_runtime_status"),
            "input_acceptance_status": data.get("input_acceptance_status"),
        }

    @staticmethod
    def _git_summary(artifact_by_type: dict[str, dict[str, Any]]) -> dict[str, Any]:
        record = artifact_by_type.get("git_change_record")
        if record is None:
            return DashboardProjectionBuilder._empty_section(
                "No git_change_record artifact found."
            )
        data = _data(record)
        dirty_tree_policy = (
            data.get("dirty_tree_policy")
            if isinstance(data.get("dirty_tree_policy"), dict)
            else {}
        )
        proposal_binding = (
            data.get("proposal_binding")
            if isinstance(data.get("proposal_binding"), dict)
            else {}
        )
        working_tree = (
            data.get("working_tree")
            if isinstance(data.get("working_tree"), dict)
            else {}
        )
        return {
            "status": dirty_tree_policy.get("status") or "unknown",
            "base_revision": data.get("base_revision"),
            "commit_sha": data.get("commit_sha"),
            "changed_file_count": len(data.get("changed_files") or []),
            "diff_hash": data.get("diff_hash"),
            "proposal_id": proposal_binding.get("proposal_id"),
            "proposal_base_revision": proposal_binding.get("base_revision"),
            "working_tree_dirty": working_tree.get("is_dirty"),
            "blocks_proposal_binding": dirty_tree_policy.get(
                "blocks_proposal_binding"
            ),
            "rollback_hint": data.get("rollback_hint"),
        }

    @staticmethod
    def _artifact_dag(envelopes: list[dict[str, Any]]) -> dict[str, Any]:
        nodes = []
        edges = []
        present_types = {
            str(envelope.get("artifact_type"))
            for envelope in envelopes
            if envelope.get("artifact_type")
        }
        for envelope in envelopes:
            artifact_type = str(envelope.get("artifact_type"))
            nodes.append(
                {
                    "artifact_type": artifact_type,
                    "status": envelope.get("status"),
                    "producer_stage": envelope.get("producer_stage"),
                }
            )
            for upstream in envelope.get("inputs", {}).get("upstream_artifacts", []):
                if not isinstance(upstream, dict):
                    continue
                upstream_type = upstream.get("artifact_type")
                edges.append(
                    {
                        "from": upstream_type,
                        "to": artifact_type,
                        "status": "present" if upstream_type in present_types else "missing",
                    }
                )
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _collect_blockers(envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        blockers = []
        for envelope in envelopes:
            artifact_type = envelope.get("artifact_type")
            data = envelope.get("data") if isinstance(envelope.get("data"), dict) else {}
            for blocker in list(envelope.get("blockers") or []) + list(data.get("blockers") or []):
                if isinstance(blocker, dict):
                    blockers.append(
                        {
                            "artifact_type": artifact_type,
                            "stage": blocker.get("stage"),
                            "code": blocker.get("code"),
                            "message": _sanitize_value(
                                blocker.get("message"),
                                key_path=("message",),
                            ),
                        }
                    )
        return blockers

    @staticmethod
    def _empty_section(message: str) -> dict[str, Any]:
        return {
            "status": "empty",
            "message": message,
        }


def _data(envelope: dict[str, Any] | None) -> dict[str, Any]:
    if not envelope:
        return {}
    data = envelope.get("data")
    return data if isinstance(data, dict) else {}


def _dict_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _sanitize_value(value: Any, *, key_path: tuple[str, ...]) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): (
                _REDACTED
                if _is_sensitive_key(str(key).lower())
                else _sanitize_value(item, key_path=key_path + (str(key),))
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return {
            "count": len(value),
            "items": [
                _sanitize_value(item, key_path=key_path)
                for item in value[:5]
            ],
        }
    if isinstance(value, str):
        return _sanitize_string(value, key_path=key_path)
    return value


def _sanitize_string(value: str, *, key_path: tuple[str, ...]) -> str:
    key = key_path[-1].lower() if key_path else ""
    if _is_sensitive_key(key):
        return _REDACTED
    if _is_path_key(key) or _looks_like_path(value):
        return _redact_path(value)
    if len(value) > _MAX_STRING_CHARS:
        return value[:_MAX_STRING_CHARS] + _TRUNCATED_SUFFIX
    return value


def _is_sensitive_key(key: str) -> bool:
    return any(part in key for part in _SENSITIVE_KEY_PARTS)


def _is_path_key(key: str) -> bool:
    return any(part == key or key.endswith(f"_{part}") for part in _PATH_KEY_PARTS)


def _looks_like_path(value: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE_PATH.match(value) or _POSIX_ABSOLUTE_PATH.match(value))


def _redact_path(value: str) -> str:
    name = Path(value).name
    if not name or name == value:
        return "[redacted_path]"
    return f"[redacted_path:{name}]"
