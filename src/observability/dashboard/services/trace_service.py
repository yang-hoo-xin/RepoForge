"""TraceService – read and parse traces from logs/traces.jsonl.

Provides a typed, filterable interface over the raw JSONL trace log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path

logger = logging.getLogger(__name__)

# Default path to the traces file (absolute, CWD-independent)
DEFAULT_TRACES_PATH = resolve_path("logs/traces.jsonl")

DEFAULT_MAX_STRING_CHARS = 2000
MAX_PRODUCT_RECORD_BYTES = 16 * 1024 * 1024
_REDACTED = "[redacted]"
_TRUNCATED_SUFFIX = "...[truncated]"
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
    "file_path",
    "path",
    "source_path",
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_POSIX_ABSOLUTE_PATH = re.compile(r"^/(?:[^/\0]+)(?:/[^/\0]+)*$")


class TraceService:
    """Read-only service for querying recorded traces.

    Args:
        traces_path: Path to the JSONL file.  Defaults to
            ``logs/traces.jsonl``.
    """

    def __init__(
        self,
        traces_path: Optional[str | Path] = None,
        *,
        include_raw: bool = False,
        safe_mode: Optional[bool] = None,
        max_string_chars: int = DEFAULT_MAX_STRING_CHARS,
    ) -> None:
        self.traces_path = Path(traces_path) if traces_path else DEFAULT_TRACES_PATH
        self.include_raw = include_raw and safe_mode is not True
        self.max_string_chars = max(32, max_string_chars)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_runs(self, *, allowed_roots, cursor=0, limit=100):
        from .run_records import RunRecordBrowser, _sanitize
        return _sanitize(RunRecordBrowser(allowed_roots).discover(cursor=cursor, limit=limit), [])

    def inspect_run(self, *, allowed_roots, run_id, record_cursor=0, record_limit=200):
        from .run_records import RunRecordBrowser, _sanitize
        return _sanitize(RunRecordBrowser(allowed_roots).inspect(run_id, record_cursor=record_cursor, record_limit=record_limit), [])

    def read_record(self, *, allowed_roots, record_id, offset=0, limit=65536, expected_sha256=None):
        from .run_records import RunRecordBrowser
        return RunRecordBrowser(allowed_roots).read_record(record_id, offset=offset, limit=limit, expected_sha256=expected_sha256)

    def verify_run_cloud(self, *, allowed_roots, run_id, trace_id, expected_source_fingerprint):
        """Explicit readback only; no upload or product-state changes.

        为什么要有它：云端核验必须由人主动触发且绑定当前原件，打开本地页面不能暗中联网或重写结果。
        """
        from .run_records import RunRecordBrowser, _sanitize
        return _sanitize(RunRecordBrowser(allowed_roots).verify_cloud(run_id, trace_id,
            expected_source_fingerprint=expected_source_fingerprint), [])

    def list_traces(
        self,
        trace_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return traces in reverse-chronological order.

        Args:
            trace_type: Filter by ``trace_type`` field (e.g.
                ``"ingestion"`` or ``"query"``).  ``None`` = all.
            limit: Maximum number of traces to return.

        Returns:
            List of trace dicts (newest first).
        """
        traces = self._load_all()

        if trace_type:
            traces = [t for t in traces if t.get("trace_type") == trace_type]

        # Newest first
        traces.sort(key=lambda t: t.get("started_at", ""), reverse=True)

        return [self._sanitize_trace(t) for t in traces[:limit]]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single trace by its ``trace_id``.

        Returns:
            Trace dict, or ``None`` if not found.
        """
        for t in self._load_all():
            if t.get("trace_id") == trace_id:
                return self._sanitize_trace(t)
        # Pre-manifest product failures use the same local trace storage but a
        # complete immutable execution file, avoiding an unbounded JSONL join.
        if isinstance(trace_id, str) and re.fullmatch(r"[a-f0-9]{32}", trace_id):
            directory = (self.traces_path.parent / "execution").resolve()
            path = (directory / (trace_id + ".json")).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Retained execution escapes the local trace directory.")
            if path.is_file():
                document = json.loads(path.read_text(encoding="utf-8"))
                summary = document.get("execution_trace") if isinstance(document, dict) else None
                if not isinstance(summary, dict) or summary.get("trace_id") != trace_id:
                    raise ValueError("Retained execution identity does not match the requested trace.")
                return self._sanitize_trace(document)
        return None

    def get_stage_timings(self, trace: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stage timings from a trace.

        Returns:
            List of dicts with keys: stage_name, elapsed_ms, data.
            Ordered by appearance.
        """
        stages = trace.get("stages", [])
        timings: List[Dict[str, Any]] = []
        for s in stages:
            # The raw stage dict has: stage, timestamp, data (dict), elapsed_ms
            # Extract the inner 'data' dict directly rather than flattening
            stage_data = s.get("data")
            if stage_data is None:
                stage_data = {
                    key: value
                    for key, value in s.items()
                    if key not in {"stage", "timestamp", "elapsed_ms"}
                }
            if not isinstance(stage_data, dict):
                stage_data = {}
            timings.append(
                {
                    "stage_name": s.get("stage"),
                    "elapsed_ms": s.get("elapsed_ms", 0),
                    "data": stage_data,
                }
            )
        return timings

    def inspect_product_run(
        self,
        *,
        allowed_root: str | Path,
        run_root: str | Path,
        run_id: str,
        continuation_root: str | Path | None = None,
        include_local_raw: bool = False,
    ) -> dict[str, Any]:
        """Project one explicitly selected Product run and optional continuation.

        Only local files under *allowed_root* are accepted. Raw request/response
        bodies are read only from hash-bound ``local_detail_refs`` and only when
        the caller explicitly opts in.
        """
        if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
            raise ValueError("Product run_id is invalid.")
        allowed = self._existing_directory(allowed_root, label="Allowed Runs Root")
        original = self._selected_directory(run_root, allowed=allowed, label="Run Root", allow_equal=True)
        if original.name != run_id:
            raise ValueError("Run Root basename does not match run_id.")
        roots = [("original", original)]
        if continuation_root is not None and str(continuation_root).strip():
            continuation = self._selected_directory(
                continuation_root, allowed=original / "continuations", label="Continuation Root",
            )
            roots.append(("continuation", continuation))

        segments = [self._product_segment(
            kind, root, run_id, include_local_raw, selected_run_root=original,
        ) for kind, root in roots]
        calls = []
        for segment in segments:
            for call in segment["calls"]:
                calls.append({**call, "call_number": len(calls) + 1})
        latest = segments[-1]
        original_budget = segments[0].get("model_call_budget") or {}
        latest_budget = latest.get("model_call_budget") or {}
        findings = (latest_budget.get("by_stage") or {}).get("requirement_findings") or {}
        projection = {
            "run_id": run_id,
            "read_only": True,
            "query_execution": {"model_calls_performed": 0, "writes_performed": False},
            "segments": segments,
            "calls": calls,
            "call_accounting": {
                "original": self._budget_counts(original_budget),
                "latest": self._budget_counts(latest_budget),
                "latest_requirement_findings": self._budget_counts(findings),
                "actual_generation_records": len(calls),
                "counting_rule": "Each generation with its parent model.request is one call.",
            },
            "diagnostic_timeline": self._diagnostic_timeline(segments),
            "schema_failure_drilldown": self._schema_failure_drilldown(segments),
            "blind_spots": [
                "request.json is an application-layer request record, not a byte-for-byte HTTP packet.",
                "response_format is a service output constraint; it is displayed separately from model messages and does not prove the model read the full schema text.",
                "Ollama Schema-to-grammar handling was not recorded; support is unknown from these run records.",
                "Missing parse, schema, business-validation, reservation, invocation, or transport facts remain unknown or not reached.",
            ],
        }
        # Scope: all display fields, including summaries and explicitly selected raw bodies.
        # 为什么要有它：完整性检查读取原字节，凭据保护只作用于最终展示；摘要和恢复字段不能绕过相同保护。
        from src.mcp_server.safety import Redactor

        from .run_records import _sanitize
        notes: list[str] = []
        sanitized = _sanitize(projection, notes)
        protected = Redactor.configured()(sanitized)
        if protected != sanitized:
            notes.append("credential_assignment_or_configured_value")
        protected["redaction"] = {
            "applied": bool(notes), "count": len(notes), "notes": sorted(set(notes)),
            "scope": "display_only_original_bytes_unchanged",
            "coverage": "entire_product_run_display_including_selected_raw_records",
            "validation_basis": "original_bytes_before_display_redaction",
            "original_files_modified": False,
        }
        return protected

    @staticmethod
    def _budget_counts(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"maximum": None, "consumed": None, "remaining": None}
        return {key: value.get(key) if type(value.get(key)) is int else None
                for key in ("maximum", "consumed", "remaining")}

    @staticmethod
    def _existing_directory(value: str | Path, *, label: str) -> Path:
        path = Path(value)
        if not path.is_absolute() or path.is_symlink():
            raise ValueError(f"{label} must be an absolute, non-symlink directory.")
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"{label} is not a directory.")
        return resolved

    @classmethod
    def _selected_directory(cls, value: str | Path, *, allowed: Path, label: str, allow_equal: bool = False) -> Path:
        path = cls._existing_directory(value, label=label)
        allowed = allowed.resolve(strict=True)
        # Scope: an explicitly authorized run directory can itself be selected; never broaden its parent.
        # 为什么要有它：仅授权一个任务目录是更窄的权限，不能为打开它反而要求授权父目录。
        if path == allowed and not allow_equal or not path.is_relative_to(allowed):
            raise ValueError(f"{label} is outside the allowed root.")
        return path

    @staticmethod
    def _read_product_json(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
        selected = path
        if not selected.is_absolute():
            selected = root / selected
        if selected.is_symlink():
            raise ValueError("Product record links are not allowed.")
        selected = selected.resolve(strict=True)
        if not selected.is_relative_to(root):
            raise ValueError("Product record escapes its selected run root.")
        info = selected.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PRODUCT_RECORD_BYTES:
            raise ValueError("Product record size or type is invalid.")
        raw = selected.read_bytes()
        if len(raw) > MAX_PRODUCT_RECORD_BYTES:
            raise ValueError("Product record is too large.")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Product record must contain a JSON object.")
        return value, hashlib.sha256(raw).hexdigest()

    def _product_segment(self, kind: str, root: Path, run_id: str, include_raw: bool,
                         *, selected_run_root: Path) -> dict[str, Any]:
        if kind == "continuation":
            try:
                (root / "run_manifest.json").lstat()
            except FileNotFoundError:
                return self._result_only_continuation(root, run_id, selected_run_root=selected_run_root)
        manifest, manifest_hash = self._read_product_json(root / "run_manifest.json", root=root)
        result, result_hash = self._read_product_json(root / "run_result.json", root=root)
        if manifest.get("run_id") != run_id or result.get("run_id") != run_id:
            raise ValueError("Product run record identity does not match run_id.")
        if kind == "original":
            terminal = manifest.get("terminal_record")
            if not isinstance(terminal, dict) or terminal.get("sha256") != result_hash:
                raise ValueError("Original run_result is not bound by the manifest terminal reference.")
        elif not root.is_relative_to(root.parents[1] / "continuations"):
            raise ValueError("Continuation Root is outside the original continuations directory.")
        trace = manifest.get("execution_trace") if isinstance(manifest.get("execution_trace"), dict) else {}
        observations = manifest.get("execution_observations")
        if not isinstance(observations, dict) or any(not isinstance(row, dict) for row in observations.values()):
            raise ValueError("Product execution observations are missing or invalid.")
        if any(row.get("trace_id") != trace.get("trace_id") for row in observations.values()):
            raise ValueError("Product observation trace identity mismatch.")
        ordered = sorted(observations.values(), key=lambda row: (row.get("started_at") or "", row.get("id") or ""))
        calls = []
        by_id = {row.get("id"): row for row in ordered}
        for generation in (row for row in ordered if row.get("kind") == "generation"):
            parent = by_id.get(generation.get("parent_id"))
            if not isinstance(parent, dict) or parent.get("name") != "model.request":
                raise ValueError("Generation has no bound parent model.request observation.")
            calls.append(self._product_call(kind, root, parent, generation, include_raw))
        budget = result.get("model_call_budget") if isinstance(result.get("model_call_budget"), dict) else (
            manifest.get("model_call_budget") if isinstance(manifest.get("model_call_budget"), dict) else None)
        diagnostics = []
        for row in ordered:
            audit = (row.get("attributes") or {}).get("requirement_findings_attempt_audit")
            if isinstance(audit, list):
                diagnostics.extend(self._native_audit_events(kind, row, audit))
        if kind == "continuation":
            cause = manifest.get("initial_findings_cause_chain")
            if isinstance(cause, dict):
                diagnostics.extend(self._offline_cause_events(cause))
        blockers = [{key: row.get(key) for key in ("owner", "stage", "code", "message", "source_node")}
                    for row in result.get("blockers", []) if isinstance(row, dict)]
        planning = self._planning_decision(root, ordered)
        evidence_query_count = sum(row.get("name") == "evidence.source_query" for row in ordered)
        patch_receipts = result.get("patch_execution_receipts")
        patch_receipt_count = len(patch_receipts) if isinstance(patch_receipts, list) else None
        initial_patch = ((budget or {}).get("by_stage") or {}).get("initial_patch") or {}
        reassessment = self._initial_findings_reassessment(
            result, ordered, root=root, selected_run_root=selected_run_root,
        )
        return {
            "segment": kind,
            "record_format": "product_manifest_and_result",
            "manifest_status": "recorded",
            "run_root": str(root),
            "manifest_ref": {"path": str(root / "run_manifest.json"), "sha256": manifest_hash},
            "result_ref": {"path": str(root / "run_result.json"), "sha256": result_hash,
                           "binding": "manifest_terminal_sha256" if kind == "original" else "locally_read_unmodified_record"},
            "trace": {key: trace.get(key) for key in ("trace_id", "session_id", "environment", "mode", "cloud_readback")},
            "status": manifest.get("status") or result.get("status") or "unknown",
            "model_call_count": result.get("model_call_count", manifest.get("model_call_count")),
            "current_invocation_model_call_count": result.get(
                "current_invocation_model_call_count", manifest.get("current_invocation_model_call_count")),
            "model_call_budget": budget,
            "calls": calls,
            "diagnostics": diagnostics,
            "blockers": blockers,
            "post_findings_gate": {
                "findings_status": "accepted" if any(row.get("status") == "accepted" for row in diagnostics) else "not_accepted",
                "supplemental_query_observation_count": evidence_query_count,
                "supplemental_query_status": "not_observed_in_selected_continuation" if evidence_query_count == 0 else "observed",
                "planning_model_selection": planning,
                "patch_execution_receipt_count": patch_receipt_count,
                "initial_patch_budget_consumed": initial_patch.get("consumed") if type(initial_patch.get("consumed")) is int else None,
            },
            "initial_findings_reassessment": reassessment,
            "runtime_identity": {
                "ollama_server_version": manifest.get("ollama_server_version"),
                "model": manifest.get("model"),
                "local_model_identity": manifest.get("local_model_identity")
                    if isinstance(manifest.get("local_model_identity"), dict) else None,
                "provider_adapter": (manifest.get("model_admission") or {}).get("provider_adapter")
                    if isinstance(manifest.get("model_admission"), dict) else None,
                "llama_cpp_version": None,
                "llama_cpp_version_status": "unknown_not_recorded",
            },
            "observation_count": len(ordered),
        }

    @classmethod
    def _continuation_reference(cls, reference: Any, *, root: Path, expected: Path) -> tuple[dict[str, Any], dict[str, str]]:
        """Read a recorded exact reference inside the caller's already selected root."""
        if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
                or not isinstance(reference.get("sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", reference["sha256"]) is None):
            raise ValueError("Continuation reference is missing its exact path or SHA256.")
        path = Path(reference["path"])
        if not path.is_absolute() or path.resolve(strict=True) != expected.resolve(strict=True):
            raise ValueError("Continuation reference does not match the selected record.")
        for part in (path, *path.parents):
            if part == root:
                break
            attributes = part.lstat()
            if part.is_symlink() or getattr(attributes, "st_file_attributes", 0) & 0x400:
                raise ValueError("Continuation record links are not allowed.")
        document, digest = cls._read_product_json(path, root=root)
        if digest != reference["sha256"]:
            raise ValueError("Continuation record SHA256 mismatch.")
        return document, {"path": str(path.resolve()), "sha256": digest}

    def _result_only_continuation(self, root: Path, run_id: str, *, selected_run_root: Path) -> dict[str, Any]:
        """Show a hash-bound recovery result without borrowing the parent's trace.

        为什么要有它：恢复可以复用候选而不产生模型调用；缺失manifest不能伪造，也不能把历史调用重复计入恢复。
        """
        if not root.is_relative_to(selected_run_root / "continuations"):
            raise ValueError("Continuation Root is outside the selected run's continuations directory.")
        result, result_hash = self._read_product_json(root / "run_result.json", root=root)
        if result.get("run_id") != run_id:
            raise ValueError("Product run record identity does not match run_id.")
        result_ref = {"path": str(root / "run_result.json"), "sha256": result_hash}
        view = {"segment": "continuation", "run_root": str(root), "record_format": "result_without_manifest",
            "recorded_entry": result.get("entry"), "manifest_ref": None, "manifest_status": "not_recorded",
            "result_ref": {**result_ref, "binding": "locally_read_identity_not_yet_confirmed"},
            "identity_status": "unconfirmed", "reported_status": result.get("status"), "status": "unconfirmed",
            "trace": {}, "trace_status": "not_recorded", "observation_count": 0, "calls": [], "diagnostics": [],
            "model_call_count": None, "current_invocation_model_call_count": None, "model_call_budget": None,
            "blockers": [], "post_findings_gate": {}, "initial_findings_reassessment": {},
            "runtime_identity": {"llama_cpp_version_status": "unknown_not_recorded"},
            "counting_scope": "No parent generation records are attributed to this continuation; missing trace is not proof of zero calls."}
        required = ("parent_run_id", "parent_manifest_ref", "parent_result_ref", "continuation_claim_ref", "fresh_result_ref")
        missing = [key for key in required if result.get(key) is None]
        for filename in ("completion.json", "selection.json", "graph_result.json"):
            try:
                (root / filename).lstat()
            except FileNotFoundError:
                missing.append(filename)
        if missing:
            view["missing_identity_fields"] = missing
            return view
        if result["parent_run_id"] != run_id:
            raise ValueError("Continuation parent run identity mismatch.")
        for key, expected in (("run_root", selected_run_root), ("continuation_root", root)):
            if key in result and Path(result[key]).resolve(strict=True) != expected:
                raise ValueError("Continuation declared root mismatch.")
        parent_manifest, manifest_ref = self._continuation_reference(result["parent_manifest_ref"],
            root=selected_run_root, expected=selected_run_root / "run_manifest.json")
        parent_result, parent_ref = self._continuation_reference(result["parent_result_ref"],
            root=selected_run_root, expected=selected_run_root / "run_result.json")
        if parent_manifest.get("run_id") != run_id or parent_result.get("run_id") != run_id:
            raise ValueError("Continuation parent record identity mismatch.")
        claim, claim_ref = self._continuation_reference(result["continuation_claim_ref"], root=root, expected=root / "selection.json")
        fresh, fresh_ref = self._continuation_reference(result["fresh_result_ref"], root=root, expected=root / "graph_result.json")
        if fresh.get("run_id") != run_id:
            raise ValueError("Continuation graph result identity mismatch.")
        for key, expected in (("parent_manifest_ref", selected_run_root / "run_manifest.json"),
                              ("parent_result_ref", selected_run_root / "run_result.json")):
            self._continuation_reference(claim.get(key), root=selected_run_root, expected=expected)
            if claim[key]["sha256"] != result[key]["sha256"]:
                raise ValueError("Continuation selection differs from result parent binding.")
        completion, completion_hash = self._read_product_json(root / "completion.json", root=root)
        for key, expected, digest in (("result_ref", root / "run_result.json", result_hash),
                ("claim_ref", root / "selection.json", claim_ref["sha256"]),
                ("fresh_result_ref", root / "graph_result.json", fresh_ref["sha256"])):
            self._continuation_reference(completion.get(key), root=root, expected=expected)
            if completion[key]["sha256"] != digest:
                raise ValueError("Continuation completion differs from result binding.")
        counters = [result[key] for key in ("current_invocation_model_call_count", "model_calls_performed") if key in result]
        if any(type(value) is not int or value < 0 for value in counters) or len(set(counters)) > 1:
            raise ValueError("Continuation current-invocation model counts conflict.")
        view.update(record_format="hash_bound_continuation_result", identity_status="verified_parent_and_completion_refs",
            result_ref={**result_ref, "binding": "completion_sha256_and_selected_parent_refs"}, status=result.get("status") or "unknown",
            model_call_count=result.get("model_call_count") if type(result.get("model_call_count")) is int else None,
            current_invocation_model_call_count=counters[0] if counters else None,
            model_call_budget=result.get("model_call_budget") if isinstance(result.get("model_call_budget"), dict) else None,
            new_provider_sends=result.get("new_provider_sends") if type(result.get("new_provider_sends")) is int else None,
            identity_refs={"parent_manifest": manifest_ref, "parent_result": parent_ref, "selection": claim_ref,
                "fresh_result": fresh_ref, "completion": {"path": str(root / "completion.json"), "sha256": completion_hash}},
            blockers=[{key: row.get(key) for key in ("owner", "stage", "code", "message", "source_node")}
                for row in result.get("blockers", []) if isinstance(row, dict)])
        return view

    @classmethod
    def _initial_findings_reassessment(
        cls,
        result: dict[str, Any],
        observations: list[dict[str, Any]],
        *,
        root: Path,
        selected_run_root: Path,
    ) -> dict[str, Any]:
        """Project the recorded query→Findings chain without executing retrieval."""
        records: list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                reassessment = value.get("reassessment")
                if (isinstance(reassessment, dict)
                        and isinstance(reassessment.get("queries"), list)
                        and isinstance(reassessment.get("per_query_execution"), list)):
                    dependency = value.get("dependency_api_supplement")
                    if not isinstance(dependency, dict):
                        augmentation = value.get("source_augmentation")
                        augmentation_budget = augmentation.get("budget_summary") if isinstance(augmentation, dict) else None
                        dependency = augmentation_budget.get("dependency_api_supplement") if isinstance(augmentation_budget, dict) else None
                    initial = value.get("initial_findings_reassessment")
                    records.append((reassessment, dependency if isinstance(dependency, dict) else None,
                                    initial if isinstance(initial, dict) else None))
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(result)
        if not records:
            return {"status": "not_recorded"}

        def identity_matches(record: tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]) -> bool:
            reassessment, _, initial = record
            prior = initial.get("prior_evidence_ref") if isinstance(initial, dict) else None
            return (isinstance(prior, dict) and isinstance(prior.get("artifact_hash"), str)
                    and reassessment.get("old_evidence_hash") == prior.get("artifact_hash"))

        matching = [record for record in records if identity_matches(record)]
        reassessment, dependency, initial = (matching or records)[-1]
        dependency = dependency or {}
        execution_by_id = {row.get("query_id"): row for row in reassessment["per_query_execution"]
                           if isinstance(row, dict) and isinstance(row.get("query_id"), str)}
        groups = []
        for row in reassessment["queries"]:
            if not isinstance(row, dict):
                continue
            metadata_queries = row.get("metadata_queries") if isinstance(row.get("metadata_queries"), list) else None
            executed = execution_by_id.get(row.get("query_id"), {})
            groups.append({
                "query_id": row.get("query_id"),
                "requirement_id": row.get("requirement_id"),
                "source_query": row.get("query"),
                "gap_ids": row.get("gap_ids") if isinstance(row.get("gap_ids"), list) else None,
                "current_fact_needed": row.get("current_fact_needed"),
                "modification_impact": row.get("modification_impact"),
                "authorized_source": row.get("authorized_source"),
                "metadata_queries": metadata_queries,
                "metadata_query_count": len(metadata_queries) if metadata_queries is not None else None,
                "source_retrieval_status": executed.get("retrieval_status"),
                "source_evidence_ids": executed.get("evidence_ids")
                    if isinstance(executed.get("evidence_ids"), list) else [],
            })

        metadata_observations = []
        for row in observations:
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
            if attrs.get("tool") == "search_dotnet_metadata" or attrs.get("retrieval_method") == "search_dotnet_metadata":
                metadata_observations.append({"observation_id": row.get("id"), "name": row.get("name"),
                                              "status": row.get("status")})

        prior_ids: set[str] = set()
        prior_status = "unknown_missing_or_unbound"
        prior = initial.get("prior_evidence_ref") if isinstance(initial, dict) else None
        if isinstance(prior, dict) and isinstance(prior.get("path"), str) and isinstance(prior.get("artifact_hash"), str):
            try:
                prior_record, _ = cls._read_product_json(Path(prior["path"]), root=selected_run_root)
                items = (prior_record.get("data") or {}).get("items")
                if prior_record.get("artifact_hash") == prior["artifact_hash"] and isinstance(items, list):
                    prior_ids = {item["evidence_id"] for item in items if isinstance(item, dict)
                                 and isinstance(item.get("evidence_id"), str)}
                    prior_status = "verified_artifact_identity"
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        findings_generation = next((row for row in reversed(observations)
            if row.get("kind") == "generation"
            and isinstance(row.get("attributes"), dict)
            and row["attributes"].get("model_role") == "requirement_findings"), None)
        input_rows: list[dict[str, Any]] = []
        output_findings: list[dict[str, Any]] = []
        request_integrity = response_integrity = "unknown_missing_or_unbound"
        if isinstance(findings_generation, dict):
            attrs = findings_generation["attributes"]
            refs = attrs.get("local_detail_refs") if isinstance(attrs.get("local_detail_refs"), dict) else {}
            for role in ("request", "response"):
                ref = refs.get(role)
                if not isinstance(ref, dict) or cls._local_ref_summary(ref, root=root)["integrity"] != "verified_bytes":
                    continue
                try:
                    value, _ = cls._read_product_json(Path(ref["path"]), root=root)
                    if role == "request":
                        request_integrity = "verified_bytes"
                        messages = value.get("messages") if isinstance(value.get("messages"), list) else []
                        user_content = next((item.get("content") for item in messages
                            if isinstance(item, dict) and item.get("role") == "user"
                            and isinstance(item.get("content"), str)), None)
                        payload = json.loads(user_content) if isinstance(user_content, str) else None
                        rows = payload.get("source_observations") if isinstance(payload, dict) else None
                        input_rows = rows if isinstance(rows, list) else []
                    else:
                        response_integrity = "verified_bytes"
                        rows = value.get("findings")
                        output_findings = rows if isinstance(rows, list) else []
                except (OSError, ValueError, json.JSONDecodeError):
                    continue

        input_by_id = {row.get("evidence_id"): row for row in input_rows
                       if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)}
        new_ids = sorted(set(input_by_id) - prior_ids) if prior_status == "verified_artifact_identity" else []
        cited = {evidence_id for row in output_findings if isinstance(row, dict)
                 for evidence_id in row.get("evidence_ids", []) if isinstance(evidence_id, str)}
        new_inputs = [{
            "evidence_id": evidence_id,
            "source_type": input_by_id[evidence_id].get("source_type"),
            "source_path": input_by_id[evidence_id].get("source_path"),
            "symbol": input_by_id[evidence_id].get("symbol"),
            "span": input_by_id[evidence_id].get("span"),
            "entered_findings_input": True,
            "cited_by_findings_output": evidence_id in cited,
        } for evidence_id in new_ids]

        def transport(role: str) -> dict[str, Any] | None:
            row = next((item for item in reversed(observations)
                if item.get("kind") == "generation"
                and isinstance(item.get("attributes"), dict)
                and item["attributes"].get("model_role") == role), None)
            if row is None:
                return None
            attrs = row["attributes"]
            return {key: attrs.get(key) for key in (
                "transport_started", "response_received", "response_complete", "http_status", "response_origin")}

        return {
            "status": "recorded",
            "stop": ({key: result["initial_findings_reassessment_stop"].get(key) for key in (
                "code", "cycle", "prior_item_count", "new_item_count", "new_evidence_ids",
                "sufficiency_status", "not_started", "fact_boundary",
            )} if isinstance(result.get("initial_findings_reassessment_stop"), dict) else None),
            "query_group_count": len(groups),
            "query_groups": groups,
            "metadata_supplement": {
                "query_source": dependency.get("query_source"),
                "status": dependency.get("status"),
                "reason": dependency.get("reason"),
                "metadata_query_count": dependency.get("metadata_query_count"),
                "tool_calls_completed": dependency.get("tool_calls_completed"),
                "model_calls_performed": dependency.get("model_calls_performed"),
                "visible_metadata": dependency.get("visible_metadata")
                    if isinstance(dependency.get("visible_metadata"), dict) else None,
                "recorded_metadata_observation_count": len(metadata_observations),
                "recorded_metadata_observations": metadata_observations,
                "observation_count_basis": "tool_or_retrieval_method_equals_search_dotnet_metadata",
            },
            "transport_boundaries": {
                "initial_findings_queries": transport("initial_findings_queries"),
                "requirement_findings": transport("requirement_findings"),
                "metadata_supplement": "not_requested" if dependency.get("status") == "not_requested" else "see_recorded_status",
            },
            "evidence_revision": {
                "prior_identity_status": prior_status,
                "prior_item_count": reassessment.get("prior_item_count"),
                "recorded_new_item_count": reassessment.get("new_item_count"),
                "recorded_new_evidence_ids": reassessment.get("new_evidence_ids")
                    if isinstance(reassessment.get("new_evidence_ids"), list) else None,
                "findings_input_item_count": len(input_by_id),
                "request_integrity": request_integrity,
                "response_integrity": response_integrity,
                "new_inputs": new_inputs,
            },
            "findings_outcomes": [{
                "relation_index": row.get("relation_index"),
                "relationship": row.get("relationship"),
                "unresolved_gap_count": len(row.get("unresolved_gaps", []))
                    if isinstance(row.get("unresolved_gaps"), list) else None,
            } for row in output_findings if isinstance(row, dict)],
            "terminal": {
                "status": result.get("status"),
                "blockers": [{key: row.get(key) for key in ("code", "message", "source_node")}
                             for row in result.get("blockers", []) if isinstance(row, dict)],
            },
            "scope": "recorded_request_input_and_execution_facts_not_model_attention_or_query_quality",
        }

    def _planning_decision(self, root: Path, observations: list[dict[str, Any]]) -> dict[str, Any] | None:
        row = next((item for item in observations if item.get("name") == "planning.decision"), None)
        if row is None:
            return None
        attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
        refs = attrs.get("local_detail_refs") if isinstance(attrs.get("local_detail_refs"), dict) else {}
        ref = refs.get("planning_receipt")
        result = {
            "observation_id": row.get("id"), "status": row.get("status"),
            "started_at": row.get("started_at"), "finished_at": row.get("finished_at"),
            "decision_status": attrs.get("decision_status"),
            "reason_status": attrs.get("reason_status"),
            "receipt": self._local_ref_summary(ref, root=root) if isinstance(ref, dict) else None,
        }
        if not isinstance(ref, dict) or result["receipt"]["integrity"] != "verified_bytes":
            result["receipt_status"] = "unknown_missing_or_unbound"
            return result
        receipt, actual_hash = self._read_product_json(Path(ref["path"]), root=root)
        if actual_hash != ref.get("sha256"):
            raise ValueError("Planning receipt hash changed after verification.")
        result["receipt_status"] = "verified_bytes"
        result["selection"] = {
            key: receipt.get(key) for key in (
                "status", "selection_mode", "provider", "model", "automatic_retry",
                "model_turns_attempted", "model_responses_received", "operator_target_ids_supplied",
                "evidence_payload_sha256", "feature_payload_sha256", "failure",
            )
        }
        return result

    def _product_call(self, segment: str, root: Path, request_row: dict[str, Any],
                      generation: dict[str, Any], include_raw: bool) -> dict[str, Any]:
        attrs = generation.get("attributes") if isinstance(generation.get("attributes"), dict) else {}
        request_attrs = request_row.get("attributes") if isinstance(request_row.get("attributes"), dict) else {}
        details = attrs.get("local_detail_refs") if isinstance(attrs.get("local_detail_refs"), dict) else {}
        references = {role: self._local_ref_summary(ref, root=root) for role, ref in details.items()
                      if role in {"request", "response", "provider_request"} and isinstance(ref, dict)}
        raw = {}
        if include_raw:
            for role, ref in details.items():
                if role in {"request", "response", "provider_request"} and isinstance(ref, dict):
                    raw[role] = self._read_local_detail(ref, root=root, role=role)
        usage = attrs.get("usage") if isinstance(attrs.get("usage"), dict) else None
        return {
            "segment": segment,
            "request_span_id": request_row.get("id"),
            "generation_span_id": generation.get("id"),
            "parent_id": request_row.get("parent_id"),
            "role": attrs.get("model_role") or request_attrs.get("model_role") or "unknown",
            "model": attrs.get("model") or request_attrs.get("model") or "unknown",
            "provider": attrs.get("provider") or request_attrs.get("provider") or "unknown",
            "status": generation.get("status"),
            "started_at": generation.get("started_at"),
            "finished_at": generation.get("finished_at"),
            "elapsed_ms": generation.get("elapsed_ms"),
            "transport": {key: attrs.get(key) if key in attrs else None for key in (
                "transport_started", "response_received", "response_complete", "http_status", "response_origin")},
            "usage": usage,
            "input_payload_sha256": attrs.get("input_payload_sha256"),
            "output_sha256": attrs.get("output_sha256"),
            "provider_body_sha256": attrs.get("provider_body_sha256"),
            "provider_request_sha256": attrs.get("provider_request_sha256"),
            "provider_request_bytes": attrs.get("provider_request_bytes"),
            "provider_request_metadata": attrs.get("provider_request_metadata")
                if isinstance(attrs.get("provider_request_metadata"), dict) else None,
            "local_evidence": references,
            "raw": raw,
            "counting": "one generation plus its parent request span equals one model call",
        }

    @staticmethod
    def _local_ref_summary(ref: dict[str, Any], *, root: Path) -> dict[str, Any]:
        path = Path(str(ref.get("path", "")))
        selected = path.resolve() if path.is_absolute() else (root / path).resolve()
        within = selected.is_relative_to(root)
        actual = None
        if within and selected.is_file() and not selected.is_symlink() and selected.stat().st_size <= MAX_PRODUCT_RECORD_BYTES:
            actual = hashlib.sha256(selected.read_bytes()).hexdigest()
        expected = ref.get("sha256")
        return {"file": selected.name if within else "[outside_selected_run]", "sha256": expected,
                "integrity": "verified_bytes" if isinstance(expected, str) and actual == expected else
                             "outside_selected_run" if not within else "missing_or_hash_mismatch"}

    def _read_local_detail(self, ref: dict[str, Any], *, root: Path, role: str) -> Any:
        summary = self._local_ref_summary(ref, root=root)
        if summary["integrity"] != "verified_bytes":
            raise ValueError("Local detail reference is not hash-bound inside the selected run.")
        path = Path(str(ref["path"])).resolve()
        raw = path.read_bytes()
        if role == "request":
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Application request record must be a JSON object.")
            return {
                "record_kind": "application_layer_request_not_raw_http",
                "messages": self._sanitize_value(value.get("messages"), key_path=("messages",)),
                "response_format": self._sanitize_value(value.get("response_format"), key_path=("response_format",)),
                "other_fields": self._sanitize_value({key: item for key, item in value.items()
                    if key not in {"messages", "response_format"}}, key_path=("other_fields",)),
                "schema_visibility": "service_output_constraint_not_proof_full_schema_was_model_visible",
            }
        if role == "provider_request":
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Native provider request record must be a JSON object.")
            return {
                "record_kind": "native_provider_request_selected_fields",
                "format": self._sanitize_value(value.get("format"), key_path=("format",)),
                "model": value.get("model") if isinstance(value.get("model"), str) else None,
                "stream": value.get("stream") if type(value.get("stream")) is bool else None,
                "think": value.get("think") if type(value.get("think")) is bool else None,
                "messages": "not_displayed_by_trace_drilldown",
                "raw_body": "not_displayed_by_trace_drilldown",
            }
        return {"record_kind": "application_layer_response", "text": self._sanitize_string(
            raw.decode("utf-8"), key_path=("response",))}

    @staticmethod
    def _native_audit_events(segment: str, observation: dict[str, Any], rows: list[Any]) -> list[dict[str, Any]]:
        events = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            diagnostic = row.get("diagnostic") if isinstance(row.get("diagnostic"), dict) else {}
            accepted = row.get("status") == "accepted"
            events.append({
                "segment": segment,
                "origin": "native_recorded_audit",
                "attempt_index": row.get("attempt_index"),
                "status": row.get("status"),
                "error_code": row.get("error_code"),
                "time": observation.get("finished_at"),
                "elapsed_ms": observation.get("elapsed_ms"),
                "validation": {
                    "parsing": "passed" if diagnostic.get("phase") == "response_schema_admission" or accepted else "unknown",
                    "schema": {"status": "passed" if accepted else "failed" if diagnostic else "unknown",
                               "failure_category": diagnostic.get("failure_category"),
                               "json_path": diagnostic.get("json_path"),
                               "schema_keyword": diagnostic.get("schema_keyword"),
                               "allowed_boundary": diagnostic.get("allowed_boundary")},
                    "business": "accepted" if accepted else "not_reached_after_schema_failure" if diagnostic else "unknown",
                },
                "transport_boundary": row.get("call_boundary") if isinstance(row.get("call_boundary"), dict) else "unknown",
            })
        return events

    @staticmethod
    def _offline_cause_events(cause: dict[str, Any]) -> list[dict[str, Any]]:
        events = []
        first = cause.get("first_rejection_reconstruction")
        if isinstance(first, dict):
            diagnostic = first.get("diagnostic") if isinstance(first.get("diagnostic"), dict) else {}
            events.append({"segment": "historical_supplement", "origin": cause.get("reconstruction_origin"),
                "attempt_index": first.get("attempt_index"), "status": first.get("status"),
                "error_code": first.get("error_code"), "time": None, "elapsed_ms": None,
                "validation": {"parsing": "passed_in_offline_replay", "schema": "passed_in_offline_replay",
                               "business": {"status": "rejected", "violated_rules": diagnostic.get("violated_rules"),
                                            "relation_index": diagnostic.get("relation_index")}},
                "history_recorded_live": False})
        if cause.get("correction_payload_prepared") is True:
            events.append({"segment": "historical_supplement", "origin": cause.get("reconstruction_origin"),
                "attempt_index": 2, "status": "prepared_not_sent", "error_code": "PRODUCT_MODEL_CALL_BUDGET_EXHAUSTED",
                "time": None, "elapsed_ms": None, "validation": {"parsing": "not_reached", "schema": "not_reached",
                    "business": "not_reached"}, "request_sent": cause.get("correction_model_request_sent"),
                "request_sent_basis": cause.get("correction_model_request_sent_basis"),
                "historical_transport_boundary": cause.get("historical_correction_boundary") or "unknown",
                "history_recorded_live": False})
        return events

    @staticmethod
    def _diagnostic_timeline(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        native, supplemental = [], []
        for segment in segments:
            for row in segment.get("diagnostics", []):
                (supplemental if row.get("origin") == "offline_exact_input_replay" else native).append(row)
        native.sort(key=lambda row: row.get("time") or "")
        return [*supplemental, *native]

    @classmethod
    def _schema_failure_drilldown(cls, segments: list[dict[str, Any]]) -> dict[str, Any]:
        """Separate five evidence layers without inferring an unrecorded grammar."""
        continuation = next((row for row in segments if row["segment"] == "continuation"), None)
        current_call = next((row for row in (continuation or {}).get("calls", [])
                             if row.get("role") == "requirement_findings_correction"), None)
        current_audit = next((row for row in (continuation or {}).get("diagnostics", [])
                              if row.get("origin") == "native_recorded_audit"), None)
        request = (current_call or {}).get("raw", {}).get("request")
        provider_request = (current_call or {}).get("raw", {}).get("provider_request")
        response = (current_call or {}).get("raw", {}).get("response")
        schema = request.get("response_format") if isinstance(request, dict) else None
        schema_summary = cls._schema_summary(schema)
        source_path = Path(__file__).resolve().parents[4] / "src" / "libs" / "llm" / "ollama_llm.py"
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None
        original_runtime = next((row.get("runtime_identity") for row in segments
                                 if row["segment"] == "original"), {}) or {}
        continuation_runtime = (continuation or {}).get("runtime_identity") or {}
        schema_facts = ((current_audit or {}).get("validation") or {}).get("schema")
        native_format = provider_request.get("format") if isinstance(provider_request, dict) else None
        application_schema_hash = cls._canonical_json_sha256(schema)
        native_format_hash = cls._canonical_json_sha256(native_format)
        response_text = response.get("text") if isinstance(response, dict) else None
        response_parse = "unknown_not_opened"
        if isinstance(response_text, str):
            try:
                json.loads(response_text)
                response_parse = "parsed_json_during_read_only_view"
            except (TypeError, ValueError):
                response_parse = "invalid_json_during_read_only_view"
        return {
            "scope": "current_requirement_findings_correction",
            "layer_1_schema_builder": {
                "status": "captured_application_response_format" if isinstance(schema, dict) else "unknown_not_opened",
                "schema_sha256": application_schema_hash,
                "constraints": schema_summary,
                "full_schema_in_model_messages": "not_demonstrated_by_application_messages",
                "basis": "hash_verified_application_request_record" if isinstance(schema, dict) else "raw_not_selected",
            },
            "layer_2_application_to_native_provider": {
                "application_response_format": "captured" if isinstance(schema, dict) else "unknown_not_opened",
                "native_provider_format": native_format if isinstance(native_format, dict) else "unknown_not_captured",
                "native_provider_format_sha256": native_format_hash,
                "provider_body_sha256": (current_call or {}).get("provider_body_sha256"),
                "provider_request_sha256": (current_call or {}).get("provider_request_sha256"),
                "provider_request_bytes": (current_call or {}).get("provider_request_bytes"),
                "provider_request_metadata": (current_call or {}).get("provider_request_metadata"),
                "current_adapter_source_analysis": {
                    "behavior": "current OllamaLLM source copies response_format to the /api/chat format field",
                    "source": "src/libs/llm/ollama_llm.py",
                    "source_sha256": source_hash,
                    "binding": "current_workspace_source_not_run_bound",
                },
                "actual_preservation_result": (
                    "verified_equal_application_schema" if application_schema_hash is not None
                    and application_schema_hash == native_format_hash else
                    "recorded_format_differs_from_application_schema" if native_format_hash is not None else
                    "unknown_not_captured"
                ),
                "native_messages_display": "not_displayed",
            },
            "layer_3_provider_runtime_support": {
                "continuation_recorded_ollama_version": continuation_runtime.get("ollama_server_version"),
                "original_run_recorded_ollama_version": original_runtime.get("ollama_server_version"),
                "llama_cpp_version": continuation_runtime.get("llama_cpp_version"),
                "llama_cpp_version_status": continuation_runtime.get("llama_cpp_version_status"),
                "model": (current_call or {}).get("model"),
                "original_run_model_identity": original_runtime.get("local_model_identity"),
                "official_documentation": {
                    "claim": "Ollama documents JSON Schema structured output through the /api/chat format field.",
                    "url": "https://docs.ollama.com/api/chat#structured-outputs",
                    "scope": "official_general_capability_not_this_run",
                },
                "version_source_analysis": {
                    "ollama_version": "0.34.2",
                    "api_types": "ChatRequest.Format is json.RawMessage",
                    "server_route": "the recorded version passes req.Format into CompletionRequest.Format",
                    "sources": [
                        "https://github.com/ollama/ollama/blob/v0.34.2/api/types.go",
                        "https://github.com/ollama/ollama/blob/v0.34.2/server/routes.go",
                    ],
                    "scope": "fixed_version_source_analysis_not_actual_conversion_capture",
                },
                "schema_to_grammar_result": "unknown_not_recorded",
                "combination_support_result": "unknown_for_this_exact_run",
            },
            "layer_4_raw_response_against_recorded_schema": {
                "response_readback": "hash_verified_local_detail" if isinstance(response_text, str) else "unknown_not_opened",
                "response_json_parse": response_parse,
                "response_sha256": (current_call or {}).get("output_sha256"),
                "schema_violation": schema_facts if isinstance(schema_facts, dict) else "unknown_not_recorded",
                "error_code": (current_audit or {}).get("error_code"),
            },
            "layer_5_local_validators": {
                "application_parse": "recorded_as_reached_before_schema_admission" if current_audit else "unknown",
                "schema_validator": schema_facts if isinstance(schema_facts, dict) else "unknown_not_recorded",
                "business_validator": ((current_audit or {}).get("validation") or {}).get("business", "unknown"),
                "historical_first_business_rejection": "offline_exact_input_replay_only",
                "negative_training_status": "not_recorded_as_dpo_negative",
            },
        }

    @staticmethod
    def _schema_summary(schema: Any) -> dict[str, Any] | str:
        if not isinstance(schema, dict):
            return "unknown_not_opened"
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        findings = properties.get("findings") if isinstance(properties.get("findings"), dict) else {}
        item = findings.get("items") if isinstance(findings.get("items"), dict) else {}
        return {
            "root_type": schema.get("type"),
            "root_required": schema.get("required") if isinstance(schema.get("required"), list) else None,
            "findings_min_items": findings.get("minItems"),
            "findings_max_items": findings.get("maxItems"),
            "item_required": item.get("required") if isinstance(item.get("required"), list) else None,
            "item_one_of_branches": len(item.get("oneOf")) if isinstance(item.get("oneOf"), list) else 0,
        }

    @staticmethod
    def _canonical_json_sha256(value: Any) -> str | None:
        if not isinstance(value, dict):
            return None
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Sanitization helpers
    # ------------------------------------------------------------------

    def _sanitize_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dashboard-safe trace unless raw debug access is explicit."""
        if self.include_raw:
            return trace
        sanitized = self._sanitize_value(trace, key_path=())
        if isinstance(sanitized, dict):
            return sanitized
        return {"trace_summary": sanitized}

    def _sanitize_value(self, value: Any, *, key_path: tuple[str, ...]) -> Any:
        if isinstance(value, dict):
            return {
                str(key): (
                    _REDACTED
                    if self._is_sensitive_key(str(key).lower())
                    else self._sanitize_value(
                        item,
                        key_path=key_path + (str(key),),
                    )
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._sanitize_value(item, key_path=key_path)
                for item in value
            ]
        if isinstance(value, str):
            return self._sanitize_string(value, key_path=key_path)
        return value

    def _sanitize_string(self, value: str, *, key_path: tuple[str, ...]) -> str:
        key = key_path[-1].lower() if key_path else ""
        if self._is_sensitive_key(key):
            return _REDACTED
        if self._is_path_key(key) or self._looks_like_path(value):
            return self._redact_path(value)
        if len(value) > self.max_string_chars:
            return value[: self.max_string_chars] + _TRUNCATED_SUFFIX
        return value

    @staticmethod
    def _is_sensitive_key(key: str) -> bool:
        return any(part in key for part in _SENSITIVE_KEY_PARTS)

    @staticmethod
    def _is_path_key(key: str) -> bool:
        return any(part == key or key.endswith(f"_{part}") for part in _PATH_KEY_PARTS)

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        return bool(
            _WINDOWS_ABSOLUTE_PATH.match(value)
            or _POSIX_ABSOLUTE_PATH.match(value)
        )

    @staticmethod
    def _redact_path(value: str) -> str:
        # Recorded path syntax is independent of the dashboard host OS.
        windows_path = PureWindowsPath(value)
        if not value.startswith("/") and (windows_path.drive or "\\" in value):
            name = windows_path.name
        else:
            name = PurePosixPath(value).name
        if not name or name == value:
            return "[redacted_path]"
        return f"[redacted_path:{name}]"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Dict[str, Any]]:
        """Parse every line in the JSONL file.

        Silently skips malformed lines.
        """
        if not self.traces_path.exists():
            return []

        traces: List[Dict[str, Any]] = []
        with self.traces_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed trace line: %s", line[:80])
        return traces
