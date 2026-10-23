"""Persist a derived developer view after the original execution facts commit.

为什么要有它：汇总失败不能改变已经发生的执行，恢复收尾也必须保留原任务身份。
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from uuid import uuid4


def _read(path):
    value = json.loads(_safe_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("OUTCOME_RECORD_OBJECT_REQUIRED")
    return value


def _safe_path(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), "st_file_attributes", 0)
                                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError("OUTCOME_LINK_NOT_ALLOWED")
    return path


def _write(path, value, *, expected_bytes=None):
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if expected_bytes is not None and path.read_bytes() != expected_bytes:
        raise ValueError("OUTCOME_MANIFEST_CHANGED")
    if path.exists() and path.read_bytes() == raw:
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if expected_bytes is not None and path.read_bytes() != expected_bytes:
        temporary.unlink()
        raise ValueError("OUTCOME_MANIFEST_CHANGED")
    os.replace(temporary, path)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _immutable(root, folder, value):
    digest = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    path = _safe_path(root / folder / (digest + ".json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError("OUTCOME_IMMUTABLE_RECORD_CHANGED")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _find_run_root(artifact_store_root):
    store = _safe_path(artifact_store_root).resolve(strict=True)
    # Existing stores are below their run root; never discover unrelated runs.
    for candidate in (store, *store.parents):
        if (candidate / "run_manifest.json").is_file():
            return candidate
    raise ValueError("OUTCOME_RUN_MANIFEST_UNAVAILABLE")


def finalize_run_outcome(*, run_root, run_id, boundary_ref=None, evidence_roots=()):
    """Best-effort summary; callers have already committed their original result."""
    root = Path(run_root)
    failure = {"outcome_summary_status": "summary_unavailable"}
    try:
        from src.observability.dashboard.services.run_evidence import query_run_evidence

        root = _safe_path(root).resolve(strict=True)
        manifest_path = _safe_path(root / "run_manifest.json")
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        if manifest.get("run_id") != run_id:
            raise ValueError("OUTCOME_RUN_IDENTITY_CHANGED")
        if boundary_ref is not None:
            boundaries = manifest.setdefault("outcome_boundaries", [])
            if boundary_ref not in boundaries:
                boundaries.append(boundary_ref)
            _write(manifest_path, manifest, expected_bytes=manifest_raw)
        report = query_run_evidence(allowed_roots=[root], run_id=run_id, evidence_roots=evidence_roots)
        outcome = report.get("outcome")
        if not isinstance(outcome, dict):
            raise ValueError("OUTCOME_PROJECTION_UNAVAILABLE")
        summary = {"schema": "run_outcome_summary_v1", "run_id": run_id,
            "request_id": manifest.get("request_id"), "outcome": outcome,
            "terminal_record": manifest.get("terminal_record"),
            "boundary_refs": manifest.get("outcome_boundaries", []),
            "complete": report.get("complete", False), "query_status": report.get("status")}
        reference = _immutable(root, "run-outcomes", summary)
        projection = {"outcome_summary_ref": reference, "outcome_summary_status": "available"}
        # Reload so an unrelated lifecycle field is never replaced by our old view.
        current_raw = manifest_path.read_bytes()
        current = json.loads(current_raw)
        if current.get("run_id") != run_id:
            raise ValueError("OUTCOME_RUN_IDENTITY_CHANGED")
        if ({k: v for k, v in current.items() if not k.startswith("outcome_summary_")}
                != {k: v for k, v in manifest.items() if not k.startswith("outcome_summary_")}):
            raise ValueError("OUTCOME_MANIFEST_CHANGED")
        current.update(projection)
        current.pop("outcome_summary_error", None)
        _write(manifest_path, current, expected_bytes=current_raw)
        return projection
    except Exception as error:
        failure["outcome_summary_error"] = type(error).__name__
        try:
            manifest_path = _safe_path(root / "run_manifest.json")
            current_raw = manifest_path.read_bytes()
            manifest = json.loads(current_raw)
            if manifest.get("run_id") == run_id:
                manifest.update(failure)
                manifest.pop("outcome_summary_ref", None)
                _write(manifest_path, manifest, expected_bytes=current_raw)
        except Exception:
            pass
        return failure


def finalize_continuation(*, artifact_store_root, event, result=None, checkpoint_ref=None, error=None, evidence_roots=()):
    """Save the missing continuation boundary without replacing original terminal facts."""
    try:
        root = _find_run_root(artifact_store_root)
        manifest = _read(root / "run_manifest.json")
        payload = dict(result or {})
        if payload.get("run_id", manifest["run_id"]) != manifest["run_id"]:
            raise ValueError("OUTCOME_CONTINUATION_IDENTITY_CHANGED")
        if payload.get("request_id", manifest.get("request_id")) != manifest.get("request_id"):
            raise ValueError("OUTCOME_CONTINUATION_REQUEST_CHANGED")
        # Keep the returned identity and references, never copy cumulative budgets.
        original = payload
        payload = {key: original[key] for key in (
            "status", "run_id", "request_id", "feature_id", "new_task_created", "replayed",
            "receipt_ref", "candidate_source_id", "whole_task_delivered", "v2_checkpoint_ref",
            "v2_delivery_checkpoint_ref", "resumed_from_checkpoint_ref", "resumed_from_rebase_ref",
            "resumed_from_delivery_receipt_ref", "task_timing_ref") if key in original}
        outcome = original.get("task_agent_outcome", original.get("outcome", {}))
        if isinstance(outcome, dict) and outcome.get("execution_ref"):
            payload["native_execution_ref"] = outcome["execution_ref"]
        learning = original.get("strategy_learning", {})
        if isinstance(learning, dict) and learning.get("receipt_ref"):
            payload["learning_receipt_ref"] = learning["receipt_ref"]
        boundary = {"schema": "run_outcome_boundary_v1", "run_id": manifest["run_id"],
            "request_id": manifest.get("request_id"), "event": event,
            "original_terminal_ref": manifest.get("terminal_record"), "result": payload}
        if checkpoint_ref is not None:
            store = Path(artifact_store_root).resolve(strict=True)
            path = Path(checkpoint_ref)
            try:
                path = _safe_path(path if path.is_absolute() else store / path).resolve(strict=True)
                if not path.is_relative_to(store):
                    raise ValueError("OUTCOME_CHECKPOINT_OUTSIDE_STORE")
                raw = path.read_bytes()
                checkpoint = json.loads(raw)
                field = "resumed_from_checkpoint_ref" if event.startswith("recovery") else "input_checkpoint_ref"
                boundary[field] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
                boundary["task_contract_ref"] = checkpoint.get("task_contract_ref")
            except (OSError, ValueError, TypeError, AttributeError):
                boundary["checkpoint_status"] = "unavailable_or_unverified"
        if error is not None:
            boundary.update(status="blocked", error_type=type(error).__name__,
                execution_status="not_executed" if event == "recovery_admission_rejected" else "unknown",
                reason="Resume was not dispatched" if event == "recovery_admission_rejected"
                    else "Continuation raised; retained original receipts determine effects")
        elif payload.get("status") is not None:
            boundary["status"] = payload["status"]
        reference = _immutable(root, "run-continuations", boundary)
        return {"outcome_boundary_ref": reference,
            **finalize_run_outcome(run_root=root, run_id=manifest["run_id"], boundary_ref=reference,
                evidence_roots=evidence_roots)}
    except Exception as error:
        return {"outcome_summary_status": "summary_unavailable", "outcome_summary_error": type(error).__name__}
