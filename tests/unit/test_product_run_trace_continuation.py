"""Read-only continuation views use recorded references, not a fabricated manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.observability.dashboard.services.trace_service import TraceService
from tests.unit.test_product_trace_drilldown import RUN_ID, _write, run_records


def retained_records(tmp_path):
    allowed, original, _legacy = run_records(tmp_path)
    root = original / "continuations" / "arbitrary-name-not-a-format"
    root.mkdir()
    parent_manifest = TraceService._read_product_json(original / "run_manifest.json", root=original)
    parent_result = TraceService._read_product_json(original / "run_result.json", root=original)
    identity = {"parent_manifest_ref": {"path": str(original / "run_manifest.json"), "sha256": parent_manifest[1]},
        "parent_result_ref": {"path": str(original / "run_result.json"), "sha256": parent_result[1]}}
    claim = _write(root / "selection.json", {**identity, "attempt_id": "saved-candidate-checkpoint"})
    graph = _write(root / "graph_result.json", {"run_id": RUN_ID, "current_stage": "dry_run_apply", "trace_events": []})
    result = {"entry": "retained-task-agent-handoff", "run_id": RUN_ID, "parent_run_id": RUN_ID,
        "continuation_root": str(root), "run_root": str(original), **identity,
        "continuation_claim_ref": claim, "fresh_result_ref": graph, "status": "awaiting_approval",
        "model_call_count": 2, "model_call_budget": parent_result[0]["model_call_budget"],
        "model_calls_performed": 0, "new_provider_sends": 0, "new_builds": 0,
        "actual_write_performed": False, "patch_execution_receipts": []}
    result_ref = _write(root / "run_result.json", result)
    _write(root / "completion.json", {"result_ref": result_ref, "claim_ref": claim, "fresh_result_ref": graph})
    return allowed, original, root


def inspect(allowed, original, continuation):
    return TraceService().inspect_product_run(allowed_root=allowed, run_root=original,
        run_id=RUN_ID, continuation_root=continuation)


def test_manifestless_saved_candidate_continuation_keeps_history_out_of_new_calls(tmp_path):
    allowed, original, continuation = retained_records(tmp_path)
    before = {str(p): p.read_bytes() for root in (original, continuation) for p in root.glob("*.json")}
    projected = inspect(allowed, original, continuation)
    segment = projected["segments"][-1]
    assert segment["record_format"] == "hash_bound_continuation_result"
    assert segment["identity_status"] == "verified_parent_and_completion_refs"
    assert segment["manifest_ref"] is None and segment["manifest_status"] == "not_recorded"
    assert segment["trace_status"] == "not_recorded"
    assert segment["status"] == "awaiting_approval"
    assert segment["model_call_count"] == 2
    assert segment["current_invocation_model_call_count"] == 0
    assert segment["calls"] == [] and segment["observation_count"] == 0
    assert projected["call_accounting"]["actual_generation_records"] == 1
    assert projected["call_accounting"]["latest"]["consumed"] == 2
    assert all(Path(path).read_bytes() == raw for path, raw in before.items())
    assert not (continuation / "run_manifest.json").exists()


def test_missing_continuation_links_are_explicitly_unconfirmed(tmp_path):
    allowed, original, continuation = retained_records(tmp_path)
    result = json.loads((continuation / "run_result.json").read_bytes())
    result.pop("parent_manifest_ref")
    (continuation / "run_result.json").write_text(json.dumps(result), encoding="utf-8")
    projected = inspect(allowed, original, continuation)["segments"][-1]
    assert projected["identity_status"] == "unconfirmed"
    assert "parent_manifest_ref" in projected["missing_identity_fields"]
    assert projected["status"] == "unconfirmed"
    assert projected["reported_status"] == "awaiting_approval"
    assert projected["model_call_budget"] is None and projected["current_invocation_model_call_count"] is None


def rewrite_result(root, document):
    result_ref = _write(root / "run_result.json", document)
    completion = json.loads((root / "completion.json").read_bytes())
    completion["result_ref"] = result_ref
    _write(root / "completion.json", completion)


@pytest.mark.parametrize("damage", ["parent_hash", "graph_hash", "completion_hash", "parent_identity",
    "result_identity", "outside_reference", "declared_root", "counter_conflict"])
def test_continuation_does_not_relax_hash_identity_or_selected_roots(tmp_path, damage):
    allowed, original, continuation = retained_records(tmp_path)
    result = json.loads((continuation / "run_result.json").read_bytes())
    if damage == "parent_hash":
        result["parent_manifest_ref"]["sha256"] = "0" * 64
    elif damage == "graph_hash":
        (continuation / "graph_result.json").write_text('{"changed":true}', encoding="utf-8")
    elif damage == "completion_hash":
        result["extra"] = "tampered without completion binding"
        _write(continuation / "run_result.json", result)
    elif damage == "parent_identity":
        result["parent_run_id"] = "unrelated-run"
    elif damage == "result_identity":
        result["run_id"] = "unrelated-run"
    elif damage == "outside_reference":
        outside = tmp_path / "outside-manifest.json"
        outside.write_bytes((original / "run_manifest.json").read_bytes())
        result["parent_manifest_ref"]["path"] = str(outside)
    elif damage == "declared_root":
        result["continuation_root"] = str(original)
    else:
        result["current_invocation_model_call_count"] = 2
    if damage not in {"graph_hash", "completion_hash"}:
        rewrite_result(continuation, result)
    with pytest.raises(ValueError):
        inspect(allowed, original, continuation)


def test_unknown_new_call_count_does_not_fall_back_to_historical_total(tmp_path):
    allowed, original, continuation = retained_records(tmp_path)
    result = json.loads((continuation / "run_result.json").read_bytes())
    result.pop("model_calls_performed")
    result.pop("new_provider_sends")
    rewrite_result(continuation, result)
    segment = inspect(allowed, original, continuation)["segments"][-1]
    assert segment["model_call_count"] == 2
    assert segment["current_invocation_model_call_count"] is None
    assert segment["new_provider_sends"] is None
    assert segment["calls"] == []


@pytest.mark.parametrize("filename", ["completion.json", "selection.json", "graph_result.json"])
def test_missing_completion_chain_file_is_unconfirmed(tmp_path, filename):
    allowed, original, continuation = retained_records(tmp_path)
    (continuation / filename).unlink()
    segment = inspect(allowed, original, continuation)["segments"][-1]
    assert segment["identity_status"] == "unconfirmed"
    assert filename in segment["missing_identity_fields"]


def test_bad_present_manifest_cannot_fall_back_to_other_format(tmp_path):
    allowed, original, continuation = retained_records(tmp_path)
    _write(continuation / "run_manifest.json", {"run_id": "different-run"})
    with pytest.raises(ValueError, match="identity"):
        inspect(allowed, original, continuation)


def test_selected_run_may_be_the_entire_explicit_authorized_root(tmp_path):
    _allowed, original, continuation = retained_records(tmp_path)
    result = inspect(original, original, continuation)
    assert result["segments"][-1]["identity_status"] == "verified_parent_and_completion_refs"
    assert len(result["calls"]) == 1
    sibling = original.parent / "not-authorized"
    sibling.mkdir()
    with pytest.raises(ValueError, match="outside"):
        inspect(original, sibling, None)
    with pytest.raises(ValueError, match="outside"):
        inspect(original, original, original / "continuations")


def test_bound_continuation_reference_cannot_follow_a_link(tmp_path, monkeypatch):
    allowed, original, continuation = retained_records(tmp_path)
    real_is_symlink = Path.is_symlink
    linked = continuation / "graph_result.json"
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == linked or real_is_symlink(path))
    with pytest.raises(ValueError, match="links"):
        inspect(allowed, original, continuation)


@pytest.mark.parametrize("field", ["entry", "blocker", "reported_status"])
@pytest.mark.parametrize("include_raw", [False, True])
def test_complete_continuation_display_is_redacted_after_original_byte_validation(tmp_path, field, include_raw):
    allowed, original, continuation = retained_records(tmp_path)
    path = continuation / "run_result.json"
    result = json.loads(path.read_bytes())
    secret = "CONTINUATION_CANARY_SECRET_903"
    if field == "entry":
        result["entry"] = "restored token=" + secret
    elif field == "reported_status":
        result["status"] = "failed token=" + secret
    else:
        result["blockers"] = [{"message": "cannot continue token=" + secret}]
    rewrite_result(continuation, result)
    before = path.read_bytes()
    projected = TraceService().inspect_product_run(allowed_root=allowed, run_root=original,
        run_id=RUN_ID, continuation_root=continuation, include_local_raw=include_raw)
    assert secret not in json.dumps(projected, ensure_ascii=False)
    assert projected["redaction"]["applied"] is True
    assert projected["redaction"]["scope"] == "display_only_original_bytes_unchanged"
    assert projected["redaction"]["validation_basis"] == "original_bytes_before_display_redaction"
    assert projected["segments"][-1]["identity_status"] == "verified_parent_and_completion_refs"
    assert projected["segments"][-1]["result_ref"]["sha256"] == TraceService._read_product_json(path, root=continuation)[1]
    assert path.read_bytes() == before and secret.encode() in before


def test_original_product_uses_same_display_redaction_boundary(tmp_path):
    allowed, original, _continuation = run_records(tmp_path)
    path = original / "run_manifest.json"
    document = json.loads(path.read_bytes())
    secret = "ORIGINAL_STATUS_CANARY_829"
    document["status"] = "failed token=" + secret
    _write(path, document)
    before = path.read_bytes()
    projected = TraceService().inspect_product_run(allowed_root=allowed, run_root=original, run_id=RUN_ID)
    assert secret not in json.dumps(projected, ensure_ascii=False)
    assert projected["redaction"]["applied"] is True
    assert path.read_bytes() == before


def test_redaction_does_not_make_two_different_original_results_hash_equivalent(tmp_path):
    allowed, original, continuation = retained_records(tmp_path)
    path = continuation / "run_result.json"
    result = json.loads(path.read_bytes())
    result["entry"] = "token=ORIGINAL_SENSITIVE_VALUE"
    rewrite_result(continuation, result)
    result["entry"] = "token=CHANGED_SENSITIVE_VALUE"
    _write(path, result)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        inspect(allowed, original, continuation)
