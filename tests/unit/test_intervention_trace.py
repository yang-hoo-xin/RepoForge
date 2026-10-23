"""Explicit assistance records, not human-time inference from machine clocks."""
import json

import pytest

from src.core.trace.execution_trace import ExecutionTrace
from src.core.trace.trace_collector import TraceCollector


def _event(trace, event_id="review-1", **extra):
    return trace.record_intervention(event_id=event_id, run_id="run-1", actor="human",
        kind="review", cost_scope="task_runtime", subject_ref={"candidate_source_id": "code-1"},
        **extra)


def test_declaration_uses_existing_trace_storage_without_timing_inference(tmp_path):
    trace = ExecutionTrace("trace-1", "run-1", "test")
    with trace.activate():
        first = _event(trace)
        second = _event(trace, event_id="review-2", timing_basis="declared", active_seconds=120, person_id="reviewer-1")
    collector = TraceCollector(tmp_path / "trace.jsonl")
    collector.collect(trace.context)
    saved = json.loads(collector.path.read_text(encoding="utf-8"))
    rows = {stage["data"]["id"]: stage["data"] for stage in saved["stages"]}
    assert rows[first]["attributes"]["intervention"]["active_seconds"] is None
    assert rows[second]["attributes"]["intervention"]["active_seconds"] == 120
    assert rows[second]["elapsed_ms"] != 120_000
    assert rows[first]["attributes"]["intervention"]["event_id"] == "review-1"


def test_outer_codex_help_keeps_assistance_but_has_no_human_time():
    trace = ExecutionTrace("trace-1", "run-1", "test")
    with trace.activate():
        observation = trace.record_intervention(event_id="hint-1", run_id="run-1", actor="outer_codex",
            kind="technical_hint", cost_scope="platform_development", subject_ref="platform-fix")
        with pytest.raises(ValueError, match="MODEL_DURATION_IS_NOT_HUMAN_ACTIVE_TIME"):
            trace.record_intervention(event_id="hint-2", run_id="run-1", actor="outer_codex",
                kind="patch_edit", cost_scope="task_runtime", subject_ref="candidate",
                active_seconds=60, timing_basis="declared")
    rows = {stage["data"]["id"]: stage["data"] for stage in trace.context.stages}
    event = rows[observation]["attributes"]["intervention"]
    assert event["actor"] == "outer_codex" and event["kind"] == "technical_hint"
    assert event["active_seconds"] is None and event["cost_scope"] == "platform_development"


def test_measured_time_requires_explicit_interval_and_origin():
    trace = ExecutionTrace("trace-1", "run-1", "test")
    with trace.activate():
        with pytest.raises(ValueError, match="MEASUREMENT_EVIDENCE_REQUIRED"):
            _event(trace, timing_basis="measured", active_seconds=60)
        observation = _event(trace, timing_basis="measured", person_id="reviewer-1",
            started_at="2026-10-23T10:00:00+08:00", finished_at="2026-10-23T10:01:00+08:00",
            evidence_refs=[{"path": "timer-origin.json", "sha256": "a" * 64}])
    rows = {stage["data"]["id"]: stage["data"] for stage in trace.context.stages}
    assert rows[observation]["attributes"]["intervention"]["active_seconds"] == 60


@pytest.mark.parametrize("extra", [
    {"active_seconds": 60},
    {"timing_basis": "declared", "active_seconds": float("nan")},
    {"timing_basis": "declared", "active_seconds": -1},
    {"timing_basis": "declared", "active_seconds": True},
    {"timing_basis": "declared"},
])
def test_missing_or_invalid_active_time_does_not_become_a_value(extra):
    trace = ExecutionTrace("trace-1", "run-1", "test")
    with pytest.raises(ValueError):
        _event(trace, **extra)
