"""Retrieval expectations, full batch denominators, and CLI decision semantics."""
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.types import RetrievalResult
from src.libs.evaluator.base_evaluator import NoneEvaluator
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.unified import read_ref
from src.observability.evaluation.unified_hooks import RagEvaluationHook
from tests.unit.test_unified_evaluation_adapters import TEXT, critic, sample


def run_case(tmp_path, *, expected=None, requirement=None, retrieved="other"):
    case = {"query": "What is the default?", "expected_behavior": "answer",
            "expected_chunk_ids": ["c1", "c2"] if expected is None else expected}
    if requirement is not None:
        case["retrieval_requirement"] = requirement
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"test_cases": [case]}), encoding="utf-8")
    search = MagicMock()
    search.search.return_value = [] if retrieved is None else [RetrievalResult(
        chunk_id=retrieved, score=1.0, text=TEXT, metadata={})]
    hook = RagEvaluationHook(tmp_path / "reports", samples={0: sample()}, critic=critic())
    report = EvalRunner(hybrid_search=search, evaluator=NoneEvaluator(),
        answer_generator=lambda *_: TEXT, synthetic_fallback=False, unified_hook=hook,
    ).run(path, run_metadata={"run_id": "retrieval-contract"})
    return report.to_dict()


def test_diagnostic_ids_survive_none_evaluator_without_inventing_gate(tmp_path):
    report = run_case(tmp_path)
    query = report["query_results"][0]
    result = query["unified_evaluation"]
    payload = json.loads(read_ref(result["binding"]["candidate_ref"]))
    assert payload["retrieval_diagnostic"]["expected_chunk_ids"] == ["c1", "c2"]
    assert payload["retrieval_diagnostic"]["metrics"] == {"hit_rate": 0.0, "mrr": 0.0}
    assert payload["retrieval_diagnostic"]["decision_use"] == "diagnostic_only"
    assert result["status"] == "passed"
    assert result["acceptance_status"] == "insufficient_evidence"


@pytest.mark.parametrize("retrieved,expected_status", [("other", "failed"), ("c2", "passed")])
def test_explicit_retrieval_threshold_uses_existing_any_hit_semantics(tmp_path, retrieved, expected_status):
    report = run_case(tmp_path, retrieved=retrieved,
                      requirement={"metric": "hit_rate", "threshold": 1.0})
    result = report["query_results"][0]["unified_evaluation"]
    row = next(r for r in result["criteria"] if r["dimension"] == "retrieval_quality")
    assert row["status"] == expected_status
    assert row["decision_use"] == "acceptance"
    assert result["status"] == expected_status
    if expected_status == "failed":
        assert result["acceptance_status"] == "failed"


def test_retrieval_requirement_without_truth_is_rejected_before_calls(tmp_path):
    with pytest.raises(ValueError, match="RAG_RETRIEVAL_REQUIREMENT_INVALID"):
        run_case(tmp_path, expected=[], requirement={"metric": "hit_rate", "threshold": 1.0})


@pytest.mark.parametrize("requirement", [
    {"metric": "recall", "threshold": 1}, {"metric": [], "threshold": 1},
    {"metric": "hit_rate", "threshold": True}, {"metric": "hit_rate", "threshold": 0},
    {"metric": "hit_rate", "threshold": float("nan")},
    {"metric": "hit_rate", "threshold": 2}, {}, "hit_rate",
])
def test_invalid_retrieval_requirement_is_rejected(tmp_path, requirement):
    with pytest.raises(ValueError, match="RAG_RETRIEVAL_REQUIREMENT_INVALID"):
        run_case(tmp_path, requirement=requirement)


def test_observed_empty_retrieval_is_zero_not_evaluator_error(tmp_path):
    report = run_case(tmp_path, retrieved=None,
                      requirement={"metric": "mrr", "threshold": 0.5})
    result = report["query_results"][0]["unified_evaluation"]
    row = next(r for r in result["criteria"] if r["dimension"] == "retrieval_quality")
    assert row["status"] == "failed" and row["value"] == 0.0
    assert result["acceptance_status"] == "failed"


def test_rag_batch_summary_distinguishes_trial_pass_from_acceptance(tmp_path):
    report = run_case(tmp_path, retrieved="c1")
    summary = report["unified_evaluation"]
    assert summary["status"] == "passed"
    assert summary["acceptance_status"] == "insufficient_evidence"
    assert summary["query_count"] == 1
    assert summary["counts"]["passed"] == 1
    assert summary["acceptance_counts"]["insufficient_evidence"] == 1
    assert set(summary["counts"]) == {
        "passed", "failed", "insufficient_evidence", "evaluation_error", "not_applicable"}


def test_batch_keeps_unattempted_rows_and_hook_failures():
    from src.observability.evaluation.eval_runner import EvalReport, QueryResult
    report = EvalReport(unified_evaluation_enabled=True, query_results=[
        QueryResult("first", unified_evaluation={"status": "failed", "acceptance_status": "failed"}),
        QueryResult("hook error", unified_evaluation={"status": "evaluation_error"}),
        QueryResult("unattempted"),
    ]).to_dict()
    summary = report["unified_evaluation"]
    assert summary["query_count"] == 3
    assert summary["status"] == summary["acceptance_status"] == "failed"
    assert summary["acceptance_counts"]["failed"] == 1
    assert summary["acceptance_counts"]["evaluation_error"] == 1
    assert summary["acceptance_counts"]["insufficient_evidence"] == 1
    assert summary["missing_report_count"] == 1


@pytest.mark.parametrize("status,code", [
    ("passed", 0), ("failed", 3), ("insufficient_evidence", 4),
    ("evaluation_error", 1), ("not_applicable", 5),
])
def test_cli_terminal_separates_process_exit_and_acceptance(status, code):
    from scripts.run_unified_evaluation import terminal_summary
    result = {"unified_evaluation": {"status": "passed", "acceptance_status": status}}
    process = terminal_summary(result, kind="rag", exit_code_policy="process")
    gated = terminal_summary(result, kind="rag", exit_code_policy="acceptance")
    assert process["process_status"] == "completed"
    assert process["exit_code"] == 0
    assert process["acceptance_status"] == status
    assert process["acceptance_exit_code"] == code
    assert gated["exit_code"] == code


@pytest.mark.parametrize("policy,exit_code", [("process", 0), ("acceptance", 4)])
def test_normal_rag_cli_persists_batch_and_emits_acceptance_exit(tmp_path, monkeypatch, capsys, policy, exit_code):
    import scripts.run_unified_evaluation as cli
    import src.core.settings as settings_module
    import src.core.query_engine.service_factory as factory

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"schema_version": "prelaunch-evaluation-run.v1",
        "run_id": "controlled-entry", "samples": {"0": asdict(sample())}}), encoding="utf-8")
    test_set = tmp_path / "cases.json"
    test_set.write_text(json.dumps({"test_cases": [{"query": "What is the default?",
        "expected_behavior": "answer", "expected_chunk_ids": ["c1"]}]}), encoding="utf-8")
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("controlled: true", encoding="utf-8")
    settings = SimpleNamespace(llm=SimpleNamespace(provider="controlled"))
    calls = []

    class Service:
        collection = "default"

        def _client(self):
            return object()

        def query(self, request):
            calls.append(request)
            response = {"results": [{"id": "c1", "text": TEXT, "score": 1, "metadata": {}}],
                "execution_state": {"retrieval_completed": True}, "error": None,
                "trace": {"stages": []}, "answer": TEXT, "status": "completed",
                "citation_validation": {"status": "not_required"}, "answer_observation": None}
            return SimpleNamespace(to_dict=lambda: response, evidence={})

    monkeypatch.setattr(settings_module, "load_settings", lambda *_: settings)
    monkeypatch.setattr(factory, "create_query_service", lambda *a, **k: Service())
    def forbidden(*_):
        raise AssertionError("No judge or calibration model allowed")
    monkeypatch.setattr(cli, "build_critic", forbidden)
    monkeypatch.setattr(cli, "build_ragas", forbidden)
    output = tmp_path / "output"
    assert cli.main(["rag", "--metadata", str(metadata), "--test-set", str(test_set),
        "--settings", str(settings_path), "--output-root", str(output),
        "--exit-code-policy", policy]) == exit_code
    terminal = json.loads(capsys.readouterr().out)
    result = json.loads((output / "result.json").read_bytes())
    assert len(calls) == 1
    assert terminal["process_status"] == "completed"
    assert terminal["acceptance_exit_code"] == 4
    assert result["unified_evaluation"]["acceptance_counts"]["insufficient_evidence"] == 1
    candidate = result["query_results"][0]["unified_evaluation"]
    assert json.loads(read_ref(candidate["binding"]["candidate_ref"]))["retrieval_diagnostic"]["metrics"]["hit_rate"] == 1.0
