"""Evaluation runner for batch quality assessment.

EvalRunner reads a golden test set, runs HybridSearch for each test case,
optionally generates answers, then invokes the configured Evaluator(s) to
produce a structured evaluation report.

Design Principles:
- Config-Driven: Evaluator selected via settings.yaml.
- Observable: Produces EvalReport with per-query details.
- Decoupled: Works with any BaseEvaluator implementation.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.core.trace.trace_context import TraceContext
from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator

logger = logging.getLogger(__name__)

OBSERVED = "observed"
FAILED_TO_MEASURE = "failed_to_measure"
NOT_APPLICABLE = "not_applicable"
_MEASUREMENT_STATUSES = {OBSERVED, FAILED_TO_MEASURE, NOT_APPLICABLE}
_ANSWER_OVERRIDE_MISSING = object()

# These names are the exact metrics implemented by CustomEvaluator and
# RagasEvaluator.  The runner uses them only to express prerequisites; it does
# not combine or reinterpret their scores.
_CUSTOM_RETRIEVAL_METRICS = {"hit_rate", "mrr"}
_RAGAS_ANSWER_METRICS = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
}
_RAGAS_CONTEXT_METRICS = {"faithfulness", "context_precision"}
_TEST_SET_METADATA_FIELDS = (
    "description",
    "tier",
    "purpose",
    "source",
    "authoring_origin",
    "human_verified",
    "answer_policy",
    "citation_policy",
)


@dataclass(frozen=True)
class MeasurementState:
    """Serializable state for one evaluation stage."""

    status: str
    mode: Optional[str] = None
    error_category: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in _MEASUREMENT_STATUSES:
            raise ValueError(f"Unsupported measurement status: {self.status}")

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "status": self.status,
            "mode": self.mode,
            "error_category": self.error_category,
            "error_message": self.error_message,
        }

    @classmethod
    def from_exception(cls, exc: Exception, *, mode: str) -> MeasurementState:
        return cls(
            status=FAILED_TO_MEASURE,
            mode=mode,
            error_category=type(exc).__name__,
            error_message=str(exc),
        )


@dataclass
class GoldenTestCase:
    """A single evaluation test case from the golden test set.

    Attributes:
        query: The test query string.
        expected_chunk_ids: Ground-truth chunk IDs for IR metrics.
        expected_sources: Ground-truth source file names (optional).
        reference_answer: Reference answer text for LLM-as-Judge (optional).
    """

    query: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    reference_answer: Optional[str] = None
    source_sections: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)
    diagnostic_types: list[str] = field(default_factory=list)
    expected_behavior: str | None = None
    retrieval_requirement: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenTestCase:
        return cls(
            query=data["query"],
            expected_chunk_ids=data.get("expected_chunk_ids", []),
            expected_sources=data.get("expected_sources", []),
            reference_answer=data.get("reference_answer"),
            source_sections=data.get("source_sections", []),
            source_pages=data.get("source_pages", []),
            diagnostic_types=data.get("diagnostic_types", []),
            expected_behavior=data.get("expected_behavior"),
            retrieval_requirement=deepcopy(data.get("retrieval_requirement")),
        )

    def metadata(self) -> dict[str, Any]:
        """Return suite-author facts that are not answer-generation inputs."""
        metadata = {
            "source_sections": list(self.source_sections),
            "source_pages": list(self.source_pages),
            "diagnostic_types": list(self.diagnostic_types),
            "expected_behavior": self.expected_behavior,
        }
        if self.expected_chunk_ids:
            metadata["expected_chunk_ids"] = list(self.expected_chunk_ids)
        if self.expected_sources:
            metadata["expected_sources"] = list(self.expected_sources)
        if self.retrieval_requirement is not None:
            metadata["retrieval_requirement"] = deepcopy(self.retrieval_requirement)
        return metadata


@dataclass
class QueryResult:
    """Result of evaluating a single test case.

    Attributes:
        case_index: Zero-based position in the loaded golden test set.
        query: The test query.
        retrieved_chunk_ids: IDs of chunks actually retrieved.
        retrieved_contexts: Snapshot of the final retrieved inputs, in input order.
        generated_answer: The generated answer (if applicable).
        metrics: Evaluation metrics for this query.
        metric_measurements: Per-metric measurement status.
        retrieval_measurement: Retrieval stage measurement state.
        answer_measurement: Answer stage measurement state and provenance.
        evaluator_measurement: Evaluator stage measurement state.
        elapsed_ms: Time taken for retrieval + evaluation.
        started_at: Actual case invocation start in UTC, or None if unattempted.
        finished_at: Actual case invocation end in UTC, or None if unattempted.
    """

    query: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    generated_answer: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    case_index: Optional[int] = None
    metric_measurements: Dict[str, str] = field(default_factory=dict)
    retrieval_measurement: MeasurementState = field(
        default_factory=lambda: MeasurementState(status=NOT_APPLICABLE)
    )
    answer_measurement: MeasurementState = field(
        default_factory=lambda: MeasurementState(status=NOT_APPLICABLE)
    )
    evaluator_measurement: MeasurementState = field(
        default_factory=lambda: MeasurementState(status=NOT_APPLICABLE)
    )
    citation_measurement: MeasurementState = field(
        default_factory=lambda: MeasurementState(status=NOT_APPLICABLE)
    )
    citation_required: bool = False
    reference_answer: str | None = None
    case_metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_contexts: list[dict[str, Any]] = field(default_factory=list)
    answer_observation: dict[str, Any] | None = None
    query_observation: dict[str, Any] | None = None
    # Recorded provider calls only; an empty list carries no completeness claim.
    embedding_observations: list[dict[str, Any]] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    unified_evaluation: dict[str, Any] | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report across all test cases.

    Attributes:
        query_results: Per-query evaluation results.
        aggregate_metrics: Averaged metrics across all queries.
        metric_measurements: Per-metric denominator metadata.
        expected_metrics: Declared and observed metric names.
        total_elapsed_ms: Total time for the entire evaluation.
        evaluator_name: Name of the evaluator used.
        test_set_path: Path to the golden test set file.
        run_metadata: Caller-selected JSON-compatible provenance; never settings.
    """

    query_results: List[QueryResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_elapsed_ms: float = 0.0
    evaluator_name: str = ""
    test_set_path: str = ""
    metric_measurements: Dict[str, Dict[str, int]] = field(default_factory=dict)
    expected_metrics: List[str] = field(default_factory=list)
    run_metadata: dict[str, Any] = field(default_factory=dict)
    test_set_metadata: dict[str, Any] = field(default_factory=dict)
    comparison_arm: str = "rag_on"
    unified_evaluation_enabled: bool = False

    @property
    def measurement_status(self) -> str:
        """Completeness of measurement, independent of the scores themselves."""
        counts = list(self.metric_measurements.values())
        if not counts or not any(row["observed"] for row in counts):
            return "unmeasured"
        stages_failed = any(
            stage.status == FAILED_TO_MEASURE
            for result in self.query_results
            for stage in (result.retrieval_measurement, result.answer_measurement,
                          result.evaluator_measurement, result.citation_measurement)
        )
        if stages_failed or any(row["failed_to_measure"] for row in counts):
            return "partial"
        return "complete"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise report to dictionary."""
        from .unified_hooks import summarize_rag_batch

        return {
            "unified_evaluation": summarize_rag_batch(self.query_results)
            if self.unified_evaluation_enabled else None,
            "evaluator_name": self.evaluator_name,
            "measurement_status": self.measurement_status,
            "test_set_path": self.test_set_path,
            "run_metadata": deepcopy(self.run_metadata),
            "test_set_metadata": deepcopy(self.test_set_metadata),
            "comparison_arm": self.comparison_arm,
            "total_elapsed_ms": round(self.total_elapsed_ms, 1),
            "aggregate_metrics": {
                k: round(v, 4) for k, v in self.aggregate_metrics.items()
            },
            "metric_measurements": {
                metric: dict(counts)
                for metric, counts in self.metric_measurements.items()
            },
            "expected_metrics": list(self.expected_metrics),
            "query_count": len(self.query_results),
            "query_results": [
                {
                    "unified_evaluation": deepcopy(qr.unified_evaluation),
                    "case_index": qr.case_index,
                    "query": qr.query,
                    "retrieved_chunk_ids": qr.retrieved_chunk_ids,
                    "retrieved_contexts": deepcopy(qr.retrieved_contexts),
                    "generated_answer": qr.generated_answer,
                    "reference_answer": qr.reference_answer,
                    "case_metadata": deepcopy(qr.case_metadata),
                    "metrics": {k: round(v, 4) for k, v in qr.metrics.items()},
                    "metric_measurements": dict(qr.metric_measurements),
                    "retrieval_measurement": qr.retrieval_measurement.to_dict(),
                    "answer_measurement": qr.answer_measurement.to_dict(),
                    "answer_observation": deepcopy(qr.answer_observation),
                    "query_observation": deepcopy(qr.query_observation),
                    "embedding_observations": deepcopy(qr.embedding_observations),
                    "evaluator_measurement": qr.evaluator_measurement.to_dict(),
                    "citation_measurement": qr.citation_measurement.to_dict(),
                    "citation_required": qr.citation_required,
                    "elapsed_ms": round(qr.elapsed_ms, 1),
                    "started_at": qr.started_at,
                    "finished_at": qr.finished_at,
                }
                for qr in self.query_results
            ],
        }


def load_test_set(path: str | Path) -> List[GoldenTestCase]:
    """Load golden test set from a JSON file.

    Args:
        path: Path to the golden test set JSON file.

    Returns:
        List of TestCase instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden test set not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "test_cases" not in data:
        raise ValueError(
            "Invalid golden test set format: missing 'test_cases' key."
        )

    return [GoldenTestCase.from_dict(tc) for tc in data["test_cases"]]


def load_test_set_metadata(path: str | Path) -> dict[str, Any]:
    """Load explicit suite provenance without treating it as generation input."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Golden test set not found: {file_path}")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if "human_verified" in data and type(data["human_verified"]) is not bool:
        raise ValueError("human_verified must be an exact boolean when provided.")
    return {key: deepcopy(data[key]) for key in _TEST_SET_METADATA_FIELDS if key in data}


class EvalRunner:
    """Runs batch evaluation against a golden test set.

    This class orchestrates:
    1. Loading the golden test set
    2. Running HybridSearch for each query
    3. Optionally generating answers
    4. Invoking the evaluator to score each result
    5. Aggregating metrics into an EvalReport

    Example::

        runner = EvalRunner(
            settings=settings,
            hybrid_search=hybrid_search,
            evaluator=evaluator,
        )
        report = runner.run("tests/fixtures/golden_test_set.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        settings: Any = None,
        hybrid_search: Any = None,
        evaluator: Optional[BaseEvaluator] = None,
        answer_generator: Any = None,
        answer_overrides: Optional[Dict[int, Any]] = None,
        reranker: Any = None,
        expected_metrics: Optional[Sequence[str]] = None,
        synthetic_fallback: bool = True,
        query_service: Any = None,
        unified_hook: Any = None,
    ) -> None:
        """Initialize EvalRunner.

        Args:
            settings: Application settings.
            hybrid_search: HybridSearch instance for retrieval.
            evaluator: BaseEvaluator instance for scoring.
            answer_generator: Optional callable(query, chunks) -> str
                for generating answers. If None, a simple concatenation
                is used as a placeholder.
            answer_overrides: Optional dict mapping test case index (0-based)
                to a user-provided answer value. Only an exact, non-empty str
                is accepted; an invalid present override is a measurement
                failure and is not replaced by auto-generation.
            reranker: Optional CoreReranker instance for reranking results.
            expected_metrics: Optional exact metric names expected from a
                BaseEvaluator whose abstract interface does not declare them.
            synthetic_fallback: Preserve the explicitly synthetic library
                preview when no generator is configured. Public evaluation
                entries disable this so absent answers remain absent.
        """
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.evaluator = evaluator
        self.answer_generator = answer_generator
        self.query_service = query_service
        self.answer_overrides = answer_overrides or {}
        self.reranker = reranker
        self.expected_metrics = self._validated_metric_names(expected_metrics or [])
        self.synthetic_fallback = synthetic_fallback
        self.unified_hook = unified_hook

    def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
        collection: Optional[str] = None,
        *,
        run_metadata: dict[str, Any] | None = None,
        stop_on_failure: bool = False,
        comparison_arm: str = "rag_on",
    ) -> EvalReport:
        """Run evaluation on the golden test set.

        Args:
            test_set_path: Path to golden_test_set.json.
            top_k: Number of chunks to retrieve per query.
            collection: Optional collection name filter.
            run_metadata: Explicit JSON-compatible model/index/run identities.
                Only caller-provided values are included; Settings are not read.
            stop_on_failure: Stop actual calls after the first case with a failed
                stage, retaining unattempted cases in the report and denominator.
            comparison_arm: ``rag_on`` when generation sees retrieved contexts,
                or ``rag_off`` when no retrieval is configured.

        Returns:
            EvalReport with per-query and aggregate metrics.

        Raises:
            FileNotFoundError: If test set file doesn't exist.
            ValueError: If evaluator or hybrid_search is not set.
        """
        if self.evaluator is None:
            raise ValueError("EvalRunner requires an evaluator.")
        if comparison_arm not in {"rag_on", "rag_off"}:
            raise ValueError("comparison_arm must be 'rag_on' or 'rag_off'.")

        input_ref = None
        if self.unified_hook is not None:
            from src.observability.evaluation.unified import file_ref
            input_ref = file_ref(test_set_path)
        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise ValueError("Golden test set is empty.")

        metadata = {} if run_metadata is None else deepcopy(run_metadata)
        if not isinstance(metadata, dict):
            raise TypeError("run_metadata must be a JSON-compatible dictionary.")
        # Fail before retrieval or model calls if provenance cannot be exported.
        json.dumps(metadata, allow_nan=False)
        if self.unified_hook is not None:
            # Scope: actual loader input, never caller-supplied provenance.
            # 为什么要有它：登记A与执行B必须在检索/模型调用之前被拒绝，且载入期间改变的题集不能取得旧身份。
            if file_ref(test_set_path) != input_ref:
                raise ValueError('EVALUATION_TEST_SET_CHANGED_DURING_LOAD')
            metadata['evaluation_test_set_ref'] = input_ref
            self.unified_hook.validate_run(test_cases, metadata)
            if set(self._declared_metric_names()) & {"faithfulness", "answer_relevancy", "context_precision"}:
                raise ValueError("UNIFIED_SEMANTIC_EVALUATOR_DUPLICATED")

        logger.info(
            "Starting evaluation: %d test cases, evaluator=%s",
            len(test_cases),
            type(self.evaluator).__name__,
        )

        report = EvalReport(
            evaluator_name=type(self.evaluator).__name__,
            test_set_path=str(test_set_path),
            run_metadata=metadata,
            test_set_metadata=load_test_set_metadata(test_set_path),
            comparison_arm=comparison_arm,
            unified_evaluation_enabled=self.unified_hook is not None,
        )
        declared_metrics = self._declared_metric_names()

        t0 = time.monotonic()
        stopped_by_case: int | None = None

        for idx, tc in enumerate(test_cases):
            if stopped_by_case is not None:
                stopped = MeasurementState(
                    status=FAILED_TO_MEASURE,
                    mode="run_stopped",
                    error_category="PreviousCaseFailed",
                    error_message=(
                        f"Case {idx} was not attempted because case {stopped_by_case} "
                        "failed to measure."
                    ),
                )
                report.query_results.append(QueryResult(
                    query=tc.query,
                    case_index=idx,
                    citation_required=bool(report.test_set_metadata.get("citation_policy")),
                    reference_answer=tc.reference_answer,
                    case_metadata=tc.metadata(),
                    retrieval_measurement=stopped,
                    answer_measurement=stopped,
                    evaluator_measurement=stopped,
                ))
                continue
            logger.info("Evaluating [%d/%d]: %s", idx + 1, len(test_cases), tc.query[:60])
            # Use user-provided answer override if available for this index
            answer_override = self.answer_overrides.get(
                idx,
                _ANSWER_OVERRIDE_MISSING,
            )
            started_at = datetime.now(timezone.utc).isoformat()
            qr = self._evaluate_single(
                tc, top_k=top_k, collection=collection,
                answer_override=answer_override,
                case_index=idx,
                expected_metrics=declared_metrics,
                require_context_citations=bool(report.test_set_metadata.get("citation_policy")),
            )
            qr.finished_at = datetime.now(timezone.utc).isoformat()
            qr.started_at = started_at
            if self.unified_hook is not None:
                try:
                    qr.unified_evaluation = self.unified_hook(qr, metadata)
                except Exception as error:
                    qr.unified_evaluation = {"status": "evaluation_error", "error_type": type(error).__name__, "report_ref": None}
            report.query_results.append(qr)
            if stop_on_failure and any(
                stage.status == FAILED_TO_MEASURE
                for stage in (
                    qr.retrieval_measurement, qr.answer_measurement, qr.evaluator_measurement,
                )
            ):
                stopped_by_case = idx

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        query_elapsed_ms = sum(qr.elapsed_ms for qr in report.query_results)
        report.total_elapsed_ms = max(elapsed_ms, query_elapsed_ms, 0.001)
        all_metric_names = self._all_metric_names(
            report.query_results,
            declared_metrics,
        )
        self._finalise_metric_measurements(report.query_results, all_metric_names)
        report.expected_metrics = all_metric_names
        report.aggregate_metrics = self._aggregate_metrics(report.query_results)
        report.metric_measurements = self._aggregate_metric_measurements(
            report.query_results,
            all_metric_names,
        )

        logger.info(
            "Evaluation complete: %d queries, aggregate=%s",
            len(report.query_results),
            report.aggregate_metrics,
        )

        return report

    def _evaluate_single(
        self,
        test_case: GoldenTestCase,
        top_k: int = 10,
        collection: Optional[str] = None,
        answer_override: Any = _ANSWER_OVERRIDE_MISSING,
        case_index: Optional[int] = None,
        expected_metrics: Optional[Sequence[str]] = None,
        require_context_citations: bool = False,
    ) -> QueryResult:
        """Evaluate a single test case.

        Args:
            test_case: The test case to evaluate.
            top_k: Number of results to retrieve.
            collection: Optional collection filter.
            answer_override: User-provided answer value. When present, it must
                be an exact, non-empty str and is never silently replaced.
            case_index: Zero-based identity in the loaded golden test set.
            expected_metrics: Exact metric names expected for this run.
            require_context_citations: Enforce the suite's explicit citation policy.

        Returns:
            QueryResult with metrics for this test case.
        """
        t0 = time.monotonic()
        qr = QueryResult(query=test_case.query, case_index=case_index,
                         citation_required=require_context_citations,
                         reference_answer=test_case.reference_answer,
                         case_metadata=test_case.metadata())
        expected = self._validated_metric_names(expected_metrics or [])

        # Step 1: Retrieve chunks
        retrieval_trace = TraceContext(trace_type="query")
        query_result = None
        if self.query_service is not None:
            from src.core.query_engine.query_service import QueryRequest
            query_response = self.query_service.query(QueryRequest(query=test_case.query,
                collection=collection or self.query_service.collection, top_k=top_k,
                answer=self.answer_generator is not None and answer_override is _ANSWER_OVERRIDE_MISSING,
                require_citations=require_context_citations))
            query_result = query_response.to_dict()
            qr.query_observation = deepcopy(query_result)
            retrieved_chunks = query_result["results"]
            if query_result["execution_state"].get("retrieval_completed"):
                qr.retrieval_measurement = MeasurementState(status=OBSERVED, mode="empty" if not retrieved_chunks else "results")
            else:
                qr.retrieval_measurement = MeasurementState(status=FAILED_TO_MEASURE, mode="retrieval",
                    error_category=(query_result["error"] or {}).get("type", "QueryIncomplete"),
                    error_message=(query_result["error"] or {}).get("code", query_result["status"]))
            retrieval_trace.stages = query_result["trace"].get("stages", [])
        else:
            retrieved_chunks, qr.retrieval_measurement = self._retrieve(
                test_case.query, top_k, collection, trace=retrieval_trace,
            )
        retrieval_trace.finish()
        # 为什么要有它：按查询传递既有TraceContext，避免并发检索用共享last_observation串错成本。
        qr.embedding_observations = [deepcopy(stage["data"]) for stage in retrieval_trace.stages
            if stage["stage"] == "embedding"]
        qr.retrieved_chunk_ids = [
            self._get_chunk_id(c) for c in retrieved_chunks
        ]
        qr.retrieved_contexts = [
            self._snapshot_context(c) for c in retrieved_chunks
        ]

        # Step 2: Generate answer — prefer user override, then generator, then fallback
        if qr.retrieval_measurement.status == FAILED_TO_MEASURE:
            answer = None
            qr.answer_measurement = MeasurementState(
                status=NOT_APPLICABLE,
                mode="retrieval_failure",
            )
        elif answer_override is not _ANSWER_OVERRIDE_MISSING:
            answer, qr.answer_measurement = self._measure_answer_value(
                answer_override,
                mode="answer_override",
            )
        elif query_result is not None and self.answer_generator is not None:
            answer = query_result["answer"]
            qr.answer_observation = deepcopy(query_response.evidence.get("answer_observation", query_result["answer_observation"]))
            if answer is not None and (query_result["status"] == "completed" or query_result["citation_validation"]["status"] == "failed"):
                answer, qr.answer_measurement = self._measure_answer_value(answer, mode="answer_generator")
            else:
                qr.answer_measurement = MeasurementState(status=FAILED_TO_MEASURE, mode="answer_generator",
                    error_category=(query_result["error"] or {}).get("type", "QueryIncomplete"),
                    error_message=(query_result["error"] or {}).get("code", query_result["status"]))
        else:
            answer, qr.answer_measurement = self._generate_answer(
                test_case.query,
                retrieved_chunks,
                require_context_citations=require_context_citations,
            )
            # 为什么要有它：生成器复用于多题，当前调用观测必须复制并与该题绑定，失败成本仍保留。
            observation = getattr(self.answer_generator, "last_observation", None)
            if isinstance(observation, dict):
                qr.answer_observation = deepcopy(observation)
        qr.generated_answer = answer
        if require_context_citations and qr.answer_measurement.status == OBSERVED:
            qr.citation_measurement = self._measure_context_citations(answer, retrieved_chunks)

        # Step 3: Build ground truth
        ground_truth = (
            {"ids": test_case.expected_chunk_ids}
            if test_case.expected_chunk_ids
            else None
        )

        prerequisite_statuses: Dict[str, Optional[str]] = {
            metric: self._metric_prerequisite_status(metric, test_case, qr)
            for metric in expected
        }
        qr.metric_measurements.update(
            {
                metric: status
                for metric, status in prerequisite_statuses.items()
                if status is not None
            }
        )

        # Step 4: Evaluate. A retrieval failure blocks every current evaluator
        # because BaseEvaluator.evaluate requires retrieved_chunks. If every
        # declared metric is not applicable, do not invoke an evaluator merely
        # to manufacture scores outside its prerequisites.
        # Scope: explicitly disabled legacy scorer. 为什么要有它：统一评测接管时，未启用的旧评分器应为不适用，不能因空分数字典制造假故障。
        if type(self.evaluator) is NoneEvaluator:
            qr.evaluator_measurement = MeasurementState(status=NOT_APPLICABLE, mode="disabled")
            qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
            return qr

        should_evaluate = qr.retrieval_measurement.status != FAILED_TO_MEASURE
        if expected and all(
            status in {FAILED_TO_MEASURE, NOT_APPLICABLE}
            for status in prerequisite_statuses.values()
        ):
            should_evaluate = False

        composite_children = self._applicable_composite_children(
            test_case,
            qr,
        )
        if composite_children == []:
            should_evaluate = False

        if not should_evaluate:
            qr.evaluator_measurement = MeasurementState(
                status=NOT_APPLICABLE,
                mode=(
                    "retrieval_failure"
                    if qr.retrieval_measurement.status == FAILED_TO_MEASURE
                    else "metric_prerequisites"
                ),
            )
            qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
            return qr

        try:
            metrics, child_errors = self._invoke_evaluator(
                query=test_case.query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=answer,
                ground_truth=ground_truth,
                composite_children=composite_children,
            )
            if not isinstance(metrics, dict):
                raise TypeError(
                    "Evaluator must return a dictionary of metric scores."
                )

            invalid_metrics: List[str] = []
            for metric, value in metrics.items():
                if not isinstance(metric, str) or not metric:
                    raise TypeError(
                        "Evaluator metric names must be non-empty strings."
                    )
                metric_name = metric
                prerequisite = prerequisite_statuses.get(metric_name)
                if prerequisite in {FAILED_TO_MEASURE, NOT_APPLICABLE}:
                    qr.metric_measurements[metric_name] = prerequisite
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    qr.metric_measurements[metric_name] = FAILED_TO_MEASURE
                    invalid_metrics.append(metric_name)
                    continue
                qr.metrics[metric_name] = float(value)
                qr.metric_measurements[metric_name] = OBSERVED

            missing = [
                metric
                for metric in expected
                if qr.metric_measurements.get(metric) not in {
                    OBSERVED,
                    NOT_APPLICABLE,
                    FAILED_TO_MEASURE,
                }
            ]
            for metric in missing:
                qr.metric_measurements[metric] = FAILED_TO_MEASURE

            applicable_expected = [
                metric
                for metric in expected
                if prerequisite_statuses.get(metric) is None
            ]
            failed_expected = [
                metric
                for metric in applicable_expected
                if qr.metric_measurements.get(metric) != OBSERVED
            ]
            if (
                child_errors
                or invalid_metrics
                or failed_expected
                or (not metrics and not expected)
            ):
                if child_errors:
                    category, message = self._composite_child_error_details(
                        child_errors
                    )
                elif invalid_metrics:
                    category = "InvalidMetricValue"
                    message = "Invalid metric values: " + ", ".join(invalid_metrics)
                elif failed_expected:
                    category = "MissingMetrics"
                    message = "Evaluator did not measure: " + ", ".join(failed_expected)
                else:
                    category = "EmptyMetrics"
                    message = "Evaluator returned no metrics."
                qr.evaluator_measurement = MeasurementState(
                    status=FAILED_TO_MEASURE,
                    mode="evaluation",
                    error_category=category,
                    error_message=message,
                )
            else:
                qr.evaluator_measurement = MeasurementState(
                    status=OBSERVED,
                    mode="evaluation",
                )
        except Exception as exc:
            logger.warning("Evaluation failed for '%s': %s", test_case.query[:40], exc)
            qr.metrics = {}
            qr.evaluator_measurement = MeasurementState.from_exception(
                exc,
                mode="evaluation",
            )
            for metric in expected:
                if prerequisite_statuses.get(metric) != NOT_APPLICABLE:
                    qr.metric_measurements[metric] = FAILED_TO_MEASURE

        qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return qr

    def _retrieve(
        self,
        query: str,
        top_k: int,
        collection: Optional[str],
        *,
        trace: TraceContext | None = None,
    ) -> tuple[List[Any], MeasurementState]:
        """Retrieve chunks using HybridSearch + optional Reranking.

        An unconfigured retriever is not applicable. A successful empty search
        is observed and remains distinct from a retrieval exception.
        """
        if self.hybrid_search is None:
            logger.warning("No HybridSearch configured; returning empty results.")
            return [], MeasurementState(
                status=NOT_APPLICABLE,
                mode="not_configured",
            )

        try:
            # Retrieve more candidates if reranker is enabled
            has_reranker = self.reranker is not None and getattr(self.reranker, 'is_enabled', False)
            initial_top_k = top_k * 2 if has_reranker else top_k

            trace_kwargs = {}
            if trace is not None:
                # 为什么要有它：观测是可选能力；先看签名，避免TypeError后重试造成双重检索。
                try:
                    parameters = inspect.signature(self.hybrid_search.search).parameters
                    if "trace" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD
                                                     for p in parameters.values()):
                        trace_kwargs["trace"] = trace
                except (TypeError, ValueError):
                    pass
            results = self.hybrid_search.search(
                query=query,
                top_k=initial_top_k,
                **trace_kwargs,
            )
            results = results if isinstance(results, list) else results.results

            # Apply reranking if enabled
            if has_reranker and results:
                rerank_result = self.reranker.rerank(query=query, results=results, top_k=top_k)
                results = rerank_result.results

            return results, MeasurementState(
                status=OBSERVED,
                mode="empty" if not results else "results",
            )
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", query[:40], exc)
            return [], MeasurementState.from_exception(exc, mode="retrieval")

    def _generate_answer(
        self,
        query: str,
        chunks: List[Any],
        *,
        require_context_citations: bool = False,
    ) -> tuple[Optional[str], MeasurementState]:
        """Generate an answer from retrieved chunks.

        If a custom answer_generator is provided, use it.
        Otherwise, concatenate chunk texts as an explicitly synthetic fallback.
        A configured generator failure is not replaced by the fallback.
        """
        if self.answer_generator is not None:
            try:
                answer, measurement = self._measure_answer_value(
                    self.answer_generator(query, chunks),
                    mode="answer_generator",
                )
                return answer, measurement
            except Exception as exc:
                logger.warning("Answer generation failed: %s", exc)
                return None, MeasurementState.from_exception(
                    exc,
                    mode="answer_generator",
                )

        if not self.synthetic_fallback:
            return None, MeasurementState(status=NOT_APPLICABLE, mode="not_configured")

        # Fallback: concatenate chunk texts for explicit library previews only.
        texts = []
        for c in chunks:
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, dict):
                texts.append(c.get("text", str(c)))
            elif hasattr(c, "text"):
                texts.append(str(getattr(c, "text")))
            else:
                texts.append(str(c))

        return " ".join(texts[:5]), MeasurementState(
            status=NOT_APPLICABLE,
            mode="synthetic_fallback",
        )

    @staticmethod
    def _measure_context_citations(
        answer: Optional[str],
        chunks: List[Any],
    ) -> MeasurementState:
        """Record explicit citation-policy status without erasing model output."""
        assert answer is not None
        from src.core.query_engine.citations import citation_error
        failure = citation_error(answer, len(chunks))
        if failure:
            return MeasurementState(
                status=FAILED_TO_MEASURE,
                mode="citation_policy",
                error_category=failure[0],
                error_message=failure[1],
            )
        return MeasurementState(status=OBSERVED, mode="citation_policy")

    @staticmethod
    def _measure_answer_value(
        value: Any,
        *,
        mode: str,
    ) -> tuple[Optional[str], MeasurementState]:
        """Accept only an exact, non-empty str as an observed answer."""
        if type(value) is str and value.strip():
            return value, MeasurementState(status=OBSERVED, mode=mode)

        if type(value) is str:
            detail = "received an empty or whitespace-only str"
        else:
            detail = f"received type {type(value).__name__}"
        return None, MeasurementState(
            status=FAILED_TO_MEASURE,
            mode=mode,
            error_category="InvalidAnswerValue",
            error_message=f"Answer must be an exact, non-empty str; {detail}.",
        )

    def _applicable_composite_children(
        self,
        test_case: GoldenTestCase,
        result: QueryResult,
    ) -> Optional[List[BaseEvaluator]]:
        """Select Composite children before any evaluator side effect occurs.

        ``None`` means the configured evaluator is not Composite. An empty
        list means Composite has no child whose declared metrics are safe and
        applicable for this case. Children without metrics exposed by the
        current evaluator interfaces retain Composite's ordinary behavior.
        """
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator

        if not isinstance(self.evaluator, CompositeEvaluator):
            return None

        selected: List[BaseEvaluator] = []
        for child in self.evaluator.evaluators:
            metrics = self._metric_names_from_evaluator(child)
            if not metrics:
                selected.append(child)
                continue

            prerequisites = {
                metric: self._metric_prerequisite_status(
                    metric,
                    test_case,
                    result,
                )
                for metric in metrics
            }
            blocked_answer_metric = any(
                metric in _RAGAS_ANSWER_METRICS
                and prerequisites[metric] in {FAILED_TO_MEASURE, NOT_APPLICABLE}
                for metric in metrics
            )
            all_metrics_blocked = all(
                status in {FAILED_TO_MEASURE, NOT_APPLICABLE}
                for status in prerequisites.values()
            )
            if blocked_answer_metric or all_metrics_blocked:
                continue
            selected.append(child)

        return selected

    def _invoke_evaluator(
        self,
        *,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str],
        ground_truth: Optional[Any],
        composite_children: Optional[Sequence[BaseEvaluator]],
    ) -> tuple[Dict[str, float], List[tuple[str, Exception]]]:
        """Invoke one evaluator or the preselected Composite child sequence."""
        if composite_children is None:
            metrics = self.evaluator.evaluate(  # type: ignore[union-attr]
                query=query,
                retrieved_chunks=retrieved_chunks,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
            )
            return metrics, []

        # Match CompositeEvaluator's validation, merge order, duplicate-key
        # overwrite behavior, partial-failure tolerance, and all-failed error.
        self.evaluator.validate_query(query)  # type: ignore[union-attr]
        self.evaluator.validate_retrieved_chunks(retrieved_chunks)  # type: ignore[union-attr]
        merged: Dict[str, float] = {}
        errors: List[tuple[str, Exception]] = []

        for child in composite_children:
            name = type(child).__name__
            try:
                child_metrics = child.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                )
                for key, value in child_metrics.items():
                    if key in merged:
                        logger.warning(
                            "Metric '%s' produced by multiple evaluators; "
                            "overwriting with value from %s",
                            key,
                            name,
                        )
                    merged[key] = value
            except Exception as exc:
                logger.warning("%s failed: %s", name, exc)
                errors.append((name, exc))

        if not merged and errors:
            raise RuntimeError(
                "All sub-evaluators failed:\n"
                + "\n".join(f"{name} failed: {exc}" for name, exc in errors)
            )

        return merged, errors

    @staticmethod
    def _composite_child_error_details(
        errors: Sequence[tuple[str, Exception]],
    ) -> tuple[str, str]:
        """Preserve exact child exception details in one measurement slot."""
        if len(errors) == 1:
            name, exc = errors[0]
            return type(exc).__name__, f"{name} failed: {exc}"

        return (
            "CompositeChildFailures",
            "\n".join(
                f"{name} failed [{type(exc).__name__}]: {exc}"
                for name, exc in errors
            ),
        )

    def _declared_metric_names(self) -> List[str]:
        """Read exact metric declarations exposed by current evaluators."""
        names = list(self.expected_metrics)
        evaluator_metrics = self._metric_names_from_evaluator(self.evaluator)
        if evaluator_metrics:
            names.extend(evaluator_metrics)
        else:
            evaluation_settings = getattr(self.settings, "evaluation", None)
            configured = getattr(evaluation_settings, "metrics", None)
            if isinstance(configured, (list, tuple)):
                names.extend(configured)
        return self._validated_metric_names(names)

    @classmethod
    def _metric_names_from_evaluator(
        cls,
        evaluator: Optional[BaseEvaluator],
    ) -> List[str]:
        if evaluator is None:
            return []

        from src.libs.evaluator.custom_evaluator import CustomEvaluator
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator
        from src.observability.evaluation.ragas_evaluator import RagasEvaluator

        if isinstance(evaluator, CustomEvaluator):
            return cls._validated_metric_names(evaluator.metrics)
        if isinstance(evaluator, RagasEvaluator):
            return cls._validated_metric_names(evaluator._metric_names)
        if isinstance(evaluator, CompositeEvaluator):
            names: List[str] = []
            for child in evaluator.evaluators:
                names.extend(cls._metric_names_from_evaluator(child))
            return cls._validated_metric_names(names)
        return []

    @staticmethod
    def _validated_metric_names(metrics: Sequence[str]) -> List[str]:
        names: List[str] = []
        for metric in metrics:
            if not isinstance(metric, str) or not metric:
                raise ValueError("Metric names must be non-empty strings.")
            if metric not in names:
                names.append(metric)
        return names

    @staticmethod
    def _metric_prerequisite_status(
        metric: str,
        test_case: GoldenTestCase,
        result: QueryResult,
    ) -> Optional[str]:
        if result.retrieval_measurement.status == FAILED_TO_MEASURE:
            return FAILED_TO_MEASURE
        if result.retrieval_measurement.status == NOT_APPLICABLE:
            if metric in _RAGAS_CONTEXT_METRICS or metric in _CUSTOM_RETRIEVAL_METRICS:
                return NOT_APPLICABLE
            if metric != "answer_relevancy":
                return NOT_APPLICABLE
        if metric in _CUSTOM_RETRIEVAL_METRICS and not test_case.expected_chunk_ids:
            return NOT_APPLICABLE
        if metric in _RAGAS_ANSWER_METRICS:
            if result.answer_measurement.status == FAILED_TO_MEASURE:
                return FAILED_TO_MEASURE
            if result.answer_measurement.status == NOT_APPLICABLE:
                return NOT_APPLICABLE
        return None

    @staticmethod
    def _all_metric_names(
        results: Sequence[QueryResult],
        declared_metrics: Sequence[str],
    ) -> List[str]:
        names = list(declared_metrics)
        for result in results:
            names.extend(result.metrics.keys())
            names.extend(result.metric_measurements.keys())
        return sorted(set(names))

    @staticmethod
    def _finalise_metric_measurements(
        results: Sequence[QueryResult],
        metric_names: Sequence[str],
    ) -> None:
        """Backfill every case/metric denominator position."""
        for result in results:
            for metric in metric_names:
                status = result.metric_measurements.get(metric)
                if status is None:
                    if metric in result.metrics:
                        status = OBSERVED
                    elif result.retrieval_measurement.status == FAILED_TO_MEASURE:
                        status = FAILED_TO_MEASURE
                    elif result.retrieval_measurement.status == NOT_APPLICABLE:
                        status = NOT_APPLICABLE
                    elif (
                        metric in _RAGAS_ANSWER_METRICS
                        and result.answer_measurement.status == FAILED_TO_MEASURE
                    ):
                        status = FAILED_TO_MEASURE
                    elif (
                        metric in _RAGAS_ANSWER_METRICS
                        and result.answer_measurement.status == NOT_APPLICABLE
                    ):
                        status = NOT_APPLICABLE
                    elif result.evaluator_measurement.status == NOT_APPLICABLE:
                        status = NOT_APPLICABLE
                    else:
                        status = FAILED_TO_MEASURE
                    result.metric_measurements[metric] = status

                if status != OBSERVED:
                    result.metrics.pop(metric, None)

    @staticmethod
    def _aggregate_metric_measurements(
        results: Sequence[QueryResult],
        metric_names: Sequence[str],
    ) -> Dict[str, Dict[str, int]]:
        details: Dict[str, Dict[str, int]] = {}
        total = len(results)
        for metric in metric_names:
            observed = sum(
                result.metric_measurements.get(metric) == OBSERVED
                for result in results
            )
            failed = sum(
                result.metric_measurements.get(metric) == FAILED_TO_MEASURE
                for result in results
            )
            not_applicable = sum(
                result.metric_measurements.get(metric) == NOT_APPLICABLE
                for result in results
            )
            details[metric] = {
                "total": total,
                "applicable": observed + failed,
                "observed": observed,
                "scored": observed,
                "failed_to_measure": failed,
                "not_applicable": not_applicable,
            }
        return details

    def _snapshot_context(self, chunk: Any) -> dict[str, Any]:
        """Copy the exact retrieval fields before generation/evaluation consumes them.

        HybridSearch returns RetrievalResult (chunk_id, text, score, metadata).
        The library also accepts text strings and dictionaries. Missing fields
        in those representations stay absent values; no content is reconstructed
        from a chunk ID, reference answer, or Settings.
        """
        if isinstance(chunk, str):
            text, score, metadata = chunk, None, {}
        elif isinstance(chunk, dict):
            text = chunk.get("text")
            score = chunk.get("score")
            metadata = chunk.get("metadata", {})
        else:
            text = getattr(chunk, "text", None)
            score = getattr(chunk, "score", None)
            metadata = getattr(chunk, "metadata", {})
        return deepcopy({
            "chunk_id": self._get_chunk_id(chunk),
            "text": text,
            "score": score,
            "metadata": metadata,
        })

    def _get_chunk_id(self, chunk: Any) -> str:
        """Extract chunk ID from various representations."""
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            for key in ("id", "chunk_id"):
                if key in chunk:
                    return str(chunk[key])
            return str(chunk)
        if hasattr(chunk, "chunk_id"):
            return str(getattr(chunk, "chunk_id"))
        if hasattr(chunk, "id"):
            return str(getattr(chunk, "id"))
        return str(chunk)

    @staticmethod
    def _aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
        """Compute average metrics across all query results.

        Args:
            results: List of QueryResult with per-query metrics.

        Returns:
            Dictionary of average metric values.
        """
        if not results:
            return {}

        # Collect all metric keys
        all_keys: set[str] = set()
        for qr in results:
            all_keys.update(qr.metrics.keys())

        # Average each metric over observed scores only. Measurement failures
        # remain in EvalReport.metric_measurements and are never coerced to 0.
        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [
                qr.metrics[key]
                for qr in results
                if key in qr.metrics
                and qr.metric_measurements.get(key, OBSERVED) == OBSERVED
            ]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages
