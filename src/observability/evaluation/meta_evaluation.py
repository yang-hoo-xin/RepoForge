"""Re-evaluate one bound evaluator criterion from labeled reports; never call an LLM."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json

from .review_resolution import checked_report
from .sample_registry import Sample
from .unified import STATUSES, read_ref


def _identity(value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return {"sha256": value["sha256"]}
        return {key: _identity(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_identity(item) for item in value]
    return value


def _target_criterion(item, criteria, rows):
    requested = item.get("criterion_id")
    if requested is None:
        candidates = [
            (criterion, row)
            for criterion, row in zip(criteria, rows)
            if criterion.method == "trial_judge"
        ]
        # 范围：未显式选择的元评估输入。为什么要有它：只有唯一裁判维度时才能安全推断，不能用整份报告状态代替某个裁判结论。
        if len(candidates) != 1:
            raise ValueError("META_TARGET_CRITERION_REQUIRED")
        return candidates[0]
    matches = [
        (criterion, row)
        for criterion, row in zip(criteria, rows)
        if criterion.id == requested
    ]
    if len(matches) != 1:
        raise ValueError("META_TARGET_CRITERION_INVALID")
    return matches[0]


def _target_invocation(report, criterion):
    matches = []
    for reference in report["invocation_refs"]:
        invocation = json.loads(read_ref(reference))
        if invocation["binding"] != report["binding"]:
            raise ValueError("META_INVOCATION_BINDING_MISMATCH")
        if (
            invocation.get("evaluator") == criterion.evaluator
            and invocation.get("role") == criterion.role
            and criterion.id in invocation.get("criterion_ids", [])
        ):
            matches.append((reference, invocation))
    if not criterion.applicable:
        if matches:
            raise ValueError("META_NOT_APPLICABLE_EVALUATOR_INVOKED")
        return None, {"not_invoked": True}
    if len(matches) != 1:
        raise ValueError("META_TARGET_INVOCATION_INVALID")
    # 范围：同一调用内的多维评分。为什么要有它：评估器名称相同不代表执行了目标维度；未适用维度也不能继承其他维度的调用成本。
    reference, invocation = matches[0]
    return reference, invocation.get("evaluator_identity")


def _sample_identity(item, report):
    """Scope: report-owned sample identity. 为什么要有它：数据集同名样本不能合并，也不能补造未知来源。"""
    raw = report.get("sample")
    if raw is None:
        dataset_id = item.get("dataset_id")
        if dataset_id is not None and (not isinstance(dataset_id, str) or not dataset_id.strip()):
            raise ValueError("META_SAMPLE_DATASET_INVALID")
        return None, dataset_id
    try:
        sample = asdict(Sample(**raw))
    except (TypeError, ValueError) as error:
        raise ValueError("META_SAMPLE_METADATA_INVALID") from error
    if (
        sample["sample_id"] != item["case_id"]
        or sample["family_id"] != item["family_id"]
        or ("dataset_id" in item and item["dataset_id"] != sample["dataset_id"])
    ):
        raise ValueError("META_SAMPLE_IDENTITY_MISMATCH")
    return sample, sample["dataset_id"]


def _label_provenance(item, criterion, sample, dataset_id, cache):
    """Scope: supplied case/criterion labels. 为什么要有它：引用了标签文件就必须核对其实际内容，未给来源仍可作未核准诊断。"""
    reference = item.get("label_source_ref")
    if reference is None:
        return {
            "status": "unverified", "label_source_ref": None, "label_origin": None,
            "human_verified": None, "independent_blind_review": None,
            "reason": "Expected status is caller-supplied; no bound label source was provided.",
        }
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or any(not isinstance(reference[field], str) or not reference[field]
                   for field in ("path", "sha256"))):
        raise ValueError("META_LABEL_SOURCE_REF_INVALID")
    key = (reference["path"], reference["sha256"])
    if key not in cache:
        cache[key] = json.loads(read_ref(reference))
    labels = cache[key]
    if not isinstance(labels, dict):
        raise ValueError("META_LABEL_SOURCE_INVALID")
    if labels.get("criterion_id") != criterion.id:
        raise ValueError("META_LABEL_CRITERION_MISMATCH")
    expected = labels.get("expected")
    if not isinstance(expected, dict) or item["case_id"] not in expected:
        raise ValueError("META_LABEL_CASE_MISSING")
    if expected[item["case_id"]] != item["expected_status"]:
        raise ValueError("META_LABEL_EXPECTATION_MISMATCH")
    domain = {"dataset_id": dataset_id, "family_id": item["family_id"]}
    if sample is not None:
        domain.update(partition=sample["partition"], tuning_exposed=sample["tuning_exposed"])
    for field, actual in domain.items():
        if field in labels and (actual is None or type(labels[field]) is not type(actual)
                                or labels[field] != actual):
            raise ValueError("META_LABEL_SAMPLE_METADATA_MISMATCH")
    origin = labels.get("label_origin")
    if origin is not None and (not isinstance(origin, str) or not origin.strip()):
        raise ValueError("META_LABEL_ORIGIN_INVALID")
    for field in ("human_verified", "independent_blind_review", "tuning_exposed"):
        if field in labels and type(labels[field]) is not bool:
            raise ValueError("META_LABEL_PROVENANCE_FLAG_INVALID")
    # 范围：声明与核验边界。为什么要有它：文件哈希正确只证明标签绑定，不能证明其真值、人类参与或独立裁判资格。
    return {
        "status": "bound" if origin is not None else "label_bound_origin_unverified",
        "label_source_ref": deepcopy(reference), "label_origin": origin,
        "case_id": item["case_id"], "criterion_id": criterion.id,
        "expected_status": expected[item["case_id"]],
        "human_verified": labels.get("human_verified"),
        "independent_blind_review": labels.get("independent_blind_review"),
        "declared_partition": labels.get("partition"),
        "declared_tuning_exposed": labels.get("tuning_exposed"),
        "sample_domain_fields_bound": [field for field in domain if field in labels],
        "verification_scope": "source bytes and case/criterion/expected binding; provenance flags remain source declarations",
        "source_material_verified": False,
    }


def evaluate_evaluator(items):
    selected = set()
    report_hashes = set()
    confusion = Counter()
    groups = defaultdict(list)
    perturbations = defaultdict(list)
    invocation_costs = {}
    label_sources = {}
    cases = []

    for item in items:
        expected = item["expected_status"]
        if expected not in STATUSES:
            raise ValueError("META_EXPECTED_STATUS_INVALID")
        report, criteria = checked_report(item["report_ref"])
        criterion, row = _target_criterion(item, criteria, report["criteria"])
        selection = (item["report_ref"]["sha256"], criterion.id)
        if selection in selected:
            raise ValueError("META_DUPLICATED_EVALUATION")
        selected.add(selection)
        report_hashes.add(item["report_ref"]["sha256"])

        sample, dataset_id = _sample_identity(item, report)
        provenance = _label_provenance(item, criterion, sample, dataset_id, label_sources)

        invocation_ref, evaluator_identity = _target_invocation(report, criterion)
        if invocation_ref is not None:
            invocation = json.loads(read_ref(invocation_ref))
            invocation_costs.setdefault(invocation_ref["sha256"], invocation.get("cost"))

        actual = row["status"]
        confusion[(expected, actual)] += 1
        record = {
            "case_id": item["case_id"],
            "family_id": item["family_id"],
            "dataset_id": dataset_id,
            "sample": sample,
            "sample_identity_status": "report_bound" if sample is not None else "unregistered",
            "label_provenance": provenance,
            "criterion_id": criterion.id,
            "dimension": criterion.dimension,
            "evaluator": criterion.evaluator,
            "role": criterion.role,
            "expected": expected,
            "actual": actual,
            "report_status": report["status"],
            "report_ref": item["report_ref"],
            "candidate_ref": report["binding"]["candidate_ref"],
        }
        cases.append(record)

        signature = (
            json.dumps(sample, sort_keys=True) if sample is not None
            else json.dumps([dataset_id, item["case_id"], item["family_id"]]),
            report["binding"]["candidate_ref"]["sha256"],
            report["rubric"]["version"],
            json.dumps(asdict(criterion), sort_keys=True),
            json.dumps(_identity(evaluator_identity), sort_keys=True),
            expected,
        )
        if item.get("repeat_group"):
            groups[item["repeat_group"]].append((signature, actual))
        if item.get("invariance_group"):
            perturbations[item["invariance_group"]].append((expected, actual))

    repeats = []
    for name, rows in groups.items():
        if len({row[0] for row in rows}) != 1:
            raise ValueError("META_REPEAT_INPUT_OR_RUBRIC_CHANGED")
        repeats.append(
            {
                "group": name,
                "runs": len(rows),
                "distinct_verdicts": len({row[1] for row in rows}),
                "verdict_changed": len({row[1] for row in rows}) > 1,
            }
        )
    invariance = []
    for name, rows in perturbations.items():
        if len({row[0] for row in rows}) != 1:
            raise ValueError("META_INVARIANCE_EXPECTATION_CHANGED")
        invariance.append(
            {
                "group": name,
                "variants": len(rows),
                "distinct_verdicts": len({row[1] for row in rows}),
            }
        )

    positive = sum(value for (expected, _), value in confusion.items() if expected == "passed")
    negative = sum(value for (expected, _), value in confusion.items() if expected == "failed")
    false_accept = confusion[("failed", "passed")]
    false_reject = confusion[("passed", "failed")]

    def aggregate_cost(field):
        values = [
            cost.get(field) if isinstance(cost, dict) else None
            for cost in invocation_costs.values()
        ]
        return {
            "known_subtotal": sum(
                value for value in values if type(value) is int and value >= 0
            ),
            "complete": all(type(value) is int and value >= 0 for value in values),
        }

    # 范围：被显式绑定的 criterion 级裁判结果。为什么要有它：其他执行维度的失败不能掩盖裁判误放，裁判错误也不能被整份报告状态冒充为正确拒绝。
    return {
        "schema_version": "evaluation-meta-summary.v1",
        "cases": cases,
        "sample_count": len({(row["dataset_id"], row["case_id"]) for row in cases}),
        "family_count": len({(row["dataset_id"], row["family_id"]) for row in cases}),
        "identity_count_basis": "dataset/sample and dataset/family namespaces; not independent-task or independent-family qualification",
        "label_provenance_counts": dict(Counter(row["label_provenance"]["status"] for row in cases)),
        "sample_identity_counts": dict(Counter(row["sample_identity_status"] for row in cases)),
        "report_count": len(report_hashes),
        "evaluation_count": len(cases),
        "confusion": [
            {"expected": expected, "actual": actual, "count": count}
            for (expected, actual), count in sorted(confusion.items())
            if count
        ],
        "false_accept": {
            "count": false_accept,
            "known_negative_evaluations": negative,
        },
        "false_reject": {
            "count": false_reject,
            "known_positive_evaluations": positive,
        },
        "decisive_evaluations": sum(
            value
            for (_, actual), value in confusion.items()
            if actual in {"passed", "failed"}
        ),
        "evaluation_errors": sum(
            value
            for (_, actual), value in confusion.items()
            if actual == "evaluation_error"
        ),
        "insufficient_evidence": sum(
            value
            for (_, actual), value in confusion.items()
            if actual == "insufficient_evidence"
        ),
        "repeated_fixed_inputs": repeats,
        "invariance_groups": invariance,
        "evaluation_cost": {
            field: aggregate_cost(field)
            for field in (
                "model_calls",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
            )
        },
        "cost_basis": "unique selected evaluator invocations",
        "selection_policy": "explicit criterion_id or one unambiguous trial_judge criterion",
        "qualification_status": "not_established",
        "automatic_judge_qualified": False,
        "release_authority": False,
        "scope": "Registered development/calibration evidence; not a production accuracy estimate",
    }
