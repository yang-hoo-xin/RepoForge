"""Bound labels and complete report samples, without model calls.

Scope: existing flat criterion/expected label documents and unregistered legacy
diagnostics. 为什么要有它：来源文件和样本归属不能在元评估时被忽略或提升为独立资格。
"""
from dataclasses import asdict
import json
from pathlib import Path
import socket

import pytest

from src.observability.evaluation.meta_evaluation import evaluate_evaluator
from src.observability.evaluation.sample_registry import Sample
from src.observability.evaluation.unified import Binding, Criterion, EvaluationSession, save
from tests.unit.test_evaluation_meta import item, reports
from tests.unit.test_unified_evaluation import adapter


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Meta label verification must not call a model or network.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def _report(tmp_path, *, dataset="judge-screen-v1", status="failed"):
    sample = Sample(dataset, "case", "service-contracts", "constructed", "calibration", "ai", True)
    payload = {"answer": "Supported assertion, missing another requested detail."}
    candidate = save(tmp_path / "candidate.json", payload)
    binding = Binding(dataset + ":case", "conversation", "run", candidate)
    report = EvaluationSession(
        tmp_path / "reports",
        criteria=[Criterion("grounding", "grounding", "judge", method="trial_judge")],
        evaluators={"judge": adapter(status)}, rubric_version="v1", sample=sample,
    ).evaluate(binding, payload)
    return report, sample


def _labels(tmp_path, **overrides):
    # Scope: actual retained label shape. 为什么要有它：兼容现有AI标签JSON，不强制另建注册表或人工标注。
    value = {
        "criterion_id": "grounding", "expected": {"case": "passed"},
        "label_origin": "current_assistant_review_of_retained_source",
        "partition": "calibration", "tuning_exposed": True,
        "human_verified": False, "independent_blind_review": False,
        "automatic_judge_qualified": False, "release_authority": False,
    }
    value.update(overrides)
    return save(tmp_path / "labels.json", value)


def _item(report, labels=None, **overrides):
    value = {
        "case_id": "case", "family_id": "service-contracts",
        "expected_status": "passed", "criterion_id": "grounding",
        "report_ref": report["report_ref"],
    }
    if labels is not None:
        value["label_source_ref"] = labels
    value.update(overrides)
    return value


def test_existing_ai_label_document_binds_case_criterion_and_full_sample(tmp_path):
    report, sample = _report(tmp_path)
    labels = _labels(tmp_path)
    summary = evaluate_evaluator([_item(report, labels)])
    row = summary["cases"][0]
    assert row["sample"] == asdict(sample)
    assert row["dataset_id"] == sample.dataset_id
    assert row["label_provenance"]["label_source_ref"] == labels
    assert row["label_provenance"]["label_origin"] == "current_assistant_review_of_retained_source"
    assert row["label_provenance"]["expected_status"] == "passed"
    assert row["label_provenance"]["criterion_id"] == "grounding"
    assert row["label_provenance"]["human_verified"] is False
    assert row["label_provenance"]["independent_blind_review"] is False
    assert row["label_provenance"]["source_material_verified"] is False
    assert summary["false_reject"]["count"] == 1
    assert summary["label_provenance_counts"] == {"bound": 1}
    assert summary["qualification_status"] == "not_established"
    assert summary["automatic_judge_qualified"] is summary["release_authority"] is False


@pytest.mark.parametrize(("override", "error"), [
    ({"criterion_id": "fulfillment"}, "META_LABEL_CRITERION_MISMATCH"),
    ({"expected": {"other": "passed"}}, "META_LABEL_CASE_MISSING"),
    ({"expected": {"case": "failed"}}, "META_LABEL_EXPECTATION_MISMATCH"),
    ({"dataset_id": "unrelated-dataset"}, "META_LABEL_SAMPLE_METADATA_MISMATCH"),
    ({"family_id": "unrelated-family"}, "META_LABEL_SAMPLE_METADATA_MISMATCH"),
    ({"partition": "holdout"}, "META_LABEL_SAMPLE_METADATA_MISMATCH"),
    ({"tuning_exposed": False}, "META_LABEL_SAMPLE_METADATA_MISMATCH"),
    ({"human_verified": "false"}, "META_LABEL_PROVENANCE_FLAG_INVALID"),
    ({"independent_blind_review": 1}, "META_LABEL_PROVENANCE_FLAG_INVALID"),
])
def test_supplied_label_mismatch_cannot_silently_change_meta_result(tmp_path, override, error):
    report, _ = _report(tmp_path)
    labels = _labels(tmp_path, **override)
    with pytest.raises(ValueError, match=error):
        evaluate_evaluator([_item(report, labels)])


def test_label_source_hash_is_checked_before_labels_are_used(tmp_path):
    report, _ = _report(tmp_path)
    labels = _labels(tmp_path)
    path = Path(labels["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        evaluate_evaluator([_item(report, labels)])


def test_legacy_unregistered_items_remain_explicitly_unverified(tmp_path):
    report = reports(tmp_path, ["failed"])[0]
    summary = evaluate_evaluator([item(report)])
    row = summary["cases"][0]
    assert row["sample"] is None and row["dataset_id"] is None
    assert row["sample_identity_status"] == "unregistered"
    assert row["label_provenance"]["status"] == "unverified"
    assert row["label_provenance"]["human_verified"] is None
    assert summary["false_reject"]["count"] == summary["sample_count"] == 1
    assert summary["qualification_status"] == "not_established"
    assert summary["release_authority"] is False


def test_origin_absence_and_human_claim_never_grant_qualification(tmp_path):
    report, _ = _report(tmp_path)
    labels = save(tmp_path / "origin-unverified.json", {
        "criterion_id": "grounding", "expected": {"case": "passed"},
    })
    summary = evaluate_evaluator([_item(report, labels)])
    provenance = summary["cases"][0]["label_provenance"]
    assert provenance["status"] == "label_bound_origin_unverified"
    assert provenance["label_origin"] is provenance["human_verified"] is None
    claimed = _labels(tmp_path, label_origin="human", human_verified=True,
                      independent_blind_review=True, automatic_judge_qualified=True,
                      release_authority=True)
    summary = evaluate_evaluator([_item(report, claimed)])
    assert summary["cases"][0]["sample"]["label_origin"] == "ai"
    assert summary["automatic_judge_qualified"] is summary["release_authority"] is False


def test_cross_dataset_names_are_not_merged_into_one_sample_or_family(tmp_path):
    first, first_sample = _report(tmp_path / "first", dataset="dataset-one")
    second, second_sample = _report(tmp_path / "second", dataset="dataset-two")
    summary = evaluate_evaluator([_item(first), _item(second)])
    assert summary["sample_count"] == summary["family_count"] == 2
    assert summary["cases"][0]["sample"] == asdict(first_sample)
    assert summary["cases"][1]["sample"] == asdict(second_sample)
    assert summary["sample_identity_counts"] == {"report_bound": 2}
    with pytest.raises(ValueError, match="META_SAMPLE_IDENTITY_MISMATCH"):
        evaluate_evaluator([_item(first, dataset_id="dataset-two")])


def test_same_repeat_group_cannot_merge_different_dataset_identity(tmp_path):
    first, _ = _report(tmp_path / "first", dataset="dataset-one")
    second, _ = _report(tmp_path / "second", dataset="dataset-two")
    with pytest.raises(ValueError, match="META_REPEAT_INPUT_OR_RUBRIC_CHANGED"):
        evaluate_evaluator([
            _item(first, repeat_group="same"), _item(second, repeat_group="same"),
        ])


def test_normal_meta_cli_preserves_provenance_without_model_calls(tmp_path):
    from scripts.run_unified_evaluation import main

    report, sample = _report(tmp_path)
    labels = _labels(tmp_path)
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({
        "schema_version": "evaluation-meta-input.v1", "items": [_item(report, labels)],
    }), encoding="utf-8")
    output = tmp_path / "meta"
    assert main(["meta", "--manifest", str(manifest), "--output-root", str(output)]) == 0
    summary = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert summary["cases"][0]["sample"] == asdict(sample)
    assert summary["label_provenance_counts"] == {"bound": 1}
    assert summary["false_reject"]["count"] == 1
    assert summary["qualification_status"] == "not_established"
