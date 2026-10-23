"""Derived-source admission checks; no model, compiler or hidden evaluation calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[2] / "artifacts/eval_driven_20261023/evaluate_restsharp_revision.py"
    spec = importlib.util.spec_from_file_location("derived_restsharp_eval", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def states():
    name = "src/RestSharp/A.cs"

    def state(before, after, identity):
        return SimpleNamespace(contract_ref=identity + "-contract", contract={"baseline_manifest_ref": identity + "-base",
            "allowed_edit_paths": [name]}, load=lambda _: {"files": {name: {"text": before}}},
            validate_candidate=lambda _: {"candidate_source_id": identity},
            materialize=lambda ref: {name: before if ref == identity + "-base" else after},
            changes_from_baseline=lambda _: [{"path": name, "before_text": before, "after_text": after,
                "before_sha256": "before", "after_sha256": "after", "binding_id": "binding"}])

    return state("vendor", "first model", "first"), state("first model", "second model", "second")


@pytest.mark.parametrize("failure", ["identity", "text", "missing_file", "extra_file"])
def test_derived_identity_mismatch_rejected_before_application(module, tmp_path, monkeypatch, failure):
    original, current = states()
    started = {"starting_candidate_ref": "candidate-one", "starting_candidate_source_id": "first"}
    if failure == "identity":
        started["starting_candidate_source_id"] = "forged"
    else:
        texts = {"src/RestSharp/A.cs": "first model"}
        if failure == "text":
            texts["src/RestSharp/A.cs"] = "unattributed fix"
        elif failure == "missing_file":
            texts.clear()
        else:
            texts["extra.cs"] = "extra"
        current.materialize = lambda _: texts
    monkeypatch.setattr(module.old, "apply_changes", lambda *a: pytest.fail("Must reject before any write"))
    with pytest.raises(ValueError, match="STARTING_SOURCE_ID_CHANGED|DERIVED_BASELINE_TEXT_MISMATCH"):
        module.apply_composed_changes(tmp_path, original, current, started, "candidate-two")


def test_composed_application_uses_both_actual_preimages(module, tmp_path, monkeypatch):
    original, current = states()
    target = tmp_path / "src/RestSharp/A.cs"
    target.parent.mkdir(parents=True)
    target.write_text("vendor", encoding="utf-8")
    monkeypatch.setattr(module.old.subprocess, "check_output", lambda *a, **k: b"src/RestSharp/A.cs\0")
    result = module.apply_composed_changes(tmp_path, original, current,
        {"starting_candidate_ref": "candidate-one", "starting_candidate_source_id": "first"}, "candidate-two")
    assert target.read_text() == "second model"
    assert result["starting_model_changes"] and result["new_model_changes"]


def test_running_new_native_rejected_before_lineage_or_evaluation(module, tmp_path, monkeypatch):
    native = tmp_path / "execution.json"
    native.write_text(json.dumps({"schema_version": "task-agent-product-execution.v2", "status": "running", "unknown_effects": False}))
    monkeypatch.setattr(module.old, "verified_qualification", lambda: pytest.fail("No live task scoring"))
    with pytest.raises(ValueError, match="KNOWN_TERMINAL_EXECUTION_REQUIRED"):
        module.evaluate(tmp_path / "absent-started.json", native, "candidate", tmp_path / "output", None)
    assert not (tmp_path / "output").exists()
