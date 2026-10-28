"""Selected public material is separate from evaluator-only patches and labels."""
import csv
import io
import json

import pytest

from scripts import prepare_repository_task as entry


def inputs(root, *, selected="repo_one-1", labels="['target_test']",
           pass_labels="['preservation_test']", preservation_policy=None, duplicate=False):
    rows = [{"instance_id": name, "repo": name.split("-")[0] + "/library", "base_commit": "1" * 40,
        "created_at": "2024-01-01T00:00:00Z", "problem_statement": "public request " + name,
        "patch": "PRIVATE_REFERENCE_" + name, "test_patch": "PRIVATE_TEST_" + name,
        "FAIL_TO_PASS": labels, "PASS_TO_PASS": pass_labels} for name in ("repo_one-1", "repo_two-2")]
    if duplicate:
        rows.append(rows[0])
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    source = root / "dataset.csv"
    source.write_text(text.getvalue(), encoding="utf-8", newline="")
    metadata = root / "metadata.json"
    identity_keys = ("instance_id", "repo", "base_commit", "created_at")
    metadata.write_text(json.dumps({"rows": [{key: row[key] for key in identity_keys} for row in rows],
        "source_ref": entry._ref(source)}), encoding="utf-8")
    identity = next({key: row[key] for key in identity_keys} for row in rows if row["instance_id"] == selected)
    selection = root / "selection.json"
    selected_value = {**identity, "schema_version": "external-acceptance-selection.v2",
        "status": "metadata_selected_task_content_unopened", "metadata_path": "metadata.json",
        "metadata_sha256": entry._ref(metadata)["sha256"], "dataset_source_sha256": entry._ref(source)["sha256"]}
    if preservation_policy is not None:
        selected_value["public_preservation_policy"] = preservation_policy
    selection.write_text(json.dumps(selected_value), encoding="utf-8")
    return selection, source, metadata


@pytest.mark.parametrize("task_id", ["repo_one-1", "repo_two-2"])
def test_same_preparer_opens_other_repository_without_public_private_mix(tmp_path, task_id):
    selection, source, metadata = inputs(tmp_path, selected=task_id)
    originals = [entry._ref(path) for path in (selection, source, metadata)]
    output = tmp_path / "new-output"
    result = entry.prepare(selection, output, repository_root=tmp_path)
    assert set(result) == {"opening_ref", "public_problem_ref"}
    problem = (output / "solver-input/problem_statement.txt").read_text(encoding="utf-8")
    assert problem == "public request " + task_id
    assert "PRIVATE_" not in problem and "target_test" not in problem
    assert (output / "evaluation-only/reference.patch").read_text() == "PRIVATE_REFERENCE_" + task_id
    assert (output / "evaluation-only/test.patch").read_text() == "PRIVATE_TEST_" + task_id
    opening = json.loads((output / "task-opened.json").read_text())
    assert opening["dataset_ref"] == originals[1]
    assert opening["model_calls"] == 0 and opening["builds"] == 0
    assert opening["reference_exposed_to_solver"] is False
    assert [entry._ref(path) for path in (selection, source, metadata)] == originals
    with pytest.raises(ValueError, match="FRESH_OUTPUT_ROOT_REQUIRED"):
        entry.prepare(selection, output, repository_root=tmp_path)


@pytest.mark.parametrize("change", ["metadata_hash", "dataset_hash", "identity", "duplicate"])
def test_changed_sources_and_ambiguous_identity_refused_before_outputs(tmp_path, change):
    selection, source, metadata = inputs(tmp_path, duplicate=change == "duplicate")
    if change == "metadata_hash":
        metadata.write_text("{}")
    elif change == "dataset_hash":
        source.write_text("changed")
    elif change == "identity":
        value = json.loads(selection.read_text())
        selection.write_text(json.dumps({**value, "repo": "another/library"}))
    output = tmp_path / "new-output"
    with pytest.raises(ValueError):
        entry.prepare(selection, output, repository_root=tmp_path)
    assert not output.exists()


@pytest.mark.parametrize("labels", ["[]", "['preservation_test']", "['duplicate','duplicate']", "PRIVATE_BAD_LABEL("])
def test_invalid_private_labels_never_appear_in_cli_error(tmp_path, monkeypatch, capsys, labels):
    selection, _, _ = inputs(tmp_path, labels=labels)
    actual = entry.prepare
    monkeypatch.setattr(entry, "prepare", lambda selection, output: actual(selection, output, repository_root=tmp_path))
    output = tmp_path / "new-output"
    assert entry.main(["--selection", str(selection), "--output-root", str(output)]) == 2
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "error_type": "ValueError"}
    assert not output.exists()


def test_empty_pass_to_pass_requires_explicit_native_public_suite_policy(tmp_path):
    selection, _, _ = inputs(tmp_path, pass_labels="[]")
    output = tmp_path / "new-output"
    with pytest.raises(ValueError, match="SELECTED_TASK_MATERIAL_INVALID"):
        entry.prepare(selection, output, repository_root=tmp_path)
    assert not output.exists()

    value = json.loads(selection.read_text(encoding="utf-8"))
    value["public_preservation_policy"] = "native_public_suite_excluding_fail_to_pass"
    selection.write_text(json.dumps(value), encoding="utf-8")
    result = entry.prepare(selection, output, repository_root=tmp_path)
    opening = json.loads((output / "task-opened.json").read_text(encoding="utf-8"))
    private = json.loads((output / "evaluation-only/task.json").read_text(encoding="utf-8"))
    assert result["opening_ref"] == entry._ref(output / "task-opened.json")
    assert opening["public_preservation_policy"] == private["public_preservation_policy"] == (
        "native_public_suite_excluding_fail_to_pass")
    assert private["PASS_TO_PASS"] == []
