"""Prepare selected task using existing dataset, preserving separate input roles.

为什么要有它：求解器只取得公开题面，独立验收材料另存并绑定原始数据版本。
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELECTION = REPO / "docs/evidence/restsharp-2241-selection.json"
DATASET = REPO / "artifacts/eval_driven_20261015/business-evaluation/swe-sharp-qualification/official-input-v1/evaluation-only/official-swe-sharp-bench.csv"
ROOT = Path("C:/tmp/rfext-restsharp-2241-task")


def ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, indent=2).encode())
    return ref(path)


def main():
    selected = json.loads(SELECTION.read_bytes())
    metadata = REPO / selected["metadata_path"]
    if (ROOT.exists() or ref(metadata)["sha256"] != selected["metadata_sha256"]
            or ref(DATASET)["sha256"] != selected["dataset_source_sha256"]
            or selected["status"] != "metadata_selected_task_content_unopened"):
        raise ValueError("TASK_OPENING_PRECONDITION_FAILED")
    row = [item for item in json.loads(metadata.read_bytes())["rows"] if item["instance_id"] == selected["instance_id"]]
    if len(row) != 1 or any(row[0][key] != selected[key] for key in ("repo", "base_commit", "created_at")):
        raise ValueError("SELECTED_METADATA_CHANGED")
    csv.field_size_limit(4 * 1024 * 1024)
    with DATASET.open(encoding="utf-8", newline="") as stream:
        tasks = [item for item in csv.DictReader(stream) if item["instance_id"] == selected["instance_id"]]
    if len(tasks) != 1 or tasks[0]["base_commit"] != selected["base_commit"]:
        raise ValueError("SELECTED_TASK_CHANGED")
    task = tasks[0]
    labels = {key: ast.literal_eval(task[key]) for key in ("FAIL_TO_PASS", "PASS_TO_PASS")}
    if (not labels["FAIL_TO_PASS"] or any(not isinstance(values, list) or any(not isinstance(v, str) for v in values) for values in labels.values())
            or set(labels["FAIL_TO_PASS"]) & set(labels["PASS_TO_PASS"])):
        raise ValueError("INVALID_LABELS")
    public = save(ROOT / "solver-input/problem_statement.txt", task["problem_statement"].encode())
    reference = save(ROOT / "evaluation-only/reference.patch", task["patch"].encode())
    tests = save(ROOT / "evaluation-only/test.patch", task["test_patch"].encode())
    private = save(ROOT / "evaluation-only/task.json", {key: selected[key] for key in ("instance_id", "repo", "base_commit")} | labels)
    opened = save(ROOT / "task-opened.json", {"schema_version": "external-task-opening.v1",
        "selection_ref": ref(SELECTION), "dataset_ref": ref(DATASET), "public_problem_ref": public,
        "reference_patch_ref": reference, "test_patch_ref": tests, "task_ref": private,
        "reference_exposed_to_solver": False, "hidden_tests_exposed_to_solver": False,
        "model_calls": 0, "resource_policy": "meter_only"})
    print(json.dumps({"opening_ref": opened, "public_problem_ref": public}))


if __name__ == "__main__":
    main()
