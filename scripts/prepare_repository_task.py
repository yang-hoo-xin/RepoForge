"""Open one metadata-selected task without publishing evaluator content.

为什么要有它：不同仓库复用同一题源投影，求解材料和独立验收材料分开，不再复制单题准备脚本。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.task_agent_v2_state import _no_links

REPO = Path(__file__).resolve().parents[1]


def _ref(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    return _ref(path)


def prepare(selection_path, output_root, *, repository_root=REPO):
    selection_path, output_root = Path(selection_path), Path(output_root)
    if not selection_path.is_absolute() or not output_root.is_absolute():
        raise ValueError("ABSOLUTE_PATH_REQUIRED")
    _no_links(selection_path)
    _no_links(output_root)
    if output_root.exists() or not output_root.parent.is_dir():
        raise ValueError("FRESH_OUTPUT_ROOT_REQUIRED")
    selection_ref = _ref(selection_path)
    selection = read_bound_json(selection_ref)
    if (selection.get("schema_version") != "external-acceptance-selection.v2"
            or selection.get("status") != "metadata_selected_task_content_unopened"):
        raise ValueError("TASK_OPENING_PRECONDITION_FAILED")
    metadata_path = repository_root / selection["metadata_path"]
    _no_links(metadata_path)
    metadata_ref = {"path": str(metadata_path.resolve()), "sha256": selection["metadata_sha256"]}
    metadata = read_bound_json(metadata_ref)
    identity = {key: selection[key] for key in ("instance_id", "repo", "base_commit", "created_at")}
    rows = [row for row in metadata["rows"] if row["instance_id"] == identity["instance_id"]]
    if len(rows) != 1 or any(rows[0].get(key) != value for key, value in identity.items()):
        raise ValueError("SELECTED_METADATA_CHANGED")
    dataset_ref = metadata["source_ref"]
    dataset_path = Path(dataset_ref["path"])
    _no_links(dataset_path)
    if (not dataset_path.is_absolute() or _ref(dataset_path)["sha256"] != dataset_ref["sha256"]
            or dataset_ref["sha256"] != selection["dataset_source_sha256"]):
        raise ValueError("DATASET_CHANGED")
    # 为什么要有它：所有身份与字段检查先完成再创建输出，且永不把整行或解析异常中的私有内容打印出来。
    csv.field_size_limit(4 * 1024 * 1024)
    with dataset_path.open(encoding="utf-8-sig", newline="") as stream:
        tasks = [row for row in csv.DictReader(stream) if row["instance_id"] == identity["instance_id"]]
    if len(tasks) != 1 or any(tasks[0].get(key) != value for key, value in identity.items()):
        raise ValueError("SELECTED_TASK_CHANGED")
    task = tasks[0]
    preservation_policy = selection.get("public_preservation_policy", "dataset_pass_to_pass")
    try:
        labels = {key: ast.literal_eval(task[key]) for key in ("FAIL_TO_PASS", "PASS_TO_PASS")}
        structural_labels = (bool(labels["FAIL_TO_PASS"]) and all(isinstance(values, list)
            and all(isinstance(value, str) and value for value in values) and len(values) == len(set(values))
            for values in labels.values()) and not set(labels["FAIL_TO_PASS"]) & set(labels["PASS_TO_PASS"]))
        # Scope: the selected task must name the preservation source before evaluation-only material is opened.
        # 为什么要有它：空PASS_TO_PASS不能静默表示“无需保留”；原生公共套件是更强但不同的显式验收合同。
        valid_labels = structural_labels and (
            bool(labels["PASS_TO_PASS"]) and preservation_policy == "dataset_pass_to_pass"
            or not labels["PASS_TO_PASS"]
            and preservation_policy == "native_public_suite_excluding_fail_to_pass")
    except (ValueError, SyntaxError, TypeError, KeyError):
        valid_labels = False
    if not valid_labels or not all(isinstance(task.get(key), str) and task[key].strip()
            for key in ("problem_statement", "patch", "test_patch")):
        raise ValueError("SELECTED_TASK_MATERIAL_INVALID")
    if (_ref(selection_path) != selection_ref or _ref(metadata_path) != metadata_ref
            or _ref(dataset_path) != dataset_ref):
        raise ValueError("TASK_SOURCES_CHANGED_DURING_OPENING")
    output_root.mkdir()
    public = _save(output_root / "solver-input/problem_statement.txt", task["problem_statement"].encode("utf-8"))
    reference = _save(output_root / "evaluation-only/reference.patch", task["patch"].encode("utf-8"))
    tests = _save(output_root / "evaluation-only/test.patch", task["test_patch"].encode("utf-8"))
    private = _save(output_root / "evaluation-only/task.json", {
        **identity, **labels, "public_preservation_policy": preservation_policy})
    opening = _save(output_root / "task-opened.json", {
        "schema_version": "external-task-opening.v1", "selection_ref": selection_ref,
        "metadata_ref": metadata_ref, "dataset_ref": dataset_ref, "public_problem_ref": public,
        "reference_patch_ref": reference, "test_patch_ref": tests, "task_ref": private,
        "public_preservation_policy": preservation_policy,
        "reference_exposed_to_solver": False, "hidden_tests_exposed_to_solver": False,
        "public_extraction_is_not_independence_proof": True, "model_calls": 0, "builds": 0,
        "resource_policy": selection.get("resource_policy", "not_supplied")})
    return {"opening_ref": opening, "public_problem_ref": public}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.selection, args.output_root)
    except Exception as error:
        # 为什么要有它：CSV/标签来自评估侧，默认错误输出仅给类型，不泄漏隐藏材料的异常片段。
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
