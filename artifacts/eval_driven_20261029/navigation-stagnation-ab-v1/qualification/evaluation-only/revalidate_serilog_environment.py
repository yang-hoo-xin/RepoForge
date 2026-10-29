"""Consume the existing evaluator after exact environment restoration.

为什么要有它：只有原冻结 owner 及公共输入实际复核通过后才产生准入包装，保留此前拒绝记录。
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from scripts import run_repository_product_pilot as pilot
from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def bound(reference):
    actual = ref(reference["path"])
    assert Path(actual["path"]).resolve() == Path(reference["path"]).resolve() and actual["sha256"] == reference["sha256"], "BOUND_INPUT_CHANGED"
    return json.loads(Path(reference["path"]).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"controls_rerun": False, "model_calls": 0, "qualified": False}
    try:
        preparation = bound({"path": "C:/tmp/rf-serilog-v2-pair-main/public-preparation.json", "sha256": "82c2e3e9a988abbddff6cc40e85e57a89edb7ceb1cb8be787a72130b291398ee"})
        derived_path = Path(__file__).resolve().parents[1] / "serilog-relocation/evaluator.json"
        original = bound({"path": "C:/tmp/rf-serilog-v2-pair-eval/qualification-final/evaluator.json", "sha256": "4b6f85bd95d7ed30a555f4ffb58f207b618f413a4c941b1cf8788014ac9f3bef"})
        derived_ref = ref(derived_path)
        derived = bound(derived_ref)
        assert {key for key in original if original[key] != derived[key]} == {"adoption_adapter_ref"}
        assert derived["adoption_adapter_ref"]["sha256"] == original["adoption_adapter_ref"]["sha256"]
        wrapper = {"qualified": False, "task_id": preparation["task_id"],
                   "public_source_root": preparation["source_root"],
                   "public_profile_ref": preparation["profile_ref"],
                   "problem_ref": preparation["original_problem_ref"],
                   "evaluator_ref": derived_ref,
                   "platform_ref": {"path": "C:/tmp/rfcode-serilog-v2-pair-base/execution_snapshot.json", "sha256": "4054922b0e2275409a8361008b41db2a8850a8215ae8e679747babdf58fb02ef"}}
        pilot.validate_frozen_evaluator(wrapper)
        report["original_owner_validation"] = "passed"
        profile = CSharpRepositoryProfile(wrapper["public_profile_ref"])
        profile.verify_fixed_inputs(Path(wrapper["public_source_root"]), preparation["fixed_inputs"])
        assert pilot.source_snapshot(Path(wrapper["public_source_root"])) == {row["path"]: row["sha256"] for row in preparation["source_inputs"]}, "PUBLIC_SOURCE_CHANGED"
        assert ref(wrapper["problem_ref"]["path"]) == wrapper["problem_ref"], "PUBLIC_PROBLEM_CHANGED"
        wrapper["qualified"] = True
        target = output / "qualification.json"
        target.write_text(json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report.update(qualified=True, qualification_ref=ref(target))
    except Exception as exc:
        report.update(failure_type=type(exc).__name__, failure_message=str(exc))
    report["wall_seconds"] = round(time.monotonic() - started, 3)
    target = output / "result.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
