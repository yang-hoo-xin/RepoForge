"""Final independent acceptance of a new task derived from saved model code.

为什么要有它：派生起点须逐文件证明来源，并保留两次运行的独立身份及完整模型提交链。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

PREVIOUS = Path(__file__).resolve().parents[1] / "eval_driven_20261022/evaluate_restsharp_v2_candidate.py"
spec = importlib.util.spec_from_file_location("restsharp_original_evaluation", PREVIOUS)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
h, q, require = old.h, old.q, old.require


def verify_derived_baseline(started, original_state, state):
    original = original_state.validate_candidate(started["starting_candidate_ref"])
    require(original["candidate_source_id"] == started["starting_candidate_source_id"], "STARTING_SOURCE_ID_CHANGED")
    expected = original_state.materialize(started["starting_candidate_ref"])
    actual = state.materialize(state.contract["baseline_manifest_ref"])
    require(actual == expected, "DERIVED_BASELINE_TEXT_MISMATCH")
    return {"starting_candidate_ref": started["starting_candidate_ref"],
        "starting_candidate_source_id": original["candidate_source_id"],
        "original_task_contract_ref": original_state.contract_ref,
        "new_task_contract_ref": state.contract_ref,
        "new_baseline_manifest_ref": state.contract["baseline_manifest_ref"],
        "matched_text_files": len(expected), "matched_texts_sha256": old.canonical_hash(expected)}


def validate_lineage(started_path, execution_path, candidate_ref, original_process_receipt):
    # 为什么要有它：先拒绝未封存的新任务，再核验旧候选；运行中的候选不能获得最终独立评分。
    native_ref, native, state, candidate = old.validate_native(execution_path, candidate_ref)
    started_ref = h.ref(started_path)
    started = old.read_bound_json(started_ref)
    require(started.get("new_task") is True and started.get("original_task_resumed") is False,
        "NEW_DERIVED_TASK_REQUIRED")
    config_ref = h.ref(Path(started["allocated_config"]))
    config = old.read_bound_json(config_ref)
    arguments = config["arguments"]

    def arg(flag):
        require(arguments.count(flag) == 1, "UNIQUE_CONFIG_ARGUMENT_REQUIRED")
        return arguments[arguments.index(flag) + 1]

    require(arg("--run-id") == native["run_id"] and arg("--request-id") == native["request_id"],
        "DERIVED_RUN_CONFIG_MISMATCH")
    expected_native = Path(arg("--workspace-root")) / native["run_id"] / "a/task-agent-execution/execution.json"
    require(execution_path.resolve() == expected_native.resolve(), "DERIVED_NATIVE_LOCATION_MISMATCH")
    require(Path(started["allocated_config"]).resolve().is_relative_to(started_path.parent.resolve()),
        "DERIVED_CONFIG_LOCATION_MISMATCH")
    baseline = state.load(state.contract["baseline_manifest_ref"])
    require(Path(baseline["source_root"]).resolve() == Path(arg("--target-project")).resolve(),
        "DERIVED_TARGET_MISMATCH")
    require(set(json.loads(arg("--task-agent-edit-paths"))) == set(state.contract["allowed_edit_paths"]),
        "DERIVED_AUTHORIZATION_MISMATCH")
    original_native = old.read_bound_json(started["original_native_ref"])
    diagnostic = original_native.get("status") == "unknown"
    original_ref, original, original_state, _ = old.validate_native(
        Path(started["original_native_ref"]["path"]), started["starting_candidate_ref"],
        diagnostic=diagnostic, process_receipt=original_process_receipt)
    require(original_ref == started["original_native_ref"] and original["run_id"] != native["run_id"],
        "ORIGINAL_NATIVE_IDENTITY_MISMATCH")
    proof = verify_derived_baseline(started, original_state, state)
    proof.update(started_ref=started_ref, execution_config_ref=config_ref, original_native_ref=original_ref,
        new_native_ref=native_ref, original_unknown_effects=original["unknown_effects"],
        new_task=True, original_task_resumed=False, exposure=started.get("exposure"),
        original_process_evidence=original.get("diagnostic_process_evidence"))
    return native_ref, native, state, candidate, original_state, started, proof


def apply_composed_changes(source, original_state, state, started, candidate_ref):
    verify_derived_baseline(started, original_state, state)
    original_changes = old.apply_changes(source, original_state, started["starting_candidate_ref"])
    # 为什么要有它：第二次沿用同一授权、精确前像和生产文件范围检查，旧候选实际应用后才满足新B0。
    changes = old.apply_changes(source, state, candidate_ref)
    return {"starting_model_changes": original_changes, "new_model_changes": changes}


def evaluate(started_path, execution_path, candidate_ref, output, original_process_receipt):
    native_ref, native, state, candidate, original_state, started, proof = validate_lineage(
        started_path, execution_path, candidate_ref, original_process_receipt)
    qualification = old.verified_qualification()
    require(not output.exists(), "FRESH_OUTPUT_REQUIRED")
    require(subprocess.check_output(["git", "-C", str(h.SOURCE), "rev-parse", "HEAD"], text=True).strip() == h.BASE,
        "BASE_COMMIT_CHANGED")
    output.mkdir(parents=True)
    h.ROOT = output / "evaluation-only"
    h.ROOT.mkdir()
    proof_ref = h.save(h.ROOT / "composed-source-proof.json", proof)
    h.save(output / "evaluation-request.json", {"native_execution_ref": native_ref, "candidate_ref": candidate_ref,
        "candidate_source_id": candidate["candidate_source_id"], "composed_source_proof_ref": proof_ref,
        "qualification_ref": old.QUALIFICATION, "script_ref": h.ref(Path(__file__)),
        "original_validator_ref": h.ref(PREVIOUS), "base_commit": h.BASE})
    source = h.prepare_variant("final", reference=False)
    changes = apply_composed_changes(source, original_state, state, started, candidate_ref)
    q.replace_exact(source / "test/Directory.Build.props", "<TargetFrameworks>net48;net6.0;net7.0;net8.0</TargetFrameworks>", "<TargetFrameworks>net8.0</TargetFrameworks>")
    q.replace_exact(source / "src/Directory.Build.props", "<TargetFrameworks>netstandard2.0;net471;net48;net6.0;net7.0;net8.0</TargetFrameworks>", "<TargetFrameworks>net8.0</TargetFrameworks>")
    h.run(["git", "apply", qualification["public_reproduction_ref"]["path"]], cwd=source, label="final-public-apply")
    h.run(["git", "diff", "--binary"], cwd=source, label="final-diff")
    manifest_ref = h.save(h.ROOT / "candidate-application.json", {"candidate_ref": candidate_ref,
        "candidate_source_id": candidate["candidate_source_id"], **changes})
    try:
        tested = q.test("final", source)
        result = {**old.aggregate_test_result(tested), "private_result_ref": h.save(h.ROOT / "result.json", tested)}
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result = {"status": "execution_incomplete", "tests_executed": None, "failure_type": type(error).__name__,
            "private_failure_ref": h.save(h.ROOT / "failure.json", {"type": type(error).__name__, "reason": str(error)})}
    result.update(schema_version="restsharp-derived-task-independent-result.v1", candidate_ref=candidate_ref,
        candidate_source_id=candidate["candidate_source_id"], native_execution_ref=native_ref, native_status=native["status"],
        composed_source_proof_ref=proof_ref, original_native_ref=proof["original_native_ref"],
        starting_candidate_ref=started["starting_candidate_ref"], starting_candidate_source_id=started["starting_candidate_source_id"],
        new_task=True, original_task_resumed=False, exposure=started.get("exposure"),
        task_contract_ref=state.contract_ref, qualification_ref=old.QUALIFICATION, candidate_application_ref=manifest_ref,
        model_calls=0, delivered=False, scope="New exposed-task final candidate acceptance; details evaluation-only.")
    result_ref = h.save(output / "result.json", result)
    print(json.dumps({"status": result["status"], "candidate_source_id": result["candidate_source_id"], "result_ref": result_ref}), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--started-record", type=Path, required=True)
    parser.add_argument("--execution-record", type=Path, required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--original-process-receipt", type=Path)
    args = parser.parse_args()
    return evaluate(args.started_record.resolve(), args.execution_record.resolve(), args.candidate_ref,
        args.output_root.resolve(), args.original_process_receipt.resolve() if args.original_process_receipt else None)


if __name__ == "__main__":
    raise SystemExit(main())
