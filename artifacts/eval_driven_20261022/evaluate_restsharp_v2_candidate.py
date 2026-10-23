"""Final-only independent RestSharp acceptance; evaluation-only, never solver input.

为什么要有它：只评已封存终态候选，私有验收结果不进入进行中的技术决策。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.mod_agent.model_qualification import canonical_hash  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json as _read_bound_json  # noqa: E402
from src.mod_agent.task_agent_execution import _admitted_response, _read  # noqa: E402
from src.mod_agent.task_agent_v2_state import TaskAgentV2State  # noqa: E402

spec = importlib.util.spec_from_file_location("restsharp_qualification", Path(__file__).with_name("qualify_restsharp_2241.py"))
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
h = q.h
QUALIFICATION = {"path": "C:/tmp/rfext-restsharp-2241-public-qualification-v2/qualification.json",
    "sha256": "28ff3f2c1802ed8c8a0e32bb9a102b001ca545c3002f40b74e9806bd70630d07"}
TEST_PATCH_SHA256 = "f16d785a0ea2250185b3454b9554a2c7a966a3bcf2026bd20394ec66ecf197ee"
TASK_SHA256 = "2e09e3d4393fb3337b51bda45eba889a923eb52bbc6e2b5ef3ce4887ce95e74b"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read_bound_json(reference):
    # 为什么要有它：资格原件的bytes是附加计量，产品读取器仍仅接收精确path/hash身份。
    require(set(reference) in ({"path", "sha256"}, {"path", "sha256", "bytes"}), "EXACT_FILE_REFERENCE_REQUIRED")
    if "bytes" in reference:
        require(Path(reference["path"]).stat().st_size == reference["bytes"], "REFERENCE_SIZE_CHANGED")
    return _read_bound_json({key: reference[key] for key in ("path", "sha256")})


def verified_qualification():
    qualification = read_bound_json(QUALIFICATION)
    original = read_bound_json(qualification["independent_qualification_ref"])
    require(qualification["qualified"] is True and original["qualified"] is True, "QUALIFICATION_NOT_PASSED")
    read_bound_json({"path": str(h.TASK_ROOT / "evaluation-only/task.json"), "sha256": TASK_SHA256})
    require(h.sha(h.TASK_ROOT / "evaluation-only/test.patch") == TEST_PATCH_SHA256, "TEST_PATCH_CHANGED")
    require(h.sha(Path(qualification["public_reproduction_ref"]["path"])) == qualification["public_reproduction_ref"]["sha256"], "PUBLIC_TEST_CHANGED")
    # 为什么要有它：资格摘要不能脱离当时实际执行过的报告而独立生效。
    for collection in (original["variants"], qualification["public_results"]):
        for result in collection.values():
            for reference in result["trx_refs"]:
                require(h.sha(Path(reference["path"])) == reference["sha256"], "QUALIFICATION_TRX_CHANGED")
            for process in result["processes"]:
                raw = read_bound_json(process["process_ref"])
                require(raw["exit_code"] == process["exit_code"], "QUALIFICATION_EXIT_CHANGED")
                for key in ("stdout_ref", "stderr_ref"):
                    require(h.sha(Path(raw[key]["path"])) == raw[key]["sha256"], "QUALIFICATION_LOG_CHANGED")
    return qualification


def validate_diagnostic_process(native_ref, native, process_path):
    require(process_path is not None, "DIAGNOSTIC_PROCESS_RECEIPT_REQUIRED")
    process_ref = h.ref(process_path)
    process = read_bound_json(process_ref)
    require(process.get("status") == "exited" and process.get("started") is True and process.get("timeout") is False
        and isinstance(process.get("exit_code"), int) and process.get("finished_at")
        and process.get("process_tree_ownership") == "Windows Job Object"
        and process.get("owned_process_tree_cleanup") == "job_closed_kill_on_close", "DIAGNOSTIC_PROCESS_NOT_CLOSED")
    for name in ("stdout_ref", "stderr_ref"):
        require(h.sha(Path(process[name]["path"])) == process[name]["sha256"], "DIAGNOSTIC_PROCESS_LOG_CHANGED")
    summary = json.loads(Path(process["stdout_ref"]["path"]).read_bytes())
    require(summary.get("task_agent_execution_ref") == native_ref and summary.get("run_id") == native["run_id"], "DIAGNOSTIC_PROCESS_NATIVE_MISMATCH")
    start_ref = h.ref(process_path.parent.parent / "start.json")
    result_ref = h.ref(process_path.parent.parent / "result.json")
    start, result = read_bound_json(start_ref), read_bound_json(result_ref)
    require(result.get("process") == process and result.get("original_started_ref") == start["original_started_ref"]
        and result.get("new_task_allocated") is False and start.get("new_task_allocated") is False
        and start["run_id"] == native["run_id"], "DIAGNOSTIC_PROCESS_ORIGIN_MISMATCH")
    original = read_bound_json(start["original_started_ref"])
    config = read_bound_json(start["config_ref"])
    args = config["arguments"]
    require(args[args.index("--run-id") + 1] == native["run_id"]
        and args[args.index("--request-id") + 1] == native["request_id"]
        and Path(original["allocated_config"]).resolve() == Path(start["config_ref"]["path"]).resolve()
        and process["argv"][-2:] == ["--execution-config", start["config_ref"]["path"]], "DIAGNOSTIC_CONFIG_MISMATCH")
    snapshot = read_bound_json(original["code_snapshot_ref"])
    code_root = Path(original["code_snapshot_ref"]["path"]).parent
    require(Path(process["cwd"]).resolve() == code_root.resolve()
        and Path(process["argv"][2]).resolve() == (code_root / "scripts/dsh_run_request.py").resolve(), "DIAGNOSTIC_CODE_ROOT_MISMATCH")
    for row in snapshot["files"]:
        path = code_root / row["path"]
        require(path.resolve().is_relative_to(code_root.resolve()) and h.sha(path) == row["sha256"], "DIAGNOSTIC_CODE_SNAPSHOT_CHANGED")
    turns = native["provider_turns"]
    require(bool(turns) and all(turn.get("status") == "responded" for turn in turns), "DIAGNOSTIC_PROVIDER_INFLIGHT")
    for turn in turns:
        response = read_bound_json(turn["response_ref"])
        _admitted_response(response, native["model_profile"])
        dispatch = read_bound_json(turn["dispatch_ref"])
        require(dispatch["reservation_id"] == turn["reservation_id"] and dispatch["request_ref"] == turn["request_ref"], "DIAGNOSTIC_PROVIDER_DISPATCH_CHANGED")
        read_bound_json(turn["request_ref"])
        outcome_path = Path(native_ref["path"]).parent / ("provider-outcome-" + turn["reservation_id"] + ".json")
        require(read_bound_json(h.ref(outcome_path)) == turn, "DIAGNOSTIC_PROVIDER_OUTCOME_CHANGED")
    return {"process_receipt_ref": process_ref, "service_start_ref": start_ref, "service_result_ref": result_ref,
        "code_snapshot_ref": original["code_snapshot_ref"], "execution_config_ref": start["config_ref"],
        "owned_process_tree_cleanup": process["owned_process_tree_cleanup"],
        "broker_descendant_cleanup": process.get("broker_descendant_cleanup"), "provider_responses": len(turns)}


def validate_native(execution_path, candidate_ref, *, diagnostic=False, process_receipt=None):
    native_ref = {key: value for key, value in h.ref(execution_path).items() if key != "bytes"}
    native = read_bound_json(native_ref)
    require(native.get("schema_version") == "task-agent-product-execution.v2", "NATIVE_V2_REQUIRED")
    if diagnostic:
        # 为什么要有它：独立诊断不消除原检查的未知状态，必须另有进程退出及实际响应原件。
        require(native.get("status") == "unknown" and native.get("unknown_effects") is True
            and native.get("termination", {}).get("code") == "TASK_AGENT_ACTION_OUTCOME_UNKNOWN", "DIAGNOSTIC_UNKNOWN_CHECK_REQUIRED")
        native["diagnostic_process_evidence"] = validate_diagnostic_process(native_ref, native, process_receipt)
    else:
        require(native.get("status") in {"candidate_ready", "blocked", "failed"} and native.get("unknown_effects") is False,
            "KNOWN_TERMINAL_EXECUTION_REQUIRED")
    require(native.get("candidate_ref") == candidate_ref, "FINAL_CANDIDATE_MISMATCH")
    require(native.get("source_before") == native.get("source_after") and bool(native.get("source_before")), "NATIVE_SOURCE_CHANGED")
    roots = [parent for parent in execution_path.resolve().parents if (parent / native["task_contract_ref"]).is_file()]
    require(len(roots) == 1, "UNIQUE_NATIVE_STORE_REQUIRED")
    root = roots[0]
    state = TaskAgentV2State(root, native["task_contract_ref"])
    candidate = state.validate_candidate(candidate_ref)
    require(native.get("initial_context") == native.get("final_context")
        and native["initial_context"].get("feature_spec_ref") == state.contract.get("feature_spec_ref"), "NATIVE_REQUIREMENTS_CHANGED")
    loop = _read(native["loop_result_ref"], root)
    require(loop.get("unknown_effects") is diagnostic and loop.get("termination") == native.get("termination"), "NATIVE_LOOP_NOT_TERMINAL")
    if diagnostic:
        unknown = [(key, value) for key, value in loop["call_records"].items() if value.get("status") == "unknown"]
        require(len(unknown) == 1 and unknown[0][0] == native["termination"]["detail"]["call_id"]
            and unknown[0][1]["name"] == "candidate_check", "DIAGNOSTIC_UNKNOWN_MUTATION_FORBIDDEN")
    records = native["completed_tool_records"]
    selected = candidate_ref
    visited = set()
    while selected != state.contract["baseline_manifest_ref"]:
        require(selected not in visited, "CANDIDATE_CHAIN_CYCLE")
        visited.add(selected)
        document = state.validate_candidate(selected)
        origin = document["model_origin"]
        matches = [row for row in records if row["call_id"] == origin["call_id"] and row["name"] == "candidate_submit"]
        require(len(matches) == 1, "REAL_SUBMISSION_REQUIRED")
        row = matches[0]
        require(origin == {key: row[key] for key in ("call_id", "response_ref", "arguments_ref")}, "SUBMISSION_ORIGIN_CHANGED")
        arguments, response, output = (_read(row[key], root) for key in ("arguments_ref", "response_ref", "result_ref"))
        _admitted_response(response, native["model_profile"])
        require([call for call in response["tool_calls"] if call["call_id"] == row["call_id"]]
            == [{"name": "candidate_submit", "call_id": row["call_id"], "arguments": arguments}], "MODEL_SUBMISSION_CHANGED")
        actual = loop["call_records"][row["call_id"]]
        require(actual["status"] == "completed" and actual["name"] == "candidate_submit"
            and actual["arguments_sha256"] == canonical_hash(arguments) and actual["result"]["output"] == output
            and row["event"] in loop["events"] and row["event"]["output_sha256"] == canonical_hash(output)
            and output.get("execution") == "executed" and output["value"]["candidate_ref"] == selected, "REAL_SUBMISSION_EXECUTION_REQUIRED")
        require(arguments["parent_revision_ref"] == document["parent_revision_ref"], "MODEL_PARENT_CHANGED")
        keys = ("binding_id", "source_view_ref", "before_text", "after_text")
        require([{key: item.get(key) for key in keys} for item in arguments["edits"]]
            == [{key: item.get(key) for key in keys} for item in document["changes"]], "MODEL_EDITS_CHANGED")
        selected = document["parent_revision_ref"]
    return native_ref, native, state, candidate


def apply_changes(source, state, candidate_ref):
    baseline = state.load(state.contract["baseline_manifest_ref"])["files"]
    # 为什么要有它：独立验收只能把同一基线上的许可生产代码带入，不能把模型改动的检查器当真值。
    tracked = subprocess.check_output(["git", "ls-files", "-z", "src", "gen"], cwd=source).decode().split("\0")
    for name in (name for name in tracked if name.endswith(".cs")):
        require(name in baseline and (source / name).read_bytes().decode("utf-8") == baseline[name]["text"], "BASELINE_SOURCE_IDENTITY_MISMATCH")
    changes = state.changes_from_baseline(candidate_ref)
    require(bool(changes), "NONEMPTY_FINAL_DIFF_REQUIRED")
    for change in changes:
        name = change["path"]
        relative = PurePosixPath(name)
        require(name in state.contract["allowed_edit_paths"] and name.startswith("src/RestSharp/") and name.endswith(".cs")
            and ".." not in relative.parts and not relative.is_absolute() and "\\" not in name and ":" not in name, "OUTSIDE_QUALIFIED_EDIT_SCOPE")
        target = source / name
        require(target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(source.resolve()), "INVALID_CANDIDATE_TARGET")
        require(target.read_bytes().decode("utf-8") == change["before_text"], "CANDIDATE_PREIMAGE_MISMATCH")
    for change in changes:
        (source / change["path"]).write_bytes(change["after_text"].encode("utf-8"))
    return [{key: item[key] for key in ("path", "before_sha256", "after_sha256", "binding_id")} for item in changes]


def aggregate_test_result(tested):
    independent_pass = q.accepted(tested)
    public_pass = tested["public_reproduction"] == {"count": 2, "passed": 2}
    full_pass = tested["all_outcomes"]["Failed"] == 0 and all(row["exit_code"] == 0 for row in tested["processes"])
    retained = tested["pass_to_pass"]
    preserved = (retained["missing_count"] == 0 and retained["outcome_counts"]["Passed"] > 0
        and retained["outcome_counts"]["Failed"] == 0 and retained["outcome_counts"]["Other"] == 0)
    # 为什么要有它：目标测试失败不能误报为原有行为退化，全部检查通过与保留行为通过分别统计。
    return {"status": "passed" if independent_pass and public_pass and full_pass else "failed", "tests_executed": True,
        "independent_acceptance_passed": independent_pass, "public_reproduction_passed": public_pass,
        "preservation_suite_passed": preserved, "preservation_scope": "qualified_existing_behavior_group",
        "full_selected_suite_passed": full_pass, "total_results": tested["total_results"], "outcomes": tested["all_outcomes"]}


def correct_saved_aggregate(path):
    original_ref = h.ref(path)
    original = read_bound_json(original_ref)
    tested = read_bound_json(original["private_result_ref"])
    updated = {**original, **aggregate_test_result(tested), "supersedes_aggregate_ref": original_ref,
        "reporting_correction": "Separate existing-behavior preservation from the entire suite including defect tests; no reexecution or task-state change.",
        "aggregation_script_ref": h.ref(Path(__file__))}
    return h.save(path.with_name("result-reporting-v2.json"), updated)


def evaluate(execution_path, candidate_ref, output, *, diagnostic=False, process_receipt=None):
    qualification = verified_qualification()
    native_ref, native, state, candidate = validate_native(execution_path, candidate_ref, diagnostic=diagnostic, process_receipt=process_receipt)
    require(not output.exists(), "FRESH_OUTPUT_REQUIRED")
    require(subprocess.check_output(["git", "-C", str(h.SOURCE), "rev-parse", "HEAD"], text=True).strip() == h.BASE, "BASE_COMMIT_CHANGED")
    output.mkdir(parents=True)
    h.ROOT = output / "evaluation-only"
    h.ROOT.mkdir()
    h.save(output / "evaluation-request.json", {"native_execution_ref": native_ref, "candidate_ref": candidate_ref,
        "candidate_source_id": candidate["candidate_source_id"], "task_contract_ref": state.contract_ref,
        "qualification_ref": QUALIFICATION, "script_ref": h.ref(Path(__file__)), "base_commit": h.BASE})
    # 为什么要有它：只应用测试补丁和固定环境，不引入参考修复或调用者提供的汇总答案。
    source = h.prepare_variant("final", reference=False)
    changes = apply_changes(source, state, candidate_ref)
    q.replace_exact(source / "test/Directory.Build.props", "<TargetFrameworks>net48;net6.0;net7.0;net8.0</TargetFrameworks>", "<TargetFrameworks>net8.0</TargetFrameworks>")
    q.replace_exact(source / "src/Directory.Build.props", "<TargetFrameworks>netstandard2.0;net471;net48;net6.0;net7.0;net8.0</TargetFrameworks>", "<TargetFrameworks>net8.0</TargetFrameworks>")
    h.run(["git", "apply", qualification["public_reproduction_ref"]["path"]], cwd=source, label="final-public-apply")
    h.run(["git", "diff", "--binary"], cwd=source, label="final-diff")
    manifest_ref = h.save(h.ROOT / "candidate-application.json", {"candidate_ref": candidate_ref, "candidate_source_id": candidate["candidate_source_id"], "changes": changes})
    try:
        tested = q.test("final", source)
        private_result_ref = h.save(h.ROOT / "result.json", tested)
        result = {**aggregate_test_result(tested), "private_result_ref": private_result_ref}
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        result = {"status": "execution_incomplete", "tests_executed": None, "failure_type": type(error).__name__,
            "private_failure_ref": h.save(h.ROOT / "failure.json", {"type": type(error).__name__, "reason": str(error)})}
    result.update(schema_version="restsharp-2241-independent-candidate-result.v1", candidate_ref=candidate_ref,
        candidate_source_id=candidate["candidate_source_id"], native_execution_ref=native_ref, native_status=native["status"],
        task_contract_ref=state.contract_ref, qualification_ref=QUALIFICATION, candidate_application_ref=manifest_ref,
        public_reproduction_ref=qualification["public_reproduction_ref"], model_calls=0, delivered=False,
        scope="Final candidate independent evaluation only; not product delivery or model feedback.")
    result.update(diagnostic_only=diagnostic, original_task_incomplete=diagnostic,
        original_unknown_effects=native["unknown_effects"], diagnostic_process_evidence=native.get("diagnostic_process_evidence"))
    result_ref = h.save(output / "result.json", result)
    print(json.dumps({"status": result["status"], "candidate_source_id": result["candidate_source_id"], "result_ref": result_ref}), flush=True)
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-record", type=Path, required=True)
    parser.add_argument("--candidate-ref", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--diagnostic-final-check", action="store_true")
    parser.add_argument("--process-receipt", type=Path)
    args = parser.parse_args()
    if bool(args.process_receipt) != args.diagnostic_final_check:
        parser.error("--process-receipt and --diagnostic-final-check must be used together")
    return evaluate(args.execution_record.resolve(), args.candidate_ref, args.output_root.resolve(),
        diagnostic=args.diagnostic_final_check, process_receipt=args.process_receipt.resolve() if args.process_receipt else None)


if __name__ == "__main__":
    raise SystemExit(main())
