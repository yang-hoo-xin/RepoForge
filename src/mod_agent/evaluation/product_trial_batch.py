"""Serial frozen Product trials with caller-owned delivery and explicit outcome scope."""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .task_measurement import measure_product_task
from .trial_comparison_identity import platform_identity, resource_budget_identity


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _path(value) -> Path:
    if not isinstance(value, str) or value.startswith(("\\\\", "//")):
        raise ValueError("Trial paths must be explicitly selected local paths.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Trial paths must be absolute.")
    return path.resolve()


def _read_reference(reference, *, root=None):
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Expected an exact file reference.")
    path = _path(reference["path"])
    if root is not None and not path.is_relative_to(root):
        raise ValueError("Delivery reference escapes this trial.")
    raw = path.read_bytes()
    if _hash(raw) != reference["sha256"]:
        raise ValueError("Selected reference bytes changed.")
    return raw


def prepare_trial(row):
    """Return the existing launcher invocation without importing delivery code."""
    for key in ("task_id", "arm", "run_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError("Missing trial identity: " + key)
    if row["arm"] not in {"baseline", "candidate"} or type(row.get("repeat")) is not int or row["repeat"] < 1:
        raise ValueError("Invalid arm or repeat identity.")
    run_root, allowed = _path(row["run_root"]), _path(row["allowed_root"])
    snapshot = _path(row["snapshot_root"])
    launcher = _path(row["launcher"])
    if run_root.name != row["run_id"] or not run_root.is_relative_to(allowed) or run_root == allowed:
        raise ValueError("Trial run root does not match its allowed root and run ID.")
    if run_root.is_relative_to(snapshot) or snapshot.is_relative_to(run_root):
        raise ValueError("Execution snapshot and task run must be separate.")
    if launcher.name != "run_frozen_product.py" or not launcher.is_file():
        raise ValueError("Use the existing frozen Product launcher.")
    argv = row.get("business_argv")
    if not isinstance(argv, list) or not argv or argv[0] != "run-request" or any(type(arg) is not str for arg in argv):
        raise ValueError("Pass the caller's original run-request argv.")
    for flag, expected in (("--run-id", row["run_id"]), ("--workspace-root", str(run_root.parent))):
        if argv.count(flag) != 1:
            raise ValueError("Expected one explicit " + flag)
        value = argv[argv.index(flag) + 1]
        if (_path(value) != run_root.parent if flag == "--workspace-root" else value != expected):
            raise ValueError("Business argv disagrees with the selected run.")
    command = [sys.executable, "-B", str(launcher), "--snapshot-root", str(snapshot), "--run-root", str(run_root)]
    for field, flag in (("bound_inputs", "--bound-input"), ("extra_load_refs", "--extra-load-ref")):
        for reference in row.get(field, []):
            _read_reference(reference)
            command.extend([flag, reference["path"], reference["sha256"]])
    command.extend(["--", *argv])
    deadline = row.get("process_deadline_seconds")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0):
        raise ValueError("Process deadline must be positive finite seconds.")
    return {"task_id": row["task_id"], "arm": row["arm"], "repeat": row["repeat"],
        "run_id": row["run_id"], "run_root": str(run_root), "snapshot_root": str(snapshot),
        "command": command, "process_deadline_seconds": deadline,
        "status": "prepared", "success": None, "metrics": None}


def _run_command(command, *, timeout=None, **_kwargs):
    """Bound the owned launcher tree, including its fresh Product child process."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
            start_new_session=os.name != "nt", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=30)
        stdout.seek(0)
        stderr.seek(0)
        result = subprocess.CompletedProcess(command, code, stdout.read().decode("utf-8", errors="replace"),
            stderr.read().decode("utf-8", errors="replace"))
        result.deadline_exceeded = timed_out
        return result


def _comparison_identity(manifest, result, judge_ref):
    model = manifest.get("model_admission") or {}
    budget = result.get("model_call_budget") or manifest.get("model_call_budget") or {}
    required = (manifest.get("request"), manifest.get("source_snapshot_sha256"), model.get("model"),
        model.get("model_digest"), manifest.get("ollama_patch_generation_profile"), budget.get("policy"))
    if any(value is None for value in required):
        return None
    if "maximum" not in budget or budget["maximum"] is None and budget.get("policy") != "meter_only":
        return None
    return {"request": manifest["request"], "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "model": model.get("model"), "model_digest": model.get("model_digest"),
        "provider": model.get("provider"), "execution_locality": model.get("execution_locality"),
        "model_profile": manifest.get("ollama_patch_generation_profile"),
        "budget_maximum": budget.get("maximum"), "budget_policy": budget.get("policy"),
        "source_disclosure": manifest.get("source_disclosure"), "runtime_admission": manifest.get("runtime_admission"),
        "judge_sha256": judge_ref["sha256"]}


def verify_product_model_proposal(run_root, initial_result, measurement, verification_ref):
    """Reuse immutable artifacts and native materialization to exclude developer patches."""
    from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
    from src.mod_agent.model_qualification import _verify_materialized_response

    run_root = _path(str(run_root))
    report = json.loads(_read_reference(verification_ref, root=run_root))
    if report.get("status") != "passed" or report.get("execution_origin") != "executed":
        raise ValueError("Delivery must reference an executed, passed judge report.")
    store = ArtifactStore(run_root / "a")  # The existing normal run-request artifact location.
    documents, references = {}, {}
    for kind in ("feature_spec", "evidence_pack", "implementation_plan", "patch_proposal"):
        path = _path(initial_result["artifacts"][kind])
        if not path.is_relative_to(store.root.resolve()):
            raise ValueError("Product artifact reference is outside this run's store.")
        envelope = json.loads(path.read_bytes())
        reference = ArtifactRef(feature_id=initial_result["feature_id"], artifact_type=kind,
            path=str(path), artifact_hash=envelope["artifact_hash"])
        documents[kind] = store.semantic_read_immutable(reference)["data"]
        references[kind] = {"feature_id": reference.feature_id, "artifact_type": kind,
            "path": reference.path, "artifact_hash": reference.artifact_hash}
    if report.get("implementation_plan_ref") != references["implementation_plan"]:
        raise ValueError("Judge report is not bound to the original selected Plan.")
    response_hash = report.get("model_response_sha256")
    expected = {row.get("generation", {}).get("model_response_sha256") for row in documents["patch_proposal"]["diff_intent"]}
    if not isinstance(response_hash, str) or expected != {response_hash}:
        raise ValueError("Judge report and Product proposal refer to different model outputs.")
    for call in measurement["model"].get("calls", []):
        if call.get("transport", {}).get("response_origin") != "live_provider_response":
            continue
        reference = call.get("local_evidence", {}).get("response", {})
        filename = reference.get("file")
        if reference.get("integrity") != "verified_bytes" or not isinstance(filename, str) or Path(filename).name != filename:
            continue
        files = list(run_root.rglob(filename))
        if len(files) != 1:
            continue
        raw = _read_reference({"path": str(files[0]), "sha256": reference["sha256"]}, root=run_root)
        response = json.loads(raw)
        content = response.get("content")
        if not isinstance(content, str) or _hash(content.encode("utf-8")) != response_hash:
            continue
        _verify_materialized_response(feature=documents["feature_spec"], evidence=documents["evidence_pack"],
            plan=documents["implementation_plan"], proposal=documents["patch_proposal"],
            parsed_output=json.loads(content), expected_changes_hash=report["generated_changes_sha256"], workspace_root=None)
        return {"model_response_sha256": response_hash, "proposal_ref": references["patch_proposal"],
            "verification_report_ref": verification_ref, "verification": "retained_live_native_response_matches_immutable_proposal"}
    raise ValueError("No retained live Product response supports this proposal.")


def _verify_v2_live_submission(run_root, native, delivered, measurement):
    from src.mod_agent.model_qualification import canonical_hash
    from src.mod_agent.task_agent_v2_state import load_v2_artifact

    candidate = load_v2_artifact(run_root / "a", delivered["candidate_ref"])
    origin = candidate["model_origin"]
    response = json.loads(_read_reference(origin["response_ref"], root=run_root))
    calls = [row for row in response.get("tool_calls", [])
        if row.get("call_id") == origin["call_id"] and row.get("name") == "candidate_submit"]
    turns = [row for row in native.get("provider_turns", [])
        if row.get("response_ref", {}).get("sha256") == origin["response_ref"]["sha256"]]
    if len(calls) != 1 or len(turns) != 1:
        raise ValueError("V2 candidate has no unique native submit response.")
    turn = turns[0]
    for key in ("request_ref", "provider_request_ref", "response_ref"):
        _read_reference(turn[key], root=run_root)
    # 为什么要有它：整个运行有真实响应不代表最终补丁来自它，须将具体提交与同一请求、传输和响应绑定。
    matches = [row for row in measurement["model"].get("calls", [])
        if row.get("transport", {}).get("response_origin") == "live_provider_response"
        and row["transport"].get("response_received") is True and row["transport"].get("response_complete") is True
        and row.get("response_sha256") == canonical_hash(response)
        and all(row.get("local_evidence", {}).get(role, {}).get("integrity") == "verified_bytes"
            and row["local_evidence"][role]["sha256"] == turn[key]["sha256"]
            for role, key in (("request", "request_ref"), ("provider_request", "provider_request_ref")))]
    if len(matches) != 1:
        raise ValueError("V2 final submit is not bound to one retained live model response.")
    return {"response_ref": origin["response_ref"], "call_id": origin["call_id"],
        "request_ref": turn["request_ref"], "generation_span_id": matches[0]["generation_span_id"],
        "trace_id": matches[0]["trace_id"]}


def _verify_v2_delivery(adapter_result, run_root, run_id, initial_result, judge_ref, measurement):
    """Use native v2 delivery and qualified independent verdicts, never a fake Plan.

    为什么要有它：v2候选身份和正式交付回执须经原owner验证，旧Plan检查不能替代也不能阻断新协议。
    """
    from src.mod_agent.evaluation.strategy_adoption import _evaluator_evidence
    from src.mod_agent.evaluation.strategy_native_export import validate_native_strategy_execution
    from src.mod_agent.task_agent_v2_delivery import read_completed_v2_delivery

    native_ref = initial_result.get("task_agent_execution_ref") or initial_result.get("task_agent_outcome", {}).get("execution_ref")
    native = json.loads(_read_reference(native_ref, root=run_root))
    if native.get("schema_version") != "task-agent-product-execution.v2" or native.get("run_id") != run_id:
        raise ValueError("V2 trial native execution identity mismatch.")
    receipt = adapter_result.get("v2_delivery_receipt_ref")
    if not isinstance(receipt, str):
        raise ValueError("V2 trial requires the original completed delivery receipt.")
    delivered = read_completed_v2_delivery(artifact_store_root=run_root / "a", receipt_ref=receipt,
        native_execution_ref=native_ref)
    live_submission = _verify_v2_live_submission(run_root, native, delivered, measurement)
    verdict_ref = adapter_result.get("verification_report_ref")
    verdict = json.loads(_read_reference(verdict_ref, root=run_root))
    if (verdict.get("schema_version") != "independent_strategy_verdict_v1"
            or verdict.get("independent") is not True or verdict.get("batch_complete") is not True
            or verdict.get("evaluator_ref") != judge_ref or verdict.get("task_success") is not True
            or verdict.get("preservation_passed") is not True
            or verdict.get("candidate_source_id") != delivered["candidate_source_id"]):
        raise ValueError("V2 trial independent verdict is not bound to its delivered candidate and selected judge.")
    execution = json.loads(_read_reference(verdict["execution_ref"], root=run_root))
    if (execution.get("native_execution_ref") != native_ref
            or execution.get("candidate_source_id") != delivered["candidate_source_id"]
            or execution.get("candidate_state", {}).get("candidate_ref") != delivered["candidate_ref"]):
        raise ValueError("V2 trial verdict refers to a different native execution.")
    native_evidence = validate_native_strategy_execution(verdict["execution_ref"])
    if native_evidence.get("status") != "native_execution_verified":
        raise ValueError("V2 trial native export is not verified.")
    verified = _evaluator_evidence(verdict)
    if verified.get("status") not in {"verified_repository_csharp", "verified_restricted_csharp"}:
        raise ValueError("V2 trial independent evaluator originals are not verified.")
    return {"verification": "native_v2_delivery_and_qualified_independent_verdict",
        "native_execution_ref": native_ref, "delivery_receipt_ref": receipt,
        "candidate_source_id": delivered["candidate_source_id"], "candidate_ref": delivered["candidate_ref"],
        "verification_report_ref": verdict_ref, "evaluator_evidence": verified,
        "live_submission": live_submission, "native_evidence": native_evidence}


def _delivery(adapter_result, run_root, run_id, measurement, initial_result, judge_ref=None):
    """Trust the selected delivery adapter's judgment only with retained local evidence.

    为什么要有它：开发者修复或仅生成候选都不等于产品交付；adapter负责调用公共续接与共同judge，汇总不替代它们。
    """
    if not isinstance(adapter_result, dict):
        raise ValueError("Delivery adapter must return an object.")
    status = adapter_result.get("status")
    if status not in {"delivered", "failed", "infrastructure_failure", "unknown"}:
        raise ValueError("Unknown delivery adapter status.")
    refs = adapter_result.get("evidence_refs", [])
    for reference in refs:
        _read_reference(reference, root=run_root)
    public_records = [json.loads(_read_reference(ref, root=run_root)) for ref in adapter_result.get("execution_refs", [])]
    known_zero_models = bool(public_records) and all(record.get("run_id") == run_id
        and record.get("model_calls_performed") == 0 for record in public_records)
    observed = {(row["trace_id"], row["observation_id"]) for row in measurement["tools"]["observations"]}
    tool_count, tools_known = 0, known_zero_models
    for record in public_records:
        native = record.get("execution_observations")
        trace_id = (record.get("execution_trace") or {}).get("trace_id")
        if not isinstance(native, dict) or not trace_id:
            tools_known = False
            continue
        for key, observation in native.items():
            if observation.get("id") != key:
                raise ValueError("Delivery observation identity mismatch.")
            if observation.get("kind") == "generation":
                known_zero_models = False
            if observation.get("kind") == "tool" and (trace_id, key) not in observed:
                tool_count += 1
                observed.add((trace_id, key))
    live_responses = measurement["model"].get("response_origins", {}).get("live_provider_response", 0)
    success = (status == "delivered" and adapter_result.get("source_origin") == "product_model_proposal"
        and adapter_result.get("delivery_complete") is True and adapter_result.get("acceptance_passed") is True
        and bool(refs) and bool(public_records) and live_responses > 0)
    if status == "delivered" and not success:
        status = "unknown"
    is_v2 = (initial_result.get("protocol") == "v2" or initial_result.get("technical_execution", {}).get("protocol") == "v2"
        or initial_result.get("status") == "awaiting_delivery")
    proposal_origin = None
    if success:
        proposal_origin = (_verify_v2_delivery(adapter_result, run_root, run_id, initial_result, judge_ref, measurement) if is_v2 else
            verify_product_model_proposal(run_root, initial_result, measurement, adapter_result.get("verification_report_ref")))
    return {"status": status, "success": success if status != "unknown" else None,
        "judgment_scope": "caller_delivery_adapter_and_bound_public_execution_records",
        "evidence_refs": refs, "execution_refs": adapter_result.get("execution_refs", []),
        "model_cost_known_zero": known_zero_models,
        "proposal_origin": proposal_origin,
        "additional_observed_tools": tool_count if tools_known else None,
        "reason": adapter_result.get("reason")}


def run_product_trial_batch(spec, *, execute=False, delivery_adapter=None, command_runner=None, checkpoint=None):
    """Run selected trials serially; every selected row survives failures and unknowns."""
    trials = spec.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("Select at least one trial.")
    judge_ref = spec["judge_ref"]
    _read_reference(judge_ref)
    identities, roots = set(), set()
    planned = []
    for trial in trials:
        row = prepare_trial(trial)
        identity = (row["task_id"], row["arm"], row["repeat"])
        if identity in identities or row["run_root"] in roots or row["snapshot_root"] in roots:
            raise ValueError("Duplicate trial or output identity.")
        identities.add(identity)
        roots.update((row["run_root"], row["snapshot_root"]))
        planned.append(row)
    if not execute:
        return {"executed": False, "trials": planned, "summary": summarize_trials(planned)}
    if not callable(delivery_adapter):
        raise ValueError("Execution requires the caller's public delivery adapter.")
    runner = command_runner or _run_command
    report = {"executed": True, "judge_ref": judge_ref, "trials": planned}
    for row in report["trials"]:
        row["status"] = "pending"
    for spec_row, row in zip(trials, report["trials"], strict=True):
        row["status"] = "running"
        if checkpoint:
            checkpoint(deepcopy(report))
        started = time.perf_counter()
        completed = None
        try:
            run_root, snapshot_root = _path(row["run_root"]), _path(row["snapshot_root"])
            if run_root.exists() or snapshot_root.exists():
                raise ValueError("Fresh trials cannot replace an existing run or snapshot.")
            _read_reference(judge_ref)
            try:
                completed = runner(row["command"], check=False, capture_output=True, text=True, encoding="utf-8",
                    timeout=row["process_deadline_seconds"])
            finally:
                row["normal_entry_seconds"] = time.perf_counter() - started
            row["launcher_exit_code"] = completed.returncode
            row["process_deadline_exceeded"] = getattr(completed, "deadline_exceeded", False)
            if completed.returncode or not (run_root / "run_result.json").is_file():
                # 为什么要有它：CLI 的结构化失败写入 stdout；缺少 terminal 不能覆盖原始失败。
                row["launcher_stdout"] = getattr(completed, "stdout", None)
                row["launcher_stderr"] = getattr(completed, "stderr", None)
            snapshot_path = snapshot_root / "execution_snapshot.json"
            if snapshot_path.is_file():
                snapshot_bytes = snapshot_path.read_bytes()
                snapshot_document = json.loads(snapshot_bytes)
                child = json.loads((snapshot_root / "child_execution.json").read_text(encoding="utf-8"))
                row["platform_execution"] = {"snapshot_ref": {"path": str(snapshot_path), "sha256": _hash(snapshot_bytes)},
                    **platform_identity(snapshot_document),
                    "status": child.get("status")}
            measurement = measure_product_task(allowed_root=spec_row["allowed_root"], run_root=run_root, run_id=row["run_id"])
            manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
            result = json.loads((run_root / "run_result.json").read_text(encoding="utf-8"))
            row["measurement_refs"] = [{key: segment.get(key) for key in ("manifest_ref", "result_ref", "trace")}
                for segment in measurement.get("segments", [])]
            row["observed_response_origins"] = measurement["model"].get("response_origins", {})
            row["comparison_identity"] = _comparison_identity(manifest, result, judge_ref)
            row["resource_budget"] = resource_budget_identity(snapshot_document, snapshot_root, manifest, result,
                row["process_deadline_seconds"]) if snapshot_path.is_file() else {"complete": False}
            if not row["resource_budget"]["complete"]:
                row["comparison_identity"] = None
            elif row["comparison_identity"] is not None:
                row["comparison_identity"]["effective_resource_limits"] = row["resource_budget"]["effective_limits"]
            if row["comparison_identity"] is not None:
                row["comparison_identity"]["requested_operation_flags"] = sorted(flag for flag in spec_row["business_argv"]
                    if flag in {"--confirm-isolated-write", "--allow-build", "--allow-game-launch",
                        "--allow-game-directory-write", "--run-gtfo-runtime", "--defer-approval"})
            usage = measurement["model"]["usage"]
            row["metrics"] = {"input_tokens": usage["input_tokens"]["total"],
                "output_tokens": usage["output_tokens"]["total"], "model_generation_records": measurement["model"]["generation_records"],
                "observed_tool_calls": measurement["tools"]["selected_record_total"],
                "model_elapsed_ms_sum": usage["generation_elapsed_ms_sum"], "full_delivery_cost": False}
            if completed.returncode != 0 or result.get("status") not in {"awaiting_approval", "awaiting_delivery", "completed", "built"}:
                row.update(status="product_failed", success=False, terminal_status=result.get("status"))
                continue
            delivery_started = time.perf_counter()
            _read_reference(judge_ref)
            try:
                raw_delivery = delivery_adapter({"trial": deepcopy(spec_row), "run_root": str(run_root),
                    "initial_result": deepcopy(result), "initial_manifest": deepcopy(manifest), "judge_ref": deepcopy(judge_ref)})
            finally:
                row["delivery_seconds"] = time.perf_counter() - delivery_started
            _read_reference(judge_ref)
            delivery = _delivery(raw_delivery, run_root, row["run_id"], measurement, result, judge_ref)
            row.update(delivery=delivery, status=delivery["status"], success=delivery["success"])
            row["metrics"]["full_delivery_cost"] = delivery["model_cost_known_zero"]
            if not delivery["model_cost_known_zero"]:
                for key in ("input_tokens", "output_tokens", "model_generation_records"):
                    row["metrics"][key] = None
            extra_tools = delivery["additional_observed_tools"]
            if extra_tools is None or row["metrics"]["observed_tool_calls"] is None:
                row["metrics"]["observed_tool_calls"] = None
            else:
                row["metrics"]["observed_tool_calls"] += extra_tools
        except Exception as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            if completed is not None and completed.returncode and not (run_root / "run_result.json").is_file():
                row["terminal_record_failure"] = failure
                failure = {"type": "LauncherExitError", "stage": "normal_entry",
                    "exit_code": completed.returncode,
                    "message": f"Product launcher exited with code {completed.returncode} before writing run_result.json."}
                try:
                    reported = json.loads(row.get("launcher_stdout") or "")
                except (ValueError, TypeError):
                    reported = None
                if isinstance(reported, dict):
                    if isinstance(reported.get("reason"), str) and reported["reason"]:
                        failure["message"] = reported["reason"]
                    if isinstance(reported.get("status"), str):
                        failure["reported_status"] = reported["status"]
            row.update(status="infrastructure_failure", success=False, failure=failure)
        except KeyboardInterrupt:
            row.update(status="interrupted", success=None)
            raise
        finally:
            row["execution_seconds"] = time.perf_counter() - started
            report["summary"] = summarize_trials(report["trials"])
            if checkpoint:
                checkpoint(deepcopy(report))
    return report


def summarize_trials(rows):
    tasks = {row["task_id"] for row in rows}
    def total(selected, key):
        values = [(row.get("metrics") or {}).get(key) for row in selected]
        return sum(values) if values and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values) else None
    groups = {}
    for arm in ("baseline", "candidate"):
        selected = [row for row in rows if row["arm"] == arm]
        successes = sum(row.get("success") is True for row in selected)
        seconds = [row.get("execution_seconds") for row in selected]
        input_tokens = total(selected, "input_tokens")
        output_tokens = total(selected, "output_tokens")
        # 为什么要有它：每次成功交付的Token成本包含失败运行及输出；任一用量缺失时不能只拿输入冒充总成本。
        token_cost = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        groups[arm] = {"trials": len(selected), "successful_deliveries": successes,
            "outcomes": dict(Counter(row["status"] for row in selected)),
            "completion_fraction": successes / len(selected) if selected and all(type(row.get("success")) is bool for row in selected) else None,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "observed_tool_calls": total(selected, "observed_tool_calls"),
            "tokens_per_successful_delivery": token_cost / successes if successes and token_cost is not None else None,
            "tokens_per_successful_delivery_scope": "input_plus_output_including_failed_runs",
            "p95_execution_seconds": sorted(seconds)[math.ceil(len(seconds) * .95) - 1]
                if seconds and all(type(value) in (int, float) for value in seconds) else None}
    pairs = {}
    for row in rows:
        pairs.setdefault((row["task_id"], row["repeat"]), {})[row["arm"]] = row
    platform_groups = [{(row.get("platform_execution") or {}).get("platform_files_sha256")
        for row in rows if row["arm"] == arm} for arm in ("baseline", "candidate")]
    fixed_platforms = all(len(group) == 1 and None not in group for group in platform_groups)
    comparable = fixed_platforms and bool(pairs) and all(set(pair) == {"baseline", "candidate"}
        and all((row.get("platform_execution") or {}).get("status") == "verified" for row in pair.values())
        and pair["baseline"].get("comparison_identity") is not None
        and pair["baseline"].get("comparison_identity") == pair["candidate"].get("comparison_identity") for pair in pairs.values())
    resource_comparable = comparable and all(row.get("success") is True
        and (row.get("metrics") or {}).get("full_delivery_cost") is True
        and all(type(row["metrics"].get(key)) is int and row["metrics"][key] >= 0
                for key in ("input_tokens", "output_tokens", "observed_tool_calls")) for row in rows)
    return {"distinct_task_count": len(tasks), "trial_count": len(rows), "independent_task_count": None,
        "by_arm": groups, "same_task_model_budget_judge": comparable,
        "fixed_platform_per_arm": fixed_platforms,
        "complete_paired_delivery_resource_comparison": resource_comparable,
        "input_token_reduction": (groups["baseline"]["input_tokens"] - groups["candidate"]["input_tokens"])
            if resource_comparable and all(groups[arm]["input_tokens"] is not None for arm in groups) else None,
        "scope": "selected_serial_execution_segments; repeated trials are not independent tasks; native tools only"}
