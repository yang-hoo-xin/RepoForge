"""Read-only adoption adapter for the existing restricted C# host evaluator.

Scope: fixed-image C# source sets and a frozen independent JSON case bundle.
为什么要有它：采用判断复算真实编译、隔离执行和宿主断言原件，不以 independent 字段代替独立验收。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.mod_agent.evaluation import restricted_csharp as runtime
from src.mod_agent.evaluation import restricted_owned_judge as judge
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_state import TaskAgentV2State

SCHEMA = "strategy_restricted_csharp_evaluator_v1"


def _require(value, code):
    if not value:
        raise ValueError("STRATEGY_RESTRICTED_" + code)


def _process(reference, records, *, mounts, entrypoint, command, is_runtime):
    result = read_bound_json(reference)
    judge._admit_execution(result)
    _require(result.get("completed") is True and not result.get("timed_out")
        and not result.get("output_limit_exceeded") and result.get("local_process_reaped") is True
        and result.get("cli_exit_code") == result["exit_code"] == result["container_state"]["ExitCode"],
        "PROCESS_NOT_COMPLETED")
    _require(runtime._ref(Path(result["docker_executable_ref"]["path"])) == result["docker_executable_ref"], "DOCKER_IDENTITY_CHANGED")
    for field in ("stdout", "stderr"):
        _require(runtime._hash(result[field].encode()) == result[field + "_sha256"], "PROCESS_STREAM_CHANGED")
    refs = records.get(reference["sha256"])
    _require(isinstance(refs, dict) and set(refs) == {"create", "inspect_before", "inspect_after", "cleanup"}, "PROCESS_ORIGINALS_MISSING")
    for key, filename in (("create", "create.json"), ("inspect_before", "inspect-before.json"),
                          ("inspect_after", "inspect-after.json"), ("cleanup", "cleanup.json")):
        _require(Path(refs[key]["path"]).resolve() == Path(reference["path"]).parent.resolve() / filename,
            "PROCESS_ORIGINAL_SCOPE_CHANGED")
    originals = {key: read_bound_json(value) for key, value in refs.items()}
    _require(all(row.get("exit_code") == 0 for row in originals.values()), "PROCESS_LIFECYCLE_FAILED")
    _require(all(Path(row["argv"][0]).resolve() == Path(result["docker_executable_ref"]["path"]).resolve()
        for row in originals.values()), "PROCESS_EXECUTABLE_CHANGED")
    created = originals["create"]
    container = result["container_id"]
    _require(created["stdout"].strip() == container and "--pull=never" in created["argv"]
        and runtime.IMAGE in created["argv"], "CONTAINER_CREATE_CHANGED")
    tokens = [arg.split("=", 2)[2] for arg in created["argv"] if arg.startswith("--label=codex.restricted-csharp=")]
    _require(len(tokens) == 1, "CONTAINER_LABEL_MISSING")
    for key in ("inspect_before", "inspect_after"):
        observed = originals[key]
        _require(observed["argv"][-2:] == ["inspect", container], "CONTAINER_INSPECT_CHANGED")
        documents = json.loads(observed["stdout"])
        _require(isinstance(documents, list) and len(documents) == 1, "CONTAINER_INSPECT_INVALID")
        runtime._verify_inspect(documents[0], token=tokens[0], entrypoint=entrypoint,
            command=command, mounts=mounts, runtime=is_runtime)
        if key == "inspect_after":
            _require(documents[0]["State"] == result["container_state"] and documents[0]["Image"] == result["image_id"],
                "CONTAINER_RESULT_CHANGED")
    _require(originals["cleanup"]["argv"][-3:] == ["rm", "--force", container], "CONTAINER_CLEANUP_CHANGED")
    return result


def _bundle(evaluator, bundle_ref):
    bundle = read_bound_json(bundle_ref)
    _require(bundle.get("schema_version") == "restricted_strategy_evidence_bundle_v1", "BUNDLE_REQUIRED")
    context = read_bound_json(bundle["context_ref"])
    sources = evaluator._materialize(context)
    expected = {name: runtime._hash(data) for name, data in sources.items()}
    report = read_bound_json(bundle["report_ref"])
    _require(Path(bundle["report_ref"]["path"]).resolve().is_relative_to(evaluator.output)
        and Path(bundle["private_report_ref"]["path"]).resolve().is_relative_to(evaluator.private_output),
        "EVIDENCE_TRUST_BOUNDARY_CHANGED")
    _require(report.get("schema_version") == "restricted_owned_behavior_v1"
        and report.get("verification_identity") == evaluator.verification_identity()
        and report.get("execution_origin") == "executed"
        and report.get("materialized_code_identity") == context["materialized_code_identity"], "REPORT_IDENTITY_CHANGED")
    details = read_bound_json(report["verification_ref"])
    private = read_bound_json(bundle["private_report_ref"])
    _require(bundle["private_report_ref"]["sha256"] == report["private_report_sha256"]
        and private["cases_ref"] == evaluator.cases_ref and private["task_id"] == evaluator.task_id
        and private["candidate_source_manifest"] == expected, "PRIVATE_REPORT_CHANGED")
    compilation = read_bound_json(bundle["compilation_receipt_ref"])
    compile_root = Path(bundle["compilation_receipt_ref"]["path"]).parent
    private_root = Path(bundle["private_report_ref"]["path"]).parent.resolve()
    _require(compile_root.resolve() == private_root / "compile", "COMPILATION_PRIVATE_SCOPE_CHANGED")
    inputs = {**sources, evaluator.worker_name: evaluator._worker_bytes()}
    input_manifest = {**{name: runtime._hash(value) for name, value in inputs.items()},
        "sources.rsp": runtime._hash("\n".join("/inputs/" + name for name in sorted(inputs)).encode())}
    _require(compilation.get("schema_version") == "restricted_csharp_compilation_v1"
        and compilation["image"] == runtime.IMAGE and compilation["sdk_version"] == runtime.SDK_VERSION
        and compilation["runtime_version"] == runtime.RUNTIME_VERSION
        and compilation["source_manifest"] == input_manifest
        and runtime._source_identity(compile_root / "public-input") == input_manifest, "COMPILATION_SOURCE_CHANGED")
    compiled = _process(compilation["execution_ref"], bundle["process_records"],
        mounts={"/inputs": (compile_root / "public-input", True), "/output": (compile_root / "compiler-output", False)},
        entrypoint="/bin/sh", command=["-c", runtime._COMPILE], is_runtime=False)
    _require(compilation["succeeded"] is True and compiled["exit_code"] == 0, "COMPILED_BEHAVIOR_REQUIRED")
    for field in ("binary_ref", "runtimeconfig_ref"):
        _require(runtime._ref(Path(compilation[field]["path"])) == compilation[field], "COMPILED_ARTIFACT_CHANGED")
    summary = read_bound_json(report["compilation_report_ref"])
    _require(summary["compiler_receipt_sha256"] == bundle["compilation_receipt_ref"]["sha256"]
        and summary["source_manifest"] == expected and summary["built_artifact_ref"]["sha256"] == compilation["binary_ref"]["sha256"],
        "PUBLIC_COMPILATION_CHANGED")
    cases = evaluator._case_set["cases"]
    _require(len(private["rows"]) == len(cases), "INDEPENDENT_CASES_NOT_EXECUTED")
    observed_checks = []
    for case, row in zip(cases, private["rows"]):
        _require(row["case_id"] == case["case_id"] and row["assertions"] == case["assertions"], "CASE_BINDING_CHANGED")
        run_ref = row["execution_ref"]
        run_root = Path(run_ref["path"]).parent
        _require(run_root.resolve().is_relative_to(private_root), "RUNTIME_PRIVATE_SCOPE_CHANGED")
        execution = _process(run_ref, bundle["process_records"], mounts={"/assembly": (run_root / "public-assembly", True)},
            entrypoint="dotnet", command=["/assembly/Subject.dll"], is_runtime=True)
        _require(execution["compilation_receipt_ref"] == bundle["compilation_receipt_ref"]
            and execution["stdin_sha256"] == canonical_hash(case["operation"])
            and execution["host_judge_material_mounted"] is False
            and execution["assembly_manifest"] == {"Subject.dll": compilation["binary_ref"]["sha256"],
                "Subject.runtimeconfig.json": compilation["runtimeconfig_ref"]["sha256"]}
            and runtime._source_identity(run_root / "public-assembly") == execution["assembly_manifest"], "RUN_SOURCE_OR_INPUT_CHANGED")
        observation = judge._json(execution["stdout"]) if execution["exit_code"] == 0 else None
        outcomes = [execution["exit_code"] == 0 and judge._assertion_passed(observation, assertion) for assertion in case["assertions"]]
        _require(row["observation"] == observation and row["failed_assertions"] == outcomes.count(False), "ASSERTION_RESULT_CHANGED")
        observed_checks.extend(outcomes)
    _require(observed_checks and [row["passed"] for row in details["checks"]] == observed_checks
        and details["executions"] == len(cases), "PUBLIC_CHECKS_CHANGED")
    success = all(observed_checks)
    _require(report["status"] == ("passed" if success else "behavior_failed")
        and report["verification_summary"] == {"checks": len(observed_checks),
            "failed_checks": observed_checks.count(False), "executions": len(cases)}, "FINAL_VERDICT_CHANGED")
    projected = judge.aggregate_result({**report, "report_ref": bundle["report_ref"]}, context=context)
    _require(projected["task_success"] is success, "AGGREGATE_CHANGED")
    return {"task_success": success, "preservation_passed": success, "source_manifest": expected,
        "verification_identity": evaluator.verification_identity(), "report_ref": bundle["report_ref"]}


def _evaluator(evaluator_ref):
    configuration = read_bound_json(evaluator_ref)
    _require(configuration.get("schema_version") == SCHEMA, "EVALUATOR_NOT_SUPPORTED")
    _require(configuration.get("adoption_adapter_ref") == runtime._ref(Path(__file__)), "ADOPTION_ADAPTER_CHANGED")
    evaluator = judge.RestrictedCSharpEvaluator(**configuration["kwargs"])
    _require(evaluator._case_set["eligibility"] == "independent_strategy_acceptance"
        and evaluator.public_cases_ref is None and evaluator.verification_identity() == configuration["verification_identity"],
        "EVALUATOR_NOT_INDEPENDENT_OR_CHANGED")
    qualification = read_bound_json(configuration["qualification_ref"])
    _require(qualification.get("schema_version") == "restricted_strategy_qualification_v1"
        and qualification["verification_identity"] == evaluator.verification_identity()
        and set(qualification["controls"]) == {"correct", "equivalent", "wrong"}, "QUALIFICATION_REQUIRED")
    controls = {role: _bundle(evaluator, reference) for role, reference in qualification["controls"].items()}
    _require(controls["correct"]["task_success"] and controls["equivalent"]["task_success"]
        and not controls["wrong"]["task_success"]
        and len({canonical_hash(value["source_manifest"]) for value in controls.values()}) == 3, "QUALIFICATION_CONTROL_FAILED")
    return configuration, evaluator


def validate_restricted_strategy_verdict(verdict):
    configuration, evaluator = _evaluator(verdict["evaluator_ref"])
    execution = read_bound_json(verdict["execution_ref"])
    binding = execution["candidate_state"]
    state = TaskAgentV2State(binding["artifact_store_root"], binding["task_contract_ref"])
    candidate = state.validate_candidate(binding["candidate_ref"])
    _require(candidate["candidate_source_id"] == execution["candidate_source_id"] == verdict["candidate_source_id"],
        "CANDIDATE_VERSION_CHANGED")
    materialized = state.materialize(binding["candidate_ref"])
    mapping = configuration["source_path_map"]
    _require(set(mapping.values()) == set(evaluator.sources) and set(mapping) <= set(materialized)
        and set(candidate["files"]) <= set(mapping), "CANDIDATE_SOURCE_MAP_CHANGED")
    expected = {target: runtime._hash(materialized[source].encode("utf-8")) for source, target in mapping.items()}
    result = _bundle(evaluator, verdict["evaluator_result_ref"])
    _require(result["source_manifest"] == expected and evaluator.task_id == verdict["task_id"] == execution["task_id"]
        and all(verdict[key] is result[key] for key in ("task_success", "preservation_passed")), "VERDICT_CANDIDATE_CHANGED")
    return {"status": "verified_restricted_csharp", "verification_identity": evaluator.verification_identity(),
        "qualification_ref": configuration["qualification_ref"], "report_ref": result["report_ref"],
        "scope": "fixed_image_csharp_source_set_independent_json_cases"}


def generate_restricted_strategy_verdict(*, evaluator_ref, execution_ref, evidence_bundle_ref, output_path):
    """Publish only allowed aggregate fields after validating original executions.

    为什么要有它：私有case及运行详情留在宿主评测侧，模型经验提炼只使用其另有授权的公开材料。
    """
    _configuration, evaluator = _evaluator(evaluator_ref)
    execution = read_bound_json(execution_ref)
    result = _bundle(evaluator, evidence_bundle_ref)
    verdict = {"schema_version": "independent_strategy_verdict_v1", "independent": True,
        "batch_complete": True, "task_id": execution["task_id"], "execution_ref": execution_ref,
        "evaluator_ref": evaluator_ref, "candidate_source_id": execution["candidate_source_id"],
        "task_success": result["task_success"], "preservation_passed": result["preservation_passed"],
        "evaluator_result_ref": evidence_bundle_ref}
    validate_restricted_strategy_verdict(verdict)
    return runtime._save(Path(output_path), verdict)


def bind_restricted_strategy_evidence(*, report_ref, context_ref, private_report_ref, output_path):
    """Bind the existing evaluator's completed public/private process originals."""
    report, private = read_bound_json(report_ref), read_bound_json(private_report_ref)
    _require(report["private_report_sha256"] == private_report_ref["sha256"], "PRIVATE_REPORT_CHANGED")
    compile_ref = runtime._ref(Path(private_report_ref["path"]).parent / "compile" / "compilation.json")
    compilation = read_bound_json(compile_ref)
    processes = [compilation["execution_ref"], *[row["execution_ref"] for row in private["rows"]]]
    records = {}
    for reference in processes:
        directory = Path(reference["path"]).parent
        records[reference["sha256"]] = {key: runtime._ref(directory / (name + ".json"))
            for key, name in (("create", "create"), ("inspect_before", "inspect-before"),
                ("inspect_after", "inspect-after"), ("cleanup", "cleanup"))}
    return runtime._save(Path(output_path), {"schema_version": "restricted_strategy_evidence_bundle_v1",
        "report_ref": report_ref, "context_ref": context_ref, "private_report_ref": private_report_ref,
        "compilation_receipt_ref": compile_ref, "process_records": records})


def qualify_restricted_strategy_evaluator(*, kwargs, controls, source_path_map, output_root):
    """Qualify retained controls, without executing new cases or training models."""
    evaluator = judge.RestrictedCSharpEvaluator(**kwargs)
    _require(set(controls) == {"correct", "equivalent", "wrong"}, "QUALIFICATION_REQUIRED")
    observed = {key: _bundle(evaluator, ref) for key, ref in controls.items()}
    _require(observed["correct"]["task_success"] and observed["equivalent"]["task_success"]
        and not observed["wrong"]["task_success"]
        and len({canonical_hash(row["source_manifest"]) for row in observed.values()}) == 3, "QUALIFICATION_CONTROL_FAILED")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    qualification = runtime._save(root / "qualification.json", {"schema_version": "restricted_strategy_qualification_v1",
        "verification_identity": evaluator.verification_identity(), "controls": controls})
    reference = runtime._save(root / "evaluator.json", {"schema_version": SCHEMA, "kwargs": kwargs,
        "verification_identity": evaluator.verification_identity(), "qualification_ref": qualification,
        "source_path_map": source_path_map, "adoption_adapter_ref": runtime._ref(Path(__file__))})
    _evaluator(reference)
    return reference
