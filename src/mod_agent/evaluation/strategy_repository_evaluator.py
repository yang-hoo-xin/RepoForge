"""Read-only independent acceptance of full SDK repository execution originals.

为什么要有它：采用经验必须重新核验源码、编译、测试原件；宿主工作区执行不冒充 OS 沙箱。
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_execution import _worker_binding
from src.mod_agent.task_agent_v2_state import TaskAgentV2State

SCHEMA = "strategy_repository_csharp_evaluator_v1"


def _require(value, code):
    if not value:
        raise ValueError("STRATEGY_REPOSITORY_" + code)


def _groups(tests, kind):
    groups = {}
    for row in tests:
        if kind == "trx":
            definitions = row.get("test_definitions", [])
            _require(len(definitions) == 1 and definitions[0].get("class_name")
                     and definitions[0].get("method_name"), "TEST_METHOD_IDENTITY_MISSING")
            key = definitions[0]["class_name"] + "." + definitions[0]["method_name"]
        else:
            key = row["name"]
            _require(row["passed"] == (row["expected"] == row["actual"]), "ASSERTION_RESULT_CHANGED")
        groups.setdefault(key, []).append(row["passed"])
    return groups


def _validate_test_configuration(data, test_build_outputs=None):
    # 为什么要有它：no-build只跳过构建，不自动继承先前构建的Configuration；缺省或覆盖会选到旧产物。
    command = data["test_command"]
    prefix = command[:command.index("--")] if "--" in command else command
    selected, frameworks = [], []
    for index, argument in enumerate(prefix):
        _require(not argument.startswith("@") and not re.search(r"(?:^|[;,:])configuration\s*=", argument, re.I),
                 "TEST_CONFIGURATION_OVERRIDE_UNSUPPORTED")
        # 为什么要有它：框架须匹配实际构建回执；尚未绑定的RID或输出目录覆盖继续拒绝。
        selector = re.split(r"[=:]", argument.lower(), maxsplit=1)[0]
        _require(selector not in {"--runtime", "-r", "--arch", "-a", "--os",
            "--output", "-o", "--artifacts-path", "--self-contained", "--no-self-contained",
            "-p", "/p", "/property", "-property", "--property"}, "TEST_ARTIFACT_SELECTION_UNSUPPORTED")
        if argument in {"--configuration", "-c"}:
            _require(index + 1 < len(prefix), "TEST_CONFIGURATION_VALUE_MISSING")
            selected.append(prefix[index + 1])
        elif argument.startswith(("--configuration=", "-c=")):
            selected.append(argument.split("=", 1)[1])
        if argument in {"--framework", "-f"}:
            _require(index + 1 < len(prefix), "TEST_FRAMEWORK_VALUE_MISSING")
            frameworks.append(prefix[index + 1])
        elif argument.startswith(("--framework=", "-f=")):
            frameworks.append(argument.split("=", 1)[1])
    _require(len(selected) == 1 and selected[0] in {"{configuration}", data["configuration"]},
             "TEST_CONFIGURATION_MUST_MATCH_BUILD")
    _require(all(key.lower() != "configuration" or value == data["configuration"]
                 for key, value in data.get("build_properties", {}).items()), "TEST_CONFIGURATION_BUILD_OVERRIDE")
    selectors = {"targetframework", "targetframeworks", "runtimeidentifier", "runtimeidentifiers", "outputpath",
        "baseoutputpath", "outdir", "artifactspath", "useartifactsoutput", "assemblyname", "targetname",
        "targetext", "targetpath", "selfcontained", "platform", "platformtarget",
        "appendtargetframeworktooutputpath", "appendruntimeidentifiertooutputpath"}
    _require(not selectors.intersection(key.lower() for key in data.get("build_properties", {})),
             "TEST_ARTIFACT_SELECTION_UNSUPPORTED")
    if frameworks:
        _require(len(frameworks) == 1 and re.fullmatch(r"[A-Za-z0-9_.-]+", frameworks[0]), "TEST_FRAMEWORK_INVALID")
        captured = test_build_outputs or {}
        matching = [row for row in captured.get("outputs", []) if row.get("target_framework") == frameworks[0]
            and row.get("configuration") == data["configuration"] and row.get("runtime_identifier") == ""]
        _require(captured.get("status") == "captured" and len(matching) == 1, "TEST_FRAMEWORK_OUTPUT_REQUIRED")


def _stage_facts(report, root):
    # 为什么要有它：采用评测消费后置核验完成的真实回执，不用最终汇总覆盖缺失或未知执行。
    ref = report.get("integrity_ref")
    _require(isinstance(ref, dict) and Path(ref["path"]).resolve() == root / "integrity.json", "INTEGRITY_REQUIRED")
    integrity = read_bound_json(ref)
    stages = report.get("stage_receipts", {})
    _require(integrity.get("status") == "validated" and integrity.get("stage_receipts") == stages
        and integrity.get("checks") == {key: {"status": "verified"} for key in backend.integrity_check_names(report["profile_ref"])},
        "INTEGRITY_UNCONFIRMED")
    context = integrity["context"]
    _require(all(context.get(key) == report.get(key) for key in (
        "source_root", "source_inputs", "profile_ref", "verification_identity")), "STAGE_SOURCE_CHANGED")
    if "diagnostic_files" in integrity["checks"]:
        backend.validate_diagnostic_integrity(report)
    _require({"build", "test_build", "test_build_outputs", "test_execution", "raw_test_result"} <= set(stages), "STAGE_MISSING")
    facts = {}
    for name, reference in stages.items():
        _require(Path(reference["path"]).resolve() == root / (name + "-fact.json"), "STAGE_SCOPE_CHANGED")
        fact = read_bound_json(reference)
        _require(fact.get("schema_version") == "csharp_repository_stage_fact_v1"
            and fact.get("stage") == name and fact.get("context") == context, "STAGE_IDENTITY_CHANGED")
        facts[name] = fact
    test = report["test_execution"]
    _require(facts["build"]["execution"] == report["build_execution"]
        and facts["test_build"]["execution"] == test["build"]
        and facts["test_build_outputs"]["execution"] == test["test_build_outputs"]
        and facts["test_execution"]["execution"] == {key: value for key, value in test.items() if key != "report_ref"}
        and facts["raw_test_result"]["raw_report_ref"] == test["report_ref"], "STAGE_RESULT_CHANGED")
    return facts


def _report(configuration, reference):
    report = read_bound_json(reference)
    profile = backend.CSharpRepositoryProfile(configuration["profile_ref"])
    data = profile.data
    _validate_test_configuration(data, report.get("test_execution", {}).get("test_build_outputs"))
    source = Path(report["source_root"]).resolve(strict=True)
    root = Path(reference["path"]).parent.resolve()
    _require(Path(reference["path"]).name == "report.json", "NATIVE_REPORT_REQUIRED")
    _require(report.get("execution_origin") == "executed"
             and report.get("profile_ref") == profile.reference, "EXECUTION_PROFILE_CHANGED")
    profile.verify_fixed_inputs(source, configuration["fixed_inputs"])
    _require(report["fixed_inputs"] == configuration["fixed_inputs"]
             and backend._snapshot(source) == report["source_inputs"], "SOURCE_CHANGED")
    identity = {"validator_sha256": canonical_hash(backend.verification_owner_hashes(data)),
                "inputs_sha256": canonical_hash({"profile": profile.reference,
                    "fixed_inputs": configuration["fixed_inputs"]}),
                "environment_sha256": canonical_hash(data["dotnet_ref"]), "deterministic_reuse_allowed": False}
    _require(report["verification_identity"] == identity, "RUNNER_IDENTITY_CHANGED")
    compilation_ref = report["compilation_report_ref"]
    _require(Path(compilation_ref["path"]).resolve() == root / "compilation.json", "COMPILATION_SCOPE_CHANGED")
    compilation = read_bound_json(compilation_ref)
    build_profile = compilation["build_profile"]
    expected_build = [data["dotnet_ref"]["path"], "build", str(backend._within(source, data["project_path"])),
                      "--no-restore", "--configuration", data["configuration"], "-p:UseSharedCompilation=false",
                      *(f"-p:{key}={value}" for key, value in sorted(data.get("build_properties", {}).items())),
                      f"-p:CustomBeforeMicrosoftCommonTargets={root / 'compile-inputs.targets'}",
                      f"-p:RepositoryCompileInputs={root / 'compile-inputs.txt'}",
                      "-p:BuildProjectReferences=true", "-t:Rebuild", "-m:1"]
    _require(build_profile == report["isolated_build_profile"]
             and build_profile["build_command"] == expected_build
             and Path(build_profile["working_directory"]).resolve() == source
             and build_profile["restore_command"] == [data["dotnet_ref"]["path"], "restore",
                 str(backend._within(source, data["project_path"])), "--configfile", build_profile["config_file"],
                 "--packages", build_profile["packages_path"]]
             and Path(build_profile["config_file"]).read_bytes() == backend._within(source, data["nuget_config_path"]).read_bytes(),
             "BUILD_COMMAND_CHANGED")
    for key in ("source_inputs", "compile_inputs", "build_execution", "built_artifact_refs", "profile_ref"):
        _require(compilation[key] == report[key], "COMPILATION_BINDING_CHANGED")
    _require(compilation.get("execution_origin") == "executed" and compilation.get("status") == "passed"
             and compilation["all_required_files_compiled"] is True
             and report["build_execution"].get("restore_exit_code") == 0
             and report["build_execution"].get("build_exit_code") == 0, "COMPILATION_NOT_COMPLETED")
    hashes = {row["path"]: row["sha256"] for row in report["source_inputs"]}
    production = set(configuration["production_paths"])
    test_root = PurePosixPath(data["test_project_path"]).parent
    _require(set(data["allowed_edit_paths"]) <= production <= set(hashes)
             and all(not PurePosixPath(path).is_relative_to(test_root) for path in production), "PRODUCTION_SCOPE_INVALID")
    # 为什么要有它：全部生产 C# 文件均须比较，不能只对照模型改动的一个片段。
    required_production = {path for path in hashes if not PurePosixPath(path).is_relative_to(test_root)}
    _require(required_production == production, "PRODUCTION_SOURCE_OMITTED")
    compiled = report["compile_inputs"]
    _require(set(data["allowed_edit_paths"]) <= {row["path"] for row in compiled}, "EDIT_NOT_COMPILED")
    capture = root / "compile-inputs.txt"
    captured = []
    for line in capture.read_text(encoding="utf-8-sig").splitlines():
        project, filename = line.split("|", 1)
        path = Path(filename).resolve(strict=True)
        if path.is_relative_to(source):
            captured.append({"project": str(Path(project).resolve().relative_to(source)),
                             "path": path.relative_to(source).as_posix(), "sha256": _ref(path)["sha256"]})
    _require(captured == compiled and (root / "compile-inputs.targets").read_bytes() == backend._capture_targets(),
             "COMPILER_CAPTURE_CHANGED")
    for row in compiled:
        _require(_ref(backend._within(source, row["path"]))["sha256"] == row["sha256"]
                 and (hashes.get(row["path"]) == row["sha256"] or
                      row["path"] not in hashes and "obj" in PurePosixPath(row["path"]).parts), "COMPILED_SOURCE_CHANGED")
    artifacts = report["built_artifact_refs"]
    _require(artifacts and all(_ref(row["path"]) == row for row in artifacts)
             and {Path(row["path"]).resolve() for row in artifacts} ==
                 {backend._within(source, path) for path in data["expected_artifacts"]}, "ARTIFACT_CHANGED")
    test = report["test_execution"]
    _require(test.get("status") == "executed" and test.get("exit_code") in (0, 1)
             and test["build"].get("restore_exit_code") == 0 and test["build"].get("build_exit_code") == 0,
             "TEST_EXECUTION_INCOMPLETE")
    test_ref = test["report_ref"]
    expected_report = backend._within(root / "test-output", data["test_report"]["path"])
    _require(Path(test_ref["path"]).resolve() == expected_report and _ref(expected_report) == test_ref,
             "TEST_REPORT_CHANGED")
    command = [arg.format(test_project=str(backend._within(source, data["test_project_path"])),
                         report_path=str(expected_report), configuration=data["configuration"])
               for arg in data["test_command"]]
    split = command.index("--") if "--" in command else len(command)
    for flag in ("--no-build", "--no-restore"):
        if flag not in command[:split]:
            command.insert(split, flag)
            split += 1
    _require(test["command"] == [data["dotnet_ref"]["path"], *command], "TEST_COMMAND_CHANGED")
    facts = _stage_facts(report, root)
    expected_test_build = list(backend.expected_test_build_command(data, source,
        root / "test-outputs.targets", root / "test-outputs.txt"))
    _require(facts["test_build"]["commands"]["build"] == expected_test_build
        and facts["test_execution"]["commands"]["test"] == test["command"], "TEST_BUILD_COMMAND_CHANGED")
    captured = backend._test_build_outputs(data, source, root / "test-outputs.targets", root / "test-outputs.txt")
    _require(captured == test["test_build_outputs"], "TEST_BUILD_OUTPUTS_CHANGED")
    tests, test_metadata = backend.read_repository_tests(expected_report, data, report=report)
    _require(tests == report["checks"], "TEST_CHECKS_CHANGED")
    groups = _groups(tests, data["test_report"]["format"])
    target, preservation = set(configuration["target_test_ids"]), set(configuration["preservation_test_ids"])
    _require(target and preservation and not target & preservation and target | preservation == set(groups),
             "TEST_SCOPE_INCOMPLETE")
    counts = {key: len(values) for key, values in groups.items()}
    if "test_counts" in configuration:
        _require(counts == configuration["test_counts"], "TEST_INSTANCES_CHANGED")
    success = all(row["passed"] for row in tests)
    # 为什么要有它：JSON断言程序可正常退出并报告断言失败；失败由已核验结果决定，成功仍须零退出码。
    _require((not success or test["exit_code"] == 0) and report["status"] == ("passed" if success else "behavior_failed")
             and report["failure_owner"] == (None if success else "patch_implementation"), "TEST_EXIT_OR_STATUS_CHANGED")
    _require(report["verification_summary"] == {"checks": len(tests), "failed_checks": sum(not row["passed"] for row in tests),
             "executions": 1, "test_command_executed": True}, "TEST_SUMMARY_CHANGED")
    return {"task_success": all(all(groups[key]) for key in target),
            **({"test_execution_metadata": test_metadata} if test_metadata else {}),
            "preservation_passed": all(all(groups[key]) for key in preservation),
            "source_manifest": {path: hashes[path] for path in sorted(production)}, "test_counts": counts,
            "report_ref": reference, "verification_identity": identity}


def _configuration(reference):
    config = read_bound_json(reference)
    _require(config.get("schema_version") == SCHEMA and config.get("adoption_adapter_ref") == _ref(__file__),
             "EVALUATOR_CHANGED")
    qualification = read_bound_json(config["qualification_ref"])
    _require(qualification.get("schema_version") == "repository_strategy_qualification_v1"
             and set(qualification["controls"]) == {"correct", "equivalent", "wrong"}, "QUALIFICATION_REQUIRED")
    controls = {key: _report(config, ref) for key, ref in qualification["controls"].items()}
    _require(all(controls[role]["task_success"] and controls[role]["preservation_passed"] for role in ("correct", "equivalent"))
             and not controls["wrong"]["task_success"]
             and len({canonical_hash(row["source_manifest"]) for row in controls.values()}) == 3,
             "QUALIFICATION_CONTROL_FAILED")
    return config


def qualify_repository_strategy_evaluator(*, profile_ref, controls, production_paths, target_test_ids,
                                         preservation_test_ids, output_root):
    """Bind fixed host inputs and existing executed qualification controls."""
    _require(set(controls) == {"correct", "equivalent", "wrong"}, "QUALIFICATION_REQUIRED")
    profile = backend.CSharpRepositoryProfile(profile_ref)
    correct = read_bound_json(controls["correct"])
    config = {"schema_version": SCHEMA, "profile_ref": profile_ref,
              "fixed_inputs": profile.bind(correct["source_root"]), "production_paths": sorted(production_paths),
              "target_test_ids": sorted(target_test_ids), "preservation_test_ids": sorted(preservation_test_ids),
              "execution_scope": "host_workspace_execution_no_os_isolation", "adoption_adapter_ref": _ref(__file__)}
    observed = {role: _report(config, ref) for role, ref in controls.items()}
    config["test_counts"] = observed["correct"]["test_counts"]
    _require(all(row["test_counts"] == config["test_counts"] for row in observed.values()), "TEST_INSTANCES_CHANGED")
    _require(all(observed[role]["task_success"] and observed[role]["preservation_passed"] for role in ("correct", "equivalent"))
             and not observed["wrong"]["task_success"]
             and len({canonical_hash(row["source_manifest"]) for row in observed.values()}) == 3, "QUALIFICATION_CONTROL_FAILED")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    config["qualification_ref"] = _save(root / "qualification.json", {
        "schema_version": "repository_strategy_qualification_v1", "controls": controls})
    reference = _save(root / "evaluator.json", config)
    _configuration(reference)
    return reference


def validate_repository_strategy_verdict(verdict):
    config = _configuration(verdict["evaluator_ref"])
    execution = read_bound_json(verdict["execution_ref"])
    binding = execution["candidate_state"]
    state = TaskAgentV2State(binding["artifact_store_root"], binding["task_contract_ref"])
    candidate = state.validate_candidate(binding["candidate_ref"])
    _require(candidate["candidate_source_id"] == execution["candidate_source_id"] == verdict["candidate_source_id"]
             and execution["task_id"] == verdict["task_id"], "CANDIDATE_CHANGED")
    materialized = state.materialize(binding["candidate_ref"])
    production = set(config["production_paths"])
    _require(set(candidate["files"]) <= production, "CANDIDATE_SOURCE_SCOPE_CHANGED")
    expected = {path: backend.hashlib.sha256(materialized[path].encode("utf-8")).hexdigest()
                for path in production if path in materialized}
    missing = production - set(materialized)
    if missing:
        # 为什么要有它：二进制资源不进入文本候选，但必须来自原执行冻结的生产输入，不能临时读取当前盘冒充原件。
        native = read_bound_json(execution["native_execution_ref"])
        worker_ref = native["worker_binding"]["configuration_ref"]
        _require(_worker_binding(worker_ref) == native["worker_binding"], "NATIVE_WORKER_CHANGED")
        selected = read_bound_json(worker_ref)["behavior_evaluator"]
        # 为什么要有它：冻结求解器在另一目录运行；原worker精确引用已重验，等字节后端不能因搬迁被误当成另一实现。
        _require(selected["factory_ref"]["sha256"] == _ref(backend.__file__)["sha256"]
                 and selected["factory_name"] == "make_csharp_repository_evaluator", "NATIVE_REPOSITORY_REQUIRED")
        public_profile = backend.CSharpRepositoryProfile(selected["kwargs"]["profile_ref"])
        frozen = selected["kwargs"]["expected_fixed_inputs"]
        _require(frozen["profile_sha256"] == public_profile.reference["sha256"]
                 and frozen["allowed_edit_paths"] == sorted(public_profile.data["allowed_edit_paths"]), "NATIVE_INPUT_BINDING_CHANGED")
        fixed = {row["path"]: row["sha256"] for row in frozen["fixed_files"]}
        _require(missing <= set(fixed) and not missing & set(public_profile.data["allowed_edit_paths"]),
                 "NATIVE_FIXED_INPUT_MISSING")
        expected.update({path: fixed[path] for path in missing})
    result = _report(config, verdict["evaluator_result_ref"])
    _require(result["source_manifest"] == expected and all(verdict[key] is result[key]
             for key in ("task_success", "preservation_passed")), "VERDICT_CHANGED")
    return {"status": "verified_repository_csharp", "verification_identity": result["verification_identity"],
            "qualification_ref": config["qualification_ref"], "report_ref": result["report_ref"],
            "scope": config["execution_scope"]}


def generate_repository_strategy_verdict(*, evaluator_ref, execution_ref, evidence_bundle_ref, output_path):
    """Emit aggregate results only; private test inputs stay in host evidence.

    为什么要有它：独立验收原件用于采用决定，隐藏细节不得进入策略提炼请求。
    """
    config = _configuration(evaluator_ref)
    execution = read_bound_json(execution_ref)
    result = _report(config, evidence_bundle_ref)
    verdict = {"schema_version": "independent_strategy_verdict_v1", "independent": True, "batch_complete": True,
               "task_id": execution["task_id"], "execution_ref": execution_ref, "evaluator_ref": evaluator_ref,
               "candidate_source_id": execution["candidate_source_id"], "task_success": result["task_success"],
               "preservation_passed": result["preservation_passed"], "evaluator_result_ref": evidence_bundle_ref}
    validate_repository_strategy_verdict(verdict)
    return _save(Path(output_path), verdict)
