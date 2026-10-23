"""Explicit repository Build/test backend shared by candidate and formal delivery.

为什么要有它：仓库候选与正式交付须执行同一项目和测试合同；编译成功或旧报告不等于测试通过。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import string
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4
from xml.etree import ElementTree

from src.mod_agent.build import nuget
from src.mod_agent.build.adapter import BuildAdapter, BuildCommandResult, BuildProfile
from src.mod_agent.build.dependencies import _windows_extended_length_path
from src.mod_agent.candidate_execution import materialize_candidate, materialized_code_identity
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import canonical_hash, read_bound_json

_IGNORED = {".git", ".idea", "bin", "obj", ".vs", "TestResults"}
_VERSION = "csharp_repository_profile_v1"
_RESERVED_BUILD_PROPERTIES = {"BuildProjectReferences", "RepositoryCompileInputs", "UseSharedCompilation"}


class _RepositoryNuGetProfile(nuget.NuGetIsolatedBuildProfile):
    @property
    def environment(self):
        # 为什么要有它：显式portable SDK只在本次子进程内使用本地CLI目录，不能污染用户设置或回退到另一安装。
        return {**super().environment,
                "DOTNET_ROOT": str(Path(self.restore_command[0]).resolve().parent),
                "DOTNET_CLI_HOME": str(self.run_root / "d"),
                "DOTNET_MULTILEVEL_LOOKUP": "0",
                "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
                "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
                # 为什么要有它：本次构建不保留可复用后台节点，不影响用户已有的全局编译服务。
                "MSBUILDDISABLENODEREUSE": "1",
                "DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER": "1"}


def _require(condition, code):
    if not condition:
        raise ValueError("CSHARP_REPOSITORY_" + code)


def _relative(value):
    _require(type(value) is str and bool(value) and "\\" not in value
             and not PurePosixPath(value).is_absolute() and not PureWindowsPath(value).drive
             and all(part not in {"", ".", ".."} for part in value.split("/"))
             and not any(char in value for char in "*?[]:"), "RELATIVE_PATH_REQUIRED")
    return value


def _within(root, value):
    path = (root / _relative(value)).resolve()
    _require(path.is_relative_to(root.resolve()), "PATH_ESCAPES_SOURCE")
    return path


def _snapshot(root):
    """Hash repository inputs, ignoring only generated build and VCS directories."""
    root = Path(root).resolve(strict=True)
    # 为什么要有它：真实生成器可产生超过Win32普通路径上限的文件；校验仍须拒绝链接并保持原相对源码身份。
    io_root = Path(_windows_extended_length_path(str(root))) if os.name == "nt" else root
    rows = []
    for path in sorted(io_root.rglob("*")):
        relative_path = path.relative_to(io_root)
        relative = relative_path.as_posix()
        _require(not path.is_symlink()
                 and not getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                 "LINKED_INPUT_UNSUPPORTED")
        _require(path.resolve().is_relative_to(io_root), "PATH_ESCAPES_SOURCE")
        if any(part in _IGNORED for part in relative_path.parts):
            continue
        if path.is_file():
            rows.append({"path": relative, "sha256": _ref(path)["sha256"]})
    return rows


class CSharpRepositoryProfile:
    """Immutable, host-selected project and test configuration, never model argv."""

    def __init__(self, reference):
        self._reference = deepcopy(reference)
        self._data = read_bound_json(self._reference)
        data = self._data
        required = {"profile_version", "dotnet_ref", "project_path", "test_project_path",
                    "nuget_config_path", "configuration", "allowed_edit_paths",
                    "expected_artifacts", "test_command", "test_report", "timeout_seconds"}
        _require(required <= set(data) <= required | {"build_properties", "probe_runner"},
                 "PROFILE_FIELDS_INVALID")
        if "probe_runner" in data:
            from src.mod_agent.domain.repository_probe_runner import validate_probe_runner_config
            validate_probe_runner_config(data["probe_runner"])
        _require(data["profile_version"] == _VERSION, "PROFILE_VERSION_INVALID")
        _require(_ref(data["dotnet_ref"]["path"]) == data["dotnet_ref"], "DOTNET_CHANGED")
        _require(Path(data["dotnet_ref"]["path"]).name.lower() in {"dotnet", "dotnet.exe"}, "DOTNET_REQUIRED")
        for key in ("project_path", "test_project_path", "nuget_config_path"):
            _relative(data[key])
        _require(Path(data["project_path"]).suffix in {".csproj", ".sln"}, "PROJECT_REQUIRED")
        _require(Path(data["test_project_path"]).suffix == ".csproj"
                 and PurePosixPath(data["test_project_path"]).parent.as_posix() != ".", "SEPARATE_TEST_PROJECT_REQUIRED")
        _require(type(data["configuration"]) is str and data["configuration"].isalnum(), "CONFIGURATION_INVALID")
        _require(type(data["timeout_seconds"]) is int and data["timeout_seconds"] > 0, "TIMEOUT_INVALID")
        properties = data.get("build_properties", {})
        _require(isinstance(properties, dict) and all(
            type(key) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", key)
            and key not in _RESERVED_BUILD_PROPERTIES
            and type(value) is str and 0 < len(value) <= 512
            and not any(character in value for character in "\r\n\0")
            for key, value in properties.items()), "BUILD_PROPERTIES_INVALID")
        for key in ("allowed_edit_paths", "expected_artifacts"):
            values = data[key]
            _require(isinstance(values, list) and values and len(set(values)) == len(values), key.upper() + "_INVALID")
            for value in values:
                _relative(value)
        test_root = PurePosixPath(data["test_project_path"]).parent
        _require(all(Path(value).suffix == ".cs" and not PurePosixPath(value).is_relative_to(test_root)
                     and not any(part in _IGNORED for part in PurePosixPath(value).parts)
                     for value in data["allowed_edit_paths"]), "EDIT_SCOPE_INVALID")
        report = data["test_report"]
        _require(isinstance(report, dict) and set(report) == {"path", "format"}
                 and report["format"] in {"json_assertions", "trx"}, "TEST_REPORT_INVALID")
        _relative(report["path"])
        command = data["test_command"]
        _require(isinstance(command, list) and command and command[0] in {"run", "test"}
                 and all(type(value) is str and value for value in command), "TEST_COMMAND_INVALID")
        fields = [field for value in command for _, field, spec, conversion in string.Formatter().parse(value)
                  if field is not None and not spec and not conversion]
        _require(set(fields) <= {"test_project", "report_path", "configuration"}
                 and fields.count("test_project") == 1 and fields.count("report_path") == 1,
                 "TEST_COMMAND_TEMPLATE_INVALID")
        for value in command:
            for _, field, spec, conversion in string.Formatter().parse(value):
                _require(field is None or (field in fields and not spec and not conversion),
                         "TEST_COMMAND_TEMPLATE_INVALID")
        if command[0] == "run":
            _require("--project" in command and command.index("--project") + 1 < len(command)
                     and command[command.index("--project") + 1] == "{test_project}",
                     "TEST_PROJECT_ARGUMENT_REQUIRED")
        else:
            _require(len(command) > 1 and command[1] == "{test_project}", "TEST_PROJECT_ARGUMENT_REQUIRED")

    @classmethod
    def load(cls, path):
        return cls(_ref(path))

    @property
    def reference(self):
        return deepcopy(self._reference)

    @property
    def data(self):
        return deepcopy(self._data)

    def _verify(self):
        _require(read_bound_json(self._reference) == self._data, "PROFILE_CHANGED")
        _require(_ref(self._data["dotnet_ref"]["path"]) == self._data["dotnet_ref"], "DOTNET_CHANGED")

    def bind(self, source_root):
        self._verify()
        root = Path(source_root).resolve(strict=True)
        _require(root.is_dir(), "SOURCE_DIRECTORY_REQUIRED")
        data = self._data
        for name in [data[key] for key in ("project_path", "test_project_path", "nuget_config_path")] + data["allowed_edit_paths"]:
            _require(_within(root, name).is_file(), "DECLARED_INPUT_MISSING")
        rows = _snapshot(root)
        allowed = set(data["allowed_edit_paths"])
        return {"binding_version": "csharp_repository_fixed_inputs_v1",
                "profile_sha256": self._reference["sha256"],
                "allowed_edit_paths": sorted(allowed),
                "fixed_files": [row for row in rows if row["path"] not in allowed],
                "source_paths": [row["path"] for row in rows]}

    def verify_fixed_inputs(self, source_root, binding):
        _require(self.bind(source_root) == binding, "FIXED_INPUTS_CHANGED")

    def create_nuget_profile(self, run_root, source_root):
        """Reuse one short, task-local cache for this immutable repository profile."""
        self.bind(source_root)
        task, source = Path(run_root).resolve(), Path(source_root).resolve()
        run = self.task_work_root(task)
        _require(not run.is_relative_to(source), "ENVIRONMENT_INSIDE_SOURCE")
        run.mkdir(parents=True, exist_ok=True)
        identity_path = run / "profile.sha256"
        # 为什么要有它：短路径只缩短物理目录名；复用缓存仍须匹配完整配置身份，不能覆盖其他运行输入。
        if identity_path.exists():
            _require(identity_path.read_text(encoding="ascii") == self.reference["sha256"], "TASK_CACHE_PROFILE_CHANGED")
        else:
            with identity_path.open("x", encoding="ascii") as stream:
                stream.write(self.reference["sha256"])
        config = run / "n.config"
        raw = _within(source, self._data["nuget_config_path"]).read_bytes()
        if config.exists():
            _require(config.read_bytes() == raw, "NUGET_CONFIG_CHANGED")
        else:
            config.write_bytes(raw)
        base = nuget._create_nuget_isolated_build_profile(
            run_root=run, working_directory=run, config_file=config, bepinex_dir=None,
            timeout_seconds=self._data["timeout_seconds"])
        executable = self._data["dotnet_ref"]["path"]
        project = str(_within(source, self._data["project_path"]))
        packages, appdata = run / "p", run / "a"
        # Scope: immutable host-selected MSBuild properties, never model-provided arguments.
        # 为什么要有它：正常仓库可能需要固定版本计算或其他环境属性；所有构建阶段必须使用同一显式合同，不能靠外层环境暗传。
        build_properties = tuple(f"-p:{key}={value}" for key, value in sorted(
            self._data.get("build_properties", {}).items()))
        return replace(_RepositoryNuGetProfile(**vars(base)), profile_id="csharp_repository_explicit_build_test",
                       working_directory=source,
                       packages_path=packages, http_cache_path=run / "h", scratch_path=run / "s",
                       plugins_cache_path=run / "u", appdata_path=appdata,
                       user_config_file=appdata / "NuGet" / "NuGet.Config",
                       additional_user_config_path=appdata / "NuGet" / "config",
                       restore_command=(executable, "restore", project, "--configfile", str(config), "--packages", str(packages)),
                       build_command=(executable, "build", project, "--no-restore", "--configuration",
                                      self._data["configuration"], "-p:UseSharedCompilation=false", *build_properties))

    def task_work_root(self, task_root):
        """Short physical workspace, derived only from the owning task and profile."""
        root = Path(task_root).resolve()
        selected = root / ".cs" / self.reference["sha256"][:12]
        _require(selected.resolve().is_relative_to(root), "TASK_CACHE_OUTSIDE_WORKSPACE")
        for component in (selected.parent, selected):
            if component.exists():
                _require(not component.is_symlink()
                         and not getattr(component.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                         "LINKED_TASK_CACHE_UNSUPPORTED")
        return selected


def _capture_targets():
    project = ElementTree.Element("Project")
    target = ElementTree.SubElement(project, "Target", Name="RecordRepositoryCompileInputs", BeforeTargets="CoreCompile")
    ElementTree.SubElement(target, "WriteLinesToFile", File="$(RepositoryCompileInputs)",
                           Lines="@(Compile->'$(MSBuildProjectFullPath)|%(FullPath)')",
                           Overwrite="false", Encoding="UTF-8")
    return ElementTree.tostring(project, encoding="utf-8", xml_declaration=True)


def _read_tests(path, kind):
    if kind == "json_assertions":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        _require(isinstance(value, dict) and value.get("schema_version") == "csharp-test-report.v1",
                 "TEST_REPORT_SCHEMA_INVALID")
        tests = value.get("tests")
        _require(isinstance(tests, list) and tests, "TESTS_NOT_EXECUTED")
        _require(all(isinstance(row, dict) and type(row.get("name")) is str and row["name"]
                     and type(row.get("passed")) is bool and {"expected", "actual"} <= set(row)
                     for row in tests), "TEST_ASSERTION_INVALID")
        _require(len({row["name"] for row in tests}) == len(tests), "DUPLICATE_TEST_NAME")
        return [{**row, "check_id": row["name"]} for row in tests]
    tree = ElementTree.parse(path).getroot()
    rows = tree.findall(".//{*}UnitTestResult")
    counters = tree.find(".//{*}Counters")
    summary = tree.find(".//{*}ResultSummary")
    _require(rows and counters is not None and summary is not None, "TESTS_NOT_EXECUTED")
    executed = int(counters.attrib.get("executed", "0"))
    _require(executed > 0 and executed == len(rows) and all(row.attrib.get("outcome") in {"Passed", "Failed"} for row in rows),
             "TESTS_NOT_EXECUTED")
    reference = _ref(path)
    definitions = defaultdict(list)
    for definition in tree.findall(".//{*}TestDefinitions/{*}UnitTest"):
        method = definition.find("{*}TestMethod")
        definitions[definition.get("id")].append({
            "name": definition.get("name"),
            "class_name": method.get("className") if method is not None else None,
            "method_name": method.get("name") if method is not None else None,
        })
    keys = [{"test_id": row.get("testId"), "display_name": row.get("testName") or row.get("testId")}
            for row in rows]
    _require(all(type(key["display_name"]) is str and key["display_name"] for key in keys), "TRX_RESULT_NAME_MISSING")
    groups = Counter(canonical_hash(key) for key in keys)
    occurrences = Counter()
    tests = []
    for index, (row, key) in enumerate(zip(rows, keys)):
        base_id = canonical_hash(key)
        occurrences[base_id] += 1
        local = groups[base_id] > 1 or not key["test_id"]
        # 为什么要有它：参数化结果的display name可能相同；真正重复组合只能按报告内出现位置区分，不能冒充跨运行语义身份。
        identity = {**key, **({"report_sha256": reference["sha256"], "occurrence": occurrences[base_id]} if local else {})}
        tests.append({
            "check_id": "trx:" + canonical_hash(identity), "name": key["display_name"],
            "test_id": key["test_id"], "test_definitions": definitions.get(key["test_id"], []),
            "identity_scope": "report_local_occurrence" if local else "runner_test_id_and_display_name",
            "cross_run_semantics_verified": False,
            "occurrence": occurrences[base_id] if local else None, "group_size": groups[base_id],
            "report_ref": reference, "result_index": index,
            "passed": row.attrib["outcome"] == "Passed", "expected": "Passed",
            "actual": row.attrib["outcome"], "details": "".join(row.itertext()).strip(),
        })
    _require(summary.attrib.get("outcome") == "Completed" or any(not row["passed"] for row in tests),
             "TRX_EXECUTION_INCOMPLETE")
    return tests


def _run_build(profile):
    started = time.perf_counter()
    try:
        return nuget._run_nuget_isolated_build(profile)
    except (OSError, subprocess.SubprocessError) as error:
        return nuget.NuGetIsolatedBuildExecution(
            restore_exit_code=-1, build_exit_code=None, restore_stdout="",
            restore_stderr=f"{type(error).__name__}: {error}", build_stdout="", build_stderr="",
            duration_ms=int((time.perf_counter() - started) * 1000))


class CSharpRepositoryBackend:
    """Compile the declared repository and execute its immutable public tests."""

    def __init__(self, *, profile_ref, source_root, workspace_root, output_root, expected_fixed_inputs=None):
        self.profile = CSharpRepositoryProfile(profile_ref)
        self.source_root = Path(source_root).resolve(strict=True)
        self.workspace_root = Path(workspace_root).resolve(strict=True)
        self.output_root = Path(output_root).resolve()
        _require(self.source_root.is_relative_to(self.workspace_root), "SOURCE_OUTSIDE_WORKSPACE")
        _require(not self.output_root.is_relative_to(self.source_root), "OUTPUT_INSIDE_SOURCE")
        self.fixed_inputs = deepcopy(expected_fixed_inputs) if expected_fixed_inputs is not None else self.profile.bind(self.source_root)
        self.profile.verify_fixed_inputs(self.source_root, self.fixed_inputs)

    def verification_identity(self):
        self.profile.verify_fixed_inputs(self.source_root, self.fixed_inputs)
        return {"validator_sha256": canonical_hash({
                    "owner_sha256": _ref(__file__)["sha256"],
                    "build_owner_sha256": _ref(nuget.__file__)["sha256"],
                }),
                "inputs_sha256": canonical_hash({"profile": self.profile.reference, "fixed_inputs": self.fixed_inputs}),
                "environment_sha256": canonical_hash(self.profile.data["dotnet_ref"]),
                "deterministic_reuse_allowed": False}

    def run_build_test(self, source_root, feature_id, input_execution_mode, *, candidate_context=None):
        source = Path(source_root).resolve(strict=True)
        self.profile.verify_fixed_inputs(source, self.fixed_inputs)
        before = _snapshot(source)
        identity = self.verification_identity()
        run = self.output_root / uuid4().hex
        run.mkdir(parents=True, exist_ok=False)
        profile = self.profile.create_nuget_profile(self.workspace_root, source)
        data = self.profile.data
        targets = run / "compile-inputs.targets"
        targets.write_bytes(_capture_targets())
        capture = run / "compile-inputs.txt"
        capture_args = (f"-p:CustomBeforeMicrosoftCommonTargets={targets}", f"-p:RepositoryCompileInputs={capture}",
                        "-p:BuildProjectReferences=true", "-t:Rebuild", "-m:1")
        profile = replace(profile, build_command=(*profile.build_command, *capture_args))
        build_profile = BuildProfile(profile_id=profile.profile_id, build_system="dotnet",
            working_directory=str(source), command=profile.build_command, timeout_seconds=profile.timeout_seconds,
            expected_artifacts=tuple(data["expected_artifacts"]))
        build_execution = {}

        def build_command(selected):
            _require(selected.command == profile.build_command, "BUILD_COMMAND_CHANGED")
            execution = _run_build(profile)
            build_execution.update(execution.to_dict())
            (run / "build.stdout.txt").write_text(execution.stdout, encoding="utf-8")
            (run / "build.stderr.txt").write_text(execution.stderr, encoding="utf-8")
            return BuildCommandResult(execution.exit_code, execution.stdout, execution.stderr, execution.duration_ms)

        build = BuildAdapter().run_local(feature_id=feature_id, profile=build_profile, command_runner=build_command,
                                        allow_execution=True, input_execution_mode=input_execution_mode)
        build = replace(build, stdout_path=str(run / "build.stdout.txt"), stderr_path=str(run / "build.stderr.txt"))
        checks, observations, test_execution = [], [], {"status": "not_executed"}
        status, owner = ("passed", None) if build.claims_real_build_success else ("behavior_failed", "patch_implementation")
        if build_execution.get("restore_exit_code") != 0:
            status, owner = "infrastructure_failed", "environment"
        compiled = []
        if build.claims_real_build_success:
            if capture.is_file():
                for line in capture.read_text(encoding="utf-8-sig").splitlines():
                    project, filename = line.split("|", 1)
                    path = Path(filename).resolve(strict=True)
                    if path.is_relative_to(source):
                        compiled.append({"project": str(Path(project).resolve().relative_to(source)),
                                         "path": path.relative_to(source).as_posix(), "sha256": _ref(path)["sha256"]})
            required = set(data["allowed_edit_paths"])
            if not required <= {row["path"] for row in compiled}:
                status, owner = "infrastructure_failed", "validator"
                observations.append({"check_id": "declared_project_compilation", "expected": sorted(required),
                                     "observed": {"compiled_paths": sorted({row["path"] for row in compiled})}})
            else:
                tests_profile = replace(profile,
                    restore_command=tuple(str(_within(source, data["test_project_path"])) if part == str(_within(source, data["project_path"]))
                                          else part for part in profile.restore_command),
                    build_command=(data["dotnet_ref"]["path"], "build", str(_within(source, data["test_project_path"])),
                                   "--no-restore", "--configuration", data["configuration"], "-m:1",
                                   "-p:UseSharedCompilation=false",
                                   *(f"-p:{key}={value}" for key, value in sorted(
                                       data.get("build_properties", {}).items()))))
                test_build = _run_build(tests_profile)
                test_execution["build"] = test_build.to_dict()
                (run / "tests-build.stdout.txt").write_text(test_build.stdout, encoding="utf-8")
                (run / "tests-build.stderr.txt").write_text(test_build.stderr, encoding="utf-8")
                if test_build.exit_code != 0:
                    status, owner = "infrastructure_failed", "environment" if test_build.restore_exit_code else "validator"
                    observations.append({"check_id": "test_project_build", "expected": "Test project builds.",
                                         "observed": {"exit_code": test_build.exit_code,
                                                      "stdout": test_build.stdout, "stderr": test_build.stderr}})
                else:
                    output = run / "test-output"
                    output.mkdir()
                    report_path = _within(output, data["test_report"]["path"])
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    _require(not report_path.exists(), "TEST_REPORT_ALREADY_EXISTS")
                    args = [value.format(test_project=str(_within(source, data["test_project_path"])),
                                         report_path=str(report_path), configuration=data["configuration"])
                            for value in data["test_command"]]
                    split = args.index("--") if "--" in args else len(args)
                    for flag in ("--no-build", "--no-restore"):
                        if flag not in args[:split]:
                            args.insert(split, flag)
                            split += 1
                    command = (data["dotnet_ref"]["path"], *args)
                    started = time.time_ns()
                    try:
                        executed = nuget._run_nuget_isolated_verification_command(profile, command)
                    except (OSError, subprocess.SubprocessError) as error:
                        executed = BuildCommandResult(-1, "", f"{type(error).__name__}: {error}",
                                                      int((time.time_ns() - started) / 1_000_000))
                    test_execution.update(status="executed", command=list(command), exit_code=executed.exit_code,
                                          stdout=executed.stdout, stderr=executed.stderr, duration_ms=executed.duration_ms)
                    try:
                        _require(report_path.is_file() and not report_path.is_symlink(), "TEST_REPORT_MISSING")
                        _require(report_path.resolve().is_relative_to(output), "TEST_REPORT_OUTSIDE_OUTPUT")
                        _require(report_path.stat().st_mtime_ns >= started - 2_000_000_000, "TEST_REPORT_STALE")
                        checks = _read_tests(report_path, data["test_report"]["format"])
                        test_execution["report_ref"] = _ref(report_path)
                    except (ValueError, KeyError, OSError, ElementTree.ParseError) as error:
                        status, owner = "infrastructure_failed", "validator"
                        observations.append({"check_id": "test_report", "expected": "A fresh non-empty complete test report.",
                                             "observed": {"error": str(error), "exit_code": executed.exit_code}})
                    else:
                        observations.extend({"check_id": row["check_id"], "expected": row["expected"],
                                             "observed": {"actual": row["actual"], "passed": row["passed"]}}
                                            for row in checks if not row["passed"])
                        if observations:
                            status, owner = "behavior_failed", "patch_implementation"
                        elif executed.exit_code != 0:
                            status, owner = "infrastructure_failed", "validator"
                            observations.append({"check_id": "test_command", "expected": 0,
                                                 "observed": {"exit_code": executed.exit_code}})
        if not build.claims_real_build_success:
            observations.append({"check_id": "repository_compilation", "expected": "Declared project compiles and produces its artifacts.",
                                 "observed": {"status": build.status, "diagnostics": [row.to_dict() for row in build.diagnostics],
                                              "build_execution": build_execution}})
        self.profile.verify_fixed_inputs(source, self.fixed_inputs)
        _require(_snapshot(source) == before and self.verification_identity() == identity, "INPUT_CHANGED_DURING_EXECUTION")
        _require(all(_ref(row["resolved_path"])["sha256"] == row["sha256"] for row in build.verified_artifacts),
                 "BUILT_ARTIFACT_CHANGED_DURING_TEST")
        if status != "passed" and build.claims_real_build_success:
            # 为什么要有它：正式入口沿用BuildResult，不能把编译成功交接为未经测试的交付成功。
            build = replace(build, status="failed", claims_real_build_success=False, build_status="verification_failed",
                            message="Compilation completed, but required repository tests did not pass.",
                            repair_handoff={"allowed": owner == "patch_implementation", "failure_evidence_refs": [],
                                            "error_class": "unknown"})
        compiled_paths = {row["path"] for row in compiled}
        compilation_passed = (build_execution.get("build_exit_code") == 0
                              and set(data["allowed_edit_paths"]) <= compiled_paths
                              and build.artifact_verification.get("status") == "verified")
        artifact_refs = [{"path": row["resolved_path"], "sha256": row["sha256"]}
                         for row in build.verified_artifacts]
        code_identity = deepcopy(candidate_context.get("materialized_code_identity")) if candidate_context is not None else None
        v2_binding = {}
        if candidate_context is not None and candidate_context.get("schema_version") == "task_agent_candidate_check_v2":
            # 为什么要有它：v2 的执行原件直接绑定合同及源码身份，不能构造旧技术计划来获得通过。
            v2_binding = {key: deepcopy(candidate_context[key]) for key in (
                "task_contract_ref", "binding_registry_ref", "candidate_ref", "candidate_source_id",
                "public_acceptance_ref", "source_snapshot_sha256")}
        # 为什么要有它：候选交接要核对本候选、编译工件和测试详情；正式Build没有模型候选身份时不能假造同一声明。
        compilation_record = {**v2_binding,
            "status": "passed" if compilation_passed else "failed", "execution_origin": "executed",
            "all_required_files_compiled": set(data["allowed_edit_paths"]) <= compiled_paths,
            "materialized_code_identity": code_identity,
            "built_artifact_ref": artifact_refs[0] if artifact_refs else None, "built_artifact_refs": artifact_refs,
            "source_inputs": before, "compile_inputs": compiled, "build_execution": build_execution,
            "build_profile": profile.to_dict(), "profile_ref": self.profile.reference,
            "input_execution_mode": input_execution_mode,
            "verification_scope": "isolated_candidate_compilation" if candidate_context is not None else "formal_repository_compilation",
        }
        compilation_ref = _save(run / "compilation.json", compilation_record)
        report = {**v2_binding, "status": status, "failure_owner": owner, "execution_origin": "executed",
                  "profile_ref": self.profile.reference, "fixed_inputs": self.fixed_inputs,
                  "source_inputs": before, "source_root": str(source), "verification_identity": identity,
                  "materialized_code_identity": code_identity,
                  "built_artifact_ref": compilation_record["built_artifact_ref"], "built_artifact_refs": artifact_refs,
                  "compilation_report_ref": compilation_ref,
                  "build_result": build.to_artifact_payload(), "build_execution": build_execution,
                  "isolated_build_profile": profile.to_dict(),
                  "compile_inputs": compiled, "test_execution": test_execution, "checks": checks,
                  "observations": observations, "verification_summary": {"checks": len(checks),
                  "failed_checks": sum(not row["passed"] for row in checks),
                  "executions": int(test_execution["status"] == "executed"),
                  "test_command_executed": test_execution["status"] == "executed"},
                  "verification_scope": "declared_repository_build_and_public_tests_no_game_runtime"}
        reference = _save(run / "report.json", report)
        # 为什么要有它：正常Build入口只返回BuildResult时仍可找到实际测试原件；持久化报告不写自引用，返回对象另加引用。
        build = replace(build, artifact_verification={**build.artifact_verification,
            "repository_validation_ref": reference, "repository_profile_ref": self.profile.reference,
            "repository_source_snapshot_sha256": canonical_hash(before),
            "repository_test_report_ref": test_execution.get("report_ref"),
            "repository_test_status": status})
        return {**report, "report_ref": reference, "build_result": build}

    def __call__(self, context):
        if context.get("schema_version") == "task_agent_candidate_check_v2":
            return self._check_v2(context)
        identity = self.verification_identity()
        before = _snapshot(self.source_root)
        changes = context["generated_changes"]
        _require(canonical_hash(changes) == context["generated_changes_sha256"], "CHANGE_HASH_MISMATCH")
        response = context["model_response"]
        _require(_ref(response["path"])["sha256"] == response["output_sha256"], "MODEL_RESPONSE_CHANGED")
        materialized = materialize_candidate(workspace_root=self.workspace_root,
            implementation_plan=context["implementation_plan"], changes=changes)
        _require(materialized_code_identity(materialized) == context["materialized_code_identity"], "CANDIDATE_IDENTITY_CHANGED")
        # 为什么要有它：候选源码路径影响Windows编译器，报告目录长度不应继续叠加到源码和NuGet包路径。
        scratch = self.profile.task_work_root(self.workspace_root) / "c"
        _require(not scratch.resolve().is_relative_to(self.source_root), "CANDIDATE_SCRATCH_INSIDE_SOURCE")
        _require(scratch.resolve().is_relative_to(self.workspace_root), "CANDIDATE_SCRATCH_OUTSIDE_WORKSPACE")
        if scratch.exists():
            _require(not scratch.is_symlink()
                     and not getattr(scratch.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                     "LINKED_CANDIDATE_SCRATCH_UNSUPPORTED")
        scratch.mkdir(parents=True, exist_ok=True)
        isolated = Path(tempfile.mkdtemp(prefix="", dir=scratch)) / "s"
        shutil.copytree(self.source_root, isolated, ignore=shutil.ignore_patterns(*_IGNORED))
        expected = {row["path"]: row["sha256"] for row in before}
        for relative, content in materialized.files.items():
            original = _within(self.workspace_root, relative)
            _require(original.is_relative_to(self.source_root), "EDIT_OUTSIDE_SOURCE")
            selected = original.relative_to(self.source_root).as_posix()
            _require(selected in self.profile.data["allowed_edit_paths"], "EDIT_OUTSIDE_SCOPE")
            _within(isolated, selected).write_bytes(content)
            expected[selected] = hashlib.sha256(content).hexdigest()
        _require({row["path"]: row["sha256"] for row in _snapshot(isolated)} == expected, "CANDIDATE_COPY_CHANGED")
        executed = self.run_build_test(isolated, context["implementation_plan"].get("feature_id", "repository-candidate"),
                                       "preapproval_isolated_compilation", candidate_context=context)
        _require(_snapshot(self.source_root) == before and self.verification_identity() == identity, "ORIGINAL_SOURCE_CHANGED")
        owner = executed["failure_owner"]
        build = executed["build_result"].to_artifact_payload()
        report = {"status": executed["status"], "failure_owner": owner,
                  "owner_binding_ids": [row["edit_binding_id"] for row in changes] if owner == "patch_implementation" else [],
                  "observations": executed["observations"], "execution_origin": "executed",
                  "verification_ref": executed["report_ref"], "verification_summary": executed["verification_summary"],
                  "verification_identity": identity, "verification_scope": executed["verification_scope"],
                  "materialized_code_identity": context["materialized_code_identity"],
                  "implementation_plan_ref": context["implementation_plan_ref"],
                  "model_response_sha256": response["output_sha256"],
                  "generated_changes_sha256": context["generated_changes_sha256"],
                  "source_snapshot_sha256": context["source_snapshot_sha256"],
                  "validation_results": {"full_compilation": {"status": "passed" if executed["build_execution"].get("build_exit_code") == 0 else "failed",
                      "all_required_files_compiled": set(self.profile.data["allowed_edit_paths"]) <= {row["path"] for row in executed["compile_inputs"]},
                      "report_ref": executed["compilation_report_ref"], "build_result": build},
                      "shared_test_result": {"status": executed["status"], "report_ref": executed["report_ref"]},
                      "runtime": {"status": "not_executed"}}}
        if owner == "patch_implementation":
            # 为什么要有它：候选目录每次不同不代表修复进展；原报告保留完整路径，比较投影只消除本次隔离路径。
            def stable(value):
                if isinstance(value, str):
                    return value.replace(str(isolated), "<candidate-source>").replace(isolated.as_posix(), "<candidate-source>")
                if isinstance(value, list):
                    return [stable(item) for item in value]
                if isinstance(value, dict):
                    return {key: stable(item) for key, item in value.items()}
                return value
            report["failure_state_observations"] = stable(executed["observations"])
        return {**report, "report_ref": _save(Path(executed["report_ref"]["path"]).parent / "candidate-report.json", report)}

    def _check_v2(self, context):
        """Check a validated file-level candidate without an implementation Plan.

        为什么要有它：授权和精确源码来自任务状态，技术计划不得成为执行或验收的权限来源。
        """
        from src.mod_agent.task_agent_v2_state import TaskAgentV2State

        state = TaskAgentV2State(context["artifact_store_root"], context["task_contract_ref"])
        candidate = state.validate_candidate(context["candidate_ref"])
        changes = state.changes_from_baseline(context["candidate_ref"])
        _require(context["generated_changes"] == changes, "CHANGE_CONTENT_MISMATCH")
        contract = state.contract
        baseline = state.load(contract["baseline_manifest_ref"])
        _require(context["source_snapshot_sha256"] == canonical_hash(baseline), "SOURCE_SNAPSHOT_MISMATCH")
        registry = state.load(context["binding_registry_ref"])
        _require(registry.get("contract_ref") == state.contract_ref, "REGISTRY_CONTRACT_MISMATCH")
        candidate_registry = state.load(candidate["registry_ref"])
        _require(all(registry["bindings"].get(key) == value for key, value in candidate_registry["bindings"].items()),
                 "CANDIDATE_REGISTRY_MISMATCH")
        _require(candidate["candidate_source_id"] == context["candidate_source_id"], "CANDIDATE_IDENTITY_CHANGED")
        _require(Path(baseline["source_root"]).resolve() == self.source_root, "CONTRACT_SOURCE_MISMATCH")
        _require(contract.get("execution_profile_ref") == self.profile.reference, "CONTRACT_PROFILE_MISMATCH")
        _require(bool(contract.get("public_acceptance_ref")), "PUBLIC_ACCEPTANCE_REQUIRED")
        _require(set(contract["allowed_edit_paths"]) <= set(self.profile.data["allowed_edit_paths"]), "EDIT_OUTSIDE_SCOPE")
        identity = self.verification_identity()
        _require(context["verification_identity"] == identity, "VERIFICATION_IDENTITY_CHANGED")
        acceptance = state.load(contract["public_acceptance_ref"])
        _require(acceptance.get("verification_identity") == identity, "ACCEPTANCE_VERIFIER_MISMATCH")
        before = _snapshot(self.source_root)
        hashes = {row["path"]: row["sha256"] for row in before}
        _require(all(hashes.get(path) == row["sha256"] for path, row in baseline["files"].items()),
                 "BASELINE_SOURCE_CHANGED")
        response = context["model_response"]
        _require(_ref(response["path"])["sha256"] == response["output_sha256"], "MODEL_RESPONSE_CHANGED")
        _require(candidate["model_origin"].get("response_ref") == {
            "path": response["path"], "sha256": response["output_sha256"]}, "CANDIDATE_RESPONSE_MISMATCH")
        # 为什么要有它：隔离副本沿用正式检查器的固定输入及真实报告要求，v2 只改变候选身份来源。
        scratch = self.profile.task_work_root(self.workspace_root) / "c"
        _require(not scratch.resolve().is_relative_to(self.source_root), "CANDIDATE_SCRATCH_INSIDE_SOURCE")
        _require(scratch.resolve().is_relative_to(self.workspace_root), "CANDIDATE_SCRATCH_OUTSIDE_WORKSPACE")
        if scratch.exists():
            _require(not scratch.is_symlink()
                     and not getattr(scratch.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                     "LINKED_CANDIDATE_SCRATCH_UNSUPPORTED")
        scratch.mkdir(parents=True, exist_ok=True)
        isolated = Path(tempfile.mkdtemp(prefix="", dir=scratch)) / "s"
        shutil.copytree(self.source_root, isolated, ignore=shutil.ignore_patterns(*_IGNORED))
        expected = dict(hashes)
        for relative, text in candidate["files"].items():
            _require(relative in contract["allowed_edit_paths"], "EDIT_OUTSIDE_SCOPE")
            content = text.encode("utf-8")
            _within(isolated, relative).write_bytes(content)
            expected[relative] = hashlib.sha256(content).hexdigest()
        _require({row["path"]: row["sha256"] for row in _snapshot(isolated)} == expected, "CANDIDATE_COPY_CHANGED")
        selected = {**context, "public_acceptance_ref": contract["public_acceptance_ref"]}
        executed = self.run_build_test(isolated, context["feature_id"], "preapproval_isolated_compilation",
                                       candidate_context=selected)
        _require(_snapshot(self.source_root) == before and self.verification_identity() == identity, "ORIGINAL_SOURCE_CHANGED")
        owner = executed["failure_owner"]
        binding = {key: selected[key] for key in (
            "task_contract_ref", "binding_registry_ref", "candidate_ref", "candidate_source_id",
            "public_acceptance_ref", "source_snapshot_sha256")}
        report = {**binding, "schema_version": "task_agent_candidate_report_v2",
            "status": executed["status"], "failure_owner": owner,
            "owner_binding_ids": sorted({row["binding_id"] for row in changes})
                if owner == "patch_implementation" else [],
            "observations": executed["observations"], "execution_origin": "executed",
            "verification_ref": executed["report_ref"], "verification_summary": executed["verification_summary"],
            "verification_identity": identity, "verification_scope": executed["verification_scope"],
            "profile_ref": self.profile.reference, "model_response_sha256": response["output_sha256"],
            "generated_changes_sha256": canonical_hash(context["generated_changes"]),
            "validation_results": {
                "full_compilation": {"status": "passed" if executed["build_execution"].get("build_exit_code") == 0 else "failed",
                    "all_required_files_compiled": set(self.profile.data["allowed_edit_paths"]) <= {row["path"] for row in executed["compile_inputs"]},
                    "report_ref": executed["compilation_report_ref"], "build_result": executed["build_result"].to_artifact_payload()},
                "shared_test_result": {"status": executed["status"], "report_ref": executed["report_ref"]},
                "runtime": {"status": "not_executed"}}}
        return {**report, "report_ref": _save(Path(executed["report_ref"]["path"]).parent / "candidate-report.json", report)}


def validate_candidate_report_v2(report_ref, *, context=None, require_pass=None,
                                 task_contract_ref=None, binding_registry_ref=None,
                                 candidate_ref=None, candidate_source_id=None, verification_identity=None,
                                 public_acceptance_ref=None):
    """Recheck persisted candidate, compilation and test evidence for handoff.

    为什么要有它：产品接管需核验同一版本的执行原件，不能仅信任上层保存的 passed 状态。
    """
    if "report_ref" in report_ref:
        report = read_bound_json(report_ref["report_ref"])
        _require(report == {key: value for key, value in report_ref.items() if key != "report_ref"},
                 "REPORT_CONTENT_MISMATCH")
    else:
        report = read_bound_json(report_ref)
    if require_pass is None:
        require_pass = context is None
    if context is not None:
        from src.mod_agent.task_agent_v2_state import TaskAgentV2State

        state = TaskAgentV2State(context["artifact_store_root"], context["task_contract_ref"])
        candidate = state.validate_candidate(context["candidate_ref"])
        changes = state.changes_from_baseline(context["candidate_ref"])
        _require(candidate["candidate_source_id"] == context["candidate_source_id"], "CANDIDATE_IDENTITY_CHANGED")
        registry, original_registry = state.load(context["binding_registry_ref"]), state.load(candidate["registry_ref"])
        _require(registry.get("contract_ref") == state.contract_ref and all(
            registry["bindings"].get(key) == value for key, value in original_registry["bindings"].items()),
            "CANDIDATE_REGISTRY_MISMATCH")
        _require(context["generated_changes"] == changes
                 and report.get("generated_changes_sha256") == canonical_hash(changes), "CHANGE_CONTENT_MISMATCH")
        snapshot = canonical_hash(state.load(state.contract["baseline_manifest_ref"]))
        _require(context["source_snapshot_sha256"] == report.get("source_snapshot_sha256") == snapshot,
                 "SOURCE_SNAPSHOT_MISMATCH")
        response = context["model_response"]
        _require(_ref(response["path"])["sha256"] == response["output_sha256"] == report.get("model_response_sha256")
                 and candidate["model_origin"].get("response_ref") == {
                     "path": response["path"], "sha256": response["output_sha256"]}, "CANDIDATE_RESPONSE_MISMATCH")
        _require(report.get("profile_ref") == state.contract.get("execution_profile_ref"), "CONTRACT_PROFILE_MISMATCH")
        task_contract_ref, binding_registry_ref = context["task_contract_ref"], context["binding_registry_ref"]
        candidate_ref, candidate_source_id = context["candidate_ref"], context["candidate_source_id"]
        verification_identity = context["verification_identity"]
        public_acceptance_ref = state.contract["public_acceptance_ref"]
        _require(state.load(public_acceptance_ref).get("verification_identity") == verification_identity,
                 "ACCEPTANCE_VERIFIER_MISMATCH")
    _require(report.get("schema_version") == "task_agent_candidate_report_v2", "V2_REPORT_REQUIRED")
    expected = {"task_contract_ref": task_contract_ref, "binding_registry_ref": binding_registry_ref,
                "candidate_ref": candidate_ref, "candidate_source_id": candidate_source_id,
                "public_acceptance_ref": public_acceptance_ref}
    _require(all(report.get(key) == value for key, value in expected.items()), "REPORT_CANDIDATE_MISMATCH")
    _require(report.get("verification_identity") == verification_identity, "REPORT_VERIFIER_MISMATCH")
    _require(report.get("status") in {"passed", "behavior_failed", "infrastructure_failed"}
             and report.get("execution_origin") == "executed", "CANDIDATE_CHECK_NOT_EXECUTED")
    if require_pass:
        _require(report.get("status") == "passed" and report.get("failure_owner") is None, "CANDIDATE_CHECK_NOT_PASSED")
    verification = read_bound_json(report["verification_ref"])
    compilation_ref = report["validation_results"]["full_compilation"]["report_ref"]
    compilation = read_bound_json(compilation_ref)
    _require(report["validation_results"]["shared_test_result"] == {
        "status": report["status"], "report_ref": report["verification_ref"]}, "TEST_REPORT_BINDING_MISMATCH")
    _require(verification.get("compilation_report_ref") == compilation_ref, "COMPILATION_REPORT_BINDING_MISMATCH")
    for record in (verification, compilation):
        _require(all(record.get(key) == value for key, value in expected.items()), "EXECUTION_CANDIDATE_MISMATCH")
        _require(record.get("execution_origin") == "executed", "EXECUTION_NOT_EXECUTED")
        _require(record.get("source_snapshot_sha256") == report["source_snapshot_sha256"], "EXECUTION_SNAPSHOT_MISMATCH")
        _require(record.get("profile_ref") == report["profile_ref"], "EXECUTION_PROFILE_MISMATCH")
    _require(verification.get("verification_identity") == verification_identity, "EXECUTION_VERIFIER_MISMATCH")
    _require(verification.get("status") == report["status"]
             and verification.get("failure_owner") == report.get("failure_owner")
             and verification.get("observations") == report.get("observations")
             and verification.get("verification_summary") == report.get("verification_summary"), "EXECUTION_RESULT_MISMATCH")
    if report["status"] != "passed":
        # 为什么要有它：真实失败可以进入下一轮决策，产品交付仍须显式要求成功及完整测试原件。
        return report
    _require(report.get("failure_owner") is None and compilation.get("status") == "passed", "EXECUTION_NOT_PASSED")
    _require(compilation.get("all_required_files_compiled") is True
             and compilation["build_execution"].get("restore_exit_code") == 0
             and compilation["build_execution"].get("build_exit_code") == 0, "COMPILATION_NOT_EXECUTED")
    _require(compilation.get("source_inputs") == verification.get("source_inputs")
             and compilation.get("compile_inputs") == verification.get("compile_inputs"), "COMPILE_INPUTS_MISMATCH")
    profile = CSharpRepositoryProfile(report["profile_ref"])
    source = Path(verification["source_root"]).resolve(strict=True)
    _require(_snapshot(source) == verification["source_inputs"], "CHECKED_SOURCE_CHANGED")
    profile.verify_fixed_inputs(source, verification["fixed_inputs"])
    source_hashes = {row["path"]: row["sha256"] for row in verification["source_inputs"]}
    # 为什么要有它：MSBuild生成的obj源码不在原始快照中，但仍须按实际编译文件核验哈希，不能误拒成功或放过被改写的生成输入。
    _require(set(profile.data["allowed_edit_paths"]) <= {row["path"] for row in compilation["compile_inputs"]}
             and all((source_hashes.get(row["path"]) == row["sha256"]
                      or row["path"] not in source_hashes and "obj" in PurePosixPath(row["path"]).parts)
                     and _ref(_within(source, row["path"]))["sha256"] == row["sha256"]
                     for row in compilation["compile_inputs"]),
             "COMPILED_SOURCE_MISMATCH")
    artifacts = compilation.get("built_artifact_refs")
    _require(bool(artifacts) and artifacts == verification.get("built_artifact_refs")
             and all(_ref(row["path"]) == row for row in artifacts), "BUILT_ARTIFACT_CHANGED")
    test_execution = verification["test_execution"]
    _require(test_execution.get("status") == "executed" and test_execution.get("exit_code") == 0,
             "TESTS_NOT_EXECUTED")
    test_ref = test_execution["report_ref"]
    _require(_ref(test_ref["path"]) == test_ref, "TEST_REPORT_CHANGED")
    tests = _read_tests(Path(test_ref["path"]), profile.data["test_report"]["format"])
    _require(tests == verification["checks"] and all(row["passed"] for row in tests), "TEST_RESULTS_MISMATCH")
    summary = {"checks": len(tests), "failed_checks": 0, "executions": 1, "test_command_executed": True}
    _require(report["verification_summary"] == verification["verification_summary"] == summary,
             "TEST_SUMMARY_MISMATCH")
    return report


def make_csharp_repository_evaluator(**kwargs):
    return CSharpRepositoryBackend(**kwargs)
