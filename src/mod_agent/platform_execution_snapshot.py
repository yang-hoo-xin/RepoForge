"""Local provenance for a fresh Python process executing frozen Product sources.

No workflow or model inputs are defined here. Site packages and native libraries
remain outside the source snapshot scope. 为什么要有它：recoverable source bytes
must be distinguished from a complete environment or a successful business run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import sysconfig
import traceback
import zipfile
from pathlib import Path

_ACTIVE = None


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _local_absolute(value):
    path = Path(value)
    if not path.is_absolute() or str(value).startswith(("\\\\", "//")) or path.is_symlink():
        raise ValueError("FROZEN_EXECUTION_REQUIRES_LOCAL_ABSOLUTE_PATH")
    return path.resolve()


def _write_once(path, value):
    raw = _bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path.resolve()), "sha256": _digest(raw)}


def prepare_snapshot(*, source_root, snapshot_root, business_argv, run_root=None,
                     extra_load_refs=(), bound_inputs=(), resource_dirs=("config",)):
    """Copy selected source/resources once; never modify selected Product inputs."""
    source, target = _local_absolute(source_root), _local_absolute(snapshot_root)
    selected = _local_absolute(run_root) if run_root is not None else None
    if selected is not None and selected.exists():
        raise ValueError("FROZEN_EXECUTION_SELECTED_RUN_ALREADY_EXISTS")
    directories = tuple(dict.fromkeys(("src", "scripts", *resource_dirs)))
    if target.exists() or target == source or any(target.is_relative_to(source / part) for part in directories):
        raise ValueError("FROZEN_EXECUTION_SNAPSHOT_DESTINATION_INVALID")
    if selected is not None and (target.is_relative_to(selected) or selected.is_relative_to(target)):
        raise ValueError("FROZEN_EXECUTION_SOURCE_AND_PRODUCT_ROOTS_MUST_BE_SEPARATE")
    inputs = []
    for role, references in (("extra_python_load", extra_load_refs), ("bound_input", bound_inputs)):
        for raw_path, expected in references:
            path = _local_absolute(raw_path)
            raw = path.read_bytes()
            if _digest(raw) != expected:
                raise ValueError("FROZEN_EXECUTION_EXTERNAL_REFERENCE_CHANGED")
            inputs.append({"role": role, "path": str(path), "sha256": expected,
                           "retained_path": "external/" + expected + "/" + path.name,
                           "raw": raw})
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for directory in directories:
        relative = Path(directory)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("FROZEN_EXECUTION_RESOURCE_SCOPE_INVALID")
        selected_directory = source / relative
        if not selected_directory.is_dir():
            raise ValueError("FROZEN_EXECUTION_REQUIRED_RESOURCE_MISSING: " + directory)
        for path in selected_directory.rglob("*"):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError("FROZEN_EXECUTION_SOURCE_LINK_NOT_ALLOWED")
            if not path.is_file():
                continue
            name = path.relative_to(source).as_posix()
            raw = path.read_bytes()
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(raw)
            files[name] = {"sha256": _digest(raw), "size": len(raw)}
    for item in inputs:
        destination = target / item["retained_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as stream:
                stream.write(item["raw"])
        files[item["retained_path"]] = {"sha256": item["sha256"], "size": len(item.pop("raw"))}
    archive_path = target / "platform_sources.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.write(target / name, name)
    document = {"schema_version": "frozen_product_execution_v1", "source_root": str(source),
        "snapshot_root": str(target), "selected_run_root": str(selected) if selected else None,
        "business_argv": list(business_argv), "working_directory": str(Path.cwd()),
        "python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
        "files": files, "external_refs": inputs,
        "archive_ref": {"path": str(archive_path), "sha256": _digest(archive_path.read_bytes())},
        "scope": "project_src_scripts_resources_and_explicit_external_refs_not_site_packages_or_native_libraries",
        "model_source_scope": "unchanged_original_cli_target_project_only"}
    return _write_once(target / "execution_snapshot.json", document)


def read_snapshot(reference):
    path = _local_absolute(reference["path"])
    raw = path.read_bytes()
    if _digest(raw) != reference["sha256"]:
        raise ValueError("FROZEN_EXECUTION_MANIFEST_CHANGED")
    document = json.loads(raw)
    root = _local_absolute(document["snapshot_root"])
    if path.parent != root or document.get("schema_version") != "frozen_product_execution_v1":
        raise ValueError("FROZEN_EXECUTION_MANIFEST_SCOPE_INVALID")
    return document


class FrozenExecution:
    """Observe and validate source actually executed by the fresh child process."""

    def __init__(self, reference):
        self.reference = dict(reference)
        self.document = read_snapshot(reference)
        self.root = Path(self.document["snapshot_root"])
        self.original = Path(self.document["source_root"])
        self.executed = {}
        self.failures = []
        self._compiling = False
        self._codes = {}
        self.selected_inputs = {}
        self._checking_inputs = False
        self._observing_inputs = False
        self._extra = {str(Path(row["path"]).resolve()): row for row in self.document["external_refs"]
                       if row["role"] == "extra_python_load"}
        self._runtime_roots = tuple(Path(sysconfig.get_path(name)).resolve()
                                    for name in ("stdlib", "purelib", "platlib"))

    def verify_files(self):
        previous = self._checking_inputs
        self._checking_inputs = True
        try:
            self._verify_files()
        finally:
            self._checking_inputs = previous

    def _verify_files(self):
        for name, expected in self.document["files"].items():
            path = self.root / name
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError("FROZEN_EXECUTION_SNAPSHOT_PATH_ESCAPE")
            if _digest(path.read_bytes()) != expected["sha256"]:
                raise ValueError("FROZEN_EXECUTION_SOURCE_CHANGED: " + name)
        if any(self.root.rglob("*.pyc")):
            raise ValueError("FROZEN_EXECUTION_BYTECODE_CACHE_NOT_ALLOWED")
        archive = self.document["archive_ref"]
        if _digest(Path(archive["path"]).read_bytes()) != archive["sha256"]:
            raise ValueError("FROZEN_EXECUTION_ARCHIVE_CHANGED")
        for row in self.document["external_refs"]:
            if _digest(Path(row["path"]).read_bytes()) != row["sha256"]:
                raise ValueError("FROZEN_EXECUTION_EXTERNAL_REFERENCE_CHANGED")

    def verify_loaded_modules(self):
        result = []
        for name, module in tuple(sys.modules.items()):
            if name not in {"src", "scripts"} and not name.startswith(("src.", "scripts.")):
                continue
            location = getattr(module, "__file__", None)
            if not location:
                continue
            path = Path(location).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("FROZEN_EXECUTION_LOADED_OUTSIDE_SNAPSHOT: " + name)
            relative = path.relative_to(self.root).as_posix()
            expected = self.document["files"].get(relative)
            if expected is None or _digest(path.read_bytes()) != expected["sha256"]:
                raise ValueError("FROZEN_EXECUTION_LOADED_SOURCE_CHANGED: " + name)
            result.append({"module": name, "relative_path": relative, "sha256": expected["sha256"]})
        return result

    def _source_for_code(self, filename):
        if not filename or filename.startswith("<") or not Path(filename).is_absolute():
            return None
        path = Path(filename).resolve()
        if path.is_relative_to(self.root):
            relative = path.relative_to(self.root).as_posix()
            expected = self.document["files"].get(relative)
            if expected is None:
                raise ValueError("FROZEN_EXECUTION_UNBOUND_SNAPSHOT_CODE")
            return path, expected["sha256"], "snapshot"
        if str(path) in self._extra:
            return path, self._extra[str(path)]["sha256"], "explicit_external"
        if any(path.is_relative_to(self.original / part) for part in ("src", "scripts", "tests")):
            raise ValueError("FROZEN_EXECUTION_LOADED_OUTSIDE_SNAPSHOT: " + str(path))
        if any(path.is_relative_to(root) for root in self._runtime_roots):
            return None  # Interpreter/site-package sources are explicitly out of scope.
        raise ValueError("FROZEN_EXECUTION_UNBOUND_EXTERNAL_PYTHON_LOAD: " + str(path))

    def audit(self, event, args):
        if event == "open":
            self._observe_input_open(args)
            return
        if event != "exec" or self._compiling:
            return
        try:
            code = args[0]
            selected = self._source_for_code(code.co_filename)
            if selected is None:
                if not Path(code.co_filename).is_absolute():
                    caller = sys._getframe(1).f_code.co_filename
                    if self._source_for_code(caller) is not None:
                        # Runtime-library generation remains out of scope; an
                        # unbound exec directly from project code stays unknown.
                        # 为什么要有它：相对文件名不能让主树动态代码冒充已冻结版本。
                        self.failures.append({"event": "exec", "reason": "unbound_dynamic_exec_from_project",
                                              "code_filename": code.co_filename, "caller": caller})
                return
            path, expected, scope = selected
            raw = path.read_bytes()
            if _digest(raw) != expected:
                raise ValueError("FROZEN_EXECUTION_EXECUTED_SOURCE_CHANGED: " + str(path))
            key = (str(path), expected)
            if key not in self._codes:
                self._compiling = True
                try:
                    self._codes[key] = compile(raw, code.co_filename, "exec", dont_inherit=True,
                                               optimize=sys.flags.optimize)
                finally:
                    self._compiling = False
            if code != self._codes[key]:
                raise ValueError("FROZEN_EXECUTION_CODE_DIFFERS_FROM_RETAINED_SOURCE: " + str(path))
            self.executed[str(path)] = {"path": str(path), "sha256": expected, "scope": scope}
        except Exception as error:
            self.failures.append({"event": "exec", "reason": str(error)})
            raise

    def bind_cli_inputs(self, parsed):
        """Bind the original run-request's configuration paths, without rewriting them."""
        external = {row["path"]: row for row in self.document["external_refs"] if row["role"] == "bound_input"}
        for field in ("settings", "patch_worker_config", "stage_model_call_limits", "execution_config", "recovery_config",
                      "game_folder_build_binding", "external_patch_profile", "dependency_api_config", "knowledge_evidence_config",
                      "repository_profile", "task_agent_experience_config"):
            value = getattr(parsed, field, None)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                anchor = self.root if field == "settings" else Path(self.document["working_directory"])
                path = anchor / path
            if path.is_symlink():
                self.failures.append({"event": "cli_input_binding", "reason": "CLI_INPUT_LINK_NOT_ALLOWED", "field": field})
                raise ValueError("FROZEN_EXECUTION_CLI_INPUT_LINK_NOT_ALLOWED: " + field)
            path = path.resolve()
            if path.is_relative_to(self.root):
                name = path.relative_to(self.root).as_posix()
                expected = self.document["files"].get(name)
                retained = str(path)
                scope = "frozen_resource"
            else:
                expected = external.get(str(path))
                retained = str(self.root / expected["retained_path"]) if expected else None
                scope = "explicit_bound_input"
            if expected is None:
                self.failures.append({"event": "cli_input_binding", "reason": "UNBOUND_CLI_INPUT", "field": field})
                raise ValueError("FROZEN_EXECUTION_UNBOUND_CLI_INPUT: " + field)
            if _digest(path.read_bytes()) != expected["sha256"]:
                self.failures.append({"event": "cli_input_binding", "reason": "CLI_INPUT_CHANGED", "field": field})
                raise ValueError("FROZEN_EXECUTION_CLI_INPUT_CHANGED: " + field)
            self.selected_inputs[str(path)] = {"field": field, "path": str(path),
                "sha256": expected["sha256"], "retained_path": retained, "scope": scope,
                "read_open_count": 0,
                "verification_boundary": "before_each_observed_read_open_and_entry_exit_scope_checks"}

    def _observe_input_open(self, args):
        if not self._observing_inputs or self._checking_inputs or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        selected = self.selected_inputs.get(str(path))
        mode = args[1]
        if selected is None or (isinstance(mode, str) and "r" not in mode and "+" not in mode):
            return
        self._checking_inputs = True
        try:
            digest = _digest(path.read_bytes())
            if digest != selected["sha256"]:
                raise ValueError("FROZEN_EXECUTION_CLI_INPUT_CHANGED_BEFORE_READ: " + selected["field"])
            selected["read_open_count"] += 1
        except Exception as error:
            self.failures.append({"event": "input_open", "reason": str(error), "field": selected["field"]})
            raise
        finally:
            self._checking_inputs = False

    def activate(self):
        global _ACTIVE
        self.verify_files()
        self.verify_loaded_modules()
        _ACTIVE = self
        sys.addaudithook(self.audit)
        # 为什么要有它：可信父进程给出的共享构建额度须在正常CLI启动编译器前生效，不改编译器或验收语义。
        if os.environ.get("REPOFORGE_BUILD_BUDGET_REF"):
            from src.mod_agent.build.execution_budget import install_build_budget_from_environment
            install_build_budget_from_environment()

    def check(self):
        try:
            self.verify_files()
            loaded = self.verify_loaded_modules()
            status = "verified" if not self.failures else "unknown"
            reason = None
        except Exception as error:
            loaded, status, reason = [], "unknown", str(error)
        return {"status": status, "reason": reason, "execution_version_ref": self.reference,
                "python_executable": str(Path(sys.executable).resolve()), "python_version": sys.version,
                "process_id": os.getpid(), "loaded_project_modules": loaded,
                "selected_cli_inputs": list(self.selected_inputs.values()),
                "executed_source_files": list(self.executed.values()), "failures": list(self.failures),
                "external_bindings": [{"path": row["path"], "sha256": row["sha256"],
                    "retained_path": str(self.root / row["retained_path"]), "role": row["role"],
                    "execution_observed": row["path"] in self.executed}
                    for row in self.document["external_refs"]],
                "scope": self.document["scope"]}

    def record_run_check(self, run_root, run_id, request_id, trace_id, phase):
        selected = self.document["selected_run_root"]
        if selected is None or Path(run_root).resolve() != Path(selected):
            raise ValueError("FROZEN_EXECUTION_UNSELECTED_RUN_ROOT")
        record = {**self.check(), "run_id": run_id, "request_id": request_id,
                  "trace_id": trace_id, "phase": phase}
        reference = _write_once(self.root / "execution_checks" / (trace_id + "." + phase + ".json"), record)
        return {"execution_version_ref": self.reference, "execution_version_check_ref": reference,
                "execution_version_status": record["status"]}


def current_frozen_execution():
    return _ACTIVE


def validate_cli_selection(parsed, document):
    """Check provenance selection after the original parser validates CLI syntax."""
    if parsed.command == "continue-pretransport-patch":
        from src.mod_agent.pretransport_patch_admission import validate_recovery_config
        config = json.loads(Path(parsed.recovery_config).read_bytes())
        validate_recovery_config(config)
        parent = Path(parsed.run_root).resolve(strict=True)
        expected = parent / "continuations" / ("pt-" + config["selection"]["source_attempt_ref"]["sha256"][:12])
        if (parsed.run_id != parent.name or not document["selected_run_root"]
                or expected != Path(document["selected_run_root"]).resolve()
                or expected.exists() or parsed.prepare_only):
            raise ValueError("FROZEN_EXECUTION_PRETRANSPORT_SEGMENT_SELECTION_INVALID")
        return
    if parsed.command != "run-request":
        raise ValueError("FROZEN_EXECUTION_SUPPORTS_NEW_RUN_REQUEST_OR_HELP")
    if not parsed.run_id or not document["selected_run_root"]:
        raise ValueError("FROZEN_EXECUTION_REQUIRES_EXPLICIT_RUN_SELECTION")
    actual = (Path(parsed.workspace_root) / parsed.run_id).resolve()
    if actual != Path(document["selected_run_root"]):
        raise ValueError("FROZEN_EXECUTION_UNSELECTED_RUN_ROOT")
    if actual.exists():
        raise ValueError("FROZEN_EXECUTION_SELECTED_RUN_ALREADY_EXISTS")
    target = getattr(parsed, "target_project", None)
    if target is not None:
        target, frozen = Path(target).resolve(), Path(document["snapshot_root"])
        if target.is_relative_to(frozen) or frozen.is_relative_to(target):
            raise ValueError("FROZEN_EXECUTION_PLATFORM_IS_NOT_TASK_SOURCE")


def child_main(reference):
    execution = FrozenExecution(reference)
    execution.activate()
    code = 2
    try:
        from src.mod_agent import cli

        execution.verify_loaded_modules()
        args = execution.document["business_argv"]
        parsed = cli.build_parser().parse_args(args)
        validate_cli_selection(parsed, execution.document)
        execution.bind_cli_inputs(parsed)
        try:
            execution._observing_inputs = True
            code = cli.main(args)
        except Exception:
            # Preserve the original Python entry's failure exit status and traceback.
            # 为什么要有它：来源记录不能把业务异常改写为启动参数拒绝或成功。
            code = 1
            traceback.print_exc()
        finally:
            execution._observing_inputs = False
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 1
    finally:
        try:
            _write_once(execution.root / "child_execution.json", {**execution.check(), "exit_code": code})
        except Exception as error:
            print(json.dumps({"execution_record_status": "unknown", "error_type": type(error).__name__}),
                  file=sys.stderr)
    return code
