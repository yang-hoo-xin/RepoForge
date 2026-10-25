"""Version-bound exploration with registered tests and separately submitted probes.

为什么要有它：模型可以选择判别实验，但探索结果不授予正式验收或交付权限。
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError, _no_links, _path

_PROBE_PROFILE = {
    "schema": "task-agent-probe-profile.v2", "entrypoints": {"csharp_console": {"source_prefix": "probe/", "extensions": [".cs"]}},
    "max_files": 8, "max_file_bytes": 32768, "max_total_bytes": 65536}


class ExperimentRejectedError(ValueError):
    pass


class ExperimentOutcomeUnknownError(RuntimeError):
    pass


def _require(condition, code):
    if not condition:
        raise ExperimentRejectedError(code)


def probe_submission_contract():
    """Describe the normal host-owned artifact contract without creating or executing it."""
    return {"entrypoint_id": "csharp_console", "storage_prefix": "probe/",
        "path_examples": ["Program.cs", "checks/Compare.cs", "probe/Program.cs"],
        "path_rule": "Relative POSIX .cs filenames; the host adds probe/ when absent. Existing probe/ paths remain valid.",
        "denied_paths": "Absolute paths, backslashes, traversal, dot/empty components, hidden/reserved names, non-.cs files and case-insensitive normalized duplicates.",
        **{key: _PROBE_PROFILE[key] for key in ("max_files", "max_file_bytes", "max_total_bytes")},
        "candidate_modified": False, "submission_executes_code": False}


def normalize_probe_path(value):
    """Map a logical filename to the private artifact subtree, without touching source.

    为什么要有它：内部目录前缀由宿主管理；规范化不能把越界路径或构建脚本变成可执行输入。
    """
    try:
        path = _path(value)
    except TaskAgentV2StateError as error:
        raise ExperimentRejectedError("PROBE_PATH_DENIED") from error
    _require(Path(path).suffix == ".cs" and not re.search(r'[<>"|?*\x00-\x1f\x7f]', path)
        and not any(part.startswith(".") or part.endswith(".") or part != part.strip() or re.fullmatch(
            r"(?:CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", part, re.I)
            for part in path.split("/")), "PROBE_PATH_DENIED")
    return path if path.startswith("probe/") else "probe/" + path


def _file_ref(path):
    path = Path(path)
    _no_links(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _source(state, revision):
    ref = state.contract["baseline_manifest_ref"] if revision in {"baseline", "B0"} else revision
    files = state.materialize(ref)
    if ref == state.contract["baseline_manifest_ref"]:
        identity = canonical_hash({"baseline_manifest_ref": ref,
            "files": {path: hashlib.sha256(text.encode("utf-8")).hexdigest() for path, text in sorted(files.items())}})
    else:
        identity = state.validate_candidate(ref)["candidate_source_id"]
    return ref, identity, files


def _method_groups(checks):
    groups = {}
    unresolved = []
    for check in checks:
        definitions = check.get("test_definitions", [])
        names = {definition.get("class_name", "") + "." + definition.get("method_name", "")
            for definition in definitions if isinstance(definition.get("class_name"), str)
            and isinstance(definition.get("method_name"), str)}
        if len(names) != 1:
            unresolved.append(check["check_id"])
            continue
        name = next(iter(names))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.+`]*", name) or name.endswith("."):
            unresolved.append(check["check_id"])
            continue
        groups.setdefault(name, []).append(check["check_id"])
    return groups, unresolved


class CSharpRegisteredTestRunner:
    """Run host-derived dotnet filters with the existing build/report implementation.

    为什么要有它：选择已有测试复用实际执行后端，不接受任意命令或改变正式检查器身份。
    """

    def __init__(self, backend):
        self.backend = backend

    def identity(self):
        return {"kind": "registered_csharp_tests", "backend": self.backend.verification_identity(),
            "owner_sha256": _file_ref(__file__)["sha256"], "profile_ref": self.backend.profile.reference}

    def read_public_report(self, report_ref):
        from src.mod_agent.domain.csharp_repository_backend import read_repository_tests

        path = Path(report_ref["path"])
        _no_links(path)
        _require(path.resolve().is_relative_to(self.backend.output_root.resolve()), "REPORT_OUTSIDE_REGISTERED_RUNNER")
        report = read_bound_json(report_ref)
        _require(report.get("execution_origin") == "executed"
            and report.get("profile_ref") == self.backend.profile.reference
            and report.get("verification_identity") == self.backend.verification_identity()
            and report.get("fixed_inputs") == self.backend.fixed_inputs, "CATALOG_RUNNER_IDENTITY_MISMATCH")
        execution = report.get("test_execution", {})
        _require(execution.get("status") == "executed" and execution.get("report_ref"), "TESTS_NOT_EXECUTED")
        raw = execution["report_ref"]
        _require(Path(raw["path"]).resolve().is_relative_to(path.parent.resolve())
            and _file_ref(raw["path"]) == raw, "CATALOG_TEST_REPORT_CHANGED")
        tests, _ = read_repository_tests(Path(raw["path"]), self.backend.profile.data, report=report)
        _require(bool(tests) and tests == report.get("checks"), "CATALOG_TEST_RESULTS_CHANGED")
        supported = self.backend.profile.data["test_report"]["format"] == "trx" and self.backend.profile.data["test_command"][0] == "test"
        return {"checks": tests, "report_ref": report_ref, "raw_test_report_ref": raw,
            "selection_available": supported, "scope": "method_group_all_parameter_cases",
            "runner_identity": self.identity()}

    def run_selected(self, *, state, source_revision_ref, catalog, selected, operation_id):
        from src.mod_agent.domain.csharp_repository_backend import (
            _IGNORED,
            CSharpRepositoryBackend,
            _snapshot,
            _within,
        )

        identity = self.identity()
        _require(identity == catalog["runner_identity"], "RUNNER_CHANGED")
        source_ref, source_id, files = _source(state, source_revision_ref)
        baseline = state.load(state.contract["baseline_manifest_ref"])
        _require(Path(baseline["source_root"]).resolve() == self.backend.source_root
            and state.contract.get("execution_profile_ref") == self.backend.profile.reference,
            "SOURCE_RUNNER_MISMATCH")
        before = _snapshot(self.backend.source_root)
        actual_hashes = {row["path"]: row["sha256"] for row in before}
        _require(all(actual_hashes.get(path) == row["sha256"] for path, row in baseline["files"].items()), "BASELINE_SOURCE_CHANGED")
        original = self.backend.profile.data
        command = list(original["test_command"])
        _require(command[0] == "test" and original["test_report"]["format"] == "trx", "TEST_SELECTION_UNAVAILABLE")
        split = command.index("--") if "--" in command else len(command)
        selected_filter = "|".join("FullyQualifiedName=" + row["method_group"] for row in selected)
        indices = [i for i, arg in enumerate(command[:split]) if arg == "--filter"]
        _require(len(indices) <= 1 and not any(arg.startswith("--filter=") for arg in command[:split]), "UNSUPPORTED_EXISTING_FILTER")
        if indices:
            index = indices[0]
            _require(index + 1 < split, "INVALID_EXISTING_FILTER")
            command[index + 1] = "(" + command[index + 1] + ")&(" + selected_filter + ")"
        else:
            command[split:split] = ["--filter", selected_filter]
        scratch = self.backend.profile.task_work_root(self.backend.workspace_root) / "experiments"
        _no_links(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="", dir=scratch))
        isolated = run / "s"
        shutil.copytree(self.backend.source_root, isolated, ignore=shutil.ignore_patterns(*_IGNORED))
        expected = dict(actual_hashes)
        for relative, text in files.items():
            if text != baseline["files"][relative]["text"]:
                _require(relative in state.contract["allowed_edit_paths"], "EDIT_OUTSIDE_SCOPE")
                _within(isolated, relative).write_bytes(text.encode("utf-8"))
                expected[relative] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        _require({row["path"]: row["sha256"] for row in _snapshot(isolated)} == expected, "EXPERIMENT_SOURCE_COPY_CHANGED")
        profile_ref = state.save("experiment-profile", {"schema": "task-agent-experiment-profile.v2",
            "original_profile_ref": self.backend.profile.reference, "operation_id": operation_id,
            "source_revision_ref": source_ref, "source_identity": source_id,
            "selection": selected, "derived_profile": {**original, "test_command": command}})
        profile_path = run / "profile.json"
        import json
        profile_path.write_bytes(json.dumps({**original, "test_command": command}, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        selected_backend = CSharpRepositoryBackend(profile_ref=_file_ref(profile_path), source_root=isolated,
            workspace_root=self.backend.workspace_root, output_root=run / "reports")
        result = selected_backend.run_build_test(isolated, state.contract["task_id"], "preapproval_isolated_compilation")
        _require(_snapshot(self.backend.source_root) == before and self.identity() == identity, "ORIGINAL_RUNNER_INPUT_CHANGED")
        checks = result["checks"]
        groups, unresolved = _method_groups(checks)
        selected_groups = {row["method_group"] for row in selected}
        complete = bool(checks) and not unresolved and set(groups) == selected_groups
        logs_ref = state.save("experiment-logs", {"schema": "task-agent-experiment-logs.v2",
            "test_execution": result["test_execution"], "build_execution": result["build_execution"],
            "verification_ref": result["report_ref"]})
        return {"source_revision_ref": source_ref, "source_identity": source_id, "runner_identity": identity,
            "execution_origin": "executed", "execution_kind": "registered_tests",
            "exit_code": result["test_execution"].get("exit_code", -1),
            "tests_executed": len(checks), "executed_method_groups": sorted(groups),
            "selected_method_groups": sorted(selected_groups), "selection_complete": complete,
            "passed": complete and result["status"] == "passed", "observations": result["observations"],
            "logs_ref": logs_ref, "verification_ref": result["report_ref"], "derived_profile_ref": profile_ref}


class TaskAgentExperiments:
    def __init__(self, *, state, test_runner=None, probe_profile_ref=None, probe_runner=None):
        self.state, self.test_runner, self.probe_runner = state, test_runner, probe_runner
        self.probe_profile_ref = probe_profile_ref or state.save("probe-profile", deepcopy(_PROBE_PROFILE))

    def test_catalog(self, verification_ref):
        _require(self.test_runner is not None, "TEST_RUNNER_UNAVAILABLE")
        report = self.test_runner.read_public_report(verification_ref)
        groups, unresolved = _method_groups(report["checks"])
        _require(report["checks"], "TESTS_NOT_EXECUTED")
        entries = [{"test_id": "method-" + canonical_hash({"runner": report["runner_identity"], "method": method}),
            "method_group": method, "observed_cases": len(checks), "source_check_ids": checks,
            "selection_scope": "all_parameter_cases_in_method_group"} for method, checks in sorted(groups.items())]
        payload = {"schema": "task-agent-test-catalog.v2", "task_contract_ref": self.state.contract_ref,
            "runner_identity": report["runner_identity"], "verification_ref": verification_ref,
            "raw_test_report_ref": report["raw_test_report_ref"], "entries": entries,
            "selection_available": bool(report["selection_available"] and entries), "unselectable_check_ids": unresolved}
        return {**payload, "catalog_ref": self.state.save("test-catalog", payload)}

    def registered_tests_run(self, *, source_revision_ref, catalog_ref, test_ids, operation_id, model_origin=None):
        try:
            catalog = self.state.load(catalog_ref)
        except FileNotFoundError as error:
            # 为什么要有它：目录引用不存在时尚未启动测试，应返回实际未执行的拒绝，不伪造未知副作用。
            raise ExperimentRejectedError("CATALOG_REFERENCE_NOT_FOUND: obtain catalog_ref and test_ids from test_catalog; do not invent them") from error
        _require(catalog.get("schema") == "task-agent-test-catalog.v2"
            and catalog.get("task_contract_ref") == self.state.contract_ref, "CATALOG_TASK_MISMATCH")
        _require(self.test_runner is not None and catalog["runner_identity"] == self.test_runner.identity(), "RUNNER_CHANGED")
        _require(catalog["selection_available"], "TEST_SELECTION_UNAVAILABLE")
        _require(isinstance(test_ids, list) and test_ids and len(test_ids) == len(set(test_ids)), "TEST_SELECTION_REQUIRED")
        entries = {row["test_id"]: row for row in catalog["entries"]}
        _require(all(test_id in entries for test_id in test_ids), "UNREGISTERED_TEST_ID")
        selected = [entries[test_id] for test_id in test_ids]
        ref, identity, _files = _source(self.state, source_revision_ref)
        request = {"kind": "registered_tests", "source_revision_ref": ref, "source_identity": identity,
            "catalog_ref": catalog_ref, "test_ids": test_ids, "runner_identity": catalog["runner_identity"],
            "model_origin": deepcopy(model_origin or {})}
        self.test_runner.read_public_report(catalog["verification_ref"])

        def run():
            output = self.test_runner.run_selected(state=self.state, source_revision_ref=ref,
                catalog=catalog, selected=selected, operation_id=operation_id)
            if not (output.get("tests_executed", 0) > 0
                    and set(output.get("executed_method_groups", [])) == {row["method_group"] for row in selected}):
                output = {**output, "passed": False, "selection_complete": False,
                    "failure_code": "SELECTED_TESTS_NOT_EXECUTED"}
            return output

        return self._execute(operation_id, request, run)

    def probe_submit(self, *, source_revision_ref, files, entrypoint_id, rationale, expected_observation, model_origin=None):
        profile = self.state.load(self.probe_profile_ref)
        _require(profile.get("schema") == "task-agent-probe-profile.v2", "PROBE_PROFILE_INVALID")
        entry = profile["entrypoints"].get(entrypoint_id)
        _require(entry is not None, "PROBE_ENTRYPOINT_DENIED")
        _require(entry.get("source_prefix") == "probe/" and entry.get("extensions") == [".cs"], "PROBE_PROFILE_INVALID")
        _require(isinstance(rationale, str) and bool(rationale.strip()) and isinstance(expected_observation, str)
            and bool(expected_observation.strip()), "PROBE_HYPOTHESIS_REQUIRED")
        _require(isinstance(files, list) and 0 < len(files) <= profile["max_files"], "PROBE_FILES_INVALID")
        rows, mapping, seen, total = [], [], set(), 0
        for row in files:
            _require(isinstance(row, dict) and set(row) == {"relative_path", "content"}
                and isinstance(row["content"], str), "PROBE_FILE_INVALID")
            path = normalize_probe_path(row["relative_path"])
            _require(path.casefold() not in seen, "PROBE_DUPLICATE_PATH")
            seen.add(path.casefold())
            size = len(row["content"].encode("utf-8"))
            _require(size <= profile["max_file_bytes"], "PROBE_FILE_TOO_LARGE")
            total += size
            rows.append({"relative_path": path, "content": row["content"], "sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest()})
            mapping.append({"submitted_path": row["relative_path"], "artifact_path": path})
        _require(total <= profile["max_total_bytes"], "PROBE_SUBMISSION_TOO_LARGE")
        ref, identity, _files = _source(self.state, source_revision_ref)
        payload = {"schema": "task-agent-probe.v2", "task_contract_ref": self.state.contract_ref,
            "source_revision_ref": ref, "source_identity": identity, "profile_ref": self.probe_profile_ref,
            "entrypoint_id": entrypoint_id, "files": rows, "path_mapping": mapping, "rationale": rationale,
            "expected_observation": expected_observation, "model_origin": deepcopy(model_origin or {})}
        return {"probe_artifact_ref": self.state.save("probe", payload), "source_revision_ref": ref,
            "source_identity": identity, "path_mapping": deepcopy(mapping), "executed": False}

    def experiment_run(self, *, source_revision_ref, probe_artifact_ref, operation_id, model_origin=None):
        probe = self.state.load(probe_artifact_ref)
        ref, identity, _files = _source(self.state, source_revision_ref)
        _require(probe.get("schema") == "task-agent-probe.v2" and probe.get("task_contract_ref") == self.state.contract_ref
            and probe.get("profile_ref") == self.probe_profile_ref, "PROBE_CONTRACT_CHANGED")
        _require(probe["source_revision_ref"] == ref and probe["source_identity"] == identity, "PROBE_REVISION_CHANGED")
        request = {"kind": "probe", "source_revision_ref": ref, "source_identity": identity,
            "probe_artifact_ref": probe_artifact_ref, "profile_ref": self.probe_profile_ref,
            "model_origin": deepcopy(model_origin or {})}
        replay = self._replay_probe_operation(operation_id, request)
        if replay is not None:
            return replay
        capability = self.probe_runner.verify_capability(self.probe_profile_ref) if self.probe_runner is not None else None
        qualified = (isinstance(capability, dict) and capability.get("status") == "qualified"
            and capability.get("profile_ref") == self.probe_profile_ref and capability.get("evidence_refs")
            and capability.get("execution_kind") in {"isolated", "controlled"}
            and capability.get("boundaries") == {"filesystem": "isolated", "network": "denied", "process": "isolated"})
        if qualified:
            for reference in capability["evidence_refs"]:
                evidence = self.state.load(reference) if isinstance(reference, str) else read_bound_json(reference)
                qualified = qualified and all(evidence.get(key) == capability[key]
                    for key in ("status", "profile_ref", "runner_identity", "execution_kind", "boundaries"))
        request["runner_identity"] = capability.get("runner_identity") if qualified else None
        if not qualified:
            unavailable = self._execute(operation_id, {**request, "capability": deepcopy(capability)}, None)
            return {**unavailable, "capability_reason": capability.get("reason", "qualified_runner_not_configured") if isinstance(capability, dict)
                else "qualified_runner_not_configured", "qualification_attempt_ref": capability.get("qualification_attempt_ref")
                if isinstance(capability, dict) else None}

        def run():
            _require(self.probe_runner.verify_capability(self.probe_profile_ref) == capability, "PROBE_CAPABILITY_CHANGED")
            output = self.probe_runner.run_probe(state=self.state, source_revision_ref=ref, probe=probe,
                capability=deepcopy(capability), operation_id=operation_id)
            _require(output.get("execution_kind") == capability["execution_kind"], "PROBE_EXECUTION_KIND_CHANGED")
            return output

        return self._execute(operation_id, request, run)

    def _replay_probe_operation(self, operation_id, request):
        """Resolve historical work before any runner qualification side effects.

        为什么要有它：新进程没有资格缓存；重放或未知结果必须先认出原操作，不能为了读回执再次启动容器。
        """
        _require(isinstance(operation_id, str) and operation_id, "EXPERIMENT_OPERATION_ID_REQUIRED")
        key = "experiment:" + operation_id
        expected = {**request, "task_contract_ref": self.state.contract_ref, "verification_scope": "exploration_only"}
        head = self.state.head

        def validate(stored, request_hash):
            _require(isinstance(stored, dict) and canonical_hash(stored) == request_hash
                and set(stored) in (set(expected) | {"runner_identity"}, set(expected) | {"runner_identity", "capability"}),
                "EXPERIMENT_REPLAY_REQUEST_INVALID")
            if any(stored.get(field) != value for field, value in expected.items()):
                raise TaskAgentV2StateError("operation_identity_conflict")

        if head.get("pending_operation"):
            pending = self.state.load(head["pending_operation"])
            _require(pending.get("schema") == "task-agent-operation-intent.v2"
                and pending.get("contract_ref") == self.state.contract_ref, "EXPERIMENT_REPLAY_INTENT_INVALID")
            if pending.get("operation_id") == key:
                validate(pending["request"], pending["request_hash"])
            raise TaskAgentV2StateError("operation_outcome_unknown")
        operation = head["operations"].get(key)
        if operation is None:
            return None
        result = operation["result"]
        stored = self.state.load(result["result_ref"])
        _require(stored.get("schema") == "task-agent-experiment-result.v2"
            and all(stored.get(field) == value for field, value in result.items() if field != "result_ref"),
            "EXPERIMENT_REPLAY_RESULT_INVALID")
        validate(stored["request"], operation["request_hash"])
        replay = self._execute(operation_id, stored["request"], None)
        if replay.get("status") == "capability_unavailable":
            capability = stored["request"].get("capability")
            return {**replay, "capability_reason": capability.get("reason", "qualified_runner_not_configured")
                if isinstance(capability, dict) else "qualified_runner_not_configured",
                "qualification_attempt_ref": capability.get("qualification_attempt_ref") if isinstance(capability, dict) else None}
        return replay

    def _execute(self, operation_id, request, runner):
        _require(isinstance(operation_id, str) and operation_id, "EXPERIMENT_OPERATION_ID_REQUIRED")
        request = {**request, "task_contract_ref": self.state.contract_ref, "verification_scope": "exploration_only"}
        with self.state._transaction("experiment:" + operation_id, request) as (_head, result, replay):
            if replay:
                return deepcopy(result)
            prepared = self.state.save("experiment-operation", {"schema": "task-agent-experiment-operation.v2",
                "operation_id": operation_id, "status": "prepared", "request": request})
            if runner is None:
                result.update(status="capability_unavailable", executed=False, grants_acceptance=False,
                    verification_scope="exploration_only", source_revision_ref=request["source_revision_ref"])
                result["result_ref"] = self.state.save("experiment-result", {"schema": "task-agent-experiment-result.v2",
                    **result, "request": request, "prepared_ref": prepared})
                return deepcopy(result)
            source = self.state.load(self.state.contract["baseline_manifest_ref"])
            original_hashes = {path: _file_ref(Path(source["source_root"]) / path) for path in source["files"]}
            original_heads = {key: _head[key] for key in ("current_revision_ref", "registry_ref", "plan_ref")}
            dispatched = self.state.save("experiment-operation", {"schema": "task-agent-experiment-operation.v2",
                "operation_id": operation_id, "status": "dispatched", "prepared_ref": prepared, "request": request})
            try:
                output = runner()
                _require(all(_file_ref(Path(source["source_root"]) / path) == reference
                    for path, reference in original_hashes.items())
                    and all(self.state.head[key] == value for key, value in original_heads.items()),
                    "EXPERIMENT_MUTATED_TASK_INPUTS")
                _require(output.get("execution_origin") == "executed"
                    and output.get("source_revision_ref") == request["source_revision_ref"]
                    and output.get("source_identity") == request["source_identity"]
                    and output.get("runner_identity") == request["runner_identity"]
                    and isinstance(output.get("exit_code"), int)
                    and output.get("logs_ref"), "EXPERIMENT_REPORT_BINDING_CHANGED")
                self.state.load(output["logs_ref"])
            except Exception as error:
                unknown = self.state.save("experiment-operation", {"schema": "task-agent-experiment-operation.v2",
                    "operation_id": operation_id, "status": "unknown", "dispatched_ref": dispatched,
                    "error_type": type(error).__name__, "reason": str(error), "request": request})
                raise ExperimentOutcomeUnknownError(unknown) from error
            result.update(status="completed" if output.get("passed", output["exit_code"] == 0) else "failed",
                executed=True, grants_acceptance=False, verification_scope="exploration_only", observations=deepcopy(output))
            result["result_ref"] = self.state.save("experiment-result", {"schema": "task-agent-experiment-result.v2",
                **result, "request": request, "dispatched_ref": dispatched})
            self.state.save("experiment-operation", {"schema": "task-agent-experiment-operation.v2",
                "operation_id": operation_id, "status": result["status"], "result_ref": result["result_ref"],
                "dispatched_ref": dispatched, "request": request})
        return deepcopy(result)
