"""Normal Task Agent tool adapters for registered experiments and probe artifacts.

为什么要有它：实验模块必须进入真实决策工具集合，执行结果继续使用独立的探索语义。
"""
from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from jsonschema import Draft202012Validator

from src.libs.llm.agent_turn import AgentToolDefinition
from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryBackend
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_loop import PreparedTaskAction, TaskAgentAction
from src.mod_agent.task_agent_v2_experiments import (
    CSharpRegisteredTestRunner,
    ExperimentRejectedError,
    TaskAgentExperiments,
    _source,
)


def experiment_configuration_facts(owner):
    """Expose configured choices without starting discovery or runner qualification.

    为什么要有它：模型选择实验前应知道已确定的环境限制，配置存在仍不证明实际执行资格。
    """
    facts = {"scope": "configuration_only_not_runtime_qualification",
        "probe_submission": {"status": "available", "executes_code": False},
        "registered_tests": {"status": "configuration_unknown"},
        "probe_execution": {"status": "configuration_unknown"}}
    reference = getattr(owner, "worker_configuration_ref", None)
    if reference is None:
        return facts
    worker = read_bound_json(reference)
    evaluator = worker.get("behavior_evaluator")
    if not isinstance(evaluator, dict) or evaluator.get("factory_name") != "make_csharp_repository_evaluator":
        return facts
    profile_ref = evaluator.get("kwargs", {}).get("profile_ref")
    if profile_ref is None:
        return facts
    profile = read_bound_json(profile_ref)
    if profile.get("profile_version") != "csharp_repository_profile_v1":
        return facts
    facts["repository_profile_ref"] = deepcopy(profile_ref)
    facts["registered_tests"] = {"status": "configured", "requires_current_revision_catalog": True,
        "grants_acceptance": False}
    configured = profile.get("probe_runner") is not None
    facts["probe_execution"] = {"status": "qualification_required" if configured else "unavailable",
        "reason": "runner_configuration_requires_actual_qualification" if configured else "qualified_runner_not_configured",
        "runtime_qualification_verified": False, "grants_acceptance": False}
    return facts


def _experiments(session):
    cached = getattr(session, "_experiments", None)
    if cached is None:
        worker = read_bound_json(session.owner.worker_configuration_ref)
        backend = CSharpRepositoryBackend(**worker["behavior_evaluator"]["kwargs"])
        probe_runner = None
        if backend.profile.data.get("probe_runner") is not None:
            from src.mod_agent.domain.repository_probe_runner import RepositoryProbeRunner
            probe_runner = RepositoryProbeRunner(state=session.state, backend=backend)
        cached = TaskAgentExperiments(state=session.state, test_runner=CSharpRegisteredTestRunner(backend),
            probe_runner=probe_runner)
        session._experiments = cached
    return cached


def experiment_actions(session):
    text = {"type": "string", "minLength": 1}
    specs = [
        ("test_catalog", {"type": "object", "additionalProperties": False, "properties": {
            "source_revision_ref": text, "execute_discovery": {"type": "boolean"}}},
         "Read registered method-group IDs from an executed public check. With execute_discovery=true, explicitly build and run the existing public suite at source_revision_ref (baseline is allowed) to register actual tests. Parameterized cases select the whole group; discovery does not grant finish."),
        ("probe_submit", {"type": "object", "additionalProperties": False,
            "properties": {"source_revision_ref": text, "files": {"type": "array", "minItems": 1,
                "items": {"type": "object", "additionalProperties": False,
                    "properties": {"relative_path": text, "content": {"type": "string"}},
                    "required": ["relative_path", "content"]}}, "entrypoint_id": {"enum": ["csharp_console"]},
                "rationale": text, "expected_observation": text},
            "required": ["source_revision_ref", "files", "entrypoint_id", "rationale", "expected_observation"]},
         "Save a version-bound temporary C# probe artifact under probe/*.cs. This does not run code or modify the candidate. Probe execution requires a qualified isolated runner."),
        ("experiment_run", {"type": "object", "additionalProperties": False,
            "properties": {"source_revision_ref": text, "kind": {"enum": ["registered_test", "temporary_probe"]},
                "catalog_ref": text, "test_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": text},
                "probe_artifact_ref": text, "hypothesis": text, "expected_observation": text},
            "required": ["source_revision_ref", "kind", "hypothesis", "expected_observation"]},
         "Run selected registered test groups or a submitted probe at the exact revision. Exploratory results never grant finish. An unavailable probe runner leaves other capabilities usable."),
    ]

    def prepare(name, schema, args):
        errors = list(Draft202012Validator(schema).iter_errors(args))
        if errors:
            raise ValueError("TASK_EXPERIMENT_ARGUMENTS_INVALID: " + errors[0].message)
        session.owner.check_experience_boundary()
        if "source_revision_ref" in args:
            session.state.materialize(args["source_revision_ref"])
        if name == "experiment_run":
            needed = {"catalog_ref", "test_ids"} if args["kind"] == "registered_test" else {"probe_artifact_ref"}
            if not needed <= set(args):
                raise ValueError("TASK_EXPERIMENT_SELECTION_REQUIRED")
        return PreparedTaskAction(deepcopy(args), 0, canonical_hash({"contract": session.state.contract_ref,
            "revision": session.state.current_revision_ref}))

    def execute_owned(name, args):
        session.owner.check_experience_boundary()
        origin = session._origin(name, args)
        experiments = _experiments(session)
        model_origin = {key: deepcopy(origin[key]) for key in ("call_id", "response_ref", "arguments_ref")}
        try:
            if name == "test_catalog":
                if args.get("execute_discovery") is True:
                    reference = args.get("source_revision_ref", session.state.current_revision_ref)
                    return _discover_catalog(session, experiments, reference, origin["call_id"], model_origin)
                if session.current_check is None:
                    return {"status": "catalog_unavailable", "reason": "no_executed_public_check",
                        "discovery_available": True, "grants_acceptance": False}
                check = session.state.load(session.current_check)
                return experiments.test_catalog(check["report"]["verification_ref"])
            if name == "probe_submit":
                return experiments.probe_submit(**args, model_origin=model_origin)
            session.state.save("experiment-hypothesis", {"schema": "task-agent-experiment-hypothesis.v2",
                "arguments": args, "model_origin": model_origin})
            if args["kind"] == "registered_test":
                result = experiments.registered_tests_run(source_revision_ref=args["source_revision_ref"],
                    catalog_ref=args["catalog_ref"], test_ids=args["test_ids"], operation_id=origin["call_id"],
                    model_origin=model_origin)
                _invalidate_check(session, args["source_revision_ref"], result)
                return result
            return experiments.experiment_run(source_revision_ref=args["source_revision_ref"],
                probe_artifact_ref=args["probe_artifact_ref"], operation_id=origin["call_id"], model_origin=model_origin)
        except ExperimentRejectedError as error:
            # 为什么要有它：该类型只表示已知准入拒绝，真正执行后未知异常仍交给共享循环保留unknown。
            return {"status": "rejected", "reason": str(error), "executed": False, "grants_acceptance": False}

    def execute(name, args):
        output = execute_owned(name, args)
        recorder = getattr(session.owner, "record_v2_tool_receipt", None)
        if callable(recorder):
            # 为什么要有它：实验选择也需要独立调用回执，恢复与经验提炼不能仅相信检查点中的自述结果。
            origin = session._origin(name, args)
            reference = recorder(origin, output, session)
            if reference is not None:
                origin["execution_receipt_ref"] = reference
        return output

    return [TaskAgentAction(AgentToolDefinition(name, description, schema), "upstream",
        lambda args, name=name, schema=schema: prepare(name, schema, args),
        lambda args, name=name: execute(name, args), False) for name, schema, description in specs]


def _invalidate_check(session, revision, receipt):
    # 为什么要有它：同版本后续公开执行反例不能被旧全绿掩盖；失败原件保留，交付前须重新正式验收。
    current = session.state.current_revision_ref
    if revision == "baseline":
        revision = session.state.contract["baseline_manifest_ref"]
    if receipt.get("executed") is True and receipt.get("status") == "failed" and revision == current:
        session.state.save("check-invalidation", {"schema": "task-agent-check-invalidation.v2",
            "previous_check_ref": session.current_check, "experiment_ref": receipt.get("result_ref"),
            "reason": "later_public_experiment_failure", "candidate_ref": current})
        session.current_check = None


def _discover_catalog(session, experiments, revision, operation_id, model_origin):
    from src.mod_agent.domain.csharp_repository_backend import _IGNORED, _snapshot, _within

    state, backend = session.state, experiments.test_runner.backend
    source_ref, source_id, files = _source(state, revision)
    baseline = state.load(state.contract["baseline_manifest_ref"])
    if Path(baseline["source_root"]).resolve() != backend.source_root:
        raise ExperimentRejectedError("DISCOVERY_SOURCE_MISMATCH")
    before = _snapshot(backend.source_root)
    request = {"kind": "public_test_discovery", "source_revision_ref": source_ref,
        "source_identity": source_id, "runner_identity": experiments.test_runner.identity(), "model_origin": model_origin}

    def execute():
        scratch = backend.profile.task_work_root(backend.workspace_root) / "experiments" / uuid4().hex / "s"
        shutil.copytree(backend.source_root, scratch, ignore=shutil.ignore_patterns(*_IGNORED))
        for path, text in files.items():
            if text != baseline["files"][path]["text"]:
                _within(scratch, path).write_bytes(text.encode("utf-8"))
        result = backend.run_build_test(scratch, session.feature_id, "preapproval_isolated_compilation")
        if _snapshot(backend.source_root) != before:
            raise RuntimeError("DISCOVERY_ORIGINAL_SOURCE_CHANGED")
        logs_ref = state.save("experiment-logs", {"schema": "task-agent-experiment-logs.v2",
            "verification_ref": result["report_ref"], "build_execution": result["build_execution"],
            "test_execution": result["test_execution"]})
        return {"execution_origin": "executed", "execution_kind": "public_test_discovery",
            "verification_ref": result["report_ref"], "source_identity": source_id, "source_revision_ref": source_ref,
            "runner_identity": request["runner_identity"], "logs_ref": logs_ref,
            "exit_code": result["test_execution"].get("exit_code", result["build_execution"].get("exit_code", -1)),
            "tests_executed": result["verification_summary"]["checks"],
            "passed": result["status"] == "passed", "observations": result["observations"]}

    receipt = experiments._execute(operation_id, request, execute)
    _invalidate_check(session, source_ref, receipt)
    # 为什么要有它：目录来自真实测试定义；基线也可先执行观察，不强迫模型先修改代码才获得实验选择能力。
    report_ref = receipt.get("observations", {}).get("verification_ref")
    if report_ref is None and receipt.get("result_ref"):
        stored = state.load(receipt["result_ref"])
        report_ref = stored.get("observations", {}).get("verification_ref")
    if report_ref is None:
        return receipt
    try:
        catalog = experiments.test_catalog(report_ref)
    except ExperimentRejectedError as error:
        # 为什么要有它：构建失败或测试目录不合格发生在实际执行之后，不能被外层误报为未执行的准入拒绝。
        return {**receipt, "catalog_status": "unavailable", "catalog_reason": str(error)}
    return {**catalog, "discovery_receipt": receipt, "grants_acceptance": False}
