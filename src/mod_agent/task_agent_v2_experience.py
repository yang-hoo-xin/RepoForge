"""Actual v2 decision and terminal-learning hooks using the existing provider.

为什么要有它：经验必须进入真实请求，提炼需从公开执行原件产生并计入原账本。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.libs.llm.agent_turn import AgentMessage
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.model_call_budget import call_allowance, validate_budget
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.planning.strategy_experience import StrategyExperience


def public_counterexample_kind(report):
    """Classify executed public evidence, including compiler counterexamples.

    为什么要有它：真实编译反例可改善下一次调查，但格式拒绝、未执行报告或环境失败不能被升级为试验观察。
    """
    if (report.get("status") != "behavior_failed" or report.get("execution_origin") != "executed"
            or not report.get("observations")):
        return None
    original = read_bound_json(_public_reference(report["report_ref"]))
    if any(original.get(key) != report.get(key) for key in (
            "status", "execution_origin", "observations", "verification_summary", "candidate_source_id",
            "verification_ref", "validation_results", "source_snapshot_sha256")):
        raise ValueError("TASK_V2_LEARNING_REPORT_CHANGED")
    execution = read_bound_json(_public_reference(original["verification_ref"]))
    binding = ("task_contract_ref", "candidate_ref", "candidate_source_id", "source_snapshot_sha256", "profile_ref")
    if (execution.get("execution_origin") != "executed" or execution.get("status") != original["status"]
            or any(execution.get(key) != original.get(key) for key in binding)
            or execution.get("observations") != original["observations"]):
        raise ValueError("TASK_V2_LEARNING_EXECUTION_CHANGED")
    summary = execution.get("verification_summary", {})
    if (summary.get("test_command_executed") is True and summary.get("failed_checks", 0) > 0
            and execution.get("test_execution", {}).get("status") == "executed"
            and execution["test_execution"].get("report_ref")):
        return "public_business_counterexample"
    compilation_ref = execution.get("compilation_report_ref")
    if compilation_ref is None:
        return None
    if compilation_ref != original.get("validation_results", {}).get("full_compilation", {}).get("report_ref"):
        raise ValueError("TASK_V2_LEARNING_COMPILATION_CHANGED")
    compilation = read_bound_json(_public_reference(compilation_ref))
    if (compilation.get("execution_origin") != "executed"
            or any(compilation.get(key) != original.get(key) for key in binding)):
        raise ValueError("TASK_V2_LEARNING_COMPILATION_CHANGED")
    build = compilation.get("build_execution", {})
    if (compilation.get("status") == "failed" and build.get("restore_exit_code") == 0
            and build.get("build_executed") is True and type(build.get("build_exit_code")) is int
            and build["build_exit_code"] != 0
            and any(row.get("check_id") == "repository_compilation" for row in original["observations"])):
        return "public_compilation_counterexample"
    return None


def _learning_require(value, code):
    if not value:
        raise ValueError("TASK_V2_LEARNING_" + code)


def _public_reference(reference):
    _learning_require(not {"hidden", "evaluation-only"}.intersection(
        part.casefold() for part in Path(reference["path"]).parts), "NONPUBLIC_SOURCE")
    return reference


def _validated_action(call, session):
    """Read the actual model call and independent tool receipt, never inferred reasoning."""
    from src.mod_agent.task_agent_execution import _admitted_response, _read

    root = session.state.store_root
    response, arguments = _read(_public_reference(call["response_ref"]), root), _read(_public_reference(call["arguments_ref"]), root)
    _admitted_response(response, session.owner.profile.to_dict())
    matching = [(index, row) for index, row in enumerate(response.get("tool_calls", [])) if row["call_id"] == call["call_id"]]
    _learning_require(len(matching) == 1 and matching[0][1] == {
        "call_id": call["call_id"], "name": call["name"], "arguments": arguments}
        and arguments == call["arguments"], "ACTION_CHANGED")
    reference = call.get("execution_receipt_ref")
    _learning_require(reference is not None, "ACTION_RECEIPT_MISSING")
    receipt = _read(_public_reference(reference), root)
    _learning_require({key: value for key, value in receipt["origin"].items() if key != "execution_receipt_ref"}
        == {key: value for key, value in call.items() if key != "execution_receipt_ref"}, "ACTION_RECEIPT_CHANGED")
    return response, matching[0][0], receipt


def _experiment_evidence(session, completed_ids=None):
    """Verify public exploratory observations without promoting them to acceptance.

    为什么要有它：模型自主实验可以提供策略学习反例，拒绝、环境故障和探针通过不能冒充业务验收。
    """
    import hashlib

    from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile, _read_tests
    from src.mod_agent.task_agent_v2_experiments import _source
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State

    selected = []
    lineage = {session.state.contract_ref}
    previous = session.state.contract.get("historical_task_contract_ref")
    while previous is not None:
        _learning_require(previous not in lineage, "EXPERIMENT_LINEAGE_CYCLE")
        lineage.add(previous)
        previous = session.state.load(previous).get("historical_task_contract_ref")
    for call in getattr(session, "actual_calls", []):
        if call["name"] not in {"test_catalog", "experiment_run"} or completed_ids is not None and call["call_id"] not in completed_ids:
            continue
        if call.get("execution_receipt_ref") is None and completed_ids is None:
            continue
        _response, _index, receipt = _validated_action(call, session)
        output = receipt["output"]
        returned = output.get("discovery_receipt", output)
        # 为什么要有它：目录不可用说明可以伴随真实执行失败，不能把目录展示字段误当作执行回执遭篡改。
        value = {key: returned[key] for key in ("status", "executed", "grants_acceptance", "verification_scope",
            "source_revision_ref", "observations", "result_ref") if key in returned}
        if value.get("executed") is not True:
            continue
        _learning_require(value.get("grants_acceptance") is False and value.get("verification_scope") == "exploration_only",
            "EXPERIMENT_SCOPE_CHANGED")
        original = session.state.load(value["result_ref"])
        _learning_require(original.get("schema") == "task-agent-experiment-result.v2"
            and all(original.get(k) == v for k, v in value.items() if k != "result_ref"), "EXPERIMENT_RESULT_CHANGED")
        request = original["request"]
        _learning_require(request["task_contract_ref"] in lineage, "EXPERIMENT_TASK_CHANGED")
        state = TaskAgentV2State(session.state.store_root, request["task_contract_ref"])
        operation = state.head["operations"].get("experiment:" + call["call_id"], {})
        dispatched = state.load(original["dispatched_ref"])
        _learning_require(operation.get("request_hash") == canonical_hash(request) and operation.get("result") == value
            and dispatched.get("status") == "dispatched" and dispatched.get("operation_id") == call["call_id"]
            and dispatched.get("request") == request
            and request["model_origin"] == {k: call[k] for k in ("call_id", "response_ref", "arguments_ref")},
            "EXPERIMENT_EXECUTION_CHANGED")
        actual_revision = call["arguments"].get("source_revision_ref", receipt["task_head"]["current_revision_ref"])
        revision, identity, files = _source(state, actual_revision)
        observed = original["observations"]
        _learning_require(request["source_revision_ref"] == observed["source_revision_ref"] == revision
            and request["source_identity"] == observed["source_identity"] == identity
            and observed["execution_origin"] == "executed" and observed["runner_identity"] == request["runner_identity"],
            "EXPERIMENT_SOURCE_CHANGED")
        kind = None
        verification_ref = observed.get("verification_ref")
        if request["kind"] in {"public_test_discovery", "registered_tests"}:
            _learning_require(verification_ref is not None and observed["execution_kind"] in {"public_test_discovery", "registered_tests"},
                "PUBLIC_EXPERIMENT_REPORT_REQUIRED")
            report = read_bound_json(_public_reference(verification_ref))
            logs = state.load(observed["logs_ref"])
            _learning_require(logs["verification_ref"] == verification_ref
                and logs["test_execution"] == report["test_execution"] and logs["build_execution"] == report["build_execution"]
                and report["execution_origin"] == "executed"
                and observed["observations"] == report["observations"], "PUBLIC_EXPERIMENT_REPORT_CHANGED")
            profile = CSharpRepositoryProfile(report["profile_ref"])
            if request["kind"] == "public_test_discovery":
                _learning_require(report["profile_ref"] == state.contract["execution_profile_ref"], "EXPERIMENT_PROFILE_CHANGED")
            else:
                derived = state.load(observed["derived_profile_ref"])
                _learning_require(derived["original_profile_ref"] == state.contract["execution_profile_ref"]
                    and derived["source_revision_ref"] == revision and derived["source_identity"] == identity
                    and derived["derived_profile"] == read_bound_json(report["profile_ref"]), "EXPERIMENT_PROFILE_CHANGED")
            hashes = {row["path"]: row["sha256"] for row in report["source_inputs"]}
            _learning_require(all(hashes.get(path) == hashlib.sha256(text.encode("utf-8")).hexdigest()
                for path, text in files.items()), "EXPERIMENT_CHECKED_CODE_CHANGED")
            compilation = read_bound_json(_public_reference(report["compilation_report_ref"]))
            _learning_require(compilation["build_execution"] == report["build_execution"]
                and compilation["source_inputs"] == report["source_inputs"]
                and compilation["profile_ref"] == report["profile_ref"]
                and compilation["execution_origin"] == "executed", "EXPERIMENT_COMPILATION_CHANGED")
            if report["status"] == "behavior_failed" and report.get("failure_owner") == "patch_implementation":
                tests = report["test_execution"]
                if tests.get("status") == "executed" and tests.get("report_ref"):
                    test_ref = tests["report_ref"]
                    _public_reference(test_ref)
                    _learning_require(hashlib.sha256(Path(test_ref["path"]).read_bytes()).hexdigest() == test_ref["sha256"],
                        "EXPERIMENT_TEST_REPORT_CHANGED")
                    checks = _read_tests(Path(test_ref["path"]), profile.data["test_report"]["format"])
                    _learning_require(checks == report["checks"] and bool(checks), "EXPERIMENT_TEST_RESULTS_CHANGED")
                    if any(not row["passed"] for row in checks) and (request["kind"] != "registered_tests"
                            or observed.get("selection_complete") is True and not observed.get("failure_code")):
                        kind = "public_business_counterexample"
                elif compilation.get("status") == "failed":
                    build = compilation["build_execution"]
                    if (build.get("restore_exit_code") == 0 and build.get("build_executed") is True
                            and type(build.get("build_exit_code")) is int and build["build_exit_code"] != 0
                            and any(row.get("check_id") == "repository_compilation" for row in report["observations"])):
                        kind = "public_compilation_counterexample"
        selected.append({"call": call, "result_ref": value["result_ref"], "observations": observed,
            "verification_ref": verification_ref, "counterexample_kind": kind, "source_identity": identity,
            "scope": "exploration_only", "grants_acceptance": False})
    return selected


def _effective_revision(state, failed, passed):
    """Only attribute an observed fix to changed descendant code under the same checks.

    为什么要有它：同代码重复检查偶然转绿不能成为有效修订，验收条件变化也不能冒充代码改善。
    """
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State

    if (not failed.get("candidate_source_id") or failed["candidate_source_id"] == passed.get("candidate_source_id")
            or failed.get("task_contract_ref") != passed.get("task_contract_ref")
            or failed.get("verification_identity") != passed.get("verification_identity")):
        return False
    selected = TaskAgentV2State(state.store_root, passed["task_contract_ref"])
    candidate = selected.validate_candidate(passed["candidate_ref"])
    if candidate["candidate_source_id"] != passed["candidate_source_id"]:
        return False
    parent = candidate["parent_revision_ref"]
    while parent != selected.contract["baseline_manifest_ref"]:
        candidate = selected.validate_candidate(parent)
        if parent == failed.get("candidate_ref"):
            return candidate["candidate_source_id"] == failed["candidate_source_id"]
        parent = candidate["parent_revision_ref"]
    return False


class TaskAgentV2Experience:
    def __init__(self, owner, configuration_ref):
        self.owner = owner
        self.configuration_ref = deepcopy(configuration_ref)
        self.config = read_bound_json(configuration_ref)
        if (set(self.config) != {"schema_version", "store_root", "snapshot_ref", "extract_on_terminal"}
                or self.config["schema_version"] != "task_agent_experience_config_v1"
                or type(self.config["extract_on_terminal"]) is not bool):
            raise ValueError("TASK_V2_EXPERIENCE_CONFIG_INVALID")
        root = Path(self.config["store_root"])
        if not root.is_absolute() or root.is_symlink() or str(root).startswith(("\\\\", "//")):
            raise ValueError("TASK_V2_EXPERIENCE_STORE_INVALID")
        self.memory = StrategyExperience(ArtifactStore(root))
        self.snapshot = self.config["snapshot_ref"]
        self._terminal_receipt = None
        self.check_boundary()

    def check_boundary(self):
        if read_bound_json(self.configuration_ref) != self.config:
            raise ValueError("TASK_V2_EXPERIENCE_CONFIG_CHANGED")
        if self.snapshot is not None:
            self.memory.check_snapshot(self.snapshot)

    @staticmethod
    def context(session):
        kinds = {kind for row in session.public_checks
            if (kind := public_counterexample_kind(row["report"])) is not None}
        if getattr(session, "actual_calls", None):
            kinds.update(row["counterexample_kind"] for row in _experiment_evidence(session) if row["counterexample_kind"] is not None)
        # 为什么要有它：只按已执行公开证据区分适用域，编译策略不能仅因同属C#就投放到业务反例；混合来源单独保留。
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed_public_counterexamples" if kinds else "none"
        return {"runtime": "dotnet", "target_domain": "csharp_repository",
            "has_public_counterexample": bool(kinds), "counterexample_kind": kind}

    def prepare_decision_request(self, request, session):
        self.check_boundary()
        if self.snapshot is None:
            return request
        selection = self.memory.select(self.snapshot, self.context(session))
        view = self.memory.advisory_model_view(selection)
        selection_ref = self.owner._save("experience-selection-" + uuid4().hex + ".json", {
            "configuration_ref": self.configuration_ref, "selection": selection,
            "model_view": view, "model_view_sha256": canonical_hash(view)})
        # 为什么要有它：每轮只附加本轮核验的投影，旧上下文不累积重复策略或获得新授权。
        prepared = replace(request, messages=(*request.messages, AgentMessage("user", {"strategy_experience": view})))
        final_ref = self.owner._save("experience-request-" + uuid4().hex + ".json", asdict(prepared))
        self.owner._save("experience-delivery-" + uuid4().hex + ".json", {
            "selection_ref": selection_ref, "model_view_sha256": canonical_hash(view), "final_request_ref": final_ref,
            "snapshot_ref": deepcopy(self.snapshot), "kind": "prepared_provider_request"})
        return prepared

    def _episode(self, session):
        sources, selections, gains, counterexamples = [], [], [], []
        execution = read_bound_json(self.owner.execution_receipt().execution_ref)
        completed = {row["call_id"] for row in execution["completed_tool_records"]}
        failures = []
        for index, checked in enumerate(session.public_checks):
            report = checked["report"]
            ref = report["report_ref"]
            original = read_bound_json(ref)
            if any(original.get(key) != report.get(key) for key in ("status", "observations", "verification_summary", "candidate_source_id")):
                raise ValueError("TASK_V2_LEARNING_REPORT_CHANGED")
            evidence_id = "public_check_" + str(index)
            sources.append({"kind": "json_file", "evidence_id": evidence_id, **ref,
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            for field in ("status", "observations", "verification_summary"):
                selections.append({"evidence_id": evidence_id, "locator": [field]})
            kind = public_counterexample_kind(report)
            if kind is not None:
                failures.append(original)
                gains.append("public_counterexample")
                counterexamples.append({"kind": kind, "evidence_id": evidence_id,
                    "candidate_source_id": original["candidate_source_id"]})
            elif original["status"] == "passed" and any(_effective_revision(session.state, failure, original) for failure in failures):
                gains.append("effective_revision")
        for index, experiment in enumerate(_experiment_evidence(session, completed)):
            evidence_id = "public_experiment_" + str(index)
            path = Path(session.state.store_root) / experiment["result_ref"]
            reference = _public_reference({"path": str(path.resolve()), "sha256": path.stem})
            sources.append({"kind": "json_file", "evidence_id": evidence_id, **reference,
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            selections.extend({"evidence_id": evidence_id, "locator": [field]} for field in (
                "observations", "verification_scope", "grants_acceptance"))
            if experiment["counterexample_kind"] is not None:
                gains.append("public_counterexample")
                counterexamples.append({"kind": experiment["counterexample_kind"], "evidence_id": evidence_id,
                    "source_identity": experiment["source_identity"], "scope": "exploration_only"})
        if not gains:
            return None
        for index, call in enumerate(session.actual_calls):
            if call["call_id"] not in completed:
                continue
            ref = call["response_ref"]
            _response, index_in_response, action_receipt = _validated_action(call, session)
            evidence_id = "public_action_" + str(index)
            sources.append({"kind": "json_file", "evidence_id": evidence_id, **ref,
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            selections.extend({"evidence_id": evidence_id, "locator": ["tool_calls", index_in_response, field]}
                for field in ("name", "arguments"))
            result_id = evidence_id + "_result"
            sources.append({"kind": "json_file", "evidence_id": result_id, **call["execution_receipt_ref"],
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            # 为什么要有它：请求过某工具不等于实验已执行；只投影实际返回字段，不补写模型未表达的技术判断。
            for field in ("status", "executed", "grants_acceptance", "reason", "path", "text", "matches",
                    "candidate_ref", "candidate_source_id", "source_view_ref", "technical_plan_ref", "plan_ref"):
                if field in action_receipt["output"]:
                    selections.append({"evidence_id": result_id, "locator": ["output", field]})
        return self.owner._save("public-learning-episode-" + uuid4().hex + ".json", {
            "schema_version": "public_strategy_episode_v1", "visibility": "public", "data_use": "strategy_learning",
            "information_gain": sorted(set(gains)), "evidence_refs": sources, "selections": selections,
            "public_counterexamples": counterexamples,
            "applicability": self.context(session), "task_contract_ref": session.state.contract_ref,
            "execution_ref": deepcopy(self.owner.execution_receipt().execution_ref)})

    def complete_terminal_learning(self):
        if self._terminal_receipt is not None:
            return deepcopy(self._terminal_receipt)
        owner = self.owner
        before = validate_budget(owner.budget_snapshot())
        result = {"status": "not_enabled", "configuration_ref": self.configuration_ref,
            "budget_before": before, "model_calls": 0, "known_usage": [], "unknown_usage_calls": 0}
        try:
            self.check_boundary()
            if self.config["extract_on_terminal"]:
                session = getattr(owner, "_v2_session", None)
                episode = self._episode(session) if session is not None else None
                if episode is None:
                    result["status"] = "no_information_gain"
                else:
                    result["episode_ref"] = episode
                    client = owner._v2_client
                    admission = owner._save("learning-admission-" + uuid4().hex + ".json", {
                        "status": "admitted", "purpose": "strategy_learning", "provider": client.provider,
                        "model": client.model, "model_identity": owner._v2_identity(),
                        "profile_sha256": canonical_hash(owner.profile.to_dict())})

                    def reserve(request_ref):
                        self.check_boundary()
                        if owner.stop_v2_check() or call_allowance(validate_budget(owner.budget_snapshot()), "initial_patch") <= 0:
                            return False
                        intent = owner._save("learning-reservation-" + uuid4().hex + ".json", {
                            "request_ref": request_ref, "before": validate_budget(owner.budget_snapshot())})
                        if owner.reserve_model_call("initial_patch") is False:
                            return False
                        result["model_calls"] += 1
                        owner._v2_state.record_resources({"model_calls": 1}, operation_id="learning:" + intent["sha256"])
                        return True

                    def complete(request):
                        self.check_boundary()
                        observed = owner._v2_identity()
                        bind = getattr(client, "bind_context_identity", None)
                        if callable(bind):
                            bind(model_identity=observed, owner_profile_sha256=canonical_hash(owner.profile.to_dict()))
                        response = client.complete_turn(request)
                        return response

                    def account(receipt):
                        result.setdefault("call_receipts", []).append(receipt)
                        usage = receipt.get("usage")
                        if usage is None:
                            result["unknown_usage_calls"] += 1
                        else:
                            result["known_usage"].append(usage)
                            owner._v2_state.record_resources(usage, operation_id="learning-usage:" + canonical_hash(receipt))
                        owner._save("learning-call-receipt-" + uuid4().hex + ".json", receipt)
                        owner._v2_identity()

                    learned = self.memory.propose_from_episode(
                        SimpleNamespace(provider=client.provider, model=client.model, complete_turn=complete), episode,
                        admission_ref=admission, before_call=reserve, after_call=account)
                    result.update(status=learned["status"], strategy_ref=learned.get("strategy_ref"),
                        extraction_ref=learned.get("extraction_ref"))
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, message=str(error))
            if result["model_calls"] > len(result["known_usage"]) + result["unknown_usage_calls"]:
                result["unknown_usage_calls"] = result["model_calls"] - len(result["known_usage"])
        result["budget_after"] = validate_budget(owner.budget_snapshot())
        result["original_execution_ref"] = deepcopy(owner.execution_receipt().execution_ref) if owner.execution_receipt() else None
        reference = owner._save("terminal-learning-receipt-" + uuid4().hex + ".json", result)
        self._terminal_receipt = {**result, "receipt_ref": reference}
        return deepcopy(self._terminal_receipt)
