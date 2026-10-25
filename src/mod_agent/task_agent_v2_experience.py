"""Actual v2 decision and terminal-learning hooks using the existing provider.

为什么要有它：经验必须进入真实请求，提炼需从公开执行原件产生并计入原账本。
"""
from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.libs.llm.agent_turn import AgentMessage, AgentTurnRequest
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.model_call_budget import call_allowance, validate_budget
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.planning.learning_request import read_learning_request, task_context_selections
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

    from src.mod_agent.domain.csharp_repository_backend import (
        CSharpRepositoryProfile,
        read_repository_tests,
    )
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
                    checks, _ = read_repository_tests(Path(test_ref["path"]), profile.data, report=report)
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


def _catalog_configuration(configuration):
    """Validate an explicitly selected dynamic collection without changing frozen modes.

    为什么要有它：目录更新必须来自宿主明确配置，不能借终态收尾扩充旧冻结集合或策略试验快照。
    """
    value = configuration.get("episode_catalog")
    if value is None:
        return None
    _learning_require(isinstance(value, dict) and {"catalog_id"} <= set(value)
        and not set(value) - {"catalog_id", "selection"}
        and isinstance(value["catalog_id"], str) and bool(value["catalog_id"].strip()), "CATALOG_CONFIG_INVALID")
    _learning_require(configuration.get("extraction_episode_set_ref") is None, "CATALOG_AND_FROZEN_SET_CONFLICT")
    selection = value.get("selection", {})
    _learning_require(isinstance(selection, dict) and not set(selection) - {"required_episode_refs", "run_choices"},
        "CATALOG_SELECTION_INVALID")
    _learning_require(isinstance(selection.get("required_episode_refs", []), list)
        and isinstance(selection.get("run_choices", {}), dict), "CATALOG_SELECTION_INVALID")
    if configuration["extract_on_terminal"]:
        _learning_require(configuration.get("learning_request_ref") is not None
            and configuration.get("requested_strategy_scope") is not None, "CATALOG_LEARNING_PURPOSE_REQUIRED")
    return deepcopy(value)


def _register_configured_public_episode(owner, episode_ref, configuration_ref):
    """Register sealed public facts independently of whether extraction is enabled."""
    if configuration_ref is None or episode_ref is None:
        return None
    configuration = read_bound_json(configuration_ref)
    catalog = _catalog_configuration(configuration)
    if catalog is None:
        return None
    root = Path(configuration["store_root"])
    _learning_require(root.is_absolute() and not root.is_symlink() and not str(root).startswith(("\\\\", "//")),
        "CATALOG_STORE_INVALID")
    value = {"schema": "task-agent-public-episode-registration.v1", "configuration_ref": deepcopy(configuration_ref),
        "episode_ref": deepcopy(episode_ref), "catalog_id": catalog["catalog_id"], "model_calls": 0,
        "strategy_activated": False, "frozen_snapshot_modified": False}
    try:
        memory = StrategyExperience(ArtifactStore(root))
        registered = memory.register_public_episode(catalog["catalog_id"], episode_ref)
        value.update(status="registered", catalog_registration_ref=asdict(registered))
    except Exception as error:
        value.update(status="failed", error_type=type(error).__name__, message=str(error))
    name = "public-episode-registration-" + canonical_hash(value) + ".json"
    try:
        try:
            receipt_ref = owner._save(name, value)
        except FileExistsError:
            from src.mod_agent.task_agent_v2_recovery import _ref
            receipt_ref = _ref(Path(owner.output_root) / name)
            _learning_require(read_bound_json(receipt_ref) == value, "CATALOG_REGISTRATION_RECEIPT_CHANGED")
        return {**value, "receipt_ref": receipt_ref}
    except Exception as error:
        # 为什么要有它：已封存经历或已成功登记不因派生回执保存失败而消失，也不能把登记失败写成成功。
        return {**value, "receipt_ref": None, "receipt_error": {"error_type": type(error).__name__, "message": str(error)}}


def seal_current_public_episode(owner, session=None, *, configuration_ref=None):
    """Seal a live owner's retained execution even when extraction is not configured.

    为什么要有它：公开过程采集不能以经验配置或一次模型调用为前提，调用者可在正常收尾独立保存事实。
    """
    session = session if session is not None else getattr(owner, "_v2_session", None)
    if session is None or owner.execution_receipt() is None:
        return None
    collector = TaskAgentV2Experience.__new__(TaskAgentV2Experience)
    collector.owner = owner
    episode = collector._episode(session)
    if configuration_ref is None:
        configuration_ref = getattr(owner, "_episode_catalog_configuration_ref", None)
    if configuration_ref is None:
        configuration_ref = getattr(getattr(owner, "_experience", None), "configuration_ref", None)
    owner._public_episode_catalog_registration = _register_configured_public_episode(owner, episode, configuration_ref)
    return episode


def _semantic_validation_evidence(owner, session, execution):
    """Seal compact validation facts from an authenticated checkpoint, without model calls.

    为什么要有它：结构化声明可被接受、拒绝或留空，后续学习需要这些真实处理结果，而不是再复制全部状态正文。
    """
    checkpoint_ref = getattr(owner, "_public_episode_checkpoint_ref", None) or getattr(owner, "v2_checkpoint_ref", None)
    if checkpoint_ref is None:
        return None
    state = session.state
    checkpoint = state.load(checkpoint_ref)
    semantic = (checkpoint.get("owner", {}).get("working_state") or {}).get("semantic_state")
    if semantic is None:
        return None
    _learning_require(checkpoint["task_contract_ref"] == state.contract_ref
        and all(checkpoint["details"].get(key) == execution.get(key) for key in ("run_id", "request_id")),
        "SEMANTIC_CHECKPOINT_TASK_CHANGED")
    from src.mod_agent.task_agent_v2_recovery import _ref, _trusted_read, _validate_completed
    from src.mod_agent.task_agent_v2_semantic_state import validate_semantic_state

    host = state.load(checkpoint["host_config_ref"])
    _learning_require(host.get("semantic_state_enabled") is True, "SEMANTIC_POLICY_NOT_ENABLED")
    _validate_completed(checkpoint, state.store_root)
    turns = []
    for index, turn in enumerate(checkpoint["details"]["provider_turns"], 1):
        _learning_require(turn["turn"] == index, "SEMANTIC_PROVIDER_SEQUENCE_CHANGED")
        if not turn.get("response_ref"):
            continue
        request = _trusted_read(_public_reference(turn["request_ref"]), state.store_root)
        response = _trusted_read(_public_reference(turn["response_ref"]), state.store_root)
        _learning_require(isinstance(turn.get("reservation_id"), str)
            and re.fullmatch(r"[0-9a-f]{32}", turn["reservation_id"]), "SEMANTIC_PROVIDER_RESERVATION_INVALID")
        # 为什么要有它：停止检查点可能已有下一次预留但尚无响应；仅核验已落盘响应，不替未知尾部补终态。
        outcome_ref = _ref(Path(turn["request_ref"]["path"]).parent / ("provider-outcome-" + turn["reservation_id"] + ".json"))
        _learning_require(_trusted_read(_public_reference(outcome_ref), state.store_root) == turn,
            "SEMANTIC_PROVIDER_OUTCOME_CHANGED")
        turns.append({"request_ref": turn["request_ref"], "response_ref": turn["response_ref"],
            "request": request, "response": response})
    validate_semantic_state(checkpoint["owner"]["working_state"], turns)
    completed_order = {row["call_id"]: index for index, row in enumerate(execution["completed_tool_records"])}
    events = []
    for event in semantic["events"]:
        compact = {key: deepcopy(event[key]) for key in (
            "request_ref", "response_ref", "call_ids", "status", "diagnostics", "snapshot_id") if key in event}
        # 为什么要有它：校验事实需要接回实际动作索引；未在完成prefix中的调用保留未知，不能凭声明推断已执行。
        compact["action_indices"] = [completed_order[call_id] for call_id in event["call_ids"] if call_id in completed_order]
        compact["outside_completed_prefix_call_ids"] = [call_id for call_id in event["call_ids"] if call_id not in completed_order]
        events.append(compact)
    summary = {"schema": "task-agent-public-semantic-validation.v1", "task_contract_ref": state.contract_ref,
        "source_checkpoint_ref": _ref(Path(state.store_root) / checkpoint_ref),
        "source_execution_ref": deepcopy(owner.execution_receipt().execution_ref),
        "events": events,
        "validation_scope": "Retained checkpoint prefix: recorded metadata shape, visible-quote provenance and update handling only. "
            "call_ids names metadata-bearing calls, not every action in a response. "
            "Accepted/reaffirmed does not mean semantic truth, hypothesis refutation, business success or strategy adoption.",
        "model_calls": 0, "authority_changed": False}
    name = "public-semantic-validation-" + canonical_hash(summary) + ".json"
    try:
        return owner._save(name, summary)
    except FileExistsError:
        reference = _ref(Path(owner.output_root) / name)
        _learning_require(read_bound_json(reference) == summary, "SEMANTIC_VALIDATION_CHANGED")
        return reference


def _process_context(session, current_task_state=None):
    """Project current, source-bound process facts for conditional strategy selection.

    为什么要有它：过程策略需要区分尚无候选、当前检查和精确重复返回；未知条件不能被补成false或语义无进展。
    """
    from src.mod_agent.task_agent_execution import _read

    state, owner = session.state, session.owner
    checkpoint_ref = getattr(owner, "_public_episode_checkpoint_ref", None)
    if checkpoint_ref is not None:
        checkpoint = state.load(checkpoint_ref)
        _learning_require(checkpoint["task_contract_ref"] == state.contract_ref, "CONTEXT_CHECKPOINT_TASK_CHANGED")
        head, saved_session = checkpoint["task_head"], checkpoint["session"]
        current, registry, check_ref = head["current_revision_ref"], head["registry_ref"], saved_session.get("current_check")
        inflight = saved_session.get("_check_inflight")
        working = checkpoint["owner"].get("working_state")
        working_ref, locator = checkpoint_ref, ["owner", "working_state"]
    else:
        current, registry, check_ref = state.current_revision_ref, state.registry_ref, getattr(session, "current_check", None)
        inflight = getattr(session, "_check_inflight", None)
        saved = current_task_state if current_task_state is not None else getattr(owner, "_v2_strategy_context_sources", None)
        expected = {"task_contract_ref": state.contract_ref, "current_revision_ref": current,
            "binding_registry_ref": registry, "current_check_ref": check_ref}
        working, working_ref, locator = None, None, []
        if isinstance(saved, dict) and all(saved.get(key) == value for key, value in expected.items()):
            working_ref = saved.get("working_state_ref")
            if working_ref is not None:
                _learning_require(isinstance(working_ref, str)
                    and re.fullmatch(r"v2/working-state/[0-9a-f]{64}\.json", working_ref), "CONTEXT_WORKING_REF_INVALID")
                path = Path(state.store_root) / working_ref
                # 为什么要有它：working-state原件采用v1 schema，复用受限path/hash读取而非误用仅接收v2 payload的state.load。
                working = _read(_public_reference({"path": str(path.resolve()), "sha256": path.stem}), state.store_root)
    boundary = {"task_contract_ref": state.contract_ref, "current_revision_ref": current,
        "binding_registry_ref": registry, "current_check_ref": check_ref}
    sources = {"schema_version": "strategy_selection_context_sources_v1", **boundary,
        "working_state_ref": working_ref, "working_state_locator": locator,
        "origin": "retained_public_checkpoint" if checkpoint_ref is not None else "current_execution_host",
        "stagnation_scope": "exact_repeated_tool_returns_only_not_semantic_progress"}
    facts = {}
    baseline = state.contract.get("baseline_manifest_ref")
    if isinstance(current, str) and isinstance(baseline, str):
        candidate = state.validate_candidate(current) if current != baseline else None
        facts["has_candidate"] = candidate is not None
        if check_ref is None and inflight is False:
            facts["task_phase"] = "candidate_unchecked" if candidate is not None else "investigation"
        elif check_ref is not None and candidate is not None:
            checked = state.load(check_ref)
            if (checked.get("schema") == "task-agent-check.v2" and checked.get("task_contract_ref") == state.contract_ref
                    and checked.get("candidate_ref") == current
                    and checked.get("candidate_source_id") == candidate["candidate_source_id"]):
                phase = {"passed": "public_check_passed", "behavior_failed": "public_check_failed",
                    "unknown": "public_check_unknown"}.get(checked.get("status"))
                if phase is not None:
                    facts["task_phase"] = phase
    if (isinstance(working, dict) and working.get("schema") == "task-agent-working-state.v1"
            and working.get("boundary") == boundary):
        from src.mod_agent.task_agent_v2_working_state import update_working_state

        stagnation = working.get("stagnation", {})
        streak = working.get("consecutive_repeated_returns")
        if (type(stagnation.get("triggered")) is bool and type(streak) is int and streak >= 0
                and stagnation.get("consecutive_repeated_returns") == streak
                and stagnation.get("business_progress_inferred") is False):
            retained = update_working_state(working, AgentTurnRequest(messages=()), task_contract_ref=state.contract_ref,
                current_revision_ref=current, registry_ref=registry, current_check_ref=check_ref)
            if retained["stagnation"]["triggered"] is stagnation["triggered"]:
                facts["stagnation_triggered"] = stagnation["triggered"]
                facts["consecutive_repeated_returns"] = streak
    return facts, sources


def seal_saved_public_episode(*, artifact_store_root, checkpoint_ref, stop_ref):
    """Seal the verified completed prefix after a process stops, without resuming it.

    为什么要有它：硬停止可能没有原生终态，但已完成工具原件仍有学习价值；封存不能派发在途操作或伪造完整执行。
    """
    from src.mod_agent.task_agent_v2_recovery import (
        _ref,
        _validate_completed,
        _validate_host_binding,
        _write_bound,
    )
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State, _no_links, load_v2_artifact

    root = Path(artifact_store_root).resolve(strict=True)
    _no_links(root)
    checkpoint = load_v2_artifact(root, checkpoint_ref)
    _learning_require(checkpoint.get("schema") == "task-agent-session-checkpoint.v2", "CHECKPOINT_REQUIRED")
    state = TaskAgentV2State(root, checkpoint["task_contract_ref"])
    host = state.load(checkpoint["host_config_ref"])
    _validate_host_binding(state, checkpoint, host)
    for reference in host["owner_refs"]:
        _learning_require(_ref(Path(reference["path"])) == reference, "FROZEN_OWNER_CHANGED")
    # 为什么要有它：待返回调用的轮数不等于完整终态；只核实已完成prefix，不对账、恢复或改变原检查点。
    _validate_completed(checkpoint, root)
    stop = read_bound_json(_public_reference(stop_ref))
    required = {"schema", "checkpoint_ref", "task_contract_ref", "run_id", "request_id", "stop_kind", "source_ref"}
    _learning_require(required <= set(stop) and stop["schema"] == "task-agent-process-stop.v1"
        and stop["checkpoint_ref"] == checkpoint_ref and stop["task_contract_ref"] == state.contract_ref
        and stop["stop_kind"] in {"timed_out", "user_stopped", "process_failed"}, "PROCESS_STOP_INVALID")
    _learning_require(all(stop[key] == checkpoint["details"][key] == host["graph_state"][key]
        for key in ("run_id", "request_id")), "PROCESS_STOP_TASK_CHANGED")
    process = read_bound_json(_public_reference(stop["source_ref"]))
    _learning_require(isinstance(process, dict) and process.get("status") in {
        "timed_out", "timeout", "exited", "failed", "terminated", "cancelled", "interrupted", "user_stopped"},
        "PROCESS_STOP_SOURCE_INVALID")
    if stop.get("plan_ref") is not None:
        plan = read_bound_json(_public_reference(stop["plan_ref"]))
        _learning_require(plan.get("run_id") == stop["run_id"]
            and (Path(plan["run_root"]) / "a").resolve() == root
            and Path(stop["source_ref"]["path"]).resolve() == (Path(plan["root"]) / "execution/process.json").resolve()
            and process.get("argv") == plan["command"]
            and Path(process["cwd"]).resolve() == Path(plan["workspace"]).resolve(), "PROCESS_PLAN_BINDING_CHANGED")
        _learning_require(all(key not in process or process[key] == stop[key] for key in ("run_id", "request_id")),
            "PROCESS_STOP_SOURCE_TASK_CHANGED")
    else:
        _learning_require(all(process.get(key) == stop[key] for key in ("run_id", "request_id")),
            "PROCESS_STOP_SOURCE_TASK_CHANGED")
    if stop["stop_kind"] == "timed_out":
        _learning_require(process.get("timeout") is True or process["status"] in {"timed_out", "timeout"},
            "PROCESS_STOP_SOURCE_CHANGED")
    elif stop["stop_kind"] == "process_failed":
        _learning_require(process["status"] in {"failed", "terminated"}
            or type(process.get("exit_code")) is int and process["exit_code"] != 0, "PROCESS_STOP_SOURCE_CHANGED")
    else:
        _learning_require(process["status"] in {"user_stopped", "cancelled", "interrupted"}
            or process.get("stop_kind") == "user_stopped" and process["status"] == "terminated",
            "PROCESS_STOP_SOURCE_CHANGED")
    if "exit_code" in stop:
        _learning_require(stop["exit_code"] == process.get("exit_code"), "PROCESS_STOP_SOURCE_CHANGED")

    def save(name, value):
        path = root / "public-learning" / name
        _no_links(path)
        if path.exists():
            reference = _ref(path)
            _learning_require(read_bound_json(reference) == value, "SEALED_EPISODE_CHANGED")
            return reference
        return _write_bound(path, value)

    loop = checkpoint["loop"]
    records = checkpoint["details"]["completed_tool_records"]
    completed_ids = [event["call_id"] for event in loop["events"] if event.get("event") == "tool_dispatch_completed"]
    _learning_require([row["call_id"] for row in records] == completed_ids, "COMPLETED_PREFIX_CHANGED")
    loop_projection = {"schema": "task-agent-public-loop-prefix.v1", "checkpoint_ref": checkpoint_ref,
        "call_records": deepcopy(loop["call_records"])}
    loop_ref = save("public-loop-" + canonical_hash(loop_projection) + ".json", loop_projection)
    current = checkpoint["task_head"]["current_revision_ref"]
    unknowns = [{"provider_pending": "provider_response_unknown", "tool_pending": "tool_execution_unknown"}[loop["phase"]]] \
        if loop["phase"] in {"provider_pending", "tool_pending"} else []
    if checkpoint["session"].get("_check_inflight"):
        unknowns.append("public_check_inflight")
    from src.mod_agent import (
        task_agent_execution,
        task_agent_v2_experiments,
        task_agent_v2_recovery,
        task_agent_v2_state,
    )
    from src.mod_agent.domain import csharp_repository_backend
    from src.mod_agent.planning import episode_catalog, learning_context, strategy_experience

    observation = {"schema": "task-agent-public-process-observation.v1", "status": stop["stop_kind"],
        "run_id": stop["run_id"], "request_id": stop["request_id"], "task_contract_ref": state.contract_ref,
        "checkpoint_ref": checkpoint_ref, "stop_ref": deepcopy(stop_ref), "process_ref": deepcopy(stop["source_ref"]),
        "completed_tool_records": deepcopy(records), "loop_result_ref": loop_ref,
        "candidate_ref": current if current != state.contract["baseline_manifest_ref"] else None,
        "check_ref": checkpoint["session"].get("current_check"), "unknown_effects": unknowns,
        "native_terminal": False, "business_result": "not_established", "model_calls": 0,
        "collector_scope": "post_process_public_collector",
        "collector_source_refs": [_ref(Path(path)) for path in (__file__, task_agent_execution.__file__,
            task_agent_v2_experiments.__file__, task_agent_v2_recovery.__file__, task_agent_v2_state.__file__,
            csharp_repository_backend.__file__, strategy_experience.__file__, episode_catalog.__file__, learning_context.__file__)],
        "stop_attribution": "host_observed_stop; not model abandonment or hypothesis refutation"}
    if stop.get("plan_ref") is not None:
        observation["plan_ref"] = deepcopy(stop["plan_ref"])
    observation_ref = save("process-observation-" + canonical_hash(observation) + ".json", observation)
    owner = SimpleNamespace(_save=save, profile=SimpleNamespace(to_dict=lambda: checkpoint["details"]["model_profile"]),
        _public_episode_checkpoint_ref=checkpoint_ref,
        _episode_catalog_configuration_ref=host.get("experience_configuration_ref"),
        execution_receipt=lambda: SimpleNamespace(execution_ref=observation_ref))
    session = SimpleNamespace(state=state, owner=owner,
        actual_calls=deepcopy([call for call in checkpoint["session"]["actual_calls"] if call["call_id"] in completed_ids]),
        public_checks=deepcopy(checkpoint["session"]["public_checks"]))
    return seal_current_public_episode(owner, session)


class TaskAgentV2Experience:
    def __init__(self, owner, configuration_ref):
        self.owner = owner
        self.configuration_ref = deepcopy(configuration_ref)
        self.config = read_bound_json(configuration_ref)
        required = {"schema_version", "store_root", "snapshot_ref", "extract_on_terminal"}
        if (not required <= set(self.config) or set(self.config) - required - {"trial_binding_ref", "extraction_episode_set_ref", "learning_request_ref", "requested_strategy_scope", "episode_catalog"}
                or self.config["schema_version"] != "task_agent_experience_config_v1"
                or type(self.config["extract_on_terminal"]) is not bool):
            raise ValueError("TASK_V2_EXPERIENCE_CONFIG_INVALID")
        root = Path(self.config["store_root"])
        if not root.is_absolute() or root.is_symlink() or str(root).startswith(("\\\\", "//")):
            raise ValueError("TASK_V2_EXPERIENCE_STORE_INVALID")
        self.memory = StrategyExperience(ArtifactStore(root))
        self.snapshot = self.config["snapshot_ref"]
        self.extraction_episode_set_ref = deepcopy(self.config.get("extraction_episode_set_ref"))
        self.episode_catalog = _catalog_configuration(self.config)
        self.learning_request_ref = deepcopy(self.config.get("learning_request_ref"))
        self.learning_request = read_learning_request(self.learning_request_ref)
        self.requested_strategy_scope = deepcopy(self.config.get("requested_strategy_scope"))
        if self.requested_strategy_scope is not None:
            from src.mod_agent.planning.strategy_experience import _strategy
            _strategy({"applicability": self.requested_strategy_scope, "suggested_action": "pending",
                "expected_observation": "unassessed", "counterevidence": ["unassessed"]})
            _learning_require(self.learning_request is not None or self.extraction_episode_set_ref is not None or self.episode_catalog is not None,
                "EXPLICIT_LEARNING_REQUEST_REQUIRED")
        self.trial_binding = self._read_trial_binding()
        self._terminal_receipt = None
        self.check_boundary()

    def _read_trial_binding(self):
        reference = self.config.get("trial_binding_ref")
        if reference is None:
            return None
        from src.mod_agent.evaluation.strategy_adoption import _planned_pairs

        binding = read_bound_json(reference)
        if (set(binding) != {"schema_version", "protocol_ref", "task_id", "attempt", "arm"}
                or binding["schema_version"] != "strategy_trial_run_binding_v1"
                or binding["arm"] not in {"B", "B+"} or type(binding["attempt"]) is not int
                or not isinstance(binding["task_id"], str) or not binding["task_id"]):
            raise ValueError("TASK_V2_TRIAL_BINDING_INVALID")
        protocol = read_bound_json(binding["protocol_ref"])
        if protocol.get("schema_version") != "strategy_trial_protocol_v1":
            raise ValueError("TASK_V2_TRIAL_PROTOCOL_INVALID")
        planned = _planned_pairs(protocol)
        if (binding["task_id"], binding["attempt"]) not in planned:
            raise ValueError("TASK_V2_TRIAL_TASK_NOT_PLANNED")
        expected_snapshot = protocol["snapshot_ref"] if binding["arm"] == "B+" else None
        if self.snapshot != expected_snapshot:
            raise ValueError("TASK_V2_TRIAL_ARM_SNAPSHOT_MISMATCH")
        self.memory.check_snapshot(protocol["snapshot_ref"])
        # 为什么要有它：已有host配置会在首个模型请求前绑定此配置；试验元数据只证明预登记，不增加技术或执行权限。
        return {"binding": binding, "protocol": protocol}

    def check_boundary(self):
        if read_bound_json(self.configuration_ref) != self.config:
            raise ValueError("TASK_V2_EXPERIENCE_CONFIG_CHANGED")
        if read_learning_request(self.learning_request_ref) != self.learning_request:
            raise ValueError("TASK_V2_LEARNING_REQUEST_CHANGED")
        if self._read_trial_binding() != self.trial_binding:
            raise ValueError("TASK_V2_TRIAL_BINDING_CHANGED")
        if self.snapshot is not None:
            self.memory.check_snapshot(self.snapshot)
        if self.extraction_episode_set_ref is not None:
            frozen = self.memory.read(self.extraction_episode_set_ref)
            _learning_require(frozen.get("kind") == "episode_set" and frozen.get("frozen") is True
                and frozen.get("visibility") == "public" and frozen.get("data_use") == "strategy_learning",
                "FROZEN_EPISODE_SET_REQUIRED")
            bound = frozen.get("learning_request_ref")
            _learning_require(self.learning_request_ref is None or bound is None or self.learning_request_ref == bound,
                "LEARNING_REQUEST_CONFLICT")
            read_learning_request(bound)
            _learning_require(self.requested_strategy_scope is None or self.requested_strategy_scope == frozen["applicability"],
                "FROZEN_STRATEGY_SCOPE_CHANGED")

    @staticmethod
    def context_with_sources(session, current_task_state=None):
        kinds = {kind for row in session.public_checks
            if (kind := public_counterexample_kind(row["report"])) is not None}
        if getattr(session, "actual_calls", None):
            kinds.update(row["counterexample_kind"] for row in _experiment_evidence(session) if row["counterexample_kind"] is not None)
        # 为什么要有它：只按已执行公开证据区分适用域，编译策略不能仅因同属C#就投放到业务反例；混合来源单独保留。
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed_public_counterexamples" if kinds else "none"
        process, sources = _process_context(session, current_task_state)
        return {"runtime": "dotnet", "target_domain": "csharp_repository",
            "has_public_counterexample": bool(kinds), "counterexample_kind": kind, **process}, sources

    @staticmethod
    def context(session, current_task_state=None):
        return TaskAgentV2Experience.context_with_sources(session, current_task_state)[0]

    def prepare_decision_request(self, request, session):
        self.check_boundary()
        current = next((message.content["current_task_state"] for message in reversed(request.messages)
            if message.role == "user" and isinstance(message.content, dict)
            and isinstance(message.content.get("current_task_state"), dict)
            and message.content["current_task_state"].get("origin") == "execution_host"), None)
        context, sources = self.context_with_sources(session, current)
        self.owner._v2_strategy_context_sources = deepcopy(sources)
        if self.snapshot is None:
            return request
        selection = self.memory.select(self.snapshot, context)
        view = self.memory.advisory_model_view(selection)
        selection_ref = self.owner._save("experience-selection-" + uuid4().hex + ".json", {
            "configuration_ref": self.configuration_ref, "selection": selection,
            "context_sources": sources, "model_view": view, "model_view_sha256": canonical_hash(view)})
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
        completed_records = {row["call_id"]: row for row in execution["completed_tool_records"]}
        _learning_require(len(completed_records) == len(execution["completed_tool_records"]), "DUPLICATE_COMPLETED_ACTION")
        completed_order = {call_id: index for index, call_id in enumerate(completed_records)}
        completed = set(completed_records)
        failures = []
        for index, checked in enumerate(session.public_checks):
            report = checked["report"]
            ref = _public_reference(report["report_ref"])
            original = read_bound_json(ref)
            if any(original.get(key) != report.get(key) for key in ("status", "observations", "verification_summary", "candidate_source_id")):
                raise ValueError("TASK_V2_LEARNING_REPORT_CHANGED")
            evidence_id = "public_check_" + str(index)
            sources.append({"kind": "json_file", "evidence_id": evidence_id, **ref,
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            for field in ("status", "observations", "verification_summary", "candidate_ref", "candidate_source_id"):
                if field not in original:
                    continue
                selections.append({"evidence_id": evidence_id, "locator": [field], "role": "public_check"})
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
            selections.extend({"evidence_id": evidence_id, "locator": [field], "role": "public_experiment"} for field in (
                "observations", "verification_scope", "grants_acceptance"))
            if experiment["counterexample_kind"] is not None:
                gains.append("public_counterexample")
                counterexamples.append({"kind": experiment["counterexample_kind"], "evidence_id": evidence_id,
                    "source_identity": experiment["source_identity"], "scope": "exploration_only"})
        _learning_require(completed <= {call["call_id"] for call in session.actual_calls}, "COMPLETED_ACTION_MISSING")
        for call in session.actual_calls:
            if call["call_id"] not in completed:
                continue
            index = completed_order[call["call_id"]]
            _learning_require(all(completed_records[call["call_id"]].get(key) == value
                for key, value in call.items()), "COMPLETED_ACTION_CHANGED")
            ref = call["response_ref"]
            _response, index_in_response, action_receipt = _validated_action(call, session)
            evidence_id = "public_action_" + str(index)
            sources.append({"kind": "json_file", "evidence_id": evidence_id, **ref,
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            selections.extend({"evidence_id": evidence_id, "locator": ["tool_calls", index_in_response, field],
                "role": "action_request", "action_index": index}
                for field in ("name", "arguments"))
            selections.append({"evidence_id": evidence_id, "locator": ["content"],
                "role": "model_statement", "action_index": index})
            if "working_state" in _response["tool_calls"][index_in_response]["arguments"]:
                # 为什么要有它：同一次动作中的结构化声明也是明确表达；原字典、空值及非法声明均保留，不代写判断。
                selections.append({"evidence_id": evidence_id,
                    "locator": ["tool_calls", index_in_response, "arguments", "working_state"],
                    "role": "model_statement", "action_index": index})
            result_id = evidence_id + "_result"
            sources.append({"kind": "json_file", "evidence_id": result_id, **call["execution_receipt_ref"],
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            # 为什么要有它：请求过某工具不等于实验已执行；只投影实际返回字段，不补写模型未表达的技术判断。
            for field in ("status", "executed", "grants_acceptance", "reason", "path", "text", "matches",
                    "candidate_ref", "candidate_source_id", "source_view_ref", "technical_plan_ref", "plan_ref",
                    "parent_revision_ref", "source_revision_ref", "file_sha256", "root_id", "check_ref",
                    "error", "error_code", "code", "message", "argument_errors", "correctable", "execution_origin",
                    "observations", "verification_summary", "failed_checks", "feedback_ref", "truncated",
                    "offset", "next_offset", "total_chars"):
                if field in action_receipt["output"]:
                    selections.append({"evidence_id": result_id, "locator": ["output", field],
                        "role": "action_return", "action_index": index})
            for field in ("current_revision_ref", "binding_registry_ref", "technical_plan_ref"):
                if field in action_receipt.get("task_head", {}):
                    selections.append({"evidence_id": result_id, "locator": ["task_head", field],
                        "role": "source_state", "action_index": index})
        # 为什么要有它：候选就绪、未知副作用和任务交付是不同事实；仅选原生技术终态，不引入隐藏评分或开发者结论。
        sources.append({"kind": "json_file", "evidence_id": "technical_outcome",
            **_public_reference(self.owner.execution_receipt().execution_ref),
            "visibility": "public", "allowed_uses": ["strategy_learning"]})
        selections.extend({"evidence_id": "technical_outcome", "locator": [field], "role": "technical_outcome"}
            for field in ("status", "candidate_ref", "check_ref", "unknown_effects",
                "native_terminal", "business_result", "stop_attribution", "collector_scope") if field in execution)
        semantic_validation = _semantic_validation_evidence(self.owner, session, execution)
        if semantic_validation is not None:
            sources.append({"kind": "json_file", "evidence_id": "semantic_validation", **semantic_validation,
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            selections.extend({"evidence_id": "semantic_validation", "locator": [field], "role": "technical_outcome"}
                for field in ("events", "source_checkpoint_ref", "source_execution_ref", "validation_scope", "authority_changed"))
        if execution.get("schema") == "task-agent-public-process-observation.v1":
            checkpoint_path = Path(session.state.store_root) / execution["checkpoint_ref"]
            lineage = {"process_stop": execution["stop_ref"], "process_source": execution["process_ref"],
                "process_checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": checkpoint_path.stem}}
            if execution.get("plan_ref") is not None:
                lineage["process_plan"] = execution["plan_ref"]
            # 为什么要有它：投影只发公开字段，但原停止记录及检查点仍须随策略被复核，不能用一份复制摘要替代来源绑定。
            sources.extend({"kind": "json_file", "evidence_id": name, **_public_reference(reference),
                "visibility": "public", "allowed_uses": ["strategy_learning"]} for name, reference in lineage.items())
        if execution.get("loop_result_ref"):
            loop_ref = _public_reference(execution["loop_result_ref"])
            loop = read_bound_json(loop_ref)
            sources.append({"kind": "json_file", "evidence_id": "public_loop", **loop_ref,
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
            # 为什么要有它：未派发的参数错误也是实际试错经历，但必须独立列出，不能混成执行过的修改或实验。
            for call_id, record in loop.get("call_records", {}).items():
                if record.get("status") == "not_dispatched":
                    selections.extend({"evidence_id": "public_loop", "locator": ["call_records", call_id, field],
                        "role": "nonexecuted_attempt"} for field in ("name", "status", "result"))
        # 为什么要有它：无候选、重复调查和未派发拒绝也是实际过程经历；不得为启用学习补造业务反例或假设被推翻。
        if not gains and (completed or any(row["role"] == "nonexecuted_attempt" for row in selections)
                or execution.get("schema") == "task-agent-public-process-observation.v1"):
            gains.append("public_process_observation")
        if not gains:
            return None
        applicability, applicability_sources = self.context_with_sources(session)
        for key in ("task_contract_ref", "current_revision_ref", "working_state_ref", "current_check_ref"):
            reference = applicability_sources.get(key)
            if reference is None:
                continue
            path = Path(session.state.store_root) / reference
            sources.append({"kind": "json_file", "evidence_id": "context_" + key,
                **_public_reference({"path": str(path.resolve()), "sha256": path.stem}),
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
        episode = {
            "schema_version": "public_strategy_episode_v1", "visibility": "public", "data_use": "strategy_learning",
            "projection_version": "ordered_public_actions_v1",
            "information_gain": sorted(set(gains)), "evidence_refs": sources, "selections": selections,
            "public_counterexamples": counterexamples,
            "applicability": applicability, "applicability_sources": applicability_sources,
            "task_contract_ref": session.state.contract_ref,
            "task_context_selections": task_context_selections(session.state.contract, "context_task_contract_ref"),
            "run_id": execution.get("run_id"), "request_id": execution.get("request_id"),
            "execution_ref": deepcopy(self.owner.execution_receipt().execution_ref)}
        name = "public-learning-episode-" + canonical_hash(episode) + ".json"
        try:
            return self.owner._save(name, episode)
        except FileExistsError:
            from src.mod_agent.task_agent_v2_recovery import _ref

            reference = _ref(Path(self.owner.output_root) / name)
            _learning_require(read_bound_json(reference) == episode, "SEALED_EPISODE_CHANGED")
            return reference

    def seal_public_episode(self, session=None):
        """Capture immutable public facts without spending model resources.

        为什么要有它：保存经历与调用提炼模型是两件事，停用提炼或人工停止不能抹去已经发生的过程。
        """
        return seal_current_public_episode(self.owner, session, configuration_ref=self.configuration_ref)

    def complete_terminal_learning(self):
        if self._terminal_receipt is not None:
            return deepcopy(self._terminal_receipt)
        owner = self.owner
        before = validate_budget(owner.budget_snapshot())
        result = {"status": "not_enabled", "configuration_ref": self.configuration_ref,
            "budget_before": before, "model_calls": 0, "known_usage": [], "unknown_usage_calls": 0}
        try:
            self.check_boundary()
            episode = self.seal_public_episode()
            if episode is not None:
                result["episode_ref"] = episode
            registration = getattr(owner, "_public_episode_catalog_registration", None)
            if registration is not None:
                result["catalog_registration"] = deepcopy(registration)
                _learning_require(registration["status"] == "registered", "CATALOG_REGISTRATION_FAILED")
            if self.config["extract_on_terminal"]:
                if episode is None and self.extraction_episode_set_ref is None and self.episode_catalog is None:
                    result["status"] = "no_information_gain"
                else:
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
                        owner.require_model_send_window("strategy_learning")
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

                    extraction = self.memory.propose_from_episode
                    extraction_source = episode
                    if self.episode_catalog is not None:
                        selection = deepcopy(self.episode_catalog.get("selection", {}))
                        required = selection.setdefault("required_episode_refs", [])
                        if episode is not None and episode not in required:
                            required.append(deepcopy(episode))
                        selected = self.memory.freeze_catalog(self.episode_catalog["catalog_id"],
                            applicability=self.requested_strategy_scope, learning_request_ref=self.learning_request_ref,
                            selection=selection)
                        extraction, extraction_source = self.memory.propose_from_episodes, asdict(selected)
                        frozen = self.memory.read(selected)
                        result.update(episode_set_ref=deepcopy(extraction_source),
                            catalog_selection_ref=deepcopy(frozen["catalog_selection_ref"]))
                    elif self.extraction_episode_set_ref is not None:
                        extraction, extraction_source = self.memory.propose_from_episodes, self.extraction_episode_set_ref
                        result["episode_set_ref"] = deepcopy(extraction_source)
                    learned = extraction(
                        SimpleNamespace(provider=client.provider, model=client.model, complete_turn=complete), extraction_source,
                        admission_ref=admission, before_call=reserve, after_call=account,
                        learning_request_ref=self.learning_request_ref, applicability=self.requested_strategy_scope)
                    result.update(status=learned["status"], strategy_ref=learned.get("strategy_ref"),
                        extraction_ref=learned.get("extraction_ref"),
                        **{key: deepcopy(learned[key]) for key in ("learning_request_ref", "source_assessments",
                            "evidence_gaps", "assessment_kind", "unassessed_episode_ids", "gap_completeness") if key in learned})
        except Exception as error:
            result.update(status="failed", error_type=type(error).__name__, message=str(error))
            if result["model_calls"] > len(result["known_usage"]) + result["unknown_usage_calls"]:
                result["unknown_usage_calls"] = result["model_calls"] - len(result["known_usage"])
        result["budget_after"] = validate_budget(owner.budget_snapshot())
        result["original_execution_ref"] = deepcopy(owner.execution_receipt().execution_ref) if owner.execution_receipt() else None
        result["task_timing_ref"] = owner.observe_task_boundary("terminal_learning_return", status=result["status"])
        reference = owner._save("terminal-learning-receipt-" + uuid4().hex + ".json", result)
        self._terminal_receipt = {**result, "receipt_ref": reference}
        return deepcopy(self._terminal_receipt)
