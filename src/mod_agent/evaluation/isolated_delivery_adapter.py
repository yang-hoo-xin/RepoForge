"""Generic compile-only delivery through the existing delegation and continuation APIs."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.build import BuildAdapter, BuildCommandResult, BuildProfile
from src.mod_agent.build.nuget import (
    _create_nuget_isolated_build_profile,
    _run_nuget_isolated_build,
)
from src.mod_agent.evaluation.product_trial_batch import _read_reference
from src.mod_agent.evaluation.trial_comparison_identity import _bound_input
from src.mod_agent.execution.task_delegation import prepare_task_delegation
from src.mod_agent.reviewed_patch_continuation import (
    continue_reviewed_patch,
    resume_deferred_approval,
)
from src.mod_agent.workflow.interrupts import ResumePayload


def _ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _inside(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("DELIVERY_PATH_OUTSIDE_RUN")
    return path


def _same_ref(left, right):
    return (isinstance(left, dict) and isinstance(right, dict)
        and Path(left["path"]).resolve() == Path(right["path"]).resolve() and left["sha256"] == right["sha256"])


def _pretransport_delivery_binding(root, initial, initial_ref, manifest_ref):
    """Require the new segment's claim, cumulative budget and finished execution proof.

    为什么要有它：发送前恢复产生新模型代码，不能借零调用恢复分支或旧源码启动证明进入交付。
    """
    from src.mod_agent.interrupted_patch_continuation import read_interrupted_patch_execution
    from src.mod_agent.model_call_budget import validate_budget
    from src.mod_agent.model_qualification import read_bound_json, canonical_hash
    from src.mod_agent.pretransport_patch_admission import validate_selection
    selected = initial.get("parent_selection")
    validate_selection(selected)
    parent_ref = _ref(root / "run_result.json")
    parent = read_bound_json(parent_ref)
    parent_budget = validate_budget(parent["model_call_budget"])
    budget = validate_budget(initial["model_call_budget"])
    segment = _inside(root, initial["continuation_root"])
    if (initial_ref != _ref(segment / "run_result.json")
            or initial != read_interrupted_patch_execution(root, pretransport=True)
            or initial.get("entry") != "pretransport-patch-continuation"
            or initial.get("generation_origin") != "new_patch_after_verified_pretransport_failure"
            or initial.get("run_id") != root.name or initial.get("parent_run_id") != root.name
            or initial.get("parent_manifest_ref") != manifest_ref or selected["parent_manifest_ref"] != manifest_ref
            or initial.get("parent_result_ref") != parent_ref or selected["parent_result_ref"] != parent_ref
            or initial.get("request_id") != read_bound_json(manifest_ref)["request_id"]
            or budget["maximum"] != parent_budget["maximum"] or budget.get("policy") != parent_budget.get("policy")
            or budget["consumed"] != initial.get("model_call_count")
            or initial.get("prior_model_call_count") != parent_budget["consumed"]
            or initial.get("current_invocation_model_call_count") != budget["consumed"] - parent_budget["consumed"]
            or initial["current_invocation_model_call_count"] <= 0
            or set(budget["by_stage"]) != set(parent_budget["by_stage"])
            or any(row["consumed"] < parent_budget["by_stage"][role]["consumed"] for role, row in budget["by_stage"].items())
            or initial.get("history_rewritten") is not False or initial.get("actual_write_performed") is not False):
        raise ValueError("DELIVERY_PRETRANSPORT_PARENT_OR_BUDGET_CHANGED")
    scope_ref = initial["continuation_scope_ref"]
    if Path(scope_ref["path"]).resolve() != segment / "scope.json":
        raise ValueError("DELIVERY_PRETRANSPORT_SCOPE_CHANGED")
    scope = read_bound_json(scope_ref)
    claim = read_bound_json(_ref(root / "continuations" / ("pretransport-" + selected["source_attempt_ref"]["sha256"][:24] + ".json")))
    if (scope.get("selection") != selected or scope.get("inherited_budget") != parent_budget
            or claim.get("scope_sha256") != canonical_hash(scope) or claim.get("result_ref") != initial_ref):
        raise ValueError("DELIVERY_PRETRANSPORT_SCOPE_CHANGED")
    check = read_bound_json(initial["execution_version_check_ref"])
    snapshot = read_bound_json(initial["execution_version_ref"])
    if (initial.get("execution_version_status") != "verified" or check.get("status") != "verified"
            or check.get("phase") != "finished" or check.get("run_id") != root.name
            or check.get("request_id") != initial["request_id"]
            or check.get("execution_version_ref") != initial["execution_version_ref"]
            or Path(snapshot["selected_run_root"]).resolve() != segment
            or snapshot.get("business_argv", [None])[0] != "continue-pretransport-patch"):
        raise ValueError("DELIVERY_PRETRANSPORT_EXECUTION_PROOF_REQUIRED")
    return scope


def _compilation_feedback_binding(root, report, proposal, verification, store, manifest):
    """Validate existing executed compilation evidence without inventing behavior checks."""
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch, PatchApplyError

    full = report["validation_results"]["full_compilation"]
    changed = full.get("changed", {})
    actual = json.loads(_read_reference(changed["report_ref"], root=root))
    execution, build = changed.get("build_execution", {}), changed.get("build_result", {})
    if (actual != {key: value for key, value in changed.items() if key != "report_ref"}
            or full.get("status") != "passed" or full.get("changed_compilation_performed") is not True
            or changed.get("status") != "passed" or changed.get("execution_origin") != "executed"
            or changed.get("all_required_files_compiled") is not True
            or execution.get("build_executed") is not True
            or any(execution.get(key) != 0 for key in ("restore_exit_code", "build_exit_code", "exit_code"))
            or build.get("real_build_performed") is not True or build.get("claims_real_build_success") is not True
            or build.get("exit_code") != 0 or changed.get("product_apply_performed") is not False
            or changed.get("verification_scope") != "full_project_compilation_not_behavior_or_formal_build_or_runtime"):
        raise ValueError("DELIVERY_EXECUTED_COMPILATION_REQUIRED")
    model_ref = changed["model_response_ref"]
    raw = _read_reference(model_ref, root=root)
    if {row["generation"]["model_response_sha256"] for row in proposal["diff_intent"]} != {model_ref["sha256"]}:
        raise ValueError("DELIVERY_CURRENT_MODEL_RESPONSE_MISMATCH")
    digest = report.get("generated_changes_sha256")
    if (not isinstance(digest, str) or re.fullmatch("[a-f0-9]{64}", digest) is None
            or not isinstance(report.get("source_snapshot_sha256"), str)
            or re.fullmatch("[a-f0-9]{64}", report["source_snapshot_sha256"]) is None
            or any(not any(row.get("report_ref") == verification and row.get("status") == "passed"
                    and row.get("model_response_sha256") == model_ref["sha256"]
                    and row.get("generated_changes_sha256") == digest
                    for row in change["generation"].get("patch_agent_loop", {}).get("behavior_evaluation_refs", []))
                for change in proposal["diff_intent"])):
        raise ValueError("DELIVERY_COMPILED_CHANGE_BINDING_MISMATCH")
    plan_ref = ArtifactRef(**changed["implementation_plan_ref"])
    _inside(root, plan_ref.path)
    if plan_ref.artifact_type != "implementation_plan":
        raise ValueError("DELIVERY_COMPILED_PLAN_REQUIRED")
    plan = store.semantic_read_immutable(plan_ref)["data"]
    try:
        materialized = materialize_exact_text_patch(workspace_root=root,
            diff_intent=proposal["diff_intent"], implementation_plan=plan)
    except PatchApplyError as error:
        raise ValueError("DELIVERY_COMPILED_SOURCE_OR_PATCH_CHANGED") from error
    identity = {"baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256,
                "files": sorted(materialized.files)}
    if identity != report.get("materialized_code_identity") or identity != changed.get("materialized_code_identity"):
        raise ValueError("DELIVERY_COMPILED_CODE_IDENTITY_MISMATCH")
    # 为什么要有它：只读重建当前Proposal并核对真实编译输入，旧DLL或同名报告不能替代当前模型补丁验收。
    target = _inside(root, manifest["nuget_isolated_build_profile"]["working_directory"])
    compiled = _inside(root, changed["isolated_source_root"])
    inputs = json.loads(_read_reference(changed["source_inputs_ref"], root=root))
    expected = {str((root / name).relative_to(target).as_posix()): hashlib.sha256(data).hexdigest()
                for name, data in materialized.files.items()}
    if not inputs or any(inputs.get(name) != digest for name, digest in expected.items()):
        raise ValueError("DELIVERY_COMPILED_SOURCE_INPUT_MISMATCH")
    for name, digest in inputs.items():
        source = _inside(target, name)
        candidate = _inside(compiled, name)
        if (_ref(candidate)["sha256"] != digest or (name not in expected and _ref(source)["sha256"] != digest)):
            raise ValueError("DELIVERY_COMPILED_SOURCE_INPUT_MISMATCH")
    _read_reference(changed["compile_inputs_ref"], root=root)
    required = changed.get("required_files", [])
    observed = {row["path"]: row["sha256"] for row in changed.get("compile_inputs", [])}
    if not required or any(observed.get(name) != inputs.get(name) for name in required):
        raise ValueError("DELIVERY_COMPLETE_COMPILE_INPUTS_REQUIRED")
    _read_reference(changed["built_artifact_ref"], root=root)
    for key in ("stdout_ref", "stderr_ref"):
        _read_reference(changed[key], root=root)
    return {"model_response_sha256": model_ref["sha256"], "model_response_ref": model_ref,
        "compilation_report_ref": changed["report_ref"], "built_artifact_ref": changed["built_artifact_ref"],
        "generated_changes_sha256": report["generated_changes_sha256"], "source_snapshot_sha256": report["source_snapshot_sha256"],
        "materialized_code_identity": identity, "verification_scope": "compilation_only_gameplay_not_verified"}


def _task_agent_verifier_binding(*, root, initial, manifest, snapshot, worker_ref, worker,
        verification, report, proposal, execution_check_ref):
    """Recheck the distinct native-tool origin; never manufacture a Patch worker receipt.

    为什么要有它：连续执行器的候选必须绑定真实工具响应和最终计划，不能借旧worker记录格式绕过同一补丁的正式验收。
    """
    from src.mod_agent.cli import _task_agent_edit_paths, build_parser
    from src.mod_agent.proposal.builder import PatchProposalBuilder
    from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
    from src.mod_agent.task_agent_execution import (
        load_task_agent_outcome,
        validate_task_agent_handoff,
    )

    parsed = build_parser().parse_args(snapshot["business_argv"])
    if (getattr(parsed, "technical_executor", None) != "task_agent"
            or parsed.constraint_policy != "source_priority_v1" or not parsed.local_ollama
            or not parsed.defer_approval or parsed.run_gtfo_runtime):
        raise ValueError("DELIVERY_TASK_AGENT_EXECUTION_NOT_DECLARED")
    paths = _task_agent_edit_paths(parsed.task_agent_edit_paths)
    declaration = manifest.get("technical_execution", {})
    if (declaration.get("technical_executor") != "task_agent"
            or declaration.get("constraint_policy") != parsed.constraint_policy
            or declaration.get("candidate_mode", "baseline") != parsed.task_agent_candidate_mode
            or declaration.get("allowed_edit_paths_relative_to_project") != list(paths)):
        raise ValueError("DELIVERY_TASK_AGENT_SCOPE_CHANGED")
    target = _inside(root, manifest["nuget_isolated_build_profile"]["working_directory"])
    allowed = tuple(_inside(target, path).relative_to(root).as_posix() for path in paths)
    reference = initial.get("task_agent_execution_ref")
    lineage = proposal.get("context_pack", {}).get("task_agent_lineage", {})
    if (lineage.get("execution_ref") != reference or lineage.get("approval_inherited") is not False
            or lineage.get("whole_task_delivered") is not False or initial.get("patch_execution_receipts")):
        raise ValueError("DELIVERY_TASK_AGENT_ORIGIN_REQUIRED")
    store = ArtifactStore(root / "a")
    outcome = load_task_agent_outcome(reference, artifact_store_root=store.root)
    execution = json.loads(_read_reference(reference, root=root))
    claim = json.loads(_read_reference(outcome.claim_ref, root=root))
    if (execution.get("run_id") != initial.get("run_id") or execution.get("request_id") != manifest.get("request_id")
            or execution.get("policy") != asdict(PatchAgentLoopPolicy(**worker["policy"]))
            or outcome.budget_after != initial.get("model_call_budget")
            or initial.get("task_agent_outcome") != outcome.to_dict()
            or lineage.get("candidate_ref") != outcome.candidate_ref
            or lineage.get("check_ref") != outcome.check_ref
            or claim.get("initial_context") != outcome.initial_context.to_dict()):
        raise ValueError("DELIVERY_TASK_AGENT_EXECUTION_BINDING_CHANGED")
    handoff = validate_task_agent_handoff(outcome, store=store, workspace_root=root,
        expected_initial_context=outcome.initial_context, allowed_edit_paths=allowed,
        worker_configuration_ref=worker_ref,
        expected_verification_identity=execution["expected_verification_identity"],
        expected_candidate_mode=parsed.task_agent_candidate_mode)
    if (handoff.check["report"]["report_ref"] != verification
            or [PatchProposalBuilder._change_intent(change) for change in handoff.changes] != proposal["diff_intent"]
            or report.get("implementation_plan_ref") != handoff.final_context.to_dict()["implementation_plan_ref"]
            or proposal["context_pack"].get("implementation_plan_hash") != handoff.final_context.implementation_plan_ref.artifact_hash):
        raise ValueError("DELIVERY_TASK_AGENT_PROPOSAL_CHANGED")
    return {"source": "caller_bound_factory_and_native_task_tool_execution",
        "validator_sha256": report["verification_identity"]["validator_sha256"],
        "factory_ref": worker["behavior_evaluator"]["factory_ref"], "worker_configuration_ref": worker_ref,
        "execution_version_check_ref": execution_check_ref, "task_agent_execution_ref": reference,
        "candidate_ref": outcome.candidate_ref, "check_ref": outcome.check_ref}


def _bound_verifier(context, root, initial, manifest, verification, report, proposal, compilation_binding=None):
    pretransport = initial.get("entry") == "pretransport-patch-continuation"
    snapshot_ref = initial["execution_version_ref"] if pretransport else manifest["execution_version_ref"]
    snapshot_root = Path(snapshot_ref["path"]).resolve().parent if pretransport else Path(context["trial"]["snapshot_root"]).resolve()
    snapshot = json.loads(_read_reference(snapshot_ref, root=snapshot_root))
    check_ref = initial["execution_version_check_ref"] if pretransport else manifest["execution_version_check_ref"]
    check = json.loads(_read_reference(check_ref, root=snapshot_root))
    if (check.get("status") != "verified" or check.get("run_id") != initial["run_id"]
            or check.get("execution_version_ref") != snapshot_ref):
        raise ValueError("DELIVERY_VERIFIED_FROZEN_EXECUTION_REQUIRED")
    if pretransport:
        from src.mod_agent.model_qualification import read_bound_json
        scope = read_bound_json(initial["continuation_scope_ref"])
        worker_ref = scope["worker_configuration_ref"]
        if not any(row.get("role") == "bound_input" and _same_ref(worker_ref, row) for row in snapshot["external_refs"]):
            raise ValueError("DELIVERY_BOUND_WORKER_CONFIGURATION_REQUIRED")
        worker = read_bound_json(worker_ref)
    else:
        worker = _bound_input(snapshot, snapshot_root, "--patch-worker-config")
        if worker is None or not isinstance(worker.get("behavior_evaluator"), dict):
            raise ValueError("DELIVERY_BOUND_WORKER_CONFIGURATION_REQUIRED")
        argv = snapshot["business_argv"]
        worker_path = Path(argv[argv.index("--patch-worker-config") + 1]).resolve()
        worker_ref = next({"path": row["path"], "sha256": row["sha256"]} for row in snapshot["external_refs"]
            if row["role"] == "bound_input" and Path(row["path"]).resolve() == worker_path)
    selected = worker["behavior_evaluator"]
    factory = selected["factory_ref"]
    if (not any(_same_ref(worker_ref, row) for row in context["trial"].get("bound_inputs", []))
            or not any(_same_ref(factory, row) for row in context["trial"].get("extra_load_refs", []))
            or not any(row.get("role") == "extra_python_load" and row.get("execution_observed") is True
                and _same_ref(factory, row) for row in check.get("external_bindings", []))):
        raise ValueError("DELIVERY_CALLER_BOUND_EXECUTED_FACTORY_REQUIRED")
    if initial.get("task_agent_execution_ref") is not None:
        if pretransport or compilation_binding is not None:
            raise ValueError("DELIVERY_TASK_AGENT_MODE_UNSUPPORTED")
        # Scope: native provenance is checked by its exact tool/candidate/report chain.
        # 为什么要有它：共同factory校验之后必须按真实来源分派，不能要求native候选伪造旧worker的generation字段。
        return _task_agent_verifier_binding(root=root, initial=initial, manifest=manifest,
            snapshot=snapshot, worker_ref=worker_ref, worker=worker, verification=verification,
            report=report, proposal=proposal, execution_check_ref=check_ref)
    expected_kwargs = hashlib.sha256(json.dumps(selected["kwargs"], sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    response_hashes = {row["generation"]["model_response_sha256"] for row in proposal["diff_intent"]}
    response_sha = compilation_binding["model_response_sha256"] if compilation_binding is not None else report.get("model_response_sha256")
    if response_hashes != {response_sha}:
        raise ValueError("DELIVERY_CURRENT_MODEL_RESPONSE_MISMATCH")
    matches = []
    for reference in initial.get("patch_execution_receipts", []):
        receipt = json.loads(_read_reference({key: reference[key] for key in ("path", "sha256")}, root=root))
        factory_receipt = receipt.get("factory_receipt", {})
        configuration = factory_receipt.get("behavior_evaluator_configuration", {})
        if (not _same_ref(configuration.get("configuration_ref"), worker_ref)
                or not _same_ref(configuration.get("factory_ref"), factory)
                or configuration.get("factory_name") != selected["factory_name"]
                or configuration.get("kwargs_sha256") != expected_kwargs):
            continue
        for evaluation in factory_receipt.get("patch_agent_loop", {}).get("behavior_evaluations", []):
            if (evaluation.get("report_ref") == verification and evaluation.get("admission") == "accepted"
                    and evaluation.get("execution_origin") == "executed"
                    and evaluation.get("model_response_sha256") == response_sha
                    and evaluation.get("verification_identity") == report.get("verification_identity")):
                if compilation_binding is not None and (evaluation.get("callback_invoked") is not True
                        or evaluation.get("status") != "passed"
                        or any(evaluation.get(key) != compilation_binding[key] for key in
                            ("generated_changes_sha256", "source_snapshot_sha256", "materialized_code_identity"))
                        or evaluation.get("validation_results") != report.get("validation_results")):
                    continue
                matches.append(reference)
    if not matches:
        raise ValueError("DELIVERY_EXECUTED_WORKER_VERIFIER_BINDING_REQUIRED")
    return {"source": "caller_bound_factory_and_verified_execution_receipt",
        "validator_sha256": report["verification_identity"]["validator_sha256"],
        "factory_ref": factory, "worker_configuration_ref": worker_ref, "execution_version_check_ref": check_ref,
        "worker_receipt_refs": [{key: row[key] for key in ("path", "sha256")} for row in matches],
        **({"compilation": compilation_binding} if compilation_binding is not None else {})}


def prepare_delivery(context):
    """Prepare existing records/profile without approval, Apply, Build or model calls.

    为什么要有它：选择依据来自本次真实报告与原构建配置，不应为每个运行复制脚本或跳过尚未支持的验收。
    """
    root = Path(context["run_root"]).resolve(strict=True)
    config = context["trial"]["delivery"]
    if not isinstance(config.get("policy_id"), str) or not config["policy_id"].strip():
        raise ValueError("DELIVERY_CALLER_POLICY_ID_REQUIRED")
    verifier_sha = config.get("verifier_sha256")
    bound_verifier = config.get("verifier_selection") == "bound_worker_execution"
    if not bound_verifier and (not isinstance(verifier_sha, str) or re.fullmatch("[a-f0-9]{64}", verifier_sha) is None):
        raise ValueError("DELIVERY_DECLARED_VERIFIER_SHA_REQUIRED")
    manifest_ref = _ref(root / "run_manifest.json")
    initial_ref = config.get("initial_result_ref") or _ref(root / "run_result.json")
    initial, manifest = (json.loads(_read_reference(ref, root=root)) for ref in (initial_ref, manifest_ref))
    if config.get("initial_result_ref") is not None:
        parent_result = json.loads(_read_reference(_ref(root / "run_result.json"), root=root))
        from src.mod_agent.model_call_budget import validate_budget
        validate_budget(parent_result.get("model_call_budget"))
        # 为什么要有它：续接结果另存而不覆盖首次失败；交付仍须绑定原任务、原manifest和本次真实checkpoint。
        if initial.get("entry") == "pretransport-patch-continuation":
            _pretransport_delivery_binding(root, initial, initial_ref, manifest_ref)
        elif initial.get("entry") == "retained-task-agent-handoff":
            from src.mod_agent.retained_task_agent_handoff import validate_retained_handoff_result

            # Scope: native candidate recovery has its own exact history and approval checkpoint.
            # 为什么要有它：新连续执行器不应冒充旧worker回执；零模型恢复也必须核对真实完成、原失败和本次checkpoint。
            recovered = validate_retained_handoff_result(run_root=root, result_ref=initial_ref)
            if recovered != initial:
                raise ValueError("DELIVERY_RETAINED_TASK_HANDOFF_CHANGED")
        elif (initial.get("parent_run_id") != root.name
                or initial.get("parent_manifest_ref") != manifest_ref
                or initial.get("parent_result_ref") != _ref(root / "run_result.json")
                or initial.get("request_id") != manifest.get("request_id")
                or initial.get("generation_origin") != "retained_live_response"
                or initial.get("new_provider_sends") != 0
                or initial.get("model_call_budget") != parent_result.get("model_call_budget")
                or initial.get("model_call_count") != parent_result.get("model_call_count")):
            raise ValueError("DELIVERY_CONTINUATION_PARENT_BINDING_REQUIRED")
    elif initial.get("run_id") != root.name:
        raise ValueError("DELIVERY_DEFERRED_PRODUCT_RESULT_REQUIRED")
    if initial.get("status") != "awaiting_approval":
        raise ValueError("DELIVERY_DEFERRED_PRODUCT_RESULT_REQUIRED")
    if (manifest.get("objective_verification_command") or manifest.get("compiled_artifact_inspection_profile")
            or (manifest.get("runtime_admission") or {}).get("requested") is True):
        raise ValueError("DELIVERY_MISSING_SUPPORTED_OBJECTIVE_OR_RUNTIME_ADAPTER")
    checkpoint = ArtifactRef(**initial["workflow_checkpoint"])
    _inside(root, checkpoint.path)
    store = ArtifactStore(root / "a")
    saved = store.semantic_read_immutable(checkpoint)["data"]
    repository_binding = manifest.get("repository_profile")
    if repository_binding is not None:
        from src.mod_agent.domain.csharp_repository_delivery import (
            repository_binding_from_request,
            validate_repository_binding,
        )
        from src.mod_agent.product_execution import ProductExecutionRequestV1
        original_request = ProductExecutionRequestV1.from_dict(json.loads(
            saved["state_snapshot"]["runtime_summary"]["deferred_product_request_json"]))
        if repository_binding_from_request(original_request) != repository_binding:
            raise ValueError("DELIVERY_REPOSITORY_CHECKPOINT_BINDING_CHANGED")
        validate_repository_binding(repository_binding, root)
    state = saved["state_snapshot"]
    proposal_ref = ArtifactRef(feature_id=initial["feature_id"], artifact_type="patch_proposal",
        path=state["artifacts"]["patch_proposal"], artifact_hash=state["artifact_hashes"]["patch_proposal"])
    proposal = store.semantic_read_immutable(proposal_ref)["data"]
    verification = config.get("verification_report_ref")
    if verification is None:
        if config.get("verification_selection") != "current_proposal":
            raise ValueError("DELIVERY_EXPLICIT_VERIFICATION_SELECTION_REQUIRED")
        selected = {}
        lineage = proposal.get("context_pack", {}).get("task_agent_lineage")
        if lineage is not None:
            if not bound_verifier or lineage.get("execution_ref") != initial.get("task_agent_execution_ref"):
                raise ValueError("DELIVERY_TASK_AGENT_ORIGIN_REQUIRED")
            checked = json.loads(_read_reference(lineage["check_ref"], root=root))
            ref = checked["report"]["report_ref"]
            selected[(ref["path"], ref["sha256"])] = ref
        for change in proposal["diff_intent"]:
            generation = change["generation"]
            for observation in generation.get("patch_agent_loop", {}).get("behavior_evaluation_refs", []):
                if observation.get("status") == "passed" and observation.get("model_response_sha256") == generation["model_response_sha256"]:
                    ref = observation["report_ref"]
                    selected[(ref["path"], ref["sha256"])] = ref
        if len(selected) != 1:
            raise ValueError("DELIVERY_CURRENT_PROPOSAL_REPORT_NOT_UNIQUE")
        verification = next(iter(selected.values()))
    report = json.loads(_read_reference(verification, root=root))
    compilation_binding = None
    if report.get("verification_scope") == "compilation_only_gameplay_not_verified":
        if not bound_verifier:
            raise ValueError("DELIVERY_COMPILATION_BOUND_EXECUTION_REQUIRED")
        compilation_binding = _compilation_feedback_binding(root, report, proposal, verification, store, manifest)
    if repository_binding is not None:
        from src.mod_agent.domain.csharp_repository_delivery import validate_repository_candidate
        verifier_binding = validate_repository_candidate(binding=repository_binding, workspace_root=root,
            store=store, initial=initial, verification=verification, report=report, proposal=proposal)
    else:
        verifier_binding = _bound_verifier(context, root, initial, manifest, verification, report, proposal,
            compilation_binding=compilation_binding) if bound_verifier else None
    if verifier_binding is not None:
        verifier_sha = verifier_binding["validator_sha256"]
    if (report.get("status") != "passed" or report.get("execution_origin") != "executed"
            or report.get("verification_identity", {}).get("validator_sha256") != verifier_sha):
        raise ValueError("DELIVERY_VERIFIER_OR_RESULT_MISMATCH")
    if compilation_binding is not None:
        # 为什么要有它：编译事件不是游戏行为检查；正式验收项由本次调用者显式给出，仍须在正式DLL回调逐项通过。
        check_ids = config.get("final_acceptance_check_ids", [])
        if (not isinstance(check_ids, list) or any(not isinstance(item, str) or not item for item in check_ids)
                or len(set(check_ids)) != len(check_ids)
                or config.get("final_acceptance_required") is True and not check_ids):
            raise ValueError("DELIVERY_EXPLICIT_FINAL_CHECK_IDENTITIES_REQUIRED")
    else:
        details = json.loads(_read_reference(report["verification_ref"], root=root))
        rows = details.get("checks", [])
        # 为什么要有它：参数化测试可能同名，须保留每条结果的runner身份；缺少ID的旧记录才回退到显示名。
        check_ids = [row.get("check_id", row.get("name")) if repository_binding is not None else row.get("check_id", row.get("check"))
            for row in rows]
        if (not check_ids or any(not isinstance(item, str) or not item for item in check_ids)
                or len(set(check_ids)) != len(check_ids) or any(row.get("passed") is not True for row in rows)):
            raise ValueError("DELIVERY_PASSED_CHECK_IDENTITIES_REQUIRED")
    original = manifest["nuget_isolated_build_profile"]
    target = _inside(root, original["working_directory"])
    profile = (_create_nuget_isolated_build_profile(run_root=root, working_directory=target,
        config_file=_inside(root, original["config_file"]),
        bepinex_dir=_inside(root, original["bepinex_dir"]) if original.get("bepinex_dir") else None,
        game_folder_dir=_inside(root, original["game_folder_dir"]) if original.get("game_folder_dir") else None,
        timeout_seconds=original["timeout_seconds"]) if repository_binding is None else
        validate_repository_binding(repository_binding, root)[0].create_nuget_profile(root, target))
    if profile.to_dict() != original:
        raise ValueError("DELIVERY_ORIGINAL_BUILD_PROFILE_CHANGED")
    baseline = manifest["baseline_build"]
    baseline_directory = _inside(root, baseline["build_profile"]["working_directory"])
    baseline_artifact = _inside(root, baseline["artifact_path"])
    relative = baseline_artifact.relative_to(baseline_directory)
    from src.mod_agent.build.baseline_repair import validated_baseline_repair_observation
    repair_observation = validated_baseline_repair_observation(manifest, root)
    # 为什么要有它：修复失败基线时原DLL可以不存在；路径来自原构建配置，成功只由当前候选验收和正式Build判定。
    if relative.suffix.lower() != ".dll" or (repair_observation is None and (
            baseline.get("restore_exit_code") != 0 or baseline.get("build_exit_code") != 0
            or baseline.get("artifact_usable") is False
            or _ref(baseline_artifact)["sha256"] != baseline["artifact_sha256"])):
        raise ValueError("DELIVERY_FORMAL_BASELINE_DLL_REQUIRED")
    return {"run_root": root, "initial": initial, "manifest": manifest, "initial_ref": initial_ref,
        "manifest_ref": manifest_ref, "checkpoint": checkpoint, "store": store,
        "verification_report_ref": verification, "required_check_ids": check_ids,
        "profile": profile, "expected_artifact": relative.as_posix(), "target": target,
        "config": config, "model_calls_performed": 0, "approval_calls_performed": 0,
        "apply_calls_performed": 0, "build_calls_performed": 0,
        "may_initialize_run_local_build_cache": True, "verifier_identity_binding": verifier_binding,
        "repository_build_and_test": repository_binding,
        "verification_scope": report.get("verification_scope", "recorded_behavior_checks")}


def deliver(context, *, final_artifact_verifier=None, expected_initial_result_ref=None):
    started = time.perf_counter()
    try:
        prepared = prepare_delivery(context)
        # Scope: a caller's pre-Apply checks bind the exact input read again here.
        # 为什么要有它：外层已核准的文件范围不能被第二次准备时换掉的run_result或checkpoint绕过；比较必须发生在授权与Apply之前。
        if expected_initial_result_ref is not None and (
                not _same_ref(prepared["initial_ref"], expected_initial_result_ref)
                or _ref(expected_initial_result_ref["path"]) != expected_initial_result_ref):
            raise ValueError("DELIVERY_PRECHECKED_INITIAL_RESULT_CHANGED")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"status": "unknown", "source_origin": "product_model_proposal", "delivery_complete": False,
            "acceptance_passed": False, "evidence_refs": [], "execution_refs": [], "reason": str(error)}
    root, initial, store = prepared["run_root"], prepared["initial"], prepared["store"]
    if prepared["config"].get("final_acceptance_required") is True and not callable(final_artifact_verifier):
        raise ValueError("DELIVERY_FINAL_ACCEPTANCE_CALLBACK_REQUIRED")
    output = _inside(root, prepared["config"].get("output_directory", "delivery"))
    output.mkdir(exist_ok=False)
    execution_refs = []

    def save(name, value):
        path = output / name
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return _ref(path)

    try:
        checkpoint = prepared["checkpoint"]
        decision = prepare_task_delegation(artifact_store_root=str(store.root), checkpoint_ref=checkpoint,
            verification_report_ref=prepared["verification_report_ref"], required_check_ids=prepared["required_check_ids"],
            disposable_run_root=str(root), policy_id=prepared["config"]["policy_id"])
        approval = ResumePayload(resume_id=initial["run_id"] + "-policy-apply", run_id=initial["run_id"],
            request_id=initial["request_id"], feature_id=initial["feature_id"], stage="dry_run_apply",
            decision="approve", approved_by="task_delegation_policy:" + decision["sha256"],
            approved_at=datetime.now(timezone.utc).isoformat(),
            artifact_refs=tuple(store.semantic_read_immutable(checkpoint)["data"]["artifact_refs"]),
            diff_hash=initial["diff_hash"], authority_request_hash=decision["sha256"])
        save("authorization.json", {"decision_ref": decision, "initial_result_ref": prepared["initial_ref"],
            "initial_manifest_ref": prepared["manifest_ref"], "adapter_ref": _ref(__file__),
            "verifier_identity_binding": prepared["verifier_identity_binding"]})
        applied = resume_deferred_approval(artifact_store_root=str(store.root), checkpoint_ref=checkpoint,
            resume_payload=approval, task_delegation_ref=decision)
        execution_refs.append(save("apply.json", applied))
        if applied.get("status") != "applied_in_isolation":
            raise ValueError("DELIVERY_APPLY_NOT_COMPLETED")
        next_checkpoint = ArtifactRef(**applied["workflow_checkpoint"])
        build_resume = replace(approval, resume_id=initial["run_id"] + "-policy-build", stage="isolated_apply",
            artifact_refs=tuple(store.semantic_read_immutable(next_checkpoint)["data"]["artifact_refs"]))

        def build(destination, feature_id, mode):
            if Path(destination).resolve() != root or mode != "isolated_apply":
                raise ValueError("DELIVERY_BUILD_SCOPE_CHANGED")
            profile = prepared["profile"]
            selected = BuildProfile(profile_id=profile.profile_id, build_system="dotnet",
                working_directory=str(prepared["target"]), command=profile.build_command,
                timeout_seconds=profile.timeout_seconds, expected_artifacts=(prepared["expected_artifact"],))

            def runner(chosen):
                if chosen != selected:
                    raise ValueError("DELIVERY_BUILD_PROFILE_CHANGED")
                execution = _run_nuget_isolated_build(profile)
                save("build-process.json", execution.to_dict())
                return BuildCommandResult(exit_code=execution.exit_code, stdout=execution.stdout,
                    stderr=execution.stderr, duration_ms=execution.duration_ms)

            return BuildAdapter().run_local(feature_id=feature_id, profile=selected, command_runner=runner,
                allow_execution=True, input_execution_mode=mode)

        built = continue_reviewed_patch(artifact_store_root=str(store.root), checkpoint_ref=next_checkpoint,
            resume_payload=build_resume, allow_build=True,
            build_result_factory=None if prepared.get("repository_build_and_test") is not None else build)
        execution_refs.append(save("build.json", built))
        if built.get("status") != "built":
            raise ValueError("DELIVERY_BUILD_NOT_COMPLETED")
        final_acceptance = None
        assembly_ref = _ref(prepared["target"] / prepared["expected_artifact"])
        if final_artifact_verifier is not None:
            # 为什么要有它：正式产物验收完成前不能发布正常入口会认可的delivered记录。
            final_acceptance = final_artifact_verifier(assembly_ref)
            save("final-acceptance.json", final_acceptance)
            checks = final_acceptance.get("checks", [])
            ids = [row.get("check_id") for row in checks]
            if (final_acceptance.get("status") != "passed" or final_acceptance.get("assembly_ref") != assembly_ref
                    or len(ids) != len(set(ids)) or set(ids) != set(prepared["required_check_ids"])
                    or any(row.get("passed") is not True for row in checks)
                    or _ref(assembly_ref["path"]) != assembly_ref):
                raise ValueError("DELIVERY_FORMAL_ACCEPTANCE_FAILED")
        patch = output / "model.patch"
        with patch.open("x", encoding="utf-8") as stream:
            stream.write(initial["unified_diff"])
        summary = {"status": "delivered", "source_origin": "product_model_proposal", "delivery_complete": True,
            "acceptance_passed": True, "verification_report_ref": prepared["verification_report_ref"],
            "verifier_identity_binding": prepared["verifier_identity_binding"],
            "candidate_verification_scope": prepared.get("verification_scope", "recorded_behavior_checks"),
            **({"formal_acceptance_ref": _ref(output / "final-acceptance.json")} if final_acceptance is not None else {}),
            "execution_refs": execution_refs,
            "evidence_refs": [_ref(patch), _ref(prepared["target"] / prepared["expected_artifact"]), prepared["verification_report_ref"]],
            "additional_model_calls": 0, "delivery_seconds": time.perf_counter() - started}
    except Exception as error:
        summary = {"status": "failed", "source_origin": "product_model_proposal", "delivery_complete": False,
            "acceptance_passed": False, "execution_refs": execution_refs, "evidence_refs": [],
            "verification_report_ref": prepared["verification_report_ref"], "reason": str(error)}
    finally:
        for reference in (prepared["initial_ref"], prepared["manifest_ref"]):
            if _ref(reference["path"]) != reference:
                raise ValueError("DELIVERY_ORIGINAL_PRODUCT_RECORD_CHANGED")
    save("result.json", summary)
    return summary
