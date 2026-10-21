"""Role selection from bound Patch execution facts, without generation or fallback.

Qualification is local and scoped to a named target Mod, a fixed task suite,
profile, model digest, worker definition and verifier identities. It grants no
source, disclosure, Apply or Runtime authority. No qualification database exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactRef, ArtifactStore
from .evaluation.quality_policy import QualityGatePolicy, evaluate_quality_policy
from .model_admission import validate_ollama_model_identity

ROUTING_SCHEMA = "qualified_patch_model_routing_v1"
QUALIFICATION_SCHEMA = "patch_model_qualification_v1"
SUITE_SCHEMA = "patch_qualification_tasks_v1"
ROLES = frozenset({"initial_patch", "repair"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_FIELDS = ("profile_version", "base_url", "model", "context_window", "output_budget", "timeout_seconds")
_WORKER_FILES = (
    "local_entry.py", "fresh_planning_continuation.py", "proposal/patch_agent_loop.py",
    "proposal/code_generator.py", "execution/patch_apply.py", "workflow/graph.py",
    "proposal/behavior_feedback_view.py",  # 为什么要有它：反馈投影会改变worker实际收到的事实表示，仅改helper也必须改变worker身份。
    "proposal/observation_value_refs.py",  # 为什么要有它：内联引用的编码和还原影响实际反馈，独立修改该模块必须使旧资格身份失效。
    "proposal/repair_diagnosis.py",  # 为什么要有它：可选诊断的输入供给影响真实修订与调用成本，需绑定执行版本。
    "candidate_execution.py",  # 为什么要有它：共同候选物化与验收逻辑影响真实结果，必须纳入运行版本。
    "task_agent_loop.py", "task_agent_tools.py", "task_agent_owners.py",  # 为什么要有它：实验执行器的工具调度、预算与实际动作也属于本轮实现身份。
    "comparison_checkpoint.py", "proposal/failure_state.py",  # 为什么要有它：检查点准入与失败计数影响继承历史后的可执行额度。
    "planning/constraint_context.py",  # 为什么要有它：约束来源投影会改变模型实际输入，独立修改也须改变运行版本。
    "repair/upstream.py",  # 为什么要有它：原失败是否进入重新调查会改变实际决策，不能沿用缺失该反馈路径的worker身份。
    "proposal/tool_argument_feedback.py",  # 为什么要有它：只读工具参数纠错的反馈与一次额度影响worker实际执行，单独修改helper也必须使旧身份失效。
    "evidence/reassessment.py",  # 为什么要有它：相同证据的复用会改变worker续接与模型调用，复用实现漂移也须绑定到执行身份。
    "request_patch_execution.py",  # 为什么要有它：多段worker历史的累计偏移决定真实续接预算，不能只改该所有者而沿用旧执行身份。
    "build/nuget.py",  # 为什么要有它：这段原由local_entry覆盖的实现迁出后，构建代码漂移仍须改变worker身份。
    "build/baseline_repair.py",  # 为什么要有它：源码编译失败的准入与诊断供给会改变产品行为，须绑定真实执行版本。
    "evidence/compilation_candidate.py",  # 为什么要有它：有界候选来源与不足证据的处理影响编辑范围，须纳入执行版本。
    "build/dependencies.py",  # 为什么要有它：依赖快照与复制迁出入口后，同样保留原有实现身份覆盖。
)


class ModelQualificationError(ValueError):
    def __init__(self, code: str, reason: str, *, status: str = "rejected"):
        super().__init__(f"{code}: {reason}")
        self.code, self.status = code, status


def _require(value: bool, reason: str, *, pending: bool = False) -> None:
    if not value:
        raise ModelQualificationError(
            "MODEL_QUALIFICATION_PENDING" if pending else "MODEL_NOT_QUALIFIED", reason,
            status="pending" if pending else "rejected",
        )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate qualification JSON key.")
        value[key] = item
    return value


def _object(raw: bytes) -> dict[str, Any]:
    result = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON.")))
    _require(isinstance(result, dict), "Qualification documents must be objects.")
    canonical_hash(result)
    return result


def read_bound_json(ref: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(ref, Mapping) and set(ref) == {"path", "sha256"}, "Exact file reference required.")
    _require(isinstance(ref["path"], str) and Path(ref["path"]).is_absolute()
             and isinstance(ref["sha256"], str) and _HASH.fullmatch(ref["sha256"]) is not None,
             "Qualification references require absolute paths and SHA-256.")
    _require(not ref["path"].startswith(("\\\\", "//")) and not Path(ref["path"]).drive.startswith("\\\\"),
             "Qualification reads do not access network shares.")
    try:
        raw = Path(ref["path"]).read_bytes()
    except OSError as exc:
        raise ModelQualificationError("MODEL_QUALIFICATION_PENDING", "Referenced evidence is missing.", status="pending") from exc
    _require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "Referenced qualification bytes changed.")
    return _object(raw)


def worker_implementation_identity() -> dict[str, Any]:
    """Fingerprint an explicit execution surface, not Git HEAD or documentation."""
    root = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _WORKER_FILES}
    return {"schema_version": "patch_worker_execution_identity_v1",
            "implementation_sha256": canonical_hash(files), "files": files,
            "scope": "listed_worker_generation_materialization_and_graph_sources"}


def patch_execution_qualification_identity(policy: Any) -> dict[str, Any]:
    return {**worker_implementation_identity(),
            "worker_policy_sha256": canonical_hash(asdict(policy)) if policy is not None else None}


def _read_artifact(store: ArtifactStore, ref: Any, kind: str) -> dict[str, Any]:
    _require(isinstance(ref, dict) and ref.get("artifact_type") == kind,
             f"Exact {kind} evidence is missing.", pending=True)
    try:
        return store.read(ArtifactRef(**ref))
    except (ValueError, TypeError, OSError) as exc:
        raise ModelQualificationError("MODEL_NOT_QUALIFIED", f"Invalid {kind} evidence.") from exc


def _verify_materialized_response(*, feature: dict[str, Any], evidence: dict[str, Any],
                                  plan: dict[str, Any], parsed_output: dict[str, Any],
                                  proposal: dict[str, Any], expected_changes_hash: str,
                                  workspace_root: str | None = None) -> dict[str, Any] | None:
    """Reuse the original native materializer, including its BOM and exact-binding rules."""
    from src.libs.llm import ChatResponse

    from .local_entry import (
        LocalEntryError,
        _external_evidence_items,
        _patch_generation_obligation_projection,
    )
    from .proposal import LLMExactPatchGenerator, PatchGenerationError

    try:
        obligations = _patch_generation_obligation_projection(
            feature_spec=feature, implementation=plan, planned_changes=list(plan["proposed_changes"]), evidence_pack=evidence)
        generator = LLMExactPatchGenerator(None)
        prepared = generator.prepare_native_contract(
            **({"constraint_context": obligations["constraint_context"]} if "constraint_context" in obligations else {}),
            user_request=feature["user_request"], evidence_items=_external_evidence_items(evidence["items"]),
            binding_registry=plan["binding_registry"], constraints=obligations["side_effect_boundaries"],
            target_behavior=obligations["target_behavior"], acceptance_criteria=obligations["acceptance_criteria"],
            runtime_verification_plan=obligations["runtime_verification_plan"],
            required_change_units=obligations["required_change_units"],
            required_validation_by_binding={row["edit_binding_id"]: row["expected_validation"]
                                            for row in obligations["required_change_units"]},
        )
        generated = generator.materialize_prepared_native(prepared, ChatResponse(content=json.dumps(parsed_output), model="retained-response-verification"))
    except (LocalEntryError, PatchGenerationError, KeyError, TypeError, ValueError) as exc:
        raise ModelQualificationError("MODEL_NOT_QUALIFIED", "The retained response cannot be materialized under its original contracts.") from exc
    _require(canonical_hash(generated.changes) == expected_changes_hash, "Behavior report is bound to different materialized changes.")
    stored = {row["edit_binding_id"]: row for row in proposal["diff_intent"]}
    _require(len(stored) == len(proposal["diff_intent"]) == len(generated.changes)
             and set(stored) == {row["edit_binding_id"] for row in generated.changes}, "Proposal binding coverage differs from the model output.")
    _require(all(stored[row["edit_binding_id"]]["patch"] == row["patch"] for row in generated.changes),
             "Proposal code differs from the original model output after deterministic materialization.")
    if workspace_root is not None:
        from .execution.patch_apply import PatchApplyError, materialize_exact_text_patch

        try:
            materialized = materialize_exact_text_patch(workspace_root=workspace_root,
                diff_intent=generated.changes, implementation_plan=plan)
        except PatchApplyError as exc:
            raise ModelQualificationError("MODEL_QUALIFICATION_PENDING", "The fixed source baseline cannot be rematerialized.", status="pending") from exc
        return {"baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256,
                "files": sorted(materialized.files)}
    return None


def _trial_outcome(case: dict[str, Any], trial: dict[str, Any], qualification: dict[str, Any],
                   profile: dict[str, Any], version: dict[str, Any], policy_hash: str) -> dict[str, Any]:
    journal = read_bound_json(trial["attempt_ref"])
    factory = journal.get("factory_receipt", {})
    terminal = journal.get("terminal_result")
    _require(journal.get("receipt_version") == "patch_continuation_attempt_v1", "Unsupported attempt evidence.")
    _require(isinstance(terminal, dict) and isinstance(journal.get("terminal_result_sha256"), str),
             "Attempt has no verifiable terminal result.", pending=True)
    from .fresh_planning_continuation import _canonical_sha256

    _require(_canonical_sha256(terminal) == journal["terminal_result_sha256"], "Terminal result hash mismatch.")
    effective_role = "repair" if isinstance(journal.get("repair_context"), dict) else "initial_patch"
    _require(effective_role == qualification["role"], "Attempt belongs to a different model role.")
    _require(factory.get("profile") == profile, "Attempt model configuration differs from the profile.")
    _require(factory.get("model_identity", {}).get("digest") == qualification["model_digest"], "Model digest mismatch.")
    turns = factory.get("worker_turns")
    _require(isinstance(turns, list) and bool(turns)
             and len(turns) == factory.get("model_call_count") == terminal.get("execution_counts", {}).get("model")
             and all(turn.get("status") == "responded" and turn.get("response_origin") == "live_provider_response" for turn in turns),
             "Independent model execution evidence is incomplete or replayed.", pending=True)
    for turn in turns:
        response = read_bound_json(turn["response_ref"])
        _require(response.get("model") == profile["model"], "Retained model response has a different model identity.")
    final_response = factory.get("model_response", {})
    parsed_output = read_bound_json({"path": final_response.get("path"), "sha256": final_response.get("output_sha256")})
    _require(isinstance(parsed_output.get("changes"), list) and bool(parsed_output["changes"]),
             "The verified model output does not contain a structured Patch.")
    last_content = response.get("content")
    _require(isinstance(last_content, str)
             and hashlib.sha256(last_content.encode("utf-8")).hexdigest() == final_response["output_sha256"]
             and response.get("raw_response", {}).get("message", {}).get("content") == last_content,
             "Final Patch bytes do not match the last retained provider response.")
    executed_version = factory.get("qualification_execution_identity")
    _require(isinstance(executed_version, dict), "Worker execution version was not recorded.", pending=True)
    _require(executed_version.get("implementation_sha256") == version["implementation_sha256"]
             and executed_version.get("worker_policy_sha256") == policy_hash,
             "Worker implementation or policy differs from the tested version.", pending=True)
    store = ArtifactStore(case["artifact_store_root"])
    feature_ref = case["feature_spec_ref"]
    feature = _read_artifact(store, feature_ref, "feature_spec")
    _require(feature["data"].get("target_mod") == qualification["scope"]["target_mod"], "Task is outside qualification scope.")
    _require(journal.get("selected_artifacts", {}).get("feature_spec") == feature_ref,
             "Attempt does not belong to its fixed task.")
    evaluations = factory.get("patch_agent_loop", {}).get("behavior_evaluations")
    _require(isinstance(evaluations, list) and bool(evaluations), "No final behavior verification recorded.", pending=True)
    final = evaluations[-1]
    report = read_bound_json(final["report_ref"])
    _require(all(final.get(key) == report.get(key) for key in (
        "status", "generated_changes_sha256", "source_snapshot_sha256", "observations")),
        "Final behavior receipt and report disagree.")
    _require(final.get("admission") == "accepted", "Behavior evidence was not admitted.", pending=True)
    _require(final.get("execution_origin") == report.get("execution_origin") == "executed",
             "Final verification is not an explicitly executed check.", pending=True)
    identity = case.get("verification_identity")
    _require(isinstance(identity, dict) and all(isinstance(identity.get(key), str)
             and _HASH.fullmatch(identity[key]) is not None for key in
             ("validator_sha256", "inputs_sha256", "environment_sha256")),
             "Task verification identity is incomplete.", pending=True)
    _require(final.get("verification_identity") == report.get("verification_identity") == identity,
             "Verification inputs, environment or definition differ from the fixed task.", pending=True)
    matrix = read_bound_json(report["verification_ref"])
    checks = matrix.get("checks")
    summary = report.get("verification_summary", {})
    expected_checks = case.get("expected_check_ids")
    _require(isinstance(expected_checks, list) and bool(expected_checks)
             and all(isinstance(name, str) and bool(name.strip()) for name in expected_checks)
             and len(set(expected_checks)) == len(expected_checks),
             "The fixed task definition has no complete check identity set.", pending=True)
    _require(isinstance(checks, list) and bool(checks) and all(isinstance(row, dict)
             and isinstance(row.get("check"), str) and type(row.get("passed")) is bool for row in checks),
             "Verification matrix is incomplete.", pending=True)
    _require(len({row["check"] for row in checks}) == len(checks)
             and summary.get("checks") == len(checks)
             and type(summary.get("executions")) is int and summary["executions"] > 0
             and matrix.get("executions") == summary["executions"],
             "Verification matrix denominator is incomplete.", pending=True)
    _require({row["check"] for row in checks} == set(expected_checks),
             "Executed check identities differ from the fixed task definition.", pending=True)
    failed = sum(not row["passed"] for row in checks)
    _require(summary.get("failed_checks") == failed, "Reported verification totals do not match actual checks.")
    _require(matrix.get("failures") == [row for row in checks if not row["passed"]],
             "The final failure list does not match its complete check matrix.")
    _require(final.get("model_response_sha256") == factory.get("model_response", {}).get("output_sha256"),
             "Final verification is not bound to the final generated response.")
    success = report.get("status") == "passed" and failed == 0
    if success:
        _require(terminal.get("status") == "terminal" and terminal.get("terminal_stage") == "dry_run_apply",
                 "The trial did not finish the selected Patch acceptance path.", pending=True)
        proposal = _read_artifact(store, terminal.get("patch_proposal_artifact"), "patch_proposal")
        review = _read_artifact(store, terminal.get("proposal_review_artifact"), "proposal_review")
        dry_run = _read_artifact(store, terminal.get("dry_run_artifact"), "dry_run_result")
        _require(proposal["status"] == "success" and review["data"].get("decision") == "approve_dry_run"
                 and review["data"].get("allowed_next_stage") == "dry_run_apply"
                 and dry_run["status"] == "success", "Required Proposal/Review/Dry-run did not pass.")
        proposal_ref, review_ref = terminal["patch_proposal_artifact"], terminal["proposal_review_artifact"]
        for envelope, expected in ((review, (proposal_ref,)), (dry_run, (proposal_ref, review_ref))):
            upstream = envelope.get("inputs", {}).get("upstream_artifacts", [])
            _require(all(any(item.get("path") == ref["path"] and item.get("artifact_hash") == ref["artifact_hash"]
                             for item in upstream) for ref in expected), "Downstream artifacts belong to another result.")
        changes = proposal["data"].get("diff_intent", [])
        _require(bool(changes) and all(change.get("generation", {}).get("model_response_sha256") == final["model_response_sha256"]
                                      for change in changes), "Proposal does not retain the verified final model response.")
        selected = journal["selected_artifacts"]
        evidence = _read_artifact(store, selected.get("evidence_pack"), "evidence_pack")
        plan = _read_artifact(store, selected.get("implementation_plan"), "implementation_plan")
        workspace = case.get("workspace_root")
        _require(isinstance(workspace, str) and Path(workspace).is_absolute() and Path(workspace).is_dir(),
                 "The fixed source baseline is unavailable for materialization.", pending=True)
        code_identity = _verify_materialized_response(feature=feature["data"], evidence=evidence["data"], plan=plan["data"],
            parsed_output=parsed_output, proposal=proposal["data"], expected_changes_hash=final["generated_changes_sha256"],
            workspace_root=workspace)
        _require(code_identity == final.get("materialized_code_identity") == report.get("materialized_code_identity")
                 == matrix.get("materialized_code_identity"),
                 "The complete matrix is not bound to the actual materialized Patch bytes.", pending=True)
    _require(terminal.get("product_source_writes_performed") is False,
             "Qualification trial exceeds its no-write acceptance scope.")
    return {"case_id": case["case_id"], "feature_id": feature["feature_id"],
            "feature_spec_sha256": feature_ref["artifact_hash"], "task_passed": success,
            "checks": len(checks), "failed_checks": failed, "attempt_ref": trial["attempt_ref"]}


@dataclass(frozen=True)
class QualifiedProfileSelection:
    profile: Any
    model_digest: str
    role: str
    routing_ref: dict[str, str]
    evidence_refs: tuple[dict[str, str], ...]
    receipt: dict[str, Any]
    worker_policy: Any

    def validate_live_identity(self, identity: Mapping[str, Any]) -> None:
        validate_ollama_model_identity(identity, expected_base_url=self.profile.base_url,
                                      expected_model=self.profile.model)
        _require(identity.get("digest") == self.model_digest, "Selected local model differs from qualified digest.")
        for ref in (self.routing_ref, *self.evidence_refs):
            read_bound_json(ref)
        current = resolve_qualified_patch_profile(
            routing_path=self.routing_ref["path"], role=self.role,
            target_mod=self.receipt["scope"]["target_mod"], worker_policy=self.worker_policy,
        )
        _require(current.receipt == self.receipt, "Qualification facts changed after profile selection.")
        _require(worker_implementation_identity()["implementation_sha256"] == self.receipt["implementation_sha256"],
                 "Worker changed after profile selection.", pending=True)


def resolve_qualified_patch_profile(*, routing_path: str | Path, role: str, target_mod: str,
                                    worker_policy: Any) -> QualifiedProfileSelection:
    """Select exactly the configured role; never try another role or model on failure."""
    try:
        _require(role in ROLES, "Unsupported model role.")
        _require(isinstance(target_mod, str) and bool(target_mod.strip()), "An explicit target Mod scope is required.", pending=True)
        _require(worker_policy is not None, "Qualified routing requires an explicit worker policy.", pending=True)
        _require(not str(routing_path).startswith(("\\\\", "//")), "Model routing is a local configuration.")
        path = Path(routing_path).resolve(strict=True)
        routing_ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        routing = read_bound_json(routing_ref)
        _require(set(routing) == {"schema_version", "routes"} and routing["schema_version"] == ROUTING_SCHEMA,
                 "Invalid role routing configuration.")
        routes = routing["routes"]
        _require(isinstance(routes, dict) and set(routes).issubset(ROLES) and role in routes,
                 "No qualified profile configured for this exact role.", pending=True)
        _require(isinstance(routes[role], dict) and set(routes[role]) == {"qualification_ref"}, "Invalid role selection.")
        qualification_ref = routes[role]["qualification_ref"]
        qualification = read_bound_json(qualification_ref)
        _require(set(qualification) == {"schema_version", "role", "scope", "profile", "model_digest",
                 "worker_policy_sha256", "implementation_sha256", "task_suite_ref", "attempts", "quality_policy"}
                 and qualification["schema_version"] == QUALIFICATION_SCHEMA, "Invalid qualification evidence document.")
        _require(qualification["role"] == role and qualification["scope"] == {"target_mod": target_mod},
                 "Role or target scope differs from the qualification.")
        from .local_entry import LocalEntryError, LocalOllamaInitialPatchProfileV1

        declared = qualification["profile"]
        try:
            profile = LocalOllamaInitialPatchProfileV1(**{key: declared[key] for key in _PROFILE_FIELDS})
        except LocalEntryError as exc:
            raise ModelQualificationError("MODEL_NOT_QUALIFIED", "The selected profile is not an admitted local model profile.") from exc
        _require(profile.to_dict() == declared, "Only the existing exact local profile is supported; no disclosure or fallback is authorized.")
        _require(isinstance(qualification["model_digest"], str)
                 and _HASH.fullmatch(qualification["model_digest"]) is not None, "Exact model digest required.")
        version = worker_implementation_identity()
        policy_hash = canonical_hash(asdict(worker_policy))
        _require(version["implementation_sha256"] == qualification["implementation_sha256"]
                 and policy_hash == qualification["worker_policy_sha256"], "Qualification is for another worker version or policy.", pending=True)
        suite = read_bound_json(qualification["task_suite_ref"])
        _require(suite.get("schema_version") == SUITE_SCHEMA and suite.get("scope") == qualification["scope"], "Task suite scope/version mismatch.")
        cases, trials = suite.get("cases"), qualification["attempts"]
        _require(isinstance(cases, list) and len(cases) >= 3, "Qualification requires three fixed distinct tasks.", pending=True)
        _require(isinstance(trials, list), "Missing task attempt references.", pending=True)
        case_ids = [case["case_id"] for case in cases]
        _require(len(set(case_ids)) == len(case_ids)
                 and len({case["feature_spec_ref"]["feature_id"] for case in cases}) == len(cases)
                 and len({case["feature_spec_ref"]["artifact_hash"] for case in cases}) == len(cases),
                 "Repeated attempts or Feature revisions cannot inflate the task denominator.")
        trial_map = {trial["case_id"]: trial for trial in trials}
        _require(len(trial_map) == len(trials) and set(trial_map) == set(case_ids), "Task attempt coverage is incomplete or duplicated.", pending=True)
        outcomes = [_trial_outcome(case, trial_map[case["case_id"]], qualification, declared, version, policy_hash) for case in cases]
        metrics = {"task_success_rate": sum(row["task_passed"] for row in outcomes) / len(cases)}
        gate = QualityGatePolicy.from_mapping(qualification["quality_policy"])
        _require(gate.required_metric_names == ("task_success_rate",), "Only measured Patch task success is qualified by this gate.")
        quality = evaluate_quality_policy(gate, metrics, {"task_success_rate": True})
        _require(all(row["task_passed"] for row in outcomes) and quality == "pass",
                 "At least one required task's final behavior failed; schema or Review success cannot hide it.")
        receipt = {"status": "qualified", "role": role, "scope": qualification["scope"],
                   "routing_ref": routing_ref, "qualification_ref": qualification_ref,
                   "task_suite_ref": qualification["task_suite_ref"], "model_digest": qualification["model_digest"],
                   "profile_sha256": canonical_hash(declared), "implementation_sha256": version["implementation_sha256"],
                   "worker_policy_sha256": policy_hash, "distinct_task_count": len(cases),
                   "task_outcomes": outcomes, "metrics": metrics, "quality_policy": gate.to_dict(),
                   "automatic_fallback": False, "authorizes_disclosure": False, "authorizes_apply": False,
                   "claim_scope": "selected_role_target_and_fixed_executed_task_suite_only"}
        evidence_refs = tuple([qualification_ref, qualification["task_suite_ref"], *[item["attempt_ref"] for item in trials]])
        return QualifiedProfileSelection(profile, qualification["model_digest"], role, routing_ref, evidence_refs, receipt, worker_policy)
    except ModelQualificationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ModelQualificationError("MODEL_QUALIFICATION_PENDING", "Qualification evidence is incomplete or invalid.", status="pending") from exc
