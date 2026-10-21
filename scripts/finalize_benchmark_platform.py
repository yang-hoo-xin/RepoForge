"""Exact delegated platform delivery; the formal DLL is checked before delivered."""
import argparse
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_reload_trials import ref, save
from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.domain.full_project_compilation import _source_snapshot
from src.mod_agent.evaluation.isolated_delivery_adapter import deliver, prepare_delivery
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration


def declared_source_scope(*, workspace, target, proposal, implementation, allowed_paths):
    """Check the exact current materialized diff before delegated Apply.

    为什么要有它：行为验证通过和工程目录授权不能替代本次明确的文件白名单，写入后才拒绝已超出授权顺序。
    """
    from src.mod_agent.execution.patch_apply import materialize_exact_text_patch

    workspace, target = Path(workspace).resolve(strict=True), Path(target).resolve(strict=True)
    if target == workspace or not target.is_relative_to(workspace) or not target.is_dir():
        raise ValueError("BENCHMARK_APPLY_TARGET_OUTSIDE_RUN")
    if not isinstance(allowed_paths, list) or not allowed_paths or any(type(path) is not str for path in allowed_paths):
        raise ValueError("BENCHMARK_EXPLICIT_EDIT_SCOPE_REQUIRED")
    materialized = materialize_exact_text_patch(workspace_root=workspace,
        diff_intent=proposal["diff_intent"], implementation_plan=implementation)
    changed = []
    for path in materialized.files:
        selected = (workspace / path).resolve(strict=True)
        if not selected.is_file() or not selected.is_relative_to(target):
            raise ValueError("BENCHMARK_PRE_APPLY_SCOPE_CHANGED")
        relative = selected.relative_to(target).as_posix()
        if relative not in allowed_paths:
            raise ValueError("BENCHMARK_PRE_APPLY_SCOPE_CHANGED")
        changed.append(relative)
    if not changed:
        raise ValueError("BENCHMARK_PRE_APPLY_CHANGES_REQUIRED")
    return {"status": "passed", "allowed_paths": allowed_paths, "changed_paths": sorted(changed),
        "baseline_sha256": materialized.baseline_sha256, "code_sha256": materialized.code_sha256,
        "checked_before_apply": True, "apply_performed_by_this_check": False}


def finalize(ready_path, *, initial_result_ref=None, records_output=None):
    # Scope: final delivery is a new Python process and must inherit the same resource cap.
    # 为什么要有它：父进程的审计钩子不跨进程继承，正式构建也必须在启动前扣减共享额度。
    from src.mod_agent.build.execution_budget import install_build_budget_from_environment
    build_budget = install_build_budget_from_environment()
    ready = read_bound_json(ref(ready_path))
    if ready.get("development_authorization_ref") is not None and (
            build_budget is None
            or build_budget.configuration.get("authorization_ref") != ready["development_authorization_ref"]
            or build_budget.configuration.get("maximum") != ready.get("shared_build_limit")):
        raise ValueError("BENCHMARK_DEVELOPMENT_SHARED_BUILD_BUDGET_REQUIRED")
    dispatch = read_bound_json(ref(Path(ready["records_root"]) / "dispatch.json"))
    bundle = read_bound_json(ready["bundle_ref"])
    context = {"run_root": dispatch["run_root"], "trial": {"snapshot_root": ready["product_snapshot_root"],
        "bound_inputs": dispatch["bound_inputs"], "extra_load_refs": dispatch["extra_load_refs"], "delivery": {
            "policy_id": ready.get("delivery_policy_id", "benchmark10_user_delegation_v1"), "verification_selection": "current_proposal",
            "verifier_selection": "bound_worker_execution", "final_acceptance_required": True}}}
    if initial_result_ref is not None:
        # Scope: select one separately persisted continuation, never rewrite the original result.
        # 为什么要有它：保存候选恢复需要进入原交付链，但首轮失败和新审批检查点必须分别保留。
        context["trial"]["delivery"]["initial_result_ref"] = dict(initial_result_ref)
    prepared = prepare_delivery(context)
    output_records = Path(ready["records_root"]).resolve()
    if records_output is not None:
        selected_output = Path(records_output).resolve()
        if selected_output == output_records or not selected_output.is_relative_to(output_records):
            raise ValueError("BENCHMARK_DELIVERY_RECORDS_OUTSIDE_RUN")
        # Scope: append a distinct attempt record without replacing prior failed output.
        # 为什么要有它：授权前失败后的修复仍要保留原尝试，不能靠覆盖delivery-result隐藏首次失败。
        selected_output.mkdir(parents=True, exist_ok=False)
        output_records = selected_output
    report = read_bound_json(prepared["verification_report_ref"])
    if report.get("verification_summary") != {"checks": bundle["expected_check_count"], "failed_checks": 0, "executions": bundle["cases"]}:
        raise ValueError("BENCHMARK_COMPLETE_SHARED_ORACLE_REQUIRED")
    baseline = {row["path"]: row["sha256"] for row in prepared["manifest"]["source_snapshot"]}
    expected = {row["path"]: row["sha256"] for row in bundle["source_snapshot"]}
    if baseline != expected:
        raise ValueError("BENCHMARK_DELIVERY_BASELINE_CHANGED")
    state = prepared["store"].semantic_read_immutable(prepared["checkpoint"])["data"]["state_snapshot"]
    documents = {}
    for kind in ("implementation_plan", "patch_proposal"):
        selected = ArtifactRef(feature_id=prepared["initial"]["feature_id"], artifact_type=kind,
            path=state["artifacts"][kind], artifact_hash=state["artifact_hashes"][kind])
        documents[kind] = prepared["store"].semantic_read_immutable(selected)["data"]
    scope = declared_source_scope(workspace=prepared["run_root"], target=prepared["target"],
        proposal=documents["patch_proposal"], implementation=documents["implementation_plan"],
        allowed_paths=bundle["allowed_edit_paths"])
    scope_ref = save(output_records / "source-scope-before-apply.json",
        {**scope, "bundle_ref": ready["bundle_ref"], "workflow_checkpoint_ref": {
            name: getattr(prepared["checkpoint"], name) for name in ("feature_id", "artifact_type", "path", "artifact_hash")}})
    delivered_diff = {}

    def verify(assembly):
        after = {row["path"]: row["sha256"] for row in _source_snapshot(prepared["target"])}
        changed = [path for path in baseline.keys() | after.keys() if baseline.get(path) != after.get(path)]
        if set(after) != set(baseline) or any(path not in bundle["allowed_edit_paths"] for path in changed):
            raise ValueError("BENCHMARK_DELIVERED_SOURCE_SCOPE_CHANGED")
        delivered_diff.update(before=baseline, after=after, changed_paths=sorted(changed),
            generated_source_root=str(prepared["target"]), input_source_unchanged=(
                {row["path"]: row["sha256"] for row in _source_snapshot(Path(prepared["manifest"]["target_source"]))} == baseline))
        configuration = PatchWorkerConfiguration.load(dispatch["allocated_worker_ref"]["path"])
        dependency = prepared["manifest"]["build_dependency"]["game_folder"]
        return configuration.behavior_evaluator().verify_existing_assembly(assembly_ref=assembly,
            dependency_root=dependency["isolated_copy"] if dependency is not None else None,
            expected_dependency_snapshot_sha256=dependency["copy_snapshot_sha256"] if dependency is not None else None)

    result = deliver(context, final_artifact_verifier=verify, expected_initial_result_ref=prepared["initial_ref"])
    result["pre_apply_source_scope_ref"] = scope_ref
    result["delivered_source_diff"] = delivered_diff
    result["input_source_unchanged"] = delivered_diff.get("input_source_unchanged")
    return save(output_records / "delivery-result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready", required=True)
    parser.add_argument("--initial-result")
    parser.add_argument("--initial-result-sha256")
    parser.add_argument("--records-output")
    args = parser.parse_args()
    if bool(args.initial_result) != bool(args.initial_result_sha256):
        parser.error("--initial-result and --initial-result-sha256 must be supplied together")
    selected = ({"path": str(Path(args.initial_result).resolve()), "sha256": args.initial_result_sha256}
        if args.initial_result else None)
    reference = finalize(args.ready, initial_result_ref=selected, records_output=args.records_output)
    print(json.dumps(reference))
    raise SystemExit(0 if read_bound_json(reference).get("delivery_complete") is True else 2)
