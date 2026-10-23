"""Register a clearly manual trial hypothesis from the user's design principle.

为什么要有它：没有自动提炼证据时不伪造学习历史，手工试用只检验建议消费与决策影响。
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.mod_agent.artifacts import ArtifactStore  # noqa: E402
from src.mod_agent.domain.full_project_compilation import _save  # noqa: E402
from src.mod_agent.planning.strategy_experience import StrategyExperience  # noqa: E402

ROOT = Path("C:/tmp/rf-serilog-v2-pair-main")


def main():
    if not (ROOT / "public-preparation.json").is_file():
        raise ValueError("PUBLIC_TASK_PREPARATION_REQUIRED")
    source = _save(ROOT / "manual-strategy-source.json", {
        "schema_version": "external_strategy_hypothesis_source_v1", "visibility": "public",
        "origin": "user_supplied_design_principle_operationalized_by_development_agent",
        "user_text": "命中相关文件不等于采用了证据，更不等于形成了正确判断。选择一个有区分力的实验。根据结果改变调查方向。",
        "qualification": "unvalidated_manual_hypothesis_not_automatic_extraction",
        "contains_reference_patch": False, "contains_hidden_evaluation": False,
        "source_scope": "generic_user_design_discussion_not_task_specific_repair"})
    memory = StrategyExperience(ArtifactStore(ROOT / "strategy-memory"))
    proposed = memory.record_strategy("contrast-observations-before-generalizing-v1", {
        "applicability": {"runtime": "dotnet", "target_domain": "csharp_repository"},
        "suggested_action": "For a behavior-change task, contrast cases that differ in one relevant condition before generalizing an edit. Use the observation to distinguish competing explanations, and change the hypothesis or target when it is contradicted rather than repeat the same patch. Keep a case for behavior that must remain unchanged.",
        "expected_observation": "A useful contrast distinguishes explanations that merely finding a relevant file did not distinguish; a correction addresses the observed case without erasing required behavior.",
        "counterevidence": ["The compared cases do not differ along the hypothesized condition.",
            "The proposed distinction also removes behavior required by the task.",
            "Execution failed before any relevant behavior was observed."]},
        evidence_refs=[{"evidence_id": "user-design-principle", "kind": "json_file", **source,
            "visibility": "public", "allowed_uses": ["strategy_learning"]}],
        origin={"kind": "human", "actor": "external_manual_input_from_user_design_not_product_model_learning"})
    policy = _save(ROOT / "manual-strategy-trial-policy.json", {"schema_version": "strategy_trial_policy_v1",
        "allowed_origin_kinds": ["human"], "allowed_domains": {"runtime": ["dotnet"]}})
    snapshot = memory.admit_trial(proposed, policy_ref=policy)
    result = _save(ROOT / "strategy-preparation.json", {"store_root": str(memory.store.root),
        "strategy_ref": asdict(proposed), "snapshot_ref": asdict(snapshot), "source_ref": source,
        "model_calls": 0, "automatic_extraction": False, "activation": "trial_only",
        "claim_limit": "Any gain may be a conditional guidance/reminder effect; this does not establish long-term learning or migration.",
        "developer_assistance": "Prepared a generic hypothesis from the user's design principle; no task-specific patch, identifier, file target or hidden assertion supplied."})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
