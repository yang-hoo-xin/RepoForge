"""Pure failure-state projection shared by normal execution and retained trials."""
from collections.abc import Mapping

from src.mod_agent.model_qualification import canonical_hash


def source_fact_hashes(observations):
    return sorted({canonical_hash({key: item[key] for key in (
        "binding_id", "source_hash", "source_snapshot_sha256", "source_ref") if key in item})
        for item in observations if isinstance(item, Mapping)
        and (item.get("admitted_to_model") or item.get("payload_admitted"))})


def failure_signature(report, source_facts):
    # Scope: projection only, no retry policy. 为什么要有它：恢复和正常循环必须比较相同事实；计数阈值与是否继续仍由原执行owner决定。
    observations = [{key: value for key, value in item.items() if key != "result_ref"}
                    if isinstance(item, Mapping) else item
                    for item in report.get("failure_state_observations", report["observations"])]
    return canonical_hash({"code": report.get("materialized_code_identity"),
        "status": report["status"], "failure_owner": report["failure_owner"],
        "owner_binding_ids": report["owner_binding_ids"], "observations": observations,
        "verification_identity": report.get("verification_identity"), "source_facts": sorted(set(source_facts))})
