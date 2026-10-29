"""Recompute the two diagnostic arms from retained requests and receipts.

为什么要有它：五类数据必须从原件重算，未知的在途调用不能补零，也不能把工具请求等同正确交付。
"""
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import read_bound_json


def summarize(pair_path, output):
    pair = read_bound_json(_ref(pair_path))
    summaries = {}
    for arm, plan_ref in pair["plans"].items():
        plan = read_bound_json(plan_ref)
        root = Path(plan["root"])
        if not (root / "result.json").exists():
            summaries[arm] = {"status": "not_completed_or_not_started", "plan_ref": plan_ref}
            continue
        result_ref = _ref(root / "result.json")
        result = read_bound_json(result_ref)
        process = read_bound_json(result["process_ref"])
        run = Path(plan["run_root"])
        execution = run / "a/task-agent-execution"
        responses, usages, receipts, triggers = [], {}, [], []
        wire_options = []
        for path in execution.glob("provider-outcome-*.json"):
            row = read_bound_json(_ref(path))
            response = read_bound_json(row["response_ref"])
            request = read_bound_json(row["request_ref"])
            states = [m["content"]["current_task_state"] for m in request["messages"]
                if isinstance(m.get("content"), dict) and "current_task_state" in m["content"]]
            if states and states[-1].get("working_state", {}).get("stagnation", {}).get("navigation_only"):
                triggers.append(row["turn"])
            if row.get("provider_request_ref"):
                wire = read_bound_json(row["provider_request_ref"])
                wire_options.append(wire.get("options"))
            responses.append({"turn": row["turn"], "response_ref": row["response_ref"],
                "request_ref": row["request_ref"], "tool_calls": response.get("tool_calls", [])})
            usages[row["reservation_id"]] = row.get("usage") if row.get("usage_available") else None
        for path in execution.glob("tool-receipt-*.json"):
            row = read_bound_json(_ref(path))
            value = row.get("output", {})
            receipts.append({"turn": row["origin"]["turn"], "name": row["origin"]["name"],
                "status": value.get("status"), "code": value.get("code"), "receipt_ref": _ref(path)})
        responses.sort(key=lambda r: r["turn"])
        receipts.sort(key=lambda r: r["turn"])
        counts = dict(collections.Counter(r["name"] for r in receipts))
        reservations = len(list(execution.glob("resource-reserved-*.json")))
        sums = {key: sum(u[key] for u in usages.values() if u is not None and isinstance(u.get(key), int))
            for key in ("input_tokens", "output_tokens", "cached_input_tokens")}
        unknown = {key: sum(u is None or not isinstance(u.get(key), int) for u in usages.values())
            + max(0, reservations-len(usages)) for key in sums}
        native = read_bound_json(_ref(run / "run_result.json")) if (run / "run_result.json").exists() else None
        pointers = list((run / "a/v2/session-checkpoints").glob("*.json"))
        checkpoint = None
        if len(pointers) == 1:
            pointer = read_bound_json(_ref(pointers[0]))
            checkpoint = read_bound_json({"path": str(run / "a" / pointer["checkpoint_ref"]),
                "sha256": Path(pointer["checkpoint_ref"]).stem})
        candidate = (checkpoint or {}).get("task_head", {}).get("current_revision_ref")
        if checkpoint:
            contract_ref = checkpoint["task_contract_ref"]
            contract = read_bound_json({"path": str(run / "a" / contract_ref), "sha256": Path(contract_ref).stem})
            if candidate == contract["baseline_manifest_ref"]:
                candidate = None
        kinds = {"probe_submit", "experiment_run", "candidate_submit", "candidate_check", "finish"}
        chosen = [{"turn": row["turn"], "name": call["name"]} for row in responses
            for call in row["tool_calls"] if call.get("name") in kinds]
        summaries[arm] = {"run_id": plan["run_id"], "plan_ref": plan_ref, "result_ref": result_ref,
            "process_ref": result["process_ref"], "process_status": process.get("status"),
            "wall_seconds": process.get("wall_seconds"), "task_wall_seconds": result.get("task_wall_seconds"),
            "native_status": native.get("status") if native else None, "native_result_present": native is not None,
            "candidate_ref": candidate, "tool_counts": counts, "tool_receipts": receipts,
            "completed_responses": len(responses), "reserved_calls": reservations,
            "known_usage": sums, "unknown_usage_calls": unknown, "first_feedback_request_turn": min(triggers) if triggers else None,
            "first_non_navigation_choice": chosen[0] if chosen else None,
            "choices_in_first_three_feedback_responses": [c for c in chosen if triggers and min(triggers) <= c["turn"] < min(triggers)+3],
            "actual_wire_options": wire_options,
            "public_check_count": len((checkpoint or {}).get("session", {}).get("public_checks", [])),
            "source_changed_paths": result.get("changed_source_paths"),
            "independent_business_result": result.get("independent_business_result"),
            "correct_delivery": result.get("correct_delivery"),
            "human_active_seconds": None, "technical_hint_or_patch": False,
            "strategy_injected": False, "experience_gain": "not_measured",
            "owned_process_residuals": process.get("owned_process_residuals"),
            "response_refs": [r["response_ref"] for r in responses]}
    report = {"scope": "one development-exposed B0/B1 diagnostic; not hidden-pilot results",
        "pair_ref": _ref(pair_path), "arms": summaries,
        "why": "为什么要有它：原件计量保留失败与未知量；独立验收及交付缺失时不能推断正确交付或有限单位成功成本。"}
    ref = _save(Path(output), report)
    print(json.dumps(ref))


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2])
