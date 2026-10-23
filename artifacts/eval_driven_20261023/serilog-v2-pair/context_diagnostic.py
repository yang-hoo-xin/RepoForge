"""One frozen, exposed-task B capability diagnostic, not another B/B+ pair."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.dsh_run_request import (  # noqa: E402
    _configuration_arguments,
    build_product_argv,
    prepare_new_task,
)
from scripts.run_benchmark_arm import execute  # noqa: E402
from src.libs.llm.agent_turn import AgentMessage  # noqa: E402
from src.mod_agent.domain.csharp_repository_backend import _snapshot  # noqa: E402
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.evaluation.strategy_trial_preparation import _source_snapshot  # noqa: E402
from src.mod_agent.model_admission import read_ollama_model_identity  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402
from src.mod_agent.platform_execution_snapshot import (  # noqa: E402
    FrozenExecution,
    prepare_snapshot,
)
from src.mod_agent.task_agent_v2_context import model_plan_view, project_history  # noqa: E402
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, _hash  # noqa: E402
from src.mod_agent.task_agent_v2_tools import TaskAgentSessionV2  # noqa: E402

ROOT = Path("C:/tmp/rf-serilog-v2-context-main")
OLD = Path("C:/tmp/rf-serilog-v2-pair-main")
SNAPSHOT = Path("C:/tmp/rfcode-serilog-v2-context-B")
MODEL = "qwen3-coder:30b"
DIGEST = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"


def api(name, body=None, timeout=10):
    req = Request("http://127.0.0.1:11434/api/" + name,
        data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def replay():
    saved = importlib.import_module("artifacts.eval_driven_20261023.serilog-v2-pair.replay_source_pages")
    results = []
    for index, run_id in (("1", "dsh_877430f56fd74572ad51ce3118cc535c"), ("2", "dsh_39747f1de5404294bed2c8e4577b2fe6")):
        native = OLD / "paired-inputs" / index / "runs" / run_id / "a/task-agent-execution"
        execution = json.loads((native / "execution.json").read_bytes())
        request_ref = execution["provider_turns"][-1]["request_ref"]
        original = saved.decode(saved.read_ref(request_ref, native))
        state = TaskAgentV2State(native.parent, execution["task_contract_ref"])
        state.save = lambda kind, data: f"v2/{kind}/{_hash(data)}.json"
        calls = {row["call_id"]: row for row in execution["completed_tool_records"]}

        def page(result):
            if result.name not in {"source_read", "candidate_read", "source_search"}:
                return result
            call = calls[result.call_id]
            if saved.read_ref(call["result_ref"], native) != result.output:
                raise ValueError("ORIGINAL_RESULT_CHANGED")
            args = call["arguments"]
            value = (state.search_source(args["parent_revision_ref"], args["query"], args.get("root_id", "target"))
                if result.name == "source_search" else state.read_source(args["parent_revision_ref"], args["path"],
                    args.get("root_id", "target"), offset=args.get("offset", 0), limit=args.get("limit", 2000)))
            return replace(result, output={**deepcopy(result.output), "value": value})

        paged = replace(original, tool_results=tuple(page(row) for row in original.tool_results),
            historical_tool_exchanges=tuple(replace(exchange, tool_results=tuple(page(row) for row in exchange.tool_results))
                for exchange in original.historical_tool_exchanges))
        projected, facts = project_history(paged, state)
        session = TaskAgentSessionV2(state=state, evaluator=None, owner=SimpleNamespace(), feature_id="offline")
        messages = list(projected.messages)
        for index in range(len(messages) - 1, -1, -1):
            content = messages[index].content
            if isinstance(content, dict) and "current_task_state" in content:
                current = {**deepcopy(content["current_task_state"]), "history_projection": facts,
                    "model_technical_plan": model_plan_view(state)}
                messages[index] = AgentMessage("user", {"current_task_state": current})
                break
        projected = replace(projected, messages=tuple(messages), tools=tuple(item.definition for item in session.actions()))
        results.append({"run_id": run_id, "execution_ref": _ref(native / "execution.json"), "request_ref": request_ref,
            "original_admission": saved.admission(original, execution), "projection": facts,
            "projected_admission": saved.admission(projected, execution),
            "scope": "Counterfactual saved public tool results and current tool schemas only; zero model calls; no inferred later choice."})
    return results


def freeze():
    if ROOT.exists() or SNAPSHOT.exists():
        raise ValueError("DIAGNOSTIC_ALREADY_PREPARED_INSPECT_ORIGINAL")
    public = read_bound_json(_ref(OLD / "public-preparation.json"))
    baseline = read_bound_json(_ref(OLD / "baseline-result.json"))
    for key in ("request_ref", "profile_ref", "settings_ref"):
        read_bound_json(public[key])
    if (_ref(public["public_test_ref"]["path"]) != public["public_test_ref"]
            or _snapshot(Path(public["source_root"])) != public["source_inputs"]
            or baseline["status"] != "behavior_failed" or baseline["local_feed_only"] is not True):
        raise ValueError("QUALIFIED_PUBLIC_BASELINE_CHANGED")
    model = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    if model["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    ROOT.mkdir(parents=True)
    replay_ref = _save(ROOT / "saved-request-projection.json", {"results": replay(), "model_calls": 0})
    profile = read_bound_json(public["profile_ref"])
    original_config = read_bound_json(_ref(OLD / "base-execution.json"))
    args = original_config["arguments"]
    position = args.index("--workspace-root") + 1
    args[position] = str(ROOT / "runs")
    config = _save(ROOT / "base-execution.json", {"arguments": args})
    description = read_bound_json(public["request_ref"])["description"]
    selected = prepare_new_task(execution_config=config["path"], description=description)
    selected_ref = _ref(selected)
    argv = build_product_argv(execution_config=selected, description=description)
    _, options = _configuration_arguments(selected)
    run_id = options["--run-id"]
    run_root = ROOT / "runs" / run_id
    refs = [selected_ref, public["request_ref"], public["profile_ref"], public["settings_ref"]]
    platform = prepare_snapshot(source_root=REPO, snapshot_root=SNAPSHOT, business_argv=argv,
        run_root=run_root, bound_inputs=[(ref["path"], ref["sha256"]) for ref in refs])
    plan = _save(ROOT / "frozen-plan.json", {"schema": "v2-context-capability-diagnostic.v1",
        "question": "Can real B use bounded source/history reads, autonomously investigate and choose targets, then revise from public feedback if reached?",
        "exposure": "exposed development task, not U5 or a memory comparison", "planned_attempts": 1,
        "technical_assistance": "Public reproduction and platform/environment preparation; no production fix, target instructions or hidden evaluation supplied.",
        "human_active_seconds": None, "human_timing_basis": "unknown", "run_id": run_id, "run_root": str(run_root),
        "execution_config_ref": selected_ref, "platform_ref": platform, "controller_ref": _ref(__file__),
        "source_ref": _save(ROOT / "source-snapshot.json", _source_snapshot(public["source_root"], include_binary=True)),
        "request_ref": public["request_ref"], "public_preparation_ref": _ref(OLD / "public-preparation.json"),
        "baseline_ref": _ref(OLD / "baseline-result.json"), "model_identity": model, "environment": baseline["environment"],
        "repository_profile_ref": public["profile_ref"], "allowed_edit_paths": profile["allowed_edit_paths"],
        "experience": "none; terminal automatic extraction not configured; absolute capability diagnostic only",
        "wall_seconds": 1800, "request_timeout_seconds": 600, "policy": "meter_only", "saved_projection_ref": replay_ref,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "attempts_not_pooled": True, "model_calls_during_freeze": 0})
    print(json.dumps(plan))


def run():
    plan_ref = _ref(ROOT / "frozen-plan.json")
    plan = read_bound_json(plan_ref)
    if _ref(__file__) != plan["controller_ref"] or (ROOT / "started.json").exists() or Path(plan["run_root"]).exists():
        raise ValueError("INPUT_CHANGED_OR_ATTEMPT_ALREADY_STARTED")
    FrozenExecution(plan["platform_ref"]).verify_files()
    if _source_snapshot(read_bound_json(plan["public_preparation_ref"])["source_root"], include_binary=True) != read_bound_json(plan["source_ref"]):
        raise ValueError("PUBLIC_SOURCE_CHANGED")
    model = read_ollama_model_identity("http://127.0.0.1:11434", MODEL)
    if model["digest"] != DIGEST:
        raise ValueError("MODEL_CHANGED")
    clock = time.monotonic()
    warmed = api("generate", {"model": MODEL, "keep_alive": "30m", "stream": False, "options": {"num_ctx": 65536}}, timeout=180)
    warmup = _save(ROOT / "warmup.json", {"response": warmed, "seconds": time.monotonic() - clock,
        "scope": "model load only, no task prompt"})
    _save(ROOT / "started.json", {"plan_ref": plan_ref, "run_id": plan["run_id"], "execution_config_ref": plan["execution_config_ref"],
        "warmup_ref": warmup, "model_identity": model, "scripted_actions": False, "outer_supplied_solution": False})
    manifest = plan["platform_ref"]
    process = execute([sys.executable, "-I", "-B", str(SNAPSHOT / "scripts/run_frozen_product.py"),
        "--_child", manifest["path"], manifest["sha256"]], cwd=SNAPSHOT, stdin_path=None, output=ROOT / "execution",
        deadline=time.monotonic() + plan["wall_seconds"], environment={**os.environ, **plan["environment"],
            "PYTHONUTF8": "1", "DOTNET_CLI_UI_LANGUAGE": "en-US"})
    print(json.dumps({"run_id": plan["run_id"], "status": process["status"], "process_ref": _ref(ROOT / "execution/process.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.mode == "freeze" else run()
