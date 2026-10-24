"""One explicitly triggered extraction API experiment, never a product runner.

为什么要有它：真实保留反例可单独检验提炼，旧未知尝试不能被重放或清零。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.core.trace.execution_trace import ExecutionTrace  # noqa: E402
from src.libs.llm.transport_observation import observe_transport  # noqa: E402
from src.mod_agent.artifacts import ArtifactStore  # noqa: E402
from src.mod_agent.evaluation.independent_preparation import file_ref, write_once  # noqa: E402
from src.mod_agent.local_entry import (  # noqa: E402
    LocalOllamaInitialPatchAdapter,
    LocalOllamaInitialPatchProfileV1,
    _reserve_request_model_call,
)
from src.mod_agent.model_admission import (  # noqa: E402
    assert_ollama_model_identity_stable,
    read_ollama_model_identity,
    validate_ollama_context_window,
    validate_ollama_model_identity,
)
from src.mod_agent.model_call_budget import (  # noqa: E402
    METER_ONLY,
    call_allowance,
    validate_budget,
)
from src.mod_agent.model_qualification import canonical_hash, read_bound_json  # noqa: E402
from src.mod_agent.patch_transport_bodies import PatchTransportBodyRecorder  # noqa: E402
from src.mod_agent.planning.strategy_experience import StrategyExperience  # noqa: E402
from src.mod_agent.task_agent_v2_execution import (  # noqa: E402
    earliest_task_deadline_utc,
    task_deadline_utc_from_environment,
)
from src.mod_agent.task_agent_v2_state import _no_links  # noqa: E402

INTENT = Path(__file__).with_name("intent.json")
EPISODE_HASH = "dcd7dd926299d46bbf03b087563e58986ca344c6c7d6fc5ff53a23aaf9ad298c"


def preflight(intent_path=None):
    intent = read_bound_json(file_ref(Path(intent_path) if intent_path is not None else INTENT))
    if intent["episode_ref"]["sha256"] != EPISODE_HASH:
        raise ValueError("EXACT_RETAINED_EPISODE_REQUIRED")
    episode = read_bound_json(intent["episode_ref"])
    if (episode.get("visibility") != "public" or episode.get("data_use") != "strategy_learning"
            or len(episode["evidence_refs"]) != 25):
        raise ValueError("EXACT_PUBLIC_EPISODE_REQUIRED")
    original = Path(intent["original_run_root"]).resolve(strict=True)
    refs = [intent["episode_ref"]]
    for source in episode["evidence_refs"]:
        path = Path(source["path"])
        _no_links(path)
        if (source.get("kind") != "json_file" or source.get("visibility") != "public"
                or "strategy_learning" not in source.get("allowed_uses", [])
                or not path.resolve().is_relative_to(original)
                or {"hidden", "evaluation-only"}.intersection(p.casefold() for p in path.parts)):
            raise ValueError("PUBLIC_SOURCE_BOUNDARY")
        read_bound_json({"path": source["path"], "sha256": source["sha256"]})
        refs.append(file_ref(path))
    for path in sorted(Path(intent["original_store"]).rglob("*")):
        _no_links(path)
        if path.is_file():
            refs.append(file_ref(path))
    return intent, refs


def output_path(value, intent):
    path = Path(value)
    _no_links(path.absolute())
    if not path.is_absolute() or path.exists():
        raise ValueError("NEW_ABSOLUTE_OUTPUT_REQUIRED")
    path = path.resolve()
    original = Path(intent["original_run_root"]).resolve()
    if path.is_relative_to(original) or original.is_relative_to(path):
        raise ValueError("ORIGINAL_OUTPUT_OVERLAP")
    return path


def verify_original_refs(original_refs):
    integrity = []
    for reference in original_refs:
        try:
            _no_links(Path(reference["path"]).absolute())
            actual = file_ref(reference["path"])
            same = (actual["sha256"] == reference["sha256"]
                and Path(actual["path"]).resolve() == Path(reference["path"]).resolve())
            integrity.append({"reference": reference, "actual": actual, "status": "matched" if same else "changed"})
        except (OSError, ValueError) as error:
            integrity.append({"reference": reference, "status": "unreadable", "error_type": type(error).__name__})
    unchanged = (False if any(row["status"] == "changed" for row in integrity)
        else None if any(row["status"] == "unreadable" for row in integrity) else True)
    return unchanged, integrity


def execute(output_root, deadline_utc, base_url, intent_path=None):
    selected_intent = Path(intent_path) if intent_path is not None else INTENT
    intent, original_refs = preflight(selected_intent)
    root = output_path(output_root, intent)
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("PARENT_UTC_DEADLINE_WITHIN_600_SECONDS_REQUIRED")
    deadline = datetime.fromtimestamp(earliest_task_deadline_utc(
        deadline.timestamp(), task_deadline_utc_from_environment()), timezone.utc)
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if not 0 < remaining <= 600:
        raise ValueError("PARENT_UTC_DEADLINE_WITHIN_600_SECONDS_REQUIRED")
    monotonic_deadline = time.monotonic() + remaining
    profile = LocalOllamaInitialPatchProfileV1("local_ollama_initial_patch_profile_v1", base_url,
        intent["model"], intent["context_window"], intent["output_budget"], intent["request_timeout_seconds"])
    root.mkdir(parents=True, exist_ok=False)
    run_id, start = "retained_extraction_" + uuid4().hex, time.monotonic()
    producer_ref = file_ref(sys.modules[StrategyExperience.__module__].__file__)
    write_once(root / "started.json", {"run_id": run_id, "intent_ref": file_ref(selected_intent),
        "producer_ref": producer_ref,
        "script_ref": file_ref(__file__), "original_refs": original_refs,
        "requested_deadline_utc": deadline_utc, "deadline_utc": deadline.isoformat(),
        "scope": intent["scope"], "profile": profile.to_dict(), "parent_process_enforcement_required": True})
    budget = validate_budget({"policy": METER_ONLY, "count_basis": "reserved_provider_invocations_not_transport_proof",
        "maximum": None, "remaining": None, "consumed": 0,
        "by_stage": {"initial_patch": {"maximum": None, "remaining": None, "consumed": 0}}})
    receipts, events = [], []
    trace = ExecutionTrace(run_id, run_id, "retained_episode_strategy_extraction",
        recorder=lambda row: write_once(root / "trace" / (uuid4().hex + ".json"), row))
    result = {"run_id": run_id, "scope": intent["scope"], "producer_ref": producer_ref,
        "status": "not_dispatched", "strategy_ref": None,
        "original_B3_extraction_cost": "unknown_preserved_not_included", "human_active_seconds": None,
        "platform_development_cost": "separate_not_measured_by_this_attempt"}

    def time_left():
        left = min(monotonic_deadline - time.monotonic(),
            (deadline - datetime.now(timezone.utc)).total_seconds())
        if left <= 0:
            raise TimeoutError("PARENT_DEADLINE_REACHED")
        return left

    def identity():
        time_left()
        observed = read_ollama_model_identity(profile.base_url, profile.model)
        validate_ollama_model_identity(observed, expected_base_url=profile.base_url, expected_model=profile.model)
        validate_ollama_context_window(observed, requested_context_window=profile.context_window)
        if observed["digest"] != intent["digest"]:
            raise ValueError("MODEL_DIGEST_CHANGED")
        return observed

    def reserve(request_ref):
        time_left()
        if budget["consumed"] or call_allowance(budget, "initial_patch") <= 0:
            return False
        updated = _reserve_request_model_call(budget, "initial_patch")
        budget.clear()
        budget.update(updated)
        write_once(root / "reservation.json", {"request_ref": request_ref, "budget": validate_budget(budget)})
        return True

    def account(receipt):
        receipts.append(receipt)
        write_once(root / "call-receipt.json", receipt)

    try:
        before = identity()
        write_once(root / "identity-before.json", before)
        admission = write_once(root / "admission.json", {"status": "admitted", "purpose": "strategy_learning",
            "provider": "ollama_chat_local", "model": profile.model, "model_identity": before,
            "origin": "new_outer_triggered_attempt_not_original_task_resume"})
        memory = StrategyExperience(ArtifactStore(root / "strategy-memory"))

        def complete(request):
            observed = identity()
            assert_ollama_model_identity_stable(expected=before, observed=observed)
            selected_profile = replace(profile, timeout_seconds=min(600, time_left()))
            client = LocalOllamaInitialPatchAdapter._default_agent_turn_client(selected_profile)
            client.bind_context_identity(model_identity=observed, owner_profile_sha256=canonical_hash(selected_profile.to_dict()))
            event = {"request_ref": write_once(root / "application-request.json", asdict(request))}
            events.append(event)
            try:
                with trace.model_call({"provider": client.provider, "model": client.model}) as observation:
                    bodies = PatchTransportBodyRecorder(record=event, observation=observation, body_root=root)
                    with observe_transport(observation.transport_started, observation.update,
                            request_callback=observation.update, local_request_body_callback=bodies.callback):
                        time_left()
                        response = client.complete_turn(request)
                    event["response_ref"] = write_once(root / "application-response.json", asdict(response))
                    observation.record_response(usage=response.usage.to_dict(), usage_available=True,
                        response_sha256=canonical_hash(asdict(response)), response_origin="live_provider_response")
                    bodies.publish_refs()
                    return response
            finally:
                write_once(root / "transport-event.json", event)

        with trace.activate():
            extracted = memory.propose_from_episode(SimpleNamespace(provider="ollama_chat_local", model=profile.model,
                complete_turn=complete), intent["episode_ref"], admission_ref=admission, before_call=reserve, after_call=account)
        result.update(status=extracted["status"], extraction_ref=extracted.get("extraction_ref"),
            strategy_ref=extracted.get("strategy_ref"))
        if result["strategy_ref"]:
            strategy = memory.read(result["strategy_ref"])
            if strategy["state"] != "proposed" or strategy["origin"]["kind"] != "model":
                raise ValueError("MODEL_PROPOSED_ONLY_REQUIRED")
        after = identity()
        assert_ollama_model_identity_stable(expected=before, observed=after)
        write_once(root / "identity-after.json", after)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, message=str(error))
    unchanged, integrity = verify_original_refs(original_refs)
    result["extraction_status"] = result["status"]
    if unchanged is not True:
        result["status"] = "integrity_unconfirmed"
    result.update(budget=validate_budget(budget), call_receipts=receipts, provider_events=events,
        elapsed_seconds=time.monotonic() - start, trace_recording_failures=trace.failure_statistics(),
        original_refs_unchanged=unchanged, original_integrity=integrity)
    return write_once(root / "result.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root")
    parser.add_argument("--deadline-utc")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--intent", help="Explicit registered intent; default retains the original experiment.")
    args = parser.parse_args(argv)
    if not args.execute:
        intent, refs = preflight(args.intent)
        print(json.dumps({"status": "preflight_only", "scope": intent["scope"], "verified_original_files": len(refs),
            "model_calls": 0, "network_calls": 0, "writes": 0}))
        return 0
    if not args.output_root or not args.deadline_utc:
        parser.error("--execute requires --output-root and parent --deadline-utc")
    reference = execute(args.output_root, args.deadline_utc, args.base_url, args.intent)
    print(json.dumps(reference))
    return 0 if read_bound_json(reference)["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
