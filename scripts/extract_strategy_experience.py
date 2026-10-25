"""Prepare public process inputs and perform at most one separately registered extraction.

为什么要有它：保留经历可以单独学习，不能恢复旧任务、重试未知调用或为提炼另建求解流程。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.core.trace.execution_trace import ExecutionTrace
from src.libs.llm.agent_turn import _chat_message, build_ollama_context_admission
from src.libs.llm.transport_observation import observe_transport
from src.mod_agent.artifacts import ArtifactStore
from src.mod_agent.local_entry import (
    LocalOllamaInitialPatchAdapter,
    LocalOllamaInitialPatchProfileV1,
    _reserve_request_model_call,
)
from src.mod_agent.model_admission import (
    assert_ollama_model_identity_stable,
    read_ollama_model_identity,
    validate_ollama_context_window,
    validate_ollama_model_identity,
)
from src.mod_agent.model_call_budget import METER_ONLY, validate_budget
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.patch_transport_bodies import PatchTransportBodyRecorder
from src.mod_agent.planning import process_trajectory, strategy_experience
from src.mod_agent.planning.process_trajectory import (
    file_ref,
    prepare_trajectory,
    project_verified,
    read_public,
    require,
    source_context,
    verify_source,
    write_once,
)
from src.mod_agent.planning.strategy_experience import StrategyExperience
from src.mod_agent.task_agent_v2_execution import (
    earliest_task_deadline_utc,
    task_deadline_utc_from_environment,
)
from src.mod_agent.task_agent_v2_state import _no_links

_CONTEXT_OWNER_PATH = sys.modules[build_ollama_context_admission.__module__].__file__


def profile_for(record):
    require(record.get("schema_version") == "strategy_extraction_input_v1", "INPUT_SCHEMA_INVALID")
    profile = LocalOllamaInitialPatchProfileV1("local_ollama_initial_patch_profile_v1", record["base_url"],
        record["model"], record["context_window"], record["output_budget"], record["request_timeout_seconds"])
    require(isinstance(record["digest"], str) and len(record["digest"]) == 64
        and all(c in "0123456789abcdef" for c in record["digest"]), "MODEL_DIGEST_REQUIRED")
    require(type(record["process_window_seconds"]) in {int, float} and math.isfinite(record["process_window_seconds"])
        and record["process_window_seconds"] > 0,
        "BOUNDED_EXTRACTION_WINDOW_REQUIRED")
    return profile


def fresh_root(path, protected):
    root = Path(path).absolute()
    _no_links(root)
    root = root.resolve()
    require(not root.exists(), "FRESH_OUTPUT_REQUIRED")
    require(not any(root.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(root) for p in protected),
        "OUTPUT_OVERLAP")
    root.mkdir(parents=True, exist_ok=False)
    return root


def owners():
    from scripts import dsh_run_request
    from src.core.trace import execution_trace
    from src.libs.llm import agent_turn, transport_observation
    from src.mod_agent import (
        interrupted_patch_admission,
        local_entry,
        model_admission,
        model_call_budget,
        model_qualification,
        patch_transport_bodies,
        platform_execution_snapshot,
        task_agent_execution,
        task_agent_loop,
        task_agent_v2_execution,
        task_agent_v2_recovery,
        task_agent_v2_state,
    )
    from src.mod_agent.planning import task_memory

    paths = [__file__, sys.modules[ArtifactStore.__module__].__file__,
        _CONTEXT_OWNER_PATH, *[module.__file__ for module in (
        process_trajectory, strategy_experience, task_memory, execution_trace, agent_turn, transport_observation,
        interrupted_patch_admission, local_entry, model_admission, model_call_budget, model_qualification,
        patch_transport_bodies, platform_execution_snapshot, task_agent_execution, task_agent_loop,
        task_agent_v2_execution, task_agent_v2_recovery, task_agent_v2_state, dsh_run_request)]]
    return [file_ref(path) for path in sorted(set(paths))]


def retain_owners(root):
    # 为什么要有它：当前路径的hash会在后续开发时失效，真实提炼必须同时保留可恢复的实际源码字节。
    records = []
    for reference in owners():
        path = root / "sources" / reference["sha256"] / Path(reference["path"]).name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(Path(reference["path"]).read_bytes())
        retained = file_ref(path)
        require(retained["sha256"] == reference["sha256"], "SOURCE_CHANGED_DURING_RETENTION")
        records.append({"original_ref": reference, "retained_ref": retained})
    return records


def prepare(input_ref, output_root):
    record = read_public(input_ref)
    profile = profile_for(record)
    require(sum(key in record for key in ("sources", "episode_ref", "episode_set")) == 1, "EXACTLY_ONE_INPUT_MODE_REQUIRED")
    protected = [Path(input_ref["path"]).parent]
    for source in record.get("sources", []):
        if source["kind"] == "public_episode":
            protected.append(source["allowed_source_root"])
        else:
            protected.extend([Path(source["artifact_store_root"]).parent, Path(source["platform_ref"]["path"]).parent,
                Path(source["process_ref"]["path"]).parent.parent])
    if "episode_ref" in record:
        protected.append(Path(record["episode_ref"]["path"]).parent)
    if "episode_set" in record:
        protected.append(record["episode_set"]["store_root"])
    root = fresh_root(output_root, protected)
    memory = StrategyExperience(ArtifactStore(root / "strategy-memory"))
    episode_refs = []
    if "sources" in record:
        require(isinstance(record["sources"], list) and record["sources"], "SOURCES_REQUIRED")
        for index, source in enumerate(record["sources"]):
            episode_refs.append(prepare_trajectory(source, root / "projections" / str(index)))
        run_ids = [read_public(reference)["run_id"] for reference in episode_refs]
        require(len(set(run_ids)) == len(run_ids), "RUN_DUPLICATED")
        selected = memory.freeze_episode_set(episode_refs, applicability=record["applicability"], source_roles=record.get("source_roles"))
        selection = {"kind": "episode_set", "reference": asdict(selected)}
    elif "episode_set" in record:
        source = record["episode_set"]
        original = StrategyExperience(ArtifactStore(source["store_root"])).read(source["reference"])
        require(original.get("kind") == "episode_set" and original.get("frozen") is True, "FROZEN_SET_REQUIRED")
        episode_refs = [row["episode_ref"] for row in original["episodes"]]
        selected = memory.freeze_episode_set(episode_refs, applicability=original["applicability"],
            source_roles=[row["source_role"] for row in original["episodes"]])
        selection = {"kind": "episode_set", "reference": asdict(selected)}
    else:
        episode = read_public(record["episode_ref"])
        require(episode.get("visibility") == "public" and episode.get("data_use") == "strategy_learning"
            and episode.get("schema_version") == "public_strategy_episode_v1", "PUBLIC_EPISODE_REQUIRED")
        memory._public_sources(episode["evidence_refs"])
        selection = {"kind": "episode", "reference": record["episode_ref"]}
        episode_refs = [record["episode_ref"]]
    retained = retain_owners(root)
    return write_once(root / "prepared.json", {"schema_version": "prepared_strategy_extraction_v1", "input_ref": input_ref,
        "profile": profile.to_dict(), "selection": selection, "episode_refs": episode_refs,
        "memory_root": str(memory.store.root), "owner_refs": [row["original_ref"] for row in retained], "retained_owner_sources": retained,
        "protected_roots": [str(Path(p).resolve()) for p in protected],
        "model_calls": 0, "scope": "prepared public inputs; no live provider admission or model request"})


def verify_prepared(prepared_ref):
    prepared = read_public(prepared_ref)
    require(prepared.get("schema_version") == "prepared_strategy_extraction_v1" and prepared["owner_refs"] == owners(),
        "PREPARED_IMPLEMENTATION_CHANGED")
    require([row["original_ref"] for row in prepared["retained_owner_sources"]] == prepared["owner_refs"], "RETAINED_OWNER_SET_CHANGED")
    for row in prepared["retained_owner_sources"]:
        require(Path(row["retained_ref"]["path"]).resolve().is_relative_to(Path(prepared_ref["path"]).resolve().parent / "sources")
            and file_ref(row["retained_ref"]["path"]) == row["retained_ref"]
            and row["retained_ref"]["sha256"] == row["original_ref"]["sha256"], "RETAINED_OWNER_CHANGED")
    record = read_public(prepared["input_ref"])
    require(profile_for(record).to_dict() == prepared["profile"], "PREPARED_PROFILE_CHANGED")
    require(len(record.get("sources", [])) in (0, len(prepared["episode_refs"])), "PREPARED_SOURCE_COUNT_CHANGED")
    for source, reference in zip(record.get("sources", []), prepared["episode_refs"], strict=False):
        verified = verify_source(source)
        trajectory, omissions = project_verified(verified)
        episode = read_public(reference)
        refs = {r["evidence_id"]: {k: r[k] for k in ("path", "sha256")} for r in episode["evidence_refs"]}
        metadata = read_public(refs["projection"])
        require(metadata["source"] == source and metadata["omissions"] == omissions
            and metadata["projection_owner_ref"] == file_ref(process_trajectory.__file__), "PROJECTION_SOURCE_CHANGED")
        trajectory["omission_manifest_ref"] = refs["projection"]
        contract_ref, applicability = source_context(verified)
        require(read_public(refs["trajectory"]) == {"trajectory": trajectory}
            and episode["selections"] == [{"evidence_id": "trajectory", "locator": ["trajectory"]}]
            and episode["run_id"] == verified["run_id"]
            and episode["task_contract_ref"] == contract_ref and episode["applicability"] == applicability,
            "PROJECTION_CHANGED")
    memory = StrategyExperience(ArtifactStore(prepared["memory_root"]))
    selection = prepared["selection"]
    if selection["kind"] == "episode_set":
        frozen = memory.read(selection["reference"])
        require([r["episode_ref"] for r in frozen["episodes"]] == prepared["episode_refs"], "PREPARED_SET_CHANGED")
        if "sources" in record:
            require(frozen["applicability"] == record["applicability"]
                and [r["source_role"] for r in frozen["episodes"]] == record.get("source_roles", ["context"] * len(record["sources"])),
                "PREPARED_SET_CONDITIONS_CHANGED")
        elif "episode_set" in record:
            source = record["episode_set"]
            original = StrategyExperience(ArtifactStore(source["store_root"])).read(source["reference"])
            require(all(frozen[k] == original[k] for k in ("episodes", "applicability", "authority", "frozen")), "PREPARED_SET_CHANGED")
    else:
        require(selection == {"kind": "episode", "reference": record["episode_ref"]}
            and prepared["episode_refs"] == [record["episode_ref"]], "PREPARED_EPISODE_CHANGED")
    return prepared, record


def request_payload(request, profile):
    require(not request.tools and not request.tool_results and not request.historical_tool_exchanges, "EXTRACTION_TOOLS_FORBIDDEN")
    return {"model": profile.model, "messages": [_chat_message(m) for m in request.messages], "stream": False,
        "think": False, "options": {"temperature": request.temperature, "num_predict": profile.output_budget,
            "seed": 0, "num_ctx": profile.context_window}, "format": request.response_format.schema}


def invoke(memory, prepared, client, admission_ref, before_call, after_call):
    selected = prepared["selection"]
    method = memory.propose_from_episodes if selected["kind"] == "episode_set" else memory.propose_from_episode
    return method(client, selected["reference"], admission_ref=admission_ref, before_call=before_call, after_call=after_call)


def _copy_selection(prepared, memory):
    """Give each offline preview or live attempt its own extraction store.

    为什么要有它：预览的未派发记录不应污染真实尝试，原经验及已结束运行均不能被重写。
    """
    result = deepcopy_json(prepared)
    if prepared["selection"]["kind"] == "episode_set":
        original = StrategyExperience(ArtifactStore(prepared["memory_root"])).read(prepared["selection"]["reference"])
        ref = memory.freeze_episode_set([r["episode_ref"] for r in original["episodes"]],
            applicability=original["applicability"], source_roles=[r["source_role"] for r in original["episodes"]])
        result["selection"]["reference"] = asdict(ref)
    return result


def deepcopy_json(value):
    return json.loads(json.dumps(value))


def preview(prepared_ref, output_root):
    prepared, record = verify_prepared(prepared_ref)
    root = fresh_root(output_root, [Path(prepared_ref["path"]).parent, *prepared["protected_roots"]])
    memory = StrategyExperience(ArtifactStore(root / "offline-preview-memory"))
    selected = _copy_selection(prepared, memory)
    profile = profile_for(record)
    admission = write_once(root / "offline-declared-identity.json", {"status": "admitted", "purpose": "strategy_learning",
        "provider": "ollama_chat_local", "model": profile.model, "profile_sha256": canonical_hash(profile.to_dict()),
        "origin": "offline_request_construction_only", "live_model_verified": False, "dispatch_allowed": False})
    captured = {}

    def inspect(request_ref):
        from src.mod_agent.interrupted_patch_admission import retained_request_from_document
        request = retained_request_from_document(memory.read(request_ref)["request"])
        payload = request_payload(request, profile)
        captured.update(request_ref=request_ref, payload_ref=write_once(root / "provider-request-preview.json", payload),
            admission=build_ollama_context_admission(payload=payload, configured_num_ctx=profile.context_window, num_predict=profile.output_budget))
        return False

    def never_send(_request):
        raise AssertionError("OFFLINE_PREVIEW_CANNOT_SEND")

    invoke(memory, selected, SimpleNamespace(provider="ollama_chat_local", model=profile.model, complete_turn=never_send),
        admission, inspect, lambda _receipt: None)
    return write_once(root / "preview.json", {"status": "preview_only", "prepared_ref": prepared_ref, **captured,
        "model_calls": 0, "live_model_verified": False, "scope": "exact request construction; internal pending_budget belongs only to offline preview"})


def execute(prepared_ref, output_root, deadline_utc, *, client_factory=None, identity_reader=None):
    prepared, record = verify_prepared(prepared_ref)
    profile = profile_for(record)
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    require(deadline.tzinfo is not None, "UTC_DEADLINE_REQUIRED")
    deadline = earliest_task_deadline_utc(deadline.timestamp(), task_deadline_utc_from_environment())
    remaining = deadline - time.time()
    require(0 < remaining <= record["process_window_seconds"], "REGISTERED_DEADLINE_REQUIRED")
    root = fresh_root(output_root, [Path(prepared_ref["path"]).parent, *prepared["protected_roots"]])
    # 为什么要有它：同一prepared输入只允许一次真实尝试；结果未知或失败也不能靠另换输出目录重新派发。
    marker = Path(prepared_ref["path"]).parent / "execution-started.json"
    started_ref = write_once(marker, {"prepared_ref": prepared_ref, "output_root": str(root), "deadline_utc": deadline})
    write_once(root / "started.json", {"prepared_ref": prepared_ref, "started_ref": started_ref, "owner_refs": owners(),
        "profile": profile.to_dict(), "deadline_utc": deadline, "parent_process_enforcement_required": True})
    memory = StrategyExperience(ArtifactStore(root / "strategy-memory"))
    selected = _copy_selection(prepared, memory)
    budget = validate_budget({"policy": METER_ONLY, "maximum": None, "remaining": None, "consumed": 0,
        "by_stage": {"initial_patch": {"maximum": None, "remaining": None, "consumed": 0}},
        "count_basis": "reserved_provider_invocations_not_transport_proof"})
    receipts, events = [], []
    run_id, clock = "strategy_extraction_" + uuid4().hex, time.monotonic()
    trace = ExecutionTrace(run_id, run_id, "public_process_strategy_extraction",
        recorder=lambda row: write_once(root / "trace" / (uuid4().hex + ".json"), row))
    result = {"status": "not_dispatched", "run_id": run_id, "prepared_ref": prepared_ref,
        "human_active_seconds": None, "scope": "new separately accounted extraction; no source task resumed", "strategy_ref": None}

    def time_left():
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError("EXTRACTION_DEADLINE_REACHED")
        return left

    def identity():
        time_left()
        observed = (identity_reader or read_ollama_model_identity)(profile.base_url, profile.model)
        validate_ollama_model_identity(observed, expected_base_url=profile.base_url, expected_model=profile.model)
        validate_ollama_context_window(observed, requested_context_window=profile.context_window)
        require(observed["digest"] == record["digest"], "MODEL_DIGEST_CHANGED")
        return observed

    def reserve(request_ref):
        from src.mod_agent.interrupted_patch_admission import retained_request_from_document
        time_left()
        request = retained_request_from_document(memory.read(request_ref)["request"])
        payload = request_payload(request, profile)
        admission = build_ollama_context_admission(payload=payload, configured_num_ctx=profile.context_window, num_predict=profile.output_budget)
        write_once(root / "context-preflight.json", {"request_ref": request_ref, "payload_ref": write_once(root / "preflight-provider-request.json", payload),
            "context_admission": admission, "model_calls": 0})
        require(admission["admitted"] is True, "REQUEST_CONTEXT_EXCEEDED")
        if budget["consumed"]:
            return False
        updated = _reserve_request_model_call(budget, "initial_patch")
        budget.clear()
        budget.update(updated)
        write_once(root / "reservation.json", {"request_ref": request_ref, "budget": budget})
        return True

    def account(receipt):
        receipts.append(receipt)
        write_once(root / "call-receipt.json", receipt)

    try:
        observed = identity()
        write_once(root / "identity-before.json", observed)
        admission = write_once(root / "admission.json", {"status": "admitted", "purpose": "strategy_learning",
            "provider": "ollama_chat_local", "model": profile.model, "model_identity": observed,
            "profile_sha256": canonical_hash(profile.to_dict()), "origin": "new_registered_extraction_not_source_task_resume"})

        def complete(request):
            current = identity()
            assert_ollama_model_identity_stable(expected=observed, observed=current)
            client = (client_factory or LocalOllamaInitialPatchAdapter._default_agent_turn_client)(profile)
            client.bind_context_identity(model_identity=current, owner_profile_sha256=canonical_hash(profile.to_dict()))
            # The profile identity stays fixed; the transport timeout only shortens with the parent deadline.
            # 为什么要有它：剩余墙钟不能被新客户端恢复为完整超时，配置身份也不能被随意替换。
            if hasattr(client, "request_timeout_seconds"):
                client.request_timeout_seconds = min(profile.timeout_seconds, time_left())
                if hasattr(client._send, "timeout_seconds"):
                    client._send.timeout_seconds = client.request_timeout_seconds
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
            learned = invoke(memory, selected, SimpleNamespace(provider="ollama_chat_local", model=profile.model, complete_turn=complete),
                admission, reserve, account)
        result.update(status=learned["status"], strategy_ref=learned.get("strategy_ref"), extraction_ref=learned.get("extraction_ref"))
        if result["strategy_ref"]:
            strategy = memory.read(result["strategy_ref"])
            require(strategy["state"] == "proposed" and strategy["origin"]["kind"] == "model", "MODEL_PROPOSED_ONLY_REQUIRED")
        write_once(root / "identity-after.json", identity())
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, message=str(error))
    result.update(budget=budget, call_receipts=receipts, provider_events=events, elapsed_seconds=time.monotonic() - clock,
        unknown_usage_calls=max(0, budget["consumed"] - sum(r.get("usage") is not None for r in receipts)),
        trace_recording_failures=trace.failure_statistics(), model_calls_reserved=budget["consumed"])
    return write_once(root / "result.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("prepare", "preview", "execute"))
    parser.add_argument("--input-record")
    parser.add_argument("--prepared-record")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--deadline-utc")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        if not args.input_record:
            parser.error("prepare requires --input-record")
        reference = prepare(file_ref(args.input_record), args.output_root)
    else:
        if not args.prepared_record:
            parser.error("preview/execute require --prepared-record")
        if args.mode == "preview":
            reference = preview(file_ref(args.prepared_record), args.output_root)
        else:
            if not args.deadline_utc:
                parser.error("execute requires --deadline-utc")
            reference = execute(file_ref(args.prepared_record), args.output_root, args.deadline_utc)
    result = read_public(reference)
    print(json.dumps({"result_ref": reference, "status": result.get("status", "prepared"),
        "model_calls_reserved": result.get("model_calls_reserved", 0)}))
    return 0 if result.get("status") in {None, "preview_only", "completed", "insufficient_information"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
