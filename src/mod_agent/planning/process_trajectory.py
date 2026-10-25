"""Deterministic public process projection; never resume a source task.

为什么要有它：跨经历提炼应保留完整行动与原文，省略正文不能丢失身份或把未知执行写成结果。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.planning.learning_request import resolve_task_context, task_context_selections
from src.mod_agent.planning.task_memory import _select
from src.mod_agent.platform_execution_snapshot import FrozenExecution
from src.mod_agent.task_agent_v2_recovery import _validate_completed, _validate_host_binding
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, _no_links

VERSION = "complete_public_process_trajectory_v2"


def require(value, code):
    if not value:
        raise ValueError("PROCESS_TRAJECTORY_" + code)


def file_ref(path):
    path = Path(path).absolute()
    _no_links(path)
    require(path.is_file(), "FILE_REQUIRED")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_public(reference, *, root=None):
    path = Path(reference["path"])
    require(path.is_absolute() and not {"hidden", "evaluation-only"}.intersection(p.casefold() for p in path.parts),
        "NONPUBLIC_SOURCE")
    _no_links(path)
    if root is not None:
        require(path.resolve().is_relative_to(Path(root).resolve()), "REFERENCE_OUTSIDE_STORE")
    return read_bound_json(reference)


def write_once(path, value):
    path = Path(path).absolute()
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return file_ref(path)


def verify_retained_platform(frozen):
    """Historical reads bind retained external bytes, never changed live inputs.

    为什么要有它：已结束任务的来源不能随主树开发失效，也不能回退到新字节；生产派发和恢复仍使用原严格校验。
    """
    document, root = frozen.document, frozen.root.resolve()
    for name, expected in document["files"].items():
        path = root / name
        _no_links(path)
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(root), "FROZEN_PATH_ESCAPE")
        require(file_ref(path)["sha256"] == expected["sha256"], "FROZEN_FILE_CHANGED")
    require(file_ref(document["archive_ref"]["path"]) == document["archive_ref"], "FROZEN_ARCHIVE_CHANGED")
    for external in document["external_refs"]:
        retained = external["retained_path"]
        require(external["role"] in {"bound_input", "extra_python_load"}
            and retained == "external/" + external["sha256"] + "/" + Path(external["path"]).name
            and document["files"].get(retained, {}).get("sha256") == external["sha256"], "RETAINED_EXTERNAL_CHANGED")
        require(file_ref(root / retained)["sha256"] == external["sha256"], "RETAINED_EXTERNAL_CHANGED")


def verify_public_episode(source):
    """Consume only an existing episode's explicit public selectors, without checkpoints."""
    root = Path(source["allowed_source_root"]).absolute()
    _no_links(root)
    episode = read_public(source["episode_ref"], root=root)
    require(episode.get("schema_version") == "public_strategy_episode_v1" and episode.get("visibility") == "public"
        and episode.get("data_use") == "strategy_learning", "PUBLIC_EPISODE_REQUIRED")
    run_id = source.get("run_id", episode.get("run_id"))
    require(isinstance(run_id, str) and run_id and run_id in Path(source["episode_ref"]["path"]).parts
        and episode.get("run_id", run_id) == run_id, "PUBLIC_EPISODE_RUN_CHANGED")
    documents, references = {}, {}
    selected_ids = {selection["evidence_id"] for selection in [*episode["selections"],
        *episode.get("task_context_selections", [])]}
    receipt_ids = {selection["evidence_id"] + "_result" for selection in episode["selections"]
        if len(selection["locator"]) == 3 and selection["locator"][0] == "tool_calls"
        and selection["locator"][2] in {"name", "arguments"}}
    needed_ids = selected_ids | receipt_ids
    for ref in episode["evidence_refs"]:
        require(ref.get("kind") == "json_file" and ref.get("visibility") == "public"
            and "strategy_learning" in ref.get("allowed_uses", []) and ref["evidence_id"] not in references,
            "PUBLIC_EPISODE_SOURCE_INVALID")
        reference = {key: ref[key] for key in ("path", "sha256")}
        references[ref["evidence_id"]] = reference
        if ref["evidence_id"] in needed_ids:
            require_public_selection_path(reference)
            documents[ref["evidence_id"]] = read_public(reference, root=root)
    resolve_task_context(episode, documents)
    selected, actions, order, facts = {}, {}, [], []
    for selection in episode["selections"]:
        evidence_id, locator = selection["evidence_id"], selection["locator"]
        value = _select(documents[evidence_id], locator)
        selected.setdefault(evidence_id, []).append((locator, value))
        if len(locator) == 3 and locator[0] == "tool_calls" and locator[2] in {"name", "arguments"}:
            key = (evidence_id, locator[1])
            if key not in actions:
                actions[key] = {}
                order.append(key)
            actions[key][locator[2]] = value
        elif not evidence_id.startswith("public_action_"):
            facts.append({"evidence_id": evidence_id, "locator": locator, "value": value})
    records, responses, seen_responses, seen_calls = [], [], {}, set()
    for index, (evidence_id, call_index) in enumerate(order):
        action = actions[(evidence_id, call_index)]
        require(set(action) == {"name", "arguments"}, "PUBLIC_ACTION_SELECTION_INCOMPLETE")
        result_id = evidence_id + "_result"
        require(result_id in documents, "PUBLIC_ACTION_RECEIPT_MISSING")
        response, receipt = documents[evidence_id], documents[result_id]
        call = response["tool_calls"][call_index]
        require(call["call_id"] not in seen_calls, "PUBLIC_ACTION_DUPLICATED")
        seen_calls.add(call["call_id"])
        origin = receipt["origin"]
        require(origin["name"] == action["name"] and origin["call_id"] == call["call_id"]
            and origin["arguments"] == action["arguments"]
            and origin["response_ref"]["sha256"] == references[evidence_id]["sha256"]
            and Path(origin["response_ref"]["path"]).resolve() == Path(references[evidence_id]["path"]).resolve(),
            "PUBLIC_ACTION_RECEIPT_CHANGED")
        output = {locator[1]: value for locator, value in selected.get(result_id, []) if len(locator) == 2 and locator[0] == "output"}
        statements = [value for locator, value in selected[evidence_id] if locator == ["content"]]
        visible = statements[0] if statements else None
        record = {"call": {"name": call["name"], "call_id": call["call_id"], "response_ref": references[evidence_id],
            "arguments_ref": references[evidence_id], "execution_receipt_ref": references[result_id]},
            "arguments": deepcopy(action["arguments"]), "arguments_locator_prefix": ["tool_calls", call_index, "arguments"],
            "response": {"content": visible}, "receipt": {"output": output}}
        records.append(record)
        response_identity = (Path(references[evidence_id]["path"]).resolve(), references[evidence_id]["sha256"])
        if response_identity not in seen_responses:
            seen_responses[response_identity] = len(responses)
            responses.append({"turn": None, "selected_response_ordinal": len(responses) + 1, "response_ref": references[evidence_id],
                "response": {"content": visible}, "content_selected": bool(statements)})
        elif statements:
            responses[seen_responses[response_identity]].update(response={"content": visible}, content_selected=True)
    for record in records:
        reference = record["call"]["response_ref"]
        identity = (Path(reference["path"]).resolve(), reference["sha256"])
        record["response"] = deepcopy(responses[seen_responses[identity]]["response"])
    return {"source": deepcopy(source), "episode": episode, "records": records, "pending": [], "responses": responses,
        "run_id": run_id, "request_id": episode.get("request_id"), "process": {"status": "not_selected_in_original_episode"},
        "task_context_selections": deepcopy(episode.get("task_context_selections", [])),
        "task_context_documents": {row["evidence_id"]: documents[row["evidence_id"]]
            for row in episode.get("task_context_selections", [])},
        "task_context_references": {row["evidence_id"]: references[row["evidence_id"]]
            for row in episode.get("task_context_selections", [])},
        "public_selected_facts": facts, "coverage": "Only the original public episode selectors; no full checkpoint, unselected response content, private evaluation or inferred completion."}


def require_public_selection_path(reference):
    """Reject developer-state/private selectors before reading any selected bytes.

    为什么要有它：public标签不能授权回读检查点或私有评测；未选择的引用只保留metadata，不为投影展开读取。
    """
    parts = {part.casefold() for part in Path(reference["path"]).parts}
    require(not parts.intersection({"session-checkpoint", "session-checkpoints", "host-recovery-config", "technical-input-audit",
        "hidden", "evaluation-only", "independent-evaluation", "run_result.json", "run_manifest.json"}), "NONPUBLIC_SELECTION")


def verify_source(source):
    """Bind an explicit checkpoint, actual process and registered frozen request."""
    from scripts.dsh_run_request import _configuration_arguments

    require(source.get("kind") in {"checkpoint", "public_episode"}, "SOURCE_KIND_INVALID")
    if source["kind"] == "public_episode":
        require("checkpoint_ref" not in source, "PUBLIC_EPISODE_DOES_NOT_READ_CHECKPOINTS")
        return verify_public_episode(source)
    store = Path(source["artifact_store_root"]).absolute()
    _no_links(store)
    store = store.resolve(strict=True)
    checkpoint = read_public(source["checkpoint_ref"], root=store)
    path = Path(source["checkpoint_ref"]["path"])
    require(checkpoint.get("schema") == "task-agent-session-checkpoint.v2"
        and path.parent == store / "v2/session-checkpoint" and path.stem == source["checkpoint_ref"]["sha256"],
        "CHECKPOINT_INVALID")
    state = TaskAgentV2State(store, checkpoint["task_contract_ref"])
    host = state.load(checkpoint["host_config_ref"])
    _validate_host_binding(state, checkpoint, host)
    _validate_completed(checkpoint, store)
    for reference in host["owner_refs"]:
        require(file_ref(reference["path"]) == reference, "OWNER_CHANGED")
    read_public(source["execution_config_ref"])
    arguments, options = _configuration_arguments(source["execution_config_ref"]["path"])
    run_id, request_id = checkpoint["details"]["run_id"], checkpoint["details"]["request_id"]
    require(options.get("--run-id") == run_id and options.get("--request-id") == request_id
        and host["graph_state"]["run_id"] == run_id and host["graph_state"]["request_id"] == request_id
        and (Path(options["--workspace-root"]) / run_id / "a").resolve() == store, "REGISTERED_RUN_CHANGED")
    frozen = FrozenExecution(source["platform_ref"])
    verify_retained_platform(frozen)
    document = frozen.document
    business = document["business_argv"]
    require(business and business[0] == "run-request" and "--" in business
        and business[1:business.index("--")] == arguments
        and Path(document["selected_run_root"]).resolve() == store.parent, "FROZEN_REQUEST_CHANGED")
    for reference in host["owner_refs"]:
        require(Path(reference["path"]).resolve().is_relative_to(frozen.root), "OWNER_OUTSIDE_FROZEN_PLATFORM")
    process = read_public(source["process_ref"])
    argv = process.get("argv", [])
    require(len(argv) == 7 and argv[1:3] == ["-I", "-B"] and argv[4] == "--_child"
        and Path(argv[0]).resolve() == Path(document["python_executable"]).resolve()
        and Path(argv[3]).resolve() == frozen.root / "scripts/run_frozen_product.py"
        and Path(argv[5]).resolve() == Path(source["platform_ref"]["path"]).resolve()
        and argv[6] == source["platform_ref"]["sha256"], "PROCESS_ARGV_CHANGED")
    if source.get("plan_ref"):
        plan = read_public(source["plan_ref"])
        # 为什么要有它：同一绝对路径的斜杠表示不改变原件身份，路径和原SHA仍必须同时匹配。
        require(plan["run_id"] == run_id
            and Path(plan["platform_ref"]["path"]).resolve() == Path(source["platform_ref"]["path"]).resolve()
            and plan["platform_ref"]["sha256"] == source["platform_ref"]["sha256"]
            and Path(plan["run_root"]).resolve() == store.parent and plan["command"] == argv
            and Path(source["process_ref"]["path"]).resolve() == Path(plan["root"]).resolve() / "execution/process.json",
            "PROCESS_PLAN_CHANGED")
        cwd = Path(plan["workspace"]).resolve()
    else:
        cwd = frozen.root
        require(Path(source["process_ref"]["path"]).resolve() == store.parent.parent.parent / "execution/process.json",
            "PROCESS_LOCATION_CHANGED")
    require(Path(process["cwd"]).resolve() == cwd and process.get("started") is True
        and process.get("finished_at") and process.get("status") in {"timed_out", "exited", "failed", "terminated", "interrupted", "cancelled"},
        "PROCESS_NOT_CONFIRMED_STOPPED")
    records = checkpoint["details"]["completed_tool_records"]
    completed_ids = [e["call_id"] for e in checkpoint["loop"]["events"] if e.get("event") == "tool_dispatch_completed"]
    require([r["call_id"] for r in records] == completed_ids and len(set(completed_ids)) == len(completed_ids),
        "COMPLETED_PREFIX_CHANGED")
    calls = {row["call_id"]: row for row in checkpoint["session"]["actual_calls"]}
    require(len(calls) == len(checkpoint["session"]["actual_calls"]), "CALL_ID_DUPLICATED")
    verified = []
    for record in records:
        call = calls[record["call_id"]]
        require(all(record.get(k) == v for k, v in call.items()), "ACTION_CHANGED")
        response = read_public(call["response_ref"], root=store)
        arguments = read_public(call["arguments_ref"], root=store)
        receipt = read_public(call["execution_receipt_ref"], root=store)
        require([r for r in response["tool_calls"] if r["call_id"] == call["call_id"]]
            == [{"name": call["name"], "call_id": call["call_id"], "arguments": arguments}], "MODEL_ACTION_CHANGED")
        verified.append({"call": call, "response": response, "arguments": arguments, "receipt": receipt})
    pending = []
    for call in calls.values():
        if call["call_id"] not in completed_ids:
            response = read_public(call["response_ref"], root=store)
            arguments = read_public(call["arguments_ref"], root=store)
            require(any(r == {"name": call["name"], "call_id": call["call_id"], "arguments": arguments}
                for r in response["tool_calls"]), "PENDING_MODEL_ACTION_CHANGED")
            record = checkpoint["loop"]["call_records"].get(call["call_id"], {})
            pending.append({"call": call, "response": response, "arguments": arguments,
                "execution_status": record.get("status"), "recorded_feedback": record.get("result", {}).get("output")})
    responses = []
    from src.mod_agent.task_agent_execution import _admitted_response

    for turn in checkpoint["details"].get("provider_turns", []):
        if turn.get("response_ref"):
            response = read_public(turn["response_ref"], root=store)
            _admitted_response(response, checkpoint["details"]["model_profile"])
            responses.append({"turn": turn["turn"], "response_ref": turn["response_ref"], "response": response})
    return {"source": deepcopy(source), "checkpoint": checkpoint, "host": host, "process": process,
        "records": verified, "pending": pending, "responses": responses, "run_id": run_id, "request_id": request_id,
        "task_context_document": deepcopy(state.contract),
        "task_context_reference": file_ref(store / checkpoint["task_contract_ref"])}


def task_context_evidence(verified):
    """Return only verified or originally selected task-contract fields.

    为什么要有它：原任务需求有自己的来源；旧episode的身份引用不能变成补读权限。
    """
    refs, documents, selectors = [], {}, []
    if "task_context_document" in verified:
        key = "source_task_contract"
        documents[key] = verified["task_context_document"]
        selectors = task_context_selections(documents[key], key)
        if selectors:
            refs.append({"kind": "json_file", "evidence_id": key, **verified["task_context_reference"],
                "visibility": "public", "allowed_uses": ["strategy_learning"]})
    else:
        for row in verified.get("task_context_selections", []):
            original_id, key = row["evidence_id"], "original_task_" + row["evidence_id"]
            if key not in documents:
                documents[key] = verified["task_context_documents"][original_id]
                refs.append({"kind": "json_file", "evidence_id": key, **verified["task_context_references"][original_id],
                    "visibility": "public", "allowed_uses": ["strategy_learning"]})
            selectors.append({**deepcopy(row), "evidence_id": key})
    return refs, selectors, documents


def project_verified(verified):
    """Keep all actions and verbatim statements; omit only declared process-irrelevant bodies."""
    cp, actions, statements, refs, paths, omissions = verified.get("checkpoint", {}), [], {}, {}, {}, []
    loop = cp.get("loop", {"phase": "not_selected_in_original_episode"})

    def alias(value):
        if isinstance(value, str) and value.startswith("v2/"):
            if value not in refs:
                refs[value] = "r" + str(len(refs))
            return refs[value]
        if isinstance(value, dict):
            return {k: alias(v) for k, v in value.items()}
        if isinstance(value, list):
            return [alias(v) for v in value]
        return value

    def omit(index, field, value, reference, locator):
        omissions.append({"action": index, "field": field, "reference": reference, "locator": locator,
            "json_sha256": canonical_hash(value), "json_bytes": len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()),
            "characters": len(value) if isinstance(value, str) else None})

    for index, row in enumerate([*verified["records"], *verified["pending"]]):
        call, arguments, response = row["call"], deepcopy(row["arguments"]), row["response"]
        pending = "receipt" not in row
        output = row.get("receipt", {}).get("output", {})
        statement = response.get("content")
        require(statement is None or isinstance(statement, str), "MODEL_STATEMENT_NOT_TEXT")
        if statement:
            digest = hashlib.sha256(statement.encode()).hexdigest()
            statements.setdefault(digest, {"text": statement, "actions": [], "turns": []})["actions"].append(index)
        revision = arguments.get("parent_revision_ref", arguments.get("source_revision_ref"))
        args = {k: v for k, v in arguments.items() if k not in {"parent_revision_ref", "source_revision_ref", "source_view_ref",
            "catalog_ref", "probe_artifact_ref", "expected_registry_ref"}}
        if args.get("root_id") == "target":
            args.pop("root_id")
        if isinstance(args.get("path"), str):
            path = args["path"]
            paths.setdefault(path, "p" + str(len(paths)))
            args["path"] = paths[path]
        for group, fields in (("edits", {"before_text", "after_text"}), ("files", {"content"})):
            if not isinstance(args.get(group, []), list):
                continue
            for item_index, item in enumerate(args.get(group, [])):
                if not isinstance(item, dict):
                    continue
                for key in list(item):
                    if key in fields:
                        value = item.pop(key)
                        omit(index, "arguments." + group + "." + key, value, call["arguments_ref"],
                            [*row.get("arguments_locator_prefix", []), group, item_index, key])
                        item[key + "_chars"] = len(value) if isinstance(value, str) else None
                    elif key in {"binding_id", "source_view_ref"}:
                        item[key] = alias(item[key])
        kept = {k: v for k, v in output.items() if k in {"status", "executed", "reason", "code", "error_code", "message",
            "argument_errors", "correctable", "execution_origin", "truncated", "offset", "next_offset", "total_chars",
            "total_matches", "catalog_ref", "probe_artifact_ref", "candidate_ref", "check_ref", "grants_acceptance"}}
        for field, value in output.items():
            if field in {"text", "matches", "observations", "changes"}:
                omit(index, "output." + field, value, call["execution_receipt_ref"], ["output", field])
        # 为什么要有它：公开失败内容属于行动结果，不能为了上下文合格而隐去已发生的反例。
        if call["name"] in {"candidate_check", "experiment_run"} and output.get("observations"):
            kept["observations"] = deepcopy(output["observations"])
            omissions[:] = [o for o in omissions if not (o["action"] == index and o["field"] == "output.observations")]
        if pending:
            kept = ({"execution": "not_dispatched", "feedback": row.get("recorded_feedback")}
                if row.get("execution_status") == "not_dispatched" else {"execution": "unknown_not_in_completed_prefix"})
        actions.append([index, call["name"], alias(revision), alias(args), alias(kept)])
    # 为什么要有它：没有工具调用的完整回答也可能明确表达判断，不能只保留已经执行的动作对应文字。
    for row in verified["responses"]:
        statement = row["response"].get("content")
        require(statement is None or isinstance(statement, str), "MODEL_STATEMENT_NOT_TEXT")
        if statement:
            digest = hashlib.sha256(statement.encode()).hexdigest()
            occurrence = row["turn"] if row.get("turn") is not None else "selected-response-" + str(row["selected_response_ordinal"])
            statements.setdefault(digest, {"text": statement, "actions": [], "turns": []})["turns"].append(occurrence)
    summaries = {}
    for row in omissions:
        summary = summaries.setdefault(row["field"], {"action_count": 0, "json_bytes": 0})
        summary["action_count"] += 1
        summary["json_bytes"] += row["json_bytes"]
    profile = cp.get("details", {}).get("model_profile", {})
    process = verified["process"]
    trajectory = {"run_id": verified["run_id"], "request_id": verified["request_id"],
        **({"checkpoint_sha256": verified["source"]["checkpoint_ref"]["sha256"]} if cp else
           {"source_episode_sha256": verified["source"]["episode_ref"]["sha256"]}),
        "completed_actions": len(verified["records"]),
        "pending_model_actions": sum(row.get("execution_status") != "not_dispatched" for row in verified["pending"]),
        "not_dispatched_actions": sum(row.get("execution_status") == "not_dispatched" for row in verified["pending"]),
        "model_responses_recorded": len(verified["responses"]),
        "model_calls_reserved": loop.get("model_budget", {}).get("consumed"),
        "counts": dict(Counter(r["call"]["name"] for r in verified["records"])),
        "native_terminal": False, "business_result": "not_established", "process_status": process["status"],
        "process_wall_seconds": process.get("wall_seconds"),
        "phase": loop["phase"], "pending_outcome": "unknown" if loop["phase"] in {"provider_pending", "tool_pending"} else None,
        "model": {k: profile[k] for k in ("model", "context_window", "output_budget") if k in profile},
        "action_columns": ["i", "tool", "revision", "args", "result"], "actions": actions,
        "statement_columns": ["verbatim_text", "action_occurrences", "response_turns"],
        "statements": [[row["text"], row["actions"], row["turns"]] for row in statements.values()],
        "empty_visible_statement_count": sum(not row["response"].get("content") for row in verified["responses"]),
        "paths": {v: k for k, v in paths.items()}, "reference_aliases": {v: k for k, v in refs.items()},
        "argument_defaults_omitted": {"root_id": "target"}, "omission_summary": summaries,
        "scope": "All public completed actions plus explicitly pending model actions. Statements are verbatim, not established hypotheses. "
            "Source text, long search matches and code bodies are omitted for process-strategy extraction; their locators remain in the omission manifest."}
    if "public_selected_facts" in verified:
        trajectory.update(public_selected_facts=verified["public_selected_facts"], source_coverage=verified["coverage"],
            statement_columns=["verbatim_text", "action_occurrences", "selected_response_occurrences"],
            actual_model_turns="not_selected_in_original_episode",
            empty_visible_statement_count=sum(row.get("content_selected") is True and not row["response"].get("content")
                for row in verified["responses"]),
            unselected_statement_count=sum(row.get("content_selected") is not True for row in verified["responses"]),
            visible_statements_coverage="Only statements selected by the original episode; unselected content is not inspected or inferred.")
    # 为什么要有它：原任务字段由episode的独立source_task_context选择器发送一次，过程轨迹不重复占用同一事实的上下文。
    return trajectory, omissions


def source_context(verified):
    cp = verified.get("checkpoint")
    if not cp:
        return verified["episode"]["task_contract_ref"], deepcopy(verified["episode"]["applicability"])
    baseline = read_public(file_ref(Path(verified["source"]["artifact_store_root"]) / cp["task_contract_ref"]))["baseline_manifest_ref"]
    has_candidate = cp["task_head"]["current_revision_ref"] != baseline
    return cp["task_contract_ref"], {"runtime": "dotnet", "target_domain": "csharp_repository",
        "has_candidate": has_candidate, "task_phase": "candidate" if has_candidate else "investigation"}


def prepare_trajectory(source, output_root):
    verified = verify_source(source)
    output = Path(output_root).absolute()
    _no_links(output)
    require(not output.exists(), "FRESH_OUTPUT_REQUIRED")
    protected = ([Path(source["allowed_source_root"]).resolve()] if source["kind"] == "public_episode" else
        [Path(source["artifact_store_root"]).resolve().parent, Path(source["platform_ref"]["path"]).resolve().parent,
         Path(source["process_ref"]["path"]).resolve().parent.parent])
    require(not any(output.resolve().is_relative_to(p) or p.is_relative_to(output.resolve()) for p in protected), "OUTPUT_OVERLAP")
    trajectory, omissions = project_verified(verified)
    output.mkdir(parents=True, exist_ok=False)
    cp = verified.get("checkpoint")
    origins = [{k: row["call"][k] for k in ("call_id", "name", "arguments_ref", "response_ref", "execution_receipt_ref")}
        for row in verified["records"]]
    omission_ref = write_once(output / "omissions.json", {"projection_version": VERSION, "source": source,
        "projection_owner_ref": file_ref(__file__), "omissions": omissions, "actions": origins,
        "responses": [{key: row[key] for key in ("turn", "selected_response_ordinal", "response_ref") if key in row}
            for row in verified["responses"]],
        "pending_actions": [deepcopy(row["call"]) for row in verified["pending"]]})
    trajectory["omission_manifest_ref"] = omission_ref
    data_ref = write_once(output / "trajectory.json", {"trajectory": trajectory})
    contract_ref, applicability = source_context(verified)
    provenance = [("trajectory", data_ref), ("projection", omission_ref)]
    if cp:
        provenance.extend((name, source[key]) for name, key in (("checkpoint", "checkpoint_ref"), ("process", "process_ref"),
            ("configuration", "execution_config_ref"), ("platform", "platform_ref")))
    evidence = [{"kind": "json_file", "evidence_id": name, **reference, "visibility": "public", "allowed_uses": ["strategy_learning"]}
        for name, reference in provenance]
    if source.get("episode_ref"):
        evidence.append({"kind": "json_file", "evidence_id": "original_episode", **source["episode_ref"],
            "visibility": "public", "allowed_uses": ["strategy_learning"]})
    task_refs, task_selectors, _task_documents = task_context_evidence(verified)
    evidence.extend(task_refs)
    episode = {"schema_version": "public_strategy_episode_v1", "visibility": "public", "data_use": "strategy_learning",
        "information_gain": ["public_process_observation"], "run_id": verified["run_id"], "request_id": verified["request_id"],
        "task_contract_ref": contract_ref, "applicability": applicability, "evidence_refs": evidence,
        "selections": [{"evidence_id": "trajectory", "locator": ["trajectory"]}],
        "task_context_selections": task_selectors,
        "projection_version_ref": {"version": VERSION, "owner_ref": file_ref(__file__)},
        "scope": "deterministic developer-prepared public process projection; no proposed strategy supplied"}
    return write_once(output / "episode.json", episode)
