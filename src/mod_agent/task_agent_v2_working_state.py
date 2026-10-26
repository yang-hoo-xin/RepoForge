"""Persistent observations and model statements for one task, without a critic call.

为什么要有它：当前运行须记得实际失败和模型自己的陈述；计数不能伪装为业务改进或替模型决定技术方向。
"""
from __future__ import annotations

from copy import deepcopy

from src.libs.llm.agent_turn import AgentTurnProviderError
from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_v2_context import native_assistant_text, native_tool_calls
from src.mod_agent.task_agent_v2_semantic_state import strip_working_state

_SCHEMA = "task-agent-working-state.v1"


def retain_model_statement(working, response, *, response_ref=None, source_boundary=None):
    """Retain complete public response content; no inferred hypothesis or hidden reasoning.

    为什么要有它：模型不调用可选计划工具时，原生响应中的技术判断也不能在下一次工具调用后丢失。
    """
    value = deepcopy(working or {})
    value.setdefault("schema", _SCHEMA)
    value.setdefault("observed_call_ids", [])
    value.setdefault("returned_evidence_fingerprints", [])
    value.setdefault("counts", {"completed_returns": 0, "distinct_returns": 0, "repeated_returns": 0,
        "accepted_candidate_returns": 0, "executed_experiment_returns": 0, "public_check_returns": 0})
    statements = value.setdefault("model_statements", [])
    content = response.content if hasattr(response, "content") else response.get("content")
    if isinstance(content, str) and content.strip():
        fingerprint = canonical_hash(content)
        statement = next((row for row in statements if row["content_sha256"] == fingerprint), None)
        first_seen = statement is None
        if first_seen:
            statement = {"content": content, "content_sha256": fingerprint,
                "first_response_ref": deepcopy(response_ref), "response_ref": deepcopy(response_ref),
                "source_boundary": deepcopy(source_boundary), "origin": "actual_model_response_content",
                "interpretation": "Model-authored statement, not an established fact or approval."}
            statements.append(statement)
        observed_refs = value.setdefault("model_statement_observed_response_refs", [])
        response_identity = canonical_hash(response_ref) if response_ref is not None else None
        # Scope: an actual new response is an occurrence; replayed native content alone is not.
        # 为什么要有它：正文去重不能删除后来重申的时间和来源，回放又不能伪造一次新响应。
        if first_seen or response_identity is not None and response_identity not in observed_refs:
            sequence = value.get("model_statement_sequence", max(
                (row.get("last_observed_sequence", 0) for row in statements), default=0)) + 1
            value["model_statement_sequence"] = sequence
            if statement.get("source_boundary") != source_boundary or "boundary_first_observed_sequence" not in statement:
                statement["boundary_first_observed_sequence"] = sequence
            statement.update(last_observed_sequence=sequence, response_ref=deepcopy(response_ref),
                source_boundary=deepcopy(source_boundary), last_occurrence_evidence=(
                    "actual_response_reference" if response_identity is not None else "first_seen_raw_content_without_response_reference"))
            if statement.get("first_response_ref") is None and response_ref is not None:
                statement["first_response_ref"] = deepcopy(response_ref)
            value["latest_model_statement_sha256"] = fingerprint
            if response_identity is not None and response_identity not in observed_refs:
                observed_refs.append(response_identity)
    return value


def update_working_state(previous, request, *, task_contract_ref, current_revision_ref,
                         registry_ref, current_check_ref):
    """Idempotently consume completed returns, preserving distinct and repeated evidence.

    为什么要有它：恢复和上下文投影会再次看到历史；重复回放不能增加计数，也不能抹掉连续无新返回的事实。
    """
    value = deepcopy(previous or {})
    value.setdefault("schema", _SCHEMA)
    if value["schema"] != _SCHEMA:
        raise ValueError("TASK_AGENT_WORKING_STATE_SCHEMA_INVALID")
    prior_boundary = value.get("boundary")
    boundary = {"task_contract_ref": task_contract_ref, "current_revision_ref": current_revision_ref,
        "binding_registry_ref": registry_ref, "current_check_ref": current_check_ref}
    value["boundary"] = boundary
    value.setdefault("model_statements", [])
    seen_calls = value.setdefault("observed_call_ids", [])
    fingerprints = value.setdefault("returned_evidence_fingerprints", [])
    counts = value.setdefault("counts", {"completed_returns": 0, "distinct_returns": 0,
        "repeated_returns": 0, "accepted_candidate_returns": 0, "executed_experiment_returns": 0,
        "public_check_returns": 0})
    streak = value.get("consecutive_repeated_returns", 0)
    if prior_boundary is not None and prior_boundary != boundary:
        streak = 0
    if prior_boundary is not None and prior_boundary.get("task_contract_ref") != task_contract_ref:
        value.pop("public_failure_repetition", None)
    exchanges = [(row.provider_state, row.tool_results) for row in request.historical_tool_exchanges]
    exchanges.append((request.provider_state, request.tool_results))
    for message in request.messages:
        if message.role == "assistant":
            value = retain_model_statement(value, {"content": message.content})
    # Keep references to the cumulative ledgers after the copy used by statement retention.
    seen_calls, fingerprints, counts = (value[key] for key in
        ("observed_call_ids", "returned_evidence_fingerprints", "counts"))
    for native, results in exchanges:
        content = native_assistant_text(native)
        if isinstance(content, str) and content.strip():
            value = retain_model_statement(value, {"content": content})
            seen_calls, fingerprints, counts = (value[key] for key in
                ("observed_call_ids", "returned_evidence_fingerprints", "counts"))
        try:
            calls = {call.call_id: call for call in native_tool_calls(native)}
        except (AgentTurnProviderError, ValueError):
            calls = {}
        for result in results:
            identity = [result.name, result.call_id]
            if identity in seen_calls:
                continue
            seen_calls.append(identity)
            call = calls.get(result.call_id)
            # 为什么要有它：同一读取的模型状态注释变化不等于工具产生新证据，原生调用哈希仍在执行台账保留。
            arguments = strip_working_state(call.arguments) if call is not None and call.name == result.name else None
            fingerprint = canonical_hash({"name": result.name, "arguments": arguments,
                "output": result.output})
            counts["completed_returns"] += 1
            repeated = fingerprint in fingerprints
            if repeated:
                counts["repeated_returns"] += 1
                streak += 1
            else:
                fingerprints.append(fingerprint)
                counts["distinct_returns"] += 1
                streak = 0
            output = result.output if isinstance(result.output, dict) else {}
            observed = output.get("value", output)
            observed = observed if isinstance(observed, dict) else {}
            executed = output.get("execution") == "executed"
            fact = {"call_id": result.call_id, "name": result.name, "arguments": arguments,
                "returned_output_sha256": canonical_hash(result.output), "repeated_exact_return": repeated}
            value["latest_action"] = fact
            if result.name == "candidate_check":
                counts["public_check_returns"] += 1
                value["latest_public_check"] = {**fact, "output": deepcopy(result.output)}
                if observed.get("status") not in {None, "passed"}:
                    value["latest_public_failure"] = {**fact, "output": deepcopy(result.output)}
                if executed:
                    signature = public_failure_signature(observed)
                    previous_failure = value.get("public_failure_repetition") or {}
                    same = signature is not None and previous_failure.get("signature") == signature["signature"]
                    count = previous_failure.get("consecutive_checks", 0) + 1 if same else int(signature is not None)
                    value["public_failure_repetition"] = {
                        **(signature or {"signature": None, "observable_projection": None}),
                        "consecutive_checks": count, "latest_check_ref": observed.get("check_ref"),
                        "latest_call_id": result.call_id, "same_visible_failure": count >= 3,
                        "business_progress_inferred": False,
                        "scope": "Executed checks only; receipt IDs are not failure identity. "
                            "This compares selected check IDs, expected/actual observations and counts; complete public details may differ."}
            if executed and result.name in {"candidate_submit", "candidate_edit"} and observed.get("candidate_ref"):
                counts["accepted_candidate_returns"] += 1
            if executed and result.name in {"experiment_run", "experiment_execute", "probe_run", "test_run"}:
                counts["executed_experiment_returns"] += 1
    value["consecutive_repeated_returns"] = streak
    repeated_failure = (value.get("public_failure_repetition") or {}).get("same_visible_failure", False)
    value["stagnation"] = {"triggered": streak >= 3 or repeated_failure, "consecutive_repeated_returns": streak,
        "same_visible_public_failure": repeated_failure,
        "basis": "The exact-return counter compares matched tool arguments and full returned output; "
            "the public-failure signal separately compares the declared selected observation signature. Replayed call IDs count once.",
        "business_progress_inferred": False, "authority_changed": False,
        "guidance": ("Repeated returns have added no distinct recorded observation. Reconsider the current "
            "hypothesis using the recorded evidence; choose a discriminating experiment, a justified edit, "
            "a specific missing observation, or explain a concrete blocker. No patch or plan tool is mandatory.")
            if streak >= 3 else (
                "Three or more executed public checks returned the same selected check/expected/actual/count signature. "
                "Inspect the full public feedback before revising a hypothesis or selecting a discriminating experiment. "
                "This does not establish absence of technical progress; no edit, approval or plan tool is required.")
            if repeated_failure else None}
    return value


def working_state_view(working, *, statement_archive_ref=None):
    """Project current work without resending the durable narrative and audit ledgers.

    为什么要有它：后台保留完整原文和恢复台账；模型需要当前明确陈述及反馈引用，旧叙述不能重复挤占源码与实验结果。
    """
    keys = ("schema", "boundary", "counts", "consecutive_repeated_returns", "latest_model_statement_sha256")
    view = {key: deepcopy(working[key]) for key in keys if key in working}
    statements = deepcopy(working.get("model_statements", []))
    for index, statement in enumerate(statements):
        statement["statement_index"] = index
    latest = max(statements, key=lambda row: row.get("last_observed_sequence", row["statement_index"]), default=None)
    # No original is omitted without the normal caller's public archive.
    selected = [latest] if latest is not None and statement_archive_ref is not None else statements
    view["model_statements"] = selected
    for key in ("latest_action", "latest_public_check", "latest_public_failure"):
        row = working.get(key)
        if not isinstance(row, dict):
            continue
        item = {name: deepcopy(row[name]) for name in ("call_id", "name", "returned_output_sha256",
            "repeated_exact_return") if name in row}
        output = row.get("output")
        if isinstance(output, dict):
            observed = output.get("value", output)
            if isinstance(observed, dict):
                item["receipt"] = {name: deepcopy(observed[name]) for name in
                    ("status", "candidate_ref", "check_ref", "feedback_ref", "source_revision_ref") if name in observed}
        view[key] = item
    if "stagnation" in working:
        view["stagnation"] = {key: deepcopy(working["stagnation"][key]) for key in
            ("triggered", "consecutive_repeated_returns", "same_visible_public_failure", "business_progress_inferred", "guidance")
            if key in working["stagnation"]}
    if "public_failure_repetition" in working:
        view["public_failure_repetition"] = {key: deepcopy(working["public_failure_repetition"][key]) for key in
            ("signature", "consecutive_checks", "latest_check_ref", "latest_call_id", "same_visible_failure", "business_progress_inferred")
            if key in working["public_failure_repetition"]}
    view["model_statement_projection"] = {"archive_ref": statement_archive_ref,
        "total_statements": len(statements), "retained_statement_indices": [row["statement_index"] for row in selected],
        "omitted_statement_count": len(statements) - len(selected), "semantic_summary_available": False,
        "current_boundary_statement_available": any(row.get("source_boundary") == working.get("boundary") for row in selected),
        "selection_basis": "Latest actual nonempty model statement; earlier statements remain in the public archive."
            if statement_archive_ref is not None else "All original statements retained because no archive was supplied.",
        "meaning": "Raw model wording, not inferred hypothesis status. Omission does not mean a claim was resolved."}
    return view


def validate_working_state(working, *, responses, call_records):
    """Reject invented public statements or returned observations during recovery.

    为什么要有它：恢复状态会进入真实模型请求，持久化不能成为注入未发生的模型陈述和工具结果的通道。
    """
    if working is None:
        return
    if working.get("schema") != _SCHEMA:
        raise ValueError("TASK_AGENT_WORKING_STATE_SCHEMA_INVALID")
    contents = {canonical_hash(row.get("content")) for row in responses
        if isinstance(row.get("content"), str) and row["content"].strip()}
    for statement in working.get("model_statements", []):
        if (canonical_hash(statement.get("content")) != statement.get("content_sha256")
                or statement["content_sha256"] not in contents):
            raise ValueError("TASK_AGENT_WORKING_STATEMENT_NOT_OBSERVED")
    identities = working.get("observed_call_ids", [])
    if len({tuple(row) for row in identities}) != len(identities):
        raise ValueError("TASK_AGENT_WORKING_RETURN_REPLAY_COUNTED")
    for name, identifier in identities:
        record = call_records.get(identifier, {})
        result = record.get("result") or {}
        if result.get("name") != name:
            raise ValueError("TASK_AGENT_WORKING_RETURN_NOT_OBSERVED")
    counts = working.get("counts", {})
    if counts.get("completed_returns") != len(identities):
        raise ValueError("TASK_AGENT_WORKING_RETURN_COUNT_CHANGED")
    for key in ("latest_public_check", "latest_public_failure"):
        observed = working.get(key)
        if observed is not None:
            result = (call_records.get(observed["call_id"], {}).get("result") or {})
            if result.get("name") != observed["name"] or result.get("output") != observed["output"]:
                raise ValueError("TASK_AGENT_WORKING_FEEDBACK_NOT_OBSERVED")


def public_failure_signature(observed):
    """Compare only explicit public counterexamples, retaining receipts separately.

    为什么要有它：新检查的回执和临时路径不等于新业务结果；相同可见失败也不能被声称为完整因果或没有技术进展。
    """
    rows = observed.get("observations")
    if observed.get("status") != "behavior_failed" or not isinstance(rows, list) or not rows:
        return None
    public = []
    for row in rows:
        if (not isinstance(row, dict) or "check_id" not in row or "expected" not in row
                or not isinstance(row.get("observed"), dict) or "actual" not in row["observed"]):
            return None
        public.append({"check_id": deepcopy(row["check_id"]), "test_name": deepcopy(row.get("test_name")),
            "expected": deepcopy(row["expected"]), "actual": deepcopy(row["observed"]["actual"]),
            "passed": row["observed"].get("passed")})
    public.sort(key=canonical_hash)
    projection = {"status": observed["status"], "observations": public,
        "observation_count": observed.get("observation_count", len(rows)),
        "observations_truncated": observed.get("observations_truncated", False),
        "test_counts": deepcopy((observed.get("test_execution_summary") or {}).get("counts"))}
    return {"signature": canonical_hash(projection), "observable_projection": projection,
        "signature_scope": "Selected public check identity, expected/actual and counts only; not full feedback equivalence.",
        "excluded_fields": ["candidate_ref", "candidate_source_id", "check_ref", "feedback_ref",
            "raw_report_ref", "details", "source_locations", "verification_summary"]}
