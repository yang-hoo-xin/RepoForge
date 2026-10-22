"""Explicit v2 product admission without reconstructing a technical Plan."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_execution import _admitted_response, _read, _require, _worker_binding


@dataclass(frozen=True)
class TaskAgentPatchOutcomeV2:
    status: str
    termination: dict[str, Any]
    initial_context: dict[str, Any]
    final_context: dict[str, Any]
    task_contract_ref: str | None
    binding_registry_ref: str | None
    candidate_ref: str | None
    check_ref: str | None
    execution_ref: dict[str, str] | None
    loop_result_ref: dict[str, str] | None
    finish_response_ref: dict[str, str] | None
    budget_before: dict[str, Any]
    budget_after: dict[str, Any]
    schema_version: str = "task-agent-product-execution.v2"
    unknown_effects: bool = False
    whole_task_delivered: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ValidatedTaskAgentHandoffV2:
    task_contract_ref: str
    binding_registry_ref: str
    candidate_ref: str
    check_ref: str
    execution_ref: dict[str, str]
    candidate_source_id: str
    changes: tuple[dict[str, Any], ...]

    def to_dict(self):
        return asdict(self)


def validate_task_agent_handoff_v2(outcome, *, store, workspace_root, expected_initial_context,
        allowed_edit_paths, worker_configuration_ref, expected_verification_identity, **_unused):
    """Revalidate immutable provenance and executed calls at the product boundary.

    为什么要有它：v2没有旧Plan；真正的模型提交、当前候选和执行过的公开检查必须直接成为产品接收依据。
    """
    from src.mod_agent.model_call_budget import validate_budget
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State

    _require(isinstance(outcome, TaskAgentPatchOutcomeV2) and outcome.status == "candidate_ready"
        and not outcome.unknown_effects and not outcome.whole_task_delivered, "V2_HANDOFF_NOT_READY")
    execution = _read(outcome.execution_ref, store.root)
    _require(all(execution.get(k) == v for k, v in outcome.to_dict().items() if k != "execution_ref"),
        "V2_OUTCOME_CHANGED")
    _require(outcome.initial_context == outcome.final_context == expected_initial_context
        and set(expected_initial_context) == {"feature_spec_ref"}, "V2_INITIAL_CONTEXT_CHANGED")
    _require(execution.get("allowed_edit_paths") == list(allowed_edit_paths)
        and execution.get("worker_binding") == _worker_binding(worker_configuration_ref)
        and execution.get("expected_verification_identity") == expected_verification_identity,
        "V2_AUTHORITY_CHANGED")
    before, after = validate_budget(outcome.budget_before), validate_budget(outcome.budget_after)
    _require(before["maximum"] == after["maximum"] and before.get("policy") == after.get("policy")
        and after["consumed"] >= before["consumed"] and all(after["by_stage"][k]["consumed"] >= v["consumed"]
            for k, v in before["by_stage"].items()), "V2_BUDGET_CHANGED")
    state = TaskAgentV2State(store.root, outcome.task_contract_ref)
    _require(state.contract.get("feature_spec_ref") == expected_initial_context["feature_spec_ref"],
        "V2_REQUIREMENTS_CHANGED")
    candidate = state.validate_candidate(outcome.candidate_ref)
    registry = state.validate_registry(outcome.binding_registry_ref)
    from pathlib import Path
    baseline = state.load(state.contract["baseline_manifest_ref"])
    target_root = Path(baseline["source_root"]).resolve()
    expected_paths = {str((Path(workspace_root) / path).resolve()) for path in allowed_edit_paths}
    _require(registry.get("contract_ref") == outcome.task_contract_ref
        and {str((target_root / path).resolve()) for path in state.contract["allowed_edit_paths"]} == expected_paths,
        "V2_REGISTRY_AUTHORITY_CHANGED")
    loop = _read(outcome.loop_result_ref, store.root)
    _require(loop.get("status") == "completed" and loop.get("unknown_effects") is False
        and loop.get("model_budget") == after and loop.get("termination") == outcome.termination
        and outcome.termination == {"code": "TASK_AGENT_COMPLETION_ACCEPTED"}
        and type(loop.get("model_turns")) is int
        and 0 < loop["model_turns"] <= after["consumed"] - before["consumed"],
        "V2_REAL_FINISH_REQUIRED")
    records = execution.get("completed_tool_records", [])
    _require([row.get("event") for row in records] == [event for event in loop.get("events", [])
        if event.get("event") == "tool_dispatch_completed"], "V2_TOOL_SEQUENCE_CHANGED")
    seen = set()
    outputs = []
    for row in records:
        arguments = _read(row["arguments_ref"], store.root)
        response = _read(row["response_ref"], store.root)
        output = _read(row["result_ref"], store.root)
        _admitted_response(response, execution["model_profile"])
        name, call_id = row["name"], row["call_id"]
        _require(call_id not in seen, "V2_DUPLICATE_CALL")
        seen.add(call_id)
        _require([call for call in response.get("tool_calls", []) if call.get("call_id") == call_id]
            == [{"name": name, "call_id": call_id, "arguments": arguments}], "V2_TOOL_RESPONSE_CHANGED")
        actual = loop.get("call_records", {}).get(call_id, {})
        _require(actual.get("status") == "completed" and actual.get("name") == name
            and actual.get("arguments_sha256") == canonical_hash(arguments)
            and actual.get("result", {}).get("output") == output
            and row.get("event") in loop.get("events", [])
            and row["event"].get("output_sha256") == canonical_hash(output)
            and output.get("execution") == "executed", "V2_TOOL_EXECUTION_CHANGED")
        outputs.append((name, arguments, output.get("value", {}), row))
    historical_count = 0
    if state.contract.get("rebase_origin_ref") is not None:
        # 为什么要有它：重基线保留原技术会话，但旧contract的提交不能冒充新baseline的候选父链。
        from src.mod_agent.task_agent_v2_delivery import validate_v2_delivery_rebase
        origin = state.load(state.contract["rebase_origin_ref"])
        previous = TaskAgentV2State(store.root, origin["prior_task_contract_ref"])
        transition = previous.head["operations"]["delivery-rebase:" + canonical_hash(origin["receipt_ref"])]
        rebase = validate_v2_delivery_rebase(artifact_store_root=store.root, rebase_ref=transition["result"]["rebase_ref"])
        _require(rebase["new_task_contract_ref"] == state.contract_ref, "V2_REBASE_CONTRACT_CHANGED")
        historical = _read(origin["prior_execution_ref"], store.root)
        historical_records = historical["completed_tool_records"]
        historical_count = len(historical_records)
        _require(historical_count < len(records) and historical["initial_context"] == outcome.initial_context
            and historical["model_profile"] == execution["model_profile"]
            and historical["budget_before"] == outcome.budget_before
            and outcome.budget_after["consumed"] >= rebase["resource_record"]["model_budget"]["consumed"],
            "V2_REBASE_HISTORY_CHANGED")
        for original, retained in zip(historical_records, records[:historical_count]):
            _require({k: v for k, v in original.items() if k != "result_ref"}
                == {k: v for k, v in retained.items() if k != "result_ref"}
                and _read(original["result_ref"], store.root) == _read(retained["result_ref"], store.root),
                "V2_REBASE_HISTORY_CHANGED")
    active_outputs = outputs[historical_count:]
    submits = [(a, v, r) for n, a, v, r in active_outputs if n == "candidate_submit"]
    _require(bool(submits) and submits[-1][1].get("candidate_ref") == outcome.candidate_ref,
        "V2_CURRENT_CANDIDATE_CHANGED")
    selected = outcome.candidate_ref
    for arguments, value, row in reversed(submits):
        _require(value.get("candidate_ref") == selected, "V2_PARENT_SEQUENCE_CHANGED")
        document = state.validate_candidate(selected)
        _require(document.get("parent_revision_ref") == arguments.get("parent_revision_ref"),
            "V2_PARENT_CHANGED")
        keys = ("binding_id", "source_view_ref", "before_text", "after_text")
        _require([{k: edit.get(k) for k in keys} for edit in arguments.get("edits", [])]
            == [{k: edit.get(k) for k in keys} for edit in document["changes"]]
            and document.get("model_origin") == {k: row[k] for k in ("call_id", "response_ref", "arguments_ref")},
            "V2_SUBMITTED_BYTES_CHANGED")
        selected = document["parent_revision_ref"]
    _require(selected == state.contract["baseline_manifest_ref"], "V2_BASELINE_PARENT_CHANGED")
    _require(outputs and outputs[-1][0] == "finish" and outputs[-1][3]["response_ref"] == outcome.finish_response_ref
        and outputs[-1][1].get("candidate_ref") == outcome.candidate_ref
        and outputs[-1][1].get("check_ref") == outcome.check_ref, "V2_FINISH_CHANGED")
    checks = [(a, v) for n, a, v, _r in outputs if n == "candidate_check"]
    _require(any(a.get("candidate_ref") == outcome.candidate_ref and v.get("check_ref") == outcome.check_ref
        for a, v in checks), "V2_CHECK_EXECUTION_REQUIRED")
    check = (state.load(outcome.check_ref) if isinstance(outcome.check_ref, str)
        else _read(outcome.check_ref, store.root))
    report = check.get("report", check)
    _require(report.get("status") == "passed" and check.get("candidate_ref") == outcome.candidate_ref
        and check.get("candidate_source_id") == candidate["candidate_source_id"], "V2_CHECK_CHANGED")
    from src.mod_agent.domain.csharp_repository_backend import validate_candidate_report_v2
    check_context = _read(check["request_ref"], store.root)
    checked_registry = state.validate_registry(check_context["binding_registry_ref"])
    _require(check_context["task_contract_ref"] == outcome.task_contract_ref
        and all(registry["bindings"].get(k) == v for k, v in checked_registry["bindings"].items())
        and check_context["candidate_ref"] == outcome.candidate_ref
        and check_context["candidate_source_id"] == candidate["candidate_source_id"]
        and check_context["verification_identity"] == expected_verification_identity
        and check_context["model_response"] == {"path": submits[-1][2]["response_ref"]["path"],
            "output_sha256": submits[-1][2]["response_ref"]["sha256"]}, "V2_CHECK_REQUEST_CHANGED")
    validate_candidate_report_v2(report, context=check_context, require_pass=True)
    final_files = state.materialize(outcome.candidate_ref)
    bindings = {row["canonical_path"]: row for row in registry["bindings"].values()}
    changes = []
    for path, text in sorted(final_files.items()):
        original = baseline["files"][path]["text"]
        if text == original:
            continue
        _require(path in bindings, "V2_FINAL_BINDING_MISSING")
        binding = bindings[path]
        _require(binding["scope"] == "file" and binding["operation"] == "modify_existing"
            and binding["authority_ref"] == state.contract["authority_ref"]
            and binding["baseline_file_sha256"] == baseline["files"][path]["sha256"], "V2_FINAL_BINDING_CHANGED")
        changes.append({"change_id": "v2-" + canonical_hash(path)[:16], "change_type": "modify",
            "edit_binding_id": binding["binding_id"], "display_file": path, "operation": "replace",
            "rationale": "Validated cumulative candidate modification.", "evidence_refs": [],
            "expected_validation": ["bound public candidate check"], "rollback_hint": "Restore baseline bytes.",
            "patch": {"before": original, "after": text}, "baseline_file_sha256": baseline["files"][path]["sha256"]})
    _require(bool(changes), "V2_NO_CHANGE")
    return ValidatedTaskAgentHandoffV2(outcome.task_contract_ref, outcome.binding_registry_ref,
        outcome.candidate_ref, outcome.check_ref, outcome.execution_ref, candidate["candidate_source_id"],
        tuple(changes))


def revalidate_handoff_v2(value, *, store):
    """Load a retained outcome and repeat admission; serialized flags confer no trust.

    为什么要有它：Proposal、Review与Dry-run各自重新绑定原件，不能依赖先前的成功字符串。
    """
    execution = _read(value["execution_ref"], store.root)
    fields = TaskAgentPatchOutcomeV2.__dataclass_fields__
    outcome = TaskAgentPatchOutcomeV2(**{key: value["execution_ref"] if key == "execution_ref"
        else deepcopy(execution[key]) for key in fields})
    selected = validate_task_agent_handoff_v2(outcome, store=store, workspace_root=execution["workspace_root"],
        expected_initial_context=execution["initial_context"], allowed_edit_paths=execution["allowed_edit_paths"],
        worker_configuration_ref=execution["worker_binding"]["configuration_ref"],
        expected_verification_identity=execution["expected_verification_identity"])
    _require(canonical_hash(selected.to_dict()) == canonical_hash(value), "V2_HANDOFF_CHANGED")
    return selected
