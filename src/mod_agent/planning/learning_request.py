"""Public learning questions, source-task facts and source-bound model assessments.

为什么要有它：本次学习目的、原任务事实与建议的适用范围不同；任何解释都不能被写成新增事实或执行授权。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path, PurePosixPath

from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.planning.task_memory import _select
from src.mod_agent.task_agent_v2_state import _no_links

TASK_FIELDS = ("original_request", "acceptance_criteria", "non_goals")


def require(value, code):
    if not value:
        raise ValueError("LEARNING_CONTRACT_" + code)


def read_learning_request(reference):
    if reference is None:
        return None
    require(isinstance(reference, dict) and set(reference) == {"path", "sha256"}, "REFERENCE_INVALID")
    path = Path(reference["path"])
    require(path.is_absolute() and not {"hidden", "evaluation-only"}.intersection(p.casefold() for p in path.parts),
        "PUBLIC_REQUEST_REQUIRED")
    _no_links(path)
    value = read_bound_json(reference)
    require(set(value) == {"schema_version", "visibility", "data_use", "question", "origin"}
        and value["schema_version"] == "strategy_learning_request_v1"
        and value["visibility"] == "public" and value["data_use"] == "strategy_learning"
        and isinstance(value["question"], str) and bool(value["question"].strip()), "REQUEST_INVALID")
    origin = value["origin"]
    require(isinstance(origin, dict) and set(origin) == {"kind", "description"}
        and origin["kind"] in {"caller", "user", "outer_technical_assistance"}
        and isinstance(origin["description"], str) and bool(origin["description"].strip()), "ORIGIN_INVALID")
    return {**deepcopy(value), "reference": deepcopy(reference), "authority": "advisory_only"}


def task_context_selections(contract, evidence_id):
    requirements = contract.get("task_requirements")
    if not isinstance(requirements, dict):
        return []
    return [{"field": field, "evidence_id": evidence_id, "locator": ["task_requirements", field]}
        for field in TASK_FIELDS if field in requirements]


def resolve_task_context(episode, documents):
    """Read only explicitly selected original contract fields, never follow identity metadata.

    为什么要有它：旧公开经历没有选择的任务目标必须保持未知，不能为补字段扩大读取检查点或隐藏材料。
    """
    selectors = episode.get("task_context_selections", [])
    require(isinstance(selectors, list), "TASK_CONTEXT_SELECTORS_INVALID")
    fields, sources = {}, {}
    references = {row["evidence_id"]: row for row in episode.get("evidence_refs", [])}
    contract_ref = episode.get("task_contract_ref")
    if selectors:
        require(isinstance(contract_ref, str) and contract_ref.startswith("v2/contract/")
            and len(PurePosixPath(contract_ref).stem) == 64
            and all(ch in "0123456789abcdef" for ch in PurePosixPath(contract_ref).stem)
            and PurePosixPath(contract_ref).as_posix() == "v2/contract/" + PurePosixPath(contract_ref).stem + ".json",
            "TASK_CONTRACT_IDENTITY_REQUIRED")
    for selection in selectors:
        require(isinstance(selection, dict) and set(selection) == {"field", "evidence_id", "locator"}
            and selection["field"] in TASK_FIELDS and selection["field"] not in fields
            and selection["locator"] == ["task_requirements", selection["field"]]
            and selection["evidence_id"] in documents, "TASK_CONTEXT_SELECTION_INVALID")
        document = documents[selection["evidence_id"]]
        reference = references.get(selection["evidence_id"], {})
        require(reference.get("kind") == "json_file" and reference.get("sha256") == PurePosixPath(contract_ref).stem
            and isinstance(reference.get("path"), str)
            and Path(reference["path"]).as_posix().endswith("/" + contract_ref), "TASK_CONTRACT_REFERENCE_CHANGED")
        require(isinstance(document, dict) and document.get("schema") == "task-agent-contract.v2", "TASK_CONTRACT_REQUIRED")
        require(episode.get("request_id") is None or document.get("task_id") == episode["request_id"],
            "TASK_CONTRACT_REQUEST_CHANGED")
        value = _select(document, selection["locator"])
        require(isinstance(value, str) if selection["field"] == "original_request" else
            isinstance(value, list) and all(isinstance(row, str) for row in value), "TASK_REQUIREMENT_TYPE_INVALID")
        fields[selection["field"]] = value
        sources[selection["field"]] = {key: deepcopy(selection[key]) for key in ("evidence_id", "locator")}
    return {"availability": "selected_public_fields" if fields else "not_selected", "fields": fields,
        "field_sources": sources, "unknown_fields": [field for field in TASK_FIELDS if field not in fields],
        "meaning": "Original task facts, not the current learning question, proposed strategy scope or new authority."}


def selected_evidence_scope(episode, documents):
    selectors = list(episode["selections"]) + list(episode.get("task_context_selections", []))
    return [{"evidence_id": row["evidence_id"], "locator": deepcopy(row["locator"]),
        "value": _select(documents[row["evidence_id"]], row["locator"])} for row in selectors]


def feedback_schema(episode_ids):
    citation = {"type": "object", "additionalProperties": False, "properties": {
        "evidence_id": {"type": "string"}, "locator": {"type": "array", "items": {"type": ["string", "integer"]}}},
        "required": ["evidence_id", "locator"]}
    evidence = {"type": "array", "items": citation}
    assessment = {"type": "object", "additionalProperties": False, "properties": {
        "episode_id": {"type": "string", "enum": sorted(episode_ids)},
        "relation": {"type": "string", "enum": ["supporting", "opposing", "inapplicable", "insufficient"]},
        "reason": {"type": "string"}, "evidence": evidence}, "required": ["episode_id", "relation", "reason"]}
    gap = {"type": "object", "additionalProperties": False, "properties": {
        "episode_id": {"type": ["string", "null"]}, "status": {"type": "string", "enum": ["unknown", "not_observed"]},
        "description": {"type": "string"}, "evidence": evidence}, "required": ["status", "description"]}
    return {"source_assessments": {"type": "array", "items": assessment},
        "evidence_gaps": {"type": "array", "items": gap}}


def validate_feedback(assessments, gaps, scope):
    """Validate references inside the actual sent selection, without reading omitted bodies.

    为什么要有它：模型可以描述概念缺口而不编引用；一旦引用，必须属于对应经历且确实已经发送。
    """
    require(isinstance(assessments, list) and isinstance(gaps, list), "FEEDBACK_LIST_REQUIRED")

    def citations(episode_id, references):
        require(isinstance(references, list), "CITATIONS_INVALID")
        seen = set()
        for reference in references:
            require(isinstance(reference, dict) and set(reference) == {"evidence_id", "locator"}
                and episode_id in scope and isinstance(reference["evidence_id"], str)
                and isinstance(reference["locator"], list)
                and all(isinstance(part, str) or type(part) is int and part >= 0 for part in reference["locator"]),
                "CITATION_INVALID")
            key = canonical_hash(reference)
            require(key not in seen, "DUPLICATE_CITATION")
            seen.add(key)
            matches = [row for row in scope[episode_id] if row["evidence_id"] == reference["evidence_id"]
                and reference["locator"][:len(row["locator"])] == row["locator"]]
            require(bool(matches), "CITATION_NOT_SENT")
            valid = False
            for row in matches:
                try:
                    _select(row["value"], reference["locator"][len(row["locator"]):])
                    valid = True
                    break
                except ValueError:
                    continue
            require(valid, "CITATION_LOCATOR_UNAVAILABLE")

    seen = set()
    for row in assessments:
        require(isinstance(row, dict) and {"episode_id", "relation", "reason"} <= set(row)
            and not set(row) - {"episode_id", "relation", "reason", "evidence"}
            and row["episode_id"] in scope and row["episode_id"] not in seen
            and row["relation"] in {"supporting", "opposing", "inapplicable", "insufficient"}
            and isinstance(row["reason"], str) and bool(row["reason"].strip()), "ASSESSMENT_INVALID")
        seen.add(row["episode_id"])
        citations(row["episode_id"], row.get("evidence", []))
    seen_gaps = set()
    for row in gaps:
        require(isinstance(row, dict) and {"status", "description"} <= set(row)
            and not set(row) - {"episode_id", "status", "description", "evidence"}
            and row.get("episode_id") in {None, *scope}
            and row["status"] in {"unknown", "not_observed"}
            and isinstance(row["description"], str) and bool(row["description"].strip()), "GAP_INVALID")
        require(canonical_hash(row) not in seen_gaps, "DUPLICATE_GAP")
        seen_gaps.add(canonical_hash(row))
        citations(row.get("episode_id"), row.get("evidence", []))
    return {"source_assessments": deepcopy(assessments), "evidence_gaps": deepcopy(gaps),
        "assessment_kind": "model_interpretation_not_verified_fact",
        "unassessed_episode_ids": sorted(set(scope) - seen), "gap_completeness": "model_reported_not_proven_complete"}
