"""Explicit experimental source priority; default planning/prompt behavior is unchanged."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy

from src.mod_agent.evidence.identity import feature_spec_payload_sha256
from src.mod_agent.evidence.requirement_coverage import validated_requirement_coverage_view

SOURCE_PRIORITY = "source_priority_v1"
MODEL_VIEW_VERSION = "source_priority_model_view_v2"
CONSTRAINT_INSTRUCTION = (
    "Source-priority experiment: the complete original user request, declared feature requirements, "
    "hard constraints and unchanged acceptance criteria remain binding. Model-derived preservation wording "
    "is omitted from this model view and retained with its source attribution in host records; it is not "
    "an additional user command. Omitted requirement IDs and source links are tracking metadata, not instructions "
    "to preserve existing behavior. Determine required behavior from the original user contract and current source. "
    "Do not discard genuine preservation requirements or infer new edit/execution permission. "
    "If actual user requirements conflict, do not silently override either. "
    "为什么要有它：保留用户公开接口不等于保留错误内部行为，派生语句不应反复作为额外命令进入模型输入。"
)
_USER_FIELDS = ("user_request", "target_behavior", "non_goals", "acceptance_criteria", "runtime_verification_plan")
_FIELDS = {"policy", "feature_spec_sha256", "user_contract", "source_plan_boundaries",
           "retained_hard_boundaries", "coverage_projection_sha256", "hard_boundaries", "boundary_origins", "model_preservation_hints", "context_sha256"}


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_constraint_policy(value):
    if value is not None and value != SOURCE_PRIORITY:
        raise ValueError("CONSTRAINT_POLICY_INVALID")
    return value


def _strings(value):
    if not isinstance(value, (list, tuple)) or any(type(item) is not str or not item.strip() for item in value):
        raise ValueError("CONSTRAINT_CONTEXT_TEXT_INVALID")
    return list(value)


def _user_contract(feature):
    if not isinstance(feature, Mapping):
        raise ValueError("CONSTRAINT_CONTEXT_REQUEST_INVALID")
    result = {key: deepcopy(feature.get(key)) for key in _USER_FIELDS}
    for key in _USER_FIELDS[:2]:
        if type(result[key]) is not str or not result[key].strip():
            raise ValueError("CONSTRAINT_CONTEXT_REQUEST_INVALID")
    for key in _USER_FIELDS[2:]:
        result[key] = _strings(result[key])
    return result


def _boundaries(user, source_boundaries, hints, retained_hard_boundaries):
    from src.mod_agent.planning.implementation_plan import PLATFORM_SIDE_EFFECT_BOUNDARIES

    hard, origins = [], []
    derived = {hint["text"] for hint in hints}
    # Scope: classify exact source contributions, never English keywords, class names or similarity.
    # 为什么要有它：相同文本可同时来自用户声明和模型；平台、non_goals或未明来源必须保留，不能用字符串差集删除真实要求。
    for text in dict.fromkeys([*PLATFORM_SIDE_EFFECT_BOUNDARIES, *user["non_goals"], *source_boundaries, *retained_hard_boundaries]):
        sources = []
        if text in PLATFORM_SIDE_EFFECT_BOUNDARIES:
            sources.append("platform")
        if text in user["non_goals"]:
            sources.append("declared_feature_non_goal")
        if text in retained_hard_boundaries:
            sources.append("caller_retained_hard_boundary")
        if text not in derived and text in source_boundaries and not sources:
            sources.append("unclassified_retained_boundary")
        if sources:
            hard.append(text)
            origins.append({"text": text, "sources": sources})
    return hard, origins


def build_constraint_context(*, feature_spec, evidence_pack=None, plan_boundaries=None, retained_hard_boundaries=()):
    """Build one source-bound view, shared by A* and B; no original artifact is mutated."""
    user = _user_contract(feature_spec)
    feature_hash = feature_spec_payload_sha256(feature_spec)
    hints, projection = [], None
    if evidence_pack is not None:
        coverage = validated_requirement_coverage_view(evidence_pack)
        if coverage is None or coverage.feature_id != feature_spec.get("feature_id") or coverage.feature_spec_sha256 != feature_hash:
            raise ValueError("CONSTRAINT_CONTEXT_COVERAGE_IDENTITY_MISMATCH")
        section = evidence_pack["requirement_coverage"]
        projection = section["projection_identity"]["projection_sha256"]
        inputs = {row["input_obligation_id"]: row for row in section["input_obligations"]}
        for row in section["requirements"]:
            if row["role"] != "preservation":
                continue
            linked = [inputs[identity] for identity in row["input_obligation_ids"]]
            hints.append({"requirement_id": row["requirement_id"], "text": row["text"],
                "input_obligation_ids": list(row["input_obligation_ids"]),
                "input_sources": [{"input_obligation_id": item["input_obligation_id"],
                    "origin": deepcopy(item["origin"]), "text_sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest()} for item in linked],
                "authority": "derived_interpretation_not_user_command"})
    if plan_boundaries is None:
        from src.mod_agent.planning.implementation_plan import PLATFORM_SIDE_EFFECT_BOUNDARIES

        plan_boundaries = list(dict.fromkeys([*PLATFORM_SIDE_EFFECT_BOUNDARIES,
            *user["non_goals"], *(row["text"] for row in hints)]))
    boundaries = _strings(plan_boundaries)
    # Scope: only the trusted caller supplies prior/otherwise-unclassified hard boundaries.
    # 为什么要有它：恢复时同文的明确硬边界不能因同时命中当前模型推断而被降级；不从模型响应接收此来源标签。
    retained = _strings(retained_hard_boundaries)
    hard, origins = _boundaries(user, boundaries, hints, retained)
    value = {"policy": SOURCE_PRIORITY, "feature_spec_sha256": feature_hash,
        "user_contract": user, "source_plan_boundaries": boundaries,
        "retained_hard_boundaries": retained,
        "coverage_projection_sha256": projection, "hard_boundaries": hard,
        "boundary_origins": origins, "model_preservation_hints": hints}
    return {**value, "context_sha256": _sha(value)}


def validate_constraint_context(value, *, feature_spec=None, evidence_pack=None, constraints=None, retained_hard_boundaries=None,
        user_request=None, target_behavior=None, acceptance_criteria=None, runtime_verification_plan=None):
    """Validate retained context against real caller inputs; never derive authority from model output."""
    if not isinstance(value, Mapping) or set(value) != _FIELDS or value["policy"] != SOURCE_PRIORITY:
        raise ValueError("CONSTRAINT_CONTEXT_INVALID")
    context = deepcopy(dict(value))
    if context["context_sha256"] != _sha({key: item for key, item in context.items() if key != "context_sha256"}):
        raise ValueError("CONSTRAINT_CONTEXT_HASH_MISMATCH")
    if (not isinstance(context["user_contract"], Mapping) or set(context["user_contract"]) != set(_USER_FIELDS)
            or not isinstance(context["feature_spec_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", context["feature_spec_sha256"]) is None):
        raise ValueError("CONSTRAINT_CONTEXT_INVALID")
    user = _user_contract(context["user_contract"])
    hints = context["model_preservation_hints"]
    if not isinstance(hints, list):
        raise ValueError("CONSTRAINT_CONTEXT_HINTS_INVALID")
    projection = context["coverage_projection_sha256"]
    if (projection is not None and (not isinstance(projection, str) or re.fullmatch(r"[0-9a-f]{64}", projection) is None)
            or hints and projection is None):
        raise ValueError("CONSTRAINT_CONTEXT_PROVENANCE_INVALID")
    seen = set()
    for row in hints:
        if (not isinstance(row, dict) or set(row) != {"requirement_id", "text", "input_obligation_ids", "input_sources", "authority"}
                or row["authority"] != "derived_interpretation_not_user_command"
                or type(row["requirement_id"]) is not str or not row["requirement_id"]
                or type(row["text"]) is not str or not row["text"].strip()
                or not _strings(row["input_obligation_ids"]) or not isinstance(row["input_sources"], list)
                or [item.get("input_obligation_id") for item in row["input_sources"] if isinstance(item, dict)] != row["input_obligation_ids"]):
            raise ValueError("CONSTRAINT_CONTEXT_HINTS_INVALID")
        if row["requirement_id"] in seen:
            raise ValueError("CONSTRAINT_CONTEXT_HINTS_INVALID")
        seen.add(row["requirement_id"])
        for source in row["input_sources"]:
            origin = source.get("origin")
            if (set(source) != {"input_obligation_id", "origin", "text_sha256"}
                    or not isinstance(origin, dict) or set(origin) != {"field", "index"}
                    or type(origin["field"]) is not str or not origin["field"]
                    or type(origin["index"]) is not int or origin["index"] < 0
                    or type(source["text_sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", source["text_sha256"]) is None):
                raise ValueError("CONSTRAINT_CONTEXT_PROVENANCE_INVALID")
    retained = _strings(context["retained_hard_boundaries"])
    if retained_hard_boundaries is not None and retained != _strings(retained_hard_boundaries):
        raise ValueError("CONSTRAINT_CONTEXT_BOUNDARIES_CHANGED")
    hard, origins = _boundaries(user, _strings(context["source_plan_boundaries"]), hints, retained)
    if context["hard_boundaries"] != hard or context["boundary_origins"] != origins:
        raise ValueError("CONSTRAINT_CONTEXT_BOUNDARIES_CHANGED")
    if feature_spec is not None:
        if context["feature_spec_sha256"] != feature_spec_payload_sha256(feature_spec) or user != _user_contract(feature_spec):
            raise ValueError("CONSTRAINT_CONTEXT_FEATURE_CHANGED")
        if evidence_pack is not None and context != build_constraint_context(feature_spec=feature_spec,
                evidence_pack=evidence_pack, plan_boundaries=context["source_plan_boundaries"], retained_hard_boundaries=retained):
            raise ValueError("CONSTRAINT_CONTEXT_PROVENANCE_CHANGED")
    elif evidence_pack is not None:
        raise ValueError("CONSTRAINT_CONTEXT_FEATURE_REQUIRED")
    if constraints is not None and _strings(constraints) != hard:
        raise ValueError("CONSTRAINT_CONTEXT_BOUNDARIES_CHANGED")
    for key, actual in (("user_request", user_request), ("target_behavior", target_behavior),
            ("acceptance_criteria", acceptance_criteria), ("runtime_verification_plan", runtime_verification_plan)):
        if actual is not None and user[key] != (list(actual) if isinstance(actual, (list, tuple)) else actual):
            raise ValueError("CONSTRAINT_CONTEXT_USER_CONTRACT_CHANGED")
    return context


def constraint_model_view(context):
    value = validate_constraint_context(context)
    # Scope: the normal prompt still carries full request/FeatureSpec; this adds source labels, not another rewritten request.
    # 为什么要有它：原件仍保留推断全文和来源，但不能把无独立授权的派生命令再次交给模型；真实要求沿正常字段完整提供。
    return {"policy": value["policy"], "projection_version": MODEL_VIEW_VERSION,
        "omission_reason": "Derived preservation interpretations remain in the stored plan; they are not additional user instructions.",
        "omitted_model_preservation_hints": [
        {"requirement_id": row["requirement_id"], "authority": row["authority"],
         "source_fields": [deepcopy(source["origin"]) for source in row["input_sources"]]}
        for row in value["model_preservation_hints"]]}


def constraint_requirement_model_view(context, requirements):
    """Project only identity-bound derived preservation text, retaining other fields.

    为什么要有它：Planning还通过requirements提供同一推断，需同时消除重复命令；不能靠语言关键词删除真正用户要求。
    """
    value = validate_constraint_context(context)
    hints = {row["requirement_id"]: row for row in value["model_preservation_hints"]}
    projected = deepcopy(requirements)
    seen = set()
    for row in projected:
        if row["role"] != "preservation":
            continue
        hint = hints.get(row["requirement_id"])
        if (hint is None or row["text"] != hint["text"]
                or row["input_obligation_ids"] != hint["input_obligation_ids"]
                or row["requirement_id"] in seen):
            raise ValueError("CONSTRAINT_CONTEXT_REQUIREMENT_PROJECTION_CHANGED")
        seen.add(row["requirement_id"])
        row.pop("text")
        row["text_projection"] = "derived_preservation_text_omitted_original_user_contract_retained"
    if seen != set(hints):
        raise ValueError("CONSTRAINT_CONTEXT_REQUIREMENT_PROJECTION_CHANGED")
    return projected


def constraint_plan_model_view(plan, *, feature_spec):
    """Keep an upstream Plan intact except its explicitly derived model projection.

    为什么要有它：重规划不能从旧Plan的完整context重新注入派生命令；原Plan、来源和硬边界继续用于执行校验而不被改写。
    """
    projected = deepcopy(plan)
    if projected.get("constraint_context") is not None:
        context = validate_constraint_context(projected["constraint_context"], feature_spec=feature_spec,
            constraints=projected["side_effect_boundaries"])
        projected["constraint_context"] = constraint_model_view(context)
    return projected
