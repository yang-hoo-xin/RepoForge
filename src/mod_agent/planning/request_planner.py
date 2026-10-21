"""Deterministic request routing for Phase E planning."""

from __future__ import annotations

import re

from .models import RequestPlan, Uncertainty, stable_feature_id

_TERM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_TARGET_MOD_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{3,})\b")
_CJK_CHARACTER_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

_CONFIG_REQUEST_TERMS = (
    "config",
    "configuration",
    "default",
    "json",
    "yaml",
    "toml",
    "ini",
    "setting",
    "settings",
    "logging",
    "配置",
)
_RUNTIME_BEHAVIOR_TERMS = (
    "enemy",
    "enemies",
    "damage",
    "hook",
    "harmony",
    "il2cpp",
    "gameplay",
    "movement",
    "move",
    "walk",
    "walking",
    "wall",
    "walls",
    "slope",
    "slopes",
    "ceiling",
    "ceilings",
    "敌人",
    "怪物",
    "移动",
    "行走",
    "墙面",
    "斜面",
    "天花板",
    "寻路",
    "旋转",
    "表面",
    "游戏行为",
    "钩子",
)
_RUNTIME_DIAGNOSTIC_TERMS = (
    "diagnostic",
    "diagnostics",
    "trace",
    "runtime probe",
    "startup",
    "runtime",
    "诊断",
    "跟踪",
    "启动",
    "运行时",
)
_RUNTIME_RELOAD_TERMS = (
    "restart",
    "reload",
    "重启",
)


def requested_scope_text(user_request: str) -> str:
    """Preserve the complete request; semantic projection owns clause meaning."""

    return user_request


def has_config_intent(user_request: str) -> bool:
    """Return whether the request explicitly includes configuration work."""

    lowered = user_request.lower()
    return any(_contains_term(lowered, term) for term in _CONFIG_REQUEST_TERMS)


def has_runtime_behavior_intent(user_request: str) -> bool:
    """Return runtime investigation leads, not proof of a requested change."""

    requested_scope = requested_scope_text(user_request).lower()
    return any(
        _contains_term(requested_scope, term) for term in _RUNTIME_BEHAVIOR_TERMS
    )


def has_runtime_diagnostic_intent(user_request: str) -> bool:
    """Return whether the requested scope explicitly requires runtime observation."""

    from .request_interpretation import runtime_diagnostic_clauses

    return any(
        _contains_term(clause.lower(), term)
        for clause in runtime_diagnostic_clauses(user_request)
        for term in _RUNTIME_DIAGNOSTIC_TERMS
    )


def has_runtime_reload_intent(user_request: str) -> bool:
    """Return whether acceptance explicitly depends on restart or reload behavior."""

    requested_scope = requested_scope_text(user_request).lower()
    return any(
        _contains_term(requested_scope, term) for term in _RUNTIME_RELOAD_TERMS
    )


def _contains_term(text: str, term: str) -> bool:
    if term.isascii():
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


class RequestPlanner:
    """Turn natural language into a route decision without writing code."""

    def plan(
        self,
        user_request: str,
        *,
        target_mod: str | None = None,
    ) -> RequestPlan:
        """Return a structured request plan."""

        from .request_interpretation import interpret_request

        request = user_request
        from .clarified_request import ANCHORED_VERSION, projection_context, source_hint_text
        clarified = projection_context(request)
        semantic_request = source_hint_text(request) if clarified is not None and clarified["schema_version"] == ANCHORED_VERSION else request
        # 为什么要有它：route receipt路径不是用户需求，不能触发歧义或配置启发式；完整原文仍作为FeatureSpec身份保存。
        interpretation = interpret_request(semantic_request)
        interpretation["raw_request"] = request
        feature_id = stable_feature_id(request)
        terms = _TERM_RE.findall(semantic_request)
        uncertainties: list[Uncertainty] = []

        if self._is_ambiguous(semantic_request, terms):
            uncertainties.append(
                Uncertainty(
                    uncertainty_id="ambiguous_user_request",
                    severity="blocking",
                    question="What exact behavior or code outcome should be changed?",
                )
            )
            return RequestPlan(
                feature_id=feature_id,
                user_request=request,
                route="blocked_ambiguous",
                risk_level="blocked",
                target_mod=target_mod,
                requires_hook_evidence=False,
                requires_runtime_evidence=False,
                hook_requirement={
                    "required": False,
                    "reason": "blocked_ambiguous_request",
                },
                uncertainties=tuple(uncertainties),
                blocked=True,
                request_interpretation=interpretation,
            )

        resolved_target_mod = target_mod or self._extract_target_mod(semantic_request)
        quick_patch = interpretation["request_type"] == "config-only"
        requires_hook = interpretation["hook_requirement"]["required"]
        requires_runtime_diagnostics = has_runtime_diagnostic_intent(semantic_request)
        requires_runtime = interpretation["requires_runtime_evidence"]
        hook_requirement = dict(interpretation["hook_requirement"])
        route = (
            "game_hook_investigation"
            if requires_hook
            else ("quick_patch" if quick_patch else "standard_feature")
        )
        risk_level = (
            "high"
            if requires_hook or requires_runtime_diagnostics
            else ("low" if quick_patch else "medium")
        )

        if resolved_target_mod is None:
            uncertainties.append(
                Uncertainty(
                    uncertainty_id="target_mod_unresolved",
                    severity="warning",
                    question="Which target mod or plugin project should receive the change?",
                    default_action="continue_with_evidence",
                )
            )

        return RequestPlan(
            feature_id=feature_id,
            user_request=request,
            route=route,
            risk_level=risk_level,
            target_mod=resolved_target_mod,
            requires_hook_evidence=requires_hook,
            requires_runtime_evidence=requires_runtime,
            hook_requirement=hook_requirement,
            uncertainties=tuple(uncertainties),
            blocked=False,
            request_interpretation=interpretation,
        )

    @staticmethod
    def _is_ambiguous(request: str, terms: list[str]) -> bool:
        lowered = request.lower()
        if len(request) < 8:
            return True
        # Scope: the whole vague-only request, never an embedded token or path.
        # 为什么要有它：具体需求中的test、Latest或“修一下”不是歧义证据；只有没有具体对象/行为的泛指请求在这里阻断。
        english_vague = (
            r"(?:(?:please|can you|could you|would you|help me|just)\s+)*"
            r"(?:fix (?:it|that issue)|that issue|test)"
            r"(?:\s+(?:please|thanks))?[.!?]*"
        )
        chinese_vague = (
            r"(?:(?:请|麻烦你?|帮我|帮忙|处理(?:一下)?|修复|修一下|"
            r"解决(?:一下)?|测试|那个问题|这个问题|这个东西|一下|吧|谢谢)\s*)+"
            r"[。！？.!?]*"
        )
        if re.fullmatch(english_vague, lowered.strip()) or re.fullmatch(
            chinese_vague, lowered.strip()
        ):
            return True
        if len(terms) >= 2:
            return False
        return len(_CJK_CHARACTER_RE.findall(request)) < 6

    @staticmethod
    def _extract_target_mod(request: str) -> str | None:
        ignored = {
            "GTFO",
            "BepInEx",
            "Harmony",
            "IL2CPP",
            "Unity",
            "JSON",
            "PersistentID",
        }
        for match in _TARGET_MOD_RE.finditer(request):
            candidate = match.group(1)
            if candidate not in ignored:
                return candidate
        return None
