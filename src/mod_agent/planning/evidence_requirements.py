"""Reusable investigation guidance, not automatically required semantic evidence."""
from __future__ import annotations

from .models import RequestPlan

CURRENT_IMPLEMENTATION_EVIDENCE = (
    "Current implementation source and any required API or documentation facts for "
    "locating the requested change and preserving existing behavior."
)
# 为什么要有它：当前源码是调查方向，实际存在性、版本与范围由原读取和绑定代码核验。

CONFIGURATION_EVIDENCE = (
    "Current configuration representation and existing constraints relevant to the request, "
    "established from configuration data or its implementation source. Identify the applicable "
    "read/write/validation behavior needed for the change. If that implementation or configuration "
    "refers to a schema or settings definition, the referenced definition must also be observed; "
    "an unresolved reference remains an evidence gap."
)
# 为什么要有它：配置可直接由源码表示，但已经引用的外部定义不能因未找到而被视为不适用。


RUNTIME_DIAGNOSTICS_GUIDANCE = (
    "Inspect the current startup and logging implementation and any supplied or available "
    "pre-change logs relevant to the reported failure. Distinguish existing behavior and API "
    "facts needed for the change from startup success or a new diagnostic marker that can "
    "only be observed after the change. Preserve explicitly requested pre-change evidence."
)
# 范围：运行诊断的调查提示。为什么要有它：保留源码与现有日志调查方向，同时将修复后的运行证明留在原验收阶段。


def default_investigation_guidance(plan: RequestPlan) -> list[str]:
    """Guide the existing knowledge-query plan without creating extra obligations."""
    requirements = [CURRENT_IMPLEMENTATION_EVIDENCE]
    if plan.request_interpretation.get("request_type") == "config-only":
        requirements.append(CONFIGURATION_EVIDENCE)
    if plan.request_interpretation.get("request_type") == "runtime diagnostics":
        requirements.append(RUNTIME_DIAGNOSTICS_GUIDANCE)
    # 为什么要有它：quick_patch 也可能是文本工作，路由本身不足以加入配置指导；历史计划不重扫原文。
    return requirements
