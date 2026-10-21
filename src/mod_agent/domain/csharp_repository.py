"""Explicit shared C# repository scope; no command discovery, execution or game capability."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from src.mod_agent.planning.models import FeatureSpec, RequestPlan

from .models import (
    DomainProfile,
    DomainRoute,
    EvidenceQuery,
    RuntimeProfile,
    VerificationPlan,
    source_definition,
)


@dataclass(frozen=True)
class CSharpRepositoryAdapter:
    """Trusted host-provided public scope/conditions shared by all comparison arms."""

    allowed_roots: tuple[str, ...]
    public_verification_conditions: tuple[str, ...]
    domain_name = "csharp_repository"

    def __post_init__(self):
        # Scope: caller-owned explicit relative roots and public condition text only.
        # 为什么要有它：空范围、路径穿越或通配授权不能变成默认全仓库能力；适配器不负责推测隐藏测试或选择执行命令。
        if not isinstance(self.allowed_roots, tuple) or not self.allowed_roots:
            raise ValueError("CSHARP_REPOSITORY_SCOPE_REQUIRED")
        for root in self.allowed_roots:
            if (type(root) is not str or not root or root in {".", ".."}
                    or PurePosixPath(root.replace("\\", "/")).is_absolute() or PureWindowsPath(root).drive
                    or ".." in PurePosixPath(root.replace("\\", "/")).parts
                    or any(character in root for character in "*?[]")):
                raise ValueError("CSHARP_REPOSITORY_SCOPE_INVALID")
        if (not isinstance(self.public_verification_conditions, tuple) or not self.public_verification_conditions
                or any(type(text) is not str or not text.strip() for text in self.public_verification_conditions)):
            raise ValueError("CSHARP_PUBLIC_VERIFICATION_CONDITIONS_REQUIRED")

    def validate_product_request(self, request):
        # Scope: source/build/test-only product requests, never game startup/deployment.
        # 为什么要有它：中立仓库接入不能借用GTFO运行时授权；实际命令及隔离资源由三臂共享backend另行明确绑定。
        if request.target_domain != self.domain_name or request.requested_completion_scope != "compile_only":
            raise ValueError("CSHARP_REPOSITORY_COMPILE_SCOPE_REQUIRED")
        root = Path(request.isolated_workspace_root).resolve()
        target = Path(request.isolated_target_root).resolve().relative_to(root).as_posix()
        self.identify_project(str(root)).review_policy().bind_target_root(target)

    def identify_project(self, root: str) -> DomainProfile:
        project = Path(root).resolve()
        return DomainProfile(game_id=self.domain_name, display_name="Explicit C# repository",
            runtime="dotnet_build_and_test", mod_loader="not_applicable", harmony="not_applicable",
            status="experimental", allowed_roots=self.allowed_roots,
            forbidden_paths=(".git", ".env", "**/.git/**", "**/.env", "**/bin/**", "**/obj/**"),
            build_profiles=(), evidence_sources=tuple(source_definition(source_id=f"csharp_source_{index}",
                source_type="target_source", project_root=project, relative_root=path)
                for index, path in enumerate(self.allowed_roots, 1)),
            smoke_tests=self.public_verification_conditions,
            known_limitations=("Only explicit source scope and public verification conditions are declared.",
                "No SDK, dependency restore, arbitrary repository layout or real test success is established by this adapter."),
            capability_report_ref="", project_root=str(project))

    def build_feature_spec(self, plan: RequestPlan, *, planning_brief=None) -> FeatureSpec:
        if not isinstance(plan, RequestPlan):
            raise ValueError("CSHARP_REQUEST_PLAN_REQUIRED")
        selected = planning_brief.get("selected_plan", {}) if isinstance(planning_brief, dict) else {}
        non_goals = selected.get("non_goals", [])
        if not isinstance(non_goals, (list, tuple)) or any(type(item) is not str or not item.strip() for item in non_goals):
            raise ValueError("CSHARP_DECLARED_NON_GOALS_INVALID")
        blockers = tuple({"code": row.uncertainty_id.upper(), "message": row.question}
            for row in plan.uncertainties if row.severity == "blocking")
        hook = dict(plan.hook_requirement)
        if hook.get("required") is True:
            blockers += ({"code": "CSHARP_GAME_INTERCEPTION_UNSUPPORTED",
                          "message": "This repository adapter does not authorize game/method-interception execution."},)
        # Scope: exact original request and declared non-goals; only public caller-supplied verification conditions.
        # 为什么要有它：不能把游戏startup/gameplay模板、参考修复或隐藏测试答案塞入通用任务；原始需求和正式检查条件仍完整保留。
        return FeatureSpec(feature_id=plan.feature_id, user_request=plan.user_request,
            target_behavior=plan.user_request, non_goals=tuple(non_goals), target_mod=plan.target_mod,
            risk_level=plan.risk_level,
            acceptance_criteria=(f"Requested behavior is implemented: {plan.user_request}", *self.public_verification_conditions),
            evidence_requirements=("Current repository source relevant to the original request.",
                "Current callers and public API contracts needed for the requested change."),
            runtime_verification_plan=self.public_verification_conditions, hook_requirement=hook,
            blocked=plan.blocked or bool(blockers), blockers=blockers,
            request_interpretation={})

    def collect_knowledge_sources(self, profile):
        return list(profile.evidence_sources)

    def classify_request(self, user_request):
        return DomainRoute(route_id="csharp_repository_source_change", risk_level="medium",
            risk_tags=("csharp_repository",), requires_game_profile=False, blocked=False,
            reason="Use the unchanged original request and explicit shared build/test checks.")

    def build_evidence_queries(self, feature_spec):
        return [EvidenceQuery(query_terms=(feature_spec["user_request"],), required_source_types=("target_source",), blocked_if_missing=True)]

    def safe_edit_roots(self, profile):
        return list(profile.allowed_roots)

    def forbidden_paths(self, profile):
        return list(profile.forbidden_paths)

    def build_profile(self, profile):
        raise ValueError("CSHARP_BUILD_PROFILE_REQUIRES_EXPLICIT_SHARED_BACKEND")

    def runtime_profile(self, profile):
        return RuntimeProfile(profile_id="csharp_repository_no_game_runtime", log_paths=(), startup_markers=(),
            behavior_evidence_sources=(), limitations=("Game runtime is not supported by this adapter.",))

    def verification_strategy(self, feature_spec):
        return VerificationPlan(required_evidence_sources=("build_result", "shared_test_result", "compiled_artifact"),
            acceptance_checks=self.public_verification_conditions, blocked_without=("build_result", "shared_test_result"),
            limitations=("Only actual shared-backend results may establish build/test success.",))
