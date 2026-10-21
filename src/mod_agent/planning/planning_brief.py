"""Deterministic planning brief builder for mod delivery requests."""

from __future__ import annotations

from dataclasses import replace

from .evidence_requirements import default_investigation_guidance
from .models import ModPlanningBrief, RequestPlan


class ModPlanningBriefBuilder:
    """Build a deterministic planning brief from an existing request plan."""

    def build(
        self,
        plan: RequestPlan,
        *,
        target_game: str | None = None,
        constraints: tuple[str, ...] = (),
        non_goals: tuple[str, ...] = (),
    ) -> ModPlanningBrief:
        request_type = self._request_type(plan)
        blocked = plan.blocked or request_type["value"] == "unknown / needs evidence"
        decision = "needs_clarification" if plan.blocked else (
            "needs_evidence" if blocked else "ready_for_feature_spec"
        )
        blockers = tuple(self._blockers(plan, request_type))
        source_types = self._source_types(str(request_type["value"]))
        knowledge_query_plan = {
            "source_types": list(source_types),
            "queries": self._queries(plan, request_type_value=str(request_type["value"])),
        }
        normalized_goal = plan.user_request
        option = self._planning_option(
            plan,
            request_type_value=str(request_type["value"]),
        )
        selected_plan = {
            "scope": normalized_goal if not blocked else "",
            "constraints": list(constraints),
            "non_goals": list(non_goals),
            "user_confirmed": False,
        }
        handoff = {
            "feature_spec_guidance": {
                "target_behavior": normalized_goal if not blocked else "",
                "target_mod": plan.target_mod,
                "risk_level": plan.risk_level,
                "evidence_requirements": list(option["required_evidence"]),
                "acceptance_signal": option["acceptance_signal"],
            },
            "hook_discovery_guidance": (
                "Run hook discovery before implementation planning."
                if request_type["value"] == "hook-required"
                else "Hook necessity is unresolved; use current evidence to select an implementation mechanism."
            ),
            "patch_review_notes": [
                "Keep patch scope bound to retrieved evidence.",
                "Do not treat this planning brief as implementation evidence.",
            ],
            "runtime_verification_expectations": list(
                self._runtime_expectations(str(request_type["value"]))
            ),
        }
        return ModPlanningBrief(
            feature_id=plan.feature_id,
            decision=decision,
            user_intent={
                "raw_request": plan.user_request,
                "request_interpretation": dict(plan.request_interpretation),
                "normalized_goal": normalized_goal,
                "target_game": target_game,
                "target_mod": plan.target_mod,
                "expected_user_visible_behavior": normalized_goal,
            },
            request_type=request_type,
            knowledge_query_plan=knowledge_query_plan,
            retrieved_experience={
                "relevant_lessons": [],
                "failed_patterns": [],
                "reusable_patterns": [],
                "unavailable_evidence": [
                    "Automated accumulated-experience retrieval is not implemented.",
                    "Current-source evidence lookup has not run in the planning brief builder.",
                ],
            },
            planning_options=(option,),
            selected_plan=selected_plan,
            handoff=handoff,
            verification_boundary=(
                "The planning brief classifies the request and prepares evidence lookup guidance.",
                "The planning brief does not prove source evidence, generate feature_spec, generate code, run hooks, build, or validate runtime behavior.",
            ),
            next_action="ask user" if blocked else "feature_spec",
            blockers=blockers,
        )

    def build_planning_only(
        self,
        plan: RequestPlan,
        *,
        target_game: str | None = None,
        constraints: tuple[str, ...] = (),
        non_goals: tuple[str, ...] = (),
    ) -> ModPlanningBrief:
        """Build a no-write brief that stops hook work before feature_spec."""

        brief = self.build(
            plan,
            target_game=target_game,
            constraints=constraints,
            non_goals=non_goals,
        )
        if brief.request_type["value"] != "hook-required":
            return brief
        return replace(
            brief,
            decision="needs_evidence",
            next_action="hook discovery",
            blockers=(
                *brief.blockers,
                {
                    "stage": "planning_brief",
                    "code": "HOOK_EVIDENCE_REQUIRED",
                    "message": (
                        "Planning-only mode stops before feature_spec until current "
                        "source and API evidence are collected."
                    ),
                },
            ),
        )

    @staticmethod
    def _request_type(plan: RequestPlan) -> dict[str, str]:
        if plan.blocked:
            return {
                "value": "unknown / needs evidence",
                "rationale": "The request planner reported blocking ambiguity.",
            }
        value = plan.request_interpretation.get("request_type")
        if value is None:
            # Historical plans retain their recorded decision without rescanning text.
            # 为什么要有它：重读历史产物不能悄悄改写当时的需求解释。
            value = ("hook-required" if plan.requires_hook_evidence else
                     "config-only" if plan.route == "quick_patch" else "gameplay behavior")
        return {
            "value": str(value),
            "rationale": "Shared provisional request interpretation; semantic decomposition belongs to RequirementProjection.",
        }

    @staticmethod
    def _source_types(request_type_value: str) -> tuple[str, ...]:
        common = ("current_source", "prior_artifacts", "delivery_lessons", "docs")
        if request_type_value == "config-only":
            return common
        if request_type_value == "UI feedback":
            return ("current_source", "game_profile", *common[1:])
        if request_type_value == "runtime diagnostics":
            return ("current_source", "game_profile", "logs", *common[1:])
        if request_type_value == "hook-required":
            return ("current_source", "game_profile", "symbol_search", *common[1:])
        if request_type_value == "unknown / needs evidence":
            return ("docs", "prior_artifacts")
        return ("current_source", "game_profile", "symbol_search", *common[1:])

    @staticmethod
    def _queries(plan: RequestPlan, *, request_type_value: str) -> list[str]:
        queries = [
            plan.user_request,
            f"request_type:{request_type_value}",
            *default_investigation_guidance(plan),
        ]
        if plan.target_mod:
            queries.append(f"target_mod:{plan.target_mod}")
        return queries

    @staticmethod
    def _planning_option(
        plan: RequestPlan,
        *,
        request_type_value: str,
    ) -> dict[str, object]:
        hook_likelihood = "high" if plan.requires_hook_evidence else "unknown"
        # 为什么要有它：自动通用指导不能升格为用户逐项举证义务；具体能力义务仍由本所有者明确列出。
        required_evidence: list[str] = []
        # 范围：模板自动推导的证据。为什么要有它：启动成功和新诊断标记是修改后的观察，不能仅凭诊断分类要求修改前已存在；显式前置要求仍由调用方保留。
        if request_type_value == "UI feedback":
            required_evidence.append("UI or feedback surface evidence")
        elif request_type_value == "hook-required":
            required_evidence.extend(["target symbol evidence", "side effect boundary"])
        elif request_type_value == "unknown / needs evidence":
            required_evidence.append("clarified user intent")

        return {
            "option": "controlled_planning_handoff",
            "expected_complexity": "medium" if plan.risk_level == "high" else plan.risk_level,
            "hook_likelihood": hook_likelihood,
            "risk": plan.risk_level,
            "required_evidence": required_evidence,
            "acceptance_signal": (
                "User-visible requested behavior can be verified without unsupported claims."
            ),
        }

    @staticmethod
    def _runtime_expectations(request_type_value: str) -> tuple[str, ...]:
        if request_type_value == "config-only":
            return ("compile or static validation",)
        if request_type_value == "UI feedback":
            return ("compile or static validation", "user-visible feedback evidence")
        if request_type_value == "runtime diagnostics":
            return ("startup_loaded", "diagnostic marker observed")
        if request_type_value == "hook-required":
            return ("startup_loaded", "target_path_executed", "acceptance_evidence_observed")
        if request_type_value == "unknown / needs evidence":
            return ()
        return ("compile or static validation", "behavior evidence observed")

    @staticmethod
    def _blockers(
        plan: RequestPlan,
        request_type: dict[str, str],
    ) -> list[dict[str, str]]:
        blockers = [
            {
                "stage": "planning_brief",
                "code": uncertainty.uncertainty_id.upper(),
                "message": uncertainty.question,
            }
            for uncertainty in plan.uncertainties
            if uncertainty.severity == "blocking"
        ]
        if request_type["value"] == "unknown / needs evidence" and not blockers:
            blockers.append(
                {
                    "stage": "planning_brief",
                    "code": "REQUEST_TYPE_UNRESOLVED",
                    "message": "The request type needs more evidence before feature_spec handoff.",
                }
            )
        return blockers
