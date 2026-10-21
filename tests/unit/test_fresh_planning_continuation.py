from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from src.libs.llm import (
    AgentToolCall,
    AgentTurnResponse,
    AgentUsage,
    ChatResponse,
    Message,
)
from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.cli import main
from src.mod_agent.evidence import (
    RgLiveEvidenceProvider,
    build_requirement_coverage_pack,
    materialize_requirement_coverage_projection,
)
from src.mod_agent.execution import PatchApplyError
from src.mod_agent.fresh_planning_continuation import (
    PATCH_CONTINUATION_CONTRACT_VERSION,
    PLANNING_CONTINUATION_CONTRACT_VERSION,
    PatchContinuationAttemptJournal,
    PatchContinuationInputV1,
    PlanningContinuationError,
    PlanningContinuationInputV1,
    preflight_patch_continuation,
    run_fresh_patch_continuation,
    run_fresh_planning_continuation,
)
from src.mod_agent.local_entry import (
    LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
    LocalEntryError,
    LocalOllamaInitialPatchAdapter,
    LocalOllamaInitialPatchProfileV1,
    _local_patch_response_sidecar_path,
    _patch_agent_initial_messages,
    _write_local_patch_response_sidecar,
    make_exact_no_write_dry_run_factory,
)
from src.mod_agent.planning import FeatureSpec, ImplementationPlan, ImplementationPlanner
from src.mod_agent.proposal import LLMExactPatchGenerator, PatchGenerationError
from src.mod_agent.proposal.patch_agent_loop import (
    PATCH_SOURCE_INSPECTION_TOOL_NAME,
    PatchAgentLoopPolicy,
)
from src.mod_agent.workflow import build_graph, create_initial_state


def _fresh_execution_receipt() -> dict[str, object]:
    roles = ["requirement_projection", "requirement_findings"]
    return {
        "roles": roles,
        "role_call_counts": {
            "requirement_projection": 1,
            "requirement_projection_correction": 0,
            "requirement_findings": 1,
            "requirement_findings_correction": 0,
        },
        "call_count": 2,
        "correction_count": 0,
        "attempts": [
            {
                "attempt_index": index,
                "role": role,
                "duration_seconds": 0.01,
                "status": "accepted",
                "error_code": None,
                "payload_canonical_sha256": str(index) * 64,
                "schema_canonical_sha256": str(index + 2) * 64,
                "output_canonical_sha256": str(index + 4) * 64,
            }
            for index, role in enumerate(roles, start=1)
        ],
        "budget": {
            "maximum_total_calls": 4,
            "maximum_calls_by_role": {
                "requirement_projection": 1,
                "requirement_projection_correction": 1,
                "requirement_findings": 1,
                "requirement_findings_correction": 1,
            },
            "automatic_provider_fallback": False,
            "automatic_retry": False,
        },
        "admission": {
            "provider": "ollama",
            "execution_locality": "local",
            "external_disclosure_occurred": False,
        },
    }


def _feature_payload() -> dict[str, object]:
    return FeatureSpec(
        feature_id="feature_planning_continuation",
        user_request="Change the marker while preserving existing feedback.",
        target_behavior="Use ChangeMarker for the configurable marker behavior.",
        non_goals=("PreserveFeedback must remain unchanged.",),
        target_mod="SyntheticMod",
        risk_level="medium",
        acceptance_criteria=("RuntimeMarker is observed after the change.",),
        evidence_requirements=("ConfigKey must exist before patching.",),
        runtime_verification_plan=("Observe RuntimeMarker.",),
    ).to_artifact_payload()


def _projection() -> dict[str, object]:
    return {
        "requirements": [
            {
                "role": "change",
                "text": "Use ChangeMarker for the configurable marker behavior.",
                "input_obligation_indexes": [1, 2],
                "query_leads": ["ChangeMarker"],
            },
            {
                "role": "pre_patch_evidence",
                "text": "ConfigKey must exist before patching.",
                "input_obligation_indexes": [3],
                "query_leads": ["ConfigKey"],
            },
            {
                "role": "preservation",
                "text": "PreserveFeedback must remain unchanged.",
                "input_obligation_indexes": [4],
                "query_leads": ["PreserveFeedback"],
            },
            {
                "role": "post_change_acceptance",
                "text": "RuntimeMarker is observed after the change.",
                "input_obligation_indexes": [5, 6],
                "query_leads": ["RuntimeMarker"],
            },
        ],
        "relations": [
            {
                "parent_requirement_indexes": [1, 3],
                "meaning": "The editable behavior carries the preservation guard.",
            },
            {
                "parent_requirement_indexes": [2],
                "meaning": "Configuration evidence exists before patching.",
            },
            {
                "parent_requirement_indexes": [4],
                "meaning": "Runtime acceptance is evaluated after the change.",
            },
        ],
        "input_dispositions": [],
    }


def _write_sources(workspace: Path) -> Path:
    source = workspace / "generated_mods" / "SyntheticMod"
    source.mkdir(parents=True)
    behavior = source / "Behavior.cs"
    behavior.write_text(
        "public sealed class Behavior\n"
        "{\n"
        "    public void ChangeMarker()\n"
        "    {\n"
        "        KeepExistingFeedback();\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (source / "Configuration.cs").write_text(
        "public sealed class Configuration\n"
        "{\n"
        '    public const string ConfigKey = "Marker.Enabled";\n'
        "}\n",
        encoding="utf-8",
    )
    (source / "Preservation.cs").write_text(
        "public sealed class Preservation\n{\n    public void PreserveFeedback() { }\n}\n",
        encoding="utf-8",
    )
    return behavior


def _evidence_by_requirement(
    materialized: dict[str, object],
    retrieved,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in retrieved.budget_summary["per_query_execution"]:
        result.setdefault(str(row["requirement_id"]), []).extend(
            str(value) for value in row["evidence_ids"]
        )
    return result


def _positive_findings(
    materialized: dict[str, object],
    retrieved,
) -> dict[str, object]:
    evidence = _evidence_by_requirement(materialized, retrieved)
    requirements = materialized["requirements"]
    change_id = str(requirements[0]["requirement_id"])
    prepatch_id = str(requirements[1]["requirement_id"])
    return {
        "findings": [
            {
                "relation_index": 1,
                "relationship": "support",
                "evidence_ids": [evidence[change_id][0]],
                "edit_target_evidence_ids": [evidence[change_id][0]],
                "unresolved_gaps": [],
            },
            {
                "relation_index": 2,
                "relationship": "support",
                "evidence_ids": [evidence[prepatch_id][0]],
                "edit_target_evidence_ids": [],
                "unresolved_gaps": [],
            },
            {
                "relation_index": 3,
                "relationship": "missing",
                "evidence_ids": [],
                "edit_target_evidence_ids": [],
                "unresolved_gaps": ["Runtime observation occurs after patching."],
            },
        ]
    }


@dataclass(frozen=True)
class _Inputs:
    request: PlanningContinuationInputV1
    store: ArtifactStore
    workspace: Path
    behavior: Path
    feature_ref: ArtifactRef
    evidence_ref: ArtifactRef
    feature_payload: dict[str, object]


def _inputs(tmp_path: Path) -> _Inputs:
    workspace = tmp_path / "workspace"
    store_root = tmp_path / "artifact-store"
    behavior = _write_sources(workspace)
    feature = _feature_payload()
    materialized = materialize_requirement_coverage_projection(
        feature_data=feature,
        request=str(feature["user_request"]),
        projection=_projection(),
    )
    retrieved = RgLiveEvidenceProvider(workspace).collect_query_set(
        queries=tuple(dict(value) for value in materialized["queries"]),
        search_root="generated_mods/SyntheticMod",
        max_items=len(materialized["queries"]),
    )
    pack = build_requirement_coverage_pack(
        query=str(feature["user_request"]),
        materialized_projection=materialized,
        retrieved_pack=retrieved,
        findings=_positive_findings(materialized, retrieved),
    )
    pack = replace(
        pack,
        budget_summary={
            **pack.budget_summary,
            "fresh_coverage_model_execution": _fresh_execution_receipt(),
        },
    )
    assert pack.can_generate_patch is True
    store = ArtifactStore(store_root)
    feature_ref = store.write(
        feature_id=str(feature["feature_id"]),
        artifact_type="feature_spec",
        payload=feature,
        status="success",
    )
    evidence_ref = store.write(
        feature_id=str(feature["feature_id"]),
        artifact_type="evidence_pack",
        payload=pack.to_artifact_payload(),
        status="success",
        upstream_refs=[feature_ref],
    )
    request = PlanningContinuationInputV1(
        contract_version=PLANNING_CONTINUATION_CONTRACT_VERSION,
        artifact_store_root=str(store_root),
        workspace_root=str(workspace),
        feature_spec_ref=feature_ref,
        evidence_pack_ref=evidence_ref,
        planning_target_evidence_ids=tuple(
            str(value)
            for value in pack.requirement_coverage["edit_target_evidence_ids"]
        ),
    )
    return _Inputs(
        request=request,
        store=store,
        workspace=workspace,
        behavior=behavior,
        feature_ref=feature_ref,
        evidence_ref=evidence_ref,
        feature_payload=feature,
    )


def _source_hashes(workspace: Path) -> dict[str, str]:
    return {
        path.relative_to(workspace).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in workspace.rglob("*.cs")
    }


def _patch_request(inputs: _Inputs) -> PatchContinuationInputV1:
    planning = run_fresh_planning_continuation(inputs.request)
    assert planning.status == "terminal"
    assert planning.implementation_plan_artifact is not None
    return PatchContinuationInputV1(
        contract_version=PATCH_CONTINUATION_CONTRACT_VERSION,
        artifact_store_root=inputs.request.artifact_store_root,
        workspace_root=inputs.request.workspace_root,
        feature_spec_ref=inputs.feature_ref,
        evidence_pack_ref=inputs.evidence_ref,
        implementation_plan_ref=ArtifactRef(
            **planning.implementation_plan_artifact
        ),
    )


class _NativePatchLLM:
    def __init__(self, *, after_response=None, content_override: str | None = None) -> None:
        self.calls = 0
        self.after_response = after_response
        self.content_override = content_override
        self.last_content: str | None = None

    def chat(self, messages, trace=None, **kwargs):
        del trace
        self.calls += 1
        kwargs["provider_request_recorder"](
            {
                "method": "POST",
                "destination": "http://127.0.0.1:11434/api/chat",
                "body_sha256": "1" * 64,
                "body_size_bytes": 1024,
                "timeout_seconds": 30.0,
                "trust_environment": False,
                "follow_redirects": False,
                "context_admission": {"admitted": True},
                "model": "qwen3:4b-instruct",
                "output_budget": 1024,
                "context_window": 8192,
                "temperature": 0.0,
                "messages_sha256": "2" * 64,
                "response_format_sha256": "3" * 64,
            }
        )
        prompt = json.loads(messages[1].content)
        required = prompt["required_change_units"]
        contexts = prompt["edit_binding_contexts"]
        changes = []
        for item in required:
            binding_id = item["edit_binding_id"]
            source_excerpt = contexts[binding_id]["source_excerpt"]
            changes.append(
                {
                    "edit_binding_id": binding_id,
                    "rationale": f'{prompt["target_behavior"]} 精确',
                    "rollback_hint": "Restore the selected source span.",
                    "after_text": source_excerpt.replace(
                        "KeepExistingFeedback();",
                        "KeepExistingFeedback();\n        RecordRuntimeMarker();",
                    ),
                }
            )
        if self.after_response is not None:
            self.after_response()
        self.last_content = self.content_override or json.dumps(
            {"changes": changes},
            ensure_ascii=False,
            indent=2,
        ).replace("\n", "\r\n")
        return ChatResponse(
            content=self.last_content,
            model="qwen3:4b-instruct",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            raw_response={"done": True, "done_reason": "stop"},
        )


def _fixture_visible_binding_text(prompt, binding_id, messages):
    """Read only a literal or the input-only Ck reference actually supplied."""
    binding = prompt["edit_binding_contexts"][binding_id]
    if "source_excerpt" in binding:
        return binding["source_excerpt"]
    history, views = None, {}
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else message.role
        value = message.get("content") if isinstance(message, dict) else message.content
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if not isinstance(value, dict):
            continue
        if role == "assistant" and value.get("input_only") is True:
            history = value
        view = value.get("validator_feedback", {}).get("working_code_view")
        if isinstance(view, dict):
            views[view["code_sha256"]] = view
    assert history is not None and history["Ck"] in views
    change = next(row for row in history["changes"] if row["edit_binding_id"] == binding_id)
    view = views[history["Ck"]]
    file = next(row for row in view["files"] if row["path"] == binding["canonical_target_path"])
    assert file["complete"] is True and len(file["hunks"]) == 1
    offset, length, *prefix = change["after_text_ref"]
    return (prefix[0] if prefix else "") + file["hunks"][0]["text"].encode("utf-8")[offset:offset + length].decode("utf-8")


class _PatchAgentTurnClient:
    provider = "ollama_chat_local"
    model = "qwen3:4b-instruct"

    def __init__(self) -> None:
        self.requests = []

    def complete_turn(self, request):
        self.requests.append(request)
        turn = len(self.requests)
        if turn == 1:
            return AgentTurnResponse(
                provider=self.provider,
                model=self.model,
                content=None,
                tool_calls=(
                    AgentToolCall(
                        call_id="patch-call-001",
                        name=PATCH_SOURCE_INSPECTION_TOOL_NAME,
                        arguments={"binding_id": "edit_binding_001"},
                    ),
                ),
                stop_reason="tool_calls",
                usage=AgentUsage(input_tokens=40, output_tokens=5),
                provider_state={
                    "content": None,
                    "tool_calls": [{"id": "patch-call-001"}],
                },
            )
        prompt = json.loads(request.messages[1].content)
        changes = []
        for item in prompt["required_change_units"]:
            binding_id = item["edit_binding_id"]
            source_excerpt = _fixture_visible_binding_text(prompt, binding_id, request.messages)
            changes.append(
                {
                    "edit_binding_id": binding_id,
                    "rationale": "Model annotation only.",
                    "rollback_hint": "Restore the selected source span.",
                    "after_text": source_excerpt.replace(
                        "KeepExistingFeedback();",
                        "KeepExistingFeedback();\n        RecordRuntimeMarker();",
                    ),
                }
            )
        return AgentTurnResponse(
            provider=self.provider,
            model=self.model,
            content=json.dumps({"changes": changes}),
            stop_reason="stop",
            usage=AgentUsage(input_tokens=80, output_tokens=20),
        )


def test_patch_agent_prompt_uses_ref_first_bounded_supporting_excerpt() -> None:
    full_excerpt = "API behavior\n" * 200
    projected = _patch_agent_initial_messages(
        (
            Message(role="system", content="Generate Patch."),
            Message(
                role="user",
                content=json.dumps(
                    {
                        "supporting_evidence": [
                            {
                                "binding_id": "support_binding_001",
                                "evidence_id": "evidence_001",
                                "excerpt": full_excerpt,
                            }
                        ],
                        "edit_binding_contexts": {
                            "edit_binding_001": {
                                "source_excerpt": "exact editable source"
                            }
                        },
                    }
                ),
            ),
        )
    )

    payload = json.loads(projected[1].content)
    support = payload["supporting_evidence"][0]
    assert len(support["excerpt"].encode("utf-8")) <= 1024
    assert support["excerpt_truncated"] is True
    assert support["excerpt_utf8_bytes"] == len(full_excerpt.encode("utf-8"))
    assert support["excerpt_sha256"] == hashlib.sha256(
        full_excerpt.encode("utf-8")
    ).hexdigest()
    assert payload["edit_binding_contexts"]["edit_binding_001"][
        "source_excerpt"
    ] == "exact editable source"


def _local_patch_adapter(
    inputs: _Inputs,
    *,
    after_response=None,
    content_override: str | None = None,
) -> tuple[LocalOllamaInitialPatchAdapter, _NativePatchLLM]:
    identity = {
        "name": "qwen3:4b-instruct",
        "model": "qwen3:4b-instruct",
        "digest": "a" * 64,
        "modified_at": "2026-09-04T00:00:00Z",
        "size": 1,
        "details": {"context_length": 8192},
        "capabilities": ["completion"],
        "base_url": "http://127.0.0.1:11434",
    }
    llm = _NativePatchLLM(
        after_response=after_response,
        content_override=content_override,
    )
    adapter = LocalOllamaInitialPatchAdapter(
        profile=LocalOllamaInitialPatchProfileV1(
            profile_version=LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
            base_url="http://127.0.0.1:11434",
            model="qwen3:4b-instruct",
            context_window=8192,
            output_budget=1024,
            timeout_seconds=30.0,
        ),
        artifact_store=inputs.store,
        identity_reader=lambda _base_url, _model: deepcopy(identity),
        llm_factory=lambda _profile: llm,
    )
    return adapter, llm


def _write_evidence_revision(
    inputs: _Inputs,
    *,
    payload: dict[str, object],
) -> ArtifactRef:
    return inputs.store.write(
        feature_id=inputs.evidence_ref.feature_id,
        artifact_type="evidence_pack",
        payload=payload,
        status="success",
        upstream_refs=[inputs.feature_ref],
    )


def _direct_graph_state(inputs: _Inputs):
    state = create_initial_state(
        run_id="run_direct_planning_continuation",
        request_id="request_direct_planning_continuation",
        user_request=str(inputs.feature_payload["user_request"]),
        target_mod=str(inputs.feature_payload["target_mod"]),
    )
    state["feature_id"] = inputs.feature_ref.feature_id
    state["risk_level"] = inputs.feature_payload["risk_level"]
    state["current_stage"] = "evidence_retrieval"
    state["next_stage"] = "implementation_plan"
    state["artifacts"] = {
        "feature_spec": inputs.feature_ref.path,
        "evidence_pack": inputs.evidence_ref.path,
    }
    state["artifact_hashes"] = {
        "feature_spec": inputs.feature_ref.artifact_hash,
        "evidence_pack": inputs.evidence_ref.artifact_hash,
    }
    return state


def _cli_args(inputs: _Inputs) -> list[str]:
    args = [
        "continue-fresh-planning",
        "--artifact-root",
        inputs.request.artifact_store_root,
        "--workspace-root",
        inputs.request.workspace_root,
        "--feature-id",
        inputs.feature_ref.feature_id,
        "--feature-spec-path",
        inputs.feature_ref.path,
        "--feature-spec-hash",
        inputs.feature_ref.artifact_hash,
        "--evidence-pack-path",
        inputs.evidence_ref.path,
        "--evidence-pack-hash",
        inputs.evidence_ref.artifact_hash,
    ]
    for evidence_id in inputs.request.planning_target_evidence_ids:
        args.extend(["--planning-target-evidence-id", evidence_id])
    return args


def _patch_cli_args(request: PatchContinuationInputV1) -> list[str]:
    return [
        "continue-fresh-patch",
        "--artifact-root",
        request.artifact_store_root,
        "--workspace-root",
        request.workspace_root,
        "--feature-id",
        request.feature_spec_ref.feature_id,
        "--feature-spec-path",
        request.feature_spec_ref.path,
        "--feature-spec-hash",
        request.feature_spec_ref.artifact_hash,
        "--evidence-pack-path",
        request.evidence_pack_ref.path,
        "--evidence-pack-hash",
        request.evidence_pack_ref.artifact_hash,
        "--implementation-plan-path",
        request.implementation_plan_ref.path,
        "--implementation-plan-hash",
        request.implementation_plan_ref.artifact_hash,
        "--ollama-model",
        "qwen3:4b-instruct",
        "--ollama-context-window",
        "8192",
        "--ollama-output-budget",
        "1024",
        "--ollama-timeout-seconds",
        "30",
    ]


def test_exact_fresh_refs_run_existing_planning_once_and_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    source_before = _source_hashes(inputs.workspace)
    calls = 0
    original_build = ImplementationPlanner.build

    def recording_build(self, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["formal_requirement_coverage_required"] is True
        assert kwargs["planning_target_evidence_ids"] == (
            inputs.request.planning_target_evidence_ids
        )
        return original_build(self, **kwargs)

    monkeypatch.setattr(ImplementationPlanner, "build", recording_build)
    result = run_fresh_planning_continuation(inputs.request)
    public = result.to_dict()

    assert calls == 1
    assert result.status == "terminal"
    assert result.plan_blocking is False
    assert result.first_failure is None
    assert public["execution_counts"] == {
        "fresh_coverage": 0,
        "model": 0,
        "planning": 1,
        "patch": 0,
        "apply": 0,
        "build": 0,
        "runtime": 0,
    }
    assert public["stopped_before"] == "patch_proposal"
    assert public["product_source_writes_performed"] is False
    assert public["source_modified_during_run"] is False
    assert _source_hashes(inputs.workspace) == source_before
    plan_ref = ArtifactRef(**public["implementation_plan_artifact"])
    plan = inputs.store.read(plan_ref)
    assert plan["data"]["blocking"] is False
    assert {item["artifact_type"] for item in plan["inputs"]["upstream_artifacts"]} == {
        "feature_spec",
        "evidence_pack",
    }
    assert {
        ref.artifact_type
        for ref in inputs.store.list_feature_artifacts(inputs.feature_ref.feature_id)
    } == {"feature_spec", "evidence_pack", "implementation_plan"}


def test_human_assisted_continuation_without_target_selection_blocks_before_patch(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = replace(inputs.request, planning_target_evidence_ids=())

    result = run_fresh_planning_continuation(request)
    public = result.to_dict()

    assert result.status == "blocked"
    assert result.plan_blocking is True
    assert result.first_failure == {
        "code": "PLANNING_TARGET_SELECTION_REQUIRED",
        "stage": "implementation_plan",
    }
    assert public["execution_counts"]["planning"] == 1
    assert public["execution_counts"]["patch"] == 0


def test_planning_continuation_preserves_blocked_plan_and_stops(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)

    def blocked_build(_self, *, feature_spec, **_kwargs):
        return ImplementationPlan(
            feature_id=feature_spec.feature_id,
            target_files=(),
            target_symbols=(),
            evidence_refs=(),
            proposed_changes=(),
            side_effect_boundaries=(),
            test_plan=(),
            rollback_plan=(),
            open_questions=("Synthetic planning blocker.",),
            blocking=True,
            blockers=(
                {
                    "code": "SYNTHETIC_PLANNING_BLOCKER",
                    "message": "Synthetic planning blocker.",
                },
            ),
        )

    monkeypatch.setattr(ImplementationPlanner, "build", blocked_build)
    result = run_fresh_planning_continuation(inputs.request)
    public = result.to_dict()

    assert result.status == "blocked"
    assert result.plan_blocking is True
    assert result.first_failure == {
        "code": "SYNTHETIC_PLANNING_BLOCKER",
        "stage": "implementation_plan",
    }
    assert public["execution_counts"]["planning"] == 1
    assert public["execution_counts"]["patch"] == 0
    assert result.implementation_plan_artifact is not None
    assert {
        ref.artifact_type
        for ref in inputs.store.list_feature_artifacts(inputs.feature_ref.feature_id)
    } == {"feature_spec", "evidence_pack", "implementation_plan"}


@pytest.mark.parametrize(
    ("ref_field", "replacement"),
    [
        ("feature_spec_ref", "latest"),
        ("evidence_pack_ref", "wrong_hash"),
    ],
)
def test_planning_continuation_rejects_non_exact_refs_before_planning(
    tmp_path: Path,
    monkeypatch,
    ref_field: str,
    replacement: str,
) -> None:
    inputs = _inputs(tmp_path)
    feature_ref = inputs.feature_ref
    evidence_ref = inputs.evidence_ref
    if ref_field == "feature_spec_ref":
        feature_ref = ArtifactRef(
            feature_id=feature_ref.feature_id,
            artifact_type=feature_ref.artifact_type,
            path=str(
                Path(inputs.request.artifact_store_root)
                / "features"
                / feature_ref.feature_id
                / "feature_spec.json"
            ),
            artifact_hash=feature_ref.artifact_hash,
        )
    elif replacement == "wrong_hash":
        evidence_ref = ArtifactRef(
            feature_id=evidence_ref.feature_id,
            artifact_type=evidence_ref.artifact_type,
            path=evidence_ref.path,
            artifact_hash="0" * 64,
        )
    request = PlanningContinuationInputV1(
        contract_version=PLANNING_CONTINUATION_CONTRACT_VERSION,
        artifact_store_root=inputs.request.artifact_store_root,
        workspace_root=inputs.request.workspace_root,
        feature_spec_ref=feature_ref,
        evidence_pack_ref=evidence_ref,
    )
    monkeypatch.setattr(
        ImplementationPlanner,
        "build",
        lambda *_args, **_kwargs: pytest.fail("Planning must remain unreachable"),
    )

    result = run_fresh_planning_continuation(request)

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PLANNING_CONTINUATION_EXACT_REVISION_REQUIRED",
        "stage": "artifact_admission",
    }
    assert result.planning_execution_count == 0
    assert result.implementation_plan_artifact is None


def test_planning_continuation_rejects_stale_artifact_before_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    updated = deepcopy(inputs.feature_payload)
    updated["target_behavior"] = "Use a later marker behavior."
    inputs.store.write(
        feature_id=inputs.feature_ref.feature_id,
        artifact_type="feature_spec",
        payload=updated,
        status="success",
    )
    monkeypatch.setattr(
        ImplementationPlanner,
        "build",
        lambda *_args, **_kwargs: pytest.fail("Planning must remain unreachable"),
    )

    result = run_fresh_planning_continuation(inputs.request)

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PLANNING_CONTINUATION_ARTIFACT_STALE",
        "stage": "artifact_admission",
    }
    assert result.planning_execution_count == 0


def test_planning_continuation_rejects_superseded_evidence_pack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    updated = deepcopy(inputs.store.read(inputs.evidence_ref)["data"])
    updated["message"] = "A later Fresh EvidencePack revision."
    newer_ref = _write_evidence_revision(inputs, payload=updated)
    assert newer_ref.artifact_hash != inputs.evidence_ref.artifact_hash
    monkeypatch.setattr(
        ImplementationPlanner,
        "build",
        lambda *_args, **_kwargs: pytest.fail("Planning must remain unreachable"),
    )

    result = run_fresh_planning_continuation(inputs.request)

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PLANNING_CONTINUATION_ARTIFACT_STALE",
        "stage": "artifact_admission",
    }
    assert result.planning_execution_count == 0
    assert result.implementation_plan_artifact is None


def test_planning_continuation_rejects_missing_fresh_origin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    payload = deepcopy(inputs.store.read(inputs.evidence_ref)["data"])
    del payload["budget_summary"]["fresh_coverage_model_execution"]
    evidence_ref = _write_evidence_revision(inputs, payload=payload)
    request = replace(inputs.request, evidence_pack_ref=evidence_ref)
    monkeypatch.setattr(
        ImplementationPlanner,
        "build",
        lambda *_args, **_kwargs: pytest.fail("Planning must remain unreachable"),
    )

    result = run_fresh_planning_continuation(request)

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PLANNING_CONTINUATION_FRESH_ORIGIN_MISSING",
        "stage": "artifact_admission",
    }
    assert result.planning_execution_count == 0
    assert result.implementation_plan_artifact is None


@pytest.mark.parametrize(
    "rejected_role",
    ["requirement_projection", "requirement_findings"],
)
def test_planning_continuation_rejects_unrepaired_fresh_model_output(
    tmp_path: Path,
    monkeypatch,
    rejected_role: str,
) -> None:
    inputs = _inputs(tmp_path)
    payload = deepcopy(inputs.store.read(inputs.evidence_ref)["data"])
    execution = payload["budget_summary"]["fresh_coverage_model_execution"]
    attempt = next(item for item in execution["attempts"] if item["role"] == rejected_role)
    attempt["status"] = "rejected"
    attempt["error_code"] = "SYNTHETIC_REJECTION"
    evidence_ref = _write_evidence_revision(inputs, payload=payload)
    request = replace(inputs.request, evidence_pack_ref=evidence_ref)
    monkeypatch.setattr(
        ImplementationPlanner,
        "build",
        lambda *_args, **_kwargs: pytest.fail("Planning must remain unreachable"),
    )

    result = run_fresh_planning_continuation(request)

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PLANNING_CONTINUATION_FRESH_ORIGIN_INVALID",
        "stage": "artifact_admission",
    }
    assert result.planning_execution_count == 0
    assert result.implementation_plan_artifact is None


def test_planning_continuation_accepts_repaired_fresh_model_output(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = deepcopy(inputs.store.read(inputs.evidence_ref)["data"])
    execution = payload["budget_summary"]["fresh_coverage_model_execution"]
    projection_attempt, findings_attempt = execution["attempts"]
    projection_attempt["status"] = "rejected"
    projection_attempt["error_code"] = "SYNTHETIC_REJECTION"
    correction_attempt = {
        **projection_attempt,
        "attempt_index": 2,
        "role": "requirement_projection_correction",
        "status": "accepted",
        "error_code": None,
        "payload_canonical_sha256": "a" * 64,
        "schema_canonical_sha256": "b" * 64,
        "output_canonical_sha256": "c" * 64,
    }
    findings_attempt["attempt_index"] = 3
    execution["roles"] = [
        "requirement_projection",
        "requirement_projection_correction",
        "requirement_findings",
    ]
    execution["role_call_counts"]["requirement_projection_correction"] = 1
    execution["call_count"] = 3
    execution["correction_count"] = 1
    execution["attempts"] = [
        projection_attempt,
        correction_attempt,
        findings_attempt,
    ]
    evidence_ref = _write_evidence_revision(inputs, payload=payload)
    request = replace(inputs.request, evidence_pack_ref=evidence_ref)

    result = run_fresh_planning_continuation(request)

    assert result.status == "terminal"
    assert result.first_failure is None
    assert result.planning_execution_count == 1
    assert result.implementation_plan_artifact is not None


def test_planning_continuation_rejects_current_source_drift_before_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    inputs.behavior.write_text("changed outside continuation\n", encoding="utf-8")
    monkeypatch.setattr(
        ImplementationPlanner,
        "build",
        lambda *_args, **_kwargs: pytest.fail("Planning must remain unreachable"),
    )

    result = run_fresh_planning_continuation(inputs.request)

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PLANNING_CONTINUATION_SOURCE_HASH_MISMATCH",
        "stage": "source_validation",
    }
    assert result.planning_execution_count == 0


def test_planning_continuation_blocks_source_drift_before_plan_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    original_build = ImplementationPlanner.build

    def build_then_drift(self, **kwargs):
        plan = original_build(self, **kwargs)
        inputs.behavior.write_text("changed during planning\n", encoding="utf-8")
        return plan

    monkeypatch.setattr(ImplementationPlanner, "build", build_then_drift)
    result = run_fresh_planning_continuation(inputs.request)

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PLANNING_CONTINUATION_SOURCE_HASH_MISMATCH",
        "stage": "source_validation",
    }
    assert result.planning_execution_count == 1
    assert result.implementation_plan_artifact is None
    assert result.source_modified_during_run is True
    assert result.to_dict()["execution_counts"]["patch"] == 0


def test_planning_continuation_graph_rejects_state_and_control_drift(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    graph = build_graph(
        inputs.store,
        planning_continuation_mode=inputs.request,
        use_default_proposal_git_binding=False,
    )
    invalid_state = create_initial_state(
        run_id="run_invalid_planning_continuation",
        request_id="request_invalid_planning_continuation",
        user_request=str(inputs.feature_payload["user_request"]),
    )

    with pytest.raises(ValueError, match="state changed"):
        graph.invoke(invalid_state)
    with pytest.raises(ValueError, match="unrelated workflow"):
        build_graph(
            inputs.store,
            planning_continuation_mode=inputs.request,
            use_default_proposal_git_binding=False,
            patch_change_factory=lambda *_args: (),
        )


def test_planning_continuation_graph_rechecks_source_before_entry(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    graph = build_graph(
        inputs.store,
        planning_continuation_mode=inputs.request,
        use_default_proposal_git_binding=False,
    )
    state = _direct_graph_state(inputs)
    inputs.behavior.write_text("changed before direct graph entry\n", encoding="utf-8")

    with pytest.raises(PlanningContinuationError) as caught:
        graph.invoke(state)

    assert caught.value.code == "PLANNING_CONTINUATION_SOURCE_HASH_MISMATCH"
    assert caught.value.stage == "source_validation"
    assert {
        ref.artifact_type
        for ref in inputs.store.list_feature_artifacts(inputs.feature_ref.feature_id)
    } == {"feature_spec", "evidence_pack"}


def test_planning_continuation_graph_has_only_planning_reachable(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    graph = build_graph(
        inputs.store,
        planning_continuation_mode=inputs.request,
        use_default_proposal_git_binding=False,
    ).get_graph()

    assert {(edge.source, edge.target, edge.conditional) for edge in graph.edges} == {
        ("__start__", "create_implementation_plan", True),
        ("create_implementation_plan", "__end__", True),
    }


def test_exact_plan_runs_one_local_patch_review_and_no_write_dry_run(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(inputs)
    source_before = _source_hashes(inputs.workspace)

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )
    public = result.to_dict()

    assert result.status == "terminal"
    assert result.terminal_stage == "dry_run_apply"
    assert result.first_failure is None
    assert llm.calls == 1
    assert public["execution_counts"] == {
        "fresh_coverage": 0,
        "model": 1,
        "planning": 0,
        "patch": 1,
        "review": 1,
        "dry_run": 1,
        "apply": 0,
        "build": 0,
        "runtime": 0,
    }
    assert public["stopped_before"] == "exact_diff_approval"
    assert public["product_source_writes_performed"] is False
    assert public["source_modified_during_run"] is False
    assert _source_hashes(inputs.workspace) == source_before
    patch_ref = ArtifactRef(**public["patch_proposal_artifact"])
    patch = inputs.store.read(patch_ref)
    assert [
        item["artifact_type"]
        for item in patch["inputs"]["upstream_artifacts"]
    ] == ["implementation_plan", "evidence_pack"]
    assert all(
        item["require_latest"] is False
        for item in patch["inputs"]["upstream_artifacts"]
    )
    assert patch["inputs"]["upstream_artifacts"][0]["artifact_hash"] == (
        request.implementation_plan_ref.artifact_hash
    )
    assert patch["inputs"]["upstream_artifacts"][1]["artifact_hash"] == (
        request.evidence_pack_ref.artifact_hash
    )
    review = inputs.store.read(ArtifactRef(**public["proposal_review_artifact"]))
    assert review["data"]["allowed_next_stage"] == "dry_run_apply"
    dry_run = inputs.store.read(ArtifactRef(**public["dry_run_artifact"]))
    assert dry_run["data"]["execution_mode"] == "dry_run"
    assert dry_run["data"]["actual_write_performed"] is False
    receipt_ref = public["attempt_receipt"]
    receipt_path = Path(receipt_ref["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_ref["model_call_count"] == 1
    assert receipt["factory_receipt"]["model_response"]["output_sha256"]
    response_receipt = receipt["factory_receipt"]["model_response"]
    assert response_receipt["retained"] is True
    response_bytes = Path(response_receipt["path"]).read_bytes()
    assert response_bytes == llm.last_content.encode("utf-8")
    assert b"\r\n" in response_bytes
    assert "精确".encode() in response_bytes
    assert response_receipt["encoding"] == "utf-8"
    assert response_receipt["output_sha256"] == hashlib.sha256(response_bytes).hexdigest()
    assert response_receipt["output_utf8_bytes"] == len(response_bytes)
    response_contract = receipt["factory_receipt"]["patch_response_contract"]
    assert response_contract == {
        "contract_id": "patch_generation",
            "contract_version": "3",
        "schema_sha256": receipt["factory_receipt"]["provider_request"][
            "response_format_sha256"
        ],
    }
    generation = patch["data"]["diff_intent"][0]["generation"]
    assert generation["patch_response_contract"] == response_contract
    assert generation["model_input_sha256"] == receipt["factory_receipt"][
        "model_input_sha256"
    ]
    assert generation["model_response_sha256"] == response_receipt[
        "output_sha256"
    ]
    assert receipt["proposal_admitted"] is True
    assert receipt["generated_against_snapshot_sha256"] == receipt[
        "current_snapshot_sha256"
    ]


def test_exact_plan_runs_bounded_patch_agent_loop_before_dry_run(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    source_before = _source_hashes(inputs.workspace)
    identity = {
        "name": "qwen3:4b-instruct",
        "model": "qwen3:4b-instruct",
        "digest": "a" * 64,
        "modified_at": "2026-09-04T00:00:00Z",
        "size": 1,
        "details": {"context_length": 8192},
        "capabilities": ["completion", "tools"],
        "base_url": "http://127.0.0.1:11434",
    }
    unused_materializer_llm = _NativePatchLLM()
    agent_client = _PatchAgentTurnClient()
    adapter = LocalOllamaInitialPatchAdapter(
        profile=LocalOllamaInitialPatchProfileV1(
            profile_version=LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
            base_url="http://127.0.0.1:11434",
            model="qwen3:4b-instruct",
            context_window=8192,
            output_budget=1024,
            timeout_seconds=30.0,
        ),
        artifact_store=inputs.store,
        workspace_root=inputs.workspace,
        patch_agent_loop_policy=PatchAgentLoopPolicy(),
        identity_reader=lambda _base_url, _model: deepcopy(identity),
        llm_factory=lambda _profile: unused_materializer_llm,
        agent_turn_client_factory=lambda _profile: agent_client,
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
        max_model_calls=5,
    )

    assert result.status == "terminal"
    assert result.execution_counts["model"] == 2
    assert len(agent_client.requests) == 2
    assert agent_client.requests[0].tool_choice == "required"
    assert agent_client.requests[1].tool_results[0].output[
        "source_ref"
    ] == "binding://edit_binding_001"
    assert unused_materializer_llm.calls == 0
    assert _source_hashes(inputs.workspace) == source_before
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    loop = receipt["factory_receipt"]["patch_agent_loop"]
    assert loop["status"] == "completed"
    projection = receipt["factory_receipt"]["patch_agent_input_projection"]
    initial_messages = [
        {"role": item.role, "content": item.content}
        for item in agent_client.requests[0].messages[:-1]
    ]
    assert projection["initial_messages_sha256"] == hashlib.sha256(
        json.dumps(initial_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert projection["native_prepared_messages_sha256"] != projection["initial_messages_sha256"]
    assert projection["encoding_version"] in {
        "patch_agent_context_tables_v1", "patch_ref_first_bounded_excerpt_v1"
    }
    assert loop["model_turns_used"] == 2
    assert loop["tool_rounds_used"] == 1
    assert loop["observation_trace"][0]["admitted_to_model"] is True
    proposal = inputs.store.read(ArtifactRef(**result.patch_proposal_artifact))
    generation = proposal["data"]["diff_intent"][0]["generation"]
    assert generation["patch_agent_loop"]["enabled"] is True
    assert generation["patch_agent_loop"]["observation_refs"][0][
        "binding_id"
    ] == "edit_binding_001"


def test_patch_agent_runs_one_validator_correction_and_keeps_both_responses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    identity = {
        "name": "qwen3:4b-instruct",
        "model": "qwen3:4b-instruct",
        "digest": "a" * 64,
        "modified_at": "2026-09-04T00:00:00Z",
        "size": 1,
        "details": {"context_length": 8192},
        "capabilities": ["completion", "tools"],
        "base_url": "http://127.0.0.1:11434",
    }
    client = _PatchAgentTurnClient()
    adapter = LocalOllamaInitialPatchAdapter(
        profile=LocalOllamaInitialPatchProfileV1(
            profile_version=LOCAL_OLLAMA_PATCH_PROFILE_VERSION,
            base_url="http://127.0.0.1:11434",
            model="qwen3:4b-instruct",
            context_window=8192,
            output_budget=1024,
            timeout_seconds=30.0,
        ),
        artifact_store=inputs.store,
        workspace_root=inputs.workspace,
        patch_agent_loop_policy=PatchAgentLoopPolicy(),
        identity_reader=lambda _base_url, _model: deepcopy(identity),
        llm_factory=lambda _profile: _NativePatchLLM(),
        agent_turn_client_factory=lambda _profile: client,
    )
    original_materialize = LLMExactPatchGenerator.materialize_prepared_native
    materialize_calls = 0

    def fail_once(self, prepared, response, **kwargs):
        nonlocal materialize_calls
        materialize_calls += 1
        if materialize_calls == 1:
            raise PatchGenerationError(
                "Change removes every existing C# using directive.",
                diagnostics={
                    "failure_classification": (
                        "csharp_file_header_usings_removed"
                    ),
                    "binding_id": "edit_binding_001",
                    "removed_using_count": 1,
                },
            )
        return original_materialize(
            self,
            prepared,
            response,
            **kwargs,
        )

    monkeypatch.setattr(
        LLMExactPatchGenerator,
        "materialize_prepared_native",
        fail_once,
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
        max_model_calls=5,
    )

    assert result.status == "terminal"
    assert result.execution_counts["model"] == 3
    assert len(client.requests) == 3
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    factory = receipt["factory_receipt"]
    assert len(factory["model_responses"]) == 2
    assert factory["model_responses"][0]["path"] != factory[
        "model_responses"
    ][1]["path"]
    correction = factory["patch_agent_loop"]["validator_correction"]
    assert correction["status"] == "completed"
    assert correction["attempt_count"] == 1
    assert correction["validator"]["failure_classification"] == (
        "csharp_file_header_usings_removed"
    )


def test_dry_run_failure_preserves_diagnostic_and_completed_artifacts(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(inputs)
    source_before = _source_hashes(inputs.workspace)

    def fail_dry_run(*_args):
        raise PatchApplyError(
            "Unified diff introduces trailing whitespace on 1 added line(s)."
        )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=fail_dry_run,
    )

    assert result.status == "partial"
    assert result.first_failure == {
        "code": "PatchApplyError",
        "stage": "dry_run_apply",
    }
    assert result.patch_proposal_artifact is not None
    assert result.proposal_review_artifact is not None
    assert result.dry_run_artifact is None
    assert result.reason == (
        "Unified diff introduces trailing whitespace on 1 added line(s)."
    )
    assert llm.calls == 1
    assert _source_hashes(inputs.workspace) == source_before
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    assert set(receipt["artifacts"]) == {
        "patch_proposal",
        "proposal_review",
    }
    assert receipt["failure"] == {
        "code": "PatchApplyError",
        "failure_class": "PatchApplyError",
        "message": (
            "Unified diff introduces trailing whitespace on 1 added line(s)."
        ),
        "stage": "dry_run_apply",
    }


def test_patch_attempt_receipt_writes_under_real_length_artifact_root(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    admitted = preflight_patch_continuation(request)
    target_root_length = 169
    component_length = target_root_length - len(str(tmp_path)) - 1
    assert component_length > 0
    long_artifact_root = tmp_path / ("r" * component_length)
    long_artifact_root.mkdir()
    assert len(str(long_artifact_root)) == target_root_length
    run_id = "patch_continuation_" + "1" * 32

    journal = PatchContinuationAttemptJournal(
        artifact_root=long_artifact_root,
        run_id=run_id,
        request_id="patch_request_" + "2" * 32,
        request=request,
        admitted=admitted,
    )

    assert journal.path.is_file()
    assert len(str(journal.path)) <= 220
    assert journal.path.resolve().is_relative_to(long_artifact_root.resolve())
    receipt = json.loads(journal.path.read_text(encoding="utf-8"))
    assert receipt["run_id"] == run_id
    other = PatchContinuationAttemptJournal(
        artifact_root=long_artifact_root,
        run_id="patch_continuation_" + "3" * 32,
        request_id="patch_request_" + "4" * 32,
        request=request,
        admitted=admitted,
    )
    assert other.path.name != journal.path.name
    response_content = '{\r\n  "changes": [],\r\n  "note": "完整"\r\n}'
    response_path = _local_patch_response_sidecar_path(
        long_artifact_root,
        run_id=run_id,
    )
    sidecar = _write_local_patch_response_sidecar(response_path, response_content)
    assert len(str(response_path)) <= 220
    assert response_path.resolve().is_relative_to(long_artifact_root.resolve())
    assert response_path.read_bytes() == response_content.encode("utf-8")
    assert sidecar["retained"] is True
    assert _local_patch_response_sidecar_path(
        long_artifact_root,
        run_id="patch_continuation_" + "3" * 32,
    ).name != response_path.name


def test_patch_response_sidecar_first_create_installs_exact_bytes(
    tmp_path: Path,
) -> None:
    destination = _local_patch_response_sidecar_path(
        tmp_path,
        run_id="patch_continuation_first_create",
    )
    content = "first response 精确\r\n"

    receipt = _write_local_patch_response_sidecar(destination, content)

    assert destination.read_bytes() == content.encode("utf-8")
    assert receipt["retained"] is True
    assert list(destination.parent.glob("*.tmp")) == []


def test_patch_response_sidecar_same_bytes_is_idempotent(
    tmp_path: Path,
) -> None:
    destination = _local_patch_response_sidecar_path(
        tmp_path,
        run_id="patch_continuation_idempotent",
    )
    content = "same response bytes\r\n"
    first = _write_local_patch_response_sidecar(destination, content)

    second = _write_local_patch_response_sidecar(destination, content)

    assert destination.read_bytes() == content.encode("utf-8")
    assert second == first
    assert list(destination.parent.glob("*.tmp")) == []


def test_patch_response_sidecar_different_bytes_rejects_without_overwrite(
    tmp_path: Path,
) -> None:
    destination = _local_patch_response_sidecar_path(
        tmp_path,
        run_id="patch_continuation_conflict",
    )
    original = b"immutable original response\r\n"
    _write_local_patch_response_sidecar(destination, original.decode("utf-8"))

    with pytest.raises(
        LocalEntryError,
        match="LOCAL_PATCH_RESPONSE_SIDECAR_CONTENT_CONFLICT",
    ):
        _write_local_patch_response_sidecar(destination, "different response\r\n")

    assert destination.read_bytes() == original
    assert list(destination.parent.glob("*.tmp")) == []


def test_patch_response_sidecar_write_failure_retains_safe_stage_diagnostics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = _local_patch_response_sidecar_path(
        tmp_path,
        run_id="patch_continuation_write_failure",
    )

    failure = PermissionError(13, "controlled failure")
    failure.winerror = 5
    monkeypatch.setattr(os, "link", lambda *_args: (_ for _ in ()).throw(failure))

    with pytest.raises(
        LocalEntryError,
        match="LOCAL_PATCH_RESPONSE_SIDECAR_WRITE_FAILED",
    ) as captured:
        _write_local_patch_response_sidecar(destination, "response")

    assert captured.value.diagnostics == {
        "failure_boundary": "local_patch_response_sidecar_write",
        "operation": "install_hardlink",
        "os_error_type": "PermissionError",
        "errno": 13,
        "winerror": 5,
        "destination_path_length": len(str(destination)),
        "temporary_path_length": captured.value.diagnostics["temporary_path_length"],
    }
    assert captured.value.diagnostics["temporary_path_length"] > len(str(destination.parent))
    assert not destination.exists()
    assert list(destination.parent.glob("*.tmp")) == []


def test_continue_fresh_patch_cli_runs_exact_no_write_vertical(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(inputs)
    monkeypatch.setattr(
        "src.mod_agent.cli.LocalOllamaInitialPatchAdapter",
        lambda **_kwargs: adapter,
    )

    exit_code = main(_patch_cli_args(request))

    captured = capsys.readouterr()
    public = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert public["result_type"] == "patch_continuation_result_v1"
    assert public["status"] == "terminal"
    assert public["stopped_before"] == "exact_diff_approval"
    assert public["execution_counts"]["model"] == 1
    assert public["execution_counts"]["planning"] == 0
    assert public["execution_counts"]["patch"] == 1
    assert public["execution_counts"]["review"] == 1
    assert public["execution_counts"]["dry_run"] == 1
    assert llm.calls == 1
    assert llm.last_content not in captured.out
    receipt_text = Path(public["attempt_receipt"]["path"]).read_text(encoding="utf-8")
    assert llm.last_content not in receipt_text


def test_exact_selected_plan_remains_admitted_when_not_latest(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    selected_plan = inputs.store.read(request.implementation_plan_ref)
    newer_payload = deepcopy(selected_plan["data"])
    newer_payload["message"] = "A later Plan alias that is not selected."
    newer_plan_ref = inputs.store.write(
        feature_id=request.implementation_plan_ref.feature_id,
        artifact_type="implementation_plan",
        payload=newer_payload,
        status="success",
        upstream_refs=[inputs.feature_ref, inputs.evidence_ref],
    )
    assert newer_plan_ref.artifact_hash != request.implementation_plan_ref.artifact_hash
    adapter, _ = _local_patch_adapter(inputs)

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "terminal"
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    assert receipt["latest_observations"]["implementation_plan"][
        "selected_is_latest"
    ] is False
    proposal = inputs.store.read(ArtifactRef(**result.patch_proposal_artifact))
    plan_upstream = next(
        item
        for item in proposal["inputs"]["upstream_artifacts"]
        if item["artifact_type"] == "implementation_plan"
    )
    assert plan_upstream["artifact_hash"] == request.implementation_plan_ref.artifact_hash
    assert plan_upstream["require_latest"] is False


def test_source_drift_after_model_keeps_receipt_and_writes_no_proposal(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(
        inputs,
        after_response=lambda: inputs.behavior.write_text(
            "changed concurrently after model response\n",
            encoding="utf-8",
        ),
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "partial"
    assert llm.calls == 1
    assert result.first_failure == {
        "code": "PATCH_CONTINUATION_SOURCE_HASH_MISMATCH",
        "stage": "post_call_admission",
    }
    assert result.patch_proposal_artifact is None
    assert result.proposal_review_artifact is None
    assert result.dry_run_artifact is None
    assert result.execution_counts["model"] == 1
    assert result.execution_counts["patch"] == 1
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    phases = [event["phase"] for event in receipt["events"]]
    assert phases.index("model_output_persisted_before_post_call_admission") < phases.index(
        "post_call_admission_rejected"
    )
    assert receipt["factory_receipt"]["model_response"]["output_sha256"]
    assert Path(receipt["factory_receipt"]["model_response"]["path"]).is_file()
    assert receipt["proposal_admitted"] is False
    assert receipt["generated_against_snapshot_sha256"] != receipt[
        "current_snapshot_sha256"
    ]


def test_exact_plan_ref_drift_after_model_keeps_receipt_and_no_downstream_artifact(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(
        inputs,
        after_response=lambda: Path(request.implementation_plan_ref.path).write_text(
            "{}\n",
            encoding="utf-8",
        ),
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "partial"
    assert result.first_failure == {
        "code": "PATCH_CONTINUATION_ARTIFACT_INVALID",
        "stage": "post_call_admission",
    }
    assert llm.calls == 1
    assert result.patch_proposal_artifact is None
    assert result.proposal_review_artifact is None
    assert result.dry_run_artifact is None
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    phases = [event["phase"] for event in receipt["events"]]
    assert phases.index("model_output_persisted_before_post_call_admission") < phases.index(
        "post_call_admission_rejected"
    )
    assert receipt["factory_receipt"]["model_response"]["output_sha256"]
    assert Path(receipt["factory_receipt"]["model_response"]["path"]).is_file()
    assert receipt["proposal_admitted"] is False


def test_malformed_native_response_keeps_full_sidecar_and_no_proposal(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    malformed = '{"changes": [完整'
    adapter, llm = _local_patch_adapter(
        inputs,
        content_override=malformed,
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "partial"
    assert llm.calls == 1
    assert result.patch_proposal_artifact is None
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    response_receipt = receipt["factory_receipt"]["model_response"]
    assert response_receipt["retained"] is True
    assert Path(response_receipt["path"]).read_bytes() == malformed.encode("utf-8")


def test_legacy_pair_mismatch_response_is_retained_but_not_materialized(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    legacy_mismatch = json.dumps(
        {
            "changes": [
                {
                    "change_id": "wrong_change",
                    "edit_binding_id": "edit_binding_001",
                    "rationale": "old pair response",
                    "rollback_hint": "restore",
                    "after_text": "changed",
                }
            ]
        },
        ensure_ascii=False,
    )
    adapter, _ = _local_patch_adapter(
        inputs,
        content_override=legacy_mismatch,
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "partial"
    assert result.patch_proposal_artifact is None
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    factory_receipt = receipt["factory_receipt"]
    assert factory_receipt["failure"]["diagnostics"]["failure_classification"] == (
        "native_response_fields_invalid"
    )
    response_receipt = factory_receipt["model_response"]
    assert response_receipt["retained"] is True
    assert Path(response_receipt["path"]).read_bytes() == legacy_mismatch.encode("utf-8")


def test_sidecar_write_failure_blocks_before_parse_and_proposal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(inputs)
    monkeypatch.setattr(
        "src.mod_agent.local_entry._write_local_patch_response_sidecar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LocalEntryError("LOCAL_PATCH_RESPONSE_SIDECAR_WRITE_FAILED")
        ),
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "partial"
    assert llm.calls == 1
    assert result.patch_proposal_artifact is None
    receipt = json.loads(
        Path(result.attempt_receipt["path"]).read_text(encoding="utf-8")
    )
    assert receipt["factory_receipt"]["model_response"] is None
    assert receipt["factory_receipt"]["failure"]["message"] == (
        "LOCAL_PATCH_RESPONSE_SIDECAR_WRITE_FAILED"
    )


def test_pre_call_snapshot_race_blocks_with_zero_model_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(inputs)
    monkeypatch.setattr(
        "src.mod_agent.workflow.graph._patch_continuation_current_snapshot_sha256",
        lambda _mode: "f" * 64,
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "blocked"
    assert result.first_failure == {
        "code": "PATCH_CONTINUATION_SOURCE_CHANGED_BEFORE_MODEL",
        "stage": "pre_call_admission",
    }
    assert result.execution_counts["model"] == 0
    assert llm.calls == 0
    assert result.patch_proposal_artifact is None


def test_patch_continuation_rejects_non_exact_model_call_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, llm = _local_patch_adapter(inputs)
    monkeypatch.setattr(
        adapter,
        "patch_attempt_receipt",
        lambda: {"model_call_count": 2, "model_response": {"output_sha256": "a" * 64}},
    )

    result = run_fresh_patch_continuation(
        request,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
    )

    assert result.status == "partial"
    assert result.first_failure == {
        "code": "PATCH_CONTINUATION_MODEL_CALL_COUNT_INVALID",
        "stage": "post_call_admission",
    }
    assert llm.calls == 1
    assert result.patch_proposal_artifact is None


def test_patch_continuation_graph_has_only_patch_review_dry_run_reachable(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    adapter, _ = _local_patch_adapter(inputs)
    graph = build_graph(
        inputs.store,
        patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace),
        use_default_proposal_git_binding=False,
        proposal_source_snippet_char_limit=10_000,
        patch_continuation_mode=request,
        patch_continuation_attempt_recorder=lambda _event: None,
    ).get_graph()

    assert {(edge.source, edge.target, edge.conditional) for edge in graph.edges} == {
        ("__start__", "generate_patch_proposal", True),
        ("generate_patch_proposal", "review_proposal", True),
        ("generate_patch_proposal", "__end__", True),
        ("review_proposal", "dry_run_apply", True),
        ("review_proposal", "__end__", True),
        ("dry_run_apply", "__end__", True),
    }


def test_human_assisted_continue_fresh_planning_cli_selects_targets_and_stops(
    tmp_path: Path,
    capsys,
) -> None:
    inputs = _inputs(tmp_path)

    exit_code = main(_cli_args(inputs))

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert result["result_type"] == "planning_continuation_result_v1"
    assert result["status"] == "terminal"
    assert result["stopped_before"] == "patch_proposal"
    assert result["execution_counts"]["fresh_coverage"] == 0
    assert result["execution_counts"]["model"] == 0
    assert result["execution_counts"]["planning"] == 1
    assert result["execution_counts"]["patch"] == 0
    plan = inputs.store.read(ArtifactRef(**result["implementation_plan_artifact"]))
    assert {
        binding["evidence_id"]
        for binding in plan["data"]["binding_registry"]["edit_bindings"]
    } == set(inputs.request.planning_target_evidence_ids)


def test_continue_fresh_planning_cli_returns_one_blocked_json(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)

    def blocked_build(_self, *, feature_spec, **_kwargs):
        return ImplementationPlan(
            feature_id=feature_spec.feature_id,
            target_files=(),
            target_symbols=(),
            evidence_refs=(),
            proposed_changes=(),
            side_effect_boundaries=(),
            test_plan=(),
            rollback_plan=(),
            open_questions=("Synthetic planning blocker.",),
            blocking=True,
            blockers=(
                {
                    "code": "SYNTHETIC_PLANNING_BLOCKER",
                    "message": "Synthetic planning blocker.",
                },
            ),
        )

    monkeypatch.setattr(ImplementationPlanner, "build", blocked_build)

    exit_code = main(_cli_args(inputs))

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 1
    assert captured.err == ""
    assert result["status"] == "blocked"
    assert result["first_failure"] == {
        "code": "SYNTHETIC_PLANNING_BLOCKER",
        "stage": "implementation_plan",
    }
    assert result["implementation_plan_artifact"] is not None


def test_continue_fresh_planning_cli_admission_failure_is_terminal_json(
    tmp_path: Path,
    capsys,
) -> None:
    inputs = _inputs(tmp_path)
    args = _cli_args(inputs)
    args[args.index("--evidence-pack-hash") + 1] = "not-a-sha256"

    exit_code = main(args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 2
    assert captured.err == ""
    assert result["status"] == "blocked"
    assert result["first_failure"] == {
        "code": "PLANNING_CONTINUATION_REF_INVALID",
        "stage": "request",
    }
    assert result["implementation_plan_artifact"] is None


def test_continue_fresh_planning_cli_reports_partial_store_write(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    inputs = _inputs(tmp_path)
    original_write = ArtifactStore._write_text_atomic

    def fail_plan_latest(
        store: ArtifactStore,
        path: Path,
        content: str,
        *,
        capture_replace_error: bool = False,
    ) -> None:
        latest_plan_path = store._artifact_path(
            inputs.feature_ref.feature_id,
            "implementation_plan",
        )
        if path == latest_plan_path:
            raise OSError("injected latest ImplementationPlan write failure")
        original_write(
            store,
            path,
            content,
            capture_replace_error=capture_replace_error,
        )

    monkeypatch.setattr(ArtifactStore, "_write_text_atomic", fail_plan_latest)

    exit_code = main(_cli_args(inputs))

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert exit_code == 2
    assert captured.err == ""
    assert result["status"] == "partial"
    assert result["first_failure"] == {
        "code": "PLANNING_CONTINUATION_PLAN_STORE_PARTIAL",
        "stage": "implementation_plan_store",
    }
    assert result["artifact_write"]["outcome"] == "partial"
    assert result["artifact_write"]["commit_started"] is True
    assert result["artifact_write"]["revision_state"] == "valid"
    assert result["artifact_write"]["pending_marker_state"] == "present"
    assert result["artifact_write"]["failure_stage"] == "latest_alias_write"
    assert result["artifact_write"]["recovery_available"] is True
    assert result["execution_counts"]["planning"] == 1
    assert result["implementation_plan_artifact"] is not None
    assert Path(result["implementation_plan_artifact"]["path"]).is_file()
    assert not inputs.store._artifact_path(
        inputs.feature_ref.feature_id,
        "implementation_plan",
    ).exists()
