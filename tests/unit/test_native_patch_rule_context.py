"""Compact fixed rules preserve native data and the existing authority contracts.

The policy allowance comes from the retained developer request with SHA-256
94f571639f7d13d751e521ba0f994d230a168b486953846d6c2f66314c75aa12.
That private request stays outside repository fixtures; only public owned source
definitions and synthetic observations are used here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import prepare_patch_sft_data as data
from src.libs.llm import ChatResponse
from src.mod_agent.local_entry import (
    LocalOllamaInitialPatchAdapter,
    _patch_agent_initial_messages,
    _restore_patch_agent_context,
)
from src.mod_agent.proposal import LLMExactPatchGenerator, PatchGenerationError
from src.mod_agent.proposal.code_generator import _patch_system_prompt
from src.mod_agent.proposal.patch_agent_loop import (
    PatchAgentLoopPolicy,
    PlanBoundSourceInspector,
    _source_tool_definition,
)
from src.mod_agent.proposal.working_view import build_materialized_patch_view
from tests.unit.test_patch_code_generator import _large_projection_fixture


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Native rule tests must not use a real model or network.")
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)


@pytest.mark.parametrize("case_id", ["paging_partial_tail", "snapshot_nested_tags"])
def test_owned_native_context_keeps_source_failed_code_and_full_view(tmp_path, case_id):
    definition = next(row for row in data.load_tasks().TASKS if row["case_id"] == case_id)
    workspace, plan, generator, prepared, _ = data.prepare_contract(definition, tmp_path / "case")
    raw, generated, materialized = data.structured_output(
        definition, "C1", tmp_path / "case", plan, generator, prepared, workspace)
    initial = _patch_agent_initial_messages(prepared.messages)
    projected = _restore_patch_agent_context(json.loads(initial[1].content))
    projected.pop("patch_agent_context_policy")
    assert projected == json.loads(prepared.messages[1].content)
    policy = PatchAgentLoopPolicy(require_source_inspection=False, finalize_after_first_observation=False,
        enable_supporting_reads=True, allow_batched_reads=True, reuse_covered_source_observations=True,
        max_tool_rounds=6, max_total_model_turns=5)
    inspector = PlanBoundSourceInspector(workspace_root=workspace, implementation_plan=plan, policy=policy)
    tool = _source_tool_definition(inspector)
    assert initial[0].content.startswith(prepared.messages[0].content)
    assert initial[0].content.count("You may add internal declarations, state and logic") == 1
    assert tool.parameters == {"type": "object", "additionalProperties": False,
        "required": ["binding_id"], "properties": {"binding_id": {
            "type": "string", "enum": list(inspector.available_read_binding_ids)}}}
    assert tool.strict is True
    view = build_materialized_patch_view(materialized, context_radius_lines=policy.context_radius_lines)
    feedback = {"status": "behavior_failed", "generated_changes_sha256": "a" * 64,
        "observations": [{"check_id": "offline_input_preserved", "input": {"case_id": case_id,
            "initial_state": {"source": "owned fixture"}, "operations": [{"step": 1, "action": "fixture"}]},
            "expected": "original expectation", "observed": "original failed observation"}],
        "working_code_view": view}
    before = deepcopy(feedback)
    recovery = LocalOllamaInitialPatchAdapter._revision_messages(raw, feedback, message_type="patch_repair_context")
    assert recovery[0].content == raw and recovery[1].content["validator_feedback"] == before
    assert feedback == before
    assert materialized.files["OwnedTraining/Subject.cs"] == definition["sources"]["C1"].encode()
    assert all(file["complete"] and file["unified_diff"] and file["hunks"] for file in view["files"])
    assert Path(workspace, "OwnedTraining/Subject.cs").read_bytes() == definition["sources"]["B0"].encode()
    assert generator.materialize_prepared_native(prepared, ChatResponse(content=raw, model="offline")).changes == generated.changes


def test_multiple_bindings_keep_native_schema_and_reject_added_authority():
    fixture = _large_projection_fixture()
    generator = LLMExactPatchGenerator(None)
    prepared = generator.prepare_native_contract(
        user_request=fixture["user_request"], evidence_items=fixture["evidence_items"],
        binding_registry=fixture["binding_registry"], constraints=fixture["constraints"],
        required_change_units=fixture["required_change_units"],
        required_validation_by_binding=fixture["required_validation_by_binding"])
    expected_ids = [row["edit_binding_id"] for row in fixture["required_change_units"]]
    assert len(expected_ids) > 1
    schema = prepared.response_schema["properties"]["changes"]
    assert schema["minItems"] == schema["maxItems"] == len(expected_ids)
    assert schema["items"]["required"] == ["edit_binding_id", "rationale", "rollback_hint", "after_text"]
    assert schema["items"]["additionalProperties"] is False
    result = generator.materialize_prepared_native(prepared,
        ChatResponse(content=json.dumps(fixture["response"]), model="offline"))
    assert [row["edit_binding_id"] for row in result.changes] == expected_ids
    invalid = deepcopy(fixture["response"])
    invalid["changes"][0]["path"] = "outside-the-authorized-binding.cs"
    with pytest.raises(PatchGenerationError) as raised:
        generator.materialize_prepared_native(prepared, ChatResponse(content=json.dumps(invalid), model="offline"))
    assert raised.value.diagnostics["failure_classification"] == "native_response_fields_invalid"


@pytest.mark.parametrize("native,plan_bound", [(False, False), (False, True), (True, False), (True, True)])
def test_shared_prompt_allows_required_internal_implementation_without_new_authority(native, plan_bound):
    text = _patch_system_prompt(require_native_schema=native, plan_bound_response=plan_bound)
    assert "Implement the complete request with exact-text edits." in text
    assert "Minimal means avoiding unrelated changes, not omitting required behavior or necessary state." in text
    assert "Do not fabricate existing files, binding ids, existing symbols, or external dependency APIs." in text
    assert "You may add internal declarations, state and logic inside authorized edit bindings" in text
    assert "preserve public interfaces required by the user" in text
    assert "This grants no additional paths or edit authority." in text
    assert "Never invent paths/symbols/APIs/bindings" not in text
    assert "Never invent a file path, symbol, API" not in text
    assert "supporting_evidence is read-only" in text
    assert "required_change_units" in text and "edit_binding_id" in text
    assert "verified_member_accesses" in text
    if native:
        assert "Native v3: optional before_text" in text
        assert "Omit Plan/binding-derived" in text
    else:
        assert "preserve its exact change_id and edit_binding_id" in text
        assert "before_text is copied verbatim" in text


@pytest.mark.parametrize("reuse,deduplicate", [(False, False), (False, True), (True, False)])
def test_source_description_retains_policy_specific_read_costs(reuse, deduplicate):
    from types import SimpleNamespace

    policy = PatchAgentLoopPolicy(reuse_covered_source_observations=reuse,
        deduplicate_source_excerpts=deduplicate)
    tool = _source_tool_definition(SimpleNamespace(policy=policy, available_read_binding_ids=("bound_a",)))
    description = tool.description
    # The declared read scope agrees with its executable argument surface.
    assert tool.strict and tool.parameters["additionalProperties"] is False
    assert tool.parameters["required"] == ["binding_id"]
    assert set(tool.parameters["properties"]) == {"binding_id"}
    assert tool.parameters["properties"]["binding_id"]["enum"] == ["bound_a"]
    assert "no Evidence/Coverage or Plan reassessment" in description
    assert "B0 (no generated edits)" in description and "no paths" in description
    assert "assistant Patch" in description and "latest failure feedback" in description
    assert "Supporting reads grant no edits" in description
    assert "API bindings" in description and "full saved Evidence" in description
    assert "not live inspection" in description and "Sequential calls" in description
    if reuse:
        assert "New windows cost 1 read" in description
        assert "visible refs cost 0" in description and "consume bytes" in description
    else:
        assert "dispatched binding read costs one read" in description
        assert "including in a batch" in description
        assert "visible refs cost 0" not in description
    assert ("excerpt_ref" in description) is (reuse or deduplicate)
    assert ("source_excerpt_span" in description) is reuse
    assert ("excerpt_parts" in description) is reuse
    assert "target_span" in description and "keep full text" in description


@pytest.mark.parametrize("modules", [
    ("src.mod_agent.local_entry", "src.mod_agent.proposal.code_generator", "src.mod_agent.proposal.patch_agent_loop"),
    ("src.mod_agent.proposal.code_generator", "src.mod_agent.proposal.patch_agent_loop", "src.mod_agent.local_entry"),
])
def test_rule_owners_import_in_fresh_processes(modules):
    program = "\n".join("import " + name for name in modules)
    result = subprocess.run([sys.executable, "-B", "-c", program],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
