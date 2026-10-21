"""Reuse exact visible failed response text, never a hash-only or changed version."""

import hashlib

from src.libs.llm import AgentMessage
from src.mod_agent.local_entry import LocalOllamaInitialPatchAdapter


def test_revision_reuses_exact_visible_patch_and_preserves_new_failure():
    raw = '{"changes":[{"after_text":"measured failing code"}]}'
    visible = (AgentMessage(role="user", content="Preserve independent behavior."),
               AgentMessage(role="assistant", content=raw))
    feedback = {"observations": [{"expected": 30, "observed": 10}]}
    messages = LocalOllamaInitialPatchAdapter._revision_messages(
        raw, feedback, message_type="patch_behavior_correction", visible_messages=visible,
    )
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content["validator_feedback"] == feedback
    assert messages[0].content["previous_response_ref"]["content_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert "earlier complete assistant Patch" in messages[0].content["required_action"]
    assert messages[0].content["revision_semantics"] == "complete_replacement_relative_to_B0"


def test_changed_or_absent_visible_response_must_include_full_current_code():
    raw = '{"changes":[{"after_text":"current code"}]}'
    for visible in ((), (AgentMessage(role="assistant", content=raw + " "),),
                    (AgentMessage(role="user", content=raw),)):
        messages = LocalOllamaInitialPatchAdapter._revision_messages(
            raw, {"observations": []}, message_type="patch_behavior_correction", visible_messages=visible,
        )
        assert len(messages) == 2
        assert messages[0].role == "assistant" and messages[0].content == raw
        assert "previous_response_ref" not in messages[1].content



def test_recovery_uses_same_code_refs_when_source_read_cache_is_disabled(monkeypatch):
    import json
    from dataclasses import asdict
    from types import SimpleNamespace
    import pytest
    from src.libs.llm.agent_turn import AgentTurnRequest
    from src.mod_agent import local_entry as owner
    from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
    from src.mod_agent.proposal.behavior_feedback_view import restore_visible_patch_response
    code = "class Example { }\n" * 160
    earlier = json.dumps({"changes": [{"edit_binding_id": "b", "after_text": code, "rationale": "prior"}]})
    latest = json.dumps({"changes": [{"edit_binding_id": "b", "after_text": code, "rationale": "latest"}]})
    prior = AgentTurnRequest(messages=(AgentMessage("user", "All original requirements"), AgentMessage("assistant", earlier)))
    adapter = LocalOllamaInitialPatchAdapter.__new__(LocalOllamaInitialPatchAdapter)
    adapter.patch_agent_loop_policy = PatchAgentLoopPolicy(reuse_covered_source_observations=False)
    adapter._repair_input_projection = None
    adapter._worker_response_admission = None
    adapter._tool_argument_correction_used = False
    adapter._prior_worker_budget = {"consumed_observation_bytes": 0, "consumed_model_turns": 0, "consumed_tool_rounds": 0}
    adapter._receipt = {"model_call_count": 1, "patch_agent_loop": {}}
    adapter._recovery_observation_bytes = 0
    adapter._recovery_observation_usage = lambda: (0, 0)
    adapter._upstream_effects = {"evidence": "assess_and_continue", "planning": "assess_and_continue"}
    adapter._emit = lambda *_: None
    captured = {}
    def loop(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(completed=False, final_content=None, model_turns_used=0,
            final_request=kwargs["resume_request"], termination={"code": "FIXTURE_STOP_BEFORE_PROVIDER"},
                tool_argument_correction_used=False,
            to_dict=lambda: {"usage_by_turn": []})
    monkeypatch.setattr(owner, "run_patch_agent_loop", loop)
    feedback = {"observations": [{"expected": [], "observed": ["unexpected"]}]}
    estimated_bytes = adapter._revision_payload_bytes(latest, feedback, prior)
    correction = {}
    with pytest.raises(owner.LocalEntryError, match="FIXTURE_STOP_BEFORE_PROVIDER"):
        adapter._run_recovery_revision(previous_response=latest, feedback=feedback,
            message_type="patch_behavior_correction", generation_request=prior,
            inspector=None, agent_client=None, response_format=None,
            correction_state=correction, run_id="fixture", response_label="fixture")
    additions = captured["resume_request"].messages[len(prior.messages):]
    assert restore_visible_patch_response(additions[0].content, prior.messages) == json.loads(latest)
    assert additions[1].content["validator_feedback"] == feedback
    assert captured["resume_request"].messages[:len(prior.messages)] == prior.messages
    assert captured["policy"].reuse_covered_source_observations is False
    assert captured["consumed_tool_rounds"] == 0
    actual_bytes = len(owner._patch_agent_context_json([asdict(row) for row in additions]).encode())
    assert actual_bytes == estimated_bytes == correction["feedback_budget"]["feedback_utf8_bytes"]
    assert adapter._receipt["model_call_count"] == 1
