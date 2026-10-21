"""Real-length Windows ledger paths, legacy replay and truncated-identity conflicts."""
import json
from uuid import uuid4

import pytest

from src.mod_agent.planning.clarification_state import ClarificationState, digest, reference, save
from src.mod_agent.planning import clarification_state as state_module


def state(workspace, *, conversation="conversation", turn="turn", clarification_for=None, description="request"):
    return ClarificationState(workspace=workspace, conversation_id=conversation, turn_id=turn,
        clarification_for=clarification_for, request={"description": description})


def route_records(workspace, conversation, turn):
    root = workspace / ".dsh-routing" / ("route_" + uuid4().hex)
    root.mkdir(parents=True)
    save(root / "request.json", {"conversation_id": conversation, "turn_id": turn})
    result = {"conversation_id": conversation, "turn_id": turn, "route_id": root.name,
              "status": "needs_clarification", "question": "Which project?", "preview_only": False,
              "routing_record_path": str(root / "result.json"), "routing_model_calls_reserved": 0,
              "routing_elapsed_seconds": 0}
    save(root / "result.json", result)
    save(root / "conversation_receipt.json", {"request_ref": reference(root / "request.json"),
                                              "result_ref": reference(root / "result.json")})
    return root, result


def record(selected, *, already_begun=False):
    if not already_begun:
        assert selected.begin(None) is None
    root, result = route_records(selected.workspace, selected.conversation_id, selected.turn_id)
    selected.claim(root, False)
    selected.close()
    return result


def legacy(workspace):
    root = workspace / ".dsh-routing/conversations" / digest("conversation")
    root.mkdir(parents=True)
    run, result = route_records(workspace, "conversation", "old")
    claim_path = root / (digest("old") + ".json")
    save(claim_path, {"turn_id": "old", "request_hash": digest({"description": "request", "clarification_for": None}),
        "created_ns": 1, "route_id": run.name, "result_path": str(run / "result.json"),
        "clarification_for": None, "preview_only": False})
    return root, claim_path, result


def test_actual_long_workspace_uses_short_components_and_replays_without_new_claim(tmp_path):
    padding = max(1, 129 - len(str(tmp_path)) - 1)
    workspace = tmp_path / ("w" * padding)
    workspace.mkdir()
    selected = state(workspace)
    hypothetical_old = workspace / ".dsh-routing/conversations" / digest("conversation") / (digest("turn") + ".json")
    assert len(str(hypothetical_old)) > 260
    assert len(str(selected.turn)) <= 259 and len(selected.root.name) == len(selected.turn.stem) == 32
    result = record(selected)
    claim_bytes = selected.turn.read_bytes()
    claim = json.loads(claim_bytes)
    assert claim["conversation_sha256"] == digest("conversation") and claim["turn_sha256"] == digest("turn")
    assert claim["conversation_id"] == "conversation" and claim["turn_id"] == "turn"
    again = state(workspace)
    assert again.begin(None) == result and not again.locked
    assert again.turn.read_bytes() == claim_bytes


def test_old_full_length_claim_replays_and_new_turn_uses_short_name_without_migration(short_workspace):
    root, old_claim, result = legacy(short_workspace)
    before = {path: path.read_bytes() for path in short_workspace.rglob("*") if path.is_file()}
    previous = state(short_workspace, turn="old")
    assert previous.root == root and previous.turn == old_claim
    assert previous.begin(None) == result and not previous.locked
    next_turn = state(short_workspace, turn="new", clarification_for=result["route_id"], description="answer")
    assert next_turn.root == root and len(next_turn.turn.stem) == 32
    assert next_turn.begin(result["route_id"]) is None
    assert next_turn.parent == result
    record(next_turn, already_begun=True)
    assert not (root.parent / digest("conversation")[:32]).exists()
    assert not (root / "session.identity").exists()
    assert all(path.read_bytes() == original for path, original in before.items())


def test_legacy_full_directory_can_continue_at_the_reported_workspace_length(tmp_path):
    workspace = tmp_path / ("w" * max(1, 129 - len(str(tmp_path)) - 1))
    workspace.mkdir()
    root = workspace / ".dsh-routing/conversations" / digest("conversation")
    root.mkdir(parents=True)
    selected = state(workspace)
    assert selected.root == root and len(str(selected.turn)) <= 259
    record(selected)
    assert selected.turn.is_file()


def test_both_conversation_layouts_are_rejected_without_moving_either(tmp_path):
    selected = state(tmp_path)
    legacy_root = selected.root.parent / digest("conversation")
    legacy_root.mkdir()
    with pytest.raises(ValueError, match="CONVERSATION_LAYOUT_CONFLICT"):
        state(tmp_path)
    assert selected.root.is_dir() and legacy_root.is_dir()


@pytest.mark.parametrize("field", ["conversation_id", "conversation_sha256"])
def test_truncated_conversation_identity_conflicts_are_explicit(tmp_path, field):
    selected = state(tmp_path)
    identity = json.loads(selected.identity.read_bytes())
    identity[field] = "different"
    selected.identity.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="CONVERSATION_IDENTITY_CONFLICT"):
        state(tmp_path)


def test_short_directory_without_identity_is_not_assumed_to_be_our_session(tmp_path):
    (tmp_path / ".dsh-routing/conversations" / digest("conversation")[:32]).mkdir(parents=True)
    with pytest.raises(ValueError, match="CONVERSATION_IDENTITY_CONFLICT"):
        state(tmp_path)


def test_both_turn_layouts_are_rejected_before_replay(tmp_path):
    selected = state(tmp_path)
    record(selected)
    other = selected.root / (digest("turn") + ".json")
    other.write_bytes(selected.turn.read_bytes())
    with pytest.raises(ValueError, match="TURN_LAYOUT_CONFLICT"):
        state(tmp_path)


@pytest.mark.parametrize("field", ["turn_id", "turn_sha256", "conversation_id", "conversation_sha256"])
def test_short_turn_full_identity_must_match_its_directory_and_name(tmp_path, field):
    selected = state(tmp_path)
    record(selected)
    claim = json.loads(selected.turn.read_bytes())
    claim[field] = "another"
    selected.turn.write_text(json.dumps(claim), encoding="utf-8")
    with pytest.raises(ValueError, match="TURN_IDENTITY_CONFLICT"):
        state(tmp_path).begin(None)


def test_collision_in_history_fails_before_creating_another_turn(tmp_path):
    selected = state(tmp_path)
    record(selected)
    claim = json.loads(selected.turn.read_bytes())
    claim["turn_sha256"] = "different"
    selected.turn.write_text(json.dumps(claim), encoding="utf-8")
    subsequent = state(tmp_path, turn="new")
    with pytest.raises(ValueError, match="TURN_IDENTITY_CONFLICT"):
        subsequent.begin(None)
    assert not subsequent.turn.exists() and not subsequent.lock.exists()


@pytest.mark.parametrize("identity", ["conversation", "turn"])
def test_distinct_full_hashes_with_same_short_prefix_cannot_replay_each_other(tmp_path, monkeypatch, identity):
    original_digest = state_module.digest
    def collided_digest(value):
        if value == "collision_a":
            return "a" * 32 + "0" * 32
        if value == "collision_b":
            return "a" * 32 + "1" * 32
        return original_digest(value)
    monkeypatch.setattr(state_module, "digest", collided_digest)
    first = state(tmp_path, **{identity: "collision_a"})
    record(first)
    with pytest.raises(ValueError, match="IDENTITY_CONFLICT"):
        second = state(tmp_path, **{identity: "collision_b"})
        second.begin(None)
