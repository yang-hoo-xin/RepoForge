"""Exercise persisted task state and real source identities.

为什么要有它：文件、权限与候选反例必须通过真实状态层检验。
"""
import json

import pytest

from src.mod_agent.task_agent_v2_state import (
    TaskAgentV2State,
    TaskAgentV2StateError,
    save_v2_artifact,
)


@pytest.fixture
def state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "A.cs").write_bytes(b"a = 1;\n")
    (source / "B.cs").write_bytes(b"b = 2;\n")
    (source / "private.cs").write_text("private", encoding="utf-8")
    (source / ".env").write_text("secret", encoding="utf-8")
    dep = tmp_path / "dependency"
    dep.mkdir()
    (dep / "D.cs").write_text("dependency", encoding="utf-8")
    store = tmp_path / "artifacts"
    authority = save_v2_artifact(store, "authority", {
        "schema": "task-agent-authority.v2", "scope": "file", "operation": "modify_existing",
        "allowed_edit_paths": ["A.cs", "B.cs"], "read_roots": {"target": str(source), "dep": str(dep)},
    })
    return TaskAgentV2State.create(store, "task", source, {"dep": dep}, ["A.cs", "B.cs"], authority)


def edit(state, path, before, after, operation_id=None):
    view = state.read_source(state.current_revision_ref, path)
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], path)
    return state.submit(state.current_revision_ref, [{"binding_id": binding["binding_id"],
        "source_view_ref": view["source_view_ref"], "before_text": before, "after_text": after}],
        operation_id=operation_id, model_origin={"call_id": "test-call", "response_ref": "actual-response"})


def test_empty_registry_reads_target_and_dependency_without_edit_permission(state):
    assert state.load(state.registry_ref)["bindings"] == {}
    assert state.read_source("baseline", "private.cs")["text"] == "private"
    dep = state.read_source("baseline", "D.cs", "dep")
    with pytest.raises(TaskAgentV2StateError, match="source_view_not_edit_authority"):
        state.add_binding(state.registry_ref, dep["source_view_ref"], "D.cs")
    with pytest.raises(TaskAgentV2StateError, match="read_path_denied"):
        state.read_source("baseline", ".env")


def test_abandon_wrong_target_add_another_and_restore_first(state):
    first = edit(state, "A.cs", "1", "9")
    identity = first["candidate_source_id"]
    view = state.read_source(state.current_revision_ref, "B.cs")
    state.add_binding(state.registry_ref, view["source_view_ref"], "B.cs")
    state.update_plan({"target": "B.cs", "hypothesis": "A was irrelevant"})
    assert state.validate_candidate(first["candidate_ref"])["candidate_source_id"] == identity
    assert state.materialize(state.current_revision_ref)["A.cs"] == "a = 9;\n"
    second = edit(state, "B.cs", "2", "3")
    final = edit(state, "A.cs", "9", "1")
    assert state.materialize(final["candidate_ref"])["B.cs"] == "b = 3;\n"
    assert state.validate_candidate(final["candidate_ref"])["parent_revision_ref"] == second["candidate_ref"]
    assert [change["path"] for change in state.changes_from_baseline(final["candidate_ref"])] == ["B.cs"]
    restored = TaskAgentV2State(state.store_root, state.contract_ref)
    assert restored.current_revision_ref == final["candidate_ref"]
    assert restored.materialize(final["candidate_ref"]) == state.materialize(final["candidate_ref"])


def test_stale_view_and_stale_parent_are_rejected(state):
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    edit(state, "A.cs", "1", "2")
    edits = [{"binding_id": binding["binding_id"], "source_view_ref": view["source_view_ref"], "before_text": "2", "after_text": "3"}]
    with pytest.raises(TaskAgentV2StateError, match="stale_or_invalid_source_view"):
        state.submit(state.current_revision_ref, edits)
    with pytest.raises(TaskAgentV2StateError, match="stale_parent_revision"):
        state.submit("baseline", edits)


def test_plan_cannot_authorize_and_registry_uses_cas(state):
    old_registry = state.registry_ref
    state.update_plan({"allowed_edit_paths": ["private.cs"]})
    view = state.read_source("baseline", "private.cs")
    with pytest.raises(TaskAgentV2StateError, match="edit_path_denied"):
        state.add_binding(state.registry_ref, view["source_view_ref"], "private.cs")
    edit(state, "A.cs", "1", "2")
    view = state.read_source(state.current_revision_ref, "B.cs")
    with pytest.raises(TaskAgentV2StateError, match="stale_registry"):
        state.add_binding(old_registry, view["source_view_ref"], "B.cs")


def test_precise_preimage_and_no_whole_file_fallback(state):
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    edits = [{"binding_id": binding["binding_id"], "source_view_ref": view["source_view_ref"], "after_text": "new"}]
    with pytest.raises(TaskAgentV2StateError, match="explicit_nonempty"):
        state.validate_submission("baseline", edits)
    edits[0]["before_text"] = "missing"
    with pytest.raises(TaskAgentV2StateError, match="before_text_not_unique"):
        state.submit("baseline", edits)
    assert state.materialize("baseline")["A.cs"] == "a = 1;\n"


def test_operation_replay_does_not_resubmit_or_recharge(state):
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    edits = [{"binding_id": binding["binding_id"], "source_view_ref": view["source_view_ref"], "before_text": "1", "after_text": "2"}]
    first = state.submit("baseline", edits, operation_id="submit-1")
    assert state.submit("baseline", edits, operation_id="submit-1") == first
    state.record_resources({"tokens": 10}, "usage-1")
    state.record_resources({"tokens": 10}, "usage-1")
    restored = TaskAgentV2State(state.store_root, state.contract_ref)
    assert restored.head["resources"] == {"tokens": 10}
    with pytest.raises(TaskAgentV2StateError, match="operation_identity_conflict"):
        state.record_resources({"tokens": 11}, "usage-1")


def test_unknown_operation_is_not_reexecuted(state):
    with pytest.raises(OSError, match="interrupted"):
        with state._transaction("uncertain", {"kind": "test"}):
            raise OSError("interrupted")
    restored = TaskAgentV2State(state.store_root, state.contract_ref)
    assert restored.head["pending_operation"]
    with pytest.raises(TaskAgentV2StateError, match="operation_outcome_unknown"):
        restored.update_plan({"retry": True})


def test_artifact_tamper_and_candidate_forgery_are_rejected(state):
    candidate = edit(state, "A.cs", "1", "2")
    row = state.load(candidate["candidate_ref"])
    row["files"]["private.cs"] = "changed"
    forged = state.save("candidate", row)
    with pytest.raises(TaskAgentV2StateError, match="candidate_content_mismatch"):
        state.materialize(forged)
    (state.store_root / forged).write_text(json.dumps({"schema": "changed.v2"}), encoding="utf-8")
    with pytest.raises(TaskAgentV2StateError, match="artifact_hash_mismatch"):
        state.load(forged)


def test_region_authority_is_not_promoted_and_missing_authority_is_rejected(state):
    source = state.load(state.contract["baseline_manifest_ref"])["source_root"]
    authority = state.load(state.contract["authority_ref"])
    authority["scope"] = "region"
    ref = state.save("authority", authority)
    with pytest.raises(TaskAgentV2StateError, match="unsupported_scope"):
        TaskAgentV2State.create(state.store_root, "other", source, authority_ref=ref)
    authority["scope"] = "file"
    authority.pop("allowed_edit_paths")
    ref = state.save("authority", authority)
    with pytest.raises(TaskAgentV2StateError, match="invalid_authority"):
        TaskAgentV2State.create(state.store_root, "other", source, authority_ref=ref)


def test_search_observes_current_candidate_and_snapshot_is_immutable(state):
    source = state.load(state.contract["baseline_manifest_ref"])["source_root"]
    from pathlib import Path
    Path(source, "A.cs").write_text("external change", encoding="utf-8")
    assert state.read_source("baseline", "A.cs")["text"] == "a = 1;\n"
    result = edit(state, "A.cs", "1", "99")
    matches = state.search_source(result["candidate_ref"], "99")
    assert matches["total_matches"] == 1
    view = state.load(matches["matches"][0]["source_view_ref"])
    assert view["parent_revision_ref"] == result["candidate_ref"]


@pytest.mark.parametrize("path", ["../A.cs", "/A.cs", "C:/A.cs", "x/../A.cs", "A\\B.cs"])
def test_read_path_traversal_rejected(state, path):
    with pytest.raises(TaskAgentV2StateError, match="invalid_path"):
        state.read_source("baseline", path)


def test_restored_contract_cannot_expand_authority(state):
    contract = dict(state.contract)
    contract["allowed_edit_paths"].append("private.cs")
    forged = state.save("contract", contract)
    with pytest.raises(TaskAgentV2StateError, match="invalid_contract_authority"):
        TaskAgentV2State(state.store_root, forged)


def test_interrupted_head_commit_cannot_repeat_candidate(state, monkeypatch):
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    edits = [{"binding_id": binding["binding_id"], "source_view_ref": view["source_view_ref"], "before_text": "1", "after_text": "2"}]
    original_write = state._write_head
    writes = []

    def crash_on_commit(head):
        writes.append(head.copy())
        if len(writes) == 2:
            raise OSError("crash after candidate materialized")
        original_write(head)

    monkeypatch.setattr(state, "_write_head", crash_on_commit)
    with pytest.raises(OSError, match="crash after candidate"):
        state.submit("baseline", edits, operation_id="crashed-submit")
    restored = TaskAgentV2State(state.store_root, state.contract_ref)
    assert restored.head["pending_operation"]
    with pytest.raises(TaskAgentV2StateError, match="operation_outcome_unknown"):
        restored.submit("baseline", edits, operation_id="crashed-submit")
    assert len(list((state.root / "candidate").glob("*.json"))) == 1


def test_crlf_source_identity_is_not_normalized(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "A.cs").write_bytes(b"a = 1;\r\n")
    store = tmp_path / "artifacts"
    auth = save_v2_artifact(store, "authority", {"scope": "file", "operation": "modify_existing",
        "allowed_edit_paths": ["A.cs"], "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "crlf", source, allowed_edit_paths=["A.cs"], authority_ref=auth)
    candidate = edit(state, "A.cs", "1", "2")
    assert state.materialize(candidate["candidate_ref"])["A.cs"] == "a = 2;\r\n"


def test_multiple_edits_use_one_parent_and_identity_excludes_plan_registry(state):
    view = state.read_source("baseline", "A.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "A.cs")
    edits = [{"binding_id": binding["binding_id"], "source_view_ref": view["source_view_ref"],
        "before_text": before, "after_text": after} for before, after in [("a", "changed"), ("1", "5")]]
    state.validate_submission("baseline", edits)
    first = state.submit("baseline", edits)
    assert state.materialize(first["candidate_ref"])["A.cs"] == "changed = 5;\n"
    state.update_plan({"hypothesis": "now inspect B"})
    second_view = state.read_source(state.current_revision_ref, "B.cs")
    state.add_binding(state.registry_ref, second_view["source_view_ref"], "B.cs")
    edit(state, "B.cs", "2", "3")
    same_code = edit(state, "B.cs", "3", "2")
    assert same_code["candidate_source_id"] == first["candidate_source_id"]


def test_forged_registry_is_rejected_before_writing_candidate(state):
    edit(state, "A.cs", "1", "2")
    registry = state.load(state.registry_ref)
    binding_id = next(iter(registry["bindings"]))
    registry["bindings"][binding_id]["canonical_path"] = "private.cs"
    forged = state.save("registry", registry)
    head = state.head
    head["registry_ref"] = forged
    state._write_head(head)
    view = state.read_source(state.current_revision_ref, "private.cs")
    with pytest.raises(TaskAgentV2StateError, match="invalid_binding_authority"):
        state.submit(state.current_revision_ref, [{"binding_id": binding_id,
            "source_view_ref": view["source_view_ref"], "before_text": "private", "after_text": "bad"}])
    assert len(list((state.root / "candidate").glob("*.json"))) == 1
