"""Bound model-visible reads without weakening the exact parent/source preimage."""
from types import SimpleNamespace

import pytest

from src.mod_agent.task_agent_loop import TaskActionInputError, TaskCandidateRevisionInputError
from src.mod_agent.task_agent_v2_state import (
    TaskAgentV2State,
    TaskAgentV2StateError,
    _casefold_match_offset,
    save_v2_artifact,
)
from tests.unit.test_task_agent_v2_admission import action, response
from tests.unit.test_task_agent_v2_admission import session as session  # noqa: F401
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def large_source(tmp_path):
    source = tmp_path / "large-source"
    source.mkdir()
    text = "首行😀\n" + "a" * 12000 + "\nint value = 1;\n"
    (source / "A.cs").write_bytes(text.encode("utf-8"))
    store = tmp_path / "large-artifacts"
    authority = save_v2_artifact(store, "authority", {
        "schema": "task-agent-authority.v2", "scope": "file", "operation": "modify_existing",
        "allowed_edit_paths": ["A.cs"], "read_roots": {"target": str(source)}})
    return TaskAgentV2State.create(store, "large", source, {}, ["A.cs"], authority), text


def test_unicode_long_line_paging_reassembles_exact_snapshot(tmp_path):
    state, text = large_source(tmp_path)
    chunks, offset, hashes = [], 0, set()
    while True:
        page = state.read_source("baseline", "A.cs", offset=offset, limit=4000)
        chunks.append(page["text"])
        hashes.add(page["file_sha256"])
        assert page["total_chars"] == len(text) and page["total_lines"] == 3
        assert len(page["text"]) <= 4000 and page["text_scope"] == "read_window"
        stored = state.load(page["source_view_ref"])
        assert stored["text"] == text
        assert stored["read_window"] == {"offset": offset, "end": offset + len(page["text"])}
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert "".join(chunks) == text and len(hashes) == 1
    eof = state.read_source("baseline", "A.cs", offset=len(text), limit=10)
    assert eof["text"] == "" and eof["next_offset"] is None


def test_page_edit_requires_visible_preimage_and_current_parent(tmp_path):
    state, text = large_source(tmp_path)
    first = state.read_source("baseline", "A.cs", offset=0, limit=4000)
    binding = state.add_binding(state.registry_ref, first["source_view_ref"], "A.cs")
    edit = {"binding_id": binding["binding_id"], "source_view_ref": first["source_view_ref"],
        "before_text": "value = 1", "after_text": "value = 2"}
    with pytest.raises(TaskAgentV2StateError, match="before_text_outside_read_window"):
        state.validate_submission("baseline", [edit])
    with pytest.raises(TaskAgentV2StateError, match="before_text_outside_read_window"):
        state.submit("baseline", [edit])
    final = state.read_source("baseline", "A.cs", offset=12000, limit=100)
    edit["source_view_ref"] = final["source_view_ref"]
    state.validate_submission("baseline", [edit])
    candidate = state.submit("baseline", [edit])["candidate_ref"]
    assert state.materialize(candidate)["A.cs"] == text.replace("value = 1", "value = 2")
    rebound = state.load(candidate)
    rebound["changes"][0]["source_view_ref"] = first["source_view_ref"]
    with pytest.raises(TaskAgentV2StateError, match="before_text_outside_read_window"):
        state.validate_candidate(state.save("candidate", rebound))
    with pytest.raises(TaskAgentV2StateError, match="stale_or_invalid_source_view"):
        state.submit(candidate, [edit])
    forged = state.load(final["source_view_ref"])
    forged["read_window"]["end"] = len(text) + 1
    with pytest.raises(TaskAgentV2StateError, match="stale_or_invalid_source_view"):
        state._validate_view(forged, state.contract["baseline_manifest_ref"], "A.cs")


def test_search_snippet_is_bounded_and_does_not_attest_unseen_file_text(tmp_path):
    state, text = large_source(tmp_path)
    match = state.search_source("baseline", "value")["matches"][0]
    view = state.load(match["source_view_ref"])
    assert view["text"][view["read_window"]["offset"]:view["read_window"]["end"]] == match["text"]
    assert match["offset"] == text.index("int value")
    state._validate_visible_preimage(view, "value = 1")
    with pytest.raises(TaskAgentV2StateError, match="before_text_outside_read_window"):
        state._validate_visible_preimage(view, "首行")
    long_line = state.search_source("baseline", "a" * 10)["matches"][0]
    assert len(long_line["text"]) == 512 and long_line["text_truncated"] is True


@pytest.mark.parametrize("prefix,needle,query", [("x" * 1000, "TARGET", "target"),
    ("ß" * 1000, "Straße", "STRASSE"), ("😀" * 1000, "ß", "ss")])
def test_casefold_location_uses_original_character_offsets(prefix, needle, query):
    text = prefix + needle
    assert _casefold_match_offset(text, query) == len(prefix)


@pytest.mark.parametrize("window", [{"offset": -1}, {"limit": 0}, {"limit": 8001}, {"offset": 1000}])
def test_bad_window_is_correctable_without_dispatch_or_write(session, window):
    before = set(session.state.root.rglob("*"))
    with pytest.raises(TaskActionInputError):
        action(session, "source_read").prepare({"parent_revision_ref": "baseline", "path": "A.cs", **window})
    assert set(session.state.root.rglob("*")) == before


def test_tool_defaults_and_same_session_correction_then_page_edit(session, tmp_path):
    session.state, text = large_source(tmp_path)
    requests = []

    def complete(request):
        requests.append(request)
        last = request.tool_results[-1].output if request.tool_results else None
        if len(requests) == 1:
            return response("source_read", {"parent_revision_ref": "baseline", "path": "A.cs", "offset": len(text) + 1}, "bad-window")
        if len(requests) == 2:
            assert last["status"] == "rejected_not_executed"
            return response("source_read", {"parent_revision_ref": "baseline", "path": "A.cs"}, "read-page")
        if len(requests) == 3:
            assert len(last["value"]["text"]) == 2000 and last["value"]["next_offset"] == 2000
            binding = session.state.add_binding(session.state.registry_ref, last["value"]["source_view_ref"], "A.cs")
            edit = {"binding_id": binding["binding_id"], "source_view_ref": last["value"]["source_view_ref"],
                "before_text": "value = 1", "after_text": "value = 2"}
            with pytest.raises(TaskCandidateRevisionInputError) as error:
                action(session, "candidate_submit").prepare({"parent_revision_ref": session.state.current_revision_ref, "edits": [edit]})
            assert "before_text_outside_read_window" in error.value.argument_errors
            return response("candidate_read", {"parent_revision_ref": session.state.current_revision_ref,
                "path": "A.cs", "offset": 12000}, "later-page")
        raise RuntimeError("controlled fixture stops after page delivered")

    result = session.run(SimpleNamespace(complete_turn=complete), lambda _event: None)
    assert result.tool_argument_correction_used and len(requests) == 4, result.termination
    assert requests[-1].tool_results[-1].output["value"]["text"] == text[12000:]
    assert session.state.current_revision_ref == session.state.contract["baseline_manifest_ref"]
