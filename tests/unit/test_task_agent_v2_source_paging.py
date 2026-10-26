"""Bound model-visible reads without weakening the exact parent/source preimage."""
from copy import deepcopy
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


def large_source(tmp_path, text=None):
    source = tmp_path / "large-source"
    source.mkdir()
    text = text if text is not None else "首行😀\n" + "a" * 12000 + "\nint value = 1;\n"
    (source / "A.cs").write_bytes(text.encode("utf-8"))
    store = tmp_path / "large-artifacts"
    authority = save_v2_artifact(store, "authority", {
        "schema": "task-agent-authority.v2", "scope": "file", "operation": "modify_existing",
        "allowed_edit_paths": ["A.cs"], "read_roots": {"target": str(source)}})
    return TaskAgentV2State.create(store, "large", source, {}, ["A.cs"], authority), text


def test_search_read_args_select_line_context_with_unicode_crlf_and_exact_preimage(tmp_path):
    lines = [f"// 行{number}😀\r\n" for number in range(1, 331)]
    lines[325] = "int navigation_target = 1;\r\n"
    state, text = large_source(tmp_path, "".join(lines))
    match = state.search_source("baseline", "navigation_target")["matches"][0]
    assert match["line"] == 326 and match["line_offset"] == sum(map(len, lines[:325]))
    assert match["read_args"] == {"parent_revision_ref": state.current_revision_ref, "path": "A.cs", "root_id": "target",
        "start_line": 323, "line_count": 40}
    page = state.read_source(**match["read_args"])
    assert page["text"] == "".join(lines[322:])
    assert page["line_window"] == {"start_line": 323, "end_line": 330, "start_column_chars": 0, "ends_mid_line": False}
    assert page["read_window_chars"] == {"offset_chars": sum(map(len, lines[:322])), "end_chars": len(text)}
    assert page["next_read"] is None
    stored = state.load(page["source_view_ref"])
    with pytest.raises(TaskAgentV2StateError, match="before_text_outside_read_window"):
        state._validate_visible_preimage(stored, lines[0])
    change = {"path": "A.cs", "source_view_ref": page["source_view_ref"],
        "before_text": "navigation_target = 1", "after_text": "navigation_target = 2"}
    candidate = state.submit("baseline", [change])["candidate_ref"]
    assert state.materialize(candidate)["A.cs"] == text.replace("navigation_target = 1", "navigation_target = 2")
    with pytest.raises(TaskAgentV2StateError, match="stale_or_invalid_source_view"):
        state.submit(candidate, [{**change, "before_text": "navigation_target = 2", "after_text": "navigation_target = 3"}])


def test_line_read_clipping_switches_to_character_continuation_without_repeating_long_line(tmp_path):
    state, text = large_source(tmp_path)
    page = state.read_source("baseline", "A.cs", start_line=1)
    assert len(page["text"]) == 8000 and page["line_window"]["ends_mid_line"] is True
    assert page["next_read"]["offset_chars"] == 8000 and "start_line" not in page["next_read"]
    pieces, offsets = [page["text"]], [page["read_window"]["offset"]]
    while page["next_read"]:
        page = state.read_source(**page["next_read"])
        pieces.append(page["text"])
        assert page["read_window"]["offset"] > offsets[-1]
        offsets.append(page["read_window"]["offset"])
        assert len(page["text"]) <= 8000
    assert "".join(pieces) == text


@pytest.mark.parametrize("coordinates", [{"start_line": 2}, {"line_count": 2}, {"offset_chars": 2},
    {"limit_chars": 2}, {"offset": 2}, {"limit": 2}, {}])
def test_coordinate_family_defaults_preserve_unicode_and_crlf(tmp_path, coordinates):
    state, text = large_source(tmp_path, "😀汉\r\nbeta\r\nlast")
    page = state.read_source("baseline", "A.cs", **coordinates)
    if "start_line" in coordinates:
        expected = "beta\r\nlast"
    elif "line_count" in coordinates:
        expected = "😀汉\r\nbeta\r\n"
        assert page["next_read"]["start_line"] == 3
        assert state.read_source(**page["next_read"])["text"] == "last"
    elif "offset_chars" in coordinates or "offset" in coordinates:
        expected = text[2:]
    elif "limit_chars" in coordinates or "limit" in coordinates:
        expected = text[:2]
        assert page["next_read"]["offset_chars"] == 2
    else:
        expected = text
    assert page["text"] == expected
    assert page["text"] == text[page["read_window_chars"]["offset_chars"]:page["read_window_chars"]["end_chars"]]
    assert page["total_lines"] == 3
    eof = state.read_source("baseline", "A.cs", offset_chars=len(text), limit_chars=2)
    assert eof["text"] == "" and eof["next_read"] is None and eof["line_window"]["end_line"] is None


def test_empty_file_line_read_and_default_read_are_bounded(tmp_path):
    state, _ = large_source(tmp_path, "")
    for coordinates in ({}, {"start_line": 1}, {"offset_chars": 0}):
        page = state.read_source("baseline", "A.cs", **coordinates)
        assert page["text"] == "" and page["next_read"] is None and page["total_lines"] == 0
        assert page["read_window_chars"] == {"offset_chars": 0, "end_chars": 0}


def test_explicit_character_overlap_must_show_the_whole_exact_preimage(tmp_path):
    state, text = large_source(tmp_path, "0123456789αβγδ\r\nsuffix")
    first = state.read_source("baseline", "A.cs", offset_chars=0, limit_chars=12)
    change = {"path": "A.cs", "source_view_ref": first["source_view_ref"],
        "before_text": "9αβγ", "after_text": "replacement"}
    with pytest.raises(TaskAgentV2StateError, match="before_text_outside_read_window"):
        state.submit("baseline", [change])
    overlap = state.read_source("baseline", "A.cs", offset_chars=9, limit_chars=8)
    assert overlap["line_window"]["start_column_chars"] == 9
    change["source_view_ref"] = overlap["source_view_ref"]
    candidate = state.submit("baseline", [change])["candidate_ref"]
    assert state.materialize(candidate)["A.cs"] == text.replace("9αβγ", "replacement")


@pytest.mark.parametrize("coordinates", [{"start_line": 1, "offset_chars": 0}, {"line_count": 1, "limit": 1},
    {"offset_chars": 0, "offset": 0}, {"start_line": 0}, {"start_line": True}, {"line_count": 0},
    {"start_line": 999}, {"offset_chars": -1}, {"limit_chars": 8001}, {"limit_chars": True}])
def test_invalid_or_mixed_coordinate_families_do_not_save_a_view(state, coordinates):
    before = set(state.root.rglob("*"))
    with pytest.raises(TaskAgentV2StateError, match="invalid_source_read_window"):
        state.read_source("baseline", "A.cs", **coordinates)
    assert set(state.root.rglob("*")) == before


def test_search_read_args_preserve_dependency_root_and_cannot_create_edit_authority(state):
    hit = state.search_source("baseline", "dependency", root_id="dep")["matches"][0]
    assert hit["read_args"]["root_id"] == "dep"
    assert hit["read_args"]["parent_revision_ref"] == state.contract["read_roots"]["dep"]
    page = state.read_source(**hit["read_args"])
    assert page["text"] == "dependency"
    with pytest.raises(TaskAgentV2StateError, match="source_view_not_edit_authority"):
        state.add_binding(state.registry_ref, page["source_view_ref"], "D.cs")
    with pytest.raises(TaskAgentV2StateError, match="read_root_denied"):
        state.read_source(**{**hit["read_args"], "root_id": "unregistered"})
    with pytest.raises(TaskAgentV2StateError, match="dependency_revision_mismatch"):
        state.read_source(**{**hit["read_args"], "parent_revision_ref": state.current_revision_ref})


def test_normal_tool_schema_and_raw_call_identity_support_new_and_legacy_coordinates(session, tmp_path):
    from src.mod_agent.model_qualification import canonical_hash

    session.state, text = large_source(tmp_path)
    raw = [
        {"parent_revision_ref": "baseline", "path": "A.cs", "start_line": 3, "working_state": {}},
        {"parent_revision_ref": "baseline", "path": "A.cs", "offset_chars": 12000, "limit_chars": 100, "working_state": {}},
        {"parent_revision_ref": "baseline", "path": "A.cs", "offset": 0, "limit": 20, "working_state": {}},
    ]
    for name in ("source_read", "candidate_read"):
        definition = action(session, name).definition
        properties = definition.parameters["properties"]
        assert {"start_line", "line_count", "offset_chars", "limit_chars"} <= set(properties)
        assert not {"offset", "limit"}.intersection(properties)
    for args in raw:
        assert action(session, "source_read").prepare(deepcopy(args)).value == {k: v for k, v in args.items() if k != "working_state"}
    with pytest.raises(TaskActionInputError) as error:
        action(session, "source_read").prepare({**raw[0], "offset_chars": 0})
    assert "invalid_source_read_window" in error.value.argument_errors
    index = [0]

    def complete(_request):
        if index[0] == len(raw):
            raise RuntimeError("controlled fixture stops after reads")
        i = index[0]
        index[0] += 1
        return response("candidate_read" if i == 1 else "source_read", deepcopy(raw[i]), "navigation-" + str(i))

    result = session.run(SimpleNamespace(complete_turn=complete), session.observe_dispatch_event)
    expected = ["int value = 1;\n", text[12000:12100], text[:20]]
    for i, origin in enumerate(session.actual_calls):
        assert origin["arguments"] == raw[i]
        record = result.call_records[origin["call_id"]]
        assert record["arguments_sha256"] == canonical_hash(raw[i])
        assert record["result"]["output"]["value"]["text"] == expected[i]
    assert session.state.current_revision_ref == session.state.contract["baseline_manifest_ref"]


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


@pytest.mark.parametrize("target_mode", ["binding_id", "path"])
def test_page_edit_requires_visible_preimage_and_current_parent(tmp_path, target_mode):
    state, text = large_source(tmp_path)
    first = state.read_source("baseline", "A.cs", offset=0, limit=4000)
    target = {"path": "A.cs"}
    if target_mode == "binding_id":
        target = {"binding_id": state.add_binding(state.registry_ref, first["source_view_ref"], "A.cs")["binding_id"]}
    edit = {**target, "source_view_ref": first["source_view_ref"],
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
