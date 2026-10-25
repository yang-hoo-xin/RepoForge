"""Public process projections preserve raw statements, action identity and omissions."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.mod_agent.planning import process_trajectory as projection


def verified_rows():
    rows = []
    def ref(name):
        return {"path": "C:/original/" + name + ".json", "sha256": "2" * 64}
    for index, content in enumerate(["Inspect a competing explanation.", "", "Inspect a competing explanation."]):
        arguments = {"parent_revision_ref": "v2/snapshot/" + "1" * 64 + ".json", "path": "A.cs", "root_id": "target", "offset": index * 100}
        rows.append({"call": {"call_id": str(index), "name": "source_read", "arguments_ref": ref("args-" + str(index)),
            "response_ref": ref("response-" + str(index)), "execution_receipt_ref": ref("receipt-" + str(index))},
            "arguments": arguments, "response": {"content": content},
            "receipt": {"output": {"text": "SOURCE_BODY_NOT_A_STATEMENT" * 1000,
                "offset": index * 100, "next_offset": index * 100 + 100, "total_chars": 3000}}})
    return {"checkpoint": {"details": {"model_profile": {"model": "local", "context_window": 65536}},
        "loop": {"phase": "provider_pending"}}, "source": {"checkpoint_ref": {"sha256": "3" * 64}},
        "run_id": "run-a", "request_id": "request-a", "process": {"status": "timed_out"}, "records": rows, "pending": [],
        "responses": [{"turn": i + 1, "response": row["response"]} for i, row in enumerate(rows)]}


def test_all_actions_and_verbatim_statement_occurrences_survive_body_omission():
    source = verified_rows()
    original = deepcopy(source)
    view, omitted = projection.project_verified(source)
    assert source == original
    assert view["completed_actions"] == 3 and len(view["actions"]) == 3
    assert view["statements"] == [["Inspect a competing explanation.", [0, 2], [1, 3]]]
    assert view["empty_visible_statement_count"] == 1
    assert view["pending_outcome"] == "unknown" and view["business_result"] == "not_established"
    assert list(view["paths"].values()) == ["A.cs"]
    assert list(view["reference_aliases"].values()) == [original["records"][0]["arguments"]["parent_revision_ref"]]
    assert len(omitted) == 3 and all(row["locator"] == ["output", "text"] for row in omitted)
    assert "SOURCE_BODY_NOT_A_STATEMENT" not in str(view)
    assert [row[3]["offset"] for row in view["actions"]] == [0, 100, 200]


def test_pending_model_action_is_not_completed_and_keeps_its_explicit_statement():
    source = verified_rows()
    pending = deepcopy(source["records"][0])
    pending.pop("receipt")
    pending["call"]["name"] = "candidate_check"
    pending["response"]["content"] = "I will inspect the public result."
    source["pending"].append(pending)
    source["checkpoint"]["loop"]["phase"] = "tool_pending"
    view, _ = projection.project_verified(source)
    assert view["completed_actions"] == 3 and view["pending_model_actions"] == 1
    assert view["actions"][-1][-1] == {"execution": "unknown_not_in_completed_prefix"}
    assert view["statements"][-1] == ["I will inspect the public result.", [3], []]


def test_text_only_response_and_known_rejection_are_not_lost_or_falsely_executed():
    source = verified_rows()
    source["responses"].append({"turn": 4, "response": {"content": "A different hypothesis is still possible."}})
    rejected = deepcopy(source["records"][0])
    rejected.pop("receipt")
    rejected.update(execution_status="not_dispatched", recorded_feedback={"argument_errors": ["missing field"]})
    source["pending"].append(rejected)
    view, _ = projection.project_verified(source)
    assert view["completed_actions"] == 3 and view["not_dispatched_actions"] == 1
    assert view["statements"][-1] == ["A different hypothesis is still possible.", [], [4]]
    assert view["actions"][-1][-1] == {"execution": "not_dispatched", "feedback": {"argument_errors": ["missing field"]}}


def test_null_model_content_remains_unexpressed():
    source = verified_rows()
    for row in source["records"]:
        row["response"]["content"] = None
    view, _ = projection.project_verified(source)
    assert view["statements"] == [] and view["empty_visible_statement_count"] == 3


def test_public_check_failure_is_preserved_even_though_source_text_is_omitted():
    source = verified_rows()
    source["records"][0]["call"]["name"] = "candidate_check"
    observed = [{"expected": 3, "actual": 4, "details": "Public failure"}]
    source["records"][0]["receipt"]["output"].update(status="behavior_failed", observations=observed)
    view, omitted = projection.project_verified(source)
    assert view["actions"][0][-1]["observations"] == observed
    assert not any(o["field"] == "output.observations" for o in omitted)


def test_private_location_is_rejected_before_open(tmp_path):
    hidden = projection.write_once(tmp_path / "hidden" / "report.json", {"answer": "do not read"})
    with pytest.raises(ValueError, match="NONPUBLIC_SOURCE"):
        projection.read_public(hidden)


def test_reference_cannot_escape_declared_original_store(tmp_path):
    source = projection.write_once(tmp_path / "another" / "data.json", {})
    with pytest.raises(ValueError, match="REFERENCE_OUTSIDE_STORE"):
        projection.read_public(source, root=tmp_path / "run")


@pytest.fixture
def registered_prefix(tmp_path, monkeypatch):
    from src.mod_agent.platform_execution_snapshot import prepare_snapshot
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact

    original, code = tmp_path / "original", tmp_path / "code"
    for directory in ("src", "scripts", "config"):
        (code / directory).mkdir(parents=True)
    (code / "src/owner.py").write_text("# frozen owner", encoding="utf-8")
    (code / "scripts/run_frozen_product.py").write_text("# never executed", encoding="utf-8")
    (code / "config/settings.json").write_text("{}", encoding="utf-8")
    project, run = original / "project", original / "runs/run-test"
    project.mkdir(parents=True)
    (project / "A.cs").write_text("a=1;", encoding="utf-8")
    arguments = ["--workspace-root", str(run.parent), "--target-project", str(project),
        "--run-id", "run-test", "--request-id", "request-test"]
    config = projection.write_once(original / "config.json", {"arguments": arguments})
    mutable = original / "mutable-controller.py"
    mutable.write_text("# original controller", encoding="utf-8")
    mutable_ref = projection.file_ref(mutable)
    platform = prepare_snapshot(source_root=code, snapshot_root=tmp_path / "frozen", run_root=run,
        business_argv=["run-request", *arguments, "--", "Public task"],
        bound_inputs=[(config["path"], config["sha256"]), (mutable_ref["path"], mutable_ref["sha256"])])
    frozen = projection.read_public(platform)
    store = run / "a"
    authority = save_v2_artifact(store, "authority", {"scope": "file", "operation": "modify_existing",
        "allowed_edit_paths": ["A.cs"], "read_roots": {"target": str(project)}})
    state = TaskAgentV2State.create(store, "request-test", project, allowed_edit_paths=["A.cs"], authority_ref=authority,
        task_requirements={"original_request": "Correct the public task without changing interfaces.",
            "acceptance_criteria": ["The declared public behavior passes."], "non_goals": ["No API rename."]})
    host_ref = state.save("host-recovery-config", {"graph_state": {"run_id": "run-test", "request_id": "request-test"},
        "owner_refs": [projection.file_ref(tmp_path / "frozen/src/owner.py")]})
    checkpoint = {"schema": "task-agent-session-checkpoint.v2", "task_contract_ref": state.contract_ref,
        "host_config_ref": host_ref, "details": {"run_id": "run-test", "request_id": "request-test", "model_profile": {"model": "local"},
            "completed_tool_records": [], "provider_turns": []}, "task_head": state.head,
        "session": {"actual_calls": []}, "loop": {"phase": "provider_pending", "call_records": {}, "events": []}}
    checkpoint_ref = projection.file_ref(store / state.save("session-checkpoint", checkpoint))
    process = projection.write_once(original / "execution/process.json", {"started": True, "finished_at": "2026-10-25T00:00:00Z",
        "status": "timed_out", "cwd": str(tmp_path / "frozen"), "argv": [frozen["python_executable"], "-I", "-B",
            str(tmp_path / "frozen/scripts/run_frozen_product.py"), "--_child", platform["path"], platform["sha256"]]})
    # Existing recovery-owner authority checks have their own tests; these fixtures isolate the new external bindings.
    monkeypatch.setattr(projection, "_validate_host_binding", lambda *_args: None)
    return {"kind": "checkpoint", "artifact_store_root": str(store), "checkpoint_ref": checkpoint_ref,
        "execution_config_ref": config, "platform_ref": platform, "process_ref": process}


def test_registered_frozen_prefix_can_project_without_writing_originals(registered_prefix, tmp_path):
    source = registered_prefix
    root = Path(source["process_ref"]["path"]).parent.parent
    before = {str(p): projection.file_ref(p) for p in root.rglob("*") if p.is_file()}
    episode_ref = projection.prepare_trajectory(source, tmp_path / "prepared")
    episode = projection.read_public(episode_ref)
    assert episode["run_id"] == "run-test" and episode["information_gain"] == ["public_process_observation"]
    assert before == {str(p): projection.file_ref(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("changed_hash", [False, True])
def test_plan_reference_path_representation_keeps_exact_hash(registered_prefix, tmp_path, changed_hash):
    source = deepcopy(registered_prefix)
    process = projection.read_public(source["process_ref"])
    source["plan_ref"] = projection.write_once(tmp_path / "plan.json", {
        "run_id": "run-test", "run_root": str(Path(source["artifact_store_root"]).parent),
        "root": str(Path(source["process_ref"]["path"]).parent.parent),
        "workspace": process["cwd"], "command": process["argv"],
        "platform_ref": {"path": Path(source["platform_ref"]["path"]).as_posix(),
                         "sha256": "0" * 64 if changed_hash else source["platform_ref"]["sha256"]},
    })
    if changed_hash:
        with pytest.raises(ValueError, match="PROCESS_PLAN_CHANGED"):
            projection.verify_source(source)
    else:
        assert projection.verify_source(source)["run_id"] == "run-test"


@pytest.mark.parametrize("target", ["process", "configuration", "frozen_owner"])
def test_rehashed_cross_run_or_process_identity_cannot_be_projected(registered_prefix, target):
    import json

    source = deepcopy(registered_prefix)
    if target == "frozen_owner":
        path = Path(source["platform_ref"]["path"]).parent / "src/owner.py"
        path.write_text("# different owner", encoding="utf-8")
    else:
        key = "process_ref" if target == "process" else "execution_config_ref"
        value = projection.read_public(source[key])
        if target == "process":
            value["argv"][-1] = "0" * 64
        else:
            value["arguments"][value["arguments"].index("--run-id") + 1] = "another-run"
        path = Path(source[key]["path"])
        path.write_text(json.dumps(value), encoding="utf-8")
        source[key] = projection.file_ref(path)
    with pytest.raises(ValueError, match="OWNER_CHANGED|REGISTERED_RUN_CHANGED|PROCESS_ARGV_CHANGED"):
        projection.verify_source(source)


def test_projection_cannot_write_into_source_attempt(registered_prefix):
    original = Path(registered_prefix["process_ref"]["path"]).parent.parent
    with pytest.raises(ValueError, match="OUTPUT_OVERLAP"):
        projection.prepare_trajectory(registered_prefix, original / "new-projection")
    assert not (original / "new-projection").exists()


def test_historical_collection_uses_retained_controller_without_replacing_live_bytes(registered_prefix):
    frozen = projection.FrozenExecution(registered_prefix["platform_ref"])
    external = next(r for r in frozen.document["external_refs"] if r["path"].endswith("mutable-controller.py"))
    Path(external["path"]).write_text("# later development", encoding="utf-8")
    with pytest.raises(ValueError, match="EXTERNAL_REFERENCE_CHANGED"):
        frozen.verify_files()
    assert projection.verify_source(registered_prefix)["run_id"] == "run-test"
    assert Path(external["path"]).read_text() == "# later development"


def test_retained_external_tamper_and_snapshot_path_escape_are_rejected(registered_prefix, tmp_path):
    from types import SimpleNamespace

    frozen = projection.FrozenExecution(registered_prefix["platform_ref"])
    document = deepcopy(frozen.document)
    escaped = tmp_path / "escaped.py"
    escaped.write_text("# outside snapshot", encoding="utf-8")
    document["files"]["../escaped.py"] = {"sha256": projection.file_ref(escaped)["sha256"]}
    with pytest.raises(ValueError, match="FROZEN_PATH_ESCAPE"):
        projection.verify_retained_platform(SimpleNamespace(document=document, root=frozen.root))
    external = frozen.document["external_refs"][-1]
    (frozen.root / external["retained_path"]).write_text("# replaced", encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_FILE_CHANGED"):
        projection.verify_source(registered_prefix)


def test_existing_public_episode_consumes_only_original_selectors_without_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "run-public"
    arguments = {"path": "A.cs", "parent_revision_ref": "v2/snapshot/original.json"}
    response = projection.write_once(root / "response.json", {"content": "UNSELECTED_VISIBLE_TEXT",
        "tool_calls": [{"call_id": "call-1", "name": "source_read", "arguments": arguments}]})
    receipt = projection.write_once(root / "receipt.json", {"origin": {"call_id": "call-1", "name": "source_read",
        "arguments": arguments, "response_ref": response}, "output": {"text": "source body", "status": "returned"}})
    refs = [{"kind": "json_file", "evidence_id": key, **ref, "visibility": "public", "allowed_uses": ["strategy_learning"]}
        for key, ref in (("public_action_0", response), ("public_action_0_result", receipt))]
    episode = projection.write_once(root / "episode.json", {"schema_version": "public_strategy_episode_v1",
        "visibility": "public", "data_use": "strategy_learning", "task_contract_ref": "original-contract",
        "applicability": {"runtime": "dotnet"}, "evidence_refs": refs, "selections": [
            {"evidence_id": "public_action_0", "locator": ["tool_calls", 0, "name"]},
            {"evidence_id": "public_action_0", "locator": ["tool_calls", 0, "arguments"]},
            {"evidence_id": "public_action_0_result", "locator": ["output", "text"]}]})
    source = {"kind": "public_episode", "episode_ref": episode, "allowed_source_root": str(root), "run_id": "run-public"}
    monkeypatch.setattr(projection, "TaskAgentV2State", lambda *_args: pytest.fail("Public episode must not read a checkpoint"))
    verified = projection.verify_source(source)
    view, omitted = projection.project_verified(verified)
    assert view["completed_actions"] == 1 and view["statements"] == []
    assert "UNSELECTED_VISIBLE_TEXT" not in str(view) and len(omitted) == 1
    assert "source_episode_sha256" in view and "checkpoint_sha256" not in view
    derived = projection.read_public(projection.prepare_trajectory(source, tmp_path / "derived"))
    assert derived["applicability"] == {"runtime": "dotnet"}
    assert derived["task_context_selections"] == []
    with pytest.raises(ValueError, match="DOES_NOT_READ_CHECKPOINTS"):
        projection.verify_source({**source, "checkpoint_ref": {"path": "forbidden"}})


@pytest.mark.parametrize("select_excluded", [False, True])
def test_episode_batches_count_responses_once_and_never_read_unselected_checkpoint(tmp_path, monkeypatch, select_excluded):
    root = tmp_path / "run-batch"
    calls = [{"call_id": "call-" + str(i), "name": "source_read", "arguments": {"path": path}}
        for i, path in enumerate(("A.cs", "B.cs"))]
    response = projection.write_once(root / "response.json", {"content": "One response, two requested reads.", "tool_calls": calls})
    refs, selections = [], []
    for i, call in enumerate(calls):
        key = "public_action_" + str(i)
        receipt = projection.write_once(root / ("receipt-" + str(i) + ".json"), {
            "origin": {**call, "response_ref": response}, "output": {"text": "public source"}})
        refs.extend({"kind": "json_file", "evidence_id": name, **ref, "visibility": "public", "allowed_uses": ["strategy_learning"]}
            for name, ref in ((key, response), (key + "_result", receipt)))
        selections.extend({"evidence_id": key, "locator": ["tool_calls", i, field]} for field in ("name", "arguments"))
    selections.append({"evidence_id": "public_action_1", "locator": ["content"]})
    excluded = str(root / "v2/session-checkpoint/do-not-open.json")
    refs.append({"kind": "json_file", "evidence_id": "excluded", "path": excluded, "sha256": "0" * 64,
        "visibility": "public", "allowed_uses": ["strategy_learning"]})
    if select_excluded:
        selections.append({"evidence_id": "excluded", "locator": ["private"]})
    episode = projection.write_once(root / "episode.json", {"schema_version": "public_strategy_episode_v1", "visibility": "public",
        "data_use": "strategy_learning", "task_contract_ref": "original-contract", "applicability": {"runtime": "dotnet"},
        "evidence_refs": refs, "selections": selections})
    source = {"kind": "public_episode", "episode_ref": episode, "allowed_source_root": str(root), "run_id": "run-batch"}
    original_read, opened = projection.read_public, []

    def guarded(reference, **kwargs):
        assert reference["path"] != excluded, "Excluded state must not even be opened"
        opened.append(reference["path"])
        return original_read(reference, **kwargs)

    monkeypatch.setattr(projection, "read_public", guarded)
    if select_excluded:
        with pytest.raises(ValueError, match="NONPUBLIC_SELECTION"):
            projection.verify_source(source)
    else:
        verified = projection.verify_source(source)
        assert len(verified["responses"]) == 1 and verified["responses"][0]["turn"] is None
        view, _ = projection.project_verified(verified)
        assert view["completed_actions"] == 2 and view["model_responses_recorded"] == 1
        assert view["statements"] == [["One response, two requested reads.", [0, 1], ["selected-response-1"]]]
        assert view["actual_model_turns"] == "not_selected_in_original_episode"
    assert excluded not in opened


def test_new_process_episode_carries_verified_original_task_context(registered_prefix, tmp_path):
    from src.mod_agent.artifacts import ArtifactStore
    from src.mod_agent.planning.learning_request import resolve_task_context
    from src.mod_agent.planning.strategy_experience import StrategyExperience
    verified = projection.verify_source(registered_prefix)
    view, _ = projection.project_verified(verified)
    assert "source_task_context" not in view  # Sent once as a separate episode field, not duplicated in process facts.
    episode_ref = projection.prepare_trajectory(registered_prefix, tmp_path / "with-task-context")
    episode = projection.read_public(episode_ref)
    memory = StrategyExperience(ArtifactStore(tmp_path / "memory"))
    resolved = resolve_task_context(episode, memory._public_sources(episode["evidence_refs"]))
    assert resolved["fields"]["original_request"] == "Correct the public task without changing interfaces."
    assert resolved["fields"]["non_goals"] == ["No API rename."]
    assert resolved["field_sources"]["original_request"]["locator"] == ["task_requirements", "original_request"]
