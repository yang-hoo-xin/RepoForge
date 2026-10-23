"""Retained fixture evidence queried through the normal read-only command."""
import hashlib
import json

import pytest

from scripts import dsh_run_request as bridge
from tests.unit.test_task_agent_v2_admission import session as session
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _fixture(tmp_path, *, status="delivered", native=None):
    workspace = tmp_path / "runs"
    root = workspace / "one"
    native = native or {"schema_version": "task-agent-product-execution.v2", "run_id": "one",
        "status": "candidate_checked", "provider_turns": [], "completed_tool_records": []}
    native["loop_result_ref"] = _save(root / "loop.json", {"call_records": {}})
    native_ref = _save(root / "native" / "execution.json", native)
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": status, "execution_ref": native_ref})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    config = tmp_path / "config.json"
    _save(config, {"arguments": ["--workspace-root", str(workspace)]})
    return config, root


def _query(config, capsys, run_id="one"):
    assert bridge.main(["--execution-config", str(config), "--inspect-run", run_id,
        "--view", "evidence", "--format", "json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_normal_delivery_readonly_and_no_dispatch(tmp_path, capsys, monkeypatch):
    config, root = _fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(bridge.cli, "main", lambda *a: pytest.fail("execution attempted"))
    monkeypatch.setattr(bridge, "advance_task", lambda **k: pytest.fail("recovery attempted"))
    report = _query(config, capsys)
    assert report["terminal"]["status"] == "delivered"
    assert report["query_effects"] == {"model_calls": 0, "builds": 0, "tests": 0, "recoveries": 0, "writes": 0}
    assert len(report["executions"]) == 1
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["missing", "tampered", "outside", "oversize"])
def test_untrustworthy_references_never_claim_complete(tmp_path, capsys, kind):
    config, root = _fixture(tmp_path)
    path = root / "detail.json"
    ref = _save(path, {"private_payload": "never deliver this as verified"})
    if kind == "missing":
        ref["path"] = str(root / "absent.json")
    elif kind == "tampered":
        path.write_text("{}", encoding="utf-8")
    elif kind == "outside":
        ref = _save(tmp_path / "outside.json", {"marker": "FORBIDDEN CONTENT"})
    else:
        path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
        ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((root / "run_manifest.json").read_text())
    _save(root / "run_manifest.json", {**manifest, "detail_ref": ref})
    report = _query(config, capsys)
    assert report["complete"] is False
    statuses = {row["status"] for row in report["issues"]}
    assert {"missing": "missing", "tampered": "hash_mismatch", "outside": "forbidden", "oversize": "truncated"}[kind] in statuses
    assert "FORBIDDEN CONTENT" not in json.dumps(report)
    if kind == "oversize":
        assert "truncated" in statuses
    if kind == "outside":
        assert "forbidden" in statuses


@pytest.mark.parametrize("record", ["terminal", "response"])
@pytest.mark.parametrize("value", [[], None, "not a record"])
def test_non_object_json_record_never_claims_complete(tmp_path, capsys, record, value):
    config, root = _fixture(tmp_path)
    if record == "terminal":
        ref = _save(root / "run_result.json", value)
        _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": ref})
    else:
        response = _save(root / "response.json", value)
        native = {"schema_version": "task-agent-product-execution.v2", "run_id": "one",
            "provider_turns": [{"turn": 1, "reservation_id": "r-1",
                "request_ref": _save(root / "request.json", {"task": "investigate"}),
                "response_ref": response}],
            "completed_tool_records": [], "loop_result_ref": _save(root / "loop.json", {"call_records": {}})}
        native_ref = _save(root / "native" / "execution.json", native)
        terminal = _save(root / "run_result.json", {"run_id": "one", "status": "unknown", "execution_ref": native_ref})
        _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    report = _query(config, capsys)
    assert report["complete"] is False
    assert report["status"] == "partial"
    assert any(row["status"] == "unreadable" and "JSON object" in row.get("reason", "")
        for row in report["issues"])
    if record == "response":
        assert not report["executions"][0]["turns"][0]["response_observed"]


def test_duplicate_business_id_is_ambiguous_and_bound_config_conflicts(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    _save(root.parent / "copy" / "run_manifest.json", {"run_id": "one"})
    report = _query(config, capsys)
    assert report["status"] == "ambiguous" and not report["complete"]
    _save(config, {"arguments": ["--workspace-root", str(root.parent), "--run-id", "other"]})
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence"]) == 2
    assert "IDENTITY_CONFLICT" in json.loads(capsys.readouterr().out)["reason"]


def test_distinct_request_reservations_count_but_native_copies_do_not(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    turns = []
    for number in (1, 2):
        turns.append({"turn": number, "reservation_id": f"reserve-{number}",
            "request_ref": _save(root / f"request-{number}.json", {"same": "input"}),
            "response_ref": _save(root / f"response-{number}.json", {"usage": {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 3}})})
    native = {"schema_version": "task-agent-product-execution.v2", "run_id": "one",
        "provider_turns": turns, "completed_tool_records": [], "loop_result_ref": _save(root / "loop.json", {"call_records": {}})}
    ref = _save(root / "native" / "execution.json", native)
    copy = _save(root / "snapshot" / "execution.json", native)
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": "unknown", "execution_ref": ref, "snapshot_ref": copy})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    report = _query(config, capsys)
    assert len(report["executions"]) == 2
    assert report["cost"]["requests"] == 2
    assert report["cost"]["input_tokens"]["known_sum"] == 20
    assert report["cost"]["cached_input_tokens"]["known_sum"] == 6
    assert not report["cost"]["whole_task_cost_complete"]


def test_relative_source_identity_is_not_a_guessed_record_path(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    _save(root / "run_manifest.json", {"run_id": "one", "source_inputs": [{"path": "src/Real.cs", "sha256": "a" * 64}]})
    report = _query(config, capsys)
    assert any(link["status"] == "identity_only" and link["target"] == "src/Real.cs" for link in report["links"])
    assert not any(row["path"].endswith("Real.cs") for row in report["records"])


def test_postcheck_unknown_preserves_fact_without_becoming_delivery(tmp_path, capsys):
    config, root = _fixture(tmp_path, status="unknown")
    fact = _save(root / "checks" / "test-fact.json", {"schema": "csharp_repository_stage_fact_v1", "stage": "test_execution",
        "context": {"candidate_ref": "candidate-A", "candidate_source_id": "source-A"}, "execution": {"exit_code": 1}})
    _save(root / "checks" / "integrity-error.json", {"schema": "csharp_repository_integrity_v1", "status": "unknown", "stage_receipts": [fact]})
    report = _query(config, capsys)
    assert report["terminal"]["status"] == "unknown"
    assert any(row["stage"] == "test_execution" and row["execution"]["exit_code"] == 1 for row in report["stage_facts"])
    assert all(row["claim"] == "recorded_execution_fact_not_acceptance" for row in report["stage_facts"])


def test_separate_diagnostic_and_checkpoint_not_merged_as_model_turns(tmp_path, capsys):
    config, root = _fixture(tmp_path, status="unknown")
    checkpoint = _save(root / "session-checkpoints" / "checkpoint.json", {"schema": "task-agent-checkpoint.v2"})
    _save(root / "diagnostic" / "execution.json", {"schema_version": "task-agent-product-execution.v2", "run_id": "diagnostic_other",
        "provider_turns": [], "resumed_from_checkpoint_ref": checkpoint})
    report = _query(config, capsys)
    assert len(report["executions"]) == 1
    assert any(row["kind"] == "separate_run_not_merged" and row["run_id"] == "diagnostic_other" for row in report["lineage"])
    assert any(row.get("claim") == "checkpoint_presence_does_not_prove_resume" for row in report["lineage"])


def test_product_tool_catalog_exposes_only_scoped_public_feedback(session):
    names = {action.definition.name for action in session.actions()}
    assert "check_feedback_read" in names
    assert not names & {"query_run_evidence", "inspect_evidence", "read_record", "inspect_run"}


def test_explicit_result_resume_reference_is_verified_not_new_success(tmp_path, capsys):
    config, root = _fixture(tmp_path, status="unknown")
    checkpoint = _save(root / "session-checkpoints" / "checkpoint.json", {"schema": "task-agent-checkpoint.v2"})
    result = json.loads((root / "run_result.json").read_text())
    terminal = _save(root / "run_result.json", {**result, "resumed_from_checkpoint_ref": checkpoint})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    report = _query(config, capsys)
    recorded = next(row for row in report["lineage"] if row["kind"] == "resumed_from_checkpoint_ref")
    assert recorded["status"] == "reference_verified"
    assert recorded["claim"] == "explicit_recorded_lineage_not_new_execution_or_success"
    assert report["terminal"]["status"] == "unknown" and report["cost"]["requests"] == 0


def test_duplicate_response_alias_is_ambiguous_not_guessed_turn(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    call = {"call_id": "call-1", "name": "plan_update", "arguments": {"plan": {}}}
    response = {"tool_calls": [call]}
    turns = [{"turn": number, "reservation_id": f"r-{number}",
        "request_ref": _save(root / f"request-{number}.json", {"input": "same"}),
        "response_ref": _save(root / f"response-{number}.json", response)} for number in (1, 2)]
    native = {"schema_version": "task-agent-product-execution.v2", "run_id": "one", "provider_turns": turns,
        "loop_result_ref": _save(root / "loop.json", {"call_records": {}}),
        "completed_tool_records": [{**call, "response_ref": _save(root / "origin-response.json", response)}]}
    native_ref = _save(root / "native" / "execution.json", native)
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": "unknown", "execution_ref": native_ref})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    report = _query(config, capsys)
    assert report["executions"][0]["actions"][0]["turn"] is None
    assert any(row["status"] == "ambiguous" and row.get("call_id") == "call-1" for row in report["issues"])


def test_context_roots_do_not_expand_business_run_discovery(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    extra = tmp_path / "external"
    _save(extra / "duplicate" / "run_manifest.json", {"run_id": "one"})
    context = _save(extra / "context.json", {"run_id": "one", "model_calls": 0})
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence",
        "--evidence-root", str(extra), "--evidence-context", context["path"]]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] != "ambiguous"
    assert any(row.get("sha256") == context["sha256"] for row in report["records"])
    assert not any("duplicate" in row.get("path", "") for row in report["records"])


def test_context_native_pointer_cannot_be_overridden_by_run_id_label(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    other = _save(root / "other-native.json", {"schema_version": "task-agent-product-execution.v2", "run_id": "other"})
    context = _save(root / "wrong-context.json", {"run_id": "one", "native_execution_ref": other,
        "independent_acceptance_passed": True})
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence",
        "--evidence-context", context["path"]]) == 0
    report = json.loads(capsys.readouterr().out)
    assert any(issue["status"] == "unrelated_context" for issue in report["issues"])
    assert report["linked_executions"] == []
    assert report["outcome"]["result"]["confirmed_correct_deliveries"] == 0


def test_context_does_not_authorize_its_directory_or_hidden_references(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    extra = tmp_path / "external"
    private = _save(extra / "private.json", {"hidden_answer": "DO_NOT_READ"})
    context = _save(extra / "context.json", {"run_id": "one", "private_result_ref": private})
    args = ["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence",
        "--evidence-context", context["path"]]
    assert bridge.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert any(row["status"] == "forbidden" for row in report["issues"])
    assert bridge.main([*args, "--evidence-root", str(extra)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not any(row.get("path") == private["path"] for row in report["records"])
    assert "DO_NOT_READ" not in json.dumps(report)


def test_unrelated_context_and_derived_summary_are_not_adopted(tmp_path, capsys):
    config, root = _fixture(tmp_path)
    unrelated = _save(root / "unrelated.json", {"run_id": "other", "delivered": True})
    derived = _save(root / "run-outcomes" / "old.json", {"schema_version": "run_outcome_summary_v1",
        "run_id": "one", "native_execution_ref": {"path": "C:/forbidden/old.json", "sha256": "a" * 64}})
    manifest = json.loads((root / "run_manifest.json").read_text())
    _save(root / "run_manifest.json", {**manifest, "outcome_summary_ref": derived})
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence",
        "--evidence-context", unrelated["path"]]) == 0
    report = json.loads(capsys.readouterr().out)
    assert any(row["status"] == "unrelated_context" for row in report["issues"])
    assert not any(row.get("path") == derived["path"] for row in report["records"])
    assert not any("C:/forbidden" in str(row) for row in report["issues"])


@pytest.mark.parametrize("other_args", [[], ["--new-task"], ["--inspect-run", "one"]])
def test_context_flags_require_evidence_view_without_actions(tmp_path, capsys, other_args):
    config, root = _fixture(tmp_path)
    assert bridge.main(["--execution-config", str(config), "--evidence-context", str(root / "run_result.json"),
        *other_args]) == 2
    assert "INSPECTION" in json.loads(capsys.readouterr().out)["reason"]


@pytest.mark.parametrize("mismatch", [None, "candidate", "provider"])
def test_initial_feedback_requires_launch_candidate_and_actual_provider_match(tmp_path, capsys, mismatch):
    config, root = _fixture(tmp_path, status="unknown")
    parent = tmp_path / "parent"
    candidate = {"schema": "task-agent-candidate.v2", "candidate_source_id": "code-A"}
    raw = json.dumps(candidate, sort_keys=True).encode()
    candidate_ref = "v2/candidate/" + hashlib.sha256(raw).hexdigest() + ".json"
    _save(parent / "a" / candidate_ref, candidate)
    previous = _save(parent / "a" / "task-agent-execution" / "execution.json",
        {"schema_version": "task-agent-product-execution.v2", "run_id": "parent", "candidate_ref": candidate_ref,
         "provider_turns": [], "completed_tool_records": [], "loop_result_ref": _save(parent / "loop.json", {})})
    public = {"verification_summary": {"checks": 1, "failed_checks": 1},
        "observations": [{"check_id": "opaque", "expected": "Passed", "observed": {"actual": "Failed"}}]}
    feedback = _save(parent / "public.json", public)
    payload = {"public_check": {"summary": public["verification_summary"], "observations": public["observations"]}}
    content = {"task_requirements": {"original_request": "Investigate\n" + json.dumps(payload)}}
    provider = {"messages": [{"role": "user", "content": json.dumps(content)}]}
    if mismatch == "provider":
        provider = {"messages": [{"role": "user", "content": "public_check Failed"}]}
    turn = {"turn": 1, "reservation_id": "new-call", "request_ref": _save(root / "request.json", content),
        "provider_request_ref": _save(root / "http.json", provider),
        "provider_request_capture_boundary": "loopback_httpx_request_event",
        "response_ref": _save(root / "response.json", {"usage": {"input_tokens": 10, "output_tokens": 2}})}
    native = _save(root / "native" / "execution.json", {"schema_version": "task-agent-product-execution.v2",
        "run_id": "one", "provider_turns": [turn], "completed_tool_records": [], "loop_result_ref": _save(root / "loop.json", {})})
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": "unknown", "execution_ref": native})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    launch = _save(root / "launch.json", {"run_id": "one", "new_task": True, "original_task_resumed": False,
        "original_native_ref": previous, "starting_candidate_ref": candidate_ref,
        "starting_candidate_source_id": "wrong" if mismatch == "candidate" else "code-A", "public_feedback_ref": feedback})
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence",
        "--evidence-context", launch["path"], "--evidence-root", str(parent)]) == 0
    report = json.loads(capsys.readouterr().out)
    links = report["executions"][0]["feedback_links"]
    assert bool(links) is (mismatch is None)
    if links:
        assert links[0]["missing_fields"] == ["test_name", "assertion_details", "location"]
        assert links[0]["included_in_provider_request"] and not links[0]["adoption_or_repair_proven"]
    assert report["terminal"]["status"] == "unknown"
    assert len(report["linked_executions"]) == 1


@pytest.mark.parametrize("shape", ["stages", "execution_observations"])
def test_actual_trace_intervention_is_consumed_without_inferring_active_time(tmp_path, capsys, shape):
    from src.core.trace.execution_trace import ExecutionTrace

    config, root = _fixture(tmp_path, status="unknown")
    trace = ExecutionTrace("trace", "session", "test")
    with trace.activate():
        trace.record_intervention(event_id="rescue-1", run_id="one", actor="outer_codex",
            kind="technical_hint", cost_scope="task_runtime", subject_ref="request-1")
        trace.record_intervention(event_id="review-1", run_id="one", actor="human",
            kind="review", cost_scope="task_runtime", subject_ref="candidate-1", person_id="person-1",
            active_seconds=90, timing_basis="declared")
    document = trace.context.to_dict() if shape == "stages" else {
        "execution_observations": {row["data"]["id"]: row["data"] for row in trace.context.stages}}
    ref = _save(root / "trace.json", document)
    assert bridge.main(["--execution-config", str(config), "--inspect-run", "one", "--view", "evidence",
        "--evidence-context", ref["path"]]) == 0
    report = json.loads(capsys.readouterr().out)
    interventions = report["outcome"]["interventions"]
    assert interventions["observed_event_count"] == 2
    assert interventions["technical_assistance_observed"] is True
    assert interventions["human_active_seconds"] == 90
    assert interventions["total_technical_assistance"] is None
    assert interventions["human_active_time_scope"] == "observed_explicit_events_only_not_whole_task"


def test_learning_usage_follows_semantically_verified_response_not_copied_usage(tmp_path, capsys):
    from dataclasses import asdict

    from src.mod_agent.artifacts import ArtifactStore

    config, root = _fixture(tmp_path, status="unknown")
    native_ref = _save(root / "native" / "execution.json", {"schema_version": "task-agent-product-execution.v2",
        "run_id": "one", "status": "unknown", "provider_turns": [], "completed_tool_records": [],
        "loop_result_ref": _save(root / "loop.json", {})})
    terminal = _save(root / "run_result.json", {"run_id": "one", "status": "unknown", "execution_ref": native_ref})
    _save(root / "run_manifest.json", {"run_id": "one", "terminal_record": terminal})
    usage = {"input_tokens": 40, "output_tokens": 3, "cached_input_tokens": 10}
    response = ArtifactStore(root / "memory").write(feature_id="learning_response", artifact_type="strategy_experience",
        payload={"kind": "response", "schema_version": "1.0", "response": {"usage": usage}})
    _save(root / "native" / "terminal-learning-receipt-unit.json", {
        "original_execution_ref": native_ref, "model_calls": 1, "budget_after": {"consumed": 1},
        "call_receipts": [{"response_ref": asdict(response), "usage": usage}]})
    report = _query(config, capsys)
    cost = report["outcome"]["costs"]["task_runtime"]
    assert cost["known_learning_responses"] == 1
    assert cost["unknown_learning_calls"] == 0
    assert cost["total_tokens"] == 43
    # A forged receipt cannot turn absent or altered response bytes into known usage.
    _save(root / "native" / "terminal-learning-receipt-unit.json", {
        "original_execution_ref": native_ref, "model_calls": 1, "budget_after": {"consumed": 1},
        "call_receipts": [{"response_ref": {**asdict(response), "artifact_hash": "f" * 64}, "usage": usage}]})
    report = _query(config, capsys)
    cost = report["outcome"]["costs"]["task_runtime"]
    assert cost["known_learning_responses"] == 0
    assert cost["unknown_learning_calls"] == 1
