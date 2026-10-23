"""Public originals retain diagnostic detail and cannot become arbitrary file reads."""
import pytest

from src.mod_agent.public_failure_feedback import failed_checks_text, project_failed_checks
from src.mod_agent.task_agent_v2_tools import TaskAgentSessionV2
from tests.unit.test_task_agent_v2_admission import session as session
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def test_original_details_are_bounded_with_actual_source_location(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    file = source / "Test.cs"
    file.write_text("line1\nline2\n")
    details = "Assertion actual=old expected=new\n in " + str(file) + ":line 2\n" + "x" * 2500
    row = {"check_id": "test-hash", "name": "Tests.UpdateExisting", "expected": "Passed", "actual": "Failed",
        "passed": False, "details": details}
    ref = {"path": "public.trx", "sha256": "a" * 64}
    output = project_failed_checks([row], raw_report_ref=ref, source_root=source)[0]
    assert output["test_name"] == row["name"] and output["details"] == details[:2000]
    assert output["details_truncated"] and output["details_total_chars"] == len(details)
    assert output["raw_report_ref"] == ref
    assert output["source_locations"] == [{"path": "Test.cs", "line": 2, "origin": "reported_public_stack_trace"}]
    assert "x" * 2500 in failed_checks_text([row])
    simple = project_failed_checks([{k: v for k, v in row.items() if k not in {"name", "details"}}],
        raw_report_ref=ref, source_root=source)[0]
    assert "test_name" not in simple and "details" not in simple and "source_locations" not in simple


def test_feedback_read_rejects_nonfeedback_and_unobserved_check(session):
    invalid = session.state.save("probe", {"schema": "task-agent-probe.v2", "text": "private data"})
    with pytest.raises(ValueError, match="PUBLIC_FEEDBACK_SCOPE_DENIED"):
        session._feedback_record(invalid)
    fake_check = session.state.save("check", {"schema": "task-agent-check.v2"})
    record = session.state.save("public-check-feedback", {"schema": "task-agent-public-check-feedback.v2",
        "task_contract_ref": session.state.contract_ref, "check_ref": fake_check})
    with pytest.raises(ValueError, match="PUBLIC_FEEDBACK_CHECK_NOT_OBSERVED"):
        session._feedback_record(record)


def test_feedback_reader_returns_exact_bounded_chunk_and_rejects_out_of_range(session, monkeypatch):
    from src.mod_agent.task_agent_loop import TaskActionInputError

    record = {"candidate_ref": "candidate", "candidate_source_id": "identity", "check_ref": "check", "text": "真实失败" * 3000}
    monkeypatch.setattr(session, "_feedback_record", lambda _ref: record)
    monkeypatch.setattr(session, "_origin", lambda *_args: {"call_id": "read"})
    output = session._execute_owned("check_feedback_read", {"feedback_ref": "bound", "offset": 1, "limit": 7})
    assert output["text"] == record["text"][1:8] and output["next_offset"] == 8 and output["truncated"]
    action = next(item for item in TaskAgentSessionV2.actions(session) if item.definition.name == "check_feedback_read")
    with pytest.raises(TaskActionInputError, match="SOURCE_ARGUMENTS_INVALID"):
        action.prepare({"feedback_ref": "bound", "offset": len(record["text"]) + 1})


@pytest.mark.parametrize("extra_in_fixed", [False, True])
def test_saved_projection_rejects_added_tests_against_original_worker_inputs(monkeypatch, extra_in_fixed):
    from types import SimpleNamespace

    from src.mod_agent import public_failure_feedback as feedback
    from src.mod_agent import task_agent_execution

    frozen = {"source_paths": ["A.cs", "PublicTests.cs"]}
    binding = {"configuration_ref": "worker"}
    records = {
        "native": {"schema_version": "task-agent-product-execution.v2", "task_contract_ref": "contract",
            "candidate_ref": "candidate", "worker_binding": binding},
        "worker": {"behavior_evaluator": {"factory_name": "make_csharp_repository_evaluator",
            "kwargs": {"profile_ref": "profile", "expected_fixed_inputs": frozen}}},
    }
    monkeypatch.setattr(feedback, "read_bound_json", records.__getitem__)
    monkeypatch.setattr(task_agent_execution, "_worker_binding", lambda _ref: binding)
    report = {"fixed_inputs": {"source_paths": [*frozen["source_paths"], "HiddenTests.cs"]} if extra_in_fixed else frozen,
        "source_inputs": [{"path": path} for path in [*frozen["source_paths"], "HiddenTests.cs"]]}
    monkeypatch.setattr(feedback, "validated_public_checks", lambda _ref: (report, []))
    current = SimpleNamespace(contract_ref="contract", contract={"execution_profile_ref": "profile"},
        validate_candidate=lambda _ref: {"candidate_source_id": "identity"})
    with pytest.raises(ValueError, match="PUBLIC_FEEDBACK_INPUTS_CHANGED"):
        feedback.project_saved_candidate_failure(report_ref="report", state=current,
            candidate_ref="candidate", original_execution_ref="native")
