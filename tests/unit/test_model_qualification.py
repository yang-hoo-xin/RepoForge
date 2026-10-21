"""Qualification mechanics over synthetic production-path receipts, never real model credentials."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.mod_agent.cli import main
from src.mod_agent.evaluation.quality_policy import QualityGatePolicy
from src.mod_agent.fresh_planning_continuation import (
    _patch_attempt_identity,
    run_fresh_patch_continuation,
)
from src.mod_agent.model_qualification import (
    QUALIFICATION_SCHEMA,
    ROUTING_SCHEMA,
    SUITE_SCHEMA,
    ModelQualificationError,
    _verify_materialized_response,
    canonical_hash,
    resolve_qualified_patch_profile,
    worker_implementation_identity,
)
from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration
from tests.unit.test_fresh_planning_continuation import _feature_payload
from tests.unit.test_patch_repair_binding import _binding_case
from tests.unit.test_patch_repair_cli import (
    _FACTORY_SOURCE,
    _install_transport,
    _public_case,
    _sha,
    _write_json,
)

_MATRIX_FACTORY = _FACTORY_SOURCE.replace(
    '        path = self.output_root / ("report-%03d.json" % self.calls)',
    '''        check = {"check": "repair_marker", "passed": passed, "expected": True, "actual": passed}
        preserved = all("KeepExistingFeedback();" in item["patch"]["after_text"] for item in context["generated_changes"])
        preservation = {"check": "preservation", "passed": preserved, "expected": True, "actual": preserved}
        matrix_checks = [check, preservation]
        matrix = {"checks": matrix_checks, "failures": [item for item in matrix_checks if not item["passed"]],
                  "executions": 1, "materialized_code_identity": context["materialized_code_identity"]}
        matrix_path = self.output_root / ("matrix-%03d.json" % self.calls)
        matrix_raw = json.dumps(matrix, sort_keys=True).encode()
        matrix_path.write_bytes(matrix_raw)
        report["verification_ref"] = {"path": str(matrix_path), "sha256": digest(matrix_raw)}
        report["verification_summary"] = {"checks": 2, "failed_checks": len(matrix["failures"]), "executions": 1}
        path = self.output_root / ("report-%03d.json" % self.calls)''',
)


def _ref(path):
    return {"path": str(path), "sha256": _sha(Path(path).read_bytes())}


def _matrix_case(tmp_path, monkeypatch, suffix):
    feature = _feature_payload()
    feature["feature_id"] = f"qualification_task_{suffix}"
    feature["user_request"] += f" Fixed synthetic task identity {suffix}."
    with monkeypatch.context() as patch:
        patch.setattr("tests.unit.test_fresh_planning_continuation._feature_payload", lambda: deepcopy(feature))
        case = _public_case(tmp_path)
    case.factory.write_text(_MATRIX_FACTORY, encoding="utf-8")
    case.config["behavior_evaluator"]["factory_ref"] = _ref(case.factory)
    _write_json(case.config_path, case.config)
    return case


@pytest.fixture
def evidence(tmp_path, monkeypatch, capsys):
    cases, trials, reports = [], [], []
    for index in range(3):
        case = _matrix_case(tmp_path / f"task-{index}", monkeypatch, str(index))
        with monkeypatch.context() as patch:
            transport = _install_transport(patch, tmp_path / f"wire-{index}", scenario="direct_pass")
            assert main(case.argv) == 0
            output = json.loads(capsys.readouterr().out)
            assert len(transport.chat_requests) == 2
        journal_path = Path(output["attempt_receipt"]["path"])
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        factory = journal["factory_receipt"]
        evaluation = factory["patch_agent_loop"]["behavior_evaluations"][-1]
        cases.append({"case_id": f"fixed-task-{index}", "artifact_store_root": str(case.inputs.store.root),
                      "workspace_root": str(case.inputs.workspace),
                      "feature_spec_ref": journal["selected_artifacts"]["feature_spec"],
                      "expected_check_ids": ["repair_marker", "preservation"],
                      "verification_identity": evaluation["verification_identity"]})
        trials.append({"case_id": f"fixed-task-{index}", "attempt_ref": _ref(journal_path)})
        reports.append((journal_path, journal))
    scope = {"target_mod": "SyntheticMod"}
    suite_path = _write_json(tmp_path / "suite.json", {"schema_version": SUITE_SCHEMA, "scope": scope, "cases": cases})
    policy = QualityGatePolicy("quality_gate_policy_v1", "all-fixed-patch-tasks", ("task_success_rate",), {"task_success_rate": 1.0})
    qualification = {"schema_version": QUALIFICATION_SCHEMA, "role": "repair", "scope": scope,
                     "profile": factory["profile"], "model_digest": factory["model_identity"]["digest"],
                     "worker_policy_sha256": factory["qualification_execution_identity"]["worker_policy_sha256"],
                     "implementation_sha256": factory["qualification_execution_identity"]["implementation_sha256"],
                     "task_suite_ref": _ref(suite_path), "attempts": trials, "quality_policy": policy.to_dict()}
    qualification_path = _write_json(tmp_path / "qualification.json", qualification)
    routing = {"schema_version": ROUTING_SCHEMA, "routes": {"repair": {"qualification_ref": _ref(qualification_path)}}}
    routing_path = _write_json(tmp_path / "routing.json", routing)
    return {"path": routing_path, "routing": routing, "qualification_path": qualification_path,
            "qualification": qualification, "suite_path": suite_path, "cases": cases,
            "trials": trials, "reports": reports, "policy": PatchWorkerConfiguration.load(case.config_path).policy}


def _refresh(evidence):
    _write_json(evidence["qualification_path"], evidence["qualification"])
    evidence["routing"]["routes"]["repair"]["qualification_ref"] = _ref(evidence["qualification_path"])
    _write_json(evidence["path"], evidence["routing"])


def _resolve(evidence, **kwargs):
    return resolve_qualified_patch_profile(routing_path=evidence["path"],
        role=kwargs.get("role", "repair"), target_mod=kwargs.get("target_mod", "SyntheticMod"),
        worker_policy=evidence["policy"])


def test_selection_uses_three_distinct_executed_tasks_and_the_existing_profile(evidence):
    selection = _resolve(evidence)
    assert selection.profile.to_dict() == evidence["qualification"]["profile"]
    assert selection.receipt["distinct_task_count"] == 3
    assert selection.receipt["metrics"] == {"task_success_rate": 1.0}
    assert selection.receipt["authorizes_disclosure"] is False
    assert selection.receipt["authorizes_apply"] is False
    assert selection.receipt["automatic_fallback"] is False
    assert selection.receipt["implementation_sha256"] == worker_implementation_identity()["implementation_sha256"]


@pytest.mark.parametrize("mutation", ["final_failure", "missing_version", "duplicate_task", "scope", "no_role", "matrix_missing", "matrix_trimmed", "matrix_code_identity", "raw_relabelled"])
def test_unqualified_or_incomplete_facts_do_not_select_an_alternative(evidence, mutation):
    if mutation == "scope":
        with pytest.raises(ModelQualificationError):
            _resolve(evidence, target_mod="AnotherMod")
        return
    if mutation == "no_role":
        with pytest.raises(ModelQualificationError) as error:
            _resolve(evidence, role="initial_patch")
        assert error.value.status == "pending"
        return
    if mutation == "duplicate_task":
        suite = json.loads(evidence["suite_path"].read_text(encoding="utf-8"))
        suite["cases"][1]["feature_spec_ref"] = suite["cases"][0]["feature_spec_ref"]
        _write_json(evidence["suite_path"], suite)
        evidence["qualification"]["task_suite_ref"] = _ref(evidence["suite_path"])
    else:
        path, journal = evidence["reports"][0]
        if mutation == "missing_version":
            del journal["factory_receipt"]["qualification_execution_identity"]
        elif mutation == "raw_relabelled":
            retained = journal["factory_receipt"]["model_response"]
            raw_path = Path(retained["path"])
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw["changes"][0]["after_text"] += "\n// relabelled response"
            _write_json(raw_path, raw)
            retained["output_sha256"] = _sha(raw_path.read_bytes())
        else:
            final = journal["factory_receipt"]["patch_agent_loop"]["behavior_evaluations"][-1]
            report_path = Path(final["report_ref"]["path"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if mutation == "matrix_missing":
                Path(report["verification_ref"]["path"]).unlink()
            else:
                matrix_path = Path(report["verification_ref"]["path"])
                matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
                if mutation == "matrix_trimmed":
                    matrix["checks"] = matrix["checks"][:1]
                    report["verification_summary"]["checks"] = 1
                elif mutation == "matrix_code_identity":
                    matrix["materialized_code_identity"]["code_sha256"] = "sha256:" + "0" * 64
                else:
                    matrix["checks"][0].update(passed=False, actual=False)
                    matrix["failures"] = [row for row in matrix["checks"] if not row["passed"]]
                _write_json(matrix_path, matrix)
                report.update(verification_ref=_ref(matrix_path))
                if mutation == "final_failure":
                    report["status"] = "behavior_failed"
                    report["verification_summary"]["failed_checks"] = 1
                _write_json(report_path, report)
                final.update(status=report["status"], report_ref=_ref(report_path))
        _write_json(path, journal)
        evidence["qualification"]["attempts"][0]["attempt_ref"] = _ref(path)
    _refresh(evidence)
    with pytest.raises(ModelQualificationError) as error:
        _resolve(evidence)
    if mutation in {"missing_version", "matrix_missing"}:
        assert error.value.status == "pending"


def test_qualified_flag_and_external_profile_cannot_authorize_selection(evidence):
    evidence["qualification"]["qualified"] = True
    _refresh(evidence)
    with pytest.raises(ModelQualificationError):
        _resolve(evidence)
    del evidence["qualification"]["qualified"]
    evidence["qualification"]["profile"]["base_url"] = "https://external.example"
    _refresh(evidence)
    with pytest.raises(ModelQualificationError):
        _resolve(evidence)


def test_public_cli_consumes_qualified_profile_and_rejects_explicit_conflict(evidence, tmp_path, monkeypatch, capsys):
    case = _matrix_case(tmp_path / "consumer", monkeypatch, "consumer")
    for flag in ("--ollama-model", "--ollama-context-window", "--ollama-output-budget", "--ollama-timeout-seconds"):
        index = case.argv.index(flag)
        del case.argv[index:index + 2]
    case.argv.extend(["--qualified-model-routing", str(evidence["path"])])
    transport = _install_transport(monkeypatch, tmp_path / "consumer-wire", scenario="direct_pass")
    assert main(case.argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(transport.chat_requests) == 2
    journal = json.loads(Path(output["attempt_receipt"]["path"]).read_text(encoding="utf-8"))
    assert journal["factory_receipt"]["model_qualification"]["status"] == "qualified"
    assert journal["qualified_model_routing_ref"] == _ref(evidence["path"])
    prior_calls = len(transport.requests)
    assert main(case.argv) == 0
    capsys.readouterr()
    assert len(transport.requests) == prior_calls
    conflict = [*case.argv, "--ollama-model", "some-other-model"]
    assert main(conflict) != 0
    blocked = json.loads(capsys.readouterr().out)
    assert "MODEL_QUALIFIED_PROFILE_ARGUMENT_CONFLICT" in blocked["reason"]
    assert len(transport.requests) == prior_calls
    past_journal = evidence["reports"][0][1]
    final_report = past_journal["factory_receipt"]["patch_agent_loop"]["behavior_evaluations"][-1]
    report = json.loads(Path(final_report["report_ref"]["path"]).read_text(encoding="utf-8"))
    Path(report["verification_ref"]["path"]).write_text('{"changed":true}', encoding="utf-8")
    assert main(case.argv) != 0
    capsys.readouterr()
    assert len(transport.requests) == prior_calls
    api_request = replace(case.request, attempt_id=case.argv[case.argv.index("--attempt-id") + 1],
        worker_configuration_ref=_ref(case.config_path), worker_profile_sha256=canonical_hash(evidence["qualification"]["profile"]),
        qualified_model_routing_ref=_ref(evidence["path"]))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A failed cached qualification cannot generate or verify again.")
    cached = run_fresh_patch_continuation(api_request, patch_change_factory=forbidden,
        dry_run_factory=forbidden, max_model_calls=case.config["policy"]["max_total_model_turns"])
    assert cached.status == "blocked"
    assert cached.first_failure["code"] in {"MODEL_NOT_QUALIFIED", "MODEL_QUALIFICATION_PENDING"}
    assert len(transport.requests) == prior_calls


def test_model_digest_drift_blocks_generation_without_fallback(evidence):
    selection = _resolve(evidence)
    profile = selection.profile
    with pytest.raises(ModelQualificationError):
        selection.validate_live_identity({"name": profile.model, "model": profile.model,
            "digest": "b" * 64, "base_url": profile.base_url, "details": {}, "capabilities": []})


def test_worker_fingerprint_does_not_depend_on_git_or_documentation():
    identity = worker_implementation_identity()
    assert not any(path.endswith(".md") or ".git" in path for path in identity["files"])
    assert identity["implementation_sha256"] == canonical_hash(identity["files"])


def test_proposal_cannot_keep_a_valid_response_hash_while_changing_its_actual_code(evidence):
    journal = evidence["reports"][0][1]
    def read_data(ref):
        return json.loads(Path(ref["path"]).read_text(encoding="utf-8"))["data"]
    selected = journal["selected_artifacts"]
    proposal = read_data(journal["terminal_result"]["patch_proposal_artifact"])
    parsed = json.loads(Path(journal["factory_receipt"]["model_response"]["path"]).read_text(encoding="utf-8"))
    expected = journal["factory_receipt"]["patch_agent_loop"]["behavior_evaluations"][-1]["generated_changes_sha256"]
    arguments = {"feature": read_data(selected["feature_spec"]), "evidence": read_data(selected["evidence_pack"]),
                 "plan": read_data(selected["implementation_plan"]), "parsed_output": parsed,
                 "proposal": proposal, "expected_changes_hash": expected}
    _verify_materialized_response(**arguments)
    proposal["diff_intent"][0]["patch"]["after_text"] += "\n// different bytes"
    with pytest.raises(ModelQualificationError, match="Proposal code differs"):
        _verify_materialized_response(**arguments)


def test_optional_routing_ref_changes_attempt_identity_without_changing_the_profile(tmp_path):
    request = _binding_case(tmp_path / "case")["request"]
    route = _write_json(tmp_path / "route.json", {"schema_version": ROUTING_SCHEMA, "routes": {}})
    selected = replace(request, qualified_model_routing_ref=_ref(route))
    assert _patch_attempt_identity(selected, 5) != _patch_attempt_identity(request, 5)
    assert selected.worker_profile_sha256 == request.worker_profile_sha256
