"""V2 checker uses real immutable state with controlled build/test processes."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.mod_agent.domain import csharp_repository_backend as owner
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.task_agent_v2_state import TaskAgentV2State
from tests.unit.test_csharp_repository_backend import _backend, _fake_execution, _profile, _write


def _prepared(tmp_path, *, extra_target=False):
    profile, source = _profile(tmp_path)
    if extra_target:
        _write(source / "Library/Other.cs", "public class Other {}")
        data = profile.data
        data["allowed_edit_paths"].append("Library/Other.cs")
        extended = _save(tmp_path / "extended-profile.json", data)
        profile = owner.CSharpRepositoryProfile(extended)
    backend = _backend(tmp_path, profile, source)
    store = tmp_path / "artifacts"
    initial = object.__new__(TaskAgentV2State)
    initial.store_root, initial.root = store, store / "v2"
    authority_ref = initial.save("authority", {"scope": "file", "operation": "modify_existing",
        "read_roots": {"target": str(source)}, "allowed_edit_paths": profile.data["allowed_edit_paths"]})
    acceptance = initial.save("acceptance", {"verification_identity": backend.verification_identity()})
    state = TaskAgentV2State.create(store, "test-v2", source,
        allowed_edit_paths=profile.data["allowed_edit_paths"], authority_ref=authority_ref,
        contract_fields={"execution_profile_ref": profile.reference, "public_acceptance_ref": acceptance})
    view = state.read_source("baseline", "Library/Arithmetic.cs")
    binding = state.add_binding(state.registry_ref, view["source_view_ref"], "Library/Arithmetic.cs")
    response = tmp_path / "response.json"
    _write(response, '{"kind":"candidate_submit"}')
    candidate = state.submit("baseline", [{"binding_id": binding["binding_id"],
        "source_view_ref": view["source_view_ref"], "before_text": "=> value;", "after_text": "=> value * 2;"}],
        model_origin={"response_ref": _ref(response)})
    context = {"schema_version": "task_agent_candidate_check_v2", "artifact_store_root": str(store),
        "feature_id": "test-v2", "task_contract_ref": state.contract_ref, "binding_registry_ref": state.registry_ref,
        "candidate_ref": candidate["candidate_ref"], "candidate_source_id": candidate["candidate_source_id"],
        "generated_changes": state.changes_from_baseline(candidate["candidate_ref"]),
        "source_snapshot_sha256": owner.canonical_hash(state.load(state.contract["baseline_manifest_ref"])),
        "model_response": {"path": str(response), "output_sha256": _ref(response)["sha256"]},
        "verification_identity": backend.verification_identity()}
    return backend, state, context


def _validate(result, state, context):
    return owner.validate_candidate_report_v2(result["report_ref"],
        **{key: context[key] for key in ("task_contract_ref", "binding_registry_ref", "candidate_ref",
            "candidate_source_id", "verification_identity")},
        public_acceptance_ref=state.contract["public_acceptance_ref"])


def test_v2_candidate_build_and_product_report_preserve_identity_without_plan(tmp_path, monkeypatch):
    backend, state, context = _prepared(tmp_path)
    _fake_execution(monkeypatch)
    def reject_v1(**kwargs):
        raise AssertionError("v2 must not materialize a v1 Plan")
    monkeypatch.setattr(owner, "materialize_candidate", reject_v1)
    original = (backend.source_root / "Library/Arithmetic.cs").read_bytes()
    result = backend(context)
    validated = _validate(result, state, context)
    assert owner.validate_candidate_report_v2(result, context=context, require_pass=True) == validated
    assert validated["status"] == "passed"
    assert validated["candidate_source_id"] == context["candidate_source_id"]
    assert "implementation_plan_ref" not in validated
    execution = owner.read_bound_json(result["verification_ref"])
    assert execution["materialized_code_identity"] is None
    assert execution["candidate_source_id"] == context["candidate_source_id"]
    assert "=> value * 2;" in (Path(execution["source_root"]) / "Library/Arithmetic.cs").read_text()
    assert (backend.source_root / "Library/Arithmetic.cs").read_bytes() == original
    assert result["verification_summary"]["checks"] == 1


@pytest.mark.parametrize("field,value,code", [
    ("candidate_source_id", "wrong", "CANDIDATE_IDENTITY_CHANGED"),
    ("source_snapshot_sha256", "wrong", "SOURCE_SNAPSHOT_MISMATCH"),
    ("verification_identity", {}, "VERIFICATION_IDENTITY_CHANGED"),
])
def test_v2_rejects_stale_candidate_context_before_execution(tmp_path, monkeypatch, field, value, code):
    backend, _, context = _prepared(tmp_path)
    calls = _fake_execution(monkeypatch)
    context[field] = value
    with pytest.raises(ValueError, match=code):
        backend(context)
    assert not calls


def test_v2_rejects_changed_baseline_even_on_authorized_path(tmp_path, monkeypatch):
    backend, _, context = _prepared(tmp_path)
    calls = _fake_execution(monkeypatch)
    _write(backend.source_root / "Library/Arithmetic.cs", "other source")
    with pytest.raises(ValueError, match="BASELINE_SOURCE_CHANGED"):
        backend(context)
    assert not calls


@pytest.mark.parametrize("settings", [{"tests": []}, {"missing": True}, {"stale": True}, {"test_exit": 2}])
def test_v2_rejects_invalid_public_test_execution(tmp_path, monkeypatch, settings):
    backend, state, context = _prepared(tmp_path)
    _fake_execution(monkeypatch, **settings)
    result = backend(context)
    assert result["status"] == "infrastructure_failed"
    with pytest.raises(ValueError, match="CANDIDATE_CHECK_NOT_PASSED"):
        _validate(result, state, context)


def test_v2_business_failure_is_actionable_and_not_handoff_success(tmp_path, monkeypatch):
    backend, state, context = _prepared(tmp_path)
    _fake_execution(monkeypatch, tests=[{"name": "double", "passed": False, "expected": 6, "actual": 3}])
    result = backend(context)
    assert result["status"] == "behavior_failed"
    assert result["owner_binding_ids"]
    assert result["observations"][0]["observed"]["actual"] == 3
    assert owner.validate_candidate_report_v2(result, context=context)["status"] == "behavior_failed"
    with pytest.raises(ValueError, match="CANDIDATE_CHECK_NOT_PASSED"):
        _validate(result, state, context)


def test_v2_handoff_rejects_an_old_report_under_new_candidate_identity(tmp_path, monkeypatch):
    backend, state, context = _prepared(tmp_path)
    _fake_execution(monkeypatch)
    result = backend(context)
    tampered = deepcopy(result)
    tampered.pop("report_ref")
    tampered["candidate_source_id"] = "different-candidate"
    result["report_ref"] = _save(tmp_path / "misbound-report.json", tampered)
    changed = {**context, "candidate_source_id": "different-candidate"}
    with pytest.raises(ValueError, match="EXECUTION_CANDIDATE_MISMATCH"):
        _validate(result, state, changed)


def test_v2_handoff_rechecks_public_test_original(tmp_path, monkeypatch):
    backend, state, context = _prepared(tmp_path)
    _fake_execution(monkeypatch)
    result = backend(context)
    execution = owner.read_bound_json(result["verification_ref"])
    _write(Path(execution["test_execution"]["report_ref"]["path"]), "{}")
    with pytest.raises(ValueError, match="TEST_REPORT_CHANGED"):
        _validate(result, state, context)


def test_v2_rejects_changed_final_diff_before_build(tmp_path, monkeypatch):
    backend, _, context = _prepared(tmp_path)
    calls = _fake_execution(monkeypatch)
    context["generated_changes"][0]["after_text"] = "unrelated content"
    with pytest.raises(ValueError, match="CHANGE_CONTENT_MISMATCH"):
        backend(context)
    assert not calls


def test_v2_rejects_other_model_response_even_when_it_is_hash_bound(tmp_path, monkeypatch):
    backend, _, context = _prepared(tmp_path)
    calls = _fake_execution(monkeypatch)
    different = tmp_path / "different-response.json"
    _write(different, '{"kind":"different"}')
    context["model_response"] = {"path": str(different), "output_sha256": _ref(different)["sha256"]}
    with pytest.raises(ValueError, match="CANDIDATE_RESPONSE_MISMATCH"):
        backend(context)
    assert not calls


def test_v2_added_binding_does_not_change_existing_candidate_check_identity(tmp_path, monkeypatch):
    backend, state, context = _prepared(tmp_path, extra_target=True)
    _fake_execution(monkeypatch)
    build = owner.nuget._run_nuget_isolated_build
    def compile_both(profile):
        result = build(profile)
        args = [part for part in profile.build_command if "RepositoryCompileInputs=" in part]
        if args:
            capture = Path(args[0].split("=", 1)[1])
            source = profile.working_directory
            _write(capture, capture.read_text() + f"{source / 'Library/Library.csproj'}|{source / 'Library/Other.cs'}\n")
        return result
    monkeypatch.setattr(owner.nuget, "_run_nuget_isolated_build", compile_both)
    view = state.read_source(state.current_revision_ref, "Library/Other.cs")
    state.add_binding(state.registry_ref, view["source_view_ref"], "Library/Other.cs")
    assert state.registry_ref != context["binding_registry_ref"]
    context["binding_registry_ref"] = state.registry_ref
    result = backend(context)
    assert result["candidate_source_id"] == context["candidate_source_id"]
    assert owner.validate_candidate_report_v2(result, context=context, require_pass=True)["status"] == "passed"
