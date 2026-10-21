"""Repository contract survives ordinary product and Build checkpoints."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.csharp_repository_delivery import (
    repository_binding_from_request,
    validate_repository_binding,
    validate_repository_changes,
)
from src.mod_agent.errors import LocalEntryError
from src.mod_agent.product_execution import (
    ProductExecutionInputBindingV1,
    ProductExecutionRequestV1,
)
from src.mod_agent.product_executor import (
    _verify_source_revision_binding,
    source_snapshot,
    source_snapshot_sha256,
)
from src.mod_agent.proposal.patch_agent_loop import PatchAgentLoopPolicy
from tests.unit.test_csharp_repository_backend import _profile
from tests.unit.test_product_execution import _request


def _binding(tmp_path):
    profile, source = _profile(tmp_path)
    fixed = profile.bind(source)
    worker = tmp_path / "repository-worker.json"
    worker.write_text(json.dumps({"schema_version": "patch_worker_configuration_v1",
        "policy": asdict(PatchAgentLoopPolicy()), "enable_recovery_loop": True,
        "behavior_evaluator": {"factory_ref": backend._ref(backend.__file__),
            "factory_name": "make_csharp_repository_evaluator", "kwargs": {
                "profile_ref": profile.reference, "source_root": str(source), "workspace_root": str(tmp_path),
                "output_root": str(tmp_path / "checks"), "expected_fixed_inputs": fixed}}}), encoding="utf-8")
    return profile, source, {"configuration_ref": profile.reference, "fixed_inputs": fixed,
        "target_relative_root": "source", "worker_configuration_ref": backend._ref(worker)}


def test_optional_fifth_binding_roundtrips_only_for_csharp_compile_scope(tmp_path):
    original, _ = _request(tmp_path)
    binding = ProductExecutionInputBindingV1.create("repository_build_and_test", {"declared": "schema test only"})
    selected = replace(original, target_domain="csharp_repository",
        input_bindings=(*original.input_bindings, binding))
    assert ProductExecutionRequestV1.from_dict(selected.to_dict()) == selected
    assert repository_binding_from_request(selected) == {"declared": "schema test only"}
    assert repository_binding_from_request(original) is None
    with pytest.raises(ValueError):
        replace(original, input_bindings=(*original.input_bindings, binding))
    with pytest.raises(ValueError):
        replace(selected, requested_completion_scope="runtime_workflow")


def test_explicit_repository_profile_resolves_project_ambiguity_without_bypassing_git_binding(tmp_path):
    profile, source, _ = _binding(tmp_path)
    snapshot = source_snapshot(source)
    kwargs = {"source": source, "snapshot": snapshot, "snapshot_sha256": source_snapshot_sha256(snapshot)}
    with pytest.raises(LocalEntryError, match="Multiple source projects"):
        _verify_source_revision_binding(**kwargs, binding=None)
    _verify_source_revision_binding(**kwargs, binding=None, repository_profile_ref=profile.reference)
    with pytest.raises(LocalEntryError, match="Generic source revision binding shape"):
        _verify_source_revision_binding(**kwargs, binding={"binding_version": "product_source_revision_binding_v2"},
            repository_profile_ref=profile.reference)


def test_repository_delivery_preserves_fixed_inputs_and_exact_edit_paths(tmp_path):
    profile, source, binding = _binding(tmp_path)
    assert validate_repository_binding(binding, tmp_path)[0].reference == profile.reference
    validate_repository_changes(binding, tmp_path, [{"path": "source/Library/Arithmetic.cs"}])
    with pytest.raises(ValueError, match="EDIT_SCOPE_CHANGED"):
        validate_repository_changes(binding, tmp_path, [{"path": "source/tests/Program.cs"}])
    (source / "tests/Program.cs").write_text("changed test", encoding="utf-8")
    with pytest.raises(ValueError, match="FIXED_INPUTS_CHANGED"):
        validate_repository_binding(binding, tmp_path)


def test_repository_delivery_checks_original_and_executing_factory_bytes(tmp_path):
    _, _, binding = _binding(tmp_path)
    path = Path(binding["worker_configuration_ref"]["path"])
    value = json.loads(path.read_bytes())
    copied = tmp_path / "retained-backend.py"
    copied.write_bytes(Path(backend.__file__).read_bytes())
    value["behavior_evaluator"]["factory_ref"] = backend._ref(copied)
    path.write_text(json.dumps(value), encoding="utf-8")
    binding["worker_configuration_ref"] = backend._ref(path)
    validate_repository_binding(binding, tmp_path)
    copied.write_bytes(copied.read_bytes() + b"\n# drift")
    with pytest.raises(ValueError, match="CHECKER_CHANGED"):
        validate_repository_binding(binding, tmp_path)
    value["behavior_evaluator"]["factory_ref"] = backend._ref(copied)
    path.write_text(json.dumps(value), encoding="utf-8")
    binding["worker_configuration_ref"] = backend._ref(path)
    with pytest.raises(ValueError, match="CHECKER_CHANGED"):
        validate_repository_binding(binding, tmp_path)


def test_repository_delivery_selects_real_origin_and_never_falls_back_from_broken_native_claim(tmp_path, monkeypatch):
    from src.mod_agent.domain import csharp_repository_delivery as owner
    from src.mod_agent.patch_worker_configuration import PatchWorkerConfiguration

    _, _, binding = _binding(tmp_path)
    worker = PatchWorkerConfiguration.load(binding["worker_configuration_ref"]["path"])
    identity = worker.behavior_evaluator().verification_identity()
    invoked = []
    monkeypatch.setattr(owner, "_validate_workflow_candidate",
        lambda **kwargs: invoked.append(kwargs) or {"source": "verified-workflow-fixture"})
    arguments = dict(binding=binding, workspace_root=tmp_path, store=None, initial={},
        verification={}, report={"verification_identity": identity}, proposal={"context_pack": {}})
    assert owner.validate_repository_candidate(**arguments)["source"] == "verified-workflow-fixture"
    with pytest.raises(ValueError, match="NATIVE_EXECUTION_REQUIRED"):
        owner.validate_repository_candidate(**{**arguments,
            "proposal": {"context_pack": {"task_agent_lineage": {"execution_ref": "missing"}}}})
    assert len(invoked) == 1


def test_workflow_delivery_rejects_replaced_report_before_receipt_selection(tmp_path):
    from src.mod_agent.domain import csharp_repository_delivery as owner

    _, _, binding = _binding(tmp_path)
    original = {"status": "passed"}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="WORKFLOW_REPORT_CHANGED"):
        owner._validate_workflow_candidate(binding=binding, root=tmp_path, store=None, initial={},
            verification=backend._ref(path), report={**original, "observations": ["changed"]},
            proposal={}, worker=None, current_identity={})


def test_public_build_continuation_cannot_replace_repository_tests_with_callback(tmp_path, monkeypatch):
    from src.mod_agent import reviewed_patch_continuation as owner

    monkeypatch.setattr(owner, "_reviewed_build_inputs", lambda *_: (
        {}, {"repository_build_and_test": {"bound": "original contract"}}, None, tmp_path))
    with pytest.raises(ValueError, match="REPOSITORY_BUILD_FACTORY_OVERRIDE_FORBIDDEN"):
        owner.continue_reviewed_patch(artifact_store_root=tmp_path / "a", checkpoint_ref=None, resume_payload=None,
            allow_build=True, build_result_factory=lambda *_: pytest.fail("Weaker Build must not run"))


@pytest.mark.parametrize("checks, expected_ids", [
    ([{"name": "public arithmetic assertion", "passed": True}], ["public arithmetic assertion"]),
    ([{"name": "same display", "check_id": "case-a", "passed": True},
      {"name": "same display", "check_id": "case-b", "passed": True}], ["case-a", "case-b"]),
    ([{"name": "first", "check_id": "duplicate", "passed": True},
      {"name": "second", "check_id": "duplicate", "passed": True}], None),
])
def test_delivery_projection_reads_repository_test_names_without_changing_legacy_format(tmp_path, monkeypatch, checks, expected_ids):
    from src.mod_agent.artifacts import ArtifactStore
    from src.mod_agent.domain import csharp_repository_delivery as delivery
    from src.mod_agent.evaluation.isolated_delivery_adapter import prepare_delivery
    from src.mod_agent.local_entry import _create_nuget_isolated_build_profile
    from tests.unit.test_isolated_delivery_adapter import _save, _setup

    context, root, manifest, proposal, report_ref = _setup(tmp_path, monkeypatch)
    declared = {"fixture": "repository boundary already verified by separate owner tests"}
    manifest["repository_profile"] = declared
    _save(root / "run_manifest.json", manifest)
    report = json.loads(Path(report_ref["path"]).read_bytes())
    report["verification_ref"] = _save(root / "checks/named-details.json",
        {"checks": checks})
    updated = _save(root / "checks/named-report.json", report)
    proposal["diff_intent"][0]["generation"]["patch_agent_loop"]["behavior_evaluation_refs"][0]["report_ref"] = updated
    original_read = ArtifactStore.semantic_read_immutable

    def read(store, reference):
        value = original_read(store, reference)
        if reference.artifact_type == "workflow_checkpoint":
            value["data"]["state_snapshot"]["runtime_summary"] = {"deferred_product_request_json": "{}"}
        return value

    monkeypatch.setattr(ArtifactStore, "semantic_read_immutable", read)
    monkeypatch.setattr(ProductExecutionRequestV1, "from_dict", lambda _value: SimpleNamespace())
    monkeypatch.setattr(delivery, "repository_binding_from_request", lambda _request: declared)
    target = root / "generated_mods/ArbitraryTask"
    restored_profile = _create_nuget_isolated_build_profile(run_root=root, working_directory=target,
        config_file=target / "nuget.config", bepinex_dir=None, timeout_seconds=23)
    monkeypatch.setattr(delivery, "validate_repository_binding", lambda *_: (
        SimpleNamespace(create_nuget_profile=lambda *_args: restored_profile), target))
    monkeypatch.setattr(delivery, "validate_repository_candidate", lambda **_: {"validator_sha256": "a" * 64})
    if expected_ids is None:
        with pytest.raises(ValueError, match="PASSED_CHECK_IDENTITIES_REQUIRED"):
            prepare_delivery(context)
        return
    result = prepare_delivery(context)
    assert result["required_check_ids"] == expected_ids
    assert result["repository_build_and_test"] == declared
    assert result["model_calls_performed"] == result["build_calls_performed"] == 0
