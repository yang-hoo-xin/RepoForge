"""Shared C# domain integration with controlled Build/test doubles, never real compilation."""
import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from src.mod_agent.build import BuildAdapter, BuildCommandResult, BuildProfile
from src.mod_agent.domain.csharp_repository import CSharpRepositoryAdapter
from src.mod_agent.evidence import EvidenceItem, EvidencePack
from src.mod_agent.evidence.models import DETERMINISTIC_EVIDENCE_AUTHORITY
from src.mod_agent.execution.patch_apply import ExactTextPatchApplier
from src.mod_agent.planning import RequestPlanner
from src.mod_agent.product_execution import ProductExecutionContractError
from src.mod_agent.product_executor import (
    ProductExecutionAdaptersV1,
    execute_product_execution,
    preflight_product_execution,
    source_snapshot,
    source_snapshot_sha256,
)
from tests.unit.test_product_execution import (
    _blocked_evidence_adapters,
    _formalize_product_test_pack,
    _human_semantic_approve_from_dry_run,
    _request,
)

CHECKS = ("The declared shared public Build/test checks pass on this exact revision.",
          "The declared output artifact is retained with its source identity.")


def test_default_product_adapter_has_no_domain_override():
    assert _blocked_evidence_adapters().domain_adapter is None


def test_arbitrary_object_is_not_an_admitted_domain_adapter(tmp_path):
    request, _ = _request(tmp_path)
    with pytest.raises(ProductExecutionContractError):
        replace(_blocked_evidence_adapters(), domain_adapter=object()).validate_for(request)


@pytest.mark.parametrize("roots", [(), (".",), ("../outside",), ("C:/outside",), ("/outside",), ("repo/*",)])
def test_neutral_adapter_requires_explicit_relative_source_scope(roots):
    with pytest.raises(ValueError, match="SCOPE"):
        CSharpRepositoryAdapter(roots, CHECKS)


def test_neutral_feature_keeps_original_request_and_declared_public_conditions():
    text="Fix Arithmetic.Double while preserving all existing public signatures."
    adapter=CSharpRepositoryAdapter(("repo",),CHECKS)
    plan=RequestPlanner().plan(text,target_mod="ArithmeticLibrary")
    feature=adapter.build_feature_spec(plan,planning_brief={"selected_plan":{"non_goals":["Do not rename public methods."]}})
    assert feature.user_request==feature.target_behavior==text
    assert feature.non_goals==("Do not rename public methods.",)
    assert feature.runtime_verification_plan==CHECKS
    assert not any(value in feature.runtime_verification_plan for value in ("startup_loaded","target_path_executed","acceptance_evidence_observed"))
    assert not feature.request_interpretation
    with pytest.raises(ValueError,match="SHARED_BACKEND"):
        adapter.build_profile(adapter.identify_project("."))


def test_domain_override_cannot_authorize_game_runtime_or_a_different_scope(tmp_path):
    request,_=_request(tmp_path)
    adapter=CSharpRepositoryAdapter(("repo",),CHECKS)
    adapters=replace(_blocked_evidence_adapters(),domain_adapter=adapter)
    for candidate in (request,replace(request,target_domain=adapter.domain_name),
                      replace(request,target_domain=adapter.domain_name,requested_completion_scope="runtime_workflow")):
        with pytest.raises(ProductExecutionContractError):
            adapters.validate_for(candidate)


@pytest.mark.parametrize("bad_identity", [None, "feature_id", "user_request"])
def test_public_csharp_library_runs_original_product_graph_through_shared_build_check(tmp_path, monkeypatch, bad_identity):
    # Scope: one visible synthetic library, scripted planning/patch and explicitly controlled build/check doubles.
    # 为什么要有它：验证的是正常产品阶段及共享适配边界，不把替身输出当真实编译、隐藏题通过或架构优势。
    def no_network(*args,**kwargs):
        raise AssertionError("No provider/network is permitted")
    monkeypatch.setattr("socket.socket.connect",no_network)
    original,store=_request(tmp_path)
    source=tmp_path/"library-source"
    target=Path(original.isolated_workspace_root)/"repo"
    source.mkdir()
    target.mkdir()
    before="public static class Arithmetic { public static int Double(int value) { return value; } }\n"
    after=before.replace("return value;","return value * 2;")
    project='<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>\n'
    for root in (source,target):
        (root/"Arithmetic.cs").write_text(before,encoding="utf-8")
        (root/"ArithmeticLibrary.csproj").write_text(project,encoding="utf-8")
    request=replace(original,user_request="Make Arithmetic.Double return twice its integer input and preserve its public signature.",
        target_domain="csharp_repository",target_mod="ArithmeticLibrary",source_root=str(source.resolve()),
        isolated_target_root=str(target.resolve()),source_snapshot_sha256=source_snapshot_sha256(source_snapshot(source)),
        isolated_source_snapshot_sha256=source_snapshot_sha256(source_snapshot(target)))
    seen={"check_calls":0,"feature":None,"dry_run":None}
    applier=ExactTextPatchApplier()
    def evidence(_state,feature):
        seen["feature"]=deepcopy(feature)
        raw=(target/"Arithmetic.cs").read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        pack=EvidencePack(query=request.user_request,readiness="sufficient",can_generate_patch=True,
            items=(EvidenceItem(evidence_id="public_library_source",source_type="target_source",
                source_path="repo/Arithmetic.cs",symbol="Double",span={"start_line":1,"end_line":1},
                retrieval_method="symbol_exact",score=1.0,claim_supported="The public fixture contains Arithmetic.Double.",
                excerpt=raw.decode(),source_ref=str(target/"Arithmetic.cs"),source_hash=digest,file_hash=digest,doc_hash=digest,
                authority=DETERMINISTIC_EVIDENCE_AUTHORITY),),retrieval_methods_used=("symbol_exact",))
        return _formalize_product_test_pack(query=request.user_request,feature_data=feature,retrieved_pack=pack)
    def patch(_state,plan,_evidence):
        rows=deepcopy(plan["proposed_changes"])
        assert len(rows)==1
        # Scope: the shared planner authorized the method, not replacement of its enclosing class.
        # 为什么要有它：受控fixture也必须遵守真实binding边界，不能为通过测试放松正常Review。
        rows[0]["patch"]={"kind":"exact_text_replace","before_text":"return value;","after_text":"return value * 2;","expected_occurrences":1}
        return tuple(rows)
    def dry_run(proposal,plan,review):
        seen["dry_run"]=applier.run(workspace_root=request.isolated_workspace_root,proposal=proposal,
            implementation_plan=plan,review=review,allow_write=False)
        return seen["dry_run"]
    def apply(proposal,plan,review):
        return applier.run(workspace_root=request.isolated_workspace_root,proposal=proposal,
            implementation_plan=plan,review=review,allow_write=True,
            approved_diff_hash=seen["dry_run"]["diff_hash"],
            human_semantic_decision=_human_semantic_approve_from_dry_run(seen["dry_run"]))
    def build(feature_id,input_execution_mode):
        def runner(profile):
            assert (target/"Arithmetic.cs").read_text(encoding="utf-8")==after
            seen["check_calls"]+=1
            artifact=target/"bin/ArithmeticLibrary.dll"
            artifact.parent.mkdir()
            artifact.write_bytes(b"CONTROLLED FIXTURE; NOT A COMPILED ASSEMBLY")
            return BuildCommandResult(exit_code=0,stdout="Controlled build/check adapter reached",stderr="",duration_ms=1)
        return BuildAdapter().run_local(feature_id=feature_id,
            profile=BuildProfile(profile_id="public_fixture_shared_backend",build_system="dotnet",working_directory=str(target),
                command=("declared-fixture-build-and-check",),timeout_seconds=30,expected_artifacts=("bin/ArithmeticLibrary.dll",)),
            command_runner=runner,allow_execution=True,input_execution_mode=input_execution_mode)
    class SelectedDomain(CSharpRepositoryAdapter):
        def build_feature_spec(self, plan, *, planning_brief=None):
            feature=super().build_feature_spec(plan,planning_brief=planning_brief)
            return replace(feature,**{bad_identity:"changed contract identity"}) if bad_identity else feature
    adapters=ProductExecutionAdaptersV1(evidence_pack_factory=evidence,patch_change_factory=patch,
        planning_target_selection_factory=lambda _feature,_pack:("public_library_source",),dry_run_factory=dry_run,
        isolated_apply_factory=apply,build_result_factory=build,runtime_evidence_factory=None,
        compiled_artifact_verifier=None,source_change_verifier=None,
        domain_adapter=SelectedDomain(("repo",),CHECKS))
    preflight=preflight_product_execution(request,adapters,store)
    assert preflight.preflight_status=="ready",preflight
    result=execute_product_execution(preflight,adapters,store)
    if bad_identity:
        assert result.execution_disposition!="succeeded"
        assert seen["feature"] is None and seen["check_calls"]==0
        assert (target/"Arithmetic.cs").read_text(encoding="utf-8")==before
        return
    assert result.execution_disposition=="succeeded",(result.failure_stage,[row.message for row in result.blockers])
    assert seen["check_calls"]==1
    assert seen["feature"]["user_request"]==seen["feature"]["target_behavior"]==request.user_request
    assert seen["feature"]["runtime_verification_plan"]==list(CHECKS)
    assert (source/"Arithmetic.cs").read_text(encoding="utf-8")==before
    assert (target/"Arithmetic.cs").read_text(encoding="utf-8")==after
    nodes=[row.node_name for row in result.store_attempts]
    for node in ("plan_request","planning_brief","draft_feature_spec","retrieve_evidence","create_implementation_plan",
                 "generate_patch_proposal","review_proposal","dry_run_apply","isolated_apply","build"):
        assert node in nodes
