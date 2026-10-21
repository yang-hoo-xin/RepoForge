"""Pre-change knowledge and post-change runtime observations have different owners."""
from copy import deepcopy
import pytest
from src.mod_agent.planning import RequestPlanner,ModPlanningBriefBuilder,FeatureSpecBuilder
from src.mod_agent.evidence import EvidencePack,build_requirement_coverage_pack,default_feature_requirement_projection,materialize_requirement_coverage_projection
from src.mod_agent.evidence.requirement_coverage import requirement_input_obligation_model_view


def build(task_request):
 plan=RequestPlanner().plan(task_request,target_mod='VisualSignal')
 brief=ModPlanningBriefBuilder().build(plan).to_artifact_payload()
 return plan,brief,FeatureSpecBuilder().build(plan,planning_brief=brief)


@pytest.mark.parametrize('task_request',[
 'make Load enable diagnostics',
 '游戏更新后 Hook 失效，保留其他功能；当前依赖 Il2CppInterop.Runtime.dll；修复后检查正常加载。',
 'Add a new diagnostic marker at plugin startup. Verify the marker after the change.',
])
def test_generated_diagnostics_evidence_is_not_a_pre_patch_success_requirement(task_request):
 plan,brief,spec=build(task_request)
 assert plan.request_interpretation['request_type']=='runtime diagnostics'
 assert 'log marker or startup evidence' not in spec.evidence_requirements
 view=requirement_input_obligation_model_view(spec.to_artifact_payload())
 runtime=[r for r in view if r['origin']['field']=='runtime_verification_plan']
 assert runtime and all(not r['required_before_patch_policy'] for r in runtime)
 assert {'startup_loaded','diagnostic marker observed'}<=set(spec.runtime_verification_plan)
 assert spec.user_request==spec.target_behavior==task_request
 assert any('startup and logging' in q.lower() for q in brief['knowledge_query_plan']['queries'])


def test_explicit_log_prerequisite_is_not_removed_by_matching_template_text():
 plan,brief,_=build('Inspect the supplied startup log before changing the plugin.')
 explicit=['log marker or startup evidence','Read supplied startup.log before changing the initialization path.']
 brief['handoff']['feature_spec_guidance']['evidence_requirements']=explicit
 snapshot=deepcopy(brief)
 spec=FeatureSpecBuilder().build(plan,planning_brief=brief)
 assert spec.evidence_requirements==tuple(explicit) and brief==snapshot
 obligations=requirement_input_obligation_model_view(spec.to_artifact_payload())
 selected=[r for r in obligations if r['origin']['field']=='evidence_requirements']
 assert all(r['required_before_patch_policy'] and r['allowed_roles']==['pre_patch_evidence'] for r in selected)


def test_removing_generated_runtime_prerequisite_does_not_allow_source_free_patch():
 _,_,spec=build('Add a diagnostic marker at startup while preserving existing behavior.')
 feature=spec.to_artifact_payload()
 projection=materialize_requirement_coverage_projection(feature_data=feature,request=spec.user_request,
  projection=default_feature_requirement_projection(feature))
 pack=build_requirement_coverage_pack(query=spec.user_request,materialized_projection=projection,
  retrieved_pack=EvidencePack.empty(query=spec.user_request,reason='No current source observed'),findings={'findings':[]})
 assert not pack.can_generate_patch and not pack.requirement_coverage['edit_target_evidence_ids']


def test_explicit_interception_still_requires_method_timing_and_side_effect_evidence():
 plan,_,spec=build('Hook Update in VisualSignal and add a startup diagnostic marker.')
 assert plan.requires_hook_evidence
 assert {'target method','call timing','side effect boundary','concrete hook target method identity'}<=set(spec.evidence_requirements)
