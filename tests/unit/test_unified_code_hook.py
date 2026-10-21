"""Final candidate selection must not inherit an earlier patch verdict."""
from dataclasses import asdict
from copy import deepcopy
import pytest
from src.observability.evaluation import unified_hooks as hooks
from src.observability.evaluation.unified import save,observation
from src.observability.evaluation.sample_registry import Sample


def candidate(run, name='old'):
    root=run/name
    code={'baseline_sha256':'b'*64,'code_sha256':'c'*64,'files':['Plugin.cs']}
    built=save(root/'controlled-assembly.json',{'controlled_fixture':True})
    compiled=save(root/'compiled.json',{'status':'passed','execution_origin':'executed',
        'all_required_files_compiled':True,'materialized_code_identity':code,
        'model_response_ref':{'sha256':name},'built_artifact_ref':built})
    detail=save(root/'details.json',{'status':'passed','execution_origin':'executed',
        'materialized_code_identity':code,'built_artifact_ref':built,
        'checks':[{'check_id':'controlled-behavior','passed':True}]})
    raw={'status':'passed','failure_owner':None,'owner_binding_ids':[],'observations':[],
        'execution_origin':'executed','source_snapshot_sha256':'f'*64,
        'materialized_code_identity':code,'model_response_sha256':name,
        'generated_changes_sha256':name+'-change','verification_identity':{'validator_sha256':'a'*64},
        'verification_ref':detail,'verification_summary':{'checks':1,'failed_checks':0,'executions':1},
        'validation_results':{'full_compilation':{'status':'passed','all_required_files_compiled':True,'report_ref':compiled}}}
    reference=save(root/'validation.json',raw)
    return {**raw,'report_ref':reference,'admission':'accepted','callback_invoked':True}


def run_hook(tmp_path, evaluations, *, selected='old', product_status='awaiting_approval'):
    run=tmp_path/'run'
    receipt=save(run/'receipt.json',{'factory_receipt':{'model_response':{'output_sha256':selected},
        'patch_agent_loop':{'behavior_evaluations':evaluations}}})
    hook=hooks.CodeEvaluationHook(sample=Sample('d','s','f','constructed','regression','executable_reference',True),
        runtime_required=False,allowed_edit_paths=['Plugin.cs'])
    return hook({'route_id':'route','status':product_status,'product_result':{
        'run_id':'run','run_root':str(run),'patch_execution_receipts':[receipt]}},routing_root=tmp_path/'route')

@pytest.mark.parametrize('selected,expected', [('new','insufficient_evidence'),('old','passed')])
def test_old_candidate_report_does_not_validate_unchecked_final_candidate(tmp_path,monkeypatch,selected,expected):
    run=tmp_path/'run';run.mkdir()
    evaluation=candidate(run)
    receipt=save(run/'receipt.json',{'factory_receipt':{'model_response':{'output_sha256':selected},
        'patch_agent_loop':{'behavior_evaluations':[evaluation]}}})
    class Checked:
        def __call__(self,binding,payload,criteria,root):
            evidence=save(root/'check.json',{'status':'passed'})
            return {'results':[observation(binding,c,'passed',evidence=[evidence]) for c in criteria],
                'cost':dict(model_calls=0,input_tokens=0,output_tokens=0,cached_input_tokens=0),
                'evaluator_identity':{'evaluation_method':'execution','version':'unit-owned-validator'}}
    monkeypatch.setattr(hooks,'ExecutionReportAdapter',Checked)
    hook=hooks.CodeEvaluationHook(sample=Sample('d','s','f','constructed','regression','executable_reference',True),runtime_required=False,allowed_edit_paths=['Plugin.cs'])
    result=hook({'route_id':'route','status':'awaiting_approval','product_result':{
        'run_id':'run','run_root':str(run),'patch_execution_receipts':[receipt]}},routing_root=tmp_path/'route')
    assert result['status']==result['acceptance_status']==expected
    assert result['product_status']=='awaiting_approval'
    assert len(result['candidate_reports'])==1


@pytest.mark.parametrize('admission,expected',[('rejected','evaluation_error'),('pending','insufficient_evidence'),(None,'insufficient_evidence')])
def test_unadmitted_owner_report_cannot_validate_candidate(tmp_path,admission,expected):
    evaluation=candidate(tmp_path/'run')
    evaluation['admission']=admission
    evaluation['admission_error']='Behavior observation is not bound to the current input.'
    result=run_hook(tmp_path,[evaluation],product_status='blocked')
    assert result['status']==result['acceptance_status']==expected
    assert result['candidate_reports']==[]
    assert result['skipped_reports'][0]['report_ref']==evaluation['report_ref']
    assert result['skipped_reports'][0]['receipt_ref']['sha256']
    assert result['skipped_reports'][0]['admission']==admission


@pytest.mark.parametrize('field', ['source_snapshot_sha256','materialized_code_identity','status',
                                  'failure_owner','owner_binding_ids','observations','verification_identity'])
def test_accepted_journal_must_match_persisted_owner_report(tmp_path,field):
    evaluation=candidate(tmp_path/'run')
    evaluation[field]='mismatched-journal-value'
    result=run_hook(tmp_path,[evaluation])
    assert result['acceptance_status']=='evaluation_error'
    assert result['candidate_reports']==[]
    assert result['skipped_reports'][0]['code']=='CODE_RECEIPT_REPORT_BINDING_MISMATCH'


def test_latest_rejected_observation_cannot_inherit_same_candidate_prior_pass(tmp_path):
    evaluation=candidate(tmp_path/'run')
    rejected={**deepcopy(evaluation),'admission':'rejected'}
    result=run_hook(tmp_path,[evaluation,rejected],product_status='blocked')
    assert result['acceptance_status']=='evaluation_error'
    assert len(result['candidate_reports'])==1
    assert len(result['skipped_reports'])==1


def test_old_rejected_candidate_does_not_override_new_accepted_candidate(tmp_path):
    old=candidate(tmp_path/'run');old['admission']='rejected'
    new=candidate(tmp_path/'run','new')
    result=run_hook(tmp_path,[old,new],selected='new')
    assert result['acceptance_status']=='passed'
    assert len(result['candidate_reports'])==1
    assert len(result['skipped_reports'])==1


def test_rejected_evaluator_without_report_is_still_an_evaluation_error(tmp_path):
    result=run_hook(tmp_path,[{'status':'infrastructure_failed','admission':'rejected',
        'failure_owner':'unknown','model_response_sha256':'old','generated_changes_sha256':'old-change'}],
        product_status='blocked')
    assert result['acceptance_status']=='evaluation_error'
    assert result['skipped_reports'][0]['report_ref'] is None
