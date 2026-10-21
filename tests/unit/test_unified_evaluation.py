from dataclasses import asdict
import json
import pytest
from src.observability.evaluation.unified import Binding, Criterion, EvaluationSession, observation, save

@pytest.fixture
def subject(tmp_path):
    payload={'answer':'a candidate'}
    return Binding('task','conversation','run',save(tmp_path/'candidate.json',payload)),payload

def adapter(status='passed', mutate=None, value=None):
    def run(binding,payload,criteria,root):
        evidence=save(root/'original.json',{'observed':status})
        rows=[observation(binding,c,status,evidence=[evidence],value=value) for c in criteria]
        if mutate:mutate(binding,payload,rows,evidence)
        return {'results':rows,'cost':{'model_calls':0,'input_tokens':0,'output_tokens':0,'cached_input_tokens':0},'evaluator_identity':{'version':'test-v1'}}
    return run

def evaluate(tmp_path,subject,criteria,adapters):
    return EvaluationSession(tmp_path/'evaluations',criteria=criteria,evaluators=adapters,rubric_version='rubric-v1').evaluate(*subject)

@pytest.mark.parametrize('status',['passed','failed','insufficient_evidence','evaluation_error'])
def test_preserves_four_applicable_states(tmp_path,subject,status):
    r=evaluate(tmp_path,subject,[Criterion('behavior','behavior','code')],{'code':adapter(status)})
    assert r['status']==status
    assert r['criteria'][0]['binding']==asdict(subject[0])
    assert r['evaluation_cost']['model_calls']=={'known_subtotal':0,'complete':True}

def test_not_applicable_does_not_invoke_judge(tmp_path,subject):
    r=evaluate(tmp_path,subject,[Criterion('runtime','runtime','code',applicable=False)],{})
    assert r['status']=='not_applicable' and r['invocation_refs']==[]

def test_high_semantic_score_cannot_rescue_behavior_failure(tmp_path,subject):
    r=evaluate(tmp_path,subject,[Criterion('behavior','behavior','code'),Criterion('support','support','critic',False)],{'code':adapter('failed'),'critic':adapter(value=1.0)})
    assert r['status']=='failed'

def test_required_runtime_missing_remains_pending(tmp_path,subject):
    r=evaluate(tmp_path,subject,[Criterion('behavior','behavior','code'),Criterion('runtime','runtime','game')],{'code':adapter(),'game':adapter('insufficient_evidence')})
    assert r['status']=='insufficient_evidence'

def test_duplicate_primary_dimension_rejected_before_calls(tmp_path,subject):
    with pytest.raises(ValueError,match='DUPLICATE_PRIMARY'):
        evaluate(tmp_path,subject,[Criterion('a','support','ragas'),Criterion('b','support','critic')],{})

def test_calibration_retained_separately_and_does_not_gate(tmp_path,subject):
    r=evaluate(tmp_path,subject,[Criterion('main','support','critic'),Criterion('audit','support','ragas',False,role='calibration')],{'critic':adapter(),'ragas':adapter('failed')})
    assert r['status']=='passed' and len(r['criteria'])==2

def test_same_evaluator_invoked_once_for_multiple_dimensions(tmp_path,subject):
    calls=[];base=adapter()
    def invoke(*args):calls.append(1);return base(*args)
    r=evaluate(tmp_path,subject,[Criterion('a','support','critic'),Criterion('b','relevance','critic')],{'critic':invoke})
    assert r['status']=='passed' and len(calls)==1

@pytest.mark.parametrize('change',['candidate','conversation','run','task','evidence','payload','binding','applicability','score'])
def test_rejects_stale_or_mutated_evidence(tmp_path,subject,change):
    def mutate(binding,payload,rows,evidence):
        if change in {'conversation','run','task'}:rows[0]['binding'][change+'_id']='another'
        elif change=='candidate':rows[0]['binding']['candidate_ref']['sha256']='0'*64
        elif change=='evidence':rows[0]['evidence_refs'][0]['sha256']='0'*64
        elif change=='payload':payload['answer']='different'
        elif change=='binding':binding.candidate_ref['sha256']='0'*64
        elif change=='applicability':rows[0]['status']='not_applicable'
        else:rows[0]['value']=float('nan')
    r=evaluate(tmp_path,subject,[Criterion('a','behavior','code')],{'code':adapter(mutate=mutate)})
    assert r['status']=='evaluation_error'

def test_threshold_is_not_just_advisory(tmp_path,subject):
    r=evaluate(tmp_path,subject,[Criterion('a','support','ragas',threshold=.9)],{'ragas':adapter(value=.1)})
    assert r['status']=='evaluation_error'

def test_failed_judge_is_not_content_failure_or_zero_usage(tmp_path,subject):
    def fail(*_):raise TimeoutError('provider unavailable')
    r=evaluate(tmp_path,subject,[Criterion('a','support','critic')],{'critic':fail})
    assert r['status']=='evaluation_error'
    assert r['evaluation_cost']['model_calls']['complete'] is False
    assert r['criteria'][0]['value'] is None

@pytest.mark.parametrize('name',['../../outside','..','a/b'])
def test_evaluator_name_cannot_select_output_path(name):
    with pytest.raises(ValueError):Criterion('a','behavior',name)
def test_deep_local_evidence_paths_remain_readable(tmp_path,subject):
    from src.observability.evaluation.unified import read_ref
    deep=tmp_path/('a'*100)/('b'*100)
    r=evaluate(deep,subject,[Criterion('a','behavior','code')],{'code':adapter()})
    assert r['status']=='passed'
    assert json.loads(read_ref(r['report_ref']))['status']=='passed'


@pytest.mark.parametrize('execution_status,trial_status,acceptance',[
    ('passed','passed','insufficient_evidence'),
    ('passed','failed','insufficient_evidence'),
    ('failed','passed','failed'),
    ('passed','evaluation_error','evaluation_error'),
])
def test_trial_judge_neither_approves_nor_vetoes_execution(tmp_path,subject,execution_status,trial_status,acceptance):
    def run(*args):
        result=adapter(execution_status)(*args)
        result['evaluator_identity']['evaluation_method']='execution'
        return result
    result=evaluate(tmp_path,subject,[Criterion('behavior','behavior','code',method='execution'),
        Criterion('support','grounding','critic')],{'code':run,'critic':adapter(trial_status)})
    assert result['acceptance_status']==acceptance
    assert result['release_authority'] is False


def test_execution_only_report_can_validate_candidate_without_judge(tmp_path,subject):
    def run(*args):
        result=adapter()(*args);result['evaluator_identity']['evaluation_method']='execution';return result
    result=evaluate(tmp_path,subject,[Criterion('behavior','behavior','code',method='execution')],{'code':run})
    assert result['acceptance_status']=='passed'
    assert result['qualification_status']=='execution_checks_only'
    assert result['decision_scope']=='candidate_validation'


def test_model_result_cannot_claim_execution_basis(tmp_path,subject):
    def run(*args):
        result=adapter()(*args)
        result['evaluator_identity']['evaluation_method']='execution'
        result['cost']['model_calls']=1
        return result
    result=evaluate(tmp_path,subject,[Criterion('support','support','critic',method='execution')],{'critic':run})
    assert result['acceptance_status']=='evaluation_error'
