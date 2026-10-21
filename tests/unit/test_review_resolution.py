from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pytest
from src.observability.evaluation.unified import Binding,Criterion,EvaluationSession,save,read_ref,file_ref
from src.observability.evaluation.review_resolution import resolve_review,resolve_rag_batch
from tests.unit.test_unified_evaluation import adapter


def fixture(root, *, behavior='passed', runtime=False, reviewer_status='passed', number=0,
            critic_status='evaluation_error', execution_dimension='behavior'):
    payload={'query':'State and retry?','answer':'Indeterminate; never retry.',
             'contexts':[{'text':'Indeterminate; never retry.'}],'expected_behavior':'answer'}
    candidate=save(root/'candidate.json',payload)
    bound=Binding('task'+str(number),'conversation','run'+str(number),candidate)
    def execution(*args):
        result=adapter(behavior)(*args);result['evaluator_identity']['evaluation_method']='execution';return result
    criteria=[Criterion('behavior',execution_dimension,'code',method='execution'),
              Criterion('grounding','grounding','critic'),Criterion('fulfillment','task_fulfillment','critic')]
    evaluators={'code':execution,'critic':adapter(critic_status)}
    if runtime:
        criteria.append(Criterion('runtime','runtime','native',method='execution'))
        def missing(*args):
            r=adapter('insufficient_evidence')(*args);r['evaluator_identity']['evaluation_method']='execution';return r
        evaluators['native']=missing
    report=EvaluationSession(root/'auto',criteria=criteria,evaluators=evaluators,rubric_version='v1').evaluate(bound,payload)
    review={'schema_version':'assisted-semantic-review.v1','original_automated_report_ref':report['report_ref'],
        'binding':asdict(bound),'reviewer_origin':'current_codex_conversation_assistant','human_review':False,
        'independent_blind_review':False,'release_authority':False,'retained_candidate_ref':candidate,
        'criteria':[{'dimension':'grounding','status':reviewer_status},{'dimension':'task_fulfillment','status':reviewer_status}],
        'status':reviewer_status,'reason':'Compared both requirements with bound context',
        'source_spans':[{'context_number':1,'start':0,'end':len('Indeterminate; never retry.'),'quote':'Indeterminate; never retry.'}]}
    return report,review


def test_explicit_review_resolves_semantics_without_overwriting_original(tmp_path):
    report,review=fixture(tmp_path);before=read_ref(report['report_ref'])
    r=resolve_review(report['report_ref'],save(tmp_path/'review.json',review),tmp_path/'resolved')
    assert r['status']=='passed' and r['original_status']=='evaluation_error'
    assert r['additional_model_calls']==0 and r['review_usage']=='unavailable_for_chat_review'
    assert not r['human_review'] and not r['release_authority'] and not r['automated_judge_qualified']
    assert read_ref(report['report_ref'])==before


@pytest.mark.parametrize('behavior,runtime,expected',[('failed',False,'failed'),('passed',True,'insufficient_evidence')])
def test_semantic_pass_cannot_rescue_execution_or_missing_game(tmp_path,behavior,runtime,expected):
    report,review=fixture(tmp_path,behavior=behavior,runtime=runtime)
    assert resolve_review(report['report_ref'],save(tmp_path/'review.json',review),tmp_path/'result')['status']==expected


@pytest.mark.parametrize('status',['failed','insufficient_evidence','evaluation_error'])
def test_review_preserves_distinct_negative_and_unknown_states(tmp_path,status):
    report,review=fixture(tmp_path,reviewer_status=status)
    assert resolve_review(report['report_ref'],save(tmp_path/'review.json',review),tmp_path/'result')['status']==status


@pytest.mark.parametrize('mutation',['candidate','run','conversation','task','report','retained','quote','offset','boolean_offset','dimension','duplicate','not_applicable','human','blind','authority','summary','missing_grounding_evidence'])
def test_rejects_stale_fabricated_or_out_of_scope_review(tmp_path,mutation):
    report,review=fixture(tmp_path)
    if mutation in {'run','conversation','task'}:review['binding'][mutation+'_id']='changed'
    elif mutation=='candidate':review['binding']['candidate_ref']['sha256']='0'*64
    elif mutation=='report':review['original_automated_report_ref']['sha256']='0'*64
    elif mutation=='retained':review['retained_candidate_ref']=save(tmp_path/'other.json',{'answer':'other'})
    elif mutation=='quote':review['source_spans'][0]['quote']='invented'
    elif mutation=='offset':review['source_spans'][0]['end']=999
    elif mutation=='boolean_offset':review['source_spans'][0]['context_number']=True
    elif mutation=='dimension':review['criteria'][0]['dimension']='behavior'
    elif mutation=='duplicate':review['criteria'].append(review['criteria'][0])
    elif mutation=='not_applicable':review['criteria'][0]['status']='not_applicable'
    elif mutation=='human':review['human_review']=True
    elif mutation=='blind':review['independent_blind_review']=True
    elif mutation=='authority':review['release_authority']=True
    elif mutation=='summary':review['status']='failed'
    else:review['source_spans']=[]
    assert resolve_review(report['report_ref'],save(tmp_path/'review.json',review),tmp_path/'result')['status']=='evaluation_error'


def make_batch(tmp_path,*,missing=False,duplicate=False):
    queries=[];reviews=[]
    for i in range(2):
        report,review=fixture(tmp_path/str(i),number=i)
        queries.append({'case_index':i,'query':'State and retry?','unified_evaluation':report})
        if not (missing and i==1):reviews.append(save(tmp_path/str(i)/'review.json',review))
    original=save(tmp_path/'batch.json',{'query_count':2,'query_results':queries})
    if duplicate:reviews.append(reviews[0])
    batch=save(tmp_path/'reviews.json',{'schema_version':'assisted-semantic-review-batch.v1','original_result_ref':original,'review_refs':reviews})
    selection=save(tmp_path/'selection.json',{'schema_version':'judge-selection.v1','selected':'current_codex_conversation_assisted_review',
        'human_review':False,'execute_or_apply_authority':False,'review_batch_ref':batch})
    return original,selection


def test_batch_missing_review_retains_denominator_and_prior_failure(tmp_path):
    original,selection=make_batch(tmp_path,missing=True)
    r=resolve_rag_batch(original,selection,tmp_path/'resolved')
    assert r['query_count']==2 and r['counts']['passed']==1 and r['counts']['evaluation_error']==1
    assert r['status']=='evaluation_error'


def test_duplicate_review_does_not_inflate_passes(tmp_path):
    original,selection=make_batch(tmp_path,duplicate=True)
    with pytest.raises(ValueError,match='DUPLICATE_REVIEW'):resolve_rag_batch(original,selection,tmp_path/'resolved')


def test_normal_cli_resolution_never_calls_generator_or_judge(tmp_path,monkeypatch,capsys):
    import scripts.run_unified_evaluation as cli
    original,selection=make_batch(tmp_path)
    def forbidden(*_):raise AssertionError('No new model invocation allowed')
    monkeypatch.setattr(cli,'build_critic',forbidden);monkeypatch.setattr(cli,'build_ragas',forbidden)
    assert cli.main(['resolve','--automated-result',original['path'],'--selection',selection['path'],'--output-root',str(tmp_path/'resolved')])==0
    result=json.loads((tmp_path/'resolved/result.json').read_bytes())
    assert result['status']=='passed' and result['query_count']==2
    assert result['counts']['passed']==2 and result['additional_model_calls']==0
    assert not result['automated_judge_qualified']


def test_stored_acceptance_cannot_override_unresolved_criteria(tmp_path):
    from src.observability.evaluation.review_resolution import checked_report

    report, _ = fixture(tmp_path)
    original = json.loads(read_ref(report['report_ref']))
    assert original['acceptance_status'] == 'evaluation_error'
    original['acceptance_status'] = 'passed'
    changed = save(tmp_path / 'incorrect-summary.json', original)

    with pytest.raises(ValueError, match='ORIGINAL_ACCEPTANCE_INVALID'):
        checked_report(changed)


def test_partial_review_rejects_forged_assisted_decision_use(tmp_path):
    report, review = fixture(tmp_path, critic_status='passed')
    original = json.loads(read_ref(report['report_ref']))
    assert original['acceptance_status'] == 'insufficient_evidence'
    original['criteria'][2]['decision_use'] = 'assisted_review'
    forged = save(tmp_path / 'forged-assisted.json', original)
    review['original_automated_report_ref'] = forged
    review['criteria'] = review['criteria'][:1]
    review['reason'] = 'Only grounding was reviewed'

    result = resolve_review(forged, save(tmp_path / 'review.json', review), tmp_path / 'resolved')

    assert result['status'] == 'evaluation_error'
    assert result['error_code'] == 'RESOLUTION_ORIGINAL_DECISION_USE_INVALID'


@pytest.mark.parametrize('index,decision_use', [(0, 'trial_only'), (1, 'acceptance'), (1, None)])
def test_original_decision_use_must_match_criterion_method(tmp_path, index, decision_use):
    from src.observability.evaluation.review_resolution import checked_report

    report, _ = fixture(tmp_path)
    original = json.loads(read_ref(report['report_ref']))
    if decision_use is None:
        original['criteria'][index].pop('decision_use')
    else:
        original['criteria'][index]['decision_use'] = decision_use
    changed = save(tmp_path / 'wrong-decision-use.json', original)

    with pytest.raises(ValueError, match='ORIGINAL_DECISION_USE_INVALID'):
        checked_report(changed)


def test_partial_review_keeps_unreviewed_trial_pass_pending(tmp_path):
    report, review = fixture(tmp_path, critic_status='passed')
    review['criteria'] = review['criteria'][:1]

    result = resolve_review(report['report_ref'], save(tmp_path / 'review.json', review), tmp_path / 'resolved')

    assert result['status'] == 'insufficient_evidence'
    assert result['criteria'][2]['decision_use'] == 'trial_only'


def test_semantic_review_cannot_override_required_citation_failure(tmp_path):
    report, review = fixture(tmp_path, behavior='failed', execution_dimension='citation_policy')
    result = resolve_review(report['report_ref'], save(tmp_path / 'review.json', review), tmp_path / 'resolved')

    assert result['status'] == 'failed'
    assert result['criteria'][0]['dimension'] == 'citation_policy'
    assert result['criteria'][0]['decision_use'] == 'acceptance'
    assert result['criteria'][0]['status'] == 'failed'


def test_review_cannot_select_execution_citation_dimension(tmp_path):
    report, review = fixture(tmp_path, behavior='failed', execution_dimension='citation_policy')
    review['criteria'].append({'dimension': 'citation_policy', 'status': 'passed'})
    result = resolve_review(report['report_ref'], save(tmp_path / 'review.json', review), tmp_path / 'resolved')

    assert result['status'] == 'evaluation_error'
    assert result['error_code'] == 'RESOLUTION_REVIEW_SCOPE_INVALID'
