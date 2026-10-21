from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from src.core.types import RetrievalResult
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.llm.agent_turn import AgentTurnResponse, AgentUsage
from src.libs.llm.transport_observation import transport_started, transport_response_received
from src.observability.evaluation.unified import Binding, Criterion, EvaluationSession, save
from src.observability.evaluation.unified_adapters import CriticAdapter, RagasAdapter
from src.observability.evaluation.unified_hooks import RagEvaluationHook
from src.observability.evaluation.sample_registry import Sample, validate_samples
from src.observability.evaluation.eval_runner import EvalRunner

TEXT='The feature defaults to disabled.'
DIGEST='a'*64

def annotation():
    return {'labels':{'retrieval':'retrieve','relevance':'relevant','support':'fully_supported','utility':5},
            'answer_kind':'factual','unsupported_factual_claims':False,'core_claim':TEXT,
            'core_verdict':'supported','completeness':'complete','issue':'',
            'evidence_quotes':[{'context_index':1,'quote':TEXT,'relation':'supports_core'}]}

class Client:
    model='unit-critic';provider='ollama_chat_local';context_window=32768
    def __init__(self,value=None):self.value=value or annotation();self.calls=0
    def complete_turn(self,request):
        self.calls+=1
        assert 'expected_behavior' not in request.messages[-1].content
        transport_started();transport_response_received(http_status=200,response_complete=True)
        return AgentTurnResponse(self.provider,self.model,json.dumps(self.value),stop_reason='stop',usage=AgentUsage(10,20),
            provider_diagnostic={'measured_usage':{'input_tokens':10,'output_tokens':20}})

def sample(partition='calibration',family='config'):
    return Sample('suite','case',family,'constructed',partition,'executable_reference',partition!='holdout')

def critic(client=None,reader=None):
    return CriticAdapter(client or Client(),identity_reader=reader or (lambda:{'digest':DIGEST}),expected_digest=DIGEST)

def execute(tmp_path,adapter,payload):
    binding=Binding('task','conversation','run',save(tmp_path/'candidate.json',payload))
    return EvaluationSession(tmp_path/'eval',criteria=[Criterion('support','grounding','critic')],evaluators={'critic':adapter},rubric_version='v1').evaluate(binding,payload)

def payload():return {'query':'What is the default?','answer':TEXT,'contexts':[{'id':'c1','text':TEXT}],'expected_behavior':'answer'}

def test_critic_reuses_labels_and_records_real_transport_observation(tmp_path):
    result=execute(tmp_path,critic(),payload())
    assert result['status']=='passed'
    assert result['evaluation_cost']['model_calls']=={'known_subtotal':1,'complete':True}
    assert result['evaluation_cost']['input_tokens']['known_subtotal']==10
    assert result['evaluation_cost']['cached_input_tokens']['complete'] is False

def test_fabricated_quote_is_judge_error_not_content_failure(tmp_path):
    a=annotation();a['evidence_quotes'][0]['quote']='This quote does not exist.'
    result=execute(tmp_path,critic(Client(a)),payload())
    assert result['status']=='evaluation_error'

def test_fourth_context_is_not_silently_dropped(tmp_path):
    a=annotation();a['evidence_quotes'][0]['context_index']=4
    p=payload();p['contexts']=[{'text':'irrelevant source'}]*3+[{'text':TEXT}]
    assert execute(tmp_path,critic(Client(a)),p)['status']=='passed'

def test_model_identity_change_is_judge_error(tmp_path):
    identities=iter([{'digest':DIGEST},{'digest':'b'*64}])
    assert execute(tmp_path,critic(reader=lambda:next(identities)),payload())['status']=='evaluation_error'

def test_normal_eval_runner_automatically_attaches_unified_result(tmp_path):
    path=tmp_path/'gold.json';path.write_text(json.dumps({'test_cases':[{'query':'What is the default?','expected_chunk_ids':['c1'],'expected_behavior':'answer'}]}))
    search=MagicMock();search.search.return_value=[RetrievalResult(chunk_id='c1',score=1.0,text=TEXT,metadata={})]
    client=Client();hook=RagEvaluationHook(tmp_path/'reports',samples={0:sample()},critic=critic(client))
    report=EvalRunner(hybrid_search=search,evaluator=CustomEvaluator(metrics=['hit_rate']),answer_generator=lambda *_:TEXT,
        synthetic_fallback=False,unified_hook=hook).run(path,top_k=4,run_metadata={'run_id':'actual-run'})
    result=report.to_dict()['query_results'][0]['unified_evaluation']
    assert result['status']=='passed' and client.calls==1
    assert result['sample']['family_id']=='config'
    assert result['binding']['run_id']=='actual-run'
    assert result['report_ref']['sha256']

def test_registry_rejects_seen_holdout():
    with pytest.raises(ValueError):Sample('suite','x','f','constructed','holdout','ai',True)

def test_registry_rejects_related_cross_partition_samples():
    a=sample();b=Sample('suite','unseen-name','config','constructed','holdout','human',False)
    with pytest.raises(ValueError,match='FAMILY_CROSSES_HOLDOUT'):validate_samples([a,b])

def test_ragas_keeps_missing_measurement_out_of_zero_scores(tmp_path):
    class Missing:
        def evaluate(self,**kwargs):return {}
    binding=Binding('task','conversation','run',save(tmp_path/'candidate.json',payload()))
    result=EvaluationSession(tmp_path/'eval',criteria=[Criterion('support','grounding','ragas',threshold=1.0)],
        evaluators={'ragas':RagasAdapter(Missing())},rubric_version='v1').evaluate(binding,payload())
    assert result['status']=='evaluation_error' and result['criteria'][0]['value'] is None
    assert result['evaluation_cost']['model_calls']['complete'] is False

def test_uncalibrated_semantic_pass_cannot_be_acceptance_pass(tmp_path):
    result=execute(tmp_path,critic(),payload())
    assert result['status']=='passed'
    assert result['acceptance_status']=='insufficient_evidence'
    assert result['criteria'][0]['decision_use']=='trial_only'
    assert result['qualification_status']=='not_established'


def test_native_context_rejection_is_measured_without_sending(tmp_path):
    from src.libs.llm.agent_turn import OllamaChatAgentAdapter
    def send(*_):raise AssertionError('Context gate must reject before transport')
    client=OllamaChatAgentAdapter(base_url='http://127.0.0.1:11434',model='unit-critic',
        max_output_tokens=1024,context_window=8192,request_sender=send)
    p=payload();p['contexts'][0]['text']=TEXT*600
    r=execute(tmp_path,critic(client),p)
    assert r['status']=='evaluation_error'
    assert r['evaluation_cost']['model_calls']=={'known_subtotal':0,'complete':True}
    from src.observability.evaluation.unified import read_ref
    audit=json.loads(read_ref(r['criteria'][0]['evidence_refs'][0]))
    assert audit['failure_boundary']=='provider_request_gate'
    assert audit['failure_classification']=='provider_request_context_budget_exceeded'
    assert audit['context_admission']['admitted'] is False
    assert r['evaluation_cost']['input_tokens']['complete'] is False


def test_disabled_legacy_scorer_is_not_a_spurious_failure(tmp_path):
    from src.libs.evaluator.base_evaluator import NoneEvaluator
    path=tmp_path/'gold.json';path.write_text(json.dumps({'test_cases':[{'query':'What is the default?','expected_chunk_ids':['c1'],'expected_behavior':'answer'}]}))
    search=MagicMock();search.search.return_value=[RetrievalResult(chunk_id='c1',score=1.0,text=TEXT,metadata={})]
    hook=RagEvaluationHook(tmp_path/'reports',samples={0:sample()},critic=critic())
    report=EvalRunner(hybrid_search=search,evaluator=NoneEvaluator(),answer_generator=lambda *_:TEXT,
        synthetic_fallback=False,unified_hook=hook).run(path,run_metadata={'run_id':'normal-entry'})
    q=report.to_dict()['query_results'][0]
    assert q['evaluator_measurement']['status']=='not_applicable'
    assert q['evaluator_measurement']['mode']=='disabled'
    assert q['citation_required'] is False
    assert q['unified_evaluation']['status']=='passed'
    assert q['unified_evaluation']['acceptance_status']=='insufficient_evidence'
    citation=next(row for row in q['unified_evaluation']['criteria'] if row['dimension']=='citation_policy')
    assert citation['status']=='not_applicable' and citation['applicable'] is False


def test_required_citation_failure_cannot_pass_unified_execution_gate(tmp_path):
    from src.libs.evaluator.base_evaluator import NoneEvaluator
    path=tmp_path/'gold.json';path.write_text(json.dumps({'citation_policy':'cite retrieved contexts',
        'test_cases':[{'query':'What is the default?','expected_chunk_ids':['c1'],'expected_behavior':'answer'}]}))
    search=MagicMock();search.search.return_value=[RetrievalResult(chunk_id='c1',score=1.0,text=TEXT,metadata={})]
    hook=RagEvaluationHook(tmp_path/'reports',samples={0:sample()},critic=critic())
    report=EvalRunner(hybrid_search=search,evaluator=NoneEvaluator(),answer_generator=lambda *_:TEXT,
        synthetic_fallback=False,unified_hook=hook).run(path,run_metadata={'run_id':'citation-required'})
    query=report.to_dict()['query_results'][0]
    result=query['unified_evaluation']
    citation=next(row for row in result['criteria'] if row['dimension']=='citation_policy')
    assert query['citation_required'] is True
    assert query['citation_measurement']['error_category']=='MissingContextCitation'
    assert citation['status']=='failed' and citation['applicable'] is True
    assert result['status']==result['acceptance_status']=='failed'


def test_assisted_normal_entry_produces_pending_review_without_judge(tmp_path):
    from src.libs.evaluator.base_evaluator import NoneEvaluator
    from src.observability.evaluation.unified import read_ref
    path=tmp_path/'gold.json';path.write_text(json.dumps({'test_cases':[{'query':'What is the default?','expected_chunk_ids':['c1'],'expected_behavior':'answer'}]}))
    search=MagicMock();search.search.return_value=[RetrievalResult(chunk_id='c1',score=1.0,text=TEXT,metadata={})]
    hook=RagEvaluationHook(tmp_path/'reports',samples={0:sample()})
    report=EvalRunner(hybrid_search=search,evaluator=NoneEvaluator(),answer_generator=lambda *_:TEXT,
        synthetic_fallback=False,unified_hook=hook).run(path,run_metadata={'run_id':'assisted-normal'})
    result=report.to_dict()['query_results'][0]['unified_evaluation']
    assert result['status']==result['acceptance_status']=='insufficient_evidence'
    assert result['evaluation_cost']['model_calls']=={'complete':True,'known_subtotal':0}
    assert result['criteria'][0]['status']=='passed'
    grounding=next(row for row in result['criteria'] if row['dimension']=='grounding')
    request=json.loads(read_ref(grounding['evidence_refs'][0]))
    assert request['binding']==result['binding']
    assert request['required_dimensions']==['grounding','task_fulfillment']


@pytest.mark.parametrize('options',[['--review-mode','trial'],['--review-mode','assisted','--judge-profile','unused.json']])
def test_cli_rejects_ambiguous_judge_mode_before_work(tmp_path,options):
    from scripts.run_unified_evaluation import main
    with pytest.raises(SystemExit) as error:
        main(['rag','--metadata','unused.json','--settings','unused.yaml','--test-set','unused.json','--output-root',str(tmp_path/'output'),*options])
    assert error.value.code==2 and not (tmp_path/'output').exists()
