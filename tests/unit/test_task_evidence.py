from copy import deepcopy
import pytest
from src.observability.evaluation.task_evidence import summarize_observations,collect_task_evidence
from src.observability.evaluation.unified import save


def row(i='1',usage=None,transport=True):
 return {'id':i,'trace_id':'trace','name':'model.request','elapsed_ms':20,'attributes':{'transport_started':transport,'usage':usage or {}},'status':'failed' if usage is None else 'completed'}


def test_failed_model_usage_stays_unknown_not_zero():
 r=summarize_observations([row(usage={'input_tokens':10,'output_tokens':2}),row('2')])
 assert r['transport_calls']=={'known_subtotal':2,'complete':True}
 assert r['usage']['input_tokens']=={'known_subtotal':10,'complete':False}
 assert r['usage']['cached_input_tokens']['complete'] is False


def test_identical_observation_is_not_double_counted():
 a=row(usage={'input_tokens':10,'output_tokens':2,'cached_input_tokens':0})
 r=summarize_observations([a,deepcopy(a)])
 assert r['observation_count']==1 and r['usage']['input_tokens']['known_subtotal']==10


def test_conflicting_same_identity_is_rejected():
 a=row();b=deepcopy(a);b['elapsed_ms']=40
 with pytest.raises(ValueError,match='CONFLICTING'):summarize_observations([a,b])


def test_unobserved_transport_is_not_guessed():
 r=summarize_observations([row(transport=None)])
 assert r['transport_calls']['complete'] is False
 assert r['elapsed_aggregation']=='not_summed_nested_stages'


def test_timeout_without_domain_result_retains_missing_evidence(tmp_path):
 process=save(tmp_path/'process.json',{'timeout':True,'wall_seconds':600})
 qualification=save(tmp_path/'qualification.json',{'task_id':'C4'})
 r=collect_task_evidence(task_id='C4',attempt_id='new',result_path=tmp_path/'missing.json',process_ref=process,
  qualification_ref=qualification,output_root=tmp_path/'out',revision_scope='current')
 assert r['timed_out'] and r['cost'] is None and r['completeness']=='partial'
 assert not r['whole_task_delivered'] and r['collection_model_calls']==0
