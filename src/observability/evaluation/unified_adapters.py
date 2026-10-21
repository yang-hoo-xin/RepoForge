"""Adapters for existing execution reports and existing semantic evaluators."""
from __future__ import annotations
from dataclasses import asdict, replace
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
from .unified import file_ref, observation, read_ref, save


def zero_model_cost():
    return dict(model_calls=0,input_tokens=0,output_tokens=0,cached_input_tokens=0)


def response(rows,cost,identity):
    return {'results':rows,'cost':cost,'evaluator_identity':identity}


class ExecutionReportAdapter:
    """Consume the normal owner's bound validation; never compile a second time."""
    def __call__(self,binding,payload,criteria,root):
        reference=payload['validation_report_ref']
        report=json.loads(read_ref(reference))
        if report.get('status')=='passed' and report.get('execution_origin')!='executed':
            raise ValueError('EXECUTION_OBSERVATION_REQUIRED')
        identity=payload['candidate_identity']
        for key in ('materialized_code_identity','model_response_sha256','generated_changes_sha256'):
            if report.get(key)!=identity.get(key) or identity.get(key) is None:
                raise ValueError('EXECUTION_REPORT_CANDIDATE_MISMATCH')
        details={}
        if report.get('verification_ref'):
            details=json.loads(read_ref(report['verification_ref']))
            if details.get('materialized_code_identity')!=identity['materialized_code_identity']:
                raise ValueError('EXECUTION_OBSERVATION_CANDIDATE_MISMATCH')
            if details.get('built_artifact_ref'):read_ref(details['built_artifact_ref'])
        full=report.get('validation_results',{}).get('full_compilation',{})
        rows=[]
        for c in criteria:
            reason='Existing owner observation';status='insufficient_evidence'
            if c.dimension=='compilation':
                if full.get('status')=='passed' and full.get('all_required_files_compiled') is True:
                    compiled=json.loads(read_ref(full['report_ref']))
                    if (compiled.get('status')!='passed' or compiled.get('execution_origin')!='executed'
                            or compiled.get('all_required_files_compiled') is not True
                            or compiled.get('materialized_code_identity')!=identity['materialized_code_identity']
                            or (compiled.get('model_response_ref') or {}).get('sha256')!=identity['model_response_sha256']
                            or (details and details.get('built_artifact_ref')!=compiled.get('built_artifact_ref'))):
                        raise ValueError('COMPILED_ARTIFACT_CANDIDATE_MISMATCH')
                    read_ref(compiled['built_artifact_ref']);status='passed'
                elif report.get('failure_owner')=='patch_implementation' and report.get('status') in {'compilation_failed','failed'}:
                    status='failed'
                elif report.get('failure_owner') in {'validator','infrastructure'}:
                    status='evaluation_error'
            elif c.dimension=='behavior':
                checks=details.get('checks',[])
                if report.get('status')=='passed' and checks and all(x.get('passed') is True for x in checks):
                    status='passed'
                elif report.get('status')=='behavior_failed' and report.get('failure_owner')=='patch_implementation':
                    status='failed'
                elif report.get('failure_owner') in {'validator','infrastructure'}:
                    status='evaluation_error'
            elif c.dimension=='modification_scope':
                allowed=payload.get('allowed_edit_paths')
                actual=identity['materialized_code_identity'].get('files')
                if allowed is not None:
                    for path in [*allowed,*(actual or [])]:
                        if not isinstance(path,str) or not path or '\\' in path or ':' in path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
                            raise ValueError('CODE_SCOPE_PATH_INVALID')
                    status='passed' if actual and set(actual)<=set(allowed) else 'failed'
                    reason='Materialized candidate paths against registered task scope'
            elif c.dimension=='runtime':
                # Scope: generic managed probe reports. 为什么要有它：真实游戏的专用证据合同未接入时，不能把managed runtime字段或布尔值当成实机通过。
                reason='Native runtime evidence must be verified by its task-specific owner'
            else:
                raise ValueError('EXECUTION_CRITERION_UNSUPPORTED')
            refs=[reference]+([report['verification_ref']] if details else [])
            rows.append(observation(binding,c,status,evidence=refs,reason=reason))
        return response(rows,dict(zero_model_cost(),execution_reused=True),
                        {'evaluation_method':'execution','adapter_ref':file_ref(__file__),'owner_identity':report.get('verification_identity')})


class CriticAdapter:
    """Reuse the Critic request/schema/quote validation, outside its training filter."""
    def __init__(self,client,*,identity_reader,expected_digest):
        self.client=client;self.identity_reader=identity_reader;self.expected_digest=expected_digest

    def __call__(self,binding,payload,criteria,root):
        from jsonschema import Draft202012Validator
        from src.core.trace.execution_trace import ExecutionTrace
        from src.mod_agent import upstream_model_trace
        from src.mod_agent.evaluation import self_rag_critic as critic
        contexts=payload.get('contexts',[])
        if not payload.get('answer'):
            return response([observation(binding,c,'insufficient_evidence',reason='No answer to evaluate') for c in criteria],zero_model_cost(),{'adapter_ref':file_ref(__file__),'critic_ref':file_ref(critic.__file__)})
        candidate={'question':payload['query'],'answer':payload['answer'],
                   'contexts':[{'context_index':i,'chunk_id':v.get('id',v.get('chunk_id',str(i))), 'text':v['text']} for i,v in enumerate(contexts,1)],
                   'source_constraint':{'decision_stage':'before_retrieval','mode':'supplied_contexts_only'}}
        request=critic.critic_request(candidate)
        schema=deepcopy(request.response_format.schema)
        schema['properties']['evidence_quotes']['items']['properties']['context_index']['maximum']=max(1,len(contexts))
        request=replace(request,response_format=replace(request.response_format,name='unified_critic_labels_v1',schema=schema))
        request_ref=save(root/'request.json',asdict(request))
        events={};cost=dict(model_calls=None,input_tokens=None,output_tokens=None,cached_input_tokens=None)
        identity={'adapter_ref':file_ref(__file__),'critic_ref':file_ref(critic.__file__),
                  'provider':self.client.provider,'model':self.client.model,'digest':self.expected_digest,
                  'context_window':getattr(self.client,'context_window',None),'request_ref':request_ref,
                  'client_configuration':{key:getattr(self.client,key,None) for key in ('max_output_tokens','context_window','seed','request_timeout_seconds')}}
        before=None;annotation=None;bindings=None;error=None;failure={};raw_refs=[request_ref]
        trace=ExecutionTrace(binding.run_id,binding.conversation_id,'local',recorder=lambda r:events.update({r['id']:r}),root_name='evaluation.critic')
        try:
            before=self.identity_reader()
            if before.get('digest')!=self.expected_digest:raise ValueError('CRITIC_MODEL_IDENTITY_MISMATCH')
            with trace.activate():
                result=upstream_model_trace.trace_upstream_model_call(role='self_rag_critic',provider=self.client.provider,
                    model=self.client.model,input_payload=candidate,request_document=asdict(request),body_root=root,
                    invoke=lambda:self.client.complete_turn(request))
            raw_refs.append(save(root/'response.json',{'provider':result.provider,'model':result.model,
                'content':result.content,'stop_reason':result.stop_reason,'usage':asdict(result.usage)}))
            usage=upstream_model_trace._usage(result)
            if usage:cost.update({k:usage.get(k) for k in ('input_tokens','output_tokens','cached_input_tokens')})
            if self.identity_reader()!=before:raise ValueError('CRITIC_MODEL_IDENTITY_CHANGED')
            if result.model!=self.client.model or result.provider!=self.client.provider or result.stop_reason!='stop' or result.tool_calls:
                raise ValueError('CRITIC_RESPONSE_INCOMPLETE')
            annotation=critic._strict_json(result.content)
            Draft202012Validator(schema).validate(annotation)
            bindings=critic.annotation_bindings(candidate,annotation)
            if critic.annotation_semantic_errors(candidate,annotation):raise ValueError('CRITIC_INCONSISTENT_LABELS')
            if bindings['core_claim'] is None or any(q['span'] is None for q in bindings['evidence_quotes']):
                raise ValueError('CRITIC_QUOTE_NOT_IN_BOUND_INPUT')
            if annotation['core_verdict'] in {'supported','contradicted'}:
                required='supports_core' if annotation['core_verdict']=='supported' else 'contradicts_core'
                if not any(q['relation']==required for q in bindings['evidence_quotes']):
                    raise ValueError('CRITIC_SUPPORTING_QUOTE_REQUIRED')
        except Exception as exc:
            error=type(exc).__name__
            diagnostic=getattr(exc,'diagnostic',{})
            # Scope: known local context gate only. 为什么要有它：保留预检拒绝的可解释证据，不能把异常对象中的任意数据或凭据直接写入日志。
            if diagnostic.get('failure_classification')=='provider_request_context_budget_exceeded':
                failure={'failure_boundary':'provider_request_gate',
                         'failure_classification':'provider_request_context_budget_exceeded',
                         'context_admission':{k:v for k,v in diagnostic.get('context_admission',{}).items()
                             if k in {'admitted','configured_num_ctx','num_predict','provider_body_utf8_bytes',
                                      'conservative_required_context','estimated_required_context','estimated_prompt_tokens'}
                             and type(v) in {bool,int}}}
        calls=[r for r in events.values() if r['name']=='model.request']
        transports=[r['attributes'].get('transport_started') for r in calls]
        cost['model_calls']=sum(v is True for v in transports) if transports and all(type(v) is bool for v in transports) else (0 if not calls else None)
        audit_ref=save(root/'judgment.json',{'binding':asdict(binding),'annotation':annotation,'quote_bindings':bindings,
                                           'error_type':error,**failure,'model_identity':before,'raw_refs':raw_refs,'events':list(events.values()),'cost':cost})
        rows=[]
        for c in criteria:
            status='insufficient_evidence';reason='Judge did not establish this criterion'
            if error:status='evaluation_error';reason=error
            else:
                unsupported=annotation['unsupported_factual_claims'];kind=annotation['answer_kind']
                if c.dimension=='grounding':
                    if unsupported is True or annotation['core_verdict']=='contradicted':status='failed'
                    elif unsupported is False and annotation['labels']['support']=='fully_supported':status='passed'
                    elif unsupported is False and kind=='refusal':status='passed'
                elif c.dimension=='task_fulfillment':
                    expected=payload.get('expected_behavior')
                    if expected=='refuse':status='passed' if kind=='refusal' and unsupported is False else 'insufficient_evidence' if kind=='unknown' else 'failed'
                    elif expected=='answer':
                        if kind=='refusal' or annotation['completeness']=='incomplete':status='failed'
                        elif kind in {'factual','mixed','nonfactual'} and annotation['completeness']=='complete':status='passed'
                elif c.dimension=='evidence_relevance':
                    status={'relevant':'passed','irrelevant':'failed'}.get(annotation['labels']['relevance'],'insufficient_evidence')
                else:raise ValueError('CRITIC_CRITERION_UNSUPPORTED')
                reason=annotation['issue'] or 'Bound Critic annotation'
            rows.append(observation(binding,c,status,evidence=[audit_ref],reason=reason))
        return response(rows,cost,identity)


class RagasAdapter:
    """Ragas scores remain attributable calibration/primary observations."""
    def __init__(self,evaluator):self.evaluator=evaluator
    def __call__(self,binding,payload,criteria,root):
        observation_data={};scores={};error=None
        try:
            scores=self.evaluator.evaluate(query=payload['query'],retrieved_chunks=payload['contexts'],
                generated_answer=payload['answer'],observation_sink=lambda value:observation_data.update(value))
        except Exception as exc:error=type(exc).__name__
        record=save(root/'ragas.json',{'binding':asdict(binding),'scores':scores,'observation':observation_data,'error_type':error})
        rows=[]
        for c in criteria:
            value=scores.get('faithfulness' if c.dimension=='grounding' else c.dimension)
            if error or value is None:status='evaluation_error'
            elif c.threshold is None:status='insufficient_evidence'
            else:status='passed' if value>=c.threshold else 'failed'
            rows.append(observation(binding,c,status,evidence=[record],value=value,
                reason=error or ('No registered acceptance threshold' if c.threshold is None else 'Registered Ragas threshold')))
        identity={'adapter_ref':file_ref(__file__),'evaluator':type(self.evaluator).__name__}
        identity.update(observation_data.get('evaluator_identity',{}))
        return response(rows,observation_data.get('cost',dict(model_calls=None,input_tokens=None,output_tokens=None,cached_input_tokens=None)),identity)