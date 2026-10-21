"""Automatic unified reports at existing RAG and natural-request entry boundaries."""
from dataclasses import asdict
from collections import Counter
import json
import math
from pathlib import Path
from uuid import uuid4
from .sample_registry import Sample,admit_samples,bind_sample_input
from .unified import Binding,Criterion,EvaluationSession,STATUSES,read_ref,save
from .unified_adapters import ExecutionReportAdapter,zero_model_cost,response


def summarize_rag_batch(queries):
    """Summarize observations and acceptance separately over every registered query."""
    def statuses(field):
        values=[]
        for query in queries:
            result=query.unified_evaluation
            if result is None:
                values.append('insufficient_evidence');continue
            value=result.get(field)
            if value is None and result.get('status')=='evaluation_error':value='evaluation_error'
            values.append(value if value in STATUSES else 'evaluation_error')
        return Counter(values)
    def aggregate(counts):
        return next((s for s in ('failed','evaluation_error','insufficient_evidence') if counts[s]),
                    'passed' if counts['passed'] else 'not_applicable')
    counts=statuses('status');acceptance=statuses('acceptance_status')
    # Scope: all registered batch rows. 为什么要有它：跳过、失败和未评测题均占分母，试验裁判的通过不能转成正式验收通过。
    return {'schema_version':'unified-evaluation-batch.v1','query_count':len(queries),
            'status':aggregate(counts),'acceptance_status':aggregate(acceptance),
            'counts':{s:counts[s] for s in sorted(STATUSES)},
            'acceptance_counts':{s:acceptance[s] for s in sorted(STATUSES)},
            'missing_report_count':sum(q.unified_evaluation is None for q in queries),
            'decision_scope':'candidate_validation_not_final_delivery','release_authority':False}


class RagEvaluationHook:
    def __init__(self,output_root,*,samples,critic=None,ragas=None,calibration_indices=(),registry_refs=()):
        self.root=Path(output_root).resolve();self.samples=dict(samples)
        self.registry_refs=tuple(registry_refs)
        self.sample_admission=admit_samples(list(self.samples.values()),registry_refs=self.registry_refs)
        self.input_bindings={}
        self.critic=critic if critic is not None else self.pending_review;self.ragas=ragas;self.calibration=set(calibration_indices)
        if not self.calibration<=set(self.samples) or (self.calibration and ragas is None):
            raise ValueError('RAG_CALIBRATION_SELECTION_INVALID')
    def validate_run(self,cases,metadata):
        if set(self.samples)!=set(range(len(cases))):raise ValueError('RAG_SAMPLE_REGISTRY_INCOMPLETE')
        if not isinstance(metadata.get('run_id'),str) or not metadata['run_id']:
            raise ValueError('RAG_EVALUATION_RUN_ID_REQUIRED')
        if any(case.expected_behavior not in {'answer','refuse'} for case in cases):
            raise ValueError('RAG_EXPECTED_BEHAVIOR_REQUIRED')
        for index,case in enumerate(cases):
            binding=({'kind':'rag','test_set_ref':metadata['evaluation_test_set_ref'],'case_index':index}
                     if metadata.get('evaluation_test_set_ref') else None)
            bind_sample_input(self.samples[index],self.sample_admission,binding,query=case.query)
            self.input_bindings[index]=binding
            if (not isinstance(case.expected_chunk_ids,list)
                    or any(not isinstance(item,str) or not item for item in case.expected_chunk_ids)):
                raise ValueError('RAG_EXPECTED_CHUNK_IDS_INVALID')
            requirement=case.retrieval_requirement
            if requirement is not None and (
                    not isinstance(requirement,dict) or set(requirement)!={'metric','threshold'}
                    or not isinstance(requirement['metric'],str) or requirement['metric'] not in {'hit_rate','mrr'}
                    or type(requirement['threshold']) not in (int,float)
                    or not math.isfinite(requirement['threshold'])
                    or not 0<requirement['threshold']<=1 or not case.expected_chunk_ids):
                raise ValueError('RAG_RETRIEVAL_REQUIREMENT_INVALID')
    def __call__(self,qr,metadata):
        sample=self.samples[qr.case_index];directory=self.root/uuid4().hex
        from src.libs.evaluator.custom_evaluator import CustomEvaluator
        expected=qr.case_metadata.get('expected_chunk_ids',[])
        requirement=qr.case_metadata.get('retrieval_requirement')
        retrieval_status=qr.retrieval_measurement.status
        measured=retrieval_status=='observed' and bool(expected)
        metrics={}
        if measured:
            metrics=(CustomEvaluator().evaluate(qr.query,qr.retrieved_chunk_ids,ground_truth={'ids':expected})
                     if qr.retrieved_chunk_ids else {'hit_rate':0.0,'mrr':0.0})
        diagnostic={'expected_chunk_ids':expected,'expected_sources':qr.case_metadata.get('expected_sources',[]),
            'retrieved_chunk_ids':qr.retrieved_chunk_ids,'metrics':metrics,
            'measurement_status':retrieval_status if expected else 'not_applicable',
            'decision_use':'acceptance' if requirement is not None else 'diagnostic_only'}
        payload={'query':qr.query,'answer':qr.generated_answer,'contexts':qr.retrieved_contexts,
                 'evaluation_input_binding':self.input_bindings.get(qr.case_index),
                 'expected_behavior':qr.case_metadata.get('expected_behavior'),
                 'answer_measurement':qr.answer_measurement.to_dict(),
                  'retrieval_measurement':qr.retrieval_measurement.to_dict(),
                  'citation_measurement':qr.citation_measurement.to_dict(),
                  'citation_required':qr.citation_required,
                  'retrieval_diagnostic':diagnostic,'retrieval_requirement':requirement}
        subject=save(directory/'candidate.json',payload)
        task_id=sample.dataset_id+':'+sample.sample_id
        # Scope: independent single-query evaluation conversations. 为什么要有它：此处明确分配评测会话ID，不把多个独立问题伪称为真实用户多轮对话。
        binding=Binding(task_id,metadata['run_id']+':'+sample.sample_id,metadata['run_id'],subject)
        criteria=[Criterion('generation','generation','pipeline',method='execution'),
                  Criterion('citation','citation_policy','pipeline',method='execution',
                            applicable=payload['citation_required']),
                  Criterion('grounding','grounding','critic'),
                  Criterion('fulfillment','task_fulfillment','critic'),
                  Criterion('context_relevance','evidence_relevance','critic',required=False)]
        # Scope: an explicit case acceptance requirement only. 为什么要有它：旧expected_chunk_ids仅是诊断真值，只有登记的阈值可以升级为验收；复用hit_rate/MRR语义，不引入全命中假设。
        if requirement is not None:
            criteria.append(Criterion('retrieval','retrieval_quality','pipeline',method='execution',
                                      threshold=requirement['threshold']))
        evaluators={'pipeline':self.pipeline,'critic':self.critic}
        if qr.case_index in self.calibration:
            criteria.append(Criterion('ragas_grounding','grounding','ragas',required=False,role='calibration',threshold=1.0))
            evaluators['ragas']=self.ragas
        report=EvaluationSession(directory/'evaluations',criteria=criteria,evaluators=evaluators,
                                 rubric_version='rag-grounded-answer-v3',sample=sample,
                                 registry_refs=self.registry_refs,
                                 input_binding=self.input_bindings.get(qr.case_index)).evaluate(binding,payload)
        return report
    @staticmethod
    def pending_review(binding,payload,criteria,root):
        from .unified import observation
        # Scope: explicitly assisted workflow. 为什么要有它：未配置合格自动裁判时正常任务仍可产出候选和复核请求，不能隐式调用退役模型或把未评分当成通过。
        request=save(root/'review-request.json',{'schema_version':'semantic-review-request.v1',
            'binding':asdict(binding),'required_dimensions':[c.dimension for c in criteria if c.required],
            'optional_dimensions':[c.dimension for c in criteria if not c.required],
            'candidate_ref':binding.candidate_ref,'review_status':'pending',
            'expected_review_schema':'assisted-semantic-review.v1',
            'reason':'Review answer against bound contexts and requirements; keep original evidence and cite source spans.',
            'human_review':False,'release_authority':False})
        return response([observation(binding,c,'insufficient_evidence',evidence=[request],reason='Awaiting explicitly selected semantic review') for c in criteria],
            zero_model_cost(),{'evaluation_method':'pending_review','model_invoked':False,'request_ref':request})

    @staticmethod
    def pipeline(binding,payload,criteria,root):
        from .unified import observation
        record=save(root/'stages.json',{'binding':asdict(binding),'answer':payload['answer_measurement'],
                                       'retrieval':payload['retrieval_measurement'],
                                       'citation':payload['citation_measurement'],
                                       'retrieval_diagnostic':payload['retrieval_diagnostic'],
                                       'retrieval_requirement':payload['retrieval_requirement']})
        rows=[]
        for criterion in criteria:
            value=None
            if criterion.dimension=='generation':
                states=[payload[k]['status'] for k in ('answer_measurement','retrieval_measurement')]
                status='failed' if 'failed_to_measure' in states else 'passed' if payload['answer'] else 'insufficient_evidence'
                reason='Normal query execution stages'
            elif criterion.dimension=='citation_policy':
                citation=payload['citation_measurement']
                # Scope: an applicable deterministic citation policy. 为什么要有它：引用缺失或越界是候选未满足必需项，不能在统一门禁里被语义高分掩盖。
                status={'observed':'passed','failed_to_measure':'failed',
                        'not_applicable':'insufficient_evidence'}.get(citation['status'],'evaluation_error')
                reason=citation.get('error_message') or 'Required context citation policy'
            elif criterion.dimension=='retrieval_quality':
                requirement=payload['retrieval_requirement']
                diagnostic=payload['retrieval_diagnostic']
                value=diagnostic['metrics'].get(requirement['metric'])
                if diagnostic['measurement_status']=='failed_to_measure':status='evaluation_error'
                elif value is None:status='insufficient_evidence'
                else:status='passed' if value>=requirement['threshold'] else 'failed'
                reason='Explicit retrieval requirement: '+requirement['metric']
            else:
                raise ValueError('RAG_PIPELINE_CRITERION_UNSUPPORTED')
            rows.append(observation(binding,criterion,status,evidence=[record],reason=reason,value=value))
        return response(rows,zero_model_cost(),{'evaluation_method':'execution','owner':'EvalRunner execution observations','model_calls_reused':True})


class CodeEvaluationHook:
    def __init__(self,*,sample:Sample,runtime_required:bool,allowed_edit_paths=None,registry_refs=(),input_binding=None):
        if type(runtime_required) is not bool:raise ValueError('CODE_RUNTIME_APPLICABILITY_REQUIRED')
        self.sample=sample;self.runtime_required=runtime_required
        self.registry_refs=tuple(registry_refs)
        self.sample_admission=admit_samples([sample],registry_refs=self.registry_refs)
        self.input_binding=input_binding
        self.sample_input_admission=bind_sample_input(sample,self.sample_admission,input_binding)
        self.allowed_edit_paths=None if allowed_edit_paths is None else tuple(allowed_edit_paths)
    def __call__(self,result,*,routing_root):
        product=result.get('product_result') or {}
        if not product.get('run_id') or not product.get('run_root'):
            return {'status':'insufficient_evidence','reason':'No product run/candidate','candidate_reports':[]}
        run_root=Path(product['run_root']).resolve();run_id=product['run_id']
        selected=[];skipped=[];decisions=[];selected_response=None
        def skip(evaluation,receipt_ref,index,*,status,code,reason):
            skipped.append({'receipt_ref':receipt_ref,'report_ref':evaluation.get('report_ref'),
                'evaluation_index':index,'model_response_sha256':evaluation.get('model_response_sha256'),
                'admission':evaluation.get('admission'),'failure_owner':evaluation.get('failure_owner'),
                'status':status,'code':code,'reason':reason})
            decisions.append((evaluation.get('model_response_sha256'),status,status))
        receipts=product.get('patch_execution_receipts',[])
        if isinstance(receipts,dict):receipts=[receipts]
        for receipt in receipts:
            ref={'path':receipt['path'],'sha256':receipt['sha256']}
            if not Path(ref['path']).resolve().is_relative_to(run_root):raise ValueError('CODE_RECEIPT_OUTSIDE_RUN')
            value=json.loads(read_ref(ref));factory=value.get('factory_receipt') or {}
            selected_response=(factory.get('model_response') or {}).get('output_sha256')
            for index,evaluation in enumerate((factory.get('patch_agent_loop') or {}).get('behavior_evaluations',[]),1):
                report_ref=evaluation.get('report_ref')
                if report_ref and not Path(report_ref['path']).resolve().is_relative_to(run_root):raise ValueError('CODE_REPORT_OUTSIDE_RUN')
                # 范围：实际产品owner的行为报告journal。为什么要有它：被拒绝或尚未接纳的观察保留供追查，但不能越过原入口的输入、源码和观察绑定校验取得通过资格。
                if evaluation.get('admission')!='accepted':
                    status='insufficient_evidence' if evaluation.get('admission') in (None,'pending') else 'evaluation_error'
                    skip(evaluation,ref,index,status=status,code='CODE_REPORT_NOT_ADMITTED',
                         reason='Product owner did not admit this behavior observation')
                    continue
                if not report_ref:
                    skip(evaluation,ref,index,status='insufficient_evidence',code='CODE_REPORT_REFERENCE_MISSING',
                         reason='Admitted journal observation has no retained report')
                    continue
                report=json.loads(read_ref(report_ref))
                identity={k:report.get(k) for k in ('materialized_code_identity','model_response_sha256','generated_changes_sha256','source_snapshot_sha256')}
                if any(v is None for v in identity.values()):
                    skip(evaluation,ref,index,status='insufficient_evidence',code='CODE_CANDIDATE_BINDING_INCOMPLETE',
                         reason='No complete candidate/source binding')
                    continue
                shared=('status','failure_owner','owner_binding_ids','observations','failure_state_observations',
                        'execution_origin','verification_identity','verification_ref','verification_summary','validation_results')
                if (any(evaluation.get(key)!=value for key,value in identity.items())
                        or any(evaluation.get(key)!=report.get(key) for key in shared)):
                    skip(evaluation,ref,index,status='evaluation_error',code='CODE_RECEIPT_REPORT_BINDING_MISMATCH',
                         reason='Admitted journal and persisted behavior report disagree')
                    continue
                payload={'candidate_identity':identity,'validation_report_ref':report_ref,
                         'receipt_ref':ref,'evaluation_index':index,'allowed_edit_paths':self.allowed_edit_paths,
                         'evaluation_input_binding':self.input_binding}
                directory=Path(routing_root)/'unified-evaluation'/uuid4().hex
                candidate=save(directory/'candidate.json',payload)
                conversation=result.get('conversation_id') or result['route_id']
                binding=Binding(self.sample.dataset_id+':'+self.sample.sample_id,conversation,run_id,candidate)
                criteria=[Criterion('compilation','compilation','execution',method='execution'),Criterion('behavior','behavior','execution',method='execution'),Criterion('scope','modification_scope','execution',method='execution'),
                          Criterion('runtime','runtime','execution',applicable=self.runtime_required,method='execution')]
                result_report=EvaluationSession(directory/'evaluations',criteria=criteria,
                    evaluators={'execution':ExecutionReportAdapter()},rubric_version='code-candidate-v3',sample=self.sample,
                    registry_refs=self.registry_refs,input_binding=self.input_binding).evaluate(binding,payload)
                selected.append(result_report)
                decisions.append((identity['model_response_sha256'],result_report['status'],result_report['acceptance_status']))
        # 范围：当前最终模型响应的最新观察。为什么要有它：最新拒绝或缺证据必须覆盖同候选较早通过，也不能把另一候选的成功移给最终候选。
        current=[row for row in decisions if selected_response is not None and row[0]==selected_response]
        return {'status':current[-1][1] if current else 'insufficient_evidence',
                'acceptance_status':current[-1][2] if current else 'insufficient_evidence',
                'selected_model_response_sha256':selected_response,
                'decision_scope':'candidate_validation_not_final_delivery','candidate_reports':selected,
                'skipped_reports':skipped,'product_status':result.get('status'),'product_run_id':run_id,
                'sample':asdict(self.sample),'runtime_required':self.runtime_required}
