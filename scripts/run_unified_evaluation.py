"""Run registered pre-launch evaluations through the existing product/RAG entries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from src.observability.evaluation.sample_registry import Sample,admit_samples
from src.observability.evaluation.unified import file_ref,save,disk_path
from src.observability.evaluation.unified_adapters import CriticAdapter,RagasAdapter
from src.observability.evaluation.unified_hooks import CodeEvaluationHook,RagEvaluationHook


def terminal_summary(result,*,kind,exit_code_policy='process'):
    """Preserve process exit compatibility while exposing a usable acceptance gate."""
    codes={'passed':0,'failed':3,'insufficient_evidence':4,'evaluation_error':1,'not_applicable':5}
    decision=result if kind=='resolve' else result.get('unified_evaluation') or {}
    acceptance=decision.get('status') if kind=='resolve' else decision.get('acceptance_status')
    acceptance=acceptance if acceptance in codes else 'insufficient_evidence'
    process_code=1 if kind=='resolve' and result.get('status')=='evaluation_error' else 0
    # Scope: evaluation CLI callers. 为什么要有它：旧调用者以0判断报告生成成功；验收门禁必须显式选择，并暴露五态对应的退出码，避免把进程成功当作交付成功。
    return {'status':decision.get('status','insufficient_evidence'),
            'process_status':'completed' if process_code==0 else 'evaluation_error',
            'acceptance_status':acceptance,'acceptance_exit_code':codes[acceptance],
            'exit_code_policy':exit_code_policy,
            'exit_code':codes[acceptance] if exit_code_policy=='acceptance' else process_code}


def build_critic(profile):
    from src.libs.llm.agent_turn import OllamaChatAgentAdapter
    from src.mod_agent.local_entry import _read_ollama_model_identity
    keys={'base_url','model','digest','context_window','max_output_tokens','request_timeout_seconds','seed'}
    if set(profile)!=keys:raise ValueError('UNIFIED_JUDGE_PROFILE_FIELDS_INVALID')
    client=OllamaChatAgentAdapter(**{k:v for k,v in profile.items() if k!='digest'})
    return CriticAdapter(client,identity_reader=lambda:_read_ollama_model_identity(profile['base_url'],profile['model']),expected_digest=profile['digest'])


def build_ragas(profile):
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
    # Scope: local pre-launch calibration. 为什么要有它：校准复用已登记本地Judge，不隐式换成外部模型或申请新的数据外发权限。
    from src.libs.llm.agent_turn import validate_loopback_http_base_url
    base=validate_loopback_http_base_url(profile['base_url'])
    settings=SimpleNamespace(llm=SimpleNamespace(provider='openai',model=profile['model'],
        api_key='local-evaluation',base_url=base.rstrip('/')+'/v1',temperature=0.0,
        max_tokens=profile['max_output_tokens']))
    return RagasAdapter(RagasEvaluator(settings=settings,metrics=['faithfulness']))


def retain_sources(root):
    paths=[Path(__file__), *sorted((REPO/'src/observability/evaluation').glob('*.py')),
           REPO/'scripts/dsh_request_routing.py',REPO/'src/mod_agent/evaluation/self_rag_critic.py',
           REPO/'src/mod_agent/upstream_model_trace.py',REPO/'src/libs/llm/agent_turn.py']
    entries=[]
    for path in paths:
        rel=path.relative_to(REPO);target=root/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        data=path.read_bytes()
        with target.open('xb') as f:f.write(data)
        entries.append({'original':file_ref(path),'retained':file_ref(target)})
    return save(root/'sources.json',entries)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='kind',required=True)
    for name in ('code','rag'):
        p=sub.add_parser(name);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
        p.add_argument('--exit-code-policy',choices=('process','acceptance'),default='process',
            help='process preserves report-completion exits; acceptance: pass=0, error=1, fail=3, insufficient=4, not applicable=5')
        if name=='code':
            p.add_argument('--execution-config',type=Path,required=True);p.add_argument('--description-file',type=Path,required=True)
            p.add_argument('--runtime-required',action=argparse.BooleanOptionalAction,required=True)
        else:
            p.add_argument('--settings',type=Path,required=True);p.add_argument('--test-set',type=Path,required=True)
            p.add_argument('--judge-profile',type=Path);p.add_argument('--review-mode',choices=('assisted','trial'));p.add_argument('--collection',default='default')
            p.add_argument('--top-k',type=int,default=3);p.add_argument('--calibration-case',type=int,action='append',default=[])
    p=sub.add_parser('resolve')
    p.add_argument('--automated-result',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--exit-code-policy',choices=('process','acceptance'),default='process',
        help='Select process completion or candidate acceptance exit status')
    p=sub.add_parser('meta')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p=sub.add_parser('registry',help='Validate registered sample history without invoking a model or task')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.kind=='rag':
        args.review_mode=args.review_mode or ('trial' if args.judge_profile else 'assisted')
        if args.review_mode=='trial' and args.judge_profile is None:
            parser.error('trial review requires --judge-profile')
        if args.review_mode=='assisted' and (args.judge_profile is not None or args.calibration_case):
            parser.error('assisted mode does not call a judge; remove --judge-profile and --calibration-case')
    root=disk_path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=False)
    if args.kind=='registry':
        manifest=json.loads(args.manifest.read_bytes())
        if (manifest.get('schema_version')!='evaluation-sample-admission.v1'
                or not isinstance(manifest.get('samples'),list) or not manifest['samples']
                or not isinstance(manifest.get('registry_refs'),list)):
            raise ValueError('EVALUATION_SAMPLE_MANIFEST_REQUIRED')
        admission_ref=save(root/'admission.json',{'kind':'registry','manifest_ref':file_ref(args.manifest),
            'source_manifest_ref':retain_sources(root),'model_calls_allowed':0,'scope':'sample_registration_only'})
        try:
            result=admit_samples([Sample(**row) for row in manifest['samples']],registry_refs=manifest['registry_refs'])
            result.update(status='registered' if manifest['registry_refs'] else 'unregistered_development_metadata',
                          admission_ref=admission_ref,additional_model_calls=0)
        except (TypeError,ValueError,OSError) as error:
            result={'schema_version':'evaluation-sample-admission.v1','status':'evaluation_error',
                'error_code':str(error) if isinstance(error,ValueError) else type(error).__name__,
                'admission_ref':admission_ref,'additional_model_calls':0,'release_authority':False}
        ref=save(root/'result.json',result)
        print(json.dumps({'result_ref':ref,'status':result['status'],'scope':'sample_registration_not_business_acceptance'}))
        return 1 if result['status']=='evaluation_error' else 0
    if args.kind=='resolve':
        from src.observability.evaluation.review_resolution import resolve_rag_batch
        admission={'kind':'resolve','automated_result_ref':file_ref(args.automated_result),
                   'selection_ref':file_ref(args.selection),'source_manifest_ref':retain_sources(root)}
        save(root/'admission.json',admission)
        try:
            result=resolve_rag_batch(admission['automated_result_ref'],admission['selection_ref'],root)
        except Exception as exc:
            result={'schema_version':'unified-evaluation-resolution-batch.v1','status':'evaluation_error',
                    'error_code':str(exc) if isinstance(exc,ValueError) else type(exc).__name__,
                    'additional_model_calls':0,'release_authority':False,'admission_ref':file_ref(root/'admission.json')}
        ref=save(root/'result.json',result)
        terminal=terminal_summary(result,kind=args.kind,exit_code_policy=args.exit_code_policy)
        print(json.dumps({'result_ref':ref,**terminal,'scope':'assisted_candidate_validation'},ensure_ascii=False))
        return terminal['exit_code']
    if args.kind=='meta':
        from src.observability.evaluation.meta_evaluation import evaluate_evaluator
        manifest=json.loads(args.manifest.read_bytes())
        if manifest.get('schema_version')!='evaluation-meta-input.v1' or not isinstance(manifest.get('items'),list) or not manifest['items']:
            raise ValueError('EVALUATION_META_INPUT_REQUIRED')
        admission_ref=save(root/'admission.json',{'kind':'meta','manifest_ref':file_ref(args.manifest),
            'source_manifest_ref':retain_sources(root),'model_calls_allowed':0,
            'scope':'criterion-bound evaluator calibration, not release qualification'})
        result=evaluate_evaluator(manifest['items'])
        result.update(input_manifest_ref=file_ref(args.manifest),admission_ref=admission_ref)
        ref=save(root/'result.json',result)
        print(json.dumps({'result_ref':ref,'evaluation_count':result['evaluation_count'],
            'scope':'criterion_bound_evaluator_meta_evaluation'},ensure_ascii=False))
        return 0
    metadata=json.loads(args.metadata.read_bytes())
    if metadata.get('schema_version')!='prelaunch-evaluation-run.v1' or not metadata.get('run_id'):
        raise ValueError('UNIFIED_RUN_METADATA_REQUIRED')
    admitted={'kind':args.kind,'metadata_ref':file_ref(args.metadata),'source_manifest_ref':retain_sources(root)}
    if args.kind=='code':
        from scripts.dsh_request_routing import run_natural_request
        sample=Sample(**metadata['sample'])
        if not metadata.get('allowed_edit_paths'):raise ValueError('CODE_ALLOWED_EDIT_PATHS_REQUIRED')
        hook=CodeEvaluationHook(sample=sample,runtime_required=args.runtime_required,allowed_edit_paths=metadata['allowed_edit_paths'],
                                registry_refs=metadata.get('registry_refs',[]),input_binding={
                                    'kind':'code','description_ref':file_ref(args.description_file),
                                    'execution_config_ref':file_ref(args.execution_config)})
        admitted.update(profile_ref=file_ref(args.execution_config),request_ref=file_ref(args.description_file),runtime_required=args.runtime_required)
        admitted['sample_admission']=hook.sample_admission
        admitted['sample_input_admission']=hook.sample_input_admission
        save(root/'admission.json',admitted)
        result=run_natural_request(execution_config=str(args.execution_config),description=args.description_file.read_text(encoding='utf-8'),
            conversation_id=metadata.get('conversation_id') or metadata['run_id'],turn_id=metadata.get('turn_id','initial'),unified_evaluation=hook)
    else:
        from src.core.settings import load_settings
        from src.core.query_engine.service_factory import create_query_service
        from src.libs.evaluator.base_evaluator import NoneEvaluator
        from src.observability.evaluation.eval_runner import EvalRunner
        profile=json.loads(args.judge_profile.read_bytes()) if args.judge_profile else None
        hook=RagEvaluationHook(root/'assessments',samples={int(k):Sample(**v) for k,v in metadata['samples'].items()},
            critic=build_critic(profile) if profile else None,ragas=build_ragas(profile) if args.calibration_case else None,
            calibration_indices=args.calibration_case,registry_refs=metadata.get('registry_refs',[]))
        admitted['sample_admission']=hook.sample_admission
        settings=load_settings(args.settings)
        options={}
        if settings.llm.provider=='ollama':
            from src.mod_agent.local_entry import _read_ollama_model_identity
            identity_reader=lambda:_read_ollama_model_identity(settings.llm.base_url,settings.llm.model)
            generator_identity=identity_reader()
            if not settings.query or generator_identity.get('digest')!=settings.query.model_revision:
                raise ValueError('RAG_GENERATOR_IDENTITY_MISMATCH')
            admitted['generator_model_identity']=generator_identity
            options['model_identity_provider']=identity_reader
        service=create_query_service(settings,collection=args.collection,**options)
        admitted.update(settings_ref=file_ref(args.settings),test_set_ref=file_ref(args.test_set),judge_profile_ref=file_ref(args.judge_profile) if args.judge_profile else None,review_mode=args.review_mode,top_k=args.top_k,calibration_cases=args.calibration_case)
        save(root/'admission.json',admitted)
        # QueryService performs retrieval and generation; its real response/usage is consumed once by EvalRunner.
        report=EvalRunner(settings=settings,evaluator=NoneEvaluator(),query_service=service,
            answer_generator=service._client(),synthetic_fallback=False,unified_hook=hook).run(args.test_set,
            top_k=args.top_k,collection=args.collection,run_metadata={'run_id':metadata['run_id'],'purpose':'prelaunch_validation'})
        result=report.to_dict()
    reference=save(root/'result.json',result)
    terminal=terminal_summary(result,kind=args.kind,exit_code_policy=args.exit_code_policy)
    print(json.dumps({'result_ref':reference,**terminal,'scope':'prelaunch_evaluation_not_production_traffic'},ensure_ascii=False))
    return terminal['exit_code']


if __name__=='__main__':raise SystemExit(main())
