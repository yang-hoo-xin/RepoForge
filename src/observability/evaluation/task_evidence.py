"""Collect result, trajectory, resource cost and evaluator evidence from one owned run."""
from collections import Counter
import json
from pathlib import Path
from .unified import file_ref, read_ref, save, disk_path


def summarize_observations(observations):
    rows=list(observations.values()) if isinstance(observations,dict) else list(observations)
    unique={}
    for row in rows:
        key=(row['trace_id'],row['id'])
        if key in unique and unique[key]!=row:
            raise ValueError('TASK_EVIDENCE_CONFLICTING_OBSERVATION')
        unique[key]=row
    rows=list(unique.values());models=[r for r in rows if r['name']=='model.request']
    def count(field):
        values=[r.get('attributes',{}).get(field) for r in models]
        return {'known_subtotal':sum(v is True for v in values),'complete':all(type(v) is bool for v in values)}
    def usage(field):
        values=[r.get('attributes',{}).get('usage',{}).get(field) for r in models]
        return {'known_subtotal':sum(v for v in values if type(v) is int and v>=0),
                'complete':all(type(v) is int and v>=0 for v in values)}
    # Scope: raw observed stages. 为什么要有它：嵌套、并行阶段不能相加冒充总耗时，重复事件和未知用量不能制造节省。
    return {'observation_count':len(rows),'event_counts':dict(Counter(r['name'] for r in rows)),
        'model_request_observations':len(models),'transport_calls':count('transport_started'),
        'usage':{k:usage(k) for k in ('input_tokens','output_tokens','cached_input_tokens')},
        'stages':[{'id':r['id'],'trace_id':r['trace_id'],'name':r['name'],'parent_id':r.get('parent_id'),
                   'status':r.get('status'),'elapsed_ms':r.get('elapsed_ms')} for r in rows],
        'elapsed_aggregation':'not_summed_nested_stages','source_query_events_are_not_total_tool_calls':True}


def collect_task_evidence(*,task_id,attempt_id,result_path,process_ref,qualification_ref,output_root,revision_scope):
    process=json.loads(read_ref(process_ref));qualification=json.loads(read_ref(qualification_ref))
    record={'schema_version':'task-evidence-bundle.v1','task_id':task_id,'attempt_id':attempt_id,
            'revision_scope':revision_scope,'process_ref':process_ref,'evaluator_qualification_ref':qualification_ref,
            'wall_seconds':process.get('wall_seconds'),'timed_out':process.get('timeout'),
            'outcome':None,'trajectory':None,'cost':None,'evaluation':None,'missing':[],
            'collection_model_calls':0,'collection_builds':0,'whole_task_delivered':False}
    if not disk_path(result_path).exists():
        record['missing'].append('normal_entry_result_missing_after_process_terminal')
    else:
        record['result_ref']=file_ref(result_path);result=json.loads(read_ref(record['result_ref']))
        product=result.get('product_result') or {}
        record['outcome']={'status':result.get('status'),'conversation_id':result.get('conversation_id'),
            'run_id':product.get('run_id'),'product_model_call_count':product.get('model_call_count'),
            'product_source_writes':product.get('product_source_writes_performed'),
            'actual_write_performed':product.get('actual_write_performed')}
        record['evaluation']=result.get('unified_evaluation')
        record['cost']={'routing_usage':result.get('routing_usage'),
            'routing_transport_calls':result.get('routing_transport_calls'),
            'routing_elapsed_seconds':result.get('routing_elapsed_seconds'),
            'product_usage':None,'scope':'routing and product recorded separately, never added to nested stage durations'}
        trace=product.get('execution_trace') or {}
        location=(trace.get('local_observation_source') or {}).get('path')
        if location and product.get('run_root') and Path(location).resolve().is_relative_to(Path(product['run_root']).resolve()):
            manifest_ref=file_ref(location);manifest=json.loads(read_ref(manifest_ref))
            if 'execution_observations' not in manifest:record['missing'].append('execution_observations_missing')
            else:
                measured=summarize_observations(manifest['execution_observations'])
                record['trajectory']={'manifest_ref':manifest_ref,'json_pointer':'/execution_observations',**measured}
                record['cost']['product_usage']=measured['usage']
        else:record['missing'].append('bound_product_trace_unavailable')
        if record['evaluation'] is None:record['missing'].append('unified_evaluation_unavailable')
    record['completeness']='partial' if record['missing'] else 'four_evidence_sections_present'
    record['interpretation']='Section presence does not prove task success, judge qualification for native runtime, or causal improvement.'
    return dict(record,report_ref=save(output_root/'task-evidence.json',record))
