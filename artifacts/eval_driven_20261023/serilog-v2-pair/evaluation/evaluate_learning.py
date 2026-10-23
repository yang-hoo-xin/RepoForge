"""Single-run independent candidate evaluation; not a paired strategy verdict.

为什么要有它：本轮真实候选须独立核验，不能借旧配对协议、旧分数或参考实现制造完成。
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.product_trial_batch import _verify_v2_live_submission
from src.mod_agent.evaluation.strategy_repository_evaluator import _configuration, _report
from src.mod_agent.evaluation.task_measurement import measure_product_task
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_execution import _admitted_response, _worker_binding
from src.mod_agent.task_agent_v2_state import TaskAgentV2State

MAIN = Path('C:/tmp/rf-serilog-v2-learning-main')
PLAN_REF = {'path': str(MAIN / 'frozen-plan.json'),
            'sha256': '60876e7c687c73f34af4186c7d58463dbef579155055f854e1ca7129300f72b6'}
EVALUATOR_REF = {'path': 'C:/tmp/rf-serilog-v2-pair-eval/qualification-final/evaluator.json',
                 'sha256': '4b6f85bd95d7ed30a555f4ffb58f207b618f413a4c941b1cf8788014ac9f3bef'}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def terminal_gate(process, started, plan, plan_ref):
    require(process.get('finished_at') and process.get('status') in
            {'exited', 'timed_out', 'infrastructure_failed'}, 'PROCESS_NOT_TERMINAL')
    require(started['plan_ref'] == plan_ref and started['run_id'] == plan['run_id'], 'RUN_CHANGED')
    require(plan['platform_ref']['path'] in process['argv'] and plan['platform_ref']['sha256'] in process['argv'],
            'PROCESS_PLATFORM_CHANGED')


def verify_origin(native, candidate, read):
    origin = candidate['model_origin']
    matches = [r for r in native['completed_tool_records'] if r['call_id'] == origin['call_id']]
    require(len(matches) == 1, 'SUBMIT_ORIGIN_MISSING')
    call = matches[0]
    require(call['name'] == 'candidate_submit' and all(call[k] == v for k, v in origin.items()), 'SUBMIT_ORIGIN_CHANGED')
    response, arguments, output = (read(call[k]) for k in ('response_ref', 'arguments_ref', 'result_ref'))
    _admitted_response(response, native['model_profile'])
    require([r for r in response['tool_calls'] if r['call_id'] == call['call_id']] ==
            [{'call_id': call['call_id'], 'name': call['name'], 'arguments': arguments}], 'RESPONSE_ARGUMENTS_CHANGED')
    require(candidate['parent_revision_ref'] == arguments['parent_revision_ref'] and
            len(candidate['changes']) == len(arguments['edits']) and all(
                all(change.get(k) == v for k, v in edit.items())
                for change, edit in zip(candidate['changes'], arguments['edits'])), 'CANDIDATE_EDIT_CHANGED')
    loop = read(native['loop_result_ref'])
    require(loop['termination'] == native['termination'] and loop['model_budget'] == native['budget_after'], 'LOOP_CHANGED')
    recorded = loop['call_records'][call['call_id']]
    require(recorded['status'] == 'completed' and recorded['name'] == call['name']
            and recorded['arguments_sha256'] == canonical_hash(arguments)
            and recorded['result']['output'] == output and output.get('execution') == 'executed', 'SUBMIT_NOT_EXECUTED')
    require(call['event']['output_sha256'] == canonical_hash(output) and call['event'] in loop['events'], 'SUBMIT_EVENT_CHANGED')
    value = output['value']
    require(value['candidate_ref'] == native['candidate_ref'] and value['candidate_source_id'] == candidate['candidate_source_id'],
            'SUBMIT_CANDIDATE_CHANGED')
    receipt = read(call['execution_receipt_ref'])
    receipt_origin = {k: v for k, v in call.items() if k not in {'execution_receipt_ref', 'role', 'event', 'result_ref'}}
    require(receipt['origin'] == receipt_origin and receipt['output'] == value,
            'SUBMIT_EXECUTION_RECEIPT_CHANGED')
    require(receipt['task_head']['current_revision_ref'] == native['candidate_ref'], 'RECEIPT_CANDIDATE_CHANGED')
    events = [e for e in native['provider_turns'] if e.get('response_ref') and read(e['response_ref']) == response]
    require(len(events) == 1, 'PROVIDER_ORIGIN_MISSING')
    event = events[0]
    outcome = _ref(Path(event['request_ref']['path']).parent / ('provider-outcome-' + event['reservation_id'] + '.json'))
    require(read(outcome) == event, 'PROVIDER_OUTCOME_CHANGED')
    dispatch = read(event['dispatch_ref'])
    require(dispatch['request_ref'] == event['request_ref'] and dispatch['reservation_id'] == event['reservation_id']
            and dispatch['status'] == 'requested', 'PROVIDER_DISPATCH_CHANGED')
    read(event['request_ref'])
    return {'model_origin': origin, 'submit_result_ref': call['result_ref'], 'provider_outcome_ref': outcome,
            'submit_execution_receipt_ref': call['execution_receipt_ref'],
            'normalized_request_ref': event['request_ref']}


def verify_live_transport(run_root, native, candidate, measurement):
    # 为什么要有它：这里只传候选身份，不虚构交付；规范化请求与实际发送请求分别绑定到同一次真实响应。
    facts = {'candidate_ref': native['candidate_ref'], 'candidate_source_id': candidate['candidate_source_id']}
    live = _verify_v2_live_submission(run_root, native, facts, measurement)
    matches = [e for e in native['provider_turns'] if e.get('response_ref', {}).get('sha256') ==
               candidate['model_origin']['response_ref']['sha256']]
    require(len(matches) == 1, 'LIVE_PROVIDER_ORIGIN_AMBIGUOUS')
    return {'normalized_request_ref': live['request_ref'],
            'actual_provider_request_ref': matches[0]['provider_request_ref'],
            'response_ref': live['response_ref'], 'generation_span_id': live['generation_span_id'],
            'trace_id': live['trace_id'], 'verification': 'same_submit_bound_to_live_provider_transport'}


def inspect():
    plan = read_bound_json(PLAN_REF)
    public = read_bound_json(plan['public_preparation_ref'])
    process_path = MAIN / 'execution/process.json'
    if not process_path.is_file():
        return {'status': 'waiting_for_terminal_process'}
    process_ref = _ref(process_path)
    process = read_bound_json(process_ref)
    terminal_gate(process, read_bound_json(_ref(MAIN / 'started.json')), plan, PLAN_REF)
    native_path = Path(plan['run_root']) / 'a/task-agent-execution/execution.json'
    facts = {'process_ref': process_ref, 'original_process_status': process['status'], 'delivered': False,
             'scope': 'single_candidate_business_evaluation_not_U2_adoption_or_normal_task_completion'}
    if not native_path.is_file():
        return {**facts, 'status': 'native_missing'}
    native_ref = _ref(native_path)
    native = read_bound_json(native_ref)
    facts.update(native_ref=native_ref, native_status=native['status'])
    if native.get('unknown_effects') or native['status'] == 'unknown':
        return {**facts, 'status': 'native_unknown'}
    if not native.get('candidate_ref'):
        return {**facts, 'status': 'no_candidate'}
    root = Path(plan['run_root']) / 'a'
    require(native['run_id'] == plan['run_id'], 'NATIVE_RUN_CHANGED')
    state = TaskAgentV2State(root, native['task_contract_ref'])
    candidate = state.validate_candidate(native['candidate_ref'])
    def read(reference):
        require(Path(reference['path']).resolve().is_relative_to(root.resolve()), 'NATIVE_REFERENCE_ESCAPES_RUN')
        return read_bound_json(reference)
    facts['origin_evidence'] = verify_origin(native, candidate, read)
    measurement = measure_product_task(allowed_root=Path(plan['run_root']).parent,
        run_root=Path(plan['run_root']), run_id=plan['run_id'])
    facts['origin_evidence']['live_transport'] = verify_live_transport(Path(plan['run_root']), native, candidate, measurement)
    worker_ref = native['worker_binding']['configuration_ref']
    require(_worker_binding(worker_ref) == native['worker_binding'], 'WORKER_CHANGED')
    selected = read_bound_json(worker_ref)['behavior_evaluator']
    require(selected['factory_ref']['sha256'] == _ref(backend.__file__)['sha256'] and
            selected['factory_name'] == 'make_csharp_repository_evaluator', 'BACKEND_CHANGED')
    profile = backend.CSharpRepositoryProfile(selected['kwargs']['profile_ref'])
    require(profile.reference == public['profile_ref'], 'PUBLIC_PROFILE_CHANGED')
    profile.verify_fixed_inputs(public['source_root'], selected['kwargs']['expected_fixed_inputs'])
    facts.update(candidate_ref=native['candidate_ref'], candidate_source_id=candidate['candidate_source_id'])
    return {'status': 'ready', 'facts': facts, 'plan': plan, 'public': public, 'state': state,
            'candidate': candidate, 'fixed_inputs': selected['kwargs']['expected_fixed_inputs'],
            'config': _configuration(EVALUATOR_REF)}


def production_bytes(relative, materialized, fixed, public_root):
    if relative in materialized:
        return materialized[relative].encode('utf-8')
    require(relative in fixed, 'MISSING_PRODUCTION_SOURCE')
    raw = (public_root / relative).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == fixed[relative], 'FIXED_BINARY_CHANGED')
    try:
        raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw
    raise ValueError('MISSING_TEXT_CANDIDATE')


def execute(context, output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    config, state = context['config'], context['state']
    materialized = state.materialize(context['facts']['candidate_ref'])
    fixed = {r['path']: r['sha256'] for r in context['fixed_inputs']['fixed_files']}
    production = set(config['production_paths'])
    require(set(context['candidate']['files']) <= production, 'CANDIDATE_OUTSIDE_PRODUCTION')
    control = read_bound_json(read_bound_json(config['qualification_ref'])['controls']['correct'])
    expected = {}
    for row in control['source_inputs']:
        relative = row['path']
        target = source / relative
        require(target.resolve().is_relative_to(source.resolve()), 'PATH_ESCAPE')
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in production:
            raw = production_bytes(relative, materialized, fixed, Path(context['public']['source_root']))
            expected[relative] = hashlib.sha256(raw).hexdigest()
        else:
            raw = (Path(control['source_root']) / relative).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == row['sha256'], 'FIXED_TEST_CHANGED')
        target.write_bytes(raw)
    require(set(expected) == production, 'PRODUCTION_OMITTED')
    profile = backend.CSharpRepositoryProfile(config['profile_ref'])
    profile.verify_fixed_inputs(source, config['fixed_inputs'])
    os.environ.update(context['plan']['environment'])
    cache = profile.task_work_root(output)
    cache.mkdir(parents=True)
    origin = profile.task_work_root(Path('C:/tmp/rf-serilog-v2-pair-eval/correct'))
    for name in ('p', 'h'):
        shutil.copytree(origin / name, cache / name)
    executor = backend.CSharpRepositoryBackend(profile_ref=profile.reference, source_root=source,
        workspace_root=output, output_root=output / 'checks', expected_fixed_inputs=config['fixed_inputs'])
    refs = {}
    try:
        result = executor.run_build_test(source, 'serilog-learning-independent', 'independent_evaluation')
        refs['report_ref'] = result['report_ref']
        report = read_bound_json(result['report_ref'])
        refs.update(raw_test_result_ref=report['test_execution'].get('report_ref'),
                    stage_receipts=report.get('stage_receipts'), integrity_ref=report.get('integrity_ref'))
        if report['test_execution'].get('status') != 'executed':
            return {'status': 'incomplete', **refs}
        verdict = _report(config, result['report_ref'])
        require(verdict['source_manifest'] == expected, 'WRONG_CANDIDATE_RESULT')
        return {'status': 'evaluated', **refs, 'task_success': verdict['task_success'],
                'preservation_passed': verdict['preservation_passed']}
    except Exception as error:
        refs['retained_stage_refs'] = [_ref(path) for path in sorted((output / 'checks').glob('*/*-fact.json'))]
        refs['integrity_error_refs'] = [_ref(path) for path in sorted((output / 'checks').glob('*/integrity-error.json'))]
        return {'status': 'unknown', **refs, 'error_type': type(error).__name__, 'error': str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        context = inspect()
    except Exception as error:
        print(json.dumps({'status': 'unknown', 'task_success': None, 'preservation_passed': None,
            'model_calls': 0, 'build_calls': 0, 'error_type': type(error).__name__, 'error': str(error)}))
        return
    if context['status'] != 'ready' or not args.execute:
        print(json.dumps({'status': context['status'], 'task_success': None, 'preservation_passed': None,
            'model_calls': 0, 'build_calls': 0}))
        return
    require(args.output is not None, 'OUTPUT_REQUIRED')
    output = args.output.resolve()
    for protected in (MAIN, Path(context['public']['source_root']), Path(EVALUATOR_REF['path']).parents[1]):
        require(not output.is_relative_to(protected.resolve()) and not protected.resolve().is_relative_to(output),
                'OUTPUT_OVERLAPS_ORIGINAL_INPUT')
    clock = time.monotonic()
    try:
        result = execute(context, output)
    except Exception as error:
        # 为什么要有它：准备失败不是业务失败；不得在既有输出目录补写覆盖原件。
        print(json.dumps({'status': 'unknown', 'task_success': None, 'preservation_passed': None,
            'model_calls': 0, 'error_type': type(error).__name__, 'error': str(error)}))
        return
    record = {'task_success': None, 'preservation_passed': None, **result, **context['facts'],
        'plan_ref': PLAN_REF, 'evaluator_ref': EVALUATOR_REF, 'model_calls': 0,
        'evaluation_seconds': time.monotonic() - clock, 'cost_scope': 'independent_evaluation_only',
        'training_or_model_input_written': False, 'paired_comparison': False}
    reference = _save(output / 'result.json', record)
    print(json.dumps({'result_ref': reference, 'status': result['status'],
        'task_success': record['task_success'], 'preservation_passed': record['preservation_passed']}))


if __name__ == '__main__':
    main()
