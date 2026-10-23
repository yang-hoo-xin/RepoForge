"""Explicit private evaluation of one terminal v2 Polly candidate.

为什么要有它：复用独立测试和执行器，但不把历史两控制资格冒充 U2 三控制采用资格。
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.public_failure_feedback import validated_public_checks
from src.mod_agent.task_agent_execution import _worker_binding
from src.mod_agent.task_agent_v2_recovery import (
    _validate_completed,
    _validate_host_binding,
    _validate_resource_chain,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2State, load_v2_artifact

MAIN = Path('C:/tmp/rf-polly-v2-context-main')
QUALIFICATION = Path('C:/tmp/rfext-polly-2164-qualification/qualification.json')
QUALIFICATION_SHA = 'f25745b4d0e819e15602795c82a6b567eabc26763e26d522bf6d0211603f4980'
OPENING = Path('C:/tmp/rfext-polly-2164-task/task-opened.json')
OPENING_SHA = 'b1a523ca21ca2871f60924d1449c2cf36ab92f2ed8b8d22cbdd5cfa736889731'
QUALIFIER_SHA = 'b1f04c80026aaa25c88c253c1e3124023d1a9d580787b88b06954a3e5505ac1d'
BASE = '5d81f99ef5f523032c17b6b59b32d575ec9dfb06'
CHECKPOINT = 'v2/session-checkpoint/ebcc987d5f37c15be7f32519296010ae9cb2a5824f6c285e09af55f7d9ffcd0b.json'
CONTRACT = 'v2/contract/9f6d8f5672d8e76fd1a281ee7f76e226b7346019120e2ab30ac630d79b465d83.json'
CANDIDATE = 'v2/candidate/9a148035ed8548dfb575f5b52a0baa78d7a1b1551818e7881f962be17f4ddb39.json'
SOURCE_ID = '9bc542ef53307233f0aa266054f189cb75936f21683aa092dcdcf945cd6bc074'


def bound(reference):
    return read_bound_json({k: reference[k] for k in ('path', 'sha256')})


def prerequisites(native_path=None, checkpoint_diagnostic=False):
    plan_ref = _ref(MAIN / 'frozen-plan.json')
    plan = read_bound_json(plan_ref)
    public = bound(plan['public_input_ref'])
    qualification = bound({'path': str(QUALIFICATION), 'sha256': QUALIFICATION_SHA})
    assert qualification['qualified'] and qualification['base_commit'] == BASE
    assert qualification['public_problem_ref']['sha256'] == public['problem_ref']['sha256']
    process_path = MAIN / 'execution/process.json'
    if not process_path.exists():
        return {'status': 'waiting_for_terminal_process', 'task_success': None}
    process = read_bound_json(_ref(process_path))
    assert process.get('finished_at') and process.get('status') in {'exited', 'timed_out', 'infrastructure_failed'}, 'PROCESS_NOT_TERMINAL'
    started = read_bound_json(_ref(MAIN / 'started.json'))
    assert started['plan_ref'] == plan_ref and started['run_id'] == plan['run_id']
    assert plan['platform_ref']['path'] in process['argv'] and plan['platform_ref']['sha256'] in process['argv']
    origin_evidence = {}
    if checkpoint_diagnostic:
        assert process['status'] == 'timed_out'
        root = Path(plan['run_root']) / 'a'
        checkpoint = load_v2_artifact(root, CHECKPOINT)
        assert checkpoint['task_contract_ref'] == CONTRACT
        state = TaskAgentV2State(root, CONTRACT)
        assert state.head == checkpoint['task_head']
        assert checkpoint['loop']['phase'] == 'tool_pending'
        host = state.load(checkpoint['host_config_ref'])
        _validate_host_binding(state, checkpoint, host)
        assert host['graph_state']['run_id'] == plan['run_id']
        assert all(_ref(r['path']) == r for r in host['owner_refs'])
        _validate_resource_chain(state, checkpoint, host)
        _validate_completed(checkpoint, root, host)
        candidate = state.validate_candidate(CANDIDATE)
        assert state.current_revision_ref == CANDIDATE and candidate['candidate_source_id'] == SOURCE_ID
        origin = candidate['model_origin']
        calls = [c for c in checkpoint['session']['actual_calls'] if c['call_id'] == origin['call_id']]
        assert len(calls) == 1
        call = calls[0]
        assert call['name'] == 'candidate_submit' and all(call[k] == origin[k] for k in origin)
        receipt = read_bound_json(call['execution_receipt_ref'])
        assert receipt['output']['candidate_ref'] == CANDIDATE and receipt['output']['candidate_source_id'] == SOURCE_ID
        arguments = read_bound_json(origin['arguments_ref'])
        assert candidate['parent_revision_ref'] == arguments['parent_revision_ref']
        assert len(candidate['changes']) == len(arguments['edits'])
        assert all(all(change[k] == edit[k] for k in edit) for change, edit in zip(candidate['changes'], arguments['edits']))
        response = read_bound_json(origin['response_ref'])
        matches = [e for e in checkpoint['details']['provider_turns'] if e.get('response_ref')
                   and read_bound_json(e['response_ref']) == response]
        assert len(matches) == 1
        event = matches[0]
        dispatch = read_bound_json(event['dispatch_ref'])
        assert dispatch['request_ref'] == event['request_ref'] and dispatch['reservation_id'] == event['reservation_id']
        assert dispatch['status'] == 'requested'
        read_bound_json(event['request_ref'])
        origin_evidence = {'checkpoint_ref': _ref(root / CHECKPOINT), 'scope': 'checkpoint_candidate_diagnostic',
            'model_origin': origin, 'submit_receipt_ref': call['execution_receipt_ref'],
            'original_task_status': 'timed_out', 'original_public_check': 'unknown',
            'original_whole_task_finished': False, 'original_delivered': False}
        native = {**checkpoint['details'], 'task_contract_ref': CONTRACT, 'candidate_ref': CANDIDATE,
                  'status': 'checkpoint_candidate_diagnostic'}
        native_ref, roots = None, [root]
    else:
        path = Path(native_path).resolve()
        assert path.is_relative_to(Path(plan['run_root']).resolve()), 'NATIVE_OUTSIDE_RUN'
        if not path.is_file():
            return {'status': 'native_missing', 'task_success': None}
        native_ref = _ref(path)
        native = read_bound_json(native_ref)
        roots = [p for p in path.parents if (p / native['task_contract_ref']).is_file()]
        origin_evidence = {'scope': 'candidate_business_evaluation_only', 'autonomy_proven': False}
    if native.get('status') == 'unknown' or native.get('unknown_effects'):
        return {'status': 'native_unknown', 'task_success': None, 'native_ref': native_ref}
    if not native.get('candidate_ref'):
        return {'status': 'no_candidate', 'task_success': None, 'native_ref': native_ref}
    assert len(roots) == 1
    state = TaskAgentV2State(roots[0], native['task_contract_ref'])
    candidate = state.validate_candidate(native['candidate_ref'])
    worker_ref = native['worker_binding']['configuration_ref']
    assert _worker_binding(worker_ref) == native['worker_binding']
    worker = read_bound_json(worker_ref)['behavior_evaluator']
    assert worker['factory_ref']['sha256'] == _ref(backend.__file__)['sha256']
    assert worker['kwargs']['profile_ref'] == public['profile_ref']
    profile = backend.CSharpRepositoryProfile(public['profile_ref'])
    profile.verify_fixed_inputs(public['source_root'], worker['kwargs']['expected_fixed_inputs'])
    return {'status': 'ready', 'plan': plan, 'plan_ref': plan_ref, 'public': public,
            'process_ref': _ref(process_path), 'native_ref': native_ref, 'native_status': native['status'],
            'state': state, 'candidate': candidate, 'candidate_ref': native['candidate_ref'],
            'profile': profile, 'worker': worker, 'origin_evidence': origin_evidence}


def assess_labels(qualifier, observed, task):
    targets = qualifier.label_outcomes(observed, task['FAIL_TO_PASS'])
    preserved = qualifier.label_outcomes(observed, task['PASS_TO_PASS'])
    extras = qualifier.label_outcomes(observed, sorted(set(observed) - set(task['FAIL_TO_PASS']) - set(task['PASS_TO_PASS'])))
    valid = all(r['missing_count'] == 0 and r['outcome_counts']['Other'] == 0 for r in (targets, preserved))
    return {'status': 'evaluated' if valid else 'unknown',
        'task_success': targets['outcome_counts']['Failed'] == 0 if valid else None,
        'preservation_passed': preserved['outcome_counts']['Failed'] == 0 if valid and task['PASS_TO_PASS'] else None,
        'preservation_scope': 'registered_PASS_TO_PASS' if task['PASS_TO_PASS'] else 'no_registered_preservation_labels',
        'target_label_summary': targets, 'preservation_label_summary': preserved,
        'additional_observed_tests': extras, 'test_count': sum(len(v) for v in observed.values())}


def execute(context, output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'
    source.mkdir()
    state, profile, public = context['state'], context['profile'], context['public']
    materialized = state.materialize(context['candidate_ref'])
    fixed = {r['path']: r['sha256'] for r in context['worker']['kwargs']['expected_fixed_inputs']['fixed_files']}
    public_root = Path(public['source_root'])
    # 为什么要有它：生产内容全部取当前候选；固定测试只取已绑定公开输入，之后施加登记的独立测试补丁。
    for row in backend._snapshot(public_root):
        relative = row['path']
        target = source / relative
        assert target.resolve().is_relative_to(source.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith('test/'):
            assert fixed.get(relative) == row['sha256'], 'PUBLIC_TEST_CHANGED'
            raw = (public_root / relative).read_bytes()
        elif relative in materialized:
            raw = materialized[relative].encode('utf-8')
        else:
            raw = (public_root / relative).read_bytes()
            assert fixed.get(relative) == backend.hashlib.sha256(raw).hexdigest()
            try:
                raw.decode('utf-8')
            except UnicodeDecodeError:
                pass
            else:
                raise ValueError('MISSING_TEXT_CANDIDATE')
        target.write_bytes(raw)
    opening = bound({'path': str(OPENING), 'sha256': OPENING_SHA})
    patch_ref = {k: opening['test_patch_ref'][k] for k in ('path', 'sha256')}
    assert _ref(patch_ref['path']) == patch_ref
    qualifier_path = Path(__file__).parents[1] / 'eval_driven_20261021/qualify_polly_2164.py'
    assert _ref(qualifier_path)['sha256'] == QUALIFIER_SHA
    spec = importlib.util.spec_from_file_location('polly_original_qualifier', qualifier_path)
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    assert all(p.startswith('test/') for p in qualifier.patch_paths(Path(patch_ref['path'])))
    for label, flags in [('patch-check', ['--check']), ('patch-apply', [])]:
        command = ['git', 'apply', *flags, patch_ref['path']]
        result = subprocess.run(command, cwd=source, capture_output=True)
        _save(output / (label + '.json'), {'argv': command, 'exit_code': result.returncode,
            'stdout': result.stdout.decode('utf-8', errors='replace'), 'stderr': result.stderr.decode('utf-8', errors='replace')})
        if result.returncode:
            return {'status': 'test_patch_unavailable', 'task_success': None, 'preservation_passed': None}
    for relative, text in materialized.items():
        if not relative.startswith('test/'):
            assert (source / relative).read_bytes() == text.encode('utf-8'), 'PRODUCTION_CHANGED'
    os.environ.update(context['plan']['environment'])
    cache = profile.task_work_root(output)
    cache.mkdir(parents=True)
    for name, origin in [('p', 'C:/tmp/rfext-polly-packages'), ('h', 'C:/tmp/rfext-polly-http-cache')]:
        shutil.copytree(origin, cache / name)
    executor = backend.CSharpRepositoryBackend(profile_ref=profile.reference, source_root=source,
        workspace_root=output, output_root=output / 'checks', expected_fixed_inputs=profile.bind(source))
    execution_refs = {}
    try:
        result = executor.run_build_test(source, 'polly-context-independent', 'independent_evaluation')
        execution_refs['report_ref'] = result['report_ref']
        report = read_bound_json(result['report_ref'])
        execution_refs.update(raw_test_result_ref=report['test_execution'].get('report_ref'),
            stage_receipts=report.get('stage_receipts', {}), integrity_ref=report.get('integrity_ref'))
        if report['test_execution'].get('status') != 'executed':
            return {'status': 'incomplete', 'task_success': None, 'preservation_passed': None, 'report_ref': result['report_ref']}
        validated_public_checks(result['report_ref'])
        observed = qualifier.trx_results(Path(report['test_execution']['report_ref']['path']).parent)
        task = bound(opening['task_ref'])
        return {**assess_labels(qualifier, observed, task), **execution_refs, 'test_patch_ref': patch_ref}
    except Exception as error:
        return {'status': 'unknown', 'task_success': None, 'preservation_passed': None,
            'error_type': type(error).__name__, 'error': str(error), **execution_refs}


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--native')
    modes.add_argument('--checkpoint-diagnostic', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        context = prerequisites(args.native, args.checkpoint_diagnostic)
    except Exception as error:
        print(json.dumps({'status': 'unknown', 'task_success': None, 'build_calls': 0,
            'model_calls': 0, 'error_type': type(error).__name__, 'error': str(error)}))
        return
    if context['status'] != 'ready' or not args.execute:
        print(json.dumps({'status': context['status'], 'task_success': None, 'build_calls': 0, 'model_calls': 0}))
        return
    assert args.output is not None
    evaluation_start = time.monotonic()
    result = execute(context, args.output.resolve())
    reference = _save(args.output / 'result.json', {**result, 'native_ref': context['native_ref'],
        'candidate_ref': context['candidate_ref'], 'candidate_source_id': context['candidate']['candidate_source_id'],
        'native_status': context['native_status'], 'plan_ref': context['plan_ref'], 'process_ref': context['process_ref'],
        'qualification_ref': {'path': str(QUALIFICATION), 'sha256': QUALIFICATION_SHA}, 'model_calls': 0,
        'scope': context['origin_evidence']['scope'], 'origin_evidence': context['origin_evidence'],
        'limitations': 'one exposed task, net8.0; historical baseline/reference only, no equivalent control; not U2 adoption qualification',
        'independent_evaluation_wall_seconds': time.monotonic() - evaluation_start,
        'cost_scope': 'evaluation costs separate from solver run; no solver or learning input'})
    print(json.dumps({'result_ref': reference, 'status': result['status'],
        'task_success': result['task_success'], 'preservation_passed': result['preservation_passed']}))


if __name__ == '__main__':
    main()
