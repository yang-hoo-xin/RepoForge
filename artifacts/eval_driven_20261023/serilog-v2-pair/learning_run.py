"""One normal frozen B task with its existing terminal-learning hook enabled.

为什么要有它：真实经历提炼需经正常入口和原累计资源执行，不由外层脚本指定技术动作或代修。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.dsh_run_request import (  # noqa: E402
    _configuration_arguments,
    build_product_argv,
    evidence_locator,
    prepare_new_task,
)
from scripts.run_benchmark_arm import execute  # noqa: E402
from src.mod_agent.domain.csharp_repository_backend import _snapshot  # noqa: E402
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.evaluation.strategy_trial_preparation import _source_snapshot  # noqa: E402
from src.mod_agent.model_admission import read_ollama_model_identity  # noqa: E402
from src.mod_agent.model_qualification import read_bound_json  # noqa: E402
from src.mod_agent.platform_execution_snapshot import (  # noqa: E402
    FrozenExecution,
    prepare_snapshot,
)
from src.mod_agent.public_failure_feedback import validated_public_checks  # noqa: E402

ROOT = Path('C:/tmp/rf-serilog-v2-learning-main')
OLD = Path('C:/tmp/rf-serilog-v2-pair-main')
SNAPSHOT = Path('C:/tmp/rfcode-serilog-v2-learning-B')
INTENT = Path(__file__).with_name('learning-intent.json')


def public_inputs():
    public_ref = _ref(OLD / 'public-preparation.json')
    public = read_bound_json(public_ref)
    baseline_ref = _ref(OLD / 'baseline-result.json')
    baseline = read_bound_json(baseline_ref)
    for key in ('request_ref', 'profile_ref', 'settings_ref', 'public_test_ref'):
        if _ref(public[key]['path']) != public[key]:
            raise ValueError('PUBLIC_INPUT_CHANGED')
    if _snapshot(Path(public['source_root'])) != public['source_inputs']:
        raise ValueError('PUBLIC_SOURCE_CHANGED')
    if baseline['status'] != 'behavior_failed' or baseline['local_feed_only'] is not True:
        raise ValueError('PUBLIC_BASELINE_NOT_QUALIFIED')
    validated_public_checks(baseline['report_ref'])
    return public_ref, public, baseline_ref, baseline


def freeze():
    if ROOT.exists() or SNAPSHOT.exists():
        raise ValueError('ALREADY_PREPARED_INSPECT_EXISTING_ATTEMPT')
    intent_ref = _ref(INTENT)
    intent = read_bound_json(intent_ref)
    public_ref, public, baseline_ref, baseline = public_inputs()
    model = read_ollama_model_identity('http://127.0.0.1:11434', intent['model'])
    if model['digest'] != intent['model_digest']:
        raise ValueError('MODEL_CHANGED')
    ROOT.mkdir(parents=True)
    experience_ref = _save(ROOT / 'experience.json', {'schema_version': 'task_agent_experience_config_v1',
        'store_root': str(ROOT / 'strategy-memory'), 'snapshot_ref': None, 'extract_on_terminal': True})
    args = read_bound_json(_ref(OLD / 'base-execution.json'))['arguments']
    args[args.index('--workspace-root') + 1] = str(ROOT / 'runs')
    if '--task-agent-experience-config' in args:
        raise ValueError('UNEXPECTED_EXISTING_STRATEGY_INPUT')
    args.extend(['--task-agent-experience-config', experience_ref['path']])
    config = _save(ROOT / 'base-execution.json', {'arguments': args})
    description = read_bound_json(public['request_ref'])['description']
    selected = prepare_new_task(execution_config=config['path'], description=description)
    argv = build_product_argv(execution_config=selected, description=description)
    _, options = _configuration_arguments(selected)
    run_id = options['--run-id']
    run_root = ROOT / 'runs' / run_id
    refs = [_ref(selected), public['profile_ref'], public['settings_ref'], public['request_ref'], experience_ref]
    platform = prepare_snapshot(source_root=REPO, snapshot_root=SNAPSHOT, business_argv=argv,
        run_root=run_root, bound_inputs=[(ref['path'], ref['sha256']) for ref in refs])
    plan = _save(ROOT / 'frozen-plan.json', {'schema': 'real-b-terminal-learning-task.v1',
        'intent_ref': intent_ref, 'run_id': run_id, 'run_root': str(run_root),
        'execution_config_ref': _ref(selected), 'platform_ref': platform, 'controller_ref': _ref(__file__),
        'public_preparation_ref': public_ref, 'public_baseline_ref': baseline_ref,
        'source_ref': _save(ROOT / 'source-snapshot.json', _source_snapshot(public['source_root'], include_binary=True)),
        'experience_ref': experience_ref, 'model_identity': model, 'environment': baseline['environment'],
        'wall_seconds': intent['new_independent_local_wall_seconds'],
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'planned_attempts': 1, 'scripted_actions': False, 'model_calls_during_freeze': 0,
        'inspection': evidence_locator(execution_config=selected, run_id=run_id)})
    print(json.dumps({'plan_ref': plan, 'run_id': run_id}))


def run():
    plan_ref = _ref(ROOT / 'frozen-plan.json')
    plan = read_bound_json(plan_ref)
    if _ref(__file__) != plan['controller_ref'] or (ROOT / 'started.json').exists() or Path(plan['run_root']).exists():
        raise ValueError('INPUT_CHANGED_OR_ALREADY_STARTED')
    intent = read_bound_json(plan['intent_ref'])
    public_ref, public, baseline_ref, baseline = public_inputs()
    if public_ref != plan['public_preparation_ref'] or baseline_ref != plan['public_baseline_ref']:
        raise ValueError('PUBLIC_BINDING_CHANGED')
    if _source_snapshot(public['source_root'], include_binary=True) != read_bound_json(plan['source_ref']):
        raise ValueError('PUBLIC_SOURCE_CHANGED')
    FrozenExecution(plan['platform_ref']).verify_files()
    model = read_ollama_model_identity('http://127.0.0.1:11434', intent['model'])
    if model['digest'] != intent['model_digest']:
        raise ValueError('MODEL_CHANGED')
    body = json.dumps({'model': intent['model'], 'keep_alive': '30m', 'stream': False,
        'options': {'num_ctx': intent['context_window']}}).encode()
    clock = time.monotonic()
    with urlopen(Request('http://127.0.0.1:11434/api/generate', data=body,
            headers={'Content-Type': 'application/json'}), timeout=180) as response:
        warmup = _save(ROOT / 'warmup.json', {'response': json.load(response),
            'seconds': time.monotonic() - clock, 'scope': 'model load without task prompt'})
    _save(ROOT / 'started.json', {'plan_ref': plan_ref, 'run_id': plan['run_id'],
        'execution_config_ref': plan['execution_config_ref'], 'warmup_ref': warmup,
        'outer_supplied_solution': False, 'scripted_actions': False, 'human_active_seconds': None,
        'human_timing_basis': 'unknown', 'assistance': intent['model_help']})
    manifest = plan['platform_ref']
    process = execute([sys.executable, '-I', '-B', str(SNAPSHOT / 'scripts/run_frozen_product.py'),
        '--_child', manifest['path'], manifest['sha256']], cwd=SNAPSHOT, stdin_path=None,
        output=ROOT / 'execution', deadline=time.monotonic() + plan['wall_seconds'],
        environment={**os.environ, **baseline['environment'], 'PYTHONUTF8': '1', 'DOTNET_CLI_UI_LANGUAGE': 'en-US'})
    print(json.dumps({'run_id': plan['run_id'], 'status': process['status'],
        'process_ref': _ref(ROOT / 'execution/process.json'), 'inspection': plan['inspection']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'run'])
    arguments = parser.parse_args()
    freeze() if arguments.mode == 'freeze' else run()
