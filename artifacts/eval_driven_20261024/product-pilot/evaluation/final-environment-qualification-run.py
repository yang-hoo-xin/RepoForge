"""One fixed six-item common-environment qualification; no model or retry loop.

为什么要有它：相同冻结执行与诊断边界必须真实覆盖固定控制，旧环境通过不能代替本批资格。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.strategy_repository_evaluator import (
    _groups,
    qualify_repository_strategy_evaluator,
)
from src.mod_agent.model_qualification import read_bound_json
from src.mod_agent.platform_execution_snapshot import FrozenExecution

ROOT = Path('C:/tmp/rfqual-final-1531')
OLD = Path('C:/tmp/rfext-spectre-1531-evaluation/public-contract-v1')
PLATFORM = {'path': 'C:/tmp/rfdiag-real-1531/platform/execution_snapshot.json',
            'sha256': 'f0f39c46dc3ade97b37d50fe5db638ccbc98dc2a05b5a8df98bfb24f1e90cd40'}
REPO = Path(__file__).resolve().parents[4]
TEST_ROOT = 'test/Spectre.Console.Cli.Tests/'
INITIALIZER = TEST_ROOT + 'VerifyConfiguration.cs'
ROLES = ('public-baseline', 'public-correct', 'public-equivalent',
         'independent-correct', 'independent-equivalent', 'independent-wrong')


def require(value, reason):
    if not value:
        raise ValueError(reason)


def prepare():
    require({path.name for path in ROOT.iterdir()} == {'process'}, 'FRESH_OUTPUT_REQUIRED')
    previous = read_bound_json(_ref(OLD / 'configuration.json'))
    real = read_bound_json(_ref('C:/tmp/rfdiag-real-1531/configuration.json'))
    adapter = real['initializer_ref']
    require(_ref(adapter['path']) == adapter, 'FIXED_INITIALIZER_CHANGED')
    profile_ref = _ref('C:/tmp/rfdiag-real-1531/profile.json')
    require(profile_ref['sha256'] == '0b84cccbaac2ad1cbbd92d1b44f08a5a775a87f9164cd96a1645511c34a86b6f', 'PROFILE_CHANGED')
    shutil.copyfile(profile_ref['path'], ROOT / 'profile.json')
    profile = backend.CSharpRepositoryProfile(_ref(ROOT / 'profile.json'))
    public_original = read_bound_json(_ref('C:/tmp/rfext-spectre-1531-evaluation/public-preparation-v2.json'))
    original_tests = {row['path']: row['sha256'] for row in public_original['source_inputs']
                      if row['path'].startswith(TEST_ROOT)}
    sources = {}
    for role in ROLES:
        if role == 'public-baseline':
            origin = Path(previous['public_source_root'])
        elif role.startswith('public-'):
            origin = OLD / role / 'source'
        else:
            origin = OLD / 'controls-v2' / role.removeprefix('independent-') / 'source'
        inputs = backend._snapshot(origin)
        actual = {row['path']: row['sha256'] for row in inputs}
        require(all(actual.get(path) == digest for path, digest in original_tests.items()), 'ORIGINAL_PUBLIC_TEST_OR_EXPECTATION_CHANGED')
        if role.startswith('public-'):
            require({path for path in actual if path.startswith(TEST_ROOT)} == set(original_tests), 'PRIVATE_TEST_IN_PUBLIC_SOURCE')
        source = ROOT / role / 'source'
        source.mkdir(parents=True)
        for row in inputs:
            selected = backend._within(source, row['path'])
            selected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin / row['path'], selected)
            require(_ref(selected)['sha256'] == row['sha256'], 'SOURCE_COPY_CHANGED')
        shutil.copyfile(adapter['path'], source / INITIALIZER)
        expected = {**actual, INITIALIZER: adapter['sha256']}
        require({row['path']: row['sha256'] for row in backend._snapshot(source)} == expected, 'ONLY_FIXED_INITIALIZER_MAY_CHANGE')
        sources[role] = {'source_root': str(source), 'origin_root': str(origin),
            'origin_manifest_ref': _save(ROOT / 'inputs' / (role + '-origin.json'), {'files': inputs}),
            'source_manifest_ref': _save(ROOT / 'inputs' / (role + '-source.json'), {'files': backend._snapshot(source)})}
    cache = profile.task_work_root(ROOT)
    cache.mkdir(parents=True)
    origin = profile.task_work_root(Path('C:/tmp/rfdiag-real-1531'))
    clock = time.monotonic()
    for name in ('p', 'h'):
        if (origin / name).is_dir():
            shutil.copytree(origin / name, cache / name)
    return {'profile_ref': profile.reference, 'sources': sources, 'initializer_ref': adapter,
        'cache_copy_seconds': time.monotonic() - clock, 'cache_origin': str(origin), 'cache_root': str(cache),
        'private_labels_ref': previous['task_ref'], 'public_fixture_ref': previous['public_test_ref'],
        'original_public_tests_and_expected_bytes_unchanged': True,
        'public_source_root': sources['public-baseline']['source_root'],
        'public_source_snapshot_ref': sources['public-baseline']['source_manifest_ref']}


def execute_case(prepared, role):
    profile = backend.CSharpRepositoryProfile(prepared['profile_ref'])
    source = Path(prepared['sources'][role]['source_root'])
    clock = time.monotonic()
    runner = backend.CSharpRepositoryBackend(profile_ref=profile.reference, source_root=source,
        workspace_root=ROOT, output_root=ROOT / role / 'checks', expected_fixed_inputs=profile.bind(source))
    report = runner.run_build_test(source, role, 'fixed_common_environment_qualification')
    diagnostic = backend.validate_diagnostic_integrity(read_bound_json(report['report_ref']))
    tests = backend._read_tests(Path(report['test_execution']['report_ref']['path']), 'trx')
    public = [row for row in tests if any('RepoForgePublicVersionConflictTests' in item.get('class_name', '')
              for item in row['test_definitions'])]
    private = [row for row in tests if any('IndependentVersionConflictTests' in item.get('class_name', '')
               for item in row['test_definitions'])]
    output = {'role': role, 'status': report['status'], 'report_ref': report['report_ref'],
        'trx_ref': report['test_execution']['report_ref'], 'integrity_ref': report['integrity_ref'],
        'diagnostics_ref': report['diagnostics_ref'], 'stage_receipts': report['stage_receipts'],
        'tests': len(tests), 'failed': sum(not row['passed'] for row in tests),
        'public_cases': len(public), 'public_passed': sum(row['passed'] for row in public),
        'independent_cases': len(private), 'independent_passed': sum(row['passed'] for row in private),
        'original_cases': len(tests) - len(public) - len(private),
        'original_failed': sum(not row['passed'] for row in tests if row not in public and row not in private),
        'expected_copies': len(diagnostic['expected_copies']), 'received_files': len(diagnostic['received_refs']),
        'all_source_test_expected_bytes_unchanged': diagnostic['all_source_test_and_expectation_bytes_unchanged'],
        'elapsed_seconds': time.monotonic() - clock, 'model_calls': 0}
    reference = _save(ROOT / 'receipts' / (role + '.json'), output)
    print(json.dumps({'role': role, 'receipt_ref': reference, **{key: output[key] for key in (
        'status', 'tests', 'failed', 'public_passed', 'independent_cases', 'independent_passed')}}), flush=True)
    require(output['original_cases'] == 377 and output['public_cases'] == 8 and output['expected_copies'] == 89,
            'FIXED_TEST_OR_EXPECTATION_SCOPE_CHANGED')
    if role == 'public-baseline':
        require(output['tests'] == 385 and output['status'] == 'behavior_failed' and output['failed'] == 2
                and output['public_passed'] == 6 and output['original_failed'] == 0, 'PUBLIC_BASELINE_CONTRADICTION')
    elif role != 'independent-wrong':
        require(output['status'] == 'passed' and output['failed'] == 0, 'POSITIVE_CONTROL_CONTRADICTION')
    else:
        require(output['status'] == 'behavior_failed' and output['failed'] > 0, 'WRONG_CONTROL_NOT_REJECTED')
    return output


def main():
    clock = time.monotonic()
    guard = FrozenExecution(PLATFORM)
    guard.verify_files()
    guard.verify_loaded_modules()
    receipts, prepared, evaluator = {}, None, None
    result = {'qualified': False, 'evaluator_ref': None, 'model_calls': 0,
        'fixed_selection': list(ROLES), 'snapshot_ref': PLATFORM, 'controller_ref': _ref(__file__),
        'intent_ref': _ref(REPO / 'artifacts/eval_driven_20261024/product-pilot/final-environment-qualification-intent.json'),
        'deadline_utc': float(os.environ['REPOFORGE_TASK_DEADLINE_UTC']), 'automatic_retry': False,
        'official_benchmark_score': False, 'paired_product_score': False}
    try:
        prepared = prepare()
        result['preparation_ref'] = _save(ROOT / 'prepared.json', prepared)
        for role in ROLES:
            require(time.time() < result['deadline_utc'], 'CUMULATIVE_WINDOW_EXHAUSTED')
            receipts[role] = execute_case(prepared, role)
        labels = read_bound_json(prepared['private_labels_ref'])
        correct = read_bound_json(receipts['independent-correct']['report_ref'])
        groups = _groups(backend._read_tests(Path(correct['test_execution']['report_ref']['path']), 'trx'), 'trx')
        target = labels['FAIL_TO_PASS']
        production = [row['path'] for row in correct['source_inputs'] if not row['path'].startswith(TEST_ROOT)]
        evaluator = qualify_repository_strategy_evaluator(profile_ref=prepared['profile_ref'],
            controls={name: receipts['independent-' + name]['report_ref'] for name in ('correct', 'equivalent', 'wrong')},
            production_paths=production, target_test_ids=target, preservation_test_ids=sorted(set(groups) - set(target)),
            output_root=ROOT / 'qualification')
        require(guard.check()['status'] == 'verified', 'FROZEN_PLATFORM_CHANGED')
        result.update(qualified=True, evaluator_ref=evaluator, status='qualified_public_issue_environment')
    except Exception as error:
        result.update(status='preparation_blocked', error_type=type(error).__name__, error=str(error))
    result.update(receipts={role: _ref(ROOT / 'receipts' / (role + '.json')) for role in receipts},
        frozen_execution_check_ref=_save(ROOT / 'frozen-execution-check.json', guard.check()),
        elapsed_seconds=time.monotonic() - clock)
    if prepared is not None:
        result.update(profile_ref=prepared['profile_ref'], public_source_root=prepared['public_source_root'],
            public_source_snapshot_ref=prepared['public_source_snapshot_ref'],
            cache_copy_seconds=prepared['cache_copy_seconds'])
    reference = _save(ROOT / 'result.json', result)
    print(json.dumps({'result_ref': reference, 'qualified': result['qualified'], 'evaluator_ref': result['evaluator_ref']}), flush=True)


if __name__ == '__main__':
    main()
