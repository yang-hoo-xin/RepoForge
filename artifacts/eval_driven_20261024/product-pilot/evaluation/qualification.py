"""Configuration layer over existing repository execution and qualification APIs.

为什么要有它：新题只声明私有控制输入、公开复现和环境范围，不复制平台执行器或修改旧冻结原件。
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from src.mod_agent.domain import csharp_repository_backend as backend
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.evaluation.strategy_repository_evaluator import (
    _groups,
    qualify_repository_strategy_evaluator,
)
from src.mod_agent.model_qualification import read_bound_json


def require(value, code):
    if not value:
        raise ValueError(code)


def load(reference):
    value = read_bound_json(reference)
    require(value['schema_version'] == 'repository_control_configuration_v1', 'CONFIGURATION_VERSION')
    require(value['edit_prefixes'] == ['src/Spectre.Console.Cli/', 'src/Spectre.Console/'], 'EDIT_SCOPE')
    require(len(value['expected_artifacts']) == 2 and len(set(value['expected_artifacts'])) == 2, 'ARTIFACT_SCOPE')
    require(value['source_ready'] is False or value['source_ready'] is True, 'SOURCE_READY_TYPE')
    source, output = Path(value['source_root']).resolve(), Path(value['output_root']).resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'OUTPUT_OVERLAPS_SOURCE')
    read_bound_json(value['task_ref'])
    require(_ref(value['public_test_ref']['path']) == value['public_test_ref'], 'PUBLIC_TEST_CHANGED')
    require(_ref(value['dotnet_ref']['path']) == value['dotnet_ref'], 'SDK_CHANGED')
    require(value['profile_timeout_seconds'] <= 300 and value['profile_timeout_seconds'] > 0, 'COMMAND_TIMEOUT')
    return value


def apply_private_control(source, control):
    for patch in control.get('patches', []):
        require(_ref(patch['ref']['path']) == patch['ref'], 'PRIVATE_PATCH_CHANGED')
        command = ['git', 'apply', *('--exclude=' + name for name in patch.get('excludes', [])), patch['ref']['path']]
        subprocess.run(command, cwd=source, capture_output=True, check=True, timeout=30)
    for replacement in control.get('replacements', []):
        path = backend._within(source, replacement['path'])
        text = path.read_text(encoding='utf-8')
        require(text.count(replacement['before']) == replacement['count'], 'PRIVATE_VARIANT_PREIMAGE_CHANGED')
        path.write_text(text.replace(replacement['before'], replacement['after']), encoding='utf-8')
    for item in control.get('files', []):
        require(_ref(item['ref']['path']) == item['ref'], 'CONTROL_INPUT_CHANGED')
        target = backend._within(source, item['relative_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), 'CONTROL_FILE_ALREADY_EXISTS')
        shutil.copyfile(item['ref']['path'], target)


def prepare(value, reference):
    require(value['source_ready'] is True, 'WAIT_FOR_EXPLICIT_BLOB_READY')
    root, base = Path(value['output_root']), Path(value['source_root'])
    require(not (root / 'prepared.json').exists(), 'ALREADY_PREPARED')
    task = read_bound_json(value['task_ref'])
    require(task['instance_id'] == value['task_id'] and task['base_commit'] == value['base_commit'], 'TASK_CHANGED')
    controls = read_bound_json(value['private_controls_ref'])
    for role in ('baseline', 'correct', 'equivalent', 'wrong', 'public-baseline', 'public-correct', 'public-equivalent'):
        source = Path(value['public_source_root']) if role == 'public-baseline' else root / role / 'source'
        if role == 'public-baseline' and value.get('reuse_public_source') is True:
            expected = {row['path']: row['sha256'] for row in backend._snapshot(base)}
            expected[value['public_test_destination']] = value['public_test_ref']['sha256']
            require({row['path']: row['sha256'] for row in backend._snapshot(source)} == expected, 'REUSED_PUBLIC_SOURCE_CHANGED')
            continue
        shutil.copytree(base, source, ignore=shutil.ignore_patterns(*backend._IGNORED))
        apply_private_control(source, controls[role])
        shutil.copyfile(value['public_test_ref']['path'], backend._within(source, value['public_test_destination']))
    baseline = backend._snapshot(base)
    allowed = [row['path'] for row in baseline if row['path'].endswith('.cs')
               and any(row['path'].startswith(prefix) for prefix in value['edit_prefixes'])]
    profile_data = {'profile_version': 'csharp_repository_profile_v1',
        'dotnet_ref': value['dotnet_ref'], 'project_path': 'src/Spectre.Console.Cli/Spectre.Console.Cli.csproj',
        'test_project_path': 'test/Spectre.Console.Cli.Tests/Spectre.Console.Cli.Tests.csproj',
        'nuget_config_path': 'NuGet.Config', 'configuration': 'Release', 'allowed_edit_paths': allowed,
        'expected_artifacts': value['expected_artifacts'],
        'build_properties': {'MinVerSkip': 'true', 'UseBuildTimeTools': 'false'},
        'test_command': ['test', '{test_project}', '--configuration', '{configuration}', '--framework', 'net8.0',
                         '--logger', 'trx;LogFileName={report_path}'],
        'test_report': {'format': 'trx', 'path': 'tests.trx'}, 'timeout_seconds': value['profile_timeout_seconds']}
    if value.get('reuse_profile_ref') is not None:
        profile = value['reuse_profile_ref']
        require(read_bound_json(profile) == profile_data, 'REUSED_PROFILE_CHANGED')
    else:
        profile = _save(root / 'profile.json', profile_data)
    if value.get('relocate_snapshots', True):
        adapter = importlib.import_module(value['snapshot_adapter_module'])
        require(_ref(adapter.__file__) == value['snapshot_adapter_ref'], 'SNAPSHOT_ADAPTER_CHANGED')
        adapter.ROOT, adapter.BASE = root, base
        adapter.adapt_snapshot_output()
    else:
        for role in ('correct', 'equivalent', 'wrong'):
            shutil.copytree(root / role / 'source', root / 'controls-v2' / role / 'source',
                            ignore=shutil.ignore_patterns(*backend._IGNORED))
    target = root / 'controls-v2/baseline/source'
    shutil.copytree(root / 'baseline/source', target, ignore=shutil.ignore_patterns(*backend._IGNORED))
    if value.get('relocate_snapshots', True):
        initializer = 'test/Spectre.Console.Cli.Tests/VerifyConfiguration.cs'
        shutil.copyfile(root / 'controls-v2/correct/source' / initializer, target / initializer)
    return _save(root / 'prepared.json', {'configuration_ref': reference, 'profile_ref': profile,
        'baseline_manifest': baseline, 'task_ref': value['task_ref'], 'private_controls_ref': value['private_controls_ref'],
        'public_test_ref': value['public_test_ref'], 'public_test_class': value['public_test_class'],
        'public_source_root': value['public_source_root'],
        'model_calls': 0, 'scope': 'original multitarget build; net8.0 execution only; no operating-system sandbox'})


def run(value, role, attempt_id=None):
    root = Path(value['output_root'])
    prepared = read_bound_json(_ref(root / 'prepared.json'))
    profile = backend.CSharpRepositoryProfile(prepared['profile_ref'])
    selected = root if role.startswith('public-') else root / 'controls-v2'
    source = Path(value['public_source_root']) if role == 'public-baseline' else selected / role / 'source'
    workspace = source.parent if role == 'public-baseline' else root
    started = time.monotonic()
    result = {'role': role, 'model_calls': 0, 'source_root': str(source), 'profile_ref': profile.reference}
    try:
        runner = backend.CSharpRepositoryBackend(profile_ref=profile.reference, source_root=source,
            workspace_root=workspace, output_root=selected / role / 'checks', expected_fixed_inputs=profile.bind(source))
        report = runner.run_build_test(source, role, 'independent_evaluator_qualification')
        result.update(status=report['status'], report_ref=report['report_ref'])
    except Exception as error:
        result.update(status='incomplete', error_type=type(error).__name__, error=str(error))
    reports = list((selected / role / 'checks').rglob('tests.trx'))
    if len(reports) == 1:
        rows = backend._read_tests(reports[0], 'trx')
        groups = _groups(rows, 'trx')
        labels = read_bound_json(value['task_ref']) if not role.startswith('public-') else {'FAIL_TO_PASS': [], 'PASS_TO_PASS': []}
        target = set(labels['FAIL_TO_PASS'])
        public = [row for row in rows if any(value['public_test_class'] in item.get('class_name', '')
                  for item in row['test_definitions'])]
        independent = [row for row in rows if value.get('independent_test_class') and any(
            value['independent_test_class'] in item.get('class_name', '') for item in row['test_definitions'])]
        result.update(trx_ref=_ref(reports[0]), actual_tests=len(rows), failures=sum(not row['passed'] for row in rows),
            official_target_labels=len(target), official_labels_found=sum(label in groups for label in target),
            official_target_instances=sum(len(groups.get(label, [])) for label in target),
            official_target_passed=sum(sum(groups.get(label, [])) for label in target),
            public_tests=len(public), public_passed=sum(row['passed'] for row in public),
            official_preservation_labels=len(labels['PASS_TO_PASS']),
            independent_tests=len(independent), independent_passed=sum(row['passed'] for row in independent),
            original_other_tests=len(rows) - len(public) - len(independent),
            original_other_failures=sum(not row['passed'] for row in rows if row not in public and row not in independent))
        result['label_origin'] = value.get('label_origin', 'upstream_task')
        if value.get('label_origin') == 'independent_public_contract':
            result.update(independent_target_labels=result['official_target_labels'],
                independent_target_instances=result['official_target_instances'],
                independent_target_passed=result['official_target_passed'])
            for key in ('official_target_labels', 'official_labels_found', 'official_target_instances',
                        'official_target_passed', 'official_preservation_labels'):
                result[key] = None
        if role.startswith('public-'):
            result.update(official_label_scope='not_applicable_public_only', official_target_labels=None,
                official_labels_found=None, official_target_instances=None, official_target_passed=None,
                official_preservation_labels=None)
    result['retained_stage_refs'] = [_ref(path) for path in sorted((selected / role / 'checks').glob('*/*-fact.json'))]
    result['integrity_error_refs'] = [_ref(path) for path in sorted((selected / role / 'checks').glob('*/integrity-error.json'))]
    result['elapsed_seconds'] = time.monotonic() - started
    suffix = '-' + attempt_id if attempt_id else ''
    return _save(selected / (role + '-result' + suffix + '.json'), result)


def qualify(value):
    root = Path(value['output_root'])
    prepared = read_bound_json(_ref(root / 'prepared.json'))
    controls = {role: read_bound_json(_ref(root / 'controls-v2' / (role + '-result.json')))['report_ref']
                for role in ('correct', 'equivalent', 'wrong')}
    correct = read_bound_json(controls['correct'])
    target = read_bound_json(value['task_ref'])['FAIL_TO_PASS']
    groups = _groups(backend._read_tests(Path(correct['test_execution']['report_ref']['path']), 'trx'), 'trx')
    production = [row['path'] for row in correct['source_inputs']
                  if not row['path'].startswith('test/Spectre.Console.Cli.Tests/')]
    evaluator = qualify_repository_strategy_evaluator(profile_ref=prepared['profile_ref'], controls=controls,
        production_paths=production, target_test_ids=target, preservation_test_ids=sorted(set(groups) - set(target)),
        output_root=root / 'qualification')
    return _save(root / 'qualification-result.json', {'status': 'qualified', 'evaluator_ref': evaluator,
        'controls': controls, 'official_labels_ref': value['task_ref'], 'model_calls': 0,
        'official_preservation_labels_count': len(read_bound_json(value['task_ref'])['PASS_TO_PASS']),
        'additional_preservation_scope': 'other actual tests; not official PASS_TO_PASS labels',
        'label_origin': value.get('label_origin', 'upstream_task'),
        'official_benchmark_score': False if value.get('label_origin') == 'independent_public_contract' else None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('static', 'prepare', 'baseline', 'correct', 'equivalent', 'wrong',
                                         'public-baseline', 'public-correct', 'public-equivalent', 'qualify'))
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--configuration-sha256', required=True)
    parser.add_argument('--attempt-id')
    args = parser.parse_args()
    reference = {'path': str(args.configuration.resolve()), 'sha256': args.configuration_sha256}
    value = load(reference)
    require(args.attempt_id is None or args.attempt_id.replace('-', '').isalnum(), 'ATTEMPT_ID_INVALID')
    stop = Path(value['output_root']) / 'dispatch-stop.json'
    if stop.is_file() and args.action != 'static':
        print(json.dumps({'status': 'dispatch_stopped_by_explicit_scope_revision', 'stop_ref': _ref(stop),
                          'model_calls': 0, 'build_calls': 0}), flush=True)
        return
    if args.action == 'static':
        print(json.dumps({'status': 'static_only', 'source_ready': value['source_ready'], 'model_calls': 0, 'build_calls': 0}))
        return
    if args.action == 'prepare':
        result = prepare(value, reference)
    elif args.action == 'qualify':
        result = qualify(value)
    else:
        result = run(value, args.action, args.attempt_id)
    print(json.dumps({'result_ref': result}), flush=True)


if __name__ == '__main__':
    main()
