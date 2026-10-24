"""Evaluator-only control runner; never imported into solver input.

为什么要有它：独立控制保留实际构建和测试原件，不把参考实现或隐藏测试回流产品模型。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from src.mod_agent.domain import csharp_repository_backend as backend  # noqa: E402
from src.mod_agent.domain.full_project_compilation import _ref, _save  # noqa: E402
from src.mod_agent.evaluation.strategy_repository_evaluator import _groups  # noqa: E402

ROOT = Path('C:/tmp/rfext-spectre-1552-evaluation')
BASE = Path('C:/tmp/rfext-spectre-1552')
PRIVATE = Path('C:/tmp/rfext-spectre-1552-task/evaluation-only')
PUBLIC = REPO / 'artifacts/eval_driven_20261024/spectre-1552/RepoForgePublicHelpAliasTests.cs'


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    labels = json.loads((PRIVATE / 'task.json').read_text(encoding='utf-8'))
    inputs = {name: _ref(PRIVATE / name) for name in
              ('task.json', 'test.patch', 'reference.patch', 'equivalent.patch', 'wrong.patch')}
    sources = {}
    for role in ('baseline', 'correct', 'equivalent', 'wrong'):
        source = ROOT / role / 'source'
        shutil.copytree(BASE, source, ignore=shutil.ignore_patterns(*backend._IGNORED))
        for name in (['test.patch', 'reference.patch'] if role in ('correct', 'equivalent') else ['test.patch']):
            subprocess.run(['git', 'apply', str(PRIVATE / name)], cwd=source, check=True, timeout=30, capture_output=True)
        if role == 'equivalent':
            subprocess.run(['git', 'apply', str(PRIVATE / 'equivalent.patch')], cwd=source,
                           check=True, timeout=30, capture_output=True)
        if role == 'wrong':
            subprocess.run(['git', 'apply', str(PRIVATE / 'wrong.patch')], cwd=source,
                           check=True, timeout=30, capture_output=True)
        shutil.copyfile(PUBLIC, source / 'test/Spectre.Console.Cli.Tests/RepoForgePublicHelpAliasTests.cs')
        sources[role] = {'source': str(source), 'source_inputs': backend._snapshot(source)}
    allowed = [row['path'] for row in sources['baseline']['source_inputs']
               if row['path'].startswith('src/Spectre.Console.Cli/') and row['path'].endswith('.cs')]
    profile_ref = _save(ROOT / 'profile.json', {
        'profile_version': 'csharp_repository_profile_v1',
        'dotnet_ref': _ref('C:/tmp/rf-dotnet-8.0.302/sdk/dotnet.exe'),
        'project_path': 'src/Spectre.Console.Cli/Spectre.Console.Cli.csproj',
        'test_project_path': 'test/Spectre.Console.Cli.Tests/Spectre.Console.Cli.Tests.csproj',
        'nuget_config_path': 'NuGet.Config', 'configuration': 'Release',
        'allowed_edit_paths': allowed,
        'expected_artifacts': ['src/Spectre.Console.Cli/bin/Release/net8.0/Spectre.Console.Cli.dll'],
        'build_properties': {'MinVerSkip': 'true', 'UseBuildTimeTools': 'false'},
        'test_command': ['test', '{test_project}', '--configuration', '{configuration}', '--framework', 'net8.0',
                         '--logger', 'trx;LogFileName={report_path}'],
        'test_report': {'format': 'trx', 'path': 'tests.trx'}, 'timeout_seconds': 300,
    })
    _save(ROOT / 'prepared.json', {'sources': sources, 'profile_ref': profile_ref, 'labels_ref': inputs['task.json'],
        'inputs': inputs, 'public_ref': _ref(PUBLIC), 'base_commit': labels['base_commit'], 'model_calls': 0,
        'scope': 'original framework build; only net8.0 execution; host workspace not OS sandbox',
        'deadline_seconds': 1200, 'command_limit_seconds': 600,
        'baseline_manifest': backend._snapshot(BASE), 'official_pass_to_pass_count': len(labels['PASS_TO_PASS'])})
    print(json.dumps({'status': 'prepared', 'prepared_ref': _ref(ROOT / 'prepared.json')}), flush=True)


def run(role, adapted=False):
    prepared = json.loads((ROOT / 'prepared.json').read_text(encoding='utf-8'))
    profile = backend.CSharpRepositoryProfile(prepared['profile_ref'])
    selected = ROOT / 'controls-v2' if adapted else ROOT
    source = selected / role / 'source'
    started = time.monotonic()
    value = {'role': role, 'model_calls': 0, 'source_root': str(source), 'profile_ref': profile.reference}
    try:
        runner = backend.CSharpRepositoryBackend(profile_ref=profile.reference, source_root=source,
            workspace_root=selected / role, output_root=selected / role / 'checks', expected_fixed_inputs=profile.bind(source))
        report = runner.run_build_test(source, role, 'independent_evaluator_qualification')
        value.update(status=report['status'], report_ref=report['report_ref'])
    except Exception as error:
        value.update(status='incomplete', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    reports = list((selected / role / 'checks').rglob('tests.trx'))
    if len(reports) == 1:
        rows = backend._read_tests(reports[0], 'trx')
        groups = _groups(rows, 'trx')
        labels = json.loads(Path(prepared['labels_ref']['path']).read_text(encoding='utf-8'))
        target = set(labels['FAIL_TO_PASS'])
        public = [row for row in rows if any('RepoForgePublicHelpAliasTests' in item.get('class_name', '') for item in row['test_definitions'])]
        value.update(trx_ref=_ref(reports[0]), actual_tests=len(rows), failures=sum(not row['passed'] for row in rows),
            official_target_labels=len(target), official_labels_found=sum(label in groups for label in target),
            official_target_instances=sum(len(groups.get(label, [])) for label in target),
            official_target_passed=sum(sum(groups.get(label, [])) for label in target),
            public_tests=len(public), public_passed=sum(row['passed'] for row in public),
            original_other_tests=len(rows) - len(public) - sum(len(groups.get(label, [])) for label in target),
            original_other_failures=sum(not row['passed'] for row in rows if row not in public
                and not any((item['class_name'] + '.' + item['method_name']) in target for item in row['test_definitions'])),
            groups=groups)
    value['elapsed_seconds'] = time.monotonic() - started
    reference = _save(selected / (role + '-result.json'), value)
    print(json.dumps({'role': role, 'status': value['status'], 'result_ref': reference,
                      **{key: value.get(key) for key in ('actual_tests', 'failures', 'official_target_passed', 'public_passed')}}), flush=True)


def adapt_snapshot_output():
    """Keep snapshot comparison bytes unchanged; only relocate writable received files.

    为什么要有它：测试工具生成诊断文件不能修改被冻结源码；该适配不改变预期文本和比较规则。
    """
    selected = ROOT / 'controls-v2'
    selected.mkdir(exist_ok=False)
    initializer = '''namespace Spectre.Console.Tests;

public static class VerifyConfiguration
{
    [ModuleInitializer]
    public static void Init()
    {
        Verifier.DerivePathInfo((sourceFile, projectDirectory, type, method) =>
        {
            var original = Expectations.Initialize(sourceFile, projectDirectory, type, method);
            var directory = System.IO.Path.Combine(projectDirectory, "TestResults", "verify",
                System.IO.Path.GetRelativePath(projectDirectory, original.Directory));
            System.IO.Directory.CreateDirectory(directory);
            foreach (var file in System.IO.Directory.EnumerateFiles(original.Directory, "*.verified.*"))
            {
                System.IO.File.Copy(file, System.IO.Path.Combine(directory, System.IO.Path.GetFileName(file)), true);
            }
            return new PathInfo(directory, original.TypeName, original.MethodName);
        });
    }
}
'''
    refs = {}
    for role in ('correct', 'equivalent', 'wrong'):
        source = selected / role / 'source'
        shutil.copytree(ROOT / role / 'source', source, ignore=shutil.ignore_patterns(*backend._IGNORED, '*.received.*'))
        path = source / 'test/Spectre.Console.Cli.Tests/VerifyConfiguration.cs'
        path.write_text(initializer, encoding='utf-8')
        refs[role] = _ref(path)
    _save(selected / 'snapshot-output-environment.json', {'reason': 'failed snapshots must not create files in immutable source',
        'original_initializer_ref': _ref(BASE / 'test/Spectre.Console.Cli.Tests/VerifyConfiguration.cs'),
        'adapted_initializer_refs': refs, 'original_expectations_unchanged': True,
        'output_directory': 'test/Spectre.Console.Cli.Tests/TestResults/verify', 'model_calls': 0,
        'scope': 'independent qualification environment only; not a business implementation change'})
    print(json.dumps({'status': 'snapshot_output_adapted', 'root': str(selected)}), flush=True)


def prepare_public_control():
    source = ROOT / 'public-correct' / 'source'
    shutil.copytree(ROOT / 'correct' / 'source', source, ignore=shutil.ignore_patterns(*backend._IGNORED))
    subprocess.run(['git', 'apply', '-R', str(PRIVATE / 'test.patch')], cwd=source,
                   check=True, timeout=30, capture_output=True)
    _save(ROOT / 'public-correct-preparation.json', {'source_root': str(source),
        'source_inputs': backend._snapshot(source), 'public_ref': _ref(PUBLIC),
        'condition': 'reference implementation with original tests plus public reproduction only; no hidden test.patch',
        'model_calls': 0})
    print(json.dumps({'status': 'public_control_prepared'}), flush=True)


def qualify():
    from src.mod_agent.evaluation.strategy_repository_evaluator import (
        qualify_repository_strategy_evaluator,
    )
    selected = ROOT / 'controls-v2'
    prepared = json.loads((ROOT / 'prepared.json').read_text(encoding='utf-8'))
    controls = {role: json.loads((selected / (role + '-result.json')).read_text(encoding='utf-8'))['report_ref']
                for role in ('correct', 'equivalent', 'wrong')}
    correct = json.loads(Path(controls['correct']['path']).read_text(encoding='utf-8'))
    labels = json.loads(Path(prepared['labels_ref']['path']).read_text(encoding='utf-8'))
    groups = _groups(backend._read_tests(Path(correct['test_execution']['report_ref']['path']), 'trx'), 'trx')
    target = labels['FAIL_TO_PASS']
    production = [row['path'] for row in correct['source_inputs']
                  if not row['path'].startswith('test/Spectre.Console.Cli.Tests/')]
    evaluator = qualify_repository_strategy_evaluator(profile_ref=prepared['profile_ref'], controls=controls,
        production_paths=production, target_test_ids=target,
        preservation_test_ids=sorted(set(groups) - set(target)), output_root=selected / 'qualification')
    result = _save(ROOT / 'qualification-result.json', {'status': 'qualified', 'evaluator_ref': evaluator,
        'controls': controls, 'official_labels_ref': prepared['labels_ref'], 'model_calls': 0,
        'official_preservation_labels_count': len(labels['PASS_TO_PASS']),
        'additional_preservation_scope': 'all other actually executed tests, not official PASS_TO_PASS labels',
        'scope': 'net8.0 test execution; original multitarget build; host workspace, not OS sandbox'})
    print(json.dumps({'status': 'qualified', 'result_ref': result}), flush=True)


def summarize():
    prepared = json.loads((ROOT / 'prepared.json').read_text(encoding='utf-8'))
    rows = {}
    for name, path in [('initial_correct', ROOT / 'correct-result.json'),
                       ('initial_baseline', ROOT / 'baseline-result.json'),
                       ('public_correct', ROOT / 'public-correct-result.json'),
                       *[(role, ROOT / 'controls-v2' / (role + '-result.json'))
                         for role in ('correct', 'equivalent', 'wrong')]]:
        if path.is_file():
            raw = json.loads(path.read_text(encoding='utf-8'))
            rows[name] = {key: raw.get(key) for key in ('status', 'actual_tests', 'failures',
                'official_target_labels', 'official_labels_found', 'official_target_instances',
                'official_target_passed', 'public_tests', 'public_passed', 'original_other_tests',
                'original_other_failures', 'elapsed_seconds', 'error_type', 'error')}
            rows[name].update(result_ref=_ref(path), report_ref=raw.get('report_ref'), trx_ref=raw.get('trx_ref'))
        else:
            rows[name] = {'status': 'not_completed'}
    qualified = ROOT / 'qualification-result.json'
    actual_processes = [ROOT / 'correct-process/process.json', ROOT / 'baseline-process/process.json',
                        ROOT / 'public-correct-process/process.json',
                        *[ROOT / 'controls-v2' / (role + '-process/process.json')
                          for role in ('correct', 'equivalent', 'wrong')]]
    processes = [json.loads(path.read_text(encoding='utf-8')) for path in actual_processes if path.is_file()]
    summary = {'schema_version': 'repository_control_qualification_summary_v1',
        'status': 'qualified' if qualified.is_file() else 'incomplete',
        'task_id': 'spectreconsole__spectre-console-1552', 'base_commit': prepared['base_commit'],
        'qualification_runner_ref': _ref(__file__),
        'private_controls_ref': _ref(ROOT / 'controls-v2/control-inputs.json'),
        'profile_ref': prepared['profile_ref'], 'source_files': len(prepared['baseline_manifest']),
        'original_source_unchanged': backend._snapshot(BASE) == prepared['baseline_manifest'],
        'controls': rows, 'qualified_result_ref': _ref(qualified) if qualified.is_file() else None,
        'process_refs': [_ref(path) for path in actual_processes if path.is_file()],
        'recorded_process_wall_seconds': sum(item['wall_seconds'] for item in processes),
        'process_time_scope': 'serial owned execution windows; excludes artifact preparation, cache copy and parent investigation',
        'official_preservation_labels_count': prepared['official_pass_to_pass_count'],
        'additional_preservation_scope': 'other actual tests, not official preservation labels',
        'environment_adaptation_ref': _ref(ROOT / 'controls-v2/snapshot-output-environment.json'),
        'scope': prepared['scope'], 'model_calls': 0, 'autonomous_success': False,
        'limitations': ['qualification controls only, not a product-model result',
                       'not official Linux or multi-runtime test execution',
                       'host process execution, not operating-system sandbox',
                       'initial baseline integrity failure retained, not converted into trusted feedback']}
    reference = _save(Path(__file__).with_name('result.json'), summary)
    print(json.dumps({'summary_ref': reference, 'status': summary['status']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'baseline', 'correct', 'equivalent', 'wrong', 'adapt', 'qualify',
                                         'prepare-public', 'public-correct', 'summarize'))
    parser.add_argument('--adapted', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'adapt':
        adapt_snapshot_output()
    elif args.action == 'qualify':
        qualify()
    elif args.action == 'prepare-public':
        prepare_public_control()
    elif args.action == 'summarize':
        summarize()
    else:
        run(args.action, args.adapted)
