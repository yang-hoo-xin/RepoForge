"""One bounded real Verify reproduction through frozen shared repository sources.

为什么要有它：只改变测试诊断输出位置，并以真实执行原件区分业务失败与源码完整性失败。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('C:/tmp/rfdiag-real-1531')
REPO = Path(__file__).resolve().parents[4]
OLD = Path('C:/tmp/rfext-spectre-1531-evaluation/v2/public-correct')
INITIALIZER = 'test/Spectre.Console.Cli.Tests/VerifyConfiguration.cs'


def ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(reference):
    if ref(reference['path']) != reference:
        raise ValueError('REFERENCE_CHANGED')
    return json.loads(Path(reference['path']).read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
    return ref(path)


def prepare():
    from src.mod_agent.platform_execution_snapshot import prepare_snapshot

    ROOT.mkdir(parents=True, exist_ok=True)
    if {path.name for path in ROOT.iterdir()} != {'process'}:
        raise ValueError('FRESH_OWNED_PROCESS_DIRECTORY_REQUIRED')
    started = time.monotonic()
    script = ref(__file__)
    adapter = ref(Path(__file__).with_name('diagnostic-integration-VerifyConfiguration.cs'))
    intent = ref(REPO / 'artifacts/eval_driven_20261024/product-pilot/diagnostic-integration-intent.json')
    result = ref(OLD.parent / 'public-correct-result.json')
    if result['sha256'] != '424663640506936d86023e903d24d594b559d33718874f01e2ba31e68e611552':
        raise ValueError('RETAINED_ORIGINAL_CHANGED')
    original = read(result)
    integrity_ref = original['integrity_error_refs'][0]
    old_integrity = read(integrity_ref)
    snapshot = prepare_snapshot(source_root=REPO, snapshot_root=ROOT / 'platform',
        business_argv=['real-verify-diagnostic-reproduction'],
        extra_load_refs=[(script['path'], script['sha256'])],
        bound_inputs=[(item['path'], item['sha256']) for item in (adapter, intent, result, integrity_ref)])
    source = ROOT / 'source'
    source.mkdir()
    before = old_integrity['context']['source_inputs']
    expected = {row['path']: row['sha256'] for row in before}
    for relative, digest in expected.items():
        origin, target = OLD / 'source' / relative, source / relative
        if not target.resolve().is_relative_to(source.resolve()) or ref(origin)['sha256'] != digest:
            raise ValueError('RETAINED_SOURCE_CHANGED')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    known = read(ref(REPO / 'artifacts/eval_driven_20261024/product-pilot/evaluation/public-failure-analysis-addendum.json'))
    allowed_received = set(known['new_file_paths_since_recorded_public_check_inputs'])
    observed = set()
    pending = [OLD / 'source']
    while pending:
        for path in pending.pop().iterdir():
            if path.is_symlink():
                raise ValueError('RETAINED_LINK_UNSUPPORTED')
            if path.is_dir():
                if path.name not in {'bin', 'obj', '.generated'}:
                    pending.append(path)
            elif path.is_file():
                observed.add(path.relative_to(OLD / 'source').as_posix())
    if observed - set(expected) != allowed_received:
        raise ValueError('UNDECLARED_OMITTED_INPUT')
    shutil.copyfile(adapter['path'], source / INITIALIZER)
    old_profile = read(original['profile_ref'])
    profile = {**old_profile, 'timeout_seconds': 300,
        'diagnostics': {'kind': 'verify_snapshots_v1',
            'expectation_root': 'test/Spectre.Console.Cli.Tests/Expectations',
            'compiler_output_roots': [prefix + '/' + name for prefix in (
                'src/Spectre.Console', 'src/Spectre.Console.Cli', 'src/Spectre.Console.Testing',
                'test/Spectre.Console.Cli.Tests') for name in ('bin', 'obj')],
            'received_markers': ['.received.', '.DotNet8_0.received.']}}
    profile_ref = save(ROOT / 'profile.json', profile)
    build_fact = read(old_integrity['stage_receipts']['build'])
    restore = build_fact['commands']['restore']
    old_cache = Path(restore[restore.index('--packages') + 1]).parent
    config = save(ROOT / 'configuration.json', {'snapshot_ref': snapshot, 'profile_ref': profile_ref,
        'original_result_ref': result, 'original_integrity_ref': integrity_ref, 'initializer_ref': adapter,
        'source_before': before, 'source_root': str(source), 'omitted_known_received': sorted(allowed_received),
        'cache_origin': str(old_cache), 'controller_ref': script, 'intent_ref': intent,
        'prepare_seconds': time.monotonic() - started, 'model_calls': 0})
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'platform'), 'PYTHONDONTWRITEBYTECODE': '1'}
    command = [sys.executable, '-B', str(Path(__file__).resolve()), 'execute']
    # Fresh interpreter imports src/scripts exclusively from the selected snapshot.
    completed = subprocess.run(command, cwd=ROOT / 'platform', env=environment,
        capture_output=True, timeout=max(1, float(environment['REPOFORGE_TASK_DEADLINE_UTC']) - time.time()))
    (ROOT / 'backend.stdout.bin').write_bytes(completed.stdout)
    (ROOT / 'backend.stderr.bin').write_bytes(completed.stderr)
    save(ROOT / 'controller.json', {'configuration_ref': config, 'command': command,
        'exit_code': completed.returncode, 'stdout_ref': ref(ROOT / 'backend.stdout.bin'),
        'stderr_ref': ref(ROOT / 'backend.stderr.bin'), 'model_calls': 0})
    print(json.dumps({'configuration_ref': config, 'exit_code': completed.returncode}), flush=True)


def execute():
    from src.mod_agent.domain import csharp_repository_backend as backend
    from src.mod_agent.platform_execution_snapshot import FrozenExecution

    config = read(ref(ROOT / 'configuration.json'))
    guard = FrozenExecution(config['snapshot_ref'])
    guard.activate()
    source, clock = Path(config['source_root']), time.monotonic()
    profile = backend.CSharpRepositoryProfile(config['profile_ref'])
    current = backend._snapshot(source)
    expected = {row['path']: row['sha256'] for row in config['source_before']}
    expected[INITIALIZER] = config['initializer_ref']['sha256']
    if {row['path']: row['sha256'] for row in current} != expected:
        raise ValueError('ONLY_INITIALIZER_MAY_CHANGE')
    cache = profile.task_work_root(ROOT)
    cache.mkdir(parents=True)
    cache_clock = time.monotonic()
    for name in ('p', 'h'):
        origin = Path(config['cache_origin']) / name
        if origin.exists():
            shutil.copytree(origin, cache / name)
    cache_seconds = time.monotonic() - cache_clock
    value = {'status': 'unknown', 'model_calls': 0, 'snapshot_ref': config['snapshot_ref'],
        'profile_ref': config['profile_ref'], 'source_root': str(source), 'cache_copy_seconds': cache_seconds,
        'source_preserved_except_fixed_initializer': True, 'original_result_ref': config['original_result_ref']}
    backend_clock = time.monotonic()
    try:
        runner = backend.CSharpRepositoryBackend(profile_ref=profile.reference, source_root=source,
            workspace_root=ROOT, output_root=ROOT / 'checks', expected_fixed_inputs=profile.bind(source))
        result = runner.run_build_test(source, 'real-verify-diagnostics', 'independent_implementation_verification')
        value.update(status=result['status'], report_ref=result['report_ref'],
            verification_summary=result['verification_summary'], diagnostics_ref=result.get('diagnostics_ref'),
            integrity_ref=result['integrity_ref'], stage_receipts=result['stage_receipts'])
        backend.validate_diagnostic_integrity(read(result['report_ref']))
    except Exception as error:
        value.update(status='unknown', error_type=type(error).__name__, error=str(error))
        value['retained_stage_refs'] = [ref(path) for path in sorted((ROOT / 'checks').glob('*/*-fact.json'))]
        value['integrity_error_refs'] = [ref(path) for path in sorted((ROOT / 'checks').glob('*/integrity-error.json'))]
    value.update(backend_seconds=time.monotonic() - backend_clock, execution_seconds=time.monotonic() - clock,
        frozen_execution_check_ref=save(ROOT / 'frozen-execution-check.json', guard.check()),
        claim='diagnostic mechanism only; not implementation acceptance or evaluator qualification')
    reference = save(ROOT / 'result.json', value)
    print(json.dumps({'result_ref': reference, 'status': value['status']}), flush=True)


if __name__ == '__main__':
    prepare() if sys.argv[1:] == ['prepare'] else execute()
