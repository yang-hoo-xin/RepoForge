"""Native JSONL accounting and Windows-owned timeout cleanup, without models."""
import json
import os
import subprocess
import sys
import time

import pytest

from scripts.run_benchmark_arm import (
    codex_argv,
    codex_events,
    codex_resume_argv,
    execute,
    source_snapshot,
)


def test_jsonl_keeps_cached_tokens_distinct_and_unknown_calls(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in [
        {'type': 'thread.started', 'thread_id': 'native'},
        {'type': 'turn.completed', 'usage': {'input_tokens': 100, 'cached_input_tokens': 40, 'output_tokens': 20}},
        {'type': 'item.completed', 'item': {'type': 'command_execution', 'command': 'test'}},
    ]) + '\nnot-json\n', encoding='utf-8')
    result = codex_events(path)
    assert result['usage_totals'] == {'input_tokens': 100, 'cached_input_tokens': 40, 'output_tokens': 20}
    assert result['provider_send_count'] is None and result['invalid_jsonl_lines'][0]['line'] == 4
    assert len(result['tool_items']) == 1


def test_explicit_codex_configuration_has_no_invented_token_cap(tmp_path):
    args = codex_argv('codex', tmp_path / 'workspace', tmp_path / 'last.txt', [tmp_path / 'host'])
    assert args[args.index('--model') + 1] == 'gpt-5.6-sol'
    assert 'model_reasoning_effort="medium"' in args and '--ignore-user-config' in args
    assert not any('model_context_window=' in value or 'rollout_budget' in value for value in args)
    assert '--dangerously-bypass-approvals-and-sandbox' not in args
    assert 'permissions.benchmark.network.enabled=false' in args
    assert '--ephemeral' not in args


def test_resume_keeps_frozen_settings_and_exact_session(tmp_path):
    original = codex_argv('codex', tmp_path / 'workspace', tmp_path / 'first.txt', [tmp_path / 'host'])
    session = '12345678-1234-4234-9234-123456789abc'
    resumed = codex_resume_argv(original, session, tmp_path / 'second.txt')
    assert resumed[-2:] == [session, '-'] and '--last' not in resumed
    assert resumed[resumed.index('--model') + 1] == 'gpt-5.6-sol'
    assert [original[i+1] for i, x in enumerate(original) if x == '-c'] == [resumed[i+1] for i, x in enumerate(resumed) if x == '-c']
    with pytest.raises(ValueError):
        codex_resume_argv(original, 'latest', tmp_path / 'out')


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object contract')
def test_deadline_kills_owned_child_tree_and_retains_raw_output(tmp_path):
    delayed = tmp_path / 'escaped.txt'
    child = f'import time;from pathlib import Path;time.sleep(2);Path({str(delayed)!r}).write_text("escaped")'
    parent = 'import subprocess,sys,time;print("started",flush=True);subprocess.Popen([sys.executable,"-c",' + repr(child) + ']);time.sleep(60)'
    result = execute([sys.executable, '-I', '-B', '-c', parent], cwd=tmp_path, stdin_path=None,
                     output=tmp_path / 'process', deadline=time.monotonic() + .5)
    assert result['timeout'] is True and result['status'] == 'timed_out'
    assert 'started' in (tmp_path / 'process/stdout.bin').read_text()
    time.sleep(2.1)
    assert not delayed.exists()
    assert result['owned_process_tree_cleanup'] == 'job_closed_kill_on_close'
    assert result['broker_descendant_cleanup'] == 'not_established_by_generic_job_ownership'


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object contract')
@pytest.mark.parametrize('environment_mode', ['same', 'absent', 'later'])
def test_owned_process_passes_actual_utc_deadline_without_extending_inherited_window(tmp_path, monkeypatch, environment_mode):
    inherited = time.time() + 60
    monkeypatch.setenv('REPOFORGE_TASK_DEADLINE_UTC', str(inherited))
    env = {**os.environ, 'REPOFORGE_TASK_DEADLINE_UTC': str(inherited)}
    if environment_mode == 'absent':
        env.pop('REPOFORGE_TASK_DEADLINE_UTC')
    elif environment_mode == 'later':
        env['REPOFORGE_TASK_DEADLINE_UTC'] = str(inherited + 120)
    child = 'import os;print(os.environ.get("REPOFORGE_TASK_DEADLINE_UTC"),flush=True)'
    result = execute([sys.executable, '-I', '-B', '-c', child], cwd=tmp_path, stdin_path=None,
        output=tmp_path / 'bounded', deadline=time.monotonic() + 300, environment=env)
    assert result['exit_code'] == 0
    assert result['deadline_utc'] == inherited
    assert float((tmp_path / 'bounded/stdout.bin').read_text()) == inherited


def test_source_audit_ignores_build_outputs_but_catches_protected_edits(tmp_path):
    (tmp_path / 'Plugin.cs').write_text('before')
    (tmp_path / 'Project.csproj').write_text('fixed')
    before = source_snapshot(tmp_path)
    (tmp_path / 'obj').mkdir()
    (tmp_path / 'obj/generated.cs').write_text('generated')
    assert source_snapshot(tmp_path) == before
    (tmp_path / 'Project.csproj').write_text('changed')
    assert source_snapshot(tmp_path)['Project.csproj'] != before['Project.csproj']


@pytest.mark.skipif(os.name != 'nt', reason='Windows junction boundary')
def test_junction_in_build_output_is_rejected_before_traversal(tmp_path):
    workspace = tmp_path / 'workspace'
    outside = tmp_path / 'outside'
    (workspace / 'obj').mkdir(parents=True)
    outside.mkdir()
    sentinel = outside / 'private.txt'
    sentinel.write_text('unchanged')
    junction = workspace / 'obj' / 'escape'
    result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(junction), str(outside)], capture_output=True)
    assert result.returncode == 0
    with pytest.raises(ValueError, match='REPARSE'):
        source_snapshot(workspace)
    assert sentinel.read_text() == 'unchanged'
