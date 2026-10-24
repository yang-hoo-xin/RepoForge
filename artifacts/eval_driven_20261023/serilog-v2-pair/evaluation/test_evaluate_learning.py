"""Pure admission counterexamples: no model, SDK, or private test reads."""
import importlib
import json
from pathlib import Path

import pytest

owner = importlib.import_module('artifacts.eval_driven_20261023.serilog-v2-pair.evaluation.evaluate_learning')


def test_live_process_cannot_admit_evaluation():
    with pytest.raises(ValueError, match='PROCESS_NOT_TERMINAL'):
        owner.terminal_gate({'status': 'running'}, {}, {}, {})


def test_terminal_other_run_cannot_admit_evaluation():
    with pytest.raises(ValueError, match='RUN_CHANGED'):
        owner.terminal_gate({'status': 'exited', 'finished_at': 'done'},
            {'plan_ref': 'old', 'run_id': 'old'}, {'run_id': 'new'}, 'new')


def test_terminal_other_platform_cannot_admit_evaluation():
    with pytest.raises(ValueError, match='PROCESS_PLATFORM_CHANGED'):
        owner.terminal_gate({'status': 'exited', 'finished_at': 'done', 'argv': ['wrong']},
            {'plan_ref': 'p', 'run_id': 'r'}, {'run_id': 'r', 'platform_ref': {'path': 'platform', 'sha256': 'hash'}}, 'p')


def test_text_candidate_always_replaces_template_content():
    assert owner.production_bytes('a.cs', {'a.cs': 'candidate'}, {}, None) == b'candidate'


def test_missing_production_never_falls_back_to_teacher():
    with pytest.raises(ValueError, match='MISSING_PRODUCTION_SOURCE'):
        owner.production_bytes('a.cs', {}, {}, None)


def test_unknown_submit_origin_is_not_autonomous_source():
    with pytest.raises(ValueError, match='SUBMIT_ORIGIN_MISSING'):
        owner.verify_origin({'completed_tool_records': []}, {'model_origin': {'call_id': 'missing'}}, lambda _: None)


def test_different_response_reference_is_rejected_before_any_read():
    with pytest.raises(ValueError, match='SUBMIT_ORIGIN_CHANGED'):
        owner.verify_origin({'completed_tool_records': [{'call_id': 'c', 'name': 'candidate_submit', 'response_ref': 'wrong'}]},
            {'model_origin': {'call_id': 'c', 'response_ref': 'actual'}}, lambda _: pytest.fail('read not expected'))


def test_missing_native_process_is_read_only(monkeypatch):
    monkeypatch.setattr(owner, 'read_bound_json', lambda ref: {'public_preparation_ref': {}})
    monkeypatch.setattr(owner.Path, 'is_file', lambda _: False)
    assert owner.inspect() == {'status': 'waiting_for_terminal_process'}


@pytest.fixture
def transport(monkeypatch):
    from src.mod_agent.evaluation import product_trial_batch
    from src.mod_agent.task_agent_v2_state import load_v2_artifact

    response = {'tool_calls': [{'call_id': 'submit', 'name': 'candidate_submit', 'arguments': {}}]}
    normalized = {'path': '/run/normalized.json', 'sha256': 'normalized'}
    wire = {'path': '/run/wire.json', 'sha256': 'wire'}
    response_ref = {'path': '/run/response.json', 'sha256': 'response'}
    candidate = {'candidate_source_id': 'source', 'model_origin': {'call_id': 'submit', 'response_ref': response_ref}}
    native = {'candidate_ref': 'candidate', 'provider_turns': [{'request_ref': normalized,
        'provider_request_ref': wire, 'response_ref': response_ref}]}
    measurement = {'model': {'calls': [{'transport': {'response_origin': 'live_provider_response',
        'response_received': True, 'response_complete': True}, 'response_sha256': owner.canonical_hash(response),
        'local_evidence': {'request': {'integrity': 'verified_bytes', 'sha256': 'normalized'},
                           'provider_request': {'integrity': 'verified_bytes', 'sha256': 'wire'}},
        'generation_span_id': 'span', 'trace_id': 'trace'}]}}
    monkeypatch.setattr(importlib.import_module(load_v2_artifact.__module__), 'load_v2_artifact', lambda *a: candidate)
    monkeypatch.setattr(product_trial_batch, '_read_reference',
        lambda reference, **kw: json.dumps(response).encode() if reference == response_ref else b'{}')
    return native, candidate, measurement


def test_live_final_submit_keeps_normalized_and_wire_refs_distinct(transport):
    result = owner.verify_live_transport(Path('/run'), *transport)
    assert result['normalized_request_ref']['sha256'] == 'normalized'
    assert result['actual_provider_request_ref']['sha256'] == 'wire'


def test_synthetic_final_response_is_rejected_even_with_another_live_response(transport):
    native, candidate, measurement = transport
    synthetic = measurement['model']['calls'][0]
    synthetic['transport']['response_origin'] = 'scripted_response'
    other = {**synthetic, 'transport': {**synthetic['transport'], 'response_origin': 'live_provider_response'},
             'response_sha256': 'some-other-response'}
    measurement['model']['calls'].append(other)
    with pytest.raises(ValueError, match='not bound to one retained live'):
        owner.verify_live_transport(Path('/run'), native, candidate, measurement)


def test_missing_actual_provider_request_is_rejected(transport):
    native, candidate, measurement = transport
    native['provider_turns'][0].pop('provider_request_ref')
    with pytest.raises(KeyError, match='provider_request_ref'):
        owner.verify_live_transport(Path('/run'), native, candidate, measurement)


def test_wrong_wire_hash_for_final_response_is_rejected(transport):
    native, candidate, measurement = transport
    measurement['model']['calls'][0]['local_evidence']['provider_request']['sha256'] = 'other-wire'
    with pytest.raises(ValueError, match='not bound to one retained live'):
        owner.verify_live_transport(Path('/run'), native, candidate, measurement)


@pytest.fixture
def configured(tmp_path):
    main = tmp_path / 'main'
    plan_ref = owner._save(main / 'frozen-plan.json', {'run_id': 'r',
        'platform_ref': {'path': 'frozen-platform', 'sha256': 'frozen-hash'}})
    value = {'schema_version': 'single_repository_candidate_consumer_v1',
        'plan_ref': plan_ref, 'evaluator_ref': {'path': 'private-evaluator', 'sha256': 'evaluator-hash'},
        'run_root': str(main / 'runs/r'),
        'public': {'profile_ref': {'path': 'public-profile', 'sha256': 'profile-hash'},
                   'source_root': str(tmp_path / 'public-source')}}
    return main, value, owner._save(tmp_path / 'consumer.json', value)


def test_configuration_is_explicit_and_does_not_mutate_legacy_plan(configured):
    main, value, reference = configured
    result = owner.consumer_configuration(reference)
    assert result['main'] == main.resolve()
    assert result['plan']['run_root'] == str(main / 'runs/r')
    assert result['plan']['environment'] == {}
    assert owner.read_bound_json(value['plan_ref']).get('run_root') is None
    assert result['public'] == value['public']


@pytest.mark.parametrize('field', ['plan_ref', 'evaluator_ref', 'public', 'run_root'])
def test_configuration_missing_required_field_rejected(configured, field):
    _, value, reference = configured
    value.pop(field)
    reference = owner._save(Path(reference['path']).with_name('missing-field.json'), value)
    with pytest.raises(ValueError, match='CONSUMER_CONFIGURATION_INVALID'):
        owner.consumer_configuration(reference)


def test_modified_configuration_cannot_use_old_hash(configured):
    _, value, reference = configured
    value['public']['source_root'] = 'different-source'
    Path(reference['path']).write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        owner.consumer_configuration(reference)


@pytest.mark.parametrize('kind', ['other_id', 'other_main'])
def test_configuration_cannot_select_another_run(configured, kind):
    main, value, reference = configured
    value['run_root'] = str(main / ('runs/other' if kind == 'other_id' else 'other-root/r'))
    reference = owner._save(Path(reference['path']).with_name('different-run.json'), value)
    with pytest.raises(ValueError, match='CONSUMER_RUN_IDENTITY_CHANGED'):
        owner.consumer_configuration(reference)


def test_bound_plan_hash_is_revalidated(configured):
    _, value, reference = configured
    Path(value['plan_ref']['path']).write_text(json.dumps({'run_id': 'different'}), encoding='utf-8')
    with pytest.raises(ValueError):
        owner.consumer_configuration(reference)


def test_configured_nonterminal_entry_does_not_load_judge_or_build(configured, monkeypatch):
    _, _, reference = configured
    monkeypatch.setattr(owner, '_configuration', lambda *args: pytest.fail('judge must not load'))
    monkeypatch.setattr(owner, 'execute', lambda *args: pytest.fail('must not Build'))
    assert owner.inspect(reference) == {'status': 'waiting_for_terminal_process'}


def test_legacy_configuration_keeps_old_reference_selection(monkeypatch):
    plan = {'public_preparation_ref': {'path': 'old-public', 'sha256': 'old-public-hash'}}
    monkeypatch.setattr(owner, 'read_bound_json', lambda ref: plan if ref == owner.PLAN_REF else {'legacy': True})
    result = owner.consumer_configuration()
    assert result['main'] == owner.MAIN
    assert result['plan_ref'] == owner.PLAN_REF
    assert result['evaluator_ref'] == owner.EVALUATOR_REF
    assert result['plan'] is plan
    assert result['public'] == {'legacy': True}


@pytest.mark.parametrize('native,status', [
    ({'status': 'blocked', 'candidate_ref': None}, 'no_candidate'),
    ({'status': 'unknown', 'candidate_ref': None}, 'native_unknown'),
])
def test_terminal_without_usable_candidate_never_starts_judge(configured, monkeypatch, native, status):
    main, value, reference = configured
    owner._save(main / 'execution/process.json', {'status': 'exited', 'finished_at': 'done',
        'argv': ['frozen-platform', 'frozen-hash']})
    owner._save(main / 'started.json', {'plan_ref': value['plan_ref'], 'run_id': 'r'})
    owner._save(main / 'runs/r/a/task-agent-execution/execution.json', native)
    monkeypatch.setattr(owner, '_configuration', lambda *args: pytest.fail('no usable candidate'))
    result = owner.inspect(reference)
    assert result['status'] == status
    assert result['delivered'] is False


def test_execute_flag_cannot_bypass_nonterminal_gate(configured, monkeypatch, capsys):
    _, _, reference = configured
    monkeypatch.setattr(owner, 'execute', lambda *args: pytest.fail('must not Build'))
    monkeypatch.setattr('sys.argv', ['consumer', '--configuration', reference['path'],
        '--configuration-sha256', reference['sha256'], '--execute', '--output', 'unused'])
    owner.main()
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'waiting_for_terminal_process'
    assert result['build_calls'] == 0
    assert result['task_success'] is None
