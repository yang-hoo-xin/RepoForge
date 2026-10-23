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
