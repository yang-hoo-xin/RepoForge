"""Saved-request defaults remain separate from real model arguments and authority."""
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.model_qualification import canonical_hash
from src.mod_agent.task_agent_loop import TaskCandidateRevisionInputError
from src.mod_agent.task_agent_v2_recovery import (
    _validate_completed,
    restore_session,
    session_snapshot,
)
from src.mod_agent.task_agent_v2_state import TaskAgentV2StateError
from src.mod_agent.task_agent_v2_tools import resolved_origin_arguments, validate_origin_request
from tests.unit.test_task_agent_v2_admission import action, response
from tests.unit.test_task_agent_v2_admission import session as session  # noqa: F401
from tests.unit.test_task_agent_v2_state import edit
from tests.unit.test_task_agent_v2_state import state as state  # noqa: F401


def bind(session):
    request = session.owner.prepare_decision_request(session.initial_request(), session)
    reference = session.owner._save('actual-request-' + str(len(session.actual_calls)) + '.json', asdict(request))
    session.bind_actual_request(reference)
    return reference


def test_omitted_revision_executes_and_retains_raw_request_resolution(session):
    reference = bind(session)
    raw = {'path': 'A.cs'}
    session.response_admission(response('source_read', raw, 'read-default'))
    selected = action(session, 'source_read')
    prepared = selected.prepare(raw)
    session.observe_dispatch_event({'event': 'tool_dispatch_started', 'call_id': 'read-default'})
    result = selected.execute(prepared.value)
    origin = session.actual_calls[0]
    assert result['text'] == 'a = 1;\n'
    assert raw == origin['arguments'] == {'path': 'A.cs'}
    assert json.loads(Path(origin['arguments_ref']['path']).read_text()) == raw
    assert origin['request_ref'] == reference
    assert origin['effective_arguments'] == prepared.value
    assert origin['argument_resolution']['defaulted_fields']['parent_revision_ref']['value'] == session.state.current_revision_ref
    assert resolved_origin_arguments(origin) == prepared.value


def test_request_time_parent_survives_state_advance_for_read_but_cannot_edit(session):
    bind(session)
    baseline = session.state.current_revision_ref
    old_view = session.state.read_source(baseline, 'A.cs')['source_view_ref']
    edit(session.state, 'A.cs', '1', '2')
    read = action(session, 'source_read').prepare({'path': 'A.cs'})
    assert read.value['parent_revision_ref'] == baseline
    assert read.version_identity == canonical_hash({'contract': session.state.contract_ref, 'revision': baseline})
    with pytest.raises(TaskAgentV2StateError, match='stale_parent_revision'):
        action(session, 'candidate_submit').prepare({'edits': [{'path': 'A.cs', 'source_view_ref': old_view,
            'before_text': '1', 'after_text': '3'}]})


def test_explicit_history_and_invalid_explicit_values_are_not_overridden(session):
    edit(session.state, 'A.cs', '1', '2')
    bind(session)
    selected = action(session, 'source_read')
    assert selected.prepare({'path': 'A.cs', 'parent_revision_ref': 'baseline'}).value['parent_revision_ref'] == 'baseline'
    with pytest.raises(Exception):
        selected.prepare({'path': 'A.cs', 'parent_revision_ref': 'not-a-revision'})


def test_registry_default_does_not_refresh_after_actual_request(session):
    bind(session)
    view = session.state.read_source('baseline', 'A.cs')['source_view_ref']
    session.state.add_binding(session.state.registry_ref, view, 'A.cs')
    with pytest.raises(TaskAgentV2StateError):
        action(session, 'binding_add').prepare({'source_view_ref': view, 'target_selector': 'A.cs',
            'rationale': 'arithmetic', 'expected_validation': 'value'})


@pytest.mark.parametrize('missing', ['source_view_ref', 'before_text', 'after_text'])
def test_edit_evidence_is_still_explicit(session, missing):
    bind(session)
    item = {'path': 'A.cs', 'source_view_ref': session.state.read_source('baseline', 'A.cs')['source_view_ref'],
            'before_text': '1', 'after_text': '2'}
    item.pop(missing)
    with pytest.raises(TaskCandidateRevisionInputError):
        action(session, 'candidate_submit').prepare({'edits': [item]})


def test_origin_tampering_and_unsent_request_are_rejected(session):
    bind(session)
    session.response_admission(response('source_read', {'path': 'A.cs'}, 'read'))
    origin = session.actual_calls[0]
    validate_origin_request(origin, [{'request_ref': origin['request_ref'], 'response_ref': origin['response_ref']}])
    with pytest.raises(ValueError, match='REQUEST_NOT_SENT'):
        validate_origin_request(origin, [])
    changed = deepcopy(origin)
    changed['effective_arguments']['parent_revision_ref'] = 'baseline'
    with pytest.raises(ValueError, match='RESOLUTION_CHANGED'):
        resolved_origin_arguments(changed)
    stripped = {key: value for key, value in origin.items()
        if key not in {'request_ref', 'effective_arguments', 'argument_resolution'}}
    with pytest.raises(ValueError, match='REQUEST_REQUIRED'):
        resolved_origin_arguments(stripped)
    legacy = {'name': 'source_read', 'arguments': {'parent_revision_ref': 'baseline', 'path': 'A.cs'}}
    assert resolved_origin_arguments(legacy) == legacy['arguments']
    assert resolved_origin_arguments({'name': 'test_catalog', 'arguments': {}}) == {}
    changed = deepcopy(origin)
    changed['argument_resolution']['defaulted_fields']['parent_revision_ref']['locator'][-1] = 'baseline_revision_ref'
    with pytest.raises(ValueError, match='RESOLUTION_CHANGED'):
        resolved_origin_arguments(changed)


def test_checkpoint_restores_saved_request_without_live_head_default(session):
    reference = bind(session)
    before = session.state.current_revision_ref
    snapshot = session_snapshot(session)
    edit(session.state, 'A.cs', '1', '2')
    session._actual_request_ref = None
    restore_session(session, snapshot)
    assert session._actual_request_ref == reference
    assert action(session, 'source_read').prepare({'path': 'A.cs'}).value['parent_revision_ref'] == before


def test_completed_read_cannot_delete_all_host_resolution_fields(session):
    session.owner.output_root = session.state.store_root / 'resolution-recovery'
    session.owner.output_root.mkdir()
    bind(session)
    raw = {'path': 'A.cs'}
    session.response_admission(response('source_read', raw, 'read'))
    origin = session.actual_calls[0]
    output = session.state.read_source(session.state.current_revision_ref, 'A.cs')
    returned = {'execution': 'executed', 'value': output}
    event = {'event': 'tool_dispatch_completed', 'call_id': 'read', 'role': 'source_read',
        'output_sha256': canonical_hash(returned)}
    receipt = {'origin': deepcopy(origin), 'output': output,
        'task_head': {'contract_ref': session.state.contract_ref},
        'session': {'current_check': None, 'public_checks': []}}
    origin['execution_receipt_ref'] = session.owner._save('read-receipt.json', receipt)
    checkpoint = {'task_contract_ref': session.state.contract_ref,
        'session': {'actual_calls': [origin], 'current_check': None, 'public_checks': []},
        'details': {'model_profile': session.owner.profile.to_dict(), 'provider_turns': [
            {'request_ref': origin['request_ref'], 'response_ref': origin['response_ref']}]},
        'loop': {'phase': 'dispatch_ready', 'events': [event], 'call_records': {'read': {
            'status': 'completed', 'arguments_sha256': canonical_hash(raw), 'result': {'output': returned}}}}}
    _validate_completed(checkpoint, session.state.store_root)
    changed = deepcopy(checkpoint)
    forged_origin = changed['session']['actual_calls'][0]
    for field in ('request_ref', 'effective_arguments', 'argument_resolution'):
        forged_origin.pop(field)
        receipt['origin'].pop(field)
    forged_origin['execution_receipt_ref'] = session.owner._save('forged-read-receipt.json', receipt)
    with pytest.raises(ValueError, match='REQUEST_REQUIRED'):
        _validate_completed(changed, session.state.store_root)


def test_current_candidate_check_and_probe_defaults_keep_model_selection_explicit(session):
    candidate = edit(session.state, 'A.cs', '1', '2')['candidate_ref']
    bind(session)
    assert action(session, 'candidate_check').prepare({}).value == {'candidate_ref': candidate}
    probe = {'files': [{'relative_path': 'Program.cs', 'content': 'class Arithmetic {}'}],
        'entrypoint_id': 'csharp_console', 'rationale': 'addition', 'expected_observation': 'sum'}
    assert action(session, 'probe_submit').prepare(probe).value['source_revision_ref'] == candidate
    with pytest.raises(ValueError, match='SELECTION_REQUIRED'):
        action(session, 'experiment_run').prepare({'kind': 'temporary_probe', 'hypothesis': 'addition', 'expected_observation': 'sum'})
    assert 'source_view_ref' in action(session, 'candidate_submit').definition.parameters['properties']['edits']['items']['required']
    for selected in session.actions():
        assert 'working_state' not in selected.definition.parameters.get('required', [])


@pytest.mark.parametrize('name', ['candidate_check', 'finish'])
def test_missing_candidate_is_not_defaulted_to_baseline_or_dispatched(session, name):
    reference = bind(session)
    current = json.loads(Path(reference['path']).read_bytes())['messages'][-1]['content']['current_task_state']
    assert current.get('current_candidate_ref') is None
    session.response_admission(response(name, {}, 'not-ready'))
    assert session.actual_calls[-1]['effective_arguments'] == {}
    with pytest.raises(ValueError, match='TASK_AGENT_CANDIDATE_NOT_READY'):
        action(session, name).prepare({})
    assert session.current_check is None and session.public_checks == []
    assert session.state.current_revision_ref == session.state.contract['baseline_manifest_ref']


def test_plan_update_reports_source_revision_without_inventing_a_candidate(session):
    bind(session)
    raw = {'plan': {'hypotheses': ['inspect source']}}
    session.response_admission(response('plan_update', raw, 'plan'))
    selected = action(session, 'plan_update')
    result = selected.execute(selected.prepare(raw).value)
    assert result['candidate_ref'] is None
    assert result['current_revision_ref'] == session.state.current_revision_ref


def test_candidate_binding_is_request_bound_and_rejects_forged_readiness(session):
    baseline_request = bind(session)
    session.response_admission(response('candidate_check', {}, 'before-candidate'))
    candidate = edit(session.state, 'A.cs', '1', '2')['candidate_ref']
    with pytest.raises(ValueError, match='TASK_AGENT_CANDIDATE_NOT_READY'):
        action(session, 'candidate_check').prepare({})
    current_request = bind(session)
    assert action(session, 'candidate_check').prepare({}).value == {'candidate_ref': candidate}
    document = json.loads(Path(current_request['path']).read_bytes())
    document['messages'][-1]['content']['current_task_state']['current_candidate_ref'] = None
    forged = session.owner._save('forged-candidate-boundary.json', document)
    with pytest.raises(ValueError, match='ACTUAL_REQUEST_BOUNDARY_CHANGED'):
        session.bind_actual_request(forged)
    assert baseline_request != current_request


def test_missing_check_for_real_candidate_is_correctable_but_explicit_invalid_values_are_not(session):
    candidate = edit(session.state, 'A.cs', '1', '2')['candidate_ref']
    bind(session)
    with pytest.raises(ValueError, match='TASK_AGENT_CANDIDATE_NOT_READY'):
        action(session, 'finish').prepare({})
    for raw in ({'candidate_ref': 7}, {'candidate_ref': 'baseline'}, {'candidate_ref': 'forged'}):
        with pytest.raises(ValueError) as error:
            action(session, 'candidate_check').prepare(raw)
        assert 'TASK_AGENT_CANDIDATE_NOT_READY' not in str(error.value)
    assert action(session, 'candidate_check').prepare({'candidate_ref': candidate}).value['candidate_ref'] == candidate


def test_legacy_saved_request_defaults_are_not_reinterpreted_as_new_readiness(session):
    from src.mod_agent.task_agent_v2_tools import resolve_tool_arguments

    reference = bind(session)
    document = json.loads(Path(reference['path']).read_bytes())
    document['messages'][-1]['content']['current_task_state'].pop('current_candidate_ref', None)
    legacy = session.owner._save('legacy-request.json', document)
    session.bind_actual_request(legacy)
    effective, resolution = resolve_tool_arguments('candidate_check', {}, legacy)
    assert effective == {'candidate_ref': session.state.current_revision_ref}
    assert resolution['defaulted_fields']['candidate_ref']['locator'][-1] == 'current_revision_ref'


def test_failed_public_episode_rejects_self_consistent_unsent_request(session):
    from src.mod_agent.model_qualification import read_bound_json
    from src.mod_agent.task_agent_v2_experience import seal_current_public_episode
    from src.mod_agent.task_agent_v2_tools import resolve_tool_arguments

    session.owner.output_root = session.state.store_root / 'failed-episode'
    session.owner.output_root.mkdir()
    actual_request_ref = bind(session)
    raw = {'path': 'A.cs'}
    session.response_admission(response('source_read', raw, 'read'))
    origin = session.actual_calls[0]
    output = session.state.read_source(session.state.current_revision_ref, 'A.cs')
    receipt = {'origin': deepcopy(origin), 'output': output, 'task_head': deepcopy(session.state.head)}
    origin['execution_receipt_ref'] = session.owner._save('actual-read-receipt.json', receipt)
    execution = {'schema_version': 'task-agent-product-execution.v2', 'status': 'blocked',
        'candidate_ref': None, 'check_ref': None, 'unknown_effects': False,
        'run_id': 'arithmetic-failure', 'request_id': 'no-candidate',
        'completed_tool_records': [deepcopy(origin)], 'provider_turns': [
            {'request_ref': actual_request_ref, 'response_ref': origin['response_ref']}]}
    execution_ref = session.owner._save('actual-execution.json', execution)
    session.owner.execution_receipt = lambda: SimpleNamespace(execution_ref=execution_ref)
    episode_ref = seal_current_public_episode(session.owner, session)
    episode = read_bound_json(episode_ref)
    assert episode['information_gain'] == ['public_process_observation']
    requested = [item for item in episode['selections'] if item['role'] == 'action_request']
    assert [item['locator'][-1] for item in requested] == ['name', 'arguments']
    assert read_bound_json(origin['arguments_ref']) == raw
    # All forged local records agree, but this copied request was never the provider turn's input.
    copied = session.owner._save('unsent-request.json', read_bound_json(actual_request_ref))
    origin['request_ref'] = copied
    origin['effective_arguments'], origin['argument_resolution'] = resolve_tool_arguments('source_read', raw, copied)
    receipt['origin'] = {key: value for key, value in origin.items() if key != 'execution_receipt_ref'}
    origin['execution_receipt_ref'] = session.owner._save('forged-read-receipt.json', receipt)
    execution['completed_tool_records'] = [deepcopy(origin)]
    execution_ref = session.owner._save('forged-execution.json', execution)
    with pytest.raises(ValueError, match='REQUEST_NOT_SENT'):
        seal_current_public_episode(session.owner, session)
