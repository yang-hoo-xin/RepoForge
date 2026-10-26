"""Product handoff recomputes current-ID defaults without rewriting model calls."""
from copy import deepcopy
from dataclasses import replace

import pytest

from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_handoff import validate_task_agent_handoff_v2
from src.mod_agent.task_agent_v2_tools import resolve_tool_arguments
from tests.unit.test_task_agent_v2_handoff import handoff_case as handoff_case  # noqa: F401
from tests.unit.test_task_agent_v2_handoff import save


@pytest.mark.parametrize('tamper', [None, 'effective', 'request_provenance'])
def test_handoff_consumes_saved_request_defaults_and_rejects_host_forgery(handoff_case, tamper):
    c = handoff_case
    execution, loop = deepcopy(c.execution), deepcopy(c.loop)
    turns = []
    finish_ref = None
    for row in execution['completed_tool_records']:
        if row['name'] not in {'candidate_check', 'finish'}:
            continue
        raw = {}
        response = read_bound_json(row['response_ref'])
        response['tool_calls'][0]['arguments'] = raw
        row['response_ref'] = save(c.root, row['name'] + '-default-response.json', response)
        row['arguments_ref'] = save(c.root, row['name'] + '-default-arguments.json', raw)
        current = {'origin': 'execution_host', 'task_contract_ref': c.task.contract_ref,
            'current_revision_ref': c.outcome.candidate_ref, 'binding_registry_ref': c.task.registry_ref,
            'current_check_ref': c.outcome.check_ref}
        row['request_ref'] = save(c.root, row['name'] + '-actual-request.json',
            {'messages': [{'role': 'user', 'content': {'current_task_state': current}}]})
        row['effective_arguments'], row['argument_resolution'] = resolve_tool_arguments(row['name'], raw, row['request_ref'])
        loop['call_records'][row['call_id']]['arguments_sha256'] = canonical_hash(raw)
        turns.append({'request_ref': row['request_ref'], 'response_ref': row['response_ref']})
        if row['name'] == 'finish':
            finish_ref = row['response_ref']
    execution['provider_turns'] = turns
    outcome = replace(c.outcome, finish_response_ref=finish_ref, loop_result_ref=save(c.root, 'default-loop.json', loop))
    execution.update(outcome.to_dict())
    if tamper == 'effective':
        execution['completed_tool_records'][-1]['effective_arguments']['candidate_ref'] = 'baseline'
    elif tamper == 'request_provenance':
        execution['provider_turns'] = turns[:-1]
    outcome = replace(outcome, execution_ref=save(c.root, 'default-execution.json', execution))
    if tamper is None:
        handoff = validate_task_agent_handoff_v2(outcome, **c.kwargs)
        assert handoff.candidate_ref == c.outcome.candidate_ref
    else:
        with pytest.raises(ValueError, match='RESOLUTION_CHANGED|REQUEST_NOT_SENT'):
            validate_task_agent_handoff_v2(outcome, **c.kwargs)
