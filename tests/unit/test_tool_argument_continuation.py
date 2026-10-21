"""Real adapter/Graph/CLI callers with scripted providers; no live model or Build.

Scope: the task-wide argument-correction bit and existing numeric budgets only.
为什么要有它：换Plan并丢弃native历史不能重发纠错机会，也不能继承旧binding权限或添加预算字段。
"""
import json
from pathlib import Path

import httpx
import pytest

from src.libs.llm.agent_turn import AgentToolCall, AgentTurnResponse, AgentUsage
from src.mod_agent.fresh_planning_continuation import run_fresh_patch_continuation
from src.mod_agent.local_entry import make_exact_no_write_dry_run_factory
from src.mod_agent.proposal.patch_agent_loop import PATCH_SOURCE_INSPECTION_TOOL_NAME, PatchAgentLoopPolicy
from tests.unit.test_fresh_planning_continuation import _inputs, _patch_request, _PatchAgentTurnClient, _source_hashes
from tests.unit.test_patch_behavior_feedback import _adapter, _BehaviorOracle
from tests.unit.test_patch_repair_cli import _message_jsons
from tests.unit.test_patch_upstream_cli import _case, _run, _UpstreamTransport

EXHAUSTED = 'PATCH_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED'
BUDGET_KEYS = {'consumed_model_turns', 'consumed_tool_rounds', 'consumed_observation_bytes'}


class _ArgumentThenPatch(_PatchAgentTurnClient):
    def complete_turn(self, request):
        turn = len(self.requests) + 1
        if turn > 2:
            return super().complete_turn(request)
        self.requests.append(request)
        arguments = {} if turn == 1 else {'binding_id': 'edit_binding_001'}
        return AgentTurnResponse(provider=self.provider, model=self.model, content=None,
            tool_calls=(AgentToolCall(call_id=f'argument-caller-{turn}', name=PATCH_SOURCE_INSPECTION_TOOL_NAME,
                                     arguments=arguments),), stop_reason='tool_calls',
            usage=AgentUsage(input_tokens=20, output_tokens=5),
            provider_state={'content':None, 'tool_calls':[{'id':f'argument-caller-{turn}'}]})


@pytest.mark.parametrize('inherited', [False, True], ids=['normal-correction', 'inherited-monotonic-used'])
def test_normal_adapter_entry_carries_correction_bit_without_expanding_budget(tmp_path, inherited):
    inputs = _inputs(tmp_path)
    request = _patch_request(inputs)
    before = _source_hashes(inputs.workspace)
    client = _ArgumentThenPatch()
    adapter = _adapter(inputs, client, _BehaviorOracle(tmp_path/'oracle'))
    adapter.enable_recovery_loop = True
    adapter.patch_agent_loop_policy = PatchAgentLoopPolicy(finalize_after_first_observation=False)
    if inherited:
        adapter.bind_tool_argument_correction_used(True)
        adapter.bind_tool_argument_correction_used(False)
    assert set(adapter._prior_worker_budget) == BUDGET_KEYS
    result = run_fresh_patch_continuation(request, patch_change_factory=adapter,
        dry_run_factory=make_exact_no_write_dry_run_factory(inputs.workspace), max_model_calls=5)
    receipt = json.loads(Path(result.attempt_receipt['path']).read_bytes())['factory_receipt']
    assert receipt['tool_argument_correction_used'] is True
    assert receipt['patch_agent_loop']['tool_argument_correction_used'] is True
    assert set(adapter._prior_worker_budget) == BUDGET_KEYS
    assert _source_hashes(inputs.workspace) == before
    assert any(tool.name == PATCH_SOURCE_INSPECTION_TOOL_NAME for tool in client.requests[0].tools)
    if inherited:
        assert len(client.requests) == 1
        assert receipt['patch_agent_loop']['termination']['code'] == EXHAUSTED
        assert result.patch_proposal_artifact is None
    else:
        assert result.status == 'terminal', result
        assert len(client.requests) == 3
        assert receipt['patch_agent_loop']['tool_rounds_used'] == 1
        assert client.requests[1].tool_results[0].output['tool_argument_correction']['used'] is True


class _ArgumentReplanTransport(_UpstreamTransport):
    def __init__(self, root, case):
        super().__init__(root, case)
        self.first_bad = False
        self.replanned_bad = False
        self.replanned_payload = None

    def __call__(self, request):
        if request.method != 'POST' or request.url.path != '/api/chat':
            return super().__call__(request)
        payload = json.loads(request.content)
        values = [value for _,value in _message_jsons(payload)]
        planning = any('allowed_edit_evidence_ids' in value for value in values)
        first = not self.first_bad
        second = self.planning_sent and not planning and not self.replanned_bad
        if not (first or second):
            return super().__call__(request)
        self.first_bad = True
        if second:
            self.replanned_bad = True
            self.replanned_payload = payload
        if first:
            assert any(tool['function']['name'] == PATCH_SOURCE_INSPECTION_TOOL_NAME for tool in payload.get('tools', []))
        self.requests.append((request.method, str(request.url)))
        self.chat_requests.append((request.content, payload))
        self.stage_roles.append('patch')
        return self._response(request, {'role':'assistant','content':'','tool_calls':[{'function':{
            'name':PATCH_SOURCE_INSPECTION_TOOL_NAME, 'arguments':{}}}]})


def test_public_cli_new_plan_keeps_used_bit_without_history_and_preserves_final_turn_budget(tmp_path, monkeypatch, capsys):
    case = _case(tmp_path, monkeypatch)
    # Keep the legal recovery evaluator and its actual reserved turns. The new Plan
    # receives only a submission turn; independently assert the inherited bit instead
    # of attributing the tool-unavailable rejection to that bit.
    # 为什么要有它：正常caller的真实预算不能为测试让路，也不能把另一预算门的拒绝伪报为纠错耗尽。
    before = _source_hashes(case.inputs.workspace)
    handler = _ArgumentReplanTransport(tmp_path/'wire', case)
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', lambda _self,request:transport.handle_request(request))
    def forbidden(*args, **kwargs):
        raise AssertionError('No real network or provider is permitted')
    monkeypatch.setattr('socket.socket.connect', forbidden)
    monkeypatch.setattr('socket.socket.connect_ex', forbidden)
    code,result = _run(case,capsys)
    assert code != 0, result
    assert handler.stage_roles == ['patch','patch','planning','patch'], {key:result.get(key) for key in ('status','reason','first_failure')}
    journal = json.loads(Path(result['attempt_receipt']['path']).read_bytes())
    first,second = journal['patch_execution_segments']
    assert first['selected_artifacts']['implementation_plan'] != second['selected_artifacts']['implementation_plan']
    for segment in (first,second):
        assert segment['factory_receipt']['tool_argument_correction_used'] is True
    second_loop = second['factory_receipt']['patch_agent_loop']
    assert second_loop['termination']['code'] == 'PATCH_AGENT_TOOL_FORBIDDEN'
    assert second_loop['model_turns_used'] == 1 and second_loop['tool_rounds_used'] == 0
    assert not any(message['role']=='tool' for message in handler.replanned_payload['messages'])
    state = next(value for _,value in _message_jsons(handler.replanned_payload)
                 if value.get('message_type')=='patch_agent_loop_state')
    assert state['tool_argument_correction_used'] is True and state['prior_observations']==[]
    assert not any(tool['function']['name']==PATCH_SOURCE_INSPECTION_TOOL_NAME for tool in handler.replanned_payload.get('tools', []))
    assert result['execution_counts']['model'] == 4
    assert result['execution_counts']['planning'] == 1
    assert result['execution_counts']['apply'] == result['execution_counts']['build'] == result['execution_counts']['runtime'] == 0
    assert _source_hashes(case.inputs.workspace) == before
