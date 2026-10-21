import json
import pytest
from src.observability.evaluation.meta_evaluation import evaluate_evaluator
from src.observability.evaluation.unified import Binding,Criterion,EvaluationSession,save
from tests.unit.test_unified_evaluation import adapter


def reports(tmp_path,statuses):
    payload={'answer':'fixed answer'};binding=Binding('task','conversation','run',save(tmp_path/'subject.json',payload))
    return [EvaluationSession(tmp_path/'eval',criteria=[Criterion('a','quality','judge')],evaluators={'judge':adapter(s)},rubric_version='v1').evaluate(binding,payload) for s in statuses]

def item(report,expected='passed',**extra):
    return dict(case_id='case',family_id='family',expected_status=expected,report_ref=report['report_ref'],**extra)

def test_meta_reports_error_coverage_instead_of_claiming_zero_misjudgment(tmp_path):
    rs=reports(tmp_path,['passed','failed','evaluation_error','insufficient_evidence'])
    summary=evaluate_evaluator([item(r) for r in rs])
    assert summary['evaluation_count']==4 and summary['sample_count']==1
    assert summary['false_reject']['count']==1 and summary['evaluation_errors']==1
    assert summary['insufficient_evidence']==1 and summary['decisive_evaluations']==2

def test_meta_fixed_input_repetition_and_registered_invariance(tmp_path):
    rs=reports(tmp_path,['passed','failed'])
    summary=evaluate_evaluator([item(r,repeat_group='same-input',invariance_group='presentation') for r in rs])
    assert summary['repeated_fixed_inputs'][0]['verdict_changed'] is True
    assert summary['invariance_groups'][0]['distinct_verdicts']==2

def test_meta_rejects_counting_same_report_twice(tmp_path):
    r=reports(tmp_path,['passed'])[0]
    with pytest.raises(ValueError,match='DUPLICATED'):evaluate_evaluator([item(r),item(r)])

def test_meta_repeat_requires_same_rubric(tmp_path):
    rs=reports(tmp_path,['passed','passed'])
    raw=json.loads(open(rs[1]['report_ref']['path'],encoding='utf-8').read());raw['rubric']['version']='changed'
    rs[1]['report_ref']=save(tmp_path/'changed.json',raw)
    with pytest.raises(ValueError,match='REPEAT_INPUT_OR_RUBRIC_CHANGED'):
        evaluate_evaluator([item(r,repeat_group='same') for r in rs])


def test_meta_scores_selected_judge_instead_of_aggregate_report(tmp_path):
    payload = {'answer': 'fixed answer'}
    binding = Binding(
        'task',
        'conversation',
        'run',
        save(tmp_path / 'multi-subject.json', payload),
    )
    report = EvaluationSession(
        tmp_path / 'multi-eval',
        criteria=[
            Criterion('behavior', 'behavior', 'execution'),
            Criterion('grounding', 'grounding', 'judge'),
        ],
        evaluators={
            'execution': adapter('failed'),
            'judge': adapter('passed'),
        },
        rubric_version='v1',
    ).evaluate(binding, payload)

    assert report['status'] == 'failed'
    summary = evaluate_evaluator([
        item(report, expected='failed', criterion_id='grounding')
    ])

    assert summary['false_accept']['count'] == 1
    assert summary['cases'][0]['actual'] == 'passed'
    assert summary['cases'][0]['report_status'] == 'failed'


def test_meta_requires_target_for_multi_criterion_report(tmp_path):
    payload = {'answer': 'fixed answer'}
    binding = Binding(
        'task',
        'conversation',
        'run',
        save(tmp_path / 'ambiguous-subject.json', payload),
    )
    report = EvaluationSession(
        tmp_path / 'ambiguous-eval',
        criteria=[
            Criterion('grounding', 'grounding', 'judge'),
            Criterion('fulfillment', 'task_fulfillment', 'judge'),
        ],
        evaluators={'judge': adapter('passed')},
        rubric_version='v1',
    ).evaluate(binding, payload)

    with pytest.raises(ValueError, match='TARGET_CRITERION_REQUIRED'):
        evaluate_evaluator([item(report)])


def test_meta_cli_persists_criterion_bound_summary(tmp_path):
    from scripts.run_unified_evaluation import main

    report = reports(tmp_path, ['failed'])[0]
    manifest = tmp_path / 'meta-input.json'
    manifest.write_text(json.dumps({
        'schema_version': 'evaluation-meta-input.v1',
        'items': [item(report, expected='failed', criterion_id='a')],
    }), encoding='utf-8')
    output = tmp_path / 'meta-output'

    assert main([
        'meta',
        '--manifest', str(manifest),
        '--output-root', str(output),
    ]) == 0
    result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    assert result['evaluation_count'] == 1
    assert result['false_accept']['count'] == 0
    assert result['cases'][0]['criterion_id'] == 'a'
    assert result['selection_policy'].startswith('explicit criterion_id')


def test_meta_not_applicable_dimension_can_share_an_invoked_evaluator(tmp_path):
    payload = {'answer': 'fixed answer'}
    binding = Binding('task', 'conversation', 'run', save(tmp_path / 'candidate.json', payload))
    report = EvaluationSession(
        tmp_path / 'eval',
        criteria=[Criterion('support', 'grounding', 'judge'),
                  Criterion('citation', 'citation', 'judge', applicable=False)],
        evaluators={'judge': adapter('passed')},
        rubric_version='v1',
    ).evaluate(binding, payload)

    summary = evaluate_evaluator([
        item(report, expected='not_applicable', criterion_id='citation')
    ])

    assert summary['cases'][0]['actual'] == 'not_applicable'
    assert summary['evaluation_cost']['model_calls'] == {'known_subtotal': 0, 'complete': True}


def test_meta_multiple_dimensions_count_shared_judge_cost_once(tmp_path):
    payload = {'answer': 'fixed answer'}
    binding = Binding('task', 'conversation', 'run', save(tmp_path / 'candidate.json', payload))

    def judge(*args):
        result = adapter('passed')(*args)
        result['cost'].update(model_calls=1, input_tokens=30, output_tokens=10)
        return result

    report = EvaluationSession(
        tmp_path / 'eval',
        criteria=[Criterion('support', 'grounding', 'judge'),
                  Criterion('fulfillment', 'task_fulfillment', 'judge')],
        evaluators={'judge': judge},
        rubric_version='v1',
    ).evaluate(binding, payload)
    summary = evaluate_evaluator([
        item(report, criterion_id='support'), item(report, criterion_id='fulfillment')
    ])

    assert summary['evaluation_count'] == 2
    assert summary['report_count'] == summary['sample_count'] == 1
    assert summary['evaluation_cost']['model_calls'] == {'known_subtotal': 1, 'complete': True}
    assert summary['evaluation_cost']['input_tokens']['known_subtotal'] == 30


def test_meta_rejects_invocation_that_did_not_evaluate_selected_criterion(tmp_path):
    from src.observability.evaluation.unified import read_ref

    report = reports(tmp_path, ['passed'])[0]
    raw = json.loads(read_ref(report['report_ref']))
    invocation = json.loads(read_ref(raw['invocation_refs'][0]))
    invocation['criterion_ids'] = ['another']
    raw['invocation_refs'] = [save(tmp_path / 'wrong-invocation.json', invocation)]
    report['report_ref'] = save(tmp_path / 'wrong-report.json', raw)

    with pytest.raises(ValueError, match='TARGET_INVOCATION_INVALID'):
        evaluate_evaluator([item(report, criterion_id='a')])
