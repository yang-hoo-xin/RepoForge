"""Registry history guards; fixtures below are controls, not business holdouts."""
from dataclasses import asdict, replace
import json

import pytest

from src.observability.evaluation.sample_registry import Sample, admit_samples
from src.observability.evaluation.unified import file_ref, save


def sample(**kwargs):
    return replace(Sample('dataset', 'sample', 'family', 'constructed', 'regression', 'ai', True), **kwargs)


def registry(tmp_path, name, values, *, previous=()):
    evidence = tmp_path / (name + '-provenance.json')
    evidence.write_text(json.dumps({'scope': 'constructed registry control'}), encoding='utf-8')
    ref = file_ref(evidence)
    return save(tmp_path / (name + '.json'), {
        'schema_version': 'evaluation-sample-registry.v1',
        'records': [{'sample': asdict(value), 'origin_id': origin, 'origin_ref': ref,
                     'label_refs': [ref], 'exposure_refs': [ref] if value.tuning_exposed else []}
                    for value, origin in values],
        'prior_registry_refs': list(previous),
    })


def test_legacy_development_remains_explicitly_unqualified():
    result = admit_samples([sample()])
    assert result['registration_status'] == 'unregistered_development_metadata'
    assert result['independence_verified'] is False


def test_holdout_without_registry_rejected():
    with pytest.raises(ValueError, match='HOLDOUT_REGISTRY_REQUIRED'):
        admit_samples([sample(partition='holdout', tuning_exposed=False)])


def test_renamed_dataset_sample_and_family_cannot_hide_known_origin(tmp_path):
    old = registry(tmp_path, 'old', [(sample(), 'original-issue')])
    new = sample(dataset_id='new-dataset', sample_id='paraphrase', family_id='renamed',
                 partition='holdout', tuning_exposed=False)
    current = registry(tmp_path, 'current', [(new, 'original-issue')], previous=[old])
    with pytest.raises(ValueError, match='HOLDOUT_PREVIOUSLY_EXPOSED'):
        admit_samples([new], registry_refs=[current])


def test_same_key_historical_exposure_cannot_be_reset(tmp_path):
    old = registry(tmp_path, 'old', [(sample(), 'original-issue')])
    new = sample(partition='holdout', tuning_exposed=False)
    current = registry(tmp_path, 'current', [(new, 'original-issue')], previous=[old])
    with pytest.raises(ValueError, match='HOLDOUT_PREVIOUSLY_EXPOSED'):
        admit_samples([new], registry_refs=[current])


def test_unknown_renamed_sample_is_not_registered(tmp_path):
    known = sample(partition='holdout', tuning_exposed=False)
    ref = registry(tmp_path, 'known', [(known, 'one-issue')])
    with pytest.raises(ValueError, match='SAMPLE_NOT_REGISTERED'):
        admit_samples([replace(known, sample_id='unknown')], registry_refs=[ref])


def test_same_sample_name_in_unrelated_datasets_is_not_merged(tmp_path):
    independent = sample(dataset_id='other', partition='holdout', tuning_exposed=False)
    ref = registry(tmp_path, 'mixed', [(sample(), 'old-issue'), (independent, 'other-issue')])
    result = admit_samples([independent], registry_refs=[ref])
    assert result['registration_status'] == 'registered'
    assert result['declared_origin_count'] == 1
    assert result['independence_verified'] is False
    assert result['semantic_labels_verified'] is False


def test_consumed_holdout_can_be_retained_as_exposed_regression(tmp_path):
    holdout = sample(partition='holdout', tuning_exposed=False)
    old = registry(tmp_path, 'old', [(holdout, 'one-issue')])
    current = registry(tmp_path, 'current', [(sample(), 'one-issue')], previous=[old])
    assert admit_samples([sample()], registry_refs=[current])['registration_status'] == 'registered'


@pytest.mark.parametrize('changed', ['registry', 'provenance'])
def test_changed_bound_bytes_rejected(tmp_path, changed):
    ref = registry(tmp_path, 'known', [(sample(), 'one-issue')])
    path = tmp_path / ('known.json' if changed == 'registry' else 'known-provenance.json')
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='EVIDENCE_CHANGED'):
        admit_samples([sample()], registry_refs=[ref])


def test_missing_registered_label_basis_rejected(tmp_path):
    ref = registry(tmp_path, 'known', [(sample(), 'one-issue')])
    value = json.loads((tmp_path / 'known.json').read_bytes())
    value['records'][0]['label_refs'] = []
    ref = save(tmp_path / 'missing-label.json', value)
    with pytest.raises(ValueError, match='LABEL_REFERENCE_REQUIRED'):
        admit_samples([sample()], registry_refs=[ref])


def test_same_family_different_origin_cannot_split_inside_dataset(tmp_path):
    holdout = sample(sample_id='variant', partition='holdout', tuning_exposed=False)
    ref = registry(tmp_path, 'mixed', [(sample(), 'one'), (holdout, 'two')])
    with pytest.raises(ValueError, match='HOLDOUT_PREVIOUSLY_EXPOSED'):
        admit_samples([holdout], registry_refs=[ref])


@pytest.mark.parametrize('entry', ['code', 'rag', 'session'])
def test_normal_entry_rejects_unregistered_holdout_before_execution(tmp_path, entry):
    from src.observability.evaluation.unified import EvaluationSession
    from src.observability.evaluation.unified_hooks import CodeEvaluationHook, RagEvaluationHook
    holdout = sample(partition='holdout', tuning_exposed=False)
    with pytest.raises(ValueError, match='HOLDOUT_REGISTRY_REQUIRED'):
        if entry == 'code':
            CodeEvaluationHook(sample=holdout, runtime_required=False)
        elif entry == 'rag':
            RagEvaluationHook(tmp_path, samples={0: holdout})
        else:
            EvaluationSession(tmp_path, criteria=[], evaluators={}, rubric_version='control', sample=holdout)


def bound_registry(tmp_path, value, input_binding):
    registry(tmp_path, 'known', [(value, 'controlled-independent-origin')])
    document = json.loads((tmp_path / 'known.json').read_bytes())
    document['records'][0]['input_binding'] = input_binding
    return save(tmp_path / 'input-bound-registry.json', document)


def rag_inputs(tmp_path):
    from src.observability.evaluation.eval_runner import load_test_set
    a = save(tmp_path / 'a.json', {'test_cases': [{'query': 'registered question A', 'expected_behavior': 'answer'}]})
    b = save(tmp_path / 'b.json', {'test_cases': [{'query': 'exposed question B', 'expected_behavior': 'answer'}]})
    binding = {'kind': 'rag', 'test_set_ref': a, 'case_index': 0}
    return a, b, binding, load_test_set


def test_rag_loader_cannot_label_other_test_set_as_registered_holdout(tmp_path):
    from src.libs.evaluator.base_evaluator import NoneEvaluator
    from src.observability.evaluation.eval_runner import EvalRunner
    from src.observability.evaluation.unified_hooks import RagEvaluationHook
    holdout = sample(partition='holdout', tuning_exposed=False)
    a, b, binding, _ = rag_inputs(tmp_path)
    ref = bound_registry(tmp_path, holdout, binding)
    hook = RagEvaluationHook(tmp_path / 'evaluations', samples={0: holdout}, registry_refs=[ref])
    class NeverQuery:
        calls = 0
        def query(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError('No business/provider call allowed')
    service = NeverQuery()
    runner = EvalRunner(evaluator=NoneEvaluator(), query_service=service, unified_hook=hook)
    with pytest.raises(ValueError, match='REGISTERED_INPUT_MISMATCH'):
        runner.run(b['path'], run_metadata={'run_id': 'control', 'evaluation_test_set_ref': a})
    assert service.calls == 0


def test_correct_rag_source_selector_is_admitted_without_running_model(tmp_path):
    from src.observability.evaluation.unified_hooks import RagEvaluationHook
    holdout = sample(partition='holdout', tuning_exposed=False)
    a, _, binding, loader = rag_inputs(tmp_path)
    ref = bound_registry(tmp_path, holdout, binding)
    hook = RagEvaluationHook(tmp_path / 'evaluations', samples={0: holdout}, registry_refs=[ref])
    hook.validate_run(loader(a['path']), {'run_id': 'control', 'evaluation_test_set_ref': a})
    assert hook.input_bindings[0] == binding
    assert hook.sample_admission['independence_verified'] is False


def test_direct_session_cannot_assign_registered_identity_to_other_query(tmp_path):
    from src.observability.evaluation.unified import Binding, EvaluationSession
    holdout = sample(partition='holdout', tuning_exposed=False)
    _, _, binding, _ = rag_inputs(tmp_path)
    ref = bound_registry(tmp_path, holdout, binding)
    session = EvaluationSession(tmp_path / 'eval', criteria=[], evaluators={}, rubric_version='control',
                                sample=holdout, registry_refs=[ref], input_binding=binding)
    payload = {'query': 'another question', 'evaluation_input_binding': binding}
    candidate = save(tmp_path / 'candidate.json', payload)
    with pytest.raises(ValueError, match='INPUT_QUERY_MISMATCH'):
        session.evaluate(Binding('task', 'conversation', 'run', candidate), payload)


def test_code_cli_rejects_wrong_description_before_product_entry(tmp_path, monkeypatch):
    from scripts import dsh_request_routing, run_unified_evaluation
    holdout = sample(partition='holdout', tuning_exposed=False)
    original = save(tmp_path / 'original-request.json', {'request': 'original'})
    wrong = save(tmp_path / 'wrong-request.json', {'request': 'different'})
    profile = save(tmp_path / 'profile.json', {'scope': 'controlled-unused-profile'})
    ref = bound_registry(tmp_path, holdout, {'kind': 'code', 'description_ref': original,
                                            'execution_config_ref': profile})
    metadata = save(tmp_path / 'metadata.json', {'schema_version': 'prelaunch-evaluation-run.v1',
        'run_id': 'control', 'sample': asdict(holdout), 'allowed_edit_paths': ['one.py'], 'registry_refs': [ref]})
    called = []
    monkeypatch.setattr(dsh_request_routing, 'run_natural_request', lambda **kwargs: called.append(kwargs))
    with pytest.raises(ValueError, match='REGISTERED_INPUT_MISMATCH'):
        run_unified_evaluation.main(['code', '--metadata', metadata['path'], '--output-root', str(tmp_path / 'out'),
            '--execution-config', profile['path'], '--description-file', wrong['path'], '--no-runtime-required'])
    assert called == []


def test_registry_inventory_does_not_open_execution_inputs(tmp_path, monkeypatch):
    from src.observability.evaluation import unified
    holdout = sample(partition='holdout', tuning_exposed=False)
    private = {'path': str(tmp_path / 'sealed-not-opened.json'), 'sha256': 'c' * 64}
    ref = bound_registry(tmp_path, holdout, {'kind': 'rag', 'test_set_ref': private, 'case_index': 0})
    original = unified.read_ref
    def public_refs_only(reference):
        assert reference != private, 'Metadata admission must not read sealed task bytes'
        return original(reference)
    monkeypatch.setattr(unified, 'read_ref', public_refs_only)
    result = admit_samples([holdout], registry_refs=[ref])
    assert result['registration_status'] == 'registered'
    assert result['independence_verified'] is False


@pytest.mark.parametrize('query', [None, '', '   '])
def test_direct_bound_rag_session_requires_actual_query(tmp_path, query):
    from src.observability.evaluation.unified import Binding, EvaluationSession
    holdout = sample(partition='holdout', tuning_exposed=False)
    _, _, binding, _ = rag_inputs(tmp_path)
    ref = bound_registry(tmp_path, holdout, binding)
    session = EvaluationSession(tmp_path / 'eval', criteria=[], evaluators={}, rubric_version='control',
                                sample=holdout, registry_refs=[ref], input_binding=binding)
    payload = {'query': query, 'evaluation_input_binding': binding}
    candidate = save(tmp_path / 'candidate.json', payload)
    with pytest.raises(ValueError, match='INPUT_QUERY_REQUIRED'):
        session.evaluate(Binding('task', 'conversation', 'run', candidate), payload)
