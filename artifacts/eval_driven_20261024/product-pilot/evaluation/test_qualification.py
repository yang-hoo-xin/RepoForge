"""Pure preparation checks; no SDK, source archive, model or hidden data use."""
import importlib
import json
from types import SimpleNamespace

import pytest

owner = importlib.import_module('artifacts.eval_driven_20261024.product-pilot.evaluation.qualification')


@pytest.fixture
def configured(tmp_path):
    sdk = tmp_path / 'dotnet.exe'
    sdk.write_bytes(b'not executable; identity fixture only')
    public = tmp_path / 'Public.cs'
    public.write_text('// public fixture', encoding='utf-8')
    task = owner._save(tmp_path / 'private-task.json', {'instance_id': 'fixture', 'base_commit': 'base'})
    value = {'schema_version': 'repository_control_configuration_v1', 'source_ready': False,
        'source_root': str(tmp_path / 'not-downloaded'), 'output_root': str(tmp_path / 'evaluation'),
        'edit_prefixes': ['src/Spectre.Console.Cli/', 'src/Spectre.Console/'],
        'expected_artifacts': ['Cli.dll', 'Console.dll'], 'public_test_ref': owner._ref(public),
        'dotnet_ref': owner._ref(sdk), 'task_ref': task, 'profile_timeout_seconds': 300}
    return value, owner._save(tmp_path / 'config.json', value)


def test_static_validation_does_not_require_source_or_execute(configured):
    value, reference = configured
    assert owner.load(reference) == value


def test_prepare_refuses_before_explicit_source_ready(configured):
    value, reference = configured
    with pytest.raises(ValueError, match='WAIT_FOR_EXPLICIT_BLOB_READY'):
        owner.prepare(value, reference)
    assert not owner.Path(value['output_root']).exists()


@pytest.mark.parametrize('scope', [[], ['src/Spectre.Console.Cli/']])
def test_cli_only_or_empty_scope_rejected(configured, scope):
    value, reference = configured
    value['edit_prefixes'] = scope
    reference = owner._save(owner.Path(reference['path']).with_name('scope.json'), value)
    with pytest.raises(ValueError, match='EDIT_SCOPE'):
        owner.load(reference)


def test_only_one_artifact_cannot_claim_two_project_scope(configured):
    value, reference = configured
    value['expected_artifacts'] = ['Cli.dll']
    reference = owner._save(owner.Path(reference['path']).with_name('artifact.json'), value)
    with pytest.raises(ValueError, match='ARTIFACT_SCOPE'):
        owner.load(reference)


def test_output_cannot_overlap_source(configured):
    value, reference = configured
    value['output_root'] = str(owner.Path(value['source_root']) / 'inside')
    reference = owner._save(owner.Path(reference['path']).with_name('overlap.json'), value)
    with pytest.raises(ValueError, match='OUTPUT_OVERLAPS_SOURCE'):
        owner.load(reference)


def test_modified_configuration_rejected(configured):
    value, reference = configured
    value['source_ready'] = True
    owner.Path(reference['path']).write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        owner.load(reference)


def test_public_input_hash_checked(configured):
    value, reference = configured
    owner.Path(value['public_test_ref']['path']).write_text('// modified', encoding='utf-8')
    with pytest.raises(ValueError, match='PUBLIC_TEST_CHANGED'):
        owner.load(reference)


def test_private_replacement_cannot_escape_source(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    with pytest.raises(ValueError):
        owner.apply_private_control(source, {'replacements': [{'path': '../outside.cs'}]})


def test_private_replacement_requires_exact_preimage(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    target = source / 'file.cs'
    target.write_text('actual', encoding='utf-8')
    with pytest.raises(ValueError, match='PRIVATE_VARIANT_PREIMAGE_CHANGED'):
        owner.apply_private_control(source, {'replacements': [
            {'path': 'file.cs', 'before': 'missing', 'after': 'never', 'count': 1}]})
    assert target.read_text(encoding='utf-8') == 'actual'


def test_public_root_uses_its_declared_parent_workspace(tmp_path, monkeypatch):
    root, public = tmp_path / 'evaluation', tmp_path / 'public-source'
    calls = []
    monkeypatch.setattr(owner, '_ref', lambda path: {'path': str(path), 'sha256': 'fixture'})
    monkeypatch.setattr(owner, 'read_bound_json', lambda ref: {'profile_ref': {'path': 'profile', 'sha256': 'fixture'}})
    monkeypatch.setattr(owner.backend, 'CSharpRepositoryProfile', lambda ref: SimpleNamespace(
        reference=ref, bind=lambda source: {'fixed': 'fixture'}))

    def constructor(**kwargs):
        calls.append(kwargs)
        assert kwargs['source_root'].is_relative_to(kwargs['workspace_root'])
        return SimpleNamespace(run_build_test=lambda *args: {'status': 'controlled_no_execution', 'report_ref': {}})

    monkeypatch.setattr(owner.backend, 'CSharpRepositoryBackend', constructor)
    result = owner.run({'output_root': str(root), 'public_source_root': str(public)}, 'public-baseline', 'fixture')
    assert result['path'].endswith('public-baseline-result-fixture.json')
    assert calls[0]['workspace_root'] == public.parent
    assert calls[0]['source_root'] == public
