"""Pure diagnostic file-policy checks, not real Verify relocation execution."""
import importlib
import shutil

import pytest

owner = importlib.import_module('artifacts.eval_driven_20261024.product-pilot.evaluation.declared_diagnostics')


@pytest.fixture
def tree(tmp_path):
    source = tmp_path / 'source'
    expected = source / 'test/Expectations/Root.verified.txt'
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b'fixed expected text\n')
    code = source / 'code.cs'
    code.write_bytes(b'// original source\n')
    before = [{'path': path.relative_to(source).as_posix(), 'sha256': owner._ref(path)['sha256']}
              for path in (expected, code)]
    copied = source / 'test/TestResults/verify/Expectations/Root.verified.txt'
    copied.parent.mkdir(parents=True)
    shutil.copyfile(expected, copied)
    received = copied.with_name('Root.DotNet8_0.received.txt')
    received.write_bytes(b'deliberately different observation; not a new expectation\n')
    return source, before, expected, copied, received


def check(tree):
    return owner.validate(tree[0], tree[1], expectation_root='test/Expectations',
                          diagnostic_root='test/TestResults/verify/Expectations')


def test_exact_expected_copy_and_declared_diagnostic_are_accepted(tree):
    value = check(tree)
    assert value['status'] == 'validated_declared_diagnostics'
    assert len(value['expected_copies']) == len(value['received_refs']) == 1
    assert value['expected_copies'][0]['origin_ref'] == owner._ref(tree[2])


@pytest.mark.parametrize('path', ['new.cs', 'test/TestResults/unexpected.txt',
                                  'test/TestResults/verify/Expectations/unregistered.received.txt',
                                  'bin/not-declared.cs'])
def test_arbitrary_new_files_not_silently_ignored(tree, path):
    target = tree[0] / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'unknown')
    with pytest.raises(ValueError, match='UNDECLARED_NEW_FILES'):
        check(tree)


def test_changed_expected_copy_rejected(tree):
    tree[3].write_bytes(b'changed expectation copy')
    with pytest.raises(ValueError, match='EXPECTED_COPY_CHANGED'):
        check(tree)


def test_changed_original_expectation_rejected(tree):
    tree[2].write_bytes(b'changed original')
    with pytest.raises(ValueError, match='ORIGINAL_SOURCE_TEST_OR_EXPECTATION_CHANGED'):
        check(tree)


def test_changed_original_source_rejected(tree):
    (tree[0] / 'code.cs').write_bytes(b'changed source')
    with pytest.raises(ValueError, match='ORIGINAL_SOURCE_TEST_OR_EXPECTATION_CHANGED'):
        check(tree)


@pytest.mark.parametrize('root', ['../outside', 'C:/outside'])
def test_diagnostic_policy_cannot_escape_source(tree, root):
    with pytest.raises(ValueError):
        owner.validate(tree[0], tree[1], expectation_root='test/Expectations', diagnostic_root=root)
