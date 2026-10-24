"""Validate one explicitly declared Verify diagnostic tree, not arbitrary new files.

为什么要有它：TestResults被构建器忽略不等于其中任意内容可信，期望副本仍须绑定原来源字节。
"""
from pathlib import Path

from src.mod_agent.domain.csharp_repository_backend import _relative, _within
from src.mod_agent.domain.full_project_compilation import _ref


def validate(source, before, *, expectation_root, diagnostic_root, compiler_output_roots=()):
    source = Path(source).resolve()
    expectation_root, diagnostic_root = _relative(expectation_root), _relative(diagnostic_root)
    _within(source, expectation_root)
    _within(source, diagnostic_root)
    compiler_output_roots = {_relative(path) for path in compiler_output_roots}
    original = {_relative(row['path']): row['sha256'] for row in before}
    for relative, digest in original.items():
        path = _within(source, relative)
        if not path.is_file() or _ref(path)['sha256'] != digest:
            raise ValueError('ORIGINAL_SOURCE_TEST_OR_EXPECTATION_CHANGED')
    expectations = {path: digest for path, digest in original.items()
                    if path.startswith(expectation_root + '/') and '.verified.' in path}
    allowed_copies, allowed_received = {}, set()
    for path, digest in expectations.items():
        relative = Path(path).relative_to(expectation_root).as_posix()
        copied = diagnostic_root + '/' + relative
        allowed_copies[copied] = {'sha256': digest, 'origin_ref': {'path': str(source / path), 'sha256': digest}}
        allowed_received.add(copied.replace('.verified.', '.DotNet8_0.received.'))
        allowed_received.add(copied.replace('.verified.', '.received.'))
    files, pending = [], [source]
    while pending:
        for path in pending.pop().iterdir():
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError('DIAGNOSTIC_LINK_OR_ESCAPE')
            if path.is_dir():
                if path.relative_to(source).as_posix() not in compiler_output_roots:
                    pending.append(path)
            elif path.is_file():
                files.append(path)
    copies, received, unknown = [], [], []
    for path in files:
        relative = path.relative_to(source).as_posix()
        if relative in original:
            continue
        elif relative in allowed_copies:
            value = allowed_copies[relative]
            if _ref(path)['sha256'] != value['sha256'] or _ref(value['origin_ref']['path']) != value['origin_ref']:
                raise ValueError('EXPECTED_COPY_CHANGED_OR_SOURCE_CHANGED')
            copies.append({'copy_ref': _ref(path), 'origin_ref': value['origin_ref']})
        elif relative in allowed_received:
            received.append(_ref(path))
        else:
            unknown.append(relative)
    if unknown:
        raise ValueError('UNDECLARED_NEW_FILES: ' + repr(unknown))
    return {'status': 'validated_declared_diagnostics', 'diagnostic_root': diagnostic_root,
        'declared_compiler_output_roots': sorted(compiler_output_roots),
        'expected_copies': copies, 'received_refs': received, 'unknown_new_files': [],
        'all_original_source_test_expectation_bytes_unchanged': True,
        'scope': 'only exact configured compiler output roots excluded; all other files compared with exact originals or enumerated diagnostic names'}
