"""Read saved public failures only; no SDK, hidden materials or source changes.

为什么要有它：快照差异与需求适用性需要独立判断，参考控制的输出不是正确性真值。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import unified_diff
from pathlib import Path

from src.mod_agent.domain.csharp_repository_backend import _read_tests
from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import read_bound_json

ROOT = Path('C:/tmp/rfext-spectre-1531-evaluation/v2')
PUBLIC = Path('C:/tmp/rfext-spectre-1531-public-v2')
TEST = Path('test/Spectre.Console.Cli.Tests')
RESULT = {'path': str(ROOT / 'public-correct-result.json'),
          'sha256': '424663640506936d86023e903d24d594b559d33718874f01e2ba31e68e611552'}


def method_body(text, name):
    match = re.search(r'public\s+(?:async\s+)?[^\n]+\b' + re.escape(name) + r'\(', text)
    if match is None:
        raise ValueError('PUBLIC_METHOD_NOT_FOUND')
    opening = text.index('{', match.start())
    # These saved public methods have no braces in string literals. Keep the
    # assumption explicit rather than pretending to implement a C# parser.
    depth, end = 0, opening
    for end in range(opening, len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                break
    if depth:
        raise ValueError('PUBLIC_METHOD_BLOCK_UNBALANCED')
    return text[match.start():end + 1], text[:match.start()].count('\n') + 1, text[:end].count('\n') + 1


def normalized(lines):
    return [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]


def main():
    record = read_bound_json(RESULT)
    tests = _read_tests(Path(record['trx_ref']['path']), 'trx')
    failed = [row for row in tests if not row['passed']]
    assert len(failed) == record['failures'] == 34
    control_tests = ROOT / 'public-correct/source' / TEST
    public_help = PUBLIC / TEST / 'Unit/CommandAppTests.Help.cs'
    text = public_help.read_text(encoding='utf-8-sig')
    assert public_help.read_bytes() == (control_tests / 'Unit/CommandAppTests.Help.cs').read_bytes()
    rows, pairs = [], {}
    for row in failed:
        definition, = row['test_definitions']
        assert definition['class_name'].endswith('CommandAppTests+Help')
        directory = re.search(r'Directory: ([^\r\n]+)', row['details']).group(1).strip()
        received = re.search(r'^\s*-?\s*Received:\s*(.+\.received\.txt)$', row['details'], re.MULTILINE).group(1).strip()
        verified = re.search(r'^\s*Verified:\s*(.+\.verified\.txt)$', row['details'], re.MULTILINE).group(1).strip()
        received, verified = Path(directory) / received, Path(directory) / verified
        assert received.resolve().is_relative_to(control_tests.resolve())
        assert verified.resolve().is_relative_to(control_tests.resolve())
        baseline_verified = PUBLIC / TEST / verified.relative_to(control_tests)
        assert baseline_verified.read_bytes() == verified.read_bytes()
        expected = verified.read_text(encoding='utf-8-sig').splitlines()
        actual = received.read_text(encoding='utf-8-sig').splitlines()
        expected_version = [line for line in expected if '--version' in line]
        actual_version = [line for line in actual if '--version' in line]
        expected_other = [line for line in expected if '--version' not in line]
        actual_other = [line for line in actual if '--version' not in line]
        remainder_equal = normalized(expected_other) == normalized(actual_other)
        if len(expected_version) == 1 and not actual_version and remainder_equal:
            category = 'builtin_version_row_removed_only' if expected_other == actual_other else 'builtin_version_row_removed_with_padding'
        else:
            category = 'other_difference_requires_review'
        body, start, end = method_body(text, definition['method_name'])
        declarations = re.findall(r'(?:SetDefaultCommand|AddCommand|AddBranch)<([^>]+)>', body)
        selected = {'test_name': row['name'], 'test_method': definition['method_name'],
            'source_ref': _ref(public_help), 'source_line_start': start, 'source_line_end': end,
            'source_excerpt': body, 'declared_command_or_settings_types': declarations,
            'calls_SetApplicationVersion': 'SetApplicationVersion(' in body,
            'assigns_ApplicationVersion': bool(re.search(r'ApplicationVersion\s*=', body)),
            'references_custom_version_fixture': bool(re.search(r'\bVersion(?:Command|Settings)\b', body)),
            'run_expression': re.search(r'var result = fixture\.Run\((.*?)\);', body, re.DOTALL).group(1),
            'category': category, 'verified_ref': _ref(verified), 'received_ref': _ref(received),
            'verified_matches_original_public_bytes': True,
            'removed_version_rows': expected_version, 'received_version_rows': actual_version,
            'other_lines_equal_after_horizontal_whitespace_normalization': remainder_equal,
            'full_diff': '\n'.join(unified_diff(expected, actual, fromfile=verified.name, tofile=received.name)),
            'residual_nonversion_diff': '\n'.join(unified_diff(normalized(expected_other), normalized(actual_other))),
            'test_report_ref': row['report_ref'], 'test_result_index': row['result_index']}
        rows.append(selected)
        pairs[(str(verified), str(received))] = {'verified_ref': selected['verified_ref'],
            'received_ref': selected['received_ref'], 'category': category}
    support_paths = [PUBLIC / 'src/Spectre.Console.Testing/Cli/CommandAppTester.cs',
        PUBLIC / 'src/Spectre.Console.Cli/ConfiguratorExtensions.cs',
        PUBLIC / TEST / 'Data/Help/CustomHelpProvider.cs',
        PUBLIC / TEST / 'Data/Help/RenderMarkupHelpProvider.cs',
        PUBLIC / TEST / 'Data/Help/RedirectHelpProvider.cs',
        PUBLIC / TEST / 'Data/Settings/VersionSettings.cs']
    output = {'schema_version': 'saved_public_failure_analysis_v1', 'source_result_ref': RESULT,
        'public_requirement_ref': _ref('C:/tmp/rfext-spectre-1531-task/solver-input/problem_statement.txt'),
        'trx_ref': record['trx_ref'], 'original_terminal_status': record['status'],
        'original_integrity_error_refs': record['integrity_error_refs'],
        'scope': 'only saved public tests, original public source and public verified/received outputs',
        'model_calls': 0, 'build_calls': 0, 'originals_modified': False,
        'reference_production_source_or_hidden_materials_read': False,
        'failed_test_instances': len(rows), 'unique_test_methods': len({row['test_method'] for row in rows}),
        'unique_snapshot_pairs': len(pairs), 'categories': dict(Counter(row['category'] for row in rows)),
        'all_remainders_equal_after_horizontal_whitespace_normalization': all(
            row['other_lines_equal_after_horizontal_whitespace_normalization'] for row in rows),
        'explicit_application_version_cases': sum(row['calls_SetApplicationVersion'] or row['assigns_ApplicationVersion'] for row in rows),
        'custom_version_fixture_cases': sum(row['references_custom_version_fixture'] for row in rows),
        'whole_public_help_file_mentions_SetApplicationVersion': 'SetApplicationVersion' in text,
        'whole_public_help_file_mentions_version_fixture': bool(re.search(r'\bVersion(?:Command|Settings)\b', text)),
        'supporting_public_source_refs': [_ref(path) for path in support_paths],
        'snapshots': list(pairs.values()), 'tests': rows,
        'judgment': {
            'fact': 'all observed failures concern non-custom-version help snapshots; old version row disappears',
            'contract_inference': 'the public issue requests removal of a conflict with a user-defined version option; it does not authorize changing all non-conflicting default help',
            'decision_supported_by_public_evidence': 'do not update or exclude the 34 snapshots solely to accept the reference control; retain applicable baseline expectations until an explicit compatibility decision',
            'unknown': 'the intended broader upstream API policy cannot be established from this issue alone',
            'not_claimed': 'no unrelated semantic differences in these snapshot remainders does not prove absence of regressions outside these observations'}}
    reference = _save(Path(__file__).with_name('public-failure-analysis.json'), output)
    print(json.dumps({'analysis_ref': reference, **{key: output[key] for key in (
        'failed_test_instances', 'unique_test_methods', 'unique_snapshot_pairs', 'categories',
        'all_remainders_equal_after_horizontal_whitespace_normalization',
        'explicit_application_version_cases', 'custom_version_fixture_cases')}}), flush=True)


def addendum():
    output = Path(__file__).parent
    analysis = read_bound_json(_ref(output / 'public-failure-analysis.json'))
    test_root = PUBLIC / TEST
    declarations = []
    for path in (test_root / 'Data').rglob('*.cs'):
        text = path.read_text(encoding='utf-8-sig')
        for match in re.finditer(r'CommandOption\("([^"]+)"', text):
            aliases = [re.split(r'[ =<]', alias.strip(), maxsplit=1)[0] for alias in match.group(1).split('|')]
            if set(aliases).intersection({'-v', '--version'}):
                declarations.append({'source_ref': _ref(path), 'line': text[:match.start()].count('\n') + 1,
                                     'template': match.group(1)})
    result = read_bound_json(analysis['source_result_ref'])
    integrity = read_bound_json(result['integrity_error_refs'][0])
    source = Path(integrity['context']['source_root'])
    before = {row['path'] for row in integrity['context']['source_inputs']}
    ignored = {'.git', '.idea', 'bin', 'obj', '.vs', 'TestResults'}
    # Enumerate paths only; do not inspect reference production implementations.
    after = {path.relative_to(source).as_posix() for path in source.rglob('*')
             if path.is_file() and not ignored.intersection(path.relative_to(source).parts)}
    extra = sorted(after - before)
    localized = [row for row in analysis['tests'] if 'Localised' in row['test_method']]
    value = {'schema_version': 'saved_public_failure_analysis_addendum_v1',
        'analysis_ref': _ref(output / 'public-failure-analysis.json'), 'model_calls': 0, 'build_calls': 0,
        'source_scope': 'original public test Data and original public docs; no reference implementation or hidden files',
        'exact_custom_version_option_declarations_in_original_public_Data': declarations,
        'failing_Help_file_references_that_fixture': False,
        'source_setup_facts': {'explicit_version_setters_in_Help': 0,
            'custom_version_settings_or_command_mentions_in_Help': 0,
            'CommandAppTester_creates_fresh_CommandApp_per_Run': True,
            'CommandAppTester_has_no_application_version_setter': True,
            'custom_help_version_string_is_footer_only': True},
        'documentation': {'source_ref': _ref(PUBLIC / 'src/Spectre.Console.Cli/ConfiguratorExtensions.cs'),
            'line': 85, 'text': 'Overrides the auto-detected version of the application.'},
        'unique_pair_categories': dict(Counter(row['category'] for row in analysis['snapshots'])),
        'localized_test_instances': len(localized),
        'localized_unique_snapshot_pairs': len({(row['verified_ref']['path'], row['received_ref']['path'])
                                               for row in localized}),
        'new_file_paths_since_recorded_public_check_inputs': extra, 'new_files': len(extra),
        'all_new_files_are_received_snapshots': all('.received.' in path for path in extra),
        'other_missing_paths': sorted(before - after),
        'original_terminal_still': result['status'], 'integrity_status_still': integrity['status'],
        'judgment': 'public issue supports fixing custom-version conflicts, not blanket removal of default version help; retain baseline expectations under the current preservation contract',
        'limits': 'no other text differences in these 34 comparisons does not prove absence of unrelated regressions elsewhere'}
    reference = _save(output / 'public-failure-analysis-addendum.json', value)
    print(json.dumps({'addendum_ref': reference, 'new_files': len(extra),
        'all_new_received': value['all_new_files_are_received_snapshots'],
        'localized_instances': len(localized), 'localized_pairs': value['localized_unique_snapshot_pairs'],
        'unique_pair_categories': value['unique_pair_categories'], 'missing_paths': len(value['other_missing_paths']),
        'custom_version_declarations': declarations}), flush=True)


if __name__ == '__main__':
    addendum() if sys.argv[1:] == ['--addendum'] else main()
