"""Resolve explicitly selected assisted reviews without rerunning or replacing originals."""
from collections import Counter
from copy import deepcopy
import json
from .unified import Binding, Criterion, STATUSES, canonical, decision, read_ref, save

SEMANTIC_DIMENSIONS = frozenset({'grounding', 'task_fulfillment', 'evidence_relevance'})
REVIEWER = 'current_codex_conversation_assistant'


def checked_report(reference):
    report = json.loads(read_ref(reference))
    if report.get('schema_version') != 'unified-evaluation.v1':
        raise ValueError('RESOLUTION_REPORT_SCHEMA_INVALID')
    Binding(**report['binding'])
    criteria = [Criterion(**c) for c in report['rubric']['criteria']]
    rows = report['criteria']
    if len(rows) != len(criteria) or len({c.id for c in criteria}) != len(criteria):
        raise ValueError('RESOLUTION_CRITERIA_INVALID')
    for c, row in zip(criteria, rows):
        if (row.get('criterion_id') != c.id or row.get('binding') != report['binding']
                or row.get('status') not in STATUSES
                or any(row.get(k) != getattr(c, k) for k in ('dimension', 'evaluator', 'role', 'required', 'applicable'))):
            raise ValueError('RESOLUTION_ROW_BINDING_INVALID')
        expected_use = 'acceptance' if c.method == 'execution' else 'trial_only'
        if row.get('decision_use') != expected_use:
            raise ValueError('RESOLUTION_ORIGINAL_DECISION_USE_INVALID')
        if row['status'] in {'passed', 'failed'} and not row.get('evidence_refs'):
            raise ValueError('RESOLUTION_DECISION_EVIDENCE_MISSING')
        for ref in row['evidence_refs']:
            read_ref(ref)
    for ref in report['invocation_refs']:
        if json.loads(read_ref(ref))['binding'] != report['binding']:
            raise ValueError('RESOLUTION_INVOCATION_BINDING_INVALID')
    if decision(criteria, rows) != report['status']:
        raise ValueError('RESOLUTION_ORIGINAL_DECISION_INVALID')
    acceptance_rows = [dict(row, status='insufficient_evidence')
                       if criterion.method == 'trial_judge' and row['status'] in {'passed', 'failed'}
                       else row for criterion, row in zip(criteria, rows)]
    # 范围：重读统一报告。为什么要有它：哈希验证只证明字节未变，验收摘要仍须与分项结果一致，未复核裁判结果不能借摘要字段变成通过。
    if decision(criteria, acceptance_rows) != report.get('acceptance_status'):
        raise ValueError('RESOLUTION_ORIGINAL_ACCEPTANCE_INVALID')
    return report, criteria


def _checked_review(reference, report_ref, report, criteria):
    review = json.loads(read_ref(reference))
    if (review.get('schema_version') != 'assisted-semantic-review.v1'
            or review.get('original_automated_report_ref') != report_ref
            or review.get('binding') != report['binding']):
        raise ValueError('RESOLUTION_REVIEW_BINDING_MISMATCH')
    if (review.get('reviewer_origin') != REVIEWER or review.get('human_review') is not False
            or review.get('independent_blind_review') is not False
            or review.get('release_authority') is not False):
        raise ValueError('RESOLUTION_REVIEW_ORIGIN_INVALID')
    original = read_ref(report['binding']['candidate_ref'])
    if read_ref(review['retained_candidate_ref']) != original:
        raise ValueError('RESOLUTION_RETAINED_CANDIDATE_MISMATCH')
    payload = json.loads(original)
    eligible = {c.dimension for c in criteria if c.applicable and c.role == 'primary'
                and c.method == 'trial_judge' and c.dimension in SEMANTIC_DIMENSIONS}
    rows = review['criteria']
    if (not rows or len({r['dimension'] for r in rows}) != len(rows)
            or any(r['dimension'] not in eligible or r['status'] not in STATUSES - {'not_applicable'} for r in rows)
            or not isinstance(review.get('reason'), str) or not review['reason'].strip()):
        raise ValueError('RESOLUTION_REVIEW_SCOPE_INVALID')
    if decision([Criterion('item'+str(i), r['dimension'], 'review') for i, r in enumerate(rows)], rows) != review['status']:
        raise ValueError('RESOLUTION_REVIEW_DECISION_INVALID')
    spans = review.get('source_spans')
    if not isinstance(spans, list):
        raise ValueError('RESOLUTION_SOURCE_SPANS_INVALID')
    for span in spans:
        number, start, end = (span.get(k) for k in ('context_number', 'start', 'end'))
        contexts = payload.get('contexts', [])
        if (any(type(v) is not int for v in (number, start, end)) or not 1 <= number <= len(contexts)
                or not 0 <= start < end <= len(contexts[number - 1]['text'])
                or contexts[number - 1]['text'][start:end] != span.get('quote')):
            raise ValueError('RESOLUTION_SOURCE_QUOTE_MISMATCH')
    if any(r['dimension'] == 'grounding' and r['status'] == 'passed' for r in rows) and not spans:
        if payload.get('expected_behavior') != 'refuse' or not isinstance(review.get('absence_check'), str) or not review['absence_check'].strip():
            raise ValueError('RESOLUTION_GROUNDING_EVIDENCE_MISSING')
    return review


def resolve_review(report_ref, review_ref, output_root):
    """Caller explicitly selects the record; review content grants no Apply authority."""
    record = {'schema_version': 'unified-evaluation-resolution.v1',
              'original_report_ref': report_ref, 'review_ref': review_ref,
              'decision_scope': 'assisted_candidate_validation',
              'human_review': False, 'independent_blind_review': False,
              'automated_judge_qualified': False, 'release_authority': False,
              'additional_model_calls': 0, 'review_usage': 'unavailable_for_chat_review'}
    try:
        original, criteria = checked_report(report_ref)
        record.update(binding=original['binding'], sample=original.get('sample'))
        review = _checked_review(review_ref, report_ref, original, criteria)
        reviewed = {r['dimension']: r for r in review['criteria']}
        rows = deepcopy(original['criteria'])
        applied_criteria = set()
        # Scope: explicit semantic review only. 为什么要有它：复核可解决裁判故障，但不能改写编译、行为、修改范围和实机验收，也不能使旧候选重新通过。
        for c, row in zip(criteria, rows):
            if c.dimension in reviewed and c.method == 'trial_judge' and c.role == 'primary' and c.applicable:
                applied_criteria.add(c.id)
                row.update(status=reviewed[c.dimension]['status'], reason=review['reason'],
                           decision_use='assisted_review', evidence_refs=[review_ref],
                           original_evaluator=row['evaluator'], evaluator='assisted_review', value=None)
        # 范围：本次复核实际覆盖的合法维度。为什么要有它：原报告自报assisted_review不能让未复核裁判结果取得验收效力，只有本次已校验并应用的criterion可以升级。
        accepted = [dict(r, status='insufficient_evidence') if c.method == 'trial_judge'
                    and c.id not in applied_criteria and r['status'] in {'passed', 'failed'} else r
                    for c, r in zip(criteria, rows)]
        record.update(status=decision(criteria, accepted), criteria=rows,
                      original_status=original['status'], original_acceptance_status=original['acceptance_status'],
                      original_evaluation_cost=original['evaluation_cost'], reviewer_origin=review['reviewer_origin'])
        read_ref(report_ref); read_ref(review_ref); read_ref(original['binding']['candidate_ref'])
    except Exception as exc:
        record.update(status='evaluation_error', error_code=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
    return dict(record, report_ref=save(output_root / 'result.json', record))


def resolve_rag_batch(automated_result_ref, selection_ref, output_root):
    """An explicit host-side selection connects all original rows to their reviews."""
    selection = json.loads(read_ref(selection_ref))
    if (selection.get('schema_version') != 'judge-selection.v1'
            or selection.get('selected') != 'current_codex_conversation_assisted_review'
            or selection.get('human_review') is not False
            or selection.get('execute_or_apply_authority') is not False):
        raise ValueError('RESOLUTION_SELECTION_INVALID')
    batch_ref = selection['review_batch_ref']
    batch = json.loads(read_ref(batch_ref))
    if batch.get('schema_version') != 'assisted-semantic-review-batch.v1' or batch['original_result_ref'] != automated_result_ref:
        raise ValueError('RESOLUTION_BATCH_BINDING_MISMATCH')
    original = json.loads(read_ref(automated_result_ref))
    queries = original['query_results']
    if original['query_count'] != len(queries):
        raise ValueError('RESOLUTION_QUERY_COUNT_INVALID')
    reviews = {}
    for ref in batch['review_refs']:
        review = json.loads(read_ref(ref)); key = canonical(review['original_automated_report_ref'])
        if key in reviews:
            raise ValueError('RESOLUTION_DUPLICATE_REVIEW')
        reviews[key] = ref
    refs = [q['unified_evaluation']['report_ref'] for q in queries]
    if len({canonical(ref) for ref in refs}) != len(refs) or set(reviews) - {canonical(ref) for ref in refs}:
        raise ValueError('RESOLUTION_UNKNOWN_OR_DUPLICATED_CANDIDATE')
    results = []
    for index, (query, ref) in enumerate(zip(queries, refs)):
        embedded = dict(query['unified_evaluation']); embedded.pop('report_ref')
        if embedded != json.loads(read_ref(ref)):
            raise ValueError('RESOLUTION_EMBEDDED_RESULT_MISMATCH')
        review = reviews.get(canonical(ref))
        if review:
            result = resolve_review(ref, review, output_root / 'candidates' / str(index))
        else:
            base, _ = checked_report(ref)
            result = {'status': base['acceptance_status'], 'binding': base['binding'],
                      'original_report_ref': ref, 'review_applied': False}
        results.append({'case_index': query['case_index'], 'query': query['query'], 'resolution': result})
    # Scope: preserve the full denominator. 为什么要有它：未复核和失败样本不能从汇总消失，辅助复核通过也不能变成自动裁判的成绩。
    counts = Counter(r['resolution']['status'] for r in results)
    status = next((s for s in ('failed', 'evaluation_error', 'insufficient_evidence') if counts[s]),
                  'passed' if results and counts['passed'] == len(results) else 'not_applicable')
    read_ref(automated_result_ref); read_ref(selection_ref); read_ref(batch_ref)
    return {'schema_version': 'unified-evaluation-resolution-batch.v1', 'status': status,
            'automated_result_ref': automated_result_ref, 'selection_ref': selection_ref,
            'review_batch_ref': batch_ref, 'query_count': len(results), 'query_results': results,
            'counts': {s: counts[s] for s in sorted(STATUSES)}, 'additional_model_calls': 0,
            'scope': 'assisted_candidate_validation', 'automated_judge_qualified': False,
            'human_review': False, 'release_authority': False}
