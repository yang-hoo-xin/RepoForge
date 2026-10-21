"""Candidate-bound pre-launch evaluation; no evaluator can grant execution authority."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable
from uuid import uuid4

STATUSES = frozenset({'passed', 'failed', 'insufficient_evidence', 'evaluation_error', 'not_applicable'})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def disk_path(path):
    value = str(Path(path).absolute())
    if value.startswith(("//", "\\\\")):
        if not re.match(r"^\\\\\?\\[A-Za-z]:\\", value):
            raise ValueError('EVALUATION_LOCAL_PATH_REQUIRED')
        return Path(value)
    # Scope: local Windows paths only. 为什么要有它：深层任务路径需要扩展路径格式写入，不能因此丢失已完成的评测；UNC和设备路径仍拒绝。
    return Path("\\\\?\\" + value) if os.name == 'nt' else Path(value)


def file_ref(path):
    disk = disk_path(path).resolve(strict=True)
    public = str(disk)
    if os.name == 'nt' and public.startswith("\\\\?\\"):
        public = public[4:]
    return {'path': public, 'sha256': hashlib.sha256(disk.read_bytes()).hexdigest()}


def read_ref(reference):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}
            or not isinstance(reference['path'], str)
            or not Path(reference['path']).is_absolute()
            or not isinstance(reference['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', reference['sha256'])):
        raise ValueError('EVALUATION_REFERENCE_INVALID')
    data = disk_path(reference['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('EVALUATION_EVIDENCE_CHANGED')
    return data


def save(path, value):
    path = disk_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value))
    return file_ref(path)


@dataclass(frozen=True)
class Criterion:
    id: str
    dimension: str
    evaluator: str
    required: bool = True
    applicable: bool = True
    role: str = 'primary'
    threshold: float | None = None
    method: str = 'trial_judge'

    def __post_init__(self):
        if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,127}', v) for v in (self.id, self.dimension, self.evaluator)):
            raise ValueError('EVALUATION_CRITERION_ID_REQUIRED')
        if type(self.required) is not bool or type(self.applicable) is not bool:
            raise ValueError('EVALUATION_CRITERION_FLAGS_INVALID')
        if self.method not in {'execution', 'trial_judge'}:
            raise ValueError('EVALUATION_METHOD_INVALID')
        if self.role not in {'primary', 'calibration'} or (self.role == 'calibration' and self.required):
            raise ValueError('EVALUATION_CALIBRATION_CANNOT_GATE')
        if self.threshold is not None and (type(self.threshold) not in (float, int) or not math.isfinite(self.threshold)):
            raise ValueError('EVALUATION_THRESHOLD_INVALID')


@dataclass(frozen=True)
class Binding:
    task_id: str
    conversation_id: str
    run_id: str
    candidate_ref: dict

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.task_id, self.conversation_id, self.run_id)):
            raise ValueError('EVALUATION_TASK_CONVERSATION_RUN_REQUIRED')
        read_ref(self.candidate_ref)


def observation(binding, criterion, status, *, evidence=(), reason='', value=None):
    if status not in STATUSES:
        raise ValueError('EVALUATION_STATUS_INVALID')
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError('EVALUATION_SCORE_INVALID')
    return {'binding': asdict(binding), 'criterion_id': criterion.id, 'status': status,
            'reason': reason, 'value': value, 'evidence_refs': list(evidence)}


def validate_policy(criteria):
    ids = set()
    dimensions = set()
    for item in criteria:
        if item.id in ids:
            raise ValueError('EVALUATION_DUPLICATE_CRITERION')
        ids.add(item.id)
        if item.role == 'primary':
            if item.dimension in dimensions:
                raise ValueError('EVALUATION_DUPLICATE_PRIMARY_DIMENSION')
            dimensions.add(item.dimension)


def decision(criteria, rows):
    selected = [r['status'] for c, r in zip(criteria, rows)
                if c.required and c.applicable and c.role == 'primary']
    if not selected:
        return 'not_applicable'
    # Scope: required, applicable primary criteria only. 为什么要有它：语义高分不能抵消行为失败，缺实机也不能误记为交付通过。
    for status in ('failed', 'evaluation_error', 'insufficient_evidence'):
        if status in selected:
            return status
    return 'passed' if all(s == 'passed' for s in selected) else 'insufficient_evidence'


class EvaluationSession:
    """One evaluation per immutable candidate and rubric; adapters remain task-specific."""
    def __init__(self, output_root, *, criteria, evaluators: dict[str, Callable], rubric_version, sample=None,
                 registry_refs=(), input_binding=None):
        self.root = Path(output_root).resolve()
        self.criteria = tuple(criteria)
        validate_policy(self.criteria)
        if not isinstance(rubric_version, str) or not rubric_version.strip():
            raise ValueError('EVALUATION_RUBRIC_VERSION_REQUIRED')
        self.rubric_version = rubric_version
        self.evaluators = dict(evaluators)
        self.sample = None if sample is None else asdict(sample)
        from .sample_registry import admit_samples, bind_sample_input
        self.registry_refs = tuple(registry_refs)
        # Scope: every evaluation entry, including direct sessions.
        # 为什么要有它：跳过CLI或Hook不能把未登记的自报holdout写成正式独立样本；登记仍不证明标签语义。
        self.sample_admission = None if sample is None else admit_samples([sample], registry_refs=self.registry_refs)
        self.input_binding = input_binding
        self.sample_input_admission = None if sample is None else bind_sample_input(sample, self.sample_admission, input_binding)

    def evaluate(self, binding: Binding, payload):
        # Scope: pre/post evaluation binding. 为什么要有它：结果必须对应同一任务、会话、运行与候选，不能把旧候选成绩移给新补丁。
        read_ref(binding.candidate_ref)
        if self.sample is not None:
            from .sample_registry import Sample, admit_samples, bind_sample_input
            if admit_samples([Sample(**self.sample)], registry_refs=self.registry_refs) != self.sample_admission:
                raise ValueError('EVALUATION_SAMPLE_ADMISSION_CHANGED')
            if (self.sample_input_admission['status'] == 'input_bound' and self.input_binding['kind'] == 'rag'
                    and (not isinstance(payload.get('query'), str) or not payload['query'].strip())):
                raise ValueError('EVALUATION_INPUT_QUERY_REQUIRED')
            bind_sample_input(Sample(**self.sample), self.sample_admission, self.input_binding,
                              query=payload.get('query'))
            if self.sample_input_admission['status'] == 'input_bound' and payload.get('evaluation_input_binding') != self.input_binding:
                raise ValueError('EVALUATION_CANDIDATE_INPUT_BINDING_MISMATCH')
        if canonical(payload) != read_ref(binding.candidate_ref):
            raise ValueError('EVALUATION_PAYLOAD_BINDING_MISMATCH')
        expected_binding = asdict(binding)
        expected_payload = canonical(payload)
        directory = self.root / uuid4().hex
        started = time.monotonic()
        invocation_refs, invocation_costs, by_id = [], [], {}
        groups = {}
        for criterion in self.criteria:
            if not criterion.applicable:
                by_id[criterion.id] = observation(binding, criterion, 'not_applicable', reason='Excluded by registered applicability')
            else:
                groups.setdefault((criterion.evaluator, criterion.role), []).append(criterion)
        for (name, role), criteria in groups.items():
            t0 = time.monotonic()
            record = {'evaluator': name, 'role': role, 'binding': expected_binding,
                      'criterion_ids': [c.id for c in criteria], 'status': 'started', 'cost': None}
            try:
                callback = self.evaluators[name]
                selected_payload = json.loads(expected_payload)
                response = callback(binding, selected_payload, tuple(criteria), disk_path(directory / (name + '-' + role)))
                if asdict(binding) != expected_binding or canonical(selected_payload) != expected_payload:
                    raise ValueError('EVALUATION_INPUT_MUTATED')
                if not isinstance(response, dict) or set(response) != {'results', 'cost', 'evaluator_identity'}:
                    raise ValueError('EVALUATION_ADAPTER_PROTOCOL_INVALID')
                cost = response['cost']
                if not isinstance(cost, dict) or not isinstance(response['evaluator_identity'], dict) or not response['evaluator_identity']:
                    raise ValueError('EVALUATION_COST_OR_IDENTITY_REQUIRED')
                for field in ('model_calls', 'input_tokens', 'output_tokens', 'cached_input_tokens'):
                    value = cost.get(field)
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError('EVALUATION_COST_INVALID')
                if any(c.method == 'execution' for c in criteria) and (
                        cost.get('model_calls') != 0 or response['evaluator_identity'].get('evaluation_method') != 'execution'):
                    raise ValueError('EVALUATION_EXECUTION_BASIS_REQUIRED')
                record.update(cost=cost, evaluator_identity=response['evaluator_identity'])
                expected = {c.id: c for c in criteria}
                results = response['results']
                if not isinstance(results, list) or len(results) != len(criteria):
                    raise ValueError('EVALUATION_CRITERION_RESULT_MISSING')
                pending = {}
                for row in results:
                    if not isinstance(row, dict) or row.get('binding') != expected_binding:
                        raise ValueError('EVALUATION_RESULT_BINDING_MISMATCH')
                    key = row.get('criterion_id')
                    if key not in expected or key in pending or row.get('status') not in STATUSES:
                        raise ValueError('EVALUATION_CRITERION_RESULT_INVALID')
                    if row['status'] == 'not_applicable':
                        raise ValueError('EVALUATION_APPLICABILITY_CANNOT_BE_CHANGED_BY_JUDGE')
                    if row.get('value') is not None and (type(row['value']) not in (int, float) or not math.isfinite(row['value'])):
                        raise ValueError('EVALUATION_SCORE_INVALID')
                    criterion = expected[key]
                    if criterion.threshold is not None and row['status'] in {'passed', 'failed'}:
                        expected_status = 'passed' if row.get('value') is not None and row['value'] >= criterion.threshold else 'failed'
                        if row.get('value') is None or row['status'] != expected_status:
                            raise ValueError('EVALUATION_THRESHOLD_VERDICT_MISMATCH')
                    evidence = row.get('evidence_refs')
                    if not isinstance(evidence, list) or (row['status'] in {'passed', 'failed'} and not evidence):
                        raise ValueError('EVALUATION_DECISION_EVIDENCE_REQUIRED')
                    for reference in evidence:
                        read_ref(reference)
                    pending[key] = row
                read_ref(binding.candidate_ref)
                by_id.update(pending)
                record['status'] = 'evaluation_error' if any(r['status'] == 'evaluation_error' for r in results) else 'completed'
            except Exception as error:
                # Scope: evaluator errors, never content scores. 为什么要有它：API/解析/绑定故障是评估执行失败，不能伪造成内容0分或触发盲目代码修订。
                record.update(status='evaluation_error', error_type=type(error).__name__)
                for criterion in criteria:
                    by_id[criterion.id] = observation(binding, criterion, 'evaluation_error', reason=type(error).__name__)
            invocation_costs.append(record['cost'])
            record['elapsed_seconds'] = time.monotonic() - t0
            invocation_refs.append(save(directory / (name + '-' + role + '.json'), record))
        try:
            read_ref(binding.candidate_ref)
        except Exception:
            for criterion in self.criteria:
                if criterion.applicable:
                    by_id[criterion.id] = observation(binding, criterion, 'evaluation_error', reason='Candidate changed during evaluation')
        rows = [dict(by_id[c.id], binding=json.loads(canonical(expected_binding)), dimension=c.dimension, evaluator=c.evaluator,
                     role=c.role, required=c.required, applicable=c.applicable,
                     decision_use='acceptance' if c.method == 'execution' else 'trial_only') for c in self.criteria]
        # Scope: unqualified model verdicts. 为什么要有它：候选裁判的正负评分均不直接决定验收，错误和缺证据也必须显式保留；执行失败仍阻止放行。
        acceptance_rows = [dict(r, status='insufficient_evidence')
                           if c.method == 'trial_judge' and r['status'] in {'passed', 'failed'} else r
                           for c, r in zip(self.criteria, rows)]
        has_trial = any(c.required and c.applicable and c.role == 'primary' and c.method == 'trial_judge' for c in self.criteria)
        report = {'schema_version': 'unified-evaluation.v1', 'binding': expected_binding,
                  'sample': self.sample, 'sample_admission': self.sample_admission,
                  'sample_input_admission': self.sample_input_admission,
                  'decision_scope': 'trial_observation' if has_trial else 'candidate_validation',
                  'acceptance_status': decision(self.criteria, acceptance_rows),
                  'qualification_status': 'not_established' if has_trial else 'execution_checks_only',
                  'release_authority': False,
                  'rubric': {'version': self.rubric_version, 'criteria': [asdict(c) for c in self.criteria]},
                  'status': decision(self.criteria, rows), 'criteria': rows,
                  'invocation_refs': invocation_refs,
                  'evaluation_cost': {field: {'known_subtotal': sum((c or {}).get(field) or 0 for c in invocation_costs),
                    'complete': all(c is not None and c.get(field) is not None for c in invocation_costs)}
                    for field in ('model_calls', 'input_tokens', 'output_tokens', 'cached_input_tokens')},
                  'elapsed_seconds': time.monotonic() - started,
                  'authority': 'evaluation_only_not_apply_or_release_authorization'}
        return dict(report, report_ref=save(directory / 'result.json', report))
