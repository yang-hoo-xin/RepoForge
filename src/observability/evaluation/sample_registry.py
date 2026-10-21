"""Dataset membership and provenance, distinct from assertions and repeated runs."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Sample:
    dataset_id: str
    sample_id: str
    family_id: str
    source_kind: str
    partition: str
    label_origin: str
    tuning_exposed: bool

    def __post_init__(self):
        if any(not isinstance(v,str) or not v.strip() for v in (self.dataset_id,self.sample_id,self.family_id)):
            raise ValueError('EVALUATION_SAMPLE_ID_REQUIRED')
        if self.source_kind not in {'real_request','constructed','replayed'}:
            raise ValueError('EVALUATION_SAMPLE_SOURCE_INVALID')
        if self.partition not in {'development','regression','calibration','holdout','boundary'}:
            raise ValueError('EVALUATION_SAMPLE_PARTITION_INVALID')
        if self.label_origin not in {'human','executable_reference','ai','mixed','unlabeled'}:
            raise ValueError('EVALUATION_SAMPLE_LABEL_ORIGIN_INVALID')
        if type(self.tuning_exposed) is not bool or (self.partition=='holdout' and self.tuning_exposed):
            raise ValueError('EVALUATION_EXPOSED_SAMPLE_NOT_HOLDOUT')


def validate_samples(samples):
    identities=set();families={}
    for item in samples:
        key=(item.dataset_id,item.sample_id)
        if key in identities:raise ValueError('EVALUATION_DUPLICATE_SAMPLE')
        identities.add(key)
        families.setdefault((item.dataset_id,item.family_id),set()).add(item.partition)
    # Scope: independent holdouts. 为什么要有它：同源变体跨开发和验收分区会泄漏修复模式，重复运行不能增加独立任务数。
    if any('holdout' in parts and len(parts)>1 for parts in families.values()):
        raise ValueError('EVALUATION_FAMILY_CROSSES_HOLDOUT')
    return {'samples':len(identities),'families':len(families),'membership':[asdict(s) for s in samples],
            'business_distribution_claimed':False}


def admit_samples(samples, *, registry_refs=()):
    """Check a declared registry history; this does not certify semantic labels."""
    import json
    from .unified import read_ref

    samples = tuple(samples)
    result = validate_samples(samples)
    references = list(registry_refs)
    if any(sample.partition == 'holdout' for sample in samples) and not references:
        raise ValueError('EVALUATION_HOLDOUT_REGISTRY_REQUIRED')
    result.update(
        schema_version='evaluation-sample-admission.v1',
        registration_status='registered' if references else 'unregistered_development_metadata',
        registry_refs=references, history_registry_refs=[], registered_origins=[],
        history_coverage='provided_registry_chain_only', independence_verified=False,
        semantic_labels_verified=False, release_authority=False,
    )
    if not references:
        return result

    records, visited, pending = [], set(), list(references)
    while pending:
        reference = pending.pop(0)
        raw = read_ref(reference)
        key = (reference['path'], reference['sha256'])
        if key in visited:
            continue
        visited.add(key)
        document = json.loads(raw)
        if (not isinstance(document, dict)
                or document.get('schema_version') != 'evaluation-sample-registry.v1'
                or not isinstance(document.get('records'), list)
                or not isinstance(document.get('prior_registry_refs'), list)):
            raise ValueError('EVALUATION_REGISTRY_DOCUMENT_INVALID')
        document_keys = set()
        for record in document['records']:
            if (not isinstance(record, dict) or set(record) not in ({
                    'sample', 'origin_id', 'origin_ref', 'label_refs', 'exposure_refs'}, {
                    'sample', 'origin_id', 'origin_ref', 'label_refs', 'exposure_refs', 'input_binding'})
                    or not isinstance(record['origin_id'], str) or not record['origin_id'].strip()
                    or not isinstance(record['label_refs'], list)
                    or not isinstance(record['exposure_refs'], list)):
                raise ValueError('EVALUATION_REGISTRY_RECORD_INVALID')
            try:
                registered = Sample(**record['sample'])
            except (TypeError, ValueError) as error:
                raise ValueError('EVALUATION_REGISTRY_SAMPLE_INVALID') from error
            identity = (registered.dataset_id, registered.sample_id)
            if identity in document_keys:
                raise ValueError('EVALUATION_REGISTRY_DUPLICATE_SAMPLE')
            document_keys.add(identity)
            # Scope: immutable public provenance refs, not private test answers.
            # 为什么要有它：来源和标签声明须绑定实际字节，登记本身不授予人类真值或发布资格。
            read_ref(record['origin_ref'])
            for evidence in (*record['label_refs'], *record['exposure_refs']):
                read_ref(evidence)
            if registered.label_origin != 'unlabeled' and not record['label_refs']:
                raise ValueError('EVALUATION_REGISTRY_LABEL_REFERENCE_REQUIRED')
            if record.get('input_binding') is not None:
                input_identity(record['input_binding'], verify_content=False)
            records.append((registered, record))
        result['history_registry_refs'].append(reference)
        pending.extend(document['prior_registry_refs'])

    selected = []
    for sample in samples:
        identity = (sample.dataset_id, sample.sample_id)
        matching = [(prior, record) for prior, record in records
                    if (prior.dataset_id, prior.sample_id) == identity]
        exact = [record for prior, record in matching if prior == sample]
        if not exact:
            raise ValueError('EVALUATION_SAMPLE_NOT_REGISTERED')
        if len({record['origin_id'] for _, record in matching}) != 1:
            raise ValueError('EVALUATION_REGISTERED_ORIGIN_CHANGED')
        record = exact[0]
        selected.append((sample, record))
        if sample.partition == 'holdout':
            # Scope: known exposure across every supplied registry generation.
            # 为什么要有它：换dataset/sample/family名不能洗掉同一origin的已知曝光；不声称发现未登记的语义别名。
            related = [prior for prior, item in records
                       if item['origin_id'] == record['origin_id']
                       or (prior.dataset_id, prior.family_id) == (sample.dataset_id, sample.family_id)]
            if any(prior.tuning_exposed or prior.partition != 'holdout' for prior in related):
                raise ValueError('EVALUATION_HOLDOUT_PREVIOUSLY_EXPOSED')
        result['registered_origins'].append({
            'dataset_id': sample.dataset_id, 'sample_id': sample.sample_id,
            'origin_id': record['origin_id'], 'origin_ref': record['origin_ref'],
            'label_refs': record['label_refs'], 'exposure_refs': record['exposure_refs'],
            'input_binding': record.get('input_binding'),
        })
    for sample, record in selected:
        if sample.partition == 'holdout' and any(
                item['origin_id'] == record['origin_id']
                and (other.partition != 'holdout' or other.tuning_exposed)
                for other, item in selected):
            raise ValueError('EVALUATION_ORIGIN_CROSSES_HOLDOUT')
    result['declared_origin_count'] = len({record['origin_id'] for _, record in selected})
    return result


def input_identity(binding, *, verify_content=True):
    """Validate source bytes/selector; code profiles also retain location identity."""
    from .unified import read_ref
    from pathlib import Path
    import re
    fields = {'rag': {'kind', 'test_set_ref', 'case_index'},
              'code': {'kind', 'description_ref', 'execution_config_ref'}}
    if (not isinstance(binding, dict) or binding.get('kind') not in fields
            or set(binding) != fields[binding['kind']]):
        raise ValueError('EVALUATION_INPUT_BINDING_INVALID')
    identity = {'kind': binding['kind']}
    for key, value in binding.items():
        if key.endswith('_ref'):
            if (not isinstance(value, dict) or set(value) != {'path', 'sha256'}
                    or not isinstance(value['path'], str) or not Path(value['path']).is_absolute()
                    or not isinstance(value['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', value['sha256'])):
                raise ValueError('EVALUATION_INPUT_REFERENCE_INVALID')
            # Scope: registry inventory inspects only public input-reference metadata.
            # 为什么要有它：封存题面在准入盘点时不能被读取；原件字节校验留给被选定的实际执行入口。
            if verify_content:
                read_ref(value)
            # Scope: code execution profiles may resolve paths relative to their location.
            # 为什么要有它：同字节profile换目录可能指向另一个工程，不能只靠文件内容哈希认成同任务。
            identity[key] = value['sha256'] if binding['kind'] == 'rag' else dict(value)
    if binding['kind'] == 'rag':
        if type(binding['case_index']) is not int or binding['case_index'] < 0:
            raise ValueError('EVALUATION_INPUT_SELECTOR_INVALID')
        identity['case_index'] = binding['case_index']
    return identity


def bind_sample_input(sample, admission, binding, *, query=None):
    """Bind a registered sample to the actual evaluated input, not its display ID."""
    import json
    from .unified import read_ref
    matching = [row for row in admission['registered_origins']
                if (row['dataset_id'], row['sample_id']) == (sample.dataset_id, sample.sample_id)]
    expected = matching[0].get('input_binding') if matching else None
    # Scope: registered holdout execution, not metadata-only inventory.
    # 为什么要有它：合法登记A不能给另一个test-set或description B贴身份；旧开发记录可保留未绑定状态但不冒充独立验收。
    if expected is None:
        if sample.partition == 'holdout':
            raise ValueError('EVALUATION_HOLDOUT_INPUT_BINDING_REQUIRED')
        return {'status': 'unbound_development_input', 'independence_verified': False}
    if binding is None or input_identity(expected) != input_identity(binding):
        raise ValueError('EVALUATION_REGISTERED_INPUT_MISMATCH')
    if query is not None and binding['kind'] == 'rag':
        document = json.loads(read_ref(binding['test_set_ref']))
        cases = document.get('test_cases', document.get('cases')) if isinstance(document, dict) else document
        index = binding['case_index']
        if (not isinstance(cases, list) or index >= len(cases)
                or not isinstance(cases[index], dict) or cases[index].get('query') != query):
            raise ValueError('EVALUATION_INPUT_QUERY_MISMATCH')
    return {'status': 'input_bound', 'registered_input': expected, 'actual_input': binding,
            'independence_verified': False}
