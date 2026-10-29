"""Preserve both stopped runs without exporting private acceptance inputs.

为什么要有它：实际请求、回执和公开源码可单独复核；私有验收与大型构建缓存仍留在原位置，不删除原件。
"""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path('C:/tmp/rfn2241v2')
destination = Path(__file__).resolve().parent / 'run-records.zip'
included, excluded = {}, {}
with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        reason = None
        if 'platform' in parts and 'external' in parts:
            reason = 'bound_external_material_retained_locally_not_exported'
        elif any(part in {'.cs', 'bin', 'obj', '__pycache__', '.git'} for part in parts):
            reason = 'generated_build_or_interpreter_cache_retained_at_original_path'
        elif parts[0].endswith('-platform-source') or (len(parts) == 1 and path.suffix == '.zip'):
            reason = 'duplicate_export_frozen_platform_copy_included_instead'
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        assert path.resolve().is_relative_to(root.resolve()), 'ARCHIVE_SOURCE_ESCAPE'
        data = path.read_bytes()
        archive.writestr(relative.as_posix(), data)
        included[relative.as_posix()] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    manifest = {'original_root': str(root), 'files': included, 'excluded_counts': excluded,
        'original_files_deleted': False,
        'scope': 'Public run records, runtime source and public task inputs; external bound files, private acceptance and build caches remain at their original locations.'}
    archive.writestr('RECORDS-MANIFEST.json', json.dumps(manifest, indent=2))
with zipfile.ZipFile(destination) as archive:
    for name, row in included.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == row['sha256'], name
receipt = {'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    'archive_bytes': destination.stat().st_size, 'record_count': len(included),
    'included_bytes': sum(row['bytes'] for row in included.values()),
    'excluded_counts': excluded, 'all_included_hashes_verified': True, 'original_files_deleted': False,
    'scope': manifest['scope']}
destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt), flush=True)
