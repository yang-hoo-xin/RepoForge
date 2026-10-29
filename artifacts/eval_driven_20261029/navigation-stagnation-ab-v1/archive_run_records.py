"""Copy completed diagnostic records without duplicating build caches.

为什么要有它：原始请求、回执、冻结代码与失败输出另存于项目证据目录；大型环境缓存留在原位置，不删除原件。
"""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path('C:/tmp/rfnav2103-20261029-v1')
destination = Path(__file__).resolve().parent / 'serilog-run-records.zip'
included, excluded = {}, {}
with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        reason = None
        if any(part in {'.cs', 'bin', 'obj', '__pycache__', '.git'} for part in parts):
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
                'scope': 'Run records, frozen platforms and public sources; not a self-contained SDK/package environment.'}
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
