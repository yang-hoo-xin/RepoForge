"""Restore only the SDK bytes required by the existing Serilog qualification.

为什么要有它：环境恢复可重复核验，但不能改写既有资格、源码、控制或模型配置。
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import httpx

OUT = Path(__file__).resolve().parent
TARGET = Path('C:/Users/10160/AppData/Local/Temp/rfext1020/sdk')
MANIFEST = 'https://dotnetcli.blob.core.windows.net/dotnet/release-metadata/8.0/releases.json'
EXPECTED = 'b9eace03c8471717e3f98873527005dbd9a92367b954f8c48484d2b7b78efbac'
started = time.monotonic()
if TARGET.exists():
    raise RuntimeError('ORIGINAL_SDK_TARGET_ALREADY_EXISTS_NO_OVERWRITE')
if shutil.disk_usage(TARGET.anchor).free < 3 * 1024**3:
    raise RuntimeError('INSUFFICIENT_DISK_FOR_BOUNDED_RESTORE')
with httpx.Client(timeout=60, follow_redirects=True) as client:
    response = client.get(MANIFEST)
    response.raise_for_status()
    metadata = response.json()
    matches = [s for r in metadata['releases'] for s in r.get('sdks', [r['sdk']]) if s['version'] == '8.0.100']
    chosen = [f for s in matches for f in s['files'] if f['rid'] == 'win-x64' and f['name'].endswith('.zip')]
    assert len(chosen) == 1, 'SDK_ARCHIVE_IDENTITY_AMBIGUOUS'
    item = chosen[0]
    archive = OUT / 'dotnet-sdk-8.0.100-win-x64.zip'
    digest = hashlib.sha512()
    with archive.open('xb') as sink, client.stream('GET', item['url']) as download:
        download.raise_for_status()
        last = 0
        for block in download.iter_bytes(1024 * 1024):
            if time.monotonic() - started > 600:
                raise TimeoutError('SDK_RESTORE_DOWNLOAD_BUDGET_EXHAUSTED')
            sink.write(block)
            digest.update(block)
            if sink.tell() - last >= 32 * 1024**2:
                last = sink.tell()
                print(json.dumps({'downloaded_bytes': last}), flush=True)
    assert digest.hexdigest().lower() == item['hash'].lower(), 'OFFICIAL_ARCHIVE_HASH_MISMATCH'
with zipfile.ZipFile(archive) as package:
    assert hashlib.sha256(package.read('dotnet.exe')).hexdigest() == EXPECTED, 'BOUND_EXECUTABLE_HASH_MISMATCH'
    assert all((TARGET / n).resolve().is_relative_to(TARGET.resolve()) for n in package.namelist()), 'ARCHIVE_PATH_ESCAPE'
    TARGET.mkdir(parents=True, exist_ok=False)
    package.extractall(TARGET)
assert hashlib.sha256((TARGET / 'dotnet.exe').read_bytes()).hexdigest() == EXPECTED
result = {
    'status': 'original_sdk_restored_pending_original_qualification_validation',
    'sdk_version': '8.0.100', 'manifest_url': MANIFEST, 'archive_url': item['url'],
    'archive_sha512': digest.hexdigest(), 'archive_bytes': archive.stat().st_size,
    'target': str(TARGET), 'dotnet_sha256': EXPECTED,
    'wall_seconds': round(time.monotonic() - started, 3),
    'qualification_modified': False, 'controls_rerun': False,
    'why': '为什么要有它：下载包完整性与原资格绑定分别核验，不能仅靠版本号宣称环境一致。',
}
(OUT / 'sdk-restoration.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False), flush=True)
