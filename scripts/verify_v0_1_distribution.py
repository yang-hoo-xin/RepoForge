"""Verify the RepoForge v0.1 wheel and source distribution boundaries."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path


def _ref(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def verify_distribution(dist_dir: Path, expected_version: str) -> dict[str, object]:
    dist_dir = dist_dir.resolve()
    wheel_name = f"repoforge-{expected_version}-py3-none-any.whl"
    sdist_name = f"repoforge-{expected_version}.tar.gz"
    wheel = dist_dir / wheel_name
    sdist = dist_dir / sdist_name
    if not wheel.is_file() or not sdist.is_file():
        raise ValueError("V0_1_DISTRIBUTION_FILES_MISSING")

    dist_info = f"repoforge-{expected_version}.dist-info"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = email.message_from_bytes(archive.read(f"{dist_info}/METADATA"))
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        license_files = [name for name in names if name.startswith(f"{dist_info}/licenses/")]
    if metadata["Name"] != "repoforge" or metadata["Version"] != expected_version:
        raise ValueError("V0_1_DISTRIBUTION_METADATA_MISMATCH")
    if metadata["License"] != "MIT" or not any(name.endswith("/LICENSE") for name in license_files):
        raise ValueError("V0_1_DISTRIBUTION_LICENSE_MISSING")
    expected_entries = {
        "mcp-server = src.mcp_server.server:main",
        "mod-agent = src.mod_agent.cli:main",
    }
    if set(entry_points.splitlines()) - {"", "[console_scripts]"} != expected_entries:
        raise ValueError("V0_1_DISTRIBUTION_ENTRYPOINT_MISMATCH")
    forbidden_wheel = [
        name
        for name in names
        if name.startswith(("tests/", "artifacts/", "workspaces/", "scripts/"))
    ]
    if forbidden_wheel:
        raise ValueError("V0_1_WHEEL_CONTAINS_DEVELOPMENT_FILES")

    prefix = f"repoforge-{expected_version}/"
    required_sdist = {prefix + name for name in ("README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml")}
    with tarfile.open(sdist) as archive:
        sdist_names = set(archive.getnames())
    missing_sdist = sorted(required_sdist - sdist_names)
    if missing_sdist:
        raise ValueError("V0_1_SDIST_RELEASE_FILES_MISSING")

    return {
        "schema_version": "repoforge-v0.1-distribution-verification.v1",
        "status": "passed",
        "version": expected_version,
        "wheel_ref": _ref(wheel),
        "sdist_ref": _ref(sdist),
        "wheel_file_count": len(names),
        "wheel_console_scripts": sorted(expected_entries),
        "wheel_forbidden_development_files": [],
        "sdist_required_release_files": sorted(required_sdist),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_distribution(args.dist_dir, args.expected_version)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
