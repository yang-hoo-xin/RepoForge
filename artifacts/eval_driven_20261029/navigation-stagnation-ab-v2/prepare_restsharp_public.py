"""Materialize the existing qualified public RestSharp inputs, without execution.

为什么要有它：新诊断只复用原公开输入与构建配置，准备阶段不运行模型、控制组或私有验收。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile

ADMISSION = Path(__file__).with_name("qualification-reuse-restsharp.json")
ADMISSION_SHA = "c3455e5a04a7ebc9af3bfac52adf326da4f0e9676aeb2122f4aad236ddf42bd6"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bound(reference):
    raw = Path(reference["path"]).read_bytes()
    if digest(raw) != reference["sha256"]:
        raise ValueError("BOUND_PUBLIC_INPUT_CHANGED")
    return raw


def reference(path):
    return {"path": str(path.resolve()), "sha256": digest(path.read_bytes())}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    admission = json.loads(bound({"path": str(ADMISSION), "sha256": ADMISSION_SHA}))
    output = args.output_root.resolve()
    if output.exists():
        raise ValueError("FRESH_OUTPUT_REQUIRED")
    inputs = admission["public_input_refs"]
    payloads = {key: bound(ref) for key, ref in inputs.items()}
    profile_raw = bound(admission["repository_profile_ref"])
    bound(admission["sdk_ref"])
    source = admission["source"]
    base = source["base_commit"]
    observed = subprocess.check_output(["git", "-C", source["path"], "rev-parse", "HEAD"], text=True).strip()
    if observed != base:
        raise ValueError("BASE_COMMIT_CHANGED")
    if subprocess.check_output(["git", "-C", source["path"], "status", "--porcelain"]):
        raise ValueError("BASE_CHECKOUT_NOT_CLEAN")
    archive = subprocess.check_output(["git", "-C", source["path"], "archive", "--format=tar", base])
    original = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or ":" in member.name:
                raise ValueError("ARCHIVE_PATH_INVALID")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("ARCHIVE_LINK_OR_SPECIAL_FILE")
            blob = bundle.extractfile(member).read()
            checked_out = (Path(source["path"]) / name.as_posix()).read_bytes()
            # 为什么要有它：原独立验收按生产源码前像字节比对，Windows CRLF 检出不能被 archive 的 LF 字节静默替换。
            if checked_out != blob and checked_out.replace(b"\r\n", b"\n") != blob:
                raise ValueError("CHECKOUT_DIFFERS_FROM_BASE_ARCHIVE")
            original[name.as_posix()] = checked_out
    output.mkdir(parents=True)
    target = output / "source"
    target.mkdir()
    for name, raw in original.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    # 为什么要有它：与原准备入口保持相同补丁字节和顺序，不能从历史模型候选构造新基线。
    for key in ("public_reproduction_ref", "environment_patch_ref"):
        subprocess.run(["git", "apply", "--check", "-"], input=payloads[key], cwd=target, check=True, capture_output=True)
        subprocess.run(["git", "apply", "-"], input=payloads[key], cwd=target, check=True, capture_output=True)
    for name, raw in original.items():
        if name.endswith(".cs") and name.startswith(("src/", "gen/")):
            if (target / name).read_bytes() != raw:
                raise ValueError("PRODUCTION_BASELINE_CHANGED")
    problem = output / "problem_statement.txt"
    problem.write_bytes(payloads["public_problem_ref"])
    profile_path = output / "repository-profile.json"
    profile_path.write_bytes(profile_raw)
    profile = CSharpRepositoryProfile.load(profile_path)
    binding = profile.bind(target)
    profile.verify_fixed_inputs(target, binding)
    manifest = [{"path": p.relative_to(target).as_posix(), "sha256": digest(p.read_bytes())}
                for p in sorted(target.rglob("*")) if p.is_file()]
    result = {"schema_version": "restsharp_public_input_preparation.v1", "source_root": str(target),
        "base_commit": base, "archive_sha256": digest(archive), "admission_ref": reference(ADMISSION),
        "preparation_script_ref": reference(Path(__file__)), "original_public_input_refs": inputs,
        "profile_ref": reference(profile_path), "problem_ref": reference(problem),
        "fixed_input_binding": binding, "source_manifest": manifest,
        "source_manifest_sha256": digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()),
        "production_cs_matches_base": True, "base_checkout_bytes_preserved": True,
        "archive_comparison": "Exact bytes or Git CRLF checkout normalization only",
        "model_calls": 0, "build_or_tests_executed": False,
        "controls_rerun": False, "hidden_material_supplied": False,
        "preparation_wall_seconds": time.perf_counter() - started}
    receipt = output / "public-preparation.json"
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "prepared", "receipt_ref": reference(receipt), "source_root": str(target)}))


if __name__ == "__main__":
    main()
