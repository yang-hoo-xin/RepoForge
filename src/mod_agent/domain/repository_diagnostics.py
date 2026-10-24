"""Explicit snapshot diagnostic outputs with immutable expectation provenance.

为什么要有它：诊断写入独立输出目录，原源码和评分依据保持可核验，未登记的新文件不能被目录名掩盖。
"""
from __future__ import annotations

import re
import shutil
import stat
from pathlib import Path, PurePosixPath

from src.mod_agent.domain.full_project_compilation import _ref, _save
from src.mod_agent.model_qualification import read_bound_json


def require(condition, code):
    if not condition:
        raise ValueError("REPOSITORY_DIAGNOSTICS_" + code)


def validate_policy(policy):
    from src.mod_agent.domain.csharp_repository_backend import _relative
    require(isinstance(policy, dict) and set(policy) == {
        "kind", "expectation_root", "compiler_output_roots", "received_markers"}
        and policy["kind"] == "verify_snapshots_v1", "POLICY_INVALID")
    _relative(policy["expectation_root"])
    roots, markers = policy["compiler_output_roots"], policy["received_markers"]
    require(isinstance(roots, list) and len(set(roots)) == len(roots), "COMPILER_ROOTS_INVALID")
    for value in roots:
        _relative(value)
        require(PurePosixPath(value).name in {"bin", "obj"}, "COMPILER_ROOTS_INVALID")
    require(isinstance(markers, list) and markers and len(set(markers)) == len(markers)
        and all(isinstance(value, str) and re.fullmatch(r"\.(?:[A-Za-z0-9_-]+\.)?received\.", value)
                for value in markers), "RECEIVED_MARKERS_INVALID")


def bind_policy(source, policy, source_inputs):
    from src.mod_agent.domain.csharp_repository_backend import _within
    validate_policy(policy)
    source = Path(source).resolve(strict=True)
    expected = _within(source, policy["expectation_root"])
    require(expected.is_dir(), "EXPECTATION_ROOT_MISSING")
    projects = {PurePosixPath(row["path"]).parent for row in source_inputs if row["path"].endswith(".csproj")}
    require(all(PurePosixPath(value).parent in projects for value in policy["compiler_output_roots"]),
        "COMPILER_ROOT_NOT_PROJECT_OUTPUT")
    require(not any(PurePosixPath(policy["expectation_root"]).is_relative_to(PurePosixPath(root))
        for root in policy["compiler_output_roots"]), "EXPECTATIONS_INSIDE_COMPILER_OUTPUT")
    require(any(PurePosixPath(row["path"]).is_relative_to(PurePosixPath(policy["expectation_root"]))
        and ".verified." in PurePosixPath(row["path"]).name for row in source_inputs), "NO_EXPECTATIONS")


def inventory(root, excluded=()):
    root = Path(root).resolve(strict=True)
    files, pending = {}, [root]
    while pending:
        for path in pending.pop().iterdir():
            require(not path.is_symlink() and not getattr(path.lstat(), "st_file_attributes", 0)
                & stat.FILE_ATTRIBUTE_REPARSE_POINT and path.resolve().is_relative_to(root), "LINK_OR_ESCAPE")
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                if relative not in excluded:
                    pending.append(path)
            elif path.is_file():
                files[relative] = _ref(path)["sha256"]
    return dict(sorted(files.items()))


def prepare(source, run, policy, context):
    source, run = Path(source).resolve(strict=True), Path(run).resolve(strict=True)
    require(not run.is_relative_to(source) and not source.is_relative_to(run), "OUTPUT_OVERLAPS_SOURCE")
    bind_policy(source, policy, context["source_inputs"])
    excluded = list(policy["compiler_output_roots"])
    if (source / ".git").is_dir():
        excluded.append(".git")
    before = inventory(source, excluded)
    destination = run / "diagnostics"
    destination.mkdir(exist_ok=False)
    expectation_root = source / policy["expectation_root"]
    copies, received = {}, set()
    for relative, digest in before.items():
        path = source / relative
        if path.is_relative_to(expectation_root) and ".verified." in path.name:
            output_relative = path.relative_to(expectation_root).as_posix()
            target = destination / output_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(_ref(target)["sha256"] == digest, "EXPECTED_COPY_CHANGED")
            copies[output_relative] = {"origin_ref": _ref(path), "copy_ref": _ref(target)}
            for marker in policy["received_markers"]:
                received.add(str(PurePosixPath(output_relative).with_name(path.name.replace(".verified.", marker))))
    reference = _save(run / "diagnostic-preparation.json", {
        "schema_version": "repository_diagnostic_preparation_v1", "context": context, "policy": policy,
        "diagnostic_root": str(destination), "source_inventory": before, "excluded_source_roots": excluded,
        "expected_copies": copies, "allowed_received": sorted(received)})
    return reference, {"REPOFORGE_DIAGNOSTIC_ROOT": str(destination),
        "REPOFORGE_EXPECTATION_ROOT": str(expectation_root)}


def validate(preparation_ref, context):
    preparation = read_bound_json(preparation_ref)
    require(preparation.get("schema_version") == "repository_diagnostic_preparation_v1"
        and preparation["context"] == context, "PREPARATION_CHANGED")
    root = Path(preparation_ref["path"]).resolve().parent
    destination = root / "diagnostics"
    require(preparation["diagnostic_root"] == str(destination), "OUTPUT_ROOT_CHANGED")
    source = Path(context["source_root"])
    policy = read_bound_json(context["profile_ref"]).get("diagnostics")
    require(preparation["policy"] == policy, "POLICY_CHANGED")
    bind_policy(source, policy, context["source_inputs"])
    declared = list(policy["compiler_output_roots"])
    require(preparation["excluded_source_roots"] in (declared, [*declared, ".git"]), "SOURCE_EXCLUSIONS_CHANGED")
    require(inventory(source, preparation["excluded_source_roots"]) == preparation["source_inventory"],
        "SOURCE_TEST_EXPECTATION_OR_NEW_FILE_CHANGED")
    observed = inventory(destination)
    expected = {PurePosixPath(row["path"]).relative_to(policy["expectation_root"]).as_posix(): row
        for row in context["source_inputs"]
        if PurePosixPath(row["path"]).is_relative_to(PurePosixPath(policy["expectation_root"]))
        and ".verified." in PurePosixPath(row["path"]).name}
    require(set(preparation["expected_copies"]) == set(expected), "EXPECTED_COPY_SET_CHANGED")
    received = {str(PurePosixPath(relative).with_name(PurePosixPath(relative).name.replace(".verified.", marker)))
        for relative in expected for marker in policy["received_markers"]}
    require(preparation["allowed_received"] == sorted(received), "RECEIVED_ALLOWLIST_CHANGED")
    for relative, pair in preparation["expected_copies"].items():
        require(Path(pair["origin_ref"]["path"]).resolve() == source / expected[relative]["path"]
            and pair["origin_ref"]["sha256"] == expected[relative]["sha256"]
            and _ref(pair["origin_ref"]["path"]) == pair["origin_ref"]
            and _ref(pair["copy_ref"]["path"]) == pair["copy_ref"]
            and pair["origin_ref"]["sha256"] == pair["copy_ref"]["sha256"]
            and Path(pair["copy_ref"]["path"]).resolve() == destination / relative, "EXPECTED_COPY_CHANGED")
    extra = set(observed) - set(preparation["expected_copies"])
    require(extra <= set(preparation["allowed_received"]), "UNDECLARED_DIAGNOSTIC_FILE")
    return {"schema_version": "repository_diagnostic_validation_v1", "status": "validated",
        "context": context, "preparation_ref": preparation_ref, "diagnostic_root": str(destination),
        "expected_copies": preparation["expected_copies"],
        "received_refs": [_ref(destination / relative) for relative in sorted(extra)],
        "all_source_test_and_expectation_bytes_unchanged": True}


def validate_receipt(reference, context, *, environment):
    saved = read_bound_json(reference)
    require(Path(reference["path"]).name == "diagnostic-validation.json", "RECEIPT_LOCATION_CHANGED")
    root = Path(reference["path"]).resolve().parent
    require(Path(saved["preparation_ref"]["path"]).resolve() == root / "diagnostic-preparation.json",
        "PREPARATION_LOCATION_CHANGED")
    current = validate(saved["preparation_ref"], context)
    require(saved == current, "RECEIPT_CHANGED")
    policy = read_bound_json(context["profile_ref"])["diagnostics"]
    require(isinstance(environment, dict)
        and environment.get("REPOFORGE_DIAGNOSTIC_ROOT") == str(root / "diagnostics")
        and environment.get("REPOFORGE_EXPECTATION_ROOT") == str(Path(context["source_root"]) / policy["expectation_root"]),
        "EXECUTION_ENVIRONMENT_CHANGED")
    return saved
