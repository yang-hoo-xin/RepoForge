"""Optional shared build-attempt cap at the existing subprocess boundary.

This is resource admission, not a compiler/verifier replacement or source authority.
为什么要有它：多个正常入口子进程共享批准的构建额度，必须在启动前预留，失败和重启不能重领。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

ENVIRONMENT_KEY = "REPOFORGE_BUILD_BUDGET_REF"
_INSTALLED = None


class BuildBudgetError(ValueError):
    pass


def _raw(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _local(path):
    selected = Path(path)
    if not selected.is_absolute() or str(selected).startswith(("\\\\", "//")):
        raise BuildBudgetError("BUILD_BUDGET_LOCAL_ABSOLUTE_PATH_REQUIRED")
    for parent in (selected, *selected.parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if parent.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise BuildBudgetError("BUILD_BUDGET_LINK_FORBIDDEN")
    return selected.resolve()


def _read(reference):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", reference["sha256"])):
        raise BuildBudgetError("BUILD_BUDGET_REFERENCE_INVALID")
    path = _local(reference["path"])
    if path.stat().st_size > 262144:
        raise BuildBudgetError("BUILD_BUDGET_CONFIGURATION_TOO_LARGE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise BuildBudgetError("BUILD_BUDGET_CONFIGURATION_CHANGED")
    return json.loads(raw)


def _reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_build_budget(authorization_ref):
    """Reuse a stable, write-once ledger beside the explicit authorization.

    为什么要有它：改变输出目录不能产生新的总额度；此文件仅约束资源，不授予Apply或执行权限。
    """
    authorization = _read(authorization_ref)
    maximum = authorization.get("shared_build_limit")
    if type(maximum) is not int or not 1 <= maximum <= 1000:
        raise BuildBudgetError("BUILD_BUDGET_LIMIT_INVALID")
    parent = _local(authorization_ref["path"]).parent
    ledger = _local(parent / "runtime-ledger")
    ledger.mkdir(exist_ok=True)
    selected = _local(ledger / "build-budget.json")
    value = {"schema_version": "shared-build-attempt-budget.v1", "maximum": maximum,
             "authorization_ref": authorization_ref, "count_basis": "reserved_compiler_process_attempts_not_successes"}
    raw = _raw(value)
    try:
        with selected.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if selected.read_bytes() != raw:
            raise BuildBudgetError("BUILD_BUDGET_EXISTING_GRANT_DIFFERS") from None
    return _reference(selected)


def prepare_build_meter(origin_ref, *, ledger_root):
    """Record compiler counts; retain all actual subprocess timeout protections.

    为什么要有它：2026-10-20用户撤销实验次数配额，构建次数仅用于计量，不充当源码权限或执行审批。
    """
    _read(origin_ref)
    ledger = _local(ledger_root)
    ledger.mkdir(parents=True, exist_ok=True)
    selected = _local(ledger / "build-budget.json")
    raw = _raw({"schema_version": "shared-build-attempt-budget.v1",
        "resource_policy": "meter_only", "maximum": None, "origin_ref": origin_ref,
        "count_basis": "reserved_compiler_process_attempts_not_successes"})
    try:
        with selected.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if selected.read_bytes() != raw:
            raise BuildBudgetError("BUILD_METER_EXISTING_ORIGIN_DIFFERS") from None
    return _reference(selected)


def compiler_command(executable, arguments):
    """Recognize direct .NET build/MSBuild and compiler launches used by the product."""
    if isinstance(arguments, (list, tuple)):
        words = [os.fsdecode(part) for part in arguments]
    elif isinstance(arguments, (str, bytes)):
        words = [part.strip('"') for part in shlex.split(os.fsdecode(arguments), posix=False)]
    else:
        return None
    if not words:
        return None
    name = os.fsdecode(executable or words[0]).strip('"').replace("\\", "/").rsplit("/", 1)[-1].lower()
    args = [part.lower().strip('"') for part in words[1:]]
    if name in {"msbuild", "msbuild.exe", "csc", "csc.exe", "vbc", "vbc.exe"}:
        return words
    if name in {"dotnet", "dotnet.exe"}:
        if args and args[0] in {"build", "msbuild", "publish", "pack"}:
            return words
        if args and args[0].replace("\\", "/").rsplit("/", 1)[-1] in {"csc.dll", "vbc.dll", "msbuild.dll"}:
            return words
        if args and args[0] == "exec" and any(part.replace("\\", "/").rsplit("/", 1)[-1]
                in {"csc.dll", "vbc.dll", "msbuild.dll"} for part in args[1:]):
            return words
    return None


class SharedBuildBudget:
    def __init__(self, reference):
        self.reference = dict(reference)
        self.configuration = _read(self.reference)
        self.path = _local(reference["path"])
        config = self.configuration
        self.meter_only = config.get("resource_policy") == "meter_only"
        if self.meter_only:
            if config.get("schema_version") != "shared-build-attempt-budget.v1" or config.get("maximum") is not None:
                raise BuildBudgetError("BUILD_METER_CONFIGURATION_INVALID")
            _read(config["origin_ref"])
            return
        if (config.get("schema_version") != "shared-build-attempt-budget.v1"
                or type(config.get("maximum")) is not int or not 1 <= config["maximum"] <= 1000):
            raise BuildBudgetError("BUILD_BUDGET_CONFIGURATION_INVALID")
        authorization = _read(config["authorization_ref"])
        if authorization.get("shared_build_limit") != config["maximum"]:
            raise BuildBudgetError("BUILD_BUDGET_AUTHORIZED_LIMIT_CHANGED")
        expected = _local(config["authorization_ref"]["path"]).parent / "runtime-ledger" / "build-budget.json"
        if self.path != expected:
            raise BuildBudgetError("BUILD_BUDGET_LEDGER_LOCATION_CHANGED")

    def reserve(self, command, cwd=None):
        # Scope: immutable slots; failed launches are conservatively still consumed.
        # 为什么要有它：预留后崩溃存在执行不确定性，不能清零或删除槽位来重新获得构建次数。
        if _read(self.reference) != self.configuration:
            raise BuildBudgetError("BUILD_BUDGET_CONFIGURATION_CHANGED")
        if not self.meter_only and _read(self.configuration["authorization_ref"]).get("shared_build_limit") != self.configuration["maximum"]:
            raise BuildBudgetError("BUILD_BUDGET_AUTHORIZED_LIMIT_CHANGED")
        ordinals = count(1) if self.meter_only else range(1, self.configuration["maximum"] + 1)
        for ordinal in ordinals:
            target = _local(self.path.parent / f"build-{ordinal:04d}.json")
            value = {"ordinal": ordinal, "budget_ref": self.reference, "process_id": os.getpid(),
                     "reserved_at": datetime.now(timezone.utc).isoformat(),
                     "command_sha256": hashlib.sha256(_raw(command)).hexdigest(),
                     "working_directory": None if cwd is None else os.fsdecode(cwd),
                     "status": "reserved_before_process_spawn_outcome_not_inferred"}
            try:
                with target.open("xb") as stream:
                    stream.write(_raw(value))
                return _reference(target)
            except FileExistsError:
                continue
        raise BuildBudgetError("BUILD_BUDGET_EXHAUSTED")

    def audit(self, event, args):
        if event != "subprocess.Popen":
            return
        command = compiler_command(args[0], args[1])
        if command is not None:
            self.reserve(command, args[2])

    def snapshot(self):
        reservations = []
        paths = (sorted(path for path in self.path.parent.glob("build-*.json")
            if re.fullmatch(r"build-\d{4,}\.json", path.name)) if self.meter_only else
            (self.path.parent / f"build-{ordinal:04d}.json" for ordinal in range(1, self.configuration["maximum"] + 1)))
        for selected in paths:
            path = _local(selected)
            if path.exists():
                reservations.append(_reference(path))
        return {"budget_ref": self.reference, "maximum": self.configuration["maximum"],
                "reserved": len(reservations),
                "remaining": None if self.meter_only else self.configuration["maximum"] - len(reservations),
                "reservations": reservations, "count_basis": self.configuration["count_basis"]}


def install_build_budget_from_environment():
    """Enable only the host-supplied resource cap; ordinary execution is unchanged."""
    global _INSTALLED
    encoded = os.environ.get(ENVIRONMENT_KEY)
    if not encoded:
        return None
    reference = json.loads(encoded)
    if _INSTALLED is not None:
        if _INSTALLED.reference != reference:
            raise BuildBudgetError("BUILD_BUDGET_PROCESS_GRANT_CHANGED")
        return _INSTALLED
    budget = SharedBuildBudget(reference)
    sys.addaudithook(budget.audit)
    _INSTALLED = budget
    return budget
