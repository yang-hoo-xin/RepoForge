"""Shared attempt limits reserve before compiler launch, without running a compiler.

为什么要有它：模型次数与编译次数是不同资源，跨入口/重启不能多领构建额度。
"""
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.mod_agent.build.execution_budget import (
    BuildBudgetError,
    SharedBuildBudget,
    compiler_command,
    prepare_build_budget,
    prepare_build_meter,
)


def grant(tmp_path, limit=2):
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps({"shared_build_limit": limit}), encoding="utf-8")
    source = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return source, prepare_build_budget(source)


def test_reservations_are_shared_across_owners_and_preparation_does_not_reset(tmp_path):
    source, reference = grant(tmp_path)
    first, second = SharedBuildBudget(reference), SharedBuildBudget(reference)
    first.reserve(["dotnet", "build"], str(tmp_path))
    assert prepare_build_budget(source) == reference
    second.reserve(["dotnet", "exec", "csc.dll"], str(tmp_path))
    assert first.snapshot()["reserved"] == 2 and second.snapshot()["remaining"] == 0
    with pytest.raises(BuildBudgetError, match="EXHAUSTED"):
        SharedBuildBudget(reference).reserve(["dotnet", "build"])


def test_parallel_claims_cannot_overshoot_the_grant(tmp_path):
    _source, reference = grant(tmp_path, 3)

    def reserve(_index):
        try:
            return SharedBuildBudget(reference).reserve(["dotnet", "build"])
        except BuildBudgetError as error:
            assert str(error) == "BUILD_BUDGET_EXHAUSTED"
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reserve, range(16)))
    assert sum(result is not None for result in results) == 3
    assert SharedBuildBudget(reference).snapshot()["reserved"] == 3


@pytest.mark.parametrize("arguments", [
    ["dotnet", "build", "Example.csproj"],
    ["dotnet", "exec", "C:/sdk/Roslyn/bincore/csc.dll", "@response.rsp"],
    ["dotnet", "C:/sdk/Roslyn/bincore/csc.dll", "@response.rsp"],
    ["dotnet", "msbuild"],
    ["MSBuild.exe", "Example.csproj"],
    '"C:\\Program Files\\dotnet\\dotnet.exe" build "C:\\example project\\a.csproj"',
])
def test_recognizes_product_compiler_commands(arguments):
    assert compiler_command(None, arguments)


@pytest.mark.parametrize("arguments", [["dotnet", "restore"], ["dotnet", "--version"],
                                      ["dotnet", "Probe.dll"], ["python", "script.py"]])
def test_runtime_and_environment_queries_are_not_build_attempts(arguments):
    assert compiler_command(None, arguments) is None


def test_changed_authorization_and_relocated_ledger_fail_closed(tmp_path):
    source, reference = grant(tmp_path)
    budget = SharedBuildBudget(reference)
    Path(source["path"]).write_text('{"shared_build_limit":100}', encoding="utf-8")
    with pytest.raises(BuildBudgetError, match="CONFIGURATION_CHANGED"):
        budget.reserve(["dotnet", "build"])
    assert budget.snapshot()["reserved"] == 0


def test_real_audit_hook_refuses_before_os_spawn_without_a_compiler(tmp_path):
    _source, reference = grant(tmp_path, 1)
    program = """
import json, os, subprocess, sys
from src.mod_agent.build.execution_budget import ENVIRONMENT_KEY, install_build_budget_from_environment, BuildBudgetError
os.environ[ENVIRONMENT_KEY] = sys.argv[1]
budget = install_build_budget_from_environment()
assert install_build_budget_from_environment() is budget
budget.reserve(['dotnet', 'build'])
try:
    subprocess.run([sys.argv[2], 'build'], check=False)
except BuildBudgetError as error:
    assert str(error) == 'BUILD_BUDGET_EXHAUSTED'
else:
    raise AssertionError('Compiler subprocess must not start')
print(json.dumps(budget.snapshot()))
"""
    result = subprocess.run([sys.executable, "-B", "-c", program, json.dumps(reference),
                             str(tmp_path / "nonexistent" / "dotnet.exe")],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["reserved"] == 1


def test_meter_counts_beyond_old_eight_build_quota_without_authorization(tmp_path):
    source = tmp_path / "protocol.json"
    source.write_text('{"resource_policy":"meter_only"}', encoding="utf-8")
    origin = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    reference = prepare_build_meter(origin, ledger_root=tmp_path / "attempt/runtime-ledger")
    first, second = SharedBuildBudget(reference), SharedBuildBudget(reference)
    with ThreadPoolExecutor(max_workers=4) as executor:
        reservations = list(executor.map(lambda _: second.reserve(["dotnet", "build"]), range(12)))
    assert len({value["path"] for value in reservations}) == 12
    assert prepare_build_meter(origin, ledger_root=tmp_path / "attempt/runtime-ledger") == reference
    result = first.snapshot()
    assert result["reserved"] == 12
    assert result["maximum"] is None and result["remaining"] is None
    assert first.configuration["origin_ref"] == origin
    assert "authorization_ref" not in first.configuration


def test_meter_new_attempt_does_not_reset_or_replace_previous_records(tmp_path):
    source = tmp_path / "protocol.json"
    source.write_text('{"resource_policy":"meter_only"}', encoding="utf-8")
    origin = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    old = SharedBuildBudget(prepare_build_meter(origin, ledger_root=tmp_path / "first/runtime-ledger"))
    old.reserve(["dotnet", "build"])
    before = {row["path"]: row["sha256"] for row in old.snapshot()["reservations"]}
    current = SharedBuildBudget(prepare_build_meter(origin, ledger_root=tmp_path / "second/runtime-ledger"))
    current.reserve(["dotnet", "build"])
    assert current.snapshot()["reserved"] == old.snapshot()["reserved"] == 1
    assert {row["path"]: row["sha256"] for row in old.snapshot()["reservations"]} == before
