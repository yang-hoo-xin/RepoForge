"""Revalidate qualification process originals; controlled transport, no Docker."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.domain import repository_probe_runner as module
from src.mod_agent.domain.full_project_compilation import _save
from src.mod_agent.model_qualification import read_bound_json
from tests.unit.test_repository_probe_runner import bundle, owner


@pytest.fixture
def qualified(tmp_path, monkeypatch):
    selected = owner(tmp_path)
    environment = {"engine_id": "controlled-engine", "server_version": "controlled", "image_id": "controlled-image",
        "image_environment": [], "evidence_refs": []}
    monkeypatch.setattr(selected, "_environment", lambda: (["controlled-docker", "--context", "default"], environment))
    observed = {"candidate_value": 7, "uid": 65534, "source_readonly": True, "assembly_readonly": True,
        "root_readonly": True, "network_denied": True, "host_canary_absent": True,
        "process_status": "NoNewPrivs:\t1\nCapEff:\t00000000\n", "pids_max": "64",
        "memory_max": "1073741824", "cpu_max": "100000 100000"}
    files = {"logs/status.json": json.dumps({"restore_exit_code": 0, "publish_exit_code": 0}),
        "logs/sdk-version.txt": "8.0.100", "logs/restore.txt": "controlled restore", "logs/publish.txt": "controlled compile",
        "logs/compile-inputs.txt": "/work/repository/Library/Library.csproj|/work/repository/Library/Value.cs\n",
        "artifacts/RepoForge.Probe.dll": "controlled assembly", "artifacts/RepoForge.Probe.deps.json":
            json.dumps({"libraries": {"RepoForgeQualificationTarget/1.0": {"type": "project"}}})}
    calls = []
    def execute(*, root, docker, environment, image, mounts, runtime):
        calls.append(runtime)
        root.mkdir()
        token, container = "qualification-token", ("b" if runtime else "c") * 64
        entrypoint, command = ("dotnet", ["/assembly/RepoForge.Probe.dll"]) if runtime else ("/bin/sh", ["-c", module._BUILD_SCRIPT])
        policy = module._policy(runtime)
        document = {"Image": environment["image_id"], "State": {"Running": False, "ExitCode": 0},
            "Config": {"Image": image, "User": "65534:65534", "Entrypoint": [entrypoint], "Cmd": command,
                "Labels": {"codex.repository-probe": token}, "Env": list(module._ENV)},
            "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False, "CapDrop": ["ALL"],
                "CapAdd": None, "SecurityOpt": ["no-new-privileges"], "PidsLimit": policy["pids"], "Memory": policy["memory"],
                "MemorySwap": policy["memory"], "NanoCpus": policy["nano_cpus"], "Tmpfs": policy["tmpfs"],
                "Init": True, "Devices": None, "VolumesFrom": None, "PidMode": "", "LogConfig": {"Type": "none"}},
            "Mounts": [{"Destination": dest, "Type": "bind", "Source": str(path), "RW": False} for dest, path in mounts.items()]}
        created = _save(root / "create.json", {"argv": [*docker, "create", "--pull=never", "--label=codex.repository-probe=" + token],
            "exit_code": 0, "stdout": container, "stderr": ""})
        refs = {name: _save(root / (name + ".json"), {"argv": [*docker, "inspect", container], "exit_code": 0,
            "stdout": json.dumps([document]), "stderr": ""}) for name in ("inspect-before", "inspect-after")}
        _save(root / "cleanup.json", {"argv": [*docker, "rm", "--force", container], "exit_code": 0, "stdout": "", "stderr": ""})
        stdout = json.dumps(observed) if runtime else bundle(files)
        execution = {"completed": True, "policy_verified": True, "runtime": runtime, "exit_code": 0,
            "timed_out": False, "output_limit_exceeded": False, "local_process_reaped": True,
            "container_id": container, "container_state": document["State"], "image": image,
            "image_id": environment["image_id"], "stdout": stdout, "stderr": "",
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(), "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "create_ref": created, "inspect_before_ref": refs["inspect-before"], "inspect_after_ref": refs["inspect-after"]}
        return {**execution, "receipt_ref": _save(root / "execution.json", execution)}
    monkeypatch.setattr(module, "execute_sdk_container", execute)
    profile = "v2/probe-profile/controlled.json"
    capability = selected.verify_capability(profile)
    assert capability["status"] == "qualified", capability
    assert calls == [False, True]
    return selected, environment, profile, capability, calls


def reuse(selected, environment, reference, monkeypatch):
    data = deepcopy(selected.backend.profile.data)
    data["probe_runner"]["qualification_ref"] = reference
    backend = SimpleNamespace(profile=SimpleNamespace(data=data, reference={"path": "new-profile", "sha256": "d" * 64}))
    replacement = module.RepositoryProbeRunner(state=selected.state, backend=backend)
    monkeypatch.setattr(replacement, "_environment", lambda: (["controlled-docker", "--context", "default"], environment))
    monkeypatch.setattr(replacement, "_qualify", lambda *args: pytest.fail("Saved proof must not silently launch qualification"))
    return replacement


def test_qualification_reference_is_not_a_recursive_profile_identity(qualified, monkeypatch):
    selected, environment, profile, capability, calls = qualified
    replacement = reuse(selected, environment, capability["evidence_refs"][0], monkeypatch)
    assert replacement.backend.profile.reference != selected.backend.profile.reference
    assert replacement.identity(environment) == selected.identity(environment)
    assert replacement.verify_capability(profile)["status"] == "qualified"
    assert selected.verify_capability(profile)["status"] == "qualified"
    assert calls == [False, True]
    replacement.backend.profile.data["configuration"] = "Debug"
    assert replacement.verify_capability(profile)["status"] == "unavailable"


@pytest.mark.parametrize("change,reason", [
    ({"execution_refs": []}, "QUALIFICATION_EXECUTION_REFS_REQUIRED"),
    ({"qualification_candidate": "model_claim"}, "QUALIFICATION_IDENTITY_CHANGED"),
    ({"controls": {"source_readonly": False}}, "QUALIFICATION_BOUNDARY_FAILED"),
])
def test_qualified_summary_without_matching_control_originals_is_rejected(qualified, monkeypatch, change, reason):
    selected, environment, profile, capability, _ = qualified
    reference = capability["evidence_refs"][0]
    proof = {**read_bound_json(reference), **change}
    changed = _save(Path(reference["path"]).with_name("changed-proof.json"), proof)
    observed = reuse(selected, environment, changed, monkeypatch).verify_capability(profile)
    assert observed["status"] == "unavailable" and reason in observed["reason"]


@pytest.mark.parametrize("fault,reason", [("network", "CONTAINER_POLICY_MISMATCH"), ("exit", "QUALIFICATION_EXECUTION_NOT_SUCCESSFUL")])
def test_policy_verified_flag_does_not_override_actual_inspect_or_exit(qualified, monkeypatch, fault, reason):
    selected, environment, profile, capability, _ = qualified
    proof = read_bound_json(capability["evidence_refs"][0])
    attempt = read_bound_json(proof["qualification_attempt_ref"])
    captured = attempt["result"]["runtime_execution"]
    root = Path(captured["receipt_ref"]["path"]).parent
    if fault == "network":
        inspected = read_bound_json(captured["inspect_before_ref"])
        document = json.loads(inspected["stdout"])
        document[0]["HostConfig"]["NetworkMode"] = "host"
        captured["inspect_before_ref"] = _save(root / "changed-inspect.json", {**inspected, "stdout": json.dumps(document)})
    else:
        captured["exit_code"] = 7
    captured["receipt_ref"] = _save(root / "changed-execution.json", {key: value for key, value in captured.items() if key != "receipt_ref"})
    proof["execution_refs"][1] = captured["receipt_ref"]
    proof["qualification_attempt_ref"] = _save(root.parent / "changed-attempt.json", attempt)
    changed = _save(root.parent / "changed-proof.json", proof)
    observed = reuse(selected, environment, changed, monkeypatch).verify_capability(profile)
    assert observed["status"] == "unavailable" and reason in observed["reason"]


def test_cached_qualification_rechecks_immutable_proof(qualified):
    selected, _, profile, capability, calls = qualified
    path = Path(capability["evidence_refs"][0]["path"])
    path.write_text("{}")
    assert selected.verify_capability(profile)["status"] == "unavailable"
    assert calls == [False, True]
