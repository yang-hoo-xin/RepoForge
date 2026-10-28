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


def kernel_observation(version, *, group="/docker/subject", mount_root="/docker/subject", cpu_mount="cpu"):
    values = {"pids.max": "64\n", "memory.max": "1073741824\n", "memory.limit_in_bytes": "1073741824\n",
        "cpu.max": "100000 100000\n", "cpu.cfs_quota_us": "100000\n", "cpu.cfs_period_us": "100000\n"}
    suffix = "" if group in {mount_root, "/"} else group.removeprefix(mount_root).lstrip("/")
    suffix = "/" + suffix if suffix else ""
    if version == "2":
        membership = "0::" + group + "\n"
        mounts = f"29 23 0:26 {mount_root} /sys/fs/cgroup ro - cgroup2 cgroup rw\n"
        files = {"/sys/fs/cgroup" + suffix + "/" + name: values[name] for name in ("pids.max", "memory.max", "cpu.max")}
    else:
        # 为什么要有它：实际宿主有0::条目却仅挂载v1；必须按controller定位，不能把条目当v2资格。
        membership = f"12:pids:{group}\n8:memory:{group}\n5:cpu,cpuacct:{group}\n0::{group}\n"
        mounts, files = "", {}
        for controller, destination, names in (("pids", "pids", ["pids.max"]),
            ("memory", "memory", ["memory.limit_in_bytes"]), ("cpu,cpuacct", cpu_mount, ["cpu.cfs_quota_us", "cpu.cfs_period_us"])):
            mounts += f"29 23 0:26 {mount_root} /sys/fs/cgroup/{destination} ro - cgroup cgroup rw,{controller}\n"
            files.update({"/sys/fs/cgroup/" + destination + suffix + "/" + name: values[name] for name in names})
    return {"proc_self_cgroup": membership, "proc_self_mountinfo": mounts,
        "versions": dict.fromkeys(("pids", "memory", "cpu"), version), "files": files}


@pytest.fixture(params=["1", "2"], ids=["cgroup-v1", "cgroup-v2"])
def qualified(tmp_path, monkeypatch, request):
    selected = owner(tmp_path)
    environment = {"engine_id": "controlled-engine", "server_version": "controlled", "image_id": "controlled-image",
        "image_environment": [], "evidence_refs": []}
    monkeypatch.setattr(selected, "_environment", lambda: (["controlled-docker", "--context", "default"], environment))
    observed = {"candidate_value": 7, "uid": 65534, "source_readonly": True, "assembly_readonly": True,
        "root_readonly": True, "network_denied": True, "host_canary_absent": True,
        "process_status": "NoNewPrivs:\t1\nCapEff:\t00000000\n", "pids_max": "64",
        "memory_max": "1073741824", "cpu_max": "100000 100000", "cgroup_observation": kernel_observation(request.param)}
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


def test_new_runner_recovers_same_qualification_workspace_without_repeating_containers(qualified, monkeypatch):
    selected, environment, profile, capability, calls = qualified
    replacement = module.RepositoryProbeRunner(state=selected.state, backend=selected.backend)
    monkeypatch.setattr(replacement, "_environment", lambda: (
        ["controlled-docker", "--context", "default"], environment))
    recovered = replacement.verify_capability(profile)
    assert recovered == capability
    assert calls == [False, True]


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


@pytest.mark.parametrize("version,group,mount_root,cpu_mount", [
    ("1", "/docker/subject", "/docker/subject", "cpu"),
    ("1", "/docker/subject/child", "/docker/subject", "cpu,cpuacct"),
    ("1", "/", "/", "cpu"),
    ("2", "/docker/subject", "/", "unused"),
    ("2", "/", "/", "unused"),
])
def test_kernel_limits_resolve_current_membership_and_mounted_controller(version, group, mount_root, cpu_mount):
    observation = kernel_observation(version, group=group, mount_root=mount_root, cpu_mount=cpu_mount)
    observed = {"pids_max": "64", "memory_max": "1073741824", "cpu_max": "100000 100000",
        "cgroup_observation": observation}
    module._verify_cgroup_controls(observed)
    assert module._cgroup_control_values(observation) == {key: observed[key] for key in ("pids_max", "memory_max", "cpu_max")}


def test_actual_v1_layout_with_unified_membership_does_not_claim_v2():
    observation = kernel_observation("1")
    observation["proc_self_cgroup"] = observation["proc_self_cgroup"].replace("5:cpu,cpuacct:", "2:cpu:") + "3:cpuacct:/docker/subject\n"
    observation["proc_self_mountinfo"] = observation["proc_self_mountinfo"].replace("rw,cpu,cpuacct", "rw,cpu")
    values = module._cgroup_control_values(observation)
    assert values["cpu_max"] == "100000 100000"
    assert module._cgroup_location("cpu", observation["proc_self_cgroup"], observation["proc_self_mountinfo"]) == (
        "/sys/fs/cgroup/cpu", "1")


@pytest.mark.parametrize("version", ["1", "2"])
def test_namespace_root_cannot_use_limits_from_an_unrelated_mounted_subtree(version):
    # 为什么要有它：即使子树限额恰好相同，也不能用它代替当前namespace根的实际资源边界。
    observation = kernel_observation(version, group="/", mount_root="/docker/unrelated")
    with pytest.raises(ValueError, match="CGROUP_MOUNT_INVALID"):
        module._cgroup_control_values(observation)


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize("fault", ["missing", "malformed", "unlimited", "wrong", "wrong_cpu_period", "reported_value", "version", "unselected_path"])
def test_kernel_resource_limits_cannot_be_missing_or_replaced_by_claims(version, fault):
    observation = kernel_observation(version)
    observed = {"pids_max": "64", "memory_max": "1073741824", "cpu_max": "100000 100000",
        "cgroup_observation": observation}
    pids = next(path for path in observation["files"] if path.endswith("pids.max"))
    if fault == "missing":
        observation["files"].pop(pids)
    elif fault in {"malformed", "unlimited", "wrong"}:
        observation["files"][pids] = {"malformed": "64 garbage", "unlimited": "max", "wrong": "128"}[fault]
    elif fault == "wrong_cpu_period":
        cpu = next(path for path in observation["files"] if path.endswith("cpu.cfs_period_us" if version == "1" else "cpu.max"))
        observation["files"][cpu] = "200000" if version == "1" else "100000 200000"
    elif fault == "reported_value":
        observed["pids_max"] = "128"
    elif fault == "version":
        observation["versions"]["pids"] = "2" if version == "1" else "1"
    else:
        observation["files"]["/unrelated/pids.max"] = "64"
    with pytest.raises(ValueError, match="CGROUP_|QUALIFICATION_BOUNDARY_FAILED"):
        module._verify_cgroup_controls(observed)


@pytest.mark.parametrize("fault", ["no_membership", "no_mount", "ambiguous", "traversal"])
def test_unresolved_kernel_hierarchy_is_rejected(fault):
    observation = kernel_observation("1")
    if fault == "no_membership":
        observation["proc_self_cgroup"] = "8:memory:/docker/subject\n"
    elif fault == "no_mount":
        observation["proc_self_mountinfo"] = ""
    elif fault == "ambiguous":
        observation["proc_self_mountinfo"] += "30 23 0:26 /docker/subject /other ro - cgroup cgroup rw,pids\n"
    else:
        observation["proc_self_cgroup"] = observation["proc_self_cgroup"].replace("/docker/subject", "/../subject")
    with pytest.raises(ValueError, match="CGROUP_"):
        module._cgroup_control_values(observation)
