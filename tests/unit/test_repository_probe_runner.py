"""Controlled SDK transport tests; not evidence of actual container isolation."""
import base64
import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from src.mod_agent.domain import repository_probe_runner as runner


def config():
    return {"kind": "docker_sdk_project_v1", "image": "mcr.microsoft.com/dotnet/sdk@sha256:" + "a" * 64,
        "sdk_version": "8.0.100", "target_framework": "net8.0", "project_path": "Library/Library.csproj"}


@pytest.mark.parametrize("endpoint,expected", [("npipe:////./pipe/dockerDesktopLinuxEngine", True),
    ("unix:///var/run/docker.sock", True), ("npipe:////remote-host/pipe/docker", False),
    ("unix://remote-host/run/docker.sock", False), ("unix:////remote-host/run/docker.sock", False),
    ("ssh://host", False), ("tcp://127.0.0.1:2375", False), ("unix:///run/docker.sock?host=other", False)])
def test_docker_endpoint_is_local_not_just_a_local_protocol(endpoint, expected):
    assert runner._local_docker_endpoint(endpoint) is expected


def owner(tmp_path):
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact

    source = tmp_path / "state-source"
    source.mkdir()
    (source / "A.cs").write_text("class A {}", encoding="utf-8")
    store = tmp_path / "state"
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": ["A.cs"], "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "repository-probe-runner-fixture", source,
        allowed_edit_paths=["A.cs"], authority_ref=authority)
    return runner.RepositoryProbeRunner(state=state,
        backend=SimpleNamespace(profile=SimpleNamespace(data={"probe_runner": config(), "configuration": "Release"},
            reference={"path": "controlled-profile", "sha256": "b" * 64})))


def _directory_link_or_reparse(path, target, monkeypatch):
    try:
        path.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        path.mkdir()
    original_lstat = Path.lstat

    class ReparseStat:
        def __init__(self, observed):
            self.observed = observed
            self.st_mode = observed.st_mode
            self.st_file_attributes = getattr(observed, "st_file_attributes", 0) | 0x400

        def __getattr__(self, name):
            return getattr(self.observed, name)

    def lstat(selected, *args, **kwargs):
        observed = original_lstat(selected, *args, **kwargs)
        return ReparseStat(observed) if Path(selected) == path else observed

    monkeypatch.setattr(Path, "lstat", lstat)


def bundle(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            if isinstance(content, tarfile.TarInfo):
                archive.addfile(content)
            else:
                content = content.encode()
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return "REPOFORGE_SDK_BUNDLE_V1\n" + base64.b64encode(stream.getvalue()).decode() + "\n"


@pytest.mark.parametrize("change", [{"image": "mcr.microsoft.com/dotnet/sdk:8.0"},
    {"project_path": "../other.csproj"}, {"docker_context": "remote"}, {"target_framework": "net8.0-windows"}])
def test_runner_configuration_never_admits_unpinned_or_remote_execution(change):
    with pytest.raises(ValueError):
        runner.validate_probe_runner_config({**config(), **change})


def test_repository_profile_loads_optional_runner_without_environment_actions(tmp_path, monkeypatch):
    from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile
    from tests.unit.test_csharp_repository_backend import _profile

    profile, source = _profile(tmp_path)
    data = {**profile.data, "probe_runner": config()}
    path = tmp_path / "with-probe.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runner.RepositoryProbeRunner, "_environment", lambda *_args: pytest.fail("Profile load must be read-only"))
    loaded = CSharpRepositoryProfile.load(path)
    assert loaded.data["probe_runner"] == config()
    loaded.verify_fixed_inputs(source, loaded.bind(source))
    data["probe_runner"]["image"] = "sdk:latest"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="PINNED_SDK_IMAGE_REQUIRED"):
        CSharpRepositoryProfile.load(path)


@pytest.mark.parametrize("name", ["../escape", "logs/../../escape", "/logs/absolute", "unknown/file", "logs/x:stream",
    "artifacts/NUL.dll", "artifacts/COM1", "logs/CON", "logs/output.", "logs/output ", "logs/a?b", "logs/LPT².txt"])
def test_build_archive_rejects_escaped_or_unexpected_paths(tmp_path, name):
    with pytest.raises(ValueError):
        runner.extract_build_bundle(bundle({name: "x"}), tmp_path / "unpack")


def test_build_archive_rejects_links_and_case_collisions(tmp_path):
    link = tarfile.TarInfo("artifacts/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../escape"
    with pytest.raises(ValueError, match="ARCHIVE_TYPE_DENIED"):
        runner.extract_build_bundle(bundle({"link": link}), tmp_path / "links")
    with pytest.raises(ValueError, match="ARCHIVE_PATH_COLLISION"):
        runner.extract_build_bundle(bundle({"logs/A": "a", "logs/a": "b"}), tmp_path / "case")


def test_harness_references_actual_project_and_accepts_only_probe_csharp(tmp_path):
    selected = owner(tmp_path)
    selected._harness(tmp_path / "harness", [{"relative_path": "probe/Program.cs", "content": "class Program {}"}])
    project = ElementTree.parse(tmp_path / "harness/RepoForge.Probe.csproj").getroot()
    assert project.find("ItemGroup/ProjectReference").get("Include") == "/work/repository/Library/Library.csproj"
    assert project.find("PropertyGroup/EnableDefaultCompileItems").text == "false"
    assert "https" not in (tmp_path / "harness/NuGet.Config").read_text()
    with pytest.raises(ValueError, match="PROBE_SOURCE_INVALID"):
        selected._harness(tmp_path / "bad-harness", [{"relative_path": "probe/overwrite.targets", "content": "x"}])


def test_missing_engine_is_unavailable_not_qualified(tmp_path, monkeypatch):
    selected = owner(tmp_path)
    monkeypatch.setattr(selected, "_environment", lambda: (_ for _ in ()).throw(ValueError("DOCKER_ENGINE_UNAVAILABLE")))
    monkeypatch.setattr(selected, "_qualify", lambda *args: pytest.fail("No qualification without engine"))
    capability = selected.verify_capability({"path": "profile", "sha256": "c" * 64})
    assert capability["status"] == "unavailable"
    assert capability["reason"] == "DOCKER_ENGINE_UNAVAILABLE"


def test_sdk_probe_materializes_selected_candidate_without_changing_original(tmp_path, monkeypatch):
    from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact
    from tests.unit.test_csharp_repository_backend import _backend, _profile
    from tests.unit.test_task_agent_v2_state import edit

    profile, source = _profile(tmp_path)
    profile_path = tmp_path / "probe-profile.json"
    profile_path.write_text(json.dumps({**profile.data, "probe_runner": config()}))
    profile = CSharpRepositoryProfile.load(profile_path)
    backend = _backend(tmp_path, profile, source)
    store = tmp_path / "state"
    allowed = profile.data["allowed_edit_paths"]
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": allowed, "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "probe-test", source, allowed_edit_paths=allowed, authority_ref=authority,
        contract_fields={"execution_profile_ref": profile.reference})
    original = (source / allowed[0]).read_bytes()
    candidate = edit(state, allowed[0], "=> value;", "=> value * 2;")
    selected = runner.RepositoryProbeRunner(state=state, backend=backend)
    monkeypatch.setattr(selected, "_environment", lambda: ([], {}))
    observed = []

    def execute(root, candidate_root, files, docker, environment, **_kwargs):
        observed.append((candidate_root / allowed[0]).read_text())
        assert (candidate_root / "Library/Library.csproj").read_bytes() == (source / "Library/Library.csproj").read_bytes()
        return {"source_manifest": runner._tree(candidate_root), "dependency_manifest": [], "compiled": False,
            "passed": False, "exit_code": 1, "runtime_execution": None, "compile_execution": {"stderr": "controlled failure"}}

    monkeypatch.setattr(selected, "_compile_and_run", execute)
    outcome = selected.run_probe(state=state, source_revision_ref=candidate["candidate_ref"],
        probe={"files": [{"relative_path": "probe/Program.cs", "content": "class Program {}"}]},
        capability={"runner_identity": selected.identity({}), "evidence_refs": []}, operation_id="controlled-probe")
    assert "=> value * 2;" in observed[0]
    assert outcome["source_identity"] == candidate["candidate_source_id"]
    assert outcome["passed"] is False and outcome["observations"]["compiled"] is False
    assert (source / allowed[0]).read_bytes() == original


def test_normal_probe_uses_short_recoverable_workspace_for_bound_packages(short_workspace, monkeypatch):
    """The experiment seam owns path shortening; callers do not choose a short task root."""
    from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile
    from src.mod_agent.model_qualification import canonical_hash
    from src.mod_agent.task_agent_v2_experiments import TaskAgentExperiments
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact
    from tests.unit.test_csharp_repository_backend import _backend, _profile

    profile, source = _profile(short_workspace)
    feed = short_workspace / "feed"
    feed.mkdir()
    long_name = "runtime.opensuse.13.2-x64.runtime.native.system.security.cryptography.openssl.4.3.0.nupkg"
    for index in range(197):
        name = long_name if index == 0 else f"package.{index:03d}.1.0.0.nupkg"
        (feed / name).write_bytes(f"package-{index}".encode())
    packages = runner.offline_package_manifest(feed)
    runner_config = {**config(), "offline_packages": {
        "path": str(feed), "manifest_sha256": canonical_hash(packages)}}
    profile_path = short_workspace / "probe-profile.json"
    profile_path.write_text(json.dumps({**profile.data, "probe_runner": runner_config}), encoding="utf-8")
    profile = CSharpRepositoryProfile.load(profile_path)
    backend = _backend(short_workspace, profile, source)

    # This durable task path is legal for content-addressed JSON, but the old
    # repository-probes/run-<uuid>/offline nesting makes the longest package
    # target exceed the reproduced Win32 failure length.
    store = short_workspace / ("e" * 64) / "state"
    allowed = profile.data["allowed_edit_paths"]
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": allowed, "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "short-workspace-probe", source, allowed_edit_paths=allowed,
        authority_ref=authority, contract_fields={"execution_profile_ref": profile.reference})

    container_calls = []
    build_files = {"logs/status.json": json.dumps({"restore_exit_code": 0, "publish_exit_code": 0}),
        "logs/sdk-version.txt": runner_config["sdk_version"], "logs/restore.txt": "restored",
        "logs/publish.txt": "built", "logs/compile-inputs.txt": "/work/repository/Library/Library.csproj|source.cs\n",
        "artifacts/RepoForge.Probe.dll": "controlled assembly", "artifacts/RepoForge.Probe.deps.json":
            json.dumps({"libraries": {"Library/1.0": {"type": "project"}}})}

    def execute(**kwargs):
        container_calls.append(kwargs)
        return {"completed": True, "exit_code": 0,
            "stdout": "{\"actual\":6}" if kwargs["runtime"] else bundle(build_files), "stderr": ""}

    monkeypatch.setattr(runner, "execute_sdk_container", execute)
    environment = {"image_id": "controlled-image", "image_environment": []}
    monkeypatch.setattr(runner.RepositoryProbeRunner, "_environment", lambda *_args: ([], deepcopy(environment)))

    selected = runner.RepositoryProbeRunner(state=state, backend=backend)
    experiments = TaskAgentExperiments(state=state, probe_runner=selected)
    capability_payload = {"status": "qualified", "profile_ref": experiments.probe_profile_ref,
        "execution_kind": "isolated", "runner_identity": selected.identity(environment),
        "boundaries": {"filesystem": "isolated", "network": "denied", "process": "isolated"}}
    capability_ref = state.save("probe-qualification", capability_payload)
    capability = {**capability_payload, "evidence_refs": [capability_ref]}
    monkeypatch.setattr(runner.RepositoryProbeRunner, "verify_capability", lambda *_args: deepcopy(capability))
    submitted = experiments.probe_submit(source_revision_ref=state.current_revision_ref,
        files=[{"relative_path": "Program.cs", "content": "class Program {}"}], entrypoint_id="csharp_console",
        rationale="Measure the candidate", expected_observation="The probe runs")
    first = experiments.experiment_run(source_revision_ref=state.current_revision_ref,
        probe_artifact_ref=submitted["probe_artifact_ref"], operation_id="short-workspace-operation")
    output = first["observations"]
    evidence = state.load(output["logs_ref"])
    mapping = state.load(evidence["execution_workspace_ref"])
    execution_root = Path(mapping["execution_root"])
    evidence_root = Path(mapping["evidence_root"])
    assert mapping["identity"] == {"schema": "task-agent-execution-workspace-identity.v2",
        "task_contract_ref": state.contract_ref, "kind": "probe", "operation_id": "short-workspace-operation",
        "source_revision_ref": state.current_revision_ref, "source_identity": output["source_identity"],
        "artifact_store_root": str(state.store_root)}
    assert not execution_root.is_relative_to(state.root)
    assert evidence_root.is_relative_to(state.root / "repository-probes")
    materialized = runner.offline_package_manifest(execution_root / "offline")
    assert len(materialized) == 197
    expected = {Path(row["path"]).name: row["sha256"] for row in packages}
    assert {row["path"]: row["sha256"] for row in materialized} == expected
    assert len(str(execution_root / "offline" / long_name)) < 248

    restarted = TaskAgentExperiments(state=state, probe_profile_ref=experiments.probe_profile_ref,
        probe_runner=runner.RepositoryProbeRunner(state=state, backend=backend))
    replay = restarted.experiment_run(source_revision_ref=state.current_revision_ref,
        probe_artifact_ref=submitted["probe_artifact_ref"], operation_id="short-workspace-operation")
    assert replay == first
    assert len(container_calls) == 2


def test_recovered_workspace_mapping_cannot_rebind_the_execution_root(tmp_path):
    from src.mod_agent.execution_workspace import ExecutionWorkspaceError

    selected = owner(tmp_path)
    manager = selected.execution_workspaces
    workspace = manager.allocate(kind="probe", operation_id="bound-root",
        source_revision_ref=selected.state.current_revision_ref, source_identity="a" * 64)
    mapping = selected.state.load(workspace.mapping_ref)
    forged_ref = selected.state.save("execution-workspace", {
        **mapping, "execution_root": str((tmp_path / "forged-execution-root").resolve())})
    index = manager.evidence_root / "workspace-mappings" / (workspace.workspace_id + ".json")
    index.write_text(json.dumps({"schema": "task-agent-execution-workspace-binding.v2",
        "workspace_id": workspace.workspace_id, "mapping_ref": forged_ref}), encoding="utf-8")

    with pytest.raises(ExecutionWorkspaceError, match="EXECUTION_WORKSPACE_MAPPING_SCOPE_INVALID"):
        manager.allocate(kind="probe", operation_id="bound-root",
            source_revision_ref=selected.state.current_revision_ref, source_identity="a" * 64)


@pytest.mark.parametrize("linked_parent", ["workspace-mappings", "probe"])
def test_execution_workspace_rejects_linked_evidence_parents(tmp_path, monkeypatch, linked_parent):
    from src.mod_agent.execution_workspace import ExecutionWorkspaceError

    selected = owner(tmp_path)
    manager = selected.execution_workspaces
    outside = tmp_path / ("outside-" + linked_parent)
    outside.mkdir()
    manager.evidence_root.mkdir(parents=True, exist_ok=True)
    linked = manager.evidence_root / linked_parent
    _directory_link_or_reparse(linked, outside, monkeypatch)

    with pytest.raises(ExecutionWorkspaceError, match="EXECUTION_WORKSPACE_EVIDENCE_LINKED"):
        manager.allocate(kind="probe", operation_id="linked-evidence-" + linked_parent,
            source_revision_ref=selected.state.current_revision_ref, source_identity="a" * 64)
    assert not any(outside.iterdir())


@pytest.mark.parametrize("linked_level", ["root", "parent"])
def test_execution_workspace_rejects_a_linked_configured_evidence_root(tmp_path, monkeypatch, linked_level):
    from src.mod_agent.execution_workspace import ExecutionWorkspaceError, ExecutionWorkspaceManager

    selected = owner(tmp_path)
    outside = tmp_path / ("outside-configured-" + linked_level)
    outside.mkdir()
    linked = tmp_path / ("linked-configured-" + linked_level)
    _directory_link_or_reparse(linked, outside, monkeypatch)
    evidence_root = linked if linked_level == "root" else linked / "evidence"

    with pytest.raises(ExecutionWorkspaceError, match="EXECUTION_WORKSPACE_EVIDENCE_LINKED"):
        ExecutionWorkspaceManager(state=selected.state, evidence_root=evidence_root,
            execution_base=tmp_path / "execution-base")


@pytest.mark.parametrize(("failure", "reason"), [
    ("source_missing", "REPOSITORY_PROBE_OFFLINE_PACKAGE_SOURCE_MISSING"),
    ("target_missing", "REPOSITORY_PROBE_EXECUTION_TARGET_WRITE_FAILED"),
    ("target_denied", "REPOSITORY_PROBE_EXECUTION_TARGET_PERMISSION_DENIED"),
])
def test_normal_probe_preserves_distinct_package_preflight_failures(short_workspace, monkeypatch, failure, reason):
    from src.mod_agent.domain.csharp_repository_backend import CSharpRepositoryProfile
    from src.mod_agent.model_qualification import canonical_hash
    from src.mod_agent.task_agent_v2_experiments import (
        ExperimentOutcomeUnknownError,
        TaskAgentExperiments,
    )
    from src.mod_agent.task_agent_v2_state import TaskAgentV2State, save_v2_artifact
    from tests.unit.test_csharp_repository_backend import _backend, _profile

    profile, source = _profile(short_workspace)
    feed = short_workspace / "missing-feed"
    manifest_sha256 = "0" * 64
    if failure != "source_missing":
        feed = short_workspace / "feed"
        feed.mkdir()
        (feed / "package.1.0.0.nupkg").write_bytes(b"package")
        manifest_sha256 = canonical_hash(runner.offline_package_manifest(feed))
    runner_config = {**config(), "offline_packages": {"path": str(feed), "manifest_sha256": manifest_sha256}}
    profile_path = short_workspace / "probe-profile.json"
    profile_path.write_text(json.dumps({**profile.data, "probe_runner": runner_config}), encoding="utf-8")
    profile = CSharpRepositoryProfile.load(profile_path)
    backend = _backend(short_workspace, profile, source)
    store = short_workspace / "state"
    allowed = profile.data["allowed_edit_paths"]
    authority = save_v2_artifact(store, "authority", {"schema": "task-agent-authority.v2", "scope": "file",
        "operation": "modify_existing", "allowed_edit_paths": allowed, "read_roots": {"target": str(source)}})
    state = TaskAgentV2State.create(store, "package-preflight-" + failure, source,
        allowed_edit_paths=allowed, authority_ref=authority,
        contract_fields={"execution_profile_ref": profile.reference})
    environment = {"image_id": "controlled-image", "image_environment": []}
    monkeypatch.setattr(runner.RepositoryProbeRunner, "_environment", lambda *_args: ([], deepcopy(environment)))
    selected = runner.RepositoryProbeRunner(state=state, backend=backend)
    experiments = TaskAgentExperiments(state=state, probe_runner=selected)
    capability_payload = {"status": "qualified", "profile_ref": experiments.probe_profile_ref,
        "execution_kind": "isolated", "runner_identity": selected.identity(environment),
        "boundaries": {"filesystem": "isolated", "network": "denied", "process": "isolated"}}
    capability = {**capability_payload,
        "evidence_refs": [state.save("probe-qualification", capability_payload)]}
    monkeypatch.setattr(runner.RepositoryProbeRunner, "verify_capability", lambda *_args: deepcopy(capability))
    monkeypatch.setattr(runner, "execute_sdk_container", lambda **_kwargs: pytest.fail("Preflight must stop before Docker"))
    submitted = experiments.probe_submit(source_revision_ref=state.current_revision_ref,
        files=[{"relative_path": "Program.cs", "content": "class Program {}"}], entrypoint_id="csharp_console",
        rationale="Measure the candidate", expected_observation="The probe runs")
    if failure != "source_missing":
        original_write = Path.write_bytes

        def fail_target(path, data):
            if path.suffix == ".nupkg" and path.parent.name == "offline":
                error = PermissionError if failure == "target_denied" else FileNotFoundError
                raise error("controlled target failure")
            return original_write(path, data)

        monkeypatch.setattr(Path, "write_bytes", fail_target)
    with pytest.raises(ExperimentOutcomeUnknownError) as captured:
        experiments.experiment_run(source_revision_ref=state.current_revision_ref,
            probe_artifact_ref=submitted["probe_artifact_ref"], operation_id="package-preflight")
    unknown = state.load(str(captured.value))
    assert unknown["reason"] == reason
    assert unknown["status"] == "unknown"


@pytest.mark.parametrize("restore,publish", [(1, 125), (0, 1), (0, 0)])
def test_full_project_build_failure_is_feedback_and_success_runs_separate_runtime(tmp_path, monkeypatch, restore, publish):
    selected = owner(tmp_path)
    root = tmp_path / "execution"
    candidate = root / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "Source.cs").write_text("class Source {}")
    calls = []
    restore_output = "error NU1101: Unable to find package Missing.Package" if restore else "restore observation"
    files = {"logs/status.json": json.dumps({"restore_exit_code": restore, "publish_exit_code": publish}),
        "logs/sdk-version.txt": "8.0.100", "logs/restore.txt": restore_output, "logs/publish.txt": "compile observation",
        "logs/compile-inputs.txt": "\ufeff/work/repository/Library/Library.csproj|/work/repository/源.cs\n",
        "artifacts/RepoForge.Probe.dll": "controlled assembly", "artifacts/RepoForge.Probe.deps.json":
            json.dumps({"libraries": {"Library/1.0": {"type": "project"}}})}

    def execute(**kwargs):
        calls.append(kwargs)
        return {"completed": True, "exit_code": 0, "stdout": "observed candidate" if kwargs["runtime"] else bundle(files), "stderr": ""}

    monkeypatch.setattr(runner, "execute_sdk_container", execute)
    result = selected._compile_and_run(root, candidate, [{"relative_path": "probe/Program.cs", "content": "class Program {}"}], [], {})
    assert result["source_manifest"][0]["path"] == "Source.cs"
    assert calls[0]["mounts"]["/candidate"] == candidate
    assert result["build_diagnostics"]["publish"] == "compile observation"
    if restore or publish:
        assert len(calls) == 1 and not result["passed"] and result["runtime_execution"] is None
        if restore:
            assert result["dependency_failure"] == {"kind": "offline_package_missing", "diagnostic_codes": ["NU1101"]}
    else:
        assert len(calls) == 2 and result["passed"]
        assert set(calls[1]["mounts"]) == {"/candidate", "/assembly"}
        assert result["compile_inputs"] == ["/work/repository/Library/Library.csproj|/work/repository/源.cs"]


def test_container_inspection_rejects_writable_source_and_network(tmp_path):
    policy = runner._policy(True)
    source = tmp_path / "source"
    document = {"Image": "image-id", "Config": {"Image": config()["image"], "User": "65534:65534",
        "Entrypoint": ["dotnet"], "Cmd": ["/assembly/RepoForge.Probe.dll"], "Labels": {"codex.repository-probe": "token"},
        "Env": list(runner._ENV)}, "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
        "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "PidsLimit": policy["pids"], "Memory": policy["memory"],
        "MemorySwap": policy["memory"], "NanoCpus": policy["nano_cpus"], "Tmpfs": policy["tmpfs"], "Init": True,
        "LogConfig": {"Type": "none"}}, "Mounts": [{"Destination": "/candidate", "Type": "bind", "RW": False, "Source": str(source)}]}
    kwargs = dict(image=config()["image"], image_id="image-id", token="token", mounts={"/candidate": source},
        entrypoint="dotnet", command=["/assembly/RepoForge.Probe.dll"], runtime=True, image_environment=[])
    runner.verify_container_inspect(document, **kwargs)
    changed = deepcopy(document)
    changed["Mounts"][0]["RW"] = True
    with pytest.raises(ValueError, match="CONTAINER_POLICY_MISMATCH"):
        runner.verify_container_inspect(changed, **kwargs)
    changed = deepcopy(document)
    changed["HostConfig"]["NetworkMode"] = "host"
    with pytest.raises(ValueError, match="CONTAINER_POLICY_MISMATCH"):
        runner.verify_container_inspect(changed, **kwargs)
