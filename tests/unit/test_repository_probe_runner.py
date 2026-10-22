"""Controlled SDK transport tests; not evidence of actual container isolation."""
import base64
import io
import json
import tarfile
from copy import deepcopy
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
    return runner.RepositoryProbeRunner(state=SimpleNamespace(root=tmp_path / "state"),
        backend=SimpleNamespace(profile=SimpleNamespace(data={"probe_runner": config(), "configuration": "Release"},
            reference={"path": "controlled-profile", "sha256": "b" * 64})))


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

    def execute(root, candidate_root, files, docker, environment):
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


@pytest.mark.parametrize("restore,publish", [(1, 125), (0, 1), (0, 0)])
def test_full_project_build_failure_is_feedback_and_success_runs_separate_runtime(tmp_path, monkeypatch, restore, publish):
    selected = owner(tmp_path)
    root = tmp_path / "execution"
    candidate = root / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "Source.cs").write_text("class Source {}")
    calls = []
    files = {"logs/status.json": json.dumps({"restore_exit_code": restore, "publish_exit_code": publish}),
        "logs/sdk-version.txt": "8.0.100", "logs/restore.txt": "restore observation", "logs/publish.txt": "compile observation",
        "logs/compile-inputs.txt": "/work/repository/Library/Library.csproj|/work/repository/Source.cs\n",
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
    else:
        assert len(calls) == 2 and result["passed"]
        assert set(calls[1]["mounts"]) == {"/candidate", "/assembly"}


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
