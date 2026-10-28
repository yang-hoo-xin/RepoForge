"""Pinned SDK-project probes in separately constrained build/runtime containers.

为什么要有它：探针必须引用被选候选的真实工程图，隔离资格来自实际容器与固定控制实验。
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from uuid import uuid4
from xml.etree import ElementTree

from src.mod_agent import execution_workspace as execution_workspace_module
from src.mod_agent.evaluation import restricted_csharp as restricted
from src.mod_agent.model_qualification import canonical_hash, read_bound_json
from src.mod_agent.task_agent_v2_state import _no_links, _path

_KIND = "docker_sdk_project_v1"
_SOURCE_LIMIT = 128 * 1024 * 1024
_PACKAGE_LIMIT = 512 * 1024 * 1024
_ARCHIVE_LIMIT = 64 * 1024 * 1024
_EXPANDED_LIMIT = 128 * 1024 * 1024
_OUTPUT_LIMIT = 1024 * 1024
_ENV = ("DOTNET_CLI_TELEMETRY_OPTOUT=1", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1",
    "DOTNET_NOLOGO=1", "DOTNET_CLI_HOME=/tmp/dotnet-home", "DOTNET_MULTILEVEL_LOOKUP=0",
    "MSBUILDDISABLENODEREUSE=1", "DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER=1",
    "NUGET_PACKAGES=/work/packages", "NUGET_HTTP_CACHE_PATH=/tmp/nuget-http",
    "NUGET_PLUGINS_CACHE_PATH=/tmp/nuget-plugins")
_IGNORED = {".git", ".idea", ".vs", "bin", "obj", "testresults", ".venv", "__pycache__",
    ".env", ".aws", ".ssh", ".codex", "credentials"}
_BUILD_SCRIPT = r'''set -eu
mkdir -p /work/repository /work/harness /work/packages /work/artifacts /work/logs
cp -R /candidate/. /work/repository/
cp -R /harness/. /work/harness/
cd /work/repository
set --
while IFS= read -r flag; do set -- "$@" "$flag"; done < /harness/build-flags.txt
dotnet --version > /work/logs/sdk-version.txt 2>&1 || true
: > /work/logs/restore.txt
: > /work/logs/publish.txt
: > /work/logs/compile-inputs.txt
set +e
dotnet restore /work/harness/RepoForge.Probe.csproj --configfile /harness/NuGet.Config --packages /work/packages "$@" -m:1 > /work/logs/restore.txt 2>&1
restored=$?
published=125
if [ "$restored" -eq 0 ]; then
  dotnet publish /work/harness/RepoForge.Probe.csproj --no-restore --output /work/artifacts "$@" -p:UseSharedCompilation=false -p:BuildProjectReferences=true -p:CustomBeforeMicrosoftCommonTargets=/harness/capture.targets -p:RepositoryCompileInputs=/work/logs/compile-inputs.txt -m:1 > /work/logs/publish.txt 2>&1
  published=$?
fi
set -e
printf '{"restore_exit_code":%s,"publish_exit_code":%s}\n' "$restored" "$published" > /work/logs/status.json
tar -czf /work/bundle.tgz -C /work logs artifacts
printf 'REPOFORGE_SDK_BUNDLE_V1\n'
base64 /work/bundle.tgz | tr -d '\n'
printf '\n'
'''
_QUALIFICATION_SOURCE = r'''using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
class Program {
  [DllImport("libc")] static extern uint getuid();
  static bool Denied(string path) { try { File.WriteAllText(path, "denied"); return false; } catch { return true; } }
  static string Unescape(string value) => Regex.Replace(value, @"\\([0-7]{3})", m => ((char)Convert.ToInt32(m.Groups[1].Value, 8)).ToString());
  static void Absolute(string value) {
    if (!value.StartsWith("/") || value.Split('/').Any(p => p == "." || p == ".."))
      throw new InvalidDataException("Invalid cgroup path");
  }
  static (string directory, string version) Locate(string controller, string membership, string mountinfo) {
    var entries = membership.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(line => line.TrimEnd('\r').Split(':', 3)).ToArray();
    var own = entries.Where(row => row.Length == 3 && row[1].Split(',').Contains(controller)).ToArray();
    bool unified = own.Length == 0;
    if (unified) own = entries.Where(row => row.Length == 3 && row[0] == "0" && row[1] == "").ToArray();
    if (own.Length != 1) throw new InvalidDataException("Missing or ambiguous cgroup membership: " + controller);
    string group = own[0][2]; Absolute(group);
    var directories = new HashSet<string>();
    foreach (string line in mountinfo.Split('\n', StringSplitOptions.RemoveEmptyEntries)) {
      var sides = line.TrimEnd('\r').Split(" - ", 2, StringSplitOptions.None);
      if (sides.Length != 2) continue;
      var left = sides[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
      var right = sides[1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
      if (left.Length < 6 || right.Length < 3 || right[0] != (unified ? "cgroup2" : "cgroup")) continue;
      if (!unified && !right[2].Split(',').Contains(controller)) continue;
      string root = Unescape(left[3]), mount = Unescape(left[4]); Absolute(root); Absolute(mount);
      string relative;
      if (group == root) relative = "";
      else if (root == "/") relative = group.Substring(1);
      else if (group.StartsWith(root.TrimEnd('/') + "/", StringComparison.Ordinal)) relative = group.Substring(root.TrimEnd('/').Length + 1);
      else continue;
      directories.Add(mount.TrimEnd('/') + (relative == "" ? "" : "/" + relative));
    }
    if (directories.Count != 1) throw new InvalidDataException("Missing or ambiguous cgroup mount: " + controller);
    return (directories.Single(), unified ? "2" : "1");
  }
  static async Task<int> Main() {
    string membership = File.ReadAllText("/proc/self/cgroup"), mountinfo = File.ReadAllText("/proc/self/mountinfo");
    var versions = new Dictionary<string, string>();
    var files = new Dictionary<string, string>();
    string Read(string controller, string v1, string v2) {
      var located = Locate(controller, membership, mountinfo);
      versions[controller] = located.version;
      string path = located.directory + "/" + (located.version == "1" ? v1 : v2);
      if (!files.ContainsKey(path)) files[path] = File.ReadAllText(path);
      return files[path].Trim();
    }
    string pids = Read("pids", "pids.max", "pids.max");
    string memory = Read("memory", "memory.limit_in_bytes", "memory.max");
    string cpu = Read("cpu", "cpu.cfs_quota_us", "cpu.max");
    if (versions["cpu"] == "1") cpu += " " + Read("cpu", "cpu.cfs_period_us", "cpu.max");
    bool networkDenied = false;
    using(var socket = new TcpClient()) { try { await socket.ConnectAsync("192.0.2.1", 9).WaitAsync(TimeSpan.FromSeconds(2)); }
      catch { networkDenied = true; } }
    Console.WriteLine(JsonSerializer.Serialize(new {
      candidate_value = RepoForgeQualificationTarget.Value.Get(), uid = getuid(),
      source_readonly = Denied("/candidate/qualification-write.cs"),
      assembly_readonly = Denied("/assembly/qualification-write.txt"),
      root_readonly = Denied("/etc/repoforge-qualification-write"), network_denied = networkDenied,
      host_canary_absent = Environment.GetEnvironmentVariable("REPOFORGE_QUALIFY_HOST_ONLY") == null,
      process_status = File.ReadAllText("/proc/self/status"),
      pids_max = pids, memory_max = memory, cpu_max = cpu,
      cgroup_observation = new { proc_self_cgroup = membership, proc_self_mountinfo = mountinfo, versions, files }
    }));
    return 0;
  }
}'''


def _require(condition, reason):
    if not condition:
        raise ValueError("REPOSITORY_PROBE_" + reason)


def _cgroup_location(controller, membership, mountinfo):
    """Resolve the current process's controller path from kernel proc originals.

    为什么要有它：v1独立层级和v2统一层级不能靠固定目录或Docker配置代替实际内核读取。
    """
    def absolute(value):
        _require(isinstance(value, str) and value.startswith("/")
            and not any(part in {".", ".."} for part in value.split("/")), "CGROUP_PATH_INVALID")
        return value

    entries = [line.rstrip("\r").split(":", 2) for line in membership.splitlines() if line]
    own = [row for row in entries if len(row) == 3 and controller in row[1].split(",")]
    unified = not own
    if unified:
        own = [row for row in entries if len(row) == 3 and row[:2] == ["0", ""]]
    _require(len(own) == 1, "CGROUP_MEMBERSHIP_INVALID")
    group = absolute(own[0][2])
    directories = set()
    for line in mountinfo.splitlines():
        sides = line.split(" - ", 1)
        if len(sides) != 2:
            continue
        left, right = sides[0].split(), sides[1].split()
        if len(left) < 6 or len(right) < 3 or right[0] != ("cgroup2" if unified else "cgroup"):
            continue
        if not unified and controller not in right[2].split(","):
            continue
        root, mount = (absolute(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value))
            for value in left[3:5])
        if group == root:
            relative = ""
        elif root == "/":
            relative = group[1:]
        elif group.startswith(root.rstrip("/") + "/"):
            relative = group[len(root.rstrip("/")) + 1:]
        else:
            continue
        directories.add(mount.rstrip("/") + ("/" + relative if relative else ""))
    _require(len(directories) == 1, "CGROUP_MOUNT_INVALID")
    return directories.pop(), "2" if unified else "1"


def _cgroup_control_values(observation):
    """Recompute limits using the recorded kernel paths and unmodified file values."""
    _require(isinstance(observation, dict) and set(observation) == {
        "proc_self_cgroup", "proc_self_mountinfo", "versions", "files"}
        and isinstance(observation["proc_self_cgroup"], str) and isinstance(observation["proc_self_mountinfo"], str)
        and isinstance(observation["versions"], dict) and isinstance(observation["files"], dict), "CGROUP_OBSERVATION_INVALID")
    versions, paths, values = {}, set(), {}
    for controller, key, v1_names, v2_names in (
        ("pids", "pids_max", ["pids.max"], ["pids.max"]),
        ("memory", "memory_max", ["memory.limit_in_bytes"], ["memory.max"]),
        ("cpu", "cpu_max", ["cpu.cfs_quota_us", "cpu.cfs_period_us"], ["cpu.max"]),
    ):
        directory, version = _cgroup_location(controller, observation["proc_self_cgroup"], observation["proc_self_mountinfo"])
        versions[controller] = version
        names = v1_names if version == "1" else v2_names
        selected = [directory + "/" + name for name in names]
        _require(all(isinstance(observation["files"].get(path), str) for path in selected), "CGROUP_FILE_MISSING")
        paths.update(selected)
        values[key] = " ".join(observation["files"][path].strip() for path in selected)
    _require(observation["versions"] == versions and set(observation["files"]) == paths, "CGROUP_SOURCE_CHANGED")
    return values


def _verify_cgroup_controls(observed):
    values = _cgroup_control_values(observed.get("cgroup_observation"))
    _require(all(observed.get(key) == value for key, value in values.items())
        and values["pids_max"] == "64" and values["memory_max"] == "1073741824"
        and values["cpu_max"].split() == ["100000", "100000"], "QUALIFICATION_BOUNDARY_FAILED")


def _local_docker_endpoint(endpoint):
    # 为什么要有它：named-pipe和unix协议名本身不证明本机，远端authority不能越过本地执行范围。
    return isinstance(endpoint, str) and bool(
        re.fullmatch(r"npipe:/{4}\./pipe/[^/\\\x00-\x20?#]+", endpoint)
        or re.fullmatch(r"unix:///(?!/)[^\x00-\x20?#\\]+", endpoint))


def validate_probe_runner_config(value):
    required = {"kind", "image", "sdk_version", "target_framework", "project_path"}
    _require(isinstance(value, dict) and required <= set(value)
        and set(value) <= required | {"docker_context", "offline_packages", "qualification_ref"}, "CONFIG_FIELDS_INVALID")
    _require(value["kind"] == _KIND, "RUNNER_KIND_UNSUPPORTED")
    _require(re.fullmatch(r"mcr\.microsoft\.com/dotnet/sdk@sha256:[0-9a-f]{64}", value["image"]), "PINNED_SDK_IMAGE_REQUIRED")
    _require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value["sdk_version"]), "SDK_VERSION_REQUIRED")
    _require(re.fullmatch(r"net(?:[6-9]|[1-9][0-9])\.0", value["target_framework"]), "FRAMEWORK_UNSUPPORTED")
    _require(_path(value["project_path"]).endswith(".csproj"), "SDK_PROJECT_REQUIRED")
    _require(value.get("docker_context", "desktop-linux" if os.name == "nt" else "default") in {"desktop-linux", "default"}, "LOCAL_DOCKER_CONTEXT_REQUIRED")
    if "offline_packages" in value:
        package = value["offline_packages"]
        _require(isinstance(package, dict) and set(package) == {"path", "manifest_sha256"}
            and Path(package["path"]).is_absolute() and re.fullmatch(r"[0-9a-f]{64}", package["manifest_sha256"]), "OFFLINE_PACKAGE_MANIFEST_REQUIRED")
    if "qualification_ref" in value:
        _require(isinstance(value["qualification_ref"], dict) and set(value["qualification_ref"]) == {"path", "sha256"}, "QUALIFICATION_REFERENCE_INVALID")
    return deepcopy(value)


def _ref(path):
    path = Path(path)
    _no_links(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _save(path, value):
    return restricted._save(Path(path), value)


def _control(root, name, argv, *, host_canary=None):
    env = {**os.environ, **({"REPOFORGE_QUALIFY_HOST_ONLY": host_canary} if host_canary else {})}
    result = subprocess.run(argv, capture_output=True, timeout=20, env=env)
    captured = result.stdout.decode("utf-8", "replace")
    if name.startswith("inspect") or name == "image":
        try:
            documents = json.loads(captured)
            for document in documents:
                values = document.get("Config", {}).get("Env", [])
                document.get("Config", {})["Env"] = [value.split("=", 1)[0] + "=<redacted>"
                    if re.search(r"secret|token|password|credential|api_key|proxy", value.split("=", 1)[0], re.I) else value for value in values]
            captured = json.dumps(documents, ensure_ascii=False, sort_keys=True)
        except (ValueError, TypeError):
            captured = "<unparseable inspect output>"
    reference = _save(Path(root) / (name + ".json"), {"argv": argv, "exit_code": result.returncode,
        "stdout": captured, "raw_stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr": result.stderr.decode("utf-8", "replace")})
    return result, reference


def _tree(root):
    root = Path(root).resolve(strict=True)
    _no_links(root)
    rows = []
    for path in sorted(root.rglob("*")):
        _no_links(path)
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "sha256": _ref(path)["sha256"], "bytes": path.stat().st_size})
    return rows


def offline_package_manifest(root):
    root = Path(root).resolve(strict=True)
    _no_links(root)
    rows = []
    total = 0
    for path in sorted(root.rglob("*.nupkg")):
        _no_links(path)
        _require(path.resolve().is_relative_to(root), "PACKAGE_PATH_ESCAPES_ROOT")
        total += path.stat().st_size
        _require(total <= _PACKAGE_LIMIT, "OFFLINE_PACKAGE_LIMIT")
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": _ref(path)["sha256"], "bytes": path.stat().st_size})
    _require(sum(row["bytes"] for row in rows) <= _PACKAGE_LIMIT, "OFFLINE_PACKAGE_LIMIT")
    return rows


def _policy(runtime):
    return {"memory": 1073741824 if runtime else 2147483648, "pids": 64 if runtime else 128,
        "nano_cpus": 1000000000, "tmpfs": {"/tmp": "rw,nosuid,nodev,noexec,size=64m",
            **({"/operation-output": "rw,nosuid,nodev,noexec,size=64m"} if runtime else
               {"/work": "rw,nosuid,nodev,size=1073741824,mode=1777"})}}


def verify_container_inspect(document, *, image, image_id, token, mounts, entrypoint, command, runtime, image_environment):
    config, host, expected = document["Config"], document["HostConfig"], _policy(runtime)
    environment = dict(item.split("=", 1) for item in image_environment)
    environment.update(item.split("=", 1) for item in _ENV)
    checks = [config["Image"] == image, document["Image"] == image_id, config["User"] == "65534:65534",
        config["Entrypoint"] == [entrypoint], config["Cmd"] == command,
        config.get("Labels", {}).get("codex.repository-probe") == token,
        dict(item.split("=", 1) for item in config["Env"]) == environment,
        len(config["Env"]) == len(environment),
        host["NetworkMode"] == "none", host["ReadonlyRootfs"] is True, host["Privileged"] is False,
        host["CapDrop"] == ["ALL"], not host.get("CapAdd"),
        host["SecurityOpt"] in (["no-new-privileges"], ["no-new-privileges=true"]),
        host["PidsLimit"] == expected["pids"], host["Memory"] == expected["memory"],
        host["MemorySwap"] == expected["memory"], host["NanoCpus"] == expected["nano_cpus"],
        host["Tmpfs"] == expected["tmpfs"], host.get("Init") is True, not host.get("Devices"),
        not host.get("VolumesFrom"), not host.get("PidMode"), host["LogConfig"]["Type"] == "none",
        len(document["Mounts"]) == len(mounts)]
    for destination, source in mounts.items():
        selected = [row for row in document["Mounts"] if row["Destination"] == destination]
        checks.append(len(selected) == 1 and selected[0]["Type"] == "bind" and selected[0]["RW"] is False
            and restricted._same_mount_source(selected[0]["Source"], Path(source)))
    _require(all(checks), "CONTAINER_POLICY_MISMATCH")


def execute_sdk_container(*, root, docker, environment, image, mounts, runtime):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    token = uuid4().hex
    policy = _policy(runtime)
    entrypoint, command = ("dotnet", ["/assembly/RepoForge.Probe.dll"]) if runtime else ("/bin/sh", ["-c", _BUILD_SCRIPT])
    argv = [*docker, "create", "--name=repoforge-probe-" + token, "--label=codex.repository-probe=" + token,
        "--pull=never", "--platform=linux/amd64", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=65534:65534", "--pids-limit=" + str(policy["pids"]),
        "--memory=" + str(policy["memory"]), "--memory-swap=" + str(policy["memory"]), "--cpus=1",
        "--init", "--interactive", "--log-driver=none", "--workdir=/tmp", "--entrypoint=" + entrypoint]
    argv.extend("--tmpfs=" + path + ":" + options for path, options in policy["tmpfs"].items())
    argv.extend("--env=" + value for value in _ENV)
    for destination, source in mounts.items():
        _no_links(Path(source))
        _require("," not in str(source), "MOUNT_PATH_UNSUPPORTED")
        argv.extend(["--mount", f"type=bind,source={Path(source).resolve()},target={destination},readonly"])
    argv.extend([image, *command])
    created, created_ref = _control(root, "create", argv, host_canary=token)
    _require(created.returncode == 0, "CONTAINER_CREATE_FAILED")
    container = created.stdout.decode().strip()
    _require(re.fullmatch(r"[a-f0-9]{64}", container), "CONTAINER_ID_INVALID")
    primary = None
    try:
        before, before_ref = _control(root, "inspect-before", [*docker, "inspect", container])
        _require(before.returncode == 0, "INSPECT_FAILED")
        document = json.loads(before.stdout)[0]
        verify_container_inspect(document, image=image, image_id=environment["image_id"], token=token,
            mounts=mounts, entrypoint=entrypoint, command=command, runtime=runtime,
            image_environment=environment["image_environment"])
        execution = restricted._attach([*docker, "start", "--attach", "--interactive", container], stdin=b"",
            timeout=15 if runtime else 180, output_limit=_OUTPUT_LIMIT if runtime else _ARCHIVE_LIMIT * 2,
            stop=lambda: _control(root, "kill", [*docker, "kill", container]))
        after, after_ref = _control(root, "inspect-after", [*docker, "inspect", container])
        _require(after.returncode == 0, "RESULT_INSPECT_FAILED")
        final = json.loads(after.stdout)[0]
        verify_container_inspect(final, image=image, image_id=environment["image_id"], token=token,
            mounts=mounts, entrypoint=entrypoint, command=command, runtime=runtime,
            image_environment=environment["image_environment"])
        execution.update(container_id=container, image=image, image_id=final["Image"], container_state=final["State"],
            completed=not final["State"]["Running"] and execution["exit_code"] == final["State"]["ExitCode"]
                and not execution["timed_out"] and not execution["output_limit_exceeded"],
            create_ref=created_ref, inspect_before_ref=before_ref, inspect_after_ref=after_ref,
            policy_verified=True, runtime=runtime)
        execution["receipt_ref"] = _save(root / "execution.json", execution)
        return execution
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            removed, _reference = _control(root, "cleanup", [*docker, "rm", "--force", container])
            _require(removed.returncode == 0, "CONTAINER_CLEANUP_FAILED")
        except BaseException as error:
            if primary is None:
                raise
            primary.container_cleanup_error = str(error)


def extract_build_bundle(output, destination):
    lines = output.splitlines()
    _require(len(lines) == 2 and lines[0] == "REPOFORGE_SDK_BUNDLE_V1", "BUILD_PROTOCOL_INVALID")
    packed = base64.b64decode(lines[1], validate=True)
    _require(len(packed) <= _ARCHIVE_LIMIT, "BUILD_ARCHIVE_LIMIT")
    root = Path(destination)
    _no_links(root)
    root.mkdir(parents=True, exist_ok=False)
    total, seen = 0, set()
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
        for member in archive:
            relative = _path(member.name.rstrip("/"))
            _require(relative.split("/", 1)[0] in {"logs", "artifacts"} and not re.search(r"[\x00-\x1f]", relative), "ARCHIVE_PATH_DENIED")
            # 为什么要有它：Linux归档返回Windows宿主时，设备名、流及规范化别名不能绕过目标文件范围。
            _require(all(not re.search(r'[<>"|?*]', part) and not part.endswith((".", " "))
                and not re.fullmatch(r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³]",
                    part.split(".", 1)[0], re.I) for part in relative.split("/")), "ARCHIVE_WINDOWS_PATH_DENIED")
            _require(not member.issym() and not member.islnk() and (member.isdir() or member.isfile()), "ARCHIVE_TYPE_DENIED")
            key = relative.casefold()
            _require(key not in seen and len(seen) < 10000, "ARCHIVE_PATH_COLLISION")
            seen.add(key)
            total += member.size
            _require(total <= _EXPANDED_LIMIT and member.size <= _EXPANDED_LIMIT, "BUILD_EXPANSION_LIMIT")
            target = root / relative
            _no_links(target)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                content = archive.extractfile(member).read(member.size + 1)
                _require(len(content) == member.size, "ARCHIVE_CONTENT_TRUNCATED")
                with target.open("xb") as stream:
                    stream.write(content)
    return _tree(root)


class RepositoryProbeRunner:
    def __init__(self, *, state, backend):
        self.state, self.backend = state, backend
        self.config = validate_probe_runner_config(backend.profile.data["probe_runner"])
        self.root = state.root / "repository-probes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_workspaces = execution_workspace_module.ExecutionWorkspaceManager(
            state=state, evidence_root=self.root)
        self._capability = None
        self._attempted = False
        self._qualification_attempt_ref = None

    def _environment(self):
        root = self.root / ("environment-" + uuid4().hex)
        root.mkdir()
        docker = restricted._docker()
        docker[-1] = self.config.get("docker_context", "desktop-linux" if os.name == "nt" else "default")
        context, context_ref = _control(root, "context", [*docker, "context", "inspect", docker[-1]])
        _require(context.returncode == 0, "DOCKER_CONTEXT_UNAVAILABLE")
        endpoint = json.loads(context.stdout)[0]["Endpoints"]["docker"]["Host"]
        _require(_local_docker_endpoint(endpoint), "REMOTE_DOCKER_CONTEXT_DENIED")
        info, info_ref = _control(root, "engine", [*docker, "info", "--format", "{{json .}}"])
        _require(info.returncode == 0, "DOCKER_ENGINE_UNAVAILABLE")
        server = json.loads(info.stdout)
        _require(server["OSType"] == "linux" and server["Architecture"] in {"x86_64", "amd64"}, "LINUX_AMD64_ENGINE_REQUIRED")
        image, image_ref = _control(root, "image", [*docker, "image", "inspect", self.config["image"]])
        _require(image.returncode == 0, "PINNED_IMAGE_UNAVAILABLE")
        selected = json.loads(image.stdout)[0]
        _require(self.config["image"] in selected.get("RepoDigests", []) and selected["Os"] == "linux"
            and selected["Architecture"] == "amd64", "PINNED_IMAGE_IDENTITY_CHANGED")
        identity = {"engine_id": server["ID"], "server_version": server["ServerVersion"], "kernel_version": server["KernelVersion"],
            "security_options": server.get("SecurityOptions", []), "image_id": selected["Id"], "image": self.config["image"],
            "docker_ref": _ref(docker[0]), "context": docker[-1]}
        return docker, {**identity, "image_environment": selected["Config"].get("Env", []),
            "evidence_refs": [context_ref, info_ref, image_ref]}

    def identity(self, environment):
        configuration = {key: value for key, value in self.config.items() if key != "qualification_ref"}
        # 为什么要有它：资格引用不能参与其自身所证明的身份；其他实际工程、权限及执行参数仍完整冻结。
        execution_profile = {**self.backend.profile.data, "probe_runner": configuration}
        return {"kind": _KIND, "config_sha256": canonical_hash(configuration),
            "environment": {key: value for key, value in environment.items() if key not in {"evidence_refs", "image_environment"}},
            "runner_ref": _ref(__file__), "restricted_transport_ref": _ref(restricted.__file__),
            "execution_workspace_ref": _ref(execution_workspace_module.__file__),
            "repository_execution_profile_sha256": canonical_hash(execution_profile), "platform": "linux/amd64"}

    def _verify_qualification_execution(self, captured, *, environment, mounts, runtime):
        reference = captured["receipt_ref"]
        execution = read_bound_json(reference)
        _require(execution == {key: value for key, value in captured.items() if key != "receipt_ref"}, "QUALIFICATION_EXECUTION_CHANGED")
        _require(execution.get("completed") is True and execution.get("policy_verified") is True
            and execution.get("runtime") is runtime and execution.get("exit_code") == 0
            and execution.get("timed_out") is False and execution.get("output_limit_exceeded") is False
            and execution.get("local_process_reaped") is True and execution["container_state"]["Running"] is False
            and execution["container_state"]["ExitCode"] == 0
            and execution["image"] == self.config["image"] and execution["image_id"] == environment["image_id"],
            "QUALIFICATION_EXECUTION_NOT_SUCCESSFUL")
        for field in ("stdout", "stderr"):
            _require(hashlib.sha256(execution[field].encode("utf-8")).hexdigest() == execution[field + "_sha256"],
                "QUALIFICATION_STREAM_CHANGED")
        created = read_bound_json(execution["create_ref"])
        _require(created["exit_code"] == 0 and created["stdout"].strip() == execution["container_id"]
            and "--pull=never" in created["argv"], "QUALIFICATION_CREATE_CHANGED")
        tokens = [arg.split("=", 2)[2] for arg in created["argv"] if arg.startswith("--label=codex.repository-probe=")]
        _require(len(tokens) == 1, "QUALIFICATION_LABEL_CHANGED")
        entrypoint, command = ("dotnet", ["/assembly/RepoForge.Probe.dll"]) if runtime else ("/bin/sh", ["-c", _BUILD_SCRIPT])
        image_env = [value.split("=", 1)[0] + "=<redacted>"
            if re.search(r"secret|token|password|credential|api_key|proxy", value.split("=", 1)[0], re.I) else value
            for value in environment["image_environment"]]
        for name in ("inspect_before_ref", "inspect_after_ref"):
            record = read_bound_json(execution[name])
            _require(record["exit_code"] == 0 and record["argv"][-2:] == ["inspect", execution["container_id"]],
                "QUALIFICATION_INSPECT_CHANGED")
            documents = json.loads(record["stdout"])
            _require(len(documents) == 1, "QUALIFICATION_INSPECT_CHANGED")
            verify_container_inspect(documents[0], image=self.config["image"], image_id=environment["image_id"],
                token=tokens[0], mounts=mounts, entrypoint=entrypoint, command=command, runtime=runtime, image_environment=image_env)
            if name == "inspect_after_ref":
                _require(documents[0]["State"] == execution["container_state"], "QUALIFICATION_CONTAINER_STATE_CHANGED")
        cleanup = read_bound_json(_ref(Path(reference["path"]).parent / "cleanup.json"))
        _require(cleanup["exit_code"] == 0 and cleanup["argv"][-3:] == ["rm", "--force", execution["container_id"]],
            "QUALIFICATION_CLEANUP_FAILED")
        return execution

    def _read_qualification(self, reference, *, identity, probe_profile_ref, environment):
        """Recompute stored qualification from controls and process originals.

        为什么要有它：外部引用和缓存都不能用qualified或policy_verified自述跳过实际隔离原件。
        """
        proof = read_bound_json(reference)
        _require(proof.get("schema") == "task-agent-probe-qualification.v2" and proof.get("status") == "qualified"
            and proof.get("runner_identity") == identity and proof.get("profile_ref") == probe_profile_ref
            and proof.get("qualification_candidate") == "trusted_control_fixture"
            and proof.get("execution_kind") == "isolated"
            and proof.get("boundaries") == {"filesystem": "isolated", "network": "denied", "process": "isolated"},
            "QUALIFICATION_IDENTITY_CHANGED")
        attempt = read_bound_json(proof["qualification_attempt_ref"])
        _require(attempt.get("runner_identity") == identity and attempt.get("qualification_candidate") == "trusted_control_fixture",
            "QUALIFICATION_ATTEMPT_CHANGED")
        result = attempt["result"]
        _require(result.get("passed") is True and result.get("compiled") is True
            and result.get("runtime_execution") is not None, "QUALIFICATION_EXECUTION_FAILED")
        expected_refs = [result["compile_execution"]["receipt_ref"], result["runtime_execution"]["receipt_ref"]]
        _require(proof.get("execution_refs") == expected_refs and expected_refs[0] != expected_refs[1], "QUALIFICATION_EXECUTION_REFS_REQUIRED")
        workspace_reference = proof.get("workspace_mapping_ref")
        if workspace_reference is None:
            root = Path(proof["qualification_attempt_ref"]["path"]).parent
        else:
            try:
                mapping_path = Path(workspace_reference["path"]).resolve()
                mapping_ref = mapping_path.relative_to(self.state.store_root).as_posix()
                workspace = self.execution_workspaces.recover(mapping_ref=mapping_ref, kind="qualification",
                    operation_id=canonical_hash({"runner_identity": identity, "profile_ref": probe_profile_ref}))
                _require(_ref(mapping_path) == workspace_reference
                    and workspace.evidence_root.resolve()
                        == Path(proof["qualification_attempt_ref"]["path"]).parent.resolve()
                    and attempt.get("workspace_mapping_ref") == workspace_reference,
                    "QUALIFICATION_WORKSPACE_MAPPING_CHANGED")
            except (KeyError, OSError, ValueError, execution_workspace_module.ExecutionWorkspaceError):
                raise ValueError("QUALIFICATION_WORKSPACE_MAPPING_CHANGED") from None
            root = workspace.execution_root
        candidate, harness, packages, bundle = root / "candidate", root / "harness", root / "offline", root / "bundle"
        _require(_tree(candidate) == result["source_manifest"] and _tree(bundle) == result["bundle_manifest"],
            "QUALIFICATION_INPUT_OR_ARTIFACT_CHANGED")
        project = candidate / self.config["project_path"]
        expected_project = '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>' + self.config["target_framework"] + '</TargetFramework><AssemblyName>RepoForgeQualificationTarget</AssemblyName></PropertyGroup></Project>'
        _require(project.read_bytes() == expected_project.encode("utf-8")
            and (project.parent / "Value.cs").read_bytes() == b'namespace RepoForgeQualificationTarget { public static class Value { public static int Get() => 7; } }'
            and (harness / "probe/Program.cs").read_bytes() == _QUALIFICATION_SOURCE.encode("utf-8"), "QUALIFICATION_CONTROL_SOURCE_CHANGED")
        self._verify_qualification_execution(result["compile_execution"], environment=environment,
            mounts={"/candidate": candidate, "/harness": harness, "/offline": packages}, runtime=False)
        runtime_record = self._verify_qualification_execution(result["runtime_execution"], environment=environment,
            mounts={"/candidate": candidate, "/assembly": bundle / "artifacts"}, runtime=True)
        _require(json.loads((bundle / "logs/status.json").read_text()) == result["build_status"] == {
            "restore_exit_code": 0, "publish_exit_code": 0}
            and (bundle / "logs/sdk-version.txt").read_text().strip() == result["sdk_version"] == self.config["sdk_version"],
            "QUALIFICATION_BUILD_CHANGED")
        observed = json.loads(runtime_record["stdout"])
        _verify_cgroup_controls(observed)
        _require(proof["controls"] == observed and observed["candidate_value"] == 7 and observed["uid"] == 65534
            and all(observed[name] is True for name in ("source_readonly", "assembly_readonly", "root_readonly", "network_denied", "host_canary_absent"))
            and re.search(r"NoNewPrivs:\s+1", observed["process_status"])
            and re.search(r"CapEff:\s+0+\s", observed["process_status"])
            and observed["pids_max"] == "64" and observed["memory_max"] == "1073741824"
            and observed["cpu_max"].split() == ["100000", "100000"], "QUALIFICATION_BOUNDARY_FAILED")
        return proof

    def verify_capability(self, probe_profile_ref):
        try:
            docker, environment = self._environment()
            identity = self.identity(environment)
            if self._capability is not None:
                _require(self._capability["runner_identity"] == identity, "RUNNER_IDENTITY_CHANGED")
                self._read_qualification(self._capability["evidence_refs"][0], identity=identity,
                    probe_profile_ref=probe_profile_ref, environment=environment)
                return deepcopy(self._capability)
            if self.config.get("qualification_ref"):
                reference = self.config["qualification_ref"]
                proof = self._read_qualification(reference, identity=identity, probe_profile_ref=probe_profile_ref, environment=environment)
            else:
                _require(not self._attempted, "QUALIFICATION_ALREADY_FAILED")
                self._attempted = True
                proof, reference = self._qualify(docker, environment, identity, probe_profile_ref)
                proof = self._read_qualification(reference, identity=identity, probe_profile_ref=probe_profile_ref, environment=environment)
            self._capability = {key: deepcopy(proof[key]) for key in ("status", "profile_ref", "runner_identity", "execution_kind", "boundaries")}
            self._capability["evidence_refs"] = [reference]
            return deepcopy(self._capability)
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
            return {"status": "unavailable", "reason": str(error), "execution_kind": "isolated",
                "profile_ref": probe_profile_ref, "environment_inspected": True,
                "qualification_attempt_ref": self._qualification_attempt_ref}

    def _bound_packages(self):
        binding = self.config.get("offline_packages")
        if binding is None:
            return None, [], []
        source = Path(binding["path"])
        try:
            _require(stat.S_ISDIR(source.stat().st_mode), "OFFLINE_PACKAGE_SOURCE_MISSING")
            rows = offline_package_manifest(source)
        except FileNotFoundError as error:
            raise ValueError("REPOSITORY_PROBE_OFFLINE_PACKAGE_SOURCE_MISSING") from error
        except PermissionError as error:
            raise ValueError("REPOSITORY_PROBE_OFFLINE_PACKAGE_SOURCE_PERMISSION_DENIED") from error
        _require(canonical_hash(rows) == binding["manifest_sha256"], "OFFLINE_PACKAGES_CHANGED")
        names = set()
        materialized = []
        for row in rows:
            name = PurePosixPath(row["path"]).name
            _require(name.casefold() not in names, "OFFLINE_PACKAGE_COLLISION")
            names.add(name.casefold())
            materialized.append(name)
        return source, rows, materialized

    def _packages(self, root, prepared=None):
        source, rows, names = prepared or self._bound_packages()
        try:
            root.mkdir()
        except PermissionError as error:
            raise ValueError("REPOSITORY_PROBE_EXECUTION_TARGET_PERMISSION_DENIED") from error
        except OSError as error:
            raise ValueError("REPOSITORY_PROBE_EXECUTION_TARGET_WRITE_FAILED") from error
        if source is None:
            return []
        for row, name in zip(rows, names):
            origin, target = source / row["path"], root / name
            try:
                data = origin.read_bytes()
            except FileNotFoundError as error:
                raise ValueError("REPOSITORY_PROBE_OFFLINE_PACKAGE_SOURCE_MISSING") from error
            except PermissionError as error:
                raise ValueError("REPOSITORY_PROBE_OFFLINE_PACKAGE_SOURCE_PERMISSION_DENIED") from error
            try:
                target.write_bytes(data)
            except PermissionError as error:
                raise ValueError("REPOSITORY_PROBE_EXECUTION_TARGET_PERMISSION_DENIED") from error
            except OSError as error:
                raise ValueError("REPOSITORY_PROBE_EXECUTION_TARGET_WRITE_FAILED") from error
            _require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                "OFFLINE_PACKAGE_SOURCE_CHANGED")
        _require(offline_package_manifest(source) == rows, "OFFLINE_PACKAGES_CHANGED")
        copied = offline_package_manifest(root)
        _require({row["path"]: (row["sha256"], row["bytes"]) for row in copied}
            == {name: (row["sha256"], row["bytes"]) for row, name in zip(rows, names)},
            "OFFLINE_PACKAGE_COPY_CHANGED")
        return rows

    def _harness(self, root, files):
        from src.mod_agent.domain.csharp_repository_backend import _capture_targets

        root.mkdir()
        project = ElementTree.Element("Project", Sdk="Microsoft.NET.Sdk")
        properties = ElementTree.SubElement(project, "PropertyGroup")
        for key, value in {"OutputType": "Exe", "TargetFramework": self.config["target_framework"],
                "AssemblyName": "RepoForge.Probe", "UseAppHost": "false", "EnableDefaultCompileItems": "false"}.items():
            ElementTree.SubElement(properties, key).text = value
        items = ElementTree.SubElement(project, "ItemGroup")
        ElementTree.SubElement(items, "ProjectReference", Include="/work/repository/" + self.config["project_path"])
        for row in files:
            path = _path(row["relative_path"])
            _require(path.startswith("probe/") and path.endswith(".cs"), "PROBE_SOURCE_INVALID")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(row["content"].encode("utf-8"))
            ElementTree.SubElement(items, "Compile", Include=path)
        (root / "RepoForge.Probe.csproj").write_bytes(ElementTree.tostring(project, encoding="utf-8", xml_declaration=True))
        (root / "NuGet.Config").write_bytes(b'<configuration><packageSources><clear/><add key="offline" value="/offline"/></packageSources></configuration>')
        (root / "capture.targets").write_bytes(_capture_targets())
        flags = ["-p:Configuration=" + self.backend.profile.data["configuration"]]
        for key, value in sorted(self.backend.profile.data.get("build_properties", {}).items()):
            _require(re.fullmatch(r"[A-Za-z0-9_.+-]{1,128}", value)
                and not re.search(r"credential|password|secret|token|restore|sdkspath|custombefore|customafter", key, re.I), "BUILD_PROPERTY_UNSUPPORTED")
            flags.append("-p:" + key + "=" + value)
        (root / "build-flags.txt").write_bytes(("\n".join(flags) + "\n").encode())

    def _compile_and_run(self, root, candidate, files, docker, environment, *, evidence_root=None,
                         prepared_packages=None):
        evidence_root = Path(evidence_root) if evidence_root is not None else Path(root)
        harness, packages = root / "harness", root / "offline"
        self._harness(harness, files)
        dependencies = self._packages(packages, prepared_packages)
        inputs = {"candidate": _tree(candidate), "harness": _tree(harness), "offline": _tree(packages)}
        compiled = execute_sdk_container(root=evidence_root / "compile", docker=docker, environment=environment, image=self.config["image"],
            mounts={"/candidate": candidate, "/harness": harness, "/offline": packages}, runtime=False)
        _require(inputs == {"candidate": _tree(candidate), "harness": _tree(harness), "offline": _tree(packages)}, "BUILD_INPUT_MUTATED")
        if not compiled["completed"] or compiled["exit_code"] != 0:
            return {"compile_execution": compiled, "runtime_execution": None, "source_manifest": inputs["candidate"],
                "dependency_manifest": dependencies, "compiled": False, "passed": False, "exit_code": compiled["exit_code"] or -1}
        bundle = root / "bundle"
        manifest = extract_build_bundle(compiled["stdout"], bundle)
        status = json.loads((bundle / "logs/status.json").read_text())
        sdk = (bundle / "logs/sdk-version.txt").read_text().strip()
        succeeded = status["restore_exit_code"] == status["publish_exit_code"] == 0 and sdk == self.config["sdk_version"]
        diagnostics = {name: (bundle / ("logs/" + name + ".txt")).read_bytes()[:32768].decode("utf-8", "replace")
            for name in ("restore", "publish")}
        diagnostic_codes = sorted(set(re.findall(r"\bNU[0-9]{4}\b", diagnostics["restore"])))
        dependency_failure = None
        if status["restore_exit_code"] != 0:
            dependency_failure = {"kind": "offline_package_missing"
                if set(diagnostic_codes) & {"NU1101", "NU1102"} else "restore_failed",
                "diagnostic_codes": diagnostic_codes}
        base = {"compile_execution": compiled, "runtime_execution": None, "source_manifest": inputs["candidate"],
            "dependency_manifest": dependencies, "bundle_manifest": manifest, "build_status": status, "sdk_version": sdk,
            "compiled": succeeded, "passed": False, "exit_code": status["publish_exit_code"] or status["restore_exit_code"],
            "build_diagnostics": diagnostics, "build_diagnostics_limit_bytes_each": 32768}
        if dependency_failure is not None:
            base["dependency_failure"] = dependency_failure
        if not succeeded:
            return base
        compile_inputs = (bundle / "logs/compile-inputs.txt").read_text(encoding="utf-8-sig").splitlines()
        _require(any(row.split("|", 1)[0] == "/work/repository/" + self.config["project_path"] for row in compile_inputs), "REFERENCED_PROJECT_NOT_COMPILED")
        assembly = bundle / "artifacts"
        _require((assembly / "RepoForge.Probe.dll").is_file() and (assembly / "RepoForge.Probe.deps.json").is_file(), "PROBE_ASSEMBLY_MISSING")
        deps = json.loads((assembly / "RepoForge.Probe.deps.json").read_text())
        _require(any(value.get("type") == "project" and not name.startswith("RepoForge.Probe/")
            for name, value in deps["libraries"].items()), "PROJECT_REFERENCE_DEPENDENCY_MISSING")
        runtime = execute_sdk_container(root=evidence_root / "runtime", docker=docker, environment=environment, image=self.config["image"],
            mounts={"/candidate": candidate, "/assembly": assembly}, runtime=True)
        _require(_tree(bundle) == manifest and _tree(candidate) == inputs["candidate"], "RUNTIME_INPUT_MUTATED")
        return {**base, "runtime_execution": runtime, "compile_inputs": compile_inputs,
            "project_dependencies": deps["libraries"], "passed": runtime["completed"] and runtime["exit_code"] == 0,
            "exit_code": runtime["exit_code"] if runtime["exit_code"] is not None else -1}

    def _qualify(self, docker, environment, identity, probe_profile_ref):
        prepared_packages = self._bound_packages()
        projected = ["offline/" + name for name in prepared_packages[2]]
        projected.extend(("candidate/" + self.config["project_path"], "harness/probe/Program.cs",
            "harness/RepoForge.Probe.csproj", "bundle/artifacts/RepoForge.Probe.dll"))
        workspace = self.execution_workspaces.allocate(kind="qualification",
            operation_id=canonical_hash({"runner_identity": identity, "profile_ref": probe_profile_ref}),
            projected_paths=projected)
        root, evidence_root = workspace.execution_root, workspace.evidence_root
        mapping_ref = _ref(self.state.store_root / workspace.mapping_ref)
        completed = evidence_root / "qualification.json"
        attempted = evidence_root / "qualification-attempt.json"
        if workspace.recovered:
            _require(completed.is_file() and attempted.is_file(), "QUALIFICATION_WORKSPACE_INCOMPLETE")
            proof_ref = _ref(completed)
            proof = read_bound_json(proof_ref)
            self._qualification_attempt_ref = proof.get("qualification_attempt_ref")
            return proof, proof_ref
        candidate = root / "candidate"
        project = candidate / self.config["project_path"]
        project.parent.mkdir(parents=True)
        project.write_text('<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>' + self.config["target_framework"]
            + '</TargetFramework><AssemblyName>RepoForgeQualificationTarget</AssemblyName></PropertyGroup></Project>', encoding="utf-8")
        (project.parent / "Value.cs").write_text('namespace RepoForgeQualificationTarget { public static class Value { public static int Get() => 7; } }', encoding="utf-8")
        result = self._compile_and_run(root, candidate,
            [{"relative_path": "probe/Program.cs", "content": _QUALIFICATION_SOURCE}], docker, environment,
            evidence_root=evidence_root, prepared_packages=prepared_packages)
        self._qualification_attempt_ref = _save(attempted, {"runner_identity": identity,
            "result": result, "qualification_candidate": "trusted_control_fixture",
            "workspace_mapping_ref": mapping_ref})
        _require(result["passed"] and result["runtime_execution"] is not None, "QUALIFICATION_EXECUTION_FAILED")
        observed = json.loads(result["runtime_execution"]["stdout"])
        _verify_cgroup_controls(observed)
        _require(observed["candidate_value"] == 7 and observed["uid"] == 65534
            and all(observed[name] is True for name in ("source_readonly", "assembly_readonly", "root_readonly", "network_denied", "host_canary_absent"))
            and re.search(r"NoNewPrivs:\s+1", observed["process_status"])
            and re.search(r"CapEff:\s+0+\s", observed["process_status"])
            and observed["pids_max"] == "64" and observed["memory_max"] == "1073741824"
            and observed["cpu_max"].split() == ["100000", "100000"], "QUALIFICATION_BOUNDARY_FAILED")
        proof = {"schema": "task-agent-probe-qualification.v2", "status": "qualified", "profile_ref": probe_profile_ref,
            "runner_identity": identity, "execution_kind": "isolated",
            "boundaries": {"filesystem": "isolated", "network": "denied", "process": "isolated"},
            "controls": observed, "execution_refs": [result["compile_execution"]["receipt_ref"], result["runtime_execution"]["receipt_ref"]],
            "qualification_attempt_ref": self._qualification_attempt_ref, "workspace_mapping_ref": mapping_ref,
            "scope": "linux_amd64_sdk_project_graph_with_frozen_offline_packages", "qualification_candidate": "trusted_control_fixture"}
        return proof, _save(completed, proof)

    def run_probe(self, *, state, source_revision_ref, probe, capability, operation_id):
        from src.mod_agent.task_agent_v2_experiments import _source

        docker, environment = self._environment()
        identity = self.identity(environment)
        _require(identity == capability["runner_identity"], "CAPABILITY_CHANGED")
        reference, source_id, files = _source(state, source_revision_ref)
        baseline = state.load(state.contract["baseline_manifest_ref"])
        _require(Path(baseline["source_root"]).resolve() == self.backend.source_root
            and state.contract["execution_profile_ref"] == self.backend.profile.reference, "CANDIDATE_PROJECT_CHANGED")
        self.backend.profile.verify_fixed_inputs(self.backend.source_root, self.backend.fixed_inputs)
        source_paths = []
        for path in sorted(self.backend.source_root.rglob("*")):
            relative = path.relative_to(self.backend.source_root)
            if any(part.casefold() in _IGNORED or part.casefold().startswith(".env.") for part in relative.parts):
                continue
            _no_links(path)
            if path.is_file():
                name = relative.as_posix()
                _require(not re.search(r"[|\x00-\x1f]", name), "SOURCE_PATH_UNSUPPORTED")
                source_paths.append((path, relative, name))
        prepared_packages = self._bound_packages()
        projected = ["candidate/" + name for _path_value, _relative_value, name in source_paths]
        projected.extend("offline/" + name for name in prepared_packages[2])
        projected.extend(("harness/RepoForge.Probe.csproj", "harness/NuGet.Config",
            "harness/capture.targets", "harness/build-flags.txt", "bundle/artifacts/RepoForge.Probe.dll"))
        projected.extend("harness/" + _path(row["relative_path"]) for row in probe["files"])
        workspace = self.execution_workspaces.allocate(kind="probe", operation_id=operation_id,
            source_revision_ref=reference, source_identity=source_id, projected_paths=projected)
        _require(not workspace.recovered, "EXECUTION_WORKSPACE_ALREADY_USED")
        root, evidence_root = workspace.execution_root, workspace.evidence_root
        candidate = root / "candidate"
        candidate.mkdir(parents=True)
        original = []
        total = 0
        for path, relative, name in source_paths:
            _require(total + path.stat().st_size <= _SOURCE_LIMIT, "SOURCE_LIMIT")
            data = path.read_bytes()
            original.append({"path": name, "sha256": hashlib.sha256(data).hexdigest()})
            if name in baseline["files"]:
                _require(hashlib.sha256(data).hexdigest() == baseline["files"][name]["sha256"], "BASELINE_CHANGED")
                data = files[name].encode("utf-8")
            if path.name.casefold() in {"nuget.config", "web.config", "app.config"}:
                _require(b"packagesourcecredentials" not in data.lower(), "CREDENTIAL_CONFIG_DENIED")
            total += len(data)
            _require(total <= _SOURCE_LIMIT, "SOURCE_LIMIT")
            target = candidate / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        project = candidate / self.config["project_path"]
        _require(project.is_file(), "PROJECT_NOT_IN_CANDIDATE")
        document = ElementTree.fromstring(project.read_bytes())
        _require(document.get("Sdk") or document.find("Sdk") is not None, "SDK_STYLE_PROJECT_REQUIRED")
        result = self._compile_and_run(root, candidate, probe["files"], docker, environment,
            evidence_root=evidence_root, prepared_packages=prepared_packages)
        _require(all(_ref(self.backend.source_root / row["path"])["sha256"] == row["sha256"] for row in original), "ORIGINAL_SOURCE_MUTATED")
        logs = state.save("probe-execution", {"schema": "task-agent-probe-execution.v2", "operation_id": operation_id,
            "source_revision_ref": reference, "source_identity": source_id, "probe": probe,
            "runner_identity": identity, "qualification_refs": capability["evidence_refs"], "execution": result,
            "execution_workspace_ref": workspace.mapping_ref,
            "project_path": self.config["project_path"], "host_home_mounted": False,
            "host_environment_inherited": False, "source_credential_screening": "known_paths_and_nuget_credentials_only",
            "network_enabled": False})
        runtime = result["runtime_execution"]
        return {"execution_origin": "executed", "execution_kind": "isolated", "source_revision_ref": reference,
            "source_identity": source_id, "runner_identity": identity, "exit_code": result["exit_code"],
            "passed": result["passed"], "logs_ref": logs, "compiled_project_path": self.config["project_path"],
            "execution_workspace_ref": workspace.mapping_ref,
            "source_manifest_sha256": canonical_hash(result["source_manifest"]),
            "dependency_manifest_sha256": canonical_hash(result["dependency_manifest"]),
            "observations": {"compiled": result["compiled"], "build_status": result.get("build_status"),
                "dependency_failure": result.get("dependency_failure"),
                "build_diagnostics": result.get("build_diagnostics"),
                "stdout": runtime["stdout"] if runtime else "", "stderr": runtime["stderr"] if runtime else result["compile_execution"]["stderr"]},
            "verification_scope": "exploration_only_linux_sdk_project_not_independent_acceptance"}
