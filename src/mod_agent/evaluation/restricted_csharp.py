"""Fixed-image C# execution for a trusted host judge, without judge-data mounts.

Only source bytes and one JSON operation cross the boundary. Compilation uses
the SDK compiler directly: no project builds, package restore, analyzers or
model-selected commands. Runtime files live on bounded tmpfs; observations
return on bounded stdout rather than an untrusted host-writable bind mount.
为什么要有它：隔离必须由真实容器配置和观测支持，不能由调用者填一个标签来声明。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

IMAGE = "mcr.microsoft.com/dotnet/sdk@sha256:4da9dff9f77f86ae7380a41332b838118f25b96eb27c4ac776325b285a331be0"
SDK_VERSION = "7.0.401"
RUNTIME_VERSION = "7.0.11"
COMPILE_TIMEOUT = 60
RUN_TIMEOUT = 15
OUTPUT_LIMIT = 1_048_576
SOURCE_LIMIT = 1_048_576
OPERATION_LIMIT = 65_536
_COMPILE = r'''set -eu
printf '/nologo\n/target:exe\n/langversion:latest\n/deterministic+\n/out:/output/Subject.dll\n' > /output/compiler.rsp
for reference in /usr/share/dotnet/packs/Microsoft.NETCore.App.Ref/7.0.11/ref/net7.0/*.dll; do
  printf '/reference:%s\n' "$reference" >> /output/compiler.rsp
done
dotnet /usr/share/dotnet/sdk/7.0.401/Roslyn/bincore/csc.dll @/output/compiler.rsp @/inputs/sources.rsp
'''
_ENV = ("DOTNET_CLI_TELEMETRY_OPTOUT=1", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1",
        "DOTNET_NOLOGO=1", "DOTNET_CLI_HOME=/tmp/dotnet-home")


@dataclass(frozen=True)
class CSharpCompilation:
    output_root: str
    succeeded: bool
    binary_ref: dict[str, str] | None
    runtimeconfig_ref: dict[str, str] | None
    receipt_ref: dict[str, str]


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> bytes:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    if path.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise ValueError("RESTRICTED_CSHARP_LINK_NOT_ALLOWED")
    return path.read_bytes()


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _hash(_read_bytes(path))}


def _bound_bytes(path: Path, reference: dict[str, str], error: str) -> bytes:
    value = _read_bytes(path)
    if {"path": str(path.resolve()), "sha256": _hash(value)} != reference:
        raise ValueError(error)
    return value


def _save(path: Path, value: Any) -> dict[str, str]:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode()
    with path.open("xb") as stream:
        stream.write(data)
    reference = {"path": str(path.resolve()), "sha256": _hash(data)}
    _bound_bytes(path, reference, "RESTRICTED_CSHARP_RECEIPT_PUBLICATION_CHANGED")
    return reference


def _docker() -> list[str]:
    executable = shutil.which("docker") or "C:/Program Files/Docker/Docker/resources/bin/docker.exe"
    if not Path(executable).is_file():
        raise ValueError("RESTRICTED_CSHARP_DOCKER_UNAVAILABLE")
    return [executable, "--context", "desktop-linux"]


def _command(root: Path, name: str, arguments: list[str]) -> subprocess.CompletedProcess:
    started = time.monotonic()
    result = subprocess.run(arguments, capture_output=True, timeout=20)
    _save(root / (name + ".json"), {"argv": arguments, "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": result.stdout.decode("utf-8", "replace"),
        "stderr": result.stderr.decode("utf-8", "replace")})
    return result


def _source_identity(root: Path) -> dict[str, str]:
    result = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(directory) / name
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            if (path.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    or not path.resolve().is_relative_to(root.resolve())):
                raise ValueError("RESTRICTED_CSHARP_LINK_NOT_ALLOWED")
        for name in files:
            path = Path(directory) / name
            result[path.relative_to(root).as_posix()] = _hash(path.read_bytes())
    return result


def _same_mount_source(actual: str, expected: Path) -> bool:
    if os.path.normcase(os.path.normpath(actual)) == os.path.normcase(str(expected.resolve())):
        return True
    value = expected.resolve().as_posix()
    return bool(re.match(r"^[A-Za-z]:/", value) and actual == (
        "/run/desktop/mnt/host/" + value[0].lower() + value[2:]))


def _verify_inspect(document, *, token, entrypoint, command, mounts, runtime):
    config, host = document["Config"], document["HostConfig"]
    expected_tmpfs = {"/tmp": "rw,nosuid,nodev,noexec,size=64m"}
    if runtime:
        expected_tmpfs["/operation-output"] = "rw,nosuid,nodev,noexec,size=64m"
    predicates = [config["Image"] == IMAGE, config["User"] == "65534:65534",
        config["Entrypoint"] == [entrypoint], config["Cmd"] == command,
        config.get("Labels", {}).get("codex.restricted-csharp") == token,
        all(item in config["Env"] for item in _ENV), host["NetworkMode"] == "none",
        host["ReadonlyRootfs"] is True, host["Privileged"] is False,
        host["CapDrop"] == ["ALL"], not host.get("CapAdd"),
        host["SecurityOpt"] in (["no-new-privileges"], ["no-new-privileges=true"]),
        host["PidsLimit"] == 64, host["Memory"] == 1_073_741_824,
        host["MemorySwap"] == 1_073_741_824, host["NanoCpus"] == 1_000_000_000,
        host["Tmpfs"] == expected_tmpfs, host.get("Init") is True,
        not host.get("Devices"), not host.get("VolumesFrom"), not host.get("PidMode"),
        host["LogConfig"]["Type"] == "none"]
    actual = document["Mounts"]
    predicates.append(len(actual) == len(mounts))
    for destination, (source, readonly) in mounts.items():
        matching = [item for item in actual if item["Destination"] == destination]
        predicates.append(len(matching) == 1 and matching[0]["Type"] == "bind"
            and matching[0]["RW"] is not readonly
            and _same_mount_source(matching[0]["Source"], source))
    if not all(predicates):
        raise ValueError("RESTRICTED_CSHARP_INSPECT_POLICY_MISMATCH")


def _attach(arguments, *, stdin, timeout, stop, output_limit=OUTPUT_LIMIT):
    if type(output_limit) is not int or not 0 < output_limit <= 134_217_728:
        raise ValueError("RESTRICTED_CSHARP_OUTPUT_LIMIT_INVALID")
    process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    excess = threading.Event()

    def collect(name, stream):
        while chunk := stream.read(8192):
            remaining = output_limit - len(buffers[name])
            buffers[name].extend(chunk[:max(remaining, 0)])
            if len(chunk) > remaining:
                excess.set()

    def supply():
        try:
            process.stdin.write(stdin)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    readers = [threading.Thread(target=collect, args=(name, getattr(process, name)), daemon=True)
               for name in buffers]
    writer = threading.Thread(target=supply, daemon=True)
    started = time.monotonic()
    timed_out = False
    primary_error = None
    cleanup_errors = []

    def snapshot():
        return {"argv": arguments, "exit_code": process.returncode,
            "timed_out": timed_out, "output_limit_exceeded": excess.is_set(),
            "elapsed_seconds": time.monotonic() - started,
            "local_process_id": process.pid, "local_process_reaped": process.poll() is not None,
            "local_cleanup_errors": list(cleanup_errors),
            **{name: bytes(value).decode("utf-8", "replace") for name, value in buffers.items()},
            **{name + "_sha256": _hash(bytes(value)) for name, value in buffers.items()},
            "output_hash_scope": "captured prefix only" if excess.is_set() else "complete stream"}

    try:
        for reader in readers:
            reader.start()
        writer.start()
        while process.poll() is None:
            if excess.is_set() or time.monotonic() - started >= timeout:
                timed_out = not excess.is_set()
                stop()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                break
            time.sleep(0.02)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        except BaseException as cleanup_error:
            cleanup_errors.append({"operation": "reap_attach_process", "error": str(cleanup_error)})
        for thread in [*readers, writer]:
            if thread.ident is not None:
                try:
                    thread.join(timeout=5)
                    if thread.is_alive():
                        cleanup_errors.append({"operation": "join_pipe_thread", "error": "thread still alive"})
                except BaseException as cleanup_error:
                    cleanup_errors.append({"operation": "join_pipe_thread", "error": str(cleanup_error)})
        if not any(thread.is_alive() for thread in [*readers, writer]):
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError as cleanup_error:
                    cleanup_errors.append({"operation": "close_pipe", "error": str(cleanup_error)})
        if primary_error is not None:
            primary_error.attach_execution = snapshot()
        elif cleanup_errors:
            error = RuntimeError("RESTRICTED_CSHARP_ATTACH_CLEANUP_FAILED")
            error.attach_execution = snapshot()
            raise error
    return snapshot()


def _execute(root, *, mounts, entrypoint, command, stdin, runtime):
    prefix, token = _docker(), uuid4().hex
    argv = [*prefix, "create", "--name=codex-csharp-" + token,
        "--label=codex.restricted-csharp=" + token, "--pull=never", "--platform=linux/amd64",
        "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--user=65534:65534", "--pids-limit=64", "--memory=1g", "--memory-swap=1g", "--cpus=1",
        "--init", "--interactive", "--log-driver=none", "--workdir=/tmp",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m", "--entrypoint=" + entrypoint]
    if runtime:
        argv.append("--tmpfs=/operation-output:rw,nosuid,nodev,noexec,size=64m")
    argv.extend("--env=" + value for value in _ENV)
    for destination, (source, readonly) in mounts.items():
        argv.extend(["--mount", f"type=bind,source={source},target={destination}" + (",readonly" if readonly else "")])
    argv.extend([IMAGE, *command])
    created = _command(root, "create", argv)
    if created.returncode:
        raise RuntimeError("RESTRICTED_CSHARP_CONTAINER_CREATE_FAILED")
    container = created.stdout.decode().strip()
    if not re.fullmatch(r"[a-f0-9]{64}", container):
        raise RuntimeError("RESTRICTED_CSHARP_CONTAINER_ID_INVALID")
    primary_error = None
    try:
        before = _command(root, "inspect-before", [*prefix, "inspect", container])
        if before.returncode:
            raise RuntimeError("RESTRICTED_CSHARP_INSPECT_FAILED")
        document = json.loads(before.stdout)[0]
        _verify_inspect(document, token=token, entrypoint=entrypoint, command=command,
                        mounts=mounts, runtime=runtime)
        result = _attach([*prefix, "start", "--attach", "--interactive", container],
            stdin=stdin, timeout=RUN_TIMEOUT if runtime else COMPILE_TIMEOUT,
            stop=lambda: _command(root, "kill", [*prefix, "kill", container]))
        after = _command(root, "inspect-after", [*prefix, "inspect", container])
        if after.returncode:
            raise RuntimeError("RESTRICTED_CSHARP_RESULT_INSPECT_FAILED")
        final = json.loads(after.stdout)[0]
        _verify_inspect(final, token=token, entrypoint=entrypoint, command=command,
                        mounts=mounts, runtime=runtime)
        result.update(container_id=container, image=IMAGE, image_id=final["Image"],
            container_state=final["State"], inspect_policy_verified=True,
            stdin_sha256=_hash(stdin), docker_executable_ref=_ref(Path(prefix[0])),
            implementation_ref=_ref(Path(__file__)))
        result["cli_exit_code"] = result["exit_code"]
        result["exit_code"] = None if final["State"]["Running"] else final["State"]["ExitCode"]
        result["completed"] = (not final["State"]["Running"]
            and result["cli_exit_code"] == result["exit_code"]
            and not result["timed_out"] and not result["output_limit_exceeded"])
        _save(root / "execution.json", result)
        return result
    except BaseException as exc:
        primary_error = exc
        if hasattr(exc, "attach_execution"):
            try:
                _save(root / "attach-failure.json", {"error_type": type(exc).__name__,
                    "error": str(exc), "execution": exc.attach_execution})
            except BaseException as recording_error:
                exc.attach_failure_recording_error = str(recording_error)
        raise
    finally:
        try:
            removed = _command(root, "cleanup", [*prefix, "rm", "--force", container])
            if removed.returncode:
                raise RuntimeError("RESTRICTED_CSHARP_CONTAINER_CLEANUP_FAILED")
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.container_cleanup_error = str(cleanup_error)


def compile_csharp(*, sources: Mapping[str, bytes], output_root: str | Path) -> CSharpCompilation:
    """Compile supplied public sources only; output_root must be a new directory."""
    if not isinstance(sources, Mapping) or not 1 <= len(sources) <= 16:
        raise ValueError("RESTRICTED_CSHARP_SOURCES_INVALID")
    sources = dict(sources)
    for name, content in sources.items():
        parts = PurePosixPath(name).parts if isinstance(name, str) else ()
        if (not parts or any(part in {".", ".."} for part in parts)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.cs", name)
                or type(content) is not bytes):
            raise ValueError("RESTRICTED_CSHARP_SOURCE_PATH_INVALID")
    if sum(map(len, sources.values())) > SOURCE_LIMIT:
        raise ValueError("RESTRICTED_CSHARP_SOURCE_LIMIT")
    if len({name.casefold() for name in sources}) != len(sources):
        raise ValueError("RESTRICTED_CSHARP_SOURCE_PATH_COLLISION")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs, output = root / "public-input", root / "compiler-output"
    inputs.mkdir()
    output.mkdir()
    for name, content in sources.items():
        target = inputs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    response_bytes = "\n".join("/inputs/" + name for name in sorted(sources)).encode()
    (inputs / "sources.rsp").write_bytes(response_bytes)
    original = {**{name: _hash(content) for name, content in sources.items()},
                "sources.rsp": _hash(response_bytes)}
    if _source_identity(inputs) != original:
        raise ValueError("RESTRICTED_CSHARP_INPUT_STAGING_CHANGED")
    result = _execute(root, mounts={"/inputs": (inputs, True), "/output": (output, False)},
        entrypoint="/bin/sh", command=["-c", _COMPILE], stdin=b"", runtime=False)
    binary = output / "Subject.dll"
    succeeded = (result["exit_code"] == result["container_state"]["ExitCode"] == 0
        and result["completed"] and binary.is_file())
    runtimeconfig = output / "Subject.runtimeconfig.json"
    runtimeconfig_bytes = None
    if succeeded:
        runtimeconfig_bytes = json.dumps({"runtimeOptions": {"tfm": "net7.0",
            "framework": {"name": "Microsoft.NETCore.App", "version": RUNTIME_VERSION}}}).encode()
        runtimeconfig.write_bytes(runtimeconfig_bytes)
    if _source_identity(inputs) != original:
        raise RuntimeError("RESTRICTED_CSHARP_INPUT_CHANGED")
    output_manifest = _source_identity(output)
    binary_ref = runtimeconfig_ref = None
    if succeeded:
        binary_ref = {"path": str(binary.resolve()), "sha256": output_manifest["Subject.dll"]}
        runtimeconfig_ref = {"path": str(runtimeconfig.resolve()),
                             "sha256": _hash(runtimeconfig_bytes)}
        _bound_bytes(binary, binary_ref, "RESTRICTED_CSHARP_BINARY_CHANGED")
        _bound_bytes(runtimeconfig, runtimeconfig_ref, "RESTRICTED_CSHARP_BINARY_CHANGED")
    payload = {"schema_version": "restricted_csharp_compilation_v1", "image": IMAGE,
        "sdk_version": SDK_VERSION, "runtime_version": RUNTIME_VERSION, "succeeded": succeeded,
        "source_manifest": original, "output_manifest": output_manifest,
        "binary_ref": binary_ref, "runtimeconfig_ref": runtimeconfig_ref,
        "execution_ref": _ref(root / "execution.json")}
    receipt = _save(root / "compilation.json", payload)
    return CSharpCompilation(str(root), succeeded, payload["binary_ref"], payload["runtimeconfig_ref"], receipt)


def run_csharp(*, compilation: CSharpCompilation, operation: Any, output_root: str | Path) -> dict[str, Any]:
    """Run one JSON operation; runtime files stay ephemeral and never return to host."""
    stdin = json.dumps(operation, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(stdin) > OPERATION_LIMIT:
        raise ValueError("RESTRICTED_CSHARP_OPERATION_LIMIT")
    prepared = Path(compilation.output_root).resolve()
    receipt = json.loads(_bound_bytes(prepared / "compilation.json", compilation.receipt_ref,
        "RESTRICTED_CSHARP_COMPILATION_RECEIPT_CHANGED"))
    if (receipt["schema_version"] != "restricted_csharp_compilation_v1"
            or receipt["sdk_version"] != SDK_VERSION or receipt["runtime_version"] != RUNTIME_VERSION
            or receipt["image"] != IMAGE or not receipt["succeeded"] or not compilation.succeeded
            or any(receipt[name] != getattr(compilation, name) for name in ("binary_ref", "runtimeconfig_ref"))):
        raise ValueError("RESTRICTED_CSHARP_COMPILATION_NOT_ADMITTED")
    selected_bytes = {}
    for name, reference in (("Subject.dll", compilation.binary_ref),
                            ("Subject.runtimeconfig.json", compilation.runtimeconfig_ref)):
        path = prepared / "compiler-output" / name
        selected_bytes[name] = _bound_bytes(path, reference, "RESTRICTED_CSHARP_BINARY_CHANGED")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    assembly = root / "public-assembly"
    assembly.mkdir()
    for name, content in selected_bytes.items():
        (assembly / name).write_bytes(content)
    original = {name: _hash(content) for name, content in selected_bytes.items()}
    if _source_identity(assembly) != original:
        raise ValueError("RESTRICTED_CSHARP_ASSEMBLY_STAGING_CHANGED")
    result = _execute(root, mounts={"/assembly": (assembly, True)}, entrypoint="dotnet",
        command=["/assembly/Subject.dll"], stdin=stdin, runtime=True)
    if _source_identity(assembly) != original:
        raise RuntimeError("RESTRICTED_CSHARP_ASSEMBLY_CHANGED")
    result.update(compilation_receipt_ref=compilation.receipt_ref, assembly_manifest=original,
        operation_output="bounded 64MiB tmpfs; process observations returned on stdout",
        runtime_file_outputs_returned=False, runtime_files_persistent=False,
        host_judge_material_mounted=False)
    result["receipt_ref"] = _save(root / "run.json", result)
    return result
