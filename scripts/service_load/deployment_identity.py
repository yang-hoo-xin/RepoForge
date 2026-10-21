"""Read-only deployment identity checks; never start, stop, rebuild or migrate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
PROJECT = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
PRODUCT_ROLES = {"api", "worker", "prepare"}
# Scope: Docker returns only identity/state fields, never Config.Env or health logs.
# 为什么要有它：版本核对不需要凭据，过滤应发生在Docker输出侧而不是先读取完整inspect再脱敏。
INSPECT_FORMAT = (
    '{"id":{{json .Id}},"name":{{json .Name}},'
    '"project":{{json (index .Config.Labels "com.docker.compose.project")}},'
    '"role":{{json (index .Config.Labels "com.docker.compose.service")}},'
    '"image_id":{{json .Image}},"state":{'
    '"Status":{{json .State.Status}},"Running":{{json .State.Running}},'
    '"Paused":{{json .State.Paused}},"Restarting":{{json .State.Restarting}},'
    '"Dead":{{json .State.Dead}},"OOMKilled":{{json .State.OOMKilled}},'
    '"Pid":{{json .State.Pid}},"ExitCode":{{json .State.ExitCode}},'
    '"HealthStatus":{{if .State.Health}}{{json .State.Health.Status}}{{else}}null{{end}}},'
    '"mounts":[{{range $i,$m := .Mounts}}{{if $i}},{{end}}'
    '{"type":{{json $m.Type}},"name":{{json $m.Name}},'
    '"destination":{{json $m.Destination}}}{{end}}]}'
)


def _mounts(value):
    if not isinstance(value, list):
        raise ValueError("VOLUME_RECORD_INVALID")
    rows = []
    for mount in value:
        if (not isinstance(mount, dict) or mount.get("type") != "volume"
                or not isinstance(mount.get("name"), str) or not mount["name"]
                or not isinstance(mount.get("destination"), str)
                or not mount["destination"].startswith("/")):
            # Scope: this contract verifies named volumes, not unknown bind-mount sources.
            # 为什么要有它：旧记录没有bind源身份，不能只比较目的地就把不同宿主文件判为相同。
            raise ValueError("NAMED_VOLUME_IDENTITY_REQUIRED")
        rows.append((mount["name"], mount["destination"]))
    if len({destination for _, destination in rows}) != len(rows):
        raise ValueError("DUPLICATE_VOLUME_DESTINATION")
    return sorted(rows)


def _validate_expected(project, image_id, roles):
    if not isinstance(project, str) or not PROJECT.fullmatch(project):
        raise ValueError("PROJECT_IDENTITY_INVALID")
    if not isinstance(image_id, str) or not IMAGE_ID.fullmatch(image_id):
        raise ValueError("RELEASE_IMAGE_IDENTITY_INVALID")
    if not PRODUCT_ROLES.issubset(roles) or "redis" not in roles:
        raise ValueError("REQUIRED_ROLE_MISSING_FROM_RECEIPT")
    for role, expected in roles.items():
        if (not isinstance(role, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", role)
                or type(expected["count"]) is not int or not 1 <= expected["count"] <= 128
                or not isinstance(expected["image_id"], str)
                or not IMAGE_ID.fullmatch(expected["image_id"])
                or type(expected["healthcheck_required"]) is not bool):
            raise ValueError("ROLE_IDENTITY_INVALID")
        if role in PRODUCT_ROLES and expected["image_id"] != image_id:
            raise ValueError("RECEIPT_PRODUCT_IMAGES_DIFFER")
        _mounts(expected["mounts"])
    if any(roles[role]["count"] != 1 for role in ("api", "prepare", "redis")):
        raise ValueError("SINGLETON_ROLE_COUNT_INVALID")
    return {"project": project, "image_id": image_id, "roles": roles}


def expectations_from_receipt(receipt):
    """Read the original successful launch record; container IDs are not identities."""
    if not isinstance(receipt, dict) or receipt.get("status") != "started":
        raise ValueError("SUCCESSFUL_START_RECEIPT_REQUIRED")
    rows = receipt.get("containers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("RECORDED_CONTAINERS_REQUIRED")
    roles = {}
    identifiers = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not CONTAINER_ID.fullmatch(row["id"]) or row["id"] in identifiers):
            raise ValueError("RECORDED_CONTAINER_ID_INVALID")
        identifiers.add(row["id"])
        role = row.get("role")
        if not isinstance(role, str):
            raise ValueError("RECORDED_ROLE_INVALID")
        state = row.get("state") or {}
        health = state.get("HealthStatus", (state.get("Health") or {}).get("Status"))
        entry = {"count": 1, "image_id": row.get("image_id"), "mounts": row.get("mounts"),
                 "healthcheck_required": health is not None}
        if role in roles:
            if (roles[role]["image_id"] != entry["image_id"]
                    or roles[role]["healthcheck_required"] != entry["healthcheck_required"]
                    or _mounts(roles[role]["mounts"]) != _mounts(entry["mounts"])):
                raise ValueError("RECORDED_REPLICAS_DIFFER")
            roles[role]["count"] += 1
        else:
            roles[role] = entry
    return _validate_expected(receipt.get("project"), receipt.get("image_id"), roles)


def expectations_from_compose(configured, image_id, image_ids, role_counts):
    """Initial launch uses the same checker, with prelaunch Compose expectations."""
    roles = {}
    for role, count in role_counts.items():
        service = configured["services"][role]
        mounts = []
        for mount in service.get("volumes", []):
            if mount.get("type") != "volume":
                raise ValueError("NAMED_VOLUME_IDENTITY_REQUIRED")
            volume = configured["volumes"][mount["source"]]
            mounts.append({"type": "volume", "name": volume["name"], "destination": mount["target"]})
        healthcheck = service.get("healthcheck") or {}
        roles[role] = {"count": count, "image_id": image_ids[role], "mounts": mounts,
                       "healthcheck_required": bool(healthcheck and not healthcheck.get("disable", False)
                                                    and healthcheck.get("test") != ["NONE"])}
    return _validate_expected(configured["name"], image_id, roles)


def inspect_project(project, command):
    """Only list and minimally inspect containers in the specified Compose project."""
    if not isinstance(project, str) or not PROJECT.fullmatch(project):
        raise ValueError("PROJECT_IDENTITY_INVALID")
    identifiers = command(["docker", "ps", "-a", "--no-trunc", "--filter",
                           "label=com.docker.compose.project=" + project, "--format", "{{.ID}}"]).splitlines()
    if (len(identifiers) > 128 or len(set(identifiers)) != len(identifiers)
            or any(not CONTAINER_ID.fullmatch(identifier) for identifier in identifiers)):
        raise ValueError("CONTAINER_LIST_INVALID")
    observations = []
    for identifier in identifiers:
        row = json.loads(command(["docker", "container", "inspect", "--format", INSPECT_FORMAT, identifier]))
        if not isinstance(row, dict) or row.get("id") != identifier:
            raise ValueError("INSPECT_CONTAINER_ID_CHANGED")
        if set(row) != {"id", "name", "project", "role", "image_id", "state", "mounts"}:
            raise ValueError("INSPECT_FIELDS_INVALID")
        if (not isinstance(row["state"], dict)
                or set(row["state"]) != {"Status", "Running", "Paused", "Restarting", "Dead",
                                         "OOMKilled", "Pid", "ExitCode", "HealthStatus"}
                or not isinstance(row["mounts"], list)
                or any(not isinstance(mount, dict) or set(mount) != {"type", "name", "destination"}
                       for mount in row["mounts"])):
            raise ValueError("INSPECT_FIELDS_INVALID")
        observations.append(row)
    return observations


def check_observations(expected, observations):
    """Compare real labels, immutable image IDs, role counts, volumes and state."""
    checks = []

    def check(name, passed, **details):
        checks.append({"check": name, "passed": bool(passed), **details})

    identifiers = [row.get("id") for row in observations]
    check("unique_valid_container_ids",
          all(isinstance(identifier, str) and CONTAINER_ID.fullmatch(identifier) for identifier in identifiers)
          and len(set(identifiers)) == len(identifiers))
    counts = Counter(row.get("role") for row in observations)
    check("role_counts", counts == Counter({role: value["count"] for role, value in expected["roles"].items()}),
          expected={role: value["count"] for role, value in expected["roles"].items()},
          actual=dict(counts))
    for row in observations:
        role = row.get("role")
        identity = {"container_id": row.get("id"), "role": role}
        check("compose_project_label", row.get("project") == expected["project"], **identity)
        spec = expected["roles"].get(role)
        check("registered_service_label", spec is not None, **identity)
        if spec is None:
            continue
        check("registered_image_id", row.get("image_id") == spec["image_id"], **identity)
        try:
            mounts_match = _mounts(row.get("mounts")) == _mounts(spec["mounts"])
        except ValueError:
            mounts_match = False
        check("registered_named_volumes", mounts_match, **identity)
        state = row.get("state") or {}
        safe_state = all(state.get(name) is False for name in ("Paused", "Restarting", "Dead", "OOMKilled"))
        if role == "prepare":
            correct_state = (state.get("Status") == "exited" and state.get("Running") is False
                             and type(state.get("ExitCode")) is int and state["ExitCode"] == 0
                             and type(state.get("Pid")) is int and state["Pid"] == 0)
        else:
            correct_state = (state.get("Status") == "running" and state.get("Running") is True
                             and type(state.get("Pid")) is int and state["Pid"] > 0)
            # Scope: a recorded/configured health check cannot disappear into an assumed healthy state.
            # 为什么要有它：运行中的进程不等于健康，原有健康检查缺失应失败；未配置检查的Worker不凭空要求健康数据。
            health = state.get("HealthStatus")
            check("registered_service_health",
                  health == "healthy" if spec["healthcheck_required"] else health in (None, "healthy"),
                  **identity)
        check("prepare_completed" if role == "prepare" else "service_running", safe_state and correct_state,
              **identity)
    passed = all(item["passed"] for item in checks)
    return {"schema": "service-deployment-identity.v1", "status": "verified" if passed else "mismatch",
            "passed": passed, "project": expected["project"], "checks": checks, "containers": observations,
            "scope": "Read-only deployment identity and process state; not HTTP, restart, data recovery or delivery acceptance"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 180 or not re.fullmatch(r"[0-9a-f]{64}", args.receipt_sha256):
        parser.error("Use a 1..180 second timeout and the original receipt SHA256")
    raw = args.receipt.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != args.receipt_sha256:
        raise ValueError("START_RECEIPT_HASH_CHANGED")
    expected = expectations_from_receipt(json.loads(raw))
    started = time.monotonic()

    def command(command_args):
        remaining = args.timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("IDENTITY_CHECK_DEADLINE")
        completed = subprocess.run(command_args, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=min(10, remaining))
        if completed.returncode:
            raise RuntimeError("DOCKER_READ_UNAVAILABLE")
        return completed.stdout.strip()

    # Scope: create the new receipt before Docker access; historical files stay untouched.
    # 为什么要有它：输出路径冲突不能在查询之后才发现，更不能覆盖启动原件或以前的验收结果。
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            result = check_observations(expected, inspect_project(expected["project"], command))
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired, TimeoutError) as error:
            result = {"schema": "service-deployment-identity.v1", "status": "unavailable",
                      "passed": False, "error_type": type(error).__name__,
                      "scope": "No verified deployment conclusion; read-only Docker query failed"}
        result.update(receipt_ref={"path": str(args.receipt.resolve()), "sha256": actual_hash},
                      elapsed_seconds=time.monotonic() - started)
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "output": str(args.output)}))
    return 0 if result["passed"] else 3 if result["status"] == "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
