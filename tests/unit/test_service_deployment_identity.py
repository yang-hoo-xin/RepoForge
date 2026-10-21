"""Saved deployment observations and command doubles only; no Docker or network."""
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "deployment_identity", ROOT / "scripts/service_load/deployment_identity.py"
)
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)
RECEIPT = ROOT / "artifacts/eval_driven_20261016/service-deployment/instance-v1/start-result.json"


@pytest.fixture
def receipt():
    return json.loads(RECEIPT.read_bytes())


def recorded_observations(receipt):
    # Historical role and project were validated by the launch owner, not observed again here.
    rows = []
    for original in receipt["containers"]:
        row = {name: deepcopy(original[name]) for name in ("id", "name", "role", "image_id", "mounts")}
        row["project"] = receipt["project"]
        row["state"] = {name: original["state"][name]
                        for name in ("Status", "Running", "Paused", "Restarting", "Dead", "OOMKilled", "Pid", "ExitCode")}
        row["state"]["HealthStatus"] = (original["state"].get("Health") or {}).get("Status")
        rows.append(row)
    return rows


def configured_compose(receipt):
    config = {"name": receipt["project"], "services": {},
              "volumes": {"data": {"name": receipt["project"] + "_service-data"},
                          "redis": {"name": receipt["project"] + "_redis-data"}},
              "networks": {"backend": {"name": receipt["project"] + "_backend"}}}
    for row in receipt["containers"]:
        role = row["role"]
        volume = "redis" if role == "redis" else "data"
        config["services"][role] = {
            "image": "redis:7.4-alpine" if role == "redis" else receipt["image_tag"],
            "volumes": [{"type": "volume", "source": volume, "target": row["mounts"][0]["destination"]}],
        }
        if row["state"].get("Health"):
            config["services"][role]["healthcheck"] = {"test": ["CMD", "controlled-health-check"]}
    return config


def test_saved_real_deployment_is_offline_positive_without_changing_receipt(receipt):
    before = RECEIPT.read_bytes()
    expected = identity.expectations_from_receipt(receipt)
    assert identity.check_observations(expected, recorded_observations(receipt))["passed"]
    assert RECEIPT.read_bytes() == before


def test_recreated_container_ids_are_allowed(receipt):
    rows = recorded_observations(receipt)
    for index, row in enumerate(rows):
        row["id"] = f"{index + 10:064x}"
        row["state"]["Pid"] = 0 if row["role"] == "prepare" else index + 100
    assert identity.check_observations(identity.expectations_from_receipt(receipt), rows)["passed"]


def test_shared_checker_rejects_duplicate_observations_even_with_correct_role_counts(receipt):
    rows = recorded_observations(receipt)
    workers = [row for row in rows if row["role"] == "worker"]
    workers[1]["id"] = workers[0]["id"]
    report = identity.check_observations(identity.expectations_from_receipt(receipt), rows)
    assert not report["passed"]
    assert next(check for check in report["checks"] if check["check"] == "unique_valid_container_ids")["passed"] is False


@pytest.mark.parametrize("role", ["api", "redis"])
def test_registered_healthcheck_must_not_disappear(receipt, role):
    rows = recorded_observations(receipt)
    next(row for row in rows if row["role"] == role)["state"]["HealthStatus"] = None
    report = identity.check_observations(identity.expectations_from_receipt(receipt), rows)
    assert not report["passed"]
    assert any(check["check"] == "registered_service_health" and check["role"] == role
               and not check["passed"] for check in report["checks"])


def test_current_receipt_minimal_state_keeps_health_requirement(receipt):
    receipt["containers"] = recorded_observations(receipt)
    expected = identity.expectations_from_receipt(receipt)
    assert expected["roles"]["api"]["healthcheck_required"]
    assert expected["roles"]["redis"]["healthcheck_required"]
    assert not expected["roles"]["worker"]["healthcheck_required"]


@pytest.mark.parametrize("fault", [
    "image", "volume_name", "volume_destination", "extra_volume", "missing_worker", "extra_worker",
    "project", "service_label", "prepare_running", "prepare_failed", "worker_exited",
    "oom", "paused", "restarting", "dead", "unhealthy", "bind_mount",
])
def test_identity_state_and_role_mismatches_fail(receipt, fault):
    rows = recorded_observations(receipt)
    worker = next(row for row in rows if row["role"] == "worker")
    prepare = next(row for row in rows if row["role"] == "prepare")
    if fault == "image":
        worker["image_id"] = "sha256:" + "0" * 64
    elif fault == "volume_name":
        worker["mounts"][0]["name"] = "different-old-data"
    elif fault == "volume_destination":
        worker["mounts"][0]["destination"] = "/wrong"
    elif fault == "extra_volume":
        worker["mounts"].append({"type": "volume", "name": "extra", "destination": "/other"})
    elif fault == "missing_worker":
        rows.remove(worker)
    elif fault == "extra_worker":
        rows.append(deepcopy(worker))
    elif fault == "project":
        worker["project"] = "different-project"
    elif fault == "service_label":
        worker["role"] = "other-service"
    elif fault == "prepare_running":
        prepare["state"].update(Status="running", Running=True, Pid=1)
    elif fault == "prepare_failed":
        prepare["state"]["ExitCode"] = 1
    elif fault == "worker_exited":
        worker["state"].update(Status="exited", Running=False, Pid=0)
    elif fault == "unhealthy":
        worker["state"]["HealthStatus"] = "unhealthy"
    elif fault == "bind_mount":
        worker["mounts"][0]["type"] = "bind"
    else:
        worker["state"][{"oom": "OOMKilled", "paused": "Paused", "restarting": "Restarting", "dead": "Dead"}[fault]] = True
    report = identity.check_observations(identity.expectations_from_receipt(receipt), rows)
    assert not report["passed"] and report["status"] == "mismatch"


@pytest.mark.parametrize("fault", ["failed_start", "mixed_images", "missing_role", "duplicate_id"])
def test_invalid_expected_record_cannot_define_success(receipt, fault):
    if fault == "failed_start":
        receipt["status"] = "failed"
    elif fault == "mixed_images":
        receipt["containers"][0]["image_id"] = "sha256:" + "0" * 64
    elif fault == "missing_role":
        receipt["containers"] = [row for row in receipt["containers"] if row["role"] != "prepare"]
    else:
        receipt["containers"].append(deepcopy(receipt["containers"][0]))
    with pytest.raises(ValueError):
        identity.expectations_from_receipt(receipt)


def fake_docker(rows, calls):
    def command(argv):
        calls.append(argv)
        if argv[1] == "ps":
            assert argv[2:4] == ["-a", "--no-trunc"]
            return "\n".join(row["id"] for row in rows)
        assert argv[:4] == ["docker", "container", "inspect", "--format"]
        assert argv[4] == identity.INSPECT_FORMAT
        assert ".Config.Env" not in argv[4] and ".State.Health.Log" not in argv[4]
        return json.dumps(next(row for row in rows if row["id"] == argv[-1]))
    return command


def test_shared_query_is_minimal_read_only_and_validates_labels(receipt):
    rows, calls = recorded_observations(receipt), []
    actual = identity.inspect_project(receipt["project"], fake_docker(rows, calls))
    assert actual == rows
    assert len(calls) == 1 + len(rows)
    assert all(call[1] in {"ps", "container"} for call in calls)
    assert not {"Env", "Health.Log", "Config"}.intersection(json.dumps(actual).split('"'))


def test_unexpected_sensitive_fields_cannot_enter_query_result(receipt):
    rows = recorded_observations(receipt)
    rows[0]["state"]["Health.Log"] = "controlled-sensitive-marker"
    with pytest.raises(ValueError, match="INSPECT_FIELDS_INVALID"):
        identity.inspect_project(receipt["project"], fake_docker(rows, []))


def test_initial_compose_and_retained_record_share_expectations(receipt):
    expected = identity.expectations_from_receipt(receipt)
    initial = identity.expectations_from_compose(
        configured_compose(receipt), receipt["image_id"],
        {row["role"]: row["image_id"] for row in receipt["containers"]},
        {"api": 1, "worker": 2, "prepare": 1, "redis": 1},
    )
    assert initial == expected


@pytest.mark.parametrize("unavailable", [False, "oserror", "docker_exit", "timeout"])
def test_cli_uses_original_record_and_exclusive_new_output(receipt, tmp_path, monkeypatch, unavailable):
    rows, calls = recorded_observations(receipt), []
    read = fake_docker(rows, calls)
    output = tmp_path / "identity.json"
    before = RECEIPT.read_bytes()

    def run(argv, **kwargs):
        assert kwargs["capture_output"] and 0 < kwargs["timeout"] <= 10
        if unavailable == "oserror":
            raise OSError("controlled unavailable, not a deployment mismatch")
        if unavailable == "docker_exit":
            return SimpleNamespace(returncode=1, stdout="", stderr="controlled private error must not be copied")
        if unavailable == "timeout":
            raise identity.subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout=read(argv), stderr="")

    monkeypatch.setattr(identity.subprocess, "run", run)
    argv = ["--receipt", str(RECEIPT), "--receipt-sha256", hashlib.sha256(before).hexdigest(),
            "--output", str(output)]
    assert identity.main(argv) == (3 if unavailable else 0)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == ("unavailable" if unavailable else "verified")
    assert "controlled private" not in output.read_text(encoding="utf-8")
    assert RECEIPT.read_bytes() == before
    previous_calls = len(calls)
    with pytest.raises(FileExistsError):
        identity.main(argv)
    assert len(calls) == previous_calls


@pytest.mark.parametrize("fault", [None, "image", "volume", "missing_worker"])
def test_first_launch_normal_caller_uses_shared_checks_without_full_inspect(receipt, tmp_path, monkeypatch, fault):
    spec = importlib.util.spec_from_file_location(
        "identity_start_instance", ROOT / "artifacts/eval_driven_20261016/service-deployment/start_instance.py"
    )
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    monkeypatch.setattr(launch, "INSTANCE", tmp_path / "new-instance")
    monkeypatch.setattr(launch.sys, "argv", ["start_instance.py", "--execute", "--image-tag", receipt["image_tag"],
                                           "--image-id", receipt["image_id"]])
    rows, calls = recorded_observations(receipt), []
    worker = next(row for row in rows if row["role"] == "worker")
    if fault == "image":
        worker["image_id"] = "sha256:" + "0" * 64
    elif fault == "volume":
        worker["mounts"][0]["name"] = "wrong-data"
    elif fault == "missing_worker":
        rows.remove(worker)
    read = fake_docker(rows, calls)
    started = False
    checks = []
    shared_check = launch.check_observations

    def checked(expected, observations):
        checks.append((expected, observations))
        return shared_check(expected, observations)

    monkeypatch.setattr(launch, "check_observations", checked)

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def bind(self, _address):
            pass

    monkeypatch.setattr(launch.socket, "socket", Probe)

    def run(argv, **_kwargs):
        nonlocal started
        if argv[1:3] == ["image", "inspect"]:
            assert argv[-2:] == ["--format", "{{.Id}}"]
            result = next(row["image_id"] for row in rows if row["role"] == "redis") if argv[3] == "redis:7.4-alpine" else receipt["image_id"]
        elif "config" in argv:
            result = json.dumps(configured_compose(receipt))
        elif "up" in argv:
            started, result = True, ""
        elif argv[1] in {"volume", "network"} or argv[1] == "ps" and not started:
            result = ""
        else:
            result = read(argv)
        return SimpleNamespace(returncode=0, stdout=result, stderr="")

    monkeypatch.setattr(launch.subprocess, "run", run)
    assert launch.main() == (1 if fault else 0)
    assert len(checks) == 1
    result = json.loads((launch.INSTANCE / "start-result.json").read_text(encoding="utf-8"))
    assert result["identity_check"]["passed"] is (fault is None)
    assert result["status"] == ("failed" if fault else "started")
    if fault:
        assert result["error"] == "DEPLOYMENT_IDENTITY_MISMATCH"
    assert result["containers"] == rows
