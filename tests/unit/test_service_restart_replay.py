"""Controlled HTTP responses only; no network, containers, or business execution."""
import importlib.util
import json
import time
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("restart_replay", Path(__file__).parents[2] / "scripts/service_load/restart_replay.py")
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)
URL = "http://127.0.0.1:18088"
FINISHED = time.time()


def job():
    return {"id": "a" * 32, "state": "completed", "attempt": 1, "finished_at": FINISHED, "result": {"status": "completed",
        "results": [{"metadata": {"source_path": replay.SOURCE}}],
        "usage": {"answer_transport_calls": 0, "rewrite_transport_calls": 0}}}


def prepare():
    def call(method, _path, _body, _key, _timeout):
        return {"status": 202, "body": {"created": True, "job": job()}} if method == "POST" else {"status": 200, "body": job()}
    return replay.observe("prepare", URL, call, key="fixed-key", retention_seconds=86400)


def test_prepare_and_verify_keep_original_request_result_and_attempt():
    receipt = prepare()
    assert receipt["passed"] and receipt["idempotency_key"] == "fixed-key"
    before = deepcopy(receipt)
    calls = []

    def call(method, path, body, key, _timeout):
        calls.append((method, path, deepcopy(body), key))
        if method == "GET":
            return {"status": 200, "body": job()}
        if body == receipt["body"]:
            return {"status": 200, "body": {"created": False, "job": job()}}
        return {"status": 409, "body": {"detail": "Conflict"}}

    result = replay.observe("verify", URL, call, receipt=receipt)
    assert result["passed"] and receipt == before
    assert result["http_request_attempts"] == 3
    assert calls[1][2:] == (receipt["body"], "fixed-key")
    assert calls[2][2] != receipt["body"] and calls[2][3] == "fixed-key"
    assert "restart" in result["scope"]


@pytest.mark.parametrize("change", ["url", "agent", "result_hash", "failed_prepare", "wrong_source", "model_calls"])
def test_invalid_receipt_rejected_before_any_http(change):
    receipt = prepare()
    if change == "url":
        receipt["url"] = "http://other.invalid"
    elif change == "agent":
        receipt["body"]["kind"] = "agent"
    elif change == "result_hash":
        receipt["result_sha256"] = "0" * 64
    elif change == "failed_prepare":
        receipt["passed"] = False
    elif change == "wrong_source":
        receipt["job"]["result"]["results"][0]["metadata"]["source_path"] = "wrong"
    else:
        receipt["job"]["result"]["usage"]["answer_transport_calls"] = 1
    result = replay.observe("verify", URL, lambda *_args: pytest.fail("invalid receipt made HTTP request"), receipt=receipt)
    assert result["passed"] is False


@pytest.mark.parametrize("change", ["id", "result", "attempt", "created", "status", "conflict"])
def test_replayed_job_or_conflict_mismatch_cannot_pass(change):
    receipt = prepare()

    def call(method, _path, body, _key, _timeout):
        if method == "GET":
            return {"status": 200, "body": job()}
        if body != receipt["body"]:
            return {"status": 202 if change == "conflict" else 409, "body": {}}
        value = job()
        if change == "id":
            value["id"] = "b" * 32
        elif change == "result":
            value["result"]["extra"] = "changed"
        elif change == "attempt":
            value["attempt"] = 2
        return {"status": 202 if change == "status" else 200,
            "body": {"created": change == "created", "job": value}}

    assert replay.observe("verify", URL, call, receipt=receipt)["passed"] is False


@pytest.mark.parametrize("failure", ["domain", "source", "missing_usage", "false_usage"])
def test_prepare_requires_real_domain_source_and_integer_zero_usage(failure):
    value = job()
    if failure == "domain":
        value["result"]["status"] = "failed"
    elif failure == "source":
        value["result"]["results"] = []
    elif failure == "missing_usage":
        value["result"].pop("usage")
    else:
        value["result"]["usage"]["answer_transport_calls"] = False
    def call(method, *_args):
        return {"status": 202, "body": {"created": True, "job": value}} if method == "POST" else {"status": 200, "body": value}
    assert replay.observe("prepare", URL, call)["passed"] is False


def test_poll_deadline_retains_admission_without_resubmission():
    now, calls = [0.0], []
    def call(method, *_args):
        calls.append(method)
        return {"status": 202, "body": {"created": True, "job": job()}} if method == "POST" else {"status": 200, "body": {"state": "queued"}}
    result = replay.observe("prepare", URL, call, timeout=1, clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n))
    assert not result["passed"] and result["error_type"] == "TimeoutError"
    assert result["job_id"] == "a" * 32 and calls.count("POST") == 1 and now[0] <= 1.0001


def test_cli_refuses_existing_receipt_before_http(tmp_path, monkeypatch):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("original", encoding="utf-8")
    monkeypatch.setenv("SERVICE_API_TOKEN", "TOKEN_CANARY")
    monkeypatch.setattr(replay, "http_transport", lambda *_args: pytest.fail("output collision reached HTTP"))
    with pytest.raises(FileExistsError):
        replay.main(["prepare", "--url", URL, "--receipt", str(receipt)])
    assert receipt.read_text() == "original"


def test_cli_writes_no_token_and_never_uses_implicit_url(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SERVICE_API_TOKEN", "TOKEN_CANARY")
    with pytest.raises(SystemExit):
        replay.main(["prepare", "--receipt", str(tmp_path / "unused.json")])
    def transport(url, token):
        assert url == URL and token == "TOKEN_CANARY"
        return lambda *_args: {"status": 503, "body": {"detail": "TOKEN_CANARY"}}
    monkeypatch.setattr(replay, "http_transport", transport)
    output = tmp_path / "receipt.json"
    assert replay.main(["prepare", "--url", URL, "--receipt", str(output)]) == 2
    assert "TOKEN_CANARY" not in output.read_text() + capsys.readouterr().out
    assert json.loads(output.read_text())["error_type"] == "CredentialInEvidence"


def test_http_does_not_follow_redirects():
    assert replay.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.invalid") is None


def test_expired_receipt_is_inconclusive_without_replaying_request():
    receipt = prepare()
    receipt["job"]["finished_at"] = time.time() - 60
    receipt["retention_seconds"] = 1
    calls = []

    def call(*args):
        calls.append(args)
        return {"status": 404, "body": {"detail": "Job not found"}}

    before = deepcopy(receipt)
    result = replay.observe("verify", URL, call, receipt=receipt)
    assert result["outcome"] == "inconclusive"
    assert result["reason"] == "retention_window_elapsed"
    assert result["http_request_attempts"] == 0
    assert not result["passed"] and not calls and receipt == before


def test_legacy_receipt_without_retention_is_not_a_persistence_failure():
    receipt = prepare()
    receipt.pop("retention_seconds", None)
    calls = []

    def call(*args):
        calls.append(args)
        return {"status": 404, "body": {"detail": "Job not found"}}

    result = replay.observe("verify", URL, call, receipt=receipt)
    assert result["outcome"] == "inconclusive"
    assert result["reason"] == "retention_policy_unrecorded"
    assert not result["passed"] and not calls


def test_missing_record_inside_known_window_still_fails():
    receipt, calls = prepare(), []

    def call(method, *_args):
        calls.append(method)
        return {"status": 404, "body": {"detail": "Job not found"}}

    result = replay.observe("verify", URL, call, receipt=receipt, utc_clock=lambda: FINISHED + 1)
    assert result["outcome"] == "failed" and not result["passed"]
    assert result["checks"][-1]["check"] == "retained_job_unchanged"
    assert result["http_request_attempts"] == 1
    assert calls == ["GET"]


def test_expiry_approaching_after_lookup_stops_before_post():
    receipt, calls, now = prepare(), [], [FINISHED + 1]

    def call(method, *_args):
        calls.append(method)
        now[0] = FINISHED + 86400 - 5
        return {"status": 200, "body": job()}

    result = replay.observe("verify", URL, call, receipt=receipt, utc_clock=lambda: now[0])
    assert result["outcome"] == "inconclusive" and result["reason"] == "retention_window_too_short"
    assert calls == ["GET"] and result["retained_lookup"]["body"] == job()


def test_expiry_during_lookup_is_not_reported_as_lost_persistence():
    receipt, now = prepare(), [FINISHED + 1]

    def call(method, *_args):
        assert method == "GET"
        now[0] = FINISHED + 86401
        return {"status": 404, "body": {"detail": "Job not found"}}

    result = replay.observe("verify", URL, call, receipt=receipt, utc_clock=lambda: now[0])
    assert result["outcome"] == "inconclusive" and result["reason"] == "retention_window_elapsed"
    assert result["retained_lookup"]["status"] == 404


@pytest.mark.parametrize("change,reason", [
    ("no_time", "terminal_timestamp_unavailable"),
    ("boolean_time", "terminal_timestamp_unavailable"),
    ("future_time", "clock_before_completion"),
])
def test_unreliable_terminal_timestamp_cannot_authorize_replay(change, reason):
    receipt = prepare()
    receipt["job"]["finished_at"] = {"no_time": None, "boolean_time": True, "future_time": FINISHED + 100}[change]
    result = replay.observe("verify", URL, lambda *_: pytest.fail("uncertain time made request"),
        receipt=receipt, utc_clock=lambda: FINISHED + 1)
    assert result["outcome"] == "inconclusive" and result["reason"] == reason


@pytest.mark.parametrize("retention", [True, 0, -1, 1.5, "86400"])
def test_invalid_retention_policy_never_submits(retention):
    result = replay.observe("prepare", URL, lambda *_: pytest.fail("invalid retention submitted"),
        retention_seconds=retention)
    assert result["outcome"] == "failed" and not result["passed"]


def test_verify_cannot_extend_retention_recorded_by_prepare():
    result = replay.observe("verify", URL, lambda *_: pytest.fail("changed retention made request"),
        receipt=prepare(), retention_seconds=172800)
    assert result["outcome"] == "failed"
    assert result["checks"][-1]["check"] == "declared_retention_matches_receipt"


def test_legacy_receipt_can_use_explicit_policy_without_being_modified():
    receipt, calls = prepare(), []
    receipt.pop("retention_seconds")
    before = deepcopy(receipt)

    def call(method, _path, body, _key, _timeout):
        calls.append(method)
        if method == "GET":
            return {"status": 200, "body": job()}
        return ({"status": 200, "body": {"created": False, "job": job()}}
            if body == receipt["body"] else {"status": 409, "body": {}})

    result = replay.observe("verify", URL, call, receipt=receipt, retention_seconds=86400,
        utc_clock=lambda: FINISHED + 1)
    assert result["passed"] and calls == ["GET", "POST", "POST"] and receipt == before
    assert result["retention_window"]["actual_expiry_observed"] is False


def test_cli_expired_sample_returns_distinct_inconclusive_exit_code(tmp_path, monkeypatch, capsys):
    receipt = prepare()
    receipt["job"]["finished_at"] = time.time() - 86401
    source, output = tmp_path / "original.json", tmp_path / "observed.json"
    source.write_text(json.dumps(receipt), encoding="utf-8")
    before = source.read_bytes()
    monkeypatch.setenv("SERVICE_API_TOKEN", "TOKEN_CANARY")
    monkeypatch.setattr(replay, "http_transport",
        lambda *_: lambda *_args: pytest.fail("expired receipt reached HTTP"))
    assert replay.main(["verify", "--url", URL, "--receipt", str(source), "--output", str(output)]) == 3
    result = json.loads(output.read_bytes())
    assert result["outcome"] == "inconclusive" and result["reason"] == "retention_window_elapsed"
    assert json.loads(capsys.readouterr().out)["outcome"] == "inconclusive"
    assert source.read_bytes() == before and "TOKEN_CANARY" not in output.read_text()


@pytest.mark.parametrize("phase", ["same_request", "conflict"])
@pytest.mark.parametrize("new_job", [False, True])
def test_post_response_after_retention_window_preserves_observation_without_verdict(phase, new_job):
    receipt, now, calls = prepare(), [FINISHED + 1], []

    def call(method, _path, body, _key, _timeout):
        calls.append(method)
        if method == "GET":
            return {"status": 200, "body": job()}
        same = body == receipt["body"]
        if same == (phase == "same_request"):
            now[0] = FINISHED + 86401
            if new_job:
                return {"status": 202, "body": {"created": True, "job": {"id": "b" * 32}}}
        return ({"status": 200, "body": {"created": False, "job": job()}} if same
            else {"status": 409, "body": {"detail": "Conflict"}})

    result = replay.observe("verify", URL, call, receipt=receipt, utc_clock=lambda: now[0])
    assert result["outcome"] == "inconclusive" and result["reason"] == "retention_window_elapsed"
    assert result["http_request_attempts"] == len(calls) == (2 if phase == "same_request" else 3)
    assert result["last_http_response"]["method"] == "POST"
    if new_job:
        assert result["last_http_response"]["response"]["body"]["created"] is True
    assert result["retention_window"]["observed_at"] == now[0]


@pytest.mark.parametrize("elapsed", [11.0, 91.0], ids=["request-budget", "total-budget"])
def test_response_after_total_deadline_cannot_pass(elapsed):
    receipt, now = prepare(), [0.0]

    def call(method, *_args):
        assert method == "GET"
        now[0] = elapsed
        return {"status": 200, "body": job()}

    result = replay.observe("verify", URL, call, receipt=receipt, clock=lambda: now[0],
        utc_clock=lambda: FINISHED + 1)
    assert result["outcome"] == "failed" and result["error_type"] == "TimeoutError"
    assert result["last_http_response"]["response"]["status"] == 200
    assert result["http_request_attempts"] == 1
