"""Observe HTTP state before/after an externally managed restart; never restart it."""
import argparse
import hashlib
import json
import math
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

SOURCE = "src/ingestion/storage/bm25_indexer.py"
TERMINAL = {"completed", "failed", "timed_out", "indeterminate", "cancelled"}


class ObservationInconclusiveError(Exception):
    """The retained sample cannot establish persistence in the selected window."""


def request_body():
    return {"kind": "query", "payload": {"query": "BM25Indexer inverted postings IDF",
        "collection": "service_load", "top_k": 5, "answer": False, "rewrite": False, "rerank": False},
        "timeout_seconds": 60.0, "max_attempts": 1}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def url_identity(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("Explicit HTTP URL without credentials/query/fragment is required")
    return value.rstrip("/")


def contains_secret(value, token):
    if isinstance(value, str):
        return token in value
    if isinstance(value, dict):
        return any(contains_secret(key, token) or contains_secret(item, token) for key, item in value.items())
    return isinstance(value, list) and any(contains_secret(item, token) for item in value)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Scope: explicit HTTP target only. 为什么要有它：验证脚本不能随重定向把Bearer凭据发给另一个地址。
        return None


def http_transport(url, token):
    opener = build_opener(NoRedirect())

    def call(method, path, body, key, timeout):
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + token}
        if key is not None:
            headers["Idempotency-Key"] = key
        request = Request(url + path, method=method, headers=headers,
            data=None if body is None else json.dumps(body).encode("utf-8"))
        started = time.monotonic()
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read().decode("utf-8")
            if token in raw:
                raise ValueError("Response contains credential")
            value = json.loads(raw)
            if contains_secret(value, token):
                raise ValueError("Response contains credential")
            return {"status": response.status, "body": value, "elapsed_seconds": time.monotonic() - started}

    return call


def completed(job):
    result = job.get("result") or {}
    usage = result.get("usage") or {}
    return (job.get("state") == "completed" and type(job.get("attempt")) is int and job["attempt"] == 1
        and result.get("status") == "completed"
        and any(row.get("metadata", {}).get("source_path") == SOURCE for row in result.get("results", []))
        and all(type(usage.get(name)) is int and usage[name] == 0
                for name in ("answer_transport_calls", "rewrite_transport_calls")))


def observe(mode, url, call, *, receipt=None, key=None, timeout=90.0, retention_seconds=None,
            clock=time.monotonic, utc_clock=time.time, sleep=time.sleep):
    report = {"mode": mode, "url": url, "passed": False, "outcome": "failed", "checks": [],
        "http_request_attempts": 0,
        "scope": "HTTP persistence and replay only; restart ownership/evidence is external"}
    started = clock()
    window = None

    def check(name, passed, response=None):
        report["checks"].append({"check": name, "passed": bool(passed), "response": response})
        if not passed:
            raise ValueError(name)

    def check_window(request_seconds=0.0):
        if window is None:
            return
        now = utc_clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError("Invalid wall clock")
        report["retention_window"]["observed_at"] = now
        if now < window["finished_at"]:
            raise ObservationInconclusiveError("clock_before_completion")
        if now >= window["expires_at"]:
            raise ObservationInconclusiveError("retention_window_elapsed")
        if now + request_seconds >= window["expires_at"]:
            raise ObservationInconclusiveError("retention_window_too_short")

    def request(method, path, body=None, request_key=None):
        remaining = timeout - (clock() - started)
        if remaining <= 0:
            raise TimeoutError("Observation deadline exhausted")
        request_seconds = min(10.0, remaining)
        check_window(request_seconds)
        report["http_request_attempts"] += 1
        sent_at = clock()
        response = call(method, path, body, request_key, request_seconds)
        # Scope: preserve an executed request before deciding whether its observation is still admissible.
        # 为什么要有它：慢响应可能跨过TTL或观察截止；不能误报恢复通过，也不能把已发生的POST记成未执行。
        report["last_http_response"] = {"method": method, "path": path, "response": response}
        if mode == "verify" and method == "GET":
            report["retained_lookup"] = response
        check_window()
        received_at = clock()
        if received_at - started >= timeout or received_at - sent_at > request_seconds:
            raise TimeoutError("Response arrived outside the observation budget")
        return response

    try:
        if retention_seconds is not None:
            check("valid_declared_retention", type(retention_seconds) is int and retention_seconds > 0)
        if mode == "prepare":
            key = key or "restart-replay-" + uuid4().hex
            check("valid_idempotency_key", isinstance(key, str) and re.fullmatch(r"[\x21-\x7e]{1,128}", key) is not None)
            body = request_body()
            report.update(body=body, idempotency_key=key, expected_source=SOURCE)
            admitted = request("POST", "/v1/jobs", body, key)
            job = admitted["body"].get("job") or {}
            check("new_job_admitted", admitted["status"] == 202 and admitted["body"].get("created") is True
                and re.fullmatch(r"[a-f0-9]{32}", str(job.get("id"))) is not None, admitted)
            report["job_id"] = job["id"]
            polls = 0
            while True:
                row = request("GET", "/v1/jobs/" + report["job_id"])
                polls += 1
                if row["status"] != 200 or row["body"].get("state") in TERMINAL:
                    break
                remaining = timeout - (clock() - started)
                if remaining <= 0:
                    raise TimeoutError("Observation deadline exhausted")
                sleep(min(0.1, remaining))
            report["poll_count"] = polls
            check("actual_query_completion_source_and_zero_model_calls", row["status"] == 200
                and row["body"].get("id") == report["job_id"] and completed(row["body"]), row)
            report["job"] = row["body"]
            report["result_sha256"] = digest(row["body"]["result"])
            if retention_seconds is not None:
                report["retention_seconds"] = retention_seconds
                report["retention_basis"] = "caller_declared_deployment_policy_not_server_expiry_observation"
        elif mode == "verify":
            # Scope: retained fixed query only. 为什么要有它：receipt不能扩大为Agent或任意提交；错误地址/失败准备也不能获得重放资格。
            valid = (isinstance(receipt, dict) and receipt.get("mode") == "prepare" and receipt.get("passed") is True
                and receipt.get("url") == url and digest(receipt.get("body")) == digest(request_body())
                and receipt.get("expected_source") == SOURCE
                and isinstance(receipt.get("idempotency_key"), str)
                and re.fullmatch(r"[\x21-\x7e]{1,128}", receipt["idempotency_key"]) is not None
                and re.fullmatch(r"[a-f0-9]{32}", str(receipt.get("job_id"))) is not None
                and isinstance(receipt.get("job"), dict) and receipt["job"].get("id") == receipt["job_id"]
                and completed(receipt["job"]) and receipt.get("result_sha256") == digest(receipt["job"]["result"]))
            check("prepare_receipt_valid_for_exact_target", valid)
            report.update(job_id=receipt["job_id"], prepare_sha256=digest(receipt))
            recorded_retention = receipt.get("retention_seconds")
            if recorded_retention is not None:
                check("recorded_retention_valid", type(recorded_retention) is int and recorded_retention > 0)
                check("declared_retention_matches_receipt", retention_seconds is None or retention_seconds == recorded_retention)
            selected_retention = recorded_retention if recorded_retention is not None else retention_seconds
            if selected_retention is None:
                raise ObservationInconclusiveError("retention_policy_unrecorded")
            finished = receipt["job"].get("finished_at")
            if type(finished) not in (int, float) or not math.isfinite(finished) or finished <= 0:
                raise ObservationInconclusiveError("terminal_timestamp_unavailable")
            deadline = finished + selected_retention
            check("finite_retention_deadline", math.isfinite(deadline))
            window = {"finished_at": finished, "retention_seconds": selected_retention, "expires_at": deadline}
            # Scope: a declared test window, not proof Redis deleted or retained a key.
            # 为什么要有它：过期样本不能证明重启丢数据，也不能重放成新任务；每个请求前重验窗口但不延长TTL或改旧receipt。
            report["retention_window"] = {**window,
                "basis": "caller_declared_deployment_policy_and_retained_server_finished_at",
                "actual_expiry_observed": False}
            check_window()

            def same(job):
                return (isinstance(job, dict) and completed(job) and job.get("id") == receipt["job_id"]
                    and job.get("attempt") == receipt["job"]["attempt"]
                    and digest(job.get("result")) == receipt["result_sha256"])

            row = request("GET", "/v1/jobs/" + receipt["job_id"])
            check("retained_job_unchanged", row["status"] == 200 and same(row["body"]), row)
            row = request("POST", "/v1/jobs", receipt["body"], receipt["idempotency_key"])
            check("same_request_not_executed_again", row["status"] == 200
                and row["body"].get("created") is False and same(row["body"].get("job")), row)
            changed = deepcopy(receipt["body"])
            changed["payload"]["query"] += " changed-body-conflict"
            row = request("POST", "/v1/jobs", changed, receipt["idempotency_key"])
            check("changed_request_conflicts", row["status"] == 409, row)
        else:
            raise ValueError("Unknown mode")
        report["passed"] = True
        report["outcome"] = "passed"
    except ObservationInconclusiveError as error:
        report.update(outcome="inconclusive", reason=str(error))
    except (Exception, KeyboardInterrupt) as error:
        report["error_type"] = type(error).__name__
    report["elapsed_seconds"] = clock() - started
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retention-seconds", type=int,
        help="Confirmed deployment retention policy; stored by prepare or supplied for a legacy receipt.")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.timeout) or not 1 <= args.timeout <= 180
            or args.mode == "verify" and (args.output is None or args.key is not None)
            or args.mode == "prepare" and args.output is not None
            or args.retention_seconds is not None and args.retention_seconds <= 0):
        parser.error("Use a 1..180s budget, positive retention, --output only for verify, and --key only for prepare")
    url = url_identity(args.url)
    token = os.environ["SERVICE_API_TOKEN"]
    if not token or "\r" in token or "\n" in token:
        raise ValueError("A nonempty Bearer token is required")
    receipt = None if args.mode == "prepare" else json.loads(args.receipt.read_bytes())
    destination = args.receipt if args.mode == "prepare" else args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Scope: new evidence files only. 为什么要有它：在请求前独占输出文件，不能覆盖旧receipt或验证结果后才发现无法保存。
    with destination.open("x", encoding="utf-8") as stream:
        report = observe(args.mode, url, http_transport(url, token), receipt=receipt, key=args.key, timeout=args.timeout,
            retention_seconds=args.retention_seconds)
        if contains_secret(report, token):
            report = {"mode": args.mode, "passed": False, "outcome": "failed", "error_type": "CredentialInEvidence"}
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"passed": report["passed"], "outcome": report["outcome"], "mode": args.mode}))
    return 0 if report["passed"] else 3 if report["outcome"] == "inconclusive" else 2


if __name__ == "__main__":
    raise SystemExit(main())
