"""Independent read-only browser security checks using disposable synthetic records.

为什么要有它：浏览授权记录不能扩大目录权限、泄露凭据或修改原始运行；测试不改真实历史目录。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from src.observability.dashboard.services.trace_service import TraceService


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def make_run(root: Path, name: str = "run-alpha") -> Path:
    run = root / name
    write_json(run / "run_manifest.json", {
        "run_id": name, "request_id": "request-" + name,
        "started_at": "2026-10-19T00:00:00+00:00", "status": "blocked",
        "terminal_stage": "blocked", "execution_observations": {},
    })
    write_json(run / "run_result.json", {"run_id": name, "current_stage": "blocked"})
    return run


@pytest.fixture
def browser(tmp_path):
    root = tmp_path / "allowed"
    run = make_run(root)
    service = TraceService(tmp_path / "not-used.jsonl")
    return service, root, run


def run_entry(service, root, run):
    cursor = 0
    while True:
        result = service.discover_runs(allowed_roots=[str(root)], cursor=cursor, limit=2)
        assert result["read_only"] is True
        for row in result["runs"]:
            if Path(row["run_root"]).resolve() == run.resolve():
                return row
        if result["next_cursor"] is None:
            raise AssertionError(f"Synthetic run was not discovered: {result}")
        assert result["next_cursor"] > cursor
        cursor = result["next_cursor"]


def record_entry(service, root, run, relative):
    selected = run_entry(service, root, run)
    cursor = 0
    while True:
        page = service.inspect_run(allowed_roots=[str(root)], run_id=selected["run_id"],
            record_cursor=cursor, record_limit=2)
        assert page["read_only"] is True
        for row in page["records"]:
            if row["relative_path"].replace("\\", "/") == relative:
                return row
        if page["next_record_cursor"] is None:
            raise AssertionError(f"Synthetic record was not enumerated: {relative}")
        assert page["next_record_cursor"] > cursor
        cursor = page["next_record_cursor"]


def read_all(service, roots, record_id, *, limit=65536):
    offset, chunks, responses = 0, [], []
    expected_total = None
    while True:
        result = service.read_record(allowed_roots=[str(root) for root in roots],
            record_id=record_id, offset=offset, limit=limit)
        assert result["status"] in {"ok", "corrupt"}, result
        assert result["read_only"] is True
        span = result["range"]
        assert span["offset"] == offset and span["length"] == len(result["text"])
        if expected_total is None:
            expected_total = span["total_chars"]
        assert span["total_chars"] == expected_total
        chunks.append(result["text"])
        responses.append(result)
        if span["complete"]:
            assert span["next_offset"] is None
            break
        assert span["next_offset"] == offset + span["length"] > offset
        offset = span["next_offset"]
        assert len(responses) < 10000, "Pagination did not terminate"
    value = "".join(chunks)
    assert len(value) == expected_total
    return value, responses


def assert_denied(result, statuses=("forbidden",)):
    assert result["status"] in statuses, result
    assert result["text"] == ""
    assert result["json_value"] is None
    assert result["read_only"] is True


def inventory(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


@pytest.mark.parametrize("locator", [
    "../outside.json", "..\\outside.json", "/etc/passwd", "C:\\Windows\\win.ini",
    "\\\\server\\share\\secret.json", "record.json:private", "record.json::$DATA",
])
def test_arbitrary_path_and_ads_are_not_record_locators(browser, locator):
    service, root, _run = browser
    assert_denied(service.read_record(allowed_roots=[str(root)], record_id=locator))


def test_opaque_record_id_cannot_be_reused_under_another_or_prefix_root(browser, tmp_path):
    service, root, run = browser
    write_json(run / "inside.json", {"value": "authorized"})
    record = record_entry(service, root, run, "inside.json")
    sibling = tmp_path / "allowed-extra"
    sibling_run = make_run(sibling, "run-other")
    secret = "PRIVATE_OTHER_ROOT_83da"
    write_json(sibling_run / "outside.json", {"value": secret})
    outside = record_entry(service, sibling, sibling_run, "outside.json")
    assert_denied(service.read_record(allowed_roots=[str(root)], record_id=outside["record_id"]))
    assert_denied(service.read_record(allowed_roots=[str(sibling)], record_id=record["record_id"]))
    assert_denied(service.read_record(allowed_roots=[], record_id=record["record_id"]))


def test_hash_bound_reference_does_not_grant_access_to_outside_content(browser, tmp_path):
    service, root, run = browser
    secret = "DO_NOT_READ_EXTERNAL_REFERENCE_d867"
    outside = write_json(tmp_path / "private" / "secret.json", {"secret": secret})
    write_json(run / "linked.json", {"detail_ref": {"path": str(outside),
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}})
    record = record_entry(service, root, run, "linked.json")
    result = service.read_record(allowed_roots=[str(root)], record_id=record["record_id"])
    assert result["status"] == "ok"
    assert secret not in json.dumps(result, ensure_ascii=False)
    assert result["links"]
    assert all(row["target_record_id"] is None for row in result["links"])
    assert all(row["status"] == "forbidden" for row in result["links"])


def test_missing_and_hash_changed_records_are_explicit_without_content(browser):
    service, root, run = browser
    path = write_json(run / "version.json", {"version": 1})
    selected = record_entry(service, root, run, "version.json")
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(path, {"version": 2, "password": "NEW_PRIVATE_VALUE_264a"})
    changed = service.read_record(allowed_roots=[str(root)], record_id=selected["record_id"],
        expected_sha256=original_sha)
    assert_denied(changed, ("hash_mismatch", "changed"))
    path.unlink()
    assert_denied(service.read_record(allowed_roots=[str(root)], record_id=selected["record_id"]), ("missing",))


@pytest.mark.parametrize("include_raw", [False, True])
def test_nested_json_credentials_and_bearer_across_pages_remain_redacted(browser, include_raw):
    _service, root, run = browser
    service = TraceService(root / "not-used.jsonl", include_raw=include_raw)
    secrets = ["API_PRIVATE_40f8", "NESTED_PRIVATE_764c", "BearerPrivate44a9" * 8]
    payload = {"api_key": secrets[0], "payload": json.dumps({"access_token": secrets[1],
        "usage": {"input_tokens": 11, "output_tokens": 7, "cached_input_tokens": 3}}, ensure_ascii=False),
        "log": "prefix " + "p" * 61 + " Authorization: Bearer " + secrets[2] + "\nlast line",
        "usage": {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168, "cached_input_tokens": 80}}
    path = write_json(run / "sensitive.json", payload)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    selected = record_entry(service, root, run, "sensitive.json")
    text, pages = read_all(service, [root], selected["record_id"], limit=64)
    exposed = json.dumps(pages, ensure_ascii=False) + text
    for secret in [*secrets, "BearerPrivate44a9"]:
        assert secret not in exposed
    restored = json.loads(text)
    assert restored["usage"] == payload["usage"]
    assert json.loads(restored["payload"])["usage"] == {"input_tokens": 11, "output_tokens": 7, "cached_input_tokens": 3}
    assert any(row["redaction"]["applied"] for row in pages)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("field", ["SERVICE_API_TOKEN", "api_token", "credentials", "secrets", "client_secret", "refresh_token"])
def test_application_credential_keys_are_protected_in_scalar_and_container_forms(browser, field):
    service, root, run = browser
    secret = "PROJECT_CREDENTIAL_VALUE_539e"
    write_json(run / "credential.json", {field: {"opaque": [secret]}, "output_tokens": 12})
    selected = record_entry(service, root, run, "credential.json")
    text, pages = read_all(service, [root], selected["record_id"], limit=32)
    assert secret not in text + json.dumps(pages, ensure_ascii=False)
    assert json.loads(text)["output_tokens"] == 12
    assert any(row["redaction"]["applied"] for row in pages)


def test_discovery_summary_and_timeline_do_not_bypass_default_redaction(browser):
    service, root, run = browser
    secret = "METADATA_PRIVATE_BEARER_e93a"
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "failed: Authorization: Bearer " + secret
    write_json(run / "run_manifest.json", manifest)
    write_json(run / "event.json", {"name": "Authorization: Bearer " + secret, "status": "blocked"})
    discovered = service.discover_runs(allowed_roots=[str(root)])
    assert secret not in json.dumps(discovered, ensure_ascii=False)
    selected = next(row for row in discovered["runs"] if Path(row["run_root"]).resolve() == run.resolve())
    inspection = service.inspect_run(allowed_roots=[str(root)], run_id=selected["run_id"])
    assert secret not in json.dumps(inspection, ensure_ascii=False)


def test_schema_field_descriptions_are_not_confused_with_credential_values(browser):
    service, root, run = browser
    schema = {"type": "object", "properties": {
        "api_key": {"type": "string", "description": "The configured key is provided by the caller."},
        "token": {"type": "integer", "description": "An explicitly documented numeric field."}}}
    secret = "ACTUAL_SCHEMA_NEIGHBOR_CREDENTIAL_91a0"
    write_json(run / "schema.json", {"response_format": {"schema": schema},
        "configuration": {"api_key": secret}, "usage": {"input_tokens": 14}})
    record = record_entry(service, root, run, "schema.json")
    text, _pages = read_all(service, [root], record["record_id"])
    safe = json.loads(text)
    assert safe["response_format"]["schema"] == schema
    assert secret not in text
    assert safe["usage"]["input_tokens"] == 14


def test_secret_schema_default_is_a_value_not_just_a_field_description(browser):
    service, root, run = browser
    secret = "SCHEMA_DEFAULT_PRIVATE_VALUE_17fb"
    write_json(run / "schema-default.json", {"parameters": {"type": "object", "properties": {
        "api_key": {"type": "string", "description": "Provider credential", "default": secret}}}})
    record = record_entry(service, root, run, "schema-default.json")
    text, pages = read_all(service, [root], record["record_id"])
    assert secret not in text + json.dumps(pages, ensure_ascii=False)
    definition = json.loads(text)["parameters"]["properties"]["api_key"]
    assert definition["type"] == "string" and definition["description"] == "Provider credential"


def test_linked_plain_log_masks_actual_service_environment_key(browser):
    service, root, run = browser
    secret = "ENVIRONMENT_PRIVATE_VALUE_d791"
    log = run / "stderr.log"
    log.write_text("before\nSERVICE_API_TOKEN=" + secret + "\nafter\n", encoding="utf-8")
    write_json(run / "log-reference.json", {"stderr_ref": {"path": str(log),
        "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}})
    selected = record_entry(service, root, run, "log-reference.json")
    reference = service.read_record(allowed_roots=[str(root)], record_id=selected["record_id"])
    link = next(row for row in reference["links"] if row["status"] == "ok")
    text, pages = read_all(service, [root], link["target_record_id"], limit=32)
    assert "before" in text and "after" in text
    assert secret not in text + json.dumps(pages, ensure_ascii=False)
    assert secret in log.read_text(encoding="utf-8")


def test_configured_secret_literal_is_hidden_without_a_field_or_bearer_label(browser, monkeypatch):
    service, root, run = browser
    secret = "RRFIX_LITERAL_VALUE_c03ab677e7f9514e"
    monkeypatch.setenv("REPOFORGE_BROWSER_TEST_API_KEY", secret)
    path = write_json(run / "literal.json", {"content": "prefix " + secret + " suffix", "input_tokens": 9})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    selected = record_entry(service, root, run, "literal.json")
    text, pages = read_all(service, [root], selected["record_id"], limit=11)
    assert secret not in text + json.dumps(pages, ensure_ascii=False)
    assert json.loads(text)["input_tokens"] == 9
    assert any(row["redaction"]["applied"] for row in pages)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("filename", [".env", ".env.local", "auth.key", "id_rsa"])
def test_explicit_credential_files_do_not_become_unprotected_utf8_logs(browser, filename):
    service, root, run = browser
    secret = "SENSITIVE_FILE_LITERAL_9c247089"
    private = run / filename
    private.write_text(secret + "\n", encoding="utf-8")
    write_json(run / "private-reference.json", {"detail_ref": {"path": str(private),
        "sha256": hashlib.sha256(private.read_bytes()).hexdigest()}})
    selected = record_entry(service, root, run, "private-reference.json")
    reference = service.read_record(allowed_roots=[str(root)], record_id=selected["record_id"])
    assert secret not in json.dumps(reference, ensure_ascii=False)
    assert reference["links"]
    link = reference["links"][0]
    if link.get("target_record_id") is None:
        assert link["status"] == "forbidden"
    else:
        result = service.read_record(allowed_roots=[str(root)], record_id=link["target_record_id"])
        assert secret not in json.dumps(result, ensure_ascii=False)
        assert result["status"] == "forbidden" or result["redaction"]["applied"]
    assert secret in private.read_text(encoding="utf-8")


def test_private_key_block_inside_ordinary_record_is_protected(browser):
    service, root, run = browser
    secret = "PRIVATE_KEY_BODY_CANARY_736add"
    pem = "-----BEGIN PRIVATE KEY-----\n" + secret + "\n-----END PRIVATE KEY-----"
    write_json(run / "diagnostic.json", {"content": "captured stderr:\n" + pem + "\nafter", "output_tokens": 2})
    selected = record_entry(service, root, run, "diagnostic.json")
    text, pages = read_all(service, [root], selected["record_id"], limit=23)
    assert secret not in text + json.dumps(pages, ensure_ascii=False)
    assert json.loads(text)["output_tokens"] == 2
    assert any(row["redaction"]["applied"] for row in pages)


def test_corrupt_json_has_safe_complete_text_and_visible_parse_issue(browser):
    service, root, run = browser
    secret = "CORRUPT_PRIVATE_CREDENTIAL_b21a"
    path = run / "broken.json"
    raw = '{"authorization": "Bearer ' + secret + '", "unfinished": '
    path.write_text(raw, encoding="utf-8")
    selected = record_entry(service, root, run, "broken.json")
    text, pages = read_all(service, [root], selected["record_id"], limit=32)
    assert all(row["status"] == "corrupt" and row["json_value"] is None for row in pages)
    assert any(row["issues"] for row in pages)
    assert secret not in json.dumps(pages, ensure_ascii=False)
    assert "unfinished" in text
    assert path.read_text(encoding="utf-8") == raw


def test_unknown_large_json_reaches_last_value_and_browsing_changes_no_files(browser):
    service, root, run = browser
    payload = {"unknown_schema": "a-new-producer.v928", "rows": [
        {"index": index, "text": "large-record-" + "z" * 245} for index in range(8192)],
        "tail": {"marker": "EXACT_END_OF_LARGE_JSON_43b2", "input_tokens": 17}}
    path = write_json(run / "new-producer.json", payload)
    assert path.stat().st_size > 2 * 1024 * 1024
    before = inventory(root)
    selected = record_entry(service, root, run, "new-producer.json")
    text, pages = read_all(service, [root], selected["record_id"])
    assert len(pages) > 2 and json.loads(text) == payload
    assert pages[-1]["range"]["complete"] is True
    assert "EXACT_END_OF_LARGE_JSON_43b2" in pages[-1]["text"]
    assert inventory(root) == before


def test_valid_file_replaced_by_symlink_cannot_escape_root(browser, tmp_path):
    service, root, run = browser
    path = write_json(run / "swap.json", {"normal": True})
    outside = write_json(tmp_path / "outside.json", {"secret": "SYMLINK_PRIVATE_310e"})
    record = record_entry(service, root, run, "swap.json")
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Host cannot create file symlink: {type(error).__name__}")
    try:
        assert_denied(service.read_record(allowed_roots=[str(root)], record_id=record["record_id"]))
    finally:
        path.unlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary")
def test_junction_replacement_and_junction_root_are_rejected(browser, tmp_path):
    service, root, run = browser
    folder = run / "records"
    write_json(folder / "entry.json", {"normal": True})
    record = record_entry(service, root, run, "records/entry.json")
    preserved = run / "records-preserved"
    folder.rename(preserved)
    outside = tmp_path / "outside"
    make_run(outside, "outside-run")
    write_json(outside / "entry.json", {"secret": "JUNCTION_PRIVATE_4ea1"})
    created = subprocess.run(["cmd", "/c", "mklink", "/J", str(folder), str(outside)],
        capture_output=True, timeout=10)
    if created.returncode:
        pytest.skip("Host cannot create a directory junction")
    try:
        assert folder.resolve() == outside.resolve()
        assert_denied(service.read_record(allowed_roots=[str(root)], record_id=record["record_id"]))
        discovered = service.discover_runs(allowed_roots=[str(folder)])
        assert discovered["runs"] == [] and discovered["issues"]
    finally:
        # Only remove this exact synthetic junction, never recursively delete its target.
        # 为什么要有它：越界测试的清理不能误删所指向目录，原合成输入仍可由pytest统一回收。
        assert folder.resolve() == outside.resolve()
        folder.rmdir()
