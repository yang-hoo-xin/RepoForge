"""Independent ASGI API instances sharing real Redis; not an HTTP TCP load test.

为什么要有它：单实例替身不能证明多实例共享限流额度及幂等状态，真实Redis必须参与入站验收。
"""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from uuid import uuid4

import pytest

redis = pytest.importorskip("redis", reason="Real Redis API integration requires redis")
pytest.importorskip("fastapi", reason="API integration requires service dependencies")
from fastapi.testclient import TestClient

from src.service_layer.api import create_app
from src.service_layer.config import ServiceConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def api_pair(tmp_path, record_property):
    url = os.environ.get("SERVICE_TEST_REDIS_URL")
    if not url:
        pytest.skip("Set SERVICE_TEST_REDIS_URL to an explicitly selected real Redis")
    settings = tmp_path / "settings.yaml"
    settings.write_text("query: {}\n", encoding="utf-8")
    config = ServiceConfig(
        redis_url=url,
        namespace="service-api-test-" + uuid4().hex,
        settings_path=settings,
        work_root=tmp_path / "jobs",
        api_token=uuid4().hex,
        requests_per_minute=3,
        max_pending=10,
    )
    apps = [create_app(config), create_app(config)]
    stores = [app.state.job_store for app in apps]
    assert stores[0] is not stores[1]
    assert stores[0].client.connection_pool is not stores[1].client.connection_pool
    assert stores[0].key_prefix == stores[1].key_prefix
    record_property("transport_scope", "ASGI TestClient plus real Redis TCP; no API TCP/load claim")
    record_property("namespace", config.namespace)
    try:
        try:
            for store in stores:
                assert store.client.ping()
        except redis.RedisError as error:
            pytest.fail(f"Configured Redis unavailable: {type(error).__name__}; no fake fallback", pytrace=False)
        with ExitStack() as stack:
            clients = [stack.enter_context(TestClient(app)) for app in apps]
            yield clients, stores, {"Authorization": "Bearer " + config.api_token}
    finally:
        # 为什么要有它：共享Redis中仅删除本fixture随机namespace的键，禁止flush或清理其他任务。
        try:
            owner = stores[0]
            keys = list(owner.client.scan_iter(match=owner.key_prefix + "*"))
            if keys:
                assert all(key.startswith(owner.key_prefix) for key in keys)
                owner.client.delete(*keys)
            assert not list(owner.client.scan_iter(match=owner.key_prefix + "*"))
            record_property("namespace_cleanup", "verified_empty")
        finally:
            for store in stores:
                store.client.close()


def test_instances_share_rate_budget_and_idempotent_admission(api_pair, record_property):
    clients, stores, auth = api_pair
    body = {"kind": "query", "payload": {"query": "Explain JSON arrays"}}

    def submit(instance, key):
        return clients[instance].post("/v1/jobs", headers={**auth, "Idempotency-Key": key}, json=body)

    first = submit(0, "shared-request")
    assert first.status_code == 202, first.text
    replay = submit(1, "shared-request")
    assert replay.status_code == 200, replay.text
    second = submit(1, "second-request")
    assert second.status_code == 202, second.text
    first_id, second_id = first.json()["job"]["id"], second.json()["job"]["id"]
    assert first_id != second_id
    assert first.json()["created"] is True and second.json()["created"] is True
    assert replay.json()["created"] is False
    assert replay.json()["job"]["id"] == first_id
    assert replay.headers["location"] == first.headers["location"] == "/v1/jobs/" + first_id

    # 为什么要有它：两实例各自还未达到3次，合计已满；必须都拒绝新键，排除仅进程内限流。
    rejected = [submit(0, "new-over-limit-a"), submit(1, "new-over-limit-b")]
    for response in rejected:
        assert response.status_code == 429, response.text
        assert response.json()["detail"] == "Rate limit exceeded"
        assert 1 <= int(response.headers["retry-after"]) <= 60

    store = stores[0]
    queued = set(store.client.zrange(store.key_prefix + "ready", 0, -1))
    assert queued == {first_id, second_id}
    assert all(store.get(job_id)["state"] == "queued" for job_id in queued)
    assert len(list(store.client.scan_iter(match=store.key_prefix + "job:*"))) == 2
    assert len(list(store.client.scan_iter(match=store.key_prefix + "idem:*"))) == 2
    for instance_store in stores:
        snapshot = instance_store.metrics()
        assert snapshot["queued"] == snapshot["pending"] == 2
        assert snapshot["running"] == 0
        assert snapshot["counters"]["submitted"] == 2
        assert snapshot["counters"]["duplicate_submits"] == 1
    metrics = clients[1].get("/metrics")
    assert metrics.status_code == 200
    assert "service_jobs_submitted_total 2\n" in metrics.text
    assert "service_jobs_idempotency_hits_total 1\n" in metrics.text
    record_property("submission_statuses", "202,200,202,429,429")
    record_property("unique_jobs_queued", 2)


def test_compact_and_legacy_completed_results_keep_normal_api_shape(api_pair, record_property):
    clients, stores, auth = api_pair
    body = {"kind": "query", "payload": {"query": "controlled result representation"}}
    headers = {**auth, "Idempotency-Key": "completed-result"}
    submitted = clients[0].post("/v1/jobs", headers=headers, json=body)
    assert submitted.status_code == 202
    job_id = submitted.json()["job"]["id"]
    store = stores[0]
    claimed = store.claim("controlled-storage-check", 30)
    assert claimed["id"] == job_id
    expected = {"status": "completed", "answer": "保留结果", "data": [None, False, 0, [], {}]}
    assert store.finish(job_id, claimed["lease_token"], "completed", result=expected)
    key = store.key_prefix + "job:" + job_id
    compact = json.loads(store.client.get(key))
    assert compact["attempts_history"][-1]["_result_from_job"] == {"version": 1, "attempt": 1}
    first = clients[0].get("/v1/jobs/" + job_id, headers=auth)
    second = clients[1].get("/v1/jobs/" + job_id, headers=auth)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["state"] == "completed" and first.json()["result"] == expected
    assert "attempts_history" not in first.json() and "_result_from_job" not in first.text
    repeated = clients[1].post("/v1/jobs", headers=headers, json=body)
    assert repeated.status_code == 200 and repeated.json()["created"] is False
    assert repeated.json()["job"] == first.json()

    # Scope: only this synthetic fixture's private namespace, preserving its TTL.
    # 为什么要有它：旧编码必须经实际Redis和正常API读取，不能只用手写decoder的返回值假装兼容。
    last = compact["attempts_history"][-1]
    last.pop("_result_from_job")
    last["result_json"] = compact["result_json"]
    old_bytes = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    store.client.set(key, old_bytes, keepttl=True)
    assert 0 < store.client.pttl(key) <= 86400000
    for client in clients:
        response = client.get("/v1/jobs/" + job_id, headers=auth)
        assert response.status_code == 200 and response.json() == first.json()
    assert store.get(job_id)["attempts_history"][-1]["result"] == expected
    assert store.metrics()["pending"] == 0
    record_property("representation_scope", "new writer and legacy stored encoding; current API readers only")
