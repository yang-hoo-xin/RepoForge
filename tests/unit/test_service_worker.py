"""Worker/backend contract tests; fake Redis/provider boundaries, no model or game."""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.query_engine.query_service import QueryResponse, QueryService
from src.core.settings import QuerySettings
from src.service_layer import backends, execute, worker


class CacheClient:
    def __init__(self):
        self.values, self.writes = {}, []

    def ping(self):
        return True

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, **options):
        self.values[key] = value
        self.writes.append((key, value, options))


@pytest.fixture(autouse=True)
def cache_owner_boundary(monkeypatch):
    """Fake only the cache storage owner; actual Lua capacity has separate checks."""
    class CacheStore:
        def __init__(self, client, namespace, **policy):
            self.client, self.namespace, self.policy = client, namespace, policy

        def cache_key(self, digest):
            namespace = hashlib.sha256(self.namespace.encode()).hexdigest()
            return "agent-service:{" + namespace + "}:cache:" + digest

        def cache_get(self, digest):
            raw = self.client.get(self.cache_key(digest))
            return self.client.get(self.namespace + ":query-cache:" + digest) if raw is None else raw

        def cache_put(self, digest, encoded):
            self.client.set(self.cache_key(digest), encoded, ex=self.policy["cache_ttl_seconds"])

    monkeypatch.setattr(backends, "RedisJobStore", CacheStore)


def test_redis_cache_retains_binding_integrity_and_ttl():
    client = CacheClient()
    cache = backends.RedisQueryCache(client, "test", 30)
    binding = {"index":{"revision":"one"},"model":"m","operation":"answer_v3_original_source_context"}
    value = {"answer":"supported [context 1]"}
    cache.put(binding,value)
    assert client.writes[0][2] == {"ex":30}
    assert cache.get(binding) == value
    assert cache.get({**binding,"model":"different"}) is None
    key, raw, _ = client.writes[0]
    corrupt = json.loads(raw)
    corrupt["value"]["answer"] = "changed"
    client.values[key] = json.dumps(corrupt)
    assert cache.get(binding) is None


def test_query_owner_does_not_publish_partial_failure_to_redis():
    client = CacheClient()
    calls = []
    class FailingModel:
        def chat(self, *args, **kwargs):
            calls.append(1)
            raise RuntimeError("controlled provider failure")
    settings = SimpleNamespace(query=QuerySettings(cache_enabled=True, model_revision="fixture"),
        llm=SimpleNamespace(provider="ollama",model="fixture",temperature=0,max_tokens=128))
    search = SimpleNamespace(search=lambda **kw:[{"chunk_id":"one","text":"supported text","metadata":{}}])
    service = QueryService(settings=settings,hybrid_search=search,reranker=None,llm=FailingModel(),
        cache=backends.RedisQueryCache(client,"test",30),index_identity_provider=lambda:{"index":"one"},
        model_identity_provider=lambda:{"model":"fixture"})
    response = service.query({"query":"question","rerank":False})
    assert response.data["status"] == "failed" and calls == [1]
    assert client.writes == []


def config(tmp_path):
    return SimpleNamespace(work_root=tmp_path,settings_path=tmp_path/"settings.yaml",agent_profiles={},
        namespace="test",redis_url="redis://:PRIVATE_PASSWORD@127.0.0.1:6379/0",query_cache_ttl=30,
        max_job_seconds=10,lease_seconds=3,retry_delay_seconds=1,retention_seconds=86400,max_pending=1000,
        capacity_bytes=64*1024**2,cache_capacity_bytes=16*1024**2)


def test_queued_settings_drift_rejected_before_query_owner(tmp_path):
    cfg = config(tmp_path)
    cfg.settings_path.write_text("changed")
    with pytest.raises(backends.BackendError,match="SETTINGS_BINDING_CHANGED"):
        backends.run_query({"query":"q","_settings_sha256":"0"*64},cfg,job_id="job",timeout_seconds=5,
            service_factory=lambda *a,**k:pytest.fail("query owner invoked"),redis_client=CacheClient())


def test_query_public_response_excludes_private_evidence(tmp_path,monkeypatch):
    cfg = config(tmp_path)
    cfg.settings_path.write_text("test configuration")
    @dataclass(frozen=True)
    class Settings:
        query: QuerySettings
    monkeypatch.setattr("src.core.settings.load_settings",lambda _:Settings(QuerySettings()))
    fake = SimpleNamespace(query=lambda _:QueryResponse({"status":"completed","answer":"ok"},{"raw_secret":"PRIVATE"}))
    response = backends.run_query({"query":"q","_settings_sha256":hashlib.sha256(cfg.settings_path.read_bytes()).hexdigest()},
        cfg,job_id="job",timeout_seconds=5,redis_client=CacheClient(),service_factory=lambda *a,**k:fake)
    assert response == {"status":"completed","answer":"ok"}
    assert isinstance(fake.cache,backends.RedisQueryCache)


@pytest.mark.parametrize("dense_enabled",[False,True])
def test_sparse_no_model_cache_is_explicit_but_model_identity_is_not_invented(tmp_path,monkeypatch,dense_enabled):
    cfg = config(tmp_path)
    cfg.settings_path.write_text("fixed settings")
    @dataclass(frozen=True)
    class Settings:
        query: QuerySettings
        retrieval: object
        llm: object
    original = Settings(QuerySettings(cache_enabled=False),
        SimpleNamespace(dense_enabled=dense_enabled,dense_provider="vector_store" if dense_enabled else "disabled"),
        SimpleNamespace(provider="ollama",model="unverified-model",temperature=0,max_tokens=128))
    monkeypatch.setattr("src.core.settings.load_settings",lambda _:original)
    client, searches, factory_options = CacheClient(), [], []
    search = SimpleNamespace(search=lambda **kw:searches.append(kw) or [{"chunk_id":"one","text":"realistic fixture text","metadata":{}}])
    def factory(settings,collection,**kwargs):
        factory_options.append(kwargs)
        assert settings.query.cache_enabled is True and original.query.cache_enabled is False
        return QueryService(settings=settings,hybrid_search=search,reranker=None,collection=collection,
            cache=client,index_identity_provider=lambda:{"fixture_index":"v1"},
            model_identity_provider=kwargs.get("model_identity_provider",lambda:None))
    payload = {"query":"lookup","answer":False,"rerank":False,"rewrite":False,
               "_settings_sha256":hashlib.sha256(cfg.settings_path.read_bytes()).hexdigest()}
    first = backends.run_query(payload,cfg,job_id="first",timeout_seconds=10,redis_client=client,service_factory=factory)
    second = backends.run_query(payload,cfg,job_id="second",timeout_seconds=10,redis_client=client,service_factory=factory)
    assert first["status"] == second["status"] == "completed"
    if dense_enabled:
        assert factory_options == [{},{}] and len(searches) == 2 and not client.writes
        assert second["cache"]["disabled_reason"] == "model_identity_unavailable"
    else:
        assert len(searches) == 1 and second["cache"]["result"] == "hit"
        assert second["cache"]["model_identity"]["kind"] == "no_model_required"


def test_agent_uses_registered_pinned_profile_and_stable_conversation(tmp_path,monkeypatch):
    cfg = config(tmp_path)
    profile = tmp_path/"profile.json"
    profile.write_text('{"server_profile":true}')
    cfg.agent_profiles = {"registered":profile}
    monkeypatch.setattr(backends,"_profile",lambda *a:None)
    payload = {"profile_id":"registered","description":"modify this project",
               "_profile_sha256":hashlib.sha256(profile.read_bytes()).hexdigest()}
    calls = []
    def route(**kwargs):
        calls.append(kwargs)
        return {"status":"needs_clarification"}
    backends.run_agent(payload,cfg,job_id="abc",routing=route)
    backends.run_agent(payload,cfg,job_id="abc",routing=route)
    assert calls[0] == calls[1]
    assert calls[0]["conversation_id"] == "service-abc" and calls[0]["turn_id"] == "job-abc"
    assert Path(calls[0]["execution_config"]).read_bytes() == profile.read_bytes()
    profile.write_text("changed")
    with pytest.raises(backends.BackendError,match="PROFILE_BINDING_CHANGED"):
        backends.run_agent(payload,cfg,job_id="abc",routing=route)
    assert len(calls) == 2


@pytest.mark.parametrize("field,value", [("defer_approval",False),("run_gtfo_runtime",True),
    ("allow_game_launch",True),("allow_game_directory_write",True),("runtime_plugin_destination","C:/GTFO/mod.dll")])
def test_agent_profile_cannot_admit_runtime_or_implicit_apply(monkeypatch,field,value):
    parsed = {"defer_approval":True,"confirm_isolated_write":True,"local_ollama":True,"allow_source_disclosure":False}
    parsed[field] = value
    monkeypatch.setattr("scripts.dsh_run_request.validate_product_profile",lambda **kw:([],SimpleNamespace(**parsed)))
    with pytest.raises(backends.BackendError,match="PROFILE_EFFECTS_NOT_ALLOWED"):
        backends._profile("server-profile.json","request")


def test_service_respects_provider_authorization_from_product_owner(monkeypatch):
    parsed = SimpleNamespace(defer_approval=True, confirm_isolated_write=True,
                             local_ollama=False, allow_source_disclosure=True)
    monkeypatch.setattr("scripts.dsh_run_request.validate_product_profile", lambda **kw:([],parsed))
    assert backends._profile("registered-authorized-profile.json", "request") is parsed
    # This tests delegation only; the current normal DSH owner remains loopback-only.
    def reject(**kwargs):
        raise ValueError("PRODUCT_DISCLOSURE_AUTHORIZATION_REQUIRED")
    monkeypatch.setattr("scripts.dsh_run_request.validate_product_profile", reject)
    with pytest.raises(ValueError, match="DISCLOSURE_AUTHORIZATION_REQUIRED"):
        backends._profile("registered-profile.json", "request")


def test_agent_rejects_client_executor_override(tmp_path):
    with pytest.raises(backends.BackendError,match="PAYLOAD_FIELDS_INVALID"):
        backends.run_agent({"product_executor":"shell"},config(tmp_path),job_id="job")


class Store:
    def __init__(self,kind="query"):
        self.job = {"id":"a"*32,"kind":kind,"payload":{"query":"q"},"state":"running", "attempt":1,
                    "max_attempts":2,"timeout_seconds":10,"lease_token":"token"}
        self.finished, self.retried = [], []

    def claim(self,*args):
        return self.job

    def heartbeat(self,*args):
        return True

    def finish(self,*args,**kwargs):
        self.finished.append((args,kwargs))
        return True

    def retry(self,*args):
        self.retried.append(args)
        return True


def runner_result(envelope=None, **process):
    def run(argv,**kwargs):
        request = json.loads(Path(argv[argv.index("--request")+1]).read_bytes())
        assert "PRIVATE_PASSWORD" not in json.dumps(request)
        assert kwargs["environment"]["SERVICE_CHILD_REDIS_URL"].startswith("redis://")
        if envelope is not None:
            execute.save(argv[argv.index("--result")+1],{"schema":"service-attempt-result.v1",
                "job_id":request["job"]["id"],"attempt":request["job"]["attempt"],**envelope})
        return {"status":"exited","started":True,"exit_code":0,"timeout":False,**process}
    return run


def test_completed_job_preserves_failed_business_status(tmp_path):
    store = Store()
    worker.run_once(store,config(tmp_path),worker_id="test",runner=runner_result(
        {"ok":True,"safe_retry":False,"result":{"status":"failed","error":{"code":"QUERY_ANSWER_EMPTY"}}}))
    assert store.finished[0][0][2] == "completed"
    assert store.finished[0][1]["result"]["status"] == "failed" and not store.retried
    assert len(list(tmp_path.glob("jobs/*/*/request.json"))) == 1
    assert len(list(tmp_path.glob("jobs/*/*/attempt.json"))) == 1


@pytest.mark.parametrize("kind,safe,expected", [("query",True,"retry"),("query",False,"failed"),("agent",True,"indeterminate")])
def test_only_explicit_safe_query_failure_requests_retry(tmp_path,kind,safe,expected):
    store = Store(kind)
    worker.run_once(store,config(tmp_path),worker_id="test",runner=runner_result(
        {"ok":False,"safe_retry":safe,"execution_state":"not_started" if kind == "query" else "unknown",
         "error":{"code":"QUERY_CACHE_UNAVAILABLE_BEFORE_EXECUTION"}}))
    if expected == "retry":
        assert len(store.retried) == 1 and not store.finished
    else:
        assert not store.retried and store.finished[0][0][2] == expected


@pytest.mark.parametrize("kind,expected", [("query","timed_out"),("agent","indeterminate")])
def test_timed_out_process_does_not_publish_a_late_success(tmp_path,kind,expected):
    store = Store(kind)
    worker.run_once(store,config(tmp_path),worker_id="test",runner=runner_result(
        {"ok":True,"safe_retry":False,"result":{"status":"completed"}},timeout=True,exit_code=124))
    assert store.finished[0][0][2] == expected and store.finished[0][1]["result"] is None
    assert not store.retried


def test_lost_lease_suppresses_publication_but_retains_attempt(tmp_path,monkeypatch):
    class Lost(worker.LeaseHeartbeat):
        def start(self):
            self.lost.set()
        def stop(self):
            pass
    monkeypatch.setattr(worker,"LeaseHeartbeat",Lost)
    store = Store()
    worker.run_once(store,config(tmp_path),worker_id="test",runner=runner_result(
        {"ok":True,"safe_retry":False,"result":{"status":"completed"}}))
    assert not store.finished and not store.retried
    receipt = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert receipt["publication"] == "suppressed_lease_lost" and receipt["result_ref"]


def test_public_result_redacts_credentials_without_inventing_delivery():
    value = {"status":"awaiting_approval","raw_response":{"private":"x"},"api_token":"key",
             "message":"credential PRIVATE_PASSWORD","usage":{"total_tokens":5}}
    assert backends.public_result(value,["PRIVATE_PASSWORD"]) == {
        "status":"awaiting_approval","message":"credential [redacted]","usage":{"total_tokens":5}}


def test_child_result_does_not_expose_private_business_envelope(tmp_path,monkeypatch):
    request = {"schema":"service-attempt-request.v1","job":{"id":"job","attempt":1,"kind":"query",
        "payload":{},"timeout_seconds":5},"server_config":{"namespace":"test","settings_path":"x",
        "work_root":str(tmp_path),"agent_profiles":{},"query_cache_ttl":30}}
    execute.save(tmp_path/"request.json",request)
    monkeypatch.setattr(backends,"run_query",lambda *a,**kw:{"status":"completed","answer":"ok","raw_response":{"secret":"x"}})
    execute.execute_request(tmp_path/"request.json",tmp_path/"result.json")
    result = json.loads((tmp_path/"result.json").read_bytes())
    assert result["ok"] is True and "raw_response" not in result["result"]
    assert "raw_response" in json.loads((tmp_path/"business-result.private.json").read_bytes())


def test_false_safe_retry_on_semantic_failure_is_rejected(tmp_path):
    store = Store()
    worker.run_once(store,config(tmp_path),worker_id="test",runner=runner_result(
        {"ok":False,"safe_retry":True,"execution_state":"not_started","error":{"code":"SEMANTIC_VALIDATION_FAILED"}}))
    assert not store.retried and store.finished[0][0][2] == "failed"


def test_actual_heartbeat_thread_fences_late_result(tmp_path):
    class LosingStore(Store):
        def __init__(self):
            super().__init__()
            self.observed = threading.Event()
        def heartbeat(self,*args):
            self.observed.set()
            return False
    store, cfg = LosingStore(), config(tmp_path)
    cfg.lease_seconds = 0.15  # Controlled short interval, not the production ServiceConfig.
    completed = runner_result({"ok":True,"safe_retry":False,"result":{"status":"completed"}})
    def runner(*args,**kwargs):
        assert store.observed.wait(2)
        return completed(*args,**kwargs)
    worker.run_once(store,cfg,worker_id="test",runner=runner)
    assert not store.finished and not store.retried


def test_finish_fence_rejection_is_not_a_second_publication(tmp_path):
    store = Store()
    def rejected(*args,**kwargs):
        store.finished.append((args,kwargs))
        return False
    store.finish = rejected
    worker.run_once(store,config(tmp_path),worker_id="test",runner=runner_result(
        {"ok":True,"safe_retry":False,"result":{"status":"completed"}}))
    assert len(store.finished) == 1
    evidence = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert evidence["publication"] == "suppressed_fence_rejected"


def test_unknown_finish_reply_does_not_publish_another_terminal_state(tmp_path):
    store = Store()
    def lost_reply(*args, **kwargs):
        store.finished.append((args, kwargs))
        raise ConnectionError("reply lost after possible Redis commit")
    store.finish = lost_reply
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner_result(
        {"ok":True,"safe_retry":False,"result":{"status":"completed"}}))
    assert len(store.finished) == 1 and not store.retried
    evidence = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert evidence["publication"] == "store_response_unknown"


def test_cleanup_failure_cannot_publish_success(tmp_path):
    store = Store("agent")
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner_result(
        {"ok":True,"safe_retry":False,"result":{"status":"awaiting_approval"}},
        status="infrastructure_failed", cleanup_error_type="PermissionError"))
    assert store.finished[0][0][2] == "indeterminate"
    assert store.finished[0][1]["result"] is None


def test_worker_does_not_disclose_http_auth_to_business_child(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVICE_API_TOKEN", "http-only-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "server-registered-provider-key")
    def runner(*args, **kwargs):
        assert "SERVICE_API_TOKEN" not in kwargs["environment"]
        assert kwargs["environment"]["OPENAI_API_KEY"] == "server-registered-provider-key"
        assert isinstance(kwargs["cancel_event"], threading.Event)
        return runner_result({"ok":True,"safe_retry":False,"result":{"status":"completed"}})(*args, **kwargs)
    store = Store()
    worker.run_once(store, config(tmp_path), worker_id="test", runner=runner)
    assert store.finished[0][0][2] == "completed"


def test_bounded_real_child_keeps_output_and_nonzero_exit(tmp_path):
    result = execute.execute_process([sys.executable,"-I","-B","-c","print('controlled child'); raise SystemExit(7)"],
        cwd=tmp_path,output=tmp_path/"process",deadline=time.monotonic()+10)
    assert result["exit_code"] == 7 and result["timeout"] is False
    assert b"controlled child" in Path(result["stdout_ref"]["path"]).read_bytes()


def test_bounded_real_child_obeys_absolute_deadline(tmp_path):
    started = time.monotonic()
    result = execute.execute_process([sys.executable,"-I","-B","-c","import time; time.sleep(60)"],
        cwd=tmp_path,output=tmp_path/"process",deadline=started+1)
    assert result["timeout"] is True and result["exit_code"] is not None
    assert time.monotonic()-started < 10


def test_precancelled_child_never_starts(tmp_path):
    cancel = threading.Event()
    cancel.set()
    result = execute.execute_process([sys.executable, "-c", "raise SystemExit(99)"],
        cwd=tmp_path, output=tmp_path/"process", deadline=time.monotonic()+10, cancel_event=cancel)
    assert result["status"] == "cancelled_before_start" and result["started"] is False
    assert not (tmp_path/"process").exists()


def test_real_lease_loss_stops_owned_child_and_retains_evidence(tmp_path):
    class LosingStore(Store):
        def heartbeat(self, *args):
            return False
    store, cfg = LosingStore("agent"), config(tmp_path)
    cfg.lease_seconds = 0.3
    def runner(argv, **kwargs):
        return execute.execute_process([sys.executable, "-I", "-B", "-c",
            "import time; print('owned child started',flush=True); time.sleep(60)"], **kwargs)
    started = time.monotonic()
    worker.run_once(store, cfg, worker_id="test", runner=runner)
    receipt = json.loads(next(tmp_path.glob("jobs/*/*/attempt.json")).read_bytes())
    assert not store.finished and not store.retried
    assert receipt["publication"] == "suppressed_lease_lost"
    assert receipt["local_process_cancel_on_lease_loss"] is True
    assert receipt["process"]["cancelled"] is True and receipt["process"]["timeout"] is False
    assert receipt["process"]["provider_server_state"] == "not_established_by_client_process_cleanup"
    assert time.monotonic()-started < 5


def test_real_cancel_terminates_tool_descendant(tmp_path):
    ready, leaked = tmp_path/"descendant-ready", tmp_path/"leaked-side-effect"
    child = tmp_path/"child.py"
    child.write_text("import time\nfrom pathlib import Path\n"
        f"Path({str(ready)!r}).write_text('ready')\ntime.sleep(1.5)\n"
        f"Path({str(leaked)!r}).write_text('unwanted late effect')\ntime.sleep(60)\n", encoding="utf-8")
    parent = tmp_path/"parent.py"
    parent.write_text("import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-I','-B',{str(child)!r}])\ntime.sleep(60)\n", encoding="utf-8")
    cancel = threading.Event()
    def observe_then_cancel():
        until = time.monotonic()+5
        while not ready.exists() and time.monotonic() < until:
            time.sleep(0.02)
        cancel.set()
    observer = threading.Thread(target=observe_then_cancel)
    observer.start()
    result = execute.execute_process([sys.executable,"-I","-B",str(parent)],cwd=tmp_path,
        output=tmp_path/"process", deadline=time.monotonic()+8, cancel_event=cancel)
    observer.join(timeout=6)
    assert ready.exists() and result["cancelled"] and not result["timeout"]
    time.sleep(1.6)
    assert not leaked.exists()
    assert result["owned_process_tree_cleanup"] in {"job_closed_kill_on_close", "owned_process_group_terminated"}


def configure_worker_loop(monkeypatch, tmp_path, store):
    cfg = config(tmp_path)
    cfg.retention_seconds, cfg.max_pending, cfg.poll_seconds = 60, 10, 0.1
    monkeypatch.setattr("src.service_layer.config.ServiceConfig.from_env", lambda:cfg)
    monkeypatch.setattr(worker, "worker_redis_client", lambda *args, **kwargs:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr("src.service_layer.store.RedisJobStore", lambda *args, **kwargs:store)
    return cfg


def test_worker_recovers_transient_poll_errors_without_replaying_agent(tmp_path, monkeypatch, capsys):
    import redis
    class StopLoopError(Exception):
        pass
    class RestartingStore(Store):
        def __init__(self):
            super().__init__("agent")
            self.recovery_calls, self.claims = 0, 0
        def recover_expired(self):
            self.recovery_calls += 1
            if self.recovery_calls == 1:
                raise redis.ConnectionError("redis://:PRIVATE_PASSWORD@host unavailable")
        def claim(self, *args):
            self.claims += 1
            if self.claims == 1:
                raise redis.TimeoutError("PRIVATE_PASSWORD: claim may already be committed")
            if self.claims == 2:
                return None  # Unknown claimed Agent remains Store-owned; no local reconstruction.
            raise StopLoopError()
    store = RestartingStore()
    cfg = configure_worker_loop(monkeypatch, tmp_path, store)
    sleeps = []
    monkeypatch.setattr(worker.WorkerShutdown, "wait", lambda self, delay:sleeps.append(delay))
    with pytest.raises(StopLoopError):
        worker.main(["--worker-id", "test"])
    assert store.claims == 3 and store.recovery_calls == 4
    assert sleeps == [cfg.retry_delay_seconds, cfg.retry_delay_seconds*2, cfg.poll_seconds]
    assert not store.finished and not store.retried
    assert not list(tmp_path.glob("jobs/*/*/request.json"))
    logged = capsys.readouterr().err
    assert "PRIVATE_PASSWORD" not in logged and "redis://" not in logged
    assert [json.loads(line)["error_type"] for line in logged.splitlines()] == ["ConnectionError", "TimeoutError"]


@pytest.mark.parametrize("once,error_name", [(True,"ConnectionError"),(False,"AuthenticationError"),(False,"ResponseError")])
def test_worker_once_or_permanent_store_error_exits_without_retry(tmp_path, monkeypatch, capsys, once, error_name):
    import redis
    class Unavailable:
        def recover_expired(self):
            raise getattr(redis.exceptions,error_name)("PRIVATE_PASSWORD")
        def claim(self, *args):
            pytest.fail("claim must not run after recovery failed")
    configure_worker_loop(monkeypatch, tmp_path, Unavailable())
    monkeypatch.setattr(worker.time, "sleep", lambda _:pytest.fail("unexpected retry"))
    assert worker.main(["--once"] if once else []) == 2
    logged = capsys.readouterr().err
    assert "PRIVATE_PASSWORD" not in logged and json.loads(logged)["retry"] is False
