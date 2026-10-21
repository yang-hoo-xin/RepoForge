"""Authenticated job submission; the API process never runs the business owners."""
from __future__ import annotations

import hashlib
import secrets
import time
from pathlib import Path

import redis
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from src.service_layer.config import ServiceConfig
from src.service_layer.http_limits import BodyLimitMiddleware
from src.service_layer.schemas import JobSubmission
from src.service_layer.store import (
    CapacityExceeded,
    CapacityPolicyMismatch,
    CapacityStateError,
    IdempotencyConflict,
    QueueFull,
    RedisJobStore,
)


def public_job(job):
    # 为什么要有它：租约token是worker写入凭证，内部配置hash和路径也不应随队列状态暴露。
    fields = ("id", "kind", "state", "attempt", "max_attempts", "timeout_seconds",
              "created_at", "queued_at", "started_at", "finished_at", "result", "error")
    return {name: job.get(name) for name in fields}


def create_app(config: ServiceConfig | None = None, *, store=None):
    config = config or ServiceConfig.from_env()
    if not config.api_token:
        raise ValueError("Set SERVICE_API_TOKEN before starting the HTTP service")
    store = store or RedisJobStore(
        redis.Redis.from_url(config.redis_url, decode_responses=True, socket_timeout=5,
                             socket_connect_timeout=5),
        namespace=config.namespace, retention_seconds=config.retention_seconds,
        max_pending=config.max_pending,
        capacity_bytes=config.capacity_bytes, cache_capacity_bytes=config.cache_capacity_bytes,
        cache_ttl_seconds=config.query_cache_ttl,
    )
    registry = CollectorRegistry()
    requests = Counter("service_http_requests_total", "HTTP responses", ["route", "method", "status"], registry=registry)
    duration = Histogram("service_http_request_seconds", "HTTP request latency, excluding asynchronous job execution",
                         ["route", "method"], registry=registry)
    app = FastAPI(title="RAG and controlled Agent service", version="0.1.0")
    app.add_middleware(BodyLimitMiddleware)
    app.state.job_store = store
    app.state.service_config = config

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # 为什么要有它：校验错误不回显整份任务或异常对象；非法Unicode也必须稳定返回422，不能在序列化错误时变成500。
        def safe(value):
            return str(value).encode("utf-8", "backslashreplace").decode("utf-8")[:500]

        return JSONResponse(status_code=422, content={"detail": [
            {"type": safe(item["type"]), "loc": [safe(part) for part in item["loc"]],
             "msg": safe(item["msg"])} for item in error.errors()
        ]})

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            # 为什么要有它：模板路由保持有限标签集合，不能把job id、问题文本或凭据写进监控标签。
            method = request.method if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"} else "OTHER"
            requests.labels(route, method, str(status)).inc()
            duration.labels(route, method).observe(time.perf_counter() - started)

    def authorized(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization.encode(), ("Bearer " + config.api_token).encode()):
            raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})

    def available(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except CapacityExceeded:
            # Scope: refusal before new work is admitted, including a new rate identity.
            # 为什么要有它：容量不足不能返回已接单，也不能绕过限流记录占用；返回可重试的背压，不回显内部数据。
            raise HTTPException(429, "Service capacity reached", headers={"Retry-After": "1"}) from None
        except (CapacityPolicyMismatch, CapacityStateError):
            raise HTTPException(503, "Job store capacity policy unavailable", headers={"Retry-After": "1"}) from None
        except redis.RedisError:
            raise HTTPException(503, "Job store unavailable", headers={"Retry-After": "1"}) from None

    @app.get("/health")
    def health():
        available(store.client.ping)
        return {"status": "ok", "service": "rag-agent", "execution": "separate_worker"}

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(authorized)])
    def submit(body: JobSubmission, response: Response,
               idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=128,
                                              pattern=r"^[\x21-\x7e]+$")):
        allowed, wait = available(store.rate_limit, "configured_owner", config.requests_per_minute, 60)
        if not allowed:
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(max(1, int(wait)))})
        if body.timeout_seconds > config.max_job_seconds:
            raise HTTPException(422, "Requested timeout exceeds the configured limit")
        payload = dict(body.payload)
        selected = config.settings_path if body.kind == "query" else config.agent_profiles.get(payload["profile_id"])
        if selected is None:
            raise HTTPException(422, "Unknown registered Agent profile")
        try:
            digest = hashlib.sha256(Path(selected).read_bytes()).hexdigest()
        except OSError:
            raise HTTPException(503, "Configured business input is unavailable") from None
        payload["_settings_sha256" if body.kind == "query" else "_profile_sha256"] = digest
        try:
            job, created = available(store.submit, body.kind, payload, idempotency_key,
                                     body.timeout_seconds, body.max_attempts)
        except IdempotencyConflict:
            raise HTTPException(409, "Idempotency key is already bound to a different request") from None
        except QueueFull:
            raise HTTPException(429, "Queue capacity reached", headers={"Retry-After": "1"}) from None
        response.status_code = 202 if created else 200
        response.headers["Location"] = "/v1/jobs/" + job["id"]
        return {"job": public_job(job), "created": created,
                "completion_scope": "Job completion preserves the domain result; it does not imply Agent delivery."}

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorized)])
    def status(job_id: str):
        import re
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise HTTPException(404, "Job not found")
        job = available(store.get, job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return public_job(job)

    @app.get("/v1/features/{feature_id}/{view}", dependencies=[Depends(authorized)])
    def artifacts(feature_id: str, view: str):
        if config.artifact_root is None:
            raise HTTPException(404, "Artifact inspection is not configured")
        if view not in {"artifacts", "status", "dashboard"}:
            raise HTTPException(404, "Unknown artifact view")
        from urllib.parse import quote

        from src.mod_agent.api.http_server import ModAgentHttpApi
        from src.mod_agent.artifacts import ArtifactStore

        outcome = ModAgentHttpApi(ArtifactStore(config.artifact_root)).handle_get(
            "/features/" + quote(feature_id, safe="") + "/" + view)
        return JSONResponse(outcome.payload, status_code=outcome.status_code, headers=outcome.headers)

    @app.get("/metrics")
    def metrics():
        from src.service_layer.metrics import queue_metrics

        snapshot = available(store.metrics)
        body = generate_latest(registry) + queue_metrics(snapshot)
        return Response(body, media_type="text/plain; version=0.0.4; charset=utf-8")

    return app
