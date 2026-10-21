"""Explicit service configuration; business execution stays in existing owners."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    redis_url: str = "redis://127.0.0.1:6379/0"
    namespace: str = "agent-service"
    work_root: Path = Path("data/service_jobs")
    settings_path: Path = Path("config/settings.yaml")
    agent_profiles: dict[str, Path] = field(default_factory=dict)
    query_cache_ttl: int = 3600
    query_execution_mode: str = "resident_sparse"
    max_job_seconds: float = 600
    lease_seconds: float = 30
    poll_seconds: float = 0.5
    retry_delay_seconds: float = 2
    retention_seconds: int = 86400
    max_pending: int = 1000
    capacity_bytes: int = 64 * 1024 * 1024
    cache_capacity_bytes: int = 16 * 1024 * 1024
    requests_per_minute: int = 120
    api_token: str = field(default="", repr=False, compare=False)
    artifact_root: Path | None = None

    def __post_init__(self):
        import re

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.namespace):
            raise ValueError("Invalid service namespace")
        if not self.redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("Redis URL is required")
        if self.query_execution_mode not in {"isolated", "resident_sparse"}:
            raise ValueError("Invalid query execution mode")
        for name in ("max_job_seconds", "lease_seconds", "poll_seconds", "retry_delay_seconds"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("query_cache_ttl", "retention_seconds", "max_pending", "requests_per_minute"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # Scope: namespace encoded-data policy, not Redis/RSS memory limits.
        # 为什么要有它：API与Worker必须使用相同且精确的额度；布尔、超大浮点和缓存超过总额不能悄悄改变准入。
        for name in ("capacity_bytes", "cache_capacity_bytes"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 2**40:
                raise ValueError(f"{name} must be an integer between 1 and 2**40")
        if self.capacity_bytes < 8192:
            raise ValueError("capacity_bytes must cover the fixed 8192-byte metadata reservation")
        if self.cache_capacity_bytes > self.capacity_bytes:
            raise ValueError("cache_capacity_bytes must not exceed capacity_bytes")
        if self.lease_seconds < 3 or self.poll_seconds >= self.lease_seconds / 3:
            raise ValueError("Lease must leave room for heartbeat renewal")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) for key in self.agent_profiles):
            raise ValueError("Invalid registered profile id")
        for name in ("work_root", "settings_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())
        object.__setattr__(self, "agent_profiles", {key: Path(value).resolve() for key, value in self.agent_profiles.items()})
        if self.artifact_root is not None:
            object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())

    def to_dict(self):
        # 为什么要有它：子进程需Redis/业务配置，但不需要HTTP入口凭据；内部配置不作为API响应输出。
        return {name: ({key: str(path) for key, path in value.items()} if name == "agent_profiles"
                       else str(value) if isinstance(value, Path) else value)
                for name, value in vars(self).items() if name != "api_token"}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    @classmethod
    def from_env(cls):
        return cls(
            redis_url=os.environ.get("SERVICE_REDIS_URL", "redis://127.0.0.1:6379/0"),
            namespace=os.environ.get("SERVICE_NAMESPACE", "agent-service"),
            work_root=Path(os.environ.get("SERVICE_WORK_ROOT", "data/service_jobs")),
            settings_path=Path(os.environ.get("SERVICE_SETTINGS", "config/settings.yaml")),
            agent_profiles=json.loads(os.environ.get("SERVICE_AGENT_PROFILES", "{}")),
            query_cache_ttl=int(os.environ.get("SERVICE_QUERY_CACHE_TTL", "3600")),
            query_execution_mode=os.environ.get("SERVICE_QUERY_EXECUTION_MODE", "resident_sparse"),
            max_job_seconds=float(os.environ.get("SERVICE_MAX_JOB_SECONDS", "600")),
            lease_seconds=float(os.environ.get("SERVICE_LEASE_SECONDS", "30")),
            poll_seconds=float(os.environ.get("SERVICE_POLL_SECONDS", "0.5")),
            retry_delay_seconds=float(os.environ.get("SERVICE_RETRY_DELAY_SECONDS", "2")),
            retention_seconds=int(os.environ.get("SERVICE_RETENTION_SECONDS", "86400")),
            max_pending=int(os.environ.get("SERVICE_MAX_PENDING", "1000")),
            capacity_bytes=int(os.environ.get("SERVICE_CAPACITY_BYTES", str(64 * 1024 * 1024))),
            cache_capacity_bytes=int(os.environ.get("SERVICE_CACHE_CAPACITY_BYTES", str(16 * 1024 * 1024))),
            requests_per_minute=int(os.environ.get("SERVICE_REQUESTS_PER_MINUTE", "120")),
            api_token=os.environ.get("SERVICE_API_TOKEN", ""),
            artifact_root=Path(os.environ["SERVICE_ARTIFACT_ROOT"]) if os.environ.get("SERVICE_ARTIFACT_ROOT") else None,
        )
