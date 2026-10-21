"""Synchronous Redis jobs with atomic idempotency, leases and bounded recovery."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any
from uuid import uuid4

from redis import Redis
from redis.exceptions import ResponseError

from src.service_layer.capacity import (
    CACHE_CAPACITY_BYTES,
    CACHE_ENTRY_BYTES,
    CAPACITY_BYTES,
    CapacityExceeded,
    CapacityPolicyMismatch,
    CapacityStateError,
    policy_json,
)
from src.service_layer.capacity import (
    LUA as CAPACITY_LUA,
)


class IdempotencyConflict(ValueError):
    """The supplied key already identifies different request bytes (HTTP 409)."""

    def __init__(self, job_id: str):
        super().__init__("Idempotency key already belongs to a different request")
        self.job_id = job_id


class QueueFull(RuntimeError):
    """The namespace has reached its nonterminal-job admission limit."""


class ResultTooLargeError(ValueError):
    """A valid result exceeded the local byte limit, before any Redis operation."""

    def __init__(self, size_bytes: int, limit_bytes: int):
        super().__init__("JSON value exceeds UTF-8 size limit")
        self.size_bytes, self.limit_bytes = size_bytes, limit_bytes


_TERMINAL = frozenset({"completed", "failed", "timed_out", "indeterminate", "cancelled"})
_MAX_PAYLOAD_BYTES = 128 * 1024
_MAX_RESULT_BYTES = 2 * 1024 * 1024
_MAX_ERROR_BYTES = 64 * 1024

# 为什么要有它：幂等判定、队列修改和租约fence必须在同一次Redis原子执行中完成，不能由并发客户端先读后写。
# 为什么要有它：时间统一到毫秒，避免cjson时间精度和zset分值不同而在正常到期边界误报损坏。
_SCRIPT = r"""
local op, prefix = ARGV[1], ARGV[2]
local args = cjson.decode(ARGV[3])
local retention = tonumber(ARGV[4])
local capacity = tonumber(ARGV[5])
local ready, processing, stats = KEYS[1], KEYS[2], KEYS[3]
local stamp = redis.call('TIME')
local now_ms = tonumber(stamp[1]) * 1000 + math.floor(tonumber(stamp[2]) / 1000)
local now = now_ms / 1000
""" + CAPACITY_LUA + r"""
local function seconds(value) return math.ceil(value * 1000) / 1000 end
local function jobkey(id) return prefix .. 'job:' .. id end
local function idemkey(job) return prefix .. 'idem:' .. job._idempotency_digest end
local function count(name) write('HINCRBY', stats, name, 1) end
local function reply(value) return cjson.encode(value) end
local function readjob(id)
    local raw = redis.call('GET', jobkey(id))
    if not raw then return nil end
    return cjson.decode(raw)
end
local function validindex(job)
    local raw = redis.call('GET', idemkey(job))
    if not raw then return false end
    local index = cjson.decode(raw)
    return index.job_id == job.id and index.request_hash == job._request_hash
end
local function saveactive(job,created)
    local encoded=cjson.encode(job)
    capacity_job(job,encoded,false,created)
    write('SET', jobkey(job.id), encoded)
    write('PERSIST', idemkey(job))
end
local function saveterminal(job)
    write('ZREM', ready, job.id)
    write('ZREM', processing, job.id)
    job.lease_token = cjson.null
    job.lease_until = cjson.null
    job.finished_at = now
    local encoded=cjson.encode(job)
    capacity_job(job,encoded,true,false)
    write('SET', jobkey(job.id), encoded, 'PX', retention)
    write('PEXPIRE', idemkey(job), retention)
    count(job.state)
end
local function fence(job, token)
    return job and job.state == 'running' and job.lease_token == token
        and type(job.lease_until) == 'number' and job.lease_until > now
end
local function closeattempt(job, state, result_json, error_json, cost_unknown)
    local attempt = job.attempts_history[#job.attempts_history]
    attempt.state = state
    attempt.finished_at = now
    attempt.result_json = result_json
    attempt.error_json = error_json
    attempt.cost_unknown = cost_unknown
end
local function recovery_prior_error(job)
    -- Scope: validate this job's complete recovery history before any batch write.
    -- 为什么要有它：错误正文已在attempt历史中；先验证同job的后向引用，不能复制转义正文或在第100项损坏时先改前99项。
    local history, number = job.attempts_history, job.attempt
    if type(history) ~= 'table' or type(number) ~= 'number' or number < 1 or number % 1 ~= 0
        or type(job.error_json) ~= 'string' then return false end
    local count_entries = 0
    for index,_ in pairs(history) do
        if type(index) ~= 'number' or index % 1 ~= 0 or index < 1 or index > number then return false end
        count_entries = count_entries + 1
    end
    if count_entries ~= number then return false end
    for index=1,number do
        local attempt = history[index]
        if type(attempt) ~= 'table' or attempt.attempt ~= index or type(attempt.error_json) ~= 'string' then return false end
        if index == number then
            if attempt.state ~= 'running' or attempt.error_json ~= 'null' then return false end
        else
            if attempt.state ~= 'retry_scheduled' and attempt.state ~= 'lease_expired' then return false end
            local valid, value = pcall(cjson.decode, attempt.error_json)
            if not valid then return false end
            -- Scope: only server-produced lease-expiry v2 has reference semantics.
            -- 为什么要有它：普通应用错误里的自由字段不能指定恢复操作；旧v1正文和retry原错误只作为历史数据保留。
            if attempt.state == 'lease_expired' and type(value) == 'table'
                and value.schema == 'service-lease-expired-error.v2' then
                if value.code ~= 'lease_expired' or value.cost_unknown ~= true then return false end
                local prior = value.prior_error_ref
                if index == 1 or history[index-1].error_json == 'null' then
                    if prior ~= cjson.null then return false end
                else
                    if type(prior) ~= 'table' or prior.version ~= 1 or type(prior.attempt) ~= 'number'
                        or prior.attempt ~= index-1 or prior.field ~= 'error' then return false end
                    local fields = 0
                    for key,_ in pairs(prior) do
                        if key ~= 'version' and key ~= 'attempt' and key ~= 'field' then return false end
                        fields = fields + 1
                    end
                    if fields ~= 3 then return false end
                end
            end
        end
    end
    if number == 1 then
        if job.error_json ~= 'null' then return false end
    elseif job.error_json ~= history[number-1].error_json then
        return false
    end
    local prior = cjson.null
    if job.error_json ~= 'null' then prior = {version=1, attempt=number-1, field='error'} end
    return true, prior
end

if op == 'submit' then
    local ikey = prefix .. 'idem:' .. args.idempotency_digest
    local prior = redis.call('GET', ikey)
    if prior then
        local index = cjson.decode(prior)
        if index.request_hash ~= args.request_hash then
            count('idempotency_conflicts')
            return respond({code='conflict', job_id=index.job_id})
        end
        local job = readjob(index.job_id)
        if not job or not validindex(job) then return redis.error_reply('JOB_INDEX_INCONSISTENT') end
        count('duplicate_submits')
        return respond({code='existing', job=job})
    end
    if redis.call('ZCARD', ready) + redis.call('ZCARD', processing) >= capacity then
        count('queue_rejections')
        return respond({code='full'})
    end
    if redis.call('EXISTS', jobkey(args.job_id)) ~= 0 then return redis.error_reply('JOB_ID_COLLISION') end
    local job = {id=args.job_id, kind=args.kind, payload_json=args.payload_json, state='queued',
        attempt=0, max_attempts=args.max_attempts, timeout_seconds=args.timeout_seconds,
        created_at=now, started_at=cjson.null, finished_at=cjson.null, available_at=now,
        lease_token=cjson.null, lease_until=cjson.null, attempt_deadline_at=cjson.null,
        worker_id=cjson.null, result_json='null', error_json='null', attempts_history={},
        _request_hash=args.request_hash, _idempotency_digest=args.idempotency_digest}
    write('SET', ikey, reply({job_id=job.id,request_hash=args.request_hash}))
    saveactive(job,true)
    write('ZADD', ready, now, job.id)
    count('submitted')
    return respond({code='created', job=job})
elseif op == 'claim' then
    local ids = redis.call('ZRANGEBYSCORE', ready, '-inf', now, 'LIMIT', 0, 1)
    if #ids == 0 then return respond({code='empty'}) end
    local job = readjob(ids[1])
    if not job or job.state ~= 'queued' or not validindex(job) or job.attempt >= job.max_attempts then
        return redis.error_reply('READY_JOB_INCONSISTENT')
    end
    job.attempt = job.attempt + 1
    job.state = 'running'
    job.started_at = now
    job.attempt_deadline_at = now + seconds(job.timeout_seconds)
    job.lease_until = now + seconds(args.lease_seconds)
    job.lease_token = args.lease_token
    job.worker_id = args.worker_id
    table.insert(job.attempts_history, {attempt=job.attempt, state='running', started_at=now,
        finished_at=cjson.null, lease_token=args.lease_token, worker_id=args.worker_id,
        result_json='null', error_json='null', cost_unknown=cjson.null})
    write('ZREM', ready, job.id)
    write('ZADD', processing, job.lease_until, job.id)
    saveactive(job)
    count('claimed')
    return respond({code='claimed',job=job})
elseif op == 'heartbeat' or op == 'finish' or op == 'retry' then
    local job = readjob(args.job_id)
    if not fence(job,args.lease_token) then
        count('late_writes_rejected')
        return respond({code='rejected'})
    end
    if not validindex(job) then return redis.error_reply('RUNNING_JOB_INDEX_INCONSISTENT') end
    if op == 'heartbeat' then
        job.lease_until = now + seconds(args.lease_seconds)
        write('ZADD',processing,job.lease_until,job.id)
        saveactive(job)
        return respond({code='accepted'})
    elseif op == 'finish' then
        job.state=args.state
        job.result_json=args.result_json
        job.error_json=args.error_json
        closeattempt(job,args.state,args.result_json,args.error_json,
            args.state == 'indeterminate' and true or cjson.null)
        -- Scope: only one completed non-null result. 为什么要有它：顶层保留原始JSON，最后attempt只存版本化同job引用，不能删除旧失败、错误或其他尝试结果。
        if args.state == 'completed' and args.result_json ~= 'null' then
            local last = job.attempts_history[#job.attempts_history]
            last.result_json = nil
            last._result_from_job = {version=1, attempt=job.attempt}
        end
        saveterminal(job)
        return respond({code='accepted'})
    else
        if job.kind ~= 'query' then return respond({code='rejected'}) end
        job.error_json=args.error_json
        write('ZREM',processing,job.id)
        if job.attempt >= job.max_attempts then
            job.state='failed'
            closeattempt(job,'failed','null',args.error_json,cjson.null)
            saveterminal(job)
        else
            closeattempt(job,'retry_scheduled','null',args.error_json,cjson.null)
            job.state='queued'
            job.lease_token=cjson.null
            job.lease_until=cjson.null
            job.available_at=now+seconds(args.delay_seconds)
            write('ZADD',ready,job.available_at,job.id)
            saveactive(job)
            count('retries')
        end
        return respond({code='accepted'})
    end
elseif op == 'recover' then
    local ids = redis.call('ZRANGEBYSCORE',processing,'-inf',now,'LIMIT',0,100)
    local jobs, prior_errors = {}, {}
    for _,id in ipairs(ids) do
        local job=readjob(id)
        if not job or job.state~='running' or not validindex(job) or job.lease_until>now then
            return redis.error_reply('EXPIRED_JOB_INCONSISTENT')
        end
        local history_valid, prior_error = recovery_prior_error(job)
        if not history_valid then return redis.error_reply('RECOVERY_ERROR_HISTORY_INCONSISTENT') end
        table.insert(jobs,job)
        table.insert(prior_errors,prior_error)
    end
    local result={expired=0,queries_requeued=0,queries_exhausted=0,agents_indeterminate=0}
    for index,job in ipairs(jobs) do
        -- Scope: new public errors use v2 references, not byte-identical v1 nested error bodies.
        -- 为什么要有它：public_job仅返回顶层error，必须明确结构演进；_job不递归展开，原错误仍可在服务端同job历史核对。
        local error_json=reply({schema='service-lease-expired-error.v2',code='lease_expired',cost_unknown=true,
            prior_error_ref=prior_errors[index],reason='Worker lease expired; previous execution cost/effects may be unknown'})
        closeattempt(job,'lease_expired','null',error_json,true)
        job.error_json=error_json
        job.lease_token=cjson.null
        job.lease_until=cjson.null
        write('ZREM',processing,job.id)
        if job.kind=='query' and job.attempt<job.max_attempts then
            job.state='queued'
            job.available_at=now
            write('ZADD',ready,now,job.id)
            saveactive(job)
            result.queries_requeued=result.queries_requeued+1
            count('retries')
        else
            job.state=job.kind=='agent' and 'indeterminate' or 'failed'
            saveterminal(job)
            if job.kind=='agent' then result.agents_indeterminate=result.agents_indeterminate+1
            else result.queries_exhausted=result.queries_exhausted+1 end
        end
        result.expired=result.expired+1
        count('expired_leases')
    end
    -- Scope: successful recovery removes exactly this prevalidated batch from processing.
    -- 为什么要有它：队列写入现在统一延后提交；不能把提交前ZCOUNT误当完成后的剩余数，也不能为读结果再提交一次。
    result.remaining_expired=redis.call('ZCOUNT',processing,'-inf',now)-#jobs
    return respond(result)
elseif op == 'rate_limit' then
    local key=prefix .. 'rate:' .. args.identity_digest
    local raw=redis.call('GET',key)
    local ttl=raw and redis.call('PTTL',key) or args.window_ms
    if ttl<0 then return redis.error_reply('RATE_LIMIT_EXPIRATION_MISSING') end
    local previous=raw and tonumber(raw) or 0
    if not valid_integer(previous) then return redis.error_reply('RATE_LIMIT_VALUE_INVALID') end
    -- Scope: a bounded rate counter, not unlimited INCR history.
    -- 为什么要有它：超过最高合法limit后继续加数字没有业务收益，饱和计数让每个身份的预留有固定上界。
    local value=math.min(previous+1,1000001)
    local encoded=tostring(value)
    local deadline=now_ms+args.window_ms
    if raw then
        local row=charge('r:'..args.identity_digest)
        if not row or row.kind~='rate' or row.expires<=0 then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
        deadline=row.expires
    end
    -- Scope: existing rate identities keep their original deadline and seven-byte maximum value reservation.
    -- 为什么要有它：9变10不应争抢新额度，读取PTTL再加TIME也不能逐次延长限流窗口。
    capacity_item('rate',args.identity_digest,'1000001',math.max(0,deadline-now_ms))
    planned['r:'..args.identity_digest].expires=deadline
    write('SET',key,encoded,'PXAT',deadline)
    return respond({allowed=value<=args.limit,retry_after=value<=args.limit and 0 or math.max(0,deadline-now_ms)/1000})
elseif op == 'cache_put' then
    if #args.encoded>policy.cache_entry_bytes then cap_error('SERVICE_CACHE_ENTRY_TOO_LARGE') end
    capacity_item('cache',args.digest,args.encoded,policy.cache_ttl_ms)
    write('SET',prefix..'cache:'..args.digest,args.encoded,'PX',policy.cache_ttl_ms)
    return respond({code='accepted'})
elseif op == 'capacity_reclaim' then
    local released=capacity_commit()
    return reply({released=released,remaining_expired=redis.call('ZCOUNT',expiry,'-inf',now_ms)})
elseif op == 'metrics' then
    local raw=redis.call('HGETALL',stats)
    local counters={}
    for i=1,#raw,2 do counters[raw[i]]=tonumber(raw[i+1]) end
    local queued=redis.call('ZCARD',ready)
    local running=redis.call('ZCARD',processing)
    return reply({queued=queued,running=running,pending=queued+running,
        ready=redis.call('ZCOUNT',ready,'-inf',now),expired_leases=redis.call('ZCOUNT',processing,'-inf',now),
        counters=counters,timestamp=now,capacity=capacity_view()})
end
return redis.error_reply('UNKNOWN_JOB_OPERATION')
"""


def _positive(value: Any, name: str, maximum: float, *, zero: bool = False) -> float:
    if (type(value) not in (int, float) or value < 0 or (not zero and value == 0)
            or value > maximum or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if zero else 'positive'} number <= {maximum}")
    return float(value)


def _integer(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return value


def _text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be nonempty UTF-8 text of at most {maximum} bytes")
    return value


def _json(value: Any, maximum: int, *, size_error: type[ValueError] | None = None) -> str:
    def validate(item: Any) -> None:
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError("Only JSON values are accepted")
    try:
        validate(value)
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        size_bytes = len(text.encode("utf-8"))
        if size_bytes > maximum:
            if size_error is not None:
                raise size_error(size_bytes, maximum)
            raise ValueError("JSON value exceeds UTF-8 size limit")
        return text
    except (TypeError, RecursionError, OverflowError) as exc:
        raise ValueError("Unsupported JSON value") from exc


def _job(value: dict) -> dict:
    result = {key: item for key, item in value.items() if not key.startswith("_")}
    encoded_result = result["result_json"]
    # 为什么要有它：Redis cjson会把空数组重编码为空对象；业务JSON按原始字符串保存，才能保留payload/result/error的完整类型和值。
    for key in ("payload", "result", "error"):
        result[key] = json.loads(result.pop(key + "_json"))
    history = result["attempts_history"]
    if history == {}:
        history = []
    # Scope: decode without modifying retained input or sharing result objects.
    # 为什么要有它：兼容还原只能改变本次读取视图，顶层与attempt结果必须和旧格式一样独立。
    history = [dict(attempt) for attempt in history]
    for index, attempt in enumerate(history):
        if "_result_from_job" in attempt:
            marker = attempt.pop("_result_from_job")
            # Scope: exact last completed attempt of this job, never a generic missing-field fallback.
            # 为什么要有它：错误版本、错位引用和同时存在的结果不能借顶层内容伪装成有效历史记录。
            if (not isinstance(marker, dict) or set(marker) != {"version", "attempt"}
                    or type(marker.get("version")) is not int or marker["version"] != 1
                    or type(marker.get("attempt")) is not int or marker["attempt"] < 1
                    or type(result.get("attempt")) is not int or type(attempt.get("attempt")) is not int
                    or marker["attempt"] != result["attempt"] or marker["attempt"] != attempt["attempt"]
                    or index != len(history) - 1 or len(history) != result["attempt"]
                    or result.get("state") != "completed" or attempt.get("state") != "completed"
                    or "result_json" in attempt or result["result"] is None):
                raise ValueError("SERVICE_JOB_RESULT_REFERENCE_INVALID")
            attempt["result_json"] = encoded_result
        for key in ("result", "error"):
            attempt[key] = json.loads(attempt.pop(key + "_json"))
        for key in ("started_at", "finished_at"):
            if attempt.get(key) is not None:
                attempt[key] = float(attempt[key])
    result["attempts_history"] = history
    for key in ("created_at", "started_at", "finished_at", "available_at", "lease_until", "attempt_deadline_at"):
        if result.get(key) is not None:
            result[key] = float(result[key])
    return result


class RedisJobStore:
    """Single Redis deployment; max_pending bounds queued plus running jobs.

    为什么要有它：活动作业和幂等映射不设TTL；仅终态按保留期一起到期，排队或长任务不能悄悄变成可重复的新任务。
    """

    def __init__(self, client: Redis, namespace: str = "agent-service",
                 retention_seconds: float = 86400, max_pending: int = 1000, *,
                 capacity_bytes: int = CAPACITY_BYTES, cache_capacity_bytes: int = CACHE_CAPACITY_BYTES,
                 cache_ttl_seconds: int = 3600):
        self.client = client
        self.namespace = _text(namespace, "namespace", 256)
        self.retention_seconds = _positive(retention_seconds, "retention_seconds", 365 * 86400)
        self.max_pending = _integer(max_pending, "max_pending", 100000)
        self.capacity_bytes, self.cache_capacity_bytes = capacity_bytes, cache_capacity_bytes
        self.cache_ttl_seconds = _integer(cache_ttl_seconds, "cache_ttl_seconds", 365*86400)
        self.capacity_policy = policy_json(capacity_bytes, cache_capacity_bytes,
            max(1, math.ceil(self.retention_seconds*1000)), self.cache_ttl_seconds*1000)
        # 为什么要有它：命名空间、限流身份和幂等标识都只以摘要进入Redis键，键名不暴露问题文本或凭据。
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self.key_prefix = "agent-service:{" + digest + "}:"
        self._keys = [self.key_prefix + name for name in ("ready", "processing", "metrics", "capacity", "capacity-expiry")]
        self._script = client.register_script(_SCRIPT)

    def _call(self, operation: str, arguments: dict) -> dict:
        try:
            value = self._script(keys=self._keys, args=[operation, self.key_prefix,
                json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                max(1, math.ceil(self.retention_seconds * 1000)), self.max_pending,
                self.capacity_policy, self.namespace + ":query-cache:"])
        except ResponseError as error:
            # Scope: only exact fixed capacity codes become typed admission errors.
            # 为什么要有它：未知Redis错误可能发生在部分写之后，不能当成安全拒绝或自动重试；原错误与dirty留给owner核查。
            code = str(error).split(" ", 1)[0]
            if code in {"SERVICE_CAPACITY_EXCEEDED", "SERVICE_CACHE_CAPACITY_EXCEEDED", "SERVICE_CACHE_ENTRY_TOO_LARGE"}:
                raise CapacityExceeded(code) from error
            if code == "SERVICE_CAPACITY_POLICY_MISMATCH":
                raise CapacityPolicyMismatch(code) from error
            if code in {"SERVICE_CAPACITY_STATE_INVALID", "SERVICE_CAPACITY_DIRTY", "SERVICE_CAPACITY_LEGACY_NAMESPACE",
                        "SERVICE_CAPACITY_INITIALIZATION_SCAN_LIMIT", "SERVICE_CAPACITY_KEY_TYPE_INVALID", "SERVICE_CAPACITY_JOB_RESERVATION_INVALID",
                        "SERVICE_CAPACITY_JOB_RESERVATION_EXCEEDED"}:
                raise CapacityStateError(code) from error
            raise
        return json.loads(value)

    def cache_key(self, digest: str) -> str:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid cache digest")
        return self.key_prefix + "cache:" + digest

    def cache_get(self, digest: str):
        raw = self.client.get(self.cache_key(digest))
        # Scope: legacy cache is read-only. 为什么要有它：升级不删除旧缓存或迁移旧数据，任何新写仍须核准完整namespace策略。
        return self.client.get(self.namespace + ":query-cache:" + digest) if raw is None else raw

    def cache_put(self, digest: str, encoded: str) -> None:
        self.cache_key(digest)
        if not isinstance(encoded, str):
            raise ValueError("Encoded cache value must be text")
        if len(encoded.encode("utf-8")) > CACHE_ENTRY_BYTES:
            raise CapacityExceeded("SERVICE_CACHE_ENTRY_TOO_LARGE")
        self._call("cache_put", {"digest": digest, "encoded": encoded})

    def reclaim_expired_capacity(self) -> dict:
        """Explicit bounded maintenance; ordinary writes also reclaim at most 100."""
        return self._call("capacity_reclaim", {})

    @staticmethod
    def _id(job_id: str) -> str:
        if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            raise ValueError("Invalid job ID")
        return job_id

    @staticmethod
    def _token(token: str) -> str:
        return _text(token, "lease_token", 256)

    def submit(self, kind: str, payload: dict, idempotency_key: str,
               timeout_seconds: float, max_attempts: int) -> tuple[dict, bool]:
        if not isinstance(kind, str) or kind not in {"query", "agent"} or not isinstance(payload, dict):
            raise ValueError("kind must be query or agent and payload must be an object")
        key = _text(idempotency_key, "idempotency_key", 512)
        timeout = _positive(timeout_seconds, "timeout_seconds", 86400)
        attempts = _integer(max_attempts, "max_attempts", 10)
        body = _json(payload, _MAX_PAYLOAD_BYTES)
        canonical = _json({"kind": kind, "payload": payload, "timeout_seconds": timeout,
                           "max_attempts": attempts}, _MAX_PAYLOAD_BYTES + 1024)
        answer = self._call("submit", {"job_id": uuid4().hex, "kind": kind, "payload_json": body,
            "timeout_seconds": timeout, "max_attempts": attempts,
            "idempotency_digest": hashlib.sha256(key.encode("utf-8")).hexdigest(),
            "request_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()})
        if answer["code"] == "conflict":
            raise IdempotencyConflict(answer["job_id"])
        if answer["code"] == "full":
            raise QueueFull("Nonterminal job capacity reached")
        return _job(answer["job"]), answer["code"] == "created"

    def get(self, job_id: str) -> dict | None:
        raw = self.client.get(self.key_prefix + "job:" + self._id(job_id))
        return None if raw is None else _job(json.loads(raw))

    def claim(self, worker_id: str, lease_seconds: float) -> dict | None:
        # 为什么要有它：worker负责执行timeout并收尾；独立租约让仍持有所有权的worker能记录timed_out，同时拒绝真正失租的迟到结果。
        answer = self._call("claim", {"worker_id": _text(worker_id, "worker_id", 256),
            "lease_seconds": _positive(lease_seconds, "lease_seconds", 3600), "lease_token": uuid4().hex})
        return None if answer["code"] == "empty" else _job(answer["job"])

    def heartbeat(self, job_id: str, lease_token: str, lease_seconds: float) -> bool:
        return self._call("heartbeat", {"job_id": self._id(job_id), "lease_token": self._token(lease_token),
            "lease_seconds": _positive(lease_seconds, "lease_seconds", 3600)})["code"] == "accepted"

    def finish(self, job_id: str, lease_token: str, state: str,
               result: Any = None, error: Any = None) -> bool:
        if not isinstance(state, str) or state not in _TERMINAL:
            raise ValueError("Unsupported terminal state")
        # Scope: result byte limit only, before _call. 为什么要有它：仅这一typed异常能证明尚未发送终态写入，不能把一般校验错误或未知网络回复误当可安全再次发布。
        return self._call("finish", {"job_id": self._id(job_id), "lease_token": self._token(lease_token),
            "state": state, "result_json": _json(result, _MAX_RESULT_BYTES, size_error=ResultTooLargeError),
            "error_json": _json(error, _MAX_ERROR_BYTES)})["code"] == "accepted"

    def retry(self, job_id: str, lease_token: str, error: Any, delay_seconds: float) -> bool:
        # 为什么要有它：只有只读query可按次数重试；Agent可能已写文件，未知副作用必须由owner核对而非盲重跑。
        return self._call("retry", {"job_id": self._id(job_id), "lease_token": self._token(lease_token),
            "error_json": _json(error, _MAX_ERROR_BYTES),
            "delay_seconds": _positive(delay_seconds, "delay_seconds", 86400, zero=True)})["code"] == "accepted"

    def recover_expired(self) -> dict:
        """Recover at most 100 expired leases per call; retain unknown attempt costs."""
        return self._call("recover", {})

    def rate_limit(self, identity: str, limit: int, window_seconds: float) -> tuple[bool, float]:
        who = _text(identity, "identity", 1024)
        answer = self._call("rate_limit", {"identity_digest": hashlib.sha256(who.encode("utf-8")).hexdigest(),
            "limit": _integer(limit, "limit", 1000000),
            "window_ms": max(1, math.ceil(_positive(window_seconds, "window_seconds", 86400) * 1000))})
        return answer["allowed"], float(answer["retry_after"])

    def metrics(self) -> dict:
        answer = self._call("metrics", {})
        answer["timestamp"] = float(answer["timestamp"])
        answer["counters_scope"] = "Cumulative namespace events, including expired retained terminal records"
        answer["pending_scope"] = "Queued plus running; no terminal records counted toward admission capacity"
        return answer
