"""Logical encoded-byte admission for the existing service Redis script, not RSS."""
from __future__ import annotations

import json

CAPACITY_BYTES = 64 * 1024**2
CACHE_CAPACITY_BYTES = 16 * 1024**2
CACHE_ENTRY_BYTES = 2 * 1024**2
BASE_BYTES = 8192
ITEM_METADATA_BYTES = 1024
JOB_METADATA_BYTES = 8192
ATTEMPT_METADATA_BYTES = 4096
VERSION = "service-capacity.v1"


class CapacityExceeded(RuntimeError):  # noqa: N818 - stable service admission exception name
    def __init__(self, code="SERVICE_CAPACITY_EXCEEDED"):
        self.code = code
        super().__init__(code)


class CapacityPolicyMismatch(RuntimeError):  # noqa: N818 - stable service admission exception name
    def __init__(self, code="SERVICE_CAPACITY_POLICY_MISMATCH"):
        self.code = code
        super().__init__(code)


class CapacityStateError(RuntimeError):
    def __init__(self, code="SERVICE_CAPACITY_STATE_INVALID"):
        self.code = code
        super().__init__(code)


def policy_json(capacity_bytes, cache_capacity_bytes, retention_ms, cache_ttl_ms):
    for value in (capacity_bytes, cache_capacity_bytes):
        if type(value) is not int or not 1 <= value <= 2**40:
            raise ValueError("Capacity bytes must be positive bounded integers")
    if capacity_bytes < BASE_BYTES or cache_capacity_bytes > capacity_bytes:
        raise ValueError("Capacity must cover fixed metadata and the cache sub-budget")
    return json.dumps({"version": VERSION, "capacity_bytes": capacity_bytes,
        "cache_capacity_bytes": cache_capacity_bytes, "retention_ms": retention_ms,
        "cache_ttl_ms": cache_ttl_ms, "cache_entry_bytes": CACHE_ENTRY_BYTES}, sort_keys=True, separators=(",", ":"))


def reservation_bytes(payload_bytes, attempts):
    # Scope: valid new jobs only, with Python payload/result/error and attempt limits.
    # 为什么要有它：两份非completed结果各可2MiB，外层字符串再次转义；每次原错误保留，worker_id每UTF8字节最多6字节，不能用平均样本估计生命周期上界。
    if type(payload_bytes) is not int or not 0 <= payload_bytes <= 128*1024 or type(attempts) is not int or not 1 <= attempts <= 10:
        raise ValueError("Invalid lifecycle reservation input")
    # Top-level 8KiB covers identifiers, timestamps, queue/idem and ledger metadata;
    # each 4KiB covers its <=1536-byte escaped worker id plus fixed attempt fields.
    return 2*payload_bytes + 4*(2*1024**2) + 2*(attempts+1)*(64*1024) + JOB_METADATA_BYTES + attempts*ATTEMPT_METADATA_BYTES


LUA = r"""
local quota, expiry = KEYS[4], KEYS[5]
local policy_raw, legacy_prefix = ARGV[6], ARGV[7]
local policy = cjson.decode(policy_raw)
local BASE, ITEM = 8192, 1024
local writes, planned = {}, {}
local function cap_error(code) error(code, 0) end
local function write(...) table.insert(writes, {...}) end
-- Scope: logical quotas fit 13 decimal digits; expiry is a separate epoch-ms domain.
-- 为什么要有它：cjson默认14位有效数比Lua double精度更小，不能把2^53当成持久JSON无损保证。
local function valid_integer(value,maximum) return type(value)=='number' and value>=0 and value<=(maximum or 1099511627776) and value%1==0 end
local function charge(field)
    local raw=redis.call('HGET',quota,field)
    if not raw then return nil end
    if #raw>512 then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    local ok,row=pcall(cjson.decode,raw)
    if not ok or type(row)~='table' or not valid_integer(row.bytes) or row.bytes<ITEM
        or not valid_integer(row.expires,9999999999999) or type(row.id)~='string'
        or (row.kind~='job' and row.kind~='cache' and row.kind~='rate') then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    local tag=row.kind=='job' and 'j:' or row.kind=='cache' and 'c:' or 'r:'
    if field~=tag..row.id or #row.id~=(row.kind=='job' and 32 or 64)
        or not string.match(row.id,'^[0-9a-f]+$') then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    -- Scope: charge records have only bounded internal fields covered by ITEM metadata.
    -- 为什么要有它：账面额度不能忽略可增长的附加字段，损坏记录也不能指定任意释放对象。
    local expected={kind=true,id=true,bytes=true,expires=true}
    if row.kind=='job' then
        expected.idem=true; expected.reserved=true
        if type(row.idem)~='string' or #row.idem~=64 or not string.match(row.idem,'^[0-9a-f]+$')
            or not valid_integer(row.reserved) or row.reserved<row.bytes then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    end
    local count=0
    for key,_ in pairs(row) do
        if not expected[key] then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
        count=count+1
    end
    if count~=(row.kind=='job' and 6 or 4) then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    return row
end
local function capacity_view()
    local saved=redis.call('HGET',quota,'policy')
    local configured=policy
    if saved then
        local ok,value=pcall(cjson.decode,saved)
        if not ok or type(value)~='table' or value.version~='service-capacity.v1'
            or not valid_integer(value.capacity_bytes) or value.capacity_bytes<BASE
            or not valid_integer(value.cache_capacity_bytes) or value.cache_capacity_bytes<1
            or value.cache_capacity_bytes>value.capacity_bytes then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
        configured=value
    end
    return {capacity_bytes=configured.capacity_bytes,cache_capacity_bytes=configured.cache_capacity_bytes,
        charged_bytes=tonumber(redis.call('HGET',quota,'used') or '0'),
        cache_charged_bytes=tonumber(redis.call('HGET',quota,'cache_used') or '0'),
        dirty=(redis.call('HGET',quota,'dirty') or '')~='',initialized=saved~=false,
        accounting_version='service-capacity.v1'}
end
local function capacity_guard()
    local saved=redis.call('HGET',quota,'policy')
    if saved and saved~=policy_raw then cap_error('SERVICE_CAPACITY_POLICY_MISMATCH') end
    if not saved then
        -- Scope: empty namespaces only; at most 16 SCAN pages / 4096 returned names.
        -- 为什么要有它：空页不代表扫描完成，也不能只查ready漏掉旧终态或旧cache；大共享DB可明确拒绝，不能无限遍历或默默清零。
        local cursor,seen='0',0
        for page=1,16 do
            local batch=redis.call('SCAN',cursor,'COUNT',100)
            cursor=batch[1]; seen=seen+#batch[2]
            if seen>4096 then cap_error('SERVICE_CAPACITY_INITIALIZATION_SCAN_LIMIT') end
            for _,key in ipairs(batch[2]) do
                if string.sub(key,1,#prefix)==prefix or string.sub(key,1,#legacy_prefix)==legacy_prefix then
                    cap_error('SERVICE_CAPACITY_LEGACY_NAMESPACE')
                end
            end
            if cursor=='0' then break end
        end
        if cursor~='0' then cap_error('SERVICE_CAPACITY_INITIALIZATION_SCAN_LIMIT') end
        redis.call('HSET',quota,'policy',policy_raw,'used',BASE,'cache_used',0,'dirty','')
    end
    if redis.call('HGET',quota,'dirty')~='' then cap_error('SERVICE_CAPACITY_DIRTY') end
    local used,cached=tonumber(redis.call('HGET',quota,'used')),tonumber(redis.call('HGET',quota,'cache_used'))
    if not valid_integer(used) or not valid_integer(cached) or used<BASE or cached>used
        or used>policy.capacity_bytes or cached>policy.cache_capacity_bytes then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    return used,cached
end
local function capacity_cleanup(used,cached)
    local releases,total,cache_total={},0,0
    local ids=redis.call('ZRANGEBYSCORE',expiry,'-inf',now_ms,'LIMIT',0,100)
    -- Scope: confirm all selected expired allocations before releasing any of them.
    -- 为什么要有它：TTL索引可能早于实际过期，job和idem必须都消失；不按时间猜测，也不在坏的第100项前先释放99项。
    for _,field in ipairs(ids) do
        local row=charge(field)
        if not row then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
        local absent=false
        if row.expires>0 and row.expires<=now_ms then
            if row.kind=='job' then
                if type(row.idem)~='string' then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
                absent=redis.call('EXISTS',prefix..'job:'..row.id,prefix..'idem:'..row.idem)==0
                    and not redis.call('ZSCORE',ready,row.id) and not redis.call('ZSCORE',processing,row.id)
            else absent=redis.call('EXISTS',prefix..row.kind..':'..row.id)==0 end
        end
        if absent then
            table.insert(releases,field); total=total+row.bytes
            if row.kind=='cache' then cache_total=cache_total+row.bytes end
        end
    end
    if used-total<BASE or cached-cache_total<0 then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    if #releases>0 then
        -- Scope: physical write failure is not a rollback. 为什么要有它：先保留旧额度和dirty；只有全部删除成功才释放，否则所有后续写拒绝，需要明确人工修复。
        redis.call('HSET',quota,'dirty','reclaim','used',used,'cache_used',cached)
        for _,field in ipairs(releases) do
            redis.call('HDEL',quota,field); redis.call('ZREM',expiry,field)
        end
        used,cached=used-total,cached-cache_total
        redis.call('HSET',quota,'used',used,'cache_used',cached,'dirty','')
    end
    return used,cached,#releases
end
local function capacity_job(job,encoded,terminal,created)
    local field='j:'..job.id
    local row=charge(field)
    local reserve=2*#job.payload_json+8388608+2*(job.max_attempts+1)*65536+8192+job.max_attempts*4096
    if created then
        if row then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    elseif not row or row.kind~='job' or row.idem~=job._idempotency_digest or row.reserved~=reserve
        or row.expires~=0 or row.bytes~=reserve then cap_error('SERVICE_CAPACITY_JOB_RESERVATION_INVALID') end
    local actual=#encoded+#(prefix..'job:'..job.id)+#(prefix..'idem:'..job._idempotency_digest)
        +#cjson.encode({job_id=job.id,request_hash=job._request_hash})+ITEM
    -- Scope: every proposed job write, including all 100 recover candidates, is encoded before commit.
    -- 为什么要有它：生命周期预留是可核查上界，不是允许超限写入的估计；坏数据不能在前面若干任务已改后才被发现。
    if actual>reserve then cap_error('SERVICE_CAPACITY_JOB_RESERVATION_EXCEEDED') end
    planned[field]={kind='job',id=job.id,idem=job._idempotency_digest,reserved=reserve,
        bytes=terminal and actual or reserve,expires=terminal and (now_ms+retention) or 0}
end
local function capacity_item(kind,id,encoded,ttl)
    local field=(kind=='cache' and 'c:' or 'r:')..id
    planned[field]={kind=kind,id=id,bytes=#encoded+#(prefix..kind..':'..id)+ITEM,expires=now_ms+ttl}
end
local function validate_writes()
    local checked,increments={},{}
    for _,command in ipairs(writes) do
        local name,key=command[1],command[2]
        local expected=(name=='ZADD' or name=='ZREM') and 'zset' or name=='HINCRBY' and 'hash' or 'string'
        if checked[key] and checked[key]~=expected then cap_error('SERVICE_CAPACITY_KEY_TYPE_INVALID') end
        if not checked[key] then
            local actual=redis.call('TYPE',key).ok
            if actual~='none' and actual~=expected then cap_error('SERVICE_CAPACITY_KEY_TYPE_INVALID') end
            checked[key]=expected
        end
        if name=='HINCRBY' then
            local field=command[3]
            local value=increments[field]
            if value==nil then
                local raw=redis.call('HGET',key,field) or '0'
                value=tonumber(raw)
                if not valid_integer(value) or string.format('%.0f',value)~=raw then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
            end
            if not valid_integer(value) or value>=1099511627776 then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
            increments[field]=value+1
        end
    end
end
local function capacity_commit()
    -- Scope: predictable key/type/counter errors precede any business/control mutation.
    -- 为什么要有它：已知坏数据必须先拒，dirty只表示真实写入过程中未完成，不能为测试制造本可避免的半写。
    validate_writes()
    local used,cached=capacity_guard()
    local released
    used,cached,released=capacity_cleanup(used,cached)
    local peak,peak_cache,settled,settled_cache=used,cached,used,cached
    local fields,final={'dirty',op,'used',used,'cache_used',cached},{'dirty',''}
    for field,row in pairs(planned) do
        local old=charge(field); local before=old and old.bytes or 0
        if old and old.kind~=row.kind then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
        local maximum=math.max(before,row.bytes)
        peak=peak+maximum-before; settled=settled+row.bytes-before
        if row.kind=='cache' then peak_cache=peak_cache+maximum-before; settled_cache=settled_cache+row.bytes-before end
        local conservative={}
        for key,value in pairs(row) do conservative[key]=value end
        conservative.bytes=maximum
        table.insert(fields,field); table.insert(fields,cjson.encode(conservative))
        table.insert(final,field); table.insert(final,cjson.encode(row))
    end
    if peak>policy.capacity_bytes then cap_error('SERVICE_CAPACITY_EXCEEDED') end
    if peak_cache>policy.cache_capacity_bytes then cap_error('SERVICE_CACHE_CAPACITY_EXCEEDED') end
    if settled<BASE or settled_cache<0 or settled_cache>settled then cap_error('SERVICE_CAPACITY_STATE_INVALID') end
    if #writes>0 or next(planned) then
        fields[4],fields[6]=peak,peak_cache
        -- Scope: all business mutations in this invocation share one preceding control HSET.
        -- 为什么要有它：额度先保守占用且dirty与之同写；物理OOM/WRONGTYPE/未知回复不能被叙述为Lua事务回滚。
        redis.call('HSET',quota,unpack(fields))
        for _,command in ipairs(writes) do redis.call(unpack(command)) end
        for field,row in pairs(planned) do
            if row.expires>0 then redis.call('ZADD',expiry,row.expires,field)
            else redis.call('ZREM',expiry,field) end
        end
        table.insert(final,'used'); table.insert(final,settled)
        table.insert(final,'cache_used'); table.insert(final,settled_cache)
        redis.call('HSET',quota,unpack(final))
    end
    return released
end
local function respond(value)
    local encoded=cjson.encode(value)
    capacity_commit()
    return encoded
end
"""
