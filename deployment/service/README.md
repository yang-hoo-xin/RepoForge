# RAG / Agent 服务

FastAPI 受理任务，Redis 原子维护队列、幂等键、限流和租约，独立 worker 调用既有 `QueryService` 与正常自然语言 Agent 入口。每次尝试有独立预算、输入、业务结果和进程记录；符合条件的纯稀疏查询可复用有界常驻解释器，但每次仍创建新的 `QueryService`。Prometheus 从 API 收集 HTTP 与队列指标。

## 本机启动

当前容量版本不能直接接管已有、未登记容量的Redis数据。升级时先确认旧实例没有在途任务，使用明确的新Compose项目名、版本镜像、新数据卷及端口；保留旧实例、镜像和卷。不要只在旧Redis中更换命名空间，也不要运行旧版本读取器处理新格式数据。下面默认启动命令仅适用于全新开发实例；已部署实例的升级和回退需要按版本证据核对。为什么要有它：新准入规则会拒绝未计量旧数据，清空旧数据、混用版本或共享未计量的物理内存都不是兼容方案。

从仓库根目录创建 `deployment/service/.env`，设置两个不同的随机值：`SERVICE_API_TOKEN`、`SERVICE_REDIS_PASSWORD`。可参考 `.env.example`；已有 `.env` 不覆盖。凭据仅从环境注入，不提交或写入日志。为什么要有它：服务访问权限与队列凭据需要独立管理，部署镜像及实验原件不应携带密钥。

```powershell
docker compose --env-file deployment/service/.env -f deployment/service/compose.yaml up -d --build --scale worker=2
```

初始化容器会生成真实仓库源码的 BM25 / Chroma 示例语料，首次正常查询逐题核对来源。重启复用哈希相同的语料；源码变化时初始化明确报错，应选择新语料目录/卷并保留旧数据。为什么要有它：旧索引不能被静默当成新版本，初始化也不应删除历史证据。

默认地址：API `http://127.0.0.1:8080/docs`，Prometheus `http://127.0.0.1:9095`。Redis 仅发布到本机 `16379`，其他容器间连接使用项目网络。当前认证和配额面向一个受信任操作方，共享同一命名空间；不是多租户 RBAC。

停止服务并保留数据：

```powershell
docker compose --env-file deployment/service/.env -f deployment/service/compose.yaml down
```

持久数据在本项目的命名卷中；不使用 `down -v` 清理实验记录。为什么要有它：服务重启与删除历史数据是不同操作。

## Worker 执行模式与回退

范围：本 Compose 仅向 worker 注入 `SERVICE_QUERY_EXECUTION_MODE`，默认 `resident_sparse`；API、语料初始化和默认每分钟120次请求的配额不变。当前 worker 以 Python 直接作为 Linux 容器 PID 1 运行。只有这一环境、请求显式设置 `answer=false`、`rewrite=false`、`rerank=false`，且服务器配置关闭 dense/rewrite/rerank、dense provider 为 `disabled`、向量存储为 Chroma 时，才采用常驻解释器。Windows、非 PID 1 的 Linux、Agent 和其他查询继续走原逐尝试隔离子进程路径；加入 init/tini 使 Python 不再是 PID 1 时也会回退。为什么要有它：复用只覆盖已验证的无模型路径，不放宽其他任务的进程隔离。

范围：常驻模式只复用解释器及 Chroma 进程资源，不复用 `QueryService` 对象、请求记录或先前 job 的结果；每次仍走完整 `execute_request`、settings/hash、索引和缓存检查。为什么要有它：生命周期提速不能改变来源正确性、同 request_id 跨 job 的隔离或既有缓存语义。

范围：常驻生命周期中，每个 worker 最多保有一个 resident，最多处理128次请求后回收，空闲约60秒后由 worker 轮询清理；settings、collection、源码、BM25或数据库文件替换身份变化会重建，不适用请求先关闭闲置 resident 再走隔离路径。超时、失租、协议或业务异常会销毁该 worker 自有进程组；清理未确认时保留句柄并阻止启动第二个 resident。每次尝试有独立 stdout/stderr/Trace；`request_completed` 和 `request_exit_code` 表示请求完成，不把仍存活的进程伪报为 OS 退出0。父死亡保护限定于上述 Linux PID 1 容器边界，不宣称通用于所有 Linux 运行方式。为什么要有它：复用资源仍需有界回收、准确的完成语义和不可跨尝试污染的证据。

回退配置为：

```dotenv
SERVICE_QUERY_EXECUTION_MODE=isolated
```

范围：将模式变更应用到现有部署时，先等待活动尝试结束，再仅重建 worker 并保持原实例数、命名空间、卷和其他配置；本配置文件更新不会自动修改已运行的容器。为什么要有它：模式回退不应中断未知副作用、删除历史证据，或同时改变其他性能因素。

## HTTP 合同

业务接口使用 `Authorization: Bearer <SERVICE_API_TOKEN>`。提交时另带 `Idempotency-Key`（最多128个可打印ASCII字符）。

| 接口 | 用途 |
|---|---|
| `POST /v1/jobs` | 新任务返回202；相同幂等键及绑定请求重放返回200和同一job id |
| `GET /v1/jobs/{id}` | 排队、尝试次数、终态、公开业务结果 |
| `GET /v1/features/{id}/{view}` | 配置 ArtifactStore 后复用原只读 artifacts/status/dashboard 接口 |
| `GET /health` | API 与 Redis 连通性 |
| `GET /metrics` | HTTP 计数/延迟、Redis队列及生命周期累计事件 |

同一幂等键绑定不同请求或执行配置返回409；限流/容量返回429及 `Retry-After`；Redis不可用返回503。请求体在JSON解析前限制大小和读取时间，参数拒绝未知字段、非法类型及非有限数字。为什么要有它：重放不能生成第二个任务，坏输入也不能进入执行队列。

示例检索请求：

```json
{
  "kind": "query",
  "payload": {
    "query": "RedisJobStore IdempotencyConflict",
    "collection": "service_load",
    "top_k": 5,
    "answer": false,
    "rewrite": false,
    "rerank": false
  },
  "timeout_seconds": 60,
  "max_attempts": 2
}
```

`timeout_seconds` 是每次尝试从worker领取开始的执行预算，包含子进程启动；排队时间另记。`max_attempts`最多3。HTTP客户端停止轮询不等于服务端取消任务。活动作业和幂等记录保留到终态后才开始保留期TTL。

`completed` 表示后端返回了结构化结果，必须继续检查领域结果。Agent的澄清、待批准、partial和实际交付各自保留，不能凭HTTP200或队列completed声明代码交付。为什么要有它：基础设施完成与用户需求满足属于不同层次。

## Agent 注册

服务器通过 `SERVICE_AGENT_PROFILES` 注册 `profile_id → 绝对profile路径`，客户端只提交id和需求：

```json
{"kind":"agent","payload":{"profile_id":"registered_project","description":"修改这个已登记工程的需求"},"timeout_seconds":600,"max_attempts":1}
```

API入队绑定profile字节，worker核验相同内容后调用既有 `dsh_request_routing.run_natural_request`。profile必须符合产品入口原有授权与执行约束；服务不接受客户端提供执行器或任意本地路径。失租或执行结果不明的Agent进入 `indeterminate`，不自动重跑。为什么要有它：不确定的源码或工具副作用不能通过盲重试重复执行。

默认Compose演示运行检索，Agent注册表为空。Windows/.NET/GTFO回归profile由Windows本机的同一API和worker模块执行，使用独立Redis命名空间与工作目录；Linux容器只能执行与其文件系统、工具和产品入口兼容的profile。示例本机启动入口：

```powershell
# 先在该进程环境设置SERVICE_API_TOKEN、SERVICE_REDIS_URL、SERVICE_NAMESPACE、
# SERVICE_SETTINGS、SERVICE_WORK_ROOT、SERVICE_AGENT_PROFILES。
.venv\Scripts\python.exe -B -m uvicorn src.service_layer.api:create_app --factory --host 127.0.0.1 --port 8082
.venv\Scripts\python.exe -B -m src.service_layer.worker
```

同一队列的API和worker必须部署一致的注册配置及可访问数据。为什么要有它：跨机器路径不可自动互换，队列中的配置哈希不能替代实际执行依赖。

## 超时、恢复与缓存

### 编码数据配额与在途预留（2026-10-16）

范围：同一服务命名空间默认总额度64MiB、缓存子额度16MiB；8192字节固定元数据预占不随TTL归零。提交时按payload、最多尝试数、最大结果/错误及转义上界预留整个生命周期；完成后按实际编码和有限元数据结算。缓存与限流共用总额度，单缓存记录最多2MiB；续租和结束使用已接任务的预留，不与新任务重新竞争。为什么要有它：终态任务和缓存不受max_pending限制，只限制排队长度不能保证在途任务有空间保存结果。

范围：这是逻辑编码数据预算，不是Redis内存或容器RSS上限。Compose另显式使用128MiB `maxmemory`、`noeviction`及256MiB容器限制；它们不能保证物理OOM下所有Lua仍可执行，也不覆盖同Redis的外部写入、复制或AOF缓冲。为什么要有它：编码长度、Redis对象开销和进程内存不同，物理余量与长期稳定性仍需实际测量。

范围：API/Worker/查询子进程必须使用一致容量策略、保留期及缓存TTL。新命名空间仅在有界完整扫描确认没有旧job/cache等键后启用；大共享数据库扫描未完成也明确拒绝。TTL每次最多核对100项，实际数据与索引均消失后才释放额度；未知中途写入保留未完成标记和保守占用，不自动清零。容量不足返回429，策略冲突/旧数据/未完成写入返回503；GET仍可核查原记录。为什么要有它：不能把Lua原子执行误当失败回滚，也不能用重启或重复提交绕过未知副作用。

[真实Redis/API验证](../../artifacts/eval_driven_20261016/service-capacity/candidate-02/result.json)55/55通过、零跳过，含并发最后一份额度、缓存竞争后的十次尝试与结果保存、TTL及ACL中途拒写；14.485秒，容器退出0且未OOM。首轮53通过、2个百任务fixture在准备时被新额度拒绝；只为这两项显式配置更高合成额度后复验，生产默认和原断言未变。该结果不等于版本部署、百并发、模型负载或长期稳定性已通过。

租约恢复错误已改用 `service-lease-expired-error.v2` 同任务历史引用，旧错误保留，不再每次复制转义正文。[上一轮真实检查](../../artifacts/eval_driven_20261015/service-bottleneck/memory-analysis/recovery-error-growth/candidate-v1/result.json)48/48通过；新顶层错误结构是明确的v2变化，不宣称与旧嵌套JSON逐字一致。

### 终态结果的内部保存格式

范围：新 writer 仅对 `completed` 且结果非 `null` 的 job 去掉最后 attempt 中重复的 `result_json`，顶层原始 JSON 保留，最后 attempt 改存私有 `_result_from_job={version:1,attempt:当前编号}`。读取器严格核验版本、最后位置、任务与尝试编号、成功状态和无冲突字段，再独立还原原有公开及内部返回；旧无标记记录照读，不迁移旧 Redis/AOF，历史失败、错误、租约、TTL 和统计不变。为什么要有它：减少一份已确认同值的结果，不能借去重删历史或改变 API 行为。

范围：API 与所有 worker 必须作为同版本整套升级，确认旧 reader 全部停止后才能启用新 writer；不能把该格式宣称为与旧版本滚动兼容。回退旧 reader 前仍需使用兼容读取的新版本或另行制定数据转换方案，本改动不自动回写历史。为什么要有它：旧 reader 强制读取每个 attempt 的 `result_json`，无法解析新引用；部署版本混用会使已有任务不可读取。

范围：此去重只减少终态 job 的重复结果编码，不替代总容量预算，也不清理或淘汰活动 job、幂等记录、查询缓存和 AOF。实际字节节省、正常 Redis/API 兼容和低内存行为需分别验证，不能把投影字节差当成 RSS 节省。为什么要有它：更小的单条记录仍会随任务量和保留期增长，且 Redis 内存及持久化开销不等于 JSON 长度。

2026-10-15 [真实Redis存储对照](../../artifacts/eval_driven_20261015/service-bottleneck/memory-analysis/result-compaction-runtime-v1/attempt-v3/result.json)使用相同2,293份历史输入与结果，逐项确认任务和尝试结果一致。任务键 `STRLEN` 总和174,828,517→88,863,918字节，`MEMORY USAGE` 总和198,336,712→99,360,968字节（减少49.90%）；实际Redis与ASGI集成33/33、零跳过，保留24小时TTL。该对照未重新执行业务或模型，不证明RSS、用户等待、费用或长期稳定性收益。测试容器已退出0，原部署及AOF未改。

最终完整查询结果缓存的 `SET` 遇到Redis连接、超时或服务端拒绝时，现在交给原查询层处理并记录 `complete_result_cache_write_failed`，不把缓存成功或命中写入结果；编程错误不被吞掉。18项受控异常测试通过，包含正常 `execute_request` 调用链。范围仅为最后的完整结果缓存写入，不代表初始读取、中间缓存或任务终态发布已验证低内存容错。为什么要有它：可重建缓存故障不应抹掉已完成检索，但任务状态存储与缓存有不同的可靠性要求。

完整结果超过原2MiB限制时，Store仅对合法JSON字节超限抛出“尚未发送Redis写入”的专用异常；Worker保留原结果文件，按原租约发布一次小型 `SERVICE_RESULT_TOO_LARGE` 服务失败说明，明确失败阶段为结果发布。`backend_returned=true`不意味着领域任务成功，未知网络回复仍不补发，也不自动重跑业务。43项聚焦检查通过，但该超限分支尚未运行完整真实Redis Worker场景。为什么要有它：服务无法发布结果和业务本身失败应分开说明，原件不能被截断或丢失。

[低内存实测](../../artifacts/eval_driven_20261015/service-bottleneck/memory-analysis/redis-low-memory-v1/attempt-v1/result.json)在独立Redis7.4.11、`noeviction`下观察8个串行小案例：提交/重复提交/续租/缓存写入被拒，读取仍可用，完成/重排/恢复的部分Lua仍能在超过`maxmemory`后执行；本次索引均一致。尚未把内存阈值配置应用到原部署；准入背压、在途预留、并发和持续容量仍待验证。为什么要有它：不淘汰键不等于所有在途操作可继续，配置阈值也不是进程RSS或Lua写入的严格上限。

Worker 收到 `SIGTERM` / `SIGINT` 后停止领取任务，打断空闲轮询或故障退避，并立即通过现有执行器取消正在运行的本机子进程。这不是等待任务自然完成；Query 的已确认取消记为 `cancelled`，已启动 Agent 的执行结果记为 `indeterminate`，不自动重跑。信号到达前已完成的有效结果仍经过原租约 token 检查。为什么要有它：停止服务不能冒称业务完成，也不能因重启重复源码或工具副作用。

范围为 Worker 自有连接和进程：Redis URL 解析后固定连接/读取超时各 3 秒、SDK 自动重试为零；未知写入回复不重发，队列轮询仍使用原外层退避。停止时心跳线程总共最多等待 1 秒，未退出按失租处理；随后尝试关闭 resident、Redis 并恢复信号处理器，某一步失败不跳过其余清理。为什么要有它：URL 参数不能隐式扩大等待或重发 claim/finish，旧 owner 不能发布不确定租约下的结果。

Compose 的 Worker 硬停止等待改为 45 秒，供执行器的进程清理等待（异常路径可累计约 20 秒）、最后一次 resident 清理（最多一次 10 秒等待）、心跳退出及 Redis I/O 使用；不是任意内核、文件系统、DNS 或网络故障下的正常退出保证。超过该窗口仍可能被容器强制终止，未确认结果保留未知；外部模型服务器是否停止不由客户端取消证明。为什么要有它：原 15 秒无法覆盖已知叠加路径，配置预算不能替代真实容器停止验证。

worker续租并用租约token提交结果，过期owner不能覆盖新owner。进程取消清理Windows Job / POSIX进程组；关闭客户端进程并不证明外部模型服务器已经停止。短暂Redis连接故障采用退避，永久认证/配置错误明确失败。

Query只对已确认尚未执行的可恢复缓存连接故障，或失租后的只读任务在额度内重试；全部attempt保留。Agent禁止自动重跑。为什么要有它：恢复需遵守原执行副作用边界和重试额度。

Redis结果缓存复用业务层的配置、源码、索引和模型身份。纯稀疏路径绑定实际 Chroma 正文/metadata 的稳定语义哈希，避免旧正文命中和跨进程SQLite文件抖动。此检查包含全量读取/哈希成本。模型身份未知的生成路径不会凭空启用缓存。

## 可重算验证

真实HTTP烟测：`scripts/service_load/http_smoke.py`。异常联调：`runtime_faults.py`。负载和容器采样见 [压测说明](../../scripts/service_load/README.md)，五组编排入口为 `run_matrix.py`。

已登记部署的只读身份检查使用 `python scripts/service_load/deployment_identity.py --receipt <原start-result.json> --receipt-sha256 <原件SHA256> --output <新结果文件>`。它核对实际项目／服务标签、角色数量、镜像ID、命名卷与挂载位置、初始化退出及运行／健康状态；容器重建后的新ID允许，重复ID或缺失角色不允许。输出父目录须已存在，旧记录不覆盖。首次启动脚本也复用同一检查器，不再读取完整Docker环境或健康日志。为什么要有它：首次发布通过不能替代后续重复启动的版本核验，可变tag、错误卷或漏启动Worker需要能被发现。

身份检查退出0仅表示所查身份和进程状态符合原记录，退出2表示不一致，退出3表示Docker不可查询；不证明卷内容、HTTP业务、数据恢复或整个部署验收通过，也不执行任何重启、停止或迁移。真实重启验收需另保留前后身份检查及下述有效窗口内的任务重放。为什么要有它：只读检查不能被当成实际重启，环境不可用也不能伪装成版本正确。

重启前后幂等检查复用 `scripts/service_load/restart_replay.py`，它不负责重启容器。先核准实际部署的 `SERVICE_RETENTION_SECONDS`（当前默认86400），在 `prepare --url ... --receipt ... --retention-seconds 86400` 保存一份新记录；外部完成同版本、同卷的受控重启后，用 `verify --url ... --receipt ... --output ...` 核对。API凭据仍只从环境读取，旧记录及输出不覆盖。为什么要有它：检查对象必须是这次重启前真实完成的任务，不能用多天前的样本或新任务冒充持久化恢复。

验证窗口根据调用者声明的部署保留期与原服务返回的 `finished_at` 计算，要求服务与观察端时钟可比较；不是实际Redis删除或TTL观测。已超过窗口、缺保留期/终态时间或剩余窗口不足以完成请求时，输出 `outcome=inconclusive`、退出码3，不重放POST，也不算恢复成功或数据丢失。窗口内记录缺失仍失败；完整核对成功退出0，其他失败退出2。旧记录可在核准原部署策略后显式补传 `--retention-seconds`，不会改写原文件；新记录已保存的期限不能在验证时延长。为什么要有它：TTL自然过期与重启故障不能混为一谈，等待环境恢复也不能成为重领任务或延长保留期的理由。

每次响应返回后也核对TTL、单请求和总观察预算；跨窗响应保留实际请求次数及最后响应，再报告无法判定或超时，不把已发生的POST改写为未执行。网络socket超时不是整个响应的硬进程截止，本脚本保证拒绝使用晚到观察作通过依据，不承诺强制中断任意阻塞网络调用。为什么要有它：请求前尚未到期不能保证响应返回仍有效，观察工具的时间保证也必须与实际实现一致。

2026-10-15 的停止修复已通过56项worker测试及11项常驻进程相关回归。[空闲及连接退避实测](../../artifacts/eval_driven_20261015/service-bottleneck/shutdown-qualification/run-normal-v2/result.json)与[在途故障注入](../../artifacts/eval_driven_20261015/service-bottleneck/shutdown-qualification/run-active-v2/result.json)均使用原10秒停止窗口，四个worker退出0且PID为0。在途查询保留取消状态，Agent保留结果不确定，未自动重试或领取哨兵任务。这是实际worker.main、Redis和自有子进程的停止验证，不是实际模型交付，也不证明模型服务器已取消。首配置缺项失败及追加约20秒执行成本保留；测试Redis清理退出137另记，不能据此宣布整个服务已具备完整优雅停机或长期稳定性。

吞吐以完成的不同job计数，P95注明样本量、nearest-rank算法及完整窗口；提交延迟、任务端到端延迟、排队和执行分别统计。容器资源只包含显式指定的本项目服务，不含外部模型或宿主总资源。为什么要有它：202、重复返回或提前失败都不能被包装成任务吞吐或提速。
