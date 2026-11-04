# RepoForge 文档

RepoForge 将自然语言需求推进到有证据、可验收、可授权应用的代码变更。这是项目文档的统一入口：正文保存在仓库，任务和 PR 在 GitHub 管理，索引关联来源版本及验证结果。

**查找资料：** [可搜索总览](index.html) · [索引数据](document-index.json) · [GitHub 任务](https://github.com/yang-hoo-xin/RepoForge/issues) · [Pull requests](https://github.com/yang-hoo-xin/RepoForge/pulls)。在 GitHub 阅读源码时，可在本地打开 `docs/index.html` 使用搜索，或直接阅读下列 Markdown 入口。

**维护资料：** [存放与清理约定](document-management.md) · [登记表](document-registry.json)。索引中的 GitHub 状态带读取时间；分支文档迁入不代表相应功能代码已经合入。

## 当前工作线

| 工作线 | 正文与记录 | 状态边界 |
| --- | --- | --- |
| 竞争性评测 T1—T11 | [原冻结规格](specs/competitive-delivery-evaluation.md) · [父 Issue #3](https://github.com/yang-hoo-xin/RepoForge/issues/3) | 原协议与后续开发试验分别记录，不以准备或单题结果替代批次完成 |
| 开发者试用版 B | [产品规格](specs/developer-preview-b.md) · [父任务 #17](https://github.com/yang-hoo-xin/RepoForge/issues/17) | B1—B5 任务按 GitHub 记录；[B4 #21](https://github.com/yang-hoo-xin/RepoForge/issues/21) 是安装后完整流程，B5 才是外部采用 |
| 本地模型候选编辑与验收接线 | [接口与验证记录](research/candidate-editing-20261103.md) | 关联 S1/S2/S3 及原分支；工程检查与模型结果分别记录 |
| B1 费用恢复修复 | [独立审查](research/b1-spending-review-20261103.md) | 来源 `21cb246`，实现保留在独立分支 |
| 编辑权限 | [ADR-0005](adr/0005-task-permission-scope-and-file-creation.md) | 设计接受不等于权限已实施或自动授予 |
| HTML 分析 | [报告来源](reports/sources.json) | 历史分析材料，按来源与版本查阅 |
| 文档管理与草稿生命周期 | [任务 #23](https://github.com/yang-hoo-xin/RepoForge/issues/23) · [草稿 PR #16](https://github.com/yang-hoo-xin/RepoForge/pull/16) | 独立分支已公开，主线合入由审阅决定 |

原工作区中的状态、计划及分析副本用于追溯；新的正式资料按[维护约定](document-management.md)登记。为什么要有它：一个入口应说明各项工作的实际归属，而不要求读者从聊天标题或相同目录名猜测。

## 了解项目与实测结果

| 阅读目标 | 入口 | 内容 |
| --- | --- | --- |
| 快速了解产品 | [项目首页](../README.md) | 目标、核心能力、工作流与快速开始 |
| 查看已完成验证 | [实测报告](PUBLIC_RESULTS.md) | 服务化、真实 Agent 入口、模块 EDD、受控代码交付与绑定版本的 CI 结果 |
| 核对服务测量数字 | [负载数值摘录](evidence/service-load-20261009.json) | 五组条件、完整分母、结果与原摘要 SHA-256 |
| 理解交付目标 | [产品契约](ACTIVE_PRODUCT_CONTRACT.md) | 新建、修改、继续修复，以及验收与执行边界 |

## 运行与接入

| 阅读目标 | 入口 | 内容 |
| --- | --- | --- |
| 启动服务演示 | [服务部署与 HTTP 接口](../deployment/service/README.md) | Compose、认证、异步作业、Agent profile 与停止方式 |
| 理解 HTTP 到业务的调用 | [服务业务接线](../src/service_layer/backends.py) | 查询服务复用、登记 profile 和自然语言 Agent 入口 |
| 使用命令行入口 | [CLI 实现](../src/mod_agent/cli.py)、[自然语言路由](../scripts/dsh_request_routing.py) | 命令参数与新建、修改、续接、澄清的路由 |
| 配置补丁续接 | [Patch 上游续接](PATCH_UPSTREAM_CONTINUATION.md) | Evidence / Planning 配置和原 worker 的接入方式 |
| 查看模型选择机制 | [模型资格路由](MODEL_QUALIFIED_ROUTING.md) | 可选角色选择、身份绑定与显式配置关系 |

## 理解架构与评测

| 阅读目标 | 入口 | 内容 |
| --- | --- | --- |
| 了解工作流与职责 | [架构说明](ARCHITECTURE_OVERVIEW.md)、[LangGraph 图](../src/mod_agent/workflow/graph.py) | 控制流、领域边界与具体实现；架构文档中的里程碑保留其原始阶段语境 |
| 查看产物持久化 | [ArtifactStore](../src/mod_agent/artifacts/store.py) | 业务产物、身份与谱系的保存入口 |
| 理解成功与成本计数 | [评测驱动开发](EVALUATION_DRIVEN_DEVELOPMENT.md) | 补丁可用、任务交付、自主程度，以及实验分母与预算 |
| 运行模块评测 | [模块 runner](../scripts/run_module_evaluation.py)、[场景清单](../tests/fixtures/evaluation/module_coverage_v1.json) | 模块、场景、工程用例与执行尝试的组织 |
| 理解负载测量 | [压测说明](../scripts/service_load/README.md)、[汇总器](../scripts/service_load/summarize.py) | 任务吞吐、终态延迟、幂等、限流和资源采样口径 |
| 查看持续集成 | [CI 工作流](../.github/workflows/ci.yml) | 依赖环境、检查选集与运行步骤；最近已记录结果见实测报告 |

## 研究与项目沿革

[检索架构决策](RETRIEVAL_ARCHITECTURE_DECISIONS.md)和[检索验证计划](RETRIEVAL_VALIDATION_PLAN.md)记录设计选择、待验证假设与实验方案。[成果清单](PROJECT_OUTCOMES.md)、[项目状态](PROJECT_STATE.md)及[实施队列](IMPLEMENTATION_QUEUE.md)保留按日期推进的研发记录，历史段落描述当时的条件与结论。

更早的快照包括[状态归档](PROJECT_STATE.history-20260926.md)、[产品合同归档](ACTIVE_PRODUCT_CONTRACT.history-20260926.md)和[评测周期归档](EVALUATION_FIRST_CYCLE.history-20260927.md)。这些材料用于追溯演进；当前对外结果集中在[实测报告](PUBLIC_RESULTS.md)，可执行接口由相应代码和配置定义。
