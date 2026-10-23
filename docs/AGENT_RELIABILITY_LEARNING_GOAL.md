# RepoForge 自主开发与策略经验：实施设计

设计调研基线：`fdf92b727d6cb5d0fa56feecc713b5e6508ab2d5`。用户主设计与五处局部修订已合并，主设计不另行重写；以下为首选设计和完成条件，实现与运行证据单独见[项目进度](PROJECT_STATE.md)。应用内目标已由用户更新为U1—U5，当前持续实施U1—U4并穿插真实B。为什么要有它：设计、代码实现、受控通路和真实自主结果必须分别判断。

原Goal、R01—R18、14个子验收、历史原件和适用专项继续有效；已取消的三项v0.1发布前置限制与一般实验资源续批不恢复，真实权限和平台硬限制保留。为什么要有它：局部架构改造不能删除长期合同或重新引入已撤销门槛。

以“配置更新后，已有查询对象仍读取旧值，同时必须保留正常缓存行为”为例，我选择以下目标过程：
1. Agent 最初认为问题在写入端，只修改 ConfigWriter.cs，得到候选 C1。公开复现显示：新建对象能读到新值，已有对象仍读旧值。这反驳了当前假设。
2. Agent 选择已有生命周期测试，并提出“更新前／更新后创建对象 × 首次／重复查询”的临时实验，观察对象持有的配置版本和缓存命中。
3. 实验指向 SnapshotStore.cs 与 CachedQuery.cs。Agent 读取当前源码，说明修改理由及预期验证；平台确认这两个文件已在授权范围内，直接建立精确 binding，不再要求另一轮 Evidence、Planning 模型批准。
4. Agent 修订技术计划，撤销 C1 中不必要的修改，形成跨文件候选 C2。初始计划选过的文件不必继续修改，也不需要制造无效补丁满足计划。
5. 实验结束后进程中断。恢复时保留 C2、实验回执、公开反例、已否定假设和累计消耗；已执行实验不重复运行，未知执行先对账。
6. Agent 根据反馈继续修订，经冻结的公开验收后提交候选；产品执行层核验具体 diff、当前前像及有效授权，完成 Apply、Build 和适用验收。隐藏最终评测只评价冻结结果，不回流求解。
这是本方案的集成验收场景，不是已有能力声明。为什么要有它：跨文件修复必须同时证明自主选择、版本连续性、权限约束和最终交付，组件分别存在不能替代完整过程。
## 一、源码基线与本轮结论
设计调研时核对的提交为 fdf92b727d6cb5d0fa56feecc713b5e6508ab2d5，当时工作区无未提交改动。当前实施状态见[项目进度](PROJECT_STATE.md)。它与 4dcfe5f 的差异只有三份目标文档，相关产品源码相同。
调查期间收到原任务交接，并只读核验了 B12 原件：B12 已结束，累计预留 12 次模型调用，其中 Task Agent 8 次；首候选编译失败，业务测试执行数为 0，后续没有形成有效新候选，未交付。终止码为 TASK_AGENT_TOOL_ARGUMENT_CORRECTION_EXHAUSTED。见[运行原件](C:/tmp/rfext-polly-2164-b12/run.json)和[执行回执](C:/tmp/rfext-polly-2164-b12/runs/dsh_c2c6047791294963973d1bc10ac42dd3/a/task-agent-execution/execution.json)。本轮没有重启、恢复或接管该运行。
当前事实与设计判断分开如下：

| 类别 | 结论 |
| --- | --- |
| 已实现 | 连续 Task Agent 工具循环、当前候选读取与修订、跨重新规划保留修改、真实候选检查、正式交付接管、同进程共享模型预算。 |
| 已实现但范围有限 | TaskMemory 有来源绑定、适用条件和检索；当前注入 Planning 请求。已有恢复入口能够承接特定已完成候选的交接失败。 |
| 尚未实现完整通路 | 直接确定性扩展 binding、Agent 自选测试和临时探针、策略经验进入 Task Agent 实际请求、正常入口的完整决策会话恢复。 |
| 工程判断 | 保留 LangGraph、ArtifactStore、候选物化和交付后端；改动职责归属及其调用者，不另建一套平行执行平台。 |
| 未获证明 | 真实模型能够稳定自主修复、策略经验具有跨任务净收益。本设计不预先宣称这些结果。 |



首选方案是四项职责分离，并以版本化任务会话连接它们。 适用范围首先是现有本地、可丢弃隔离工程、compile-only 的 Task Agent 路径；不顺带扩大外发、游戏运行或部署权限。为什么要有它：可以沿已有产品入口实施，同时避免把架构升级误当成新增执行授权。

| 职责 | 唯一责任方 | 内容 |
| --- | --- | --- |
| 任务合同 | 产品入口及可信授权持有者 | 原始需求、明确保留义务、授权来源、路径与操作范围、公开验收、资源政策。 |
| 技术计划 | Task Agent | 假设、拟修改目标、实验选择、已否定解释、下一步验证；允许修订。 |
| 真实执行 | 平台工具与执行后端 | 源码读取凭据、binding、候选物化、实验派发、日志、资源预占和回执。 |
| 独立验收 | 独立于候选生成的验收 owner | 冻结公开验收及正式交付检查；隐藏评测由求解之外的评估方负责。 |



以上分工不让技术计划、经验或模型自述产生权限。为什么要有它：技术判断需要可变，授权来源和执行真实性必须能够独立核验。
## 二、对应当前调用链及迁移位置
当前正常链路为：
CLI run-request → run_local_request → execute_product_execution → LangGraph 前置需求／证据／规划 → generate_patch_proposal → TaskAgentExecution → TaskAgentSession → run_task_agent_loop → 候选检查 → validate_task_agent_handoff → Proposal／Review／Dry-run／Apply／Build／适用评估
当前新增目标的实际支路是：
source_search → search 引用 → upstream(evidence) → Planning selector → ImplementationPlanner.build → 新 plan 内 binding_registry → 当前候选重新映射
当前经验支路则是：
TaskMemory.retrieve → memory_model_view → ModelPlanningTargetSelector 请求
它尚未进入 Task Agent 的实际行动请求。
下面的映射是实施范围；不是只增加几个模块名称。

| 现有文件／关键函数 | 当前职责 | 拟改职责与接口 | 保留的执行边界 | 验收方式 |
| --- | --- | --- | --- | --- |
| [local_entry.py：执行器选择与装配 (line 6889)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/local_entry.py:6889)、[graph.py：规划后路由 (line 1530)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/workflow/graph.py:1530) | 前置规划完成后创建 Task Agent；入口限制本地、隔离、compile-only 等组合。 | 增加显式 v2 协议装配；先建立任务合同，允许技术计划和 binding 集合为空。技术证据不足进入调查，不再一律挡在 Agent 前。 | 需求歧义、源码身份无效、权限不足等真实准入阻断仍保留；不能笼统忽略旧 blockers。 | 正常 CLI 在“尚未确定编辑目标”时能进入调查；无授权时不能提交候选。为什么要有它：调查应能解决技术未知，不能绕过真实准入。 |
| [task_agent_execution.py：prepare_task_agent_contract (line 107)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/task_agent_execution.py:107) | 从 plan 的 proposed_changes 生成每项必改义务。 | v2 分别消费任务合同、平台注册表和技术计划；计划只作为可修订决策资料。 | 原始需求、用户保持义务、操作权限不因计划变化而消失。 | 初始多选文件可保持不变，少选文件可授权内扩展。为什么要有它：计划不是第二份授权，也不是必须完成的补丁清单。 |
| [implementation_plan.py：binding 构造 (line 359)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/planning/implementation_plan.py:359)、[code_generator.py：必改覆盖检查 (line 2053)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/proposal/code_generator.py:2053) | 规划器拥有 registry；补丁必须精确覆盖计划 change IDs。 | v2 移除“选中即必改”和精确覆盖计划的门；改为逐项验证实际编辑的 binding、前像和操作。 | v1 历史语义不改写；v2 仍拒绝越权、歧义前像和无效编辑。 | 同一任务放弃错误目标、增加正确目标，能够形成最小有效补丁。为什么要有它：必须修改的是业务错误，不是初始计划的所有条目。 |
| [task_agent_tools.py：_search／_prepare_upstream (line 749)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/task_agent_tools.py:749)、[task_agent_owners.py：upstream (line 385)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/task_agent_owners.py:385) | 新搜索引用须经 Evidence、Planning 才能成为编辑目标。 | 增加平台 binding_add；Agent 可直接 plan_update。Evidence／Planning 保留为可选复杂调查或咨询能力。 | 搜索命中本身不授权；所有 owner 仍使用同一资源账户。 | 新 binding 不调用上游模型；来源失效、超范围请求明确拒绝。为什么要有它：普通目标调整不需要固定两轮模型审议。 |
| [bindings/resolver.py (line 84)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/bindings/resolver.py:84)、[patch_apply.py：materialize_exact_text_patch (line 139)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/execution/patch_apply.py:139) | 从 plan 取 binding，与 Evidence 对齐并精确物化。 | 增加 from_registry_v2；物化函数显式接收已验证 registry，不生成假 plan 迎合旧接口。 | 路径规范化、源码 hash、范围、唯一前像、编码、精确 diff 均保留。 | 篡改引用、路径逃逸、错误版本及错误前像全部拒绝。为什么要有它：移除计划门禁不能移除执行校验。 |
| [current_candidate.py (line 20)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/current_candidate.py:20) | 换 plan 要重新表达相同代码；修订绑定父候选。 | v2 代码身份独立于 plan／registry；增加目标不重造候选，真正编辑才生成子候选。 | 父版本、累计修改、当前前像、原始基线全部保留。 | 改 A 后新增 B，A 的修改仍在；撤销 A 必须明确修订。为什么要有它：计划变化不能隐式丢代码。 |
| [task_agent_tools.py：_check／completion_check (line 1006)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/task_agent_tools.py:1006)、[CSharpRepositoryBackend (line 331)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/domain/csharp_repository_backend.py:331) | 执行固定完整检查，报告还绑定 plan/context。 | 保留完整公开验收；另增探索执行接口。验收绑定代码、完整检查规格及环境，移除技术计划身份依赖。 | 探索不能设置正式通过；实际执行、测试数量和报告来源仍须核验。 | 子集通过不能 finish；同代码换验收／环境不得套旧结果。为什么要有它：实验信息与交付资格不同。 |
| [task_agent_execution.py：validate_task_agent_handoff (line 196)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/task_agent_execution.py:196)、[task_delegation.py (line 116)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/execution/task_delegation.py:116) | 重建真实候选、检查、finish，交给产品审批和交付。 | v2 验证合同、授权、registry 来源、候选链和验收；技术计划只附审计引用。 | 有效人审／任务委托来源、精确 diff、活源前像及副作用范围不变。 | 错候选、旧报告、伪造工具调用或授权都不能交付。为什么要有它：Agent 完成只代表候选就绪。 |
| [task_memory.py (line 308)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/planning/task_memory.py:308)、[model_selection.py (line 447)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/planning/model_selection.py:447) | 来源绑定事实、条件检索；经验进入 Planning。 | 保留 v1；增加策略 v2、生命周期、冻结快照和 Task Agent 决策投影。 | 经验只建议，不能改目标、权限、验收或预算。 | 检查实际 provider request，而非只检查检索命中。为什么要有它：经验必须真正到达决策位置。 |
| [task_agent_loop.py (line 163)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/task_agent_loop.py:163)、[retained_task_agent_handoff.py (line 76)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/retained_task_agent_handoff.py:76) | 循环已有历史恢复参数；正常入口未接通完整会话恢复。后者只续接特定交付错误。 | 同一执行 owner 增加 checkpoint 重建；旧 retained handoff 保持其限定用途。 | 不重发未知动作，不清零纠错、无进展计数或累计资源。 | 在发送前后、检查前后中断，经正常 CLI 恢复。为什么要有它：恢复会话不能等同于重开任务。 |
| [memory_comparison.py (line 71)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/src/mod_agent/evaluation/memory_comparison.py:71)、[prepare_memory_trials.py (line 33)](C:/Users/10160/.codex/visualizations/2026/08/11/019ff15f-9eac-7b13-aad5-faffde4b36b7/rag_capability_mainline/scripts/prepare_memory_trials.py:33) | 已冻结共同条件、经验开关和独立目录；trial 脚本硬编码少数工程。 | 复用冻结与漂移检查，抽出公共派发；任务配置来自 manifest，增加真实请求送达与负迁移记录。 | 原试验和原结果不改写；隐藏评分不进入求解。 | 非原固定工程能够通过正常入口配对执行。为什么要有它：已有比较基础可以复用，但不能冒充通用评测能力。 |



## 三、关键接口和状态已经选定
以下接口适用于 Task Agent v2；旧 v1 继续按原 schema 读取和重放。ArtifactStore 注册新版本并显式区分消费者，不静默重解释历史产物。为什么要有它：当前运行和历史证据不能被新语义追溯改写。
任务合同与技术状态。
TaskContractV2
  task_id
  original_request_ref
  feature_spec_ref
  requirement_sources
  authority_ref
  baseline_manifest_ref
  public_acceptance_ref
  execution_profile_ref
  resource_policy_ref

TaskAgentContextV2
  task_contract_ref
  binding_registry_ref
  technical_plan_ref
  evidence_index_ref
  candidate_head_ref

TechnicalPlanV2
  hypotheses
  intended_targets
  proposed_experiments
  rejected_hypotheses
  supporting_observation_refs
任务合同保存原始需求及真实来源；模型推导的解释、内部保持建议进入技术计划，不能仅因写入 FeatureSpec 就升级为用户硬要求。初始没有明确编辑目标时，registry 可以为空；但首次候选的每个实际编辑都必须已有有效 binding。为什么要有它：允许先调查，不等于允许先收任意补丁再补授权。
首版沿用宿主明确授权的现有文件集合，支持 modify_existing；不从它推导 create、delete、rename。范围外目标返回扩权需求；范围内增加目标不重复请示。为什么要有它：路径授权与操作授权必须明确，同时避免把普通技术调整变成人工审批。
新 binding 由平台生成。首版只接受现有文件的明确整文件修改授权，binding 固定为文件级，包含 binding_id、authority_ref、baseline_manifest_ref、canonical_path、operation=modify_existing、scope=file、baseline_file_sha256。target_selector 仅为规范文件路径，行号只用于读取展示。区域授权不得扩为文件授权；不支持时返回 unsupported_scope，既有区域受限路径保留。为什么要有它：权限范围可以稳定，编辑前像必须随父候选变化，简化定位不能扩大权限。

实际编辑绑定 parent_revision_ref、该父版本的 source_view_ref（文件、hash与读取内容），以及唯一精确匹配的 before_text。文件变化后重新读取，旧视图不能用于新编辑。空 registry 的读取依据任务读取范围和源码快照，baseline 可调查尚未绑定的文件；只读依赖按根标识与只读快照访问，不提供写权限。为什么要有它：模型必须先调查再选择目标，读取凭据不能成为编辑授权。
模型调用：
binding_add(
  expected_registry_ref,
  source_observation_ref,
  source_revision_ref,       # B0 或当前候选
  operation,
  target_selector,          # canonical file path; file scope only
  rationale,
  expected_validation
)
可信 owner 从会话取得合同和授权，模型不能传入一份自称有效的权限记录。返回：
BindingDecision
  status: added | existing | rejected
  registry_ref
  binding_id?
  baseline_preimage_ref?
  current_source_view_ref?
  rejection_code?
平台执行顺序固定为：核验 observation 原件和来源 → 解析规范路径及目标范围 → 核对路径／操作权限和保护对象 → 核对 B0 与当前候选字节 → 建立不可变 binding → 原子发布 registry 新版本。模型的理由和预期验证作为技术依据保存，不由平台判断“理由正确”才授予原有权限。为什么要有它：绑定是可复算的执行事实，技术判断由后续实验和验收检验。
拒绝条件固定为：引用缺失或篡改、过期源码／候选、路径逃逸或链接逃逸、目标范围不唯一、前像不一致、路径或操作未授权、触及保护的验收资产、registry 版本竞争，以及执行器不支持该操作。拒绝不清空已有候选或检查。为什么要有它：错误的扩展请求不能破坏已经完成的工作。
首版继续采用精确文本编辑，不引入 AST 重写。首次和后续提交统一为：
candidate_submit(
  parent_revision_ref,      # 首次为 B0，后续为 Cn
  edits: [{
    binding_id,
    source_view_ref,
    before_text,
    after_text,
    rationale
  }]
)
before_text 必填且必须在当前授权范围内唯一匹配；省略它不再隐含“替换整段源码”。无变化提交返回结构化反馈，沿用累计纠错政策。为什么要有它：同一种编辑语义应贯穿首次和后续修订，避免格式转换损坏当前代码。
代码身份定为：
candidate_source_id =
  hash(version, B0_manifest_hash, canonical_overlay_manifest)
overlay 记录规范路径、操作及实际文件字节 hash，不包含技术计划、理由或 binding 集合；无变化文件不进入 overlay。为什么要有它：同一代码不能因增加可编辑目标或改写计划而变成另一个被测版本。

| 变化 | 当前候选 | 既有检查 | 授权处理 |
| --- | --- | --- | --- |
| 修改技术计划 | 不变 | 保留原事实及符合条件的资格 | 不产生新权限 |
| 新增未使用 binding | 不变 | 不因注册表增补自动失效 | 核验新增目标原有权限 |
| 实际编辑／撤销编辑 | 生成子候选 | 旧结果留作历史；新代码须验收 | 每项编辑核验 binding |
| 公开验收、依赖或 runner 改变 | 代码可保持 | 原结果不能证明新检查合同 | 新执行仍须获准 |
| 撤销权限或实际源码漂移 | 保留证据 | 不再足以支持交付 | 停止相关执行并重新准入 |



检查资格使用 candidate_source_id + check_spec_hash + runner_identity + environment/dependency_identity，并核验真实执行、完整性和时效。再次请求执行时仍遵守现有 deterministic_reuse_allowed；保存旧事实不等于获准跳过一次新检查。为什么要有它：测试范围、环境和执行政策也是结果成立的条件。
自主实验与三类检查。

模型通过 probe_submit(source_revision_ref, files:[{relative_path,content}], entrypoint_id, rationale, expected_observation) 提交不可变 probe_artifact_ref。平台按已配置 runner profile 校验路径、大小、类型和固定入口；提交只写实验工件，不运行或修改候选；修订产生新引用，再由 experiment_run 绑定准确候选执行。禁止提交工程脚本或覆盖路径绕过 runner。为什么要有它：模型必须有实际产生探针的入口，提交内容与授权执行需要分开。

没有合格隔离环境时只关闭探针执行，不阻塞已有测试选择、U1或U2。为什么要有它：环境缺口应限定对应能力，不能成为全部主线的统一等待条件。
experiment_run(
  source_revision_ref,
  kind: registered_test | temporary_probe,
  test_ids? | probe_artifact_ref?,
  hypothesis,
  expected_observation,
  runner_profile_ref,
  requested_limits
) -> ExperimentReceipt
receipt 至少包含候选身份、测试集合或探针 hash、真实 runner／环境、执行状态、观察与断言、退出码、stdout／stderr 引用、资源消耗；固定 acceptance_eligible=false。为什么要有它：实验结果必须可复核，同时不能伪装成完整验收。
- **测试选择：**Agent 从平台注册的测试目录选择 ID；平台解析为受限命令参数，不接受任意 shell。零测试执行不能报告成功。为什么要有它：保留模型选择实验的能力，同时保证执行范围可确定。
- **临时探针：**Agent 创建版本化 probe artifact；平台将候选和探针放入独立执行区，探针不得覆盖候选源码或公开验收文件。为什么要有它：可修改的诊断程序不能污染交付内容或裁判。
- **执行隔离：**任意模型探针只交给具备实际文件、进程、环境变量和网络限制的 runner。默认无网络、无宿主凭据、最小挂载、限制子进程及超时；能力缺失返回 capability_unavailable，不退回普通子进程。为什么要有它：当前 NuGet 构建执行器继承宿主环境并使用普通 subprocess，临时目录不是安全沙箱。
- **结果回流：**探索结果和公开验收失败进入下一次 Task Agent 请求；隐藏最终评测不进入请求、经验提炼或当批调参。为什么要有它：探索用于判断假设，公开验收用于继续修复，隐藏评测用于独立测量。
- **执行恢复：**状态为 prepared → dispatched → completed|failed|unknown。中断后先查询同一 operation ID；不能确认是否执行时保留 unknown，不自动重跑。为什么要有它：没有收到结果不等于动作没有发生。
公开 candidate_check 保持完整、冻结的必需检查集合，模型不能通过选择测试子集缩小它；finish 只产生 candidate_ready。为什么要有它：模型可以决定如何调查，但不能自行降低完成标准。
策略经验。

U2 的生产、试用和采用链：run_local_request 统一终态收尾由 TaskAgentExecution 提供已落盘行动与反馈；有公开反例、有效修订或被反驳假设等信息增量时，按 episode hash 幂等调用 TaskMemory.propose_from_episode()，保存已准入模型的实际请求响应及来源。record_strategy() 核验数据使用范围并写 proposed；配对准备通过 admit_trial() 根据来源、适用域与试验政策形成冻结 trial 快照。评测模块 decide_strategy_adoption() 消费独立原件、冻结条件、实际送达和行动回执，生成政策版本绑定的决定；transition(expected_revision, decision_ref) 执行保持试用、限定启用、停用或退役，收窄另建 revision 后重新试用。为什么要有它：责任链必须落实到谁调用、谁写入和谁作出采用决定，不能仅定义状态名称。

不做逐轮反思。提炼计入原资源账本；meter_only 继续只计量，有确实配置的额度才沿用剩余量，不能恢复已撤销的人工续批门槛。无信息增量可不提炼。隐藏评测只在批次结束后将规定范围结果提供给采用判断，隐藏测试、答案与细节不进入提炼请求；小样本正向结果仍可保留 trial。为什么要有它：学习成本必须真实，来源可追溯、策略有效与成熟程度是不同结论。

快照冻结内容、版本和选择条件；每次决策和执行边界另核最新撤回状态。撤回可停止受影响任务与批次，不能替换新策略进旧快照，已发动作和成本保留。人工策略先只计策略消费与试用机制，自动提炼链经真实入口验证后才计自动提炼通路，迁移收益仍需新任务。为什么要有它：冻结条件与撤回都必须真实生效，人工提供策略不能冒充系统自动学习。
保留现有来源绑定事实记录，新增策略经验版本：
StrategyExperienceV2
  lesson_id, revision, previous_revision_ref
  authority: advisory_only
  trigger
  mechanism_hypothesis
  suggested_actions
  expected_observations
  counter_indications
  reconsider_when
  applicability:
    required_facts, excluded_facts, dependency_constraints
  provenance:
    episode_refs, observation_refs, counterexample_refs
    extraction_origin, extractor_request_ref, extractor_response_ref
  state: proposed | trial | active | suspended | retired
  evaluation_refs
事实引用与策略推断分开：平台验证来源、候选和数据使用范围；策略仍是可反驳的假设。来源可以是公开反例、有效修订、实验推翻假设及任务终态；开发者撰写条目标注人工来源，隐藏评测内容不作为求解经验。为什么要有它：真实经历支持提出策略，但不能自动证明策略可迁移。
生命周期采用以下默认选择：

| 转换 | 条件和处理 |
| --- | --- |
| proposed → trial | 来源有效、适用条件明确、数据可用于学习；进入限定试验，不逐条默认等待人工审批。 |
| trial → active | 新任务独立结果支持其声明的作用范围；附采用决定和反例，不凭使用次数或模型置信度升级。 |
| 修订／收窄 | 新建 revision，修改适用条件或建议，保留旧版本与反例。 |
| active/trial → suspended | 来源失效、条件冲突、权限诱导或有证据的负迁移；默认检索立即排除。 |
| suspended → 新版 trial | 修改后重新试用；不原地抹掉失败。 |
| → retired | 版本被替代或决定弃用；保留历史，禁止新任务注入。 |



为什么要有它：经验需要可自动试用，也必须可执行地撤回；当前文字形式的 invalidation_conditions 不足以实现停用。
适用性返回 match|mismatch|unknown；硬前提未知时默认不注入。来源文件 hash 用于验证经历真实性，不能机械要求目标任务具有同一仓库 hash；目标适用性依据显式运行时、依赖、API 和机制条件判断。为什么要有它：既不能失去来源约束，也不能把跨任务迁移锁死为复用原题。
实际注入位置定为：
local_entry.run_local_request
  → TaskAgentExecution(experience_snapshot_ref, experience_provider)
  → TaskAgentSession.initial_request
  → run_task_agent_loop.prepare_decision_request
  → 实际 complete_turn 请求保存与发送
每次决策前从冻结快照按当前事实筛选；保存 lesson refs + selection reasons + model_view hash + final request ref。检索不需要每轮额外模型评分；新反例、事实变化或主动请求触发重新筛选。为什么要有它：经验必须进入真实决策，不能只存在于检索报告或前置 Planning。
B 和 B+ 使用同样工具说明、版本语义及纠错反馈。针对 Task Agent 经验效果的比较，前置 Planning 输入相同；B 经验列表为空，B+ 增加冻结策略，不能同时偷偷改变 Planning 经验。为什么要有它：工具可用性和跨任务策略收益必须分开归因。
负迁移记录关联：策略版本、匹配依据、实际送达请求、模型自述采用／拒绝、真实行动、公开反例、独立结果、配对差异及处置。模型自述只算自述，无法归因就记录 unknown。为什么要有它：不能把所有失败归咎经验，也不能把注入或引用经验计为收益。
上下文投影和恢复。
持久状态至少包括：
TaskSessionCheckpoint
  task_id, run_id, continuation_seq, last_event_seq
  task_contract_ref, authority_ref
  baseline_manifest_ref
  model_identity, worker_ref, execution_profile_ref
  technical_plan_ref, binding_registry_ref
  candidate_head_ref, candidate_lineage_refs
  evidence_index_ref, public_failure_refs
  experiment_refs, accepted_check_ref
  execution_history_refs, call_records_ref
  correction_state, no_progress_state
  experience_snapshot_ref, experience_receipt_refs
  resource_ledger_ref, pending_operations
  delivery_state, applied_candidate_ref
资源账本保存：原政策和额度、分阶段及总模型预占、已知 token、未知用量、读取与观察字节、构建／实验次数及资源、累计执行时间、适用的绝对截止时间。继续复用现有预算回调，由任务级持久 owner 实现，不让每个循环维护新资源池。为什么要有它：同一任务的继续执行不能因换进程或换阶段重获预算。
所有有成本或副作用的动作遵循：
持久化 intent／operation ID → 持久化预占 → 执行 → 保存结果和实际用量 → 发布状态
相同 operation ID 幂等返回原状态；不同原生 call ID 不因内容相似被随意合并。发送后结果未知仍保留预占，未知 token 不填零。为什么要有它：中断窗口必须可对账，不能同时制造免费重试和漏执行。
正常入口增加 run-request --resume-task-agent <checkpoint>，复用同一个装配链。恢复先核验身份、取得单一执行锁、重建账本、对账 pending，再恢复候选及决策。meter_only 仍为无上限且累计；有额度的任务沿用原剩余量，截止时间不重新计算。为什么要有它：恢复的是原任务，不是配置相似的新任务。
模型上下文是持久状态的投影：保留合同与权限摘要、当前代码身份、可用 binding、最后公开反例、未决／已否定假设、未完成动作、累计资源和经验版本；完整源码、旧日志通过可验证引用再读。原生未闭合调用链必须保留合法对应关系，压缩摘要不能伪装成历史真实工具调用。为什么要有它：减少上下文不应改变身份、事实或执行资格。
正式交付继续由产品层拥有。已 Apply 的候选要记入 applied_candidate_ref；后续修订的实际 Apply 前像以该工作副本的真实字节为准，累计候选仍能从 B0 重建。公开失败可恢复同一 Task Agent；隐藏评分不触发继续求解；Apply 状态未知先对账。为什么要有它：恢复既要保留累计代码，也要面对实际已经发生的写入。
## 四、实施拆分：四个代码单元，加一个真实评测单元
第一项代码改动：修改 prepare_task_agent_contract 的权威输入来源，并在同一单元接通独立 registry。 不先做提示词补丁，也不只删除计划覆盖检查。

| 单元 | 最小代码变化 | 必要测试 | 正常入口验证与完成条件 |
| --- | --- | --- | --- |
| U1：合同、binding 和候选解耦 | 注册 v2 合同／registry；增加平台注册 owner；改 prepare_task_agent_contract、resolver、候选物化、生成器及 graph 路由；统一首次／后续编辑格式。先建立共用持久 intent、资源预占和回执基础。 | 授权内新增目标、越权拒绝、空 registry 调查、初始误选无需必改、前像歧义、增加 binding 不丢候选、实际代码变化使检查不再适用；v1 回放。 | 经正常 run-request --technical-executor task_agent --task-agent-protocol v2，用受控模型响应完成“误选 A→新增 B/C→保留并修订候选→产品接管”。通过才算接口接通。为什么要有它：先解除阻断自主修复的实际门禁，并给后续功能提供可恢复执行基础。 |
| U2：经验进入决策与评测合同 | 扩展 TaskMemory v2、状态与冻结快照；接入每次决策请求；扩展披露及 ArtifactStore consumer 校验；扩展现有配对 manifest、曝光登记和汇总字段。 | 实际 provider request 中 B/B+ 仅策略不同；未知条件、停用、过期来源排除；负迁移关联；快照漂移拒绝；提炼和检索成本入账。 | 同一普通 CLI 任务分别关闭／启用经验，保存实际请求及 receipt；通路通过不宣称模型收益。U1 接口确定后即可并行开发，不等 Polly 成功。为什么要有它：经验不是未来附加项，必须尽早进入实际决策路径。 |
| U3：测试选择与临时探针 | C# backend 增加测试目录与子集执行；增加 probe artifact 和受限 runner 适配；沿现有候选物化及统一操作回执执行。 | 选中测试真实执行、零测试拒绝；错候选／旧探针报告拒绝；不能覆盖验收；网络、宿主文件、凭据和进程树限制真实生效；能力不足明确不可执行。 | 正常入口由 Agent 选择子集、提出探针、收到结果并修订；另用第二种既有任务配置验证共享接口。没有真实受限 runner 时，探针部分保持未完成。为什么要有它：工具 schema 存在不能替代实际实验能力。 |
| U4：完整恢复与交付连续性 | 将 U1 账本接到 checkpoint；增加正常入口 resume；重建上下文投影、候选、经验快照和未决动作；完成跨进程 handoff 恢复、正式失败回流及 Apply 前像衔接。 | 发送前后、结果落盘前后、公开失败后、handoff 前后中断；预算／纠错／无进展累计不增加余额；未知动作不重发；旧报告、错版本交付拒绝。 | 新进程恢复开头的跨文件任务，通过正式验收及交付，原始失败和总成本可追溯。为什么要有它：局部历史恢复不能替代用户可以继续使用的任务恢复。 |
| U5：冻结后的真实 B/B+ | 通用 manifest 派发接入正常入口；复用独立评测接口，记录动作、结果、人工帮助与负迁移。 | 派发前冻结核验、批次漂移和撤回处理、隐藏数据隔离、汇总分母检查。 | 真实模型完成预登记的新任务批次，给出保持试用、收窄或停用结论；允许负结果结束试验。为什么要有它：平台通路不等于自主行为，单题成功不等于经验迁移。 |



U2 可以先完成评测合同格式，正式批次在实际采用的工具和平台版本确定后才冻结。U5 不以 B12、Polly 成功或全部长期能力完成为前置；只要求所测动作有效、反馈有效、版本一致、资源和恢复政策可冻结。为什么要有它：早做经验接口与公平冻结并不冲突。
协议默认化、经验启用与探针启用分别决定。v2 先保持显式开关，依据正常入口完整性、权限及版本反例、代表性真实任务和兼容性决定是否默认采用，不以U1—U4受控通路通过自动默认化。经验启用依据冻结的v2 B/B+结果及适用域；U5的经验对照不单独证明v2优于旧协议，后者需要另有可比证据。探针按runner实际能力独立启用，环境不足不撤回已验证的合同、binding或经验接口。为什么要有它：协议效果、经验效果和执行隔离是三个不同问题，采用范围须对应实际证据。

U1 增加显式 v2 handoff 校验与产品消费分支，输出 ValidatedTaskAgentHandoffV2，包含 task_contract_ref、binding_registry_ref、candidate_ref、check_ref、execution_ref、candidate_source_id 和 changes。校验真实 submit／finish 来源、候选父链、合同授权、registry、精确代码与公开验收，同步修改 product_execution 记录校验、product_executor、graph 与 Proposal 消费，禁止伪造旧Plan兼容。为什么要有它：新候选必须穿过真实产品边界，不能停在返回成功的适配函数。

U1 的产品接管定义为：同一候选进入正常 Proposal、Review、Dry-run，生成与该候选绑定的待交付记录。U1 同时包含空registry调查、文件binding、统一父候选精确编辑、最小持久操作和资源记录、v2候选检查及此最小交接。受控回归覆盖误选A、调查、新增B/C、撤销错误修改、公开检查和接管，以及旧报告、错候选、授权篡改；只计平台通路。首项从 prepare_task_agent_contract 的权威输入拆分开始，提交范围包含上述调用者。跨进程恢复、正式交付失败回流和已Apply前像连续性留U4，U2/U3/U4不是U1前置。为什么要有它：U1须独立兑现自己的完成条件，同时不能把最小接管说成完整交付。

## 五、评测设计及采用决定
三种证据分别报告：

| 证据层 | 怎么测 | 能证明什么 |
| --- | --- | --- |
| 通路测试 | 受控模型响应＋真实工具／执行后端，按指定动作覆盖跨文件修复和中断 | 平台能执行这些动作，身份与资源处理正确 |
| 真实模型自主行为 | 只提供任务、共同工具规则及该臂允许的经验；观察模型实际选择 | 模型是否主动调查、设计有判别力的实验、依据反例改变修改 |
| 独立业务结果 | 对冻结交付候选运行独立验收，保护题目要求及保持义务 | 最终是否解决业务问题、是否发生退化 |



三者不相互替代；开发者补充公开复现标为辅助条件，不能算模型自主实验。为什么要有它：接线正确、行动自主和结果正确是不同结论。
B/B+ 共同冻结：平台提交与相关源码 hash、模型 digest 和推理设置、原始任务与初始工程、共同前置资料、工具定义和说明、请求模板及投影策略、授权、runner／依赖、公开验收、资源政策、恢复政策和停止条件。B+ 唯一新增本批策略经验；两组实际动作和消耗可以不同。为什么要有它：动作差异是待测结果，共同条件变化则会破坏解释。
首批工程选择为四个未曝光任务：两个相同机制、不同实现或仓库的近迁移任务；一个表面相似但前提不成立的任务；一个盲目套用策略会破坏保持义务的任务。每臂一首尝试，顺序预登记并交替；这一批用于决定下一步，不支撑普遍可靠性声明。为什么要有它：首批同时覆盖收益和负迁移，不只挑容易受益的题。
训练／提炼资料、开发回归、迁移验证、隐藏最终评测分别登记。Polly 等已用于调试的题只进入开发回归；经验批次前冻结，批次中产生的新经验留到后续版本。隐藏评测在候选冻结后执行，评分不回流该批求解。为什么要有它：曝光和批次内调参会把记忆题答案伪装成迁移收益。
报告至少包含全部启动任务的正确交付率、有效执行比例、有效条件下成功率、有效修订、保持义务、人工技术帮助、总时间／调用／token／未知量、经验提炼检索成本和负迁移。前置失败保留在端到端分母。为什么要有它：分离故障原因有助修复，不能通过剔除失败改善成绩。
采用口径定为：
- **继续限定试用：**新任务出现正确交付改善，且没有关键保持义务退化；或正确结果相同，总成本中的时间和 token 均不恶化且至少一项改善。小批次只支持继续验证，不直接宣布普遍收益。为什么要有它：优先交付质量，效率收益必须包含提炼和检索成本。
- **收窄：**收益只在某类生命周期、API 或版本条件成立，则用新 revision 收窄适用条件；不把不适用任务排除后重写原成绩。为什么要有它：条件化策略可以有用，但作用范围必须真实。
- **停用策略版本：**出现有证据的错误套用、保持义务退化，或成本抵消收益；仅等同工具提醒时改归因到共同工具规则。为什么要有它：经验必须允许负结果改变采用决定。
- **停止当前实现扩大使用：**出现越权、隐藏资料泄漏或错误版本交付，先修复执行边界；不是通过多跑几次平均掉。为什么要有它：这些是结果有效性的前提。
- **放弃本策略／重新选择机制：**在有效公平条件下持续无净收益，保留关闭状态；不以“经验还不够多”无限续试。为什么要有它：首选设计需要可以被证据推翻。
在线停用命中冻结批次时，停止新派发，并在下一安全决策／执行边界停止受影响任务；已经发送的请求、动作和成本照实保留。批次记为 interrupted_by_strategy_revocation，不补跑新版 B+ 凑成旧配对。为什么要有它：撤回本身可能由负迁移触发，删除这些记录会产生选择偏差。
仍有三项环境或效果未知，但默认选择已定：

| 未知事项 | 默认选择 | 改变条件 |
| --- | --- | --- |
| 当前机器是否具备合格探针沙箱 | 保留固定测试能力；任意探针返回能力不可用 | 真实隔离反例测试通过后启用 |
| 哪些新任务满足未曝光且可执行 | 按公开 manifest 与曝光登记选择上述四类；不足则明确缺样本 | 找到合格任务后排期，不影响 U1—U4 |
| 策略是否具有净迁移收益 | 默认关闭，仅限定 trial | 独立新任务结果支持后限定启用 |

## 2026-10-23 增量：共享运行证据查询与统一入口

保留 U1—U5。在公共反馈及阶段事实修复收口后，复用 TraceService、RunRecordBrowser、context_chain 与运行状态读取，接通按业务运行 ID 查询的共享视图；不建立第二套存储、不搬迁历史原件，不另建 Dashboard、数据库或 MCP 服务。为什么要有它：开发调查、现有界面和评测程序需要共用引用关联，不应各自猜测目录或解析同一日志。

统一入口为 `python -m scripts.dsh_run_request --execution-config "<绝对配置路径>" --inspect-run "<业务运行ID>" --view evidence --format markdown`，程序使用 `--format json`。本节登记实施目标，是否可用以正常入口验证记录为准；默认仅标准输出，不生成报告或派发模型、Build、测试、恢复。为什么要有它：查询与执行分离，接口声明不能冒充已验证能力。

关联实际模型输入／响应、请求动作、工具执行回执、候选父链与精确改动、检查原件、公开反馈投影、其进入后续实际请求的事实，以及后续行动、验收和交付引用。分别显示请求准备、发送尝试、取得响应、反馈包含和已执行动作；采用源码、模型和运行身份及精确引用，不以时间相邻推断因果。输出累计已知成本、原件索引及缺失、歧义、不可访问、篡改和截断状态，不自动评价理解、采纳或业务成功。为什么要有它：证据查询应还原能够核对的事实，不替代语义判断或独立验收。

开发 Agent 的使用约定登记在 AGENTS.md；Dashboard 使用同一服务，CI 消费结构化输出。产品 Task Agent 和 U2 仍使用各自受限的公开反馈／经历接口，不直接读取完整开发视图；引用不授予目录权限，隐藏评测、凭据和外部修法不通过此模块进入求解请求。为什么要有它：共享关联逻辑不等于共享所有内容或新增产品权限。

正常入口验收覆盖：ID与配置解析；正常交付、检查后置异常、恢复及另开诊断的区分；缺失、篡改、越界、大文件截断不冒充完整；开发视图不成为产品工具；查询前后原件不变且没有执行派发。新运行返回实际运行ID及可复制检查入口。为什么要有它：用户必须能通过真实入口查到对应事实，不能只交付函数与演示报告。

插队暂缓 U2 仓库验收适配的构建／测试配置一致性收尾，保留未提交工作；上述入口反例完成后返回 U2—U4 剩余集成和跨任务验证，U5不取消。本项不自动启动 RestSharp 第三次运行，也不是全部 U2 工作的前置。为什么要有它：查询服务于交付，不成为新增长期平台主线。
