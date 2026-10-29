# Repository Instructions For Codex

Work toward the current user's outcome. These agreements can change with the task;
they do not freeze an old architecture or override platform permissions.

1. Establish the requested outcome, observable acceptance and current work boundary.
   Use the user's explicit limits, invoked workflow and existing authorization to
   distinguish explanation, review, investigation, implementation and execution.
   Honor analysis-only requests; complete authorized fixes and checks without
   repeated permission. Ask only when missing information changes scope or authority.
   为什么要有它：交付应对应当前请求；既不能把讨论扩成实施，也不能让已授权工作反复等待批准。

2. Read the source, configuration and callers needed for the current question.
   Use `rg` and existing engineering navigation when useful. Consult architecture,
   product or coverage documents for the relevant contract. Load skills and their
   references only for the workflow they serve.
   为什么要有它：按需上下文能减少目标冲突，不需要每次先扫描整库、加载整套文档或生成任务包。

3. Reuse the current LangGraph, ArtifactStore and domain modules where they fit.
   Change interfaces, validators or ownership when needed to meet the current
   acceptance, and update their callers. Related fixes may land together within
   that boundary; isolate factors when claiming causal improvement. Keep other
   discoveries in the existing queue. When work is interrupted, record what is
   deferred, why, and the step after which the original queue will be reconsidered.
   为什么要有它：修复可以改变技术方向，但发现相邻问题不能自动延长工作链。

4. Preserve unrelated WIP and original run evidence. Inspect the affected diff;
   do not reset other work. Keep attempts' actual source, model, permissions,
   budgets and outcomes. Treat old packets, playbooks and task memory as historical
   hints whose applicability must be checked.
   为什么要有它：共享工作区需要可恢复性，旧任务的目标、停止条件和允许目录不能覆盖当前请求。

5. Bind product-source Apply to the exact diff, current source preimage, allowed
   paths and effects. Reuse valid human approval or explicit task delegation,
   recording its real origin. Honor the selected disclosure, Build, deployment
   and runtime scope, including game access when applicable;
   model output grants no authority. Do not expose secrets or bypass platform denial.
   为什么要有它：去掉重复审批仍须守住实际源码与副作用边界，平台开发和产品执行各按当前授权推进。

6. Verify changed behavior and the normal entry path with sufficient focused
   checks. Preserve the first failure, fix its direct cause, and rerun affected
   checks. Broaden only for an unresolved concern or selected release requirement.
   Seek focused independent review for authorization, effects or result validity;
   otherwise continue delivery once checks pass.
   为什么要有它：有意义的反例和真实验收能检验改动，固定追加测试与全面等待审阅会拖慢进度。

7. Parallelize independent investigation, implementation or review when it helps.
   Keep code ownership clear and integrate stable results; do not create another
   report or audit merely because a subtask or optional module exists.
   为什么要有它：并行要减少等待，不能把一个需求拆成更多流程产物。

8. Give one concise result linked to original evidence: delivery, necessary cost,
   boundary checks and any direct blocker. Distinguish developer-assisted work,
   product-model outcomes and independent evaluation, retaining unknowns. Use
   experimental protocols and skill schemas only for their selected purpose.
   Each added rule states its scope and “为什么要有它”; keep the explanation with
   the decision instead of repeating it on every ordinary field or test assertion.
   为什么要有它：真实开发积累可复查成果，重复报告和补写成功叙述不会补齐待测数据。

9. Use standard engineering terms and plain language in progress and documentation.
   Explain the concrete problem, mechanism, verified result and remaining work;
   define necessary project-specific terms. Distinguish current progress from
   history, and preserve executable identifiers, schemas, paths, hashes and evidence
   when improving prose.
   为什么要有它：用户需要能理解、检索和解释技术成果，而不是先学习项目自创词汇；改善表达也不能破坏程序接口或历史证据。

10. **Run evidence:** before inspecting a run, attributing its failure, assessing
    capability from runs or choosing the next experiment, read
    [Run evidence](docs/agents/run-evidence.md) and use its shared entry.
    为什么要有它：运行归因需要同一身份下的原件；具体查询步骤只在这些任务中加载。

11. For changes to decisions, tools, feedback, context, retrieval, models,
    experience or resource policy, state the expected effect, comparison baseline
    and applicable verification in the existing task record before implementation.
    Before preparing a real Agent experiment, making an Agent capability claim,
    or running or reporting that verification, read
    [Evaluation](docs/agents/evaluation.md). Complete already authorized, registered
    verification when ready, within its resource policy and stop/return conditions.
    Ordinary refactoring and documentation need focused checks, not full model evaluation.
    为什么要有它：验证应随已授权工作完成；代码修改本身不自动授权下一轮实验。

## Agent skills

### Issue tracker

RepoForge issues, specs and implementation tickets use GitHub Issues; pull requests are not a request surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Use `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human` and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout: domain language lives in the root `GLOSSARY.md` and hard-to-reverse decisions live in `docs/adr/`. See `docs/agents/domain.md`.
