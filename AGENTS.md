# Repository Instructions For Codex

Work toward the current user's outcome. These agreements can change with the task;
they do not freeze an old architecture or override platform permissions.

1. Establish the requested result and observable acceptance, then investigate,
   implement, run and repair until that result is delivered. Make reasonable
   implementation choices. Ask only when missing information materially changes
   scope or authority; reuse authorization already given.
   为什么要有它：完成条件决定何时结束，局部实现、分析产物或等待重复批准不能替代交付。

2. Read the source, configuration and callers needed for the current question.
   Use `rg` and existing engineering navigation when useful. Consult architecture,
   product or coverage documents for the relevant contract. Load skills and their
   references only for the workflow they serve.
   为什么要有它：按需上下文能减少目标冲突，不需要每次先扫描整库、加载整套文档或生成任务包。

3. Reuse the current LangGraph, ArtifactStore and domain modules where they fit.
   Redesign interfaces, validators or ownership when evidence warrants it, and
   update their callers. Related fixes may land together; isolate factors only
   when claiming a causal improvement. Extract code when responsibilities need it.
   为什么要有它：已有实现是复用起点，错误的表示和重复门禁也必须能够修改。

4. Preserve unrelated WIP and original run evidence. Inspect the affected diff;
   do not reset other work. Keep attempts' actual source, model, permissions,
   budgets and outcomes. Treat old packets, playbooks and task memory as historical
   hints whose applicability must be checked.
   为什么要有它：共享工作区需要可恢复性，旧任务的目标、停止条件和允许目录不能覆盖当前请求。

5. Bind product-source Apply to the exact diff, current source preimage, allowed
   paths and effects. Reuse valid human approval or explicit task delegation,
   recording its real origin. Honor disclosure, Build, deployment and game scope;
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
   Every added rule includes its scope and “为什么要有它”.
   为什么要有它：真实开发积累可复查成果，重复报告和补写成功叙述不会补齐待测数据。

9. In user-facing progress, work-in-progress summaries, and project documentation,
   prefer standard engineering terms and plain language. Explain the task, concrete
   problem, technical mechanism and trade-off, verified result, and remaining work.
   Introduce a necessary project-specific name with its general concept and purpose;
   do not use internal field/state names or management shorthand as the result.
   Keep executable identifiers, schemas, paths, hashes, and historical run evidence
   unchanged when improving prose. Clearly separate current progress from history.
   为什么要有它：用户需要能理解、检索和解释技术成果，而不是先学习项目自创词汇；改善表达也不能破坏程序接口或历史证据。
