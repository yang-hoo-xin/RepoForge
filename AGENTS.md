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

10. When asked to inspect a run's evidence, attribute its failure, assess a capability,
    or choose its next experiment, use the shared evidence entry:
    `python -m scripts.dsh_run_request --execution-config "<absolute registered config>" --inspect-run "<business run ID>" --view evidence --format json`
    (`markdown` is the human-readable format). Follow returned references to the
    relevant actual requests, patches and checks; report missing or ambiguous facts.
    Establish the configuration once and reuse it when applicable; never choose the
    latest directory to resolve conflicting identities. Ordinary code edits do not
    require scanning historical runs. Natural-language requests are this working
    agreement, not a registered slash command or automatic trigger. Keep the
    developer view out of product-model inputs; their existing public-feedback and
    experience interfaces retain their own visibility and authorization boundaries.
    为什么要有它：共享入口只有进入实际调查流程才能减少错误归因，统一查询也不能扩张产品模型的可见范围。

11. For changes to decisions, tools, feedback, context, retrieval, models,
    experience or resource policy, state the expected effect, comparison baseline
    and applicable verification in the existing task record before implementation.
    Once usable, run that verification within existing authorization and resource
    policy; do not wait for a resume request. Freeze common conditions and identify
    the main tested change. Without a comparable baseline, report absolute results.
    Ordinary refactoring and documentation do not require full model evaluation.
    Reuse native artifacts, Trace, resource and delivery receipts; retain all
    attempts, failures, regressions, unknowns and the versions of mid-run fixes.
    Record external Codex root-cause advice, fixes and patch edits as technical
    assistance, separately from approval, review and environment repair. Human
    active time requires measurement or explicit declaration, never waiting-time
    inference. Separate task, independent-evaluation and platform-development
    costs. Do not change hidden acceptance, reveal its answers, or expand product
    authority for measurement. A supported negative result completes an experiment.
    为什么要有它：工程推进应默认留下可复算的结果与成本，避免漏掉外层救援、失败尝试和未知量，也不为取得漂亮数字延长实验或扩大权限。

## Agent skills

### Issue tracker

RepoForge issues, specs and implementation tickets use GitHub Issues; pull requests are not a request surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Use `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human` and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout: domain language lives in the root `GLOSSARY.md` and hard-to-reverse decisions live in `docs/adr/`. See `docs/agents/domain.md`.
