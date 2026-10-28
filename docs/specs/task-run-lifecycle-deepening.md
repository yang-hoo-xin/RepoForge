# Deepen the normal Task Run lifecycle behind one interface

## Problem Statement

Adding one real Task Run behavior currently requires coordinated changes across task opening, command dispatch, Task Agent execution, Action handling, recovery, terminal projection, pilot execution and evidence inspection. The user cannot rely on one normal entry to open or resume a repository task and receive one Run Outcome with one Evidence Bundle; several callers instead sequence shallow interfaces and reinterpret the same run facts.

## Solution

Provide one deep Task Run module with one execute interface. It accepts an in-process Open or Resume command, derives Task Authority from bound host configuration, sends every Effect through one internal admission seam, preserves one Task Run identity across recovery, and returns either a recoverable suspended result or a completed result containing the unique Run Outcome and Evidence Bundle references. DSH and pilot remain separate adapters that call this same interface.

## User Stories

1. As a RepoForge user, I want one normal entry for a repository task, so that I do not need to understand internal preparation and dispatch order.
2. As a RepoForge user, I want interruption recovery to use the same Task Run entry, so that recovery cannot silently become a new attempt.
3. As a RepoForge user, I want the same run and request identity after recovery, so that cost and outcome attribution remain correct.
4. As a RepoForge user, I want a recoverable interruption distinguished from a completed failure, so that the platform does not invent a terminal result.
5. As a RepoForge user, I want exactly one Run Outcome for a completed Task Run, so that different adapters cannot disagree about the result.
6. As a RepoForge user, I want one Evidence Bundle that references original facts, so that the Run Outcome can be recomputed without copying or rewriting evidence.
7. As a task owner, I want Task Intent separated from Task Authority, so that a model request or technical plan cannot grant file or Effect permissions.
8. As a task owner, I want every Effect admitted against current Authority, Binding, source preimage and version, so that stale or unauthorized changes do not execute.
9. As a task owner, I want an Effect Receipt recorded before an Observation is exposed, so that the agent cannot treat a request as an executed fact.
10. As a task owner, I want unknown Effect state retained across interruption, so that recovery reconciles instead of repeating a possible side effect.
11. As a Task Agent, I want Action results returned as Observations from verified Receipts, so that I can revise a technical hypothesis using actual feedback.
12. As a Task Agent, I want source read, Patch Candidate submission and public check to remain one continuous Task Run, so that investigation can become a verified code change.
13. As a Task Agent, I want cumulative resources preserved after recovery, so that interruption does not reset budgets or hide cost.
14. As a pilot runner, I want to call the same Task Run interface as the normal product adapter, so that evaluation does not exercise a parallel product path.
15. As a pilot runner, I want only Run Outcome and Evidence Bundle references, so that pilot aggregation does not reinterpret internal result files.
16. As a maintainer, I want the technical state machine to have one owner, so that a new Action does not require changes in every adapter.
17. As a maintainer, I want recovery and idempotency to remain behind the Task Run seam, so that tests do not depend on checkpoint file layout.
18. As a maintainer, I want the replaced preparation, dispatch, resume and result-projection interfaces deleted, so that the deepening does not create a third execution path.
19. As a maintainer, I want one vertical external-seam test, so that internal refactoring does not rewrite the test contract.
20. As a project owner, I want the architecture chain to stop after this tracer bullet is green, so that T1, T2 and T3 return to the active queue.

## Implementation Decisions

- The external interface is one execute_task_run operation with two in-process command variants: Open and Resume. No persisted command schema or new CLI option is introduced. 为什么要有它：一个用例只能有一个正常入口，内部类型足以表达当前差异。
- Open receives a bound execution configuration and Task Intent; it derives Task Authority and initial Binding from verified host inputs. 为什么要有它：调用者和模型都不能把参数当成新增权限。
- Resume receives the existing Task Run identity and exact current resume reference; it never discovers the latest directory or allocates a new identity. 为什么要有它：恢复必须继续原预算、Effect 和证据链。
- The result is either Suspended with the exact resume reference or Completed with one Run Outcome reference and one Evidence Bundle reference. 为什么要有它：可恢复中断不能伪装成失败，完成结果也不能由多个 adapter 重写。
- The Task Run module owns open-versus-resume selection and terminal finalization. The Task Agent execution module owns the technical state machine; the recovery journal owns checkpoint reconciliation and idempotency. 为什么要有它：所有权应集中但不把所有实现塞进一个函数。
- Action execution crosses one internal Effect admission seam. Authority, Binding, source preimage, version and stable operation identity are verified before the Effect; the Effect owner records a Receipt before creating an Observation. 为什么要有它：工具可用和 Action 请求都不是副作用授权或执行事实。
- A Task Run that fails before Task Authority or the edit Binding registry is created still completes with its Intent, Authority status set to not established, and Binding status set to source-revision-only; it does not invent missing edit Binding references. 为什么要有它：失败也属于Task Run，Evidence Bundle必须保留已建立的源码Binding以及尚未建立的权限事实。
- Action extensibility remains behind the seam through internal handlers; handler registration is not part of the external interface. 为什么要有它：未来 Action 不应扩大每个调用者必须理解的 interface。
- DSH and pilot remain two adapters. DSH handles human-facing command encoding and inspection; pilot handles frozen batch conditions, ordering and measurement. 为什么要有它：两个真实调用场景证明 seam 有价值，合并 adapter 只会制造更大的条件分支。
- Product graph behavior may remain behind a local adapter, but it no longer owns a parallel public Task Run entry or a second technical outcome projection. 为什么要有它：大文件不是首要问题，重复生命周期所有权才是接线成本来源。
- The first slice ends at Patch Candidate, public check, interruption recovery, Run Outcome and Evidence Bundle. Delivery and deployment are not implied. 为什么要有它：候选交接与正确交付是不同结果，首条切片不能扩大主张。
- Expand, migrate and contract work may land sequentially, but the architecture item remains incomplete until the compatibility interfaces and their interface-specific tests are deleted. 为什么要有它：expand-only 兼容层会形成第三条长期运行链。

### Required deletion list

- Remove pilot calls that separately prepare a task and build product command arguments. 为什么要有它：pilot 必须消费同一 Task Run interface，不能编排正常生命周期。
- Remove the normal Task Agent v2 path that publicly wraps product dispatch through a second command entry. 为什么要有它：正常执行只能有一个外部 seam。
- Remove the task-advance branch that starts a missing v2 task; Open and Resume must select through the Task Run interface. 为什么要有它：启动与恢复分支不能在另一个 shallow interface 中继续增长。
- Remove direct use of the recovery entry from the normal DSH and pilot path; Resume invokes the existing recovery implementation behind the Task Run seam and consumes its authoritative references. 为什么要有它：恢复不应在正常链中产生第二个结果所有者。
- Remove local duplicate assembly of the Task Agent technical outcome; it passes the unique outcome and evidence references. 为什么要有它：最后一次字段复制会重新引入结果漂移。
- Remove pilot logic that executes a Task Agent plan and then interprets the run result independently. 为什么要有它：评测 adapter 不能成为第二个产品实现。
- Remove tests that exist only for deleted interfaces after equivalent external-seam coverage exists. 为什么要有它：旧测试会把已删除的 interface 继续冻结成隐式合同。

## Testing Decisions

- The confirmed external seam is execute_task_run; tests invoke behavior only through DSH Open/Resume. Deterministic provider, evaluator and Effect-interruption adapters may be injected behind the seam, but tests do not call or assert the Task Agent owner, Action handlers, recovery journal or checkpoint layout directly. 为什么要有它：interface 是测试面，同时受控中断需要一个可替换的内部Effect adapter。
- The tracer bullet opens one repository Task Run, reads source, executes one Action, records one Observation, forms a Patch Candidate, runs the public check, interrupts after an Effect Receipt, resumes through the same interface, and returns one completed result. 为什么要有它：只有完整纵向切片能证明正常入口、Effect 和恢复真的接通。
- The deterministic provider and evaluator are existing local-substitutable adapters; the test does not require a live model or Docker. 为什么要有它：架构行为应在 CI 中稳定复现，真实模型收益属于 T1/T2/T3 队列。
- The test asserts stable run/request identity, one execution of the interrupted Effect, no duplicated cumulative resource charge, candidate/check identity, one Run Outcome and one Evidence Bundle containing Intent, Authority, Binding, Receipt, Observation, checkpoint and outcome references. 为什么要有它：这些是 deep module 的外部不变量，而不是文件布局断言。
- Existing permission, source-preimage, forged Receipt and superseded checkpoint counterexamples remain green. 为什么要有它：deepening 不能通过减少安全验证换取窄 interface。
- Completion requires the focused tracer test, affected regression suite, Ruff fatal check, Standards review, Spec review and the full remote CI workflow. 为什么要有它：本地单题成功不能证明旧路径已删除或集成入口仍工作。

## Out of Scope

- Legacy findings, interrupted-patch and runtime continuation paths are not migrated in this slice. 为什么要有它：它们不属于首条 repository Task Agent v2 tracer bullet。
- Delivery, deployment, independent business acceptance and correct-delivery claims are not added. 为什么要有它：Run Outcome 与正确交付的验收范围不同。
- T1 Docker acceptance, T2 real-model contrast and the T3 authority-equivalent permission experiment remain deferred. 为什么要有它：架构插队不能改写原队列的真实状态。
- The work does not split the full local entry module into smaller files merely to reduce line count. 为什么要有它：文件拆分本身不增加 module depth 或 locality。
- The work does not add speculative Action types, event buses, persisted schemas or CLI parameters. 为什么要有它：首条切片只实现已确认用例。
- Existing routing, strategy-preparation and main-CLI callers outside the six-module architecture scope may retain lower-level preparation or recovery helpers, but those helpers cannot remain a normal DSH/pilot Task Run path. 为什么要有它：删除范围必须真实，不能为了全局改名突破本次限定模块。

## Further Notes

The fixed starting point is commit 3240c27699619aaafcb4c55dfc55d193a5d8c949. Historical Task Runs and evaluation attempts remain immutable and are not restarted. After the contract ticket deletes the compatibility interfaces and CI is green, stop this architecture chain and reselect T1, T2 or T3.
