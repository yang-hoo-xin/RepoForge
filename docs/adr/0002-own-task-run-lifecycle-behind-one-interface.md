# Own the Task Run lifecycle behind one interface

Status: accepted

RepoForge will expose one normal Task Run interface, conceptually `execute_task_run(OpenTaskRun | ResumeTaskRun)`, to both the DSH and pilot adapters. `OpenTaskRun` carries a bound execution configuration and Task Intent; Task Authority is derived from host-bound configuration rather than supplied by the caller. `ResumeTaskRun` carries the existing Task Run identity and exact current resume reference. 为什么要有它：打开与恢复必须保持同一运行身份、权限和累计资源，调用者不能通过另一入口扩张权限或重开一次尝试。

The interface returns either a recoverable suspended result or a completed result containing exactly one Run Outcome reference and one Evidence Bundle reference. The Task Run module owns lifecycle selection and finalization; the Task Agent execution state machine remains its internal implementation, the recovery journal owns checkpoint reconciliation and idempotency, and Action execution crosses one internal Effect admission seam that records a Receipt before exposing an Observation. 为什么要有它：中断不是失败或第二个结果，效果事实也不能由动作请求或模型陈述推断。

## Considered options

- Always return a completed result. Rejected because a recoverable interruption would be overloaded as an exception, failure or unknown terminal state. 为什么要有它：可恢复中断需要保持同一 Task Run，而不是伪造终态。
- Let callers pass an Authority reference and expose a generic handler registry. Rejected because it widens the interface toward the implementation and makes Authority look caller-defined. 为什么要有它：扩展性应留在 deep module 内部，任务意图和工具 schema 都不授予权限。
- Merge the DSH and pilot adapters. Rejected because they are two real adapters with different responsibilities; both should share the Task Run seam instead. 为什么要有它：合并 adapter 只会集中条件分支，不会增加 module depth。

## Consequences

- The first tracer bullet must migrate opening and recovery through the same interface and delete the replaced direct entry paths; an indefinitely retained compatibility path blocks completion. 为什么要有它：expand-only 改动会形成第三条运行链，继续制造 Shotgun Surgery。
- The first slice covers the repository Task Agent v2 path through Patch Candidate, public check, interruption and recovery; legacy findings, interrupted-patch, runtime continuation and delivery remain outside this decision. 为什么要有它：首条切片要纵向完整，同时不能自动扩成全库重构。
- Tests cross only the Task Run interface and assert Task Run identity, one-time Effect execution, cumulative resources, the unique Run Outcome and the Evidence Bundle. 为什么要有它：interface 是测试 seam，内部文件布局和 helper 不应成为新的测试合同。
