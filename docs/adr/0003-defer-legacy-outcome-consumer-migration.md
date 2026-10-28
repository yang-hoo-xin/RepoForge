# Defer legacy outcome consumer migration from the Task Run slice

Status: accepted

The normal DSH and pilot Task Run paths now share one Open/Resume interface, preserve exact Intent and Receipt identity, and return one Run Outcome with a verified Evidence Bundle. RepoForge accepts that scoped slice as complete and defers removal of legacy outcome dictionaries used by the main CLI, delivery, handoff, outcome ledger and Dashboard to an independent follow-up. 为什么要有它：继续迁移会突破原六模块范围并延长架构插队，而当前绿色 seam 已经降低下一项功能的接线成本。

## Considered options

- Expand the current architecture item to every legacy outcome consumer. Rejected because it turns one tracer bullet into a cross-product migration and delays T1, T2 and T3 again. 为什么要有它：架构收益不能以无限扩大工作链为代价。
- Pretend the legacy fields were removed globally. Rejected because those consumers still exist and remain supported. 为什么要有它：阶段完成必须准确限定范围，不能改写现有代码事实。

## Consequences

- The scoped Task Run architecture item is complete for the normal DSH/pilot path after commit `da6822c` and CI run 40649694005. 为什么要有它：完成状态需要绑定同一版本代码和远端验证。
- Legacy outcome consumer migration remains a separate deferred item and does not block returning to T1, T2 or T3. 为什么要有它：后续清理继续可见，但不再占用真实能力实验的前置路径。
- Claims must say “normal Task Run path unified,” not “all outcome formats removed.” 为什么要有它：局部架构交付不能扩大成全产品清理主张。
