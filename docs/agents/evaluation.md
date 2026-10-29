# Evaluation

Read this when executing or reporting verification for changes to agent decisions,
tools, feedback, context, retrieval, models, experience or resource policy. It also
applies when preparing a real Agent experiment or a capability claim. It does not
turn ordinary refactoring or documentation into a model evaluation.
为什么要有它：能力主张需要专门证据，普通工程检查不必承担整套评测流程。

## Scope and completion

1. Use the expected effect, baseline and verification already recorded for the
   current work. Before a new real run, establish its protocol, resource policy,
   stop conditions and return point within the existing user authorization. Once
   that registered verification is ready, execute it without waiting for another
   resume request. New findings return to the queue unless already in this scope.
   为什么要有它：保留授权内的自主执行，同时让实验有明确终点，避免修复后不断追加新运行。

2. Freeze the common conditions and identify the main tested change when claiming
   a causal improvement. Without a comparable baseline, report absolute results.
   Close the experiment when its protocol's completion or stop condition is met;
   an evidence-supported negative result is a completed experiment. It does not
   mean the long-term product goal is complete.
   为什么要有它：失败同样能支持技术决定，追求正结果不能成为延长实验的理由。

## Evidence and attribution

3. Reuse native artifacts, Trace, resource and delivery receipts. Retain attempts,
   failures, regressions, unknowns and the versions of any mid-run fixes. Treat
   hidden acceptance and its answers as protected inputs; evaluation does not
   grant broader product authority or permission to alter acceptance.
   为什么要有它：指标须能从原件复算，不能通过删失败、混版本或泄漏答案改善成绩。

4. Record external Codex root-cause advice, fixes and patch edits as technical
   assistance, separately from approval, review and environment repair. Human
   active time requires actual timing or explicit declaration, never an inference
   from waiting time. Separate task, independent-evaluation and platform-development
   costs, retaining unknowns.
   为什么要有它：外层帮助和准备成本不能隐去，模型运行时间也不能冒充人的主动投入。

## Applicable specifications

For the competitive A/B/B+ batch and its five metric categories, use the
[competitive evaluation specification](../specs/competitive-autonomous-delivery-evaluation.md)
and its approved version or amendments. For a bounded development experiment,
use that experiment's registered task record; the competitive batch's task counts,
models and deadlines do not automatically become its conditions.
为什么要有它：共用证据纪律，不等于所有诊断都必须运行完整竞争性批次。
