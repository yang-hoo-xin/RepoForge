# Run evidence

Read this when inspecting a task run, attributing a failure, assessing capability
from execution evidence or selecting the next experiment. Ordinary code edits do
not require historical-run investigation. 为什么要有它：按任务触发取证，避免把历史扫描变成所有开发的前置流程。

1. Establish the absolute registered execution configuration and business Run ID.
   Reuse that identity while applicable; resolve conflicting identities from their
   registrations, not by choosing the latest directory.
   为什么要有它：目录的新旧不能证明它属于当前讨论的任务。

2. Use the shared read-only entry with the project's configured Python environment:

   ```text
   python -m scripts.dsh_run_request --execution-config "<absolute registered config>" --inspect-run "<business run ID>" --view evidence --format json
   ```

   Use `--format markdown` for the human-readable view. Follow returned references
   to the actual requests, responses, patches and checks relevant to the question.
   Report missing or ambiguous facts as such. Completion means the conclusion is
   linked to inspected originals, with remaining gaps identified.
   为什么要有它：汇总是调查入口，不能替代实际执行原件或把缺失项补成确定事实。

3. Keep the developer evidence view out of product-model inputs. Public feedback
   and experience use their existing visibility and authorization boundaries.
   These natural-language working agreements do not create a registered slash
   command or an automatic execution trigger.
   为什么要有它：共享查询方便开发调查，但不增加产品模型的可见范围或执行权限。
