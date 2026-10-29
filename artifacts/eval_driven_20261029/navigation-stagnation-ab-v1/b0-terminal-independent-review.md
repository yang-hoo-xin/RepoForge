# B0 terminal review

Decision: apply the preregistered `platform_failure` rule and leave B1 unstarted. Preserve B0 without retry. 为什么要有它：正常入口的终态转换已实际异常，不能在看到结果后把平台故障缩窄为进程超时或启动失败。

Reviewed run: `dsh_20a78f8b74d5454bae86319f28c1aaae`, under `C:/tmp/rfnav2103-20261029-v1/B0`.

The shared evidence entry was invoked with the plan's registered execution configuration, the exact run ID, `--view evidence --format json`, using the repository `.venv/Scripts/python.exe`. It returned `status=partial`, `complete=false`, and a hash-verified `run_result.json`. Its 24 reference-access issues are retained as incompleteness; this review does not infer that each issue is a platform defect. The first invocation with system Python failed before inspection because that interpreter lacks `langgraph`; the repository interpreter succeeded without model calls or run recovery.

Two separate findings:

- Model/task result: `task_agent_outcome.status=blocked`; `loop-result.json` records `TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED`, 11 model turns and 11 of 16 model reservations consumed, 32,158 admitted observation bytes, no candidate and no check. These are retained negative observations. The stop label alone does not establish why the budget charge failed; the loop maps exceptions during observation charging to that label.
- Platform result: `execution/process.json` records exit code 1, no timeout and no owned residual processes. `execution/stderr.bin` records `DSH_TASK_RUN_OUTCOME_INVALID` at the frozen `scripts/dsh_run_request.py:1108`. The task's top-level `run_result.json` has `current_stage=blocked` but no `status`. `_completed_task_run` at lines 1104–1108 requires a nonempty top-level string status. Frozen `src/mod_agent/local_entry.py:6217` (`_persist_graph_terminal_result`) guarantees `current_stage`, serializes that summary, and writes `status` to the manifest rather than guaranteeing it in the result. Both preserved outcome summaries have the correct run/request IDs, so the missing result status is independently sufficient to trigger this rejection.

The parent later sealed a process outcome; that does not erase the child's normal-entry exception. `result.json` has no delivery process, no changed source paths, and unknown independent-business/correct-delivery fields. No business acceptance was executed, and no B0/B1 improvement claim is available.

Evidence references:

- `C:/tmp/rfnav2103-20261029-v1/B0/result.json`
- `C:/tmp/rfnav2103-20261029-v1/B0/execution/process.json`
- `C:/tmp/rfnav2103-20261029-v1/B0/execution/stderr.bin`
- `C:/tmp/rfnav2103-20261029-v1/B0/runs/dsh_20a78f8b74d5454bae86319f28c1aaae/run_result.json` (SHA-256 `7b0650671df041b98da70c8f950f2c415d371f50fa555893dc905bd8c97537b4`)
- `C:/tmp/rfnav2103-20261029-v1/B0/runs/dsh_20a78f8b74d5454bae86319f28c1aaae/a/task-agent-execution/loop-result.json` (SHA-256 `0d1dc1a40b8487a101084f3d511b4f052912ea27bb603d6fa16c2225acb5e586`)
- `C:/tmp/rfnav2103-20261029-v1/B0/platform/scripts/dsh_run_request.py:1104`
- `C:/tmp/rfnav2103-20261029-v1/B0/platform/src/mod_agent/local_entry.py:6217`
- `intent.json`, `decision_rule.platform_failure`

Review scope: read-only investigation plus this review artifact. No product changes, task-answer access, model invocation, restart, or B1 launch. The existing T2/T1/owner-level-stop queue remains unchanged; this failed comparison closes at the documented phase boundary for re-selection rather than automatically opening another repair chain.
