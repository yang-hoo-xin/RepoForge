# B0 evidence and measurement audit

Run: `dsh_20a78f8b74d5454bae86319f28c1aaae`. Read-only review after completion; no product execution, frozen-input change, or task technical rescue performed by this audit.

## Conclusion

This pair cannot estimate navigation-feedback benefit. B0 exhausted a three-read/search quota; the B1 intervention requires eight executed navigation returns. The planned intervention is unreachable under the effective common resource policy. This is a missed pre-run experiment compatibility check, not evidence that the intervention or autonomous repair capability failed. B1 remains unstarted. Preserve this B0 attempt and its costs; do not rerun it under the same identity.

B0 also exited with `DSH_TASK_RUN_OUTCOME_INVALID` after the technical loop recorded `TASK_AGENT_OBSERVATION_BUDGET_EXHAUSTED`. These are distinct failures: a resource stop inside the loop and a platform outcome-boundary failure outside it.

## Evidence entry and original references

Started from the shared `scripts.dsh_run_request --inspect-run ... --view evidence --format json` entry using the execution configuration below. Its evidence view is partial: it reports outside-selected-run reference restrictions and a reference-edge limit. Referenced run-local originals were inspected directly; no missing evidence was treated as zero.

Base directory for relative references in this audit: `C:/tmp/rfnav2103-20261029-v1/B0/runs/dsh_20a78f8b74d5454bae86319f28c1aaae/`.

| Original | SHA-256 | Relevant location |
| --- | --- | --- |
| `../.dsh-inputs/dsh_20a78f8b74d5454bae86319f28c1aaae/execution.json` | `e3c0f8bea609b340c0bed63b9b11c9fa291a4ad9ee6897e8fc59aa5cffbceaa7` | Registered execution configuration |
| `repository-worker.json` | `43cb7835cdc991a8c7bb64d5f07464efdf594ddf03a7119f636a7381d8101da0` | `/policy/max_tool_rounds` = 3; `/policy/max_observation_bytes` = 32768; `/policy/max_total_model_turns` = 16 |
| `a/task-agent-execution/loop-result.json` | `0d1dc1a40b8487a101084f3d511b4f052912ea27bb603d6fa16c2225acb5e586` | `/events/13/read_budget/limit` = 3; `/events/13/read_budget/remaining` = 0; `/read_units` = 3; `/observation_bytes` = 32158; `/termination/code` |
| `a/task-agent-execution/execution.json` | `e3b9f559a2cdb238ed2348243745b265d9ebc4a83ad147e7e265ebd1ccca407b` | `/provider_turns`, `/candidate_ref`, `/check_ref`, `/budget_after` |
| `run_manifest.json` | `23a2b14ae05cc6473f8c03be6cd906b6f3baf1391b8531a310208eae1d4ccee1` | `/execution_observations`, `/model_call_count` |
| `run_result.json` | `7b0650671df041b98da70c8f950f2c415d371f50fa555893dc905bd8c97537b4` | `/task_agent_outcome`, `/model_call_budget` |

The effective quota originates in the frozen B0 platform: `src/mod_agent/local_entry.py:7995` constructs `PatchAgentLoopPolicy` without overriding the read or observation defaults; `src/mod_agent/proposal/patch_agent_loop.py:100` defaults `max_tool_rounds` to 3 and line 89 defaults observation bytes to 32768. `src/mod_agent/task_agent_execution.py:541` charges those limits when the policy is not meter-only. The task uses `task_total_only`, not meter-only. The Task Contract's referenced `a/v2/resource/1c5211f51b35eade0767c874a9f2a67687d663bbd939e083aa05ac1bdaabc27a.json` records 16 provider reservations and deadline, but does not expose the separate three-read default. Thus the quota is an effective frozen-platform constructor default materialized in `repository-worker.json`, not an extra user-authorized task restriction inferred by this audit.

The frozen B1 `src/mod_agent/task_agent_v2_working_state.py:161` requires `navigation >= 8`; its guidance at line 168 explicitly counts executed search/read returns and excludes replayed call IDs. Three possible executed navigations cannot reach eight.

## Response and action classification

All 11 responses are complete real provider responses. Request, actual HTTP-body, and response reference hashes were verified for all 11 turns. The 9 origin references (arguments/request/response) across the 3 tool receipts also verified. Provider responses and receipts agree on call identity.

| Turns | Result | Count |
| --- | --- | ---: |
| 1, 2, 3 | `source_search` actually executed | 3 |
| 4, 5, 9 | New `source_search` rejected for exhausted read budget; feedback appears in next request | 3 |
| 6, 8 | New `source_read` rejected for exhausted read budget; feedback appears in next request | 2 |
| 7 | No tool call; plain response causes output-format correction | 1 |
| 10 | New `history_read` rejected for exhausted read budget; feedback appears in request 11 | 1 |
| 11 | New `source_search` rejected; rejection cannot fit remaining observation allowance and is retained in `/unadmitted_tool_results/0` | 1 |

Therefore: 10 proposed tool calls, 3 executed, 7 rejected before execution. Of the 7, 6 rejection responses were delivered and the last was retained but not delivered. There are 6 rejected source-navigation calls (4 searches, 2 reads), plus 1 history-read rejection. None is a replay. Only 3 actions count toward the navigation threshold, not 11. There is no candidate, probe, check, or independent business acceptance. Later model statements acknowledge the exhausted budget but still request reads; this is an observed behavior under this constrained configuration, not a treatment comparison.

## Five measurement categories

| Category | Defensible report |
| --- | --- |
| Correct delivery | B0: 0 verified correct deliveries / 1 started attempt; candidate 0, changed source 0, independent acceptance not run. Raw `correct_delivery` and `independent_business_result` remain null. B1: 1 planned slot, not started because comparison invalidated; no invented outcome or B0/B1 difference. Preserve both planned slots and their statuses. |
| Human input and waiting | B0 task wall time 631.75 s (`B0/result.json`); owned child process 631.5 s (`B0/execution/process.json`), exited code 1, not timed out. Human active minutes unknown; do not infer from wall time. Environment qualification and this audit belong to platform development/independent review, not task technical repair. No quality-matched efficiency comparison exists. |
| Unit successful cost | All 11 recorded provider responses: 224385 input tokens, 645 output tokens, 152539 cached input tokens. Cached tokens are a component of input, not an added token total. Raw provider counts independently sum to these values. All 11 `generation` observations are `patch.generate`; no additional auxiliary model generation is observed in this run, and pre-loop budget consumed is 0. Shared evidence still reports whole-task cost completeness false; missing accounting outside these records, money, hardware cost, and human minutes remain unknown. With zero verified successes, no finite cost per success is calculated. |
| Repair after counterexample | Business-error K = 0 and compilation-error K = 0: no candidate was checked and no public business/compile counterexample was delivered. k = 0 for both, rates undefined rather than 0%. Budget/format rejections are separate process feedback, not business counterexamples. |
| Judgment and experience | No B1 run, no intervention exposure, no strategy trial or transfer estimate. The experiment identifies a resource-policy compatibility defect and outcome-boundary failure. It cannot support a positive or negative causal claim about navigation feedback, strategy adoption, or transfer. |

Accounting cross-check: raw response `prompt_eval_count`, `eval_count`, and `prompt_eval_cached_count` match recorded usage for every turn. The manifest's 11 generation records, 11 technical-loop reservations and 11 persisted actual responses agree. Do not double-count `task.agent.turn` and `model.request` trace wrappers as additional model calls.

External failure evidence: `C:/tmp/rfnav2103-20261029-v1/B0/execution/stderr.bin` SHA-256 `55255485499b83d73b1621de631fc2f50064f338841d00e4f823e47856b6cf51` ends at frozen `scripts/dsh_run_request.py:1108` with `DSH_TASK_RUN_OUTCOME_INVALID`. `B0/execution/process.json` records no owned process residuals; broker-descendant cleanup is explicitly not established by generic job ownership. No claim is made that every external resource has been reclaimed.

At audit time B1 contains preparation artifacts but neither `started.json` nor `result.json`. This report introduces no new execution permission or modified evaluation rule.
