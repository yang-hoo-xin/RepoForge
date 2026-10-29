# Independent measurement audit: navigation-stagnation ab-v2

## Conclusion and bounded technical decision

On the single development-exposed task `restsharp__restsharp-2241`, B1 received the navigation feedback in its actual ninth HTTP request, but its next three completed responses chose `source_read`, `source_read`, and `source_search`. All three executed while read allowance remained. The preregistered primary behavioral endpoint therefore failed. Neither arm produced a candidate. This is a supported negative result for this advisory feedback on this task/configuration; it does not establish that the model generally cannot repair software.

Do not adopt or advertise this feedback as a demonstrated remedy for action stagnation. It may remain diagnostic telemetry, but has no demonstrated repair benefit from this experiment. Close this registered comparison without tuning the threshold, extending the budget, or rerunning this task for a positive result. Return to the existing queue rather than creating another experiment here. 为什么要有它：the registered negative-result stopping condition bounds this recommendation and prevents outcome-driven repetition.

## Method and identity

Read `docs/agents/run-evidence.md` and `docs/agents/evaluation.md`; started each arm's review with the shared evidence entry and its own `plan.execution_config_ref`. B0's detailed audit is in `b0-measurement-audit.md`. For B1 the registered execution configuration is `C:/tmp/rfn2241v2/B1/runs/.dsh-inputs/dsh_1d07d6daa6004163bba32f162a2902ca/execution.json`, SHA-256 `53ca2c2434082d1e4e01f0b73f5ec79259895bd1b20be8cb6143e12140f6e70e`.

Pair: `C:/tmp/rfn2241v2/pair.json`, SHA-256 `dd190b32339e0e815b4eb9c990d1c4b175251c62d73153de830b78fe6bc978f1`.

Both arms use the same common platform commit `3569e61bf541288a02485f93b5cc43341d1f5c3a`; the registered ablation changes only `navigation_only = navigation >= NAVIGATION_STAGNATION_THRESHOLD` to `navigation_only = False` in B0. Both effective worker policies have 16 model reservations, 12 read/search operations and 262144 observation bytes. Captured HTTP options agree with the common model configuration: context 65536, output 4096, seed 0, temperature 0. The fixed sequential order is B0 then B1; machine/cache/order effects are not claimed eliminated.

The independent check recomputed completed-response usage from raw provider counts, classified execution from tool receipts, and counted reservations from their originals. These agree with `observed-pair-summary.json`; all 60 path/hash references in that projection verified. Actual provider-body references and receipt-origin references were also verified. This does not turn that projection into a complete whole-task ledger.

## Observations

| Measurement | B0 | B1 |
| --- | ---: | ---: |
| Run ID | `dsh_a31162dcceab4c349922da4734e1782d` | `dsh_1d07d6daa6004163bba32f162a2902ca` |
| Reserved / completed model calls | 16 / 15 | 15 / 14 |
| In-flight call with unknown response and usage | 1 | 1 |
| Executed searches / source reads | 6 / 6 | 5 / 7 |
| Rejected navigation requests | 3 | 2 |
| Candidates / checks / experiments | 0 / 0 / 0 | 0 / 0 / 0 |
| Owned-process duration | 900.062 s | 900.062 s |
| Outer task elapsed time | 901.031 s | 900.984 s |
| Known input tokens | 330966 | 305641 |
| Known output tokens | 848 | 822 |
| Known cached input tokens, included in input | 193132 | 160736 |
| Changed source / forbidden source paths | 0 / 0 | 0 / 0 |

Both runs end at the parent hard deadline, with sealed parent process episodes and no reported owned-process residuals. Broker-descendant cleanup is not established by generic job ownership. The two incomplete provider calls must not be converted into zero-token calls or completed failures.

B0 requests 13–15 are new read/read/search requests refused before execution; feedback is delivered in requests 14–16. B1 requests 13–14 are new read/search requests refused before execution; feedback is delivered in requests 14–15. These are not replays. Working-state completed-return counts include refusal feedback and are not counts of executed navigation.

B0's actual HTTP inputs keep navigation feedback disabled. B1 request 8 has navigation count 7 and no guidance; request 9 has count 8, `navigation_only = true`, `triggered = true` and the complete registered advisory guidance. Responses 9–11 still choose navigation. Response text describes further reading rather than a non-navigation proposal. The primary failure precedes the exhausted-read refusals, so those later refusals do not explain away this endpoint.

## Five-category reporting

| Category | Report and limitation |
| --- | --- |
| Correct delivery | B0 0/1 and B1 0/1 verified deliveries among started attempts; observed difference 0. These are two attempts on one exposed task, not two independent tasks. No candidate means independent business acceptance and same-candidate delivery were not executed; their raw fields remain null. Do not report a grader failure or a measured repair-success advantage. |
| Human input and waiting | Both tasks hit their deadline; process and outer durations are shown separately. Human active minutes are unknown, not inferred from elapsed time. No quality-matched successful tasks exist, so user-effort or faster-delivery savings cannot be estimated. Audit, preparation and platform work remain separate from task-model execution. |
| Unit-success cost | Known sums across both arms are 636607 input, 1670 output and 353868 cached input tokens, plus two unknown in-flight calls. Cached input is included in input. No extra auxiliary generation type is observed in either manifest; trace wrappers are not extra calls. Whole-task money, hardware, and human cost remain unknown. With zero successes, finite cost per success is undefined. Lower B1 known tokens do not prove efficiency: it completed one fewer response and has an unfinished call. |
| Repair after public counterexample | For both arms, business-error K = 0 and compilation-error K = 0, k = 0, rates undefined. No candidate/check means no delivered business or compilation counterexample. Budget refusals are process feedback and are counted separately. |
| Judgment and experience | Feedback delivery succeeded; the specified behavioral response did not occur. Strategy experience was off, so adoption, transfer and negative transfer were not measured. This result concerns one feedback mechanism on one exposed task, not general autonomy, competitive evaluation performance, or long-term learning. |

Raw provider usage fields (`prompt_eval_count`, `eval_count`, `prompt_eval_cached_count`) match the persisted normalized usage for every completed response. The known noncached input components are B0 137834 and B1 144905. Both final generation records show transport started and no received response/usage. There are no independently established zero costs for those pending calls.

## Evidence gaps and references

The shared evidence views are partial because timeout prevented final `run_result.json`, technical `execution.json`, and `loop-result.json` from being written. Their zero-request cost projections are missing aggregation, not zero usage. Individual trace/HTTP/response/receipt originals and parent outcome evidence support the counts above; child-owned final outcome completeness remains a known limitation.

B1 run-relative references are under `C:/tmp/rfn2241v2/B1/runs/dsh_1d07d6daa6004163bba32f162a2902ca/`:

| Original | SHA-256 / observation |
| --- | --- |
| `run_manifest.json` | `e86eea2cadefcd4f6bc168c5fcfeffcb0f12570a7c3f6f8d25cdc3803727f4fb`; generation records and body references |
| `a/task-agent-execution/model_trace_bodies/caea15e6fee345ff.provider_request.json` | `0a83333839bd46778f3ad93ad58c9725853747bb4275c1e7f6815dd7e40effb6`; ninth actual HTTP body; decode message content for `/current_task_state/working_state/stagnation` |
| `a/task-agent-execution/response-009.json` | `76c8bde7579d37bff2d6e78855e478540676dc409c9a04852729c150d67a5ff0`; source read |
| `a/task-agent-execution/response-010.json` | `1a5d546187510a4fd0636ededdf23bc6d5bb7a7a0b85182e7542d6e2a7dc62a0`; source read |
| `a/task-agent-execution/response-011.json` | `bae236bed3923945c6838f7d5556f4cc2655aa271a05af7c445dc4a01970f82f`; source search |
| `a/task-agent-execution/model_trace_bodies/a4098eb3616047e7.provider_request.json` | `557dc8752546e25877a12035879173a71434f835737287ecdb704b54c959ec55`; fifteenth actual request, response unknown |
| `C:/tmp/rfn2241v2/B1/result.json` | `32f7d3676d8ae16e3acb26eb66f232cc779c448371c716d0eed42752e3310556`; outer result and sealed episode refs |
| `C:/tmp/rfn2241v2/B1/execution/process.json` | `f93662bf3552e6fac7e14d52623fdd05cab0ac868c01b01a38d134fca99a93e9`; timeout and owned-process cleanup |

No model execution, historical replay, task rescue, acceptance alteration or hidden-answer inspection was performed by this audit. The original queue remains: T2 broader safe-input correction, T1 full two-repository acceptance, and child-owned graceful stop; their next order is a separate decision after this bounded comparison closes.
