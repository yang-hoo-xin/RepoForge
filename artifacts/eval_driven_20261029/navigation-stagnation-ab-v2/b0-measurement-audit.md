# B0 measurement audit — ab-v2

Read-only audit of completed B0 `dsh_a31162dcceab4c349922da4734e1782d`. No frozen inputs changed, no model launched, no advice supplied to B1, and no hidden acceptance answers inspected. This is a single-arm measurement report; B1 was still running when assigned, so no paired conclusion is made.

Identity: `C:/tmp/rfn2241v2/pair.json` SHA-256 verified as `dd190b32339e0e815b4eb9c990d1c4b175251c62d73153de830b78fe6bc978f1`. Started with the shared evidence entry using `B0/plan.json`'s execution configuration `B0/runs/.dsh-inputs/dsh_a31162dcceab4c349922da4734e1782d/execution.json` (SHA-256 `b3468e37091fe8cfe684b9dc09d52fe18ee0bb2f3c1c5d62e6843652376d2b3f`).

## Verified observations

- Effective worker policy: 16 model reservations, 12 read/search operations, 262144 observation bytes. The actual provider request bodies carry maximum 16 model calls. Requests 14–16 deliver exhausted-read feedback with limit 12, used 12, remaining 0.
- Baseline navigation intervention is disabled in all 16 actual captured HTTP request bodies: `/messages/*/content` decoded as JSON contains `/current_task_state/working_state/stagnation/navigation_only = false` and `guidance = null`. Request 9 records eight actual navigation returns, while retaining those disabled values. The intervention is therefore not accidentally active in B0 after its threshold.
- 15 complete provider responses and 16 reservations/actual request captures. Turns 1–12 execute six searches and six source reads. Turns 13 and 14 request source reads, turn 15 requests a search; all three are rejected before execution for exhausted read allowance. Their feedback appears in requests 14, 15 and 16 respectively. Distinct call IDs and matching receipts establish that these are new requests, not replays.
- No candidate, check or experiment was executed; no source paths changed. `completed_returns = 15` in request 16 includes rejection feedback and must not be reported as 15 executed navigation operations.
- The sixteenth generation has `transport_started = true`, `response_received = false`, `usage_available = false`. Its actual HTTP body exists, but response and usage remain unknown after the parent timeout. Do not count it as a completed response or zero-cost call.
- Owned process time is **900.062 seconds**, exit 124, timeout true. The outer result records **901.031 seconds** task wall time including controller overhead; these are different measurements. No owned process residuals are reported; broker-descendant cleanup is explicitly not established by generic job ownership.

## Usage and five measurement categories

All 15 persisted raw provider responses independently sum to **330966 input tokens, 848 output tokens, 193132 cached input tokens**. Cached input is part of input, not additional input: the known noncached component is 137834. Persisted response usage agrees with raw `prompt_eval_count`, `eval_count`, and `prompt_eval_cached_count` for every completed response. Recorded reasoning tokens total zero for those responses only. Unknown sixteenth-call usage is additional and remains unknown.

The manifest has exactly 16 `patch.generate` observations (15 completed, one running at capture), with no other generation types observed. `model.request` and `task.agent.turn` are wrappers, not additional calls. All 32 manifest-linked request and provider-body references verified by SHA-256; all 36 arguments/request/response origin references across the 12 execution receipts also verified.

| Category | B0-only report |
| --- | --- |
| Correct delivery | 0 verified deliveries / 1 started attempt; raw `correct_delivery` and independent business result remain null because independent acceptance/delivery did not occur. No paired difference yet. |
| Human input and waiting | Report 900.062 s process and 901.031 s task elapsed separately. Human active minutes and comparable-quality savings are unknown. Audit/platform preparation are not task-model success. |
| Unit success cost | Known token subtotal above plus one unknown in-flight call. Whole-task money/hardware/human cost unknown. Zero successes means no finite unit-success cost. |
| Repair after counterexample | Business and compilation K = 0, k = 0, rates undefined. Three budget refusals are process feedback, not business/compile counterexamples. |
| Judgment/experience | Navigation baseline is confirmed disabled. No B0/B1 benefit, negative-transfer, or strategy-adoption claim can be made before B1 audit and comparison. |

The shared evidence view is partial: `run_result.json` is missing after the timeout; final `execution.json` and `loop-result.json` were not written. Its `cost.requests = 0` and known sums zero are an incomplete projection, **not zero actual usage**. The verified manifest, individual requests/responses, receipts, and parent result above are necessary to report this interrupted run accurately. Whole-task cost completeness remains false.

## Original references

Run-relative paths below are under `C:/tmp/rfn2241v2/B0/runs/dsh_a31162dcceab4c349922da4734e1782d/`.

| File | SHA-256 / relevant pointer |
| --- | --- |
| `repository-worker.json` | `b4b0f49c255520320e99ae5bd0a624c0e1c8628edb8c7c5709bf2c1d69617fe4`; `/policy` |
| `run_manifest.json` | `4710dca29f08baad4c5f440126795ff7ce3a82fe4363b3fc460bbdff7a71813f`; `/execution_observations` |
| `a/task-agent-execution/request-009.json` | `1b5961f47742f17949825d1b121af8666b3f425e22bb8ca9fffb25ece68e56db`; `/messages/2/content/current_task_state/working_state/stagnation` |
| `a/task-agent-execution/request-014.json` | `6d92c6d4f69b8f170ab3e7ba6749f309f3d0514b18b73aadcacbbf9a1f83156d`; `/tool_results/0/output/read_budget` |
| `a/task-agent-execution/request-016.json` | `66d46a8060ff157c7d584239f4c9feecc6f474c4a051639d208f46babacf48d0`; delivered rejection and final reservation context |
| `a/task-agent-execution/model_trace_bodies/546e9da54e5743c7.provider_request.json` | `2d1a95905add85a8cffb4a9349e98af38f9f5c2b9f2b5de29d46ba70e692ade1`; actual sixteenth HTTP request body |
| `C:/tmp/rfn2241v2/B0/execution/process.json` | `0c81a75bf58abb0f6a310421070ec075ebf4fc00dc7753eda5dd948f6ddea0ca`; `/wall_seconds`, `/timeout`, `/owned_process_residuals` |
| `C:/tmp/rfn2241v2/B0/execution-facts.json` | `83f20d18da7703a404247dd11b7ee2f7318598f1c9adfd4395bb18a9a59e0164`; `/task_wall_seconds`, `/changed_source_paths` |

No changes or additional experiment are authorized by this audit. B1 evidence and cross-arm comparability remain pending.
