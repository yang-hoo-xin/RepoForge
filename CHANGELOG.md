# Changelog

This file records user-visible RepoForge changes. Release claims remain bound to
the corresponding Git revision, tests, and evidence manifest.

## [0.1.0] - Unreleased

### Added

- Evidence-bound Coding Agent evaluation and execution across current-source
  investigation, candidate generation, real checks, traceable results, and
  controlled delivery.
- Local model execution through Ollama plus configurable external-provider
  adapters.
- Versioned artifacts, execution traces, interrupted-task continuation, and
  explicit result states that keep failures and unknown outcomes visible.
- Explicit repository profiles for supported SDK-style C# projects, including
  declared editable files, build inputs, test commands, reports, and expected
  artifacts.
- FastAPI, worker, Redis, MCP, dashboard, and Prometheus integration around the
  shared application modules.

### Experimental continuous-task additions

- Optional v2 task execution retains model statements and public observations
  across capacity-based history projection and recovery. These are attributable
  working records, not a proven semantic summary of the model's understanding.
- Path-based candidate edits bind authorized existing files atomically while
  preserving exact source preimages and the older binding-ID interface.
- Public process evidence can be sealed after a bounded failed attempt, without
  resuming it or inventing a native successful terminal result.
- Source-checkout tooling can prepare, preview, and perform one version-bound
  extraction from selected public episodes with explicit omissions and costs.
- Controlled tests cover the normal interfaces; the latest real repair and
  multi-episode learning results remain negative. No general repair or transfer
  benefit is claimed, and these research paths remain opt-in.

为什么要有它：新增可选机制需要进入版本说明，但受控接线和真实负结果不能被表述为自主修复或学习收益。

### Release boundaries

- This is an initial-development release; public interfaces may change.
- Verified examples do not establish arbitrary-repository support or a general
  autonomous repair success rate.
- Developer-assisted results, deterministic fixtures, environment
  qualification, and autonomous model results are reported separately.
- Autonomous C5 repair, live Docker restart/replay, and external Serilog model
  delivery are post-v0.1 work and are not release blockers.

[0.1.0]: https://github.com/yang-hoo-xin/RepoForge/releases/tag/v0.1.0
