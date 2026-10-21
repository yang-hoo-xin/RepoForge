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

### Release boundaries

- This is an initial-development release; public interfaces may change.
- Verified examples do not establish arbitrary-repository support or a general
  autonomous repair success rate.
- Developer-assisted results, deterministic fixtures, environment
  qualification, and autonomous model results are reported separately.
- Autonomous C5 repair, live Docker restart/replay, and external Serilog model
  delivery are post-v0.1 work and are not release blockers.

[0.1.0]: https://github.com/yang-hoo-xin/RepoForge/releases/tag/v0.1.0
