# Issue tracker: GitHub

RepoForge issues, specs and implementation tickets live in GitHub Issues for `yang-hoo-xin/RepoForge`. Use the `gh` CLI for operations when available.

## Conventions

- Create: `gh issue create --title "..." --body-file <path>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,labels,assignees`.
- Comment: `gh issue comment <number> --body-file <path>`.
- Label or assign: `gh issue edit <number> --add-label "..." --add-assignee "..."`.
- Close: `gh issue close <number> --comment "..."`.

Infer the repository from the current Git remote. Use temporary body files for multiline text so shell interpolation cannot change issue content.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill meanings

- “Publish to the issue tracker” means create a GitHub issue.
- “Fetch the relevant ticket” means read the issue body and comments.
- `/to-spec` creates the parent specification issue.
- `/to-tickets` creates agent-ready child issues in dependency order.

## Wayfinding operations

- A map is one issue labelled `wayfinder:map`.
- Decision tickets are child issues and use one `wayfinder:<type>` label.
- Create all child issues before adding blocking edges.
- Prefer GitHub native sub-issues and `blocked_by` dependencies; when unavailable, use explicit `Part of` and `Blocked by` lines.
- Claim a frontier ticket by assigning it before work begins.
- Resolve a ticket by commenting with the decision, closing it, and adding a linked one-line gist to the map.
