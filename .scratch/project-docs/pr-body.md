RepoForge now has a single repository document entry point and an offline searchable catalog linking specifications, ADRs, analyses, tasks and PRs. The change imports the accepted developer-preview specification, permissions decision, two review summaries and four static HTML reports, while preserving source versions and unmerged-code boundaries.

Drafts keep a stable ID through update, promotion and retirement. Promotion requires the raw draft hash and a decision reference, relocates Markdown links and retains a validated redirect. AGENTS, the maintenance skill and CI checks establish the ongoing convention. Six local task drafts are now published as Issues and replaced by redirects; the developer-preview parent is #17.

Validation: 56 tests passed, 1 Windows symlink-privilege skip; Ruff, skill format, imported-file hashes, document links, actual draft-to-formal CLI smoke and committed-checkout index verification passed. No product-model calls or runtime implementation merges were performed.

Closes #23 when merged. This remains a draft PR for user review; it does not close evaluation Issue #3 or the developer-preview implementation tasks.
