# Domain Docs

RepoForge uses a single-context domain-document layout.

## Before exploring

- Read the root `GLOSSARY.md` when the task uses or changes RepoForge domain language.
- Read ADRs under `docs/adr/` that affect the area being changed.
- If either location is absent or irrelevant, continue without creating placeholder documents.

## Language

Use the canonical terms in `GLOSSARY.md` in issues, specifications, tests and user-facing results. When a needed concept is missing, first check whether it is a real domain concept or ordinary engineering vocabulary; add only domain concepts.

## Decisions

Record an ADR only when a decision is hard to reverse, surprising without context and based on a real trade-off. If proposed work contradicts an ADR, identify the contradiction before implementation.
