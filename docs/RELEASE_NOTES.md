# v0.1 Release Notes

## 2026-10-13 maintenance update

Default entry and per-launcher seed isolation fixed; runtime failure exit status,
transport limits, truncated-response rejection, stage/API usage records and full
offline CI added.
The experimental dynamic hierarchy is intentionally not merged into this release.

## Included

- Semantic frontend with explicit Event Graph schema and DSO / supply-chain examples in its prompt.
- Event-graph contract checks for evidence references, confidence normalization and field shape.
- Control, Knowledge, Matcher and Executor runtime.
- Curated formal Rule and Knowledge databases for finance and inventory / supply-chain demonstrations.
- One command-line demo entry point and offline regression tests.

## Excluded intentionally

- API keys and `.env` files.
- Model weight caches.
- Previous experiment runs, training checkpoints, traces and backup archives.
- Original Git history.

## Not a claim

This release is not a claim that the system can solve arbitrary domains or that the current Rule network is complete. It is a reproducible v0.1 demonstration of the present two-domain reasoning pipeline.
