# ASE development inputs

- `runtime/`: 48 evidence-only cases and their split. Both method examples are excluded.
- `examples/`: two source-grounded method examples, their exact source excerpts,
  original 50-case source cohort and preparation provenance. These are method inputs,
  not model outputs or independent evaluation cases.
- `evaluation/`: 50 original gold records, used only by offline scoring with the
  fixed 48-case selection. These files are never passed to inference workers.
- `evidence/`: 558 collected source items and 23 source image references. The
  release bundle changes only image paths; source text and image bytes are unchanged.
- `experiment_two.json`: explicit source input paths, hashes, ordered IDs and model
  setting. It has no dependency on an existing prediction or run manifest.
- `release_manifest.json`: original-to-release paths and hashes; historical absolute
  paths are provenance only. Current unversioned sources are not historical snapshots.

All 48 cases have been inspected during development; do not call this an unseen
holdout. Run commands and evaluation boundaries are in the Benchmark guide.
