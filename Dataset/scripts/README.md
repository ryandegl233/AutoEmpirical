# Dataset construction tools

These utilities collect source evidence, reconstruct candidates, repair stage
lineage and audit the normalized dataset. They were previously kept in `fixcode/`.
Run them from the repository root; use `--help` to inspect each CLI first.

```bash
python -m Dataset.scripts.repair_stage1_stage2_lineage --help
python -m Dataset.scripts.fetch_pytorch_stage1_candidates --help
python -m pytest tests/data -q
```

Collectors require network access and, for GitHub sources, the `gh` CLI and its
local authentication. Download caches belong in `Dataset/cache/` and stay local.
Repair commands may modify CSVs or audit files: inspect their arguments and use a
separate dataset copy when reconstructing. Historical audit scripts can require
the original Git revisions and source reports referenced in their source.

The existing `reports/*_reconstruction/` paths remain the provenance locations of
already published source artifacts. New reconstruction documentation belongs in
`Dataset/reconstruction/`. Model predictions and experiment summaries are not
dataset source evidence and must stay in ignored result/run directories.
