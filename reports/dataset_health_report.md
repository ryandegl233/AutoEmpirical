# AutoEmpirical Dataset Health Report

Generated on 2026-06-25 after Stage 1 to Stage 2 lineage repairs, Stage 2 to
Stage 3 lineage repairs, and timestamp field cleanup.

## Dataset And Grain Summary

| Stage | Rows | Columns | Unique `record_id` | Duplicate `record_id` excess | Unique `issue_url` | Duplicate `issue_url` excess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage1 | 35,391 | 23 | 35,391 | 0 | 35,054 | 337 |
| stage2 | 4,197 | 23 | 4,197 | 0 | 4,183 | 14 |
| stage3 | 2,041 | 23 | 2,041 | 0 | 2,032 | 9 |

## Stage 1 To Stage 2 Lineage

- Every Stage 2 row now has a matching Stage 1 row with the same `paper_id` and `issue_url`.
- 1,597 previously missing Stage 1 rows were restored from Stage 2 source fields.
- Downstream label fields were cleared from the 1,597 inserted Stage 1 rows.
- Repair details are recorded in `reports/stage1_stage2_lineage_repairs.csv`.

## Stage 2 To Stage 3 Lineage

- Every Stage 3 row now has a matching Stage 2 row with the same `paper_id`, `issue_url`, and `record_id`.
- 21 previously missing Stage 2 rows were restored from Stage 3 source fields.
- Final annotation fields were cleared from the 21 inserted Stage 2 rows.
- Existing Stage 3 labels were preserved except for duplicate-key cleanup, yielding the current 2,041-row annotated set.
- Repair details are recorded in `reports/stage2_stage3_lineage_repairs.csv`.

## Timestamp Cleanup

- Numeric `updated_at` values that represented TXBug confirmed-duration fields were removed from the three unified stage files and the ICSE 2024 per-paper splits.
- 53 affected ICSE 2024 records were backfilled from local TXBug reconstruction artifacts by matching `issue_url`.
- 4 affected records had no authoritative local timestamp and now use `not_available_in_source`.
- Repair details are recorded in `reports/timestamp_repairs.csv`.

## ICSME 2022 DL-Performance Evidence Reconstruction

- The existing 6,835 / 2,265 / 136 cohorts, `record_id` values, stage lineage,
  and gold labels were preserved.
- Issue metadata was recovered for 6,829 of 6,835 Stage 1 records (5,575
  PyTorch and 1,254 TensorFlow records); all 6,829 have titles and 6,793 have
  non-empty bodies after whitespace normalization. Six PyTorch issues returned
  404 through both GraphQL and REST and remain explicitly marked inaccessible.
- The author artifact expected 34,766 comments. The benchmark retains 34,739
  currently visible comments after enforcing the author snapshot count; five
  unique issues have fewer currently visible comments than the historical
  count and are listed in the mismatch audit.
- All 173 unique fixing commits referenced by the author fixing tables were
  recovered. They contain 710 changed-file entries: 709 expose patches and one
  does not. One Stage 3 row (PyTorch issue 17502) is absent from the author
  fixing-commit tables and is not supplemented by inference.
- Cohort membership and comment counts are anchored to author artifact commit
  `24a19cd03a607a5051686ea2e275ac9dad511626`. Reconstructed GitHub text is
  `current_unversioned`, not an exact February 2021 snapshot.
- Full provenance and audits are under
  `reports/dl_performance_reconstruction/`; structured evidence is stored in
  `Dataset/evidence/icse2022_dl_performance_evidence.jsonl`.

## Remaining Known Issues

- `record_id` is globally unique and same-paper `issue_url` duplicates have been removed.
- `issue_url` can still repeat across different papers because multiple studies may analyze the same issue or commit; use grouped-URL or paper-level evaluation splits.
- Some historical audit CSV files describe earlier repair passes and are retained for provenance even when current counts differ after cleanup.
