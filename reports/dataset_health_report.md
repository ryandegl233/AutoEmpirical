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

## Remaining Known Issues

- `record_id` is globally unique and same-paper `issue_url` duplicates have been removed.
- `issue_url` can still repeat across different papers because multiple studies may analyze the same issue or commit; use grouped-URL or paper-level evaluation splits.
- Some historical audit CSV files describe earlier repair passes and are retained for provenance even when current counts differ after cleanup.
