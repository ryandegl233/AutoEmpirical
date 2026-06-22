# AutoEmpirical Dataset Health Report

Generated on 2026-06-16.

## Dataset And Grain Summary

This handoff package contains the current three-stage AutoEmpirical reproduction dataset. The intended grain is one source issue, commit, or bug record per row, scoped by `paper_id` and `source_project`.

| Stage | File | Rows | Columns | Intended Use |
| --- | --- | ---: | ---: | --- |
| Stage 1 Raw | `Dataset/stage1.csv` | 33,822 | 23 | Candidate records before human filtering |
| Stage 2 Filtered | `Dataset/stage2.csv` | 4,199 | 23 | Human-filtered bug-relevant records |
| Stage 3 Annotated | `Dataset/stage3.csv` | 2,050 | 23 | Human-annotated records with final labels |

All three stage files use the same 23-column schema:

`record_id`, `paper_id`, `source_project`, `issue_url`, `title`, `body`, `comments`, `created_at`, `updated_at`, `state`, `symptom`, `root_cause`, `bug_type`, `component`, `sub_component`, `trigger_condition`, `consequence`, `fix_type`, `severity_or_impact`, `original_label_json`, `source_file`, `source_sheet`, `source_row_index`.

Machine-readable check outputs:

- `reports/data_quality_metrics.csv`
- `reports/duplicate_key_rows.csv`
- `reports/SHA256SUMS.txt`

## Checks Performed

- CSV readability for all stage files.
- Schema equality across Stage 1, Stage 2, and Stage 3.
- Row count and paper count checks.
- Non-empty coverage for core fields.
- Candidate key uniqueness checks for `record_id` and `issue_url`.
- Stage 3 label coverage checks for `symptom`, `root_cause`, `bug_type`, `component`, and `fix_type`.
- Metadata consistency spot check against `metadata/dataset_metadata.csv` and `metadata/paper_dataset_summary.csv`.

## Health Summary

| Stage | Schema OK | Papers | `record_id` Unique | Duplicate `record_id` Rows | `issue_url` Unique | Duplicate `issue_url` Rows | `symptom` Coverage | `root_cause` Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 | yes | 7 | 33,799 / 33,822 | 23 | 33,749 / 33,822 | 73 | 1,270 / 33,822 | 1,272 / 33,822 |
| Stage 2 | yes | 7 | 4,177 / 4,199 | 22 | 4,163 / 4,199 | 36 | 2,067 / 4,199 | 2,069 / 4,199 |
| Stage 3 | yes | 7 | 2,042 / 2,050 | 8 | 2,032 / 2,050 | 18 | 2,050 / 2,050 | 2,050 / 2,050 |

Stage 3 is analysis-ready for label prediction experiments because the final target labels are fully populated. Stage 1 and Stage 2 intentionally contain partial labels because they preserve earlier human workflow stages.

## Included Papers

| Paper ID | Stage 1 | Stage 2 | Stage 3 |
| --- | ---: | ---: | ---: |
| `ase2022_towards_understanding_the_faults_of` | 3,859 | 684 | 682 |
| `icse2021_iot_bugs_and_development_challenges` | 5,565 | 323 | 320 |
| `issta2024_bugs_in_pods_understanding_bugs` | 8,271 | 429 | 429 |
| `icse2023_an_empirical_study_on_bugs` | 2,205 | 194 | 194 |
| `icse2024_understanding_transaction_bugs_in_database` | 7,775 | 140 | 140 |
| `fse2021_an_exploratory_study_of_autopilot` | 569 | 168 | 142 |
| `icse2022_an_empirical_study_on_performance` | 5,578 | 2,261 | 143 |

## Findings

### Medium: `record_id` Is Not A Strict Primary Key

Evidence: Stage 1 has 23 duplicate `record_id` rows, Stage 2 has 22, and Stage 3 has 8. The duplicate rows are saved in `reports/duplicate_key_rows.csv`.

Why it matters: evaluation code should not assume `record_id` alone is a strict primary key. Using it as the only merge key can duplicate predictions or inflate metrics.

Recommended remediation: use a composite key of `paper_id`, `source_project`, `issue_url`, and row position when joining predictions back to source data. If a new repo needs a strict ID, create a derived `handoff_row_id` after deciding whether duplicate rows are true duplicates or intentionally repeated records from source artifacts.

### Medium: `issue_url` Duplicates Exist Within And Across Papers

Evidence: Stage 1 has 73 duplicate `issue_url` rows, Stage 2 has 36, and Stage 3 has 18. Some duplicates are exact repeated records; others reflect the same GitHub issue appearing in more than one empirical study, especially PyTorch issues shared by PyTorch-bug and performance-bug papers.

Why it matters: for cross-paper evaluation, the same URL can appear in train and test if splits are made only by row. This creates leakage.

Recommended remediation: use paper-level splits for cross-paper generalization, or group by normalized `issue_url` before splitting.

### Low: Older Phase-1 Quality Report Is Stale

Evidence: the source repository's `data/processed/quality_report.md` describes a 6,262-row tfjs-only Phase-1 dataset. This does not match the current three-stage dataset in this handoff package.

Why it matters: future agents should not use that old report as the current dataset health source.

Recommended remediation: treat this file, `reports/dataset_health_report.md`, as the current health summary for migration.

## Assumptions And Open Questions

- Stage 3 is the current gold-label file for classification experiments.
- Stage 2 is the positive set for filtering/verification experiments.
- Stage 1 is the candidate pool for filtering/verification experiments.
- The duplicate rows were not removed in this package because the user asked for a migration-ready handoff, not destructive cleaning.
- Before model training, decide whether duplicates should be preserved as source fidelity or consolidated into unique bug records.
