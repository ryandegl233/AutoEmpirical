# AutoEmpirical Dataset

This directory is the dataset entry point for AutoEmpirical. It contains the unified stage files and per-paper splits. Metadata and health reports remain in the top-level `metadata/` and `reports/` directories.

## Dataset Layout

| Location | Purpose |
| --- | --- |
| `stage1.csv` | Unified raw candidate records before human filtering |
| `stage2.csv` | Unified human-filtered bug-relevant records |
| `stage3.csv` | Unified final human-labeled records |
| `by_paper/<paper_id>/` | Per-paper stage splits for paper-level experiments |
| `../metadata/dataset_metadata.csv` | Paper-level metadata, counts, paths, and notes |
| `../metadata/data_dictionary.md` | Field definitions |
| `../reports/dataset_health_report.md` | Human-readable data quality report |
| `../reports/data_quality_metrics.csv` | Machine-readable quality metrics |
| `../reports/duplicate_key_rows.csv` | Duplicate `record_id` and `issue_url` evidence |
| `../reports/SHA256SUMS.txt` | File checksums for the frozen dataset package |

## Workflow Stages

| Stage | File | Meaning | Rows |
| --- | --- | --- | ---: |
| Stage 1 Raw | `stage1.csv` | Raw candidate records before human filtering | 33,822 |
| Stage 2 Filtered | `stage2.csv` | Human-filtered bug-relevant records | 4,199 |
| Stage 3 Annotated | `stage3.csv` | Final human-labeled records | 2,050 |

Use Stage 1 to Stage 2 for filtering or verification experiments. Use Stage 2 to Stage 3 for label prediction experiments.

## Included Studies

| Paper ID | Venue | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | ---: | ---: | ---: |
| `ase2022_towards_understanding_the_faults_of` | ASE | 3,859 | 684 | 682 |
| `icse2021_iot_bugs_and_development_challenges` | ICSE | 5,565 | 323 | 320 |
| `issta2024_bugs_in_pods_understanding_bugs` | ISSTA | 8,271 | 429 | 429 |
| `icse2023_an_empirical_study_on_bugs` | ICSE | 2,205 | 194 | 194 |
| `icse2024_understanding_transaction_bugs_in_database` | ICSE | 7,775 | 140 | 140 |
| `fse2021_an_exploratory_study_of_autopilot` | FSE | 569 | 168 | 142 |
| `icse2022_an_empirical_study_on_performance` | ICSME | 5,578 | 2,261 | 143 |

Two earlier candidate studies were excluded because their Stage 1 to Stage 2 filtering rate was 0%, which made them unsuitable for the current filtering-task formulation.

## Schema

All three stage files use the same 23-column schema:

```text
record_id, paper_id, source_project, issue_url, title, body, comments,
created_at, updated_at, state, symptom, root_cause, bug_type, component,
sub_component, trigger_condition, consequence, fix_type, severity_or_impact,
original_label_json, source_file, source_sheet, source_row_index
```

See `../metadata/data_dictionary.md` and `../metadata/stage1_label_dictionary.md` for field definitions.

## Data Health

The latest health check was run on 2026-06-16.

| Stage | Schema OK | Papers | `record_id` unique | Duplicate `record_id` rows | `issue_url` unique | Duplicate `issue_url` rows | Final-label coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 | yes | 7 | 33,799 / 33,822 | 23 | 33,749 / 33,822 | 73 | partial |
| Stage 2 | yes | 7 | 4,177 / 4,199 | 22 | 4,163 / 4,199 | 36 | partial |
| Stage 3 | yes | 7 | 2,042 / 2,050 | 8 | 2,032 / 2,050 | 18 | 100% |

Stage 3 is analysis-ready for label prediction experiments. Stage 1 and Stage 2 intentionally contain partial labels because they preserve earlier human workflow stages.

## Experimental Design Notes

- Use paper-level splits for cross-paper generalization.
- Use grouped `issue_url` splits when evaluating within a mixed-paper setting.
- Avoid row-level random splits because duplicate issue URLs can leak across train and test.
- Do not assume `record_id` or `issue_url` is a strict primary key.
- If a strict experiment ID is needed, derive a new row ID after deciding how duplicates should be handled.
