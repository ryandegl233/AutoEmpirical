# AutoEmpirical Dataset

_Dataset documentation updated on 2026-08-03._

This directory is the canonical entry point for the unified three-stage dataset,
the seven paper-level splits, and reconstructed evidence sidecars. Metadata and
audits are stored in the repository-level `metadata/` and `reports/`
directories.

## Layout

| Location | Purpose |
| --- | --- |
| `stage1.csv` | Unified candidate records before paper-specific filtering |
| `stage2.csv` | Unified records retained as study-relevant bugs |
| `stage3.csv` | Unified records with final human taxonomy labels |
| `by_paper/<paper_id>/` | Per-paper Stage 1, Stage 2, and Stage 3 files |
| `evidence/*.jsonl` | Structured reconstructed evidence and retrieval status |
| `../metadata/dataset_metadata.csv` | Paper counts, paths, and study notes |
| `../metadata/data_dictionary.md` | Field definitions |
| `../reports/dataset_health_report.md` | Dataset-wide quality report |
| `../reports/SHA256SUMS.txt` | Checksums for tracked dataset and report artifacts |

## Workflow stages

| Stage | Rows | Intended task |
| --- | ---: | --- |
| Stage 1 | 35,391 | Candidate collection and Stage 2 filtering |
| Stage 2 | 4,197 | Study-relevant cohort and Stage 3 labeling input |
| Stage 3 | 2,041 | Gold taxonomy labels and final evaluation cohort |

Stage membership and labels are gold information. Experiment builders must
construct task-specific inputs that exclude the target membership or taxonomy
fields; loading a later-stage CSV directly into a model prompt can leak the
answer.

## Included studies and research objects

| Paper ID | Venue | Primary research object | Stage 1 / 2 / 3 |
| --- | --- | --- | ---: |
| `ase2022_towards_understanding_the_faults_of` | ASE 2022 | GitHub issues | 4,184 / 683 / 682 |
| `fse2021_an_exploratory_study_of_autopilot` | FSE 2021 | GitHub issues and pull requests | 567 / 168 / 142 |
| `icse2021_iot_bugs_and_development_challenges` | ICSE 2021 | GitHub issues and pull requests | 5,548 / 320 / 320 |
| `icse2022_an_empirical_study_on_performance` | ICSME 2022 | GitHub issues plus fixing commits | 6,835 / 2,265 / 136 |
| `icse2023_an_empirical_study_on_bugs` | ICSE 2023 | PyTorch GitHub issues | 2,207 / 194 / 194 |
| `icse2024_understanding_transaction_bugs_in_database` | ICSE 2024 | Bug reports, issues, and mailing-list threads | 7,775 / 140 / 140 |
| `issta2024_bugs_in_pods_understanding_bugs` | ISSTA 2024 | Commits | 8,275 / 427 / 427 |

ISSTA 2024 is commit-only: its core evidence is the commit URL and
`code_diff`; issue comments are not part of that paper's research-object
definition.

## Schema

The three unified stage files have the same 24 columns:

```text
record_id, paper_id, source_project, issue_url, title, body, comments,
created_at, updated_at, state, symptom, root_cause, bug_type, component,
sub_component, trigger_condition, consequence, fix_type, severity_or_impact,
original_label_json, source_file, source_sheet, source_row_index, code_diff
```

The six issue/report-oriented paper splits retain the 23-column common schema.
The ISSTA 2024 per-paper files have the same 24 columns as the unified files
because `code_diff` is part of that commit-only study's primary evidence.

See [the data dictionary](../metadata/data_dictionary.md) and
[Stage 1 label dictionary](../metadata/stage1_label_dictionary.md) for field
semantics.

## Evidence sidecars

| Sidecar | Coverage |
| --- | --- |
| `evidence/fse2021_autopilot_evidence.jsonl` | Current public issue/PR evidence for 567 Autopilot records |
| `evidence/icse2021_iot_discussion_evidence.jsonl` | Current discussion evidence for 5,548 IoT records |
| `evidence/icse2022_dl_performance_evidence.jsonl` | Issue, retained-comment, and fixing-commit evidence for 6,835 DL records |
| `evidence/icse2024_transaction_bugs_stage1_evidence.jsonl` | Multi-source discussion status for all 7,775 Transaction Bugs candidates |
| `evidence/icse2024_transaction_bugs_evidence.jsonl` | Multi-source discussion status for the final 140-record cohort |

`no_comments_in_source` means the source explicitly exposed zero discussion.
`comments_unavailable_in_source` means discussion could not be recovered and
must not be treated as a verified zero.

## Reconstruction status and limitations

- **Autopilot:** 567/567 records resolved, with one source-URL correction.
- **IoT:** 3,728 records have discussion, 1,110 have verified zero discussion,
  and 710 have unavailable sources.
- **DL performance:** all 6,835 cohort records have structured evidence; six
  current issue URLs return 404, five unique issues expose fewer comments than
  the author snapshot, and one Stage 3 row is absent from the author fixing
  tables. All 173 unique referenced fixing commits were recovered.
- **PyTorch:** Stage 1 integrates an exact-count 2,205 convergence
  reconstruction plus two Stage 2 lineage supplements. The exact-count cutoff
  conflicts with the paper's stated date, so the set is not represented as the
  original frozen author snapshot.
- **Transaction Bugs:** Stage 1 retrieval status is 6,825 with discussion, 948
  verified zero, and two unavailable; the final cohort is 127 with discussion,
  11 verified zero, and two unavailable.
- **ISSTA 2024:** all Stage 1/2/3 rows have non-empty commit diffs; comment
  sentinels are not missing-data indicators for this commit-only study.

Reconstructed web evidence is generally classified as `current_unversioned`.
Consult the corresponding `../reports/*_reconstruction/README.md` before
making claims about historical content.

## Data health

The latest dataset-wide health report was generated on 2026-06-25:

| Stage | Papers | Unique `record_id` | Duplicate `record_id` rows | Duplicate `issue_url` rows | Final-label coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 | 7 | 35,391 / 35,391 | 0 | 337 | Partial by design |
| Stage 2 | 7 | 4,197 / 4,197 | 0 | 14 | Partial by design |
| Stage 3 | 7 | 2,041 / 2,041 | 0 | 9 | 100% |

The later information reconstructions preserve row counts, record identities,
stage membership, and gold labels. Their paper-specific integration audits are
the authoritative checks for changed evidence fields.

## Experimental design

- Use paper-level splits for cross-paper generalization.
- Use grouped `paper_id` + `issue_url` splits within a paper or mixed corpus.
- Do not use unrestricted row-level random splits.
- Treat `record_id` as the global row key and `issue_url` as a grouping key.
- Select an explicit evidence mode for DL experiments: `issue_only`,
  `issue_discussion`, or `full_fix_evidence`.
- For ISSTA, use commit diffs rather than issue-discussion assumptions.
