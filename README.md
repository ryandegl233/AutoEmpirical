# AutoEmpirical Dataset Handoff

This folder is a migration-ready handoff for a new, cleaner AutoEmpirical reproduction repository. It keeps the dataset and project context, but intentionally excludes the old forked data-processing code and the current MAS v2 design as implementation targets.

## Current Project Status

The project is currently at the dataset-freeze and baseline-search stage.

Completed:

- Built a three-stage empirical software-fault dataset from 7 retained papers.
- Excluded papers whose Stage 1 to Stage 2 filtering rate is 0%.
- Standardized Stage 1, Stage 2, and Stage 3 into one 23-column schema.
- Split the dataset by paper under `data/by_paper/`.
- Rechecked dataset health on 2026-06-16.
- Prepared a new baseline-first research plan.

Not completed:

- Do not implement or continue MAS v2 yet.
- Do not claim a new MAS contribution before running baseline experiments.
- Do not treat old `data/processed/quality_report.md` from the source repo as current health evidence.

## Folder Structure

```text
handoff_autoempirical_dataset_2026-06-16/
  README.md
  data/
    stage1.csv
    stage2.csv
    stage3.csv
    by_paper/
      <paper_id>/
        stage1.csv
        stage2.csv
        stage3.csv
  metadata/
    dataset_metadata.csv
    dataset_metadata.md
    paper_dataset_summary.csv
    paper_dataset_overview.md
    data_dictionary.md
    stage1_label_dictionary.md
    prompts.yaml
  reports/
    dataset_health_report.md
    data_quality_metrics.csv
    duplicate_key_rows.csv
    SHA256SUMS.txt
  research/
    baseline_research_plan.md
```

## Dataset Meaning

| Stage | File | Meaning | Rows |
| --- | --- | --- | ---: |
| Stage 1 Raw | `data/stage1.csv` | Raw candidate records before human filtering | 33,822 |
| Stage 2 Filtered | `data/stage2.csv` | Human-filtered bug-relevant records | 4,199 |
| Stage 3 Annotated | `data/stage3.csv` | Final human-labeled records | 2,050 |

Use Stage 1 to Stage 2 for filtering/verification experiments. Use Stage 2 to Stage 3 for label prediction experiments.

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

## Health Check Result

See `reports/dataset_health_report.md` for the full report.

Short version:

- All three stage files are readable CSVs.
- All three stage files share the expected 23-column schema.
- Stage 3 has full coverage for `symptom`, `root_cause`, `bug_type`, `component`, and `fix_type`.
- `record_id` and `issue_url` are not strict unique keys; duplicates are documented in `reports/duplicate_key_rows.csv`.
- Use paper-level or grouped-URL splits to avoid leakage.

## Research Direction

The next agent should not start by implementing a new MAS. The next agent should first run baselines:

1. Majority and heuristic baselines.
2. Classical text classifiers such as TF-IDF plus logistic regression or linear SVM.
3. Single-LLM zero-shot baseline.
4. Single-LLM few-shot baseline.
5. Self-consistency or majority-vote LLM baseline.
6. Retrieval-augmented single-LLM baseline.

Only after these baselines fail in specific, measured ways should a new MAS be designed. The intended argument is:

> Existing empirical-study baselines fail on specific cross-paper filtering and labeling cases in this dataset. The new MAS is designed to address those observed failures.

See `research/baseline_research_plan.md`.

## Recommended First Commands In A New Repo

```powershell
python -m pip install pandas scikit-learn pyyaml
python - <<'PY'
import pandas as pd
for stage in ["stage1", "stage2", "stage3"]:
    df = pd.read_csv(f"data/{stage}.csv")
    print(stage, df.shape, df["paper_id"].nunique())
PY
```

Then implement a small baseline runner that supports:

- `--task stage2_filter`
- `--task stage3_label`
- `--split leave_one_paper_out`
- `--split grouped_issue_url`

## Notes For Future Agents

- Yanjie wants the repository simplified. Do not migrate old forked data-processing code unless it is explicitly needed.
- Do not batch-delete files. If cleanup is needed, ask Yanjie to delete old files manually or delete one explicit file at a time only after approval.
- Treat this folder as the clean migration source.
- Treat the source repository as a working archive, not the target structure.
