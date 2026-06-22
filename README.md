# AutoEmpirical: Benchmarking Automated Empirical Software Fault Analysis

AutoEmpirical is a dataset-first benchmark project for studying whether LLMs and later multi-agent systems can reproduce key steps in empirical software fault studies. The current repository contains a frozen cross-paper dataset, metadata, health reports, and a baseline-first experimental roadmap.

This repository is organized into two main parts:

1. [The AutoEmpirical Dataset](./Dataset/)
2. [The AutoEmpirical Benchmark](./Benchmark/)

The detailed data files are kept in `Dataset/`. Metadata tables, health reports, and the research plan are kept in `metadata/`, `reports/`, and `research/`.

This repository is still under active development. The current project state is dataset-ready and benchmark-planning-ready; baseline implementation has not started yet.

## Overview

Empirical software fault studies usually involve three labor-intensive steps:

1. Collecting candidate software fault records.
2. Filtering candidates into a study-specific valid bug set.
3. Assigning human taxonomy labels such as symptom, root cause, bug type, component, and fix type.

AutoEmpirical turns these steps into a unified three-stage dataset so automated methods can be evaluated against human empirical-study workflows.

## Content

### The AutoEmpirical Dataset

The dataset contains seven retained empirical software fault studies and three workflow stages:

| Stage | File | Meaning | Rows |
| --- | --- | --- | ---: |
| Stage 1 Raw | `Dataset/stage1.csv` | Raw candidate records before human filtering | 33,822 |
| Stage 2 Filtered | `Dataset/stage2.csv` | Human-filtered bug-relevant records | 4,199 |
| Stage 3 Annotated | `Dataset/stage3.csv` | Final human-labeled records | 2,050 |

See [Dataset/README.md](./Dataset/README.md) for the dataset layout, included studies, schema notes, and data health summary.

### The AutoEmpirical Benchmark

The benchmark is intended to evaluate two tasks:

| Task | Input | Gold target | Purpose |
| --- | --- | --- | --- |
| Stage 2 filtering | `Dataset/stage1.csv` | Membership in `Dataset/stage2.csv` | Predict whether a candidate record should be accepted as bug-relevant |
| Stage 3 labeling | `Dataset/stage2.csv` | Labels in `Dataset/stage3.csv` | Predict empirical-study labels such as `symptom`, `root_cause`, `bug_type`, `component`, and `fix_type` |

Before designing a new MAS, the project should run simpler baselines first: majority and heuristic baselines, TF-IDF classifiers, single-LLM zero-shot and few-shot baselines, self-consistency, and retrieval-augmented single-LLM baselines.

See [Benchmark/README.md](./Benchmark/README.md) and [research/baseline_research_plan.md](./research/baseline_research_plan.md) for the benchmark plan.

## Repository Structure

```text
AutoEmpirical/
  README.md
  Dataset/
    README.md
    stage1.csv
    stage2.csv
    stage3.csv
    by_paper/
      <paper_id>/
        stage1.csv
        stage2.csv
        stage3.csv
  Benchmark/
    README.md
    configs/
    scripts/
    src/
    results/
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

## Quick Start

Install minimal analysis dependencies:

```powershell
python -m pip install pandas scikit-learn pyyaml
```

Verify that the dataset loads:

```powershell
@'
import pandas as pd

for stage in ["stage1", "stage2", "stage3"]:
    df = pd.read_csv(f"Dataset/{stage}.csv", low_memory=False)
    print(stage, df.shape, df["paper_id"].nunique())
'@ | python -
```

Expected output:

```text
stage1 (33822, 23) 7
stage2 (4199, 23) 7
stage3 (2050, 23) 7
```

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Dataset consolidation | Complete | Three-stage dataset and per-paper splits are present |
| Metadata | Complete | Paper-level metadata and data dictionary are present |
| Health check | Complete | Latest report is under `reports/` |
| Baseline plan | Ready | Baseline sequence is documented under `research/` |
| Baseline implementation | Not started | Next engineering step |
| MAS redesign | Blocked by baselines | Should wait for measured baseline failures |

## Important Notes

- Keep this repository dataset-first and baseline-first.
- Do not continue the previous MAS v2 design as the default next step.
- Do not claim a new MAS contribution before baseline experiments have been run.
- Use paper-level splits or grouped `issue_url` splits to avoid issue-level leakage.
- Treat `record_id` and `issue_url` as non-strict keys; duplicate rows are documented in `reports/duplicate_key_rows.csv`.
- Preserve `reports/SHA256SUMS.txt` or regenerate it whenever dataset files change.

## Citation

If you use this repository, please cite the related AutoEmpirical paper when the final citation is available.

```bibtex
@article{yu2025autoempirical,
  title  = {AutoEmpirical: LLM-based Automated Research for Empirical Software Fault Analysis},
  author = {Yu, Yanjie and others},
  year   = {2025},
  note   = {Citation details to be updated}
}
```

## Contact

Maintainer: Yanjie Yu
