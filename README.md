# AutoEmpirical: Benchmarking Automated Empirical Software Fault Analysis

_Repository status updated on 2026-09-28._

AutoEmpirical is a dataset-first benchmark for evaluating whether automated
methods can reproduce the collection, filtering, and taxonomy-labeling steps of
empirical software fault studies. The repository contains a repaired
three-stage dataset for seven papers, source-evidence sidecars, provenance
audits, shared single-LLM and CAMEL multi-agent baselines for all seven papers,
and an adaptive workflow for fault filtering and taxonomy annotation.

## Workflow

| Stage | Research operation | Unified file | Rows |
| --- | --- | --- | ---: |
| Stage 1 | Collect candidate research objects | `Dataset/stage1.csv` | 35,391 |
| Stage 2 | Retain study-relevant bugs | `Dataset/stage2.csv` | 4,197 |
| Stage 3 | Assign study taxonomies | `Dataset/stage3.csv` | 2,041 |

The research object is paper-specific. Six studies primarily analyze issues,
pull requests, or other bug-report records; ISSTA 2024 Bugs in Pods analyzes
commits. See the [dataset guide](./Dataset/README.md) for the exact object and
evidence boundary of every paper.

## Current coverage

| Area | Status |
| --- | --- |
| Unified and per-paper datasets | Available for all seven retained studies |
| Information reconstruction | Integrated for Autopilot, IoT, DL performance, PyTorch Stage 1, and Transaction Bugs |
| Commit diffs | Integrated for all ISSTA 2024 stages |
| Single-LLM and CAMEL MAS | Shared entry points implemented for all seven papers |
| Adaptive workflow | Independent analysis, evidence verification, arbitration and execution audits |
| Experiment inputs | Frozen seven-paper cohorts with paper-specific codebooks |
| Evidence tools | External source capture, supplemental evidence and source-graph experiments implemented |

The unified dataset is unchanged. Recent work adds benchmark implementations,
frozen experiment inputs, evidence tools and reusable dataset construction scripts. See the
[benchmark guide](./Benchmark/README.md) for experiment commands and protocols.

## Repository structure

```text
AutoEmpirical/
  Dataset/
    stage1.csv
    stage2.csv
    stage3.csv
    by_paper/
    evidence/
    scripts/
    reconstruction/
  Benchmark/
    configs/
    inputs/
    scripts/
    src/
    results/                    # local outputs
    runs/                       # local outputs
  tests/
  metadata/
    dataset_metadata.csv
    data_dictionary.md
    prompts.yaml
  reports/
    dataset_health_report.md
    *_reconstruction/
    SHA256SUMS.txt
  research/
    baseline_research_plan.md
```

## Quick start

Fetch the large source files and install the analysis dependencies:

```powershell
git lfs pull
python -m pip install -r requirements.txt
```

Verify the unified dataset:

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
stage1 (35391, 24) 7
stage2 (4197, 24) 7
stage3 (2041, 24) 7
```

Run the repository tests with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

For the implemented experiment commands and provider settings, see the
[benchmark guide](./Benchmark/README.md). CAMEL-based runs additionally require:

```powershell
python -m pip install -r Benchmark/requirements-mas.txt
```

## Documentation

- [Dataset layout, object types, and provenance](./Dataset/README.md)
- [Benchmark implementations and protocols](./Benchmark/README.md)
- [Field definitions](./metadata/data_dictionary.md)
- [Paper-level metadata](./metadata/dataset_metadata.md)
- [Dataset health report](./reports/dataset_health_report.md)
- [Baseline research plan](./research/baseline_research_plan.md)

## Citation

If you use this repository, please cite the related AutoEmpirical paper when
the final citation is available.

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

Email: Ryandegl@outlook.com
