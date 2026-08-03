# AutoEmpirical: Benchmarking Automated Empirical Software Fault Analysis

_Repository status updated on 2026-08-03._

AutoEmpirical is a dataset-first benchmark for evaluating whether automated
methods can reproduce the collection, filtering, and taxonomy-labeling steps of
empirical software fault studies. The repository contains a repaired
three-stage dataset for seven papers, source-evidence sidecars, provenance
audits, and executable single-LLM and multi-agent baselines for the ASE 2022 and
ISSTA 2024 cohorts.

## 📋 Workflow

| Stage | Research operation | Unified file | Rows |
| --- | --- | --- | ---: |
| Stage 1 | Collect candidate research objects | `Dataset/stage1.csv` | 35,391 |
| Stage 2 | Retain study-relevant bugs | `Dataset/stage2.csv` | 4,197 |
| Stage 3 | Assign study taxonomies | `Dataset/stage3.csv` | 2,041 |

The research object is paper-specific. Six studies primarily analyze issues,
pull requests, or other bug-report records; ISSTA 2024 BugsInPy analyzes
commits. See the [dataset guide](./Dataset/README.md) for the exact object and
evidence boundary of every paper.

## 📊 Current coverage

| Area | Status |
| --- | --- |
| Unified and per-paper datasets | Available for all seven retained studies |
| Information reconstruction | Integrated for Autopilot, IoT, DL performance, PyTorch Stage 1, and Transaction Bugs |
| Commit diffs | Integrated for all ISSTA 2024 stages |
| Single-LLM experiments | Implemented for ASE 2022 and ISSTA 2024 |
| Multi-agent experiments | Implemented for ASE 2022 and ISSTA 2024 |
| Other paper-specific baselines | Not yet implemented |

The repository records retrieval limitations explicitly. Reconstructed public
discussion is generally `current_unversioned`, not a claim that GitHub or other
sources still expose the exact historical text seen by the original authors.

## 📁 Repository structure

```text
AutoEmpirical/
  Dataset/
    stage1.csv
    stage2.csv
    stage3.csv
    by_paper/
    evidence/
  Benchmark/
    scripts/
    src/
    results/
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

## ⚡ Quick start

Install the minimal analysis dependencies:

```powershell
python -m pip install pandas scikit-learn pyyaml
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
python -m pytest -q
```

For the implemented experiment commands and provider settings, see the
[benchmark guide](./Benchmark/README.md). CAMEL-based runs additionally require:

```powershell
python -m pip install -r Benchmark/requirements-mas.txt
```

## ⚠️ Experimental safeguards

- Build Stage 2 targets from membership in the gold Stage 2 cohort, but do not
  expose Stage 2 or Stage 3 labels in model inputs.
- Build Stage 3 inputs only from the information fields permitted by the chosen
  evidence mode; keep gold taxonomy fields evaluation-only.
- Use paper-level splits for cross-paper generalization and grouped
  `paper_id` + `issue_url` splits for within-paper evaluation.
- Treat `record_id` as the global row key. An `issue_url` may repeat across
  papers and must not cross train/test boundaries.
- Keep sentinel values such as `no_comments_in_source` distinct from
  `comments_unavailable_in_source`.
- Regenerate `reports/SHA256SUMS.txt` whenever a tracked dataset or report file
  changes.

## 📚 Documentation

- [Dataset layout, object types, and provenance](./Dataset/README.md)
- [Benchmark implementations and results](./Benchmark/README.md)
- [Field definitions](./metadata/data_dictionary.md)
- [Paper-level metadata](./metadata/dataset_metadata.md)
- [Dataset health report](./reports/dataset_health_report.md)
- [Baseline research plan](./research/baseline_research_plan.md)

## 📝 Citation

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

## 👤 Contact

Maintainer: Yanjie Yu

Email: Ryandegl@outlook.com
