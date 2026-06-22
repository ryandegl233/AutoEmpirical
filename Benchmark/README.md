# AutoEmpirical Benchmark

This directory is the benchmark entry point for AutoEmpirical. It is intended to contain baseline implementations, experiment configurations, reusable benchmark code, and generated result tables.

The current repository state is baseline-planning-ready. Benchmark implementation has not started yet.

## Benchmark Tasks

| Task | Input | Gold target | Main metrics |
| --- | --- | --- | --- |
| Stage 2 filtering | `../Dataset/stage1.csv` | Membership in `../Dataset/stage2.csv` | Precision, recall, F1, false positive categories, false negative categories |
| Stage 3 labeling | `../Dataset/stage2.csv` | Labels in `../Dataset/stage3.csv` | Exact match, macro-F1, micro-F1, per-label F1 |

## Planned Baselines

Run baselines before designing a new MAS:

1. Majority and heuristic baselines.
2. Classical text classifiers such as TF-IDF plus logistic regression or linear SVM.
3. Single-LLM zero-shot baseline.
4. Single-LLM few-shot baseline.
5. Self-consistency or majority-vote LLM baseline.
6. Retrieval-augmented single-LLM baseline.

The intended research logic is baseline first, failure analysis second, MAS design third.

## Directory Layout

```text
Benchmark/
  README.md
  configs/
    README.md
  scripts/
    README.md
  src/
    README.md
  results/
    README.md
```

## Expected Implementation Interface

The first implementation target should be a deterministic baseline runner with a stable command-line interface:

```powershell
python Benchmark/scripts/run_baseline.py --task stage2_filter --split leave_one_paper_out
python Benchmark/scripts/run_baseline.py --task stage3_label --split leave_one_paper_out
python Benchmark/scripts/run_baseline.py --task stage2_filter --split grouped_issue_url
python Benchmark/scripts/run_baseline.py --task stage3_label --split grouped_issue_url
```

Suggested options:

| Option | Values | Meaning |
| --- | --- | --- |
| `--task` | `stage2_filter`, `stage3_label` | Select the benchmark task |
| `--split` | `leave_one_paper_out`, `grouped_issue_url` | Select the evaluation split |
| `--baseline` | `majority`, `heuristic`, `tfidf_logreg`, `tfidf_svm`, `llm_zero_shot`, `llm_few_shot`, `llm_self_consistency`, `llm_rag` | Select the baseline |
| `--label-field` | `symptom`, `root_cause`, `bug_type`, `component`, `fix_type` | Select the Stage 3 label field |
| `--output-dir` | path | Store predictions, metrics, and logs |

## Reporting Requirements

Each benchmark run should report:

- macro-F1 and micro-F1;
- per-paper performance;
- invalid-output rate for LLM baselines;
- abstention or uncertain rate;
- cost per 100 records for LLM baselines;
- latency per 100 records;
- sampled failure-mode categories.

## Related Plan

See `../research/baseline_research_plan.md` for the full baseline-first roadmap and the motivation for delaying MAS design until baseline failure modes are measured.
