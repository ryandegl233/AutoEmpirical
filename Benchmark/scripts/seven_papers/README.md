# Seven-paper utilities

`verify_prepared_cohorts.py` audits released input identity, prompt projections,
labels and shared source URLs. Its default report is local under `Benchmark/cache/`.

The remaining tools consume local runs; inspect `--help` for arguments:

- `summarize_runs.py`: summarize run artifacts.
- `verify_full_evidence.py`: audit prediction/evidence bindings.
- `build_prediction_audit_index.py`: build an index for local review.
- `render_full_results.py`: render the fixed five-domain, two-engine study
  (100 filtering and 50 annotation records per domain/engine).

All produced predictions, result tables, metrics and audit indexes stay under
ignored `Benchmark/results/` or `Benchmark/runs/`; they are not release artifacts.
