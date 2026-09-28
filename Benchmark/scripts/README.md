# Benchmark entry points

Run commands from the repository root. Use `--help` for supported arguments.

| Program | Purpose |
| --- | --- |
| `prepare_paper_benchmarks.py` | Prepare or verify seven-paper frozen inputs |
| `run_paper_llm_baseline.py` | SingleLLM native filtering/annotation |
| `run_paper_camel_mas_baseline.py` | CAMEL MAS native filtering/annotation |
| `run_adaptive_empirical_workflow.py` | Adaptive workflow, evidence and rule policies |
| `collect_external_evidence.py` | Bounded source collection and offline replay |
| `run_experiment_two.py` | Four-arm protocol; use `tools/run_experiment_two_sharded.py` for case isolation |
| `prepare_baseline_trust_root.py` | Prepare and explicitly register a local baseline |
| `evaluate_baseline_preservation_experiment.py`, `evaluate_targeted_sla_experiment.py` | Offline scoring of locally supplied runs |
| `seven_papers/` | Input verification and local result audits |
| `ase2022_dev48/` | Method-example preparation and A/B runs |

Original ASE and ISSTA preparation/baseline entry points remain available.
Frozen inputs belong in `Benchmark/inputs/`; new outputs belong in ignored
`Benchmark/runs/` or `Benchmark/results/`. See the [benchmark guide](../README.md).
